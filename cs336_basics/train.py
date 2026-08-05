import torch
import numpy as np
from numpy.typing import NDArray
import torch.nn as nn
from math import sqrt,cos,pi,log
from einops import einsum
from einops import rearrange
from jaxtyping import Bool, Float, Int
from torch import Tensor
from collections.abc import Callable, Iterable
from typing import Optional
import os
from typing import BinaryIO,IO
import argparse, sys, json, time

from tool import data_loader, save_checkpoint, load_checkpoint
from transformer import transformer_lm
from optimizer import AdamW, cross_entropy, cosine_learning_rate, grad_global_norm,gradient_clipping


# ============ 第零层：辅助函数 ============
# 主循环里只留「每一步都要做的事」，其余都收进这里，否则 main() 会长到看不清骨架。

def resolve_device(name: str) -> torch.device:
    """把命令行中的设备名称转换成 torch.device。
    当输入 auto 时，按 CUDA、MPS、CPU 的优先级选择设备。"""
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda 但这台机器没有可用的 CUDA")
    return torch.device(name)


def load_tokens(
        path: str, #token文件路径
        dtype: str, #如unit16
        context_length: int,  #每条训练样本所含token数量
) -> NDArray:
    """以内存映射方式打开.bin token 文件，并进行长度检查。
    """
    data = np.memmap(path, dtype=np.dtype(dtype), mode="r")

    if len(data) < context_length + 1:
        raise ValueError(f" 数据只有 {len(data)} 个 token，装不下 context_length={context_length}")

    return data


def make_fixed_batches(
        data: NDArray,
        batch_size: int,
        context_length: int,
        device: torch.device,
        num_batches: int,
        seed: int,
) -> list[tuple[Tensor, Tensor]]:
    """提前从验证集随机采样固定的一组 batch。
    之后每次验证都使用同样的数据，避免验证曲线因为每次随机样本不同而抖动。
    """
    state = np.random.get_state()
    np.random.seed(seed)
    batches = [data_loader(data, batch_size, context_length, device) for _ in range(num_batches)]
    np.random.set_state(state)
    return batches


@torch.no_grad()
def evaluate(model: nn.Module, batches: list[tuple[Tensor, Tensor]]) -> float:
    """在固定验证 batch 上计算平均交叉熵，并禁止构建反向传播计算图"""
    model.eval()
    total = 0.0
    for inputs, targets in batches:
        total += cross_entropy(model(inputs), targets).item()
    model.train()
    return total / len(batches)




def log_jsonl(path: str, record: dict) -> None:
    """每条日志追加一行 JSON。Section 6/7 画曲线时直接读这个文件，不必去 grep stdout。"""
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


# ============ 第一层:参数解析函数 ============
# 只负责定义参数、返回配置对象,不碰任何训练逻辑。
# 单独拆出来的好处：可以脱离命令行，直接用 parse_args(["--lr", "1e-3", ...]) 测试或做 sweep。

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a transformer LM")

    # --- 数据 ---
    parser.add_argument("--train_data", type=str, required=True, help="训练集 token id 数组(.npy 或裸 .bin)")
    parser.add_argument("--val_data", type=str, required=True, help="验证集 token id 数组")
    parser.add_argument("--out_dir", type=str, required=True, help="checkpoint / 日志 / 配置的输出目录")
    parser.add_argument("--data_dtype", type=str, default="uint16",
                        help="裸二进制文件的 dtype；.npy 会自带 dtype，此项被忽略")

    # --- 模型（对照 transformer_lm.__init__） ---
    parser.add_argument("--vocab_size", type=int, default=10000)
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--d_ff", type=int, default=1344)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=16)
    parser.add_argument("--rope_theta", type=float, default=10000.0)

    # --- 优化器与调度 ---
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="（保留但不生效：每一步的 lr 都由 cosine 调度写进 param_groups）")
    parser.add_argument("--betas", type=float, nargs=2, default=(0.9, 0.95), metavar=("BETA1", "BETA2"))
    parser.add_argument("--eps",type = float, default = 1e-8)
    parser.add_argument("--alpha_max", type=float, default=3e-4)
    parser.add_argument("--alpha_min", type=float, default=3e-5)
    parser.add_argument("--T_w", type=int, default=0, help="线性 warmup 步数 T_w")
    parser.add_argument("--T_c", type=int, default=None,
                        help="余弦周期结束步数 T_c，默认等于 --total_steps")
    parser.add_argument("--total_steps", type=int, default=5000)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    # --- 训练过程 ---
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="cpu")
    parser.add_argument("--seed", type=int, default=0, help="同时设 torch 和 numpy，缺一不可复现")
    parser.add_argument("--log_interval", type=int, default=20)
    parser.add_argument("--eval_interval", type=int, default=200)
    parser.add_argument("--eval_batches", type=int, default=10, help="每次验证平均多少个固定 batch")
    parser.add_argument("--checkpoint_interval", type=int, default=500, help="每隔多少步覆盖写 ckpt_last.pt")
    parser.add_argument("--milestone_interval", type=int, default=0,
                        help=">0 时每隔这么多步额外另存一个 ckpt_{step}.pt（0 表示只保留最新）")
    parser.add_argument("--resume_from", type=str, default=None)
    parser.add_argument("--overfit", action="store_true",
                        help="只用一个固定 batch 反复训练：验收检查②，loss 必须能压到接近 0")

    return parser.parse_args(argv)


# ============ 第二层:main(args) —— 完整训练流程 ============
# 接受第一层返回的配置对象，本身不解析命令行 —— 这样才能被别的脚本 import 后直接调用。

def main(args: argparse.Namespace) -> int:
    # ---------- 区块 1：配置校验 + 落盘 ----------
    # 早失败比晚失败便宜：下面几条如果错了，报错会发生在很深的地方，信息完全没法读。
    if args.d_model % args.num_heads != 0:
        raise ValueError(f"d_model={args.d_model} 不能被 num_heads={args.num_heads} 整除")
    if args.T_c is None:
        args.T_c = args.total_steps
    if args.T_w >= args.T_c:
        raise ValueError(f"T_w={args.T_w} 必须小于 T_c={args.T_c}")
    if args.batch_size <= 0 or args.total_steps <= 0 or args.eval_batches <= 0:
        raise ValueError("batch_size、total_steps、eval_batches 都必须为正")

    os.makedirs(args.out_dir, exist_ok=True)
    config = vars(args)
    with open(os.path.join(args.out_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2, ensure_ascii=False, default=str)
    metrics_path = os.path.join(args.out_dir, "metrics.jsonl")
    print("[config] " + json.dumps(config, ensure_ascii=False, default=str))

    # ---------- 区块 2：准备阶段 ----------
    # ① 种子：data_loader 走 numpy 全局随机数，模型初始化走 torch，两个都要设
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ② 设备
    device = resolve_device(args.device)
    print(f"[device] {device}")

    # ③ 数据（memmap + 值域自检）
    train_data = load_tokens(args.train_data, args.data_dtype, args.vocab_size, args.context_length)
    val_data = load_tokens(args.val_data, args.data_dtype, args.vocab_size, args.context_length)

    # ④ 模型
    body = transformer_lm(args.vocab_size,
                          args.context_length,
                          args.num_layers,
                          args.d_model,
                          args.num_heads,
                          args.d_ff,
                          args.rope_theta,
                          device)
    body.train()  #进入train模式
    n_params = sum(p.numel() for p in body.parameters())
    print(f"[model] {n_params:,} parameters")

    # ⑤ 优化器
    optimizer = AdamW(body.parameters(),
                      args.alpha_max,
                      tuple(args.betas),
                      args.eps,
                      args.weight_decay)

    # ⑥ 可选：恢复 checkpoint
    # 存的是「下一步该跑第几步」，所以直接拿它当循环起点。
    # 注意：恢复时必须用同一套模型配置——RoPE 的 cos/sin 表是 persistent=False 的 buffer，
    # 不在 checkpoint 里，配置不一致时它会被静默地重新算错，load_state_dict 察觉不到。
    start_step = 0
    if args.resume_from is not None:
        start_step = load_checkpoint(args.resume_from, body, optimizer)
        print(f"[resume] 从 {args.resume_from} 恢复，从第 {start_step} 步继续")

    # ⑦ 固定验证 batch（以及 --overfit 时固定的那一个训练 batch）
    val_batches = make_fixed_batches(val_data, args.batch_size, args.context_length,
                                     device, args.eval_batches, args.seed + 1)
    fixed_train_batch = None
    if args.overfit:
        fixed_train_batch = data_loader(train_data, args.batch_size, args.context_length, device)
        print("[overfit] 只用一个固定 batch 训练，loss 应该一路降到接近 0")

    # ---------- 区块 3 + 4：主循环 + 周期性任务 ----------
    t0 = time.perf_counter()
    tokens_per_step = args.batch_size * args.context_length
    loss_value = float("nan")

    for step in range(start_step, args.total_steps):
        # 1) 算学习率，写进 param_groups
        #    必须写进 group['lr']——AdamW.step 读的就是它。写成 self.lr 之类的实例属性会静默失效。
        lr = cosine_learning_rate(step, args.alpha_max, args.alpha_min, args.T_w, args.T_c)
        for group in optimizer.param_groups:
            group["lr"] = lr

        # 2) 采一个 batch
        if fixed_train_batch is not None:
            inputs, targets = fixed_train_batch
        else:
            inputs, targets = data_loader(train_data, args.batch_size, args.context_length, device)

        # 3) 前向 + loss：cross_entropy 支持 (..., vocab) 对 (...)，不用手动 flatten
        logits = body(inputs)
        loss = cross_entropy(logits, targets)

        # 4) 清梯度 + 反向
        #    PyTorch 的梯度是累加的，不清零等于第 t 步用了前 t 步梯度之和。清零必须在 backward 之前。
        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        # 5) 梯度裁剪：夹在 backward 和 step 之间。放前面没梯度可裁，放后面参数已经被推飞了。
        need_log = (step % args.log_interval == 0) or (step == args.total_steps - 1)
        pre_clip_norm = grad_global_norm(body.parameters()) if need_log else None
        gradient_clipping(body.parameters(), args.grad_clip)

        # 6) 更新
        optimizer.step()

        loss_value = loss.item()
        if step == start_step and args.resume_from is None:
            # 验收检查①：随机初始化的模型对下一个 token 近似均匀预测，交叉熵应 ≈ ln(V)
            print(f"[sanity] 初始 loss = {loss_value:.4f}，期望 ≈ ln(vocab_size) = {log(args.vocab_size):.4f}")

        # 7) 周期性任务
        if need_log:
            elapsed = time.time() - t0
            done = step - start_step + 1
            print(f"step {step:6d} | loss {loss_value:8.4f} | lr {lr:.3e} | gnorm {pre_clip_norm:8.3f}"
                  f" | {elapsed:7.1f}s | {done * tokens_per_step / max(elapsed, 1e-9):9,.0f} tok/s")
            log_jsonl(metrics_path, {"step": step, "train_loss": loss_value, "lr": lr,
                                     "grad_norm": pre_clip_norm, "elapsed": elapsed})

        if args.eval_interval > 0 and ((step + 1) % args.eval_interval == 0 or step + 1 == args.total_steps):
            val_loss = evaluate(body, val_batches)
            print(f"step {step:6d} | val loss {val_loss:8.4f}")
            log_jsonl(metrics_path, {"step": step, "val_loss": val_loss, "elapsed": time.time() - t0})

        if args.checkpoint_interval > 0 and (step + 1) % args.checkpoint_interval == 0:
            # 定期覆盖同一个文件（崩溃恢复用）
            save_checkpoint(body, optimizer, step + 1, os.path.join(args.out_dir, "ckpt_last.pt"))
            if args.milestone_interval > 0 and (step + 1) % args.milestone_interval == 0:
                # 关键节点另存（事后分析用）
                save_checkpoint(body, optimizer, step + 1, os.path.join(args.out_dir, f"ckpt_{step + 1}.pt"))

    # ---------- 区块 5：收尾 ----------
    save_checkpoint(body, optimizer, args.total_steps, os.path.join(args.out_dir, "ckpt_final.pt"))
    final_val = evaluate(body, val_batches)
    total_time = time.time() - t0
    print(f"[done] {args.total_steps - start_step} steps in {total_time:.1f}s"
          f" | final train loss {loss_value:.4f} | final val loss {final_val:.4f}")
    log_jsonl(metrics_path, {"step": args.total_steps, "final_val_loss": final_val, "elapsed": total_time})
    return 0


# ============ 第三层：__main__ 守卫 ============
# 只在「直接运行这个文件」时才解析命令行、启动训练。
# import 这个模块（比如别的脚本想复用 main）不会触发任何副作用。

if __name__ == "__main__":
    args = parse_args()
    sys.exit(main(args))
