import torch
import numpy as np
from numpy.typing import NDArray
import torch.nn as nn
from math import sqrt,cos,pi
from einops import einsum
from einops import rearrange
from jaxtyping import Bool, Float, Int
from torch import Tensor
from collections.abc import Callable, Iterable
from typing import Optional
import os
from typing import BinaryIO,IO
import argparse, sys
from tool import data_loader, save_checkpoint, load_checkpoint
from transformer import transformer_lm
from optimizer import AdamW

# ============ 第一层:参数解析函数 ============
# 只负责定义参数、返回配置对象,不碰任何训练逻辑。
# 单独拆出来的好处：可以脱离命令行，直接用 parse_args(["--lr", "1e-3", ...]) 测试或做 sweep。

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a transformer LM")

    # --- 数据 ---
    parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument("--val_data", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)

    # --- 模型（对照 transformer_lm.__init__） ---
    parser.add_argument("--vocab_size", type=int, default=10000)
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--d_ff", type=int, default=1344)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=16)
    parser.add_argument("--rope_theta", type=float, default=10000.0)

    # --- 优化器与调度 ---
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--betas",type = tuple, default = (0.9,0.95))
    parser.add_argument("--eps",type = float, default = 1e-8)
    parser.add_argument("--lr_max", type=float, default=3e-4)
    parser.add_argument("--lr_min", type=float, default=3e-5)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--total_steps", type=int, default=5000)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    # --- 训练过程 ---
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    parser.add_argument("--eval_interval", type=int, default=200)
    parser.add_argument("--checkpoint_interval", type=int, default=500)
    parser.add_argument("--resume_from", type=str, default=None)

    return parser.parse_args(argv)


# ============ 第二层:main(args) —— 完整训练流程 ============
# 接受第一层返回的配置对象，本身不解析命令行 —— 这样才能被别的脚本 import 后直接调用。

def main(args: argparse.Namespace) -> int:
    # 区块 2：准备阶段
    # - 选设备、加载 train/val 的 memmap 数据
    # - 构造 transformer_lm、构造 AdamW
    # - 如果 args.resume_from 不为空，从 checkpoint 恢复
    #bpe 先省略
    x = np.arange(10000)
    inputs, tragets = data_loader(x,args.batch_size,
                                  args.context_length,
                                  args.device)
    body = transformer_lm(args.vocab_size,
                          args.context_length,
                          args.num_layers,
                          args.d_model,
                          args.num_heads,
                          args.d_ff,
                          args.rope_theta,
                          args.device)
    optimizer = AdamW(body.parameters(),
                      args.lr,
                      args.betas,
                      args.eps,
                      args.weight_decay)
    


    # 区块 3 + 4：主循环 + 周期性任务
    # - 按 args.total_steps 循环
    # - 每步：取 batch → forward → cross_entropy → backward → grad_clip → 写 lr → step
    # - 每隔 args.eval_interval 步跑一次验证集评估、打日志
    # - 每隔 args.checkpoint_interval 步存一次 checkpoint
    ...  # TODO

    # 区块 5：收尾
    # - 存最终 checkpoint
    ...  # TODO

    return 0


# ============ 第三层：__main__ 守卫 ============
# 只在「直接运行这个文件」时才解析命令行、启动训练。
# import 这个模块（比如别的脚本想复用 main）不会触发任何副作用。

if __name__ == "__main__":
    args = parse_args()
    sys.exit(main(args))