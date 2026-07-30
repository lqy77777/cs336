# §5.3 `train.py` 代码走读与脚手架待补清单

> 承接 [Section5.3_训练脚本实现指南.md](Section5.3_训练脚本实现指南.md)（讲「要做什么」），本文讲**代码里实际是怎么做的、为什么这么选**，以及**现有脚手架还缺哪些零件**。
> 只动了 `cs336_basics/train.py`，其余文件一行未改——所有对别的文件的建议都写在第 5 节。
> 日期：2026-07-30。

---

## 0. 先跑起来

```bash
# 真实训练（数据必须是已经 tokenize 好的 token id 数组）
uv run python -m cs336_basics.train \
    --train_data data/tinystories_train.bin \
    --val_data   data/tinystories_val.bin \
    --out_dir    runs/exp01 \
    --data_dtype uint16 \
    --vocab_size 10000 --context_length 256 \
    --d_model 512 --d_ff 1344 --num_layers 4 --num_heads 16 \
    --lr_max 3e-4 --lr_min 3e-5 --warmup_steps 200 --total_steps 5000 \
    --batch_size 32 --device auto

# 验收检查②：单 batch 过拟合，loss 必须掉到接近 0
uv run python -m cs336_basics.train --train_data ... --val_data ... --out_dir runs/overfit \
    --overfit --weight_decay 0 --total_steps 200
```

`out_dir` 里会得到四样东西：

| 文件 | 内容 |
|---|---|
| `config.json` | 本次运行的完整超参数（三周后回来看实验时唯一的依据） |
| `metrics.jsonl` | 每条日志一行 JSON，Section 6/7 画曲线直接 `pd.read_json(..., lines=True)` |
| `ckpt_last.pt` | 每 `--checkpoint_interval` 步覆盖写，崩溃恢复用 |
| `ckpt_final.pt` | 收尾存档；`--milestone_interval > 0` 时还会有 `ckpt_{step}.pt` |

---

## 1. 文件骨架：从三层变成四层

指南里说的三层（解析 / `main` / `__main__` 守卫）保持不变，只是在最上面多加了一层**辅助函数**：

| 函数 | 职责 | 为什么要单独拆出来 |
|---|---|---|
| `resolve_device` | `auto` → cuda/mps/cpu，并对「要 cuda 但没有 cuda」直接报错 | 免得在 Mac / 服务器之间来回改命令行 |
| `load_tokens` | memmap 打开 + 值域自检 | 这一段有 10 行，塞进 `main` 会淹没主循环 |
| `make_fixed_batches` | 预先采一组固定不变的 batch | 验证集必须固定，理由见 §2.7 |
| `evaluate` | 在固定 batch 上求平均 loss | `@torch.no_grad()` 直接挂在函数上，比在 `main` 里写 `with` 更不容易漏 |
| `grad_global_norm` | 裁剪前的梯度范数（只用于日志） | 见 §2.10 |
| `log_jsonl` | 追加一行 JSON | 见 §0 |

**判断标准**：主循环里只留「每一步都要做的事」。凡是「准备阶段做一次」或者「细节多但概念简单」的，都往上提。这样 `for step in range(...)` 那一段能一眼看完七个步骤。

---

## 2. `main()` 逐区块走读

### 2.1 区块 1：先校验，再落盘

```python
if args.d_model % args.num_heads != 0: raise ValueError(...)
if args.cosine_cycle_steps is None: args.cosine_cycle_steps = args.total_steps
if args.warmup_steps >= args.cosine_cycle_steps: raise ValueError(...)
```

三条断言对应三种「不断言就会很难查」的错误：

- `d_model % num_heads != 0` → 报错会发生在 `rearrange` 里，信息是一串形状不匹配，看不出根因；
- `T_c` 没给 → 默认等于 `total_steps`（最常见的意图就是「整个训练走完一个余弦周期」）；
- `T_w >= T_c` → `cosine_learning_rate` 里的 `(t-T_w)/(T_c-T_w)` 会**除零**，得到 `inf`/`nan`，然后学习率变成 `nan`，然后**所有参数变成 `nan`**，而 loss 打出来是 `nan` 时你已经离现场很远了。

校验通过后立刻把 `vars(args)` 写进 `out_dir/config.json` 并打印一遍。这不是洁癖：`learning_rate` 实验要跑几十次，「这条曲线是哪组参数」是个真实问题。

### 2.2 种子：两个都要设

```python
torch.manual_seed(args.seed)
np.random.seed(args.seed)
```

`data_loader` 用的是 **numpy 全局**随机数（`np.random.randint`），模型初始化用的是 **torch** 的。只设一个 → 另一半仍然是随机的 → 复现失败。

### 2.3 数据：memmap + 值域自检

```python
data = np.load(path, mmap_mode="r")          # .npy
data = np.memmap(path, dtype=..., mode="r")  # 裸 .bin
```

两种格式都支持：`.npy` 自带 dtype 信息（推荐），裸二进制需要 `--data_dtype` 手动对齐。

值域自检那一句是整个脚本里性价比最高的三行：

```python
probe = data[: min(len(data), 10_000_000)]
if int(probe.min()) < 0 or int(probe.max()) >= vocab_size: raise ValueError(...)
```

实测（故意把 uint16 的文件按 int32 读）：

```
ValueError: train 的 token id 落在 [96, 6291551]，超出 vocab_size=100。
            多半是 --data_dtype（当前 int32）和保存时不一致。
```

两个 `uint16` 被拼成一个巨大的 int32，一眼就露馅。**dtype 读错是不会报错的**，没有这一句，你会训练几个小时然后发现模型什么都没学会。

> 只抽查前 1000 万个 token（读 20 MB，毫秒级）。全量扫一遍 500M token 的文件要读满整个文件，不值得。代价是文件尾部损坏抓不到——可以接受。

### 2.4 模型构造：`device` 传两次是故意的

```python
body = transformer_lm(..., device)   # 直接在目标设备上创建参数
body.to(device)                      # 兜底
```

`transformer_lm` 会把 `device` 一路传给 `Linear` / `Embedding` / `RMSNorm` / RoPE，参数本来就直接建在目标设备上（不必先在 CPU 建好再搬，省一次拷贝和一份峰值内存）。`.to(device)` 是兜底——将来某个子模块忘了接 `device` 参数时不至于崩。

顺带打印参数量，方便对照 §资源核算的估算。

### 2.5 优化器：构造时传的是 `lr_max`，不是 `args.lr`

```python
optimizer = AdamW(body.parameters(), args.lr_max, tuple(args.betas), args.eps, args.weight_decay)
```

因为主循环每一步都会把 `group['lr']` 覆盖掉，**构造时给的 lr 只是个占位值**。传 `lr_max` 而不是 `args.lr` 的理由：万一哪天把调度那两行注释掉了，脚本仍然跑在一个合理的学习率上，而不是一个和调度器毫不相干的 `1e-3`。

`--lr` 本身保留了（不删你的参数），但 help 里标了「不生效」——见 §4。

### 2.6 恢复 checkpoint 的 off-by-one

**约定：`save_checkpoint` 里存的 `iteration` 是「下一步该跑第几步」，不是「刚跑完第几步」。**

```python
save_checkpoint(body, optimizer, step + 1, ...)   # 存的时候 +1
start_step = load_checkpoint(...)                 # 恢复时直接当循环起点
for step in range(start_step, args.total_steps):  # 无缝衔接
```

这样恢复端一个 `+1` 都不用写。反过来（存 `step`）的话，`range(start_step + 1, ...)` 里那个 `+1` 迟早会被漏掉或者写重，**症状是某一步被跑两次或者跳过一步**——loss 曲线上看不出来。

实测（跑 30 步存档 → 恢复跑到 60）：

```
[resume] 从 .../ckpt_last.pt 恢复，从第 30 步继续
step     29 | loss   2.4705       ← 第一次运行的最后一步
step     30 | loss   2.4866       ← 恢复后的第一步，loss 连续
```

⚠️ 恢复时**模型配置必须和存档时一致**。RoPE 的 `cos/sin` 表是 `persistent=False` 的 buffer，不在 checkpoint 里，是构造时按 `context_length` / `rope_theta` 重新算的——配置写错了表就算错了，而 `load_state_dict` 完全察觉不到（它只管 `weight` 这类真参数）。`config.json` 就是给恢复时对照用的。

### 2.7 固定验证 batch：为什么要存取 numpy 的 RNG state

```python
state = np.random.get_state()
np.random.seed(seed)
batches = [data_loader(...) for _ in range(num_batches)]
np.random.set_state(state)
```

两层考虑：

1. **验证集必须固定**。每次验证都重新随机采样的话，曲线的抖动来自采样噪声而不是模型，你根本没法判断模型有没有变好。所以预先采好一组，之后一直喂这一组。
2. **采完要把 RNG 状态还回去**。`data_loader` 吃的是 numpy 全局随机数——如果在这里连采 10 次却不还原，训练用的随机序列就会**跟着 `--eval_batches` 变**。那意味着「只改了验证 batch 数量」这种和模型完全无关的操作，会让整条训练曲线变样，两次实验没法对照。

第 2 点是那种「不出错就永远想不到、出错了要查半天」的地方。

### 2.8 主循环的七步

```python
for step in range(start_step, args.total_steps):
    lr = cosine_learning_rate(step, lr_max, lr_min, warmup_steps, cosine_cycle_steps)
    for group in optimizer.param_groups: group["lr"] = lr   # ① 写进 param_groups
    inputs, targets = data_loader(...)                      # ②
    logits = body(inputs); loss = cross_entropy(logits, targets)  # ③
    optimizer.zero_grad(set_to_none=True); loss.backward()  # ④ 清零必须在 backward 之前
    gradient_clipping(body.parameters(), args.grad_clip)    # ⑤ 夹在 backward 和 step 之间
    optimizer.step()                                        # ⑥
    ...                                                     # ⑦ 日志 / 验证 / 存档
```

三个顺序约束在代码里的落点：

| 约束 | 落点 | 放错了会怎样 |
|---|---|---|
| 清梯度在反向之前 | ④ 里 `zero_grad` 紧挨着 `backward` 写在同一行区域 | 梯度累加，第 t 步等效学习率被放大 t 倍，**不报错** |
| 裁剪夹在反向和更新之间 | ⑤ 在 ④ 和 ⑥ 之间 | 放前面没梯度可裁；放后面参数已经被推飞 |
| lr 写进 `param_groups` | ① 显式遍历 `optimizer.param_groups` | 若 AdamW 读的是实例属性，调度器**静默失效** |

关于第三条——你的 `AdamW.step` 里写的是 `lr = group["lr"]`（optimizer.py:44），**契约是对的**，调度器能生效。实测日志里 lr 走的是「0 → 1e-3 线性上升 → 余弦下降到 3e-5」，不是水平线：

```
step  0 | lr 0.000e+00      ← warmup 起点
step 10 | lr 1.000e-03      ← warmup 结束，到达 lr_max
step 20 | lr 9.074e-04      ← 余弦下降
step 59 | lr 3.096e-05      ← 接近 lr_min
```

另外 `cross_entropy` 直接吃 `(B, m, vocab)` 对 `(B, m)`，**不需要手动 flatten 成二维**——你的实现里 `gather` 和 `mean` 都是按最后一维广播的，前面有多少批量维都无所谓。

### 2.9 `zero_grad(set_to_none=True)`

置 `None` 而不是填 0：省一次显存写入，也让「某个参数从头到尾没参与计算」这种 bug 更容易暴露（`p.grad` 一直是 `None` 而不是一片 0）。`gradient_clipping` 和 `AdamW.step` 里都写了 `if p.grad is None: continue`，兼容。

### 2.10 梯度范数只在打日志的那几步算

```python
need_log = (step % args.log_interval == 0) or (step == args.total_steps - 1)
pre_clip_norm = grad_global_norm(body.parameters()) if need_log else None
gradient_clipping(body.parameters(), args.grad_clip)
```

`gradient_clipping` 内部已经算过一次全局范数，但**没有返回它**（见 §5-B1）。想记录「裁剪前的范数」只能再算一遍。每步都算等于白白多遍历一次全部参数，所以只在日志步算——`--log_interval 20` 的话开销摊薄到 1/20。

为什么值得记：**梯度范数的尖峰比 loss 更早预警发散**。loss 炸掉的时候已经晚了，范数持续攀升的时候还来得及降学习率。

### 2.11 checkpoint 策略：覆盖 + 里程碑

```python
save_checkpoint(..., "ckpt_last.pt")                       # 每 checkpoint_interval 步覆盖
if milestone_interval > 0 and ...: save_checkpoint(..., f"ckpt_{step+1}.pt")   # 另存
```

只留最新的 → 训练崩了没法回退到更早的健康状态；每次都另存 → 磁盘塞满，训练在深夜静静死掉。默认 `--milestone_interval 0`（只留最新），要做事后分析时再打开。

---

## 3. 新增的 `--overfit` 开关

```python
if args.overfit:
    fixed_train_batch = data_loader(...)   # 只采一次，之后每步都喂它
```

这是把指南 §5-② 那个「最强 sanity check」做成了一个开关，四行代码：拿一个 batch 反复训练，**loss 必须能降到接近 0**。

实测（`--overfit --weight_decay 0`，200 步）：

```
step   0 | loss 4.9982
step  50 | loss 0.0845
step 199 | loss 0.0105     ← 训练 loss 压到 0.01
[done] final train loss 0.0105 | final val loss 1.4807   ← 验证 loss 高，正是过拟合的样子
```

这一条通过 = 模型结构、前向、反向、优化器**全链路正常**，剩下的问题都是超参数问题。它的价值在于**把「实现错误」和「超参数不好」彻底分开**——没有它，你会在两者之间反复横跳。

---

## 4. 对 `parse_args` 的改动清单

**修正 1 处：**

| 参数 | 原来 | 现在 | 原因 |
|---|---|---|---|
| `--betas` | `type=tuple` | `type=float, nargs=2` | `type=tuple` 是坏的：argparse 会把命令行字符串 `"0.9,0.95"` 喂给 `tuple()`，得到 `('0','.','9',',','0','.','9','5')`——**8 个字符**。不传 `--betas` 时用默认值没事，一传就炸，而且报错发生在 AdamW 内部。改成 `nargs=2` 后写法是 `--betas 0.9 0.95` |

**新增 8 个（都给了默认值，不传也能跑）：**

| 参数 | 默认 | 为什么必须有 |
|---|---|---|
| `--seed` | 0 | 不设种子就没有复现，验收检查③（中断恢复）根本没法做 |
| `--data_dtype` | `uint16` | 读裸二进制必须知道 dtype，读错静默出垃圾 |
| `--cosine_cycle_steps` | `None`→`total_steps` | `cosine_learning_rate` 的 $T_c$。原脚手架只有 `warmup_steps` 和 `total_steps`，$T_c$ 没有出口；两者分开才能做「余弦周期比训练长/短」的实验 |
| `--log_interval` | 20 | 原来只有 eval / checkpoint 两个周期，训练 loss 打印的频率没有出口 |
| `--eval_batches` | 10 | 单个 batch 的验证 loss 噪声太大，曲线没法看，必须平均多个 |
| `--milestone_interval` | 0 | 「覆盖 + 另存」两种 checkpoint 策略的开关 |
| `--overfit` | False | 验收检查②，见 §3 |
| `--device` 增加 `auto` | `cpu` | 在 Mac(mps) / 服务器(cuda) / 本机(cpu) 之间切换不用改命令行。默认值保持 `cpu` 不变 |

**保留但不生效 1 个：**

- `--lr`：调度器每步都会覆盖 `group['lr']`，它注定不生效。没有删（那是你的参数），只在 help 里标注了。**建议直接删掉**，或者改名成 `--lr` 作为 `--lr_max` 的别名——两个都叫「学习率」的参数并存，三周后一定会拿错。

---

## 5. 脚手架还缺什么

### A. 必须补，否则跑不了真实数据 —— 数据准备管道

`train.py` 吃的是**已经 tokenize 好的 token id 数组**。从原始 `.txt` 到这个数组，中间这一段目前是空的：

| 缺什么 | 现状 | 位置 |
|---|---|---|
| **A1. tokenizer 存盘** | `train_bpe` 只把 `(vocab, merges)` 返回到内存，进程一退就没了 | `bpe.py` 没有对应的序列化函数 |
| **A2. `Tokenizer.from_files`** | 函数体是 `pass`（bpe.py:163-170），返回 `None` | 和 A1 是一对：存不了也读不回来 |
| **A3. 语料编码脚本** | 没有「读 .txt → `encode_iterable` → `np.array(dtype=uint16)` → 存盘」这一步 | 建议新建 `cs336_basics/prepare_data.py` |

补 A3 时的三个要点：

- **dtype 选 `uint16`**：vocab 10000 < 65536，用 uint16 比 int32 省一半磁盘和一半 I/O。存的时候用什么，`--data_dtype` 就得填什么。
- **用 `encode_iterable` 逐行喂**，不要把整个 TinyStories 读成一个字符串再 `encode`——内存扛不住。
- **train / val 分开存两个文件**，不要在一个数组里切——切片边界附近的样本会跨越两边，造成数据泄漏。

> ⚠️ 另外 `Tokenizer.encode` 目前对**每个 pretoken 都重新跑一遍完整的 merge 循环**（每轮还要重建 `pairs` 集合和 `ranks` 字典）。语料里 `" the"` 这种 pretoken 会出现上百万次，每次都从头算。加一个 `dict[tuple[bytes,...], list[int]]` 缓存（或 `functools.lru_cache`）能把 tokenize 整个语料的时间降一到两个数量级。**这一步不优化，光 tokenize 就要跑很久。**

### B. 建议补，不补也能跑

| # | 位置 | 问题 | 建议 |
|---|---|---|---|
| **B1** | `optimizer.py: gradient_clipping` | 内部算了全局范数但不返回，想记日志只能再算一遍（§2.10） | 末尾 `return norm`，`train.py` 就能白拿这个指标 |
| **B2** | `tool.py: data_loader` | 用 numpy **全局**随机数，采样的随机状态没法保存 → 严格意义上中断恢复不可复现（喂的 batch 序列变了） | 加一个 `rng: np.random.Generator = None` 参数，把 generator 的 state 一起存进 checkpoint |
| **B3** | `tool.py: save_checkpoint` | 只存 model / opt / iteration，不存 RNG 状态，也不存超参数配置 | 把 `vars(args)` 和 RNG state 一起塞进去。现在的替代方案是 `train.py` 单独写 `config.json`，但它和 checkpoint 是分离的，容易对不上 |
| **B4** | `tool.py: load_checkpoint` | `torch.load` 没有 `map_location` | 在 GPU 上存的 checkpoint 拿到 CPU 机器上会直接报错。加 `map_location="cpu"` 最省心 |
| **B5** | `optimizer.py: AdamW.step` | 注释写着「必须原地运算」，但 `m = beta_1 * m + (1-beta_1) * grad` 实际上**每步都新建张量**（`m.mul_(b1).add_(grad, alpha=1-b1)` 才是原地） | 功能完全正确，只是每步多分配 2× 参数量的显存并多一次拷贝。CPU 上跑小模型无所谓，上 GPU 跑大模型时值得改 |
| **B6** | `optimizer.py: AdamW.step` | weight decay 写在 Adam 更新**之前**，handout 的伪代码是写在之后 | 差异是 $O(\alpha^2)$ 量级，`test_adamw` 对两种顺序都放行（它同时对照 PyTorch 和参考实现）。知道有这回事即可 |
| **B7** | 实验记录 | 只有 stdout + `metrics.jsonl` | `pyproject.toml` 里已经装了 `wandb`。Section 6/7 要跑几十次实验，接上 wandb 会比自己读 jsonl 舒服很多 |

> 顺带确认两个**没有问题**的地方，省得你怀疑：
> - `data_loader` 的 `np.random.randint(0, n-m)` 边界是对的：起点最大 `n-m-1`，targets 最远取到 `n-1`，不越界。
> - `gradient_clipping` 开头的 `p = list(p)` 已经把生成器固化了，`train.py` 里传 `model.parameters()` 不会踩「第二遍遍历拿到空」的坑。

---

## 6. 验收：四个检查怎么做

按顺序做，**前面的不过就别做后面的**。以下是在一份 20 万 token 的合成数据（周期序列，vocab=100）上的实测结果。

### ① 初始 loss ≈ ln(vocab_size) ✅

```
[sanity] 初始 loss = 5.0027，期望 ≈ ln(vocab_size) = 4.6052
```

脚本会在第一步自动打这一行（恢复训练时不打，因为那时模型已经不是随机初始化了）。

**略高于 $\ln V$ 是正常的**：$\ln V$ 对应「logits 完全相等」的理想情形，随机初始化的 logits 有方差，而交叉熵在 logits 有方差时**只会更高**。差 0.4 属于正常范围；差到 15+ 才说明初始化方差失控。真实配置（vocab=10000）下这个值应该落在 **9.21 稍偏上**。

### ② 单 batch 过拟合 ✅

见 §3，实测 loss 200 步压到 0.0105。**这是最强的一条**，通过了就说明实现没问题。

### ③ 中断恢复曲线连续 ✅

见 §2.6，实测 step 29 → step 30 的 loss 连续（2.4705 → 2.4866）。

⚠️ 严格意义上的「两条曲线完全重合」目前**做不到**，因为 `data_loader` 的随机状态没有存进 checkpoint（B2/B3）——恢复后 numpy 的随机序列是从 `seed` 重新开始的，喂的 batch 和不中断时不同。**能看的是「loss 没有跳变」**，这已经能抓出「模型/优化器状态没恢复对」这类真 bug（那种情况下 loss 会明显反弹）。

### ④ 学习率曲线符合预期 ✅

见 §2.8 的实测数据，也可以直接画：

```python
import pandas as pd
df = pd.read_json("runs/exp01/metrics.jsonl", lines=True)
df.plot(x="step", y="lr")          # 应该是：线性上升 → 余弦下降 → 平台
df.plot(x="step", y=["train_loss", "val_loss"])
```

**如果 lr 是一条水平线**，说明调度器没生效——去查「lr 有没有写进 `param_groups`」和「AdamW 有没有从 `group['lr']` 读」。

---

## 7. 相关笔记

- [Section5.3_训练脚本实现指南.md](Section5.3_训练脚本实现指南.md) —— 「要做什么」，本文是「怎么做的」
- [Section5_训练循环任务清单.md](Section5_训练循环任务清单.md) —— §5.1 / §5.2 的零件
- [argparse使用手册.md](argparse使用手册.md) —— `type` / `nargs` / `action` 的语义（§4 那个 `type=tuple` 的坑）
- [4.2_SGD优化器与Optimizer_API详解.md](4.2_SGD优化器与Optimizer_API详解.md) —— `param_groups` 与 `state` 的契约
- [register_buffer详解.md](register_buffer详解.md) —— `persistent=False` 与 checkpoint 的关系（§2.6 的坑）
- [torch.no_grad详解.md](torch.no_grad详解.md) —— `evaluate` 上那个装饰器
- [距离训练一个模型还缺什么.md](距离训练一个模型还缺什么.md)
