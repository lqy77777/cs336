# `train.py` 逐函数逐行详解与代码审查

> 审查对象：当前版本的 `cs336_basics/train.py`，并对照它直接调用的 `tool.py`、`optimizer.py` 和 `transformer.py`。
>
> 本文只解释和审查代码，没有修改任何训练源码。行号以审查时的文件版本为准。

## 0. 先说结论

`train.py` 的总体结构是合理的：辅助函数、参数解析、`main()` 和 `__main__` 守卫分层清楚，数据使用 memmap，训练循环的 `forward → loss → zero_grad → backward → clip → step` 顺序也正确。

但在开始正式训练之前，有几项必须处理：

| 优先级 | 问题 | 影响 |
|---|---|---|
| P0 | `optimizer.gradient_clipping()` 覆盖了全局范数变量，导致不同参数的梯度交替缩小和放大 | 直接破坏训练正确性，必须先修 |
| P1 | 当前只有 train token `.bin`，还缺用同一个 tokenizer 编码得到的 validation `.bin` | 不能做独立验证；拿 train 兼作 val 会造成数据泄漏 |
| P1 | `python -m cs336_basics.train` 会因 `from tool import ...` 失败 | 包模式无法启动，只能直接运行脚本 |
| P1 | 新实验复用同一个 `out_dir` 时，旧 `metrics.jsonl` 会被继续追加 | 多次实验的数据混在一起，曲线失真 |
| P1 | `milestone_interval` 被嵌套在普通 checkpoint 条件内 | 不会真的按 milestone 周期保存，只在两个周期同时命中时保存 |
| P1 | checkpoint 不保存 NumPy/Torch RNG 状态 | 中断恢复后 batch 序列不同，无法严格复现实验 |
| P2 | 最后一轮已经验证一次，循环结束后又对同一模型、同一固定 batch 验证一次 | 纯重复计算 |
| P2 | `--lr` 参数完全不生效 | 接口冗余，容易误导实验配置 |
| P2 | `log_interval=0` 未校验，会发生取模除零 | 参数边界会导致运行时崩溃 |
| P2 | `load_tokens()` 未检查一维整数数组，且最小长度判断多要求了一个 token | 可能静默接受错误 `.npy`，也会拒绝本来合法的最小数据 |

已经实际验证的结果：

- `tests/test_optimizer.py`：AdamW 和 cosine learning rate 测试通过。
- `tests/test_data.py`：data loader 测试通过。
- `tests/test_serialization.py`：checkpoint 测试通过。
- `tests/test_nn_utils.py::test_gradient_clipping`：失败。
- 用三个梯度都为 `10` 的单元素参数测试，裁剪上限为 `1` 时，当前实现得到约 `[0.577, 173.202, 0.577]`，裁剪后的全局范数约为 `173.204`，显然违背“裁剪到不超过 1”的目标。

## 1. `train.py` 在整个项目中的职责

这个文件不负责：

- 从 `.txt` 训练 BPE；
- 把文本编码成 token ID；
- 定义 Transformer 各个层；
- 实现 AdamW、交叉熵和数据采样。

它负责把已经实现好的组件串起来：

```text
train.bin / valid.bin
→ memmap 打开
→ 随机采 batch
→ Transformer forward
→ cross entropy
→ backward
→ gradient clipping
→ AdamW step
→ 日志 / 验证 / checkpoint
```

输入包括：

- 训练 token 文件；
- 验证 token 文件；
- 命令行模型与优化器参数；
- 可选 checkpoint。

输出包括：

- `config.json`；
- `metrics.jsonl`；
- `ckpt_last.pt`；
- 可选 `ckpt_{step}.pt`；
- `ckpt_final.pt`。

## 2. import 区域审查

当前第 1–14 行导入了很多并未在 `train.py` 中使用的名字。静态检查报告 13 个未使用导入和一处“一行多个 import”。

### 2.1 实际使用的导入

| 导入 | 用途 |
|---|---|
| `torch` | 设备、随机种子、`no_grad`、Tensor 计算 |
| `numpy as np` | memmap、随机状态、数据抽查 |
| `NDArray` | 类型标注 |
| `torch.nn as nn` | `evaluate()` 的模型类型标注 |
| `log` | 打印初始 loss 的理论参考值 `ln(vocab_size)` |
| `Tensor` | batch 类型标注 |
| `os` | 创建目录、拼接输出路径 |
| `argparse` | 命令行参数解析 |
| `sys` | `sys.exit(main(args))` |
| `json` | 配置与 JSONL 日志 |
| `time` | 训练耗时和吞吐量 |

### 2.2 未使用的导入

```python
sqrt, cos, pi
einsum, rearrange
Bool, Float, Int
Callable, Iterable
Optional
BinaryIO, IO
```

这些大多像是从其他模块复制过来的。它们不影响结果，但增加阅读成本，让人误以为 `train.py` 会直接进行张量布局或数学计算。

### 2.3 可简化的 import

从当前使用情况看，可以缩减到类似：

```python
import argparse
import json
import os
import sys
import time
from math import log

import numpy as np
import torch
import torch.nn as nn
from numpy.typing import NDArray
from torch import Tensor
```

本地模块导入还涉及启动方式问题，见第 11 节。

## 3. `resolve_device()`

### 3.1 作用

把命令行中的设备名称转换成 `torch.device`。当输入 `auto` 时，按 CUDA、MPS、CPU 的优先级选择设备。

### 3.2 输入

```python
name: str
```

正常来自：

```text
auto / cpu / cuda / mps
```

### 3.3 输出

```python
torch.device
```

### 3.4 逐行解释

第 24 行：

```python
def resolve_device(name: str) -> torch.device:
```

定义函数，输入字符串，输出 PyTorch 设备对象。

第 25 行：

```python
if name == "auto":
```

只有在自动模式下才进行硬件探测。

第 26–27 行：

```python
if torch.cuda.is_available():
    return torch.device("cuda")
```

如果 NVIDIA CUDA 可用，优先选择 CUDA。

第 28–29 行：

```python
if torch.backends.mps.is_available():
    return torch.device("mps")
```

如果 CUDA 不可用但 Apple MPS 可用，选择 MPS。

第 30 行：

```python
return torch.device("cpu")
```

前两者都不可用时回退 CPU。

第 31–32 行：

```python
if name == "cuda" and not torch.cuda.is_available():
    raise RuntimeError(...)
```

用户明确要求 CUDA，但机器不支持时，尽早给出可读错误。

第 33 行：

```python
return torch.device(name)
```

处理显式的 `cpu`、`mps` 或 `cuda`。

### 3.5 问题与改进

显式 `cuda` 有可用性检查，但显式 `mps` 没有。可以对称地增加：

```python
if name == "mps" and not torch.backends.mps.is_available():
    raise RuntimeError("--device mps 但这台机器没有可用的 MPS")
```

默认参数现在是 `cpu`，所以用户不写 `--device auto` 时，即使 Mac 支持 MPS 也不会自动使用。这个选择不算错误，但要确认是否符合预期。

## 4. `load_tokens()`

### 4.1 作用

以内存映射方式打开 `.npy` 或裸 `.bin` token 文件，并进行长度和值域抽查。

### 4.2 输入

```python
path: str
dtype: str
vocab_size: int
context_length: int
tag: str
```

- `path`：token 文件路径；
- `dtype`：裸 `.bin` 的元素类型，例如 `uint16`；
- `vocab_size`：合法 token ID 上界；
- `context_length`：每条训练样本的输入长度；
- `tag`：错误信息中的数据集名称，例如 `train` 或 `val`。

### 4.3 输出

返回 NumPy 内存映射数组。对 `.npy` 通常是带 mmap 的数组；对 `.bin` 是 `np.memmap`。

### 4.4 逐行解释

第 49–50 行：

```python
if path.endswith(".npy"):
    data = np.load(path, mmap_mode="r")
```

`.npy` 自带 dtype 和 shape，因此不需要手动指定；`mmap_mode="r"` 表示只读按需加载。

第 51–52 行：

```python
else:
    data = np.memmap(path, dtype=np.dtype(dtype), mode="r")
```

其他后缀全部按裸二进制读取。裸文件没有头部，所以必须依赖调用方提供正确 dtype。

第 54–55 行：

```python
if len(data) < context_length + 2:
    raise ValueError(...)
```

检查数据能否提供一个输入窗口及其右移一位的 target。

这里存在一个 off-by-one：`data_loader()` 对长度 `n` 和上下文 `m` 使用 `randint(0, n-m)`；只要 `n=m+1`，起点 0 就合法，inputs 使用 `[0, m-1]`，targets 使用 `[1, m]`。因此正确最小长度是：

```python
context_length + 1
```

实测 `17` 个 token、`context_length=16` 本来能够产生一个 batch，却被当前检查拒绝。

第 57 行：

```python
probe = data[: min(len(data), 10_000_000)]
```

只抽查前 1000 万个 token，避免为检查而读取完整 GB 级文件。

第 58 行：

```python
lo, hi = int(probe.min()), int(probe.max())
```

统计抽样最小和最大 token ID，并转换成 Python int 便于打印和 JSON 处理。

第 59–63 行：

```python
if lo < 0 or hi >= vocab_size:
    raise ValueError(...)
```

ID 必须属于 `[0, vocab_size)`。裸文件 dtype 读错时，经常会产生巨大整数，这项检查能较早发现问题。

第 64 行打印数据规模、dtype、抽样值域。

第 65 行返回内存映射数组。

### 4.5 问题与改进

第一，应该检查一维：

```python
if data.ndim != 1:
    raise ValueError(f"{tag} 必须是一维 token 序列")
```

第二，应该检查整数 dtype：

```python
if not np.issubdtype(data.dtype, np.integer):
    raise TypeError(f"{tag} dtype 必须是整数，实际为 {data.dtype}")
```

否则一个值都在范围内的 float `.npy` 会通过 min/max，随后在 `torch.tensor(..., dtype=int64)` 时被静默截断。

第三，长度判断应改为：

```python
if len(data) < context_length + 1:
```

第四，当前只抽查文件开头，不能发现尾部损坏。这是速度与完整性的取舍，不是错误；可以在数据准备阶段做一次全量校验或保存哈希。

第五，当前已经在 tokenizer JSON 中记录了 `.bin` 的 dtype 和 shape，但 `train.py` 不读取这份 metadata，仍要求命令行重复填写 `--data_dtype` 和 `--vocab_size`。可以保持显式 CLI，也可以新增 manifest/config 参数来消除重复来源；两种方案只选一种，避免配置不一致。

## 5. `make_fixed_batches()`

### 5.1 作用

提前从验证集随机采样固定的一组 batch。之后每次验证都使用同样的数据，避免验证曲线因为每次随机样本不同而抖动。

### 5.2 输入

```python
data: NDArray
batch_size: int
context_length: int
device: torch.device
num_batches: int
seed: int
```

### 5.3 输出

```python
list[tuple[Tensor, Tensor]]
```

列表中的每一项是 `(inputs, targets)`，形状通常都是：

```text
(batch_size, context_length)
```

### 5.4 逐行解释

第 83 行：

```python
state = np.random.get_state()
```

保存 NumPy 全局随机数生成器的当前状态。

第 84 行：

```python
np.random.seed(seed)
```

临时切换到固定验证 seed。

第 85 行：

```python
batches = [data_loader(...) for _ in range(num_batches)]
```

连续调用 `data_loader()`，提前生成固定 batch。由于 `data_loader()` 直接把结果放到 `device`，这些 batch 会常驻 CPU/GPU/MPS 内存。

第 86 行：

```python
np.random.set_state(state)
```

恢复训练采样原本的随机状态，避免 `eval_batches` 数量影响之后的训练 batch 序列。

第 87 行返回固定 batch 列表。

### 5.5 问题与改进

如果第 85 行抛异常，第 86 行不会执行，全局 RNG 状态会被留在验证 seed。最低限度应该使用 `try/finally`。

更好的长期设计是让 `data_loader()` 接收独立的：

```python
np.random.Generator
```

训练 RNG 和验证 RNG 各自独立，就不需要保存、切换和恢复全局状态。

默认配置下固定 batch 占用不大：

```text
10 batches × 32 × 256 × 2 tensors × 8 bytes ≈ 1.25 MiB
```

但更大的 `eval_batches`、`batch_size` 和 `context_length` 会让它们长期占据设备内存。可选优化是只保存固定起点索引，验证时再从 memmap 生成 tensor。

## 6. `evaluate()`

### 6.1 作用

在固定验证 batch 上计算平均交叉熵，并禁止构建反向传播计算图。

### 6.2 输入

```python
model: nn.Module
batches: list[tuple[Tensor, Tensor]]
```

### 6.3 输出

```python
float
```

即所有验证 batch loss 的算术平均。

### 6.4 逐行解释

第 90 行：

```python
@torch.no_grad()
```

装饰整个函数，关闭 autograd 图构建，降低验证内存和计算开销。

第 93 行：

```python
model.eval()
```

切换到评估模式。当前模型没有 dropout，但保持这个习惯是正确的。

第 94 行初始化 Python 浮点累加器。

第 95–96 行：

```python
for inputs, targets in batches:
    total += cross_entropy(model(inputs), targets).item()
```

逐 batch 前向，计算平均交叉熵，`.item()` 把零维 Tensor 变成 Python float。

第 97 行：

```python
model.train()
```

恢复训练模式。

第 98 行返回平均 loss。

### 6.5 问题与改进

函数无条件恢复到 train 模式，不能保持调用前状态。更稳妥的是：

```python
was_training = model.training
model.eval()
try:
    ...
finally:
    model.train(was_training)
```

如果函数只在训练循环中调用，当前行为不会出错，但作为通用辅助函数不够严谨。

还可以用 `torch.inference_mode()` 替代 `no_grad()`，验证场景通常略省开销。不过收益远小于模型前向本身。

函数本身没有检查空 batch；`main()` 已确保 `eval_batches > 0`，所以当前调用路径安全。

## 7. `log_jsonl()`

### 7.1 作用

把一条 Python 字典序列化为单行 JSON，并追加到日志文件末尾。

### 7.2 输入

```python
path: str
record: dict
```

### 7.3 输出

无返回值，副作用是追加文件内容。

### 7.4 逐行解释

第 105 行：

```python
with open(path, "a") as f:
```

以追加模式打开文件。每次调用结束都会关闭文件，因此已经写完的日志即使训练崩溃也通常能保留。

第 106 行：

```python
f.write(json.dumps(record) + "\n")
```

每条记录独占一行。整个 `.jsonl` 不是一个 JSON 数组，读取时需要逐行解析或使用 `pandas.read_json(..., lines=True)`。

### 7.5 问题与改进

应该显式指定：

```python
encoding="utf-8"
```

更重要的是，`main()` 总是使用追加模式，却不会在“全新实验”开始时清空旧文件。复用同一个 `out_dir` 会产生：

```text
旧实验 metrics
+ 新实验 metrics
```

而 `config.json` 又会被新实验覆盖，最终同一个 metrics 文件对应两套配置。

建议明确规则：

- 新实验且没有 `--resume_from`：要求 out_dir 为空，或清空 `metrics.jsonl`；
- 恢复训练：继续追加现有 metrics；
- 更稳妥：每次实验都使用唯一 run 目录。

每次日志都 open/close 有少量开销，但默认每 20 步一次，不是训练瓶颈，而且它提高了崩溃后的日志完整性，不必急着优化。

## 8. `parse_args()`

### 8.1 作用

定义全部命令行参数，并返回 `argparse.Namespace`。

### 8.2 输入

```python
argv: list[str] | None = None
```

- `None`：解析真实的 `sys.argv[1:]`；
- list：解析调用者提供的参数，便于测试或 sweep。

### 8.3 输出

```python
argparse.Namespace
```

### 8.4 参数逐项解释

#### 数据参数

| 参数 | 作用 | 备注 |
|---|---|---|
| `--train_data` | 训练 token 文件 | 必填，支持 `.npy` 或裸 `.bin` |
| `--val_data` | 验证 token 文件 | 必填，必须与 train 使用同一个 tokenizer |
| `--out_dir` | 配置、日志、checkpoint 目录 | 必填；复用目录有日志混合风险 |
| `--data_dtype` | 裸 `.bin` 的 dtype | 默认 `uint16`；`.npy` 分支忽略 |

#### 模型参数

| 参数 | 作用 |
|---|---|
| `--vocab_size` | 词表大小，也是输出 logits 最后一维 |
| `--context_length` | 每个样本的 token 长度和 RoPE 最大长度 |
| `--d_model` | Transformer 隐藏维度 |
| `--d_ff` | SwiGLU FFN 中间维度 |
| `--num_layers` | Transformer block 数量 |
| `--num_heads` | 注意力头数 |
| `--rope_theta` | RoPE 基频参数 |

#### 优化器与学习率参数

| 参数 | 作用 | 审查结论 |
|---|---|---|
| `--lr` | 声称是学习率 | 实际完全不生效，应删除或改为 `alpha_max` 的别名 |
| `--betas` | AdamW 的两个动量系数 | `nargs=2` 正确；显式传入时先得到 list，构造优化器前转 tuple |
| `--eps` | AdamW 数值稳定项 | 应校验大于 0 |
| `--alpha_max` | warmup 顶点/余弦起点学习率 | 当前优化器初始 lr 也使用它 |
| `--alpha_min` | 余弦结束后的最小学习率 | 应校验非负且不大于 alpha_max |
| `--T_w` | warmup 步数 | 0 表示不 warmup；应校验非负 |
| `--T_c` | 余弦周期结束步数 | None 时在 main 中设成 total_steps |
| `--total_steps` | 训练更新总步数 | 必须为正 |
| `--weight_decay` | AdamW 解耦权重衰减 | 应校验非负 |
| `--grad_clip` | 梯度全局范数上限 | 应校验大于 0；当前底层实现有严重 bug |

#### 训练过程参数

| 参数 | 作用 | 审查结论 |
|---|---|---|
| `--batch_size` | 每步样本数 | 必须为正 |
| `--device` | cpu/cuda/mps/auto | 默认 cpu，不会自动启用 Mac MPS |
| `--seed` | Torch 和 NumPy 初始 seed | checkpoint 未保存 RNG state，恢复仍不严格复现 |
| `--log_interval` | 训练日志周期 | 必须大于 0，否则 `step % 0` 崩溃 |
| `--eval_interval` | 验证周期 | 小于等于 0 时禁用周期验证，但结尾仍会验证 |
| `--eval_batches` | 每次验证平均多少个固定 batch | 必须为正 |
| `--checkpoint_interval` | `ckpt_last.pt` 周期 | 小于等于 0 时禁用周期 checkpoint |
| `--milestone_interval` | 独立里程碑 checkpoint 周期 | 当前实现被错误嵌套，不能独立生效 |
| `--resume_from` | 恢复 checkpoint 路径 | 模型结构和原配置必须一致 |
| `--overfit` | 固定一个训练 batch 反复拟合 | 用于训练链路 sanity check |

第 160 行：

```python
return parser.parse_args(argv)
```

真正执行解析并返回 Namespace。

### 8.5 参数校验缺口

当前 main 只校验了部分参数。建议至少补充：

```text
vocab_size > 0
context_length > 0
d_model > 0
d_ff > 0
num_layers > 0
num_heads > 0
T_w >= 0
T_c > T_w
0 <= alpha_min <= alpha_max
0 <= beta1, beta2 < 1
eps > 0
weight_decay >= 0
grad_clip > 0
log_interval > 0
eval_interval >= 0
checkpoint_interval >= 0
milestone_interval >= 0
```

## 9. `main()`：完整训练流程

### 9.1 输入和输出

输入：

```python
args: argparse.Namespace
```

输出：

```python
int
```

成功训练后返回 `0`，交给 `sys.exit()` 作为进程退出码。

### 9.2 第 166–176 行：配置校验

第 169–170 行：

```python
if args.d_model % args.num_heads != 0:
    raise ValueError(...)
```

多头注意力要求每头维度 `d_model // num_heads` 是整数。还应先保证 `num_heads > 0`，否则这里会先除零。

第 171–172 行：

```python
if args.T_c is None:
    args.T_c = args.total_steps
```

未指定余弦周期时，让它覆盖整个训练过程。

第 173–174 行确保 warmup 结束早于 cosine 周期结束，避免调度公式分母为 0 或负数。

第 175–176 行确保 batch、总步数和验证 batch 数量为正。这里遗漏了 `log_interval` 等前述参数。

### 9.3 第 178–183 行：输出目录和配置

第 178 行：

```python
os.makedirs(args.out_dir, exist_ok=True)
```

创建输出目录；目录已存在时不报错。

第 179 行：

```python
config = vars(args)
```

取得 Namespace 内部字典。它不是副本，而是 `args.__dict__` 本身；当前后面没有继续修改 args，所以没有实际问题。为了语义清晰可写 `vars(args).copy()`。

第 180–181 行将配置覆盖写入 `config.json`。`default=str` 能兜底非 JSON 类型，但也可能把本应暴露的错误静默转成字符串；当前 argparse 值本来都是 JSON 兼容类型，不需要它。

第 182 行生成 metrics 路径。

第 183 行打印完整配置。

关键问题是：config 覆盖写，metrics 追加写。复用 out_dir 时两者生命周期不一致。

### 9.4 第 185–197 行：随机种子、设备和数据

第 187 行设置 Torch seed，影响模型初始化等 Torch 随机操作。

第 188 行设置 NumPy seed，影响 `data_loader()` 的随机起点。

第 191 行解析设备，第 192 行打印。

第 195–196 行分别以内存映射打开 train 和 val，并进行抽样检查。

当前项目中已经生成了 train `.bin`，但还需要用同一个 `tinystories_tokenizer.json` 编码 validation 文本，得到独立 validation `.bin`。不能在 validation 上重新训练 tokenizer，也不应把 train `.bin` 同时传给 `--train_data` 和 `--val_data`。

### 9.5 第 198–210 行：模型构造

第 199–206 行把命令行模型参数传给 `transformer_lm`。

参数 `device` 在构造时传入，使各层参数、RoPE buffer 和 attention mask 的设备配置一致。

第 207 行：

```python
body.to(device)
```

因为模型已经在目标设备构造，这一行通常不会搬运数据，看起来有些冗余；但它可以兜底未来某个子模块忘记接受 device 的情况。

需要注意：不能简单改成“始终在 CPU 构造，再 `.to(device)`”而不改 `transformer.py`。attention 模块保存了 `self.device` 并用它创建 mask 和 position，`.to()` 不会自动修改普通 Python 属性 `self.device`。

第 208 行 `body.train()` 是显式声明。PyTorch 模块初始化后默认就是训练模式，所以当前属于可读性冗余，不是性能问题。

第 209 行统计全部可训练和不可训练 Parameter 的元素数量；当前模型 Parameter 都参与训练。

第 210 行打印参数量。

### 9.6 第 212–219 行：优化器

构造自定义 AdamW：

```python
optimizer = AdamW(
    body.parameters(),
    args.alpha_max,
    tuple(args.betas),
    args.eps,
    args.weight_decay,
)
```

主循环会每步覆盖 param group 的 `lr`，因此这里的 `alpha_max` 是初始占位值。

这也说明 `args.lr` 没有任何消费者，应该删除，避免用户传 `--lr` 后以为训练会变化。

### 9.7 第 221–228 行：恢复 checkpoint

第 225 行默认从 0 开始。

第 226–228 行在指定 checkpoint 时加载：

- 模型 state_dict；
- 优化器 state_dict；
- 下一步 iteration。

保存时传 `step + 1`，恢复后直接 `range(start_step, total_steps)`，off-by-one 设计正确。

但当前 checkpoint 不包含：

- NumPy RNG state；
- Torch RNG state；
- CUDA/MPS RNG state；
- args/config；
- 已累计训练耗时。

因此恢复后模型和优化器连续，但训练 batch 序列从 seed 起点重新开始，不能与不中断训练得到完全相同的曲线。

`tool.load_checkpoint()` 还缺少 `map_location`。CUDA 保存的 checkpoint 在 CPU/MPS 机器上恢复可能失败。

还应检查：

```python
0 <= start_step < args.total_steps
```

否则循环可能一次都不执行，`loss_value` 保持 `nan`，最后仍保存一个语义错误的 final checkpoint。

### 9.8 第 230–236 行：固定验证 batch 和 overfit batch

第 231–232 行使用 `seed + 1` 采固定验证 batch。

第 233 行默认没有固定训练 batch。

第 234–236 行启用 `--overfit` 时只采一次训练 batch，之后每步重复使用它。若 loss 不能下降到接近 0，应优先怀疑模型、损失、梯度、优化器或训练循环实现。

当前梯度裁剪 bug 会污染这个 sanity check，所以必须先修 clipping，再用 overfit 判断其余链路。

### 9.9 第 238–242 行：循环计时和状态

第 239 行：

```python
t0 = time.time()
```

记录本次进程训练阶段开始时间。用于计算持续时间时，`time.perf_counter()` 更适合，因为它是单调时钟，不受系统时间调整影响。

第 240 行计算每步处理的 input token 数量：

```python
batch_size * context_length
```

第 241 行把最终训练 loss 初始化为 NaN，供循环结束后打印。

### 9.10 第 243–249 行：学习率

循环范围：

```python
range(start_step, total_steps)
```

每个 `step` 表示即将执行的更新编号。

第 246 行调用 cosine schedule。

第 247–248 行把学习率写入每个 optimizer param group。自定义 AdamW 的 `step()` 正是从 `group["lr"]` 读取，所以调度确实生效。

默认 `T_c=total_steps`，但实际最后一个 step 是 `total_steps-1`，因此最后一次更新的 lr 会略高于 `alpha_min`。这是常见的端点约定差异，不是严重错误；如果要求最后一步精确等于最小值，需要统一 schedule 对 step 的定义。

### 9.11 第 250–255 行：采训练 batch

overfit 模式直接复用固定 batch；普通模式调用 `data_loader()`：

```python
inputs, targets = data_loader(
    train_data,
    batch_size,
    context_length,
    device,
)
```

`data_loader()` 从一维 memmap 随机选择 B 个起点，构造 inputs 和右移一位的 targets，再复制到目标设备。

### 9.12 第 256–259 行：前向与 loss

第 257 行：

```python
logits = body(inputs)
```

输出形状为：

```text
(batch_size, context_length, vocab_size)
```

第 258 行使用自定义 cross entropy。它支持前置任意 batch 维，因此不需要手动 flatten。

### 9.13 第 260–264 行：清梯度与反向

第 262 行：

```python
optimizer.zero_grad(set_to_none=True)
```

清除上一轮梯度。`set_to_none=True` 通常比写入全 0 更省一次内存操作。

第 263 行：

```python
loss.backward()
```

沿计算图计算每个 Parameter 的梯度。

这两行放在 forward 后、clipping 和 optimizer step 前，顺序正确。

### 9.14 第 265–269 行：梯度范数与裁剪

第 266 行：

```python
need_log = (
    step % args.log_interval == 0
    or step == args.total_steps - 1
)
```

决定本步是否记录训练日志。若 `log_interval=0`，这里会触发 `ZeroDivisionError`，所以必须在进入循环前校验大于 0。

第 267 行只在日志步额外计算一次裁剪前全局范数。

第 268 行每步调用 `gradient_clipping()`。

这里有两层问题：

1. `gradient_clipping()` 本身存在 P0 正确性 bug，见第 12.1 节。
2. 日志步会计算两次相同全局范数：`grad_global_norm()` 一次，`gradient_clipping()` 内部又一次。

修复底层函数后，让它返回“原始全局范数”，main 可以直接写：

```python
pre_clip_norm = gradient_clipping(
    body.parameters(),
    args.grad_clip,
)
```

然后只在 `need_log` 时输出该值，无需第二遍遍历参数。

### 9.15 第 270–276 行：参数更新和初始 sanity check

第 271 行执行 AdamW 更新。

第 273 行把本步 forward 得到的 loss 转成 Python float。这个 loss 对应更新前的参数，这是训练日志的通常定义。

第 274–276 行只在非恢复训练的第一步打印：

```text
初始 loss 与 ln(vocab_size) 的对照
```

虽然打印发生在 `optimizer.step()` 之后，但 `loss_value` 来自更新前 forward，因此仍然是初始模型 loss。

### 9.16 第 278–285 行：训练日志

第 280 行计算从本进程主循环开始后的总耗时。

第 281 行计算本次进程已经完成多少步。恢复训练时从 1 重新计数，而不是从整个实验累计步数计数。

第 282–283 行打印 step、loss、lr、梯度范数、时间和平均 token/s。

吞吐量分母包含验证和 checkpoint 时间，所以表示端到端平均吞吐量，不是纯训练 step 吞吐量。

第 284–285 行把训练指标追加到 JSONL。

恢复训练时 `elapsed` 会重新从 0 开始，但 metrics 文件会继续追加，导致同一 JSONL 中 elapsed 不再单调。可以增加 `session_id`，或在 checkpoint 中保存累计 elapsed。

### 9.17 第 287–290 行：周期验证

条件含义：

```python
eval_interval > 0
并且
当前完成步数命中 eval_interval，或者已经是最后一步
```

因此只要 `eval_interval > 0`，最后一步一定会验证。

第 288 行调用固定 batch evaluate，第 289 行打印，第 290 行记录 JSONL。

### 9.18 第 292–297 行：checkpoint

第 292–294 行按 `checkpoint_interval` 覆盖保存 `ckpt_last.pt`，用于崩溃恢复。

第 295–297 行的问题是 milestone 条件嵌套在普通 checkpoint 条件内部：

```python
if 命中 checkpoint_interval:
    保存 last
    if 命中 milestone_interval:
        保存 milestone
```

假设：

```text
checkpoint_interval = 500
milestone_interval = 750
```

帮助文本说每 750 步保存 milestone，但实际只有同时被 500 和 750 整除时才保存，即每 1500 步一次。

两个条件应该彼此独立：

```python
if checkpoint_interval > 0 and ...:
    save last

if milestone_interval > 0 and ...:
    save milestone
```

### 9.19 第 299–306 行：收尾

第 300 行无条件保存 `ckpt_final.pt`。

第 301 行再次验证。

但如果 `eval_interval > 0`，最后一个循环 step 已经因为：

```python
step + 1 == total_steps
```

验证过相同模型和相同固定 batch。模型在循环结束到这里之间没有更新，因此两个 loss 完全相同。可以保存最后一次 `val_loss` 并复用，避免重复 forward。

如果 `eval_interval <= 0`，循环中没有验证，收尾时再验证一次是合理的。

第 302 行计算本次进程训练耗时。

第 303–304 行打印完成步数、最终 train loss 和 final val loss。

第 305 行追加 final 指标。它使用 `step=total_steps`，而最后一次训练/验证日志通常是 `total_steps-1`，需要在分析脚本中理解这套约定。

第 306 行返回成功退出码 0。

## 10. `__main__` 守卫

第 313 行：

```python
if __name__ == "__main__":
```

只有直接运行文件时才执行下面两行。import `train` 时不会自动启动训练。

第 314 行解析命令行。

第 315 行调用 main，并把返回值交给 `sys.exit()`。

这个层次设计正确，使 `parse_args()` 和 `main()` 可以被测试或其他脚本复用。

## 11. 启动方式与本地导入错误

当前导入写法：

```python
from tool import data_loader, save_checkpoint, load_checkpoint
from transformer import transformer_lm
from optimizer import ...
```

直接运行脚本时可用：

```bash
python cs336_basics/train.py --help
```

但按 Python 包运行时失败：

```bash
python -m cs336_basics.train --help
```

实测错误：

```text
ModuleNotFoundError: No module named 'tool'
```

如果项目希望支持包模式，应使用相对导入：

```python
from .tool import data_loader, load_checkpoint, save_checkpoint
from .transformer import transformer_lm
from .optimizer import ...
```

然后统一使用：

```bash
python -m cs336_basics.train ...
```

不要同时长期维护两套启动方式；选择包模式通常更稳妥。

## 12. 必须优先修复的错误

### 12.1 P0：梯度裁剪实现会放大梯度

`optimizer.py` 当前核心逻辑：

```python
norm = grad_global_norm(p)
if norm > M:
    with torch.no_grad():
        for parameter in p:
            if parameter.grad is None:
                continue
            norm = M / (norm + eps)
            parameter.grad *= norm
```

问题是循环内部把 `norm` 从“原始全局范数”覆盖成“缩放比例”。下一个参数又拿这个缩放比例计算新的比例。

假设原始全局范数为 `17.32`，上限为 `1`：

```text
第一个参数：scale = 1 / 17.32 ≈ 0.0577，正确缩小
第二个参数：scale = 1 / 0.0577 ≈ 17.32，错误放大
第三个参数：scale = 1 / 17.32 ≈ 0.0577，再次缩小
```

正确思路是只计算一次 scale：

```python
norm = grad_global_norm(parameters)
if norm > max_norm:
    scale = max_norm / (norm + eps)
    for parameter in parameters:
        if parameter.grad is not None:
            parameter.grad.mul_(scale)
return norm
```

这项修复同时允许 `train.py` 删除日志步额外的 `grad_global_norm()` 调用。

### 12.2 P1：当前还缺 validation token 文件

当前 `jsons` 中只有训练语料生成的：

```text
tinystories_tokens.bin
```

`train.py` 要求独立的 `--val_data`。正确流程是：

```text
在 train 文本上训练 tokenizer
→ 保存 tokenizer JSON
→ 用该 tokenizer 编码 train 文本
→ 用同一个 tokenizer 编码 validation 文本
→ 得到两个 .bin
```

不能在 validation 文本上重新训练另一套 tokenizer，因为 train 和 val 的 ID 语义必须一致。

### 12.3 P1：新运行污染旧 metrics

新实验使用旧 out_dir 时，config 被覆盖而 metrics 被追加。训练开始前必须区分“新运行”和“resume”。

### 12.4 P1：checkpoint 恢复不完整

要严格恢复，需要保存并恢复：

- Python/NumPy/Torch RNG；
- GPU RNG（如适用）；
- 模型和优化器状态；
- iteration；
- 与模型结构相关的配置；
- 可选累计训练时间。

另外 checkpoint 建议先写临时文件，再使用 `os.replace()` 原子替换 `ckpt_last.pt`，避免进程在写到一半时留下损坏文件。

## 13. 可以简化或去重的地方

### 13.1 删除未使用 import

静态检查发现 13 个未使用导入。删除后文件开头会明显清爽。

### 13.2 删除或重新定义 `--lr`

`args.lr` 没有被读取。最简单是删除；若希望保留常见参数名，可以把 `--lr` 直接作为 `alpha_max` 的命令行名称，不要同时存在两个“最大学习率”。

### 13.3 梯度范数只计算一次

修复 `gradient_clipping()` 让它返回原始 norm，删除：

```python
pre_clip_norm = grad_global_norm(...)
```

### 13.4 复用最后一次 validation loss

最后一步周期验证与循环后的 final evaluate 重复。维护一个：

```python
last_val_loss: float | None
```

最后已经验证过就直接复用。

### 13.5 独立 milestone 条件

把 milestone checkpoint 从普通 checkpoint if 中移出。代码更短，语义也与参数帮助一致。

### 13.6 使用独立 RNG

让 `data_loader()` 接收 `np.random.Generator` 后：

- `make_fixed_batches()` 不再需要保存/恢复全局 state；
- checkpoint 可以明确保存 train RNG state；
- 验证参数不会影响训练采样。

### 13.7 不必继续过度拆分 `main()`

当前 main 已按配置、准备、循环、收尾分区。可以抽出 `validate_args()` 和 checkpoint/RNG 辅助函数，但不建议把每三行都拆成函数，否则控制流会更难追踪。

适合新增的有限几个辅助函数：

```text
validate_args(args)
build_model(args, device)
save_training_state(...)
load_training_state(...)
```

训练 step 本身最好仍连续保留在主循环中。

## 14. 性能优化建议

在正确性修复之后，再考虑性能。

### 14.1 低风险优化

- `evaluate()` 使用 `torch.inference_mode()`；
- 使用 `time.perf_counter()` 统计耗时；
- 梯度范数避免重复计算；
- 验证 loss 避免末尾重复计算；
- 在 CUDA 上考虑 autocast 混合精度；MPS 需先验证算子和数值稳定性。

### 14.2 固定验证 batch 的设备内存

默认只占约 1.25 MiB，不需要急着改。配置放大后，可以只保存 NumPy 起点索引，验证时动态构造 tensor。

### 14.3 自定义 AdamW 的分配

`optimizer.py` 中：

```python
m = beta_1 * m + (1-beta_1) * grad
v = beta_2 * v + (1-beta_2) * grad ** 2
```

每步会创建新 tensor。可改为原地更新以减少峰值内存和分配，但必须保持作业测试通过。这个优化优先级低于梯度裁剪正确性。

### 14.4 `torch.compile` 和 fused optimizer

它们可能提高训练速度，但当前作业使用自定义模块与自定义 AdamW，兼容性和调试成本较高。应在基线正确、overfit 通过、checkpoint 可恢复之后再尝试。

## 15. 建议的修复顺序

1. 修复 `gradient_clipping()`，运行 `test_gradient_clipping`。
2. 用现有 tokenizer 编码 validation，生成独立 val `.bin`。
3. 增加参数校验，尤其是 `log_interval`、模型维度和优化器参数。
4. 修复新运行/恢复运行的 metrics 生命周期。
5. 把 milestone checkpoint 条件移出普通 checkpoint 条件。
6. checkpoint 保存/恢复 RNG state 和配置，并支持 `map_location`。
7. 修复包相对导入，统一启动方式。
8. 删除未使用 import 和无效 `--lr`。
9. 去掉重复梯度范数与重复 final evaluate。
10. 先跑 `--overfit` 小实验，再开始完整 TinyStories 训练。

## 16. 最终训练前检查清单

- [ ] `test_gradient_clipping` 通过；
- [ ] train 和 val `.bin` 都存在，并使用同一 tokenizer；
- [ ] 两个 `.bin` 的 dtype 都与 `--data_dtype` 一致；
- [ ] train/val token ID 均小于 vocab size；
- [ ] `--overfit` 能把单 batch loss 压到接近 0；
- [ ] 初始 loss 大致接近 `ln(10000) ≈ 9.21`，允许随机 logits 方差使其略高；
- [ ] 新实验使用新的 out_dir，或明确清理旧 metrics；
- [ ] checkpoint 可以在目标设备上加载；
- [ ] resume 的模型配置与原配置一致；
- [ ] 磁盘空间足够保存模型、AdamW 状态和多个 checkpoint；
- [ ] 先进行几十步 smoke test，再启动长时间训练。

## 17. 一句话评价

`train.py` 的训练骨架和分层思路是好的，主要冗余集中在 import、无效 `--lr`、重复梯度范数和重复最终验证；真正阻止正式训练的不是这些冗余，而是底层梯度裁剪错误、缺少独立 validation token 文件，以及不完整的恢复/日志语义。应先修正确性，再做简化和性能优化。
