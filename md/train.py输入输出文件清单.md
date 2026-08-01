# train.py 输入输出文件清单

跑一次 `python cs336_basics/train.py ...`，磁盘上需要哪些文件才能开始、会产生哪些文件、每个文件具体怎么被读/写——按"输入"和"输出"两大类整理，串联之前几篇笔记（[memmap内存映射详解.md](memmap内存映射详解.md)、[train.py函数与变量作用整理.md](train.py函数与变量作用整理.md)）里已经讲过的细节。

## 一、需要准备好的输入文件

### 1. 训练集 / 验证集 token 数组（必需，`--train_data` / `--val_data`）

**是什么**：tokenizer 处理完文本之后，产出的一份纯 token id 序列，存成磁盘文件。格式可以是两种之一：

| 格式 | 后缀 | 是否自描述 | 打开方式 |
|---|---|---|---|
| NumPy 原生格式 | `.npy` | 是（自带 dtype/shape 头部） | `np.load(path, mmap_mode="r")` |
| 裸二进制 | 任意（脚本里判断非 `.npy` 都走这条） | 否（必须手动告知 dtype） | `np.memmap(path, dtype=..., mode="r")` |

**读取代码位置**：`load_tokens`（`train.py:44-73`），两个分支的判断依据是文件名后缀：

```python
if path.endswith(".npy"):
    data = np.load(path, mmap_mode="r")
else:
    data = np.memmap(path, dtype=np.dtype(dtype), mode="r")
```

**读取方式的关键点**：两种方式都是**内存映射**（memory-mapped），不会把整个文件一次性读进内存——数据留在磁盘，真正被访问到的那一小段才会被操作系统按需加载进物理内存。这是为了让上百 GB 的语料也能被"打开"而不撑爆内存。详见 [memmap内存映射详解.md](memmap内存映射详解.md)。

**读取前的自检**（同样在 `load_tokens` 里）：
- 长度检查：`len(data) >= context_length + 2`，否则整份数据连一个训练窗口都塞不下。
- 值域检查：抽样前 1000 万个 token，检查 `min/max` 是否落在 `[0, vocab_size)` 之间——这是抓"`--data_dtype` 传错"的最便宜手段，dtype 读错不会自然报错，只会把字节流解释成垃圾数字。

**准备时要注意**：如果传的是裸二进制文件（非 `.npy`），必须通过 `--data_dtype`（默认 `uint16`）告诉脚本这份数据是用什么 dtype 存的——这个值必须和 tokenizer 存文件时用的 dtype **完全一致**，否则会读出乱码但不一定报错（除非值域自检抓到了）。

### 2.（可选）续训的 checkpoint 文件（`--resume_from`）

**是什么**：之前某次训练存下来的 `.pt` 文件（`ckpt_last.pt` / `ckpt_final.pt` / `ckpt_{step}.pt` 之一），内含模型权重、优化器状态、训练步数三样东西的打包。

**读取代码位置**：`load_checkpoint`（`tool.py:41-49`）：

```python
obj = torch.load(src)
model.load_state_dict(obj['model'])
optimizer.load_state_dict(obj['opt'])
return obj['iteration']
```

**读取方式的关键点**：
- `torch.load` 把整个 `.pt` 文件反序列化成一个 Python 字典。
- `model.load_state_dict(...)` / `optimizer.load_state_dict(...)` 是**原地写入**——把恢复出来的数值填回调用方已经构造好的 `body`、`optimizer` 对象里，函数本身不返回模型/优化器。
- 只有 `iteration`（一个 int）作为返回值传出来，因为整数不可变，没法"原地修改"。

**准备时要注意**：续训时命令行传入的模型结构参数（`d_model`/`num_heads`/`context_length`/`rope_theta` 等）必须和保存这份 checkpoint 时**完全一致**——因为 `state_dict` 里只存了权重数值，不存模型结构；而且 RoPE 的 cos/sin 表是 `persistent=False` 的 buffer，根本不会被存进 checkpoint，配置不一致时这张表会被静默地按新配置重新算出来，`load_state_dict` 察觉不到这种不一致。

## 二、命令行本身（不是文件，但决定了一切）

所有其他配置（模型结构、优化器超参、训练步数、各种 interval）都通过命令行参数传入，由 `parse_args`（`train.py:121-168`）解析成 `argparse.Namespace` 对象。这一步不涉及文件读写，但决定了后面输出文件里 `config.json` 的具体内容。

## 三、训练过程中会产生的输出文件

全部输出都落在 `--out_dir` 指定的目录下（该目录由 `os.makedirs(args.out_dir, exist_ok=True)` 自动创建，不存在也不会报错）。

### 1. `config.json`（训练开始时写一次）

**内容**：这次训练用到的**全部**命令行参数，来自 `vars(args)` 把 `Namespace` 转成字典。

**写入代码位置**：`main` 区块 1（`train.py:186-189`）：

```python
config = vars(args)
with open(os.path.join(args.out_dir, "config.json"), "w") as f:
    json.dump(config, f, indent=2, ensure_ascii=False, default=str)
```

**写入方式的关键点**：
- `"w"` 覆盖模式——每次训练开始都会重写这份文件，不会累积历史。
- `indent=2` 让文件保持人类可读的缩进格式（这是"给人回头查阅"的文件，不是高频写入的日志，所以格式化开销无所谓）。
- `default=str`：兜底处理 `json` 不认识的类型（比如某个参数如果被解析成了非基础类型），遇到就转成字符串再序列化，避免直接抛异常中断训练。

**怎么读回来**：普通的 `json.load(open(path))` 即可，得到一个字典。

### 2. `metrics.jsonl`（训练全程持续追加写入）

**内容**：每一条训练指标记录——训练 loss/学习率/梯度范数（每 `log_interval` 步一条）、验证 loss（每 `eval_interval` 步一条）、训练结束时的最终验证结果（一条）。同一个文件里混着不同"种类"的记录，靠字段名区分（比如 `train_loss` vs `val_loss` vs `final_val_loss`）。

**写入代码位置**：`log_jsonl`（`train.py:111-114`），在主循环里被调用三处（第 284、290 行）和收尾时一处（第 305 行）：

```python
def log_jsonl(path: str, record: dict) -> None:
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")
```

**写入方式的关键点**：
- `"a"` 追加模式——每次调用只在文件末尾加一行，不需要读出旧内容再整体重写。
- 采用 **JSONL**（JSON Lines）格式：文件不是一个整体的 JSON 数组，而是"每一行都是一个独立、完整的 JSON 对象"。好处是即使训练中途崩溃，已经写完的每一行仍然是完整可解析的 JSON,不会像"一个大 JSON 数组"那样因为一次意外中断就损坏整个文件的语法结构。

**怎么读回来**：不能直接 `json.load(f)`（整个文件不是一个合法 JSON 值），必须逐行读、逐行解析：

```python
import json
records = []
with open("metrics.jsonl") as f:
    for line in f:
        records.append(json.loads(line))
```

如果要用 pandas 分析画图，`pandas.read_json(path, lines=True)` 可以直接读取这种格式，一步到位得到 DataFrame。

### 3. Checkpoint 文件（`.pt`，按 interval 周期性产生 + 训练结束固定产生一份）

| 文件名 | 何时产生 | 会不会被覆盖 | 用途 |
|---|---|---|---|
| `ckpt_last.pt` | 每 `checkpoint_interval` 步 | 会（每次都覆盖同名文件） | 崩溃恢复；`--resume_from` 通常指向它 |
| `ckpt_{step}.pt` | `milestone_interval > 0` 且该步同时命中 `checkpoint_interval` 和 `milestone_interval` 时 | 不会（文件名带步数，各自独立） | 事后分析、对比不同阶段的模型 |
| `ckpt_final.pt` | 训练循环彻底结束后，无条件产生一次 | 不会 | 标志"这次训练完整跑完"；作为最终交付模型或后续继续训练的起点 |

**写入代码位置**：都调用同一个 `save_checkpoint`（`tool.py:30-39`）：

```python
def save_checkpoint(model, optimizer, iteration, out):
    state_model = model.state_dict()
    state_opt = optimizer.state_dict()
    obj = {'model': state_model, 'opt': state_opt, 'iteration': iteration}
    torch.save(obj, out)
```

**写入方式的关键点**：
- 把模型的 `state_dict()`（各层权重）、优化器的 `state_dict()`（AdamW 的一阶/二阶动量估计等）、当前步数三样打包进一个字典，用 `torch.save` 整体序列化写盘。
- **为什么优化器状态也必须存**：如果只存模型权重，续训时优化器的动量状态会归零重来，即使权重完全一样，续训后前几步的更新方向和步长也会和"从未中断"的情况不同，训练曲线会在续训点出现不该有的抖动。
- `iteration` 存的语义是**下一步该从哪继续**——保存时统一传 `step + 1`（不是 `step`），恢复时直接拿这个值当循环起始点用，不需要再额外 `+1`。

**怎么读回来**：调用 `load_checkpoint`（见上面"输入文件"第 2 条），或者最原始地直接 `torch.load(path)` 拿到那个字典，手动取 `obj['model']`/`obj['opt']`/`obj['iteration']`。

**跨设备加载的注意事项**：`load_checkpoint` 内部的 `torch.load(src)` 没有传 `map_location` 参数——如果 checkpoint 是在有 CUDA 的机器上存的，现在要在只有 CPU/MPS 的机器上加载，默认行为会尝试把张量放回原来的 CUDA 设备，如果当前机器没有 CUDA 会直接报错。跨设备加载需要显式 `torch.load(src, map_location=device)`。

### 4. 终端标准输出（不是文件，但同样是一种"输出"）

训练过程中会往 stdout 打印一系列带 `[tag]` 前缀的提示（`[config]`、`[device]`、`[data]`、`[model]`、`[resume]`、`[sanity]`、`[overfit]`、常规的 `step ... | loss ...` 训练/验证日志行、`[done]`），如果想保留这份记录，需要自己在运行时重定向（如 `python train.py ... > train.log 2>&1`）——脚本本身不会自动把这些内容存成文件，只有 `metrics.jsonl` 才是脚本主动落盘的结构化日志。

## 四、文件依赖关系总览

```
【输入】
  --train_data (.npy 或裸 .bin)  ──┐
  --val_data   (.npy 或裸 .bin)  ──┼── load_tokens (memmap 打开 + 自检) ──> 内存映射数组
  --data_dtype (裸文件才需要)     ──┘
  --resume_from (可选, .pt)      ──── load_checkpoint (torch.load + load_state_dict) ──> 恢复模型/优化器状态 + 起始步数

【输出，全部落在 --out_dir 下】
  config.json      ← json.dump,       训练开始时写一次（"w" 覆盖）
  metrics.jsonl    ← log_jsonl,       训练全程持续追加（"a" 追加，JSONL 格式）
  ckpt_last.pt     ← save_checkpoint, 每 checkpoint_interval 步覆盖写
  ckpt_{step}.pt   ← save_checkpoint, 命中 milestone_interval 时额外独立写（可能有多个）
  ckpt_final.pt    ← save_checkpoint, 训练结束后无条件写一次
```

## 五、跑一次完整训练前的自查清单

1. 训练集、验证集文件是否存在，格式（`.npy` 还是裸二进制）是否和 `--data_dtype` 对应一致。
2. `--out_dir` 是否有写权限（`os.makedirs` 会自动建目录，但目录所在的父路径必须可写）。
3. 如果是续训（`--resume_from`），模型结构相关的参数是否和保存这份 checkpoint 时完全一致。
4. 如果 checkpoint 是跨设备产出的（比如在云端 GPU 训练、现在想在本地 Mac 上续训/推理），需要注意 `load_checkpoint` 目前没有处理 `map_location`，直接加载可能报错。
