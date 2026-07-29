# Assignment 1 — Section 5「Training loop」任务清单

> 整理自 `cs336_assignment1_basics.pdf` 第 34–36 页（Section 5 全部内容，到 Section 6 之前）。
> 承接 [Training任务清单.md](Training任务清单.md)（Section 4）。
> 本文只梳理**要做什么、怎么验收**，不含任何实现。

## 0. 本节定位

Section 2 有了 tokenizer，Section 3 有了模型，Section 4 有了 loss 和优化器。Section 5 把它们**拼成一个能真正跑起来的训练脚本**：

- **喂数据** —— 从一条超长 token 序列里采样 batch
- **存/取进度** —— checkpoint，让训练中断后能接着跑
- **主循环** —— 把所有零件串起来的脚本

做完这一节，你就可以开始真正训练模型了（Section 6 起全是实验）。

---

## 1. 任务总览（共 3 个 Problem，7 分）

| # | Problem | 分值 | 类型 | Adapter | 测试命令 |
|---|---|---|---|---|---|
| 1 | `data_loading` | 2 | 编程 | `run_get_batch` | `uv run pytest -k test_get_batch` |
| 2 | `checkpointing` | 1 | 编程 | `run_save_checkpoint`<br>`run_load_checkpoint` | `uv run pytest -k test_checkpointing` |
| 3 | `training_together` | 4 | **脚本（无自动测试）** | — | — |

**注意分值分布**：`training_together` 一个人占了 4/7 分，但**没有任何自动测试**。它的验收方式是 Section 6/7 的实验——脚本写得好不好，直接决定后面几十个小时的实验体验。

---

## 2. 逐项拆解

### ☐ 任务 1：`data_loading`（2 分，§5.1，p.34）

#### 数据长什么样

所有文档（网页、源码文件…）被**拼接成一条超长的 token 序列** $x = (x_1, \dots, x_n)$，文档之间插入 `<|endoftext|>` 作为分隔符。

这是 LM 预训练的标准做法。handout 列出了三条好处：

1. **采样平凡**——任意合法起点 $i$ 都能切出一条训练样本
2. **无需 padding**——所有样本等长，硬件利用率高，也更容易增大 batch size
3. **无需全量载入内存**——配合 mmap 就能处理超过内存的数据集

#### 要交什么

| 项 | 内容 |
|---|---|
| **输入** | `dataset`（1D numpy 整数数组）、`batch_size`、`context_length`、`device`（如 `'cpu'` / `'cuda:0'` / `'mps'`） |
| **输出** | 一个 tuple：`(inputs, targets)` |
| **形状** | 两个都是 `(batch_size, context_length)` |
| **类型** | `torch.LongTensor`（token ID） |
| **位置** | 两个都要在 `device` 上 |

#### 采样语义

handout 的例子（$B=1, m=3$）：

$$([x_2, x_3, x_4],\ [x_3, x_4, x_5])$$

也就是：**target 是 input 右移一位**。选定起点 $i$ 后，input 取 $m$ 个 token，target 取紧随其后错开一位的 $m$ 个 token。

#### ⚠️ 合法起点的范围（最容易 off-by-one 的地方）

handout 说：

> any $1 \le i \le n - m$ gives a valid training sequence

这是 **1-indexed**。转成 Python 的 0-index，测试里写得更直白（[test_data.py:36-39](../tests/test_data.py#L36-L39)）：

```
num_possible_starting_indices = len(dataset) - context_length
assert max(starting_indices) == num_possible_starting_indices - 1
assert min(starting_indices) == 0
```

以测试的参数（$n=100$, $m=7$）为例：合法起点是 **0 到 92**，共 **93** 个。

**自己推一遍**：target 的最后一个元素下标是多少？它必须 $\le n-1$。由此反推 $i$ 的上界。**这个边界必须精确**——测试会断言 `max` 和 `min` 恰好等于这两个值，多一个少一个都过不了。

#### 测试还查了什么（[test_data.py](../tests/test_data.py)）

跑 1000 轮 × batch 32，共 32000 个样本，断言：

1. **形状**：两个张量都是 `(32, 7)`
2. **偏移**：`y == x + 1`（因为测试数据是 `np.arange(0,100)`，相邻 token 恰好差 1）
3. **起点覆盖**：最小值是 0，最大值是 92
4. **分布均匀**：每个起点出现的次数必须落在 $\mu \pm 5\sigma$ 内

第 4 条是硬性要求：**必须是均匀随机采样**。顺序遍历、固定 stride、或者任何有偏的采样方式都会挂。

#### mmap（handout 用加粗强调的部分）

> **be sure to load the dataset in memory-mapped mode** (via `np.memmap` or the flag `mmap_mode='r'` to `np.load`)

原理：`mmap` 是一个 Unix 系统调用，把磁盘文件映射到虚拟内存，**访问到哪一段才真正读哪一段**。于是你可以「假装」整个数据集都在内存里。

handout 明确提醒的两点：

- **dtype 必须和保存时匹配**——不匹配的话读出来的是一堆乱码，而且**不报错**
- **建议显式验证数据看起来正常**，例如检查有没有超出 vocab size 的值

> 这又是一个「静默失败」：dtype 弄错（比如 `uint16` 读成 `int32`）不会有任何异常，只是你的 token ID 全是垃圾，模型永远学不会东西。花两分钟做一次范围检查，能省下几小时的困惑。

#### 💡 Low-Resource Tip（handout 专门给的提示框）

在 CPU 或 Apple Silicon 上训练时：

- CPU → device 字符串 `'cpu'`
- Apple Silicon（M 系列芯片）→ `'mps'`
- **数据和模型必须在同一个 device 上**

参考资料：
- https://docs.pytorch.org/docs/stable/mps.html
- https://docs.pytorch.org/docs/stable/notes/mps.html

#### 需要注意的坑

- **dtype 转换**：numpy 数组常存成 `uint16`（省空间），但 PyTorch 这边做 embedding 索引和 cross-entropy target 都需要 `int64`
- **两个张量都要搬 device**，别只搬了一个
- **起点上界**：见上文，测试会精确断言
- **memmap 切片**：从 memmap 取出来的是 array-like，转成 tensor 时注意是否需要先复制成真正的 ndarray

---

### ☐ 任务 2：`checkpointing`（1 分，§5.2，p.35–36）

#### 为什么需要

handout 给了两个理由：

1. **训练会中断**——作业超时、机器故障。没有 checkpoint 就得从头再来
2. **想研究训练动态**——事后回看不同阶段的模型、从各阶段采样

#### 一个 checkpoint 必须包含三样东西

| 要存什么 | 为什么 |
|---|---|
| **模型权重** | 显然 |
| **优化器状态** | AdamW 是**有状态**的：$m$、$v$。不存的话恢复训练相当于重新 warmup，loss 曲线会有一个明显的跳变 |
| **迭代次数** | 用来**恢复学习率调度**——不知道走到第几步，cosine schedule 就接不上 |

> 这一条和我们讨论过的**契约 2** 直接相关：优化器状态必须放在 `self.state` 里，`state_dict()` 才能把它序列化出来。如果你把 $m$、$v$ 存成了实例属性，这里存出来的 checkpoint 会**缺失动量**——而且存和读都不报错，只有训练曲线会告诉你有问题。

#### 两个函数的签名（handout 明确指定）

**`save_checkpoint(model, optimizer, iteration, out)`**

| 参数 | 类型 |
|---|---|
| `model` | `torch.nn.Module` |
| `optimizer` | `torch.optim.Optimizer` |
| `iteration` | `int` |
| `out` | `str \| os.PathLike \| BinaryIO \| IO[bytes]` |

**`load_checkpoint(src, model, optimizer) -> int`**

| 参数 | 类型 |
|---|---|
| `src` | `str \| os.PathLike \| BinaryIO \| IO[bytes]` |
| `model` | `torch.nn.Module` |
| `optimizer` | `torch.optim.Optimizer` |
| **返回** | `int`——保存时的迭代次数 |

#### 用到的 PyTorch API

handout 直接点名了这几个：

| API | 作用 |
|---|---|
| `module.state_dict()` | 拿到所有可学习权重 |
| `module.load_state_dict()` | 恢复权重（**原地**） |
| `optimizer.state_dict()` / `load_state_dict()` | 同上，用于优化器 |
| `torch.save(obj, dest)` | 把对象（含张量的 dict、普通 Python 对象如 int）写到路径或 file-like object |
| `torch.load(src)` | 读回来 |

handout 说 `obj` 用 dict 是典型选择，但**格式随你**——只要自己读得回来。

#### 测试断言了什么（[test_serialization.py:57-120](../tests/test_serialization.py#L57)）

流程：建模型 → 跑 10 步 → 存 → **新建一个模型和优化器** → 读 → 比较。

1. `assert it == loaded_iterations`——返回值必须对
2. 模型 `state_dict` 的 **key 集合**一致
3. 每个权重张量数值 `allclose`
4. 优化器 `state_dict` 相等（含 $m$、$v$、$t$）

**关键点**：`load_checkpoint` 必须**原地恢复**传进来的 `model` 和 `optimizer`，不是返回新对象。测试拿的是它自己创建的那两个对象去比对。

#### 需要注意的坑

- **参数顺序必须一致**。`optimizer.state_dict()` 里的 key 是**整数索引**（按 `param_groups` 的顺序编号），不是 Parameter 对象。所以恢复时模型结构必须完全一致，否则状态会张冠李戴——**而且通常不报错**
- **`torch.load` 的 `weights_only`**：新版本 PyTorch 默认为 `True`。存纯粹的 state_dict + int 没问题，但如果你往 checkpoint 里塞了自定义对象，读的时候会报错
- **file-like object 也要支持**：测试用的是路径，但签名声明支持两者，别写死成只接受路径

---

### ☐ 任务 3：`training_together`（4 分，§5.3，p.36）

#### 要交什么

> Write a script that runs a training loop to train your model on user-provided input.

handout **推荐**（recommend，不是强制）至少支持四件事：

1. **可配置模型和优化器的各种超参数**——建议做成命令行参数
2. **用 `np.memmap` 内存高效地加载训练集和验证集**
3. **把 checkpoint 序列化到用户指定的路径**
4. **定期记录训练和验证性能**——输出到 console，和/或外部服务如 Weights & Biases

#### handout 特意强调的一句

> It will pay off to make it easy to start training runs with different hyperparameters, since **you will be doing these many times later**

这不是客套话。往后的实验包括：学习率扫描（2 B200 小时）、batch size 变化、4 个消融实验、OWT 主实验、leaderboard（10 B200 小时）。**你会反复启动几十次训练。** 现在多花一小时把命令行接口和日志做好，后面能省下十几小时。

#### 要组装的零件清单

到这里为止你写过的所有东西，都要在这个脚本里汇合：

| 来源 | 零件 |
|---|---|
| §2 | Tokenizer（离线把语料转成 token 数组） |
| §3 | `TransformerLM` |
| §4.1 | `cross_entropy` |
| §4.3 | `AdamW` |
| §4.4 | cosine 学习率调度（含 warmup） |
| §4.5 | 梯度裁剪 |
| §5.1 | `get_batch` |
| §5.2 | `save_checkpoint` / `load_checkpoint` |

#### 主循环的顺序（把各节的结论汇总）

```
每一步：
  1. 用当前 iteration 算出学习率，写进 optimizer 的 param_groups   ← §4.4
  2. 采一个 batch                                                 ← §5.1
  3. 前向 → cross_entropy 得到标量 loss                            ← §4.1
  4. zero_grad → backward                                         ← §4.2
  5. 梯度裁剪                                                     ← §4.5
  6. optimizer.step()                                             ← §4.3
  7. 定期：算验证 loss（记得 no_grad）、打印日志、存 checkpoint     ← §5.2
```

第 1 步和第 5 步的位置尤其容易放错。第 5 步必须在 backward 之后、step 之前；第 1 步写的是 `group["lr"]`——这就是为什么我们说 `step` 里必须从 `group[...]` 读超参数而不是 `self.lr`。

#### 建议自己加的东西（handout 没要求但会很有用）

- **训练 loss 和验证 loss 分开记录**，验证 loss 用固定的 batch 或固定 seed，才有可比性
- **记录 wall-clock 时间**——后面的实验按 B200 小时计费
- **梯度范数也记一下**——训练不稳定时，梯度范数的曲线比 loss 曲线更早给出预警
- **checkpoint 定期覆盖 + 关键节点另存**，避免磁盘被塞满

---

## 3. 依赖关系

```
data_loading  ──┐
                ├──→ training_together ──→ Section 6/7 的所有实验
checkpointing ──┘
        ↑
   依赖 AdamW 的 state 契约（§4.3）
```

前两个任务互相独立，可以并行做。`training_together` 依赖前两个 **加上** Section 3、4 的全部产出。

---

## 4. 验收清单

- [ ] `uv run pytest -k test_get_batch` 通过
- [ ] `uv run pytest -k test_checkpointing` 通过
- [ ] 训练脚本能从命令行接收超参数
- [ ] 训练脚本用 memmap 加载数据，且验证过 token 值域正常
- [ ] 能存 checkpoint，杀掉进程后能接着跑，**且 loss 曲线在接续处平滑**（这一条是检验优化器状态有没有真的恢复的黄金标准）
- [ ] 定期输出 train / val loss

最后一条的检验方法值得单独说：**跑 20 步不中断** vs **跑 10 步 → 存 → 重启 → 再跑 10 步**，两条 loss 曲线应当完全重合。对不上就说明有状态没存全。

---

## 5. 相关笔记

- [Training任务清单.md](Training任务清单.md) —— Section 4
- [4.2_SGD优化器与Optimizer_API详解.md](4.2_SGD优化器与Optimizer_API详解.md) —— `state_dict` 的两条数据契约
- [gather与高级索引详解.md](gather与高级索引详解.md)
- [如何构造Iterable和Iterator.md](如何构造Iterable和Iterator.md)
