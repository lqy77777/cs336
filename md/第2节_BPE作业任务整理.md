# Section 2（BPE 分词器）作业任务整理

> 依据讲义《第 2 节：字节对编码（BPE）分词器》整理的任务清单与整体框架，本节共 **42 分**。
> 注意：本文件只整理"要做什么"，不包含任何实现思路之外的解法——具体实现需要自己完成。

---

## 一、本节目标总览

实现一个**字节级（byte-level）BPE 分词器**，分两大块：

1. **训练**：给定文本文件，训练出词表（vocab）和合并规则列表（merges）；
2. **使用**：加载词表和合并规则，实现文本 ↔ token ID 的编码与解码。

这个分词器之后会用于把 TinyStories / OpenWebText 数据编码成整数序列，供后续语言模型训练使用。

---

## 二、任务清单（按题目）

### 书面题（理解性问题，写一两句话即可）

| 题目 | 分值 | 交付物 |
|------|------|--------|
| **(unicode1)** 理解 Unicode | 1 分 | 3 个小问，各一句话：`chr(0)` 是什么字符；`__repr__()` 与打印表示的区别；该字符出现在文本中的现象 |
| **(unicode2)** Unicode 编码 | 3 分 | 3 个小问：为何用 UTF-8 而非 UTF-16/32；给出使 `decode_utf8_bytes_to_str_wrong` 出错的字节串示例并解释；给出一个无法解码的两字节序列并解释 |

### 编程题（核心实现，占大头）

| 题目 | 分值 | 交付物 |
|------|------|--------|
| **(train_bpe)** BPE 训练函数 | 15 分 | 一个训练函数：输入 `input_path`、`vocab_size`、`special_tokens`，输出 `vocab: dict[int, bytes]` 和 `merges: list[tuple[bytes, bytes]]`；实现适配器 `adapters.run_train_bpe`，通过 `uv run pytest tests/test_train_bpe.py` |
| **(tokenizer)** Tokenizer 类 | 15 分 | 一个 `Tokenizer` 类，含 `__init__`、`from_files`、`encode`、`encode_iterable`、`decode` 五个接口；实现适配器 `adapters.get_tokenizer`，通过 `uv run pytest tests/test_tokenizer.py` |

### 实验题（跑实验 + 简答）

| 题目 | 分值 | 内容 |
|------|------|------|
| **(train_bpe_tinystories)** | 2 分 | 在 TinyStories 上训练，vocab_size = 10,000，含 `<|endoftext|>`；序列化 vocab/merges 到磁盘；回答训练耗时/内存、最长 token 是什么；用 profiler 找出训练中最耗时的部分 |
| **(train_bpe_expts_owt)** | 2 分 | 在 OpenWebText 上训练，vocab_size = 32,000；序列化到磁盘；回答最长 token；对比两个分词器的异同 |
| **(tokenizer_experiments)** | 4 分 | (a) 各采样 10 篇文档，计算两个分词器的压缩比（字节数/token 数）；(b) 用 TinyStories 分词器编码 OWT 样本会怎样；(c) 估计吞吐量（字节/秒），推算对 825 GB 的 Pile 分词要多久；(d) 把训练集/验证集编码为 `uint16` 的 NumPy 数组并解释为何 `uint16` 合适 |

---

## 三、大致框架

### Part A：BPE 训练（对应 2.4–2.5 节）

讲义规定的训练流程分三步：

1. **词表初始化**
   - 初始词表 = 全部 256 个字节值，再加上所有特殊 token（如 `<|endoftext|>`）。
   - 最终词表大小 = 初始字节表 + 合并次数 + 特殊 token 数，不超过 `vocab_size`。

2. **预分词（Pre-tokenization）**
   - 使用 GPT-2 风格的正则 `PAT`（讲义 2.4 节给出）对文本做粗切分，得到"预 token → 出现次数"的统计。
   - 讲义要求用 `re.finditer` 而不是 `findall`（避免把所有结果存进内存）。
   - **先按特殊 token 切分文本**（`re.split` + 注意 `re.escape`），特殊 token 是硬边界：两侧的文本不能发生合并，特殊 token 本身也不参与合并统计。对应测试 `test_train_bpe_special_tokens`。
   - 性能瓶颈主要在这一步：讲义建议用 `multiprocessing` 并行，分块边界要落在 `<|endoftext|>` 的开头（可直接照搬 `cs336_basics/pretokenization_example.py` 里的分块起始代码）。

3. **迭代计算合并（Merges）**
   - 每轮：统计所有相邻字节对的频次 → 取频次最高的一对 → 合并成新 token 加入词表 → 记录进 merges 列表（按创建顺序）。
   - **不跨预 token 边界**统计/合并字节对。
   - **平局规则**：频次相同时，选**字典序更大**的那一对（对 bytes 二元组直接取 `max`）。
   - 效率提示（讲义 2.5 节）：朴素做法每轮全量重扫太慢；每次合并后只有与被合并对重叠的字节对计数会变化，可以对计数建索引做**增量更新**。合并这部分在 Python 中无法并行。
   - 建议先在小数据（如 TinyStories 验证集）上调通，再上完整数据；用 `cProfile` / `py-spy` 找瓶颈。

   可以先照 Sennrich et al. 的算法 1 写一个低效版本验证理解（讲义中的 `bpe_example` 手算示例可用来核对：语料 low/lower/widest/newest，前 6 次合并应为 `s t, e st, o w, l ow, w est, n e`）。

### Part B：Tokenizer 类（对应 2.6 节）

**接口要求**（讲义推荐的签名）：

- `__init__(self, vocab, merges, special_tokens=None)` — 用给定词表、合并列表、特殊 token 构造；特殊 token 若不在词表中要追加进去。
- `from_files(cls, vocab_filepath, merges_filepath, special_tokens=None)` — 类方法，从磁盘序列化文件（格式与你训练代码的输出一致）构造。
- `encode(self, text: str) -> list[int]` — 文本 → token ID 序列。
- `encode_iterable(self, iterable) -> Iterator[int]` — 对字符串可迭代对象（如文件句柄）惰性产出 token ID，用于内存放不下的大文件。
- `decode(self, ids: list[int]) -> str` — token ID 序列 → 文本。

**编码流程**（与训练呼应）：

1. 预分词（与训练时完全相同的正则），每个预 token 转成 UTF-8 字节序列；
2. 在每个预 token 内部，**按 merges 的创建顺序**依次应用合并（不跨预 token 边界）；
3. 查词表得到整数 ID。讲义 2.6.1 有一个 `'the cat ate'` 的完整手算示例可用来核对。

**注意点**：

- 特殊 token 在编码时必须保持为单个 token，不能被拆开。
- 大文件要分块处理（`encode_iterable`），且保证 token 不跨分块边界，结果要与一次性编码一致。
- 解码时，非法字节序列用 Unicode 替换字符 `U+FFFD` 处理（`bytes.decode` 的 `errors='replace'`）。

---

## 四、测试与验证方式

- 先在 `tests/adapters.py` 里实现两个适配器：`run_train_bpe` 和 `get_tokenizer`；
- 然后运行：
  - `uv run pytest tests/test_train_bpe.py`（训练部分，含特殊 token 测试）
  - `uv run pytest tests/test_tokenizer.py`（编解码部分）
- 两组测试都要全部通过。

## 五、资源限制与数据

| 实验 | 时间上限 | 内存上限 | 数据文件（在 `data/` 下） |
|------|---------|---------|--------------------------|
| TinyStories 训练（10K 词表） | ≤ 30 分钟（优化好可 < 2 分钟） | ≤ 30 GB | `TinyStoriesV2-GPT4-train.txt` / `-valid.txt` |
| OpenWebText 训练（32K 词表） | ≤ 12 小时 | ≤ 100 GB | `owt_train.txt` / `owt_valid.txt` |

均无需 GPU。

## 六、最终交付物清单（自查）

- [ ] unicode1 三个小问的书面回答
- [ ] unicode2 三个小问的书面回答（含示例）
- [ ] BPE 训练函数 + `adapters.run_train_bpe`，通过 `test_train_bpe.py`
- [ ] TinyStories 10K 分词器：vocab/merges 序列化文件 + 耗时/内存/最长 token 简答 + profiling 简答
- [ ] OpenWebText 32K 分词器：vocab/merges 序列化文件 + 最长 token 简答 + 两分词器对比简答
- [ ] `Tokenizer` 类 + `adapters.get_tokenizer`，通过 `test_tokenizer.py`
- [ ] 实验 (a)–(c) 简答：压缩比、跨域分词现象、吞吐量估算
- [ ] 实验 (d)：TinyStories 和 OWT 的训练/验证集编码为 `uint16` NumPy 数组 + 解释为何 `uint16` 合适
