# BPE Tokenizer 实现清单

> 依据:`cs336_assignment1_basics.pdf` 第 2 章。本文件只列出**需要实现什么**(接口、功能、输入输出),不含实现细节。

## 总览:两大交付物

| 交付物 | 位置 | 对接的 adapter | 对应测试 |
|--------|------|----------------|----------|
| BPE 训练函数 | `cs336_basics/`(如 `bpe.py`) | `tests/adapters.py` 中的 `run_train_bpe` | `uv run pytest tests/test_train_bpe.py` |
| `Tokenizer` 类 | `cs336_basics/` | `tests/adapters.py` 中的 `get_tokenizer` | `uv run pytest tests/test_tokenizer.py` |

`adapters.py` 里只写胶水代码(调用你的实现),不放实质逻辑。

---

## 一、BPE 训练函数(Problem `train_bpe`,15 分)

**功能**:给定一个文本文件路径,训练一个 byte-level BPE tokenizer,产出词表和合并规则。

### 输入参数

| 参数 | 类型 | 含义 |
|------|------|------|
| `input_path` | `str` | 训练语料文本文件的路径 |
| `vocab_size` | `int` | 最终词表大小上限(含 256 个初始字节 + 合并产生的 token + 特殊 token) |
| `special_tokens` | `list[str]` | 要加入词表的特殊 token。训练时作为硬分割边界,不参与 merge 统计 |

### 返回值

| 返回值 | 类型 | 含义 |
|--------|------|------|
| `vocab` | `dict[int, bytes]` | 词表:token ID → token 字节串 |
| `merges` | `list[tuple[bytes, bytes]]` | 合并规则列表,`(A, B)` 表示 A 与 B 合并;**按创建顺序排列** |

### 必须满足的行为规则(讲义规定)

- 词表从 256 个字节值初始化,特殊 token 也要有固定 ID
- 预分词使用 GPT-2 正则 `PAT`(讲义 2.4 节给出),用 `regex` 包 + `re.finditer`
- 预分词前先按特殊 token **split** 语料(`re.split` + `re.escape`),保证 merge 不跨文档边界
- merge 不跨 pre-token 边界
- 频次并列时选**字典序较大**的 pair
- 合并进行到词表达到 `vocab_size` 为止

### 配套实验(写进 writeup)

- `train_bpe_tinystories`(2 分):TinyStories,vocab 10K;报告耗时/内存、最长 token。限时 30 分钟(提示:并行预分词后应能 < 2 分钟)
- `train_bpe_expts_owt`(2 分):OpenWebText,vocab 32K;报告最长 token,对比两个 tokenizer

---

## 二、`Tokenizer` 类(Problem `tokenizer`,15 分)

**功能**:给定词表 + 合并规则,在文本(`str`)与 token ID 序列(`list[int]`)之间双向转换,支持用户自定义特殊 token(若不在词表中则追加)。

### 需要的方法

#### `__init__(self, vocab, merges, special_tokens=None)`

用给定的词表、合并规则、(可选)特殊 token 列表构造 tokenizer。

| 参数 | 类型 |
|------|------|
| `vocab` | `dict[int, bytes]` |
| `merges` | `list[tuple[bytes, bytes]]` |
| `special_tokens` | `list[str] \| None`(默认 `None`) |

#### `from_files(cls, vocab_filepath, merges_filepath, special_tokens=None)`(类方法)

从磁盘上序列化好的词表和合并规则文件构造并返回 `Tokenizer`(文件格式与你训练代码的输出格式一致)。

| 参数 | 类型 |
|------|------|
| `vocab_filepath` | `str` |
| `merges_filepath` | `str` |
| `special_tokens` | `list[str] \| None` |

#### `encode(self, text: str) -> list[int]`

把输入文本编码为 token ID 序列。要求:

- 先预分词,再对每个 pre-token **按训练时的创建顺序**依次应用 merges
- 特殊 token 必须保持为单个 token,绝不被拆开

#### `encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]`

接收字符串的可迭代对象(如文件句柄),返回**惰性生成** token ID 的迭代器。用途:对内存放不下的大文件做常数内存编码。注意 token 不能跨 chunk 边界,否则结果与整体编码不一致。

#### `decode(self, ids: list[int]) -> str`

把 token ID 序列解码回文本:查词表取字节串、拼接、按 UTF-8 解码。非法字节序列须替换为官方替换字符 U+FFFD(`errors='replace'`)。

---

## 三、配套实验(Problem `tokenizer_experiments`,4 分)

- (a) 各取 10 篇 TinyStories / OWT 文档,用对应 tokenizer 编码,计算压缩比(bytes/token)
- (b) 用 TinyStories tokenizer 编码 OWT 样本,对比压缩比并定性描述
- (c) 估算 tokenizer 吞吐量(bytes/s),推算编码 825GB(The Pile)需要多久
- (d) 用两个 tokenizer 分别把各自的训练集/验证集编码为 token ID 序列并保存(后续训练 LM 要用)

---

## 四、书面题(不涉及代码)

- `unicode1`(1 分):`chr(0)` 相关的三个小问题
- `unicode2`(3 分):UTF-8 vs UTF-16/32;分析错误的逐字节 decode 函数并举反例;给出无法解码的双字节序列

---

## 建议推进顺序

1. 两道 unicode 书面题(建立字节层面直觉)
2. 手推讲义 2.4 节 `bpe_example` 小例子
3. 朴素版训练函数 → 通过 `test_train_bpe`
4. 性能优化(并行预分词 + 增量 pair 计数),先用小的 debug 数据集迭代
5. `Tokenizer` 类 → 通过 `test_tokenizer`
6. 跑三组实验,写 writeup
