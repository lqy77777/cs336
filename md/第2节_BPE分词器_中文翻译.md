# 2 字节对编码（Byte-Pair Encoding, BPE）分词器

> 译自 CS336 Assignment 1 (basics): Building a Transformer LM，Version 26.0.3，第 2 节。仅供个人学习使用。

在作业的第一部分，我们将训练并实现一个**字节级（byte-level）的字节对编码（BPE）分词器** [Sennrich et al., 2016; Wang et al., 2019]。具体来说，我们会把任意的（Unicode）字符串表示为字节序列，并在这个字节序列上训练 BPE 分词器。之后，我们将用这个分词器把文本（字符串）编码成 token（整数序列），用于语言建模。

## 2.1 Unicode 标准

Unicode 是一种文本编码标准，它把字符映射为整数**码点（code point）**。截至 Unicode 17.0（2025 年 9 月发布），该标准共定义了 159,801 个字符，覆盖 172 种文字系统。例如，字符 "s" 的码点是 115（通常记作 `U+0073`，其中 `U+` 是约定的前缀，`0073` 是 115 的十六进制表示），而字符 "牛" 的码点是 29275。在 Python 中，可以用 `ord()` 函数把单个 Unicode 字符转换成它的整数表示，用 `chr()` 函数把整数 Unicode 码点转换成对应字符的字符串。

```python
>>> ord('牛')
29275
>>> chr(29275)
'牛'
```

---

### 习题 (unicode1)：理解 Unicode（1 分）

(a) `chr(0)` 返回的是什么 Unicode 字符？

**交付物**：一句话回答。

(b) 这个字符的字符串表示（`__repr__()`）与它的打印表示有什么不同？

**交付物**：一句话回答。

(c) 当这个字符出现在文本中会发生什么？你可以在 Python 解释器里试试下面的代码，看看结果是否符合你的预期：

```python
>>> chr(0)
>>> print(chr(0))
>>> "this is a test" + chr(0) + "string"
>>> print("this is a test" + chr(0) + "string")
```

**交付物**：一句话回答。

---

## 2.2 Unicode 编码

虽然 Unicode 标准定义了从字符到码点（整数）的映射，但直接在 Unicode 码点上训练分词器并不现实：词表会大得离谱（约 15 万项）而且非常稀疏（因为很多字符极少出现）。因此，我们改用 **Unicode 编码（encoding）**，它把一个 Unicode 字符转换为一个**字节序列**。Unicode 标准本身定义了三种编码：UTF-8、UTF-16 和 UTF-32，其中 UTF-8 是互联网上占主导地位的编码（超过 98% 的网页使用它）。

要把 Unicode 字符串编码成 UTF-8，可以使用 Python 的 `encode()` 函数。要访问 Python `bytes` 对象底层的字节值，可以直接对它迭代（例如调用 `list()`）。最后，可以用 `decode()` 函数把 UTF-8 字节串解码回 Unicode 字符串。

```python
>>> test_string = "hello! こんにちは!"
>>> utf8_encoded = test_string.encode("utf-8")
>>> print(utf8_encoded)
b'hello! \xe3\x81\x93\xe3\x82\x93\xe3\x81\xab\xe3\x81\xa1\xe3\x81\xaf!'
>>> print(type(utf8_encoded))
<class 'bytes'>
>>> # 获取编码后字符串的字节值（0 到 255 之间的整数）。
>>> list(utf8_encoded)
[104, 101, 108, 108, 111, 33, 32, 227, 129, 147, 227, 130, 147, 227, 129, 171, 227, 129,
161, 227, 129, 175, 33]
>>> # 一个字节不一定对应一个 Unicode 字符！
>>> print(len(test_string))
13
>>> print(len(utf8_encoded))
23
>>> print(utf8_encoded.decode("utf-8"))
hello! こんにちは!
```

通过把 Unicode 码点转换成字节序列（例如通过 UTF-8 编码），我们实际上是把一个码点序列（21 位整数，共 159,801 个有效取值）变换成了一个字节值序列（0 到 255 范围内的整数）。这个长度为 256 的字节词表处理起来要**容易得多**。使用字节级分词时，我们也无需担心「词表外（out-of-vocabulary）」token 的问题，因为我们知道**任何**输入文本都可以表示为 0 到 255 之间的整数序列。

---

### 习题 (unicode2)：Unicode 编码（3 分）

(a) 为什么我们更倾向于在 UTF-8 编码的字节上训练分词器，而不是 UTF-16 或 UTF-32？对比一下这几种编码在各种输入字符串上的输出可能会有帮助。

**交付物**：一到两句话回答。

(b) 考虑下面这个（错误的）函数，它本意是把 UTF-8 字节串解码成 Unicode 字符串。为什么这个函数是错误的？请给出一个会产生错误结果的输入字节串示例。

```python
def decode_utf8_bytes_to_str_wrong(bytestring: bytes):
    return "".join([bytes([b]).decode("utf-8") for b in bytestring])

>>> decode_utf8_bytes_to_str_wrong("hello".encode("utf-8"))
'hello'
```

**交付物**：一个使 `decode_utf8_bytes_to_str_wrong` 产生错误输出的输入字节串示例，并用一句话解释这个函数为什么是错误的。

(c) 给出一个无法解码为任何 Unicode 字符的两字节序列。

**交付物**：一个示例，并用一句话解释。

---

## 2.3 子词分词（Subword Tokenization）

字节级分词虽然可以缓解词级分词器面临的词表外问题，但把文本切成字节会导致输入序列极长。这会拖慢模型训练：一个 10 个词的句子在词级语言模型中可能只有 10 个 token，但在字符级模型中可能有 50 个甚至更多 token（取决于词的长度）。处理这些更长的序列意味着模型每一步都要做更多计算。此外，在字节序列上做语言建模也更困难，因为更长的输入序列在数据中造成了长程依赖。

**子词分词**是词级分词器与字节级分词器之间的折中方案。注意，字节级分词器的词表有 256 个条目（字节取值为 0 到 255）。子词分词器用**更大的词表规模**来换取**对输入字节序列更好的压缩**。例如，如果字节序列 `b'the'` 在原始文本训练数据中频繁出现，那么给它在词表中分配一个条目，就能把这个原本 3 个 token 的序列压缩成单个 token。

我们如何选择要加入词表的子词单元呢？Sennrich et al. [3] 提出使用**字节对编码**（BPE；Gage [5]）——一种压缩算法，它迭代地把出现频率最高的一对字节替换（「合并」，merge）为一个新的、未使用过的索引。注意，这个算法向词表中添加子词 token，是为了最大化对输入序列的压缩——如果一个词在输入文本中出现得足够多，它就会被表示成单个子词单元。

用 BPE 构造词表的子词分词器通常被称为 **BPE 分词器**。在本作业中，我们将实现一个**字节级** BPE 分词器：词表项是字节或合并后的字节序列。这在词表外问题的处理和可控的输入序列长度两方面都能兼得。构造 BPE 分词器词表的过程被称为「**训练**」BPE 分词器。

## 2.4 BPE 分词器训练

BPE 分词器的训练过程包含三个主要步骤。

### 词表初始化（Vocabulary initialization）

分词器词表是一个从字节串 token 到整数 ID 的一一映射。由于我们训练的是字节级 BPE 分词器，初始词表就是所有字节的集合。字节有 256 种可能取值，所以初始词表大小为 256。

### 预分词（Pre-tokenization）

有了词表之后，原则上你可以统计文本中各字节彼此相邻出现的次数，然后从频率最高的字节对开始合并。但这样做计算开销很大，因为每次合并都要完整扫一遍语料。此外，直接跨越整个语料合并字节，可能会产生仅在标点上不同的 token（例如 `dog!` 和 `dog.`）。这些 token 会得到完全不同的 token ID，尽管它们在语义上很可能高度相似（毕竟只差一个标点）。

为避免这个问题，我们对语料进行**预分词（pre-tokenize）**。你可以把它理解为一次粗粒度的分词，帮助我们统计字符对出现的次数。举个例子，单词 `'text'` 可能是一个出现了 10 次的预 token。这时统计字符 't' 和 'e' 相邻出现的次数时，我们看到单词 'text' 中 't' 和 'e' 相邻，就可以直接把计数加 10，而不用逐个扫描语料。由于我们训练的是字节级 BPE 模型，每个预 token 都表示为 UTF-8 字节序列。

Sennrich et al. [3] 最初的 BPE 实现只是简单地按空白字符切分（即 `s.split(" ")`）来做预分词。基于 SentencePiece 的分词器（例如 Llama 1 和 Llama 2 的分词器）中仍能见到这种方法。

大多数现代分词器使用基于正则表达式的预分词器，这一做法始于 GPT-2 [Radford et al., 6]。我们将使用原始正则的一个稍微美化过的版本，取自 `github.com/openai/tiktoken/pull/234/files`：

```python
>>> PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
```

交互式地用这个预分词器切分一些文本，有助于更好地理解它的行为：

```python
>>> # 需要 `regex` 包
>>> import regex as re
>>> re.findall(PAT, "some text that i'll pre-tokenize")
['some', ' text', ' that', ' i', "'ll", ' pre', '-', 'tokenize']
```

不过，在你自己的代码中使用它时，应该用 `re.finditer`，以避免在构建「预 token → 计数」映射时把所有预分词结果都存下来。

### 计算 BPE 合并（Compute BPE merges）

现在我们已经把输入文本切成了预 token，并把每个预 token 表示为 UTF-8 字节序列，接下来就可以计算 BPE 合并（即训练 BPE 分词器）了。概括地说，BPE 算法迭代地统计每一对相邻字节的出现次数，找出频率最高的那一对（"A", "B"），然后把这一最高频对（"A", "B"）的所有出现全部**合并**，即替换为一个新 token "AB"。这个新合并出的 token 会被加入词表；因此，BPE 训练结束后的最终词表大小等于初始词表大小（我们这里是 256）加上训练过程中执行的 BPE 合并操作次数。为了提高 BPE 训练效率，我们**不考虑跨越预 token 边界的字节对**。[^1] 计算合并时，若字节对频率出现并列，我们通过**优先选择字典序更大的那一对**来确定性地打破平局。例如，如果 ("A", "B")、("A", "C")、("B", "ZZ") 和 ("BA", "A") 的频率并列最高，我们会合并 ("BA", "A")：

```python
>>> max([("A", "B"), ("A", "C"), ("B", "ZZ"), ("BA", "A")])
('BA', 'A')
```

### 特殊 token（Special tokens）

通常，会用某些字符串（例如 `<|endoftext|>`）来编码元数据（例如文档之间的边界）。编码文本时，我们往往希望把这些字符串当作「特殊 token」处理：它们永远不应被拆分成多个 token（即始终作为单个 token 保留）。例如，序列结束字符串 `<|endoftext|>` 应始终作为单个 token（即单个整数 ID）保留，这样我们才知道语言模型何时该停止生成。这些特殊 token 必须加入词表，从而拥有对应的固定 token ID。

Sennrich et al. [3] 的算法 1 给出了一个（效率不高的）BPE 分词器训练实现（基本就是按照我们上面列出的步骤）。作为第一个练习，实现并测试这个函数来检验你的理解，可能会很有帮助。

[^1]: 注意，Sennrich et al. [3] 的原始 BPE 表述中包含一个词尾（end-of-word）token。我们在训练字节级 BPE 模型时不添加词尾 token，因为所有字节（包括空白和标点）都已包含在模型词表中。既然我们显式地表示了空格和标点，学到的 BPE 合并自然会反映这些词边界。

---

### 示例 (bpe_example)：BPE 训练示例

这是一个取自 Sennrich et al. [3] 的风格化示例。考虑一个由以下文本组成的语料：

```
low low low low low
lower lower widest widest widest
newest newest newest newest newest newest
```

并且词表中有一个特殊 token `<|endoftext|>`。

**词表**：我们用特殊 token `<|endoftext|>` 和 256 个字节值初始化词表。

**预分词**：为了简化并聚焦于合并过程，本例中假设预分词只按空白切分。预分词并计数后，我们得到如下频率表：

```
{low: 5, lower: 2, widest: 3, newest: 6}
```

把它表示为 `dict[tuple[bytes, ...], int]` 会很方便，例如 `{(l,o,w): 5, …}`。注意，在 Python 中即使单个字节也是 `bytes` 对象；Python 没有表示单个字节的 `byte` 类型，就像它没有表示单个字符的 `char` 类型一样。

**合并**：我们首先查看每一对相邻字节，并把它们所在单词的频率求和：`{lo: 7, ow: 7, we: 8, er: 2, wi: 3, id: 3, de: 3, es: 9, st: 9, ne: 6, ew: 6}`。字节对 `('e', 's')` 和 `('s', 't')` 并列最高，于是取字典序更大的那对 `('s', 't')`。然后我们合并预 token，得到 `{(l,o,w): 5, (l,o,w,e,r): 2, (w,i,d,e,st): 3, (n,e,w,e,st): 6}`。

第二轮中，我们看到 `(e, st)` 是最常见的字节对（计数为 9），于是合并得到 `{(l,o,w): 5, (l,o,w,e,r): 2, (w,i,d,est): 3, (n,e,w,est): 6}`。继续下去，最终得到的合并序列为 `['s t', 'e st', 'o w', 'l ow', 'w est', 'n e', 'ne west', 'w i', 'wi d', 'wid est', 'low e', 'lowe r']`。

如果只取 6 次合并，我们得到 `['s t', 'e st', 'o w', 'l ow', 'w est', 'n e']`，词表元素为 `[<|endoftext|>, [...256 个字节字符], st, est, ow, low, west, ne]`。

用这个词表和合并集合，单词 `newest` 会被分词为 `[ne, west]`。

---

## 2.5 BPE 分词器训练实验

让我们在 TinyStories 数据集上训练一个字节级 BPE 分词器。查找/下载数据集的说明见第 1 节。开始之前，建议先看看 TinyStories 数据集，感受一下数据里都有什么。

### 并行化预分词（Parallelizing pre-tokenization)

你会发现主要瓶颈在预分词这一步。可以用内置库 `multiprocessing` 对代码做并行化来加速预分词。具体来说，我们建议在并行实现预分词时，对语料进行分块（chunk），同时确保分块边界恰好落在某个特殊 token 的开头。你可以直接照搬以下链接中的起始代码来获得分块边界，然后用这些边界把工作分发到各个进程：

```
https://github.com/stanford-cs336/assignment1-basics/blob/main/cs336_basics/pretokenization_example.py
```

这种分块方式永远是合法的，因为我们本来就不希望跨文档边界进行合并。就本作业而言，你总是可以这样切分。不必担心「收到一个不包含 `<|endoftext|>` 的超大语料」这种极端情况。

### 预分词前移除特殊 token（Removing special tokens before pre-tokenization）

在用正则表达式（`re.finditer`）运行预分词之前，你应该先把语料（或你的分块，如果是并行实现）中的所有特殊 token 剥离出去。务必**按特殊 token 切分（split）**文本，使得它们所分隔的文本之间不会发生任何合并。例如，如果语料（或分块）是 `[Doc 1]<|endoftext|>[Doc 2]`，你应该按特殊 token `<|endoftext|>` 切分，然后分别对 `[Doc 1]` 和 `[Doc 2]` 做预分词，这样文档边界两侧就不会发生合并。换句话说，特殊 token 在训练时定义了硬性的切分边界，但它们本身不应参与合并计数。可以用 `re.split`，以 `"|".join(special_tokens)` 作为分隔符来实现（注意小心使用 `re.escape`，因为特殊 token 中可能包含 `|`）。测试 `test_train_bpe_special_tokens` 会检验这一点。

### 优化合并步骤（Optimizing the merging step）

上面风格化示例中 BPE 训练的朴素实现之所以慢，是因为每次合并都要遍历所有字节对来找出最高频的那一对。然而，每次合并之后，唯一会发生变化的字节对计数是那些与被合并对有重叠的。因此，可以通过对所有字节对的计数建立索引并做**增量更新**，而不是每次都显式遍历所有字节对来统计频率，从而提升 BPE 训练速度。这种缓存策略能带来显著加速，不过要注意：BPE 训练中的合并部分在 Python 中是**无法**并行化的。

> **低资源小贴士：性能分析（Profiling）**
>
> 你应该使用 `cProfile` 或 `py-spy` 之类的性能分析工具找出实现中的瓶颈，集中精力优化瓶颈部分。

> **低资源小贴士：「缩小规模」（Downscaling）**
>
> 与其直接在完整的 TinyStories 数据集上训练分词器，我们建议先在一小部分数据（一个「调试数据集」）上训练。例如，可以先在 TinyStories 验证集上训练分词器——它只有 2.2 万篇文档，而不是 212 万篇。这体现了一个通用策略：只要有可能就缩小规模来加快开发速度，例如使用更小的数据集、更小的模型规模等。选择调试数据集或超参配置的规模需要仔细斟酌：你希望调试集大到足以呈现与完整配置相同的瓶颈（这样你做的优化才能泛化过去），但又不能大到跑一次要等很久。

---

### 习题 (train_bpe)：BPE 分词器训练（15 分）

**交付物**：编写一个函数：给定输入文本文件的路径，训练一个（字节级）BPE 分词器。你的 BPE 训练函数应（至少）处理以下输入参数：

**输入**

- `input_path: str` — BPE 分词器训练数据文本文件的路径。
- `vocab_size: int` — 一个正整数，定义最终词表的最大规模（包括初始字节词表、合并产生的词表项以及所有特殊 token）。
- `special_tokens: list[str]` — 要加入词表的字符串列表。这些特殊 token 在训练中被视为硬性边界，阻止跨越其所在位置的合并，但在计算合并统计量时不应把它们计入。

你的 BPE 训练函数应返回得到的词表和合并列表：

**输出**

- `vocab: dict[int, bytes]` — 分词器词表，是从 `int`（词表中的 token ID）到 `bytes`（token 字节串）的映射。
- `merges: list[tuple[bytes, bytes]]` — 训练产生的 BPE 合并列表。每个列表项是一个 `bytes` 二元组 `(<token1>, <token2>)`，表示 `<token1>` 与 `<token2>` 被合并。合并列表应按创建顺序排列。

要用我们提供的测试来检验你的 BPE 训练函数，你首先需要实现测试适配器 `adapters.run_train_bpe`，然后运行 `uv run pytest tests/test_train_bpe.py`。你的实现应能通过所有测试。可选地（这可能需要投入大量时间），你可以用某种系统级语言实现训练方法的关键部分，例如 C++（可考虑 `cppyy` 或 `nanobind`）或 Rust（用 PyO3）。如果你这么做，要注意哪些操作需要复制内存、哪些可以直接读取 Python 内存，并确保留下构建说明，或确保仅用 `pyproject.toml` 就能完成构建。另外注意，GPT-2 的正则表达式在大多数正则引擎中支持不佳，而在支持它的引擎中大多也太慢。我们验证过 Oniguruma 速度尚可且支持负向前瞻（negative lookahead），但 Python 的 `regex` 包甚至更快。

---

### 习题 (train_bpe_tinystories)：在 TinyStories 上训练 BPE（2 分）

(a) 在 TinyStories 数据集上训练一个字节级 BPE 分词器，最大词表规模为 10,000。务必把 TinyStories 的特殊 token `<|endoftext|>` 加入词表。将得到的词表和合并序列化到磁盘，以便后续检查。训练用了多少时间和内存？词表中最长的 token 是什么？它合理吗？

**资源要求**：≤ 30 分钟（无需 GPU），≤ 30 GB 内存

**提示**：借助在预分词阶段使用 `multiprocessing` 以及下面两个事实，BPE 训练应该可以做到 2 分钟以内：

- (a) `<|endoftext|>` token 在数据文件中用于分隔文档。
- (b) `<|endoftext|>` token 在应用 BPE 合并之前会被作为特殊情况处理。

**交付物**：一到两句话回答。

(b) 对你的代码做性能分析。分词器训练过程中哪一部分耗时最多？

**交付物**：一到两句话回答。

---

接下来，我们尝试在 OpenWebText 数据集上训练一个字节级 BPE 分词器。和之前一样，建议先看看数据集，更好地了解其内容。

### 习题 (train_bpe_expts_owt)：在 OpenWebText 上训练 BPE（2 分）

(a) 在 OpenWebText 数据集上训练一个字节级 BPE 分词器，最大词表规模为 32,000。将得到的词表和合并序列化到磁盘，以便后续检查。词表中最长的 token 是什么？它合理吗？

**资源要求**：≤ 12 小时（无需 GPU），≤ 100 GB 内存

**交付物**：一到两句话回答。

(b) 对比在 TinyStories 与 OpenWebText 上训练得到的分词器，分析它们的异同。

**交付物**：一到两句话回答。

---

## 2.6 BPE 分词器：编码与解码

在上一部分中，我们实现了一个在输入文本上训练 BPE 分词器的函数，得到了分词器词表和 BPE 合并列表。现在，我们将实现一个 BPE 分词器：加载给定的词表和合并列表，用它们把文本编码成 token ID，或把 token ID 解码回文本。

### 2.6.1 编码文本

用 BPE 编码文本的过程与我们训练 BPE 词表的方式相呼应。主要有以下几步。

**第 1 步：预分词。** 我们首先对序列做预分词，并把每个预 token 表示为 UTF-8 字节序列，与 BPE 训练时完全一样。我们将在**每个预 token 内部**把这些字节合并成词表元素，各预 token 独立处理（不跨预 token 边界合并）。

**第 2 步：应用合并。** 然后，我们取出 BPE 训练时产生的词表元素合并序列，**按其创建顺序**将其应用到我们的预 token 上。

---

#### 示例 (bpe_encoding)：BPE 编码示例

例如，假设输入字符串是 `'the cat ate'`，词表是 `{0: b' ', 1: b'a', 2: b'c', 3: b'e', 4: b'h', 5: b't', 6: b'th', 7: b' c', 8: b' a', 9: b'the', 10: b' at'}`，学到的合并是 `[(b't', b'h'), (b' ', b'c'), (b' ', b'a'), (b'th', b'e'), (b' a', b't')]`。首先，预分词器会把这个字符串切分成 `['the', ' cat', ' ate']`。然后我们逐个处理每个预 token，应用 BPE 合并。

第一个预 token `'the'` 初始表示为 `[b't', b'h', b'e']`。查看合并列表，我们发现第一个可应用的合并是 `(b't', b'h')`，应用后预 token 变为 `[b'th', b'e']`。然后回到合并列表，发现下一个可应用的合并是 `(b'th', b'e')`，应用后预 token 变为 `[b'the']`。最后再查看合并列表，发现没有更多可应用的合并了（因为整个预 token 已经合并成了单个 token），于是 BPE 合并应用完毕。对应的整数序列是 `[9]`。

对其余预 token 重复这一过程：预 token `' cat'` 应用 BPE 合并后表示为 `[b' c', b'a', b't']`，对应整数序列 `[7, 1, 5]`。最后一个预 token `' ate'` 应用 BPE 合并后是 `[b' at', b'e']`，对应整数序列 `[10, 3]`。因此，编码整个输入字符串的最终结果是 `[9, 7, 1, 5, 10, 3]`。

---

**特殊 token。** 你的分词器在编码文本时应能正确处理用户定义的特殊 token（在构造分词器时提供）。

**内存考量。** 假设我们要对一个无法放入内存的大文本文件进行分词。为了高效地对这个大文件（或任何其他数据流）分词，我们需要把它切成可管理的分块（chunk），逐块处理，使内存复杂度保持常数级，而不是随文本大小线性增长。这样做时，需要确保 token 不会跨越分块边界，否则得到的分词结果就会与「把整个序列放进内存一次性分词」这种朴素方法不一致。

### 2.6.2 解码文本

要把整数 token ID 序列解码回原始文本，只需在词表中查出每个 ID 对应的条目（一个字节序列），把它们拼接起来，然后把这些字节解码成 Unicode 字符串。注意，输入的 ID 序列并不保证能映射成合法的 Unicode 字符串（用户可能输入任意的整数 ID 序列）。如果输入的 token ID 无法产生合法的 Unicode 字符串，你应该用官方的 Unicode 替换字符 `U+FFFD` 来替换这些非法字节。[^2] `bytes.decode` 的 `errors` 参数控制如何处理 Unicode 解码错误，使用 `errors='replace'` 会自动把无法解码的数据替换为替换标记。

[^2]: 关于 Unicode 替换字符的更多信息，参见 `en.wikipedia.org/wiki/Specials_(Unicode_block)#Replacement_character`。

---

### 习题 (tokenizer)：实现分词器（15 分）

**交付物**：实现一个 `Tokenizer` 类：给定词表和合并列表，把文本编码成整数 ID，并把整数 ID 解码成文本。你的分词器还应支持用户提供的特殊 token（若词表中没有则追加进词表）。我们推荐以下接口：

- `def __init__(self, vocab, merges, special_tokens=None)` — 用给定的词表、合并列表以及（可选的）特殊 token 列表构造分词器。该函数应接受以下参数：
  - `vocab: dict[int, bytes]`
  - `merges: list[tuple[bytes, bytes]]`
  - `special_tokens: list[str] | None = None`
- `def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None)` — 类方法，从序列化的词表和合并列表（格式与你的 BPE 训练代码输出相同）以及（可选的）特殊 token 列表构造并返回一个 `Tokenizer`。该方法应接受以下额外参数：
  - `vocab_filepath: str`
  - `merges_filepath: str`
  - `special_tokens: list[str] | None = None`
- `def encode(self, text: str) -> list[int]` — 把输入文本编码成 token ID 序列。
- `def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]` — 给定一个字符串的可迭代对象（例如一个 Python 文件句柄），返回一个惰性产出 token ID 的生成器。这是对无法直接载入内存的大文件进行内存高效分词所必需的。
- `def decode(self, ids: list[int]) -> str` — 把 token ID 序列解码成文本。

要用我们提供的测试检验你的 `Tokenizer`，你首先需要实现测试适配器 `adapters.get_tokenizer`，然后运行 `uv run pytest tests/test_tokenizer.py`。你的实现应能通过所有测试。

---

## 2.7 实验

### 习题 (tokenizer_experiments)：分词器实验（4 分）

(a) 从 TinyStories 和 OpenWebText 中各采样 10 篇文档。用你之前训练好的 TinyStories 分词器和 OpenWebText 分词器（词表规模分别为 10K 和 32K），把这些采样文档编码成整数 ID。每个分词器的压缩比（字节数/ token 数）是多少？

**交付物**：一到两句话回答。

(b) 如果用 TinyStories 分词器去对你的 OpenWebText 样本做分词，会发生什么？对比压缩比，和/或定性地描述现象。

**交付物**：一到两句话回答。

(c) 估计你的分词器的吞吐量（例如以字节/秒计）。对 Pile 数据集（825 GB 文本）分词需要多长时间？

**交付物**：一到两句话回答。

(d) 用你的 TinyStories 和 OpenWebText 分词器，分别把对应的训练集和开发集编码成整数 token ID 序列。我们之后会用它来训练语言模型。我们建议把 token ID 序列化为数据类型 `uint16` 的 NumPy 数组。为什么 `uint16` 是一个合适的选择？

**交付物**：一到两句话回答。
