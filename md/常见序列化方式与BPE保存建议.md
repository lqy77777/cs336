# 常见序列化方式与 BPE 保存建议

## 1. 什么是序列化

序列化（serialization）是把内存中的对象转换成可以保存到磁盘或通过网络传输的数据；反序列化（deserialization）则是把这些数据恢复成程序中的对象。

例如，BPE 训练结束后得到：

```python
vocab: dict[int, bytes]
merges: list[tuple[bytes, bytes]]
```

如果不进行序列化，它们会随着 Python 进程退出而消失。保存后，下次可以直接加载 tokenizer，而不必重新训练。

序列化格式没有一种在所有场景下都“最好”。选择时主要考虑：

- 是否需要人直接阅读和检查
- 是否必须保留 Python 对象的原始类型
- 是否需要跨编程语言使用
- 文件大小和读写速度是否重要
- 数据来源是否可信
- 格式能否长期稳定地兼容新旧版本

## 2. 常见序列化方式对比

| 格式 | 可读性 | 跨语言 | 类型保留 | 主要优点 | 主要缺点 | 适合场景 |
|---|---:|---:|---:|---|---|---|
| JSON | 高 | 高 | 较弱 | 通用、易检查、标准库支持 | 不直接支持 `bytes`、tuple、整数键 | 配置、接口、需要检查的模型元数据 |
| Pickle | 无 | 低 | 很强 | 几乎可以原样保存 Python 对象，代码最少 | 加载不可信文件可能执行恶意代码；依赖 Python | 自己本地生成、短期使用的 Python 对象 |
| YAML | 高 | 高 | 一般 | 适合人工编辑复杂配置 | 解析规则复杂；加载不可信内容也需注意安全 | 配置文件，不适合大量 BPE 数据 |
| MessagePack | 无 | 高 | 较强 | 比 JSON 紧凑，原生支持二进制数据 | 需要第三方库；不方便人工检查 | 网络传输、紧凑的跨语言数据 |
| Protocol Buffers | 无 | 很高 | 由 schema 决定 | 类型明确、体积小、兼容性和跨语言能力强 | 需要定义和维护 `.proto` schema | 长期维护的生产系统和服务通信 |
| NumPy `.npy` / `.npz` | 无 | 一般 | 适合数组 | 保存数值数组简单快速；`.npy` 支持 memmap | 不适合任意嵌套 Python 对象 | 大型 token ID 数组、模型数值数据 |
| CSV | 高 | 高 | 弱 | 简单、表格软件可打开 | 不适合嵌套结构和任意二进制数据 | 简单二维表格 |
| Parquet | 低 | 高 | 适合表格 | 压缩好、列式读取高效、带 schema | 不适合 BPE merge 这种小型嵌套结构 | 大型结构化数据集和数据分析 |

## 3. JSON：可读性和通用性较好

JSON 原生只支持对象、数组、字符串、数字、布尔值和 `null`。它不能直接保存 Python 的 `bytes`：

```python
import json

json.dumps({0: b"hello"})  # TypeError
```

BPE 是 byte-level 的，token 可能包含任意字节，不一定是合法 UTF-8。因此不能简单地用下面的方法保存：

```python
token.decode("utf-8", errors="replace")
```

`errors="replace"` 会把不同的非法字节替换成同一个字符，导致信息丢失，之后无法准确恢复原始 `bytes`。

更稳妥的做法是将每段 `bytes` 转成十六进制字符串：

```python
b"hello".hex()            # "68656c6c6f"
bytes.fromhex("68656c6c6f")  # b"hello"
```

### 保存 BPE 的 `vocab` 和 `merges`

```python
import json

vocab_data = {
    str(token_id): token_bytes.hex()
    for token_id, token_bytes in vocab.items()
}

merges_data = [
    [left.hex(), right.hex()]
    for left, right in merges
]

with open("vocab.json", "w", encoding="utf-8") as f:
    json.dump(vocab_data, f, ensure_ascii=False, indent=2)

with open("merges.json", "w", encoding="utf-8") as f:
    json.dump(merges_data, f, ensure_ascii=False, indent=2)
```

这里有两个类型转换需要特别留意：

- JSON 对象的键必须是字符串，所以 `int` token ID 保存后会变成字符串。
- JSON 没有 tuple，merge pair 保存后会表现为数组。

### 恢复 BPE 的 `vocab` 和 `merges`

```python
import json

with open("vocab.json", "r", encoding="utf-8") as f:
    vocab_data = json.load(f)

with open("merges.json", "r", encoding="utf-8") as f:
    merges_data = json.load(f)

vocab = {
    int(token_id): bytes.fromhex(token_hex)
    for token_id, token_hex in vocab_data.items()
}

merges = [
    (bytes.fromhex(left), bytes.fromhex(right))
    for left, right in merges_data
]
```

merge 的先后顺序决定 BPE 规则的优先级，因此必须使用有序的 JSON 数组保存 `merges`，不能把它改成集合或无序映射。

## 4. Pickle：Python 中最省事

Pickle 可以直接保留这里的 `dict[int, bytes]`、list 和 tuple：

```python
import pickle

with open("vocab.pkl", "wb") as f:
    pickle.dump(vocab, f)

with open("merges.pkl", "wb") as f:
    pickle.dump(merges, f)
```

读取时：

```python
with open("vocab.pkl", "rb") as f:
    vocab = pickle.load(f)

with open("merges.pkl", "rb") as f:
    merges = pickle.load(f)
```

Pickle 的优点是代码短、类型恢复准确；缺点是文件不可读、跨语言能力差，而且绝对不要加载来源不可信的 pickle 文件，因为反序列化过程可能执行恶意代码。

## 5. NumPy：适合 token ID，不适合 BPE 规则

编码后的 token ID 是同一种整数组成的大型数组，适合保存为 `.npy`：

```python
import numpy as np

tokens = np.asarray(token_ids, dtype=np.uint16)
np.save("tokens.npy", tokens)
```

读取大型数组时可以使用内存映射：

```python
tokens = np.load("tokens.npy", mmap_mode="r")
```

但是 `vocab` 和 `merges` 是变长字节串组成的嵌套结构，不适合强行转换为普通 NumPy 数组。使用 `dtype=object` 最终通常仍依赖 pickle，也失去了 NumPy 数组紧凑、规则的优势。

## 6. 对当前 BPE 作业的推荐

如果目标是“保存到磁盘以便进一步检查”，推荐使用 **JSON + 十六进制编码**：

- 文件可以直接打开检查结构。
- 任意字节都可以无损保存和恢复。
- 不依赖 Python 专属的 Pickle 格式。
- `from_files()` 可以明确地完成字符串键到整数键、十六进制到 `bytes` 的恢复。

如果只是个人本地临时使用，并且确定文件完全可信，Pickle 是最省事的选择。它并非总体上“最好”，只是对纯 Python 原型最方便。

建议按数据职责分别选择格式：

| 要保存的数据 | 推荐格式 |
|---|---|
| `vocab`、`merges`，且需要检查或长期保存 | JSON + hex |
| `vocab`、`merges`，仅限可信的 Python 本地实验 | Pickle |
| 大型 token ID 数组 | `.npy`；超大文件配合 memmap |
| 需要与其他 tokenizer 工具兼容 | 采用该工具规定的 `vocab.json` / `merges.txt` 等标准格式 |
| 跨语言生产系统中的稳定接口 | Protocol Buffers 或明确约定的 MessagePack schema |

## 7. 保存后应该验证什么

序列化完成不代表一定正确。至少应该检查一次往返一致性（round-trip）：

```python
assert loaded_vocab == vocab
assert loaded_merges == merges
```

还可以重新构造 tokenizer，检查编码和解码结果：

```python
loaded_tokenizer = Tokenizer(
    loaded_vocab,
    loaded_merges,
    special_tokens=["<|endoftext|>"],
)

text = "A short story.<|endoftext|>"
ids = loaded_tokenizer.encode(text)
assert loaded_tokenizer.decode(ids) == text
```

长期保存时，还可以在文件中加入 `format_version`、词表大小和特殊 token 等元数据。这样未来修改文件结构时，可以根据版本选择正确的加载逻辑。

## 8. 一句话结论

没有绝对最好的序列化格式：对当前 byte-level BPE 作业，**JSON + hex 最适合检查和可靠保存，Pickle 最适合可信环境下的快速实验，而大型 token ID 应单独保存为 `.npy` 或内存映射文件**。
