# `pretokenization_example.py` 逐行解读

> 文件位置：`cs336_basics/pretokenization_example.py`
> 这是课程官方提供的起始代码，讲义 2.5 节明确说明可以直接照搬使用。
> 它解决的问题是：**为并行化预分词把大文件切成若干块（chunk），并保证每个切分点恰好落在特殊 token（如 `<|endoftext|>`）的开头**，这样各块可以互相独立地统计预 token 计数，不会在文档边界两侧产生错误的合并。

---

## 整体结构

文件只有两部分：

1. `find_chunk_boundaries()` 函数 —— 计算切分点（返回文件内的字节偏移量列表）；
2. 底部的 Usage 示例 —— 演示如何用这些切分点逐块读取文件（串行版，供你改成多进程并行版）。

---

## 第 1–2 行：导入

```python
import os
from typing import BinaryIO
```

- **第 1 行** `import os`：这里只用到 `os.SEEK_END` 这一个常量（见第 17 行），它是 `file.seek()` 的定位模式参数。
- **第 2 行** `from typing import BinaryIO`：导入类型注解 `BinaryIO`，表示"以二进制模式打开的文件对象"。它只用于函数签名的类型提示，不影响运行；但它传达了一个重要约定——**这个函数要求文件以 `"rb"`（二进制）模式打开**，因为所有偏移量都是按字节算的。

---

## 第 5–9 行：函数签名

```python
def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
```

- `file: BinaryIO`：已打开的二进制文件对象。
- `desired_num_chunks: int`：**期望的**块数，一般等于你要开的进程数（比如 CPU 核数）。注意是"期望"——实际返回的块数可能更少（见第 49 行）。
- `split_special_token: bytes`：作为切分标记的特殊 token，注意类型是 `bytes` 而不是 `str`（例如 `b"<|endoftext|>"`），因为我们是在字节流里查找它。
- 返回值 `list[int]`：一串**字节偏移量**，长度为"块数 + 1"。第 `i` 块就是文件的 `[boundaries[i], boundaries[i+1])` 区间。

## 第 10–13 行：docstring

```python
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
```

说明两件事：① 切出的块可以**独立计数**（这正是并行的前提）；② 如果多个边界猜测点向后搜索时撞到了同一个特殊 token，去重后**块数可能少于期望值**。

## 第 14 行：入参断言

```python
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"
```

防御性检查：如果你误传了字符串 `"<|endoftext|>"` 而不是字节串 `b"<|endoftext|>"`，这里会直接报错。原因是第 34 行读出来的 `mini_chunk` 是 `bytes`，`bytes.find()` 只能查找 `bytes`——`str` 和 `bytes` 在 Python 3 里不能混用。

## 第 16–19 行：获取文件总字节数

```python
    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)
```

- **第 17 行** `file.seek(0, os.SEEK_END)`：把文件指针移动到"距文件**末尾** 0 字节"处，即文件结尾。`seek(offset, whence)` 的第二个参数指定参照点，`os.SEEK_END` 表示从末尾算起。
- **第 18 行** `file.tell()`：返回当前指针位置。因为指针刚被移到末尾，这个位置就等于**文件总字节数**。这是不把文件读进内存就获取其大小的标准手法（对几个 GB 的语料很重要）。
- **第 19 行** `file.seek(0)`：把指针移回文件开头，恢复现场，避免影响后面的读取。

## 第 21 行：估算每块大小

```python
    chunk_size = file_size // desired_num_chunks
```

整除得到每块的**理论**大小（字节）。这只是初始猜测，后面会把每个边界向后挪到最近的特殊 token 处。

## 第 23–26 行：初始化边界猜测

```python
    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size
```

- **第 25 行**：生成 `desired_num_chunks + 1` 个均匀分布的偏移量：`[0, chunk_size, 2*chunk_size, ..., desired_num_chunks*chunk_size]`。n 块需要 n+1 个端点，第 i 块是 `[boundaries[i], boundaries[i+1])`（注释里说的"含起点、不含终点"就是这个意思）。
- **第 26 行**：由于第 21 行是整除，`desired_num_chunks * chunk_size` 可能略小于 `file_size`（余数被丢掉了），所以强制把最后一个端点改成 `file_size`，保证最后一块能覆盖到文件真正的结尾。

## 第 28 行：设置"向后探测"的读取步长

```python
    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time
```

每次向后读 4096 字节（4KB，恰好是常见的磁盘页大小）来查找特殊 token，而不是一次读一大段。这样即使边界离下一个 `<|endoftext|>` 很远，也只按需读取少量数据。

## 第 30–46 行：把每个中间边界对齐到特殊 token

```python
    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size
```

- **第 30 行** `for bi in range(1, len(chunk_boundaries) - 1)`：只调整**中间**的边界。第一个边界（0，文件开头）和最后一个边界（`file_size`，文件结尾）天然合法，不需要动。
- **第 31 行**：取出这个边界当前的猜测位置。
- **第 32 行** `file.seek(initial_position)`：把文件指针移到猜测位置，准备从这里开始向后扫描。
- **第 33 行** `while True`：不断向后读，直到找到特殊 token 或读到文件尾。
- **第 34 行** `file.read(mini_chunk_size)`：从当前指针读最多 4096 字节。`read()` 会自动推进指针，所以循环每转一圈就往后看 4KB。
- **第 37–39 行**：`read()` 返回空字节串 `b""` 表示已到文件末尾（EOF）——说明从猜测位置到文件结尾都没有特殊 token，那就把这个边界直接设为 `file_size`，退出循环。（去重后这个边界会和最后一个端点合并，块数因此变少。）
- **第 42 行** `mini_chunk.find(split_special_token)`：在这 4KB 里查找特殊 token，返回**它在 `mini_chunk` 内的相对下标**，找不到返回 `-1`。
- **第 43–45 行**：找到了，就把边界更新为 `initial_position + found_at`——即特殊 token 在**整个文件**中的绝对偏移（块内相对位置 + 本次 mini chunk 的起始位置）。注意边界落在特殊 token 的**第一个字节**上，所以特殊 token 本身归属于**下一块**的开头。
- **第 46 行**：这 4KB 里没找到，就把 `initial_position` 前移 4096，继续读下一段。（文件指针已经被 `read` 自动推进了，这行只是同步记录"当前 mini chunk 从哪开始"，供第 44 行计算绝对偏移用。）

> **一个值得注意的边界情况**：如果特殊 token 恰好**跨越两个 mini chunk 的接缝**（比如 `<|endo` 在这 4KB 结尾、`ftext|>` 在下 4KB 开头），`find` 在两段里都查不到它，会被跳过、继续找下一个出现位置。对本作业来说这不影响正确性——边界只是会落到再往后的一个特殊 token 上，块划分依然合法，只是没那么均匀。

## 第 48–49 行：去重并排序后返回

```python
    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))
```

如果两个相邻的初始猜测点向后扫描时找到了**同一个**特殊 token（在特殊 token 稀疏、或块很小的情况下会发生），它们会得到相同的偏移量。`set()` 去重、`sorted()` 恢复升序。这就是"实际块数可能少于期望块数"的来源——对应 docstring 第二句。

---

## 第 52–62 行：使用示例（Usage）

```python
## Usage
with open(..., "rb") as f:
    num_processes = 4
    boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

    # The following is a serial implementation, but you can parallelize this
    # by sending each start/end pair to a set of processes.
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
        # Run pre-tokenization on your chunk and store the counts for each pre-token
```

- **第 53 行** `open(..., "rb")`：`...` 是**占位符**（Python 的 `Ellipsis` 字面量），你要换成实际的语料路径，比如 `data/TinyStoriesV2-GPT4-train.txt`。**这个文件原样是跑不起来的**——这段 Usage 写在模块顶层，import 时就会执行并报错，所以它是给你"抄进自己代码里改"的，不是直接运行的。`"rb"` 二进制模式与前面按字节 seek/read 的逻辑配套。
- **第 54 行**：期望块数，示例取 4；实际中通常取你打算开的进程数（如 `multiprocessing.cpu_count()`）。
- **第 55 行**：调用上面的函数拿到边界列表，特殊 token 传的是**字节串** `b"<|endoftext|>"`。
- **第 59 行** `zip(boundaries[:-1], boundaries[1:])`：把边界列表两两配对成 `(start, end)` 区间。例如 `[0, 100, 250, 400]` 会配出 `(0,100), (100,250), (250,400)`——正是每一块的起止偏移。
- **第 60 行** `f.seek(start)`：跳到这一块的起点。
- **第 61 行** `f.read(end - start)`：精确读出这一块的字节，然后 `.decode("utf-8", errors="ignore")` 解码成字符串。`errors="ignore"` 表示丢弃无法解码的字节——由于边界都对齐在特殊 token 上（UTF-8 中 ASCII 字符不会出现在多字节字符中间），正常情况下块的切口不会劈开一个多字节字符，这里只是兜底。
- **第 62 行**（注释）：留给你完成的部分——对 `chunk` 做预分词（先按特殊 token `re.split`，再用 GPT-2 正则 `re.finditer`），统计每个预 token 的出现次数。
- **第 57–58 行**（注释）：提示这个循环是**串行**的；并行化时，把每个 `(start, end)` 连同文件路径发给一个工作进程（例如 `multiprocessing.Pool`），每个进程自己打开文件、seek 到 start、读自己的那块并统计，最后把各进程的计数字典合并。

---

## 为什么要这样设计（对应讲义 2.5 节）

1. **为什么边界必须对齐特殊 token？** `<|endoftext|>` 是文档分隔符，训练 BPE 时本来就不允许跨文档合并。把切分点放在它上面，任意切分都不会破坏合并统计的正确性——这就是讲义说"这种分块方式永远是合法的"的原因。
2. **为什么不直接把文件均分？** 均分点可能落在一个单词甚至一个多字节 UTF-8 字符中间，两个进程会各得到半个预 token，计数就错了。
3. **为什么用 seek/read 而不是整个读入？** 语料可能有几个 GB（OWT），逐块 seek/read 让内存占用与块大小而非文件大小成正比。

## 使用时的注意事项

- 传给函数的特殊 token 必须是 `bytes`（第 14 行的 assert 会拦住 `str`）。
- 拿到 chunk 字符串后，别忘了讲义的要求：先用 `re.split`（配合 `re.escape`）按特殊 token 把 chunk 切成文档，再在每篇文档内部跑预分词正则——特殊 token 本身不参与合并计数。
- 多进程并行时，不要把打开的文件对象传给子进程（文件句柄不能跨进程共享指针），传"路径 + start + end"让子进程自己打开更稳妥。
