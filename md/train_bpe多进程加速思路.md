# `train_bpe` 多进程加速思路 & 实测规划

> 承接对话：`num_processes` 目前只控制切成几段边界，`bpe.py` 里第74-81行的 pre-tokenization 循环是**单进程串行**跑的，没有真正并行。本文整理"如果要改成真正多进程"的思路，以及配合你新建的 `cs336_basics/bpe_train.py` 做实测的规划。
> **不含任何代码改动**——`bpe.py` / `bpe_train.py` 都没有被动过，这里只是设计笔记。

---

## 1. 现状：为什么现在是单进程

```python
# bpe.py:71-81
num_processes = 4    #可改
with open(input_path, "rb") as f:
    boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
    for start, end in zip(boundaries[:-1], boundaries[1:]):   # 普通 for 循环
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
        for para in re.split(escaped, chunk):
            for matched in re.finditer(PAT, para):
                key = tuple(bytes([i]) for i in matched.group().encode('utf-8'))
                frequency[key] += 1
```

`find_chunk_boundaries` 只负责"切出边界在哪"，真正读取 + 正则匹配 + 计数这部分，是主进程一段一段顺序处理的。`num_processes=4` 这个名字容易让人误以为已经并行，实际语义只是"文件切成4段来读"。

**验证方法**：跑的时候用 `htop` 看 CPU 占用，单进程版本只有一个核心会满载，其余核心闲置。

---

## 2. 改造思路：三步

### 2.1 把单个 chunk 的处理逻辑拆成一个顶层函数

`multiprocessing` 要求传给子进程的函数能被 `pickle`。现在这段逻辑是写在 `train_bpe` 内部、直接闭包用外层变量（`escaped`），要拆成一个独立的、参数自包含的顶层函数：

```python
def _count_chunk(args: tuple[str, int, int, str]) -> Counter:
    input_path, start, end, escaped = args
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
    local_freq = Counter()
    for para in re.split(escaped, chunk):
        for matched in re.finditer(PAT, para):
            key = tuple(bytes([i]) for i in matched.group().encode('utf-8'))
            local_freq[key] += 1
    return local_freq
```

**关键点**：每个子进程要**自己重新 `open` 文件**，不能把已打开的 `BinaryIO` 对象传给子进程——文件句柄不能跨进程共享（`pickle` 不了）。所以传的是 `input_path`（路径字符串）+ `start`/`end`（字节偏移量），子进程内部自己开文件、`seek`、`read`。

### 2.2 用 `multiprocessing.Pool` 分发 + 合并

```python
from multiprocessing import Pool

with Pool(num_processes) as pool:
    tasks = [(input_path, s, e, escaped) for s, e in zip(boundaries[:-1], boundaries[1:])]
    results = pool.map(_count_chunk, tasks)

frequency = Counter()
for local_freq in results:
    frequency.update(local_freq)   # Counter 支持直接相加/update 合并
```

`pool.map` 把 `tasks` 里每个元素分给一个 worker 进程执行 `_count_chunk`，各自返回一个局部 `Counter`；主进程收集后用 `update`（等价于逐 key 相加）合并成全局频率表。这一步合并是单进程串行的，但通常远小于 pre-tokenization 本身的耗时。

### 2.3 `if __name__ == "__main__":` 守卫是硬性要求

`multiprocessing` 在 macOS/Windows（默认 `spawn` 方式）下会**重新 import 主模块**来创建子进程。这正是你在 [Section5.3_训练脚本实现指南.md](Section5.3_训练脚本实现指南.md#L46-L57) 里已经确认过的坑，同样的道理适用在这里：

- 如果 `train_bpe` 只是被 `import` 后调用（比如从 `bpe_train.py` 里 `from bpe import train_bpe` 再调用），本身没问题，因为顶层没有直接执行训练逻辑；
- **但 `bpe_train.py` 自己已经写对了**——`main()` 里跑训练，外面套了 `if __name__ == '__main__':`（[bpe_train.py:16-17](../cs336_basics/bpe_train.py#L16-L17)）。这个守卫本来是给"脚本自己会启动多进程"这件事准备的，现在派上用场了：如果 `train_bpe` 内部改成用 `Pool`，`bpe_train.py` 已经具备了安全跑多进程的前提条件，不用再改。

---

## 3. 需要权衡的点

| 考虑因素 | 说明 |
|---|---|
| **进程创建开销** | 尤其 Windows/macOS 的 `spawn`：每个子进程都要重新 import 整个模块。文件不大、`num_processes` 设太多时，创建/销毁开销可能超过并行收益 |
| **`num_processes` 怎么定** | 目前硬编码 `4`。改造后建议从 `**kwargs` 读出来（比如 `kwargs.get("num_processes", os.cpu_count())`），让调用者能控制。本机 `os.cpu_count()` 实测是 24 |
| **边界数量上限** | `find_chunk_boundaries` 按 `<|endoftext|>` 对齐边界，如果 `desired_num_chunks` 远大于文档数，切出来的边界会重复，第53行 `sorted(set(...))` 去重后**并行度可能达不到你设的数字** |
| **`re.split` + `escaped` 的正则对象** | 每个子进程会各自 `re.compile`（隐式，因为 `re.split` / `re.finditer` 内部有缓存），不共享主进程的正则编译结果，属于正常开销，不是 bug |
| **返回值大小** | 每个 `local_freq` 是一个 `Counter`，chunk 越多、词表越大，`pickle` 传回主进程的数据量越大——如果 pretoken 种类是几十万级别，这部分序列化开销值得关注 |

---

## 4. 配合 `bpe_train.py` 做实测规划

现状：`data/` 目录**目前是空的**，`bpe_train.py` 里写的路径 `data/TinyStoriesV2-GPT4-valid.txt` 还不存在（本文写作时确认过，`ls data/` 为空）。所以下面是数据就位后的操作规划，不是已经跑出的结果。

### 4.1 `bpe_train.py` 现在做的事

```python
# cs336_basics/bpe_train.py
def main():
    t = time.time()
    vocab_size = 10000
    special_tokens = ['<|endoftext|>']
    input_path = 'data/TinyStoriesV2-GPT4-valid.txt'
    vocab = train_bpe(input_path, vocab_size, special_tokens)[0]
    print(time.time()- t)
```

一个最小的计时脚本：跑一次 `train_bpe`，打印总耗时。这正好是做"单进程 vs 多进程"对比实验的骨架，稍微扩展一下就能用。

### 4.2 建议的实测步骤（数据下载好之后）

1. **先用 `valid` 集**（比 `train` 集小得多）跑通一次，拿到单进程的耗时基线——`bpe_train.py` 现在这样就够用。

2. **如果改造成多进程版本**，把 `main()` 扩展成可以传 `num_processes` 的对比实验：
   ```python
   for n in [1, 2, 4, 8, os.cpu_count()]:
       t = time.time()
       train_bpe(input_path, vocab_size, special_tokens, num_processes=n)
       print(f"num_processes={n}: {time.time()-t:.1f}s")
   ```
   这样能画出"进程数 vs 耗时"的曲线，找到收益开始打平（甚至因为进程开销反而变慢）的拐点，而不是凭感觉设一个数字。

3. **验证正确性不受影响**：多进程只是把统计过程拆开再合并，`vocab` / `merges` 的结果理论上应该和单进程版本**完全一致**（因为合并的是词频统计，不依赖处理顺序）。可以在小数据集上同时跑单进程和多进程版本，断言两边返回的 `vocab` 和 `merges` 相同，作为改造后的正确性回归检查。

4. **再上 `train` 集**（体量大得多，官方 TinyStories 训练集是 GB 级别），这时候多进程的收益才会明显体现——`valid` 集通常不够大，单进程可能几秒就跑完，进程创建开销占比会显得不划算，容易得出"多进程没用"的错误结论。

### 4.3 计时时要注意的干扰因素

- **第一次运行文件系统缓存是冷的**，如果磁盘不是 SSD，首次读取会比后续几次慢，建议重复跑 2-3 次取稳定值。
- **`num_processes` 对比实验里，进程数从小到大依次跑**，避免机器上同时有其他重负载进程干扰计时。
- 记得关注 `find_chunk_boundaries` 的边界去重效应（见上面表格）——如果传的 `num_processes` 比文档数还多，实际并行度会被压低，曲线可能不是单调下降的。

---

## 5. 相关笔记

- [bpe两版效率实测对比.md](bpe两版效率实测对比.md) —— 之前做过的效率对比，方法论可以直接复用到这次的多进程对比
- [Section5.3_训练脚本实现指南.md](Section5.3_训练脚本实现指南.md#L46-L57) —— `if __name__ == "__main__":` 守卫的必要性（多进程 spawn 场景）
- [Section5.3_train.py代码走读与待补清单.md](Section5.3_train.py代码走读与待补清单.md) —— §5-A3 提到的数据准备管道，跟这里的多进程 tokenize 是同一条链路上的两个环节
