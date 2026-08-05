# `lru_cache` 与 BPE pre-token 缓存详解

## 1. 当前优化解决了什么问题

Tokenizer 的编码流程可以简化成：

```text
文本
→ 特殊 token 切分
→ 正则预分词
→ 对每个 pre-token 执行 BPE merge
→ token ID
```

TinyStories 中存在大量重复的 pre-token。例如：

```text
" the"
" little"
" was"
" said"
"."
```

前导空格属于 pre-token 的一部分，因此 `"the"` 和 `" the"` 是两个不同的缓存键。

BPE 编码是确定性的：只要同一个 `Tokenizer` 的 `vocab` 和 `merges` 不变，同一个 pre-token 每次都会得到完全相同的 token ID 序列。例如：

```text
" little" → (421, 83)
```

没有缓存时，`" little"` 每出现一次，程序都会重新执行完整的 BPE merge。加入缓存后，只有第一次出现时需要计算，后续出现时可以直接取出 `(421, 83)`。

## 2. 修改前的执行过程

修改前，每一个 pre-token 都需要执行：

```python
pretoken = matched.group()
encoded = pretoken.encode("utf-8")
keys = tuple(bytes([byte_id]) for byte_id in encoded)
```

然后反复执行：

```python
pairs = set(zip(keys[:-1], keys[1:]))
ranks = {pair: self.rank[pair] for pair in pairs}
min_pair = min(ranks, key=ranks.get)
```

每轮 merge 还会创建新的 `temp` 列表和 `keys` tuple，直到找不到可用的 merge。

假设语料中有 `N` 次 pre-token，而单次 merge 的平均成本为 `C_merge`，修改前的主要计算量近似为：

```text
N × C_merge
```

即使某个词已经计算过几十万次，程序也不会记住之前的结果。

## 3. 当前的缓存实现

当前代码把单个 pre-token 的 BPE 编码提取成独立方法：

```python
@lru_cache(maxsize=100_000)
def _encode_pretoken(
    self,
    pretoken: str,
) -> tuple[int, ...]:
    keys = tuple(
        bytes([byte_id])
        for byte_id in pretoken.encode("utf-8")
    )

    # 执行 BPE merge，最后返回 token ID tuple
    ...
```

`encode()` 中先取得正则匹配产生的字符串，再调用缓存方法：

```python
for matched in re.finditer(PAT, para):
    pretoken = matched.group()
    ids.extend(self._encode_pretoken(pretoken))
```

特殊 token 不进入普通 pre-token 缓存，而是直接查询词表：

```python
special_id = self.reversed_vocab[
    para.encode("utf-8")
]
ids.append(special_id)
```

这样可以保证 `<|endoftext|>` 被视为一个整体，不会被普通 BPE 逻辑拆开。

## 4. `@lru_cache` 装饰器实际上做了什么

下面的写法：

```python
@lru_cache(maxsize=100_000)
def _encode_pretoken(self, pretoken):
    ...
```

概念上相当于：

```python
_encode_pretoken = lru_cache(maxsize=100_000)(
    _encode_pretoken
)
```

也就是说，类中原来的函数被一个缓存包装器替代。每次调用时，包装器先根据参数构造缓存键，再决定是否执行原函数体。

因为这是实例方法，缓存键在概念上不只是：

```python
pretoken
```

而是：

```python
(self, pretoken)
```

因此，不同 `Tokenizer` 实例中的同一个 `" the"` 不会错误地共享结果。不同 tokenizer 可能有不同的 `vocab` 和 `merges`，所以把 `self` 作为缓存键的一部分是必要的。

## 5. 缓存未命中时发生什么

第一次调用：

```python
tokenizer._encode_pretoken(" little")
```

缓存中还不存在对应键，因此发生 cache miss：

```text
1. 根据 (self, " little") 查询缓存
2. 没有找到结果
3. 执行 _encode_pretoken() 函数体
4. 转换 UTF-8 bytes
5. 构造 bytes tuple
6. 执行全部 BPE merge
7. 得到 token ID tuple
8. 把输入键和输出结果存入缓存
9. 返回结果
```

可以抽象成：

```python
cache[(self, " little")] = (421, 83)
```

第一次调用不会比原来明显更快，因为完整计算仍然必须执行，而且多了一次缓存查询和保存。

## 6. 缓存命中时发生什么

之后再次调用：

```python
tokenizer._encode_pretoken(" little")
```

包装器能够找到之前保存的结果，因此发生 cache hit：

```text
1. 根据 (self, " little") 查询缓存
2. 找到 (421, 83)
3. 更新该条目的最近使用顺序
4. 直接返回 (421, 83)
```

这一次不会进入 `_encode_pretoken()` 的函数体，所以会跳过：

- `pretoken.encode("utf-8")`
- 为每个字节创建单字节 `bytes`
- 构造 `keys` tuple
- 构造相邻 pair 集合
- 构造 rank 字典
- 搜索最高优先级 merge
- 反复创建 `temp` 和新 tuple
- 把最终 bytes 查询为 token ID

缓存命中仍然需要对字符串计算或读取哈希、查询哈希表，并把返回的 ID 加到最终 `ids` 中，但这些操作通常远小于完整 BPE merge 的成本。

## 7. 为什么缓存键选在字符串阶段

更早的缓存方案使用：

```python
keys: tuple[bytes, ...]
```

作为缓存键。这种方案虽然能跳过 merge，但每次调用前仍然必须执行：

```text
str
→ UTF-8 bytes
→ 多个单字节 bytes
→ tuple
→ 查询缓存
```

当前方案直接使用 `matched.group()` 已经产生的 `str`：

```python
pretoken = matched.group()
ids.extend(self._encode_pretoken(pretoken))
```

因此命中缓存时，连 bytes tuple 都不用构造。对于缓存命中率接近 100% 的语料，这项差异非常明显。

## 8. 为什么返回 tuple 而不是 list

缓存值使用：

```python
tuple[int, ...]
```

而不是：

```python
list[int]
```

tuple 不可变，调用方无法意外修改缓存中的结果。例如，如果缓存返回 list，下面的代码可能破坏未来所有相同 pre-token 的结果：

```python
cached_ids = tokenizer._encode_pretoken(" little")
cached_ids.append(9999)
```

使用 tuple 后，编码完整文本时仍然可以通过 `extend()` 加入结果：

```python
ids.extend(cached_ids)
```

## 9. LRU 的含义和淘汰规则

LRU 是 Least Recently Used，即“最近最少使用”。

`maxsize=100_000` 表示最多保留 100,000 个缓存条目。这里的上限是条目数量，不是 100,000 字节。

每次缓存命中时，该条目会被移动到“最近使用”的位置。缓存已满后出现新 pre-token 时，会删除最长时间没有被访问的条目：

```text
缓存已经有 100,000 项
→ 出现一个新的 pre-token
→ 删除最久未使用的一项
→ 保存新项
```

这种策略适合自然语言，因为常用词会不断被访问并保留下来，偶尔出现一次的名字、数字和罕见字符串更容易被淘汰。

缓存内部主要依赖哈希表完成平均 `O(1)` 查询，同时维护最近使用顺序。`O(1)` 是平均复杂度；字符串哈希本身仍与字符串长度有关，但 pre-token 通常很短。

## 10. 修改前后计算量的区别

假设：

- pre-token 总出现次数为 `N`
- 不同 pre-token 数量为 `U`
- `U` 远小于 `N`

修改前主要计算量近似为：

```text
N × C_merge
```

修改后主要计算量近似为：

```text
U × C_merge
+ (N - U) × C_cache_lookup
```

因为缓存查询远快于 BPE merge，而且 TinyStories 中 `U << N`，所以总时间显著下降。

缓存并没有加速下面这些操作：

- 从磁盘读取文本
- `<|endoftext|>` 切分
- 正则表达式预分词
- 把 token ID 加入结果
- 将 NumPy batch 写入 `.bin`

它加速的是最重复、最昂贵的“单个 pre-token BPE merge”。

## 11. 本项目中的实测结果

在约 21 MB 的 TinyStories validation 文件上，使用同一个 10,000 大小的 tokenizer：

| 指标 | 修改前 | 字符串 pre-token 缓存后 |
|---|---:|---:|
| 编码时间 | 约 17.78 秒 | 约 1.70 秒 |
| token 数量 | 5,461,747 | 5,461,747 |
| 加速比 | 1 倍 | 约 10.5 倍 |

一次独立冷缓存实验得到的缓存统计约为：

```text
cache hits:   5,405,890
cache misses:    13,111
cache size:      13,111
hit rate:        99.758%
```

“冷缓存”表示实验开始前没有提前放入任何结果。即使如此，在一次完整遍历中，后半部分语料仍能复用前面已经见过的 pre-token。

修改前后产生的 token 数和完整 token 序列 SHA-256 相同：

```text
2fb3ee7ab58c997518072c55946e71e43877ff7d72128713511656e2d34c1642
```

这说明缓存只改变了计算方式，没有改变 tokenization 结果。

## 12. `lru_cache` 会不会占用内存

会。缓存是典型的“用内存换时间”。

每个缓存条目至少需要保存：

- 作为键的 `self` 引用和 pre-token 字符串
- 作为值的 token ID tuple
- 哈希表条目的管理开销
- LRU 最近使用顺序的管理开销

`lru_cache` 会对参数和返回值保持强引用。只要条目仍在缓存中，对应的 pre-token 字符串和 token ID tuple 就不会被垃圾回收。

在之前的独立进程实验中：

| 实现 | 峰值 RSS |
|---|---:|
| 无缓存 | 约 67.69 MiB |
| 100,000 上限的缓存 | 约 71.91 MiB |

这次数据只产生约 13,000 个不同 pre-token，所以峰值 RSS 约增加 4.2 MiB。RSS 是整个 Python 进程的指标，包含解释器、NumPy 和其他对象，因此这个差值只能作为实际效果参考，不能视为缓存对象的精确字节数。

如果语料产生接近 100,000 个缓存条目，内存增量可能达到几十 MB，取决于字符串长度和每个 token ID tuple 的长度。

## 13. 为什么不能使用无限缓存

下面的写法没有条目上限：

```python
@lru_cache(maxsize=None)
```

如果 GB 级语料包含大量唯一的名字、网址、数字、代码片段或随机字符串，缓存会持续增长，最终可能占用大量内存。

当前使用：

```python
@lru_cache(maxsize=100_000)
```

可以限制条目数量。需要注意，它限制的是条目数，不是精确内存；一个很长的 pre-token 和一个很短的 pre-token 都只算一个条目。

可以根据内存情况尝试：

```python
maxsize=50_000
maxsize=100_000
maxsize=200_000
```

然后比较命中率、编码时间和峰值内存。

## 14. 如何查看和清空缓存

查看统计：

```python
info = tokenizer._encode_pretoken.cache_info()
print(info)
```

输出类似：

```text
CacheInfo(
    hits=5405890,
    misses=13111,
    maxsize=100000,
    currsize=13111,
)
```

计算命中率：

```python
total = info.hits + info.misses
hit_rate = info.hits / total if total else 0.0
print(f"Cache hit rate: {hit_rate:.2%}")
```

清空缓存：

```python
tokenizer._encode_pretoken.cache_clear()
```

清空后，下一次遇到每个 pre-token 时会重新发生 cache miss。

查看配置：

```python
print(tokenizer._encode_pretoken.cache_parameters())
```

## 15. `vocab` 或 `merges` 改变时必须注意

缓存结果依赖：

```text
Tokenizer 实例
+ pre-token
+ vocab
+ merges
```

`lru_cache` 不知道 `self.vocab` 或 `self.merges` 的内部内容是否被修改。创建 tokenizer 并产生缓存后，如果原地修改它们，旧缓存可能变成错误结果。

因此应该把训练完成的 tokenizer 视为不可变对象。如果确实修改了 `vocab`、`merges`、`rank` 或反向词表，必须调用：

```python
tokenizer._encode_pretoken.cache_clear()
```

更稳妥的做法是创建一个新的 `Tokenizer` 实例。

## 16. 多个 Tokenizer 实例的注意事项

把 `@lru_cache` 直接写在实例方法上时，缓存包装器位于类的方法上。缓存键包含 `self`，所以结果不会在不同 tokenizer 之间混淆，但所有实例共享同一个最大条目池。

缓存会强引用出现在缓存键中的 `self`。如果一个长时间运行的程序不断创建并丢弃许多 tokenizer，旧实例可能因为仍有缓存条目而暂时不能释放，直到对应条目被淘汰或缓存被清空。

当前脚本通常只创建一个 tokenizer，因此这不是实际问题。如果程序会频繁创建不同 tokenizer，可以在不再使用时清空缓存，或者设计每个实例独立的缓存包装器。

## 17. 多进程编码的注意事项

不同 Python 进程拥有独立内存，因此每个 worker 都会有自己的 LRU cache：

```text
worker 1 → 独立缓存
worker 2 → 独立缓存
worker 3 → 独立缓存
```

缓存不会自动跨进程共享。多进程意味着：

- 每个进程都要经历自己的冷缓存阶段
- 相同 pre-token 可能在不同进程中各计算一次
- 缓存内存占用会随进程数增加

因此已经有约 10 倍单进程缓存加速时，不一定还需要多进程编码。应该先测量单进程是否已经满足需求，再决定是否接受多进程的内存和通信成本。

## 18. 什么情况下缓存收益较小

缓存最适合重复度高的自然语言。下面这些数据可能命中率较低：

- 大量互不相同的随机字符串
- UUID、哈希值和长数字序列
- 唯一 URL 或日志 ID
- 高度多样化的代码和压缩后文本
- 已经预先打乱成几乎不重复的片段

如果大部分 pre-token 只出现一次，缓存会增加查询和内存管理开销，却很少复用结果。这时应通过 `cache_info()` 观察命中率，而不是默认缓存一定有帮助。

## 19. 一个与缓存独立的内存问题

当前 `_encode_pretoken()` 中如果使用：

```python
self.rank[pair]
```

而 `self.rank` 是 `defaultdict`，查询不存在的 pair 会把该 pair 自动插入字典。GB 级编码时，`rank` 可能不必要地增长。

更稳妥的是：

```python
self.rank.get(pair, self.max)
```

这项修改不是 LRU cache 的一部分，但可以避免另一种隐藏的内存增长，而且不会改变 merge 结果。

## 20. 缓存不会被序列化

`tokenizer.json` 只需要保存：

- `vocab`
- `merges`
- `special_tokens`
- 格式版本等元数据

LRU cache 是运行时加速状态，不应该写入 JSON。重新加载 tokenizer 后缓存为空，随后会在实际编码过程中逐渐建立。

## 21. 总结

修改前，每次出现 pre-token 都重新执行完整 BPE merge：

```text
每次出现 → bytes 转换 → pair/rank → 多轮 merge → token ID
```

修改后：

```text
第一次出现 → 完整计算并保存
再次出现   → 哈希查询并直接返回 token ID tuple
```

TinyStories 的 pre-token 重复率很高，因此冷缓存单次遍历就能达到约 99.76% 的命中率，使 validation 编码从约 17.78 秒降到约 1.70 秒，同时 token 数和完整输出哈希保持一致。

代价是缓存会占用额外内存。`maxsize=100_000` 通过 LRU 淘汰把条目数量限制在固定范围内；在当前实验中实际缓存约 13,000 项，峰值 RSS 只增加约 4.2 MiB。对当前 TinyStories tokenizer，这是一次收益很高的“用少量内存换大量时间”的优化。
