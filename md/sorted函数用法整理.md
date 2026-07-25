# Python `sorted()` 函数用法整理

## 1. 基本用法

`sorted(iterable)` —— 返回一个**新的 list**,元素按默认规则升序排列。原对象**不受影响**。

```python
>>> sorted([3, 7, 2])
[2, 3, 7]
>>> sorted("hello")              # 对字符串迭代,逐字符排序,返回的是 list 不是 str
['e', 'h', 'l', 'l', 'o']
>>> sorted({"b": 1, "a": 2})     # 对 dict 迭代拿到的是 key
['a', 'b']
>>> sorted({3, 1, 2})            # set 无序,排序后得到有序 list
[1, 2, 3]
```

**要点**:不管传进去的是 list、str、set、dict 还是任何可迭代对象,`sorted` 的返回值**永远是 list**。

## 2. `sorted()` vs `list.sort()`

这是最容易混的一对:

| | `sorted(x)` | `x.sort()` |
|---|---|---|
| 适用对象 | 任何可迭代对象 | **只有 list** |
| 是否修改原对象 | 否 | 是(原地排序) |
| 返回值 | 新的 list | **`None`** |

```python
>>> nums = [3, 1, 2]
>>> sorted(nums)
[1, 2, 3]
>>> nums                    # 原 list 没变
[3, 1, 2]

>>> nums.sort()             # 原地改,返回 None
>>> nums
[1, 2, 3]

>>> result = nums.sort()    # 典型错误
>>> print(result)
None
```

**易错点**:`x = x.sort()` 会把 `x` 变成 `None`。想要新列表用 `sorted`,想省内存原地改用 `.sort()`,但别接返回值。

## 3. `reverse` 参数:降序

```python
>>> sorted([3, 7, 2], reverse=True)
[7, 3, 2]
>>> sorted(["apple", "kiwi", "fig"], reverse=True)
['kiwi', 'fig', 'apple']
```

## 4. `key` 参数:自定义排序标准

`sorted(iterable, key=func)` —— 排序时不直接比较元素本身,而是比较 `func(元素)` 的结果;**列表里放的仍然是原始元素**。

```python
>>> sorted(["kiwi", "fig", "banana"], key=len)
['fig', 'kiwi', 'banana']            # 按长度排,返回的还是原字符串
>>> sorted([-5, 3, -8], key=abs)
[3, -5, -8]                          # 按绝对值排,元素保持原样
>>> sorted(["Amy", "bob", "Cat"], key=str.lower)
['Amy', 'bob', 'Cat']                # 忽略大小写
```

`key` 函数对每个元素**只调用一次**(结果被缓存起来复用),所以即使 `key` 比较费时也不用担心重复计算。

## 5. `key` 返回元组:多级排序

`key` 返回元组时,排序会"先比第一位,相等再比第二位",因为元组比较本身就是逐位进行的。

```python
>>> people = [("Bob", 30), ("Amy", 25), ("Zoe", 30), ("Cat", 25)]
>>> sorted(people, key=lambda p: (p[1], p[0]))
[('Amy', 25), ('Cat', 25), ('Bob', 30), ('Zoe', 30)]
# 先按年龄升序,年龄相同时再按名字字典序
```

### 混合升降序

`reverse=True` 是**作用于整体**的,没法只让某一级反过来。两种解决办法:

**办法一:数值取负**(只对数字有效)

```python
>>> sorted(people, key=lambda p: (-p[1], p[0]))
[('Bob', 30), ('Zoe', 30), ('Amy', 25), ('Cat', 25)]
# 年龄降序,同龄时名字升序
```

**办法二:利用稳定性分两次排**(见第 6 节),对字符串等无法取负的类型也适用

```python
>>> tmp = sorted(people, key=lambda p: p[0])          # 先按次要标准排
>>> sorted(tmp, key=lambda p: p[1], reverse=True)     # 再按主要标准排
[('Bob', 30), ('Zoe', 30), ('Amy', 25), ('Cat', 25)]
```

## 6. 稳定性(stable sort)

Python 的排序是**稳定的**:`key` 值相等的元素,排序后**保持它们原来的相对先后顺序**。

```python
>>> data = [("b", 1), ("a", 2), ("c", 1)]
>>> sorted(data, key=lambda x: x[1])
[('b', 1), ('c', 1), ('a', 2)]       # ("b",1) 原本在 ("c",1) 前面,排完还是
```

这个性质是第 5 节"办法二"能成立的基础:**从最次要的标准开始排,一路排到最主要的标准**,前面的排序结果会被后面的排序保留下来。

底层算法是 **Timsort**,时间复杂度 `O(n log n)`,对部分有序的数据有额外优化。

## 7. 对 `dict` 排序

直接 `sorted(d)` 只对 key 排序。想按 value 或拿到键值对,需要显式处理:

```python
>>> counts = {"a": 3, "b": 7, "c": 5}
>>> sorted(counts)                                  # 只排 key
['a', 'b', 'c']
>>> sorted(counts, key=counts.get)                  # 按 value 排,结果仍是 key 的列表
['a', 'c', 'b']
>>> sorted(counts.items(), key=lambda kv: kv[1])    # 用 .items() 拿键值对
[('a', 3), ('c', 5), ('b', 7)]
>>> dict(sorted(counts.items(), key=lambda kv: kv[1], reverse=True))
{'b': 7, 'c': 5, 'a': 3}                            # 排完再转回 dict
```

## 8. 排序 `bytes` 和元组(BPE 场景相关)

`bytes` 之间按**字节值的字典序**比较,规则和字符串类似——先逐字节比,前缀更短的排在前面:

```python
>>> sorted([b"the", b"a", b"th"])
[b'a', b'th', b'the']
>>> b"Z" < b"a"                  # 大写字母的字节值小于小写
True
>>> sorted([b" t", b"t", b"T"])  # 空格(0x20)最小
[b' t', b'T', b't']
```

元组的比较是**逐位进行**的,所以 `bytes` 组成的 pair 可以直接排:

```python
>>> sorted([(b"t", b"h"), (b"t", b"a"), (b"a", b"z")])
[(b'a', b'z'), (b't', b'a'), (b't', b'h')]
# 先比第 0 位,相同时再比第 1 位
```

这正是 `train_bpe` 里用 `max(pairs, key=lambda x: (pairs[x], x))` 打破平局时依赖的性质——`x` 本身就是可比较的。

## 9. 本项目里的两个实例

**`cs336_basics/bpe.py` 第 150 行**

```python
sorted(special_tokens, key=len, reverse=True)
```

按长度**降序**排列 special tokens。原因:后面要把它们拼成正则的多选分支 `(A|B|C)`,而正则的 `|` 是"先到先得"而非"最长优先"。当一个 token 是另一个的前缀时(如 `<|endoftext|>` 和 `<|endoftext|><|endoftext|>`),必须把长的放前面,否则长的永远匹配不到。详见 [tests/test_tokenizer.py](../tests/test_tokenizer.py) 里的 `test_overlapping_special_tokens`。

**`cs336_basics/bpe.py` 第 53 行**

```python
sorted(set(chunk_boundaries))
```

`sorted(set(x))` 是一个常见组合技:`set` 去重,`sorted` 恢复顺序(因为 set 是无序的)。这里用来保证分块边界既不重复、又保持递增。

## 10. 补充:`operator` 模块

用 `lambda` 取下标/属性时,`operator` 里的现成函数更快也更清晰:

```python
>>> from operator import itemgetter, attrgetter
>>> sorted(people, key=itemgetter(1))            # 等价于 key=lambda p: p[1]
>>> sorted(people, key=itemgetter(1, 0))         # 等价于 key=lambda p: (p[1], p[0])
>>> sorted(objs, key=attrgetter("name"))         # 按对象的 .name 属性排
```

## 11. 常见陷阱

- **`x.sort()` 返回 `None`**——最高频的错误,别写 `x = x.sort()`
- **`sorted` 返回的永远是 list**——传进去 str 或 dict,出来的都是 list,不会保持原类型
- **`key` 的返回值只用于比较,不会替换元素**——排完列表里放的还是原始元素
- **`reverse=True` 作用于整体**——想要"一级降序、一级升序",用取负或分两次排(第 5 节)
- **元素之间必须可比较**——混合类型会报 `TypeError`:

  ```python
  >>> sorted([1, "a"])
  TypeError: '<' not supported between instances of 'str' and 'int'
  ```

- **别拿排序当"取最大/最小"用**——只要一个极值时,`max()` / `min()` 是 `O(n)`,排序是 `O(n log n)`;要前 k 个用 `heapq.nlargest()` / `nsmallest()`
- **`key=len` 不加括号**——传的是函数对象本身,写成 `key=len()` 会立刻报错
