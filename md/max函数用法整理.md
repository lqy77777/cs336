# Python `max()` 函数用法整理

## 1. 基本用法:单个可迭代对象

`max(iterable)` —— 返回其中最大的元素,按默认比较规则(数字比大小,字符串/元组按字典序)。

```python
>>> max([3, 7, 2])
7
>>> max("hello")          # 对字符串迭代,逐字符比较
'o'
>>> max(["banana", "apple", "cherry"])
'cherry'
>>> max([(1, 2), (1, 5), (0, 9)])   # 元组按字典序,逐位比较
(1, 5)
```

## 2. 可变参数形式:多个独立的值

`max(a, b, c, ...)` —— 不传一个可迭代对象,而是直接传多个独立参数,返回其中最大的一个。

```python
>>> max(3, 7, 2)
7
>>> max("a", "z", "m")
'z'
>>> max((1, 2), (1, 3))    # 两个元组作为两个独立参数比较
(1, 3)
```

## 3. `key` 参数:自定义比较标准

`max(iterable, key=func)` —— 比较时不直接看元素本身,而看 `func(元素)` 的结果;**返回值仍是原始元素**,不是 `func` 算出来的值。

```python
>>> max(["kiwi", "fig", "banana"], key=len)
'banana'                     # 按长度比较,但返回的是原字符串
>>> max([-5, 3, -8], key=abs)
-8                           # 按绝对值比较,返回原始的 -8
>>> max(["Amy", "bob", "Cat"], key=str.lower)
'Cat'                        # 忽略大小写比较
```

## 4. `key` 返回元组:多级排序 / 打破平局

`key` 函数可以返回一个元组,实现"先比第一个标准,相等时再比第二个标准"——因为元组比较本身就是逐位进行的。

```python
>>> people = [("Bob", 30), ("Amy", 30), ("Zoe", 25)]
>>> max(people, key=lambda p: (p[1], p[0]))
('Bob', 30)                  # 年龄(p[1])优先比较,30 并列时再比名字(p[0])

>>> products = [("apple", 3), ("banana", 3), ("kiwi", 1)]
>>> max(products, key=lambda p: (p[1], p[0]))
('banana', 3)                # 数量并列时,按名字字典序选更大的
```

## 5. 对 `dict` 使用 `max`

直接迭代一个 `dict` 拿到的是**它的 key**,不是 value,也不是 `(key, value)` 对。

```python
>>> counts = {"a": 3, "b": 7, "c": 5}
>>> max(counts)                       # 不加 key:只比较 key 本身(这里是字符串)
'c'
>>> max(counts, key=counts.get)       # 加 key:按 value 大小选 key
'b'
>>> max(counts.items(), key=lambda kv: kv[1])   # 用 .items() 拿 (key, value) 对
('b', 7)                              # 这时返回值是整个 (key, value) 元组
```

**易错点**:不加 `key` 时,`max(counts)` 只会按 key 本身排序,和 value 毫无关系——想按 value 排序,必须显式传 `key`。

## 6. `default` 参数:处理空序列

不加 `default` 时,对空的可迭代对象调用 `max` 会报错;加上 `default` 可以避免。

```python
>>> max([])
Traceback (most recent call last):
    ...
ValueError: max() iterable argument is empty
>>> max([], default=0)
0
>>> max([], default=None)
```

## 7. 常见陷阱

- **并列时默认返回"最先遍历到的那个"**,不会自动按元素本身排序去打破平局——需要用第 4 节的元组 `key` 技巧显式指定平局规则
- **`key` 函数的返回值只用来比较,不是最终返回值**——`max` 返回的永远是原始可迭代对象里的某个元素
- **迭代 `dict` 默认只拿到 key**,想要 value 参与比较,要么用 `key=dict.get` 这类间接查值的方式,要么改用 `.items()` 直接迭代 `(key, value)` 对
- `max(a, b, key=...)`(第 2 节 + 第 3 节组合)同样合法:两个独立参数各自算一次 `key`,比较结果
