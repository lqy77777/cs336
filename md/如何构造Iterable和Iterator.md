# 如何构造 Iterable / Iterator

> 通用 Python 语法参考。所有例子都和作业无关,具体应用需要自己设计。

## 1. 定义回顾

- **Iterable**:有 `__iter__()` 方法的对象,可以被 `for x in obj:` 遍历
- **Iterator**:有 `__iter__()`(返回自己)**和** `__next__()`(产出下一个值,没有更多值时抛出 `StopIteration`)的对象
- 每个 Iterator 也是 Iterable(反过来不成立)——`list` 是 Iterable 但不是 Iterator

---

## 2. 构造 Iterable 的方式

### 2.1 直接用内置容器
`list`、`tuple`、`set`、`dict`、`str` 本身就是 Iterable,不需要额外构造:

```python
>>> for x in [1, 2, 3]:
...     print(x)
```

### 2.2 自定义类实现 `__iter__`

```python
class MyCollection:
    def __init__(self, data):
        self.data = data
    def __iter__(self):
        return iter(self.data)   # 返回一个 Iterator
```

---

## 3. 构造 Iterator 的方式

### 3.1 生成器函数(最常用、最推荐)

函数体内只要出现 `yield` 关键字,这个函数就变成**生成器函数**——调用它不会立即执行函数体,而是返回一个**生成器对象**,这个对象天然满足 Iterator 协议。

```python
def count_up_to(n):
    i = 1
    while i <= n:
        yield i
        i += 1

>>> gen = count_up_to(3)
>>> gen
<generator object count_up_to at 0x...>
>>> next(gen)
1
>>> next(gen)
2
>>> for x in count_up_to(3):   # 也可以直接用 for 遍历
...     print(x)
1
2
3
```

**`yield` 的行为**:执行到 `yield value` 时,函数**暂停**,把 `value` 交给调用者;下次调用 `next()` 时,从暂停的地方**继续**执行,而不是从头开始。这和 `return` 完全不同——`return` 会立刻结束函数并交出最终结果,`yield` 是"交出一个值,但保留现场,以后还能回来接着跑"。

### 3.2 生成器表达式

和列表推导式长得很像,但用圆括号,而且是**惰性**求值(不会一次性把结果都算出来):

```python
>>> squares = (x * x for x in range(5))
>>> squares
<generator object <genexpr> at 0x...>
>>> list(squares)
[0, 1, 4, 9, 16]
```

对比列表推导式 `[x*x for x in range(5)]`——那个会立即算出一个完整列表(Iterable,但不是 Iterator);圆括号版本是惰性的生成器(Iterator)。

### 3.3 `iter()` 内置函数

把一个 Iterable **转换**成 Iterator:

```python
>>> it = iter([1, 2, 3])
>>> next(it)
1
>>> next(it)
2
```

### 3.4 手动实现协议(自定义类)

不用 `yield`,自己写一个类,同时实现 `__iter__`(返回自己)和 `__next__`(每次调用产出下一个值,没有更多时 `raise StopIteration`):

```python
class CountUpTo:
    def __init__(self, n):
        self.n = n
        self.current = 1

    def __iter__(self):
        return self

    def __next__(self):
        if self.current > self.n:
            raise StopIteration
        value = self.current
        self.current += 1
        return value
```

这种写法比生成器函数啰嗦得多,通常只在需要额外功能(比如支持从多个独立起点重新遍历、暴露额外方法)时才用。**大多数场景,生成器函数(3.1)就够了**,不需要手写这一整套协议。

---

## 4. 几个容易踩的点

- **生成器只能遍历一次**:耗尽之后再 `next()` 会一直抛 `StopIteration`,想重新遍历要重新调用生成器函数,拿到一个**新的**生成器对象
- **函数里只要有一个 `yield`,整个函数就是生成器函数**——哪怕这个 `yield` 藏在很深的 `if` 分支里,也会让整个函数的调用方式变成"返回生成器",而不是"直接执行返回值"
- **生成器函数里也能用 `return`**:效果是提前结束生成、触发 `StopIteration`,但不会像普通函数那样把 `return` 后面的值正常返回给调用者(它会变成 `StopIteration` 异常的一个附加值,一般用不到)

---

## 5. 选择哪种方式的经验法则

| 场景 | 推荐方式 |
|------|----------|
| 简单的"边算边产出"逻辑 | 生成器函数(`yield`) |
| 从已有数据做一次简单变换 | 生成器表达式 |
| 已经有一个 Iterable,想要 Iterator | `iter()` |
| 需要复杂状态管理、多方法接口 | 手动实现 `__iter__`/`__next__` 的类 |

对于"输入某个来源、边处理边产出结果"这种典型场景,生成器函数几乎总是最简单直接的选择。
