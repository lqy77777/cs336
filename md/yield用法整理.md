# `yield` 用法整理

> 通用 Python 语法参考,配合 [如何构造Iterable和Iterator.md](如何构造Iterable和Iterator.md) 一起看。所有例子都和作业无关。

## 1. 基本用法:`yield` 让函数变成生成器

一个函数体内只要出现 `yield`,这个函数就变成了**生成器函数**——调用它不会执行函数体,而是立刻返回一个**生成器对象**;真正的执行,是在你对这个生成器调用 `next()`(或者用 `for` 遍历)时才逐步发生的。

```python
def simple_gen():
    print("开始")
    yield 1
    print("中间")
    yield 2
    print("结束")

>>> g = simple_gen()      # 这一行什么都不打印!函数体还没开始执行
>>> next(g)
开始
1
>>> next(g)
中间
2
>>> next(g)
结束
Traceback (most recent call last):
    ...
StopIteration
```

每次 `next()`,函数从**上次暂停的地方**继续跑,直到遇到下一个 `yield`(交出值、再次暂停),或者函数体自然结束(抛出 `StopIteration`)。

## 2. `yield` vs `return`

| | `return` | `yield` |
|---|---|---|
| 效果 | 立即结束函数,交出**一个**最终结果 | 暂停函数,交出**一个**值,以后还能回来接着跑 |
| 函数下次被调用 | 从头重新开始执行 | 不会"再调用"——继续用 `next()` 从暂停处往下走 |
| 一个函数里能有几个 | 通常一个(多个也行,但只有第一个执行到的生效) | 可以有任意多个,每个都会被依次触发 |

在生成器函数里也能写 `return`(不带值,或者 `return` 后面跟一个值),效果是**提前结束生成**,触发 `StopIteration`,不会像普通函数那样把值正常"返回"给调用者。

## 3. 循环里用 `yield`(最常见的模式)

这是 `yield` 最典型的使用场景——把"逐个产出一系列值"的逻辑写成一个循环:

```python
def evens_up_to(n):
    i = 0
    while i <= n:
        yield i
        i += 2

>>> list(evens_up_to(10))
[0, 2, 4, 6, 8, 10]
```

好处是:如果 `n` 非常大,这种写法**不需要一次性把所有结果存进一个列表**——每次只在内存里保留"当前算到哪了"这一点点状态,符合"常数内存"的要求。对比一下等价的"先建列表再返回"写法:

```python
def evens_up_to_list(n):
    result = []
    i = 0
    while i <= n:
        result.append(i)
        i += 2
    return result   # 一次性占用和 n 成正比的内存
```

两者用起来（`for x in ...`）看起来差不多,但内存特性完全不同。

## 4. `yield from`:把另一个可迭代对象的内容逐个转交出去

如果你在一个生成器函数里,想把**另一个**可迭代对象(列表、另一个生成器、别的函数调用结果……)里的每个元素依次交出去,可以用 `yield from`,而不用手写循环:

```python
def relay(iterable):
    yield from iterable

>>> list(relay([1, 2, 3]))
[1, 2, 3]
```

等价于手写:

```python
def relay(iterable):
    for x in iterable:
        yield x
```

`yield from` 只是这个常见模式的简写,但语义上更清楚地表达"我在转发别的东西的输出"。常见场景:一个生成器函数需要依次调用**多个**子任务、把每个子任务产出的所有结果依次交出去:

```python
def all_results(tasks):
    for task in tasks:
        yield from process(task)   # 假设 process(task) 返回一个列表或另一个生成器
```

## 5. 消费生成器的几种方式

```python
gen = evens_up_to(6)

for x in gen:            # 方式一:for 循环,最常见
    print(x)

next(gen)                 # 方式二:手动逐个取,拿到 StopIteration 说明耗尽了

list(gen)                 # 方式三:一次性耗尽,收集成列表(如果不在乎内存优势的话)
```

## 6. 常见陷阱

- **生成器只能遍历一次**。耗尽之后(`StopIteration` 触发过),再遍历它只会立刻结束,什么都不产出。想"重新来一遍",需要重新调用生成器函数,拿到一个**新的**生成器对象。
- **函数体在调用时不会立即执行**,只有第一次 `next()`(或者进入 `for` 循环)时才会跑到第一个 `yield`。如果函数开头有一些"提前检查参数合法性"的代码,这些代码**不会**在调用生成器函数的那一刻执行,而是延迟到第一次真正取值时——如果需要立即校验,这是个容易被忽略的坑。
- **调试时容易忘记"暂停"这件事**:生成器函数里 `yield` 之后的代码,要等**下一次** `next()` 才会执行,不是紧接着当前这次调用就跑完——不熟悉这个执行模型时,容易对执行顺序产生误解,建议像本文件开头那个 `print` 例子一样,自己动手跑一遍观察执行顺序。
