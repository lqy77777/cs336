# `einsum` / `einops` / `einx` 介绍

> 通用 API 参考,配合 [PyTorch_Tensor基础.md](PyTorch_Tensor基础.md) 使用。所有例子都是自己构造的通用例子,和作业无关。

## 为什么需要这些工具(简述)

普通写法(`view`/`reshape`/`transpose`/`@`)要处理"批量维在哪、维度顺序对不对"这类问题,经常需要好几步操作,而且从代码本身很难看出"每个维度到底代表什么"。`einsum` 系工具让你**直接在表达式里给每个维度起名字**,代码本身就是文档——写错维度顺序、忘了转置这类 bug 会大幅减少。详细动机见 [第3节_Transformer架构知识整理.md](第3节_Transformer架构知识整理.md) 3.2 节。

三者的关系:`torch.einsum` 是 PyTorch 原生的、最基础的版本;`einops` 在此基础上提供了更好读的具名坐标轴记号,外加 `rearrange`/`reduce`/`repeat` 等配套操作;`einx` 是比 `einops` 更通用、更激进的下一代库,能把"重排 + 归约 + 张量收缩"揉进一次调用。

---

## 1. `torch.einsum`

### 1.1 记号规则

`torch.einsum("下标模式", 输入张量们)`。下标模式的语法:

- 每个输入张量对应一组字母,每个字母代表一个维度
- 输出用 `->` 隔开,写明结果保留哪些字母(维度)
- **同一个字母在多个输入里重复出现** → 沿这一维做"逐元素相乘再求和"(这是矩阵乘法的本质)
- **某个字母没有出现在输出里** → 这一维会被求和掉(归约消失)
- **字母只出现在一个输入、也出现在输出里** → 这一维原样保留

### 1.2 常见例子

**向量点积**:
```python
>>> import torch
>>> a = torch.tensor([1.0, 2.0, 3.0])
>>> b = torch.tensor([4.0, 5.0, 6.0])
>>> torch.einsum("i,i->", a, b)
tensor(32.)     # 1*4 + 2*5 + 3*6,i 在两个输入里重复、且不出现在输出 → 求和归约
```

**矩阵乘法**:
```python
>>> A = torch.randn(3, 4)
>>> B = torch.randn(4, 5)
>>> torch.einsum("ij,jk->ik", A, B).shape
torch.Size([3, 5])     # j 重复出现、不在输出里 → 被收缩掉,等价于 A @ B
```

**批量矩阵乘法**(多一个 batch 维,原样保留):
```python
>>> A = torch.randn(8, 3, 4)
>>> B = torch.randn(8, 4, 5)
>>> torch.einsum("bij,bjk->bik", A, B).shape
torch.Size([8, 3, 5])   # b 在两个输入和输出里都出现 → 原样保留,不参与收缩
```

**转置**(只重排字母顺序,不做任何求和):
```python
>>> A = torch.randn(3, 4)
>>> torch.einsum("ij->ji", A).shape
torch.Size([4, 3])
```

**沿某一维求和**:
```python
>>> A = torch.randn(3, 4)
>>> torch.einsum("ij->i", A).shape       # 相当于 A.sum(dim=1)
torch.Size([3])
```

---

## 2. `einops`

`einops` 不用单字母,而是用**有意义的名字**做坐标轴标记,可读性更高。核心提供四个函数:`rearrange`、`einsum`、`reduce`、`repeat`。

### 2.1 `rearrange`:重排/合并/拆分维度

```python
from einops import rearrange

>>> x = torch.randn(2, 3, 4)        # 想象成 (batch, height, width)
>>> rearrange(x, "b h w -> b w h").shape          # 转置最后两维
torch.Size([2, 4, 3])

>>> rearrange(x, "b h w -> b (h w)").shape        # 合并两个维度
torch.Size([2, 12])

>>> y = torch.randn(2, 12)
>>> rearrange(y, "b (h w) -> b h w", h=3).shape   # 拆分成两个维度(需要指明其中一个的大小)
torch.Size([2, 3, 4])
```

`(h w)` 这种括号语法表示"把 `h` 和 `w` 这两个维度合并成一个",拆分时需要额外传参数(比如 `h=3`)告诉它怎么拆。

### 2.2 `einsum`(einops 版本):具名坐标轴

和 `torch.einsum` 语义一样,但用单词代替单字母,可读性大幅提升:

```python
from einops import einsum

>>> A = torch.randn(3, 4)
>>> B = torch.randn(4, 5)
>>> einsum(A, B, "d_in d_hidden, d_hidden d_out -> d_in d_out").shape
torch.Size([3, 5])
```

对比 `torch.einsum("ij,jk->ik", A, B)`——功能完全一样,但读代码的人不用去猜 `i`、`j`、`k` 分别指什么。

### 2.3 `reduce`:带命名坐标轴的归约

```python
from einops import reduce

>>> x = torch.randn(2, 3, 4)
>>> reduce(x, "b h w -> b h", "sum").shape     # 沿 w 求和
torch.Size([2, 3])
>>> reduce(x, "b h w -> b h", "mean").shape    # 沿 w 求均值
torch.Size([2, 3])
```

第三个参数是归约方式的字符串(`"sum"`、`"mean"`、`"max"` 等)。

### 2.4 `repeat`:广播式复制/扩展维度

```python
from einops import repeat

>>> x = torch.randn(3)
>>> repeat(x, "d -> b d", b=5).shape     # 复制成 5 份,新增一个 batch 维
torch.Size([5, 3])
```

用于"我有一个不带 batch 维的张量,想把它扩展到和某个 batch 一致"这种场景。

---

## 3. `einx`

`einx` 是比 `einops` 更通用的下一代库(讲义提醒:没有 `einops` 那么久经考验,遇到问题可以退回 `einops` 或原生 PyTorch)。核心区别:`einx` 能把"重排 + 归约 + 张量收缩"**揉进一次表达式**,不需要像 `einops` 那样分成 `rearrange` + `einsum` 两步。

### 3.1 `einx.dot`:einsum 的等价物

```python
import einx

>>> A = torch.randn(3, 4)
>>> B = torch.randn(4, 5)
>>> einx.dot("i j, j k -> i k", A, B).shape
torch.Size([3, 5])
```

### 3.2 一次表达式同时做重排和收缩(einops 做不到这么简洁的地方)

假设有一个形状 `(batch, height, width, channel)` 的张量,想对**每个 channel 独立**、在展平后的像素维度上做一次线性变换(权重形状 `(height*width, height*width)`)。用 `einops` 需要先 `rearrange` 展平/换轴,再 `einsum`,再 `rearrange` 换回来——三步。用 `einx.dot`,可以直接在一个表达式里同时表达"输入的四维怎么理解"和"矩阵乘法怎么做":

```python
>>> x = torch.randn(2, 32, 32, 3)     # (batch, row, col, channel)
>>> B = torch.randn(32*32, 32*32)
>>> out = einx.dot(
...     "batch row_in col_in channel, (row_out col_out) (row_in col_in) "
...     "-> batch row_out col_out channel",
...     x, B, row_in=32, col_in=32
... )
>>> out.shape
torch.Size([2, 32, 32, 3])
```

这一步"合并 `(row col)`、做矩阵乘法、再拆回 `(row col)`"在 `einops` 里需要三次调用,`einx` 一次表达式就写完了。

---

## 4. 三者怎么选

| | 记号方式 | 特点 |
|---|---------|------|
| `torch.einsum` | 单字母 | PyTorch 原生,不需要额外依赖,但字母不如单词好读 |
| `einops` | 具名单词 | 可读性好,`rearrange`/`reduce`/`repeat` 覆盖了大多数日常需求,足够成熟稳定 |
| `einx` | 具名单词,表达力更强 | 能把多步操作揉进一次调用,更简洁,但相对新、坑可能更多 |

讲义的建议:**没接触过 einsum 记号的,先学 `einops`**(先读它的文档);已经熟悉 `einops` 的,可以进一步学更通用的 `einx`。日常写 `Linear`、attention 这类模块时,`einops` 的 `rearrange` + `einsum` 组合基本够用。
