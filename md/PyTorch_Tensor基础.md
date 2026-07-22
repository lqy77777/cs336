# PyTorch Tensor 基础

> 通用 PyTorch API 参考,配合第 3 节(Transformer 架构)的实现使用。所有例子都和作业无关。

## 1. 什么是 Tensor

`torch.Tensor` 是 PyTorch 的核心数据结构——一个多维数组,行为上和 NumPy 的 `ndarray` 很像,但多了两个关键能力:

- **可以放在 GPU 上运算**(`.to("cuda")`)
- **支持自动求导**(`requires_grad=True` 时,PyTorch 会自动记录计算图,用于反向传播)

```python
>>> import torch
>>> t = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
>>> t
tensor([[1., 2.],
        [3., 4.]])
```

## 2. 创建 Tensor

```python
>>> torch.tensor([1, 2, 3])          # 从 Python 列表/嵌套列表构造
>>> torch.zeros(2, 3)                 # 全 0,形状 (2, 3)
>>> torch.ones(2, 3)                  # 全 1
>>> torch.randn(2, 3)                 # 标准正态分布随机数
>>> torch.empty(2, 3)                 # 未初始化(值不确定,只是分配内存)
>>> torch.arange(5)                   # tensor([0, 1, 2, 3, 4])
```

## 3. 核心属性:`shape` / `dtype` / `device`

```python
>>> t = torch.randn(2, 3, 4)
>>> t.shape           # torch.Size([2, 3, 4]),也可以用 t.size()
>>> t.dtype           # torch.float32(默认浮点类型)
>>> t.device           # device(type='cpu')
>>> t.ndim             # 3(维度数量)
```

**`dtype`**:数据类型,常见的有 `torch.float32`、`torch.float64`、`torch.int64`(整数,`torch.long` 是它的别名)、`torch.bool`。

**`device`**:数据存放在哪块硬件上,`"cpu"` 或 `"cuda"`(GPU)。**两个 tensor 要做运算,必须在同一个 device 上**,否则会报错。

```python
>>> a = torch.randn(3)
>>> b = a.to(torch.float64)     # 转换 dtype,返回新 tensor
>>> c = a.to("cuda")            # 转移到 GPU(如果可用),返回新 tensor
```

这也是为什么很多模块的 `__init__` 会接受 `device=None, dtype=None` 这两个参数——在构造参数(权重)时就指定好它们存放在哪、精度是多少。

## 4. 索引与切片

和 Python 列表 / NumPy 的用法基本一致:

```python
>>> t = torch.arange(12).reshape(3, 4)
>>> t
tensor([[ 0,  1,  2,  3],
        [ 4,  5,  6,  7],
        [ 8,  9, 10, 11]])
>>> t[0]              # 第 0 行: tensor([0, 1, 2, 3])
>>> t[:, 1]            # 第 1 列: tensor([1, 5, 9])
>>> t[0, 2]             # 单个元素: tensor(2)
>>> t[1:, :2]           # 切片: 第 1 行开始的所有行,前 2 列
```

**用另一个整数 tensor 做索引**("花式索引",Embedding 查表的核心机制):

```python
>>> table = torch.tensor([10.0, 20.0, 30.0, 40.0])
>>> ids = torch.tensor([2, 0, 3])
>>> table[ids]
tensor([30., 10., 40.])
```

## 5. 形状变换

### `view` / `reshape`:改变形状,元素总数不变

```python
>>> t = torch.arange(6)
>>> t.view(2, 3)
tensor([[0, 1, 2],
        [3, 4, 5]])
```

`view` 要求底层内存是"连续的"(contiguous),否则会报错;`reshape` 更宽松,必要时会自动拷贝数据。不确定用哪个时,`reshape` 更安全。

### `transpose` / `permute`:交换维度顺序

```python
>>> t = torch.randn(2, 3, 4)
>>> t.transpose(0, 1).shape     # 交换第 0、1 维: torch.Size([3, 2, 4])
>>> t.permute(2, 0, 1).shape    # 按给定顺序重排所有维: torch.Size([4, 2, 3])
```

### `squeeze` / `unsqueeze`:增删长度为 1 的维度

```python
>>> t = torch.randn(3, 1, 4)
>>> t.squeeze(1).shape        # 去掉第 1 维(长度是 1): torch.Size([3, 4])
>>> t2 = torch.randn(3, 4)
>>> t2.unsqueeze(0).shape     # 在第 0 维插入一个长度 1 的维: torch.Size([1, 3, 4])
```

`unsqueeze` 常用于给张量"添加一个批量维",方便和另一个更高维的张量做广播运算。

## 6. 广播(Broadcasting)

两个形状不同的 tensor 做逐元素运算时,PyTorch 会尝试"广播"——从**最后一维**开始比较,维度相等、或其中一个是 1、或缺失,就可以自动扩展匹配:

```python
>>> a = torch.randn(3, 4)
>>> b = torch.randn(4)        # 形状 (4,),自动广播成 (3, 4)
>>> (a + b).shape
torch.Size([3, 4])

>>> c = torch.randn(3, 1)
>>> (a + c).shape             # (3,4) 和 (3,1) 广播成 (3,4)
torch.Size([3, 4])
```

这是"批量维"能够优雅处理的关键机制——不需要手动复制数据,PyTorch 在计算时"假装"扩展过了。

## 7. 逐元素运算 vs 矩阵运算

**逐元素(elementwise)**:`+`、`-`、`*`、`/` 对两个形状相同(或可广播)的 tensor,是**逐个位置**分别运算:

```python
>>> a = torch.tensor([1.0, 2.0, 3.0])
>>> b = torch.tensor([10.0, 20.0, 30.0])
>>> a * b
tensor([10., 40., 90.])      # 不是矩阵乘法,是对应位置相乘
```

**矩阵乘法**:用 `@` 运算符或 `torch.matmul`,**不是** `*`:

```python
>>> A = torch.randn(3, 4)
>>> B = torch.randn(4, 5)
>>> (A @ B).shape             # torch.Size([3, 5])
```

`@` 对**批量矩阵乘法**也生效——如果 `A` 形状是 `(batch, m, n)`,`B` 是 `(batch, n, p)`,`A @ B` 会对每个 batch 独立做矩阵乘法,输出 `(batch, m, p)`,不需要写循环。

## 8. 沿指定维度做归约

`sum`、`mean`、`max` 等归约操作,可以指定 `dim` 参数,只沿着**这一维**归约,其他维保留:

```python
>>> t = torch.randn(2, 3, 4)
>>> t.sum(dim=-1).shape        # 沿最后一维求和: torch.Size([2, 3])
>>> t.mean(dim=-1).shape       # 沿最后一维求均值: torch.Size([2, 3])
>>> t.max(dim=-1)               # 返回 (最大值, 对应下标) 两部分
```

`dim=-1` 表示"最后一维"(比写死具体数字更灵活,不用管前面有多少个批量维)。

**`keepdim` 参数**:归约后是否保留被归约的那一维(变成长度 1),而不是直接消失:

```python
>>> t.sum(dim=-1).shape                  # torch.Size([2, 3])
>>> t.sum(dim=-1, keepdim=True).shape    # torch.Size([2, 3, 1])
```

`keepdim=True` 常用于:归约完的结果还要和原张量做广播运算(比如"每个元素除以它所在那一维的和")——保留这一维,广播才能自动对齐。

## 9. `nn.Parameter` 和 `nn.Module`

`nn.Module` 是所有神经网络层的基类(自定义层需要继承它)。`nn.Parameter` 是 `Tensor` 的一个子类,标记"这个张量是需要被训练的参数"——包在 `nn.Parameter` 里的张量,会自动被 `nn.Module` 记录,出现在 `.parameters()` 里,也会自动参与梯度计算:

```python
import torch.nn as nn

class MyLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(dim))

    def forward(self, x):
        return x * self.weight
```

要点:
- 必须调用 `super().__init__()`(基类构造函数),否则参数注册机制不生效
- `forward` 方法定义"输入怎么变成输出",调用层实例(`layer(x)`)时会自动触发 `forward`

## 10. `register_buffer`:非训练状态

有些张量需要跟着模块走(存、读、跟 `.to(device)` 一起移动),但**不是**可训练参数(比如固定的常量、预先算好的查找表)。这种情况用 `register_buffer`:

```python
class MyLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.register_buffer("scale", torch.ones(dim), persistent=False)
```

`persistent=False` 表示这个 buffer **不需要**被保存进模型的 checkpoint(通常用于"可以随时重新计算出来"的缓存值,没必要占存档空间)。

## 11. 常见陷阱

- **`view` 要求内存连续**:对做过 `transpose`/`permute` 的张量直接 `.view(...)` 经常报错(`RuntimeError: view size is not compatible...`),需要先 `.contiguous()`,或者干脆用 `.reshape(...)` 代替
- **`*` 不是矩阵乘法**:矩阵乘法要用 `@` 或 `torch.matmul`,写错很容易得到形状"恰好能广播、但语义完全错误"的结果,不会报错但结果是错的
- **dtype 不匹配**:`float32` 的张量和 `float64` 的张量做运算可能报错或静默提升精度,需要显式 `.to(dtype)` 对齐
- **device 不匹配**:CPU 上的张量和 GPU 上的张量不能直接运算,报错信息通常很明确(`Expected all tensors to be on the same device`)
- **原地(in-place)操作**:方法名带下划线后缀的(比如 `.add_()`、`.mul_()`)是**原地修改**,不返回新张量;不带下划线的(`.add()`、`.mul()`)返回新张量,原张量不变——训练时对需要梯度的张量做原地操作容易破坏计算图,不确定的话优先用不带下划线的版本
