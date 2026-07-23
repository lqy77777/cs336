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

### `torch.arange`:生成等差整数/浮点数序列

和 Python 内置的 `range()` 用法很像,但返回的是 tensor:

```python
>>> torch.arange(5)              # 只给终点(不含):0 到 4
tensor([0, 1, 2, 3, 4])
>>> torch.arange(2, 8)           # 起点、终点(不含)
tensor([2, 3, 4, 5, 6, 7])
>>> torch.arange(0, 10, 2)       # 起点、终点(不含)、步长
tensor([0, 2, 4, 6, 8])
>>> torch.arange(5, dtype=torch.float32)   # 指定 dtype
tensor([0., 1., 2., 3., 4.])
```

三个参数分别是 `start`(默认 0)、`end`(必填,**不包含**这个值本身)、`step`(默认 1)——和 Python 的 `range(start, end, step)` 规则完全一致,同样支持 `device`、`dtype` 关键字参数。这是生成"位置序列"(比如 `0, 1, ..., seq_len-1`)、"下标序列"最常用的方式。

**dtype 自动推断**:不显式指定 `dtype` 时,参数**全是整数** → 返回整数 tensor(`int64`);**任意一个参数是浮点数** → 返回浮点 tensor(`float32`):

```python
>>> torch.arange(1, 4).dtype          # 全整数参数
torch.int64
>>> torch.arange(1, 7/2 + 1)          # 7/2 是浮点除法,端点变成浮点数
tensor([1., 2., 3., 4.])              # 结果自动是 float32
```

注意 Python 的 `/` 永远是浮点除法(哪怕两边都是整数,`6/2` 也是 `3.0` 不是 `3`)——所以端点表达式里只要出现过 `/`,`arange` 的结果就是浮点 tensor。这个推断有时"恰好帮了忙"(后续要做浮点运算时不用再转),但依赖它不如显式写 `dtype=` 清楚。

### `torch.linspace`:在区间内生成指定数量的等间隔点

和 `arange` 不同,`linspace` **不是**按步长生成,而是"给定区间两端和点数,自动算出间隔":

```python
>>> torch.linspace(0, 1, steps=5)
tensor([0.0000, 0.2500, 0.5000, 0.7500, 1.0000])
```

`arange` 关心"步长是多少",`linspace` 关心"一共要几个点"——**两端(`start`、`end`)都会被包含**在结果里(这点和 `arange` 的"不包含终点"不一样),需要哪种取决于你是想控制间隔大小,还是控制点的数量。

### `torch.full`:生成指定形状、值全部相同的 tensor

```python
>>> torch.full((2, 3), 7.0)
tensor([[7., 7., 7.],
        [7., 7., 7.]])
```

比 `torch.zeros`/`torch.ones` 更通用——可以填任意常数,不局限于 0 或 1。

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

索引 tensor 可以是**任意形状**(不只是一维)——结果的形状 = 索引 tensor 的形状 + 被索引 tensor 除第一维外剩下的形状。比如用形状 `(batch, seq_len)` 的 `ids` 去索引形状 `(vocab, d)` 的表,结果是 `(batch, seq_len, d)`。顺带一提:**einops 没有"索引"这类操作**——`rearrange`/`einsum`/`reduce` 只做重排和归约,查表这件事只能用原生索引。

### 步长切片与 `stop` 的"不包含"规则

切片完整语法是 `[start:stop:step]`,三段都可省略:

```python
>>> t = torch.arange(10)
>>> t[0::2]        # 从 0 开始每隔 2 取一个(偶数下标)
tensor([0, 2, 4, 6, 8])
>>> t[1::2]        # 奇数下标
tensor([1, 3, 5, 7, 9])
>>> t[:-1]          # stop=-1:切到最后一个元素之前,不包含它
tensor([0, 1, 2, 3, 4, 5, 6, 7, 8])
```

**`stop` 永远不包含**,负数下标也一样——`-1` 指"最后一个元素的位置",写在 `stop` 上就是"排除最后一个"。另外注意:如果一个维度长度是偶数,`[0::2]` 和 `[0:-1:2]` 结果完全相同(最后一个下标是奇数,本来就取不到),后者的 `-1` 是多余的。

### 省略号 `...`:保留任意数量的前置维度

`...`(Ellipsis)在索引里代表"前面所有维度原样保留",只对写出来的那几维做操作:

```python
>>> t = torch.randn(2, 3, 8)
>>> t[..., 0::2].shape        # 只切最后一维,前面维度全部保留
torch.Size([2, 3, 4])
```

**一个高危陷阱**:`t[-1]` 和 `t[..., -1]` 完全不同——前者是对**第一维**取下标(把整个第一维砍掉只剩最后一份,批量数据直接丢失),后者才是对**最后一维**取下标。同理 `t[-1][0::2]` 和 `t[..., 0::2]` 是两码事:想"保留所有批量维、只操作最后一维",必须用 `...` 开头的写法。

**einops 对比(处理"相邻两两配对"场景)**:除了步长切片,还可以用 `rearrange` 把配对结构显式暴露成一个维度:

```python
>>> from einops import rearrange
>>> pairs = rearrange(t, '... (n two) -> ... n two', two=2)   # (2,3,8) -> (2,3,4,2)
>>> pairs[..., 0]        # 每对的第一个成员,等价于 t[..., 0::2]
>>> pairs[..., 1]        # 每对的第二个成员,等价于 t[..., 1::2]
```

两种写法结果一致——步长切片更短,`rearrange` 把"最后一维是 4 组、每组 2 个"的意图写得更显式,读代码时不用心算步长语义。

## 5. 形状变换

### `view` / `reshape`:改变形状,元素总数不变

```python
>>> t = torch.arange(6)
>>> t.view(2, 3)
tensor([[0, 1, 2],
        [3, 4, 5]])
```

`view` 要求底层内存是"连续的"(contiguous),否则会报错;`reshape` 更宽松,必要时会自动拷贝数据。不确定用哪个时,`reshape` 更安全。

### `transpose` / `permute` / `.T`:交换维度顺序

```python
>>> t = torch.randn(2, 3, 4)
>>> t.transpose(0, 1).shape     # 交换第 0、1 维: torch.Size([3, 2, 4])
>>> t.permute(2, 0, 1).shape    # 按给定顺序重排所有维: torch.Size([4, 2, 3])
```

**`.T` 属性**:二维矩阵转置的简写:

```python
>>> A = torch.randn(3, 4)
>>> A.T.shape                    # torch.Size([4, 3]),等价于 A.transpose(0, 1)
```

注意:`.T` 只应该用在**二维**张量上——对三维及以上的张量,`.T` 的语义是"反转所有维度顺序"(且新版 PyTorch 已对这种用法发出弃用警告),通常不是你想要的。高维张量想"只转置最后两维"(比如批量矩阵转置),用 **`.mT`**(matrix transpose)或 `transpose(-2, -1)`。线性层权重是二维的,`weight.T` 安全;attention 里对带批量维的 K 做转置时,要用 `.mT` 或 `transpose(-2, -1)`,不能用 `.T`。

### `squeeze` / `unsqueeze`:增删长度为 1 的维度

```python
>>> t = torch.randn(3, 1, 4)
>>> t.squeeze(1).shape        # 去掉第 1 维(长度是 1): torch.Size([3, 4])
>>> t2 = torch.randn(3, 4)
>>> t2.unsqueeze(0).shape     # 在第 0 维插入一个长度 1 的维: torch.Size([1, 3, 4])
```

`unsqueeze` 常用于给张量"添加一个批量维",方便和另一个更高维的张量做广播运算。

### `stack` / `cat`:把多个 tensor 合并成一个

**`torch.stack`**:沿着一个**新增的**维度,把多个**形状完全相同**的 tensor 堆起来:

```python
>>> a = torch.tensor([1, 2, 3])
>>> b = torch.tensor([4, 5, 6])
>>> torch.stack([a, b], dim=0).shape     # 在最前面新增一维: torch.Size([2, 3])
>>> torch.stack([a, b], dim=-1).shape    # 在最后面新增一维: torch.Size([3, 2])
```

`dim` 参数决定新维度插在哪个位置——`dim=-1` 时,`[a_i, b_i]` 会被放在一起,常用于"把两个对应位置的元素两两配对"这种场景。

**`torch.cat`**:沿着一个**已经存在的**维度做拼接,**不产生新维度**,要求除了拼接的那一维,其余维度都要一致:

```python
>>> a = torch.randn(2, 3)
>>> b = torch.randn(2, 3)
>>> torch.cat([a, b], dim=0).shape     # 沿第 0 维拼接: torch.Size([4, 3])
>>> torch.cat([a, b], dim=1).shape     # 沿第 1 维拼接: torch.Size([2, 6])
```

`stack` 和 `cat` 最容易混淆的地方:`stack` 会**增加一个维度**(把两个 `(3,)` 的向量堆成 `(2,3)`),`cat` 维度数量**不变**(把两个 `(2,3)` 拼成 `(4,3)` 或 `(2,6)`)。

### `flatten`:把多个维度合并成一个

```python
>>> t = torch.randn(2, 3, 4)
>>> t.flatten(start_dim=1).shape     # 从第 1 维开始,把后面的维度全部合并: torch.Size([2, 12])
```

和 `reshape(...)` 效果类似,但 `flatten(start_dim=...)` 更直接地表达"把从某一维开始的所有维度压平"这个意图。

### 组合技巧:用 `stack` + 合并维度,实现"交替排列"

把两个形状 `(..., n)` 的 tensor 交替穿插成一个 `(..., 2n)` 的 tensor(比如 `a_0, b_0, a_1, b_1, ...`),标准做法是:

```python
>>> a = torch.tensor([1, 3, 5])
>>> b = torch.tensor([2, 4, 6])
>>> torch.stack([a, b], dim=-1).reshape(-1)
tensor([1, 2, 3, 4, 5, 6])
```

先用 `stack(..., dim=-1)` 把每一对 `[a_i, b_i]` 紧挨着放在一起(形状变成 `(n, 2)`),再把最后两维合并成一维——因为 `stack` 之后每一对元素在内存里本来就是相邻的,合并出来的结果自然就是交替排列,不需要手写循环去交叉插入。

**三种"合并最后两维"的等价写法对比**(带批量维时的差异是重点):

```python
>>> stacked = torch.stack([a, b], dim=-1)       # 形状 (..., n, 2)

# 写法一:rearrange,模式里必须用 ... 涵盖前置批量维
>>> rearrange(stacked, '... n two -> ... (n two)')

# 写法二:flatten,从倒数第二维开始压平,天然适配任意维度数
>>> stacked.flatten(start_dim=-2)

# 写法三:reshape(只适合确切知道目标形状时)
>>> stacked.reshape(*stacked.shape[:-2], -1)
```

**陷阱**:`rearrange(stacked, 'n two -> (n two)')` 这种**不带 `...`** 的模式,要求输入**恰好二维**——只要 `stacked` 前面还有批量维(三维及以上),就会报 `EinopsError: Wrong shape: expected 2 dims`。einops 的模式字符串声明了几个轴,输入就必须恰好是几维(`...` 是唯一的"任意维数"通配)。`flatten(start_dim=-2)` 没有这个问题,不管前面几维都能用。

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

**一维向量在 `@` 里的特殊规则**:一维 tensor 本身没有"行/列"身份,但出现在 `@` **左侧**时被临时当作行向量、**右侧**时当作列向量,算完自动去掉临时补的那一维——这也是"`(in,)`、`(batch, in)`、`(batch, seq, in)` 三种输入都能被同一行 `x @ W.T` 正确处理"的原因。

**einsum 对比**:

```python
# 二维矩阵乘法
A @ B                                                  # 原生
einsum(A, B, 'i j, j k -> i k')                        # einops 等价

# 线性层模式:x 带任意批量维,权重 W 形状 (d_out, d_in)
x @ W.T                                                 # 原生:需要手动转置
einsum(x, W, '... d_in, d_out d_in -> ... d_out')       # einops:靠轴名自动对齐,不用转置
```

einsum 版本的优势:**不需要记"该转置谁"**——两个输入里同名的轴(`d_in`)自动被收缩,剩下的轴按输出模式排列,"列向量数学记号 vs 行向量代码惯例"那笔转置账它替你算了。简单场景用 `@` 更短;容易转置错、批量维多的场景 einsum 更稳。

## 8. 沿指定维度做归约

`sum`、`mean`、`max` 等归约操作,可以指定 `dim` 参数,只沿着**这一维**归约,其他维保留:

```python
>>> t = torch.randn(2, 3, 4)
>>> t.sum(dim=-1).shape        # 沿最后一维求和: torch.Size([2, 3])
>>> t.mean(dim=-1).shape       # 沿最后一维求均值: torch.Size([2, 3])
```

`dim=-1` 表示"最后一维"(比写死具体数字更灵活,不用管前面有多少个批量维)。

### `max`/`min` 带 `dim` 时返回的是"二元组",不是单个 tensor

```python
>>> t = torch.tensor([[3., 9., 1.],
...                   [7., 2., 8.]])
>>> result = t.max(dim=-1)
>>> result
torch.return_types.max(
values=tensor([9., 8.]),
indices=tensor([1, 2]))
>>> result.values       # 最大值本身——等价写法: result[0]
tensor([9., 8.])
>>> result.indices      # 最大值所在的下标——等价写法: result[1]
tensor([1, 2])
```

带 `dim` 的 `max`/`min` 返回**命名元组**(`values` + `indices` 两部分)。**直接拿整个返回值去参与运算会报错**(`TypeError: unsupported operand type(s) for -: 'Tensor' and 'torch.return_types.max'`)——必须先用 `.values` 或 `[0]` 取出数值那一半。

三个相关的辨析:

- **不带 `dim` 的 `torch.max(t)`** 返回的是**单个标量 tensor**(全局最大值,没有下标)——和带 `dim` 的版本返回类型完全不同,容易混淆
- 只想要下标,直接用 `torch.argmax(t, dim=...)` 更省事
- `sum`/`mean` 没有这个问题——它们不管带不带 `dim`,返回的都是单纯的 tensor

**einops 对比**:`reduce` 也能做沿轴归约,且**只返回数值、没有下标**,不存在"二元组"问题:

```python
>>> from einops import reduce
>>> reduce(t, '... d -> ...', 'max')            # 等价于 t.max(dim=-1).values
>>> reduce(t, '... d -> ... 1', 'max')          # 等价于 keepdim=True 的版本
>>> reduce(t, '... d -> ... 1', 'sum')          # sum/mean/min/prod 同理
```

模式右侧写 `1` 就是"保留这一维、长度变 1"(对应 `keepdim=True`);右侧不写就是这一维直接消失(对应 `keepdim=False`)——用轴名把归约意图写得更显式,不用记 `keepdim` 这个参数名。

**`keepdim` 参数**:归约后是否保留被归约的那一维(变成长度 1),而不是直接消失:

```python
>>> t.sum(dim=-1).shape                  # torch.Size([2, 3])
>>> t.sum(dim=-1, keepdim=True).shape    # torch.Size([2, 3, 1])
```

`keepdim=True` 常用于:归约完的结果还要和原张量做广播运算(比如"每个元素除以它所在那一维的和")——保留这一维,广播才能自动对齐。

## 9. 常用逐元素数学函数

这些函数对 tensor 里的**每个元素**独立做同一种数学运算,输入输出形状完全一致。

### `torch.sqrt`:开根号

```python
>>> t = torch.tensor([4.0, 9.0, 16.0])
>>> torch.sqrt(t)
tensor([2., 3., 4.])
>>> t.sqrt()          # 等价写法,方法调用
tensor([2., 3., 4.])
```

注意:输入必须是**非负**的浮点数,对负数求平方根会得到 `nan`(不会报错,但结果无意义)。

**`math.sqrt` vs `torch.sqrt`**:Python 标准库的 `math.sqrt` 只接受**单个 Python 数字**、返回 Python `float`——适合算标量超参数(比如初始化用的 σ,输入是普通数字不是 tensor);`torch.sqrt` 接受 tensor、逐元素开方。对 tensor 调 `math.sqrt` 会报错;算一个纯 Python 标量时用 `math.sqrt` 更直接,不用先包成 tensor 再拆出来。

### `**` 运算符 / `torch.pow`:幂运算

```python
>>> t = torch.tensor([1.0, 2.0, 3.0])
>>> t ** 2                    # 逐元素平方,最常见写法
tensor([1., 4., 9.])
>>> torch.pow(t, 2)           # 等价
tensor([1., 4., 9.])
```

**反过来,"标量为底、tensor 为指数"同样成立**:

```python
>>> base = 10.0
>>> exponents = torch.tensor([0.0, 1.0, 2.0])
>>> base ** exponents
tensor([  1.,  10., 100.])
```

底数固定、指数逐元素变化——构造几何级数(比如位置编码里 `Θ^{(2k-2)/d}` 这类"不同维度不同频率"的序列)就是这个模式,一行搞定,不需要循环。数学上 `a^x = e^{x·ln(a)}`,所以它也总能用 `torch.exp(x * math.log(a))` 改写,但直接用 `**` 更清楚。

### `torch.exp` / `torch.log`:指数、自然对数

```python
>>> t = torch.tensor([0.0, 1.0, 2.0])
>>> torch.exp(t)              # e^t,逐元素
tensor([1.0000, 2.7183, 7.3891])
>>> torch.log(torch.exp(t))   # log 是 exp 的逆运算
tensor([0., 1., 2.])
```

softmax、交叉熵这类计算会大量用到这两个函数。

### `torch.sigmoid`:S 形压缩函数

```python
>>> t = torch.tensor([-2.0, 0.0, 2.0])
>>> torch.sigmoid(t)          # σ(t) = 1 / (1 + e^(-t)),逐元素,输出落在 (0, 1) 区间
tensor([0.1192, 0.5000, 0.8808])
```

SiLU 激活(`x * sigmoid(x)`)、GLU 门控、二分类概率输出都以它为基础。直接调 `torch.sigmoid` 比自己用 `exp` 拼公式在数值上更稳定(内部对正负输入分别做了防溢出处理)。

### `torch.sin` / `torch.cos`:三角函数

```python
>>> angles = torch.tensor([0.0, 1.5708, 3.1416])
>>> torch.sin(angles)
tensor([ 0.0000e+00,  1.0000e+00, -7.3464e-06])
>>> torch.cos(angles)
tensor([ 1.0000e+00, -3.6200e-06, -1.0000e+00])
```

逐元素,输入按**弧度**(不是角度)。旋转类的几何变换(位置编码)会用到——注意区分两种角色:**对"角度表"求 `cos`/`sin` 得到旋转系数**,和**拿这些系数与数据做逐元素乘法**,是两步不同的操作;把 `torch.cos` 直接套在数据本身上(把数据当角度)是一类容易犯的概念错误。

### `torch.rsqrt`:平方根的倒数(顺带一提)

```python
>>> t = torch.tensor([4.0, 16.0])
>>> torch.rsqrt(t)            # 1 / sqrt(t)
tensor([0.5000, 0.2500])
```

比先 `sqrt` 再算倒数更高效(底层是一步到位的运算),很多归一化层的参考实现里会用它代替"先开根号再做除法"。

### 一个组合起来的例子:实现"均方根"这个量

```python
>>> a = torch.tensor([1.0, 2.0, 3.0, 4.0])
>>> rms = torch.sqrt((a ** 2).mean())     # 先平方,沿所有元素求均值,再开根号
>>> rms
tensor(2.7386)
```

这个组合(平方 → 沿某一维求均值 → 加一个小常数保证数值稳定 → 开根号)正是很多归一化层的核心计算模式——具体沿哪一维求均值,取决于要保留哪些维度不被归约掉(参考第 8 节 `dim` 和 `keepdim` 的用法)。

## 10. `nn.Parameter` 和 `nn.Module`

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

## 11. `register_buffer`:非训练状态

有些张量需要跟着模块走(存、读、跟 `.to(device)` 一起移动),但**不是**可训练参数(比如固定的常量、预先算好的查找表)。这种情况用 `register_buffer`:

```python
class MyLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.register_buffer("scale", torch.ones(dim), persistent=False)
```

`persistent=False` 表示这个 buffer **不需要**被保存进模型的 checkpoint(通常用于"可以随时重新计算出来"的缓存值,没必要占存档空间)。

## 12. 常见陷阱

- **`view` 要求内存连续**:对做过 `transpose`/`permute` 的张量直接 `.view(...)` 经常报错(`RuntimeError: view size is not compatible...`),需要先 `.contiguous()`,或者干脆用 `.reshape(...)` 代替
- **`*` 不是矩阵乘法**:矩阵乘法要用 `@` 或 `torch.matmul`,写错很容易得到形状"恰好能广播、但语义完全错误"的结果,不会报错但结果是错的
- **dtype 不匹配**:`float32` 的张量和 `float64` 的张量做运算可能报错或静默提升精度,需要显式 `.to(dtype)` 对齐
- **device 不匹配**:CPU 上的张量和 GPU 上的张量不能直接运算,报错信息通常很明确(`Expected all tensors to be on the same device`)
- **原地(in-place)操作**:方法名带下划线后缀的(比如 `.add_()`、`.mul_()`)是**原地修改**,不返回新张量;不带下划线的(`.add()`、`.mul()`)返回新张量,原张量不变——训练时对需要梯度的张量做原地操作容易破坏计算图,不确定的话优先用不带下划线的版本
