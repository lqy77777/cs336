# einops 用法整理

版本：`einops 0.8.2`（本文所有示例均在该版本下实测通过）。

## 0. einops 解决的是什么问题

PyTorch 原生的 `reshape`、`permute`、`transpose`、`sum(dim=...)` 都用**数字下标**指定维度，写多了容易记混——`x.permute(0, 2, 1, 3)` 到底是把哪个维度换到哪去了，脱离上下文根本看不出来。

einops 用**具名字符串**代替数字下标：

```python
x.permute(0, 2, 1, 3)                          # 数字下标,要对照 shape 才看得懂
rearrange(x, 'b h s d -> b s h d')             # 字符串直接说明白:交换 h 和 s
```

一共只有几个核心函数，但覆盖了 `reshape` / `permute` / `transpose` / `sum` / `mean` / `repeat` / `tile` / `einsum` 的大部分日常用法。

### 返回值与是否操作输入——总览

在看具体函数之前先说结论，因为这件事比 API 本身更容易踩坑：

**einops 的所有函数都不是原地操作（in-place）——它们从不修改传入的张量本身，永远返回一个新的 tensor 对象（`x = f(x)` 才能让这个"新结果"生效，单独调用 `f(x)` 而不接收返回值什么都不会改变）。**

但"返回新对象"和"返回的数据是不是独立拷贝"是两回事——后者才是真正需要注意的地方：

| 函数 | 是否原地修改输入 | 返回的是新 tensor 对象 | 返回的数据是否与输入共享内存 |
|---|---|---|---|
| `rearrange` | 否 | 是 | **看情况**：能用 view 表示就共享内存，做不到就拷贝 |
| `reduce` | 否 | 是 | 否，总是新分配内存 |
| `repeat` | 否 | 是 | **看情况，且容易踩坑**：广播式复制往往共享内存（见 3.4 节） |
| `einsum` | 否 | 是 | 否，总是新分配内存 |
| `pack` | 否 | 是 | 否，拷贝进新的连续内存 |
| `unpack` | 否 | 是（返回多个新对象） | **是**，返回的是指向 `pack` 结果内部的视图 |
| `Rearrange` / `Reduce`（layers） | 否 | 是 | 与对应的函数版本完全一致 |

下面每一节里会具体说明"什么情况下共享内存"，因为这直接决定了"改结果会不会连带改坏原始数据"。

---

## 1. `rearrange`：维度的重排、合并、拆分

`rearrange(tensor, pattern, **axes_lengths)`。pattern 的形式是 `'输入轴 -> 输出轴'`，箭头两边给维度起名字，靠名字对应关系描述变换。

### 1.1 转置（等价于 permute / transpose）

```python
>>> x = torch.arange(24).reshape(2, 3, 4)
>>> rearrange(x, 'a b c -> b a c').shape
torch.Size([3, 2, 4])
```

箭头右边的顺序就是新的维度顺序——比 `x.permute(1, 0, 2)` 直观得多。

**返回值与是否操作输入**：不修改 `x` 本身，返回一个新的 tensor 对象。纯转置（不涉及内存重新排布，PyTorch 底层就是靠改 stride 实现）返回的是 `x` 的**视图（view）**，和 `x` 共享同一块底层内存：

```python
>>> x = torch.arange(12.0).reshape(3, 4)
>>> y = rearrange(x, 'a b -> b a')
>>> y[0, 0] = 999.0
>>> x[0, 0]
tensor(999.)          # x 也被改了!因为 y 只是 x 的另一种"看法",不是独立拷贝
```

这不是 einops 的怪癖，而是和 `x.t()` / `x.permute(...)` 完全一致的 PyTorch 视图语义——einops 只是转发到底层的 `permute`。**如果打算修改结果又不想连累原始张量，要显式 `.clone()`。**

### 1.2 合并维度：用括号 `(...)`

```python
>>> rearrange(x, 'a b c -> a (b c)').shape
torch.Size([2, 12])
```

`(b c)` 表示把 `b` 和 `c` 两个轴压成一个轴，大小是 `b * c`。等价于 `x.reshape(2, -1)`，但明确写出了"是把 b 和 c 合并"，而不是笼统的"剩下的全部拍平"。

**返回值与是否操作输入**：同样不修改输入，是否共享内存**取决于原始张量在内存里是否连续、以及合并的轴是否相邻**。上面这个例子里 `x` 本身连续、`b` 和 `c` 又是相邻的两个轴，所以能直接用 view 表示、共享内存；但如果先转置再合并，PyTorch 无法只靠改 stride 表示出结果，就会自动拷贝一份新内存：

```python
>>> x2 = torch.arange(12.0).reshape(3, 4)
>>> y2 = rearrange(x2, 'a b -> (b a)')     # 转置后再合并,内存不再连续
>>> y2[0] = 999.0
>>> x2[0, 0]
tensor(0.)             # x2 没被改——这种情况下返回的是拷贝
```

**这一点无法靠看 pattern 字符串直接判断，是共享内存还是拷贝，取决于具体的 shape 和内存布局。** 如果代码逻辑依赖"改了返回值原始数据不受影响"，不要凭猜测，要么显式 `.clone()`，要么用 `.data_ptr()` 实际检查一下。

### 1.3 拆分维度：同样用括号，但要指定至少一个子轴的大小

```python
>>> y = torch.arange(24).reshape(2, 12)
>>> rearrange(y, 'a (b c) -> a b c', b=3).shape
torch.Size([2, 3, 4])
```

拆分是合并的逆操作。因为光看 `12` 拆不出 `b` 和 `c` 分别是多少，必须显式传 `b=3`（或 `c=4`），另一个会被自动推导。

**常见报错**：如果传的 `b` 不能整除原维度大小，或者你干脆没有告诉它任何一个子轴的大小，`einops.EinopsError` 会直接抛出来，而不是像裸 `reshape` 那样默默给你一个形状对但语义错的结果。这是 einops 相比手写 reshape 更安全的地方。

### 1.4 省略号 `...`：批量维度不想每次都写全

```python
>>> w = torch.rand(2, 3, 4, 5)
>>> rearrange(w, '... h w -> ... (h w)').shape
torch.Size([2, 3, 20])
```

`...` 代表"前面随便多少个维度，原样保留"。写 Transformer 代码时，batch 维、头数维经常不想每次都显式命名，用 `...` 省事。

### 1.5 元素个数必须对得上

`rearrange` 只做**维度的重新排布**，不做数值计算，所以箭头两边涉及的**元素总数必须相等**。这是判断"该用 rearrange 还是该用 reduce"的关键分界线：

```python
>>> rearrange(x, 'a b c -> a c')          # b 去哪了?
EinopsError: ...
```

丢掉一个维度但不说明怎么处理它（求和？取平均？），einops 直接报错，不会像某些框架那样帮你瞎猜。**这正是它比 `.view()` / `.reshape()` 更不容易埋雷的地方。**

---

## 2. `reduce`：带聚合操作的降维

`reduce(tensor, pattern, reduction, **axes_lengths)`。当你确实想丢掉某个维度、且明确知道要怎么聚合时用它。

```python
>>> y = torch.rand(2, 3, 4)
>>> reduce(y, 'a b c -> a b', 'mean').shape     # 对 c 求平均
torch.Size([2, 3])
>>> reduce(y, 'a b c -> a', 'sum').shape        # 对 b 和 c 都求和
torch.Size([2])
```

**返回值与是否操作输入**：不修改 `y` 本身；返回值**总是新分配的独立内存**，不会和输入共享——这一点比 `rearrange` 简单干脆，因为求和/平均等聚合运算本质上就是"从多个输入元素算出一个新值"，不可能只靠改 stride 表示，必然要写进新内存。放心大胆地修改 `reduce` 的返回值，不会连累原始张量。

`reduction` 参数支持 `'mean'`、`'sum'`、`'max'`、`'min'`、`'prod'`，以及你自己传一个可调用对象。

### 保留维度但压成 1（不是彻底丢掉）

```python
>>> reduce(y, 'a b c -> a 1 c', 'max').shape
torch.Size([2, 1, 4])
```

输出模式里写字面量 `1`，表示"这个位置的维度保留下来，但压缩成 1"——等价于 `torch.max(y, dim=1, keepdim=True)`，但省去了 `keepdim` 这种容易忘记加的参数。

### `rearrange` 与 `reduce` 的分界线

| | 元素总数 | 用途 |
|---|---|---|
| `rearrange` | 前后相等 | 换位置、合并、拆分 |
| `reduce` | 后面变少 | 求和、平均、取最值 |

箭头右边"丢掉了某个左边出现过的轴名"——这是需要用 `reduce` 而不是 `rearrange` 的信号。

---

## 3. `repeat`：广播式复制

`repeat(tensor, pattern, **axes_lengths)`，和 `reduce` 正好相反——**元素总数变多**。

### 3.1 增加一个全新的维度

```python
>>> z = torch.rand(3, 4)
>>> repeat(z, 'h w -> b h w', b=5).shape
torch.Size([5, 3, 4])
```

`b` 是输出里全新出现的轴名，必须显式给出它的大小。原有的 `(3,4)` 数据被复制了 5 份。

**返回值与是否操作输入 —— 这是本文最重要的一个坑**：`repeat` 不修改 `z` 本身，返回新对象。但这种"增加全新维度"的复制，**底层用的是广播（等价于 `expand`），复制出来的 5 份实际上指向同一块内存**，并不是真的写了 5 份数据出去：

```python
>>> r = repeat(z, 'h w -> b h w', b=5)
>>> r.stride()
(0, 4, 1)              # 第 0 维(也就是"复制"出来的那一维) stride=0
```

`stride=0` 意味着这一维在内存里根本不存在、只是逻辑上假装有 5 份。后果是——**往其中一份写入，会同时改变所有份，还会连带改坏原始输入**：

```python
>>> x = torch.arange(4.0)
>>> r = repeat(x, 'w -> b w', b=3)
>>> r[0, 0] = 999.0
>>> r
tensor([[999.,   1.,   2.,   3.],
        [999.,   1.,   2.,   3.],      # 没有单独修改第 0 行,三行全变了
        [999.,   1.,   2.,   3.]])
>>> x
tensor([999.,   1.,   2.,   3.])       # 原始输入也被改坏了!
```

**这是最容易埋雷的一步：`repeat` 看起来是"复制了 n 份独立数据"，实际上给你的是共享内存的视图。** 只要接下来只做**只读**操作（矩阵乘、求和、当作另一个函数的输入等），完全没问题、而且更省内存；但凡打算**原地修改**其中一份，必须先 `.clone()`：

```python
r = repeat(x, 'w -> b w', b=3).clone()   # 需要独立可写时,显式拷贝
```

### 3.2 沿着某个已有维度平铺

```python
>>> repeat(z, 'h w -> h (r w)', r=2).shape
torch.Size([3, 8])
```

`(r w)` 表示：先把 `w` 复制 `r` 份，再和原来的 `w` 合并成一个更大的轴。这是 `torch.tile` 的等价写法。

### 3.3 插入一个大小为 1 的新轴（两种等价写法）

```python
>>> repeat(z, 'h w -> h w 1').shape
torch.Size([3, 4, 1])
>>> repeat(z, 'h w -> h w c', c=1).shape       # 效果相同,只是显式起了个名字
torch.Size([3, 4, 1])
```

等价于 `z.unsqueeze(-1)`，但你可以顺手把新插入的轴命名，方便后面维护代码时理解这个维度是干什么用的。

---

## 4. `einops.einsum`：带轴名的爱因斯坦求和

PyTorch 原生 `torch.einsum` 用单字母下标（`'ij,jk->ik'`），维度一多字母就不够用、且可读性差。`einops.einsum` 允许用任意长度的名字：

```python
>>> from einops import einsum
>>> a = torch.rand(2, 3)
>>> b = torch.rand(3, 4)
>>> einsum(a, b, 'i j, j k -> i k').shape        # 矩阵乘法
torch.Size([2, 4])

>>> c = torch.rand(2, 3, 4)
>>> d = torch.rand(2, 3, 4)
>>> einsum(c, d, 'b s d, b s d -> b s').shape    # 逐位置点积,常见于 attention score
torch.Size([2, 3])
```

**和 `torch.einsum` 的关键区别**：`einops.einsum` 的张量参数在前、pattern 字符串在最后（`einsum(a, b, pattern)`），而 `torch.einsum` 是 pattern 在最前（`torch.einsum(pattern, a, b)`）。两者别混着写。

**返回值与是否操作输入**：不修改 `a`、`b`，返回一个新分配的独立张量——道理和 `reduce` 一样，求和类运算不可能只靠 view 表示。**梯度会正常保留**：如果输入 `requires_grad=True`，返回结果可以正常 `.backward()`，einops 这一层只是重新组织了调用方式，底层实际执行的还是 PyTorch 原生算子，自动微分不受影响（`rearrange` / `reduce` / `repeat` 同理，全部支持 autograd）。

**没有隐式求和**：`torch.einsum` 里如果某个下标只出现在输入没出现在输出，会被自动隐式求和；`einops.einsum` 要求你在 `->` 右边显式写清楚保留哪些轴，不允许有歧义的隐式行为——这跟 `rearrange` 报错设计的哲学是一致的：**宁可让你多写几个字符，也不要帮你"猜"。**

---

## 5. `einops.layers`：把变换封装成 `nn.Module`

写 `nn.Sequential` 时，中间想插入一次维度变换，直接用函数不行（`Sequential` 里的每一项必须是 `nn.Module`）。`einops.layers.torch` 提供了对应的 Module 封装：

```python
>>> from einops.layers.torch import Rearrange, Reduce
>>> layer = Rearrange('b c h w -> b (c h w)')
>>> layer(torch.rand(2, 3, 4, 5)).shape
torch.Size([2, 60])
```

`Rearrange(pattern)` / `Reduce(pattern, reduction)` 的参数和函数版完全一致，只是包了一层 `nn.Module` 的壳，可以直接放进 `nn.Sequential(...)`。

**返回值与是否操作输入**：`nn.Module.__call__`（也就是 `layer(x)`）本身不修改 `x`，返回值就是内部调用对应函数版本（`rearrange` / `reduce`）的结果——所以是否共享内存，完全遵循 1、2 两节里各自的规则，没有额外差异。

---

## 6. `pack` / `unpack`：变长的合并与拆分

处理"几个形状不完全相同、但除了某个维度外都一样"的张量时很有用，比如把多个头的 Q/K/V 打包再拆开。

```python
>>> from einops import pack, unpack
>>> p1 = torch.rand(2, 3)
>>> p2 = torch.rand(2, 5)
>>> packed, ps = pack([p1, p2], 'b *')
>>> packed.shape
torch.Size([2, 8])
>>> ps
[torch.Size([3]), torch.Size([5])]

>>> u1, u2 = unpack(packed, ps, 'b *')
>>> u1.shape, u2.shape
(torch.Size([2, 3]), torch.Size([2, 5]))
```

`*` 是"这个位置的大小允许在各个输入之间不同"的占位符。`pack` 返回打包后的张量和一份"怎么拆回去"的形状记录 `ps`；`unpack` 拿着这份记录就能精确复原。

**返回值与是否操作输入**：`pack` 不修改 `p1`、`p2`，返回的 `packed` 是**拷贝**进一块新连续内存的结果（因为要把两个形状不同的张量首尾拼接，物理上必须搬数据）：

```python
>>> packed[0, 0] = 999.0
>>> p1[0, 0]
tensor(0.)              # p1 没被改
```

但 `unpack` 相反——它把 `packed` 切片切开还给你，**返回的是指向 `packed` 内部的视图，不是拷贝**：

```python
>>> u1, u2 = unpack(packed, ps, 'b *')
>>> u1[0, 0] = -1.0
>>> packed[0, 0]
tensor(-1.)              # packed 被连带改了
```

这符合直觉——`unpack` 存在的意义就是"分别去操作 pack 之前的各个部分"，如果每次都拷贝一份反而违背了它的设计初衷。但如果你打算独立修改某一份 unpack 出来的结果、又不想影响 `packed`，同样需要显式 `.clone()`。

---

## 7. 辅助函数

### `parse_shape`：把 shape 转成带名字的字典，方便后续复用

```python
>>> from einops import parse_shape
>>> x = torch.rand(2, 3, 4)
>>> parse_shape(x, 'b s d')
{'b': 2, 's': 3, 'd': 4}
```

常见用法：先 `parse_shape` 拿到 `b`、`s`、`d` 这些数字，后面别的 `rearrange` 调用需要传 `axes_lengths` 时直接复用，不用重复写 `x.shape[0]` 这种容易錯位的下标。

### `asnumpy`：脱离框架取值

```python
from einops import asnumpy
asnumpy(x)   # 不管 x 是 torch/tf/jax 的张量,统一转成 numpy array
```

---

## 8. 与原生 PyTorch 写法对照表

| 需求 | PyTorch 原生 | einops |
|---|---|---|
| 转置两个维度 | `x.transpose(1, 2)` | `rearrange(x, 'a b c -> a c b')` |
| 任意顺序重排 | `x.permute(2, 0, 1)` | `rearrange(x, 'a b c -> c a b')` |
| 展平 | `x.reshape(a, -1)` | `rearrange(x, 'a b c -> a (b c)')` |
| 拆分一个维度 | `x.reshape(a, b, c)`（容易拆错顺序） | `rearrange(x, 'a (b c) -> a b c', b=...)` |
| 求平均降维 | `x.mean(dim=-1)` | `reduce(x, '... d -> ...', 'mean')` |
| keepdim | `x.max(dim=1, keepdim=True)` | `reduce(x, 'a b c -> a 1 c', 'max')` |
| 沿新维度复制 | `x.unsqueeze(0).expand(n, -1, -1)` | `repeat(x, 'h w -> n h w', n=...)` |
| 插入新轴 | `x.unsqueeze(-1)` | `repeat(x, 'h w -> h w 1')` |
| 矩阵乘 | `torch.einsum('ij,jk->ik', a, b)` | `einsum(a, b, 'i j, j k -> i k')` |

---

## 9. 为什么它比原生写法"更安全"

这是 einops 设计上反复强调的一点，贯穿前面几节，这里单独总结：

- **元素数量必须自洽**：`rearrange` 前后元素总数必须相等，对不上直接报 `EinopsError`，不会给你一个形状对但数据错位的"静默 bug"（这是裸 `reshape` 最容易踩的坑——形状凑巧对上了，但数据其实被弄乱了顺序）
- **没有隐式行为**：`reduce` 必须显式指定聚合方式，`einops.einsum` 不允许隐式求和，`rearrange` 不允许丢维度不聚合——处处要求你把意图写清楚
- **可读性即文档**：`'b h s d -> b s h d'` 本身就在告诉读代码的人"这是把 head 和 seq 换了位置"，而 `permute(0, 2, 1, 3)` 需要你对照上下文才知道在干什么

## 10. 常见陷阱

- **`repeat` 的"复制"经常是共享内存的广播视图，写入会同时污染所有副本和原始输入**——这是本文里最容易造成静默数据错误的一点（1.4 节 / 3.1 节都有实测），需要独立可写时必须显式 `.clone()`
- **`unpack` 返回的是指向 `pack` 结果内部的视图，不是拷贝**（6 节）——修改 unpack 出来的某一份会连带改变 `packed`
- **`rearrange` 是否共享内存取决于内存布局，不能只看 pattern 字符串判断**——纯转置通常是 view，转置后再合并通常变成拷贝，拿不准就 `.clone()` 或查 `.data_ptr()`
- **`einops.einsum` 的参数顺序和 `torch.einsum` 相反**：`einops.einsum(a, b, pattern)` vs `torch.einsum(pattern, a, b)`，混用会直接报错或得到完全不相关的结果
- **拆分维度时至少指定一个子轴大小**：`rearrange(y, 'a (b c) -> a b c')` 不传 `b=` 或 `c=` 会报错，因为单靠总数拆不出两个未知数
- **合并/拆分只能对元素总数做文章，不能做数值变换**：想要"每隔一个取一个"这种下采样，不是 `rearrange` 的工作，要用切片或 `reduce`
- **箭头右边出现输入没有的轴名**——比如打错字——会直接报 `EinopsError`，这通常意味着你少写了一个 `->` 右边该出现的字母，而不是 einops 本身的问题
