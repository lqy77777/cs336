# `torch.gather` 与高级索引机制详解

> 通用 PyTorch API 参考,配合 [PyTorch_Tensor基础.md](PyTorch_Tensor基础.md) 使用。所有例子都和作业无关,输出均已实际运行验证。

## 零、三种"取元素"的方式

PyTorch 里从张量取元素有三条完全不同的路径,搞混它们是最常见的 bug 来源:

| 方式 | 写法 | 返回 | 典型用途 |
|---|---|---|---|
| **基本索引** | `a[1]`、`a[:, 2]`、`a[1:3]` | **视图**(共享内存) | 切片、降维 |
| **高级索引** | `a[tensor]`、`a[t1, t2]`、`a[mask]` | **拷贝** | 查表、重排、筛选 |
| **gather 系列** | `torch.gather`、`take_along_dim` | 拷贝 | 沿某一维**逐位置**取不同下标 |

贯穿全文的示例张量:

```python
import torch
a = torch.arange(12).reshape(3, 4)
# tensor([[ 0,  1,  2,  3],
#         [ 4,  5,  6,  7],
#         [ 8,  9, 10, 11]])
```

---

## 一、基本索引(对照组)

用**整数**和**切片**索引,得到的是原张量的视图:

```python
a[1]        # tensor([4, 5, 6, 7])      整数 → 降一维
a[1, 2]     # tensor(6)                 两个整数 → 标量
a[:, 1]     # tensor([1, 5, 9])         切片 + 整数
a[0:2]      # 前两行
```

**视图意味着共享内存**:

```python
v = a[1]
v[0] = 100
a[1, 0]     # 100  ← 原张量被改了
```

这是和高级索引最本质的区别,下面会反复用到。

---

## 二、高级索引(Advanced Indexing)

只要索引里出现了**张量**(整数张量或布尔张量),就进入高级索引模式。

### 2.1 单个索引张量:形状替换规则

**规则**:用形状为 $S$ 的索引张量索引第 0 维,结果形状 = $S$ + **剩余各维原样保留**。

$$\texttt{a.shape} = (n, d_1, d_2, \dots),\quad \texttt{idx.shape} = S \;\Longrightarrow\; \texttt{a[idx].shape} = S + (d_1, d_2, \dots)$$

```python
idx = torch.tensor([2, 0, 2])
a[idx]
# tensor([[ 8,  9, 10, 11],
#         [ 0,  1,  2,  3],
#         [ 8,  9, 10, 11]])
# 形状: (3,) + (4,) = (3, 4)
```

注意三个特点:

- **可以乱序**(先取第 2 行再取第 0 行)
- **可以重复**(第 2 行取了两次)
- **可以改变长度**(索引 5 个元素就得到 5 行)

索引张量本身可以是多维的:

```python
idx2 = torch.tensor([[0, 1],
                     [2, 2]])       # (2, 2)
a[idx2].shape                       # torch.Size([2, 2, 4])
```

$(2,2) + (4,) = (2,2,4)$——**索引张量的形状原封不动地成为输出的前导维**。

> 这正是 `nn.Embedding` 的工作原理:权重表 `(num_embeddings, embedding_dim)` 被形状 `(batch, seq)` 的 token id 索引,得到 `(batch, seq, embedding_dim)`。输出的 batch 结构**由索引张量创造**。

### 2.2 多个索引张量:坐标配对,不是笛卡尔积

这是最容易误解的一点。**提供多个索引张量时,它们按元素"拉链式"配对成坐标**:

```python
rows = torch.tensor([0, 1, 2])
cols = torch.tensor([3, 0, 1])
a[rows, cols]       # tensor([3, 4, 9])
```

取的是 $(0,3), (1,0), (2,1)$ 这**三个**位置,而不是 $3\times3=9$ 个组合。结果形状 = 索引张量广播后的形状 `(3,)`。

**心智模型**:`a[i, j]` 中的 `i` 和 `j` 是两个"下标流",它们逐位对齐,每一对给出一个输出元素。

### 2.3 想要笛卡尔积?靠广播

索引张量之间遵循**标准广播规则**,利用这一点可以做出网格:

```python
r = torch.tensor([[0], [2]])     # (2, 1)
c = torch.tensor([[1, 3]])       # (1, 2)
a[r, c]
# tensor([[ 1,  3],
#         [ 9, 11]])
# 形状 (2, 2)
```

`(2,1)` 与 `(1,2)` 广播成 `(2,2)`,于是取出 $(0,1),(0,3),(2,1),(2,3)$ 四个位置。

**记忆法**:配对是默认行为,笛卡尔积要自己用广播"造"出来。

### 2.4 `arange` 技巧:每行取不同的一个元素

由 2.2 直接推出的经典用法——想"第 $i$ 行取第 $t_i$ 个元素",就让行下标走 `arange`:

```python
pick = torch.tensor([3, 0, 1])
a[torch.arange(a.size(0)), pick]     # tensor([3, 4, 9])
```

这是二维情形下最易读的写法。但**前导维一多就麻烦**:三维 `(B, S, V)` 时要写成

```python
x[torch.arange(B)[:, None], torch.arange(S)[None, :], pick]
```

每多一个前导维就要多构造一个 `arange` 并手动对齐形状。这正是 `gather` 存在的价值(见第三节)。

### 2.5 布尔掩码索引

用布尔张量索引会**拉平**成一维,只保留 `True` 的位置:

```python
a[a % 2 == 0]     # tensor([ 0,  2,  4,  6,  8, 10])
a[a > 6]          # tensor([ 7,  8,  9, 10, 11])
```

掩码形状必须和被索引的那几维匹配。因为结果长度依赖数据内容,**输出形状是动态的**——这会让它无法用于要求静态形状的场景(如 `torch.compile` 的某些路径、导出 ONNX)。需要保形状时改用 `torch.where` 或 `masked_fill`。

### 2.6 高级索引返回**拷贝**

```python
b = torch.arange(12).reshape(3, 4)
cp = b[torch.tensor([1])]
cp[0, 0] = 999
b[1, 0]      # 4  ← 原张量没变
```

对比 2.1 节切片的例子(原张量被改成了 100)。

**为什么必须是拷贝**:高级索引可以乱序、重复、跳跃地取元素,结果在内存里根本无法用"起点 + stride"描述,所以只能新开一块内存。

**但作为赋值左值时是原地写入**:

```python
b[torch.tensor([0, 2])] = 0      # 直接修改 b 的第 0、2 行
```

### 2.7 重复下标赋值:`index_put_` 与累加

下标重复时,普通赋值的行为是"**后写覆盖先写**"(且在并行设备上顺序不确定):

```python
z = torch.zeros(4)
z[torch.tensor([0, 0, 2])] = torch.tensor([1., 1., 5.])
# tensor([1., 0., 5., 0.])   ← 两次写 0 号位,只留下一次的结果
```

要**累加**必须显式声明:

```python
z = torch.zeros(4)
z.index_put_((torch.tensor([0, 0, 2]),), torch.tensor([1., 1., 5.]), accumulate=True)
# tensor([2., 0., 5., 0.])   ← 0 号位是 1+1
```

反向传播中梯度按累加语义处理,这正是同一个 token 在序列里出现多次时 embedding 梯度能正确相加的原因。

### 2.8 陷阱:切片与索引张量混用时的维度顺序

当高级索引**被切片隔开**时,广播出来的维度会被挪到**最前面**:

```python
y = torch.arange(24).reshape(2, 3, 4)

y[:, torch.tensor([0, 2]), :].shape
# torch.Size([2, 2, 4])     ← 索引张量夹在中间,维度留在原位

y[torch.tensor([0]), :, torch.tensor([1, 2])].shape
# torch.Size([2, 3])        ← 两个索引张量被切片隔开,广播维 (2,) 跑到最前
```

第二个例子里,`(1,)` 和 `(2,)` 广播成 `(2,)` 并置于最前,中间那个切片维 `3` 排在后面。这条规则继承自 NumPy,极易踩坑。**建议:混用时先拆成两步,或者干脆改用 `gather`。**

---

## 三、`torch.gather`

### 3.1 语义

```
torch.gather(input, dim, index)
```

以三维、`dim=2` 为例,输出的定义是:

$$\texttt{out}[i][j][k] = \texttt{input}[i][j][\;\texttt{index}[i][j][k]\;]$$

**一句话记住**:除了 `dim` 那一维,其余维度的下标**原样照抄**;只有 `dim` 维的下标被 `index` 里的值替换掉。

这正好补上了高级索引的短板——它天然地"逐位置配对",不需要手工构造 `arange`。

### 3.2 三条硬性规则

1. **`index.dim()` 必须等于 `input.dim()`**(维度**数量**相同,不是形状相同)
2. **对所有 $d \ne \texttt{dim}$,要求 `index.size(d) <= input.size(d)`**
3. **输出形状 == `index` 的形状**(这条最好记,也最有用)

### 3.3 例子集

**(a) 二维,`dim=1`,每行取一个**

```python
index = torch.tensor([[3], [0], [1]])     # (3, 1)
torch.gather(a, 1, index)
# tensor([[3],
#         [4],
#         [9]])                            # (3, 1)
```

等价于 2.4 的 `a[arange(3), pick]`,区别只在于 gather 保留了尾部的长度 1 维。

**(b) 二维,`dim=1`,每行取多个(可重复)**

```python
index = torch.tensor([[0, 3],
                      [1, 1],
                      [2, 0]])            # (3, 2)
torch.gather(a, 1, index)
# tensor([[ 0,  3],
#         [ 5,  5],
#         [10,  8]])
```

第 1 行取了两次同一个元素——完全合法。注意输出形状 `(3,2)` = index 形状。

**(c) 二维,`dim=0`,每列取一个**

```python
index = torch.tensor([[0, 1, 2, 0]])      # (1, 4)
torch.gather(a, 0, index)
# tensor([[ 0,  5, 10,  3]])
```

第 0 列取第 0 行、第 1 列取第 1 行、第 2 列取第 2 行、第 3 列取第 0 行。

**(d) 三维,`dim=2`**

```python
x = torch.arange(24).reshape(2, 3, 4)
index = torch.tensor([[[0], [2], [3]],
                      [[1], [1], [0]]])   # (2, 3, 1)
torch.gather(x, 2, index).squeeze(-1)
# tensor([[ 0,  6, 11],
#         [13, 17, 20]])
```

**这就是 gather 相对高级索引的核心优势**:前导维从 1 个变成 2 个,代码**一个字都不用改**;换成 `arange` 写法则要多构造一个索引网格。

### 3.4 关键陷阱:gather **不广播**

规则 3 说"输出形状 = index 形状",这意味着 index 小了不会报错,而是**静默地只处理一部分**:

```python
torch.gather(a, 1, torch.tensor([[1]]))    # index 形状 (1,1)
# tensor([[1]])       ← 只取了第 0 行!其余两行被无声丢弃
```

对比 `take_along_dim`(见第四节),后者会广播成 `(3,1)`。**这是 gather 最阴险的一个坑**——形状不对时它不报错,只是给你一个更小的结果。

### 3.5 常见报错对照

| 报错信息 | 原因 | 处理 |
|---|---|---|
| `Index tensor must have the same number of dimensions as input tensor` | index 少一维 | `unsqueeze(-1)` 补上 |
| `gather(): Expected dtype int32/int64 for index` | index 是 float | `.long()` |
| `index -1 is out of bounds for dimension 1 with size 4` | **gather 不支持负数索引** | 先 `index % size` 转正 |

最后一条值得强调:

```python
torch.gather(a, 1, torch.tensor([[-1], [0], [1]]))   # RuntimeError
a[torch.tensor([-1])]                                 # 正常,取最后一行
```

**高级索引支持负数下标,gather 不支持。**

---

## 四、`take_along_dim`:gather 的宽松版

```python
torch.take_along_dim(input, indices, dim)
```

语义与 gather 相同(NumPy `take_along_axis` 的对应物),但**在非 `dim` 维上允许广播**:

```python
torch.take_along_dim(a, torch.tensor([[3], [0], [1]]), dim=1)
# tensor([[3], [4], [9]])          ← 和 gather 一致

torch.take_along_dim(a, torch.tensor([[1]]), dim=1)
# tensor([[1], [5], [9]])          ← 广播到 (3,1),每行都取第 1 列

torch.gather(a, 1, torch.tensor([[1]]))
# tensor([[1]])                    ← gather 只给一行
```

**取舍**:`take_along_dim` 更宽容、名字更自明;`gather` 更严格,形状写错时更容易在别处暴露出来。想要"形状必须精确对上"的安全感就用 gather,想少写几个 `expand` 就用 `take_along_dim`。

---

## 五、`scatter`:gather 的逆运算

`gather` 是"按下标**取**",`scatter_` 是"按下标**放**":

```python
oh = torch.zeros(3, 4)
oh.scatter_(1, torch.tensor([[3], [0], [1]]), 1.0)
# tensor([[0., 0., 0., 1.],
#         [1., 0., 0., 0.],
#         [0., 1., 0., 0.]])
```

这是构造 one-hot 的标准手法。下标重复时若要累加,用 `scatter_add_`。

**`gather` 的反向传播恰好就是 `scatter_add`**——取过的位置收到梯度,取了两次的位置梯度翻倍:

```python
w = torch.arange(12.).reshape(3, 4).requires_grad_(True)
torch.gather(w, 1, torch.tensor([[0, 0], [1, 2], [3, 3]])).sum().backward()
w.grad
# tensor([[2., 0., 0., 0.],      ← 第 0 列取了两次
#         [0., 1., 1., 0.],
#         [0., 0., 0., 2.]])     ← 第 3 列取了两次
```

理解这一点就明白了:**查表操作是完全可微的**,梯度会沿着"被查过的路径"原路返回并累加。

---

## 六、选型决策表

| 你想做的事 | 推荐 | 理由 |
|---|---|---|
| 按行号取**整行**(查表 / embedding) | 高级索引 `w[ids]` | 索引张量的形状直接成为输出前导维 |
| 每行取**同一个**列 | 基本索引 `a[:, j]` | 返回视图,零拷贝 |
| 每行取**不同的一个**元素(二维) | `a[arange(n), idx]` | 最易读 |
| 每行取**不同的**元素(**任意前导维**) | **`gather` / `take_along_dim`** | 前导维增加时代码不变 |
| 按布尔条件筛选 | 掩码 `a[mask]` | 但输出形状动态 |
| 按下标**写入** | `scatter_` / `index_put_` | 注意重复下标要不要累加 |
| 需要保持形状、不能拉平 | `torch.where` / `masked_fill` | 掩码索引会拉平 |

**一条经验法则**:如果索引的语义是"**为已经存在的每个位置挑一个下标**",用 gather;如果是"**用下标创造出新的批次结构**",用高级索引。

---

## 七、易错点清单

1. **gather 的 index 少一维** → `RuntimeError`。先 `unsqueeze`,用完再 `squeeze`。
2. **gather 不广播** → 形状小了不报错,静默丢数据。这是最难发现的一个坑。
3. **gather 不接受负数下标**,高级索引接受。
4. **index 必须是整数类型**(int32/int64),float 会报错。
5. **多个索引张量是配对,不是笛卡尔积**。要笛卡尔积得自己用广播凑。
6. **高级索引返回拷贝**,想原地改必须写成赋值语句的左边。
7. **重复下标赋值默认覆盖而非累加**,要累加得用 `index_put_(..., accumulate=True)` 或 `scatter_add_`。
8. **切片与索引张量混用时维度顺序会变**,被切片隔开的广播维会跑到最前面。
9. **布尔掩码索引会拉平**,输出形状依赖数据内容。

---

## 八、速查表

```python
# ---------- 基本索引(视图) ----------
a[1]                 # 取一行,降维
a[:, 1]              # 取一列,降维
a[0:2]               # 切片

# ---------- 高级索引(拷贝) ----------
a[idx]               # (S,) + 剩余维;查表
a[rows, cols]        # 坐标配对,逐位取标量
a[r[:,None], c]      # 广播成网格 → 笛卡尔积
a[arange(n), pick]   # 每行取不同的一个(仅二维方便)
a[a > 0]             # 布尔筛选,拉平

# ---------- gather 系列 ----------
torch.gather(x, dim, index)          # index.dim() == x.dim();输出形状 == index 形状
torch.take_along_dim(x, idx, dim)    # 同上,但允许广播

# ---------- 写入 ----------
x.scatter_(dim, index, src)                    # 按下标放
x.scatter_add_(dim, index, src)                # 按下标累加
x.index_put_((idx,), val, accumulate=True)     # 重复下标累加
```

---

## 九、相关笔记

- [PyTorch_Tensor基础.md](PyTorch_Tensor基础.md)
- [einops用法整理.md](einops用法整理.md)
- [transformer.py开发知识点总结.md](transformer.py开发知识点总结.md)
