# `transformer.py` 开发过程知识点总结

> 按写作顺序整理:`Linear` → `Embedding` → `RMSNorm` → `SiLU`/`Feedforward` → `RoPE` → `softmax`。每个组件下列出:实际踩过的坑、当前代码逐行要注意的点、以及每一处"原生写法 vs einops 写法"的对比。API 细节的展开讲解见 [PyTorch_Tensor基础.md](PyTorch_Tensor基础.md)。

---

## 〇、文件头部(import)

- `from math import sqrt`:`math.sqrt` 只接受单个 Python 数字、返回 Python `float`——用来算标量超参数(初始化的 σ)正合适;对 tensor 要用 `torch.sqrt`(逐元素),两者不能混用
- `from einops import einsum, rearrange`:注意 `einops.einsum` 的参数顺序是"先张量、后模式串"(`einsum(a, b, 'i,j -> i j')`),和 `torch.einsum`("先模式串、后张量")相反
- `from jaxtyping import Float`:目前只是导入了还没用上——jaxtyping 标注(`Float[Tensor, "... d"]`)是可选的文档手段,运行时不强制检查;讲义推荐接口用的都是普通 `torch.Tensor` 标注

---

## 一、`Linear`

### Linear:踩过的坑

1. **`nn.Parameter(out_features, in_features)` 是错的**——`nn.Parameter(data, requires_grad=True)` 第一个参数必须是已存在的 tensor。正确顺序:先 `torch.empty(out_features, in_features, device=device, dtype=dtype)` 造形状/设备/精度都正确的空 tensor,再包进 `nn.Parameter`
2. **初始化必须在 `__init__` 里、只做一次**——放进 `forward` 会导致每次前向传播都把权重重新随机化,训练结果被不断抹掉
3. **测试 adapter 里不能 `linear.weight = weights` 直接赋值**——`nn.Module.__setattr__` 拦截:已注册为 `Parameter` 的名字,再赋值时必须还是 `Parameter`(或 `None`),普通 tensor 会报 `TypeError`。正确方式是 `load_state_dict({"weight": weights})`(按名字拷贝值,不替换对象)

### Linear:逐行要点

- 第 23 行 `self.sigma = sqrt(2 / (out_features + in_features))`:σ 是**标准差**(方差 `2/(d_in+d_out)` 开根号),`trunc_normal_` 要的是 std 不是方差;这里输入是纯 Python 数字,用 `math.sqrt` 正确
- 第 25 行 `nn.init.trunc_normal_(w, mean=0.0, std=σ, a=-3σ, b=3σ)`:末尾下划线 = **原地操作**,在已有 tensor 上填充;`a`/`b` 是绝对上下界,不是"几倍标准差"
- 第 29 行 `x @ self.weight.T`:
  - 为什么要转置:数学记号列向量(`y=Wx`)vs PyTorch 批量在前的行向量惯例(`y=xW^T`),存 `W` 用 `W^T` 算
  - **`.T` 只安全用于二维**——高维张量上 `.T` 反转所有维度(已弃用警告),想转置最后两维要用 `.mT` 或 `transpose(-2,-1)`。权重是二维,这里没问题
  - `@` 自动处理任意批量维:`(in,)`、`(batch,in)`、`(batch,seq,in)` 都能被同一行正确处理(一维张量在 `@` 左侧临时当行向量)

### Linear:einops 对比

```python
x @ self.weight.T                                              # 原生:需要记得转置
einsum(x, self.weight, '... d_in, d_out d_in -> ... d_out')    # einops:轴名自动对齐,免转置
```

einsum 版本把"转置账"交给轴名去算,批量维用 `...` 表达;这里维度关系简单,两种都可以,`@` 更短。

---

## 二、`Embedding`

### Embedding:踩过的坑

1. **`self.embedding(token_ids)` 圆括号调用是错的**——`Parameter` 是 tensor,不可调用,报 `TypeError: 'Parameter' object is not callable`;查表用**方括号索引** `self.weight[token_ids]`
2. **属性命名要按 PyTorch 惯例叫 `self.weight`**——当下单独测试无所谓(adapter 的 key 自己定),但以后加载外部参考 `state_dict`(key 是按官方惯例固定的)会对不上,提前统一省得返工

### Embedding:逐行要点

- 第 40 行 `self.sigma = 1`:Embedding 的初始化是固定 `std=1`、截断 `[-3,3]`,**不是** `Linear` 那个按维度算的公式——两个模块初始化方案不同
- 第 46 行 `self.weight[token_ids]`:花式索引,结果形状 = `token_ids` 的形状 + 表除第一维外剩下的形状——`(batch, seq)` 的 ids 索引 `(vocab, d)` 的表得到 `(batch, seq, d)`

### Embedding:einops 对比

**没有对比可做**——einops 的三件套(`rearrange`/`einsum`/`reduce`)只做重排、收缩、归约,**不做索引/查表**。这一行只能用原生索引,没有 einops 等价物。

---

## 三、`RMSNorm`

### RMSNorm:踩过的坑

1. **`__init__` 存 `self.d_eps`、`forward` 用 `self.eps`,命名不一致** → `AttributeError`。教训:属性名直接沿用构造参数名
2. **`torch.sum` 写成了公式要求的 `torch.mean`**——形状一模一样(靠 `keepdim=True`),任何形状检查都发现不了,数值却整体错了 `d_model` 倍。"形状对、数值错"是最隐蔽的一类 bug,只有和参考值比对才能揪出来
3. **`g` 写成 `torch.ones(d_model, 1)` 多了一维**——广播会对齐到错误的维度(倒数第二维),要么报错、要么(更危险)碰巧广播成功但语义全错。正确形状是一维 `(d_model,)`
4. **`device`/`dtype` 传给了 `nn.Parameter(...)`**——`nn.Parameter` 不接受这两个参数,它们必须传给**构造 tensor 的函数**(`torch.ones(d_model, device=..., dtype=...)`)

### RMSNorm:逐行要点

- 第 61-62、65 行(升/降精度骨架):`in_dtype = x.dtype` → `x.to(torch.float32)` → 算 → `result.to(in_dtype)`。动机:低精度(fp16/bf16)下平方运算容易溢出成 `inf` → `NaN`,内部升到 fp32 保稳定,出口降回原精度保持接口不变
- 第 63 行 `torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)`:
  - `x ** 2` 逐元素平方;`mean(dim=-1)` 只在特征维归约,批量维保留
  - **`keepdim=True` 必须有**——下一行要拿它和 `x` 做除法,不保留这一维,广播可能对齐错维度(正方形形状时甚至不报错、悄悄算错)
  - `+ self.eps` 在开根号**里面**(公式定义如此),防止除零
- 第 64 行 `(x / rms) * self.weight`:`rms` 形状 `(..., 1)` 广播到 `(..., d)`;`self.weight` 形状 `(d,)` 从右对齐广播到最后一维——两次广播都是"对齐到最后一维",正确

### RMSNorm:einops 对比

```python
torch.mean(x ** 2, dim=-1, keepdim=True)          # 原生:靠 dim + keepdim 两个参数
reduce(x ** 2, '... d -> ... 1', 'mean')          # einops:右侧写 1 = 保留该维长度 1
```

`reduce` 模式串右侧写 `1` 等价于 `keepdim=True`、不写等价于 `False`——把"归约哪一维、保不保留"直接写在轴名里,不容易忘 `keepdim`。

---

## 四、`SiLU` / `Feedforward`(SwiGLU)

### Feedforward:踩过的坑

1. **一开始把 SwiGLU 写成了普通函数**——adapter 需要"构造实例 → `load_state_dict` 加载参考权重 → 调用"这套流程,只有 `nn.Module` 类做得到;`SiLU` 无状态,保持普通函数正确
2. **忘了 `super().__init__()`** → `AttributeError: cannot assign module before Module.__init__() call`(这次触发点是子模块赋值,和 `Parameter` 赋值同一个机制)
3. **`load_state_dict` 的 key 少了 `.weight` 后缀**——嵌套子模块的 key 是点号路径:`self.W_1` 是 `Linear` 实例、它内部权重叫 `weight`,完整 key 是 `"W_1.weight"` 不是 `"W_1"`
4. **`Linear(d_ff, d_model, ...)` 参数顺序反了(三行全反)**——报错信息里"checkpoint 期望形状"和"当前模型形状"恰好互换,就是参数顺序颠倒的信号
5. **`Linear(out_features=d_ff, in_features=d_model, device, dtype)` 语法错误**——关键字参数之后不能再跟位置参数(`SyntaxError: positional argument follows keyword argument`),而且**语法错误会让整个文件无法导入**,所有依赖它的测试文件全部崩掉。要么全位置、要么后面的也全写成 `device=device, dtype=dtype`

### Feedforward:逐行要点

- 第 67-68 行 `SiLU(x) = x * torch.sigmoid(x)`:逐元素,任意形状输入都行;讲义明确允许用 `torch.sigmoid`(数值稳定);`*` 是逐元素乘不是矩阵乘
- 第 79-81 行:`W_1`、`W_3` 是 `d_model → d_ff`(升维),`W_2` 是 `d_ff → d_model`(降回)——`W_2` 方向和另外两个相反,最容易写反的地方
- 第 83 行 `self.W_2(SiLU(self.W_1(x)) * self.W_3(x))`:
  - 子模块用 `self.W_1(x)` 这种**调用**方式(触发 `__call__` → `forward`),不要写 `self.W_1.forward(x)`
  - 中间的 `*` 是 GLU 门控的**逐元素**相乘,两边形状都是 `(..., d_ff)`,正好对齐

### Feedforward:einops 对比

这一层的三次线性变换都发生在 `Linear` 内部(见第一节的对比);`SiLU(...) * self.W_3(x)` 是纯逐元素乘法,**einops 没有逐元素运算的对应物**(它不做 pointwise 运算),这行没有 einops 写法。

---

## 五、`RotaryPositionalEmbedding`(RoPE)

### RoPE:踩过的坑

1. **`register_buffer` 忘了 `persistent=False`**——cos/sin 表是确定性算出来的,不训练、不该存 checkpoint;默认 `persistent=True` 会让它进 `state_dict`,以后加载外部参考权重时因"多出 unexpected key"报错
2. **`torch.arange` 忘了传 `device`**——buffer 固定生成在 CPU,模型上 GPU 后 forward 里设备不匹配
3. **`x[-1][0:-1:2]` 三重错误**:`x[-1]` 对第一维取下标,砍掉了批量维;切片切在错误的维度;后续 stack 出的形状和 `'a b -> (a b)'` 模式对不上。正确:`x[..., 0::2]` 用 `...` 保留全部前置维、只切最后一维
4. **`torch.cos(x_1)` 把数据当角度**——概念错误:旋转公式里的 cos/sin 是**预先按位置算好的系数表**(查表取出),和数据相乘;不是对数据本身求三角函数。症状:算好的 `cosine`/`sine` 两个变量从没被用过,这是发现此类错误的线索
5. **`rearrange(temp, 'a b -> (a b)')` 遇到批量维报 `EinopsError`**——einops 模式声明几个轴,输入就必须恰好几维;要用 `'... a b -> ... (a b)'`(或 `flatten(start_dim=-2)`)

### RoPE:`__init__` 逐行要点

- 第 94 行 `torch.arange(max_seq_len, device=device, dtype=torch.float32)`:位置序列显式转浮点——`arange` 全整数参数默认给 `int64`,后面要进外积和三角函数,浮点更稳
- 第 95 行 `torch.arange(1, (d_k/2)+1, device=device)`:**`d_k/2` 是浮点除法,端点是浮点数 → `arange` 自动推断返回浮点 tensor**——这里"恰好"类型对了,但依赖隐式推断;`k` 从 1 数到 `d_k/2`,对应公式里 `k ∈ {1,...,d/2}`
- 第 96 行 `theta ** (-(2*k-2)/d_k)`:**标量为底、tensor 为指数**的幂运算——把公式里的"除以 `Θ^{(2k-2)/d}`"改写成"乘以负指数幂",一行算出整串频率
- 第 97 行 `einsum(position, frequency, 'i,j -> i j')`:外积——两个输入名字**不同**(`i`、`j`)且**都保留**在输出里 → 不收缩任何维,`rope[i][j] = position[i] * frequency[j]`,正是角度表 `θ_{i,k}`
- 第 98-99 行:`persistent=False` 已加;`torch.cos`/`torch.sin` 逐元素作用在整张角度表上

### RoPE:外积的三种写法对比

```python
einsum(position, frequency, 'i,j -> i j')          # einops:名字不同+都保留 = 外积
position[:, None] * frequency[None, :]              # 原生广播:列 × 行 自动扩成矩阵
torch.outer(position, frequency)                    # 专用函数,最短
```

三者完全等价。einsum 版本的意义在于"不收缩"这个语义直接写在了模式里;广播版本要求读者心算 `None`(即 `unsqueeze`)的效果;`torch.outer` 最直接但只支持一维输入。

### RoPE:`forward` 逐行要点

- 第 106-107 行 `self.rope_cos[token_positions]`:花式索引查表(和 Embedding 同一机制)——`(..., seq_len)` 的位置索引 `(max_seq_len, d_k/2)` 的表,得到 `(..., seq_len, d_k/2)`;**不能假设位置是 `0,1,2,...`**,增量生成/padding 场景下两者不等价
- 第 108 行 `x[..., 0:-1:2]`:功能正确,但 `-1` 是**多余的**——`d_k` 是偶数,最后一个下标是奇数,`0::2` 本来就取不到它,`0:-1:2` 和 `0::2` 结果相同,后者更简洁(第 109 行 `1::2` 就是这种风格)
- 第 110-111 行:2D 旋转公式 `y1 = x1·cos − x2·sin`、`y2 = x1·sin + x2·cos`——四个操作数形状都是 `(..., seq_len, d_k/2)`,纯逐元素运算
- 第 112 行 `torch.stack([y_1, y_2], dim=-1)`:新增最后一维把每对 `(y1_i, y2_i)` 紧挨着放,形状 `(..., seq_len, d_k/2, 2)`
- 第 113 行 `rearrange(temp, '... a b -> ... (a b)')`:合并最后两维,恢复交替排列 `y1_0, y2_0, y1_1, y2_1, ...`——`...` 前缀是它能容纳任意批量维的关键

### RoPE:拆对与合并的 einops 对比

```python
# 拆出每对的两个成员
x[..., 0::2], x[..., 1::2]                                    # 原生:步长切片
pairs = rearrange(x, '... (n two) -> ... n two', two=2)        # einops:显式暴露配对维
pairs[..., 0], pairs[..., 1]                                    # 再取每对成员,与上排等价

# 合并回交替排列
rearrange(temp, '... a b -> ... (a b)')                        # einops(当前写法)
temp.flatten(start_dim=-2)                                      # 原生:天然适配任意维数
```

`rearrange` 版本把"最后一维是 n 组每组 2 个"的结构写成了文档;`flatten(start_dim=-2)` 不用写模式串、不会有维度数不匹配的问题。两边都对,选可读性偏好即可。

---

## 六、`softmax`

### softmax:踩过的坑

1. **`x - torch.max(x, dim=i, keepdim=True)` 报 `TypeError`**——带 `dim` 的 `max` 返回**命名元组** `(values, indices)`,不是 tensor,不能直接参与运算;必须 `[0]` 或 `.values` 取出数值部分(当前代码第 116 行已用 `[0]` 修正)

### softmax:逐行要点

- 第 116 行 `torch.exp(x - torch.max(x, dim=i, keepdim=True)[0])`:
  - 减最大值 = 数值稳定技巧(softmax 对整体平移不变,让最大值变 0,`exp` 不会溢出成 `inf`)
  - `keepdim=True` 让最大值形状 `(..., 1)`,减法正确广播到 `x` 的对应维——没有它,正方形形状下会"碰巧广播成功但对齐错维度",不报错、结果全错
  - `exp` 是逐元素函数,**没有也不需要 `dim`**——作用在整个张量上不会"波及"别的维度;只有 `max`/`sum` 这类归约才需要指定维度
  - 辨析:**不带 `dim` 的 `torch.max(t)` 返回单个标量 tensor**(无下标),和带 `dim` 版本返回类型不同
- 第 117 行 `torch.sum(shifted, dim=i, keepdim=True)`:分母只在第 `i` 维内求和(其余维度每种组合独立归一化);同样必须 `keepdim=True` 供下一行除法广播。变量名 `denomitor` 拼错了(应为 `denominator`)——不影响运行,风格问题
- 第 118 行:逐元素除法,输出形状与输入相同,沿第 `i` 维求和处处为 1
- 整体:函数**不依赖输入总维数**——1 维 logits、2 维 batch、4 维注意力分数 `(batch, heads, seq, seq)` 全部适用,因为 `dim`/`keepdim` 的写法天然维度无关

### softmax:einops 对比

```python
torch.max(x, dim=i, keepdim=True)[0]              # 原生:要记得 [0] 拆元组
reduce(x, '... d -> ... 1', 'max')                # einops:只返回数值,没有元组问题(仅当 i 是最后一维)

torch.sum(shifted, dim=i, keepdim=True)           # 原生
reduce(shifted, '... d -> ... 1', 'sum')          # einops 等价
```

注意局限:`reduce` 的模式串按**位置**描述轴,`'... d -> ... 1'` 只能表达"最后一维"这种固定位置——像本函数这样 `dim=i` 是**运行时参数**、任意维都可能被归约的场景,原生 `dim=i` 写法反而是唯一自然的选择。这是"einops 并非处处更优"的一个具体例子。

---

## 七、贯穿始终的通用知识点

### 通用:`nn.Module` 相关

- `__init__` 第一行必须 `super().__init__()`,否则 `Parameter`/子模块赋值都会报错
- 只有 `nn.Parameter`/子模块赋值被自动追踪(`.parameters()`、`.to(device)`、`state_dict()`);普通属性是"隐形"的;`register_buffer` 介于两者之间(跟着走、不训练),`persistent=False` 再免除存档
- `load_state_dict` 按名字拷贝值,嵌套 key 用点号路径;报错信息(Missing/Unexpected/size mismatch)会精确指出命名或形状问题
- 调用子模块用 `module(x)` 不用 `.forward(x)`

### 通用:张量操作

- 归约(`sum`/`mean`/`max`)靠 `dim` 选维、`keepdim` 保形——归约结果还要与原张量运算时,`keepdim=True` 几乎总是必须的
- 带 `dim` 的 `max`/`min` 返回命名元组,要 `[0]`/`.values` 拆;`sum`/`mean` 不存在这个问题
- 逐元素函数(`exp`/`sqrt`/`sigmoid`/`sin`/`cos`)无 `dim` 概念,整张量独立处理
- 花式索引(`table[ids]`)是查表类操作的统一机制(Embedding、RoPE 查 cos/sin 表)
- `...` 是"保留任意前置维度"的通配:索引里(`x[..., 0::2]`)和 einops 模式里(`'... a b -> ... (a b)'`)语义一致
- `x[-1]` 砍第一维、`x[..., -1]` 取最后一维——一字之差,前者丢批量数据

### 通用:einops 使用心得(本文件所有对比的总结)

- **einsum 的威力**:轴名自动对齐收缩,免手写转置(`Linear`);"名字不同+都保留"即外积(RoPE 角度表)
- **reduce 的威力**:`'-> ... 1'` 替代 `keepdim=True`,归约意图写进模式串
- **rearrange 的威力**:`(n two)` 这类分组语法把"两两配对"的结构写成文档
- **硬性规则**:模式串声明几个轴,输入必须恰好几维;`...` 是唯一的任意维数通配,带批量维的场景几乎总要带上它
- **einops 做不了的**:索引/查表(Embedding)、逐元素运算(SiLU 门控)、以 `dim` 为运行时参数的归约(softmax 的 `dim=i`)——这些场景原生写法是唯一或更自然的选择

### 通用:测试与调试方法论

- 测试通过 ≠ 代码没问题:只证明"测试覆盖到的输入上结果正确"。形状对、数值错的 bug(`sum`/`mean` 之差、`keepdim` 缺失的错误广播)靠形状检查发现不了,必须数值比对
- 报错信息往往已经精确指出问题(缺失的 key、互换的形状、不可调用的类型),仔细读报错本身是最快的排查方式
- 语法错误(如关键字/位置参数混用)会让整个文件无法导入、**所有**测试文件连带崩溃——先保证文件能被解析,再谈逻辑对错
