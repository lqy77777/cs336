# 第 3 节:Transformer 语言模型架构 —— 知识整理

> 依据:`cs336_assignment1_basics.pdf` 第 13-27 页(3 Transformer Language Model Architecture)。本文件只讲**概念、数学公式、设计动机**,不含实现细节或代码。

## 3.0 语言模型的输入输出是什么

语言模型接收一批(batched)整数 token ID 序列(`torch.Tensor`,形状 `(batch_size, sequence_length)`),输出一个在词表上的(批量)归一化概率分布(形状 `(batch_size, sequence_length, vocab_size)`),表示"给定前缀,下一个词是什么"的预测。

- **训练时**:用这些"下一词预测"和真实的下一个词计算交叉熵损失(这是第 4 节的内容,不在本文件范围)
- **推理(生成文本)时**:只取序列**最后一个位置**的预测分布,从中选一个 token(取最大概率、或采样),接到输入序列末尾,重复这个过程

## 3.1 Transformer LM 整体结构

给定 token ID 序列:

1. **输入嵌入(Input Embedding)**:把整数 ID 转成稠密向量
2. 依次通过 `num_layers` 个 **Transformer Block**
3. 最后做一次 **归一化(Norm)**
4. 通过一个学到的线性投影("output embedding" / "LM head")得到每个位置的下一词预测 logits

对应讲义 Figure 1:`Inputs → Token Embedding → [Transformer Block] × num_layers → Norm → Linear(Output Embedding) → Softmax → Output Probabilities`

### Token Embeddings
输入 `(batch_size, sequence_length)` 的整数张量,输出 `(batch_size, sequence_length, d_model)` 的向量张量——每个 token ID 被映射成一个 `d_model` 维的向量。

### Pre-norm Transformer Block(整体概念)
标准 decoder-only Transformer 由 `num_layers` 个**结构相同**的 Transformer block 堆叠而成。每个 block 接收 `(batch_size, sequence_length, d_model)`,输出同样形状——**block 内部做两件事**:通过 self-attention **跨序列聚合信息**,通过 feed-forward 层**做非线性变换**。

讲义采用的是 **"pre-norm"** 变体(和原始 Transformer 论文的 "post-norm" 相对),即:归一化放在每个子层的**输入端**,而不是输出端(细节见 3.4 节)。这种设计现在是主流 LLM(GPT-3、LLaMA、PaLM 等)采用的标准做法。

## 3.2 备注:批处理(Batching)、Einsum 与高效计算

Transformer 里反复出现"对很多个批量维度做同样的运算"这个模式,常见的三种"批量维"是:

- **batch 内的样本**:对每个样本跑一样的 forward
- **序列长度**:RMSNorm、feed-forward 这类"逐位置"操作,对序列里每个位置做的事完全一样
- **attention head**:多头 attention 本质上是把 attention 操作在多个 head 上批量执行

讲义**强烈推荐使用 einsum 记号**(`torch.einsum`,或更框架无关的 `einops`/`einx` 库),原因是:

- 大多数机器学习运算,都可以归结为"维度调整(reshape/transpose 之类) + 张量收缩(矩阵乘法之类) + 偶尔的逐元素非线性函数"这三件事的组合
- `einops.rearrange`:重排/拼接/拆分任意维度
- `einops.einsum`:做任意维度的张量收缩
- Einsum 记法是**自解释的**(self-documenting)——直接把每个维度的名字写在表达式里,比一堆 `view`/`reshape`/`transpose` 更清楚、更不容易出 shape 错误

讲义给了三个例子(`einstein_example1/2/3`),分别展示:批量矩阵乘法、广播逐元素乘法、跨通道的像素混合线性变换——核心思想都是"用维度名字代替死记硬背的维度顺序"。

### 3.2.1 数学记号与内存排布

数学写法上有两种惯例:

- **行向量**记号:`y = xW^T`,`W ∈ R^{d_out × d_in}`,`x ∈ R^{1 × d_in}`——批量化时在 `x` 最外层加 batch 维
- **列向量**记号:`y = Wx`,`x ∈ R^{d_in}`——批量化时 batch 维要放在**最后**

讲义**主要采用列向量记号**(`y = Wx`),这是数学/线性代数里更常见的写法。但要注意:PyTorch/NumPy 默认是**行主序(row-major)**内存排布,如果用纯矩阵乘法记号(而不是 einsum),需要按行向量惯例(`y = xW^T`)加转置。**用 einsum 的话,只要维度轴标对了名字,这个问题就不存在**——这是 einsum 记号的又一个好处。

## 3.3 基础组件:Linear 和 Embedding

### 3.3.1 参数初始化

讲义给出的具体初始化方案(后续作业会详细讨论原理,这里先照用):

| 参数类型 | 分布 | 截断范围 |
|---|---|---|
| Linear 权重 | `N(μ=0, σ²=2/(d_in+d_out))` | `[-3σ, 3σ]` |
| Embedding | `N(μ=0, σ²=1)` | `[-3, 3]` |
| RMSNorm 增益 | 全 1 | 无 |

应使用 `torch.nn.init.trunc_normal_` 来实现截断正态分布初始化。

### Linear 模块(概念)
`y = Wx`——标准线性变换,**不含 bias 项**(跟随现代主流 LLM 的做法)。要求:
- 继承 `torch.nn.Module`
- 参数存成 `W`(不是 `W^T`),用 `nn.Parameter` 包装
- 不能用 `nn.Linear` 或 `nn.functional.linear`

### Embedding 模块(概念)
输入 token ID(`torch.LongTensor`,形状 `(batch_size, sequence_length)`),从形状为 `(vocab_size, d_model)` 的嵌入矩阵里按 ID 索引取出对应向量。要求同样是自己实现(不能用 `nn.Embedding`)。

## 3.4 Pre-Norm Transformer Block

每个 Transformer block 有两个子层:**多头自注意力**、**逐位置前馈网络**。原始 Transformer 论文用的是"post-norm"(归一化在子层输出之后)。讲义采用的是"**pre-norm**"变体——归一化放在每个子层的**输入端**,子层的输出通过残差连接直接加回主干,外加**最后一个 block 之后再做一次额外的归一化**。

Pre-norm 的直觉:从输入嵌入到最终输出之间,存在一条不经过任何归一化的"干净残差流(residual stream)",据信能改善梯度流动。这是当前 LLM(GPT-3、LLaMA、PaLM 等)的标准架构。

### 3.4.1 RMSNorm(均方根层归一化)

比标准 LayerNorm 更简单的归一化方式,不做"减均值"这一步:

```
RMSNorm(a_i) = (a_i / RMS(a)) * g_i
RMS(a) = sqrt( (1/d_model) * Σ a_i² + ε )
```

其中 `g_i` 是**可学习的增益参数**(一共 `d_model` 个),`ε` 是防止除零的小常数(常取 `1e-5`)。

**数值稳定性要点**:计算前需要把输入**上采样(upcast)到 `float32`**(防止平方运算时溢出),计算完再转回原来的 dtype。

### 3.4.2 逐位置前馈网络(Position-Wise Feed-Forward)

原始 Transformer 用两层线性变换夹一个 ReLU,中间层维度通常是输入维度的 4 倍。**现代 LLM 的两个改进**:换用不同的激活函数,并引入门控机制。

**SiLU(又叫 Swish)激活函数**:
```
SiLU(x) = x · σ(x) = x / (1 + e^{-x})
```
形状和 ReLU 相似,但在 0 附近是光滑的(不像 ReLU 有一个尖角)。

**GLU(Gated Linear Unit,门控线性单元)**:
```
GLU(x, W1, W2) = σ(W1·x) ⊙ W2·x
```
`⊙` 表示逐元素乘法。GLU 被认为能"通过提供一条线性的梯度通路,缓解深层网络的梯度消失问题,同时保留非线性能力"。

**SwiGLU**(把 SiLU 和 GLU 结合,Llama 3、Qwen 2.5 等模型采用):
```
FFN(x) = SwiGLU(x, W1, W2, W3) = W2 · (SiLU(W1·x) ⊙ W3·x)
```
其中 `x ∈ R^{d_model}`,`W1, W3 ∈ R^{d_ff × d_model}`,`W2 ∈ R^{d_model × d_ff}`,惯例上 `d_ff = (8/3) × d_model`,实现时取一个方便硬件计算的、离这个值最近的 64 的倍数。

讲义引用了提出 SwiGLU 的论文里一句很实在的话:"我们没有解释为什么这些架构选择有效;把它们的成功归因于神圣的恩典(divine benevolence)"——提醒你这类设计选择更多是靠实验验证,而不是有严格的理论推导。

### 3.4.3 相对位置编码:RoPE(Rotary Positional Embedding)

**目的**:给模型注入"token 在序列里的位置"这个信息。

**核心思想**:对位置 `i` 处的 query 向量 `q^(i) = W_q x^(i) ∈ R^d`,应用一个**依赖位置的旋转矩阵** `R^i`,得到 `q'^(i) = R^i q^(i)`。`R^i` 把 embedding 里的每一对相邻元素 `(q_{2k-1}, q_{2k})` 当作一个 2D 向量,旋转角度为:

```
θ_{i,k} = i / Θ^{(2k-2)/d}      对 k ∈ {1, ..., d/2}
```

`Θ` 是一个常数超参数。每一对 2D 分量对应一个 2×2 的旋转子矩阵:

```
R_k^i = [ cos(θ_{i,k})   -sin(θ_{i,k}) ]
        [ sin(θ_{i,k})    cos(θ_{i,k}) ]
```

完整的 `R^i` 是这些 2×2 子块拼成的**块对角矩阵**(其余位置全 0)。

**实现上的效率提示**:不需要真的构造出完整的 `d×d` 矩阵——可以利用块对角结构直接实现旋转。`cos(θ_{i,k})`、`sin(θ_{i,k})` 这些值在同一序列内、跨层、跨 batch 都是**可复用**的,可以只算一次、存成缓冲区(用 `self.register_buffer(persistent=False)`,而不是 `nn.Parameter`,因为这些值不需要被训练)。**Key 向量 `k^(j)` 也要做完全相同的旋转过程**(用各自的位置 `j`)。**这一层没有可学习参数**。

### 3.4.4 Scaled Dot-Product Attention(缩放点积注意力)

**Softmax**(先作为一个独立组件):
```
softmax(v)_i = exp(v_i) / Σ_j exp(v_j)
```

**数值稳定性**:`exp(v_i)` 对较大的 `v_i` 会变成 `inf`,导致 `inf/inf = NaN`。利用 softmax 对"给所有输入加同一个常数"不变的性质,通常做法是先**减去这一维上的最大值**,让新的最大值变成 0,再算 softmax。

**Attention 操作**:
```
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) · V
```
`Q ∈ R^{n×d_k}`,`K ∈ R^{m×d_k}`,`V ∈ R^{m×d_v}`——注意 Q、K、V 是**这个操作的输入**,不是可学习参数本身(可学习参数是产生 Q/K/V 的那些投影矩阵)。

**Masking(掩码)**:mask 是一个形状 `(n, m)` 的布尔矩阵,`(i,j)` 位置为 `True` 表示 query `i` **可以**看到(attend to)key `j`。计算上更高效的做法不是真的去截断子序列,而是:在 softmax **之前**,把预 softmax 的分数矩阵里,mask 为 `False` 的位置**加上 `-∞`**——这样 softmax 之后这些位置的权重自然变成 0。

### 3.4.5 因果多头自注意力(Causal Multi-Head Self-Attention)

**多头注意力**:
```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h)
head_i = Attention(Q_i, K_i, V_i)
```
`Q_i`、`K_i`、`V_i` 是 `Q`、`K`、`V` 沿着 embedding 维度切出的第 `i` 份(每份大小 `d_k` 或 `d_v`)。

**多头自注意力**(接上输入/输出投影):
```
MultiHeadSelfAttention(x) = W_O · MultiHead(W_Q x, W_K x, W_V x)
```
`W_Q, W_K ∈ R^{h·d_k × d_model}`,`W_V ∈ R^{h·d_v × d_model}`,`W_O ∈ R^{d_model × h·d_v}`。因为 Q/K/V 会被切分到各个 head,可以把 `W_Q`、`W_K`、`W_V` 想象成对每个 head 分别存在(总共只需要三次矩阵乘法就能同时算出所有 head 的投影)。

**因果掩码(Causal Masking)**:语言模型不能"偷看"未来的 token——预测位置 `i` 的下一词时,只能用到 `t_1, ..., t_i`,不能用到 `t_{i+1}, ..., t_n`(否则训练目标会被信息泄露"作弊")。做法是让 token `i` **只能 attend 到位置 `j ≤ i`** 的 token,可以用 `torch.triu` 或者广播式的下标比较构造这个 mask,并复用已经实现好的、支持 masking 的 scaled dot-product attention。

**RoPE 的应用位置**:RoPE 只作用在 **query 和 key** 向量上,**不作用在 value 向量**上。多头场景下,"head"这个维度要被当作一个额外的"批量维"来处理——也就是说,每个 head 各自独立地对自己的 query/key 做同样的 RoPE 旋转(用相同的位置信息)。

## 3.5 完整的 Transformer LM

### 组装一个 Transformer Block

Block 的第一个子层(多头自注意力部分):
```
y = x + MultiHeadSelfAttention(RMSNorm(x))
```
第二个子层(feed-forward 部分)结构类似:先 RMSNorm,再过 SwiGLU FFN,再残差相加。这正是"pre-norm"的具体体现——归一化在子层**输入**处,主干通过残差连接直接传递。

### 组装完整模型

按 Figure 1:输入 → embedding → `num_layers` 个 Transformer block → 最终的 norm → LM head(线性层)→ 未归一化的 logits(在词表维度上)。

## 资源核算(Resource Accounting)背景知识

Transformer 里绝大多数 FLOPs 来自矩阵乘法。核算方法:

1. 把 forward pass 里所有矩阵乘法列出来
2. 用下面这条规则把每个矩阵乘法换算成 FLOPs

**规则**:`A ∈ R^{m×n}`、`B ∈ R^{n×p}`,矩阵乘法 `AB` 需要 **`2mnp`** 次浮点运算(FLOPs)。

推导:`(AB)[i,j] = A[i,:]·B[:,j]`,这个点积需要 `n` 次乘法 + `n` 次加法(共 `2n` FLOPs);`AB` 一共有 `m×p` 个元素,所以总数是 `(2n)(mp) = 2mnp`。

这条规则是后面分析 GPT-2 XL 等具体模型配置的 FLOPs 时唯一需要的核心工具——对模型里每一处矩阵乘法(QKV 投影、`QK^T`、加权求和、输出投影、FFN 里的三个矩阵乘)分别应用这条规则、再求和。
