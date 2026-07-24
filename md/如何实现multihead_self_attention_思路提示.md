# 如何实现 3.4.5 Causal Multi-Head Self-Attention(思路提示)

> 这是实现思路的提示,不含具体代码。具体怎么写需要自己完成。

## 这一节在整体架构里的位置

- 3.4.4 实现的是通用的 `scaled_dot_product_attention`:给定任意 `Q`、`K`、`V`,算注意力,不关心它们从哪来。
- 3.4.5 要在这基础上构建"多头自注意力",拆开看是三个概念叠在一起:
  1. **自注意力(self-attention)**:`Q`、`K`、`V` 都来自同一个输入 `x`(区别于交叉注意力从两个不同来源产生 `Q` 和 `K`/`V`)
  2. **多头(multi-head)**:把 `d_model` 维的表示切成 `h` 个头,每个头独立做一次注意力,再把结果拼接起来
  3. **因果(causal)**:每个位置只能看到自己和之前的位置,不能看到未来——这是自回归语言模型训练时的硬性要求

## 公式回顾

$$
\text{MultiHead}(Q,K,V) = \text{Concat}(\text{head}_1,...,\text{head}_h), \quad \text{head}_i = \text{Attention}(Q_i,K_i,V_i)
$$

$$
\text{MultiHeadSelfAttention}(x) = W_O \cdot \text{MultiHead}(W_Q x, W_K x, W_V x)
$$

可学习参数:$W_Q, W_K \in \mathbb{R}^{hd_k \times d_{model}}$,$W_V \in \mathbb{R}^{hd_v \times d_{model}}$,$W_O \in \mathbb{R}^{d_{model} \times hd_v}$。按讲义约定 $d_k = d_v = d_{model}/h$。

## 核心思路:三次矩阵乘法,而不是循环 `h` 次

讲义原文提示:"you should be computing the key, value, and query projections as a total of three matrix multiplies"。

也就是说:**不要**写类似"对每个头单独做一次投影,循环 `h` 次"的写法——而是一次性算出整个 `W_Q x`(输出维度是 `h*d_k`),再把这个大向量**切分**成 `h` 份分给每个头。

想一下:用什么操作能把形状 `(..., seq_len, h*d_k)` 的张量,变成 `(..., h, seq_len, d_k)`?——这需要先把最后一维拆成 `(h, d_k)` 两部分,再把 `h` 这个新轴挪到 `seq_len` 前面(让它变成又一个"批量维")。回忆你在 RoPE 里已经用过 `rearrange` 做过"拆分/合并"操作(`'... a b -> ... (a b)'` 那个方向),现在需要的是类似操作的逆方向,外加一次维度换位。

这样处理之后,你其实可以**直接复用**你已经写好的 `scaled_dot_product_attention`——因为它的签名允许任意数量的前置批量维 `...`,而"头"正好可以被当成又一个批量维,不需要为多头单独重写注意力逻辑。

## Causal mask 怎么构造

讲义提示:可以用 `torch.triu`,也可以用广播的下标比较(broadcasted index comparison)。

- 回忆 3.4.4 里 mask 的语义:`(i,j)` 位置为 `True` 表示"query i 可以看 key j"。causal 要求:`j <= i` 时为 `True`,`j > i` 时为 `False`。
- 如果用 `torch.triu`:查一下它的 `diagonal` 参数如何控制"保留哪条对角线以上/以下的元素",想清楚你要的是上三角还是下三角、要不要包含对角线本身,以及它默认填的是什么值(和你需要的布尔语义是否一致,需不需要转换)。
- 如果用广播下标比较:构造两个从 `0` 到 `seq_len-1` 的下标序列,一个代表 query 位置、一个代表 key 位置,用 `<=` 比较,直接广播出一个布尔矩阵。想一下这两个序列的形状分别应该是什么(比如一个是列向量、一个是行向量),才能广播成 `(seq_len, seq_len)`。

mask 算出来之后,直接传给你已经写好的 `scaled_dot_product_attention` 的 `mask` 参数即可,不需要重新实现"填 `-inf`"那部分逻辑。

## RoPE 怎么套进来

讲义明确提示两点:

1. RoPE 只作用于 `Q` 和 `K`,**不**作用于 `V`。
2. "头"这一维要被当成一个批量维——因为多头注意力里每个头是独立做注意力的,而 RoPE 的旋转本身只依赖"位置",跟"是哪个头"无关,所以同一份旋转应该不加区分地应用到每个头上。

结合你 `RotaryPositionalEmbedding.forward` 的签名(`x`, `token_positions`)以及它对 `...` 批量维的支持,想一下:应该在"拆分成 `h` 个头"**之后**调用 RoPE(这样输入形状里自然带有"头"这个批量维,靠 `...` 自动处理),还是拆分之前?哪种顺序能让 RoPE 里 `d_k` 这个参数、以及它对任意数量前置批量维的支持,自然地兼容"头"这个新增维度?

## 建议的实现步骤拆分

1. `__init__`:创建三个 `Linear(d_model, h*d_k)` 分别对应 Q/K/V 投影,和一个 `Linear(h*d_v, d_model)` 对应输出投影 `W_O`(回忆 3.3.2 的结论,要不要偏置)。
2. `forward(x)`:
   - 分别算 `W_Q x`、`W_K x`、`W_V x`(三次矩阵乘法,而不是每个头单独算一次)
   - 用 `rearrange` 把三者的最后一维拆成 `(h, d_k)` 或 `(h, d_v)`,并把头维挪到序列维之前
   - (如果这次作业要求)对 `Q`、`K` 做 RoPE 旋转
   - 构造 causal mask
   - 调用你已经写好的 `scaled_dot_product_attention`,传入 `Q`、`K`、`V`、`mask`
   - 把输出的"头"维和 `d_v` 维重新拼回一个 `h*d_v` 维(前面 `rearrange` 操作的逆过程)
   - 过 `W_O` 投影,得到 `(..., seq_len, d_model)`

## 建议的验证方式

1. 每一步打印/检查中间张量的形状是否符合预期(比如拆分成头之后应该是 `(batch, h, seq_len, d_k)`)
2. 对一个很小的 `seq_len`(比如 4)手动检查你构造的 causal mask 是否恰好是预期的上/下三角形状
3. 确认 `h * d_k == d_model`——如果用 `d_k = d_model // h`,注意能否整除
4. 运行 `uv run pytest -k test_multihead_self_attention` 验证
