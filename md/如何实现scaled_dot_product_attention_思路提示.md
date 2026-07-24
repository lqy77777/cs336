# 如何实现 `scaled_dot_product_attention`(思路提示)

> 这是实现思路的提示,不含具体代码。具体怎么写需要自己完成。

## 公式回顾

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V
$$

- $Q \in \mathbb{R}^{n \times d_k}$,$K \in \mathbb{R}^{m \times d_k}$,$V \in \mathbb{R}^{m \times d_v}$
- `Q`、`K`、`V` 都是这个操作的**输入**,不是可学习参数本身

## 两个容易忽略的概念点

**`d_k` 和 `d_v` 不需要相等**:`Q`/`K` 必须共享 `d_k`(点积要求最后一维对齐),但 `V` 用自己独立的 `d_v`——两者角色不同(`d_k` 决定"怎么算相关性",`d_v` 决定"实际内容有多少维")。讲义给函数规格时特意用了两个不同的名字,实现时不要假设它们相等。

**`n`(query 数量)和 `m`(key/value 数量)也不需要相等**:自注意力场景下两者恰好都等于 `seq_len`,但 attention 这个操作本身更通用(交叉注意力、KV cache 场景下两者会不一样)。`K` 和 `V` 的数量永远一致(配对的),但和 `Q` 的数量是独立的。

## 函数接口

讲义要求:

- 输入:`Q`、`K` 形状 `(batch_size, ..., seq_len, d_k)`,`V` 形状 `(batch_size, ..., seq_len, d_v)`,`...` 是任意数量的额外批量维
- 可选参数:一个布尔 mask,形状 `(seq_len, seq_len)`,`True` 表示"可以看"、`False` 表示"看不到"
- 输出:`(batch_size, ..., seq_len, d_v)`

## 拆成四步

### 第一步:算 `QK^T / sqrt(d_k)`

`K` 需要转置成 `(..., d_k, seq_len)` 才能和 `Q` 做矩阵乘法——但 `K` 带批量维,**不能用 `.T`**(只对二维安全,高维会反转所有维度)。需要只转置**最后两维**、不动前面批量维的写法(提示:`PyTorch_Tensor基础.md` 里补充过这个知识点,有专门处理最后两维转置的属性/方法)。

`@` 本身会自动处理批量维,转置转对了地方,这一步不需要额外操心批量维对齐。

`sqrt(d_k)` 里 `d_k` 是普通整数,不是 tensor。

### 第二步:如果给了 mask,把 `False` 的位置设成 `-inf`

要在**softmax 之前**,把分数矩阵里 `mask` 为 `False` 的位置替换成 `-inf`。这是"按条件选择性替换某些位置的值"的操作——查一下 PyTorch 有没有专门做"按布尔条件从两个来源里选值"或者"按布尔条件填充某个常数"的函数,注意条件要不要取反。

另外想一下:`mask` 形状 `(seq_len, seq_len)`,分数矩阵形状 `(batch_size, ..., seq_len, seq_len)`,直接参与运算会不会自动广播对齐?回忆广播规则"从最后一维开始比较"。

### 第三步:对处理过的分数矩阵做 softmax

复用你自己写好的 `softmax(x, dim)` 函数。想一下该对哪一维做——这个分数矩阵最后一维代表"这个 query 能看到的所有 key",要归一化的正是这一维。

### 第四步:和 `V` 相乘

softmax 算出来的权重(`..., seq_len, seq_len`)和 `V`(`..., seq_len, d_v`)做矩阵乘法,得到 `(..., seq_len, d_v)`。

## 建议的验证方式

1. 先不加 mask,拿一个不带批量维的小例子(比如 `d_k=2`、`seq_len=3`)手动算一遍,检查输出形状和一两个具体数值
2. 再测试加了 mask 的情况,确认被屏蔽的位置权重确实是 0(讲义要求:mask 为 `True` 的位置权重加起来是 1,`False` 的位置权重是 0)
3. 分别测试三维输入和四维输入(讲义提到 `test_scaled_dot_product_attention` 测三阶张量,`test_4d_scaled_dot_product_attention` 测四阶张量),确认批量维数量不影响正确性
