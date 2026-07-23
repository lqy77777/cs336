# 如何实现 RoPE Class(思路提示)

> 这是实现思路的提示,不含具体代码——配合 [RoPE旋转位置编码详解.md](RoPE旋转位置编码详解.md) 的原理讲解一起看。具体怎么写需要自己完成。

## 推荐接口(讲义给出)

```python
def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None)
def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor
```

## `__init__`:预先算好 cos/sin 表

**目标**:算出一张形状为 `(max_seq_len, d_k/2)` 的表,每一行对应一个位置 `i`,每一列对应一个分组 `k`,存的是 `θ_{i,k}` 的 cos 值和 sin 值(两张表)。

需要解决的子问题:

1. **怎么生成分组编号 `k` 对应的那部分指数**:公式是 `θ_{i,k} = i / Θ^{(2k-2)/d}`。想一下用 `torch.arange` 怎么生成 `k = 0, 1, ..., d_k/2 - 1`(注意 Python 习惯从 0 开始编号,对应公式里的 `k=1,...,d/2`,指数部分要相应调整),再算出 `Θ^{(2k-2)/d}` 这一串数(每个 `k` 一个值,形状 `(d_k/2,)`)
2. **怎么生成位置序列**:`i = 0, 1, ..., max_seq_len - 1`,同样用 `torch.arange`
3. **怎么把"位置"和"每组的频率"组合成一张二维表**:有一个长度 `max_seq_len` 的位置序列、一个长度 `d_k/2` 的频率序列,想要的是形状 `(max_seq_len, d_k/2)` 的角度表,每个位置 `(i,k)` 存的是 `i / Θ^{(2k-2)/d}`。可以用广播机制——一个 `(max_seq_len, 1)` 的列和一个 `(1, d_k/2)` 的行相乘,会自动广播成 `(max_seq_len, d_k/2)`;或者用 `einsum`/`einops.einsum` 表达"两个一维向量的外积"这种操作
4. **对这张角度表分别取 `cos`、`sin`**,存成两个 buffer——用 `register_buffer(persistent=False)`,因为这些值不需要训练、也不需要存进 checkpoint

## `forward`:两步——先取出对应位置的 cos/sin,再做旋转

### 第一步:根据 `token_positions` 取出正确的 cos/sin 行

讲义提醒过,不能想当然假设位置就是 `0,1,2,...`,而要用 `token_positions` 里给的实际值去查表。用一个整数 tensor 去索引另一个 tensor 的某一维("花式索引",和 `Embedding` 查表用的是同一个技巧):`token_positions` 形状是 `(..., seq_len)`,预先算好的 cos/sin 表形状是 `(max_seq_len, d_k/2)`,用 `token_positions` 去索引这张表的**第一维**,取出形状 `(..., seq_len, d_k/2)` 的结果。

### 第二步:把 `x` 拆成"每对里的第一个"和"第二个"分量,分别做旋转

讲义的配对方式是**相邻两个元素**为一对:`(q_1,q_2), (q_3,q_4), ...`,不是"前一半、后一半"这种切法。PyTorch 的切片支持指定"步长"(`x[start:stop:step]`),想一下怎么用这种写法从 `x` 的最后一维里,分别取出"所有奇数位置的元素"和"所有偶数位置的元素"这两组,各自形状 `(..., seq_len, d_k/2)`。

拿到这两组分量(记作 `x1`、`x2`)之后,套用 2D 旋转公式:

$$
x_1' = x_1 \cos\theta - x_2 \sin\theta, \qquad x_2' = x_1 \sin\theta + x_2 \cos\theta
$$

### 第三步:把 `x1'`、`x2'` 按原来交替的顺序拼回一个 `d_k` 维的结果

这一步需要把两个 `(..., d_k/2)` 的张量,重新"交织"回一个 `(..., d_k)` 的张量(顺序是 `x1'_0, x2'_0, x1'_1, x2'_1, ...`)。想一下 `torch.stack` 配合 `reshape`/`view` 能不能拼出这个交替排列(先把两个张量在一个新维度上堆起来,再把最后两维合并成一维,顺序应该正好是交替的)。

## 建议的验证方式

先用最小的例子手动验证:`d_k=2`(只有一对),`max_seq_len` 随便设几个,构造一个简单的 `x` 和对应的 `token_positions`,手算一下某个具体位置该旋转成什么样,和代码算出来的结果对一下。确认 `d_k=2` 的简单情况对了,再考虑更大的 `d_k` 和带批量维的情况。
