# 第 3 节:Transformer 语言模型架构 —— 实现清单

> 依据:`cs336_assignment1_basics.pdf` 第 13-27 页。本文件只列出**需要实现什么**(接口、参数、返回值),不含公式推导或实现细节——公式部分见 [第3节_Transformer架构知识整理.md](第3节_Transformer架构知识整理.md)。

## 总览:11 个 Problem,共 29 分

| # | Problem | 分值 | 类型 |
|---|---------|------|------|
| 1 | `linear` | 1 | 代码 |
| 2 | `embedding` | 1 | 代码 |
| 3 | `rmsnorm` | 1 | 代码 |
| 4 | `positionwise_feedforward` | 2 | 代码 |
| 5 | `rope` | 2 | 代码 |
| 6 | `softmax` | 1 | 代码 |
| 7 | `scaled_dot_product_attention` | 5 | 代码 |
| 8 | `multihead_self_attention` | 5 | 代码 |
| 9 | `transformer_block` | 3 | 代码 |
| 10 | `transformer_lm` | 3 | 代码 |
| 11 | `transformer_accounting` | 5 | 书面题(a-e 五问) |

每个代码类的 Problem,都要求:实现对应的类/函数 → 在 `tests/adapters.py` 里补全对应的 `run_xxx` adapter(胶水代码)→ 跑 `uv run pytest -k test_xxx` 验证。

---

## 1. `linear`(1 分)—— Linear 模块

实现一个 `Linear` 类,继承 `torch.nn.Module`,接口对齐 PyTorch 内置的 `nn.Linear`,但**没有 bias**。

**`__init__(self, in_features, out_features, device=None, dtype=None)`**

| 参数 | 类型 | 含义 |
|------|------|------|
| `in_features` | `int` | 输入的最后一维大小 |
| `out_features` | `int` | 输出的最后一维大小 |
| `device` | `torch.device \| None` | 参数存放的设备 |
| `dtype` | `torch.dtype \| None` | 参数的数据类型 |

**`forward(self, x: torch.Tensor) -> torch.Tensor`**:对输入做线性变换。

**要求**:
- 继承 `nn.Module`,调用父类构造函数
- 权重存成 `W`(不是 `W^T`),用 `nn.Parameter` 包装
- 不能用 `nn.Linear` 或 `nn.functional.linear`
- 用 `torch.nn.init.trunc_normal_` 按讲义给定的分布初始化

**Adapter**:`adapters.run_linear` | **测试**:`uv run pytest -k test_linear`

---

## 2. `embedding`(1 分)—— Embedding 模块

实现一个 `Embedding` 类,继承 `torch.nn.Module`,接口对齐 `nn.Embedding`。

**`__init__(self, num_embeddings, embedding_dim, device=None, dtype=None)`**

| 参数 | 类型 | 含义 |
|------|------|------|
| `num_embeddings` | `int` | 词表大小 |
| `embedding_dim` | `int` | 嵌入向量维度(即 `d_model`) |
| `device` | `torch.device \| None` | 参数存放的设备 |
| `dtype` | `torch.dtype \| None` | 参数的数据类型 |

**`forward(self, token_ids: torch.Tensor) -> torch.Tensor`**:按 token ID 查表,返回对应的嵌入向量。

**要求**:
- 继承 `nn.Module`,调用父类构造函数
- 嵌入矩阵用 `nn.Parameter` 包装,`d_model` 是最后一维
- 不能用 `nn.Embedding` 或 `nn.functional.embedding`
- 同样用 `torch.nn.init.trunc_normal_` 初始化

**Adapter**:`adapters.run_embedding` | **测试**:`uv run pytest -k test_embedding`

---

## 3. `rmsnorm`(1 分)—— RMSNorm

实现 `RMSNorm`,继承 `torch.nn.Module`。

**`__init__(self, d_model, eps=1e-5, device=None, dtype=None)`**

| 参数 | 类型 | 含义 |
|------|------|------|
| `d_model` | `int` | 模型的隐藏维度 |
| `eps` | `float`,默认 `1e-5` | 数值稳定性用的小常数 |
| `device` | `torch.device \| None` | 参数存放的设备 |
| `dtype` | `torch.dtype \| None` | 参数的数据类型 |

**`forward(self, x: torch.Tensor) -> torch.Tensor`**:输入形状 `(batch_size, sequence_length, d_model)`,输出同形状。

**要求**:计算前把输入**上采样到 `float32`**,计算完再转回原 dtype。

**Adapter**:`adapters.run_rmsnorm` | **测试**:`uv run pytest -k test_rmsnorm`

---

## 4. `positionwise_feedforward`(2 分)—— SwiGLU 前馈网络

实现 SwiGLU 前馈网络(SiLU 激活 + GLU 门控)。

**要求**:
- `d_ff` 约等于 `(8/3) × d_model`,取最接近的、能被 64 整除的值(硬件效率考虑)
- 可以用 `torch.sigmoid` 提升数值稳定性

**Adapter**:`adapters.run_swiglu` | **测试**:`uv run pytest -k test_swiglu`

---

## 5. `rope`(2 分)—— 旋转位置编码

实现 `RotaryPositionalEmbedding` 类。

**`__init__(self, theta, d_k, max_seq_len, device=None)`**

| 参数 | 类型 | 含义 |
|------|------|------|
| `theta` | `float` | RoPE 用的 `Θ` 常数 |
| `d_k` | `int` | query/key 向量的维度 |
| `max_seq_len` | `int` | 输入可能达到的最大序列长度 |
| `device` | `torch.device \| None` | 缓冲区存放的设备 |

**`forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor`**

| 参数 | 类型 | 含义 |
|------|------|------|
| `x` | `torch.Tensor`,形状 `(..., seq_len, d_k)` | 需要支持任意数量的前置批量维 |
| `token_positions` | `torch.Tensor`,形状 `(..., seq_len)` | 指定 `x` 里每个位置在序列中的下标 |

返回:和 `x` 同形状的张量。

**要求**:用 `token_positions` 去索引(可能预先算好的)cos/sin 张量。

**Adapter**:`adapters.run_rope` | **测试**:`uv run pytest -k test_rope`

---

## 6. `softmax`(1 分)

写一个函数,对张量的**指定维度**做 softmax。

**接口**:接受一个张量和一个维度下标 `i`,返回同形状的张量,第 `i` 维变成归一化的概率分布。

**要求**:用"减去第 `i` 维最大值"的技巧保证数值稳定。

**Adapter**:`adapters.run_softmax` | **测试**:`uv run pytest -k test_softmax_matches_pytorch`

---

## 7. `scaled_dot_product_attention`(5 分)

实现缩放点积注意力函数。

**要求**:
- 输入:`Q`/`K` 形状 `(batch_size, ..., seq_len, d_k)`,`V` 形状 `(batch_size, ..., seq_len, d_v)`(`...` 代表任意数量的额外批量维)
- 输出:形状 `(batch_size, ..., seq_len, d_v)`
- 支持一个可选的布尔 mask,形状 `(seq_len, seq_len)`——mask 值为 `True` 的位置,attention 概率应正确参与归一化(加总为 1);为 `False` 的位置,概率应为 0

**Adapter**:`adapters.run_scaled_dot_product_attention`
**测试**:`uv run pytest -k test_scaled_dot_product_attention`(三阶张量)、`uv run pytest -k test_4d_scaled_dot_product_attention`(四阶张量)

---

## 8. `multihead_self_attention`(5 分)

实现因果多头自注意力,继承 `torch.nn.Module`。

**要求**:
- 至少接受 `d_model: int`(Transformer block 输入维度)、`num_heads: int`(注意力头数)两个构造参数
- 遵循 `d_k = d_v = d_model / num_heads`
- 需要正确实现**因果掩码**(阻止看到未来 token)
- RoPE 只作用于 Q、K,不作用于 V;head 维度要被当作批量维处理

**Adapter**:`adapters.run_multihead_self_attention` | **测试**:`uv run pytest -k test_multihead_self_attention`

**注(选做加分方向)**:可以尝试把 Q、K、V 三个投影矩阵合并成一个,只做一次矩阵乘法。

---

## 9. `transformer_block`(3 分)—— 完整的 Pre-Norm Transformer Block

实现 pre-norm Transformer block(组合 `multihead_self_attention` 和 `positionwise_feedforward`,各自搭配 RMSNorm 和残差连接)。

**要求**:至少接受以下构造参数:

| 参数 | 类型 | 含义 |
|------|------|------|
| `d_model` | `int` | Transformer block 输入维度 |
| `num_heads` | `int` | 多头自注意力的头数 |
| `d_ff` | `int` | 逐位置前馈网络内层维度 |

**Adapter**:`adapters.run_transformer_block` | **测试**:`uv run pytest -k test_transformer_block`

---

## 10. `transformer_lm`(3 分)—— 完整的 Transformer 语言模型

把 embedding、`num_layers` 个 Transformer block、最终 norm、LM head 组装成完整模型。

**要求**:至少接受 `transformer_block` 的全部构造参数,外加:

| 参数 | 类型 | 含义 |
|------|------|------|
| `vocab_size` | `int` | 词表大小(决定 token embedding 矩阵的维度) |
| `context_length` | `int` | 最大上下文长度(决定 RoPE cos/sin 缓冲区的维度) |
| `num_layers` | `int` | 使用的 Transformer block 数量 |

**Adapter**:`adapters.run_transformer_lm` | **测试**:`uv run pytest -k test_transformer_lm`

---

## 11. `transformer_accounting`(5 分)—— 书面题,不涉及代码

先手动列出 Transformer forward pass 里**所有**的矩阵乘法,再用 `2mnp` 这条规则(见知识整理文件)换算成 FLOPs。分五小问:

**(a)** 给定 GPT-2 XL 配置(`vocab_size=50257`、`context_length=1024`、`num_layers=48`、`d_model=1600`、`num_heads=25`、`d_ff=4288`):模型有多少可训练参数?按 float32 单精度计算,加载这个模型需要多少内存?
**Deliverable**:一到两句话。

**(b)** 找出 GPT-2 XL forward pass 里所有的矩阵乘法,假设输入序列长度为 `context_length`(1024 个 token),这些矩阵乘法总共需要多少 FLOPs?
**Deliverable**:矩阵乘法清单(附描述)+ 总 FLOPs 数。

**(c)** 根据上面的分析,模型的哪些部分消耗了最多的 FLOPs?
**Deliverable**:一到两句话。

**(d)** 对 GPT-2 small(12 层、`d_model=768`、12 头)、GPT-2 medium(24 层、`d_model=1024`、16 头)、GPT-2 large(36 层、`d_model=1280`、20 头)重复上面的分析。随着模型规模增大,各部分占 FLOPs 的比例如何变化?
**Deliverable**:每个模型给出"各组件 FLOPs 占总 FLOPs 比例"的细分,加一到两句话描述比例随规模的变化趋势。

**(e)** 把 GPT-2 XL 的 `context_length` 增大到 16384,总 FLOPs 如何变化?各组件的相对 FLOPs 占比如何变化?
**Deliverable**:一到两句话。

---

## 建议的推进顺序

1. **3.3 节基础组件**:`linear` → `embedding`(两个都简单,先建立信心)
2. **3.4 节 block 内部组件,自底向上**:`rmsnorm` → `positionwise_feedforward` → `rope` → `softmax` → `scaled_dot_product_attention` → `multihead_self_attention`(每一个都依赖前面的,建议做完一个就跑对应测试再往下走)
3. **3.5 节组装**:`transformer_block`(组合 attention + FFN)→ `transformer_lm`(组合完整模型)
4. **`transformer_accounting`** 书面题,可以在写代码的间隙做,和代码进度没有强依赖(但需要先理解 3.4/3.5 节每个组件的矩阵乘法结构)
