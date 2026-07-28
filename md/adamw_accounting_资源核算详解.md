# Problem (adamw_accounting)：AdamW 训练资源核算（2 分）详解

> 本文对应 `cs336_assignment1_basics.pdf` 第 4.3 节末尾的 `adamw_accounting` 问题。
> 包含题干翻译、涉及的知识点讲解、完整推导与数值解答。
> 所有数值均用 `Fraction`（精确分数）计算，避免浮点累积误差。

---

## 0. 题干翻译

**问题（adamw_accounting）：用 AdamW 训练时的资源核算（2 分）**

让我们计算运行 AdamW 需要多少内存和计算量。假设每个张量都用 float32。

**(a)** 运行 AdamW 需要多少**峰值内存**？请按**参数、激活值、梯度、优化器状态**四部分分解你的答案。用 `batch_size` 和模型超参数（`vocab_size`, `context_length`, `num_layers`, `d_model`, `num_heads`）表示。假设 $d_{ff} = \frac{8}{3} \times d_{model}$。

为简化计算，激活值只考虑以下部分：

- Transformer block
  - RMSNorm(s)
  - 多头自注意力子层：$QKV$ 投影、$QK^\top$ 矩阵乘、softmax、值的加权求和、输出投影
  - 逐位置前馈网络（SwiGLU）：$W_1$、$W_2$、门控分支上的 SiLU、逐元素乘积、$W_3$
- 最终的 RMSNorm
- 输出嵌入（output embedding）
- logits 上的交叉熵

**交付物**：参数、激活值、梯度、优化器状态各自的代数表达式，以及总和。

**(b)** 代入 GPT-2 XL 形状的模型，得到一个只依赖 `batch_size` 的表达式。在 80GB 内存限制下能用的最大 batch size 是多少？

**交付物**：形如 $a \cdot \text{batch\_size} + b$ 的表达式（$a, b$ 为具体数值），以及最大 batch size。

**(c)** 运行 AdamW 的一步需要多少 FLOPs？

**交付物**：一个代数表达式，附简要说明。

**(d)** 模型 FLOPs 利用率（MFU）定义为实测吞吐量（tokens/秒）与硬件理论峰值 FLOP 吞吐量之比。NVIDIA H100 对 "float32"（实际是 TensorFloat-32，实质是 "bfloat19"）运算的理论峰值是 495 teraFLOP/s。假设你能达到 50% MFU，在单张 H100 上以 batch size 1024 训练 GPT-2 XL 400K 步需要多久？依照 Kaplan 和 Hoffmann 等人的做法，假设反向传播的 FLOPs 是前向的两倍。

**交付物**：训练所需小时数，附简要说明。

---

## 1. 前置知识点

### 1.1 float32 = 4 字节

一个 float32 数占 **4 字节**。所有内存计算都是「元素个数 × 4」。

单位换算本文统一用二进制 GB（GiB）：$1\text{GB} = 2^{30} = 1{,}073{,}741{,}824$ 字节。

### 1.2 训练时显存的四大来源

这是本题的核心分类，也是理解大模型训练显存瓶颈的基础：

| 来源 | 大小 | 说明 |
|---|---|---|
| **参数**（parameters） | $P$ | 模型权重本身 |
| **梯度**（gradients） | $P$ | 每个参数都要存一份梯度，**形状与参数完全相同** |
| **优化器状态**（optimizer state） | $2P$ | AdamW 为每个参数维护一阶矩 $m$ 和二阶矩 $v$，**各一份** |
| **激活值**（activations） | 正比于 $\text{batch\_size}$ | 前向传播的中间结果，反向传播时需要用到 |

**关键洞察**：前三项加起来是 $4P$，**与 batch size 无关**（常数项）；只有激活值随 batch size 线性增长。这正是 (b) 问要求写成 $a \cdot \text{batch\_size} + b$ 的原因。

> 这也解释了一个实践现象：显存不够时，减小 batch size 只能压缩激活值那部分，参数+梯度+优化器状态这 $4P$ 是**压不掉**的硬底线。想突破它就得换手段（混合精度、ZeRO 分片、梯度检查点等）。

### 1.3 为什么梯度和参数一样大

反向传播要对每个可训练参数计算 $\partial \mathcal{L}/\partial \theta_i$，每个参数对应一个梯度值 → 形状完全相同。

### 1.4 为什么 AdamW 的优化器状态是 $2P$

回看 AdamW 算法（PDF Algorithm 1）：

```
m ← β₁m + (1-β₁)g      ← 一阶矩(动量)，形状同 θ
v ← β₂v + (1-β₂)g²     ← 二阶矩(梯度平方的滑动平均)，形状同 θ
```

对照你在 [training.py:53-54](../cs336_basics/training.py#L53-L54) 的实现：

```python
m = state.get('m', torch.zeros(p.shape, ...))
v = state.get('v', torch.zeros(p.shape, ...))
```

两个都是 `p.shape` → 各占 $P$，合计 $2P$。（迭代计数 `t` 是标量，可忽略。）

**对比 SGD**：无状态，优化器状态为 0。这就是「AdamW 用额外内存换取稳定性和收敛速度」的具体含义——参数相关显存从 $2P$（参数+梯度）涨到 $4P$，**翻倍**。

### 1.5 为什么要保存激活值

反向传播的链式法则需要前向传播的中间结果。例如 $y = Wx$ 对 $W$ 求导得 $\partial \mathcal{L}/\partial W = (\partial \mathcal{L}/\partial y) \cdot x^\top$——**需要 $x$**，也就是这一层的输入。所以前向算出的中间张量不能立即丢弃，要一直留到反向传播用完。

（`torch.no_grad()` 之所以能省显存，正是因为它告诉 PyTorch「不用留这些中间结果」。）

---

## 2. (a) 峰值内存的代数表达式

### 符号约定

| 符号 | 含义 |
|---|---|
| $V$ | vocab_size |
| $L$ | context_length（序列长度） |
| $N$ | num_layers |
| $d$ | d_model |
| $h$ | num_heads |
| $B$ | batch_size |
| $d_{ff}$ | $\frac{8}{3}d$（题目规定） |

### 2.1 参数量 $P$

逐个模块清点（本实现**无 bias**，`Linear` 只有 `weight`）：

**Token embedding**：形状 $(V, d)$ → $Vd$

**每个 Transformer block**：

| 组件 | 形状 | 参数量 |
|---|---|---|
| RMSNorm ×2 | 各 $(d,)$ | $2d$ |
| $W_Q, W_K, W_V, W_O$ | 各 $(d, d)$ | $4d^2$ |
| $W_1, W_3$ | 各 $(d_{ff}, d)$ | $2d \cdot d_{ff}$ |
| $W_2$ | $(d, d_{ff})$ | $d \cdot d_{ff}$ |

FFN 合计 $3 d\, d_{ff} = 3d \cdot \frac{8}{3}d = \boxed{8d^2}$ ——这就是题目设 $d_{ff} = \frac{8}{3}d$ 的用意：**让 SwiGLU 的三个矩阵总参数量恰好等于普通 FFN（$4d$ 中间维、两个矩阵）的 $8d^2$**，便于横向比较。

每块合计：$4d^2 + 8d^2 + 2d = 12d^2 + 2d$

**最终 RMSNorm**：$d$

**输出嵌入 / lm_head**：形状 $(V, d)$ → $Vd$

$$\boxed{P = 2Vd + N(12d^2 + 2d) + d}$$

> 注意：本实现的 embedding 和 lm_head **不共享权重**（weight tying），所以是 $2Vd$ 而非 $Vd$。

### 2.2 梯度

$$\text{gradients} = P \text{ 个元素}$$

### 2.3 优化器状态

$$\text{optimizer state} = 2P \text{ 个元素}$$

### 2.4 激活值 $A$（每个 batch 元素）

按题目指定的清单逐项计算。设 batch 内单个样本，序列长 $L$：

**每个 Transformer block：**

| 项 | 张量形状 | 元素数 |
|---|---|---|
| RMSNorm ×2 | $(L, d)$ | $2Ld$ |
| $QKV$ 投影 | 各 $(L, d)$ | $3Ld$ |
| $QK^\top$ | $(h, L, L)$ | $hL^2$ |
| softmax | $(h, L, L)$ | $hL^2$ |
| 加权求和 | $(L, d)$ | $Ld$ |
| 输出投影 | $(L, d)$ | $Ld$ |
| $W_1$ 输出 | $(L, d_{ff})$ | $L d_{ff}$ |
| $W_3$ 输出 | $(L, d_{ff})$ | $L d_{ff}$ |
| SiLU（门控分支） | $(L, d_{ff})$ | $L d_{ff}$ |
| 逐元素乘积 | $(L, d_{ff})$ | $L d_{ff}$ |
| $W_2$ 输出 | $(L, d)$ | $Ld$ |

注意力小计：$3Ld + 2hL^2 + Ld + Ld = 5Ld + 2hL^2$

FFN 小计：$4L d_{ff} + Ld = 4L \cdot \frac{8}{3}d + Ld = \frac{32}{3}Ld + Ld = \frac{35}{3}Ld$

**每块合计**：$2Ld + 5Ld + 2hL^2 + \frac{35}{3}Ld = \frac{56}{3}Ld + 2hL^2$

**尾部：**

| 项 | 形状 | 元素数 |
|---|---|---|
| 最终 RMSNorm | $(L, d)$ | $Ld$ |
| 输出嵌入（logits） | $(L, V)$ | $LV$ |
| 交叉熵 | $(L, V)$ | $LV$ |

$$\boxed{A = B\left[N\left(\frac{56}{3}Ld + 2hL^2\right) + Ld + 2LV\right]}$$

**两个值得注意的结构性观察：**

1. $2hL^2$ 项对序列长度是**二次**的——这是注意力机制的著名瓶颈。当 $L$ 很大时，这一项会主导显存占用（也是 FlashAttention 等工作要解决的问题）。
2. $2LV$ 项在 $V$ 很大时（GPT-2 是 50257）相当可观，且**不随层数增长**——但它是 $V$ 的一次项，实测下面会看到它占了尾部的绝大部分。

### 2.5 总峰值内存

$$\boxed{\text{Memory} = 4\big[\underbrace{P}_{\text{参数}} + \underbrace{P}_{\text{梯度}} + \underbrace{2P}_{\text{优化器}} + \underbrace{A}_{\text{激活}}\big] \text{ 字节} = 4(4P + A) \text{ 字节}}$$

---

## 3. (b) 代入 GPT-2 XL

### GPT-2 XL 形状

$$V = 50257,\quad L = 1024,\quad N = 48,\quad d = 1600,\quad h = 25$$

### 参数量

| 项 | 计算 | 数值 |
|---|---|---|
| embedding | $Vd$ | 80,411,200 |
| 每块 attention | $4d^2$ | 10,240,000 |
| 每块 FFN | $8d^2$ | 20,480,000 |
| 每块 RMSNorm | $2d$ | 3,200 |
| 每块合计 | $12d^2 + 2d$ | 30,723,200 |
| × 48 层 | | 1,474,713,600 |
| final RMSNorm | $d$ | 1,600 |
| lm_head | $dV$ | 80,411,200 |
| **$P$ 总计** | | **1,635,537,600（≈1.64B）** |

> 常说 GPT-2 XL 是 1.5B 参数，这里算出 1.64B——差异来自本题设定的架构（SwiGLU + RMSNorm + RoPE，无 bias、无学习式位置嵌入、不共享词嵌入），是"GPT-2 XL **形状**"而非原版 GPT-2 XL。

### 激活量（每样本）

| 项 | 数值 |
|---|---|
| 每块 RMSNorm ×2 | 3,276,800 |
| QKV 投影 | 4,915,200 |
| $QK^\top$ | 26,214,400 |
| softmax | 26,214,400 |
| 加权求和 | 1,638,400 |
| 输出投影 | 1,638,400 |
| FFN | 19,114,666 |
| **每块合计** | **83,012,266** |
| × 48 层 | 3,984,588,800 |
| 尾部（$Ld + 2LV$） | 104,564,736 |
| **$A$ 总计（每样本）** | **4,089,153,536** |

注意：$QK^\top$ + softmax 两项就占了每块的 $\frac{52{,}428{,}800}{83{,}012{,}266} \approx 63\%$ ——注意力矩阵是激活值的大头。

### 显存分解

| 来源 | 表达式 | 字节 | GB |
|---|---|---|---|
| 参数 | $4P$ | 6,542,150,400 | **6.093** |
| 梯度 | $4P$ | 6,542,150,400 | **6.093** |
| 优化器状态 $m, v$ | $8P$ | 13,084,300,800 | **12.186** |
| **静态小计** | $16P$ | 26,168,601,600 | **24.371** |
| 激活（每样本） | $4A$ | 16,356,614,144 | **15.233** |

### 最终表达式

$$\boxed{\text{Memory (GB)} = 15.233 \cdot \text{batch\_size} + 24.371}$$

### 最大 batch size

$$15.233 B + 24.371 \le 80 \implies B \le 3.652$$

$$\boxed{\text{最大 batch\_size} = 3}$$

验证：
- $B = 3$：$15.233 \times 3 + 24.371 = \mathbf{70.07}$ GB ✓
- $B = 4$：$15.233 \times 4 + 24.371 = \mathbf{85.30}$ GB ✗（超出 80GB）

> **实践启示**：一张 80GB 的 H100 训练 1.6B 模型，batch size 只能到 3。而 (d) 问要求 batch size = 1024——这意味着实际训练必须用**梯度累积**（把 1024 拆成很多个 microbatch 累积梯度）或**多卡并行**。这正是大模型训练需要分布式的直接原因。

---

## 4. (c) AdamW 一步的 FLOPs

对照 PDF Algorithm 1 逐行数**每个参数元素**上的浮点运算：

| 算法行 | 运算 | FLOPs/元素 |
|---|---|---|
| 7 | $\alpha_t \leftarrow \alpha\frac{\sqrt{1-\beta_2^t}}{1-\beta_1^t}$ | 标量运算，$O(1)$，可忽略 |
| 8 | $\theta \leftarrow \theta - \alpha\lambda\theta$ | 1 乘 + 1 减 = **2** |
| 9 | $m \leftarrow \beta_1 m + (1-\beta_1)g$ | 2 乘 + 1 加 = **3** |
| 10 | $v \leftarrow \beta_2 v + (1-\beta_2)g^2$ | $g^2$、$\beta_2 v$、$(1-\beta_2)g^2$ 三次乘 + 1 加 = **4** |
| 11 | $\theta \leftarrow \theta - \alpha_t \frac{m}{\sqrt{v}+\varepsilon}$ | sqrt、加 $\varepsilon$、除、乘 $\alpha_t$、减 = **5** |

$$\boxed{\text{FLOPs}_{\text{AdamW}} \approx 14 P}$$

代入 GPT-2 XL：$14 \times 1.636\times 10^9 \approx 2.29 \times 10^{10} = \mathbf{22.9}$ GFLOPs

**说明**：常数 14 取决于怎么数（比如把 $\alpha\lambda$ 视为预计算的标量、sqrt 算 1 次运算等），不同数法会得到 10~15 之间的值。**关键结论不是这个常数，而是「AdamW 一步的开销正比于参数量 $P$、与 batch size 和序列长度无关」**。

对比 (d) 中算出的一步前向+反向 $1.08 \times 10^{16}$ FLOPs：

$$\frac{2.29 \times 10^{10}}{1.08 \times 10^{16}} \approx 0.0002\%$$

**优化器更新的计算开销完全可以忽略。** AdamW 的代价在**内存**（$2P$ 的状态）而非**计算**——这是理解它的关键。

---

## 5. (d) 训练 400K 步需要多久

### 5.1 前向传播 FLOPs

矩阵乘法 FLOPs 的标准计法：$(m \times k)$ 矩阵乘 $(k \times n)$ 矩阵需要 $2mnk$ FLOPs（每个输出元素做 $k$ 次乘加，一次乘加 = 2 FLOPs）。

设 $B = 1024$，token 总数 $BL = 1024 \times 1024 = 1{,}048{,}576$。

**每个 Transformer block：**

| 操作 | FLOPs |
|---|---|
| $QKV$ 投影（3 个 $d\times d$） | $3 \times 2BLd^2 = 6BLd^2$ |
| $QK^\top$ | $2BhL^2 \cdot \frac{d}{h} = 2BL^2d$ |
| 加权求和（$\text{attn} \times V$） | $2BL^2d$ |
| 输出投影 | $2BLd^2$ |
| FFN（$W_1, W_2, W_3$，每个 $d \times d_{ff}$） | $3 \times 2BL\,d\,d_{ff} = 6BL \cdot \frac{8}{3}d^2 = 16BLd^2$ |

每块合计：$\boxed{24BLd^2 + 4BL^2d}$

**lm_head**：$2BLdV$

$$\text{FLOPs}_{\text{fwd}} = N(24BLd^2 + 4BL^2d) + 2BLdV$$

代入数值：

| 项 | 数值 |
|---|---|
| 每块 | $7.130 \times 10^{13}$ |
| × 48 层 | $3.422 \times 10^{15}$ |
| lm_head | $1.686 \times 10^{14}$ |
| **前向合计** | $\mathbf{3.591 \times 10^{15}}$ |

### 5.2 加上反向传播

题目指定：反向 = 2 × 前向。

$$\text{FLOPs}_{\text{step}} = 3 \times \text{FLOPs}_{\text{fwd}} = 1.077 \times 10^{16}$$

> **为什么反向是前向的 2 倍**：前向每个矩阵乘 $y = Wx$ 算一次；反向要算**两个**梯度——对输入的 $\partial\mathcal{L}/\partial x = W^\top \delta$ 和对权重的 $\partial\mathcal{L}/\partial W = \delta x^\top$，各自的计算量与前向相当，故为 $2\times$。

### 5.3 总计算量与耗时

$$\text{总 FLOPs} = 1.077\times 10^{16} \times 400{,}000 = 4.309 \times 10^{21}$$

有效算力：$495 \times 10^{12} \times 50\% = 2.475 \times 10^{14}$ FLOP/s

$$t = \frac{4.309 \times 10^{21}}{2.475 \times 10^{14}} = 1.741 \times 10^7 \text{ 秒}$$

$$\boxed{\approx 4{,}836 \text{ 小时} \approx 202 \text{ 天} \approx 0.55 \text{ 年}}$$

---

## 6. 结论汇总

| 问 | 答案 |
|---|---|
| **(a)** 参数 | $P = 2Vd + N(12d^2+2d) + d$ |
| | 梯度 $= P$，优化器状态 $= 2P$ |
| | 激活 $A = B\left[N\left(\frac{56}{3}Ld + 2hL^2\right) + Ld + 2LV\right]$ |
| | 总内存 $= 4(4P + A)$ 字节 |
| **(b)** | $\text{Memory (GB)} = 15.233 \cdot \text{batch\_size} + 24.371$，最大 **batch_size = 3** |
| **(c)** | $\approx 14P \approx 22.9$ GFLOPs，与前向反向相比可忽略 |
| **(d)** | $\approx$ **4,836 小时**（约 202 天） |

## 7. 从这道题该学到什么

1. **显存的四分法**：参数 / 梯度 / 优化器状态 / 激活。前三者是 $4P$ 的硬底线，与 batch size 无关；只有激活可以通过减小 batch 来压缩。

2. **AdamW 的真实代价在内存不在计算**：优化器状态让参数相关显存**翻倍**（$2P \to 4P$），但一步更新的 FLOPs 只占总计算的万分之二。

3. **注意力的 $L^2$ 项**：激活值里的 $2hL^2$ 对序列长度是二次增长，实测在 GPT-2 XL 配置下占了每块激活的 63%。这是长上下文的核心障碍。

4. **单卡训练大模型不现实**：80GB 卡上 1.6B 模型只能跑 batch size 3，而目标 batch size 是 1024——必须靠梯度累积或分布式并行。而且即便算力拉满 50% MFU，单卡也要跑 202 天。

5. **$d_{ff} = \frac{8}{3}d$ 的设计意图**：让 SwiGLU 的三矩阵结构（$3d \cdot d_{ff} = 8d^2$）与传统 FFN 的两矩阵结构（$2 \cdot d \cdot 4d = 8d^2$）参数量持平，从而在做架构对比实验时是公平的。
