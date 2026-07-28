# Assignment 1 — Section 4「Training a Transformer LM」任务清单

> 整理自 `cs336_assignment1_basics.pdf` 第 28–34 页(Section 4 全部内容,到 Section 5 之前)。
> 本文只梳理**要做什么、怎么验收**,不含任何实现。

## 0. 本节定位

模型(Section 3)已经写完,数据(Section 2)也能变成 token 了。Section 4 补齐**训练所需的三块基础设施**:

- **Loss** —— 交叉熵
- **Optimizer** —— AdamW(先用 SGD 热身)
- **训练稳定性工具** —— 学习率调度、梯度裁剪

Section 5 才把它们拼成完整训练循环。

---

## 1. 任务总览(共 6 个 Problem,8 分)

| # | Problem | 分值 | 类型 | Adapter | 测试命令 |
|---|---|---|---|---|---|
| 1 | `cross_entropy` | 1 | 编程 | `run_cross_entropy` | `uv run pytest -k test_cross_entropy` |
| 2 | `learning_rate_tuning` | 1 | **写作** | — | — |
| 3 | `adamw` | 2 | 编程 | `get_adamw_cls` | `uv run pytest -k test_adamw` |
| 4 | `adamw_accounting` | 2 | **写作(4 小问)** | — | — |
| 5 | `learning_rate_schedule` | 1 | 编程 | `get_lr_cosine_schedule` | `uv run pytest -k test_get_lr_cosine_schedule` |
| 6 | `gradient_clipping` | 1 | 编程 | `run_gradient_clipping` | `uv run pytest -k test_gradient_clipping` |

**4 个编程题 + 2 个写作题。** 写作题的答案进 `writeup.pdf`,代码进 `code.zip`。

---

## 2. 逐项拆解

### ☐ 任务 1:`cross_entropy`(1 分,§4.1,p.29)

**要交什么**:一个函数,输入预测 logits $o_i$ 和目标 $x_{i+1}$,输出 $\ell_i = -\log \text{softmax}(o_i)[x_{i+1}]$。

**讲义明确列出的三条要求**:

1. **减去最大元素**保证数值稳定
2. **尽可能让 log 和 exp 相互抵消**(不要先算出概率再取对数)
3. **处理任意多的 batch 维度,返回批次上的平均值**。约定同 §3.2:**batch 类维度总在前,词表维在最后**

**相关公式**:
- 损失定义:式 (16)、(17)
- 困惑度:$\text{perplexity} = \exp\left(\frac1m \sum_{i=1}^m \ell_i\right)$,式 (18)。训练不用它,但**评估时要报告**,后面 Section 6 会用

**自验收**(测试之外):随机初始化的模型,loss 应约等于 $\ln(\text{vocab\_size})$ —— TinyStories 词表 10000 时约 9.2,GPT-2 词表 50257 时约 10.82。

---

### ☐ 任务 2:`learning_rate_tuning`(1 分,§4.2.1,p.31)—— 写作题

**要做什么**:用讲义 p.30 给出的那个 **toy 训练循环**(`weights = 5 * torch.randn((10,10))`,loss 是 `(weights**2).mean()`),把学习率依次换成 **1e1、1e2、1e3**,各跑 **10 次迭代**,观察 loss 的行为。

**要交什么**:一到两句话,描述你观察到的现象 —— loss 衰减更快、更慢,还是发散(随训练增大)?

**注意**:这题不需要提交代码,也没有测试。但要真的跑一遍,别凭想象写。

---

### ☐ 任务 3:`adamw`(2 分,§4.3,p.32)

**要交什么**:继承 `torch.optim.Optimizer` 实现 AdamW,严格按讲义 **Algorithm 1**(p.32)。

**构造参数**:学习率 $\alpha$,以及 $\beta = (\beta_1, \beta_2)$、$\varepsilon$、权重衰减 $\lambda$。

**讲义强调的实现要点**:

- 用基类提供的 `self.state` 字典保存每个参数的状态(即两个矩估计 $m$、$v$)
- **$t$ 从 1 开始**,不是 0
- 第 7 行的偏差修正:$\alpha_t \leftarrow \alpha \cdot \frac{\sqrt{1-\beta_2^t}}{1-\beta_1^t}$
- 第 8 行的权重衰减是 **decoupled** 的(独立于梯度更新那一步),这正是 AdamW 区别于 Adam 的地方 —— 注意它在算法里出现的**位置**

**超参数常识**(讲义提到,做实验时有用):典型取 $(\beta_1,\beta_2) = (0.9, 0.999)$;LLaMA、GPT-3 等大模型常用 $(0.9, 0.95)$;$\varepsilon$ 取 $10^{-8}$ 量级。

---

### ☐ 任务 4:`adamw_accounting`(2 分,§4.3,p.32–33)—— 写作题,4 小问

前提:**所有张量都用 float32**。

**(a) 峰值显存的代数表达式**
按四类分解:**参数 / 激活 / 梯度 / 优化器状态**。用 `batch_size` 和模型超参(`vocab_size`, `context_length`, `num_layers`, `d_model`, `num_heads`)表示,假设 $d_{ff} = \frac83 d_{model}$。

激活部分**只统计**讲义列出的这些(别多算也别少算):
- Transformer block:RMSNorm(s);多头注意力子层的 **QKV 投影、$QK^\top$ 矩阵乘、softmax、对 V 的加权求和、输出投影**;FFN(SwiGLU)的 **$W_1$、$W_2$、门分支上的 SiLU、逐元素乘、$W_3$**
- 最后的 RMSNorm
- output embedding
- logits 上的 cross-entropy

**交付**:参数、激活、梯度、优化器状态**各自**的代数式,以及总和。

**(b) 代入 GPT-2 XL,化成只依赖 `batch_size` 的形式**
交付:形如 $a \cdot \text{batch\_size} + b$ 的表达式(给出 $a$、$b$ 的数值),以及 **80GB 显存下的最大 batch size**。

**(c) 一步 AdamW 的 FLOPs**
交付:代数表达式 + 简要理由。

**(d) 训练时长估算**
条件:H100 峰值 **495 TFLOP/s**(float32/TF32),假设 **MFU = 50%**,GPT-2 XL,**400K 步**,batch size **1024**,单卡。按 Kaplan / Hoffmann 的惯例,**反向传播的 FLOPs 是前向的 2 倍**。
交付:训练需要多少**小时**,附简要理由。

> 提示:(a)(b) 可以直接复用你在 `transformer_accounting` 里已经做过的参数量拆解;(d) 需要前向 FLOPs 的结果。两题是连着的。

---

### ☐ 任务 5:`learning_rate_schedule`(1 分,§4.4,p.33–34)

**要交什么**:一个函数,输入 $t$、$\alpha_{max}$、$\alpha_{min}$、$T_w$(warmup 步数)、$T_c$(余弦退火结束步),返回该步的学习率。

**三段式定义**(讲义原文):

| 阶段 | 条件 | 学习率 |
|---|---|---|
| Warm-up | $t < T_w$ | $\alpha_t = \dfrac{t}{T_w}\alpha_{max}$ |
| Cosine annealing | $T_w \le t \le T_c$ | $\alpha_t = \alpha_{min} + \dfrac12\left(1+\cos\left(\dfrac{t-T_w}{T_c-T_w}\pi\right)\right)(\alpha_{max}-\alpha_{min})$ |
| Post-annealing | $t > T_c$ | $\alpha_t = \alpha_{min}$ |

这是 LLaMA 使用的调度。**边界点**($t=T_w$、$t=T_c$)最容易出错,写完手动代入这两个值检查连续性。

---

### ☐ 任务 6:`gradient_clipping`(1 分,§4.5,p.34)

**要交什么**:一个函数,输入**一组参数**和最大 $\ell_2$ 范数 $M$,**原地修改**每个参数的梯度。

**规则**:计算所有参数梯度拼起来的**全局** $\ell_2$ 范数 $\|g\|_2$。若 $\|g\|_2 < M$ 则不动;否则整体缩放 $\dfrac{M}{\|g\|_2 + \varepsilon}$。

**$\varepsilon = 10^{-6}$**(PyTorch 默认值)。注意讲义那句话:**缩放后的范数会略小于 $M$**——这是 $\varepsilon$ 带来的,不是 bug。

关键词:**全局范数**(不是每个张量各自裁剪)、**原地修改**、在 `backward()` 之后、`optimizer.step()` 之前调用。

---

## 3. 建议顺序与依赖

```
cross_entropy ──┐
                ├─→ (Section 5 训练循环)
adamw ──────────┤
learning_rate_schedule ─┤
gradient_clipping ──────┘

learning_rate_tuning  ← 独立,只依赖讲义 p.30 的 SGD 示例
adamw_accounting      ← 依赖 transformer_accounting 的结果
```

推荐节奏:

1. 先把 **4 个编程题**做完并全部通过 pytest(它们互不依赖,可以按分值从小到大做)
2. 顺手做 `learning_rate_tuning`(跑一次实验,几分钟)
3. 最后做 `adamw_accounting`(纯推导,最费脑,但可复用已有结果)

---

## 4. 完成度检查

- [ ] `test_cross_entropy` 通过
- [ ] `test_adamw` 通过
- [ ] `test_get_lr_cosine_schedule` 通过
- [ ] `test_gradient_clipping` 通过
- [ ] `learning_rate_tuning` 的一到两句话已写进 writeup
- [ ] `adamw_accounting` (a)(b)(c)(d) 四问答案已写进 writeup
- [ ] 四个 adapter 函数都已在 `adapters.py` 中接好(只做转发,不写实质逻辑)

一次性跑完本节全部测试:

```bash
uv run pytest -k "cross_entropy or adamw or lr_cosine or gradient_clipping"
```

---

## 5. 相关笔记

- [transformer.py开发知识点总结.md](transformer.py开发知识点总结.md)
- [激活函数与门控FFN整理.md](激活函数与门控FFN整理.md)
- [torch.nn.Module介绍.md](torch.nn.Module介绍.md)
