# Section 7：Experiments 整理

> 对应 `cs336_assignment1_basics.pdf` 第 38～45 页。本文按当前版本讲义整理，并结合本仓库的已有训练结果说明下一步应该做什么。

## 1. Section 7 的核心目标

Section 7 不再要求实现一个孤立的 Transformer 算子，而是要求把前面完成的 tokenizer、模型、优化器、训练循环、checkpoint 和 decoder 组合起来，开展一组**可复现、可比较、可解释**的语言模型实验。

整套工作的主线是：

```text
可靠的实验记录
    ↓
TinyStories 基线模型
    ↓
学习率与 batch size 调参
    ↓
生成文本，定性检查模型质量
    ↓
RMSNorm / pre-norm / RoPE / SwiGLU 消融
    ↓
OpenWebText 实验
    ↓
自选改进与 leaderboard
```

这一节真正训练的是实验能力：一次只改变一个因素，保持其他条件一致，记录 loss 曲线和运行时间，再用证据解释结果。

## 2. 7.1 实验基础设施与实验日志

### 2.1 为什么要记录实验

Section 7 会运行许多配置。如果只看终端最后一行，很快就会无法回答：

- 这个 checkpoint 使用了哪个学习率？
- 两次实验使用的是不是同一个 tokenizer 和验证集？
- loss 是在相同 token 预算下比较的吗？
- 哪一次发生了发散，发生在第几步？
- 更好的最终 loss 是否只是因为它训练得更久？

所以讲义把实验追踪本身作为一个单独的 deliverable。

### 2.2 至少应该记录什么

每次实验建议保存：

| 类别 | 建议字段 |
| --- | --- |
| 实验身份 | `run_name`、时间、代码版本或 commit |
| 数据 | 数据集、tokenizer、词表大小、训练/验证 token 文件 |
| 模型 | `context_length`、`d_model`、`d_ff`、层数、头数、RoPE theta、参数量 |
| 优化器 | `alpha_max`、`alpha_min`、warmup、cosine decay、betas、epsilon、weight decay |
| 训练 | batch size、总步数、总 token 数、seed、device、梯度裁剪阈值 |
| 过程指标 | step、wall-clock time、train loss、validation loss、learning rate、gradient norm |
| 性能指标 | tokens/s、峰值显存（如果可用）、总训练时间 |
| 产物 | checkpoint、生成样本、学习曲线、实验结论 |

讲义明确要求 loss 曲线能够使用两类横轴：

1. gradient step；
2. wall-clock time。

只记录 step 不足以评价运行效率，只记录时间又不容易分析优化过程。

### 2.3 本仓库已有基础

当前代码已经具备一部分实验基础设施：

- `runs/<experiment>/config.json`：保存一次实验的配置；
- `runs/<experiment>/metrics.jsonl`：记录 step、loss、学习率、梯度范数和 elapsed time；
- `ckpt_last.pt`：中断恢复；
- `ckpt_final.pt`：最终模型；
- `decoding.py`：从 checkpoint 加载模型并生成文本。

后续主要还缺：统一的实验命名、自动绘制对比曲线、不同实验的汇总表，以及架构消融开关。

## 3. 7.2 TinyStories 基线实验

TinyStories 是一个语言简单、结构重复、适合训练小语言模型的数据集。Section 7 先在它上面验证整个训练系统，然后进行超参数搜索和架构比较。

### 3.1 讲义给出的基线超参数

| 超参数 | 推荐值 |
| --- | ---: |
| vocabulary size | 10,000 |
| context length | 256 |
| `d_model` | 512 |
| `d_ff`（SwiGLU） | 1,344 |
| RoPE theta | 10,000 |
| Transformer layers | 4 |
| attention heads | 16 |
| 标准训练 token 预算 | 327,680,000 |

总训练 token 数按下面的式子计算：

$$
N_{tokens} = batch\_size \times total\_steps \times context\_length.
$$

学习率、warmup、AdamW 的 betas/epsilon、weight decay 等参数需要通过实验确定。

### 3.2 低资源路线

如果使用 CPU 或 Apple Silicon，讲义允许把 token 预算降低到约 40M，并把目标 validation loss 从 1.45 放宽到 2.00。

讲义给出的低资源示例是：

$$
32 \times 5000 \times 256 = 40,960,000\ tokens.
$$

这正是当前 `runs/experiment01/config.json` 使用的配置。讲义报告的参考结果是 M4 Max 上：

- CPU 约 1 小时 22 分钟；
- MPS 约 36 分钟；
- step 5000 时 validation loss 约 1.80。

这些数字是参考值，不是要求不同硬件必须达到完全相同的速度。

低资源路线还要注意：

- cosine schedule 应在最后一个训练 step 到达最小学习率；
- MPS 不建议启用 `torch.set_float32_matmul_precision("high")`；
- CPU 可以尝试 `torch.compile(model)`；
- MPS 可以尝试 `torch.compile(model, backend="aot_eager")`。

### 3.3 先做的 sanity checks

在投入长时间训练前，讲义建议：

1. 尝试过拟合单个 minibatch，正确实现通常能把 loss 很快降到接近 0；
2. 用 debugger 检查各层中间 tensor shape；
3. 监控 activation、weight 和 gradient norm，排查爆炸或消失；
4. 确认 dataloader、validation 和 checkpoint 不构成明显性能瓶颈；
5. 确认模型确实使用 batch 计算，而不是逐样本循环。

## 4. 学习率实验（learning_rate）

这是 TinyStories 中最重要的调参实验。

### 4.1 要回答的问题

1. 不同最大学习率对应怎样的训练曲线和最终 loss？
2. 哪些学习率稳定收敛，哪些学习率发散？
3. 最好的学习率与“稳定性边缘”（edge of stability）有什么关系？

### 4.2 Deliverables

- 多个学习率的 learning curves；
- 每次实验的最终 loss，发散实验明确标记 divergence；
- 学习率搜索策略说明；
- 至少一条使用过大学习率而发散的曲线；
- 对最佳学习率和发散临界点关系的分析；
- 标准路线 TinyStories validation loss 不高于 1.45；低资源路线可按讲义放宽到 2.00。

### 4.3 推荐搜索方法

先进行短程、对数尺度搜索，再对候选值进行完整训练。例如围绕当前的 `3e-4`，可以先尝试若干更小和更大的数量级。具体候选值应根据短程曲线继续调整，而不是机械地完成一个固定列表。

比较时应保持以下因素一致：

- tokenizer 和 tokenized dataset；
- 模型结构；
- batch size、context length 和总 token 预算；
- seed 和固定 validation batches；
- warmup、学习率调度方式和其他 AdamW 参数。

“发散”不应只表示某一步 loss 略有波动，而应有明确证据，例如 loss 持续急剧升高、出现 NaN/Inf，或梯度范数失控且无法恢复。

## 5. Batch size 实验（batch_size_experiment）

### 5.1 实验要求

从 batch size 1 一直测试到设备内存允许的上限，并包含若干中间值；讲义特别提到 64 和 128 这类常见配置。

### 5.2 Deliverables

- 不同 batch size 的 learning curves；
- 必要时为不同 batch size 重新调学习率；
- 简要讨论 batch size 对训练质量和效率的影响。

### 5.3 如何公平比较

如果 batch size 改变但 total steps 不变，总训练 token 数也会改变，比较就混入了额外训练数据量这个变量。更公平的方法是保持总 token 预算不变：

$$
total\_steps = \frac{target\_tokens}{batch\_size \times context\_length}.
$$

需要同时观察：

- 相同 token 预算下的 validation loss；
- 相同 wall-clock time 下的 validation loss；
- tokens/s；
- 梯度噪声和曲线波动；
- 是否出现内存不足。

大 batch 通常能提高硬件利用率，但不保证在固定计算预算下总是得到最好的泛化结果，而且最佳学习率可能随 batch size 改变。

## 6. 文本生成实验（generate）

使用 Section 6 完成的 decoder 和训练 checkpoint 生成文本，并通过 temperature 与 top-p 调整生成行为。

### Deliverables

- 至少 256 个生成 token，或者生成到第一个 `<|endoftext|>` 为止；
- 简短评价输出的流畅性；
- 至少分析两个影响生成质量的因素。

可分析的因素包括：

- validation loss 和训练是否充分；
- 模型规模；
- 训练 token 数；
- tokenizer 质量；
- prompt；
- temperature；
- top-p；
- context length；
- 数据集本身的质量与多样性。

生成实验不能只挑一个看起来最好的样本。建议固定多个 prompt，并记录 sampling 参数和随机种子，这样不同 checkpoint 才能合理比较。

## 7. 7.3 架构消融实验

消融实验的原则是：只改变被研究的组件，其余训练条件尽量保持一致。

| 实验 | 对照组 | 实验组 | 要提交什么 |
| --- | --- | --- | --- |
| 删除 RMSNorm | 原始 pre-norm Transformer | 删除所有 RMSNorm | 原学习率曲线、降低学习率后的最佳曲线、稳定性分析 |
| pre-norm vs post-norm | 残差分支前做 RMSNorm | 残差相加后做 RMSNorm | 两者 learning curve 对比 |
| RoPE vs NoPE | 使用 RoPE | 完全不提供位置编码 | validation learning curve 对比 |
| SwiGLU vs SiLU | `W2(SiLU(W1x) * W3x)` | `W2 SiLU(W1x)` | 参数量近似匹配的曲线和分析 |

### 7.1 删除 RMSNorm

目标是观察 normalization 对训练稳定性的影响：

- 在原来的最佳学习率下是否发散？
- 降低学习率后能否重新稳定？
- 即使稳定，收敛速度和最终 validation loss 是否变差？

### 7.2 Pre-norm 改为 post-norm

当前模型使用：

$$
z = x + \operatorname{Attention}(\operatorname{RMSNorm}(x)),
$$

$$
y = z + \operatorname{FFN}(\operatorname{RMSNorm}(z)).
$$

post-norm 对照改为：

$$
z = \operatorname{RMSNorm}(x + \operatorname{Attention}(x)),
$$

$$
y = \operatorname{RMSNorm}(z + \operatorname{FFN}(z)).
$$

重点比较两者的训练稳定性和 learning curve，而不只是最终一个 loss 数字。

### 7.3 RoPE 改为 NoPE

NoPE 实验完全删除显式位置信息，但仍保留 causal mask。实验目的是验证 decoder-only Transformer 是否能仅借助 causal structure 学到足够的位置相关模式，以及这种做法在当前规模下损失多少性能。

### 7.4 SwiGLU 改为 SiLU FFN

SiLU 基线为：

$$
\operatorname{FFN}_{SiLU}(x) = W_2\operatorname{SiLU}(W_1x).
$$

因为 SiLU FFN 只有两个权重矩阵，而 SwiGLU 有三个，为了近似匹配参数量：

- SwiGLU 使用约 `d_ff = 8/3 × d_model`，当前取 1344；
- SiLU baseline 使用 `d_ff = 4 × d_model`，即当前 `d_model=512` 时取 2048。

若参数量不匹配，实验无法判断差异来自 gating 还是单纯来自参数数量。

## 8. 7.4 OpenWebText 实验

OpenWebText 比 TinyStories 更真实、更复杂、更多样，也包含更多噪声。讲义要求使用与 TinyStories 相同的模型架构和训练迭代数训练 OpenWebText 模型，并提醒可能需要重新调整学习率和 batch size。

### Deliverables

- OpenWebText 的 learning curve；
- 解释它和 TinyStories loss 的差异，以及这些 loss 应该如何理解；
- 与 TinyStories 相同格式的生成文本；
- 分析为什么相同模型和计算预算下，OpenWebText 输出通常更差。

不能直接把两个数据集上的 loss 当作完全同尺度的能力分数。数据的熵、词汇分布、文体多样性、噪声和 tokenizer 都会影响交叉熵。

如果计算资源有限，讲义允许继续在 TinyStories 上测试架构修改，以 validation loss 作为主要指标。

## 9. 7.5 自选修改与 Leaderboard

最后一部分要求自行改进模型或训练方法，并在限制内尽量降低 OpenWebText validation loss。

### 规则

- 单次最终提交在 B200 上最多运行 45 分钟，即 0.75 B200-hour；
- 只能使用课程提供的 OpenWebText 训练数据；
- 其他模型与优化修改基本不受限制；
- naive baseline validation loss 为 5.0，讲义期望最终结果优于它。

### 可尝试方向

- input embedding 与 LM head weight tying；
- 参考 Llama 3、Qwen 2.5 等公开架构；
- 参考 modded-nanogpt 的小规模预训练优化；
- 改进初始化、优化器、学习率调度或训练效率；
- 在 TinyStories 或 OpenWebText 小子集上先筛选修改，再进行最终运行。

### Deliverables

- 最终 validation loss；
- 横轴明确为 wall-clock time、且总时间少于 45 分钟的曲线；
- 对所有修改的说明；
- leaderboard submission。

需要注意：小规模、短时间训练中有效的技巧，不一定能推广到更大规模预训练。

## 10. Section 7 完整交付清单

- [ ] 实验追踪代码；
- [ ] 完整实验日志文档；
- [ ] 多学习率曲线与搜索策略；
- [ ] 至少一个发散学习率及 edge-of-stability 分析；
- [ ] 达到 TinyStories 目标 validation loss；
- [ ] 多 batch size 曲线与讨论；
- [ ] 至少 256-token 的生成样本或生成到 EOS；
- [ ] 生成流畅性评价及至少两个影响因素；
- [ ] 删除 RMSNorm 的曲线和分析；
- [ ] pre-norm 与 post-norm 对比曲线；
- [ ] RoPE 与 NoPE 对比曲线；
- [ ] 参数量近似匹配的 SwiGLU 与 SiLU 对比；
- [ ] OpenWebText 曲线、生成文本和数据集差异分析；
- [ ] 自选改进、45 分钟曲线、最终 validation loss 和 leaderboard 提交。

## 11. 当前仓库状态与首要问题

### 11.1 已经完成的部分

当前 `experiment01` 已完成一次低资源 TinyStories 训练：

| 项目 | 当前结果 |
| --- | ---: |
| batch size | 32 |
| steps | 5,000 |
| context length | 256 |
| 总 token 数 | 40,960,000 |
| 参数量 | 22,696,448 |
| 最后记录的 train loss | 1.8932 |
| 总训练时间 | 9,417 秒，约 2 小时 37 分 |

模型已经能够生成类似英语故事的短文本，说明训练主路径基本工作。

### 11.2 当前 validation loss 无效

`experiment01` 记录的最终 validation loss 是 8.9161，但这个数字不能用于 Section 7。

原因是：

- 训练集使用 `tinystories_train_tokenizer.json`；
- 验证集使用另一套独立训练的 `tinystories_valid_tokenizer.json`；
- 两者的 vocab 和 merges 均不相同；
- 因而相同 token ID 在训练和验证阶段代表不同字节串。

语言模型的输出维度虽然仍然是 10,000，但 ID 语义已经错位，所以 validation loss 没有可解释性。

正确做法是：只在训练集上训练 tokenizer，然后使用**同一份训练 tokenizer**分别编码训练集和验证集。现有模型不一定需要重新训练；可以先用训练 tokenizer 重新编码 validation 文本，再重新评估 `ckpt_final.pt`。

在修正验证数据前，不应该开始学习率优劣、batch size 或架构消融的正式结论，因为所有实验的主要比较指标都不可靠。

## 12. 推荐执行顺序

1. 使用训练 tokenizer 重新编码 TinyStories validation；
2. 用现有 `ckpt_final.pt` 重新计算可信的 validation loss；
3. 做单 minibatch overfit 测试；
4. 完成绘图脚本，让 `metrics.jsonl` 能按 step 和 elapsed time 绘图；
5. 做短程学习率 sweep，找到稳定区间和发散边界；
6. 用候选学习率跑完整低资源实验；
7. 在固定 token 预算下比较 batch size；
8. 保存至少 256-token 的生成样本及 sampling 参数；
9. 依次完成四个架构消融，每次只改变一个因素；
10. 资源允许时进行 OpenWebText 和 leaderboard 实验。

## 13. 推荐实验记录模板

每次实验可以使用下面的模板：

```markdown
## run_name

### 假设

这次实验想验证什么？预期会发生什么？

### 与 baseline 的唯一差异

- 例如：alpha_max 从 3e-4 改为 1e-3。

### 固定条件

- dataset/tokenizer：
- model config：
- token budget：
- seed：
- validation batches：

### 结果

- final train loss：
- best/final validation loss：
- total time：
- tokens/s：
- 是否稳定：
- checkpoint：
- learning curve：

### 观察与结论

- 曲线说明了什么？
- 是否支持原假设？
- 下一次实验应该改变什么？
```

## 14. 一句话总结

Section 7 的重点不是“再训练一次模型”，而是建立可信的验证和日志基础，在控制变量的前提下系统比较超参数与架构，并用 learning curves、生成样本和实验记录支持自己的结论。
