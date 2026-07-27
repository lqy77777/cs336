# 激活函数与门控前馈网络:ReLU / SiLU / GLU / SwiGLU

> 通用概念参考,配合 [transformer.py开发知识点总结.md](transformer.py开发知识点总结.md) 使用。本文只讲数学定义与设计动机,不涉及作业实现。

## 零、先明确一件事:为什么需要非线性

若前馈网络只是 $W_2(W_1 x)$,那么无论堆多少层,整体始终等价于**一个**线性变换 $W x$——深度带来的表达力为零。激活函数的作用就是在两次线性变换之间插入非线性,让网络能逼近任意函数。

下文的四个名字里,前两个(ReLU、SiLU)是**逐元素的激活函数**,后两个(GLU、SwiGLU)是**整层的结构**。这是最容易混淆的一点:它们不在同一个抽象层级上。

---

## 一、ReLU(Rectified Linear Unit,修正线性单元)

$$\mathrm{ReLU}(x) = \max(0, x)$$

负数全部截断为 0,正数原样通过。

**优点**

- **极其便宜**:一次比较,没有指数、没有除法。
- **缓解梯度消失**:正区间导数恒为 1,梯度反向传播时不衰减。这是它当年取代 sigmoid/tanh 的核心原因——后者在饱和区导数趋近 0,深层网络根本训不动。
- **稀疏激活**:大约一半的神经元输出为 0,带来某种正则效果。

**缺点**

- **Dying ReLU(神经元死亡)**:一旦某个神经元的输入长期为负,它的梯度恒为 0,权重再也不会更新,这个神经元就永久失效了。
- **在 $x=0$ 处不可导**:实践中直接约定该点导数取 0 或 1(次梯度),工程上不构成问题,但数学上不光滑。
- **输出非零中心**:输出恒非负,会让下一层的梯度方向产生系统性偏置。

**历史地位**:2017 年原始 Transformer 论文的 FFN 用的就是 ReLU,即 $\mathrm{FFN}(x) = W_2\,\mathrm{ReLU}(W_1 x)$。

---

## 二、SiLU / Swish(Sigmoid Linear Unit)

$$\mathrm{SiLU}(x) = x \cdot \sigma(x), \qquad \sigma(x) = \frac{1}{1 + e^{-x}}$$

其中 $\sigma$ 是 sigmoid 函数,把任意实数压到 $(0, 1)$。

**怎么理解这个式子**:$\sigma(x)$ 是一个取值在 0 到 1 之间的**门**,而这个门的开合程度由 $x$ **自己**决定——所以 SiLU 被称为 **self-gating(自门控)**。$x$ 很大时 $\sigma(x) \to 1$,输出 $\approx x$(像 ReLU 的正区间);$x$ 很负时 $\sigma(x) \to 0$,输出 $\to 0$(像 ReLU 的负区间)。所以 **SiLU 是 ReLU 的平滑版本**。

**关键性质**

- **处处光滑可导**,没有 ReLU 在原点的折角。
- **非单调**:在 $x \approx -1.278$ 处取到最小值 $\approx -0.278$,然后回升。这是它和 ReLU 最本质的形态差异——负区间不是简单归零,而是保留了一小段"凹陷"。
- **负区间梯度不为零**:缓解了 dying ReLU 问题,处于负区间的神经元仍能恢复。
- **代价**:需要算 $e^{-x}$,比 ReLU 贵。在 memory-bound 的场景下这个开销常被访存掩盖,但在小算子上是可测的。

**SiLU 和 Swish 是同一个东西吗?**

基本是,名字来自两条独立的研究线索:

- **SiLU** 由 Elfwing 等人(2017)在强化学习背景下提出(更早 Hendrycks & Gimpel 的 GELU 论文里也提到过这个形式)。
- **Swish** 由 Ramachandran 等人(2017)通过**自动搜索**激活函数发现,一般形式是 $x \cdot \sigma(\beta x)$,其中 $\beta$ 可以是可学习参数。

当 $\beta = 1$ 时 Swish 就是 SiLU。$\beta \to \infty$ 时 $\sigma(\beta x)$ 趋近阶跃函数,Swish 退化为 ReLU;$\beta = 0$ 时退化为线性函数 $x/2$。所以 $\beta$ 是一个在"线性"和"ReLU"之间连续插值的旋钮。现代 LLM 里几乎都固定 $\beta = 1$,两个名字就通用了。

**顺带一提 GELU**(同一家族的兄弟):$\mathrm{GELU}(x) = x \cdot \Phi(x)$,其中 $\Phi$ 是标准正态的累积分布函数。形状和 SiLU 极其接近,BERT 和 GPT-2 用的是它。

---

## 三、GLU(Gated Linear Unit,门控线性单元)

**注意:这不是一个激活函数,而是一整个层的结构。**

$$\mathrm{GLU}(x, W, V) = \sigma(xW) \odot (xV)$$

其中 $\odot$ 是逐元素相乘(Hadamard 积)。

出自 Dauphin 等人(2016)的门控卷积语言模型。它的做法是:把输入用**两组不同的权重**各投影一次,得到两路:

- **门(gate)路**:$xW$ 过 sigmoid,变成 $(0,1)$ 之间的系数;
- **内容(value)路**:$xV$ **不过任何非线性**——"Gated **Linear** Unit"里的 "linear" 指的就是这一路。

然后逐元素相乘:门决定内容里每个通道**能通过多少**。

**这和逐元素激活函数有什么本质区别?**

ReLU/SiLU 施加的是**固定的**逐元素变换——同一个位置的输出只取决于它自己的输入值。GLU 引入的是**乘性交互**:门的值由**另一路投影**算出,而那一路看到的是**整个输入向量**。于是"某个通道是否通过"变成了一个数据相关、且依赖全局上下文的决策。这种乘性门控的表达力严格强于逐元素非线性。

**代价**:两路投影意味着参数量翻倍。

---

## 四、SwiGLU

把 GLU 里的 sigmoid 门换成 SiLU/Swish 门,就得到 SwiGLU:

$$\mathrm{SwiGLU}(x, W_1, W_3) = \mathrm{SiLU}(xW_1) \odot (xW_3)$$

放进前馈网络里,完整的一层是:

$$\mathrm{FFN}_{\mathrm{SwiGLU}}(x) = W_2\Big(\mathrm{SiLU}(x W_1) \odot (x W_3)\Big)$$

出自 Shazeer(2020)的《GLU Variants Improve Transformer》。那篇论文系统地把 GLU 的门控函数换成各种激活(ReGLU、GEGLU、SwiGLU……),发现门控变体普遍优于原始 FFN。

论文里有一句名句,值得记住:

> "We offer no explanation as to why these architectures seem to work; we attribute their success, as all else, to divine benevolence."
> (我们无法解释这些架构为什么有效;和其它一切一样,我们把它的成功归因于神的恩典。)

也就是说——**这是纯粹的经验发现,至今没有令人满意的理论解释**。

### 关键细节:为什么 $d_{ff} = \frac{8}{3} d_{model}$

原始 FFN 只有 **2 个**权重矩阵,SwiGLU 有 **3 个**。若沿用 $d_{ff} = 4 d_{model}$,参数量会平白多出 50%,和基线的对比就不公平了。于是:

- 原始 FFN 参数量:$2 \times d_{model} \times 4 d_{model} = 8 d_{model}^2$
- SwiGLU 参数量:$3 \times d_{model} \times d_{ff}$

令两者相等,解得 $d_{ff} = \frac{8}{3} d_{model} \approx 2.67 d_{model}$。

这就是各种实现里那个看起来很怪的 $8/3$ 的来历。实际工程中还会把结果**向上取整到 64 的倍数**,以便对齐 GPU 的 tile 尺寸、让 GEMM 跑在高效路径上。

### 谁在用

LLaMA 全系、PaLM、Mistral、Qwen、Gemma 等主流开源与闭源模型的 FFN 基本都是 SwiGLU,且**不带 bias**。

---

## 五、一张表理清关系

| 名称 | 公式 | 层级 | 权重矩阵数(FFN 中) | 门控 | 光滑 | 代表模型 |
|---|---|---|---|---|---|---|
| ReLU | $\max(0,x)$ | 逐元素激活 | 2 | ✗ | ✗ | 原始 Transformer |
| GELU | $x \cdot \Phi(x)$ | 逐元素激活 | 2 | 自门控 | ✓ | BERT、GPT-2 |
| SiLU / Swish | $x \cdot \sigma(x)$ | 逐元素激活 | 2 | 自门控 | ✓ | (作为 SwiGLU 的组件) |
| GLU | $\sigma(xW) \odot (xV)$ | 层结构 | 3 | ✓ | ✓ | 门控卷积语言模型 |
| SwiGLU | $\mathrm{SiLU}(xW_1) \odot (xW_3)$ | 层结构 | 3 | ✓ | ✓ | LLaMA、PaLM、Qwen |

**记忆线索**:

- **Swi**GLU = **Swi**sh + **GLU** —— 名字本身就是配方。
- 逐元素激活 → 2 个矩阵,$d_{ff} = 4d$;门控结构 → 3 个矩阵,$d_{ff} = \frac{8}{3}d$。

---

## 六、常见困惑澄清

**Q: GLU 的名字里为什么有 "linear"?**
因为内容那一路 $xV$ 是纯线性的,不过任何激活。非线性完全来自门那一路,以及两路相乘这个动作本身。

**Q: SiLU 的负值区间会不会造成问题?**
不会,反而是优点。它让处于负区间的神经元保留了非零梯度,不至于像 ReLU 那样"死掉"。那个 $\approx -0.278$ 的最小值也给了网络表达"轻微负响应"的能力。

**Q: 门控为什么更强?**
关键在**乘性**。逐元素激活是加性结构里的固定弯折;门控是让一部分网络输出去**调制**另一部分,这类似于注意力机制的思想——用数据决定信息流的通断。

**Q: 现代实现为什么都去掉 bias?**
bias 参数量占比极低但收益难以观测,还额外占用优化器状态和通信带宽。PaLM、LLaMA 等都验证了去掉后质量无损。

**Q: PyTorch 里有现成的吗?**
有:`torch.nn.functional.relu`、`torch.nn.functional.silu`、`torch.nn.functional.gelu`、`torch.nn.GLU`。**但 CS336 作业要求自己实现**,这里只作对照参考。

---

## 七、延伸阅读

- Dauphin et al., *Language Modeling with Gated Convolutional Networks* (2016) —— GLU 的出处
- Ramachandran et al., *Searching for Activation Functions* (2017) —— Swish 的出处
- Shazeer, *GLU Variants Improve Transformer* (2020) —— SwiGLU 的出处,只有 5 页,强烈建议通读
- 课程主页 cs336.stanford.edu 对应讲次的 slides
