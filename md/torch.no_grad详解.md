# `torch.no_grad()` 详解

> 通用 PyTorch API 参考，配合 [PyTorch_Tensor基础.md](PyTorch_Tensor基础.md) 使用。所有例子都和作业无关，输出均已实际运行验证。

## 一、一句话

`torch.no_grad()` 是一个**上下文管理器**：在它的作用域内，所有张量运算**不被 autograd 记录**——不保存反向传播需要的中间结果，也不构建计算图。

```python
with torch.no_grad():
    ...   # 这里面的运算，autograd 全程不参与
```

## 二、具体关闭了什么

```python
x = torch.ones(3, requires_grad=True)

y = x * 2
with torch.no_grad():
    z = x * 2

y.requires_grad   # True
z.requires_grad   # False
y.grad_fn          # <MulBackward0 object at ...>   ← 记录了怎么算出来的
z.grad_fn          # None                            ← 没有记录
```

两个直接后果：

1. **不保存中间结果** → 省显存（推理时最大的收益）
2. **不构建 grad_fn 链** → 省一点时间，且结果无法 `.backward()`

## 三、三个典型用途

### 1. 推理 / 验证

算 validation loss、生成文本时不需要梯度。不加 `no_grad` 的话，PyTorch 会为每一层保存激活值，**显存可能翻好几倍**——序列一长直接 OOM。

### 2. 优化器更新参数 / 手动修改梯度

参数是 `requires_grad=True` 的**叶子节点**，直接原地修改会被 autograd 拦下：

```python
p = torch.nn.Parameter(torch.ones(3))
p -= 0.1
# RuntimeError: a leaf Variable that requires grad is being used in an in-place operation.
```

包进 `no_grad` 就合法了：

```python
with torch.no_grad():
    p -= 0.1
# 正常执行，p 变成 tensor([0.9, 0.9, 0.9])
```

这正是自己实现 `Optimizer.step()`（SGD / AdamW）时必须用到的模式——原地修改 `p` 或 `p.grad`，但不想让这次修改进入计算图。

### 3. 手动初始化 / clamp 权重

比如训练中途想限制某些参数的范围。

## 四、`no_grad` vs `.data` —— 安全性差很多

两者都能绕过 autograd 拦截原地修改，但 `.data` 是遗留 API，它**完全绕过了 autograd 的版本计数器**（version counter）。如果你改的张量正好是反向传播需要用到的值，autograd **察觉不到**，会用被篡改的值算出错误的梯度——且**不会有任何警告**。

实测（`exp` 的反向传播需要用到它自己的输出值）：

```python
a = torch.tensor([1.0], requires_grad=True)
b = a.exp()          # 正确梯度应为 e ≈ 2.71828

# 用 .data 篡改 b
b.data *= 100
b.sum().backward()
a.grad   # 271.828   ← 静默给出错误梯度，无任何异常
```

```python
c = torch.tensor([1.0], requires_grad=True)
d = c.exp()

# 用 no_grad 篡改 d
with torch.no_grad():
    d *= 100
d.sum().backward()
# RuntimeError: one of the variables needed for gradient computation
# has been modified by an inplace operation
```

**`no_grad` 会响亮地报错，`.data` 会安静地给你错的答案。** 这是"静默失败远比响亮失败昂贵"这条原则的又一个例子——同类问题也出现在高级索引的原地写入、`squeeze(dim)` 维度不为 1 时的静默无操作等场景（见 [gather与高级索引详解.md](gather与高级索引详解.md)）。

**结论：写新代码时优先用 `no_grad`，把这道安全网留着。只有明确知道自己在做什么、且性能确实是瓶颈时，才考虑 `.data`。**

## 五、几个容易混淆的近亲

| API | 区别 |
|---|---|
| `torch.inference_mode()` | 比 `no_grad` **更激进**：连版本计数器都不维护，更快更省，但产出的张量**不能再参与任何 autograd 计算**——连 `.requires_grad_(True)` 都不允许。纯推理场景（不会再拿这个结果做任何训练相关操作）用它 |
| `x.detach()` | 作用于**单个张量**，返回一个脱离计算图的副本（**共享内存**）。`no_grad` 作用于**一整块代码** |
| `p.requires_grad_(False)` | **冻结参数**——连梯度都不计算。这是持久的属性修改，不是临时作用域 |
| `model.eval()` | ⚠️ **完全不同的东西**，见下文 |

`inference_mode` 的限制实测：

```python
with torch.inference_mode():
    m = x * 2
m.requires_grad_(True)
# RuntimeError: Setting requires_grad=True on inference tensor
# outside InferenceMode is not allowed.
```

`detach()` 的共享内存特性：

```python
y = x * 2
z = y.detach()
z.requires_grad   # False
z.data_ptr() == y.data_ptr()   # True   ← 同一块内存，只是脱离了计算图
```

## 六、两个高频坑

### 坑 1：`no_grad` ≠ `model.eval()`

- `no_grad` 管的是**要不要记录梯度**
- `model.eval()` 管的是 **Dropout / BatchNorm 的行为模式**（Dropout 在 eval 模式下不再随机丢弃；BatchNorm 改用运行时统计量而非当前 batch 的统计量）

推理时通常**两个都要**。只写 `no_grad` 而忘了 `eval()`，Dropout 仍在随机丢弃神经元，验证 loss 会莫名偏高且每次运行结果都不一样。

> 本次作业的 `transformer_lm` 没有 Dropout 和 BatchNorm，暂时不受影响——但这是写任何 PyTorch 推理代码都要记住的常识。

### 坑 2：不小心把需要梯度的前向包进去了

```python
with torch.no_grad():
    loss = model(x) ...      # ❌ 前向被 no_grad 包住了
loss.backward()
# RuntimeError: element 0 of tensors does not require grad
#               and does not have a grad_fn
```

好在这个错误是**响亮**的，一眼能看出来是哪里的问题。

## 七、用法：上下文管理器 vs 装饰器

**作为 `with` 块**（最常见）：

```python
with torch.no_grad():
    val_loss = compute_loss(model, val_data)
```

**作为装饰器**，让整个函数体都不记录梯度：

```python
@torch.no_grad()
def evaluate(model, data):
    ...
```

实测两种写法效果一致：

```python
@torch.no_grad()
def f(x):
    return x * 2

r = f(x)
r.requires_grad   # False
```

写生成函数、验证函数时用装饰器标注，比每次记得手动加 `with` 更可靠、更不容易漏。

## 八、速查表

```python
# ---- 推理 / 验证 ----
with torch.no_grad():
    output = model(x)

@torch.no_grad()
def evaluate(...): ...

# ---- 优化器手写 step() 里修改参数 ----
with torch.no_grad():
    p -= lr * grad          # 合法；不加 no_grad 会报叶子节点错误

# ---- 单个张量脱离计算图（不是整段代码）----
z = y.detach()               # 共享内存，requires_grad=False

# ---- 纯推理，追求极致性能，且不再需要 autograd ----
with torch.inference_mode():
    output = model(x)

# ---- 冻结参数，连梯度都不算（持久生效，不是临时块）----
p.requires_grad_(False)
```

## 九、易错点清单

1. **`.data` 绕过 autograd 版本计数器**，篡改被反向传播依赖的值会**静默**给出错误梯度。优先用 `no_grad`，它会**响亮**报错。
2. **`no_grad` ≠ `model.eval()`**——两者管的是完全不同的东西，推理时通常都要加。
3. **`inference_mode` 产出的张量不能再进入任何 autograd 计算**，比 `no_grad` 更严格，别在还需要梯度的地方用它。
4. **别把需要 `.backward()` 的前向传播包进 `no_grad`**——虽然报错很直白，但仍是常见的手误。
5. **叶子节点**（`requires_grad=True` 且非计算图产物，如模型参数）**不能直接原地修改**，必须在 `no_grad` 或用 `.data`；后者应作为最后手段而非默认选择。

## 十、相关笔记

- [PyTorch_Tensor基础.md](PyTorch_Tensor基础.md)
- [gather与高级索引详解.md](gather与高级索引详解.md) —— §2.6 原地 vs 拷贝，同一主题的另一种体现
- [4.2_SGD优化器与Optimizer_API详解.md](4.2_SGD优化器与Optimizer_API详解.md) —— `step()` 里原地修改参数的实际应用场景
