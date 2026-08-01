# 计算图与 autograd 机理详解

> 通用 PyTorch API 参考，配合 [torch.no_grad详解.md](torch.no_grad详解.md) 使用。所有例子都和作业无关，输出均已实际运行验证。

## 一、一句话

PyTorch 的自动微分（autograd）机制在你写**前向计算**代码的同时，悄悄在背后搭建一张记录"每个值是怎么算出来的"的有向图——**计算图（computation graph）**。`loss.backward()` 做的事就是沿着这张图**反向**走一遍，用链式法则算出 loss 对每个参数的偏导数。

```python
logits = body(inputs)              # 前向：一边计算数值，一边偷偷建图
loss = cross_entropy(logits, targets)
loss.backward()                     # 反向：沿着刚才建的图，反着走一遍求梯度
```

## 二、计算图长什么样

计算图的节点是张量，边是"由哪个操作算出来的"。以最简单的例子说明：

```python
x = torch.tensor([2.0], requires_grad=True)
y = x ** 2
z = y * 3
```

对应的图：

```
x --(pow, 指数2)--> y --(mul, 乘3)--> z
```

每个由运算产生的张量都带一个 `.grad_fn` 属性，记录"我是被哪个操作、用哪些输入算出来的"：

```python
z.grad_fn                        # <MulBackward0 object at ...>
z.grad_fn.next_functions         # ((<PowBackward0 ...>, 0), (None, 0))
y.grad_fn                        # <PowBackward0 object at ...>
y.grad_fn.next_functions         # ((<AccumulateGrad ...>, 0),)
```

`next_functions` 就是这个节点在图上指向的"上游"——`z` 的上游是算出 `y` 的 `PowBackward0`；`y` 的上游是 `AccumulateGrad`，这是叶子节点专属的终点标记（见下一节）。图是在**前向传播执行的过程中**动态搭建出来的，每执行一个张量运算就多一个节点，而不是提前静态声明好的。

## 三、叶子节点 vs 非叶子节点

```python
x.is_leaf   # True  —— x 是用户直接创建、requires_grad=True 的张量
y.is_leaf   # False —— y 是运算的产物
```

- **叶子节点（leaf）**：用户直接创建的、`requires_grad=True` 的张量——典型例子就是模型的参数（`nn.Parameter`）。它们是计算图的"起点"，没有 `grad_fn`（或者说 `grad_fn` 是 `None`），取而代之的是一个特殊标记 `AccumulateGrad`，表示"梯度算到这里就该往 `.grad` 里累加，不用再往上游传了"。
- **非叶子节点**：所有由运算产生的中间张量（比如 `y`、`z`），它们有 `grad_fn`，记录着自己的来源。

**只有叶子节点的梯度会被保留在 `.grad` 里**，非叶子节点默认不保留：

```python
z.backward()
x.grad   # tensor([12.])   —— dz/dx = d(3x²)/dx = 6x = 12，正确
y.grad   # None，并且访问它会触发 UserWarning
```

这也是为什么优化器只关心 `model.parameters()`（全部是叶子节点）的 `.grad`——中间层的激活值本来就不需要长期保留梯度,如果确实需要看某个中间张量的梯度（调试用），可以显式调用 `y.retain_grad()`。

## 四、梯度是"累加"的，不是"覆盖"的

```python
x = torch.tensor([1.0], requires_grad=True)
y = x * 2
y.backward()
x.grad                # tensor([2.])

y2 = x * 2
y2.backward()
x.grad                # tensor([4.])   ← 不是 2，是 2+2 累加起来的！
```

`backward()` 每次都是把新算出的梯度**加到** `.grad` 现有值上，而不是覆盖。这正是训练循环里第 268 行 `optimizer.zero_grad(set_to_none=True)` 必须放在 `loss.backward()` **之前**执行的原因——不清零，第 t 步实际用来更新参数的就是"前 t 步梯度之和"，方向完全错误。这个累加设计不是 bug，而是刻意为之：它是为了支持"梯度累积"（gradient accumulation）这种技巧——想要更大的有效 batch size 但显存不够时，可以连续跑几个小 batch 都不清零、只在最后一次才清零+更新，效果上近似于一次大 batch 的梯度。

## 五、计算图用完即焚：为什么不能 `backward()` 两次

```python
x = torch.tensor([1.0], requires_grad=True)
y = x ** 2
y.backward()
y.backward()
# RuntimeError: Trying to backward through the graph a second time
# (or directly access saved tensors after they have already been freed).
```

默认情况下，`backward()` 执行完之后，这张计算图占用的中间结果（反向传播需要用到的"保存的张量"，比如某些操作的输入值）会被**释放**，为的是节省显存——训练一个大模型时，如果每一步的计算图都无限期保留着，显存很快就会耗尽。

如果确实需要对同一张图反向传播多次（比较少见，比如同一个前向结果要算多个不同的 loss 各自反传），需要显式传 `retain_graph=True`：

```python
y.backward(retain_graph=True)   # 这次不释放
y.backward()                     # 图还在，可以再传一次
x.grad                           # tensor([4.])，两次调用的梯度按累加语义叠加
```

正常的训练循环里**不需要**这个参数——每一步都是"重新前向（建一张新图）→ 反向（用完就释放）"，图从不跨步复用。如果你的代码报了这个错，几乎总是意味着某个 Tensor（比如某个中间结果或 loss）被跨步保留、又被拿去 `backward()` 了第二次，这通常是写训练循环时的逻辑错误，而不是需要用 `retain_graph=True` 来"修复"的场景。

## 六、`backward()` 只能对标量调用（隐式情况下）

```python
x = torch.tensor([1.0, 2.0], requires_grad=True)
y = x ** 2
y.backward()
# RuntimeError: grad can be implicitly created only for scalar outputs
```

`backward()` 本质上算的是"给定输出的梯度，反推每个上游变量的梯度"——如果不显式告诉它"输出这边的梯度是多少"，它默认假设你在对一个**标量**求梯度，隐式地把"起点梯度"设为 `1.0`（对应数学上 `d(loss)/d(loss) = 1`）。这也是为什么 `loss` 必须是一个 0 维的标量——`cross_entropy` 内部用 `torch.mean(...)` 把所有位置的损失平均成一个数，就是为了保证能直接 `.backward()`。

如果输出不是标量，必须显式传入一个和输出同形状的"梯度种子"：

```python
y.backward(torch.ones_like(y))
x.grad   # tensor([2., 4.])   —— 相当于对 y.sum() 求梯度
```

日常写训练代码几乎不会用到这个用法——只要保证 loss 是标量（几乎所有训练场景都是如此），直接 `loss.backward()` 就够了。

## 七、和这个训练脚本的对应关系

结合 `cs336_basics/train.py` 主循环（第 264-283 行）来看，一步训练里计算图经历的完整生命周期：

```python
logits = body(inputs)              # 建图：从 inputs 到 logits 的整条链路被记录
loss = cross_entropy(logits, targets)   # 继续建图：logits → loss，且 loss 是标量

optimizer.zero_grad(set_to_none=True)   # 清空叶子节点（模型参数）的 .grad，为这次反传腾地方
loss.backward()                          # 反向：沿着图从 loss 走到每个参数，把梯度存进 .grad；走完图被释放

gradient_clipping(body.parameters(), args.grad_clip)   # 读取/修改 .grad，不涉及计算图
optimizer.step()                         # 读取 .grad + group["lr"]，更新参数数值（脱离计算图，在 no_grad 语境下进行）
```

- **每一步都是全新的一张图**：`step` 循环下一轮重新调用 `body(inputs)`，会重新构建一张全新的计算图，和上一步的图完全无关（旧图早已在上一次 `backward()` 后被释放）。
- **`.grad` 是唯一"跨图存活"的东西**：参数的 `.grad` 属性在多次 `backward()` 之间是持久的（除非手动清零），这也是为什么"忘记清零"会导致梯度污染——它不属于计算图本身，图释放不影响它。

## 八、易错点清单

1. **梯度默认累加，不清零就是"前 t 步梯度之和"**——`zero_grad()` 必须在每次 `backward()` 之前调用。
2. **只有叶子节点的 `.grad` 会被保留**——中间张量默认没有 `.grad`，需要就显式 `retain_grad()`。
3. **计算图默认用完即焚**——同一张图不能 `backward()` 两次，除非显式 `retain_graph=True`（正常训练循环用不到这个参数）。
4. **`backward()` 隐式调用要求输出是标量**——loss 函数设计时要确保最终返回单个数值（如用 `mean`/`sum` 聚合），否则要手动传梯度种子。
5. **"建图"只在 `requires_grad=True` 的张量参与运算时发生**——如果某段代码被 `torch.no_grad()` 包住，或者输入张量本身 `requires_grad=False`，PyTorch 根本不会记录这段计算的 `grad_fn`，之后对它 `.backward()` 会报"没有 grad_fn"的错误（见 [torch.no_grad详解.md](torch.no_grad详解.md) 的坑 2）。

## 九、相关笔记

- [torch.no_grad详解.md](torch.no_grad详解.md) —— 如何临时关闭计算图的记录，以及和 `model.eval()`、`.detach()` 的区别
- [train.py函数与变量作用整理.md](train.py函数与变量作用整理.md) —— 主循环里 `backward()` 前后的完整步骤上下文
