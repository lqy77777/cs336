# `register_buffer` 详解

> 通用 PyTorch API 参考,配合 [torch.nn.Module介绍.md](torch.nn.Module介绍.md) 使用。所有例子都和作业无关。

## 是什么

`register_buffer` 是 `nn.Module` 提供的一个方法,用来注册一个"跟着模块走,但不参与训练"的 tensor——它解决的是一个具体的空白:`nn.Parameter` 太"重"(会被训练),普通属性又太"轻"(不会被 `nn.Module` 的管理机制追踪)。

## 基本用法

```python
class MyLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("scale", torch.ones(4))
```

调用之后,`self.scale` 就能像普通属性一样访问(`self.scale`),但它被 `nn.Module` **正式追踪**了。

## 它解决的问题:普通属性"不被追踪"

如果直接写 `self.scale = torch.ones(4)`(普通赋值,不用 `register_buffer`),这个 tensor 对 `nn.Module` 来说是**隐形的**——回忆一下 `nn.Parameter` 那部分的实验:普通 tensor 属性不会出现在 `.parameters()` 里,同理它也**不会**跟着 `model.to("cuda")` 一起被搬到 GPU 上,也**不会**被存进 `state_dict()`。如果模块的 `forward` 里用到这个 tensor,一旦模型整体搬去了 GPU、这个 tensor 却还留在 CPU 上,就会报 device 不匹配的错误。

`register_buffer` 就是专门解决这个问题的——让这个 tensor 也享受"跟着模块走"的待遇,但**不参与训练**(不会出现在 `.parameters()` 里,优化器看不到它,也不会计算梯度)。

## `persistent` 参数:要不要存进 checkpoint

```python
self.register_buffer("scale", torch.ones(4), persistent=True)   # 默认值
self.register_buffer("scale", torch.ones(4), persistent=False)
```

- **`persistent=True`(默认)**:这个 buffer 会被包含进 `state_dict()`,也就是说存模型 checkpoint 时会把它的值也存进去,加载时也会去 checkpoint 里找它
- **`persistent=False`**:这个 buffer **依然会**跟着 `.to(device)` 走、依然能通过 `self.scale` 访问,但**不会**出现在 `state_dict()` 里——既不会被存进 checkpoint,加载 checkpoint 时也不会去找它、更不会因为它"缺失"而报错

## 什么场景该用 `persistent=False`

典型场景:模块内部需要一个**完全确定性算出来的**中间量(比如某种预计算表)——只要给定构造时的超参数,任何时候都能重新算出**一模一样**的值,不依赖任何训练数据,也不是学出来的。既然随时能重新算,存进 checkpoint 就是纯粹的浪费(占用磁盘空间,存的还是一份可以瞬间重算出来的东西)。

还有一个更实际的原因:如果之后要把这个模块装进一个更大的模型、加载**外部提供的参考权重**做测试——那份参考 `state_dict` 通常**不会包含**这种"实现细节相关"的预计算表(它只包含真正训练出来的权重)。如果这个模块把这张表注册成 `persistent=True`(或者更糟,注册成 `nn.Parameter`),`load_state_dict` 就会因为"缺了这个 key"或者"多了这个 key"而报错。用 `persistent=False`,这张表压根不会被 `state_dict()`/`load_state_dict()` 关心,也就不会有这个冲突。

## 一张表总结四种选择的区别

| | 参与训练(出现在 `.parameters()`) | 跟着 `.to(device)` 走 | 存进 `state_dict()` |
|---|---|---|---|
| `nn.Parameter` | ✓ | ✓ | ✓ |
| `register_buffer(persistent=True)`(默认) | ✗ | ✓ | ✓ |
| `register_buffer(persistent=False)` | ✗ | ✓ | ✗ |
| 普通属性(`self.x = tensor`) | ✗ | ✗ | ✗ |
