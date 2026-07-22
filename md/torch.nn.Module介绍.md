# `torch.nn.Module` 介绍

> 通用 PyTorch API 参考,配合 [PyTorch_Tensor基础.md](PyTorch_Tensor基础.md) 使用。所有例子都是自己构造的通用例子,和作业无关。

## 1. 是什么

`nn.Module` 是 PyTorch 里**所有神经网络层/模型的基类**。它本身不定义具体计算,而是提供一整套"管理基础设施":自动追踪参数、自动追踪子模块、批量迁移设备/精度、保存和加载状态、递归遍历整个模型结构。你写的每一个层(`Linear`、`RMSNorm`……)以及最终的整个模型,都要继承它。

## 2. 基本结构

```python
import torch
import torch.nn as nn

class MyLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()          # 必须调用,注册机制靠它初始化
        self.weight = nn.Parameter(torch.randn(dim))

    def forward(self, x):
        return x * self.weight
```

**三个要点**:
- 继承 `nn.Module`
- `__init__` 里第一行调用 `super().__init__()`——这一步在内部建立好参数/子模块的追踪表,漏掉这一步后续的自动注册机制会失效
- 定义 `forward` 方法,描述"输入怎么变成输出"

**调用模块时,用 `layer(x)`,不要直接调 `layer.forward(x)`**:

```python
>>> layer = MyLayer(4)
>>> layer(torch.randn(4))     # 正确:触发 __call__,间接调用 forward,还会顺带处理 hook 等机制
>>> layer.forward(torch.randn(4))  # 能跑,但绕过了 __call__ 里的额外逻辑,不推荐
```

## 3. 参数自动注册:`nn.Parameter`

只要把一个 `nn.Parameter` 对象赋值给 `self.某个属性`,`nn.Module` 就会自动把它记录下来:

```python
class MyLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(dim))   # 自动被记录

>>> layer = MyLayer(4)
>>> list(layer.parameters())
[Parameter containing: tensor([...], requires_grad=True)]
```

`nn.Parameter` 是 `torch.Tensor` 的子类,唯一的区别是它默认 `requires_grad=True`,并且被 `nn.Module` 特殊对待(自动出现在 `.parameters()` 里)。**普通 tensor 赋值给 `self.xxx` 不会被自动记录**——这是判断"这个数据是不是可训练参数"的分界线。

## 4. 子模块自动注册(嵌套)

`nn.Module` 也可以包含**别的 `nn.Module`** 作为属性,同样会被自动记录:

```python
class Block(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.layer1 = MyLayer(dim)
        self.layer2 = MyLayer(dim)

    def forward(self, x):
        return self.layer2(self.layer1(x))

>>> block = Block(4)
>>> list(block.parameters())     # 自动收集 layer1 和 layer2 各自的参数,一共 2 个
[Parameter containing: ..., Parameter containing: ...]
```

这个"嵌套自动收集"的机制,正是搭建 Transformer(block 包含 attention/FFN,模型包含多个 block)的基础——最外层模型调用 `.parameters()`,能拿到**整棵树**上所有层的所有参数,不需要手动一层层去收集。

## 5. 遍历参数和子模块

```python
>>> list(block.parameters())              # 所有参数(tensor 列表,不带名字)
>>> list(block.named_parameters())        # 带名字: [("layer1.weight", tensor), ("layer2.weight", tensor)]
>>> list(block.children())                 # 直接子模块(只有一层): [layer1, layer2]
>>> list(block.named_children())           # 带名字的直接子模块
>>> list(block.modules())                  # 递归所有模块(包括自己): [block, layer1, layer2]
```

`named_parameters()` 在调试时特别有用——想知道某个参数具体来自模型的哪一层,看名字就知道。

## 6. 容器类:`nn.ModuleList` / `nn.Sequential` / `nn.ModuleDict`

讲义明确允许使用这几个"容器类"(它们本身不含可学习逻辑,只是帮你管理一组子模块)。

**`nn.ModuleList`**:像 Python 列表一样存一组模块,但会被正确注册(普通 Python `list` 存 `nn.Module` **不会**被自动注册,参数会"隐身"、`.parameters()` 找不到它们——这是个常见陷阱):

```python
class Stack(nn.Module):
    def __init__(self, dim, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([MyLayer(dim) for _ in range(num_layers)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
```

这正是"堆叠 `num_layers` 个 Transformer block"这类场景的标准写法。

**`nn.Sequential`**:比 `ModuleList` 更进一步,连 `forward` 里的循环调用都帮你写好了(按顺序依次调用):

```python
seq = nn.Sequential(MyLayer(4), MyLayer(4))
seq(torch.randn(4))    # 自动依次调用 layer1、layer2
```

适合"纯粹按顺序、没有任何分支/额外逻辑"的场景;如果 block 之间要传额外参数(比如 RoPE 需要的位置信息),`Sequential` 的固定调用方式可能不够灵活,这时候用 `ModuleList` 自己写循环更合适。

## 7. 设备和精度迁移

```python
>>> layer = MyLayer(4)
>>> layer.to("cuda")           # 递归地把所有参数和 buffer 都搬到 GPU 上
>>> layer.to(torch.float64)     # 递归地把所有参数和 buffer 都转成 float64
>>> layer.cuda()                 # .to("cuda") 的简写
>>> layer.cpu()                  # 搬回 CPU
```

对最外层模型调用一次 `.to(...)`,会**递归**作用到所有子模块、所有参数、所有 buffer,不需要一个个手动处理。

## 8. 保存与加载:`state_dict` / `load_state_dict`

`state_dict()` 把整个模块(及其所有子模块)的参数,导出成一个"名字 → tensor"的字典;`load_state_dict()` 反过来把这样一个字典的值加载进模块:

```python
>>> sd = layer.state_dict()
>>> sd
OrderedDict([('weight', tensor([...]))])

>>> new_layer = MyLayer(4)
>>> new_layer.load_state_dict(sd)     # 把 sd 里的值加载进 new_layer 对应名字的参数
```

**这个方法在写测试 adapter 时会直接用到**:讲义在 `linear` 那道题里明确提示——"adapter 应该把给定的权重加载进你的 `Linear` 模块,你可以用 `Module.load_state_dict` 来做这件事"。也就是说,测试会提供一份**参考权重**,你的 adapter 需要把这份权重塞进你自己构造的模块实例里,再跑 `forward` 比对输出——`load_state_dict` 就是"塞权重"这一步用的工具。

**注意**:`load_state_dict` 要求字典里的**名字**和你模块里参数的名字**完全对应**——这也是为什么讲义反复强调"变量该叫 `W`"这类命名细节,命名和测试提供的 `state_dict` 对不上,加载就会报错。

## 9. `register_buffer`(简要,详见 Tensor 基础文件)

和 `nn.Parameter` 类似,但**不参与训练**(不出现在 `.parameters()` 里、不参与梯度更新),只是"跟着模块走"的固定数据(比如 RoPE 预先算好的 cos/sin 表)。同样会被 `.to(device)` 递归迁移。

## 10. `.train()` / `.eval()` 模式

```python
>>> model.train()    # 切换到训练模式
>>> model.eval()      # 切换到评估/推理模式
```

某些层(比如 Dropout、BatchNorm)在训练和推理时行为不同,这两个方法负责切换这个全局标志(同样会递归作用到所有子模块)。本次作业里用到的 RMSNorm/Attention 等组件在这两种模式下行为一致,但写代码时保持这个惯例是好习惯,以后遇到有状态差异的层就不会漏掉。

## 11. 常见陷阱

- **忘记调用 `super().__init__()`**:参数注册机制不会初始化,后续给 `self.xxx` 赋值 `nn.Parameter` 时会直接报错
- **用普通 Python `list`/`dict` 存一组 `nn.Module`**,而不是 `nn.ModuleList`/`nn.ModuleDict`:模块本身能正常调用,但**不会**被注册,`.parameters()`、`.to(device)`、`state_dict()` 都会"看不见"这些参数——这是最容易踩、也最难发现的坑之一(代码能跑,但训练时这部分参数根本没有被优化器更新)
- **`state_dict()` 的 key 和模块内部属性名不一致**:加载会直接报 `KeyError` 或形状不匹配的错误
