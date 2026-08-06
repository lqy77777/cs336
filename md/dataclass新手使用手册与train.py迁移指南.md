# dataclass 新手使用手册与 train.py 改造指南

## 1. 最终目标

训练程序只使用一个 TrainConfig 数据类管理全部配置。

配置流程如下：

~~~text
make_config()
    ↓
TrainConfig
    ↓
main(config)
    ↓
训练、验证、日志和 checkpoint
~~~

核心结构：

~~~python
from dataclasses import dataclass


@dataclass
class TrainConfig:
    train_data: str
    val_data: str
    out_dir: str
    batch_size: int = 32


def make_config() -> TrainConfig:
    return TrainConfig(
        train_data="data/train.bin",
        val_data="data/valid.bin",
        out_dir="runs/baseline",
    )


def main(config: TrainConfig) -> int:
    print(config.batch_size)
    return 0


if __name__ == "__main__":
    config = make_config()
    raise SystemExit(main(config))
~~~

所有训练参数都在 Python 代码中设置。要修改 batch size，只需要修改：

~~~python
batch_size=64
~~~

---

# 第一部分：dataclass 基础

## 2. dataclass 是什么

dataclass 是 Python 标准库提供的数据类，适合表示一组有名称、有类型、有默认值的数据。

~~~python
from dataclasses import dataclass


@dataclass
class ExampleConfig:
    batch_size: int = 32
    learning_rate: float = 3e-4
    device: str = "cpu"
~~~

创建对象：

~~~python
config = ExampleConfig()
~~~

读取字段：

~~~python
print(config.batch_size)       # 32
print(config.learning_rate)    # 0.0003
print(config.device)           # cpu
~~~

覆盖默认值：

~~~python
config = ExampleConfig(
    batch_size=64,
    device="cuda",
)
~~~

打印对象：

~~~python
print(config)
~~~

结果类似：

~~~text
ExampleConfig(batch_size=64, learning_rate=0.0003, device='cuda')
~~~

## 3. 字段、类型和默认值

下面一行表示一个配置字段：

~~~python
batch_size: int = 32
~~~

| 部分 | 含义 |
|---|---|
| batch_size | 字段名称 |
| int | 类型标注 |
| 32 | 默认值 |

### 3.1 必填字段

没有默认值的字段必须在创建对象时提供：

~~~python
@dataclass
class TrainConfig:
    train_data: str
    out_dir: str
    batch_size: int = 32
~~~

创建：

~~~python
config = TrainConfig(
    train_data="data/train.bin",
    out_dir="runs/test",
)
~~~

如果缺少 train_data 或 out_dir，Python 会直接报错。

### 3.2 必填字段必须放在默认字段之前

错误写法：

~~~python
@dataclass
class BadConfig:
    batch_size: int = 32
    train_data: str
~~~

正确写法：

~~~python
@dataclass
class GoodConfig:
    train_data: str
    batch_size: int = 32
~~~

### 3.3 可选字段

checkpoint 路径可能不存在：

~~~python
resume_from: str | None = None
~~~

它可以是：

~~~python
resume_from="runs/test/ckpt_last.pt"
~~~

也可以是：

~~~python
resume_from=None
~~~

### 3.4 类型标注不会自动检查运行时类型

下面虽然声明了 int：

~~~python
batch_size: int = 32
~~~

普通 Python 仍可能允许：

~~~python
ExampleConfig(batch_size="wrong")
~~~

类型标注主要用于：

- 帮助阅读；
- IDE 自动补全；
- 静态类型检查。

关键数值范围仍然要由 __post_init__ 主动检查。

## 4. 使用 __post_init__ 检查配置

数据类完成字段初始化后，会自动调用 __post_init__：

~~~python
@dataclass
class ExampleConfig:
    batch_size: int = 32
    d_model: int = 512
    num_heads: int = 16

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size 必须为正")
        if self.num_heads <= 0:
            raise ValueError("num_heads 必须为正")
        if self.d_model % self.num_heads != 0:
            raise ValueError("d_model 必须能被 num_heads 整除")
~~~

下面的配置会在训练开始前报错：

~~~python
ExampleConfig(batch_size=0)
~~~

__post_init__ 也可以补全派生字段。

例如 T_c 默认等于 total_steps：

~~~python
@dataclass
class ExampleConfig:
    total_steps: int = 5000
    T_c: int | None = None

    def __post_init__(self) -> None:
        if self.T_c is None:
            self.T_c = self.total_steps
~~~

创建：

~~~python
config = ExampleConfig(total_steps=8000)
~~~

结果：

~~~python
config.T_c == 8000
~~~

## 5. 使用 asdict() 转成字典

训练配置通常需要保存到 config.json。

~~~python
from dataclasses import asdict, dataclass


@dataclass
class ExampleConfig:
    batch_size: int = 32
    betas: tuple[float, float] = (0.9, 0.95)


config = ExampleConfig()
config_dict = asdict(config)
~~~

结果：

~~~python
{
    "batch_size": 32,
    "betas": (0.9, 0.95),
}
~~~

保存：

~~~python
with open("config.json", "w") as f:
    json.dump(asdict(config), f, indent=2)
~~~

JSON 会把 tuple 写成数组，这是正常行为：

~~~json
{
  "batch_size": 32,
  "betas": [0.9, 0.95]
}
~~~

## 6. 可变默认值与 default_factory

list、dict、set 等可变对象不应直接作为默认值。

错误写法：

~~~python
@dataclass
class BadConfig:
    tags: list[str] = []
~~~

正确写法：

~~~python
from dataclasses import dataclass, field


@dataclass
class GoodConfig:
    tags: list[str] = field(default_factory=list)
~~~

你的 betas 是不可变 tuple：

~~~python
betas: tuple[float, float] = (0.9, 0.95)
~~~

所以不需要 default_factory。

## 7. 使用 replace() 创建配置变体

多个实验只修改少量字段时，可以使用 dataclasses.replace：

~~~python
from dataclasses import replace


base_config = TrainConfig(
    train_data="data/train.bin",
    val_data="data/valid.bin",
    out_dir="runs/base",
)

small_lr_config = replace(
    base_config,
    out_dir="runs/lr_1e-4",
    alpha_max=1e-4,
)

large_batch_config = replace(
    base_config,
    out_dir="runs/batch_64",
    batch_size=64,
)
~~~

replace() 会创建新对象，不会修改 base_config。

新对象还会重新执行 __post_init__。

---

# 第二部分：当前项目的完整 TrainConfig

## 8. 需要导入什么

在 train.py 顶部加入：

~~~python
from dataclasses import asdict, dataclass, replace
from typing import Literal
~~~

其中：

- dataclass 用来声明配置类；
- asdict 用来保存配置；
- replace 用来创建实验变体；
- Literal 用来表达允许的设备名称。

如果暂时不用 replace，可以不导入它：

~~~python
from dataclasses import asdict, dataclass
~~~

## 9. 完整配置类

把下面的 TrainConfig 放在 import 之后、辅助函数之前：

~~~python
@dataclass
class TrainConfig:
    # 必填路径
    train_data: str
    val_data: str
    out_dir: str

    # 数据
    data_dtype: str = "uint16"

    # 模型
    vocab_size: int = 10_000
    context_length: int = 256
    d_model: int = 512
    d_ff: int = 1344
    num_layers: int = 4
    num_heads: int = 16
    rope_theta: float = 10_000.0

    # AdamW 与学习率调度
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    alpha_max: float = 3e-4
    alpha_min: float = 3e-5
    T_w: int = 0
    T_c: int | None = None
    total_steps: int = 5000
    weight_decay: float = 0.1
    grad_clip: float = 1.0

    # 训练过程
    batch_size: int = 32
    device: Literal["auto", "cpu", "cuda", "mps"] = "cpu"
    seed: int = 0
    log_interval: int = 20
    eval_interval: int = 200
    eval_batches: int = 10
    checkpoint_interval: int = 500
    milestone_interval: int = 0
    resume_from: str | None = None
    overfit: bool = False

    def __post_init__(self) -> None:
        self._validate_model()
        self._resolve_and_validate_schedule()
        self._validate_optimizer()
        self._validate_training()

    def _validate_model(self) -> None:
        if self.vocab_size <= 0:
            raise ValueError("vocab_size 必须为正")
        if self.context_length <= 0:
            raise ValueError("context_length 必须为正")
        if self.d_model <= 0 or self.d_ff <= 0:
            raise ValueError("d_model 和 d_ff 必须为正")
        if self.num_layers <= 0 or self.num_heads <= 0:
            raise ValueError("num_layers 和 num_heads 必须为正")
        if self.d_model % self.num_heads != 0:
            raise ValueError(
                f"d_model={self.d_model} 不能被 num_heads={self.num_heads} 整除"
            )

    def _resolve_and_validate_schedule(self) -> None:
        if self.total_steps <= 0:
            raise ValueError("total_steps 必须为正")

        if self.T_c is None:
            self.T_c = self.total_steps

        assert self.T_c is not None

        if self.T_w < 0:
            raise ValueError("T_w 不能为负数")
        if self.T_w >= self.T_c:
            raise ValueError(f"T_w={self.T_w} 必须小于 T_c={self.T_c}")
        if self.alpha_min < 0 or self.alpha_max < 0:
            raise ValueError("学习率不能为负数")
        if self.alpha_min > self.alpha_max:
            raise ValueError("alpha_min 不能大于 alpha_max")

    def _validate_optimizer(self) -> None:
        if len(self.betas) != 2:
            raise ValueError("betas 必须包含 beta_1 和 beta_2")

        beta_1, beta_2 = self.betas

        if not 0.0 <= beta_1 < 1.0:
            raise ValueError(f"beta_1={beta_1} 必须位于 [0, 1)")
        if not 0.0 <= beta_2 < 1.0:
            raise ValueError(f"beta_2={beta_2} 必须位于 [0, 1)")
        if self.eps < 0:
            raise ValueError("eps 不能为负数")
        if self.weight_decay < 0:
            raise ValueError("weight_decay 不能为负数")

    def _validate_training(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size 必须为正")
        if self.eval_batches <= 0:
            raise ValueError("eval_batches 必须为正")
        if self.grad_clip <= 0:
            raise ValueError("grad_clip 必须为正")

        valid_devices = {"auto", "cpu", "cuda", "mps"}
        if self.device not in valid_devices:
            raise ValueError(f"device={self.device!r} 不在 {valid_devices} 中")
~~~

### 为什么把检查拆成多个方法

也可以把所有检查都写在 __post_init__ 中，但会变得很长。

拆开后职责更清晰：

| 方法 | 负责内容 |
|---|---|
| _validate_model | 模型结构 |
| _resolve_and_validate_schedule | 学习率调度 |
| _validate_optimizer | AdamW |
| _validate_training | 训练过程 |

下划线表示这些方法主要供 TrainConfig 内部使用。

### 为什么没有 lr 字段

当前训练实际使用 alpha_max 作为最大学习率，并由余弦调度每一步更新学习率。

有效字段是：

~~~text
alpha_max
alpha_min
T_w
T_c
~~~

单独的 lr 字段没有参与训练，所以不应继续保留。

---

# 第三部分：修改 main()

## 10. main() 接收 TrainConfig

将训练入口定义为：

~~~python
def main(config: TrainConfig) -> int:
~~~

函数内部统一使用 config.xxx。

设置随机种子：

~~~python
torch.manual_seed(config.seed)
np.random.seed(config.seed)
~~~

设备：

~~~python
device = resolve_device(config.device)
~~~

数据：

~~~python
train_data = load_tokens(
    config.train_data,
    config.data_dtype,
    config.context_length,
)

val_data = load_tokens(
    config.val_data,
    config.data_dtype,
    config.context_length,
)
~~~

模型：

~~~python
body = transformer_lm(
    config.vocab_size,
    config.context_length,
    config.num_layers,
    config.d_model,
    config.num_heads,
    config.d_ff,
    config.rope_theta,
    device,
)
~~~

AdamW：

~~~python
optimizer = AdamW(
    body.parameters(),
    config.alpha_max,
    config.betas,
    config.eps,
    config.weight_decay,
)
~~~

因为 betas 在 TrainConfig 中已经是 tuple，所以不需要再次写：

~~~python
tuple(config.betas)
~~~

学习率：

~~~python
lr = cosine_learning_rate(
    step,
    config.alpha_max,
    config.alpha_min,
    config.T_w,
    config.T_c,
)
~~~

## 11. 配置验证不再放进 main()

配置检查都由 TrainConfig.__post_init__ 完成。

main() 应假设接收到的是合法配置，专注执行训练。

这样可以避免两套检查规则不一致：

~~~text
创建 TrainConfig
       ↓
自动检查配置
       ↓
配置合法后进入 main
~~~

## 12. 使用 asdict() 保存配置

不要把数据类对象和配置字典都命名为 config。

推荐：

~~~python
os.makedirs(config.out_dir, exist_ok=True)

config_dict = asdict(config)

with open(os.path.join(config.out_dir, "config.json"), "w") as f:
    json.dump(
        config_dict,
        f,
        indent=2,
        ensure_ascii=False,
        default=str,
    )

metrics_path = os.path.join(config.out_dir, "metrics.jsonl")

print(
    "[config] "
    + json.dumps(config_dict, ensure_ascii=False, default=str)
)
~~~

这里：

- config 是 TrainConfig 对象；
- config_dict 是普通字典；
- config.json 保存最终实际生效的配置。

由于 T_c 已在 __post_init__ 中补全，保存结果中的 T_c 是最终整数。

---

# 第四部分：创建和使用配置

## 13. 使用 make_config()

推荐把当前实验参数集中放在一个函数中：

~~~python
def make_config() -> TrainConfig:
    return TrainConfig(
        # 数据与输出
        train_data="data/train.bin",
        val_data="data/valid.bin",
        out_dir="runs/baseline",

        # 模型
        vocab_size=10_000,
        context_length=256,
        d_model=512,
        d_ff=1344,
        num_layers=4,
        num_heads=16,
        rope_theta=10_000.0,

        # AdamW 与调度
        betas=(0.9, 0.95),
        eps=1e-8,
        alpha_max=3e-4,
        alpha_min=3e-5,
        T_w=100,
        T_c=5000,
        total_steps=5000,
        weight_decay=0.1,
        grad_clip=1.0,

        # 训练过程
        batch_size=32,
        device="cpu",
        seed=0,
        log_interval=20,
        eval_interval=200,
        eval_batches=10,
        checkpoint_interval=500,
        milestone_interval=0,
        resume_from=None,
        overfit=False,
    )
~~~

这里把所有字段都写出来，适合第一次核对配置。

熟悉默认值后，可以只写必填字段和需要覆盖的字段：

~~~python
def make_config() -> TrainConfig:
    return TrainConfig(
        train_data="data/train.bin",
        val_data="data/valid.bin",
        out_dir="runs/baseline",
        device="cpu",
        T_w=100,
    )
~~~

没有写出的字段使用 TrainConfig 中的默认值。

## 14. 程序入口

train.py 底部写成：

~~~python
if __name__ == "__main__":
    config = make_config()
    raise SystemExit(main(config))
~~~

也可以不用 make_config：

~~~python
if __name__ == "__main__":
    config = TrainConfig(
        train_data="data/train.bin",
        val_data="data/valid.bin",
        out_dir="runs/baseline",
    )
    raise SystemExit(main(config))
~~~

使用 make_config 的好处是底部入口更短，配置位置更容易找到。

## 15. 完整骨架

省略辅助函数和训练循环后，train.py 的整体结构如下：

~~~python
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np
import torch


@dataclass
class TrainConfig:
    train_data: str
    val_data: str
    out_dir: str
    batch_size: int = 32
    total_steps: int = 5000
    device: Literal["auto", "cpu", "cuda", "mps"] = "cpu"

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size 必须为正")
        if self.total_steps <= 0:
            raise ValueError("total_steps 必须为正")


def make_config() -> TrainConfig:
    return TrainConfig(
        train_data="data/train.bin",
        val_data="data/valid.bin",
        out_dir="runs/baseline",
    )


def main(config: TrainConfig) -> int:
    os.makedirs(config.out_dir, exist_ok=True)

    config_dict = asdict(config)
    with open(os.path.join(config.out_dir, "config.json"), "w") as f:
        json.dump(config_dict, f, indent=2)

    # 准备数据、模型和优化器
    # 执行训练循环
    # 保存 checkpoint

    return 0


if __name__ == "__main__":
    config = make_config()
    raise SystemExit(main(config))
~~~

---

# 第五部分：训练实验

## 16. 创建调试配置

调试时可以使用很小的配置：

~~~python
def make_debug_config() -> TrainConfig:
    return TrainConfig(
        train_data="data/train.bin",
        val_data="data/valid.bin",
        out_dir="runs/debug",
        batch_size=2,
        total_steps=10,
        eval_interval=5,
        checkpoint_interval=5,
        device="cpu",
        overfit=True,
    )
~~~

入口改为：

~~~python
if __name__ == "__main__":
    config = make_debug_config()
    raise SystemExit(main(config))
~~~

## 17. 创建恢复训练配置

~~~python
def make_resume_config() -> TrainConfig:
    return TrainConfig(
        train_data="data/train.bin",
        val_data="data/valid.bin",
        out_dir="runs/baseline_resumed",
        resume_from="runs/baseline/ckpt_last.pt",
        total_steps=10_000,
    )
~~~

恢复时，模型结构字段必须与原 checkpoint 一致：

- vocab_size
- context_length
- d_model
- d_ff
- num_layers
- num_heads
- rope_theta

建议查看原实验保存的 config.json 进行核对。

## 18. 创建学习率实验

~~~python
def run_learning_rate_experiments() -> None:
    base_config = TrainConfig(
        train_data="data/train.bin",
        val_data="data/valid.bin",
        out_dir="runs/base",
    )

    for alpha_max in (1e-4, 3e-4, 1e-3):
        experiment = replace(
            base_config,
            alpha_max=alpha_max,
            out_dir=f"runs/lr_{alpha_max}",
        )
        main(experiment)
~~~

每个实验必须使用不同 out_dir，否则日志和 checkpoint 可能相互覆盖。

## 19. 从其他 Python 文件使用

其他脚本可以直接导入：

~~~python
from train import TrainConfig, main


config = TrainConfig(
    train_data="data/train.bin",
    val_data="data/valid.bin",
    out_dir="runs/external",
    batch_size=4,
    total_steps=20,
)

main(config)
~~~

这种方式适合：

- notebook；
- 单元测试；
- 调试脚本；
- 实验管理程序。

---

# 第六部分：测试和排错

## 20. 检查语法

~~~bash
uv run python -m py_compile cs336_basics/train.py
~~~

## 21. 单独测试 TrainConfig

合法配置：

~~~python
config = TrainConfig(
    train_data="train.bin",
    val_data="valid.bin",
    out_dir="runs/test",
    batch_size=16,
)

assert config.batch_size == 16
assert config.betas == (0.9, 0.95)
assert config.T_c == config.total_steps
~~~

非法配置：

~~~python
try:
    TrainConfig(
        train_data="train.bin",
        val_data="valid.bin",
        out_dir="runs/test",
        batch_size=0,
    )
except ValueError as error:
    print(error)
~~~

应该在训练开始前得到清晰错误。

## 22. 做一次短训练

~~~python
config = TrainConfig(
    train_data="data/train.bin",
    val_data="data/valid.bin",
    out_dir="runs/smoke_test",
    batch_size=2,
    total_steps=5,
    eval_interval=5,
    checkpoint_interval=5,
    device="cpu",
)
~~~

检查：

- 模型能够正常构建；
- config.json 包含完整字段；
- T_c 已补全为最终整数；
- AdamW 收到 tuple 类型的 betas；
- checkpoint 能正常保存和恢复。

## 23. 常见错误

### 错误一：函数参数名和函数体不一致

如果入口是：

~~~python
def main(config: TrainConfig) -> int:
~~~

函数体也必须使用：

~~~python
config.batch_size
~~~

### 错误二：把配置对象覆盖成字典

不要写：

~~~python
config = asdict(config)
~~~

这会让后续 config.batch_size 失效。

应该写：

~~~python
config_dict = asdict(config)
~~~

### 错误三：以为类型标注会自动检查

普通 dataclass 不会自动执行严格类型验证，关键检查必须保留在 __post_init__。

### 错误四：把必填字段放在默认字段后

所有无默认值字段必须写在有默认值字段之前。

### 错误五：使用可变默认值

list、dict、set 使用：

~~~python
field(default_factory=...)
~~~

### 错误六：多组实验共用 out_dir

不同实验应使用不同输出目录，否则日志、配置和 checkpoint 会互相覆盖。

### 错误七：过早使用 frozen=True

~~~python
@dataclass(frozen=True)
~~~

会禁止普通字段修改，但当前 __post_init__ 需要把 T_c 补全为 total_steps。

新手阶段先不要使用 frozen=True。

### 错误八：重复维护验证规则

配置检查应集中在 TrainConfig 中，main() 只负责训练流程。

---

## 24. 是否拆分多个数据类

当前项目建议使用一个扁平 TrainConfig。

项目扩大后可以拆成：

~~~python
@dataclass
class DataConfig:
    train_data: str
    val_data: str
    data_dtype: str = "uint16"


@dataclass
class ModelConfig:
    vocab_size: int = 10_000
    context_length: int = 256
    d_model: int = 512
    d_ff: int = 1344
    num_layers: int = 4
    num_heads: int = 16


@dataclass
class OptimizerConfig:
    alpha_max: float = 3e-4
    alpha_min: float = 3e-5
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    weight_decay: float = 0.1


@dataclass
class TrainConfig:
    data: DataConfig
    model: ModelConfig
    optimizer: OptimizerConfig
~~~

访问方式会变为：

~~~python
config.model.d_model
config.optimizer.alpha_max
~~~

当前 CS336 作业使用一个扁平 TrainConfig 更简单。

## 25. 最终检查清单

- [ ] 导入 dataclass、asdict 和 Literal
- [ ] 定义完整 TrainConfig
- [ ] 把配置验证放进 __post_init__
- [ ] 不保留无效的 lr 字段
- [ ] 定义 make_config()
- [ ] main() 接收 TrainConfig
- [ ] main() 内统一使用 config.xxx
- [ ] 使用 asdict(config) 保存 config.json
- [ ] 配置字典使用 config_dict 名称
- [ ] 在程序入口调用 make_config()
- [ ] 使用不同 out_dir 区分实验
- [ ] 用 py_compile 检查语法
- [ ] 单独测试合法和非法配置
- [ ] 做一次短训练
- [ ] 检查 checkpoint 保存与恢复

最终结构：

~~~text
TrainConfig
    ↓
make_config()
    ↓
main(config)
    ↓
训练完成
~~~
