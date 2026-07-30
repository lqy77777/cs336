# argparse 使用手册（新手向）

标准库 `argparse`，Python 3.x 通用，无需额外安装。本文是纯粹的语法参考——`argparse` 是通用的命令行工具，怎么解析参数和你写什么训练逻辑是两件事，所以这里放心给完整代码。真正到 `train.py` 里填超参、写训练循环，那部分还是你自己的活。

---

## 0. 先搞清楚一件事：`main` 和命令行参数没有关系

C 语言里 `int main(int argc, char **argv)` 是语言规定的固定签名。**Python 没有这个规定**——`main` 只是一个普通函数，名字随便起，接不接参数、接几个，全由你自己决定。

真正连接"命令行"和"你的代码"的，是解释器启动时塞进 `sys.argv` 的那份原始字符串列表：

```python
import sys
print(sys.argv)
```

```bash
$ python foo.py --lr 0.001 --steps 100
['foo.py', '--lr', '0.001', '--steps', '100']
```

注意：**全都是字符串**，`'0.001'` 不是 `float`，`'100'` 不是 `int`。你完全可以手动切片解析 `sys.argv[1:]`，但很快会写出一堆重复代码：判断某个 flag 在不在、取它后面那个值、转类型、类型不对时报错……`argparse` 就是把这套重复劳动封装好的标准库。

---

## 1. 最小可运行的例子

```python
import argparse

parser = argparse.ArgumentParser(description="一个示例程序")
parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
args = parser.parse_args()

print(args.lr, type(args.lr))
```

```bash
$ python foo.py --lr 0.01
0.01 <class 'float'>

$ python foo.py
0.001 <class 'float'>          # 没传就用 default
```

三步走：

1. `ArgumentParser()` 造一个解析器对象
2. `add_argument(...)` 声明每一个参数长什么样、类型是什么、默认值是什么
3. `parse_args()` 真正去读 `sys.argv`，返回一个 **`Namespace` 对象**——`args.lr` 这种属性访问方式，就是从这里来的

`parse_args()` 还自带一个你没写但白送的功能：

```bash
$ python foo.py --help
usage: foo.py [-h] [--lr LR]

一个示例程序

options:
  -h, --help  show this help message and exit
  --lr LR     learning rate
```

这也是为什么 `argparse` 比手切 `sys.argv` 值得——`--help` 不用你多写一行代码。

---

## 2. 位置参数 vs 可选参数

**名字前面有没有 `--`，决定了它是"位置参数"还是"可选参数"**，这是 argparse 里最容易搞混的一点。

```python
parser.add_argument("data_path")       # 位置参数：必须给，靠位置识别，不带 --
parser.add_argument("--lr")            # 可选参数：靠名字识别，可以省略（如果有 default）
```

```bash
$ python foo.py ./data/train.txt --lr 0.01
```

```python
print(args.data_path)   # './data/train.txt'
print(args.lr)          # '0.01'
```

区别：

| | 位置参数 | 可选参数 |
|---|---|---|
| 写法 | `add_argument("name")` | `add_argument("--name")` |
| 命令行怎么写 | 直接写值，靠顺序对应 | `--name value` |
| 默认是否必填 | **是**（不给就报错） | 否（除非 `required=True`） |
| 适合什么场景 | 每次运行都必须给的东西，比如输入文件路径 | 有合理默认值、平时不用改的超参 |

**新手常见踩坑**：以为 `add_argument("lr")` 和 `add_argument("--lr")` 是同一回事，其实前者是位置参数，命令行要写成 `python foo.py 0.01`（不带 `--lr`），和后者的调用方式完全不同。训练脚本里的超参基本都应该用 `--` 前缀的可选参数——你不会想强制记住几十个超参的固定顺序。

---

## 3. 常用选项逐个过

### 3.1 `type`：字符串怎么转成你要的类型

```python
parser.add_argument("--lr", type=float)
parser.add_argument("--steps", type=int)
parser.add_argument("--name", type=str)      # str 是默认值，可以不写
```

`argparse` 会自动做转换，**转换失败会直接报错退出**，不会让一个非法字符串偷偷混进你的程序：

```bash
$ python foo.py --steps abc
usage: foo.py [-h] [--steps STEPS]
foo.py: error: argument --steps: invalid int value: 'abc'
```

这比手动 `int(sys.argv[i])` 强的地方在于：报错信息里直接告诉你是哪个参数、传了什么非法值。

### 3.2 `default`：不传时用什么

```python
parser.add_argument("--batch-size", type=int, default=32)
```

不写 `default` 的可选参数，不传的话值是 `None`——这个很容易忘，写完记得想一下"如果用户没传这个 flag，我的代码后面拿到 `None` 会不会炸"。

### 3.3 `required`：可选参数也能强制必填

```python
parser.add_argument("--data-path", type=str, required=True)
```

不传就直接报错，不会等到程序跑到一半才因为 `None` 崩溃：

```bash
$ python foo.py
foo.py: error: the following arguments are required: --data-path
```

### 3.4 `choices`：限定合法取值集合

```python
parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
```

传了集合之外的值会直接拒绝：

```bash
$ python foo.py --device tpu
foo.py: error: argument --device: invalid choice: 'tpu' (choose from 'cpu', 'cuda', 'mps')
```

适合那种"只有几个合法取值，输错了应该立刻报错，而不是跑到一半才发现"的参数，比如学习率调度类型、优化器种类。

### 3.5 `action="store_true"`：开关型布尔参数

```python
parser.add_argument("--verbose", action="store_true")
```

这类参数命令行上**不跟值**，出现就是 `True`，不出现就是 `False`（默认）：

```bash
$ python foo.py --verbose
# args.verbose == True

$ python foo.py
# args.verbose == False
```

配套的还有 `action="store_false"`——出现就是 `False`，默认 `True`，用得少一些，容易让人confuse，一般不推荐。

### 3.6 `BooleanOptionalAction`：想要 `--flag` / `--no-flag` 成对出现

Python 3.9+ 可用。比单纯的 `store_true` 更清楚地表达"这是一个可以显式关闭的开关"：

```python
parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True)
```

```bash
$ python foo.py --compile        # args.compile == True
$ python foo.py --no-compile     # args.compile == False
$ python foo.py                  # args.compile == True（用 default）
```

### 3.7 `nargs`：一个参数接收多个值

```python
parser.add_argument("--betas", type=float, nargs=2)       # 恰好 2 个
parser.add_argument("--layers", type=int, nargs="+")      # 1 个或更多
parser.add_argument("--tags", type=str, nargs="*")        # 0 个或更多
```

```bash
$ python foo.py --betas 0.9 0.95 --layers 2 4 8
```

```python
args.betas    # [0.9, 0.95]
args.layers   # [2, 4, 8]
```

### 3.8 短选项 + `dest`：自定义属性名

```python
parser.add_argument("--learning-rate", "-lr", dest="lr", type=float, default=1e-3)
```

- 短横线 `-lr` 只是命令行上少打几个字，两个都能用
- `dest="lr"` 决定了最终 `Namespace` 上的属性名叫什么。**不写 `dest` 的话，`argparse` 会自动把参数名的横线转下划线**：`--learning-rate` 不写 `dest` 时默认对应 `args.learning_rate`。

### 3.9 `help` 和 `%(default)s`

```python
parser.add_argument(
    "--steps", type=int, default=1000,
    help="训练步数（默认: %(default)s）",
)
```

`%(default)s` 是 argparse 的占位符语法，会自动填成真实的默认值，不用手动同步两处：

```bash
$ python foo.py --help
  --steps STEPS  训练步数（默认: 1000）
```

---

## 4. 完整例子：一个像样的训练脚本参数表

把上面的选项拼起来，大概长这样（这里只是展示 argparse 语法本身，超参具体怎么定见你的训练脚本设计）：

```python
import argparse

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a language model")

    # 数据相关：必填，没有合理默认值
    parser.add_argument("--train-data", type=str, required=True)
    parser.add_argument("--val-data", type=str, required=True)

    # 模型超参：有默认值，平时不用每次都传
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=16)

    # 训练超参
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--steps", type=int, default=5000)

    # 限定取值集合
    parser.add_argument("--lr-schedule", choices=["cosine", "constant"], default="cosine")
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")

    # 开关型
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=False)

    return parser

if __name__ == "__main__":
    args = build_parser().parse_args()
    print(vars(args))   # 把 Namespace 转成 dict，方便打印/存盘
```

```bash
$ python train.py --train-data data/train.bin --val-data data/val.bin \
    --lr 1e-3 --batch-size 64 --device cuda --compile

{'train_data': 'data/train.bin', 'val_data': 'data/val.bin', 'd_model': 512,
 'num_layers': 4, 'num_heads': 16, 'lr': 0.001, 'batch_size': 64,
 'steps': 5000, 'lr_schedule': 'cosine', 'device': 'cuda', 'compile': True}
```

`vars(args)` 这个技巧值得记住：它把 `Namespace` 转成普通 `dict`，方便你打日志、存进 JSON、或者用 `**vars(args)` 展开去构造别的对象。

---

## 5. `main(argv=None)`：为什么很多标准库/工具都这么写

如果 `parse_args()` 永远读全局的 `sys.argv`，那你就没法在单元测试里方便地喂给它一组假参数——只能真的去改 `sys.argv`，很脏。解法是让 `main` 接受一个**可选的**参数列表：

```python
def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)   # argv=None 时，parse_args 内部自动回退到 sys.argv[1:]
    print(vars(args))
    ...
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
```

好处是这样一来 `main` 既能从命令行跑，也能被别的 Python 代码直接调用，不用经过 shell：

```python
# 单元测试里：
from train import main
assert main(["--train-data", "x.bin", "--val-data", "y.bin", "--steps", "2"]) == 0
```

```python
# 或者写个 sweep 脚本，绕过命令行直接跑多组参数：
for lr in [1e-4, 3e-4, 1e-3]:
    main(["--train-data", "x.bin", "--val-data", "y.bin", "--lr", str(lr)])
```

`sys.exit(main())` 也顺带把返回值变成了进程的退出码——`main` 返回非 0，外层的 shell 脚本或 CI 就能感知到"这次跑失败了"。

---

## 6. 常见坑

**1. 横线转下划线，容易对不上。**
命令行上是 `--learning-rate`，但 `args.learning-rate` 是非法 Python 语法，argparse 会自动帮你转成 `args.learning_rate`。忘记这条转换规则，会在代码里手滑写出 `args.learning-rate` 这种语法错误，或者以为没有自动转换而多此一举写 `dest`。

**2. `type=bool` 是个陷阱，几乎总是不该用。**
```python
parser.add_argument("--flag", type=bool)   # 别这么写
```
Python 里 `bool("False")` 的结果是 `True`——因为任何非空字符串都是"真"。也就是说 `--flag False` 会被解析成 `True`，行为和直觉完全相反。想要布尔参数，用第 3.5/3.6 节的 `action="store_true"` 或 `BooleanOptionalAction`，不要用 `type=bool`。

**3. 没传的可选参数默认是 `None`，不是你以为的"某个合理的空值"。**
```python
parser.add_argument("--tags", nargs="*")
```
不传 `--tags` 时 `args.tags` 是 `None`，不是空列表 `[]`——想让它默认空列表，要显式写 `default=[]`。

**4. 位置参数和 `nargs="*"`/`"+"` 混用时，解析顺序会变得不直观。**
新手阶段尽量少让位置参数和可变长度的参数共存，容易解析出乎意料的结果。

**5. `required=True` 只对可选参数（带 `--`）有意义。**
位置参数本来就是必填的，给它加 `required=True` 是无效写法（部分版本甚至会报错）。

---

## 7. 速查表

| 需求 | 写法 |
|---|---|
| 必填的位置参数 | `add_argument("path")` |
| 必填的可选参数 | `add_argument("--path", required=True)` |
| 有默认值的可选参数 | `add_argument("--lr", type=float, default=1e-3)` |
| 限定取值集合 | `add_argument("--mode", choices=[...])` |
| 开关（默认关） | `add_argument("--flag", action="store_true")` |
| 开关（可显式关闭） | `add_argument("--flag", action=argparse.BooleanOptionalAction, default=True)` |
| 接收多个值 | `add_argument("--xs", type=int, nargs="+")` |
| 自定义属性名 | `add_argument("--learning-rate", dest="lr", type=float)` |
| help 里显示默认值 | `help="... (default: %(default)s)"` |
| 从字符串列表解析（测试用） | `parser.parse_args(["--lr", "0.01"])` |
| Namespace 转 dict | `vars(args)` |

---

## 8. 相关笔记

- [Section5.3_训练脚本实现指南.md](Section5.3_训练脚本实现指南.md) —— 训练脚本的整体结构（区块 1「解析参数」对应本文全部内容）
- [距离训练一个模型还缺什么.md](距离训练一个模型还缺什么.md) —— `train.py` 目前是空的，argparse 是补上「区块 1」的第一步
