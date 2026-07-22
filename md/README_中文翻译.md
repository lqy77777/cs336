# CS336 2025 春季 作业 1：基础（Basics）

关于本次作业的完整说明，请参阅作业讲义：
[cs336_assignment1_basics.pdf](./cs336_assignment1_basics.pdf)

如果你发现作业讲义或代码中存在任何问题，欢迎提交 GitHub issue，或直接开一个包含修复的 pull request。

## 环境搭建（Setup）

### 环境（Environment）
我们使用 `uv` 来管理环境，以保证可复现性、可移植性和易用性。
请在[这里](https://github.com/astral-sh/uv#installation)安装 `uv`（推荐），或者运行 `pip install uv` / `brew install uv`。
我们建议先阅读一下[这里](https://docs.astral.sh/uv/guides/projects/#managing-dependencies)关于使用 `uv` 管理项目的内容（你不会后悔的！）。

现在你可以用以下命令运行仓库中的任意代码：
```sh
uv run <python_file_path>
```
必要时环境会被自动解析并激活。

### 运行单元测试（Run unit tests）

```sh
uv run pytest
```

初始状态下，所有测试都应该以 `NotImplementedError` 失败。
要将你的实现接入测试，请补全 [./tests/adapters.py](./tests/adapters.py) 中的函数。

### 下载数据（Download data）
下载 TinyStories 数据以及 OpenWebText 的一个子样本

``` sh
mkdir -p data
cd data

wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
gunzip owt_train.txt.gz
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
gunzip owt_valid.txt.gz

cd ..
```
