# 如何在 Mac 电脑上配置 Linux 环境

> 操作指南,所有命令需要你自己在终端里执行。

## 0. 先说清楚一个前提:Mac 默认已经比 Windows 更接近 Linux

macOS 底层是 **Darwin**,一个 BSD 家族的**类 Unix 系统**,和 Linux 一样遵循 **POSIX 标准**。这也是为什么在这次 CS336 作业开发过程中,同一份代码:

- `import resource`(POSIX 专属的内存限制模块)在 Mac 上能正常导入,不会像 Windows 那样报 `ModuleNotFoundError`
- 文件默认用 **UTF-8** 编码读写,不会像中文版 Windows(默认 GBK)那样把 UTF-8 编码的 vocab/merges 文件读成乱码,进而触发 `KeyError`

所以严格来说,**大多数情况下 Mac 不需要额外"配置 Linux 环境"就能正常跑这门课的代码**。这份文档主要讲两件事:①为什么 Mac 已经"够用"了;②如果你确实需要**完全等价的 Linux 环境**(不是"类 Unix",而是"就是 Linux"),有哪些方式,分别适合什么场景。

---

## 1. macOS(BSD)和 Linux 的差异到底在哪

虽然都是类 Unix 系统,但 macOS 的用户态工具链来自 **BSD**,Linux 发行版(Ubuntu 等)的用户态工具链来自 **GNU**,两者在一些命令行工具的参数、行为上**不完全一致**,比如:

- `sed`、`grep`、`tar`、`date` 等命令的部分参数在 BSD 版本和 GNU 版本里写法不同(比如 `sed -i` 在 BSD 上要求跟一个备份后缀参数,GNU 上可以省略)
- 默认的文件系统大小写敏感性不同(macOS 默认大小写不敏感,Linux 默认大小写敏感)
- 某些非常底层的系统调用行为、性能特征(比如多进程/多线程调度细节)可能存在细微差异

对 CS336 这门课来说,这些差异**通常不会影响到 Python 代码本身的正确性**(Python 标准库在 POSIX 系统上的行为是一致的),但如果你之后需要:

- 跟课程提供的、跑在 Linux 服务器上的参考环境做**逐字节对齐的复现**
- 用到某些只在 Linux 上打包发布、没有 macOS 原生版本的工具/驱动(常见于 CUDA 相关工具链——Mac 上除极少数情况外本来就没有 NVIDIA GPU,如果作业后续涉及 CUDA/Triton,Mac 用户大概率都需要一个真正的 Linux 环境远程或本地跑)
- 想要一个和同学、助教完全一致的"标准化"开发环境,减少"在我电脑上是好的"这类环境差异排查成本

这些情况下,才值得专门装一个真正的 Linux 环境。

---

## 2. 方式一:Docker(推荐,最常见的用法)

如果你的需求是"跑一个隔离的、和 Linux 服务器一致的运行环境",而不是要一个完整的桌面系统,**Docker** 是最轻量、最常用的方案——本质是用容器技术跑 Linux 用户态,而不是完整虚拟机,启动快、资源占用小。

```bash
brew install --cask docker
```

安装后打开 Docker Desktop,之后就可以用容器跑一个 Linux 环境,比如临时进入一个 Ubuntu 容器:

```bash
docker run -it --rm ubuntu:22.04 bash
```

**Apple Silicon(M 系列芯片)注意事项**:你的 Mac 是 **M4**,属于 **ARM64 架构**,和大多数云服务器/课程参考环境用的 **x86_64(AMD64)架构不同**。拉取镜像时:
- 默认会拉取 ARM64 版本的镜像(如果该镜像提供了 ARM64 构建),运行速度快,原生执行
- 如果某个镜像只有 x86_64 版本,Docker 会通过模拟(QEMU)运行,**明显更慢**,且个别底层库可能出现兼容性问题

需要强制指定架构时可以加 `--platform`:
```bash
docker run --platform linux/amd64 -it --rm ubuntu:22.04 bash
```

---

## 3. 方式二:Colima(轻量级,Docker 的开源替代品)

如果不想用 Docker Desktop(比如不想要它的图形界面、或者在意 Docker Desktop 的授权限制),`Colima` 是一个更轻量的选择,底层原理类似(用一个小型 Linux 虚拟机跑容器),命令行体验和 Docker 基本兼容:

```bash
brew install colima docker
colima start
docker run -it --rm ubuntu:22.04 bash
```

---

## 4. 方式三:完整的 Linux 虚拟机(需要图形界面/完整系统时)

如果你需要的不只是"跑几个命令的容器",而是一个**完整的、独立的 Linux 桌面/服务器系统**(比如需要装很多系统级依赖、长期开发、或者想要更接近真实 Linux 服务器的体验),可以用虚拟机软件:

- **UTM**(免费,Apple Silicon 原生支持,基于 QEMU):
  ```bash
  brew install --cask utm
  ```
  装好后手动下载 Ubuntu 的 **ARM64** 安装镜像(`.iso`),在 UTM 里创建新虚拟机安装。

- **Parallels Desktop** / **VMware Fusion**:商业软件(Fusion 现在个人使用免费),图形化程度更高,对 Apple Silicon 的优化和易用性通常优于 UTM,适合不想折腾配置细节的情况。

**同样要注意架构**:一定要下载 Ubuntu 官方的 **ARM64(aarch64)版本**镜像,不要下载 x86_64 版本,否则要么装不上,要么要靠模拟运行,速度会明显变慢。

---

## 5. 方式四:Multipass(专门用来快速拉起 Ubuntu 虚拟机)

如果只是想要一个纯命令行的 Ubuntu 环境,不需要图形界面,`Multipass`(Canonical 官方出品)比通用虚拟机软件更省心,一条命令就能拉起一个 Ubuntu 虚拟机:

```bash
brew install --cask multipass
multipass launch --name cs336-linux
multipass shell cs336-linux
```

---

## 6. 该选哪种方式

| 需求 | 推荐方式 |
|------|----------|
| 只是想验证代码在"纯 Linux 环境"里能不能跑通 | Docker 或 Colima(容器,最轻量、启动最快) |
| 需要长期开发、装很多系统依赖 | Multipass 或完整虚拟机(UTM/Parallels) |
| 需要图形界面 | UTM / Parallels Desktop / VMware Fusion |
| 只是想解决"某个 Python 模块在我电脑上导入报错"这类问题 | **通常不需要**,先确认是不是编码/依赖版本问题,Mac 本身已经是类 Unix 系统 |
| 涉及 CUDA/GPU 相关的作业内容 | Mac 本机(不管是否原生还是虚拟机)都没有 NVIDIA GPU,需要连接远程 Linux 服务器/云 GPU,而不是本地虚拟化方案 |

---

## 7. 常见问题

- **M 系列芯片(ARM)上跑 x86_64 的 Linux 镜像会怎样**:能跑(通过模拟),但速度会明显下降,遇到用到特殊 CPU 指令集的库时还可能出现兼容性问题。优先找 ARM64/aarch64 版本的镜像。
- **Mac 上要不要也像 Windows 那样纠结 `resource` 模块、编码问题**:不需要,这两个具体问题在 macOS 上天然不存在,因为 macOS 本身就是 POSIX 兼容、默认 UTF-8 编码的系统。
- **`brew` 命令不存在**:说明还没装 Homebrew(Mac 上最常用的包管理器),先执行:
  ```bash
  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
  ```
- **CUDA/GPU 相关的作业内容怎么办**:无论是原生 macOS 还是本地虚拟化的 Linux,Apple Silicon 的 Mac 都没有 NVIDIA GPU,涉及 CUDA/Triton 的部分需要连接学校提供的 GPU 集群或云端 GPU 实例,不是靠本地配置 Linux 环境能解决的。

---

## 常用命令速查

| 命令 | 作用 |
|------|------|
| `brew install --cask docker` | 安装 Docker Desktop |
| `docker run -it --rm ubuntu:22.04 bash` | 临时跑一个 Ubuntu 容器并进入 shell |
| `docker run --platform linux/amd64 ...` | 强制以 x86_64 架构运行(会走模拟,较慢) |
| `brew install colima docker` | 安装 Colima(轻量 Docker 替代) |
| `colima start` | 启动 Colima 的 Linux 虚拟机 |
| `brew install --cask utm` | 安装 UTM(通用虚拟机软件) |
| `brew install --cask multipass` | 安装 Multipass(快速 Ubuntu 虚拟机工具) |
| `multipass launch --name <名字>` | 创建一个新的 Ubuntu 虚拟机 |
| `multipass shell <名字>` | 进入虚拟机的 shell |
