# 如何在 Windows 电脑上配置 Linux 环境(WSL)

> 操作指南,所有命令需要你自己在终端(PowerShell / WSL 终端)里执行。

## 0. 为什么需要这么做

在 CS336 的作业开发过程中,Windows 环境下踩到过两类典型的"平台差异"坑,根源都是**Windows 不是类 Unix 系统**:

1. **某些 Python 标准库模块在 Windows 上不存在**
   比如 `resource` 模块(`tests/test_tokenizer.py` 里用来做内存限制),这是 POSIX 专属模块,Windows 没有对应的底层系统调用,`import resource` 会直接报 `ModuleNotFoundError`。

2. **默认文件编码不一致导致乱码/`KeyError`**
   Mac/Linux 系统默认用 UTF-8 读写文件;中文版 Windows 的默认编码通常是 GBK/CP936。像 `test_tokenizer.py` 里 `open(vocab_path)` 这种没有显式指定 `encoding="utf-8"` 的写法,在 Windows 上会用错误的编码去解析 UTF-8 文件,读出乱码字符(比如 `'臓'`),进而导致 `KeyError`。

**根本原因**:Mac 和 Linux 都是"类 Unix"系统,共享 POSIX 标准和相近的系统调用;Windows 走的是完全不同的技术路线(Win32 API)。这类差异以后大概率还会在别的地方反复出现。

**解决思路**:与其每次单独排查、逐个打补丁,不如直接在 Windows 里跑一个**真正的 Linux 环境**,从源头上避免这一整类问题——这就是 WSL(Windows Subsystem for Linux)。

---

## 1. WSL 是什么

WSL 不是"沙盒",更准确的说法是:**在 Windows 里内置了一台轻量级 Linux 虚拟机**。

- **WSL 1**:不是虚拟机,是一个"翻译层"——把 Linux 系统调用实时转换成 Windows 内核能理解的调用,没有真正的 Linux 内核,兼容性和性能都不如 WSL 2。
- **WSL 2**(现在的默认版本):通过 Hyper-V 虚拟化技术,运行了一个**真正的 Linux 内核**。它的文件系统是一个存放在物理磁盘上的**虚拟磁盘文件(`.vhdx`)**,里面是完整的 ext4 文件系统。

装好之后,`resource` 模块、UTF-8 默认编码这些"Unix 环境的隐含假设",在 WSL 里全部自动满足,不需要再对课程代码做任何平台兼容性修改。

---

## 2. 安装 WSL

在 PowerShell(管理员模式)里执行:

```powershell
wsl --install
```

默认会安装最新的 WSL 2 和 Ubuntu 发行版。安装完成后重启电脑,首次启动 Ubuntu 会要求你设置一个 Linux 用户名和密码(和 Windows 账号无关,是 WSL 内部单独的账号)。

确认安装成功、且版本是 WSL 2:

```powershell
wsl --list --verbose
```

---

## 3. (可选)把 WSL 从 C 盘迁移到 D 盘

WSL 默认把虚拟磁盘装在 **C 盘**(`C:\Users\<你>\AppData\Local\Packages\...\ext4.vhdx`)。如果 C 盘空间紧张,而作业数据集(TinyStories + OpenWebText,共约 14GB)又要下载进 WSL,建议先搬到 D 盘。

**官方支持的方法:导出 → 注销 → 导入**

```powershell
# 1. 确认发行版名字
wsl --list --verbose

# 2. 导出成 tar 文件(会打包当前所有数据)
wsl --export Ubuntu D:\WSL\ubuntu-backup.tar

# 3. 注销当前这个(确保第2步导出成功后再执行)
wsl --unregister Ubuntu

# 4. 导入到 D 盘新位置(会新建一个 .vhdx 文件)
wsl --import Ubuntu D:\WSL\Ubuntu D:\WSL\ubuntu-backup.tar
```

完成后可以删掉备份用的 `ubuntu-backup.tar`。

**注意**:`wsl --import` 方式创建的发行版,默认登录用户会变成 `root`,需要手动指定回你自己的用户:

在 WSL 里编辑(或创建)`/etc/wsl.conf`,加入:
```ini
[user]
default=你的用户名
```
保存后在 PowerShell 里执行 `wsl --shutdown` 重启 WSL 生效。

> 之后你在 WSL 里 `clone` 仓库、下载数据集,只要都在 WSL 自己的文件系统内(比如 `~/cs336`),就都会落在 D 盘,不会占用 C 盘空间。

---

## 4. 数据会不会用完就消失

不会。WSL 2 的整个 Linux 文件系统存在一个 `.vhdx` 虚拟磁盘文件里,这个文件本身持久保存在物理硬盘上,关机、重启电脑、重启 WSL 都不影响里面的文件。

**会导致数据丢失的情况**:
- 手动执行 `wsl --unregister <发行版名>`(会连虚拟磁盘一起删除,无法恢复)
- 极端情况下断电/强制关机导致 `.vhdx` 文件损坏(概率低,但存在)
- 物理磁盘本身空间不足,写入失败

**建议**:代码部分保持养成 `git commit` + `push` 到 GitHub 的习惯,双重保险;那 14GB 的数据集本身不需要也不该传到 GitHub(体积太大),正常使用 WSL 不用担心它自己消失。

---

## 5. 把仓库和数据放进 WSL

**不要**通过 `/mnt/d/...` 直接在 WSL 里访问 Windows 盘符上已有的文件来长期使用——WSL2 访问 Windows 文件系统(DrvFs/9p 协议)有明显的 I/O 性能损耗,尤其是大文件的顺序读取(比如 BPE 训练要整个扫一遍 12GB 的语料)。

推荐做法:**直接在 WSL 里重新 clone 仓库、重新下载数据**,让所有文件从一开始就落在 WSL 原生的 ext4 文件系统里。

```bash
# 进入 WSL 终端
cd ~
git clone https://github.com/你的用户名/仓库名.git
cd 仓库名/data   # 或对应的数据目录
```

国内直接访问 huggingface.co 可能很慢,可以用镜像站 `hf-mirror.com`(把 URL 里的 `huggingface.co` 换成 `hf-mirror.com`):

```bash
curl -L -O https://hf-mirror.com/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
curl -L -O https://hf-mirror.com/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt
curl -L -O https://hf-mirror.com/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
curl -L -O https://hf-mirror.com/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
gunzip owt_train.txt.gz
gunzip owt_valid.txt.gz
```

如果文件比较大、下载慢,可以换成多线程下载工具 `aria2c`(比单线程 curl/wget 明显快):

```bash
sudo apt install aria2
aria2c -x 16 -s 16 -k 1M <URL>
```

---

## 6. 重新配置 git 身份和 GitHub 认证

**WSL 和 Windows 是两套独立的环境**,Windows 上用 `gh auth login` 保存的登录状态**不会自动同步**到 WSL,需要在 WSL 里单独走一遍:

```bash
# 装 git(部分发行版自带,没有的话装一下)
sudo apt install git

# 配置身份(commit 记录里显示谁提交的,和 GitHub 认证无关)
git config --global user.name "你的名字"
git config --global user.email "你的邮箱"

# 安装 GitHub CLI 并登录(用来做实际的身份认证)
sudo apt install gh
gh auth login
```

> 补充说明两者的区别:`gh auth login` 解决的是"有没有权限访问/推送 GitHub 仓库"(身份认证);`user.name`/`user.email` 只是写入每次 commit 元数据里的作者信息,是纯本地配置,Git 不会从 `gh` 登录状态里自动读取,两者互不替代。

配置完之后,在 WSL 里 push/pull 到 GitHub 和在 Windows 上操作完全一样,不会因为环境切换而受影响。

---

## 7. 用 VS Code 连接 WSL(Remote - WSL 扩展)

装好 VS Code 的 **Remote - WSL** 扩展后,在 WSL 终端里进入项目目录,输入:

```bash
code .
```

VS Code 窗口会打开,左下角出现绿色的 `WSL: Ubuntu` 标签,代表:

- 界面仍然显示在 Windows 上(体验和平时一样)
- 但**文件读写、终端 shell、调试运行、扩展执行**全部发生在 WSL 里面
- 打开的终端是 WSL 的 bash,不是 PowerShell
- Python 解释器会选到你在 WSL 里装的那个环境(比如 WSL 里的 conda 环境),`resource` 模块、编码问题自然就不存在了

简单说:**它是一座桥——UI 留在 Windows,"干活"的部分交给 WSL 里的 Linux 环境**。

---

## 8. 常见问题

- **`import resource` 报错**:说明还在用 Windows 原生 Python 环境跑代码,没有真正切换到 WSL 内的解释器,检查 VS Code 左下角是不是显示 `WSL: Ubuntu`,以及终端选的是不是 WSL 的 bash。
- **测试报编码相关的 `KeyError`**:同上,或者是仓库代码里 `open()` 没有显式指定 `encoding="utf-8"`,即使在 WSL 里也建议养成显式写编码参数的习惯,不依赖系统默认值。
- **访问 Windows 盘符里的文件很慢**:`/mnt/c`、`/mnt/d` 这种跨文件系统访问本身就有性能损耗,长期使用的项目/数据建议直接放进 WSL 自己的文件系统(如 `~/`)。
- **同一份代码,Windows 上跑得比 Mac 慢**:先排除是不是这类平台兼容性问题导致的(比如误用了低效的兼容写法);如果两边都在各自原生环境下正常运行,速度差异更可能是硬件差异(单核 IPC、有没有杀毒软件实时扫描等),不一定是代码问题。

---

## 常用命令速查

| 命令 | 作用 |
|------|------|
| `wsl --install` | 安装 WSL 和默认发行版 |
| `wsl --list --verbose` | 查看已安装的发行版及版本 |
| `wsl --export <名字> <路径>.tar` | 导出发行版为 tar 备份 |
| `wsl --unregister <名字>` | 注销发行版(**会删除所有数据**) |
| `wsl --import <名字> <目标路径> <tar路径>` | 从备份导入到新位置 |
| `wsl --shutdown` | 关闭所有 WSL 实例,使配置改动生效 |
| `code .` | 在当前目录下用 VS Code 打开(需要装 Remote-WSL 扩展) |
