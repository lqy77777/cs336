# WSL + Linux 环境笔记

记录时间：2026-07-24

## 1. 当前环境快照

| 项目 | 值 |
|---|---|
| WSL 版本 | WSL2 |
| 发行版 | Ubuntu 26.04 LTS (Resolute Raccoon) |
| 内核 | 6.18.33.2-microsoft-standard-WSL2 |
| 默认用户 | lambert7 |
| systemd | 已启用 (`/etc/wsl.conf` 中 `[boot] systemd=true`) |
| 网络模式 | mirrored（`.wslconfig` 中设置，Windows 与 WSL 共享同一网段/localhost 直通）|
| GPU | NVIDIA RTX 3060 Ti，通过 `/usr/lib/wsl/lib/nvidia-smi` 直通可用 |
| CPU | 24 核 |
| 内存 | WSL 侧约 7.6Gi（未在 `.wslconfig` 里手动设置 `memory=`，是 WSL2 默认按物理内存比例分配的结果）|
| 项目路径 | `/home/lambert7/cs336`（在 Linux 文件系统里，不在 `/mnt/c`）|
| VSCode 连接方式 | Remote-WSL（`~/.vscode-server` 已存在，说明是用 VSCode 的 WSL 扩展连进来的）|

## 2. 每次开机如何回到当前这个页面

你现在这个"页面"是：**Windows 上的 VSCode，通过 Remote-WSL 扩展连接到 WSL2 里的 Ubuntu，并打开了 `/home/lambert7/cs336` 这个文件夹**，其中带有 Claude Code 插件在跑。

复原步骤（按推荐程度排序）：

1. **最简单：直接打开 VSCode，让它自动重连**
   - Windows 开机后启动 VSCode（不需要手动先开 WSL，VSCode 会自动拉起 WSL2 虚拟机）。
   - 如果左下角显示过 "WSL: Ubuntu" 的最近窗口，从「File > Open Recent」里选回 `\\wsl.localhost\Ubuntu\home\lambert7\cs336` 即可，会自动重连。

2. **从 WSL 侧手动打开**
   - 打开 Windows Terminal 或任意终端，输入 `wsl` 进入 Ubuntu。
   - `cd ~/cs336`
   - `code .` （这个命令能用，是因为 `~/.vscode-server` 里已经装好了 remote-cli）
   - 这会自动在 Windows 上拉起 VSCode 窗口并连接到当前目录。

3. **从 Windows Terminal 直接定位**
   - Windows Terminal 里新建一个 "Ubuntu" 标签页，默认就会落在 `/home/lambert7`，`cd cs336` 后按上面方式 `code .` 即可。

> 小提示：VSCode 里如果装了「Remote - WSL」扩展并且最近打开过这个文件夹，通常**不需要手动敲命令**，直接打开 VSCode 选最近项目最快。

## 3. 额外值得知道的 WSL/Linux 知识

- **文件系统位置很重要**：项目在 `/home/lambert7/cs336`（Linux ext4 文件系统），而不是 `/mnt/c/...`。这是对的做法——把代码放在 Linux 原生文件系统里，I/O 性能比放在 `/mnt/c`（Windows NTFS 通过 9P 协议访问）快一个数量级以上。不要把大项目搬到 `/mnt/c` 下开发。

- **GPU 直通**：`nvidia-smi` 能看到 RTX 3060 Ti，说明 WSL2 的 GPU 直通（CUDA on WSL）配置正常，PyTorch/CUDA 训练任务应该能直接用 GPU，不需要额外配置。

- **内存分配**：WSL2 默认会占用大约一半物理内存作为上限，当前看到约 7.6Gi。如果以后跑 CS336 的训练任务遇到 OOM 而 Windows 侧内存还有富余，可以在 **Windows 用户目录**（`C:\Users\<你的用户名>\.wslconfig`）里加：
  ```
  [wsl2]
  networkingMode=mirrored
  memory=16GB
  processors=24
  ```
  改完后需要在 Windows PowerShell 里执行 `wsl --shutdown` 再重新打开 WSL 才会生效（这会关闭所有 WSL 发行版，注意保存工作）。

- **networkingMode=mirrored**：当前已经开启了镜像网络模式，这意味着 WSL 和 Windows 共享同一个网络视角，`localhost` 可以直接互通（比如 Windows 上跑的服务，WSL 里 `curl localhost:PORT` 能直接访问，反之亦然），不需要再查 WSL 的虚拟网卡 IP。

- **systemd 已开启**：意味着可以正常用 `systemctl` 管理服务（比如以后要跑 Docker daemon、数据库等），不需要用 WSL1 时代那些绕过 systemd 的 hack。

- **重启 WSL 的方式**：
  - 只重启某个发行版：`wsl --terminate Ubuntu`（在 Windows 侧执行）
  - 完全关闭所有 WSL 虚拟机（改 `.wslconfig` 后必须做）：`wsl --shutdown`
  - 这两个命令都要在 Windows 的 PowerShell/CMD 里执行，不是在 Linux 里。

- **Git 凭证**：当前 `git config` 里 `credential.https://github.com.helper` 用的是 `gh auth git-credential`，也就是通过 GitHub CLI (`gh`) 管理凭证。如果以后 push/pull 突然要求重新认证，跑 `gh auth login` 或 `gh auth refresh` 而不是去折腾 SSH key。

- **`code` 命令为什么能用**：它不是系统自带的，是 VSCode 的 Remote-WSL 扩展在第一次连接时，把一份 `remote-cli` 装进了 `~/.vscode-server/bin/<版本号>/bin/remote-cli/code` 并加进了 PATH。如果哪天升级了 VSCode 版本导致这个命令失效，重新用 VSCode 打开一次这个文件夹通常就会自动修好。

- **磁盘空间**：Linux 侧（`/`）在一个约 1TB 的虚拟磁盘上，当前只用了 2.8G，空间很宽裕；Windows 的 C 盘只剩 48G 左右，如果以后要下载大的数据集/模型 checkpoint，优先放到 Linux 侧 `/home/lambert7` 下而不是 `/mnt/c`，既快又不占 C 盘空间。
