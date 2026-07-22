# 把本地代码上传到 GitHub、以及后续同步

> 操作指南,所有命令需要你自己在终端执行。

## 0. 当前情况说明

查过之后发现两点需要先处理:

1. **当前的 `origin` 指向的是斯坦福课程官方仓库**(`stanford-cs336/assignment1-basics`),不是你自己的仓库——这是当初 `git clone` 留下的。不能直接往这个地址推送。
2. **`data/` 目录(约 13GB,TinyStories + OpenWebText)还没有被 `.gitignore` 排除**。这个必须先处理,否则:
   - GitHub 对单个文件有 100MB 的限制,`data/` 里的大文件会直接推送失败
   - 就算能推送,也不该把训练数据这种大文件放进 git 仓库(git 不适合管理这类数据)

---

## 1. 准备:更新 `.gitignore`

打开项目根目录的 `.gitignore` 文件,在**开头**加上这几行:

```
# 下载的训练数据,不应纳入版本控制
data/

# macOS 系统文件
.DS_Store
```

保存后,用下面的命令确认 `data/` 确实被排除了(不应该出现在输出里):

```bash
git status
```

---

## 2. 把课程仓库改名为 `upstream`,腾出 `origin` 这个名字

这样做的好处:以后课程如果更新了作业代码(README 里提到过"if there are any updates, we will notify you so you can `git pull`"),你依然能从 `upstream` 拉取更新;而你自己的改动会推送到 `origin`(你自己的仓库),两者不冲突。

**这一步是必须的前提**,不管你接下来用哪种方式创建自己的仓库(命令行还是网页),都需要先把现有的 `origin` 腾出来:

```bash
git remote rename origin upstream

# 验证一下,应该能看到 upstream 指向课程仓库
git remote -v
```

---

## 3. 创建你自己的仓库并推送(两种方式选一种)

### 方式一:命令行(需要先安装 GitHub CLI)

如果还没装 `gh`:

```bash
brew install gh
gh auth login       # 引导你完成一次性的浏览器授权登录
```

装好并登录之后,一条命令搞定"创建仓库 + 设为 origin + 推送":

```bash
gh repo create 仓库名 --private --source=. --remote=origin --push
```

- `--private`:仓库设为私有(作业解答建议私有,避免被其他同学看到引发学术诚信问题;想公开就换成 `--public`)
- `--source=.`:用当前目录作为仓库内容来源
- `--remote=origin`:自动把新仓库设成名字叫 `origin` 的远程(因为上一步已经把原来的 `origin` 腾出来了,这里不会冲突)
- `--push`:创建完直接推送,不需要再手动执行 `git push`

做完这一步,直接跳到第 4 节,不需要再看下面"方式二"。

### 方式二:网页创建 + 手动关联

1. 登录 [github.com](https://github.com)
2. 右上角点 `+` → `New repository`
3. 填写仓库名(比如 `cs336-hw1`)
4. 选择 **Public** 或 **Private**(建议 **Private**,理由同上)
5. **不要**勾选"Initialize this repository with a README"之类的选项(本地已经有内容了,勾选会导致远程和本地历史冲突)
6. 点 `Create repository`

创建完成后,页面会显示一个仓库地址,形如:

```
https://github.com/你的用户名/仓库名.git
```

把它接上,并完成首次提交推送:

```bash
# 把你自己的仓库地址设成新的 origin
git remote add origin https://github.com/你的用户名/仓库名.git

# 验证一下,应该能看到 origin 和 upstream 两个,指向不同地址
git remote -v

# 查看有哪些改动/新文件(确认 data/ 不在列表里)
git status

# 把改动加入暂存区
git add .

# 提交,写一句描述性的信息
git commit -m "Add BPE tokenizer and Transformer implementation"

# 首次推送,-u 会把本地的 main 分支和 origin 的 main 分支关联起来
git push -u origin main
```

如果 GitHub 上新建的仓库默认分支名不是 `main`(有些账号设置默认是 `master`),把上面命令里的 `main` 换成实际分支名(可以先用 `git branch` 确认本地当前分支叫什么)。

---

## 4. 之后有新改动,如何再次"更新"上传

日常开发流程,重复这三步就够了(不需要再加 `-u`,因为已经关联过分支了):

```bash
git status                          # 看看改了哪些文件
git add .                           # 或者 git add 具体文件名,只提交部分改动
git commit -m "描述这次改了什么"
git push
```

---

## 5. 如何把远程的改动同步回本地

**情况一:你在 GitHub 网页上直接改过文件**(比如网页编辑器改了 README),本地要同步:

```bash
git pull origin main
```

**情况二:想拉取课程官方仓库(`upstream`)的更新**:

```bash
git fetch upstream                  # 先把 upstream 的更新拉下来,但不自动合并
git log upstream/main --oneline -5  # 看看 upstream 最近有什么新提交
git merge upstream/main             # 确认要合并的话,再执行这一步
```

这一步**可能会产生冲突**(如果你自己改过的文件,课程那边也改了同一处)。如果遇到冲突,git 会在冲突的文件里标出 `<<<<<<<`、`=======`、`>>>>>>>` 这样的标记,需要你手动决定保留哪部分,改完后:

```bash
git add 冲突的文件名
git commit
```

---

## 6. 常见问题

- **`push` 时提示某个文件太大 / 被拒绝**:说明有大文件没被 `.gitignore` 排除掉,检查是不是 `data/` 漏了,或者别的地方生成了大文件(比如序列化的 vocab/merges,如果很大也建议加进 `.gitignore`)
- **`push` 被拒绝,提示 remote 有本地没有的提交**(比如你在另一台设备上推送过):先 `git pull origin main`,解决完可能的冲突,再 `git push`
- **想确认某个文件到底会不会被上传**:`git check-ignore -v 文件路径`,如果有输出说明会被忽略,并且会告诉你是 `.gitignore` 里哪一行导致的
- **`gh repo create` 报 `origin` 已存在**:说明第 2 步(改名成 `upstream`)还没做,先执行 `git remote rename origin upstream` 再重试

---

## 常用命令速查

| 命令 | 作用 |
|------|------|
| `git status` | 查看当前改动状态 |
| `git remote -v` | 查看当前配置的远程仓库地址 |
| `git add .` | 把所有改动加入暂存区 |
| `git commit -m "..."` | 提交暂存区的改动 |
| `git push` | 推送到远程(默认 origin) |
| `git pull` | 拉取远程改动并合并到本地 |
| `git fetch upstream` | 只拉取 upstream 的更新,不自动合并 |
| `git log --oneline -10` | 查看最近 10 条提交记录 |
| `gh repo create` | 用命令行创建一个新的 GitHub 仓库 |
