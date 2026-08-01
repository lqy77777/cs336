# torch.cuda 与 torch.backends.mps 用法整理

背景：`train.py` 里的 `resolve_device` 函数用到了这两个模块来做设备自动选择，借此机会整理一下它们的常用 API。

## resolve_device 在做什么

```python
def resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda 但这台机器没有可用的 CUDA")
    return torch.device(name)
```

- `--device auto`：按优先级探测 CUDA > MPS > CPU，返回第一个可用的。
- `--device cuda` 但机器没有 CUDA：**提前失败**（fail fast），在这个函数里就直接报错，报错信息明确。
- `--device mps` 但机器没有 MPS：**没有对应检查**，这里不会报错，`torch.device("mps")` 会正常构造出来；错误会推迟到后面第一次 `.to("mps")` 真正搬数据时才由 PyTorch 抛出（fail late）。这是代码里一个不对称但影响不大的小遗漏。

## torch.cuda —— 管理 NVIDIA GPU

**检测与信息查询**
```python
torch.cuda.is_available()        # 有没有可用的 CUDA GPU
torch.cuda.device_count()        # 有几块 GPU
torch.cuda.get_device_name(0)    # 0 号 GPU 型号名
torch.cuda.current_device()      # 当前默认用第几块
```

**放数据/模型到 GPU**
```python
x = x.to("cuda")      # 或 "cuda:0" 指定具体卡
x = x.cuda()           # 等价写法，老代码常见
model = model.to("cuda")
```

**显存管理**（调 OOM 常用）
```python
torch.cuda.memory_allocated()       # 张量实际占用显存（字节）
torch.cuda.memory_reserved()        # 缓存分配器预留的显存，通常 >= allocated
torch.cuda.max_memory_allocated()   # 峰值占用
torch.cuda.empty_cache()            # 把缓存但未使用的显存还给系统
torch.cuda.reset_peak_memory_stats()
```

**同步与计时**（GPU 是异步执行的）
```python
torch.cuda.synchronize()   # 阻塞等待已提交的 kernel 全部跑完
```
> 精确计时某一步耗时时，必须在 `time.time()` 前后都调用 `synchronize()`，否则测的只是"提交 kernel"的时间，不是"kernel 跑完"的时间——新手最容易踩的坑。

**混合精度训练**
```python
torch.autocast("cuda")           # 自动混合精度上下文（新写法）
torch.cuda.amp.GradScaler()      # fp16 训练防止梯度下溢
```

## torch.backends.mps —— 管理 Apple Silicon GPU

API 比 `torch.cuda` 简单很多，因为苹果生态通常只有一块 GPU，没有多卡概念。

```python
torch.backends.mps.is_available()   # 硬件+系统版本是否满足可用条件
torch.backends.mps.is_built()       # 当前这份 PyTorch 编译时有没有包含 MPS 支持
```

- `is_built()` 问的是"包本身支不支持"，`is_available()` 还会再检查"当前硬件/系统是否满足"。理论上可能 `is_built()==True` 但 `is_available()==False`（比如系统版本太老）。

**放数据/模型到 GPU**
```python
x = x.to("mps")
model = model.to("mps")
```
没有 `x.mps()` 简写，也没有多卡切换接口。

**运行时操作在 `torch.mps` 里，不在 `torch.backends.mps`**
```python
torch.mps.synchronize()
torch.mps.empty_cache()
torch.mps.current_allocated_memory()
```

**MPS 相比 CUDA 的明显阉割**
- 显存统计 API 更新、更不成熟。
- 部分算子在 MPS 上没实现，训练时会直接报 `NotImplementedError`。

## 为什么检测函数在 `torch.backends.mps`，而不是 `torch.mps`

这是 PyTorch API 设计上的历史遗留分裂，不是刻意为之：

- `torch.backends.<name>`（如 `torch.backends.cudnn`、`torch.backends.mkldnn`、`torch.backends.mps`）是一套**后端能力探测/全局开关**的命名规范，回答"这个后端能不能用 / 要不要开某特性"，返回值多是 bool 或用来设置全局 flag（如 `torch.backends.cudnn.benchmark = True`）。
- `torch.cuda`、`torch.mps` 这类顶层命名空间放的是**运行时操作**——真正管理设备状态（显存、流、同步）。
- `torch.cuda` 是最早、最成熟的后端，历史上把探测函数（`is_available`）和运行时操作（`memory_allocated` 等）**都塞进了同一个命名空间**，没有严格区分，所以 `torch.cuda.is_available()` 和运行时 API 混在一起。
- MPS 是后加入的后端（2022 年左右），这时候 PyTorch 团队已经倾向用 `torch.backends.<name>` 规范来放探测类 API，所以 `is_available()`/`is_built()` 被放进了 `torch.backends.mps`；但独立的 `torch.mps` 命名空间也存在，只是加得更晚、功能更少。

一句话总结：CUDA 的检测函数重复暴露在两处（历史包袱），MPS 则从一开始就严格遵循了"探测归 backends，操作归顶层"的划分。

## 遗留的思考题

1. 如果在没有 MPS 的机器（如 Linux）上传 `--device mps`，`resolve_device` 不会报错，但后续第一次 `.to("mps")` 时 PyTorch 会抛出类似 `RuntimeError: PyTorch is not built with MPS enabled` 的错误——为什么这种"延迟报错"通常没有太大问题（相对于在入口处提前检查）？
2. 如果之后要在训练循环里做显存监控，为什么不能把 CUDA 和 MPS 一视同仁地都调用 `torch.cuda.memory_allocated()`，而需要保留 `device.type` 字符串来分支处理？
3. `torch.backends.cuda.is_built()` 和 `torch.cuda.is_available()` 语义上分别问的是什么？自己在本机验证一下两者的返回值差异。
