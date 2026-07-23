# `bpe.py` 开发过程知识点总结

> 按写作顺序整理:`find_chunk_boundaries`(课程提供)→ `train_bpe`(签名 → 初始词表 → 预分词 → merge 循环)→ `Tokenizer`(`__init__` → `encode` → `encode_iterable` → `decode`)→ adapter 对接。每部分列出:实际踩过的坑、当前代码逐行要点、以及开发中真实比较过的"等价写法对比"。姊妹篇见 [transformer.py开发知识点总结.md](transformer.py开发知识点总结.md)。

---

## 〇、文件头部(import 与 PAT)

- `import regex as re`:**必须用第三方 `regex` 包,不能用标准库 `re`**——PAT 里的 `\p{L}`(Unicode 字母类)、`\p{N}`(数字类)标准库不支持;起名 `re` 是讲义示例的惯例,注意别和标准库搞混
- `from collections import Counter`:计数专用 dict 子类,`counter[key] += 1` 对不存在的 key 自动当 0 处理
- 第 8 行 `PAT`:GPT-2 预分词正则,**必须逐字符照抄讲义原文**。踩过两次坑:
  1. 三引号字符串里手动换行——换行符和缩进空格会成为正则内容的一部分,行为彻底改变
  2. 少了 `| `(竖线加空格)导致六个分支被焊成五个(`\p{N}+` 和 `[^\s\p{L}\p{N}]+` 之间)
- **验证咒语**:`re.findall(PAT, "some text that i'll pre-tokenize")` 必须输出 `['some', ' text', ' that', ' i', "'ll", ' pre', '-', 'tokenize']`——30 秒验证,挡住几小时调试
- PAT 六个分支恰好覆盖任意字符(字母/数字/空白/其余),所以 `finditer` 的所有匹配拼回去等于原文本、不丢字符——**这是 PAT 的设计性质,不是 `finditer` 的通用保证**(换一条覆盖不全的正则立刻会丢)

---

## 一、`find_chunk_boundaries`(课程提供,允许原样使用)

### chunk 边界:要点

- 第 22-24 行:`seek(0, SEEK_END)` → `tell()` → `seek(0)` 是"O(1) 获取文件大小"的标准惯用法,不读内容
- 第 31 行:最后一个边界强制设为 `file_size`——整除有余数,`n * chunk_size` 可能小于真实大小,不修正会丢尾部数据
- 第 35 行:跳过首尾边界(0 和 `file_size` 天然合法),只调整中间的
- 第 44、49 行:边界对齐到特殊 token 的**开头**——除第一块外,每个 chunk 都以 `<|endoftext|>` 开头;`found_at` 是迷你块内相对偏移,要加 `initial_position` 换算成绝对位置
- 第 54 行 `sorted(set(...))`:多个猜测点可能对齐到同一位置,去重后块数可能少于期望——这就是函数注释"May return fewer chunks"的原因
- **已知边界情况**:token 恰好横跨两个 4096 迷你块交界时会被漏检,边界滑到下一个 token——代价是负载均衡略差,**正确性不受影响**(边界仍在某个 token 开头)
- 设计哲学:成本只和"猜测点到最近 token 的距离"成正比,与文件总大小无关——几十 GB 语料也只在每个猜测点附近读几 KB

---

## 二、`train_bpe`:函数签名(第 56-61 行)

### 签名:踩过的坑

1. **返回注解写成 `-> vocab: dict[...], merges: list[...]:`**——Python 返回注解只能是**单个类型表达式**,不能给返回值起名、不能逗号并列;两个返回值的惯例是单个 `tuple[dict[int,bytes], list[tuple[bytes,bytes]]]`
2. **`special_tokens: list[int]`** → 应为 `list[str]`(特殊 token 是字符串)
3. **第一个参数收 `file: BinaryIO`** → 应收 `input_path: str | os.PathLike`。三层原因:文件对象**不能 pickle**、多进程没法共享(每个 worker 要拿路径自己 open);打开方式(`"rb"`)是实现内部需求,不该交给调用者;`with open` 生命周期自管
4. **联合类型写成 `str,os.PathLike`(逗号)**——参数列表里逗号是"分隔两个参数",联合类型的连接符是 `|`(Python 3.10+)
5. `**kwargs`:收集多余关键字参数成字典,容错垫片——测试多传参数不报错;自己的函数里可选

---

## 三、`train_bpe`:初始词表(第 62-66 行)

### 词表:踩过的坑

1. **`bytes(i)` vs `bytes([i])`**——`bytes(5)` 是"5 个零字节",`bytes([5])` 才是"数值为 5 的单字节"。用错时 ID 0 对应 `b''`、ID 1 对应 `b'\x00'`……全错。自查:`vocab[65] == b'A'`

### 词表:逐行要点

- 第 63 行:全部 256 个字节值都要进词表(不是"语料里出现过的"),这是 byte-level BPE 无 OOV 保证的来源
- 第 64-65 行:`enumerate` 解包(序号+元素),特殊 token 用 `.encode('utf-8')` 转成**一整块** bytes 作为单个词表条目——它从一开始就是完整单元,不拆、不参与 merge
- 第 66 行:`index` 记录下一个可用 ID,merge 循环每加一个新 token 就 +1
- ID 分配顺序(特殊 token 在前在后)测试不检查——`test_train_bpe` 只比较 key 集合和 value 集合

### 词表:写法对比

```python
for k, v in enumerate(special_tokens):        # 当前写法:手工偏移 256+k
    vocab[256 + k] = v.encode('utf-8')

for k, v in enumerate(special_tokens, start=256):   # start 参数消掉硬编码偏移
    vocab[k] = v.encode('utf-8')
```

`enumerate` 优于 `range(len(...))`+下标:少一次下标访问的出错机会、意图直读;`start` 参数还能消掉手工偏移(当前代码保留了 `256+k`,能跑,但 256 出现两处要同步)。

---

## 四、`train_bpe`:预分词(第 68-82 行)

### 预分词:踩过的坑

1. **累积器 `frequency` 一度放在 chunk 循环内**——每轮被重置,只剩最后一个 chunk 的数据;必须在循环外建、循环内累加
2. **拿 `Match` 对象当 Counter 的 key**——Match 按对象身份比较,同文本的两次匹配是两个不同对象,永远不会累加;要 `.group()` 取文本
3. **对 `str` 调 `.decode()`**——方向搞反:`str.encode()` 字符串→字节,`bytes.decode()` 字节→字符串;`chunk` 已经 decode 过,`match.group()` 是 str,要 **encode**
4. **`tuple(match.group().encode())` 拆出整数元组**——迭代 bytes 得到 int;要 `tuple(bytes([i]) for i in ...)` 逐个包回单字节 bytes。键必须是 `tuple[bytes, ...]`,否则后面 merge 的"相邻对拼接"(`bytes + bytes`)无法进行
5. **`escaped`/`'|'.join` 在循环内重复计算**——不变量挪出循环
6. **忘了先按特殊 token split**——`<|endoftext|>` 的字节会污染 merge 统计,`test_train_bpe_special_tokens`(检查词表条目不含 `b"<|"`)必挂

### 预分词:逐行要点

- 第 71 行 `'|'.join([re.escape(s) for s in special_tokens])`:**先对每个 token 单独 escape、再用 `|` 拼接**——顺序反了会把自己加的 `|` 分隔符也转义掉。`re.escape` 的必要性:`<|endoftext|>` 自带 `|`,不转义会被解析成"或"
- 第 74 行:`b"<|endoftext|>"` 目前**硬编码**——严格来说应从 `special_tokens` 参数推导(遗留项,现有测试恰好都传这个 token 所以没暴露)
- 第 77 行 `.decode("utf-8", errors="ignore")`:切块边界对齐到 ASCII 的 `<`,理论上不会切坏多字节字符,`ignore` 是防御性兜底——但它会**静默丢数据**,开发期可临时换 `strict` 让切块 bug 响亮暴露
- 第 78 行 `re.split(escaped, chunk)`:训练阶段特殊 token 被**剥离**(不保留)——它不参与统计,split 掉正是目的;段首尾的空字符串喂给 PAT 得到空结果,无害
- 第 79 行 `re.finditer`:讲义明确要求用它而不是 `findall`
- 第 81-82 行:`frequency` 的键是"单字节 bytes 的元组"(如 `(b't',b'h',b'e')`),值是出现次数——正是讲义 2.4 节的频次表形态

### 预分词:写法对比

```python
re.findall(PAT, para)        # 立即求值,一次性把所有匹配存成 list[str]——内存 O(全部匹配)
re.finditer(PAT, para)       # 惰性迭代器,逐个产出 Match——边扫边计数,扫过即丢
```

```python
frequency[key] += 1                      # Counter:不存在自动当 0,语义即文档
d[key] = d.get(key, 0) + 1               # dict.get:每处都要手写默认值 0
if key in d: d[key] += 1 else: ...       # 最啰嗦,四行
```

`Counter` 附赠 `most_common(n)`、计数器相加合并(多进程汇总正好用)——计数场景首选。

---

## 五、`train_bpe`:merge 循环(第 83-116 行)

### merge:踩过的坑

1. **`temp = frequency` 不是复制,是别名**——两个名字指向同一对象,边遍历边写同一字典 → `RuntimeError: dictionary changed size during iteration`;必须 `temp = Counter()` 新建对象
2. **`tuple(t[i], t[i+1])` 报 TypeError**——`tuple()` 只接受一个可迭代参数;二元组用字面量 `(t[i], t[i+1])`。`return tuple(vocab, merges)` 同错,直接 `(vocab, merges)`
3. **`max` 的 key 函数两版错误**:`lambda x: (x[1], x[0])`——迭代 dict 拿到的是 key(pair 本身),`x[1]` 是 pair 第二个字节不是频次;`(dict.get, x)`——类方法对象,对所有 x 恒等,频次彻底没参与比较。正确:`lambda x: (pairs[x], x)`,元组比较先比频次、并列退到 pair 字典序(正好实现讲义"取字典序较大"的平局规则)
4. **`while index <= vocab_size` off-by-one**——会多合并一轮,词表超限一条;`<` 才对
5. **`for i in range(...)` 内 `i += 1` 无效**——for 的循环变量由 range 决定,体内赋值不影响下一轮;"合并消耗两格、下一步跳两格"必须用 `while` + 手动步进
6. **元组最后一个元素被系统性丢弃**——两个变种:`if i == len(t)-1` 永远不触发(range 到不了);合并发生在倒数第二位时跳过了补尾检查。用 `(a,b,c,d)` 合并 `(b,c)` 手动追踪暴露:结果丢了 `d`。修法:补尾检查放在**两个分支的步进之后**执行
7. **`while i < range(len(t)-1)` 报 TypeError**——int 和 range 对象不能比大小,循环上界应是数字 `len(t)-1`

### merge:逐行要点

- 第 88-89、99-100 行 `if len(t) <= 1: continue`:单元素 pre-token 凑不成相邻对、也永远不会再变,跳过是安全的(它只是中间账本,不是最终答案);同时防住"空循环产出空元组、词条消失"的 bug
- 第 90-92 行:pair 统计只在**每个 pre-token 元组内部**遍历相邻位置——天然满足"merge 不跨 pre-token 边界"
- 第 93 行 `max(pairs, key=lambda x: (pairs[x], x))`:`max` 迭代 dict 拿到 key、返回的也是 key(pair 本身,不是频次);`bytes` 元组的比较就是字典序,平局规则零成本实现
- 第 94-96 行:一次合并同时更新三处——`vocab[index] = max_pair[0] + max_pair[1]`(**bytes 用 `+` 拼接**,合并后的整体);`merges.append(max_pair)`(**合并前的两半**,顺序即创建顺序,`encode` 要按此重放);`index += 1`
- 第 103-112 行(扫描替换):`while` 手动步进——匹配上时 `append(t[i]+t[i+1])` 且额外 `i += 1`(跳过被消耗的两格);第 110-111 行补尾检查在步进后执行,覆盖"最后一步是非合并"和"合并跳跃后恰剩一个落单元素"两种收尾
- `(x,x,x)` 合并 `(x,x)` 得 `(xx, x)`——从左到右贪心、已消耗位置不重叠,BPE 标准行为
- **性能注记**:每轮全量重算 `pairs` + 重建 `frequency` 是朴素实现——讲义指出可增量更新(只有与被合并 pair 重叠的计数会变)。实测:corpus.en/500 词表 1.18s(参考实现 0.38s、toy 3s 之间);5MB/1000 词表 6.2s,呈超线性——真跑 TinyStories(2.2GB/10000)前大概率要做并行预分词+增量更新

### merge:写法对比

```python
max(pairs, key=lambda x: (pairs[x], x))    # 当前:一步到位,平局规则内建
pairs.most_common(1)                        # Counter 自带,但只按频次排——平局时不保证字典序,不满足讲义
sorted / heapq.nlargest                     # 只要第一名时都比 max 重
```

---

## 六、`Tokenizer.__init__`(第 118-128 行)

### init:踩过的坑

1. **`sorted(special_tokens, key=len)` 默认升序**——重叠 token 场景(`<|endoftext|>` 和它的两连)要**长的优先匹配**,必须 `reverse=True`;正则 `|` 按写在前面的分支优先,不会自动选最长
2. **`sorted(None)` 崩溃**——`special_tokens` 可为 `None`,要条件式:`None if special_tokens is None else sorted(...)`

### init:逐行要点

- 第 121 行注解 `dict[int:bytes]`:**冒号是笔误**,这写出来是切片语法(`dict[slice(int,bytes)]`),应为 `dict[int, bytes]`(逗号)——注解不影响运行所以没炸,但语义是错的
- 第 128 行 `reversed_vocab = {v: k for k, v in vocab.items()}`:反转字典(值查键),`encode` 末尾"bytes → ID"全靠它——**在 `__init__` 建一次、所有 `encode` 调用复用**,这正是"Tokenizer 为什么是类"的核心理由(有状态 + 一次性预处理)。前提:vocab 值无重复,反转才无损
- **遗留项**:讲义要求"special token 不在 vocab 里时追加进词表",目前没实现(现有测试的辅助函数替学生做了这步,没暴露)
- `from_files`(第 130-136 行)仍是 `pass`——序列化格式要和自己训练代码的存盘格式配套,pickle(省事、二进制)或 json(可读、需自行处理 bytes)都可

---

## 七、`Tokenizer.encode`(第 137-180 行)

### encode:踩过的坑

1. **`re.split` 不带捕获组会把特殊 token 丢掉**——训练时"丢"是目的,编码时 token 必须保留并映射成 ID;给整个模式包一层**普通括号(捕获组)**,split 结果里分隔符本身也会出现(与文本片段交替)
2. **`tuple(para.encode('utf-8'))` 把特殊 token 拆成 13 个整数**——`tuple(bytes)` 是"拆开",`(x,)`(括号+逗号)才是"整体包成单元素元组"。拆开后:进合并循环被污染成 int 元组 → 查 `reversed_vocab`(key 全是 bytes)→ `KeyError`
3. **`item in self.special_tokens` 恒为 False**——左边是 bytes/tuple、右边是 list[str],类型不同永不相等;后来发现根本不需要这个判断:特殊 token 包成单元素元组后,`len(item) == 1` 一个条件同时保护"特殊 token"和"已完全合并的 pre-token"
4. **合并循环丢尾元素的 bug 在这里"复发"**——`train_bpe` 修过的同款问题,重写时漏了补尾逻辑,症状:`"Hello"` 被逐轮啃到只剩 `'H'`、emoji 只剩 1 字节解码成 �、该发生的合并因原料提前被啃掉而没发生。教训:**同一逻辑第二次手写时,第一次踩过的坑会原样重现**——要么复用代码,要么把当时的反例(`(a,b,c,d)` 合并 `(b,c)`)重新跑一遍
5. **`para` 可能就是特殊 token 本身,不能无差别跑 PAT**——否则 `<|endoftext|>` 被拆成 `<`、`|`、`endoftext`……违反"特殊 token 永不拆分"

### encode:逐行要点

- 第 140 行:`special_tokens is None` 单独走无 split 的分支——`for s in None` 会 TypeError
- 第 141 行 `'(' + '|'.join(...) + ')'`:捕获组保留分隔符;`self.special_tokens` 已在 `__init__` 按长度降序,最长 token 优先命中
- 第 143-144 行:`para` 命中特殊 token 时,`(para.encode('utf-8'),)` 包成**单元素元组**——形态与"完全合并的 pre-token"统一,后续零特判
- 第 153-155 行:**按 `self.merges` 的创建顺序逐条重放**——顺序错一位结果就不同(先学 `th` 还是 `he` 影响后续能否继续合并),这是 `merges` 必须有序的原因
- 第 158-159 行 `len(item) == 1: continue`:一石二鸟(特殊 token + 已合并完的词)
- 第 162-171 行:与 `train_bpe` 第 103-112 行**同一扫描替换算法的另一种等价写法**——`while j <= len(item)-2` 即 `j < len(item)-1`;补尾检查 `if j == len(item)-1: append(item[j])` 放在步进后,和 train_bpe 的 `if i == len(t)-2: append(t[i+1])` 检查点不同但覆盖等价。两处独立手写增加了维护成本(改一处忘另一处的风险)
- 第 174-179 行:查 `reversed_vocab` 把每个 bytes 换成 ID——单元素直接查 `item[0]`,多元素逐个查
- **性能注记**:外层遍历全部 merges(词表规模下近万条)、内层遍历所有 pre-token,即使某词早已合并到底——正确性无损,大语料编码时是优化点

### encode:写法对比

```python
re.split(escaped, text)              # 无捕获组:分隔符被丢弃(训练用)
re.split('(' + escaped + ')', text)  # 有捕获组:分隔符保留在结果里(编码用)
```

```python
tuple(b'ab')     # 拆开 → (97, 98) 整数元组
(b'ab',)         # 整体包 → (b'ab',) 单元素元组——一对括号一个逗号,语义天差地别
```

---

## 八、`Tokenizer.encode_iterable`(第 182-189 行)

### iterable:逐行要点

- 内部定义生成器函数 `out`,`return out(iterable)` 返回**生成器对象**——满足 `Iterator[int]` 返回类型;函数体内有 `yield` 就是生成器函数,调用不执行、首次 `next()` 才跑
- `yield from self.encode(string)`:把一行的整个 ID 列表逐个转交——等价于 `for id in ...: yield id` 的简写
- 内存特性:同一时刻只持有"当前一行 + 这一行的 ID 列表",不随文件总大小增长——`test_encode_iterable_memory_usage`(Linux 限 1MB)验证的正是这个
- 切块安全性:按**行**切,PAT 的空白分支不会产生跨越 `\n` 之后还回头合并的 pre-token——"逐行编码拼接 == 整体一次编码"由 `test_encode_iterable_tinystories_matches_tiktoken` 与 tiktoken 逐值比对确认
- 对照:`encode` 允许整段进内存(对应测试标了 xfail),`encode_iterable` 必须常数内存——两个方法设计目标不同

---

## 九、`Tokenizer.decode`(第 191-195 行)

### decode:踩过的坑

1. **逐 token 分别 `.decode()` 再拼字符串**——byte-level BPE 不保证每个 token 是完整 UTF-8 字符,多字节字符(如 3 字节汉字、4 字节 emoji)可能被切进两个相邻 token,半个字符单独解码直接 `UnicodeDecodeError`。正确顺序:**先把所有 bytes 拼成一整块,最后统一 decode 一次**——`b'\xe7'` 单独解码炸、`b'\xe7\x89\x9b'` 整体解码得 `'牛'`

### decode:逐行要点

- 第 192-194 行:逐 ID 查 `self.vocab` 收集 bytes 进列表
- 第 195 行 `b''.join(text).decode('utf-8', errors='replace')`:`b''.join` 是 bytes 版的 join(O(n) 一次分配);`errors='replace'` 把非法字节换成 U+FFFD(�)而不是崩——用户可传任意 ID 序列,讲义明确要求

### decode:写法对比

```python
text = ''
for token in ids:
    text += vocab[token].decode('utf-8')   # 错误×2:半字符崩 + 循环内 += 是 O(n²)

b''.join(pieces).decode('utf-8', errors='replace')   # 正确:先拼字节,一次解码,线性时间
```

循环内 `+=` 拼接不可变序列(str/bytes)每次整体重新分配拷贝,n 次累计 O(n²)——`join` 先算总长一次分配,O(n)。与 `list.extend`(原地,均摊线性)vs `list = list + other`(每次新建)是同一条原理。

---

## 十、adapter 对接与测试

- `from cs336_basics.bpe import train_bpe, Tokenizer`:包名来自 `pyproject.toml` 的 `name`,`uv` 把项目装成包,**导入认包名不认目录层级**——`tests/` 和 `cs336_basics/` 平级不构成障碍
- `raise NotImplementedError` 必须**整行替换**——`raise` 立即中断,它后面加的代码永远执行不到
- `bpe.py` 导入 `Counter`,`adapters.py` 不用重复导入——每个模块的名字在**它自己定义处**的命名空间解析,对调用方透明
- `run_train_bpe` 转发:按位置传参即可,参数名不必与自己函数一致;`get_tokenizer`:构造 `Tokenizer(vocab, merges, special_tokens)` 返回实例
- 测试结果:`test_train_bpe.py` 3/3 通过(速度 1.18s,介于参考 0.38s 与 toy 3s 之间);`test_tokenizer.py` 23 通过 + 2 跳过(macOS 无 rlimit)

---

## 十一、当前已知遗留项(备忘)

1. 第 74 行切块 token 硬编码 `b"<|endoftext|>"`,未从 `special_tokens` 推导
2. 第 72 行 `num_processes = 4` 只用于切块,实际预分词仍是**串行**——真正的 `multiprocessing` 并行未接入(macOS spawn 模式注意 `if __name__ == "__main__"` 保护)
3. merge 循环是朴素全量重算——TinyStories 全量训练前需评估增量更新优化
4. `from_files` 未实现(与训练侧的序列化格式配套设计)
5. `__init__` 未实现"special token 缺失时追加进 vocab"
6. 第 121 行类型注解 `dict[int:bytes]` 冒号应为逗号
7. `encode` 与 `train_bpe` 的扫描替换逻辑两处独立手写——已因此复发过一次丢尾 bug,可考虑抽成共用函数

---

## 十二、贯穿始终的通用方法论

### 方法论:str 与 bytes 的世界观

- `str.encode('utf-8')` → bytes;`bytes.decode('utf-8')` → str,方向不可混;`errors=` 三档:`strict`(默认,炸)/`replace`(U+FFFD)/`ignore`(静默丢)
- **迭代/索引 bytes 得到 int**(0-255),不是单字节 bytes;`bytes([i])` 包回去——本文件三处独立用到(初始词表、预分词、坑 7 的反面)
- `bytes + bytes` 是拼接(同 str/list 的 `+` 语义);bytes 元组比较即逐字节字典序——平局规则免费实现
- UTF-8 变长(1-4 字节):任意字节位置切一刀可能切在字符中间——切块要对齐 ASCII 分隔符、解码要先拼后 decode,两处坑同根

### 方法论:调试与验证

- **手动追踪小例子**是抓逻辑 bug 的主力:讲义 2.4 节例子验证 merge 全流程、`(a,b,c,d)` 合并 `(b,c)` 抓丢尾、`(x,x,x)` 验证贪心不重叠
- "形状/类型能跑"≠"结果对":Match 当 key、`(dict.get, x)`、`item in special_tokens` 恒 False——都不报错,结果悄悄全错;**算出来却从没被用过的变量**(`cosine`/`sine` 同款)是此类错误的信号
- 讲义原文和 adapter docstring 是最终规格——"merges 存合并前的两半且有序"、"用 finditer 别存列表"这类细节都写在原文里
- 同一 bug 会在重写同一逻辑时复发——修 bug 时把反例记下来,重写后重跑
