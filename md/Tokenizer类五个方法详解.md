# Tokenizer 类五个方法详解

> 依据:`cs336_assignment1_basics.pdf` 2.6 节。本文件只讲每个方法**做什么、接受什么、输出什么**,不含实现细节。

## 1. `__init__(self, vocab, merges, special_tokens=None)`

**做什么**:构造函数,创建 `Tokenizer` 对象时被自动调用。把传进来的词表、合并规则、特殊 token 存到对象自己身上(变成实例属性),供后面 `encode`/`decode` 反复使用——类能把状态记住,避免每次调用都重新传一遍这些大对象。

**接受什么**:
- `vocab: dict[int, bytes]`——训练阶段产出的词表
- `merges: list[tuple[bytes, bytes]]`——训练阶段产出的、按创建顺序排列的合并规则
- `special_tokens: list[str] | None = None`——用户想让这个 tokenizer 认识的特殊 token,可选

**输出什么**:没有返回值。副作用是产生一个可以调用 `encode`/`decode` 的对象。

**注意**:如果 `special_tokens` 里有的 token **不在** `vocab` 里,应该把它**追加**进词表(分配新 ID),不能假设调用者已经把特殊 token 放进 `vocab` 了。

---

## 2. `from_files(cls, vocab_filepath, merges_filepath, special_tokens=None)`(类方法)

**做什么**:`__init__` 的"另一条路"——不是直接传内存里现成的 `vocab`/`merges`,而是从**磁盘上序列化好的文件**里读出这两样东西,再构造出一个 `Tokenizer` 对象。对应 `train_bpe` 训练完之后"序列化到磁盘"这一步产出的文件,这里是反过来把它们读回来用。

**接受什么**:
- `vocab_filepath: str`——序列化后的词表文件路径
- `merges_filepath: str`——序列化后的合并规则文件路径
- `special_tokens: list[str] | None = None`——同上

**输出什么**:一个 `Tokenizer` 实例(和 `__init__` 直接构造出来的对象,行为完全一样)。

**注意**:文件格式要和自己 `train_bpe` 保存时用的格式对应上——序列化格式(比如怎么存 `bytes`,JSON 还是别的)需要自己设计并保持前后一致。

---

## 3. `encode(self, text: str) -> list[int]`

**做什么**:把一整段文本编码成 token ID 序列。流程和训练时"同构":先预分词切出 pre-token,再对每个 pre-token 按 `self.merges` 的**创建顺序**依次应用合并规则,最后每个合并到底的 token 查 `self.vocab` 的反向映射(bytes → ID),拼成结果列表。要正确处理 `self.special_tokens`(整体保留、不被拆开,且要处理"更长的特殊 token 优先匹配"这种重叠情况)。

**接受什么**:`text: str`——一段完整的、已经在内存里的字符串。

**输出什么**:`list[int]`——一次性返回的、完整的 ID 列表。

**注意**:讲义和测试都明确允许这个方法**不用**考虑内存效率,可以整段读进内存处理。

---

## 4. `encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]`

**做什么**:`encode` 的"流式版本"。用于处理**放不进内存的大文件**——不是一次性拿到全部文本,而是从一个可迭代的字符串来源(讲义举例:一个打开的文件句柄,逐行产出字符串)里,**边读边编码、边产出 ID**,整个过程只占用常数级内存,不随输入总长度增长。

**接受什么**:`iterable: Iterable[str]`——任何"可以被遍历、每次拿到一个字符串"的东西,最典型的就是一个打开的文件对象。

**输出什么**:`Iterator[int]`——一个**惰性**的 ID 生成器,不是提前算好的 `list`。调用方可以用 `for id in tokenizer.encode_iterable(f):` 逐个取用,不需要等全部算完。

**注意**:跨越"块"边界处理文本时,不能让一个 token 被硬生生切成两半,否则结果会和"一次性整体编码"不一致。

---

## 5. `decode(self, ids: list[int]) -> str`

**做什么**:反向操作——把一串 token ID 还原成文本。查 `self.vocab` 把每个 ID 转成对应的 `bytes`,全部拼接,再整体解码成 `str`。

**接受什么**:`ids: list[int]`——一串整数 ID(不保证是 `encode` 产出的合法序列,用户可能传入任意整数)。

**输出什么**:`str`——还原出的文本。

**注意**:输入 ID 不保证对应合法的 UTF-8 字节序列,解码时要用 `errors='replace'`,把非法字节替换成 U+FFFD,而不是直接抛异常崩溃。

---

## 五者关系小结

- `__init__` / `from_files`:数据准备,只在构造时执行一次
- `encode`:核心、工作量最大,一次性处理
- `encode_iterable`:在 `encode` 基础上多了"流式惰性、常数内存"的要求
- `decode`:相对简单,`encode` 的逆操作
