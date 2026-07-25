import os
from typing import BinaryIO
import regex as re
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator

#gpt-2正则
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks
    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

def train_bpe(
        input_path: str | os.PathLike,
        vocab_size: int,
        special_tokens: list[str],
        **kwargs,
) -> tuple[dict[int,bytes], list[tuple[bytes, bytes]]]:
    #1.先创建初始词表(256+special tokens)
    vocab = {i : bytes([i]) for i in range(256)}    #bytes函数的用法
    for k,v in enumerate(special_tokens):
        vocab[256 + k] = v.encode('utf-8')
    index = 256 + len(special_tokens)   #记录下一个新词的索引

    #2.读取文件,确定分割点,然后pre-tokenization
    frequency = Counter()  #pre-tokenization后的frequecy_table
    #转义后的special_tokens
    escaped = '|'.join([re.escape(s) for s in special_tokens])
    num_processes = 4    #可改
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore") #跳过无法解码的字节
            for para in re.split(escaped, chunk):
                for matched in re.finditer(PAT, para):
                    #迭代一个 bytes 对象,取出来的每个元素是 int(0-255 之间的数值),不是 bytes
                    key = tuple(bytes([i]) for i in matched.group().encode('utf-8'))
                    frequency[key] += 1
    #3.merge
    merges = []
    
    pairs = Counter()   #记录相邻字节出现的次数
    pairs_to_frequency = defaultdict(set)
    for key, count in frequency.items():
        if len(key) <= 1:   #特殊情形：单个字节无需处理
            continue
        for i in range(len(key)-1):
            pair = (key[i],key[i+1])   #相邻字节对
            pairs_to_frequency[pair].add(key)
            pairs[pair] += count
    while index < vocab_size:
        #次数打平的话返回字典序高的字节对  #对字典使用max函数得到的是key而不是整个字典
        max_pair = max(pairs, key = lambda x: (pairs[x], x))
        new_token = max_pair[0] + max_pair[1]  #两个字节拼接在一起 
        vocab[index] = new_token 
        merges.append(max_pair)
        #接下来是更新frequency
        copys = pairs_to_frequency[max_pair].copy()
        for key in copys:
            count = frequency[key]
            length = len(key)
            temp = []
            i = 0
            while i < length-1:
                if (key[i],key[i+1]) != max_pair:
                    temp.append(key[i])
                else:
                    if i != 0:
                        pairs[(key[i-1],key[i])] -= count
                        pairs[(key[i-1],new_token)] += count
                    if i != length -2:
                        pairs[(key[i+1],key[i+2])] -= count
                        pairs[(new_token,key[i+2])] += count
                    temp.append(new_token)
                    i += 1
                if i == length - 2:
                    temp.append(key[i+1])
                i += 1
            new_key = tuple(temp)
            frequency[new_key] = count
            frequency.pop(key)
            if len(new_key) == 1:
                pass
            else:
                old_pairs = list(zip(key[:-1],key[1:]))
                new_pairs = list(zip(new_key[:-1],new_key[1:]))
                for old in old_pairs:
                    pairs_to_frequency[old].discard(key)                       
                for new in new_pairs:
                    pairs_to_frequency[old].discard(key)
                    pairs_to_frequency[new].add(new_key)
        pairs.pop(max_pair)
        index += 1
         

    return (vocab,merges)

class Tokenizer():
    def __init__(
            self, 
            vocab: dict[int,bytes],
            merges: list[tuple[bytes,bytes]],
            special_tokens: list[str] = None,
    ):
        self.vocab = vocab
        self.merges = merges
        self.max = len(merges)
        self.special_tokens = None if special_tokens is None else sorted(special_tokens, key = len, reverse = True)
        self.reversed_vocab = {v : k for k,v in vocab.items()}
        self.rank = defaultdict(lambda: self.max)
        for i in range(len(merges)):
            self.rank[merges[i]] = i
        self.escaped = '(' + '|'.join([re.escape(s) for s in self.special_tokens]) + ')' if special_tokens is not None else None
    @classmethod
    def from_files(
        cls, 
        vocab_filepath: str, 
        merges_filepath: str, 
        special_tokens: list[str] = None,
    ):
        pass
    def encode(self, text: str) -> list[int]:
        result = []
        ids = []
        #1.把一整段文本切成互不影响的 pretoken 单元
        if self.special_tokens is not None:
            #括号会让被分割的东西也保留在split返回的列表里
            for para in re.split(self.escaped, text):
                if para in self.special_tokens:
                    result.append((para.encode('utf-8'),))
                else:
                    for matched in re.finditer(PAT, para):
                        item = tuple(bytes([i]) for i in matched.group().encode('utf-8'))
                        result.append(item)
        else:
            for matched in re.finditer(PAT, text):
                    item = tuple(bytes([i]) for i in matched.group().encode('utf-8'))
                    result.append(item)
        #2.merge  

        for keys in result:
            if len(keys) == 1:
                ids.append(self.reversed_vocab[keys[0]])
            else:
                pairs = set(zip(keys[:-1],keys[1:]))
                while True:
                    ranks = {pair:self.rank[pair] for pair in pairs}
                    min_pair = min(ranks,key = ranks.get)
                    if ranks[min_pair] == self.max:
                        for key in keys:
                            ids.append(self.reversed_vocab[key])
                        break
                    i = 0
                    temp = []
                    while i < len(keys) - 1:
                        pair = (keys[i],keys[i+1])
                        if pair != min_pair:
                            temp.append(keys[i])
                        else:
                            new_token = keys[i] + keys[i+1]
                            if i != 0:
                                pairs.add((keys[i-1],new_token))
                            if i != len(keys)-2:
                                pairs.add((new_token,keys[i+2]))
                            temp.append(new_token)
                            i += 1
                        if i == len(keys) - 2:
                            temp.append(keys[i+1])
                        i += 1
                    keys = tuple(temp)
                    ranks[min_pair] = self.max
                    pairs.discard(min_pair)
                    if len(keys) == 1:
                        ids.append(self.reversed_vocab[keys[0]])
                        break

                
        return ids

    def encode_iterable(
            self, 
            iterable: Iterable[str]
    ) -> Iterator[int]:
        def out(inp):
            for string in inp:
                yield from self.encode(string)
        return out(iterable)
    
    def decode(self, ids: list[int]) -> str:
        text = []
        for token in ids:
            text.append(self.vocab[token])
        return b''.join(text).decode('utf-8',errors = 'replace')