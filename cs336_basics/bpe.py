import os
from typing import BinaryIO
import regex as re
from collections import Counter
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
    vocab = {i : bytes([i]) for i in range(256)}
    for k,v in enumerate(special_tokens):
        vocab[256 + k] = v.encode('utf-8')
    index = 256 + len(special_tokens)

    #2.读取文件,确定分割点,然后pre-tokenization
    frequency = Counter()  #pre-tokenization后的frequecy_table
    #转义后的special_tokens
    escaped = '|'.join([re.escape(s) for s in special_tokens])
    num_processes = 4    #可改
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            for para in re.split(escaped, chunk):
                for matched in re.finditer(PAT, para):
                    #迭代一个 bytes 对象,取出来的每个元素是 int(0-255 之间的数值),不是 bytes
                    key = tuple(bytes([i]) for i in matched.group().encode('utf-8'))
                    frequency[key] += 1
    #3.merge
    merges = []
    while index < vocab_size:
        pairs = Counter()
        for t, count in frequency.items():
            if len(t) <= 1:
                continue
            for i in range(len(t)-1):
                k = (t[i],t[i+1])
                pairs[k] += count
        max_pair = max(pairs, key = lambda x: (pairs[x], x))
        vocab[index] = max_pair[0] + max_pair[1]
        merges.append(max_pair)
        index += 1
        temp = Counter()
        for t, count in frequency.items():
            if len(t) <= 1:
                continue
            middle = []
            i = 0
            while i < len(t)-1:
                k = (t[i],t[i+1])
                if k != max_pair:
                    middle.append(t[i])
                else:
                    middle.append(t[i]+t[i+1])
                    i += 1
                if i == len(t) - 2:
                        middle.append(t[i+1])
                i += 1
            temp[tuple(middle)] = count
        frequency = temp

    return (vocab,merges)

class Tokenizer():
    def __init__(
            self, 
            vocab: dict[int:bytes],
            merges: list[tuple[bytes,bytes]],
            special_tokens: list[str] = None,
    ):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = None if special_tokens is None else sorted(special_tokens, key = len, reverse = True)
        self.reversed_vocab = {v : k for k,v in vocab.items()}
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
        if self.special_tokens is not None:
            escaped = '(' + '|'.join([re.escape(s) for s in self.special_tokens]) + ')'
            for para in re.split(escaped, text):
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
        i, num_update = 0, len(self.merges)
        while i < num_update:
            pick = self.merges[i]
            for n in range(len(result)):
                item = result[n]
                if len(item) == 1:
                    continue
                middle = []
                j = 0
                while j <= len(item) - 2:
                    k = (item[j],item[j+1])
                    if k != pick:
                        middle.append(item[j])
                    else:
                        middle.append(item[j]+item[j+1])
                        j += 1
                    j += 1
                    if j == len(item) - 1:
                        middle.append(item[j])
                result[n] = tuple(middle)
            i += 1
        for item in result:
            if len(item) == 1:
                ids.append(self.reversed_vocab[item[0]])
            else:
                for token in item:
                    ids.append(self.reversed_vocab[token])
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