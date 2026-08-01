import os
from typing import BinaryIO
import regex as re
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from bpe import train_bpe, Tokenizer
import time

def main():
    t = time.time()
    vocab_size = 10000
    special_tokens = ['<|endoftext|>']
    input_path = 'data/TinyStoriesV2-GPT4-valid.txt'
    vocab = train_bpe(input_path, vocab_size, special_tokens)[0]
    print(time.time()- t)
if __name__ == '__main__':
    main()