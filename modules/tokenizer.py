import gzip
import html
import os
from functools import lru_cache

import ftfy
import numpy as np
import regex as re
from partd.file import token


def default_bpe():
    return "C:\\project\\caption\\bpe_simple_vocab_16e6.txt.gz"


"""
生成一个从 UTF-8 字节值（0-255）到对应的 Unicode 字符的映射。
确保映射避免使用空白字符和控制字符（这些字符可能会干扰 BPE 的处理）。
返回一个字典，用于后续的字节到 Unicode 的转换。
"""
def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a corresponding list of unicode strings.
    The reversible bpe codes work on unicode strings.
    This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
    When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
    This is a signficant percentage of your normal, say, 32K bpe vocab.
    To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
    And avoids mapping to whitespace/control characters the bpe code barfs on.
    """

    """
      返回 UTF-8 字节和 Unicode 字符串的映射。
      BPE 工作在 Unicode 字符串上，需要较大的词汇表（例如 5K）来覆盖 10B token 数据集，否则会产生大量 UNK（未知词）。
      通过字节到 Unicode 的查找表，避免直接使用空白和控制字符（这些字符可能导致 BPE 处理出错）。
    """
    bs = list(range(ord("!"), ord("~")+1))+list(range(ord("¡"), ord("¬")+1))+list(range(ord("®"), ord("ÿ")+1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8+n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


def get_pairs(word):
    """Return set of symbol pairs in a word.
    Word is represented as tuple of symbols (symbols being variable-length strings).
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


"""
这个函数用于修复文本中的字符问题，并将 HTML 实体转义字符转化为正常的字符。最终，它会去除文本两端的多余空白字符。
"""
def basic_clean(text):
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()

"""
这个函数用于将文本中的多余空白字符（如多个空格、制表符、换行符等）替换为单一的空格，并去除文本两端的多余空白字符。
"""
def whitespace_clean(text):
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text


class SimpleTokenizer(object):
    def __init__(self, bpe_path: str = default_bpe()):
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        merges = gzip.open(bpe_path).read().decode("utf-8").split('\n')
        merges = merges[1:49152-256-2+1]
        merges = [tuple(merge.split()) for merge in merges]
        vocab = list(bytes_to_unicode().values())
        vocab = vocab + [v+'</w>' for v in vocab]
        for merge in merges:
            vocab.append(''.join(merge))
        vocab.extend(['<|startoftext|>', '<|endoftext|>'])
        self.encoder = dict(zip(vocab, range(len(vocab))))
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        self.cache = {'<|startoftext|>': '<|startoftext|>', '<|endoftext|>': '<|endoftext|>'}
        self.pat = re.compile(r"""<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""", re.IGNORECASE)

        self.vocab = self.encoder

    # 这段代码定义了 bpe 方法，这是实现字节对编码（BPE, Byte Pair Encoding）的核心部分。
    # 其作用是对输入的 token（通常是一个词或符号）进行 BPE 分解，逐步将词拆分成更小的子词或字符。
    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token[:-1]) + ( token[-1] + '</w>',)
        pairs = get_pairs(word)

        if not pairs:
            return token+'</w>'

        while True:
            bigram = min(pairs, key = lambda pair: self.bpe_ranks.get(pair, float('inf')))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word)-1 and word[i+1] == second:
                    new_word.append(first+second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = ' '.join(word)
        self.cache[token] = word
        return word

    # 主要用于将输入的文本进行编码，转换成由 BPE（字节对编码）标记组成的列表
    def encode(self, text):
        bpe_tokens = []
        text = whitespace_clean(basic_clean(text)).lower()
        for token in re.findall(self.pat, text):
            token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
            bpe_tokens.extend(self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' '))
        return bpe_tokens

    def decode_list(self, tokens):
        t = tokens[0]
        return [str(self.decoder.get(token, '[UNK]')).replace('</w>',' ') for token in tokens]

    # 将 BPE 编码后的标记（tokens）解码回原始文本
    def decode(self, tokens):
        text = ''.join([self.decoder.get(token, '[UNK]') for token in tokens])
        text = bytearray([self.byte_decoder[c] for c in text]).decode('utf-8', errors="replace").replace('</w>', ' ')
        return text

    # 将输入的文本转换为一个由 BPE（字节对编码）标记组成的列表
    def tokenize(self, text):
        tokens = []
        text = whitespace_clean(basic_clean(text)).lower()
        for token in re.findall(self.pat, text):
            token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
            tokens.extend(bpe_token for bpe_token in self.bpe(token).split(' '))
        return tokens

    def convert_tokens_to_ids(self, tokens):
        return [self.encoder[bpe_token] for bpe_token in tokens]