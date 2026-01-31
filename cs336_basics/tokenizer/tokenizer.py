import json
import os
import regex as re
import heapq
from tqdm import tqdm

from collections import Counter
from collections import defaultdict
from multiprocessing import Pool
from typing import Iterable, Iterator
from cs336_basics.tokenizer.pretokenization_example import find_chunk_boundaries


class DecreasingBytes:
    """让 bytes 的比较逻辑反转，用于堆中寻找最大字典序"""

    def __init__(self, data: bytes):
        self.data = data

    def __lt__(self, other):
        # 反转比较：我们想要字典序大的被认为"小"（优先弹出）
        return self.data > other.data

    def __le__(self, other):
        return self.data >= other.data

    def __gt__(self, other):
        return self.data < other.data

    def __ge__(self, other):
        return self.data <= other.data

    def __eq__(self, other):
        return self.data == other.data


def one_merge_step_naive(corpus, vocab, merges):
    # 步骤 1：统计所有相邻字节对的频率
    pair_freq = defaultdict(int)
    for ids, count in corpus:
        # 遍历每个单词内的所有相邻字节对
        for i in range(len(ids) - 1):
            pair = (ids[i], ids[i + 1])
            pair_freq[pair] += count
    if not pair_freq:
        return False

    # 步骤 2：找出最频繁的对
    max_freq_pair = max(
        pair_freq.items(), key=lambda x: (x[1], vocab[x[0][0]], vocab[x[0][1]])
    )[0]
    # 步骤 3：为这对分配新的 token ID，并更新 vocab 和 merges
    new_id = len(vocab)
    vocab[new_id] = vocab[max_freq_pair[0]] + vocab[max_freq_pair[1]]
    merges.append((vocab[max_freq_pair[0]], vocab[max_freq_pair[1]]))
    # 步骤 4：在 corpus 中执行合并
    for item in corpus:
        old_ids = item[0]
        new_ids = []
        i = 0
        while i < len(old_ids):
            if (
                i < len(old_ids) - 1
                and old_ids[i] == max_freq_pair[0]
                and old_ids[i + 1] == max_freq_pair[1]
            ):
                new_ids.append(new_id)
                i += 2
            else:
                new_ids.append(old_ids[i])
                i += 1
        item[0] = new_ids
    return True


def one_merge_step_with_index(corpus, vocab, merges, pair_counts, pair_to_words, heap):
    if not pair_counts:
        return False
    # 步骤 1：找出最频繁的对 (使用堆优化)
    max_freq_pair = None
    while heap:
        neg_count, _, _, pair = heapq.heappop(heap)
        # 检查是否是过时的数据
        if pair in pair_counts and pair_counts[pair] == -neg_count:
            max_freq_pair = pair
            break

    if max_freq_pair is None:
        return False
    # 步骤 2：为这对分配新的 token ID，并更新 vocab 和 merges
    new_id = len(vocab)
    vocab[new_id] = vocab[max_freq_pair[0]] + vocab[max_freq_pair[1]]
    merges.append((vocab[max_freq_pair[0]], vocab[max_freq_pair[1]]))

    # 步骤 3：找到需要修改的words
    target_word_indices = list(pair_to_words[max_freq_pair])
    changed_pairs = set()

    for word_idx in target_word_indices:
        item = corpus[word_idx]
        old_ids, count = item
        new_ids = []
        # 这个思路很重要，先清理这个词在全局的信息，然后更新这个词的id，再将这个词加到全局信息中
        for i in range(len(old_ids) - 1):
            p = (old_ids[i], old_ids[i + 1])
            pair_counts[p] -= count
            changed_pairs.add(p)
            if pair_counts[p] <= 0:
                del pair_counts[p]
            pair_to_words[p].discard(word_idx)

        i = 0
        while i < len(old_ids):
            if (
                i < len(old_ids) - 1
                and old_ids[i] == max_freq_pair[0]
                and old_ids[i + 1] == max_freq_pair[1]
            ):
                new_ids.append(new_id)
                i += 2
            else:
                new_ids.append(old_ids[i])
                i += 1
        item[0] = new_ids

        for i in range(len(new_ids) - 1):
            p = (new_ids[i], new_ids[i + 1])
            pair_counts[p] += count
            changed_pairs.add(p)
            pair_to_words[p].add(word_idx)

    # 将更新后的 pair 重新加入堆
    for p in changed_pairs:
        if p in pair_counts:
            heapq.heappush(
                heap,
                (
                    -pair_counts[p],
                    DecreasingBytes(vocab[p[0]]),
                    DecreasingBytes(vocab[p[1]]),
                    p,
                ),
            )

    return True


# 1. 这是一个由子进程执行的函数
def process_chunk(path, start, end, special_tokens, gpt2_pat):
    counts = Counter()
    with open(path, "rb") as f:
        f.seek(start)
        # 只读取属于自己的那一块
        data = f.read(end - start)
        text = data.decode("utf-8", errors="ignore")

        # 在这块文本内，按照特殊 Token 进行物理隔离
        if special_tokens:
            special_pat = "|".join([re.escape(t) for t in special_tokens])
            fragments = re.split(special_pat, text)
        else:
            fragments = [text]

        # 对每个片段进行 GPT-2 预分词正则统计
        for frag in fragments:
            for m in re.finditer(gpt2_pat, frag):
                counts[m.group(0).encode("utf-8")] += 1
    return counts


def train_bpe(
    input_path: str | os.PathLike, vocab_size: int, special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    # 初始化词表
    vocab = {i: bytes([i]) for i in range(256)}
    for special_token in special_tokens:
        vocab[len(vocab)] = special_token.encode(
            "utf-8"
        )  # 编码成bytes序列，如 b'<|endoftext|>'
    # 计算需要的合并次数
    num_merges = vocab_size - len(vocab)
    merges: list[tuple[bytes, bytes]] = []

    # 利用GPT-2的pre-tokenization规则进行划分
    gpt2_pat = (
        r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    )
    # 1. 统计原始单词频率
    with open(input_path, "rb") as f:
        num_processes = os.cpu_count() # 在cpu核数=128的情况下，测试会失败。因为开启这个多进程需要花费很多时间，所以测试的时候把它设置为1就行
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
    tasks = [
        (input_path, start, end, special_tokens, gpt2_pat)
        for start, end in zip(boundaries[:-1], boundaries[1:])
    ]

    with Pool(num_processes) as pool:
        partial_results = pool.starmap(process_chunk, tasks)

    word_counts = Counter()
    for res in partial_results:
        word_counts.update(res)

    # 2. 转换为 corpus 结构：List[ [Token_ID列表], 出现次数 ]
    corpus = []
    pair_counts = defaultdict(int)
    pair_to_words = defaultdict(set)
    for i, (word_bytes, count) in enumerate(word_counts.items()):
        word_ids = list(word_bytes)
        corpus.append([word_ids, count])
        for j in range(len(word_ids) - 1):
            pair = (word_ids[j], word_ids[j + 1])
            pair_counts[pair] += count
            pair_to_words[pair].add(i)

    # 3. 初始化堆 (考虑字典序决胜规则)
    heap = []
    for pair, count in pair_counts.items():
        heapq.heappush(
            heap,
            (
                -count,
                DecreasingBytes(vocab[pair[0]]),
                DecreasingBytes(vocab[pair[1]]),
                pair,
            ),
        )

    # 4. BPE 合并循环
    for _ in tqdm(range(num_merges), desc="Training BPE"):
        if not one_merge_step_with_index(
            corpus, vocab, merges, pair_counts, pair_to_words, heap
        ):
            break

    return vocab, merges


class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        """Construct a tokenizer from a given vocabulary, list of merges, and (optionally) a list of special tokens.
        This function should accept the following parameters:
        """
        self.vocab = vocab
        self.inverse_vocab = {v: k for k, v in vocab.items()}
        self.merges = merges
        self.init_merge_ids()
        self.special_tokens = special_tokens if special_tokens is not None else []
        self.gpt2_pat = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        self.cache = {}  # 简单的单词缓存

    def init_merge_ids(self):
        """
        提前计算好id合并规则
        """
        self.merge_ids_rank = {}
        for i, (bytes1, bytes2) in enumerate(self.merges):
            id1 = self.inverse_vocab[bytes1]
            id2 = self.inverse_vocab[bytes2]
            merge_id = self.inverse_vocab[bytes1 + bytes2]
            self.merge_ids_rank[(id1, id2)] = (i, merge_id)

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None,
    ):
        """Class method that constructs and return a Tokenizer from a serialized vocabulary and list of merges
        (in the same format that your BPE training code output) and (optionally) a list of special tokens.
        This method should accept the following additional parameters:

        vocab_filepath: str
        merges_filepath: str
        special_tokens: list[str] | None = None
        """
        with open(vocab_filepath, "r") as f:
            vocab_data = json.load(f)
        vocab = {int(k): bytes(v) for k, v in vocab_data.items()}
        merges = []
        with open(merges_filepath, "r") as f:
            for line in f:
                parts = line.split()
                if len(parts) != 2:
                    continue
                hex1, hex2 = parts
                merges.append((bytes.fromhex(hex1), bytes.fromhex(hex2)))
        return cls(vocab, merges, special_tokens)

    def _bpe(self, word_ids):
        # 以token_ids为核心，去查看里面的pair是否存在merges中，如果存在多个，那么需要rank最小的那个
        while len(word_ids) >= 2:
            min_rank = float("inf")
            min_pair = None
            min_new_id = None
            for i in range(len(word_ids) - 1):
                pair = (word_ids[i], word_ids[i + 1])
                if pair in self.merge_ids_rank:
                    rank, new_id = self.merge_ids_rank[pair]
                    if rank < min_rank:
                        min_rank = rank
                        min_pair = pair
                        min_new_id = new_id

            if min_pair == None:
                break

            new_word_ids = []
            i = 0
            while i < len(word_ids):
                if (
                    i < len(word_ids) - 1
                    and word_ids[i] == min_pair[0]
                    and word_ids[i + 1] == min_pair[1]
                ):
                    new_word_ids.append(min_new_id)
                    i += 2
                else:
                    new_word_ids.append(word_ids[i])
                    i += 1
            word_ids = new_word_ids

        return word_ids

    def _encode_standard_text(self, text: str) -> list[int]:
        words = re.findall(self.gpt2_pat, text)
        res_ids = []
        # 将每一个word拆分成ids
        for word in words:
            if word in self.cache:
                res_ids.extend(self.cache[word])
                continue
            # word_ids = list(word.encode('utf-8')) 不能假设字节值就是这个字符的ID，比如 \xc2的字节值是194，但是对应的ID是99
            word_bytes = word.encode("utf-8")
            word_ids = [self.inverse_vocab[bytes([b])] for b in word_bytes]

            word_ids = self._bpe(word_ids)

            self.cache[word] = word_ids
            res_ids.extend(word_ids)
        return res_ids

    def encode(self, text: str) -> list[int]:
        """Encode an input text into a sequence of token IDs."""
        if not self.special_tokens:
            return self._encode_standard_text(text)
        # 用special tokens来切分text，并且要包含special tokens
        sorted_special_tokens = sorted(self.special_tokens, key=len, reverse=True)
        special_pat = "(" + "|".join(re.escape(t) for t in sorted_special_tokens) + ")"
        parts = re.split(special_pat, text)
        res_ids = []
        for part in parts:
            if part in sorted_special_tokens:
                res_ids.append(self.inverse_vocab[part.encode("utf-8")])
            else:
                res_ids.extend(self._encode_standard_text(part))
        return res_ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """Given an iterable of strings (e.g., a Python file handle), return a generator that lazily yields token IDs. This is required for memory-efficient tokenization of large files that we cannot directly load into memory."""
        for text in iterable:
            for idx in self.encode(text):
                yield idx

    def decode(self, ids: list[int]) -> str:
        """Decode a sequence of token IDs into text."""
        token_bytes = [self.vocab[idx] for idx in ids]
        concatenated_bytes = b"".join(token_bytes)
        # 使用 errors='replace' 防止某些切片导致非法 UTF-8
        return concatenated_bytes.decode("utf-8", errors="replace")


if __name__ == "__main__":
    input_path = (
        "/home/jkd/online_course_learning/cs336/assignment1-basics/data/for_debug.txt"
    )
    vocab_size = 1000
    special_tokens = ["<|endoftext|>", "<|nan|>"]
    vocab, merges = train_bpe(input_path, vocab_size, special_tokens)
    # print(vocab, merges)
