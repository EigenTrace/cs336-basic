from cs336_basics.pretokenization_example import find_chunk_boundaries
import os
from multiprocessing import Pool
from collections import Counter, defaultdict
import regex as re

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
_PAT = re.compile(PAT)


def count(chunk: tuple) -> Counter[tuple[bytes, ...]]:
    start, end, input_path, special_tokens = chunk
    dic_count = Counter()
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk2 = f.read(end - start).decode("utf-8", "ignore")
    if special_tokens:
        delimit = "|".join(re.escape(token) for token in special_tokens)
        chunk3 = re.split(delimit, chunk2)
    else:
        chunk3 = [chunk2]
    for c in chunk3:
        for word in _PAT.finditer(c):
            b = word.group().encode("utf-8")
            dic_count[tuple(bytes([w]) for w in b)] += 1

    return dic_count


def merge(
    byte_count: Counter[tuple[bytes, ...]], size: int, special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    dic = {}
    for i in range(256):
        dic[i] = bytes([i])
    for token in special_tokens:
        dic[len(dic)] = token.encode("utf-8")
    word = []
    frq = []
    process = []

    pair_count = Counter()
    index = defaultdict(set)
    for w in byte_count:
        word.append(w)
        frq.append(byte_count[w])

        for pair in zip(w, w[1:]):
            pair_count[pair] += byte_count[w]
            index[pair].add(len(word) - 1)

    while len(dic) < size:
        if not pair_count:
            break
        best = max(pair_count, key=lambda p: (pair_count[p], p))
        num = pair_count[best]
        dic[len(dic)] = best[0] + best[1]
        process.append(best)

        for wid in list(index[best]):
            w = word[wid]
            f = frq[wid]

            old_pair = Counter(zip(w, w[1:]))

            new_word = []
            i = 0
            while i < len(w):

                if i < len(w) - 1 and w[i] == best[0] and w[i + 1] == best[1]:
                    new_word.append(best[0] + best[1])
                    i += 2
                else:
                    new_word.append(w[i])
                    i += 1
            word[wid] = new_word

            new_pair = Counter(zip(new_word, new_word[1:]))

            for p, c in (new_pair - old_pair).items():
                pair_count[p] += f * c
                index[p].add(wid)
            for p, c in (old_pair - new_pair).items():
                pair_count[p] -= f * c
                if pair_count[p] == 0:
                    del pair_count[p]
                if new_pair[p] == 0:
                    index[p].discard(wid)
        del pair_count[best]
        del index[best]
    return dic, process


def train_bpe(
    input_path: str, vocab_size: int, special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    num_processes = os.cpu_count()
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

    delimit = [
        (start, end, input_path, special_tokens)
        for start, end in zip(boundaries[:-1], boundaries[1:])
    ]
    byte_count = Counter()
    with Pool(num_processes) as pool:
        for dic in pool.imap_unordered(count, delimit):
            byte_count.update(dic)

    result, process = merge(byte_count, vocab_size, special_tokens)

    return result, process
