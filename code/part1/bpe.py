from cs336_basics.pretokenization_example import find_chunk_boundaries
import os
from multiprocessing import Pool
from collections import Counter
import regex as re  

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
_PAT = re.compile(PAT)

def count(chunk:tuple)->Counter[tuple[bytes,...]]:
    start,end,input_path,special_tokens=chunk
    dic_count=Counter()
    with open(input_path,'rb') as f:
        f.seek(start)
        chunk2=f.read(end-start).decode("utf-8","ignore")
    delimit="|".join(re.escape(token) for token in special_tokens)
    chunk3=re.split(delimit,chunk2)
    for c in chunk3:
        for word in _PAT.finditer(c):
            b=word.group().encode("utf-8")
            dic_count[tuple(bytes([w]) for w in b)]+=1

    return dic_count


def merge(byte_count:Counter[tuple[bytes,...]],size->int)->tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    dic={}
    for i in range(256):
        dic[i]=bytes([i])
    
    num=Counter()
    process=[]
    for word in byte_count:
        for i in range(len(word)-1):
            pair=(word[i],word[i+1])
            num[pair]+=byte_count[word]

    while len(dic)<size:
        key,count=num.most_common(1)
        process.append(key)
        


def train_bpe(
        input_path: str,
        vocab_size: int,
        special_tokens: list[str]
) ->  tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    num_processes = os.cpu_count()
    with open(input_path,'rb') as f:
        boundaries=find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
    
    delimit=[(start,end,input_path,special_tokens) for start,end in zip(boundaries[:-1],boundaries[1:]) ]
    byte_count=Counter()
    with Pool(num_processes) as pool:
        for dic in pool.imap_unordered(count,delimit):
            byte_count.update(dic)
    
    result,process=merge(byte_count,vocab_size)

    return result,process
