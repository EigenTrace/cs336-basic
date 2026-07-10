import os
import pickle
from .tokenizer import Tokenizer
import sys
import time
import resource
import numpy as np

def peak_memory_mb() -> float:
    """Peak RSS across this process and its worker children."""
    rss = (
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        + resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    )
    # macOS reports ru_maxrss in bytes, Linux in kibibytes.
    divisor = 1024 ** 2 if sys.platform == "darwin" else 1024
    return rss / divisor

def main()->None:
    with open("assignment/part1/tinystories_bpe/vocab.pkl","rb") as f:
        vocab=pickle.load(f)
    with open("assignment/part1/tinystories_bpe/merges.pkl",'rb') as f:
        merges=pickle.load(f)
    tiny_token=Tokenizer(vocab,merges,["<|endoftext>|"])

    with open('data/TinyStoriesV2-GPT4-train.txt',encoding='utf-8') as f:
        text=f.read(1_000_000)
    chunks=[c for c in text.split("<|endoftext|>") if c.strip()][:10]
    
    lennum=0
    lenchar=0
    for c in chunks:
        lenchar+=len(c.encode("utf-8"))
        tinycode=tiny_token.encode(c)
        lennum+=len(tinycode)
    
    print(f"For tiny dataset. byte count:{lenchar}  token count:{lennum}  compression ratio:{lenchar/lennum}")




    with open("assignment/part1/owt_bpe/vocab.pkl","rb") as f:
        vocab1=pickle.load(f)
    with open("assignment/part1/owt_bpe/merges.pkl","rb") as f:
        merges1=pickle.load(f)
    owt_token=Tokenizer(vocab1,merges1,["<|endoftext|>"])

    with open("data/owt_train.txt",encoding="utf-8") as f:
        text=f.read(1_000_000)
    chunks=[c for c in text.split("<|endoftext|>") if c.strip()][:10]
    lennum=0
    lenchar=0
    s=time.time()
    for c in chunks:
        lenchar+=len(c.encode("utf-8"))
        tinycode=owt_token.encode(c)
        lennum+=len(tinycode)
    e=time.time()
    t=e-s
    print(f"For owt dataset. byte count:{lenchar}  token count:{lennum}  compression ratio:{lenchar/lennum}")
    print(f"encode time:{e-s}s  encode speed:{(lenchar)/(e-s)} bytes/s")

    lennum=0
    for c in chunks:
        tinycode=tiny_token.encode(c)
        lennum+=len(tinycode)
    print(f"For owt dataset using tiny tokenizer. byte count:{lenchar}  token count:{lennum}  compression ratio:{lenchar/lennum}")

    if not os.path.exists("data/tiny_train.npy"):
        with open("data/TinyStoriesV2-GPT4-train.txt", encoding="utf-8") as f:
            arr = np.fromiter(tiny_token.encode_iterable(f), dtype=np.uint16)
            np.save("data/tiny_train.npy", arr)

    

    if not os.path.exists("data/owt_valid.npy"):
        with open("data/owt_valid.txt", encoding="utf-8") as f:
            arr = np.fromiter(owt_token.encode_iterable(f), dtype=np.uint16)
            np.save("data/owt_valid.npy", arr)

    if not os.path.exists("data/tiny_valid.npy"):
        with open("data/TinyStoriesV2-GPT4-valid.txt", encoding="utf-8") as f:
            arr = np.fromiter(tiny_token.encode_iterable(f), dtype=np.uint16)
            np.save("data/tiny_valid.npy", arr)


    if not os.path.exists("data/owt_train.npy"):
        with open("data/owt_train.txt", encoding="utf-8") as f:
            arr = np.fromiter(owt_token.encode_iterable(f), dtype=np.uint16)
            np.save("data/owt_train.npy", arr)
    
if __name__=="__main__":
    main()