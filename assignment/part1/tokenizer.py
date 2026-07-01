import pickle
import regex as re
from collections.abc import Iterable
from collections.abc import Iterator

class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens
        # Add any special tokens not already present in the vocab.
        if special_tokens:
            existing = set(vocab.values())
            for token in special_tokens:
                b = token.encode("utf-8")
                if b not in existing:
                    vocab[len(vocab)] = b
                    existing.add(b)
        self.inverse_vocab={}
        for k,v in vocab.items():
            self.inverse_vocab[v]=k
        # Merge priority: earlier merges have lower rank (higher priority).
        self.ranks={pair:i for i,pair in enumerate(merges)}
        # Cache encoded ids per pre-token (pre-tokens repeat heavily in text).
        self._cache={}

    @classmethod
    def from_files(cls, vocab_filepath:str, merges_filepath:str, special_tokens:list[str] | None = None):
        with open(vocab_filepath,"rb") as f:
            vocab=pickle.load(f)
        with open(merges_filepath,"rb") as f:
            merges=pickle.load(f)
        return cls(vocab,merges,special_tokens)
    
    def encodeword(self,word:tuple[bytes,...])->list[int]:
        if word in self._cache:
            return self._cache[word]
        w=list(word)
        while len(w)>=2:
            # Pick the adjacent pair with the highest merge priority (lowest rank).
            best_rank=None
            best_i=None
            for i in range(len(w)-1):
                r=self.ranks.get((w[i],w[i+1]))
                if r is not None and (best_rank is None or r<best_rank):
                    best_rank=r
                    best_i=i
            if best_i is None:
                break   # no adjacent pair is mergeable
            w[best_i:best_i+2]=[w[best_i]+w[best_i+1]]
        result=[self.inverse_vocab[key] for key in w]
        self._cache[word]=result
        return result
                
                

    
    def encode(self, text: str) -> list[int]:
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        _PAT = re.compile(PAT)

        if self.special_tokens:
            # Longest first so overlapping specials (e.g. "<|eot|><|eot|>") win.
            specials=sorted(self.special_tokens,key=len,reverse=True)
            delimit="("+"|".join(re.escape(token) for token in specials)+")"
            chunk=re.split(delimit,text)
        else:
            chunk=[text]
        special_set=set(self.special_tokens or [])
        result=[]
        for c in chunk:
            if not c:
                continue
            if c in special_set:
                result.append(self.inverse_vocab[c.encode("utf-8")])
                continue
            for word in _PAT.finditer(c):
                word=word.group().encode('utf-8')
                word=tuple(bytes([b]) for b in word)
                num=self.encodeword(word)
                result+=num
        return result
    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for s in iterable:
            ids=self.encode(s)
            for id in ids:
                yield id


    def decode(self, ids: list[int]) -> str:
        byte=b''
        for id in ids:
            byte+=self.vocab[id]
        return byte.decode('utf-8', errors='replace')



