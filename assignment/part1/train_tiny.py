import os
import sys
import time
import pickle
import resource

from .bpe import train_bpe


def peak_memory_mb() -> float:
    """Peak RSS across this process and its worker children."""
    rss = (
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        + resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    )
    # macOS reports ru_maxrss in bytes, Linux in kibibytes.
    divisor = 1024 ** 2 if sys.platform == "darwin" else 1024
    return rss / divisor


def main() -> None:
    input_path = "data/TinyStoriesV2-GPT4-train.txt"
    vocab_size = 10000
    special_tokens = ["<|endoftext|>"]
    out_dir = "assignment/part1/tinystories_bpe"
    os.makedirs(out_dir, exist_ok=True)

    start = time.perf_counter()
    vocab, merges = train_bpe(input_path, vocab_size, special_tokens)
    elapsed = time.perf_counter() - start

    # --- serialize results for later inspection / reuse ---
    with open(os.path.join(out_dir, "vocab.pkl"), "wb") as f:
        pickle.dump(vocab, f)
    with open(os.path.join(out_dir, "merges.pkl"), "wb") as f:
        pickle.dump(merges, f)

    # --- report ---
    longest = max(vocab.values(), key=len)
    print(f"training time:   {elapsed:.1f} s ({elapsed/60:.2f} min)")
    print(f"peak memory:     {peak_memory_mb():.1f} MB")
    print(f"vocab size:      {len(vocab)}")
    print(f"num merges:      {len(merges)}")
    print(f"longest token:   {longest!r}  (len {len(longest)})")
    print(f"saved to:        {out_dir}/vocab.pkl, {out_dir}/merges.pkl")


if __name__ == "__main__":
    main()