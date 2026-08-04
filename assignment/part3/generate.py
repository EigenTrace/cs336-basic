"""Generate text from a trained TinyStories language model.

Usage (from repo root):
    uv run python -m assignment.part3.generate
    uv run python -m assignment.part3.generate "Once upon a time"
"""

import pickle
import sys

import torch

from assignment.part1.tokenizer import Tokenizer
from assignment.part2.all import transformer_lm
from assignment.part3.all import Decoding

# ---- must match the config the checkpoint was trained with ----
VOCAB_SIZE = 10000
CONTEXT_LENGTH = 256
D_MODEL = 512
NUM_LAYERS = 4
NUM_HEADS = 16
D_FF = 1344
ROPE_THETA = 10000.0

CKPT = "ckpt_tinyv1.pt"
VOCAB_PKL = "assignment/part1/tinystories_bpe/vocab.pkl"
MERGES_PKL = "assignment/part1/tinystories_bpe/merges.pkl"
SPECIAL = "<|endoftext|>"


def load_model(ckpt_path: str, device: str) -> torch.nn.Module:
    """Rebuild the model and load trained weights (ignores optimizer state)."""
    model = transformer_lm(
        VOCAB_SIZE, CONTEXT_LENGTH, D_MODEL, NUM_LAYERS, NUM_HEADS, D_FF, ROPE_THETA
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt["model_state"]
    # torch.compile prefixes keys with "_orig_mod." -- strip it if present
    state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()
    print(f"loaded {ckpt_path} (trained {ckpt['ite']} steps)")
    return model


def main() -> None:
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    prompt_text = sys.argv[1] if len(sys.argv) > 1 else "Once upon a time"

    with open(VOCAB_PKL, "rb") as f:
        vocab = pickle.load(f)
    with open(MERGES_PKL, "rb") as f:
        merges = pickle.load(f)
    tokenizer = Tokenizer(vocab, merges, [SPECIAL])
    stop_id = tokenizer.inverse_vocab[SPECIAL.encode("utf-8")]

    model = load_model(CKPT, device)

    prompt_ids = tokenizer.encode(prompt_text)
    prompt = torch.tensor([prompt_ids], dtype=torch.long, device=device)  # (1, seq)

    # try a few sampling settings so you can compare
    for tem, p in [(1.0, 0.95), (0.8, 0.9), (0.6, 0.9)]:
        out = Decoding(model, prompt, max_len=256, tem=tem, p=p, stop_id=stop_id)
        text = tokenizer.decode(out[0].tolist())
        print(f"\n{'=' * 70}\ntemperature={tem}  top_p={p}\n{'=' * 70}")
        print(text)


if __name__ == "__main__":
    main()
