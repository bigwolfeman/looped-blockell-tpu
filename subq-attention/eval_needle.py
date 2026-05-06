"""Needle-in-a-Haystack evaluation for subquadratic attention methods.

Tests whether the attention can retrieve a specific fact placed at a known
position within a long context of filler text. No pretrained model needed —
we train a small model, inject a "needle" pattern, and check if it's retrieved.

For untrained models, this tests the attention mechanism's raw retrieval
capability: can the model attend to the right position?
"""

import argparse
import math
import random

import torch
import torch.nn.functional as F

from model import SubQTransformer, ModelConfig


def make_needle_batch(
    seq_len: int,
    batch_size: int,
    vocab_size: int = 49152,
    needle_len: int = 16,
    device: str = "cuda",
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    """Create sequences with a unique needle pattern at random positions.

    Returns (input_ids, labels, needle_positions).
    Labels are -100 everywhere except the token right after the needle,
    where the label is a specific "answer" token.
    """
    ANSWER_TOKEN = 42
    NEEDLE_MARKER = 1337

    ids = torch.randint(100, vocab_size - 100, (batch_size, seq_len), device=device)
    labels = torch.full((batch_size, seq_len), -100, device=device, dtype=torch.long)
    positions = []

    for b in range(batch_size):
        max_pos = seq_len - needle_len - 2
        pos = random.randint(0, max_pos)
        positions.append(pos)

        ids[b, pos] = NEEDLE_MARKER
        ids[b, pos + 1 : pos + needle_len] = ANSWER_TOKEN
        ids[b, pos + needle_len] = NEEDLE_MARKER

        # The query at the end asks "what was the needle?"
        query_pos = seq_len - 3
        ids[b, query_pos] = NEEDLE_MARKER
        labels[b, query_pos + 1] = ANSWER_TOKEN

    return ids, labels, positions


def eval_needle(model, seq_len: int, batch_size: int = 8, n_batches: int = 20):
    """Evaluate needle retrieval accuracy."""
    device = next(model.parameters()).device
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for _ in range(n_batches):
            ids, labels, positions = make_needle_batch(
                seq_len, batch_size, device=device
            )
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(ids)

            logits = out["logits"]
            for b in range(batch_size):
                query_pos = seq_len - 3
                pred = logits[b, query_pos + 1].argmax().item()
                if pred == 42:  # ANSWER_TOKEN
                    correct += 1
                total += 1

    acc = correct / total if total > 0 else 0
    model.train()
    return acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--seq_lengths", type=int, nargs="+", default=[2048, 4096, 8192, 16384])
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--n_batches", type=int, default=20)
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, weights_only=False)
    cfg = ckpt["config"]
    model = SubQTransformer(cfg).cuda()
    model.load_state_dict(ckpt["model_state_dict"])

    for sl in args.seq_lengths:
        if sl > cfg.max_seq_len:
            print(f"  seq_len={sl}: SKIP (exceeds max_seq_len={cfg.max_seq_len})")
            continue
        acc = eval_needle(model, sl, args.batch_size, args.n_batches)
        print(f"  seq_len={sl:6d}: needle accuracy = {acc:.1%}")
