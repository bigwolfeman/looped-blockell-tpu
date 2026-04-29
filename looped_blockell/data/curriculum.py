"""Multi-phase curriculum data loader: Dolma → StarCoder → OpenHermes.

Each phase can have different:
- Dataset (HF streaming)
- Sequence length
- Batch size
- Tokenizer (all use StarCoder2 for now — shared vocab)

Usage::

    phases = [
        CurriculumPhase("dolma", "allenai/dolma", end_step=200000, seq_len=1024, batch_size=20),
        CurriculumPhase("starcoder", "bigcode/starcoderdata", end_step=280000, seq_len=2048, batch_size=10),
        CurriculumPhase("openhermes", "teknium/OpenHermes-2.5", end_step=300000, seq_len=2048, batch_size=20),
    ]
    loader = CurriculumLoader(phases, tokenizer_name="bigcode/starcoder2-7b")
    for step in range(300000):
        x, y = loader.get_batch(step)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import jax.numpy as jnp


@dataclass
class CurriculumPhase:
    name: str
    dataset_id: str
    end_step: int
    seq_len: int = 1024
    batch_size: int = 20
    n_epochs: int = 1
    dataset_split: str = "train"
    text_field: str = "text"
    buffer_tokens: int = 50_000_000


class CurriculumLoader:
    """Streams data from multiple HF datasets in sequence.

    Automatically transitions between phases based on step count.
    Each phase has its own dataset, seq_len, and batch_size.
    """

    def __init__(
        self,
        phases: list[CurriculumPhase],
        tokenizer_name: str = "bigcode/starcoder2-7b",
    ):
        from transformers import AutoTokenizer

        self.phases = sorted(phases, key=lambda p: p.end_step)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
        self.eos_id = self.tokenizer.eos_token_id

        self._current_phase_idx = 0
        self._stream = None
        self._buf = None
        self._buf_len = 0
        self._cursor = 0
        self._current_epoch = 0
        self.total_tokens_consumed = 0

        self._init_phase(0)

    @property
    def current_phase(self) -> CurriculumPhase:
        return self.phases[self._current_phase_idx]

    @property
    def phase_name(self) -> str:
        return self.current_phase.name

    def _init_phase(self, idx: int):
        self._current_phase_idx = idx
        phase = self.phases[idx]
        self._buf = np.zeros(phase.buffer_tokens, dtype=np.int32)
        self._buf_len = 0
        self._cursor = 0
        self._current_epoch = 0
        self._init_stream()
        self._fill_buffer()
        print(f"  Curriculum phase: {phase.name} (seq={phase.seq_len}, batch={phase.batch_size})")

    def _init_stream(self):
        from datasets import load_dataset
        phase = self.current_phase
        self._current_epoch += 1
        print(f"  Starting {phase.dataset_id} stream "
              f"(epoch {self._current_epoch}/{phase.n_epochs})...")

        ds_kwargs = dict(split=phase.dataset_split, streaming=True)
        if phase.dataset_id == "allenai/dolma":
            ds_kwargs["trust_remote_code"] = True

        self._stream = iter(load_dataset(phase.dataset_id, **ds_kwargs))

    def _fill_buffer(self):
        phase = self.current_phase
        tokens = []
        while len(tokens) < phase.buffer_tokens:
            try:
                sample = next(self._stream)
            except StopIteration:
                if self._current_epoch < phase.n_epochs:
                    self._init_stream()
                    continue
                else:
                    break
            text = sample.get(phase.text_field, "")
            if not text:
                continue
            ids = self.tokenizer.encode(text, add_special_tokens=False)
            ids.append(self.eos_id)
            tokens.extend(ids)

        n = min(len(tokens), phase.buffer_tokens)
        if n == 0:
            print(f"  WARNING: No tokens available for phase {phase.name}")
            return
        self._buf[:n] = np.array(tokens[:n], dtype=np.int32)
        self._buf_len = n
        self._cursor = 0
        print(f"  Buffer filled: {n:,} tokens "
              f"(total consumed: {self.total_tokens_consumed / 1e9:.2f}B)")

    def get_batch(self, step: int) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Get next batch, transitioning phases as needed."""
        # Check if we need to advance to next phase
        while (self._current_phase_idx < len(self.phases) - 1
               and step >= self.current_phase.end_step):
            self._init_phase(self._current_phase_idx + 1)

        phase = self.current_phase
        window = phase.seq_len + 1

        seqs = []
        for _ in range(phase.batch_size):
            if self._cursor + window > self._buf_len:
                self._fill_buffer()
                if self._buf_len < window:
                    raise RuntimeError(f"Data exhausted in phase {phase.name}")
            seq = self._buf[self._cursor:self._cursor + window]
            seqs.append(seq)
            self._cursor += phase.seq_len
            self.total_tokens_consumed += phase.seq_len

        seqs = np.stack(seqs)
        x = jnp.asarray(seqs[:, :-1])
        y = jnp.asarray(seqs[:, 1:])
        return x, y
