# SPDX-License-Identifier: Apache-2.0
"""Deterministic tokenized workloads for the Ray SFT batching benchmark."""

from math import gcd

import torch
from torch.utils.data import Dataset


REFERENCE_CONTEXT = 16_384
LENGTHS = {
    "mixed": (
        (2_048,) * 8
        + (3_072,) * 8
        + (4_096,) * 6
        + (6_144,) * 4
        + (8_192,) * 3
        + (12_288,) * 2
        + (16_384,)
    ),
    "narrow": (2_816, 3_072, 3_328),
}
EVALUATION_OFFSETS = {"mixed-calibration": 65_536, "mixed-holdout": 65_792}
MAX_EVALUATION_SAMPLES = 256
SAMPLE_STRIDE = 97


class SyntheticDataset(Dataset):
    """Supply the same tokenized samples to every arm of a workload."""

    def __init__(self, parquet_files, tokenizer, config, processor=None, max_samples=-1):
        if not isinstance(parquet_files, str) or (parquet_files not in LENGTHS and parquet_files not in EVALUATION_OFFSETS):
            raise ValueError("SyntheticDataset requires mixed, narrow, mixed-calibration, or mixed-holdout")
        if config.pad_mode != "no_padding":
            raise ValueError("SyntheticDataset requires data.pad_mode=no_padding")
        if processor is not None:
            raise ValueError("SyntheticDataset is for text models")
        if config.max_length < 2 or tokenizer.vocab_size < 2:
            raise ValueError("SyntheticDataset needs a sequence and vocabulary of size at least two")

        self.vocab_size = tokenizer.vocab_size
        self.sample_offset = EVALUATION_OFFSETS.get(parquet_files, 0)
        workload = "mixed" if self.sample_offset else parquet_files
        self.lengths = [max(2, length * config.max_length // REFERENCE_CONTEXT) for length in LENGTHS[workload]]
        default_size = MAX_EVALUATION_SAMPLES if self.sample_offset else 1_024
        self.size = max_samples if max_samples > 0 else default_size
        if self.sample_offset:
            if self.size > MAX_EVALUATION_SAMPLES:
                raise ValueError(f"Evaluation split must contain at most {MAX_EVALUATION_SAMPLES} samples")
            if getattr(config, "train_max_samples", -1) > min(EVALUATION_OFFSETS.values()):
                raise ValueError("Training samples reach the reserved evaluation index range")
            modulus = self.vocab_size - 1
            period = modulus // gcd(SAMPLE_STRIDE, modulus)
            if self.sample_offset + self.size > period:
                raise ValueError("Vocabulary is too small for disjoint synthetic evaluation identities")

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        length = self.lengths[index % len(self.lengths)]
        positions = torch.arange(length, dtype=torch.long)
        input_ids = 1 + (positions * 17 + (index + self.sample_offset) * SAMPLE_STRIDE) % (self.vocab_size - 1)
        supervised = max(1, min(512, length // 4))
        loss_mask = (positions >= length - supervised).long()
        return {"input_ids": input_ids, "position_ids": positions, "loss_mask": loss_mask}
