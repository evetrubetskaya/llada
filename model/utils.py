from __future__ import annotations

import torch


def get_mask_from_lengths(lengths: torch.LongTensor['B'], T: int | None = None) -> torch.BoolTensor['B T']:
    T = lengths.max() if T is None else T
    pos = torch.arange(T, dtype=torch.long, device=lengths.device)  # [T]
    mask = pos[None, :] < lengths[:, None]  # [1, T] < [B, 1] = [B, T]
    return mask  # [B, T]


def next_multiple(x: int, divisor: int = 128) -> int:
    remainder = x % divisor
    pad = (divisor - remainder) if remainder > 0 else 0
    return x + pad
