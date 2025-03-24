from __future__ import annotations

import torch


def get_mask_from_lengths(lengths: torch.LongTensor['B']) -> torch.BoolTensor['B T']:
    T = lengths.max()
    pos = torch.arange(T, dtype=torch.long, device=lengths.device)  # [T]
    mask = pos[None, :] < lengths[:, None]  # [1, T] < [B, 1] = [B, T]
    return mask  # [B, T]
