from dataclasses import dataclass


@dataclass
class SmallConfig:
    dim: int = 768
    n_heads: int = 6
    dim_mult: float = 4.0
    n_layers: int = 12
    vocab_size: int = 50257
    # eos and pad tokens are the same
    eos_idx: int = 50256
    pad_idx: int = 50256


@dataclass
class MediumConfig:
    dim: int = 1024
    n_heads: int = 16
    dim_mult: float = 4.0
    n_layers: int = 24
    vocab_size: int = 50257
    # eos and pad tokens are the same
    eos_idx: int = 50256
    pad_idx: int = 50256


@dataclass
class LargeConfig:
    dim: int = 1536
    n_heads: int = 16
    dim_mult: float = 4.0
    n_layers: int = 24
    vocab_size: int = 50257
    # eos and pad tokens are the same
    eos_idx: int = 50256
    pad_idx: int = 50256
