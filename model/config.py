from dataclasses import dataclass


@dataclass
class LLaDAConfig:
    dim: int = 768
    n_heads: int = 6
    dim_mult: float = 4.0
    n_layers: int = 12
    vocab_size: int = 50257
    eos_idx: int = 50256
    pad_idx: int = 50256
