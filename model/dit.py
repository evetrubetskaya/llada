from __future__ import annotations

import math

import torch

from model.utils import next_multiple


def rms_norm(x: torch.FloatTensor['B T D']) -> torch.FloatTensor['B T D']:
    return torch.nn.functional.rms_norm(x, (x.shape[-1],))  # [B, T, D]


class CastedLinear(torch.nn.Linear):
    def __init__(self, in_features: int, out_features: int):
        super().__init__(in_features, out_features, bias=False)

    def reset_parameters(self) -> None:
        std = 0.5 * (self.in_features ** -0.5)  # 0.5 is a bit better than the default 1/sqrt(3) suggested in modded-nanogpt
        bound = (3 ** 0.5) * std
        with torch.no_grad():
            self.weight.uniform_(-bound, bound)

    def forward(self, x: torch.FloatTensor['... in_features']) -> torch.FloatTensor['... out_features']:
        return torch.nn.functional.linear(x, self.weight.type_as(x))  # [..., out_features]


class FFN(torch.nn.Module):
    def __init__(self, dim: int, dim_mult: float = 4.0):
        super().__init__()
        self.dim = dim
        self.dim_mult = dim_mult
        self.dim_hidden = dim_hidden = int(dim_mult * dim)
        
        self.fc1 = CastedLinear(dim, dim_hidden)
        self.fc2 = CastedLinear(dim_hidden, dim)
    
    def forward(self, x: torch.FloatTensor['B T D']) -> torch.FloatTensor['B T D']:
        x = self.fc1(x)  # [B, T, D] -> [B, T, dim_hidden]
        x = x.relu().square()  # [B, T, dim_hidden]
        x = self.fc2(x)  # [B, T, dim_hidden] -> [B, T, D]
        return x  # [B, T, D]


class RotaryEmbedding(torch.nn.Module):
    def __init__(self, d: int, max_length: int = 4096):
        super().__init__()
        self.d = d
        self.max_length = max_length
        
        # half-truncate RoPE (w/ base freq tuning) suggested by modded-nanogpt
        angular_freq = (1.0 / 1024.0) ** torch.linspace(0, 1, steps=d//4, dtype=torch.float32)
        angular_freq = torch.cat([angular_freq, angular_freq.new_zeros(d//4)])
        t = torch.arange(max_length, dtype=torch.float32)
        theta = torch.einsum('i,j -> ij', t, angular_freq)
        self.cos = torch.nn.Buffer(theta.cos(), persistent=False)
        self.sin = torch.nn.Buffer(theta.sin(), persistent=False)

    def forward(self, x: torch.FloatTensor['B T H d']) -> torch.FloatTensor['B T H d']:
        assert self.cos.shape[0] >= x.shape[1]
        dtype = x.dtype
        with torch.amp.autocast(x.device.type, enabled=False):
            cos, sin = self.cos[None, :x.shape[1], None, :], self.sin[None, :x.shape[1], None, :]
            x1, x2 = x.to(dtype=torch.float32).chunk(2, dim=-1)  # 2 x [B, T, H, d//2]
            y1 = x1 * cos + x2 * sin  # [B, T, H, d//2]
            y2 = x1 * (-sin) + x2 * cos  # [B, T, H, d//2]
            x = torch.cat((y1, y2), 3)  # [B, T, H, d]
        return x.to(dtype)  # [B, T, H, d]


class MHSA(torch.nn.Module):
    
    USE_FLASH_ATTN = True
    
    def __init__(self, dim: int, n_heads: int, scale: float | None = 0.12, max_length: int = 4096):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.dim_head = dim // n_heads
        self.scale = self.dim_head ** 0.5 if scale is None else scale  # default scale=0.12 is suggested by modded-nanogpt
        self.max_length = max_length
        
        std = 0.5 * (dim ** -0.5)
        bound = (3 ** 0.5) * std  # improved init scale by modded-nanogpt
        self.Wqkv = torch.nn.Parameter(torch.empty(3, dim, dim).uniform_(-bound, bound))
        self.lambdas = torch.nn.Parameter(torch.tensor([0.5, 0.5]))
        self.rope = RotaryEmbedding(self.dim_head, max_length=max_length)
        self.proj = CastedLinear(dim, dim)

    def forward(
        self,
        x: torch.FloatTensor['B T D'],
        ve: torch.FloatTensor['B T D'],
        mask: torch.BoolTensor['B T 1'],
        attn_mask: torch.BoolTensor['B 1 T T']
    ) -> torch.FloatTensor['B T D']:
        B, T, D = x.shape
        
        # compute Q, K, V projections
        QKV = torch.nn.functional.linear(x, self.Wqkv.flatten(end_dim=1).type_as(x))  # [B, T, 3D]
        Q, K, V = QKV.reshape(B, T, 3 * self.n_heads, self.dim_head).chunk(3, -2)  # 3 x [B, T, H, d]
        
        # apply norm to Q and K for better stability and apply rope
        Q = self.rope(rms_norm(Q))  # [B, T, H, d]
        K = self.rope(rms_norm(K))  # [B, T, H, d]
        
        if ve is None:
            # skip mid-layers token value embeddings
            V = self.lambdas[0] * V  # [B, T, H, d]
        else:
            V = self.lambdas[0] * V + self.lambdas[1] * ve.view_as(V).contiguous()  # [B, T, H, d]
        
        # compute attention
        mask_expanded = mask[:, None]  # [B, 1, T, 1]
        Q = torch.where(mask_expanded, Q.transpose(1, 2).contiguous(), 0.0)  # [B, H, T, d]
        K = torch.where(mask_expanded, K.transpose(1, 2).contiguous(), 0.0)  # [B, H, T, d]
        V = torch.where(mask_expanded, V.transpose(1, 2).contiguous(), 0.0)  # [B, H, T, d]
        if self.USE_FLASH_ATTN:
            attn_mask = (~attn_mask).to(Q.dtype) * torch.finfo(Q.dtype).min
            x = torch.nn.functional.scaled_dot_product_attention(Q, K, V, attn_mask=attn_mask, scale=0.12)  # [B, H, T, d]
        else:
            # fallback to naive implementation
            Q = Q / math.sqrt(self.scale)  # [B, H, T, d]
            KT = K.transpose(-2, -1).contiguous() / math.sqrt(self.scale)  # [B, H, T, d]
            attn_scores = torch.matmul(Q, KT)  # [B, H, T, d] @ [B, H, d, T] = [B, H, T, T]
            attn_scores = torch.masked_fill(attn_scores, ~attn_mask, torch.finfo(attn_scores.dtype).min)  # [B, H, T, T]
            attn_weights = torch.nn.functional.softmax(attn_scores, dim=-1)  # [B, H, T, T]
            x = torch.matmul(attn_weights, V)  # [B, H, T, d]
        
        # project back to output dimension
        x = x.transpose(2, 1).contiguous().reshape(B, T, D)  # [B, H, T, d] -> [B, T, H, d] -> [B, T, D]
        x = torch.where(mask, self.proj(x), 0.0)  # [B, T, D]
        
        return x  # [B, T, D]


class DiTBlock(torch.nn.Module):
    def __init__(self, dim: int, n_heads: int, dim_mult: float = 4.0, max_length: int = 4096):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.dim_mult = dim_mult
        self.max_length = max_length
        
        self.lambdas = torch.nn.Parameter(torch.tensor([1., 0.]))
        self.attn = MHSA(dim, n_heads, max_length=max_length)
        self.ffn = FFN(dim, dim_mult=dim_mult)
    
    def forward(
        self,
        x: torch.FloatTensor['B T D'],
        x0: torch.FloatTensor['B T D'],
        ve: torch.FloatTensor['B T D'] | None,
        mask: torch.BoolTensor['B T 1'],
        mod_params: torch.FloatTensor['B 1|T 6D'],
        attn_mask: torch.BoolTensor['B 1 T T']
    ) -> torch.FloatTensor['B T D']:
        # unpacking current layer modulation params
        std1, mean1, alpha1, std2, mean2, alpha2 = mod_params.chunk(6, -1)  # 6 x [B, 1|T, D]
        # mixing current signal x with original x0 suggested by modded-nanogpt
        x = self.lambdas[0] * x + self.lambdas[1] * x0  # [B, T, D]
        # apply self-attention
        x_attn = rms_norm(x) * (1.0 + std1) + mean1  # [B, T, D]
        x_attn = alpha1 * self.attn(x_attn, ve, mask, attn_mask)  # [B, T, D]
        x = x + x_attn  # [B, T, D]
        # apply ffn
        x_ffn = rms_norm(x) * (1.0 + std2) + mean2 # [B, T, D]
        x_ffn = torch.where(mask, alpha2 * self.ffn(x_ffn), 0.0)  # [B, T, D]
        x = x + x_ffn  # [B, T, D]
        return x  # [B, T, D]


def _abs_sincos_timestep_embedder(
    timesteps: torch.FloatTensor['B T'],
    dim: int = 768,
    scale: float = 1000.0
) -> torch.FloatTensor['B T D']:
    dtype = timesteps.dtype
    timesteps = timesteps.float()  # [B, T]
    with torch.amp.autocast(timesteps.device.type, enabled=False):
        half_dim = dim // 2
        t = 9.210340371976184 / (half_dim - 1)
        dims_half = torch.arange(half_dim, dtype=timesteps.dtype, device=timesteps.device)  # [D//2]
        t = (dims_half * -t).exp()  # [D//2]
        t = scale * timesteps.unsqueeze(-1) * t.unsqueeze(0).unsqueeze(0)  # [B, T, D//2]
        t = torch.cat([t.sin(), t.cos()], dim=-1)  # [B, T, D]
    return t.to(dtype)  # [B, T, D]


class TimestepEmbedder(torch.nn.Module):
    def __init__(self, dim: int, scale: float = 1000.0):
        super().__init__()
        self.dim = dim
        self.scale = scale
        
        self.fc1 = CastedLinear(dim, 4*dim)
        self.fc2 = CastedLinear(4*dim, dim)
    
    def forward(self, timesteps: torch.FloatTensor['B T']) -> torch.FloatTensor['B T D']:
        t = _abs_sincos_timestep_embedder(timesteps, self.dim, scale=self.scale)  # [B, T, D]
        t = self.fc1(t)  # [B, T, 4D]
        t = torch.nn.functional.silu(t)  # [B, T, 4D]
        t = self.fc2(t)  # [B, T, D]
        return torch.nn.functional.silu(t)  # [B, T, D]


class DiTHead(torch.nn.Module):
    def __init__(self, dim: int, vocab_size: int):
        super().__init__()
        self.dim = dim
        self.vocab_size = vocab_size
        
        self.t_proj = CastedLinear(dim, 2*dim)
        self.linear = CastedLinear(dim, vocab_size)
    
    def forward(self, x: torch.FloatTensor['B T D'], t: torch.FloatTensor['B 1|T D']) -> torch.FloatTensor['B T vocab_size_pad128']:
        mean, scale = self.t_proj(t).chunk(2, -1)  # 2 x [B, 1|T D]
        x = rms_norm(x)  # [B, T, D]
        x = x * (scale + 1.0) + mean  # [B, T, D]
        logits = self.linear(x).float()  # [B, T, vocab_size_pad128]
        # tanh softcapping suggested by modded-nanogpt from gemma 2 paper
        logits = 30 * torch.sigmoid(logits / (7.5 * x.size(-1)**0.5))  # [B, T, vocab_size_pad128]
        return logits  # [B, T, vocab_size_pad128]


class DiT(torch.nn.Module):
    def __init__(self, dim: int, n_heads: int, dim_mult: float, n_layers: int, vocab_size: int = 50257, max_length: int = 4096):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.dim_mult = dim_mult
        self.n_layers = n_layers
        # following common practice we pad vocabulary size to the multiple of 128 to improve the speed
        self.vocab_size = vocab_size
        self.max_length = max_length
        
        self.v_embedders = torch.nn.ModuleList([torch.nn.Embedding(vocab_size, dim) for _ in range(3)])
        self.x_embedder = torch.nn.Embedding(vocab_size, dim)
        self.t_embedder = TimestepEmbedder(dim)
        # instead of projecting modulation params at each layer,
        # we will do all-layers projection at once in advance
        self.t_proj = CastedLinear(dim, n_layers * 6 * dim)
        self.layers = torch.nn.ModuleList([
            DiTBlock(dim, n_heads, dim_mult, max_length=max_length)
            for _ in range(n_layers)
        ])
        self.head = DiTHead(dim, next_multiple(vocab_size, divisor=128))
        # tying weights of embedder and head to improve memory usage
        self.x_embedder.weight = self.head.linear.weight
    
    def forward(
        self,
        xt: torch.LongTensor['B T'],
        t: torch.FloatTensor['B 1|T'],
        mask: torch.BoolTensor['B T 1'],
        attn_mask: torch.BoolTensor['B 1 T T']
    ) -> torch.FloatTensor['B T vocab_size_pad128']:
        ves = [v_embedder(xt) for v_embedder in self.v_embedders]
        # 012 ... 012 structure on token value embeddings suggested by modded-nanogpt
        ves = [ves[0], ves[1], ves[2]] + [None] * (self.n_layers - 6) + [ves[0], ves[1], ves[2]]  # list[n_layers] x [B, T, D]
        
        x = x0 = rms_norm(self.x_embedder(xt))  # [B, T, D]
        
        t = self.t_embedder(t)  # [B, 1|T, D]
        mod_params = self.t_proj(t).chunk(self.n_layers, -1)  # n_layers x [B, 1|T, 6 * D]
        
        for i, layer in enumerate(self.layers):
            x = layer(x, x0, ves[i], mask, mod_params[i], attn_mask)  # [B, T, D]

        logits = self.head(x, t)  # [B, T, vocab_size_pad128]
        return logits  # [B, T, vocab_size_pad128]
