from __future__ import annotations

import torch


USE_FLASH_ATTN = True


class FFN(torch.nn.Module):
    def __init__(self, dim: int, dim_mult: float = 4.0):
        super().__init__()
        self.dim = dim
        self.dim_mult = dim_mult
        self.dim_hidden = dim_hidden = int(dim_mult * dim)
        
        self.fc1 = torch.nn.Linear(dim, dim_hidden, bias=False)
        self.activation = torch.nn.GELU()
        self.fc2 = torch.nn.Linear(dim_hidden, dim, bias=False)
    
    def forward(self, x: torch.FloatTensor['B T D']) -> torch.FloatTensor['B T D']:
        x = self.fc1(x)  # [B, T, D] -> [B, T, dim_hidden]
        x = self.activation(x)  # [B, T, dim_hidden]
        x = self.fc2(x)  # [B, T, dim_hidden] -> [B, T, D]
        return x


def rotate_half(x: torch.FloatTensor['B n_heads T dim_head']) -> torch.FloatTensor['B n_heads T dim_head']:
    x = x.clone()  # [B, n_heads, T, dim_head]
    x1, x2 = x.chunk(2, -1)  # [B, n_heads, T, dim_head//2]
    out = torch.cat([-x2.clone(), x1.clone()], dim=-1)  # [B, n_heads, T, dim_head]
    return out  # [B, n_heads, T, dim_head]


class RotaryEmbedding(torch.nn.Module):
    def __init__(self, dim_head: int, base: int = 10000):
        super().__init__()
        self.dim_head = dim_head
        self.base = base

    def forward(
        self,
        x: torch.FloatTensor['B n_heads T dim_head'],
        pos: torch.LongTensor['B T']
    ) -> torch.FloatTensor['B n_heads T dim_head']:
        dtype = x.dtype
        device = x.device
        x = x.float()  # [B, n_heads, T, dim_head]
        with torch.amp.autocast(device.type, enabled=False):
            dims_half = torch.arange(0, self.dim_head, 2, dtype=torch.float32, device=device)  # [dim_head//2]
            inv_freq = 1.0 / (self.base ** (dims_half / self.dim_head))  # [dim_head//2]
            freqs = torch.einsum("bi,j->bij", pos, inv_freq)  # [B, T, dim_head//2]
            embds = torch.cat([freqs, freqs], dim=-1)  # [B, T, dim_head]
            cos = embds.cos().unsqueeze(1)  # [B, 1, T, dim_head]
            sin = embds.sin().unsqueeze(1)  # [B, 1, T, dim_head]
            x_rotated = rotate_half(x)  # [B, n_heads, T, dim_head]
            x_embd = (x * cos) + (x_rotated * sin)  # [B, n_heads, T, dim_head]
        x_embd = x_embd.to(dtype)  # [B, n_heads, T, dim_head]
        return x_embd  # [B, n_heads, T, dim_head]


class MHSA(torch.nn.Module):
    def __init__(self, dim: int, n_heads: int):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.dim_head = dim // n_heads
        self.scale = self.dim_head ** 0.5
        
        self.W_qkv = torch.nn.Linear(dim, 3*dim, bias=False)
        self.norm_q = torch.nn.LayerNorm(self.dim_head)
        self.norm_k = torch.nn.LayerNorm(self.dim_head)
        self.W_o = torch.nn.Linear(dim, dim, bias=False)
        self.rope = RotaryEmbedding(self.dim_head)

    def forward(
        self,
        x: torch.FloatTensor['B T D'],
        attn_mask: torch.BoolTensor['B 1 T T'],
        pos: torch.LongTensor['B T']
    ) -> torch.FloatTensor['B T D']:
        B, T, D = x.shape
        
        # compute Q, K, V projections
        QKV = self.W_qkv(x)  # [B, T, 3D]
        Q, K, V = QKV.chunk(3, -1)  # 3 x [B, T, D]
        
        # split into multiple heads
        Q = Q.reshape(B, T, self.n_heads, self.dim_head).transpose(1, 2).contiguous()  # [B, T, D] -> [B, T, H, d] -> [B, H, T, d]
        K = K.reshape(B, T, self.n_heads, self.dim_head).transpose(1, 2).contiguous()  # [B, H, T, d]
        V = V.reshape(B, T, self.n_heads, self.dim_head).transpose(1, 2).contiguous()  # [B, H, T, d]
        
        # apply layer norm to Q and K for better stability
        Q = self.norm_q(Q)  # [B, H, T, d]
        K = self.norm_k(K)  # [B, H, T, d]
        
        # apply rope
        Q = self.rope(Q, pos)  # [B, H, T, d]
        K = self.rope(K, pos)  # [B, H, T, d]
        
        # compute attention
        if USE_FLASH_ATTN:
            dtype = Q.dtype
            Q = Q.to(torch.float16)
            K = K.to(torch.float16)
            V = V.to(torch.float16)
            attn_mask = (~attn_mask).to(torch.float16) * torch.finfo(torch.float16).min
            attn_output = torch.nn.functional.scaled_dot_product_attention(Q, K, V, attn_mask=attn_mask)  # [B, H, T, d]
            attn_output = attn_output.to(dtype)
        else:
            # fallback to naive implementation
            Q = Q / self.scale  # [B, H, T, d]
            KT = K.transpose(-2, -1).contiguous() / self.scale  # [B, H, T, d]
            attn_scores = torch.matmul(Q, KT)  # [B, H, T, d] @ [B, H, d, T] = [B, H, T, T]
            attn_scores = torch.masked_fill(attn_scores, ~attn_mask, torch.finfo(attn_scores.dtype).min)  # [B, H, T, T]
            attn_weights = torch.nn.functional.softmax(attn_scores, dim=-1)  # [B, H, T, T]
            attn_output = torch.matmul(attn_weights, V)  # [B, H, T, d]
        
        # project back to output dimension
        output = attn_output.transpose(2, 1).contiguous().reshape(B, T, D)  # [B, H, T, d] -> [B, T, H, d] -> [B, T, D]
        output = self.W_o(output)  # [B, T, D]
        return output  # [B, T, D]


class TransformerLayer(torch.nn.Module):
    def __init__(self, dim: int, n_heads: int, dim_mult: float = 4.0):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.dim_mult = dim_mult
        
        self.norm_1 = torch.nn.LayerNorm(dim)
        self.attn = MHSA(dim, n_heads)
        self.norm_2 = torch.nn.LayerNorm(dim)
        self.ffn = FFN(dim, dim_mult=dim_mult)
    
    def forward(
        self,
        x: torch.FloatTensor['B T D'],
        mod_params: torch.FloatTensor['B 1|T 6D'],
        attn_mask: torch.BoolTensor['B 1 T T'],
        pos: torch.LongTensor['B T']
    ) -> torch.FloatTensor['B T D']:
        # unpacking current layer modulation params
        std_1, mean_1, alpha_1, std_2, mean_2, alpha_2 = mod_params.chunk(6, -1)  # 6 x [B, 1|T, D]
        
        # applying self-attention
        x_attn = self.norm_1(x)  # [B, T, D]
        x_attn = (x_attn * std_1 + mean_1)  # [B, T, D]
        x_attn = self.attn(x_attn, attn_mask, pos)  # [B, T, D]
        x_attn = x_attn * alpha_1  # [B, T, D]
        x = x + x_attn  # [B, T, D]
        
        # applying ffn
        x_ffn = self.norm_2(x)  # [B, T, D]
        x_ffn = (x_ffn * std_2 + mean_2) # [B, T, D]
        x_ffn = self.ffn(x_ffn)  # [B, T, D]
        x_ffn = x_ffn * alpha_2  # [B, T, D]
        x = x + x_ffn  # [B, T, D]
        
        return x


class DiT(torch.nn.Module):
    def __init__(self, dim: int, n_heads: int, dim_mult: float, n_layers: int, vocab_size: int):
       super().__init__()
       self.dim = dim
       self.n_heads = n_heads
       self.dim_mult = dim_mult
       self.n_layers = n_layers
       self.vocab_size = vocab_size
       
       self.embedder = torch.nn.Embedding(vocab_size, dim)
       self.t_embedder = torch.nn.Linear(1, dim, bias=False)
       # instead of projecting modulation params at each layer,
       # we will do all-layers projection at once in advance
       self.t_proj = torch.nn.Linear(dim, n_layers * 6 * dim, bias=False)
       self.layers = torch.nn.ModuleList([
           TransformerLayer(dim, n_heads, dim_mult)
           for _ in range(n_layers)
       ])
       self.norm = torch.nn.LayerNorm(dim)
       self.head = torch.nn.Linear(dim, vocab_size, bias=False)
    
    def forward(
        self,
        x_t: torch.LongTensor['B T'],
        t: torch.FloatTensor['B 1|T'],
        attn_mask: torch.BoolTensor['B 1 T T']
    ) -> torch.FloatTensor['B T vocab_size']:
        x = self.embedder(x_t)  # [B, T, D]
        t = t.unsqueeze(-1)  # [B, 1|T, 1]
        t = self.t_embedder(t)  # [B, 1|T, D]
        t = self.t_proj(t)  # [B, 1|T, n_layers * 6 * D]
        mod_params = t.chunk(self.n_layers, -1)  # n_layers x [B, 1|T, 6 * D]
        
        pos = torch.arange(x.shape[1], device=x.device)  # [T]
        pos = pos[None, :]  # [1, T]
        pos = pos.repeat(x.shape[0], 1)  # [B, T]
        for i, layer in enumerate(self.layers):
            x = layer(x, mod_params[i], attn_mask, pos)  # [B, T, D]
        
        x = self.norm(x)  # [B, T, D]
        logits = self.head(x)  # [B, T, vocab_size]
        return logits  # [B, T, vocab_size]
