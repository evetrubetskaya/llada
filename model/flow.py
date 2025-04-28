from __future__ import annotations

from tqdm import tqdm

import torch


_MAX_LENGTH = 1024

SOLVERS = [
    'euler',
    'ddim'
]


class CategoricalFlowMatching(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, vocab_size: int, eos_idx: int = 50256):
        super().__init__()
        self.model = model
        self.vocab_size = vocab_size
        self.eos_idx = eos_idx
    
    def forward(self, x_1: torch.FloatTensor['B T']) -> tuple[torch.FloatTensor, torch.FloatTensor]:
        B = x_1.shape[0]
        T = x_1.shape[1]
        
        # sample random timesteps on trajectories
        t = torch.rand(B, 1, dtype=torch.float32, device=x_1.device)  # [B, 1]
        t_bc = t[:, None]  # [B, 1, 1], broadcasted
        
        # interpolate between categorical dists
        p_1 = torch.nn.functional.one_hot(x_1, num_classes=self.vocab_size)  # [B, T, vocab_size]
        p_0 = torch.full(
            (B, T, self.vocab_size),
            fill_value=1/self.vocab_size,
            dtype=torch.float32,
            device=x_1.device
        )  # [B, T, vocab_size]
        p_t = t_bc * p_1 + (1 - t_bc) * p_0  # [B, T, vocab_size]
        
        # sample x_t from interpolated cat dist
        x_t = torch.distributions.Categorical(p_t).sample()  # [B, T]
        attn_mask = torch.ones(B, 1, T, T, dtype=torch.bool, device=x_1.device)  # [B, 1, T, T]
        
        # predict original sequence
        logits = self.model(x_t, t, attn_mask)  # [B, T, vocab_size]
        
        # compute loss
        logits_flat = logits.reshape(B*T, -1).contiguous()  # [B*T, vocab_size]
        x_1_flat = x_1.reshape(B*T)  # [B*T]
        loss = torch.nn.functional.cross_entropy(logits_flat, x_1_flat)  # (logits=[B, num_classes], labels=[B])
        
        # we also track accuracy
        with torch.no_grad():
            preds = logits_flat.argmax(-1)  # [nnz]
            accuracy = (preds == x_1_flat).float().mean()
        return loss, accuracy
    
    @torch.no_grad()
    def sample(
        self,
        B: int,
        timesteps: int = 10,
        solver: str = 'euler',
        verbose: bool = True
    ) -> tuple[
        list[torch.LongTensor['Ti']],
        list[torch.LongTensor['B T']]
    ]:
        device = next(self.parameters()).device
        
        # prepare solver timesteps grid
        h = 1 / timesteps
        grid = torch.linspace(0, 1 - h, steps=timesteps, device=device)  # [timesteps]
        
        # we start from uniform cat dist
        p_0 = torch.full(
            (B, _MAX_LENGTH, self.vocab_size),
            fill_value=1/self.vocab_size,
            dtype=torch.float32,
            device=device
        )  # [B, T, vocab_size]
        p_t = p_0.clone()  # [B, T, vocab_size]
        
        # run solver
        attn_mask = torch.ones(B, 1, _MAX_LENGTH, _MAX_LENGTH, dtype=torch.bool, device=device)  # [B, 1, T, T]
        states = []
        for i, t in enumerate((tqdm(grid) if verbose else grid)):
            t = t.repeat(B)[:, None]  # [B, 1]
            x_t = torch.distributions.Categorical(p_t).sample()  # [B, T]
            states.append(x_t.clone())
            # at each step we estimate how p_1 looks like
            logits = self.model(x_t, t, attn_mask)  # [B, T, vocab_size]
            p_1 = logits.softmax(-1)  # [B, T, vocab_size]
            if i < grid.shape[0] - 1:
                t_bc = t[:, None]  # [B, 1, 1]
                # now we obtain next distribution by doing a solver step
                match solver:
                    case 'euler':
                        u_t = (p_1 - p_t) / (1.0 - t_bc)  # [B, T, vocab_size]
                        p_t = p_t + h * u_t  # [B, T, vocab_size]
                    case 'ddim':
                        p_t = (t_bc + h) * p_1 + (1 - (t_bc + h)) * p_0  # [B, T, vocab_size]
        
        # sampling final sequence
        x_1 = torch.distributions.Categorical(p_1).sample()  # [B, T]
        states.append(x_1.clone())
        
        x_1 = x_1.cpu().tolist()  # [B, T]
        seqs = []
        for i in range(B):
            try:
                j = x_1[i].index(self.eos_idx)
                seqs.append(x_1[i][:j])
            except ValueError:
                seqs.append(x_1[i])
        
        return seqs, states  # list x [Ti], list x [B, T]
