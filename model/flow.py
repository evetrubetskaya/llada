from __future__ import annotations

from tqdm import tqdm

import torch

from model.utils import get_mask_from_lengths


SOLVERS = [
    'euler',
    'ddim'
]


class CategoricalFlowMatching(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, vocab_size: int):
        super().__init__()
        self.model = model
        self.vocab_size = vocab_size
    
    def forward(
        self,
        x_1: torch.FloatTensor['B T'],
        lengths: torch.LongTensor['B']
    ) -> tuple[torch.FloatTensor, torch.FloatTensor]:
        B = x_1.shape[0]
        T = x_1.shape[1]
        
        # sample random timesteps on trajectories
        t = torch.rand(B, dtype=torch.float32, device=x_1.device)  # [B]
        t_bc = t[:, None, None]  # [B, 1, 1], broadcasted
        
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
        mask = get_mask_from_lengths(lengths)  # [B, T]
        
        # predict original sequence
        logits = self.model(x_t, t, mask)  # [B, T, vocab_size]
        
        # compute loss
        mask_flat = mask.reshape(B*T).contiguous()  # [B*T]
        logits_flat = logits.reshape(B*T, -1).contiguous()  # [B*T, vocab_size]
        logits_nnz = logits_flat[mask_flat]  # [nnz, vocab_size]
        x_1_nnz = x_1.reshape(B*T).contiguous()[mask_flat]  # [nnz]
        loss = torch.nn.functional.cross_entropy(logits_nnz, x_1_nnz)  # (logits=[B, num_classes], labels=[B])
        
        # we also track accuracy
        with torch.no_grad():
            preds = logits_nnz.argmax(-1)  # [nnz]
            accuracy = (preds == x_1_nnz).float().mean()
        return loss, accuracy

    @torch.no_grad()
    def sample(
        self,
        lengths: torch.LongTensor['B'],
        timesteps: int = 10,
        solver: str = 'euler',
        verbose: bool = True
    ) -> tuple[
        list[torch.LongTensor['Ti']],
        list[torch.LongTensor['B T']]
    ]:
        B = lengths.shape[0]
        T = lengths.max()
        mask = get_mask_from_lengths(lengths)  # [B, T]
        
        device = next(self.parameters()).device
        
        # prepare solver timesteps grid
        h = 1 / timesteps
        grid = torch.linspace(0, 1 - h, steps=timesteps, device=device)  # [timesteps]
        
        # we start from uniform cat dist
        p_0 = torch.full(
            (B, T, self.vocab_size),
            fill_value=1/self.vocab_size,
            dtype=torch.float32,
            device=device
        )  # [B, T, vocab_size]
        p_t = p_0.clone()  # [B, T, vocab_size]
        
        # run solver
        states = []
        for i, t in enumerate((tqdm(grid) if verbose else grid)):
            t = t.repeat(B)  # [B]
            x_t = torch.distributions.Categorical(p_t).sample()  # [B, T]
            states.append(x_t.clone())
            # at each step we estimate how p_1 looks like
            logits = self.model(x_t, t, mask)  # [B, T, vocab_size]
            p_1 = logits.softmax(-1)  # [B, T, vocab_size]
            if i < grid.shape[0] - 1:
                t_bc = t[:, None, None]  # [B, 1, 1]
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
        
        seqs = [x_1[i, :lengths[i]] for i in range(B)]  # list x [Ti]
        
        return seqs, states  # list x [Ti], list x [B, T]
