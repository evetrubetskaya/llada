from __future__ import annotations

from tqdm import tqdm

import torch

from model.utils import get_mask_from_lengths, next_multiple


_SEQLEN_MULTIPLE = 128


def topk_filtering(logits: torch.FloatTensor['B T vocab_size'], topk: int = 50) -> torch.FloatTensor['B T vocab_size']:
    _, topk_indices = torch.topk(logits, topk, dim=-1)  # [B, T, top_k]
    topk_mask = torch.zeros_like(logits, dtype=torch.bool)  # [B, T, vocab_size]
    topk_mask.scatter_(-1, topk_indices, True)
    logits = torch.where(topk_mask, logits, torch.tensor(float('-inf'), device=logits.device, dtype=logits.dtype))  # [B, T, vocab_size]
    return logits  # [B, T, vocab_size]


class CategoricalFlowMatching(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, vocab_size: int = 50257, eos_idx: int = 50256):
        super().__init__()
        self.model = model
        self.vocab_size = vocab_size
        self.vocab_size_pad128 = next_multiple(vocab_size, divisor=128)
        self.eos_idx = eos_idx
    
    def compile(self, mode: str, fullgraph: bool = True, dynamic: bool = True) -> None:
        if self.training:
            self._forward_impl = torch.compile(self._forward_impl, mode=mode, fullgraph=fullgraph, dynamic=dynamic)
        else:
            self.model = torch.compile(self.model, mode=mode, fullgraph=fullgraph, dynamic=dynamic)
    
    def kt(self, t: torch.FloatTensor | float) -> torch.FloatTensor | float:
        # best schedule according to https://arxiv.org/pdf/2407.15595
        return 2 * t**2
    
    def dkt(self, t: torch.FloatTensor | float) -> torch.FloatTensor | float:
        # kt derivative w.r.t. t
        return 4 * t

    def _forward_impl(self, x1: torch.LongTensor['B T']) -> tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        dtype = next(self.parameters()).dtype
        device = next(self.parameters()).device
        
        B, T = x1.shape[:2]
        
        # sample random timesteps on trajectories
        t = torch.rand(B, 1, dtype=dtype, device=device)  # [B, 1]
        kt = self.kt(t)  # [B, 1]
        # get an unconditional couping of (x0, x1)
        x0 = torch.randint_like(x1, self.vocab_size)  # [B, T]
        # interpolate between x0 and x1
        mask_use_x0 = torch.rand(B, T, device=device) > kt  # [B, T]
        xt = torch.where(mask_use_x0, x0, x1)  # [B, T]
        
        # for variable-length sequences, use an all-true padding mask so the model can attend to every token
        # as well as predict eos tokens; this allows the model to control sequence length
        # by predicting eos tokens as needed; this was also used in the llada paper https://arxiv.org/pdf/2502.09992
        mask = torch.ones(B, T, 1, dtype=torch.bool, device=device)  # [B, T, 1]
        attn_mask = torch.ones(B, 1, T, T, dtype=torch.bool, device=device)  # [B, 1, T, T]
        # predict posterior p(x1|xt) with the model
        logits = self.model(xt, kt, mask, attn_mask)  # [B, T, vocab_size_pad128]
        
        # compute x1 cross-entropy loss
        logits_flat = logits.reshape(-1, self.vocab_size_pad128)  # [B*T, vocab_size_pad128]
        x1_flat = x1.reshape(-1)  # [B*T]
        loss = torch.nn.functional.cross_entropy(logits_flat, x1_flat)
        
        # we also track accuracy and perplexity at timestep t
        with torch.no_grad():
            preds = logits.argmax(-1)  # [B, T]
            preds_flat = preds.reshape(-1)  # [B*T]
            mask_eos = x1_flat == self.eos_idx  # [B*T]
            accuracy = ((preds_flat == x1_flat).to(dtype) * mask_eos / mask_eos.sum()).sum()
            perplexity = loss.exp()
        return loss, accuracy, perplexity
    
    def forward(self, x1: torch.LongTensor['B T']) -> tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        # padding x1 length to the next multiple of _SEQLEN_MULTIPLE
        T = x1.shape[-1]
        if T % _SEQLEN_MULTIPLE != 0:
            T_padded = next_multiple(T, divisor=_SEQLEN_MULTIPLE)
            x1 = torch.nn.functional.pad(x1, (0, T_padded - T), value=self.eos_idx)
        # running forward pass
        loss, accuracy, perplexity = self._forward_impl(x1)
        return loss, accuracy, perplexity
    
    @torch.no_grad()
    def sample(
        self,
        B: int,
        T: int = 1024,
        prompts: list[torch.LongTensor['Pi']] | None = None,
        timesteps: int = 256,
        topk: int | None = None,
        temperature: float = 1.0,
        verbose: bool = True,
        parse_outputs: bool = True
    ) -> tuple[
        list[torch.LongTensor['Ti']],
        list[torch.LongTensor['B T']]
    ]:
        device = next(self.parameters()).device
        
        # prepare solver timesteps grid
        grid = torch.cat([
            torch.linspace(0.0, 1.0-1.0/timesteps, steps=timesteps, dtype=torch.float32, device=device),
            torch.ones(1, dtype=torch.float32, device=device)
        ])  # [timesteps + 1]
        
        # prepare terminal state and insert prompts in the beginning if any provided
        xt = torch.randint(0, self.vocab_size, size=(B, T), dtype=torch.long, device=device)  # [B, T]
        xprompt = torch.ones_like(xt) * self.eos_idx  # [B, T]
        if prompts is not None:
            prompt_lengths = torch.LongTensor(list(map(lambda x: x.shape[-1], prompts))).to(device)  # [B]
            # insert prompts into xt and initialize prompts mask to not let the model change prompt tokens
            for i in range(B):
                xprompt[i, :prompts[i].shape[-1]] = prompts[i]  # [Pi]
            mask_prompt = get_mask_from_lengths(prompt_lengths, T=T)  # [B, T]
            xt = mask_prompt * xprompt + (~mask_prompt) * xt  # [B, T]
        else:
            mask_prompt = torch.zeros(B, T, dtype=torch.bool, device=device)  # [B, T]
        
        # running sampling
        mask = torch.ones(B, T, 1, dtype=torch.bool, device=device)  # [B, T, 1], all-trues for now
        attn_mask = torch.ones(B, 1, T, T, dtype=torch.bool, device=device)  # [B, 1, T, T], all-trues for now
        states = [xt.clone()]
        if verbose: pb = tqdm(range(timesteps))
        for i, t in enumerate(grid):
            if i == grid.shape[-1] - 1:
                continue
            else:
                # compute current step size
                h = grid[i+1] - grid[i]
                kt = self.kt(t)
                dkt = self.dkt(t)
                h_adaptive = torch.min(h, (1 - kt) / dkt)
                
                # obtain posterior p(x1|xt) estimate with the model
                logits = self.model(xt, kt * torch.ones(B, 1, device=device), mask, attn_mask)  # [B, T, vocab_size_pad128]
                logits = logits[..., :self.vocab_size].float()  # [B, T, vocab_size]
                logits = logits / temperature  # [B, T, vocab_size]
                logits = topk_filtering(logits) if topk is not None else logits  # [B, T, vocab_size]
                p1t = logits.softmax(-1)  # [B, T, vocab_size]
                
                # obtain velocity ut from posterior p(x1|xt) estimate
                ut = dkt / (1.0 - kt) * p1t  # [B, T, vocab_size]
                dirac_xt = torch.nn.functional.one_hot(xt, num_classes=self.vocab_size)  # [B, T, vocab_size]
                pt = torch.where(dirac_xt.to(torch.bool), torch.zeros_like(ut), ut)  # [B, T, vocab_size]
                
                # sample next state xt ~ pt
                intensity = pt.sum(dim=-1)  # [B, T], asssuming ut(xt|xt,x1) := 0s
                mask_jump = torch.rand((B, T), device=device) < (1.0 - (-h_adaptive * intensity).exp())  # [B, T]
                if mask_jump.sum() > 0:
                    xt[mask_jump] = torch.multinomial(pt[mask_jump] / pt[mask_jump].sum(-1, keepdim=True), 1, replacement=True).squeeze(-1)  # [nnz]
                # preserve prompt in updated xt
                xt = torch.where(mask_prompt, xprompt, xt)  # [B, T]
                states.append(xt.clone())
                
                if verbose: pb.update()
        
        if verbose: pb.close()
        
        x1 = xt.clone()  # [B, T]
        if parse_outputs:
            x1 = x1.cpu().tolist()  # [B, T]
            seqs = []
            for i in range(B):
                try:
                    j = x1[i].index(self.eos_idx)
                    seqs.append(x1[i][:j])
                except ValueError:
                    seqs.append(x1[i])
            return seqs, states  # list x [Ti], list x [B, T]
        return x1, states  # [B, T], list x [B, T]
