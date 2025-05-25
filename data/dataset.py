from __future__ import annotations

import numpy as np

import torch

from datasets import load_dataset, interleave_datasets
from transformers import GPT2TokenizerFast


class StreamingDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        rank: int,
        world_size: int,
        paths: list[str],
        names: list[str] | None = None,
        splits: list[str] | None = None
    ):
        self.rank = rank
        self.world_size = world_size
        self.paths = paths
        self.names = names
        self.splits = splits
        
        dataset = interleave_datasets([
            load_dataset(path, name=name, split=split, streaming=True, trust_remote_code=True)
            for path, name, split in zip(paths, names, splits)
        ])
        self.shard = dataset.shard(num_shards=world_size, index=rank)

    def skip(self, offset: int) -> None:
        if offset: self.shard = self.shard.skip(offset)

    def __iter__(self):
        for item in self.shard:
            yield item


class BaseBatchCollator(object):
    def __init__(self, max_length: int = 1024, eos_idx: int = 50256, pad_idx: int = 50256):
        self.max_length = max_length
        self.eos_idx = eos_idx
        self.pad_idx = pad_idx
        
        self.tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')


class PretrainBatchCollator(BaseBatchCollator):
    def __call__(self, batch: list[dict[str, str]]) -> tuple[torch.LongTensor['B T'], torch.LongTensor['B']]:
        texts = [item['text'] + '<|endoftext|>' for item in batch]
        tokens = [self.tokenizer.encode(text, max_length=16384, truncation=True) for text in texts]
        B = len(tokens)
        lengths = torch.LongTensor(list(map(len, tokens))).clamp_max(self.max_length)  # [B]
        T = lengths.max()
        tokens_padded = torch.ones(B, T, dtype=torch.long) * self.pad_idx  # [B, T]
        for i, seq in enumerate(tokens):
            if len(seq) > T:
                max_offset = len(seq) - T
                offset = np.random.choice(max_offset)
                tokens_padded[i, :] = torch.LongTensor(seq[offset:offset+T])
            else:
                tokens_padded[i, :len(seq)] = torch.LongTensor(seq)
        return tokens_padded, lengths  # [B, T], [B]


class SFTBatchCollator(BaseBatchCollator):
    def __call__(self, batch: list[dict[str, str]]) -> tuple[torch.LongTensor['B T'], torch.LongTensor['B']]:
        texts = [item['instruction'] + item['response'] + '<|endoftext|>' for item in batch]
        tokens = [self.tokenizer.encode(text, max_length=16384, truncation=True) for text in texts]
        B = len(tokens)
        lengths = torch.LongTensor(list(map(len, tokens))).clamp_max(self.max_length)  # [B]
        T = lengths.max()
        tokens_padded = torch.ones(B, T, dtype=torch.long) * self.pad_idx  # [B, T]
        for i, seq in enumerate(tokens):
            tokens_padded[i, :T] = torch.LongTensor(seq[:T])
        return tokens_padded, lengths  # [B, T], [B]
