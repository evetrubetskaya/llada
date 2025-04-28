import numpy as np

import torch

from datasets import load_dataset
from transformers import AutoTokenizer


class StreamingDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        rank: int,
        world_size: int,
        path: str,
        name: str | None = None,
        split: str | None = None
    ):
        self.rank = rank
        self.world_size = world_size
        self.path = path
        self.name = name
        self.split = split
        
        dataset = load_dataset(path, name=name, split=split, streaming=True)
        self.shard = dataset.shard(num_shards=world_size, index=rank)

    def __iter__(self):
        for item in self.shard:
            yield item


class BaseBatchCollator(object):
    def __init__(self, max_length: int = 1024, pad_idx: int = 50256):
        self.max_length = max_length
        self.pad_idx = pad_idx
        
        self.tokenizer = AutoTokenizer.from_pretrained('gpt2')


class PretrainBatchCollator(BaseBatchCollator):
    def __call__(self, batch: list[dict[str, str]]):
        texts = [item['text'] + '<|endoftext|>' for item in batch]
        tokens = [self.tokenizer.encode(text) for text in texts]
        B = len(tokens)
        lenghts = torch.LongTensor(list(map(len, tokens)))  # [B]
        tokens_padded = torch.ones(B, self.max_length, dtype=torch.long) * self.pad_idx  # [B, T]
        for i, seq in enumerate(tokens):
            if len(seq) <= self.max_length:
                tokens_padded[i, :len(seq)] = torch.LongTensor(seq)
            else:
                max_start = len(seq) - self.max_length
                start = np.random.choice(max_start)
                tokens_padded[i, :] = torch.LongTensor(seq[start:start+self.max_length])
        return tokens_padded, lenghts  # [B, T], [B]


class SFTBatchCollator(BaseBatchCollator):
    def __call__(self, batch: list[dict[str, str]]):
        texts = [item['instruction'] + item['response'] + '<|endoftext|>' for item in batch]
        tokens = [self.tokenizer.encode(text) for text in texts]
        B = len(tokens)
        lenghts = torch.LongTensor(list(map(len, tokens)))  # [B]
        T = max(map(len, tokens))
        tokens_padded = torch.ones(B, T, dtype=torch.long) * self.pad_idx  # [B, T]
        for i, seq in enumerate(tokens):
            # @TODO: think about how to cope with max length
            tokens_padded[i, :len(seq)] = torch.LongTensor(seq)
        return tokens_padded, lenghts  # [B, T], [B]
