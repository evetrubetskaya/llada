from __future__ import annotations

import json
from string import punctuation
from typing import Any

from tokenizers import Tokenizer
from tokenizers.trainers import BpeTrainer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Split


ALPHABET = list('qwertyuiopasdfghjklzxcvbnmQWERTYUIOPASDFGHJKLZXCVBNM1234567890\n\t' + punctuation)

BPE_MAX_VOCAB_SIZE = 2**14  # 16384


class BPETokenizer(object):
    """Byte-pair encoding (BPE) level tokenizer."""
    
    def __init__(self, sep: str = ' ', backend: Tokenizer | None = None, **kwargs):
        self.sep = sep
        
        if backend is None:
            self.backend = Tokenizer(BPE())
            self.backend.pre_tokenizer = Split(self.sep, behavior='merged_with_next')
            self.is_trained = True
        else:
            self.backend = backend
            self.is_trained = False
        self.vocab = self.backend.get_vocab()
    
    @classmethod
    def from_checkpt(cls, path: str, **kwargs) -> BPETokenizer:
        with open(path, 'r', encoding='utf-8') as f:
            state_dict = json.load(f)
        sep = state_dict['sep']
        backend = Tokenizer.from_str(state_dict['backend_str'])
        return cls(sep=sep, backend=backend)
    
    def state_dict(self) -> dict[str, Any]:
        return {
            'sep': self.sep,
            'backend_str': self.backend.to_str(pretty=True)
        }
    
    def save(self, path: str) -> None:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.state_dict(), f, ensure_ascii=False, indent=4)
    
    def size(self) -> int:
        """Size of tokenizer vocabulary."""
        return len(self.vocab)
    
    def train(
        self,
        texts: iter,
        max_vocab_size: int = BPE_MAX_VOCAB_SIZE,
        min_frequency: int = 5,
        max_token_length: int = 8
    ) -> None:
        """Computes and sets vocab to use for future tokenization."""
        
        trainer = BpeTrainer(
            vocab_size=max_vocab_size,
            min_frequency=min_frequency,
            initial_alphabet=ALPHABET,
            max_token_length=max_token_length
        )
        self.backend.train_from_iterator(texts, trainer=trainer)
        
        self.vocab = self.backend.get_vocab()
        self.is_trained = True
    
    def encode(self, s: str) -> dict[str, Any]:
        """Encodes given string into tokens. Returns dict with tokens and their vocab indices."""
        
        assert self.is_trained, 'Tokenizer is not trained. Before encoding run `tokenizer.train(...)` on your texts.'
        
        x = self.backend.encode(s)
        indices, tokens = x.ids, x.tokens
        
        return dict(tokens=tokens, indices=indices)
    
    def decode(self, indices: list[int]) -> str:
        return self.backend.decode(indices).replace('  ', '_________').replace(' ', '').replace('_________', ' ')
