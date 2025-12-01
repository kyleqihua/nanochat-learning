import torch
import torch.nn as nn

class GPTConfig:
    def __init__(self, vocab_size=50304, n_layer=12, n_head=6, n_embd=768):
        self.vocab_size = vocab_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd