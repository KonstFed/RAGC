import torch
from torch import nn

class Projector(nn.Module):
    """Projects original embedding into node space."""

    def __init__(self, orig_emb_size: int, node_emb_size: int):
        super().__init__()
        _relu = nn.ReLU()
        self.compose = nn.Sequential(
            nn.Linear(orig_emb_size, 1024),
            _relu,
            nn.Linear(1024, 768),
            _relu,
            nn.Linear(768, node_emb_size),
        )

    def forward(self, og_embeddings: torch.Tensor) -> torch.Tensor:
        return self.compose(og_embeddings)