import torch
from torch import nn

class Projector(nn.Module):
    """Projects original embedding into node space."""

    def __init__(self, orig_emb_size: int, node_emb_size: int):
        super().__init__()
        self.compose = nn.Sequential(
            nn.Linear(orig_emb_size, 1024),
            nn.ReLU(),
            #nn.Linear(1024, 2048),
            #nn.ReLU(),
            #nn.Linear(2048, 2048),
            #nn.ReLU(),
            #nn.Linear(2048, 768),
            #nn.ReLU(),
            nn.Linear(1024, node_emb_size),
        )

    def forward(self, og_embeddings: torch.Tensor) -> torch.Tensor:
        return self.compose(og_embeddings)
