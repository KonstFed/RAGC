import torch
import torch.nn.functional as F
from torch import nn


class TripletLoss:
    def __init__(self, **loss_kwargs: dict):
        self._triplet_loss = nn.TripletMarginLoss(
            **loss_kwargs,
        )

    def __call__(self, anchor_embeddings: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        # normalize using L2 so TripletMarginLoss (which uses Euclidean distance)
        # will compute cosine loss
        # L2 normalize embeddings (important: dim=1)
        anchor_embeddings = F.normalize(anchor_embeddings, p=2, dim=1)
        positive = F.normalize(positive, p=2, dim=1)
        negative = F.normalize(negative, p=2, dim=1)
        return self._triplet_loss(anchor_embeddings, positive, negative)


class CosineSimLoss:
    def __init__(self, loss_kwargs: dict):
        self._cos_los = nn.CosineEmbeddingLoss(**loss_kwargs)

    def __call__(self, anchor_embeddings: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        left = torch.concat([anchor_embeddings, anchor_embeddings], axis=0)
        right = torch.concat([positive, negative], dim=0)
        target = torch.concat(
            [torch.ones(len(positive), device=positive.device), -torch.ones(len(negative), device=positive.device)]
        )
        return self._cos_los(left, right, target)


class SimpleClassificationLoss:
    def __init__(self, loss_kwargs: dict):
        self._bce_loss = nn.BCELoss(**loss_kwargs)

    def __call__(self, anchor_embeddings: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        left = torch.concat([anchor_embeddings, anchor_embeddings], axis=0)
        right = torch.concat([positive, negative], dim=0)
        sim = F.cosine_similarity(left, right, dim=1)
        sim = (sim + 1) / 2
        target = torch.concat(
            [torch.ones(len(positive), device=positive.device), torch.zeros(len(negative), device=positive.device)],
        )
        return self._bce_loss(sim, target)
