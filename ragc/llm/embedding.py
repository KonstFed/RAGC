from abc import ABC, abstractmethod

import numpy as np
from pydantic import BaseModel, ConfigDict


class BaseEmbedder(ABC):
    @abstractmethod
    def embed(self, inputs: list[str] | str) -> np.ndarray:
        """Generate embeddings for given inputs."""


class BaseEmbederConfig(BaseModel):
    def create(self) -> BaseEmbedder:
        """Factory method that creates instance of BaseEmbedder."""
        raise NotImplementedError

    model_config = ConfigDict(
        extra="ignore",
        frozen=True,
    )
