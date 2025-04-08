from abc import ABC, abstractmethod
from typing import Literal
from pathlib import Path

import torch
from pydantic import BaseModel, ConfigDict
from torch_geometric.data import Data

from ragc.graphs.common import Node


class BaseRetrieval(ABC):
    """Базовый класс для всех retrieval."""

    def __init__(self, graph: Data) -> None:
        self.graph = graph

    @abstractmethod
    def _retrieve(self, query: str | torch.Tensor) -> torch.Tensor:
        """Get relevant nodes as indexes."""
        raise NotImplementedError

    @abstractmethod
    def retrieve(self, query: str | torch.Tensor) -> list[Node]:
        """Get relevant nodes as Node."""


class BaseCachedRetrieval(BaseRetrieval):
    def __init__(self, graph: Data, cache_index_path: Path | None = None) -> None:
        self.cache_index_path = cache_index_path
        self.graph = graph

    @abstractmethod
    def save_index(self, path: Path) -> None:
        """Save index if exists.

        Args:
            path (Path): save path

        """

    @abstractmethod
    def load_index(self, path: Path) -> None:
        """Load index from path.

        Args:
            path (Path): load path

        """

    @abstractmethod
    def _index(self) -> None:
        """Index graph."""

    def _init_index(self, index_path: Path | None = None) -> None:
        """Init index by calculating or loading from cache.

        If `index_path` provided and no index found will save index to the path.

        Args:
            index_path (Path | None, optional): index path. Defaults to None.

        """
        if index_path is None:
            self._index()
            return

        if index_path.exists():
            self.load_index(path=index_path)
            return

        self._index()
        self.save_index(path=index_path)


class BaseRetievalConfig(BaseModel):
    type: str

    def create(self, graph: Data) -> BaseRetrieval:
        """Factory method that creates instance of retrieval."""
        raise NotImplementedError

    model_config = ConfigDict(
        extra="ignore",
        frozen=True,
        arbitrary_types_allowed=True,
    )


class NoRetrievalConfig(BaseRetievalConfig, BaseRetrieval):
    """Config for no retrieval."""

    type: Literal["no_retrieval"] = "no_retrieval"

    def _retrieve(self, query: str | torch.Tensor) -> torch.Tensor:
        return None

    def retrieve(self, query: str | torch.Tensor) -> list[Node]:
        return []

    def create(self, graph: Data):
        return self
