from abc import ABC, abstractmethod
from pathlib import Path

import networkx as nx
from pydantic import BaseModel, ConfigDict

from ragc.graphs.common import Node


class BaseRetrieval(ABC):
    """Базовый класс для всех retrieval."""

    def __init__(self, graph: nx.MultiDiGraph, cache_index_path: Path | None = None) -> None:
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

    @abstractmethod
    def retrieve(self, query: str, n_elems: int, ignore_nodes: list[str] | None = None) -> list[Node]:
        """Получить релевантные куски кода."""
        raise NotImplementedError


class BaseRetievalConfig(BaseModel):
    def create(self, graph: nx.MultiDiGraph, cache_index_path: Path | None = None) -> BaseRetrieval:
        """Factory method that creates instance of retrieval."""
        raise NotImplementedError

    model_config = ConfigDict(
        extra="ignore",
        frozen=True,
    )
