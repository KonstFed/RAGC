from abc import ABC, abstractmethod
from pathlib import Path

import networkx as nx

from ragc.graphs.common import Node

class BaseRetrieval(ABC):
    """Базовый класс для всех retrieval."""

    def __init__(self, repo_path: Path, graph: nx.MultiDiGraph) -> None:
        self.graph = graph
        self.repo_path = repo_path

    @abstractmethod
    def retrieve(self, query: str, n_elems: int) -> list[Node]:
        """Получить релевантные куски кода."""
        raise NotImplementedError
