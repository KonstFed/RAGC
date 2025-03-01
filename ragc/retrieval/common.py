from pathlib import Path

import networkx as nx

from ragc.graphs.common import BaseGraphParser

class BaseRetrieval:
    """Базовый класс для всех retrieval."""

    def __init__(self, repo_path: Path, parser: BaseGraphParser) -> None:
        self.parser = parser
        self.repo_path = repo_path

    def retrieve(self, query: str, n_elems: int) -> list[str]:
        """Получить релевантные куски кода."""
        raise NotImplementedError
