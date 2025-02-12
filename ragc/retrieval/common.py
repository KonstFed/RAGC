from pathlib import Path

import networkx as nx


class BaseRetrieval:
    """Базовый класс для всех retrieval."""

    def __init__(self, repo_path: Path) -> None:
        self.repo_path = repo_path

    def retrieve(self, query: str, n_elems: int) -> list[str]:
        """Получить релевантные куски кода."""
        raise NotImplementedError
