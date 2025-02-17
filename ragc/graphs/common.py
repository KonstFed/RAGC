from abc import ABC, abstractmethod
from pathlib import Path

import networkx as nx

class BaseGraphParser(ABC):

    @abstractmethod
    def parse(repo_path: Path) -> nx.MultiDiGraph:
        raise NotImplementedError