from abc import ABC, abstractmethod
from pathlib import Path

import networkx as nx

class BaseGraphParser(ABC):

    @abstractmethod
    def parse(self, repo_path: Path) -> nx.MultiDiGraph:
        raise NotImplementedError
    
    @abstractmethod
    def parse_into_files(self, repo_path: Path) -> nx.MultiDiGraph:
        raise NotImplementedError
