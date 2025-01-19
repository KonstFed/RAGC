from pathlib import Path

import networkx as nx

from common import BaseRetrieval
from utils import get_call_graph


class PageRankRetrieval(BaseRetrieval):
    def __init__(self, repo_path: Path, semantic_graph: nx.MultiDiGraph) -> None:
        super().__init__(repo_path)
        _call_graph = get_call_graph(semantic_graph)
        _pagerank_mapping = nx.pagerank(_call_graph)
        self._top_nodes = sorted(
            _pagerank_mapping.items(), key=lambda x: x[1], reverse=True,
        )

    def retrieve(self, query: str, n_elems: int) -> list[str]:
        return (n[0] for n in self._top_nodes[:n_elems])
