from pathlib import Path

import networkx as nx

from ragc.retrieval import BaseRetrieval
from ragc.graphs import BaseGraphParser

class PageRankRetrieval(BaseRetrieval):
    def __init__(self, repo_path: Path, parser: BaseGraphParser) -> None:
        super().__init__(repo_path, parser)
        self.graph = parser.parse_into_files(repo_path=repo_path)
        _pagerank_mapping = nx.pagerank(self.graph)
        self._top_nodes = sorted(
            _pagerank_mapping.items(),
            key=lambda x: x[1],
            reverse=True,
        )

    def retrieve(self, query: str, n_elems: int) -> list[tuple[str,str]]:
        relevant_nodes = [n[0] for n in self._top_nodes[:n_elems]]
        relevant_texts = [
            (n, self.graph.nodes(data=True)[n]["code"]) for n in relevant_nodes
        ]
        return relevant_texts
