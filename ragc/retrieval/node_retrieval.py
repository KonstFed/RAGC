from pathlib import Path
from typing import Literal

import networkx as nx
import numpy as np

from ragc.graphs import NodeType
from ragc.graphs.common import Node
from ragc.graphs.utils import get_file_graph
from ragc.llm import EmbedderConfig
from ragc.llm.embedding import BaseEmbedder
from ragc.retrieval.common import BaseRetievalConfig, BaseRetrieval


class BaseEmbRetieval(BaseRetrieval):
    """Base class for retrieval using graph and embeddings."""

    def __init__(
        self,
        graph: nx.MultiDiGraph,
        embedder: BaseEmbedder,
        cache_index_path: Path | None = None,
    ) -> None:
        super().__init__(graph, cache_index_path=cache_index_path)
        self.embedder = embedder
        self.embeddings = None
        self.index2node = None
        self.node2emb = None
        self._init_index(index_path=cache_index_path)

    def get_sorted_similar_nodes(self, query_emb: np.ndarray) -> list[str, float]:
        cosine_sim = self.embeddings @ query_emb.T
        cosine_sim = cosine_sim.reshape(cosine_sim.shape[0])
        cosine_sim = [(self.index2node[i], sim) for i, sim in enumerate(cosine_sim)]
        cosine_sim.sort(key=lambda p: p[1], reverse=True)

        return cosine_sim

    def load_index(self, path: Path) -> None:
        nodes = sorted(self.graph.nodes)
        combined_embeddings = np.load(path)
        if combined_embeddings.shape[0] != len(nodes):
            raise ValueError("Number of embeddings in loaded index do not match up with number of nodes in graph")

        self.node2emb = dict(zip(nodes, combined_embeddings, strict=True))
        self._embedding_postprocessing()

    def save_index(self, path: Path) -> None:
        nodes = sorted(self.graph.nodes)
        combined_embeddings = [self.node2emb[n] for n in nodes]
        combined_embeddings = np.array(combined_embeddings, dtype=np.float32)
        np.save(path, combined_embeddings)

    def _index(self) -> None:
        node_idx = []
        all_files = []
        for n, data in self.graph.nodes(data=True):
            all_files.append(data["code"])
            node_idx.append(n)

        embeddings = self.embedder.embed(all_files)

        self.node2emb = dict(zip(node_idx, embeddings, strict=True))
        self._embedding_postprocessing()

    def _embedding_postprocessing(self) -> None:
        self.index2node = []
        self.embeddings = []

        for node in sorted(self.graph.nodes):
            self.index2node.append(node)
            self.embeddings.append(self.node2emb[node])

        self.embeddings = np.array(self.embeddings, dtype=np.float32)
        # normalize embeddings for cosine similarity
        self.embeddings = self.embeddings / np.linalg.norm(self.embeddings, axis=1, keepdims=True)

    def retrieve(self, query: str, n_elems: int, ignore_nodes: list[str] | None = None) -> list[Node]:
        """Получить релевантные куски кода."""
        if ignore_nodes is None:
            ignore_nodes = set()
        query_emb = self.embedder.embed(query)
        query_emb = query_emb / np.linalg.norm(query_emb)
        best_results = self.get_sorted_similar_nodes(query_emb)
        best_results = filter(lambda p: p[0] not in ignore_nodes, best_results)

        out = []
        for _ in range(n_elems):
            node, _sim = next(best_results)
            node_data = self.graph.nodes(data=True)[node]
            out.append(Node.model_validate(node_data))

        return out


class FileEmbRetrieval(BaseEmbRetieval):
    def __init__(self, graph: nx.MultiDiGraph, embedder: BaseEmbedder, cache_index_path: Path | None = None) -> None:
        file_graph = get_file_graph(graph=graph)
        super().__init__(file_graph, embedder, cache_index_path=cache_index_path)


class LowGranularityRetrieval(BaseEmbRetieval):
    def __init__(self, graph: nx.MultiDiGraph, embedder: BaseEmbedder, cache_index_path: Path | None = None) -> None:
        not_file_nodes = [n for n, attr in graph.nodes(data=True) if attr["type"] != NodeType.FILE]
        graph = graph.subgraph(not_file_nodes)

        super().__init__(graph, embedder, cache_index_path=cache_index_path)


class EmbRetrievalConfig(BaseRetievalConfig):
    embeder_config: EmbedderConfig


class FileEmbRetrievalConfig(EmbRetrievalConfig):
    type: Literal["file_retrieval"] = "file_retrieval"

    def create(self, graph: nx.MultiDiGraph) -> FileEmbRetrieval:
        return FileEmbRetrieval(graph=graph, embedder=self.embeder_config.create(), cache_index_path=self.cache_index_path)


class LowGranularityRetrievalConfig(EmbRetrievalConfig):
    type: Literal["low_granularity_retrieval"] = "low_granularity_retrieval"

    def create(self, graph: nx.MultiDiGraph) -> LowGranularityRetrieval:
        return LowGranularityRetrieval(
            graph=graph,
            embedder=self.embeder_config.create(),
            cache_index_path=self.cache_index_path,
        )
