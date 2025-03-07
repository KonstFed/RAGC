from pathlib import Path

import networkx as nx
import numpy as np
import ollama

from ragc.graphs import NodeType
from ragc.graphs.utils import get_file_graph
from ragc.retrieval import BaseRetrieval
from ragc.utils import load_secrets

SECRETS = load_secrets()


class BaseEmbRetieval(BaseRetrieval):
    """Base class for retrieval using graph and embeddings."""

    def __init__(self, repo_path: Path, graph: nx.MultiDiGraph, emb_model: str) -> None:
        super().__init__(repo_path, graph)
        self.emb_model = emb_model
        self.ollama_client = ollama.Client(host=SECRETS["OLLAMA_URL"])
        self.embeddings = None
        self.index2node = None
        self._init_index()

    def get_sorted_similar_nodes(self, query_emb: np.ndarray) -> list[str, float]:
        cosine_sim = self.embeddings @ query_emb.T
        cosine_sim = cosine_sim.reshape(cosine_sim.shape[0])
        cosine_sim = [(self.index2node[i], sim) for i, sim in enumerate(cosine_sim)]
        cosine_sim.sort(key=lambda p: p[1], reverse=True)

        return cosine_sim

    def _init_index(self) -> None:
        all_files = []
        node_idx = []
        for n, data in self.graph.nodes(data=True):
            all_files.append(data["code"])
            node_idx.append(n)

        self.index2node = dict(enumerate(node_idx))
        response = self.ollama_client.embed(model=self.emb_model, input=all_files)
        embeddings = np.array(response.embeddings)
        # normalize embeddings or not
        self.embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        node2emb = dict(zip(node_idx, self.embeddings, strict=True))
        nx.set_node_attributes(self.graph, node2emb, "embedding")

    def retrieve(self, query: str, n_elems: int) -> list[tuple[str, str]]:
        """Получить релевантные куски кода."""
        query_emb = self.ollama_client.embed(model=self.emb_model, input=query).embeddings
        query_emb = query_emb / np.linalg.norm(query_emb)
        best_results = self.get_sorted_similar_nodes(query_emb)

        out = []
        for node, _sim in best_results[:n_elems]:
            node_data = self.graph.nodes(data=True)[node]
            out.append((node, node_data["code"]))
        return out


class FileEmbRetrieval(BaseEmbRetieval):
    def __init__(self, repo_path: Path, graph: nx.MultiDiGraph, emb_model: str) -> None:
        file_graph = get_file_graph(graph=graph)
        super().__init__(repo_path, file_graph, emb_model)


class LowGranularityRetrieval(BaseEmbRetieval):
    def __init__(self, repo_path: Path, graph: nx.MultiDiGraph, emb_model: str) -> None:
        not_file_nodes = [n for n, attr in graph.nodes(data=True) if attr["type"] != NodeType.FILE]
        graph = graph.subgraph(not_file_nodes)

        super().__init__(repo_path, graph, emb_model)
