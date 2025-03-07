from abc import abstractmethod
from pathlib import Path

import networkx as nx
import numpy as np
import ollama

from ragc.graphs import BaseGraphParser, NodeType
from ragc.retrieval import BaseRetrieval
from ragc.utils import load_secrets


SECRETS = load_secrets()

class BaseEmbRetieval(BaseRetrieval):
    def __init__(self, repo_path: Path, parser: BaseGraphParser, emb_model: str) -> None:
        super().__init__(repo_path, parser)        
        self.emb_model = emb_model
        self.ollama_client = ollama.Client(host=SECRETS["OLLAMA_URL"])
        self.embeddings = None
        self.index2node = None
        self.graph = self._init_graph()
        self._init_index()

    @abstractmethod
    def _init_graph(self) -> nx.MultiDiGraph:
        raise NotImplementedError

    def get_sorted_similar_nodes(self, emb: np.ndarray) -> list[str, float]:
        cosine_sim = self.embeddings @ emb.T
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

        self.index2node = {i:n for i, n in enumerate(node_idx)}
        response = self.ollama_client.embed(model=self.emb_model, input=all_files)
        embeddings = np.array(response.embeddings)
        # normalize embeddings or not
        self.embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        node2emb = {n:emb for n, emb in zip(node_idx, self.embeddings, strict=True)}
        nx.set_node_attributes(self.graph, node2emb, "embedding")

    
    def retrieve(self, query: str, n_elems: int) -> list[tuple[str,str]]:
        query_emb = ollama.embed(model=self.emb_model, input=query).embeddings
        query_emb = query_emb / np.linalg.norm(query_emb)
        best_results = self.get_sorted_similar_nodes(query_emb)

        out = []
        for node, sim in best_results[:n_elems]:
            node_data = self.graph.nodes(data=True)[node]
            out.append((node, node_data["code"]))          
        return out
    
class FileEmbRetrieval(BaseEmbRetieval):

    def _init_graph(self) -> None:
         return self.parser.parse_into_files(repo_path=self.repo_path)

class LowGranularityRetrieval(BaseEmbRetieval):
    def _init_graph(self) -> None:
        graph = self.parser.parse(repo_path=self.repo_path)
        not_file_nodes = [n for n, attr in graph.nodes(data=True) if attr["type"] != NodeType.FILE]
        graph = graph.subgraph(not_file_nodes)
        return graph
