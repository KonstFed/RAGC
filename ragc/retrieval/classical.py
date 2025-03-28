from typing import Literal

import torch
import torch.nn.functional as F
from pcst_fast import pcst_fast
from torch_geometric.data import Data

from ragc.graphs.common import NodeTypeNumeric
from ragc.retrieval.common import BaseRetievalConfig, BaseRetrieval


class SimpleEmbRetrieval(BaseRetrieval):
    ind2node: torch.tensor = None
    embeddings: torch.tensor = None

    def __init__(self, graph: Data, ntop: int) -> None:
        # graph should have x parameters which represents embeddings for each node
        self.ntop = ntop
        self.graph = graph
        super().__init__(graph)
        self._init_embeddings()

    def _init_embeddings(self) -> None:
        not_file_mask = self.graph.type != NodeTypeNumeric.FILE.value
        self.ind2node = torch.nonzero(not_file_mask).squeeze()

        self.embeddings = self.graph.x[not_file_mask]
        self.embeddings = F.normalize(self.embeddings, p=2, dim=1)

    def retrieve(self, query: str | torch.Tensor) -> torch.Tensor:
        if isinstance(query, str):
            raise ValueError("пока не сделали для строк")

        query = F.normalize(query, p=2, dim=0)
        cosine_dist = self.embeddings @ query
        cosine_dist = torch.abs(cosine_dist)
        _, indices = torch.topk(cosine_dist, k=min(self.ntop, len(cosine_dist)))
        return self.ind2node[indices]


class SimpleEmbRetrievalConfig(BaseRetievalConfig):
    type: Literal["simple_emb"] = "simple_emb"
    ntop: int

    def create(self, graph: Data) -> SimpleEmbRetrieval:
        return SimpleEmbRetrieval(graph=graph, ntop=self.ntop)


def cosine_distance(embeddings: torch.Tensor, query_emb: torch.Tensor):
    embeddings = F.normalize(embeddings, p=2, dim=1)
    query_emb = F.normalize(query_emb, p=2, dim=0)
    dist = 1 - torch.abs(embeddings @ query_emb)
    return dist


def get_prize_and_cost(graph: Data, query_emb: torch.Tensor, k: int = 5) -> tuple[torch.Tensor, torch.Tensor]:
    k = min(k, graph.num_nodes)
    not_file_mask = graph.type != NodeTypeNumeric.FILE.value

    distances = -torch.ones_like(not_file_mask, dtype=graph.x.dtype)
    distances[not_file_mask] = -cosine_distance(graph.x[not_file_mask], query_emb)

    top_indices = torch.topk(distances, k)[1]
    prizes = torch.zeros_like(not_file_mask, dtype=graph.x.dtype)

    for i, ind in enumerate(top_indices):
        prizes[ind] = k - i

    costs = torch.ones(graph.edge_index.shape[1], dtype=torch.float64)
    return prizes, costs


def pcst(
    graph: Data,
    query_emb: torch.Tensor,
    k: int,
    root: int = -1,
    n_subgraphs: int = 1,
) -> tuple[torch.tensor, torch.tensor]:
    prize, cost = get_prize_and_cost(graph, query_emb, k)
    vertices, edges = pcst_fast(
        graph.edge_index.numpy().transpose(),
        prize.numpy(),
        cost.numpy(),
        root,
        n_subgraphs,
        "gw",
        0,
    )
    vertices, edges = torch.from_numpy(vertices), torch.from_numpy(edges)
    return vertices, edges


class PCSTRetrieval(BaseRetrieval):
    def __init__(self, graph: Data, n_subgraphs: int, ntop: int):
        self.n_subgraphs = n_subgraphs
        self.ntop = ntop
        super().__init__(graph)

    def retrieve(self, query: str | torch.Tensor) -> torch.Tensor:
        if isinstance(query, str):
            raise ValueError("пока не сделали для строк")

        vertices, _ = pcst(graph=self.graph, query_emb=query, k=self.ntop, n_subgraphs=self.n_subgraphs)
        return vertices


class PCSTConfig(BaseRetievalConfig):
    type: Literal["pcst"] = "pcst"

    n_subgraphs: int = 1
    ntop: int

    def create(self, graph: Data) -> "PCSTRetrieval":
        return PCSTRetrieval(graph=graph, n_subgraphs=self.n_subgraphs, ntop=self.ntop)
