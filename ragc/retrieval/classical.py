from typing import Literal

import torch
import torch.nn.functional as F
from pcst_fast import pcst_fast
from torch_geometric.data import Data

from ragc.graphs.common import Node, NodeTypeNumeric, EdgeTypeNumeric
from ragc.graphs.utils import pyg_extract_node, apply_mask, mask_nodes
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

    def _retrieve(self, query: str | torch.Tensor) -> torch.Tensor:
        if isinstance(query, str):
            raise ValueError("пока не сделали для строк")

        query = F.normalize(query, p=2, dim=0)
        cosine_dist = self.embeddings @ query
        cosine_dist = torch.abs(cosine_dist)
        _, indices = torch.topk(cosine_dist, k=min(self.ntop, len(cosine_dist)))
        return self.ind2node[indices]

    def retrieve(self, query: str | torch.Tensor) -> list[Node]:
        indices = self._retrieve(query=query)
        return pyg_extract_node(graph=self.graph, indices=indices.tolist())


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
    def __init__(self, graph: Data, n_subgraphs: int, ntop: int, for_prompt: bool = False):
        self.n_subgraphs = n_subgraphs
        self.ntop = ntop
        self.for_prompt = for_prompt
        super().__init__(graph)

    def _retrieve(self, query: str | torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(query, str):
            raise ValueError("пока не сделали для строк")

        vertices, edges = pcst(graph=self.graph, query_emb=query, k=self.ntop, n_subgraphs=self.n_subgraphs)
        return vertices, edges

    def retrieve(self, query: str | torch.Tensor) -> list[Node]:
        vertices, edges = self._retrieve(query=query)
        if not self.for_prompt:
            return pyg_extract_node(graph=self.graph, indices=vertices.tolist())

        # perform removing to reduce context size
        # task PCST result from graph
        node_mask = torch.zeros(self.graph.num_nodes, dtype=torch.bool)
        node_mask[vertices] = True
        edge_mask = torch.zeros(self.graph.num_edges, dtype=torch.bool)
        edge_mask[edges] = True
        masked_graph = apply_mask(self.graph, node_mask, edge_mask)

        # Remove files
        file_nodes = torch.where(masked_graph.type == NodeTypeNumeric.FILE.value)[0]
        _node_msk, _edge_msk = mask_nodes(masked_graph, nodes=file_nodes)
        masked_graph = apply_mask(masked_graph, _node_msk, _edge_msk)

        # Remove nodes whose code is already included into some other node code
        own_edges = masked_graph.edge_index[:, masked_graph.edge_type == EdgeTypeNumeric.OWNER.value]
        nodes_with_owner = own_edges[1, :]
        _node_msk, _edge_msk = mask_nodes(masked_graph, nodes_with_owner)
        final_graph = apply_mask(masked_graph, _node_msk, _edge_msk)

        return pyg_extract_node(graph=final_graph)


class PCSTConfig(BaseRetievalConfig):
    type: Literal["pcst"] = "pcst"

    n_subgraphs: int = 1
    ntop: int

    for_prompt: bool = False

    def create(self, graph: Data) -> "PCSTRetrieval":
        return PCSTRetrieval(
            graph=graph,
            n_subgraphs=self.n_subgraphs,
            ntop=self.ntop,
            for_prompt=self.for_prompt,
        )
