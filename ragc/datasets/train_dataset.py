import copy

import networkx as nx
import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.transforms import BaseTransform
from torch_geometric.transforms import ComposeFilters

from ragc.graphs.common import EdgeTypeNumeric, NodeTypeNumeric
from ragc.graphs.transforms import get_call_neighbors, get_callee_subgraph, graph2pyg, mask_node
from ragc.llm.embedding import BaseEmbedder


class Embed(BaseTransform):
    """Embeddes all nodes except Files"""

    def __init__(self, embedder: BaseEmbedder):
        self.embedder = embedder
        super().__init__()

    def __call__(self, data: Data) -> Data:
        data_c = copy.copy(data)
        not_file_mask = data_c.type != NodeTypeNumeric.FILE.value
        all_code = [data_c.code[i] for i in torch.where(not_file_mask)[0]]
        embeddings = torch.from_numpy(self.embedder.embed(all_code))
        # embeddings should be 2d tensor
        node_embeddings = torch.zeros(data.num_nodes, embeddings.shape[1])
        node_embeddings[not_file_mask] = embeddings

        data_c.x = node_embeddings
        return data_c


class FilterZeroEvalCandidates(BaseTransform):
    def __call__(self, data: Data) -> bool:
        return len(get_candidates(data)) != 0


def get_target_nodes(graph: Data, node: int, node_mask: torch.Tensor, edge_mask: torch.Tensor) -> torch.Tensor:
    call_edges_mask = (
        (graph.edge_index[0] == node) & (graph.edge_type == EdgeTypeNumeric.CALL.value) & (graph.edge_index[1] != node)
    )
    # remove all known connections
    call_edges_mask = call_edges_mask & ~edge_mask

    # all call nodes
    nodes = graph.edge_index[1][call_edges_mask]
    nodes = torch.unique(nodes)

    # target is only in known graph
    known_nodes = torch.where(node_mask)[0]
    nodes = torch.tensor(list(set(nodes.tolist()) & set(known_nodes.tolist())))
    return nodes


def get_candidates(graph: Data) -> list[tuple[int, torch.Tensor, torch.Tensor]]:
    func_nodes = torch.where(graph.type == NodeTypeNumeric.FUNCTION.value)[0]

    candidates = []
    for i, f_node in enumerate(func_nodes):
        caller_nodes, _ = get_call_neighbors(graph=graph, node=int(f_node), out=True)
        if len(caller_nodes) < 1:
            continue

        callee_mask, callee_edge_mask = get_callee_subgraph(graph=graph, node=int(f_node))

        callee_mask = ~callee_mask
        callee_edge_mask = ~callee_edge_mask

        node_mask, edge_mask = mask_node(graph=graph, node=f_node)

        edge_mask = callee_edge_mask & edge_mask
        node_mask = callee_mask & node_mask

        target_nodes = get_target_nodes(graph, f_node, node_mask, edge_mask)

        if len(target_nodes) == 0 or node_mask.sum() < 5:
            continue

        candidates.append((f_node, node_mask, edge_mask, target_nodes))

    return candidates


class GraphDataset(InMemoryDataset):
    def __init__(self, root: str, graphs: list[nx.MultiDiGraph], transform=None, pre_transform=None, pre_filter=None):
        self._graphs = graphs
        super().__init__(root, transform, pre_transform, pre_filter)
        self.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return ["original_graph.pt"]

    def process(self) -> None:
        data_list = []
        for graph in self._graphs:
            pyg_graph = graph2pyg(graph)
            data_list.append(pyg_graph)

        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]

        if self.pre_filter is not None:
            data_list = list(filter(self.pre_filter, data_list))

        # Save the processed data
        self.save(data_list, self.processed_paths[0])
