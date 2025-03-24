import copy

import networkx as nx
import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.transforms import BaseTransform

from ragc.graphs.common import EdgeTypeNumeric, NodeTypeNumeric
from ragc.graphs.transforms import get_call_neighbors, get_callee_subgraph, graph2pyg, mask_node
from ragc.llm.embedding import BaseEmbedder


class PreEmbed(BaseTransform):
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


class RetrievalEvaluationDataset(InMemoryDataset):
    def __init__(self, root: str, graphs: list[nx.MultiDiGraph], transform=None, pre_transform=None, pre_filter=None):
        self._graphs = graphs
        super().__init__(root, transform, pre_transform, pre_filter)
        self.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return "data.pt"

    def _collate_candidate(
        self, graph: Data, node: int, node_mask: torch.Tensor, edge_mask: torch.Tensor, target_nodes: torch.Tensor
    ) -> Data:
        candidate_data = Data(
            x=graph.x,  # Node features
            type=graph.type,
            code=graph.code,
            edge_index=graph.edge_index,  # Edge indices
            edge_type=graph.edge_type,  # Edge types
            y=target_nodes,  # Target nodes (labels)
            node_mask=node_mask,  # Node mask for the candidate
            edge_mask=edge_mask,  # Edge mask for the candidate
            x_node=node,  # The function node for which the candidate was generated
        )
        return candidate_data

    def process(self) -> None:
        data_list = []
        for graph in self._graphs:
            pyg_graph = graph2pyg(graph)

            if self.pre_transform is not None:
                pyg_graph = self.pre_transform(pyg_graph)

            candidates = get_candidates(pyg_graph)
            if len(candidates) == 0:
                continue

            for candidate in candidates:
                f_node, node_mask, edge_mask, target_nodes = candidate
                candidate_data = self._collate_candidate(pyg_graph, f_node, node_mask, edge_mask, target_nodes)

                # Create a Data object for each candidate

                # Append the candidate data to the list
                data_list.append(candidate_data)

        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]

        if self.pre_filter is not None:
            data_list = [self.pre_filter(d) for d in data_list]

        # Collate the list of Data objects into a single large Data object
        data, slices = self.collate(data_list)

        # Save the processed data
        torch.save((data, slices), self.processed_paths[0])
