from copy import deepcopy

import networkx as nx
import torch
from torch_geometric.data import Data
from torch_geometric.utils.convert import from_networkx

from ragc.graphs.common import EdgeType, EdgeTypeNumeric, NodeType, NodeTypeNumeric, Node

nodetype2idx = {
    NodeType.FUNCTION: NodeTypeNumeric.FUNCTION.value,
    NodeType.CLASS: NodeTypeNumeric.CLASS.value,
    NodeType.FILE: NodeTypeNumeric.FILE.value,
    NodeType.FUNCTION.value: NodeTypeNumeric.FUNCTION.value,
    NodeType.CLASS.value: NodeTypeNumeric.CLASS.value,
    NodeType.FILE.value: NodeTypeNumeric.FILE.value,
}

idx2nodetype = {
    NodeTypeNumeric.FUNCTION.value: NodeType.FUNCTION,
    NodeTypeNumeric.CLASS.value: NodeType.CLASS,
    NodeTypeNumeric.FILE.value: NodeType.FILE,
}

edgetype2idx = {
    EdgeType.CALL.value: EdgeTypeNumeric.CALL.value,
    EdgeType.OWNER.value: EdgeTypeNumeric.OWNER.value,
    EdgeType.IMPORT.value: EdgeTypeNumeric.IMPORT.value,
    EdgeType.INHERITED.value: EdgeTypeNumeric.INHERITED.value,
    EdgeType.CALL: EdgeTypeNumeric.CALL.value,
    EdgeType.OWNER: EdgeTypeNumeric.OWNER.value,
    EdgeType.IMPORT: EdgeTypeNumeric.IMPORT.value,
    EdgeType.INHERITED: EdgeTypeNumeric.INHERITED.value,
}


def reverse_inheritance(graph: nx.MultiDiGraph) -> None:
    remove_edges = []
    for u, v, data in graph.edges(data=True):
        if data["type"] != EdgeType.INHERITED.value:
            continue

        remove_edges.append((u, v))

    for u, v in remove_edges:
        graph.remove_edge(u, v)
        graph.add_edge(u, v, type=EdgeType.INHERITED.value)


def graph2pyg(graph: nx.MultiDiGraph) -> Data:
    graph = deepcopy(graph)
    pyg_graph = from_networkx(graph)
    pyg_graph.edge_type = torch.tensor([edgetype2idx[edge] for edge in pyg_graph.edge_type])
    pyg_graph.type = torch.tensor([nodetype2idx[node] for node in pyg_graph.type])
    return pyg_graph


def get_call_neighbors(graph: Data, node: int, out: bool = True) -> tuple[list[int], torch.Tensor]:
    dir_idx = 0 if out else 1
    opp_dir_idx = (dir_idx + 1) % 2

    mask = graph.edge_index[dir_idx] == node
    mask = mask & (graph.edge_index[opp_dir_idx] != node)

    out_edge_idx = torch.where(mask)[0]
    out_edge_idx = [ind for ind in out_edge_idx if graph.edge_type[ind] == EdgeTypeNumeric.CALL.value]
    out_edge_idx = torch.tensor(out_edge_idx)

    if len(out_edge_idx) == 0:
        return [], None
    successor_nodes = graph.edge_index[opp_dir_idx][out_edge_idx].tolist()
    return successor_nodes, out_edge_idx


def get_callee_subgraph(graph: Data, node: int) -> tuple[torch.Tensor, torch.Tensor]:
    visited = set()
    stack = [node]
    while len(stack) > 0:
        cur_node = stack.pop(0)
        callees_nodes, _ = get_call_neighbors(graph=graph, node=cur_node, out=False)
        callees_nodes = set(callees_nodes)
        stack.extend(callees_nodes.difference(visited))
        visited = visited.union(callees_nodes)

    node_mask = torch.zeros(graph.num_nodes, dtype=torch.bool)
    edge_mask = torch.zeros(graph.num_edges, dtype=torch.bool)

    for node in visited:
        node_mask[node] = True
        node_edge_mask = (graph.edge_index[0] == node) | (graph.edge_index[1] == node)
        edge_mask = edge_mask | node_edge_mask & (graph.edge_type == EdgeTypeNumeric.CALL.value)
    return node_mask, edge_mask


def mask_node(graph: Data, node: int) -> tuple[torch.Tensor, torch.Tensor]:
    edge_mask = (graph.edge_index[0] == node) | (graph.edge_index[1] == node)
    edge_mask = ~edge_mask
    node_mask = torch.ones(graph.num_nodes, dtype=torch.bool)
    node_mask[node] = False
    return node_mask, edge_mask


def get_callee_mask(graph: Data, node: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Get mask for graph excluding node and its callee subgraph.

    Args:
        graph (Data): graph in torch_geometric.Data
        node (int): node index

    Returns:
        tuple[torch.Tensor, torch.Tensor]: node mask and edge mask

    """
    callee_mask, callee_edge_mask = get_callee_subgraph(graph=graph, node=node)

    callee_mask = ~callee_mask
    callee_edge_mask = ~callee_edge_mask

    node_mask, edge_mask = mask_node(graph=graph, node=node)
    edge_mask = callee_edge_mask & edge_mask
    node_mask = callee_mask & node_mask
    return node_mask, edge_mask


def apply_mask(graph: Data, node_mask: torch.Tensor, edge_mask: torch.Tensor) -> Data:
    # Get indices of kept nodes
    kept_nodes = torch.where(node_mask)[0]

    # Create mapping from original node indices to new indices
    idx_map = torch.zeros_like(node_mask, dtype=torch.long)
    idx_map[kept_nodes] = torch.arange(len(kept_nodes), device=node_mask.device)

    # Apply masks and remap edge indices
    return Data(
        x=graph.x[node_mask],
        edge_index=idx_map[graph.edge_index[:, edge_mask]],
        name=[graph.name[i] for i in kept_nodes] if graph.name else None,
        docstring=[graph.docstring[i] for i in kept_nodes] if graph.name else None,
        signature=[graph.signature[i] for i in kept_nodes] if graph.name else None,
        type=graph.type[node_mask] if graph.type is not None else None,
        code=[graph.code[i] for i in kept_nodes] if graph.code else None,
        file_path=[graph.file_path[i] for i in kept_nodes] if graph.file_path else None,
        edge_type=graph.edge_type[edge_mask] if graph.edge_type is not None else None,
    )


def pyg_extract_node(graph: Data, indices: list[int]) -> list[Node]:
    """Extract Node representation from PYG graph located with indices."""
    nodes = []
    for idx in indices:
        n = Node(
            name=graph.name[idx],
            type=idx2nodetype[int(graph.type[idx])],
            docstring=graph.docstring[idx],
            signature=graph.signature[idx],
            code=graph.code[idx],
            file_path=graph.file_path[idx],
        )
        nodes.append(n)
    return nodes
