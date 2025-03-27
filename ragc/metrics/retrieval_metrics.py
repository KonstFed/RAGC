import warnings

import torch
from torch_geometric.data import Data

from ragc.retrieval.common import BaseRetievalConfig
from ragc.datasets.train_dataset import GraphDataset
from ragc.graphs.common import EdgeTypeNumeric, NodeTypeNumeric
from ragc.graphs.transforms import get_call_neighbors, get_callee_subgraph, graph2pyg, mask_node, apply_mask


def _get_target_nodes(graph: Data, node: int, node_mask: torch.Tensor, edge_mask: torch.Tensor) -> torch.Tensor:
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


def _get_eval_candidates(graph: Data) -> list[tuple[int, torch.Tensor, torch.Tensor]]:
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

        target_nodes = _get_target_nodes(graph, f_node, node_mask, edge_mask)

        if len(target_nodes) == 0 or node_mask.sum() < 5:
            continue

        candidates.append((f_node, node_mask, edge_mask, target_nodes))

    return candidates


def make_predictions(
    dataset: GraphDataset, retreival_cfg: BaseRetievalConfig
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    predicted_results = []
    target_results = []
    for i in range(len(dataset)):
        graph = dataset[i]
        candidates = _get_eval_candidates(graph)

        if len(candidates) == 0:
            warnings.warn(f"graph with index {i} got zero evaluation candidates. Thus, it is skipped", stacklevel=1)

        for predict_node, node_mask, edge_mask, target_nodes in candidates:
            eval_graph = apply_mask(graph, node_mask, edge_mask)
            retrieval = retreival_cfg.create(eval_graph)

            # пока что используем эбмеддинг кода, но нужно доставать, docstring
            predicted_nodes = retrieval.retrieve(graph.x[predict_node])
            predicted_results.append(predicted_nodes)
            target_results.append(target_nodes)

    return predicted_results, target_results


def count_recall(predicted: list[torch.Tensor], target: list[torch.Tensor]) -> float:
    if len(predicted) != len(target):
        raise ValueError(
            f"predictions and target do not have same size. Predictions: {len(predicted)}, target: {len(target)}",
        )
    recall = []
    for pred_nodes, target_nodes in zip(predicted, target, strict=True):
        c_recall = set(pred_nodes.tolist()).intersection(target_nodes.tolist())
        recall.append(c_recall)

    return sum(recall) / len(recall)
