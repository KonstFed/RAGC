import torch
from torch_geometric.data import HeteroData


def get_all_owned_nodes(
    graph: HeteroData,
    nodes: dict[str, torch.Tensor],
    return_mask: bool = False,
) -> dict[str, torch.Tensor]:
    """Breadth first search of all nodes that are owned by given nodes."""
    visited = {n_type: torch.zeros(graph[n_type].num_nodes, dtype=torch.bool) for n_type in graph.node_types}
    stack = []
    for node_type, node_indices in nodes.items():
        for n_idx in node_indices.tolist():
            stack.append((n_idx, node_type))
            visited[node_type][n_idx] = True

    while len(stack) > 0:
        node_idx, node_type = stack.pop(0)

        for possible_dst in ["CLASS", "FUNCTION"]:
            edge_index = graph[node_type, "OWNER", possible_dst].edge_index
            owned_nodes = edge_index[1, torch.where(edge_index[0] == node_idx)[0]]

            for owned_n in owned_nodes:
                if visited[possible_dst][owned_n]:
                    continue
                visited[possible_dst][owned_n] = True
                stack.append((owned_n, possible_dst))

    if return_mask:
        return visited
    return {node_type: torch.where(mask)[0] for node_type, mask in visited.items() if mask.sum() != 0}


def greedy_scan(graph: HeteroData, node_idx: int, node_type: str) -> dict[str, torch.Tensor]:
    """Breadth first search ignoring direction.

    Returns bool mask for each node type."""
    visited = {n_type: torch.zeros(graph[n_type].num_nodes, dtype=torch.bool) for n_type in graph.node_types}
    visited[node_type][node_idx] = True
    stack = [(node_idx, node_type)]

    while len(stack) > 0:
        cur_node_idx, cur_node_type = stack.pop(0)

        for rel in graph.edge_types:
            edge_index = graph[rel].edge_index

            if rel[2] == cur_node_type:
                # incoming edges
                inc_type = rel[0]
                incomin_nodes = edge_index[0, edge_index[1] == cur_node_idx]
                stack.extend([(n, inc_type) for n in incomin_nodes if not visited[inc_type][n]])
                visited[inc_type][incomin_nodes] = True

            if rel[0] == cur_node_type:
                # outcoming edges
                out_type = rel[2]
                outcoming_nodes = edge_index[1, edge_index[0] == cur_node_idx]
                stack.extend([(n, out_type) for n in outcoming_nodes if not visited[out_type][n]])
                visited[out_type][outcoming_nodes] = True
    return visited


def get_all_components(graph: HeteroData) -> dict[str, torch.Tensor]:
    """Map graph into connected components

    It uses assumption that all nodes are either FILE or owned by only one file."""

    # -1 means not inited
    # value represent index value
    inited_files = -1 * torch.ones(graph["FILE"].num_nodes, dtype=torch.int16)
    cls_idx = 0
    components = []
    while (inited_files == -1).sum() != 0:
        # choose first file index that is not present in clusters
        ind = torch.where(inited_files == -1)[0][0]

        # make greedy scan
        mask = greedy_scan(graph, ind, "FILE")
        components.append(mask)

        for file_id in torch.where(mask["FILE"])[0].tolist():
            inited_files[file_id] = cls_idx

        cls_idx += 1

    if (inited_files == -1).sum() != 0:
        raise ValueError("Sanity check failed. Some files are not inited")

    return components


def remove_caller_subgraph(graph: HeteroData, nodes: torch.Tensor, return_mask: bool = False) -> HeteroData:
    """Removes all nodes that has dependency on given nodes recursively.

    Node A considered dependent of Node B if any of these true
    - A calls B
    - A imports B
    - B owns A
    - A inherits B
    - A and B are owned by same function or class
    """
    if len(nodes) == 0:
        if return_mask:
            return {n: torch.ones(graph[n].num_nodes, dtype=torch.bool) for n in graph.node_types}
        return graph

    stack = [(n, "FUNCTION") for n in nodes.tolist()]
    visited = {n: torch.zeros(graph[n].num_nodes, dtype=torch.bool) for n in graph.node_types}

    visited["FUNCTION"][nodes] = True

    # all relations that show ala CALL relation
    DEPENDENCY_RELATIONS = [
        ("FUNCTION", "CALL", "FUNCTION"),
        ("FILE", "IMPORT", "FILE"),
        ("FILE", "IMPORT", "CLASS"),
        ("FILE", "IMPORT", "FUNCTION"),
        ("FILE", "CALL", "FUNCTION"),
        ("CLASS", "INHERITED", "CLASS"),
    ]

    while len(stack) > 0:
        cur_node, node_type = stack.pop(0)
        # remove all owned nodes
        owned_nodes = get_all_owned_nodes(graph, {node_type: torch.tensor([cur_node])})
        for owned_node_type, node_indices in owned_nodes.items():
            stack.extend([(n, owned_node_type) for n in node_indices.tolist() if not visited[owned_node_type][n]])
            visited[owned_node_type][node_indices] = True

        # remove all owners of this node beside FILE
        for own_node_type in ["FUNCTION", "CLASS"]:
            f_owner_rel = (own_node_type, "OWNER", node_type)
            if f_owner_rel not in graph.edge_types:
                continue
            edge_index = graph[f_owner_rel].edge_index
            f_owners = edge_index[0, torch.where(edge_index[1] == cur_node)[0]]
            if len(f_owners) != 0:
                stack.extend([(n, own_node_type) for n in f_owners if not visited[own_node_type][n]])
                visited[own_node_type][f_owners] = True

        cur_dep_rels = [rel for rel in DEPENDENCY_RELATIONS if rel[2] == node_type]
        for rel in cur_dep_rels:
            if rel not in graph.edge_types:
                continue
            edge_index = graph[rel].edge_index

            # it is nodes that are dependent on current
            dependent_nodes = edge_index[0, edge_index[1] == cur_node]
            for dep_node in dependent_nodes:
                if visited[rel[0]][dep_node]:
                    # already visited
                    continue
                visited[rel[0]][dep_node] = True
                stack.append((dep_node, rel[0]))

    if return_mask:
        # inverse because mask shows callers
        return {k: ~v for k, v in visited.items()}

    subset_indices = {k: torch.where(~v)[0] for k, v in visited.items()}
    return graph.subgraph(subset_indices)
