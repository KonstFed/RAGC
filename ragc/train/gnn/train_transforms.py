import torch
from torch_geometric.data import HeteroData
from torch_geometric.transforms import BaseTransform
from ragc.graphs.hetero_utils import remove_caller_subgraph, get_all_components

class InverseEdges(BaseTransform):
    """Inverse all edges for hetero graph."""

    def __init__(self, rev_suffix: str = "_REV"):
        self.rev_suffix = rev_suffix

    def forward(self, data: HeteroData):
        new_edge_types = {}
        edge_attrs = {}
        edge_types_to_remove = []
        # Collect all original edge types and their attributes
        for edge_type in data.edge_types:
            src, rel, dst = edge_type
            edge_index = data[edge_type].edge_index
            edge_attrs[edge_type] = {
                "edge_index": edge_index,
            }

            # Create reverse edge type
            new_rel = rel + self.rev_suffix
            new_edge_type = (dst, new_rel, src)
            new_edge_types[edge_type] = new_edge_type
            edge_types_to_remove.append(edge_type)

        for edge_type in edge_types_to_remove:
            del data[edge_type]

        # Add reversed edges
        for edge_type, new_edge_type in new_edge_types.items():
            src, rel, dst = edge_type
            edge_index = edge_attrs[edge_type]["edge_index"]

            # Reverse edges
            reversed_edge_index = edge_index.flip([0])

            # Store reversed edges
            data[dst, new_edge_type[1], src].edge_index = reversed_edge_index
        return data


def sample_same_link_pairs(
    graph: HeteroData,
    link_type: tuple[str, str, str],
    n_samples: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample for triplet loss from same links only.

    Args:
        graph (HeteroData): from where to sample
        link_type (tuple[str, str, str]): which type of edge to sample. Example `("FUNCTION", "CALL", "FUNCTION)`
        n_samples (int): upper bound of number samples to get.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: first is positive edge index and second is negative.
        It is guaranteed that pos and neg has equal shape.

    """
    c_edge_index = graph[link_type].edge_index
    n_samples = min(n_samples, c_edge_index.size(1))

    # positive pairs
    perm = torch.randperm(c_edge_index.size(1))
    idx = perm[: min(n_samples, len(perm))]
    pos_samples = c_edge_index[:, idx]

    # negative pairs
    possible_neg_samples = torch.randint(graph[link_type[2]].num_nodes, size=(n_samples,), device=c_edge_index.device)
    _mask = torch.zeros(n_samples, dtype=torch.bool)

    for i, (caller_id, neg_ind) in enumerate(zip(pos_samples[0], possible_neg_samples, strict=True)):
        if ((c_edge_index[0] == caller_id) & (c_edge_index[1] == neg_ind)).sum() == 0:
            _mask[i] = True

    possible_neg_samples = torch.stack([pos_samples[0], possible_neg_samples], axis=0)
    neg_samples = possible_neg_samples[:, _mask]
    pos_samples = pos_samples[:, _mask]

    return pos_samples, neg_samples


class SamplePairs(BaseTransform):
    """Sample for training pairs."""

    def __init__(self, links: list[tuple[str, str, str]], sample_ratio: float, min_size: int, max_size: int):
        self.links = links
        self.sample_ratio = sample_ratio
        self.min_size = min_size
        self.max_size = max_size

    def forward(
        self,
        data: HeteroData,
    ) -> tuple[HeteroData, dict[tuple[str, str, str], tuple[torch.Tensor, torch.Tensor]]]:
        samples = {}
        for link in self.links:
            edge_index = data[link].edge_index
            num_func_calls = edge_index.shape[1]

            n_samples = max(round(num_func_calls * self.sample_ratio), self.min_size)
            n_samples = min(self.max_size, n_samples)

            pos, neg = sample_same_link_pairs(data, link, n_samples)
            samples[link] = (pos, neg)
        data.samples = samples
        return data

class SampleCallPairsSubgraph(BaseTransform):
    """Sample from graph pairs of function that need to count retrieval metrics.

    To avoid dataleak will also make subgraph."""

    def get_func_candidates(self, graph: HeteroData, mask: dict[str, torch.Tensor]):
        call_edge_index = graph["FUNCTION", "CALL", "FUNCTION"].edge_index
        not_called = torch.ones(graph["FUNCTION"].num_nodes, dtype=torch.bool)
        not_called[call_edge_index[1]] = False

        # only use func from that component
        not_called = not_called & mask["FUNCTION"]
        # should call someone
        cand_edge_index = call_edge_index[:, not_called[call_edge_index[0]]]
        not_called = cand_edge_index[0].unique()

        reduced_graph_mask = remove_caller_subgraph(graph, not_called, return_mask=True)

        cand_edge_index = cand_edge_index[:, torch.where(reduced_graph_mask["FUNCTION"][cand_edge_index[1]])[0]]
        return reduced_graph_mask, cand_edge_index

    def forward(self, data: HeteroData) -> HeteroData:
        components = get_all_components(data)
        graph_mask = {n: torch.zeros(data[n].num_nodes, dtype=torch.bool) for n in data.node_types}

        all_candidates = torch.tensor([], dtype=torch.int64)
        for comp in components:
            # it is required that components do not intersect
            this_comp_mask, candidates = self.get_func_candidates(data, mask=comp)
            for k in graph_mask:
                graph_mask[k] = graph_mask[k] | (this_comp_mask[k] & comp[k])

            all_candidates = torch.concat([all_candidates, candidates], dim=1)

        if all_candidates.shape[1] == 0:
            data.pairs = torch.zeros((2, 0), dtype=torch.int64)
            data.init_embs = torch.zeros(0)
            return data

        # torch unique for some reason make them in
        all_candidates, _indices = torch.sort(all_candidates, dim=1, stable=True)

        new_indices = []
        mapping = {}
        embs = []
        n_sampled_nodes = 0
        for s_func in all_candidates[0]:
            s_func = int(s_func)
            if s_func in mapping:
                idx = mapping[s_func]
            else:
                mapping[s_func] = n_sampled_nodes
                idx = n_sampled_nodes
                n_sampled_nodes += 1
                embs.append(data["FUNCTION"].x[s_func])

            new_indices.append(idx)

        assert len(all_candidates[0].unique()) == len(embs), f"{len(embs)} {len(all_candidates[0].unique())}"
        assert len(embs) == n_sampled_nodes

        embs = torch.stack(embs)
        new_indices = torch.tensor(new_indices, dtype=torch.int64)

        # new node to old node mapping
        new_mapping = torch.nonzero(graph_mask["FUNCTION"]).flatten().tolist()
        new_dst = []

        for dst_idx in all_candidates[1]:
            new_dst.append(new_mapping.index(dst_idx))

        new_dst = torch.tensor(new_dst, dtype=torch.int64)
        pairs = torch.stack([new_indices, new_dst], dim=0)

        new_graph = data.subgraph(graph_mask)
        new_graph.pairs = pairs
        new_graph.init_embs = embs

        # for l, r in zip(embs, data["FUNCTION"].x[all_candidates[0].unique()], strict=True):
        #     if not torch.allclose(l, r):
        #         raise ValueError

        return new_graph
