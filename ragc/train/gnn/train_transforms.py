import torch
from torch_geometric.data import HeteroData
from torch_geometric.transforms import BaseTransform


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
