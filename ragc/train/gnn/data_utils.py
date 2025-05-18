import torch
from torch.utils.data import Dataset, random_split
from torch_geometric.data import Batch, HeteroData
from torch_geometric.transforms import BaseTransform

from ragc.datasets.train_dataset import TorchGraphDataset


class MapDataset(Dataset):
    """Dataset mapper.

    Given a dataset, creates a dataset which applies a mapping function
    to its items (lazily, only when an item is called).
    Note that data is not cloned/copied from the initial dataset.
    """

    def __init__(self, dataset: Dataset, map_fn: BaseTransform):
        self.dataset = dataset
        self.map = map_fn

    def __getitem__(self, index: int):
        return self.map(self.dataset[index])

    def __len__(self):
        return len(self.dataset)


def train_val_test_split(
    ds: TorchGraphDataset,
    ratios: list[float],
    train_tf: BaseTransform,
    val_tf: BaseTransform,
    test_tf: BaseTransform,
) -> tuple[MapDataset, MapDataset, MapDataset]:
    """Split dataset into training, validation and test part."""
    if len(ratios) != 3:
        raise ValueError(f"Not correct ratios {ratios}")

    train_ds, val_ds, test_ds = random_split(ds, ratios)

    train_ds = MapDataset(train_ds, train_tf)
    val_ds = MapDataset(val_ds, val_tf)
    test_ds = MapDataset(val_ds, test_tf)

    return train_ds, val_ds, test_ds


def collate_with_samples(batch: list[HeteroData]) -> Batch:
    """Collate hetero graphs with sampled pairs for training."""
    samples = [b.samples for b in batch]
    for b in batch:
        del b.samples

    hetero_batch = Batch.from_data_list(batch)
    pairs = {}
    pairs_ptr = {}

    for graph_idx, sample in enumerate(samples):
        for link_type in sample:
            src, _edge, dst = link_type
            src_offset = hetero_batch[src].ptr[graph_idx]
            dst_offset = hetero_batch[dst].ptr[graph_idx]

            pos, neg = sample[link_type]

            if pos.shape[1] == 0 or neg.shape[1] == 0:
                continue

            pos[0] += src_offset
            pos[1] += dst_offset

            neg[0] += src_offset
            neg[1] += dst_offset

            if link_type not in pairs:
                pairs[link_type] = (pos, neg)
                pairs_ptr[link_type] = [0]
            else:
                prev_pos, prev_neg = pairs[link_type]
                pos = torch.cat([prev_pos, pos], dim=1)
                neg = torch.cat([prev_neg, neg], dim=1)
                pairs[link_type] = (pos, neg)

                ptr_list = pairs_ptr[link_type]
                ptr_list.append(prev_pos.shape[1])

    hetero_batch.samples = pairs
    # hetero_batch.pair_ptr = pairs_ptr # doesnt work yet
    # print(pairs_ptr)
    return hetero_batch

def collate_with_positives(batch: list[HeteroData]) -> Batch:
    """Collate hetero graphs with sampled pairs for training."""
    def _has_all_positives(positives: dict[str, torch.Tensor]):
        return all(v.shape[0] != 0 for v in positives.values())

    batch = [b for b in batch if _has_all_positives(b.positives)]
    positives = [b.positives for b in batch]
    for b in batch:
        del b.positives

    hetero_batch = Batch.from_data_list(batch)
    collated_positives = {}
    positive_ptr = {}
    for graph_idx, sample in enumerate(positives):
        for link_type in sample:
            _src, _edge, dst = link_type
            src_offset = hetero_batch[dst].ptr[graph_idx]

            pos = sample[link_type]

            if pos.shape[0] == 0:
                continue

            pos += src_offset

            if link_type not in collated_positives:
                positive_ptr[link_type] = [0]
                collated_positives[link_type] = pos
            else:
                prev_pos = collated_positives[link_type]
                pos = torch.cat([prev_pos, pos], dim=0)
                collated_positives[link_type] = pos
                positive_ptr[link_type] = positive_ptr[link_type] + [len(prev_pos)]

    positive_ptr = {k:[*v, len(collated_positives[k])] for k,v in positive_ptr.items()}
    hetero_batch.positives = collated_positives
    hetero_batch.positives_ptr = positive_ptr
    return hetero_batch


def unbatch_with_positives(batch: Batch) -> list[HeteroData]:
    positives = batch.positives
    positives_ptr = batch.positives_ptr

    graphs = batch.to_data_list()

    for i in range(len(graphs)):
        graphs[i].positives = {}

    for link_type in positives_ptr:
        _src, _edge_, dst = link_type
        for i in range(len(graphs)):
            start = positives_ptr[link_type][i]
            end = positives_ptr[link_type][i+1]
            cur_positives = positives[link_type][start:end].clone()
            cur_positives -= batch[dst].ptr[i]
            graphs[i].positives[link_type] = cur_positives

    return graphs


def collate_for_validation(batch: list[HeteroData]) -> Batch:
    """Collate hetero graphs with sampled pairs for validation."""
    batch = [b for b in batch if b.pairs.shape[1] != 0]
    pairs = [b.pairs for b in batch]
    init_embs = [b.init_embs for b in batch]
    for b in batch:
        del b.pairs
        del b.init_embs

    hetero_batch = Batch.from_data_list(batch)

    collated_pairs = None
    collated_embs = None
    emb_ptr = []

    init_embs_offset = 0
    in_graph_offsets = hetero_batch["FUNCTION"].ptr
    for i in range(len(pairs)):
        edge_index = pairs[i]

        # init embs offset
        edge_index[0] += init_embs_offset
        init_embs_offset = edge_index[0].max() + 1

        # update in graph connections
        edge_index[1] += in_graph_offsets[i]

        # collate embs

        if collated_pairs is None:
            collated_pairs = edge_index
            collated_embs = init_embs[i]
            emb_ptr.append(0)
        else:
            emb_ptr.append(len(collated_embs))
            collated_pairs = torch.concat([collated_pairs, edge_index], dim=1)
            collated_embs = torch.concat([collated_embs, init_embs[i]], dim=0)

    emb_ptr.append(len(collated_embs))
    hetero_batch["pairs"] = collated_pairs
    hetero_batch["init_embs"] = collated_embs
    hetero_batch["init_embs_ptr"] = torch.tensor(emb_ptr)
    return hetero_batch



def has_duplicate_edges(edge_index: torch.Tensor) -> bool:
    edge_pairs = edge_index.t()  # shape [num_edges, 2]
    unique_edge_pairs = torch.unique(edge_pairs, dim=0)
    return unique_edge_pairs.size(0) < edge_pairs.size(0)


def stable_sort_edge_index(edge_index: torch.Tensor) -> torch.Tensor:
    edge_pairs = edge_index.t()  # shape [B, 2]

    # First sort by dst (secondary key), then stable sort by src (primary key)
    idx = torch.argsort(edge_pairs[:, 1], stable=True)
    edge_pairs = edge_pairs[idx]
    idx = torch.argsort(edge_pairs[:, 0], stable=True)
    edge_pairs = edge_pairs[idx]

    return edge_pairs.t().contiguous()