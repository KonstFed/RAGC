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
) -> tuple[Dataset, Dataset, Dataset]:
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
            else:
                prev_pos, prev_neg = pairs[link_type]
                pos = torch.cat([prev_pos, pos], dim=1)
                neg = torch.cat([prev_neg, neg], dim=1)
                pairs[link_type] = (pos, neg)
    hetero_batch.samples = pairs
    return hetero_batch



def collate_for_validation(batch: list[HeteroData]) -> Batch:
    """Collate hetero graphs with sampled pairs for validation."""
    pairs = [b.pairs for b in batch if b.pairs.shape[1] != 0]
    init_embs = [b.init_embs for b in batch if len(b.init_embs) != 0]
    for b in batch:
        del b.pairs
        del b.init_embs

    hetero_batch = Batch.from_data_list(batch)

    collated_pairs = None
    collated_embs = None

    init_embs_offset = 0
    in_graph_offsets = hetero_batch["FUNCTION"].ptr
    for i in range(len(pairs)):
        edge_index = pairs[i]

        # init embs offset
        edge_index[0] += init_embs_offset
        init_embs_offset = edge_index[0].max()

        # update in graph connections
        edge_index[1] += in_graph_offsets[i]

        # collate embs

        if collated_pairs is None:
            collated_pairs = edge_index
            collated_embs = init_embs[i]
        else:
            collated_pairs = torch.concat([collated_pairs, edge_index], dim=1)
            collated_embs = torch.concat([collated_embs, init_embs[i]], dim=0)

    hetero_batch["pairs"] = collated_pairs
    hetero_batch["init_embs"] = collated_embs
    return hetero_batch
