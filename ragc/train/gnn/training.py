import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader
from torch_geometric.data import HeteroData
from torch_geometric.transforms import Compose
from tqdm import tqdm

from ragc.datasets.train_dataset import TorchGraphDataset
from ragc.graphs.hetero_transforms import DropIsolated, InitFileEmbeddings, RemoveExcessInfo, ToHetero
from ragc.train.gnn.data_utils import collate_with_samples, train_val_test_split, collate_for_validation
from ragc.train.gnn.models.hetero_graphsage import HeteroGraphSAGE
from ragc.train.gnn.train_transforms import InverseEdges, SamplePairs, SampleCallPairsSubgraph


def compute_loss(
    model,
    loss_fn,
    batch: HeteroData,
    node_embeddings,
) -> torch.Tensor | None:
    pairs_dict = batch.samples

    f_loss = None
    for link_type in pairs_dict:
        src, _, dst = link_type
        pos_idx, neg_idx = pairs_dict[link_type]
        pos_idx.to(batch["FILE"].x.device)
        neg_idx.to(batch["FILE"].x.device)

        # get embeddings
        anchor_idx = pos_idx[0, :]
        if len(anchor_idx) == 0:
            return None

        anchor_embs = batch[src].x[anchor_idx]
        cur_projector = model.proj_map[link_type]
        anchor_embs = cur_projector(anchor_embs)

        positive_embs = node_embeddings[dst][pos_idx[1, :]]
        negative_embs = node_embeddings[dst][neg_idx[1, :]]

        loss = loss_fn(anchor_embs, positive_embs, negative_embs)
        if f_loss is None:
            f_loss = loss
        else:
            f_loss += loss

    return f_loss


def train_epoch(model, loss_fn, device, optimizer, train_loader):
    model.train()
    losses = []
    bar = tqdm(train_loader)

    for batched_graph in bar:
        optimizer.zero_grad()
        batched_graph.to(device)
        node_embeddings = model(batched_graph.x_dict, batched_graph.edge_index_dict)

        loss = compute_loss(model, loss_fn, batched_graph, node_embeddings)
        if loss is None:  # no func TODO fix sampling
            continue

        loss.backward()

        optimizer.step()

        loss = loss.cpu().item()
        losses.append(loss)

        avg_loss = sum(losses) / len(losses)
        bar.set_description(f"avg loss: {avg_loss}; loss: {loss}")


def predict(model, batch: HeteroData, node_embeddings: dict[str, torch.Tensor]):
    pairs_dict = batch.samples
    predictions = {}
    target = {}
    for link_type, (pos_idx, neg_idx) in pairs_dict.items():
        src, _, dst = link_type
        pos_idx.to(batch["FILE"].x.device)
        neg_idx.to(batch["FILE"].x.device)

        # get embeddings
        anchor_idx = pos_idx[0, :]
        anchor_embs = batch[src].x[anchor_idx]
        cur_projector = model.proj_map[link_type]
        anchor_embs = cur_projector(anchor_embs)

        positive_embs = node_embeddings[dst][pos_idx[1, :]]
        negative_embs = node_embeddings[dst][neg_idx[1, :]]

        pred_positive = F.cosine_similarity(anchor_embs, positive_embs, dim=1)
        pred_positive = (pred_positive + 1) / 2
        pred_negative = F.cosine_similarity(anchor_embs, negative_embs, dim=1)
        pred_negative = (pred_negative + 1) / 2

        predictions[link_type] = torch.concat(
            [pred_positive, pred_negative],
            dim=0,
        )
        target[link_type] = torch.concat(
            [
                torch.ones(len(pred_positive)),
                torch.zeros(len(pred_negative)),
            ],
        )

    return predictions, target


def eval_classification(model, device, loader):
    model.eval()
    bar = tqdm(loader)

    g_predictions = {}
    g_target = {}
    counter = {}
    with torch.no_grad():
        for batched_graph in bar:
            batched_graph.to(device)
            node_embeddings = model(batched_graph.x_dict, batched_graph.edge_index_dict)
            predictions, targets = predict(model, batched_graph, node_embeddings)
            for link_type in predictions:
                preds = predictions[link_type]
                counter[link_type] = counter.get(link_type, 0) + len(preds)
                if link_type in g_predictions:
                    g_predictions[link_type] = torch.concat([g_predictions[link_type], preds], dim=0)
                else:
                    g_predictions[link_type] = preds

                c_target = targets[link_type]
                if link_type in g_target:
                    g_target[link_type] = torch.concat([g_target[link_type], c_target], dim=0)
                else:
                    g_target[link_type] = c_target

    metrics = {}
    for link_type in g_target:
        assert (counter[link_type] == g_target[link_type].shape[0]) and (
            counter[link_type] == g_predictions[link_type].shape[0]
        )
        score = roc_auc_score(g_target[link_type].cpu().numpy(), g_predictions[link_type].cpu().numpy())
        metrics[link_type] = score
    return metrics


def retrieve(
    model, node_embeddings: dict[str, torch.Tensor], init_embeddings: torch.Tensor, k: int = 0
) -> torch.Tensor:
    projected_embs = model.proj_map["FUNCTION", "CALL", "FUNCTION"](init_embeddings)

    # l2 normalize for cos distance
    projected_embs = F.normalize(projected_embs, p=2, dim=1)

    f_embs = node_embeddings["FUNCTION"]
    f_embs = F.normalize(f_embs, dim=1, p=2)

    cosine_distances = projected_embs @ f_embs.T
    _values, indices = torch.topk(cosine_distances, k=k, dim=1)
    return indices


def eval_retrieval(model, device, loader, k: int = 3) -> dict[str, float]:
    """Compute retrieval metrics."""
    model.eval()
    bar = tqdm(loader)

    actual = []
    predicted = []
    with torch.no_grad():
        for batched_graph in bar:
            batched_graph.to(device)
            node_embeddings = model(batched_graph.x_dict, batched_graph.edge_index_dict)
            # nodes for which we want to predict
            to_predict = batched_graph.pairs[0].unique()

            # get actual relevant nodes
            c_actual = [[] for _ in range(len(to_predict))]
            for n, relevant_f in batched_graph.pairs.T:
                c_actual[n].append(relevant_f)

            og_embs = batched_graph.init_embs[to_predict]
            pred_relevant = retrieve(model, node_embeddings, og_embs, k=k)

            actual.extend(c_actual)
            predicted.extend(pred_relevant.tolist())

    assert len(actual) == len(predicted)

    recalls = []
    precisions = []
    for actual_nodes, pred_nodes in zip(actual, predicted, strict=True):
        tp = len(set(actual_nodes).intersection(pred_nodes))
        recall = tp / len(actual_nodes)
        precision = tp / len(pred_nodes)
        recalls.append(recall)
        precisions.append(precision)

    return {
        "recall": sum(recalls) / len(recalls),
        "precision": sum(precisions) / len(precisions),
    }


def train():
    BATCH_SIZE = 10
    pair_sampler = SamplePairs([("FUNCTION", "CALL", "FUNCTION"), ("CLASS", "OWNER", "FUNCTION")], 0.2, 10, 100)

    train_transform = Compose(
        [
            ToHetero(),
            RemoveExcessInfo(),
            DropIsolated("FILE"),
            InitFileEmbeddings(),
            SamplePairs([("FUNCTION", "CALL", "FUNCTION"), ("CLASS", "OWNER", "FUNCTION")], 0.2, 10, 100),
            InverseEdges(rev_suffix=""),
        ]
    )

    val_transform = Compose(
        [
            ToHetero(),
            RemoveExcessInfo(),
            SampleCallPairsSubgraph(),
            DropIsolated("FILE"),
            InitFileEmbeddings(),
            InverseEdges(rev_suffix=""),
        ]
    )

    ds = TorchGraphDataset(
        root="data/torch_cache/repobench",
    )
    train_ds, val_ds, test_ds = train_val_test_split(
        ds,
        [0.6, 0.2, 0.2],
        train_tf=train_transform,
        val_tf=val_transform,
        test_tf=val_transform,
    )

    loader = DataLoader(train_ds, batch_size=BATCH_SIZE, collate_fn=collate_with_samples, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, collate_fn=collate_for_validation, shuffle=False)

    device = torch.device("cpu")

    model = HeteroGraphSAGE(768, 768, 768, 3)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

    triplet_loss = nn.TripletMarginLoss(
        margin=1.0,
        p=2,
        swap=False,
        reduction="mean",
    )

    for epoch in range(300):
        print(f"--------Epoch {epoch}-------")
        print("Training")
        # train_epoch(model, triplet_loss, device, optimizer, loader)
        print("Calculate metrics:")
        metrics = eval_retrieval(model, device, val_loader)
        print(metrics)


if __name__ == "__main__":
    train()
