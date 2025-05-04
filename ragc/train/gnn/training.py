from copy import copy, deepcopy

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
from ragc.train.gnn.data_utils import (
    collate_for_validation,
    collate_with_samples,
    train_val_test_split,
    collate_with_positives,
)
from ragc.train.gnn.models.hetero_graphsage import HeteroGraphSAGE
from ragc.train.gnn.train_transforms import InverseEdges, SampleCallPairsSubgraph, SamplePairs, PositiveSampler, HardSampler
from ragc.train.gnn.losses import TripletLoss


class Trainer:
    def __init__(
        self, dataset: TorchGraphDataset, model: HeteroGraphSAGE, loss_fn, optimizer, batch_size: int, retrieve_k: int
    ):
        self.model = model
        self.loss_fn = loss_fn
        self.retrieve_k = retrieve_k

        self.optimizer = optimizer
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model.to(self.device)

        train_transform = Compose(
            [
                ToHetero(),
                RemoveExcessInfo(),
                DropIsolated("FILE"),
                InitFileEmbeddings(),
                SamplePairs([("FUNCTION", "CALL", "FUNCTION"), ("CLASS", "OWNER", "FUNCTION")], 0.2, 10, 100),
                InverseEdges(rev_suffix=""),
            ],
        )

        # train_transform = Compose(
        #     (
        #         ToHetero(),
        #         RemoveExcessInfo(),
        #         DropIsolated("FILE"),
        #         InitFileEmbeddings(),
        #         PositiveSampler([("FUNCTION", "CALL", "FUNCTION"), ("CLASS", "OWNER", "FUNCTION")], 0.2, 10, 100),
        #         InverseEdges(rev_suffix=""),
        #     )
        # )

        val_transform = Compose(
            [
                ToHetero(),
                RemoveExcessInfo(),
                SampleCallPairsSubgraph(),
                DropIsolated("FILE"),
                InitFileEmbeddings(),
                InverseEdges(rev_suffix=""),
            ],
        )

        self.train_ds, self.val_ds, self.test_ds = train_val_test_split(
            dataset,
            [0.6, 0.2, 0.2],
            train_tf=train_transform,
            val_tf=val_transform,
            test_tf=val_transform,
        )
        # self.train_loader = DataLoader(
        #     self.train_ds,
        #     batch_size=batch_size,
        #     collate_fn=collate_with_positives,
        #     shuffle=True,
        # )
        self.train_loader = DataLoader(
            self.train_ds,
            batch_size=batch_size,
            collate_fn=collate_with_samples,
            shuffle=True,
        )
        self.val_loader = DataLoader(
            self.val_ds,
            batch_size=batch_size,
            collate_fn=collate_for_validation,
            shuffle=False,
        )

        self.hard_sampler = HardSampler()
        # self.clas_val_loader = DataLoader(class_val_ds, batch_size=batch_size, collate_fn=collate_with_samples, shuffle=False)

    ###----Training functions below----###

    def compute_loss(self, batch: HeteroData, node_embeddings: dict[str, torch.Tensor]) -> torch.Tensor:
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
            cur_projector = self.model.proj_map[link_type]
            anchor_embs = cur_projector(anchor_embs)

            positive_embs = node_embeddings[dst][pos_idx[1, :]]
            negative_embs = node_embeddings[dst][neg_idx[1, :]]

            loss = self.loss_fn(anchor_embs, positive_embs, negative_embs)
            if f_loss is None:
                f_loss = loss
            else:
                f_loss += loss

        return f_loss

    # def compute_loss(self, batch: HeteroData, node_embeddings: dict[str, torch.Tensor]) -> torch.Tensor:
    #     with torch.autograd.set_detect_anomaly(True):
    #         positives_dict = batch.positives

    #         f_loss = None
    #         for link_type in positives_dict:
    #             src, _, dst = link_type
    #             positives = positives_dict[link_type]
    #             positives.to(batch["FILE"].x.device)

    #             mapping = -1 * torch.ones(batch[src].x.shape[0], dtype=torch.int64)
    #             mapping = mapping.to(positives.device)
    #             mapping[positives] = torch.arange(len(positives), device=self.device)

    #             anchor_embs = batch[src].x[positives]
    #             cur_projector = self.model.proj_map[link_type]
    #             anchor_embs = cur_projector(anchor_embs)

    #             positive_pairs, negative_pairs = self.hard_sampler.forward(copy(batch), link_type, node_embeddings, anchor_embs)
    #             positive_embs = node_embeddings[src][positive_pairs[0, :]]
    #             negative_embs = node_embeddings[src][negative_pairs[0, :]]

    #             assert (positive_pairs[1] != negative_pairs[1]).sum() == 0
    #             anchor_embs_indices = mapping[positive_pairs[1]]
    #             assert (anchor_embs_indices == -1).sum() == 0

    #             anchor_embs = anchor_embs[anchor_embs_indices]
    #             loss = self.loss_fn(anchor_embs, positive_embs, negative_embs)
    #             if f_loss is None:
    #                 f_loss = loss
    #             else:
    #                 f_loss += loss

    #     return f_loss

    def train_epoch(self) -> None:
        self.model.train()
        losses = []
        bar = tqdm(self.train_loader)

        for batched_graph in bar:
            self.optimizer.zero_grad()
            batched_graph.to(self.device)
            node_embeddings = self.model(batched_graph.x_dict, batched_graph.edge_index_dict)

            loss = self.compute_loss(batched_graph, node_embeddings)
            if loss is None:  # no func TODO fix sampling
                continue

            loss.backward()

            self.optimizer.step()

            loss = loss.cpu().item()
            losses.append(loss)

            avg_loss = sum(losses) / len(losses)
            bar.set_description(f"avg loss: {avg_loss}; loss: {loss}")



    ###----Validation functions below----###

    def classification_metrics(
        self,
        batch: HeteroData,
        node_embeddings: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Get tuple of predictions and targets."""
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
            cur_projector = self.model.proj_map[link_type]
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

    def validate_epoch(self, loader: DataLoader) -> dict[str, float]:
        """Validate given loader with classification metrics for provided in batch pairs.

        Return dict of classifcation metrics.
        """
        self.model.eval()
        bar = tqdm(loader)

        g_predictions = {}
        g_target = {}
        counter = {}
        with torch.no_grad():
            for batched_graph in bar:
                batched_graph.to(self.device)
                node_embeddings = self.model(batched_graph.x_dict, batched_graph.edge_index_dict)
                predictions, targets = self.classification_metrics(batched_graph, node_embeddings)
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

    def validate_retrieval_epoch(self, loader: DataLoader, k: int) -> dict[str, float]:
        """Compute retrieval metrics."""
        self.model.eval()
        bar = tqdm(loader)

        actual = []
        predicted = []
        with torch.no_grad():
            for batched_graph in bar:
                if "pairs" not in batched_graph:
                    continue
                batched_graph.to(self.device)
                node_embeddings = self.model(batched_graph.x_dict, batched_graph.edge_index_dict)
                # nodes for which we want to predict

                og_embs = batched_graph.init_embs
                pred_relevant = self.model.retrieve(
                    batched_graph,
                    node_embeddings["FUNCTION"],
                    og_embs,
                    batched_graph.init_embs_ptr,
                    k=k,
                )

                # get actual relevant nodes
                c_actual = [[] for _ in range(len(og_embs))]
                c_predicted = [[] for _ in range(len(og_embs))]
                for n, relevant_f in batched_graph.pairs.T:
                    c_actual[n].append(int(relevant_f))
                    c_predicted[n] = pred_relevant[n]

                actual.extend(c_actual)
                predicted.extend(c_predicted)

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

    def train(self):
        for epoch in range(300):
            print(f"--------Epoch {epoch}-------")
            print("Training")
            self.train_epoch()

            # print("Simple acc metrics")
            # class_metrics = self.validate_epoch(self.val_loader)
            # print(class_metrics)

            print("Retrieval metrics:")
            retrieval_metrics = self.validate_retrieval_epoch(self.val_loader, k=self.retrieve_k)
            print(retrieval_metrics)


def train():
    ds = TorchGraphDataset(
        root="data/torch_cache/repobench",
    )

    triplet_loss = TripletLoss(
        margin=1.0,
        p=2,
        swap=False,
        reduction="mean",
    )

    model = HeteroGraphSAGE(768, 768, 768, 3)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    trainer = Trainer(model=model, loss_fn=triplet_loss, dataset=ds, batch_size=10, optimizer=optimizer, retrieve_k=3)
    trainer.train()


if __name__ == "__main__":
    torch.manual_seed(0)
    train()
