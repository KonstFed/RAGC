import torch
from torch_geometric.nn import HeteroConv, SAGEConv

from ragc.train.gnn.models.projector import Projector
import torch.nn.functional as F
from torch_geometric.data import Batch, HeteroData


class HeteroGraphSAGE(torch.nn.Module):
    """Directed graphsage."""

    def __init__(self, orig_emb_size: int, hidden_dim: int, out_channels: int, num_layers: int):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers):
            if i == num_layers - 1:
                layer_conv = self._init_conv_layer(hidden_dim=out_channels)
            else:
                layer_conv = self._init_conv_layer(hidden_dim=hidden_dim)

            # Create a SAGEConv for each edge type
            conv = HeteroConv(layer_conv, aggr="mean")
            self.convs.append(conv)

        self.proj_map = self._init_projectors(orig_emb_size=orig_emb_size, node_emb_size=out_channels)

    def _init_conv_layer(self, hidden_dim: int) -> dict[tuple[str, str, str], torch.nn.Module]:
        file_conv = SAGEConv((-1, -1), hidden_dim, aggr="mean")
        file_call_conv = SAGEConv((-1, -1), hidden_dim, "mean")
        own_conv = SAGEConv((-1, -1), hidden_dim, "mean")
        call_conv = SAGEConv((-1, -1), hidden_dim, "mean")

        block_map = {
            # file own relations
            ("FILE", "OWNER", "CLASS"): file_conv,
            ("FILE", "OWNER", "FUNCTION"): file_conv,
            # file calls
            ("FILE", "CALL", "FUNCTION"): file_call_conv,
            ("FILE", "IMPORT", "FILE"): file_call_conv,
            ("FILE", "IMPORT", "CLASS"): file_call_conv,
            ("FILE", "IMPORT", "FUNCTION"): file_call_conv,
            # own relations
            ("CLASS", "OWNER", "CLASS"): own_conv,
            ("CLASS", "OWNER", "FUNCTION"): own_conv,
            ("FUNCTION", "OWNER", "CLASS"): own_conv,
            ("FUNCTION", "OWNER", "FUNCTION"): own_conv,
            # call relations
            ("CLASS", "CALL", "FUNCTION"): call_conv,
            ("CLASS", "INHERITED", "CLASS"): call_conv,
            ("FUNCTION", "CALL", "FUNCTION"): call_conv,
        }
        return block_map

    def _init_projectors(self, orig_emb_size: int, node_emb_size: int) -> dict[tuple[str, str, str], torch.nn.Module]:
        self.file_own_proj = Projector(orig_emb_size, node_emb_size)
        self.file_call_proj = Projector(orig_emb_size, node_emb_size)
        self.own_call_proj = Projector(orig_emb_size, node_emb_size)
        self.call_proj = Projector(orig_emb_size, node_emb_size)

        proj_map = {
            # file own relations
            ("FILE", "OWNER", "CLASS"): self.file_own_proj,
            ("FILE", "OWNER", "FUNCTION"): self.file_own_proj,
            # file calls
            ("FILE", "CALL", "FUNCTION"): self.file_call_proj,
            ("FILE", "IMPORT", "FILE"): self.file_call_proj,
            ("FILE", "IMPORT", "CLASS"): self.file_call_proj,
            ("FILE", "IMPORT", "FUNCTION"): self.file_call_proj,
            # own relations
            ("CLASS", "OWNER", "CLASS"): self.own_call_proj,
            ("CLASS", "OWNER", "FUNCTION"): self.own_call_proj,
            ("FUNCTION", "OWNER", "CLASS"): self.own_call_proj,
            ("FUNCTION", "OWNER", "FUNCTION"): self.own_call_proj,
            # call relations
            ("CLASS", "CALL", "FUNCTION"): self.call_proj,
            ("CLASS", "INHERITED", "CLASS"): self.call_proj,
            ("FUNCTION", "CALL", "FUNCTION"): self.call_proj,
        }
        return proj_map

    def forward(self, x, edge_dict) -> dict[str, torch.Tensor]:
        for conv in self.convs:
            # Process all edge types and update node features
            x_dict = conv(x, edge_dict)
            # Apply ReLU to all node types
            x_dict = {key: x.relu() for key, x in x_dict.items()}

        return x_dict

    def retrieve_single(
        self,
        link_type: tuple[str, str, str],
        query: torch.Tensor,
        node_embeddings: torch.Tensor,
        k: int,
    ) -> torch.Tensor:
        projected = self.proj_map[link_type](query)
        projected = F.normalize(projected, dim=0)
        node_embeddings = F.normalize(node_embeddings, dim=1)
        cos_sim = torch.matmul(projected, node_embeddings.T)
        _values, indices = cos_sim.topk(k=min(k, node_embeddings.shape[0]))
        return indices

    def retrieve(
        self,
        batch: Batch,
        fund_node_embeddings: torch.Tensor,
        embeddings: torch.Tensor,
        emb_ptr: torch.Tensor,
        k: int,
        # projected_ptr: torch.Tensor,
        # node_ptr: torch.Tensor,
    ) -> list[list[int]]:
        relation_type = ("FUNCTION", "CALL", "FUNCTION")
        node_ptr = batch["FUNCTION"].ptr

        projected_embs = self.proj_map[relation_type](embeddings)

        # l2 normalize for cos distance
        projected_embs = F.normalize(projected_embs, p=2, dim=1)
        node_embs = F.normalize(fund_node_embeddings, dim=1, p=2)

        out = []
        for i, (pr_ptr, c_node_ptr) in enumerate(zip(emb_ptr.tolist()[:-1], node_ptr.tolist()[:-1], strict=True)):
            pr_end = emb_ptr[i + 1]
            cur_proj_embs = projected_embs[pr_ptr:pr_end]

            node_end = node_ptr[i + 1]
            cur_node_embs = node_embs[c_node_ptr:node_end]

            cosine_distances = cur_proj_embs @ cur_node_embs.T
            cur_k = min(k, cosine_distances.shape[1])
            _values, indices = torch.topk(cosine_distances, k=cur_k, dim=1)
            indices += c_node_ptr
            out.extend(indices.tolist())

        return out
