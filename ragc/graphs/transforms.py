import copy
from abc import ABC, abstractmethod
from typing import Literal

import torch
from pydantic import BaseModel
import networkx as nx
from torch_geometric.data import Data, HeteroData
from torch_geometric.transforms import BaseTransform

from ragc.graphs.common import EdgeTypeNumeric, NodeTypeNumeric
from ragc.llm.embedding import BaseEmbederConfig, BaseEmbedder
from ragc.llm.types import EmbedderConfig
from ragc.graphs.utils import get_callee_mask, apply_mask, mask_node, graph2pyg

# For GNN which combinations of edges and nodes are allowed
ALLOWED_COMBINATIONS: list[tuple[NodeTypeNumeric, EdgeTypeNumeric, NodeTypeNumeric]] = []

# OWNERSHIP
ALLOWED_COMBINATIONS.extend(
    [
        (f, EdgeTypeNumeric.OWNER, s)
        for f in [NodeTypeNumeric.FILE, NodeTypeNumeric.CLASS, NodeTypeNumeric.FUNCTION]
        for s in [NodeTypeNumeric.CLASS, NodeTypeNumeric.FUNCTION]
    ],
)

# CALLS
ALLOWED_COMBINATIONS.extend(
    [
        (f, EdgeTypeNumeric.CALL, NodeTypeNumeric.FUNCTION)
        for f in [NodeTypeNumeric.FILE, NodeTypeNumeric.CLASS, NodeTypeNumeric.FUNCTION]
    ],
)

# IMPORTS
ALLOWED_COMBINATIONS.extend(
    [
        (NodeTypeNumeric.FILE, EdgeTypeNumeric.IMPORT, s)
        for s in [NodeTypeNumeric.FILE, NodeTypeNumeric.CLASS, NodeTypeNumeric.FUNCTION]
    ],
)

# INHERITANCE
ALLOWED_COMBINATIONS.append((NodeTypeNumeric.CLASS, EdgeTypeNumeric.INHERITED, NodeTypeNumeric.CLASS))


class BaseTransformConfig(BaseModel, ABC):
    """Config for transforms for graphs"""

    @abstractmethod
    def create(self) -> BaseTransform:
        raise NotImplementedError


class ToPYG(BaseTransform, BaseTransformConfig):
    type: Literal["to_pyg"] = "to_pyg"

    def forward(self, data: nx.MultiDiGraph) -> Data:
        return graph2pyg(data)

    def create(self):
        return self


class ToHetero(BaseTransform, BaseTransformConfig):
    type: Literal["to_hetero"] = "to_hetero"

    @classmethod
    def to_hetero(graph: Data) -> HeteroData:
        """Transform graph to HeteroData for simplier training."""
        h_graph = HeteroData()

        node_map_by_cat = {}

        for node_type in [NodeTypeNumeric.FILE, NodeTypeNumeric.CLASS, NodeTypeNumeric.FUNCTION]:
            node_mask = graph.type == node_type.value
            node_idx = torch.where(node_mask)[0]

            idx_map = -torch.ones_like(node_mask, dtype=torch.long)
            idx_map[node_idx] = torch.arange(len(node_idx), device=node_mask.device)
            node_map_by_cat[node_type] = idx_map

            h_graph[node_type.name].x = graph.x[node_mask]
            h_graph[node_type.name].signature = [graph.signature[i] for i in node_idx]
            h_graph[node_type.name].docstring = [graph.docstring[i] for i in node_idx]
            h_graph[node_type.name].name = [graph.name[i] for i in node_idx]

        for f, edge, s in ALLOWED_COMBINATIONS:
            f_types = graph.type[graph.edge_index[0, :]]
            f_mask = f_types == f.value
            s_types = graph.type[graph.edge_index[1, :]]
            s_mask = s_types == s.value

            edge_mask = (graph.edge_type == edge.value) & f_mask & s_mask

            cur_edge_index = graph.edge_index[:, edge_mask]
            cur_edge_index[0, :] = node_map_by_cat[f][cur_edge_index[0, :]]
            cur_edge_index[1, :] = node_map_by_cat[s][cur_edge_index[1, :]]

            h_graph[f.name, edge.name, s.name].edge_index = cur_edge_index

        return h_graph

    def forward(self, data: Data) -> HeteroData:
        return self.to_hetero(data)

    def create(self):
        return self


class EmbedTransform(BaseTransform):
    """Embeddes all nodes except Files."""

    def __init__(self, embedder: BaseEmbedder):
        self.embedder = embedder
        super().__init__()

    def forward(self, data: Data) -> Data:
        data_c = copy.copy(data)
        # SUGGESTION: maybe not ignore file, make a parameter
        not_file_mask = data_c.type != NodeTypeNumeric.FILE.value
        all_code = [data_c.code[i] for i in torch.where(not_file_mask)[0]]
        embeddings = self.embedder.embed(all_code).cpu()
        # embeddings should be 2d tensor
        node_embeddings = torch.zeros(data.num_nodes, embeddings.shape[1])
        node_embeddings[not_file_mask] = embeddings

        data_c.x = node_embeddings
        return data_c


class EmbedTransformConfig(BaseTransformConfig):
    type: Literal["embed_transform"] = "embed_transform"
    embedder: EmbedderConfig

    def create(self) -> EmbedTransform:
        embedder = self.embedder.create()
        return EmbedTransform(embedder=embedder)


class MaskNodes(BaseTransform):
    """Mask given nodes from the graph including all callees nodes."""

    def __init__(self, nodes2mask: list[str], mask_callee: bool):
        self.nodes2mask = nodes2mask
        self.mask_callee = mask_callee

    def _find_node(self, graph: Data, node: str) -> int:
        for i, name in enumerate(graph.name):
            if name == node:
                return i

        raise ValueError("Node not found")

    def forward(self, data: Data) -> Data:
        node_mask = None
        edge_mask = None
        for node_name in self.nodes2mask:
            node = self._find_node(graph=data, node=node_name)

            if self.mask_callee:
                cur_node_mask, cur_edge_mask = get_callee_mask(graph=data, node=node)
            else:
                cur_node_mask, cur_edge_mask = mask_node(graph=data, node=node)

            if node_mask is None:
                node_mask = cur_node_mask
                edge_mask = cur_edge_mask
            else:
                node_mask = node_mask & cur_node_mask
                edge_mask = edge_mask & cur_edge_mask

        if node_mask is None:
            print("ACHTUNG")
            return data

        return apply_mask(graph=data, node_mask=node_mask, edge_mask=edge_mask)


class MaskNodeskConfig(BaseTransformConfig):
    type: Literal["mask_node"] = "mask_node"
    node2mask: list[str]
    mask_callee: bool = True

    def create(self) -> MaskNodes:
        return MaskNodes(self.node2mask, mask_callee=self.mask_callee)


Transform = MaskNodeskConfig | EmbedTransformConfig | ToHetero | ToPYG