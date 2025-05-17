import copy
from abc import ABC, abstractmethod
from typing import Literal

import networkx as nx
import torch
from pydantic import BaseModel
from torch_geometric.data import Data, HeteroData
from torch_geometric.transforms import BaseTransform

from ragc.graphs.common import NodeTypeNumeric
from ragc.graphs.utils import apply_mask, get_callee_mask, graph2pyg, mask_node
from ragc.llm.embedding import BaseEmbedder
from ragc.llm.types import EmbedderConfig


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


class EmbedTransform(BaseTransform):
    """Embeddes all nodes except Files."""

    def __init__(self, embedder: BaseEmbedder, embed_docstring:bool):
        self.embedder = embedder
        self.embed_docstring = embed_docstring
        super().__init__()

    def emb_docstring(self, data: Data) -> Data:
        # this mask state which nodes had meaningful doctring
        mask = [i for i, d in enumerate(data.docstring) if len(d) != 0]
        data.docstring_mask = torch.tensor(mask, dtype=torch.int64)

        # костыль чтобы знать размер эмбэдинга
        t = self.embedder.embed(["aaaa"])[0]

        docstring_embeddings = torch.zeros((data.num_nodes, t.shape[0]))
        docstrings = [d for d in data.docstring if len(d) != 0]
        if len(docstrings) != 0:
            embeddings = self.embedder.embed(docstrings).cpu()
            for i, g_i in enumerate(mask):
                docstring_embeddings[g_i] = embeddings[i]

        data.docstring_embeddings = docstring_embeddings
        return data

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

        if self.embed_docstring:
            data_c = self.emb_docstring(data_c)
        return data_c

class EmbedTransformConfig(BaseTransformConfig):
    type: Literal["embed_transform"] = "embed_transform"
    embedder: EmbedderConfig

    embed_docstring: bool

    def create(self) -> EmbedTransform:
        embedder = self.embedder.create()
        return EmbedTransform(embedder=embedder, embed_docstring=self.embed_docstring)



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


class MaskNodesConfig(BaseTransformConfig):
    type: Literal["mask_node"] = "mask_node"
    node2mask: list[str]
    mask_callee: bool = True

    def create(self) -> MaskNodes:
        return MaskNodes(self.node2mask, mask_callee=self.mask_callee)
