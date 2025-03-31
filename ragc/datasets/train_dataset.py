import copy

import networkx as nx
import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.transforms import BaseTransform

from ragc.graphs.common import NodeTypeNumeric
from ragc.graphs.transforms import graph2pyg
from ragc.llm.embedding import BaseEmbedder


class Embed(BaseTransform):
    """Embeddes all nodes except Files"""

    def __init__(self, embedder: BaseEmbedder):
        self.embedder = embedder
        super().__init__()

    def __call__(self, data: Data) -> Data:
        data_c = copy.copy(data)
        not_file_mask = data_c.type != NodeTypeNumeric.FILE.value
        all_code = [data_c.code[i] for i in torch.where(not_file_mask)[0]]
        embeddings = self.embedder.embed(all_code)
        # embeddings should be 2d tensor
        node_embeddings = torch.zeros(data.num_nodes, embeddings.shape[1])
        node_embeddings[not_file_mask] = embeddings

        data_c.x = node_embeddings
        return data_c

class TorchGraphDataset(InMemoryDataset):
    def __init__(self, root: str, graphs: list[nx.MultiDiGraph], transform=None, pre_transform=None, pre_filter=None):
        self._graphs = graphs
        super().__init__(root, transform, pre_transform, pre_filter)
        self.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return ["original_graph.pt"]

    def process(self) -> None:
        data_list = []
        for graph in self._graphs:
            pyg_graph = graph2pyg(graph)
            data_list.append(pyg_graph)

        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]

        if self.pre_filter is not None:
            data_list = list(filter(self.pre_filter, data_list))

        # Save the processed data
        self.save(data_list, self.processed_paths[0])
