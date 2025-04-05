import warnings
from pathlib import Path

import networkx as nx
from pydantic import BaseModel, model_validator
from torch_geometric.data import InMemoryDataset
from torch_geometric.transforms import Compose
from tqdm import tqdm

from ragc.graphs import GraphParserConfig
from ragc.graphs.transforms import Transform


class TorchGraphDataset(InMemoryDataset):
    """Dataset for training GNN."""

    def __init__(self, root: str, graphs: list[nx.MultiDiGraph], transform=None, pre_transform=None, pre_filter=None):
        self._graphs = graphs
        super().__init__(root, transform, pre_transform, pre_filter)
        self.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return ["original_graph.pt"]

    def process(self) -> None:
        data_list = self._graphs

        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in tqdm(data_list)]

        if self.pre_filter is not None:
            data_list = list(filter(self.pre_filter, data_list))

        # Save the processed data
        self.save(data_list, self.processed_paths[0])


class TorchGraphDatasetConfig(BaseModel):
    """Config for torch_geometric dataset."""

    root_path: Path
    repos_path: Path | None = None
    graphs_path: Path | None = None

    parser: GraphParserConfig | None = None

    pre_transform: list[Transform] | None = None
    transform: list[Transform] | None = None

    @model_validator(mode="after")
    def _validate_general(self):
        if self.repos_path is not None and self.graphs_path is not None:
            raise ValueError("If you want to init graphs you should provide either repos_path or graphs_path not both")

        if self.repos_path is not None and self.parser is None:
            raise ValueError("parser should be provided if repos_path is provided")

        return self

    def _parse_graphs(self) -> list[nx.MultiDiGraph]:
        """Parse graphs from repo."""
        parser = self.parser.create()
        graphs = []
        for repo_p in self.repos_path.iterdir():
            try:
                graph = parser.parse(repo_path=repo_p)
                graphs.append(graph)
            except Exception as e:
                _wrn_msg = f"Failed to parse {repo_p.absolute()} with error:\n{e}"
                warnings.warn(_wrn_msg, stacklevel=1)
        return graphs

    def _load_graphs(self) -> list[nx.MultiDiGraph]:
        """Read .gml graphs."""
        graphs = []
        for graph_p in self.graphs_path.iterdir():
            graph = nx.read_gml(graph_p)
            graphs.append(graph)
        return graphs

    def create(self) -> TorchGraphDataset:
        graphs = None
        if self.repos_path is not None:
            graphs = self._parse_graphs()
        elif self.graphs_path is not None:
            graphs = self._load_graphs()

        pre_transform = None
        if self.pre_transform is not None:
            pre_transform = Compose([t.create() for t in self.pre_transform])

        transform = None
        if self.transform is not None:
            transform = Compose([t.create() for t in self.transform])

        return TorchGraphDataset(
            root=self.root_path,
            graphs=graphs,
            transform=transform,
            pre_transform=pre_transform,
        )
