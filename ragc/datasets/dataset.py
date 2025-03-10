import shutil
import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import networkx as nx
from pydantic import BaseModel
from tqdm import tqdm

from ragc.graphs import BaseGraphParser, read_graph, save_graph
from ragc.graphs.common import BaseGraphParserConfig
from ragc.retrieval.common import BaseRetievalConfig, BaseRetrieval
from ragc.inference import InferenceConfig, Inference


class AbstractCacheDataset(ABC):
    elements_cache_p: list[Path]
    elements: list[BaseModel]

    def __init__(self, cache_path: Path, config: BaseModel, in_memory: bool = False):
        self.cache_path = cache_path
        self.in_memory = in_memory
        self.config_template = config
        self.elements_cache_p = []
        self.elements = []
        self.load()

    def get_repo_id(self, repo_path: Path) -> str:
        return repo_path.name

    def add(self, repo_paths: list[Path], progress_bar: bool = True) -> list[bool]:
        bar = tqdm(repo_paths) if progress_bar else repo_paths

        successful_repos = []
        for repo_p in bar:
            repo_cache_path = self.cache_path / repo_p.name
            repo_cache_path.mkdir(exist_ok=True)
            try:
                self.add_single_repo(repo_path=repo_p, repo_cache_path=repo_cache_path)
                result = True
            except Exception as e:
                _wrn_msg = f"Failed add {repo_p}. Error {type(e).__name__}:\n{e}"
                warnings.warn(_wrn_msg)

                shutil.rmtree(repo_cache_path)
                result = False

            successful_repos.append(result)

        return successful_repos

    def load(self) -> None:
        _all_folders = [p for p in self.cache_path.iterdir() if p.is_dir()]
        for repo_cache_path in _all_folders:
            if not self.check_cache(repo_cache_path):
                # cache is bad
                _wrn_msg = f"Cache for {repo_cache_path.name} is bad. Thus, skipping it"
                warnings.warn(_wrn_msg)
                continue
            self.elements_cache_p.append(repo_cache_path)

            if self.in_memory:
                loaded = self.load_single_repo(repo_cache_path=repo_cache_path)
                self.elements.append(loaded)

    @abstractmethod
    def create_config(self, cache_path: Path) -> BaseModel:
        """Changes cache path in config template."""

    @abstractmethod
    def check_cache(self, repo_cache_path: Path) -> bool:
        """Check if cache is correct"""

    @abstractmethod
    def load_single_repo(self, repo_cache_path: Path) -> dict[str, Any]:
        """Load repo."""

    @abstractmethod
    def add_single_repo(self, repo_path: Path, repo_cache_path: Path) -> dict[str, Any]:
        """Add repository to cache."""

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if self.in_memory:
            return self.elements[idx]

        return self.load_single_repo(repo_cache_path=self.elements_cache_p[idx])

    def __len__(self) -> int:
        return len(self.elements_cache_p)


class GraphDataset(AbstractCacheDataset):
    def __init__(self, cache_path: Path, parser_cfg: BaseGraphParserConfig, in_memory: bool = False):
        super().__init__(cache_path=cache_path, config=parser_cfg, in_memory=in_memory)

    def create_config(self, cache_path: Path) -> BaseGraphParserConfig:
        new_cfg = self.config_template.model_copy(update={"cache_path": cache_path / f"{cache_path.name}.gml"})
        return new_cfg

    def add_single_repo(self, repo_path: Path, repo_cache_path: Path) -> nx.MultiDiGraph:
        """Add repository to cache."""
        cur_parser_cfg = self.create_config(cache_path=repo_cache_path)
        parser = cur_parser_cfg.create()
        graph = parser.parse(repo_path=repo_path)
        return graph

    def check_cache(self, repo_cache_path: Path) -> bool:
        """Check if cache is correct."""
        cached_repo_p = repo_cache_path / f"{repo_cache_path.name}.gml"
        return cached_repo_p.exists()

    def load_single_repo(self, repo_cache_path: Path) -> dict[str, nx.MultiDiGraph]:
        """Load repo."""
        graph = read_graph(repo_cache_path / f"{repo_cache_path.name}.gml")
        return {"graph": graph}


class RetrievalDataset(GraphDataset):
    def __init__(
        self,
        cache_path: Path,
        parser_cfg: BaseGraphParserConfig,
        retrieval_cfg: BaseRetievalConfig,
        in_memory: bool = False,
    ):
        self.retrieval_cfg = retrieval_cfg
        super().__init__(cache_path=cache_path, parser_cfg=parser_cfg, in_memory=in_memory)

    def create_config(self, cache_path: Path) -> tuple[BaseGraphParserConfig, BaseRetievalConfig]:
        graph_cfg = super().create_config(cache_path)
        retrieval_cfg = self.retrieval_cfg.model_copy(
            update={"cache_index_path": cache_path / f"{cache_path.name}.npy"},
        )
        return graph_cfg, retrieval_cfg

    def add_single_repo(self, repo_path: Path, repo_cache_path: Path) -> BaseRetrieval:
        graph_cfg, retrieval_cfg = self.create_config(cache_path=repo_cache_path)
        parser = graph_cfg.create()
        graph = parser.parse(repo_path=repo_path)
        retrieval = retrieval_cfg.create(graph=graph)
        return retrieval

    def check_cache(self, repo_cache_path: Path) -> bool:
        """Check if cache is correct."""
        sup_res = super().check_cache(repo_cache_path)
        if not sup_res:
            return False

        cache_index_path = repo_cache_path / f"{repo_cache_path.name}.npy"
        return cache_index_path.exists()

    def load_single_repo(self, repo_cache_path: Path) -> dict[str, nx.MultiDiGraph | BaseRetrieval]:
        super_data = super().load_single_repo(repo_cache_path)

        cache_index_path = repo_cache_path / f"{repo_cache_path.name}.npy"  # maybe not .npy in future

        cur_cfg = self.retrieval_cfg.model_copy(update={"cache_index_path": cache_index_path})
        retrieval = cur_cfg.create(graph=super_data["graph"])

        super_data["retrieval"] = retrieval
        return super_data


class InferenceDataset(RetrievalDataset):
    def __init__(self, cache_path: Path, inference_cfg: InferenceConfig, in_memory=False):
        self.inference_cfg = inference_cfg
        super().__init__(
            cache_path, parser_cfg=inference_cfg.parser, retrieval_cfg=inference_cfg.retrieval, in_memory=in_memory
        )

    def create_config(self, cache_path: Path) -> InferenceConfig:
        parser_cfg, retrieval_cfg = super().create_config(cache_path)
        return self.inference_cfg.model_copy(update={"parser": parser_cfg, "retrieval": retrieval_cfg})

    def load_single_repo(self, repo_cache_path: Path) -> Inference:
        inference_cfg: InferenceConfig = self.create_config(cache_path=repo_cache_path)
        return inference_cfg.create()
