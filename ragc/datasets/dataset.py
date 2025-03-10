import warnings
import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import networkx as nx
from tqdm import tqdm

from ragc.graphs import BaseGraphParser, read_graph, save_graph
from ragc.retrieval.common import BaseRetievalConfig


class AbstractCacheDataset(ABC):
    def __init__(self, cache_path: Path, in_memory: bool = False):
        self.cache_path = cache_path
        self.in_memory = in_memory
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
    def __init__(self, cache_path: Path, parser: BaseGraphParser, in_memory: bool = False):
        self.parser = parser
        super().__init__(cache_path, in_memory=in_memory)

    def add_single_repo(self, repo_path: Path, repo_cache_path: Path) -> nx.MultiDiGraph:
        """Add repository to cache."""
        graph = self.parser.parse(repo_path=repo_path)
        save_graph(graph=graph, save_path=repo_cache_path / f"{self.get_repo_id(repo_path)}.gml")
        return {"graph": graph}

    def check_cache(self, repo_cache_path: Path) -> bool:
        """Check if cache is correct"""
        cached_repo_p = repo_cache_path / f"{repo_cache_path.name}.gml"
        return cached_repo_p.exists()

    def load_single_repo(self, repo_cache_path: Path) -> nx.MultiDiGraph:
        """Load repo."""
        graph = read_graph(repo_cache_path / f"{repo_cache_path.name}.gml")
        return {"graph": graph}


class RetrievalDataset(GraphDataset):
    def __init__(
        self,
        cache_path: Path,
        parser: BaseGraphParser,
        retrieval_cfg: BaseRetievalConfig,
        in_memory: bool = False,
    ):
        self.retrieval_cfg = retrieval_cfg
        super().__init__(cache_path, parser, in_memory=in_memory)

    def add_single_repo(self, repo_path: Path, repo_cache_path: Path) -> dict[str, Any]:
        super_data = super().add_single_repo(repo_path, repo_cache_path)

        cache_index_path = repo_cache_path / f"{self.get_repo_id(repo_path)}.npy"  # maybe not .npy in future

        cur_cfg = self.retrieval_cfg.model_copy(update={"cache_index_path": cache_index_path})
        retrieval = cur_cfg.create(graph=super_data["graph"])

        super_data["retrieval"] = retrieval
        return super_data

    def check_cache(self, repo_cache_path: Path) -> bool:
        """Check if cache is correct"""
        sup_res = super().check_cache(repo_cache_path)
        if not sup_res:
            return False

        cache_index_path = repo_cache_path / f"{repo_cache_path.name}.npy"
        return cache_index_path.exists()

    def load_single_repo(self, repo_cache_path: Path) -> dict[str, Any]:
        super_data = super().load_single_repo(repo_cache_path)

        cache_index_path = repo_cache_path / f"{repo_cache_path.name}.npy"  # maybe not .npy in future

        cur_cfg = self.retrieval_cfg.model_copy(update={"cache_index_path": cache_index_path})
        retrieval = cur_cfg.create(graph=super_data["graph"])

        super_data["retrieval"] = retrieval
        return super_data


class InferenceDataset(RetrievalDataset):
    def __init__(self, cache_path, parser, retrieval_cfg, in_memory = False):
        super().__init__(cache_path, parser, retrieval_cfg, in_memory)