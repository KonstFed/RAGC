from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import Annotated

import networkx as nx
from pydantic import BaseModel, ConfigDict, Field, field_validator


class BaseGraphParser(ABC):
    @abstractmethod
    def parse(self, repo_path: Path) -> nx.MultiDiGraph:
        raise NotImplementedError

    @abstractmethod
    def parse_into_files(self, repo_path: Path) -> nx.MultiDiGraph:
        raise NotImplementedError

    @abstractmethod
    def to_standart(self, graph: nx.MultiDiGraph) -> nx.MultiDiGraph:
        raise NotImplementedError


class NodeType(Enum):
    CLASS = "class"
    FUNCTION = "function"
    FILE = "file"

    def __str__(self):
        return self.value


class EdgeType(Enum):
    IMPORT = "import"
    OWNER = "owner"
    CALL = "call"
    INHERITED = "inherited"

    def __str__(self):
        return self.value


class Node(BaseModel):
    name: Annotated[str, Field(pattern=r"^[A-Za-z0-9._-]+$")]
    type: NodeType
    code: str
    file_path: Path

    @field_validator("file_path", mode="before")
    @classmethod
    def _normalize_file_path(cls, file_p: str) -> Path:
        return Path(file_p)

    model_config = ConfigDict(
        extra="ignore",
        frozen=True,
    )


class Edge(BaseModel):
    type: EdgeType

    model_config = ConfigDict(
        extra="ignore",
        frozen=True,
    )


def save_graph(graph: nx.Graph, save_path: Path) -> None:
    nx.write_gml(graph, save_path, stringizer=str)


def read_graph(read_path: Path) -> nx.Graph:
    graph = nx.read_gml(read_path)
    for _, attr in graph.nodes(data=True):
        attr["type"] = NodeType(attr["type"])
        attr["file_path"] = Path(attr["file_path"])
    return graph
