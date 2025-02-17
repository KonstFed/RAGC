import json
from pathlib import Path
import networkx as nx

from reprocess.re_processors import (
    JsonConverter,
    GraphBuilder,
    CloneRepository,
    Compose,
    RegExpFinder,
)
from reprocess.re_container import ReContainer
from tempfile import TemporaryDirectory

from .common import BaseGraphParser

# TODO: проблема тут что если функция вызывает функцию класса.
# то он поставит ссылку до создания обьекта но не до вызова функции.
# и вообще call graph он не делает. Однако можно решить с помощью Code2Flow


def _add_edges(graph: nx.DiGraph, comp: dict) -> None:
    """Adds edges to the graph."""
    comp_id = comp["component_id"]
    graph.add_edge(comp["file_id"], comp_id, type="includes")

    for link_comp_id in comp["linked_component_ids"]:
        graph.add_edge(comp_id, link_comp_id, type="calls")

    comp_name_parts = comp["component_name"].split(".")

    for node, attr in graph.nodes(data=True):
        if attr["component_type"] == "file":
            continue
        if not attr["component_name"].startswith(comp["component_name"]):
            continue

        c_atrrs = attr["component_name"].split(".")
        if len(c_atrrs) != len(comp_name_parts) + 1:
            continue

        graph.add_edge(comp_id, node, type="includes")


def _process_graph(graph: dict) -> nx.DiGraph:
    processed_graph = nx.DiGraph()

    for file in graph["files"]:
        processed_graph.add_node(
            file["file_id"],
            component_type="file",
            **{k: v for k, v in file.items() if k != "__class__"},
        )

    for c_comp in graph["code_components"]:
        processed_graph.add_node(c_comp["component_id"], **c_comp)

    for c_comp in graph["code_components"]:
        _add_edges(processed_graph, c_comp)

    return processed_graph


class ReprocessParser(BaseGraphParser):
    def __init__(self):
        composition_list = [
            GraphBuilder(),
            JsonConverter(),
        ]

        self.composition = Compose(composition_list)

    def parse(self, repo_path: Path) -> nx.DiGraph:
        with TemporaryDirectory() as tmp_p:
            container = ReContainer(
                repo_path.name,
                str(repo_path),
                tmp_p,
            )
            self.composition(container)

            with (Path(tmp_p) / repo_path.name / "data.json").open("r") as f:
                graph = json.load(f)

        return _process_graph(graph=graph)


class ReprocessFileParser(BaseGraphParser):
    def parse(repo_path) -> nx.DiGraph:
        general_graph = super().parse()

