import json
from pathlib import Path
import networkx as nx
from collections import deque

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
# не распознаёт async def


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


    @staticmethod
    def to_file_only(graph: nx.DiGraph):


        file_only_g = nx.DiGraph()


        for node, attr in graph.nodes(data=True):
            if attr["component_type"] != "file":
                continue
            file_only_g.add_node(node, **attr)


        for node, attr in list(file_only_g.nodes(data=True)):

            q = deque(graph.successors(node))
            visited = set(graph.successors(node))
            while q:
                cur_comp = q.popleft()
                
                for succ_comp in graph.successors(cur_comp):
                    edge_data = graph.get_edge_data(cur_comp, succ_comp)
                    if edge_data["type"] != "includes":
                        continue
                    if succ_comp in visited:
                        continue
                    q.append(succ_comp)
                    visited.add(succ_comp)


            related_files = set()

            for visited_node in visited:
                for called in graph.successors(visited_node):
                    # print(graph.nodes(data=True)[called])
                    attr = graph.nodes(data=True)[called]
                    file_id = attr["file_id"]
                    if not graph.has_node(file_id):
                        print("AAAAA")
                    
                    related_files.add(file_id)
            # print(related_files)
            if "component_type" not in attr:
                print(attr)
            for rel_f in related_files:
                if not file_only_g.has_node(rel_f):
                    raise ValueError
                    print(rel_f)
                file_only_g.add_edge(node, rel_f, type="calls")

        return file_only_g

    def parse(repo_path) -> nx.DiGraph:
        general_graph = super().parse()
        return ReprocessFileParser.to_file_only(general_graph)
