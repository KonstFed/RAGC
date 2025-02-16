import json
from pathlib import Path
import networkx as nx

# TODO: проблема тут что если функция вызывает функцию класса.
# то он поставит ссылку до создания обьекта но не до вызова функции.
# и вообще call graph он не делает. Однако можно решить с помощью Code2Flow

def _add_edges(graph: nx.MultiDiGraph, comp: dict) -> None:
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


def process_graph(graph: dict) -> nx.MultiDiGraph:
    processed_graph = nx.MultiDiGraph()

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


def read_and_process(graph_path: str | Path) -> nx.MultiDiGraph:
    graph_path = Path(graph_path)
    with graph_path.open("r") as f:
        graph = json.load(f)

    return process_graph(graph)
