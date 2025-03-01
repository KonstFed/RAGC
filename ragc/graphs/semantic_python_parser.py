from tempfile import TemporaryDirectory
from pathlib import Path

import networkx as nx
import plotly.graph_objects as go
from pydantic import parse_obj_as

from semantic_parser import SemanticGraphBuilder
from ragc.graphs.common import (
    Node,
    Edge,
    NodeType,
    EdgeType,
    BaseGraphParser,
)


class SemanticParser(BaseGraphParser):
    color2class: dict[str, str] = {
        "green": "file",
        "blue": "class",
        "orange": "function",
    }

    node2type: dict[str, str] = {
        "file": NodeType.FILE,
        "class": NodeType.CLASS,
        "function": NodeType.FUNCTION,
    }

    edge2type: dict[str, str] = {
        "Encapsulation": EdgeType.OWNER,
        "Invoke": EdgeType.CALL,
        "Import": EdgeType.IMPORT,
        "Ownership": EdgeType.OWNER,
        "Class Hierarchy": EdgeType.INHERITED,
    }

    def __init__(self):
        self._builder = SemanticGraphBuilder()

    def _process(self, graph: nx.MultiDiGraph, repo_path: Path) -> nx.MultiDiGraph:
        new_types = {
            node: self.color2class[attr["color"]]
            for node, attr in graph.nodes(data=True)
        }

        nx.set_node_attributes(graph, new_types, "type")

        file_nodes: list[str] = [
            node for node, attr in graph.nodes(data=True) if attr["type"] == "file"
        ]

        new_values = {}
        for node in file_nodes:
            attr = graph.nodes(data=True)[node]
            with (repo_path / attr["file_path"]).open("r") as f:
                file_code = f.read()

            if "body" in graph.nodes(data=True)[node]:
                raise ValueError(graph.nodes(data=True)[node])

            new_values[node] = file_code

        nx.set_node_attributes(graph, new_values, "body")

        return graph

    def _relabel(self, graph: nx.MultiDiGraph, repo_path: Path) -> nx.MultiDiGraph:
        mapping = {}
        file_path_map = {}
        for node in graph.nodes:
            if "C." != node[:2]:
                _err = f"All nodes should start form 'C.'. Got instead {node}"
                raise ValueError(_err)
            new_node = node.removeprefix("C.")
            new_node = Path(new_node).relative_to(repo_path)

            # remove .py and extract file_path
            parts = list(new_node.parts)
            filename = list(filter(lambda p: p[1].endswith(".py"), enumerate(parts)))
            if len(filename) != 1:
                raise ValueError(f"incorrect path {new_node}")

            idx, filename = filename[0]
            file_path = Path(*parts[: idx + 1])

            parts[idx] = parts[idx].removesuffix(".py")

            mapping[node] = ".".join(parts)
            file_path_map[node] = file_path

        nx.set_node_attributes(graph, file_path_map, "file_path")
        nx.relabel_nodes(graph, mapping, copy=False)
        return graph
    
    def _fix_code_ident(self, graph: nx.MultiDiGraph) -> nx.MultiDiGraph:
        for _, attr in graph.nodes(data=True):
            if attr["type"] == "file":
                continue
            code = attr["body"]
            ident_pos = attr["start_point"][1]
            code = " " * ident_pos + code
            lines = code.splitlines()
            min_indent = min((len(line) - len(line.lstrip())) for line in lines if line.strip())
            stripped_lines = [line[min_indent:] if line.strip() else line for line in lines]
            code = "\n".join(stripped_lines)
            attr["body"] = code
        return graph

    def to_standart(self, graph: nx.MultiDiGraph) -> nx.MultiDiGraph:
        norm_graph = nx.MultiDiGraph()
        for node, attr in graph.nodes(data=True):
            new_node = Node(
                name=node,
                type=self.node2type[attr["type"]],
                code=attr["body"],
                file_path=attr["file_path"],
            )
            norm_graph.add_node(node, **new_node.model_dump())

        for n_from, n_to, edge_data in graph.edges(data=True):
            new_edge = Edge(type=self.edge2type[edge_data["type"]])
            norm_graph.add_edge(n_from, n_to, **new_edge.model_dump())
        return norm_graph

    def parse(self, repo_path: Path) -> nx.MultiDiGraph:
        repo_path = repo_path.absolute()

        with TemporaryDirectory() as t:
            self._builder.build_from_one(str(repo_path), t, gsave=True, gprint=False)

            t = Path(t)
            graph_file = next(t.iterdir())
            graph = nx.read_gml(graph_file)

        graph = self._relabel(graph=graph, repo_path=repo_path)
        graph = self._process(graph=graph, repo_path=repo_path)
        graph = self._fix_code_ident(graph=graph)
        graph = self.to_standart(graph=graph)
        return graph

    def parse_into_files(self, repo_path: Path) -> nx.MultiDiGraph:
        graph = self.parse(repo_path=repo_path)
        file_nodes = [n for n, attr in graph.nodes(data=True) if attr["type"] == NodeType.FILE]
        return graph.subgraph(file_nodes).copy()


def draw_graph(graph: nx.MultiDiGraph, seed=2243324):
    node_color_map = {
        "class": "blue",
        "function": "orange",
        "file": "green",
    }
    edge_color_map = {
        "Import": "red",
        "Encapsulation": "blue",
        "Invoke": "green",
        "Ownership": "yellow",
    }
    g_draw = graph.copy()
    # Get positions for layout
    pos = nx.spring_layout(g_draw, seed=seed)

    # Create node traces
    node_x, node_y, node_colors, node_labels = [], [], [], []
    for node, (x, y) in pos.items():
        node_x.append(x)
        node_y.append(y)
        node_colors.append(
            node_color_map[g_draw.nodes[node]["type"]]
        )  # Color based on type
        node_labels.append(f"Node {node} (Type {g_draw.nodes[node]['type']})")  # Label

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        text=node_labels,
        marker=dict(size=20, color=node_colors, line=dict(width=2, color="black")),
        textposition="top center",
    )

    # Create edge traces
    edge_traces = []
    for u, v in g_draw.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        for _, t in g_draw[u][v].items():
            edge_color = edge_color_map[t["type"]]
            edge_traces.append(
                go.Scatter(
                    x=[x0, x1],
                    y=[y0, y1],
                    mode="lines",
                    line=dict(width=2, color=edge_color),
                    hoverinfo="text",
                    text=f"Edge {u}-{v} (type: {t['type']})",
                )
            )

    # Combine all traces
    fig = go.Figure(data=edge_traces + [node_trace])

    # Layout adjustments
    fig.update_layout(
        title="Interactive Network Graph with Plotly",
        showlegend=False,  # We will add a manual legend
        hovermode="closest",
        margin=dict(b=20, l=20, r=20, t=40),
        height=1000,
        width=1000,
    )

    # Add legend manually
    legend_items = [
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker=dict(size=15, color=color),
            name=f"Type {typ}",
        )
        for typ, color in node_color_map.items()
    ] + [
        go.Scatter(
            x=[None],
            y=[None],
            mode="lines",
            line=dict(width=2, color=color),
            name=f"Weight {weight}",
        )
        for weight, color in edge_color_map.items()
    ]

    fig.add_traces(legend_items)

    # Show the interactive graph
    fig.show()
