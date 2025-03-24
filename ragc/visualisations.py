from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
from pyvis.network import Network

from ragc.graphs.common import EdgeType, NodeType


def plot_graph(semantic_graph: nx.MultiDiGraph, save_path: Path | str, plot: bool = False) -> None:
    """Draws and saves semantic graph to png. In addition calls matplotlib if needed"""
    graph = deepcopy(semantic_graph)
    save_path = Path(save_path)
    # Extract node attributes (color, nesting) for visualization
    COLOR_SCHEMA = {
        NodeType.CLASS: "blue",
        NodeType.FILE: "green",
        NodeType.FUNCTION: "orange",
    }

    EDGES_COLORS = {
        EdgeType.OWNER: "gold",
        EdgeType.IMPORT: "red",
        EdgeType.CALL: "green",
        EdgeType.INHERITED: "pink",
    }

    # Edge styles for the graph
    EDGES_STYLES = {
        EdgeType.OWNER: "bold",
        EdgeType.IMPORT: "dashed",
        EdgeType.CALL: "bold",
        EdgeType.INHERITED: "dashed",
    }

    node_colors = {n: COLOR_SCHEMA[attr["type"]] for n, attr in graph.nodes(data=True)}
    # nx.set_node_attributes(graph, color_map, "color")
    # node_colors = nx.get_node_attributes(graph, 'color'

    for n, attr in graph.nodes(data=True):
        graph.nodes(data=True)[n].pop("name")
    # Create a DOT (graph description) representation
    dot = nx.nx_pydot.to_pydot(graph)  # Coxnvert the graph to a DOT format

    # Set node styles, colors, and labels
    for node in graph.nodes():
        pydot_node = dot.get_node(str(node))[0]  # Get the corresponding node in the DOT representation
        pydot_node.set_fillcolor(node_colors.get(node, "white"))  # Set fill color, default to white
        pydot_node.set_style("filled")  # Make the node visually filled
        pydot_node.set_label(node.split("/")[-1])  # Label with the last path segment

    # Set edge labels and styles
    for edge in graph.edges(keys=True):
        source, target, key = edge  # Extract source, target, and key for MultiDiGraph edges
        edge_type = str(graph.edges[source, target, key].get("type", "Unknown"))  # Get the edge type
        edge_color = EDGES_COLORS.get(edge_type, "black")  # Get the edge color based on the type
        edge_style = EDGES_STYLES.get(edge_type, "solid")  # Get the edge style based on the type

        # Get the corresponding edge in the DOT representation
        pydot_edge = dot.get_edge(str(source), str(target))[key]

        # Apply attributes to the edge
        pydot_edge.set_label(edge_type)  # Set edge label to indicate the type
        pydot_edge.set_style(edge_style)  # Set style for the edge
        pydot_edge.set_color(edge_color)  # Set color for the edge

    # Save the resulting graph visualization as a PNG image
    dot.write_png(save_path)  # Write the DOT representation to a PNG file
    if plot:
        img = plt.imread(save_path)
        plt.imshow(img)


def plot_interactive(graph: nx.MultiDiGraph, save_path: Path | str) -> None:
    # Create a working copy without modifying original graph
    draw_g = deepcopy(graph)

    # NODE_COLORS = {
    #     NodeType.CLASS: "blue",
    #     NodeType.FILE: "green",
    #     NodeType.FUNCTION: "orange",
    # }

    # EDGE_COLORS = {
    #     EdgeType.OWNER: "gold",
    #     EdgeType.IMPORT: "red",
    #     EdgeType.CALL: "green",
    #     EdgeType.INHERITED: "pink",
    # }

    NODE_COLORS = {
        NodeType.CLASS: "#FF6B6B",  # Red
        NodeType.FUNCTION: "#4D96FF",  # Blue
        NodeType.FILE: "#6BCB77",  # Green
    }

    EDGE_COLORS = {
        EdgeType.IMPORT: "#FFD93D",  # Yellow
        EdgeType.OWNER: "#8E44AD",   # Darker Purple (more distinct)
        EdgeType.CALL: "#3498DB",    # Bright Blue (high contrast with purple)
        EdgeType.INHERITED: "#FF69B4",  # Pink
    }

    # Apply node styling
    node_name_map = {}
    for node, data in draw_g.nodes(data=True):
        del data["name"]
        del data["code"]
        file_path: Path = data.pop("file_path")
        new_name = node.removeprefix(".".join(file_path.parts).removesuffix(".py") + ".")
        # node_name_map[node] = new_name

        node_type = data.pop("type")
        data.update({"color": NODE_COLORS.get(node_type, "#999999"), "shape": "box", "font": {"size": 14}})

    # Apply edge styling
    for u, v, key, data in draw_g.edges(keys=True, data=True):
        edge_type = EdgeType(data.pop("type"))
        data.update({"color": EDGE_COLORS.get(edge_type, "#CCCCCC"), "arrows": "to", "width": 2})

    draw_g = nx.relabel_nodes(draw_g, node_name_map)

    # Generate visualization
    net = Network(
        notebook=False,
        directed=True,
        cdn_resources="in_line",
        bgcolor="#FFFFFF",
        height="800px",
        width="100%",
    )
    net.from_nx(draw_g)
    net.show(save_path, notebook=False)
