from pathlib import Path
from copy import deepcopy

import networkx as nx
import matplotlib.pyplot as plt

from ragc.graphs.common import Node, Edge, NodeType, EdgeType

def plot_graph(semantic_graph: nx.MultiDiGraph, save_path: Path | str, plot: bool = False) -> None:
    """Draws and saves semantic graph to png. In addition calls matplotlib if needed"""
    graph = deepcopy(semantic_graph)
    save_path = Path(save_path)
    # Extract node attributes (color, nesting) for visualization
    COLOR_SCHEMA = {
        NodeType.CLASS: "blue",
        NodeType.FILE: "green",
        NodeType.FUNCTION: "orange"
    }

    EDGES_COLORS = {
        EdgeType.OWNER: 'gold',
        EdgeType.IMPORT: 'red',
        EdgeType.CALL: 'green',
        EdgeType.INHERITED: 'pink'
    }

    # Edge styles for the graph
    EDGES_STYLES = {
        EdgeType.OWNER: 'bold',
        EdgeType.IMPORT: 'dashed',
        EdgeType.CALL: 'bold',
        EdgeType.INHERITED: 'dashed'
    }


    node_colors = {n:COLOR_SCHEMA[attr["type"]] for n, attr in graph.nodes(data=True)}
    # nx.set_node_attributes(graph, color_map, "color")
    # node_colors = nx.get_node_attributes(graph, 'color'

    for n, attr in graph.nodes(data=True):
        graph.nodes(data=True)[n].pop("name")
    # Create a DOT (graph description) representation
    dot = nx.nx_pydot.to_pydot(graph)  # Coxnvert the graph to a DOT format

    # Set node styles, colors, and labels
    for node in graph.nodes():
        pydot_node = dot.get_node(str(node))[0]  # Get the corresponding node in the DOT representation
        pydot_node.set_fillcolor(node_colors.get(node, 'white'))  # Set fill color, default to white
        pydot_node.set_style("filled")  # Make the node visually filled
        pydot_node.set_label(node.split("/")[-1])  # Label with the last path segment

    # Set edge labels and styles
    for edge in graph.edges(keys=True):
        source, target, key = edge  # Extract source, target, and key for MultiDiGraph edges
        edge_type = str(graph.edges[source, target, key].get('type', 'Unknown'))  # Get the edge type
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
