import networkx as nx

from ragc.graphs.common import BaseNode, FileNode, 

def plot_graph(graph: nx.MultiDiGraph):
    # Extract node attributes (color, nesting) for visualization
    node_colors = nx.get_node_attributes(graph, 'color')  # Get colors of nodes

    # Create a DOT (graph description) representation
    dot = nx.nx_pydot.to_pydot(self.graph)  # Coxnvert the graph to a DOT format

    # Set node styles, colors, and labels
    for node in self.graph.nodes():
        pydot_node = dot.get_node(str(node))[0]  # Get the corresponding node in the DOT representation
        pydot_node.set_fillcolor(node_colors.get(node, 'white'))  # Set fill color, default to white
        pydot_node.set_style("filled")  # Make the node visually filled
        pydot_node.set_label(node.split("/")[-1])  # Label with the last path segment

    # Set edge labels and styles
    for edge in self.graph.edges(keys=True):
        source, target, key = edge  # Extract source, target, and key for MultiDiGraph edges
        edge_type = self.graph.edges[source, target, key].get('type', 'Unknown')  # Get the edge type
        edge_color = EDGES_COLORS.get(edge_type, "black")  # Get the edge color based on the type
        edge_style = EDGES_STYLES.get(edge_type, "solid")  # Get the edge style based on the type

        # Get the corresponding edge in the DOT representation
        pydot_edge = dot.get_edge(str(source), str(target))[key]

        # Apply attributes to the edge
        pydot_edge.set_label(edge_type)  # Set edge label to indicate the type
        pydot_edge.set_style(edge_style)  # Set style for the edge
        pydot_edge.set_color(edge_color)  # Set color for the edge

    # Save the resulting graph visualization as a PNG image
    png_name = f'{self.path_to_repo.split(chr(92))[-1]}.png'  # Generate PNG file name
    dot.write_png(png_name)  # Write the DOT representation to a PNG file
    #
    # Optionally, display the generated graph image
    img = Image.open(png_name)  # Open the generated PNG image
    img.show()  # Display the image