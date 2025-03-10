import networkx as nx

from ragc.graphs.common import NodeType


def get_file_graph(graph: nx.MultiDiGraph) -> nx.MultiDiGraph:
    """Return graph with only files as nodes and import relation with them."""
    file_nodes = [n for n, attr in graph.nodes(data=True) if attr["type"] == NodeType.FILE]
    return graph.subgraph(file_nodes).copy()
