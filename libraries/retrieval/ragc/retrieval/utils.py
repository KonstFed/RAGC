import networkx as nx


def get_call_graph(g: nx.MultiDiGraph) -> nx.DiGraph:
    """Получить subgraph, содержащий только вызовы."""
    # Define a filtering condition (e.g., keep edges with weight > 3)
    call_edges = [(u, v, d) for u, v, d in g.edges(data=True) if d["type"] == "Invoke"]

    # Create a new graph with the filtered edges
    call_graph = nx.DiGraph()
    call_graph.add_edges_from(call_edges)
    return call_graph
