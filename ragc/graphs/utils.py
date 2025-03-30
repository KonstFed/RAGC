import networkx as nx

import ast

from ragc.graphs.common import NodeType


def get_file_graph(graph: nx.MultiDiGraph) -> nx.MultiDiGraph:
    """Return graph with only files as nodes and import relation with them."""
    file_nodes = [n for n, attr in graph.nodes(data=True) if attr["type"] == NodeType.FILE]
    return graph.subgraph(file_nodes).copy()


def extract_function_info(code: str) -> tuple[str | None, str | None]:
    """Extract function signatures and docstrings."""
    tree = ast.parse(code)

    signature = None
    docstring = None

    for node in tree.body:
        if not isinstance(node, ast.FunctionDef):  # Check if it's a function
            continue

        signature = f"{node.name}({', '.join(arg.arg for arg in node.args.args)})"
        docstring = ast.get_docstring(node)  # Extract docstring
        return signature, docstring

    return signature, docstring


def extract_class_info(code) -> str | None:
    """Extract class docstring if it has one."""
    tree = ast.parse(code)

    for node in tree.body:
        if isinstance(node, ast.ClassDef):  # Check if it's a class
            docstring = ast.get_docstring(node)  # Extract docstring
            return docstring

    return None
