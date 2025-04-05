import ast

from ragc.graphs.common import NodeType

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
