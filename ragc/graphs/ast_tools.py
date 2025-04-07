import ast

import tree_sitter_python as tspython
from tree_sitter import Language, Parser


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


def to_zero_ident(code: str) -> str:
    """Set code snippet to zero ident."""
    def _check_empty(line: str) -> bool:
        if len(line) == 0:
            return True

        return all(l == " " for l in line)

    lines = code.splitlines()
    if len(lines) == 0:
        return code
    min_indent = min((len(line) - len(line.lstrip())) for line in lines if not _check_empty(line))
    stripped_lines = [line if _check_empty(line) else line[min_indent:] for line in lines]
    code = "\n".join(stripped_lines)
    return code


class PythonExtractHelper:
    def __init__(self):
        self.py_language = Language(tspython.language())
        self.parser = Parser(self.py_language)

    def extract_body(self, code: str) -> str:
        """Extract function/class body, handling UTF-8 correctly."""
        tree = self.parser.parse(bytes(code, "utf-8"))
        root_node = tree.root_node

        lines = code.splitlines(keepends=True)  # Preserve line endings for accurate columns

        for node in root_node.children:
            if node.type in ("function_definition", "class_definition"):
                body_node = node.child_by_field_name("body")
                if not body_node or body_node.start_point == body_node.end_point:
                    continue  # Skip empty bodies

                start_line, start_col = body_node.start_point
                start_col = 0
                end_line, end_col = body_node.end_point

                # Extract lines covered by the body
                body_lines = []
                for i in range(start_line, end_line + 1):
                    line = lines[i]
                    if i == start_line:
                        line = line[start_col:]  # Trim start column
                    if i == end_line:
                        line = line[:end_col]  # Trim end column
                    body_lines.append(line)

                body_text = "".join(body_lines)
                return to_zero_ident(body_text)

        return ""  # No function/class found
