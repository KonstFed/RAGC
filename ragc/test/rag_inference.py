import os
from pathlib import Path
from typing import Any, Dict, Literal

from ragc.test.utils import extract_signature

from .test_config import TestInference


def build_prompt(
    completion_path: str,
    namespace: str,
    signature: str,
    requirement: Dict[str, str],
    completion_type: Literal["function", "method"],
) -> str:
    requirement_str = "".join(f"## {key}\n{value}" for key, value in requirement.items())
    prompt = f"""Your task is to generate a Python {completion_type} based on the following details:

# Completion path
`{completion_path}`

# Namespace
`{namespace}`

# Requirements
{requirement_str}

# {completion_type.title()} signature
```
{signature}
```
# Answer format
It is very important, that your answer should only include a {completion_type} body without additional text and explanations of any kind.
"""

    return prompt


def _get_correct_namespace(completion_path: str, project_path: str, namespace: str) -> dict:
    file_path = Path(completion_path).relative_to(project_path)

    parts = list(file_path.parts)
    # remove python file
    parts[-1] = parts[-1].removesuffix(".py")
    if file_path.name == "__init__.py":
        parts = parts[:-1]

    namespace_parts = namespace.split(".")
    match_idx = 0
    for i in range(len(parts)):
        if parts[i] != namespace_parts[0]:
            continue

        is_match = True

        for j in range(i, len(parts)):
            if parts[j] != namespace_parts[j - i]:
                is_match = False
                break

        if is_match:
            match_idx = i
            break

    namespace = ".".join(parts[:match_idx] + namespace.split("."))

    if file_path.name == "__init__.py":
        _parts = list(file_path.parts)
        _parts[-1] = _parts[-1].removesuffix(".py")
        namespace = namespace.replace(".".join(file_path.parts[:-1]), ".".join(_parts))

    return namespace


def generate(task: Dict[str, Any], repos_dir: str | os.PathLike, test_inference: TestInference, **kwargs) -> str:
    config_path = kwargs.get("config_path", None)
    if not config_path:
        raise ValueError("No RAG config is specified!")

    task_path = os.path.join(repos_dir, task["completion_path"])

    # build prompt
    prompt = build_prompt(
        completion_path=task["completion_path"],
        namespace=task["namespace"],
        signature=extract_signature(task_path, task["signature_position"]),
        requirement=task["requirement"],
        completion_type=task["type"],
    )

    namespace = _get_correct_namespace(task["completion_path"], task["project_path"], task["namespace"])

    # inference
    generation, _meta = test_inference(
        repo_name=Path(task["project_path"]).parts[-1],
        prompt=prompt,
        node_namespace=namespace,
    )

    result = {
        "namespace": task["namespace"],
        "completion": generation,
    }

    return result
