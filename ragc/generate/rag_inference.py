import os

from typing import Dict, Literal, Any
from pathlib import Path

from ragc.generate.utils import extract_signature
from ragc.utils import load_config
from .test_config import TestInference, TestInferenceConfig


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


def generate(task: Dict[str, Any], repos_dir: str | os.PathLike, **kwargs) -> str:
    config_path = kwargs.get("config_path", None)
    if not config_path:
        raise ValueError("No RAG config is specified!")

    repo_path = os.path.join(repos_dir, task["completion_path"].split("/")[0])
    task_path = os.path.join(repos_dir, task["completion_path"])

    # build prompt
    prompt = build_prompt(
        completion_path=task["completion_path"],
        namespace=task["namespace"],
        signature=extract_signature(task_path, task["signature_position"]),
        requirement=task["requirement"],
        completion_type=task["type"],
    )

    # inference
    cfg: TestInferenceConfig = load_config(TestInferenceConfig, config_path)
    inference: TestInference = cfg.create()
    inference(
        repo_name=Path(task["project_path"]).parts[-1],
        prompt=prompt,
        node_namespace=".".join(task["namespace"].split(".")[1:]),
    )
    generation = inference(prompt)

    return generation
