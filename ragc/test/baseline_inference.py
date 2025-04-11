import os

from typing import Dict, Literal, Any

from ragc.test.utils import extract_signature


def build_prompt(
        completion_path: str,
        namespace: str,
        signature: str,
        requirement: Dict[str, str],
        domain: str,
        completion_type: Literal['function', 'method']
) -> str:
    raise NotImplementedError


def generate(
        task: Dict[str, Any],
        repos_dir: str | os.PathLike,
        **kwargs
) -> str:
    raise NotImplementedError
