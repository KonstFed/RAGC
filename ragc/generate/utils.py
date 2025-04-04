import os
import json
import textwrap

from typing import List, Tuple, Dict, Any


def load_tasks(path: str | os.PathLike) -> List[Dict[str, Any]]:
    tasks = []
    with open(path, 'r') as f:
        for line in f:
            js = json.loads(line)
            tasks.append(js)
    
    return tasks


def extract_signature(completion_path: str | os.PathLike, signature_position: Tuple[int, int]) -> str:
    start, end = signature_position
    with open(completion_path, 'r') as f:
        completion_lines = f.read().split('\n')[start - 1:end]
    
    signature = '\n'.join(textwrap.dedent(line) for line in completion_lines)
    return signature
