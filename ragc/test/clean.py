import re

import pandas as pd

def extract_python_code_blocks(text):
    """Extract only Python code blocks from text, ignoring all other content."""
    pattern = r"```python(.*?)```"
    matches = re.finditer(pattern, text, re.DOTALL)

    code_blocks = []
    for match in matches:
        code_block = match.group(1).strip()
        if code_block:
            code_blocks.append(code_block)

    return code_blocks


def clean_single(code: str) -> str:
    """Clean single code instance."""
    try:
        code = extract_python_code_blocks(code)[0]
    finally:
        return code


def clean(completions: pd.DataFrame) -> pd.DataFrame:
    """Standartize LLM output for insertion into tests."""
    completions["completion"] = completions["completion"].apply(clean_single)
    return completions
