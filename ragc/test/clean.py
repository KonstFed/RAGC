import re

import black
import pandas as pd

from ragc.graphs.ast_tools import PythonExtractHelper, to_zero_ident


def extract_python_code_blocks(text):
    """Extract code blocks."""
    pattern = r'```python(.*?)```'
    code_blocks = re.findall(pattern, text, re.DOTALL)

    cleaned_blocks = []
    for block in code_blocks:
        lines = block.split('\n')
        start = 0
        while start < len(lines) and lines[start].strip() == '':
            start += 1
        end = len(lines) - 1
        while end >= 0 and lines[end].strip() == '':
            end -= 1
        if start > end:
            cleaned_block = ''
        else:
            cleaned_lines = lines[start:end+1]
            cleaned_block = '\n'.join(cleaned_lines)
        cleaned_blocks.append(cleaned_block)
    return cleaned_blocks


def clean_single(code: str, helper: PythonExtractHelper) -> str:
    """Clean single code instance."""
    # TODO: solve import problem
    # import are cutted out if we take only func body
    #
    # - classes are not inserted correctly (only their function?)

    try:
        code = extract_python_code_blocks(code)[0]
        code = helper.extract_body(code)
        code = to_zero_ident(code)
        code = black.format_str(code, mode=black.Mode())
    finally:
        return code


def clean(completions: pd.DataFrame) -> pd.DataFrame:
    """Standartize LLM output for insertion into tests."""
    helper = PythonExtractHelper()
    completions["completion"] = completions["completion"].apply(clean_single, helper=helper)
    return completions


