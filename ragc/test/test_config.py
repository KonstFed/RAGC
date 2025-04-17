import warnings
from pathlib import Path
from typing import Iterator, Literal

import pandas as pd
from pydantic import BaseModel
from torch_geometric.data import Data
from tqdm import tqdm

from ragc.datasets.train_dataset import TorchGraphDataset, TorchGraphDatasetConfig
from ragc.graphs import Node
from ragc.graphs.transforms import MaskNodes
from ragc.graphs.utils import pyg_extract_node
from ragc.inference import InferenceConfig
from ragc.test.utils import extract_signature, map_cross_file_dependency


def build_prompt(
    completion_path: str,
    namespace: str,
    signature: str,
    requirement: dict[str, str],
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
    """Transform namespace of evocodebench into graph namespace."""
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


class TestInference:
    def __init__(
        self,
        dataset: TorchGraphDataset,
        inference_cfg: InferenceConfig,
        task_path: Path,
        repos_path: Path,
        only_with_cross_file: bool = True,
        use_gold_context: bool = False,
    ):
        self.dataset = dataset
        self.inference_cfg = inference_cfg

        self.repos_path = repos_path
        self.task_path = task_path
        self.only_with_cross_file = only_with_cross_file
        self.use_gold_context = use_gold_context

        _task = pd.read_json(task_path, lines=True)
        _task = _task[_task["project_path"].apply(lambda x: x.split("/")[-1]).isin(dataset.get_repos_names())]

        if only_with_cross_file:
            _task = self._clean_dependecies(_task)

        self.tasks = _task

    def _clean_dependecies(self, tasks: pd.DataFrame) -> pd.DataFrame:
        def _present_in_graph(task) -> bool:
            repo_name = Path(task["project_path"]).parts[-1]
            namespace = _get_correct_namespace(task["completion_path"], task["project_path"], task["namespace"])

            graph = self._prepare_graph(repo_name=repo_name, node_namespace=namespace)

            gold_nodes = self._get_gold_snippets(task_row=task, graph=graph)
            return gold_nodes is not None

        tasks["cross_file"] = tasks["dependency"].apply(lambda x: x["cross_file"])
        tasks = tasks[tasks["cross_file"].apply(len) > 0]
        return tasks[tasks.apply(_present_in_graph, axis=1)]

    def _prepare_graph(self, repo_name: str, node_namespace: str) -> Data:
        """Get graph from node_namespace."""
        graph = self.dataset.get_by_name(repo_name)
        t = MaskNodes([node_namespace], mask_callee=True)
        graph = t(graph)
        return graph

    def retrieve(self, repo_name: str, prompt: str, node_namespace: str) -> list[Node]:
        graph = self._prepare_graph(repo_name, node_namespace)
        inference = self.inference_cfg.create(graph=graph)
        return inference.retrieve(prompt)

    def __call__(self, repo_name: str, prompt: str, node_namespace: str) -> tuple[str, dict]:
        graph = self._prepare_graph(repo_name, node_namespace)
        inference = self.inference_cfg.create(graph=graph)
        return inference(query=prompt)

    def _generate_with_golden(
        self, repo_name: str, prompt: str, node_namespace: str, golden_nodes: list[Node]
    ) -> tuple[str, dict]:
        graph = self._prepare_graph(repo_name, node_namespace)
        inference = self.inference_cfg.create(graph=graph)
        return inference.generate_with_context(query=prompt, nodes=golden_nodes)

    def generate_completion(self, progress_bar: bool = True) -> Iterator[dict[str, str]]:
        """Pipeline for evocodebench generation."""
        bar = tqdm(self.tasks.iterrows(), total=len(self.tasks)) if progress_bar else self.tasks.iterrows()
        for _, task in bar:
            task_path = self.repos_path / task["completion_path"]
            namespace = _get_correct_namespace(task["completion_path"], task["project_path"], task["namespace"])
            repo_name = Path(task["project_path"]).parts[-1]

            prompt = build_prompt(
                completion_path=task["completion_path"],
                namespace=task["namespace"],
                signature=extract_signature(task_path, task["signature_position"]),
                requirement=task["requirement"],
                completion_type=task["type"],
            )
            if not self.use_gold_context:
                generation, _meta = self(
                    repo_name=repo_name,
                    prompt=prompt,
                    node_namespace=namespace,
                )
            else:
                graph = self._prepare_graph(repo_name=repo_name, node_namespace=namespace)
                gold_nodes = self._get_gold_snippets(task_row=task, graph=graph)
                if gold_nodes is None:
                    # gold dependencies are not in our graph
                    warnings.warn(f"{task['namespace']} got cross file reference not found in graph.")
                    continue
                generation, _meta = self._generate_with_golden(
                    repo_name=repo_name,
                    prompt=prompt,
                    node_namespace=namespace,
                    golden_nodes=gold_nodes,
                )

            result = {
                "namespace": task["namespace"],
                "completion": generation,
            }
            yield result

    def _get_gold_snippets(self, task_row, graph: Data) -> list[Node] | None:
        indices = []
        for dependency in task_row["dependency"]["cross_file"]:
            cur_idx = map_cross_file_dependency(dependency, task_row["project_path"], graph)

            if cur_idx is None:
                return None

            indices.append(cur_idx)

        return pyg_extract_node(graph, indices)

    def generate_retrieval_pairs(self, progress_bar: bool = True) -> Iterator[tuple[list[str], list[str]]]:
        """Pipeline for evocodebench generation."""
        bar = tqdm(self.tasks.iterrows(), total=len(self.tasks)) if progress_bar else self.tasks.iterrows()
        for _, task in bar:
            repo_name = Path(task["project_path"]).parts[-1]
            task_path = self.repos_path / task["completion_path"]
            namespace = _get_correct_namespace(task["completion_path"], task["project_path"], task["namespace"])

            graph = self._prepare_graph(repo_name=repo_name, node_namespace=namespace)

            gold_nodes = self._get_gold_snippets(task_row=task, graph=graph)
            if gold_nodes is None:
                # gold dependencies are not in our graph
                warnings.warn(f"{task['namespace']} got cross file reference not found in graph.")
                continue

            prompt = build_prompt(
                completion_path=task["completion_path"],
                namespace=task["namespace"],
                signature=extract_signature(task_path, task["signature_position"]),
                requirement=task["requirement"],
                completion_type=task["type"],
            )
            retrieved_nodes = self.retrieve(
                repo_name=Path(task["project_path"]).parts[-1],
                prompt=prompt,
                node_namespace=namespace,
            )

            yield gold_nodes, retrieved_nodes


class TestInferenceConfig(BaseModel):
    """Cached inference for test metrics."""

    inference: InferenceConfig
    dataset: TorchGraphDatasetConfig

    task_path: Path
    repos_path: Path
    only_with_cross_file: bool = True
    use_gold_context: bool = False

    def create(self) -> TestInference:
        return TestInference(
            dataset=self.dataset.create(),
            inference_cfg=self.inference,
            task_path=self.task_path,
            repos_path=self.repos_path,
            only_with_cross_file=self.only_with_cross_file,
            use_gold_context=self.use_gold_context,
        )
