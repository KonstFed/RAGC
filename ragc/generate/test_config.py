from pydantic import BaseModel

from ragc.datasets.train_dataset import TorchGraphDatasetConfig, TorchGraphDataset
from ragc.graphs.transforms import MaskNodes
from ragc.inference import InferenceConfig


class TestInference:
    def __init__(self, dataset: TorchGraphDataset, inference_cfg: InferenceConfig):
        self.dataset = dataset
        self.inference_cfg = inference_cfg

    def __call__(self, repo_name: str, prompt: str, node_namespace: str) -> str:
        graph = self.dataset.get_by_name(repo_name)
        if graph is None:
            raise ValueError(f"repo {repo_name} is not present in dataset")
        t = MaskNodes([node_namespace], mask_callee=True)
        graph = t(graph)
        inference = self.inference_cfg.create(graph=graph)
        return inference(query=prompt)


class TestInferenceConfig(BaseModel):
    """Cached inference for test metrics."""

    inference: InferenceConfig
    dataset: TorchGraphDatasetConfig

    def create(self) -> TestInference:
        return TestInference(dataset=self.dataset.create(), inference_cfg=self.inference)
