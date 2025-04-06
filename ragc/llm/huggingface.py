from typing import Literal

import torch
from transformers import AutoModel, AutoTokenizer

from ragc.llm.embedding import BaseEmbedder, BaseEmbederConfig


class HuggingFaceEmbedder(BaseEmbedder):
    def __init__(self, model_name: str, max_batch_size: int, max_length: int | None, store_in_gpu: bool = False):
        self.model_name = model_name
        self.max_batch_size = max_batch_size
        self.store_in_gpu = store_in_gpu

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

        if max_length is None:
            self.max_length = self.tokenizer.model_max_length
        else:
            self.max_length = max_length

        if store_in_gpu:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(device)

    def embed(self, inputs: list[str] | str) -> torch.Tensor:
        if isinstance(inputs, str):
            inputs = [inputs]

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

        batch_size = min(self.max_batch_size, len(inputs))
        embeddings = []

        for i in range(0, len(inputs), batch_size):
            batch = inputs[i : min(i + batch_size, len(inputs))]
            tokenized_inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=self.max_length,
            ).to(device)
            with torch.no_grad():
                outputs = self.model(**tokenized_inputs)

            embeddings.append(outputs.last_hidden_state.mean(dim=1))

        embeddings = torch.cat(embeddings)

        if not self.store_in_gpu:
            embeddings = embeddings.cpu()
            self.model.to(torch.device("cpu"))

        return embeddings


class HuggingFaceEmbedderConfig(BaseEmbederConfig):
    type: Literal["hugging_face"] = "hugging_face"

    model_name: str
    max_batch_size: int
    max_length: int | None = None
    store_in_gpu: bool = True

    def create(self) -> HuggingFaceEmbedder:
        return HuggingFaceEmbedder(
            model_name=self.model_name,
            max_batch_size=self.max_batch_size,
            max_length=self.max_length,
            store_in_gpu=self.store_in_gpu,
        )
