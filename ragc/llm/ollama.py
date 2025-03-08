from typing import Literal

import numpy as np
import ollama
from pydantic import HttpUrl

from ragc.llm.embedding import BaseEmbedder, EmbederConfig
from ragc.llm.generator import BaseGenerator, GeneratorConfig


class OllamaEmbedder(BaseEmbedder):
    def __init__(self, ollama_url: str, emb_model: str):
        self.emb_model = emb_model
        self.client = ollama.Client(host=ollama_url)

    def embed(self, inputs: list[str] | str) -> np.ndarray:
        response = self.client.embed(model=self.emb_model, input=inputs)
        embeddings = np.array(response.embeddings, dtype=np.float32)
        return embeddings


class OllamaEmbedderConfig(EmbederConfig):
    type: Literal["ollama_embedding"] = "ollama_embedding"
    ollama_url: HttpUrl
    emb_model: str

    def create(self) -> OllamaEmbedder:
        return OllamaEmbedder(
            ollama_url=str(self.ollama_url),
            emb_model=self.emb_model,
        )


class OllamaGenerator(BaseGenerator):
    def __init__(self, ollama_url: str, model: str):
        self.client = ollama.Client(ollama_url)
        self.model = model

    def generate(self, prompt: str) -> str:
        response = self.client.generate(model=self.model, prompt=prompt)
        return response.response


class OllamaGeneratorConfig(GeneratorConfig):
    ollama_url: HttpUrl
    model: str

    def create(self) -> OllamaGenerator:
        return OllamaGenerator(
            ollama_url=self.ollama_url,
            model=self.model,
        )
