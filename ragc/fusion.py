from abc import ABC, abstractmethod
from typing import Literal

from pydantic import BaseModel, ConfigDict

from ragc.graphs.common import Node
from ragc.llm import BaseGenerator, GeneratorConfig
from ragc.prompt import PromptConfig, PromptTemplate


class BaseFusion(ABC):
    @abstractmethod
    def fuse_and_generate(self, query: str, relevant_nodes: list[Node]) -> dict[str, str]:
        """Fuse relevant nodes into generator and produce answer to query."""


class PromptFusion(BaseFusion):
    def __init__(self, generator: BaseGenerator, prompt_template: PromptTemplate):
        self.generator = generator
        self.prompt_template = prompt_template

    def fuse_and_generate(self, query: str, relevant_nodes: list[Node]) -> str:
        """Fuse relevant node via simple prompt injection."""
        prompt = self.prompt_template.inject(query=query, context=relevant_nodes)
        answer = self.generator.generate(prompt=prompt)
        return {
            "answer": answer,
            "prompt": prompt,
        }

class BaseFusionConfig(ABC, BaseModel):
    model_config = ConfigDict(
        extra="ignore",
        frozen=True,
    )

    @abstractmethod
    def create(self) -> BaseFusion:
        """Create Fusion instance."""


class PromptFusionConfig(BaseFusionConfig):
    type: Literal["prompt"] = "prompt"
    prompt: PromptConfig
    generator: GeneratorConfig

    def create(self) -> PromptFusion:
        return PromptFusion(
            generator=self.generator.create(),
            prompt_template=self.prompt.create(),
        )


FusionConfig = PromptFusionConfig
