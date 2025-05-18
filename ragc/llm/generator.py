from abc import ABC, abstractmethod

from pydantic import BaseModel, ConfigDict

from ragc.graphs.common import Node


class BaseGenerator:
    def generate(self, prompt: str) -> str:
        """Generate for given input."""


class AugmentedGenerator:
    def generate(self, query: str, relevant_nodes: list[Node]) -> str:
        """Generate for given input."""


class BaseGeneratorConfig(BaseModel):
    def create(self) -> BaseGenerator:
        """Factory method that creates instance of BaseGenerator."""
        raise NotImplementedError

    model_config = ConfigDict(
        extra="ignore",
        frozen=True,
    )


class AugmentedGeneratorConfig(BaseModel):
    def create(self) -> AugmentedGenerator:
        """Factory method that creates instance of AugmentedGenerator."""
        raise NotImplementedError

    model_config = ConfigDict(
        extra="ignore",
        frozen=True,
    )
