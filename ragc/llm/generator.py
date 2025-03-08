from abc import ABC, abstractmethod

from pydantic import BaseModel, ConfigDict


class BaseGenerator(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Generate embeddings for given inputs."""


class GeneratorConfig(BaseModel):

    def create(self) -> BaseGenerator:
        """Factory method that creates instance of BaseGenerator."""
        raise NotImplementedError

    model_config = ConfigDict(
        extra="ignore",
        frozen=True,
    )
