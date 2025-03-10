from pathlib import Path

from pydantic import BaseModel, ConfigDict

from ragc.retrieval.common import Node


class PromptTemplate:
    def __init__(self, prompt_template: str, context_prompt_template: str):
        self.prompt_template = prompt_template
        self.context_prompt_template = context_prompt_template

    def inject(self, query: str, context: list[Node]) -> str:
        context_prompt = ""

        for node in context:
            context_prompt += self.context_prompt_template.format(**node.model_dump())

        prompt = self.prompt_template.format(context=context_prompt, query=query)
        return prompt


class RawPromptConfig(BaseModel):
    prompt_template: str
    context_prompt_template: str

    def create(self) -> PromptTemplate:
        return PromptTemplate(
            prompt_template=self.prompt_template,
            context_prompt_template=self.context_prompt_template,
        )

    model_config = ConfigDict(
        extra="ignore",
        frozen=True,
    )


class FilePromptConfig(BaseModel):
    prompt_template_p: Path
    context_prompt_template_p: Path

    def create(self) -> PromptTemplate:
        with self.prompt_template_p.open() as f:
            prompt_template = f.read()

        with self.context_prompt_template_p.open() as f:
            context_prompt_template = f.read()

        return PromptTemplate(prompt_template=prompt_template, context_prompt_template=context_prompt_template)

    model_config = ConfigDict(
        extra="ignore",
        frozen=True,
    )


PromptConfig = RawPromptConfig | FilePromptConfig
