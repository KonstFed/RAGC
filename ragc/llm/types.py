from ragc.llm.huggingface import HuggingFaceEmbedderConfig, CompletionGeneratorConfig
from ragc.llm.ollama import OllamaEmbedderConfig

EmbedderConfig = OllamaEmbedderConfig | HuggingFaceEmbedderConfig
GeneratorConfig = CompletionGeneratorConfig
