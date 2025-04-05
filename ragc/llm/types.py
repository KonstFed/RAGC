from ragc.llm.huggingface import HuggingFaceEmbedderConfig
from ragc.llm.ollama import OllamaEmbedderConfig, OllamaGeneratorConfig

EmbedderConfig = OllamaEmbedderConfig | HuggingFaceEmbedderConfig
GeneratorConfig = OllamaGeneratorConfig
