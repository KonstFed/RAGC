from ragc.llm.huggingface import HuggingFaceEmbedderConfig, HuggingFaceGeneratorConfig
from ragc.llm.ollama import OllamaEmbedderConfig, OllamaGeneratorConfig

EmbedderConfig = OllamaEmbedderConfig | HuggingFaceEmbedderConfig
GeneratorConfig = OllamaGeneratorConfig | HuggingFaceGeneratorConfig
