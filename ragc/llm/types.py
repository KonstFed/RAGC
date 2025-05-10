from ragc.llm.huggingface import HuggingFaceEmbedderConfig, DeepseekGreedyGeneratorConfig
from ragc.llm.ollama import OllamaEmbedderConfig

EmbedderConfig = OllamaEmbedderConfig | HuggingFaceEmbedderConfig
GeneratorConfig = DeepseekGreedyGeneratorConfig
