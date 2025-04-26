from ragc.graphs.hetero_transforms import ToHetero
from ragc.graphs.semantic_python_parser import SemanticParserConfig
from ragc.graphs.transforms import EmbedTransformConfig, MaskNodesConfig, ToPYG

GraphParserConfig = SemanticParserConfig
Transform = MaskNodesConfig | EmbedTransformConfig | ToHetero | ToPYG