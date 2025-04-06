from pathlib import Path

from pydantic import BaseModel, Field
from torch_geometric.data import Data

from ragc.fusion import BaseFusion, FusionConfig
from ragc.graphs import GraphParserConfig
from ragc.llm.types import EmbedderConfig
from ragc.llm.embedding import BaseEmbedder
from ragc.retrieval import RetrievalConfig
from ragc.retrieval.common import BaseRetrieval


class Inference:
    def __init__(
        self,
        query_embedder: BaseEmbedder,
        retrieval: BaseRetrieval,
        fusion: BaseFusion,
    ):
        self.query_embedder = query_embedder
        self.retrieval = retrieval
        self.fusion = fusion

    def __call__(
        self,
        query: str,
    ):
        query_emb = self.query_embedder.embed([query])[0]
        relevant_nodes = self.retrieval.retrieve(query=query_emb)
        return self.fusion.fuse_and_generate(query=query, relevant_nodes=relevant_nodes)


class InferenceConfig(BaseModel):
    parser: GraphParserConfig = Field(discriminator="type")
    query_embedder: EmbedderConfig = Field(discriminator="type")
    retrieval: RetrievalConfig = Field(discriminator="type")
    fusion: FusionConfig = Field(discriminator="type")
    n_elems: int = 5

    def create(self, repo_path: Path | None = None, graph: Data | None = None) -> Inference:
        if repo_path is not None:
            # TODO deprecated support for using without dataset
            raise ValueError("Not supported inference for repository without dataset")
            parser = self.parser.create()
            graph = parser.parse(repo_path=repo_path)
        # else:
        #     graph = graph
        fusion = self.fusion.create()
        retrieval = self.retrieval.create(graph)
        embedder = self.query_embedder.create()
        return Inference(retrieval=retrieval, fusion=fusion, query_embedder=embedder)


if __name__ == "__main__":
    import argparse

    from ragc.utils import load_config

    arg_parser = argparse.ArgumentParser(description="Inference parser")
    arg_parser.add_argument("-c", "--config", type=Path, required=True, help="path to .yaml config")
    arg_parser.add_argument("-r", "--repo_path", type=Path, help="Path to repo to question", required=True)
    arg_parser.add_argument("query", type=str, help="Your query")
    args = arg_parser.parse_args()
    cfg: InferenceConfig = load_config(InferenceConfig, args.config)
    inference = cfg.create()
    inference(args.query, repo_path=args.repo_path.absolute().resolve())
