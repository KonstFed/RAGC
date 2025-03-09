from pathlib import Path

import networkx as nx
from pydantic import BaseModel, Field

from ragc.fusion import BaseFusion, FusionConfig
from ragc.graphs import BaseGraphParser, GraphParserConfig
from ragc.retrieval import RetrievalConfig
from ragc.retrieval.common import BaseRetievalConfig


class Inference:
    def __init__(
        self, parser: BaseGraphParser, retrieval_cfg: BaseRetievalConfig, fusion: BaseFusion, n_elems: int = 5,
    ):
        self.parser = parser
        self.retrieval_cfg = retrieval_cfg
        self.fusion = fusion
        self.n_elems = n_elems

    def __call__(
        self,
        query: str,
        repo_path: Path | str | None = None,
        graph: nx.MultiDiGraph | None = None,
        ignore_nodes: list[str] | None = None,
    ):
        if repo_path is not None:
            graph = self.parser.parse(repo_path=Path(repo_path))

        # TODO: надо придумать механизм умного ретриавала (сколько возращать? Пока просто 5 элементов)
        retrieval = self.retrieval_cfg.create(graph=graph)
        relevant_nodes = retrieval.retrieve(query=query, n_elems=self.n_elems, ignore_nodes=ignore_nodes)
        return self.fusion.fuse_and_generate(query=query, relevant_nodes=relevant_nodes)


class InferenceConfig(BaseModel):
    parser: GraphParserConfig = Field(discriminator="type")
    retrieval: RetrievalConfig = Field(discriminator="type")
    fusion: FusionConfig = Field(discriminator="type")

    n_elems: int = 5

    def create(self) -> Inference:
        fusion = self.fusion.create()
        return Inference(parser=self.parser, retrieval_cfg=self.retrieval, fusion=fusion, n_elems=self.n_elems)

if __name__ == "__main__":
    import argparse
    from ragc.utils import load_config, save_config

    arg_parser = argparse.ArgumentParser(description="Inference parser")
    arg_parser.add_argument("-c", "--config", type=Path, required=True, help="path to .yaml config")
    arg_parser.add_argument("-r","--repo_path", type=Path, help="Path to repo to question", required=True)
    arg_parser.add_argument("query", type=str, help="Your query")
    args = arg_parser.parse_args()
    cfg: InferenceConfig = load_config(InferenceConfig, args.config)
    inference = cfg.create()
    inference(args.query, repo_path=args.repo_path.absolute().resolve())
