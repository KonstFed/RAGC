from pathlib import Path

from ragc.llms import OllamaWrapper
from ragc.retrieval.pagerank import PageRankRetrieval
from ragc.graphs import SemanticParser

if __name__ == "__main__":
    repo_path = Path("/home/konstfed/Documents/diplom/RAGC/data/repositories/holostyak-bot")
    parser = SemanticParser()
    graph = parser.parse_into_files(repo_path)
    retrieval = PageRankRetrieval(repo_path=repo_path, semantic_graph=graph)
    relevant = retrieval.retrieve("", 5)
    for file, el in relevant:
        print(file)