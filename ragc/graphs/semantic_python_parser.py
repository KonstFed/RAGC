from tempfile import TemporaryDirectory
from pathlib import Path

import networkx as nx
from semantic_parser import SemanticGraphBuilder

from ragc.graphs.common import BaseGraphParser

class SemanticParser(BaseGraphParser):
    def __init__(self):
        self._builder = SemanticGraphBuilder()

    def parse(self, repo_path: Path) -> nx.MultiDiGraph:
        with TemporaryDirectory() as t:
                
            self._builder.build_from_one(str(repo_path), t, gsave=True,
                                    gprint=False) 
            
            t = Path(t)
            graph_file = next(t.iterdir())
            graph = nx.read_gml(graph_file)
        
        return graph