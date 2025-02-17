import glob
from pathlib import Path
from tempfile import TemporaryDirectory

import networkx as nx
from code2flow import code2flow

from .common import BaseGraphParser


class Code2FlowParser(BaseGraphParser):
    # некорректно работает если есть файлы с одинаковыми именами
    # если есть одинаково названные функции

    def parse(repo_path: Path) -> nx.MultiDiGraph:
        with TemporaryDirectory() as tmp:
            result_p = Path(tmp) / "graph.json"
            
            code2flow(str(repo_path), str(result_p), language="py")


        return super().parse()