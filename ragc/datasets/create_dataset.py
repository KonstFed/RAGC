from pathlib import Path
import argparse

from pydantic import BaseModel

from ragc.datasets.train_dataset import TorchGraphDatasetConfig
from ragc.utils import load_config

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--evocodebench", type=Path, required=False, help="Path to evocodebench repos")
    parser.add_argument("config", type=Path, help="Path to dataset config")
    args = parser.parse_args()
    config: TorchGraphDatasetConfig = load_config(TorchGraphDatasetConfig, args.config)

    if args.evocodebench is not None:
        global_p = args.evocodebench
        for domain in global_p.iterdir():
            for repo_p in domain.iterdir():
                config.add_repo(repo_p.absolute())

    config.create()
