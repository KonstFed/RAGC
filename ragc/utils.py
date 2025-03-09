from pathlib import Path

import yaml
from pydantic import BaseModel


def load_config(cls: type[BaseModel], cfg_p: Path) -> BaseModel:
    with cfg_p.open("r") as f:
        data = yaml.safe_load(f)

    return cls.model_validate(data)
