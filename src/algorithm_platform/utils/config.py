import yaml
from dataclasses import dataclass
from typing import Any, Dict

@dataclass
class Config:
    raw: Dict[str, Any]

    def __getitem__(self, item):
        return self.raw[item]


def load_config(path: str) -> Config:
    with open(path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    return Config(raw=data)
