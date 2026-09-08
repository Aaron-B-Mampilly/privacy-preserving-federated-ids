"""Load the project's YAML configuration.

Every hyperparameter fixed by the frozen design spec lives in
configs/config.yaml. Code should call load_config() and read values
from the returned dict rather than hardcoding numbers, so a change to
e.g. window_size is a one-line config edit, not a source-code hunt.
"""

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


def load_config(config_path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not config:
        raise ValueError(f"Config file is empty or invalid: {config_path}")

    return config
