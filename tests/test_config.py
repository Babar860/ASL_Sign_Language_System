from pathlib import Path

import pytest
import yaml

from src.config import load_config


def test_config_loads_root_file():
    config = load_config("config.yaml")
    assert config["model"]["num_classes"] == 24


def test_missing_config_key_fails(tmp_path: Path):
    config = load_config("config.yaml")
    del config["model"]["dropout_rate"]
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(KeyError):
        load_config(path)
