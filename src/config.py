"""Configuration loading and validation for the ASL application."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


REQUIRED_SCHEMA: dict[str, tuple[str, ...]] = {
    "dataset": ("train_csv", "test_csv"),
    "model": (
        "checkpoint_path",
        "word_checkpoint_path",
        "word_model_type",
        "num_classes",
        "dropout_rate",
        "word_frame_count",
        "word_image_size",
        "word_confidence_threshold",
        "word_margin_threshold",
        "word_stability_frames",
    ),
    "training": (
        "epochs",
        "word_epochs",
        "batch_size",
        "word_batch_size",
        "word_cache_videos",
        "word_validate_videos",
        "word_use_bbox",
        "word_split_strategy",
        "word_keypoints_dir",
        "learning_rate",
        "weight_decay",
        "scheduler",
        "scheduler_patience",
        "scheduler_factor",
    ),
    "augmentation": (
        "rotation_degrees",
        "horizontal_flip",
        "brightness_jitter",
        "contrast_jitter",
    ),
    "inference": (
        "confidence_threshold",
        "roi_size",
        "roi_x",
        "roi_y",
        "camera_index",
        "target_fps",
    ),
    "avatar": (
        "signs_dir",
        "word_videos_dir",
        "sign_to_text_words_dir",
        "sign_to_text_manifest",
        "min_display_duration_ms",
        "max_display_duration_ms",
        "display_duration_ms",
        "space_duration_ms",
        "no_sign_duration_ms",
        "image_size",
        "max_input_chars",
    ),
    "logging": ("log_dir", "training_log_file", "log_level"),
}


def validate_config(config: dict[str, Any]) -> None:
    """Validate that all required config sections and keys are present.

    Args:
        config: Parsed YAML configuration.

    Raises:
        KeyError: If a required section or key is missing.
        ValueError: If basic numeric constraints are violated.
    """
    missing: list[str] = []
    for section, keys in REQUIRED_SCHEMA.items():
        if section not in config or not isinstance(config[section], dict):
            missing.append(section)
            continue
        for key in keys:
            if key not in config[section]:
                missing.append(f"{section}.{key}")
    if missing:
        raise KeyError("Missing required config parameters: " + ", ".join(missing))

    dropout = float(config["model"]["dropout_rate"])
    if not 0.3 <= dropout <= 0.5:
        raise ValueError("model.dropout_rate must be in [0.3, 0.5]")
    if int(config["model"]["num_classes"]) != 24:
        raise ValueError("model.num_classes must be 24")
    if float(config["training"]["weight_decay"]) < 1e-4:
        raise ValueError("training.weight_decay must be at least 1e-4")
    duration = int(config["avatar"]["display_duration_ms"])
    min_duration = int(config["avatar"]["min_display_duration_ms"])
    max_duration = int(config["avatar"]["max_display_duration_ms"])
    if not min_duration <= duration <= max_duration:
        raise ValueError("avatar.display_duration_ms must be between min and max duration")


def load_config(path: str | Path = "config.yaml") -> dict[str, Any]:
    """Load and validate the YAML configuration file.

    Args:
        path: Path to the YAML file.

    Returns:
        The parsed configuration dictionary.

    Raises:
        FileNotFoundError: If the config file is missing.
        ValueError: If the YAML content is not a mapping or is invalid.
        KeyError: If required parameters are absent.
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)

    if not isinstance(loaded, dict):
        raise ValueError(f"{config_path} must contain a YAML mapping")

    validate_config(loaded)
    return loaded
