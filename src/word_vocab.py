"""Shared word vocabulary helpers for Text-to-Sign and Sign-to-Text."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def normalize_word_label(label: str) -> str:
    """Normalize a word or phrase label for matching filenames and WLASL glosses."""
    return " ".join(label.strip().lower().replace("_", " ").replace("-", " ").split())


def compact_word_key(label: str) -> str:
    """Return a compact alphanumeric key for tolerant vocabulary matching."""
    return "".join(ch for ch in normalize_word_label(label) if ch.isalnum())


def display_word_label(label: str) -> str:
    """Format a normalized label for display."""
    normalized = normalize_word_label(label)
    if normalized == "i":
        return "I"
    return " ".join(part.capitalize() if part != "i" else "I" for part in normalized.split())


def text_to_sign_vocabulary(config: dict[str, Any]) -> list[str]:
    """Read supported Text-to-Sign word labels from configured MP4 filenames."""
    words_dir = Path(config["avatar"]["word_videos_dir"])
    if not words_dir.exists():
        return []

    labels: dict[str, str] = {}
    for video_path in words_dir.rglob("*.mp4"):
        normalized = normalize_word_label(video_path.stem)
        if normalized:
            labels[normalized] = display_word_label(normalized)
    return [labels[key] for key in sorted(labels)]

