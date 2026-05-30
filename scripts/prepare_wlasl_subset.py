"""Prepare a WLASL subset for the supported Text-to-Sign word vocabulary."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.word_vocab import compact_word_key, display_word_label, normalize_word_label, text_to_sign_vocabulary


VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--dataset-path", default=None, help="Existing WLASL directory or ZIP/archive path.")
    parser.add_argument("--download", action="store_true", help="Download WLASL with kagglehub first.")
    parser.add_argument("--dry-run", action="store_true", help="Only report matching words and video counts.")
    args = parser.parse_args()

    config = load_config(args.config)
    dataset_path = Path(args.dataset_path) if args.dataset_path else None
    if args.download:
        import kagglehub

        dataset_path = Path(kagglehub.dataset_download("risangbaskoro/wlasl-processed"))
    if dataset_path is None:
        dataset_path = _default_kaggle_cache()
    if dataset_path is None or not dataset_path.exists():
        raise FileNotFoundError("WLASL dataset not found. Pass --dataset-path or use --download.")

    vocabulary = text_to_sign_vocabulary(config)
    phrase_labels = [label for label in vocabulary if " " in normalize_word_label(label)]
    word_vocabulary = [label for label in vocabulary if " " not in normalize_word_label(label)]
    target_by_key = {compact_word_key(label): display_word_label(label) for label in word_vocabulary}
    if not target_by_key:
        raise ValueError("No Text-to-Sign MP4 vocabulary found.")

    output_dir = Path(config["avatar"]["sign_to_text_words_dir"])
    manifest_path = Path(config["avatar"]["sign_to_text_manifest"])
    records = _prepare_from_zip(dataset_path, target_by_key, output_dir, dry_run=args.dry_run)
    if records is None:
        records = _prepare_from_directory(dataset_path, target_by_key, output_dir, dry_run=args.dry_run)

    matched_words = sorted({record["label"] for record in records})
    missing_words = sorted(set(target_by_key.values()) - set(matched_words))
    print(f"Text-to-Sign vocabulary: {len(vocabulary)} MP4 labels")
    if phrase_labels:
        print(f"Skipping {len(phrase_labels)} phrase label(s) for WLASL word training: {', '.join(phrase_labels)}")
    print(f"WLASL target vocabulary: {len(target_by_key)} single-word labels")
    print(f"Matched in WLASL metadata/files: {len(matched_words)} words, {len(records)} videos")
    if missing_words:
        print("Missing from WLASL subset:")
        print(", ".join(missing_words))

    if not args.dry_run:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with manifest_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["path", "label", "video_id", "split"])
            writer.writeheader()
            writer.writerows(records)
        print(f"Wrote manifest: {manifest_path}")


def _default_kaggle_cache() -> Path | None:
    root = Path.home() / ".cache" / "kagglehub" / "datasets" / "risangbaskoro" / "wlasl-processed"
    if not root.exists():
        return None
    archives = sorted(root.glob("*.archive"), key=lambda path: path.stat().st_mtime, reverse=True)
    return archives[0] if archives else root


def _prepare_from_zip(
    dataset_path: Path,
    target_by_key: dict[str, str],
    output_dir: Path,
    dry_run: bool = False,
) -> list[dict[str, str]] | None:
    if dataset_path.is_dir():
        return None
    if not zipfile.is_zipfile(dataset_path):
        raise ValueError(
            f"{dataset_path} is not a complete readable archive yet. "
            "Run this script again with --download after the Kaggle download finishes."
        )

    with zipfile.ZipFile(dataset_path) as archive:
        metadata_name = _find_metadata_name(archive.namelist())
        entries = json.loads(archive.read(metadata_name).decode("utf-8"))
        video_names = [name for name in archive.namelist() if Path(name).suffix.lower() in VIDEO_EXTENSIONS]
        video_by_id = {Path(name).stem: name for name in video_names}
        return _collect_records(entries, target_by_key, output_dir, dry_run, lambda video_id: video_by_id.get(video_id), archive)


def _prepare_from_directory(
    dataset_path: Path,
    target_by_key: dict[str, str],
    output_dir: Path,
    dry_run: bool = False,
) -> list[dict[str, str]]:
    metadata_path = _find_metadata_path(dataset_path)
    entries = json.loads(metadata_path.read_text(encoding="utf-8"))
    video_paths = [path for path in dataset_path.rglob("*") if path.suffix.lower() in VIDEO_EXTENSIONS]
    video_by_id = {path.stem: path for path in video_paths}
    return _collect_records(entries, target_by_key, output_dir, dry_run, lambda video_id: video_by_id.get(video_id), None)


def _find_metadata_name(names: list[str]) -> str:
    candidates = [name for name in names if Path(name).name.lower().startswith("wlasl") and name.lower().endswith(".json")]
    if not candidates:
        candidates = [name for name in names if name.lower().endswith(".json")]
    if not candidates:
        raise FileNotFoundError("No WLASL JSON metadata found in archive.")
    return candidates[0]


def _find_metadata_path(dataset_path: Path) -> Path:
    candidates = [path for path in dataset_path.rglob("*.json") if path.name.lower().startswith("wlasl")]
    if not candidates:
        candidates = list(dataset_path.rglob("*.json"))
    if not candidates:
        raise FileNotFoundError("No WLASL JSON metadata found in dataset directory.")
    return candidates[0]


def _collect_records(
    entries: list[dict[str, Any]],
    target_by_key: dict[str, str],
    output_dir: Path,
    dry_run: bool,
    resolve_video: Any,
    archive: zipfile.ZipFile | None,
) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    output_dir.mkdir(parents=True, exist_ok=True)
    for entry in entries:
        gloss = normalize_word_label(str(entry.get("gloss", "")))
        label = target_by_key.get(compact_word_key(gloss))
        if label is None:
            continue
        for instance in entry.get("instances", []):
            video_id = str(instance.get("video_id", "")).strip()
            source = resolve_video(video_id)
            if not video_id or source is None:
                continue
            split = str(instance.get("split", "train"))
            destination = output_dir / compact_word_key(label) / f"{video_id}.mp4"
            if not dry_run and not destination.exists():
                destination.parent.mkdir(parents=True, exist_ok=True)
                if archive is not None:
                    with archive.open(str(source)) as src, destination.open("wb") as dst:
                        shutil.copyfileobj(src, dst)
                else:
                    shutil.copy2(Path(source), destination)
            records.append(
                {
                    "path": str(destination.as_posix()),
                    "label": label,
                    "video_id": video_id,
                    "split": split,
                }
            )
    return records


if __name__ == "__main__":
    main()
