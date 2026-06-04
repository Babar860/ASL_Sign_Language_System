"""Prepare a WLASL manifest and word list for the highest-sample glosses."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.word_vocab import compact_word_key, display_word_label


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", default="archive (1).zip")
    parser.add_argument("--count", type=int, default=1500)
    parser.add_argument("--output", default="assets/sign_to_text/wlasl_top1500_manifest.csv")
    parser.add_argument("--words-output", default="assets/sign_to_text/wlasl_top1500_words.txt")
    parser.add_argument("--extract-dir", default="assets/sign_to_text/wlasl_top1500_words")
    parser.add_argument("--max-samples-per-word", type=int, default=0)
    parser.add_argument("--copy-videos", action="store_true")
    args = parser.parse_args()

    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists() or not zipfile.is_zipfile(dataset_path):
        raise FileNotFoundError(f"Readable WLASL zip not found: {dataset_path}")

    output_path = Path(args.output)
    words_output = Path(args.words_output)
    extract_dir = Path(args.extract_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    words_output.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(dataset_path) as archive:
        entries = json.loads(archive.read("WLASL_v0.3.json").decode("utf-8"))
        video_names = {Path(name).stem: name for name in archive.namelist() if name.lower().endswith(".mp4")}
        counts = _count_available_videos(entries, video_names)
        selected = [label for label, _ in counts.most_common(args.count)]
        selected_set = set(selected)
        records = _records(
            entries,
            selected_set,
            video_names,
            archive,
            extract_dir,
            copy_videos=args.copy_videos,
            max_samples_per_word=max(0, int(args.max_samples_per_word)),
        )

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "label", "video_id", "split", "frame_start", "frame_end", "bbox"])
        writer.writeheader()
        writer.writerows(records)

    words_output.write_text("\n".join(selected) + "\n", encoding="utf-8")

    print(f"Selected words: {len(selected)}")
    print(f"Manifest rows: {len(records)}")
    print(f"Wrote manifest: {output_path}")
    print(f"Wrote word list: {words_output}")
    if not args.copy_videos:
        print("Videos were not copied. Re-run with --copy-videos when you are ready to extract the large dataset.")


def _count_available_videos(entries: list[dict[str, Any]], video_names: dict[str, str]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for entry in entries:
        label = display_word_label(str(entry.get("gloss", "")))
        for instance in entry.get("instances", []):
            if str(instance.get("video_id", "")) in video_names:
                counts[label] += 1
    return counts


def _records(
    entries: list[dict[str, Any]],
    selected: set[str],
    video_names: dict[str, str],
    archive: zipfile.ZipFile,
    extract_dir: Path,
    copy_videos: bool,
    max_samples_per_word: int = 0,
) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for entry in entries:
        label = display_word_label(str(entry.get("gloss", "")))
        if label not in selected:
            continue
        label_dir = compact_word_key(label)
        copied_for_label = 0
        for instance in entry.get("instances", []):
            if max_samples_per_word and copied_for_label >= max_samples_per_word:
                break
            video_id = str(instance.get("video_id", ""))
            archive_name = video_names.get(video_id)
            if archive_name is None:
                continue
            destination = extract_dir / label_dir / f"{video_id}.mp4"
            if copy_videos and not destination.exists():
                destination.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(archive_name) as src, destination.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
            records.append(
                {
                    "path": str(destination.as_posix()),
                    "label": label,
                    "video_id": video_id,
                    "split": str(instance.get("split", "train")),
                    "frame_start": str(instance.get("frame_start", 1)),
                    "frame_end": str(instance.get("frame_end", -1)),
                    "bbox": json.dumps(instance.get("bbox", [])),
                }
            )
            copied_for_label += 1
    return records


if __name__ == "__main__":
    main()
