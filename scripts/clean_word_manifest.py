"""Create a cleaned Sign-to-Text word manifest by removing unreadable videos."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2

from src.config import load_config


def main() -> None:
    config = load_config("config.yaml")
    manifest_path = Path(config["avatar"]["sign_to_text_manifest"])
    if not manifest_path.exists() and manifest_path.name.endswith("_clean.csv"):
        manifest_path = manifest_path.with_name(manifest_path.name.replace("_clean.csv", ".csv"))
    clean_path = manifest_path.with_name(f"{manifest_path.stem}_clean.csv")
    rejected_path = manifest_path.with_name(f"{manifest_path.stem}_rejected.csv")

    rows: list[dict[str, str]] = []
    rejected: list[dict[str, str]] = []
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            path = Path(row["path"])
            if not path.is_absolute():
                path = Path.cwd() / path
            ok, reason = _is_readable_video(path, int(config["model"]["word_frame_count"]))
            if ok:
                rows.append(row)
            else:
                rejected.append({**row, "reason": reason})

    with clean_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "label", "video_id", "split"])
        writer.writeheader()
        writer.writerows(rows)

    with rejected_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "label", "video_id", "split", "reason"])
        writer.writeheader()
        writer.writerows(rejected)

    print(f"Readable videos: {len(rows)}")
    print(f"Rejected videos: {len(rejected)}")
    print(f"Wrote clean manifest: {clean_path}")
    print(f"Wrote rejected manifest: {rejected_path}")


def _is_readable_video(path: Path, min_frames: int) -> tuple[bool, str]:
    if not path.exists():
        return False, "missing file"
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        return False, "could not open"
    frames = 0
    try:
        while frames < max(1, min_frames):
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            frames += 1
    finally:
        capture.release()
    if frames == 0:
        return False, "no readable frames"
    return True, ""


if __name__ == "__main__":
    main()
