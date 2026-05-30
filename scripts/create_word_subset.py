"""Create a smaller word manifest from the highest-sample labels."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="assets/sign_to_text/word_manifest_clean.csv")
    parser.add_argument("--output", default="assets/sign_to_text/word_manifest_top15.csv")
    parser.add_argument("--count", type=int, default=15)
    args = parser.parse_args()

    source = Path(args.source)
    rows = list(csv.DictReader(source.open("r", encoding="utf-8", newline="")))
    counts = Counter(row["label"] for row in rows)
    selected = [label for label, _ in counts.most_common(args.count)]
    selected_rows = [row for row in rows if row["label"] in set(selected)]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "label", "video_id", "split"])
        writer.writeheader()
        writer.writerows(selected_rows)

    print(f"Selected {len(selected)} words and {len(selected_rows)} videos")
    for label in selected:
        print(f"{label}: {counts[label]}")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
