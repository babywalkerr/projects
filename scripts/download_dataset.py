from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import urllib.request
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_URLS = [
    "https://huggingface.co/datasets/AlexSham/Toxic_Russian_Comments/resolve/main/train.jsonl",
    "https://huggingface.co/datasets/AlexSham/Toxic_Russian_Comments/resolve/main/test.jsonl",
]

TEXT_KEYS = ("text", "comment", "comment_text", "message", "content")
LABEL_KEYS = ("label", "labels", "toxic", "is_toxic", "target", "class")


def pick_text(row: dict[str, Any]) -> str:
    for key in TEXT_KEYS:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    for value in row.values():
        if isinstance(value, str) and len(value.strip()) > 5 and "__label__" not in value:
            return value.strip()
    raise ValueError(f"Cannot find text field in row keys: {list(row)}")


def pick_label(row: dict[str, Any]) -> tuple[int, str]:
    for key in LABEL_KEYS:
        if key in row:
            return normalize_label(row[key])
    for value in row.values():
        if isinstance(value, (int, float, bool)):
            return normalize_label(value)
    for value in row.values():
        if isinstance(value, str) and "__label__" in value:
            return normalize_label(value)
    raise ValueError(f"Cannot find label field in row keys: {list(row)}")


def normalize_label(value: Any) -> tuple[int, str]:
    raw = str(value).strip()
    lower = raw.lower()
    if isinstance(value, bool):
        return (1 if value else 0), raw
    if isinstance(value, (int, float)):
        return (1 if float(value) >= 0.5 else 0), raw
    if any(token in lower for token in ("normal", "neutral", "non-toxic", "__label__normal")):
        return 0, raw
    if lower in {"0", "false", "no"}:
        return 0, raw
    return 1, raw


def iter_jsonl(url: str):
    request = urllib.request.Request(url, headers={"User-Agent": "toxicity-moderation-ru/0.1"})
    with urllib.request.urlopen(request, timeout=60) as response:
        for raw_line in response:
            line = raw_line.decode("utf-8").strip()
            if line:
                yield json.loads(line)


def main() -> int:
    parser = argparse.ArgumentParser(description="Download open Russian toxic comments dataset into labeled.csv.")
    parser.add_argument("--output", default=str(PROJECT_ROOT / "labeled.csv"))
    parser.add_argument("--limit", type=int, default=12000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--urls", nargs="*", default=DEFAULT_URLS)
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    errors = 0
    for url in args.urls:
        print(f"Reading {url}")
        for item in iter_jsonl(url):
            try:
                text = pick_text(item)
                label, source_label = pick_label(item)
                rows.append({"text": text, "label": label, "source_label": source_label})
            except Exception:
                errors += 1
            if args.limit and len(rows) >= args.limit:
                break
        if args.limit and len(rows) >= args.limit:
            break

    if not rows:
        raise SystemExit("No rows were downloaded. Check network access or dataset URL.")

    random.Random(args.seed).shuffle(rows)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["text", "label", "source_label"])
        writer.writeheader()
        writer.writerows(rows)

    toxic = sum(row["label"] for row in rows)
    print(f"Saved {len(rows)} rows to {output}")
    print(f"Toxic: {toxic}; normal: {len(rows) - toxic}; skipped rows: {errors}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
