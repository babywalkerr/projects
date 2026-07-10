from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_URL = "https://raw.githubusercontent.com/LDNOOBW/List-of-Dirty-Naughty-Obscene-and-Otherwise-Bad-Words/master/ru"


def main() -> int:
    parser = argparse.ArgumentParser(description="Download external Russian profanity list.")
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--output", default=str(PROJECT_ROOT / "data" / "profanity_external_ru.txt"))
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(args.url, timeout=30) as response:
        raw = response.read().decode("utf-8")

    entries = [part.strip() for part in raw.replace("\n", " ").split(" ") if part.strip()]
    output.write_text("\n".join(sorted(set(entries))) + "\n", encoding="utf-8")
    print(f"Saved {len(set(entries))} entries to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

