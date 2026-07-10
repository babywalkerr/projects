from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Export errors from saved chain scores at a threshold.")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    with Path(args.predictions).open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    errors = []
    for row in rows:
        label = int(row["label"])
        score = float(row["chain_score"])
        pred = 1 if score >= args.threshold else 0
        if pred == label:
            continue
        errors.append(
            {
                **row,
                "prediction": pred,
                "error_type": "false_positive" if label == 0 else "false_negative",
            }
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(errors[0].keys()) if errors else list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(errors)
    print(f"wrote {len(errors)} errors to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
