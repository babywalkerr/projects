from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from toxicity_moderation.config import Settings
from toxicity_moderation.llm import HuggingFaceLocalClassifier


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate local HF toxicity model on held-out chain errors.")
    parser.add_argument("--errors", default=str(PROJECT_ROOT / "outputs" / "policy_clean_holdout" / "chain_errors.csv"))
    parser.add_argument("--output", default=str(PROJECT_ROOT / "outputs" / "policy_clean_holdout" / "hf_error_review.csv"))
    parser.add_argument("--summary", default=str(PROJECT_ROOT / "outputs" / "policy_clean_holdout" / "hf_error_review.json"))
    args = parser.parse_args()

    settings = Settings(llm_provider="hf_local")
    hf = HuggingFaceLocalClassifier(model=settings.hf_toxic_model, cache_dir=str(settings.hf_cache_dir))

    with Path(args.errors).open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    reviewed: list[dict[str, Any]] = []
    fixed = 0
    still_wrong = 0
    skipped = 0
    for row in rows:
        result = hf.moderate(row["text"])
        if result.skipped:
            skipped += 1
            hf_pred = ""
            hf_conf = 0.0
        else:
            hf_pred = 1 if result.toxic else 0
            hf_conf = result.confidence
            if hf_pred == int(row["label"]):
                fixed += 1
            else:
                still_wrong += 1
        reviewed.append(
            {
                **row,
                "hf_prediction": hf_pred,
                "hf_confidence": hf_conf,
                "hf_would_fix": hf_pred != "" and int(hf_pred) == int(row["label"]),
            }
        )

    summary = {
        "input_errors": len(rows),
        "hf_fixed_errors": fixed,
        "hf_still_wrong": still_wrong,
        "hf_skipped": skipped,
        "note": "This evaluates HF only on known chain errors, so it is diagnostic and not an official standalone metric.",
    }
    write_csv(Path(args.output), reviewed)
    Path(args.summary).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
