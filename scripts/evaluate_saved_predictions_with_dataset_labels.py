from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from toxicity_moderation.data import load_dataset
from toxicity_moderation.config import Settings
from toxicity_moderation.lexicon import ProfanityFilter


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate saved prediction scores against current dataset labels.")
    parser.add_argument("--data", default=str(PROJECT_ROOT / "data" / "labeled_policy_clean.csv"))
    parser.add_argument("--predictions", default=str(PROJECT_ROOT / "outputs" / "hf_holdout" / "hf_holdout_predictions.csv"))
    parser.add_argument("--threshold", type=float, default=0.32)
    args = parser.parse_args()

    dataset = load_dataset(args.data, text_column="text", label_column="label")
    lexicon = ProfanityFilter(Settings().profanity_paths)
    with Path(args.predictions).open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    y = [dataset.labels[int(row["dataset_index"])] for row in rows]
    scores = [
        1.0 if lexicon.find(row["text"]).blocked else float(row["hf_score"])
        for row in rows
    ]
    pred = [1 if score >= args.threshold else 0 for score in scores]
    report = {
        "rows": len(rows),
        "threshold": args.threshold,
        "accuracy": float(accuracy_score(y, pred)),
        "precision_toxic": float(precision_score(y, pred, zero_division=0)),
        "recall_toxic": float(recall_score(y, pred, zero_division=0)),
        "f1_toxic": float(f1_score(y, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, scores)),
        "tp": int(sum(1 for yt, yp in zip(y, pred) if yt == 1 and yp == 1)),
        "tn": int(sum(1 for yt, yp in zip(y, pred) if yt == 0 and yp == 0)),
        "fp": int(sum(1 for yt, yp in zip(y, pred) if yt == 0 and yp == 1)),
        "fn": int(sum(1 for yt, yp in zip(y, pred) if yt == 1 and yp == 0)),
        "note": "Diagnostic: uses previously saved test comments/scores with current policy labels.",
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
