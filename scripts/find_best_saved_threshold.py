from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from sklearn.metrics import roc_auc_score


def main() -> int:
    parser = argparse.ArgumentParser(description="Find exact best threshold from saved prediction scores.")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--score-column", default="chain_score")
    args = parser.parse_args()

    with Path(args.predictions).open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    y = [int(row["label"]) for row in rows]
    scores = [float(row[args.score_column]) for row in rows]
    pairs = sorted(zip(scores, y), key=lambda item: item[0], reverse=True)
    positives = sum(y)
    negatives = len(y) - positives

    tp = 0
    fp = 0
    fn = positives
    tn = negatives
    best_acc = None
    best_f1 = None
    auc = float(roc_auc_score(y, scores))

    index = 0
    while index < len(pairs):
        threshold = pairs[index][0]
        while index < len(pairs) and pairs[index][0] == threshold:
            _, label = pairs[index]
            if label == 1:
                tp += 1
                fn -= 1
            else:
                fp += 1
                tn -= 1
            index += 1
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 2 * precision * recall / max(1e-12, precision + recall)
        item = {
            "threshold": threshold,
            "accuracy": float((tp + tn) / max(1, len(y))),
            "precision_toxic": float(precision),
            "recall_toxic": float(recall),
            "f1_toxic": float(f1),
            "roc_auc": auc,
            "tp": int(tp),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
        }
        if best_acc is None or item["accuracy"] > best_acc["accuracy"]:
            best_acc = item
        if best_f1 is None or item["f1_toxic"] > best_f1["f1_toxic"]:
            best_f1 = item

    print(json.dumps({"best_accuracy": best_acc, "best_f1": best_f1}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
