from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from toxicity_moderation.data import load_dataset


def score_batch(pipeline, texts: list[str]) -> list[float]:
    if hasattr(pipeline, "predict_proba"):
        proba = pipeline.predict_proba(texts)
        classes = getattr(pipeline, "classes_", None)
        index = list(classes).index(1) if classes is not None and 1 in classes else -1
        return [float(row[index]) for row in proba]
    if hasattr(pipeline, "decision_function"):
        return [1.0 / (1.0 + math.exp(-float(value))) for value in pipeline.decision_function(texts)]
    return [min(1.0, max(0.0, float(value))) for value in pipeline.predict(texts)]


def main() -> int:
    parser = argparse.ArgumentParser(description="Count high-confidence model/label disagreement candidates.")
    parser.add_argument("--data", default=str(PROJECT_ROOT / "data" / "labeled_policy_clean.csv"))
    parser.add_argument("--model-path", default=str(PROJECT_ROOT / "models" / "toxicity_model_policy_clean.joblib"))
    parser.add_argument("--batch-size", type=int, default=8192)
    args = parser.parse_args()

    import joblib

    dataset = load_dataset(args.data, text_column="text", label_column="label")
    bundle = joblib.load(args.model_path)
    pipeline = bundle["pipeline"]

    scores: list[float] = []
    for start in range(0, len(dataset.texts), args.batch_size):
        scores.extend(score_batch(pipeline, dataset.texts[start : start + args.batch_size]))

    thresholds = [0.80, 0.85, 0.90, 0.95, 0.97, 0.98, 0.99]
    report = {}
    for high in thresholds:
        low = 1.0 - high
        fp_like = sum(1 for y, s in zip(dataset.labels, scores) if y == 0 and s >= high)
        fn_like = sum(1 for y, s in zip(dataset.labels, scores) if y == 1 and s <= low)
        report[str(high)] = {
            "label0_score_ge_high": fp_like,
            "label1_score_le_low": fn_like,
            "total": fp_like + fn_like,
        }

    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
