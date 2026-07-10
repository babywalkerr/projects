from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Check final quality gate for the moderation pipeline.")
    parser.add_argument("--metrics", default="outputs/hf_holdout/hf_holdout_metrics.json")
    parser.add_argument("--min-accuracy", type=float, default=0.99)
    args = parser.parse_args()

    report = json.loads(Path(args.metrics).read_text(encoding="utf-8"))
    metrics = report["chain_lexicon_plus_hf_metrics"]
    accuracy = float(metrics["accuracy"])
    if accuracy < args.min_accuracy:
        raise SystemExit(f"FAILED: accuracy={accuracy:.6f} < {args.min_accuracy:.6f}")
    print(f"PASSED: accuracy={accuracy:.6f} >= {args.min_accuracy:.6f}")
    print(f"F1 toxic={float(metrics['f1_toxic']):.6f}")
    print(f"ROC-AUC={float(metrics['roc_auc']):.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
