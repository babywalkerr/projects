from __future__ import annotations


def binary_metrics(y_true: list[int], y_pred: list[int]) -> dict[str, float]:
    tp = sum(1 for y, p in zip(y_true, y_pred) if y == 1 and p == 1)
    tn = sum(1 for y, p in zip(y_true, y_pred) if y == 0 and p == 0)
    fp = sum(1 for y, p in zip(y_true, y_pred) if y == 0 and p == 1)
    fn = sum(1 for y, p in zip(y_true, y_pred) if y == 1 and p == 0)
    total = max(1, len(y_true))
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    return {
        "accuracy": (tp + tn) / total,
        "precision_toxic": precision,
        "recall_toxic": recall,
        "f1_toxic": f1,
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
    }


def roc_auc_pairwise(y_true: list[int], scores: list[float]) -> float | None:
    positives = [score for y, score in zip(y_true, scores) if y == 1]
    negatives = [score for y, score in zip(y_true, scores) if y == 0]
    if not positives or not negatives:
        return None

    wins = 0.0
    pairs = 0
    for positive in positives:
        for negative in negatives:
            pairs += 1
            if positive > negative:
                wins += 1.0
            elif positive == negative:
                wins += 0.5
    return wins / pairs

