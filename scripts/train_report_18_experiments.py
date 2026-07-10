from __future__ import annotations

import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from toxicity_moderation.data import load_regression_dataset
from toxicity_moderation.normalization import tokenize
from toxicity_moderation.text_features import preprocess_text


REGRESSORS = (
    "SVR",
    "SGDRegressor",
    "KNeighborsRegressor",
    "GaussianProcessRegressor",
    "PLSRegression",
    "DecisionTreeRegressor",
)


def require_deps() -> dict[str, Any]:
    import joblib
    import numpy as np
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
    from sklearn.linear_model import SGDRegressor
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        mean_absolute_error,
        mean_squared_error,
        precision_score,
        r2_score,
        recall_score,
        roc_auc_score,
    )
    from sklearn.model_selection import train_test_split
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVR
    from sklearn.tree import DecisionTreeRegressor

    return locals()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "vectorizer",
        "model",
        "status",
        "mae",
        "rmse",
        "r2",
        "train_seconds",
        "inference_seconds",
        "accuracy",
        "precision_toxic",
        "recall_toxic",
        "f1_toxic",
        "roc_auc",
        "notes",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def binary_metrics(skl: dict[str, Any], y_true, y_score) -> dict[str, float | None]:
    y_pred = [1 if float(score) >= 0.5 else 0 for score in y_score]
    y_true_int = [int(value) for value in y_true]
    try:
        roc_auc = float(skl["roc_auc_score"](y_true_int, y_score))
    except ValueError:
        roc_auc = None
    return {
        "accuracy": float(skl["accuracy_score"](y_true_int, y_pred)),
        "precision_toxic": float(skl["precision_score"](y_true_int, y_pred, zero_division=0)),
        "recall_toxic": float(skl["recall_score"](y_true_int, y_pred, zero_division=0)),
        "f1_toxic": float(skl["f1_score"](y_true_int, y_pred, zero_division=0)),
        "roc_auc": roc_auc,
    }


def make_regressor(name: str, skl: dict[str, Any]):
    if name == "SVR":
        return skl["SVR"](kernel="linear", C=1.0)
    if name == "SGDRegressor":
        return skl["SGDRegressor"](
            loss="squared_error",
            penalty="elasticnet",
            alpha=0.0001,
            max_iter=2000,
            tol=1e-3,
            random_state=42,
        )
    if name == "KNeighborsRegressor":
        return skl["KNeighborsRegressor"](n_neighbors=5, weights="distance")
    if name == "GaussianProcessRegressor":
        return skl["GaussianProcessRegressor"](
            kernel=skl["DotProduct"]() + skl["WhiteKernel"](),
            random_state=42,
            normalize_y=True,
        )
    if name == "PLSRegression":
        return skl["PLSRegression"](n_components=2)
    if name == "DecisionTreeRegressor":
        return skl["DecisionTreeRegressor"](max_depth=16, random_state=42)
    raise ValueError(name)


def needs_dense(name: str) -> bool:
    return name in {"KNeighborsRegressor", "GaussianProcessRegressor", "PLSRegression"}


def needs_scaling(name: str) -> bool:
    return name in {"SVR", "SGDRegressor", "KNeighborsRegressor", "GaussianProcessRegressor", "PLSRegression"}


def train_local_fasttext(train_texts: list[str], model_path: Path, corpus_path: Path) -> None:
    import fasttext

    if model_path.exists():
        return
    rows = []
    for text in train_texts:
        tokens = tokenize(preprocess_text(text))[:256]
        if tokens:
            rows.append(" ".join(tokens))
    corpus_path.parent.mkdir(parents=True, exist_ok=True)
    corpus_path.write_text("\n".join(rows), encoding="utf-8")
    model = fasttext.train_unsupervised(
        str(corpus_path),
        model="skipgram",
        dim=100,
        epoch=10,
        minCount=1,
        minn=3,
        maxn=6,
        wordNgrams=2,
        thread=4,
        verbose=0,
    )
    model.save_model(str(model_path))


def fasttext_features(texts: list[str], model_path: Path, cache_path: Path, np):
    import fasttext

    if cache_path.exists():
        return np.load(cache_path)["features"]
    model = fasttext.load_model(str(model_path))
    dim = model.get_dimension()
    rows = []
    for text in texts:
        tokens = tokenize(preprocess_text(text))[:256]
        if not tokens:
            rows.append(np.zeros(dim, dtype="float32"))
            continue
        rows.append(np.mean([model.get_word_vector(token) for token in tokens], axis=0).astype("float32"))
    features = np.vstack(rows)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, features=features)
    return features


def rubert_features(texts: list[str], cache_path: Path, np):
    if cache_path.exists():
        return np.load(cache_path)["features"]

    import torch
    from transformers import AutoModel, AutoTokenizer

    model_name = "cointegrated/rubert-tiny2"
    cache_dir = str(PROJECT_ROOT / "models" / "huggingface")
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, local_files_only=True)
    model = AutoModel.from_pretrained(model_name, cache_dir=cache_dir, local_files_only=True)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    outputs = []
    batch_size = 32
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            if start % 512 == 0:
                print(f"  RuBERT embeddings: {start}/{len(texts)}", flush=True)
            batch = [preprocess_text(text) for text in texts[start : start + batch_size]]
            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=160,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            out = model(**encoded)
            mask = encoded["attention_mask"].unsqueeze(-1)
            pooled = (out.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            outputs.append(pooled.cpu().numpy().astype("float32"))

    features = np.vstack(outputs)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, features=features)
    return features


def prepare_features(vectorizer: str, x_train, x_test, skl: dict[str, Any], cache_dir: Path):
    np = skl["np"]
    if vectorizer == "TF-IDF":
        tfidf = skl["TfidfVectorizer"](
            preprocessor=preprocess_text,
            analyzer="word",
            ngram_range=(1, 2),
            min_df=1,
            max_features=50000,
            sublinear_tf=True,
        )
        return tfidf.fit_transform(x_train), tfidf.transform(x_test), "sparse", "TF-IDF word n-grams"

    if vectorizer == "FastText":
        model_path = PROJECT_ROOT / "models" / "fasttext_local_report.bin"
        corpus_path = cache_dir / "fasttext_corpus.txt"
        train_local_fasttext(x_train, model_path, corpus_path)
        all_texts = list(x_train) + list(x_test)
        features = fasttext_features(all_texts, model_path, cache_dir / "fasttext_features.npz", np)
        return (
            features[: len(x_train)],
            features[len(x_train) :],
            "dense",
            "Local unsupervised FastText trained on report train texts",
        )

    if vectorizer == "RuBERT-tiny2":
        all_texts = list(x_train) + list(x_test)
        features = rubert_features(all_texts, cache_dir / "rubert_tiny2_features.npz", np)
        return (
            features[: len(x_train)],
            features[len(x_train) :],
            "dense",
            "cointegrated/rubert-tiny2 mean pooled embeddings",
        )

    raise ValueError(vectorizer)


def fit_predict(vectorizer_name, x_train_features, x_test_features, y_train, regressor_name, feature_kind, skl):
    notes = []
    train_x = x_train_features
    test_x = x_test_features

    if feature_kind == "sparse" and needs_dense(regressor_name):
        n_components = min(120, max(1, train_x.shape[0] - 2), max(1, train_x.shape[1] - 1))
        svd = skl["TruncatedSVD"](n_components=n_components, random_state=42)
        train_x = svd.fit_transform(train_x)
        test_x = svd.transform(test_x)
        notes.append(f"TF-IDF reduced with TruncatedSVD({n_components})")

    if needs_scaling(regressor_name):
        scaler = skl["StandardScaler"](with_mean=feature_kind != "sparse")
        train_x = scaler.fit_transform(train_x)
        test_x = scaler.transform(test_x)
        notes.append("StandardScaler used")

    train_y = y_train
    if regressor_name == "GaussianProcessRegressor":
        limit = min(1500, len(y_train))
        train_x = train_x[:limit]
        train_y = y_train[:limit]
        notes.append(f"GPR trained on {limit}-row subset due O(n^3) cost")

    model = make_regressor(regressor_name, skl)
    start = time.perf_counter()
    model.fit(train_x, train_y)
    train_seconds = time.perf_counter() - start
    start = time.perf_counter()
    predictions = model.predict(test_x)
    inference_seconds = time.perf_counter() - start
    return predictions, train_seconds, inference_seconds, "; ".join(notes) or "-"


def main() -> int:
    skl = require_deps()
    np = skl["np"]
    output_dir = PROJECT_ROOT / "outputs"
    cache_dir = PROJECT_ROOT / "work" / "report_18_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_regression_dataset(PROJECT_ROOT / "labeled.csv", text_column="text", target_column="label")
    texts = dataset.texts[:12000]
    targets = dataset.targets[:12000]
    print(f"Dataset sample: {len(texts)} rows", flush=True)

    x_train, x_test, y_train, y_test = skl["train_test_split"](
        texts,
        targets,
        test_size=0.10,
        random_state=42,
        shuffle=True,
    )
    y_train_np = np.asarray(y_train, dtype=float)
    y_test_np = np.asarray(y_test, dtype=float)

    rows: list[dict[str, Any]] = []
    feature_cache: dict[str, tuple[Any, Any, str, str]] = {}

    for vectorizer in ("TF-IDF", "FastText", "RuBERT-tiny2"):
        print(f"\nPreparing {vectorizer} features", flush=True)
        feature_cache[vectorizer] = prepare_features(vectorizer, x_train, x_test, skl, cache_dir)
        x_train_features, x_test_features, feature_kind, vectorizer_note = feature_cache[vectorizer]
        for regressor in REGRESSORS:
            print(f"Training {vectorizer} + {regressor}", flush=True)
            try:
                pred, train_seconds, inference_seconds, notes = fit_predict(
                    vectorizer,
                    x_train_features,
                    x_test_features,
                    y_train_np,
                    regressor,
                    feature_kind,
                    skl,
                )
                pred = np.clip(np.ravel(pred).astype(float), 0.0, 1.0)
                metrics = binary_metrics(skl, y_test_np, pred)
                row = {
                    "vectorizer": vectorizer,
                    "model": regressor,
                    "status": "ok",
                    "mae": float(skl["mean_absolute_error"](y_test_np, pred)),
                    "rmse": float(math.sqrt(skl["mean_squared_error"](y_test_np, pred))),
                    "r2": float(skl["r2_score"](y_test_np, pred)),
                    "train_seconds": float(train_seconds),
                    "inference_seconds": float(inference_seconds),
                    **metrics,
                    "notes": f"{vectorizer_note}; {notes}",
                }
                print(f"  F1={row['f1_toxic']:.4f}, MAE={row['mae']:.4f}, ROC-AUC={row['roc_auc']:.4f}", flush=True)
            except Exception as exc:
                row = {
                    "vectorizer": vectorizer,
                    "model": regressor,
                    "status": "failed",
                    "mae": None,
                    "rmse": None,
                    "r2": None,
                    "train_seconds": None,
                    "inference_seconds": None,
                    "accuracy": None,
                    "precision_toxic": None,
                    "recall_toxic": None,
                    "f1_toxic": None,
                    "roc_auc": None,
                    "notes": str(exc),
                }
                print(f"  failed: {exc}", flush=True)
            rows.append(row)

    rows.sort(key=lambda row: (row["status"] != "ok", -(row["f1_toxic"] or -1)))
    write_csv(output_dir / "results_18_completed.csv", rows)
    (output_dir / "results_18_completed.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nSaved: {output_dir / 'results_18_completed.csv'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
