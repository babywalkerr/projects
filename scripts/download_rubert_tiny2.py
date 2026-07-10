from __future__ import annotations

import argparse
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description="Download RuBERT tiny2 embeddings model locally.")
    parser.add_argument("--model", default="cointegrated/rubert-tiny2")
    parser.add_argument("--cache-dir", default=str(PROJECT_ROOT / "models" / "huggingface"))
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    from transformers import AutoModel, AutoTokenizer

    print(f"Downloading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, cache_dir=str(cache_dir))
    print(f"Downloading model: {args.model}")
    model = AutoModel.from_pretrained(args.model, cache_dir=str(cache_dir))
    print(f"Loaded model type: {model.config.model_type}, hidden_size={model.config.hidden_size}")
    print(f"Tokenizer vocab size: {len(tokenizer)}")
    print(f"Saved Hugging Face cache under: {cache_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
