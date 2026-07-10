from __future__ import annotations

import argparse
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description="Download a Hugging Face text-classification model locally.")
    parser.add_argument("--model", default="s-nlp/russian_toxicity_classifier")
    parser.add_argument("--cache-dir", default=str(PROJECT_ROOT / "models" / "huggingface"))
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

    print(f"Downloading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, cache_dir=str(cache_dir))
    print(f"Downloading model: {args.model}")
    model = AutoModelForSequenceClassification.from_pretrained(args.model, cache_dir=str(cache_dir))
    print("Running a short local inference check")
    classifier = pipeline("text-classification", model=model, tokenizer=tokenizer, truncation=True)
    print(classifier("Ты мне нравишься. Я тебя люблю"))
    print(classifier("Ты идиот и ничего не понимаешь."))
    print(f"Saved Hugging Face cache under: {cache_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
