from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from toxicity_moderation.pipeline import ModerationPipeline


def main() -> int:
    pipeline = ModerationPipeline()
    comments = [
        "Спасибо за помощь, все понятно.",
        "Ты идиот и ничего не понимаешь.",
        "Что за дерьмо вы опять сделали.",
    ]
    for comment in comments:
        print(json.dumps({"text": comment, "result": pipeline.moderate(comment)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

