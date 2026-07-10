#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PROJECT_ROOT="$(pwd)"
VENV_DIR="${PROJECT_ROOT}/.venv-mac"
PYTHON_BIN="${VENV_DIR}/bin/python"

if ! command -v python3 >/dev/null 2>&1; then
  echo "Python 3 не найден. Установи Python 3.11/3.12 для macOS и запусти скрипт снова."
  exit 1
fi

if [ ! -x "${PYTHON_BIN}" ]; then
  echo "Создаю отдельное окружение для macOS: .venv-mac"
  python3 -m venv "${VENV_DIR}"
fi

echo "Обновляю pip и ставлю зависимости..."
"${PYTHON_BIN}" -m pip install --upgrade pip
"${PYTHON_BIN}" -m pip install -r requirements.txt
"${PYTHON_BIN}" -m pip install -e .

if ! "${PYTHON_BIN}" -c "import transformers, torch" >/dev/null 2>&1; then
  echo "Ставлю зависимости для локального HF-слоя..."
  "${PYTHON_BIN}" -m pip install "torch>=2.3" "transformers>=4.42" "sentencepiece>=0.2" "accelerate>=0.33"
fi

if [ -f "${PROJECT_ROOT}/models/toxicity_model_policy_clean.joblib" ]; then
  export MODEL_PATH="${PROJECT_ROOT}/models/toxicity_model_policy_clean.joblib"
elif [ -f "${PROJECT_ROOT}/models/toxicity_model.joblib" ]; then
  export MODEL_PATH="${PROJECT_ROOT}/models/toxicity_model.joblib"
else
  echo "Не найдена модель слоя 2 в models/. Сайт запустится, но API будет без регрессионной модели."
fi

export HF_HOME="${PROJECT_ROOT}/models/huggingface"
export HF_TOXIC_MODEL="${HF_TOXIC_MODEL:-s-nlp/russian_toxicity_classifier}"

if [ -d "${HF_HOME}/models--s-nlp--russian_toxicity_classifier" ]; then
  export LLM_PROVIDER="${LLM_PROVIDER:-hf_local}"
else
  echo "Локальная HF-модель не найдена в models/huggingface. Запускаю без HF-слоя."
  echo "Чтобы скачать ее на Mac: ${PYTHON_BIN} scripts/download_hf_model.py --model s-nlp/russian_toxicity_classifier --cache-dir models/huggingface"
  export LLM_PROVIDER="${LLM_PROVIDER:-disabled}"
fi

export REGRESSION_BLOCK_THRESHOLD="${REGRESSION_BLOCK_THRESHOLD:-0.5}"
export REGRESSION_REVIEW_THRESHOLD="${REGRESSION_REVIEW_THRESHOLD:-0.35}"

echo
echo "Сайт запускается:"
echo "  http://127.0.0.1:8000"
echo "Документация API:"
echo "  http://127.0.0.1:8000/docs"
echo
echo "Остановить сервер: Ctrl+C"
echo

"${PYTHON_BIN}" -m uvicorn app.main:app --host 127.0.0.1 --port 8000
