#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <qa.json> <dataset-root> [output.json]"
  exit 1
fi

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
INPUT_PATH="$1"
DATASET_ROOT="$2"
OUTPUT_PATH="${3:-${ROOT_DIR}/outputs/sama.json}"

if [[ -f "${ROOT_DIR}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${ROOT_DIR}/.env"
  set +a
fi

: "${API_MODEL_NAME:?Set API_MODEL_NAME in the environment or .env}"
: "${API_MODEL_NAME_VLM:?Set API_MODEL_NAME_VLM in the environment or .env}"

if [[ -z "${API_KEY:-${OPENAI_API_KEY:-}}" ]]; then
  echo "Set API_KEY or OPENAI_API_KEY for the planner."
  exit 1
fi

mkdir -p "$(dirname "$OUTPUT_PATH")"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"

python3 -m multivideo_agent.eval_multivideo \
  --dataset custom \
  --dataset_folder "$DATASET_ROOT" \
  --input_path "$INPUT_PATH" \
  --prompt_version "${PROMPT_VERSION:-default}" \
  --clip_fps "${CLIP_FPS:-2}" \
  --max_frames "${MAX_FRAMES:-32}" \
  --max_workers "${MAX_WORKERS:-1}" \
  --save_path "$OUTPUT_PATH"
