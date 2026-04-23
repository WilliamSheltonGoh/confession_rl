#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════
#  infer.sh  MODEL_TYPE  DATASET  [OPTIONS...]
#
#  MODEL_TYPE : base   → load base model directly
#               train  → load latest VERL checkpoint
#                        (supports full HF checkpoint OR base+LoRA adapter)
#
#  DATASET    : kandk  → kandk inference.parquet
#               boolq  → boolq_inference.parquet
# ══════════════════════════════════════════════════════════════
set -euo pipefail



usage() {
    echo ""
    echo "  Usage: bash infer.sh <MODEL_TYPE> <DATASET> [extra args]"
    echo ""
    echo "    MODEL_TYPE  : base | train"
    echo "    DATASET     : kandk | boolq"
    echo ""
    exit 1
}

[[ $# -lt 2 ]] && usage

MODEL_TYPE="${1,,}"
DATASET="${2,,}"
shift 2

if [[ "$MODEL_TYPE" != "base" && "$MODEL_TYPE" != "train" ]]; then
    echo "[ERROR] MODEL_TYPE must be 'base' or 'train', got: '$MODEL_TYPE'"
    usage
fi

if [[ "$DATASET" != "kandk" && "$DATASET" != "boolq" ]]; then
    echo "[ERROR] DATASET must be 'kandk' or 'boolq', got: '$DATASET'"
    usage
fi

# ── Paths ─────────────────────────────────────────────────────
BASE_MODEL="${BASE_MODEL:-/root/autodl-tmp/models/Qwen3-4B-Instruct-2507}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/root/autodl-tmp/checkpoints/self_rl_data}"
KANDK_DATA="${KANDK_DATA:-./data/kandk_data/inference.parquet}"
BOOLQ_DATA="${BOOLQ_DATA:-./data/boolq_data/boolq_inference.parquet}"
OUTPUT_DIR="${OUTPUT_DIR:-./inference_results}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY_SCRIPT="${SCRIPT_DIR}/infertest.py"

# ── Fix libgomp warning if OMP_NUM_THREADS is invalid ─────────
if [[ -z "${OMP_NUM_THREADS:-}" || ! "${OMP_NUM_THREADS}" =~ ^[0-9]+$ ]]; then
    export OMP_NUM_THREADS=1
fi

echo ""
echo "══════════════════════════════════════════════════════"
echo "  model_type : ${MODEL_TYPE}"
echo "  dataset    : ${DATASET}"
echo "  base_model : ${BASE_MODEL}"
echo "  checkpoint : ${CHECKPOINT_DIR}"
echo "══════════════════════════════════════════════════════"

if [[ ! -f "$PY_SCRIPT" ]]; then
    echo "[ERROR] infertest.py not found at: ${PY_SCRIPT}"
    exit 1
fi

if [[ "$MODEL_TYPE" == "base" && ! -d "$BASE_MODEL" ]]; then
    echo "[ERROR] Base model directory not found: ${BASE_MODEL}"
    exit 1
fi

if [[ "$MODEL_TYPE" == "train" && ! -d "$CHECKPOINT_DIR" ]]; then
    echo "[ERROR] Checkpoint directory not found: ${CHECKPOINT_DIR}"
    exit 1
fi

if [[ "$DATASET" == "kandk" && ! -f "$KANDK_DATA" ]]; then
    echo "[ERROR] K&K inference parquet not found: ${KANDK_DATA}"
    exit 1
fi

if [[ "$DATASET" == "boolq" && ! -f "$BOOLQ_DATA" ]]; then
    echo "[ERROR] BoolQ inference parquet not found: ${BOOLQ_DATA}"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

python "${PY_SCRIPT}" \
    --model_type      "${MODEL_TYPE}" \
    --dataset         "${DATASET}" \
    --base_model_path "${BASE_MODEL}" \
    --checkpoint_dir  "${CHECKPOINT_DIR}" \
    --kandk_data      "${KANDK_DATA}" \
    --boolq_data      "${BOOLQ_DATA}" \
    --output_dir      "${OUTPUT_DIR}" \
    "$@"
