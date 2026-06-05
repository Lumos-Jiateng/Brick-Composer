#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# run_qwen3_8b.sh
# Evaluate the base Qwen3-VL-8B-Instruct model (port 8001) on the full
# Designer_supervision_assembly_testing.json dataset.
#
# Model   : Qwen/Qwen3-VL-8B-Instruct  (served as qwen-3-vl)
# Port    : 8001
# Results : results_pose_qwen3_8b/<task_id>/result.json
# Summary : results_pose_qwen3_8b/summary_<timestamp>.json
#
# Inference is resumable: already-completed tasks are skipped automatically.
# After inference, runs success-rate evaluation (t=10 LDU, r=45°) and
# symmetry-aware re-aggregation.
#
# Usage:
#   bash run_qwen3_8b.sh
#   bash run_qwen3_8b.sh --thinking        # enable chain-of-thought
#   bash run_qwen3_8b.sh --obj 77756       # single model
#   bash run_qwen3_8b.sh --limit 50        # quick test
# ---------------------------------------------------------------------------

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON="${PYTHON:-/shared/nas/data/m1/jiateng5/anaconda3/envs/vllm/bin/python3.12}"
MODEL="qwen3_8b"
RESULTS_DIR="results_pose_qwen3_8b"

echo "============================================================"
echo "  Inference  :  $MODEL"
echo "  Endpoint   :  http://172.22.225.5:8001/v1  (qwen-3-vl / Qwen3-VL-8B-Instruct)"
echo "  Results    :  $RESULTS_DIR"
echo "============================================================"

# --- Inference ---------------------------------------------------------------
"$PYTHON" evaluate_assembly.py --model "$MODEL" "$@"

echo ""
echo "============================================================"
echo "  Success-rate evaluation  (t=10 LDU, r=45°)"
echo "============================================================"

"$PYTHON" compute_success_rate.py \
    --results_dir "$RESULTS_DIR" \
    --t_ldu 10 \
    --r_deg 45

echo ""
echo "============================================================"
echo "  Symmetry-aware re-aggregation"
echo "============================================================"

"$PYTHON" reeval_with_symmetry.py \
    --results_dir "$RESULTS_DIR"

echo ""
echo "Done.  All outputs are under: $SCRIPT_DIR/$RESULTS_DIR"
