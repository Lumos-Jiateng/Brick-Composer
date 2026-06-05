#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# run_qwen35_27b.sh
# Evaluate the base Qwen3.5-27B model (port 8013) on the full
# Designer_supervision_assembly_testing.json dataset.
#
# Model   : Qwen/Qwen3.5-27B  (served with --reasoning-parser qwen3)
# Port    : 8013
# Results : results_pose_qwen35_27b/<task_id>/result.json
# Summary : results_pose_qwen35_27b/summary_<timestamp>.json
#
# Inference is resumable: already-completed tasks are skipped automatically.
# After inference, runs success-rate evaluation (t=10 LDU, r=45°) and
# symmetry-aware re-aggregation.
#
# Usage:
#   bash run_qwen35_27b.sh
#   bash run_qwen35_27b.sh --thinking        # enable chain-of-thought (recommended for this model)
#   bash run_qwen35_27b.sh --obj 77756       # single model
#   bash run_qwen35_27b.sh --limit 50        # quick test
# ---------------------------------------------------------------------------

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Use the vllm_qwen35 conda env that serves this model
PYTHON="${PYTHON:-/shared/nas/data/m1/jiateng5/anaconda3/envs/vllm_qwen35/bin/python3.12}"
# Fall back to vllm env if above is not available for client-side packages
if [ ! -f "$PYTHON" ]; then
    PYTHON="/shared/nas/data/m1/jiateng5/anaconda3/envs/vllm/bin/python3.12"
fi

MODEL="qwen35_27b"
RESULTS_DIR="results_pose_qwen35_27b"

echo "============================================================"
echo "  Inference  :  $MODEL"
echo "  Endpoint   :  http://172.22.225.5:8013/v1  (Qwen/Qwen3.5-27B)"
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
