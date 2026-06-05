#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# run_real_only.sh
# Evaluate the real_only fine-tuned model (qwen3 @ port 8031) on the full
# Designer_supervision_assembly_testing.json dataset.
#
# Results → results_pose_real_only/<task_id>/result.json
# Summary → results_pose_real_only/summary_<timestamp>.json
#
# Inference is resumable: already-completed tasks are skipped automatically.
# After inference, runs success-rate evaluation (t=10 LDU, r=45°) and
# symmetry-aware re-aggregation.
#
# Usage:
#   bash run_real_only.sh
#   bash run_real_only.sh --thinking        # enable chain-of-thought
#   bash run_real_only.sh --obj 77756       # single model
# ---------------------------------------------------------------------------

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON="${PYTHON:-/shared/nas/data/m1/jiateng5/anaconda3/envs/vllm/bin/python3.12}"
MODEL="real_only"
RESULTS_DIR="results_pose_real_only"

echo "============================================================"
echo "  Inference  :  $MODEL"
echo "  Endpoint   :  http://172.22.225.5:8031/v1  (qwen3)"
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
