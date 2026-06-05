# MLLM Brick Pose Estimation

Evaluates Vision-Language Models on the **Brick Pose Estimation** subtask: given the
build state before/after a step plus the brick render(s), predict the global 6-DoF pose
(translation + rotation) of the brick(s) placed at that step.

Inference is served by a local [vLLM](https://github.com/vllm-project/vllm) OpenAI-compatible
endpoint (host/port configured in `api_client.py`).

## File structure

| File | Purpose |
|---|---|
| `api_client.py` | vLLM endpoint config + `call_model()` (edit host/port/model name here) |
| `evaluate_assembly.py` | Main inference + evaluation over the assembly test set |
| `pose_metrics.py` | Translation error (LDU) and symmetry-aware geodesic rotation error |
| `compute_success_rate.py` | Step-wise success rate under translation/rotation thresholds |
| `reeval_with_symmetry.py` | Symmetry-aware re-aggregation of saved results |
| `aggregate_by_object.py` | Aggregates per-step metrics up to per-object summaries |
| `run_*.sh` | End-to-end run scripts for each model variant |

## Run scripts

| Script | Model variant |
|---|---|
| `run_qwen3_8b.sh` | Qwen-3-VL-8B (direct) |
| `run_qwen35_27b.sh` | Qwen-3.5-VL-27B (direct) |
| `run_real_only.sh` | Fine-tuned on Designer Supervision only |
| `run_real_synthetic.sh` | Fine-tuned on Designer Supervision + Synthetic Experience |

Each script runs inference, then success-rate evaluation (default `t=10` LDU, `r=45°`),
then symmetry-aware re-aggregation.

## Usage

```bash
# Full pipeline for the real+synthetic fine-tuned model
bash run_real_synthetic.sh

# Chain-of-thought ("thinking") mode
bash run_real_synthetic.sh --thinking

# Single object
bash run_real_synthetic.sh --obj 77756

# Or call the evaluator directly
python evaluate_assembly.py --model real_only --limit 50
```

## Metrics

| Metric | Description |
|---|---|
| `translation_error_ldu` | Euclidean `\|t_pred - t_gt\|` in LDraw units |
| `geodesic_error_deg` | Symmetry-aware geodesic rotation angle: `min_{g∈G} geodesic(R_pred, R_gt·g)` |
| step-wise success rate | Fraction of steps with translation < `t` LDU **and** rotation < `r`° |

Results are written under `results_pose_<model>/` (not tracked in this repo).

## Data

This task consumes the **Designer Supervision** dataset (per-step ground-truth poses in
`brick_pose.json`). See the [dataset links in the main README](../../README.md#datasets).
