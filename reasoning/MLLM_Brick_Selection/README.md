# MLLM Brick Selection Evaluation

Evaluates a Vision-Language Model (Qwen3-VL via vLLM) on the LEGO brick assembly
step-selection task.

## File structure

| File | Purpose |
|---|---|
| `prompt.py` | **Edit this** to change what the model sees — system prompt & user message builder |
| `api_client.py` | vLLM server config + `call_model()` — edit host/port/model name here |
| `evaluate.py` | `.dat` filename extraction, per-task metrics, aggregate reporting |
| `run.py` | Main pipeline: loads tasks, calls model, writes results |
| `results/` | JSON result files written here |

## Prerequisites

```bash
conda activate base   # openai, Pillow must be installed
```

## Usage

```bash
# Full evaluation (all 53 objects, ~2500 non-blank steps)
conda run -n base python run.py

# Single object
conda run -n base python run.py --obj 1704

# Single object + single step (quick smoke-test)
conda run -n base python run.py --obj 1704 --step 0

# Use catalog without captions
conda run -n base python run.py --obj 1704 --no_caption

# Enable Qwen3 chain-of-thought thinking mode
conda run -n base python run.py --obj 1704 --thinking

# Custom output path
conda run -n base python run.py --output results/my_run.json
```

## Metrics

| Metric | Description |
|---|---|
| `exact_match_acc` | % of steps where predicted multiset == GT multiset |
| `mean_recall` | Average fraction of GT bricks found in prediction |
| `mean_precision` | Average fraction of predicted bricks that are correct |
| `mean_f1` | Harmonic mean of recall & precision |
| `single_*` | Above metrics restricted to single-brick steps |
| `module_*` | Above metrics restricted to module (multi-brick) steps |

Results are written to `results/eval_<timestamp>.json`.

## Modifying the prompt

Open `prompt.py` and edit:
- `SYSTEM_PROMPT` — the system-turn text
- `build_user_message()` — the order, wording, and images sent in the user turn

No other file needs to change.
