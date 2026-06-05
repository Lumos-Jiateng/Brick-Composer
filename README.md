# Brick-Composer: MLLMs Construct Everything from Building Blocks

**[Project Page](https://lumos-jiateng.github.io/Brick-Composer/)** | **[Paper](docs/assets/Brick-Composer-paper.pdf)**

Official repository for **"Brick-Composer: MLLMs Construct Everything from Building Blocks"**

Jiateng Liu, Bingxuan Li, Zhenhailong Wang, Rushi Wang, Kaiwen Hong, Cheng Qian, Jiayu Liu, Denghui Zhang, Katherine Driggs-Campbell, Manling Li, Heng Ji

UIUC · Stevens Institute of Technology · Northwestern University

---

## About

We study whether multimodal large language models (MLLMs) can read arbitrary designs and construct real-world objects from reusable building blocks. We formulate brick assembly as a **sequential decision-making** problem with two subtasks:

- **Brick Selection** — identify the target component from a candidate catalog grid.
- **Brick Pose Estimation** — predict where (position) and how (6-DoF orientation) the selected brick should be placed.

We introduce **BC-Bench**, a benchmark for evaluating MLLMs on assembly with diverse bricks, and propose **Brick-Composer**, a learning framework combining **Human Design Sparks**, **World Feedback**, and **Synthetic Experience**.

## Key Results

| Approach | Selection ↑ | Trans. Err ↓ | Rot. Err ↓ | Step-Wise SR ↑ |
|---|---|---|---|---|
| Qwen-3-VL-8B (Direct) | 22.76 | 210.14 | 62.47 | 0.36 |
| Brick-Composer | **68.21** | **65.63** | **37.97** | **14.27** |

- **3×** improvement in brick selection accuracy
- Strict step-level assembly success from <1% → **~15%**

## Datasets

All datasets are hosted on the Hugging Face Hub under [`Lumos-Jiateng`](https://huggingface.co/Lumos-Jiateng):

| Dataset | Hub repo | What it contains |
|---|---|---|
| **Brick / Design data** | [`Lumos-Jiateng/bricklink_lego_design`](https://huggingface.co/datasets/Lumos-Jiateng/bricklink_lego_design) | BrickLink part library and per-object LEGO designs (the brick vocabulary and source designs). |
| **Designer Supervision** | [`Lumos-Jiateng/designer_supervision`](https://huggingface.co/datasets/Lumos-Jiateng/designer_supervision) | Step-by-step supervision (renders + selection/pose ground truth) for 102 real, human-designed objects — the *Human Design Sparks* signal. |
| **Synthetic Experience** | [`Lumos-Jiateng/brick_synthetic`](https://huggingface.co/datasets/Lumos-Jiateng/brick_synthetic) | Large-scale synthetic assembly trajectories for the *Synthetic Experience* stage. |

Quick download example:

```bash
# e.g. the Designer Supervision archive
hf download Lumos-Jiateng/designer_supervision Designer_supervision.zip \
  --repo-type dataset --local-dir .
unzip Designer_supervision.zip
```

```python
from huggingface_hub import snapshot_download
snapshot_download(repo_id="Lumos-Jiateng/brick_synthetic", repo_type="dataset")
```

## Repository layout

```
Brick-Composer/
├── README.md
├── docs/                                   # project website (GitHub Pages source)
└── reasoning/                              # evaluation code for the two subtasks
    ├── MLLM_Brick_Selection/               # Brick Selection eval pipeline
    │   ├── prompt.py                        # system prompt + user-message builder
    │   ├── api_client.py                    # vLLM endpoint config + call_model()
    │   ├── run.py                           # main selection pipeline
    │   ├── evaluate.py                       # selection metrics
    │   ├── evaluate_with_color.py           # color-aware selection metrics
    │   └── eval_test_split.py
    └── MLLM_Brick_Pose_Estimation/         # Brick Pose Estimation eval pipeline
        ├── api_client.py                    # vLLM endpoint config + call_model()
        ├── evaluate_assembly.py             # main inference + evaluation
        ├── pose_metrics.py                  # translation + symmetry-aware rotation error
        ├── compute_success_rate.py          # step-wise success rate
        ├── reeval_with_symmetry.py
        ├── aggregate_by_object.py
        └── run_*.sh                         # per-model run scripts
```

See [`reasoning/MLLM_Brick_Selection/README.md`](reasoning/MLLM_Brick_Selection/README.md) and
[`reasoning/MLLM_Brick_Pose_Estimation/README.md`](reasoning/MLLM_Brick_Pose_Estimation/README.md)
for per-task instructions.

## Getting started

The evaluation code talks to a local [vLLM](https://github.com/vllm-project/vllm)
OpenAI-compatible server. The minimal client dependencies are:

```bash
pip install openai pillow
```

1. Serve a Vision-Language Model with vLLM (e.g. Qwen-3-VL) and note its host/port.
2. Set the endpoint in the relevant `api_client.py`.
3. Download the datasets above and point the scripts at the extracted folders.
4. Run a subtask, e.g.:

```bash
# Brick Selection
cd reasoning/MLLM_Brick_Selection && python run.py

# Brick Pose Estimation
cd reasoning/MLLM_Brick_Pose_Estimation && bash run_real_synthetic.sh
```

## Citation

```bibtex
@misc{liu2026brickcomposer,
  title  = {Brick-Composer: MLLMs Construct Everything from Building Blocks},
  author = {Liu, Jiateng and Li, Bingxuan and Wang, Zhenhailong and Wang, Rushi and Hong, Kaiwen and Qian, Cheng and Liu, Jiayu and Zhang, Denghui and Driggs-Campbell, Katherine and Li, Manling and Ji, Heng},
  year   = {2026},
  url    = {https://github.com/Lumos-Jiateng/Brick-Composer}
}
```
