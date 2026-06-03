# Brick-Composer: MLLMs Construct Everything from Building Blocks

**[Project Page](https://lumos-jiateng.github.io/Brick-Composer/)** | **[Paper](docs/assets/Brick-Composer-paper.pdf)**

Official repository for **"Brick-Composer: MLLMs Construct Everything from Building Blocks"**

Jiateng Liu, Bingxuan Li, Zhenhailong Wang, Rushi Wang, Kaiwen Hong, Cheng Qian, Jiayu Liu, Denghui Zhang, Katherine Driggs-Campbell, Manling Li, Heng Ji

UIUC · Stevens Institute of Technology · Northwestern University

---

## About

We study whether multimodal large language models (MLLMs) can read arbitrary designs and construct real-world objects from reusable building blocks. We formulate brick assembly as a sequential decision-making problem with two subtasks:

- **Brick Selection**: identifying the target component from a candidate grid
- **Brick Pose Estimation**: predicting where and how the selected brick should be placed

We introduce **BC-Bench**, a benchmark for evaluating MLLMs on assembly with diverse bricks, and propose **Brick-Composer**, a learning framework combining Human Design Sparks, World Feedback, and Synthetic Experience.

## Key Results

| Approach | Selection ↑ | Trans. Err ↓ | Rot. Err ↓ | Step-Wise SR ↑ |
|---|---|---|---|---|
| Qwen-3-VL-8B (Direct) | 22.76 | 210.14 | 62.47 | 0.36 |
| Brick-Composer | **68.21** | **65.63** | **37.97** | **14.27** |

- **3×** improvement in brick selection accuracy
- Strict step-level assembly success from <1% → **~15%**

## Citation

```bibtex
@misc{liu2026brickcomposer,
  title  = {Brick-Composer: MLLMs Construct Everything from Building Blocks},
  author = {Liu, Jiateng and Li, Bingxuan and Wang, Zhenhailong and Wang, Rushi and Hong, Kaiwen and Qian, Cheng and Liu, Jiayu and Zhang, Denghui and Driggs-Campbell, Katherine and Li, Manling and Ji, Heng},
  year   = {2026},
  url    = {https://github.com/Lumos-Jiateng/Brick-Composer}
}
```
