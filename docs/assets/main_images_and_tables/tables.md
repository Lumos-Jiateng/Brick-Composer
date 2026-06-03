# Brick-Composer Main Images and Tables

## Table 2: Main evaluation of state-of-the-art MLLMs on BC-Bench

| Model | Overall Selection Acc. (%) ↑ | Overall PE Trans. Err. (LDU) ↓ | Overall PE Rot. Err. (°) ↓ | Overall Step-Wise SR (%) ↑ | Best Object Selection Acc. (%) ↑ | Best Object PE Trans. Err. (LDU) ↓ | Best Object PE Rot. Err. (°) ↓ | Best Object Step-Wise SR (%) ↑ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Gemma-3-12B | 4.35 | 269.09 | 63.00 | 0.09 | 17.24 | 41.86 | 12.86 | 2.22 |
| InternVL-3.5-8B | 13.24 | 221.69 | 88.43 | 0.00 | 31.25 | 41.54 | 35.00 | 0.00 |
| Qwen-3-VL-8B | 22.76 | 210.14 | 62.47 | 0.36 | 55.17 | 43.81 | 12.86 | 4.44 |
| Qwen-3.5-VL-27B | 37.44 | 314.32 | 82.94 | 0.18 | 75.86 | 64.47 | 34.84 | 4.44 |
| GPT-5.4 | 43.88 | 310.78 | 74.85 | 0.18 | 93.10 | 67.94 | 40.00 | 2.22 |

## Table 3: Performance improvements on BC-Bench

| Model Group | Approach | Overall Selection Acc. (%) ↑ | Overall PE Trans. Err. (LDU) ↓ | Overall PE Rot. Err. (°) ↓ | Overall Step-Wise SR (%) ↑ | Best Object Selection Acc. (%) ↑ | Best Object PE Trans. Err. (LDU) ↓ | Best Object PE Rot. Err. (°) ↓ | Best Object Step-Wise SR (%) ↑ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Gemma-3-12B | Direct Prompting | 4.35 | 269.09 | 63.00 | 0.09 | 17.24 | 41.86 | 12.86 | 2.22 |
| Gemma-3-12B | World Feedback (P) | – | 273.41 | 62.26 | 0.00 | – | 39.46 | 14.53 | 2.22 |
| Gemma-3-12B | Designer Supervision | 15.59 | 201.51 | 55.69 | 0.72 | 26.67 | 36.91 | 15.00 | 4.44 |
| Gemma-3-12B | World Feedback (L) | – | 170.33 | 51.49 | 1.99 | – | 27.56 | 12.86 | 9.07 |
| Gemma-3-12B | Brick-Composer | 17.95 | 123.69 | 45.43 | 4.35 | 52.87 | 24.63 | 12.86 | 18.75 |
| Qwen-3-8B-VL | Direct Prompting | 22.76 | 210.14 | 62.47 | 0.36 | 55.17 | 43.81 | 12.86 | 4.44 |
| Qwen-3-8B-VL | World Feedback (P) | – | 226.33 | 65.66 | 0.27 | – | 42.36 | 12.86 | 4.44 |
| Qwen-3-8B-VL | Designer Supervision | 48.29 | 162.82 | 57.81 | 5.40 | 73.24 | 27.13 | 12.95 | 8.92 |
| Qwen-3-8B-VL | World Feedback (L) | – | 137.26 | 52.65 | 6.24 | – | 21.46 | 11.64 | 15.36 |
| Qwen-3-8B-VL | Brick-Composer | 68.21 | 65.63 | 37.97 | 14.27 | 90.65 | 14.29 | 0.00 | 41.63 |
