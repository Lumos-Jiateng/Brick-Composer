#!/Users/jiateng5/anaconda3/bin/python
"""
run.py
------
Main VLM evaluation pipeline for LEGO brick selection.

Usage
-----
# Evaluate all objects, all non-blank steps:
  python run.py

# Single object:
  python run.py --obj 1704

# Single object + single step:
  python run.py --obj 1704 --step 0

# Use no-caption catalog image:
  python run.py --no_caption

# Enable Qwen3 thinking mode:
  python run.py --thinking

# Custom output path:
  python run.py --output results/my_run.json
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import math
import shutil
import time
from pathlib import Path

from PIL import Image

from api_client import call_model, img_to_b64
from evaluate import aggregate_metrics, compute_metrics, print_summary
from prompt import SYSTEM_PROMPT, build_user_message

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

TASKS_DIR          = Path("/Users/jiateng5/research/BrickLink_data_statistics/reasoning/tasks")
RESULTS_DIR        = Path(__file__).parent / "results"
RESULTS_MODULE_DIR = Path(__file__).parent / "results_module"

# View grid layout: 6 views arranged as 3 columns × 2 rows
VIEW_GRID_COLS   = 3
VIEW_THUMB_WIDTH = 480   # resize each view to this width before grid assembly


# ---------------------------------------------------------------------------
# Image utilities
# ---------------------------------------------------------------------------

def _resize_keep_aspect(img: Image.Image, target_w: int) -> Image.Image:
    ratio    = target_w / img.width
    target_h = int(img.height * ratio)
    return img.resize((target_w, target_h), Image.LANCZOS)


def concat_views_to_image(image_paths: list[str],
                           cols: int = VIEW_GRID_COLS) -> Image.Image:
    """
    Load *image_paths*, resize to VIEW_THUMB_WIDTH, assemble into a grid,
    and return the PIL Image (useful for both saving and encoding).
    """
    images = [_resize_keep_aspect(Image.open(p).convert("RGB"), VIEW_THUMB_WIDTH)
              for p in image_paths]
    w, h = images[0].size
    rows = math.ceil(len(images) / cols)
    grid = Image.new("RGB", (cols * w, rows * h), color=(255, 255, 255))
    for i, img in enumerate(images):
        grid.paste(img, (i % cols * w, i // cols * h))
    return grid


def image_to_b64(img: Image.Image) -> str:
    """Encode a PIL Image to a base64 PNG string."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


# ---------------------------------------------------------------------------
# Task loading
# ---------------------------------------------------------------------------

def load_tasks(
    obj_filter:  str | None,
    step_filter: int | None,
) -> list[tuple[str, dict]]:
    """
    Return [(object_folder_name, task_dict), ...] from all tasks.json files,
    applying optional object-ID and step-index filters.
    """
    tasks: list[tuple[str, dict]] = []

    for obj_dir in sorted(TASKS_DIR.iterdir()):
        if not obj_dir.is_dir():
            continue

        if obj_filter is not None:
            obj_id = obj_dir.name.split("_")[0]
            if obj_id != obj_filter:
                continue

        manifest_path = obj_dir / "tasks.json"
        if not manifest_path.exists():
            continue

        manifest = json.loads(manifest_path.read_text())
        for task in manifest["tasks"]:
            if step_filter is not None and task["step_k"] != step_filter:
                continue
            tasks.append((obj_dir.name, task))

    return tasks


# ---------------------------------------------------------------------------
# Single-task runner
# ---------------------------------------------------------------------------

VIEW_ORDER = ["Back", "Bottom", "Left", "Right", "Up", "front"]


def run_task(
    task:            dict,
    use_no_caption:  bool        = False,
    enable_thinking: bool        = False,
    save_dir:        Path | None = None,
    step_dir:        Path | None = None,
) -> dict:
    """
    Run one task through the VLM and return a result dict.

    If *step_dir* is given, original/target view images are loaded from the
    center-cropped files in the tasks/ folder (original_<view>.png /
    target_<view>.png) instead of the raw step_wise_orthogonal_views paths
    stored in tasks.json.

    If *save_dir* is given, the three input images are written there:
        original_views.png  – 3×2 grid of original views
        target_views.png    – 3×2 grid of target views
        catalog.png         – catalog image used for this call

    Blank steps are returned immediately as skipped (no API call made).
    """
    if task.get("is_blank"):
        return {
            "step_k":       task["step_k"],
            "view_folder":  task["view_folder"],
            "ground_truth": task.get("ground_truth", ""),
            "gt_bricks":    task.get("gt_bricks", []),
            "skipped":      True,
            "reason":       "blank step",
        }

    imgs      = task["input_images"]
    is_module = task.get("ground_truth", "").startswith('module "')

    # ── Build 3×2 view grids as PIL Images ───────────────────────────────
    if step_dir is not None:
        # Use center-cropped images from the tasks/ directory
        original_paths = [str(step_dir / f"original_{v}.png") for v in VIEW_ORDER]
        target_paths   = [str(step_dir / f"target_{v}.png")   for v in VIEW_ORDER]
    else:
        original_paths = imgs["original"]
        target_paths   = imgs["target"]

    original_img = concat_views_to_image(original_paths)
    target_img   = concat_views_to_image(target_paths)

    # ── Catalog (load as PIL so we can optionally save a copy) ────────────
    catalog_key  = "catalog_remaining_no_caption" if use_no_caption else "catalog_remaining"
    catalog_path = Path(imgs[catalog_key])
    catalog_img  = Image.open(catalog_path).convert("RGB")

    # ── Optionally save input images into the task subfolder ─────────────
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        original_img.save(save_dir / "original_views.png")
        target_img.save(save_dir   / "target_views.png")
        shutil.copy(catalog_path,    save_dir / "catalog.png")

    # ── Encode to base64 for the API ─────────────────────────────────────
    original_b64 = image_to_b64(original_img)
    target_b64   = image_to_b64(target_img)
    catalog_b64  = image_to_b64(catalog_img)

    user_content = build_user_message(original_b64, target_b64, catalog_b64,
                                      is_module=is_module)

    t0 = time.time()
    raw_output = call_model(
        SYSTEM_PROMPT,
        user_content,
        enable_thinking=enable_thinking,
    )
    elapsed = round(time.time() - t0, 2)

    metrics = compute_metrics(
        raw_output,
        task["gt_bricks"],
        task["ground_truth"],
        gt_position=task.get("gt_position"),
    )

    return {
        "step_k":              task["step_k"],
        "view_folder":         task["view_folder"],
        "ground_truth":        task["ground_truth"],
        "gt_bricks":           task["gt_bricks"],
        "gt_position":         task.get("gt_position"),
        "catalog_brick_count": task.get("catalog_brick_count"),
        "raw_output":          raw_output,
        "elapsed_s":           elapsed,
        **metrics,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a VLM on the LEGO brick selection task."
    )
    parser.add_argument(
        "--obj",  type=str, default=None,
        help="Only evaluate this object ID (e.g. 1704).",
    )
    parser.add_argument(
        "--step", type=int, default=None,
        help="Only evaluate this step index, 0-based (e.g. 0).",
    )
    parser.add_argument(
        "--no_caption", action="store_true",
        help="Use catalog_remaining_no_caption.png instead of the labeled version.",
    )
    parser.add_argument(
        "--thinking", action="store_true",
        help="Enable Qwen3 chain-of-thought thinking mode.",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Path to write the JSON results (default: results/eval_<timestamp>.json).",
    )
    parser.add_argument(
        "--results_dir", type=str, default=None,
        help="Override the output directory for per-step result.json files "
             "(default: results/ and results_module/ next to this script).",
    )
    return parser.parse_args()


def main() -> None:
    global RESULTS_DIR, RESULTS_MODULE_DIR
    args = parse_args()
    if args.results_dir:
        RESULTS_DIR        = Path(args.results_dir)
        RESULTS_MODULE_DIR = Path(args.results_dir + "_module")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_MODULE_DIR.mkdir(parents=True, exist_ok=True)

    tasks = load_tasks(args.obj, args.step)
    if not tasks:
        print("No tasks found — check --obj / --step filters.")
        return

    print(f"Loaded {len(tasks)} tasks  "
          f"(obj={args.obj or 'all'}, step={args.step if args.step is not None else 'all'})")

    results_by_obj: dict[str, list[dict]] = {}
    all_results:    list[dict]            = []
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    for obj_name, task in tasks:
        gt_label = task.get("ground_truth", "blank")
        print(f"  [{obj_name}] step {task['step_k']:04d}  gt={gt_label!r:<30}", end="", flush=True)

        obj_id      = obj_name.split("_")[0]
        step_k      = task["step_k"]
        view_folder = task["view_folder"]
        folder_name = f"{obj_id}_step_{step_k:04d}_{view_folder}"

        # Determine output root before running (is_module known from ground_truth)
        is_module_task = task.get("ground_truth", "").startswith('module "')
        root_dir  = RESULTS_MODULE_DIR if is_module_task else RESULTS_DIR
        task_dir  = root_dir / folder_name
        task_dir.mkdir(parents=True, exist_ok=True)

        # ── Resume: skip tasks that already have a saved result ───────────
        result_json_path = task_dir / "result.json"
        if result_json_path.exists():
            print("SKIP (already done)")
            result = json.loads(result_json_path.read_text())
            result["object"] = obj_name
            all_results.append(result)
            results_by_obj.setdefault(obj_name, []).append(result)
            continue

        step_dir = TASKS_DIR / obj_name / f"step_{step_k:04d}_{view_folder}"
        result   = run_task(task, args.no_caption, args.thinking,
                            save_dir=task_dir, step_dir=step_dir)
        result["object"] = obj_name
        all_results.append(result)
        results_by_obj.setdefault(obj_name, []).append(result)

        if result.get("skipped"):
            print("SKIP")
        elif result.get("is_module"):
            ec  = "✓" if result.get("either_correct")    else "✗"
            bm  = "✓" if result.get("brick_exact_match") else "✗"
            pm  = "✓" if result.get("position_any_match") else "✗"
            f1  = result.get("f1", 0.0)
            cnt = result.get("catalog_brick_count", "?")
            print(f"[module] either={ec} brick={bm} pos={pm} f1={f1:.3f} "
                  f"catalog={cnt}  pred={result['predicted_bricks']}")
        else:
            ec  = "✓" if result.get("either_correct")   else "✗"
            bc  = "✓" if result.get("brick_correct")    else "✗"
            pc  = "✓" if result.get("position_correct") else "✗"
            cnt = result.get("catalog_brick_count", "?")
            print(f"[single] either={ec} brick={bc} pos={pc} "
                  f"catalog={cnt}  pred={result['predicted_bricks']}")

        # ── Per-task result JSON (inside the subfolder) ────────────────────
        result_json = task_dir / "result.json"
        result_json.write_text(json.dumps(result, indent=2, ensure_ascii=False))

    # ── Aggregate and display ──────────────────────────────────────────────
    agg = aggregate_metrics(all_results)
    print_summary(agg)

    # ── Summary file ──────────────────────────────────────────────────────
    run_config = {
        "obj_filter":      args.obj,
        "step_filter":     args.step,
        "use_no_caption":  args.no_caption,
        "enable_thinking": args.thinking,
        "timestamp":       timestamp,
    }
    if args.output:
        summary_path = Path(args.output)
    else:
        tag = f"_{args.obj}" if args.obj else ""
        summary_path = RESULTS_DIR / f"summary{tag}_{timestamp}.json"

    summary_payload = {
        "run_config":  run_config,
        "summary":     agg,
        "by_object":   {
            obj: aggregate_metrics(results)
            for obj, results in results_by_obj.items()
        },
    }
    summary_path.write_text(json.dumps(summary_payload, indent=2, ensure_ascii=False))
    print(f"\nSingle-brick results → {RESULTS_DIR}/<obj>_step_<k>_<view>/")
    print(f"Module results       → {RESULTS_MODULE_DIR}/<obj>_step_<k>_<view>/")
    print(f"Summary              → {summary_path}")


if __name__ == "__main__":
    main()
