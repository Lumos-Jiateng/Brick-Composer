#!/usr/bin/env python3
"""
evaluate_with_color.py
----------------------
Evaluates a VLM on LEGO brick selection using the **tasks_with_color** task
root.  Each step directory already contains pre-rendered composite images
(``original_orthogonal_views.png``, ``target_orthogonal_views.png``) and
catalog images, so no view concatenation is needed.

Results are written under **results_color** / **results_color_module**.

Usage
-----
  python evaluate_with_color.py --test_set ../tasks_with_color/test.json
  python evaluate_with_color.py --obj 1704
  python evaluate_with_color.py --obj 1704 --step 0
  python evaluate_with_color.py --no_caption
  python evaluate_with_color.py --output results_color/my_run.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from api_client import call_model, img_to_b64
from evaluate import aggregate_metrics, compute_metrics, print_summary
from prompt import SYSTEM_PROMPT, build_user_message

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

TASKS_DIR = Path(
    "/shared/nas/data/m1/jiateng5/BrickComposer/reasoning/tasks_with_color"
)

_BASE = Path(__file__).parent.resolve()
# Defaults — overridden at runtime when --results_dir is passed
RESULTS_DIR = _BASE / "results_color"
RESULTS_MODULE_DIR = _BASE / "results_color_module"


# ---------------------------------------------------------------------------
# Task loading
# ---------------------------------------------------------------------------

def load_tasks(
    obj_filter: str | None,
    step_filter: int | None,
    test_set: set[str] | None = None,
) -> list[tuple[str, dict]]:
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
            if test_set is not None:
                key = f"{obj_dir.name}/step_{task['step_k']:04d}_{task['view_folder']}"
                if key not in test_set:
                    continue
            tasks.append((obj_dir.name, task))

    return tasks


# ---------------------------------------------------------------------------
# Single task execution
# ---------------------------------------------------------------------------

def run_task(
    task: dict,
    use_no_caption: bool = False,
    enable_thinking: bool = False,
) -> dict:
    if task.get("is_blank"):
        return {
            "step_k":       task["step_k"],
            "view_folder":  task["view_folder"],
            "ground_truth": task.get("ground_truth", ""),
            "gt_bricks":    task.get("gt_bricks", []),
            "skipped":      True,
            "reason":       "blank step",
        }

    step_dir = Path(task["task_dir"])
    is_module = task.get("ground_truth", "").startswith('module "')

    catalog_key = (
        "catalog_remaining_no_caption" if use_no_caption else "catalog_remaining"
    )
    catalog_path = Path(task["input_images"][catalog_key])

    original_b64 = img_to_b64(step_dir / "original_orthogonal_views.png")
    target_b64   = img_to_b64(step_dir / "target_orthogonal_views.png")
    catalog_b64  = img_to_b64(catalog_path)

    user_content = build_user_message(
        original_b64, target_b64, catalog_b64, is_module=is_module,
    )

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
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a VLM on LEGO brick selection using **tasks_with_color** "
            "and **results_color**."
        ),
    )
    parser.add_argument(
        "--obj", type=str, default=None,
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
        help="Path to JSON summary (default: results_color/summary_<timestamp>.json).",
    )
    parser.add_argument(
        "--test_set", type=str, default=None,
        help=(
            "Path to a JSON file listing the tasks to evaluate "
            "(e.g. tasks_with_color/test.json).  Each entry must be "
            "'<obj_name>/step_<k:04d>_<view_folder>'."
        ),
    )
    parser.add_argument(
        "--results_dir", type=str, default=None,
        help=(
            "Override the base results directory "
            "(default: results_color/ next to this script).  "
            "Module results go into <results_dir>_module/."
        ),
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    global RESULTS_DIR, RESULTS_MODULE_DIR

    args = parse_args()

    if args.results_dir:
        RESULTS_DIR = Path(args.results_dir)
        RESULTS_MODULE_DIR = Path(args.results_dir + "_module")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_MODULE_DIR.mkdir(parents=True, exist_ok=True)

    test_set: set[str] | None = None
    if args.test_set:
        test_set = set(json.loads(Path(args.test_set).read_text()))

    tasks = load_tasks(args.obj, args.step, test_set)
    if not tasks:
        print("No tasks found — check TASKS_DIR and --obj / --step / --test_set.")
        return

    print(
        f"Loaded {len(tasks)} tasks  "
        f"(obj={args.obj or 'all'}, step={args.step if args.step is not None else 'all'})",
    )
    print(f"Task root: {TASKS_DIR}")

    results_by_obj: dict[str, list[dict]] = {}
    all_results: list[dict] = []
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    for obj_name, task in tasks:
        gt_label = task.get("ground_truth", "blank")
        print(
            f"  [{obj_name}] step {task['step_k']:04d}  gt={gt_label!r:<30}",
            end="",
            flush=True,
        )

        obj_id = obj_name.split("_")[0]
        step_k = task["step_k"]
        view_folder = task["view_folder"]
        folder_name = f"{obj_id}_step_{step_k:04d}_{view_folder}"

        is_module_task = task.get("ground_truth", "").startswith('module "')
        root_dir = RESULTS_MODULE_DIR if is_module_task else RESULTS_DIR
        task_dir = root_dir / folder_name
        task_dir.mkdir(parents=True, exist_ok=True)

        result_json_path = task_dir / "result.json"
        if result_json_path.exists():
            print("SKIP (already done)")
            result = json.loads(result_json_path.read_text())
            result["object"] = obj_name
            all_results.append(result)
            results_by_obj.setdefault(obj_name, []).append(result)
            continue

        result = run_task(task, args.no_caption, args.thinking)
        result["object"] = obj_name
        all_results.append(result)
        results_by_obj.setdefault(obj_name, []).append(result)

        if result.get("skipped"):
            print("SKIP")
        elif result.get("is_module"):
            ec = "✓" if result.get("either_correct") else "✗"
            bm = "✓" if result.get("brick_exact_match") else "✗"
            pm = "✓" if result.get("position_any_match") else "✗"
            f1 = result.get("f1", 0.0)
            cnt = result.get("catalog_brick_count", "?")
            print(
                f"[module] either={ec} brick={bm} pos={pm} f1={f1:.3f} "
                f"catalog={cnt}  pred={result['predicted_bricks']}",
            )
        else:
            ec = "✓" if result.get("either_correct") else "✗"
            bc = "✓" if result.get("brick_correct") else "✗"
            pc = "✓" if result.get("position_correct") else "✗"
            cnt = result.get("catalog_brick_count", "?")
            print(
                f"[single] either={ec} brick={bc} pos={pc} "
                f"catalog={cnt}  pred={result['predicted_bricks']}",
            )

        result_json_path.write_text(
            json.dumps(result, indent=2, ensure_ascii=False),
        )

    agg = aggregate_metrics(all_results)
    print_summary(agg)

    run_config = {
        "obj_filter":         args.obj,
        "step_filter":        args.step,
        "use_no_caption":     args.no_caption,
        "enable_thinking":    args.thinking,
        "test_set":           args.test_set,
        "timestamp":          timestamp,
        "tasks_dir":          str(TASKS_DIR),
        "results_dir":        str(RESULTS_DIR),
        "results_module_dir": str(RESULTS_MODULE_DIR),
    }
    if args.output:
        summary_path = Path(args.output)
    else:
        tag = f"_{args.obj}" if args.obj else ""
        summary_path = RESULTS_DIR / f"summary{tag}_{timestamp}.json"

    summary_payload = {
        "run_config": run_config,
        "summary":    agg,
        "by_object":  {
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
