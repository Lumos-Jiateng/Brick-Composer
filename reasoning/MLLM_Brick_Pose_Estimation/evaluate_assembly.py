#!/usr/bin/env python3
"""
evaluate_assembly.py
--------------------
Evaluates fine-tuned VLMs on LEGO brick global pose estimation using the
Designer_supervision_assembly_testing.json evaluation set.

Each entry in the testing JSON provides the exact prompt format the model was
fine-tuned on (system + human turn with <image> placeholders).  The four images
are supplied in order:
  1. original_orthogonal_views.png  — assembly BEFORE this step
  2. target_orthogonal_views.png    — assembly AFTER this step
  3. part render(s)                 — annotated render per brick
  4. current_state_axes.png         — 7-view composite with LDU tick labels

Ground truth is read from brick_pose.json in:
  Designer_supervision/data/<model_folder>/<step>/brick_pose.json

Evaluation metrics (per brick, symmetry-aware at inference time):
  translation_error_ldu   — Euclidean |t_pred - t_gt| in LDU
  geodesic_error_deg      — Symmetry-aware geodesic angle (degrees):
                            min_{g in G}  geodesic(R_pred,  R_gt @ g)

Results are written to:
  results_pose_<model>/<model_folder>_step_<k>/result.json
  results_pose_<model>/summary_<timestamp>.json

Usage
-----
  python evaluate_assembly.py --model real_only
  python evaluate_assembly.py --model real_synthetic
  python evaluate_assembly.py --model real_only --obj 77756
  python evaluate_assembly.py --model real_only --thinking
  python evaluate_assembly.py --model real_only --limit 50
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from pathlib import Path
from typing import Any

from api_client import call_model, img_to_b64
from pose_metrics import (
    aggregate_pose_metrics,
    aggregate_step_metrics,
    compute_translation_error,
    compute_translation_offsets,
    parse_pose_output,
    print_summary,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

TESTING_JSON = Path(
    "/shared/nas/data/m1/jiateng5/BrickComposer"
    "/LlamaFactory/data/Designer_supervision_assembly_testing.json"
)
GT_DATA_DIR = Path(
    "/shared/nas/data/m1/jiateng5/BrickComposer/Designer_supervision/data"
)
SYMMETRY_DIR = Path(
    "/shared/nas/data/m1/jiateng5/BrickComposer"
    "/simulator_visualization/part_simulation/renders_all_text_qwen"
)

_BASE = Path(__file__).parent.resolve()

# ---------------------------------------------------------------------------
# Model endpoint configuration
# ---------------------------------------------------------------------------

MODEL_CONFIGS: dict[str, dict] = {
    "real_only": {
        "base_url":    "http://172.22.225.5:8031/v1",
        "model_name":  "qwen3",
        "results_dir": "results_pose_real_only",
    },
    "real_synthetic": {
        "base_url":    "http://172.22.225.5:8032/v1",
        "model_name":  "qwen3-vl-8b",
        "results_dir": "results_pose_real_synthetic",
    },
    # Base / reference models (not fine-tuned on assembly data)
    "qwen35_27b": {
        "base_url":    "http://172.22.225.5:8013/v1",
        "model_name":  "Qwen/Qwen3.5-27B",
        "results_dir": "results_pose_qwen35_27b",
    },
    "qwen3_8b": {
        "base_url":    "http://172.22.225.5:8001/v1",
        "model_name":  "qwen-3-vl",
        "results_dir": "results_pose_qwen3_8b",
    },
}


# ---------------------------------------------------------------------------
# Pure-Python symmetry helpers  (identical to reeval_with_symmetry.py)
# ---------------------------------------------------------------------------

def _mat(rows: list) -> list:
    return [list(r) for r in rows]


def _transpose(m: list) -> list:
    return [[m[r][c] for r in range(3)] for c in range(3)]


def _matmul(a: list, b: list) -> list:
    return [
        [sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)]
        for i in range(3)
    ]


def _mats_approx_equal(a: list, b: list, tol: float = 1e-5) -> bool:
    return all(abs(a[i][j] - b[i][j]) <= tol for i in range(3) for j in range(3))


def _rot_x(deg: float) -> list:
    r = math.radians(deg); c, s = math.cos(r), math.sin(r)
    return [[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]]


def _rot_y(deg: float) -> list:
    r = math.radians(deg); c, s = math.cos(r), math.sin(r)
    return [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]]


def _rot_z(deg: float) -> list:
    r = math.radians(deg); c, s = math.cos(r), math.sin(r)
    return [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]


def _close_group(generators: list) -> list:
    _I = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    group: list = [_I]
    for g in generators:
        if not any(_mats_approx_equal(g, h) for h in group):
            group.append(g)
    changed = True
    while changed:
        changed = False
        additions: list = []
        for a in group:
            for b in group:
                prod = _matmul(a, b)
                if not any(_mats_approx_equal(prod, h) for h in group) and \
                   not any(_mats_approx_equal(prod, h) for h in additions):
                    additions.append(prod)
                    changed = True
        group.extend(additions)
    return group


def build_symmetry_group(symmetry: dict | None) -> list:
    """Build proper-rotation symmetry group from part metadata."""
    _I = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    if not symmetry:
        return [_I]
    rot_str = (symmetry.get("rotational_symmetry") or "none").lower()
    _fn = {"x": _rot_x, "y": _rot_y, "z": _rot_z}
    if "360" in rot_str:
        m = re.search(r"about\s+([xyz])", rot_str)
        axis = m.group(1) if m else "y"
        return [_fn[axis](float(d)) for d in range(360)]
    generators: list = []
    if rot_str.count("about") > 1:
        generators += [_rot_x(180.0), _rot_y(180.0), _rot_z(180.0)]
    elif rot_str != "none":
        angle_m = re.search(r"(\d+)°", rot_str)
        axis_m  = re.search(r"about\s+([xyz])", rot_str)
        if angle_m and axis_m:
            generators.append(_fn[axis_m.group(1)](float(angle_m.group(1))))
    sym_yz = bool(symmetry.get("sym_yz_plane"))
    sym_xz = bool(symmetry.get("sym_xz_plane"))
    sym_xy = bool(symmetry.get("sym_xy_plane"))
    if sym_yz and sym_xz:
        generators.append(_rot_z(180.0))
    if sym_yz and sym_xy:
        generators.append(_rot_y(180.0))
    if sym_xz and sym_xy:
        generators.append(_rot_x(180.0))
    return _close_group(generators)


def load_part_symmetry(brick_name: str, ldraw_color: int | None) -> dict | None:
    """Load the symmetry dict from renders_all_text_qwen/<part_id>[_c<color>].json."""
    part_id = brick_name.removesuffix(".dat")
    candidates = [SYMMETRY_DIR / f"{part_id}.json"]
    if ldraw_color is not None:
        candidates.insert(0, SYMMETRY_DIR / f"{part_id}_c{ldraw_color}.json")
    for path in candidates:
        if path.exists():
            try:
                return json.loads(path.read_text()).get("symmetry")
            except Exception:
                pass
    return None


def _geodesic_deg(R1: list, R2: list) -> float:
    R_rel = _matmul(_transpose(R1), R2)
    trace = R_rel[0][0] + R_rel[1][1] + R_rel[2][2]
    cos_val = max(-1.0, min(1.0, (trace - 1.0) / 2.0))
    return math.degrees(math.acos(cos_val))


def compute_geodesic_error_sym(
    pred_transform: list | None,
    gt_transform: list | None,
    symmetry_group: list,
) -> float | None:
    """
    Symmetry-aware geodesic error:
        min_{g in G}  geodesic(R_pred,  R_gt @ g)
    """
    if pred_transform is None or gt_transform is None:
        return None
    try:
        R_pred = _mat(pred_transform)
        R_gt   = _mat(gt_transform)
        return min(
            _geodesic_deg(R_pred, _matmul(R_gt, g))
            for g in symmetry_group
        )
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Task loading
# ---------------------------------------------------------------------------

def load_tasks(
    obj_filter: str | None,
    limit: int | None,
    testing_json_path: Path = TESTING_JSON,
) -> list[dict]:
    """
    Load evaluation tasks from Designer_supervision_assembly_testing.json.

    Each task dict contains:
        model_folder   : str  — e.g. "77756_Lego_Racers_-_Rocket_Racer_Car"
        step_k         : int
        step_dir       : Path — path to the step directory
        system_prompt  : str
        human_message  : str  — raw text with <image> placeholders
        image_paths    : list[Path]  — 4 image paths in order
        task_id        : str  — used for result directory name
    """
    entries = json.loads(testing_json_path.read_text())
    tasks: list[dict] = []

    for entry in entries:
        images = entry.get("images", [])
        if not images:
            continue

        # Derive step_dir and model_folder from the first image path
        step_dir     = Path(images[0]).parent
        model_folder = step_dir.parent.name
        step_name    = step_dir.name            # "step_XXXX"
        try:
            step_k = int(step_name.split("_")[1])
        except (IndexError, ValueError):
            continue

        if obj_filter is not None:
            model_id = model_folder.split("_")[0]
            if model_id != obj_filter:
                continue

        task_id = f"{model_folder}_step_{step_k:04d}"
        tasks.append({
            "model_folder":  model_folder,
            "step_k":        step_k,
            "step_dir":      step_dir,
            "system_prompt": entry["system"],
            "human_message": entry["conversations"][0]["value"],
            "image_paths":   [Path(p) for p in images],
            "task_id":       task_id,
        })

        if limit is not None and len(tasks) >= limit:
            break

    return tasks


# ---------------------------------------------------------------------------
# User content builder — converts <image> placeholders to API content list
# ---------------------------------------------------------------------------

def build_user_content(human_message: str, image_paths: list[Path]) -> list[dict]:
    """
    Convert a human message with <image> placeholders into the OpenAI content
    list format, substituting each placeholder with the corresponding image.

    Args:
        human_message : raw text with N occurrences of <image>
        image_paths   : exactly N image paths (in order)

    Returns:
        List of {"type": "text", "text": ...} and
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
        blocks.
    """
    parts = human_message.split("<image>")
    content: list[dict] = []

    for i, text_part in enumerate(parts):
        # Strip leading/trailing newlines from text segments while keeping
        # internal structure, but only add non-empty text blocks.
        stripped = text_part.strip("\n")
        if stripped:
            content.append({"type": "text", "text": stripped})

        if i < len(image_paths):
            img_path = image_paths[i]
            if not img_path.exists():
                # Replace missing image with a short notice in text
                content.append({
                    "type": "text",
                    "text": f"[Image {i+1} unavailable: {img_path.name}]",
                })
            else:
                b64 = img_to_b64(img_path)
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64}"},
                })

    return content


# ---------------------------------------------------------------------------
# Single task execution
# ---------------------------------------------------------------------------

def run_task(
    task: dict,
    base_url: str,
    model_name: str,
    sym_cache: dict,
    enable_thinking: bool = False,
) -> dict:
    """
    Run pose-estimation inference for one assembly step.

    Returns a result dict compatible with the standard result.json schema.
    Skips steps where brick_pose.json is missing or has no bricks.
    Computes symmetry-aware geodesic error at evaluation time.
    """
    step_k       = task["step_k"]
    model_folder = task["model_folder"]
    step_dir     = task["step_dir"]
    task_id      = task["task_id"]

    # ── Ground truth from brick_pose.json ─────────────────────────────────────
    bp_path = step_dir / "brick_pose.json"
    if not bp_path.exists():
        return {
            "task_id":       task_id,
            "model_folder":  model_folder,
            "step_k":        step_k,
            "skipped":       True,
            "reason":        "brick_pose.json missing",
        }

    brick_pose = json.loads(bp_path.read_text())

    if brick_pose.get("is_blank"):
        return {
            "task_id":       task_id,
            "model_folder":  model_folder,
            "step_k":        step_k,
            "skipped":       True,
            "reason":        "blank step",
        }

    gt_bricks = brick_pose.get("bricks", [])
    if not gt_bricks:
        return {
            "task_id":       task_id,
            "model_folder":  model_folder,
            "step_k":        step_k,
            "skipped":       True,
            "reason":        "no bricks in brick_pose.json",
        }

    # ── Check images exist ─────────────────────────────────────────────────────
    image_paths = task["image_paths"]
    missing = [str(p) for p in image_paths if not p.exists()]
    if missing:
        return {
            "task_id":       task_id,
            "model_folder":  model_folder,
            "step_k":        step_k,
            "skipped":       True,
            "reason":        f"missing images: {missing}",
        }

    # ── Build user content ─────────────────────────────────────────────────────
    user_content = build_user_content(task["human_message"], image_paths)

    # ── Call model ────────────────────────────────────────────────────────────
    t0 = time.time()
    try:
        raw_output = call_model(
            system_prompt=task["system_prompt"],
            user_content=user_content,
            enable_thinking=enable_thinking,
            base_url=base_url,
            model_name=model_name,
        )
    except RuntimeError as exc:
        err_str = str(exc)
        if "longer than the maximum model length" in err_str or "context" in err_str.lower():
            return {
                "task_id":       task_id,
                "model_folder":  model_folder,
                "step_k":        step_k,
                "skipped":       True,
                "reason":        f"context_length_exceeded: {err_str[:200]}",
            }
        raise
    elapsed = round(time.time() - t0, 2)

    # ── Parse and evaluate ────────────────────────────────────────────────────
    predictions = parse_pose_output(raw_output, num_bricks=len(gt_bricks))

    brick_results: list[dict] = []
    for pred, gt in zip(predictions, gt_bricks):
        brick_name  = gt["brick_name"]
        ldraw_color = gt.get("ldraw_color")

        offsets = compute_translation_offsets(pred, gt)
        t_err   = compute_translation_error(pred, gt)

        # Symmetry-aware rotation error
        cache_key = (brick_name, ldraw_color)
        if cache_key not in sym_cache:
            sym_meta = load_part_symmetry(brick_name, ldraw_color)
            sym_cache[cache_key] = build_symmetry_group(sym_meta)
        sym_group = sym_cache[cache_key]

        r_err = compute_geodesic_error_sym(
            pred.get("transform"),
            gt.get("transform"),
            sym_group,
        )

        brick_results.append({
            "brick_name":            brick_name,
            "ldraw_color":           ldraw_color,
            # Ground truth
            "gt_x":                  gt["x"],
            "gt_y":                  gt["y"],
            "gt_z":                  gt["z"],
            "gt_transform":          gt["transform"],
            # Model prediction
            "pred_x":                pred["x"],
            "pred_y":                pred["y"],
            "pred_z":                pred["z"],
            "pred_transform":        pred["transform"],
            # Signed per-axis offsets (pred − gt)
            "offset_x":              offsets["dx"] if offsets else None,
            "offset_y":              offsets["dy"] if offsets else None,
            "offset_z":              offsets["dz"] if offsets else None,
            # Scalar error metrics
            "translation_error_ldu": t_err,
            "geodesic_error_deg":    r_err,        # symmetry-aware
            "symmetry_group_size":   len(sym_group),
        })

    return {
        "task_id":      task_id,
        "model_folder": model_folder,
        "step_k":       step_k,
        "num_bricks":   len(gt_bricks),
        "raw_output":   raw_output,
        "elapsed_s":    elapsed,
        "brick_results": brick_results,
        "step_summary":  aggregate_step_metrics(brick_results),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate fine-tuned VLMs on LEGO brick pose estimation "
            "using Designer_supervision_assembly_testing.json."
        )
    )
    parser.add_argument(
        "--model", type=str, required=True,
        choices=list(MODEL_CONFIGS.keys()),
        help="Which model checkpoint to evaluate.",
    )
    parser.add_argument(
        "--obj", type=str, default=None,
        help=(
            "Only evaluate steps whose model folder starts with this ID "
            "(e.g. 77756 → 77756_Lego_Racers_-_Rocket_Racer_Car)."
        ),
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Max number of tasks to evaluate (useful for quick tests).",
    )
    parser.add_argument(
        "--thinking", action="store_true",
        help="Enable Qwen3 chain-of-thought thinking mode.",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Path for the summary JSON (default: results_pose_<model>/summary_<ts>.json).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    cfg       = MODEL_CONFIGS[args.model]
    base_url  = cfg["base_url"]
    model_name = cfg["model_name"]
    results_dir = _BASE / cfg["results_dir"]
    results_dir.mkdir(parents=True, exist_ok=True)

    tasks = load_tasks(obj_filter=args.obj, limit=args.limit)
    if not tasks:
        print(
            "No tasks found — check TESTING_JSON path and --obj filter."
        )
        return

    print(
        f"Model    : {args.model}  ({model_name} @ {base_url})"
    )
    print(
        f"Tasks    : {len(tasks)}"
        f"  (obj={args.obj or 'all'}, limit={args.limit or 'none'})"
    )
    print(f"Results  : {results_dir}")
    print(f"Symmetry : enabled (computed at inference time)")

    all_results:    list[dict]      = []
    results_by_obj: dict[str, list] = {}
    sym_cache:      dict            = {}
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    for task in tasks:
        task_id      = task["task_id"]
        model_folder = task["model_folder"]
        step_k       = task["step_k"]
        task_dir     = results_dir / task_id
        task_dir.mkdir(parents=True, exist_ok=True)
        result_path  = task_dir / "result.json"

        print(
            f"  [{model_folder}] step {step_k:04d}",
            end="",
            flush=True,
        )

        # Resume: skip already-completed tasks
        if result_path.exists():
            print("  SKIP (already done)")
            result = json.loads(result_path.read_text())
            all_results.append(result)
            results_by_obj.setdefault(model_folder, []).append(result)
            continue

        result = run_task(
            task,
            base_url=base_url,
            model_name=model_name,
            sym_cache=sym_cache,
            enable_thinking=args.thinking,
        )
        all_results.append(result)
        results_by_obj.setdefault(model_folder, []).append(result)

        # Per-step summary line
        if result.get("skipped"):
            print(f"  SKIP ({result.get('reason', '')})")
        else:
            brick_res = result.get("brick_results", [])
            t_errs = [b["translation_error_ldu"] for b in brick_res
                      if b.get("translation_error_ldu") is not None]
            r_errs = [b["geodesic_error_deg"] for b in brick_res
                      if b.get("geodesic_error_deg") is not None]
            t_str = f"t_err={sum(t_errs)/len(t_errs):.1f} LDU" if t_errs else "t_err=?"
            r_str = f"R_err={sum(r_errs)/len(r_errs):.1f}°"   if r_errs else "R_err=?"
            n     = result.get("num_bricks", 0)
            print(f"  n={n}  {t_str}  {r_str}")

        result_path.write_text(
            json.dumps(result, indent=2, ensure_ascii=False)
        )

    # ── Aggregation ───────────────────────────────────────────────────────────
    agg = aggregate_pose_metrics(all_results)
    print_summary(agg)

    # ── Write summary JSON ────────────────────────────────────────────────────
    run_config: dict[str, Any] = {
        "model":            args.model,
        "model_name":       model_name,
        "base_url":         base_url,
        "obj_filter":       args.obj,
        "limit":            args.limit,
        "enable_thinking":  args.thinking,
        "symmetry_aware":   True,
        "timestamp":        timestamp,
        "testing_json":     str(TESTING_JSON),
        "results_dir":      str(results_dir),
    }
    per_obj_agg = {
        obj: aggregate_pose_metrics(results)
        for obj, results in results_by_obj.items()
    }

    if args.output:
        summary_path = Path(args.output)
    else:
        tag = f"_{args.obj}" if args.obj else ""
        summary_path = results_dir / f"summary{tag}_{timestamp}.json"

    summary_path.write_text(
        json.dumps(
            {"run_config": run_config, "summary": agg, "by_object": per_obj_agg},
            indent=2,
            ensure_ascii=False,
        )
    )
    print(f"\nPer-step results → {results_dir}/<task_id>/result.json")
    print(f"Summary          → {summary_path}")


if __name__ == "__main__":
    main()
