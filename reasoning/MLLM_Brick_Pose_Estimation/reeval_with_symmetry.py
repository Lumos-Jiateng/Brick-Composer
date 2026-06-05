#!/usr/bin/env python3
"""
reeval_with_symmetry.py
-----------------------
Re-evaluates rotation errors in existing results_pose_<model>/ result.json
files using symmetry-aware geodesic error.

For each per-brick result the script:
  1. Loads the brick's symmetry metadata from SYMMETRY_DIR.
  2. Builds its proper-rotation symmetry group.
  3. Recomputes geodesic_error_deg as min_{g in G} geodesic(R_pred, R_gt @ g).

Translation errors are unchanged.  The existing result.json files are NOT
modified; a new summary JSON is written instead.

Note: evaluate_assembly.py already computes symmetry-aware geodesic at
inference time, so this script is mainly useful for re-aggregating or
cross-checking those results.

Output
------
  results_pose_<model>/summary_symmetry_<timestamp>.json

Usage
-----
  python reeval_with_symmetry.py --results_dir results_pose_real_only
  python reeval_with_symmetry.py --results_dir results_pose_real_synthetic
  python reeval_with_symmetry.py --results_dir results_pose_real_only --output my_summary.json
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_BASE = Path(__file__).parent.resolve()
SYMMETRY_DIR = Path(
    "/shared/nas/data/m1/jiateng5/BrickComposer"
    "/simulator_visualization/part_simulation/renders_all_text_qwen"
)


# ---------------------------------------------------------------------------
# Pure-Python 3×3 matrix helpers
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


# ---------------------------------------------------------------------------
# Rotation matrix constructors (pure Python SO(3))
# ---------------------------------------------------------------------------

def _rot_x(deg: float) -> list:
    r = math.radians(deg)
    c, s = math.cos(r), math.sin(r)
    return [[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]]


def _rot_y(deg: float) -> list:
    r = math.radians(deg)
    c, s = math.cos(r), math.sin(r)
    return [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]]


def _rot_z(deg: float) -> list:
    r = math.radians(deg)
    c, s = math.cos(r), math.sin(r)
    return [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]


def _close_group(generators: list) -> list:
    """
    Close a list of 3×3 rotation matrices under multiplication.
    Always includes the identity.
    """
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


# ---------------------------------------------------------------------------
# Symmetry group builder (identical to CoT_Eval pose_metrics.py)
# ---------------------------------------------------------------------------

def build_symmetry_group(symmetry: dict | None) -> list:
    """
    Build the proper-rotation symmetry group from a part's symmetry metadata.

    Returns a list of 3×3 rotation matrices g such that R_gt @ g is an
    equally valid ground-truth rotation.  Always contains the identity.
    """
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


# ---------------------------------------------------------------------------
# Symmetry metadata loader (identical to CoT_Eval evaluate_pose.py)
# ---------------------------------------------------------------------------

def load_part_symmetry(brick_name: str, ldraw_color: int | None) -> dict | None:
    """
    Load the "symmetry" dict from the part's JSON in SYMMETRY_DIR.

    Lookup order:
      1. <part_id>_c<ldraw_color>.json  (color-specific)
      2. <part_id>.json                 (default / color-neutral)
    """
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


# ---------------------------------------------------------------------------
# Symmetry-aware geodesic error
# ---------------------------------------------------------------------------

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
    Minimum geodesic angle (degrees) over all symmetry-equivalent GT orientations:
        error = min_{g in G}  geodesic(R_pred,  R_gt @ g)
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
# Aggregate helpers
# ---------------------------------------------------------------------------

def _avg(lst: list) -> float | None:
    return round(sum(lst) / len(lst), 4) if lst else None


def _median(lst: list) -> float | None:
    if not lst:
        return None
    s = sorted(lst)
    n = len(s)
    mid = n // 2
    return round(s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2.0, 4)


def aggregate_step_metrics(brick_results: list[dict]) -> dict[str, Any]:
    trans_errors = [
        b["translation_error_ldu"]
        for b in brick_results
        if b.get("translation_error_ldu") is not None
    ]
    rot_errors = [
        b["geodesic_error_deg"]
        for b in brick_results
        if b.get("geodesic_error_deg") is not None
    ]
    return {
        "num_bricks":             len(brick_results),
        "translation_evaluated":  len(trans_errors),
        "translation_mean_ldu":   _avg(trans_errors),
        "translation_median_ldu": _median(trans_errors),
        "rotation_evaluated":     len(rot_errors),
        "rotation_mean_deg":      _avg(rot_errors),
        "rotation_median_deg":    _median(rot_errors),
    }


def aggregate_pose_metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    brick_results: list[dict] = []
    for r in results:
        if r.get("skipped"):
            continue
        brick_results.extend(r.get("brick_results", []))

    if not brick_results:
        return {"evaluated_bricks": 0}

    trans_errors = [
        b["translation_error_ldu"]
        for b in brick_results
        if b.get("translation_error_ldu") is not None
    ]
    rot_errors = [
        b["geodesic_error_deg"]
        for b in brick_results
        if b.get("geodesic_error_deg") is not None
    ]
    evaluated_steps = sum(1 for r in results if not r.get("skipped"))
    return {
        "total_steps":            len(results),
        "skipped_blank":          sum(1 for r in results if r.get("skipped")),
        "evaluated_steps":        evaluated_steps,
        "total_bricks":           len(brick_results),
        "translation_evaluated":  len(trans_errors),
        "translation_mean_ldu":   _avg(trans_errors),
        "translation_median_ldu": _median(trans_errors),
        "rotation_evaluated":     len(rot_errors),
        "rotation_mean_deg":      _avg(rot_errors),
        "rotation_median_deg":    _median(rot_errors),
    }


def print_summary(agg: dict[str, Any]) -> None:
    w = 32
    print("\n" + "=" * 62)
    print("  Global Pose Evaluation Summary  (symmetry-aware)")
    print("=" * 62)
    for k, v in agg.items():
        val = f"{v:.4f}" if isinstance(v, float) else str(v)
        print(f"    {k.replace('_', ' ').title():<{w}} {val}")
    print("=" * 62)


# ---------------------------------------------------------------------------
# Main re-evaluation loop
# ---------------------------------------------------------------------------

def reeval(results_dir: Path, output_path: Path | None = None) -> None:
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    # Collect all task directories (each contains a result.json)
    task_dirs = sorted(
        d for d in results_dir.iterdir()
        if d.is_dir() and (d / "result.json").exists()
    )

    if not task_dirs:
        print(f"No result.json files found under {results_dir}")
        return

    print(f"Found {len(task_dirs)} task directories under {results_dir}")
    print(f"Symmetry data dir: {SYMMETRY_DIR}")
    print()

    all_results:    list[dict]      = []
    results_by_obj: dict[str, list] = {}

    sym_cache: dict[tuple, list] = {}   # (brick_name, ldraw_color) → group

    for i, task_dir in enumerate(task_dirs, 1):
        result = json.loads((task_dir / "result.json").read_text())

        if result.get("skipped"):
            obj_name = result.get("object", task_dir.name)
            all_results.append(result)
            results_by_obj.setdefault(obj_name, []).append(result)
            continue

        brick_results_orig = result.get("brick_results", [])
        brick_results_new: list[dict] = []

        for b in brick_results_orig:
            brick_name  = b.get("brick_name", "")
            ldraw_color = b.get("ldraw_color")

            cache_key = (brick_name, ldraw_color)
            if cache_key not in sym_cache:
                sym_meta = load_part_symmetry(brick_name, ldraw_color)
                sym_cache[cache_key] = build_symmetry_group(sym_meta)
            sym_group = sym_cache[cache_key]

            new_geodesic = compute_geodesic_error_sym(
                b.get("pred_transform"),
                b.get("gt_transform"),
                sym_group,
            )

            new_b = dict(b)
            new_b["geodesic_error_deg"]  = new_geodesic
            new_b["symmetry_group_size"] = len(sym_group)
            brick_results_new.append(new_b)

        new_result = dict(result)
        new_result["brick_results"] = brick_results_new
        new_result["step_summary"]  = aggregate_step_metrics(brick_results_new)

        obj_name = result.get("object", task_dir.name)
        all_results.append(new_result)
        results_by_obj.setdefault(obj_name, []).append(new_result)

        if i % 100 == 0 or i == len(task_dirs):
            print(f"  Processed {i}/{len(task_dirs)} tasks …")

    # Aggregate
    agg = aggregate_pose_metrics(all_results)
    print_summary(agg)

    per_obj_agg = {
        obj: aggregate_pose_metrics(results)
        for obj, results in results_by_obj.items()
    }

    run_config: dict[str, Any] = {
        "source":          str(results_dir),
        "symmetry_dir":    str(SYMMETRY_DIR),
        "symmetry_aware":  True,
        "timestamp":       timestamp,
        "note": (
            "Rotation errors recomputed with symmetry-aware geodesic metric "
            "(min over symmetry group).  Translation errors unchanged. "
            "Original result.json files are NOT modified."
        ),
    }

    if output_path is None:
        output_path = results_dir / f"summary_symmetry_{timestamp}.json"

    output_path.write_text(
        json.dumps(
            {"run_config": run_config, "summary": agg, "by_object": per_obj_agg},
            indent=2,
            ensure_ascii=False,
        )
    )
    print(f"\nSummary written → {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Re-evaluate rotation errors using the symmetry-aware geodesic metric. "
            "Use --results_dir to point at results_pose_real_only or "
            "results_pose_real_synthetic."
        )
    )
    parser.add_argument(
        "--results_dir", type=str, default=None,
        help=(
            "Path to results directory.  Relative paths resolved from script dir. "
            "Default: results_pose_real_only."
        ),
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Path for the output summary JSON.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.results_dir:
        rd = Path(args.results_dir)
        if not rd.is_absolute():
            rd = _BASE / rd
    else:
        rd = _BASE / "results_pose_real_only"
    out = Path(args.output) if args.output else None
    reeval(rd, out)
