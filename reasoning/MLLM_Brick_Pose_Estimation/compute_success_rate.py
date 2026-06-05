#!/usr/bin/env python3
"""
compute_success_rate.py
-----------------------
Rule-based success-rate evaluation over results_pose_<model>/ outputs.

A brick prediction is "correct" when BOTH conditions hold:
  1. Translation: |offset_x| ≤ T_LDU  AND  |offset_y| ≤ T_LDU  AND  |offset_z| ≤ T_LDU
  2. Rotation   : symmetry-aware geodesic error < R_DEG

A step is "correct" when every brick in the step is correct.

Thresholds (editable):
  T_LDU = 10   (LDraw Units, per axis)
  R_DEG = 45   (degrees, after symmetry regularization)

Usage
-----
  python compute_success_rate.py --results_dir results_pose_real_only
  python compute_success_rate.py --results_dir results_pose_real_synthetic
  python compute_success_rate.py --results_dir results_pose_real_only --t_ldu 20 --r_deg 30
  python compute_success_rate.py --results_dir results_pose_real_only --output my_success.json
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

_BASE        = Path(__file__).parent.resolve()
SYMMETRY_DIR = Path(
    "/shared/nas/data/m1/jiateng5/BrickComposer"
    "/simulator_visualization/part_simulation/renders_all_text_qwen"
)


# ---------------------------------------------------------------------------
# Pure-Python 3×3 matrix helpers
# ---------------------------------------------------------------------------

def _mat(rows):
    return [list(r) for r in rows]

def _transpose(m):
    return [[m[r][c] for r in range(3)] for c in range(3)]

def _matmul(a, b):
    return [[sum(a[i][k]*b[k][j] for k in range(3)) for j in range(3)] for i in range(3)]

def _mats_approx_equal(a, b, tol=1e-5):
    return all(abs(a[i][j]-b[i][j]) <= tol for i in range(3) for j in range(3))

def _rot_x(deg):
    r=math.radians(deg); c,s=math.cos(r),math.sin(r)
    return [[1,0,0],[0,c,-s],[0,s,c]]

def _rot_y(deg):
    r=math.radians(deg); c,s=math.cos(r),math.sin(r)
    return [[c,0,s],[0,1,0],[-s,0,c]]

def _rot_z(deg):
    r=math.radians(deg); c,s=math.cos(r),math.sin(r)
    return [[c,-s,0],[s,c,0],[0,0,1]]

def _close_group(generators):
    _I=[[1,0,0],[0,1,0],[0,0,1]]
    group=[_I]
    for g in generators:
        if not any(_mats_approx_equal(g,h) for h in group):
            group.append(g)
    changed=True
    while changed:
        changed=False; additions=[]
        for a in group:
            for b in group:
                prod=_matmul(a,b)
                if not any(_mats_approx_equal(prod,h) for h in group) and \
                   not any(_mats_approx_equal(prod,h) for h in additions):
                    additions.append(prod); changed=True
        group.extend(additions)
    return group

def build_symmetry_group(symmetry):
    _I=[[1,0,0],[0,1,0],[0,0,1]]
    if not symmetry:
        return [_I]
    rot_str=(symmetry.get("rotational_symmetry") or "none").lower()
    _fn={"x":_rot_x,"y":_rot_y,"z":_rot_z}
    if "360" in rot_str:
        m=re.search(r"about\s+([xyz])",rot_str); axis=m.group(1) if m else "y"
        return [_fn[axis](float(d)) for d in range(360)]
    generators=[]
    if rot_str.count("about")>1:
        generators+=[_rot_x(180),_rot_y(180),_rot_z(180)]
    elif rot_str!="none":
        am=re.search(r"(\d+)°",rot_str); axm=re.search(r"about\s+([xyz])",rot_str)
        if am and axm:
            generators.append(_fn[axm.group(1)](float(am.group(1))))
    sym_yz=bool(symmetry.get("sym_yz_plane"))
    sym_xz=bool(symmetry.get("sym_xz_plane"))
    sym_xy=bool(symmetry.get("sym_xy_plane"))
    if sym_yz and sym_xz: generators.append(_rot_z(180))
    if sym_yz and sym_xy: generators.append(_rot_y(180))
    if sym_xz and sym_xy: generators.append(_rot_x(180))
    return _close_group(generators)

def load_part_symmetry(brick_name, ldraw_color):
    part_id=brick_name.removesuffix(".dat")
    candidates=[SYMMETRY_DIR/f"{part_id}.json"]
    if ldraw_color is not None:
        candidates.insert(0, SYMMETRY_DIR/f"{part_id}_c{ldraw_color}.json")
    for path in candidates:
        if path.exists():
            try:
                return json.loads(path.read_text()).get("symmetry")
            except Exception:
                pass
    return None

def geodesic_sym(pred_tf, gt_tf, sym_group):
    if pred_tf is None or gt_tf is None:
        return None
    try:
        R_pred=_mat(pred_tf); R_gt=_mat(gt_tf)
        best=float("inf")
        for g in sym_group:
            Rg=_matmul(R_gt,g)
            R_rel=_matmul(_transpose(R_pred),Rg)
            trace=R_rel[0][0]+R_rel[1][1]+R_rel[2][2]
            cv=max(-1.0,min(1.0,(trace-1.0)/2.0))
            best=min(best,math.degrees(math.acos(cv)))
        return best
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def compute(t_ldu: float, r_deg: float, results_dir: Path, output_path: Path | None) -> None:
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    task_dirs = sorted(
        d for d in results_dir.iterdir()
        if d.is_dir() and (d / "result.json").exists()
    )
    print(f"Results dir       : {results_dir}")
    print(f"Tasks found       : {len(task_dirs)}")
    print(f"Translation thresh: |dx|,|dy|,|dz| ≤ {t_ldu} LDU")
    print(f"Rotation thresh   : symmetry-aware geodesic < {r_deg}°\n")

    sym_cache: dict = {}

    # Counters
    total_steps     = 0
    skipped_steps   = 0
    evaluated_steps = 0
    correct_steps   = 0

    total_bricks    = 0
    correct_bricks  = 0
    # bricks where prediction was parseable (no None offsets/transform)
    evaluable_bricks = 0
    # breakdown reasons for brick failure
    fail_trans_only  = 0
    fail_rot_only    = 0
    fail_both        = 0

    by_obj: dict[str, dict] = {}

    for task_dir in task_dirs:
        result = json.loads((task_dir / "result.json").read_text())
        total_steps += 1

        if result.get("skipped"):
            skipped_steps += 1
            continue

        evaluated_steps += 1
        obj_name = result.get("object", task_dir.name)
        if obj_name not in by_obj:
            by_obj[obj_name] = {
                "evaluated_steps": 0, "correct_steps": 0,
                "total_bricks": 0, "correct_bricks": 0,
            }

        brick_results = result.get("brick_results", [])
        step_all_correct = True

        for b in brick_results:
            total_bricks += 1
            by_obj[obj_name]["total_bricks"] += 1

            ox = b.get("offset_x")
            oy = b.get("offset_y")
            oz = b.get("offset_z")

            # Translation check
            if ox is None or oy is None or oz is None:
                # Can't evaluate this brick
                step_all_correct = False
                continue

            evaluable_bricks += 1
            trans_ok = abs(ox) <= t_ldu and abs(oy) <= t_ldu and abs(oz) <= t_ldu

            # Rotation check (symmetry-aware)
            brick_name  = b.get("brick_name", "")
            ldraw_color = b.get("ldraw_color")
            cache_key   = (brick_name, ldraw_color)
            if cache_key not in sym_cache:
                sym_meta = load_part_symmetry(brick_name, ldraw_color)
                sym_cache[cache_key] = build_symmetry_group(sym_meta)
            sym_group = sym_cache[cache_key]

            geo = geodesic_sym(b.get("pred_transform"), b.get("gt_transform"), sym_group)
            rot_ok = (geo is not None) and (geo < r_deg)

            brick_correct = trans_ok and rot_ok
            if brick_correct:
                correct_bricks += 1
                by_obj[obj_name]["correct_bricks"] += 1
            else:
                step_all_correct = False
                if not trans_ok and not rot_ok:
                    fail_both += 1
                elif not trans_ok:
                    fail_trans_only += 1
                else:
                    fail_rot_only += 1

        by_obj[obj_name]["evaluated_steps"] += 1
        if step_all_correct:
            correct_steps += 1
            by_obj[obj_name]["correct_steps"] += 1

    # ── Print results ─────────────────────────────────────────────────────────
    def pct(a, b):
        return f"{100*a/b:.2f}%" if b else "N/A"

    print("=" * 62)
    print("  Success Rate  (rule-based regularization)")
    print("=" * 62)
    print(f"  Steps  : {total_steps} total, {skipped_steps} skipped, "
          f"{evaluated_steps} evaluated")
    print(f"  Correct steps  (all bricks pass): "
          f"{correct_steps} / {evaluated_steps}  =  {pct(correct_steps, evaluated_steps)}")
    print()
    print(f"  Bricks : {total_bricks} total, {evaluable_bricks} evaluable")
    print(f"  Correct bricks : "
          f"{correct_bricks} / {evaluable_bricks}  =  {pct(correct_bricks, evaluable_bricks)}")
    print()
    print("  Failure breakdown (evaluable bricks that failed):")
    failed = evaluable_bricks - correct_bricks
    print(f"    Translation only failed : {fail_trans_only}  ({pct(fail_trans_only, evaluable_bricks)})")
    print(f"    Rotation only failed    : {fail_rot_only}  ({pct(fail_rot_only, evaluable_bricks)})")
    print(f"    Both failed             : {fail_both}  ({pct(fail_both, evaluable_bricks)})")
    print("=" * 62)

    # ── Per-object breakdown ──────────────────────────────────────────────────
    print("\nPer-object step success rate:")
    per_obj_out: dict[str, Any] = {}
    for obj, s in sorted(by_obj.items()):
        ev  = s["evaluated_steps"]
        cor = s["correct_steps"]
        b_ev  = s["total_bricks"]
        b_cor = s["correct_bricks"]
        step_rate  = round(cor / ev, 4) if ev else None
        brick_rate = round(b_cor / b_ev, 4) if b_ev else None
        per_obj_out[obj] = {
            "evaluated_steps":  ev,
            "correct_steps":    cor,
            "step_success_rate":  step_rate,
            "total_bricks":     b_ev,
            "correct_bricks":   b_cor,
            "brick_success_rate": brick_rate,
        }
        print(f"  {obj:35s}  steps {cor}/{ev} ({pct(cor,ev)})  "
              f"bricks {b_cor}/{b_ev} ({pct(b_cor,b_ev)})")

    # ── Write JSON ────────────────────────────────────────────────────────────
    output: dict[str, Any] = {
        "run_config": {
            "translation_threshold_ldu": t_ldu,
            "rotation_threshold_deg":    r_deg,
            "symmetry_aware_rotation":   True,
            "source":    str(results_dir),
            "timestamp": timestamp,
            "note": (
                "A brick is 'correct' iff |offset_x|,|offset_y|,|offset_z| <= t_ldu "
                "AND symmetry-aware geodesic_error_deg < r_deg. "
                "A step is 'correct' iff all its bricks are correct."
            ),
        },
        "summary": {
            "total_steps":           total_steps,
            "skipped_steps":         skipped_steps,
            "evaluated_steps":       evaluated_steps,
            "correct_steps":         correct_steps,
            "step_success_rate":     round(correct_steps / evaluated_steps, 4) if evaluated_steps else None,
            "total_bricks":          total_bricks,
            "evaluable_bricks":      evaluable_bricks,
            "correct_bricks":        correct_bricks,
            "brick_success_rate":    round(correct_bricks / evaluable_bricks, 4) if evaluable_bricks else None,
            "failure_trans_only":    fail_trans_only,
            "failure_rot_only":      fail_rot_only,
            "failure_both":          fail_both,
        },
        "by_object": per_obj_out,
    }

    if output_path is None:
        output_path = results_dir / f"success_rate_t{int(t_ldu)}_r{int(r_deg)}_{timestamp}.json"

    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print(f"\nResults written → {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--results_dir", type=str, default=None,
        help=(
            "Path to results directory (e.g. results_pose_real_only or "
            "results_pose_real_synthetic).  Relative paths are resolved from "
            "the script directory.  Default: results_pose_real_only."
        ),
    )
    p.add_argument("--t_ldu", type=float, default=10.0,
                   help="Per-axis translation threshold in LDU (default: 10)")
    p.add_argument("--r_deg", type=float, default=45.0,
                   help="Rotation threshold in degrees (default: 45)")
    p.add_argument("--output", type=str, default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.results_dir:
        rd = Path(args.results_dir)
        if not rd.is_absolute():
            rd = _BASE / rd
    else:
        rd = _BASE / "results_pose_real_only"
    compute(args.t_ldu, args.r_deg, rd, Path(args.output) if args.output else None)
