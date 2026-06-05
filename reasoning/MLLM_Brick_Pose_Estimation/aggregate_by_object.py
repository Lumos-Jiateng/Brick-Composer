#!/usr/bin/env python3
"""
aggregate_by_object.py
----------------------
Aggregate per-step result.json files in a results_pose_*/ folder up to the
OBJECT level (= model_folder, e.g. 77756_Lego_Racers_-_Rocket_Racer_Car), then
report:
  - the macro average across objects
  - the SINGLE best object (chosen by highest step success rate, ties broken
    by lowest translation error then lowest rotation error)

Six numbers per folder:
  avg_object_translation_ldu     best_object_translation_ldu
  avg_object_rotation_deg        best_object_rotation_deg
  avg_object_success_rate        best_object_success_rate

The per-step result.json files already contain symmetry-aware
geodesic_error_deg (symmetry_group_size > 1), so no re-computation is needed.

Success rate uses the same thresholds as compute_success_rate.py:
  translation  : |offset_x|,|offset_y|,|offset_z| <= 10 LDU
  rotation     : geodesic_error_deg < 45°
  step correct iff ALL bricks in the step are correct
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean


T_LDU = 10.0
R_DEG = 45.0


# ---------------------------------------------------------------------------
# Per-step helpers
# ---------------------------------------------------------------------------

def step_metrics(result: dict) -> tuple[float | None, float | None, bool | None]:
    """
    Returns (step_mean_translation_ldu, step_mean_rotation_deg, step_correct).
    Returns (None, None, None) for skipped steps.
    """
    if result.get("skipped"):
        return None, None, None

    bricks = result.get("brick_results", [])
    if not bricks:
        return None, None, None

    t_errs = [b["translation_error_ldu"] for b in bricks
              if b.get("translation_error_ldu") is not None]
    r_errs = [b["geodesic_error_deg"] for b in bricks
              if b.get("geodesic_error_deg") is not None]

    if not t_errs and not r_errs:
        return None, None, None

    t_mean = mean(t_errs) if t_errs else None
    r_mean = mean(r_errs) if r_errs else None

    def _brick_correct(b: dict) -> bool:
        ox, oy, oz = b.get("offset_x"), b.get("offset_y"), b.get("offset_z")
        ge = b.get("geodesic_error_deg")
        if None in (ox, oy, oz, ge):
            return False
        return (abs(ox) <= T_LDU and abs(oy) <= T_LDU and abs(oz) <= T_LDU
                and ge < R_DEG)

    step_correct = all(_brick_correct(b) for b in bricks)
    return t_mean, r_mean, step_correct


# ---------------------------------------------------------------------------
# Per-object aggregation
# ---------------------------------------------------------------------------

def aggregate_folder(results_dir: Path) -> dict[str, dict]:
    """
    Returns {object_name: {translation_ldu, rotation_deg, success_rate,
                           num_steps}}.
    """
    by_obj: dict[str, list[tuple[float | None, float | None, bool | None]]] = {}

    for task_dir in sorted(results_dir.iterdir()):
        if not task_dir.is_dir():
            continue
        rp = task_dir / "result.json"
        if not rp.exists():
            continue
        try:
            result = json.loads(rp.read_text())
        except Exception:
            continue

        obj = result.get("model_folder")
        if not obj:
            # Fallback: derive from task_id by stripping "_step_XXXX"
            tid = result.get("task_id") or task_dir.name
            obj = tid.rsplit("_step_", 1)[0]

        by_obj.setdefault(obj, []).append(step_metrics(result))

    summary: dict[str, dict] = {}
    for obj, step_results in by_obj.items():
        evaluated = [(t, r, s) for (t, r, s) in step_results if t is not None or r is not None or s is not None]
        if not evaluated:
            continue
        t_vals = [t for (t, _, _) in evaluated if t is not None]
        r_vals = [r for (_, r, _) in evaluated if r is not None]
        s_vals = [s for (_, _, s) in evaluated if s is not None]
        summary[obj] = {
            "num_steps":       len(evaluated),
            "translation_ldu": round(mean(t_vals), 4) if t_vals else None,
            "rotation_deg":    round(mean(r_vals), 4) if r_vals else None,
            "success_rate":    round(sum(s_vals) / len(s_vals), 4) if s_vals else None,
        }
    return summary


def pick_best_object(per_obj: dict[str, dict]) -> str:
    """
    Best object = highest success rate, then lowest translation,
    then lowest rotation.
    """
    def _key(obj: str) -> tuple:
        m = per_obj[obj]
        return (
            -(m["success_rate"] or 0.0),
            (m["translation_ldu"] if m["translation_ldu"] is not None else float("inf")),
            (m["rotation_deg"]    if m["rotation_deg"]    is not None else float("inf")),
        )
    return min(per_obj.keys(), key=_key)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def report_folder(folder: Path) -> dict:
    per_obj = aggregate_folder(folder)
    if not per_obj:
        raise SystemExit(f"No evaluable per-step results under {folder}")

    objects = sorted(per_obj.keys())
    avg_t = round(mean(per_obj[o]["translation_ldu"] for o in objects
                       if per_obj[o]["translation_ldu"] is not None), 4)
    avg_r = round(mean(per_obj[o]["rotation_deg"]    for o in objects
                       if per_obj[o]["rotation_deg"]    is not None), 4)
    avg_s = round(mean(per_obj[o]["success_rate"]    for o in objects
                       if per_obj[o]["success_rate"]    is not None), 4)

    best_overall_obj = pick_best_object(per_obj)
    best_overall     = per_obj[best_overall_obj]

    # Best PER METRIC (matches trivial_baseline_summary.json format)
    def _arg_best(key: str, better):
        cands = [(o, per_obj[o][key]) for o in objects if per_obj[o][key] is not None]
        return better(cands, key=lambda kv: kv[1]) if cands else (None, None)

    best_trans_obj, best_trans_val = _arg_best("translation_ldu", min)
    best_rot_obj,   best_rot_val   = _arg_best("rotation_deg",    min)
    best_sr_obj,    best_sr_val    = _arg_best("success_rate",    max)

    payload = {
        "folder":            str(folder),
        "num_objects":       len(objects),
        "thresholds":        {"t_ldu": T_LDU, "r_deg": R_DEG},
        "macro_average": {
            "translation_ldu": avg_t,
            "rotation_deg":    avg_r,
            "success_rate":    avg_s,
        },
        "best_per_metric": {
            "translation_ldu": {"object": best_trans_obj, "value": best_trans_val},
            "rotation_deg":    {"object": best_rot_obj,   "value": best_rot_val},
            "success_rate":    {"object": best_sr_obj,    "value": best_sr_val},
        },
        "best_object_by_success_rate": {
            "name":            best_overall_obj,
            "num_steps":       best_overall["num_steps"],
            "translation_ldu": best_overall["translation_ldu"],
            "rotation_deg":    best_overall["rotation_deg"],
            "success_rate":    best_overall["success_rate"],
        },
        "per_object": {o: per_obj[o] for o in objects},
    }
    return payload


def print_report(name: str, payload: dict) -> None:
    print()
    print("=" * 82)
    print(f"  {name}")
    print(f"  {payload['folder']}")
    print(f"  {payload['num_objects']} objects   |   "
          f"thresholds: |offset|≤{T_LDU} LDU, geodesic<{R_DEG}°")
    print("=" * 82)
    avg = payload["macro_average"]
    bpm = payload["best_per_metric"]
    print(f"  {'Metric':<28} {'Macro Avg':>12} {'Best Value':>12}   Best object")
    print(f"  {'-'*28} {'-'*12} {'-'*12}   {'-'*30}")
    print(f"  {'Translation error (LDU)':<28} "
          f"{avg['translation_ldu']:>12.4f} {bpm['translation_ldu']['value']:>12.4f}   {bpm['translation_ldu']['object']}")
    print(f"  {'Rotation error  (deg)':<28} "
          f"{avg['rotation_deg']:>12.4f} {bpm['rotation_deg']['value']:>12.4f}   {bpm['rotation_deg']['object']}")
    print(f"  {'Success rate    (frac)':<28} "
          f"{avg['success_rate']:>12.4f} {bpm['success_rate']['value']:>12.4f}   {bpm['success_rate']['object']}")
    print("=" * 82)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "folders", nargs="+",
        help="One or more results_pose_*/ directories to aggregate.",
    )
    p.add_argument(
        "--write_summary", action="store_true",
        help="Also write summary_by_object.json next to each input folder.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    for f in args.folders:
        folder = Path(f).resolve()
        if not folder.is_dir():
            print(f"[skip] not a directory: {folder}")
            continue
        payload = report_folder(folder)
        print_report(folder.name, payload)

        if args.write_summary:
            out = folder / "summary_by_object.json"
            out.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
            print(f"  → {out}")


if __name__ == "__main__":
    main()
