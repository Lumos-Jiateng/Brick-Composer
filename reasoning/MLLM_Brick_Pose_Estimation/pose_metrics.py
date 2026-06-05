"""
pose_metrics.py
---------------
Parsing model output and computing evaluation metrics for global pose estimation.

Metrics per brick:
  offset_x/y/z            — Signed per-axis translation error (pred − gt) in LDU
  translation_error_ldu   — Euclidean distance |t_pred - t_gt| in LDraw Units
                             (1 LDU = 0.4 mm; 1 stud = 20 LDU = 8 mm)
  geodesic_error_deg      — Geodesic angle between predicted and GT rotation
                             matrices, in degrees:
                             θ = arccos( (trace(R_pred.T @ R_gt) - 1) / 2 )

Step-level summary (aggregate over all bricks in one step):
  step_summary            — mean/median translation error and rotation error

Cross-step aggregate metrics (mean and median over all evaluated bricks):
  translation_mean_ldu / translation_median_ldu
  rotation_mean_deg    / rotation_median_deg
"""

from __future__ import annotations

import math
import re
from typing import Any


# ---------------------------------------------------------------------------
# Output parser
# ---------------------------------------------------------------------------

def parse_pose_output(text: str, num_bricks: int) -> list[dict]:
    """
    Parse the model's raw text into a list of pose dicts, one per brick.

    Expected format per brick (header required for multi-brick steps):
      Brick <N>:
      Translation: x=<float>, y=<float>, z=<float>
      Rotation:
      [<r00>, <r01>, <r02>]
      [<r10>, <r11>, <r12>]
      [<r20>, <r21>, <r22>]

    Returns:
        List of length `num_bricks`.  Missing predictions are represented as
        {"x": None, "y": None, "z": None, "transform": None}.
    """
    results: list[dict] = []

    if num_bricks > 1:
        # Split on "Brick N:" headers, drop any preamble before the first header
        sections = re.split(r"Brick\s+\d+\s*:", text, flags=re.IGNORECASE)
        sections = sections[1:]  # discard preamble
    else:
        sections = [text]

    for section in sections[:num_bricks]:
        results.append(_parse_single_pose(section))

    # Pad to expected length
    _null = {"x": None, "y": None, "z": None, "transform": None}
    while len(results) < num_bricks:
        results.append(dict(_null))

    return results


def _parse_single_pose(text: str) -> dict:
    """
    Parse x, y, z translation and a 3×3 rotation matrix from one text block.
    """
    result: dict = {"x": None, "y": None, "z": None, "transform": None}

    # Translation — accept "x=<val>", "x: <val>", "x = <val>", etc.
    _float = r"(-?[\d]+(?:\.[\d]+)?(?:[eE][+-]?[\d]+)?)"
    for axis in ("x", "y", "z"):
        m = re.search(rf"{axis}\s*[=:]\s*{_float}", text, re.IGNORECASE)
        if m:
            result[axis] = float(m.group(1))

    # Rotation matrix — find 3 bracket-enclosed rows of 3 floats each
    row_pat = (
        r"\[\s*" + _float + r"\s*,\s*" + _float + r"\s*,\s*" + _float + r"\s*\]"
    )
    rows = re.findall(row_pat, text)
    if len(rows) >= 3:
        result["transform"] = [
            [float(rows[0][0]), float(rows[0][1]), float(rows[0][2])],
            [float(rows[1][0]), float(rows[1][1]), float(rows[1][2])],
            [float(rows[2][0]), float(rows[2][1]), float(rows[2][2])],
        ]

    return result


# ---------------------------------------------------------------------------
# Per-brick error computation
# ---------------------------------------------------------------------------

def compute_translation_offsets(pred: dict, gt: dict) -> dict | None:
    """
    Signed per-axis translation offsets (pred − gt) in LDU.

    Args:
        pred : dict with keys "x", "y", "z" (floats or None).
        gt   : brick entry from brick_pose.json with keys "x", "y", "z".

    Returns:
        {"dx": float, "dy": float, "dz": float}, or None if prediction is
        incomplete.
    """
    if pred["x"] is None or pred["y"] is None or pred["z"] is None:
        return None
    return {
        "dx": round(pred["x"] - gt["x"], 4),
        "dy": round(pred["y"] - gt["y"], 4),
        "dz": round(pred["z"] - gt["z"], 4),
    }


def compute_translation_error(pred: dict, gt: dict) -> float | None:
    """
    Euclidean distance between predicted and ground-truth translation (LDU).

    Args:
        pred : dict with keys "x", "y", "z" (floats or None).
        gt   : brick entry from brick_pose.json with keys "x", "y", "z".

    Returns:
        Distance in LDU, or None if prediction is incomplete.
    """
    if pred["x"] is None or pred["y"] is None or pred["z"] is None:
        return None
    dx = pred["x"] - gt["x"]
    dy = pred["y"] - gt["y"]
    dz = pred["z"] - gt["z"]
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def compute_geodesic_error(pred: dict, gt: dict) -> float | None:
    """
    Geodesic angle (degrees) between predicted and ground-truth rotation matrix.

    The geodesic distance between two rotation matrices R1, R2 is:
        θ = arccos( (trace(R1.T @ R2) - 1) / 2 )

    Args:
        pred : dict with key "transform" (list[list[float, 3], 3] or None).
        gt   : brick entry from brick_pose.json with key "transform".

    Returns:
        Angle in degrees in [0, 180], or None if prediction is incomplete.
    """
    if pred["transform"] is None:
        return None
    try:
        R_pred = _mat(pred["transform"])
        R_gt   = _mat(gt["transform"])
        # Relative rotation
        R_rel  = _matmul(_transpose(R_pred), R_gt)
        trace  = R_rel[0][0] + R_rel[1][1] + R_rel[2][2]
        # Clamp to [-1, 1] to guard against numerical noise
        cos_val = max(-1.0, min(1.0, (trace - 1.0) / 2.0))
        return math.degrees(math.acos(cos_val))
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Pure-Python 3×3 matrix helpers (avoid mandatory numpy dependency at import)
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


# ---------------------------------------------------------------------------
# Aggregate metrics
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
    """
    Compute mean/median translation and rotation error for one step's bricks.

    Args:
        brick_results : list of brick result dicts from a single step, each
                        containing "translation_error_ldu" and
                        "geodesic_error_deg" (possibly None).

    Returns:
        Dict with step-level counts and mean/median for both metrics.
    """
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
        "num_bricks":                len(brick_results),
        "translation_evaluated":     len(trans_errors),
        "translation_mean_ldu":      _avg(trans_errors),
        "translation_median_ldu":    _median(trans_errors),
        "rotation_evaluated":        len(rot_errors),
        "rotation_mean_deg":         _avg(rot_errors),
        "rotation_median_deg":       _median(rot_errors),
    }


def aggregate_pose_metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Aggregate per-step results into summary statistics.

    Each result dict may contain a "brick_results" list, each entry holding
    "translation_error_ldu" and "geodesic_error_deg" (possibly None).
    """
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
        "total_steps":               len(results),
        "skipped_blank":             sum(1 for r in results if r.get("skipped")),
        "evaluated_steps":           evaluated_steps,
        "total_bricks":              len(brick_results),
        "translation_evaluated":     len(trans_errors),
        "translation_mean_ldu":      _avg(trans_errors),
        "translation_median_ldu":    _median(trans_errors),
        "rotation_evaluated":        len(rot_errors),
        "rotation_mean_deg":         _avg(rot_errors),
        "rotation_median_deg":       _median(rot_errors),
    }


# ---------------------------------------------------------------------------
# Pretty-print helper
# ---------------------------------------------------------------------------

def print_summary(agg: dict[str, Any]) -> None:
    w = 32
    print("\n" + "=" * 62)
    print("  Global Pose Evaluation Summary")
    print("=" * 62)
    for k, v in agg.items():
        val = f"{v:.4f}" if isinstance(v, float) else str(v)
        print(f"    {k.replace('_', ' ').title():<{w}} {val}")
    print("=" * 62)
