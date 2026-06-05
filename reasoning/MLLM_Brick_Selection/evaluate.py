"""
evaluate.py
-----------
Parsing model output and computing evaluation metrics.

Single-brick steps
    The model must output exactly one brick + position.
    Result: True / False  (either_correct = brick_correct OR position_correct)

Module steps
    The model outputs one line per component brick.
    Result: recall / precision / F1 at the brick-multiset level,
            plus either_correct = brick_exact_match OR position_any_match.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def extract_dat_files(text: str) -> list[str]:
    """
    Extract all ``xxxxx.dat`` filenames from raw model output.
    Returns lower-case names, preserving order and duplicates.

    >>> extract_dat_files("3023.dat, row 2, col 4")
    ['3023.dat']
    """
    return [m.lower() for m in re.findall(r"[\w\-]+\.dat", text, re.IGNORECASE)]


def extract_positions(text: str) -> list[dict[str, int]]:
    """
    Extract (row, col) pairs from model output.
    Matches patterns like "row 2, col 4" / "row:2 col:4".

    >>> extract_positions("3023.dat, row 2, col 4")
    [{'row': 2, 'col': 4}]
    """
    pattern = r"row\s*[:\s]\s*(\d+)[,\s]+col\s*[:\s]\s*(\d+)"
    return [
        {"row": int(r), "col": int(c)}
        for r, c in re.findall(pattern, text, re.IGNORECASE)
    ]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _pos_in_candidates(pos: dict[str, int], gt_position: list[dict]) -> bool:
    """True if *pos* matches any candidate_positions entry across all gt_position items."""
    for entry in gt_position:
        for cand in entry.get("candidate_positions", []):
            if cand["row"] == pos["row"] and cand["col"] == pos["col"]:
                return True
    return False


# ---------------------------------------------------------------------------
# Per-task metrics — single-brick path
# ---------------------------------------------------------------------------

def _metrics_single(
    raw_output:  str,
    gt_bricks:   list[str],
    gt_position: list[dict] | None,
) -> dict[str, Any]:
    """
    Evaluate a single-brick step.

    Only the FIRST predicted brick and FIRST predicted position are considered
    (the model was asked for exactly one).

    Returns
    -------
    brick_correct    – first predicted .dat == GT brick (case-insensitive)
    position_correct – first predicted (row,col) is in GT candidate_positions
    either_correct   – brick_correct OR position_correct  ← primary metric
    """
    predicted_bricks    = extract_dat_files(raw_output)
    predicted_positions = extract_positions(raw_output)

    gt_brick = gt_bricks[0].lower() if gt_bricks else ""

    brick_correct = bool(predicted_bricks) and predicted_bricks[0].lower() == gt_brick

    position_correct = False
    if gt_position and predicted_positions:
        position_correct = _pos_in_candidates(predicted_positions[0], gt_position)

    either_correct = brick_correct or position_correct

    return {
        "predicted_bricks":    predicted_bricks,
        "predicted_positions": predicted_positions,
        "brick_correct":       brick_correct,
        "position_correct":    position_correct,
        "either_correct":      either_correct,
        "is_module":           False,
    }


# ---------------------------------------------------------------------------
# Per-task metrics — module path
# ---------------------------------------------------------------------------

def _metrics_module(
    raw_output:  str,
    gt_bricks:   list[str],
    gt_position: list[dict] | None,
) -> dict[str, Any]:
    """
    Evaluate a module step (multiple bricks).

    brick_exact_match  – predicted brick multiset == GT brick multiset
    recall / precision / f1  – multiset-intersection partial credit
    position_any_match – any predicted (row,col) in GT candidate_positions
    either_correct     – brick_exact_match OR position_any_match
    """
    predicted_bricks    = extract_dat_files(raw_output)
    predicted_positions = extract_positions(raw_output)

    gt_counter   = Counter(b.lower() for b in gt_bricks)
    pred_counter = Counter(b.lower() for b in predicted_bricks)

    tp        = sum((gt_counter & pred_counter).values())
    recall    = tp / sum(gt_counter.values())
    precision = tp / sum(pred_counter.values()) if pred_counter else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0 else 0.0
    )
    brick_exact_match = (gt_counter == pred_counter)

    position_any_match = False
    if gt_position and predicted_positions:
        position_any_match = any(
            _pos_in_candidates(p, gt_position) for p in predicted_positions
        )

    either_correct = brick_exact_match or position_any_match

    return {
        "predicted_bricks":    predicted_bricks,
        "predicted_positions": predicted_positions,
        "brick_exact_match":   brick_exact_match,
        "position_any_match":  position_any_match,
        "either_correct":      either_correct,
        "recall":              round(recall,    4),
        "precision":           round(precision, 4),
        "f1":                  round(f1,        4),
        "is_module":           True,
    }


# ---------------------------------------------------------------------------
# Public dispatcher
# ---------------------------------------------------------------------------

def compute_metrics(
    raw_output:   str,
    gt_bricks:    list[str],
    ground_truth: str,
    gt_position:  list[dict] | None = None,
) -> dict[str, Any]:
    """
    Dispatch to the appropriate evaluation path based on step type.

    Blank steps (empty gt_bricks) are returned as skipped.
    """
    is_module = ground_truth.startswith('module "')

    if not gt_bricks:
        predicted_bricks    = extract_dat_files(raw_output)
        predicted_positions = extract_positions(raw_output)
        return {
            "predicted_bricks":    predicted_bricks,
            "predicted_positions": predicted_positions,
            "either_correct":      False,
            "is_module":           is_module,
            "note":                "blank step — not evaluated",
        }

    if is_module:
        return _metrics_module(raw_output, gt_bricks, gt_position)
    else:
        return _metrics_single(raw_output, gt_bricks, gt_position)


# ---------------------------------------------------------------------------
# Aggregate metrics
# ---------------------------------------------------------------------------

def aggregate_metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Aggregate per-task results.

    Single-brick tasks → accuracy (either_correct rate)
    Module tasks       → mean F1, recall, precision + either_correct rate
    """
    evaluated = [
        r for r in results
        if not r.get("skipped") and r.get("note") != "blank step — not evaluated"
    ]
    if not evaluated:
        return {"evaluated": 0}

    single  = [r for r in evaluated if not r.get("is_module")]
    modules = [r for r in evaluated if r.get("is_module")]

    def _rate(lst: list[dict], key: str) -> float:
        if not lst:
            return 0.0
        return round(sum(bool(r.get(key)) for r in lst) / len(lst), 4)

    def _avg(lst: list[dict], key: str) -> float:
        vals = [r[key] for r in lst if r.get(key) is not None]
        return round(sum(vals) / len(vals), 4) if vals else 0.0

    return {
        "total_tasks":          len(results),
        "evaluated":            len(evaluated),
        "skipped_blank":        sum(1 for r in results if r.get("skipped")),
        # ── Single-brick ──────────────────────────────────────────────────
        "single_count":         len(single),
        "single_either_acc":    _rate(single, "either_correct"),
        "single_brick_acc":     _rate(single, "brick_correct"),
        "single_position_acc":  _rate(single, "position_correct"),
        # ── Module ────────────────────────────────────────────────────────
        "module_count":         len(modules),
        "module_either_acc":    _rate(modules, "either_correct"),
        "module_brick_exact":   _rate(modules, "brick_exact_match"),
        "module_mean_f1":       _avg(modules,  "f1"),
        "module_mean_recall":   _avg(modules,  "recall"),
        "module_mean_precision": _avg(modules, "precision"),
        "module_position_acc":  _rate(modules, "position_any_match"),
    }


# ---------------------------------------------------------------------------
# Pretty-print helper
# ---------------------------------------------------------------------------

def print_summary(agg: dict[str, Any]) -> None:
    w = 28
    print("\n" + "=" * 55)
    print("  Evaluation Summary")
    print("=" * 55)
    sections = {
        "Overall":        ["total_tasks", "evaluated", "skipped_blank"],
        "Single-brick":   [k for k in agg if k.startswith("single_")],
        "Module":         [k for k in agg if k.startswith("module_")],
    }
    for section, keys in sections.items():
        print(f"\n  [{section}]")
        for k in keys:
            if k not in agg:
                continue
            v   = agg[k]
            val = f"{v:.4f}" if isinstance(v, float) else str(v)
            print(f"    {k.replace('_',' ').title():<{w}} {val}")
    print("=" * 55)
