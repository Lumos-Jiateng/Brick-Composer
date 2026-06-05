#!/usr/bin/env python3
"""
eval_test_split.py
------------------
Re-evaluates metrics for the FT model using only the test split
(tasks_with_color/test.json), reading pre-computed result.json files
from results_color/ and results_color_module/.

Usage
-----
  python eval_test_split.py
  python eval_test_split.py --test_set ../tasks_with_color/test.json
  python eval_test_split.py --output test_split_summary.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_BASE          = Path(__file__).parent.resolve()
RESULTS_DIR    = _BASE / "results_color"
RESULTS_MOD_DIR = _BASE / "results_color_module"
DEFAULT_TEST_SET = (
    Path(__file__).parent.parent / "tasks_with_color" / "test.json"
)

# ---------------------------------------------------------------------------
# Reuse aggregate_metrics / print_summary from the non-FT directory
# ---------------------------------------------------------------------------

sys.path.insert(0, str(_BASE))
from evaluate import aggregate_metrics, print_summary  # noqa: E402


# ---------------------------------------------------------------------------
# Key conversion helpers
# ---------------------------------------------------------------------------

def test_key_to_folder(key: str) -> str:
    """
    Convert a test.json key → result folder name.

    test.json entry : "10145_Micro_P38/step_0002_340"
    folder name     : "10145_step_0002_340"
                       ^obj_id ^step_part ^view
    """
    obj_name, step_part = key.split("/", 1)          # "10145_Micro_P38", "step_0002_340"
    obj_id = obj_name.split("_")[0]                  # "10145"
    _, step_k_str, view = step_part.split("_", 2)    # "step", "0002", "340"
    return f"{obj_id}_step_{step_k_str}_{view}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute test-split metrics for FT results.")
    p.add_argument(
        "--test_set", type=str, default=str(DEFAULT_TEST_SET),
        help="Path to test.json (default: ../tasks_with_color/test.json).",
    )
    p.add_argument(
        "--output", type=str, default=None,
        help="Optional path to write JSON summary.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    test_set_path = Path(args.test_set)
    if not test_set_path.exists():
        print(f"ERROR: test set file not found: {test_set_path}", file=sys.stderr)
        sys.exit(1)

    test_keys: list[str] = json.loads(test_set_path.read_text())
    test_folders: set[str] = {test_key_to_folder(k) for k in test_keys}
    print(f"Test set: {len(test_keys)} entries  ({test_set_path})")

    all_results: list[dict] = []
    missing: list[str] = []

    for folder_name in sorted(test_folders):
        # Try single-brick results first, then module results
        for root_dir in (RESULTS_DIR, RESULTS_MOD_DIR):
            result_path = root_dir / folder_name / "result.json"
            if result_path.exists():
                result = json.loads(result_path.read_text())
                all_results.append(result)
                break
        else:
            missing.append(folder_name)

    print(f"Found   : {len(all_results)} result files")
    if missing:
        print(f"Missing : {len(missing)} (no result.json found)")
        for m in missing[:10]:
            print(f"  {m}")
        if len(missing) > 10:
            print(f"  ... and {len(missing) - 10} more")

    if not all_results:
        print("No results to aggregate.", file=sys.stderr)
        sys.exit(1)

    agg = aggregate_metrics(all_results)
    print_summary(agg)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "test_set":    str(test_set_path),
            "found":       len(all_results),
            "missing":     len(missing),
            "summary":     agg,
        }
        out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
        print(f"\nSummary written → {out_path}")


if __name__ == "__main__":
    main()
