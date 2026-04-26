"""scripts/filter_prompt_pool.py - filter the SFT dataset for GRPO training.

GRPO compute is wasted on prompts where the optimal answer is "match
PyMatching" — the model has nothing new to learn there. The filter keeps
only:

  * Prompts where PyMatching is wrong (PM logical-error distance > 0)
  * OR prompts with multi-error syndromes (>= 2 true errors), since these
    are where MWPM ambiguity matters and beat-rate has headroom.

Output is written to data/grpo_prompt_pool.jsonl (path configurable via
``--out``). The GRPO trainer loads from this filtered file via the
GRPO_PROMPT_POOL_PATH config knob.

Run::

    python -m scripts.filter_prompt_pool
    python -m scripts.filter_prompt_pool --in data/sft_dataset.jsonl \
        --out data/grpo_prompt_pool.jsonl

Exit codes:
    0  filtered pool written and audit passed
    1  audit failed (filtered pool too small or curriculum collapse)
    2  input file missing
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path
from typing import Iterable


def _is_kept(rec: dict) -> tuple[bool, str]:
    """Return (kept, reason) for a single SFT record."""
    actual = int(rec.get("actual_observable_flip", 0))
    pm_pred = int(rec.get("pymatching_observable_pred", 0))
    pm_wrong = (pm_pred != actual)
    if pm_wrong:
        return True, "pm_wrong"
    n_true_x = len(rec.get("true_x_errors", []))
    n_true_z = len(rec.get("true_z_errors", []))
    if (n_true_x + n_true_z) >= 2:
        return True, "multi_error"
    return False, "easy"


def main(argv: Iterable[str] = ()) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="in_path", type=str,
                        default="data/sft_dataset.jsonl")
    parser.add_argument("--out", dest="out_path", type=str,
                        default="data/grpo_prompt_pool.jsonl")
    parser.add_argument("--min-pool-size", type=int, default=500,
                        help="audit floor on filtered pool size")
    args = parser.parse_args(list(argv))

    src = Path(args.in_path)
    if not src.exists():
        print(f"[filter] FATAL: input dataset not found: {src}",
              file=sys.stderr)
        return 2

    in_rows: list[dict] = []
    with src.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            in_rows.append(json.loads(line))

    kept: list[dict] = []
    reasons = collections.Counter()
    for rec in in_rows:
        keep, why = _is_kept(rec)
        reasons[why] += 1
        if keep:
            kept.append(rec)

    out = Path(args.out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for rec in kept:
            f.write(json.dumps(rec) + "\n")

    n_in = len(in_rows)
    n_out = len(kept)
    pct = 100.0 * n_out / max(1, n_in)
    print()
    print("GRPO PROMPT POOL FILTER")
    print("=" * 23)
    print(f"input:           {src}  ({n_in} rows)")
    print(f"output:          {out}  ({n_out} rows, {pct:.1f}% retained)")
    print(f"keep reasons:    {dict(reasons)}")

    # Per-level breakdown of the filtered pool.
    by_level = collections.Counter(
        r.get("level", "unknown") for r in kept
    )
    print(f"per-level mix:   {dict(by_level)}")

    # Audit gates.
    failed: list[str] = []
    if n_out < args.min_pool_size:
        failed.append(
            f"filtered pool {n_out} < {args.min_pool_size} (--min-pool-size). "
            f"Either lower the floor or check the input dataset."
        )
    # Each curriculum level should have at least *some* representation.
    for level in ("L1_warmup", "L2_target", "L3_stretch"):
        if by_level.get(level, 0) == 0:
            failed.append(
                f"filtered pool has zero {level} prompts; the GRPO curriculum "
                f"mix cannot sample this level."
            )

    if failed:
        print()
        print("AUDIT FAILED:")
        for f in failed:
            print(f"  - {f}")
        return 1
    print()
    print(f"AUDIT PASSED — pool ready at {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
