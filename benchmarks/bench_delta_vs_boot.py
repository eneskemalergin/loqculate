"""Wall time and agreement: delta-method LOQ vs bootstrap LOQ.

Uses the 27-peptide demo calibration set with the same data harness as the CF
full-pipeline ~30 ms median context in ``bench_knot_vs_curvefit``.

Reports:

* Full-pipeline wall time: ``fit(...).loq()`` (bootstrap) vs
  ``fit(...).loq_delta()`` (analytical).
* LOQ-only wall time on a shared fitted model (fit cost excluded).
* Agreement pilot: delta vs bootstrap LOQ, split by linear-segment size
  ``n_L`` among both-finite pairs.

Run from the repository root::

    python benchmarks/bench_delta_vs_boot.py
    python benchmarks/bench_delta_vs_boot.py --quick
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np

_REPO = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(Path(__file__).parent))

from _helpers import DEMO_DATA, DEMO_MAP, _json_safe  # noqa: E402

from loqculate.io import read_calibration_data  # noqa: E402
from loqculate.models.piecewise_cf import PiecewiseCF  # noqa: E402

SEED = 42


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Delta-method vs bootstrap LOQ: wall time and agreement"
    )
    p.add_argument(
        "--n_boot_reps",
        type=int,
        default=200,
        help="Bootstrap replicates for bootstrap LOQ (default: 200)",
    )
    p.add_argument(
        "--n_timing",
        type=int,
        default=5,
        help="Timing repetitions per peptide (default: 5)",
    )
    p.add_argument(
        "--save",
        type=str,
        default=str(_REPO / "tmp" / "results" / "bench_delta_vs_boot.json"),
        metavar="PATH",
        help="Write JSON results to PATH",
    )
    p.add_argument(
        "--quick",
        action="store_true",
        help="Fast smoke mode: n_boot_reps=50, n_timing=3",
    )
    return p.parse_args()


def _load_peptides() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    data = read_calibration_data(str(DEMO_DATA), str(DEMO_MAP))
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for pep in np.unique(data.peptide):
        m = data.peptide == pep
        out[str(pep)] = (data.concentration[m], data.area[m])
    return out


def _n_lin(cf: PiecewiseCF) -> int:
    return int(np.sum(cf.x_ > cf.params_["knot_x"]))


def _rel_diff(a: float, b: float) -> float | None:
    if not (np.isfinite(a) and np.isfinite(b)):
        return None
    denom = max(abs(a), abs(b), 1e-30)
    return float(abs(a - b) / denom)


def _timing_aggregate(
    boot_medians: list[float],
    delta_medians: list[float],
    speedups: list[float],
    *,
    n_boot_reps: int,
    n_timing: int,
    workload: str,
) -> dict[str, Any]:
    return {
        "boot_median_ms": float(np.median(boot_medians)),
        "delta_median_ms": float(np.median(delta_medians)),
        "median_speedup_boot_over_delta": float(np.median(speedups)),
        "ratio_of_medians_boot_over_delta": float(
            np.median(boot_medians) / np.median(delta_medians)
        ),
        "n_peptides": len(boot_medians),
        "n_boot_reps": n_boot_reps,
        "n_timing": n_timing,
        "seed": SEED,
        "workload": workload,
    }


def exp_full_pipeline(
    peptides: dict[str, tuple[np.ndarray, np.ndarray]],
    *,
    n_boot_reps: int,
    n_timing: int,
) -> dict[str, Any]:
    """Time fit+LOQ for bootstrap vs delta on each peptide."""
    print(f"\nFull-pipeline timing  (n_boot_reps={n_boot_reps}, n_timing={n_timing})")
    print("  boot: PiecewiseCF(n_boot_reps=N).fit(x, y).loq()")
    print("  delta: PiecewiseCF(n_boot_reps=0).fit(x, y).loq_delta()")
    print("  Timing loops are separate (all boot reps, then all delta reps).")

    rows: list[dict[str, Any]] = []
    boot_medians: list[float] = []
    delta_medians: list[float] = []
    speedups: list[float] = []

    for pep, (x, y) in peptides.items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            PiecewiseCF(n_boot_reps=n_boot_reps, seed=SEED).fit(x, y).loq()
            PiecewiseCF(n_boot_reps=0, seed=SEED).fit(x, y).loq_delta()

        boot_times: list[float] = []
        for _ in range(n_timing):
            t0 = time.perf_counter()
            PiecewiseCF(n_boot_reps=n_boot_reps, seed=SEED).fit(x, y).loq()
            boot_times.append(time.perf_counter() - t0)

        delta_times: list[float] = []
        for _ in range(n_timing):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                t0 = time.perf_counter()
                PiecewiseCF(n_boot_reps=0, seed=SEED).fit(x, y).loq_delta()
                delta_times.append(time.perf_counter() - t0)

        boot_ms = float(np.median(boot_times) * 1e3)
        delta_ms = float(np.median(delta_times) * 1e3)
        speedup = boot_ms / delta_ms if delta_ms > 0 else float("nan")
        rows.append(
            {
                "peptide": pep,
                "boot_median_ms": boot_ms,
                "delta_median_ms": delta_ms,
                "speedup_boot_over_delta": speedup,
            }
        )
        boot_medians.append(boot_ms)
        delta_medians.append(delta_ms)
        speedups.append(speedup)
        print(
            f"  {pep[:24]:<24}  boot={boot_ms:7.2f} ms  "
            f"delta={delta_ms:7.3f} ms  speedup={speedup:6.1f}×"
        )

    agg = _timing_aggregate(
        boot_medians,
        delta_medians,
        speedups,
        n_boot_reps=n_boot_reps,
        n_timing=n_timing,
        workload=(
            "full_pipeline: fit+bootstrap_loq vs fit+loq_delta on demo 27 peptides; "
            "same data as bench_knot_vs_curvefit full-pipeline timing"
        ),
    )
    print(
        f"\n  Aggregate median: boot={agg['boot_median_ms']:.2f} ms  "
        f"delta={agg['delta_median_ms']:.3f} ms  "
        f"median speedup={agg['median_speedup_boot_over_delta']:.1f}×  "
        f"(ratio of medians={agg['ratio_of_medians_boot_over_delta']:.1f}×)"
    )
    return {"per_peptide": rows, "aggregate": agg}


def exp_loq_only(
    peptides: dict[str, tuple[np.ndarray, np.ndarray]],
    *,
    n_boot_reps: int,
    n_timing: int,
) -> dict[str, Any]:
    """Time LOQ alone on a shared fitted model (fit cost excluded)."""
    print(f"\nLOQ-only timing  (n_boot_reps={n_boot_reps}, n_timing={n_timing})")
    print("  Timing loops are separate (all boot reps, then all delta reps).")

    rows: list[dict[str, Any]] = []
    boot_medians: list[float] = []
    delta_medians: list[float] = []
    speedups: list[float] = []

    for pep, (x, y) in peptides.items():
        cf = PiecewiseCF(n_boot_reps=n_boot_reps, seed=SEED).fit(x, y)

        boot_times: list[float] = []
        for _ in range(n_timing):
            cf._loq_cache.clear()
            cf._boot_summary = None
            cf._x_grid = None
            t0 = time.perf_counter()
            cf.loq()
            boot_times.append(time.perf_counter() - t0)

        delta_times: list[float] = []
        for _ in range(n_timing):
            cf._loq_cache.clear()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                t0 = time.perf_counter()
                cf.loq_delta()
                delta_times.append(time.perf_counter() - t0)

        boot_ms = float(np.median(boot_times) * 1e3)
        delta_ms = float(np.median(delta_times) * 1e3)
        speedup = boot_ms / delta_ms if delta_ms > 0 else float("nan")
        rows.append(
            {
                "peptide": pep,
                "boot_median_ms": boot_ms,
                "delta_median_ms": delta_ms,
                "speedup_boot_over_delta": speedup,
            }
        )
        boot_medians.append(boot_ms)
        delta_medians.append(delta_ms)
        speedups.append(speedup)

    agg = _timing_aggregate(
        boot_medians,
        delta_medians,
        speedups,
        n_boot_reps=n_boot_reps,
        n_timing=n_timing,
        workload="loq_only on shared fit; fit excluded from timer",
    )
    print(
        f"  Aggregate median: boot={agg['boot_median_ms']:.2f} ms  "
        f"delta={agg['delta_median_ms']:.3f} ms  "
        f"median speedup={agg['median_speedup_boot_over_delta']:.1f}×  "
        f"(ratio of medians={agg['ratio_of_medians_boot_over_delta']:.1f}×)"
    )
    return {"per_peptide": rows, "aggregate": agg}


def exp_agreement(
    peptides: dict[str, tuple[np.ndarray, np.ndarray]],
    *,
    n_boot_reps: int,
) -> dict[str, Any]:
    """Compare delta vs bootstrap LOQ; split both-finite pairs by median n_L."""
    print(f"\nAgreement pilot  (n_boot_reps={n_boot_reps}, seed={SEED})")

    rows: list[dict[str, Any]] = []
    for pep, (x, y) in peptides.items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            cf = PiecewiseCF(n_boot_reps=n_boot_reps, seed=SEED).fit(x, y)
            n_lin = _n_lin(cf)
            delta = float(cf.loq_delta())
            boot = float(cf.loq())
        rows.append(
            {
                "peptide": pep,
                "n_lin": n_lin,
                "delta_loq": delta,
                "boot_loq": boot,
                "both_finite": bool(np.isfinite(delta) and np.isfinite(boot)),
                "rel_diff": _rel_diff(delta, boot),
            }
        )

    both = [r for r in rows if r["both_finite"] and r["rel_diff"] is not None]

    def _spread(group: list[dict[str, Any]]) -> dict[str, Any]:
        if not group:
            return {"n": 0, "median_rel_diff": None, "p90_rel_diff": None, "n_lin_range": None}
        diffs = np.asarray([r["rel_diff"] for r in group], dtype=float)
        n_lins = [int(r["n_lin"]) for r in group]
        return {
            "n": int(diffs.size),
            "median_rel_diff": float(np.median(diffs)),
            "p90_rel_diff": float(np.percentile(diffs, 90)),
            "n_lin_range": [min(n_lins), max(n_lins)],
        }

    if both:
        n_lin_cut = float(np.median([r["n_lin"] for r in both]))
        smaller = [r for r in both if r["n_lin"] <= n_lin_cut]
        larger = [r for r in both if r["n_lin"] > n_lin_cut]
    else:
        n_lin_cut = None
        smaller, larger = [], []

    summary = {
        "n_peptides": len(rows),
        "n_both_finite": len(both),
        "n_delta_inf": int(sum(1 for r in rows if not np.isfinite(r["delta_loq"]))),
        "n_boot_inf": int(sum(1 for r in rows if not np.isfinite(r["boot_loq"]))),
        "all_both_finite": _spread(both),
        "smaller_n_lin": _spread(smaller),
        "larger_n_lin": _spread(larger),
        "n_lin_median_cut": n_lin_cut,
        "n_boot_reps": n_boot_reps,
        "seed": SEED,
        "note": (
            "Record-only pilot on demo 27 peptides. Splits both-finite pairs at the "
            "median n_L of that subset. Not a pass/fail gate; not manuscript coverage."
        ),
    }
    print(
        f"  both finite: {summary['n_both_finite']}/{summary['n_peptides']}  "
        f"delta_inf={summary['n_delta_inf']}  boot_inf={summary['n_boot_inf']}"
    )
    if n_lin_cut is not None:
        print(f"  median n_L cut among both-finite: {n_lin_cut:.0f}")
    for label, key in (
        ("all", "all_both_finite"),
        ("smaller/equal n_L", "smaller_n_lin"),
        ("larger n_L", "larger_n_lin"),
    ):
        s = summary[key]
        if s["n"] == 0:
            print(f"  {label}: n=0")
        else:
            print(
                f"  {label}: n={s['n']}  n_L={s['n_lin_range']}  "
                f"median |rel_diff|={s['median_rel_diff']:.3f}  "
                f"p90={s['p90_rel_diff']:.3f}"
            )
    return {"per_peptide": rows, "summary": summary}


def main() -> None:
    args = _parse_args()
    if args.quick:
        args.n_boot_reps = 50
        args.n_timing = 3
        print("Quick mode: n_boot_reps=50, n_timing=3")

    if not DEMO_DATA.exists() or not DEMO_MAP.exists():
        sys.exit(f"Demo data missing: {DEMO_DATA} / {DEMO_MAP}")

    peptides = _load_peptides()
    print(f"Loaded {len(peptides)} peptides from {DEMO_DATA.name}")

    output: dict[str, Any] = {
        "meta": {
            "demo_data": str(DEMO_DATA),
            "demo_map": str(DEMO_MAP),
            "n_peptides": len(peptides),
            "n_boot_reps": args.n_boot_reps,
            "n_timing": args.n_timing,
            "seed": SEED,
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "machine": platform.machine(),
            "processor": platform.processor(),
            "command": (
                f"python benchmarks/bench_delta_vs_boot.py{' --quick' if args.quick else ''}"
            ),
        }
    }

    output["full_pipeline"] = exp_full_pipeline(
        peptides, n_boot_reps=args.n_boot_reps, n_timing=args.n_timing
    )
    output["loq_only"] = exp_loq_only(
        peptides, n_boot_reps=args.n_boot_reps, n_timing=args.n_timing
    )
    output["agreement"] = exp_agreement(peptides, n_boot_reps=args.n_boot_reps)

    save_path = Path(args.save)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_text(json.dumps(_json_safe(output), indent=2))
    print(f"\nWrote {save_path}")

    fp = output["full_pipeline"]["aggregate"]
    lo = output["loq_only"]["aggregate"]
    ag = output["agreement"]["summary"]
    print("\nSummary for plan notes")
    print(
        f"  full_pipeline median: boot={fp['boot_median_ms']:.2f} ms  "
        f"delta={fp['delta_median_ms']:.3f} ms  "
        f"median speedup={fp['median_speedup_boot_over_delta']:.1f}×"
    )
    print(
        f"  loq_only median:      boot={lo['boot_median_ms']:.2f} ms  "
        f"delta={lo['delta_median_ms']:.3f} ms  "
        f"median speedup={lo['median_speedup_boot_over_delta']:.1f}×"
    )
    print(
        f"  agreement both_finite={ag['n_both_finite']}/{ag['n_peptides']}  "
        f"smaller n_L median_rel_diff={ag['smaller_n_lin']['median_rel_diff']}  "
        f"larger n_L median_rel_diff={ag['larger_n_lin']['median_rel_diff']}"
    )


if __name__ == "__main__":
    main()
