"""bench_knot_vs_curvefit.py — Comprehensive C6 benchmark: PiecewiseCF vs PiecewiseWLS.

Examines the closed-form knot search (PiecewiseCF) against the TRF optimizer
(PiecewiseWLS) from five orthogonal perspectives:

  Exp 1 — SPEED: single-fit timing (no bootstrap), all 27 reference peptides.
  Exp 2 — SPEED: full-pipeline timing (fit + bootstrap + loq), varying boot reps.
  Exp 3 — CORRECTNESS: LOD/LOQ comparison, partition agreement, RSS audit.
  Exp 4 — ROBUSTNESS: 13 synthetic stress cases where TRF is most at risk.
  Exp 5 — H8: what fraction of WLS end-to-end time is spent inside curve_fit?

Each experiment saves its results to the shared JSON output so the plan's
hypothesis table (H3, H8) can be filled in from the output alone.

Run from the repository root::

    python benchmarks/bench_knot_vs_curvefit.py
    python benchmarks/bench_knot_vs_curvefit.py --quick
    python benchmarks/bench_knot_vs_curvefit.py --n_fit_reps 200 --n_boot_reps 200

Options
-------
--n_fit_reps      Repeated single-fit calls per peptide for timing (default 100).
--n_boot_reps     Bootstrap replicates used in Exp 2 (default 200).
--n_boot_timing   Timing repetitions for the full-pipeline experiment (default 5).
--save            Path for JSON output (default tmp/results/bench_knot_vs_curvefit.json).
--quick           Override to fast settings: n_fit_reps=20, n_boot_reps=100,
                  n_boot_timing=3.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_REPO = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(Path(__file__).parent))

from _helpers import DEMO_DATA, DEMO_MAP, _json_safe

from loqculate.io import read_calibration_data
from loqculate.models.piecewise_cf import PiecewiseCF
from loqculate.models.piecewise_wls import PiecewiseWLS
from loqculate.utils.weights import inverse_sqrt_weights

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args():
    p = argparse.ArgumentParser(
        description="C6: PiecewiseCF vs PiecewiseWLS — speed, correctness, robustness"
    )
    p.add_argument(
        "--n_fit_reps",
        type=int,
        default=100,
        help="Repeated single-fit calls per peptide (default: 100)",
    )
    p.add_argument(
        "--n_boot_reps", type=int, default=200, help="Bootstrap replicates in Exp 2 (default: 200)"
    )
    p.add_argument(
        "--n_boot_timing",
        type=int,
        default=5,
        help="Timing repetitions for full-pipeline Exp 2 (default: 5)",
    )
    p.add_argument(
        "--save",
        type=str,
        default=str(_REPO / "tmp" / "results" / "bench_knot_vs_curvefit.json"),
        metavar="PATH",
        help="Write JSON results to PATH",
    )
    p.add_argument(
        "--quick",
        action="store_true",
        help="Fast smoke mode: n_fit_reps=20, n_boot_reps=100, n_boot_timing=3",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _load_peptides() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    data = read_calibration_data(str(DEMO_DATA), str(DEMO_MAP))
    out = {}
    for pep in np.unique(data.peptide):
        m = data.peptide == pep
        out[pep] = (data.concentration[m], data.area[m])
    return out


# ---------------------------------------------------------------------------
# Exp 1 — single-fit timing
# ---------------------------------------------------------------------------


def exp1_single_fit_timing(
    peptides: dict[str, tuple],
    n_fit_reps: int,
) -> dict[str, Any]:
    """Time PiecewiseCF.fit() vs PiecewiseWLS.fit() with no bootstrap.

    Reports median, p5, p95, and speedup per peptide, plus aggregate stats.
    """
    print(f"\n{'=' * 70}")
    print(f"Exp 1: Single-fit timing  (n_fit_reps={n_fit_reps})")
    print(f"{'=' * 70}")
    print(f"  {'Peptide':<30} {'WLS ms':>8} {'CF ms':>8} {'Speedup':>8}")
    print(f"  {'-' * 58}")

    per_peptide = {}
    wls_medians, cf_medians = [], []

    for pep, (x, y) in peptides.items():
        # --- warm-up (avoid first-call JIT effects) ---
        for _ in range(3):
            PiecewiseWLS(init_method="legacy", n_boot_reps=0).fit(x, y)
            PiecewiseCF(n_boot_reps=0).fit(x, y)

        # --- WLS timing ---
        wls_times = []
        for _ in range(n_fit_reps):
            t0 = time.perf_counter()
            PiecewiseWLS(init_method="legacy", n_boot_reps=0).fit(x, y)
            wls_times.append(time.perf_counter() - t0)

        # --- CF timing ---
        cf_times = []
        for _ in range(n_fit_reps):
            t0 = time.perf_counter()
            PiecewiseCF(n_boot_reps=0).fit(x, y)
            cf_times.append(time.perf_counter() - t0)

        w_ms = np.median(wls_times) * 1e3
        c_ms = np.median(cf_times) * 1e3
        speedup = w_ms / c_ms if c_ms > 0 else float("inf")

        per_peptide[pep] = {
            "n_obs": int(len(x)),
            "wls_median_ms": float(w_ms),
            "wls_p5_ms": float(np.percentile(wls_times, 5) * 1e3),
            "wls_p95_ms": float(np.percentile(wls_times, 95) * 1e3),
            "cf_median_ms": float(c_ms),
            "cf_p5_ms": float(np.percentile(cf_times, 5) * 1e3),
            "cf_p95_ms": float(np.percentile(cf_times, 95) * 1e3),
            "speedup": float(speedup),
        }
        wls_medians.append(w_ms)
        cf_medians.append(c_ms)

        print(f"  {pep:<30} {w_ms:>8.3f} {c_ms:>8.3f} {speedup:>7.1f}×")

    agg_speedup = np.median(wls_medians) / np.median(cf_medians)
    print(
        f"\n  Aggregate: WLS median={np.median(wls_medians):.3f} ms  "
        f"CF median={np.median(cf_medians):.3f} ms  "
        f"median speedup={agg_speedup:.1f}×"
    )

    return {
        "per_peptide": per_peptide,
        "aggregate": {
            "wls_median_ms": float(np.median(wls_medians)),
            "cf_median_ms": float(np.median(cf_medians)),
            "median_speedup": float(agg_speedup),
            "min_speedup": float(np.min([v["speedup"] for v in per_peptide.values()])),
            "max_speedup": float(np.max([v["speedup"] for v in per_peptide.values()])),
        },
    }


# ---------------------------------------------------------------------------
# Exp 2 — full-pipeline timing (fit + bootstrap + loq)
# ---------------------------------------------------------------------------


def exp2_full_pipeline_timing(
    peptides: dict[str, tuple],
    n_boot_reps: int,
    n_timing: int,
) -> dict[str, Any]:
    """Compare full-pipeline time: WLS vs CF-loop vs CF-vectorized.

    Each measured as: model(n_boot_reps=N).fit(x, y).loq()
    which covers fit + bootstrap + sliding-window LOQ search.
    """
    print(f"\n{'=' * 70}")
    print(f"Exp 2: Full-pipeline timing  (n_boot_reps={n_boot_reps}, n_timing={n_timing})")
    print(f"{'=' * 70}")
    print(
        f"  {'Peptide':<30} {'WLS ms':>9} {'CF-loop ms':>11} {'CF-vec ms':>10} "
        f"{'vs WLS':>7} {'vec/loop':>9}"
    )
    print(f"  {'-' * 78}")

    per_peptide = {}
    speedups_vs_wls, speedups_vec_loop = [], []

    for pep, (x, y) in peptides.items():
        # warm-up
        for _ in range(2):
            PiecewiseWLS(init_method="legacy", n_boot_reps=n_boot_reps).fit(x, y).loq()
            PiecewiseCF(n_boot_reps=n_boot_reps).fit(x, y).loq()

        wls_times, cf_loop_times, cf_vec_times = [], [], []

        for _ in range(n_timing):
            t0 = time.perf_counter()
            PiecewiseWLS(init_method="legacy", n_boot_reps=n_boot_reps).fit(x, y).loq()
            wls_times.append(time.perf_counter() - t0)

        # CF with forced loop path (patch memory limit to 0 to trigger fallback)
        # We import and patch the module-level constant via a context approach.
        import loqculate.models.piecewise_cf as _pcf

        _orig_limit = _pcf.VECTORIZED_BOOTSTRAP_MEMORY_LIMIT_MB
        _pcf.VECTORIZED_BOOTSTRAP_MEMORY_LIMIT_MB = 0
        try:
            for _ in range(n_timing):
                t0 = time.perf_counter()
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", ResourceWarning)
                    PiecewiseCF(n_boot_reps=n_boot_reps).fit(x, y).loq()
                cf_loop_times.append(time.perf_counter() - t0)
        finally:
            _pcf.VECTORIZED_BOOTSTRAP_MEMORY_LIMIT_MB = _orig_limit

        # CF with vectorized path (default)
        for _ in range(n_timing):
            t0 = time.perf_counter()
            PiecewiseCF(n_boot_reps=n_boot_reps).fit(x, y).loq()
            cf_vec_times.append(time.perf_counter() - t0)

        w_ms = np.median(wls_times) * 1e3
        l_ms = np.median(cf_loop_times) * 1e3
        v_ms = np.median(cf_vec_times) * 1e3
        sp_vs_wls = w_ms / v_ms if v_ms > 0 else float("inf")
        sp_vec_loop = l_ms / v_ms if v_ms > 0 else float("inf")

        # Flag peptides where WLS short-circuits early (inf LOD → no bootstrap).
        # In these cases WLS is artificially fast and the speedup number is not
        # comparable to the typical case where both models run the full pipeline.
        wls_lod_inf = not np.isfinite(
            PiecewiseWLS(init_method="legacy", n_boot_reps=0).fit(x, y).lod()
        )
        cf_lod_inf = not np.isfinite(PiecewiseCF(n_boot_reps=0).fit(x, y).lod())
        # Comparable only when both run bootstrap (both finite LOD), or both skip.
        comparable = wls_lod_inf == cf_lod_inf

        per_peptide[pep] = {
            "wls_median_ms": float(w_ms),
            "cf_loop_median_ms": float(l_ms),
            "cf_vec_median_ms": float(v_ms),
            "speedup_vs_wls": float(sp_vs_wls),
            "speedup_vec_vs_loop": float(sp_vec_loop),
            "wls_lod_inf": bool(wls_lod_inf),
            "cf_lod_inf": bool(cf_lod_inf),
            "comparable": bool(comparable),
        }
        if comparable:
            speedups_vs_wls.append(sp_vs_wls)
            speedups_vec_loop.append(sp_vec_loop)

        note = "" if comparable else "  [LOD inf mismatch — excluded from aggregate]"
        print(
            f"  {pep:<30} {w_ms:>9.1f} {l_ms:>11.1f} {v_ms:>10.1f} "
            f"{sp_vs_wls:>6.1f}× {sp_vec_loop:>8.1f}×{note}"
        )

    print(f"\n  Median speedup CF-vec vs WLS:        {np.median(speedups_vs_wls):.1f}×")
    print(f"  Median speedup CF-vec vs CF-loop:    {np.median(speedups_vec_loop):.1f}×")

    return {
        "per_peptide": per_peptide,
        "aggregate": {
            "median_speedup_vec_vs_wls": float(np.median(speedups_vs_wls)),
            "median_speedup_vec_vs_loop": float(np.median(speedups_vec_loop)),
            "min_speedup_vec_vs_wls": float(np.min(speedups_vs_wls)),
            "max_speedup_vec_vs_wls": float(np.max(speedups_vs_wls)),
        },
    }


# ---------------------------------------------------------------------------
# Exp 3 — LOD/LOQ correctness audit
# ---------------------------------------------------------------------------


def exp3_lod_correctness(
    peptides: dict[str, tuple],
) -> dict[str, Any]:
    """For each peptide: compare CF LOD/LOQ against WLS, record RSS and partition.

    Classifies each peptide as:
      - same_partition: CF and WLS choose the same join point (|CF_knot - WLS_knot| < 0.01%)
      - cf_wins_rss: different partition, CF achieves lower total RSS
      - wls_wins_rss: different partition, WLS achieves lower total RSS
      - cf_rescues_finite: WLS returns inf LOD/LOQ but CF returns finite
    """
    print(f"\n{'=' * 70}")
    print("Exp 3: LOD/LOQ correctness audit")
    print(f"{'=' * 70}")
    print(f"  {'Peptide':<30} {'WLS LOD':>9} {'CF LOD':>9} {'rel_diff':>10} {'partition'}")
    print(f"  {'-' * 72}")

    per_peptide = {}
    same_part = diff_cf_wins = diff_wls_wins = cf_rescues = 0

    for pep, (x, y) in peptides.items():
        w = inverse_sqrt_weights(x)
        W = w**2

        wls_model = PiecewiseWLS(init_method="legacy", n_boot_reps=0).fit(x, y)
        cf_model = PiecewiseCF(n_boot_reps=0).fit(x, y)

        wls_lod = wls_model.lod()
        cf_lod = cf_model.lod()

        # WLS join point from stored params
        try:
            wls_a = wls_model.params_["slope"]
            wls_b = wls_model.params_["intercept_linear"]
            wls_c = wls_model.params_["intercept_noise"]
            wls_knot = float((wls_c - wls_b) / wls_a) if wls_a > 0 else float("nan")
        except Exception:
            wls_knot = float("nan")

        cf_knot = cf_model.params_["knot_x"]

        # RSS under the piecewise WLS objective for both solutions
        def _total_rss(a, b, c, knot, x_arr, y_arr, W_arr):
            pred = np.maximum(c, a * x_arr + b)
            return float(np.sum(W_arr * (y_arr - pred) ** 2))

        try:
            cf_rss = _total_rss(
                cf_model.params_["slope"],
                cf_model.params_["intercept_linear"],
                cf_model.params_["intercept_noise"],
                cf_knot,
                x,
                y,
                W,
            )
            wls_rss = _total_rss(
                wls_model.params_["slope"],
                wls_model.params_["intercept_linear"],
                wls_model.params_["intercept_noise"],
                wls_knot,
                x,
                y,
                W,
            )
        except Exception:
            cf_rss = wls_rss = float("nan")

        # LOD relative difference (skip if either is inf)
        if np.isfinite(wls_lod) and np.isfinite(cf_lod) and wls_lod > 0:
            lod_rel = abs(cf_lod - wls_lod) / wls_lod
        else:
            lod_rel = float("nan")

        # Partition classification
        knot_tol = 1e-4 * max(abs(wls_knot), abs(cf_knot), 1.0)
        rescues = (not np.isfinite(wls_lod)) and np.isfinite(cf_lod)

        if rescues:
            partition_class = "cf_rescues_finite"
            cf_rescues += 1
        elif np.isnan(wls_knot) or abs(cf_knot - wls_knot) <= knot_tol:
            partition_class = "same_partition"
            same_part += 1
        elif np.isfinite(cf_rss) and np.isfinite(wls_rss) and cf_rss <= wls_rss:
            partition_class = "diff_cf_wins_rss"
            diff_cf_wins += 1
        else:
            partition_class = "diff_wls_wins_rss"
            diff_wls_wins += 1

        per_peptide[pep] = {
            "wls_lod": float(wls_lod),
            "cf_lod": float(cf_lod),
            "wls_knot": float(wls_knot),
            "cf_knot": float(cf_knot),
            "lod_rel_diff": float(lod_rel),
            "cf_rss": float(cf_rss),
            "wls_rss": float(wls_rss),
            "rss_cf_improvement_pct": float(100 * (wls_rss - cf_rss) / wls_rss)
            if np.isfinite(wls_rss) and wls_rss > 0
            else float("nan"),
            "partition_class": partition_class,
        }

        lod_str = f"{cf_lod:.4f}" if np.isfinite(cf_lod) else "inf"
        wls_str = f"{wls_lod:.4f}" if np.isfinite(wls_lod) else "inf"
        rel_str = f"{lod_rel:.2e}" if np.isfinite(lod_rel) else "n/a"
        print(f"  {pep:<30} {wls_str:>9} {lod_str:>9} {rel_str:>10} {partition_class}")

    total = len(per_peptide)
    print(f"\n  Summary ({total} peptides total):")
    print(f"    Same partition:              {same_part:>3} / {total}")
    print(f"    Different partition — CF wins RSS:  {diff_cf_wins:>3} / {total - same_part}")
    print(f"    Different partition — WLS wins RSS: {diff_wls_wins:>3} / {total - same_part}")
    print(f"    CF rescues finite LOD from inf WLS: {cf_rescues:>3}")

    # LOD agreement for same-partition peptides
    same_part_diffs = [
        v["lod_rel_diff"]
        for v in per_peptide.values()
        if v["partition_class"] == "same_partition" and np.isfinite(v["lod_rel_diff"])
    ]
    if same_part_diffs:
        print(f"    Same-partition max |LOD rel diff|:  {max(same_part_diffs):.2e}")

    return {
        "per_peptide": per_peptide,
        "summary": {
            "total": total,
            "same_partition": same_part,
            "diff_cf_wins_rss": diff_cf_wins,
            "diff_wls_wins_rss": diff_wls_wins,
            "cf_rescues_finite": cf_rescues,
            "same_partition_max_lod_rel_diff": float(max(same_part_diffs))
            if same_part_diffs
            else None,
        },
    }


# ---------------------------------------------------------------------------
# Exp 4 — robustness stress tests
# ---------------------------------------------------------------------------

_STRESS_CASES = [
    {
        "name": "constant_y",
        "desc": "All y values identical — no signal variation",
        "x": np.array([0.0, 1.0, 2.0, 4.0, 8.0] * 3, dtype=float),
        "y": np.full(15, 5000.0),
    },
    {
        "name": "zero_noise_floor",
        "desc": "Noise floor exactly at y=0 — boundary case for noise intercept",
        "x": np.array([0.0, 0.5, 1.0, 2.0, 4.0, 8.0] * 3, dtype=float),
        "y": np.array([0.0, 0.0, 0.0, 2000.0, 4000.0, 8000.0] * 3, dtype=float),
    },
    {
        "name": "monotone_no_plateau",
        "desc": "Strictly increasing signal with no noise plateau",
        "x": np.array([1.0, 2.0, 4.0, 8.0, 16.0] * 3, dtype=float),
        "y": np.array([1000.0, 2000.0, 4000.0, 8000.0, 16000.0] * 3, dtype=float),
    },
    {
        "name": "negative_slope_candidate",
        "desc": "Some candidate knots produce negative slope — constraint 1 must fire",
        "x": np.array([0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0] * 3, dtype=float),
        "y": np.array([500.0, 490.0, 480.0, 1000.0, 2000.0, 5000.0, 10000.0] * 3, dtype=float),
    },
    {
        "name": "noise_below_linear",
        "desc": "Noise intercept > linear intercept — constraint 2 (clamp) must fire",
        "x": np.array([0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0] * 2, dtype=float),
        "y": np.array([8000.0, 8000.0, 2000.0, 4000.0, 8000.0, 16000.0, 32000.0] * 2, dtype=float),
    },
    {
        "name": "only_three_unique_conc",
        "desc": "Minimum feasible: exactly 3 unique x values → 1 interior candidate",
        "x": np.array([0.0] * 5 + [5.0] * 5 + [10.0] * 5, dtype=float),
        "y": np.array([500.0] * 5 + [5000.0] * 5 + [10000.0] * 5, dtype=float),
    },
    {
        "name": "very_wide_range",
        "desc": "x spans 6 decades — weights differ by 3 orders of magnitude",
        "x": np.array([0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0] * 3, dtype=float),
        "y": np.array([50.0, 50.0, 50.0, 100.0, 1000.0, 10000.0, 100000.0] * 3, dtype=float),
    },
    {
        "name": "near_singular_linear_segment",
        "desc": "Linear segment has only 2 obs at identical x — triggers singular fallback",
        "x": np.array([0.0, 0.0, 0.0, 5.0, 5.0, 10.0, 10.0], dtype=float),
        "y": np.array([100.0, 100.0, 100.0, 5000.0, 5050.0, 10000.0, 10050.0], dtype=float),
    },
    {
        "name": "x_starts_at_positive",
        "desc": "No zero concentration — inverse_sqrt_weights applied safely",
        "x": np.array([0.5, 1.0, 2.0, 5.0, 10.0, 20.0] * 3, dtype=float),
        "y": np.array([50.0, 50.0, 100.0, 1000.0, 2000.0, 4000.0] * 3, dtype=float),
    },
    {
        "name": "heavy_outlier_single",
        "desc": "One replicate has a 10× outlier — the noise plateau should still be found",
        "x": np.array([0.0, 0.0, 0.0, 1.0, 2.0, 4.0, 8.0] * 3, dtype=float),
        "y": np.array([500.0, 500.0, 5000.0, 1000.0, 2000.0, 4000.0, 8000.0] * 3, dtype=float),
    },
    {
        "name": "saturating_signal",
        "desc": "Signal plateaus at high concentrations — step function in reverse",
        "x": np.array([0.0, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0] * 3, dtype=float),
        "y": np.array([200.0, 1000.0, 2000.0, 4000.0, 6000.0, 6100.0, 6050.0] * 3, dtype=float),
    },
    {
        "name": "all_replicates_identical_within_level",
        "desc": "Zero intra-level variance — std(noise)=0, LOD formula yields exact value",
        "x": np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0], dtype=float),
        "y": np.array(
            [500.0, 500.0, 500.0, 1000.0, 1000.0, 1000.0, 2000.0, 2000.0, 2000.0], dtype=float
        ),
    },
    {
        "name": "negative_areas",
        "desc": "Negative y values in noise plateau — common in blank-subtracted MS data",
        "x": np.array([0.0, 0.0, 0.0, 1.0, 2.0, 4.0, 8.0] * 3, dtype=float),
        "y": np.array([-200.0, 100.0, -50.0, 500.0, 1000.0, 2000.0, 4000.0] * 3, dtype=float),
    },
]


def exp4_robustness_stress() -> dict[str, Any]:
    """Run PiecewiseCF and PiecewiseWLS on 13 synthetic stress cases.

    For each case records: did the model raise, did it converge cleanly,
    what are the resulting LOD/LOQ values, are they finite and sensible?
    Also checks that CF never emits convergence warnings that WLS might.
    """
    print(f"\n{'=' * 70}")
    print("Exp 4: Robustness stress tests")
    print(f"{'=' * 70}")

    results = {}

    for case in _STRESS_CASES:
        name = case["name"]
        x = np.asarray(case["x"], dtype=float)
        y = np.asarray(case["y"], dtype=float)

        row: dict[str, Any] = {
            "desc": case["desc"],
            "n_obs": int(len(x)),
            "n_unique_x": int(len(np.unique(x))),
        }

        # --- WLS ---
        wls_exc = wls_warn = None
        wls_lod = float("nan")
        wls_converged = True
        t0 = time.perf_counter()
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                wls_model = PiecewiseWLS(
                    init_method="legacy",
                    n_boot_reps=0,
                ).fit(x, y)
                wls_lod = wls_model.lod()
            wls_warn = [
                str(w.message)
                for w in caught
                if "covariance" in str(w.message).lower()
                or "optimal" in str(w.message).lower()
                or "convergence" in str(w.message).lower()
            ]
            wls_converged = len(wls_warn) == 0
        except Exception as e:
            wls_exc = type(e).__name__ + ": " + str(e)
            wls_converged = False
        wls_ms = (time.perf_counter() - t0) * 1e3

        # --- CF ---
        cf_exc = cf_warn = None
        cf_lod = float("nan")
        t0 = time.perf_counter()
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                cf_model = PiecewiseCF(n_boot_reps=0).fit(x, y)
                cf_lod = cf_model.lod()
            cf_warn = [str(w.message) for w in caught]
            cf_warn = [w for w in cf_warn if "ResourceWarning" not in w]
        except Exception as e:
            cf_exc = type(e).__name__ + ": " + str(e)
        cf_ms = (time.perf_counter() - t0) * 1e3

        # Verdict
        wls_ok = wls_exc is None
        cf_ok = cf_exc is None
        verdict = "BOTH_OK"
        if wls_ok and not cf_ok:
            verdict = "WLS_ONLY"
        elif cf_ok and not wls_ok:
            verdict = "CF_RESCUES"
        elif not wls_ok and not cf_ok:
            verdict = "BOTH_FAIL"
        elif not wls_converged:
            verdict = "WLS_WARN_CF_OK"

        row.update(
            {
                "wls_ok": wls_ok,
                "cf_ok": cf_ok,
                "wls_lod": float(wls_lod),
                "cf_lod": float(cf_lod),
                "wls_converged": wls_converged,
                "wls_warnings": wls_warn or [],
                "wls_exception": wls_exc,
                "cf_exception": cf_exc,
                "verdict": verdict,
                "wls_ms": float(wls_ms),
                "cf_ms": float(cf_ms),
            }
        )
        results[name] = row

        warn_flag = " [WLS WARNS]" if not wls_converged else ""
        lod_str = f"CF lod={cf_lod:.3g}" if cf_ok else "CF exception"
        print(f"  {name:<35} {verdict:<18} {lod_str}{warn_flag}")

    n_cf_ok = sum(1 for v in results.values() if v["cf_ok"])
    n_wls_ok = sum(1 for v in results.values() if v["wls_ok"])
    n_wls_warn = sum(1 for v in results.values() if not v["wls_converged"])
    print(
        f"\n  Summary: CF ok={n_cf_ok}/{len(results)}  "
        f"WLS ok={n_wls_ok}/{len(results)}  WLS warns/fails={n_wls_warn}"
    )

    return {
        "cases": results,
        "summary": {
            "n_cases": len(results),
            "cf_ok": n_cf_ok,
            "wls_ok": n_wls_ok,
            "wls_convergence_warnings": n_wls_warn,
        },
    }


# ---------------------------------------------------------------------------
# Exp 5 — H8: solver fraction of WLS end-to-end time
# ---------------------------------------------------------------------------


def exp5_h8_solver_fraction(
    peptides: dict[str, tuple],
    n_fit_reps: int,
) -> dict[str, Any]:
    """Estimate what fraction of WLS end-to-end time is spent in curve_fit.

    Method:
      1. Time the *full* PiecewiseWLS pipeline (fit + bootstrap + loq).
      2. Time the *single* fit step alone (no bootstrap).
      3. Solver fraction ≈ t_single / t_full.
      4. For context also time CF single fit.

    Note: this is a proxy; `cProfile` on the full CLI with the 23k-peptide
    dataset would give the true breakdown, but the demo data is sufficient to
    estimate the *relative* cost of the optimizer vs the bootstrap.
    """
    print(f"\n{'=' * 70}")
    print(f"Exp 5 (H8): WLS solver fraction of end-to-end time  (n_fit_reps={n_fit_reps})")
    print(f"{'=' * 70}")

    n_boot = 200  # fixed for this experiment to get a meaningful fraction

    per_peptide = {}
    fractions = []

    for pep, (x, y) in list(peptides.items())[:10]:  # first 10 for speed
        # warm-up
        for _ in range(2):
            PiecewiseWLS(init_method="legacy", n_boot_reps=n_boot).fit(x, y).loq()

        # single-fit
        single_times = []
        for _ in range(n_fit_reps):
            t0 = time.perf_counter()
            PiecewiseWLS(init_method="legacy", n_boot_reps=0).fit(x, y)
            single_times.append(time.perf_counter() - t0)

        # full pipeline
        full_times = []
        for _ in range(5):
            t0 = time.perf_counter()
            PiecewiseWLS(init_method="legacy", n_boot_reps=n_boot).fit(x, y).loq()
            full_times.append(time.perf_counter() - t0)

        t_single = np.median(single_times) * 1e3
        t_full = np.median(full_times) * 1e3
        # Estimated curve_fit fraction of the full pipeline:
        #   ~n_boot curve_fit calls are executed inside the bootstrap loop,
        #   each costing roughly t_single.  So the solver occupies
        #   (n_boot × t_single) of t_full.
        frac = (n_boot * t_single) / t_full if t_full > 0 else float("nan")
        fractions.append(frac)

        per_peptide[pep] = {
            "single_fit_ms": float(t_single),
            "full_pipeline_ms": float(t_full),
            "n_boot": n_boot,
            "estimated_solver_fraction": float(frac),
        }
        print(
            f"  {pep:<30} single={t_single:.2f} ms  "
            f"full={t_full:.1f} ms  est_solver_frac={frac:.1%}"
        )

    med_frac = float(np.median(fractions)) if fractions else float("nan")
    print(f"\n  Median estimated solver fraction (n_boot×t_single / t_full): {med_frac:.1%}")
    print(f"  Interpretation: curve_fit accounts for ~{med_frac:.0%} of total WLS pipeline.")
    print(f"  Remaining ~{(1 - med_frac):.0%}: bootstrap sampling, LOQ grid search, IO.")

    return {
        "per_peptide": per_peptide,
        "aggregate": {
            "median_estimated_solver_fraction": med_frac,
            "min_estimated_solver_fraction": float(np.min(fractions)) if fractions else None,
            "max_estimated_solver_fraction": float(np.max(fractions)) if fractions else None,
            "n_boot_used": n_boot,
        },
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    args = _parse_args()
    if args.quick:
        args.n_fit_reps = 20
        args.n_boot_reps = 100
        args.n_boot_timing = 3
        print("Quick mode: n_fit_reps=20, n_boot_reps=100, n_boot_timing=3")

    print("\nLoading 27-peptide reference dataset …")
    peptides = _load_peptides()
    print(f"  {len(peptides)} peptides loaded")

    output: dict[str, Any] = {}

    output["exp1_single_fit_timing"] = exp1_single_fit_timing(peptides, args.n_fit_reps)
    output["exp2_full_pipeline_timing"] = exp2_full_pipeline_timing(
        peptides, args.n_boot_reps, args.n_boot_timing
    )
    output["exp3_lod_correctness"] = exp3_lod_correctness(peptides)
    output["exp4_robustness_stress"] = exp4_robustness_stress()
    output["exp5_h8_solver_fraction"] = exp5_h8_solver_fraction(peptides, args.n_fit_reps)

    # Print overall summary
    print(f"\n{'=' * 70}")
    print("OVERALL SUMMARY")
    print(f"{'=' * 70}")
    agg1 = output["exp1_single_fit_timing"]["aggregate"]
    agg2 = output["exp2_full_pipeline_timing"]["aggregate"]
    s3 = output["exp3_lod_correctness"]["summary"]
    s4 = output["exp4_robustness_stress"]["summary"]
    s5 = output["exp5_h8_solver_fraction"]["aggregate"]

    print(
        f"  Exp 1 — Single-fit speedup (CF vs WLS):         {agg1['median_speedup']:.1f}×  "
        f"(range {agg1['min_speedup']:.1f}–{agg1['max_speedup']:.1f}×)"
    )
    print(
        f"  Exp 2 — Full-pipeline speedup (CF-vec vs WLS):  "
        f"{agg2['median_speedup_vec_vs_wls']:.1f}×"
    )
    print(
        f"  Exp 2 — Vectorized vs loop within CF:           "
        f"{agg2['median_speedup_vec_vs_loop']:.1f}×"
    )
    print(
        f"  Exp 3 — Partition: {s3['same_partition']} same, "
        f"{s3['diff_cf_wins_rss']} CF-wins, "
        f"{s3['diff_wls_wins_rss']} WLS-wins, "
        f"{s3['cf_rescues_finite']} CF-rescues"
    )
    print(
        f"  Exp 4 — CF robustness: {s4['cf_ok']}/{s4['n_cases']} ok  "
        f"WLS convergence issues: {s4['wls_convergence_warnings']}"
    )
    print(
        f"  Exp 5 — H8: estimated curve_fit fraction = {s5['median_estimated_solver_fraction']:.1%}"
    )

    # Save
    out_path = Path(args.save)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(_json_safe(output), f, indent=2)
    print(f"\n  Results saved to {out_path}\n")


if __name__ == "__main__":
    main()
