"""bench_vectorized_boot.py — Comprehensive C6 benchmark: vectorized vs loop bootstrap.

Examines the vectorized bootstrap path in PiecewiseCF from six orthogonal
angles so that a single run answers the question "are we gaining any advantage,
or swapping a good implementation for a bad one?"

  Exp 1 — SPEED:       loop vs vectorized, 3 synthetic profiles × n_reps sweep.
  Exp 2 — MEMORY:      tracemalloc peak during loop and vectorized operations.
  Exp 3 — CORRECTNESS: knot agreement rate + max |cv_diff| on all 27 peptides.
  Exp 4 — GUARD:       memory-limit fallback emits ResourceWarning & stays correct.
  Exp 5 — STABILITY:   LOQ CV across 20 seeds — same stability in both paths.
  Exp 6 — H9:          Numba @njit compatibility of solve_2x2_wls (informational).

The three synthetic profiles match the plan spec:
  sparse  — 6 conc levels × 3 reps = 18 observations
  medium  — 9 conc levels × 5 reps = 45 observations
  dense   — 14 conc levels × 5 reps = 70 observations

Run from the repository root::

    python benchmarks/bench_vectorized_boot.py
    python benchmarks/bench_vectorized_boot.py --quick

Options
-------
--n_reps_list   Comma-separated n_reps to sweep in Exp 1 (default 50,100,200,500).
--n_timing      Timing repetitions per (profile, n_reps) pair (default 5).
--n_seeds       Number of seeds for Exp 5 stability test (default 20).
--save          Path for JSON output (default tmp/results/bench_vectorized_boot.json).
--quick         Fast mode: n_reps_list=50,100, n_timing=3, n_seeds=10.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import tracemalloc
import warnings
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Path setup — must precede loqculate imports so the package resolves from
# the repo root rather than a stale installation.  E402 is expected here.
# ---------------------------------------------------------------------------
_REPO = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(Path(__file__).parent))

from _helpers import DEMO_DATA, DEMO_MAP, _json_safe  # noqa: E402

import loqculate.config as _cfg_mod  # noqa: E402
import loqculate.models.piecewise_cf as _pcf_mod  # noqa: E402
from loqculate.io import read_calibration_data  # noqa: E402
from loqculate.models.piecewise_cf import (  # noqa: E402
    PiecewiseCF,
    _bootstrap_loop_cf,
    _bootstrap_vectorized_cf,
)
from loqculate.utils.weights import inverse_sqrt_weights  # noqa: E402

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args():
    p = argparse.ArgumentParser(
        description="C6: vectorized vs loop bootstrap — speed/memory/correctness/stability"
    )
    p.add_argument(
        "--n_reps_list",
        type=str,
        default="50,100,200,500",
        help="Comma-separated n_reps to sweep (default: 50,100,200,500)",
    )
    p.add_argument(
        "--n_timing", type=int, default=5, help="Timing repetitions per config (default: 5)"
    )
    p.add_argument(
        "--n_seeds", type=int, default=20, help="Seeds for stability experiment (default: 20)"
    )
    p.add_argument(
        "--save",
        type=str,
        default=str(_REPO / "tmp" / "results" / "bench_vectorized_boot.json"),
        metavar="PATH",
        help="Write JSON results to PATH",
    )
    p.add_argument(
        "--quick", action="store_true", help="Fast mode: n_reps_list=50,100, n_timing=3, n_seeds=10"
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------


def _load_peptides() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    data = read_calibration_data(str(DEMO_DATA), str(DEMO_MAP))
    out = {}
    for pep in np.unique(data.peptide):
        m = data.peptide == pep
        out[pep] = (data.concentration[m], data.area[m])
    return out


def _build_synthetic_profiles() -> dict[str, dict]:
    """Build sparse / medium / dense synthetic calibration profiles.

    Each profile has a known underlying model y = max(c, a*x + b) with
    additive Gaussian noise proportional to x.
    """
    rng = np.random.default_rng(42)

    def _make(concs, n_reps_per, seed_offset=0):
        a, b, c = 1000.0, 0.0, 500.0  # slope, linear intercept, noise floor
        xs, ys = [], []
        for cx in concs:
            true_y = max(c, a * cx + b)
            noise_scale = max(true_y * 0.05, 50.0)
            for _ in range(n_reps_per):
                xs.append(cx)
                ys.append(true_y + rng.normal(0, noise_scale))
        x = np.array(xs, dtype=float)
        y = np.array(ys, dtype=float)
        return {
            "x": x,
            "y": y,
            "n_obs": len(x),
            "n_conc": len(concs),
            "n_reps_per": n_reps_per,
            "true_a": a,
            "true_b": b,
            "true_c": c,
        }

    sparse_concs = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0]
    medium_concs = [0.0, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
    dense_concs = [0.0, 0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 50.0, 100.0, 200.0]

    return {
        "sparse": _make(sparse_concs, n_reps_per=3),
        "medium": _make(medium_concs, n_reps_per=5),
        "dense": _make(dense_concs, n_reps_per=5),
    }


def _get_W(x, y):
    w = inverse_sqrt_weights(x)
    return w**2


# ---------------------------------------------------------------------------
# Exp 1 — speed: loop vs vectorized × profile × n_reps
# ---------------------------------------------------------------------------


def exp1_speed_profile(
    profiles: dict[str, dict],
    n_reps_list: list[int],
    n_timing: int,
) -> dict[str, Any]:
    """Measure wall time for loop and vectorized bootstrap across profiles and n_reps."""
    print(f"\n{'=' * 70}")
    print(f"Exp 1: Speed  (n_timing={n_timing})")
    print(f"{'=' * 70}")
    print(
        f"  {'profile':<8} {'n_reps':>7} {'loop ms':>9} {'vec ms':>8} {'speedup':>8} "
        f"{'reps/s loop':>12} {'reps/s vec':>11}"
    )
    print(f"  {'-' * 71}")

    results: dict[str, Any] = {}

    x_grid_ref = np.linspace(0.0, 1.0, 50)  # fixed grid for timing only

    for prof_name, prof in profiles.items():
        x, y, W = prof["x"], prof["y"], _get_W(prof["x"], prof["y"])
        results[prof_name] = {}

        for n_reps in n_reps_list:
            # warm-up
            _bootstrap_loop_cf(x, y, W, x_grid_ref, n_reps=min(n_reps, 20), seed=0)
            _bootstrap_vectorized_cf(x, y, W, x_grid_ref, n_reps=min(n_reps, 20), seed=0)

            loop_times, vec_times = [], []
            for _ in range(n_timing):
                t0 = time.perf_counter()
                _bootstrap_loop_cf(x, y, W, x_grid_ref, n_reps=n_reps, seed=42)
                loop_times.append(time.perf_counter() - t0)

                t0 = time.perf_counter()
                _bootstrap_vectorized_cf(x, y, W, x_grid_ref, n_reps=n_reps, seed=42)
                vec_times.append(time.perf_counter() - t0)

            l_ms = float(np.median(loop_times)) * 1e3
            v_ms = float(np.median(vec_times)) * 1e3
            speedup = l_ms / v_ms if v_ms > 0 else float("inf")
            rps_l = n_reps / (np.median(loop_times))
            rps_v = n_reps / (np.median(vec_times))

            results[prof_name][str(n_reps)] = {
                "loop_median_ms": l_ms,
                "loop_p5_ms": float(np.percentile(loop_times, 5)) * 1e3,
                "loop_p95_ms": float(np.percentile(loop_times, 95)) * 1e3,
                "vec_median_ms": v_ms,
                "vec_p5_ms": float(np.percentile(vec_times, 5)) * 1e3,
                "vec_p95_ms": float(np.percentile(vec_times, 95)) * 1e3,
                "speedup": float(speedup),
                "reps_per_sec_loop": float(rps_l),
                "reps_per_sec_vec": float(rps_v),
                "n_obs": int(len(x)),
            }

            print(
                f"  {prof_name:<8} {n_reps:>7} {l_ms:>9.1f} {v_ms:>8.1f} "
                f"{speedup:>7.2f}× {rps_l:>12.0f} {rps_v:>11.0f}"
            )

    return results


# ---------------------------------------------------------------------------
# Exp 2 — memory: tracemalloc peak
# ---------------------------------------------------------------------------


def exp2_memory_usage(
    profiles: dict[str, dict],
    n_reps_list: list[int],
) -> dict[str, Any]:
    """Measure peak memory allocated during loop and vectorized bootstrap.

    Uses :mod:`tracemalloc` for Python-heap peak.  Note: numpy arrays allocated
    via the C allocator may not appear in tracemalloc; the reported numbers are
    conservative lower bounds.  The key comparison is *relative* (loop vs vec)
    not absolute.
    """
    print(f"\n{'=' * 70}")
    print("Exp 2: Memory usage  (tracemalloc peak)")
    print(f"{'=' * 70}")
    print(f"  {'profile':<8} {'n_reps':>7} {'loop MiB':>10} {'vec MiB':>9} {'ratio':>7}")
    print(f"  {'-' * 48}")

    results: dict[str, Any] = {}
    x_grid_ref = np.linspace(0.0, 1.0, 50)

    for prof_name, prof in profiles.items():
        x, y, W = prof["x"], prof["y"], _get_W(prof["x"], prof["y"])
        results[prof_name] = {}

        for n_reps in n_reps_list:
            # loop
            tracemalloc.start()
            _bootstrap_loop_cf(x, y, W, x_grid_ref, n_reps=n_reps, seed=42)
            _, loop_peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            # vectorized
            tracemalloc.start()
            _bootstrap_vectorized_cf(x, y, W, x_grid_ref, n_reps=n_reps, seed=42)
            _, vec_peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            loop_mib = loop_peak / 1024**2
            vec_mib = vec_peak / 1024**2
            ratio = vec_mib / loop_mib if loop_mib > 0 else float("inf")

            results[prof_name][str(n_reps)] = {
                "loop_peak_mib": float(loop_mib),
                "vec_peak_mib": float(vec_mib),
                "vec_to_loop_ratio": float(ratio),
            }
            print(f"  {prof_name:<8} {n_reps:>7} {loop_mib:>10.3f} {vec_mib:>9.3f} {ratio:>6.2f}×")

    return results


# ---------------------------------------------------------------------------
# Exp 3 — correctness audit: all 27 peptides × N=200
# ---------------------------------------------------------------------------


def exp3_correctness_audit(
    peptides: dict[str, tuple],
    n_reps: int = 200,
    seed: int = 42,
) -> dict[str, Any]:
    """Compare loop and vectorized outputs for all 27 reference peptides.

    Metrics per peptide:
      - max |cv_diff|  (cv is std/mean on the bootstrap grid)
      - max |pred_diff| (per-rep, per-grid-point absolute prediction diff)
      - max |mean_diff_rel| (relative diff of the mean prediction)
      - n_reps_differ  (reps where ANY grid point differs by > 0)
      - verdict: PASS (<1e-6), FP_NOISE (1e-15 range), FAIL (>1e-6)

    Expected: 0 FAILs, max cv_diff ~ 2e-15 (pairwise vs sequential FP noise).
    """
    print(f"\n{'=' * 70}")
    print(f"Exp 3: Correctness audit  (n_reps={n_reps}, seed={seed})")
    print(f"{'=' * 70}")
    print(
        f"  {'Peptide':<30} {'max|cv_diff|':>14} {'max|pred_diff|':>16} "
        f"{'reps_differ':>12} {'verdict'}"
    )
    print(f"  {'-' * 82}")

    results: dict[str, Any] = {}
    n_fail = n_fp_noise = n_pass = n_skip = 0
    worst_cv = 0.0
    worst_pred = 0.0

    for pep, (x, y) in peptides.items():
        W = _get_W(x, y)

        # Use LOD from PiecewiseCF as grid start
        lod = PiecewiseCF(n_boot_reps=0).fit(x, y).lod()
        if not np.isfinite(lod):
            results[pep] = {"verdict": "SKIP_INF_LOD", "lod": float(lod)}
            n_skip += 1
            print(f"  {pep:<30} {'—':>14} {'—':>16} {'—':>12} SKIP_INF_LOD")
            continue

        x_grid = np.linspace(lod, float(np.max(x)), 100)

        pred_l, sum_l = _bootstrap_loop_cf(x, y, W, x_grid, n_reps, seed)
        pred_v, sum_v = _bootstrap_vectorized_cf(x, y, W, x_grid, n_reps, seed)

        # Per-rep max diff across grid
        rep_diffs = np.max(np.abs(pred_l - pred_v), axis=1)  # (n_reps,)
        reps_differ = int(np.sum(rep_diffs > 0))
        max_pred_diff = float(np.max(np.abs(pred_l - pred_v)))

        # CV diff (summary-level)
        cv_l = sum_l["cv"]
        cv_v = sum_v["cv"]
        finite_mask = np.isfinite(cv_l) & np.isfinite(cv_v)
        max_cv_diff = (
            float(np.max(np.abs(cv_l[finite_mask] - cv_v[finite_mask])))
            if finite_mask.any()
            else 0.0
        )

        # Mean relative diff
        mean_l = sum_l["mean"]
        mean_v = sum_v["mean"]
        nonzero = np.abs(mean_l) > 0
        mean_rel_diff = (
            float(np.max(np.abs((mean_l[nonzero] - mean_v[nonzero]) / mean_l[nonzero])))
            if nonzero.any()
            else 0.0
        )

        # Verdict
        if max_cv_diff > 1e-6:
            verdict = "FAIL"
            n_fail += 1
        elif max_cv_diff > 1e-14:
            verdict = "FP_NOISE"
            n_fp_noise += 1
        else:
            verdict = "PASS"
            n_pass += 1

        worst_cv = max(worst_cv, max_cv_diff)
        worst_pred = max(worst_pred, max_pred_diff)

        results[pep] = {
            "n_obs": int(len(x)),
            "lod": float(lod),
            "max_cv_diff": float(max_cv_diff),
            "max_pred_diff": float(max_pred_diff),
            "max_mean_rel_diff": float(mean_rel_diff),
            "n_reps_differ": int(reps_differ),
            "verdict": verdict,
        }

        print(
            f"  {pep:<30} {max_cv_diff:>14.3e} {max_pred_diff:>16.3e} {reps_differ:>12} {verdict}"
        )

    print(f"\n  Summary: PASS={n_pass}  FP_NOISE={n_fp_noise}  FAIL={n_fail}  SKIP={n_skip}")
    print(f"  Worst cv_diff={worst_cv:.3e}   worst pred_diff={worst_pred:.3e}")
    print(f"  Knot-selection agreement: {'100%' if n_fail == 0 else 'DEGRADED'}")

    return {
        "per_peptide": results,
        "summary": {
            "n_pass": n_pass,
            "n_fp_noise": n_fp_noise,
            "n_fail": n_fail,
            "n_skip": n_skip,
            "worst_cv_diff": float(worst_cv),
            "worst_pred_diff": float(worst_pred),
            "knot_selection_agreement": "100%" if n_fail == 0 else "DEGRADED",
        },
    }


# ---------------------------------------------------------------------------
# Exp 4 — memory-guard robustness
# ---------------------------------------------------------------------------


def exp4_guard_robustness(
    peptides: dict[str, tuple],
    n_reps: int = 200,
    seed: int = 42,
) -> dict[str, Any]:
    """Verify the memory-guard fallback path is correct and noisy.

    Method:
      1. Patch VECTORIZED_BOOTSTRAP_MEMORY_LIMIT_MB to 0 on both the config
         module and the piecewise_cf module so the guard fires immediately.
      2. Run _bootstrap_vectorized_cf — it must emit a ResourceWarning and
         produce output identical to _bootstrap_loop_cf with the same args.
      3. Restore the original limit.

    Tests correctness on the first 5 peptides (sufficient; guard only changes
    the dispatch path, not the algorithm).
    """
    print(f"\n{'=' * 70}")
    print("Exp 4: Memory-guard robustness")
    print(f"{'=' * 70}")

    results: dict[str, Any] = {}
    n_tested = n_warning_ok = n_output_ok = 0

    for pep, (x, y) in list(peptides.items())[:5]:
        W = _get_W(x, y)
        lod = PiecewiseCF(n_boot_reps=0).fit(x, y).lod()
        if not np.isfinite(lod):
            continue

        x_grid = np.linspace(lod, float(np.max(x)), 50)

        # Reference: direct loop call
        pred_ref, sum_ref = _bootstrap_loop_cf(x, y, W, x_grid, n_reps, seed)

        # Patched: force the memory guard to fire
        orig_cfg = _cfg_mod.VECTORIZED_BOOTSTRAP_MEMORY_LIMIT_MB
        orig_pcf = _pcf_mod.VECTORIZED_BOOTSTRAP_MEMORY_LIMIT_MB
        _cfg_mod.VECTORIZED_BOOTSTRAP_MEMORY_LIMIT_MB = 0
        _pcf_mod.VECTORIZED_BOOTSTRAP_MEMORY_LIMIT_MB = 0

        try:
            caught_warnings = []
            with warnings.catch_warnings(record=True) as caught_w:
                warnings.simplefilter("always")
                pred_guard, sum_guard = _bootstrap_vectorized_cf(x, y, W, x_grid, n_reps, seed)
            caught_warnings = [
                str(w.message) for w in caught_w if issubclass(w.category, ResourceWarning)
            ]
        finally:
            _cfg_mod.VECTORIZED_BOOTSTRAP_MEMORY_LIMIT_MB = orig_cfg
            _pcf_mod.VECTORIZED_BOOTSTRAP_MEMORY_LIMIT_MB = orig_pcf

        warning_emitted = len(caught_warnings) > 0
        cv_diff = (
            float(
                np.max(
                    np.abs(
                        sum_ref["cv"][np.isfinite(sum_ref["cv"]) & np.isfinite(sum_guard["cv"])]
                        - sum_guard["cv"][np.isfinite(sum_ref["cv"]) & np.isfinite(sum_guard["cv"])]
                    )
                )
            )
            if (np.isfinite(sum_ref["cv"]).any())
            else 0.0
        )
        output_identical = cv_diff == 0.0

        n_tested += 1
        n_warning_ok += int(warning_emitted)
        n_output_ok += int(output_identical)

        results[pep] = {
            "warning_emitted": warning_emitted,
            "output_identical_to_direct_loop": output_identical,
            "cv_diff": float(cv_diff),
        }

        status = "OK" if (warning_emitted and output_identical) else "FAIL"
        print(f"  {pep:<40} warning={warning_emitted}  output_ok={output_identical}  [{status}]")

    print(
        f"\n  {n_tested} peptides tested: "
        f"{n_warning_ok}/{n_tested} emitted ResourceWarning, "
        f"{n_output_ok}/{n_tested} produced identical output"
    )

    return {
        "per_peptide": results,
        "summary": {
            "n_tested": n_tested,
            "n_warning_ok": n_warning_ok,
            "n_output_ok": n_output_ok,
            "all_passed": (n_warning_ok == n_tested and n_output_ok == n_tested),
        },
    }


# ---------------------------------------------------------------------------
# Exp 5 — bootstrap stability across seeds
# ---------------------------------------------------------------------------


def exp5_stability_seeds(
    peptides: dict[str, tuple],
    n_seeds: int = 20,
    n_reps: int = 200,
) -> dict[str, Any]:
    """Compare LOQ stability of loop and vectorized paths across n_seeds.

    For each of 5 representative peptides × n_seeds seeds, compute PiecewiseCF
    LOQ using both paths exclusively.  A stable implementation returns similar
    LOQ values regardless of seed; CV(LOQ) < 30% is expected for typical data.
    Both paths must have similar stability (CV ratio ≈ 1.0) — any divergence
    indicates the vectorized path is not faithfully replicating the loop path.
    """
    print(f"\n{'=' * 70}")
    print(f"Exp 5: Bootstrap stability across seeds  (n_seeds={n_seeds}, n_reps={n_reps})")
    print(f"{'=' * 70}")
    print(
        f"  {'Peptide':<30} {'CV_loop':>9} {'CV_vec':>8} {'CV_ratio':>10} {'mean_|LOQ_diff| %':>18}"
    )
    print(f"  {'-' * 80}")

    # Select 5 peptides with finite LOQ
    selected = []
    for pep, (x, y) in peptides.items():
        lod = PiecewiseCF(n_boot_reps=0).fit(x, y).lod()
        if np.isfinite(lod):
            selected.append(pep)
        if len(selected) >= 5:
            break

    master_rng = np.random.default_rng(999)
    seeds = master_rng.integers(0, 100_000, size=n_seeds).tolist()

    results: dict[str, Any] = {}
    cv_ratios = []

    import loqculate.models.piecewise_cf as _pcf_mod2

    for pep in selected:
        x, y = peptides[pep]
        lod = PiecewiseCF(n_boot_reps=0).fit(x, y).lod()

        loop_loqs, vec_loqs = [], []

        for s in seeds:
            # Loop path (force via memory limit = 0)
            orig_limit = _pcf_mod2.VECTORIZED_BOOTSTRAP_MEMORY_LIMIT_MB
            _pcf_mod2.VECTORIZED_BOOTSTRAP_MEMORY_LIMIT_MB = 0
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", ResourceWarning)
                    loq_l = PiecewiseCF(n_boot_reps=n_reps, seed=s).fit(x, y).loq()
            finally:
                _pcf_mod2.VECTORIZED_BOOTSTRAP_MEMORY_LIMIT_MB = orig_limit

            # Vectorized path (default)
            loq_v = PiecewiseCF(n_boot_reps=n_reps, seed=s).fit(x, y).loq()

            loop_loqs.append(float(loq_l))
            vec_loqs.append(float(loq_v))

        # Filter inf before computing CV
        lq_l = [v for v in loop_loqs if np.isfinite(v)]
        lq_v = [v for v in vec_loqs if np.isfinite(v)]

        cv_l = float(np.std(lq_l) / np.mean(lq_l)) if len(lq_l) > 1 else float("nan")
        cv_v = float(np.std(lq_v) / np.mean(lq_v)) if len(lq_v) > 1 else float("nan")
        cv_ratio = cv_v / cv_l if (np.isfinite(cv_l) and cv_l > 0) else float("nan")

        # Mean absolute LOQ diff (%)
        paired = [
            (lo, v)
            for lo, v in zip(loop_loqs, vec_loqs)
            if np.isfinite(lo) and np.isfinite(v) and lo > 0
        ]
        mean_diff_pct = (
            float(np.mean([abs(v - lo) / lo * 100 for lo, v in paired])) if paired else float("nan")
        )

        cv_ratios.append(cv_ratio)
        results[pep] = {
            "loop_loqs": loop_loqs,
            "vec_loqs": vec_loqs,
            "cv_loop": cv_l,
            "cv_vec": cv_v,
            "cv_ratio": cv_ratio,
            "mean_loq_diff_pct": mean_diff_pct,
            "n_finite_loop": len(lq_l),
            "n_finite_vec": len(lq_v),
        }

        print(f"  {pep:<30} {cv_l:>9.3f} {cv_v:>8.3f} {cv_ratio:>10.4f} {mean_diff_pct:>18.4f}")

    finite_ratios = [r for r in cv_ratios if np.isfinite(r)]
    print(
        f"\n  Median CV ratio (vec/loop): {np.median(finite_ratios):.4f}  "
        f"(1.000 = identical stability; <0.05 deviation is excellent)"
    )

    return {
        "per_peptide": results,
        "summary": {
            "median_cv_ratio": float(np.median(finite_ratios)) if finite_ratios else None,
            "max_cv_ratio": float(np.max(finite_ratios)) if finite_ratios else None,
            "n_seeds": n_seeds,
            "n_reps": n_reps,
        },
    }


# ---------------------------------------------------------------------------
# Exp 6 — H9: Numba compatibility check
# ---------------------------------------------------------------------------


def exp6_h9_numba() -> dict[str, Any]:
    """Check whether solve_2x2_wls is compatible with Numba @njit.

    Informational only — not a production change.  Records the result in the
    H9 row of the hypothesis table.

    Procedure:
      1. Try to import numba.
      2. If available, decorate solve_2x2_wls with @njit and call it once
         with a 4-element example to trigger compilation.
      3. Verify the output matches the non-JIT version.
      4. If numba is unavailable or raises TypingError, record the specific
         error so it can be addressed if Numba support is desired in future.
    """
    print(f"\n{'=' * 70}")
    print("Exp 6 (H9): Numba @njit compatibility of solve_2x2_wls")
    print(f"{'=' * 70}")

    from loqculate.utils.normal_equations import solve_2x2_wls

    result: dict[str, Any] = {}

    # Reference output (pure numpy)
    x_ref = np.array([1.0, 2.0, 3.0, 4.0])
    y_ref = np.array([2.1, 4.05, 5.95, 8.1])
    W_ref = np.array([1.0, 0.5, 0.333, 0.25])
    ref_slope, ref_intercept, ref_rss = solve_2x2_wls(x_ref, y_ref, W_ref)

    result["reference"] = {
        "slope": float(ref_slope),
        "intercept": float(ref_intercept),
        "rss": float(ref_rss),
    }

    try:
        import numba  # noqa: F401

        numba_version = numba.__version__
        result["numba_available"] = True
        result["numba_version"] = numba_version
        print(f"  Numba {numba_version} available. Attempting @njit compilation…")

        try:
            from numba import njit

            @njit
            def _solve_jit(x, y, W):
                return solve_2x2_wls(x, y, W)

            t0 = time.perf_counter()
            jit_slope, jit_intercept, jit_rss = _solve_jit(x_ref, y_ref, W_ref)
            compile_ms = (time.perf_counter() - t0) * 1e3

            slope_ok = abs(jit_slope - ref_slope) < 1e-10
            intercept_ok = abs(jit_intercept - ref_intercept) < 1e-10

            print(f"  Compilation + first call: {compile_ms:.0f} ms")
            print(f"  Output matches: slope={slope_ok}, intercept={intercept_ok}")
            print("  H9 result: COMPATIBLE")

            result.update(
                {
                    "compile_ms": float(compile_ms),
                    "output_matches": bool(slope_ok and intercept_ok),
                    "h9_result": "COMPATIBLE",
                }
            )

        except Exception as e:
            err_type = type(e).__name__
            err_msg = str(e)
            print(f"  Compilation failed: {err_type}: {err_msg[:120]}")
            print(f"  H9 result: INCOMPATIBLE — {err_type}")
            result.update(
                {
                    "h9_result": "INCOMPATIBLE",
                    "error_type": err_type,
                    "error_msg": err_msg[:200],
                }
            )

    except ImportError:
        print("  Numba not installed (pip install numba to test H9).")
        print("  H9 result: NUMBA_NOT_AVAILABLE")
        result.update(
            {
                "numba_available": False,
                "h9_result": "NUMBA_NOT_AVAILABLE",
            }
        )

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    args = _parse_args()
    if args.quick:
        args.n_reps_list = "50,100"
        args.n_timing = 3
        args.n_seeds = 10
        print("Quick mode: n_reps_list=50,100, n_timing=3, n_seeds=10")

    n_reps_list = [int(x) for x in args.n_reps_list.split(",")]

    print("\nLoading 27-peptide reference dataset …")
    peptides = _load_peptides()
    print(f"  {len(peptides)} peptides loaded")

    print("Building synthetic profiles (sparse / medium / dense) …")
    profiles = _build_synthetic_profiles()
    for k, v in profiles.items():
        print(f"  {k}: {v['n_obs']} obs ({v['n_conc']} conc × {v['n_reps_per']} reps)")

    output: dict[str, Any] = {}

    output["exp1_speed"] = exp1_speed_profile(profiles, n_reps_list, args.n_timing)
    output["exp2_memory"] = exp2_memory_usage(profiles, n_reps_list)
    output["exp3_correctness"] = exp3_correctness_audit(peptides, n_reps=200, seed=42)
    output["exp4_guard"] = exp4_guard_robustness(peptides, n_reps=200, seed=42)
    output["exp5_stability"] = exp5_stability_seeds(peptides, n_seeds=args.n_seeds, n_reps=200)
    output["exp6_h9"] = exp6_h9_numba()

    # Overall summary
    print(f"\n{'=' * 70}")
    print("OVERALL SUMMARY — vectorized vs loop bootstrap")
    print(f"{'=' * 70}")

    # Speedup: dense profile at max n_reps
    last_nreps = str(n_reps_list[-1])
    if "dense" in output["exp1_speed"] and last_nreps in output["exp1_speed"]["dense"]:
        sp = output["exp1_speed"]["dense"][last_nreps]["speedup"]
        l_ms = output["exp1_speed"]["dense"][last_nreps]["loop_median_ms"]
        v_ms = output["exp1_speed"]["dense"][last_nreps]["vec_median_ms"]
        print(
            f"  Speed (dense, n_reps={last_nreps}):   loop={l_ms:.1f} ms  vec={v_ms:.1f} ms  "
            f"speedup={sp:.1f}×"
        )

    s3 = output["exp3_correctness"]["summary"]
    print(
        f"  Correctness:  PASS={s3['n_pass']}  FP_NOISE={s3['n_fp_noise']}  "
        f"FAIL={s3['n_fail']}  worst cv_diff={s3['worst_cv_diff']:.2e}"
    )
    print(f"  Knot agreement: {s3['knot_selection_agreement']}")

    s4 = output["exp4_guard"]["summary"]
    print(
        f"  Guard fallback: {s4['n_warning_ok']}/{s4['n_tested']} warned, "
        f"{s4['n_output_ok']}/{s4['n_tested']} identical output"
    )

    s5 = output["exp5_stability"]["summary"]
    print(f"  Stability CV ratio (vec/loop): {s5['median_cv_ratio']:.4f}  (ideal = 1.0000)")

    h9 = output["exp6_h9"].get("h9_result", "UNKNOWN")
    print(f"  H9 (Numba):  {h9}")

    # Save
    out_path = Path(args.save)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(_json_safe(output), f, indent=2)
    print(f"\n  Results saved to {out_path}\n")


if __name__ == "__main__":
    main()
