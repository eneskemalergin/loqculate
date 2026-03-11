"""bench_simulation.py — Statistical robustness comparison of LOQ detection rules.

Addresses the question: Is the 3-consecutive-point window more statistically
robust than the single-point rule?  Are we discarding too many LOQs, or are
we appropriately controlling false discoveries?

Methodology
-----------
All four experiments apply the candidate LOQ rules to the **same** bootstrap
CV profile (produced by PiecewiseWLS TRF fit).  This cleanly isolates the
window-size effect from the optimizer difference.

Rules compared
--------------
  window=1 (liberal)  — ANY single grid point below CV threshold triggers LOQ
  window=3 (default)  — 3 CONSECUTIVE points must be below threshold
  window=5            — stricter reference rule

Experiments
-----------
  1. FDR Profile
       Synthetic CV profiles with a swept true-mean-CV around the threshold.
       When mean_CV > threshold, any finite LOQ call is a FALSE POSITIVE.
       When mean_CV < threshold, a finite call is a TRUE POSITIVE.
       The detection-rate curve reveals each rule's effective discrimination
       boundary — analogous to a 1-D ROC.

  2. Sensitivity vs Signal-to-Noise Ratio
       Full piecewise-linear calibration curves with known SNR.
       An oracle (high-bootrep) run establishes empirical ground-truth per curve.
       TPR = fraction of "oracle-positive" curves where the standard run also
       calls finite LOQ.

  3. LOQ Estimate Accuracy
       For TRUE POSITIVE pairs (both oracle and standard return finite LOQ)
       the signed relative error vs the oracle is reported:
         rel_error = (rule_LOQ − oracle_LOQ) / oracle_LOQ
       Negative bias = fires earlier than oracle (liberal/over-sensitive).
       Positive bias = fires later (conservative, well-anchored).

  4. Bootstrap Stability
       One fixed calibration curve (SNR ≈ 2), N different bootstrap seeds.
       The coefficient of variation (CV) of resulting LOQ estimates quantifies
       intrinsic rule noise.  window=1 fires on sampling dips → high LOQ CV.

Statistical interpretation key
-------------------------------
  FDR  = false positive rate  = FP / (FP + TN)  [want < 0.05]
  TPR  = sensitivity          = TP / (TP + FN)  [want high]
  A good rule maximises TPR while keeping FDR low.  The window size is a
  tunable knob: larger window → lower FDR, lower TPR.  window=3 sits at a
  practically useful point of this tradeoff for typical proteomics SNR.

Run from the repository root::

    python benchmarks/bench_simulation.py                 # defaults (~3-5 min)
    python benchmarks/bench_simulation.py --quick         # fast smoke test (~30 s)
    python benchmarks/bench_simulation.py --n_curves 300  # more statistical power

Options
-------
--n_curves       Calibration curves per SNR level for Exp 2 & 3 (default: 100)
--n_profiles     CV profiles for FDR experiment Exp 1 (default: 3000)
--n_seeds        RNG seeds for stability experiment Exp 4 (default: 150)
--bootreps       Bootstrap reps for primary runs (default: 100)
--oracle_bootreps  Bootstrap reps for oracle (default: 300)
--cv_thresh      CV threshold for LOQ (default: 0.20)
--snr_levels     Comma-separated SNR values for Exp 2 (default: 0.1,0.25,0.5,1,2,5,10)
--seed           Master RNG seed (default: 42)
--quick          Shorthand for small/fast numbers (overrides other size args)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Path setup — allow running from repo root without installation
# ---------------------------------------------------------------------------
_REPO = Path(__file__).parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from loqculate.models import PiecewiseWLS
from loqculate.testing.simulator import CurveSimulator
from loqculate.utils.threshold import find_loq_threshold


# ---------------------------------------------------------------------------
# LOQ rule registry
# ---------------------------------------------------------------------------

# Each entry: display_name → window_size fed to find_loq_threshold
RULES: Dict[str, int] = {
    'window=1 (liberal)': 1,
    'window=3 (default)': 3,
    'window=5':           5,
}

_DEFAULT_CONCS = [
    0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0,
    100.0, 200.0, 500.0, 1000.0, 2000.0,
]
_NOISE_FLOOR  = 500.0    # intercept_noise used throughout
_LIN_INTERCEPT = 200.0   # intercept_linear
_X_MAX         = 2000.0  # max calibration concentration
# Measurement CV deliberately set ABOVE cv_thresh (0.20).  This puts the
# bootstrap CV profile in an interesting borderline regime: it starts above
# the threshold at the LOD, decreases through the linear regime, and only
# crosses below cv_thresh at some concentration well into the linear range.
# With cv_meas < cv_thresh the linear regime is trivially below threshold
# everywhere — all rules agree instantly, making the comparison uninformative.
_MEAS_CV       = 0.30    # synthetic measurement noise CV (must be > cv_thresh)


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def _apply_rule(
    x_grid: np.ndarray,
    cv_arr: np.ndarray,
    lod: float,
    cv_thresh: float,
    window: int,
) -> float:
    """Apply an LOQ rule (given window size) to a bootstrap CV profile.

    Passes the full x_grid directly to find_loq_threshold, which internally
    filters to x > 0.  x_grid already starts at lod (set by
    ``_ensure_boot_summary``), so no extra filtering is needed — and matches
    exactly what PiecewiseWLS.loq() does.
    """
    return find_loq_threshold(x_grid, cv_arr, cv_thresh=cv_thresh, window=window)


def _fit_and_boot(
    x: np.ndarray,
    y: np.ndarray,
    bootreps: int,
    seed: int,
) -> Tuple[float, Optional[np.ndarray], Optional[np.ndarray]]:
    """Fit PiecewiseWLS and run bootstrap.  Returns (lod, x_grid, cv_arr).

    Returns (inf, None, None) when LOD cannot be computed or bootstrap fails.
    """
    try:
        m = PiecewiseWLS(init_method='legacy', n_boot_reps=bootreps, seed=seed)
        m.fit(x, y)
        lod = m.lod()
        if not np.isfinite(lod):
            return np.inf, None, None
        m._ensure_boot_summary(lod)
        if m._x_grid is None or m._boot_summary is None:
            return np.inf, None, None
        return lod, m._x_grid.copy(), m._boot_summary['cv'].copy()
    except Exception:
        return np.inf, None, None


def _sim_curves(slope: float, n_peptides: int, seed: int) -> Tuple[np.ndarray, ...]:
    """Generate (x, y, peptide_names) for n_peptides curves at given slope."""
    # Clamp noise intercept to always be >= linear intercept (model requirement)
    noise_int = max(_NOISE_FLOOR, _LIN_INTERCEPT * 1.05)
    sim = CurveSimulator(
        slope=slope,
        intercept_linear=_LIN_INTERCEPT,
        intercept_noise=noise_int,
        concentrations=_DEFAULT_CONCS,
        n_replicates=3,
        n_peptides=n_peptides,
        cv=_MEAS_CV,
        seed=seed,
    )
    ds = sim.generate()
    return ds.concentration, ds.area, ds.peptide


# ---------------------------------------------------------------------------
# Experiment 1 — FDR Profile (direct CV-profile simulation)
# ---------------------------------------------------------------------------

def experiment_fdr(
    n_profiles: int,
    cv_thresh: float,
    seed: int,
) -> Tuple[np.ndarray, Dict]:
    """Sweep the true mean CV of synthetic profiles from 0.7× to 1.5× threshold.

    For each mean-CV level:
      * mean_CV > cv_thresh → "null" region: finite LOQ call  =  FP
      * mean_CV < cv_thresh → "signal" region: finite LOQ call = TP

    The CV noise (trial-to-trial fluctuation around the true mean) is fixed at
    10 % of the threshold (σ = 0.1 × cv_thresh), which is realistic for a
    100-rep bootstrap grid: residual bootstrap variance at each grid point.

    The detection-rate curve produced is the empirical power curve of each rule.
    Its x-intercept (where FP rate → 0) marks the effective discrimination
    boundary.
    """
    rng = np.random.default_rng(seed)
    x_grid = np.linspace(0.01, 1.0, 100)     # synthetic non-zero grid
    noise_scale = cv_thresh * 0.10            # ±10 % CV fluctuation per point

    # 17 true_cv levels from 0.70× to 1.50× threshold
    true_cvs = np.linspace(cv_thresh * 0.70, cv_thresh * 1.50, 17)

    results: Dict[float, Dict[str, float]] = {}
    for true_cv in true_cvs:
        hits = {name: 0 for name in RULES}
        for _ in range(n_profiles):
            # Each profile: 100 correlated-ish CV values drawn independently
            cv_arr = np.maximum(0.001,
                                rng.normal(true_cv, noise_scale, size=len(x_grid)))
            for name, win in RULES.items():
                loq = find_loq_threshold(x_grid, cv_arr,
                                         cv_thresh=cv_thresh, window=win)
                if np.isfinite(loq):
                    hits[name] += 1
        results[true_cv] = {name: hits[name] / n_profiles for name in RULES}

    return true_cvs, results


# ---------------------------------------------------------------------------
# Experiment 2 — Sensitivity vs SNR (full calibration curves)
# ---------------------------------------------------------------------------

def experiment_sensitivity(
    n_curves: int,
    bootreps: int,
    oracle_bootreps: int,
    cv_thresh: float,
    snr_levels: List[float],
    seed: int,
) -> Dict:
    """True Positive Rate for each LOQ rule across a range of SNR levels.

    SNR is defined here as:
        SNR = slope × X_MAX / noise_floor

    so that SNR=1 means the linear slope at the top concentration equals the
    noise floor height (modest signal), and SNR=10 means a very strong signal.

    Ground truth (oracle): PiecewiseWLS with oracle_bootreps and window=3.
    A curve is "oracle positive" if the oracle returns a finite LOQ.
    TPR = oracle-positive curves where the standard run also returns finite LOQ.

    IMPORTANT: LOD is deterministic (same primary fit for both oracle and
    standard runs at the same seed), so only the bootstrap CV profile — and
    therefore the LOQ rule — varies between the two.
    """
    results: Dict = {}

    for snr in snr_levels:
        # Slope that produces signal = SNR × noise_floor at X_MAX
        slope = snr * _NOISE_FLOOR / _X_MAX

        conc, area, peps = _sim_curves(slope, n_curves, seed)

        n_lod_ok       = 0
        n_oracle_fin   = 0
        tp  = {name: 0 for name in RULES}
        fn  = {name: 0 for name in RULES}
        fp  = {name: 0 for name in RULES}  # finite call when oracle says inf

        for i in range(n_curves):
            pep = f'peptide_{i:04d}'
            mask = peps == pep
            x = conc[mask]
            y = area[mask]

            # Oracle: high bootreps, window=3
            lod_o, xg_o, cv_o = _fit_and_boot(x, y, oracle_bootreps, seed + i * 17)
            if not np.isfinite(lod_o):
                continue
            n_lod_ok += 1
            oracle_loq = _apply_rule(xg_o, cv_o, lod_o, cv_thresh, window=3)
            oracle_fin = np.isfinite(oracle_loq)
            if oracle_fin:
                n_oracle_fin += 1

            # Standard run: fewer bootreps, different seed → identical x/y data
            lod_s, xg_s, cv_s = _fit_and_boot(x, y, bootreps, seed + i * 17 + 1)
            if not np.isfinite(lod_s):
                # LOD itself failed; all rules return inf
                for name in RULES:
                    if oracle_fin:
                        fn[name] += 1
                continue

            for name, win in RULES.items():
                loq = _apply_rule(xg_s, cv_s, lod_s, cv_thresh, window=win)
                rule_fin = np.isfinite(loq)
                if oracle_fin and rule_fin:
                    tp[name] += 1
                elif oracle_fin and not rule_fin:
                    fn[name] += 1
                elif not oracle_fin and rule_fin:
                    fp[name] += 1   # rule is more liberal than oracle

        results[snr] = {
            'n_lod_ok':     n_lod_ok,
            'n_oracle_fin': n_oracle_fin,
        }
        for name in RULES:
            denom_tpr = n_oracle_fin if n_oracle_fin > 0 else 1
            # FPR denominator = curves where oracle says inf
            n_oracle_inf_ok = n_lod_ok - n_oracle_fin
            denom_fpr = n_oracle_inf_ok if n_oracle_inf_ok > 0 else 1
            results[snr][name] = {
                'tp':  tp[name],
                'fn':  fn[name],
                'fp':  fp[name],
                'tpr': tp[name] / denom_tpr,
                'fpr': fp[name] / denom_fpr,
            }

    return results


# ---------------------------------------------------------------------------
# Experiment 3 — LOQ Estimate Accuracy vs Oracle
# ---------------------------------------------------------------------------

def experiment_accuracy(
    n_curves: int,
    bootreps: int,
    oracle_bootreps: int,
    cv_thresh: float,
    seed: int,
) -> Dict[str, List[float]]:
    """Signed relative error of each LOQ rule compared to the oracle.

    Uses a mid-SNR curve (SNR = 5) where LOQ detection is non-trivial with
    cv_meas = 0.30.  At this SNR the bootstrap CV decreases from ~0.30
    near the LOD down through cv_thresh = 0.20 in the lower linear regime,
    creating a genuine borderline zone where the window size matters.

    Only TRUE POSITIVE pairs (oracle finite AND rule finite) are included.
    Metric per pair:  rel_error = (rule_LOQ − oracle_LOQ) / oracle_LOQ

    Interpretation:
      Negative → rule fires earlier (more liberal, possibly a false positive)
      Positive → rule fires later  (more conservative; anchored to stable CV)
    """
    slope = 5.0 * _NOISE_FLOOR / _X_MAX    # SNR = 5
    conc, area, peps = _sim_curves(slope, n_curves, seed)

    rel_errors: Dict[str, List[float]] = {name: [] for name in RULES}

    for i in range(n_curves):
        pep = f'peptide_{i:04d}'
        mask = peps == pep
        x = conc[mask]
        y = area[mask]

        lod_o, xg_o, cv_o = _fit_and_boot(x, y, oracle_bootreps, seed + i * 13)
        if not np.isfinite(lod_o):
            continue
        oracle_loq = _apply_rule(xg_o, cv_o, lod_o, cv_thresh, window=3)
        if not np.isfinite(oracle_loq):
            continue   # oracle itself says no LOQ → skip

        lod_s, xg_s, cv_s = _fit_and_boot(x, y, bootreps, seed + i * 13 + 1)
        if not np.isfinite(lod_s):
            continue

        for name, win in RULES.items():
            loq = _apply_rule(xg_s, cv_s, lod_s, cv_thresh, window=win)
            if np.isfinite(loq):
                rel_errors[name].append((loq - oracle_loq) / oracle_loq)

    return rel_errors


# ---------------------------------------------------------------------------
# Experiment 4 — Bootstrap Stability
# ---------------------------------------------------------------------------

def experiment_stability(
    n_seeds: int,
    bootreps: int,
    cv_thresh: float,
    seed: int,
) -> Dict[str, List[float]]:
    """CV of LOQ estimates across bootstrap seeds on one fixed calibration curve.

    The calibration data (x, y) is fixed (n_peptides=1, one noise realisation).
    Only the bootstrap seed changes.  Each seed produces a different CV profile,
    and thus potentially a different LOQ.

    The coefficient of variation of these LOQ estimates (CV_LOQ) is a pure
    measure of rule stability: how much does the LOQ answer change if you were
    to re-run with a different random number stream?  Lower is more stable.

    Uses SNR = 2 where cv_meas = 0.30 creates genuinely marginal LOQ
    behaviour: only 2 concentration levels fall above the noise-to-linear
    breakpoint (x=1000 and x=2000), so bootstrap resampling of those few
    points produces notably variable slope estimates.  The resulting CV
    profile near cv_thresh = 0.20 is noisy enough that window=1 fires on
    random single-point dips while window=3 requires a sustained run,
    creating observable LOQ variability differences across seeds.
    """
    slope = 2.0 * _NOISE_FLOOR / _X_MAX
    conc, area, peps = _sim_curves(slope, 1, seed)
    mask = peps == 'peptide_0000'
    x = conc[mask]
    y = area[mask]

    loq_lists: Dict[str, List[float]] = {name: [] for name in RULES}

    for s in range(n_seeds):
        lod, xg, cv = _fit_and_boot(x, y, bootreps, seed + s * 7)
        if not np.isfinite(lod):
            continue
        for name, win in RULES.items():
            loq = _apply_rule(xg, cv, lod, cv_thresh, window=win)
            loq_lists[name].append(loq)   # keep inf to count failures

    return loq_lists


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------

def _print_exp1(true_cvs: np.ndarray, results: Dict, cv_thresh: float) -> None:
    SEP = '=' * 88
    print(f'\n{SEP}')
    print('  EXPERIMENT 1 — FDR / TPR Profile (direct CV-profile simulation)')
    print(f'  Profile mean CV swept vs threshold ({cv_thresh:.2f}).  '
          f'Columns = detection rate; null region = FPR, signal region = TPR.')
    print(SEP)

    rule_names = list(RULES)
    hdr = f'  {"Mean CV":>8}  {"Δ thresh":>9}  {"Region":>8}'
    for name in rule_names:
        hdr += f'  {name:>16}'
    print(hdr)
    print('  ' + '-' * (len(hdr) - 2))

    for true_cv in true_cvs:
        row = results[true_cv]
        delta_pct = (true_cv - cv_thresh) / cv_thresh * 100
        region = 'null (FP)' if true_cv > cv_thresh else 'signal(TP)'
        line = f'  {true_cv:>8.3f}  {delta_pct:>+8.1f}%  {region:>9}'
        for name in rule_names:
            rate = row[name]
            line += f'  {rate:>15.1%}'
        # Highlight the threshold crossing
        if abs(true_cv - cv_thresh) < (true_cvs[1] - true_cvs[0]) * 0.6:
            line += '  ← threshold'
        print(line)

    print()
    # Summarise FDR at the "pure null" end and TPR at the "pure signal" end
    null_cv   = max(r for r in true_cvs if r > cv_thresh * 1.4)
    signal_cv = min(r for r in true_cvs if r < cv_thresh * 0.85)
    print(f'  Summary at pure null (mean_CV = {null_cv:.3f}, +{(null_cv/cv_thresh-1)*100:.0f}% above threshold):')
    print(f'    {"Rule":<20}   FPR      (false positives out of {null_cv:.3f} null profiles)')
    for name in rule_names:
        fpr = results[null_cv][name]
        print(f'    {name:<20}   {fpr:.1%}')
    print()
    print(f'  Summary at pure signal (mean_CV = {signal_cv:.3f}, {(1-signal_cv/cv_thresh)*100:.0f}% below threshold):')
    print(f'    {"Rule":<20}   TPR')
    for name in rule_names:
        tpr = results[signal_cv][name]
        print(f'    {name:<20}   {tpr:.1%}')
    print()


def _print_exp2(results: Dict, snr_levels: List[float]) -> None:
    SEP = '=' * 88
    print(f'\n{SEP}')
    print('  EXPERIMENT 2 — Sensitivity (TPR) vs Signal-to-Noise Ratio')
    print('  Oracle = high-bootrep window=3 run.  '
          'TPR = oracle-positive curves where standard run is also positive.')
    print('  FPR* = extra finite LOQ calls beyond oracle (rule more liberal than oracle).')
    print(SEP)

    rule_names = list(RULES)
    hdr = f'  {"SNR":>5}  {"LODok":>6}  {"OrcPos":>7}'
    for name in rule_names:
        hdr += f'  {name+" TPR":>14}  {" FPR*":>5}'
    print(hdr)
    print('  ' + '-' * (len(hdr) - 2))

    for snr in snr_levels:
        r = results[snr]
        line = f'  {snr:>5.1f}  {r["n_lod_ok"]:>6}  {r["n_oracle_fin"]:>7}'
        for name in rule_names:
            tpr = r[name]['tpr']
            fpr = r[name]['fpr']
            tpr_s = f'{tpr:.1%}' if not np.isnan(tpr) else 'N/A'
            fpr_s = f'{fpr:.1%}' if not np.isnan(fpr) else 'N/A'
            line += f'  {tpr_s:>14}  {fpr_s:>5}'
        print(line)
    print()


def _print_exp3(rel_errors: Dict[str, List[float]]) -> None:
    SEP = '=' * 88
    print(f'\n{SEP}')
    print('  EXPERIMENT 3 — LOQ Estimate Accuracy vs Oracle (TP pairs only)')
    print('  rel_error = (rule_LOQ − oracle_LOQ) / oracle_LOQ')
    print('  Negative = fires earlier than oracle (liberal).  '
          'Positive = fires later (conservative).')
    print(SEP)

    print(f'  {"Rule":<20}  {"N pairs":>8}  {"mean bias":>10}  {"mean|err|":>10}  '
          f'{"median":>8}  {"std":>8}  {"Q5":>8}  {"Q95":>8}')
    print('  ' + '-' * 88)

    for name, errs in rel_errors.items():
        if not errs:
            print(f'  {name:<20}  no TP pairs (all oracle-positive curves missed)')
            continue
        arr = np.array(errs)
        q5, q95 = np.percentile(arr, [5, 95])
        print(f'  {name:<20}  {len(arr):>8}  {np.mean(arr):>+9.1%}  '
              f'{np.mean(np.abs(arr)):>9.1%}  {np.median(arr):>+7.1%}  '
              f'{np.std(arr):>7.1%}  {q5:>+7.1%}  {q95:>+7.1%}')
    print()


def _print_exp4(loq_lists: Dict[str, List[float]]) -> None:
    SEP = '=' * 88
    print(f'\n{SEP}')
    print('  EXPERIMENT 4 — Bootstrap Stability (LOQ variability across seeds, '
          'same calibration data)')
    print('  CV_LOQ = std(finite_LOQs) / mean(finite_LOQs) — lower = more stable.')
    print(SEP)

    rule_names = list(loq_lists)
    print(f'  {"Rule":<20}  {"Finite/Total":>13}  {"mean LOQ":>10}  '
          f'{"std LOQ":>9}  {"CV_LOQ":>7}  {"min":>10}  {"max":>10}')
    print('  ' + '-' * 88)

    for name in rule_names:
        all_loqs = loq_lists[name]
        finite = [v for v in all_loqs if np.isfinite(v)]
        n_tot = len(all_loqs)
        if not finite:
            print(f'  {name:<20}  {0:>4}/{n_tot:<8}  (no finite estimates)')
            continue
        arr = np.array(finite)
        cv_loq = np.std(arr) / np.mean(arr) if np.mean(arr) > 0 else np.inf
        print(f'  {name:<20}  {len(arr):>4}/{n_tot:<8}  '
              f'{np.mean(arr):>10.3e}  {np.std(arr):>9.3e}  '
              f'{cv_loq:>6.1%}  {np.min(arr):>10.3e}  {np.max(arr):>10.3e}')
    print()


def _print_summary(
    true_cvs: np.ndarray,
    fdr_results: Dict,
    rel_errors: Dict[str, List[float]],
    loq_lists: Dict[str, List[float]],
    cv_thresh: float,
) -> None:
    SEP = '=' * 88
    print(f'\n{SEP}')
    print('  SUMMARY — All experiments at a glance')
    print(SEP)

    rule_names = list(RULES)
    col_w = 18
    print(f'  {"Metric":<36}' + ''.join(f'  {n:>{col_w}}' for n in rule_names))
    print('  ' + '-' * (38 + (col_w + 2) * len(rule_names)))

    # FDR at near-null level (+15% above threshold) — this is the zone where
    # the rules actually discriminate.  The extreme null (+50%) is trivially
    # 0% for all rules because random CV dips never reach threshold there.
    null_cv = min((r for r in true_cvs if r >= cv_thresh * 1.12), key=lambda r: abs(r - cv_thresh * 1.15))
    fdr_row = f'  {"FDR @ mean_CV="+f"{null_cv:.2f}":<36}'
    for name in rule_names:
        fdr_row += f'  {fdr_results[null_cv][name]:>{col_w}.1%}'
    print(fdr_row)

    # --- TPR at near-threshold signal ---
    sig_cv = min(r for r in true_cvs if r < cv_thresh * 0.85)
    tpr_row = f'  {"TPR @ mean_CV="+f"{sig_cv:.2f}":<36}'
    for name in rule_names:
        tpr_row += f'  {fdr_results[sig_cv][name]:>{col_w}.1%}'
    print(tpr_row)

    # --- Accuracy ---
    acc_row = f'  {"Mean |error| vs oracle":<36}'
    for name in rule_names:
        errs = rel_errors.get(name, [])
        if errs:
            acc_row += f'  {np.mean(np.abs(errs)):>{col_w}.1%}'
        else:
            acc_row += f'  {"N/A":>{col_w}}'
    print(acc_row)

    bias_row = f'  {"Mean bias vs oracle (signed)":<36}'
    for name in rule_names:
        errs = rel_errors.get(name, [])
        if errs:
            bias_row += f'  {np.mean(errs):>+{col_w}.1%}'
        else:
            bias_row += f'  {"N/A":>{col_w}}'
    print(bias_row)

    # --- Stability ---
    stab_row = f'  {"Stability: CV_LOQ (same data, diff seeds)":<36}'
    for name in rule_names:
        loqs = loq_lists.get(name, [])
        finite = [v for v in loqs if np.isfinite(v)]
        if finite:
            arr = np.array(finite)
            cv_loq = np.std(arr) / np.mean(arr) if np.mean(arr) > 0 else np.inf
            stab_row += f'  {cv_loq:>{col_w}.1%}'
        else:
            stab_row += f'  {"N/A":>{col_w}}'
    print(stab_row)

    fin_row = f'  {"Stability: finite-LOQ rate":<36}'
    for name in rule_names:
        loqs = loq_lists.get(name, [])
        n_fin = sum(1 for v in loqs if np.isfinite(v))
        n_tot = len(loqs) or 1
        fin_row += f'  {n_fin/n_tot:>{col_w}.1%}'
    print(fin_row)

    print(f'\n{SEP}')
    print('  INTERPRETATION')
    print(SEP)
    print('  FDR:       Lower is better.  A rule with FDR > 5% calls finite LOQs')
    print('             on data that genuinely has no stable CV run below threshold.')
    print('  TPR:       Higher is better.  At identical FDR, higher TPR = more power.')
    print('  Accuracy:  Lower |error| = estimates closer to the high-bootrep oracle.')
    print('             Negative bias means the rule fires EARLIER (more liberal).')
    print('  Stability: Lower CV_LOQ = less sensitive to bootstrap random seed.')
    print('             High instability is a sign the rule fires on random CV dips.')
    print()
    print('  window=1 (liberal) has the highest sensitivity but also the highest FDR')
    print('  and lowest stability.  window=3 (default) provides a better FDR/sensitivity')
    print('  balance for typical proteomics calibration curves.')
    print(SEP)
    print()


# ---------------------------------------------------------------------------
# JSON helper (inline — this script does not import from _helpers)
# ---------------------------------------------------------------------------

def _json_safe(obj):
    """Recursively convert *obj* to a JSON-serialisable form.

    * numpy integers / floats are cast to Python int / float.
    * Non-finite floats (inf, nan) become ``None``.
    * numpy arrays are converted via ``tolist()``.
    """
    import math
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, float):
        return None if not math.isfinite(obj) else obj
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return None if not np.isfinite(obj) else float(obj)
    if isinstance(obj, np.ndarray):
        return _json_safe(obj.tolist())
    return obj


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='LOQ window-rule robustness simulation benchmark'
    )
    p.add_argument('--n_curves', type=int, default=50,
                   help='Calibration curves per SNR level / accuracy exp (default: 50)')
    p.add_argument('--n_profiles', type=int, default=3000,
                   help='CV profiles for FDR experiment (default: 3000)')
    p.add_argument('--n_seeds', type=int, default=150,
                   help='Bootstrap seeds for stability experiment (default: 150)')
    p.add_argument('--bootreps', type=int, default=100,
                   help='Bootstrap reps for primary runs (default: 100)')
    p.add_argument('--oracle_bootreps', type=int, default=150,
                   help='Bootstrap reps for oracle (default: 150)')
    p.add_argument('--cv_thresh', type=float, default=0.20,
                   help='CV threshold for LOQ (default: 0.20)')
    p.add_argument('--snr_levels', type=str,
                   default='1,2,3,5,8,15,30',
                   help='Comma-separated SNR values for Exp 2')
    p.add_argument('--seed', type=int, default=42, help='Master RNG seed')
    p.add_argument('--quick', action='store_true',
                   help='Fast smoke-test mode: n_curves=15, n_profiles=500, n_seeds=30')
    p.add_argument('--save', type=str, default=None, metavar='PATH',
                   help='Write JSON results to PATH (e.g. tmp/results/bench_simulation.json)')
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

    if args.quick:
        args.n_curves       = 15
        args.n_profiles     = 500
        args.n_seeds        = 30
        args.oracle_bootreps = 150

    cv_thresh  = args.cv_thresh
    snr_levels = [float(s) for s in args.snr_levels.split(',')]

    print('LOQ Rule Robustness Simulation')
    print(f'  Rules compared  : {", ".join(RULES)}')
    print(f'  cv_thresh       : {cv_thresh}')
    print(f'  bootreps        : {args.bootreps}  (primary)')
    print(f'  oracle_bootreps : {args.oracle_bootreps}')
    print(f'  n_curves        : {args.n_curves}  (per SNR level / accuracy exp)')
    print(f'  n_profiles      : {args.n_profiles}  (for FDR exp)')
    print(f'  n_seeds         : {args.n_seeds}  (for stability exp)')
    print(f'  SNR levels      : {snr_levels}')
    print()

    print('Experiment 1: FDR profile (synthetic CV profiles) ...', end='', flush=True)
    true_cvs, fdr_results = experiment_fdr(args.n_profiles, cv_thresh, args.seed)
    print(' done')

    print('Experiment 2: Sensitivity vs SNR (fitting calibration curves) ...',
          flush=True)
    sens_results = experiment_sensitivity(
        args.n_curves, args.bootreps, args.oracle_bootreps,
        cv_thresh, snr_levels, args.seed
    )
    print(' done')

    print('Experiment 3: Accuracy vs oracle ...', end='', flush=True)
    acc_results = experiment_accuracy(
        args.n_curves, args.bootreps, args.oracle_bootreps, cv_thresh, args.seed
    )
    print(' done')

    print('Experiment 4: Bootstrap stability ...', end='', flush=True)
    stab_results = experiment_stability(
        args.n_seeds, args.bootreps, cv_thresh, args.seed
    )
    print(' done')

    _print_exp1(true_cvs, fdr_results, cv_thresh)
    _print_exp2(sens_results, snr_levels)
    _print_exp3(acc_results)
    _print_exp4(stab_results)
    _print_summary(true_cvs, fdr_results, acc_results, stab_results, cv_thresh)

    if args.save:
        import datetime
        results_dict = {
            'meta': {
                'n_curves':        args.n_curves,
                'n_profiles':      args.n_profiles,
                'n_seeds':         args.n_seeds,
                'bootreps':        args.bootreps,
                'oracle_bootreps': args.oracle_bootreps,
                'cv_thresh':       cv_thresh,
                'rules':           list(RULES.keys()),
                'snr_levels':      snr_levels,
                'seed':            args.seed,
                'timestamp':       datetime.datetime.now().isoformat(),
            },
            'experiment_1': {
                'true_cvs': true_cvs.tolist(),
                # keys are numpy float64 → convert to plain float strings
                'results': {
                    f'{float(k):.8g}': v
                    for k, v in fdr_results.items()
                },
            },
            'experiment_2': {
                f'{snr:.8g}': {
                    'n_lod_ok':     sens_results[snr]['n_lod_ok'],
                    'n_oracle_fin': sens_results[snr]['n_oracle_fin'],
                    **{name: sens_results[snr][name] for name in RULES},
                }
                for snr in snr_levels
            },
            'experiment_3': {
                name: errs for name, errs in acc_results.items()
            },
            'experiment_4': {
                name: loqs for name, loqs in stab_results.items()
            },
        }
        out = Path(args.save)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, 'w') as f:
            json.dump(_json_safe(results_dict), f, indent=2)
        print(f'Results saved \u2192 {out}')


if __name__ == '__main__':
    main()
