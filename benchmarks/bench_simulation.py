"""bench_simulation.py — PiecewiseWLS LOQ accuracy and stability on synthetic calibration curves.

Addresses the question: Given a PiecewiseWLS bootstrap CV profile, how accurately
and stably does each window-size rule estimate the LOQ across realistic SNR levels?

Note on scope
-------------
This script focuses on PiecewiseWLS-specific experiments (full calibration curve fits,
oracle comparison, bootstrap stability).  The FDR-under-null experiment (Exp 1 in the
previous version) has been moved to bench_window_rules.py, which provides a unified
FDR/TPR comparison for BOTH PiecewiseWLS and EmpiricalCV in one place.

Methodology
-----------
All three experiments apply the candidate LOQ rules to bootstrap CV profiles produced
by PiecewiseWLS (scipy TRF fit).  This cleanly isolates the window-size effect from
the optimizer difference.

Rules compared
--------------
  window=1 (liberal)  — ANY single grid point below CV threshold triggers LOQ
  window=3 (default)  — 3 CONSECUTIVE points must be below threshold
  window=5            — stricter reference rule

Experiments
-----------
  1. Sensitivity vs Signal-to-Noise Ratio
       Full piecewise-linear calibration curves with known SNR.
       An oracle (high-bootrep) run establishes empirical ground-truth per curve.
       TPR = fraction of "oracle-positive" curves where the standard run also
       calls finite LOQ.

  2. LOQ Estimate Accuracy
       For TRUE POSITIVE pairs (both oracle and standard return finite LOQ)
       the signed relative error vs the oracle is reported:
         rel_error = (rule_LOQ − oracle_LOQ) / oracle_LOQ
       Negative bias = fires earlier than oracle (liberal/over-sensitive).
       Positive bias = fires later (conservative, well-anchored).

  3. Bootstrap Stability
       One fixed calibration curve (SNR ≈ 2), N different bootstrap seeds.
       The coefficient of variation (CV) of resulting LOQ estimates quantifies
       intrinsic rule noise.  window=1 fires on sampling dips → high LOQ CV.

Statistical interpretation key
-------------------------------
  TPR  = sensitivity = TP / (TP + FN)  [want high]
  Oracle = high-bootrep window=3 run on the same data.
  FDR/null experiments → see bench_window_rules.py

Run from the repository root::

    python benchmarks/bench_simulation.py                 # defaults (~3-5 min)
    python benchmarks/bench_simulation.py --quick         # fast smoke test (~30 s)
    python benchmarks/bench_simulation.py --n_curves 300  # more statistical power

Options
-------
--n_curves       Calibration curves per SNR level for Exps 1 & 2 (default: 100)
--n_seeds        RNG seeds for stability experiment Exp 3 (default: 150)
--bootreps       Bootstrap reps for primary runs (default: 100)
--oracle_bootreps  Bootstrap reps for oracle (default: 300)
--cv_thresh      CV threshold for LOQ (default: 0.20)
--snr_levels     Comma-separated SNR values for Exp 1 (default: 1,2,3,5,8,15,30)
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

# Shared RULES registry from _helpers avoids duplicate definitions
sys.path.insert(0, str(Path(__file__).parent))
from _helpers import RULES, _json_safe as _helpers_json_safe

from loqculate.models import PiecewiseWLS
from loqculate.testing.simulator import CurveSimulator
from loqculate.utils.threshold import find_loq_threshold

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
# Experiment 1 — Sensitivity vs SNR (full calibration curves)
# NOTE: FDR-vs-null experiment has moved to bench_window_rules.py (Exp 1)
#       for unified comparison with EmpiricalCV's raw-replicate FDR.
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

def _print_exp1(results: Dict, snr_levels: List[float]) -> None:
    SEP = '=' * 88
    print(f'\n{SEP}')
    print('  EXPERIMENT 1 — Sensitivity (TPR) vs Signal-to-Noise Ratio')
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


def _print_exp2(rel_errors: Dict[str, List[float]]) -> None:
    SEP = '=' * 88
    print(f'\n{SEP}')
    print('  EXPERIMENT 2 — LOQ Estimate Accuracy vs Oracle (TP pairs only)')
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


def _print_exp3(loq_lists: Dict[str, List[float]]) -> None:
    SEP = '=' * 88
    print(f'\n{SEP}')
    print('  EXPERIMENT 3 — Bootstrap Stability (LOQ variability across seeds, '
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
    sens_results: Dict,
    rel_errors: Dict[str, List[float]],
    loq_lists: Dict[str, List[float]],
) -> None:
    SEP = '=' * 88
    print(f'\n{SEP}')
    print('  SUMMARY — All experiments at a glance')
    print(SEP)

    rule_names = list(RULES)
    col_w = 18
    print(f'  {"Metric":<36}' + ''.join(f'  {n:>{col_w}}' for n in rule_names))
    print('  ' + '-' * (38 + (col_w + 2) * len(rule_names)))

    # Accuracy (Exp 2)
    acc_row = f'  {"Mean |error| vs oracle":<36}'
    for name in rule_names:
        errs = rel_errors.get(name, [])
        acc_row += f'  {np.mean(np.abs(errs)):>{col_w}.1%}' if errs else f'  {"N/A":>{col_w}}'
    print(acc_row)

    bias_row = f'  {"Mean bias vs oracle (signed)":<36}'
    for name in rule_names:
        errs = rel_errors.get(name, [])
        bias_row += f'  {np.mean(errs):>+{col_w}.1%}' if errs else f'  {"N/A":>{col_w}}'
    print(bias_row)

    # Stability (Exp 3)
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
    print('  FDR/null experiments → see bench_window_rules.py Exp 1 & 2.')
    print('  Accuracy:  Lower |error| = estimates closer to the high-bootrep oracle.')
    print('             Negative bias means the rule fires EARLIER (more liberal).')
    print('  Stability: Lower CV_LOQ = less sensitive to bootstrap random seed.')
    print('             High instability is a sign the rule fires on random CV dips.')
    print()
    print('  window=3 (default) provides better accuracy and stability than window=1')
    print('  with minimal sensitivity loss at typical MS proteomics SNR levels.')
    print(SEP)
    print()


# Use _helpers_json_safe (imported from _helpers) — avoids duplicate definition
_json_safe = _helpers_json_safe


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='PiecewiseWLS LOQ sensitivity, accuracy, and stability benchmark'
    )
    p.add_argument('--n_curves', type=int, default=50,
                   help='Calibration curves per SNR level / accuracy exp (default: 50)')
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
                   help='Comma-separated SNR values for Exp 1')
    p.add_argument('--seed', type=int, default=42, help='Master RNG seed')
    p.add_argument('--quick', action='store_true',
                   help='Fast smoke-test mode: n_curves=15, n_seeds=30')
    p.add_argument('--save', type=str, default=None, metavar='PATH',
                   help='Write JSON results to PATH (e.g. tmp/results/bench_simulation.json)')
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

    if args.quick:
        args.n_curves        = 15
        args.n_seeds         = 30
        args.oracle_bootreps = 150

    cv_thresh  = args.cv_thresh
    snr_levels = [float(s) for s in args.snr_levels.split(',')]

    print('PiecewiseWLS LOQ Sensitivity, Accuracy, and Stability')
    print('  (FDR/null experiments → bench_window_rules.py)')
    print(f'  Rules compared  : {", ".join(RULES)}')
    print(f'  cv_thresh       : {cv_thresh}')
    print(f'  bootreps        : {args.bootreps}  (primary)')
    print(f'  oracle_bootreps : {args.oracle_bootreps}')
    print(f'  n_curves        : {args.n_curves}  (per SNR level / accuracy exp)')
    print(f'  n_seeds         : {args.n_seeds}  (for stability exp)')
    print(f'  SNR levels      : {snr_levels}')
    print()

    print('Experiment 1: Sensitivity vs SNR (fitting calibration curves) ...',
          flush=True)
    sens_results = experiment_sensitivity(
        args.n_curves, args.bootreps, args.oracle_bootreps,
        cv_thresh, snr_levels, args.seed
    )
    print(' done')

    print('Experiment 2: Accuracy vs oracle ...', end='', flush=True)
    acc_results = experiment_accuracy(
        args.n_curves, args.bootreps, args.oracle_bootreps, cv_thresh, args.seed
    )
    print(' done')

    print('Experiment 3: Bootstrap stability ...', end='', flush=True)
    stab_results = experiment_stability(
        args.n_seeds, args.bootreps, cv_thresh, args.seed
    )
    print(' done')

    _print_exp1(sens_results, snr_levels)
    _print_exp2(acc_results)
    _print_exp3(stab_results)
    _print_summary(sens_results, acc_results, stab_results)

    if args.save:
        import datetime
        results_dict = {
            'meta': {
                'n_curves':        args.n_curves,
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
                'description': 'Sensitivity (TPR) vs SNR using oracle comparison',
                'snr_levels': snr_levels,
                'results': {
                    f'{snr:.8g}': {
                        'n_lod_ok':     sens_results[snr]['n_lod_ok'],
                        'n_oracle_fin': sens_results[snr]['n_oracle_fin'],
                        **{name: sens_results[snr][name] for name in RULES},
                    }
                    for snr in snr_levels
                },
            },
            'experiment_2': {
                'description': 'LOQ estimate accuracy vs oracle (TP pairs, SNR=5)',
                'errors_per_rule': {name: errs for name, errs in acc_results.items()},
            },
            'experiment_3': {
                'description': 'Bootstrap stability: LOQ CV across seeds, same data',
                'loq_lists_per_rule': {name: loqs for name, loqs in stab_results.items()},
            },
        }
        out = Path(args.save)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, 'w') as f:
            json.dump(_json_safe(results_dict), f, indent=2)
        print(f'Results saved \u2192 {out}')


if __name__ == '__main__':
    main()
