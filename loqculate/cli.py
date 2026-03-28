"""Unified CLI entry-point for loqculate.

Usage examples
--------------
# PiecewiseCF (default)
loqculate fit data.tsv conc_map.csv

# PiecewiseWLS (legacy)
loqculate fit data.tsv conc_map.csv --model piecewise_wls

# Empirical CV
loqculate fit data.tsv conc_map.csv --model cv_empirical

# Two models side-by-side
loqculate compare data.tsv conc_map.csv --models piecewise_cf,piecewise_wls
"""

from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from tqdm import tqdm

from loqculate import __version__
from loqculate.config import (
    DEFAULT_BOOT_REPS,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CV_THRESH,
    DEFAULT_MIN_LINEAR_POINTS,
    DEFAULT_MIN_NOISE_POINTS,
    DEFAULT_MODEL,
    DEFAULT_SLIDING_WINDOW,
    DEFAULT_STD_MULT,
)
from loqculate.io import apply_multiplier, read_calibration_data, stream_csv_writer
from loqculate.models import MODEL_REGISTRY

# ---------------------------------------------------------------------------
# Worker functions (module-level so ProcessPoolExecutor can pickle them)
# ---------------------------------------------------------------------------


def _process_chunk(
    x_sorted: np.ndarray,
    y_sorted: np.ndarray,
    peptides_sorted: np.ndarray,
    chunk: list,  # list of (start, end) index pairs
    model_name: str,
    model_kwargs: dict,
    std_mult: float,
    cv_thresh: float,
) -> list[dict]:
    """Process a chunk of peptides and return a list of result dicts."""
    model_class = MODEL_REGISTRY[model_name]
    results = []

    for start, end in chunk:
        pep = str(peptides_sorted[start])
        x = x_sorted[start:end]
        y = y_sorted[start:end]

        row = {"peptide": pep, "LOD": np.inf, "LOQ": np.inf}
        try:
            model = model_class(**model_kwargs)
            model.fit(x, y)
            row["LOD"] = model.lod(std_mult) if model.supports_lod() else np.inf
            row["LOQ"] = model.loq(cv_thresh)
            row.update(model.params_)
        except Exception as exc:
            sys.stderr.write(f"ERROR processing {pep}: {exc}\n")

        results.append(row)

    return results


# ---------------------------------------------------------------------------
# fit sub-command
# ---------------------------------------------------------------------------


def _run_fit(args: argparse.Namespace) -> None:
    data = read_calibration_data(args.curve_data, args.filename_concentration_map, fmt=args.format)

    if args.multiplier_file:
        data = apply_multiplier(data, args.multiplier_file)

    # Sort contiguously by peptide for zero-copy IPC
    sort_idx = np.argsort(data.peptide, kind="stable")
    peps_s = data.peptide[sort_idx]
    x_s = data.concentration[sort_idx]
    y_s = data.area[sort_idx]

    uniq_peps, group_starts = np.unique(peps_s, return_index=True)
    group_ends = np.append(group_starts[1:], len(peps_s))

    cs = args.chunk_size
    chunk_list = [
        list(zip(group_starts[i : i + cs], group_ends[i : i + cs]))
        for i in range(0, len(uniq_peps), cs)
    ]

    piecewise_kwargs: dict = {
        "n_boot_reps": args.bootreps,
        "min_noise_points": args.min_noise_points,
        "min_linear_points": args.min_linear_points,
        "sliding_window": args.sliding_window,
    }
    # Only piecewise models accept the full kwargs set.
    model_kwargs = piecewise_kwargs if args.model.startswith("piecewise") else {}

    output_dir = Path(args.output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    fom_path = output_dir / "figuresofmerit.csv"
    columns = ["peptide", "LOD", "LOQ", "slope", "intercept_linear", "intercept_noise"]

    n_threads = args.n_threads if args.n_threads != -1 else None

    with stream_csv_writer(fom_path, columns) as write_row:
        with ProcessPoolExecutor(max_workers=n_threads) as executor:
            futures = [
                executor.submit(
                    _process_chunk,
                    x_s,
                    y_s,
                    peps_s,
                    chunk,
                    args.model,
                    model_kwargs,
                    args.std_mult,
                    args.cv_thresh,
                )
                for chunk in chunk_list
            ]

            for future in tqdm(
                as_completed(futures), total=len(futures), desc="processing peptides"
            ):
                try:
                    for row in future.result():
                        write_row(row)

                        if args.plot == "y":
                            _plot_one(row, x_s, y_s, peps_s, args)
                except Exception as exc:
                    sys.stderr.write(f"Chunk error: {exc}\n")

    sys.stdout.write(f"Results written to {fom_path}\n")


def _plot_one(row, x_s, y_s, peps_s, args):
    """Fit and plot a single peptide (called from the main process)."""
    from loqculate.models import MODEL_REGISTRY
    from loqculate.plotting import plot_calibration, plot_cv_profile

    pep = row["peptide"]
    mask = peps_s == pep
    x = x_s[mask]
    y = y_s[mask]

    try:
        model = MODEL_REGISTRY[args.model]()
        model.fit(x, y)
        if args.model == "cv_empirical":
            plot_cv_profile(model, pep, output_path=args.output_path)
        else:
            plot_calibration(model, x, y, pep, output_path=args.output_path)
    except Exception as exc:
        sys.stderr.write(f"Plot error for {pep}: {exc}\n")


# ---------------------------------------------------------------------------
# compare sub-command
# ---------------------------------------------------------------------------


def _run_compare(args: argparse.Namespace) -> None:
    from loqculate.models import MODEL_REGISTRY
    from loqculate.plotting import plot_model_comparison

    data = read_calibration_data(args.curve_data, args.filename_concentration_map, fmt=args.format)
    if args.multiplier_file:
        data = apply_multiplier(data, args.multiplier_file)

    model_names = [m.strip() for m in args.models.split(",")]
    for nm in model_names:
        if nm not in MODEL_REGISTRY:
            sys.exit(f"Unknown model '{nm}'.  Available: {list(MODEL_REGISTRY.keys())}")

    output_dir = Path(args.output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    sort_idx = np.argsort(data.peptide, kind="stable")
    peps_s = data.peptide[sort_idx]
    x_s = data.concentration[sort_idx]
    y_s = data.area[sort_idx]

    uniq_peps, group_starts = np.unique(peps_s, return_index=True)
    group_ends = np.append(group_starts[1:], len(peps_s))

    for pep_idx, pep in enumerate(tqdm(uniq_peps, desc="comparing models")):
        s, e = int(group_starts[pep_idx]), int(group_ends[pep_idx])
        x = x_s[s:e]
        y = y_s[s:e]

        fitted_models = {}
        for nm in model_names:
            try:
                m = MODEL_REGISTRY[nm]()
                m.fit(x, y)
                fitted_models[nm] = m
            except Exception as exc:
                sys.stderr.write(f"{nm} failed on {pep}: {exc}\n")

        if args.plot == "y" and fitted_models:
            plot_model_comparison(
                fitted_models, x, y, peptide_name=str(pep), output_path=output_dir
            )

    sys.stdout.write(f"Comparison plots written to {output_dir}\n")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    _default_threads = max(1, (os.cpu_count() or 1) - 2)

    parser = argparse.ArgumentParser(
        prog="loqculate",
        description="loqculate: fit LOD/LOQ calibration curves.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    sub = parser.add_subparsers(dest="command", required=True)

    # ---- shared arguments ------------------------------------------------
    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument("curve_data", type=str, help="Quantitative data file")
    shared.add_argument(
        "filename_concentration_map", type=str, help="CSV mapping filenames to concentrations"
    )
    shared.add_argument("--std_mult", default=DEFAULT_STD_MULT, type=float)
    shared.add_argument("--cv_thresh", default=DEFAULT_CV_THRESH, type=float)
    shared.add_argument("--bootreps", default=DEFAULT_BOOT_REPS, type=int)
    shared.add_argument("--min_noise_points", default=DEFAULT_MIN_NOISE_POINTS, type=int)
    shared.add_argument("--min_linear_points", default=DEFAULT_MIN_LINEAR_POINTS, type=int)
    shared.add_argument("--sliding_window", default=DEFAULT_SLIDING_WINDOW, type=int)
    shared.add_argument("--multiplier_file", type=str, default=None)
    shared.add_argument("--output_path", default=os.getcwd(), type=str)
    shared.add_argument("--plot", default="y", choices=["y", "n"])
    shared.add_argument(
        "--format",
        default="auto",
        choices=[
            "auto",
            "encyclopedia",
            "skyline",
            "diann_report",
            "diann_matrix",
            "spectronaut",
            "generic",
        ],
    )
    shared.add_argument("--chunk_size", default=DEFAULT_CHUNK_SIZE, type=int)
    shared.add_argument(
        "--n_threads", default=_default_threads, type=int, help="Worker processes (-1 = all CPUs)"
    )

    # ---- fit -------------------------------------------------------------
    p_fit = sub.add_parser(
        "fit", parents=[shared], help="Fit a single model and write figuresofmerit.csv"
    )
    p_fit.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        choices=list(MODEL_REGISTRY.keys()),
        help=(
            "Calibration model. piecewise_cf (default): closed-form knot search, "
            "globally optimal, no convergence failures. "
            "piecewise_wls: TRF optimizer, retained for comparison. "
            "cv_empirical: non-parametric CV profile, no LOD."
        ),
    )

    # ---- compare ---------------------------------------------------------
    p_cmp = sub.add_parser(
        "compare", parents=[shared], help="Run multiple models and generate overlay plots"
    )
    p_cmp.add_argument(
        "--models",
        default="piecewise_cf,piecewise_wls",
        help="Comma-separated list of model keys to compare. "
        f"Available: {list(MODEL_REGISTRY.keys())}",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "fit":
        _run_fit(args)
    elif args.command == "compare":
        _run_compare(args)


if __name__ == "__main__":
    main()
