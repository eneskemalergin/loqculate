# Benchmarks

Reproduce from `benchmarks/`. Scratch JSON in a local directory is not a published number. I will not quote a speedup that I cannot name a harness for. Scripts that import `old/calculate-loq.py` need `uv sync --extra legacy` (`lmfit`).

## What each script measures

- `bench_knot_vs_curvefit.py`: PiecewiseCF vs PiecewiseWLS on the 27-peptide demo (`data/demo/one_protein.csv`). Single-fit timing, full pipeline at `n_boot=200` by default, LOD/LOQ agreement, synthetic stress cases.
- `bench_vectorized_boot.py`: vectorized CF bootstrap vs the loop fallback.
- `bench_delta_vs_boot.py`: PiecewiseCF bootstrap LOQ vs `loq_delta()` on that same demo, `B=200`, seed 42, five timing reps by default.
- `bench_window_rules.py`: FDR under null, including window rules. Used with `comparison_report.ipynb`.
- `bench_simulation.py`, `bench_n_concentrations.py`, `bench_n_replicates.py`, `bench_real_data.py`: older WLS/CV/original comparisons.

`comparison_report.ipynb` is a frozen report for OriginalWLS, OriginalCV, PiecewiseWLS, and EmpiricalCV. It does not include PiecewiseCF.

Machine notes in older text assumed an AMD Ryzen 9 3950X. Re-runs on another host will differ.

## Numbers I am willing to attach to a file

v0.3.0 CHANGELOG, from `bench_knot_vs_curvefit.py` / `bench_vectorized_boot.py` on that machine: full pipeline (vectorized CF vs WLS, `n_boot=200`) recorded as 15.8x; single-fit median 2.7x (range 1.3-6.3x on 27 peptides); vectorized vs loop bootstrap 6.8x at 500 replicates. Those are changelog records, not constants of the algorithm.

v0.4.0 CHANGELOG, `bench_delta_vs_boot.py`, demo 27 peptides, $B=200$, seed 42: LOQ-only medians about $32\,\mathrm{ms}$ bootstrap vs $0.16\,\mathrm{ms}$ delta; full `fit`+LOQ about $34\,\mathrm{ms}$ vs $1.0\,\mathrm{ms}$. That is a vectorized grid, not a closed-form confidence interval.

FDR figures that mix concentration density with replicate count belonged to different experiments in `comparison_report.ipynb`. The per-model pages quote the notebook in that context. I am not reprinting a single table that pretends they were one design.

```bash
uv run python benchmarks/bench_knot_vs_curvefit.py
uv run python benchmarks/bench_delta_vs_boot.py
uv run python benchmarks/bench_window_rules.py
```
