"""CLI contract tests for --fast and LOQ routing."""

from __future__ import annotations

import io
from contextlib import redirect_stdout
from types import SimpleNamespace

import numpy as np
import pytest

from loqculate import PiecewiseCF
from loqculate.cli import _loq_for_cli, _process_chunk, _run_fit, build_parser


def _finite_delta_arrays() -> tuple[np.ndarray, np.ndarray]:
    """Return x, y for a curve with finite analytical delta LOQ."""
    noise_x = np.repeat([0.1, 1.0], 12)
    lin_x = 10.0 + 10.0 * np.arange(6, dtype=float)
    x = np.concatenate([noise_x, lin_x])
    y = np.where(x <= 1.0, 5.0, 5.0 + (x - 1.0))
    return x, y


def _delta_inf_bootstrap_finite_arrays() -> tuple[np.ndarray, np.ndarray]:
    """Return x, y with n_L < 3 (delta LOQ inf) but finite bootstrap LOQ."""
    x = np.concatenate([np.repeat([1.0, 2.0, 3.0], 8), np.array([4.0, 5.0])])
    y = np.where(x <= 3.0, 10.0, 10.0 + 5.0 * (x - 3.0))
    y = y + np.random.default_rng(1).normal(0.0, 0.3, size=y.shape)
    return x, y


def test_parser_wires_fast_flag() -> None:
    """fit parses --fast as store_true; default is False."""
    parser = build_parser()
    assert parser.parse_args(["fit", "curve.tsv", "map.csv"]).fast is False
    assert parser.parse_args(["fit", "curve.tsv", "map.csv", "--fast"]).fast is True


def test_run_fit_forwards_fast_to_worker(tmp_path, monkeypatch) -> None:
    """_run_fit must pass args.fast into _process_chunk (not drop the flag)."""
    import loqculate.cli as cli
    from loqculate.io.readers import CalibrationData

    data = CalibrationData(
        peptide=np.array(["P", "P"]),
        concentration=np.array([1.0, 10.0]),
        area=np.array([5.0, 50.0]),
    )
    monkeypatch.setattr(cli, "read_calibration_data", lambda *a, **k: data)

    captured: dict = {}

    class _FakeFuture:
        def result(self):
            return [
                {
                    "peptide": "P",
                    "LOD": 1.0,
                    "LOQ": 2.0,
                    "slope": 1.0,
                    "intercept_linear": 0.0,
                    "intercept_noise": 0.0,
                }
            ]

    class _FakePool:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def submit(self, fn, *args):
            captured["fn"] = fn
            captured["fast"] = args[8]
            captured["model"] = args[4]
            return _FakeFuture()

    monkeypatch.setattr(cli, "ProcessPoolExecutor", _FakePool)
    monkeypatch.setattr(cli, "as_completed", lambda futures: list(futures))

    args = SimpleNamespace(
        fast=True,
        model="piecewise_cf",
        curve_data="curve.tsv",
        filename_concentration_map="map.csv",
        format="auto",
        multiplier_file=None,
        chunk_size=10,
        bootreps=0,
        min_noise_points=3,
        min_linear_points=3,
        sliding_window=3,
        output_path=str(tmp_path),
        n_threads=1,
        std_mult=2.0,
        cv_thresh=0.2,
        plot="n",
    )
    _run_fit(args)

    assert captured["fn"] is cli._process_chunk
    assert captured["model"] == "piecewise_cf"
    assert captured["fast"] is True

    args.fast = False
    _run_fit(args)
    assert captured["fast"] is False


def test_fast_refused_for_non_cf_model() -> None:
    """--fast with a non-CF model exits with a clear error."""
    args = SimpleNamespace(
        fast=True,
        model="piecewise_wls",
        curve_data="unused",
        filename_concentration_map="unused",
    )
    with pytest.raises(SystemExit, match="piecewise_cf"):
        _run_fit(args)


def test_process_chunk_fast_refuses_non_cf() -> None:
    """Worker rejects --fast for non-CF models."""
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([1.0, 2.0, 3.0])
    peps = np.array(["A", "A", "A"])
    with pytest.raises(ValueError, match="piecewise_cf"):
        _process_chunk(x, y, peps, [(0, 3)], "piecewise_wls", {}, 2.0, 0.2, fast=True)


def test_fast_uses_finite_delta_loq() -> None:
    """--fast with finite delta LOQ uses the analytical value (deterministic)."""
    x, y = _finite_delta_arrays()
    peps = np.array(["P"] * len(x))
    kwargs = {"n_boot_reps": 0, "seed": 42}

    rows_a = _process_chunk(x, y, peps, [(0, len(x))], "piecewise_cf", kwargs, 2.0, 0.2, True)
    rows_b = _process_chunk(x, y, peps, [(0, len(x))], "piecewise_cf", kwargs, 2.0, 0.2, True)

    cf = PiecewiseCF(**kwargs).fit(x, y)
    delta = cf.loq(method="delta")
    assert np.isfinite(delta)
    assert rows_a[0]["LOQ"] == delta
    assert rows_b[0]["LOQ"] == delta
    assert rows_a[0]["LOQ"] == rows_b[0]["LOQ"]


def test_without_fast_uses_bootstrap_loq() -> None:
    """Without --fast, LOQ matches default bootstrap loq(), not delta."""
    x, y = _finite_delta_arrays()
    peps = np.array(["P"] * len(x))
    kwargs = {"n_boot_reps": 0, "seed": 42}

    rows = _process_chunk(x, y, peps, [(0, len(x))], "piecewise_cf", kwargs, 2.0, 0.2, False)
    cf = PiecewiseCF(**kwargs).fit(x, y)
    delta = cf.loq(method="delta")
    boot = cf.loq()
    assert np.isfinite(delta), "pre-condition: delta LOQ must be finite on this curve"
    assert np.isinf(boot), "pre-condition: n_boot_reps=0 bootstrap LOQ must be inf"
    assert rows[0]["LOQ"] == boot
    assert rows[0]["LOQ"] != delta


def test_fast_falls_back_to_bootstrap_when_delta_inf() -> None:
    """--fast falls back to bootstrap when delta LOQ is infinite (H7)."""
    x, y = _delta_inf_bootstrap_finite_arrays()
    peps = np.array(["P"] * len(x))
    kwargs = {"n_boot_reps": 100, "seed": 42}

    cf = PiecewiseCF(**kwargs).fit(x, y)
    assert int(np.sum(cf.x_ > cf.params_["knot_x"])) < 3
    assert np.isinf(cf.loq(method="delta"))
    assert np.isinf(cf.loq_delta())
    boot = cf.loq()
    assert np.isfinite(boot), "pre-condition: bootstrap LOQ must be finite for this curve"

    rows = _process_chunk(x, y, peps, [(0, len(x))], "piecewise_cf", kwargs, 2.0, 0.2, True)
    assert rows[0]["LOQ"] == boot


def test_api_delta_does_not_fall_back() -> None:
    """Pure API method='delta' stays infinite when delta is unavailable."""
    x, y = _delta_inf_bootstrap_finite_arrays()
    cf = PiecewiseCF(n_boot_reps=100, seed=42).fit(x, y)
    assert np.isinf(cf.loq(method="delta"))
    assert np.isfinite(cf.loq())


def test_loq_for_cli_fast_fallback_helper() -> None:
    """_loq_for_cli returns bootstrap after infinite delta under fast=True."""
    x, y = _delta_inf_bootstrap_finite_arrays()
    cf = PiecewiseCF(n_boot_reps=100, seed=42).fit(x, y)
    assert np.isinf(cf.loq(method="delta"))
    got = _loq_for_cli(cf, cv_thresh=0.2, fast=True)
    assert np.isfinite(got)
    assert got == cf.loq(method="bootstrap")


def test_fit_help_mentions_fast_without_fda_ema() -> None:
    """fit --help documents --fast and does not claim FDA/EMA PI mandate."""
    parser = build_parser()
    buf = io.StringIO()
    with pytest.raises(SystemExit), redirect_stdout(buf):
        parser.parse_args(["fit", "--help"])
    help_text = buf.getvalue()
    assert "--fast" in help_text
    assert "delta" in help_text.lower()
    assert "FDA" not in help_text
    assert "EMA" not in help_text
