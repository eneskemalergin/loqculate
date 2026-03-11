"""Single-peptide calibration curve plot."""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-interactive backend; callers can override before import
import matplotlib.pyplot as plt

from loqculate.models.base import CalibrationModel
from loqculate.config import DEFAULT_CV_THRESH, DEFAULT_STD_MULT

plt.style.use('seaborn-v0_8-whitegrid')


def plot_calibration(
    model: CalibrationModel,
    x: np.ndarray,
    y: np.ndarray,
    peptide_name: str = '',
    output_path: Optional[Union[str, Path]] = None,
    show: bool = False,
    std_mult: float = DEFAULT_STD_MULT,
    cv_thresh: float = DEFAULT_CV_THRESH,
) -> Optional[plt.Figure]:
    """Plot calibration curve: scatter + fit + bootstrap CI + LOD/LOQ markers.

    Parameters
    ----------
    model:
        A **fitted** :class:`~loqculate.models.base.CalibrationModel`.
    x, y:
        Raw observations used to fit the model.
    peptide_name:
        Title for the figure.
    output_path:
        Directory (or full file path) to save the PNG.  If *None*, the figure
        is not saved.
    show:
        If *True*, the figure is returned (and displayed in interactive
        sessions); otherwise ``plt.close(fig)`` is called and *None* returned.
    std_mult:
        LOD standard deviation multiplier (forwarded to ``model.lod()``).
    cv_thresh:
        CV threshold (forwarded to ``model.loq()``).

    Returns
    -------
    matplotlib.figure.Figure or None
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    lod = model.lod(std_mult) if model.supports_lod() else np.inf
    loq = model.loq(cv_thresh)

    fig, axes = plt.subplots(2, 1, figsize=(5, 7))
    fig.suptitle(peptide_name)

    # ---- Top panel: signal vs concentration ----
    ax_top = axes[0]
    ax_top.scatter(x, y, color='steelblue', s=20, label='_nolegend_', zorder=3)

    # Fit line
    x_line = np.linspace(0, float(np.max(x)), 300)
    ax_top.plot(x_line, model.predict(x_line), color='green', linewidth=1.4,
                label='fit')

    # Bootstrap CI band (if the model supports it)
    try:
        lower, upper = model.prediction_interval(x_line)
        ax_top.fill_between(x_line, lower, upper, color='gold', alpha=0.3,
                            label='95 % PI')
    except (NotImplementedError, RuntimeError):
        pass

    if np.isfinite(lod):
        ax_top.axvline(lod, color='m', linewidth=1.2, label=f'LOD = {lod:.3e}')
    if np.isfinite(loq):
        ax_top.axvline(loq, color='c', linewidth=1.2, label=f'LOQ = {loq:.3e}')

    ax_top.set_ylabel('signal')
    ax_top.ticklabel_format(style='sci', axis='both', scilimits=(0, 0))
    ax_top.legend(fontsize=8, loc='upper left')

    # ---- Bottom panel: CV profile ----
    ax_bot = axes[1]

    # If the model cached a bootstrap summary, use it
    if hasattr(model, '_x_grid') and model._x_grid is not None and model._boot_summary is not None:
        xg = model._x_grid
        cv_arr = model._boot_summary['cv']
        ax_bot.plot(xg, cv_arr, color='black', linewidth=1.2, label='bootstrap CV')

    ax_bot.axhline(cv_thresh, color='red', linestyle='--', linewidth=1.0,
                   label=f'{cv_thresh:.0%} CV')
    if np.isfinite(lod):
        ax_bot.axvline(lod, color='m', linewidth=1.2)
    if np.isfinite(loq):
        ax_bot.axvline(loq, color='c', linewidth=1.2)

    ax_bot.set_xlabel('concentration')
    ax_bot.set_ylabel('CV')
    ax_bot.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    ax_bot.set_ylim(bottom=0)

    plt.tight_layout()

    if output_path is not None:
        out = Path(output_path)
        if out.is_dir():
            out = out / f'{peptide_name}.png'
        fig.savefig(out, bbox_inches='tight')

    if show:
        return fig

    plt.close(fig)
    return None
