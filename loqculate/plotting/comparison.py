"""Multi-model overlay plot for comparing LOD/LOQ across models."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Union

import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from loqculate.config import DEFAULT_CV_THRESH, DEFAULT_STD_MULT
from loqculate.models.base import CalibrationModel

plt.style.use('seaborn-v0_8-whitegrid')

# Colour cycle for model lines / markers
_COLOURS = ['steelblue', 'darkorange', 'forestgreen', 'crimson', 'purple']


def plot_model_comparison(
    models_dict: Dict[str, CalibrationModel],
    x: np.ndarray,
    y: np.ndarray,
    peptide_name: str = '',
    output_path: Optional[Union[str, Path]] = None,
    show: bool = False,
    std_mult: float = DEFAULT_STD_MULT,
    cv_thresh: float = DEFAULT_CV_THRESH,
) -> Optional[plt.Figure]:
    """Overlay multiple fitted models on one peptide's calibration data.

    Parameters
    ----------
    models_dict:
        Mapping of model label → fitted
        :class:`~loqculate.models.base.CalibrationModel` instance.  Example::

            {'Piecewise WLS': pw_model, 'Empirical CV': cv_model}

    x, y:
        Raw observations.
    peptide_name:
        Figure title.
    output_path:
        Directory or full path.  *None* → not saved.
    show:
        Return figure instead of closing.
    std_mult, cv_thresh:
        Forwarded to each model's ``lod()`` / ``loq()``.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    fig, axes = plt.subplots(2, 1, figsize=(7, 8))
    fig.suptitle(f'{peptide_name} — model comparison')

    ax_top = axes[0]
    ax_bot = axes[1]

    # Raw data (shared across models)
    ax_top.scatter(x, y, color='black', s=15, zorder=4, label='data')
    ax_bot.axhline(cv_thresh, color='black', linestyle='--', linewidth=0.8,
                   label=f'{cv_thresh:.0%} CV')

    x_line = np.linspace(0, float(np.max(x)), 300)

    for idx, (label, model) in enumerate(models_dict.items()):
        colour = _COLOURS[idx % len(_COLOURS)]

        # Fit line (only for models that have a meaningful predict)
        try:
            ax_top.plot(x_line, model.predict(x_line), color=colour,
                        linewidth=1.4, linestyle='-', label=label)
        except (NotImplementedError, RuntimeError):
            pass

        lod = model.lod(std_mult) if model.supports_lod() else np.inf
        loq = model.loq(cv_thresh)

        lod_label = f'{label} LOD={lod:.2e}' if np.isfinite(lod) else f'{label} LOD=∞'
        loq_label = f'{label} LOQ={loq:.2e}' if np.isfinite(loq) else f'{label} LOQ=∞'

        if np.isfinite(lod):
            ax_top.axvline(lod, color=colour, linestyle=':', linewidth=1.2,
                           label=lod_label)
            ax_bot.axvline(lod, color=colour, linestyle=':', linewidth=1.0)
        if np.isfinite(loq):
            ax_top.axvline(loq, color=colour, linestyle='--', linewidth=1.2,
                           label=loq_label)
            ax_bot.axvline(loq, color=colour, linestyle='--', linewidth=1.0)

        # CV profile
        if hasattr(model, '_x_grid') and model._x_grid is not None and model._boot_summary is not None:
            ax_bot.plot(model._x_grid, model._boot_summary['cv'], color=colour,
                        linewidth=1.2, label=f'{label} CV')
        elif hasattr(model, 'cv_table_') and model.cv_table_:
            concs = np.array(sorted(c for c in model.cv_table_ if c > 0))
            cvs = np.array([model.cv_table_[c] for c in concs])
            ax_bot.scatter(concs, cvs, color=colour, s=20, label=f'{label} CV')

    ax_top.set_ylabel('signal')
    ax_top.ticklabel_format(style='sci', axis='both', scilimits=(0, 0))
    ax_top.legend(fontsize=7, loc='upper left')

    ax_bot.set_xlabel('concentration')
    ax_bot.set_ylabel('CV')
    ax_bot.set_ylim(bottom=0)
    ax_bot.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    ax_bot.legend(fontsize=7, loc='upper right')

    plt.tight_layout()

    if output_path is not None:
        out = Path(output_path)
        if out.is_dir():
            out = out / f'{peptide_name}_comparison.png'
        fig.savefig(out, bbox_inches='tight')

    if show:
        return fig
    plt.close(fig)
    return None
