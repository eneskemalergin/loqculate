"""CV-only plot for EmpiricalCV (no regression fit overlay)."""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from loqculate.config import DEFAULT_CV_THRESH
from loqculate.models.cv_empirical import EmpiricalCV

plt.style.use('seaborn-v0_8-whitegrid')


def plot_cv_profile(
    model: EmpiricalCV,
    peptide_name: str = '',
    output_path: Optional[Union[str, Path]] = None,
    show: bool = False,
    cv_thresh: float = DEFAULT_CV_THRESH,
) -> Optional[plt.Figure]:
    """Plot CV vs concentration for an :class:`~loqculate.models.cv_empirical.EmpiricalCV` model.

    Parameters
    ----------
    model:
        A **fitted** :class:`~loqculate.models.cv_empirical.EmpiricalCV`.
    peptide_name:
        Figure title.
    output_path:
        Directory or full file path for PNG output.  None → not saved.
    show:
        Return the figure instead of closing it.
    cv_thresh:
        Threshold reference line.
    """
    model._check_is_fitted()
    loq = model.loq(cv_thresh)

    concs = np.array(sorted(model.cv_table_.keys()))
    cvs = np.array([model.cv_table_[c] for c in concs])

    fig, ax = plt.subplots(figsize=(6, 4))
    fig.suptitle(peptide_name)

    ax.scatter(concs, cvs, color='steelblue', s=30, zorder=3)
    ax.axhline(cv_thresh, color='red', linestyle='--', linewidth=1.0,
               label=f'{cv_thresh:.0%} CV threshold')
    if np.isfinite(loq):
        ax.axvline(loq, color='c', linewidth=1.2, label=f'LOQ = {loq:.3e}')

    ax.set_xlabel('concentration')
    ax.set_ylabel('CV')
    ax.set_ylim(bottom=0)
    ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    ax.legend(fontsize=8)
    plt.tight_layout()

    if output_path is not None:
        out = Path(output_path)
        if out.is_dir():
            out = out / f'{peptide_name}_cv.png'
        fig.savefig(out, bbox_inches='tight')

    if show:
        return fig
    plt.close(fig)
    return None
