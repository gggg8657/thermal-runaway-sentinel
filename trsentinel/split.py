"""Fleet split. Every split in this project is **by cell**, never by cycle.

Splitting by cycle would put cycle 300 of a cell in the fit set and cycle 325 of
the same cell in the test set; the twin would then be scored on a cell whose
individual thermocouple offset and cooling it had already seen, and the
false-alarm rate would be optimistic in exactly the way that matters.
"""
from __future__ import annotations

import numpy as np

#: Fractions, so the same code splits the 46-cell Severson fleet 10/12/24 and
#: the 30-cell NASA fleet 7/8/15: some to identify the twin, some to calibrate
#: the conformal band, and the majority never seen until the false-alarm rate
#: is measured.
F_FIT, F_CAL = 10 / 46, 12 / 46


def cell_split(n_cells, seed=0, f_fit=F_FIT, f_cal=F_CAL):
    n_fit = int(round(f_fit * n_cells))
    n_cal = int(round(f_cal * n_cells))
    perm = np.random.default_rng(seed).permutation(n_cells)
    return {"fit": np.sort(perm[:n_fit]),
            "cal": np.sort(perm[n_fit:n_fit + n_cal]),
            "test": np.sort(perm[n_fit + n_cal:])}


def windows_of(d, cells):
    return np.where(np.isin(d["cell_index"], np.asarray(cells)))[0]


def subsample(idx, n, seed=0):
    if len(idx) <= n:
        return np.asarray(idx)
    return np.sort(np.random.default_rng(seed).choice(idx, n, replace=False))
