"""Scoring: conformal thresholds, false-alarm rates, lead times."""
from __future__ import annotations

import numpy as np

from .sentinel import alarm_index, persistence_statistic

#: nominal false-alarm rates the whole study is reported at
ALPHAS = [0.20, 0.10, 0.05, 0.02, 0.01]
#: windows at the start of each cell's life used to register its own offset
N_REGISTER = 3


def conformal_threshold(cal_scores, alpha):
    """Split-conformal threshold with the finite-sample correction."""
    s = np.sort(np.asarray(cal_scores, float))
    n = len(s)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    return float(np.inf) if k > n else float(s[k - 1])


def per_cell_offsets(res, cell_index, cycle, n_register=N_REGISTER):
    """Each cell's own residual offset, from its first few healthy windows.

    A deployed sentinel gets to watch a cell while it is known-good before it
    has to judge it. Registering that offset is the difference between asking
    "is this cell hotter than the fleet's model" and "is this cell hotter than
    it used to be", and the two give very different false-alarm rates. The
    windows spent on registration are excluded from every later evaluation.
    """
    offsets, used = {}, np.zeros(len(res), bool)
    for c in np.unique(cell_index):
        sel = np.where(cell_index == c)[0]
        sel = sel[np.argsort(cycle[sel])][:n_register]
        offsets[int(c)] = float(np.median([np.nanmean(res[j]) for j in sel]))
        used[sel] = True
    return offsets, used


def scores_for(res, idx, cell_index, persist, offsets=None):
    out = []
    for j in idx:
        r = res[j] if offsets is None else res[j] - offsets[int(cell_index[j])]
        out.append(persistence_statistic(r, persist))
    return np.array(out, float)


def far_table(cal_scores, test_scores, test_cells, alphas=ALPHAS):
    """Empirical false-alarm rate on held-out healthy cells, per nominal alpha."""
    rows = []
    for a in alphas:
        q = conformal_threshold(cal_scores, a)
        fired = np.asarray(test_scores) > q
        per_cell = [float(np.mean(fired[test_cells == c]))
                    for c in np.unique(test_cells)]
        rows.append({
            "alpha": a, "threshold_K": q,
            "far": float(np.mean(fired)),
            "n_test_windows": int(len(test_scores)),
            "n_fired": int(fired.sum()),
            "cells_with_any_alarm": int(sum(1 for p in per_cell if p > 0)),
            "n_test_cells": int(len(per_cell)),
            "worst_cell_far": float(max(per_cell)) if per_cell else 0.0,
        })
    return rows


def lead_time(residual, q_model, raw, q_raw, dt_s, persist):
    """Detection times of the residual alarm and of a model-free baseline.

    Both alarms use the identical persistence rule and are calibrated to the
    same nominal false-alarm rate on the same healthy cells, so the difference
    between them is the value of the physics twin and nothing else.
    """
    i_model = alarm_index(residual, q_model, persist)
    i_raw = alarm_index(raw, q_raw, persist)
    return {
        "t_model_s": None if i_model is None else float(i_model * dt_s),
        "t_raw_s": None if i_raw is None else float(i_raw * dt_s),
        "lead_s": (None if (i_model is None or i_raw is None)
                   else float((i_raw - i_model) * dt_s)),
        "model_fired": i_model is not None,
        "raw_fired": i_raw is not None,
    }
