"""TRI / Severson fast-charging dataset: real V/I/T from 46 LFP 18650 cells.

Source: Severson et al., *Data-driven prediction of battery cycle life before
capacity degradation*, Nature Energy 4, 383-391 (2019). Cells are A123
APR18650M1A LFP/graphite, 1.1 Ah nominal, cycled in a ~30 C forced-convection
chamber, with a thermocouple on the can. 5 s logging of I, V and T.

Two operating windows are extracted, and the contrast between them is the point:

``discharge``
    4C constant current to 2.0 V. **Every cell in the dataset is discharged
    identically**, so this window is a fixed, endlessly repeated duty cycle.
``charge``
    The first 200 s of the fast charge. Each cell runs one of 72 policies --
    a first constant-current step anywhere from 3.6C to 8C, sometimes stepping
    down to a second rate inside the window. **The duty cycle differs from cell
    to cell**, which is what makes it a real test of whether a model of the cell
    is worth having.

Both windows stay strictly inside constant-current control. During the 2.0 V and
3.6 V constant-voltage holds the cycler controls voltage rather than current,
and a current-driven twin cannot reproduce a clamped terminal voltage.

The raw archive is a ~2.8 GB pickle that is *not* in this repo;
`scripts/prepare_data.py` reads it once and writes the rectangular npz files
that everything downstream loads.
"""
from __future__ import annotations

import numpy as np

SEVERSON_META = {
    "source": "Severson et al. 2019 (TRI/Stanford/MIT fast-charging dataset), batch1",
    "cell": "A123 APR18650M1A, LFP/graphite 18650, 1.1 Ah nominal, 3.3 V",
    "chamber": "30 C forced convection",
    "channels": "I [A], V [V], T [degC] (thermocouple on the can), 5 s sampling",
}

#: uniform resampling step, matching the raw logger
DT_S = 5.0

#: Per phase: window length, and the current band the whole window must sit in.
#: 780 s is the longest window that lies inside the 4C constant-current
#: discharge of every cycle kept (the shortest, an aged cell at 0.88 Ah / 4 A,
#: lasts ~790 s). 200 s likewise stays inside the constant-current part of the
#: fast charge for all but a handful of the fastest policies.
PHASES = {
    "discharge": {"window_s": 780.0, "onset": -3.5, "lo": -np.inf, "hi": -3.5,
                  "soc0": 1.0,
                  "protocol": "4C constant current to 2.0 V, identical for every cell"},
    "charge": {"window_s": 200.0, "onset": 3.0, "lo": 1.5, "hi": np.inf,
               "soc0": 0.0,
               "protocol": "first 200 s of the cell's fast-charge policy, "
                           "constant current at 3.6C-8C (some policies step "
                           "down to a second rate inside the window)"},
}


def window_grid(phase="discharge"):
    return np.arange(0.0, PHASES[phase]["window_s"] + DT_S / 2, DT_S)


def extract_window(t_min, I, V, T, Qd, phase="discharge"):
    """Cut one constant-current window out of one raw cycle.

    Returns None when the cycle does not contain a clean one -- the first and
    last cycles of a run, cycles whose discharge is already too short, and a
    handful of logger glitches.
    """
    cfg = PHASES[phase]
    t = np.asarray(t_min, float) * 60.0          # the archive stores minutes
    I, V, T, Qd = (np.asarray(x, float) for x in (I, V, T, Qd))
    if t.size < 50 or not np.all(np.diff(t) >= 0):
        return None

    on = I < cfg["onset"] if phase == "discharge" else I > cfg["onset"]
    if on.sum() < 20:
        return None
    t0 = t[int(np.argmax(on))]
    if t[-1] - t0 < cfg["window_s"]:
        return None
    inside = (t >= t0) & (t <= t0 + cfg["window_s"])
    if np.any(I[inside] < cfg["lo"]) or np.any(I[inside] > cfg["hi"]):
        return None

    grid = window_grid(phase)
    sel = slice(max(int(np.argmax(on)) - 5, 0), None)
    out = {"t": grid}
    for name, arr in (("I", I), ("V", V), ("T", T), ("Qd", Qd)):
        out[name] = np.interp(grid + t0, t[sel], arr[sel])

    # Ambient: the chamber setpoint is 30 C, but each channel sits a little off
    # it. The near-rest just before the 4C discharge -- the tail of the 3.6 V
    # constant-voltage hold, where the current has tapered to a few tens of mA
    # -- is the best local estimate available, and it is a measurement, not a
    # fitted parameter. The charge window borrows the same cycle's estimate.
    dis = int(np.argmax(I < -3.5)) if np.any(I < -3.5) else None
    if dis is None:
        return None
    td = t[dis]
    pre = (t >= td - 120) & (t < td) & (np.abs(I) < 0.15)
    if pre.sum() < 5:
        return None
    out["T_amb"] = float(np.mean(T[pre]))
    out["Qd_cycle"] = float(np.nanmax(Qd))

    if not np.all(np.isfinite(np.concatenate([out["I"], out["V"], out["T"]]))):
        return None
    if not (20.0 < out["T_amb"] < 45.0):
        return None
    if not (0.6 < out["Qd_cycle"] < 1.3):
        return None
    return out


def load_severson_windows(path="data/severson_discharge.npz"):
    """Load a prepared window set.

    Returns a dict with rectangular arrays `I,V,T` of shape (n_windows, n_t),
    the shared time grid `t`, and per-window `cell_index`, `cycle`, `T_amb`,
    `Qd_cycle`, plus the list of cell names.
    """
    z = np.load(path, allow_pickle=True)
    d = {k: z[k] for k in z.files}
    d["cells"] = list(d["cells"])
    d["phase"] = str(d["phase"]) if "phase" in d else "discharge"
    return d


def window_dict(d, i, phase=None):
    """One window as the twin expects it (`t, I, V, T, T_amb, T0, soc0, ...`).

    `soc0` is not fitted: a discharge window starts from a cell the cycler has
    just held at 3.6 V, a charge window from one it has just held at 2.0 V.
    """
    phase = phase or str(d.get("phase", "discharge"))
    return {
        "t": d["t"], "I": d["I"][i], "V": d["V"][i], "T": d["T"][i],
        "T_amb": float(d["T_amb"][i]), "T0": float(d["T"][i][0]),
        "Qd_cycle": float(d["Qd_cycle"][i]),
        "cell": str(d["cells"][int(d["cell_index"][i])]),
        "cell_index": int(d["cell_index"][i]), "cycle": int(d["cycle"][i]),
        "phase": phase, "soc0": PHASES[phase]["soc0"],
    }
