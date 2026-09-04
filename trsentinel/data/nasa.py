"""NASA PCoE randomised battery ageing data: a second chemistry, second lab.

Source: B. Saha and K. Goebel, *Battery Data Set*, NASA Ames Prognostics Center
of Excellence. 18650 Li-ion cells (LCO/graphite, 2 Ah nominal) cycled to failure
with V, I and a **thermocouple temperature** logged at ~19 s.

Two things make this worth carrying alongside the Severson corpus, at the cost
of one more chemistry to parameterise:

* **A different chemistry and a different thermal environment.** LCO rather than
  LFP, and still air rather than a forced-convection chamber, so the same twin
  and the same conformal machinery are exercised twice on genuinely unlike cells.
* **A heterogeneous duty cycle inside one dataset.** The discharge current runs
  from 1 A to 4 A and the chamber sits at 4 C, 24 C or 43 C depending on the
  cell. The Severson discharge, by contrast, is the same 4C for all 46 cells --
  which is why a model-free temperature threshold does so well there.

`scripts/prepare_nasa.py` reads the .mat archives and writes the same
rectangular npz layout the Severson loader produces, so everything downstream is
shared.
"""
from __future__ import annotations

import numpy as np

NASA_META = {
    "source": "NASA Ames PCoE Battery Data Set (Saha & Goebel)",
    "cell": "18650 Li-ion, LCO/graphite, 2 Ah nominal",
    "chamber": "4 / 24 / 43 C, still air, varies by cell",
    "channels": "I [A], V [V], T [degC] (thermocouple), ~19 s sampling",
    "protocol": "constant-current discharge, 1-4 A depending on the cell",
}

#: The shortest constant-current discharge in the corpus that is kept runs a
#: little over 900 s (a 4 A cell at 4 C), so 900 s is the longest common window.
WINDOW_S = 900.0
DT_S = 20.0
#: current must stay on its plateau, within this fraction of its median
PLATEAU_TOL = 0.12


def window_grid():
    return np.arange(0.0, WINDOW_S + DT_S / 2, DT_S)


def extract_window(t, I, V, T, T_amb, capacity_Ah):
    """Cut the constant-current part of one NASA discharge cycle."""
    t, I, V, T = (np.asarray(x, float).ravel() for x in (t, I, V, T))
    if t.size < 20 or not np.all(np.diff(t) > 0):
        return None
    on = I < -0.5
    if on.sum() < 15:
        return None
    i0 = int(np.argmax(on))
    t0 = t[i0]
    if t[-1] - t0 < WINDOW_S:
        return None
    inside = (t >= t0) & (t <= t0 + WINDOW_S)
    Iw = I[inside]
    med = float(np.median(Iw))
    if med > -0.5 or np.any(np.abs(Iw - med) > PLATEAU_TOL * abs(med)):
        return None

    grid = window_grid()
    sel = slice(max(i0 - 2, 0), None)
    out = {"t": grid}
    for name, arr in (("I", I), ("V", V), ("T", T)):
        out[name] = np.interp(grid + t0, t[sel], arr[sel])
    out["T_amb"] = float(T_amb)
    out["Qd_cycle"] = float(capacity_Ah)
    if not np.all(np.isfinite(np.concatenate([out["I"], out["V"], out["T"]]))):
        return None
    # the per-cycle Capacity field is measured down to the 2.7 V cutoff and is
    # nonsense for cycles that never got there; a cell that held this current
    # for the whole window demonstrably had at least this much charge in it
    if not (1.05 * abs(med) * WINDOW_S / 3600.0 < out["Qd_cycle"] < 3.0):
        return None
    # the cell must start near its own chamber, or the "ambient" is not one
    if abs(out["T"][0] - out["T_amb"]) > 12.0:
        return None
    return out
