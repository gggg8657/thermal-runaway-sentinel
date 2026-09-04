"""PyBAMM twin: energy balance, fault injection, and the staggered coupling.

Skipped when PyBAMM is not installed, so CI (numpy only) stays green.
"""
import sys

import numpy as np

try:
    import pybamm  # noqa: F401
except Exception:
    print("pybamm not installed - skipping"); sys.exit(0)

from trsentinel.fault import InternalShort, simulate_short
from trsentinel.twin_pybamm import CELL_AREA_M2, PybammTwin, TwinParams

THETA = TwinParams(k_area=0.7, k_cap=0.9, R_contact=0.006, h_conv=30.0, c_scale=1.5)


def _window(n=157, dt=5.0, T_amb=30.0):
    t = np.arange(n) * dt
    return {"t": t, "I": np.full(n, -4.0), "V": np.full(n, 3.2),
            "T": np.full(n, T_amb), "T_amb": T_amb, "T0": T_amb,
            "Qd_cycle": 1.05}


def test_twin_runs_and_heats_a_discharging_cell():
    r = PybammTwin("SPMe").simulate(_window(), THETA)
    assert r["ok"], r["termination"]
    assert r["T"][0] == 30.0
    assert 1.0 < r["T"][-1] - 30.0 < 30.0, "4C discharge should warm the cell"
    assert 2.0 < r["V"].min() and r["V"].max() < 3.7


def test_stepped_and_monolithic_solvers_agree_when_healthy():
    w = _window()
    a = PybammTwin("SPMe").simulate(w, THETA)
    b = simulate_short(w, THETA, None)
    assert b["ok"]
    assert np.max(np.abs(a["T"] - b["T"])) < 1e-2
    assert np.max(np.abs(a["V"] - b["V"])) < 1e-3


def test_injected_short_power_matches_the_lumped_energy_balance():
    """A steady Q_s must raise the steady-state temperature by Q_s / (hA)."""
    w = _window()
    base = simulate_short(w, THETA, None)
    hot = simulate_short(w, THETA, InternalShort(R0_ohm=50.0, onset_s=0.0))
    assert hot["ok"]
    dT = hot["T"] - base["T"]
    Q = np.nanmedian(hot["Q_short"][10:])
    ceiling = Q / (THETA.h_conv * CELL_AREA_M2)
    assert 0 < dT[-1] < ceiling * 1.05, "heat injection overshoots its own steady state"
    assert dT[-1] > 0.5 * ceiling, "780 s is >2 thermal time constants; expect most of it"


def test_a_worse_short_is_always_hotter():
    w = _window()
    base = simulate_short(w, THETA, None)
    peaks = []
    for R in (500.0, 100.0, 20.0):
        r = simulate_short(w, THETA, InternalShort(R0_ohm=R, onset_s=100.0))
        assert r["ok"]
        peaks.append(np.nanmax(r["T"] - base["T"]))
    assert peaks[0] < peaks[1] < peaks[2]


def test_coupling_error_is_small_at_the_5_s_step():
    w = _window()
    f = InternalShort(R0_ohm=500.0, onset_s=100.0, R_end_ohm=1.0, tau_s=120.0)
    coarse = simulate_short(w, THETA, f, dt=5.0)
    fine = simulate_short(w, THETA, f, dt=1.25)
    assert coarse["ok"] and fine["ok"]
    assert np.nanmax(np.abs(coarse["T"] - fine["T"])) < 0.25


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
