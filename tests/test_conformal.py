"""Conformal machinery, the fault's resistance law, and the reduced-order twin.

numpy only -- this is what CI runs. The PyBAMM twin has its own test in
`tests/test_twin_pybamm.py`, which skips when PyBAMM is not installed.
"""
import numpy as np

from trsentinel.fault import InternalShort
from trsentinel.metrics import conformal_threshold, far_table, per_cell_offsets
from trsentinel.sentinel import ThermalTwin, alarm_index, persistence_statistic


def test_persistence_statistic_matches_the_alarm_rule():
    rng = np.random.default_rng(0)
    for _ in range(200):
        r = rng.normal(size=60)
        p = int(rng.integers(1, 8))
        s = persistence_statistic(r, p)
        for q in (s - 1e-9, s + 1e-9):
            fired = alarm_index(r, q, p) is not None
            assert fired == (s > q), "scalar score disagrees with the alarm rule"


def test_conformal_threshold_controls_the_false_alarm_rate():
    """P(score > q) <= alpha for exchangeable data, at finite n."""
    rng = np.random.default_rng(1)
    for alpha in (0.2, 0.05):
        fired = []
        for _ in range(4000):
            cal = rng.standard_t(df=3, size=200)
            q = conformal_threshold(cal, alpha)
            fired.append(rng.standard_t(df=3) > q)
        rate = float(np.mean(fired))
        assert rate <= alpha + 3 * np.sqrt(alpha * (1 - alpha) / 4000), \
            f"alpha={alpha}: empirical {rate:.4f} exceeds the guarantee"


def test_conformal_threshold_is_infinite_when_calibration_is_too_small():
    assert np.isinf(conformal_threshold(np.arange(5.0), alpha=0.01))


def test_far_table_shapes():
    rng = np.random.default_rng(2)
    cal, test = rng.normal(size=300), rng.normal(size=400)
    cells = rng.integers(0, 8, size=400)
    rows = far_table(cal, test, cells, alphas=[0.1, 0.05])
    assert [r["alpha"] for r in rows] == [0.1, 0.05]
    assert rows[0]["far"] >= rows[1]["far"]


def test_per_cell_offsets_uses_only_the_earliest_windows():
    res = np.array([[0.0], [1.0], [5.0], [0.0], [2.0]])
    cell = np.array([0, 0, 0, 1, 1])
    cycle = np.array([30, 10, 20, 50, 25])
    offs, used = per_cell_offsets(res, cell, cycle, n_register=2)
    assert offs[0] == 3.0          # median of cycles 10 and 20 -> (1+5)/2
    assert offs[1] == 1.0          # median of cycles 25 and 50 -> (2+0)/2
    assert used.tolist() == [False, True, True, True, True]


def test_internal_short_resistance_law():
    s = InternalShort(R0_ohm=100.0, onset_s=50.0)
    R = s.resistance(np.array([0.0, 49.0, 50.0, 500.0]))
    assert np.isinf(R[0]) and np.isinf(R[1])
    assert R[2] == 100.0 and R[3] == 100.0

    g = InternalShort(R0_ohm=500.0, onset_s=0.0, R_end_ohm=1.0, tau_s=100.0)
    Rg = g.resistance(np.array([0.0, 100.0, 1e4]))
    assert abs(Rg[0] - 500.0) < 1e-9
    assert 1.0 < Rg[1] < 500.0 and abs(Rg[2] - 1.0) < 1e-6


def test_rom_twin_recovers_a_known_lumped_cell():
    """The reduced-order twin must invert its own forward model exactly."""
    R_over_C, h_over_C, dt, n = 0.02, 1 / 300.0, 5.0, 200
    t = np.arange(n) * dt
    I = np.where((t // 200) % 2 == 0, -4.0, 0.0)
    T = np.empty(n); T[0] = 30.0
    for k in range(1, n):
        T[k] = T[k - 1] + dt * (R_over_C * I[k - 1] ** 2 - h_over_C * (T[k - 1] - 30.0))
    w = {"t": t, "I": I, "T": T, "T_amb": 30.0}
    twin = ThermalTwin().fit_windows([w])
    assert abs(twin.R_over_C - R_over_C) < 2e-3
    assert abs(twin.h_over_C - h_over_C) < 2e-4
    assert np.max(np.abs(twin.predict_window(w) - T)) < 0.05


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
