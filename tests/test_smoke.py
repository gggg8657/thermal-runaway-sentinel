"""Smoke test: early warning fires before visible runaway, no healthy false alarm."""
from trsentinel import (
    healthy_fleet, faulty_cell, ThermalTwin, ConformalBand, early_warning,
)


def _setup():
    fleet = healthy_fleet(n_cells=30, seed=0)
    cal = healthy_fleet(n_cells=15, seed=100)
    twin = ThermalTwin().fit(fleet)
    band = ConformalBand(alpha=0.01).calibrate([twin.residual(c) for c in cal])
    return fleet, twin, band


def test_detects_before_runaway():
    fleet, twin, band = _setup()
    onset = 1100
    cell = faulty_cell(t_onset=onset)
    warn = early_warning(twin.residual(cell), band, persist=20)
    assert warn is not None, "no early warning fired on a faulty cell"
    assert warn >= onset - 5, "warned implausibly before the fault existed"


def test_no_false_alarm_on_healthy():
    fleet, twin, band = _setup()
    hc = healthy_fleet(n_cells=1, seed=555)[0]
    assert early_warning(twin.residual(hc), band, persist=20) is None


if __name__ == "__main__":
    test_detects_before_runaway()
    test_no_false_alarm_on_healthy()
    print("ok")
