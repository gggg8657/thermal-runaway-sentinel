"""End-to-end smoke demo on a SYNTHETIC fleet -- runs with numpy only.

Every cell here comes out of `trsentinel/cell.py`, a toy lumped simulator. The
demo exists so the mechanism -- twin, residual, conformal band, persistence
rule -- can be read and run in one file with no data and no PyBAMM. **None of
its numbers are measurements**, and none of them appear in the README's results.
The real-data pipeline is `scripts/run_all.sh`.
"""
import numpy as np
from trsentinel import (
    healthy_fleet, faulty_cell, ThermalTwin, ConformalBand, early_warning,
)


def main():
    fleet = healthy_fleet(n_cells=30, seed=0)
    cal = healthy_fleet(n_cells=15, seed=100)

    twin = ThermalTwin().fit(fleet)
    band = ConformalBand(alpha=0.01).calibrate([twin.residual(c) for c in cal])

    onset = 1100
    cell = faulty_cell(t_onset=onset)
    res = twin.residual(cell)
    warn = early_warning(res, band, persist=20)

    # "visible runaway": when temperature exceeds healthy max by 10 C
    healthy_Tmax = max(c["T"].max() for c in fleet)
    runaway_idx = int(np.argmax(cell["T"] > healthy_Tmax + 10.0))
    runaway_idx = runaway_idx or len(cell["T"])

    print("SYNTHETIC simulator -- these numbers are not measurements.")
    print(f"fault onset step:        {onset}")
    print(f"early-warning fired at:  {warn}")
    print(f"visible runaway at:      {runaway_idx}")
    if warn is not None:
        print(f"lead time before runaway: {runaway_idx - warn} steps")
    print(f"conformal band (1% FAR): {band.hi:.3f} C")

    # false-alarm check on a held-out healthy cell
    hc = healthy_fleet(n_cells=1, seed=555)[0]
    fa = early_warning(twin.residual(hc), band, persist=20)
    print(f"healthy cell false alarm: {'NONE' if fa is None else fa}")
    return warn, runaway_idx, fa


if __name__ == "__main__":
    main()
