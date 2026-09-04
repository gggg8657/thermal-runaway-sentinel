"""Extract constant-current discharge windows from the NASA PCoE archives.

    python scripts/prepare_nasa.py --root data/nasa/ext

Writes `data/nasa_discharge.npz` in the same layout as the Severson sets, plus
`data/prepare_report_nasa.json`.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import time
from pathlib import Path

import numpy as np
import scipy.io as sio

from trsentinel.data.nasa import NASA_META, extract_window, window_grid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/nasa/ext")
    ap.add_argument("--stride", type=int, default=2, help="keep every Nth cycle")
    ap.add_argument("--out", default="data/nasa_discharge.npz")
    ap.add_argument("--report", default="data/prepare_report_nasa.json")
    args = ap.parse_args()

    t0 = time.time()
    files, seen = [], set()
    for f in sorted(glob.glob(os.path.join(args.root, "**", "*.mat"),
                              recursive=True)):
        name = os.path.basename(f)[:-4]
        if name in seen:                      # the archives overlap
            continue
        seen.add(name); files.append((name, f))

    grid = window_grid()
    I, V, T, ci, cyc, amb, qd, cells = [], [], [], [], [], [], [], []
    rejected = 0
    for name, f in files:
        try:
            cycles = sio.loadmat(f, simplify_cells=True)[name]["cycle"]
        except Exception:
            continue
        kept_any = False
        n_dis = 0
        for c in np.atleast_1d(cycles):
            if c.get("type") != "discharge":
                continue
            n_dis += 1
            if n_dis % args.stride:
                continue
            d = c["data"]
            cap = np.asarray(d.get("Capacity", np.nan), float).ravel()
            if cap.size == 0 or not np.isfinite(cap[0]):
                rejected += 1
                continue
            w = extract_window(d["Time"], d["Current_measured"],
                               d["Voltage_measured"], d["Temperature_measured"],
                               c["ambient_temperature"], cap[0])
            if w is None:
                rejected += 1
                continue
            if not kept_any:
                cells.append(name); kept_any = True
            I.append(w["I"]); V.append(w["V"]); T.append(w["T"])
            ci.append(len(cells) - 1); cyc.append(n_dis)
            amb.append(w["T_amb"]); qd.append(w["Qd_cycle"])

    out = dict(t=grid, I=np.array(I), V=np.array(V), T=np.array(T),
               cell_index=np.array(ci), cycle=np.array(cyc),
               T_amb=np.array(amb), Qd_cycle=np.array(qd),
               cells=np.array(cells), phase=np.array("discharge"))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **out)

    mean_I = np.abs(out["I"]).mean(1)
    report = {
        "meta": NASA_META, "phase": "nasa_discharge",
        "protocol": NASA_META["protocol"],
        "root": args.root, "n_mat_files": len(files),
        "n_cells": len(cells), "cells": cells,
        "cycle_stride": args.stride,
        "n_windows": int(len(I)), "n_rejected_cycles": int(rejected),
        "window_s": float(grid[-1]), "dt_s": float(grid[1] - grid[0]),
        "T_amb_median_C": round(float(np.median(out["T_amb"])), 2),
        "T_amb_values_C": sorted(set(np.round(out["T_amb"], 1).tolist())),
        "peak_dT_median_C": round(float(np.median(out["T"].max(1) - out["T_amb"])), 2),
        "abs_current_median_A": round(float(np.median(np.abs(out["I"]))), 2),
        "abs_current_range_A": [round(float(mean_I.min()), 2),
                                round(float(mean_I.max()), 2)],
        "distinct_current_profiles": int(len(np.unique(np.round(mean_I, 2)))),
        "Qd_range_Ah": [round(float(np.min(out["Qd_cycle"])), 3),
                        round(float(np.max(out["Qd_cycle"])), 3)],
        "prepare_s": round(time.time() - t0, 1),
    }
    Path(args.report).write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "cells"}, indent=2))


if __name__ == "__main__":
    main()
