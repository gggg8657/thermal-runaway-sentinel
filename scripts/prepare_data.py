"""Extract the 4C-discharge windows from the raw Severson batch1 pickle.

    python scripts/prepare_data.py --raw <path to batch1.pkl> --stride 25

Writes `data/severson_discharge.npz` (a few MB) plus `data/prepare_report.json`.
The raw archive stays where it is; nothing is written outside this repo.
"""
from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np

from trsentinel.data.severson import (
    PHASES, SEVERSON_META, extract_window, window_grid,
)

DEFAULT_RAW = ("/home/dongjukim/Documents/workspace/repos/PyBAMM_Inverse/"
               "data/raw/batch1.pkl")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=DEFAULT_RAW)
    ap.add_argument("--stride", type=int, default=25,
                    help="keep every Nth cycle of each cell")
    ap.add_argument("--first-cycle", type=int, default=10)
    ap.add_argument("--phase", default="discharge", choices=sorted(PHASES))
    ap.add_argument("--out", default=None)
    ap.add_argument("--report", default=None)
    args = ap.parse_args()
    args.out = args.out or f"data/severson_{args.phase}.npz"
    args.report = args.report or f"data/prepare_report_{args.phase}.json"

    t0 = time.time()
    with open(args.raw, "rb") as f:
        bat = pickle.load(f)
    load_s = time.time() - t0

    cells = sorted(bat, key=lambda s: int(s.split("cell")[1]))
    grid = window_grid(args.phase)
    I, V, T, cell_ix, cyc, amb, qd = [], [], [], [], [], [], []
    rejected = 0
    for ci, name in enumerate(cells):
        cyc_dict = bat[name]["cycles"]
        keys = sorted(cyc_dict, key=int)
        for k in keys:
            n = int(k)
            if n < args.first_cycle or n % args.stride:
                continue
            c = cyc_dict[k]
            w = extract_window(c["t"], c["I"], c["V"], c["T"], c["Qd"],
                               phase=args.phase)
            if w is None:
                rejected += 1
                continue
            I.append(w["I"]); V.append(w["V"]); T.append(w["T"])
            cell_ix.append(ci); cyc.append(n)
            amb.append(w["T_amb"]); qd.append(w["Qd_cycle"])

    out = dict(
        t=grid, I=np.array(I), V=np.array(V), T=np.array(T),
        cell_index=np.array(cell_ix), cycle=np.array(cyc),
        T_amb=np.array(amb), Qd_cycle=np.array(qd),
        cells=np.array(cells), phase=np.array(args.phase),
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **out)

    report = {
        "meta": SEVERSON_META,
        "phase": args.phase,
        "protocol": PHASES[args.phase]["protocol"],
        "raw_pickle": args.raw,
        "raw_load_s": round(load_s, 1),
        "n_cells": len(cells),
        "cycle_stride": args.stride,
        "n_windows": int(len(I)),
        "n_rejected_cycles": int(rejected),
        "window_s": float(grid[-1]),
        "dt_s": float(grid[1] - grid[0]),
        "T_amb_median_C": round(float(np.median(out["T_amb"])), 2),
        "peak_dT_median_C": round(float(np.median(out["T"].max(1) - out["T_amb"])), 2),
        "abs_current_median_A": round(float(np.median(np.abs(out["I"]))), 2),
        "abs_current_range_A": [round(float(np.min(np.abs(out["I"]))), 2),
                                round(float(np.max(np.abs(out["I"]))), 2)],
        "distinct_current_profiles": int(len(np.unique(
            np.round(np.abs(out["I"]).mean(1), 2)))),
        "Qd_range_Ah": [round(float(np.min(out["Qd_cycle"])), 3),
                        round(float(np.max(out["Qd_cycle"])), 3)],
        "prepare_s": round(time.time() - t0, 1),
    }
    Path(args.report).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
