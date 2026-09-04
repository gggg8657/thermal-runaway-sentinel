"""False-alarm rate of the conformal band on **real healthy cells**.

    PYTHONPATH=. python scripts/eval_far.py --workers 54

This is the real-data half of the project. Nothing here is simulated: the twin
is run over every prepared 4C-discharge window of all 46 Severson cells, the
conformal threshold is calibrated on the 12 calibration cells, and the alarm
rate is measured on the 24 test cells -- cells that were used neither to
identify the twin nor to set the threshold. A conformal band promises
`P(alarm | healthy) <= alpha`; whether that survives being calibrated on one set
of physical cells and applied to another is an empirical question, and this is
the script that answers it.

Writes `runs/far.json` and `runs/residuals.npz`.
"""
from __future__ import annotations

import argparse
import json
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from trsentinel.data.severson import load_severson_windows, window_dict
from trsentinel.metrics import (
    ALPHAS, N_REGISTER, far_table, per_cell_offsets, scores_for,
)
from trsentinel.sentinel import ThermalTwin
from trsentinel.split import cell_split, subsample, windows_of
from trsentinel.twin_pybamm import PybammTwin, TwinParams

_D = _THETA = _KIND = _CHEM = None


def _init(path, theta_vec, kind, chemistry):
    global _D, _THETA, _KIND, _CHEM
    _D = load_severson_windows(path)
    _THETA = TwinParams.from_vector(theta_vec)
    _KIND, _CHEM = kind, chemistry


def _residual_one(i):
    w = window_dict(_D, i)
    r = PybammTwin(_KIND, chemistry=_CHEM).simulate(w, _THETA)
    return int(i), r["T"], bool(r["ok"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/severson_discharge.npz")
    ap.add_argument("--fit", default="runs/twin_fit.json")
    ap.add_argument("--workers", type=int, default=54)
    ap.add_argument("--persist", type=int, default=6,
                    help="consecutive samples above the threshold")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/far.json")
    ap.add_argument("--residuals", default="runs/residuals.npz")
    args = ap.parse_args()

    fit = json.loads(Path(args.fit).read_text())
    theta = TwinParams(**fit["theta"])
    kind = fit["model"].split()[0]
    chem = fit.get("chemistry", "Prada2013")

    d = load_severson_windows(args.data)
    sp = cell_split(len(d["cells"]), seed=args.seed)
    n_w = len(d["cell_index"])

    t0 = time.time()
    Tpred = np.full(d["T"].shape, np.nan)
    ok = np.zeros(n_w, bool)
    with Pool(args.workers, initializer=_init,
              initargs=(args.data, theta.as_vector(), kind, chem)) as pool:
        for i, Ti, o in pool.imap_unordered(_residual_one, range(n_w), chunksize=4):
            Tpred[i] = Ti
            ok[i] = o
    twin_s = time.time() - t0

    # reduced-order twin, fitted on the same cells, for the comparison
    rom = ThermalTwin().fit_windows(
        [window_dict(d, i) for i in subsample(windows_of(d, sp["fit"]), 60, seed=1)])
    Trom = np.array([rom.predict_window(window_dict(d, i)) for i in range(n_w)])

    res_pb = d["T"] - Tpred
    res_rom = d["T"] - Trom
    # model-free baseline: the raw temperature rise over ambient, same alarm rule
    res_raw = d["T"] - d["T_amb"][:, None]

    cal_idx = windows_of(d, sp["cal"])
    test_idx = windows_of(d, sp["test"])
    ci = d["cell_index"]

    report = {
        "data": json.loads(Path(
            f"data/prepare_report_{d['phase']}.json").read_text()),
        "phase": str(d["phase"]),
        "twin_fit": args.fit,
        "persist_samples": args.persist,
        "persist_s": args.persist * float(d["t"][1] - d["t"][0]),
        "n_windows": int(n_w),
        "twin_solve_failures": int((~ok).sum()),
        "twin_wall_s": round(twin_s, 1),
        "twin_s_per_window": round(twin_s * args.workers / n_w, 3),
        "split_sizes": {k: int(len(v)) for k, v in sp.items()},
        "n_cal_windows": int(len(cal_idx)), "n_test_windows": int(len(test_idx)),
        "variants": {},
    }

    for name, res in (("pybamm_spme", res_pb), ("rom_lumped", res_rom),
                      ("no_model_dT_over_ambient", res_raw)):
        entry = {}
        # (a) fleet-calibrated: threshold from other cells, applied as is
        cs = scores_for(res, cal_idx, ci, args.persist)
        ts = scores_for(res, test_idx, ci, args.persist)
        entry["fleet_calibrated"] = far_table(cs, ts, ci[test_idx])
        # (b) cell-registered: each cell's own offset from its first 3 windows
        offs, used = per_cell_offsets(res, ci, d["cycle"], N_REGISTER)
        cal2 = cal_idx[~used[cal_idx]]
        test2 = test_idx[~used[test_idx]]
        cs2 = scores_for(res, cal2, ci, args.persist, offs)
        ts2 = scores_for(res, test2, ci, args.persist, offs)
        entry["cell_registered"] = far_table(cs2, ts2, ci[test2])
        entry["n_cal_windows_registered"] = int(len(cal2))
        entry["n_test_windows_registered"] = int(len(test2))
        entry["residual_rms_test_K"] = float(np.sqrt(np.nanmean(res[test_idx] ** 2)))
        entry["residual_bias_spread_K"] = float(np.std(
            [np.nanmean(res[ci == c]) for c in np.unique(ci)]))
        report["variants"][name] = entry

    np.savez_compressed(args.residuals, res_pybamm=res_pb, res_rom=res_rom,
                        T_pred_pybamm=Tpred, ok=ok,
                        cell_index=ci, cycle=d["cycle"], t=d["t"])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    for k, v in report["variants"].items():
        print(f"--- {k}  rms={v['residual_rms_test_K']:.3f} K")
        for tag in ("fleet_calibrated", "cell_registered"):
            print("   ", tag, [f"a={r['alpha']}->FAR={r['far']:.3f} q={r['threshold_K']:.2f}K"
                               for r in v[tag]])
    print("twin wall", report["twin_wall_s"], "s;", report["twin_solve_failures"], "failures")


if __name__ == "__main__":
    main()
