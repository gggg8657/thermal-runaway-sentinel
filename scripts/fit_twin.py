"""Identify the PyBAMM twin's four scalar parameters against real cycles.

    PYTHONPATH=. python scripts/fit_twin.py --workers 54

Nelder-Mead on `T_rmse + 10 K/V * V_rmse`, evaluated over a subsample of the
*fit* cells' windows only. The calibration and test cells are never touched
here. Writes `runs/twin_fit.json`.
"""
from __future__ import annotations

import argparse
import json
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from trsentinel.data.severson import load_severson_windows, window_dict
from trsentinel.split import cell_split, subsample, windows_of
from trsentinel.twin_pybamm import PybammTwin, TwinParams

#: 1 V of terminal-voltage error is traded against 10 K of temperature error
V_WEIGHT_K_PER_V = 10.0

_D = None
_KIND = "SPMe"


def _init(path, kind):
    global _D, _KIND
    _D = load_severson_windows(path)
    _KIND = kind


def _score_one(arg):
    i, vec = arg
    w = window_dict(_D, i)
    tw = PybammTwin(_KIND)
    r = tw.simulate(w, TwinParams.from_vector(vec))
    if not r["ok"]:
        return {"i": int(i), "ok": False, "v_rmse": np.nan, "t_rmse": np.nan,
                "t_bias": np.nan, "termination": r["termination"]}
    return {"i": int(i), "ok": True,
            "v_rmse": float(np.sqrt(np.mean((r["V"] - w["V"]) ** 2))),
            "t_rmse": float(np.sqrt(np.mean((r["T"] - w["T"]) ** 2))),
            "t_bias": float(np.mean(w["T"] - r["T"])),
            "termination": r["termination"]}


def score(pool, idx, vec):
    rows = pool.map(_score_one, [(i, vec) for i in idx])
    ok = [r for r in rows if r["ok"]]
    frac_ok = len(ok) / max(len(rows), 1)
    if not ok:
        return {"objective": 1e3, "frac_ok": 0.0, "n": len(rows)}
    v = float(np.mean([r["v_rmse"] for r in ok]))
    t = float(np.mean([r["t_rmse"] for r in ok]))
    # a window the solver could not finish is a modelling failure, not a free pass
    return {"objective": t + V_WEIGHT_K_PER_V * v + 5.0 * (1 - frac_ok),
            "v_rmse": v, "t_rmse": t, "frac_ok": frac_ok, "n": len(rows),
            "t_bias": float(np.mean([r["t_bias"] for r in ok])),
            "t_rmse_p95": float(np.percentile([r["t_rmse"] for r in ok], 95)),
            "v_rmse_p95": float(np.percentile([r["v_rmse"] for r in ok], 95))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/severson_discharge.npz")
    ap.add_argument("--cross-data", default=None,
                    help="second window set to score the fitted twin on, "
                         "without refitting (the cross-duty-cycle check)")
    ap.add_argument("--kind", default="SPMe", choices=["SPM", "SPMe", "DFN"])
    ap.add_argument("--workers", type=int, default=54)
    ap.add_argument("--n-fit-windows", type=int, default=60)
    ap.add_argument("--n-eval-windows", type=int, default=150)
    ap.add_argument("--maxiter", type=int, default=260)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/twin_fit.json")
    args = ap.parse_args()

    d = load_severson_windows(args.data)
    sp = cell_split(len(d["cells"]), seed=args.seed)
    fit_idx = subsample(windows_of(d, sp["fit"]), args.n_fit_windows, seed=1)

    x0 = TwinParams(k_area=0.7, k_cap=0.9, R_contact=0.006, h_conv=30.0,
                    c_scale=1.5).as_vector()
    # optimise in log space: every parameter is a positive scale
    lo = np.array([0.3, 0.4, 1e-4, 5.0, 0.4])
    hi = np.array([2.5, 1.6, 6e-2, 200.0, 4.0])

    hist = []
    t0 = time.time()
    with Pool(args.workers, initializer=_init,
              initargs=(args.data, args.kind)) as pool:
        def obj(z):
            vec = np.clip(np.exp(z), lo, hi)
            s = score(pool, fit_idx, vec)
            hist.append({"vec": vec.tolist(), **{k: v for k, v in s.items()}})
            return s["objective"]

        res = minimize(obj, np.log(x0), method="Nelder-Mead",
                       options={"maxiter": args.maxiter, "xatol": 1e-3,
                                "fatol": 1e-4, "adaptive": True})
        best = np.clip(np.exp(res.x), lo, hi)
        theta = TwinParams.from_vector(best)
        fit_s = time.time() - t0

        # held-out scoring, on cells the optimiser never saw
        out_scores = {}
        for name in ("fit", "cal", "test"):
            idx = subsample(windows_of(d, sp[name]), args.n_eval_windows, seed=2)
            out_scores[name] = score(pool, idx, best)

        # the reduced-order twin this project started with, on the same windows,
        # for the "did the physics twin actually change anything" comparison
        from trsentinel.sentinel import ThermalTwin
        rom = ThermalTwin().fit_windows(
            [window_dict(d, i) for i in subsample(windows_of(d, sp["fit"]),
                                                  args.n_fit_windows, seed=1)])
        rom_scores = {}
        for name in ("fit", "cal", "test"):
            idx = subsample(windows_of(d, sp[name]), args.n_eval_windows, seed=2)
            errs = []
            for i in idx:
                w = window_dict(d, i)
                Tp = rom.predict_window(w)
                errs.append((float(np.sqrt(np.mean((Tp - w["T"]) ** 2))),
                             float(np.mean(w["T"] - Tp))))
            rom_scores[name] = {
                "t_rmse": float(np.mean([e[0] for e in errs])),
                "t_rmse_p95": float(np.percentile([e[0] for e in errs], 95)),
                "t_bias": float(np.mean([e[1] for e in errs])),
                "n": len(idx), "frac_ok": 1.0}

    cross = {}
    if args.cross_data:
        with Pool(args.workers, initializer=_init,
                  initargs=(args.cross_data, args.kind)) as pool:
            dc = load_severson_windows(args.cross_data)
            spc = cell_split(len(dc["cells"]), seed=args.seed)
            for name in ("fit", "cal", "test"):
                idx = subsample(windows_of(dc, spc[name]), args.n_eval_windows,
                                seed=2)
                cross[name] = score(pool, idx, best)
            cross["phase"] = str(dc["phase"])

    out = {
        "model": args.kind + " + lumped thermal (PyBAMM)",
        "phase": str(d["phase"]),
        "data": args.data,
        "cross_phase": cross,
        "chemistry": "Prada2013 LFP/graphite, Chen2020 thermal properties",
        "theta": theta.to_dict(),
        "theta_names": list(TwinParams.NAMES),
        "n_fit_windows": len(fit_idx),
        "n_eval_windows_per_split": args.n_eval_windows,
        "split_cells": {k: [str(d["cells"][i]) for i in v] for k, v in sp.items()},
        "split_sizes": {k: int(len(v)) for k, v in sp.items()},
        "objective": "T_rmse[K] + 10 * V_rmse[V] + 5*(1-frac_solved)",
        "n_objective_evals": len(hist),
        "fit_wall_s": round(fit_s, 1),
        "pybamm": out_scores,
        "rom_lumped_lstsq": rom_scores,
        "rom_params": rom.to_dict(),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items() if k != "split_cells"}, indent=2))


if __name__ == "__main__":
    main()
