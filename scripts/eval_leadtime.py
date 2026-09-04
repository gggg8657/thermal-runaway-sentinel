"""Lead time under a **simulated** internal short. Nothing here is real-data.

    PYTHONPATH=. python scripts/eval_leadtime.py --workers 54

No public cycling dataset carries a labelled internal-short onset, so the fault
is injected into the physics twin (`trsentinel/fault.py`) and every number this
script writes is a simulation result. What keeps it from being a toy is where
the fault is injected:

    T_observed(t) = T_measured_real(t) + [ T_twin_faulted(t) - T_twin_healthy(t) ]

The trace a detector sees is a **real measured temperature trace from a real
held-out cell**, plus the temperature increment the short causes. So the alarm
has to clear the real cell's real residual noise floor -- the twin's own
modelling error against that cell -- not a clean simulated baseline. The
threshold is the one calibrated on real healthy cells in `scripts/eval_far.py`,
never re-tuned here.

Writes `runs/leadtime.json`.
"""
from __future__ import annotations

import argparse
import json
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from trsentinel.data.severson import load_severson_windows, window_dict
from trsentinel.fault import InternalShort, simulate_short
from trsentinel.metrics import (
    conformal_threshold, lead_time, per_cell_offsets, scores_for,
)
from trsentinel.split import cell_split, subsample, windows_of
from trsentinel.twin_pybamm import TwinParams

#: constant soft shorts, plus one that worsens with time
FAULTS = {
    "R=500 ohm (constant)": InternalShort(R0_ohm=500.0, onset_s=200.0),
    "R=200 ohm (constant)": InternalShort(R0_ohm=200.0, onset_s=200.0),
    "R=100 ohm (constant)": InternalShort(R0_ohm=100.0, onset_s=200.0),
    "R=50 ohm (constant)": InternalShort(R0_ohm=50.0, onset_s=200.0),
    "R=20 ohm (constant)": InternalShort(R0_ohm=20.0, onset_s=200.0),
    "worsening 500->1 ohm (tau=120 s)": InternalShort(
        R0_ohm=500.0, onset_s=100.0, R_end_ohm=1.0, tau_s=120.0),
}

#: nominal false-alarm rates the lead time is reported at
ALPHAS_REPORT = [0.05, 0.01]

_D = _THETA = _KIND = None


def _init(path, theta_vec, kind):
    global _D, _THETA, _KIND
    _D = load_severson_windows(path)
    _THETA = TwinParams.from_vector(theta_vec)
    _KIND = kind


def _run_window(i):
    """Healthy twin + every fault, on one real window."""
    w = window_dict(_D, i)
    healthy = simulate_short(w, _THETA, None, kind=_KIND)
    out = {"i": int(i), "ok": healthy["ok"], "T_healthy": healthy["T"],
           "faults": {}}
    if not healthy["ok"]:
        return out
    for name, f in FAULTS.items():
        r = simulate_short(w, _THETA, f, kind=_KIND)
        out["faults"][name] = {
            "ok": r["ok"], "dT": r["T"] - healthy["T"],
            "Q_peak_W": float(np.nanmax(r["Q_short"])),
            "I_peak_A": float(np.nanmax(r["I_short"])),
        }
    return out


def _verify_coupling(path, theta, kind, i):
    d = load_severson_windows(path)
    w = window_dict(d, i)
    f = FAULTS["worsening 500->1 ohm (tau=120 s)"]
    a = simulate_short(w, theta, f, kind=kind, dt=5.0)
    b = simulate_short(w, theta, f, kind=kind, dt=1.25)
    return {"window": int(i), "dt_coarse_s": 5.0, "dt_fine_s": 1.25,
            "max_abs_dT_K": float(np.nanmax(np.abs(a["T"] - b["T"]))),
            "max_abs_dV_V": float(np.nanmax(np.abs(a["V"] - b["V"])))}


def _thresholds(d, res_pb, res_raw, cal_idx, ci, persist, alphas):
    """Alarm thresholds, per variant, taken from the real healthy calibration
    cells exactly as `scripts/eval_far.py` sets them. Nothing is re-tuned on
    faulted data."""
    out = {}
    offs_pb, used_pb = per_cell_offsets(res_pb, ci, d["cycle"])
    offs_raw, used_raw = per_cell_offsets(res_raw, ci, d["cycle"])
    for variant in ("fleet_calibrated", "cell_registered"):
        if variant == "cell_registered":
            cp, cr = cal_idx[~used_pb[cal_idx]], cal_idx[~used_raw[cal_idx]]
            op, orr = offs_pb, offs_raw
        else:
            cp = cr = cal_idx
            op = orr = None
        sp_ = scores_for(res_pb, cp, ci, persist, op)
        sr_ = scores_for(res_raw, cr, ci, persist, orr)
        out[variant] = {
            "offsets_residual": op, "offsets_raw": orr, "used": used_pb,
            "q": {a: {"residual": conformal_threshold(sp_, a),
                      "no_model": conformal_threshold(sr_, a)} for a in alphas},
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/severson_discharge.npz")
    ap.add_argument("--fit", default="runs/twin_fit.json")
    ap.add_argument("--residuals", default="runs/residuals.npz")
    ap.add_argument("--workers", type=int, default=54)
    ap.add_argument("--n-windows", type=int, default=120)
    ap.add_argument("--persist", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/leadtime.json")
    args = ap.parse_args()

    fit = json.loads(Path(args.fit).read_text())
    theta = TwinParams(**fit["theta"])
    kind = fit["model"].split()[0]

    d = load_severson_windows(args.data)
    dt_s = float(d["t"][1] - d["t"][0])
    sp = cell_split(len(d["cells"]), seed=args.seed)
    ci = d["cell_index"]
    res_pb = np.load(args.residuals)["res_pybamm"]
    res_raw = d["T"] - d["T_amb"][:, None]

    thr = _thresholds(d, res_pb, res_raw, windows_of(d, sp["cal"]), ci,
                      args.persist, ALPHAS_REPORT)

    # score on test cells, skipping the windows spent registering each cell
    test_idx = windows_of(d, sp["test"])
    test_idx = test_idx[~thr["cell_registered"]["used"][test_idx]]
    idx = subsample(test_idx, args.n_windows, seed=7)

    t0 = time.time()
    with Pool(args.workers, initializer=_init,
              initargs=(args.data, theta.as_vector(), kind)) as pool:
        runs = pool.map(_run_window, idx)
    sim_s = time.time() - t0
    coupling = _verify_coupling(args.data, theta, kind, int(idx[0]))

    out = {
        "SIMULATED": ("every lead time below comes from a fault injected into "
                      "the physics twin and added to a real measured temperature "
                      "trace; no measured runaway event is involved anywhere"),
        "fault_model": ("resistance across the electrodes: shunt current V/R "
                        "drawn inside the cell plus V^2/R of ohmic heat added to "
                        "the lumped energy balance"),
        "persist_samples": args.persist, "dt_s": dt_s,
        "n_windows": int(len(idx)),
        "n_test_cells": int(len(np.unique(ci[idx]))),
        "sim_wall_s": round(sim_s, 1),
        "sim_s_per_window_per_core": round(sim_s * args.workers / (len(idx) * (1 + len(FAULTS))), 3),
        "coupling_check": coupling,
        "thresholds_K": {v: {str(a): thr[v]["q"][a] for a in ALPHAS_REPORT}
                         for v in thr},
        "healthy_control": {}, "faults": {},
    }

    def score(dT, name, variant, alpha):
        """Alarm times for one fault, one variant, one alpha, over all windows."""
        q = thr[variant]["q"][alpha]
        op, orr = thr[variant]["offsets_residual"], thr[variant]["offsets_raw"]
        rows = []
        for run in runs:
            if not run["ok"]:
                continue
            i = run["i"]
            inc = dT(run)
            if inc is None:
                continue
            T_obs = d["T"][i] + inc
            r = T_obs - run["T_healthy"]
            raw = T_obs - d["T_amb"][i]
            if op is not None:
                r = r - op[int(ci[i])]
                raw = raw - orr[int(ci[i])]
            lt = lead_time(r, q["residual"], raw, q["no_model"], dt_s, args.persist)
            lt["peak_dT_K"] = float(np.nanmax(inc))
            rows.append(lt)
        return rows

    def summarise(rows, onset_s):
        """Detection is only counted after onset; an alarm before onset is a
        false alarm from the *real* residual and is reported separately."""
        n = len(rows)
        pre = [r for r in rows if r["t_model_s"] is not None and r["t_model_s"] < onset_s]
        pre_raw = [r for r in rows if r["t_raw_s"] is not None and r["t_raw_s"] < onset_s]
        det = [r for r in rows if r["t_model_s"] is not None and r["t_model_s"] >= onset_s]
        det_raw = [r for r in rows if r["t_raw_s"] is not None and r["t_raw_s"] >= onset_s]
        both = [r for r in rows
                if r["t_model_s"] is not None and r["t_model_s"] >= onset_s
                and r["t_raw_s"] is not None and r["t_raw_s"] >= onset_s]
        return {
            "n_windows": n,
            "detection_rate_residual": round(len(det) / n, 4) if n else None,
            "detection_rate_no_model": round(len(det_raw) / n, 4) if n else None,
            "false_alarm_before_onset_residual": round(len(pre) / n, 4) if n else None,
            "false_alarm_before_onset_no_model": round(len(pre_raw) / n, 4) if n else None,
            "median_detect_after_onset_s": (
                round(float(np.median([r["t_model_s"] for r in det])) - onset_s, 1)
                if det else None),
            "median_no_model_after_onset_s": (
                round(float(np.median([r["t_raw_s"] for r in det_raw])) - onset_s, 1)
                if det_raw else None),
            "median_lead_vs_no_model_s": (
                round(float(np.median([r["t_raw_s"] - r["t_model_s"] for r in both])), 1)
                if both else None),
            "n_detected_by_residual_only": int(sum(
                1 for r in rows
                if (r["t_model_s"] is not None and r["t_model_s"] >= onset_s)
                and not (r["t_raw_s"] is not None and r["t_raw_s"] >= onset_s))),
            "n_detected_by_no_model_only": int(sum(
                1 for r in rows
                if (r["t_raw_s"] is not None and r["t_raw_s"] >= onset_s)
                and not (r["t_model_s"] is not None and r["t_model_s"] >= onset_s))),
        }

    # healthy control: the identical scoring path with no fault at all, so the
    # simulated half and the real-data half are tied together by one number
    for variant in thr:
        for a in ALPHAS_REPORT:
            rows = score(lambda run: np.zeros_like(run["T_healthy"]), None, variant, a)
            key = f"{variant} alpha={a}"
            out["healthy_control"][key] = {
                "far_residual": round(float(np.mean([r["model_fired"] for r in rows])), 4),
                "far_no_model": round(float(np.mean([r["raw_fired"] for r in rows])), 4),
                "n_windows": len(rows),
            }

    for name, f in FAULTS.items():
        entry = {"fault": f.to_dict()}
        okruns = [r for r in runs if r["ok"] and r["faults"].get(name, {}).get("ok")]
        entry["n_windows_simulated"] = len(okruns)
        if okruns:
            entry["peak_short_power_W"] = round(float(np.median(
                [r["faults"][name]["Q_peak_W"] for r in okruns])), 3)
            entry["peak_shunt_current_A"] = round(float(np.median(
                [r["faults"][name]["I_peak_A"] for r in okruns])), 4)
            entry["peak_dT_K"] = round(float(np.median(
                [np.nanmax(r["faults"][name]["dT"]) for r in okruns])), 3)
        for variant in thr:
            for a in ALPHAS_REPORT:
                rows = score(
                    lambda run, nm=name: (run["faults"][nm]["dT"]
                                          if run["faults"].get(nm, {}).get("ok") else None),
                    name, variant, a)
                entry[f"{variant} alpha={a}"] = summarise(rows, f.onset_s)
        out["faults"][name] = entry

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(json.dumps({k: out[k] for k in
                      ("healthy_control", "thresholds_K", "coupling_check",
                       "sim_wall_s")}, indent=2))
    for k, v in out["faults"].items():
        print(f"--- {k}: Q={v.get('peak_short_power_W')} W  peak dT={v.get('peak_dT_K')} K")
        for vv in ("cell_registered alpha=0.05", "fleet_calibrated alpha=0.05"):
            s = v[vv]
            print(f"    {vv}: det={s['detection_rate_residual']} (no-model {s['detection_rate_no_model']}) "
                  f"t_det={s['median_detect_after_onset_s']}s lead={s['median_lead_vs_no_model_s']}s")


if __name__ == "__main__":
    main()
