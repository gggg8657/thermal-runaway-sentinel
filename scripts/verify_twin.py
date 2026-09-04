"""Verify the twin's energy balance before anything downstream trusts it.

    PYTHONPATH=. python scripts/verify_twin.py

Two checks on real windows, both of which the twin has to pass before a
temperature residual computed from it means anything:

1. **Heat generation against the terminal energy balance.** All the heat a cell
   releases has to come from `I (U - V)`, the gap between its open-circuit and
   terminal voltage. PyBAMM's own heat terms are summed and compared against
   that gap.
2. **The lumped thermal ODE.** `C dT/dt = Q - hA (T - T_amb)` is re-integrated
   from the twin's own reported heat and compared with the twin's own reported
   temperature -- if the two disagree, the ambient-offset trick that both the
   contact-resistance heat and the injected fault heat rely on is wrong.

Writes `runs/verify.json`.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from trsentinel.data.severson import load_severson_windows, window_dict
from trsentinel.fault import InternalShort, simulate_short
from trsentinel.split import cell_split, subsample, windows_of
from trsentinel.twin_pybamm import (
    CELL_AREA_M2, PybammTwin, TwinParams, borrowed_keys, build_model,
)


def energy_balance(window, theta, kind="SPMe", chemistry="Prada2013"):
    import pybamm

    t = np.asarray(window["t"], float)
    tw = PybammTwin(kind, chemistry=chemistry)
    p = tw._params_for(theta, t, window["I"], window["T_amb"], window["T0"],
                       window["Qd_cycle"])
    sim = pybamm.Simulation(build_model(kind), parameter_values=p)
    sol = sim.solve(t_eval=[t[0], t[-1]], t_interp=t,
                    initial_soc=float(window.get("soc0", 1.0)))
    vol = p["Cell volume [m3]"]
    Q = np.asarray(sol["Volume-averaged total heating [W.m-3]"].entries) * vol
    Q_contact = np.asarray(window["I"], float) ** 2 * theta.R_contact
    U = np.asarray(sol["Bulk open-circuit voltage [V]"].entries)
    V = np.asarray(sol["Terminal voltage [V]"].entries)
    ideal = np.abs(window["I"]) * (U - V) * np.sign(U - V)
    T = np.asarray(sol["Volume-averaged cell temperature [C]"].entries)

    # re-integrate the lumped ODE from the twin's own heat
    hA = theta.h_conv * CELL_AREA_M2
    # effective heat capacity, read off the twin's own response
    dt = float(t[1] - t[0])
    dTdt = np.gradient(T, dt)
    net = (Q + Q_contact) - hA * (T - window["T_amb"])
    C_eff = float(np.sum(net * dTdt) / np.sum(dTdt ** 2))
    T_re = np.empty_like(T); T_re[0] = T[0]
    for k in range(1, len(T)):
        T_re[k] = T_re[k - 1] + dt / C_eff * (
            Q[k - 1] + Q_contact[k - 1] - hA * (T_re[k - 1] - window["T_amb"]))
    return {
        "Q_pybamm_W": float(np.median(Q)),
        "Q_contact_W": float(np.median(Q_contact)),
        "Q_terminal_balance_W": float(np.median(ideal)),
        "closure": float(np.median(Q + Q_contact) / np.median(ideal)),
        "C_eff_J_per_K": C_eff,
        "hA_W_per_K": hA,
        "tau_s": C_eff / hA,
        "lumped_reintegration_max_err_K": float(np.max(np.abs(T_re - T))),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--out", default="runs/verify.json")
    args = ap.parse_args()

    out = {}
    for phase, data, fit in (("discharge", "data/severson_discharge.npz",
                              "runs/twin_fit.json"),
                             ("charge", "data/severson_charge.npz",
                              "runs/twin_fit_charge.json"),
                             ("nasa_discharge", "data/nasa_discharge.npz",
                              "runs/twin_fit_nasa.json")):
        if not Path(fit).exists():
            continue
        d = load_severson_windows(data)
        cfg = json.loads(Path(fit).read_text())
        chem = cfg.get("chemistry", "Prada2013")
        theta = TwinParams(**cfg["theta"])
        idx = subsample(windows_of(d, cell_split(len(d["cells"]))["test"]),
                        args.n, seed=11)
        rows = [energy_balance(window_dict(d, i), theta, chemistry=chem)
                for i in idx]
        out[phase] = {k: round(float(np.median([r[k] for r in rows])), 5)
                      for k in rows[0]}
        out[phase]["n_windows"] = len(rows)

        # the staggered fault coupling, at the step used and at a quarter of it
        w = window_dict(d, int(idx[0]))
        f = InternalShort(R0_ohm=20.0, onset_s=50.0)
        dt = float(d["t"][1] - d["t"][0])
        a = simulate_short(w, theta, f, dt=dt, chemistry=chem)
        b = simulate_short(w, theta, f, dt=dt / 4, chemistry=chem)
        out[phase]["fault_coupling_step_vs_quarter_step_max_K"] = round(
            float(np.nanmax(np.abs(a["T"] - b["T"]))), 5)
        out[phase]["chemistry"] = chem
        out[phase]["borrowed_parameters"] = len(borrowed_keys(chem))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
