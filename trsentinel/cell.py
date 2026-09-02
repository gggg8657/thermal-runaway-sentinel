"""Reduced-order Li-ion cell simulator (electro-thermal).

A lumped equivalent-circuit + thermal-ODE model, integrated with explicit
Euler. Good enough to generate physically-plausible V/I/T traces for both
healthy cells and cells developing an internal short (a thermal-runaway
precursor). The real project calibrates this against PyBAMM / NASA-PCoE data.
"""
from __future__ import annotations
import numpy as np


def ocv(soc):
    """Open-circuit voltage vs state of charge (smooth NMC-like curve)."""
    return 3.2 + 0.9 * soc - 0.25 * (1 - soc) ** 2


def simulate(
    n=1800, dt=1.0, R0=0.045, Cth=45.0, h=0.55, T_amb=25.0, cap_Ah=2.0,
    fault=None, seed=0,
):
    """Simulate one cell under a repeating charge/discharge current.

    fault: None (healthy) or dict(t_onset=int, q_rate=float, R_growth=float)
      describing an internal short that injects heat and grows resistance.
    Returns dict of arrays: t, I, V, T, soc.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n) * dt
    # +/- 1C current with rest phases
    period = 300
    I = np.where((t % period) < period / 2, cap_Ah, -cap_Ah)
    I[(t % period > period - 30)] = 0.0

    soc = 0.6
    T = T_amb
    V, Tarr, socarr = np.zeros(n), np.zeros(n), np.zeros(n)
    q_short = np.zeros(n)
    R = R0
    for k in range(n):
        if fault and k >= fault["t_onset"]:
            prog = (k - fault["t_onset"]) / max(n - fault["t_onset"], 1)
            R = R0 * (1 + fault.get("R_growth", 0.0) * prog)
            q_short[k] = fault.get("q_rate", 0.0) * prog  # W, grows in
        soc = np.clip(soc + I[k] * dt / (cap_Ah * 3600), 0.02, 0.99)
        v = ocv(soc) - I[k] * R
        heat = I[k] ** 2 * R + q_short[k]
        T = T + dt / Cth * (heat - h * (T - T_amb))
        V[k] = v + rng.normal(0, 0.004)
        Tarr[k] = T + rng.normal(0, 0.05)
        socarr[k] = soc
    return {"t": t, "I": I, "V": V, "T": Tarr, "soc": socarr}


def healthy_fleet(n_cells=30, length=1800, seed=0):
    return [simulate(n=length, seed=s) for s in range(seed, seed + n_cells)]


def faulty_cell(length=1800, t_onset=1100, q_rate=6.0, R_growth=2.5, seed=999):
    return simulate(
        n=length, seed=seed,
        fault={"t_onset": t_onset, "q_rate": q_rate, "R_growth": R_growth},
    )
