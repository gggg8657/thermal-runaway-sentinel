"""A physically parameterised internal short, injected into the PyBAMM twin.

**Everything in this module is simulation.** There is no labelled internal-short
event in any public cycling dataset this project could reach, so the fault is
modelled rather than observed, and every number that depends on it is labelled
as simulated where it is reported.

The fault is the standard equivalent-circuit picture of an internal soft short
(Feng et al., *J. Power Sources* 2016): a resistance `R_s` bridging the
electrodes. Two things follow, and both are applied:

1. **A shunt current** `I_s = V / R_s` is drawn *inside* the cell, on top of
   whatever the cycler applies. The twin sees `I_applied + I_s`, so the extra
   electrode kinetics, diffusion and ohmic losses that current causes are
   modelled by PyBAMM rather than assumed.
2. **Ohmic heat** `Q_s = V * I_s = V^2 / R_s` is released at the short itself.
   That power is external to PyBAMM's own heat-generation terms, so it is added
   to the lumped energy balance. In the lumped model the balance is
   `C dT/dt = Q_pybamm + Q_s - hA (T - T_amb)`, which is identical to
   `C dT/dt = Q_pybamm - hA (T - T_amb_eff)` with
   `T_amb_eff = T_amb + Q_s / (hA)` -- so the injection is done exactly, through
   the ambient temperature, with no change to PyBAMM's thermal submodel.

`I_s` and `Q_s` depend on `V`, which depends on them, so the two are coupled
with a staggered explicit scheme: each 5 s step uses the voltage at the start of
the step. `scripts/verify_twin.py` reruns one window at a quarter of the step
and reports the difference, so the size of that approximation is measured
rather than assumed.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

from .twin_pybamm import CELL_AREA_M2, PybammTwin, TwinParams, build_model


@dataclass
class InternalShort:
    """Resistance across the electrodes, optionally worsening with time.

    `R_end_ohm` / `tau_s` describe a short that grows: dendrite penetration or a
    separator melt-through does not hold a fixed resistance. With `R_end_ohm`
    left at None the short is a constant resistance, which is the cleaner
    single-number sweep.
    """

    R0_ohm: float = 100.0
    onset_s: float = 200.0
    R_end_ohm: float | None = None
    tau_s: float = 200.0

    def resistance(self, t):
        t = np.asarray(t, float)
        R = np.full(t.shape, np.inf)
        on = t >= self.onset_s
        if self.R_end_ohm is None:
            R[on] = self.R0_ohm
        else:
            dt = t[on] - self.onset_s
            R[on] = self.R_end_ohm + (self.R0_ohm - self.R_end_ohm) * np.exp(-dt / self.tau_s)
        return R

    def to_dict(self):
        d = asdict(self)
        d["kind"] = "constant" if self.R_end_ohm is None else "exponentially worsening"
        return d


def simulate_short(window, theta: TwinParams, short: InternalShort | None,
                   kind="SPMe", dt=None, initial_soc=None,
                   chemistry="Prada2013"):
    """Run the twin over one window with (or without) an internal short.

    Returns `V, T, I_short, Q_short, t` on the window's own time grid. With
    `short=None` this is the healthy twin run through the same stepping code,
    so a healthy/faulted difference is not contaminated by a change of solver
    path.
    """
    import pybamm

    t = np.asarray(window["t"], float)
    step = float(t[1] - t[0]) if dt is None else float(dt)
    if initial_soc is None:
        initial_soc = float(window.get("soc0", 1.0))
    tw = PybammTwin(kind, chemistry=chemistry)
    p = tw._params_for(theta, t, window["I"], window["T_amb"], window["T0"],
                       window["Qd_cycle"], initial_soc)
    p["Current function [A]"] = p["Current function [A]"] + pybamm.InputParameter("I_short")
    # _params_for already folded the contact-resistance heat into the ambient
    p["Ambient temperature [K]"] = (p["Ambient temperature [K]"]
                                    + pybamm.InputParameter("dT_amb"))
    hA = theta.h_conv * CELL_AREA_M2

    sim = pybamm.Simulation(build_model(kind), parameter_values=p)
    n_steps = int(round((t[-1] - t[0]) / step))
    ts = t[0] + step * np.arange(n_steps + 1)
    R_of_t = short.resistance(ts) if short is not None else np.full(ts.shape, np.inf)

    V_now = float(window["V"][0])
    I_s_hist, Q_s_hist = [0.0], [0.0]
    sol = None
    try:
        for k in range(n_steps):
            R = R_of_t[k]
            I_s = 0.0 if not np.isfinite(R) else max(V_now, 0.0) / R
            Q_s = V_now * I_s
            sol = sim.step(step, inputs={"I_short": I_s, "dT_amb": Q_s / hA},
                           save=True)
            V_now = float(sol["Terminal voltage [V]"].entries[-1])
            I_s_hist.append(I_s); Q_s_hist.append(Q_s)
        V = np.asarray(sol["Terminal voltage [V]"].entries, float)
        T = np.asarray(sol["Volume-averaged cell temperature [C]"].entries, float)
        tt = np.asarray(sol.t, float)
        ok = True
    except Exception as exc:                                  # solver failure
        return {"ok": False, "termination": f"FAILED: {type(exc).__name__}",
                "V": np.full(len(t), np.nan), "T": np.full(len(t), np.nan),
                "I_short": np.full(len(t), np.nan),
                "Q_short": np.full(len(t), np.nan), "t": t}

    # sim.step(save=True) repeats the joining time point between steps
    return {
        "ok": ok, "termination": "final time",
        "V": np.interp(t, tt, V), "T": np.interp(t, tt, T),
        "I_short": np.interp(t, ts, I_s_hist),
        "Q_short": np.interp(t, ts, Q_s_hist), "t": t,
    }
