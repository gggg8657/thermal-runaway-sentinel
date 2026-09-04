"""PyBAMM electro-thermal digital twin for the sentinel.

The twin is a **single-particle model with electrolyte (SPMe)** coupled to a
**lumped thermal** energy balance, run on the LFP/graphite chemistry of
`Prada2013` -- the same chemistry as the A123 cells in the Severson dataset.
Given a measured current profile, an ambient temperature and a starting cell
temperature, it predicts the terminal voltage and the can temperature a
*healthy* cell should show. Nothing about the measured temperature enters the
prediction except its value at t=0, which is why the residual
`T_measured - T_twin` is informative about abnormal heat.

Four scalar parameters are identified against real cycles (`scripts/fit_twin.py`):

===========  ====================================================================
`k_area`     electrode-area multiplier; through the current density it sets the
             cell's polarisation at 4C
`k_cap`      lithium inventory per amp-hour of *measured* discharge capacity. The
             capacity of each window is a measurement, not a fit: the electrode
             maximum concentrations are scaled by `k_cap * Qd_cycle`, so an aged
             cell is modelled with the inventory it actually still has
`R_contact`  series contact resistance [Ohm]
`h_conv`     total heat transfer coefficient [W.m-2.K-1] to the 30 C chamber
`c_scale`    multiplier on the component specific heat capacities, i.e. the
             cell's effective thermal mass
===========  ====================================================================

`Prada2013` carries no thermal or current-collector data, so those entries are
borrowed from `Chen2020` (generic Li-ion material properties) -- see
`BORROWED_FROM_CHEN2020`. Cell volume and cooling area are *not* borrowed: they
are the true 18650 can geometry.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

#: 18650 can: 18 mm diameter, 65 mm height
CELL_R_M, CELL_H_M = 0.009, 0.065
CELL_VOLUME_M3 = np.pi * CELL_R_M ** 2 * CELL_H_M
CELL_AREA_M2 = 2 * np.pi * CELL_R_M * CELL_H_M + 2 * np.pi * CELL_R_M ** 2

#: Prada2013 has no thermal/current-collector entries; these come from Chen2020.
BORROWED_FROM_CHEN2020 = [
    "Negative current collector thickness [m]",
    "Negative current collector conductivity [S.m-1]",
    "Positive current collector thickness [m]",
    "Positive current collector conductivity [S.m-1]",
    "Negative current collector density [kg.m-3]",
    "Negative current collector specific heat capacity [J.kg-1.K-1]",
    "Negative electrode density [kg.m-3]",
    "Negative electrode specific heat capacity [J.kg-1.K-1]",
    "Separator density [kg.m-3]",
    "Separator specific heat capacity [J.kg-1.K-1]",
    "Positive electrode density [kg.m-3]",
    "Positive electrode specific heat capacity [J.kg-1.K-1]",
    "Positive current collector density [kg.m-3]",
    "Positive current collector specific heat capacity [J.kg-1.K-1]",
]
_HEAT_CAPACITY_KEYS = [k for k in BORROWED_FROM_CHEN2020 if "specific heat" in k]

#: Prada2013 electrode height at its native 2.3 Ah rating
_REF_HEIGHT_M = 0.6


@dataclass
class TwinParams:
    k_area: float = 0.80        # x reference electrode height
    k_cap: float = 0.45         # x Qd_cycle -> electrode max-concentration scale
    R_contact: float = 0.012    # Ohm
    h_conv: float = 30.0        # W.m-2.K-1
    c_scale: float = 1.0        # x component specific heat capacities

    NAMES = ("k_area", "k_cap", "R_contact", "h_conv", "c_scale")

    def as_vector(self):
        return np.array([self.k_area, self.k_cap, self.R_contact,
                         self.h_conv, self.c_scale])

    @classmethod
    def from_vector(cls, v):
        return cls(*(float(x) for x in v))

    def to_dict(self):
        return asdict(self)


def base_parameter_values():
    """Prada2013 (LFP/graphite) completed with Chen2020 thermal properties."""
    import pybamm

    p = pybamm.ParameterValues("Prada2013")
    donor = pybamm.ParameterValues("Chen2020")
    for k in BORROWED_FROM_CHEN2020:
        p[k] = donor[k]
    p["Cell volume [m3]"] = CELL_VOLUME_M3
    p["Cell cooling surface area [m2]"] = CELL_AREA_M2
    p["Nominal cell capacity [A.h]"] = 1.1
    p["Lower voltage cut-off [V]"] = 1.6
    p["Upper voltage cut-off [V]"] = 4.0
    return p


_QUIET = False


def build_model(kind="SPMe"):
    global _QUIET
    import pybamm

    if not _QUIET:                      # 60 workers x 1200 windows of warnings
        pybamm.set_logging_level("ERROR")
        _QUIET = True
    opts = {"thermal": "lumped", "contact resistance": "true"}
    cls = {"SPM": pybamm.lithium_ion.SPM, "SPMe": pybamm.lithium_ion.SPMe,
           "DFN": pybamm.lithium_ion.DFN}[kind]
    return cls(options=opts)


class PybammTwin:
    """SPMe + lumped thermal, driven by a measured current trace."""

    def __init__(self, kind="SPMe", theta: TwinParams | None = None):
        self.kind = kind
        self.theta = theta or TwinParams()
        self._base = None

    # -- internals -------------------------------------------------------
    def _params_for(self, theta, t, I_meas, T_amb_C, T0_C, Qd_Ah,
                    short_ohm=None, short_onset_s=None):
        import pybamm

        if self._base is None:
            self._base = base_parameter_values()
        p = pybamm.ParameterValues(self._base)
        p["Electrode height [m]"] = _REF_HEIGHT_M * theta.k_area
        for el in ("negative", "positive"):
            k = f"Maximum concentration in {el} electrode [mol.m-3]"
            p[k] = self._base[k] * theta.k_cap * Qd_Ah
        p["Contact resistance [Ohm]"] = theta.R_contact
        p["Total heat transfer coefficient [W.m-2.K-1]"] = theta.h_conv
        for k in _HEAT_CAPACITY_KEYS:
            p[k] = self._base[k] * theta.c_scale
        p["Ambient temperature [K]"] = T_amb_C + 273.15
        p["Initial temperature [K]"] = T0_C + 273.15
        # PyBAMM signs current positive on discharge; the archive signs it negative
        p["Current function [A]"] = pybamm.Interpolant(
            t, -np.asarray(I_meas, float), pybamm.t, name="I_applied",
            interpolator="linear")
        return p

    # -- public ----------------------------------------------------------
    def simulate(self, window, theta: TwinParams | None = None, initial_soc=None):
        """Predict V and T over one measured window.

        `window` is a dict with `t, I, T_amb, T0` (and anything else, ignored).
        Returns `dict(V, T, ok, termination, n_t)`; on a solver failure `ok` is
        False and the traces are padded with their last solved value, so a
        caller can still score the window instead of silently dropping it.
        """
        import pybamm

        theta = theta or self.theta
        if initial_soc is None:
            initial_soc = float(window.get("soc0", 1.0))
        t = np.asarray(window["t"], float)
        p = self._params_for(theta, t, window["I"], window["T_amb"],
                             window["T0"], window["Qd_cycle"])
        model = build_model(self.kind)
        sim = pybamm.Simulation(model, parameter_values=p)
        try:
            sol = sim.solve(t_eval=[t[0], t[-1]], t_interp=t,
                            initial_soc=initial_soc)
            V = np.asarray(sol["Terminal voltage [V]"].entries, float)
            T = np.asarray(sol["Volume-averaged cell temperature [C]"].entries,
                           float)
            term = str(sol.termination)
        except Exception as exc:                       # solver / model failure
            V = np.array([]); T = np.array([]); term = f"FAILED: {type(exc).__name__}"
        n = len(t)
        ok = len(V) == n
        if len(V) == 0:
            V = np.full(n, np.nan); T = np.full(n, np.nan)
        elif len(V) < n:
            V = np.concatenate([V, np.full(n - len(V), V[-1])])
            T = np.concatenate([T, np.full(n - len(T), T[-1])])
        return {"V": V, "T": T, "ok": bool(ok), "termination": term,
                "n_solved": int(min(len(V), n))}
