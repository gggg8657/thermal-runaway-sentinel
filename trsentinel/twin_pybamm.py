"""PyBAMM electro-thermal digital twin for the sentinel.

The twin is a **single-particle model with electrolyte (SPMe)** coupled to a
**lumped thermal** energy balance, run on the LFP/graphite chemistry of
`Prada2013` -- the same chemistry as the A123 cells in the Severson dataset.
Given a measured current profile, an ambient temperature and a starting cell
temperature, it predicts the terminal voltage and the can temperature a
*healthy* cell should show. Nothing about the measured temperature enters the
prediction except its value at t=0, which is why the residual
`T_measured - T_twin` is informative about abnormal heat.

Six scalar parameters are identified against real cycles (`scripts/fit_twin.py`):

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
`dUdT`       full-cell entropic coefficient [mV.K-1], added to the positive
             electrode's OCP entropic change. `Prada2013` sets both electrodes'
             entropic change to zero, so without this the twin has no reversible
             heat at all -- and the measured cells visibly *cool* for the first
             minute of a fast charge, which is exactly that term
===========  ====================================================================

`Prada2013` carries no thermal or current-collector data, so those entries are
borrowed from `Chen2020` (generic Li-ion material properties) -- see
`borrowed_keys()`. Cell volume and cooling area are *not* borrowed: they are the
true 18650 can geometry.

Two heat terms that PyBAMM leaves out by default are put back, because both are
the same size as the faults this project is trying to detect:

* **Heat of mixing** (`"heat of mixing": "true"`). Without it the SPMe's heat
  generation is ~0.35 W against an `I (U - V)` energy balance of ~0.71 W at 4C.
* **Contact-resistance ohmic heat** `I^2 R_contact`. The contact resistance
  option adds the voltage drop but not the dissipation. It is known ahead of
  time from the measured current, so it is added to the lumped balance exactly,
  through the ambient term -- the same device `trsentinel/fault.py` uses:
  `C dT/dt = Q + Q_c - hA(T - T_amb)` is `C dT/dt = Q - hA(T - T_amb - Q_c/hA)`.

`scripts/verify_twin.py` reports how well the balance closes after both.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

#: 18650 can: 18 mm diameter, 65 mm height
CELL_R_M, CELL_H_M = 0.009, 0.065
CELL_VOLUME_M3 = np.pi * CELL_R_M ** 2 * CELL_H_M
CELL_AREA_M2 = 2 * np.pi * CELL_R_M * CELL_H_M + 2 * np.pi * CELL_R_M ** 2

#: Chemistry sets that carry no thermal or current-collector data get those
#: entries filled from Chen2020 (generic Li-ion material properties). Which keys
#: were borrowed is discovered by building the model and is reported, not
#: hidden -- see `borrowed_keys()`.
DONOR = "Chen2020"
_HEAT_CAPACITY_KEYS = [
    f"{d} specific heat capacity [J.kg-1.K-1]"
    for d in ("Negative current collector", "Negative electrode", "Separator",
              "Positive electrode", "Positive current collector")
]
_BORROWED: dict[str, list[str]] = {}

@dataclass
class TwinParams:
    k_area: float = 0.80        # x reference electrode height
    k_cap: float = 0.45         # x Qd_cycle -> electrode max-concentration scale
    R_contact: float = 0.012    # Ohm
    h_conv: float = 30.0        # W.m-2.K-1
    c_scale: float = 1.0        # x component specific heat capacities
    dUdT_mV_per_K: float = 0.0  # full-cell entropic coefficient

    NAMES = ("k_area", "k_cap", "R_contact", "h_conv", "c_scale",
             "dUdT_mV_per_K")
    #: physically defensible search box; a fit that lands on a bound is flagged
    LO = np.array([0.3, 0.4, 1e-4, 5.0, 0.4, -1.0])
    HI = np.array([2.5, 2.5, 6e-2, 300.0, 8.0, 1.0])

    def as_vector(self):
        return np.array([self.k_area, self.k_cap, self.R_contact,
                         self.h_conv, self.c_scale, self.dUdT_mV_per_K])

    @classmethod
    def from_vector(cls, v):
        return cls(*(float(x) for x in v))

    def to_dict(self):
        return asdict(self)


def base_parameter_values(chemistry="Prada2013", nominal_Ah=1.1,
                          v_lo=1.6, v_hi=4.0):
    """A chemistry set completed with whatever thermal data it is missing.

    Cell volume and cooling area are *not* borrowed: they are the true 18650 can
    geometry, which both datasets this repo uses are built from.
    """
    import pybamm
    import re as _re

    p = pybamm.ParameterValues(chemistry)
    donor = pybamm.ParameterValues(DONOR)
    borrowed = []
    for _ in range(80):                    # fill until the model can be built
        try:
            pybamm.Simulation(build_model("SPM"),
                              parameter_values=pybamm.ParameterValues(p)).build()
            break
        except KeyError as exc:
            m = _re.search(r"Parameter '([^']+)' not found", str(exc))
            if not m or m.group(1) not in donor:
                raise
            p[m.group(1)] = donor[m.group(1)]
            borrowed.append(m.group(1))
    _BORROWED[chemistry] = borrowed
    p["Cell volume [m3]"] = CELL_VOLUME_M3
    p["Cell cooling surface area [m2]"] = CELL_AREA_M2
    p["Nominal cell capacity [A.h]"] = nominal_Ah
    p["Lower voltage cut-off [V]"] = v_lo
    p["Upper voltage cut-off [V]"] = v_hi
    return p


_ENTROPY_KEY = "Positive electrode OCP entropic change [V.K-1]"


def _add_entropic_change(p, base, dUdT):
    """Add a constant entropic coefficient on top of whatever the set carries.

    For a chemistry with no entropy data (Prada2013) this *is* the entropic
    coefficient; for one that has a literature curve (Ramadass2004) it is a
    correction to it, and the fit is free to leave it at zero.
    """
    import inspect

    b = base[_ENTROPY_KEY]
    if not callable(b):
        p[_ENTROPY_KEY] = float(b) + dUdT
        return
    n = len(inspect.signature(b).parameters)
    if n == 1:
        p[_ENTROPY_KEY] = lambda sto, _b=b, _d=dUdT: _b(sto) + _d
    else:
        p[_ENTROPY_KEY] = lambda sto, c, _b=b, _d=dUdT: _b(sto, c) + _d


def borrowed_keys(chemistry="Prada2013"):
    """Which parameters this chemistry had to borrow from the donor set."""
    if chemistry not in _BORROWED:
        base_parameter_values(chemistry)
    return list(_BORROWED[chemistry])


_QUIET = False


def build_model(kind="SPMe"):
    global _QUIET
    import pybamm

    if not _QUIET:                      # 60 workers x 1200 windows of warnings
        pybamm.set_logging_level("ERROR")
        _QUIET = True
    opts = {"thermal": "lumped", "contact resistance": "true",
            "heat of mixing": "true"}
    cls = {"SPM": pybamm.lithium_ion.SPM, "SPMe": pybamm.lithium_ion.SPMe,
           "DFN": pybamm.lithium_ion.DFN}[kind]
    return cls(options=opts)


class PybammTwin:
    """SPMe + lumped thermal, driven by a measured current trace."""

    def __init__(self, kind="SPMe", theta: TwinParams | None = None,
                 chemistry="Prada2013", nominal_Ah=1.1):
        self.kind = kind
        self.chemistry = chemistry
        self.nominal_Ah = nominal_Ah
        self.theta = theta or TwinParams()
        self._base = None

    # -- internals -------------------------------------------------------
    def _params_for(self, theta, t, I_meas, T_amb_C, T0_C, Qd_Ah,
                    short_ohm=None, short_onset_s=None):
        import pybamm

        if self._base is None:
            self._base = base_parameter_values(self.chemistry, self.nominal_Ah)
        p = pybamm.ParameterValues(self._base)
        p["Electrode height [m]"] = self._base["Electrode height [m]"] * theta.k_area
        for el in ("negative", "positive"):
            k = f"Maximum concentration in {el} electrode [mol.m-3]"
            p[k] = self._base[k] * theta.k_cap * Qd_Ah
        p["Contact resistance [Ohm]"] = theta.R_contact
        p["Total heat transfer coefficient [W.m-2.K-1]"] = theta.h_conv
        for k in _HEAT_CAPACITY_KEYS:
            p[k] = self._base[k] * theta.c_scale
        _add_entropic_change(p, self._base, theta.dUdT_mV_per_K * 1e-3)
        # I^2 R_contact is dissipated in the cell but is not in PyBAMM's heat
        # generation; it is known from the measured current, so it goes into
        # the lumped balance exactly, as an ambient offset.
        hA = theta.h_conv * CELL_AREA_M2
        q_contact = np.asarray(I_meas, float) ** 2 * theta.R_contact / hA
        p["Ambient temperature [K]"] = T_amb_C + 273.15 + pybamm.Interpolant(
            t, q_contact, pybamm.t, name="dT_contact", interpolator="linear")
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
