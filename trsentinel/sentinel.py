"""Physics-residual + conformal early-warning for thermal runaway.

The digital twin predicts the temperature a *healthy* cell should show given
its own current and voltage. The residual (measured - predicted) is scored
against a conformal band calibrated on a healthy fleet, giving a
distribution-free early-warning flag with a controlled false-alarm rate.
"""
from __future__ import annotations
import numpy as np


class ThermalTwin:
    """Predict expected temperature rise from ohmic heating (healthy model).

    Fits lumped params (effective R/Cth/h) by least squares on healthy cells,
    then predicts T from I and ambient. No fault term — so a real fault shows
    up as a growing positive residual.
    """

    def __init__(self, dt=1.0, T_amb=25.0):
        self.dt, self.T_amb = dt, T_amb
        self.R = self.h = self.Cth = None

    def _features(self, cell):
        I, T = cell["I"], cell["T"]
        dT = np.gradient(T, self.dt)
        # Cth*dT/dt = I^2 R - h (T - T_amb)  ->  linear in [R, h]
        A = np.column_stack([I ** 2, -(T - self.T_amb)])
        return A, dT

    def fit(self, healthy_cells):
        A = np.vstack([self._features(c)[0] for c in healthy_cells])
        b = np.concatenate([self._features(c)[1] for c in healthy_cells])
        # absorb Cth into the coefficients: dT = (R/Cth) I^2 - (h/Cth)(T-Tamb)
        coef, *_ = np.linalg.lstsq(A, b, rcond=None)
        self.R_over_C, self.h_over_C = coef
        return self

    def predict_T(self, cell):
        I, T = cell["I"], cell["T"]
        T_pred = np.empty_like(T)
        T_pred[0] = T[0]
        for k in range(1, len(T)):
            dT = self.R_over_C * I[k - 1] ** 2 - self.h_over_C * (
                T_pred[k - 1] - self.T_amb
            )
            T_pred[k] = T_pred[k - 1] + self.dt * dT
        return T_pred

    def residual(self, cell):
        return cell["T"] - self.predict_T(cell)

    # -- real-data interface --------------------------------------------
    # The synthetic path above assumes one global ambient and dt=1 s. Real
    # windows carry their own ambient and time grid, so the fit and the
    # rollout take them per window. The information the model is given is
    # exactly the information the PyBAMM twin gets: I(t), T_amb, and T(0).

    @staticmethod
    def _rollout(I, T0, T_amb, dt, R_over_C, h_over_C):
        """Forward-Euler rollout, vectorised over a batch of windows."""
        I = np.atleast_2d(np.asarray(I, float))
        T0 = np.atleast_1d(np.asarray(T0, float))
        T_amb = np.atleast_1d(np.asarray(T_amb, float))
        out = np.empty_like(I)
        out[:, 0] = T0
        for k in range(1, I.shape[1]):
            out[:, k] = out[:, k - 1] + dt * (
                R_over_C * I[:, k - 1] ** 2
                - h_over_C * (out[:, k - 1] - T_amb))
        return out

    def fit_windows(self, windows):
        """Fit (R/C, h/C) by minimising the *rollout* error, not dT/dt error.

        Regressing on a finite-difference dT/dt is the quick way, but it scores
        a one-step-ahead prediction while the twin is used as a free-running
        simulator; the PyBAMM twin is scored on its trajectory, so this one is
        too, and the comparison between them is then like for like. The
        finite-difference solution is used as the starting point.
        """
        from scipy.optimize import least_squares

        dt = float(windows[0]["t"][1] - windows[0]["t"][0])
        I = np.array([w["I"] for w in windows], float)
        T = np.array([w["T"] for w in windows], float)
        T_amb = np.array([w["T_amb"] for w in windows], float)

        A = np.column_stack([I.ravel() ** 2, -(T - T_amb[:, None]).ravel()])
        b = np.gradient(T, dt, axis=1).ravel()
        x0, *_ = np.linalg.lstsq(A, b, rcond=None)

        def resid(x):
            return (self._rollout(I, T[:, 0], T_amb, dt, x[0], x[1]) - T).ravel()

        sol = least_squares(resid, np.maximum(x0, 1e-8), xtol=1e-12, ftol=1e-12)
        self.R_over_C, self.h_over_C = float(sol.x[0]), float(sol.x[1])
        return self

    def predict_window(self, w):
        """Forward-Euler rollout of the fitted lumped ODE from T(0)."""
        dt = float(w["t"][1] - w["t"][0])
        return self._rollout(w["I"], w["T"][0], w["T_amb"], dt,
                             self.R_over_C, self.h_over_C)[0]

    def to_dict(self):
        return {"R_over_C_K_per_J": getattr(self, "R_over_C", None),
                "h_over_C_per_s": getattr(self, "h_over_C", None)}


class ConformalBand:
    """Distribution-free upper band from healthy-cell residuals.

    Pointwise version, kept for the synthetic demo. `WindowConformal` below is
    what the real-data pipeline uses: it controls the false-alarm rate *per
    window*, which is the quantity an operator actually cares about.
    """

    def __init__(self, alpha=0.01):
        self.alpha, self.hi = alpha, None

    def calibrate(self, residuals):
        r = np.concatenate([np.abs(x) for x in residuals])
        self.hi = np.quantile(r, 1 - self.alpha)
        return self

    def exceed(self, residual):
        return residual > self.hi


def persistence_statistic(residual, persist=6):
    """Largest residual level held for `persist` consecutive samples.

    `alarm(q) == (persistence_statistic(r, p) > q)` for every threshold q, so
    one scalar per window summarises the whole persistence-filtered alarm rule
    and can be used directly as a conformal nonconformity score.
    """
    r = np.asarray(residual, float)
    if persist <= 1:
        return float(np.nanmax(r))
    n = len(r) - persist + 1
    if n <= 0:
        return float(np.nanmin(r))
    windows = np.lib.stride_tricks.sliding_window_view(r, persist)
    return float(np.nanmax(np.nanmin(windows, axis=1)))


def alarm_index(residual, threshold, persist=6):
    """First index at which the residual has stayed above `threshold`."""
    over = np.asarray(residual, float) > threshold
    run = 0
    for k, o in enumerate(over):
        run = run + 1 if o else 0
        if run >= persist:
            return k - persist + 1
    return None


class WindowConformal:
    """Split-conformal alarm threshold with a per-window false-alarm guarantee.

    The nonconformity score of a window is `persistence_statistic(residual)`.
    With `n` exchangeable healthy calibration windows, the threshold

        q = the ceil((n+1)(1-alpha))-th smallest calibration score

    gives `P(alarm on a new healthy window) <= alpha` -- a finite-sample
    statement, no distributional assumption. Exchangeability across *cells* is
    the part that has to be checked empirically, and `scripts/eval_far.py`
    checks it.
    """

    def __init__(self, alpha=0.05, persist=6):
        self.alpha, self.persist = alpha, persist
        self.q = None
        self.n_cal = 0

    def calibrate(self, residuals):
        scores = np.array([persistence_statistic(r, self.persist)
                           for r in residuals], float)
        n = len(scores)
        k = int(np.ceil((n + 1) * (1 - self.alpha)))
        self.n_cal = n
        self.q = float(np.inf) if k > n else float(np.sort(scores)[k - 1])
        self.cal_scores = scores
        return self

    def score(self, residual):
        return persistence_statistic(residual, self.persist)

    def alarms(self, residual):
        return self.score(residual) > self.q

    def alarm_index(self, residual):
        return alarm_index(residual, self.q, self.persist)


def early_warning(residual, band, persist=20):
    """Flag the first index where the residual stays above band `persist` steps."""
    over = band.exceed(residual).astype(int)
    run = 0
    for k, o in enumerate(over):
        run = run + 1 if o else 0
        if run >= persist:
            return k - persist + 1
    return None
