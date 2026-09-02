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


class ConformalBand:
    """Distribution-free upper band from healthy-cell residuals."""

    def __init__(self, alpha=0.01):
        self.alpha, self.hi = alpha, None

    def calibrate(self, residuals):
        r = np.concatenate([np.abs(x) for x in residuals])
        self.hi = np.quantile(r, 1 - self.alpha)
        return self

    def exceed(self, residual):
        return residual > self.hi


def early_warning(residual, band, persist=20):
    """Flag the first index where the residual stays above band `persist` steps."""
    over = band.exceed(residual).astype(int)
    run = 0
    for k, o in enumerate(over):
        run = run + 1 if o else 0
        if run >= persist:
            return k - persist + 1
    return None
