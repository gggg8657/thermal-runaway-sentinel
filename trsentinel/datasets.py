"""The three real-cell window sets this project evaluates on, in one place.

Everything downstream -- cost benchmark, figures, report -- iterates over this
dict, so adding a dataset means adding an entry and nothing else.
"""
from __future__ import annotations

DATASETS = {
    "discharge": {
        "label": "Severson 4C discharge",
        "sublabel": "identical duty cycle for every cell",
        "corpus": "Severson (TRI) batch1, LFP 18650",
        "data": "data/severson_discharge.npz",
        "prepare_report": "data/prepare_report_discharge.json",
        "fit": "runs/twin_fit.json",
        "far": "runs/far.json",
        "leadtime": "runs/leadtime.json",
        "residuals": "runs/residuals.npz",
        "chemistry": "Prada2013",
    },
    "charge": {
        "label": "Severson fast charge",
        "sublabel": "each cell's own 3.6C-8C policy",
        "corpus": "Severson (TRI) batch1, LFP 18650",
        "data": "data/severson_charge.npz",
        "prepare_report": "data/prepare_report_charge.json",
        "fit": "runs/twin_fit_charge.json",
        "far": "runs/far_charge.json",
        "leadtime": "runs/leadtime_charge.json",
        "residuals": "runs/residuals_charge.npz",
        "chemistry": "Prada2013",
    },
    "nasa": {
        "label": "NASA PCoE discharge",
        "sublabel": "1-4 A, chamber at 4 / 24 / 43 C",
        "corpus": "NASA Ames PCoE, LCO 18650",
        "data": "data/nasa_discharge.npz",
        "prepare_report": "data/prepare_report_nasa.json",
        "fit": "runs/twin_fit_nasa.json",
        "far": "runs/far_nasa.json",
        "leadtime": "runs/leadtime_nasa.json",
        "residuals": "runs/residuals_nasa.npz",
        "chemistry": "Ramadass2004",
    },
}
