"""What one twin evaluation costs, so the deployment question has a number.

    PYTHONPATH=. python scripts/bench_cost.py

Single-core, no parallelism, on real windows: the wall clock to simulate one
window and the real-time factor that follows from it. Writes `runs/cost.json`.
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np

from trsentinel.data.severson import load_severson_windows, window_dict
from trsentinel.datasets import DATASETS
from trsentinel.fault import simulate_short
from trsentinel.sentinel import ThermalTwin
from trsentinel.split import subsample, windows_of
from trsentinel.split import cell_split
from trsentinel.twin_pybamm import PybammTwin, TwinParams


def timeit(fn, n):
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t0) / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--out", default="runs/cost.json")
    args = ap.parse_args()

    out = {"cpu": platform.processor() or platform.machine(),
           "note": "single core, no parallelism", "phases": {}}
    for phase, cfg in DATASETS.items():
        if not Path(cfg["fit"]).exists():
            continue
        d = load_severson_windows(cfg["data"])
        meta = json.loads(Path(cfg["fit"]).read_text())
        theta = TwinParams(**meta["theta"])
        chem = meta.get("chemistry", "Prada2013")
        idx = subsample(windows_of(d, cell_split(len(d["cells"]))["test"]),
                        args.n, seed=3)
        ws = [window_dict(d, i) for i in idx]
        tw = PybammTwin("SPMe", chemistry=chem)
        it = iter(ws * 3)
        t_pb = timeit(lambda: tw.simulate(next(it), theta), len(ws))
        rom = ThermalTwin().fit_windows(ws)
        it2 = iter(ws * 3)
        t_rom = timeit(lambda: rom.predict_window(next(it2)), len(ws))
        it3 = iter(ws * 3)
        t_step = timeit(lambda: simulate_short(next(it3), theta, None,
                                              chemistry=chem), len(ws))
        span = float(d["t"][-1])
        out["phases"][phase] = {
            "label": cfg["label"],
            "chemistry": chem,
            "window_s": span,
            "n_timesteps": int(len(d["t"])),
            "pybamm_twin_s_per_window": round(t_pb, 4),
            "pybamm_twin_realtime_factor": round(span / t_pb, 1),
            "pybamm_stepped_s_per_window": round(t_step, 4),
            "reduced_order_twin_s_per_window": round(t_rom, 6),
            "reduced_order_realtime_factor": round(span / t_rom, 1),
            "fleet_of_1000_cells_core_seconds_per_cycle": round(1000 * t_pb, 1),
        }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
