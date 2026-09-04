"""Render the README figures from the finished runs.

    PYTHONPATH=. python scripts/make_figures.py

Reads `runs/` + `data/` and writes PNGs into `assets/`. Every number plotted
comes from a JSON or npz produced by the evaluation scripts -- nothing is typed
in by hand. Figures whose content is simulated say so on the figure itself, not
only in the caption.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from trsentinel.data.severson import load_severson_windows, window_dict  # noqa: E402
from trsentinel.datasets import DATASETS  # noqa: E402
from trsentinel.metrics import per_cell_offsets  # noqa: E402
from trsentinel.split import cell_split, windows_of  # noqa: E402
from trsentinel.twin_pybamm import PybammTwin, TwinParams  # noqa: E402

plt.rcParams.update({"font.size": 9, "figure.dpi": 140, "savefig.bbox": "tight",
                     "axes.grid": True, "grid.alpha": 0.25})

C_MEAS, C_TWIN, C_ROM, C_RAW = "#222222", "#c1272d", "#1f6fb4", "#8a8a8a"
SIM_BOX = dict(boxstyle="round,pad=0.3", fc="#fff3cd", ec="#b8860b", lw=0.8)


def _sim_badge(ax, text="SIMULATED FAULT"):
    ax.text(0.99, 0.02, text, transform=ax.transAxes, ha="right", va="bottom",
            fontsize=7.5, bbox=SIM_BOX, zorder=10)


def fig_twin_fit(out, theta, keys, n=3):
    """Measured vs twin V and T on real held-out cells, every window set."""
    fig, axes = plt.subplots(2, len(keys), figsize=(4.5 * len(keys), 5.0),
                             squeeze=False)
    for col, phase in enumerate(keys):
        cfg = DATASETS[phase]
        d = load_severson_windows(cfg["data"])
        sp = cell_split(len(d["cells"]), seed=0)
        idx = windows_of(d, sp["test"])
        pick = idx[np.linspace(0, len(idx) - 1, n).astype(int)]
        tw = PybammTwin("SPMe", chemistry=cfg["chemistry"])
        for k, i in enumerate(pick):
            w = window_dict(d, i)
            r = tw.simulate(w, theta[phase])
            lab = f"{w['cell'].replace('batch1_', '')} cyc {w['cycle']}, |I|={np.abs(w['I']).mean():.1f} A"
            axes[0, col].plot(w["t"], w["V"], color=C_MEAS, lw=1.2,
                              label="measured" if k == 0 else None)
            axes[0, col].plot(w["t"], r["V"], color=C_TWIN, lw=1.0, ls="--",
                              label="PyBAMM twin" if k == 0 else None)
            axes[1, col].plot(w["t"], w["T"], color=C_MEAS, lw=1.2, label=lab)
            axes[1, col].plot(w["t"], r["T"], color=C_TWIN, lw=1.0, ls="--")
        axes[0, col].set_title(f"{cfg['label']}\n{cfg['sublabel']}", fontsize=9)
        axes[0, col].set_ylabel("terminal voltage [V]")
        axes[1, col].set_ylabel("cell temperature [$^\\circ$C]")
        axes[1, col].set_xlabel("time in window [s]")
        axes[0, col].legend(fontsize=7, loc="best")
        axes[1, col].legend(fontsize=6.5, loc="best")
    fig.suptitle("PyBAMM SPMe + lumped thermal twin against real Severson cells "
                 "(solid: measured, dashed: twin)", fontsize=9.5)
    fig.tight_layout()
    fig.savefig(out); plt.close(fig)


def fig_far(keys, out):
    """Empirical false-alarm rate on real healthy cells vs the nominal alpha."""
    fig, axes = plt.subplots(1, len(keys), figsize=(4.6 * len(keys), 3.9),
                             squeeze=False)
    styles = {"pybamm_spme": (C_TWIN, "o", "PyBAMM SPMe residual"),
              "rom_lumped": (C_ROM, "s", "reduced-order twin residual"),
              "no_model_dT_over_ambient": (C_RAW, "^", "no model: T - T$_{amb}$")}
    for ax, phase in zip(axes[0], keys):
        far = json.loads(Path(DATASETS[phase]["far"]).read_text())
        lim = [0.004, 0.32]
        ax.plot(lim, lim, color="k", lw=0.8, ls=":", label="conformal guarantee")
        for name, (c, m, lab) in styles.items():
            for variant, ls in (("cell_registered", "-"), ("fleet_calibrated", "--")):
                rows = far["variants"][name][variant]
                a = [r["alpha"] for r in rows]
                f = [max(r["far"], 1e-3) for r in rows]
                ax.plot(a, f, color=c, marker=m, ls=ls, ms=4, lw=1.2,
                        label=f"{lab}, {variant.replace('_', '-')}")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlim(*lim); ax.set_ylim(*lim)
        ax.set_xlabel("nominal false-alarm rate $\\alpha$")
        ax.set_ylabel("measured false-alarm rate, held-out real cells")
        ax.set_title(f"{DATASETS[phase]['label']}\n"
                     f"{far['split_sizes']['test']} unseen cells, "
                     f"{far['n_test_windows']} windows", fontsize=9)
    axes[0][0].legend(fontsize=6.2, loc="upper left")
    fig.suptitle("Real healthy cells only. Below the dotted line = the conformal "
                 "promise held out of sample.", fontsize=9)
    fig.tight_layout()
    fig.savefig(out); plt.close(fig)


def fig_residual(res_path, data_path, far_path, out, alpha=0.05):
    """What the detector actually watches: residual traces and their spread."""
    z = np.load(res_path)
    d = load_severson_windows(data_path)
    far = json.loads(Path(far_path).read_text())
    sp = cell_split(len(d["cells"]), seed=0)
    ci = z["cell_index"]
    res = z["res_pybamm"]
    offs, used = per_cell_offsets(res, ci, z["cycle"])
    test = windows_of(d, sp["test"]); test = test[~used[test]]
    q = [r for r in far["variants"]["pybamm_spme"]["cell_registered"]
         if r["alpha"] == alpha][0]["threshold_K"]

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6))
    reg = np.array([res[i] - offs[int(ci[i])] for i in test])
    for i in range(0, len(reg), max(len(reg) // 60, 1)):
        axes[0].plot(z["t"], reg[i], color=C_TWIN, lw=0.5, alpha=0.25)
    axes[0].axhline(q, color="k", ls="--", lw=1.1,
                    label=f"conformal threshold, $\\alpha$={alpha}: {q:.2f} K")
    axes[0].set_xlabel("time in window [s]")
    axes[0].set_ylabel("residual  T$_{meas}$ - T$_{twin}$  [K]")
    axes[0].set_title("real healthy held-out cells", fontsize=9)
    axes[0].legend(fontsize=7)

    for name, key, c in (("PyBAMM SPMe", "res_pybamm", C_TWIN),
                         ("reduced-order", "res_rom", C_ROM)):
        r = z[key][test]
        axes[1].hist(r.ravel(), bins=120, histtype="step", color=c, lw=1.2,
                     density=True, label=f"{name}  (rms {np.sqrt(np.nanmean(r**2)):.2f} K)")
    axes[1].set_xlabel("residual [K]"); axes[1].set_ylabel("density")
    axes[1].set_title("pointwise residual distribution", fontsize=9)
    axes[1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(out); plt.close(fig)


def fig_leadtime(keys, out, alpha=0.05, variant="cell_registered"):
    """Detectability and lead time vs short severity -- simulated fault."""
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8))
    marks = dict(zip(DATASETS, (("o", "-"), ("s", "--"), ("^", ":"))))
    for phase in keys:
        lt = json.loads(Path(DATASETS[phase]["leadtime"]).read_text())
        lab = DATASETS[phase]["label"]
        names = [k for k in lt["faults"] if "constant" in k]
        P = [lt["faults"][k]["peak_short_power_W"] for k in names]
        m, ls = marks[phase]
        det = [lt["faults"][k][f"{variant} alpha={alpha}"]["detection_rate_residual"]
               for k in names]
        detn = [lt["faults"][k][f"{variant} alpha={alpha}"]["detection_rate_no_model"]
                for k in names]
        axes[0].plot(P, det, color=C_TWIN, marker=m, ls=ls, lw=1.3,
                     label=f"{lab}: physics residual")
        axes[0].plot(P, detn, color=C_RAW, marker=m, ls=ls, lw=1.3,
                     label=f"{lab}: no model")
        lead = [lt["faults"][k][f"{variant} alpha={alpha}"]["median_lead_vs_no_model_s"]
                for k in names]
        axes[1].plot(P, [0 if v is None else v for v in lead], color=C_TWIN,
                     marker=m, ls=ls, lw=1.3, label=lab)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("short power at the end of the window [W]")
    axes[0].set_ylabel(f"detection rate at $\\alpha$={alpha}")
    axes[0].set_ylim(-0.03, 1.03)
    axes[0].set_title("can the short be seen at all?", fontsize=9)
    axes[0].legend(fontsize=7, loc="upper left")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("short power at the end of the window [W]")
    axes[1].set_ylabel("median lead time over the\nno-model alarm [s]")
    axes[1].set_title("how much earlier than a calibrated\ntemperature threshold?",
                      fontsize=9)
    axes[1].legend(fontsize=7)
    for ax in axes:
        _sim_badge(ax)
    fig.suptitle("Internal short injected into the twin, added to real measured "
                 "temperature traces (simulated fault, real noise floor)",
                 fontsize=9)
    fig.tight_layout(); fig.savefig(out); plt.close(fig)


def fig_trace(out, theta, phase="nasa", alpha=0.05):
    """One window, healthy and faulted, with the alarm the detector raises."""
    from trsentinel.fault import InternalShort, simulate_short

    cfg = DATASETS[phase]
    d = load_severson_windows(cfg["data"])
    sp = cell_split(len(d["cells"]), seed=0)
    idx = windows_of(d, sp["test"])
    i = int(idx[len(idx) // 2])
    w = window_dict(d, i)
    lt = json.loads(Path(cfg["leadtime"]).read_text())
    q = lt["thresholds_K"]["cell_registered"][str(alpha)]

    chem = cfg["chemistry"]
    healthy = simulate_short(w, theta, None, chemistry=chem)
    f = InternalShort(R0_ohm=20.0, onset_s=50.0)
    fault = simulate_short(w, theta, f, chemistry=chem)
    dT = fault["T"] - healthy["T"]
    T_obs = w["T"] + dT

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.5))
    axes[0].plot(w["t"], w["T"], color=C_MEAS, lw=1.4, label="measured (real cell)")
    axes[0].plot(w["t"], healthy["T"], color=C_TWIN, ls="--", lw=1.2, label="twin, healthy")
    axes[0].plot(w["t"], T_obs, color="#d1660f", lw=1.4,
                 label=f"measured + simulated short ({f.R0_ohm:.0f} $\\Omega$)")
    axes[0].axvline(f.onset_s, color="k", lw=0.8, ls=":")
    axes[0].set_xlabel("time in window [s]"); axes[0].set_ylabel("temperature [$^\\circ$C]")
    axes[0].legend(fontsize=7); axes[0].set_title(
        f"{cfg['label']}: {w['cell'].replace('batch1_', '')} cycle "
        f"{w['cycle']}, |I| = {np.abs(w['I']).mean():.1f} A", fontsize=9)

    axes[1].plot(w["t"], w["T"] - healthy["T"], color=C_TWIN, lw=1.2,
                 label="residual, healthy")
    axes[1].plot(w["t"], T_obs - healthy["T"], color="#d1660f", lw=1.4,
                 label="residual, with short")
    axes[1].axhline(q["residual"], color="k", ls="--", lw=1.1,
                    label=f"conformal threshold {q['residual']:.2f} K")
    axes[1].axvline(f.onset_s, color="k", lw=0.8, ls=":", label="short onset")
    axes[1].set_xlabel("time in window [s]"); axes[1].set_ylabel("residual [K]")
    axes[1].legend(fontsize=7); axes[1].set_title("what the sentinel scores", fontsize=9)
    _sim_badge(axes[0]); _sim_badge(axes[1])
    fig.tight_layout(); fig.savefig(out); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", default="assets")
    args = ap.parse_args()
    a = Path(args.assets); a.mkdir(exist_ok=True)

    keys = [k for k, v in DATASETS.items() if Path(v["fit"]).exists()]
    theta = {k: TwinParams(**json.loads(Path(DATASETS[k]["fit"]).read_text())["theta"])
             for k in keys}
    fig_twin_fit(a / "fig_twin_fit.png", theta, keys)
    have_far = [k for k in keys if Path(DATASETS[k]["far"]).exists()]
    fig_far(have_far, a / "fig_far.png")
    fig_residual(DATASETS["discharge"]["residuals"], DATASETS["discharge"]["data"],
                 DATASETS["discharge"]["far"], a / "fig_residual.png")
    have_lt = [k for k in keys if Path(DATASETS[k]["leadtime"]).exists()]
    fig_leadtime(have_lt, a / "fig_leadtime.png")
    trace_key = "nasa" if "nasa" in have_lt else have_lt[0]
    fig_trace(a / "fig_trace.png", theta[trace_key], trace_key)
    print("wrote", sorted(p.name for p in a.glob("*.png")))


if __name__ == "__main__":
    main()
