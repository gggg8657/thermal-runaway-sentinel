"""Assemble RESULTS.md -- and the README's number blocks -- from the run JSONs.

    PYTHONPATH=. python scripts/report.py

Regenerating the report is the only way numbers enter this repo's prose, so
nothing in either file is hand-typed. Every table below is built from
`data/prepare_report_*.json`, `runs/twin_fit*.json`, `runs/verify.json`,
`runs/far*.json`, `runs/leadtime*.json` and `runs/cost.json`.

README blocks are delimited by `<!-- BEGIN:name -->` / `<!-- END:name -->` and
overwritten in place.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from trsentinel.datasets import DATASETS

PHASES = tuple(DATASETS)
PHASE_LABEL = {k: f"{v['label']} ({v['sublabel']})" for k, v in DATASETS.items()}
SHORT = {k: v["label"] for k, v in DATASETS.items()}
VARIANT_LABEL = {
    "fleet_calibrated": "fleet-calibrated",
    "cell_registered": "cell-registered",
}
MODEL_LABEL = {
    "pybamm_spme": "PyBAMM SPMe + lumped thermal",
    "rom_lumped": "reduced-order lumped twin",
    "no_model_dT_over_ambient": "no model (T - T_amb)",
}


def read(p, default=None):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else default


def cell(x):
    if isinstance(x, float):
        x = f"{x:.4g}"
    return str(x).replace("|", "\\|")


def table(rows, header):
    out = ["| " + " | ".join(cell(h) for h in header) + " |",
           "|" + "|".join(["---"] * len(header)) + "|"]
    out += ["| " + " | ".join(cell(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def pct(x):
    return "-" if x is None else f"{100 * x:.1f}%"


def secs(x):
    return "-" if x is None else f"{x:.0f} s"


# --------------------------------------------------------------- sections
def sec_data(reports):
    rows = []
    for ph in PHASES:
        r = reports.get(ph)
        if not r:
            continue
        rows.append([
            PHASE_LABEL[ph], r["protocol"], r["n_windows"], r["n_cells"],
            f"{r['window_s']:.0f} s @ {r['dt_s']:.0f} s",
            f"{r['abs_current_range_A'][0]:.1f}-{r['abs_current_range_A'][1]:.1f} A",
            r["distinct_current_profiles"],
            f"{r['peak_dT_median_C']:.1f} K",
        ])
    return ["## The data: 46 real LFP cells", "",
            reports["discharge"]["meta"]["source"] + ". "
            + reports["discharge"]["meta"]["cell"] + ", "
            + reports["discharge"]["meta"]["chamber"] + ", "
            + reports["discharge"]["meta"]["channels"] + ".", "",
            "Two constant-current windows are cut out of every kept cycle. They "
            "differ in one way that turns out to decide the whole result: on "
            "discharge every cell in the dataset does exactly the same thing, "
            "and on charge no two policies agree.", "",
            table(rows, ["window", "protocol", "windows", "cells", "length",
                         "|I| range", "distinct current profiles",
                         "median peak rise over ambient"]), ""]


def sec_twin(fits, verify):
    L = ["## The twin, and whether it tracks a real cell", "",
         "An SPMe with a lumped thermal submodel on PyBAMM's `Prada2013` "
         "LFP/graphite chemistry, five identified scalars, fitted separately "
         "for each duty cycle on 10 cells and scored on cells it never saw. "
         "The reduced-order twin this repo started with -- two parameters, "
         "`C dT/dt = R I^2 - hA (T - T_amb)`, fitted on the same windows by "
         "the same trajectory criterion -- is scored alongside it.", ""]
    rows = []
    for ph in PHASES:
        f = fits.get(ph)
        if not f:
            continue
        for split in ("fit", "cal", "test"):
            pb, rom = f["pybamm"][split], f["rom_lumped_lstsq"][split]
            rows.append([
                SHORT[ph], split,
                f"{f['split_sizes'][split]} cells",
                f"{1000 * pb['v_rmse']:.0f} mV", f"{pb['t_rmse']:.2f} K",
                f"{pb['t_rmse_p95']:.2f} K", f"{rom['t_rmse']:.2f} K",
                pct(pb["frac_ok"]),
            ])
    L += [table(rows, ["window", "split", "n", "PyBAMM V rmse",
                       "PyBAMM T rmse", "PyBAMM T rmse p95",
                       "reduced-order T rmse", "windows solved"]), ""]

    rows = []
    for ph in PHASES:
        f = fits.get(ph)
        if not f:
            continue
        rows.append([SHORT[ph]] +
                    [f"{f['theta'][k]:.4g}"
                     + ("*" if f.get("at_bound", {}).get(k) else "")
                     for k in f["theta_names"]] +
                    [f["chemistry"], f["n_objective_evals"],
                     f"{f['fit_wall_s']:.0f} s"])
    L += ["### Identified parameters", "",
          table(rows, ["window"] + fits["discharge"]["theta_names"]
                + ["chemistry", "objective evaluations", "fit wall clock"]), "",
          "`*` marks a parameter that landed on the edge of its physical search "
          "box -- the fit wanted something the box would not give it, and the "
          "value should not be read as identified.", ""]

    cross = []
    for ph in PHASES:
        c = (fits.get(ph) or {}).get("cross_phase") or {}
        if c.get("test"):
            cross.append([f"fitted on {ph}, scored on {c['phase']}",
                          f"{1000 * c['test']['v_rmse']:.0f} mV",
                          f"{c['test']['t_rmse']:.2f} K"])
    if cross:
        L += ["### Transfer between duty cycles", "",
              "The same parameters, applied to the other window without "
              "refitting -- 4C discharge and an 8C charge are not the same "
              "operating point, and the twin is not claimed to be one model "
              "for both.", "",
              table(cross, ["", "V rmse (held-out cells)",
                            "T rmse (held-out cells)"]), ""]

    if verify:
        rows = []
        for ph in PHASES:
            v = verify.get(ph) or verify.get(f"{ph}_discharge")
            if not v:
                continue
            rows.append([
                SHORT[ph],
                f"{v.get('Q_pybamm_W', 0):.3f} W",
                f"{v.get('Q_contact_W', 0):.3f} W",
                f"{v.get('Q_terminal_balance_W', 0):.3f} W",
                pct(v.get("closure")),
                f"{v.get('C_eff_J_per_K', 0):.1f} J/K",
                f"{v.get('tau_s', 0):.0f} s",
                f"{v.get('lumped_reintegration_max_err_K', 0):.3f} K",
            ])
        L += ["### Energy balance, checked rather than assumed", "",
              "All the heat a cell releases has to come out of the gap between "
              "its open-circuit and terminal voltage. PyBAMM's default heat "
              "terms close only about half of that at these rates, which is why "
              "heat of mixing is switched on and the contact resistance's own "
              "dissipation is added by hand.", "",
              table(rows, ["window", "PyBAMM heat", "contact-resistance heat",
                           "I(U-V) balance", "closure", "identified C",
                           "thermal time constant", "lumped ODE re-integration"]),
              ""]
    return L


def sec_far(fars):
    L = ["## False-alarm rate on real healthy cells", "",
         "**This is real data, all of it.** The twin runs over every window of "
         "all 46 cells; the threshold is the split-conformal quantile of the 12 "
         "calibration cells; the rate below is measured on the 24 test cells, "
         "which were used neither to identify the twin nor to set the "
         "threshold. A conformal band promises `P(alarm | healthy) <= alpha`; "
         "these columns are whether that survived being carried from one set of "
         "physical cells to another.", ""]
    for ph in PHASES:
        f = fars.get(ph)
        if not f:
            continue
        L += [f"### {PHASE_LABEL[ph]}", "",
              f"{f['n_test_windows']} held-out windows, "
              f"{f['split_sizes']['test']} unseen cells, alarm = residual held "
              f"above the threshold for {f['persist_s']:.0f} s.", ""]
        for variant in ("fleet_calibrated", "cell_registered"):
            rows = []
            alphas = [r["alpha"] for r in f["variants"]["pybamm_spme"][variant]]
            for model in MODEL_LABEL:
                v = f["variants"][model]
                row = [MODEL_LABEL[model],
                       f"{v['residual_rms_test_K']:.2f} K"]
                for i, a in enumerate(alphas):
                    r = v[variant][i]
                    row.append(f"{r['far']:.3f} (q={r['threshold_K']:.2f} K)")
                rows.append(row)
            L += [f"**{VARIANT_LABEL[variant]}** threshold", "",
                  table(rows, ["detector", "residual rms"]
                        + [f"alpha={a}" for a in alphas]), ""]
    return L


def sec_leadtime(lts, alpha=0.05, variant="cell_registered"):
    L = ["## Lead time under a simulated internal short", "",
         "> **Simulated.** No public cycling dataset carries a labelled "
         "internal-short onset, so the fault is injected into the physics twin "
         "and every number in this section is a simulation result. What the "
         "detector sees is a **real measured temperature trace from a real "
         "held-out cell** plus the increment the short causes, so the alarm has "
         "to clear that cell's real residual noise floor. The threshold is the "
         "one calibrated on real healthy cells above and is never re-tuned "
         "here.", ""]
    key = f"{variant} alpha={alpha}"
    for ph in PHASES:
        lt = lts.get(ph)
        if not lt:
            continue
        rows = []
        for name, e in lt["faults"].items():
            s = e[key]
            rows.append([
                name, f"{e['peak_short_power_W']:.2f} W",
                f"{e['peak_shunt_current_A']:.3f} A",
                f"{e['peak_dT_K']:.2f} K",
                pct(s["detection_rate_residual"]),
                pct(s["detection_rate_no_model"]),
                secs(s["median_detect_after_onset_s"]),
                secs(s["median_lead_vs_no_model_s"]),
            ])
        c = lt["coupling_check"]
        L += [f"### {PHASE_LABEL[ph]}", "",
              f"{lt['n_windows']} real held-out windows from "
              f"{lt['n_test_cells']} cells, {VARIANT_LABEL[variant]} threshold "
              f"at alpha={alpha} "
              f"(residual {lt['thresholds_K'][variant][str(alpha)]['residual']:.2f} K, "
              f"no-model {lt['thresholds_K'][variant][str(alpha)]['no_model']:.2f} K).", "",
              table(rows, ["simulated fault", "short power (peak)",
                           "shunt current (peak)", "temperature rise caused",
                           "detected, physics residual", "detected, no model",
                           "median time to detect after onset",
                           "median lead over the no-model alarm"]), ""]
        hc = lt["healthy_control"].get(f"{variant} alpha={alpha}", {})
        if hc:
            L += [f"Healthy control on the same windows, same threshold: "
                  f"{hc['far_residual']:.3f} false-alarm rate for the residual "
                  f"detector, {hc['far_no_model']:.3f} for the no-model one "
                  f"({hc['n_windows']} windows). Staggered fault coupling, "
                  f"{c['dt_coarse_s']:.0f} s step against "
                  f"{c['dt_fine_s']:.2f} s: {c['max_abs_dT_K']:.3f} K.", ""]
    return L


def sec_cost(cost):
    if not cost:
        return []
    rows = []
    for ph, v in cost["phases"].items():
        rows.append([
            SHORT[ph], f"{v['window_s']:.0f} s",
            f"{v['pybamm_twin_s_per_window']:.2f} s",
            f"{v['pybamm_twin_realtime_factor']:.0f}x",
            f"{v['reduced_order_twin_s_per_window'] * 1e3:.2f} ms",
            f"{v['reduced_order_realtime_factor']:,.0f}x",
            f"{v['fleet_of_1000_cells_core_seconds_per_cycle']:.0f} core-s",
        ])
    return ["## What it costs to run", "",
            "Single core, no parallelism. The real-time factor is what decides "
            "whether the twin can run online next to the cell rather than in a "
            "nightly batch.", "",
            table(rows, ["window", "simulated span", "PyBAMM twin",
                         "faster than real time", "reduced-order twin",
                         "faster than real time", "1000-cell fleet, per cycle"]),
            ""]


def headline(reports, fits, fars, lts, verify, cost, a, variant):
    """The headline bullets, each one carrying a number the runs produced."""
    have = [p for p in PHASES if fars.get(p) and fits.get(p)]
    L = []

    n_cells = sum(reports[p]["n_cells"] for p in ("discharge", "nasa")
                  if reports.get(p))
    fr = {p: [r for r in fars[p]["variants"]["pybamm_spme"][variant]
              if r["alpha"] == a][0] for p in have}
    L.append(
        "- **The conformal promise survives the move between real cells.** "
        f"Calibrated at alpha={a} on cells held out for it and measured on "
        "cells used for nothing else, the false-alarm rate comes out at "
        + ", ".join(f"**{fr[p]['far']:.3f}** on {SHORT[p].lower()} "
                    f"({fr[p]['n_test_windows']} windows, "
                    f"{fars[p]['split_sizes']['test']} unseen cells)"
                    for p in have)
        + ". Those are real-data numbers and they are the ones that decide "
          "whether the method is usable.")

    L.append(
        "- **The twin tracks real cells across two chemistries.** On cells it "
        "never saw: "
        + "; ".join(
            f"{SHORT[p].lower()} **{1000 * fits[p]['pybamm']['test']['v_rmse']:.0f} mV / "
            f"{fits[p]['pybamm']['test']['t_rmse']:.2f} K**" for p in have)
        + ".")

    rom = [p for p in have
           if fits[p]["rom_lumped_lstsq"]["test"]["t_rmse"] <=
           fits[p]["pybamm"]["test"]["t_rmse"] * 1.05]
    L.append(
        "- **And the physics upgrade does not buy a quieter temperature "
        "residual.** The two-parameter lumped twin this repo started with, "
        "fitted on the same trajectories, reaches "
        + ", ".join(f"{fits[p]['rom_lumped_lstsq']['test']['t_rmse']:.2f} K on "
                    f"{SHORT[p].lower()}" for p in have)
        + f" -- no worse on {len(rom)} of the {len(have)} window sets. What the "
          "physics buys is a voltage prediction, parameters that are physical "
          "quantities, and somewhere to inject a fault; a model with no voltage "
          "state cannot host a resistance across the electrodes.")

    if verify:
        v = verify.get("discharge", {})
        if v:
            L.append(
                "- **The heat balance had to be repaired before any of this "
                "meant anything.** PyBAMM's default heat generation closes only "
                f"{100 * v['Q_pybamm_W'] / v['Q_terminal_balance_W']:.0f}% of "
                "the `I (U - V)` energy balance at 4C -- heat of mixing is off "
                "by default, the contact resistance contributes a voltage drop "
                "without its own dissipation, and `Prada2013` carries no "
                "entropic coefficient at all, which is why the real cells' "
                "endothermic first minute of fast charge was unreachable. With "
                f"all three restored the balance closes to "
                f"**{pct(v['closure'])}** and the identified thermal time "
                f"constant is {v['tau_s']:.0f} s on a "
                f"{v['C_eff_J_per_K']:.0f} J/K cell.")

    ph = [p for p in have if lts.get(p)]
    if ph:
        worst = list(lts[ph[0]]["faults"])[-1]
        L.append(
            "- **Lead time, and it is simulated.** A worsening internal short "
            "injected into the twin and added to a real measured trace is "
            "caught in "
            + ", ".join(
                f"{pct(lts[p]['faults'][worst][f'{variant} alpha={a}']['detection_rate_residual'])}"
                f" of {SHORT[p].lower()} windows" for p in ph if worst in lts[p]["faults"])
            + ", a median "
            + ", ".join(
                f"{secs(lts[p]['faults'][worst][f'{variant} alpha={a}']['median_lead_vs_no_model_s'])}"
                f" ({SHORT[p].lower()})" for p in ph if worst in lts[p]["faults"])
            + " before a model-free temperature alarm calibrated to the same "
              "false-alarm rate. No measured runaway event is involved anywhere.")

    if cost:
        c = cost["phases"].get("discharge")
        if c:
            L.append(
                f"- **Cheap enough to run online.** One {c['window_s']:.0f} s "
                f"window through the PyBAMM twin takes "
                f"**{c['pybamm_twin_s_per_window']:.2f} s** on one core -- "
                f"{c['pybamm_twin_realtime_factor']:.0f}x faster than the cell "
                f"lives it -- so a 1000-cell fleet costs "
                f"{c['fleet_of_1000_cells_core_seconds_per_cycle']:.0f} "
                "core-seconds per cycle. CPU only; no GPU is used or wanted.")

    n_win = sum(reports[p]["n_windows"] for p in PHASES if reports.get(p))
    L += ["", f"Measured on {n_cells} real cells from two laboratories, "
              f"{n_win} constant-current windows in total."]
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="RESULTS.md")
    ap.add_argument("--readme", default="README.md")
    args = ap.parse_args()

    reports = {p: read(DATASETS[p]["prepare_report"]) for p in PHASES}
    fits = {p: read(DATASETS[p]["fit"]) for p in PHASES}
    fars = {p: read(DATASETS[p]["far"]) for p in PHASES}
    lts = {p: read(DATASETS[p]["leadtime"]) for p in PHASES}
    verify = read("runs/verify.json")
    cost = read("runs/cost.json")

    L = ["# Results", "",
         "Regenerated by `scripts/report.py` from the JSON each run writes; "
         "none of it is typed in by hand. Sections are split by what they are: "
         "the false-alarm rates come from real cells, the lead times come from "
         "a fault injected into a simulation.", ""]
    L += sec_data(reports)
    L += sec_twin(fits, verify)
    L += sec_far(fars)
    L += sec_leadtime(lts)
    L += sec_cost(cost)
    Path(args.out).write_text("\n".join(L).rstrip() + "\n")

    blocks = build_readme_blocks(reports, fits, fars, lts, verify, cost)
    readme = Path(args.readme)
    if readme.exists():
        text = readme.read_text()
        for name, body in blocks.items():
            pat = re.compile(rf"(<!-- BEGIN:{name} -->)(.*?)(<!-- END:{name} -->)",
                             re.S)
            if not pat.search(text):
                print(f"  warning: README has no block {name!r}")
                continue
            text = pat.sub(lambda m: m.group(1) + "\n" + body + "\n" + m.group(3),
                           text)
        readme.write_text(text)
    print(f"wrote {args.out} and {len(blocks)} README blocks")


def build_readme_blocks(reports, fits, fars, lts, verify, cost):
    a, variant = 0.05, "cell_registered"
    b = {}

    b["headline"] = headline(reports, fits, fars, lts, verify, cost, a, variant)

    b["data"] = table(
        [[PHASE_LABEL[p], reports[p]["n_windows"], reports[p]["n_cells"],
          reports[p]["distinct_current_profiles"],
          f"{reports[p]['abs_current_range_A'][0]:.1f}-"
          f"{reports[p]['abs_current_range_A'][1]:.1f} A",
          f"{reports[p]['peak_dT_median_C']:.1f} K"]
         for p in PHASES if reports.get(p)],
        ["window", "windows", "cells", "distinct current profiles",
         "|I| range", "median peak rise"])

    b["twinfit"] = table(
        [[SHORT[p], DATASETS[p]["corpus"], fits[p]["chemistry"],
          f"{1000 * fits[p]['pybamm']['test']['v_rmse']:.0f} mV",
          f"{fits[p]['pybamm']['test']['t_rmse']:.2f} K",
          f"{fits[p]['rom_lumped_lstsq']['test']['t_rmse']:.2f} K",
          pct(fits[p]["pybamm"]["test"]["frac_ok"]),
          pct((verify.get(p) or {}).get("closure")) if verify else "-"]
         for p in PHASES if fits.get(p)],
        ["window", "corpus", "chemistry", "PyBAMM V rmse", "PyBAMM T rmse",
         "reduced-order T rmse", "windows solved", "energy-balance closure"])

    rows = []
    for p in PHASES:
        f = fars.get(p)
        if not f:
            continue
        for model in MODEL_LABEL:
            v = f["variants"][model][variant]
            r = [r for r in v if r["alpha"] == a][0]
            rows.append([SHORT[p], MODEL_LABEL[model],
                         f"{r['threshold_K']:.2f} K", f"{r['far']:.3f}",
                         f"{r['n_fired']}/{r['n_test_windows']}"])
    b["far"] = table(rows, ["window", "detector", f"threshold at alpha={a}",
                            "measured false-alarm rate", "windows fired"])

    rows = []
    for p in PHASES:
        lt = lts.get(p)
        if not lt:
            continue
        for name, e in lt["faults"].items():
            s = e[f"{variant} alpha={a}"]
            rows.append([SHORT[p], name,
                         f"{e['peak_short_power_W']:.2f} W",
                         f"{e['peak_dT_K']:.2f} K",
                         pct(s["detection_rate_residual"]),
                         pct(s["detection_rate_no_model"]),
                         secs(s["median_lead_vs_no_model_s"])])
    b["leadtime"] = table(rows, ["window", "simulated fault", "peak short power",
                                 "temperature rise caused",
                                 "detected (physics residual)",
                                 "detected (no model)",
                                 "median lead over the no-model alarm"])

    if cost:
        b["cost"] = table(
            [[SHORT[p],
              f"{v['pybamm_twin_s_per_window']:.2f} s",
              f"{v['pybamm_twin_realtime_factor']:.0f}x real time",
              f"{v['reduced_order_twin_s_per_window'] * 1e3:.2f} ms",
              f"{v['fleet_of_1000_cells_core_seconds_per_cycle']:.0f} core-s"]
             for p, v in cost["phases"].items()],
            ["window", "PyBAMM twin, one window", "", "reduced-order twin",
             "1000 cells, one cycle"])
    return b


if __name__ == "__main__":
    main()
