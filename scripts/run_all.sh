#!/usr/bin/env bash
# Everything, in order, from the raw archives to the figures and the report.
#
#   bash scripts/run_all.sh [path/to/batch1.pkl]
#
# ~1 h on a many-core CPU box; no GPU is used or wanted. The NASA archive is
# downloaded if data/nasa is not already populated.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=.
PY=${PY:-python}
RAW=${1:-/home/dongjukim/Documents/workspace/repos/PyBAMM_Inverse/data/raw/batch1.pkl}
W=${WORKERS:-40}
NASA_URL="https://phm-datasets.s3.amazonaws.com/NASA/5.+Battery+Data+Set.zip"

# ---- data
$PY scripts/prepare_data.py --raw "$RAW" --phase discharge
$PY scripts/prepare_data.py --raw "$RAW" --phase charge
if [ ! -d data/nasa/ext ]; then
  mkdir -p data/nasa && curl -sSL -o data/nasa/nasa_battery.zip "$NASA_URL"
  $PY - <<'PYX'
import glob, os, zipfile
zipfile.ZipFile("data/nasa/nasa_battery.zip").extractall("data/nasa")
for f in glob.glob("data/nasa/**/*.zip", recursive=True):
    if f.endswith("nasa_battery.zip"):
        continue
    zipfile.ZipFile(f).extractall(os.path.join("data/nasa/ext",
                                               os.path.basename(f)[:-4]))
PYX
fi
$PY scripts/prepare_nasa.py

# ---- identify the twin, once per window set (these are the slow steps)
$PY scripts/fit_twin.py --data data/severson_discharge.npz \
    --cross-data data/severson_charge.npz --out runs/twin_fit.json --workers $W &
$PY scripts/fit_twin.py --data data/severson_charge.npz \
    --cross-data data/severson_discharge.npz --out runs/twin_fit_charge.json --workers $W &
$PY scripts/fit_twin.py --data data/nasa_discharge.npz --chemistry Ramadass2004 \
    --out runs/twin_fit_nasa.json --workers $W &
wait

$PY scripts/verify_twin.py

# ---- real-data false-alarm rate, then the simulated-fault lead time
for s in "severson_discharge twin_fit far residuals leadtime" \
         "severson_charge twin_fit_charge far_charge residuals_charge leadtime_charge" \
         "nasa_discharge twin_fit_nasa far_nasa residuals_nasa leadtime_nasa"; do
  set -- $s
  $PY scripts/eval_far.py --data "data/$1.npz" --fit "runs/$2.json" \
      --out "runs/$3.json" --residuals "runs/$4.npz" --workers $W
  $PY scripts/eval_leadtime.py --data "data/$1.npz" --fit "runs/$2.json" \
      --residuals "runs/$4.npz" --out "runs/$5.json" --workers $W
done

$PY scripts/bench_cost.py
$PY scripts/make_figures.py
$PY scripts/report.py

# ---- the offline demo and the tests
$PY demo_smoke.py
$PY tests/test_smoke.py
$PY tests/test_conformal.py
$PY tests/test_twin_pybamm.py
