#!/bin/bash
# Assemble the deliverable dataset from reviewed candidates, generate synthetic scenes and validate.
#   bash solution/tools/build_all.sh <dataset_dir> <n_synthetic> <seed>
set -e
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
DS=${1:-$ROOT/dataset}
NSYN=${2:-5200}
SEED=${3:-42}
PY=${PYTHON:-$ROOT/.venv/bin/python}; [ -x "$PY" ] || PY=python
cd $ROOT

rm -rf "$DS/images" "$DS/labels" "$DS/meta.csv" "$DS/meta_real.csv" "$DS/meta_synthetic.csv" "$DS/attribution.csv"
mkdir -p "$DS/images/synthetic"
REVIEWS=""
for r in work/review/*/; do
  set_name=$(basename "$r")
  if [ -f "work/review2/$set_name/decisions.csv" ]; then continue; fi   # superseded by round-2 review
  [ -f "$r/decisions.csv" ] && REVIEWS="$REVIEWS --review $r"
done
for r in work/review2/*/; do
  [ -f "$r/decisions.csv" ] && REVIEWS="$REVIEWS --review $r"
done
$PY solution/tools/build_dataset.py $REVIEWS --out "$DS"

# generator (copy of solution/generator without caches)
rm -rf "$DS/generator"; mkdir -p "$DS/generator"
cp solution/generator/*.py solution/generator/requirements.txt solution/generator/README.md "$DS/generator/"
cp -r solution/generator/fonts "$DS/generator/fonts"

# shooting conditions of the real rows are measured from the images themselves
$PY solution/tools/estimate_conditions.py --dataset "$DS" --meta meta_real.csv

# synthetic scenes on real backgrounds (real plates replaced in place)
(cd "$DS/generator" && $PY generate.py --out "$DS" --n $NSYN --seed $SEED --backgrounds "$DS/images/real" --meta "$DS/meta_real.csv" --procedural-frac 0.25 --min-size 640 --max-size 1280)

# merge meta
$PY - "$DS" <<'EOF'
import csv, sys, os
ds = sys.argv[1]
rows = []
for name in ("meta_real.csv", "meta_synthetic.csv"):
    p = os.path.join(ds, name)
    if os.path.exists(p):
        rows += list(csv.DictReader(open(p, newline="", encoding="utf-8"), delimiter=";"))
fields = ["image", "plate_num", "plate_type", "bbox", "quad", "is_vehicle", "is_synthetic", "source", "license", "conditions"]
with open(os.path.join(ds, "meta.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fields, delimiter=";", extrasaction="ignore")
    w.writeheader(); w.writerows(rows)
print("meta.csv rows:", len(rows))
EOF

$PY solution/tools/validate_dataset.py --dataset "$DS" --report "$DS/validation_report.txt" || true
echo BUILD_ALL_DONE
