#!/bin/bash
# Build the held-out real test split from the dataset and evaluate the current weights on it.
#   bash solution/tools/eval_holdout.sh <dataset_dir> [device]
set -e
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
DS=${1:-$ROOT/dataset}; DEV=${2:-auto}
PY=${PYTHON:-$ROOT/.venv/bin/python}; [ -x "$PY" ] || PY=python
cd $ROOT
rm -rf work/test_set && $PY solution/tools/holdout_split.py --dataset "$DS" --frac 0.15 --out work/test_set --exclude-dir work/ds_v0/images/real
$PY solution/run.py --input work/test_set/images --output work/test_set/result.csv --device $DEV --debug-json work/test_set/debug.json
$PY solution/tools/evaluate.py --gt work/test_set/gt.csv --pred work/test_set/result.csv | tee work/test_set/metrics.txt
