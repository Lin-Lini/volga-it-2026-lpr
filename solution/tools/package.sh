#!/bin/bash
# Package the deliverables: release/solution.zip (code + weights + docs) and release/dataset.zip.
#   bash solution/tools/package.sh
set -e
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
PY=${PYTHON:-$ROOT/.venv/bin/python}; [ -x "$PY" ] || PY=python
cd "$ROOT"

# explanatory note -> PDF (jury reads a document, not markdown)
$PY solution/tools/md2pdf.py solution/docs/note.md solution/docs/note.pdf

rm -rf release && mkdir -p release
rsync -a --exclude '__pycache__' --exclude '*.pyc' --exclude 'runs' --exclude '.venv' --exclude '.DS_Store' \
      solution/ release/solution/
cp dataset/validation_report.txt release/solution/docs/dataset_validation_report.txt
(cd release && zip -qr solution.zip solution)
zip -qr release/dataset.zip dataset -x 'dataset/**/__pycache__/*' '*.DS_Store'
$PY - <<'PYEOF'
import hashlib, os
for f in ("release/solution.zip", "release/dataset.zip"):
    h = hashlib.sha256(open(f, "rb").read()).hexdigest()
    print(f"{os.path.getsize(f)/1e6:8.1f} MB  sha256:{h[:16]}…  {f}")
PYEOF
ls -la release
