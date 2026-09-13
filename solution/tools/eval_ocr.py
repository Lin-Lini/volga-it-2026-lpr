#!/usr/bin/env python3
"""Exact-match / CER of a recogniser checkpoint on a crops folder (labels_*.csv)."""
import argparse, glob, os, sys, collections
import cv2
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from lpr.ocr import Recognizer, apply_unknown
from lpr.plates import fix_lookalikes

ap = argparse.ArgumentParser(); ap.add_argument("--weights", required=True); ap.add_argument("--crops", required=True)
ap.add_argument("--device", default="cpu"); ap.add_argument("--unk", type=float, default=0.3); a = ap.parse_args()
rec = Recognizer(a.weights, device=a.device)
rows = []
for lab in glob.glob(os.path.join(a.crops, "labels_*.csv")):
    rows += [l.strip().split(";") for l in open(lab, encoding="utf-8") if l.strip()]
ok = tot = 0; per = collections.defaultdict(lambda: [0, 0]); cer_n = cer_d = 0
def lev(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]
for fn, text, kind in rows:
    img = cv2.imread(os.path.join(a.crops, fn))
    if img is None: continue
    if kind == "type1a":
        h = img.shape[0]; parts = [img[int(h * 0.06):int(h * 0.54)], img[int(h * 0.48):int(h * 0.97)]]
    else:
        parts = [img]
    dec = rec.read_rows(parts)
    pred = "".join(d[0] for d in dec); confs = sum((d[1] for d in dec), [])
    pred, _ = apply_unknown(pred, confs, a.unk); pred = fix_lookalikes(pred, kind)
    tot += 1; per[kind][1] += 1; cer_n += lev(pred, text); cer_d += len(text)
    if pred == text: ok += 1; per[kind][0] += 1
print(f"{os.path.basename(a.weights)}: exact {ok/tot:.4f} ({ok}/{tot})  CER {cer_n/cer_d:.4f}  " + "  ".join(f"{k}: {v[0]/v[1]:.3f} (n={v[1]})" for k, v in sorted(per.items())))
