#!/usr/bin/env python3
"""Convert a dataset meta.csv into the ground-truth CSV of the task format (image;plate_num;plate_type;is_vehicle).

    python tools/meta_to_gt.py --meta work/syn_test/meta_synthetic.csv --out work/syn_test/gt.csv
"""
import argparse
import csv
import os

ap = argparse.ArgumentParser()
ap.add_argument("--meta", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--synthetic-only", action="store_true")
a = ap.parse_args()

rows = list(csv.DictReader(open(a.meta, newline="", encoding="utf-8"), delimiter=";"))
os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
n = 0
with open(a.out, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f, delimiter=";")
    w.writerow(["image", "plate_num", "plate_type", "is_vehicle"])
    for r in rows:
        if a.synthetic_only and r.get("is_synthetic") != "1":
            continue
        w.writerow([os.path.basename(r["image"]), r["plate_num"], r["plate_type"], r.get("is_vehicle", "1")])
        n += 1
print(f"{n} plates -> {a.out}")
