#!/usr/bin/env python3
"""Materialise the held-out real test split (same hash rule as train_detector.py --holdout-frac).

    python tools/holdout_split.py --dataset ../dataset --frac 0.15 --out work/test_set
Creates <out>/images/*.jpg (copies) and <out>/gt.csv in the task's result format (image;plate_num;plate_type;is_vehicle).
"""
import argparse, csv, hashlib, os, shutil

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", required=True)
ap.add_argument("--meta", default="meta.csv")
ap.add_argument("--frac", type=float, default=0.15)
ap.add_argument("--out", required=True)
ap.add_argument("--exclude-dir", default=None, help="skip images whose file name exists in this folder (e.g. images seen by an earlier training stage)")
a = ap.parse_args()
excluded = set(os.listdir(a.exclude_dir)) if a.exclude_dir and os.path.isdir(a.exclude_dir) else set()
os.makedirs(os.path.join(a.out, "images"), exist_ok=True)
rows = list(csv.DictReader(open(os.path.join(a.dataset, a.meta), newline="", encoding="utf-8"), delimiter=";"))
held = {}
for r in rows:
    if r.get("is_synthetic", "0") == "1":
        continue
    stem = os.path.splitext(os.path.basename(r["image"]))[0]
    if os.path.basename(r["image"]) in excluded:
        continue
    if int(hashlib.md5(stem.encode()).hexdigest(), 16) % 1000 < a.frac * 1000:
        held.setdefault(r["image"], []).append(r)
with open(os.path.join(a.out, "gt.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f, delimiter=";")
    w.writerow(["image", "plate_num", "plate_type", "is_vehicle"])
    for img, rs in sorted(held.items()):
        shutil.copy(os.path.join(a.dataset, img), os.path.join(a.out, "images", os.path.basename(img)))
        for r in rs:
            w.writerow([os.path.basename(img), r["plate_num"], r["plate_type"], r["is_vehicle"]])
print(f"{len(held)} images, {sum(len(v) for v in held.values())} plates -> {a.out}")
