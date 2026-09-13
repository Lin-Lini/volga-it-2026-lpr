#!/usr/bin/env python3
"""Export rectified plate crops (+ labels) from a dataset meta.csv for recogniser training / evaluation.

    python tools/export_ocr_crops.py --dataset work/ds_v0 --meta meta_real.csv --out work/ocr_real --prefix real

Writes <out>/<prefix>_XXXXX.jpg (192x48 single-row, 192x112 two-row) and <out>/labels_<prefix>.csv
(file;text;kind). Plates of type 'other' and fully unreadable plates are skipped. With --jitter N,
N extra copies with randomly perturbed corners are written (simulates imprecise detection).
"""
import argparse
import csv
import os
import random

import cv2
import numpy as np


def rectify(img, q, w, h, rng=None, jitter=0.0):
    q = q.astype(np.float32).copy()
    if rng is not None and jitter > 0:
        pw = np.linalg.norm(q[1] - q[0])
        ph = np.linalg.norm(q[3] - q[0])
        q += np.array([[rng.uniform(-jitter, jitter) * pw, rng.uniform(-jitter, jitter) * ph] for _ in range(4)], np.float32)
    dst = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float32)
    M = cv2.getPerspectiveTransform(q, dst)
    return cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--meta", default="meta.csv")
    ap.add_argument("--out", required=True)
    ap.add_argument("--prefix", default="real")
    ap.add_argument("--jitter", type=int, default=2, help="extra jittered copies per plate")
    ap.add_argument("--min-width", type=int, default=20)
    ap.add_argument("--only-real", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    os.makedirs(a.out, exist_ok=True)
    lab = open(os.path.join(a.out, f"labels_{a.prefix}.csv"), "w", encoding="utf-8")
    n = 0
    cache = {}
    with open(os.path.join(a.dataset, a.meta), newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter=";"):
            if r["plate_type"] == "other" or not r["plate_num"] or set(r["plate_num"]) == {"#"}:
                continue
            if a.only_real and r.get("is_synthetic") == "1":
                continue
            path = os.path.join(a.dataset, r["image"])
            if path not in cache:
                cache = {path: cv2.imread(path, cv2.IMREAD_COLOR)}
            img = cache[path]
            if img is None:
                continue
            q = np.array([float(v) for v in r["quad"].split(",")], np.float32).reshape(4, 2)
            if np.linalg.norm(q[1] - q[0]) < a.min_width:
                continue
            two = r["plate_type"] == "type1a"
            w, h = (192, 112) if two else (192, 48)
            for k in range(1 + a.jitter):
                crop = rectify(img, q, w, h, rng if k else None, 0.04 if k else 0.0)
                name = f"{a.prefix}_{n:06d}_{k}.jpg"
                cv2.imwrite(os.path.join(a.out, name), crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
                lab.write(f"{name};{r['plate_num']};{r['plate_type']}\n")
            n += 1
    lab.close()
    print(f"{n} plates -> {a.out}")


if __name__ == "__main__":
    main()
