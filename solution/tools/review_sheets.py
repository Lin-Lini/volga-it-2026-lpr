#!/usr/bin/env python3
"""Build contact sheets of candidate plates for manual verification.

    python tools/review_sheets.py --review work/review/buses --per-sheet 24 [--type type1b] [--only-unreviewed]

Each tile shows the rectified crop (top) and the context crop (bottom) with the candidate index and the
model's guess. The reviewer records decisions in <review>/decisions.csv:
    id;plate_num;plate_type;is_vehicle;keep
(keep=0 rejects the candidate; plate_num uses Latin capitals and '#' for unreadable positions).
"""
from __future__ import annotations

import argparse
import csv
import os

import cv2
import numpy as np


def load_decisions(path):
    d = {}
    if os.path.exists(path):
        with open(path, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f, delimiter=";"):
                d[r["id"]] = r
    return d


def tile(review, row, tw=360, th=300):
    crop = cv2.imread(os.path.join(review, "crops", row["id"] + ".jpg"))
    ctx = cv2.imread(os.path.join(review, "ctx", row["id"] + ".jpg"))
    canvas = np.full((th, tw, 3), 30, np.uint8)
    if crop is not None:
        h, w = crop.shape[:2]
        s = min((tw - 8) / w, 120 / h)
        c = cv2.resize(crop, (int(w * s), int(h * s)))
        canvas[4:4 + c.shape[0], 4:4 + c.shape[1]] = c
    if ctx is not None:
        h, w = ctx.shape[:2]
        s = min((tw - 8) / w, (th - 150) / h)
        c = cv2.resize(ctx, (max(1, int(w * s)), max(1, int(h * s))))
        canvas[130:130 + c.shape[0], 4:4 + c.shape[1]] = c
    label = f"{row['_n']}: {row['det_type']} {row['ocr_text']} d{float(row['det_score']):.2f} v{row['is_vehicle']}"
    cv2.putText(canvas, label, (4, th - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--review", required=True)
    ap.add_argument("--per-sheet", type=int, default=24)
    ap.add_argument("--cols", type=int, default=6)
    ap.add_argument("--type", default=None)
    ap.add_argument("--min-score", type=float, default=0.0)
    ap.add_argument("--only-unreviewed", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--sort", default="score", choices=["score", "file"])
    ap.add_argument("--vehicle-only", action="store_true", help="skip candidates not on a vehicle unless det_score >= 0.7")
    ap.add_argument("--images-with-keeps", action="store_true", help="only candidates on images that already have an accepted plate")
    a = ap.parse_args()
    out = a.out or os.path.join(a.review, "sheets")
    os.makedirs(out, exist_ok=True)
    dec = load_decisions(os.path.join(a.review, "decisions.csv"))
    rows = []
    keep_imgs = set()
    if a.images_with_keeps:
        with open(os.path.join(a.review, "candidates.csv"), newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f, delimiter=";"):
                if r["id"] in dec and dec[r["id"]].get("keep") == "1":
                    keep_imgs.add(r["image"])
    with open(os.path.join(a.review, "candidates.csv"), newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter=";"):
            if a.images_with_keeps and r["image"] not in keep_imgs:
                continue
            if a.type and r["det_type"] != a.type:
                continue
            if float(r["det_score"]) < a.min_score:
                continue
            if a.only_unreviewed and r["id"] in dec:
                continue
            if a.vehicle_only and r["is_vehicle"] == "0" and float(r["det_score"]) < 0.7:
                continue
            rows.append(r)
    if a.sort == "score":
        rows.sort(key=lambda r: -float(r["det_score"]))
    for i, r in enumerate(rows):
        r["_n"] = i
    with open(os.path.join(out, "index.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["n", "id", "image", "det_type", "ocr_text"])
        for r in rows:
            w.writerow([r["_n"], r["id"], r["image"], r["det_type"], r["ocr_text"]])
    for s in range(0, len(rows), a.per_sheet):
        chunk = rows[s: s + a.per_sheet]
        tiles = [tile(a.review, r) for r in chunk]
        while len(tiles) % a.cols:
            tiles.append(np.full_like(tiles[0], 30))
        grid = np.vstack([np.hstack(tiles[i:i + a.cols]) for i in range(0, len(tiles), a.cols)])
        cv2.imwrite(os.path.join(out, f"sheet_{s // a.per_sheet:03d}.jpg"), grid, [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(f"{len(rows)} candidates -> {max(0, (len(rows) + a.per_sheet - 1) // a.per_sheet)} sheets in {out}")


if __name__ == "__main__":
    main()
