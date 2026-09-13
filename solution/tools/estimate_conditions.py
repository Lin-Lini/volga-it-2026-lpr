#!/usr/bin/env python3
"""Estimate the `conditions` field of real dataset rows from the image itself.

    python tools/estimate_conditions.py --dataset ../dataset --meta meta_real.csv [--also meta.csv] [--dry-run]

Every label is a measurement of the image, not a human judgement (documented as such in the datasheet):
  night       — low overall brightness of the scene (median V of the frame) with few bright pixels
  glare       — a saturated highlight covers a noticeable part of the rectified plate
  motion_blur — low gradient energy on the plate crop (blurred/defocused) with a directional asymmetry
  dirt        — strong dark low-frequency variation over the plate field (mud/snow patches)
  angle       — the plate quad deviates from a rectangle (perspective) by more than 8° (top ~15% of the set)
  rain / snow — not inferred automatically (needs the scene, not the plate); kept if already present
  day         — fallback when the frame is not dark
The previous value is preserved for any label the estimator does not produce (e.g. hand-set rain/snow).
"""
from __future__ import annotations

import argparse
import csv
import math
import os

import cv2
import numpy as np

AUTO = {"day", "night", "glare", "motion_blur", "dirt", "angle"}


def rectify(img, quad, w=208, h=48):
    dst = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float32)
    M = cv2.getPerspectiveTransform(quad.astype(np.float32), dst)
    return cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def quad_skew_deg(q: np.ndarray) -> float:
    """Max deviation of the quad's corner angles from 90° (perspective/rotation measure)."""
    worst = 0.0
    for i in range(4):
        a, b, c = q[(i - 1) % 4], q[i], q[(i + 1) % 4]
        v1, v2 = a - b, c - b
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 < 1e-6 or n2 < 1e-6:
            continue
        ang = math.degrees(math.acos(float(np.clip(np.dot(v1, v2) / (n1 * n2), -1, 1))))
        worst = max(worst, abs(ang - 90.0))
    return worst


def estimate(img, quad) -> set:
    out = set()
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    v = hsv[..., 2]
    if float(np.median(v)) < 70 and float((v > 200).mean()) < 0.08:
        out.add("night")
    else:
        out.add("day")
    if quad_skew_deg(quad) > 8.0:
        out.add("angle")
    crop = rectify(img, quad)
    if crop.size:
        g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        hv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        if float(((hv[..., 2] > 245) & (hv[..., 1] < 60)).mean()) > 0.06:
            out.add("glare")
        gx = float(np.abs(cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)).mean())
        gy = float(np.abs(cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)).mean())
        if max(gx, gy) < 50.0 and max(gx, gy) / max(min(gx, gy), 1e-3) > 1.4:
            out.add("motion_blur")
        low = cv2.GaussianBlur(g.astype(np.float32), (0, 0), 6)
        if float(low.std()) > 30 and float(np.median(g)) < 150:
            out.add("dirt")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--meta", default="meta_real.csv")
    ap.add_argument("--also", action="append", default=[], help="other meta files to update for the same rows")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    path = os.path.join(a.dataset, a.meta)
    rows = list(csv.DictReader(open(path, newline="", encoding="utf-8"), delimiter=";"))
    hdr = list(rows[0].keys())
    cache = {}
    updated = {}
    from collections import Counter
    stat = Counter()
    for r in rows:
        if r.get("is_synthetic") == "1":
            continue
        img_path = os.path.join(a.dataset, r["image"])
        if img_path not in cache:
            cache.clear()
            cache[img_path] = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cache[img_path]
        if img is None:
            continue
        q = np.array([float(x) for x in r["quad"].split(",")], np.float32).reshape(4, 2)
        est = estimate(img, q)
        kept = {c for c in (r.get("conditions", "") or "").split(",") if c and c not in AUTO}
        new = ",".join(sorted(est | kept))
        updated[(r["image"], r["quad"])] = new
        r["conditions"] = new
        for c in est | kept:
            stat[c] += 1
    print("conditions:", ", ".join(f"{k}={v}" for k, v in stat.most_common()))
    if a.dry_run:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=hdr, delimiter=";")
        w.writeheader(); w.writerows(rows)
    for other in a.also:
        op = os.path.join(a.dataset, other)
        if not os.path.exists(op):
            continue
        orows = list(csv.DictReader(open(op, newline="", encoding="utf-8"), delimiter=";"))
        ohdr = list(orows[0].keys())
        n = 0
        for r in orows:
            k = (r["image"], r["quad"])
            if k in updated:
                r["conditions"] = updated[k]; n += 1
        with open(op, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=ohdr, delimiter=";")
            w.writeheader(); w.writerows(orows)
        print(f"{other}: {n} rows updated")


if __name__ == "__main__":
    main()
