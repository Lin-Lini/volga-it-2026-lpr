#!/usr/bin/env python3
"""Assemble the real part of the dataset from reviewed candidates.

    python tools/build_dataset.py --review work/review/buses --review work/review/taxi ... \
        --raw-manifests raw/commons/buses/manifest.csv ... --out ../dataset

Inputs per review folder: candidates.csv (model proposals) + decisions.csv (human verdicts:
id;plate_num;plate_type;is_vehicle;keep[;quad]). Only candidates with a decision and keep=1 are used; an
image is included only if ALL its candidates were reviewed (so no unlabeled plates remain) and it has no
'drop' verdict (e.g. person is the main subject). Faces are blurred on copy. Writes images/real/*.jpg,
labels/*.txt (YOLO-pose), meta_real.csv, and attribution.csv (source page, author, license per image).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import os
import re
import shutil
import sys
from collections import defaultdict

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from blur_faces import FaceBlurrer  # noqa: E402

CLASSES = ["type1", "type1a", "type1b", "other"]
CYR2LAT = str.maketrans("АВЕКМНОРСТУХ", "ABEKMHOPCTYX")


def load_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter=";"))


def order_quad(q):
    c = q.mean(axis=0)
    ang = np.arctan2(q[:, 1] - c[1], q[:, 0] - c[0])
    q = q[np.argsort(ang)]
    return np.roll(q, -int(np.argmin(q.sum(axis=1))), axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--review", action="append", required=True, help="review folder (candidates.csv + decisions.csv); raw images are read from the folder recorded in its rawdir.txt or --raw")
    ap.add_argument("--raw", action="append", default=[], help="raw image folder for the corresponding --review (same order)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-side", type=int, default=1600)
    a = ap.parse_args()
    img_out = os.path.join(a.out, "images", "real")
    lab_out = os.path.join(a.out, "labels")
    os.makedirs(img_out, exist_ok=True)
    os.makedirs(lab_out, exist_ok=True)
    fb = FaceBlurrer()
    meta_rows, attr_rows = [], []
    n_img = 0
    counts = defaultdict(int)
    seen_sha = set()   # the same Commons file may be harvested into several raw folders
    for k, rv in enumerate(a.review):
        raw_dir = a.raw[k] if k < len(a.raw) else open(os.path.join(rv, "rawdir.txt")).read().strip()
        manifest = {r["file"]: r for r in load_csv(os.path.join(raw_dir, "manifest.csv"))}
        cands = load_csv(os.path.join(rv, "candidates.csv"))
        decs = {r["id"]: r for r in load_csv(os.path.join(rv, "decisions.csv"))}
        by_img = defaultdict(list)
        for c in cands:
            by_img[c["image"]].append(c)
        drops = {r["id"] for r in decs.values() if r.get("keep") == "drop"}
        for image, cs in sorted(by_img.items()):
            ids = [c["id"] for c in cs]
            if any(i not in decs for i in ids):
                continue                       # not fully reviewed -> unlabeled plates possible
            if any(i in drops for i in ids):
                continue                       # whole image rejected
            kept = [(c, decs[c["id"]]) for c in cs if decs[c["id"]].get("keep", "1") == "1"]
            # de-duplicate overlapping accepted candidates (keep the highest detector score)
            kept.sort(key=lambda cd: -float(cd[0]["det_score"]))
            uniq = []
            for c, d in kept:
                bx = [float(v) for v in c["bbox"].split(",")]
                dup = False
                for c2, _ in uniq:
                    b2 = [float(v) for v in c2["bbox"].split(",")]
                    ix = max(0, min(bx[0] + bx[2], b2[0] + b2[2]) - max(bx[0], b2[0]))
                    iy = max(0, min(bx[1] + bx[3], b2[1] + b2[3]) - max(bx[1], b2[1]))
                    if ix * iy > 0.5 * min(bx[2] * bx[3], b2[2] * b2[3]):
                        dup = True
                        break
                if not dup:
                    uniq.append((c, d))
            kept = uniq
            if not kept:
                continue
            m0 = manifest.get(image, {})
            key = m0.get("sha1") or m0.get("id") or image
            if key in seen_sha:
                continue
            seen_sha.add(key)
            src = os.path.join(raw_dir, image)
            img = cv2.imread(src, cv2.IMREAD_COLOR)
            if img is None:
                continue
            H0, W0 = img.shape[:2]
            cw, ch = int(cs[0]["img_w"]), int(cs[0]["img_h"])   # size at which candidates were produced
            sx, sy = W0 / cw, H0 / ch
            s = min(1.0, a.max_side / max(H0, W0))
            if s < 1.0:
                img = cv2.resize(img, (int(W0 * s), int(H0 * s)), interpolation=cv2.INTER_AREA)
            H, W = img.shape[:2]
            img, n_faces = fb.blur(img)
            m = manifest.get(image, {})
            stem = f"real_{hashlib.md5(image.encode()).hexdigest()[:10]}"
            name = stem + ".jpg"
            cv2.imwrite(os.path.join(img_out, name), img, [cv2.IMWRITE_JPEG_QUALITY, 93])
            lines = []
            for c, d in kept:
                q = np.array([float(v) for v in (d.get("quad") or c["quad"]).split(",")], np.float32).reshape(4, 2)
                q = order_quad(q * np.array([sx * s, sy * s], np.float32))
                q[:, 0] = np.clip(q[:, 0], 0, W - 1)
                q[:, 1] = np.clip(q[:, 1], 0, H - 1)
                x0, y0 = float(q[:, 0].min()), float(q[:, 1].min())
                bw, bh = float(q[:, 0].max() - x0), float(q[:, 1].max() - y0)
                ptype = d.get("plate_type") or c["det_type"]
                pn = (d.get("plate_num") if d.get("plate_num") is not None else c["ocr_text"]).upper().translate(CYR2LAT).replace(" ", "")
                isv = d.get("is_vehicle") or c["is_vehicle"]
                conds = d.get("conditions", "") or "day"
                lic = m.get("license", "") or d.get("license", "")
                if lic in ("by", "by-sa", "cc0", "pdm"):          # Openverse codes -> human-readable
                    lic = {"by": "CC BY", "by-sa": "CC BY-SA", "cc0": "CC0", "pdm": "Public domain"}[lic]
                    if m.get("license_version") and lic.startswith("CC BY"):
                        lic += " " + m["license_version"]
                source = m.get("page", "") or m.get("landing", "") or d.get("source", "own_photo")
                lines.append(f"{CLASSES.index(ptype)} {(x0 + bw / 2) / W:.6f} {(y0 + bh / 2) / H:.6f} {bw / W:.6f} {bh / H:.6f} "
                             + " ".join(f"{q[i, 0] / W:.6f} {q[i, 1] / H:.6f}" for i in range(4)))
                meta_rows.append(dict(image=f"images/real/{name}", plate_num=pn, plate_type=ptype,
                                      bbox=f"{x0:.1f},{y0:.1f},{bw:.1f},{bh:.1f}", quad=",".join(f"{v:.1f}" for v in q.reshape(-1)),
                                      is_vehicle=isv, is_synthetic=0, source=source, license=lic, conditions=conds))
                counts[ptype] += 1
            with open(os.path.join(lab_out, stem + ".txt"), "w") as f:
                f.write("\n".join(lines) + "\n")
            attr_rows.append(dict(image=f"images/real/{name}", original=image, title=m.get("title", ""), page=source,
                                  author=m.get("artist", m.get("creator", "")), license=lic, license_url=m.get("license_url", ""),
                                  faces_blurred=n_faces))
            n_img += 1
    with open(os.path.join(a.out, "meta_real.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["image", "plate_num", "plate_type", "bbox", "quad", "is_vehicle", "is_synthetic", "source", "license", "conditions"], delimiter=";")
        w.writeheader()
        w.writerows(meta_rows)
    with open(os.path.join(a.out, "attribution.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["image", "original", "title", "page", "author", "license", "license_url", "faces_blurred"], delimiter=";")
        w.writeheader()
        w.writerows(attr_rows)
    print(f"{n_img} real images, plates: {dict(counts)} -> {a.out}")


if __name__ == "__main__":
    main()
