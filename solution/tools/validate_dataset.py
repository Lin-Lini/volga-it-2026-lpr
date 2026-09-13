#!/usr/bin/env python3
"""Validate a plate dataset against the structure of the task statement (section 6).

    python tools/validate_dataset.py --dataset ../dataset [--report validation_report.txt]

Checks: folder layout, meta.csv columns / separators, plate_num masks, plate types, bbox/quad geometry
against image size and against each other, labels/*.txt presence and consistency with meta.csv,
is_vehicle / is_synthetic flags, licenses, conditions vocabulary, duplicates, and prints statistics
(images and unique plates per type, real vs synthetic, conditions, licenses).
Exit code 0 = no errors (warnings allowed), 1 = errors found.
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from collections import Counter, defaultdict

from PIL import Image

TYPES = {"type1", "type1a", "type1b", "other"}
COND = {"day", "night", "rain", "snow", "dirt", "glare", "motion_blur", "angle"}
RE_PLATE_STD = re.compile(r"^[ABEKMHOPCTYX#][0-9#]{3}[ABEKMHOPCTYX#]{2}([0-9#]{2}|[1-9#][0-9#]{2})$")
RE_PLATE_TAXI = re.compile(r"^[ABEKMHOPCTYX#]{2}[0-9#]{3}([0-9#]{2}|[1-9#][0-9#]{2})$")
REQUIRED = ["image", "plate_num", "plate_type", "bbox", "quad", "is_vehicle", "is_synthetic", "source", "license", "conditions"]
CLASSES = ["type1", "type1a", "type1b", "other"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--report", default=None)
    ap.add_argument("--check-images", action="store_true", help="open every image (slower)")
    a = ap.parse_args()
    root = a.dataset
    errors, warnings = [], []
    out_lines = []

    def log(s=""):
        out_lines.append(s)
        print(s)

    log(f"Dataset validation report: {os.path.abspath(root)}")
    for d in ["images", "images/real", "images/synthetic", "labels", "generator"]:
        if not os.path.isdir(os.path.join(root, d)):
            errors.append(f"missing folder {d}/")
    for f in ["meta.csv", "README.md", "LICENSE"]:
        if not os.path.isfile(os.path.join(root, f)):
            errors.append(f"missing file {f}")
    gen = os.path.join(root, "generator")
    if os.path.isdir(gen):
        if not any(f.endswith(".py") for f in os.listdir(gen)):
            errors.append("generator/ has no .py script")
        if not os.path.exists(os.path.join(gen, "requirements.txt")):
            warnings.append("generator/requirements.txt missing")
    meta_path = os.path.join(root, "meta.csv")
    if not os.path.isfile(meta_path):
        log("\n".join("ERROR: " + e for e in errors))
        sys.exit(1)

    with open(meta_path, newline="", encoding="utf-8") as f:
        head = f.readline()
        if ";" not in head:
            errors.append("meta.csv must use ';' as separator")
        f.seek(0)
        rows = list(csv.DictReader(f, delimiter=";"))
    cols = list(rows[0].keys()) if rows else []
    for c in REQUIRED:
        if c not in cols:
            errors.append(f"meta.csv: missing column {c}")
    if errors:
        log("\n".join("ERROR: " + e for e in errors))
        if a.report:
            open(a.report, "w").write("\n".join(out_lines))
        sys.exit(1)

    per_image = defaultdict(list)
    for r in rows:
        per_image[r["image"]].append(r)
    size_cache = {}
    stats = Counter()
    uniq = defaultdict(set)
    lic = Counter()
    src = Counter()
    conds = Counter()
    seen_rows = set()
    n_real_img = n_syn_img = 0
    for img_rel, rs in per_image.items():
        p = os.path.join(root, img_rel)
        if not os.path.isfile(p):
            errors.append(f"image not found: {img_rel}")
            continue
        if not img_rel.startswith("images/real/") and not img_rel.startswith("images/synthetic/"):
            errors.append(f"image outside images/real or images/synthetic: {img_rel}")
        try:
            with Image.open(p) as im:
                W, H = im.size
                if a.check_images:
                    im.verify()
        except Exception as e:
            errors.append(f"cannot open {img_rel}: {e}")
            continue
        size_cache[img_rel] = (W, H)
        syn_flags = {r["is_synthetic"] for r in rs}
        if len(syn_flags) > 1:
            errors.append(f"{img_rel}: mixed is_synthetic flags")
        is_syn = rs[0]["is_synthetic"] == "1"
        if is_syn != img_rel.startswith("images/synthetic/"):
            errors.append(f"{img_rel}: is_synthetic={rs[0]['is_synthetic']} but path says otherwise")
        if is_syn:
            n_syn_img += 1
        else:
            n_real_img += 1
        # labels file
        lab = os.path.join(root, "labels", os.path.splitext(os.path.basename(img_rel))[0] + ".txt")
        lab_lines = []
        if not os.path.isfile(lab):
            errors.append(f"{img_rel}: labels/{os.path.basename(lab)} missing")
        else:
            lab_lines = [ln.split() for ln in open(lab, encoding="utf-8") if ln.strip()]
            if len(lab_lines) != len(rs):
                errors.append(f"{img_rel}: {len(lab_lines)} label lines vs {len(rs)} meta rows")
        for r in rs:
            key = (r["image"], r["quad"])
            if key in seen_rows:
                errors.append(f"{img_rel}: duplicate row for quad {r['quad']}")
            seen_rows.add(key)
            t = r["plate_type"]
            if t not in TYPES:
                errors.append(f"{img_rel}: bad plate_type '{t}'")
                continue
            pn = r["plate_num"]
            if t in ("type1", "type1a"):
                if not RE_PLATE_STD.match(pn):
                    errors.append(f"{img_rel}: plate_num '{pn}' does not match mask for {t}")
            elif t == "type1b":
                if not (RE_PLATE_STD.match(pn) or RE_PLATE_TAXI.match(pn)):
                    errors.append(f"{img_rel}: plate_num '{pn}' does not match type1b masks")
            else:
                if pn and not re.match(r"^[A-Z0-9#]{1,12}$", pn):
                    warnings.append(f"{img_rel}: unusual plate_num '{pn}' for other")
            if pn and pn.count("#") == len(pn):
                warnings.append(f"{img_rel}: plate_num fully unreadable ({pn})")
            try:
                x, y, w, h = [float(v) for v in r["bbox"].split(",")]
                q = [float(v) for v in r["quad"].split(",")]
                assert len(q) == 8
            except Exception:
                errors.append(f"{img_rel}: bad bbox/quad format")
                continue
            if w <= 2 or h <= 2:
                errors.append(f"{img_rel}: degenerate bbox {r['bbox']}")
            if x < -1 or y < -1 or x + w > W + 1 or y + h > H + 1:
                errors.append(f"{img_rel}: bbox outside image ({W}x{H}): {r['bbox']}")
            qx, qy = q[0::2], q[1::2]
            if min(qx) < x - 2 or max(qx) > x + w + 2 or min(qy) < y - 2 or max(qy) > y + h + 2:
                errors.append(f"{img_rel}: quad not inside bbox")
            # clockwise from top-left: TL should have the smallest x+y and the polygon area positive in image coords
            area = 0.0
            for i in range(4):
                j = (i + 1) % 4
                area += qx[i] * qy[j] - qx[j] * qy[i]
            if area <= 0:   # image y axis points down => clockwise-on-screen polygons have positive shoelace area
                warnings.append(f"{img_rel}: quad is not clockwise")
            if r["is_vehicle"] not in ("0", "1"):
                errors.append(f"{img_rel}: is_vehicle must be 0/1")
            if r["is_synthetic"] not in ("0", "1"):
                errors.append(f"{img_rel}: is_synthetic must be 0/1")
            if not r["source"]:
                errors.append(f"{img_rel}: empty source")
            if not r["license"]:
                errors.append(f"{img_rel}: empty license")
            elif re.search(r"NC|ND|all rights|copyright", r["license"], re.I):
                errors.append(f"{img_rel}: incompatible license '{r['license']}'")
            for c in (r["conditions"].split(",") if r["conditions"] else []):
                if c not in COND:
                    warnings.append(f"{img_rel}: unknown condition '{c}'")
                conds[c] += 1
            stats[(t, "syn" if is_syn else "real")] += 1
            if not is_syn:
                uniq[t].add(pn)
            lic[r["license"]] += 1
            src[r["source"] if r["source"] in ("own_photo", "generator") else re.sub(r"[/:].*", "", r["source"])] += 1
            # cross-check with labels file
            if lab_lines:
                best = None
                for ln in lab_lines:
                    if len(ln) < 13:
                        errors.append(f"{img_rel}: label line with {len(ln)} fields (need 13)")
                        continue
                    cx, cy = float(ln[1]) * W, float(ln[2]) * H
                    if best is None or abs(cx - (x + w / 2)) + abs(cy - (y + h / 2)) < best[0]:
                        best = (abs(cx - (x + w / 2)) + abs(cy - (y + h / 2)), ln)
                if best and (best[0] > 3 or int(best[1][0]) != CLASSES.index(t)):
                    errors.append(f"{img_rel}: labels/*.txt disagree with meta row (class/box) for {pn}")

    log("")
    log("Images:   real=%d synthetic=%d total=%d" % (n_real_img, n_syn_img, n_real_img + n_syn_img))
    log("Plates:")
    log("  %-8s %8s %8s %12s" % ("type", "real", "synth", "uniq(real)"))
    for t in CLASSES:
        log("  %-8s %8d %8d %12d" % (t, stats[(t, "real")], stats[(t, "syn")], len(uniq[t])))
    req = {"type1a": (150, 50), "type1b": (300, 100), "other": (50, 0)}
    real_imgs = defaultdict(set)
    for r in rows:
        if r["is_synthetic"] == "0":
            real_imgs[r["plate_type"]].add(r["image"])
    for t, (ni, nu) in req.items():
        flag = "OK " if (len(real_imgs[t]) >= ni and len(uniq[t]) >= nu) else "BELOW"
        log(f"  recommended minimum {t}: {ni} images / {nu} unique -> have {len(real_imgs[t])} / {len(uniq[t])}  [{flag}]")
    log(f"  synthetic images: {n_syn_img} (recommended >= 5000) [{'OK' if n_syn_img >= 5000 else 'BELOW'}]")
    log("Conditions: " + ", ".join(f"{k}={v}" for k, v in conds.most_common()))
    log("Licenses:   " + ", ".join(f"{k}={v}" for k, v in lic.most_common()))
    log("Sources:    " + ", ".join(f"{k}={v}" for k, v in src.most_common(12)))
    log("")
    for wmsg in warnings[:50]:
        log("WARNING: " + wmsg)
    if len(warnings) > 50:
        log(f"... {len(warnings) - 50} more warnings")
    for e in errors[:100]:
        log("ERROR: " + e)
    if len(errors) > 100:
        log(f"... {len(errors) - 100} more errors")
    log("")
    log(f"RESULT: {'PASS' if not errors else 'FAIL'}  ({len(errors)} errors, {len(warnings)} warnings)")
    if a.report:
        with open(a.report, "w", encoding="utf-8") as f:
            f.write("\n".join(out_lines) + "\n")
    sys.exit(0 if not errors else 1)


if __name__ == "__main__":
    main()
