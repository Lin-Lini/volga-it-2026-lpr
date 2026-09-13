#!/usr/bin/env python3
"""Transfer human decisions from an older candidate set to a new one (same raw images, re-run detector).

    python tools/transfer_decisions.py --old work/review/fareast_old --new work/review/fareast [--iou 0.4]

A decision is copied when a new candidate on the same image overlaps the old candidate (bbox IoU >= --iou).
Rejections ('keep=0') are transferred the same way; 'drop' verdicts are transferred to every candidate of the image.
"""
import argparse
import csv
import os
from collections import defaultdict


def load(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter=";"))


def iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    return inter / max(1e-6, aw * ah + bw * bh - inter)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", required=True)
    ap.add_argument("--new", required=True)
    ap.add_argument("--iou", type=float, default=0.4)
    a = ap.parse_args()
    old_c = {r["id"]: r for r in load(os.path.join(a.old, "candidates.csv"))}
    old_d = load(os.path.join(a.old, "decisions.csv"))
    new_c = load(os.path.join(a.new, "candidates.csv"))
    by_img = defaultdict(list)
    for r in new_c:
        by_img[r["image"]].append(r)
    dec_path = os.path.join(a.new, "decisions.csv")
    new_d = {r["id"]: r for r in load(dec_path)}
    moved = 0
    for d in old_d:
        c = old_c.get(d["id"])
        if not c:
            continue
        ob = [float(v) for v in c["bbox"].split(",")]
        if d["keep"] == "drop":
            for r in by_img.get(c["image"], []):
                new_d[r["id"]] = dict(d, id=r["id"])
                moved += 1
            continue
        best, bi = 0.0, None
        for r in by_img.get(c["image"], []):
            v = iou(ob, [float(x) for x in r["bbox"].split(",")])
            if v > best:
                best, bi = v, r
        if bi is not None and best >= a.iou:
            new_d[bi["id"]] = dict(d, id=bi["id"])
            moved += 1
    # old accepted candidates that have no counterpart in the new set are carried over verbatim
    new_ids = {r["id"] for r in new_c}
    carried = []
    for d in old_d:
        c = old_c.get(d["id"])
        if not c or d["keep"] != "1" or d["id"] in new_d or d["id"] in new_ids:
            continue
        ob = [float(v) for v in c["bbox"].split(",")]
        if any(iou(ob, [float(x) for x in r["bbox"].split(",")]) >= a.iou for r in by_img.get(c["image"], [])):
            continue
        carried.append(c)
        new_d[d["id"]] = dict(d)
    if carried:
        cand_path = os.path.join(a.new, "candidates.csv")
        with open(cand_path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(new_c[0].keys()), delimiter=";", extrasaction="ignore")
            for c in carried:
                w.writerow(c)
        for src in ("crops", "ctx"):
            for c in carried:
                sp = os.path.join(a.old, src, c["id"] + ".jpg")
                if os.path.exists(sp):
                    os.makedirs(os.path.join(a.new, src), exist_ok=True)
                    dst = os.path.join(a.new, src, c["id"] + ".jpg")
                    if not os.path.exists(dst):
                        import shutil
                        shutil.copy(sp, dst)
        print(f"carried over {len(carried)} accepted candidates without a new counterpart")
    with open(dec_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "plate_num", "plate_type", "is_vehicle", "keep", "conditions"], delimiter=";")
        w.writeheader()
        for r in new_d.values():
            w.writerow({k: r.get(k, "") for k in w.fieldnames})
    print(f"transferred {moved} decisions -> {dec_path} ({len(new_d)} total)")


if __name__ == "__main__":
    main()
