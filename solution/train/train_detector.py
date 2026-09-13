#!/usr/bin/env python3
"""Train the plate detector (YOLO11-pose: box + 4 corner keypoints + plate type class).

    python train/train_detector.py --data work/boot --data dataset --epochs 60 --out weights/det.pt

Every --data root must contain  images/{real,synthetic}/*.jpg  and  labels/*.txt  (YOLO-pose lines:
cls cx cy w h x1 y1 x2 y2 x3 y3 x4 y4, normalised). A train/val split is materialised with symlinks
under --work and the Ultralytics trainer is run on it.
"""
from __future__ import annotations

import argparse
import glob
import os
import random
import shutil
import sys

CLASSES = ["type1", "type1a", "type1b", "other"]


def collect(root):
    imgs = []
    for sub in ("real", "synthetic"):
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            imgs += glob.glob(os.path.join(root, "images", sub, ext))
    pairs = []
    for im in sorted(imgs):
        lab = os.path.join(root, "labels", os.path.splitext(os.path.basename(im))[0] + ".txt")
        if os.path.exists(lab):
            pairs.append((im, lab, "real" in im.replace("\\", "/").split("/")))
    return pairs


def link(src, dst):
    if os.path.lexists(dst):
        os.remove(dst)
    try:
        os.symlink(os.path.abspath(src), dst)
    except OSError:
        shutil.copy(src, dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", action="append", required=True)
    ap.add_argument("--work", default="work/det_data")
    ap.add_argument("--out", default="weights/det.pt")
    ap.add_argument("--model", default="yolo11n-pose.pt", help="pretrained checkpoint or .yaml to start from")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default=None)
    ap.add_argument("--val-frac", type=float, default=0.08)
    ap.add_argument("--real-repeat", type=int, default=3, help="oversample real images N times")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--name", default="det")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--holdout-frac", type=float, default=0.0, help="exclude this fraction of REAL images (deterministic by name hash) for a final test set")
    a = ap.parse_args()

    rng = random.Random(a.seed)
    pairs = []
    for d in a.data:
        p = collect(d)
        print(f"{d}: {len(p)} labelled images ({sum(1 for x in p if x[2])} real)")
        pairs += p
    if a.holdout_frac > 0:
        import hashlib
        def held(im):
            stem = os.path.splitext(os.path.basename(im))[0]
            return int(hashlib.md5(stem.encode()).hexdigest(), 16) % 1000 < a.holdout_frac * 1000
        before = len(pairs)
        pairs = [p for p in pairs if not (p[2] and held(p[0]))]
        print(f"holdout: excluded {before - len(pairs)} real images from training")
    rng.shuffle(pairs)
    n_val = int(len(pairs) * a.val_frac)
    val, train = pairs[:n_val], pairs[n_val:]
    for split in ("train", "val"):
        for sub in ("images", "labels"):
            d = os.path.join(a.work, split, sub)
            shutil.rmtree(d, ignore_errors=True)
            os.makedirs(d, exist_ok=True)
    for split, items in (("train", train), ("val", val)):
        for im, lab, is_real in items:
            base = os.path.basename(im)
            reps = a.real_repeat if (is_real and split == "train") else 1
            for r in range(reps):
                stem, ext = os.path.splitext(base)
                nm = f"{stem}_r{r}{ext}" if r else base
                link(im, os.path.join(a.work, split, "images", nm))
                link(lab, os.path.join(a.work, split, "labels", os.path.splitext(nm)[0] + ".txt"))
    yaml_path = os.path.join(a.work, "data.yaml")
    with open(yaml_path, "w") as f:
        f.write(f"path: {os.path.abspath(a.work)}\ntrain: train/images\nval: val/images\n")
        f.write("kpt_shape: [4, 2]\nflip_idx: [1, 0, 3, 2]\n")
        f.write("names:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(CLASSES)))
    print("train", len(train), "val", len(val), "->", yaml_path)

    from ultralytics import YOLO
    model = YOLO(a.resume or a.model)
    res = model.train(data=yaml_path, epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, device=a.device, workers=a.workers,
                      project=os.path.join(a.work, "runs"), name=a.name, exist_ok=True, seed=a.seed,
                      fliplr=0.0, mosaic=1.0, close_mosaic=10, degrees=5.0, scale=0.5, translate=0.1, perspective=0.0005,
                      hsv_h=0.02, hsv_s=0.6, hsv_v=0.5, mixup=0.05, patience=30, cos_lr=True, resume=bool(a.resume),
                      pose=12.0, kobj=1.0, box=7.5, cls=1.0)
    save_dir = str(getattr(res, "save_dir", "") or getattr(model.trainer, "save_dir", os.path.join(a.work, "runs", a.name)))
    best = os.path.join(save_dir, "weights", "best.pt")
    if not os.path.exists(best):
        best = os.path.join(save_dir, "weights", "last.pt")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    shutil.copy(best, a.out)
    print("saved", a.out)


if __name__ == "__main__":
    main()
