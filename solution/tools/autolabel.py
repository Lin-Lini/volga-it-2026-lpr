#!/usr/bin/env python3
"""Pre-annotate raw photos with the current models and export crops for manual review.

    python tools/autolabel.py --input raw/commons/buses --out work/review/buses --det weights/det.pt --ocr weights/ocr.pt

Writes
    <out>/candidates.csv   one row per detected plate: id;image;quad;bbox;det_type;det_score;ocr_text;ocr_score;is_vehicle;persons
    <out>/crops/<id>.jpg   rectified plate crop (for reading)
    <out>/ctx/<id>.jpg     context crop around the plate (for judging vehicle / plate type)
Every candidate is then checked by a human (see review_sheets.py); nothing here is final annotation.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from lpr.detector import CLASSES  # noqa: E402
from lpr.pipeline import LPRPipeline, RECT_1ROW, RECT_2ROW, pick_device, rectify  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--det", default="weights/det.pt")
    ap.add_argument("--ocr", default="weights/ocr.pt")
    ap.add_argument("--veh", default="weights/yolo11n.pt")
    ap.add_argument("--conf", type=float, default=0.2)
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--min-width", type=int, default=24, help="skip plates narrower than this (px)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-per-image", type=int, default=8)
    a = ap.parse_args()

    device = pick_device(a.device)
    pipe = LPRPipeline(a.det, a.ocr, a.veh, device=device, det_imgsz=a.imgsz, det_conf=a.conf, min_conf=0.0, emit_other=True)
    from ultralytics import YOLO
    person_det = YOLO(a.veh)
    os.makedirs(os.path.join(a.out, "crops"), exist_ok=True)
    os.makedirs(os.path.join(a.out, "ctx"), exist_ok=True)
    files = sorted(f for f in os.listdir(a.input) if f.lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff")))
    if a.limit:
        files = files[: a.limit]
    done = set()
    cand_path = os.path.join(a.out, "candidates.csv")
    if os.path.exists(cand_path):
        with open(cand_path, newline="", encoding="utf-8") as f:
            done = {r["image"] for r in csv.DictReader(f, delimiter=";")}
    new = not os.path.exists(cand_path)
    with open(cand_path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter=";")
        if new:
            w.writerow(["id", "image", "quad", "bbox", "det_type", "det_score", "ocr_text", "ocr_score", "is_vehicle", "persons", "img_w", "img_h"])
        n_img = n_pl = 0
        for k, fn in enumerate(files):
            if fn in done:
                continue
            img = cv2.imread(os.path.join(a.input, fn), cv2.IMREAD_COLOR)
            if img is None:
                continue
            H, W = img.shape[:2]
            if max(H, W) > 2000:
                s = 2000 / max(H, W)
                img = cv2.resize(img, (int(W * s), int(H * s)), interpolation=cv2.INTER_AREA)
                H, W = img.shape[:2]
            res = pipe.process(img)
            if not res:
                continue
            pr = person_det.predict(img, imgsz=640, conf=0.3, classes=[0], device=device, verbose=False)[0]
            persons = 0.0
            if pr.boxes is not None and len(pr.boxes):
                for b in pr.boxes.xyxy.cpu().numpy():
                    persons = max(persons, float((b[2] - b[0]) * (b[3] - b[1]) / (W * H)))
            # plausibility filters: aspect ratio per type, size relative to the image, top-k by score
            good = []
            for r in res:
                bw, bh = r.box[2] - r.box[0], r.box[3] - r.box[1]
                if bw < a.min_width or bh < 6 or bw > 0.6 * W or bh > 0.5 * H:
                    continue
                ar = bw / max(1.0, bh)
                if r.plate_type == "type1a" and not (0.9 <= ar <= 3.0):
                    continue
                if r.plate_type in ("type1", "type1b") and not (1.8 <= ar <= 7.5):
                    continue
                good.append(r)
            res = sorted(good, key=lambda r: -r.det_score)[: a.max_per_image]
            if not res:
                continue
            n_img += 1
            for j, r in enumerate(res):
                q = np.array(r.quad, np.float32).reshape(4, 2)
                stem = os.path.splitext(fn)[0]
                cid = f"{stem[:28]}_{stem[-8:]}_{j}" if len(stem) > 36 else f"{stem}_{j}"
                cid = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in cid)
                rect = rectify(img, q, RECT_2ROW if r.plate_type == "type1a" else RECT_1ROW, margin=0.08)
                rect = cv2.resize(rect, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
                cv2.imwrite(os.path.join(a.out, "crops", cid + ".jpg"), rect, [cv2.IMWRITE_JPEG_QUALITY, 92])
                x0, y0, x1, y1 = r.box
                bw, bh = x1 - x0, y1 - y0
                cx0, cy0 = int(max(0, x0 - 2.5 * bw)), int(max(0, y0 - 4 * bh))
                cx1, cy1 = int(min(W, x1 + 2.5 * bw)), int(min(H, y1 + 3 * bh))
                ctx = img[cy0:cy1, cx0:cx1].copy()
                cv2.polylines(ctx, [(q - [cx0, cy0]).astype(int)], True, (0, 255, 0), 2)
                cv2.imwrite(os.path.join(a.out, "ctx", cid + ".jpg"), ctx, [cv2.IMWRITE_JPEG_QUALITY, 85])
                w.writerow([cid, fn, ",".join(f"{v:.1f}" for v in r.quad), f"{x0:.1f},{y0:.1f},{bw:.1f},{bh:.1f}", r.plate_type,
                            f"{r.det_score:.3f}", r.plate_num, f"{r.ocr_score:.3f}", r.is_vehicle, f"{persons:.3f}", W, H])
                n_pl += 1
            f.flush()
            if (k + 1) % 100 == 0:
                print(f"{k + 1}/{len(files)} images, {n_img} with plates, {n_pl} plates", file=sys.stderr)
    print(f"done: {n_img} images with plates, {n_pl} candidates -> {cand_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
