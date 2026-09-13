#!/usr/bin/env python3
"""Measure per-image latency of the pipeline (detector / vehicle gate / OCR / total).

    python tools/benchmark.py --input some/images --device cuda:0 [--n 200] [--imgsz 640]
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from lpr.pipeline import LPRPipeline, pick_device  # noqa: E402

HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--half", action="store_true")
    ap.add_argument("--no-vehicle-gate", action="store_true")
    a = ap.parse_args()
    device = pick_device(a.device)
    pipe = LPRPipeline(os.path.join(HERE, "weights", "det.pt"), os.path.join(HERE, "weights", "ocr.pt"),
                       None if a.no_vehicle_gate else os.path.join(HERE, "weights", "yolo11n.pt"), device=device,
                       det_imgsz=a.imgsz, half=a.half)
    pipe.warmup()
    files = sorted(f for f in os.listdir(a.input) if f.lower().endswith((".jpg", ".jpeg", ".png")))[: a.n]
    imgs = [cv2.imread(os.path.join(a.input, f)) for f in files]
    imgs = [i for i in imgs if i is not None]
    tot, det, veh, ocr, plates = [], [], [], [], 0
    for img in imgs:
        t0 = time.perf_counter()
        res = pipe.process(img)
        tot.append(time.perf_counter() - t0)
        if res:
            det.append(res[0].timings["det"]); veh.append(res[0].timings["veh"]); ocr.append(res[0].timings["ocr"])
            plates += len(res)
    tot = np.array(tot) * 1000
    print(f"device={device} imgsz={a.imgsz} half={a.half} vehicle_gate={not a.no_vehicle_gate}")
    print(f"images={len(imgs)} plates={plates}")
    print(f"total  : mean {tot.mean():.1f} ms  median {np.median(tot):.1f}  p95 {np.percentile(tot, 95):.1f}  max {tot.max():.1f}")
    if det:
        print(f"detector {1000 * np.mean(det):.1f} ms | vehicle gate {1000 * np.mean(veh):.1f} ms | ocr {1000 * np.mean(ocr):.1f} ms (images with plates)")
    print(f"throughput: {1000 / tot.mean():.1f} img/s")


if __name__ == "__main__":
    main()
