#!/usr/bin/env python3
"""Export the detector and the recogniser to ONNX (optional; for ONNX Runtime / TensorRT deployments).

    python tools/export_onnx.py --det weights/det.pt --ocr weights/ocr.pt --out weights
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from lpr.ocr import CRNN, IN_H, IN_W  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--det", default="weights/det.pt")
    ap.add_argument("--ocr", default="weights/ocr.pt")
    ap.add_argument("--out", default="weights")
    ap.add_argument("--imgsz", type=int, default=640)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.det and os.path.exists(a.det):
        from ultralytics import YOLO
        path = YOLO(a.det).export(format="onnx", imgsz=a.imgsz, dynamic=False, simplify=True, opset=12)
        print("detector ->", path)
    if a.ocr and os.path.exists(a.ocr):
        m = CRNN()
        st = torch.load(a.ocr, map_location="cpu")
        m.load_state_dict(st["model"] if "model" in st else st)
        m.eval()
        dummy = torch.zeros(1, 1, IN_H, IN_W)
        out = os.path.join(a.out, "ocr.onnx")
        torch.onnx.export(m, dummy, out, input_names=["image"], output_names=["logits"],
                          dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}}, opset_version=12)
        print("recogniser ->", out)


if __name__ == "__main__":
    main()
