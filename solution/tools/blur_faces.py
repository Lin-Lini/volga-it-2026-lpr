#!/usr/bin/env python3
"""Detect and blur faces (OpenCV YuNet). Used on every real image before it enters the dataset.

    python tools/blur_faces.py --input work/sel --output dataset/images/real
Returns per-image face counts on stdout (csv). Faces are replaced by a strong Gaussian blur + pixelation.
"""
from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(HERE, "..", "weights", "face_detection_yunet_2023mar.onnx")


class FaceBlurrer:
    def __init__(self, model=MODEL, score=0.6):
        self.det = cv2.FaceDetectorYN.create(model, "", (320, 320), score_threshold=score, nms_threshold=0.3, top_k=500)

    def faces(self, img):
        H, W = img.shape[:2]
        out = []
        # run at two scales to catch small faces
        for scale in (1.0, 2.0):
            if scale != 1.0 and max(H, W) * scale > 3000:
                continue
            im = img if scale == 1.0 else cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            self.det.setInputSize((im.shape[1], im.shape[0]))
            _, f = self.det.detect(im)
            if f is not None:
                for r in f:
                    x, y, w, h = r[:4] / scale
                    out.append((int(x), int(y), int(w), int(h)))
        return out

    def blur(self, img):
        fs = self.faces(img)
        H, W = img.shape[:2]
        for (x, y, w, h) in fs:
            x0, y0 = max(0, int(x - 0.25 * w)), max(0, int(y - 0.3 * h))
            x1, y1 = min(W, int(x + 1.25 * w)), min(H, int(y + 1.3 * h))
            if x1 <= x0 or y1 <= y0:
                continue
            roi = img[y0:y1, x0:x1]
            k = max(3, int(min(roi.shape[:2]) / 8))
            small = cv2.resize(roi, (max(1, roi.shape[1] // k), max(1, roi.shape[0] // k)), interpolation=cv2.INTER_AREA)
            pix = cv2.resize(small, (roi.shape[1], roi.shape[0]), interpolation=cv2.INTER_NEAREST)
            img[y0:y1, x0:x1] = cv2.GaussianBlur(pix, (0, 0), max(2.0, min(roi.shape[:2]) / 6))
        return img, len(fs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    a = ap.parse_args()
    os.makedirs(a.output, exist_ok=True)
    fb = FaceBlurrer()
    print("image;faces")
    for fn in sorted(os.listdir(a.input)):
        if not fn.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        img = cv2.imread(os.path.join(a.input, fn))
        if img is None:
            continue
        img, n = fb.blur(img)
        cv2.imwrite(os.path.join(a.output, fn), img, [cv2.IMWRITE_JPEG_QUALITY, 93])
        print(f"{fn};{n}")


if __name__ == "__main__":
    main()
