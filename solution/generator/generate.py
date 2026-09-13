#!/usr/bin/env python3
"""Synthetic dataset generator for Russian registration plates (types 1, 1A, 1B and negatives).

Reproducible: the whole set is a pure function of ``--seed`` (and of the background pool).

    python generate.py --out ../dataset --n 5000 --seed 42 \
        --backgrounds ../dataset/images/real --meta ../dataset/meta.csv

Outputs
    <out>/images/synthetic/synth_XXXXXX.jpg     scene images
    <out>/labels/synth_XXXXXX.txt               YOLO-pose lines: cls cx cy w h x1 y1 x2 y2 x3 y3 x4 y4 (normalised)
    <out>/meta_synthetic.csv                    rows in the meta.csv format of the task (';' separated)
    <out>/ocr/...                               optional rectified crops for the recogniser (--ocr-crops)

Scene construction: a background (a real annotated photo whose plates are *replaced* in place by freshly
rendered plates, or a procedural background) receives 1..3 plates warped with a random 3-D pose, matched
lighting, dirt, frame, occlusions and camera degradations (blur, noise, JPEG, night, rain, snow, glare).
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import random
import sys
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plate_render import Fonts, build_plate, render_plate  # noqa: E402

CLASSES = ["type1", "type1a", "type1b", "other"]
KIND_WEIGHTS = {"type1": 0.22, "type1a": 0.33, "type1b": 0.30, "other": 0.15}


# ----------------------------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------------------------
def rot_matrix(yaw: float, pitch: float, roll: float) -> np.ndarray:
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cr, sr = math.cos(roll), math.sin(roll)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
    Rz = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]])
    return Rz @ Rx @ Ry


def project_quad(w_mm: float, h_mm: float, yaw: float, pitch: float, roll: float, focal: float = 2.2) -> np.ndarray:
    """Corners (TL, TR, BR, BL) of a rotated rectangle seen by a pinhole camera; unit-normalised width."""
    pts = np.array([[-w_mm / 2, -h_mm / 2, 0], [w_mm / 2, -h_mm / 2, 0], [w_mm / 2, h_mm / 2, 0], [-w_mm / 2, h_mm / 2, 0]], dtype=np.float64)
    pts = pts / w_mm
    R = rot_matrix(yaw, pitch, roll)
    p = pts @ R.T
    z = p[:, 2] + focal
    proj = p[:, :2] * (focal / z)[:, None]
    proj -= proj.mean(axis=0)
    return proj


def quad_bbox(q: np.ndarray) -> Tuple[float, float, float, float]:
    x0, y0 = q[:, 0].min(), q[:, 1].min()
    return float(x0), float(y0), float(q[:, 0].max() - x0), float(q[:, 1].max() - y0)


def order_quad(q: np.ndarray) -> np.ndarray:
    """Return corners ordered clockwise from the top-left one."""
    c = q.mean(axis=0)
    ang = np.arctan2(q[:, 1] - c[1], q[:, 0] - c[0])
    idx = np.argsort(ang)
    q = q[idx]
    s = q.sum(axis=1)
    start = int(np.argmin(s))
    return np.roll(q, -start, axis=0)


def procedural_background(rng: random.Random, W: int, H: int) -> np.ndarray:
    nrng = np.random.RandomState(rng.randint(0, 2**31 - 1))
    base = np.array([rng.randint(20, 235) for _ in range(3)], dtype=np.float32)
    img = np.ones((H, W, 3), np.float32) * base
    # gradient
    gx = np.linspace(-1, 1, W)[None, :, None]
    gy = np.linspace(-1, 1, H)[:, None, None]
    img += (gx * rng.uniform(-40, 40) + gy * rng.uniform(-40, 40))
    # random shapes (car bodies, walls, bumpers)
    for _ in range(rng.randint(3, 14)):
        col = tuple(int(v) for v in nrng.randint(0, 255, 3))
        if rng.random() < 0.5:
            x0, y0 = rng.randint(0, W), rng.randint(0, H)
            cv2.rectangle(img, (x0, y0), (x0 + rng.randint(20, W), y0 + rng.randint(20, H)), col, -1)
        else:
            cv2.ellipse(img, (rng.randint(0, W), rng.randint(0, H)), (rng.randint(10, W // 2), rng.randint(10, H // 2)),
                        rng.uniform(0, 180), 0, 360, col, -1)
    img = np.clip(img, 0, 255).astype(np.uint8)
    # some text-like clutter (signs) so the detector learns that not every text is a plate
    for _ in range(rng.randint(0, 3)):
        txt = "".join(rng.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789") for _ in range(rng.randint(3, 8)))
        cv2.putText(img, txt, (rng.randint(0, W), rng.randint(20, H)), cv2.FONT_HERSHEY_SIMPLEX, rng.uniform(0.6, 2.5),
                    tuple(int(v) for v in nrng.randint(0, 255, 3)), rng.randint(1, 4), cv2.LINE_AA)
    img = img.astype(np.float32) + nrng.normal(0, rng.uniform(2, 12), img.shape).astype(np.float32)
    if rng.random() < 0.5:
        img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.5, 3.0))
    return np.clip(img, 0, 255).astype(np.uint8)


# ----------------------------------------------------------------------------------------------
# plate-level effects (applied on the rectified high-res plate, BGR uint8)
# ----------------------------------------------------------------------------------------------
def plate_effects(img: np.ndarray, rng: random.Random, boxes: List, conds: set) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """Returns (plate_bgr, alpha_mask, occluded_glyph_indices)."""
    H, W = img.shape[:2]
    nrng = np.random.RandomState(rng.randint(0, 2**31 - 1))
    img = img.astype(np.float32)
    alpha = np.full((H, W), 255, np.uint8)
    # frame around the plate (holder)
    if rng.random() < 0.55:
        t = rng.randint(max(2, H // 40), max(3, H // 12))
        col = rng.choice([(20, 20, 20), (40, 40, 40), (200, 200, 200), (90, 90, 90), (30, 30, 120)])
        pad = np.full((H + 2 * t, W + 2 * t, 3), col, np.float32)
        pad[t:t + H, t:t + W] = img
        img = pad
        alpha = np.full((H + 2 * t, W + 2 * t), 255, np.uint8)
        for b in boxes:
            b[1] += t; b[2] += t; b[3] += t; b[4] += t
        H, W = img.shape[:2]
    # bolts
    if rng.random() < 0.5:
        for cx in (int(W * 0.12), int(W * 0.88)):
            cv2.circle(img, (cx, int(H * rng.choice([0.2, 0.8]))), max(2, H // 30), (60, 60, 60), -1)
    # dirt blotches
    occluded = []
    if rng.random() < 0.35:
        conds.add("dirt")
        mask = np.zeros((H, W), np.float32)
        for _ in range(rng.randint(2, 10)):
            cv2.ellipse(mask, (rng.randint(0, W), rng.randint(0, H)), (rng.randint(W // 30, W // 5), rng.randint(H // 12, H // 2)),
                        rng.uniform(0, 180), 0, 360, 1.0, -1)
        mask = cv2.GaussianBlur(mask, (0, 0), max(1.0, W / 60)) * rng.uniform(0.25, 0.95)
        dirt_col = np.array(rng.choice([(60, 70, 90), (70, 90, 110), (110, 120, 130), (150, 160, 170), (40, 40, 40)]), np.float32)
        img = img * (1 - mask[..., None]) + dirt_col * mask[..., None]
        # a glyph is unreadable if dirt is dense over it
        for i, b in enumerate(boxes):
            x0, y0, x1, y1 = [int(v) for v in b[1:5]]
            region = mask[max(0, y0):max(1, y1), max(0, x0):max(1, x1)]
            if region.size and (region > 0.8).mean() > 0.8:
                occluded.append(i)
    # scratches / wear
    if rng.random() < 0.3:
        for _ in range(rng.randint(1, 6)):
            p0 = (rng.randint(0, W), rng.randint(0, H))
            p1 = (rng.randint(0, W), rng.randint(0, H))
            cv2.line(img, p0, p1, (rng.randint(80, 230),) * 3, rng.randint(1, max(2, H // 50)), cv2.LINE_AA)
    # hard occluder (tow bar, snow band, sticker)
    if rng.random() < 0.12:
        oc = np.zeros((H, W), np.uint8)
        if rng.random() < 0.5:
            x0 = rng.randint(0, W - W // 6)
            cv2.rectangle(oc, (x0, 0), (x0 + rng.randint(W // 20, W // 6), H), 255, -1)
        else:
            y0 = rng.randint(0, H - H // 4)
            cv2.rectangle(oc, (0, y0), (W, y0 + rng.randint(H // 5, H // 2)), 255, -1)
        col = np.array(rng.choice([(30, 30, 30), (230, 230, 230), (20, 60, 200), (0, 0, 180)]), np.float32)
        m = (oc > 0)[..., None]
        img = np.where(m, col, img)
        for i, b in enumerate(boxes):
            x0, y0, x1, y1 = [int(v) for v in b[1:5]]
            region = oc[max(0, y0):max(1, y1), max(0, x0):max(1, x1)]
            if region.size and (region > 0).mean() > 0.45 and i not in occluded:
                occluded.append(i)
    # lighting gradient / shading
    gx = np.linspace(-1, 1, W)[None, :]
    gy = np.linspace(-1, 1, H)[:, None]
    shade = 1.0 + gx * rng.uniform(-0.25, 0.25) + gy * rng.uniform(-0.25, 0.25)
    img = img * shade[..., None]
    # glare (specular blob)
    if rng.random() < 0.22:
        conds.add("glare")
        gl = np.zeros((H, W), np.float32)
        cv2.ellipse(gl, (rng.randint(0, W), rng.randint(0, H)), (rng.randint(W // 8, W // 2), rng.randint(H // 4, H)), rng.uniform(0, 180), 0, 360, 1.0, -1)
        gl = cv2.GaussianBlur(gl, (0, 0), W / 12) * rng.uniform(0.4, 1.0)
        img = img + 255 * gl[..., None]
    img = np.clip(img, 0, 255)
    # colour cast / contrast
    img = (img - 128) * rng.uniform(0.7, 1.15) + 128 + nrng.uniform(-12, 12, 3)[None, None, :]
    return np.clip(img, 0, 255).astype(np.uint8), alpha, occluded


# ----------------------------------------------------------------------------------------------
# image-level degradations
# ----------------------------------------------------------------------------------------------
def motion_blur(img: np.ndarray, k: int, angle: float) -> np.ndarray:
    kern = np.zeros((k, k), np.float32)
    kern[k // 2, :] = 1.0
    M = cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), angle, 1.0)
    kern = cv2.warpAffine(kern, M, (k, k))
    kern /= max(kern.sum(), 1e-6)
    return cv2.filter2D(img, -1, kern)


def scene_effects(img: np.ndarray, rng: random.Random, conds: set, light: bool = False) -> np.ndarray:
    H, W = img.shape[:2]
    nrng = np.random.RandomState(rng.randint(0, 2**31 - 1))
    f = img.astype(np.float32)
    k_ = 0.5 if light else 1.0     # probability scale for degradations
    r = rng.random()
    if r < 0.22:                      # night
        conds.add("night")
        f = f * rng.uniform(0.15, 0.5)
        f[..., 0] *= rng.uniform(1.0, 1.3)   # bluish (BGR)
        f[..., 2] *= rng.uniform(0.7, 1.0)
        # headlight / street lamp bloom
        if rng.random() < 0.6:
            gl = np.zeros((H, W), np.float32)
            cv2.circle(gl, (rng.randint(0, W), rng.randint(0, H)), rng.randint(W // 10, W // 3), 1.0, -1)
            gl = cv2.GaussianBlur(gl, (0, 0), W / 10)
            f = f + 180 * gl[..., None] * rng.uniform(0.3, 1.0)
        f += nrng.normal(0, rng.uniform(4, 14), f.shape)
    elif r < 0.30:                    # dusk / low contrast fog
        f = (f - 128) * rng.uniform(0.45, 0.8) + 128 + rng.uniform(-30, 30)
    if rng.random() < 0.15 * k_:           # rain streaks
        conds.add("rain")
        layer = np.zeros((H, W), np.float32)
        n = rng.randint(80, 400)
        ang = rng.uniform(-20, 20)
        for _ in range(n):
            x, y = rng.randint(0, W), rng.randint(0, H)
            ln = rng.randint(H // 40, H // 10)
            dx = int(ln * math.sin(math.radians(ang)))
            cv2.line(layer, (x, y), (x + dx, y + ln), 1.0, 1)
        layer = cv2.GaussianBlur(layer, (0, 0), 0.8)
        f = f * (1 - 0.35 * layer[..., None]) + 200 * 0.35 * layer[..., None]
        f = cv2.GaussianBlur(f, (0, 0), rng.uniform(0.3, 1.0))
    if rng.random() < 0.12 * k_:           # snow
        conds.add("snow")
        n = rng.randint(100, 900)
        for _ in range(n):
            cv2.circle(f, (rng.randint(0, W), rng.randint(0, H)), rng.randint(1, max(2, W // 300)), (235, 235, 235), -1)
        f = (f - 128) * rng.uniform(0.75, 0.95) + 128 + rng.uniform(5, 25)
    if rng.random() < 0.25 * k_:
        k = rng.choice([3, 5, 7] if light else [3, 5, 7, 9, 11])
        f = motion_blur(f, k, rng.uniform(0, 180))
        if k >= 7:
            conds.add("motion_blur")
    if rng.random() < 0.35 * k_:
        f = cv2.GaussianBlur(f, (0, 0), rng.uniform(0.3, 1.0 if light else 1.6))
    f += nrng.normal(0, rng.uniform(0, 9), f.shape)
    # slight colour / gamma
    gamma = rng.uniform(0.75, 1.3)
    f = 255.0 * np.power(np.clip(f, 0, 255) / 255.0, gamma)
    img = np.clip(f, 0, 255).astype(np.uint8)
    # resolution loss
    if rng.random() < 0.3 * k_:
        s = rng.uniform(0.6 if light else 0.4, 0.85)
        small = cv2.resize(img, (max(8, int(W * s)), max(8, int(H * s))), interpolation=cv2.INTER_AREA)
        img = cv2.resize(small, (W, H), interpolation=rng.choice([cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_NEAREST]))
    # JPEG
    q = rng.randint(35, 95)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
    if ok:
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return img


# ----------------------------------------------------------------------------------------------
# background pool
# ----------------------------------------------------------------------------------------------
class Backgrounds:
    def __init__(self, folder: Optional[str], meta_csv: Optional[str], rng: random.Random, upper_only: bool = False):
        self.files: List[str] = []
        self.plates: Dict[str, List[dict]] = {}
        self.upper_only = upper_only
        if folder and os.path.isdir(folder):
            for root, _, fs in os.walk(folder):
                for f in fs:
                    if f.lower().endswith((".jpg", ".jpeg", ".png")):
                        self.files.append(os.path.join(root, f))
            self.files.sort()
        if meta_csv and os.path.exists(meta_csv):
            with open(meta_csv, newline="", encoding="utf-8") as fh:
                for row in csv.DictReader(fh, delimiter=";"):
                    if row.get("is_synthetic", "0") == "1":
                        continue
                    key = os.path.basename(row["image"])
                    q = [float(v) for v in row["quad"].split(",")]
                    self.plates.setdefault(key, []).append(dict(quad=np.array(q, np.float32).reshape(4, 2), plate_type=row["plate_type"],
                                                                is_vehicle=row.get("is_vehicle", "1")))
        self.rng = rng

    def __len__(self):
        return len(self.files)

    def sample(self, W: int, H: int) -> Tuple[np.ndarray, List[dict]]:
        """Random background of size WxH plus the list of plate quads (already transformed) it contains."""
        rng = self.rng
        path = rng.choice(self.files)
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            return procedural_background(rng, W, H), []
        plates = [dict(p) for p in self.plates.get(os.path.basename(path), [])]
        h, w = img.shape[:2]
        if self.upper_only:
            img = img[: max(32, int(h * 0.45))]
            h = img.shape[0]
            plates = []
        # random crop that keeps the plates (if any) inside, then resize to WxH
        if plates and rng.random() < 0.85:
            qs = np.concatenate([p["quad"] for p in plates])
            x0, y0 = qs[:, 0].min(), qs[:, 1].min()
            x1, y1 = qs[:, 0].max(), qs[:, 1].max()
            # crop region containing all plates, sized between plate extent and full image
            cw = rng.uniform(max(x1 - x0 + 20, w * 0.3), w)
            ch = cw * H / W
            if ch > h:
                ch = h
                cw = ch * W / H
            cx0 = rng.uniform(max(0, x1 - cw), min(x0, w - cw)) if x1 - cw <= min(x0, w - cw) else max(0, min(x0, w - cw))
            cy0 = rng.uniform(max(0, y1 - ch), min(y0, h - ch)) if y1 - ch <= min(y0, h - ch) else max(0, min(y0, h - ch))
            cx0, cy0 = max(0.0, cx0), max(0.0, cy0)
        else:
            s = rng.uniform(0.35, 1.0)
            cw = w * s
            ch = cw * H / W
            if ch > h:
                ch = h
                cw = ch * W / H
            cx0 = rng.uniform(0, max(0.0, w - cw))
            cy0 = rng.uniform(0, max(0.0, h - ch))
        crop = img[int(cy0):int(cy0 + ch), int(cx0):int(cx0 + cw)]
        if crop.size == 0:
            return procedural_background(rng, W, H), []
        sx, sy = W / crop.shape[1], H / crop.shape[0]
        out = cv2.resize(crop, (W, H), interpolation=cv2.INTER_AREA if sx < 1 else cv2.INTER_LINEAR)
        kept = []
        for p in plates:
            q = (p["quad"] - np.array([cx0, cy0])) * np.array([sx, sy])
            if q[:, 0].min() < -2 or q[:, 1].min() < -2 or q[:, 0].max() > W + 2 or q[:, 1].max() > H + 2:
                continue
            p["quad"] = q
            kept.append(p)
        if len(kept) < len(plates):
            # a plate got cut off: unlabeled plate pixels would poison training -> fall back to procedural bg
            return procedural_background(rng, W, H), []
        return out, kept


# ----------------------------------------------------------------------------------------------
# main composition
# ----------------------------------------------------------------------------------------------
def choose_kind(rng: random.Random, weights: Dict[str, float]) -> str:
    ks = list(weights)
    return rng.choices(ks, weights=[weights[k] for k in ks])[0]


def make_plate(rng: random.Random, fonts: Fonts, kind: str, target_w_px: float, conds: set):
    spec = build_plate(rng, kind)
    scale = max(1.0, min(6.0, 2.2 * target_w_px / spec.width))   # render ~2x the target size for antialiasing
    pil, info = render_plate(spec, fonts, rng, scale=scale)
    bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    boxes = [list(b) for b in info["boxes"]]
    bgr, alpha, occluded = plate_effects(bgr, rng, boxes, conds)
    text = list(spec.text)
    # boxes are stored in glyph order == text order for all builders (region glyphs last, matching text)
    for i in occluded:
        if i < len(text):
            text[i] = "#"
    return spec, bgr, alpha, "".join(text), info


def paste_plate(scene: np.ndarray, plate: np.ndarray, alpha: np.ndarray, quad: np.ndarray, rng: random.Random) -> np.ndarray:
    H, W = scene.shape[:2]
    ph, pw = plate.shape[:2]
    src = np.array([[0, 0], [pw, 0], [pw, ph], [0, ph]], np.float32)
    M = cv2.getPerspectiveTransform(src, quad.astype(np.float32))
    # downscale first when the target is much smaller (proper antialiasing)
    tw = max(1.0, np.linalg.norm(quad[1] - quad[0]))
    if pw > 1.6 * tw:
        s = 1.4 * tw / pw
        plate_s = cv2.resize(plate, (max(2, int(pw * s)), max(2, int(ph * s))), interpolation=cv2.INTER_AREA)
        alpha_s = cv2.resize(alpha, (plate_s.shape[1], plate_s.shape[0]), interpolation=cv2.INTER_AREA)
        src = np.array([[0, 0], [plate_s.shape[1], 0], [plate_s.shape[1], plate_s.shape[0]], [0, plate_s.shape[0]]], np.float32)
        M = cv2.getPerspectiveTransform(src, quad.astype(np.float32))
        plate, alpha = plate_s, alpha_s
    warped = cv2.warpPerspective(plate, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    wa = cv2.warpPerspective(alpha, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT).astype(np.float32) / 255.0
    # soften the edge slightly
    wa = cv2.GaussianBlur(wa, (0, 0), 0.7)
    # match local lighting of the background
    x, y, w, h = quad_bbox(quad)
    x0, y0, x1, y1 = int(max(0, x - w)), int(max(0, y - h)), int(min(W, x + 2 * w)), int(min(H, y + 2 * h))
    local = scene[y0:y1, x0:x1].astype(np.float32).mean() if (y1 > y0 and x1 > x0) else 128.0
    gain = np.clip((local / 128.0) ** rng.uniform(0.3, 0.8), 0.35, 1.5)
    warped = np.clip(warped.astype(np.float32) * gain, 0, 255)
    out = scene.astype(np.float32) * (1 - wa[..., None]) + warped * wa[..., None]
    return np.clip(out, 0, 255).astype(np.uint8)


def random_quad(rng: random.Random, spec, W: int, H: int, conds: set) -> Optional[np.ndarray]:
    # plate width in px, log-uniform, biased to small plates (camera footage)
    wmin, wmax = 22.0, min(W * 0.55, 420.0)
    pw = math.exp(rng.uniform(math.log(wmin), math.log(wmax)))
    yaw = math.radians(rng.uniform(-50, 50)) if rng.random() < 0.7 else math.radians(rng.uniform(-12, 12))
    pitch = math.radians(rng.uniform(-30, 30)) if rng.random() < 0.6 else math.radians(rng.uniform(-8, 8))
    roll = math.radians(rng.uniform(-15, 15)) if rng.random() < 0.5 else math.radians(rng.uniform(-4, 4))
    if abs(math.degrees(yaw)) > 25 or abs(math.degrees(pitch)) > 18:
        conds.add("angle")
    q = project_quad(spec.width, spec.height, yaw, pitch, roll)
    q = q / (q[:, 0].max() - q[:, 0].min()) * pw
    bx, by, bw, bh = quad_bbox(q)
    if bw > W - 4 or bh > H - 4:
        return None
    cx = rng.uniform(bw / 2 + 2, W - bw / 2 - 2)
    cy = rng.uniform(bh / 2 + 2, H - bh / 2 - 2)
    q = q + np.array([cx, cy])
    return q


def quad_iou_overlap(q: np.ndarray, others: List[np.ndarray]) -> bool:
    x, y, w, h = quad_bbox(q)
    for o in others:
        ox, oy, ow, oh = quad_bbox(o)
        ix = max(0, min(x + w, ox + ow) - max(x, ox))
        iy = max(0, min(y + h, oy + oh) - max(y, oy))
        if ix * iy > 0.05 * min(w * h, ow * oh):
            return True
    return False


def rectify(img: np.ndarray, quad: np.ndarray, out_w: int, out_h: int, rng: Optional[random.Random] = None, jitter: float = 0.0) -> np.ndarray:
    q = quad.astype(np.float32).copy()
    if rng is not None and jitter > 0:
        w = np.linalg.norm(q[1] - q[0])
        h = np.linalg.norm(q[3] - q[0])
        q += np.array([[rng.uniform(-jitter, jitter) * w, rng.uniform(-jitter, jitter) * h] for _ in range(4)], np.float32)
    dst = np.array([[0, 0], [out_w, 0], [out_w, out_h], [0, out_h]], np.float32)
    M = cv2.getPerspectiveTransform(q, dst)
    return cv2.warpPerspective(img, M, (out_w, out_h), flags=cv2.INTER_LINEAR)


def crops_mode(a, rng: random.Random, fonts: Fonts, weights: Dict[str, float], bgs: "Backgrounds"):
    """Fast recogniser-only data: plate -> effects -> small scene patch -> degradations -> jittered rectification."""
    os.makedirs(a.ocr_crops, exist_ok=True)
    lab = open(os.path.join(a.ocr_crops, f"labels_{a.prefix}.csv"), "a", encoding="utf-8")
    stats = {k: 0 for k in CLASSES}
    for i in range(a.start, a.start + a.n):
        kind = choose_kind(rng, weights)
        if kind == "other":
            kind = rng.choice(["type1", "type1a", "type1b"])
        conds = set()
        spec_probe = build_plate(random.Random(rng.randint(0, 1 << 30)), kind)
        pw = math.exp(rng.uniform(math.log(40), math.log(300)))
        W = int(pw * rng.uniform(1.3, 2.2)); H = int(pw * spec_probe.height / spec_probe.width * rng.uniform(1.5, 3.0))
        W, H = max(48, W), max(32, H)
        if len(bgs) and rng.random() > a.procedural_frac:
            scene, _ = bgs.sample(W, H)
        else:
            scene = procedural_background(rng, W, H)
        q = random_quad(rng, spec_probe, W, H, conds)
        if q is None:
            continue
        spec, plate, alpha, text, info = make_plate(rng, fonts, kind, np.linalg.norm(q[1] - q[0]), conds)
        scene = paste_plate(scene, plate, alpha, q, rng)
        scene = scene_effects(scene, rng, conds, light=True)
        crop = rectify(scene, q, 192, 48 if spec.kind != "type1a" else 112, rng, jitter=rng.choice([0.0, 0.02, 0.04, 0.07]))
        cname = f"{a.prefix}_{i:07d}_{CLASSES.index(spec.kind)}.jpg"
        cv2.imwrite(os.path.join(a.ocr_crops, cname), crop, [cv2.IMWRITE_JPEG_QUALITY, rng.randint(70, 95)])
        lab.write(f"{cname};{text};{spec.kind}\n")
        stats[spec.kind] += 1
        if (i + 1) % 2000 == 0:
            print(f"{i + 1 - a.start}/{a.n} crops {stats}", file=sys.stderr)
    lab.close()
    print("done", stats, file=sys.stderr)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", default="scenes", choices=["scenes", "crops"], help="scenes: full images + labels; crops: recogniser crops only")
    ap.add_argument("--out", required=True, help="dataset root (images/synthetic, labels, meta_synthetic.csv are created inside)")
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--backgrounds", default=None, help="folder with real background photos")
    ap.add_argument("--meta", default=None, help="meta.csv with quads of real plates in the background photos (they get replaced)")
    ap.add_argument("--bg-upper-only", action="store_true", help="use only the upper part of background photos (no unlabeled plates)")
    ap.add_argument("--procedural-frac", type=float, default=0.25, help="fraction of purely procedural backgrounds")
    ap.add_argument("--min-size", type=int, default=640)
    ap.add_argument("--max-size", type=int, default=1280)
    ap.add_argument("--ocr-crops", default=None, help="also write rectified plate crops + labels for the recogniser into this folder")
    ap.add_argument("--prefix", default="synth")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--weights", default=None, help="kind weights, e.g. type1:0.2,type1a:0.35,type1b:0.3,other:0.15")
    a = ap.parse_args()

    rng = random.Random(a.seed)
    fonts = Fonts()
    weights = dict(KIND_WEIGHTS)
    if a.weights:
        weights = {k: float(v) for k, v in (kv.split(":") for kv in a.weights.split(","))}
    if a.mode == "crops":
        if not a.ocr_crops:
            ap.error("--ocr-crops is required in crops mode")
        crops_mode(a, rng, fonts, weights, Backgrounds(a.backgrounds, a.meta, rng, upper_only=a.bg_upper_only))
        return
    img_dir = os.path.join(a.out, "images", "synthetic")
    lbl_dir = os.path.join(a.out, "labels")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)
    if a.ocr_crops:
        os.makedirs(a.ocr_crops, exist_ok=True)
        ocr_labels = open(os.path.join(a.ocr_crops, f"labels_{a.prefix}.csv"), "a", encoding="utf-8")
    bgs = Backgrounds(a.backgrounds, a.meta, rng, upper_only=a.bg_upper_only)
    meta_path = os.path.join(a.out, "meta_synthetic.csv")
    new_meta = not os.path.exists(meta_path)
    fields = ["image", "plate_num", "plate_type", "bbox", "quad", "is_vehicle", "is_synthetic", "source", "license", "conditions"]
    with open(meta_path, "a", newline="", encoding="utf-8") as mf:
        mw = csv.DictWriter(mf, fieldnames=fields, delimiter=";")
        if new_meta:
            mw.writeheader()
        stats = {k: 0 for k in CLASSES}
        for i in range(a.start, a.start + a.n):
            W = rng.randint(a.min_size, a.max_size)
            H = int(W * rng.choice([0.5625, 0.75, 0.75, 0.6667, 1.0, 1.333]))
            W, H = (W // 32) * 32, (H // 32) * 32
            conds = set()
            use_proc = len(bgs) == 0 or rng.random() < a.procedural_frac
            if use_proc:
                scene, existing = procedural_background(rng, W, H), []
            else:
                scene, existing = bgs.sample(W, H)
            placed: List[np.ndarray] = []
            labels = []
            # 1) replace real plates in place
            for p in existing:
                q = order_quad(p["quad"])
                kind = choose_kind(rng, weights)
                spec, plate, alpha, text, info = make_plate(rng, fonts, kind, np.linalg.norm(q[1] - q[0]), conds)
                # keep the aspect of the synthetic plate: adjust quad height around its centre
                cur_h = (np.linalg.norm(q[3] - q[0]) + np.linalg.norm(q[2] - q[1])) / 2
                cur_w = (np.linalg.norm(q[1] - q[0]) + np.linalg.norm(q[2] - q[3])) / 2
                want_h = cur_w * spec.height / spec.width
                c = q.mean(axis=0)
                q = c + (q - c) * np.array([1.0, want_h / max(cur_h, 1e-3)])
                if q[:, 0].min() < 0 or q[:, 1].min() < 0 or q[:, 0].max() > W or q[:, 1].max() > H:
                    continue
                scene = paste_plate(scene, plate, alpha, q, rng)
                placed.append(q)
                labels.append(dict(kind=spec.kind, text=text, quad=q, is_vehicle=p.get("is_vehicle", "1")))
            # 2) extra plates at random poses
            n_extra = 0 if (existing and rng.random() < 0.6) else rng.choice([1, 1, 1, 2, 2, 3])
            for _ in range(n_extra):
                kind = choose_kind(rng, weights)
                spec_probe = build_plate(random.Random(rng.randint(0, 1 << 30)), kind)
                q = random_quad(rng, spec_probe, W, H, conds)
                if q is None or quad_iou_overlap(q, placed):
                    continue
                spec, plate, alpha, text, info = make_plate(rng, fonts, kind, np.linalg.norm(q[1] - q[0]), conds)
                scene = paste_plate(scene, plate, alpha, q, rng)
                placed.append(q)
                labels.append(dict(kind=spec.kind, text=text, quad=q, is_vehicle="0"))
            if not labels and rng.random() < 0.9:
                continue   # keep a few (~10% of attempts) plate-free negatives
            scene = scene_effects(scene, rng, conds)
            if "night" not in conds:
                conds.add("day")
            name = f"{a.prefix}_{i:06d}.jpg"
            cv2.imwrite(os.path.join(img_dir, name), scene, [cv2.IMWRITE_JPEG_QUALITY, 92])
            with open(os.path.join(lbl_dir, name.replace(".jpg", ".txt")), "w") as lf:
                for L in labels:
                    q = L["quad"].copy()
                    q[:, 0] = np.clip(q[:, 0], 0, W - 1)
                    q[:, 1] = np.clip(q[:, 1], 0, H - 1)
                    x, y, w, h = quad_bbox(q)
                    cls = CLASSES.index(L["kind"])
                    kp = " ".join(f"{q[k, 0] / W:.6f} {q[k, 1] / H:.6f}" for k in range(4))
                    lf.write(f"{cls} {(x + w / 2) / W:.6f} {(y + h / 2) / H:.6f} {w / W:.6f} {h / H:.6f} {kp}\n")
                    stats[L["kind"]] += 1
                    mw.writerow(dict(image=f"images/synthetic/{name}", plate_num=L["text"], plate_type=L["kind"],
                                     bbox=f"{x:.1f},{y:.1f},{w:.1f},{h:.1f}", quad=",".join(f"{v:.1f}" for v in q.reshape(-1)),
                                     is_vehicle=L["is_vehicle"], is_synthetic=1, source="generator", license="CC BY 4.0",
                                     conditions=",".join(sorted(conds))))
                    if a.ocr_crops and L["kind"] != "other" and w >= 16:
                        crop = rectify(scene, q, 192, 48 if L["kind"] != "type1a" else 112, rng, jitter=0.04)
                        cname = f"{a.prefix}_{i:06d}_{len(placed)}_{cls}.jpg"
                        cv2.imwrite(os.path.join(a.ocr_crops, cname), crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
                        ocr_labels.write(f"{cname};{L['text']};{L['kind']}\n")
                        placed.append(q)   # only to make the counter unique
            if (i + 1) % 200 == 0:
                print(f"{i + 1 - a.start}/{a.n}  plates so far: {stats}", file=sys.stderr)
    print("done", stats, file=sys.stderr)


if __name__ == "__main__":
    main()
