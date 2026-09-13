#!/usr/bin/env python3
"""Recognise registration plates in a folder of images and write the result CSV.

    python run.py --input /path/to/images --output result.csv

The input folder can also be given through the environment variable LPR_INPUT or a config file
(--config config.yaml). Everything runs offline; all weights are in ./weights.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time

import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lpr.pipeline import LPRPipeline, pick_device  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


def load_config(path):
    if not path or not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        if path.endswith(".json"):
            return json.load(f)
        try:
            import yaml
            return yaml.safe_load(f) or {}
        except ImportError:
            # minimal YAML fallback: "key: value  # comment" lines only
            cfg = {}
            for line in f:
                line = line.rstrip()
                if not line or line.lstrip().startswith("#") or ":" not in line:
                    continue
                k, v = line.split(":", 1)
                v = re.sub(r"\s+#.*$", "", v).strip().strip("'\"")
                if v.lower() in ("true", "false"):
                    v = v.lower() == "true"
                cfg[k.strip()] = v
            return cfg


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", "-i", default=os.environ.get("LPR_INPUT"), help="folder with .jpg/.png images (or env LPR_INPUT)")
    ap.add_argument("--output", "-o", default=os.environ.get("LPR_OUTPUT", "result.csv"), help="output CSV path (or env LPR_OUTPUT)")
    ap.add_argument("--config", default=os.environ.get("LPR_CONFIG", os.path.join(HERE, "config.yaml")))
    ap.add_argument("--device", default=None, help="cuda:0 | cpu | mps | auto")
    ap.add_argument("--det-weights", default=None)
    ap.add_argument("--ocr-weights", default=None)
    ap.add_argument("--vehicle-weights", default=None)
    ap.add_argument("--imgsz", type=int, default=None)
    ap.add_argument("--det-conf", type=float, default=None)
    ap.add_argument("--unk-thr", type=float, default=None, help="per-character confidence below which '#' is emitted")
    ap.add_argument("--min-conf", type=float, default=None, help="drop plates with confidence below this")
    ap.add_argument("--unk-drop-frac", type=float, default=None, help="readings with at least this fraction of '#' are dropped when confidence < --unk-drop-conf")
    ap.add_argument("--unk-drop-conf", type=float, default=None)
    ap.add_argument("--no-vehicle-gate", action="store_true", help="do not check that the plate sits on a vehicle")
    ap.add_argument("--drop-off-vehicle", action="store_true", help="omit plates that are not on a vehicle (default: keep, lower confidence)")
    ap.add_argument("--no-other", action="store_true", help="do not output plates classified as 'other'")
    ap.add_argument("--half", action="store_true", help="FP16 inference on CUDA")
    ap.add_argument("--recursive", action="store_true")
    ap.add_argument("--debug-json", default=None, help="write detailed per-plate json (boxes, quads, timings)")
    a = ap.parse_args()

    cfg = load_config(a.config)

    def opt(name, default):
        v = getattr(a, name.replace("-", "_"), None)
        if v is not None:
            return v
        return cfg.get(name, cfg.get(name.replace("-", "_"), default))

    inp = a.input or cfg.get("input")
    if not inp or not os.path.isdir(inp):
        ap.error("input folder not found; pass --input DIR, set LPR_INPUT or put `input:` in config.yaml")
    out_csv = a.output or cfg.get("output", "result.csv")
    device = pick_device(str(opt("device", "auto")))
    det_w = opt("det-weights", os.path.join(HERE, "weights", "det.pt"))
    ocr_w = opt("ocr-weights", os.path.join(HERE, "weights", "ocr.pt"))
    veh_w = opt("vehicle-weights", os.path.join(HERE, "weights", "yolo11n.pt"))
    vehicle_gate = not a.no_vehicle_gate and bool(cfg.get("vehicle_gate", True))
    emit_other = not a.no_other and bool(cfg.get("emit_other", True))
    drop_off = a.drop_off_vehicle or bool(cfg.get("drop_off_vehicle", False))

    t_load = time.perf_counter()
    pipe = LPRPipeline(det_w, ocr_w, veh_w if vehicle_gate else None, device=device, det_imgsz=int(opt("imgsz", 640)),
                       det_conf=float(opt("det-conf", 0.25)), unk_thr=float(opt("unk-thr", 0.45)), min_conf=float(opt("min-conf", 0.15)),
                       vehicle_gate=vehicle_gate, emit_other=emit_other, half=bool(a.half or cfg.get("half", False)),
                       unk_drop_frac=float(opt("unk-drop-frac", 0.34)), unk_drop_conf=float(opt("unk-drop-conf", 0.5)))
    pipe.warmup()
    print(f"[lpr] device={device} models loaded in {time.perf_counter() - t_load:.1f}s", file=sys.stderr)

    files = []
    if a.recursive:
        for root, _, fs in os.walk(inp):
            files += [os.path.join(root, f) for f in fs if f.lower().endswith(IMG_EXT)]
    else:
        files = [os.path.join(inp, f) for f in os.listdir(inp) if f.lower().endswith(IMG_EXT)]
    files.sort()
    print(f"[lpr] {len(files)} images in {inp}", file=sys.stderr)

    rows, debug = [], []
    total_t, n_done = 0.0, 0
    for path in files:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            print(f"[lpr] cannot read {path}", file=sys.stderr)
            continue
        t0 = time.perf_counter()
        res = pipe.process(img)
        dt = time.perf_counter() - t0
        total_t += dt
        n_done += 1
        name = os.path.basename(path)
        for r in res:
            conf = r.confidence
            if r.is_vehicle == 0:
                if drop_off:
                    continue
                conf = round(conf * 0.6, 4)
            rows.append(dict(image=name, plate_num=r.plate_num, plate_type=r.plate_type, confidence=f"{conf:.4f}"))
            if a.debug_json:
                debug.append(dict(image=name, **{k: v for k, v in r.__dict__.items()}))
        if n_done % 50 == 0:
            print(f"[lpr] {n_done}/{len(files)}  avg {1000 * total_t / n_done:.1f} ms/img", file=sys.stderr)

    os.makedirs(os.path.dirname(os.path.abspath(out_csv)) or ".", exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["image", "plate_num", "plate_type", "confidence"], delimiter=";")
        w.writeheader()
        w.writerows(rows)
    if a.debug_json:
        with open(a.debug_json, "w", encoding="utf-8") as f:
            json.dump(debug, f, ensure_ascii=False, indent=1)
    print(f"[lpr] done: {n_done} images, {len(rows)} plates, avg {1000 * total_t / max(1, n_done):.1f} ms/img -> {out_csv}", file=sys.stderr)


if __name__ == "__main__":
    main()
