#!/usr/bin/env python3
"""Train the CRNN recogniser on rectified plate crops.

    python train/train_ocr.py --crops work/boot/ocr --crops dataset/ocr_real --epochs 30 --out weights/ocr.pt

Each --crops folder must contain labels_*.csv files with lines  "<file>;<text>;<kind>".
Square plates (kind == type1a, 192x112 crops) are split into two row samples; single-row plates are used as is.
"""
from __future__ import annotations

import argparse
import glob
import os
import random
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from lpr.ocr import BLANK, CRNN, IN_H, IN_W, greedy_decode, preprocess  # noqa: E402
from lpr.plates import ALPHABET, CHAR2IDX, normalise  # noqa: E402


def split_rows(img: np.ndarray, text: str, rng: random.Random):
    """type1a crop (2 rows) -> [(row_img, row_text)] with jittered split."""
    h = img.shape[0]
    top_end = int(h * rng.uniform(0.50, 0.57))
    bot_start = int(h * rng.uniform(0.44, 0.52))
    top = img[int(h * rng.uniform(0.02, 0.10)):top_end]
    bot = img[bot_start:int(h * rng.uniform(0.93, 1.0))]
    return [(top, text[:4]), (bot, text[4:])]


class CropDataset(Dataset):
    def __init__(self, folders, train=True, seed=0, limit=None):
        self.items = []
        for folder in folders:
            repeat = 1
            if ":" in folder and folder.rsplit(":", 1)[1].isdigit():      # "path:N" oversamples the folder N times
                folder, repeat = folder.rsplit(":", 1)[0], int(folder.rsplit(":", 1)[1])
            for lab in glob.glob(os.path.join(folder, "labels_*.csv")):
                for line in open(lab, encoding="utf-8"):
                    parts = line.strip().split(";")
                    if len(parts) < 3:
                        continue
                    fn, text, kind = parts[0], normalise(parts[1]), parts[2]
                    if not text or any(c not in CHAR2IDX for c in text):
                        continue
                    self.items.extend([(os.path.join(folder, fn), text, kind)] * repeat)
        rng = random.Random(seed)
        rng.shuffle(self.items)
        if limit:
            self.items = self.items[:limit]
        self.train = train
        self.rng = rng

    def __len__(self):
        return len(self.items)

    def augment(self, img: np.ndarray) -> np.ndarray:
        rng = self.rng
        h, w = img.shape[:2]
        if rng.random() < 0.5:   # small affine
            M = cv2.getRotationMatrix2D((w / 2, h / 2), rng.uniform(-4, 4), rng.uniform(0.9, 1.1))
            M[0, 2] += rng.uniform(-w * 0.04, w * 0.04)
            M[1, 2] += rng.uniform(-h * 0.06, h * 0.06)
            img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
        if rng.random() < 0.3:
            img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.3, 1.2))
        if rng.random() < 0.3:
            k = rng.choice([3, 5, 7])
            kern = np.zeros((k, k), np.float32); kern[k // 2, :] = 1.0 / k
            img = cv2.filter2D(img, -1, kern)
        if rng.random() < 0.5:
            img = np.clip(img.astype(np.float32) * rng.uniform(0.6, 1.4) + rng.uniform(-40, 40), 0, 255).astype(np.uint8)
        if rng.random() < 0.3:
            img = np.clip(img.astype(np.float32) + np.random.normal(0, rng.uniform(2, 15), img.shape), 0, 255).astype(np.uint8)
        if rng.random() < 0.15:   # inverted plates (night IR cameras)
            img = 255 - img
        if rng.random() < 0.2:    # random horizontal crop of the border
            c = int(w * rng.uniform(0, 0.06))
            img = img[:, c:w - int(w * rng.uniform(0, 0.06))] if w > 2 * c + 8 else img
        return img

    def __getitem__(self, i):
        path, text, kind = self.items[i]
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            img = np.zeros((48, 192, 3), np.uint8)
        if kind == "type1a" and img.shape[0] > img.shape[1] * 0.4:
            rows = split_rows(img, text, self.rng)
            img, text = rows[self.rng.randint(0, 1)] if self.train else rows[i % 2]
        if self.train:
            img = self.augment(img)
        x = preprocess(img)
        y = [CHAR2IDX[c] for c in text]
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long), text


def collate(batch):
    xs, ys, texts = zip(*batch)
    lengths = torch.tensor([len(y) for y in ys], dtype=torch.long)
    return torch.stack(xs), torch.cat(ys), lengths, list(texts)


def evaluate(model, loader, device):
    model.eval()
    ok = tot = 0
    cer_num = cer_den = 0
    with torch.no_grad():
        for x, y, lengths, texts in loader:
            logits = model(x.to(device))
            dec = greedy_decode(logits.float().cpu())
            for (pred, _), gt in zip(dec, texts):
                tot += 1
                ok += int(pred == gt)
                cer_num += levenshtein(pred, gt)
                cer_den += len(gt)
    model.train()
    return ok / max(1, tot), cer_num / max(1, cer_den)


def levenshtein(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crops", action="append", required=True)
    ap.add_argument("--val-crops", action="append", default=[])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", default="weights/ocr.pt")
    ap.add_argument("--init", default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    random.seed(a.seed)
    np.random.seed(a.seed)
    device = a.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print("device:", device)
    full = CropDataset(a.crops, train=True, seed=a.seed, limit=a.limit)
    n_val = max(200, int(0.03 * len(full)))
    val_items = full.items[:n_val]
    full.items = full.items[n_val:]
    val = CropDataset([], train=False, seed=a.seed)
    val.items = val_items
    if a.val_crops:
        extra = CropDataset(a.val_crops, train=False, seed=a.seed)
        val.items += extra.items
    print(f"train crops: {len(full)}  val crops: {len(val)}")
    tl = DataLoader(full, batch_size=a.batch, shuffle=True, num_workers=a.workers, collate_fn=collate, drop_last=True, persistent_workers=a.workers > 0)
    vl = DataLoader(val, batch_size=256, shuffle=False, num_workers=0, collate_fn=collate)

    model = CRNN().to(device)
    if a.init and os.path.exists(a.init):
        st = torch.load(a.init, map_location="cpu")
        model.load_state_dict(st["model"] if "model" in st else st)
        print("initialised from", a.init)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    steps = a.epochs * len(tl)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps, pct_start=0.1)
    ctc = nn.CTCLoss(blank=BLANK, zero_infinity=True)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    best = -1.0
    t0 = time.time()
    for ep in range(a.epochs):
        model.train()
        tot_loss = 0.0
        for it, (x, y, lengths, _) in enumerate(tl):
            x = x.to(device)
            logits = model(x)                                   # B,T,C
            logp = logits.log_softmax(-1).permute(1, 0, 2)      # T,B,C
            T = logp.shape[0]
            in_len = torch.full((x.shape[0],), T, dtype=torch.long)
            loss = ctc(logp.cpu() if device == "mps" else logp, y, in_len, lengths)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            tot_loss += float(loss)
        acc, cer = evaluate(model, vl, device)
        print(f"epoch {ep + 1}/{a.epochs} loss {tot_loss / max(1, len(tl)):.4f}  val acc {acc:.4f} cer {cer:.4f}  ({time.time() - t0:.0f}s)", flush=True)
        if acc >= best:
            best = acc
            torch.save({"model": model.state_dict(), "alphabet": ALPHABET, "in_hw": (IN_H, IN_W), "val_acc": acc}, a.out)
    print("best val acc", best, "->", a.out)


if __name__ == "__main__":
    main()
