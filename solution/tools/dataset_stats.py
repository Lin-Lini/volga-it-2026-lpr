#!/usr/bin/env python3
"""Print dataset statistics (Markdown) from meta.csv: images / plates / unique numbers per type, real vs synthetic,
licenses, sources, conditions. Used to fill the datasheet (dataset/README.md).

    python tools/dataset_stats.py --dataset ../dataset
"""
import argparse
import csv
import os
import re
from collections import Counter, defaultdict

TYPES = ["type1", "type1a", "type1b", "other"]

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", required=True)
a = ap.parse_args()
rows = list(csv.DictReader(open(os.path.join(a.dataset, "meta.csv"), newline="", encoding="utf-8"), delimiter=";"))
real = [r for r in rows if r["is_synthetic"] == "0"]
syn = [r for r in rows if r["is_synthetic"] == "1"]

def table(rs, title):
    imgs = defaultdict(set); plates = Counter(); uniq = defaultdict(set)
    for r in rs:
        imgs[r["plate_type"]].add(r["image"]); plates[r["plate_type"]] += 1; uniq[r["plate_type"]].add(r["plate_num"])
    print(f"\n**{title}** — изображений: {len({r['image'] for r in rs})}, знаков: {len(rs)}\n")
    print("| Тип | Изображений | Знаков | Уникальных номеров |")
    print("|---|---:|---:|---:|")
    for t in TYPES:
        print(f"| {t} | {len(imgs[t])} | {plates[t]} | {len(uniq[t])} |")

table(real, "Реальные данные")
table(syn, "Синтетические данные")
lic = Counter(r["license"] for r in real)
print("\n**Лицензии реальных изображений (по знакам):** " + ", ".join(f"{k} — {v}" for k, v in lic.most_common()))
src = Counter()
for r in real:
    m = re.match(r"https?://([^/]+)/", r["source"])
    src[m.group(1) if m else r["source"]] += 1
print("\n**Источники:** " + ", ".join(f"{k} — {v}" for k, v in src.most_common()))
cond = Counter()
for r in rows:
    for c in r["conditions"].split(","):
        if c:
            cond[c] += 1
print("\n**Условия съёмки (все данные):** " + ", ".join(f"{k} — {v}" for k, v in cond.most_common()))
veh = Counter(r["is_vehicle"] for r in real)
print(f"\n**is_vehicle (реальные):** 1 — {veh.get('1', 0)}, 0 — {veh.get('0', 0)}")
unread = sum(1 for r in real if "#" in r["plate_num"])
print(f"\n**Знаков с нечитаемыми позициями (#) в реальных данных:** {unread}")
