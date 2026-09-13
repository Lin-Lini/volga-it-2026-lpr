#!/usr/bin/env python3
"""Compare a result CSV with a ground-truth CSV (both ';'-separated, task format).

    python tools/evaluate.py --gt gt.csv --pred result.csv

Matching: within an image, predictions are matched greedily to ground-truth rows by string similarity
(normalised Levenshtein). Reported per plate type and overall:
  * detection recall / precision (a GT plate counts as detected if any prediction matched it)
  * exact plate_num accuracy among detected plates, '#'-tolerant accuracy, character accuracy
  * plate_type accuracy and confusion
  * false positives on 'other' (a prediction with a standard type for a GT 'other' plate)
GT rows without plates ('' plate_num) are ignored; images absent from GT are treated as negatives.
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict

TYPES = ["type1", "type1a", "type1b", "other"]


def lev(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def sim(a, b):
    if not a and not b:
        return 1.0
    return 1.0 - lev(a, b) / max(len(a), len(b), 1)


def tolerant_equal(pred, gt):
    if len(pred) != len(gt):
        return False
    return all(p == g or g == "#" or p == "#" for p, g in zip(pred, gt))


def load(path):
    d = defaultdict(list)
    with open(path, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f, delimiter=";"):
            d[r["image"].strip()].append({k: (v or "").strip() for k, v in r.items()})
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--min-sim", type=float, default=0.4)
    a = ap.parse_args()
    gt, pr = load(a.gt), load(a.pred)
    images = set(gt) | set(pr)
    det_tp = Counter(); det_fn = Counter(); det_fp = Counter()
    exact = Counter(); tol = Counter(); n_char = Counter(); ok_char = Counter()
    type_conf = Counter()
    other_as_std = 0
    for im in sorted(images):
        g = [r for r in gt.get(im, []) if r.get("plate_num") or r.get("plate_type")]
        p = list(pr.get(im, []))
        used = set()
        for gr in g:
            best, bi = -1.0, None
            for i, prow in enumerate(p):
                if i in used:
                    continue
                s = sim(prow["plate_num"], gr["plate_num"])
                if s > best:
                    best, bi = s, i
            gtype = gr.get("plate_type", "")
            if bi is not None and best >= a.min_sim:
                used.add(bi)
                prow = p[bi]
                det_tp[gtype] += 1
                if prow["plate_num"] == gr["plate_num"]:
                    exact[gtype] += 1
                if tolerant_equal(prow["plate_num"], gr["plate_num"]):
                    tol[gtype] += 1
                n_char[gtype] += len(gr["plate_num"])
                ok_char[gtype] += sum(1 for x, y in zip(prow["plate_num"], gr["plate_num"]) if x == y)
                type_conf[(gtype, prow["plate_type"])] += 1
                if gtype == "other" and prow["plate_type"] != "other":
                    other_as_std += 1
            else:
                det_fn[gtype] += 1
                if gtype == "other":
                    pass   # not emitting 'other' is allowed
        for i, prow in enumerate(p):
            if i not in used:
                det_fp[prow["plate_type"]] += 1
    print(f"{'type':8s} {'GT':>5s} {'det':>5s} {'recall':>7s} {'exact':>7s} {'tol#':>7s} {'char':>7s} {'FP':>5s}")
    for t in TYPES:
        n = det_tp[t] + det_fn[t]
        if n == 0 and det_fp[t] == 0:
            continue
        print(f"{t:8s} {n:5d} {det_tp[t]:5d} {det_tp[t] / max(1, n):7.3f} {exact[t] / max(1, det_tp[t]):7.3f} "
              f"{tol[t] / max(1, det_tp[t]):7.3f} {ok_char[t] / max(1, n_char[t]):7.3f} {det_fp[t]:5d}")
    std = ["type1", "type1a", "type1b"]
    n_std = sum(det_tp[t] + det_fn[t] for t in std)
    tp_std = sum(det_tp[t] for t in std)
    print(f"\nstandard types: recall {tp_std / max(1, n_std):.3f}, exact-match {sum(exact[t] for t in std) / max(1, tp_std):.3f} of detected,"
          f" end-to-end exact {sum(exact[t] for t in std) / max(1, n_std):.3f}")
    print(f"type accuracy on detected standard plates: {sum(type_conf[(t, t)] for t in std) / max(1, tp_std):.3f}")
    print(f"'other' plates reported with a standard type: {other_as_std}")
    print(f"false positives (unmatched predictions): {sum(det_fp.values())}")
    print("\ntype confusion (gt -> pred):")
    for g in TYPES:
        row = " ".join(f"{type_conf[(g, p)]:5d}" for p in TYPES)
        print(f"  {g:8s} {row}")


if __name__ == "__main__":
    main()
