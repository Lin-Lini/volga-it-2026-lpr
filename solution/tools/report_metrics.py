#!/usr/bin/env python3
"""Full quality report on the held-out real test split: overall metrics, a breakdown by plate width and by
shooting conditions, and the type confusion matrix — rendered as Markdown for the explanatory note.

    python tools/report_metrics.py --gt work/test_set/gt.csv --pred work/test_set/result.csv \
        --meta ../dataset/meta.csv --out work/metrics.md
"""
from __future__ import annotations

import argparse
import csv
import os
from collections import Counter, defaultdict

TYPES = ["type1", "type1a", "type1b", "other"]
STD = ["type1", "type1a", "type1b"]


def lev(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def sim(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    return 1.0 - lev(a, b) / max(len(a), len(b), 1)


def tol_eq(pred: str, gt: str) -> bool:
    """Equality that tolerates '#' on either side (an unreadable position is not an error)."""
    return len(pred) == len(gt) and all(p == g or "#" in (p, g) for p, g in zip(pred, gt))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--meta", default=None, help="dataset meta.csv, for bbox width and conditions")
    ap.add_argument("--out", default=None)
    ap.add_argument("--min-sim", type=float, default=0.4)
    a = ap.parse_args()

    gt = defaultdict(list)
    for r in csv.DictReader(open(a.gt, newline="", encoding="utf-8-sig"), delimiter=";"):
        gt[r["image"].strip()].append({k: (v or "").strip() for k, v in r.items()})
    pr = defaultdict(list)
    for r in csv.DictReader(open(a.pred, newline="", encoding="utf-8-sig"), delimiter=";"):
        pr[r["image"].strip()].append({k: (v or "").strip() for k, v in r.items()})

    # extra attributes of ground-truth plates (width in px, shooting conditions)
    extra = {}
    if a.meta and os.path.exists(a.meta):
        for r in csv.DictReader(open(a.meta, newline="", encoding="utf-8"), delimiter=";"):
            if r.get("is_synthetic") == "1":
                continue
            key = (os.path.basename(r["image"]), r["plate_num"], r["plate_type"])
            extra[key] = (float(r["bbox"].split(",")[2]), r.get("conditions", ""))

    det_tp, det_fn, det_fp = Counter(), Counter(), Counter()
    exact, tol = Counter(), Counter()
    n_char, ok_char = Counter(), Counter()
    conf_mat = Counter()
    by_w = defaultdict(lambda: [0, 0, 0])       # bucket -> [detected, exact, total]
    by_cond = defaultdict(lambda: [0, 0, 0])
    other_as_std = 0

    for im in sorted(set(gt) | set(pr)):
        g_rows = [r for r in gt.get(im, []) if r.get("plate_num") or r.get("plate_type")]
        p_rows = list(pr.get(im, []))
        used = set()
        for g in g_rows:
            best, bi = -1.0, None
            for i, p in enumerate(p_rows):
                if i in used:
                    continue
                s = sim(p["plate_num"], g["plate_num"])
                if s > best:
                    best, bi = s, i
            gtype = g.get("plate_type", "")
            w, conds = extra.get((im, g["plate_num"], gtype), (None, ""))
            bucket = None if w is None else ("<32 px" if w < 32 else "32–64 px" if w < 64 else "64–128 px" if w < 128 else "≥128 px")
            matched = bi is not None and best >= a.min_sim
            is_exact = False
            if matched:
                used.add(bi)
                p = p_rows[bi]
                det_tp[gtype] += 1
                is_exact = p["plate_num"] == g["plate_num"]
                exact[gtype] += int(is_exact)
                tol[gtype] += int(tol_eq(p["plate_num"], g["plate_num"]))
                n_char[gtype] += len(g["plate_num"])
                ok_char[gtype] += sum(1 for x, y in zip(p["plate_num"], g["plate_num"]) if x == y)
                conf_mat[(gtype, p["plate_type"])] += 1
                if gtype == "other" and p["plate_type"] != "other":
                    other_as_std += 1
            else:
                det_fn[gtype] += 1
            if gtype in STD:
                if bucket:
                    by_w[bucket][0] += int(matched); by_w[bucket][1] += int(is_exact); by_w[bucket][2] += 1
                for c in (conds.split(",") if conds else ["day"]):
                    if c:
                        by_cond[c][0] += int(matched); by_cond[c][1] += int(is_exact); by_cond[c][2] += 1
        for i, p in enumerate(p_rows):
            if i not in used:
                det_fp[p["plate_type"]] += 1

    L = []
    n_std = sum(det_tp[t] + det_fn[t] for t in STD)
    tp_std = sum(det_tp[t] for t in STD)
    ex_std = sum(exact[t] for t in STD)
    tol_std = sum(tol[t] for t in STD)
    ch_std = sum(ok_char[t] for t in STD) / max(1, sum(n_char[t] for t in STD))
    L.append("**Итог по целевым типам (1, 1А, 1Б):** полнота обнаружения %.3f, точное совпадение номера %.3f "
             "от найденных, сквозная точность (найден И прочитан полностью верно) **%.3f**, посимвольная точность %.3f, "
             "точность определения типа %.3f." % (tp_std / max(1, n_std), ex_std / max(1, tp_std), ex_std / max(1, n_std),
                                                  ch_std, sum(conf_mat[(t, t)] for t in STD) / max(1, tp_std)))
    L.append("")
    L.append("| Тип | Знаков в эталоне | Найдено | Полнота | Точный номер | С учётом `#` | Посимвольно | Ложных |")
    L.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for t in TYPES:
        n = det_tp[t] + det_fn[t]
        if n == 0 and det_fp[t] == 0:
            continue
        L.append("| %s | %d | %d | %.3f | %.3f | %.3f | %.3f | %d |" % (
            t, n, det_tp[t], det_tp[t] / max(1, n), exact[t] / max(1, det_tp[t]), tol[t] / max(1, det_tp[t]),
            ok_char[t] / max(1, n_char[t]), det_fp[t]))
    L.append("| **целевые** | **%d** | **%d** | **%.3f** | **%.3f** | **%.3f** | **%.3f** | **%d** |" % (
        n_std, tp_std, tp_std / max(1, n_std), ex_std / max(1, tp_std), tol_std / max(1, tp_std), ch_std,
        sum(det_fp[t] for t in STD)))
    L.append("")
    L.append("Знаков типа `other` (прицепы, мотоциклы, иностранные), ошибочно выданных с целевым типом: "
             "**%d** из %d." % (other_as_std, det_tp["other"] + det_fn["other"]))
    L.append("")
    if by_w:
        L.append("**Зависимость от размера знака в кадре** (целевые типы):")
        L.append("")
        L.append("| Ширина знака | Знаков | Полнота | Сквозная точность |")
        L.append("|---|---:|---:|---:|")
        for k in ["<32 px", "32–64 px", "64–128 px", "≥128 px"]:
            if k in by_w:
                d, e, n = by_w[k]
                L.append("| %s | %d | %.3f | %.3f |" % (k, n, d / max(1, n), e / max(1, n)))
        L.append("")
    if by_cond:
        L.append("**Зависимость от условий съёмки** (целевые типы; у знака может быть несколько меток):")
        L.append("")
        L.append("| Условие | Знаков | Полнота | Сквозная точность |")
        L.append("|---|---:|---:|---:|")
        names = {"day": "день", "night": "ночь", "rain": "дождь", "snow": "снег", "dirt": "загрязнение",
                 "glare": "блики", "motion_blur": "смаз", "angle": "ракурс"}
        for k, v in sorted(by_cond.items(), key=lambda kv: -kv[1][2]):
            d, e, n = v
            L.append("| %s | %d | %.3f | %.3f |" % (names.get(k, k), n, d / max(1, n), e / max(1, n)))
        L.append("")
    L.append("**Матрица ошибок по типам** (строка — эталон, столбец — решение):")
    L.append("")
    L.append("| эталон \\ решение | " + " | ".join(TYPES) + " | не найден |")
    L.append("|---|" + "---:|" * (len(TYPES) + 1))
    for g in TYPES:
        L.append("| %s | %s | %d |" % (g, " | ".join(str(conf_mat[(g, p)]) for p in TYPES), det_fn[g]))
    text = "\n".join(L)
    print(text)
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
        open(a.out, "w", encoding="utf-8").write(text + "\n")


if __name__ == "__main__":
    main()
