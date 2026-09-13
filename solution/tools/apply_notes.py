#!/usr/bin/env python3
"""Convert compact reviewer notes into decisions.csv.

    python tools/apply_notes.py --review work/review/buses --notes notes.txt [--sheets sheets]

Notes format (one candidate per line, n = tile index from the sheet):
    12 A123BC77 type1            keep, plate text + type (is_vehicle from candidate unless v0/v1 given)
    13 X###KM777 type1a v1       keep with unreadable positions
    14 x                         reject candidate (not a plate / duplicate)
    15 drop                      reject the whole image (person is the main subject, unusable, ...)
    16 ok                        keep with the model's text and type
    17 ok type1b                 keep with the model's text but another type
Optional trailing tokens: night rain snow dirt glare motion_blur angle  -> conditions.
"""
import argparse
import csv
import os

TYPES = {"type1", "type1a", "type1b", "other"}
CONDS = {"day", "night", "rain", "snow", "dirt", "glare", "motion_blur", "angle"}
CYR2LAT = str.maketrans("АВЕКМНОРСТУХ", "ABEKMHOPCTYX")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--review", required=True)
    ap.add_argument("--notes", required=True)
    ap.add_argument("--sheets", default="sheets")
    a = ap.parse_args()
    idx = {}
    with open(os.path.join(a.review, a.sheets, "index.csv"), newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter=";"):
            idx[r["n"]] = r
    cands = {}
    with open(os.path.join(a.review, "candidates.csv"), newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter=";"):
            cands[r["id"]] = r
    dec_path = os.path.join(a.review, "decisions.csv")
    existing = {}
    if os.path.exists(dec_path):
        with open(dec_path, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f, delimiter=";"):
                existing[r["id"]] = r
    n_new = 0
    for line in open(a.notes, encoding="utf-8"):
        toks = line.strip().split()
        if not toks or toks[0].startswith("#"):
            continue
        n = toks[0]
        if n not in idx:
            print("unknown tile", n)
            continue
        cid = idx[n]["id"]
        c = cands[cid]
        row = dict(id=cid, plate_num=c["ocr_text"], plate_type=c["det_type"], is_vehicle=c["is_vehicle"], keep="1", conditions="")
        rest = toks[1:]
        if rest and rest[0] == "x":
            row["keep"] = "0"
        elif rest and rest[0] == "drop":
            row["keep"] = "drop"
        else:
            if rest and rest[0] != "ok":
                row["plate_num"] = rest[0].upper().translate(CYR2LAT)
            conds = []
            for t in rest[1:]:
                if t in TYPES:
                    row["plate_type"] = t
                elif t in ("v0", "v1"):
                    row["is_vehicle"] = t[1]
                elif t in CONDS:
                    conds.append(t)
            row["conditions"] = ",".join(conds)
        existing[cid] = row
        n_new += 1
    with open(dec_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "plate_num", "plate_type", "is_vehicle", "keep", "conditions"], delimiter=";")
        w.writeheader()
        for r in existing.values():
            w.writerow(r)
    print(f"{n_new} notes applied, {len(existing)} decisions total -> {dec_path}")


if __name__ == "__main__":
    main()
