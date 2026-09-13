#!/usr/bin/env python3
"""Harvest freely-licensed images from Wikimedia Commons with full license metadata.

Usage examples:
  python harvest_commons.py --out raw/commons/buses --search 'intitle:bus intitle:2024 filemime:image/jpeg' --max 500
  python harvest_commons.py --out raw/commons/taxi --category "Taxis in Moscow" --depth 2
Every downloaded file gets a row in <out>/manifest.csv with title, page URL, license,
license URL, author, date and the download URL, so attribution can be reproduced later.
Only CC0 / Public domain / CC BY / CC BY-SA files are kept (configurable).
"""
import argparse
import csv
import hashlib
import json
import os
import re
import sys
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor

import requests

API = "https://commons.wikimedia.org/w/api.php"
UA = os.environ.get("HARVEST_UA",
    "VolgaIT-2026-LPR-dataset/1.0 (+https://github.com/; non-commercial research)")
# Wikimedia and Openverse ask for a contact in the User-Agent: set HARVEST_UA to your own
# e.g. HARVEST_UA="my-project/1.0 (you@example.com)" python tools/harvest_commons.py ...
S = requests.Session()
S.headers["User-Agent"] = UA

ALLOWED_LICENSE_RE = re.compile(r"^(CC0|Public domain|CC BY(-SA)?( [0-9.]+)?( [a-z]{2})?|CC-BY(-SA)?[- 0-9.]*)", re.I)
MIMES = {"image/jpeg", "image/png", "image/tiff"}


def api(params, retries=5):
    params = dict(params, format="json")
    for i in range(retries):
        try:
            r = S.post(API, data=params, timeout=90)
            time.sleep(0.7)
            if r.status_code == 200:
                return r.json()
            print(f"API HTTP {r.status_code}", file=sys.stderr)
            if r.status_code == 429:
                time.sleep(30 * (i + 1))
                continue
        except requests.RequestException as e:
            print(f"API error: {e}", file=sys.stderr)
        time.sleep(2 + i * 2)
    raise RuntimeError(f"API failure: {str(params)[:300]}")


def iter_category(cat, depth, seen=None):
    seen = seen if seen is not None else set()
    if cat in seen:
        return
    seen.add(cat)
    cont = {}
    while True:
        d = api({"action": "query", "list": "categorymembers", "cmtitle": f"Category:{cat}",
                 "cmtype": "file|subcat", "cmlimit": 500, **cont})
        for m in d["query"]["categorymembers"]:
            if m["ns"] == 6:
                yield m["title"]
            elif m["ns"] == 14 and depth > 0:
                yield from iter_category(m["title"].replace("Category:", ""), depth - 1, seen)
        if "continue" in d:
            cont = d["continue"]
        else:
            break


def iter_search(query, limit):
    cont = {}
    n = 0
    while n < limit:
        d = api({"action": "query", "list": "search", "srsearch": query, "srnamespace": 6,
                 "srlimit": min(50, limit - n), **cont})
        for m in d["query"]["search"]:
            yield m["title"]
            n += 1
        if "continue" in d and n < limit:
            cont = d["continue"]
        else:
            break


def file_info(titles, width):
    out = {}
    for i in range(0, len(titles), 25):
        chunk = titles[i:i + 25]
        d = api({"action": "query", "titles": "|".join(chunk), "prop": "imageinfo",
                 "iiprop": "url|size|mime|timestamp|extmetadata|sha1", "iiurlwidth": width,
                 "iiextmetadatafilter": "LicenseShortName|License|LicenseUrl|Artist|Credit|DateTimeOriginal|Attribution|Copyrighted|UsageTerms"})
        for p in d["query"]["pages"].values():
            if "imageinfo" not in p:
                continue
            ii = p["imageinfo"][0]
            em_raw = ii.get("extmetadata") or {}
            em = {k: v.get("value", "") for k, v in em_raw.items()} if isinstance(em_raw, dict) else {}
            out[p["title"]] = dict(title=p["title"], page=f"https://commons.wikimedia.org/wiki/{urllib.parse.quote(p['title'].replace(' ', '_'))}",
                                  url=ii.get("thumburl") or ii.get("url"), orig_url=ii.get("url"), mime=ii.get("mime"),
                                  width=ii.get("width"), height=ii.get("height"), sha1=ii.get("sha1"), upload=ii.get("timestamp"),
                                  license=strip_html(em.get("LicenseShortName", "")), license_url=em.get("LicenseUrl", ""),
                                  artist=strip_html(em.get("Artist", ""))[:200], credit=strip_html(em.get("Credit", ""))[:200],
                                  date=strip_html(em.get("DateTimeOriginal", ""))[:40], copyrighted=em.get("Copyrighted", ""),
                                  usage_terms=strip_html(em.get("UsageTerms", ""))[:100])
    return out


def strip_html(s):
    return re.sub(r"<[^>]+>", "", s or "").replace("\n", " ").strip()


def license_ok(info):
    lic = info["license"]
    if not lic:
        return False
    if re.search(r"NC|ND|GFDL|FAL|non-free|fair use", lic, re.I) and not ALLOWED_LICENSE_RE.match(lic):
        return False
    return bool(ALLOWED_LICENSE_RE.match(lic))


def year_of(info):
    m = re.search(r"(19|20)\d\d", info.get("date") or "") or re.search(r"(19|20)\d\d", info.get("upload") or "")
    return int(m.group(0)) if m else None


def safe_name(title):
    base = title.replace("File:", "")
    h = hashlib.md5(base.encode()).hexdigest()[:8]
    base = re.sub(r"[^A-Za-z0-9._-]+", "_", base)[:80]
    root, ext = os.path.splitext(base)
    ext = ext.lower() if ext.lower() in (".jpg", ".jpeg", ".png", ".tif", ".tiff") else ".jpg"
    return f"{root}_{h}{ext}"


def download(info, out_dir):
    fn = safe_name(info["title"])
    path = os.path.join(out_dir, fn)
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return fn
    for i in range(4):
        try:
            r = S.get(info["url"], timeout=120)
            if r.status_code == 200 and len(r.content) > 1000:
                with open(path, "wb") as f:
                    f.write(r.content)
                return fn
            if r.status_code == 429:
                time.sleep(20 * (i + 1))
        except requests.RequestException:
            time.sleep(3)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--category", action="append", default=[])
    ap.add_argument("--search", action="append", default=[])
    ap.add_argument("--depth", type=int, default=1)
    ap.add_argument("--max", type=int, default=400, help="max files per query/category")
    ap.add_argument("--min-year", type=int, default=0)
    ap.add_argument("--width", type=int, default=1600)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--tag", default="", help="free-form tag stored in manifest (e.g. bus_yellow)")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    titles = []
    for c in a.category:
        got = list(iter_category(c, a.depth))[: a.max]
        print(f"[category] {c}: {len(got)} files", file=sys.stderr)
        titles += [(t, f"category:{c}") for t in got]
    for s in a.search:
        got = list(iter_search(s, a.max))
        print(f"[search] {s}: {len(got)} files", file=sys.stderr)
        titles += [(t, f"search:{s}") for t in got]
    src_by_title = {}
    for t, s in titles:
        src_by_title.setdefault(t, s)
    uniq = list(src_by_title)
    print(f"unique titles: {len(uniq)}", file=sys.stderr)

    infos = file_info(uniq, a.width)
    keep = []
    for t, inf in infos.items():
        if inf["mime"] not in MIMES:
            continue
        if not license_ok(inf):
            continue
        y = year_of(inf)
        if a.min_year and (y is None or y < a.min_year):
            continue
        inf["query"] = src_by_title.get(t, "")
        inf["year"] = y
        keep.append(inf)
    print(f"after license/mime/year filter: {len(keep)}", file=sys.stderr)

    man_path = os.path.join(a.out, "manifest.csv")
    existing = set()
    if os.path.exists(man_path):
        with open(man_path, newline="", encoding="utf-8") as f:
            existing = {r["title"] for r in csv.DictReader(f, delimiter=";")}
    fields = ["file", "title", "page", "license", "license_url", "artist", "credit", "date", "year", "upload", "width", "height",
              "mime", "sha1", "orig_url", "url", "query", "tag"]
    new_file = not os.path.exists(man_path)
    with open(man_path, "a", newline="", encoding="utf-8") as f, ThreadPoolExecutor(a.workers) as ex:
        w = csv.DictWriter(f, fieldnames=fields, delimiter=";", extrasaction="ignore")
        if new_file:
            w.writeheader()
        todo = [k for k in keep if k["title"] not in existing]
        for inf, fn in zip(todo, ex.map(lambda k: download(k, a.out), todo)):
            if fn:
                inf["file"] = fn
                inf["tag"] = a.tag
                w.writerow(inf)
                f.flush()
    print("done", file=sys.stderr)


if __name__ == "__main__":
    main()
