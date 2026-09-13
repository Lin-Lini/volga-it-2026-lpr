#!/usr/bin/env python3
"""Harvest CC-licensed photos through the Openverse API (aggregates Flickr, Wikimedia, etc.).

  python harvest_openverse.py --out raw/openverse/taxi --query "moscow taxi" --pages 12
Anonymous access is rate-limited, so queries are used sparingly. Each file is recorded in
manifest.csv with license, creator, landing page and source, for attribution.
"""
import argparse
import csv
import hashlib
import os
import re
import sys
import time

import requests

API = "https://api.openverse.org/v1/images/"
UA = os.environ.get("HARVEST_UA",
    "VolgaIT-2026-LPR-dataset/1.0 (+https://github.com/; non-commercial research)")
# Wikimedia and Openverse ask for a contact in the User-Agent: set HARVEST_UA to your own
# e.g. HARVEST_UA="my-project/1.0 (you@example.com)" python tools/harvest_commons.py ...
S = requests.Session()
S.headers["User-Agent"] = UA


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--query", action="append", required=True)
    ap.add_argument("--pages", type=int, default=12, help="pages of 20 per query (max 12 anonymous)")
    ap.add_argument("--licenses", default="cc0,pdm,by,by-sa")
    ap.add_argument("--skip-wikimedia", action="store_true", help="skip Commons files (harvested separately)")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    man_path = os.path.join(a.out, "manifest.csv")
    existing = set()
    if os.path.exists(man_path):
        with open(man_path, newline="", encoding="utf-8") as f:
            existing = {r["id"] for r in csv.DictReader(f, delimiter=";")}
    fields = ["file", "id", "title", "creator", "license", "license_version", "license_url", "source", "provider",
              "landing", "url", "width", "height", "query", "tag"]
    new = not os.path.exists(man_path)
    with open(man_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, delimiter=";", extrasaction="ignore")
        if new:
            w.writeheader()
        for q in a.query:
            for page in range(1, a.pages + 1):
                for attempt in range(4):
                    r = S.get(API, params=dict(q=q, page=page, page_size=20, license=a.licenses, mature="false"), timeout=60)
                    if r.status_code == 200:
                        break
                    if r.status_code == 429:
                        print("rate limited, sleeping 60s", file=sys.stderr)
                        time.sleep(60)
                    else:
                        time.sleep(3)
                if r.status_code != 200:
                    print(f"query {q} page {page}: HTTP {r.status_code}", file=sys.stderr)
                    break
                d = r.json()
                results = d.get("results", [])
                if not results:
                    break
                for it in results:
                    if it["id"] in existing:
                        continue
                    if a.skip_wikimedia and it.get("source") == "wikimedia":
                        continue
                    url = it.get("url")
                    if not url or not re.search(r"\.(jpe?g|png)(\?|$)", url, re.I):
                        continue
                    fn = f"{it['source']}_{hashlib.md5(it['id'].encode()).hexdigest()[:10]}" + (".png" if url.lower().endswith(".png") else ".jpg")
                    path = os.path.join(a.out, fn)
                    try:
                        rr = S.get(url, timeout=120)
                        if rr.status_code != 200 or len(rr.content) < 2000:
                            continue
                        with open(path, "wb") as g:
                            g.write(rr.content)
                    except requests.RequestException:
                        continue
                    row = dict(file=fn, id=it["id"], title=(it.get("title") or "")[:120], creator=(it.get("creator") or "")[:100],
                               license=it.get("license"), license_version=it.get("license_version"), license_url=it.get("license_url"),
                               source=it.get("source"), provider=it.get("provider"), landing=it.get("foreign_landing_url"), url=url,
                               width=it.get("width"), height=it.get("height"), query=q, tag=a.tag)
                    w.writerow(row)
                    f.flush()
                    existing.add(it["id"])
                print(f"{q} page {page}: {len(results)} results", file=sys.stderr)
                time.sleep(1.5)
    print("done", file=sys.stderr)


if __name__ == "__main__":
    main()
