"""Procedural renderer of vehicle registration plates (GOST R 50577-2018 and a few foreign layouts).

All geometry is in millimetres and converted to pixels with a scale factor, so plates of any
resolution can be produced. Rendering is deterministic for a given `random.Random` instance.

Plate families (``kind``):
  type1   – 520x112, white, "A123BC 77"                      (GOST type 1)
  type1a  – 290x170, white, two rows "A123 / BC 77"          (GOST type 1A)
  type1b  – 520x112, yellow, "AB123 77" (GOST) or "A123BC 77" (task statement)  (GOST type 1B)
  other   – trailer (type 2), motorcycle (type 4), tractor (type 3), police (type 20),
            military (type 5), diplomatic (type 9-11), transit (type 15), foreign (KZ, BY, UA, UZ, KG, AM, GE, EU)
"""
from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
FONT_DIR = os.path.join(HERE, "fonts")

LETTERS = "ABEKMHOPCTYX"                     # Latin letters allowed on Russian plates
LAT2CYR = dict(zip("ABEKMHOPCTYX", "АВЕКМНОРСТУХ"))
DIGITS = "0123456789"
REGIONS_2 = [f"{i:02d}" for i in range(1, 100)]
REGIONS_3 = ["102", "113", "116", "121", "122", "123", "124", "125", "126", "134", "136", "138", "142", "147", "150", "152",
             "154", "155", "156", "159", "161", "163", "164", "169", "173", "174", "177", "178", "186", "190", "193", "196",
             "197", "199", "702", "716", "750", "761", "763", "774", "777", "790", "797", "799"]

WHITE = (250, 250, 250)
BLACK = (18, 18, 18)
YELLOW_BASE = (250, 200, 20)


@dataclass
class Glyph:
    ch: str
    x: float          # left, mm
    y: float          # top, mm
    h: float          # cap height, mm
    row: int = 0      # 0 = single row / top row, 1 = bottom row


@dataclass
class PlateSpec:
    kind: str                     # type1 / type1a / type1b / other
    subkind: str                  # e.g. "type1", "trailer", "kz"
    text: str                     # plate number as it should be read (Latin, no spaces)
    width: float                  # mm
    height: float                 # mm
    bg: Tuple[int, int, int]
    fg: Tuple[int, int, int]
    glyphs: List[Glyph] = field(default_factory=list)
    lines: List[Tuple[float, float, float, float]] = field(default_factory=list)   # dividers, mm
    region_box: Optional[Tuple[float, float, float, float]] = None                  # x0,y0,x1,y1 mm
    rus: Optional[Tuple[float, float, float]] = None                                # x, y, h of "RUS" text
    flag: Optional[Tuple[float, float, float, float]] = None                        # x0,y0,x1,y1 of flag
    strip: Optional[Tuple[Tuple[int, int, int], float, str]] = None                 # (color, width mm, text) left strip
    rows: int = 1
    row_split: float = 0.0        # for two-row plates: y (mm) separating rows
    small_text: List[Tuple[str, float, float, float]] = field(default_factory=list)  # (text, x, y, h)


class Fonts:
    """Loads the GOST plate font plus fallback OFL fonts (for robustness to unusual fonts)."""

    def __init__(self, font_dir: str = FONT_DIR):
        self.paths = []
        for name in ["GOST-R-50577-93.ttf", "RobotoCondensed-wght.ttf", "Oswald-wght.ttf", "PTSansNarrow-Bold.ttf"]:
            p = os.path.join(font_dir, name)
            if os.path.exists(p):
                self.paths.append(p)
        if not self.paths:
            raise FileNotFoundError(f"no fonts in {font_dir}")
        self.gost = self.paths[0]
        self._cache: Dict[Tuple[str, int], ImageFont.FreeTypeFont] = {}

    def get(self, path: str, size: int) -> ImageFont.FreeTypeFont:
        key = (path, size)
        if key not in self._cache:
            f = ImageFont.truetype(path, size)
            try:
                if "wght" in os.path.basename(path):
                    f.set_variation_by_axes([700])
            except Exception:
                pass
            self._cache[key] = f
        return self._cache[key]

    def pick(self, rng: random.Random, gost_prob: float = 0.8) -> str:
        if rng.random() < gost_prob or len(self.paths) == 1:
            return self.gost
        return rng.choice(self.paths[1:])


# ----------------------------------------------------------------------------------------------
# Number generation
# ----------------------------------------------------------------------------------------------
def rand_region(rng: random.Random) -> str:
    return rng.choice(REGIONS_3) if rng.random() < 0.35 else rng.choice(REGIONS_2)


def rand_letters(rng: random.Random, n: int) -> str:
    return "".join(rng.choice(LETTERS) for _ in range(n))


def rand_digits(rng: random.Random, n: int, allow_all_zero: bool = False) -> str:
    while True:
        s = "".join(rng.choice(DIGITS) for _ in range(n))
        if allow_all_zero or set(s) != {"0"}:
            return s


def rand_standard_number(rng: random.Random) -> Tuple[str, str, str, str]:
    """(letter, digits, letters, region) for types 1 / 1A / 1B(task)."""
    return rng.choice(LETTERS), rand_digits(rng, 3), rand_letters(rng, 2), rand_region(rng)


# ----------------------------------------------------------------------------------------------
# Layout builders (all in mm)
# ----------------------------------------------------------------------------------------------
def _row_layout(chars: List[Tuple[str, float]], x0: float, gap: float, y_bottom: float, row: int = 0) -> Tuple[List[Glyph], float]:
    """Place characters left→right, bottom-aligned at y_bottom. chars = [(ch, height_mm)]. Returns glyphs and x end."""
    x = x0
    out = []
    for ch, h in chars:
        w = char_width(ch, h)
        out.append(Glyph(ch, x, y_bottom - h, h, row))
        x += w + gap
    return out, x - gap


def char_width(ch: str, h: float) -> float:
    """Approximate advance width for the GOST font as a fraction of cap height."""
    if ch == "1":
        return h * 0.42
    if ch in "MХXМ":
        return h * 0.68
    if ch in "ABEKHOPCTYXАВЕКМНОРСТУХ" or ch.isalpha():
        return h * 0.62
    return h * 0.58


def build_type1(rng: random.Random, yellow: bool = False, gost_taxi: bool = False) -> PlateSpec:
    W, H = 520.0, 112.0
    bg = WHITE if not yellow else YELLOW_BASE
    if gost_taxi:
        l2, d3, reg = rand_letters(rng, 2), rand_digits(rng, 3), rand_region(rng)
        text = l2 + d3 + reg
        chars = [(c, 58.0) for c in l2] + [(c, 76.0) for c in d3]
        g, xend = _row_layout(chars, 44.0, 8.0, 94.0)
        # widen gap between letter group and digit group a bit
        for gl in g[2:]:
            gl.x += 14.0
    else:
        l1, d3, l2, reg = rand_standard_number(rng)
        text = l1 + d3 + l2 + reg
        chars = [(l1, 58.0)] + [(c, 76.0) for c in d3] + [(c, 58.0) for c in l2]
        g, xend = _row_layout(chars, 36.0, 8.0, 94.0)
        for gl in g[1:]:
            gl.x += 8.0
        for gl in g[4:]:
            gl.x += 8.0
    spec = PlateSpec("type1b" if yellow else "type1", "type1b_gost" if gost_taxi else ("type1b" if yellow else "type1"),
                     text, W, H, bg, BLACK, g)
    spec.lines = [(384.0, 4.0, 384.0, H - 4.0)]
    spec.region_box = (384.0, 0.0, W, H)
    rh = 58.0 if len(reg) == 2 else 52.0
    rg, _ = _row_layout([(c, rh) for c in reg], 0, 4.0, 70.0)
    rw = sum(char_width(c, rh) for c in reg) + 4.0 * (len(reg) - 1)
    off = 384.0 + (136.0 - rw) / 2.0
    for gl in rg:
        gl.x += off
    spec.glyphs += rg
    spec.rus = (396.0, 78.0, 16.0)
    spec.flag = (456.0, 78.0, 500.0, 98.0)
    return spec


def build_type1a(rng: random.Random) -> PlateSpec:
    W, H = 290.0, 170.0
    l1, d3, l2, reg = rand_standard_number(rng)
    text = l1 + d3 + l2 + reg
    top, _ = _row_layout([(l1, 46.0)] + [(c, 58.0) for c in d3], 44.0, 7.0, 82.0, row=0)
    for gl in top[1:]:
        gl.x += 8.0
    bot, _ = _row_layout([(c, 58.0) for c in l2], 44.0, 7.0, 158.0, row=1)
    spec = PlateSpec("type1a", "type1a", text, W, H, WHITE, BLACK, top + bot, rows=2, row_split=88.0)
    spec.region_box = (168.0, 90.0, W, H)
    spec.lines = [(168.0, 90.0, 168.0, H), (168.0, 90.0, W, 90.0)]
    rh = 40.0 if len(reg) == 2 else 34.0
    rg, _ = _row_layout([(c, rh) for c in reg], 0, 3.0, 132.0, row=1)
    rw = sum(char_width(c, rh) for c in reg) + 3.0 * (len(reg) - 1)
    off = 168.0 + (122.0 - rw) / 2.0
    for gl in rg:
        gl.x += off
    spec.glyphs += rg
    spec.rus = (180.0, 138.0, 14.0)
    spec.flag = (232.0, 138.0, 274.0, 156.0)
    return spec


def build_trailer(rng: random.Random) -> PlateSpec:
    W, H = 520.0, 112.0
    l2, d4, reg = rand_letters(rng, 2), rand_digits(rng, 4), rand_region(rng)
    chars = [(c, 58.0) for c in l2] + [(c, 76.0) for c in d4]
    g, _ = _row_layout(chars, 40.0, 8.0, 94.0)
    for gl in g[2:]:
        gl.x += 14.0
    spec = PlateSpec("other", "trailer", l2 + d4 + reg, W, H, WHITE, BLACK, g)
    spec.lines = [(384.0, 4.0, 384.0, H - 4.0)]
    spec.region_box = (384.0, 0.0, W, H)
    _add_region_type1(spec, reg)
    return spec


def _add_region_type1(spec: PlateSpec, reg: str, rus=True):
    rh = 58.0 if len(reg) == 2 else 52.0
    rg, _ = _row_layout([(c, rh) for c in reg], 0, 4.0, 70.0)
    rw = sum(char_width(c, rh) for c in reg) + 4.0 * (len(reg) - 1)
    off = 384.0 + (136.0 - rw) / 2.0
    for gl in rg:
        gl.x += off
    spec.glyphs += rg
    if rus:
        spec.rus = (396.0, 78.0, 16.0)
        spec.flag = (456.0, 78.0, 500.0, 98.0)


def build_moto(rng: random.Random) -> PlateSpec:
    W, H = 190.0, 145.0
    d4, l2, reg = rand_digits(rng, 4), rand_letters(rng, 2), rand_region(rng)
    top, _ = _row_layout([(c, 58.0) for c in d4], 24.0, 6.0, 72.0, row=0)
    bot, _ = _row_layout([(c, 46.0) for c in l2], 22.0, 6.0, 132.0, row=1)
    rg, _ = _row_layout([(c, 46.0) for c in reg], 110.0, 4.0, 132.0, row=1)
    spec = PlateSpec("other", "moto", d4 + l2 + reg, W, H, WHITE, BLACK, top + bot + rg, rows=2, row_split=80.0)
    spec.lines = [(100.0, 88.0, 100.0, H - 4.0)]
    spec.small_text = [("RUS", 74.0, 78.0, 10.0)]
    return spec


def build_tractor(rng: random.Random) -> PlateSpec:
    W, H = 288.0, 206.0
    d4, l2, reg = rand_digits(rng, 4), rand_letters(rng, 2), rand_region(rng)
    top, _ = _row_layout([(c, 76.0) for c in d4], 40.0, 9.0, 98.0, row=0)
    bot, _ = _row_layout([(c, 58.0) for c in l2], 40.0, 9.0, 186.0, row=1)
    rg, _ = _row_layout([(c, 58.0) for c in reg], 170.0, 5.0, 186.0, row=1)
    spec = PlateSpec("other", "tractor", d4 + l2 + reg, W, H, WHITE, BLACK, top + bot + rg, rows=2, row_split=112.0)
    spec.lines = [(160.0, 118.0, 160.0, H - 4.0)]
    spec.small_text = [("RUS", 110.0, 110.0, 12.0)]
    return spec


def build_police(rng: random.Random) -> PlateSpec:
    W, H = 520.0, 112.0
    l1, d4, reg = rng.choice(LETTERS), rand_digits(rng, 4), rng.choice(REGIONS_2)
    chars = [(l1, 58.0)] + [(c, 76.0) for c in d4]
    g, _ = _row_layout(chars, 44.0, 10.0, 94.0)
    for gl in g[1:]:
        gl.x += 14.0
    spec = PlateSpec("other", "police", l1 + d4 + reg, W, H, (20, 60, 170), WHITE, g)
    spec.lines = [(384.0, 4.0, 384.0, H - 4.0)]
    spec.region_box = (384.0, 0.0, W, H)
    _add_region_type1(spec, reg, rus=False)
    spec.small_text = [("RUS", 410.0, 80.0, 16.0)]
    return spec


def build_military(rng: random.Random) -> PlateSpec:
    W, H = 520.0, 112.0
    d4, l2, code = rand_digits(rng, 4), rand_letters(rng, 2), rand_digits(rng, 2, allow_all_zero=True)
    chars = [(c, 76.0) for c in d4] + [(c, 58.0) for c in l2]
    g, _ = _row_layout(chars, 36.0, 9.0, 94.0)
    for gl in g[4:]:
        gl.x += 14.0
    spec = PlateSpec("other", "military", d4 + l2 + code, W, H, (25, 25, 25), WHITE, g)
    spec.lines = [(384.0, 4.0, 384.0, H - 4.0)]
    spec.region_box = (384.0, 0.0, W, H)
    _add_region_type1(spec, code, rus=False)
    spec.small_text = [("RUS", 410.0, 80.0, 16.0)]
    return spec


def build_diplomatic(rng: random.Random) -> PlateSpec:
    W, H = 520.0, 112.0
    variant = rng.choice(["CD", "D", "T"])
    d3 = rand_digits(rng, 3)
    tail = rand_digits(rng, rng.choice([1, 2, 3]))
    reg = rng.choice(REGIONS_2)
    chars = [(c, 76.0) for c in d3] + [(c, 58.0) for c in variant] + [(c, 76.0) for c in tail]
    g, _ = _row_layout(chars, 30.0, 8.0, 94.0)
    for gl in g[3:]:
        gl.x += 10.0
    for gl in g[3 + len(variant):]:
        gl.x += 10.0
    spec = PlateSpec("other", "diplomatic", d3 + variant + tail + reg, W, H, (200, 30, 30), WHITE, g)
    spec.lines = [(384.0, 4.0, 384.0, H - 4.0)]
    spec.region_box = (384.0, 0.0, W, H)
    _add_region_type1(spec, reg, rus=False)
    spec.small_text = [("RUS", 410.0, 80.0, 16.0)]
    return spec


def build_transit(rng: random.Random) -> PlateSpec:
    W, H = 520.0, 112.0
    l2, d3, l1, reg = rand_letters(rng, 2), rand_digits(rng, 3), rng.choice(LETTERS), rand_region(rng)
    chars = [(c, 58.0) for c in l2] + [(c, 76.0) for c in d3] + [(l1, 58.0)]
    g, _ = _row_layout(chars, 30.0, 8.0, 94.0)
    for gl in g[2:]:
        gl.x += 10.0
    for gl in g[5:]:
        gl.x += 10.0
    spec = PlateSpec("other", "transit", l2 + d3 + l1 + reg, W, H, WHITE, BLACK, g)
    spec.lines = [(384.0, 4.0, 384.0, H - 4.0)]
    spec.region_box = (384.0, 0.0, W, H)
    spec.strip = (YELLOW_BASE, 0.0, "")   # region field painted yellow (handled by renderer via region_fill)
    _add_region_type1(spec, reg)
    return spec


FOREIGN_LETTERS = "ABCDEFGHJKLMNOPRSTUVWXYZ"


def build_foreign(rng: random.Random) -> PlateSpec:
    country = rng.choice(["kz", "by", "ua", "uz", "kg", "am", "ge", "eu", "eu", "kz", "by"])
    W, H = 520.0, 112.0
    if country == "kz":
        d3, l3, reg = rand_digits(rng, 3), "".join(rng.choice(FOREIGN_LETTERS) for _ in range(3)), f"{rng.randint(1, 20):02d}"
        chars = [(c, 76.0) for c in d3] + [(c, 62.0) for c in l3]
        g, _ = _row_layout(chars, 70.0, 8.0, 94.0)
        for gl in g[3:]:
            gl.x += 12.0
        spec = PlateSpec("other", "kz", d3 + l3 + reg, W, H, WHITE, BLACK, g)
        spec.strip = ((30, 90, 200), 44.0, "KZ")
        spec.lines = [(410.0, 4.0, 410.0, H - 4.0)]
        rg, _ = _row_layout([(c, 62.0) for c in reg], 428.0, 4.0, 88.0)
        spec.glyphs += rg
    elif country == "by":
        d4, l2, reg = rand_digits(rng, 4), "".join(rng.choice("ABEIKMHOPCTX") for _ in range(2)), str(rng.randint(1, 7))
        chars = [(c, 76.0) for c in d4] + [(c, 62.0) for c in l2] + [("-", 62.0), (reg, 62.0)]
        g, _ = _row_layout(chars, 60.0, 9.0, 94.0)
        for gl in g[4:]:
            gl.x += 12.0
        spec = PlateSpec("other", "by", d4 + l2 + reg, W, H, WHITE, BLACK, g)
        spec.strip = ((220, 40, 40), 40.0, "BY")
    elif country == "ua":
        l2a, d4, l2b = "".join(rng.choice("ABCEHIKMOPTX") for _ in range(2)), rand_digits(rng, 4), "".join(rng.choice("ABCEHIKMOPTX") for _ in range(2))
        chars = [(c, 62.0) for c in l2a] + [(c, 76.0) for c in d4] + [(c, 62.0) for c in l2b]
        g, _ = _row_layout(chars, 64.0, 8.0, 94.0)
        for gl in g[2:]:
            gl.x += 12.0
        for gl in g[6:]:
            gl.x += 12.0
        spec = PlateSpec("other", "ua", l2a + d4 + l2b, W, H, WHITE, BLACK, g)
        spec.strip = ((40, 90, 200), 44.0, "UA")
    elif country == "uz":
        reg, l1, d3, l2 = f"{rng.randint(1, 95):02d}", rng.choice(FOREIGN_LETTERS), rand_digits(rng, 3), "".join(rng.choice(FOREIGN_LETTERS) for _ in range(2))
        chars = [(c, 62.0) for c in reg] + [("|", 70.0)] + [(l1, 62.0)] + [(c, 76.0) for c in d3] + [(c, 62.0) for c in l2]
        g, _ = _row_layout(chars, 40.0, 8.0, 94.0)
        spec = PlateSpec("other", "uz", reg + l1 + d3 + l2, W, H, WHITE, BLACK, g)
        spec.strip = ((30, 150, 90), 36.0, "UZ")
    elif country == "kg":
        reg, d3, l3 = f"{rng.randint(1, 9):02d}", rand_digits(rng, 3), "".join(rng.choice(FOREIGN_LETTERS) for _ in range(3))
        chars = [(c, 62.0) for c in reg] + [(c, 76.0) for c in d3] + [(c, 62.0) for c in l3]
        g, _ = _row_layout(chars, 60.0, 8.0, 94.0)
        for gl in g[2:]:
            gl.x += 12.0
        for gl in g[5:]:
            gl.x += 12.0
        spec = PlateSpec("other", "kg", reg + d3 + l3, W, H, WHITE, BLACK, g)
        spec.strip = ((200, 40, 40), 40.0, "KG")
    elif country == "am":
        d2, l2, d3 = rand_digits(rng, 2, True), "".join(rng.choice(FOREIGN_LETTERS) for _ in range(2)), rand_digits(rng, 3)
        chars = [(c, 76.0) for c in d2] + [(c, 62.0) for c in l2] + [(c, 76.0) for c in d3]
        g, _ = _row_layout(chars, 70.0, 9.0, 94.0)
        for gl in g[2:]:
            gl.x += 12.0
        for gl in g[4:]:
            gl.x += 12.0
        spec = PlateSpec("other", "am", d2 + l2 + d3, W, H, WHITE, BLACK, g)
        spec.strip = ((40, 90, 200), 40.0, "AM")
    elif country == "ge":
        l2a, d3, l2b = "".join(rng.choice(FOREIGN_LETTERS) for _ in range(2)), rand_digits(rng, 3), "".join(rng.choice(FOREIGN_LETTERS) for _ in range(2))
        chars = [(c, 62.0) for c in l2a] + [("-", 62.0)] + [(c, 76.0) for c in d3] + [("-", 62.0)] + [(c, 62.0) for c in l2b]
        g, _ = _row_layout(chars, 70.0, 8.0, 94.0)
        spec = PlateSpec("other", "ge", l2a + d3 + l2b, W, H, WHITE, BLACK, g)
        spec.strip = ((40, 90, 200), 40.0, "GE")
    else:  # generic EU
        n = rng.choice(["LLL-DDD", "LL-DDDD", "DDD LLL", "L DD LLL"])
        text = "".join(rng.choice(FOREIGN_LETTERS) if c == "L" else (rng.choice(DIGITS) if c == "D" else c) for c in n)
        chars = [(c, 76.0 if c.isdigit() else 62.0) for c in text if c != " "]
        g, _ = _row_layout(chars, 70.0, 10.0, 94.0)
        spec = PlateSpec("other", "eu", text.replace("-", "").replace(" ", ""), W, H, WHITE, BLACK, g)
        spec.strip = ((30, 60, 170), 40.0, rng.choice(["D", "FIN", "EST", "LV", "LT", "PL", "F", "I"]))
    return spec


OTHER_BUILDERS = {
    "trailer": build_trailer, "moto": build_moto, "tractor": build_tractor, "police": build_police,
    "military": build_military, "diplomatic": build_diplomatic, "transit": build_transit, "foreign": build_foreign,
}
OTHER_WEIGHTS = {"trailer": 0.16, "moto": 0.14, "tractor": 0.06, "police": 0.10, "military": 0.10, "diplomatic": 0.10,
                 "transit": 0.08, "foreign": 0.26}


def build_plate(rng: random.Random, kind: str) -> PlateSpec:
    if kind == "type1":
        return build_type1(rng)
    if kind == "type1a":
        return build_type1a(rng)
    if kind == "type1b":
        return build_type1(rng, yellow=True, gost_taxi=rng.random() < 0.55)
    if kind == "other":
        names = list(OTHER_WEIGHTS)
        sub = rng.choices(names, weights=[OTHER_WEIGHTS[n] for n in names])[0]
        return OTHER_BUILDERS[sub](rng)
    raise ValueError(kind)


# ----------------------------------------------------------------------------------------------
# Rasterisation
# ----------------------------------------------------------------------------------------------
def _jit(rng: random.Random, v: float, frac: float) -> float:
    return v * (1.0 + rng.uniform(-frac, frac))


def render_plate(spec: PlateSpec, fonts: Fonts, rng: random.Random, scale: float = 4.0, jitter: float = 0.06) -> Tuple[Image.Image, Dict]:
    """Rasterise a plate. Returns RGB image and a dict with per-glyph boxes (px) and row split (px)."""
    font_path = fonts.pick(rng)
    W, H = int(spec.width * scale), int(spec.height * scale)
    bg = tuple(int(min(255, max(0, c + rng.randint(-10, 8)))) for c in spec.bg)
    if spec.kind == "type1b":   # yellow varies a lot in the wild
        bg = (int(_jit(rng, 245, 0.05)), int(_jit(rng, 195, 0.12)), int(rng.uniform(0, 60)))
    fg = tuple(int(min(255, max(0, c + rng.randint(-10, 10)))) for c in spec.fg)
    img = Image.new("RGB", (W, H), bg)
    d = ImageDraw.Draw(img)
    # frame line
    r = int(8 * scale)
    lw = max(1, int(_jit(rng, 1.6, 0.4) * scale))
    inset = int(_jit(rng, 2.5, 0.4) * scale)
    d.rounded_rectangle([inset, inset, W - 1 - inset, H - 1 - inset], radius=r, outline=fg, width=lw)
    # transit: yellow region field
    if spec.subkind == "transit" and spec.region_box:
        x0, y0, x1, y1 = [int(v * scale) for v in spec.region_box]
        d.rectangle([x0 + lw, inset + lw, x1 - inset - lw, y1 - inset - lw], fill=(245, 200, 30))
    # left strip (foreign)
    if spec.strip and spec.strip[1] > 0:
        col, sw, txt = spec.strip
        d.rectangle([inset, inset, int(sw * scale), H - inset], fill=col)
        if txt:
            f = fonts.get(font_path, int(14 * scale))
            d.text((int(6 * scale), H - int(26 * scale)), txt, font=f, fill=WHITE)
    for (x0, y0, x1, y1) in spec.lines:
        d.line([(x0 * scale, y0 * scale), (x1 * scale, y1 * scale)], fill=fg, width=lw)
    boxes = []
    # --- glyph layout with measured widths -------------------------------------------------
    def prep(g):
        h = _jit(rng, g.h, jitter)
        ch = LAT2CYR.get(g.ch, g.ch)
        if ch in "-|":
            f = fonts.get(font_path, max(6, int(h * scale)))
            return ch, f, (0, 0, int(h * scale * 0.45), int(h * scale)), int(h * scale * 0.45), int(h * scale)
        f = fonts.get(font_path, max(6, int(h * scale)))
        bb = f.getbbox(ch)
        gh = max(1, bb[3] - bb[1])
        f = fonts.get(font_path, max(6, int(int(h * scale) * (h * scale) / gh)))
        bb = f.getbbox(ch)
        return ch, f, bb, (bb[2] - bb[0]), (bb[3] - bb[1])

    in_region = lambda g: spec.region_box is not None and spec.region_box[0] <= g.x < spec.region_box[2] and spec.region_box[1] <= g.y < spec.region_box[3]
    prepared = [(g, prep(g)) for g in spec.glyphs]
    groups = {}
    for g, pr in prepared:
        groups.setdefault((g.row, in_region(g)), []).append((g, pr))
    for (row, reg), items in groups.items():
        items.sort(key=lambda t: t[0].x)
        # nominal gaps between consecutive glyphs
        widths = [pr[3] for _, pr in items]
        gaps = []
        for i in range(1, len(items)):
            prev, cur = items[i - 1][0], items[i][0]
            gaps.append(max(2.0, cur.x - (prev.x + char_width(prev.ch, prev.h))) * scale)
        total = sum(widths) + sum(gaps)
        if not reg:
            # keep the group inside the plate / left of the region field
            limit = spec.width - 5.0
            if spec.region_box is not None:
                rx0, ry0, rx1, ry1 = spec.region_box
                gy0, gy1 = min(g.y for g, _ in items), max(g.y + g.h for g, _ in items)
                if gy1 > ry0 and gy0 < ry1:
                    limit = min(limit, rx0 - 5.0)
            avail = limit * scale - items[0][0].x * scale
            if total > avail:
                k = max(0.3, (avail - sum(widths)) / max(1.0, sum(gaps)))
                gaps = [gp * min(1.0, k) for gp in gaps]
                total = sum(widths) + sum(gaps)
                if total > avail:
                    shrink = avail / total
                    items = [(Glyph(g.ch, g.x, g.y + g.h * (1 - shrink), g.h * shrink, g.row), None) for g, _ in items]
                    items = [(g, prep(g)) for g, _ in items]
                    widths = [pr[3] for _, pr in items]
                    total = sum(widths) + sum(gaps)
        if reg:
            x0, _, x1, _ = spec.region_box
            x = ((x0 + x1) / 2.0) * scale - total / 2.0
        else:
            x = items[0][0].x * scale
        for i, (g, (ch, f, bb, gw, gh)) in enumerate(items):
            if i > 0:
                x += gaps[i - 1]
            xx = x + rng.uniform(-1.0, 1.0) * scale
            y_bottom = (g.y + g.h) * scale
            y = y_bottom - gh - bb[1]
            if ch == "-":
                d.rectangle([xx, y_bottom - gh * 0.55, xx + gw, y_bottom - gh * 0.45], fill=fg)
            elif ch == "|":
                d.rectangle([xx + gw * 0.4, y_bottom - gh, xx + gw * 0.6, y_bottom], fill=fg)
            else:
                d.text((xx - bb[0], y), ch, font=f, fill=fg)
            boxes.append((g.ch, xx, y_bottom - gh, xx + gw, y_bottom, g.row))
            x += gw
    for (txt, x, y, h) in spec.small_text:
        f = fonts.get(font_path, int(h * scale))
        d.text((x * scale, y * scale), txt, font=f, fill=fg)
    if spec.rus:
        x, y, h = spec.rus
        f = fonts.get(font_path, int(h * scale))
        d.text((x * scale, y * scale), "RUS", font=f, fill=fg)
    if spec.flag:
        x0, y0, x1, y1 = [int(v * scale) for v in spec.flag]
        hh = (y1 - y0) // 3
        d.rectangle([x0, y0, x1, y0 + hh], fill=(255, 255, 255))
        d.rectangle([x0, y0 + hh, x1, y0 + 2 * hh], fill=(0, 57, 166))
        d.rectangle([x0, y0 + 2 * hh, x1, y1], fill=(213, 43, 30))
        d.rectangle([x0, y0, x1, y1], outline=(60, 60, 60), width=max(1, lw // 2))
    info = dict(boxes=boxes, row_split=spec.row_split * scale, W=W, H=H, font=os.path.basename(font_path))
    return img, info


if __name__ == "__main__":
    import sys
    fonts = Fonts()
    rng = random.Random(0)
    out = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(out, exist_ok=True)
    for i, kind in enumerate(["type1", "type1a", "type1b", "type1b", "other", "other", "other", "other", "other", "other", "other", "other"]):
        spec = build_plate(rng, kind)
        img, info = render_plate(spec, fonts, rng, scale=3)
        img.save(os.path.join(out, f"plate_{i}_{spec.subkind}_{spec.text}.png"))
        print(kind, spec.subkind, spec.text, info["font"])
