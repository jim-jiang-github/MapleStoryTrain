"""yolo_export.py —— 训练集导出（渲染用子模块，标注归训练仓库）。

对外命令行刻意保持和旧版 wztool 一致，这样 batch_export2.py / boost_weak*.py
不用改调用方式：

    yolo_export.py maps --json
    yolo_export.py probe --no-boss --w 1366 --h 768      （stdin 给 id 列表）
    yolo_export.py yolo-classes --yolo-dir=DIR --no-boss --json
    yolo_export.py map-export <id> --preview --json --view=X,Y,W,H \
                   --yolo --mob-only --no-boss --seed=S \
                   --yolo-dir=DIR --max-mobs=30 [--mobs --density=3]

上游 WzToolkit 只出图、不出标注，所以这里：
  1. 用它的 _render_preview 渲 1:1 整图（层序/视差与 exe 同源）；
  2. 用它的 _composite_mobs 摆怪（贴地、动作随机、seed 可复现），拿回每只怪的像素框；
  3. 把窗口 1366×768 裁出来（和客户端画面同尺寸，不缩放）；
  4. 自己算 YOLO 框、写 classes.json / dataset.yaml。

「只标普通小怪 / 最多 N 只 / 只数窗口附近的刷怪点」这些是训练策略，全部在
本文件里，不污染上游。
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import wztk  # noqa: E402

W, H = wztk.W, wztk.H


# ---------------------------------------------------------------- maps

def cmd_maps(a) -> int:
    d = wztk.wztool().c_maps()
    print(json.dumps(d, ensure_ascii=False))
    return 0


# ---------------------------------------------------------------- probe

def probe_one(mid: str, win_w: int = W, win_h: int = H, no_boss: bool = False) -> dict:
    """这张图能不能开 1366×768 的窗口（够大、不缩放、有怪）。"""
    rec: dict = {"id": mid}
    try:
        mj, _, real = wztk.ensure_map(mid)
        minx, miny, bw, bh = wztk.bounds(mj)
        life = [l for l in mj.get("life", []) if l.get("type") == "m"]
        keep = ([l for l in life if wztk.tier_of(l.get("id", "")) == "normal"]
                if no_boss else life)
        rec.update({"world": {"x": minx, "y": miny, "w": bw, "h": bh},
                    "life_count": len(life), "small_count": len(keep),
                    "usable": bool(keep) and bw >= win_w and bh >= win_h,
                    "usable_view": bool(keep) and bw >= win_w and bh >= win_h})
        if real != mid:
            rec["link_to"] = real
        if no_boss and len(keep) < len(life):
            rec["boss_filtered"] = len(life) - len(keep)
    except Exception as e:                       # 单张坏图不拖垮整批
        rec.update({"usable": False, "usable_view": False,
                    "error": f"{type(e).__name__}: {e}"})
    return rec


def cmd_probe(a) -> int:
    ids = [ln.strip() for ln in sys.stdin.read().splitlines() if ln.strip()]
    out = [probe_one(i, a.w, a.h, a.no_boss) for i in ids]
    print(json.dumps(out, ensure_ascii=False))
    return 0


# ---------------------------------------------------------------- 类别表

def cmd_classes(a) -> int:
    ids = [ln.strip() for ln in sys.stdin.read().splitlines() if ln.strip()]
    ydir = Path(a.yolo_dir or (wztk.REPO / "train" / "data" / "yolo_r2"))
    if not ydir.is_absolute():
        ydir = wztk.REPO / ydir
    kinds, ok = set(), 0
    for mid in ids:
        try:
            mj, _, _ = wztk.ensure_map(mid)
        except Exception:
            continue
        ok += 1
        for l in mj.get("life", []):
            if l.get("type") != "m":
                continue
            k = str(l.get("id", "")).lstrip("0")
            if not k:
                continue
            if a.no_boss and wztk.tier_of(k) != "normal":
                continue
            kinds.add(k)
    reg = wztk.register_classes(ydir, [wztk.species_name(k) for k in kinds])
    print(json.dumps({"dir": str(ydir), "maps": ok, "classes": len(reg),
                      "class_file": str(ydir / "classes.json"),
                      "dataset": str(ydir / "dataset.yaml")}, ensure_ascii=False))
    return 0


# ---------------------------------------------------------------- 单窗口导出

def _parse_view(s: str) -> tuple[float, float, int, int] | None:
    p = [x for x in re.split(r"[,\s]+", (s or "").strip()) if x]
    if not p:
        return None                              # 没给 view：取地图正中
    if len(p) < 2:
        raise SystemExit("[x] --view 至少给 X,Y（世界坐标）")
    x, y = float(p[0]), float(p[1])
    w = int(p[2]) if len(p) > 2 else W
    h = int(p[3]) if len(p) > 3 else H
    return x, y, w, h


def _center_view(minx: float, miny: float, bw: float, bh: float,
                 w: int, h: int) -> tuple[float, float]:
    """窗口取地图正中；地图比窗口小就居中（窗口可以超出地图边界）。"""
    vx = minx + (bw - w) / 2
    vy = miny + (bh - h) / 2
    return round(vx), round(vy)


def _window_life(mj: dict, x: float, y: float, w: int, h: int,
                 no_boss: bool, max_mobs: int, density: int,
                 seed: str) -> list[dict]:
    """挑出「可能落进这个窗口」的刷怪点，并把巡逻区间夹到窗口附近。

    不这么做的话，_composite_mobs 会把整张图的怪都摆一遍（几十只摆到画面外），
    既慢又会把 max_mobs 的名额浪费在画外。
    """
    life = [l for l in mj.get("life", []) if l.get("type") == "m"]
    if no_boss:
        life = [l for l in life if wztk.tier_of(l.get("id", "")) == "normal"]
    spread = max(120.0, (w + 80) * 0.08)
    out = []
    for l in life:
        lx = float(l.get("x", 0))
        ly = (float(l["cy"]) if l.get("fh") and l.get("cy") is not None
              else float(l.get("y", 0)))
        if not (y - 160 <= ly <= y + h + 160):
            continue
        rx0, rx1 = l.get("rx0"), l.get("rx1")
        if rx0 is None or rx1 is None or float(rx1) <= float(rx0):
            lo, hi = lx - spread, lx + spread
        else:
            lo, hi = float(rx0), float(rx1)
        if hi < x - 60 or lo > x + w + 60:
            continue
        out.append({**l, "rx0": max(lo, x - 60), "rx1": min(hi, x + w + 60)})
    if max_mobs > 0:
        cap = max(1, math.ceil(max_mobs / max(1, density)))
        if len(out) > cap:                     # 同 seed 同子集，断点续跑可复现
            rng = random.Random(f"{seed}:cap")
            out = rng.sample(out, cap)
    return out


def _crop(src: Path, left: int, top: int, w: int, h: int):
    from PIL import Image
    im = Image.open(src).convert("RGB")
    if left >= 0 and top >= 0 and left + w <= im.width and top + h <= im.height:
        return im.crop((left, top, left + w, top + h))
    canvas = Image.new("RGB", (w, h), (16, 18, 24))     # 窗口越过地图边界
    sx0, sy0 = max(0, left), max(0, top)
    sx1, sy1 = min(im.width, left + w), min(im.height, top + h)
    if sx1 > sx0 and sy1 > sy0:
        canvas.paste(im.crop((sx0, sy0, sx1, sy1)), (sx0 - left, sy0 - top))
    return canvas


def _labels(places: list[dict], left: int, top: int, w: int, h: int,
            reg: dict[str, int]) -> tuple[list[str], dict[str, int], list[str]]:
    lines, states, kinds = [], {}, []
    for p in places:
        x1, y1 = p["x"] - left, p["y"] - top
        x2, y2 = x1 + p["w"], y1 + p["h"]
        cx1, cy1 = max(0, x1), max(0, y1)
        cx2, cy2 = min(w, x2), min(h, y2)
        bw, bh = cx2 - cx1, cy2 - cy1
        if bw < 4 or bh < 4:                   # 完全在画外 / 只剩一条边
            continue
        name = wztk.species_name(p["kind"])
        cid = reg.get(name)
        if cid is None:
            continue
        lines.append(f"{cid} {((cx1 + cx2) / 2) / w:.6f} {((cy1 + cy2) / 2) / h:.6f} "
                     f"{bw / w:.6f} {bh / h:.6f}")
        states[p.get("state", "?")] = states.get(p.get("state", "?"), 0) + 1
        kinds.append(name)
    return lines, states, kinds


def cmd_map_info(a) -> int:
    mj, _, real = wztk.ensure_map(a.id)
    minx, miny, bw, bh = wztk.bounds(mj)
    life = [l for l in mj.get("life", []) if l.get("type") == "m"]
    print(json.dumps({"id": real, "name": mj.get("name", ""),
                      "world": {"x": minx, "y": miny, "w": bw, "h": bh},
                      "life_count": len(life),
                      "life_kinds": len({str(l.get('id', '')).lstrip('0') for l in life})},
                     ensure_ascii=False))
    return 0


def cmd_map_export(a) -> int:
    v = _parse_view(a.view)
    mj0, _, real0 = wztk.ensure_map(a.id)
    if v is None:
        minx0, miny0, bw0, bh0 = wztk.bounds(mj0)
        cx, cy = _center_view(minx0, miny0, bw0, bh0, W, H)
        v = (cx, cy, W, H)
    x, y, w, h = v
    ydir = Path(a.yolo_dir or (wztk.REPO / "train" / "data" / "yolo_r2"))
    if not ydir.is_absolute():
        ydir = wztk.REPO / ydir
    (ydir / "images").mkdir(parents=True, exist_ok=True)
    (ydir / "labels").mkdir(parents=True, exist_ok=True)

    mj, outdir, real = wztk.ensure_map(a.id)
    Wm = wztk.wztool()
    base = outdir / "preview.png"
    Wm._render_preview(outdir, base)
    minx, miny, _, _ = wztk.bounds(mj)
    left, top = round(x - minx), round(y - miny)

    places: list[dict] = []
    src = base
    tmp_out: Path | None = None
    if a.mobs:
        life = _window_life(mj, x, y, w, h, a.no_boss, a.max_mobs, a.density, a.seed)
        if life:
            mj2 = {**mj, "life": life}
            # _composite_mobs 把合成结果写到 outdir/preview_mobs.png——并发导同一张
            # 地图时会互踩（A 正在裁剪，B 把半成品写进去 → "image file is truncated"）。
            # 给每个进程一个临时目录，用完即删。
            tmp_out = outdir / f"_tmp_{os.getpid()}"
            tmp_out.mkdir(parents=True, exist_ok=True)
            mob_png, places = Wm._composite_mobs(wztk.workspace(), tmp_out, mj2, base,
                                                 a.seed, density=a.density)
            if mob_png is not None:
                src = mob_png
        else:
            places = []

    img = _crop(src, left, top, w, h)
    stem = f"{a.id}_{a.seed}"
    img_path = ydir / "images" / f"{stem}.png"
    img.save(img_path)
    if tmp_out is not None:
        shutil.rmtree(tmp_out, ignore_errors=True)

    # 先按这一张实际出现的种类注册类别表（只增不改，历史 label 不会错位），
    # 再算框——顺序反了会拿到空表，框全被丢掉
    names = sorted({wztk.species_name(p["kind"]) for p in places})
    reg = wztk.register_classes(ydir, names)
    lines, states, kinds = _labels(places, left, top, w, h, reg)
    lab_path = ydir / "labels" / f"{stem}.txt"
    lab_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    out = {"dir": str(outdir), "preview": {"w": img.width, "h": img.height},
           "window": {"x": x, "y": y, "w": w, "h": h},
           "yolo": {"image": str(img_path), "labels": str(lab_path),
                    "boxes": len(lines), "states": states,
                    "classes": sorted(set(kinds))}}
    if real != a.id:
        out["link"] = {"from": a.id, "to": real}
    print(json.dumps(out, ensure_ascii=False))
    return 0


# ---------------------------------------------------------------- CLI

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="训练集导出（渲染用 WzToolkit 子模块）")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("maps")
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_maps)

    p = sub.add_parser("probe")
    p.add_argument("--no-boss", action="store_true")
    p.add_argument("--w", type=int, default=W)
    p.add_argument("--h", type=int, default=H)
    p.set_defaults(fn=cmd_probe)

    p = sub.add_parser("yolo-classes")
    p.add_argument("--yolo-dir", default="")
    p.add_argument("--no-boss", action="store_true", default=True)
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_classes)

    p = sub.add_parser("map-export")
    p.add_argument("id")
    p.add_argument("--preview", action="store_true")
    p.add_argument("--json", action="store_true")
    p.add_argument("--view", default="")
    p.add_argument("--yolo", action="store_true")
    p.add_argument("--mob-only", action="store_true")
    p.add_argument("--no-boss", action="store_true", default=True)
    p.add_argument("--seed", default="")
    p.add_argument("--yolo-dir", default="")
    p.add_argument("--max-mobs", type=int, default=0)
    p.add_argument("--mobs", action="store_true")
    p.add_argument("--density", type=int, default=1)
    p.set_defaults(fn=cmd_map_export)

    p = sub.add_parser("map-info")
    p.add_argument("id")
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_map_info)

    a = ap.parse_args(argv)
    if getattr(a, "seed", None) == "":         # 没给 seed 时用固定值，保证可复现
        a.seed = "default"
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
