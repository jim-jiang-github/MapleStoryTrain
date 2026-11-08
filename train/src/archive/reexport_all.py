# -*- coding: utf-8 -*-
"""round2 全量重导（修类别 id 漂移）。

为什么重导：并发导出时多个进程同时往 classes.json 追加新类，`id = len(cur)`
在同一位置互相覆盖，导致先写那批图的 label 指向了别人的类（实测 2569 张、162 张地图）。
label 无法事后反推正确 id，只能重出。

流程：
  1. 用 wztool yolo-classes 单进程预注册这批地图的全部类别（只留普通小怪）
  2. 现有 images/labels 整体移到 _old_all/（可回滚，不删除）
  3. 设置 WZ_CLASSES_RO=1（导出进程禁止改写类别表）→ 按原取景框/密度重导，seed 加 "fx"
  4. 轮转排序 + 单张 240s 超时，避免巨型地图堵死

用法：python train/src/archive/reexport_all.py --workers 14 [--dry-run]
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
YDIR = ROOT / "train" / "data" / "yolo_r2"
OLD = YDIR / "_old_all"
PY = [sys.executable, str(ROOT / "wztool.py")]
W, H = 1366, 768

sys.path.insert(0, str(Path(__file__).resolve().parent))
from batch_export import log  # noqa: E402


def normal_life_maps() -> set[str]:
    """当前有普通小怪的图（纯 BOSS 图排除）。"""
    sys.path.insert(0, str(ROOT))
    import wztool as Wz
    ws = Wz._need_ws()
    meta = {m["id"]: m for m in Wz.mob_meta(ws, ws / "Mob.nx")}
    out = set()
    for p in Path(ROOT / ".cache" / "map").glob("*/map.json"):
        try:
            mj = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        for l in mj.get("life", []):
            if l.get("type") != "m":
                continue
            k = str(l.get("id", "")).lstrip("0")
            if meta.get(k, {}).get("tier", "normal") == "normal":
                out.add(p.parent.name)
                break
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    res = json.loads((ROOT / "train" / "batch_r2_result.json").read_text(encoding="utf-8"))
    keep_maps = normal_life_maps()
    log(f"有小怪的地图（缓存里）: {len(keep_maps)}")

    # SKIP_MAPS：跳过渲染极慢的巨型地图（单张 5 分钟，会拖死整批）
    skip = {m.strip() for m in os.environ.get("SKIP_MAPS", "").split(",") if m.strip()}
    if skip:
        log(f"按 SKIP_MAPS 跳过 {len(skip)} 张巨图")

    only = {m.strip() for m in os.environ.get("ONLY_MAPS", "").split(",") if m.strip()}
    if only:
        log(f"只处理 ONLY_MAPS 指定的 {len(only)} 张图")

    jobs = []
    for r in res:
        if r["id"] not in keep_maps or not r.get("view") or r["id"] in skip:
            continue
        if only and r["id"] not in only:
            continue
        jobs.append((r["id"], r["view"][0], r["view"][1], r["density"],
                     r["seed"] + "fx", bool(r["neg"])))
    log(f"计划重导 {len(jobs)} 张（{len({j[0] for j in jobs})} 张地图）")

    if a.dry_run:
        log("dry-run：未改动")
        return 0

    # 1) 预注册类别（单进程，id 稳定）
    ids = sorted({j[0] for j in jobs})
    r = subprocess.run([*PY, "yolo-classes", "--yolo-dir=train/data/yolo_r2",
                        "--no-boss", "--json"],
                       input="\n".join(ids).encode(), capture_output=True,
                       cwd=str(ROOT), timeout=3600)
    if r.returncode != 0:
        log("类别预注册失败: " + r.stderr[-400:].decode("utf-8", "replace"))
        return 1
    info = json.loads(r.stdout.decode("utf-8", "replace"))
    log(f"类别预注册: {info.get('classes')} 类 / {info.get('maps')} 张图")

    # 2) 备份现有数据
    OLD.mkdir(parents=True, exist_ok=True)
    for sub in ("images", "labels"):
        (OLD / sub).mkdir(exist_ok=True)
    moved = 0
    if only:
        log("ONLY_MAPS 模式：跳过备份（现有数据已校验过，不能动）")
    for sub, pat in ([] if only else [("images", "*.png"), ("labels", "*.txt")]):
        for p in (YDIR / sub).glob(pat):
            if p.stem.endswith("fx"):
                continue          # 本轮已出的新图，保留（断点续跑）
            shutil.move(str(p), str(OLD / sub / p.name))
            moved += 1
    log(f"旧数据已备份到 {OLD.relative_to(ROOT)}（{moved} 个文件）")

    # 3) 按地图分组：同图瓦片连排，贴图/地图 json 走页缓存，比轮转快 2~3 倍；
    #    巨型地图靠 180s 超时兜底（超时的瓦片直接放弃，不拖住整批）
    by_map: dict[str, list] = {}
    for j in jobs:
        by_map.setdefault(j[0], []).append(j)
    jobs = [j for v in by_map.values() for j in v]

    env = dict(os.environ, WZ_CLASSES_RO="1")

    def job(j):
        mid, x, y, density, seed, neg = j
        if (YDIR / "images" / f"{mid}_{seed}.png").exists():
            return (mid, seed, "done")
        args = ["map-export", mid, "--preview", "--json",
                f"--view={x},{y},{W},{H}", "--yolo", "--mob-only", "--no-boss",
                f"--seed={seed}", "--yolo-dir=train/data/yolo_r2"]
        if not neg:
            args += ["--mobs", f"--density={density}"]
        try:
            p = subprocess.run([*PY, *args], capture_output=True, cwd=str(ROOT),
                               timeout=int(os.environ.get("WZ_TILE_TIMEOUT", "90")), env=env)
        except subprocess.TimeoutExpired:
            return (mid, seed, "timeout")
        if p.returncode != 0:
            return (mid, seed, p.stderr[-120:].decode("utf-8", "replace"))
        return (mid, seed, None)

    ok = skip = fail = 0
    errs = Counter()
    with cf.ThreadPoolExecutor(max_workers=a.workers) as ex:
        for i, (mid, seed, err) in enumerate(ex.map(job, jobs), 1):
            if err == "done":
                skip += 1
            elif err == "timeout":
                skip += 1
                errs["timeout"] += 1
            elif err:
                fail += 1
                errs[str(err)[:50]] += 1
            else:
                ok += 1
            if i % 200 == 0:
                log(f"进度 {i}/{len(jobs)}（成功 {ok} 跳过 {skip} 失败 {fail}）")
    log(f"重导完成：成功 {ok} · 跳过 {skip} · 失败 {fail}")
    if errs:
        log(f"错误分布: {dict(errs.most_common(3))}")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
