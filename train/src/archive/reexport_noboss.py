# -*- coding: utf-8 -*-
"""把 round2 里混进 BOSS 的样本重导（--no-boss 现在只留 tier=normal 的小怪）。

背景：最初 --no-boss 只排除 tier=big（HP≥1000万的大 BOSS），WZ 标了 boss=1
但血量不到 1000 万的小 BOSS（Alishar / Shadow Kyrin / Thanatos / 各种 Clone）
仍然被合成进图，共 1089 张、47 个类、其中多数类只有 4~12 个框，是纯长尾噪声。

做法（可回滚）：
  1. 扫 labels/ 找出含 BOSS 类的图 → 连同 label 一起移到 _old_boss/（不删除）
  2. 用 batch_r2_result.json 里的同一取景框 view / density / neg 重导，
     seed 加后缀 "nb" → 新文件名，布局重随机，但保证没有 BOSS
  3. 收尾自动 compact_classes + 复验

用法：python train/src/archive/reexport_noboss.py --workers 14 [--dry-run]
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
YDIR = ROOT / "train" / "data" / "yolo_r2"
OLD = YDIR / "_old_boss"
PY = [sys.executable, str(ROOT / "wztool.py")]
W, H = 1366, 768

sys.path.insert(0, str(Path(__file__).resolve().parent))
from batch_export import log  # noqa: E402


def boss_class_ids() -> set[int]:
    """classes.txt 里 tier 属于 boss/big 的类 id。"""
    sys.path.insert(0, str(ROOT))
    import wztool as Wz
    ws = Wz._need_ws()
    meta = {m["id"]: m for m in Wz.mob_meta(ws, ws / "Mob.nx")}
    names = [l for l in (YDIR / "classes.txt").read_text(encoding="utf-8").splitlines() if l.strip()]
    out = set()
    for i, nm in enumerate(names):
        raw = nm.split("_", 1)[0].lstrip("0")
        if meta.get(raw, {}).get("tier", "normal") != "normal":
            out.add(i)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    bad_ids = boss_class_ids()
    log(f"BOSS 级类别 {len(bad_ids)} 个")

    # 1) 找出受污染样本：既包括「labels 里还含 BOSS 的」，也包括上一轮已隔离到
    #    _old_boss 但还没补回新图的（重跑时从这里续）
    targets = []
    for p in sorted((YDIR / "labels").glob("*.txt")):
        ids = set()
        for ln in p.read_text(encoding="utf-8").splitlines():
            f = ln.split()
            if f:
                ids.add(int(f[0]))
        if ids & bad_ids:
            targets.append(p.stem)
    old_dir = OLD / "labels"
    if old_dir.exists():
        for p in sorted(old_dir.glob("*.txt")):
            if (YDIR / "images" / f"{p.stem}nb.png").exists():
                continue          # 已补回
            targets.append(p.stem)
    targets = sorted(set(targets))
    log(f"待补样本 {len(targets)} 张（含 BOSS 未隔离的 + 隔离区未补回的）")

    # 2) 查回原取景框／密度
    res = json.loads((ROOT / "train" / "batch_r2_result.json").read_text(encoding="utf-8"))
    by_stem = {f"{r['id']}_{r['seed']}": r for r in res if r.get("view")}
    miss = [s for s in targets if s not in by_stem]
    if miss:
        log(f"结果表里找不到 {len(miss)} 张（将跳过）: {miss[:5]}")
    jobs = []
    for s in targets:
        r = by_stem.get(s)
        if not r:
            continue
        jobs.append((r["id"], r["view"][0], r["view"][1], r["density"],
                     s.split("_", 1)[1] + "nb", bool(r["neg"]), s))
    log(f"待重导 {len(jobs)} 张（view 从结果表还原）")
    if not jobs:
        log("没有需要重导的样本，退出")
        return 0

    # 轮转排序：同图的瓦片打散到不同批次，避免一张巨型地图（单窗口渲染
    # 可能要 8 分钟）把全部 worker 一起堵死；快图先出，慢图自然落到队尾
    by_map: dict[str, list] = {}
    for j in jobs:
        by_map.setdefault(j[0], []).append(j)
    jobs = [by_map[m][i] for i in range(max(len(v) for v in by_map.values()))
            for m, v in by_map.items() if i < len(v)]

    if a.dry_run:
        log("dry-run：未改动任何文件")
        return 0

    # 3) 旧样本移入 _old_boss（可回滚）
    (OLD / "images").mkdir(parents=True, exist_ok=True)
    (OLD / "labels").mkdir(parents=True, exist_ok=True)
    for s in targets:
        for src, sub in ((YDIR / "images" / f"{s}.png", "images"),
                         (YDIR / "labels" / f"{s}.txt", "labels")):
            if src.exists():
                shutil.move(str(src), str(OLD / sub / src.name))
    log(f"旧样本已隔离到 {OLD.relative_to(ROOT)}")

    # 4) 并发重导（单张 240s 超时：个别巨型地图渲染极慢，宁可放弃该瓦片——
    #    它的旧图已隔离在 _old_boss，等于这张瓦片不出样本，不产生污染）
    def job(j):
        mid, x, y, density, seed, neg, _old = j
        if (YDIR / "images" / f"{mid}_{seed}.png").exists():   # 断点续跑
            return (mid, seed, "done")
        args = ["map-export", mid, "--preview", "--json",
                f"--view={x},{y},{W},{H}", "--yolo", "--mob-only", "--no-boss",
                f"--seed={seed}", "--yolo-dir=train/data/yolo_r2"]
        if not neg:
            args += ["--mobs", f"--density={density}"]
        try:
            r = subprocess.run([*PY, *args], capture_output=True, cwd=str(ROOT), timeout=240)
        except subprocess.TimeoutExpired:
            return (mid, seed, "timeout")
        if r.returncode != 0:
            return (mid, seed, r.stderr[-120:].decode("utf-8", "replace"))
        return (mid, seed, None)

    ok = fail = skip = 0
    errs = Counter()
    with cf.ThreadPoolExecutor(max_workers=a.workers) as ex:
        for i, (mid, seed, err) in enumerate(ex.map(job, jobs), 1):
            if err == "done":
                skip += 1
            elif err == "timeout":
                skip += 1
                errs["timeout"] += 1
                log(f"超时放弃 {mid}_{seed}（旧图已在 _old_boss，本瓦片不出样本）")
            elif err:
                fail += 1
                errs[str(err)[:60]] += 1
                log(f"失败 {mid}_{seed}: {err}")
            else:
                ok += 1
            if i % 100 == 0:
                log(f"进度 {i}/{len(jobs)}（成功 {ok} · 跳过 {skip} · 失败 {fail}）")
    log(f"重导完成：成功 {ok} · 跳过(已有/超时) {skip} · 失败 {fail}")
    if errs:
        log(f"错误分布: {dict(errs.most_common(3))}")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
