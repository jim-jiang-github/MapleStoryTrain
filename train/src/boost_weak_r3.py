"""round3 弱类定向加量：AP 低的类所在的地图，每块瓦片多出 3 版新 seed 样本。

数据来源：
  * train/reports/weak_classes.txt        —— 全类评估(train口径) mAP50<0.85 的类
  * train/reports/perclass_rank.txt       —— val 口径逐类表（78 类），<0.8 的也并入
  * train/data/yolo_r2/labels/{train,val} —— 实际标注里找弱类出现在哪些地图
  * train/batch_r2_result.json    —— 每张地图的瓦片坐标（复用 r2 的网格计划）

每块瓦片出 3 份新样本（tag r3）：density 1 / 2 / 3，seed 全新，
怪种组合、朝向、动作帧都重新抽。类别表 473 类已注册，导出全程只读。

用法：
  python train/boost_weak_r3.py --dry-run     # 先看规模
  WZ_CLASSES_RO=1 python train/boost_weak_r3.py --workers 12
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
YDIR = ROOT / "train" / "data" / "yolo_r2"
sys.path.insert(0, str(Path(__file__).resolve().parent))
from batch_export import log                       # noqa: E402
from batch_export2 import export_job               # noqa: E402

TAG = "r3"
MAX_MOBS = 20


def weak_names() -> set[str]:
    names: set[str] = set()
    p1 = ROOT / "train" / "reports" / "weak_classes.txt"          # 全类口径 <0.85
    if p1.exists():
        for ln in p1.read_text(encoding="utf-8").splitlines():
            f = ln.split("\t")
            if f and f[0].strip():
                names.add(f[0].strip())
    p2 = ROOT / "train" / "reports" / "perclass_rank.txt"         # val 口径 <0.8
    if p2.exists():
        for ln in p2.read_text(encoding="utf-8").splitlines():
            f = ln.split("\t")
            if len(f) >= 5 and f[0].strip() and float(f[4]) < 0.8:
                names.add(f[0].strip())
    return names


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--variants", type=int, default=3, help="每瓦片出几版（密度 1..N）")
    a = ap.parse_args()

    # 1) 弱类 id
    reg = json.loads((YDIR / "classes.json").read_text(encoding="utf-8"))  # 名->id
    wn = weak_names()
    weak_ids = {reg[n] for n in wn if n in reg}
    missing = {n for n in wn if n not in reg}
    log(f"弱类 {len(wn)} 个（类别表命中 {len(weak_ids)}"
        + (f"，表里没有: {sorted(missing)[:5]}" if missing else "") + "）")

    # 2) 扫标注找弱类所在地图
    cls_hit: dict[str, set] = {}      # map_id -> 命中的弱类id
    for sub in ("train", "val"):
        for lf in (YDIR / "labels" / sub).glob("*.txt"):
            ids = set()
            for ln in lf.read_text(encoding="utf-8").splitlines():
                f0 = ln.split()
                if f0:
                    try:
                        ids.add(int(f0[0]))
                    except ValueError:
                        pass
            hit = ids & weak_ids
            if hit:
                cls_hit.setdefault(lf.stem.split("_")[0], set()).update(hit)
    log(f"弱类分布在 {len(cls_hit)} 张地图上")

    # 3) 这些地图的瓦片坐标（r2 计划里的 view）
    res = json.loads((ROOT / "train" / "reports" / "batch_r2_result.json")
                     .read_text(encoding="utf-8"))
    tiles: dict[str, set] = {}
    for r in res:
        if not r.get("ok") or r.get("neg"):
            continue
        v = r.get("view")
        if isinstance(v, list) and len(v) == 2 and r["id"] in cls_hit:
            tiles.setdefault(r["id"], set()).add((v[0], v[1]))
    n_maps = len(tiles)
    n_tiles = sum(len(v) for v in tiles.values())
    jobs = []
    for mid, tv in sorted(tiles.items()):
        for (x, y) in sorted(tv):
            for d in range(1, a.variants + 1):
                seed = f"{TAG}{d}{abs(hash((mid, x, y, d))) % (1 << 24):06x}"
                jobs.append((mid, x, y, d, seed, False, MAX_MOBS))
    log(f"计划：{n_maps} 张地图 · {n_tiles} 块瓦片 · {len(jobs)} 个新样本"
        f"（每瓦片 {a.variants} 版，密度 1..{a.variants}，上限 {MAX_MOBS} 只）")
    if a.dry_run or not jobs:
        return 0

    # 4) 跑（同 batch_export2 的分组并行；export_job 自带断点续跑判断）
    CHUNK = 3
    by_map: dict[tuple, list] = {}
    for i, j in enumerate(jobs):
        by_map.setdefault((j[0], i // CHUNK), []).append(j)

    results, done = [], 0
    res_file = ROOT / "train" / "reports" / f"batch_{TAG}_result.json"
    with cf.ThreadPoolExecutor(max_workers=max(1, a.workers)) as ex:
        for res_list in ex.map(lambda item: [export_job(j) for j in item[1]],
                               list(by_map.items())):
            for r in res_list:
                results.append(r)
                done += 1
                kind = "续" if r.get("resumed") else (f"d{r['density']}" if r["ok"] else "失败")
                log(f"[{done}/{len(jobs)}] {r['id']} {r['seed']} {kind} 框 {r.get('boxes')}")
                if done % 100 == 0:
                    res_file.write_text(json.dumps(results, ensure_ascii=False),
                                        encoding="utf-8")
    res_file.write_text(json.dumps(results, ensure_ascii=False), encoding="utf-8")

    ok = [r for r in results if r["ok"] and not r.get("resumed")]
    # 空图留 10% 当负样本
    empties = sorted([r for r in ok if r["boxes"] == 0], key=lambda r: r["seed"])
    for r in empties[max(1, int(0.10 * len(ok))):]:
        for k in ("image", "label"):
            try:
                Path(r[k]).unlink(missing_ok=True)
            except OSError:
                pass
        r["dropped"] = True
    kept = [r for r in ok if not r.get("dropped")]
    boxes = sum(r["boxes"] for r in kept)
    log(f"\n完成：新样本 {len(kept)} 张（空图保留 {sum(1 for r in kept if r['boxes'] == 0)}）"
        f" · 怪框 {boxes} · 失败/续跑 {len(results) - len(kept)}")

    # 5) 清中间渲染图
    n = 0
    for mid in tiles:
        for pat in ("preview_view_*.png", "preview_mobs_view_*.png"):
            for p in (ROOT / ".cache" / "map" / mid).glob(pat):
                try:
                    p.unlink()
                    n += 1
                except OSError:
                    pass
    log(f"清理中间图 {n} 张")
    return 0


if __name__ == "__main__":
    sys.exit(main())
