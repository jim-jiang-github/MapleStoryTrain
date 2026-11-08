"""round2 批量导出：全部无大BOSS练级图 · 取景框 1:1 · 网格全覆盖 · 疏密两版

和 batch_export.py（round1）的区别：
  * **1:1 取景框渲染**（wztool map-export --view）：round1 是整图缩略图裁剪，
    大地图被压到 ~0.53x，和真实 1366x768 截图存在尺度域差；这里按相机 1:1 出图，
    背景带客户端视差、自动补云层，和游戏画面逐像素同尺度。
  * **只标怪物**（--mob-only）：绳/梯/地面/斜坡不标，类别用物种名（不带动作后缀）。
  * **每块瓦片两份**：density=1（怪少）+ density=3（怪多），不同 seed →
    不同的怪组合/朝向/动作帧，同一场景多套样本。
  * 再加少量纯背景负样本（--neg），压掉「树桩/装饰被误检成怪」的假阳。

覆盖策略：每张可用地图按 1366x768 网格全覆盖，相邻瓦片重合 1/3
（水平步长 911 / 垂直 512），行列末块贴地图边缘——地图每个角落都至少进一块。

用法：
  python train/src/batch_export2.py --tag r2 --maps 5        # 先小批试跑
  python train/src/batch_export2.py --tag r2                 # 全量
产出：train/data/yolo_r2/{images,labels}/ + classes.txt + dataset.yaml，
      本目录 batch_<tag>_result.json（每张图的类别/框数/路径）
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import random
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
YDIR = ROOT / "train" / "data" / "yolo_r2"
W, H = 1366, 768
STEP_X = round(W * 2 / 3)      # 911：1/3 重叠
STEP_Y = round(H * 2 / 3)      # 512
# 渲染走 WzToolkit 子模块（库方式），标注由 yolo_export.py 自己算——
# 上游 map-export 已不认 --yolo/--mob-only 这些训练参数
sys.path.insert(0, str(Path(__file__).resolve().parent))
import wztk  # noqa: E402
PY = [sys.executable, str(Path(__file__).resolve().parent / "yolo_export.py")]
CACHE_DIR = wztk.cache_dir()

from batch_export import log, tile_windows  # noqa: E402  （同一套网格逻辑）


def run_cli(args: list, timeout: int = 900) -> dict | None:
    try:
        r = subprocess.run([*PY, *args], capture_output=True, cwd=str(ROOT),
                           timeout=timeout)
    except subprocess.TimeoutExpired:
        log(f"CLI 超时({timeout}s): {' '.join(args[:2])} 放弃这块瓦片")
        return None
    if r.returncode != 0:
        tail = r.stderr[-300:].decode("utf-8", errors="replace").strip()
        if tail:
            log(f"CLI rc={r.returncode}: {' '.join(args[:2])} … {tail[-160:]}")
        return None
    try:
        return json.loads(r.stdout.decode("utf-8", errors="replace"))
    except ValueError:
        return None


def probe_view(ids: list) -> tuple[list, dict]:
    """一次 probe 全查：返回 (usable_view 的 [(id,x,y,w,h)], 统计)。"""
    r = subprocess.run([*PY, "probe", "--no-boss", "--w", str(W), "--h", str(H)],
                       input="\n".join(ids).encode(), capture_output=True,
                       cwd=str(ROOT), timeout=7200)
    if r.returncode != 0:
        log(f"probe 失败 rc={r.returncode}: {r.stderr[-300:].decode('utf-8', 'replace')}")
        return [], {}
    recs = json.loads(r.stdout.decode("utf-8", errors="replace"))
    ok = [(x["id"], x["world"]["x"], x["world"]["y"], x["world"]["w"], x["world"]["h"])
          for x in recs if x.get("usable_view")]
    stat = {"probed": len(recs), "usable": len(ok),
            "small_map": sum(1 for x in recs if not x.get("usable_view")
                             and x.get("error") is None),
            "error": sum(1 for x in recs if x.get("error"))}
    return ok, stat


def export_job(job) -> dict:
    mid, x, y, density, seed, neg, max_mobs = job
    # 断点续跑：同 tag 的任务计划是确定的（rnd 按种子生成），重跑时已出的图直接跳过
    if (YDIR / "images" / f"{mid}_{seed}.png").exists():
        return {"id": mid, "seed": seed, "ok": True, "neg": neg, "density": density,
                "view": [x, y], "boxes": -1, "states": {}, "image": "", "label": "",
                "resumed": True}
    args = ["map-export", mid, "--preview", "--json",
            f"--view={x},{y},{W},{H}", "--yolo", "--mob-only", "--no-boss",
            f"--seed={seed}", f"--yolo-dir=train/data/yolo_r2",
            f"--max-mobs={max_mobs}"]
    if not neg:
        args += ["--mobs", f"--density={density}"]
    d = run_cli(args)
    if not d or not d.get("yolo"):
        return {"id": mid, "seed": seed, "ok": False, "reason": "export 失败"}
    return {"id": mid, "seed": seed, "ok": True, "neg": neg,
            "density": density, "view": [x, y], "boxes": d["yolo"]["boxes"],
            "states": d["yolo"].get("states", {}),
            "image": str(d["yolo"]["image"]), "label": str(d["yolo"]["labels"])}


def load_map_json(mid: str) -> dict | None:
    """probe 已经把 map.json 落进 .cache/map/<id>/，直接读来筛瓦片。"""
    p = CACHE_DIR / "map" / mid / "map.json"
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def tile_value(mj: dict | None, x: float, y: float) -> tuple[float, int]:
    """瓦片里有多少地面（世界像素宽度合计）+ 命中几个刷怪点。

    纯天空/悬崖外的瓦片（没地面也没刷怪点）导出来就是空图，白白占样本。
    """
    if not mj:
        return 10 ** 9, 0                 # 读不到 map.json 就不筛
    ground = 0.0
    for f in mj.get("foothold", []):
        fy1, fy2 = float(f["y1"]), float(f["y2"])
        if max(fy1, fy2) < y - 40 or min(fy1, fy2) > y + H + 40:
            continue
        lo, hi = min(float(f["x1"]), float(f["x2"])), max(float(f["x1"]), float(f["x2"]))
        s, e = max(lo, x), min(hi, x + W)
        if e > s:
            ground += e - s
    spawns = 0
    for l in mj.get("life", []):
        if l.get("type") != "m":
            continue
        lx = float(l.get("x", 0))
        ly = (float(l.get("cy", l.get("y", 0))) if l.get("fh")
              else float(l.get("y", 0)))
        rx0, rx1 = l.get("rx0"), l.get("rx1")
        lo = float(rx0) if rx0 is not None else lx - 120
        hi = float(rx1) if rx1 is not None else lx + 120
        if hi < x - 60 or lo > x + W + 60:
            continue
        if y - 160 <= ly <= y + H + 160:
            spawns += 1
    return ground, spawns


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--maps", type=int, default=0, help="只导前 N 张可用地图（0=全部）")
    ap.add_argument("--probe-limit", type=int, default=0, help="最多探测多少张（0=全部）")
    ap.add_argument("--dense", type=int, default=3, help="怪多样本的密度倍数（默认 3）")
    ap.add_argument("--neg", type=float, default=0.08, help="额外出多少负样本（占瓦片比，默认 0.08）")
    ap.add_argument("--max-mobs", type=int, default=30,
                    help="单张画面最多几只怪（默认 30；真实的游戏画面不会挤满）")
    ap.add_argument("--workers", type=int, default=4, help="并行进程数（默认 4）")
    ap.add_argument("--dry-run", action="store_true", help="只统计规模，不导出")
    ap.add_argument("--keep-cache", action="store_true",
                    help="保留 .cache 里每块瓦片的中间渲染图（默认跑完删，全量约 13GB）")
    ap.add_argument("--tag", default="r2", help="seed 前缀 / 结果文件名")
    a = ap.parse_args()

    d = run_cli(["maps", "--json"])
    if not d:
        log("拿不到地图列表（wztool maps 失败）")
        return 1
    cands = list(d["life_maps"])
    if a.probe_limit:
        cands = cands[:a.probe_limit]
    log(f"有怪地图 {len(cands)} 张 → probe（no-boss + usable_view）…")
    usable, pstat = probe_view(cands)
    log(f"可用 {pstat.get('usable', 0)} 张（探测 {pstat.get('probed', 0)}，"
        f"太小/无小怪被排除 {pstat.get('small_map', 0)}，出错 {pstat.get('error', 0)}）")
    if a.maps:
        usable = usable[:a.maps]
    if not usable:
        log("没有可用地图，中止")
        return 1

    # 预注册类别表：把这次全部会出场的怪按 id 排序定好 id，导出时只读 classes.json。
    # （否则 8 路并发会反复创建/删除锁文件，被安全策略限流后全体卡死）
    if not a.dry_run:
        ids = [u[0] for u in usable]
        r = subprocess.run([*PY, "yolo-classes", "--yolo-dir=train/data/yolo_r2", "--no-boss", "--json"],
                           input="\n".join(ids).encode(), capture_output=True,
                           cwd=str(ROOT), timeout=1800)
        try:
            info = json.loads(r.stdout.decode("utf-8", errors="replace"))
            log(f"类别表预注册：{info['classes']} 个怪类别（{info['maps']} 张图）")
        except Exception:
            log("类别表预注册失败，导出退化为自动追加："
                + r.stderr[-200:].decode("utf-8", errors="replace"))

    rnd = random.Random(a.tag)
    jobs = []
    n_tiles = n_skip = 0
    for mid, minx, miny, bw, bh in usable:
        mj = load_map_json(mid)
        kept = []
        for x, y in tile_windows(minx, miny, bw, bh):
            ground, spawns = tile_value(mj, x, y)
            if ground < 120 and spawns == 0:
                n_skip += 1               # 纯天空/空崖：没地面也没刷怪点
                continue
            kept.append((x, y))
        n_tiles += len(kept)
        for x, y in kept:
            # 疏：density=1（自然刷怪量）；密：--dense 倍（同场景多怪）
            jobs.append((mid, x, y, 1, f"{a.tag}s{rnd.randrange(1 << 24):06x}",
                         False, a.max_mobs))
            jobs.append((mid, x, y, a.dense, f"{a.tag}d{rnd.randrange(1 << 24):06x}",
                         False, a.max_mobs))
            if rnd.random() < a.neg:
                jobs.append((mid, x, y, 1, f"{a.tag}n{rnd.randrange(1 << 24):06x}", True, 0))
    n_neg = sum(1 for j in jobs if j[5])
    log(f"{len(usable)} 张地图 · 有效瓦片 {n_tiles}（跳过纯天空 {n_skip}）· 共 {len(jobs)} 个样本"
        f"（正样本 {len(jobs) - n_neg} = 疏 {n_tiles} + 密 {n_tiles}，负样本 {n_neg}）")

    if a.dry_run:
        log("（dry-run：规划完成，未导出）")
        return 0

    # 按地图分组（渲染缓存复用），但每组分块：一张图有几十个窗口时，
    # 整张图塞给一个 worker 会把收尾拖成串行（实测最后一张图 20+ 窗口、
    # 单窗口 2 分钟）。切成 3 个一组，尾部也能并行。
    CHUNK = 3
    by_map: dict[tuple, list] = {}
    for i, j in enumerate(jobs):
        by_map.setdefault((j[0], i // CHUNK), []).append(j)
    results, done = [], 0

    def export_map(item):
        return [export_job(j) for j in item[1]]

    res_file = ROOT / "train" / "reports" / f"batch_{a.tag}_result.json"
    with cf.ThreadPoolExecutor(max_workers=max(1, a.workers)) as ex:
        for res_list in ex.map(export_map, list(by_map.items())):
            for res in res_list:
                results.append(res)
                done += 1
                if res["ok"]:
                    kind = "负样本" if res["neg"] else f"d{res['density']}"
                    log(f"[{done}/{len(jobs)}] {res['id']} {res['seed']} {kind} 框 {res['boxes']}")
                else:
                    log(f"[{done}/{len(jobs)}] {res['id']} {res['seed']} 跳过: {res['reason']}")
                if done % 100 == 0:
                    res_file.write_text(json.dumps(results, ensure_ascii=False, indent=1),
                                        encoding="utf-8")

    res_file.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    ok = [r for r in results if r["ok"]]
    resumed = [r for r in ok if r.get("resumed")]
    pos = [r for r in ok if not r["neg"] and not r.get("resumed")]
    negs = [r for r in ok if r["neg"] and not r.get("resumed")]
    # 空图（有刷怪点但一只没落进画面）留 10% 当负样本就够，其余删掉，
    # 否则负样本占比过高、正样本密度被稀释
    empties = sorted([r for r in pos if r["boxes"] == 0], key=lambda r: r["seed"])
    keep_n = max(1, int(0.10 * len(pos)))
    dropped = empties[keep_n:]
    for r in dropped:
        for k in ("image", "label"):
            try:
                Path(r[k]).unlink(missing_ok=True)
            except OSError:
                pass
        r["dropped"] = True
    if dropped:
        res_file.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    pos = [r for r in pos if not r.get("dropped")]
    boxes = sum(r["boxes"] for r in pos)
    empty_pos = sum(1 for r in pos if r["boxes"] == 0)
    states: dict[str, int] = {}
    for r in pos:
        for k, v in r["states"].items():
            states[k] = states.get(k, 0) + v
    log(f"\n完成：正样本 {len(pos)}（留作负样本的空图 {empty_pos}，剪掉 {len(dropped)}）"
        f" · 负样本 {len(negs)} · 失败 {len(results) - len(ok)}")
    log(f"怪框共 {boxes} · 每张平均 {round(boxes / max(1, len(pos)), 1)} · "
        f"动作分布 {dict(sorted(states.items()))}")
    log(f"类别表 {YDIR / 'classes.txt'} · 数据集 {YDIR / 'dataset.yaml'}")

    # 中间渲染图（每块瓦片的 1:1 底图 + 合成图）已经复制进 images/，源码可再生成，
    # 全量下来有十几 GB，默认清掉；--keep-cache 可保留（比如想复查某块瓦片）
    if not a.keep_cache:
        n, freed = 0, 0
        for mid in {u[0] for u in usable}:
            mdir = CACHE_DIR / "map" / mid
            for p in mdir.glob("preview_view_*.png"):
                try:
                    freed += p.stat().st_size
                    p.unlink()
                    n += 1
                except OSError:
                    pass
            for p in mdir.glob("preview_mobs_view_*.png"):
                try:
                    freed += p.stat().st_size
                    p.unlink()
                    n += 1
                except OSError:
                    pass
        log(f"清理中间图 {n} 张（约 {freed / 1e9:.1f} GB）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
