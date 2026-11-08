# -*- coding: utf-8 -*-
"""round2 数据全量体检：图片-标注对应 / 框坐标合法性 / 类别表四方一致 / PNG 完整性。

用法: python train/src/validate_r2.py [--dir train/data/yolo_r2] [--skip-verify]
"""
import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
YDIR = ROOT / "train" / "data" / "yolo_r2"
ERR = []
WARN = []


def err(msg):
    ERR.append(msg)


def warn(msg):
    WARN.append(msg)


def main() -> int:
    t0 = time.time()
    # 兼容平铺（导出后）与 images/{train,val}（切分后）两种布局
    imgs = {p.stem: p for p in (YDIR / "images").rglob("*.png")}
    lbls = {p.stem: p for p in (YDIR / "labels").rglob("*.txt")}
    print(f"图片 {len(imgs)} · 标签 {len(lbls)}")

    # ① 一一对应
    for s in sorted(set(imgs) - set(lbls)):
        err(f"图片无标签: {s}")
    for s in sorted(set(lbls) - set(imgs)):
        err(f"标签无图片: {s}")
    print(f"① 对应性: 图片缺失标签 {len(set(imgs)-set(lbls))} · 标签缺失图片 {len(set(lbls)-set(imgs))}")

    # ② 类别表四方一致
    reg = json.loads((YDIR / "classes.json").read_text(encoding="utf-8"))
    txt = [l for l in (YDIR / "classes.txt").read_text(encoding="utf-8").splitlines() if l.strip()]
    yaml_txt = (YDIR / "dataset.yaml").read_text(encoding="utf-8")
    yaml_names = {}
    in_names = False
    for ln in yaml_txt.splitlines():
        if ln.startswith("names:"):
            in_names = True
        elif in_names and ln.startswith("  "):
            i, nm = ln.strip().split(":", 1)
            yaml_names[int(i)] = nm.strip()
    inv_txt = {i: nm for i, nm in enumerate(txt)}
    if reg != {nm: i for i, nm in inv_txt.items()}:
        bad = [nm for nm, i in reg.items() if inv_txt.get(i) != nm][:3]
        err(f"classes.json 与 classes.txt 不一致（例: {bad}）")
    if yaml_names != inv_txt:
        n_bad = sum(1 for i in set(yaml_names) | set(inv_txt)
                    if yaml_names.get(i) != inv_txt.get(i))
        err(f"dataset.yaml names 与 classes.txt 不一致（{n_bad} 处）")
    if txt and txt[-1].splitlines():
        pass
    if any(" " in nm or "\t" in nm for nm in txt):
        bad = [nm for nm in txt if " " in nm]
        err(f"类别名含空格（YOLO 解析会炸）: {bad[:5]} 共{len(bad)}")
    nc = len(txt)
    print(f"② 类别表: json {len(reg)} · txt {nc} · yaml {len(yaml_names)} · "
          f"{'一致' if not ERR else '有问题'}")

    # ③ 标注逐行校验（全量）
    used_cls = Counter()
    bad_lines = 0
    tiny, huge, offscr = 0, 0, 0
    empty_lbl_pos, pos_cnt, neg_cnt = 0, 0, 0
    for s, p in lbls.items():
        rows = [l.split() for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
        if not rows:
            continue  # 空标签=负样本，合法
        # 负样本种子是 r2n*（--neg 出的纯背景），正样本 r2s*/r2d*，补导的是 *nb
        is_neg = "_r2n" in s
        if not is_neg:
            pos_cnt += 1
        for r in rows:
            if len(r) != 5:
                bad_lines += 1
                if bad_lines <= 5:
                    err(f"{s}: 字段数 {len(r)} ≠ 5")
                continue
            try:
                c, xc, yc, w, h = int(r[0]), *map(float, r[1:])
            except ValueError:
                bad_lines += 1
                err(f"{s}: 数值解析失败 {r}")
                continue
            if not (0 <= c < nc):
                bad_lines += 1
                err(f"{s}: 类别 id {c} 超界 (nc={nc})")
                continue
            used_cls[c] += 1
            x0, y0, x1, y1 = xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2
            if x1 - x0 <= 0 or y1 - y0 <= 0:
                bad_lines += 1
                continue
            if x0 < -0.001 or y0 < -0.001 or x1 > 1.001 or y1 > 1.001:
                offscr += 1
                if offscr <= 5:
                    err(f"{s}: 框越界 ({x0:.3f},{y0:.3f})-({x1:.3f},{y1:.3f}) cls {c}")
            if w * 1366 < 8 or h * 768 < 8:
                tiny += 1
            if w > 0.95 and h > 0.95:
                huge += 1
    neg_cnt = sum(1 for s in lbls if "_r2n" in s)
    # 正样本但空标签（怪没落进画面的瓦片）单独统计
    empty_all = [s for s, p in lbls.items() if not p.read_text(encoding="utf-8").strip()]
    empty_pos = [s for s in empty_all if "_r2n" not in s]
    total_boxes = sum(used_cls.values())
    print(f"③ 标注: 框 {total_boxes} · 坏行 {bad_lines} · 越界 {offscr} · "
          f"过小(<8px) {tiny} · 巨框 {huge}")
    print(f"   正样本 {pos_cnt} · 空标签共 {len(empty_all)}（其中正样本种子但空 {len(empty_pos)}）")
    if len(empty_pos) > 50:
        warn(f"空标签的正样本偏多: {len(empty_pos)}")
    unused = [i for i in range(nc) if i not in used_cls]
    if unused:
        warn(f"注册但零样本的类别: {len(unused)} 个 -> {[txt[i] for i in unused[:5]]}")

    # ④ PNG 完整性 + 尺寸（全量 verify）
    if "--skip-verify" not in sys.argv:
        from PIL import Image
        bad_png, wrong_size, done = 0, 0, 0
        t1 = time.time()
        for s, p in sorted(imgs.items()):
            try:
                with Image.open(p) as im:
                    im.verify()          # 校验 CRC/截断
                if done < 3 or done % 2000 == 0:
                    with Image.open(p) as im2:
                        if im2.size != (1366, 768):
                            wrong_size += 1
                            err(f"{s}: 尺寸 {im2.size} ≠ (1366,768)")
                else:
                    with Image.open(p) as im2:
                        if im2.size != (1366, 768):
                            wrong_size += 1
                            err(f"{s}: 尺寸 {im2.size}")
            except Exception as e:
                bad_png += 1
                err(f"{s}: PNG 损坏 ({type(e).__name__})")
            done += 1
            if done % 2000 == 0:
                print(f"   ...已校验 {done}/{len(imgs)} ({time.time()-t1:.0f}s)")
        print(f"④ PNG: 校验 {done} · 损坏 {bad_png} · 尺寸异常 {wrong_size}")

    # ⑤ 标注里的怪必须真的刷在这张地图里（抓类别 id 漂移/串号）
    #    类别名格式 "<mobid>_<名字>"，取 mobid 跟地图 life 比；mob 有 link 时
    #    贴图可能来自 link 目标，链上所有 id 都算合法。
    drift = []
    try:
        import wztk
        Wz = wztk.wztool()
        ws = wztk.workspace()
        meta = {m["id"]: m for m in Wz.mob_meta(ws, ws / "Mob.nx")}
        mob_of_cls = [nm.split("_", 1)[0].lstrip("0") for nm in txt]
        life_cache: dict[str, set | None] = {}

        def ok_ids(mid: str):
            if mid in life_cache:
                return life_cache[mid]
            p = wztk.cache_dir() / "map" / mid / "map.json"
            if not p.exists():
                life_cache[mid] = None
                return None
            ids = set()
            try:
                mj = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                life_cache[mid] = None
                return None
            for l in mj.get("life", []):
                if l.get("type") != "m":
                    continue
                k = str(l.get("id", "")).lstrip("0")
                seen = set()
                while k and k not in seen:
                    seen.add(k)
                    ids.add(k)
                    k = str(meta.get(k, {}).get("link") or "").lstrip("0")
            life_cache[mid] = ids
            return ids

        for s, p in lbls.items():
            ok = ok_ids(s.split("_")[0])
            if ok is None:
                continue
            ids = set()
            for ln in p.read_text(encoding="utf-8").splitlines():
                f = ln.split()
                if f:
                    try:
                        ids.add(int(f[0]))
                    except ValueError:
                        pass
            wrong = {txt[i] for i in ids if mob_of_cls[i] not in ok}
            if wrong:
                drift.append((s, sorted(wrong)[:2]))
        if drift:
            err(f"标注类别与地图刷怪表不符（id 漂移）: {len(drift)} 张，例 {drift[:3]}")
        print(f"⑤ 刷怪表一致性: 不符 {len(drift)} 张")
    except Exception as e:      # 校验本身失败也要报出来，不能静默
        warn(f"刷怪表校验无法执行: {type(e).__name__}: {e}")

    print(f"\n===== 结论（{time.time()-t0:.0f}s）=====")
    if ERR:
        print(f"✗ 错误 {len(ERR)} 条:")
        for m in ERR[:30]:
            print("  -", m)
        return 1
    print("✓ 全部通过" + (f"（警告 {len(WARN)} 条: " + "; ".join(WARN[:5]) + "）" if WARN else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
