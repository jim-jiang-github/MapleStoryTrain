"""压缩类别表：只保留「标注里真的出现过」的类，并重新编号。

为什么需要：批量导出前会把这批地图 life 里所有会出场的怪都预注册进 classes.json
（为了并发导出时类别 id 稳定、只读不写）。但有些怪的刷怪点可能一块瓦片都没命中，
预注册了却一张样本都没有——训练时这些空类是 mAP 的分母，纯拖低指标。

做法：扫一遍 labels/ 统计实际用到的 id → 按旧 id 顺序重排成 0..N-1 →
原地改写所有 label 文件的第一列 + 重写 classes.json / classes.txt / dataset.yaml。
幂等：没有空类时什么都不改。

用法：python train/src/compact_classes.py --dir train/data/yolo_r2
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="train/data/yolo_r2")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    ydir = Path(a.dir) if Path(a.dir).is_absolute() else ROOT / a.dir
    reg_file = ydir / "classes.json"
    reg: dict[str, int] = json.loads(reg_file.read_text(encoding="utf-8"))
    inv = {i: nm for nm, i in reg.items()}

    labels = sorted((ydir / "labels").rglob("*.txt"))  # 兼容平铺/切分布局
    if not labels:
        print("没有 label 文件")
        return 1
    used = Counter()
    lines_of = {}
    for p in labels:
        rows = []
        for ln in p.read_text(encoding="utf-8").splitlines():
            f = ln.split()
            if not f:
                continue
            try:
                cid = int(f[0])
            except ValueError:
                continue
            used[cid] += 1
            rows.append((cid, f[1:]))
        lines_of[p] = rows

    keep = [cid for cid in sorted(inv) if used.get(cid, 0) > 0]
    dropped = sorted(set(inv) - set(keep))
    remap = {old: new for new, old in enumerate(keep)}
    new_reg = {inv[old]: new for old, new in remap.items()}
    n_relab = sum(1 for p, rows in lines_of.items() if any(r[0] not in remap for r in rows)) \
        if dropped else 0

    print(f"类别：注册 {len(inv)} · 实际用到 {len(keep)} · 空类 {len(dropped)}")
    if dropped:
        names = ", ".join(inv[d] for d in dropped[:20])
        print(f"  空类示例：{names}{' …' if len(dropped) > 20 else ''}")
    if not dropped:
        print("没有空类，无需改动")
        return 0
    if a.dry_run:
        print("（dry-run，未改动）")
        return 0

    # 改写 label（第一列换成新 id）——先写临时文件再原子替换，避免半截文件
    for p, rows in lines_of.items():
        out = "\n".join(f"{remap[cid]} " + " ".join(rest) for cid, rest in rows if cid in remap)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(out + ("\n" if out else ""), encoding="utf-8")
        tmp.replace(p)

    reg_file.write_text(json.dumps(new_reg, ensure_ascii=False, indent=1), encoding="utf-8")
    inv2 = sorted(new_reg.items(), key=lambda kv: kv[1])
    (ydir / "classes.txt").write_text(
        "\n".join(nm for nm, _ in inv2) + "\n", encoding="utf-8")
    names = "\n".join(f"  {i}: {nm}" for nm, i in inv2)
    (ydir / "dataset.yaml").write_text(
        f"path: {ydir.resolve().as_posix()}\ntrain: images\nval: images\n"
        f"names:\n{names}\n", encoding="utf-8")
    print(f"已压缩为 {len(inv2)} 类，重写 {len(lines_of)} 个 label"
          f"（其中 {n_relab} 个含被删类别）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
