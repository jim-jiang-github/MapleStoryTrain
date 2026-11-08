"""把 train/data/yolo 摊平的 images/labels 收进 train/val（默认 8:2），重写 dataset.yaml。

幂等：wztool 每次导出都把图写在 images/ 根目录，这个脚本把「还没分过的」
按比例分进 train/val；已经分好的不动。负样本（label 只有绳梯或全空）一样参与。
"""
import hashlib
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
YDIR = ROOT / "train" / "data" / "yolo"


def main(ratio: float = 0.2, seed: int = 42, ydir: Path | None = None,
         group: bool = False) -> int:
    global YDIR
    if ydir is not None:
        YDIR = ydir
    img_root, lbl_root = YDIR / "images", YDIR / "labels"
    if not img_root.exists():
        print(f"还没有导出数据（{img_root} 不存在）")
        return 1
    for d in ("train", "val"):
        (img_root / d).mkdir(exist_ok=True)
        (lbl_root / d).mkdir(exist_ok=True)

    # 已分过的（在 train/val 里）就不动；根目录的散图按 stem 分流
    fresh = sorted(p for p in img_root.glob("*.png") if p.is_file())
    rnd = random.Random(seed)

    # 稀有类保护（仅 --group）：有些怪全游戏只刷在 1~2 张地图里，按地图分组切分时
    # 若那张图恰好落进 val，这个类就「训练时完全没见过」，val 里必然全漏检。
    # 做法：统计每个类出现在几张地图，凡含「≤2 张地图的类」的地图一律强制进 train，
    # 宁可验证集少几类，也不让训练集出现零样本类。
    must_train: set[str] = set()
    if group:
        cls_maps: dict[int, set] = {}
        for p in fresh:
            lbl = lbl_root / f"{p.stem}.txt"
            if not lbl.exists():
                continue
            mid = p.stem.split("_")[0]
            for ln in lbl.read_text(encoding="utf-8").splitlines():
                f = ln.split()
                if not f:
                    continue
                try:
                    c = int(f[0])
                except ValueError:
                    continue
                cls_maps.setdefault(c, set()).add(mid)
        rare = {c for c, ms in cls_maps.items() if len(ms) <= 2}
        for p in fresh:
            lbl = lbl_root / f"{p.stem}.txt"
            if not lbl.exists():
                continue
            ids = set()
            for ln in lbl.read_text(encoding="utf-8").splitlines():
                f = ln.split()
                if f:
                    try:
                        ids.add(int(f[0]))
                    except ValueError:
                        pass
            if ids & rare:
                must_train.add(p.stem.split("_")[0])
        print(f"稀有类保护: {len(rare)} 个类只出现在 ≤2 张地图 → "
              f"{len(must_train)} 张地图强制进 train")

    n_tr = n_va = 0
    for p in fresh:
        stem = p.stem
        lbl = lbl_root / f"{stem}.txt"
        if not lbl.exists():        # 图有 label 才收（label 缺失=损坏，跳过）
            print(f"跳过（无 label）: {p.name}")
            continue
        if group:
            # 按地图分组：同一张地图（含同瓦片的疏/密两版、重叠瓦片）只落一边，
            # 否则同背景几乎相同的图一半进训练一半进验证，指标虚高
            mid = stem.split("_")[0]
            v = int(hashlib.md5(f"{seed}:{mid}".encode()).hexdigest(), 16) % 1000
            dst = "train" if mid in must_train else ("val" if v < ratio * 1000 else "train")
        else:
            dst = "val" if rnd.random() < ratio else "train"
        shutil_move(p, img_root / dst / p.name)
        shutil_move(lbl, lbl_root / dst / lbl.name)
        if dst == "train":
            n_tr += 1
        else:
            n_va += 1

    # 统计
    counts = {}
    for d in ("train", "val"):
        imgs = list((img_root / d).glob("*.png"))
        lbls = list((lbl_root / d).glob("*.txt"))
        # 校验 label 与图一一对应
        missing = [p.stem for p in imgs
                   if not (lbl_root / d / f"{p.stem}.txt").exists()]
        counts[d] = {"images": len(imgs), "labels": len(lbls), "img_no_label": len(missing)}
        if missing:
            print(f"警告: {d} 有 {len(missing)} 张图缺 label（ultralytics 会当负样本）")

    # dataset.yaml
    reg = json.loads((YDIR / "classes.json").read_text(encoding="utf-8"))
    inv = sorted(reg.items(), key=lambda kv: kv[1])
    names = "\n".join(f"  {i}: {nm}" for nm, i in inv)
    (YDIR / "dataset.yaml").write_text(
        f"path: {YDIR.resolve().as_posix()}\ntrain: images/train\nval: images/val\n"
        f"names:\n{names}\n", encoding="utf-8")
    print(f"切分完成: train {counts['train']['images']} 张 · val {counts['val']['images']} 张"
          f"（本次新增 train {n_tr} / val {n_va}）· 类别 {len(inv)}")
    return 0


def shutil_move(src: Path, dst: Path):
    import shutil
    dst = Path(dst)
    if dst.exists():
        dst.unlink()
    shutil.move(str(src), str(dst))


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    d = ""
    for a in sys.argv[1:]:
        if a.startswith("--dir="):
            d = a.split("=", 1)[1]
    ratio = float(args[0]) if args else 0.2
    target = (Path(d) if Path(d).is_absolute() else ROOT / d) if d else None
    sys.exit(main(ratio, ydir=target, group=("--group" in sys.argv)))
