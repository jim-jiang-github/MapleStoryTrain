# -*- coding: utf-8 -*-
"""把 train/data/yolo_r2 打包成可发到别的电脑训练的 kit（单 zip）。

前置条件（脚本会逐一检查，不满足就退出，绝不打错包）：
  1. train/data/yolo_r2/images/ 根目录已经没有散图（说明 prep_split 已切分完）
  2. images/train、images/val、labels/train、labels/val 都存在且图-标一一对应
  3. classes.json / classes.txt 一致

打包内容（zip 内根目录 wz_mob_trainkit/）：
  dataset.yaml   —— 相对路径版（path: .，到哪台机器都能用）
  images/train|val、labels/train|val
  classes.txt / classes.json
  train.py       —— 一键训练脚本（imgsz=1366 + rect，1:1 原生分辨率）
  finetune.py    —— 服务器续训脚本（从 best.pt 出发，AdamW lr0=1e-4）
  requirements.txt、训练说明.md

用法：python train/src/package_kit.py [--out train/kit/wz_mob_trainkit.zip]
"""
from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
YDIR = ROOT / "train" / "data" / "yolo_r2"
PREFIX = "wz_mob_trainkit"

README = """# WZ 怪物检测训练包（YOLO）

## 这是什么
冒险岛（WZ 数据）合成导出的怪物检测数据集：1366×768 取景框 1:1 渲染、
只标怪物（无绳梯/地面）、469+ 类普通小怪、疏密两版随机布局（朝向/动作帧随机）。
train/val 按地图分组切分（同一张地图不跨集），稀有类强制进 train。

## 环境准备
1. Python 3.10+（3.11/3.12 均可）
2. 安装 PyTorch：有 NVIDIA 显卡去 https://pytorch.org 按显卡选 CUDA 版安装
3. `pip install -r requirements.txt`

## 开始训练
解压后进入 wz_mob_trainkit 目录：

    python train.py

默认 yolo11s + imgsz=1366 + rect + 100 epochs，权重首次运行自动下载（需联网）。
结果在 runs/detect/train/weights/best.pt。

要在已有模型上续训，把上一轮的 best.pt 放到本目录，然后：

    python finetune.py

它默认 AdamW + lr0=1e-4 + 40 epochs（optimizer="auto" 会把 lr0 丢掉，别用）。

## 关键参数（改 train.py 顶部）
- imgsz=1366：取景框原生宽度。ultralytics 会对齐到 stride 32 → 实际批次 1376 宽，
  rect 模式下不缩放不补边，图上的 35px 小怪喂进去还是 35px。
  **不要用 960**：960 是方形画布，1366×768 的信箱缩放会先把图缩到 0.70 倍，
  35px 的蜗牛只剩 25px，还白搭 44% 的灰边算力。
- rect=True：按原图宽高比组批次。代价是 ultralytics 会强制关掉 mosaic / mixup /
  cutmix（源码里 `hyp.mosaic = ... if not self.rect else 0.0`），想保留 mosaic
  就把 RECT 改 False，画布会变成 1376×1376 方形（算力约 2 倍）。
- batch=-1：自动按显存定 batch。手动设的话：8GB 显卡建议 batch=8；
  16GB 可 batch=16。
- 显存不够时的降级顺序：先降 batch，再降 imgsz 到 1152，最后才考虑换 n 模型。
- model：yolo11n（最快）/ yolo11s（推荐）/ yolo11m（更高精度，显存翻倍）。

## 训练完验证
    yolo val model=runs/detect/train/weights/best.pt data=dataset.yaml imgsz=1366 rect=True

## 导出 ONNX 给 yololab 用（别用默认 imgsz）
    yolo export model=best.pt format=onnx imgsz=1366 dynamic=True simplify=True
动态 shape 才能让 yololab 按矩形批次推理；定长 960×960 的旧 onnx 与新的
1366 训练分布在尺度上不一致，会掉点。

## 数据集速览
见 kit 根目录 dataset.yaml（类别表）与 classes.txt。
"""

TRAIN_PY = '''# -*- coding: utf-8 -*-
"""一键训练：WZ 怪物检测。改下面的超参再跑就行。"""
from ultralytics import YOLO

MODEL = "yolo11s.pt"   # yolo11n / yolo11s / yolo11m
IMGSZ = 1366           # 写整数，别写 [768,1366]（rect 下 Mosaic 会崩）。实际组出 800×1376
RECT = True            # True=按 1366×768 原比例组批（不缩放不补边，但会关掉 mosaic）
EPOCHS = 100
BATCH = -1             # -1 = 按显存自动
DEVICE = 0             # CPU 就写 "cpu"

if __name__ == "__main__":
    model = YOLO(MODEL)
    model.train(
        data="dataset.yaml",
        imgsz=IMGSZ,
        rect=RECT,
        epochs=EPOCHS,
        batch=BATCH,
        device=DEVICE,
        workers=8,
        project="runs/detect",
        name="wz_mob",
        close_mosaic=10,
        # 小目标多，copy_paste 有帮助（rect=True 时 mosaic 已被 ultralytics 强制置 0）
        copy_paste=0.1,
    )
    metrics = model.val(imgsz=IMGSZ, rect=RECT)
    print("mAP50-95:", metrics.box.map, "mAP50:", metrics.box.map50)
'''

REQUIREMENTS = "ultralytics>=8.3.0\n"


def fail(msg: str) -> None:
    print(f"✗ {msg}")
    sys.exit(1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="train/kit/wz_mob_trainkit.zip")
    a = ap.parse_args()
    out = Path(a.out) if Path(a.out).is_absolute() else ROOT / a.out

    # ---- 前置检查：宁可打包失败，绝不能打错 ----
    stray = list((YDIR / "images").glob("*.png"))
    if stray:
        fail(f"images/ 根目录还有 {len(stray)} 张散图没切分，先跑 prep_split")
    parts = {}
    for sub in ("train", "val"):
        imgs = {p.stem for p in (YDIR / "images" / sub).glob("*.png")}
        lbls = {p.stem for p in (YDIR / "labels" / sub).glob("*.txt")}
        if not imgs:
            fail(f"images/{sub} 是空的")
        miss = imgs - lbls
        if miss:
            fail(f"{sub} 有 {len(miss)} 张图缺 label（如 {sorted(miss)[0]}）")
        extra = lbls - imgs
        if extra:
            fail(f"{sub} 有 {len(extra)} 个 label 没有图")
        parts[sub] = (imgs, lbls)
    reg = json.loads((YDIR / "classes.json").read_text(encoding="utf-8"))
    names = [nm for nm, _ in sorted(reg.items(), key=lambda kv: kv[1])]
    if len(names) != len(set(names)):
        fail("classes.json 类别名有重复")
    n_cls = len(names)
    # label 里的 id 不能超界
    for sub, (imgs, _) in parts.items():
        mx = -1
        for p in (YDIR / "labels" / sub).glob("*.txt"):
            for ln in p.read_text(encoding="utf-8").splitlines():
                f = ln.split()
                if f:
                    mx = max(mx, int(f[0]))
        if mx >= n_cls:
            fail(f"{sub} 的 label 里类别 id={mx} 超出类别表（{n_cls}）")

    # ---- 生成相对路径 dataset.yaml（关键：不能用绝对路径） ----
    names_block = "\n".join(f"  {i}: {nm}" for i, nm in enumerate(names))
    yaml_text = (f"path: .\ntrain: images/train\nval: images/val\n"
                 f"nc: {n_cls}\nnames:\n{names_block}\n")

    # ---- 打包 ----
    files: list[tuple[Path, str]] = []
    for sub in ("train", "val"):
        for p in sorted((YDIR / "images" / sub).glob("*.png")):
            files.append((p, f"{PREFIX}/images/{sub}/{p.name}"))
        for p in sorted((YDIR / "labels" / sub).glob("*.txt")):
            files.append((p, f"{PREFIX}/labels/{sub}/{p.name}"))
    total_bytes = sum(p.stat().st_size for p, _ in files)
    n_img = sum(1 for _, arc in files if arc.endswith(".png"))
    print(f"待打包: {len(files)} 个文件（图 {n_img}）· 源数据 {total_bytes / 1e9:.1f} GB")

    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_STORED,
                         allowZip64=True) as z:
        z.writestr(f"{PREFIX}/dataset.yaml", yaml_text)
        z.writestr(f"{PREFIX}/classes.txt", "\n".join(names) + "\n")
        z.writestr(f"{PREFIX}/classes.json",
                   json.dumps(reg, ensure_ascii=False, indent=1))
        z.writestr(f"{PREFIX}/train.py", TRAIN_PY)
        finetune_py = Path(__file__).resolve().parent / "finetune.py"
        if not finetune_py.exists():
            fail(f"缺 {finetune_py}——kit 说明里承诺自带 finetune.py")
        z.writestr(f"{PREFIX}/finetune.py", finetune_py.read_text(encoding="utf-8"))
        z.writestr(f"{PREFIX}/requirements.txt", REQUIREMENTS)
        z.writestr(f"{PREFIX}/训练说明.md", README)
        for i, (p, arc) in enumerate(files, 1):
            z.write(p, arc)
            if i % 2000 == 0:
                print(f"  写入 {i}/{len(files)}")

    # ---- 自检：CRC 全量校验 + 条目数对账 ----
    with zipfile.ZipFile(out) as z:
        bad = z.testzip()
        if bad:
            fail(f"压缩包 CRC 校验失败: {bad}")
        entries = z.namelist()
        expect = len(files) + 7   # dataset.yaml/classes.txt/classes.json/train.py/finetune.py/requirements/训练说明
        if len(entries) != expect:
            fail(f"条目数不对: zip {len(entries)} vs 预期 {expect}")
    zs = out.stat().st_size
    print(f"✓ 打包完成: {out}（{zs / 1e9:.1f} GB，{len(entries)} 个条目，CRC 全部通过）")
    print(f"  train {len(parts['train'][0])} 张 · val {len(parts['val'][0])} 张 · {n_cls} 类")
    return 0


if __name__ == "__main__":
    sys.exit(main())
