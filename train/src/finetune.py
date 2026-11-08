# -*- coding: utf-8 -*-
"""服务器微调脚本（AutoDL 4090）。从上一轮 best.pt 出发继续练。

跑法（kit 解压后的根目录）：
    nohup /root/miniconda3/bin/python finetune.py > finetune.log 2>&1 &

两个已经踩过的坑，别改回去：
  1. optimizer="auto" 会把 lr0 丢掉，自己选 AdamW + 2.1e-5 这种小到学不动的学习率，
     所以必须显式 optimizer="AdamW" + lr0=1e-4；
  2. imgsz 用 1366（取景框原生宽度）。以前用 960 是方形画布，1366×768 的信箱缩放
     先把图缩到 0.70 倍，35px 的小怪只剩 25px，还白搭 44% 的灰边算力。
  3. imgsz 只能写整数，别写 [768, 1366]。虽然 check_imgsz 接受二维，但 rect=True 时
     Mosaic.__init__ 里的 `-imgsz // 2` 对 list 抛 TypeError（ultralytics 的 bug）。
     写 1366 就够：数据加载器会自己组出 800×1376（内容 1376×774 + 26px 底边），
     ratio_pad=1.0078，即 0.78% 放大，实质上就是 1:1。

rect=True 的副作用：ultralytics 在 build_transforms 里会强制把 mosaic / mixup /
cutmix 置 0（源码 `hyp.mosaic = hyp.mosaic if self.augment and not self.rect else 0.0`），
所以本脚本不再传 mosaic —— 传了也不生效。真要 mosaic 就把 RECT 改成 False，
代价是画布变成 1376×1376 方形，算力约 2 倍。
"""
from __future__ import annotations

import sys
from pathlib import Path

from ultralytics import YOLO

# ---- 超参 ----
BASE = "best.pt"        # 上一轮权重；从零练就写 yolo11s.pt
IMGSZ = 1366            # 整数！别写 [768, 1366]，rect 下会崩（见文件头坑 3）
RECT = True             # 按原比例组批：实际组出 800×1376，内容不缩放不补边
EPOCHS = 40
PATIENCE = 15
LR0 = 1e-4              # 显式给，不要靠 auto
OPTIMIZER = "AdamW"
COS_LR = True
BATCH = -1              # -1 = 按显存自动
DEVICE = ""             # "" = 自动挑 GPU
WORKERS = 8
DATA = "dataset.yaml"


def main() -> int:
    root = Path(__file__).resolve().parent
    base = root / BASE
    if BASE == "best.pt" and not base.exists():
        sys.exit(f"✗ 找不到 {base}，把上一轮的 best.pt 放到 kit 根目录，"
                 f"或把 BASE 改成 yolo11s.pt 从零练")

    model = YOLO(str(base) if base.exists() else BASE)
    res = model.train(
        data=str(root / DATA),
        imgsz=IMGSZ,
        rect=RECT,
        epochs=EPOCHS,
        patience=PATIENCE,
        batch=BATCH,
        device=DEVICE,
        workers=WORKERS,
        optimizer=OPTIMIZER,
        lr0=LR0,
        cos_lr=COS_LR,
        close_mosaic=10 if not RECT else 0,   # rect 下 mosaic 恒为 0，这里只是别报警
        copy_paste=0.1,
        project=str(root / "runs" / "detect"),
        name="wz_mob_ft",
        exist_ok=True,
        verbose=True,
    )

    best = Path(res.save_dir) / "weights" / "best.pt"
    m = YOLO(str(best))
    val = m.val(data=str(root / DATA), imgsz=IMGSZ, rect=RECT, split="val")
    print(f"\n[✓] 微调完成: {best}")
    print(f"    mAP50 = {val.box.map50:.4f}   mAP50-95 = {val.box.map:.4f}")
    print("    下一步导出（yololab 用，必须动态 shape）:")
    print(f"    yolo export model={best} format=onnx imgsz={IMGSZ} dynamic=True simplify=True")
    return 0


if __name__ == "__main__":
    sys.exit(main())
