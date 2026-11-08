# -*- coding: utf-8 -*-
"""把训练好的 best.pt 导出成 yololab 用的 onnx。

要点（2026-11 起）：
  · imgsz 必须是取景框原生宽度 1366，对齐后 1376 —— 与训练 rect 批次一致；
  · 两种导出模式，按部署形态选：
      --static  固定 shape（1×3×800×1376）。ONNX Runtime 能做完整算子融合与
                内存规划，最快；代价是只认这一个尺寸，下拉框换分辨率会直接报错。
      默认      动态 shape。任意尺寸都能跑，但每个新 shape 都要重新做一遍布局
                优化，实测比静态慢约 5%，且首帧有明显卡顿。
    取景框固定 1366×768 时用 --static；要留多分辨率切换能力就用默认。
  · 导出后自动跑一遍 onnxruntime 冒烟，把实际喂入张量打出来，方便肉眼核对。

用法：
    python train/src/export_onnx.py                      # 动态 shape
    python train/src/export_onnx.py --static             # 固定 800×1376，最快
    python train/src/export_onnx.py --static --imgsz 1152
产出：
    yololab/models/best_ft.onnx  + 最好的同名 .pt 副本
"""
from __future__ import annotations

import argparse
import math
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "yololab" / "models"


def pick_weights(explicit: str | None) -> Path:
    if explicit:
        p = Path(explicit)
        if not p.is_absolute():
            p = ROOT / p
        if not p.exists():
            sys.exit(f"✗ 找不到权重: {p}")
        return p
    cands = sorted((ROOT / "train" / "runs").glob("*/weights/best.pt"),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    if not cands:
        sys.exit("✗ train/runs/*/weights/ 下没有 best.pt，先训完或手动 --weights")
    return cands[0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", help="默认取 train/runs 下最新的 best.pt")
    ap.add_argument("--imgsz", type=int, default=1366,
                    help="取景框原生宽度；不要传 960（会把小怪缩到 0.70 倍）")
    ap.add_argument("--out-dir", default=str(MODELS))
    ap.add_argument("--no-simplify", action="store_true")
    ap.add_argument("--static", action="store_true",
                    help="固定输入 shape 导出（更快，但只认这一个尺寸，换分辨率必须重导）")
    ap.add_argument("--src-h", type=int, default=768,
                    help="取景框原生高度，仅 --static 用于推算短边对齐值（默认 768）")
    a = ap.parse_args()

    w = pick_weights(a.weights)
    out_dir = Path(a.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    from ultralytics import YOLO
    from ultralytics.utils.checks import check_imgsz

    aligned = check_imgsz(a.imgsz, stride=32, min_dim=2)
    print(f"[*] 权重: {w.relative_to(ROOT)}")
    print(f"[*] imgsz={a.imgsz} → 对齐 {aligned[0]}×{aligned[1]}（训练/推理保持一致）")

    # 静态导出要还原 rect 组批的真实张量：长边 = imgsz 对齐值，短边按同一缩放比推出来再向上对齐 stride
    if a.static:
        scale = aligned[1] / a.imgsz
        h_static = int(math.ceil(a.src_h * scale / 32) * 32)
        export_imgsz = [h_static, aligned[1]]
        print(f"[*] 静态 shape {export_imgsz[1]}×{export_imgsz[0]}"
              f"（源图 {a.imgsz}×{a.src_h} × {scale:.4f} → 短边 {a.src_h * scale:.0f} "
              f"向上对齐 {h_static}）")
    else:
        export_imgsz = a.imgsz
        print("[*] 动态 shape：任意尺寸可跑，换分辨率不必重导")

    model = YOLO(str(w))
    t0 = time.perf_counter()
    exported = Path(model.export(
        format="onnx",
        imgsz=export_imgsz,
        dynamic=not a.static,       # 静态：ONNX Runtime 可做完整融合，最快；动态：任意尺寸可跑
        simplify=not a.no_simplify,
        opset=12,
    ))
    print(f"[✓] 导出完成 {exported.name}（{exported.stat().st_size / 1e6:.1f} MB，"
          f"{time.perf_counter() - t0:.1f}s）")

    dst_onnx = out_dir / "best_ft.onnx"
    dst_pt = out_dir / "best_ft.pt"
    if exported.resolve() != dst_onnx.resolve():
        shutil.copy2(exported, dst_onnx)
        try:  # 别把中间产物 best.onnx 留在 runs/ 里
            if (ROOT / "train") in exported.resolve().parents:
                exported.unlink()
        except OSError:
            pass
    shutil.copy2(w, dst_pt)
    print(f"[✓] 已部署: {dst_onnx.relative_to(ROOT)} / {dst_pt.relative_to(ROOT)}")

    # ---- 冒烟：确认动态 shape + rect 推理的实际张量 ----
    try:
        import numpy as np
        import onnxruntime as ort
    except ImportError:
        print("[!] 没装 onnxruntime，跳过冒烟（pip install onnxruntime）")
        return 0

    sess = ort.InferenceSession(str(dst_onnx), providers=["CPUExecutionProvider"])
    inp, outp = sess.get_inputs()[0], sess.get_outputs()[0]
    print(f"[*] onnx 输入: {inp.shape}  输出: {outp.shape}")
    if a.static:
        print(f"[*] 静态 shape 已锁定 {inp.shape}；yololab 下拉框切到其他分辨率会直接报错，属预期行为")
    elif all(isinstance(d, int) for d in inp.shape[1:]):
        print("[!] 警告：要求动态导出却得到定长 shape，rect 推理会失效，检查 dynamic 是否被覆盖")

    dummy = np.zeros((768, 1366, 3), np.uint8)
    m = YOLO(str(dst_onnx))
    m.predict(dummy, imgsz=a.imgsz, rect=True, verbose=False)
    ab = m.predictor.model
    seen: list[tuple] = []
    orig_run = ab.session.run

    def patched(output_names, input_feed):
        seen.append(next(iter(input_feed.values())).shape)
        return orig_run(output_names, input_feed)

    ab.session.run = patched
    t0 = time.perf_counter()
    m.predict(dummy, imgsz=a.imgsz, rect=True, verbose=False)
    ms = (time.perf_counter() - t0) * 1000
    print(f"[✓] 1366×768 截图 → 实际喂入张量 {seen[-1]}（{ms:.0f} ms/帧 CPU）")
    if a.static:
        want = (1, 3, h_static, aligned[1])
        if tuple(seen[-1]) == want:
            print(f"    ✓ 与静态 shape 完全一致 {want}")
        else:
            print(f"    ✗ 形状不符！期望 {want}，实际 {tuple(seen[-1])}")
            return 1
    else:
        print(f"    宽应等于 {aligned[1]}，高应贴近 768（stride 对齐后会略大）："
              f"{seen[-1][2]}×{seen[-1][3]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
