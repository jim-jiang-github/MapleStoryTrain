#!/usr/bin/env python3
"""gen_data.py —— 训练图生成 · 统一入口（一个脚本管所有出图场景）。

底层引擎还是 src/ 里的各专业脚本，这里只是把常用场景收拢成一个命令，
不用记「哪个脚本配哪组参数」。

用法（在项目根目录）：
    python train\\src\\gen_data.py sample --maps 5        # 小批试跑（先看规模对不对）
    python train\\src\\gen_data.py sample --maps 5 --dry-run
    python train\\src\\gen_data.py full                   # 全量导出（疏+密两版，约 1 小时）
    python train\\src\\gen_data.py full --workers 8       # 并行进程数
    python train\\src\\gen_data.py weak                   # 弱类定向补数据（AP<0.85 的类加量）
    python train\\src\\gen_data.py weak --variants 3 --dry-run
    python train\\src\\gen_data.py noboss                 # 重导混入 BOSS 的样本（一次性修复，归档用）
    python train\\src\\gen_data.py validate               # 数据全量体检
    python train\\src\\gen_data.py review                 # 生成自检页（review.html）
    python train\\src\\gen_data.py review --sample 400 --per-map 8

产出位置：
    train/data/yolo_r2/     训练图 + 标注 + classes + dataset.yaml
    train/reports/          batch_*_result.json 等结果表
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent          # train/src
PY = sys.executable

# 子命令 -> (引擎脚本, 透传的固定参数, 说明)
SUBS = {
    "sample":  ("batch_export2.py", ["--maps"], "小批试跑（--maps N 只导前 N 张图）"),
    "full":    ("batch_export2.py", [],         "全量导出：全部练级图，疏(d1)+密(d3)两版"),
    "weak":    ("boost_weak_r3.py", [],         "弱类定向补数据（AP 低的类所在地图加量）"),
    "noboss":  ("archive/reexport_noboss.py", [], "重导混入 BOSS 的样本（一次性修复）"),
    "validate": ("validate_r2.py", [],          "数据全量体检（对应/坐标/类别/PNG 完整性）"),
    "review":  ("review_gallery.py", [],        "生成自检页 review.html"),
}

HELP = __doc__ + "\n子命令：\n" + "\n".join(
    f"  {k:<10} {v[2]}" for k, v in SUBS.items())


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help", "help"):
        print(HELP)
        return 0
    sub = sys.argv[1]
    extra = sys.argv[2:]
    if sub not in SUBS:
        print(f"[x] 未知子命令: {sub}\n")
        print(HELP)
        return 2
    script, fixed, _ = SUBS[sub]
    target = HERE / script
    if not target.exists():
        print(f"[x] 找不到引擎脚本: {target}")
        return 2
    cmd = [PY, str(target), *fixed, *extra]
    print(f"[*] {' '.join(cmd)}", flush=True)
    return subprocess.run(cmd).returncode


if __name__ == "__main__":
    sys.exit(main())
