"""round2 版终验：全新随机测试图 → best_ft 推理 → 与 GT 对比画框。

用法：python train/src/predict_test_r2.py <model.pt> [--n 6] [--conf 0.35]
产出：train/test_out/pred_*.png + match_report.json
"""
import argparse
import json
import os
import random
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PY = sys.executable
YDIR = ROOT / "train" / "data" / "yolo_r2"
OUT = Path(__file__).resolve().parents[1] / "test_out"
W, H = 1366, 768
os.environ["WZ_CLASSES_RO"] = "1"  # 只读类别表：缺类直接报错，绝不静默写坏


def run_cli(args):
    # 渲染/标注走训练仓库自己的适配层（WzToolkit 子模块只当库用）
    r = subprocess.run([PY, str(Path(__file__).resolve().parent / "yolo_export.py"), *args],
                       capture_output=True, cwd=str(ROOT))
    if r.returncode != 0:
        return None
    try:
        return json.loads(r.stdout.decode("utf-8", errors="replace"))
    except ValueError:
        return None


def iou(a, b):
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--maps", default="")
    ap.add_argument("--conf", type=float, default=0.35)
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)

    # 1) 选地图：有刷怪点、1:1 比例、尺寸够大
    if a.maps:
        picked = a.maps.split(",")
    else:
        d = run_cli(["maps", "--json"])
        cands = random.Random("testpick-r2").sample(list(d["life_maps"]), 60)
        picked = []
        for mid in cands:
            info = run_cli(["map-info", mid, "--json"])
            # --view 恒为 1:1 渲染，只需世界尺寸放得下取景框且有刷怪点
            if info and info.get("life_count"):
                w = info["world"]
                if w["w"] >= W and w["h"] >= H:
                    picked.append(mid)
            if len(picked) >= a.n:
                break
    if not picked:
        print("没找到可测试的地图")
        return 1

    # 2) 全新 seed 出图（1:1 取景框，与训练数据同规格）
    rnd = random.Random("final-test-r2")
    jobs = []
    for mid in picked[:a.n]:
        info = run_cli(["map-info", mid, "--json"])
        w = info["world"]
        x = round(rnd.uniform(w["x"], w["x"] + w["w"] - W))
        y = round(rnd.uniform(w["y"], w["y"] + w["h"] - H))
        seed = f"tst{rnd.randrange(1 << 24):06x}"
        jobs.append((mid, x, y, seed))

    tests = []
    for mid, x, y, seed in jobs:
        d = run_cli(["map-export", mid, "--preview", "--json", "--mobs",
                     f"--view={x},{y},{W},{H}", "--yolo", "--mob-only",
                     "--no-boss", f"--seed={seed}",
                     f"--yolo-dir={YDIR.relative_to(ROOT).as_posix()}"])
        if d and d.get("yolo"):
            tests.append({"id": mid, "seed": seed,
                          "image": d["yolo"]["image"], "label": d["yolo"]["labels"],
                          "n_boxes": d["yolo"]["boxes"]})
    print(f"导出 {len(tests)} 张全新测试图")
    if not tests:
        return 1

    # 3) 推理 + 匹配 + 画框
    from ultralytics import YOLO
    from PIL import Image, ImageDraw, ImageFont
    model = YOLO(a.model)
    reg = json.loads((YDIR / "classes.json").read_text(encoding="utf-8"))
    inv = {i: nm for nm, i in reg.items()}
    try:
        font = ImageFont.truetype("consola.ttf", 14)
    except Exception:
        font = ImageFont.load_default()

    report = {"images": [], "conf_thresh": a.conf}
    tot_gt = tot_pred = tot_matched = 0
    cls_stat: dict[str, dict] = {}
    for t in tests:
        img_path = Path(t["image"])
        im = Image.open(img_path).convert("RGB")
        Wpx, Hpx = im.size
        res = model.predict(str(img_path), conf=a.conf, imgsz=1366, rect=True, verbose=False)[0]
        preds = [{"cls": inv.get(int(b.cls[0]), str(int(b.cls[0]))),
                  "conf": round(float(b.conf[0]), 3),
                  "box": [float(v) for v in b.xyxy[0]]} for b in res.boxes]
        gts = []
        for ln in Path(t["label"]).read_text(encoding="utf-8").splitlines():
            p = ln.split()
            if len(p) != 5:
                continue
            c, cx, cy, w, h = (float(v) for v in p)
            gts.append({"cls": inv.get(int(c), str(int(c))),
                        "box": [(cx - w / 2) * Wpx, (cy - h / 2) * Hpx,
                                (cx + w / 2) * Wpx, (cy + h / 2) * Hpx]})
        used, matched = set(), 0
        for g in gts:
            best, bi = 0.0, -1
            for i, p in enumerate(preds):
                if i in used or p["cls"] != g["cls"]:
                    continue
                v = iou(g["box"], p["box"])
                if v > best:
                    best, bi = v, i
            if bi >= 0 and best >= 0.5:
                used.add(bi)
                matched += 1
                cls_stat.setdefault(g["cls"], {"gt": 0, "pred": 0, "hit": 0})["hit"] += 1
        for g in gts:
            cls_stat.setdefault(g["cls"], {"gt": 0, "pred": 0, "hit": 0})["gt"] += 1
        for p in preds:
            cls_stat.setdefault(p["cls"], {"gt": 0, "pred": 0, "hit": 0})["pred"] += 1
        tot_gt += len(gts)
        tot_pred += len(preds)
        tot_matched += matched

        dr = ImageDraw.Draw(im)
        # 红虚线 = 漏检 GT
        for g in gts:
            if g in [gg for gg in gts]:  # 画全部 GT 的底框太乱，只画漏检
                pass
        matched_gts = []
        for g in gts:
            hit = False
            for i, p in enumerate(preds):
                if p["cls"] == g["cls"] and iou(g["box"], p["box"]) >= 0.5:
                    hit = True
                    break
            if not hit:
                x0, y0, x1, y1 = g["box"]
                step = 6
                for xx in range(int(x0), int(x1), step * 2):
                    dr.line([xx, y0, min(xx + step, x1), y0], fill=(255, 60, 60), width=2)
                    dr.line([xx, y1, min(xx + step, x1), y1], fill=(255, 60, 60), width=2)
                for yy in range(int(y0), int(y1), step * 2):
                    dr.line([x0, yy, x0, min(yy + step, y1)], fill=(255, 60, 60), width=2)
                    dr.line([x1, yy, x1, min(yy + step, y1)], fill=(255, 60, 60), width=2)
        for p in preds:
            x0, y0, x1, y1 = p["box"]
            dr.rectangle([x0, y0, x1, y1], outline=(0, 220, 60), width=2)
            dr.text((x0 + 2, max(0, y0 - 16)), f"{p['cls']} {p['conf']:.2f}",
                    fill=(0, 220, 60), font=font)
        out_png = OUT / f"pred_{t['id']}_{t['seed']}.png"
        im.save(out_png)
        report["images"].append({"map": t["id"], "seed": t["seed"],
                                 "png": str(out_png), "gt": len(gts),
                                 "pred": len(preds), "matched": matched})
        print(f"{t['id']} seed={t['seed']}: GT {len(gts)} · 预测 {len(preds)} · 命中 {matched}")

    prec = tot_matched / tot_pred if tot_pred else 0
    rec = tot_matched / tot_gt if tot_gt else 0
    report["totals"] = {"gt": tot_gt, "pred": tot_pred, "matched": tot_matched,
                        "precision": round(prec, 4), "recall": round(rec, 4),
                        "f1": round(2 * prec * rec / (prec + rec), 4) if prec + rec else 0}
    report["per_class"] = {k: {"precision": round(v["hit"] / v["pred"], 3) if v["pred"] else None,
                               "recall": round(v["hit"] / v["gt"], 3) if v["gt"] else None,
                               **v} for k, v in cls_stat.items()}
    (OUT / "match_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n总体: 精确率 {prec:.1%} · 召回率 {rec:.1%} "
          f"（GT {tot_gt} · 预测 {tot_pred} · 命中 {tot_matched}）")
    print(f"截图与明细: {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
