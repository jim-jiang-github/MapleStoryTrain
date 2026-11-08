"""把导出的地图按「相机取景框」重建成一张和游戏里一模一样的画面。

和 `wztool map-export --region` 的区别：
  * region 是「整图预览的裁剪」：预览按整图缩放渲染，取景框会被缩到 ~0.53x。
    这里直接按 1:1 渲染 1366x768，和客户端窗口一个像素对一个像素。
  * 背景（back）是屏幕坐标 + 视差百分比画的，必须知道相机位置才能摆对；
    region 裁剪沿用的是「屏幕中心 = 画面中心」的近似，远景会整块错位。
  * 游戏里 info/cloud=1 的图，引擎会额外画云层（Map.nx 里没有 cloud.img），
    导出的背景铺不满时天空会露出纯蓝底砖（Back_dryRock_back_0 那种 256x256 纯色）。
    这里用地图自带的横向云带替换那块纯色底砖，天空才像游戏里那样是云。

用法：
  python yololab/scene.py 106000000 --at=-625,705
  python yololab/scene.py 106000000 --at=-625,705 --locator --shot 截图.jpg
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))


def _load_render_module():
    """vendor/render_map.py 是 dataclass 模块，必须先塞进 sys.modules 再 exec。"""
    p = ROOT / "vendor" / "render_map.py"
    spec = importlib.util.spec_from_file_location("_scene_render_map", p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def find_map_dir(map_id: str) -> Path:
    """先找当前 worktree，再回溯主仓库（worktree 里没有 exports/ 时很常见）。"""
    cands = [ROOT / "exports" / "maps" / map_id]
    gitfile = ROOT / ".git"
    if gitfile.is_file():
        txt = gitfile.read_text(encoding="utf-8", errors="ignore").strip()
        if txt.startswith("gitdir:"):
            # worktree 的 .git 文件指向 <主仓库>/.git/worktrees/<name>，往上三层才是主仓库根
            wtdir = Path(txt.split(":", 1)[1].strip())
            cands.append(wtdir.parent.parent.parent / "exports" / "maps" / map_id)
    for c in cands:
        if (c / "map.json").is_file():
            return c
    raise SystemExit(f"找不到地图导出目录：{map_id}（先跑 wztool map-export {map_id}）")


def _flat_texture(path: Path) -> bool:
    """是不是一整块纯色（天空底砖）：缩到 8x8 后几乎没变化。"""
    try:
        im = Image.open(path).convert("RGB").resize((8, 8))
    except Exception:
        return False
    a = np.asarray(im).astype(float)
    return float(a.std(axis=(0, 1)).max()) < 8.0


def cloud_backs(map_json: dict, img_dir: Path) -> list[dict]:
    """挑一条最宽的横向云带，改成双向平铺，用来顶掉纯色天空底砖。"""
    strips = [b for b in map_json.get("back", [])
              if int(b.get("front", 0)) == 0 and int(b.get("type", 0)) in (1, 4)
              and b.get("img") and (img_dir / b["img"]).is_file()]
    if not strips:
        return []
    strips.sort(key=lambda b: -(b.get("cx") or 0))
    b = dict(strips[0])
    tex = Image.open(img_dir / b["img"])
    b["type"] = 3                      # 双向平铺
    b["cx"], b["cy"] = tex.width, tex.height
    b["a"] = 255
    return [b]


def render_scene(map_id: str, at: tuple[float, float], size: tuple[int, int],
                 fill: bool = True) -> Image.Image:
    rm = _load_render_module()
    mdir = find_map_dir(map_id)
    mj = json.loads((mdir / "map.json").read_text(encoding="utf-8"))
    img_dir = mdir / "img"
    if fill:
        clouds = cloud_backs(mj, img_dir)
        if clouds:
            # 纯色底砖（天空蓝）换成云带；其余背景原样保留
            rest = [b for b in mj.get("back", [])
                    if not (int(b.get("type", 0)) == 3
                            and b.get("img") and (img_dir / b["img"]).is_file()
                            and _flat_texture(img_dir / b["img"]))]
            mj = {**mj, "back": clouds + rest}
    x, y = at
    w, h = size
    im = rm.render_map(mj, [str(img_dir)],
                       rm.RenderOpts(scale=1.0, max_width=10 ** 6, view=(x, y, w, h)))
    return im.convert("RGB")


def locator(map_id: str, at: tuple[float, float], size: tuple[int, int], out: Path) -> None:
    """整图预览上画一个红框，标出这张截图对应地图的哪一块。"""
    mdir = find_map_dir(map_id)
    prev = Image.open(mdir / "preview.png").convert("RGB")
    mj = json.loads((mdir / "map.json").read_text(encoding="utf-8"))
    xs, ys = [], []
    for L in mj.get("layers", []):
        for it in list(L.get("tile", [])) + list(L.get("obj", [])):
            xs += [float(it["x"]) - it.get("ox", 0), float(it["x"]) - it.get("ox", 0) + it.get("w", 0)]
            ys += [float(it["y"]) - it.get("oy", 0), float(it["y"]) - it.get("oy", 0) + it.get("h", 0)]
    minx, miny = min(xs) - 40, min(ys) - 40
    bw = max(xs) - minx + 40
    s = prev.width / bw
    x, y = at
    w, h = size
    box = [round((x - minx) * s), round((y - miny) * s),
           round((x + w - minx) * s), round((y + h - miny) * s)]
    im = prev.copy()
    d = ImageDraw.Draw(im, "RGBA")
    d.rectangle(box, outline=(255, 62, 62, 255), width=5)
    try:
        from PIL import ImageFont
        font = ImageFont.truetype("msyh.ttc", 26)
    except Exception:
        font = None
    tx = min(max(box[0], 8), im.width - 340)
    ty = max(box[1] - 42, 8)
    d.rectangle([tx, ty, tx + 330, ty + 38], fill=(0, 0, 0, 185))
    d.text((tx + 8, ty + 3), "你截图的位置（1366×768）", font=font, fill=(255, 90, 90, 255))
    out.parent.mkdir(parents=True, exist_ok=True)
    im.save(out)
    print(f"定位图: {out}")


def _label(draw: ImageDraw.ImageDraw, xy, text: str, font):
    x, y = xy
    bb = draw.textbbox((x, y), text, font=font)
    draw.rectangle([bb[0] - 8, bb[1] - 5, bb[2] + 8, bb[3] + 5], fill=(0, 0, 0, 190))
    draw.text((x, y), text, font=font, fill=(255, 255, 255, 255))


def compare(shot_path: str, sim: Image.Image, out_path: Path) -> None:
    """上：真实截图；中：同位置渲染；下：50% 叠加（看地形是否对得上）。"""
    from PIL import ImageFont
    shot = Image.open(shot_path).convert("RGB")
    if shot.size != sim.size:
        shot = shot.resize(sim.size, Image.LANCZOS)
    w, h = sim.size
    blend = Image.blend(shot, sim, 0.5)
    sheet = Image.new("RGB", (w, h * 3 + 16), (18, 18, 18))
    for i, im in enumerate((shot, sim, blend)):
        sheet.paste(im, (0, i * (h + 8)))
    d = ImageDraw.Draw(sheet, "RGBA")
    try:
        font = ImageFont.truetype("msyh.ttc", 22)
    except Exception:
        font = None
    _label(d, (16, 16), "真实游戏截图", font)
    _label(d, (16, h + 24), "资源重建（1:1，同一相机位置）", font)
    _label(d, (16, 2 * h + 32), "50% 叠加：地形贴合度", font)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path, quality=90)
    print(f"对比图: {out_path}")


def main() -> int:
    ap = argparse.ArgumentParser(description="按相机取景框把导出的地图重建成游戏画面")
    ap.add_argument("map_id", help="地图 id，如 106000000")
    ap.add_argument("--at", required=True, metavar="X,Y",
                    help="取景框左上角的世界坐标（坐标是负数时写成 --at=-625,705）")
    ap.add_argument("--size", default="1366x768", help="画面尺寸，默认 1366x768")
    ap.add_argument("--out", default="", help="输出 PNG（默认 yololab/out/<id>_<x>_<y>.png）")
    ap.add_argument("--shot", default="", help="给一张真实截图，额外生成三格对比图")
    ap.add_argument("--locator", action="store_true", help="额外生成整图预览 + 取景框红框")
    ap.add_argument("--no-sky", action="store_true", help="不补云层（保留原始蓝底）")
    a = ap.parse_args()
    try:
        x, y = (float(v) for v in a.at.replace("，", ",").split(","))
        w, h = (int(v) for v in a.size.lower().split("x"))
    except ValueError:
        print("参数格式：--at X,Y  --size 1366x768", file=sys.stderr)
        return 2
    sim = render_scene(a.map_id, (x, y), (w, h), fill=not a.no_sky)
    out = Path(a.out) if a.out else (ROOT / "yololab" / "out" / f"{a.map_id}_{int(x)}_{int(y)}.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    sim.save(out)
    print(f"画面: {out}  ({sim.width}x{sim.height})")
    if a.locator:
        locator(a.map_id, (x, y), (w, h), out.with_name(out.stem + "_where.png"))
    if a.shot:
        compare(a.shot, sim, out.with_name(out.stem + "_compare.jpg"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
