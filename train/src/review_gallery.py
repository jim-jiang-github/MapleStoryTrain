"""生成导出结果的自检页。

全量模式（默认）：把 <dir>/images 下所有图片渲染成带框缩略图，存到
<dir>/review/t/*.jpg，再生成 index.html（元数据内嵌、懒加载、分页、筛选），
双击 index.html 就能翻完整个数据集，点缩略图可打开对应原图。

抽样模式（--sample N）：旧逻辑，单文件 HTML，图片 base64 内嵌，
适合快速抽查。

用法：
  python train/src/review_gallery.py --dir train/data/yolo_r2            # 全量
  python train/src/review_gallery.py --dir train/data/yolo_r2 --sample 60 # 抽查
产出：<dir>/review/index.html（全量） / <dir>/review.html（抽样）
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import random
import re
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
PALETTE = [(216, 90, 48), (29, 158, 117), (83, 74, 183), (186, 117, 23),
           (212, 83, 126), (24, 95, 165), (99, 153, 34), (163, 45, 45)]


def load_classes(ydir: Path) -> list[str]:
    f = ydir / "classes.txt"
    return f.read_text(encoding="utf-8").splitlines() if f.exists() else []


def load_labels(p: Path) -> list[tuple[int, float, float, float, float]]:
    out = []
    if not p.exists():
        return out
    for ln in p.read_text(encoding="utf-8").splitlines():
        f = ln.split()
        if len(f) >= 5:
            try:
                out.append((int(f[0]), *map(float, f[1:5])))
            except ValueError:
                continue
    return out


def font(size: int):
    for name in ("msyh.ttc", "msyhl.ttc", "simhei.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            continue
    return ImageFont.load_default()


def render_thumb(im, labels, classes) -> Image.Image:
    """在已打开的图上画框 + 类别名（就地修改）。"""
    d = ImageDraw.Draw(im, "RGBA")
    f = font(13)
    for cid, cx, cy, bw, bh in labels:
        x0 = (cx - bw / 2) * im.width
        y0 = (cy - bh / 2) * im.height
        x1 = (cx + bw / 2) * im.width
        y1 = (cy + bh / 2) * im.height
        col = PALETTE[cid % len(PALETTE)]
        d.rectangle([x0, y0, x1, y1], outline=col, width=2)
        name = classes[cid].split("_", 1)[-1].replace("_", " ") if cid < len(classes) else str(cid)
        tw = d.textlength(name, font=f)
        ty = max(0, y0 - 16)
        d.rectangle([x0, ty, x0 + tw + 6, ty + 16], fill=col + (215,))
        d.text((x0 + 3, ty), name, font=f, fill=(255, 255, 255))
    return im


def _b64_jpeg(im) -> str:
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=82)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def _job(args):
    """子进程：渲染一张缩略图落盘，返回元数据。"""
    img_path, labs, classes, thumb_dir, max_w = args
    try:
        im = Image.open(img_path).convert("RGB")
        s = max_w / im.width
        im = im.resize((max_w, max(1, round(im.height * s))), Image.LANCZOS)
        im = render_thumb(im, labs, classes)
        out = thumb_dir / (img_path.stem + ".jpg")
        im.save(out, "JPEG", quality=80)
        return {"s": img_path.stem, "b": len(labs),
                "c": sorted({c for c, *_ in labs}), "ok": True}
    except Exception as e:  # 渲染失败也要在页面里露脸，不能静默吞掉
        return {"s": img_path.stem, "b": len(labs), "c": [], "ok": False, "err": str(e)[:120]}


def build_index_html(ydir: Path, meta: list[dict], classes: list[str],
                     stats: dict, thumb_w: int) -> None:
    """全量索引页：元数据内嵌 JSON，前端分页 + 筛选，懒加载缩略图。"""
    cls_disp = [c.split("_", 1)[-1].replace("_", " ") for c in classes]
    payload = json.dumps(
        {"meta": meta, "classes": cls_disp}, ensure_ascii=False, separators=(",", ":"))

    boxes_all = stats["boxes_all"]
    per_img = stats["per_img"]
    top = "".join(
        f"<tr><td class='cls'>{cls_disp[i] if i < len(cls_disp) else i}</td><td>{n}</td>"
        f"<td><div class='bar' style='width:{n / max(boxes_all.values()) * 220:.0f}px'></div></td></tr>"
        for i, n in sorted(boxes_all.items(), key=lambda kv: -kv[1])[:25])
    hist = "".join(
        f"<div class='hbar'><span>{k}只</span><div style='height:{max(4, v / max(per_img.values()) * 90):.0f}px'></div><em>{v}</em></div>"
        for k, v in sorted(per_img.items())[:24])

    html = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>YOLO round2 全量自检 · {ydir.name}</title>
<style>
 body{{margin:0;background:#14161c;color:#e6e6e6;font:14px/1.6 "Microsoft YaHei",system-ui,sans-serif}}
 header{{padding:20px 26px;border-bottom:1px solid #2a2d36;background:#181b22}}
 h1{{margin:0 0 6px;font-size:19px;font-weight:500}}
 .stats{{display:flex;gap:26px;flex-wrap:wrap;margin-top:10px}}
 .stat b{{display:block;font-size:20px;color:#7fd1a8;font-weight:500}}
 .stat span{{color:#8b90a0;font-size:12px}}
 section{{padding:18px 26px}}
 h2{{font-size:15px;font-weight:500;color:#c9cdd8;margin:0 0 12px}}
 .grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(330px,1fr));gap:14px}}
 .card{{margin:0;background:#1c1f27;border:1px solid #2a2d36;border-radius:10px;overflow:hidden}}
 .card img{{width:100%;display:block;min-height:90px;background:#101218}}
 .card.bad{{border-color:#a04a44}}
 figcaption{{padding:7px 10px;font-size:12px;color:#9aa0b0;display:flex;gap:8px;align-items:baseline;flex-wrap:wrap}}
 .zero{{color:#e0837a}} .many{{color:#7fd1a8}} .bad-tag{{color:#e0837a;font-weight:600}}
 table{{border-collapse:collapse;font-size:12px}}
 td{{padding:2px 10px 2px 0;color:#b9becc}} td.cls{{color:#e6e6e6}}
 .bar{{height:9px;background:#3f8f6d;border-radius:2px}}
 .bars{{display:flex;align-items:flex-end;gap:5px;height:120px;margin-top:8px}}
 .hbar{{display:flex;flex-direction:column;justify-content:flex-end;align-items:center;gap:3px;font-size:10px;color:#8b90a0}}
 .hbar div{{width:20px;background:#5b6bb5;border-radius:2px 2px 0 0}}
 .hbar em{{font-style:normal;color:#6f7484}}
 .two{{display:flex;gap:40px;flex-wrap:wrap}}
 #ctrl{{position:sticky;top:0;z-index:9;background:#181b22ee;border-bottom:1px solid #2a2d36;
        padding:10px 26px;display:flex;gap:14px;align-items:center;flex-wrap:wrap;backdrop-filter:blur(4px)}}
 #ctrl input[type=text],#ctrl input[type=number]{{background:#101218;color:#e6e6e6;border:1px solid #2a2d36;
        border-radius:6px;padding:5px 9px;font:13px "Microsoft YaHei"}}
 #ctrl input[type=text]{{width:230px}}
 #ctrl input[type=number]{{width:76px}}
 #ctrl label{{font-size:12px;color:#9aa0b0;display:flex;gap:5px;align-items:center}}
 #count{{font-size:12px;color:#8b90a0}}
 .pager{{display:flex;gap:8px;align-items:center;justify-content:center;padding:16px}}
 .pager button{{background:#232733;color:#dfe3ee;border:1px solid #2a2d36;border-radius:6px;
        padding:6px 14px;cursor:pointer;font-size:13px}}
 .pager button:disabled{{opacity:.35;cursor:default}}
 .pager button.cur{{background:#3f8f6d;border-color:#3f8f6d;color:#fff}}
 .clschip{{font-size:10px;background:#262b38;border-radius:4px;padding:0 5px;color:#9aa0b0}}
</style></head><body>
<header>
 <h1>YOLO round2 全量自检 · {ydir.name}（每张图都在，无抽样）</h1>
 <div class="stats">
  <div class="stat"><b>{stats['n_img']}</b><span>图片（全量）</span></div>
  <div class="stat"><b>{stats['n_map']}</b><span>地图</span></div>
  <div class="stat"><b>{len(classes)}</b><span>怪物类别</span></div>
  <div class="stat"><b>{stats['n_box']}</b><span>怪框总数</span></div>
  <div class="stat"><b>{round(stats['n_box'] / max(1, stats['n_img'] - stats['n_empty']), 1)}</b><span>平均每张（非空）</span></div>
  <div class="stat"><b>{stats['n_empty']}</b><span>空图/负样本</span></div>
  <div class="stat"><b>{stats['n_fail']}</b><span>渲染失败</span></div>
 </div>
</header>
<section class="two">
 <div><h2>每张图的怪数量分布</h2><div class="bars">{hist}</div></div>
 <div><h2>类别框数 Top 25</h2><table>{top}</table></div>
</section>
<div id="ctrl">
 <input type="text" id="q" placeholder="搜地图id / 文件名 / 类别名">
 <label>框数 ≥ <input type="number" id="cmin" min="0" value="0"></label>
 <label>框数 ≤ <input type="number" id="cmax" min="0" value=""></label>
 <label><input type="checkbox" id="hideEmpty"> 隐藏空图</label>
 <span id="count"></span>
</div>
<section><div class="grid" id="grid"></div>
 <div class="pager" id="pager"></div>
</section>
<script>
const D = {payload};
const M = D.meta, CLS = D.classes;
const PAGE = 240;
let view = M.slice(), page = 0;

const grid = document.getElementById('grid'), pager = document.getElementById('pager'),
      cnt = document.getElementById('count');
const esc = s => s.replace(/&/g,'&amp;').replace(/</g,'&lt;');

function applyFilter(){{
  const q = document.getElementById('q').value.trim().toLowerCase();
  const cmin = +document.getElementById('cmin').value || 0;
  const cmaxRaw = document.getElementById('cmax').value;
  const cmax = cmaxRaw === '' ? Infinity : +cmaxRaw;
  const he = document.getElementById('hideEmpty').checked;
  view = M.filter(m => {{
    if (m.b < cmin || m.b > cmax) return false;
    if (he && m.b === 0) return false;
    if (q) {{
      let hit = m.s.toLowerCase().includes(q);
      if (!hit && m.c) for (const ci of m.c) {{
        const cn = CLS[ci]; if (cn && cn.toLowerCase().includes(q)) {{ hit = true; break; }}
      }}
      if (!hit) return false;
    }}
    return true;
  }});
  page = 0;
  cnt.textContent = view.length + ' / ' + M.length + ' 张';
  render();
}}

function card(m){{
  const mm = m.s.match(/^(\\d{{9}})(?:_(.+))?$/);
  const map = mm ? mm[1] : '?', name = mm && mm[2] ? mm[2] : '';
  const chips = (m.c || []).slice(0, 4).map(ci =>
      `<span class="clschip">${{esc(CLS[ci] || ci)}}</span>`).join(' ');
  const head = m.ok
      ? `<a href="../images/${{m.s}}.png" target="_blank" style="color:inherit;text-decoration:none">`
        + `<img src="t/${{m.s}}.jpg" loading="lazy"></a>`
      : `<div style="padding:28px 10px;color:#e0837a;text-align:center">缩略图渲染失败：${{esc(m.err || '')}}</div>`;
  return `<figure class="card${{m.ok ? '' : ' bad'}}">${{head}}`
      + `<figcaption><b>${{map}}</b> · ${{esc(name)}} · `
      + `<span class="${{m.b ? 'many' : 'zero'}}">${{m.b}} 只</span> ${{chips}}`
      + `${{m.ok ? '' : ' <span class="bad-tag">✕ 渲染失败</span>'}}</figcaption></figure>`;
}}

function render(){{
  const total = Math.max(1, Math.ceil(view.length / PAGE));
  if (page >= total) page = total - 1;
  const start = page * PAGE, part = view.slice(start, start + PAGE);
  grid.innerHTML = part.map(card).join('');
  let btns = `<button ${{page === 0 ? 'disabled' : ''}} data-p="0">« 首页</button>`
      + `<button ${{page === 0 ? 'disabled' : ''}} data-p="${{page - 1}}">‹ 上一页</button>`;
  const lo = Math.max(0, page - 4), hi = Math.min(total, lo + 9);
  for (let i = lo; i < hi; i++)
    btns += `<button class="${{i === page ? 'cur' : ''}}" data-p="${{i}}">${{i + 1}}</button>`;
  btns += `<button ${{page >= total - 1 ? 'disabled' : ''}}" data-p="${{page + 1}}">下一页 ›</button>`
      + `<button ${{page >= total - 1 ? 'disabled' : ''}}" data-p="${{total - 1}}">末页 »</button>`
      + `<span style="color:#6f7484;font-size:12px">${{start + 1}}–${{Math.min(start + PAGE, view.length)}} / ${{view.length}}</span>`;
  pager.innerHTML = btns;
  pager.querySelectorAll('button[data-p]').forEach(b =>
      b.onclick = () => {{ page = +b.dataset.p; render(); window.scrollTo({{top: document.getElementById('ctrl').offsetTop - 4}}); }});
}}

document.getElementById('q').addEventListener('input', applyFilter);
document.getElementById('cmin').addEventListener('input', applyFilter);
document.getElementById('cmax').addEventListener('input', applyFilter);
document.getElementById('hideEmpty').addEventListener('change', applyFilter);
cnt.textContent = M.length + ' / ' + M.length + ' 张';
render();
</script>
</body></html>"""
    out = ydir / "review" / "index.html"
    out.write_text(html, encoding="utf-8")
    print(f"全量自检页: {out}")


def run_full(ydir: Path, imgs, lab_of, classes, workers: int, thumb_w: int) -> None:
    thumb_dir = ydir / "review" / "t"
    thumb_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(p, lab_of[p], classes, thumb_dir, thumb_w) for p in imgs]
    meta: list[dict] = []
    done = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for fut in as_completed([ex.submit(_job, j) for j in jobs]):
            meta.append(fut.result())
            done += 1
            if done % 1000 == 0 or done == len(jobs):
                print(f"  缩略图 {done}/{len(jobs)}")
    meta.sort(key=lambda m: m["s"])  # 与 imgs 排序一致
    n_fail = sum(1 for m in meta if not m["ok"])

    by_map = defaultdict(int)
    for p in imgs:
        m = re.match(r"^(\d{9})", p.stem)
        by_map[m.group(1) if m else "?"] += 1
    stats = {
        "n_img": len(imgs), "n_map": len(by_map),
        "n_box": sum(len(lab_of[p]) for p in imgs),
        "n_empty": sum(1 for p in imgs if not lab_of[p]),
        "n_fail": n_fail,
        "boxes_all": Counter(c for p in imgs for c, *_ in lab_of[p]),
        "per_img": Counter(len(lab_of[p]) for p in imgs),
    }
    build_index_html(ydir, meta, classes, stats, thumb_w)
    if n_fail:
        print(f"⚠ 有 {n_fail} 张缩略图渲染失败，页面上已标红，请检查")


def run_sample(ydir: Path, imgs, lab_of, classes, a) -> None:
    by_map = defaultdict(list)
    for p in imgs:
        m = re.match(r"^(\d{9})", p.stem)
        by_map[m.group(1) if m else "?"].append(p)
    rnd = random.Random(a.seed)
    for ps in by_map.values():
        ps.sort(key=lambda p: (-len(lab_of[p]), p.name))
        rnd.shuffle(ps[: max(2, len(ps) // 3)])
    picks, round_i = [], 0
    while len(picks) < a.sample and round_i < a.per_map:
        for m in sorted(by_map):
            group = by_map[m]
            if round_i < len(group) and len(picks) < a.sample:
                picks.append(group[round_i])
        round_i += 1
    empty = [p for p in imgs if not lab_of[p]]
    for p in empty[: max(3, a.sample // 10)]:
        if p not in picks:
            picks.append(p)

    n_box = sum(len(lab_of[p]) for p in imgs)
    boxes_all = Counter(c for p in imgs for c, *_ in lab_of[p])
    per_img = Counter(len(lab_of[p]) for p in imgs)
    n_empty = sum(1 for p in imgs if not lab_of[p])
    cls_disp = [c.split("_", 1)[-1].replace("_", " ") for c in classes]

    cards = []
    for p in picks:
        labs = lab_of[p]
        im = Image.open(p).convert("RGB")
        s = a.thumb_w / im.width
        im = im.resize((a.thumb_w, max(1, round(im.height * s))), Image.LANCZOS)
        im = render_thumb(im, labs, classes)
        thumb = _b64_jpeg(im)
        m = re.match(r"^(\d{9})_(.+)$", p.stem)
        cards.append(
            f'<figure class="card"><img src="{thumb}" loading="lazy">'
            f'<figcaption><b>{m.group(1) if m else "?"}</b> · {m.group(2) if m else p.stem}'
            f' · <span class="{"zero" if not labs else "many"}">{len(labs)} 只</span>'
            f'</figcaption></figure>')

    top = "".join(
        f"<tr><td class='cls'>{cls_disp[i] if i < len(cls_disp) else i}</td><td>{n}</td>"
        f"<td><div class='bar' style='width:{n / max(boxes_all.values()) * 220:.0f}px'></div></td></tr>"
        for i, n in sorted(boxes_all.items(), key=lambda kv: -kv[1])[:25])
    hist = "".join(
        f"<div class='hbar'><span>{k}只</span><div style='height:{max(4, v / max(per_img.values()) * 90):.0f}px'></div><em>{v}</em></div>"
        for k, v in sorted(per_img.items())[:24])

    html = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>YOLO round2 导出自检（抽样） · {ydir.name}</title>
<style>
 body{{margin:0;background:#14161c;color:#e6e6e6;font:14px/1.6 "Microsoft YaHei",system-ui,sans-serif}}
 header{{padding:20px 26px;border-bottom:1px solid #2a2d36;background:#181b22}}
 h1{{margin:0 0 6px;font-size:19px;font-weight:500}}
 .stats{{display:flex;gap:26px;flex-wrap:wrap;margin-top:10px}}
 .stat b{{display:block;font-size:20px;color:#7fd1a8;font-weight:500}}
 .stat span{{color:#8b90a0;font-size:12px}}
 section{{padding:18px 26px}}
 h2{{font-size:15px;font-weight:500;color:#c9cdd8;margin:0 0 12px}}
 .grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(330px,1fr));gap:14px}}
 .card{{margin:0;background:#1c1f27;border:1px solid #2a2d36;border-radius:10px;overflow:hidden}}
 .card img{{width:100%;display:block}}
 figcaption{{padding:7px 10px;font-size:12px;color:#9aa0b0}}
 .zero{{color:#e0837a}} .many{{color:#7fd1a8}}
 table{{border-collapse:collapse;font-size:12px}}
 td{{padding:2px 10px 2px 0;color:#b9becc}} td.cls{{color:#e6e6e6}}
 .bar{{height:9px;background:#3f8f6d;border-radius:2px}}
 .bars{{display:flex;align-items:flex-end;gap:5px;height:120px;margin-top:8px}}
 .hbar{{display:flex;flex-direction:column;justify-content:flex-end;align-items:center;gap:3px;font-size:10px;color:#8b90a0}}
 .hbar div{{width:20px;background:#5b6bb5;border-radius:2px 2px 0 0}}
 .hbar em{{font-style:normal;color:#6f7484}}
 .two{{display:flex;gap:40px;flex-wrap:wrap}}
</style></head><body>
<header>
 <h1>YOLO round2 导出自检（抽样） · {ydir.name}</h1>
 <div class="stats">
  <div class="stat"><b>{len(imgs)}</b><span>图片</span></div>
  <div class="stat"><b>{len(by_map)}</b><span>地图</span></div>
  <div class="stat"><b>{len(classes)}</b><span>怪物类别</span></div>
  <div class="stat"><b>{n_box}</b><span>怪框总数</span></div>
  <div class="stat"><b>{round(n_box / max(1, len(imgs) - n_empty), 1)}</b><span>平均每张（非空）</span></div>
  <div class="stat"><b>{n_empty}</b><span>空图/负样本</span></div>
 </div>
</header>
<section class="two">
 <div><h2>每张图的怪数量分布</h2><div class="bars">{hist}</div></div>
 <div><h2>类别框数 Top 25</h2><table>{top}</table></div>
</section>
<section><h2>抽样预览（{len(picks)} 张，每张地图最多 {a.per_map} 张）</h2>
 <div class="grid">{''.join(cards)}</div>
</section>
</body></html>"""
    out = ydir / "review.html"
    out.write_text(html, encoding="utf-8")
    print(f"自检页: {out}  （{len(picks)} 张样本 / {len(imgs)} 张总图）")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="train/data/yolo_r2")
    ap.add_argument("--sample", type=int, default=0, metavar="N",
                    help="抽样模式：只展示 N 张（默认 0 = 全量）")
    ap.add_argument("--per-map", type=int, default=2, help="抽样模式：每张地图最多取几张")
    ap.add_argument("--thumb-w", type=int, default=512, help="缩略图宽度")
    ap.add_argument("--workers", type=int, default=8, help="全量模式渲染进程数")
    ap.add_argument("--seed", default="rv")
    a = ap.parse_args()

    ydir = Path(a.dir) if Path(a.dir).is_absolute() else ROOT / a.dir
    # 兼容两种布局：平铺 images/*.png（导出后）和 images/{train,val}/*.png（切分后）
    imgs = sorted((ydir / "images").rglob("*.png"))
    if not imgs:
        print(f"{ydir} 里没有图片")
        return 1

    def label_of(p: Path) -> Path:
        rel = p.relative_to(ydir / "images")
        cand = ydir / "labels" / rel.with_suffix(".txt")
        return cand if cand.exists() else ydir / "labels" / (p.stem + ".txt")

    classes = load_classes(ydir)
    lab_of = {p: load_labels(label_of(p)) for p in imgs}
    print(f"{ydir}: {len(imgs)} 张图, {len(classes)} 类")

    if a.sample:
        run_sample(ydir, imgs, lab_of, classes, a)
    else:
        run_full(ydir, imgs, lab_of, classes, a.workers, a.thumb_w)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
