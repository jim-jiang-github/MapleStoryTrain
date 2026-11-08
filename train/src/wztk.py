"""wztk.py —— 把 WzToolkit 子模块当「库」用（不是 CLI）。

为什么要这层：
  上游 WzToolkit 的 `map-export` 只负责**出图**（渲染 + 随机摆怪），不认
  `--yolo / --mob-only / --no-boss / --max-mobs` 这些参数——那些是训练侧策略。
  所以这里 import 它的模块，用它的 `_render_preview` / `_composite_mobs` 做引擎，
  标注（YOLO 框、类别表）由训练仓库自己算。好处是：
    · 渲染/层序/相机视差永远和 exe 一致（同一份代码，不是抄的副本）；
    · 上游更新只要 `git submodule update`，不用再 merge 一个 fork。

子模块位置（按优先级）：
  1. 环境变量 WZTOOLKIT_DIR
  2. <repo>/vendor/WzToolkit      （标准情况：git submodule）
"""
from __future__ import annotations

import importlib.util
import json
import os
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]        # train/src/wztk.py -> 仓库根
W, H = 1366, 768                                  # 客户端画面固定尺寸

_W = None          # 已加载的 wztool 模块
_META_CACHE: dict[str, dict] = {}


def toolkit_root() -> Path:
    env = os.environ.get("WZTOOLKIT_DIR")
    p = Path(env) if env else (REPO / "vendor" / "WzToolkit")
    if not (p / "wztool.py").exists():
        raise SystemExit(
            f"[x] 找不到 WzToolkit：{p}\n"
            f"    子模块没拉下来？在仓库根执行：\n"
            f"      git submodule update --init --recursive")
    return p


def wztool(explicit_workspace: str | Path | None = None):
    """加载子模块的 wztool.py 成模块，并把它的缓存/工作区指向本仓库。

    wztool 的 CACHE / EXPORTS / STATE 都是模块级常量（按 __file__ 算的），
    这里改成训练仓库自己的目录——否则所有渲染缓存会写进子模块，让 submodule
    一直被标脏，而且和 exe 的缓存混在一起。
    """
    global _W
    if _W is not None:
        return _W
    tk = toolkit_root()
    spec = importlib.util.spec_from_file_location("_wztk_tool", tk / "wztool.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)          # 全是函数定义，无副作用

    mod.BIN = tk / "bin"                  # 原生工具就住在子模块里，保持不动
    mod.CACHE = REPO / ".cache"
    mod.EXPORTS = REPO / "exports"
    mod.STATE = REPO / "workspace.json"
    mod.CACHE.mkdir(parents=True, exist_ok=True)

    if explicit_workspace:
        mod._ws = Path(explicit_workspace).expanduser().resolve()
    else:
        mod._load_ws()                    # 读 <repo>/workspace.json
    _W = mod
    return mod


def workspace() -> Path:
    W_ = wztool()
    return W_._need_ws()


def set_workspace(path: str | Path) -> Path:
    """记住 .wz 所在工作区（写 <repo>/workspace.json，不进版本库）。"""
    W_ = wztool()
    p = Path(path).expanduser().resolve()
    if not p.is_dir():
        raise SystemExit(f"[x] 工作区不存在：{p}")
    W_._ws = p
    W_.STATE.write_text(json.dumps({"workspace": str(p)}, ensure_ascii=False),
                        encoding="utf-8")
    return p


def cache_dir() -> Path:
    return REPO / ".cache"


def bounds(mj: dict) -> tuple[float, float, float, float]:
    """渲染图的包围盒（含 40px pad）——和 render_file / _composite_mobs 一致。"""
    R = wztool()._render_mod()
    minx, miny, bw, bh = R.map_bounds(mj)
    return minx - 40, miny - 40, bw + 80, bh + 80


def ensure_map(mid: str) -> tuple[dict, Path, str]:
    """确保 .cache/map/<id>/ 里有 map.json + 贴图，返回 (map.json, 目录, 实际id)。

    链接图（info/link，自己没有画面）顺链到目标图，最多 5 跳防环。
    """
    W_ = wztool()
    ws = workspace()
    visited: set[str] = set()
    cur = str(mid).strip()
    for _ in range(5):
        if cur in visited:
            raise RuntimeError(f"link 成环：{cur}")
        visited.add(cur)
        outdir = cache_dir() / "map" / cur
        if not (outdir / "map.json").exists():
            outdir.parent.mkdir(parents=True, exist_ok=True)
            W_._run(["nxmap", cur, str(outdir)], ws, timeout=600)
        if not (outdir / "map.json").exists():
            raise RuntimeError(f"nxmap 没产出 map.json（{cur}）")
        mj = json.loads((outdir / "map.json").read_text(encoding="utf-8"))
        link = str(mj.get("info", {}).get("link") or "").strip()
        if link and link != cur and re.fullmatch(r"\d{9}", link):
            cur = link
            continue
        return mj, outdir, cur
    raise RuntimeError(f"link 跳数过多：{mid}")


def mob_meta() -> dict[str, dict]:
    """{怪 id: {name, tier, ...}}（tier=normal/high/big，来自 Mob.nx）。"""
    global _META_CACHE
    if _META_CACHE:
        return _META_CACHE
    W_ = wztool()
    ws = workspace()
    mobnx = ws / "Mob.nx"
    meta = {m["id"]: m for m in W_.mob_meta(ws, mobnx)} if mobnx.exists() else {}
    _META_CACHE = meta
    return meta


def species_name(kind: str) -> str:
    """怪 id -> 训练用类名（种类名，不带动作/状态后缀）。"""
    kind = str(kind).lstrip("0")
    meta = mob_meta().get(kind, {})
    nm = meta.get("name") or mob_names().get(kind) or ""
    if not nm:
        return kind
    return f"{kind}_{re.sub(r'[^0-9A-Za-z_]+', '_', nm).strip('_')}"


_MOB_NAMES: dict[str, str] = {}


def mob_names() -> dict[str, str]:
    global _MOB_NAMES
    if not _MOB_NAMES:
        try:
            _MOB_NAMES = wztool().mob_names() or {}
        except Exception:
            _MOB_NAMES = {}
    return _MOB_NAMES


def tier_of(kind: str) -> str:
    return mob_meta().get(str(kind).lstrip("0"), {}).get("tier", "normal")


# ---------------------------------------------------------------- 类别表

def load_classes(ydir: Path) -> dict[str, int]:
    try:
        return json.loads((ydir / "classes.json").read_text(encoding="utf-8"))
    except Exception:
        return {}


def register_classes(ydir: Path, names: list[str]) -> dict[str, int]:
    """把新类名追加进 classes.json / classes.txt / dataset.yaml（只增不改）。

    已有 id 一律保持不动——重导历史图时类别号不能漂移，否则老 label 全指错。
    """
    ydir.mkdir(parents=True, exist_ok=True)
    reg = load_classes(ydir)
    add = [n for n in sorted(set(names)) if n not in reg]
    # WZ_CLASSES_RO=1：终验/评估期间禁止改类别表——缺类必须报错，
    # 否则并发导出会「读-改-写」互相覆盖，label 的类别号全部错位（round2 踩过）
    if add and os.environ.get("WZ_CLASSES_RO"):
        raise RuntimeError(f"类别表只读但缺少类别: {', '.join(add[:5])}（共 {len(add)} 个）"
                           "——先跑 yolo-classes 预注册")
    for n in add:
        reg[n] = max(reg.values(), default=-1) + 1
    if add or not (ydir / "classes.txt").exists():
        write_class_files(ydir, reg)
    return reg


def write_class_files(ydir: Path, reg: dict[str, int]) -> None:
    ydir.mkdir(parents=True, exist_ok=True)
    items = sorted(reg.items(), key=lambda kv: kv[1])
    (ydir / "classes.json").write_text(
        json.dumps(reg, ensure_ascii=False, indent=1), encoding="utf-8")
    (ydir / "classes.txt").write_text(
        "\n".join(n for n, _ in items) + "\n", encoding="utf-8")
    lines = [f"path: {ydir.resolve().as_posix()}",
             "train: images", "val: images",
             f"nc: {len(items)}", "names:"]
    lines += [f"  {i}: {n}" for n, i in items]
    (ydir / "dataset.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")
