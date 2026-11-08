#!/usr/bin/env python3
"""yololab - YOLO 实时标注台（独立小工具，和 wztool 的 webui 完全分开）。

两种用法：
  1) 图片检测：把截图拖进来 / Ctrl+V 粘贴 / 选文件，页面上直接画出模型认到的框。
  2) 实时屏幕：抓游戏窗口（或整屏）逐帧推理，MJPEG 推流到浏览器，边玩边看准不准。

零 Web 框架依赖：只用标准库 http.server；推理用 ultralytics，抓屏用 mss。

启动：
    python yololab\\app.py                # 自动挑最新权重，开 http://127.0.0.1:8770
    python yololab\\app.py --model path\\to\\best.pt --port 8771
"""
from __future__ import annotations

import argparse
import colorsys
import hashlib
import json
import mimetypes
import os

import subprocess
import sys
import threading
import time
from collections import Counter, deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
STATIC = HERE / "static"
SAMPLES = HERE / "samples"

PORT_DEFAULT = 8770
HOST = "127.0.0.1"

# ---------------------------------------------------------------- 类别显示
TERRAIN = {"ground": "地面", "slope": "斜坡", "ladder": "梯子", "rope": "绳子"}
ACTION = {
    "stand": "站", "move": "移动", "fly": "飞行", "jump": "跳", "prone": "趴",
    "alert": "警戒", "swing": "挥击", "ladder": "爬梯", "rope": "爬绳",
    "attack1": "攻击1", "attack2": "攻击2", "attack3": "攻击3", "attack4": "攻击4",
    "hit1": "受击1", "hit2": "受击2", "die1": "死亡", "die2": "死亡2",
    "skill1": "技能1", "skill2": "技能2", "skill3": "技能3", "skill0": "技能0",
    "skill4": "技能4", "skill5": "技能5", "regen": "回血", "heal": "治疗",
    "summon": "召唤", "buff": "增益", "stun": "眩晕", "counter": "反击",
}
SCENE_COLOR = {
    "ground": (108, 117, 125),
    "slope": (139, 92, 246),
    "ladder": (217, 119, 6),
    "rope": (202, 138, 4),
}


def pretty_label(name: str) -> tuple[str, str]:
    """'9500358_Crimson_Balrog__attack1' -> ('Crimson Balrog · 攻击1', '9500358')"""
    if name in TERRAIN:
        return TERRAIN[name], ""
    if "__" in name:
        head, act = name.split("__", 1)
        mob_id, _, raw = head.partition("_")
        nm = raw.replace("_", " ").strip() or head
        return f"{nm} · {ACTION.get(act, act)}", mob_id if mob_id.isdigit() else ""
    return name.replace("_", " "), ""


def color_for(name: str) -> tuple[int, int, int]:
    if name in SCENE_COLOR:
        return SCENE_COLOR[name]
    h = int(hashlib.md5(name.encode("utf-8")).hexdigest()[:6], 16) / 0xFFFFFF
    r, g, b = colorsys.hsv_to_rgb(h, 0.68, 0.82)
    return int(r * 255), int(g * 255), int(b * 255)


def hex_rgb(rgb: tuple[int, int, int]) -> str:
    return "#%02x%02x%02x" % rgb


# ---------------------------------------------------------------- 权重发现
def repo_roots() -> list[Path]:
    """项目根 + （在 worktree 里跑时）主仓库根，两处都找权重。"""
    roots = [ROOT]
    try:
        out = subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "--path-format=absolute", "--git-common-dir"],
            capture_output=True, text=True, timeout=5,
        )
        if out.returncode == 0:
            common = Path(out.stdout.strip())
            main_root = common.parent if common.name == ".git" else common
            if main_root.exists() and main_root not in roots:
                roots.append(main_root)
    except Exception:
        pass
    return roots


MODEL_PATTERNS = [
    "yololab/models/*.pt",
    "models/*.pt",
    "train/runs/*/weights/best.pt",
    "train/runs/*/weights/last.pt",
    "runs/*/weights/best.pt",
]


def discover_models() -> list[dict]:
    seen: dict[str, dict] = {}
    for r in repo_roots():
        for pat in MODEL_PATTERNS:
            for p in r.glob(pat):
                key = str(p.resolve()).lower()
                if key in seen or not p.is_file():
                    continue
                try:
                    st = p.stat()
                except OSError:
                    continue
                seen[key] = {
                    "path": str(p),
                    "name": p.name,
                    "dir": str(p.parent),
                    "size_mb": round(st.st_size / 1048576, 1),
                    "mtime": int(st.st_mtime),
                    "root": str(r),
                }

    def score(m: dict) -> tuple:
        # 更"像"最终权重、更新的排前面
        base = m["name"].lower()
        pref = 0 if base == "best.pt" else (1 if base.startswith("best") else (2 if base == "last.pt" else 3))
        return (pref, -m["mtime"])

    return sorted(seen.values(), key=score)


def pick_default_model(explicit: str | None) -> str | None:
    if explicit:
        return explicit
    models = discover_models()
    # 跳过 COCO 预训练（80 类、名字带 yolov8/yolo11 的裸权重）
    for m in models:
        low = m["path"].lower().replace("\\", "/")
        if "weights/best.pt" in low or "/models/best" in low:
            return m["path"]
    for m in models:
        if "yolov8n.pt" in low_name(m["path"]) and "models" not in m["path"].lower().replace("\\", "/"):
            continue
        return m["path"]
    return models[0]["path"] if models else None


def low_name(p: str) -> str:
    return Path(p).name.lower()


# ---------------------------------------------------------------- 模型封装
class ModelBox:
    def __init__(self, path: str, device: str = "cpu"):
        from ultralytics import YOLO

        self.lock = threading.Lock()
        self.path = str(Path(path).resolve())
        t0 = time.perf_counter()
        self.model = YOLO(self.path)
        self.names = {int(k): str(v) for k, v in self.model.names.items()}
        self.device = device
        # 静态导出的 onnx 会把输入 shape 写死（如 1×3×800×1376）。
        # 这种模型换个分辨率不会报错，只会被静默 resize 回固定尺寸 —— 必须让 UI 知道。
        self.fixed_shape = self._probe_fixed_shape()
        self._warned_shape = False
        self.load_ms = round((time.perf_counter() - t0) * 1000)
        # 预热，避免第一帧特别慢
        try:
            import numpy as np

            with self.lock:
                self.model.predict(np.zeros((320, 320, 3), dtype=np.uint8), imgsz=320,
                                   verbose=False, device=self.device)
        except Exception:
            pass

    def _probe_fixed_shape(self):
        """读 onnx 输入维度；全为整数说明是静态导出，返回 (h, w)，否则 None。"""
        if not self.path.lower().endswith(".onnx"):
            return None
        try:
            import onnxruntime as ort

            sess = ort.InferenceSession(self.path, providers=["CPUExecutionProvider"])
            shape = sess.get_inputs()[0].shape
            if all(isinstance(d, int) for d in shape):
                return (int(shape[2]), int(shape[3]))
        except Exception:
            pass
        return None

    def info(self) -> dict:
        return {
            "path": self.path,
            "name": Path(self.path).name,
            "classes": len(self.names),
            "device": self.device,
            "load_ms": self.load_ms,
            "fixed_shape": list(self.fixed_shape) if self.fixed_shape else None,
        }

    def predict(self, bgr, conf: float, iou: float, imgsz: int) -> tuple[list[dict], float]:
        with self.lock:
            # 静态 onnx 只认一个 shape：传别的尺寸不会报错，而是被悄悄 resize 回去，
            # 既白改参数又让人误以为"切到 960 变快了"。提示一次就够，别刷屏。
            if self.fixed_shape and not self._warned_shape:
                fh, fw = self.fixed_shape
                if ((int(imgsz) + 31) // 32) * 32 != fw:
                    print(f"[!] 模型是静态 ONNX（锁定 {fw}×{fh}），imgsz={imgsz} 不会生效；"
                          f"要换分辨率请用 train/src/export_onnx.py 重导")
                    self._warned_shape = True
            t0 = time.perf_counter()
            # rect=True：按原图 1366×768 比例出张量（约 800×1376），不补成正方形。
            # ultralytics 推理本来默认就走 rect，这里显式写出来，免得默认值一变
            # 就悄悄退化成 1376×1376 方形画布（灰边多花一倍算力）。
            res = self.model.predict(bgr, conf=conf, iou=iou, imgsz=imgsz, rect=True,
                                     verbose=False, device=self.device)[0]
            ms = (time.perf_counter() - t0) * 1000
        dets = []
        for b in res.boxes:
            cid = int(b.cls[0])
            name = self.names.get(cid, str(cid))
            label, mob_id = pretty_label(name)
            x0, y0, x1, y1 = (float(v) for v in b.xyxy[0])
            dets.append({
                "cls": cid,
                "name": name,
                "label": label,
                "mob_id": mob_id,
                "conf": round(float(b.conf[0]), 3),
                "box": [round(x0, 1), round(y0, 1), round(x1, 1), round(y1, 1)],
                "color": hex_rgb(color_for(name)),
                "terrain": name in TERRAIN,
            })
        dets.sort(key=lambda d: -d["conf"])
        return dets, round(ms, 1)


# ---------------------------------------------------------------- 抓屏工具
def enable_dpi_awareness() -> None:
    if os.name != "nt":
        return
    try:
        import ctypes

        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        try:
            import ctypes

            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass


GAME_KEYWORDS = ("冒险岛", "maple", "메이플")


def list_windows() -> list[dict]:
    """枚举可见、够大的窗口（标题带关键词的排前面）。"""
    if os.name != "nt":
        return []
    import ctypes
    from ctypes import wintypes

    user32 = ctypes.windll.user32
    hwnds = []

    @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    def cb(hwnd, _):
        try:
            if not user32.IsWindowVisible(hwnd) or user32.IsIconic(hwnd):
                return True
            n = user32.GetWindowTextLengthW(hwnd)
            if n <= 0:
                return True
            buf = ctypes.create_unicode_buffer(n + 1)
            user32.GetWindowTextW(hwnd, buf, n + 1)
            title = buf.value.strip()
            rect = wintypes.RECT()
            user32.GetClientRect(hwnd, ctypes.byref(rect))
            w, h = rect.right - rect.left, rect.bottom - rect.top
            if w < 320 or h < 240:
                return True
            pt = wintypes.POINT(0, 0)
            user32.ClientToScreen(hwnd, ctypes.byref(pt))
            hwnds.append({"hwnd": int(hwnd), "title": title, "x": pt.x, "y": pt.y, "w": w, "h": h})
        except Exception:
            pass
        return True

    user32.EnumWindows(cb, 0)
    for h in hwnds:
        low = h["title"].lower()
        h["game"] = any(k in low for k in GAME_KEYWORDS)
    hwnds.sort(key=lambda h: (not h["game"], -(h["w"] * h["h"])))
    return hwnds


def window_client_rect(hwnd: int) -> list[int] | None:
    if os.name != "nt":
        return None
    import ctypes
    from ctypes import wintypes

    user32 = ctypes.windll.user32
    if not user32.IsWindow(hwnd):
        return None
    rect = wintypes.RECT()
    if not user32.GetClientRect(hwnd, ctypes.byref(rect)):
        return None
    pt = wintypes.POINT(0, 0)
    user32.ClientToScreen(hwnd, ctypes.byref(pt))
    w, h = rect.right - rect.left, rect.bottom - rect.top
    if w <= 0 or h <= 0:
        return None
    return [pt.x, pt.y, w, h]


# ---------------------------------------------------------------- 画框
_FONT_CACHE: dict[int, object] = {}


def get_font(size: int):
    if size in _FONT_CACHE:
        return _FONT_CACHE[size]
    from PIL import ImageFont

    font = None
    for cand in ("msyh.ttc", "C:/Windows/Fonts/msyh.ttc", "simhei.ttf", "consola.ttf"):
        try:
            font = ImageFont.truetype(cand, size)
            break
        except Exception:
            continue
    if font is None:
        font = ImageFont.load_default()
    _FONT_CACHE[size] = font
    return font


def annotate_bgr(bgr, dets: list[dict], cfg: dict):
    """在 BGR ndarray 上画框，返回 (RGB PIL Image, 实际画出的框数)。"""
    import cv2
    from PIL import Image, ImageDraw

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    im = Image.fromarray(rgb)
    dr = ImageDraw.Draw(im)
    font = get_font(14 if im.width >= 800 else 12)
    only_targets = cfg.get("only_targets", False)
    show_labels = cfg.get("show_labels", True)
    show_conf = cfg.get("show_conf", True)
    shown = 0
    for d in dets:
        if only_targets and d["terrain"]:
            continue
        shown += 1
        x0, y0, x1, y1 = d["box"]
        color = tuple(int(d["color"][i:i + 2], 16) for i in (1, 3, 5))
        dr.rectangle([x0, y0, x1, y1], outline=color, width=2)
        if not show_labels:
            continue
        text = d["label"] if show_conf else d["label"]
        if show_conf:
            text = f"{text} {d['conf']:.2f}"
        try:
            tw = int(dr.textlength(text, font=font))
            th = font.size + 4
        except Exception:
            tw, th = len(text) * 7, 16
        ty = y0 - th - 2
        if ty < 0:
            ty = y0 + 2
        dr.rectangle([x0, ty, x0 + tw + 6, ty + th], fill=color)
        dr.text((x0 + 3, ty + 2), text, fill=(255, 255, 255), font=font)
    return im, shown


def jpeg_bytes(im, quality: int = 82) -> bytes:
    import io

    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


# ---------------------------------------------------------------- 实时检测
class Live:
    def __init__(self, box: ModelBox):
        self.box = box
        self.cond = threading.Condition()
        self.thread: threading.Thread | None = None
        self.running = False
        self.cfg: dict = {}
        self.jpeg: bytes | None = None
        self.frame_id = 0
        self.last_bgr = None
        self.error: str | None = None
        self.stats = {"fps": 0.0, "ms": 0.0, "count": 0, "frame": 0,
                      "classes": {}, "region": None, "source": ""}
        self._times: deque[float] = deque(maxlen=30)

    # -- 控制
    def start(self, cfg: dict) -> dict:
        self.stop()
        self.cfg = cfg
        self.error = None
        self.running = True
        self._times.clear()
        with self.cond:
            self.jpeg = None
            self.frame_id = 0
        self.thread = threading.Thread(target=self._loop, args=(cfg,), daemon=True, name="yololab-live")
        self.thread.start()
        return {"ok": True, "source": cfg.get("source"), "region": cfg.get("region")}

    def stop(self) -> dict:
        self.running = False
        th = self.thread
        if th and th.is_alive():
            th.join(timeout=2.0)
        self.thread = None
        with self.cond:
            self.cond.notify_all()
        return {"ok": True}

    def status(self) -> dict:
        with self.cond:
            return {"running": self.running, "error": self.error, **self.stats}

    # -- 主循环
    def _loop(self, cfg: dict) -> None:
        import mss
        import numpy as np

        source = cfg.get("source", "screen")
        hwnd = int(cfg.get("hwnd") or 0)
        region = cfg.get("region")
        target_fps = max(1.0, float(cfg.get("fps") or 8))
        conf = float(cfg.get("conf") or 0.35)
        iou = float(cfg.get("iou") or 0.45)
        imgsz = int(cfg.get("imgsz") or 1366)
        src_name = cfg.get("source_name") or source

        try:
            opener = getattr(mss, "MSS", None) or mss.mss
            with opener() as sct:
                monitor = None
                tick = 0
                while self.running:
                    t0 = time.perf_counter()
                    if source == "window" and hwnd:
                        r = window_client_rect(hwnd)
                        if r:
                            region = r
                    if region:
                        monitor = {"left": int(region[0]), "top": int(region[1]),
                                   "width": int(region[2]), "height": int(region[3])}
                    else:
                        monitor = sct.monitors[1]
                        region = [monitor["left"], monitor["top"], monitor["width"], monitor["height"]]

                    shot = sct.grab(monitor)
                    bgr = np.frombuffer(shot.bgra, dtype=np.uint8).reshape(shot.height, shot.width, 4)[:, :, :3]
                    bgr = np.ascontiguousarray(bgr)

                    self.last_bgr = bgr.copy()
                    dets, ms = self.box.predict(bgr, conf, iou, imgsz)
                    im, shown = annotate_bgr(bgr, dets, cfg)
                    jpg = jpeg_bytes(im, 80)

                    counts: dict[str, dict] = {}
                    for d in dets:
                        if cfg.get("only_targets") and d["terrain"]:
                            continue
                        e = counts.setdefault(d["label"], {"label": d["label"], "name": d["name"],
                                                           "color": d["color"], "n": 0})
                        e["n"] += 1
                    classes = sorted(counts.values(), key=lambda x: -x["n"])[:40]
                    now = time.perf_counter()
                    self._times.append(now)
                    fps = 0.0
                    if len(self._times) > 1:
                        span = self._times[-1] - self._times[0]
                        if span > 0:
                            fps = round((len(self._times) - 1) / span, 1)
                    with self.cond:
                        self.jpeg = jpg
                        self.frame_id += 1
                        self.stats = {
                            "fps": fps,
                            "ms": ms,
                            "count": shown,
                            "frame": self.frame_id,
                            "classes": classes,
                            "region": list(region),
                            "source": src_name,
                        }
                        self.cond.notify_all()

                    elapsed = time.perf_counter() - t0
                    delay = 1.0 / target_fps - elapsed
                    if delay > 0:
                        time.sleep(min(delay, 0.5))
        except Exception as exc:  # 抓屏/推理异常：停下并报给前端
            self.error = f"{type(exc).__name__}: {exc}"
        finally:
            self.running = False
            with self.cond:
                self.cond.notify_all()

    # -- 推流
    def stream(self, handler) -> None:
        handler.send_response(200)
        handler.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        handler.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
        handler.send_header("Pragma", "no-cache")
        handler.end_headers()
        last = -1
        try:
            while True:
                with self.cond:
                    self.cond.wait(timeout=1.0)
                    jpg, fid = self.jpeg, self.frame_id
                    running = self.running
                if jpg is not None and fid != last:
                    last = fid
                    handler.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n")
                    handler.wfile.write(b"Content-Length: %d\r\n\r\n" % len(jpg))
                    handler.wfile.write(jpg)
                    handler.wfile.write(b"\r\n")
                    handler.wfile.flush()
                elif not running and jpg is None:
                    break
                if not running and fid == last:
                    break
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        finally:
            handler.close_connection = True


# ---------------------------------------------------------------- HTTP 服务
STATE: dict = {"box": None, "live": None, "started": time.time()}


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "yololab/1.0"

    # -- 小工具
    def log_message(self, fmt, *args):  # 静音默认访问日志（推流会很吵）
        pass

    def _send(self, code: int, body: bytes, ctype: str, extra: dict | None = None) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _json(self, obj, code: int = 200) -> None:
        self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"),
                   "application/json; charset=utf-8")

    def _body(self) -> bytes:
        n = int(self.headers.get("Content-Length") or 0)
        data = b""
        while len(data) < n:
            chunk = self.rfile.read(min(1 << 16, n - len(data)))
            if not chunk:
                break
            data += chunk
        return data

    def _json_body(self) -> dict:
        raw = self._body()
        if not raw:
            return {}
        try:
            return json.loads(raw.decode("utf-8"))
        except Exception:
            return {}

    def _serve_file(self, path: Path) -> None:
        if not path.is_file():
            self._json({"error": f"not found: {path.name}"}, 404)
            return
        ctype = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        if path.suffix == ".js":
            ctype = "text/javascript; charset=utf-8"
        if path.suffix in (".html", ".css", ".js", ".json", ".md"):
            ctype += "; charset=utf-8" if "charset" not in ctype else ""
        self._send(200, path.read_bytes(), ctype)

    # -- 路由
    def do_GET(self):
        u = urlparse(self.path)
        p = unquote(u.path)
        try:
            if p in ("/", "/index.html"):
                self._serve_file(STATIC / "index.html")
            elif p.startswith("/static/"):
                self._serve_file(STATIC / p[len("/static/"):])
            elif p.startswith("/samples/"):
                self._serve_file(SAMPLES / p[len("/samples/"):])
            elif p == "/api/config":
                box: ModelBox | None = STATE["box"]
                models = discover_models()
                for m in models:
                    try:
                        m["current"] = bool(box and Path(m["path"]).resolve() == Path(box.path))
                    except Exception:
                        m["current"] = False
                samples = []
                if SAMPLES.is_dir():
                    samples = sorted([f.name for f in SAMPLES.iterdir()
                                      if f.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp")])
                self._json({
                    "model": box.info() if box else None,
                    "models": models,
                    "samples": samples,
                    "live": STATE["live"].status() if STATE["live"] else {"running": False},
                    "windows": list_windows()[:12],
                    "platform": sys.platform,
                })
            elif p == "/api/live/status":
                live: Live = STATE["live"]
                self._json(live.status() if live else {"running": False})
            elif p == "/api/stream":
                live: Live = STATE["live"]
                if not live or not live.running:
                    self._json({"error": "live not running"}, 409)
                    return
                live.stream(self)
            elif p == "/api/live/frame.jpg":
                live: Live = STATE["live"]
                if not live or live.last_bgr is None:
                    self._json({"error": "还没有画面"}, 409)
                    return
                import cv2

                rgb = cv2.cvtColor(live.last_bgr, cv2.COLOR_BGR2RGB)
                from PIL import Image

                self._send(200, jpeg_bytes(Image.fromarray(rgb), 90), "image/jpeg")
            elif p == "/api/health":
                self._json({"ok": True, "uptime": round(time.time() - STATE["started"], 1)})
            elif p == "/favicon.ico":
                self._send(204, b"", "image/x-icon")
            else:
                self._json({"error": "unknown path"}, 404)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as exc:
            self._json({"error": f"{type(exc).__name__}: {exc}"}, 500)

    def do_POST(self):
        u = urlparse(self.path)
        p = unquote(u.path)
        q = parse_qs(u.query)
        try:
            if p == "/api/model":
                body = self._json_body()
                path = body.get("path")
                if not path or not Path(path).is_file():
                    self._json({"error": "权重文件不存在"}, 400)
                    return
                if STATE["live"]:
                    STATE["live"].stop()
                box = ModelBox(path, device=body.get("device") or "cpu")
                STATE["box"] = box
                self._json({"ok": True, "model": box.info()})
            elif p == "/api/predict":
                box: ModelBox | None = STATE["box"]
                if not box:
                    self._json({"error": "没有加载模型"}, 503)
                    return
                raw = self._body()
                if not raw:
                    self._json({"error": "空图片"}, 400)
                    return
                import cv2
                import numpy as np
                from PIL import Image

                arr = np.frombuffer(raw, np.uint8)
                bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if bgr is None:
                    self._json({"error": "图片解码失败"}, 400)
                    return
                conf = float(q.get("conf", ["0.35"])[0])
                iou = float(q.get("iou", ["0.45"])[0])
                imgsz = int(q.get("imgsz", ["1366"])[0])
                only = q.get("only_targets", ["0"])[0] in ("1", "true")
                dets, ms = box.predict(bgr, conf, iou, imgsz)
                if only:
                    dets = [d for d in dets if not d["terrain"]]
                counts = Counter(d["label"] for d in dets)
                self._json({
                    "detections": dets,
                    "ms": ms,
                    "size": [int(bgr.shape[1]), int(bgr.shape[0])],
                    "counts": dict(counts.most_common()),
                    "model": box.info(),
                })
            elif p == "/api/live/start":
                box: ModelBox | None = STATE["box"]
                live: Live = STATE["live"]
                if not box or not live:
                    self._json({"error": "没有加载模型"}, 503)
                    return
                body = self._json_body()
                cfg = {
                    "source": body.get("source", "screen"),
                    "hwnd": body.get("hwnd") or 0,
                    "source_name": body.get("source_name") or "屏幕",
                    "fps": float(body.get("fps") or 8),
                    "conf": float(body.get("conf") or 0.35),
                    "iou": float(body.get("iou") or 0.45),
                    "imgsz": int(body.get("imgsz") or 1366),
                    "only_targets": bool(body.get("only_targets")),
                    "show_labels": body.get("show_labels", True),
                    "show_conf": body.get("show_conf", True),
                    "region": body.get("region"),
                }
                if cfg["source"] == "window":
                    if not cfg["hwnd"]:
                        wins = [w for w in list_windows() if w["game"]]
                        if not wins:
                            self._json({"error": "没找到游戏窗口，改用整屏"}, 400)
                            return
                        cfg["hwnd"] = wins[0]["hwnd"]
                        cfg["source_name"] = wins[0]["title"]
                    r = window_client_rect(int(cfg["hwnd"]))
                    if r:
                        cfg["region"] = r
                res = live.start(cfg)
                self._json(res)
            elif p == "/api/live/stop":
                live: Live = STATE["live"]
                self._json(live.stop() if live else {"ok": True})
            else:
                self._json({"error": "unknown path"}, 404)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as exc:
            self._json({"error": f"{type(exc).__name__}: {exc}"}, 500)


class Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


# ---------------------------------------------------------------- 入口
def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="YOLO 实时标注台")
    ap.add_argument("--model", help="权重路径；缺省自动挑最新 best.pt")
    ap.add_argument("--port", type=int, default=PORT_DEFAULT)
    ap.add_argument("--host", default=HOST)
    ap.add_argument("--device", default="cpu", help="cpu / cuda:0")
    ap.add_argument("--threads", type=int, default=0, help="限制 torch CPU 线程数（0=默认）")
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args()

    enable_dpi_awareness()
    if a.threads > 0:
        try:
            import torch

            torch.set_num_threads(a.threads)
        except Exception:
            pass

    path = pick_default_model(a.model)
    if not path:
        print("[!] 没找到 .pt 权重。把训练好的 best.pt 放到 yololab\\models\\ 或 train\\runs\\<round>\\weights\\ 下。")
        return 2

    print(f"[*] 加载权重 {path}")
    box = ModelBox(path, device=a.device)
    print(f"[*] {box.info()['classes']} 类 · 加载 {box.load_ms} ms · device={a.device}")
    STATE["box"] = box
    STATE["live"] = Live(box)

    url = f"http://{a.host}:{a.port}"
    srv = Server((a.host, a.port), Handler)
    print(f"[*] 标注台已启动: {url}   (Ctrl+C 退出)")
    if not a.no_browser:
        threading.Timer(0.8, lambda: _open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n[*] 退出")
    finally:
        if STATE.get("live"):
            STATE["live"].stop()
        srv.server_close()
    return 0


def _open(url: str) -> None:
    try:
        import webbrowser

        webbrowser.open(url)
    except Exception:
        pass


if __name__ == "__main__":
    sys.exit(main())
