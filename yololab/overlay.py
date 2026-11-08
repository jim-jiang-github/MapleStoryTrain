#!/usr/bin/env python3
"""yololab/overlay.py —— 直接盖在游戏窗口上的 YOLO 透明遮罩层。

不是网页：一个置顶、穿透鼠标的透明窗口，精确贴在游戏客户区上面，
每帧抓游戏画面 → YOLO 推理 → 把结果画回遮罩上。

  * 地形（ground / slope）画**线**：框的上边缘就是角色站立线
  * 绳 / 梯画**竖线**：框的水平中线就是绳子本体
  * 怪物画框 + 名称/置信度

抓帧用 BitBlt 直接读游戏窗口的 DC（不是抓屏幕），所以遮罩自己画的东西
不会被下一帧拍进去，也不会因为别的窗口挡住游戏而拍到别的东西。

用法：
    python yololab\\overlay.py                 # 自动找游戏窗口 + 最新权重
    python yololab\\overlay.py --conf 0.5 --imgsz 1366 --fps 10
    python yololab\\overlay.py --list          # 只列出可见窗口

    推理和渲染是两条线程：--fps 是推理帧率上限（框更新），
    --rfps 是渲染帧率上限（默认 30）。检测框跨帧配对 + 速度估计 +
    航位推算插值，渲染端指数平滑，所以框的移动是连续的，不会跳格。

热键（Ctrl+Alt 系）：
    H 显示/隐藏遮罩    G 地面线    L 标签    P 暂停    Q 退出
    [ / ] 调置信度      1/2 调推理尺寸
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import math
import sys
import threading
import time
from ctypes import wintypes
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from app import ModelBox, discover_models, enable_dpi_awareness, pick_default_model

user32 = ctypes.windll.user32
gdi32 = ctypes.windll.gdi32

# ---------------------------------------------------------------- Win32 常量
WS_POPUP = 0x80000000
WS_EX_LAYERED = 0x00080000
WS_EX_TRANSPARENT = 0x00000020      # 鼠标穿透
WS_EX_TOPMOST = 0x00000008
WS_EX_TOOLWINDOW = 0x00000080       # 不进 alt-tab
WS_EX_NOACTIVATE = 0x08000000

SWP_NOSIZE = 0x0001
SWP_NOMOVE = 0x0002
SWP_NOACTIVATE = 0x0010
SWP_NOZORDER = 0x0004
SWP_SHOWWINDOW = 0x0040

WM_HOTKEY = 0x0312
SRCCOPY = 0x00CC0020

user32.DefWindowProcW.restype = ctypes.c_longlong
user32.DefWindowProcW.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.c_size_t, ctypes.c_ssize_t]
user32.GetDC.restype = ctypes.c_void_p
user32.GetDC.argtypes = [ctypes.c_void_p]
user32.ReleaseDC.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
user32.GetWindowDC.restype = ctypes.c_void_p
gdi32.CreateCompatibleDC.restype = ctypes.c_void_p
gdi32.CreateCompatibleDC.argtypes = [ctypes.c_void_p]
gdi32.CreateDIBSection.restype = ctypes.c_void_p
gdi32.SelectObject.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
gdi32.DeleteObject.argtypes = [ctypes.c_void_p]
gdi32.DeleteDC.argtypes = [ctypes.c_void_p]

user32.UpdateLayeredWindow.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                       ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                       ctypes.c_uint32, ctypes.c_void_p, ctypes.c_uint32]
user32.SetWindowPos.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
                                ctypes.c_int, ctypes.c_int, ctypes.c_uint]
user32.ShowWindow.argtypes = [ctypes.c_void_p, ctypes.c_int]
user32.DestroyWindow.argtypes = [ctypes.c_void_p]
user32.RegisterHotKey.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_uint, ctypes.c_uint]
user32.UnregisterHotKey.argtypes = [ctypes.c_void_p, ctypes.c_int]
user32.PeekMessageW.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint]
user32.TranslateMessage.argtypes = [ctypes.c_void_p]
user32.DispatchMessageW.argtypes = [ctypes.c_void_p]
user32.GetClientRect.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
user32.ClientToScreen.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
user32.IsWindow.argtypes = [ctypes.c_void_p]
user32.GetForegroundWindow.restype = ctypes.c_void_p
user32.IsIconic.argtypes = [ctypes.c_void_p]
user32.GetWindowTextLengthW.argtypes = [ctypes.c_void_p]
user32.GetWindowTextW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_int]
gdi32.BitBlt.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                         ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_uint32]
gdi32.CreateDIBSection.restype = ctypes.c_void_p
gdi32.CreateDIBSection.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint,
                                   ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32]

WNDPROC = ctypes.WINFUNCTYPE(ctypes.c_longlong, ctypes.c_void_p, ctypes.c_uint,
                             ctypes.c_size_t, ctypes.c_ssize_t)


class BITMAPINFOHEADER(ctypes.Structure):
    _fields_ = [("biSize", ctypes.c_uint32), ("biWidth", ctypes.c_int32),
                ("biHeight", ctypes.c_int32), ("biPlanes", ctypes.c_uint16),
                ("biBitCount", ctypes.c_uint16), ("biCompression", ctypes.c_uint32),
                ("biSizeImage", ctypes.c_uint32), ("biXPelsPerMeter", ctypes.c_int32),
                ("biYPelsPerMeter", ctypes.c_int32), ("biClrUsed", ctypes.c_uint32),
                ("biClrImportant", ctypes.c_uint32)]


class BITMAPINFO(ctypes.Structure):
    _fields_ = [("bmiHeader", BITMAPINFOHEADER), ("bmiColors", ctypes.c_uint32 * 3)]


class WNDCLASSW(ctypes.Structure):
    _fields_ = [("style", ctypes.c_uint), ("lpfnWndProc", ctypes.c_void_p),
                ("cbClsExtra", ctypes.c_int), ("cbWndExtra", ctypes.c_int),
                ("hInstance", ctypes.c_void_p), ("hIcon", ctypes.c_void_p),
                ("hCursor", ctypes.c_void_p), ("hbrBackground", ctypes.c_void_p),
                ("lpszMenuName", ctypes.c_wchar_p), ("lpszClassName", ctypes.c_wchar_p)]


class BLENDFUNCTION(ctypes.Structure):
    _fields_ = [("BlendOp", ctypes.c_ubyte), ("BlendFlags", ctypes.c_ubyte),
                ("SourceConstantAlpha", ctypes.c_ubyte), ("AlphaFormat", ctypes.c_ubyte)]


# ---------------------------------------------------------------- 游戏窗口
GAME_KEYWORDS = ("冒险岛", "maple", "메이플")


def find_game_window(keyword: str | None = None) -> int:
    """按标题关键词找可见、够大的窗口，返回 hwnd。"""
    hits: list[tuple[int, str, int]] = []

    @ctypes.WINFUNCTYPE(wintypes.BOOL, ctypes.c_void_p, ctypes.c_ssize_t)
    def cb(hwnd, _):
        n = user32.GetWindowTextLengthW(hwnd)
        if n <= 0:
            return True
        buf = ctypes.create_unicode_buffer(n + 1)
        user32.GetWindowTextW(hwnd, buf, n + 1)
        title = buf.value.strip()
        low = title.lower()
        keys = (keyword.lower(),) if keyword else GAME_KEYWORDS
        if not any(k in low for k in keys):
            return True
        if not user32.IsWindowVisible(hwnd):      # 最小化到托盘的不算
            return True
        r = wintypes.RECT()
        user32.GetClientRect(hwnd, ctypes.byref(r))
        w, h = r.right - r.left, r.bottom - r.top
        minimized = bool(user32.IsIconic(hwnd))
        # 最小化时客户区是 0x0，别用尺寸筛；恢复后再按真实尺寸判断
        if not minimized and (w < 320 or h < 240):
            return True
        hits.append((int(hwnd), title, 0 if minimized else w * h))
        return True

    user32.EnumWindows(cb, 0)
    if not hits:
        return 0
    hits.sort(key=lambda t: -t[2])               # 非最小化的优先
    return hits[0][0]


def list_windows() -> list[str]:
    rows: list[str] = []
    seen: set[str] = set()

    @ctypes.WINFUNCTYPE(wintypes.BOOL, ctypes.c_void_p, ctypes.c_ssize_t)
    def cb(hwnd, _):
        if not user32.IsWindowVisible(hwnd) or user32.IsIconic(hwnd):
            return True
        n = user32.GetWindowTextLengthW(hwnd)
        if n <= 0:
            return True
        buf = ctypes.create_unicode_buffer(n + 1)
        user32.GetWindowTextW(hwnd, buf, n + 1)
        title = buf.value.strip()
        r = wintypes.RECT()
        user32.GetClientRect(hwnd, ctypes.byref(r))
        w, h = r.right - r.left, r.bottom - r.top
        if w >= 320 and h >= 240 and title not in seen:
            seen.add(title)
            rows.append(f"  hwnd={int(hwnd):>9}  {w:>5}x{h:<5} {title}")
        return True

    user32.EnumWindows(cb, 0)
    return rows


def window_on_top_at(game_hwnd: int, rect: tuple[int, int, int, int]) -> bool:
    """游戏窗口中心点最上层是不是它自己（被别的窗口盖住就返回 False）。"""
    x, y, w, h = rect
    pt = wintypes.POINT(x + w // 2, y + h // 2)
    try:
        user32.WindowFromPoint.argtypes = [wintypes.POINT]
        user32.WindowFromPoint.restype = ctypes.c_void_p
        under = user32.WindowFromPoint(pt)
    except Exception:
        return False
    if not under:
        return False
    user32.GetAncestor.argtypes = [ctypes.c_void_p, ctypes.c_uint]
    user32.GetAncestor.restype = ctypes.c_void_p
    return int(user32.GetAncestor(under, 2) or 0) == int(game_hwnd)


def client_rect(hwnd: int) -> tuple[int, int, int, int] | None:
    """客户区在屏幕上的 (x, y, w, h)。"""
    if not hwnd or not user32.IsWindow(hwnd) or user32.IsIconic(hwnd):
        return None
    r = wintypes.RECT()
    if not user32.GetClientRect(hwnd, ctypes.byref(r)):
        return None
    w, h = r.right - r.left, r.bottom - r.top
    if w <= 0 or h <= 0:
        return None
    pt = wintypes.POINT(0, 0)
    user32.ClientToScreen(hwnd, ctypes.byref(pt))
    return pt.x, pt.y, w, h


# ---------------------------------------------------------------- 抓帧
class WindowCapture:
    """BitBlt 游戏客户区 DC —— 干净、实时、不含遮罩自身。"""

    def __init__(self) -> None:
        self._mem = None
        self._dib = None
        self._ptr = None
        self._size = (0, 0)
        self._bi = BITMAPINFO()
        self._bi.bmiHeader.biSize = ctypes.sizeof(BITMAPINFOHEADER)
        self._bi.bmiHeader.biPlanes = 1
        self._bi.bmiHeader.biBitCount = 32
        self._bi.bmiHeader.biCompression = 0

    def _ensure(self, w: int, h: int) -> None:
        if self._size == (w, h) and self._dib:
            return
        self._free()
        screen = user32.GetDC(0)
        self._mem = gdi32.CreateCompatibleDC(screen)
        self._bi.bmiHeader.biWidth = w
        self._bi.bmiHeader.biHeight = -h
        self._ptr = ctypes.c_void_p()
        self._dib = gdi32.CreateDIBSection(screen, ctypes.byref(self._bi), 0,
                                           ctypes.byref(self._ptr), None, 0)
        gdi32.SelectObject(self._mem, self._dib)
        user32.ReleaseDC(0, screen)
        self._size = (w, h)

    def _free(self) -> None:
        if self._dib:
            gdi32.DeleteObject(self._dib)
            self._dib = None
        if self._mem:
            gdi32.DeleteDC(self._mem)
            self._mem = None
        self._size = (0, 0)

    def grab(self, hwnd: int) -> np.ndarray | None:
        rect = client_rect(hwnd)
        if not rect:
            return None
        _, _, w, h = rect
        if w < 8 or h < 8:
            return None
        self._ensure(w, h)
        src = user32.GetDC(hwnd)
        if not src:
            return None
        ok = gdi32.BitBlt(self._mem, 0, 0, w, h, src, 0, 0, SRCCOPY)
        user32.ReleaseDC(hwnd, src)
        if not ok:
            return None
        buf = ctypes.string_at(self._ptr, w * h * 4)
        arr = np.frombuffer(buf, np.uint8).reshape(h, w, 4)
        return arr[:, :, :3].copy()          # BGRA -> BGR

    def close(self) -> None:
        self._free()


# ---------------------------------------------------------------- 遮罩窗口
class OverlayWindow:
    """置顶、鼠标穿透、逐像素 alpha 的透明窗口。"""

    _class_name = "YololabOverlay"
    _proc = None

    def __init__(self) -> None:
        self.hwnd = 0
        self._mem_dc = None
        self._dib = None
        self._ptr = None
        self._view = None                    # 直接指向 DIB 内存的 numpy 视图
        self._size = (0, 0)
        self._pos = (0, 0)
        self._bi = BITMAPINFO()
        self._bi.bmiHeader.biSize = ctypes.sizeof(BITMAPINFOHEADER)
        self._bi.bmiHeader.biPlanes = 1
        self._bi.bmiHeader.biBitCount = 32
        self._bi.bmiHeader.biCompression = 0
        self.visible = False
        self.on_hotkey = None             # callable(int) -> None
        self._create()

    def _create(self) -> None:
        if OverlayWindow._proc is None:
            OverlayWindow._proc = WNDPROC(self._wndproc)
        wc = WNDCLASSW()
        wc.lpfnWndProc = ctypes.cast(OverlayWindow._proc, ctypes.c_void_p)
        wc.lpszClassName = self._class_name
        wc.hInstance = 0
        user32.RegisterClassW(ctypes.byref(wc))
        self.hwnd = user32.CreateWindowExW(
            WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_TOPMOST | WS_EX_TOOLWINDOW | WS_EX_NOACTIVATE,
            self._class_name, "yololab overlay", WS_POPUP, 0, 0, 1, 1, 0, 0, 0, 0)
        if not self.hwnd:
            raise OSError("创建遮罩窗口失败")

    def _wndproc(self, hwnd, msg, wp, lp):
        if msg == WM_HOTKEY and self.on_hotkey:
            try:
                self.on_hotkey(int(wp))
            except Exception:
                pass
            return 0
        return user32.DefWindowProcW(hwnd, msg, wp, lp)

    # -- 尺寸 / 位置
    def _ensure(self, w: int, h: int) -> None:
        if self._size == (w, h) and self._dib:
            return
        self._view = None
        if self._dib:
            gdi32.DeleteObject(self._dib)
            self._dib = None
        if self._mem_dc:
            gdi32.DeleteDC(self._mem_dc)
            self._mem_dc = None
        screen = user32.GetDC(0)
        self._mem_dc = gdi32.CreateCompatibleDC(screen)
        self._bi.bmiHeader.biWidth = w
        self._bi.bmiHeader.biHeight = -h
        self._ptr = ctypes.c_void_p()
        self._dib = gdi32.CreateDIBSection(screen, ctypes.byref(self._bi), 0,
                                           ctypes.byref(self._ptr), None, 0)
        gdi32.SelectObject(self._mem_dc, self._dib)
        user32.ReleaseDC(0, screen)
        self._size = (w, h)
        if self._dib and self._ptr:
            try:
                self._view = np.ctypeslib.as_array(
                    (ctypes.c_ubyte * (w * h * 4)).from_address(self._ptr)
                ).reshape(h, w, 4)
            except Exception:
                self._view = None

    def move(self, x: int, y: int, w: int, h: int) -> None:
        self._ensure(w, h)
        self._pos = (x, y)

    # -- 贴一帧
    def paint(self, rgba: np.ndarray) -> None:
        h, w = rgba.shape[:2]
        if self._size != (w, h):
            self._ensure(w, h)
        view = self._view
        if view is not None:
            # 快路径：清零后只处理非透明像素，预乘用 64K 查表，直写 DIB
            view[:] = 0
            alphas = rgba[:, :, 3]
            ys, xs = np.nonzero(alphas)
            if len(ys):
                a = alphas[ys, xs].astype(np.uint32)
                base = a << 8
                px = rgba[ys, xs].astype(np.uint32)
                view[ys, xs, 0] = _PREMA[base | (px[:, 2] & 255)]   # B <- R
                view[ys, xs, 1] = _PREMA[base | (px[:, 1] & 255)]   # G
                view[ys, xs, 2] = _PREMA[base | (px[:, 0] & 255)]   # R <- B
                view[ys, xs, 3] = a.astype(np.uint8)                # A
        else:
            # 兜底：整帧预乘（BGRA）
            a = rgba[:, :, 3:4].astype(np.uint16)
            out = np.empty((h, w, 4), np.uint8)
            out[:, :, 0] = (rgba[:, :, 2].astype(np.uint16) * a[:, :, 0] // 255)
            out[:, :, 1] = (rgba[:, :, 1].astype(np.uint16) * a[:, :, 0] // 255)
            out[:, :, 2] = (rgba[:, :, 0].astype(np.uint16) * a[:, :, 0] // 255)
            out[:, :, 3] = a[:, :, 0]
            ctypes.memmove(self._ptr, out.tobytes(), w * h * 4)

        screen = user32.GetDC(0)
        src = wintypes.POINT(0, 0)
        dst = wintypes.POINT(self._pos[0], self._pos[1])
        size = wintypes.SIZE(w, h)
        bf = BLENDFUNCTION(0, 0, 255, 1)
        user32.UpdateLayeredWindow(self.hwnd, screen, ctypes.byref(dst), ctypes.byref(size),
                                   self._mem_dc, ctypes.byref(src), 0, ctypes.byref(bf), 2)
        user32.ReleaseDC(0, screen)
        if not self.visible:
            self.show()

    def show(self) -> None:
        user32.SetWindowPos(self.hwnd, -1, 0, 0, 0, 0,
                            SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE | SWP_SHOWWINDOW)
        self.visible = True

    def hide(self) -> None:
        user32.ShowWindow(self.hwnd, 0)   # SW_HIDE
        self.visible = False

    def close(self) -> None:
        if self.hwnd:
            user32.DestroyWindow(self.hwnd)
            self.hwnd = 0


# ---------------------------------------------------------------- 绘制
# 预乘 alpha 查表：_PREMA[(a<<8)|v] = a*v//255，省掉每帧几百万次 uint16 乘法
_IX = np.arange(65536, dtype=np.uint32)
_PREMA = ((_IX >> 8) * (_IX & 0xFF) // 255).astype(np.uint8)

TERRAIN_COLOR = {
    "ground": (52, 226, 138, 255),
    "slope": (188, 132, 255, 255),
    "ladder": (255, 150, 60, 255),
    "rope": (250, 204, 60, 255),
}
MONSTER_PALETTE = [
    (255, 82, 82), (255, 138, 61), (255, 205, 64), (120, 230, 130),
    (64, 210, 240), (96, 143, 255), (200, 120, 255), (255, 120, 200),
]

_FONTS: dict[int, object] = {}


def font(size: int):
    if size in _FONTS:
        return _FONTS[size]
    f = None
    for cand in ("msyh.ttc", "C:/Windows/Fonts/msyh.ttc", "simhei.ttf", "consola.ttf"):
        try:
            f = ImageFont.truetype(cand, size)
            break
        except Exception:
            continue
    if f is None:
        f = ImageFont.load_default()
    _FONTS[size] = f
    return f


_COLOR_MEMO: dict[str, tuple[int, int, int, int]] = {}


def monster_color(name: str) -> tuple[int, int, int, int]:
    c = _COLOR_MEMO.get(name)
    if c is not None:
        return c
    h = int.from_bytes(hashlib.md5(name.encode()).digest()[:2], "big") / 65535
    r, g, b = MONSTER_PALETTE[int(h * len(MONSTER_PALETTE)) % len(MONSTER_PALETTE)]
    c = (r, g, b, 255)
    _COLOR_MEMO[name] = c
    return c


def label_text(d: dict) -> str:
    return f"{d['label']} {d['conf']:.2f}"


# ---------------------------------------------------------------- 复用画布
class Canvas:
    """跨帧复用的 RGBA 画布。

    优先让 PIL 直接画进 numpy 共享内存（frombuffer 零拷贝），
    不支持时退回传统的 Image.new + np.asarray（每帧两次 4MB 拷贝）。
    """

    def __init__(self, w: int, h: int) -> None:
        self.w, self.h = w, h
        self.shared = False
        self.buf = np.zeros((h, w, 4), np.uint8)
        try:
            img = Image.frombuffer("RGBA", (w, h), self.buf, "raw", "RGBA", 0, 1)
            img.readonly = 0
            # 探针：画一个像素看有没有真的写进共享内存
            ImageDraw.Draw(img).point((0, 0), fill=(255, 0, 0, 255))
            if self.buf[0, 0, 0] == 255:
                self.img = img
                self.shared = True
                return
        except Exception:
            pass
        self.img = Image.new("RGBA", (w, h), (0, 0, 0, 0))

    def clear(self) -> None:
        if self.shared:
            self.buf[:] = 0
        else:
            self.img.paste((0, 0, 0, 0), (0, 0, self.w, self.h))

    def array(self) -> np.ndarray:
        return self.buf if self.shared else np.asarray(self.img)


_canvas_obj: Canvas | None = None
_canvas_key = (0, 0)


def get_canvas(w: int, h: int) -> Canvas:
    global _canvas_obj, _canvas_key
    if _canvas_obj is None or _canvas_key != (w, h):
        _canvas_obj = Canvas(w, h)
        _canvas_key = (w, h)
    return _canvas_obj


# ---------------------------------------------------------------- 插值追踪
class Track:
    __slots__ = ("name", "label", "conf", "terrain",
                 "cx", "cy", "bw", "bh", "vx", "vy", "t", "seen",
                 "rcx", "rcy", "rbw", "rbh", "alpha")


class Tracker:
    """检测框跨帧追踪 + 插值。

    推理线程 update()（约 --fps 次/秒），渲染线程 render()（约 --rfps 次/秒）：
    同名框按中心点贪心配对 → 估计速度（EMA）→ 航位推算外推一小段 →
    渲染端指数平滑。效果：框连续滑动，出现/消失有淡入淡出，而不是跳格。
    """

    DEAD_S = 0.45       # 丢失多久算真消失（淡出后删除）
    EXTRAP_S = 0.25     # 航位推算最多外推的时间
    SMOOTH_K = 16.0     # 渲染端平滑系数（越大跟得越紧、越小越顺滑）
    VMAX = 1500.0       # 速度上限 px/s，防误配对把框甩飞

    def __init__(self) -> None:
        self.tracks: list[Track] = []
        self.lock = threading.Lock()
        self.last_t = 0.0

    def update(self, dets: list[dict], now: float) -> None:
        with self.lock:
            self.last_t = now
            # 1) 候选配对：同名 + 中心距离在阈值内
            pairs: list[tuple[float, Track, dict, float, float, float, float]] = []
            for d in dets:
                x0, y0, x1, y1 = d["box"]
                cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
                bw, bh = x1 - x0, y1 - y0
                for tr in self.tracks:
                    if tr.name != d["name"]:
                        continue
                    age = now - tr.t
                    if age > 1.0:
                        continue
                    dist = math.hypot(tr.cx - cx, tr.cy - cy)
                    limit = 100.0 + math.hypot(tr.vx, tr.vy) * age * 1.5
                    if dist <= limit:
                        pairs.append((dist, tr, d, cx, cy, bw, bh))
            pairs.sort(key=lambda p: p[0])
            used_tr: set[int] = set()
            used_d: set[int] = set()
            for dist, tr, d, cx, cy, bw, bh in pairs:
                if id(tr) in used_tr or id(d) in used_d:
                    continue
                used_tr.add(id(tr))
                used_d.add(id(d))
                dt = max(1e-3, now - tr.t)
                if not tr.terrain and tr.seen > 0:
                    f = 0.5
                    tr.vx = tr.vx * (1 - f) + (cx - tr.cx) / dt * f
                    tr.vy = tr.vy * (1 - f) + (cy - tr.cy) / dt * f
                    sp = math.hypot(tr.vx, tr.vy)
                    if sp > self.VMAX:
                        tr.vx *= self.VMAX / sp
                        tr.vy *= self.VMAX / sp
                tr.cx, tr.cy, tr.bw, tr.bh = cx, cy, bw, bh
                tr.t = now
                tr.conf = d["conf"]
                tr.seen += 1
            # 2) 没配上对的检测 → 新轨迹
            for d in dets:
                if id(d) in used_d:
                    continue
                x0, y0, x1, y1 = d["box"]
                tr = Track()
                tr.name, tr.label, tr.conf = d["name"], d["label"], d["conf"]
                tr.terrain = d["name"] in TERRAIN_COLOR
                tr.cx, tr.cy = (x0 + x1) / 2, (y0 + y1) / 2
                tr.bw, tr.bh = x1 - x0, y1 - y0
                tr.vx = tr.vy = 0.0
                tr.t = now
                tr.seen = 1
                tr.rcx, tr.rcy, tr.rbw, tr.rbh = tr.cx, tr.cy, tr.bw, tr.bh
                tr.alpha = 0.0
                self.tracks.append(tr)
            # 3) 丢的轨迹留在列表里，由 render() 负责淡出后删除

    def render(self, now: float, dt: float, frozen: bool = False) -> list[dict]:
        """返回插值后的检测框列表（渲染帧率下调用）。frozen=暂停时冻结画面。"""
        out: list[dict] = []
        k = self.SMOOTH_K
        with self.lock:
            alive: list[Track] = []
            for tr in self.tracks:
                age = (self.last_t if frozen else now) - tr.t
                if not frozen and age > self.DEAD_S * 2:
                    continue                        # 丢太久，直接删
                # 目标位置：航位推算（地形不动，不外推）
                ea = 0.0 if (tr.terrain or frozen) else min(age, self.EXTRAP_S)
                tcx = tr.cx + tr.vx * ea
                tcy = tr.cy + tr.vy * ea
                s = 1.0 - math.exp(-dt * k)
                tr.rcx += (tcx - tr.rcx) * s
                tr.rcy += (tcy - tr.rcy) * s
                tr.rbw += (tr.bw - tr.rbw) * s
                tr.rbh += (tr.bh - tr.rbh) * s
                ta = 0.0 if age > 0.25 else 1.0
                tr.alpha += (ta - tr.alpha) * (1.0 - math.exp(-dt * 12.0))
                if not frozen and age > self.DEAD_S and tr.alpha < 0.04:
                    continue                        # 已淡出，删除
                alive.append(tr)
                a = max(0.0, min(1.0, tr.alpha))
                if a < 0.03:
                    continue
                x0 = tr.rcx - tr.rbw / 2
                y0 = tr.rcy - tr.rbh / 2
                out.append({
                    "name": tr.name, "label": tr.label, "conf": tr.conf,
                    "box": [x0, y0, x0 + tr.rbw, y0 + tr.rbh],
                    "alpha": a,
                })
            self.tracks = alive
        return out


def render(dets: list[dict], w: int, h: int, opt: dict) -> np.ndarray:
    """把（插值后的）检测结果画成 RGBA 画布（透明底，画布跨帧复用）。"""
    canvas = get_canvas(w, h)
    canvas.clear()
    dr = ImageDraw.Draw(canvas.img)
    lw = max(2, int(opt.get("line_width", 3)))
    show_labels = opt.get("labels", True)
    show_ground = opt.get("ground", True)
    show_targets = opt.get("targets", True)
    f = font(13)

    def line(p0, p1, color, width):
        # 先垫一层黑色描边，保证在任何地形底色上都看得清
        dr.line([p0, p1], fill=(0, 0, 0, int(color[3] * 110 // 255)), width=width + 2)
        dr.line([p0, p1], fill=color, width=width)

    def dashed(p0, p1, color, width, dash=10, gap=7):
        x0, y0 = p0
        x1, y1 = p1
        dist = math.hypot(x1 - x0, y1 - y0)
        if dist < 1:
            return
        ux, uy = (x1 - x0) / dist, (y1 - y0) / dist
        t = 0.0
        while t < dist:
            a = t
            b = min(t + dash, dist)
            line((x0 + ux * a, y0 + uy * a), (x0 + ux * b, y0 + uy * b), color, width)
            t += dash + gap

    for d in dets:
        name = d["name"]
        x0, y0, x1, y1 = d["box"]
        bw, bh = x1 - x0, max(1.0, y1 - y0)
        terrain = name in TERRAIN_COLOR
        alpha = d.get("alpha", 1.0)

        if terrain and name in ("ground", "slope"):
            if not show_ground:
                continue
            if bw < 10:                     # 太碎的段不画，免得满屏小短线
                continue
            r, g, b, _ = TERRAIN_COLOR[name]
            color = (r, g, b, int(255 * alpha))
            # 站立线 = 框的上边（wztool 的地形标注从站立线往下盖 32px）
            if name == "ground":
                line((x0, y0), (x1, y0), color, lw)
            else:
                dashed((x0, y0), (x1, y0), color, lw)
            continue

        if terrain and name in ("rope", "ladder"):
            if not show_ground:
                continue
            cx = (x0 + x1) / 2
            r, g, b, _ = TERRAIN_COLOR[name]
            color = (r, g, b, int(255 * alpha))
            line((cx, y0), (cx, y1), color, lw)
            # 两端加个小横杠，标出可攀爬范围
            cap = 5
            line((cx - cap, y0), (cx + cap, y0), color, max(2, lw - 1))
            line((cx - cap, y1), (cx + cap, y1), color, max(2, lw - 1))
            continue

        if not show_targets:
            continue
        r, g, b, _ = monster_color(name)
        ca = int(255 * alpha)
        dr.rectangle([x0, y0, x1, y1], outline=(0, 0, 0, int(120 * alpha)), width=lw + 2)
        dr.rectangle([x0, y0, x1, y1], outline=(r, g, b, ca), width=lw)
        if not show_labels:
            continue
        text = label_text(d)
        try:
            tw = int(dr.textlength(text, font=f))
        except Exception:
            tw = len(text) * 7
        th = 17
        tx = min(max(0, x0), max(0, w - tw - 8))
        ty = y0 - th - 2
        if ty < 0:
            ty = min(h - th, y0 + 2)
        dr.rectangle([tx, ty, tx + tw + 8, ty + th], fill=(0, 0, 0, int(190 * alpha)))
        dr.rectangle([tx, ty, tx + 2, ty + th], fill=(r, g, b, ca))
        dr.text((tx + 6, ty + 1), text, fill=(255, 255, 255, int(235 * alpha)), font=f)

    if opt.get("hud", True):
        send_hud(dr, w, h, opt)
    return canvas.array()


def send_hud(dr: ImageDraw.ImageDraw, w: int, h: int, opt: dict) -> None:
    stats = opt.get("_stats") or {}
    text = stats.get("text", "")
    if not text:
        return
    f = font(13)
    try:
        tw = int(dr.textlength(text, font=f))
    except Exception:
        tw = len(text) * 7
    x = max(8, (w - tw) // 2 - 12)
    y = 6
    dr.rounded_rectangle([x, y, x + tw + 24, y + 24], radius=8, fill=(0, 0, 0, 150))
    dr.text((x + 12, y + 4), text, fill=(255, 255, 255, 235), font=f)


# ---------------------------------------------------------------- 主程序
class App:
    # action -> 候选虚拟键（前一个被占用就试下一个）
    HOTKEYS = {
        1: ("toggle", [0x48]),            # H
        2: ("ground", [0x47]),            # G
        3: ("labels", [0x4B, 0x4C]),      # K（L 常被别的软件占用）
        4: ("pause", [0x4F, 0x50, 0x77]), # O / P / F8
        5: ("quit", [0x51]),              # Q
        6: ("conf_up", [0xDD]),           # ]
        7: ("conf_down", [0xDB]),         # [
        8: ("size", [0x32]),              # 2
    }
    KEYNAME = {0x48: "H", 0x47: "G", 0x4B: "K", 0x4C: "L", 0x4F: "O", 0x50: "P",
               0x77: "F8", 0x51: "Q", 0xDD: "]", 0xDB: "[", 0x32: "2"}

    def __init__(self, args):
        self.args = args
        self.opt = {
            "line_width": args.line_width,
            "labels": not args.no_labels,
            "ground": not args.no_ground,
            "targets": not args.no_targets,
            "hud": not args.no_hud,
            "_stats": {"text": ""},
        }
        self.conf = args.conf
        self.iou = args.iou
        self.imgsz = args.imgsz
        self.paused = False
        self.running = True
        self.hwnd = 0
        self._last_log = 0.0
        self._warned_min = False
        self.tracker = Tracker()
        self._inf_stats: dict = {}          # 推理线程写，渲染线程读
        self._game_active = True            # 渲染线程写，推理线程读

    # -- 初始化
    def setup(self) -> bool:
        path = pick_default_model(self.args.model)
        if not path:
            print("[x] 没找到权重：把 best.pt 放到 yololab\\models\\ 或 train\\runs\\<round>\\weights\\")
            return False
        print(f"[*] 权重 {path}")
        self.model = ModelBox(path, device=self.args.device)
        print(f"[*] {len(self.model.names)} 类 · 加载 {self.model.load_ms} ms")

        self.hwnd = find_game_window(self.args.window)
        if not self.hwnd:
            print("[x] 没找到游戏窗口（标题含 冒险岛 / maple）")
            print("    可用 --window 指定标题关键词，或 --list 看看有哪些窗口")
            return False
        n = user32.GetWindowTextLengthW(self.hwnd)
        buf = ctypes.create_unicode_buffer(n + 1)
        user32.GetWindowTextW(self.hwnd, buf, n + 1)
        self.title = buf.value
        print(f"[*] 游戏窗口「{self.title}」 hwnd={self.hwnd}")

        self.overlay = OverlayWindow()
        self.overlay.on_hotkey = self._on_hotkey
        self.cap = WindowCapture()
        self.active_keys: dict[int, str] = {}
        for hid, (name, vks) in self.HOTKEYS.items():
            for vk in vks:
                if user32.RegisterHotKey(self.overlay.hwnd, hid, 0x0002 | 0x0001, vk):
                    self.active_keys[hid] = self.KEYNAME.get(vk, str(vk))
                    break
            else:
                print(f"[!] 热键注册失败: {name}（试试用控制台命令）")
        keys = " · ".join(f"{self.KEYNAME.get(v, v)}={self.HOTKEYS[h][0]}"
                          for h, v in self.active_keys.items())
        print(f"[*] 热键 Ctrl+Alt+[{keys}]")
        return True

    # -- 热键
    def _on_hotkey(self, hid: int) -> None:
        act = self.HOTKEYS.get(hid, ("", []))[0]
        if act == "toggle":
            if self.overlay.visible:
                self.overlay.hide()
                print("[*] 遮罩已隐藏（Ctrl+Alt+H 恢复）")
            else:
                self.overlay.show()
                print("[*] 遮罩已显示")
        elif act == "ground":
            self.opt["ground"] = not self.opt["ground"]
            print(f"[*] 地形线 {'开' if self.opt['ground'] else '关'}")
        elif act == "labels":
            self.opt["labels"] = not self.opt["labels"]
            print(f"[*] 标签 {'开' if self.opt['labels'] else '关'}")
        elif act == "pause":
            self.paused = not self.paused
            print(f"[*] {'暂停' if self.paused else '继续'}")
        elif act == "quit":
            self.running = False
        elif act == "conf_up":
            self.conf = round(min(0.9, self.conf + 0.05), 2)
            print(f"[*] 置信度 {self.conf}")
        elif act == "conf_down":
            self.conf = round(max(0.05, self.conf - 0.05), 2)
            print(f"[*] 置信度 {self.conf}")
        elif act == "size":
            self.imgsz = 1152 if self.imgsz == 960 else (1366 if self.imgsz == 1152 else 960)
            print(f"[*] 推理尺寸 {self.imgsz}")

    # -- 控制台
    def console(self) -> None:
        while self.running:
            try:
                line = sys.stdin.readline()
            except Exception:
                return
            if not line:
                time.sleep(0.3)
                continue
            cmd = line.strip().lower()
            if not cmd:
                continue
            if cmd in ("q", "quit", "exit"):
                self.running = False
                return
            elif cmd in ("h", "hide"):
                self._on_hotkey(1)
            elif cmd in ("g", "ground"):
                self._on_hotkey(2)
            elif cmd in ("k", "l", "labels"):
                self._on_hotkey(3)
            elif cmd in ("o", "p", "pause"):
                self._on_hotkey(4)
            elif cmd.startswith("c ") or cmd.startswith("conf "):
                try:
                    self.conf = max(0.05, min(0.95, float(cmd.split()[1])))
                    print(f"[*] 置信度 {self.conf}")
                except Exception:
                    print("[!] 用法: c 0.5")
            elif cmd.startswith("i ") or cmd.startswith("imgsz "):
                try:
                    self.imgsz = int(cmd.split()[1])
                    print(f"[*] 推理尺寸 {self.imgsz}")
                except Exception:
                    print("[!] 用法: i 1366")
            elif cmd in ("?", "help", "h?"):
                print("  q 退出 | h 隐藏 | g 地面线 | k 标签 | o 暂停 | c 0.5 | i 1366")
            else:
                print("[!] 未知命令（? 看帮助）")

    # -- 主循环
    def run(self) -> int:
        if not self.setup():
            return 2
        threading.Thread(target=self.console, daemon=True).start()
        threading.Thread(target=self._infer_loop, daemon=True).start()
        try:
            self._render_loop()
        except KeyboardInterrupt:
            pass
        finally:
            self.shutdown()
        return 0

    # -- 推理线程：抓帧 → 推理 → 更新追踪（帧率 = --fps）
    def _infer_loop(self) -> None:
        target_dt = 1.0 / max(1.0, self.args.fps)
        times: list[float] = []
        while self.running:
            if self.paused or not self._game_active:
                times.clear()
                time.sleep(0.15)
                continue
            t0 = time.perf_counter()
            frame = self.cap.grab(self.hwnd)
            if frame is None or frame.size == 0:
                time.sleep(0.05)
                continue
            dets, ms = self.model.predict(frame, self.conf, self.iou, self.imgsz)
            now = time.perf_counter()
            self.tracker.update(dets, now)
            times.append(now)
            if len(times) > 15:
                times.pop(0)
            fps = 0.0
            if len(times) > 1:
                span = times[-1] - times[0]
                fps = (len(times) - 1) / span if span > 0 else 0
            n_mon = sum(1 for d in dets if d["name"] not in TERRAIN_COLOR)
            self._inf_stats = {"fps": fps, "ms": ms,
                               "mon": n_mon, "terr": len(dets) - n_mon}
            dt = time.perf_counter() - t0
            if dt < target_dt:
                time.sleep(min(target_dt - dt, 0.2))

    # -- 渲染线程（主线程）：插值 → 画框 → 贴窗（帧率 = --rfps）
    def _render_loop(self) -> None:
        target_dt = 1.0 / max(1.0, self.args.rfps)
        rect_prev = None
        last_t = time.perf_counter()
        rtimes: list[float] = []
        while self.running:
            # 处理热键消息
            msg = wintypes.MSG()
            while user32.PeekMessageW(ctypes.byref(msg), 0, 0, 0, 1):
                user32.TranslateMessage(ctypes.byref(msg))
                user32.DispatchMessageW(ctypes.byref(msg))

            t0 = time.perf_counter()
            rect = client_rect(self.hwnd)
            if not rect:
                self._game_active = False
                if self.overlay.visible:
                    self.overlay.hide()
                if not self._warned_min:
                    self._warned_min = True
                    print("[!] 游戏窗口被最小化了，恢复窗口后会自动开始绘制")
                time.sleep(0.25)
                continue
            if self._warned_min:
                self._warned_min = False
                print("[*] 游戏窗口已恢复，继续绘制")
            # 游戏既不在前台、又被别的窗口盖住时收起遮罩，免得线浮在别人窗口上
            if not self.args.always_on:
                fg = int(user32.GetForegroundWindow() or 0)
                if fg != self.hwnd and not window_on_top_at(self.hwnd, rect):
                    self._game_active = False
                    if self.overlay.visible:
                        self.overlay.hide()
                    time.sleep(0.2)
                    continue
            self._game_active = True
            if rect != rect_prev:
                self.overlay.move(*rect)
                rect_prev = rect
                if not self.overlay.visible:
                    self.overlay.show()

            now = time.perf_counter()
            dt = min(0.1, max(1e-3, now - last_t))
            last_t = now
            dets = self.tracker.render(now, dt, frozen=self.paused)
            _, _, w, h = rect

            rtimes.append(now)
            if len(rtimes) > 40:
                rtimes.pop(0)
            rfps = 0.0
            if len(rtimes) > 1:
                span = rtimes[-1] - rtimes[0]
                rfps = (len(rtimes) - 1) / span if span > 0 else 0

            st = self._inf_stats
            self.opt["_stats"] = {
                "text": f"{st.get('fps', 0):4.1f} fps · {st.get('ms', 0):3.0f} ms · "
                        f"怪 {st.get('mon', 0)} · 地形 {st.get('terr', 0)} · conf {self.conf:.2f}"
                        + ("  [暂停]" if self.paused else "")
            }
            self.overlay.paint(render(dets, w, h, self.opt))
            if now - self._last_log > 5:
                self._last_log = now
                print(f"[{time.strftime('%H:%M:%S')}] 推理 {st.get('fps', 0):4.1f} fps · "
                      f"{st.get('ms', 0):3.0f} ms · 渲染 {rfps:4.1f} fps · "
                      f"怪 {st.get('mon', 0)} · 地形 {st.get('terr', 0)} · conf {self.conf:.2f}")

            dt_loop = time.perf_counter() - t0
            if dt_loop < target_dt:
                time.sleep(min(target_dt - dt_loop, 0.2))

    def shutdown(self) -> None:
        self.running = False
        try:
            for hid in self.HOTKEYS:
                user32.UnregisterHotKey(self.overlay.hwnd, hid)
            self.overlay.close()
            self.cap.close()
        except Exception:
            pass
        print("[*] 已退出")


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="YOLO 游戏遮罩层（实时标注在游戏画面上）")
    ap.add_argument("--model", help="权重路径；缺省自动挑最新 best.pt")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--window", default=None, help="游戏窗口标题关键词，默认 冒险岛/maple")
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--imgsz", type=int, default=1366,
                    help="推理尺寸，默认取景框原生宽度 1366（对齐后 1376）")
    ap.add_argument("--fps", type=float, default=12.0, help="推理帧率上限（框的更新率）")
    ap.add_argument("--rfps", type=float, default=30.0, help="渲染帧率上限（插值后的框刷新率）")
    ap.add_argument("--line-width", type=int, default=3)
    ap.add_argument("--no-labels", action="store_true", help="不画名称/置信度")
    ap.add_argument("--no-ground", action="store_true", help="不画地形线")
    ap.add_argument("--no-targets", action="store_true", help="不画怪物框")
    ap.add_argument("--no-hud", action="store_true", help="不画左上角状态条")
    ap.add_argument("--always-on", action="store_true", help="游戏不在前台时也继续显示")
    ap.add_argument("--threads", type=int, default=0, help="限制 torch CPU 线程数")
    ap.add_argument("--list", action="store_true", help="列出可见窗口后退出")
    a = ap.parse_args()

    enable_dpi_awareness()
    if a.list:
        print("可见窗口：")
        for row in list_windows():
            print(row)
        return 0
    if a.threads > 0:
        try:
            import torch

            torch.set_num_threads(a.threads)
        except Exception:
            pass
    return App(a).run()


if __name__ == "__main__":
    sys.exit(main())
