#!/usr/bin/env python3
"""
CNN Figure Builder — PlotNeuralNet-style drag-and-drop diagram tool.
3-D blocks with adjustable fill opacity, grid (solid visible / dashed hidden edges), teal arrows, Picture layers.

Run:   python cnn_figure_builder.py
Extra: pip install Pillow   (PNG export + Picture layers)
       AI generate: pip install groq — put GROQ_API_KEY in .env (see .env.example). Optional: GROQ_MODEL=...
"""

import json
import os
import re
import sys
import threading

import tkinter as tk

try:
    from groq import Groq
except ImportError:
    Groq = None
from tkinter import ttk, colorchooser, messagebox, filedialog

# ── Default layer palette (PlotNeuralNet-inspired) ─────────────────────────────

WHEAT    = "#F5DEB3"
RED_POOL = "#CD5C5C"

LAYER_DEFAULTS = {
    "Input":      {"color": WHEAT,     "w":  6, "h": 90, "d": 90, "label": "Input"},
    "Picture":    {"color": "#A6ADC8", "w": 110, "h": 130, "d": 0, "label": ""},
    "Text":       {"color": "#313244", "w": 200, "h": 48, "d": 0, "label": "Caption"},
    "Conv2D":     {"color": WHEAT,     "w": 28, "h": 80, "d": 22, "label": "Conv2D"},
    "Conv3D":     {"color": WHEAT,     "w": 28, "h": 80, "d": 22, "label": "Conv3D"},
    "MaxPool":    {"color": RED_POOL,  "w":  9, "h": 62, "d": 14, "label": "MaxPool"},
    "AvgPool":    {"color": "#B03A2E", "w":  9, "h": 55, "d": 13, "label": "AvgPool"},
    "GlobalPool": {"color": WHEAT,     "w":  6, "h": 70, "d": 14, "label": "GlobalPool"},
    "BatchNorm":  {"color": "#AED6F1", "w":  6, "h": 72, "d": 22, "label": "BN"},
    "Dropout":    {"color": "#D7BDE2", "w":  6, "h": 62, "d": 22, "label": "Dropout 0.5"},
    "Flatten":    {"color": "#D5D8DC", "w":  3, "h": 60, "d": 60, "label": "Flatten"},
    "Dense":      {"color": WHEAT,     "w": 10, "h": 45, "d": 45, "label": "Dense 64"},
    "Output":     {"color": WHEAT,     "w":  6, "h": 28, "d": 28, "label": "Output"},
}

# Colours
CANVAS_LIGHT = "#FFFFFF"
CANVAS_DARK  = "#1E1E2E"
APP_BG  = "#252535"
PAL_BG  = "#1A1A2A"
PROP_BG = "#1A1A2A"
FG      = "#CDD6F4"
ACCENT  = "#89B4FA"
SEL_CLR = "#F5C2E7"
PERSP   = 0.45   # 3-D perspective factor

_APP_DIR = os.path.dirname(os.path.abspath(__file__))
_DOTENV_PATH = os.path.join(_APP_DIR, ".env")

# Groq — use official SDK (avoids HTTP 403 / Cloudflare 1010 on raw urllib). Override model in .env.
GROQ_MODEL_DEFAULT = "openai/gpt-oss-120b"

# After AI JSON: minimum horizontal distance between consecutive block centers (world x).
AI_LAYER_MIN_CENTER_GAP = 136

# AI-generated figures: enforced w/h/d (world units) for a consistent look.
AI_DEFAULT_H = 100
AI_DEFAULT_W = 70
AI_THIN_W = 20
AI_DEPTH_INPUT_CONV = 90
AI_DEPTH_OTHER_3D = 95

_LAYER_TYPES_DOC = ", ".join(f'"{k}"' for k in LAYER_DEFAULTS.keys())

_AI_PALETTE_LINES = "\n".join(
    f'  - "{lt}": color EXACTLY "{cfg["color"]}".'
    for lt, cfg in LAYER_DEFAULTS.items()
)

AI_FIGURE_SYSTEM_PROMPT = f"""You output ONLY valid JSON for a CNN diagram editor. No markdown fences, no explanation before or after.

Required root object:
{{
  "version": 1,
  "connections": [[from_layer_id, to_layer_id], ...],
  "layers": [ ... ]
}}

- "connections": optional directed edges as [source_id, target_id] using layer ids. Use [] to let the app auto-chain left-to-right by x.
- Each layer MUST have: "id" (unique positive int), "layer_type" (one of: {_LAYER_TYPES_DOC}), "x", "y", "w", "h", "d", "color", "label", "info", "opacity" (0-1).

GEOMETRY (critical — the app will normalize; match these targets):
- Height h = 100 for every 3D block (except Picture/Text use their usual frame sizes).
- Thickness w = 70 for Input, Conv2D, Conv3D, Dense, Output, BatchNorm, Dropout, GlobalPool.
- Thickness w = 20 for MaxPool, AvgPool, and Flatten (same thin width for all three).
- Depth d = 90 ONLY for Input, Conv2D, Conv3D.
- Depth d = 95 for all other 3D types (MaxPool, AvgPool, GlobalPool, BatchNorm, Dropout, Flatten, Dense, Output).
- Picture and Text: d = 0.
- Set "info" to empty string "" for every layer. Put all short text in "label" only (no secondary info lines).

COLORS (critical — use the app palette only, exact hex):
{_AI_PALETTE_LINES}
- Set "color" on each layer to the exact string above for that layer_type. Do not use random or alternate colors.
- Omit per-face color overrides (color_front, color_top, color_right) or set them to "".

SPACING (critical — blocks must not look crowded):
- Place layers in a clear horizontal chain with VISIBLE GAPS between blocks (not packed together).
- Consecutive block centers on x should be roughly 140–200 apart (typical 150–180). Wider blocks need larger steps so silhouettes do not overlap.
- Start near x=100–140 for the first block and increase x strongly for each next layer.

Layout: y around 320–380. ids 1,2,3,… in flow order.
Include "image_path": "", "image_rotation": 0, "image_tilt": 0 for 3D blocks unless Picture.

Match the user's architecture (layer types, order). Keep info empty; use label for short names only."""


def load_dotenv(path=None):
    """Load KEY=value lines from .env into os.environ (values in .env override existing)."""
    path = path or _DOTENV_PATH
    if not os.path.isfile(path):
        return
    try:
        with open(path, "r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("export "):
                    line = line[7:].strip()
                if "=" not in line:
                    continue
                key, _, val = line.partition("=")
                key = key.strip()
                val = val.strip()
                if (val.startswith('"') and val.endswith('"')) or (
                    val.startswith("'") and val.endswith("'")
                ):
                    val = val[1:-1]
                if key:
                    os.environ[key] = val
    except OSError:
        pass


def normalize_ai_layers_to_app_defaults(data):
    """After Groq returns JSON: palette colors, fixed AI w/h/d, empty info, clear per-face colors."""
    if not isinstance(data, dict):
        return
    layers = data.get("layers")
    if not isinstance(layers, list):
        return
    for d in layers:
        if not isinstance(d, dict):
            continue
        lt = d.get("layer_type")
        if lt not in LAYER_DEFAULTS:
            continue
        defs = LAYER_DEFAULTS[lt]
        d["color"] = defs["color"]
        d["info"] = ""
        for k in ("color_front", "color_top", "color_right"):
            d[k] = ""
        if lt in ("Picture", "Text"):
            d["d"] = 0
            d["w"] = int(defs["w"])
            d["h"] = int(defs["h"])
        else:
            d["h"] = AI_DEFAULT_H
            if lt in ("MaxPool", "AvgPool", "Flatten"):
                d["w"] = AI_THIN_W
            else:
                d["w"] = AI_DEFAULT_W
            if lt in ("Input", "Conv2D", "Conv3D"):
                d["d"] = AI_DEPTH_INPUT_CONV
            else:
                d["d"] = AI_DEPTH_OTHER_3D

    _spread_ai_layer_x_positions(layers)


def _spread_ai_layer_x_positions(layers):
    """Enforce minimum center-to-center x gap so blocks are visually separated."""
    items = [d for d in layers if isinstance(d, dict) and d.get("layer_type") in LAYER_DEFAULTS]
    if len(items) < 2:
        return
    try:
        keyed = [(float(d.get("x", 0)), d) for d in items]
    except (TypeError, ValueError):
        return
    keyed.sort(key=lambda t: t[0])
    _, first = keyed[0]
    prev_x = float(first.get("x", 120))
    first["x"] = prev_x
    gap = float(AI_LAYER_MIN_CENTER_GAP)
    for i in range(1, len(keyed)):
        _, d = keyed[i]
        try:
            cx = float(d.get("x", prev_x + gap))
        except (TypeError, ValueError):
            cx = prev_x + gap
        need = prev_x + gap
        if cx < need:
            cx = need
        d["x"] = cx
        prev_x = cx


# ── Colour helpers ─────────────────────────────────────────────────────────────

def _cl(v): return max(0, min(255, int(v)))

def lighten(c, f=0.25):
    r, g, b = int(c[1:3], 16), int(c[3:5], 16), int(c[5:7], 16)
    return "#{:02x}{:02x}{:02x}".format(
        _cl(r + (255-r)*f), _cl(g + (255-g)*f), _cl(b + (255-b)*f))

def darken(c, f=0.25):
    r, g, b = int(c[1:3], 16), int(c[3:5], 16), int(c[5:7], 16)
    return "#{:02x}{:02x}{:02x}".format(_cl(r*(1-f)), _cl(g*(1-f)), _cl(b*(1-f)))

def _lum(hex_c):
    r, g, b = int(hex_c[1:3], 16), int(hex_c[3:5], 16), int(hex_c[5:7], 16)
    return 0.299*r + 0.587*g + 0.114*b


def _hex_to_rgb(h):
    return (int(h[1:3], 16), int(h[3:5], 16), int(h[5:7], 16))


def blend_hex(fg_hex, bg_hex, opacity):
    """Simulate fg at opacity on top of bg (0 = invisible fg, 1 = solid fg)."""
    opacity = max(0.0, min(1.0, float(opacity)))
    fr, fg, fb = _hex_to_rgb(fg_hex)
    br, bg, bb = _hex_to_rgb(bg_hex)
    r = _cl(fr * opacity + br * (1.0 - opacity))
    g = _cl(fg * opacity + bg * (1.0 - opacity))
    b = _cl(fb * opacity + bb * (1.0 - opacity))
    return "#{:02x}{:02x}{:02x}".format(r, g, b)


# ── Layer data ─────────────────────────────────────────────────────────────────

class Layer:
    _ctr = 0

    def __init__(self, ltype, x, y):
        Layer._ctr += 1
        self.id         = Layer._ctr
        self.layer_type = ltype
        self.x, self.y  = x, y
        cfg             = LAYER_DEFAULTS[ltype]
        self.color      = cfg["color"]
        self.w          = cfg["w"]
        self.h          = cfg["h"]
        self.d          = cfg["d"]
        self.label      = cfg["label"]
        self.info       = ""
        self.opacity    = 0.7
        self.image_path = ""
        self.image_rotation = 0.0
        self.image_tilt = 0.0
        self.label_color = ""
        self.info_color = ""
        self.text_color = "#1A1E2E"
        self.text_bold = True
        self.text_font_size = 14
        if ltype == "Text":
            self.label_font_size = 14
            self.info_font_size = 12
        else:
            self.label_font_size = 9
            self.info_font_size = 8
        self.color_front = ""
        self.color_top = ""
        self.color_right = ""

    def to_dict(self):
        return {
            "id": self.id,
            "layer_type": self.layer_type,
            "x": self.x,
            "y": self.y,
            "color": self.color,
            "w": self.w,
            "h": self.h,
            "d": self.d,
            "label": self.label,
            "info": self.info,
            "opacity": float(getattr(self, "opacity", 0.7)),
            "image_path": getattr(self, "image_path", ""),
            "image_rotation": float(getattr(self, "image_rotation", 0.0)),
            "image_tilt": float(getattr(self, "image_tilt", 0.0)),
            "label_color": getattr(self, "label_color", ""),
            "info_color": getattr(self, "info_color", ""),
            "text_color": getattr(self, "text_color", "#1A1E2E"),
            "text_bold": bool(getattr(self, "text_bold", True)),
            "text_font_size": int(getattr(self, "text_font_size", 14)),
            "label_font_size": int(getattr(self, "label_font_size", 9)),
            "info_font_size": int(getattr(self, "info_font_size", 8)),
            "color_front": getattr(self, "color_front", "") or "",
            "color_top": getattr(self, "color_top", "") or "",
            "color_right": getattr(self, "color_right", "") or "",
        }

    @classmethod
    def from_dict(cls, d):
        obj = cls.__new__(cls)
        for k, v in d.items():
            setattr(obj, k, v)
        if not hasattr(obj, "info"):
            obj.info = ""
        if not hasattr(obj, "opacity"):
            obj.opacity = 0.7
        else:
            try:
                o = float(obj.opacity)
                if o > 1.0:
                    o /= 100.0
                obj.opacity = max(0.0, min(1.0, o))
            except (TypeError, ValueError):
                obj.opacity = 0.7
        if not hasattr(obj, "image_path"):
            obj.image_path = ""
        if not hasattr(obj, "image_rotation"):
            obj.image_rotation = 0.0
        else:
            try:
                obj.image_rotation = float(obj.image_rotation)
            except (TypeError, ValueError):
                obj.image_rotation = 0.0
        if not hasattr(obj, "image_tilt"):
            obj.image_tilt = 0.0
        else:
            try:
                obj.image_tilt = float(obj.image_tilt)
            except (TypeError, ValueError):
                obj.image_tilt = 0.0
        if not hasattr(obj, "label_color"):
            obj.label_color = ""
        if not hasattr(obj, "info_color"):
            obj.info_color = ""
        if not hasattr(obj, "text_color"):
            obj.text_color = "#1A1E2E"
        if not hasattr(obj, "text_bold"):
            obj.text_bold = True
        else:
            obj.text_bold = bool(obj.text_bold)
        if not hasattr(obj, "text_font_size"):
            obj.text_font_size = 14
        else:
            try:
                obj.text_font_size = int(obj.text_font_size)
            except (TypeError, ValueError):
                obj.text_font_size = 14
        if not hasattr(obj, "label_font_size"):
            if getattr(obj, "layer_type", "") == "Text":
                obj.label_font_size = int(getattr(obj, "text_font_size", 14))
            else:
                obj.label_font_size = 9
        else:
            try:
                obj.label_font_size = int(obj.label_font_size)
            except (TypeError, ValueError):
                obj.label_font_size = (
                    14 if getattr(obj, "layer_type", "") == "Text" else 9)
        if not hasattr(obj, "info_font_size"):
            if getattr(obj, "layer_type", "") == "Text":
                obj.info_font_size = max(5, int(getattr(obj, "label_font_size", 14)) - 2)
            else:
                obj.info_font_size = 8
        else:
            try:
                obj.info_font_size = int(obj.info_font_size)
            except (TypeError, ValueError):
                obj.info_font_size = (
                    12 if getattr(obj, "layer_type", "") == "Text" else 8)
        if getattr(obj, "layer_type", "") == "Text":
            obj.text_font_size = int(getattr(obj, "label_font_size", 14))
        for k in ("color_front", "color_top", "color_right"):
            if not hasattr(obj, k):
                setattr(obj, k, "")
            else:
                v = (getattr(obj, k) or "").strip()
                setattr(obj, k, v if isinstance(v, str) else "")
        return obj


# ── Application ────────────────────────────────────────────────────────────────

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        load_dotenv()
        self.title("CNN Figure Builder")
        self.geometry("1480x770")
        self.configure(bg=APP_BG)

        self.layers    = []
        self.selected  = None
        self._dragging = None
        self._pal_drag = None
        self._syncing  = False

        # Manual arrows: list of [from_layer_id, to_layer_id]. Empty → auto chain by x-order.
        self.manual_conns = []
        self._conn_pick_from = None  # first layer id when "Add arrow" mode is active
        self.selected_edge = None  # (from_id, to_id) when an arrow is selected; None otherwise

        # Toggles
        self._show_conn = True
        self._show_grid = True
        self._light_cv  = True     # white canvas by default
        self._add_arrow_mode = False
        self._photo_cache = {}

        # Canvas pan + zoom (world coords on layers)
        self._zoom = 1.0
        self._view_pan_x = 0
        self._view_pan_y = 0
        self._space_held = False
        self._canvas_panning = False
        self._pan_last_x = None
        self._pan_last_y = None

        self._build_ui()
        self.draw_all()

    # ── UI ─────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        # Toolbar
        tb = tk.Frame(self, bg="#0F0F1A", pady=5)
        tb.pack(side=tk.TOP, fill=tk.X)

        for text, cmd in (
            ("New",         self._new),
            ("Save",        self._save),
            ("Load",        self._load),
            ("Export EPS",  self._export_eps),
            ("Export PNG",  self._export_png),
            ("Auto Layout", self._auto_layout),
        ):
            self._tbtn(tb, text, cmd, side=tk.LEFT)

        ai_b = tk.Button(
            tb, text="AI generate…", command=self._open_ai_dialog,
            bg="#5B2C6F", fg="white", relief="flat",
            font=("Segoe UI", 9, "bold"), padx=10, pady=3,
            activebackground="#7D3C98", activeforeground="white")
        ai_b.pack(side=tk.LEFT, padx=6, pady=2)

        tk.Label(tb, text="CNN Figure Builder",
                 bg="#0F0F1A", fg=ACCENT,
                 font=("Segoe UI", 12, "bold")).pack(side=tk.LEFT, padx=14)

        self._tbtn(tb, "100%",         self._reset_view,     tk.RIGHT)
        self._tbtn(tb, "Zoom −",       self._zoom_out_center, tk.RIGHT)
        self._tbtn(tb, "Zoom +",       self._zoom_in_center, tk.RIGHT)
        self._bg_btn   = self._tbtn(tb, "Dark Canvas",    self._toggle_cv_bg,   tk.RIGHT)
        self._grid_btn = self._tbtn(tb, "Hide Grid",      self._toggle_grid,    tk.RIGHT)
        self._conn_btn = self._tbtn(tb, "Hide Arrows",    self._toggle_conn,    tk.RIGHT)
        self._arrow_mode_btn = self._tbtn(tb, "Add arrow (click A→B)", self._toggle_arrow_mode, tk.RIGHT)
        self._tbtn(tb, "Clear custom arrows", self._clear_manual_conns, tk.RIGHT)

        # Body
        body = tk.Frame(self, bg=APP_BG)
        body.pack(fill=tk.BOTH, expand=True)

        # Palette (left)
        pal = tk.Frame(body, bg=PAL_BG, width=150)
        pal.pack(side=tk.LEFT, fill=tk.Y)
        pal.pack_propagate(False)
        tk.Label(pal, text="Layer Palette",
                 bg=PAL_BG, fg=ACCENT,
                 font=("Segoe UI", 10, "bold"), pady=10).pack()
        for ltype, cfg in LAYER_DEFAULTS.items():
            self._pal_btn(pal, ltype, cfg["color"])
        tk.Label(pal,
                 text="\nDrag onto canvas\nor click to add",
                 bg=PAL_BG, fg="#555570",
                 font=("Segoe UI", 8), wraplength=132,
                 justify="center").pack(pady=10)

        # Canvas (centre)
        self.cv = tk.Canvas(body, bg=CANVAS_LIGHT, highlightthickness=0,
                            cursor="crosshair")
        self.cv.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.cv.bind("<Button-1>",        self._cv_click)
        self.cv.bind("<B1-Motion>",       self._cv_drag)
        self.cv.bind("<ButtonRelease-1>", self._cv_release)
        self.cv.bind("<Button-3>",        self._cv_rclick)
        self.cv.bind("<Delete>",          lambda _: self._delete_selection())
        self.cv.bind("<BackSpace>",       lambda _: self._delete_selection())
        self.cv.bind("<KeyPress-space>",  self._on_space_press)
        self.cv.bind("<KeyRelease-space>", self._on_space_release)
        self.cv.bind("<MouseWheel>", self._on_mousewheel)
        self.cv.bind("<Button-4>", self._on_mousewheel_linux)
        self.cv.bind("<Button-5>", self._on_mousewheel_linux)
        self.cv.bind("<Control-=>", self._zoom_in_center_key)
        self.cv.bind("<Control-minus>", self._zoom_out_center_key)
        self.cv.bind("<Control-0>", lambda _e: self._reset_view())
        self.cv.focus_set()

        # Properties (right) — scrollable
        prop = tk.Frame(body, bg=PROP_BG, width=304)
        prop.pack(side=tk.RIGHT, fill=tk.Y)
        prop.pack_propagate(False)
        self._props_canvas = tk.Canvas(
            prop, bg=PROP_BG, highlightthickness=0, bd=0)
        self._props_sb = ttk.Scrollbar(
            prop, orient="vertical", command=self._props_canvas.yview)
        self._props_inner = tk.Frame(self._props_canvas, bg=PROP_BG)
        self._props_win = self._props_canvas.create_window(
            (0, 0), window=self._props_inner, anchor="nw")
        self._props_canvas.configure(yscrollcommand=self._props_sb.set)
        self._props_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self._props_sb.pack(side=tk.RIGHT, fill=tk.Y)

        def _props_inner_cfg(_e=None):
            self._props_canvas.configure(scrollregion=self._props_canvas.bbox("all"))

        def _props_canvas_cfg(e):
            self._props_canvas.itemconfig(self._props_win, width=e.width)

        self._props_inner.bind("<Configure>", _props_inner_cfg)
        self._props_canvas.bind("<Configure>", _props_canvas_cfg)
        self._build_props(self._props_inner)
        self._props_bind_mousewheel(self._props_inner)

    def _tbtn(self, parent, text, cmd, side=tk.LEFT):
        b = tk.Button(parent, text=text, command=cmd,
                      bg="#313244", fg=FG, relief="flat",
                      font=("Segoe UI", 9), padx=10, pady=3,
                      activebackground="#45475A", activeforeground=FG)
        b.pack(side=side, padx=3, pady=2)
        return b

    def _pal_btn(self, parent, ltype, color):
        fg = "#2C2C2C" if _lum(color) > 140 else "white"
        btn = tk.Label(parent, text=ltype, bg=color, fg=fg,
                       font=("Segoe UI", 9, "bold"), pady=6,
                       cursor="hand2", relief="flat")
        btn.pack(fill=tk.X, padx=10, pady=2)
        btn.bind("<ButtonPress-1>",   lambda e, t=ltype: self._pd_start(e, t))
        btn.bind("<B1-Motion>",       self._pd_motion)
        btn.bind("<ButtonRelease-1>", self._pd_release)

    def _build_props(self, parent):
        tk.Label(parent, text="Properties",
                 bg=PROP_BG, fg=ACCENT,
                 font=("Segoe UI", 10, "bold"), pady=10).pack()

        self._pv_type = tk.StringVar(value="— select a layer —")
        tk.Label(parent, textvariable=self._pv_type,
                 bg=PROP_BG, fg=FG,
                 font=("Segoe UI", 9, "bold")).pack(pady=(0, 6))

        # Label / info — line breaks only where you press Enter (real \\n)
        self._plabel(parent, "Label (Enter = new line):", pady=(4, 0))
        self._txt_lbl = tk.Text(
            parent, height=2, wrap="word", bg="#313244", fg=FG,
            insertbackground=FG, relief="flat", font=("Segoe UI", 9),
            padx=4, pady=2)
        self._txt_lbl.pack(fill=tk.X, padx=12, pady=2)
        self._txt_lbl.bind("<KeyRelease>", self._on_text_props)
        self._txt_lbl.bind("<FocusOut>", self._on_text_props)

        self._plabel(parent, "Info (Enter = new line):", pady=(8, 0))
        self._txt_info = tk.Text(
            parent, height=4, wrap="word", bg="#313244", fg=FG,
            insertbackground=FG, relief="flat", font=("Segoe UI", 8),
            padx=4, pady=2)
        self._txt_info.pack(fill=tk.X, padx=12, pady=2)
        self._txt_info.bind("<KeyRelease>", self._on_text_props)
        self._txt_info.bind("<FocusOut>", self._on_text_props)

        self._plabel(parent, "Caption font sizes (points):", pady=(8, 0))
        self._pv_cap_lbl_size = tk.IntVar(value=9)
        self._pv_cap_info_size = tk.IntVar(value=8)
        r_cap_fs = tk.Frame(parent, bg=PROP_BG)
        r_cap_fs.pack(fill=tk.X, padx=12, pady=2)
        tk.Label(r_cap_fs, text="Label:", bg=PROP_BG, fg=FG,
                 font=("Segoe UI", 8)).pack(side=tk.LEFT)
        ttk.Scale(
            r_cap_fs, from_=6, to=36, variable=self._pv_cap_lbl_size,
            orient="horizontal",
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)
        tk.Label(r_cap_fs, textvariable=self._pv_cap_lbl_size, bg=PROP_BG, fg=FG,
                 font=("Segoe UI", 8), width=3).pack(side=tk.LEFT)
        r_cap_fs2 = tk.Frame(parent, bg=PROP_BG)
        r_cap_fs2.pack(fill=tk.X, padx=12, pady=2)
        tk.Label(r_cap_fs2, text="Info:", bg=PROP_BG, fg=FG,
                 font=("Segoe UI", 8)).pack(side=tk.LEFT)
        ttk.Scale(
            r_cap_fs2, from_=4, to=32, variable=self._pv_cap_info_size,
            orient="horizontal",
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)
        tk.Label(r_cap_fs2, textvariable=self._pv_cap_info_size, bg=PROP_BG, fg=FG,
                 font=("Segoe UI", 8), width=3).pack(side=tk.LEFT)
        self._pv_cap_lbl_size.trace_add("write", lambda *_: self._ap_caption_fonts())
        self._pv_cap_info_size.trace_add("write", lambda *_: self._ap_caption_fonts())

        self._frm_cap = tk.Frame(parent, bg=PROP_BG)
        self._plabel(
            self._frm_cap,
            "Label / info colors (under 3D blocks, auto if empty):",
            pady=(10, 0))
        r1 = tk.Frame(self._frm_cap, bg=PROP_BG)
        r1.pack(fill=tk.X, padx=12, pady=2)
        tk.Label(r1, text="Label:", bg=PROP_BG, fg=FG,
                 font=("Segoe UI", 8)).pack(side=tk.LEFT)
        self._cap_lbl_swatch = tk.Label(
            r1, text="  ", bg="#313244", relief="groove", width=3)
        self._cap_lbl_swatch.pack(side=tk.LEFT, padx=4)
        tk.Button(r1, text="Pick", command=self._pick_label_color,
                  bg="#313244", fg=FG, relief="flat", font=("Segoe UI", 8),
                  padx=6).pack(side=tk.LEFT, padx=2)
        tk.Button(r1, text="Auto", command=self._clear_label_color,
                  bg="#45475A", fg=FG, relief="flat", font=("Segoe UI", 8),
                  padx=6).pack(side=tk.LEFT)
        r2 = tk.Frame(self._frm_cap, bg=PROP_BG)
        r2.pack(fill=tk.X, padx=12, pady=2)
        tk.Label(r2, text="Info:", bg=PROP_BG, fg=FG,
                 font=("Segoe UI", 8)).pack(side=tk.LEFT)
        self._cap_info_swatch = tk.Label(
            r2, text="  ", bg="#313244", relief="groove", width=3)
        self._cap_info_swatch.pack(side=tk.LEFT, padx=4)
        tk.Button(r2, text="Pick", command=self._pick_info_color,
                  bg="#313244", fg=FG, relief="flat", font=("Segoe UI", 8),
                  padx=6).pack(side=tk.LEFT, padx=2)
        tk.Button(r2, text="Auto", command=self._clear_info_color,
                  bg="#45475A", fg=FG, relief="flat", font=("Segoe UI", 8),
                  padx=6).pack(side=tk.LEFT)
        self._frm_cap.pack(fill=tk.X)

        self._frm_text = tk.Frame(parent, bg=PROP_BG)
        self._plabel(self._frm_text, "Free Text layer style:", pady=(10, 0))
        self._pv_txt_bold = tk.BooleanVar(value=True)
        tk.Checkbutton(
            self._frm_text, text="Bold", variable=self._pv_txt_bold,
            command=self._ap_text_style, bg=PROP_BG, fg=FG,
            selectcolor="#313244", activebackground=PROP_BG,
            activeforeground=FG, font=("Segoe UI", 9),
        ).pack(anchor="w", padx=12, pady=2)
        tk.Label(
            self._frm_text,
            text='(Use "Caption font sizes" above for label / info text.)',
            bg=PROP_BG, fg="#6C7086", font=("Segoe UI", 8), justify="left",
        ).pack(anchor="w", padx=12, pady=(0, 4))
        r3 = tk.Frame(self._frm_text, bg=PROP_BG)
        r3.pack(fill=tk.X, padx=12, pady=4)
        tk.Label(r3, text="Text color:", bg=PROP_BG, fg=FG,
                 font=("Segoe UI", 8)).pack(side=tk.LEFT)
        self._txt_col_swatch = tk.Label(
            r3, text="  ", bg="#1A1E2E", relief="groove", width=3)
        self._txt_col_swatch.pack(side=tk.LEFT, padx=4)
        tk.Button(r3, text="Pick", command=self._pick_text_color,
                  bg="#313244", fg=FG, relief="flat", font=("Segoe UI", 8),
                  padx=6).pack(side=tk.LEFT)

        self._plabel(parent, "Fill opacity % (faces / image):", pady=(10, 0))
        self._pv_opacity = tk.IntVar(value=70)
        fro = tk.Frame(parent, bg=PROP_BG)
        fro.pack(fill=tk.X, padx=12, pady=1)
        ttk.Scale(fro, from_=0, to=100, variable=self._pv_opacity,
                  orient="horizontal").pack(side=tk.LEFT, fill=tk.X, expand=True)
        tk.Label(fro, textvariable=self._pv_opacity, bg=PROP_BG, fg=FG,
                 font=("Segoe UI", 8), width=4).pack(side=tk.RIGHT)
        self._pv_opacity.trace_add("write", lambda *_: self._ap_opacity())

        # Color
        self._plabel(parent, "Color:", pady=(10, 0))
        self._frm_color_row = tk.Frame(parent, bg=PROP_BG)
        self._frm_color_row.pack(fill=tk.X, padx=12, pady=3)
        row = self._frm_color_row
        self._swatch = tk.Label(row, text="      ", bg=WHEAT,
                                relief="groove", width=5)
        self._swatch.pack(side=tk.LEFT, padx=(0, 8))
        tk.Button(row, text="Pick Color", command=self._pick_color,
                  bg="#313244", fg=FG, relief="flat",
                  font=("Segoe UI", 9),
                  activebackground="#45475A", activeforeground=FG
                  ).pack(side=tk.LEFT)

        self._frm_face3 = tk.Frame(parent, bg=PROP_BG)
        self._plabel(
            self._frm_face3,
            "Per-face colors (3D blocks, optional):",
            pady=(8, 0))
        for label, which in (
                ("Front", "front"),
                ("Top", "top"),
                ("Right", "right"),
        ):
            frf = tk.Frame(self._frm_face3, bg=PROP_BG)
            frf.pack(fill=tk.X, padx=12, pady=2)
            tk.Label(frf, text=f"{label}:", bg=PROP_BG, fg=FG,
                     font=("Segoe UI", 8)).pack(side=tk.LEFT)
            sw = tk.Label(frf, text="  ", bg="#45475A", relief="groove", width=3)
            sw.pack(side=tk.LEFT, padx=4)
            setattr(self, f"_face_{which}_swatch", sw)
            tk.Button(
                frf, text="Pick", command=lambda w=which: self._pick_face_color(w),
                bg="#313244", fg=FG, relief="flat", font=("Segoe UI", 8),
                padx=6,
            ).pack(side=tk.LEFT, padx=2)
            tk.Button(
                frf, text="Auto", command=lambda w=which: self._clear_face_color(w),
                bg="#45475A", fg=FG, relief="flat", font=("Segoe UI", 8),
                padx=6,
            ).pack(side=tk.LEFT)
        self._frm_face3.pack(fill=tk.X)
        self._frm_face3.pack_forget()

        # Dimension sliders (Picture uses W/H as frame; Depth disabled)
        self._frm_geom = tk.Frame(parent, bg=PROP_BG)
        self._frm_geom.pack(fill=tk.X)
        self._sliders = {}
        self._scale_widgets = {}
        for prop, lbl, lo, hi in (
            ("w", "Thickness (W)",  2,  250),
            ("h", "Height (H)",     8,  260),
            ("d", "Depth (D)",      0,  220),
        ):
            self._plabel(self._frm_geom, f"{lbl}:", pady=(8, 0))
            var = tk.IntVar(value=20)
            fr = tk.Frame(self._frm_geom, bg=PROP_BG)
            fr.pack(fill=tk.X, padx=12, pady=1)
            sc = ttk.Scale(fr, from_=lo, to=hi, variable=var,
                           orient="horizontal")
            sc.pack(side=tk.LEFT, fill=tk.X, expand=True)
            tk.Label(fr, textvariable=var, bg=PROP_BG, fg=FG,
                     font=("Segoe UI", 8), width=4).pack(side=tk.RIGHT)
            var.trace_add("write", lambda *_, p=prop, v=var: self._ap_slider(p, v))
            self._sliders[prop] = var
            self._scale_widgets[prop] = sc

        self._frm_pic = tk.Frame(parent, bg=PROP_BG)
        self._plabel(self._frm_pic, "Picture file:", pady=(8, 0))
        self._pic_path_lbl = tk.Label(
            self._frm_pic, text="(none)", bg="#313244", fg=FG,
            anchor="w", font=("Segoe UI", 8), wraplength=248, justify="left")
        self._pic_path_lbl.pack(fill=tk.X, padx=12, pady=2)
        brow = tk.Frame(self._frm_pic, bg=PROP_BG)
        brow.pack(fill=tk.X, padx=12, pady=2)
        tk.Button(brow, text="Browse…", command=self._browse_picture,
                  bg="#313244", fg=FG, relief="flat", font=("Segoe UI", 9),
                  activebackground="#45475A", activeforeground=FG
                  ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))
        tk.Button(brow, text="Clear", command=self._clear_picture,
                  bg="#45475A", fg=FG, relief="flat", font=("Segoe UI", 9),
                  activebackground="#585B70", activeforeground=FG
                  ).pack(side=tk.LEFT)

        self._plabel(self._frm_pic, "Rotation ° (in plane):", pady=(8, 0))
        self._pv_rotation = tk.IntVar(value=0)
        frr = tk.Frame(self._frm_pic, bg=PROP_BG)
        frr.pack(fill=tk.X, padx=12, pady=1)
        ttk.Scale(frr, from_=-180, to=180, variable=self._pv_rotation,
                  orient="horizontal").pack(side=tk.LEFT, fill=tk.X, expand=True)
        tk.Label(frr, textvariable=self._pv_rotation, bg=PROP_BG, fg=FG,
                 font=("Segoe UI", 8), width=4).pack(side=tk.RIGHT)
        self._pv_rotation.trace_add("write", lambda *_: self._ap_rotation())

        self._plabel(self._frm_pic, "Skew ° (slant / legacy):", pady=(6, 0))
        self._pv_tilt = tk.IntVar(value=0)
        frt = tk.Frame(self._frm_pic, bg=PROP_BG)
        frt.pack(fill=tk.X, padx=12, pady=1)
        ttk.Scale(frt, from_=-45, to=45, variable=self._pv_tilt,
                  orient="horizontal").pack(side=tk.LEFT, fill=tk.X, expand=True)
        tk.Label(frt, textvariable=self._pv_tilt, bg=PROP_BG, fg=FG,
                 font=("Segoe UI", 8), width=4).pack(side=tk.RIGHT)
        self._pv_tilt.trace_add("write", lambda *_: self._ap_tilt())

        # Buttons
        self._btn_dup = tk.Button(
            parent, text="Duplicate Layer", command=self._dup,
            bg="#313244", fg=FG, relief="flat",
            font=("Segoe UI", 9), pady=4,
            activebackground="#45475A", activeforeground=FG)
        self._btn_dup.pack(fill=tk.X, padx=12, pady=(14, 3))

        tk.Button(parent, text="Delete Layer  [Del]", command=self._del,
                  bg="#F38BA8", fg="white", relief="flat",
                  font=("Segoe UI", 9), pady=4,
                  activebackground="#EBA0AC", activeforeground="white"
                  ).pack(fill=tk.X, padx=12, pady=3)

        # Reorder
        self._plabel(parent, "Connection order:", pady=(10, 0))
        row2 = tk.Frame(parent, bg=PROP_BG)
        row2.pack(padx=12, pady=2, anchor="w")
        for txt, fn in (("◀ Left", lambda: self._reorder(-1)),
                        ("Right ▶", lambda: self._reorder(1))):
            tk.Button(row2, text=txt, command=fn,
                      bg="#313244", fg=FG, relief="flat",
                      font=("Segoe UI", 8), padx=6, pady=2,
                      activebackground="#45475A", activeforeground=FG
                      ).pack(side=tk.LEFT, padx=2)

        tk.Label(parent,
                 text="\nRight-click layer or arrow → delete\n"
                      "Wheel / Zoom± = zoom (cursor or center)\n"
                      "Ctrl+0 = reset view · Hold Space+drag = pan\n"
                      "Drag to move freely\n"
                      "Text: palette → drag; Bold / size / color in panel.\n"
                      "Picture: Browse (Pillow). Scroll right panel if needed.\n"
                      "Auto Layout = even spacing\n"
                      "Arrows: auto left→right, or toolbar\n"
                      "\"Add arrow\" then click A, then B\n\n"
                      "EPS = vector (best for papers)\n"
                      "PNG export needs Pillow",
                 bg=PROP_BG, fg="#555570",
                 font=("Segoe UI", 8), justify="left"
                 ).pack(padx=12, pady=8, anchor="w")

    def _plabel(self, parent, text, pady=(4, 0)):
        tk.Label(parent, text=text,
                 bg=PROP_BG, fg=FG,
                 font=("Segoe UI", 9), anchor="w"
                 ).pack(fill=tk.X, padx=12, pady=pady)

    def _props_bind_mousewheel(self, widget):
        """Scroll properties panel when wheel is used over it (not over main canvas)."""
        if isinstance(widget, tk.Text):
            return

        def handler(e):
            if getattr(e, "delta", 0):
                d = e.delta
                if sys.platform == "darwin":
                    d = -d
                self._props_canvas.yview_scroll(int(-d / 120), "units")
            return "break"

        def linux_up(_e):
            self._props_canvas.yview_scroll(-3, "units")
            return "break"

        def linux_dn(_e):
            self._props_canvas.yview_scroll(3, "units")
            return "break"

        widget.bind("<MouseWheel>", handler)
        widget.bind("<Button-4>", linux_up)
        widget.bind("<Button-5>", linux_dn)
        for c in widget.winfo_children():
            self._props_bind_mousewheel(c)

    def _effective_label_fg(self, layer):
        c = (getattr(layer, "label_color", "") or "").strip()
        if c.startswith("#") and len(c) == 7:
            return c
        return "#1A1A2A" if self._light_cv else FG

    def _effective_info_fg(self, layer):
        c = (getattr(layer, "info_color", "") or "").strip()
        if c.startswith("#") and len(c) == 7:
            return c
        return "#555555" if self._light_cv else "#8A8FA8"

    @staticmethod
    def _valid_hex_color(s):
        return isinstance(s, str) and s.strip().startswith("#") and len(s.strip()) == 7

    def _default_face_base_hex(self, layer, which):
        """Base color for a face when no per-face override (before opacity blend)."""
        c = layer.color
        if which == "front":
            return c
        if which == "top":
            return lighten(c, 0.25)
        return darken(c, 0.22)

    def _base_face_color(self, layer, which):
        o = (getattr(layer, f"color_{which}", "") or "").strip()
        if self._valid_hex_color(o):
            return o.strip()
        return self._default_face_base_hex(layer, which)

    def _sync_face_swatches(self):
        for which in ("front", "top", "right"):
            sw = getattr(self, f"_face_{which}_swatch", None)
            if sw is None:
                return
            if not self.selected:
                sw.configure(bg="#45475A")
                continue
            o = (getattr(self.selected, f"color_{which}", "") or "").strip()
            sw.configure(bg=o if self._valid_hex_color(o) else "#45475A")

    def _caption_canvas_fonts(self, layer):
        """Logical pt sizes scaled for current zoom (matches on-canvas appearance)."""
        z = self._zoom
        cap = min(z, 2.8)
        lb = float(getattr(layer, "label_font_size", 9))
        ib = float(getattr(layer, "info_font_size", 8))
        return max(6, int(round(lb * cap))), max(5, int(round(ib * cap)))

    @staticmethod
    def _caption_info_y(ly0, label_text, fs_lbl):
        """Vertical offset for info line from label baseline (multi-line aware)."""
        if not (label_text or "").strip():
            return ly0
        n = label_text.count("\n") + 1
        return ly0 + n * int(fs_lbl * 1.22) + 6

    # ── Drawing ────────────────────────────────────────────────────────────────

    def draw_all(self):
        self.cv.delete("all")
        self._photo_cache.clear()
        self.cv.configure(bg=CANVAS_LIGHT if self._light_cv else CANVAS_DARK)
        for layer in self.layers:
            self._draw_layer(layer)
        if self._show_conn:
            self._draw_connections()
        self._draw_arrow_mode_hint()

    def _cx(self, x):
        return x * self._zoom - self._view_pan_x

    def _cy(self, y):
        return y * self._zoom - self._view_pan_y

    def _wx(self, cx):
        return (cx + self._view_pan_x) / self._zoom

    def _wy(self, cy):
        return (cy + self._view_pan_y) / self._zoom

    def _apply_zoom_at(self, factor, mx, my):
        """Zoom by factor keeping world point under (mx, my) fixed."""
        old_z = self._zoom
        z_new = max(0.12, min(8.0, old_z * factor))
        if abs(z_new - old_z) < 1e-9:
            return
        wx = (mx + self._view_pan_x) / old_z
        wy = (my + self._view_pan_y) / old_z
        self._zoom = z_new
        self._view_pan_x = wx * z_new - mx
        self._view_pan_y = wy * z_new - my
        self.draw_all()

    def _on_mousewheel(self, e):
        d = getattr(e, "delta", 0)
        if d == 0:
            return
        if sys.platform == "darwin":
            d = -d
        f = 1.12 if d > 0 else 1 / 1.12
        self._apply_zoom_at(f, e.x, e.y)

    def _on_mousewheel_linux(self, e):
        if e.num == 4:
            self._apply_zoom_at(1.12, e.x, e.y)
        elif e.num == 5:
            self._apply_zoom_at(1 / 1.12, e.x, e.y)

    def _zoom_in_center(self):
        self.cv.update_idletasks()
        self._apply_zoom_at(
            1.2,
            max(1, self.cv.winfo_width()) // 2,
            max(1, self.cv.winfo_height()) // 2,
        )

    def _zoom_out_center(self):
        self.cv.update_idletasks()
        self._apply_zoom_at(
            1 / 1.2,
            max(1, self.cv.winfo_width()) // 2,
            max(1, self.cv.winfo_height()) // 2,
        )

    def _zoom_in_center_key(self, _e=None):
        self._zoom_in_center()

    def _zoom_out_center_key(self, _e=None):
        self._zoom_out_center()

    def _reset_view(self, _e=None):
        self._zoom = 1.0
        self._view_pan_x = self._view_pan_y = 0
        self.draw_all()

    def _on_space_press(self, _e=None):
        self._space_held = True
        self.cv.config(cursor="fleur")

    def _on_space_release(self, _e=None):
        self._space_held = False
        if not self._canvas_panning:
            self.cv.config(cursor="crosshair")

    def _canvas_bg_hex(self):
        return CANVAS_LIGHT if self._light_cv else CANVAS_DARK

    def _draw_text_layer(self, layer):
        tag = f"L{layer.id}"
        x = self._cx(layer.x)
        y = self._cy(layer.y)
        fs, fs2 = self._caption_canvas_fonts(layer)
        bold = bool(getattr(layer, "text_bold", True))
        font_main = ("Segoe UI", fs, "bold") if bold else ("Segoe UI", fs)
        tc = (getattr(layer, "text_color", "") or "").strip()
        if not (tc.startswith("#") and len(tc) == 7):
            tc = "#1A1A2A" if self._light_cv else FG
        if layer.label:
            self.cv.create_text(
                x, y,
                text=layer.label,
                fill=tc,
                font=font_main,
                justify="center",
                anchor="center",
                tags=tag,
            )
        if layer.info:
            ifo = self._effective_info_fg(layer)
            if layer.label:
                nl = max(1, layer.label.count("\n") + 1)
                dy = int(nl * fs * 0.55) + 8
            else:
                dy = 0
            self.cv.create_text(
                x, y + dy,
                text=layer.info,
                fill=ifo,
                font=("Segoe UI", fs2),
                justify="center",
                anchor="n",
                tags=tag,
            )

    def _draw_picture(self, layer):
        tag = f"L{layer.id}"
        z = self._zoom
        x = self._cx(layer.x)
        y = self._cy(layer.y)
        w = max(8, int(layer.w * z))
        h = max(8, int(layer.h * z))
        bg = self._canvas_bg_hex()
        bgr = _hex_to_rgb(bg)
        op = max(0.0, min(1.0, float(getattr(layer, "opacity", 0.7))))
        path = (getattr(layer, "image_path", "") or "").strip()
        if path:
            path = os.path.normpath(path)
            if not os.path.isabs(path):
                path = os.path.abspath(path)

        if not path or not os.path.isfile(path):
            self.cv.create_rectangle(
                x - w / 2, y - h / 2, x + w / 2, y + h / 2,
                outline="#6C7086", dash=(5, 4), width=1, tags=tag)
            self.cv.create_text(
                x, y, text="Picture\nBrowse…", fill="#585B70",
                font=("Segoe UI", 9), justify="center", tags=tag)
        else:
            try:
                from PIL import Image as PILImage
                from PIL import ImageTk
            except ImportError:
                self.cv.create_text(
                    x, y, text="pip install Pillow",
                    fill="#D20F39", font=("Segoe UI", 9, "bold"), tags=tag)
                return
            try:
                _LANCZOS = getattr(
                    getattr(PILImage, "Resampling", PILImage), "LANCZOS", 1)
                _BICUBIC = getattr(
                    getattr(PILImage, "Resampling", PILImage), "BICUBIC", 3)
                pil = PILImage.open(path).convert("RGBA")
                pil.thumbnail((w, h), _LANCZOS)
                rot = float(getattr(layer, "image_rotation", 0))
                skew = float(getattr(layer, "image_tilt", 0))
                ang = -(rot + skew)
                if abs(ang) > 0.05:
                    pil = pil.rotate(
                        ang,
                        resample=_BICUBIC,
                        expand=True,
                        fillcolor=bgr + (255,),
                    )
                if op < 0.999:
                    alpha = pil.split()[3]
                    alpha = alpha.point(lambda px: int(px * op))
                    pil.putalpha(alpha)
                rgb = PILImage.new("RGB", pil.size, bgr)
                rgb.paste(pil, (0, 0), pil)
                photo = ImageTk.PhotoImage(rgb)
            except OSError:
                self.cv.create_text(
                    x, y, text="Could not load image",
                    fill="#D20F39", font=("Segoe UI", 9), tags=tag)
                return
            self._photo_cache[layer.id] = photo
            self.cv.create_image(x, y, image=photo, tags=tag)

        lbl_fg = self._effective_label_fg(layer)
        info_fg = self._effective_info_fg(layer)
        fs_lbl, fs_inf = self._caption_canvas_fonts(layer)
        ly0 = y + h / 2 + max(10, int(12 * min(z, 1.5)))
        if layer.label:
            self.cv.create_text(
                x, ly0, text=layer.label, fill=lbl_fg,
                font=("Segoe UI", fs_lbl, "bold"), anchor="n", tags=tag)
        if layer.info:
            ly_i = self._caption_info_y(ly0, layer.label, fs_lbl)
            self.cv.create_text(
                x, ly_i, text=layer.info, fill=info_fg,
                font=("Segoe UI", fs_inf), justify="center", anchor="n", tags=tag)

    # ── single 3-D block ───────────────────────────────────────────────────────

    def _draw_layer(self, layer):
        if layer.layer_type == "Picture":
            self._draw_picture(layer)
            return
        if layer.layer_type == "Text":
            self._draw_text_layer(layer)
            return

        x = self._cx(layer.x)
        y = self._cy(layer.y)
        w, h, d = layer.w, layer.h, layer.d
        c       = layer.color
        ox      = d * PERSP          # world depth offset
        oy      = d * PERSP
        z       = self._zoom
        ws      = w * z
        hs      = h * z
        oxs     = ox * z
        oys     = oy * z
        sel     = layer is self.selected
        tag     = f"L{layer.id}"

        bg  = self._canvas_bg_hex()
        op  = max(0.0, min(1.0, float(getattr(layer, "opacity", 0.7))))
        bf = self._base_face_color(layer, "front")
        bt = self._base_face_color(layer, "top")
        br = self._base_face_color(layer, "right")
        c_face = blend_hex(bf, bg, op)
        c_top = blend_hex(bt, bg, op)
        c_right = blend_hex(br, bg, op)
        oc      = SEL_CLR if sel else darken(c, 0.38)
        ow      = max(1, int((2 if sel else 1) * min(z, 3.0)))

        # ── Three visible faces ───────────────────────────────────────────────
        #    With grid on: fill only; full 12-edge wireframe (solid = visible,
        #    dashed = hidden edges at back corner). With grid off: polygon outlines.

        poly_outline = ("", 0) if self._show_grid else (oc, ow)

        # Front face
        self.cv.create_polygon(
            x - ws/2, y - hs/2,
            x + ws/2, y - hs/2,
            x + ws/2, y + hs/2,
            x - ws/2, y + hs/2,
            fill=c_face, outline=poly_outline[0], width=poly_outline[1], tags=tag)

        # Top face
        self.cv.create_polygon(
            x - ws/2,      y - hs/2,
            x + ws/2,      y - hs/2,
            x + ws/2 + oxs, y - hs/2 - oys,
            x - ws/2 + oxs, y - hs/2 - oys,
            fill=c_top, outline=poly_outline[0], width=poly_outline[1], tags=tag)

        # Right face
        self.cv.create_polygon(
            x + ws/2,      y - hs/2,
            x + ws/2 + oxs, y - hs/2 - oys,
            x + ws/2 + oxs, y + hs/2 - oys,
            x + ws/2,      y + hs/2,
            fill=c_right, outline=poly_outline[0], width=poly_outline[1], tags=tag)

        # ── Grid: CAD-style cube — solid visible edges, dashed hidden edges
        #    (three segments at back-bottom-left). No face subdivisions.

        if self._show_grid:
            dk = (5, 4)
            dw = max(1, min(ow, 2))

            def rf(v):
                return float(round(v))

            x1, y1 = rf(x - ws / 2), rf(y - hs / 2)
            x2, y2 = rf(x + ws / 2), rf(y - hs / 2)
            x3, y3 = rf(x + ws / 2), rf(y + hs / 2)
            x4, y4 = rf(x - ws / 2), rf(y + hs / 2)
            xt3, yt3 = rf(x + ws / 2 + oxs), rf(y - hs / 2 - oys)
            xt4, yt4 = rf(x - ws / 2 + oxs), rf(y - hs / 2 - oys)
            xr3, yr3 = rf(x + ws / 2 + oxs), rf(y + hs / 2 - oys)
            xr4, yr4 = rf(x - ws / 2 + oxs), rf(y + hs / 2 - oys)

            visible = (
                (x1, y1, x2, y2),
                (x2, y2, x3, y3),
                (x3, y3, x4, y4),
                (x4, y4, x1, y1),
                (x1, y1, xt4, yt4),
                (x2, y2, xt3, yt3),
                (xt4, yt4, xt3, yt3),
                (xt3, yt3, xr3, yr3),
                (xr3, yr3, x3, y3),
            )
            for xa, ya, xb, yb in visible:
                self.cv.create_line(
                    xa, ya, xb, yb,
                    fill=oc, width=ow, tags=tag)

            depth_ok = (oxs * oxs + oys * oys) > 0.25
            if depth_ok:
                hidden = (
                    (x4, y4, xr4, yr4),
                    (xr4, yr4, xt4, yt4),
                    (xr4, yr4, xr3, yr3),
                )
                for xa, ya, xb, yb in hidden:
                    self.cv.create_line(
                        xa, ya, xb, yb,
                        fill=oc, dash=dk, width=dw, tags=tag)

        # ── Labels (anchor=n so multi-line info grows down, no overlap) ───────

        lbl_fg = self._effective_label_fg(layer)
        info_fg = self._effective_info_fg(layer)
        fs_lbl, fs_inf = self._caption_canvas_fonts(layer)
        lx      = x + oxs / 2
        ly0     = y + hs / 2 + max(10, int(14 * min(z, 1.5)))

        if layer.label:
            self.cv.create_text(
                lx, ly0,
                text=layer.label,
                fill=lbl_fg,
                font=("Segoe UI", fs_lbl, "bold"),
                anchor="n",
                tags=tag,
            )
        if layer.info:
            display = layer.info
            ly_info = self._caption_info_y(ly0, layer.label, fs_lbl)
            self.cv.create_text(
                lx, ly_info,
                text=display,
                fill=info_fg,
                font=("Segoe UI", fs_inf),
                justify="center",
                anchor="n",
                tags=tag,
            )

    # ── Connection arrows ──────────────────────────────────────────────────────

    def _layer_by_id(self, lid):
        for L in self.layers:
            if L.id == lid:
                return L
        return None

    def _connection_pairs(self):
        """Return [(from_layer, to_layer), ...]. Manual list if set, else x-sorted chain."""
        if self.manual_conns:
            pairs = []
            for a_id, b_id in self.manual_conns:
                la, lb = self._layer_by_id(a_id), self._layer_by_id(b_id)
                if la is not None and lb is not None:
                    pairs.append((la, lb))
            return pairs
        ordered = sorted(self.layers, key=lambda l: l.x)
        return list(zip(ordered, ordered[1:])) if len(ordered) >= 2 else []

    def _arrow_endpoints(self, a, b):
        """Right-centre of block a → left-centre of block b (horizontal flow, teal style)."""
        oxa, oya = a.d * PERSP, a.d * PERSP
        oxb, oyb = b.d * PERSP, b.d * PERSP
        ax = a.x + a.w / 2 + oxa * 0.5
        ay = a.y + oya * (-0.15)
        bx = b.x - b.w / 2 + oxb * 0.25
        by = b.y - oyb * 0.15
        return ax, ay, bx, by

    @staticmethod
    def _dist_sq_point_segment(px, py, x1, y1, x2, y2):
        dx = x2 - x1
        dy = y2 - y1
        if dx * dx + dy * dy < 1e-12:
            return (px - x1) ** 2 + (py - y1) ** 2
        t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)))
        qx = x1 + t * dx
        qy = y1 + t * dy
        return (px - qx) ** 2 + (py - qy) ** 2

    def _hit_arrow(self, cx, cy, tol_px=14.0):
        """Return (from_id, to_id) if canvas point is near an arrow segment, else None."""
        if not self._show_conn:
            return None
        pairs = self._connection_pairs()
        if not pairs:
            return None
        tol2 = tol_px * tol_px
        best = None
        best_d = tol2 + 1.0
        # Later-drawn arrows win (same as visual stacking)
        for a, b in reversed(pairs):
            ax, ay, bx, by = self._arrow_endpoints(a, b)
            x1, y1 = self._cx(ax), self._cy(ay)
            x2, y2 = self._cx(bx), self._cy(by)
            d2 = self._dist_sq_point_segment(cx, cy, x1, y1, x2, y2)
            if d2 <= tol2 and d2 < best_d:
                best_d = d2
                best = (a.id, b.id)
        return best

    def _ensure_manual_edges(self):
        """If using auto chain, copy it into manual_conns so individual edges can be removed."""
        if self.manual_conns:
            return
        ordered = sorted(self.layers, key=lambda l: l.x)
        chain = list(zip(ordered, ordered[1:])) if len(ordered) >= 2 else []
        self.manual_conns = [[a.id, b.id] for a, b in chain]

    def _delete_edge(self, a_id, b_id):
        self._ensure_manual_edges()
        self.manual_conns = [
            p for p in self.manual_conns
            if not (p[0] == a_id and p[1] == b_id)
        ]
        self.selected_edge = None
        self.draw_all()

    def _draw_mid_dart_arrow(self, x1, y1, x2, y2, fill, width, tags):
        """Mid-segment dart head; one continuous shaft, head drawn on top (no gaps)."""
        dx = x2 - x1
        dy = y2 - y1
        L = (dx * dx + dy * dy) ** 0.5
        if L < 1e-6:
            return
        w_int = max(1, int(round(width)))
        if L < 14:
            self.cv.create_line(
                x1, y1, x2, y2,
                fill=fill, width=w_int, tags=tags,
                capstyle=tk.ROUND,
            )
            return

        ux, uy = dx / L, dy / L
        nx, ny = -uy, ux
        mx = (x1 + x2) * 0.5
        my = (y1 + y2) * 0.5

        scale = max(width * 5.0, 11.0 * min(self._zoom, 2.4))
        hb = scale * 0.40
        ht = scale * 0.36
        wing = max(scale * 0.50, w_int * 0.55 + 2.0)
        notch = scale * 0.24

        tx = mx + ux * ht
        ty = my + uy * ht
        rux = mx - ux * hb + nx * wing
        ruy = my - uy * hb + ny * wing
        rdx = mx - ux * hb - nx * wing
        rdy = my - uy * hb - ny * wing
        ncx = mx - ux * max(0.0, hb - notch)
        ncy = my - uy * max(0.0, hb - notch)

        # Full shaft underneath; dart on top covers the middle so the stroke reads continuous.
        self.cv.create_line(
            x1, y1, x2, y2,
            fill=fill, width=w_int, capstyle=tk.ROUND, tags=tags,
        )
        self.cv.create_polygon(
            tx, ty, rux, ruy, ncx, ncy, rdx, rdy,
            fill=fill, outline=fill, width=1, tags=tags,
        )

    def _draw_connections(self):
        pairs = self._connection_pairs()
        if not pairs:
            return
        base_col = "#2A9D8F" if self._light_cv else "#7EB8B0"
        hi_col = "#E9C46A" if self._light_cv else "#F9E076"
        lw = max(1.0, 2.2 * min(self._zoom, 2.5))
        sel = self.selected_edge
        for a, b in pairs:
            ax, ay, bx, by = self._arrow_endpoints(a, b)
            is_sel = sel is not None and sel[0] == a.id and sel[1] == b.id
            col = hi_col if is_sel else base_col
            w = lw + (3.0 if is_sel else 0.0)
            tag = ("conn", f"E{a.id}_{b.id}")
            self._draw_mid_dart_arrow(
                self._cx(ax), self._cy(ay),
                self._cx(bx), self._cy(by),
                col, w, tag,
            )

    def _draw_arrow_mode_hint(self):
        if not self._add_arrow_mode:
            return
        fg = "#1A1A2A" if self._light_cv else FG
        cw = max(self.cv.winfo_width(), 120)
        if self._conn_pick_from is not None:
            msg = "Arrow mode: now click the target layer."
        else:
            msg = "Arrow mode: click source layer, then target."
        self.cv.create_text(
            cw // 2, 16, text=msg, fill=fg,
            font=("Segoe UI", 9, "italic"), tags="hint")
        L = self._layer_by_id(self._conn_pick_from) if self._conn_pick_from else None
        if L:
            ox = L.d * PERSP
            cx = self._cx(L.x + ox * 0.35)
            cy = self._cy(L.y - L.h / 2 - 18 / self._zoom)
            self.cv.create_rectangle(
                cx - 28, cy - 10, cx + 28, cy + 10,
                outline=ACCENT, width=2, dash=(4, 3), tags="hint")
            self.cv.create_text(cx, cy, text="from", fill=ACCENT,
                                font=("Segoe UI", 8, "bold"), tags="hint")

    # ── Hit test ──────────────────────────────────────────────────────────────

    def _hit(self, mx, my):
        for layer in reversed(self.layers):
            if layer.layer_type == "Picture":
                if (layer.x - layer.w / 2 <= mx <= layer.x + layer.w / 2 and
                        layer.y - layer.h / 2 <= my <= layer.y + layer.h / 2):
                    return layer
                continue
            if layer.layer_type == "Text":
                if (layer.x - layer.w / 2 <= mx <= layer.x + layer.w / 2 and
                        layer.y - layer.h / 2 <= my <= layer.y + layer.h / 2):
                    return layer
                continue
            ox = layer.d * PERSP
            oy = layer.d * PERSP
            if (layer.x - layer.w / 2 <= mx <= layer.x + layer.w / 2 + ox and
                    layer.y - layer.h / 2 - oy <= my <= layer.y + layer.h / 2):
                return layer
        return None

    # ── Canvas events ──────────────────────────────────────────────────────────

    def _cv_click(self, e):
        self.cv.focus_set()
        if self._space_held:
            self._canvas_panning = True
            self._pan_last_x = e.x
            self._pan_last_y = e.y
            self._dragging = None
            return

        layer = self._hit(self._wx(e.x), self._wy(e.y))

        if self._add_arrow_mode:
            if layer is None:
                self._conn_pick_from = None
                self.draw_all()
                return
            self.selected_edge = None
            if self._conn_pick_from is None:
                self._conn_pick_from = layer.id
                self.selected = layer
                self._dragging = None
                self._sync_props()
                self.draw_all()
                return
            if self._conn_pick_from == layer.id:
                self._conn_pick_from = None
                self.draw_all()
                return
            pair = [self._conn_pick_from, layer.id]
            if pair not in self.manual_conns and [pair[1], pair[0]] not in self.manual_conns:
                self.manual_conns.append(pair)
            self._conn_pick_from = None
            self.selected = layer
            self._sync_props()
            self.draw_all()
            return

        edge = self._hit_arrow(e.x, e.y)
        if edge is not None:
            self.selected_edge = edge
            self.selected = None
            self._dragging = None
            self._sync_props()
            self.draw_all()
            return

        self.selected_edge = None
        self.selected = layer
        if layer:
            self._dragging = dict(layer=layer,
                                   sx=e.x, sy=e.y,
                                   ox=layer.x, oy=layer.y)
        else:
            self._dragging = None
        self._sync_props()
        self.draw_all()

    def _cv_drag(self, e):
        if self._canvas_panning and self._pan_last_x is not None:
            self._view_pan_x -= e.x - self._pan_last_x
            self._view_pan_y -= e.y - self._pan_last_y
            self._pan_last_x = e.x
            self._pan_last_y = e.y
            self.draw_all()
            return
        if self._dragging:
            d = self._dragging
            d["layer"].x = d["ox"] + (e.x - d["sx"])
            d["layer"].y = d["oy"] + (e.y - d["sy"])
            self.draw_all()

    def _cv_release(self, _):
        self._dragging = None
        self._canvas_panning = False
        self._pan_last_x = self._pan_last_y = None
        if not self._space_held:
            self.cv.config(cursor="crosshair")

    def _cv_rclick(self, e):
        edge = self._hit_arrow(e.x, e.y)
        if edge is not None:
            self._delete_edge(edge[0], edge[1])
            self._sync_props()
            return
        layer = self._hit(self._wx(e.x), self._wy(e.y))
        if layer:
            self.selected_edge = None
            self.selected = layer
            self._del()
            self._sync_props()

    def _delete_selection(self):
        if self.selected_edge is not None:
            self._delete_edge(self.selected_edge[0], self.selected_edge[1])
            self._sync_props()
        elif self.selected:
            self._del()

    # ── Palette drag & drop ────────────────────────────────────────────────────

    def _pd_start(self, e, ltype):
        ghost = tk.Toplevel(self)
        ghost.overrideredirect(True)
        ghost.attributes("-alpha", 0.82)
        ghost.attributes("-topmost", True)
        color = LAYER_DEFAULTS[ltype]["color"]
        fg    = "#2C2C2C" if _lum(color) > 140 else "white"
        tk.Label(ghost, text=f"  {ltype}  ",
                 bg=color, fg=fg,
                 font=("Segoe UI", 9, "bold"), pady=6).pack()
        ghost.geometry(f"+{e.x_root + 14}+{e.y_root + 14}")
        self._pal_drag = dict(type=ltype, ghost=ghost,
                               sx=e.x_root, sy=e.y_root)

    def _pd_motion(self, e):
        if self._pal_drag:
            self._pal_drag["ghost"].geometry(
                f"+{e.x_root + 14}+{e.y_root + 14}")

    def _pd_release(self, e):
        if not self._pal_drag:
            return
        self._pal_drag["ghost"].destroy()
        pd = self._pal_drag
        self._pal_drag = None

        dx  = abs(e.x_root - pd["sx"])
        dy  = abs(e.y_root - pd["sy"])
        cvx = self.cv.winfo_rootx()
        cvy = self.cv.winfo_rooty()
        cvw = self.cv.winfo_width()
        cvh = self.cv.winfo_height()

        if dx < 6 and dy < 6:
            # Plain click → append to the right of existing layers
            xs = [l.x for l in self.layers]
            x  = (max(xs) + 120) if xs else 120
            y  = self._wy(cvh / 2) if cvh > 1 else 340
        elif cvx <= e.x_root <= cvx + cvw and cvy <= e.y_root <= cvy + cvh:
            cx = e.x_root - cvx
            cy = e.y_root - cvy
            x = self._wx(cx)
            y = self._wy(cy)
        else:
            return  # released outside canvas

        layer = Layer(pd["type"], x, y)
        self.layers.append(layer)
        self.selected = layer
        self._sync_props()
        self.draw_all()

    # ── Properties sync ────────────────────────────────────────────────────────

    def _sync_props(self):
        self._syncing = True
        if self.selected:
            self._pv_type.set(self.selected.layer_type)
            self._txt_lbl.delete("1.0", "end")
            self._txt_lbl.insert("1.0", self.selected.label or "")
            self._txt_info.delete("1.0", "end")
            self._txt_info.insert("1.0", self.selected.info or "")
            try:
                self._pv_opacity.set(int(round(float(self.selected.opacity) * 100)))
            except (tk.TclError, ValueError, TypeError):
                self._pv_opacity.set(70)
            self._swatch.configure(bg=self.selected.color)
            self._sync_face_swatches()
            for p, v in self._sliders.items():
                v.set(int(getattr(self.selected, p)))
            lc = (getattr(self.selected, "label_color", "") or "").strip()
            self._cap_lbl_swatch.configure(bg=lc if lc else "#45475A")
            ic = (getattr(self.selected, "info_color", "") or "").strip()
            self._cap_info_swatch.configure(bg=ic if ic else "#45475A")
            try:
                self._pv_cap_lbl_size.set(
                    int(getattr(self.selected, "label_font_size", 9)))
            except (TypeError, ValueError):
                self._pv_cap_lbl_size.set(9)
            try:
                self._pv_cap_info_size.set(
                    int(getattr(self.selected, "info_font_size", 8)))
            except (TypeError, ValueError):
                self._pv_cap_info_size.set(8)

            if self.selected.layer_type == "Text":
                self._frm_face3.pack_forget()
                self._frm_cap.pack_forget()
                self._frm_pic.pack_forget()
                self._frm_text.pack_forget()
                self._frm_text.pack(fill=tk.X, after=self._frm_geom)
                self._pv_txt_bold.set(bool(getattr(self.selected, "text_bold", True)))
                tc = (getattr(self.selected, "text_color", "") or "#1A1E2E").strip()
                self._txt_col_swatch.configure(
                    bg=tc if tc.startswith("#") and len(tc) == 7 else "#1A1E2E")
                self._scale_widgets["d"].state(["disabled"])
            elif self.selected.layer_type == "Picture":
                self._frm_face3.pack_forget()
                self._frm_text.pack_forget()
                self._frm_cap.pack_forget()
                self._frm_cap.pack(fill=tk.X, after=self._txt_info)
                self._frm_pic.pack_forget()
                self._frm_pic.pack(fill=tk.X, after=self._frm_geom)
                self._pic_path_lbl.configure(
                    text=self._short_path(self.selected.image_path or "(none)"))
                try:
                    self._pv_rotation.set(
                        int(round(float(self.selected.image_rotation))))
                except (tk.TclError, ValueError, TypeError):
                    self._pv_rotation.set(0)
                try:
                    self._pv_tilt.set(int(round(float(self.selected.image_tilt))))
                except (tk.TclError, ValueError, TypeError):
                    self._pv_tilt.set(0)
                self._scale_widgets["d"].state(["disabled"])
            else:
                self._frm_text.pack_forget()
                self._frm_pic.pack_forget()
                self._frm_face3.pack_forget()
                self._frm_face3.pack(fill=tk.X, after=self._frm_color_row)
                self._frm_cap.pack_forget()
                self._frm_cap.pack(fill=tk.X, after=self._txt_info)
                self._scale_widgets["d"].state(["!disabled"])
        elif self.selected_edge:
            self._pv_type.set("— arrow selected —")
            self._txt_lbl.delete("1.0", "end")
            self._txt_info.delete("1.0", "end")
            self._frm_pic.pack_forget()
            self._frm_text.pack_forget()
            self._frm_cap.pack_forget()
            self._frm_face3.pack_forget()
            self._sync_face_swatches()
            self._scale_widgets["d"].state(["!disabled"])
        else:
            self._pv_type.set("— select a layer —")
            self._txt_lbl.delete("1.0", "end")
            self._txt_info.delete("1.0", "end")
            self._frm_pic.pack_forget()
            self._frm_text.pack_forget()
            self._frm_cap.pack_forget()
            self._frm_face3.pack_forget()
            self._sync_face_swatches()
            self._scale_widgets["d"].state(["!disabled"])
        self._syncing = False

        def _upd_scroll():
            b = self._props_canvas.bbox("all")
            if b:
                self._props_canvas.configure(scrollregion=b)

        self.after_idle(_upd_scroll)

    def _on_text_props(self, _evt=None):
        if self._syncing or not self.selected:
            return
        self.selected.label = self._txt_lbl.get("1.0", "end-1c")
        self.selected.info = self._txt_info.get("1.0", "end-1c")
        self.draw_all()

    def _ap_opacity(self):
        if self.selected and not self._syncing:
            try:
                self.selected.opacity = int(self._pv_opacity.get()) / 100.0
                self.draw_all()
            except (tk.TclError, ValueError):
                pass

    def _ap_tilt(self):
        if self.selected and not self._syncing:
            try:
                self.selected.image_tilt = float(self._pv_tilt.get())
                self.draw_all()
            except (tk.TclError, ValueError):
                pass

    def _ap_rotation(self):
        if self.selected and not self._syncing:
            try:
                self.selected.image_rotation = float(self._pv_rotation.get())
                self.draw_all()
            except (tk.TclError, ValueError):
                pass

    def _short_path(self, p, maxlen=46):
        if not p or p == "(none)":
            return p or "(none)"
        p = os.path.normpath(p)
        if len(p) <= maxlen:
            return p
        return "…" + p[-(maxlen - 1) :]

    def _browse_picture(self):
        pic = (
            self.selected
            if self.selected and self.selected.layer_type == "Picture"
            else None
        )
        if not pic:
            messagebox.showinfo(
                "Picture layer",
                "Select a Picture block on the canvas first.\n\n"
                "Add one from the left palette (Picture), click the canvas, "
                "then use Browse here.",
                parent=self,
            )
            return
        ft = [
            ("PNG", "*.png"),
            ("JPEG", "*.jpg *.jpeg *.jpe"),
            ("GIF", "*.gif"),
            ("WebP", "*.webp"),
            ("TIFF", "*.tif *.tiff"),
            ("BMP", "*.bmp"),
            ("All files", "*.*"),
        ]
        p = filedialog.askopenfilename(
            parent=self,
            title="Choose image",
            filetypes=ft,
            defaultextension=".png",
        )
        if not p:
            return
        p = os.path.normpath(os.path.abspath(p))
        if not os.path.isfile(p):
            messagebox.showerror(
                "Image",
                f"File not found or not readable:\n{p}",
                parent=self,
            )
            return
        pic.image_path = p
        self._pic_path_lbl.configure(text=self._short_path(p))
        self.draw_all()
        self.cv.focus_set()

    def _clear_picture(self):
        if not self.selected or self.selected.layer_type != "Picture":
            return
        self.selected.image_path = ""
        self._pic_path_lbl.configure(text="(none)")
        self.draw_all()

    def _ap_slider(self, prop, var):
        if not self.selected or self._syncing:
            return
        try:
            v = int(float(var.get()))
            if self.selected.layer_type in ("Picture", "Text"):
                if prop == "d":
                    return
                lo = 8
                setattr(self.selected, prop, max(lo, v))
            else:
                setattr(self.selected, prop, max(1, v))
            self.draw_all()
        except (tk.TclError, ValueError):
            pass

    def _pick_color(self):
        if not self.selected:
            return
        _, hc = colorchooser.askcolor(color=self.selected.color,
                                       title="Choose Layer Color",
                                       parent=self)
        if hc:
            self.selected.color = hc
            self._swatch.configure(bg=hc)
            self._sync_face_swatches()
            self.draw_all()

    def _pick_face_color(self, which):
        if not self.selected or self.selected.layer_type in ("Picture", "Text"):
            return
        cur = (getattr(self.selected, f"color_{which}", "") or "").strip()
        init = cur if self._valid_hex_color(cur) else self._default_face_base_hex(
            self.selected, which)
        _, hc = colorchooser.askcolor(
            color=init,
            title=f"{which.capitalize()} face color",
            parent=self,
        )
        if hc:
            setattr(self.selected, f"color_{which}", hc)
            getattr(self, f"_face_{which}_swatch").configure(bg=hc)
            self.draw_all()

    def _clear_face_color(self, which):
        if not self.selected or self.selected.layer_type in ("Picture", "Text"):
            return
        setattr(self.selected, f"color_{which}", "")
        getattr(self, f"_face_{which}_swatch").configure(bg="#45475A")
        self.draw_all()

    def _pick_label_color(self):
        if not self.selected:
            return
        cur = (getattr(self.selected, "label_color", "") or "").strip()
        init = cur if cur else self._effective_label_fg(self.selected)
        _, hc = colorchooser.askcolor(color=init, title="Label caption color",
                                      parent=self)
        if hc:
            self.selected.label_color = hc
            self._cap_lbl_swatch.configure(bg=hc)
            self.draw_all()

    def _clear_label_color(self):
        if not self.selected:
            return
        self.selected.label_color = ""
        self._cap_lbl_swatch.configure(bg="#45475A")
        self.draw_all()

    def _pick_info_color(self):
        if not self.selected:
            return
        cur = (getattr(self.selected, "info_color", "") or "").strip()
        init = cur if cur else self._effective_info_fg(self.selected)
        _, hc = colorchooser.askcolor(color=init, title="Info caption color",
                                      parent=self)
        if hc:
            self.selected.info_color = hc
            self._cap_info_swatch.configure(bg=hc)
            self.draw_all()

    def _clear_info_color(self):
        if not self.selected:
            return
        self.selected.info_color = ""
        self._cap_info_swatch.configure(bg="#45475A")
        self.draw_all()

    def _pick_text_color(self):
        if not self.selected or self.selected.layer_type != "Text":
            return
        tc = (getattr(self.selected, "text_color", "") or "#1A1E2E").strip()
        _, hc = colorchooser.askcolor(color=tc, title="Free text color",
                                      parent=self)
        if hc:
            self.selected.text_color = hc
            self._txt_col_swatch.configure(bg=hc)
            self.draw_all()

    def _ap_text_style(self):
        if self._syncing or not self.selected:
            return
        if self.selected.layer_type != "Text":
            return
        try:
            self.selected.text_bold = bool(self._pv_txt_bold.get())
            self.draw_all()
        except (tk.TclError, ValueError, TypeError):
            pass

    def _ap_caption_fonts(self):
        if self._syncing or not self.selected:
            return
        try:
            self.selected.label_font_size = int(self._pv_cap_lbl_size.get())
            self.selected.info_font_size = int(self._pv_cap_info_size.get())
            if self.selected.layer_type == "Text":
                self.selected.text_font_size = self.selected.label_font_size
            self.draw_all()
        except (tk.TclError, ValueError, TypeError):
            pass

    # ── Layer operations ───────────────────────────────────────────────────────

    def _del(self):
        if self.selected in self.layers:
            lid = self.selected.id
            self.layers.remove(self.selected)
            self.selected = None
            if self.manual_conns:
                self.manual_conns = [
                    p for p in self.manual_conns
                    if p[0] != lid and p[1] != lid
                ]
            if self.selected_edge and (
                    self.selected_edge[0] == lid or self.selected_edge[1] == lid):
                self.selected_edge = None
            self._sync_props()
            self.draw_all()

    def _dup(self):
        if not self.selected:
            return
        s = self.selected
        n = Layer.__new__(Layer)
        Layer._ctr += 1
        n.id = Layer._ctr
        for attr in (
            "layer_type", "color", "w", "h", "d", "label", "info",
            "opacity", "image_path", "image_rotation", "image_tilt",
            "label_color", "info_color", "text_color", "text_bold",
            "text_font_size", "label_font_size", "info_font_size",
            "color_front", "color_top", "color_right",
        ):
            setattr(n, attr, getattr(s, attr))
        n.x = s.x + 90
        n.y = s.y
        self.layers.insert(self.layers.index(s) + 1, n)
        self.selected = n
        self._sync_props()
        self.draw_all()

    def _reorder(self, direction):
        if not self.selected or self.selected not in self.layers:
            return
        i = self.layers.index(self.selected)
        j = i + direction
        if 0 <= j < len(self.layers):
            self.layers[i], self.layers[j] = self.layers[j], self.layers[i]
            self.draw_all()

    def _auto_layout(self):
        """Re-space all layers evenly left-to-right at vertical canvas centre."""
        if not self.layers:
            return
        cy      = self.cv.winfo_height() / 2 if self.cv.winfo_height() > 1 else 350
        ordered = sorted(self.layers, key=lambda l: l.x)
        x = 90
        for layer in ordered:
            layer.x = x
            layer.y = cy
            extra = (
                0 if layer.layer_type in ("Picture", "Text") else layer.d * PERSP
            )
            x += layer.w + extra + 90
        self.draw_all()

    # ── Toggle buttons ─────────────────────────────────────────────────────────

    def _toggle_conn(self):
        self._show_conn = not self._show_conn
        self._conn_btn.configure(
            text="Hide Arrows" if self._show_conn else "Show Arrows")
        self.draw_all()

    def _toggle_arrow_mode(self):
        self._add_arrow_mode = not self._add_arrow_mode
        self._conn_pick_from = None
        self._arrow_mode_btn.configure(
            relief="sunken" if self._add_arrow_mode else "flat",
            text=("Done adding arrows" if self._add_arrow_mode
                  else "Add arrow (click A→B)"),
        )
        self.draw_all()

    def _clear_manual_conns(self):
        self.manual_conns.clear()
        self._conn_pick_from = None
        self.selected_edge = None
        self.draw_all()

    def _toggle_grid(self):
        self._show_grid = not self._show_grid
        self._grid_btn.configure(
            text="Hide Grid" if self._show_grid else "Show Grid")
        self.draw_all()

    def _toggle_cv_bg(self):
        self._light_cv = not self._light_cv
        self._bg_btn.configure(
            text="Dark Canvas" if self._light_cv else "Light Canvas")
        self.draw_all()

    # ── File operations ────────────────────────────────────────────────────────

    def _new(self):
        if self.layers:
            if not messagebox.askyesno("New Figure", "Clear all layers?", parent=self):
                return
        self.layers.clear()
        self.manual_conns.clear()
        self._conn_pick_from = None
        self.selected = None
        self.selected_edge = None
        self._zoom = 1.0
        self._view_pan_x = self._view_pan_y = 0
        self._sync_props()
        self.draw_all()

    def _save(self):
        p = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            title="Save Figure",
            parent=self)
        if p:
            payload = {
                "version": 1,
                "layers": [l.to_dict() for l in self.layers],
                "connections": self.manual_conns,
            }
            with open(p, "w") as f:
                json.dump(payload, f, indent=2)
            messagebox.showinfo("Saved", f"Saved to:\n{p}", parent=self)

    def _load(self):
        p = filedialog.askopenfilename(
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            title="Load Figure",
            parent=self)
        if p:
            with open(p) as f:
                data = json.load(f)
            err = self._validate_figure_payload(data)
            if err:
                messagebox.showerror("Invalid figure", err, parent=self)
                return
            self._apply_figure_payload(data)

    def _validate_figure_payload(self, data):
        if not isinstance(data, dict):
            return "Root value must be a JSON object."
        layers = data.get("layers")
        if not isinstance(layers, list) or len(layers) == 0:
            return "Expected a non-empty \"layers\" array."
        need = ("id", "layer_type", "x", "y", "w", "h", "d", "color")
        for i, d in enumerate(layers):
            if not isinstance(d, dict):
                return f"layers[{i}] must be an object."
            for k in need:
                if k not in d:
                    return f"layers[{i}] missing required key \"{k}\"."
            if d.get("layer_type") not in LAYER_DEFAULTS:
                return (
                    f"Unknown layer_type \"{d['layer_type']}\" at layers[{i}]. "
                    f"Use one of: {', '.join(LAYER_DEFAULTS)}"
                )
        conns = data.get("connections")
        if conns is not None and not isinstance(conns, list):
            return "\"connections\" must be an array of [from_id, to_id] pairs."
        return None

    def _apply_figure_payload(self, data):
        def _norm_layer(d):
            m = dict(d)
            m.setdefault("label", "")
            m.setdefault("info", "")
            return m

        if isinstance(data, dict) and "layers" in data:
            self.layers = [
                Layer.from_dict(_norm_layer(d)) for d in data["layers"]
            ]
            self.manual_conns = data.get("connections") or []
        else:
            self.layers = [Layer.from_dict(_norm_layer(d)) for d in data]
            self.manual_conns = []
        self._conn_pick_from = None
        self.selected = None
        self.selected_edge = None
        self._zoom = 1.0
        self._view_pan_x = self._view_pan_y = 0
        if self.layers:
            Layer._ctr = max(L.id for L in self.layers)
        self._sync_props()
        self.draw_all()

    @staticmethod
    def _parse_ai_json_text(text):
        s = text.strip()
        if s.startswith("```"):
            s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
            s = re.sub(r"\s*```\s*$", "", s)
        s = s.strip()
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            pass
        start = s.find("{")
        if start < 0:
            raise ValueError("Model did not return a JSON object.")
        depth = 0
        for i in range(start, len(s)):
            if s[i] == "{":
                depth += 1
            elif s[i] == "}":
                depth -= 1
                if depth == 0:
                    return json.loads(s[start : i + 1])
        raise ValueError("Unbalanced JSON braces in model output.")

    def _open_ai_dialog(self):
        load_dotenv()

        win = tk.Toplevel(self)
        win.title("AI architecture → figure (Groq)")
        win.configure(bg=PROP_BG)
        win.geometry("600x500")
        win.transient(self)
        win.grab_set()

        tk.Label(
            win,
            text="Describe your CNN (layers, shapes, order). The model returns JSON for this app.",
            bg=PROP_BG, fg=FG, font=("Segoe UI", 9), wraplength=560, justify="left",
        ).pack(anchor="w", padx=12, pady=(10, 4))

        env_hint = (
            f"Requires: pip install groq. Key in .env (GROQ_API_KEY=…). "
            f"Optional GROQ_MODEL (default {GROQ_MODEL_DEFAULT}). "
            f"File: {_DOTENV_PATH}"
        )
        tk.Label(
            win,
            text=env_hint,
            bg=PROP_BG, fg="#6C7086", font=("Segoe UI", 8), wraplength=560, justify="left",
        ).pack(anchor="w", padx=12, pady=(0, 6))

        tk.Label(
            win, text="Your architecture:",
            bg=PROP_BG, fg=FG, font=("Segoe UI", 9, "bold"),
        ).pack(anchor="w", padx=12, pady=(8, 2))
        body = tk.Frame(win, bg=PROP_BG)
        body.pack(fill=tk.BOTH, expand=True, padx=12, pady=4)
        desc = tk.Text(
            body, height=14, wrap="word",
            bg="#313244", fg=FG, insertbackground=FG, relief="flat", font=("Segoe UI", 10),
        )
        desc.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb = ttk.Scrollbar(body, orient="vertical", command=desc.yview)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        desc.configure(yscrollcommand=sb.set)
        desc.insert("1.0", "Example: 3D CNN for video: Input 46×54×14, two Conv3D+ReLU with 32 then 64 filters, MaxPool between, GlobalMaxPool, Dense 64, Dropout, binary Output.")

        status = tk.Label(
            win, text="Ready.", bg=PROP_BG, fg="#6C7086",
            font=("Segoe UI", 8), wraplength=580, justify="left",
        )
        status.pack(anchor="w", padx=12, pady=4)

        btn_fr = tk.Frame(win, bg=PROP_BG)
        btn_fr.pack(fill=tk.X, padx=12, pady=(4, 12))

        def close():
            win.grab_release()
            win.destroy()

        def do_gen():
            load_dotenv()
            api_key = os.environ.get("GROQ_API_KEY", "").strip()
            if not api_key:
                messagebox.showwarning(
                    "API key",
                    f"Add a line to your .env file next to this script:\n\n"
                    f"GROQ_API_KEY=your_key_here\n\n"
                    f"Expected file:\n{_DOTENV_PATH}",
                    parent=win,
                )
                return
            user_txt = desc.get("1.0", "end-1c").strip()
            if not user_txt:
                messagebox.showwarning("Describe", "Enter an architecture description.", parent=win)
                return
            if self.layers and not messagebox.askyesno(
                "Replace figure?",
                "Replace the current canvas with the generated figure?",
                parent=win,
            ):
                return
            gen_btn.configure(state="disabled")
            status.configure(text="Calling Groq…", fg=ACCENT)
            win.update_idletasks()

            def worker():
                err_msg = None
                payload = None
                try:
                    if Groq is None:
                        err_msg = (
                            "The Groq Python SDK is not installed.\n"
                            "Run in a terminal:  pip install groq"
                        )
                    else:
                        load_dotenv()
                        model = (
                            os.environ.get("GROQ_MODEL", "").strip()
                            or GROQ_MODEL_DEFAULT
                        )
                        client = Groq(api_key=api_key)
                        completion = client.chat.completions.create(
                            model=model,
                            messages=[
                                {
                                    "role": "system",
                                    "content": AI_FIGURE_SYSTEM_PROMPT,
                                },
                                {"role": "user", "content": user_txt},
                            ],
                            temperature=0.25,
                            max_completion_tokens=8192,
                            stream=False,
                        )
                        content = (
                            (completion.choices[0].message.content or "")
                            if completion.choices
                            else ""
                        )
                        if not content:
                            err_msg = "Empty response from API."
                        else:
                            payload = App._parse_ai_json_text(content)
                            normalize_ai_layers_to_app_defaults(payload)
                except (json.JSONDecodeError, ValueError, KeyError) as e:
                    err_msg = f"Could not parse model JSON: {e}"
                except Exception as e:
                    err_msg = str(e)
                    if len(err_msg) < 12:
                        err_msg = repr(e)

                def finish():
                    gen_btn.configure(state="normal")
                    if err_msg:
                        status.configure(text=err_msg[:500], fg="#F38BA8")
                        messagebox.showerror("AI generate failed", err_msg[:1200], parent=win)
                        return
                    verr = self._validate_figure_payload(payload)
                    if verr:
                        status.configure(text=verr, fg="#F38BA8")
                        messagebox.showerror("Invalid figure JSON", verr, parent=win)
                        return
                    try:
                        self._apply_figure_payload(payload)
                    except Exception as e:
                        status.configure(text=str(e), fg="#F38BA8")
                        messagebox.showerror("Apply failed", str(e), parent=win)
                        return
                    status.configure(text="Figure applied on canvas.", fg="#A6E3A1")
                    messagebox.showinfo(
                        "Done",
                        "Architecture loaded on the canvas. Use Save to keep a .json copy.",
                        parent=win,
                    )

                self.after(0, finish)

            threading.Thread(target=worker, daemon=True).start()

        gen_btn = tk.Button(
            btn_fr, text="Generate figure", command=do_gen,
            bg=ACCENT, fg="#1E1E2E", relief="flat", font=("Segoe UI", 9, "bold"),
            padx=14, pady=4,
            activebackground="#B4BEFE", activeforeground="#1E1E2E",
        )
        gen_btn.pack(side=tk.LEFT, padx=(0, 8))
        tk.Button(
            btn_fr, text="Close", command=close,
            bg="#45475A", fg=FG, relief="flat", font=("Segoe UI", 9),
            padx=12, pady=4,
        ).pack(side=tk.LEFT)

        desc.focus_set()

    def _export_eps(self):
        """Vector EPS — works with zero extra packages. Best for LaTeX papers."""
        p = filedialog.asksaveasfilename(
            defaultextension=".eps",
            filetypes=[("EPS vector", "*.eps"), ("All files", "*.*")],
            title="Export EPS (vector — best for papers)",
            parent=self)
        if p:
            self.cv.postscript(file=p, colormode="color")
            messagebox.showinfo(
                "Exported EPS",
                f"Vector file saved:\n{p}\n\n"
                "Include in LaTeX:  \\includegraphics{{{p}}}",
                parent=self)

    def _export_png(self):
        """Raster PNG — requires Pillow (pip install Pillow)."""
        try:
            from PIL import ImageGrab
        except ImportError:
            messagebox.showerror(
                "Pillow not installed",
                "Run:  pip install Pillow\n"
                "Then try Export PNG again.\n\n"
                "Export EPS works right now without any extra packages.",
                parent=self)
            return

        p = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG image", "*.png"), ("All files", "*.*")],
            title="Export PNG",
            parent=self)
        if not p:
            return

        self.cv.update()
        x = self.cv.winfo_rootx()
        y = self.cv.winfo_rooty()
        w = self.cv.winfo_width()
        h = self.cv.winfo_height()
        ImageGrab.grab(bbox=(x, y, x + w, y + h)).save(p)
        messagebox.showinfo("Exported PNG", f"PNG saved:\n{p}", parent=self)


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    App().mainloop()
