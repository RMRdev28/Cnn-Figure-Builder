# CNN Figure Builder

A desktop **CNN / neural-network diagram editor** with a PlotNeuralNet-style palette: draggable **3D blocks** (convolution, pooling, dense, etc.), **teal arrows** between layers, optional **grid**, light/dark canvas, **opacity** per block, **Picture** and **Text** layers, and export to **JSON**, **SVG**, or **PNG** (PNG needs Pillow).

The app is a single Python script with a Tkinter UI—no web server.

## Requirements

- **Python 3.9+** (3.10+ recommended).
- **Tkinter** — included with the official Windows and macOS Python installers. On many Linux distributions you must install it separately, for example:
  - Debian / Ubuntu: `sudo apt install python3-tk`
  - Fedora: `sudo dnf install python3-tkinter`

## Install optional packages

From the project folder:

```bash
pip install -r requirements.txt
```

| Package | Purpose |
|--------|---------|
| **groq** | “Generate with AI” — sends a prompt to Groq and loads the returned JSON onto the canvas. |
| **Pillow** | **Picture** layers (load an image file) and **PNG** screenshot export. Without Pillow, the rest of the app still runs. |

## Configuration for AI generation

1. Copy `.env.example` to `.env`.
2. Set `GROQ_API_KEY` to your [Groq](https://console.groq.com/) API key.
3. Optionally set `GROQ_MODEL` (see `.env.example`).

If `groq` is not installed or the key is missing, AI generation is disabled; everything else works.

## Run the application

```bash
python cnn_figure_builder.py
```

Use the **Save** / **Load** actions in the app (or the menu entries wired to them) to write and read `.json` figure files.

---

## Figure JSON format

The app expects a **JSON object** at the root (not a bare array). You can **hand-write** this format or **export** a file with **Save** and edit it.

### Root object

| Key | Type | Required | Description |
|-----|------|----------|-------------|
| `version` | number | Recommended | Use `1` (what **Save** writes). |
| `layers` | array | **Yes** | Non-empty list of layer objects. |
| `connections` | array | No | Directed edges: each item is `[from_layer_id, to_layer_id]`. Use `[]` or omit to let the app **auto-chain** layers left-to-right by `x`. |

### Each layer object — required keys

Validation on **Load** requires **exactly** these keys on every layer:

| Key | Type | Description |
|-----|------|-------------|
| `id` | integer | Unique positive id (used by `connections`). |
| `layer_type` | string | One of the supported types (see below). |
| `x`, `y` | number | Canvas position in **world** coordinates. |
| `w`, `h`, `d` | number | **Thickness** (width of the block face), **height**, **depth** (3D extent). For **Picture** and **Text**, `d` is `0`. |
| `color` | string | Fill color, e.g. `"#F5DEB3"`. |

### Supported `layer_type` values

`Input`, `Picture`, `Text`, `Conv2D`, `Conv3D`, `MaxPool`, `AvgPool`, `GlobalPool`, `BatchNorm`, `Dropout`, `Flatten`, `Dense`, `Output`

Names are **case-sensitive** and must match exactly.

### Optional keys (recommended for a full round-trip)

If you omit them, the app fills sensible defaults when loading (see `Layer.from_dict` in the script):

- `label`, `info` — strings (secondary line under the block for `info` on 3D types).
- `opacity` — `0.0`–`1.0` (values `> 1` are treated as percentages and scaled).
- `image_path`, `image_rotation`, `image_tilt` — for **Picture** and 3D decoration; use `""` and `0` if unused.
- `label_color`, `info_color`, `text_color`, `text_bold`, `text_font_size`, `label_font_size`, `info_font_size`
- `color_front`, `color_top`, `color_right` — per-face overrides; empty string uses the main `color`.

### Minimal example (valid for **Load**)

```json
{
  "version": 1,
  "connections": [],
  "layers": [
    {
      "id": 1,
      "layer_type": "Input",
      "x": 120,
      "y": 350,
      "w": 28,
      "h": 80,
      "d": 90,
      "color": "#F5DEB3"
    },
    {
      "id": 2,
      "layer_type": "Conv2D",
      "x": 280,
      "y": 350,
      "w": 28,
      "h": 80,
      "d": 22,
      "color": "#F5DEB3"
    },
    {
      "id": 3,
      "layer_type": "MaxPool",
      "x": 420,
      "y": 350,
      "w": 9,
      "h": 62,
      "d": 14,
      "color": "#CD5C5C"
    }
  ]
}
```

After loading, open the properties panel to set **labels**, **opacity**, and optional **connections**. To see the **exact** field set the app uses (including fonts and image fields), place blocks on the canvas and use **Save** — then use that file as a template.

### Custom arrows

Set `connections` to explicit pairs, for example:

```json
"connections": [[1, 3], [2, 3]]
```

Each pair is **from_id → to_id**. Ids must exist in `layers`.

---

## Project files

- `cnn_figure_builder.py` — main application.
- `figure_3dcnn_architecture.json` — example figure (may include extra optional keys from a full save).
