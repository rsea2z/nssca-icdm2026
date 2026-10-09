from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


W, H = 3840, 1320
BLACK = (36, 38, 37)
TEAL = (18, 124, 120)
AMBER = (212, 154, 42)
RED = (184, 74, 58)
COBALT = (63, 111, 147)
GRAY = (216, 211, 199)
AXIS = (115, 122, 122)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Draw a 4K research-framework schematic following the research-figure harness style.")
    parser.add_argument("--output", default="figures/framework_harness.png")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(image)

    fonts = {
        "title": font(86, bold=True),
        "subtitle": font(42),
        "panel": font(48, bold=True),
        "label": font(38, bold=True),
        "small": font(30),
        "tiny": font(25),
    }

    draw_title(draw, fonts)
    draw_pipeline(draw, fonts)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    image.save(out, quality=96)
    print(out)


def draw_title(draw: ImageDraw.ImageDraw, fonts: dict[str, ImageFont.FreeTypeFont]) -> None:
    title = "Training-Free Normal Calibration for Frozen VLM Anomaly Maps"
    subtitle = "1/2/4 normal support images calibrate an exported AnomalyCLIP map without prompt tuning"
    centered(draw, title, (W // 2, 95), fonts["title"], BLACK)
    centered(draw, subtitle, (W // 2, 170), fonts["subtitle"], AXIS)


def draw_pipeline(draw: ImageDraw.ImageDraw, fonts: dict[str, ImageFont.FreeTypeFont]) -> None:
    y0, y1 = 250, 1260
    panels = [
        (150, y0, 690, y1, "Target Support", "normal-only", TEAL),
        (820, y0, 1390, y1, "Frozen Export", "AnomalyCLIP", COBALT),
        (1520, y0, 2090, y1, "Normal Memory", "prototype bank", TEAL),
        (2220, y0, 2790, y1, "Map Fusion", "two energies", AMBER),
        (2920, y0, 3690, y1, "Image Scoring", "top-0.1% score", TEAL),
    ]
    for panel in panels:
        card(draw, panel[:4], panel[2] - panel[0], fill=(250, 252, 252), outline=panel[6])
        centered(draw, panel[4], ((panel[0] + panel[2]) // 2, panel[1] + 68), fonts["panel"], BLACK)
        centered(draw, panel[5], ((panel[0] + panel[2]) // 2, panel[1] + 124), fonts["small"], panel[6])

    draw_support_panel(draw, panels[0][:4], fonts)
    draw_export_panel(draw, panels[1][:4], fonts)
    draw_memory_panel(draw, panels[2][:4], fonts)
    draw_fusion_panel(draw, panels[3][:4], fonts)
    draw_decision_panel(draw, panels[4][:4], fonts)

    for x in [740, 1440, 2140, 2840]:
        arrow(draw, (x, 690), (x + 250, 690), TEAL)


def draw_support_panel(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], fonts: dict[str, ImageFont.FreeTypeFont]) -> None:
    x0, y0, x1, _ = box
    for i, x in enumerate([230, 365, 500]):
        image_card(draw, (x, y0 + 250, x + 110, y0 + 390), TEAL if i < 2 else GRAY)
        centered(draw, "N", (x + 55, y0 + 455), fonts["label"], TEAL)
    centered(draw, "1 / 2 / 4 shots", ((x0 + x1) // 2, y0 + 560), fonts["label"], BLACK)
    patch_grid(draw, (255, y0 + 650, 585, y0 + 950), TEAL)
    centered(draw, "normal patch tokens", ((x0 + x1) // 2, y0 + 960), fonts["small"], AXIS)


def draw_export_panel(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], fonts: dict[str, ImageFont.FreeTypeFont]) -> None:
    x0, y0, x1, _ = box
    chip(draw, ((x0 + x1) // 2 - 140, y0 + 255, (x0 + x1) // 2 + 140, y0 + 515), COBALT)
    centered(draw, "frozen VLM", ((x0 + x1) // 2, y0 + 575), fonts["label"], BLACK)
    heatmap(draw, (925, y0 + 665, 1115, y0 + 900), COBALT)
    patch_grid(draw, (1165, y0 + 665, 1325, y0 + 900), COBALT)
    centered(draw, "base map", (1020, y0 + 955), fonts["tiny"], AXIS)
    centered(draw, "features", (1245, y0 + 955), fonts["tiny"], AXIS)


def draw_memory_panel(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], fonts: dict[str, ImageFont.FreeTypeFont]) -> None:
    x0, y0, x1, _ = box
    for i in range(4):
        offset = i * 34
        draw.rounded_rectangle((x0 + 150 + offset, y0 + 255 + offset, x1 - 150 + offset, y0 + 395 + offset), 24, fill=(235, 248, 247), outline=TEAL, width=5)
    centered(draw, "P_c = {p_j}", ((x0 + x1) // 2, y0 + 560), fonts["label"], BLACK)
    formula(draw, (x0 + 80, y0 + 670, x1 - 80, y0 + 790), "nearest-prototype distance", fonts)
    formula(draw, (x0 + 80, y0 + 830, x1 - 80, y0 + 950), "normality distance map", fonts)


def draw_fusion_panel(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], fonts: dict[str, ImageFont.FreeTypeFont]) -> None:
    x0, y0, x1, _ = box
    heatmap(draw, (x0 + 105, y0 + 250, x0 + 300, y0 + 500), COBALT)
    plus(draw, (x0 + 360, y0 + 375), AMBER)
    heatmap(draw, (x0 + 430, y0 + 250, x0 + 625, y0 + 500), TEAL)
    centered(draw, "zero-shot", (x0 + 202, y0 + 555), fonts["tiny"], AXIS)
    centered(draw, "prototype", (x0 + 528, y0 + 555), fonts["tiny"], AXIS)
    formula(draw, (x0 + 70, y0 + 660, x1 - 70, y0 + 785), "fused map = zero-shot + prototype", fonts)
    formula(draw, (x0 + 70, y0 + 835, x1 - 70, y0 + 960), "top-0.1% pooling", fonts)


def draw_decision_panel(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], fonts: dict[str, ImageFont.FreeTypeFont]) -> None:
    x0, y0, x1, _ = box
    heatmap(draw, (x0 + 80, y0 + 235, x0 + 330, y0 + 535), TEAL)
    gauge(draw, (x0 + 455, y0 + 240, x0 + 705, y0 + 520), AMBER)
    centered(draw, "heatmap", (x0 + 205, y0 + 595), fonts["tiny"], AXIS)
    centered(draw, "image score", (x0 + 580, y0 + 595), fonts["tiny"], AXIS)
    formula(draw, (x0 + 95, y0 + 705, x1 - 95, y0 + 830), "score = mean top-0.1% pixels", fonts)
    formula(draw, (x0 + 95, y0 + 880, x1 - 95, y0 + 1005), "global rule shared by classes", fonts)


def card(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], width: int, fill: tuple[int, int, int], outline: tuple[int, int, int]) -> None:
    draw.rounded_rectangle(shadow(box, 16), 34, fill=(226, 232, 232))
    draw.rounded_rectangle(box, 34, fill=fill, outline=outline, width=6)


def image_card(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], color: tuple[int, int, int]) -> None:
    draw.rounded_rectangle(box, 18, fill=(248, 250, 250), outline=color, width=5)
    x0, y0, x1, y1 = box
    draw.polygon([(x0 + 10, y1 - 18), (x0 + 46, y1 - 66), (x0 + 78, y1 - 31), (x1 - 10, y1 - 86), (x1 - 10, y1 - 10), (x0 + 10, y1 - 10)], fill=soft(color))
    draw.ellipse((x0 + 72, y0 + 22, x0 + 100, y0 + 50), fill=color)


def patch_grid(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], color: tuple[int, int, int]) -> None:
    x0, y0, x1, y1 = box
    draw.rounded_rectangle(box, 20, fill=(250, 252, 252), outline=color, width=5)
    cols, rows = 5, 4
    pad = 24
    cw = (x1 - x0 - 2 * pad) / cols
    ch = (y1 - y0 - 2 * pad) / rows
    for r in range(rows):
        for c in range(cols):
            fill = soft(color) if (r + c) % 3 else color
            draw.rounded_rectangle((x0 + pad + c * cw + 6, y0 + pad + r * ch + 6, x0 + pad + (c + 1) * cw - 6, y0 + pad + (r + 1) * ch - 6), 8, fill=fill)


def chip(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], color: tuple[int, int, int]) -> None:
    x0, y0, x1, y1 = box
    for x in range(x0 - 38, x1 + 39, 38):
        draw.line((x, y0 - 24, x, y0), fill=color, width=6)
        draw.line((x, y1, x, y1 + 24), fill=color, width=6)
    for y in range(y0 - 38, y1 + 39, 38):
        draw.line((x0 - 24, y, x0, y), fill=color, width=6)
        draw.line((x1, y, x1 + 24, y), fill=color, width=6)
    draw.rounded_rectangle(box, 28, fill=(239, 246, 251), outline=color, width=6)
    draw.rounded_rectangle((x0 + 52, y0 + 52, x1 - 52, y1 - 52), 22, fill="white", outline=color, width=4)
    centered(draw, "CLIP", ((x0 + x1) // 2, (y0 + y1) // 2 - 8), font(46, True), color)


def heatmap(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], color: tuple[int, int, int]) -> None:
    x0, y0, x1, y1 = box
    draw.rounded_rectangle(box, 24, fill=(250, 250, 248), outline=color, width=5)
    rows, cols = 7, 6
    for r in range(rows):
        for c in range(cols):
            t = (r + 2 * c) / (rows + 2 * cols)
            fill = blend((248, 248, 244), color, min(0.88, 0.18 + t))
            draw.rectangle((x0 + 18 + c * (x1 - x0 - 36) / cols, y0 + 18 + r * (y1 - y0 - 36) / rows, x0 + 18 + (c + 1) * (x1 - x0 - 36) / cols, y0 + 18 + (r + 1) * (y1 - y0 - 36) / rows), fill=fill)
    draw.rounded_rectangle(box, 24, outline=color, width=5)


def gauge(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], color: tuple[int, int, int]) -> None:
    x0, y0, x1, y1 = box
    draw.rounded_rectangle(box, 24, fill=(255, 252, 246), outline=color, width=5)
    cx, cy = (x0 + x1) // 2, y0 + 160
    draw.arc((cx - 90, cy - 90, cx + 90, cy + 90), 200, 340, fill=color, width=12)
    draw.line((cx, cy, cx + 58, cy - 54), fill=RED, width=8)
    draw.ellipse((cx - 12, cy - 12, cx + 12, cy + 12), fill=color)
    centered(draw, "top-k", (cx, y1 - 50), font(34, True), BLACK)


def mini_chart(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], fonts: dict[str, ImageFont.FreeTypeFont]) -> None:
    x0, y0, x1, y1 = box
    draw.rounded_rectangle(box, 20, fill="white", outline=GRAY, width=4)
    base = y1 - 48
    xs = [x0 + 70, x0 + 180, x0 + 290, x0 + 400, x0 + 510]
    zs = [base - 85, base - 100, base - 92, base - 98, base - 96]
    ours = [base - 145, base - 158, base - 168, base - 178, base - 185]
    draw.line(list(zip(xs, zs)), fill=AXIS, width=8)
    draw.line(list(zip(xs, ours)), fill=TEAL, width=8)
    for x, y in zip(xs, ours):
        draw.ellipse((x - 11, y - 11, x + 11, y + 11), fill=TEAL)
    for x, y in zip(xs, zs):
        draw.ellipse((x - 10, y - 10, x + 10, y + 10), fill=AXIS)
    centered(draw, "normal scores shift down", ((x0 + x1) // 2, y0 + 32), fonts["tiny"], BLACK)


def formula(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], text: str, fonts: dict[str, ImageFont.FreeTypeFont]) -> None:
    draw.rounded_rectangle(box, 20, fill="white", outline=GRAY, width=4)
    centered(draw, text, ((box[0] + box[2]) // 2, (box[1] + box[3]) // 2 - 4), fonts["small"], BLACK)


def tag(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], text: str, fnt: ImageFont.FreeTypeFont, color: tuple[int, int, int]) -> None:
    draw.rounded_rectangle(box, 32, fill=(255, 255, 255), outline=color, width=5)
    centered(draw, text, ((box[0] + box[2]) // 2, (box[1] + box[3]) // 2 - 2), fnt, color)


def plus(draw: ImageDraw.ImageDraw, center: tuple[int, int], color: tuple[int, int, int]) -> None:
    x, y = center
    draw.line((x - 34, y, x + 34, y), fill=color, width=12)
    draw.line((x, y - 34, x, y + 34), fill=color, width=12)


def arrow(draw: ImageDraw.ImageDraw, start: tuple[int, int], end: tuple[int, int], color: tuple[int, int, int]) -> None:
    x0, y0 = start
    x1, y1 = end
    draw.line((x0, y0, x1, y1), fill=color, width=12)
    draw.polygon([(x1, y1), (x1 - 42, y1 - 28), (x1 - 42, y1 + 28)], fill=color)


def centered(draw: ImageDraw.ImageDraw, text: str, xy: tuple[int, int], fnt: ImageFont.FreeTypeFont, fill: tuple[int, int, int]) -> None:
    box = draw.textbbox((0, 0), text, font=fnt)
    draw.text((xy[0] - (box[2] - box[0]) / 2, xy[1] - (box[3] - box[1]) / 2), text, font=fnt, fill=fill)


def soft(color: tuple[int, int, int]) -> tuple[int, int, int]:
    return blend((255, 255, 255), color, 0.22)


def blend(a: tuple[int, int, int], b: tuple[int, int, int], t: float) -> tuple[int, int, int]:
    return tuple(int(a[i] * (1 - t) + b[i] * t) for i in range(3))


def shadow(box: tuple[int, int, int, int], offset: int) -> tuple[int, int, int, int]:
    return (box[0] + offset, box[1] + offset, box[2] + offset, box[3] + offset)


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        Path("C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf"),
        Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ]
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


if __name__ == "__main__":
    main()
