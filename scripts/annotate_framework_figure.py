from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Overlay deterministic labels on the generated framework background.")
    parser.add_argument("--input", default="output/imagegen/framework-bg.png")
    parser.add_argument("--output", default="figures/framework.png")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image = Image.open(args.input).convert("RGBA")
    draw = ImageDraw.Draw(image, "RGBA")
    width, _ = image.size

    title_font = load_font(54, bold=True)
    label_font = load_font(28, bold=True)
    sub_font = load_font(20, bold=False)

    title = "Few-Shot Normal Calibration for Object-Agnostic ZSAD"
    title_box = draw.textbbox((0, 0), title, font=title_font)
    draw.text(((width - (title_box[2] - title_box[0])) / 2, 72), title, font=title_font, fill=(28, 42, 48, 255))

    labels = [
        ((54, 196, 430, 285), "Target Data", "images + 1/2/4 normal shots"),
        ((560, 196, 900, 285), "Frozen AnomalyCLIP", "base map and patch features"),
        ((990, 196, 1255, 285), "Prototype Bank", "per-class memory"),
        ((1344, 196, 1612, 285), "Map Fusion", "distance map + zero-shot map"),
        ((1726, 196, 1994, 285), "Calibrated Output", "heatmap and image score"),
    ]
    for box, header, subtitle in labels:
        draw_label(draw, box, header, subtitle, label_font, sub_font)

    notes = [
        ((548, 908, 910, 978), "No target prompt tuning"),
        ((965, 908, 1262, 978), "Normal-only supervision"),
        ((1322, 908, 1644, 978), "Plug-in calibration layer"),
    ]
    for box, text in notes:
        draw_note(draw, box, text, sub_font)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(output, quality=95)
    print(output)


def draw_label(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], header: str, subtitle: str, label_font, sub_font) -> None:
    x0, y0, x1, y1 = box
    draw.rounded_rectangle(box, radius=18, fill=(255, 255, 255, 230), outline=(40, 96, 105, 160), width=2)
    header_box = draw.textbbox((0, 0), header, font=label_font)
    sub_box = draw.textbbox((0, 0), subtitle, font=sub_font)
    draw.text(((x0 + x1 - (header_box[2] - header_box[0])) / 2, y0 + 12), header, font=label_font, fill=(25, 43, 49, 255))
    draw.text(((x0 + x1 - (sub_box[2] - sub_box[0])) / 2, y0 + 54), subtitle, font=sub_font, fill=(70, 86, 92, 255))


def draw_note(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], text: str, font) -> None:
    x0, y0, x1, y1 = box
    draw.rounded_rectangle(box, radius=20, fill=(248, 252, 251, 235), outline=(230, 121, 56, 150), width=2)
    text_box = draw.textbbox((0, 0), text, font=font)
    draw.text(
        ((x0 + x1 - (text_box[2] - text_box[0])) / 2, y0 + 22),
        text,
        font=font,
        fill=(61, 74, 80, 255),
    )


def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
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
