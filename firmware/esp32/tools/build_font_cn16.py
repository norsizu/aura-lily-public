#!/usr/bin/env python3
"""
Build firmware/esp32/assets/font_cn16.bin from a TTF/OTF font.

Output format:
  16-byte header:
    magic "CNFONT\\0\\0"
    u16 cn_count
    u16 ascii_start
    u16 ascii_end
    u16 reserved
  cn_index:  cn_count * u16 codepoints (sorted)
  cn_bitmap: cn_count * 32 bytes (16 rows * 2 bytes)
  ascii:     (ascii_end - ascii_start + 1) * 16 bytes (16 rows * 1 byte)

By default this script reuses the codepoint index embedded in the existing
font_cn16.bin, so replacing the font does not accidentally drop characters.
"""

from __future__ import annotations

import argparse
import struct
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


MAGIC = b"CNFONT\x00\x00"
ASCII_START = 32
ASCII_END = 126
CELL_W = 16
CELL_H = 16


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Build font_cn16.bin from a TTF/OTF font.")
    parser.add_argument("--font", required=True, help="Path to source TTF/OTF font for CJK glyphs.")
    parser.add_argument(
        "--ascii-font",
        help="Optional TTF/OTF font for ASCII glyphs. Defaults to --font.",
    )
    parser.add_argument(
        "--existing",
        default=str(root / "assets" / "font_cn16.bin"),
        help="Existing font_cn16.bin used as the source of codepoint coverage.",
    )
    parser.add_argument(
        "--output",
        default=str(root / "assets" / "font_cn16.bin"),
        help="Output font_cn16.bin path.",
    )
    parser.add_argument(
        "--include-text",
        action="append",
        default=[],
        help="Additional UTF-8 text whose non-ASCII codepoints should be included.",
    )
    parser.add_argument(
        "--include-file",
        action="append",
        default=[],
        help="UTF-8 file to scan for additional non-ASCII codepoints.",
    )
    parser.add_argument("--cn-size", type=int, default=12, help="Chinese glyph render size.")
    parser.add_argument("--ascii-size", type=int, default=12, help="ASCII glyph render size.")
    parser.add_argument(
        "--preserve-existing-cjk",
        action="store_true",
        help="Reuse existing CJK bitmaps and only rebuild the ASCII bitmap block.",
    )
    return parser.parse_args()


def load_existing_codepoints(path: Path) -> list[int]:
    data = path.read_bytes()
    if data[:8] != MAGIC:
        raise ValueError(f"Invalid existing font magic: {path}")
    cn_count = struct.unpack_from("<H", data, 8)[0]
    start = 16
    return list(struct.unpack_from(f"<{cn_count}H", data, start))


def load_existing_cn_bitmaps(path: Path) -> bytes:
    data = path.read_bytes()
    if data[:8] != MAGIC:
        raise ValueError(f"Invalid existing font magic: {path}")
    cn_count = struct.unpack_from("<H", data, 8)[0]
    start = 16 + cn_count * 2
    end = start + cn_count * 32
    return data[start:end]


def render_glyph(ch: str, font: ImageFont.FreeTypeFont, width: int, height: int) -> Image.Image:
    img = Image.new("1", (width, height), 0)
    draw = ImageDraw.Draw(img)
    bbox = font.getbbox(ch)
    if bbox is None:
        return img

    left, top, right, bottom = bbox
    glyph_w = right - left
    glyph_h = bottom - top
    x = (width - glyph_w) // 2 - left
    y = (height - glyph_h) // 2 - top
    draw.text((x, y), ch, fill=1, font=font)
    return img


def pack_cn_bitmap(img: Image.Image) -> bytes:
    out = bytearray()
    for y in range(CELL_H):
        row = 0
        for x in range(CELL_W):
            if img.getpixel((x, y)):
                row |= 1 << (15 - x)
        out.extend(struct.pack(">H", row))
    return bytes(out)


def pack_ascii_bitmap(img: Image.Image) -> bytes:
    out = bytearray()
    for y in range(CELL_H):
        row = 0
        for x in range(8):
            if img.getpixel((x, y)):
                row |= 1 << (7 - x)
        out.append(row)
    return bytes(out)


def main() -> None:
    args = parse_args()
    font_path = Path(args.font).expanduser().resolve()
    ascii_font_path = Path(args.ascii_font or args.font).expanduser().resolve()
    existing_path = Path(args.existing).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    if not font_path.exists():
        raise FileNotFoundError(font_path)
    if not ascii_font_path.exists():
        raise FileNotFoundError(ascii_font_path)
    if not existing_path.exists():
        raise FileNotFoundError(existing_path)

    codepoints = set(load_existing_codepoints(existing_path))
    for text in args.include_text:
        codepoints.update(ord(ch) for ch in text if ord(ch) >= 128)
    for file_name in args.include_file:
        file_text = Path(file_name).expanduser().resolve().read_text(encoding="utf-8", errors="ignore")
        codepoints.update(ord(ch) for ch in file_text if ord(ch) >= 128)
    codepoints = sorted(codepoints)
    ascii_font = ImageFont.truetype(str(ascii_font_path), args.ascii_size)

    cn_bitmaps = bytearray()
    if args.preserve_existing_cjk:
        cn_bitmaps.extend(load_existing_cn_bitmaps(existing_path))
    else:
        cn_font = ImageFont.truetype(str(font_path), args.cn_size)
        for cp in codepoints:
            ch = chr(cp)
            img = render_glyph(ch, cn_font, CELL_W, CELL_H)
            cn_bitmaps.extend(pack_cn_bitmap(img))

    ascii_bitmaps = bytearray()
    for cp in range(ASCII_START, ASCII_END + 1):
        ch = chr(cp)
        img = render_glyph(ch, ascii_font, 8, CELL_H)
        ascii_bitmaps.extend(pack_ascii_bitmap(img))

    header = bytearray(16)
    header[:8] = MAGIC
    struct.pack_into("<H", header, 8, len(codepoints))
    struct.pack_into("<H", header, 10, ASCII_START)
    struct.pack_into("<H", header, 12, ASCII_END)
    struct.pack_into("<H", header, 14, 0)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as fh:
        fh.write(header)
        fh.write(struct.pack(f"<{len(codepoints)}H", *codepoints))
        fh.write(cn_bitmaps)
        fh.write(ascii_bitmaps)

    print(f"Built {output_path}")
    print(f"  font: {font_path}")
    print(f"  ascii_font: {ascii_font_path}")
    print(f"  preserve_existing_cjk: {args.preserve_existing_cjk}")
    print(f"  codepoints: {len(codepoints)}")
    print(f"  size: {output_path.stat().st_size} bytes")


if __name__ == "__main__":
    main()
