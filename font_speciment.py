#!/usr/bin/env python
from svgrasterize import Transform, Path, FontsDB, DEFAULT_FONTS, Glyph, Layer
from typing import Any, Dict
import argparse
import json
import os
import pathlib
import subprocess
import sys
import unicodedata

TTF_2_SVG = pathlib.Path(__file__).expanduser().resolve().parent / "ttf2svg"

SVG = """\
<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <path fill="#ffffff" d="M0,0 H{width} V{height} H-{width}Z" />
  <path fill="#000000" d="{path}" />
</svg>
"""

FORMAT_SVG = "svg"
FORMAT_PATH = "path"
FORMAT_JSON = "json"
FORMAT_PNG = "png"
FORMAT_ALL = [FORMAT_SVG, FORMAT_PATH, FORMAT_JSON, FORMAT_PNG]

DEFAULT_COLS = 42
DEFAULT_SIZE = 32.0


def speciment(font, size: float = DEFAULT_SIZE, cols: int = DEFAULT_COLS) -> (Path, (int, int)):
    """Create speciment path that contains all symbols in the font"""
    if os.path.isfile(DEFAULT_FONTS):
        db = FontsDB()
        db.register_file(DEFAULT_FONTS)
        label_font = db.resolve("sans")
    else:
        label_font = font

    categories: Dict[str, Dict[str, Glyph]] = {}
    for name, glyph in font.glyphs.items():
        try:
            cname = unicodedata.category(name)
        except TypeError:
            cname = "Other"
        category = categories.get(cname)
        if category is None:
            categories[cname] = {name: glyph}
        else:
            category[name] = glyph

    scale = (size - 4) / font.units_per_em
    tr = (
        Transform()
        .translate(2, 2)
        .scale(scale, -scale)
        .translate(0, -font.units_per_em - font.descent)
    )
    subpaths: Any = []

    row = 0
    # render label
    label_path, label_width = label_font.str_to_path(
        size / 1.5,
        "{} {}".format(font.family, size),
    )
    label_tr = Transform().translate((cols * size - label_width) / 2.0, size)
    subpaths.extend(label_path.transform(label_tr).subpaths)
    # render categories
    for cname, category in sorted(categories.items()):
        if cname in {"Cc", "Zs", "Cf", "Zl", "Zp"}:
            continue

        # category name
        row += 1
        x, y = 2.0, row * size + size / 2.0
        (cname_path, offset) = label_font.str_to_path(size / 1.5, cname + " ")
        subpaths.extend(cname_path.transform(Transform().translate(x, y + size * 0.2)).subpaths)
        line = Path.from_svg("M{},{} h{}Z".format(x + offset, y, cols * size - offset - size / 3.0))
        subpaths.extend(line.stroke(2).subpaths)

        # category
        index = 0
        for _name, glyph in sorted(category.items()):
            col = index % cols
            if col == 0:
                row += 1
            offset = Transform().translate(col * size, row * size)
            if glyph.advance > font.units_per_em:
                offset = offset.scale(font.units_per_em / glyph.advance)
            path = glyph.path.transform(offset @ tr)
            if path.subpaths:
                subpaths.extend(path.subpaths)
                index += 1

    return Path(subpaths), (cols * size, (row + 1) * size)


def convert_to_svg(filename):
    filename_out, ext = os.path.splitext(os.path.basename(filename))
    if ext == ".svg":
        return filename
    filename_out = f"/tmp/{filename_out}.svg"
    subprocess.run([str(TTF_2_SVG), filename, filename_out])
    return filename_out


def main():
    parser = argparse.ArgumentParser(description="Generate font speciment")
    parser.add_argument("font", help="SVG|TTF font")
    parser.add_argument("output", nargs="?", help="output file, render to terminal if not provided")
    parser.add_argument("--format", "-f", choices=FORMAT_ALL, help="output format")
    parser.add_argument("--size", "-s", help="font size", default=DEFAULT_SIZE, type=float)
    parser.add_argument("--cols", help="number of columns", default=DEFAULT_COLS, type=int)
    args = parser.parse_args()

    font_filename = convert_to_svg(args.font)
    db = FontsDB()
    if os.path.isfile(font_filename):
        db.register_file(font_filename)
        db.resolve("")
        font = db.fonts.popitem()[1][0]
    else:
        sys.stderr.write("[info] no such file trying to find font with this name\n")
        db.register_file(DEFAULT_FONTS)
        font = db.resolve(font_filename)
    if font is None:
        sys.stderr.write(
            "[error] no such font or file does not contain fonts: {}\n".format(font_filename)
        )
        sys.exit(1)

    tr = Transform().matrix(0, 1, 0, 1, 0, 0)
    path, (width, height) = speciment(font, args.size, args.cols)

    if args.output is None:
        mask = path.mask(tr)[0]
        mask.image[...] = 1.0 - mask.image
        mask.show()
        return

    format = args.format
    if format is None:
        _, ext = os.path.splitext(args.output)
        format = ext[1:].lower()

    if format == FORMAT_JSON:
        with open(args.output, "w") as file:
            json.dump(font.names(), file)
    elif format == FORMAT_PATH:
        with open(args.output, "w") as file:
            file.write(path.to_svg())
    elif format == FORMAT_PNG:
        mask = path.mask(tr)[0]
        image = [1.0, 1.0, 1.0, 1.0] - mask.image * [1.0, 1.0, 1.0, 0.0]
        layer = Layer(image, (0, 0), False, True)
        with open(args.output, "wb") as file:
            layer.write_png(file)
    elif format == FORMAT_SVG:
        with open(args.output, "w") as file:
            file.write(SVG.format(width=int(width), height=int(height), path=path.to_svg()))
    else:
        sys.stderr.write(f"unsupported format: {format}\n")


if __name__ == "__main__":
    main()
