#!/usr/bin/env python
from svgrasterize import Transform, Path, FontsDB, DEFAULT_FONTS, Glyph
import subprocess
from typing import Any, Dict
import unicodedata
import sys
import os

TTF_2_SVG = os.path.dirname(__file__) + "/ttf2svg"

def speciment(font, size: float = 32.0) -> Path:
    """Create speciment path that contains all symbols in the font
    """
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
    cols = 32
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

    return Path(subpaths)


def convert_to_svg(filename):
    filename_out, ext = os.path.splitext(os.path.basename(filename))
    if ext == ".svg":
        return filename
    filename_out = f"/tmp/{filename_out}.svg"
    subprocess.run([TTF_2_SVG, filename, filename_out])
    return filename_out


def main():
    if len(sys.argv) > 3 or len(sys.argv) < 2:
        sys.stderr.write("Usage: {} <font.{{svg|ttf}}> [<font.path>]\n".format(sys.argv[0]))
        sys.exit(1)
    font_filename = convert_to_svg(sys.argv[1])
    path_filename = None if len(sys.argv) < 3 else sys.argv[2]

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
    path = speciment(font)
    if path_filename:
        with open(path_filename, "w") as file:
            file.write(path.to_svg())
    else:
        mask = path.mask(tr)[0]
        mask.image[...] = 1.0 - mask.image
        mask.show()


if __name__ == "__main__":
    main()
