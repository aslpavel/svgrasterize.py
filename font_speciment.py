#!/usr/bin/env python
from svgrasterize import Transform, Path, FontsDB, DEFAULT_FONTS, Glyph
from typing import Any, Dict
import unicodedata
import sys
import os


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

    row = -1
    cols = 32
    for cname, category in sorted(categories.items()):
        if cname in {"Cc", "Zs", "Cf", "Zl", "Zp"}:
            continue

        # category name
        row += 1
        x, y = 2.0, row * size + size / 2.0
        (cname_path, offset) = label_font.str_to_path(size / 1.5, cname + " ")
        subpaths.extend(cname_path.transform(Transform().translate(x, y + size * 0.2)).subpaths)
        line = Path.from_svg("M{},{} h{}Z".format(x + offset, y, cols * size - offset - 4))
        subpaths.extend(line.stroke(2).subpaths)

        # category
        for index, (_name, glyph) in enumerate(sorted(category.items())):
            col = index % cols
            if col == 0:
                row += 1
            offset = Transform().translate(col * size, row * size)
            if glyph.advance > font.units_per_em:
                offset = offset.scale(font.units_per_em / glyph.advance)
            path = glyph.path.transform(offset @ tr)
            subpaths.extend(path.subpaths)

    return Path(subpaths)


def main():
    if len(sys.argv) > 3 or len(sys.argv) < 2:
        sys.stderr.write("Usage: {} <font.svg> [<font.path>]\n".format(sys.argv[0]))
        sys.exit(1)
    font_filename = sys.argv[1]
    path_filename = None if len(sys.argv) < 3 else sys.argv[2]

    db = FontsDB()
    if os.path.isfile(font_filename):
        db.register_file(font_filename)
        db.resolve("")
        font = db.fonts.popitem()[1][0]
    else:
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
