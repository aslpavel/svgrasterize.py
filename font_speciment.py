#!/usr/bin/env python
import svgrasterize as s
import sys
import os


def main():
    if 3 < len(sys.argv) < 2:
        sys.stderr.write("Usage: {} <font.svg> [<font.path>]\n".format(sys.argv[0]))
        sys.exit(1)
    font_filename = sys.argv[1]
    path_filename = None if len(sys.argv) < 3 else sys.argv[2]

    db = s.FontsDB()
    if os.path.isfile(font_filename):
        db.register_file(font_filename)
        db.resolve("")
        font = db.fonts.popitem()[1][0]
    else:
        db.register_file(s.DEFAULT_FONTS)
        font = db.resolve(font_filename)
    if font is None:
        sys.stderr.write("[error] no such font: {}\n".format(font_filename))
        sys.exit(1)

    tr = s.Transform().matrix(0, 1, 0, 1, 0, 0)
    path = font.speciment()
    if path_filename is not None:
        with open(path_filename, "w") as file:
            file.write(path.to_svg())
    mask = path.mask(tr)[0]
    mask.image[...] = 1.0 - mask.image
    mask.show()


if __name__ == "__main__":
    main()
