#!/usr/bin/env python3
import argparse
import xml.etree.ElementTree as etree
from svgrasterize import Path, svg_transform


def main():
    parser = argparse.ArgumentParser("simple utility to apply transformation to SVG font")
    parser.add_argument("transform", help="SVG transformation to be appliied")
    parser.add_argument("font", help="SVG font")
    parser.add_argument("output", help="transormed SVG font")
    args = parser.parse_args()

    tr = svg_transform(args.transform)

    etree.register_namespace("", "http://www.w3.org/2000/svg")
    font = etree.parse(args.font)
    root = font.getroot()
    for glyph in root.findall(
        "svg:defs/svg:font/svg:glyph",
        dict(svg="http://www.w3.org/2000/svg"),
    ):
        d = glyph.attrib.get("d")
        if d is None:
            continue
        glyph.attrib["d"] = Path.from_svg(d).transform(tr).to_svg()
    font.write(args.output, xml_declaration=True)


if __name__ == "__main__":
    main()
