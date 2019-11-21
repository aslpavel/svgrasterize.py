#!/usr/bin/env python3
import os
import sys
import argparse
import xml.etree.ElementTree as etree

DEFAULT_SIZE = 48
DEFAULT_MARGIN = 10
SVG_NAMESPACE = "http://www.w3.org/2000/svg"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="path to directory with source svg files")
    parser.add_argument("output", help="output svg sprite file")
    parser.add_argument(
        "-s", "--size", default=DEFAULT_SIZE, type=int, help="size of a tile"
    )
    parser.add_argument(
        "-m",
        "--margin",
        default=DEFAULT_MARGIN,
        type=int,
        help="default margin size between tiles",
    )
    parser.add_argument(
        "-c",
        "--columns",
        type=int,
        help="number of columns in a sprite",
    )
    opts = parser.parse_args()

    if not os.path.isdir(opts.input):
        sys.stderr.write(f"[error] input argument must be a directory: {opts.input}\n")
        sys.exit(1)

    etree.register_namespace("", SVG_NAMESPACE)
    items = {}
    for file in os.listdir(opts.input):
        path = os.path.join(opts.input, file)
        if not file.endswith(".svg") or not os.path.isfile(path):
            continue
        name, _ = os.path.splitext(file)
        item = etree.parse(path).getroot()
        item.attrib.setdefault("id", name)
        items[name] = item

    columns = opts.columns or round(len(items) ** 0.5)
    rows, remainder = divmod(len(items), columns)
    if remainder > 0:
        rows += 1

    root = etree.Element(f"{{{SVG_NAMESPACE}}}svg")
    root.attrib["width"] = str(columns * (opts.size + opts.margin) + opts.margin)
    root.attrib["height"] = str(rows * (opts.size + opts.margin) + opts.margin)

    for index, (name, item) in enumerate(sorted(items.items())):
        row, column = divmod(index, columns)
        item.attrib["width"] = str(opts.size)
        item.attrib["height"] = str(opts.size)
        item.attrib["x"] = str(column * (opts.size + opts.margin) + opts.margin)
        item.attrib["y"] = str(row * (opts.size + opts.margin) + opts.margin)
        root.append(item)

    output = etree.ElementTree(root)
    output.write(opts.output)


if __name__ == "__main__":
    main()
