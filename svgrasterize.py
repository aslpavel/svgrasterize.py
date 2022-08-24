#!/usr/bin/env python
"""Very simple SVG rasterizer

NOT SUPPORTED:
    - markers
    - symbol
    - color-interpolation and filter-color-interpolation attributes

PARTIALLY SUPPORTED:
    - text (textPath is not supported)
    - fonts
        - font resolution logic is very basic
        - style font attribute is not parsed only font-* attrs are supported

KNOWN PROBLEMS:
    - multiple paths over going over the same pixels are breaking antialiasing
      (would draw all pixels with multiplied AA coverage (clamped)).
"""
from __future__ import annotations
import builtins
from email.generator import Generator
import gzip
import io
import math
import numpy as np
import numpy.typing as npt
import os
import re
import struct
import sys
import textwrap
import time
import warnings
import xml.etree.ElementTree as etree
import zlib
from functools import reduce, partial
from typing import Any, Callable, NamedTuple, List, Tuple, Optional, Dict

EPSILON = sys.float_info.epsilon
FLOAT_RE = re.compile(r"[-+]?(?:(?:\d*\.\d+)|(?:\d+\.?))(?:[Ee][+-]?\d+)?")
FLOAT = np.float64

# ------------------------------------------------------------------------------
# Layer
# ------------------------------------------------------------------------------
COMPOSE_OVER = 0
COMPOSE_OUT = 1
COMPOSE_IN = 2
COMPOSE_ATOP = 3
COMPOSE_XOR = 4
COMPOSE_PRE_ALPHA = {COMPOSE_OVER, COMPOSE_OUT, COMPOSE_IN, COMPOSE_ATOP, COMPOSE_XOR}


BBox = Tuple[float, float, float, float]
FNDArray = npt.NDArray[FLOAT]
Image = npt.NDArray[FLOAT]


class Layer(NamedTuple):
    image: Image
    offset: Tuple[int, int]
    pre_alpha: bool
    linear_rgb: bool

    @property
    def x(self) -> int:
        return self.offset[0]

    @property
    def y(self) -> int:
        return self.offset[1]

    @property
    def width(self) -> int:
        return self.image.shape[1]

    @property
    def height(self) -> int:
        return self.image.shape[0]

    @property
    def channels(self) -> int:
        return self.image.shape[2]

    @property
    def bbox(self) -> BBox:
        return (*self.offset, *self.image.shape[:2])

    def translate(self, x: int, y: int) -> Layer:
        offset = (self.x + x, self.y + y)
        return Layer(self.image, offset, self.pre_alpha, self.linear_rgb)

    def color_matrix(self, matrix: np.ndarray) -> Layer:
        """Apply color matrix transformation"""
        if not isinstance(matrix, np.ndarray) or matrix.shape != (4, 5):
            raise ValueError("expected 4x5 matrix")
        layer = self.convert(pre_alpha=False, linear_rgb=True)
        M = matrix[:, :4]
        B = matrix[:, 4]
        image = np.matmul(layer.image, M.T) + B
        np.clip(image, 0, 1, out=image)
        return Layer(image, layer.offset, pre_alpha=False, linear_rgb=True)

    def convolve(self, kernel: np.ndarray) -> Layer:
        """Convolve layer"""
        try:
            from scipy.signal import convolve

            layer = self.convert(pre_alpha=False, linear_rgb=True)
            kw, kh = kernel.shape
            image = convolve(layer.image, kernel[..., None])
            x, y = int(layer.x - kw / 2), int(layer.y - kh / 2)
            return Layer(image, (x, y), pre_alpha=False, linear_rgb=True)
        except ImportError:
            warnings.warn("Layer::convolve requires `scipy`")
            return self

    def morphology(self, x: int, y: int, method: str) -> Layer:
        """Morphology filter operation

        Morphology is essentially {min|max} pooling with [1, 1] stride
        """
        layer = self.convert(pre_alpha=True, linear_rgb=True)
        image = pooling(layer.image, ksize=(x, y), stride=(1, 1), method=method)
        return Layer(image, layer.offset, pre_alpha=True, linear_rgb=True)

    def convert(self, pre_alpha: Optional[bool] = None, linear_rgb: Optional[bool] = None) -> Layer:
        """Convert image if needed to specified alpha and colorspace"""
        pre_alpha = self.pre_alpha if pre_alpha is None else pre_alpha
        linear_rgb = self.linear_rgb if linear_rgb is None else linear_rgb

        if self.channels == 1:
            # single channel value assumed to be alpha
            return Layer(self.image, self.offset, pre_alpha, linear_rgb)

        in_image, out_offset, out_pre_alpha, out_linear_rgb = self
        out_image = None

        if out_linear_rgb != linear_rgb:
            out_image = in_image.copy()
            # convert to straight alpha first if needed
            if out_pre_alpha:
                out_image = color_pre_to_straight_alpha(out_image)
                out_pre_alpha = False
            if linear_rgb:
                out_image = color_srgb_to_linear(out_image)
            else:
                out_image = color_linear_to_srgb(out_image)
            out_linear_rgb = linear_rgb

        if out_pre_alpha != pre_alpha:
            if out_image is None:
                out_image = in_image.copy()
            if pre_alpha:
                out_image = color_straight_to_pre_alpha(out_image)
            else:
                out_image = color_pre_to_straight_alpha(out_image)
            out_pre_alpha = pre_alpha

        if out_image is None:
            return self
        return Layer(out_image, out_offset, out_pre_alpha, out_linear_rgb)

    def background(self, color: np.ndarray) -> Layer:
        layer = self.convert(pre_alpha=True, linear_rgb=True)
        image = canvas_compose(COMPOSE_OVER, color[None, None, ...], layer.image)
        return Layer(image, layer.offset, pre_alpha=True, linear_rgb=True)

    def opacity(self, opacity: float, linear_rgb=False) -> Layer:
        """Apply additional opacity"""
        layer = self.convert(pre_alpha=True, linear_rgb=linear_rgb)
        image = layer.image * opacity
        return Layer(image, layer.offset, pre_alpha=True, linear_rgb=linear_rgb)

    @staticmethod
    def compose(layers: List[Layer], method=COMPOSE_OVER, linear_rgb=False) -> Optional[Layer]:
        """Compose multiple layers into one with specified `method`

        Composition in linear RGB is correct one but SVG composes in sRGB
        by default. Only filter is composing in linear RGB by default.
        """
        if not layers:
            return None
        elif len(layers) == 1:
            return layers[0]
        images: List[Tuple[Image, Tuple[int, int]]] = []
        pre_alpha = method in COMPOSE_PRE_ALPHA
        for layer in layers:
            layer = layer.convert(pre_alpha=pre_alpha, linear_rgb=linear_rgb)
            images.append((layer.image, layer.offset))
        blend = partial(canvas_compose, method)
        if method == COMPOSE_IN:
            result = canvas_merge_intersect(images, blend)
        elif method == COMPOSE_OVER:
            result = canvas_merge_union(images, full=False, blend=blend)
        else:
            result = canvas_merge_union(images, full=True, blend=blend)
        if result is None:
            return None
        image, offset = result
        return Layer(image, offset, pre_alpha=pre_alpha, linear_rgb=linear_rgb)

    def write_png(self, output=None):
        if self.channels != 4:
            raise ValueError("Only RGBA layers are supported")
        layer = self.convert(pre_alpha=False, linear_rgb=False)
        return canvas_to_png(layer.image, output)

    def __repr__(self):
        return "Layer(x={}, y={}, w={}, h={}, pre_alpha={}, linear_rgb={})".format(
            self.x, self.y, self.width, self.height, self.pre_alpha, self.linear_rgb
        )

    def show(self, format=None):
        """Show layer on terminal if `imshow` if available

        NOTE: used only for debugging
        """
        try:
            from imshow import show

            layer = self.convert(pre_alpha=False, linear_rgb=False)
            show(layer.image, format=format)
            print()
        except ImportError:
            warnings.warn("to be able to show layer on terminal imshow is required")


def canvas_create(width, height, bg=None):
    """Create canvas of a specified size

    Returns (canvas, transform) tuple:
       canvas - float64 ndarray of (height, width, 4) shape
       transform - transform from (x, y) to canvas pixel coordinates
    """
    if bg is None:
        canvas = np.zeros((height, width, 4), dtype=FLOAT)
    else:
        canvas = np.broadcast_to(bg, (height, width, 4)).copy()
    return canvas, Transform().matrix(0, 1, 0, 1, 0, 0)


def canvas_to_png(canvas: Image, output: Optional[io.IOBase] = None) -> io.IOBase:
    """Convert (height, width, rgba{float64}) to PNG"""

    def png_pack(output: io.IOBase, tag: bytes, data: bytes):
        checksum = 0xFFFFFFFF & zlib.crc32(data, zlib.crc32(tag))
        output.write(struct.pack("!I", len(data)))
        output.write(tag)
        output.write(data)
        output.write(struct.pack("!I", checksum))

    height, width, _ = canvas.shape

    data = io.BytesIO()
    comp = zlib.compressobj(level=9)
    for row in np.round(canvas * 255.0).astype(np.uint8):
        data.write(comp.compress(b"\x00"))
        data.write(comp.compress(row.tobytes()))
    data.write(comp.flush())

    output = io.BytesIO() if output is None else output
    output.write(b"\x89PNG\r\n\x1a\n")
    png_pack(output, b"IHDR", struct.pack("!2I5B", width, height, 8, 6, 0, 0, 0)),
    png_pack(output, b"IDAT", data.getvalue()),
    png_pack(output, b"IEND", b"")

    return output


def canvas_compose(mode: int, dst: Image, src: Image) -> Image:
    """Compose two alpha premultiplied images

    https://ciechanow.ski/alpha-compositing/
    http://ssp.impulsetrain.com/porterduff.html
    """
    src_a = src[..., -1:] if len(src.shape) == 3 else src
    dst_a = dst[..., -1:] if len(dst.shape) == 3 else dst
    if mode == COMPOSE_OVER:
        return src + dst * (1 - src_a)
    elif mode == COMPOSE_OUT:
        return src * (1 - dst_a)
    elif mode == COMPOSE_IN:
        return src * dst_a
    elif mode == COMPOSE_ATOP:
        return src * dst_a + dst * (1 - src_a)
    elif mode == COMPOSE_XOR:
        return src * (1 - dst_a) + dst * (1 - src_a)
    elif isinstance(mode, tuple) and len(mode) == 4:
        k1, k2, k3, k4 = mode
        return (k1 * src * dst + k2 * src + k3 * dst + k4).clip(0, 1)
    raise ValueError(f"invalid compose mode: {mode}")


CANVAS_COMPOSE_OVER: Callable[[Image, Image], Image] = partial(canvas_compose, COMPOSE_OVER)


def canvas_merge_at(base, overlay, offset, blend=CANVAS_COMPOSE_OVER):
    """Alpha blend `overlay` on top of `base` at offset coordinate

    Updates `base` with `overlay` in place.
    """
    x, y = offset
    b_h, b_w = base.shape[:2]
    o_h, o_w = overlay.shape[:2]
    clip = lambda v, l, h: l if v < l else h if v > h else v

    b_x_low, b_x_high = clip(x, 0, b_h), clip(x + o_h, 0, b_h)
    b_y_low, b_y_high = clip(y, 0, b_w), clip(y + o_w, 0, b_w)
    effected = base[b_x_low:b_x_high, b_y_low:b_y_high]
    if effected.size == 0:
        return

    o_x_low, o_x_high = clip(-x, 0, o_h), clip(b_h - x, 0, o_h)
    o_y_low, o_y_high = clip(-y, 0, o_w), clip(b_w - y, 0, o_w)
    overlay = overlay[o_x_low:o_x_high, o_y_low:o_y_high]
    if overlay.size == 0:
        return

    effected[...] = blend(effected, overlay).clip(0, 1)
    return base


def canvas_merge_union(
    layers: List[Tuple[FNDArray, Tuple[int, int]]],
    full=True,
    blend: Callable[[FNDArray, FNDArray], FNDArray] = CANVAS_COMPOSE_OVER,
) -> Tuple[FNDArray, Tuple[int, int]]:
    """Blend multiple `layers` into single large enough image"""
    if not layers:
        raise ValueError("can not blend zero layers")
    elif len(layers) == 1:
        return layers[0]

    min_x, min_y, max_x, max_y = None, None, None, None
    for image, offset in layers:
        x, y = offset
        w, h = image.shape[:2]
        if min_x is None:
            min_x, min_y = x, y
            max_x, max_y = x + w, y + h
        else:
            min_x, min_y = min(min_x, x), min(min_y, y)
            max_x, max_y = max(max_x, x + w), max(max_y, y + h)
    width, height = max_x - min_x, max_y - min_y

    if full:
        output = None
        for image, offset in layers:
            x, y = offset
            w, h = image.shape[:2]
            ox, oy = x - min_x, y - min_y
            image_full = np.zeros((width, height, 4), dtype=FLOAT)
            image_full[ox : ox + w, oy : oy + h] = image
            if output is None:
                output = image_full
            else:
                output = blend(output, image_full)

    else:
        # this is optimization for method `over` blending
        output = np.zeros((max_x - min_x, max_y - min_y, 4), dtype=FLOAT)
        for index, (image, offset) in enumerate(layers):
            x, y = offset
            w, h = image.shape[:2]
            ox, oy = x - min_x, y - min_y
            effected = output[ox : ox + w, oy : oy + h]
            if index == 0:
                effected[...] = image
            else:
                effected[...] = blend(effected, image)

    return output, (min_x, min_y)


def canvas_merge_intersect(
    layers: List[Tuple[Image, Tuple[int, int]]],
    blend: Callable[[Image, Image], Image] = CANVAS_COMPOSE_OVER,
) -> Optional[Tuple[Image, Tuple[int, int]]]:
    """Blend multiple `layers` into single image covered by all layers"""
    if not layers:
        raise ValueError("can not blend zero layers")
    elif len(layers) == 1:
        return layers[0]

    min_x, min_y, max_x, max_y = None, None, None, None
    for layer, offset in layers:
        x, y = offset
        w, h = layer.shape[:2]
        if min_x is None and max_y is None:
            min_x, min_y = x, y
            max_x, max_y = x + w, y + h
        else:
            min_x, min_y = max(min_x, x), max(min_y, y)
            max_x, max_y = min(max_x, x + w), min(max_y, y + h)

    if min_x >= max_x or min_y >= max_y:
        return None  # empty intersection

    (first, (fx, fy)), *rest = layers
    output = first[min_x - fx : max_x - fx, min_y - fy : max_y - fy]
    w, h, c = output.shape
    if c == 1:
        output = np.broadcast_to(output, (w, h, 4))
    output = output.copy()
    for layer, offset in rest:
        x, y = offset
        output[...] = blend(output, layer[min_x - x : max_x - x, min_y - y : max_y - y])

    return output, (min_x, min_y)


def pooling(mat, ksize, stride=None, method="max", pad=False):
    """Overlapping pooling on 2D or 3D data.

    <mat>: ndarray, input array to pool.
    <ksize>: tuple of 2, kernel size in (ky, kx).
    <stride>: tuple of 2 or None, stride of pooling window.
              If None, same as <ksize> (non-overlapping pooling).
    <method>: str, 'max for max-pooling,
                   'mean' for mean-pooling.
    <pad>: bool, pad <mat> or not. If no pad, output has size
           (n-f)//s+1, n being <mat> size, f being kernel size, s stride.
           if pad, output has size ceil(n/s).

    Return <result>: pooled matrix.
    """
    m, n = mat.shape[:2]
    ky, kx = ksize
    if stride is None:
        stride = (ky, kx)
    sy, sx = stride

    if pad:
        nx = int(np.ceil(n / float(sx)))
        ny = int(np.ceil(m / float(sy)))
        size = ((ny - 1) * sy + ky, (nx - 1) * sx + kx) + mat.shape[2:]
        mat_pad = np.full(size, np.nan)
        mat_pad[:m, :n, ...] = mat
    else:
        mat_pad = mat[: (m - ky) // sy * sy + ky, : (n - kx) // sx * sx + kx, ...]

    # Get a strided sub-matrices view of an ndarray.
    s0, s1 = mat_pad.strides[:2]
    m1, n1 = mat_pad.shape[:2]
    m2, n2 = ksize
    view_shape = (1 + (m1 - m2) // stride[0], 1 + (n1 - n2) // stride[1], m2, n2) + mat_pad.shape[
        2:
    ]
    strides = (stride[0] * s0, stride[1] * s1, s0, s1) + mat_pad.strides[2:]
    view = np.lib.stride_tricks.as_strided(mat_pad, view_shape, strides=strides)

    if method == "max":
        result = np.nanmax(view, axis=(2, 3))
    elif method == "min":
        result = np.nanmin(view, axis=(2, 3))
    elif method == "mean":
        result = np.nanmean(view, axis=(2, 3))
    else:
        raise ValueError(f"invalid poll method: {method}")

    return result


def color_pre_to_straight_alpha(rgba: Image) -> Image:
    """Convert from premultiplied alpha in place"""
    rgb = rgba[..., :-1]
    alpha = rgba[..., -1:]
    np.divide(rgb, alpha, out=rgb, where=alpha > 0.0001)
    np.clip(rgba, 0, 1, out=rgba)
    return rgba


def color_straight_to_pre_alpha(rgba: Image) -> Image:
    """Convert to premultiplied alpha in place"""
    rgba[..., :-1] *= rgba[..., -1:]
    return rgba


def color_linear_to_srgb(rgba: Image) -> Image:
    """Convert pixels from linear RGB to sRGB in place"""
    rgb = rgba[..., :-1]
    small = rgb <= 0.0031308
    rgb[small] = rgb[small] * 12.92
    large = ~small
    rgb[large] = 1.055 * np.power(rgb[large], 1.0 / 2.4) - 0.055
    return rgba


def color_srgb_to_linear(rgba: Image) -> Image:
    """Convert pixels from sRGB to linear RGB in place"""
    rgb = rgba[..., :-1]
    small = rgb <= 0.04045
    rgb[small] = rgb[small] / 12.92
    large = ~small
    rgb[large] = np.power((rgb[large] + 0.055) / 1.055, 2.4)
    return rgba


# ------------------------------------------------------------------------------
# Transform
# ------------------------------------------------------------------------------
class Transform:
    __slots__: List[str] = ["m", "_m_inv"]
    m: FNDArray
    _m_inv: Optional[FNDArray]

    def __init__(self, matrix: Optional[FNDArray] = None, matrix_inv: Optional[FNDArray] = None):
        if matrix is None:
            self.m = np.identity(3)
            self._m_inv = self.m
        else:
            self.m = matrix
            self._m_inv = matrix_inv

    def __matmul__(self, other: Transform) -> Transform:
        return Transform(self.m @ other.m)

    @property
    def invert(self) -> Transform:
        if self._m_inv is None:
            self._m_inv = np.linalg.inv(self.m)
        return Transform(self._m_inv, self.m)

    def __call__(self, points: FNDArray) -> FNDArray:
        if len(points) == 0:
            return points
        return points @ self.m[:2, :2].T + self.m[:2, 2]

    def apply(self) -> Callable[[FNDArray], FNDArray]:
        M = self.m[:2, :2].T
        B = self.m[:2, 2]
        return lambda points: points @ M + B

    def matrix(
        self, m00: float, m01: float, m02: float, m10: float, m11: float, m12: float
    ) -> Transform:
        return Transform(self.m @ np.array([[m00, m01, m02], [m10, m11, m12], [0, 0, 1]]))

    def translate(self, tx: float, ty: float) -> Transform:
        return Transform(self.m @ np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]]))

    def scale(self, sx: float, sy: Optional[float] = None) -> Transform:
        sy = sx if sy is None else sy
        return Transform(self.m @ np.array([[sx, 0, 0], [0, sy, 0], [0, 0, 1]]))

    def rotate(self, angle: float) -> Transform:
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        return Transform(self.m @ np.array([[cos_a, -sin_a, 0], [sin_a, cos_a, 0], [0, 0, 1]]))

    def skew(self, ax: float, ay: float) -> Transform:
        return Transform(
            np.matmul(self.m, np.array([[1, math.tan(ax), 0], [math.tan(ay), 1, 0], [0, 0, 1]]))
        )

    def __repr__(self) -> str:
        return str(np.around(self.m, 4).tolist()[:2])

    def no_translate(self) -> Transform:
        m = self.m.copy()
        m[0, 2] = 0
        m[1, 2] = 0
        return Transform(m)


# ------------------------------------------------------------------------------
# Render scene
# ------------------------------------------------------------------------------
RENDER_FILL = 0
RENDER_STROKE = 1
RENDER_GROUP = 2
RENDER_OPACITY = 3
RENDER_CLIP = 4
RENDER_TRANSFORM = 5
RENDER_FILTER = 6
RENDER_MASK = 7


class Scene(tuple):
    __slots__: List[str] = []

    def __new__(cls, type, args):
        return tuple.__new__(cls, (type, args))

    @classmethod
    def fill(cls, path: Path, paint, fill_rule: Optional[str] = None) -> Scene:
        return cls(RENDER_FILL, (path, paint, fill_rule))

    @classmethod
    def stroke(
        cls,
        path: Path,
        paint,
        width: float,
        linecap: Optional[str] = None,
        linejoin: Optional[str] = None,
    ) -> Scene:
        return cls(RENDER_STROKE, (path, paint, width, linecap, linejoin))

    @classmethod
    def group(cls, children: List[Scene]) -> Scene:
        if not children:
            raise ValueError("group have to contain at least one child")
        if len(children) == 1:
            return children[0]
        return cls(RENDER_GROUP, children)

    def opacity(self, opacity: float) -> Scene:
        if opacity > 0.999:
            return self
        return Scene(RENDER_OPACITY, (self, opacity))

    def clip(self, clip: Path, bbox_units: bool = False) -> Scene:
        return Scene(RENDER_CLIP, (self, clip, bbox_units))

    def mask(self, mask: Path, bbox_units: bool = False) -> Scene:
        return Scene(RENDER_MASK, (self, mask, bbox_units))

    def transform(self, transform: Transform) -> Scene:
        type, args = self
        if type == RENDER_TRANSFORM:
            target, target_transform = args
            return Scene(RENDER_TRANSFORM, (target, transform @ target_transform))
        else:
            return Scene(RENDER_TRANSFORM, (self, transform))

    def filter(self, filter: Filter) -> Scene:
        return Scene(RENDER_FILTER, (self, filter))

    def render(
        self, transform: Transform, mask_only=False, viewport=None, linear_rgb=False
    ) -> Optional[Tuple[Layer, ConvexHull]]:
        """Render graph"""
        type, args = self
        if type == RENDER_FILL:
            path, paint, fill_rule = args
            if mask_only:
                return path.mask(transform, fill_rule=fill_rule, viewport=viewport)
            else:
                return path.fill(
                    transform, paint, fill_rule=fill_rule, viewport=viewport, linear_rgb=linear_rgb
                )

        elif type == RENDER_STROKE:
            path, paint, width, linecap, linejoin = args
            stroke = path.stroke(width, linecap, linejoin)
            if mask_only:
                return stroke.mask(transform, viewport=viewport)
            else:
                return stroke.fill(transform, paint, viewport=viewport, linear_rgb=linear_rgb)

        elif type == RENDER_GROUP:
            layers, hulls = [], []
            for child in args:
                layer = child.render(transform, mask_only, viewport, linear_rgb)
                if layer is None:
                    continue
                layer, hull = layer
                layers.append(layer)
                hulls.append(hull)

            group = Layer.compose(layers, COMPOSE_OVER, linear_rgb)
            if not group:
                return None
            return group, ConvexHull.merge(hulls)

        elif type == RENDER_OPACITY:
            target, opacity = args
            layer = target.render(transform, mask_only, viewport, linear_rgb)
            if layer is None:
                return None
            layer, hull = layer
            return layer.opacity(opacity, linear_rgb), hull

        elif type == RENDER_CLIP:
            target, clip, bbox_units = args
            image_result = target.render(transform, mask_only, viewport, linear_rgb)
            if image_result is None:
                return None
            image, hull = image_result

            if bbox_units:
                transform = hull.bbox_transform(transform)
            clip_result = clip.render(transform, True, viewport, linear_rgb)
            if clip_result is None:
                return None
            mask, _ = clip_result

            result = Layer.compose([mask, image], COMPOSE_IN, linear_rgb)
            if result is None:
                return None
            return result, hull

        elif type == RENDER_TRANSFORM:
            target, target_transfrom = args
            return target.render(transform @ target_transfrom, mask_only, viewport, linear_rgb)

        elif type == RENDER_MASK:
            target, mask_scene, bbox_units = args
            image_result = target.render(transform, mask_only, viewport, linear_rgb)
            if image_result is None:
                return None
            image, hull = image_result

            if bbox_units:
                transform = hull.bbox_transform(transform)
            mask_result = mask_scene.render(transform, mask_only, viewport, linear_rgb)
            if mask_result is None:
                return None
            mask, _ = mask_result
            mask = mask.convert(pre_alpha=False, linear_rgb=linear_rgb)
            mask_image = mask.image[..., :3] @ [0.2125, 0.7154, 0.072] * mask.image[..., 3]
            mask = Layer(mask_image[..., None], mask.offset, pre_alpha=False, linear_rgb=linear_rgb)

            result = Layer.compose([mask, image], COMPOSE_IN, linear_rgb)
            if result is None:
                return None
            return result, hull

        elif type == RENDER_FILTER:
            target, filter = args
            image_result = target.render(transform, mask_only, viewport, linear_rgb)
            if image_result is None:
                return None
            image, hull = image_result
            return filter(transform, image), hull

        else:
            raise ValueError(f"unhandled scene type: {type}")

    def to_path(self, transform: Transform) -> Path:
        """Try to convert whole scene to a path (used only for testing)"""

        def to_path(scene: Scene, transform: Transform):
            type, args = scene
            if type == RENDER_FILL:
                path, _paint, _fill_rule = args
                yield path.transform(transform)

            elif type == RENDER_STROKE:
                path, paint, width, linecap, linejoin = args
                yield path.transform(transform).stroke(width, linecap, linejoin)

            elif type == RENDER_GROUP:
                for child in args:
                    yield from to_path(child, transform)

            elif type == RENDER_OPACITY:
                target, _opacity = args
                yield from to_path(target, transform)

            elif type == RENDER_CLIP:
                target, _clip, _bbox_units = args
                yield from to_path(target, transform)

            elif type == RENDER_TRANSFORM:
                target, target_transfrom = args
                yield from to_path(target, transform @ target_transfrom)

            elif type == RENDER_MASK:
                target, _mask_scene, _bbox_units = args
                yield from to_path(target, transform)

            elif type == RENDER_FILTER:
                target, _filter = args
                yield from to_path(target, transform)

            else:
                raise ValueError(f"unhandled scene type: {type}")

        subpaths = [spath for path in to_path(self, transform) for spath in path.subpaths]
        return Path(subpaths)

    def __repr__(self) -> str:
        def repr_rec(scene, output, depth):
            output.write(indent * depth)
            type, args = scene
            if type == RENDER_FILL:
                path, paint, fill_rule = args
                if isinstance(paint, np.ndarray):
                    paint = format_color(paint)
                output.write(f"FILL fill_rule:{fill_rule} paint:{paint}\n")
                output.write(textwrap.indent(repr(path), indent * (depth + 1)))
                output.write("\n")
            elif type == RENDER_STROKE:
                path, paint, width, linecap, linejoin = args
                if isinstance(paint, np.ndarray):
                    paint = format_color(paint)
                output.write(f"STROKE ")
                output.write(f"width:{width} ")
                output.write(f"linecap:{linecap} ")
                output.write(f"linejoin:{linejoin} ")
                output.write(f"paint:{paint}\n")
                output.write(textwrap.indent(repr(path), indent * (depth + 1)))
                output.write("\n")
            elif type == RENDER_GROUP:
                output.write("GROUP\n")
                for child in args:
                    repr_rec(child, output, depth + 1)
            elif type == RENDER_OPACITY:
                target, opacity = args
                output.write(f"OPACITY {opacity}\n")
                repr_rec(target, output, depth + 1)
            elif type == RENDER_CLIP:
                target, clip, bbox_units = args
                output.write(f"CLIP bbox_units:{bbox_units}\n")
                output.write(indent * (depth + 1))
                output.write("CLIP_PATH\n")
                repr_rec(clip, output, depth + 2)
                output.write(indent * (depth + 1))
                output.write("CLIP_TARGET\n")
                repr_rec(target, output, depth + 2)
            elif type == RENDER_MASK:
                target, mask, bbox_units = args
                output.write(f"MASK bbox_units:{bbox_units}\n")
                output.write(indent * (depth + 1))
                output.write("MAKS_PATH\n")
                repr_rec(mask, output, depth + 2)
                output.write(indent * (depth + 1))
                output.write("MASK_TARGET\n")
                repr_rec(target, output, depth + 2)
            elif type == RENDER_TRANSFORM:
                target, transform = args
                output.write(f"TRANSFORM {transform}\n")
                repr_rec(target, output, depth + 1)
            elif type == RENDER_FILTER:
                target, filter = args
                output.write(f"FILTER {filter}\n")
                repr_rec(target, output, depth + 1)
            else:
                raise ValueError(f"unhandled scene type: {type}")
            return output

        def format_color(cs):
            return "#" + "".join(f"{c:0<2x}" for c in (cs * 255).astype(np.uint8))

        indent = "  "
        return repr_rec(self, io.StringIO(), 0).getvalue()[:-1]


# ------------------------------------------------------------------------------
# Path
# ------------------------------------------------------------------------------
PATH_LINE = 0
PATH_QUAD = 1
PATH_CUBIC = 2
PATH_ARC = 3
PATH_CLOSED = 4
PATH_UNCLOSED = 5
PATH_LINES = {PATH_LINE, PATH_CLOSED, PATH_UNCLOSED}
PATH_FILL_NONZERO = "nonzero"
PATH_FILL_EVENODD = "evenodd"
STROKE_JOIN_MITER = "miter"
STROKE_JOIN_ROUND = "round"
STROKE_JOIN_BEVEL = "bevel"
STROKE_CAP_BUTT = "butt"
STROKE_CAP_ROUND = "round"
STROKE_CAP_SQUARE = "square"


class Path:
    """Single rendering unit that can be filled or converted to stroke path

    `subpaths` is a list of tuples:
        - `(PATH_LINE, (p0, p1))` - line from p0 to p1
        - `(PATH_CUBIC, (p0, c0, c1, p1))` - cubic bezier curve from p0 to p1 with control c0, c1
        - `(PATH_QUAD, (p0, c0, p1))` - quadratic bezier curve from p0 to p1 with control c0
        - `(PATH_ARC, (center, rx, ry, phi, eta, eta_delta)` - arc with a center and to radii rx, ry
            rotated to phi angle, going from initial eta to eta + eta_delta angle.
        - `(PATH_CLOSED | PATH_UNCLOSED, (p0, p1))` - last segment of subpath `"closed"` if
           path was closed and `"unclosed"` if path was not closed. p0 - end subpath
           p1 - beginning of this subpath.
    """

    __slots__ = ["subpaths"]
    subpaths: List[List[Tuple[int, Tuple[Any, ...]]]]

    def __init__(self, subpaths):
        self.subpaths = subpaths

    def __iter__(self):
        """Iterate over subpaths"""
        return iter(self.subpaths)

    def __bool__(self) -> bool:
        return bool(self.subpaths)

    def mask(
        self,
        transform: Transform,
        fill_rule: Optional[str] = None,
        viewport: Optional[BBox] = None,
    ) -> Optional[Tuple[Layer, ConvexHull]]:
        """Render path as a mask (alpha channel only image)"""
        # convert all curves to cubic curves and lines
        lines_defs, cubics_defs = [], []
        for path in self.subpaths:
            if not path:
                continue
            for cmd, args in path:
                if cmd in PATH_LINES:
                    lines_defs.append(args)
                elif cmd == PATH_CUBIC:
                    cubics_defs.append(args)
                elif cmd == PATH_QUAD:
                    cubics_defs.append(bezier2_to_bezier3(args))
                elif cmd == PATH_ARC:
                    cubics_defs.extend(arc_to_bezier3(*args))
                else:
                    raise ValueError(f"unsupported path type: `{cmd}`")

        # transform all curves into presentation coordinate system
        lines = transform(np.array(lines_defs, dtype=FLOAT))
        cubics = transform(np.array(cubics_defs, dtype=FLOAT))

        # flattened (converted to lines) all curves
        if cubics.size != 0:
            # flatness of 0.1px gives good accuracy
            if lines.size != 0:
                lines = np.concatenate([lines, bezier3_flatten_batch(cubics, 0.1)])
            else:
                lines = bezier3_flatten_batch(cubics, 0.1)
        if lines.size == 0:
            return

        # calculate size of the mask
        min_x: int
        min_y: int
        max_x: int
        max_y: int
        min_x, min_y = np.floor(lines.reshape(-1, 2).min(axis=0)).astype(int) - 1
        max_x, max_y = np.ceil(lines.reshape(-1, 2).max(axis=0)).astype(int) + 1
        if viewport is not None:
            vx, vy, vw, vh = viewport
            min_x, min_y = max(vx, min_x), max(vy, min_y)
            max_x, max_y = min(vx + vw, max_x), min(vy + vh, max_y)
        width = max_x - min_x
        height = max_y - min_y
        if width <= 0 or height <= 0:
            return

        # create trace (signed coverage)
        trace = np.zeros((width, height), dtype=FLOAT)
        for points in lines - np.array([min_x, min_y]):
            line_signed_coverage(trace, points)

        # render mask
        mask = np.cumsum(trace, axis=1)
        if fill_rule is None or fill_rule == PATH_FILL_NONZERO:
            mask = np.fabs(mask).clip(0, 1)
        elif fill_rule == PATH_FILL_EVENODD:
            mask = np.fabs(np.remainder(mask + 1.0, 2.0) - 1.0)
        else:
            raise ValueError(f"Invalid fill rule: {fill_rule}")
        mask[mask < 1e-6] = 0  # round down to zero very small mask values

        output = Layer(mask[..., None], (min_x, min_y), pre_alpha=True, linear_rgb=True)
        return output, ConvexHull(lines)

    def fill(self, transform: Transform, paint, fill_rule=None, viewport=None, linear_rgb=True):
        """Render path by fill-ing it."""
        if paint is None:
            return None

        # create a mask
        mask = self.mask(transform, fill_rule, viewport)
        if mask is None:
            return None
        mask, hull = mask

        # create background with specified paint
        if isinstance(paint, np.ndarray) and paint.shape == (4,):
            if not linear_rgb:
                paint = color_pre_to_straight_alpha(paint.copy())
                paint = color_linear_to_srgb(paint)
                paint = color_straight_to_pre_alpha(paint)
            output = Layer(mask.image * paint, mask.offset, pre_alpha=True, linear_rgb=linear_rgb)

        elif isinstance(paint, (GradLinear, GradRadial)):
            if paint.bbox_units:
                user_tr = hull.bbox_transform(transform).invert
            else:
                user_tr = transform.invert
            # convert grad pixels to user coordinate system
            pixels = user_tr(grad_pixels(mask.bbox))

            if paint.linear_rgb is not None:
                linear_rgb = paint.linear_rgb
            image = paint.fill(pixels, linear_rgb=linear_rgb)
            # NOTE: consider optimizing calculation of grad only for unmasked points
            # masked = mask.image > EPSILON
            # painted = paint.fill(
            #     pixels[np.broadcast_to(masked, pixels.shape)].reshape(-1, 2),
            #     linear_rgb=linear_rgb,
            # )
            # image = np.zeros((mask.width, mask.height, 4), dtype=FLOAT)
            # image[np.broadcast_to(masked, image.shape)] = painted.reshape(-1)

            background = Layer(image, mask.offset, pre_alpha=True, linear_rgb=linear_rgb)

            # use `canvas_compose` directly to avoid needless allocation
            background = background.convert(pre_alpha=True, linear_rgb=linear_rgb)
            mask = mask.convert(pre_alpha=True, linear_rgb=linear_rgb)
            image = canvas_compose(COMPOSE_IN, mask.image, background.image)
            output = Layer(image, mask.offset, pre_alpha=True, linear_rgb=linear_rgb)

        elif isinstance(paint, Pattern):
            # render pattern
            pat_tr = transform.no_translate()
            if paint.scene_view_box:
                if paint.bbox_units:
                    px, py, pw, ph = paint.bbox()
                    _hx, _hy, hw, hh = hull.bbox(transform)
                    bbox = (px * hw, py * hh, pw * hw, ph * hh)
                else:
                    bbox = paint.bbox()
                pat_tr @= svg_viewbox_transform(bbox, paint.scene_view_box)
            elif paint.scene_bbox_units:
                pat_tr = hull.bbox_transform(pat_tr)
            pat_tr @= paint.transform
            result = paint.scene.render(pat_tr, linear_rgb=linear_rgb)
            if result is None:
                return None
            pat_layer, _pat_hull = result

            # repeat pattern
            repeat_tr = transform
            if paint.bbox_units:
                repeat_tr = hull.bbox_transform(repeat_tr)
            repeat_tr @= paint.transform
            repeat_tr = repeat_tr.no_translate()
            offsets = repeat_tr.invert(grad_pixels(mask.bbox))
            offsets = repeat_tr(
                np.remainder(offsets - [paint.x, paint.y], [paint.width, paint.height])
            )
            offsets = offsets.astype(int)
            corners = repeat_tr(
                [
                    [0, 0],
                    [paint.width, 0],
                    [0, paint.height],
                    [paint.width, paint.height],
                ]
            )
            max_x, max_y = corners.max(axis=0).astype(int)
            min_x, min_y = corners.min(axis=0).astype(int)
            w, h = max_x - min_x, max_y - min_y
            offsets -= [min_x, min_y]

            pat = np.zeros((w + 1, h + 1, 4))
            pat = canvas_merge_at(pat, pat_layer.image, (pat_layer.x - min_x, pat_layer.y - min_y))
            image = canvas_compose(COMPOSE_IN, mask.image, pat[offsets[..., 0], offsets[..., 1]])
            output = Layer(
                image, mask.offset, pre_alpha=pat_layer.pre_alpha, linear_rgb=pat_layer.linear_rgb
            )

        else:
            warnings.warn(f"fill method is not implemented: {paint}")
            return None

        return output, hull

    def stroke(
        self,
        width: float,
        linecap: Optional[str] = None,
        linejoin: Optional[str] = None,
    ) -> Path:
        """Convert path to stroked path"""
        curve_names = {2: PATH_LINE, 3: PATH_QUAD, 4: PATH_CUBIC}

        dist = width / 2
        outputs = []
        for path in self:
            if not path:
                continue

            # offset curves
            forward, backward = [], []
            for cmd, args in path:
                if cmd == PATH_LINE or cmd == PATH_CLOSED:
                    line = np.array(args)
                    line_forward = line_offset(line, dist)
                    if line_forward is None:
                        continue
                    forward.append(line_forward)
                    backward.append(line_offset(line, -dist))
                elif cmd == PATH_CUBIC:
                    cubic = np.array(args)
                    forward.extend(bezier3_offset(cubic, dist))
                    backward.extend(bezier3_offset(cubic, -dist))
                elif cmd == PATH_QUAD:
                    cubic = bezier2_to_bezier3(args)
                    forward.extend(bezier3_offset(cubic, dist))
                    backward.extend(bezier3_offset(cubic, -dist))
                elif cmd == PATH_ARC:
                    for cubic in arc_to_bezier3(*args):
                        forward.extend(bezier3_offset(cubic, dist))
                        backward.extend(bezier3_offset(cubic, -dist))
                elif cmd == PATH_UNCLOSED:
                    continue
                else:
                    raise ValueError(f"unsupported path type: `{cmd}`")
            closed = cmd == PATH_CLOSED
            if not forward:
                continue

            # connect curves
            curves = []
            for curve in forward:
                if not curves:
                    curves.append(curve)
                    continue
                curves.extend(stroke_line_join(curves[-1], curve, linejoin))
                curves.append(curve)
            # complete subpath if path is closed or add line cap
            if closed:
                curves.extend(stroke_line_join(curves[-1], curves[0], linejoin))
                outputs.append([(curve_names[len(curve)], np.array(curve)) for curve in curves])
                curves = []
            else:
                curves.extend(stroke_line_cap(curves[-1][-1], backward[-1][-1], linecap))
            # extend subpath with backward path
            while backward:
                curve = list(reversed(backward.pop()))
                if not curves:
                    curves.append(curve)
                    continue
                curves.extend(stroke_line_join(curves[-1], curve, linejoin))
                curves.append(curve)
            # complete subpath
            if closed:
                curves.extend(stroke_line_join(curves[-1], curves[0], linejoin))
            else:
                curves.extend(stroke_line_cap(curves[-1][-1], curves[0][0], linecap))
            outputs.append([(curve_names[len(curve)], np.array(curve)) for curve in curves])

        return Path(outputs)

    def transform(self, transform: Transform) -> Path:
        """Apply transformation to a path

        This method is usually not used directly but rather transformation is
        passed to mask/fill method.
        """
        paths_out = []
        for path_in in self.subpaths:
            path_out = []
            if not path_in:
                continue
            for cmd, args in path_in:
                if cmd == PATH_ARC:
                    cubics = arc_to_bezier3(*args)
                    for cubic in transform(cubics):
                        path_out.append((PATH_CUBIC, cubic.tolist()))
                else:
                    points = transform(np.array(args)).tolist()
                    path_out.append((cmd, points))
            paths_out.append(path_out)
        return Path(paths_out)

    def to_svg(self) -> str:
        """Convert to SVG path"""
        output = io.StringIO()
        for path in self.subpaths:
            if not path:
                continue
            cmd_prev = None
            for cmd, args in path:
                if cmd == PATH_LINE:
                    (x0, y0), (x1, y1) = args
                    if cmd_prev != cmd:
                        if cmd_prev is None:
                            output.write(f"M{x0:g},{y0:g} ")
                        else:
                            output.write("L")
                    output.write(f"{x1:g},{y1:g} ")
                    cmd_prev = PATH_LINE
                elif cmd == PATH_QUAD:
                    (x0, y0), (x1, y1), (x2, y2) = args
                    if cmd_prev != cmd:
                        if cmd_prev is None:
                            output.write(f"M{x0:g},{y0:g} ")
                        output.write("Q")
                    output.write(f"{x1:g},{y1:g} {x2:g},{y2:g} ")
                    cmd_prev = PATH_QUAD
                elif cmd in {PATH_CUBIC, PATH_ARC}:
                    if cmd == PATH_ARC:
                        cubics = arc_to_bezier3(*args)
                    else:
                        cubics = [args]
                    for args in cubics:
                        (x0, y0), (x1, y1), (x2, y2), (x3, y3) = args
                        if cmd_prev != cmd:
                            if cmd_prev is None:
                                output.write(f"M{x0:g},{y0:g} ")
                            output.write("C")
                        output.write(f"{x1:g},{y1:g} {x2:g},{y2:g} {x3:g},{y3:g} ")
                        cmd_prev = PATH_CUBIC
                elif cmd == PATH_CLOSED:
                    output.write("Z ")
                    cmd_prev = None
                elif cmd == PATH_UNCLOSED:
                    cmd_prev = None
                else:
                    raise ValueError("unhandled path type: `{cmd}`")
            output.write("\n")
        return output.getvalue()[:-1]

    @staticmethod
    def from_svg(input: str) -> Path:
        """Parse SVG path

        For more info see [SVG spec](https://www.w3.org/TR/SVG11/paths.html)
        """
        input_len = len(input)
        input_offset = 0

        WHITESPACE = set(" \t\r\n,")
        COMMANDS = set("MmZzLlHhVvCcSsQqTtAa")

        def position(is_relative, pos, dst):
            return [pos[0] + dst[0], pos[1] + dst[1]] if is_relative else dst

        def smooth(points):
            px, py = points[-1]
            cx, cy = points[-2]
            return [px * 2 - cx, py * 2 - cy]

        # parser state
        paths = []
        path = []

        args = []
        cmd = None
        pos = [0.0, 0.0]
        first = True  # true if this is a first command
        start = [0.0, 0.0]

        smooth_cubic = None
        smooth_quad = None

        while input_offset <= input_len:
            char = input[input_offset] if input_offset < input_len else None

            if char in WHITESPACE:
                # remove whitespaces
                input_offset += 1

            elif char is None or char in COMMANDS:
                # process current command
                cmd_args, args = args, []

                if cmd is None:
                    pass
                elif cmd in "Mm":
                    # terminate current path
                    if path:
                        path.append((PATH_UNCLOSED, [pos, start]))
                        paths.append(path)
                        path = []

                    is_relative = cmd == "m"
                    (move, *lineto) = chunk(cmd_args, 2)
                    pos = position(is_relative and not first, pos, move)
                    start = pos
                    for dst in lineto:
                        dst = position(is_relative, pos, dst)
                        path.append((PATH_LINE, [pos, dst]))
                        pos = dst
                # line to
                elif cmd in "Ll":
                    for dst in chunk(cmd_args, 2):
                        dst = position(cmd == "l", pos, dst)
                        path.append((PATH_LINE, [pos, dst]))
                        pos = dst
                # vertical line to
                elif cmd in "Vv":
                    if not cmd_args:
                        raise ValueError(f"command '{cmd}' expects at least one argument")
                    is_relative = cmd == "v"
                    for dst in cmd_args:
                        dst = position(is_relative, pos, [0 if is_relative else pos[0], dst])
                        path.append((PATH_LINE, [pos, dst]))
                        pos = dst
                # horizontal line to
                elif cmd in "Hh":
                    if not cmd_args:
                        raise ValueError(f"command '{cmd}' expects at least one argument")
                    is_relative = cmd == "h"
                    for dst in cmd_args:
                        dst = position(is_relative, pos, [dst, 0 if is_relative else pos[1]])
                        path.append((PATH_LINE, [pos, dst]))
                        pos = dst
                # cubic bezier curve
                elif cmd in "Cc":
                    for points in chunk(cmd_args, 6):
                        points = [position(cmd == "c", pos, point) for point in chunk(points, 2)]
                        path.append((PATH_CUBIC, [pos, *points]))
                        pos = points[-1]
                        smooth_cubic = smooth(points)
                # smooth cubic bezier curve
                elif cmd in "Ss":
                    for points in chunk(cmd_args, 4):
                        points = [position(cmd == "s", pos, point) for point in chunk(points, 2)]
                        if smooth_cubic is None:
                            smooth_cubic = pos
                        path.append((PATH_CUBIC, [pos, smooth_cubic, *points]))
                        pos = points[-1]
                        smooth_cubic = smooth(points)
                # quadratic bezier curve
                elif cmd in "Qq":
                    for points in chunk(cmd_args, 4):
                        points = [position(cmd == "q", pos, point) for point in chunk(points, 2)]
                        path.append((PATH_QUAD, [pos, *points]))
                        pos = points[-1]
                        smooth_quad = smooth(points)
                # smooth quadratic bezier curve
                elif cmd in "Tt":
                    for points in chunk(cmd_args, 2):
                        points = position(cmd == "t", pos, points)
                        if smooth_quad is None:
                            smooth_quad = pos
                        points = [pos, smooth_quad, points]
                        path.append((PATH_QUAD, points))
                        pos = points[-1]
                        smooth_quad = smooth(points)
                # elliptical arc
                elif cmd in "Aa":
                    # NOTE: `large_f`, and `sweep_f` are not float but flags which can only be
                    #       0 or 1 and as the result some svg minimizers merge them with next
                    #       float which may break current parser logic.
                    for points in chunk(cmd_args, 7):
                        rx, ry, x_axis_rot, large_f, sweep_f, dst_x, dst_y = points
                        dst = position(cmd == "a", pos, [dst_x, dst_y])
                        src, pos = pos, dst
                        if rx == 0 or ry == 0:
                            path.append((PATH_LINE, [pos, dst]))
                        else:
                            path.append(
                                (
                                    PATH_ARC,
                                    arc_svg_to_parametric(
                                        src,
                                        dst,
                                        rx,
                                        ry,
                                        x_axis_rot,
                                        large_f > 0.001,
                                        sweep_f > 0.001,
                                    ),
                                )
                            )
                # close current path
                elif cmd in "Zz":
                    if cmd_args:
                        raise ValueError(f"`z` command does not accept any argmuents: {cmd_args}")
                    path.append((PATH_CLOSED, [pos, start]))
                    if path:
                        paths.append(path)
                        path = []
                    pos = start
                else:
                    raise ValueError(f"unsuppported command '{cmd}' at: {input_offset}")

                if cmd is not None and cmd not in "CcSs":
                    smooth_cubic = None
                if cmd is not None and cmd not in "QqTt":
                    smooth_quad = None
                first = False
                input_offset += 1
                cmd = char

            else:
                # parse float arguments
                match = FLOAT_RE.match(input, input_offset)
                if match:
                    match_str = match.group(0)
                    args.append(float(match_str))
                    input_offset += len(match_str)
                else:
                    raise ValueError(f"not recognized command '{char}' at: {input_offset}")

        if path:
            path.append((PATH_UNCLOSED, [pos, start]))
            paths.append(path)

        return Path(paths)

    def is_empty(self):
        return not bool(self.subpaths)

    def __repr__(self):
        if not self.subpaths:
            return "EMPTY"
        output = io.StringIO()
        for subpath in self.subpaths:
            for type, coords in subpath:
                if type == PATH_LINE:
                    output.write(f"LINE {repr_coords(coords)}\n")
                elif type == PATH_CUBIC:
                    output.write(f"CUBIC {repr_coords(coords)}\n")
                elif type == PATH_QUAD:
                    output.write(f"QUAD {repr_coords(coords)}\n")
                elif type == PATH_ARC:
                    center, rx, ry, phi, eta, eta_delta = coords
                    output.write(f"ARC ")
                    output.write(f"{repr_coords([center])} ")
                    output.write(f"{rx:.4g} {ry:.4g} ")
                    output.write(f"{phi:.3g} {eta:.3g} {eta_delta:.3g}\n")
                elif type == PATH_CLOSED:
                    output.write("CLOSE\n")
        return output.getvalue()[:-1]


def repr_coords(coords):
    return " ".join(f"{x:.4g},{y:.4g}" for x, y in coords)


# offset along tanget to approximate circle with four bezier3 curves
CIRCLE_BEIZER_OFFSET = 4 * (math.sqrt(2) - 1) / 3


def stroke_line_cap(p0, p1, linecap=None):
    """Generate path connecting two curves p0 and p1 with a cap"""
    if linecap is None:
        linecap = STROKE_CAP_BUTT
    if np.allclose(p0, p1):
        return []
    if linecap == STROKE_CAP_BUTT:
        return [np.array([p0, p1])]
    elif linecap == STROKE_CAP_ROUND:
        seg = p1 - p0
        radius = np.linalg.norm(seg) / 2
        seg /= 2 * radius
        seg_norm = np.array([-seg[1], seg[0]])
        offset = CIRCLE_BEIZER_OFFSET * radius
        center = (p0 + p1) / 2
        midpoint = center + seg_norm * radius
        return [
            np.array([p0, p0 + seg_norm * offset, midpoint - seg * offset, midpoint]),
            np.array([midpoint, midpoint + seg * offset, p1 + seg_norm * offset, p1]),
        ]
    elif linecap == STROKE_CAP_SQUARE:
        seg = p1 - p0
        seg_norm = np.array([-seg[1], seg[0]])
        polyline = [p0, p0 + seg_norm / 2, p1 + seg_norm / 2, p1]
        return [np.array([s0, s1]) for s0, s1 in zip(polyline, polyline[1:])]
    else:
        raise ValueError(f"unkown line cap type: `{linecap}`")


def stroke_line_join(c0, c1, linejoin=None, miterlimit=4):
    """Stroke used at the joints of paths"""
    if linejoin is None:
        linejoin = STROKE_JOIN_MITER
    if linejoin == STROKE_JOIN_BEVEL:
        return [np.array([c0[-1], c1[0]])]
    _, l0 = stroke_curve_tangent(c0)
    l1, _ = stroke_curve_tangent(c1)
    if l0 is None or l1 is None:
        return [np.array([c0[-1], c1[0]])]
    if np.allclose(l0[-1], l1[0]):
        return []
    p, t0, t1 = line_intersect(l0, l1)
    if p is None or (0 <= t0 <= 1 and 0 <= t1 <= 1):
        # curves intersect or parallel
        return [np.array([c0[-1], c1[0]])]
    # FIXME correctly determine miterlength: stroke_width / sin(eta / 2)
    if abs(t0) < miterlimit and abs(t1) < miterlimit:
        if linejoin == STROKE_JOIN_MITER:
            return [np.array([c0[-1], p]), np.array([p, c1[0]])]
        elif linejoin == STROKE_JOIN_ROUND:
            # FIXME: correctly produce round instead quad curve
            return [np.array([c0[-1], p, c1[0]])]
    return [np.array([c0[-1], c1[0]])]


def stroke_curve_tangent(curve):
    """Find tangents of a curve at t = 0 and at t = 1 points"""
    segs = []
    for p0, p1 in zip(curve, curve[1:]):
        if np.allclose(p0, p1):
            continue
        segs.append([p0, p1])
    if not segs:
        return None, None
    return segs[0], segs[-1]


def chunk(vs, size):
    """Chunk list `vs` into chunk of size `size`"""
    chunks = [vs[i : i + size] for i in range(0, len(vs), size)]
    if not chunks or len(chunks[-1]) != size:
        raise ValueError(f"list {vs} can not be chunked in {size}s")
    return chunks


# ------------------------------------------------------------------------------
# Gradients
# ------------------------------------------------------------------------------
class GradLinear(NamedTuple):
    p0: np.ndarray
    p1: np.ndarray
    stops: List[Tuple[float, np.ndarray]]
    transform: Optional[Transform]
    spread: str
    bbox_units: bool
    linear_rgb: Optional[bool]

    def fill(self, pixels, linear_rgb=True):
        """Fill pixels (array of coordinates) with gradient

        Returns new array same size as pixels filled with gradient
        """
        if self.transform is not None:
            pixels = self.transform.invert(pixels)

        vec = self.p1 - self.p0
        offset = (pixels - self.p0) @ vec / np.dot(vec, vec)
        return grad_interpolate(grad_spread(offset, self.spread), self.stops, linear_rgb)


class GradRadial(NamedTuple):
    center: np.ndarray
    radius: float
    fcenter: Optional[np.ndarray]
    fradius: float
    stops: List[Tuple[float, np.ndarray]]
    transform: Optional[Transform]
    spread: str
    bbox_units: bool
    linear_rgb: Optional[bool]

    def fill(self, pixels, linear_rgb=True):
        """Fill pixels (array of coordinates) with gradient

        Returns new array same size as pixels filled with gradient.

        Two circle gradient is an interpolation between two cirles (c0, r0) and (c1, r1),
        with center `c(t) = (1 - t) * c0 + t * c1`, and radius `r(t) = (1 - t) * r0 + t * r1`.
        If we have a pixel with coordinates `p`, we should solve equation for it
        `|| c(t) - p || = r(t)` and pick solution corresponding to bigger radius.

        Solving this equation for `t`:
            || c(t) - p || = r(t)  -> At² - 2Bt + C = 0
        where:
            cd = c2 - c1
            pd = p - c1
            rd = r2 - r1
            A = cdx ^ 2 + cdy ^ 2 - rd ^ 2
            B = pdx * cdx + pdy * cdy + r1 * rd
            C = pdx ^2 + pdy ^ 2 - r1 ^ 2
        results in:
            t = (B +/- (B ^ 2 - A * C).sqrt()) / A

        [reference]: https://cgit.freedesktop.org/pixman/tree/pixman/pixman-radial-gradient.c
        """
        mask = None
        if self.transform is not None:
            pixels = self.transform.invert(pixels)

        if self.fcenter is None and self.fradius is None:
            offset = (pixels - self.center) / self.radius
            offset = np.sqrt((offset * offset).sum(axis=-1))
        else:
            fcenter = self.center if self.fcenter is None else self.fcenter
            fradius = self.fradius or 0

            # This is SVG 1.1 behavior. If focal center is outside of circle it
            # should be moved inside. But in SVG 2.0 it should produce a cone
            # shaped gradient.
            # fdist = np.linalg.norm(fcenter - self.center)
            # if fdist > self.radius:
            #     fcenter = self.center + (fcenter - self.center) * self.radius / fdist

            cd = self.center - fcenter
            pd = pixels - fcenter
            rd = self.radius - fradius
            a = (cd**2).sum() - rd**2
            b = (pd * cd).sum(axis=-1) + fradius * rd
            c = (pd**2).sum(axis=-1) - fradius**2

            det = b * b - a * c
            if (det < 0).any():
                mask = det >= 0
                det = det[mask]
                b = b[mask]
                c = c[mask]

            t0 = np.sqrt(det)
            t1 = (b + t0) / a
            t2 = (b - t0) / a

            if mask is None:
                offset = np.maximum(t1, t2)
            else:
                offset = np.zeros(mask.shape, dtype=FLOAT)
                offset[mask] = np.maximum(t1, t2)
                if fradius != self.radius:
                    # exclude negative `r(t)`
                    mask &= offset > (fradius / (fradius - self.radius))

        overlay = grad_interpolate(grad_spread(offset, self.spread), self.stops, linear_rgb)
        if mask is not None:
            overlay[~mask] = np.array([0, 0, 0, 0])

        return overlay


def grad_pixels(viewport):
    """Create pixels matrix to be filled by gradient"""
    off_x, off_y, width, height = viewport
    xs, ys = np.indices((width, height)).astype(FLOAT)
    offset = [off_x + 0.5, off_y + 0.5]
    return np.concatenate([xs[..., None], ys[..., None]], axis=2) + offset


def grad_spread(offsets, spread):
    if spread == "pad":
        return offsets
    elif spread == "repeat":
        return np.modf(offsets)[0]
    elif spread == "reflect":
        return np.fabs(np.remainder(offsets + 1.0, 2.0) - 1.0)
    raise ValueError(f"invalid spread method: {spread}")


def grad_interpolate(offset, stops, linear_rgb):
    """Create gradient by interpolating offsets from stops"""
    stops = grad_stops_colorspace(stops, linear_rgb)
    output = np.zeros((*offset.shape, 4), dtype=FLOAT)
    o_min, c_min = stops[0]
    output[offset <= o_min] = c_min
    o_max, c_max = stops[-1]
    output[offset > o_max] = c_max
    for (o0, c0), (o1, c1) in zip(stops, stops[1:]):
        mask = np.logical_and(offset > o0, offset <= o1)
        ratio = ((offset[mask] - o0) / (o1 - o0))[..., None]
        output[mask] += (1 - ratio) * c0 + ratio * c1
    return output


def grad_stops_colorspace(stops, linear_rgb=False):
    if linear_rgb:
        return stops
    output = []
    for offset, color in stops:
        color = color_pre_to_straight_alpha(color.copy())
        color = color_linear_to_srgb(color)
        color = color_straight_to_pre_alpha(color)
        output.append((offset, color))
    return output


class Pattern(NamedTuple):
    scene: Scene
    scene_bbox_units: bool
    scene_view_box: Optional[Tuple[float, float, float, float]]
    x: float
    y: float
    width: float
    height: float
    transform: Transform
    bbox_units: bool

    def bbox(self):
        return (self.x, self.y, self.width, self.height)


# ------------------------------------------------------------------------------
# Filter
# ------------------------------------------------------------------------------
FE_BLEND = 0
FE_COLOR_MATRIX = 1
FE_COMPONENT_TRANSFER = 2
FE_COMPOSITE = 3
FE_CONVOLVE_MATRIX = 4
FE_DIFFUSE_LIGHTING = 5
FE_DISPLACEMENT_MAP = 6
FE_FLOOD = 7
FE_GAUSSIAN_BLUR = 8
FE_MERGE = 9
FE_MORPHOLOGY = 10
FE_OFFSET = 11
FE_SPECULAR_LIGHTING = 12
FE_TILE = 13
FE_TURBULENCE = 14

FE_SOURCE_ALPHA = "SourceAlpha"
FE_SOURCE_GRAPHIC = "SourceGraphic"

COLOR_MATRIX_LUM = np.array(
    [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0.2125, 0.7154, 0.0721, 0, 0]], dtype=FLOAT
)
COLOR_MATRIX_HUE = np.array(
    [
        [[0.213, 0.715, 0.072], [0.213, 0.715, 0.072], [0.213, 0.715, 0.072]],
        [[0.787, -0.715, -0.072], [-0.213, 0.285, -0.072], [-0.213, -0.715, 0.928]],
        [[-0.213, -0.715, 0.928], [0.143, 0.140, -0.283], [-0.787, 0.715, 0.072]],
    ],
    dtype=FLOAT,
)


class Filter(NamedTuple):
    names: Dict[str, int]  # {name: index}
    filters: List[Tuple[int, List[Any], List[int]]]  # [(type, attrs, inputs)]

    @classmethod
    def empty(cls):
        return cls({FE_SOURCE_ALPHA: 0, FE_SOURCE_GRAPHIC: 1}, [])

    def add_filter(self, type, attrs, inputs, result):
        names = self.names.copy()
        filters = self.filters.copy()

        args = []
        for input in inputs:
            if input is None:
                args.append(len(filters) + 1)  # use previous result
            else:
                arg = self.names.get(input)
                if arg is None:
                    warnings.warn(f"unknown filter result name: {input}")
                    args.append(len(filters) + 1)  # use previous result
                else:
                    args.append(arg)

        if result is not None:
            names[result] = len(filters) + 2

        filters.append((type, attrs, args))
        return Filter(names, filters)

    def offset(self, dx, dy, input=None, result=None):
        return self.add_filter(FE_OFFSET, (dx, dy), [input], result)

    def merge(self, inputs, result=None):
        return self.add_filter(FE_MERGE, tuple(), inputs, result)

    def blur(self, std_x, std_y=None, input=None, result=None):
        return self.add_filter(FE_GAUSSIAN_BLUR, (std_x, std_y), [input], result)

    def blend(self, in1, in2, mode=None, result=None):
        return self.add_filter(FE_BLEND, (mode,), [in1, in2], result)

    def composite(self, in1, in2, mode=None, result=None):
        return self.add_filter(FE_COMPOSITE, (mode,), [in1, in2], result)

    def color_matrix(self, input, matrix, result=None):
        return self.add_filter(FE_COLOR_MATRIX, (matrix,), [input], result)

    def morphology(self, rx, ry, method, input, result=None):
        return self.add_filter(FE_MORPHOLOGY, (rx, ry, method), [input], result)

    def __call__(self, transform: Transform, source: Layer) -> Layer:
        """Execute filter on the provided source"""
        alpha = Layer(
            source.image[..., -1:] * np.array([0, 0, 0, 1]),
            source.offset,
            pre_alpha=True,
            linear_rgb=True,
        )
        stack = [alpha, source.convert(pre_alpha=False, linear_rgb=True)]

        for filter in self.filters:
            type, attrs, inputs = filter
            if type == FE_OFFSET:
                fn = filter_offset(transform, *attrs)
            elif type == FE_MERGE:
                fn = filter_merge(transform, *attrs)
            elif type == FE_BLEND:
                fn = filter_blend(transform, *attrs)
            elif type == FE_COMPOSITE:
                fn = filter_composite(transform, *attrs)
            elif type == FE_GAUSSIAN_BLUR:
                fn = filter_blur(transform, *attrs)
            elif type == FE_COLOR_MATRIX:
                fn = filter_color_matrix(transform, *attrs)
            elif type == FE_MORPHOLOGY:
                fn = filter_morphology(transform, *attrs)
            else:
                raise ValueError(f"unsupported filter type: {type}")
            stack.append(fn(*(stack[input] for input in inputs)))

        return stack[-1]


def filter_color_matrix(_transform, matrix):
    def filter_color_matrix_apply(input):
        if not isinstance(matrix, np.ndarray) or matrix.shape != (4, 5):
            warnings.warn(f"invalid color matrix: {matrix}")
            return input
        return input.color_matrix(matrix)

    return filter_color_matrix_apply


def filter_offset(transform, dx, dy):
    def filter_offset_apply(input):
        x, y = input.offset
        tx, ty = transform(transform.invert([x, y]) + [dx, dy])
        return input.translate(int(tx) - x, int(ty) - y)

    return filter_offset_apply


def filter_morphology(transform, rx, ry, method):
    def filter_morphology_apply(input):
        # NOTE:
        # I have no idea how to account for rotation, except to rotate
        # apply morphology and rotate back, but it is slow, so I'm not doing it
        ux, uy = transform([[rx, 0], [0, ry]]) - transform([[0, 0], [0, 0]])
        x = int(np.linalg.norm(ux) * 2)
        y = int(np.linalg.norm(uy) * 2)
        if x < 1 or y < 1:
            return input
        return input.morphology(x, y, method)

    return filter_morphology_apply


def filter_merge(_transform):
    def filter_merge_apply(*inputs):
        return Layer.compose(inputs, linear_rgb=True)

    return filter_merge_apply


def filter_blend(_transform, mode):
    def filter_blend_apply(in1, in2):
        warnings.warn("feBlend is not properly supported")
        return Layer.compose([in2, in1], linear_rgb=True)

    return filter_blend_apply


def filter_composite(_transform, mode):
    def filter_composite_apply(in1, in2):
        return Layer.compose([in2, in1], mode, linear_rgb=True)

    return filter_composite_apply


def filter_blur(transform, std_x, std_y=None):
    if std_y is None:
        std_y = std_x

    def filter_blur_apply(input):
        kernel = blur_kernel(transform, (std_x, std_y))
        if kernel is None:
            return input
        return input.convolve(kernel)

    return filter_blur_apply


def blur_kernel(transform, sigma):
    """Gaussian blur convolution kernel

    Gaussian kernel ginven presentation transformation and sigma in user
    coordinate system.
    """
    sigma_x, sigma_y = sigma

    # if one of the sigmas is smaller then a pixel rotatetion produces
    # incorrect degenerate state when the whole convolution is the same as over
    # a delta function. So we need to adjust it. If both simgas are smaller
    # then a pixel then gaussian blur is a no-op.
    scale_x, scale_y = np.linalg.norm(transform(np.eye(2)) - transform([0, 0]), axis=1)
    if scale_x * sigma_x < 0.5 and scale_y * sigma_y < 0.5:
        return None
    elif scale_x * sigma_x < 0.5:
        sigma_x = 0.5 / scale_x
    elif scale_y * sigma_y < 0.5:
        sigma_y = 0.5 / scale_y

    sigma = np.array([sigma_x, sigma_y])
    sigmas = 2.5
    user_box = [
        [-sigmas * sigma_x, -sigmas * sigma_y],
        [-sigmas * sigma_x, sigmas * sigma_y],
        [sigmas * sigma_x, sigmas * sigma_y],
        [sigmas * sigma_x, -sigmas * sigma_y],
    ]
    box = transform(user_box) - transform([0, 0])
    min_x, min_y = box.min(axis=0).astype(int)
    max_x, max_y = box.max(axis=0).astype(int)
    kernel_w, kernel_h = max_x - min_x, max_y - min_y
    kernel_w += ~kernel_w & 1  # make it odd
    kernel_h += ~kernel_h & 1

    user_tr = transform.invert
    kernel = user_tr(grad_pixels([-kernel_w / 2, -kernel_h / 2, kernel_w, kernel_h]))
    kernel -= user_tr([0, 0])  # remove translation
    kernel = np.exp(-np.square(kernel) / (2 * np.square(sigma)))
    kernel = kernel.prod(axis=-1)

    return kernel / kernel.sum()


def color_matrix_hue_rotate(angle):
    """Hue rotation matrix for specified angle in radians"""
    matrix = np.eye(4, 5)
    matrix[:3, :3] = np.dot(COLOR_MATRIX_HUE.T, [1, math.cos(angle), math.sin(angle)]).T
    return matrix


def color_matrix_saturate(value):
    matrix = np.eye(4, 5)
    matrix[:3, :3] = np.dot(COLOR_MATRIX_HUE.T, [1, value, 0]).T
    return matrix


# ------------------------------------------------------------------------------
# Convex Hull
# ------------------------------------------------------------------------------
class ConvexHull:
    """Convex hull using graham scan

    Points are stored in presenetation coordinate system, so we would not have
    to convert them back and force when merging.
    """

    __slots__ = ["points"]

    def __init__(self, points):
        """Construct convex hull of a set of points using graham scan"""
        if isinstance(points, np.ndarray):
            points = points.reshape(-1, 2).tolist()

        def turn(p, q, r):
            return (q[0] - p[0]) * (r[1] - p[1]) - (r[0] - p[0]) * (q[1] - p[1])

        def keep_left(hull, p):
            while len(hull) > 1 and turn(hull[-2], hull[-1], p) <= 0:
                hull.pop()
            if not hull or hull[-1] != p:
                hull.append(p)
            return hull

        points = sorted(points)
        left = reduce(keep_left, points, [])
        right = reduce(keep_left, reversed(points), [])
        left.extend(right[1:-1])

        self.points = left

    @classmethod
    def merge(cls, hulls):
        """Merge multiple convex hulls into one"""
        points = []
        for hull in hulls:
            points.extend(hull.points)
        return cls(points)

    def bbox(self, transform):
        """Bounding box in user coordinate system"""
        points = transform.invert(np.array(self.points))
        min_x, min_y = points.min(axis=0)
        max_x, max_y = points.max(axis=0)
        return [min_x, min_y, max_x - min_x, max_y - min_y]

    def bbox_transform(self, transform):
        """Transformation matrix for `objectBoundingBox` units

        Create bounding box transfrom for `objectBoundingBox`, using convex hull in the
        canvas coordinate system and current user space transformation.
        `objectBoundingBox` is a coordinate system where bounding box is a unit square.
        Returns `objectBoundingBox` transform.

        FIXME: In case of stroke we should use bounding box of the original path,
               not the stroked path.
        """
        x, y, w, h = self.bbox(transform)
        if w <= 0 and h <= 0:
            return transform
        return transform.translate(x, y).scale(w, h)

    def path(self):
        points = self.points
        lines = [(PATH_LINE, l) for l in zip(points, points[1:])]
        lines.append((PATH_CLOSED, [points[-1], points[0]]))
        return Path([lines])


# ------------------------------------------------------------------------------
# Bezier
# ------------------------------------------------------------------------------
BEZIER3_FLATNESS = np.array([[-2, 3, 0, -1], [-1, 0, 3, -2]], dtype=FLOAT)
BEZIER3_SPLIT = np.array(
    [
        [1, 0, 0, 0],
        [0.5, 0.5, 0, 0],
        [0.25, 0.5, 0.25, 0],
        [0.125, 0.375, 0.375, 0.125],
        [0.125, 0.375, 0.375, 0.125],
        [0, 0.25, 0.5, 0.25],
        [0, 0, 0.5, 0.5],
        [0, 0, 0, 1],
    ],
    dtype=FLOAT,
)
BEZIER3_MAT = np.array([[1, 0, 0, 0], [-3, 3, 0, 0], [3, -6, 3, 0], [-1, 3, -3, 1]], dtype=FLOAT)
BEZIER2_MAT = np.array([[1, 0, 0], [-2, 2, 0], [1, -2, 1]], dtype=FLOAT)
BEZIER3_ABC = np.array([[-3, 9, -9, 3], [6, -12, 6, 0], [-3, 3, 0, 0]], dtype=FLOAT)
BEZIER2_TO_BEZIER3 = np.array(
    [[1, 0, 0], [1.0 / 3, 2.0 / 3, 0], [0, 2.0 / 3.0, 1.0 / 3], [0, 0, 1]], dtype=FLOAT
)
BEZIER = {2: [[0, 1], [1, -1]], 3: BEZIER2_MAT, 4: BEZIER3_MAT}


def bezier3_split(points):
    """Split bezier3 curve in two bezier3 curves at t=0.5

    Using de Castelju construction at t=0.5
    """
    return np.matmul(BEZIER3_SPLIT, points).reshape(2, 4, 2)


def bezier3_split_batch(batch):
    """Split N bezier3 curves in 2xN bezier3 curves at t=0.5"""
    return np.moveaxis(np.dot(BEZIER3_SPLIT, batch), 0, -2).reshape((-1, 4, 2))


def bezier3_flatness_batch(batch):
    """Flattness criteria for a batch of bezier3 curves

    It is equal to `f = maxarg d(t) where d(t) = |b(t) - l(t)|, l(t) = (1 - t) * b0 + t * b3`
    for b(t) bezier3 curve with b{0..3} control points, in other words maximum distance
    from parametric line to bezier3 curve for the same parameter t. It is proven in the article
    that:
        f^2 <= 1/16 (max{u_x^2, v_x^2} + max{u_y^2, v_y^2})
    where:
        u = 3 * b1 - 2 * b0 - b3
        v = 3 * b2 - b0 - 2 * b3
    `f == 0` means completely flat so estimating upper bound is sufficient as spliting more
    than needed is not a problem for rendering.

    [Linear Approximation of Bezier Curve](https://hcklbrrfnn.files.wordpress.com/2012/08/bez.pdf)
    """
    uv = np.moveaxis(np.square(np.dot(BEZIER3_FLATNESS, batch)), 0, -1)
    return uv.max(-2).sum(-1)


def bezier3_flatten_batch(batch, flatness):
    lines = []
    flatness = (flatness**2) * 16
    while batch.size > 0:
        flat_mask = bezier3_flatness_batch(batch) < flatness
        lines.append(batch[flat_mask][..., [0, 3], :])
        batch = bezier3_split_batch(batch[~flat_mask])
    return np.concatenate(lines)


def bezier3_bbox(points):
    a, b, c = BEZIER3_ABC @ points
    det = b**2 - 4 * a * c
    t0 = (-b + np.sqrt(det)) / (2 * a)
    t1 = (-b - np.sqrt(det)) / (2 * a)
    curve = bezier_parametric(points)
    exterm = np.array([curve(t) for t in [0, 1, *t0, *t1] if 0 <= t <= 1])
    min_x, min_y = exterm.min(axis=0)
    max_x, max_y = exterm.max(axis=0)
    return (min_x, min_y, max_x - min_x, max_y - min_y)


def bezier3_offset(curve, distance):
    """Offset bezier3 curve with a list of bezier3 curves

    Offset bezier curve using Tiller-Hanson method. In short, just offset line
    segment corresponding to control points, find intersection of this lines
    and treat them as new control points.
    """

    def should_split(curve):
        c0, c1, c2, c3 = curve
        # angle(c3 - c0, c2 - c1) > 90 or < -90
        if np.dot(c3 - c0, c2 - c1) < 0:
            return True
        # control points should be on the same side of the baseline
        a0 = np.cross(c3 - c0, c1 - c0)
        a1 = np.cross(c3 - c0, c2 - c0)
        if a0 * a1 < 0:
            return True
        # distance between center mass and midpoint of a curve,
        # bigger then .1 of bounding box diagonal.
        c_mass = curve.sum(0) / 4  # center mass
        c_mid = [0.125, 0.375, 0.375, 0.125] @ curve  # t = 0.5
        dist = ((c_mass - c_mid) ** 2).sum()
        bbox_diag = ((curve.max(0) - curve.min(0)) ** 2).sum()
        return dist * 100 > bbox_diag

    outputs = []
    curves = [curve]
    while curves:
        curve = curves.pop()
        if should_split(curve) and len(outputs) < 16:
            curves.extend(reversed(bezier3_split(curve)))
            continue

        output = []
        repeat = 0
        line = None

        for p0, p1 in zip(curve, curve[1:]):
            # skip non-offsetable lines
            if np.allclose(p0, p1):
                repeat += 1
                continue
            # offset and intersect with previous
            o0, o1 = line_offset([p0, p1], distance)
            if line is not None:
                x0, _t0, _t1 = line_intersect(line, (o0, o1))
                if x0 is not None:
                    o0 = x0
                else:
                    o0 = (line[-1] + o0) / 2
            # repeat points if needed
            for _ in range(repeat + 1):
                output.append(o0)
            repeat = 0
            line = (o0, o1)

        if line is not None:
            # insert last points
            for _ in range(repeat + 1):
                output.append(o1)
            if outputs and not np.allclose(output[0], outputs[-1][-1]):
                # hack for a curve like "M0,0 C100,50 0,50 100,0"
                outputs.extend(stroke_line_cap(output[0], outputs[-1][-1], STROKE_CAP_ROUND))
            outputs.append(output)

    return np.array(outputs)


def bezier2_to_bezier3(points):
    """Convert bezier2 to bezier3 curve"""
    return BEZIER2_TO_BEZIER3 @ points


def bezier_parametric(points):
    points = np.array(points, dtype=FLOAT)
    points_count, _ = points.shape
    matrix = BEZIER.get(points_count)
    if matrix is None:
        raise ValueError("unsupported bezier curve order: {}", points_count)
    powers = np.array(range(points_count), dtype=FLOAT)
    matrix = np.dot(matrix, points)
    return lambda t: np.power(t, powers) @ matrix


def bezier_deriv_parametric(points):
    points = np.array(points, dtype=FLOAT)
    points_count, _ = points.shape
    matrix = BEZIER.get(points_count)
    if matrix is None:
        raise ValueError("unsupported bezier curve order: {}", points_count)
    powers = np.array(range(points_count - 1), dtype=FLOAT)
    deriv_matrix = (matrix * np.array(range(points_count))[..., None])[1:]
    deriv_matrix = np.dot(deriv_matrix, points)
    return lambda t: np.power(t, powers) @ deriv_matrix


# ------------------------------------------------------------------------------
# Line
# ------------------------------------------------------------------------------
def line_signed_coverage(canvas: Image, line: Tuple[Tuple[float, float], Tuple[float, float]]) -> Image:
    """Trace line on a canvas rendering signed coverage

    Implementation details:
    Line must be specified in the canvas coordinate system (that is one
    unit of length is equal to one pixel). Line is always traversed with
    scan line along `x` coordinates from lowest value to largest, and
    scan lines are going from lowest `y` value to larges.

    Based on https://github.com/raphlinus/font-rs/blob/master/src/raster.rs
    """
    floor, ceil = math.floor, math.ceil
    min, max = builtins.min, builtins.max
    h, w = canvas.shape
    X, Y = 1, 0
    p0, p1 = line[0], line[1]

    if p0[Y] == p1[Y]:
        return  # does not introduce any signed coverage
    dir, p0, p1 = (1.0, p0, p1) if p0[Y] < p1[Y] else (-1.0, p1, p0)
    dxdy = (p1[X] - p0[X]) / (p1[Y] - p0[Y])
    # Find first point to trace. Since we are going to interate over Y's
    # we should pick min(y , p0.y) as a starting y point, and adjust x
    # accordingly
    x, y = p0[X], int(max(0, p0[Y]))
    if p0[Y] < 0:
        x -= p0[Y] * dxdy
    x_next = x
    for y in range(y, min(h, ceil(p1[Y]))):
        x = x_next
        dy = min(y + 1, p1[Y]) - max(y, p0[Y])
        d = dir * dy  # signed y difference
        # find next x position
        x_next = x + dxdy * dy
        # order (x, x_next) from smaller value x0 to bigger x1
        x0, x1 = (x, x_next) if x < x_next else (x_next, x)
        # lower bound of effected x pixels
        x0_floor = floor(x0)
        x0i = int(x0_floor)
        # upper bound of effected x pixels
        x1_ceil = ceil(x1)
        x1i = int(x1_ceil)
        if x1i <= x0i + 1:
            # only goes through one pixel
            xmf = 0.5 * (x + x_next) - x0_floor  # effective height
            if x0i >= w:
                continue
            canvas[y, x0i if x0i > 0 else 0] += d * (1 - xmf)
            xi = x0i + 1
            if xi >= w:
                continue
            canvas[y, xi if xi > 0 else 0] += d * xmf  # next pixel is fully shaded
        else:
            s = 1 / (x1 - x0)
            x0f = x0 - x0_floor  # fractional part of x0
            x1f = x1 - x1_ceil + 1.0  # fraction part of x1
            a0 = 0.5 * s * (1 - x0f) ** 2  # area of the smallest x pixel
            am = 0.5 * s * x1f**2  # area of the largest x pixel
            # first pixel
            if x0i >= w:
                continue
            canvas[y, x0i if x0i > 0 else 0] += d * a0
            if x1i == x0i + 2:
                # only two pixels are covered
                xi = x0i + 1
                if xi >= w:
                    continue
                canvas[y, xi if xi > 0 else 0] += d * (1.0 - a0 - am)
            else:
                # second pixel
                a1 = s * (1.5 - x0f)
                xi = x0i + 1
                if xi >= w:
                    continue
                canvas[y, xi if xi > 0 else 0] += d * (a1 - a0)
                # second .. last pixels
                for xi in range(x0i + 2, x1i - 1):
                    if xi >= w:
                        continue
                    canvas[y, xi if xi > 0 else 0] += d * s
                # last pixel
                a2 = a1 + (x1i - x0i - 3) * s
                xi = x1i - 1
                if xi >= w:
                    continue
                canvas[y, xi if xi > 0 else 0] += d * (1.0 - a2 - am)
            if x1i >= w:
                continue
            canvas[y, x1i if x1i > 0 else 0] += d * am
    return canvas


def line_intersect(l0, l1):
    """Find intersection betwee two line segments

    Solved by solving l(t) for both lines:
    l(t) = (1 - t) * l[0] + t * l[1]
    v ~ l[1] - l[0]

    l0(ta) == l1(tb)
        l0[0] - l1[0] = [v1, -v0] @ [tb, ta]
        inv([v1, -v0]) @ (l0[0] - l1[0]) = [tb, ta]
    """
    ((x1, y1), (x2, y2)) = l0
    ((x3, y3), (x4, y4)) = l1
    det = (x4 - x3) * (y1 - y2) - (x1 - x2) * (y4 - y3)
    if abs(det) < EPSILON:
        return None, 0, 0
    t0 = ((y3 - y4) * (x1 - x3) + (x4 - x3) * (y1 - y3)) / det
    t1 = ((y1 - y2) * (x1 - x3) + (x2 - x1) * (y1 - y3)) / det
    return [x1 * (1 - t0) + x2 * t0, y1 * (1 - t0) + y2 * t0], t0, t1


def line_offset(line, distance):
    ((x1, y1), (x2, y2)) = line
    (vx, vy) = (x2 - x1, y2 - y1)
    line_len = vx * vx + vy * vy
    if line_len < EPSILON:
        return None
    line_len = math.sqrt(line_len)
    dx = -vy * distance / line_len
    dy = vx * distance / line_len
    return np.array([[x1 + dx, y1 + dy], [x2 + dx, y2 + dy]])


def line_offset_batch(batch, distance):
    """Offset a batch of line segments to specified distance"""
    norms = np.matmul([-1, 1], batch) @ [[0, 1], [-1, 0]]
    norms_len = np.sqrt((norms**2).sum(-1))[..., None]
    offset = norms * distance / norms_len
    return batch + offset[..., None, :]


def line_parametric(points):
    return bezier_parametric(points)


# ------------------------------------------------------------------------------
# Arc
# ------------------------------------------------------------------------------
def arc_to_bezier3(center, rx, ry, phi, eta, eta_delta):
    """Approximate arc with a sequence of cubic bezier curves

    [Drawing an elliptical arc using polylines, quadratic or cubic Bezier curves]
    (http://www.spaceroots.org/documents/ellipse/elliptical-arc.pdf)
    [Approximating Arcs Using Cubic Bezier Curves]
    (https://www.joecridge.me/content/pdf/bezier-arcs.pdf)

    We are using following formula to split arc segment from `eta_1` to `eta_2`
    to achieve good approximation arc is split in segments smaller then `pi / 2`.
        P0 = A(eta_1)
        P1 = P0 + alpha * A'(eta_1)
        P2 = P3 - alpha * A'(eta_2)
        P3 = A(eta_2)
    where
        A - arc parametrized by angle
        A' - derivative of arc parametrized by angle
        eta_1 = eta
        eta_2 = eta + eta_delta
        alpha = sin(eta_2 - eta_1) * (sqrt(4 + 3 * tan((eta_2 - eta_1) / 2) ** 2) - 1) / 3
    """
    M = np.array([[math.cos(phi), -math.sin(phi)], [math.sin(phi), math.cos(phi)]])
    arc = lambda a: M @ [rx * math.cos(a), ry * math.sin(a)] + center
    arc_d = lambda a: M @ [-rx * math.sin(a), ry * math.cos(a)]

    segment_max_angle = math.pi / 4  # maximum `eta_delta` of a segment
    segments = []
    segments_count = math.ceil(abs(eta_delta) / segment_max_angle)
    etas = np.linspace(eta, eta + eta_delta, segments_count + 1)
    for eta_1, eta_2 in zip(etas, etas[1:]):
        sq = math.sqrt(4 + 3 * math.tan((eta_2 - eta_1) / 2) ** 2)
        alpha = math.sin(eta_2 - eta_1) * (sq - 1) / 3

        p0 = arc(eta_1)
        p3 = arc(eta_2)
        p1 = p0 + alpha * arc_d(eta_1)
        p2 = p3 - alpha * arc_d(eta_2)
        segments.append([p0, p1, p2, p3])

    return np.array(segments)


def arc_svg_to_parametric(src, dst, rx, ry, x_axis_rot, large_flag, sweep_flag):
    """Convert arc from SVG arguments to parametric curve

    This code mostly comes from arc implementation notes from svg spec
    (Arc to Parametric)[https://www.w3.org/TR/SVG/implnote.html#ArcImplementationNotes]

    Returns arc parameters in parameteric form:
        - `src`, `dst` - usefull if implementation wants to know stat and end without recomputation
        - `center`     - center of an ellipse
        - `rx`, `ry`   - ellipse radii
        - `phi`        - ellipse rotation with regard to x axis
        - `eta`        - initial parameter angle
        - `eta_delta`  - difference between final and initial `eta` values

    Curve is specified in a following form:
        eta(t) = eta + t * eta_delta
        arc(t) = [[cos(phi), -sin(phi)], [sin(phi), cos(phi)]] @ [rx * cos(eta(t)), ry * sin(eta(t))] + [cx, cy]
    """
    rx, ry = abs(rx), abs(ry)
    src, dst = np.array(src), np.array(dst)
    phi = x_axis_rot * math.pi / 180

    cos_phi, sin_phi = math.cos(phi), math.sin(phi)
    M = np.array([[cos_phi, sin_phi], [-sin_phi, cos_phi]])
    # Eq 5.1
    x1, y1 = np.matmul(M, (src - dst) / 2)
    # scale/normalize radii (Eq 6.2)
    s = (x1 / rx) ** 2 + (y1 / ry) ** 2
    if s > 1:
        s = math.sqrt(s)
        rx *= s
        ry *= s
    # Eq 5.2
    sq = math.sqrt(max(0, (rx * ry) ** 2 / ((rx * y1) ** 2 + (ry * x1) ** 2) - 1))
    if large_flag == sweep_flag:
        sq = -sq
    center = sq * np.array([rx * y1 / ry, -ry * x1 / rx])
    cx, cy = center
    # Eq 5.3 convert center to initail coordinates
    center = np.matmul(M.T, center) + (dst + src) / 2
    # Eq 5.5-6
    v0 = np.array([1, 0])
    v1 = np.array([(x1 - cx) / rx, (y1 - cy) / ry])
    v2 = np.array([(-x1 - cx) / rx, (-y1 - cy) / ry])
    # initial angle
    eta = angle_between(v0, v1)
    # delta angle to be covered when t changes from 0..1
    eta_delta = math.fmod(angle_between(v1, v2), 2 * math.pi)
    if not sweep_flag and eta_delta > 0:
        eta_delta -= 2 * math.pi
    if sweep_flag and eta_delta < 0:
        eta_delta += 2 * math.pi

    return center, rx, ry, phi, eta, eta_delta


def arc_parametric(center, rx, ry, phi, eta, eta_delta):
    def arc(t):
        """Parametric arc curve"""
        angle = eta + t * eta_delta
        return M @ [rx * math.cos(angle), ry * math.sin(angle)] + center

    M = np.array([[math.cos(phi), -math.sin(phi)], [math.sin(phi), math.cos(phi)]])
    return arc


def arc_deriv_parametric(center, rx, ry, phi, eta, eta_delta):
    def arc_deriv(t):
        angle = eta + t * eta_delta
        return M @ [-rx * math.sin(angle), ry * math.cos(angle)] * eta_delta

    M = np.array([[math.cos(phi), -math.sin(phi)], [math.sin(phi), math.cos(phi)]])
    return arc_deriv


def angle_between(v0, v1):
    """Calculate angle between two vectors"""
    angle_cos = np.dot(v0, v1) / (np.linalg.norm(v0) * np.linalg.norm(v1))
    angle = math.acos(np.clip(angle_cos, -1, 1))
    if np.cross(v0, v1) < 0:
        angle = -angle
    return angle


# ------------------------------------------------------------------------------
# Render by samping parametic curve
# ------------------------------------------------------------------------------
POINTS: Dict[float, np.ndarray] = {}


def point_mask(d):
    P = POINTS.get(d)
    if P is not None:
        return P

    s = int(math.ceil(d)) + 2
    if s % 2 == 0:
        s += 1
    center = np.array([s, s], dtype=FLOAT) / 2.0
    samples = np.array([[-0.5, -0.5], [-0.5, 0.5], [0.5, -0.5], [0.5, 0.5], [0, 0]], dtype=FLOAT)

    rows = []
    for x in range(0, s):
        row = []
        for y in range(0, s):
            dist = np.array([x + 0.5, y + 0.5]) + samples - center
            row.append(((dist**2).sum(axis=1) < (d / 2) ** 2).sum() / 5)
        rows.append(row)
    return np.array(rows)[..., None]


def put_point(canvas, diam, color, point):
    w, h, _ = canvas.shape
    x, y = point
    mask = point_mask(diam)
    r = math.ceil(mask.shape[0] / 2)
    if x < r or x > w - r or y < r or y > h - r:
        return
    x = int(x - diam / 2)
    y = int(y - diam / 2)
    xi, yi = np.indices(mask.shape[:2])
    canvas_effected = canvas[xi + x, yi + y]
    canvas[xi + x, yi + y] = canvas_effected + mask[xi, yi] * (color - canvas_effected)
    return canvas


def sample_curve(canvas, radius, color, count, curve, curve_deriv=None):
    """Render parametric curve by sampling"""
    color_inv = 1 - color
    color_inv[3] = 1

    h, w, _ = canvas.shape
    x0, x1 = radius, w - 2 * radius
    y0, y1 = radius, h - 2 * radius
    for t in np.linspace(0, 1, count):
        p = curve(t).astype(int)
        if y0 < p[0] < y1 and x0 < p[1] < x1:
            if curve_deriv and curve_deriv(t)[0] < 0:
                color_render = color_inv
            else:
                color_render = color

            if radius > 1:
                put_point(canvas, radius, color_render, p)
            else:
                y, x = p
                canvas[y, x] = color_render
    return canvas


def sample_curve_points(canvas, ps):
    """Render curve control points by sampling lines"""
    color = svg_color("crimson")

    h, w, _ = canvas.shape
    for p0, p1 in zip(ps[::2], ps[1::2]):
        color = np.array([1, 0, 0, 1]) if p0[0] > p1[0] else np.array([0, 1, 1, 1])
        sample_curve(canvas, 1, color, 300, line_parametric([p0, p1]))
    for p in ps:
        x, y = p.astype(int)
        if 2 < x < h - 4 and 2 < y < w - 4:
            put_point(canvas, 5, color, np.array([x, y]))


# ------------------------------------------------------------------------------
# Font
# ------------------------------------------------------------------------------
class Glyph:
    __slots__ = ["unicode", "advance", "name", "path_source", "path_data"]

    def __init__(self, unicode: str, advance: float, path_source: str, name: Optional[str]):
        self.unicode = unicode
        self.advance = advance
        self.name = name
        self.path_source = path_source
        self.path_data: Optional[Path] = None

    @property
    def path(self) -> Path:
        if self.path_data is None:
            path = Path.from_svg(self.path_source)
            self.path_data = path
            return path
        else:
            return self.path_data

    def __repr__(self):
        return "Glyph(unicode={}, name={})".format(self.unicode, self.name)


class Font(NamedTuple):
    family: str
    weight: int
    style: str
    ascent: float
    descent: float
    units_per_em: float
    glyphs: Dict[str, Glyph]
    missing_glyph: Glyph
    hkern: Dict[Tuple[str, str], float]

    def str_to_glyphs(self, string: str) -> Tuple[List[Tuple[float, Glyph]], float]:
        """Convert string to a list of glyphs with offsets"""
        offset = 0.0
        output = []
        glyph_prev = None

        stack = list(reversed(string))
        while stack:
            chars = []
            while stack:
                chars.append(stack.pop())
                glyph = self.glyphs.get("".join(chars))
                if glyph is None:
                    if len(chars) == 1:
                        glyph = self.missing_glyph
                    else:
                        stack.append(chars.pop())
                        glyph = self.glyphs.get("".join(chars))
                    break
            assert glyph is not None

            if glyph_prev is not None:
                kern = self.hkern.get((glyph_prev, glyph.unicode))
                if kern is not None:
                    offset -= kern

            output.append((offset, glyph))
            offset += glyph.advance
            glyph_prev = glyph.unicode
        return output, offset

    def str_to_path(self, size: float, string: str) -> Tuple[Path, float]:
        """Convert string to a `Path` using this font"""
        subpaths = []
        scale = size / self.units_per_em
        glyphs, offset = self.str_to_glyphs(string)
        for glyph_offset, glyph in glyphs:
            for glyph_path in glyph.path:
                subpath = []
                for cmd, args in glyph_path:
                    assert cmd != PATH_ARC
                    subpath.append(
                        (cmd, [[(x + glyph_offset) * scale, -y * scale] for x, y in args])
                    )
                subpaths.append(subpath)
        return Path(subpaths), offset * scale

    def names(self):
        return {glyph.name: glyph.unicode for glyph in self.glyphs.values()}

    def __repr__(self):
        return 'Font(family="{}", weight={}, style={}, glyphs_count={})'.format(
            self.family, self.weight, self.style, len(self.glyphs)
        )


FONTS_SANS = {"arial", "verdana"}
FONTS_SERIF = {"times new roman", "times", "georgia"}
FONTS_MONO = {"iosevka", "courier", "pragmatapro"}
FONT_STYLE_NORMAL = "normal"
FONT_SIZE = 12


class FontsDB:
    __slots__ = ["fonts", "fonts_files"]

    def __init__(self):
        self.fonts = {}
        self.fonts_files = []

    def register(self, font, alias=None):
        self.fonts.setdefault(font.family.lower(), []).append(font)
        if alias is not None and alias != font.family:
            self.fonts.setdefault(alias.lower(), []).append(font)

    def register_file(self, font):
        """Register SVG file containing fonts defintion

        This is used to lazy load fonts on first call to resolve
        """
        self.fonts_files.append(font)

    def resolve(self, family, weight=None, style=None):
        # load fonts from sources if any
        while self.fonts_files:
            source = self.fonts_files.pop()
            if not os.path.isfile(source):
                warnings.warn(f"failed to find fonts file: {source}")
                continue
            svg_scene_from_filepath(source, fonts=self)

        # find family
        family = "serif" if family is None else family.lower()
        matches = self.fonts.get(family)
        if matches is None:
            if "sans" in family or family in FONTS_SANS:
                family = "sans"
            elif "serif" in family or family in FONTS_SERIF:
                family = "serif"
            elif "mono" in family or family in FONTS_MONO:
                family = "monospace"
            else:
                family = "serif"
            matches = self.fonts.get(family, self.fonts.get("seif"))
        if matches is None:
            return None

        # find style
        style = style or FONT_STYLE_NORMAL
        matches_out = [match for match in matches if match.style == style]
        if not matches_out:
            matches_out = [match for match in matches if match.style == FONT_STYLE_NORMAL]
        if not matches_out:
            return None
        matches = matches_out

        # find weight
        weight = weight or 400
        matches = list(sorted(matches, key=lambda f: abs(f.weight - weight)))

        return matches[0]


# ------------------------------------------------------------------------------
# SVG
# ------------------------------------------------------------------------------
SVG_UNITS_USER = "userSpaceOnUse"
SVG_UNITS_BBOX = "objectBoundingBox"
COLOR_RE = re.compile("#?([0-9A-Fa-f]+)$")
COLOR_RGB_RE = re.compile(r"\s*(rgba?|hsl)\(([^\)]+)\)\s*")
TRANSFORM_RE = re.compile(r"\s*(translate|scale|rotate|skewX|skewY|matrix)\s*\(([^\)]+)\)\s*")
SVG_INHERIT = {
    "color": None,
    "fill": "black",
    "fill-rule": PATH_FILL_NONZERO,
    "fill-opacity": None,
    "stroke": None,
    "stroke-opacity": None,
    "stroke-width": "1",
    "stroke-linecap": STROKE_CAP_BUTT,
    "stroke-linejoin": STROKE_JOIN_MITER,
    "stroke-miterlimit": "4",
    "font-family": "serif",
    "font-size": "12",
    "font-weight": "400",
    "text-anchor": None,
}
# fmt: off
SVG_COLORS = {
"aliceblue": "#f0f8ff", "antiquewhite": "#faebd7", "aqua": "#00ffff",
"aquamarine": "#7fffd4","azure": "#f0ffff", "beige": "#f5f5dc",
"bisque": "#ffe4c4", "black": "#000000", "blanchedalmond": "#ffebcd",
"blue": "#0000ff", "blueviolet": "#8a2be2", "brown": "#a52a2a",
"burlywood": "#deb887", "cadetblue": "#5f9ea0", "chartreuse": "#7fff00",
"chocolate": "#d2691e", "coral": "#ff7f50", "cornflowerblue": "#6495ed",
"cornsilk": "#fff8dc", "crimson": "#dc143c", "cyan": "#00ffff",
"darkblue": "#00008b", "darkcyan": "#008b8b", "darkgoldenrod": "#b8860b",
"darkgray": "#a9a9a9", "darkgrey": "#a9a9a9", "darkgreen": "#006400",
"darkkhaki": "#bdb76b", "darkmagenta": "#8b008b", "darkolivegreen": "#556b2f",
"darkorange": "#ff8c00", "darkorchid": "#9932cc", "darkred": "#8b0000",
"darksalmon": "#e9967a", "darkseagreen": "#8fbc8f", "darkslateblue": "#483d8b",
"darkslategray": "#2f4f4f", "darkslategrey": "#2f4f4f",
"darkturquoise": "#00ced1", "darkviolet": "#9400d3", "deeppink": "#ff1493",
"deepskyblue": "#00bfff", "dimgray": "#696969", "dimgrey": "#696969",
"dodgerblue": "#1e90ff", "firebrick": "#b22222", "floralwhite": "#fffaf0",
"forestgreen": "#228b22", "fuchsia": "#ff00ff", "gainsboro": "#dcdcdc",
"ghostwhite": "#f8f8ff", "gold": "#ffd700", "goldenrod": "#daa520",
"gray": "#808080", "grey": "#808080", "green": "#008000",
"greenyellow": "#adff2f", "honeydew": "#f0fff0", "hotpink": "#ff69b4",
"indianred": "#cd5c5c", "indigo": "#4b0082", "ivory": "#fffff0",
"khaki": "#f0e68c", "lavender": "#e6e6fa", "lavenderblush": "#fff0f5",
"lawngreen": "#7cfc00", "lemonchiffon": "#fffacd", "lightblue": "#add8e6",
"lightcoral": "#f08080", "lightcyan": "#e0ffff",
"lightgoldenrodyellow": "#fafad2", "lightgray": "#d3d3d3",
"lightgrey": "#d3d3d3", "lightgreen": "#90ee90", "lightpink": "#ffb6c1",
"lightsalmon": "#ffa07a", "lightseagreen": "#20b2aa", "lightskyblue": "#87cefa",
"lightslategray": "#778899", "lightslategrey": "#778899",
"lightsteelblue": "#b0c4de", "lightyellow": "#ffffe0", "lime": "#00ff00",
"limegreen": "#32cd32", "linen": "#faf0e6", "magenta": "#ff00ff",
"maroon": "#800000", "mediumaquamarine": "#66cdaa", "mediumblue": "#0000cd",
"mediumorchid": "#ba55d3", "mediumpurple": "#9370db",
"mediumseagreen": "#3cb371", "mediumslateblue": "#7b68ee",
"mediumspringgreen": "#00fa9a", "mediumturquoise": "#48d1cc",
"mediumvioletred": "#c71585", "midnightblue": "#191970", "mintcream": "#f5fffa",
"mistyrose": "#ffe4e1", "moccasin": "#ffe4b5", "navajowhite": "#ffdead",
"navy": "#000080", "oldlace": "#fdf5e6", "olive": "#808000",
"olivedrab": "#6b8e23", "orange": "#ffa500", "orangered": "#ff4500",
"orchid": "#da70d6", "palegoldenrod": "#eee8aa", "palegreen": "#98fb98",
"paleturquoise": "#afeeee", "palevioletred": "#db7093", "papayawhip": "#ffefd5",
"peachpuff": "#ffdab9", "peru": "#cd853f", "pink": "#ffc0cb", "plum": "#dda0dd",
"powderblue": "#b0e0e6", "purple": "#800080", "rebeccapurple": "#663399",
"red": "#ff0000", "rosybrown": "#bc8f8f", "royalblue": "#4169e1",
"saddlebrown": "#8b4513", "salmon": "#fa8072", "sandybrown": "#f4a460",
"seagreen": "#2e8b57", "seashell": "#fff5ee", "sienna": "#a0522d",
"silver": "#c0c0c0", "skyblue": "#87ceeb", "slateblue": "#6a5acd",
"slategray": "#708090", "slategrey": "#708090", "snow": "#fffafa",
"springgreen": "#00ff7f", "steelblue": "#4682b4", "tan": "#d2b48c",
"teal": "#008080", "thistle": "#d8bfd8", "tomato": "#ff6347",
"turquoise": "#40e0d0", "violet": "#ee82ee", "wheat": "#f5deb3",
"white": "#ffffff", "whitesmoke": "#f5f5f5", "yellow": "#ffff00",
"yellowgreen": "#9acd32",
}
# fmt: on


def svg_scene(file, fg=None, width=None, fonts=None):
    """Load SVG scene from a file object"""
    fonts = FontsDB() if fonts is None else fonts

    def svg_scene_rec(element, inherit, top=False, width=None):
        tag = element.tag.split("}")[-1]
        attrs = svg_attrs(element.attrib, inherit)
        inherit = {k: v for k, v in attrs.items() if k in SVG_INHERIT}

        group = []
        if tag == "svg":
            for child in element:
                group.extend(svg_scene_rec(child, inherit))
            if not group:
                return group
            scene = Scene.group(group)

            # determine size and transform
            x = svg_size(attrs.get("x", "0"))
            y = svg_size(attrs.get("y", "0"))
            w = svg_size(attrs.get("width"))
            h = svg_size(attrs.get("height"))
            # override height
            viewbox = None
            if w is not None and h is not None:
                viewbox = [0, 0, w, h]
            if width is not None:
                if w is not None and h is not None:
                    w, h = width, int(width * h / w)
                else:
                    w, h = width, None
            # viewbox transform
            viewbox = svg_floats(attrs.get("viewBox"), 4, 4) or viewbox
            if viewbox is not None:
                transform = svg_viewbox_transform((x, y, w, h), viewbox)
                scene = scene.transform(transform)
                _vx, _vy, vw, vh = viewbox
                if h is None and w is None:
                    h, w = vh, vw
                elif h is None:
                    h = vh * w / vw
                elif w is None:
                    w = vw * h / vh
            elif x > 0 and y > 0:
                scene = scene.transform(Transform().translate(x, y))

            if w is not None and h is not None:
                if top:
                    nonlocal size
                    size = (w, h)
                else:
                    clip = [
                        (PATH_LINE, [[x, y], [x + w, y]]),
                        (PATH_LINE, [[x + w, y], [x + w, y + h]]),
                        (PATH_LINE, [[x + w, y + h], [x, y + h]]),
                        (PATH_CLOSED, [[x, y + h], [x, y]]),
                    ]
                    scene = scene.clip(Scene.fill(Path([clip]), np.ones(4)))

            group = [scene]

        elif tag == "path":
            group.extend(svg_path(attrs, ids, fg))

        elif tag == "g":
            for child in element:
                group.extend(svg_scene_rec(child, inherit))

        elif tag == "defs":
            for child in element:
                svg_scene_rec(child, inherit)

        elif tag in ("linearGradient", "radialGradient"):
            id = attrs.get("id")
            if id is not None:
                is_linear = tag == "linearGradient"
                ids[id] = svg_grad(element, None, is_linear)
            return []

        elif tag == "clipPath":
            id = attrs.get("id")
            inherit.setdefault("fill-rule", attrs.get("clip-rule"))
            if id is not None:
                for child in element:
                    group.extend(svg_scene_rec(child, inherit))
                if group:
                    scene, group = Scene.group(group), []
                    transform = svg_transform(attrs.get("transform"))
                    if transform is not None:
                        scene = scene.transform(transform)
                    bbox_units = attrs.get("clipPathUnits")
                    ids[id] = (scene, bbox_units == SVG_UNITS_BBOX)
            return []

        elif tag == "mask":
            id = attrs.get("id")
            if id is not None:
                for child in element:
                    group.extend(svg_scene_rec(child, inherit))
                scene, group = Scene.group(group), []
                transform = svg_transform(attrs.get("transform"))
                if transform is not None:
                    scene = scene.transform(transform)
                bbox_units = attrs.get("maskContentUnits")
                ids[id] = (scene, bbox_units == SVG_UNITS_BBOX)

        elif tag == "filter":
            id = attrs.get("id")
            if id is not None:
                ids[id] = svg_filter(attrs, element)

        elif tag == "pattern":
            id = attrs.get("id")
            if id is not None:
                x = svg_float(attrs.get("x", "0"))
                y = svg_float(attrs.get("y", "0"))
                width = svg_float(attrs.get("width"))
                height = svg_float(attrs.get("height"))
                if width is None or height is None:
                    return []

                for child in element:
                    group.extend(svg_scene_rec(child, inherit))
                scene, group = Scene.group(group), []
                scene_view_box = svg_floats(attrs.get("viewBox"), 4, 4)
                scene_bbox_units = (
                    attrs.get("patternContentUnits", SVG_UNITS_USER) == SVG_UNITS_BBOX
                )
                # view_box = svg_floats(attrs.get("viewBox"), 4, 4)
                # if view_box is not None:
                #    scene_transform = svg_viewbox_transform((x, y, width, height), view_box)
                #    scene = scene.transform(scene_transform)
                #    scene_bbox_units = False

                transform = svg_transform(attrs.get("patternTransform"))
                if transform is None:
                    transform = Transform()
                bbox_units = attrs.get("patternUnits", SVG_UNITS_BBOX) == SVG_UNITS_BBOX
                ids[id] = Pattern(
                    scene,
                    scene_bbox_units,
                    scene_view_box,
                    x,
                    y,
                    width,
                    height,
                    transform,
                    bbox_units,
                )

        # shapes
        elif tag == "rect":
            x = svg_size(attrs.pop("x", "0"))
            y = svg_size(attrs.pop("y", "0"))
            width = svg_size(attrs.pop("width"))
            height = svg_size(attrs.pop("height"))
            rx = svg_size(attrs.get("rx"))
            ry = svg_size(attrs.get("ry"))
            attrs["d"] = svg_rect_to_path(x, y, width, height, rx, ry)
            group.extend(svg_path(attrs, ids, fg))

        elif tag == "circle":
            cx = svg_size(attrs.pop("cx", "0"))
            cy = svg_size(attrs.pop("cy", "0"))
            r = svg_size(attrs.pop("r"))
            attrs["d"] = svg_ellipse_to_path(cx, cy, r, r)
            group.extend(svg_path(attrs, ids, fg))

        elif tag == "ellipse":
            cx = svg_size(attrs.pop("cx", "0"))
            cy = svg_size(attrs.pop("cy", "0"))
            rx = svg_size(attrs.pop("rx"))
            ry = svg_size(attrs.pop("ry"))
            attrs["d"] = svg_ellipse_to_path(cx, cy, rx, ry)
            group.extend(svg_path(attrs, ids, fg))

        elif tag == "polygon":
            points = attrs.pop("points")
            attrs["d"] = f"M{points}z"
            group.extend(svg_path(attrs, ids, fg))

        elif tag == "polyline":
            points = attrs.pop("points")
            attrs["d"] = f"M{points}"
            group.extend(svg_path(attrs, ids, fg))

        elif tag == "line":
            x1 = svg_size(attrs.pop("x1", "0"))
            y1 = svg_size(attrs.pop("y1", "0"))
            x2 = svg_size(attrs.pop("x2", "0"))
            y2 = svg_size(attrs.pop("y2", "0"))
            attrs["d"] = f"M{x1},{y1} {x2},{y2}"
            group.extend(svg_path(attrs, ids, fg))

        elif tag in ("title", "desc", "metadata"):
            return []

        elif tag == "font":
            font = svg_font(element)
            id = attrs.get("id")
            fonts.register(font, id)
            if id is not None:
                ids[id] = font
            return []

        elif tag == "text":
            group.extend(svg_text(element, attrs, fonts, ids, fg))

        elif tag == "use":
            x = attrs.get("x")
            y = attrs.get("y")
            if x is not None or y is not None:
                attrs["transform"] = attrs.get("transform", "") + f" translate({x}, {y})"
            href = attrs.get("href")
            if href is None:
                for key, value in attrs.items():
                    if key.endswith("}href"):
                        href = value
                        break
            if href and href.startswith("#"):
                item = ids.get(href[1:])
                if isinstance(item, Scene):
                    group.append(item)

        else:
            warnings.warn(f"unsupported element type: {tag}")

        if not group:
            return group

        filter_name = attrs.get("filter")
        if filter_name is not None:
            flt = svg_url(filter_name, ids)
            if not isinstance(flt, Filter):
                warnings.warn(f"not a filter referenced {filter_name}: {type(flt)}")
            else:
                group = [Scene.group(group).filter(flt)]

        opacity = svg_float(attrs.get("opacity"))
        if opacity is not None:
            # create isolated group if opacity is present
            group = [Scene.group(group).opacity(opacity)]

        clip_path = attrs.get("clip-path")
        if clip_path is not None:
            clip = svg_url(clip_path, ids)
            if clip is None or not isinstance(clip, tuple):
                warnings.warn(f"clip path expected {clip_path}: {type(clip)}")
            else:
                clip, bbox_units = clip
                group = [Scene.group(group).clip(clip, bbox_units)]

        mask_url = attrs.get("mask")
        if mask_url is not None:
            mask = svg_url(mask_url, ids)
            if mask is None or not isinstance(mask, tuple):
                warnings.warn(f"mask expected {mask_url}: {type(mask)}")
            else:
                mask, bbox_units = mask
                group = [Scene.group(group).mask(mask, bbox_units)]

        # transform mast go last (so clip and mask would be in transformed space)
        transform = svg_transform(attrs.get("transform"))
        if transform is not None:
            group = [scene.transform(transform) for scene in group]

        id = attrs.get("id")
        if id is not None:
            ids[id] = Scene.group(group)

        return group

    ids = {}
    size = None
    tree = etree.parse(file)
    root = tree.getroot()
    inherit = dict(color=np.array([0.0, 0.0, 0.0, 1.0]) if fg is None else fg)
    group = svg_scene_rec(root, inherit, True, width)
    if not group:
        return None, ids, size
    return Scene.group(group), ids, size


def svg_scene_from_filepath(path, fg=None, width=None, fonts=None):
    """Load SVG scene from a file at specified path"""
    _, ext = os.path.splitext(path)
    path = os.path.expanduser(path)
    if ext in {".gz", ".svgz"}:
        with gzip.open(path, mode="rt", encoding="utf-8") as file:
            return svg_scene(file, fg, width, fonts)
    else:
        with open(path, encoding="utf-8") as file:
            return svg_scene(file, fg, width, fonts)


def svg_scene_from_str(string, fg=None, width=None, fonts=None):
    """Load SVG scene from a string"""
    return svg_scene(io.StringIO(string), fg, width, fonts)


def svg_attrs(attrs, inherit=None):
    style = attrs.pop("style", None)
    if style is not None:
        for attr in style.split(";"):
            if not attr.strip():
                continue
            key, value = attr.split(":", 1)
            attrs[key.strip()] = value.strip()
    if inherit is not None:
        attrs = {**inherit, **attrs}
    return attrs


def svg_viewbox_transform(bbox, viewbox):
    """Convert svg view_box and x, y, width and height cooridate to transformation

    FIXME: default value for width and height is actually 100% so it should probably
           be handled differently
    """
    vx, vy, vw, vh = viewbox
    x, y, w, h = bbox
    if h is None and w is None:
        h, w = vh, vw
    elif h is None:
        h = vh * w / vw
    elif w is None:
        w = vw * h / vh
    scale = min(w / vw, h / vh)
    translate_x = -vx + (w / scale - vw) / 2 + x / scale
    translate_y = -vy + (h / scale - vh) / 2 + y / scale
    return Transform().scale(scale).translate(translate_x, translate_y)


def svg_path(attrs, ids, fg, path=None):
    """Create scene for SVG path from its attributes"""
    if path is None:
        path_str = attrs.get("d")
        if path_str is None:
            return []
        path = Path.from_svg(path_str)

    group = []
    fill = attrs.get("fill")
    if fill is not None:
        if fill == "currentColor":
            fill = attrs.get("color")
        else:
            fill = svg_paint(fill, ids)
    elif fg is not None:
        fill = fg
    else:
        fill = np.array([0, 0, 0, 1], dtype=FLOAT)
    fill_opacity = svg_float(attrs.get("fill-opacity"))
    fill_rule = attrs.get("fill-rule", PATH_FILL_NONZERO)
    if fill is not None:
        scene = Scene.fill(path, fill, fill_rule)
        if fill_opacity is not None:
            scene = scene.opacity(fill_opacity)
        group.append(scene)

    stroke = attrs.get("stroke")
    if stroke == "currentColor":
        stroke = attrs.get("color")
    else:
        stroke = svg_paint(stroke, ids)
    stroke_width = svg_float(attrs.get("stroke-width", "1"))
    stroke_opacity = svg_float(attrs.get("stroke-opacity"))
    stroke_linecap = attrs.get("stroke-linecap")
    stroke_linejoin = attrs.get("stroke-linejoin")
    if stroke is not None:
        scene = Scene.stroke(path, stroke, stroke_width, stroke_linecap, stroke_linejoin)
        if stroke_opacity is not None:
            scene = scene.opacity(stroke_opacity)
        group.append(scene)

    return group


def svg_grad(element, parent, is_linear):
    """Create gradient object from SVG element"""
    attr = element.attrib
    parent = {} if parent is None else parent._asdict()

    transform = attr.get("gradientTransform") or attr.get("transform")
    if transform is not None:
        transform = svg_transform(transform)
    else:
        transform = parent.get("transform")

    spread = attr.get("spreadMethod", parent.get("spread", "pad"))

    units = attr.get("gradientUnits", SVG_UNITS_BBOX)
    if units == SVG_UNITS_BBOX:
        bbox_units = True
    elif units == SVG_UNITS_USER:
        bbox_units = False
    else:
        raise ValueError(f"invalid gradient unites: {units}")

    # valid gradients should have at least two stops
    stops = svg_stops(element) or parent.get("stops")
    if not stops:
        # no stops mean, paint = "none"
        return None
    elif len(stops) == 1:
        # one stop mean, paint = "{color}"
        _offset, color = stops[0]
        return color

    color_int = attr.get("color-interpolation")
    if color_int == "linearRGB":
        linear_rgb = True
    elif color_int == "sRGB":
        linear_rgb = False
    else:
        linear_rgb = None

    if is_linear:
        x1 = svg_float(attr.get("x1", "0"))
        y1 = svg_float(attr.get("y1", "0"))
        x2 = svg_float(attr.get("x2", "1"))
        y2 = svg_float(attr.get("y2", "0"))
        p0 = np.array([x1, y1])
        p1 = np.array([x2, y2])

        return GradLinear(p0, p1, stops, transform, spread, bbox_units, linear_rgb)
    else:
        cx = svg_float(attr.get("cx", "0.5"))
        cy = svg_float(attr.get("cy", "0.5"))
        fx = svg_float(attr.get("fx"))
        fy = svg_float(attr.get("fy"))
        if fx is not None or fy is not None:
            if fx is None:
                fcenter = np.array([cx, fy])
            elif fy is None:
                fcenter = np.array([fx, cy])
            else:
                fcenter = np.array([fx, fy])
        else:
            fcenter = None
        center = np.array([cx, cy])
        radius = svg_float(attr.get("r")) or 0.5
        fradius = svg_float(attr.get("fr"))

        return GradRadial(
            center, radius, fcenter, fradius, stops, transform, spread, bbox_units, linear_rgb
        )


def svg_stops(element):
    stops = []
    for stop in element:
        attr = svg_attrs(stop.attrib)
        if not stop.tag.endswith("stop"):
            continue
        offset = svg_float(attr.get("offset")) or 0
        offset = 0 if offset < 0 else 1 if offset > 1 else offset
        color = svg_color(attr["stop-color"])
        if color is None:
            continue
        opacity = attr.get("stop-opacity")
        if opacity:
            color *= float(opacity)
        stops.append((offset, color))
    stops.sort(key=lambda s: s[0])
    return stops


def svg_filter(element_attrs, element):
    filter = Filter.empty()
    for child in element:
        tag = child.tag.split("}")[-1]
        attrs = child.attrib
        result = attrs.get("result")
        input = attrs.get("in")
        if tag == "feOffset":
            dx = svg_float(attrs.get("dx", "0"))
            dy = svg_float(attrs.get("dy", "0"))
            filter = filter.offset(dx, dy, input, result)
        elif tag == "feGaussianBlur":
            stds = svg_floats(attrs.get("stdDeviation"), 1, 2)
            if stds is not None:
                if len(stds) == 1:
                    stds *= 2
                std_x, std_y = stds
                filter = filter.blur(std_x, std_y, input, result)
        elif tag == "feMerge":
            names = []
            for node in child:
                node_tag = node.tag.split("}")[-1]
                if node_tag != "feMergeNode":
                    continue
                names.append(node.get("in"))
            filter = filter.merge(names, result)
        elif tag == "feBlend":
            mode = attrs.get("mode")
            filter = filter.blend(input, attrs.get("in2"), mode, result)
        elif tag == "feComposite":
            mode_name = attrs.get("operator", "over")
            if mode_name == "over":
                mode = COMPOSE_OVER
            elif mode_name == "in":
                mode = COMPOSE_IN
            elif mode_name == "out":
                mode = COMPOSE_OUT
            elif mode_name == "atop":
                mode = COMPOSE_ATOP
            elif mode_name == "xor":
                mode = COMPOSE_XOR
            elif mode_name == "arithmetic":
                k1 = svg_float(attrs.get("k1", "0"))
                k2 = svg_float(attrs.get("k2", "0"))
                k3 = svg_float(attrs.get("k3", "0"))
                k4 = svg_float(attrs.get("k4", "0"))
                mode = (k1, k2, k3, k4)
            else:
                warnings.warn(f"unsupported composite mode: {mode_name}")
                mode = COMPOSE_OVER
            filter = filter.composite(input, attrs.get("in2"), mode, result)
        elif tag == "feColorMatrix":
            type = attrs.get("type", "matrix")
            values = attrs.get("values")
            if type == "matrix":
                if values is None:
                    matrix = np.eye(4, 5)
                else:
                    matrix = np.array(svg_floats(values, 20, 20)).reshape(4, 5)
            elif type == "saturate":
                value = 1 if values is None else svg_float(values)
                matrix = color_matrix_saturate(value)
            elif type == "hueRotate":
                angle = 0 if values is None else svg_angle(values)
                matrix = color_matrix_hue_rotate(angle)
            elif type == "luminanceToAlpha":
                matrix = COLOR_MATRIX_LUM
            else:
                matrix = None
                warnings.warn(f"unsupported color matrix type: {type}")
            if matrix is not None:
                filter = filter.color_matrix(input, matrix, result)

        elif tag == "feMorphology":
            operator = attrs.get("operator", "erode")
            if operator == "erode":
                method = "min"
            elif operator == "dilate":
                method = "max"
            else:
                method = None
                warnings.warn(f"invalid morphology operator: {operator}")
            radius = svg_floats(attrs.get("radius", "0"), 1, 2)
            if len(radius) == 1:
                rx, ry = radius[0], radius[0]
            else:
                rx, ry = radius
            if method is not None and rx > 0 and ry > 0:
                filter = filter.morphology(rx, ry, method, input, result)
        else:
            warnings.warn(f"unsupported filter type: {tag}")
    return filter


def svg_rect_to_path(x, y, width, height, rx=None, ry=None):
    """Convert rect SVG element to path

    https://www.w3.org/TR/SVG/shapes.html#RectElement
    """
    if rx is None or ry is None:
        if rx is not None:
            rx, ry = rx, rx
        elif ry is not None:
            rx, ry = ry, ry
        else:
            rx, ry = 0, 0

    ops = []
    ops.append(f"M{x + rx:g},{y:g}")
    ops.append(f"H{x + width - rx:g}")
    if rx > 0 and ry > 0:
        ops.append(f"A{rx:g},{ry:g},0,0,1,{x + width:g},{y + ry:g}")
    ops.append(f"V{y + height - ry:g}")
    if rx > 0 and ry > 0:
        ops.append(f"A{rx:g},{ry:g},0,0,1,{x + width - rx:g},{y + height:g}")
    ops.append(f"H{x + rx:g}")
    if rx > 0 and ry > 0:
        ops.append(f"A{rx:g},{ry:g},0,0,1,{x:g},{y + height - ry:g}")
    ops.append(f"V{y + ry:g}")
    if rx > 0 and ry > 0:
        ops.append(f"A{rx:g},{ry:g},0,0,1,{x + rx:g},{y:g}")
    ops.append("z")
    return " ".join(ops)


def svg_ellipse_to_path(cx, cy, rx, ry):
    """Convert ellipse SVG element to path"""
    if rx is None or ry is None:
        if rx is not None:
            rx, ry = rx, rx
        elif ry is not None:
            rx, ry = ry, ry
        else:
            return ""

    ops = []
    ops.append(f"M{cx + rx:g},{cy:g}")
    ops.append(f"A{rx:g},{ry:g},0,0,1,{cx:g},{cy + ry:g}")
    ops.append(f"A{rx:g},{ry:g},0,0,1,{cx - rx:g},{cy:g}")
    ops.append(f"A{rx:g},{ry:g},0,0,1,{cx:g},{cy - ry:g}")
    ops.append(f"A{rx:g},{ry:g},0,0,1,{cx + rx:g},{cy:g}")
    ops.append("z")
    return " ".join(ops)


def svg_transform(input):
    """Parse SVG transform"""
    if input is None:
        return None

    def args_err(name, args_len, needs):
        raise ValueError(
            "`{}` transform requires {} arguments {} where given".format(name, args_len, needs)
        )

    tr = Transform()
    input = input.strip().replace(",", " ")
    while input:
        match = TRANSFORM_RE.match(input)
        if match is None:
            raise ValueError(f"failed to parse transform: {input}")
        input = input[len(match.group(0)) :]

        op, args = match.groups()
        args = list(filter(None, args.split(" ")))
        args_len = len(args)
        if op == "matrix":
            args = list(map(float, args))
            if args_len != 6:
                args_err("matrix", args_len, 6)
            a, b, c, d, e, f = args
            tr = tr.matrix(a, c, e, b, d, f)
        elif op == "translate":
            args = list(map(float, args))
            if args_len == 2:
                tx, ty = args
            elif args_len == 1:
                tx, ty = args[0], 0
            else:
                args_err("translate", args_len, "{1,2}")
            tr = tr.translate(tx, ty)
        elif op == "scale":
            args = list(map(float, args))
            if args_len == 2:
                sx, sy = args
            elif args_len == 1:
                sx, sy = args[0], args[0]
            else:
                args_err("scale", args_len, "{1,2}")
            tr = tr.scale(sx, sy)
        elif op == "rotate":
            if args_len == 1:
                tr = tr.rotate(svg_angle(args[0]))
            elif args_len == 3:
                a = svg_angle(args[0])
                x, y = list(map(float, args[1:]))
                tr = tr.translate(x, y).rotate(a).translate(-x, -y)
            else:
                args_err("rotate", args_len, "{1,3}")
        elif op == "skewX":
            if args_len != 1:
                args_err("skewX", args_len, 1)
            tr = tr.skew(svg_angle(args[0]), 0)
        elif op == "skewY":
            if args_len != 1:
                args_err("skewY", args_len, 1)
            tr = tr.skew(0, svg_angle(args[0]))
        else:
            raise ValueError(f"invalid transform operation: {op}")

    return tr


def svg_float(text):
    if isinstance(text, float):
        return text
    if text is None:
        return None
    text = text.strip()
    if text.endswith("%"):
        return float(text[:-1]) / 100.0
    elif text.endswith("px") or text.endswith("pt"):
        return float(text[:-2])
    else:
        return float(text)


def svg_floats(text, min=None, max=None):
    if text is None:
        return None
    floats = [float(v) for v in text.replace(",", " ").split(" ") if v]
    if min is not None and len(floats) < min:
        raise ValueError(f"expected at least {min} arguments")
    if max is not None and len(floats) > max:
        raise ValueError(f"expected at most {max} arguments")
    return floats


def svg_angle(angle):
    """Convert SVG angle to radians"""
    angle = angle.strip()
    if angle.endswith("deg"):
        return float(angle[:-3]) * math.pi / 180
    elif angle.endswith("rad"):
        return float(angle[:-3])
    return float(angle) * math.pi / 180


def svg_size(size, default=None, dpi=96):
    if size is None:
        return default
    if isinstance(size, (int, float)):
        return float(size)
    size = size.strip().lower()
    match = FLOAT_RE.match(size)
    if match is None:
        warnings.warn(f"invalid size: {size}")
        return default
    value = float(match.group(0))
    units = size[match.end() :].strip()
    if not units or units == "px":
        return value
    elif units == "in":
        return value * dpi
    elif units == "cm":
        return value * dpi / 2.54
    elif units == "mm":
        return value * dpi / 25.4
    elif units == "pt":
        return value * dpi / 72.0
    elif units == "pc":
        return value * dpi / 6.0
    elif units == "em":
        return value * FONT_SIZE
    elif units == "ex":
        return value * FONT_SIZE / 2.0
    elif units == "%":
        warnings.warn("size in % is not supported")
        return value


def svg_url(url, ids):
    """Resolve SVG url"""
    match = re.match(r"url\(\#([^)]+)\)", url.strip())
    if match is None:
        return None
    target = ids.get(match.group(1))
    if target is None:
        warnings.warn(f"failed to resolve url: {url}")
        return None
    return target


def svg_paint(paint, ids):
    """Resolve SVG paint"""
    if paint is None:
        return None
    paint = paint.strip()
    if paint == "none":
        return None
    obj = svg_url(paint, ids)
    if obj is not None:
        return obj
    color = svg_color(paint)
    if color is not None:
        return color
    warnings.warn(f"invalid paint: {paint}")
    return None


def svg_color(color_str):
    """Parse SVG color

    Returns color with premultiplied alpha in linear RGB colorspace
    """
    color = None
    match = COLOR_RE.match(color_str)
    if match is not None:
        rgb = match.group(1)
        if len(rgb) in (3, 4):
            color = np.array([int(c, 16) for c in rgb], FLOAT) / 15.0
        elif len(rgb) in (6, 8):
            color = np.array([int(c, 16) for c in chunk(rgb, 2)], FLOAT) / 255.0
        else:
            raise ValueError(f"invalid hex color: {color_str}")

    match = COLOR_RGB_RE.match(color_str)
    if match is not None:
        type, args = match.groups()
        if type.strip() in ("rgb", "rgba"):
            channels = []
            for channel in filter(None, args.replace(",", " ").split(" ")):
                if channel.endswith("%"):
                    channels.append(float(channel[:-1]) / 100)
                else:
                    channels.append(float(channel) / 255.0)
            color = np.array(channels)
        else:
            raise ValueError(f"invalid rgb color: {color_str}")

    if color is not None:
        if color.shape == (3,):
            color = np.array([*color, 1.0], dtype=FLOAT)
        # convert to linear RGB colorspace
        color = color_srgb_to_linear(color)
        # convert to premultiplied alpha
        color[:3] *= color[3:]
        return color

    rgb = SVG_COLORS.get(color_str.lower().strip())
    if rgb is None:
        warnings.warn(f"invalid svg color: {color_str}")
        return None
    return svg_color(rgb)


def svg_font(element):
    glyphs = {}
    glyphs_names = {}
    hkern = {}
    missing_glyph = None
    font = None
    for child in element:
        tag = child.tag.split("}")[-1]
        attrs = svg_attrs(child.attrib, element.attrib)

        if tag == "glyph":
            name = attrs.get("glyph-name")
            unicode = attrs.get("unicode")
            advance = attrs.get("horiz-adv-x")
            path = attrs.get("d", "")
            if unicode is None or advance is None:
                continue
            glyph = Glyph(unicode, float(advance), path, name)
            glyphs[unicode] = glyph
            if name is not None:
                glyphs_names[name] = glyph

        elif tag == "missing-glyph":
            path = attrs.get("d", "")
            advance = attrs.get("horiz-adv-x")
            missing_glyph = Glyph(None, float(advance), path, "missing-glyph")

        elif tag == "font-face":
            family = attrs.get("font-family", f"{id(element)}")
            weight = svg_font_weight(attrs.get("font-weight"))
            style = attrs.get("font-style", FONT_STYLE_NORMAL)
            units_per_em = float(attrs.get("units-per-em", "2048"))
            ascent = float(attrs.get("ascent", str(units_per_em)))
            descent = float(attrs.get("descent", "0"))
            font = Font(family, weight, style, ascent, descent, units_per_em, {}, None, {})

        elif tag == "hkern":
            left = []
            u1 = attrs.get("u1")
            if u1:
                left.extend(filter(None, u1.split(",")))
            g1 = attrs.get("g1")
            if g1:
                for name in filter(None, g1.split(",")):
                    glyph = glyphs_names.get(name)
                    if glyph is not None and glyph.unicode:
                        left.append(glyph.unicode)

            right = []
            u2 = attrs.get("u2")
            if u2:
                right.extend(filter(None, u2.split(",")))
            g2 = attrs.get("g2")
            if g2:
                for name in filter(None, g2.split(",")):
                    glyph = glyphs_names.get(name)
                    if glyph is not None and glyph.unicode:
                        right.append(glyph.unicode)

            k = attrs.get("k")
            if k is None:
                continue

            kern = float(k)
            for l in left:
                for r in right:
                    hkern[(l, r)] = kern

    if font is None:
        warnings.warn("font is missing `font-face` element")
        return None
    font.glyphs.update(glyphs)
    font.hkern.update(hkern)
    if missing_glyph is not None:
        font = font._replace(missing_glyph=missing_glyph)
    return font


def svg_font_weight(weight):
    if weight is None:
        return 400
    weight = weight.lower()
    if weight == "normal":
        return 400
    elif weight == "bold":
        return 700
    return int(float(weight))


def svg_text(element, attrs, fonts, ids, fg):
    def text_from_attrs(text, attrs, offset, space):
        # handle shfits/offsest even if there is nothing to render
        ox, oy = offset
        x = svg_size(attrs.pop("x", None))
        if x is not None:
            ox = x
        dx = svg_size(attrs.pop("dx", None))
        if dx is not None:
            ox += dx

        y = svg_size(attrs.pop("y", None))
        if y is not None:
            oy = y
        dy = svg_size(attrs.pop("dy", None))
        if dy is not None:
            oy += dy

        if not text:
            return [], (ox, oy), space
        prefix, suffix = "", ""
        text = text.replace("\n", " ")
        if text[0] in " \t" and len(text) > 1 and not space:
            prefix = " "
        if text[-1] in " \t":
            suffix = " "
        text = " ".join(filter(None, text.strip().split()))
        if not text:
            return [], (ox, oy), space
        text = prefix + text + suffix

        transform = Transform().translate(ox, oy)

        size = svg_float(attrs.get("font-size", f"{FONT_SIZE}"))
        font = fonts.resolve(attrs.get("font-family"), svg_font_weight(attrs.get("font-weight")))
        if font is None:
            return [], (ox, oy), space
        path, path_offset = font.str_to_path(size, text)

        output = []
        for scene in svg_path(attrs, ids, fg, path):
            output.append(scene.transform(transform))

        return output, (ox + path_offset, oy), bool(suffix)

    def text_from_element(element, attrs, offset, space):
        chunks = []
        chunk, offset, space = text_from_attrs(element.text, attrs, offset, space)
        chunks.extend(chunk)
        for child in element:
            tag = element.tag.split("}")[-1]
            if tag in {"text", "tspan"}:
                child_attrs = svg_attrs(child.attrib, attrs)
                chunk, offset, space = text_from_element(child, child_attrs, offset, space)
                chunks.extend(chunk)
            chunk, offset, space = text_from_attrs(child.tail, attrs, offset, space)
            chunks.extend(chunk)
        return chunks, offset, space

    start_x = svg_float(attrs.get("x", "0"))
    chunks, (end_x, _end_y), _space = text_from_element(element, attrs, (0, 0), True)

    anchor = attrs.get("text-anchor")
    anchor_tr = None
    if anchor == "middle":
        anchor_tr = Transform().translate((start_x - end_x) / 2, 0)
    elif anchor == "end":
        anchor_tr = Transform().translate(start_x - end_x, 0)
    if anchor_tr is not None:
        chunks = [chunk.transform(anchor_tr) for chunk in chunks]

    return chunks


# ------------------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------------------
DEFAULT_FONTS = os.path.join(os.path.dirname(os.path.realpath(__file__)), "fonts.svgz")


def main() -> int:
    import os
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("svg", help="input SVG file")
    parser.add_argument("output", help="output PNG file")
    parser.add_argument("-bg", type=svg_color, help="set default background color")
    parser.add_argument("-fg", type=svg_color, help="set default foreground color")
    parser.add_argument("-w", "--width", type=int, help="output width")
    parser.add_argument("-id", help="render single element with specified `id`")
    parser.add_argument(
        "-t", "--transform", type=svg_transform, help="apply additional transformation"
    )
    parser.add_argument("--linear-rgb", action="store_true", help="use linear RGB for rendering")
    parser.add_argument("--fonts", nargs="*", help="paths to SVG files containing all fonts")
    parser.add_argument("--as-path", action="store_true", help="render output as svg path")
    opts = parser.parse_args()

    if not os.path.exists(opts.svg):
        sys.stderr.write(f"[error] file does not exsits: {opts.svg}\n")
        return 1

    fonts = FontsDB()
    for font in opts.fonts or [DEFAULT_FONTS]:
        fonts.register_file(font)

    transform = Transform() if opts.as_path else Transform().matrix(0, 1, 0, 1, 0, 0)
    if opts.transform:
        transform @= opts.transform

    if opts.svg.endswith(".path"):
        path = Path.from_svg(open(opts.svg).read())
        opts.bg = svg_color("white") if opts.bg is None else opts.bg
        opts.fg = svg_color("black") if opts.fg is None else opts.fg
        scene = Scene.fill(path, opts.fg)

        ids, size = {}, None
    else:
        scene, ids, size = svg_scene_from_filepath(
            opts.svg, fg=opts.fg, width=opts.width, fonts=fonts
        )
    if scene is None:
        sys.stderr.write("[error] nothing to render\n")
        return 0

    if opts.id is not None:
        size = None
        scene = ids.get(opts.id)
        if scene is None:
            sys.stderr.write(f"[error] no object with id: {opts.id}\n")
            return 1

    if opts.as_path:
        with open(opts.output if opts.output != "-" else os.dup(1), "w") as file:
            file.write(scene.to_path(transform).to_svg())
        return 0

    start = time.time()
    if size is not None:
        w, h = size
        result = scene.render(
            transform, viewport=[0, 0, int(h), int(w)], linear_rgb=opts.linear_rgb
        )
    else:
        result = scene.render(transform, linear_rgb=opts.linear_rgb)
    stop = time.time()
    sys.stderr.write("[info] rendered in {:.2f}\n".format(stop - start))
    sys.stderr.flush()
    if result is None:
        sys.stderr.write("[error] nothing to render\n")
        return 1
    output, _convex_hull = result

    if size is not None:
        w, h = size
        output = output.convert(pre_alpha=True, linear_rgb=opts.linear_rgb)
        base = np.zeros((int(h), int(w), 4), dtype=FLOAT)
        image = canvas_merge_at(base, output.image, output.offset)
        output = Layer(image, (0, 0), pre_alpha=True, linear_rgb=opts.linear_rgb)

    if opts.bg is not None:
        output = output.background(opts.bg)

    with open(opts.output if opts.output != "-" else os.dup(1), "wb") as file:
        output.write_png(file)
    return 0


if __name__ == "__main__":
    sys.exit(main())
