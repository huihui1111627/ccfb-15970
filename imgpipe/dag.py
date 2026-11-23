"""算子描述与窗口推导。

一个算子声明它输出的每个像素依赖 *输入* 上的哪个矩形邻域:

* ``radius``: 每个输入平面上的访问半径 (x, y); 逐像素算子为 (0, 0);
* ``scale``: 改变分辨率时的缩放因子; 输出像素 ``j`` 采样输入坐标 ``j / f``
  (双线性, 全图坐标系), 因此依赖范围由 ``scale_src_range`` 精确给出;
* ``global_stat``: 需要整幅输入图的统计量, 是融合链的强制断点。

所有窗口都是半开区间 ``(x0, y0, x1, y1)``, 坐标定义在算子 *输入* 网格的
全局坐标系上。窗口推导和实际执行共用同一份取整/裁剪规则 —— 这是"分块执行
与整图执行逐像素一致"的前提。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

Rect = Tuple[int, int, int, int]


@dataclass(frozen=True)
class OpSpec:
    name: str
    fn: str
    kind: str                                # point | stencil | scale | stat
    radius: Tuple[int, int] = (0, 0)
    scale: Optional[Tuple[float, float]] = None
    global_stat: bool = False
    params: Dict[str, Any] = field(default_factory=dict)
    label: str = ""

    @property
    def elementwise(self) -> bool:
        return self.kind == "point"

    @property
    def separable(self) -> bool:
        return self.kind == "stencil" and bool(self.params.get("separable", False))

    @property
    def taps(self) -> int:
        if self.kind != "stencil":
            return 1
        if self.separable:
            return 2 * (2 * self.radius[0] + 1)
        return (2 * self.radius[0] + 1) * (2 * self.radius[1] + 1)

    def short(self) -> str:
        return self.label or self.name


def point_op(name: str, fn: str, label: str = "", **params: Any) -> OpSpec:
    return OpSpec(name=name, fn=fn, kind="point", params=dict(params), label=label)


def stencil_op(name: str, fn: str, radius: Tuple[int, int], label: str = "",
               **params: Any) -> OpSpec:
    return OpSpec(name=name, fn=fn, kind="stencil", radius=tuple(radius),
                  params=dict(params), label=label)


def scale_op(name: str, fn: str, fx: float, fy: float, label: str = "",
             **params: Any) -> OpSpec:
    return OpSpec(name=name, fn=fn, kind="scale", scale=(float(fx), float(fy)),
                  params=dict(params), label=label)


def stat_op(name: str, fn: str, label: str = "", **params: Any) -> OpSpec:
    return OpSpec(name=name, fn=fn, kind="stat", global_stat=True,
                  params=dict(params), label=label)


def validate(op: OpSpec) -> None:
    if op.kind not in ("point", "stencil", "scale", "stat"):
        raise ValueError(f"未知算子类型 {op.kind!r} ({op.name})")
    if op.kind == "scale":
        if op.scale is None or op.scale[0] <= 0 or op.scale[1] <= 0:
            raise ValueError(f"缩放算子 {op.name} 需要正缩放因子")
    if op.kind == "stencil" and (op.radius[0] < 0 or op.radius[1] < 0):
        raise ValueError(f"邻域算子 {op.name} 半径为负")
    if op.global_stat != (op.kind == "stat"):
        raise ValueError(f"{op.name}: global_stat 只能用于 stat 算子")


def scale_dim(n: int, factor: float) -> int:
    return max(1, int(np.floor(n * factor + 0.5)))


def canvas_size(ops: Sequence[OpSpec], w: int, h: int) -> Tuple[int, int]:
    cw, ch = w, h
    for op in ops:
        if op.kind == "scale":
            cw = scale_dim(cw, op.scale[0])
            ch = scale_dim(ch, op.scale[1])
    return cw, ch


def scale_src_range(dst0: int, dst1: int, factor: float, src_n: int) -> Tuple[int, int]:
    """输出像素区间 [dst0, dst1) 的双线性采样需要输入 [lo, hi)。

    采样坐标 ``g = (j + 0.5) / f - 0.5``, 裁剪到 [0, n-1], 读取 floor(g) 与
    floor(g)+1 (裁剪到 n-1)。这里返回精确覆盖该集合的最小窗口。
    """
    if dst1 <= dst0 or src_n <= 0:
        return (0, 0)
    if src_n == 1:
        return (0, 1)
    hi_max = float(src_n - 1)
    g_lo = min(hi_max, max(0.0, (dst0 + 0.5) / factor - 0.5))
    g_hi = min(hi_max, max(0.0, (dst1 - 0.5) / factor - 0.5))
    if g_hi < g_lo:
        g_hi = g_lo
    lo = int(np.floor(g_lo))
    hi = int(np.floor(g_hi)) + 2
    lo = max(0, min(src_n, lo))
    hi = max(lo, min(src_n, hi))
    return lo, hi


def _clip(rect: Rect, w: int, h: int) -> Rect:
    x0, y0, x1, y1 = rect
    return (max(0, min(w, x0)), max(0, min(h, y0)),
            max(0, min(w, x1)), max(0, min(h, y1)))


class WindowEstimator:
    """对一段 *连续的* 算子推导各级窗口。

    ``in_dims`` 是这段算子输入画布的 ``(w, h)``。``dims[i]`` 是第 i 个算子
    输入的画布尺寸, ``dims[-1]`` 是这段输出的画布尺寸。
    """

    def __init__(self, ops: Sequence[OpSpec], in_dims: Tuple[int, int]):
        self.ops = list(ops)
        self.in_dims = (int(in_dims[0]), int(in_dims[1]))
        dims: List[Tuple[int, int]] = [self.in_dims]
        w, h = self.in_dims
        for op in self.ops:
            if op.kind == "scale":
                w, h = scale_dim(w, op.scale[0]), scale_dim(h, op.scale[1])
            dims.append((w, h))
        self.dims = dims

    def _single(self, idx: int, rect: Rect) -> Rect:
        op = self.ops[idx]
        w, h = self.dims[idx]
        x0, y0, x1, y1 = _clip(rect, *self.dims[idx + 1])
        if op.kind == "scale":
            lo_x, hi_x = scale_src_range(x0, x1, op.scale[0], w)
            lo_y, hi_y = scale_src_range(y0, y1, op.scale[1], h)
            return lo_x, lo_y, hi_x, hi_y
        rx, ry = op.radius
        return _clip((x0 - rx, y0 - ry, x1 + rx, y1 + ry), w, h)

    def input_rect(self, rect: Rect) -> Rect:
        r = rect
        for i in range(len(self.ops) - 1, -1, -1):
            r = self._single(i, r)
        return r

    def intermediates(self, rect: Rect) -> List[Rect]:
        """从输出窗口倒推每级 *输出* 窗口, 长度 = len(ops)。"""
        out: List[Optional[Rect]] = [None] * len(self.ops)
        r = rect
        for i in range(len(self.ops) - 1, -1, -1):
            out[i] = _clip(r, *self.dims[i + 1])
            r = self._single(i, r)
        return out  # type: ignore[return-value]

    def halo(self) -> Tuple[int, int]:
        """整段链的总邻域半径 (不含缩放重采样半径), 用于展示。"""
        hx = hy = 0
        for op in self.ops:
            if op.kind == "stencil":
                hx += op.radius[0]
                hy += op.radius[1]
        return hx, hy


def tile_rects(w: int, h: int, tile: Optional[Tuple[int, int]]) -> List[Rect]:
    """把 w x h 画布切成块; tile=None 表示整图一块。"""
    if tile is None:
        return [(0, 0, w, h)]
    tw, th = int(tile[0]), int(tile[1])
    if tw >= w and th >= h:
        return [(0, 0, w, h)]
    tw, th = max(1, tw), max(1, th)
    return [(x, y, min(x + tw, w), min(y + th, h))
            for y in range(0, h, th) for x in range(0, w, tw)]


def rect_area(r: Rect) -> int:
    return max(0, r[2] - r[0]) * max(0, r[3] - r[1])
