"""流水线、融合分段、调度方案与窗口推导。

概念
----
* **Pipeline** —— 有序的算子列表。
* **Segment** —— 融合单元。整图统计算子是强制断点: 之前的算子形成一个融合段,
  之后的算子形成下一个融合段。
* **Group** —— 段内的执行组: 相邻的逐像素算子合并成一个 :class:`Group`
  (一次遍历完成整条链, 中间结果不落内存), 邻域算子 / 缩放各自单独成组。
* **Plan** —— 调度方案: 是否融合 (``fuse``) + 每个融合段的块尺寸 (``tiles``)。
  ``fuse=False`` 时每个算子单独成组并物化整图 —— 即"逐个算子整图执行"基线。
* **StageWindows** —— 一个融合段、一个输出块所需要的每级输入/输出窗口。

窗口推导在本模块统一实现, 执行器和代价模型都调用它, 保证"推导的窗口 ==
执行的窗口"。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .dag import OpSpec, Rect, WindowEstimator, rect_area, scale_dim, validate

PIXEL_BYTES = 4 * 4          # float32 x RGBA


@dataclass
class Pipeline:
    name: str
    ops: List[OpSpec]
    note: str = ""

    def validate(self) -> None:
        if not self.ops:
            raise ValueError(f"流水线 {self.name} 没有任何算子")
        for op in self.ops:
            validate(op)
        # stat_idx 必须指向前面真实存在的 stat 算子
        for i, op in enumerate(self.ops):
            if op.fn == "hist_eq":
                j = int(op.params.get("stat_idx", -1))
                if not (0 <= j < i) or self.ops[j].kind != "stat":
                    raise ValueError(f"{op.name}: stat_idx 必须指向前面的 stat 算子")

    def op_index(self, name: str) -> int:
        for i, op in enumerate(self.ops):
            if op.name == name:
                return i
        raise KeyError(name)


@dataclass
class Group:
    kind: str                   # point | stencil | scale
    op_ids: List[int]

    @property
    def input_window_slack(self) -> Tuple[int, int]:
        return (0, 0)


@dataclass
class Segment:
    kind: str                   # "fused" | "stat"
    groups: List[Group]
    in_dims: Tuple[int, int] = (0, 0)
    out_dims: Tuple[int, int] = (0, 0)

    @property
    def op_ids(self) -> List[int]:
        return [i for g in self.groups for i in g.op_ids]


def _make_groups(ops: Sequence[OpSpec], ids: Sequence[int]) -> List[Group]:
    groups: List[Group] = []
    for i in ids:
        op = ops[i]
        if op.kind == "point" and groups and groups[-1].kind == "point":
            groups[-1].op_ids.append(i)
        else:
            groups.append(Group(op.kind, [i]))
    return groups


def build_segments(pipeline: Pipeline, in_dims: Tuple[int, int]) -> List[Segment]:
    """按整图统计断点切分, 段内合并相邻逐像素算子, 并标注画布尺寸。"""
    ops = pipeline.ops
    w, h = in_dims
    segs: List[Segment] = []
    cur: List[int] = []

    def flush() -> None:
        nonlocal w, h, cur
        if cur:
            ow, oh = _canvas_of(ops, cur, (w, h))
            segs.append(Segment("fused", _make_groups(ops, cur), (w, h), (ow, oh)))
            w, h = ow, oh
            cur = []

    for i, op in enumerate(ops):
        if op.kind == "stat":
            flush()
            segs.append(Segment("stat", [Group("stat", [i])], (w, h), (w, h)))
        else:
            cur.append(i)
    flush()
    return segs


def _canvas_of(ops: Sequence[OpSpec], ids: Sequence[int], dims: Tuple[int, int]) -> Tuple[int, int]:
    cw, ch = dims
    for i in ids:
        op = ops[i]
        if op.kind == "scale":
            cw = scale_dim(cw, op.scale[0])
            ch = scale_dim(ch, op.scale[1])
    return cw, ch


@dataclass
class Plan:
    """一个可执行的调度方案。``tiles`` 与融合段一一对应, None = 整图一块。"""

    name: str
    fuse: bool
    tiles: List[Optional[Tuple[int, int]]] = field(default_factory=list)
    note: str = ""

    def describe(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "fuse": self.fuse,
            "tiles": [list(t) if t else None for t in self.tiles],
            "note": self.note,
        }


@dataclass
class StageWindows:
    """一个融合段、一个输出块对应的全部窗口 (全局坐标, 各自画布)。"""

    w_in: List[Rect]
    w_out: List[Rect]
    tile_out: Rect

    @property
    def in_rect0(self) -> Rect:
        return self.w_in[0]


def _clip(r: Rect, dims: Tuple[int, int]) -> Rect:
    w, h = dims
    return (max(0, min(w, r[0])), max(0, min(h, r[1])),
            max(0, min(w, r[2])), max(0, min(h, r[3])))


def stage_windows(ops: Sequence[OpSpec], ids: Sequence[int],
                  in_dims: Tuple[int, int], tile_out: Rect) -> StageWindows:
    """从输出块倒推每一级需要的输入/输出窗口。

    ``w_out[k]`` 是算子 k 在它自己的输出画布上必须产出的窗口; 它向下游扩张的
    那一圈就是"按需重算的边缘"。``w_in[k] == w_out[k-1]``, 所以整段里每一级
    的计算量都可由这里精确算出。
    """
    est = WindowEstimator([ops[i] for i in ids], in_dims)
    n = len(ids)
    w_out: List[Rect] = [tile_out] * n
    r = tile_out
    for k in range(n - 1, -1, -1):
        r = _clip(r, est.dims[k + 1])
        w_out[k] = r
        r = est._single(k, r)
    w_in: List[Rect] = [r] + [w_out[k - 1] for k in range(1, n)]
    return StageWindows(w_in=w_in, w_out=w_out, tile_out=tile_out)


def group_windows(ops: Sequence[OpSpec], seg: Segment, sw: StageWindows) -> List[Tuple[Rect, Rect]]:
    """每组 (输入窗口, 输出窗口) —— 逐像素链只看首尾, 中间不落内存。"""
    pos = 0
    out: List[Tuple[Rect, Rect]] = []
    for g in seg.groups:
        n = len(g.op_ids)
        out.append((sw.w_in[pos], sw.w_out[pos + n - 1]))
        pos += n
    return out


@dataclass
class Workload:
    unique_px: int          # 理想情况下必须产出的像素 (每组输出画布之和)
    computed_px: int        # 实际计算出的输出像素 (含 halo 重算)
    tiles: int
    traffic_bytes: int      # 估算的读写字节数 (含重算)
    group_px: List[int]     # 每组理想像素数

    @property
    def recompute_px(self) -> int:
        return self.computed_px - self.unique_px

    @property
    def recompute(self) -> float:
        return (self.computed_px / self.unique_px - 1.0) if self.unique_px else 0.0


def segment_workload(ops: Sequence[OpSpec], seg: Segment, tiles: List[Rect]) -> Workload:
    """一个融合段按给定分块执行时的像素/流量工作量。"""
    est = WindowEstimator([ops[i] for i in seg.op_ids], seg.in_dims)
    # 每组理想输出像素 = 该组最后一个算子的输出画布面积
    group_px: List[int] = []
    pos = 0
    for g in seg.groups:
        pos += len(g.op_ids)
        group_px.append(rect_area((0, 0) + est.dims[pos]))
    unique = sum(group_px)
    computed = 0
    traffic = 0
    for t in tiles:
        sw = stage_windows(ops, seg.op_ids, seg.in_dims, t)
        for a, b in group_windows(ops, seg, sw):
            ai = rect_area(a)
            ao = rect_area(b)
            computed += ao
            traffic += (ai + ao) * PIXEL_BYTES
    # 整段输出的最终写出 (tile -> 整图) 已经算在最后一级的流量之外:
    # group_windows 的最后一组输出窗口就是 tile 输出, 写出到整图另算一次。
    traffic += sum(rect_area(t) for t in tiles) * PIXEL_BYTES
    return Workload(unique, computed, len(tiles), traffic, group_px)
