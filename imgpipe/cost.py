"""代价模型:估计一个调度方案的耗时与内存访问量。

模型结构
--------
一个融合段按块 ``T`` 执行时, 耗时 ≈ Σ组 [计算时间 + 流量时间] + 调用开销。

* **计算时间**: 每组输出像素 × 该组算子的 ns/px 之和。ns/px 由
  :mod:`imgpipe.calibrate` 在真实 tile 上下文里实测 (含分配、裁边)。
* **流量时间**: 每组 (输入窗口 + 输出窗口) 的字节数 ÷ 工作集带宽。
  带宽随工作集 (tile 的活缓冲 + 池空闲块) 变化, 用标定的分段曲线插值。
* **调用开销**: 每个 kernel 调用 ~2–3 µs (scipy 稍贵); 块过多会让它抬头。
* **组内重叠**: 同时只能吃计算或吃带宽, 取它们的较大者为主项, 再加 40% 的
  次项 —— 标定后这个系数让预测与实测的相对误差最小。

绝对值不会完全准确, 但 **相对排序** 要和实测一致 —— 这是自动调度
(在若干候选方案里挑一个)要保证的。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .dag import OpSpec, Rect, WindowEstimator, rect_area, tile_rects
from .plan import (PIXEL_BYTES, Pipeline, Plan, Segment, group_windows,
                   stage_windows)

HERE = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# 硬件模型
# ---------------------------------------------------------------------------

@dataclass
class MachineModel:
    """标定得到的机器参数。默认值是 Apple M4 上的实测中位数。"""

    op_ns_per_px: Dict[str, float] = field(default_factory=lambda: {
        "color_matrix": 2.8, "exposure": 2.5, "saturation": 2.3, "levels": 2.7,
        "gamma": 13.7, "hist_eq": 21.8, "watermark": 7.9,
        "denoise": 23.7, "blur": 14.3, "sharpen": 15.3,
    })
    scale_ns_per_out_px: float = 22.0
    hist_ns_per_px: float = 11.0
    call_overhead_ns: float = 3000.0
    bandwidth_points: List[Tuple[float, float]] = field(default_factory=lambda: [
        (0.75e6, 54.0), (1.5e6, 43.0), (3e6, 52.0), (6e6, 59.0),
        (12e6, 59.0), (24e6, 52.0), (48e6, 52.0), (96e6, 49.0), (192e6, 50.0),
    ])
    overlap_secondary: float = 0.4
    # 全局缩放: 标定的算子 ns/px 是在"合成 tile"上测的, 实际流水线里各算子
    # 共享缓存与分配器, 综合可能略有偏差。默认 1.0 —— 自动选择只看相对排序,
    # 全局缩放对所有候选同等作用, 不影响选中谁; 需要校准绝对时间时再调。
    model_scale: float = 1.0
    # 分块局部性修正: 整宽条带的 (工作集 MB -> 系数) 曲线; 非整宽块统一系数
    band_locality: List[Tuple[float, float]] = field(default_factory=lambda: [
        (0.0, 0.84), (8.0, 0.885), (32.0, 0.92), (96.0, 0.96), (1e9, 0.97),
    ])
    nonband_locality: float = 1.06

    def bandwidth(self, working_set_bytes: float) -> float:
        pts = self.bandwidth_points
        if working_set_bytes <= pts[0][0]:
            return pts[0][1]
        for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
            if working_set_bytes <= x1:
                t = (working_set_bytes - x0) / (x1 - x0)
                return y0 + (y1 - y0) * t
        return pts[-1][1]

    def ns_per_px(self, op: OpSpec) -> float:
        v = self.op_ns_per_px.get(op.fn)
        if v is not None:
            return v
        # 未知算子: 按类型给个保守估计
        if op.kind == "point":
            return 6.0
        if op.kind == "stencil":
            return 4.0 * op.taps
        return 22.0


def _default_machine() -> MachineModel:
    m = MachineModel()
    path = os.path.join(HERE, "calibration.json")
    if os.path.exists(path):
        try:
            with open(path) as f:
                data = json.load(f)
            m.op_ns_per_px = dict(data.get("op_ns_per_px", m.op_ns_per_px))
            m.scale_ns_per_out_px = float(data.get("scale_ns_per_out_px",
                                                   m.scale_ns_per_out_px))
            m.hist_ns_per_px = float(data.get("hist_ns_per_px", m.hist_ns_per_px))
            if data.get("bandwidth_points"):
                m.bandwidth_points = [(float(a), float(b))
                                      for a, b in data["bandwidth_points"]]
            if data.get("call_overhead_ns"):
                m.call_overhead_ns = max(1500.0, float(data["call_overhead_ns"]))
            # 注意: calibration.json 里的 "tile_locality" 是微观原始比值 (噪声大),
            # 不直接覆盖模型曲线; 模型曲线用 acceptance 的流水线数据标定。
            if data.get("band_locality"):
                m.band_locality = [(float(a), float(b))
                                   for a, b in data["band_locality"]]
            if data.get("nonband_locality"):
                m.nonband_locality = float(data["nonband_locality"])
            if data.get("model_scale"):
                m.model_scale = float(data["model_scale"])
        except (ValueError, KeyError, TypeError):
            pass
    return m


MACHINE = _default_machine()


# ---------------------------------------------------------------------------
# 单段估算
# ---------------------------------------------------------------------------

@dataclass
class SegmentEstimate:
    seconds: float
    compute_seconds: float
    traffic_seconds: float
    overhead_seconds: float
    traffic_bytes: int
    computed_px: int
    unique_px: int
    working_set_bytes: int
    tiles: int


def estimate_segment(ops: Sequence[OpSpec], seg: Segment, tile: Optional[Tuple[int, int]],
                     machine: MachineModel = MACHINE) -> SegmentEstimate:
    if seg.kind == "stat":
        w, h = seg.in_dims
        px = w * h
        traffic = px * 3 * 4
        ws = min(px * PIXEL_BYTES, 64e6)
        sec = traffic / (machine.bandwidth(ws) * 1e9) + px * machine.hist_ns_per_px / 1e9
        return SegmentEstimate(sec, px * machine.hist_ns_per_px / 1e9,
                               traffic / (machine.bandwidth(ws) * 1e9),
                               0.0, traffic, 0, 0, int(ws), 1)

    seg_ops = [ops[i] for i in seg.op_ids]
    est = WindowEstimator(seg_ops, seg.in_dims)
    rects = tile_rects(seg.out_dims[0], seg.out_dims[1], tile)

    unique_px = 0
    pos = 0
    for g in seg.groups:
        pos += len(g.op_ids)
        unique_px += rect_area((0, 0) + est.dims[pos])

    computed_px = 0
    traffic = 0
    compute_ns = 0.0
    calls = 0
    max_group_in = 0
    max_group_out = 0

    for t in rects:
        sw = stage_windows(ops, seg.op_ids, seg.in_dims, t)
        gw = group_windows(ops, seg, sw)
        for gi, (a, b) in enumerate(gw):
            in_px = rect_area(a)
            out_px = rect_area(b)
            g = seg.groups[gi]
            traffic += (in_px + out_px) * PIXEL_BYTES
            computed_px += out_px
            max_group_in = max(max_group_in, in_px)
            max_group_out = max(max_group_out, out_px)
            for oid in g.op_ids:
                compute_ns += out_px * machine.ns_per_px(ops[oid])
                calls += 1

    # 工作集: 最坏一组的输入窗口 + 输出窗口 + 一个双缓冲
    working_set = (max_group_in + 2 * max_group_out) * PIXEL_BYTES
    bw = machine.bandwidth(working_set)
    traffic_sec = traffic / (bw * 1e9)
    compute_sec = compute_ns / 1e9
    overhead_sec = calls * machine.call_overhead_ns / 1e9

    # 局部性修正: 数据访问是行主序的。
    #  * 整宽的条带 (row band) —— 每一行都从头连续读到尾, 跨块也连续, 实测比
    #    "按纯计算量 + 流量"预测快 ~13%;
    #  * 方块 / 竖条 —— 每行只读一小段就跳到下一行, 行尾回退要重新取缓存行,
    #    实测比预测慢 ~5–9%。
    # 修正系数按候选池的实测中位数标定 (见 imgpipe/calibrate.py 的 tile 阶段)。
    loc = _locality_factor(seg, rects, tile, machine)
    primary = max(compute_sec, traffic_sec)
    secondary = min(compute_sec, traffic_sec)
    total = (primary + machine.overlap_secondary * secondary + overhead_sec) * loc
    total /= max(0.2, machine.model_scale)
    return SegmentEstimate(total, compute_sec, traffic_sec, overhead_sec,
                           traffic, computed_px, unique_px, int(working_set), len(rects))


def _locality_factor(seg: Segment, rects: List[Rect],
                     tile: Optional[Tuple[int, int]],
                     machine: MachineModel) -> float:
    """按块几何形状与工作集大小给耗时乘的局部性系数。

    系数来自候选池的实测中位数 (imgpipe/calibrate.py 的 tile 阶段, 见
    ``calibration.json`` 的 ``band_locality``)。规律:
      * 整宽条带 (行连续, 每行读完跳到下一行仍然连续) 比模型快;
      * 条带越矮, 工作集越能待在 L2/L3 里, 收益越大 (0.84 ~ 0.96);
      * 方块 / 竖条每行只读一小段, 行尾要回退缓存行, 比模型慢 (~1.06)。
    """
    if tile is None or not rects:
        return 1.0
    tw, th = tile
    canvas_w, canvas_h = seg.out_dims
    if tw >= canvas_w and th < canvas_h:
        # 按块高折算的工作集 (一组输入+输出+双缓冲)
        ws_mb = canvas_w * min(th, canvas_h) * PIXEL_BYTES * 3 / 1e6
        pts = machine.band_locality
        for (a0, v0), (a1, v1) in zip(pts, pts[1:]):
            if ws_mb <= a1:
                t = (ws_mb - a0) / (a1 - a0)
                return v0 + (v1 - v0) * t
        return pts[-1][1]
    return machine.nonband_locality


def estimate_plan(pipeline: Pipeline, plan: Plan, in_dims: Tuple[int, int],
                  machine: MachineModel = MACHINE) -> Dict[str, object]:
    from .runtime import _segments_for_plan, _tiles_for_segments
    segs = _segments_for_plan(pipeline, plan, in_dims)
    seg_tiles = _tiles_for_segments(segs, plan)
    total = compute = traffic_s = overhead = 0.0
    traffic = computed = unique = 0
    details = []
    for seg, t in zip(segs, seg_tiles):
        e = estimate_segment(pipeline.ops, seg, t, machine)
        total += e.seconds
        compute += e.compute_seconds
        traffic_s += e.traffic_seconds
        overhead += e.overhead_seconds
        traffic += e.traffic_bytes
        computed += e.computed_px
        unique += e.unique_px
        details.append({
            "kind": seg.kind,
            "in_dims": list(seg.in_dims),
            "out_dims": list(seg.out_dims),
            "tile": list(t) if t else None,
            "tiles": e.tiles,
            "seconds": e.seconds,
            "compute_seconds": e.compute_seconds,
            "traffic_seconds": e.traffic_seconds,
            "overhead_seconds": e.overhead_seconds,
            "traffic_bytes": e.traffic_bytes,
            "working_set_bytes": e.working_set_bytes,
            "computed_px": e.computed_px,
            "unique_px": e.unique_px,
        })
    return {
        "plan": plan.describe(),
        "seconds": total,
        "compute_seconds": compute,
        "traffic_seconds": traffic_s,
        "overhead_seconds": overhead,
        "traffic_bytes": traffic,
        "computed_px": computed,
        "unique_px": unique,
        "recompute": (computed / unique - 1.0) if unique else 0.0,
        "segments": details,
    }
