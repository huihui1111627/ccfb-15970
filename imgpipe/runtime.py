"""执行引擎:把一条流水线按调度方案跑成结果图。

执行方式 (由 :class:`Plan` 表达)
--------------------------------
``逐算子整图``  (fuse=False)
    每个算子单独一次整图读写, 中间结果物化成整图。

``融合``  (fuse=True, tiles=None)
    融合段内不产生中间图。相邻逐像素算子合并成组: 组内用一个 tile 缓冲
    ping-pong 推进, 组与组之间也只传 tile; 邻域算子 / 缩放的上游按需多算
    一圈 halo, 算完裁掉。

``融合 + 分块``  (fuse=True, tiles=(tw, th))
    融合段再按块调度, 每块独立走完整段, 工作集只有 tile 大小。

执行器保证: 同一份 kernel 在所有方式下输出一致 —— 每个像素的计算路径与分块
方式无关 (采样、边界、水印都用全局坐标)。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import kernels as K
from .dag import OpSpec, Rect, WindowEstimator, scale_dim, tile_rects
from .plan import (Group, Pipeline, Plan, Segment, build_segments,
                   group_windows, segment_workload, stage_windows)

CHANNELS = K.CHANNELS


class BufferPool:
    """tile 级临时缓冲池。

    池按"形状签名"复用; 同一形状的缓冲只在上一块用完后才会被再次借出,
    因此峰值内存 ≈ 单块工作集 (与块数无关)。用 ``__array_interface__`` 之外
    的引用计数不可靠, 这里显式借还。
    """

    def __init__(self) -> None:
        self._free: Dict[Tuple[int, ...], List[np.ndarray]] = {}
        self._live: List[np.ndarray] = []
        self.peak_bytes = 0
        self.alloc_bytes = 0

    def acquire(self, shape: Tuple[int, ...]) -> np.ndarray:
        lst = self._free.get(shape)
        if lst:
            arr = lst.pop()
        else:
            arr = np.empty(shape, dtype=np.float32)
            self.alloc_bytes += arr.nbytes
        self._live.append(arr)
        self.peak_bytes = max(self.peak_bytes, sum(a.nbytes for a in self._live))
        return arr

    def release(self, arr: np.ndarray) -> None:
        for i, a in enumerate(self._live):
            if a is arr:
                del self._live[i]
                self._free.setdefault(arr.shape, []).append(arr)
                return

    def reset(self) -> None:
        self._live.clear()
        self._free.clear()


@dataclass
class ExecResult:
    image: np.ndarray
    segments: List[Segment]
    tiles_per_segment: List[Optional[Tuple[int, int]]]
    kernel_calls: int = 0
    computed_px: int = 0
    unique_px: int = 0
    traffic_bytes: int = 0
    stat_px: int = 0
    full_buffers: int = 0                 # 写出的整图画布个数 (输入之外)
    pool_peak_bytes: int = 0
    intermediates: List[np.ndarray] = field(default_factory=list)
    segment_info: List[Dict[str, object]] = field(default_factory=list)

    @property
    def recompute_px(self) -> int:
        return max(0, self.computed_px - self.unique_px)

    @property
    def recompute(self) -> float:
        return (self.computed_px / self.unique_px - 1.0) if self.unique_px else 0.0

    @property
    def n_tiles(self) -> int:
        total = 0
        for seg, t in zip(self.segments, self.tiles_per_segment):
            if seg.kind == "fused":
                total += len(tile_rects(seg.out_dims[0], seg.out_dims[1], t))
        return total


def _segments_for_plan(pipeline: Pipeline, plan: Plan,
                       in_dims: Tuple[int, int]) -> List[Segment]:
    if plan.fuse:
        return build_segments(pipeline, in_dims)
    ops = pipeline.ops
    w, h = in_dims
    segs: List[Segment] = []
    for i, op in enumerate(ops):
        if op.kind == "scale":
            ow, oh = scale_dim(w, op.scale[0]), scale_dim(h, op.scale[1])
        else:
            ow, oh = w, h
        if op.kind == "stat":
            segs.append(Segment("stat", [Group("stat", [i])], (w, h), (w, h)))
        else:
            segs.append(Segment("fused", [Group(op.kind, [i])], (w, h), (ow, oh)))
            w, h = ow, oh
    return segs


def _tiles_for_segments(segs: List[Segment], plan: Plan) -> List[Optional[Tuple[int, int]]]:
    tiles = plan.tiles or [None]
    out: List[Optional[Tuple[int, int]]] = []
    fi = 0
    for seg in segs:
        if seg.kind == "stat":
            out.append(None)
        else:
            out.append(tiles[fi] if fi < len(tiles) else tiles[-1])
            fi += 1
    return out


def stat_tile(dims: Tuple[int, int]) -> Optional[Tuple[int, int]]:
    return (512, 512) if max(dims) > 640 else None


def _collect_hist(src: np.ndarray, dims: Tuple[int, int],
                  tile: Optional[Tuple[int, int]]) -> np.ndarray:
    hist = np.zeros(256, dtype=np.int64)
    w, h = dims
    for r in tile_rects(w, h, tile):
        K.accumulate_hist(hist, src[:, r[1]:r[3], r[0]:r[2]])
    return hist


def execute(pipeline: Pipeline, image: np.ndarray, plan: Plan,
            retain_intermediates: bool = False) -> ExecResult:
    """执行整条流水线。``image``: (4, H, W) float32。"""
    pipeline.validate()
    if image.ndim != 3 or image.shape[0] != CHANNELS:
        raise ValueError("输入必须是 (4, H, W)")
    _, h, w = image.shape
    in_dims = (w, h)
    ops = pipeline.ops
    segs = _segments_for_plan(pipeline, plan, in_dims)
    seg_tiles = _tiles_for_segments(segs, plan)

    luts: Dict[int, np.ndarray] = {}
    res = ExecResult(image=image, segments=segs, tiles_per_segment=seg_tiles)
    src = image

    for gi, seg in enumerate(segs):
        if seg.kind == "stat":
            oid = seg.groups[0].op_ids[0]
            op = ops[oid]
            hist = _collect_hist(src, seg.in_dims, stat_tile(seg.in_dims))
            luts[oid] = K.lut_from_hist(
                hist, float(op.params.get("strength", 1.0)),
                float(op.params.get("target_mean", 0.5)))
            px = seg.in_dims[0] * seg.in_dims[1]
            res.stat_px += px
            res.traffic_bytes += px * 3 * 4
            continue

        out = np.empty((CHANNELS, seg.out_dims[1], seg.out_dims[0]), dtype=np.float32)
        res.full_buffers += 1
        tile = seg_tiles[gi]
        tiles = tile_rects(seg.out_dims[0], seg.out_dims[1], tile)
        pool = BufferPool()
        est = WindowEstimator([ops[i] for i in seg.op_ids], seg.in_dims)

        for rect in tiles:
            # 最后一级直接写进输出图的视图, 少一次 tile -> 整图的拷贝
            final = out[:, rect[1]:rect[3], rect[0]:rect[2]]
            buf = _run_tile(ops, seg, rect, src, est, luts, pool, res, final=final)
            if buf is not final:
                np.copyto(final, buf)
            pool.release(buf)

        res.pool_peak_bytes = max(res.pool_peak_bytes, pool.peak_bytes)
        if retain_intermediates and gi + 1 < len(segs):
            res.intermediates.append(out)
        src = out

        work = segment_workload(ops, seg, tiles)
        res.computed_px += work.computed_px
        res.unique_px += work.unique_px
        res.traffic_bytes += work.traffic_bytes
        res.segment_info.append({
            "index": gi,
            "in_dims": list(seg.in_dims),
            "out_dims": list(seg.out_dims),
            "tile": list(tile) if tile else None,
            "n_tiles": len(tiles),
            "groups": [{"kind": g.kind, "ops": g.op_ids,
                        "names": [ops[i].short() for i in g.op_ids]}
                       for g in seg.groups],
            "unique_px": work.unique_px,
            "computed_px": work.computed_px,
            "recompute": (work.computed_px / work.unique_px - 1.0) if work.unique_px else 0.0,
            "traffic_bytes": work.traffic_bytes,
        })

    res.image = src
    return res


def _run_tile(ops: Sequence[OpSpec], seg: Segment, rect: Rect, src: np.ndarray,
              est: WindowEstimator, luts: Dict[int, np.ndarray],
              pool: BufferPool, res: ExecResult,
              final: Optional[np.ndarray] = None) -> np.ndarray:
    """一个融合段在一个输出块上的完整执行。

    缓冲流转: ``buf`` 是当前可读的输入, ``owned`` 标记它是否来自池。
    逐像素链在"输入 + 一块 spare"之间 ping-pong —— 链中间的中间结果从不
    离开 tile 缓冲区, 这就是"逐像素融合不产生中间图"的落地。
    """
    sw = stage_windows(ops, seg.op_ids, seg.in_dims, rect)
    gw = group_windows(ops, seg, sw)

    r0 = sw.in_rect0
    buf = src[:, r0[1]:r0[3], r0[0]:r0[2]]
    owned: Optional[np.ndarray] = None
    pos = 0

    for gi, g in enumerate(seg.groups):
        g_in, g_out = gw[gi]
        ctx = K.Ctx(in_rect=g_in, out_rect=g_out,
                    in_dims=est.dims[pos], out_dims=est.dims[pos + len(g.op_ids)],
                    luts=luts)
        shape = (CHANNELS, ctx.oh, ctx.ow)
        if g.kind == "point":
            n = len(g.op_ids)
            if n == 1:
                op = ops[g.op_ids[0]]
                if gi == len(seg.groups) - 1 and final is not None:
                    # 最后一级直接写最终输出视图, 省一次 tile -> 整图拷贝
                    K.POINT_KERNELS[op.fn](ctx, op.params, buf, out=final)
                    res.kernel_calls += 1
                    if owned is not None:
                        pool.release(owned)
                    return final
                dst = pool.acquire(shape)
                K.POINT_KERNELS[op.fn](ctx, op.params, buf, out=dst)
                res.kernel_calls += 1
                if owned is not None:
                    pool.release(owned)
                buf, owned = dst, dst
            else:
                # 逐像素链只在两块 tile 缓冲之间轮换: 链上一个算子的输出写进
                # 另一块, 链中间的图素从不落盘成"整图大小的中间结果"。
                if owned is None:
                    p1 = pool.acquire(shape)
                    p2 = pool.acquire(shape)
                else:
                    p1 = owned
                    p2 = pool.acquire(shape)
                cur, nxt = buf, (p2 if p1 is owned else p1)
                # 第一级必须写进 p1 (它可能是 owned, 也可能是新借的), 所以
                # 先确认 p1 与 cur 不同。
                if p1 is cur:
                    p1, p2 = p2, p1
                last = len(g.op_ids) - 1
                for j, oid in enumerate(g.op_ids):
                    op = ops[oid]
                    if j == last and gi == len(seg.groups) - 1 and final is not None:
                        K.POINT_KERNELS[op.fn](ctx, op.params, cur, out=final)
                        res.kernel_calls += 1
                        pool.release(cur) if cur is p1 or cur is p2 else None
                        if owned is not None and owned is not cur:
                            pool.release(owned)
                        return final
                    K.POINT_KERNELS[op.fn](ctx, op.params, cur, out=nxt)
                    res.kernel_calls += 1
                    cur, nxt = nxt, (p2 if nxt is p1 else p1)
                if nxt is not cur:
                    pool.release(nxt)
                buf, owned = cur, cur
            pos += n

        elif g.kind == "stencil":
            op = ops[g.op_ids[0]]
            out = K.STENCIL_KERNELS[op.fn](ctx, op.params, buf)
            res.kernel_calls += 1
            if owned is not None:
                pool.release(owned)
            buf, owned = out, None
            pos += 1
        elif g.kind == "scale":
            op = ops[g.op_ids[0]]
            out = K.bilinear(buf, ctx.in_rect, ctx.out_rect, est.dims[pos],
                             op.scale[0], op.scale[1])
            res.kernel_calls += 1
            if owned is not None:
                pool.release(owned)
            buf, owned = out, None
            pos += 1
        else:  # pragma: no cover
            raise RuntimeError(f"stat 算子不应出现在融合段: {g.kind}")

    if owned is not None:
        return owned
    return np.array(buf, copy=True)
