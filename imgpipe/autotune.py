"""自动调度: 枚举候选方案, 用代价模型挑一个。

候选空间 (与验收脚本的候选池一致, 这样"自动选择 vs 实测最优"是在同一空间里比)
------------------------------------------------------------------------------
* 逐算子整图 (基线, 不参与选择)
* 融合, 整图一块
* 融合 + 长条块: 整宽 × N 行, N ∈ {32, 64, 96, 128, 192, 256, 384, 512}
* 融合 + 方块: S × S, S ∈ {128, 256, 384, 512}
* 融合 + 竖条: 256 × N

长条块之所以重要: 逐像素链和可分离卷积都是行主序访问, 整宽的条带在 L2 里
连续; 方块和竖条则要跨行跳读。候选里两者都留, 让代价模型按流水线决定。

每段的块尺寸沿着同一条"目标工作集"走 —— 多段画布尺寸不同, 但按目标字节数
各自折算, 覆盖了主要收益又不至于组合爆炸。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from .cost import MACHINE, MachineModel, estimate_plan
from .plan import Pipeline, Plan, build_segments

Tile = Tuple[int, int]

BAND_ROWS = (32, 64, 96, 128, 192, 256, 384, 512)
SQUARE_SIDES = (128, 256, 384, 512)
VBAR = ((256, 64), (256, 128), (256, 256))


def _fused_seg_dims(pipeline: Pipeline, in_dims: Tuple[int, int]) -> List[Tuple[int, int]]:
    return [s.out_dims for s in build_segments(pipeline, in_dims) if s.kind == "fused"]


def candidate_plans(pipeline: Pipeline, in_dims: Tuple[int, int]) -> List[Tuple[str, Plan]]:
    """候选池: (短名字, Plan)。自动选择与实测最优共用。"""
    segs = _fused_seg_dims(pipeline, in_dims)
    out: List[Tuple[str, Plan]] = [
        ("unfused", Plan("逐算子整图", fuse=False)),
        ("fused", Plan("融合-整图", fuse=True)),
    ]
    for rows in BAND_ROWS:
        tiles = [(sw, min(rows, sh)) for sw, sh in segs]
        out.append((f"band{rows}", Plan(f"融合-整宽 {rows}行", fuse=True, tiles=tiles)))
    for side in SQUARE_SIDES:
        tiles = [(min(side, sw), min(side, sh)) for sw, sh in segs]
        out.append((f"sq{side}", Plan(f"融合-方块 {side}", fuse=True, tiles=tiles)))
    for bw, bh in VBAR:
        tiles = [(min(bw, sw), min(bh, sh)) for sw, sh in segs]
        out.append((f"vb{bw}x{bh}", Plan(f"融合-竖条 {bw}x{bh}", fuse=True, tiles=tiles)))
    seen = set()
    uniq: List[Tuple[str, Plan]] = []
    for k, p in out:
        key = (p.fuse, tuple(p.tiles))
        if key in seen:
            continue
        seen.add(key)
        uniq.append((k, p))
    return uniq


@dataclass
class AutoChoice:
    plan: Plan
    kind: str
    estimated_seconds: float
    baseline_seconds: float
    ranked: List[Tuple[str, float]]

    @property
    def speedup_vs_unfused(self) -> float:
        if self.estimated_seconds <= 0:
            return 0.0
        return self.baseline_seconds / self.estimated_seconds


def choose_plan(pipeline: Pipeline, in_dims: Tuple[int, int],
                machine: MachineModel = MACHINE) -> AutoChoice:
    """按代价模型在候选池里挑估算最快的方案 (不执行)。"""
    cands = candidate_plans(pipeline, in_dims)
    evals: List[Tuple[str, Plan, float]] = []
    baseline = 0.0
    for kind, plan in cands:
        e = estimate_plan(pipeline, plan, in_dims, machine)
        evals.append((kind, plan, e["seconds"]))
        if kind == "unfused":
            baseline = e["seconds"]
    fused = [x for x in evals if x[1].fuse]
    kind, plan, est = min(fused, key=lambda x: x[2])
    ranked = sorted(((k, s) for k, _, s in evals), key=lambda t: t[1])
    return AutoChoice(plan=plan, kind=kind, estimated_seconds=est,
                      baseline_seconds=baseline or est, ranked=ranked)
