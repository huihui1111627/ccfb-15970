"""验收报告: 跑完整基准矩阵, 汇总成 JSON + 静态页面。

用法::

    python -m imgpipe.report                 # 跑默认矩阵 (5 条流水线 x 3 个尺寸)
    python -m imgpipe.report --quick         # 小矩阵, 只用于冒烟
    python -m imgpipe.report --serve 8000    # 跑完起一个本地服务器看页面

输出 ``web/report.json`` 与 ``web/index.html``。报告里同时给出自动选择与
"候选池实测最优"的差距。
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import pipelines as P
from .acceptance import CaseResult, case_to_json, run_case
from .autotune import choose_plan
from .cost import MACHINE, estimate_plan
from .dag import canvas_size
from .plan import Plan, build_segments

HERE = os.path.dirname(os.path.abspath(__file__))
WEB_DIR = os.path.join(os.path.dirname(HERE), "web")

DEFAULT_SIZES = {
    "640x480": (640, 480),
    "1280x960": (1280, 960),
    "1920x1080": (1920, 1080),
    "3000x2000": (3000, 2000),
}
QUICK_SIZES = {"640x480": (640, 480), "1280x960": (1280, 960)}


# ---------------------------------------------------------------------------
# 流水线静态信息 (算子图 / 融合分组)
# ---------------------------------------------------------------------------

def pipeline_view(pipeline: P.Pipeline, in_dims: Tuple[int, int]) -> Dict[str, object]:
    """前端画算子图与融合分组所需的结构化描述。"""
    w, h = in_dims
    segs = build_segments(pipeline, (w, h))
    ops = []
    for i, op in enumerate(pipeline.ops):
        ops.append({
            "index": i,
            "name": op.name,
            "label": op.short(),
            "fn": op.fn,
            "kind": op.kind,
            "radius": list(op.radius),
            "scale": list(op.scale) if op.scale else None,
            "global_stat": op.global_stat,
            "params": {k: v for k, v in op.params.items()
                       if isinstance(v, (int, float, str, bool))},
        })
    segments = []
    seg_index = 0
    for seg in segs:
        if seg.kind == "stat":
            segments.append({
                "index": seg_index, "kind": "stat",
                "op_ids": seg.groups[0].op_ids,
                "in_dims": list(seg.in_dims), "out_dims": list(seg.out_dims),
                "groups": [{"kind": "stat", "ops": seg.groups[0].op_ids}],
            })
            seg_index += 1
            continue
        groups = []
        for g in seg.groups:
            groups.append({"kind": g.kind, "ops": g.op_ids,
                           "names": [pipeline.ops[i].short() for i in g.op_ids]})
        segments.append({
            "index": seg_index, "kind": "fused",
            "op_ids": seg.op_ids,
            "in_dims": list(seg.in_dims), "out_dims": list(seg.out_dims),
            "groups": groups,
        })
        seg_index += 1
    return {
        "name": pipeline.name,
        "note": pipeline.note,
        "ops": ops,
        "segments": segments,
        "out_dims": list(canvas_size(pipeline.ops, w, h)),
    }


def tile_view(plan: Plan, pipeline: P.Pipeline, in_dims: Tuple[int, int]) -> List[Dict[str, object]]:
    """每段的块划分, 给前端画一张图的分块示意。"""
    from .dag import tile_rects
    segs = build_segments(pipeline, in_dims)
    fused = [s for s in segs if s.kind == "fused"]
    tiles = plan.tiles or [None]
    out = []
    for i, seg in enumerate(fused):
        t = tiles[i] if i < len(tiles) else tiles[-1]
        rects = tile_rects(seg.out_dims[0], seg.out_dims[1], t)
        out.append({
            "segment": i,
            "out_dims": list(seg.out_dims),
            "tile": list(t) if t else None,
            "count": len(rects),
            "rects": [[r[0], r[1], r[2] - r[0], r[3] - r[1]] for r in rects],
        })
    return out


# ---------------------------------------------------------------------------
# 汇总
# ---------------------------------------------------------------------------

def summarize(cases: List[CaseResult]) -> Dict[str, object]:
    ratios = [c.auto_vs_best for c in cases]
    ok = all(c.correctness_ok for c in cases)
    max_err = max((c.max_err for c in cases), default=0.0)
    # 三种方式相对未融合的加速
    def speedups(key: str) -> List[float]:
        out = []
        for c in cases:
            base = c.runs["unfused"].seconds
            r = c.runs.get(key)
            if r and r.seconds > 0:
                out.append(base / r.seconds)
        return out
    peaks = {}
    for key in ("unfused", "fused", "auto"):
        vals = [c.runs[key].peak_bytes / max(1, c.runs["unfused"].peak_bytes)
                for c in cases
                if key in c.runs and c.runs[key].peak_bytes is not None
                and c.runs["unfused"].peak_bytes]
        peaks[key] = float(np.median(vals)) if vals else 0.0
    return {
        "n_cases": len(cases),
        "all_correct": ok,
        "max_abs_error": max_err,
        "auto_vs_best_median": float(np.median(ratios)),
        "auto_vs_best_p90": float(np.percentile(ratios, 90)),
        "auto_vs_best_max": float(np.max(ratios)),
        "auto_speedup_vs_unfused_median": float(np.median(speedups("auto"))),
        "fused_speedup_vs_unfused_median": float(np.median(speedups("fused"))),
        "peak_rel_unfused": peaks.get("unfused", 1.0),
        "peak_rel_fused": peaks.get("fused", 1.0),
        "peak_rel_auto": peaks.get("auto", 1.0),
        "accuracy_of_estimates": _estimate_accuracy(cases),
    }


def _estimate_accuracy(cases: List[CaseResult]) -> Dict[str, float]:
    """代价模型的相对误差 (按方案实际/预测的中位数)。"""
    ratios = []
    for c in cases:
        for r in c.candidates:
            est = estimate_plan(P.make(c.pipeline),
                                Plan(r.plan, fuse=r.fuse, tiles=r.tiles),
                                c.in_dims)["seconds"]
            if est > 0:
                ratios.append(r.seconds / est)
    if not ratios:
        return {"median_ratio": 1.0, "p10": 1.0, "p90": 1.0}
    return {
        "median_ratio": float(np.median(ratios)),
        "p10": float(np.percentile(ratios, 10)),
        "p90": float(np.percentile(ratios, 90)),
    }


def build_report(pipelines: Optional[Sequence[str]] = None,
                 sizes: Optional[Dict[str, Tuple[int, int]]] = None,
                 measure_mem: bool = True) -> Dict[str, object]:
    pl_names = list(pipelines or P.ALL.keys())
    size_map = sizes or DEFAULT_SIZES
    cases: List[CaseResult] = []
    views = {}
    for name in pl_names:
        pl = P.make(name)
        ref_size = size_map[list(size_map)[-1]]
        views[name] = pipeline_view(pl, (ref_size[0], ref_size[1]))
        for size_name, (w, h) in size_map.items():
            print(f"[{name}] {size_name} ...", flush=True)
            cases.append(run_case(pl, w, h, size_name, measure_mem=measure_mem))

    # 让前端能画出"自动选择的块布局"
    for c in cases:
        pl = P.make(c.pipeline)
        auto = choose_plan(pl, c.in_dims)
        c_tiles = tile_view(auto.plan, pl, c.in_dims)
        views[c.pipeline].setdefault("auto_tiles", {})[c.size] = {
            "plan": auto.plan.describe(), "tiles": c_tiles,
        }

    report = {
        "cases": [case_to_json(c) for c in cases],
        "summary": summarize(cases),
        "pipelines": views,
        "machine": {
            "op_ns_per_px": MACHINE.op_ns_per_px,
            "call_overhead_ns": MACHINE.call_overhead_ns,
            "bandwidth_points": MACHINE.bandwidth_points,
        },
        "sizes": {k: list(v) for k, v in (size_map or DEFAULT_SIZES).items()},
    }
    return report


def write_outputs(report: Dict[str, object], web_dir: str = WEB_DIR) -> Tuple[str, str]:
    os.makedirs(web_dir, exist_ok=True)
    json_path = os.path.join(web_dir, "report.json")
    with open(json_path, "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    html_src = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "web", "index.html")
    return json_path, html_src


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="图像流水线调度引擎 —— 验收报告")
    ap.add_argument("--quick", action="store_true", help="小矩阵冒烟")
    ap.add_argument("--pipelines", nargs="*", help="只跑指定流水线")
    ap.add_argument("--no-mem", action="store_true", help="跳过峰值内存测量")
    ap.add_argument("--out", default=None, help="输出 JSON 路径")
    args = ap.parse_args(argv)

    report = build_report(pipelines=args.pipelines,
                          sizes=QUICK_SIZES if args.quick else DEFAULT_SIZES,
                          measure_mem=not args.no_mem)
    path, _ = write_outputs(report)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, ensure_ascii=False, indent=1)
    s = report["summary"]
    print("\n=== 汇总 ===")
    print(f"用例数 {s['n_cases']}, 全部通过逐像素核对: {s['all_correct']} (最大误差 {s['max_abs_error']:.2e})")
    print(f"自动选择 / 实测最优 = 中位 {s['auto_vs_best_median']:.3f}, "
          f"p90 {s['auto_vs_best_p90']:.3f}, 最大 {s['auto_vs_best_max']:.3f}")
    print(f"相对未融合加速: 自动 {s['auto_speedup_vs_unfused_median']:.2f}x, "
          f"融合 {s['fused_speedup_vs_unfused_median']:.2f}x")
    print(f"报告写入 {path}")


if __name__ == "__main__":
    main()
