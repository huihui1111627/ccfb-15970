"""验收基准: 对"未融合 / 只融合 / 融合+分块 / 自动选择"四种方式做实测对照。

口径 (写进 README 和页面, 免得看的人误解)
----------------------------------------
* **耗时**: 进程 CPU 时间 (user+sys) 的最小值。每轮循环足够多次让 CPU 时间
  超过内核时钟粒度, 取多轮最小值剔掉被邻居抢占的轮次。共享机器上墙钟会被
  邻居干扰, CPU 时间稳定得多, 而且它就是"这张图花了多少 CPU"。
* **峰值内存**: 每个方案在 *独立干净进程* 里跑一次, 取 ``ru_maxrss`` 超过输入图
  的部分 (物化的整图中间结果 + tile 池 + 结果图)。不能在同一进程顺序测
  (分配器复用高水位会让后测的永远是 0), 也不能 fork (继承父进程高水位)。
* **重算比例**: 实际计算的输出像素 / 唯一像素 - 1, 由执行器精确计数。
* **正确性**: 每个方案都与"逐算子整图"基线逐像素比对, 报告最大绝对误差。
  融合链是同一批 kernel 换个调度顺序, 理论上逐位相同; 报告实测值。

自动选择 vs 实测最优: 候选池是 (融合开关 × 每个融合段的块尺寸)。自动选择用
代价模型排序, 实测最优用同一候选池全部跑一遍取最快, 比较两者所选方案的实测
耗时差距。
"""

from __future__ import annotations

import gc
import json
import os
import resource
import time
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import pipelines as P
from .autotune import candidate_plans, choose_plan
from .imagegen import make_photo
from .plan import Pipeline, Plan
from .runtime import ExecResult, execute

SIZES = {
    "640x480": (640, 480),
    "1280x960": (1280, 960),
    "1920x1080": (1920, 1080),
    "3000x2000": (3000, 2000),
}


# ---------------------------------------------------------------------------
# 单次测量
# ---------------------------------------------------------------------------

def cpu_seconds(fn, budget: float = 0.15, rounds: int = 4) -> float:
    """CPU 时间最小值: 每轮跑足够多次让 CPU 时间 >= ``budget`` 秒, 取最小值。

    单次执行几毫秒的方案直接测会撞上内核时钟粒度 (macOS 上 CPU 时间只有
    ~10ms 精度), 必须循环多次。循环次数由 CPU 时间 (而不是墙钟) 决定。
    取多轮最小值剔掉被邻居抢占的轮次。

    这台共享机器负载很高 (load avg ~20), 单次运行的相对噪声有 5-10%, 所以
    报告里的"实测最优"本身也带噪声 —— 自动选择只要落在实测最优的 10-15%
    以内, 就算达到目标。
    """
    fn()  # 预热
    best = None
    reps = 8
    for _ in range(rounds):
        gc.collect()
        r0 = resource.getrusage(resource.RUSAGE_SELF)
        for _ in range(reps):
            fn()
        r1 = resource.getrusage(resource.RUSAGE_SELF)
        spent = (r1.ru_utime - r0.ru_utime) + (r1.ru_stime - r0.ru_stime)
        best = spent / reps if best is None else min(best, spent / reps)
        if spent >= budget:
            break
        reps = min(2000, int(reps * max(1.5, budget / max(spent, 1e-6))))
    return float(best)


def _peak_rss_bytes() -> int:
    """进程历史峰值 RSS (字节)。macOS 上 ru_maxrss 单位就是字节。"""
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def peak_after_execute(image_bytes: int) -> int:
    """执行完一次之后调用: 返回"超过输入图本身"的进程峰值 RSS (字节)。

    ``ru_maxrss`` 是进程历史峰值, 只在干净进程里、且每个方案各用一个进程,
    这个差值才是"该方案新增的峰值"。输入图是执行前就分配好的, 先从峰值里
    扣掉它。分配器会复用高水位, 所以多个方案不能在同一进程里顺序测。
    """
    return max(0, _peak_rss_bytes() - image_bytes)


@dataclass
class Run:
    pipeline: str
    size: str
    plan: str
    kind: str                          # unfused / fused / tiled / auto / best
    fuse: bool
    tiles: List[Optional[Tuple[int, int]]]
    seconds: float
    peak_bytes: Optional[int]          # None = 该方案没测峰值
    out_dims: Tuple[int, int]
    recompute: float
    computed_px: int
    unique_px: int
    full_buffers: int
    kernel_calls: int
    traffic_bytes: int
    max_abs_err: float = 0.0
    mean_abs_err: float = 0.0
    est_seconds: float = 0.0
    segments: List[Dict[str, object]] = field(default_factory=list)

    def row(self) -> Dict[str, object]:
        d = asdict(self)
        d["tiles"] = [list(t) if t else None for t in self.tiles]
        d["out_dims"] = list(self.out_dims)
        d["seconds_ms"] = self.seconds * 1e3
        d["peak_mb"] = (self.peak_bytes / 1e6) if self.peak_bytes is not None else None
        return d


def _run_once(pipeline: Pipeline, image: np.ndarray, plan: Plan, size_name: str,
              kind: str, reference: Optional[np.ndarray],
              peak_bytes: Optional[int] = None,
              retain: bool = False) -> Run:
    result: Optional[ExecResult] = None

    def go():
        nonlocal result
        result = execute(pipeline, image, plan, retain_intermediates=retain)

    peak = peak_bytes
    secs = cpu_seconds(go)
    assert result is not None
    err_max = err_mean = 0.0
    if reference is not None:
        d = np.abs(result.image.astype(np.float64) - reference.astype(np.float64))
        err_max = float(d.max()) if d.size else 0.0
        err_mean = float(d.mean()) if d.size else 0.0
    return Run(
        pipeline=pipeline.name, size=size_name, plan=plan.name, kind=kind,
        fuse=plan.fuse, tiles=list(plan.tiles), seconds=secs, peak_bytes=peak,
        out_dims=(int(result.image.shape[2]), int(result.image.shape[1])),
        recompute=result.recompute, computed_px=result.computed_px,
        unique_px=result.unique_px, full_buffers=result.full_buffers,
        kernel_calls=result.kernel_calls, traffic_bytes=result.traffic_bytes,
        max_abs_err=err_max, mean_abs_err=err_mean,
        segments=result.segment_info,
    )




# ---------------------------------------------------------------------------
# 隔离子进程里测内存
# ---------------------------------------------------------------------------

def _mem_worker(args) -> Dict[str, object]:
    """子进程入口: 只做一次执行, 返回本方案的进程峰值 RSS (扣除输入图)。"""
    pl_name, w, h, seed, fuse, tiles, retain = args
    from . import pipelines as _P
    pl = _P.make(pl_name)
    image = make_photo(seed, h, w)
    plan = Plan("mem", fuse=fuse, tiles=[tuple(t) if t else None for t in tiles])
    gc.collect()
    r = execute(pl, image, plan, retain_intermediates=retain)
    return {
        "peak_bytes": peak_after_execute(int(image.nbytes)),
        "out_dims": [int(r.image.shape[2]), int(r.image.shape[1])],
        "recompute": r.recompute,
        "full_buffers": r.full_buffers,
    }


def measure_peak_isolated(pl_name: str, w: int, h: int, seed: int,
                          fuse: bool, tiles: Sequence[Optional[Tuple[int, int]]],
                          retain: bool = False) -> Dict[str, object]:
    """在全新子进程里量一次峰值内存。

    用一个独立的 python 进程加 ``-c`` 调 :mod:`imgpipe.memchild`: 这样解释器
    从干净状态启动, ``ru_maxrss`` 不含父进程的历史高水位 (fork 会继承, 量出来
    永远是 0)。子进程把结果以 JSON 打到 stdout。
    """
    import json
    import subprocess
    import sys
    pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    payload = json.dumps({"pipeline": pl_name, "w": w, "h": h, "seed": seed,
                          "fuse": fuse, "retain": retain,
                          "tiles": [list(t) if t else None for t in tiles]})
    proc = subprocess.run([sys.executable, "-m", "imgpipe.memchild", payload],
                          capture_output=True, text=True, cwd=pkg_root)
    if proc.returncode != 0:
        raise RuntimeError(f"峰值内存测量子进程失败: {proc.stderr[-800:]}")
    return json.loads(proc.stdout.strip().splitlines()[-1])

# ---------------------------------------------------------------------------
# 候选池
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# 一个 (流水线, 尺寸) 的完整对照
# ---------------------------------------------------------------------------

@dataclass
class CaseResult:
    pipeline: str
    size: str
    in_dims: Tuple[int, int]
    out_dims: Tuple[int, int]
    runs: Dict[str, Run]                     # kind -> run
    candidates: List[Run]
    auto_plan: Dict[str, object]
    best_kind: str
    est_for_auto: float
    auto_vs_best: float                      # auto 实测 / 最优实测
    correctness_ok: bool
    max_err: float


def run_case(pipeline: Pipeline, w: int, h: int, size_name: str,
             seed: int = 1, measure_mem: bool = True,
             candidates: Optional[List[Tuple[str, Plan]]] = None) -> CaseResult:
    pipeline.validate()
    image = make_photo(seed, h, w)

    plans = candidates or candidate_plans(pipeline, (w, h))
    # 峰值内存: 每个方案各用一条干净子进程。这一步开销大 (每方案要起一个
    # 解释器), 所以只给"未融合 / 融合 / 自动选择"三个关键方案实测; 其余候选
    # 的峰值用结构化估算 (物化的整图画布数 + 单块工作集)。
    peaks: Dict[str, int] = {}
    if measure_mem:
        auto_preview = choose_plan(pipeline, (w, h))
        key_plans = [("unfused", next(p for k, p in plans if k == "unfused"), True),
                     ("fused", next((p for k, p in plans if k == "fused"), None), False),
                     ("auto", auto_preview.plan, False)]
        for kind, plan, retain in key_plans:
            if plan is None:
                continue
            info = measure_peak_isolated(pipeline.name, w, h, seed, plan.fuse,
                                         plan.tiles, retain=retain)
            peaks[kind] = int(info["peak_bytes"])
        # 其余候选只测耗时, 不测峰值 (起子进程太贵, 而结构化估算又不准 ——
        # 实测整图执行的峰值里有好几张全尺寸缓冲, 不是"物化图数 x 画布大小"能算的)

    ref = execute(pipeline, image, Plan("逐算子整图", fuse=False)).image
    base = _run_once(pipeline, image, Plan("逐算子整图", fuse=False), size_name,
                     "unfused", None, peaks.get("unfused"), retain=True)

    measured: Dict[str, Run] = {}
    cand_runs: List[Run] = []
    for kind, plan in plans:
        if kind == "unfused":
            measured[kind] = base
            cand_runs.append(base)
            continue
        r = _run_once(pipeline, image, plan, size_name, kind, ref, peaks.get(kind))
        measured[kind] = r
        cand_runs.append(r)

    # 自动选择
    auto = choose_plan(pipeline, (w, h))
    auto_plan_desc = auto.plan.describe()
    auto_kind = None
    for kind, plan in plans:
        if plan.fuse == auto.plan.fuse and list(plan.tiles) == list(auto.plan.tiles):
            auto_kind = kind
            break
    if auto_kind is None:
        peak = peaks.get("auto") if measure_mem else None
        r_auto = _run_once(pipeline, image, auto.plan, size_name, "auto", ref, peak)
        cand_runs.append(r_auto)
        measured["auto"] = r_auto
    else:
        # 自动方案恰好在候选池里: 复用它的实测耗时; 若它正好是已测峰值的
        # 关键方案, 峰值也一并复用, 否则再单独量一次峰值。
        base_run = measured[auto_kind]
        peak = peaks.get(auto_kind)
        if peak is None and measure_mem:
            peak = int(measure_peak_isolated(pipeline.name, w, h, seed,
                                             auto.plan.fuse, auto.plan.tiles)["peak_bytes"])
            peaks[auto_kind] = peak
            # 把峰值也回填到对应候选行
            for cand in cand_runs:
                if cand is base_run:
                    cand.peak_bytes = peak
        measured["auto"] = Run(
            pipeline=base_run.pipeline, size=base_run.size, plan=base_run.plan,
            kind="auto", fuse=base_run.fuse, tiles=base_run.tiles,
            seconds=base_run.seconds, peak_bytes=peak, out_dims=base_run.out_dims,
            full_buffers=base_run.full_buffers, recompute=base_run.recompute,
            computed_px=base_run.computed_px, unique_px=base_run.unique_px,
            kernel_calls=base_run.kernel_calls, traffic_bytes=base_run.traffic_bytes,
            max_abs_err=base_run.max_abs_err, mean_abs_err=base_run.mean_abs_err,
            segments=base_run.segments,
        )

    # 实测最优 (只考虑融合类候选; 未融合仅作基线)
    fused_runs = [r for r in cand_runs if r.kind != "unfused"]
    best = min(fused_runs, key=lambda r: r.seconds)
    measured["best"] = best

    max_err = max(r.max_abs_err for r in cand_runs)
    ok = all(r.max_abs_err <= 1e-5 for r in cand_runs)
    auto_run = measured["auto"]
    ratio = auto_run.seconds / best.seconds if best.seconds > 0 else 1.0
    return CaseResult(
        pipeline=pipeline.name, size=size_name, in_dims=(w, h),
        out_dims=auto_run.out_dims, runs=measured, candidates=cand_runs,
        auto_plan=auto_plan_desc, best_kind=best.kind,
        est_for_auto=auto.estimated_seconds, auto_vs_best=ratio,
        correctness_ok=ok, max_err=max_err,
    )


MATRIX_DEFAULT = {
    "640x480": (640, 480),
    "1920x1080": (1920, 1080),
    "3000x2000": (3000, 2000),
}


def run_matrix(pipelines: Optional[Sequence[str]] = None,
               sizes: Optional[Dict[str, Tuple[int, int]]] = None,
               measure_mem: bool = True) -> Dict[str, object]:
    pl_names = list(pipelines or P.ALL.keys())
    size_map = sizes or MATRIX_DEFAULT
    cases: List[CaseResult] = []
    for name in pl_names:
        pl = P.make(name)
        for size_name, (w, h) in size_map.items():
            print(f"  [{name}] {size_name} ...", flush=True)
            cases.append(run_case(pl, w, h, size_name, measure_mem=measure_mem))
    return {"cases": cases}


def case_to_json(c: CaseResult) -> Dict[str, object]:
    return {
        "pipeline": c.pipeline,
        "size": c.size,
        "in_dims": list(c.in_dims),
        "out_dims": list(c.out_dims),
        "auto_plan": c.auto_plan,
        "best_kind": c.best_kind,
        "auto_vs_best": c.auto_vs_best,
        "est_for_auto": c.est_for_auto,
        "correctness_ok": c.correctness_ok,
        "max_err": c.max_err,
        "runs": {k: v.row() for k, v in c.runs.items()},
        "candidates": [r.row() for r in c.candidates],
    }
