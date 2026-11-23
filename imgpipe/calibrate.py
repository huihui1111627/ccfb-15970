"""微基准校准: 测出代价模型需要的硬件参数, 写成 ``calibration.json``。

测什么
------
1. **每个算子的 ns/像素** —— 在真实的 tile 执行上下文里测 (含池缓冲复用),
   按输出像素归一化; 分几个 tile 尺寸测, 取中位数。
2. **带宽曲线** —— 对"读 2 写 1"的逐像素流, 扫工作集大小, 算有效带宽。
3. **调用开销** —— 极小块上的固定耗时。

用法::

    python -m imgpipe.calibrate            # 测量并写入 imgpipe/calibration.json
    python -m imgpipe.calibrate --check    # 只打印当前参数

测量用 CPU 时间而不是墙钟时间 —— 共享机器上墙钟会被邻居干扰, CPU 时间是
稳定的下界。
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import resource
import time
from typing import Dict, List

import numpy as np

from . import kernels as K

HERE = os.path.dirname(os.path.abspath(__file__))
CALIB_PATH = os.path.join(HERE, "calibration.json")


def cpu_seconds(fn, repeats: int = 5) -> float:
    """取 repeats 次里 CPU 时间的最小值 (最小值最不受调度干扰)。"""
    best = None
    for _ in range(repeats):
        gc.collect()
        r0 = resource.getrusage(resource.RUSAGE_SELF)
        fn()
        r1 = resource.getrusage(resource.RUSAGE_SELF)
        dt = (r1.ru_utime - r0.ru_utime) + (r1.ru_stime - r0.ru_stime)
        best = dt if best is None else min(best, dt)
    return float(best)


# ---------------------------------------------------------------------------
# 算子成本
# ---------------------------------------------------------------------------

def _op_probe():
    """返回 (名字, 半径, halo, 调用 lambda)."""
    m = [[1.06, -0.03, -0.03], [-0.04, 1.07, -0.03], [-0.02, -0.05, 1.07]]
    lut = np.linspace(0, 1, 256, dtype=np.float32)
    return [
        ("color_matrix", 0, lambda c, x: K.color_matrix(c, {"matrix": m, "bias": [0.008, 0, 0]}, x)),
        ("exposure", 0, lambda c, x: K.exposure(c, {"ev": 0.15}, x)),
        ("saturation", 0, lambda c, x: K.saturation(c, {"saturation": 1.08}, x)),
        ("levels", 0, lambda c, x: K.levels(c, {"lo": 0.03, "hi": 0.96}, x)),
        ("gamma", 0, lambda c, x: K.gamma(c, {"gamma": 0.92, "gain": 1.03}, x)),
        ("hist_eq", 0, lambda c, x: K.hist_eq(c, {"stat_idx": 0}, x)),
        ("watermark", 0, lambda c, x: K.watermark(c, {}, x)),
        ("denoise", 1, lambda c, x: K.denoise(c, {}, x)),
        ("blur", 2, lambda c, x: K.blur(c, {}, x)),
        ("sharpen", 1, lambda c, x: K.sharpen(c, {}, x)),
    ]


def measure_op_costs(sizes=(256, 512, 1024), repeats: int = 3) -> Dict[str, float]:
    """返回 {算子: ns / 输出像素}。"""
    rng = np.random.default_rng(7)
    samples: Dict[str, List[float]] = {}
    for side in sizes:
        halo = 2
        big = rng.random((4, side + 2 * halo, side + 2 * halo), dtype=np.float32)
        for name, halo, fn in _op_probe():
            if halo == 0:
                ctx = K.Ctx((0, 0, side, side), (0, 0, side, side), (side, side), (side, side),
                            luts={0: np.linspace(0, 1, 256, dtype=np.float32)})
                x = big[:, :side, :side]
            else:
                # 内部窗口: 输入比输出大 2*halo
                ctx = K.Ctx((-halo, -halo, side + halo, side + halo),
                            (0, 0, side, side), (side, side), (side, side),
                            luts={0: np.linspace(0, 1, 256, dtype=np.float32)})
                x = big[:, :side + 2 * halo, :side + 2 * halo]
            fn(ctx, x)   # 预热
            dt = cpu_seconds(lambda: fn(ctx, x), repeats=repeats)
            samples.setdefault(name, []).append(dt / (side * side) * 1e9)
    out = {}
    for name, vals in samples.items():
        # 大 tile 下逐像素成本更接近稳态; 用最大值 (最保守) 与中位数的折中
        out[name] = float(np.median(vals))
    return out


def measure_scale_cost(sizes=(256, 512), repeats: int = 3) -> float:
    """缩放的 ns / 输出像素 (0.55x, 双线性)。"""
    rng = np.random.default_rng(8)
    vals = []
    for src_side in sizes:
        src = rng.random((4, src_side, src_side), dtype=np.float32)
        f = 0.55
        dst_side = max(1, int(round(src_side * f)))
        out_rect = (0, 0, dst_side, dst_side)
        K.bilinear(src, (0, 0, src_side, src_side), out_rect, (src_side, src_side), f, f)
        dt = cpu_seconds(lambda: K.bilinear(src, (0, 0, src_side, src_side), out_rect,
                                           (src_side, src_side), f, f), repeats=repeats)
        vals.append(dt / (dst_side * dst_side) * 1e9)
    return float(np.median(vals))


def measure_hist_cost(side: int = 1024, repeats: int = 3) -> float:
    rng = np.random.default_rng(9)
    x = rng.random((4, side, side), dtype=np.float32)
    hist = np.zeros(256, dtype=np.int64)
    K.accumulate_hist(hist, x)
    dt = cpu_seconds(lambda: K.accumulate_hist(hist, x), repeats=repeats)
    return float(dt / (side * side) * 1e9)


# ---------------------------------------------------------------------------
# 带宽曲线
# ---------------------------------------------------------------------------

def measure_bandwidth(sizes_mb=(0.25, 0.5, 1, 2, 4, 8, 16, 32, 64),
                      repeats: int = 5) -> List[List[float]]:
    """读 2 写 1 的逐像素流, 返回 [(工作集字节, GB/s), ...]。"""
    rng = np.random.default_rng(10)
    pts = []
    for mb in sizes_mb:
        n = int(mb * 1e6 / 4)
        a = rng.random(n, dtype=np.float32)
        b = rng.random(n, dtype=np.float32)
        dst = np.empty_like(a)

        def run(dst=dst, a=a, b=b):
            np.multiply(a, np.float32(1.01), out=dst)
            dst += b

        run()
        dt = cpu_seconds(run, repeats=repeats)
        ws = 3 * n * 4
        pts.append([float(ws), float(ws / dt / 1e9)])
    return pts


def measure_tile_locality() -> Dict[str, object]:
    """测分块的局部性系数: 整宽条带 / 方块, 在几条仿真流水线上的实测 / 模型 比值。

    这里直接用一个简化的"逐像素链 + 邻域"在长条带和方块上跑, 与"整图"比,
    得到条带相对整图的耗时比 —— 这就是代价模型的局部性修正项。
    """
    rng = np.random.default_rng(12)
    H, W = 960, 1280
    img = rng.random((4, H, W), dtype=np.float32)
    lut = np.linspace(0, 1, 256, dtype=np.float32)

    def chain(band_h: int, band_w: int) -> None:
        for y in range(0, H, band_h):
            for x in range(0, W, band_w):
                h = min(band_h, H - y)
                w = min(band_w, W - x)
                s = img[:, y:y + h, x:x + w]
                ctx = K.Ctx((x, y, x + w, y + h), (x, y, x + w, y + h), (W, H), (W, H),
                            luts={0: lut})
                a = K.color_matrix(ctx, {"matrix": [[1.05, 0.01, -0.02], [-0.03, 1.06, -0.01],
                                                   [0.0, -0.04, 1.08]]}, s)
                b = K.exposure(ctx, {"ev": 0.1}, a)
                K.saturation(ctx, {"saturation": 1.08}, b)

    base = cpu_seconds(lambda: chain(H, W), repeats=4)
    rows = {}
    for rows_h in (32, 64, 128, 256, 512):
        t = cpu_seconds(lambda: chain(rows_h, W), repeats=4)
        rows[f"band{rows_h}"] = float(t / base)
    sq = {}
    for side in (128, 256, 512):
        t = cpu_seconds(lambda: chain(side, side), repeats=4)
        sq[f"sq{side}"] = float(t / base)
    return {"band_vs_full": rows, "square_vs_full": sq}


def measure_call_overhead(repeats: int = 9) -> float:
    """极小块上的 kernel 调用固定开销 (ns)。"""
    x = np.random.default_rng(11).random((4, 16, 16), dtype=np.float32)
    ctx = K.Ctx((0, 0, 16, 16), (0, 0, 16, 16), (16, 16), (16, 16))
    op = lambda: K.exposure(ctx, {"ev": 0.1}, x, out=x)
    op()
    dt = cpu_seconds(op, repeats=repeats)
    # 减去 256 像素的计算时间 (按 3ns/px 估)
    return float(max(0.0, dt * 1e9 - 256 * 3.0))


# ---------------------------------------------------------------------------

def calibrate(write: bool = True) -> Dict[str, object]:
    print("测量算子成本 ...")
    op_costs = measure_op_costs()
    print("测量缩放成本 ...")
    scale = measure_scale_cost()
    print("测量直方图成本 ...")
    hist = measure_hist_cost()
    print("测量带宽曲线 ...")
    bw = measure_bandwidth()
    print("测量调用开销 ...")
    call = measure_call_overhead()
    print("测量分块局部性 ...")
    loc = measure_tile_locality()

    data = {
        "op_ns_per_px": op_costs,
        "scale_ns_per_out_px": scale,
        "hist_ns_per_px": hist,
        "bandwidth_points": bw,
        "call_overhead_ns": call,
        "tile_locality": loc,
        "measured_on": "apple-m4",
        "note": "ns/输出像素; 带宽单位为 GB/s; 由 imgpipe.calibrate 生成",
    }
    if write:
        with open(CALIB_PATH, "w") as f:
            json.dump(data, f, indent=2)
        print(f"已写入 {CALIB_PATH}")
    return data


def load() -> Dict[str, object]:
    if os.path.exists(CALIB_PATH):
        with open(CALIB_PATH) as f:
            return json.load(f)
    return {}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    if args.check:
        print(json.dumps(load(), indent=2))
    else:
        calibrate()
