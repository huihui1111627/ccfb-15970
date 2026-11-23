"""正确性测试: 窗口推导、kernel、执行器一致性。

重点保证两件事:
1. 每个 kernel 在"恰好覆盖依赖范围"的窗口上算出的结果, 与它在整图上算出的
   结果逐位相同 —— 这是分块执行正确的前提。
2. 执行器的三种方式 (逐算子整图 / 融合 / 融合+分块) 输出逐像素一致。
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from imgpipe import kernels as K
from imgpipe.dag import (OpSpec, WindowEstimator, point_op, scale_dim, scale_op,
                         scale_src_range, stat_op, stencil_op, tile_rects)
from imgpipe.imagegen import make_flat, make_photo
from imgpipe.plan import Pipeline, Plan, build_segments, group_windows, stage_windows
from imgpipe.runtime import execute


# ---------------------------------------------------------------------------
# 依赖范围描述
# ---------------------------------------------------------------------------

def test_scale_src_range_covers_sampled_pixels():
    """缩放窗口推导必须覆盖双线性实际访问的所有输入像素。"""
    for factor in (0.25, 0.5, 0.55, 1.0, 1.7, 3.0):
        for src_n in (1, 2, 7, 300, 1000):
            dst_n = scale_dim(src_n, factor)
            for d0 in range(0, dst_n, max(1, dst_n // 17)):
                d1 = min(dst_n, d0 + max(1, dst_n // 9))
                lo, hi = scale_src_range(d0, d1, factor, src_n)
                g = (np.arange(d0, d1, dtype=np.float64) + 0.5) / factor - 0.5
                g = np.clip(g, 0.0, src_n - 1.0)
                used = np.concatenate([np.floor(g), np.minimum(np.floor(g) + 1, src_n - 1)])
                assert lo <= used.min() and used.max() < hi


def test_window_estimator_is_conservative():
    """推导出的输入窗口一定包含每个输出像素真实依赖的输入像素。"""
    rng = np.random.default_rng(0)
    ops = [
        point_op("色彩", "color_matrix"),
        stencil_op("模糊", "blur", (2, 2)),
        stencil_op("锐化", "sharpen", (1, 1)),
        scale_op("缩放", "bilinear", 0.6, 0.6),
        stencil_op("去噪", "denoise", (1, 1)),
    ]
    dims = (257, 193)
    est = WindowEstimator(ops, dims)
    rect = (5, 7, 61, 49)
    got = est.input_rect(rect)
    # 用蒙特卡洛反推: 随机取输出像素, 沿算子链倒推它能影响的最小输入集合
    w_out = est.dims[-1][0]
    h_out = est.dims[-1][1]
    for _ in range(200):
        x = rng.integers(rect[0], rect[2])
        y = rng.integers(rect[1], rect[3])
        lo_x, lo_y, hi_x, hi_y = x, y, x + 1, y + 1
        for i in range(len(ops) - 1, -1, -1):
            op = ops[i]
            if op.kind == "scale":
                wi, hi_ = est.dims[i]
                s0, s1 = scale_src_range(lo_x, hi_x, op.scale[0], wi)
                t0, t1 = scale_src_range(lo_y, hi_y, op.scale[1], hi_)
                lo_x, hi_x, lo_y, hi_y = s0, s1, t0, t1
            else:
                rx, ry = op.radius
                lo_x -= rx
                hi_x += rx
                lo_y -= ry
                hi_y += ry
        assert got[0] <= lo_x and hi_x <= got[2]
        assert got[1] <= lo_y and hi_y <= got[3]


# ---------------------------------------------------------------------------
# 中值网络
# ---------------------------------------------------------------------------

def test_median9_network_all_binary_inputs():
    """0/1 原理: 对所有二值输入正确 => 对所有输入正确 (排序网络性质)。"""
    for bits in itertools.product((0, 1), repeat=9):
        v = [np.full((1, 1), b, np.float32) for b in bits]
        t = np.empty((1, 1), np.float32)
        for i, j in K.MEDIAN9:
            np.minimum(v[i], v[j], out=t)
            np.maximum(v[i], v[j], out=v[j])
            np.copyto(v[i], t)
        assert v[4][0, 0] == sorted(bits)[4]


def test_median3_matches_reference():
    rng = np.random.default_rng(1)
    x = rng.random((4, 40, 55), dtype=np.float32)
    ctx = K.Ctx((0, 0, 55, 40), (0, 0, 55, 40), (55, 40), (55, 40))
    got = K.denoise(ctx, {}, x)
    pad = np.pad(x, ((0, 0), (1, 1), (1, 1)), mode="edge")
    for c in range(3):
        ref = np.median(np.stack([pad[c, dy:dy + 40, dx:dx + 55]
                                  for dy in range(3) for dx in range(3)]), axis=0)
        np.testing.assert_array_equal(got[c], ref.astype(np.float32))


# ---------------------------------------------------------------------------
# kernel 的窗口一致性: 内部窗口 == 整图结果的对应块
# ---------------------------------------------------------------------------

def _run_on_window(kernel, op_params, full, rect, halo, dims, out_rect):
    """在 full 上裁出 rect±halo 的窗口 (贴边时按 edge 复制), 跑 kernel, 返回输出块。"""
    w, h = dims
    x0 = max(0, rect[0] - halo[0])
    y0 = max(0, rect[1] - halo[1])
    x1 = min(w, rect[2] + halo[0])
    y1 = min(h, rect[3] + halo[1])
    win = full[:, y0:y1, x0:x1]
    # 窗口贴住画布边缘时, 缺的那一圈由 kernel 自己补 —— 这里把窗口按
    # kernel 期望的方式准备好 (与 runtime 里的 final/crop 语义一致)。
    ctx = K.Ctx((x0, y0, x1, y1), out_rect, dims, dims)
    # kernel 自己按 (out_rect - in_rect) 裁掉 halo, 返回的就是输出块
    return kernel(ctx, op_params, win)


@pytest.mark.parametrize("name,halo,params", [
    ("denoise", (1, 1), {}),
    ("blur", (2, 2), {}),
    ("sharpen", (1, 1), {"amount": 0.8}),
])
def test_stencil_kernel_window_invariance(name, halo, params):
    """邻域 kernel: 在带 halo 的窗口上算, 与整图算完全一致。"""
    rng = np.random.default_rng(2)
    h, w = 90, 130
    full = rng.random((4, h, w), dtype=np.float32)
    ctx_full = K.Ctx((0, 0, w, h), (0, 0, w, h), (w, h), (w, h))
    ref = K.STENCIL_KERNELS[name](ctx_full, params, full)
    for rect in [(0, 0, 40, 30), (10, 10, 60, 50), (w - 30, 0, w, 25),
                 (0, h - 20, 33, h), (w - 11, h - 9, w, h), (50, 40, 51, 41)]:
        got = _run_on_window(K.STENCIL_KERNELS[name], params, full, rect, halo, (w, h), rect)
        rx0, ry0, rx1, ry1 = rect
        np.testing.assert_array_equal(got, ref[:, ry0:ry1, rx0:rx1])


def test_point_kernels_global_coordinates():
    """逐像素 kernel 中依赖全局坐标的 (水印) 必须在分块下位置一致。"""
    rng = np.random.default_rng(3)
    h, w = 120, 160
    full = rng.random((4, h, w), dtype=np.float32)
    ctx_full = K.Ctx((0, 0, w, h), (0, 0, w, h), (w, h), (w, h))
    ref = K.watermark(ctx_full, {}, full)
    for rect in [(0, 0, 50, 40), (60, 30, 120, 90), (100, 80, w, h)]:
        out = np.empty((4, rect[3] - rect[1], rect[2] - rect[0]), np.float32)
        ctx = K.Ctx(rect, rect, (w, h), (w, h))
        K.watermark(ctx, {}, full[:, rect[1]:rect[3], rect[0]:rect[2]], out=out)
        np.testing.assert_array_equal(out, ref[:, rect[1]:rect[3], rect[0]:rect[2]])


def test_bilinear_window_invariance():
    """双线性缩放的采样坐标只依赖全局网格, 分块结果应与整图一致。"""
    rng = np.random.default_rng(4)
    h, w = 101, 157
    full = rng.random((4, h, w), dtype=np.float32)
    for fx, fy in [(0.5, 0.5), (0.7, 0.6), (1.5, 1.3), (0.31, 0.83)]:
        dw, dh = scale_dim(w, fx), scale_dim(h, fy)
        ref = K.bilinear(full, (0, 0, w, h), (0, 0, dw, dh), (w, h), fx, fy)
        est = WindowEstimator([scale_op("s", "bilinear", fx, fy)], (w, h))
        for rect in [(0, 0, dw // 3, dh // 3), (dw // 2, dh // 2, dw, dh),
                     (5, 7, 6, 8), (dw - 3, 0, dw, dh)]:
            src = est.input_rect(rect)
            sx0, sy0, sx1, sy1 = src
            got = K.bilinear(full[:, sy0:sy1, sx0:sx1], src, rect, (w, h), fx, fy)
            np.testing.assert_array_equal(got, ref[:, rect[1]:rect[3], rect[0]:rect[2]])


# ---------------------------------------------------------------------------
# 执行器: 三种方式一致性
# ---------------------------------------------------------------------------

def _all_pipelines():
    from imgpipe import pipelines as P
    return [P.make(n) for n in P.ALL]


@pytest.mark.parametrize("pl", _all_pipelines(), ids=lambda p: p.name)
@pytest.mark.parametrize("size", [(213, 157), (401, 293)])
def test_executor_consistency(pl, size):
    """逐算子整图 vs 融合 vs 融合+多种分块: 输出逐像素相同。"""
    w, h = size
    image = make_photo(11, h, w)
    ref = execute(pl, image, Plan("unfused", fuse=False)).image
    fused = execute(pl, image, Plan("fused", fuse=True)).image
    np.testing.assert_array_equal(fused, ref)
    segs = [s for s in build_segments(pl, (w, h)) if s.kind == "fused"]
    for tw, th in [(32, 32), (64, 48), (97, 131), (w, 16), (16, h)]:
        tiles = [(tw, th)] * len(segs)
        got = execute(pl, image, Plan("tiled", fuse=True, tiles=tiles)).image
        np.testing.assert_array_equal(got, ref)


def test_flat_image_and_extremes():
    """全平图与 0/1 极端值下也要一致 (防止截断/边界处理差异)。"""
    for img in (make_flat(2, 97, 131),
                np.zeros((4, 80, 120), np.float32),
                np.ones((4, 80, 120), np.float32)):
        for pl in _all_pipelines():
            ref = execute(pl, img, Plan("u", fuse=False)).image
            for tiles in (None, [(37, 23)]):
                t = None if tiles is None else tiles * len(
                    [s for s in build_segments(pl, (img.shape[2], img.shape[1]))
                     if s.kind == "fused"])
                got = execute(pl, img, Plan("f", fuse=True, tiles=t or [])).image
                np.testing.assert_array_equal(got, ref)


# ---------------------------------------------------------------------------
# 统计断点
# ---------------------------------------------------------------------------

def test_histogram_is_order_independent():
    rng = np.random.default_rng(5)
    x = rng.random((4, 233, 311), dtype=np.float32)
    whole = np.zeros(256, np.int64)
    K.accumulate_hist(whole, x)
    parts = np.zeros(256, np.int64)
    for r in tile_rects(311, 233, (64, 64)):
        K.accumulate_hist(parts, x[:, r[1]:r[3], r[0]:r[2]])
    np.testing.assert_array_equal(whole, parts)


def test_stat_op_splits_segments():
    pl = Pipeline("t", [
        point_op("a", "color_matrix"),
        stat_op("h", "histogram"),
        point_op("b", "exposure"),
    ])
    segs = build_segments(pl, (100, 100))
    assert [s.kind for s in segs] == ["fused", "stat", "fused"]


def test_recompute_ratio_bounded():
    """分块的重算比例应随块变大而单调下降, 且整图执行时为 0。"""
    from imgpipe import pipelines as P
    pl = P.make("heavy_denoise")
    img = make_photo(1, 240, 320)
    ratios = []
    for th in (16, 32, 64, 128):
        r = execute(pl, img, Plan("t", fuse=True, tiles=[(320, th)] * 3))
        ratios.append(r.recompute)
    assert ratios == sorted(ratios, reverse=True)
    assert ratios[-1] < ratios[0]
    r0 = execute(pl, img, Plan("f", fuse=True))
    assert r0.recompute == 0.0


def test_retain_intermediates_preserves_output():
    """保留中间图 (业务现状的内存行为) 不改变任何一位结果。"""
    from imgpipe import pipelines as P
    pl = P.make("color_grade")
    img = make_photo(2, 180, 240)
    a = execute(pl, img, Plan("u", fuse=False), retain_intermediates=True)
    b = execute(pl, img, Plan("u", fuse=False), retain_intermediates=False)
    np.testing.assert_array_equal(a.image, b.image)
    assert len(a.intermediates) == sum(1 for s in a.segments if s.kind == "fused") - 1


def test_lut_kernels_bit_identical_across_blocks():
    """LUT 类 (gamma / hist_eq) 必须在任意分块下逐位相同。"""
    from imgpipe import pipelines as P
    pl = P.make("color_grade")
    img = make_photo(3, 200, 260)
    ref = execute(pl, img, Plan("u", fuse=False)).image
    for tiles in [(64, 64), (200, 17), (13, 260)]:
        n = sum(1 for s in build_segments(pl, (260, 200)) if s.kind == "fused")
        got = execute(pl, img, Plan("t", fuse=True, tiles=[tiles] * n)).image
        np.testing.assert_array_equal(got, ref)
