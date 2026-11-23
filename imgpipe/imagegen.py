"""模拟图像:比纯正弦更接近真实照片的纹理 (多尺度 1/f 噪声 + 边缘 + 色偏 + 局部高光)。

用固定 seed 生成, 保证基准测试可复现。
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def _fractal_noise(rng: np.random.Generator, h: int, w: int,
                   octaves: int = 6, base: int = 8) -> np.ndarray:
    """多倍频程 value noise, 近似 1/f^2 功率谱的自然纹理。

    逐行采样: 每个输出行只取上下两条格线做水平插值, 再加进累加图。峰值内存
    只有几个行缓冲 (与图高无关)。倍频程越高格点越密, 但格点数顶到图像尺寸
    就不再加密 —— 生成器本身的峰值必须显著小于图像, 否则会污染内存基准。
    """
    acc = np.zeros((h, w), dtype=np.float32)
    gx = np.linspace(0.0, 1.0, w, dtype=np.float32)
    amp = 1.0
    total = 0.0
    for o in range(octaves):
        gh = max(2, min(h, int(np.ceil(h / max(1.0, base / (2 ** o))))))
        gw = max(2, min(w, int(np.ceil(w / max(1.0, base / (2 ** o))))))
        grid = rng.standard_normal((gh, gw)).astype(np.float32)
        # 水平插值下标 (与行无关, 预先算好)
        xs = gx * (gw - 1)
        x0 = np.clip(np.floor(xs).astype(np.int32), 0, gw - 1)
        x1 = np.minimum(x0 + 1, gw - 1)
        fx = (xs - x0).astype(np.float32)
        ys = np.arange(h, dtype=np.float32) * ((gh - 1) / max(1, h - 1)) if h > 1 \
            else np.zeros(h, dtype=np.float32)
        ys = np.clip(ys, 0.0, gh - 1)
        gy = np.floor(ys).astype(np.int32)
        wy = (ys - gy).astype(np.float32)
        gy = np.clip(gy, 0, gh - 1)
        gy1 = np.minimum(gy + 1, gh - 1)
        for i in range(h):
            r0 = grid[gy[i]]
            r1 = grid[gy1[i]]
            a = r0[x0] * (1.0 - fx) + r0[x1] * fx
            b = r1[x0] * (1.0 - fx) + r1[x1] * fx
            acc[i] += (a * (1.0 - wy[i]) + b * wy[i]) * amp
        total += amp
        amp *= 0.55
    acc /= total
    return acc


def make_photo(seed: int, h: int, w: int) -> np.ndarray:
    """生成一张 (C, H, W) float32 RGBA 的类照片图像, 取值范围 0..1。"""
    rng = np.random.default_rng(seed)
    base = _fractal_noise(rng, h, w)
    base = (base - base.min()) / max(1e-6, float(base.max() - base.min()))

    # 一些硬边缘形状 (模拟物体轮廓): 随机矩形/椭圆
    edges = np.zeros((h, w), dtype=np.float32)
    for _ in range(rng.integers(6, 14)):
        cy, cx = rng.uniform(0, h), rng.uniform(0, w)
        ry, rx = rng.uniform(h * 0.03, h * 0.22), rng.uniform(w * 0.03, w * 0.22)
        yy = np.arange(h, dtype=np.float32)[:, None]
        xx = np.arange(w, dtype=np.float32)[None, :]
        m = (((yy - cy) / max(ry, 1)) ** 2 + ((xx - cx) / max(rx, 1)) ** 2) < 1.0
        edges += m.astype(np.float32) * rng.uniform(-0.25, 0.25)
    base = base * 0.55 + 0.25 + edges * 0.5

    # 局部高光 (模拟反光)
    yy = np.arange(h, dtype=np.float32)[:, None]
    xx = np.arange(w, dtype=np.float32)[None, :]
    for _ in range(int(rng.integers(1, 4))):
        cy, cx = rng.uniform(0, h), rng.uniform(0, w)
        r = rng.uniform(h * 0.06, h * 0.18)
        g = np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * r * r))).astype(np.float32)
        base = base + g * rng.uniform(0.15, 0.4)

    # 色偏 + 通道差异纹理
    gains = rng.uniform(0.82, 1.18, size=3).astype(np.float32)
    img = np.empty((4, h, w), dtype=np.float32)
    for c in range(3):
        tint = _fractal_noise(rng, h, w, octaves=3) * 0.06
        img[c] = np.clip(base * gains[c] + tint, 0.0, 1.0)
    img[3] = 1.0

    # 传感器噪声 (真实图像都有), 逐通道生成以免多占一份整图缓冲
    sigma = 0.006
    for c in range(3):
        img[c] += rng.standard_normal((h, w), dtype=np.float32) * sigma
        np.clip(img[c], 0.0, 1.0, out=img[c])
    return img


def make_flat(seed: int, h: int, w: int) -> np.ndarray:
    """平坦色块 + 少量高频, 用于观察直方图的极端情况。"""
    rng = np.random.default_rng(seed)
    img = np.empty((4, h, w), dtype=np.float32)
    img[0] = rng.uniform(0.2, 0.4)
    img[1] = rng.uniform(0.35, 0.6)
    img[2] = rng.uniform(0.5, 0.75)
    img[3] = 1.0
    # 叠加一点棋盘格与噪声
    yy, xx = np.mgrid[0:h, 0:w]
    cb = (((yy // 16) + (xx // 16)) % 2).astype(np.float32) * 0.08 - 0.04
    img[:3] += cb[None]
    img[:3] += rng.standard_normal((3, h, w)).astype(np.float32) * 0.01
    np.clip(img[:3], 0.0, 1.0, out=img[:3])
    return np.ascontiguousarray(img)
