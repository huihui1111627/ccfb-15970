"""仿真流水线:模拟真实图像服务里的几类典型流水线。

算子按业务命名 (上传图 -> 色彩 -> 去噪 -> 锐化 -> 缩放 -> 水印), 参数取值
参照常见的相机/电商图处理配置。
"""

from __future__ import annotations

from typing import List

from .dag import OpSpec, point_op, scale_op, stat_op, stencil_op
from .plan import Pipeline

_COLOR_M = [[1.06, -0.03, -0.03], [-0.04, 1.07, -0.03], [-0.02, -0.05, 1.07]]


def photo_thumb() -> Pipeline:
    """电商主图缩略图: 色彩校正 + 去噪 + 锐化 + 缩小 + 水印。"""
    ops = [
        point_op("色彩矩阵", "color_matrix", label="色彩校正", matrix=_COLOR_M,
                 bias=[0.008, 0.004, 0.0]),
        point_op("曝光", "exposure", label="曝光", ev=0.15),
        stencil_op("中值去噪", "denoise", (1, 1), label="中值去噪"),
        stencil_op("锐化", "sharpen", (1, 1), label="锐化", amount=0.9),
        scale_op("缩放", "bilinear", 0.55, 0.55, label="缩放 0.55x"),
        point_op("饱和度", "saturation", label="饱和度", saturation=1.08),
        point_op("水印", "watermark", label="水印"),
    ]
    return Pipeline("photo_thumb", ops, "电商缩略图流水线")


def color_grade() -> Pipeline:
    """相册美颜: 一连串逐像素色彩调整 + 模糊 + 直方图均衡 + 再次色彩 + 水印。"""
    ops = [
        point_op("色阶", "levels", label="色阶", lo=0.03, hi=0.96),
        point_op("色彩矩阵", "color_matrix", label="色彩校正", matrix=_COLOR_M),
        point_op("gamma", "gamma", label="gamma", gamma=0.92, gain=1.03),
        stencil_op("高斯模糊", "blur", (2, 2), label="高斯模糊"),
        point_op("曝光", "exposure", label="曝光", ev=-0.1),
        stat_op("直方图", "histogram", label="直方图统计", strength=0.75),
        point_op("直方图均衡", "hist_eq", label="直方图均衡", stat_idx=5),
        point_op("饱和度", "saturation", label="饱和度", saturation=1.12),
        point_op("水印", "watermark", label="水印"),
    ]
    return Pipeline("color_grade", ops, "美化调色, 中间带整图直方图断点")


def downscale_chain() -> Pipeline:
    """上传后处理: 先降采样再锐化, 两段缩放。"""
    ops = [
        point_op("色彩矩阵", "color_matrix", label="色彩校正", matrix=_COLOR_M),
        stencil_op("中值去噪", "denoise", (1, 1), label="中值去噪"),
        scale_op("缩放1", "bilinear", 0.7, 0.7, label="缩放 0.7x"),
        stencil_op("锐化", "sharpen", (1, 1), label="锐化", amount=1.0),
        point_op("gamma", "gamma", label="gamma", gamma=1.04),
        scale_op("缩放2", "bilinear", 0.8, 0.8, label="缩放 0.8x"),
        point_op("水印", "watermark", label="水印"),
    ]
    return Pipeline("downscale_chain", ops, "连续两段缩放的链式处理")


def watermark_only() -> Pipeline:
    """只加水印和轻量调色 (主要受内存带宽限制)。"""
    ops = [
        point_op("色彩矩阵", "color_matrix", label="色彩校正", matrix=_COLOR_M),
        point_op("曝光", "exposure", label="曝光", ev=0.05),
        point_op("水印", "watermark", label="水印"),
    ]
    return Pipeline("watermark_only", ops, "纯逐像素, 应当接近内存带宽上限")


def heavy_denoise() -> Pipeline:
    """高质量去噪: 中值 + 模糊 + 锐化 + 均衡, 邻域算子密集。"""
    ops = [
        stencil_op("中值去噪", "denoise", (1, 1), label="中值去噪"),
        stencil_op("高斯模糊", "blur", (2, 2), label="高斯模糊"),
        point_op("色阶", "levels", label="色阶", lo=0.02, hi=0.98),
        stencil_op("锐化", "sharpen", (1, 1), label="锐化", amount=0.7),
        stat_op("直方图", "histogram", label="直方图统计", strength=0.6),
        point_op("直方图均衡", "hist_eq", label="直方图均衡", stat_idx=4),
        point_op("水印", "watermark", label="水印"),
    ]
    return Pipeline("heavy_denoise", ops, "邻域算子密集, 融合收益最大")


ALL = {
    "photo_thumb": photo_thumb,
    "color_grade": color_grade,
    "downscale_chain": downscale_chain,
    "watermark_only": watermark_only,
    "heavy_denoise": heavy_denoise,
}


def make(name: str) -> Pipeline:
    try:
        return ALL[name]()
    except KeyError:
        raise KeyError(f"未知流水线 {name!r}; 可选: {', '.join(ALL)}") from None
