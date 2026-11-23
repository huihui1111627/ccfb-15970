"""图像处理流水线的编译/执行引擎仿真。

顶层入口::

    from imgpipe import pipelines, plan, runtime, autotune
    pl = pipelines.make("photo_thumb")
    result = runtime.execute(pl, image, plan.Plan("fusion", fuse=True))

验收报告与网页: ``python -m imgpipe.report``。
"""
from .dag import OpSpec, WindowEstimator, point_op, scale_op, stat_op, stencil_op
from .plan import Pipeline, Plan, build_segments
from .runtime import execute, ExecResult

__all__ = [
    "OpSpec", "WindowEstimator", "point_op", "stencil_op", "scale_op", "stat_op",
    "Pipeline", "Plan", "build_segments", "execute", "ExecResult",
]
