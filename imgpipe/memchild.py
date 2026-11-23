"""峰值内存测量子进程入口 (由 :func:`imgpipe.acceptance.measure_peak_isolated` 调用)。"""

from __future__ import annotations

import json
import sys

from .acceptance import _mem_worker


def main() -> None:
    payload = json.loads(sys.argv[1])
    info = _mem_worker((payload["pipeline"], payload["w"], payload["h"],
                        payload["seed"], payload["fuse"], payload["tiles"],
                        payload.get("retain", False)))
    print(json.dumps(info))


if __name__ == "__main__":
    main()
