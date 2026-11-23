"""起一个本地静态服务器看页面: ``python -m imgpipe.serve``。"""

from __future__ import annotations

import argparse
import functools
import http.server
import os
import socketserver
from typing import Optional, Sequence


def serve(port: int = 8000, web_dir: Optional[str] = None) -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    root = web_dir or os.path.join(os.path.dirname(here), "web")
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=root)
    with socketserver.TCPServer(("127.0.0.1", port), handler) as httpd:
        print(f"http://127.0.0.1:{port}/  (Ctrl-C 退出)")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n已停止")


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args(argv)
    serve(args.port)


if __name__ == "__main__":
    main()
