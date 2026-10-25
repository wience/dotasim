"""Serve a run directory (runs/<name>, the one holding log.jsonl) with the dashboard at / and the plain fleet table
at /status.html: python -m dotasim.ui.serve runs/<name> [--port 8765]"""
import argparse
import http.server
import shutil
from pathlib import Path

HERE = Path(__file__).parent


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("run_dir")
    p.add_argument("--port", type=int, default=8765)
    a = p.parse_args(argv)
    run = Path(a.run_dir)
    run.mkdir(parents=True, exist_ok=True)
    shutil.copy(HERE / "status.html", run / "status.html")
    web = HERE.parent.parent / "web" / "index.html"   # the full dashboard; it reads run/status.json, log.jsonl, config.json
    if web.exists():
        shutil.copy(web, run / "index.html")

    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kw):
            super().__init__(*args, directory=str(run), **kw)

        def end_headers(self):
            self.send_header("Cache-Control", "no-store")
            super().end_headers()

        def log_message(self, *args):
            pass

    print(f"serving {run} at http://localhost:{a.port}/status.html")
    http.server.ThreadingHTTPServer(("0.0.0.0", a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
