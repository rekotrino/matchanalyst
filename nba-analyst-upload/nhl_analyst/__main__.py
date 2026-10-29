"""  python -m nhl_analyst run | update | analyze | grade | serve | export-html [out]"""
import argparse
import os
import sys

from . import pipeline
from .config import SETTINGS
from .payload import build_payload, export_json
from .storage import Store


def _store(db):
    return Store(db or SETTINGS.db_path)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="nhl_analyst")
    ap.add_argument("--db", default=None)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for n in ("run", "update"):
        s = sub.add_parser(n); s.add_argument("--no-prior", action="store_true"); s.add_argument("--max-summaries", type=int, default=600); s.add_argument("--days", type=int, default=None)
    s = sub.add_parser("analyze"); s.add_argument("--days", type=int, default=None)
    sub.add_parser("grade")
    s = sub.add_parser("serve"); s.add_argument("--port", type=int, default=8766)
    s = sub.add_parser("export-html"); s.add_argument("out", nargs="?", default="reports/nhl_board.html"); s.add_argument("--days", type=int, default=None)
    s = sub.add_parser("export-json"); s.add_argument("out", nargs="?", default="reports/nhl_board.json"); s.add_argument("--days", type=int, default=None)
    a = ap.parse_args(argv)
    if a.cmd in ("run", "update"):
        pipeline.update(_store(a.db), with_prior=not a.no_prior, max_summaries=a.max_summaries)
    if a.cmd in ("run", "analyze"):
        st = _store(a.db); pipeline.grade(st); ctx, results = pipeline.analyze(st, days_ahead=a.days)
        print(f"[analyze] {len(results)} games; picks: {sum(len(r['picks']) for r in results)}")
    if a.cmd == "grade":
        st = _store(a.db); pipeline.grade(st); print(pipeline.performance(st))
    if a.cmd == "serve":
        from nba_analyst.web import serve
        serve(a.db or SETTINGS.db_path, port=a.port, builder=build_payload, store_cls=Store, label="NHL")
    if a.cmd == "export-html":
        from nba_analyst.web import export_html
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        print("[export]", export_html(_store(a.db), a.out, a.days, builder=build_payload))
    if a.cmd == "export-json":
        print("[export]", export_json(_store(a.db), a.out, a.days))


if __name__ == "__main__":
    sys.exit(main())
