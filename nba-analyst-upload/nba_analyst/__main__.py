"""Command line entry point.

  python -m nba_analyst run            # update + analyze + report (the daily command)
  python -m nba_analyst update         # ingest only
  python -m nba_analyst analyze        # ratings + model + scan + reports from stored data
  python -m nba_analyst grade          # grade finished paper picks, print the record
  python -m nba_analyst demo           # offline synthetic run into data/demo.sqlite
"""

import argparse
import os
import sys

from . import pipeline, report
from .config import SETTINGS
from .storage import Store


def _store(path=None):
    return Store(path or SETTINGS.db_path)


def cmd_update(args):
    pipeline.update(_store(args.db), with_prior=not args.no_prior, max_summaries=args.max_summaries)


def cmd_analyze(args):
    store = _store(args.db)
    pipeline.grade(store)
    ctx, results = pipeline.analyze(store, days_ahead=args.days)
    digest, paths = report.write_reports(ctx, results, pipeline.performance(store), args.reports)
    print(f"[report] digest: {digest}")
    for p in paths:
        print(f"[report] {p}")
    return ctx, results


def cmd_run(args):
    cmd_update(args)
    cmd_analyze(args)


def cmd_grade(args):
    store = _store(args.db)
    pipeline.grade(store)
    perf = pipeline.performance(store)
    print(perf)


def cmd_players(args):
    """Rotation table for a team, or detail for one player."""
    from .analyst import players as pl
    store = _store(args.db)
    by = pl.player_games(store.player_stats())
    if args.team:
        rot = pl.team_rotation(by, args.team.upper(), top_n=args.top)
        print(f"\n{args.team.upper()} rotation by recent minutes (last-10 averages)\n")
        print(pl.player_table(rot))
    if args.name:
        matches = [g for g in by.values() if g[-1]["player"].lower() == args.name.lower()
                   or args.name.lower() in g[-1]["player"].lower()]
        if not matches:
            print(f"No player matching '{args.name}'")
            return
        games = matches[0]
        prof = pl.profile(games)
        print(f"\n{prof['player']} ({prof['team']}, {prof['position']}) — {prof['games']} games stored")
        for w in ("last5", "last10", "season"):
            b = prof[w]
            print(f"  {w:<7} n={b['n']:<3} MIN {b['min']:5.1f}  PTS {b['pts']:5.1f} (sd {b['pts_sd'] or 0:4.1f})  "
                  f"REB {b['reb']:4.1f}  AST {b['ast']:4.1f}  PRA {b['pra']:5.1f}  usage/36 {b['usage_proxy']:4.1f}"
                  + (f"  TS% {100 * b['ts_pct']:.1f}" if b.get('ts_pct') else ""))
        print("  recent:")
        for g in prof["recent"]:
            print(f"    {g['date']} {'vs' if g['home'] else '@ '} {g['opp']:<5} {g['min'] or 0:4.0f} min  "
                  f"{g['pts'] or 0:3d} pts {g['reb'] or 0:3d} reb {g['ast'] or 0:3d} ast{'  *' if g['starter'] else ''}")
        if args.line is not None and args.stat:
            proj = pl.projection(games, args.stat)
            p = pl.p_over(proj, args.line)
            hr, n = pl.hit_rate(games, args.stat, args.line)
            print(f"\n  {args.stat} O/U {args.line}: projection {proj['mean']:.1f} ± {proj['sd']:.1f} "
                  f"({proj['minutes']:.0f} min) -> P(over) {100 * p:.1f}%; hit {100 * hr:.0f}% of last {n}"
                  + ("\n  note: preseason games only - minutes are not representative; wait for regular-season games"
                     if proj.get("preseason_only") else ""))


def cmd_serve(args):
    from .web import serve
    extra = {}
    if args.with_nhl:
        from nhl_analyst.config import SETTINGS as NHL
        from nhl_analyst.payload import build_payload as nhl_payload
        from nhl_analyst.storage import Store as NhlStore
        extra["nhl"] = (args.nhl_db or NHL.db_path, nhl_payload, NhlStore)
    serve(args.db, port=args.port, extra_sources=extra)


def cmd_export_html(args):
    from .web import export_html
    print(f"[export] {export_html(_store(args.db), args.out, args.days)}")


def cmd_export_site(args):
    """Write <dir>/index.html with every sport embedded - drag the folder onto Netlify / any static host."""
    from .web import page_html
    from .payload import build_payload as nba_payload
    sources = {"nba": (args.db or SETTINGS.db_path, nba_payload, Store)}
    if not args.no_nhl:
        from nhl_analyst.config import SETTINGS as NHL
        from nhl_analyst.payload import build_payload as nhl_payload
        from nhl_analyst.storage import Store as NhlStore
        sources["nhl"] = (args.nhl_db or NHL.db_path, nhl_payload, NhlStore)
    data = {}
    for key, (db, builder, store_cls) in sources.items():
        if os.path.exists(db):
            data[key] = builder(store_cls(db), args.days)
    if not data:
        sys.exit(f"[site] no database found at {' or '.join(db for db, _, _ in sources.values())}; run `python -m nba_analyst run` first")
    os.makedirs(args.out, exist_ok=True)
    payload = {"sports": data} if len(data) > 1 else next(iter(data.values()))
    path = os.path.join(args.out, "index.html")
    if args.embed:
        with open(path, "w", encoding="utf-8") as f:
            f.write(page_html(payload))
        print(f"[site] {path}  ({', '.join(data)})  - single file with data embedded")
    else:
        import json as _json
        with open(path, "w", encoding="utf-8") as f:
            f.write(page_html())
        with open(os.path.join(args.out, "data.json"), "w", encoding="utf-8") as f:
            _json.dump(payload, f)
        stamp = "|".join(v["generated_at"] for v in data.values())
        with open(os.path.join(args.out, "version.json"), "w", encoding="utf-8") as f:
            _json.dump({"stamp": stamp, "sports": list(data)}, f)
        print(f"[site] {path} + data.json + version.json  ({', '.join(data)})  - the page checks version.json every 3 minutes")


def cmd_backtest(args):
    from . import backtest
    store = _store(args.db)
    recs = backtest.run(store, n_sims=args.sims, min_history=args.min_history, hit_weight=args.hit_weight)
    res = backtest.evaluate(recs)
    if not res.get("n"):
        print("[backtest] nothing to evaluate (need completed regular-season games with player box scores)")
        return
    os.makedirs(args.reports or "reports", exist_ok=True)
    path = backtest.report_md(res, os.path.join(args.reports or "reports", "backtest.md"))
    b, s_ = res["p_blend"], res["p_sim"]
    print(f"[backtest] n={res['n']}  Brier sim {s_['brier']:.4f} | blend {b['brier']:.4f} | recalibrated {res['p_calibrated']['brier']:.4f}")
    for side, sel in b["selections"].items():
        if sel["n"]:
            print(f"  {side}: n={sel['n']} hit {sel['hit_rate']:.3f} (pred {sel['avg_pred']:.3f}) ROI@-110 {sel['roi_at_-110']:+.3f}")
    print(f"  Platt a={res['platt']['a']:+.3f} b={res['platt']['b']:.3f}; minutes MAE {res['minutes']['mae']:.2f}")
    print(f"[backtest] report: {path}")
    if not args.no_write:
        print(f"[backtest] calibration written: {backtest.write_calibration(res)}")


def cmd_demo(args):
    from .demo import seed_demo
    db = args.db or "data/demo.sqlite"
    if os.path.exists(db):
        os.remove(db)
    store = Store(db)
    seed_demo(store)
    ctx, results = pipeline.analyze(store, days_ahead=3)
    digest, paths = report.write_reports(ctx, results, pipeline.performance(store), args.reports or "reports_demo")
    print(f"[demo] {len(results)} games analysed; digest at {digest}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="nba_analyst")
    ap.add_argument("--db", default=None, help="sqlite path (default data/nba_analyst.sqlite)")
    ap.add_argument("--reports", default=None, help="reports directory")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("run", cmd_run), ("update", cmd_update)):
        s = sub.add_parser(name)
        s.add_argument("--no-prior", action="store_true", help="skip last-season baseline pull")
        s.add_argument("--max-summaries", type=int, default=600)
        s.add_argument("--days", type=int, default=None, help="days ahead to analyse")
        s.set_defaults(fn=fn)
    s = sub.add_parser("analyze")
    s.add_argument("--days", type=int, default=None)
    s.set_defaults(fn=cmd_analyze)
    sub.add_parser("grade").set_defaults(fn=cmd_grade)
    s = sub.add_parser("players", help="player analytics: --team IND, or --name 'Tyrese Haliburton' [--stat pts --line 24.5]")
    s.add_argument("--team")
    s.add_argument("--name")
    s.add_argument("--top", type=int, default=10)
    s.add_argument("--stat", choices=["pts", "reb", "ast", "pra"])
    s.add_argument("--line", type=float)
    s.set_defaults(fn=cmd_players)
    s = sub.add_parser("serve", help="local dashboard at http://127.0.0.1:8765 (add --with-nhl for the NHL tab)")
    s.add_argument("--port", type=int, default=8765)
    s.add_argument("--with-nhl", action="store_true", help="also serve the NHL board as a tab")
    s.add_argument("--nhl-db", default=None)
    s.set_defaults(fn=cmd_serve)
    s = sub.add_parser("export-html", help="standalone dashboard page with data embedded")
    s.add_argument("out", nargs="?", default="reports/dashboard.html")
    s.add_argument("--days", type=int, default=None)
    s.set_defaults(fn=cmd_export_html)
    s = sub.add_parser("export-site", help="write site/index.html (NBA + NHL) for Netlify or any static host")
    s.add_argument("out", nargs="?", default="site")
    s.add_argument("--no-nhl", action="store_true")
    s.add_argument("--nhl-db", default=None)
    s.add_argument("--days", type=int, default=None)
    s.add_argument("--embed", action="store_true", help="single index.html with data embedded (no data.json)")
    s.set_defaults(fn=cmd_export_site)
    s = sub.add_parser("backtest", help="walk-forward check of the prop simulator; fits data/calibration.json")
    s.add_argument("--sims", type=int, default=3000)
    s.add_argument("--min-history", type=int, default=8)
    s.add_argument("--hit-weight", type=float, default=None)
    s.add_argument("--no-write", action="store_true", help="report only, do not write calibration.json")
    s.set_defaults(fn=cmd_backtest)
    sub.add_parser("demo").set_defaults(fn=cmd_demo)
    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
