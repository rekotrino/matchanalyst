"""Build the JSON payload the dashboard renders. Read-only: nothing is written
to the store. The same payload is served by `serve` and can be embedded in the
page for a standalone copy (`export-json`)."""

import json
from datetime import datetime, timedelta

from .analyst import players as pl
from .config import SETTINGS
from .ingest.espn import ET, TEAM_NAMES
from .model.game_model import fair_markets
from .pipeline import analyze_game, build_context, performance, qualifying_performance
from .scanner import select_daily
from .storage import Store


def _num(v, nd=3):
    return None if v is None else round(v, nd)


def _block(b):
    if not b:
        return None
    return {k: (_num(v) if isinstance(v, float) else v) for k, v in b.items()}


def _profile(p):
    return {"team": p["team"], "games": p["games"], "last5": _block(p["last5"]), "last10": _block(p["last10"]),
            "season": _block(p["season"]), "home_net": _num(p["home_net"]), "away_net": _num(p["away_net"]),
            "last_games": p["last_games"]}


def _player(p):
    return {"player_id": p["player_id"], "player": p["player"], "team": p["team"], "position": p["position"],
            "games": p["games"], "starter_rate": _num(p["starter_rate"]),
            "last5": _block(p["last5"]), "last10": _block(p["last10"]), "season": _block(p["season"]),
            "trend": {k: _num(v) for k, v in p["trend"].items()}, "recent": p["recent"]}


def _market(m):
    mk = m["market"]
    rows = None
    keep = ("outcome", "our_prob", "buy_price", "edge", "proj", "proj_sd", "proj_min", "hit_rate", "player", "stat", "line")
    if m["rows"]:
        rows = [{k: (_num(v) if isinstance(v, float) else v) for k, v in r.items() if k in keep} for r in m["rows"]]
    return {"market_id": mk["market_id"], "question": mk.get("question"), "type": mk["market_type"],
            "line": mk.get("line"), "liquidity": _num(mk.get("liquidity"), 1), "bid_ask": mk.get("bid_ask"),
            "outcomes": mk["outcomes"], "prices": mk["prices"], "rows": rows}


def prop_board(r, cfg=SETTINGS.betting):
    """Group priced prop markets by player and stat into ladders, and suggest a
    floor line (highest-probability over that still has value) and a value line."""
    by = {}
    recent = {}
    for t, ps in r["rotations"].items():
        for p in ps:
            recent[p["player_id"]] = p["recent"]
    for m in r["markets"]:
        mk, rows = m["market"], m["rows"]
        if not rows or "player_id" not in rows[0]:
            continue
        over = next((x for x in rows if x["outcome"].lower() in ("yes", "over")), None)
        under = next((x for x in rows if x["outcome"].lower() in ("no", "under")), None)
        if not over or not under:
            continue
        key = (over["player_id"], over["stat"])
        if key not in by:
            vals = [x.get(over["stat"]) or 0 for x in (recent.get(over["player_id"]) or [])]
            by[key] = {
                "player_id": over["player_id"], "player": over["player"], "team": over["team"], "opp": over["opp"],
                "position": over["position"], "stat": over["stat"], "stat_label": over["stat_label"],
                "proj": _num(over["proj"], 1), "proj_raw": _num(over["proj_raw"], 1), "proj_sd": _num(over["proj_sd"], 1),
                "proj_min": _num(over["proj_min"], 0), "p10": _num(max(0.0, over["proj"] - 1.28 * over["proj_sd"]), 1),
                "l10_avg": _num(over["l10_avg"], 1), "l5_avg": _num(over["l5_avg"], 1), "season_avg": _num(over["season_avg"], 1),
                "l10_sd": _num(over["l10_sd"], 1), "recent": vals, "l10_min": min(vals) if vals else None,
                "matchup": _num(over["matchup"], 3), "matchup_info": over["matchup_info"], "pace_factor": _num(over["pace_factor"], 3),
                "bump": over["bump"], "ladder": [],
            }
        by[key]["ladder"].append({
            "market_id": mk["market_id"], "line": over["line"], "p_over": _num(over["our_prob"]), "p_model": _num(over["p_model"]),
            "yes": over["price"], "no": under["price"], "yes_buy": _num(over["buy_price"]), "no_buy": _num(under["buy_price"]),
            "edge_over": _num(over["edge"]), "edge_under": _num(under["edge"]), "liquidity": _num(mk.get("liquidity"), 0),
            "hit10": _num(over["hit_rate"], 2), "hit5": _num(over["hit5"], 2), "hit_season": _num(over["hit_season"], 2),
            "hit_n": over["hit_n"], "hit_season_n": over["hit_season_n"],
        })
    dtd = {i["player"].lower() for i in r["injuries"] if (i.get("status") or "").lower() not in ("out", "")}
    out = []
    for item in by.values():
        lad = sorted(item["ladder"], key=lambda x: x["line"])
        item["ladder"] = lad
        item["floor_line"] = lad[0]["line"] if lad else None
        liquid = [x for x in lad if (x["liquidity"] or 0) >= cfg.min_liquidity]
        item["floor_liquid"] = liquid[0]["line"] if liquid else None
        # warnings: a listed day-to-day status, or a liquid market priced far below our
        # number (the market usually knows about a late scratch or minutes limit first)
        flags = []
        if item["player"].lower() in dtd:
            flags.append("day-to-day")
        if any(x["edge_over"] >= 0.25 for x in liquid):
            flags.append("market doubt")
        item["flags"] = flags
        pool = [] if flags else liquid  # thin rungs carry placeholder quotes; flagged players get no suggestion
        safe = [x for x in pool if x["edge_over"] >= -0.01 and x["p_over"] >= 0.60]
        item["safe_pick"] = max(safe, key=lambda x: (x["p_over"], x["edge_over"])) if safe else None
        value = [x for x in pool if x["p_over"] >= cfg.prob_floor and x["edge_over"] >= cfg.edge_threshold]
        item["value_pick"] = max(value, key=lambda x: x["edge_over"]) if value else None
        under = [x for x in pool if (1 - x["p_over"]) >= 0.60 and x["edge_under"] >= 0.0]
        item["under_pick"] = max(under, key=lambda x: (1 - x["p_over"], x["edge_under"])) if under else None
        item["liquid"] = bool(liquid)
        out.append(item)
    out.sort(key=lambda x: (x["team"], -(x["l10_avg"] or 0)))
    return out


def _round(o, nd=3):
    """Round every float and drop null fields in a nested structure (keeps data.json small;
    the page treats a missing field like a null one)."""
    if isinstance(o, float):
        return round(o, nd)
    if isinstance(o, dict):
        return {k: _round(v, nd) for k, v in o.items() if v is not None}
    if isinstance(o, list):
        return [_round(v, nd) for v in o]
    return o


def build_payload(store: Store, days_ahead=None):
    ctx = build_context(store)
    today = datetime.now(ET).date()
    end = today + timedelta(days=days_ahead if days_ahead is not None else SETTINGS.lookahead_days)
    start = today - timedelta(days=1)
    upcoming = store.games_between(str(start), str(end))
    results = [analyze_game(store, g, ctx, record=False) for g in upcoming]
    chosen = {p["market_id"] for p in select_daily([p for r in results for p in r["picks"] if not r["game"].get("completed")])}
    qlog = {(q["game_id"], q["player"].lower(), q["stat"], q["line"], q["side"]): q for q in store.qualifying_history()}

    games, qualifying = [], []
    for r in results:
        g, pred = r["game"], r["pred"]
        actual = {}
        if g.get("completed"):
            for row in store.game_player_stats(g["game_id"]):
                row["pra"] = (row["pts"] or 0) + (row["reb"] or 0) + (row["ast"] or 0)
                row["pr"] = (row["pts"] or 0) + (row["reb"] or 0)
                row["pa"] = (row["pts"] or 0) + (row["ast"] or 0)
                row["ra"] = (row["reb"] or 0) + (row["ast"] or 0)
                actual[row["player_id"]] = row
        for item in r["prop_items"]:
            a = actual.get(item["player_id"])
            if a is not None:
                item["actual"] = None if (a.get("dnp") or not a.get("min")) else a.get(item["stat"])
            q = item.get("qualifying")
            if q:
                logged = qlog.get((g["game_id"], item["player"].lower(), item["stat"], q["line"], q["side"]))
                if item.get("actual") is not None:
                    q["actual"] = item["actual"]
                    q["result"] = "push" if item["actual"] == q["line"] else (
                        "win" if ((item["actual"] > q["line"]) == (q["side"] == "over")) else "loss")
                elif a is not None:
                    q["result"] = "void"
                q["logged_at"] = logged["created_at"] if logged else None
                qualifying.append({**q, "game_id": g["game_id"], "date_et": g["date_et"], "completed": bool(g.get("completed")),
                                   "player": item["player"], "team": item["team"], "opp": item["opp"], "stat": item["stat"],
                                   "stat_label": item["stat_label"], "sim": item.get("sim"), "proj_min": item.get("proj_min"),
                                   "l10_avg": item.get("l10_avg"), "recent": item.get("recent"), "matchup": item.get("matchup")})
        games.append({
            "game": g, "home_name": TEAM_NAMES.get(g["home"], g["home"]), "away_name": TEAM_NAMES.get(g["away"], g["away"]),
            "pred": {k: (_num(v) if isinstance(v, float) else v) for k, v in pred.items() if k != "model_only"},
            "model_only": {k: _num(v) for k, v in pred.get("model_only", {}).items()},
            "ref": r["ref"], "ref_probs": {k: _num(v) for k, v in r["ref_probs"].items()} if r["ref_probs"] else None,
            "profiles": {t: _profile(p) for t, p in r["profiles"].items()},
            "issues": r["issues"], "ratings": r["ratings"], "rest": r["rest"],
            "injuries": r["injuries"], "rotations": {t: [_player(p) for p in ps] for t, ps in r["rotations"].items()},
            "fair": [{"type": f["type"], "label": f["label"], "p": _num(f["p"])} for f in fair_markets(pred, g["home"], g["away"])],
            "markets": [_market(m) for m in r["markets"]],
            "props": _round(r["prop_items"]), "lines": r["lines_table"], "books": r["books"],
            "dvp": r.get("dvp"), "dvp_league": r.get("dvp_league"), "dvp_games": r.get("dvp_games"),
            "picks": [{k: (_num(v) if isinstance(v, float) else v) for k, v in p.items()}
                      for p in r["picks"] if p["market_id"] in chosen],
            "profile_source": r["profile_source"],
        })

    teams = {}
    for t in TEAM_NAMES:
        rot = pl.team_rotation(ctx["by_player"], t, top_n=12)
        teams[t] = {"name": TEAM_NAMES[t], "rating": ctx["ratings"].get(t), "rotation": [_player(p) for p in rot]}

    hist = store.pick_history()
    perf = performance(store)
    b, m = SETTINGS.betting, SETTINGS.model
    return {
        "generated_at": datetime.now(ET).isoformat(timespec="minutes"),
        "season_label": f"{ctx['season'] - 1}-{str(ctx['season'])[2:]}",
        "n_cur_games": ctx["n_cur_games"], "n_prior_games": ctx["n_prior_games"],
        "config": {"bankroll": b.bankroll, "edge_threshold": b.edge_threshold, "prob_floor": b.prob_floor,
                   "min_liquidity": b.min_liquidity, "kelly_fraction": b.kelly_fraction, "max_stake_pct": b.max_stake_pct,
                   "anchor_cap": m.anchor_cap, "anchor_games": m.anchor_games, "margin_sd": m.margin_sd, "total_sd": m.total_sd},
        "league": {k: _num(v) for k, v in ctx["lg"].items()},
        "games": games, "teams": teams, "team_names": TEAM_NAMES,
        "days": sorted({x["game"]["date_et"] for x in games}), "today_et": str(today),
        "qualifying": sorted(qualifying, key=lambda q: (q["completed"], -q["p"], -(q["ev"] or 0))),
        "qualifying_perf": qualifying_performance(store),
        "filter": {"prob": b.qualify_prob, "min_ev": b.qualify_min_ev, "min_games": b.qualify_min_games,
                   "hit_rate": b.qualify_hit_rate, "robust": b.qualify_robust, "n_sims": b.n_sims},
        "picks_history": hist, "perf": {k: (_num(v) if isinstance(v, float) else v) for k, v in perf.items()},
    }


def export_json(store: Store, path, days_ahead=None):
    data = build_payload(store, days_ahead)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)
    return path
