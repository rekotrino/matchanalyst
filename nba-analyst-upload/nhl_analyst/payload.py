"""Dashboard payload for the NHL board (read-only). Uses the generic shape the
shared dashboard understands: stat names / columns come from the payload."""

import json
from datetime import datetime, timedelta

from .analyst import players as pl
from .config import SETTINGS
from .ingest.espn import ET, TEAM_NAMES
from .model import p_cover, p_total_over, summarize
from .pipeline import analyze_game, build_context, performance
from .scanner import select_daily
from .storage import Store

STAT_NAMES = {"g": "Goals", "a": "Assists", "p": "Points", "sog": "Shots on goal", "blk": "Blocked shots",
              "hits": "Hits", "sv": "Saves", "toi": "Ice time"}
STATS_ORDER = ["p", "g", "a", "sog", "sv", "blk", "hits"]


def _num(v, nd=3):
    return round(v, nd) if isinstance(v, (int, float)) and not isinstance(v, bool) else v


def _rotation_entry(p):
    return {"player_id": p["player_id"], "player": p["player"], "team": p["team"], "position": p["position"],
            "games": p["games"], "starter_rate": 0,
            "last5": {k: _num(v) for k, v in p["last5"].items()}, "last10": {k: _num(v) for k, v in p["last10"].items()},
            "season": {k: _num(v) for k, v in p["season"].items()}, "trend": {k: _num(v) for k, v in p["trend"].items()},
            "recent": p["recent"]}


def _goalie_entry(gp):
    return {"player_id": gp["player_id"], "player": gp["player"], "team": gp["team"], "position": "G", "games": gp["games"],
            "starter_rate": 0, "last5": {k: _num(v) for k, v in gp["last5"].items()},
            "last10": {k: _num(v) for k, v in gp["last10"].items()}, "season": {k: _num(v) for k, v in gp["season"].items()},
            "trend": {}, "recent": gp["recent"], "goalie": True}


def _profile(p):
    l10, l5 = p["last10"], p["last5"]
    summary = []
    if l10["n"]:
        summary.append(["Last 10", f"{l10['record']} · GF {l10['gf']:.2f} · GA {l10['ga']:.2f} · shots {l10['sf']:.1f} for / {l10['sa']:.1f} against · "
                                   f"PP {100 * (l10['pp_pct'] or 0):.0f}% · PK {100 * (l10['pk_pct'] or 0):.0f}% · Sh% {100 * (l10['sh_pct'] or 0):.1f} · Sv% {1000 * (l10['sv_pct'] or 0):.0f}"])
        summary.append(["Last 5", f"{l5['record']} · GF {l5['gf']:.2f} · GA {l5['ga']:.2f}"])
    summary.append(["Splits", f"home margin {p['home_net']:+.2f} · away margin {p['away_net']:+.2f}" if p["home_net"] is not None and p["away_net"] is not None else "–"])
    return {"team": p["team"], "games": p["games"], "summary": summary, "last_games": p["last_games"],
            "last10": {k: _num(v) for k, v in l10.items()}, "last5": {k: _num(v) for k, v in l5.items()}}


def _market(m):
    mk, rows = m["market"], m["rows"]
    return {"market_id": mk["market_id"], "question": mk.get("question"), "type": mk["market_type"], "line": mk.get("line"),
            "liquidity": _num(mk.get("liquidity"), 1), "bid_ask": mk.get("bid_ask"), "outcomes": mk["outcomes"], "prices": mk["prices"],
            "rows": [{k: (_num(v) if isinstance(v, float) else v) for k, v in r.items()} for r in rows] if rows else None}


def prop_board(r, cfg=SETTINGS.betting):
    """Ladders from standard lines (fair prices) merged with any listed market."""
    G = r["game"]
    listed = {}
    for m in r["markets"]:
        rows = m["rows"]
        if rows and "player" in rows[0]:
            over = next((x for x in rows if x["outcome"].lower() in ("yes", "over")), None)
            under = next((x for x in rows if x["outcome"].lower() in ("no", "under")), None)
            if over and under:
                listed[(over["player"], over["stat"], over["line"])] = (m["market"], over, under)
    dtd = {i["player"].lower() for i in r["injuries"] if (i.get("status") or "").lower() not in ("out", "")}
    out = []
    for team, lad in r["ladders"].items():
        opp = G["away"] if team == G["home"] else G["home"]
        for (player, stat), e in lad.items():
            proj = e["proj"]
            ladder = []
            for ent in e["entries"]:
                key = (player, stat, ent["line"])
                mk, over, under = listed.get(key, (None, None, None))
                row = {"market_id": mk["market_id"] if mk else None, "line": ent["line"], "p_over": _num(ent["p_over"]),
                       "p_model": _num(ent["p_model"]), "hit10": _num(ent["hit10"], 2), "hit5": _num(ent["hit5"], 2),
                       "hit_season": _num(ent["hit_season"], 2), "hit_n": ent["hit_n"], "hit_season_n": ent["hit_season_n"],
                       "fair_over": _num(ent["p_over"]), "fair_under": _num(1 - ent["p_over"]),
                       "yes": over["price"] if over else None, "no": under["price"] if under else None,
                       "yes_buy": _num(over["buy_price"]) if over else None, "no_buy": _num(under["buy_price"]) if under else None,
                       "edge_over": _num(over["edge"]) if over else None, "edge_under": _num(under["edge"]) if under else None,
                       "liquidity": _num(mk.get("liquidity"), 0) if mk else 0}
                ladder.append(row)
            flags = []
            if player.lower() in dtd:
                flags.append("day-to-day")
            if e.get("starter_note"):
                flags.append("starter unconfirmed")
            liquid = [x for x in ladder if (x["liquidity"] or 0) >= cfg.min_liquidity]
            # with no market, the "floor we take" is the highest-probability standard line at 60%+ (fair price shown)
            pool = [] if "day-to-day" in flags else (liquid or ladder)
            safe = [x for x in pool if x["p_over"] >= 0.60 and (x["edge_over"] is None or x["edge_over"] >= -0.01)]
            value = [x for x in liquid if x["p_over"] >= cfg.prob_floor and (x["edge_over"] or -1) >= cfg.edge_threshold]
            under = [x for x in liquid if (1 - x["p_over"]) >= 0.60 and (x["edge_under"] or -1) >= 0]
            vals = e["recent"]
            out.append({
                "player_id": e["player_id"], "player": player, "team": team, "opp": opp, "position": e["position"], "stat": stat,
                "stat_label": STAT_NAMES[stat], "proj": _num(proj["mean"], 2), "proj_raw": _num(proj["mean"] / (e["factor"] or 1), 2),
                "proj_sd": _num(proj["sd"], 2), "proj_min": _num(proj["minutes"], 1), "p10": None,
                "l10_avg": _num(e["l10"], 2), "l5_avg": _num(e["l5"], 2), "season_avg": _num(e["season"], 2), "l10_sd": _num(e["l10_sd"], 2),
                "recent": vals, "l10_min": min(vals) if vals else None, "matchup": _num(e["factor"], 3), "matchup_info": None,
                "pace_factor": 1.0, "bump": 0, "rate60": _num(e.get("rate60"), 2), "starter_note": e.get("starter_note"),
                "ladder": ladder, "floor_line": ladder[0]["line"] if ladder else None,
                "floor_liquid": liquid[0]["line"] if liquid else (ladder[0]["line"] if ladder else None),
                "flags": flags, "liquid": bool(liquid), "fair_only": not liquid,
                "safe_pick": max(safe, key=lambda x: (x["p_over"], x["edge_over"] or 0)) if safe else None,
                "value_pick": max(value, key=lambda x: x["edge_over"]) if value else None,
                "under_pick": max(under, key=lambda x: (1 - x["p_over"], x["edge_under"])) if under else None,
            })
    out.sort(key=lambda x: (x["team"], -(x["l10_avg"] or 0)))
    return out


def fair_markets(S, home, away):
    rows = []
    for line in (-2.5, -1.5, 1.5, 2.5):
        rows.append({"type": "spread", "label": f"{home} {line:+.1f}", "p": _num(p_cover(S, True, line))})
    for line in (4.5, 5.5, 6.5, 7.5):
        rows.append({"type": "total", "label": f"Over {line}", "p": _num(p_total_over(S, line))})
    return rows


def build_payload(store: Store, days_ahead=None):
    ctx = build_context(store)
    today = datetime.now(ET).date()
    end = today + timedelta(days=days_ahead if days_ahead is not None else SETTINGS.lookahead_days)
    results = [analyze_game(store, g, ctx, record=False) for g in store.games_upcoming(str(today), str(end))]
    chosen = {p["market_id"] for p in select_daily([p for r in results for p in r["picks"]])}
    games = []
    for r in results:
        g, pred = r["game"], r["pred"]
        S = summarize(pred["exp_home"], pred["exp_away"])
        rotations = {t: [_rotation_entry(p) for p in ps] + [_goalie_entry(gp) for gp in r["goalies"][t]] for t, ps in r["rotations"].items()}
        games.append({
            "game": g, "home_name": TEAM_NAMES.get(g["home"], g["home"]), "away_name": TEAM_NAMES.get(g["away"], g["away"]),
            "pred": {"exp_home_pts": _num(pred["exp_home"], 2), "exp_away_pts": _num(pred["exp_away"], 2), "exp_margin": _num(pred["exp_home"] - pred["exp_away"], 2),
                     "exp_total": _num(pred["exp_total"], 2), "p_home_win": _num(pred["p_home_win"]), "p_tie_reg": _num(pred["p_tie_reg"]),
                     "anchor_weight": _num(pred["anchor_weight"]), "notes": pred["notes"], "pace": None},
            "model_only": {"exp_home_pts": _num(pred["model_only"]["exp_home"], 2), "exp_away_pts": _num(pred["model_only"]["exp_away"], 2),
                           "exp_margin": _num(pred["model_only"]["exp_home"] - pred["model_only"]["exp_away"], 2),
                           "exp_total": _num(pred["model_only"]["exp_total"], 2), "p_home_win": _num(pred["model_only"]["p_home_win"])},
            "ref": r["ref"], "ref_probs": {k: _num(v) for k, v in r["ref_probs"].items()} if r["ref_probs"] else None,
            "profiles": {t: _profile(p) for t, p in r["profiles"].items()}, "issues": r["issues"],
            "ratings": {t: ({"bars": [["Attack", 100 * (v.get("att") or 0), True], ["Defence", -100 * (v.get("def") or 0), True]],
                             "n": v.get("n", 0)} if v else None) for t, v in r["ratings"].items()},
            "rest": r["rest"], "injuries": r["injuries"], "rotations": rotations,
            "fair": fair_markets(S, g["home"], g["away"]), "markets": [_market(m) for m in r["markets"]],
            "picks": [{k: (_num(v) if isinstance(v, float) else v) for k, v in p.items()} for p in r["picks"] if p["market_id"] in chosen],
            "props": r["prop_items"], "lines": r["lines_table"], "books": r["books"], "profile_source": r["profile_source"],
        })
    teams = {}
    for t in TEAM_NAMES:
        rot = [_rotation_entry(p) for p in pl.team_rotation(ctx["by_skater"], t, top_n=18)]
        rot += [_goalie_entry(gp) for gp in pl.team_goalies(ctx["by_goalie"], t)]
        v = ctx["ratings"].get(t)
        teams[t] = {"name": TEAM_NAMES[t], "rating": ({"off": 100 * v["att"], "def": 100 * v["def"], "pace": 0} if v else None), "rotation": rot}
    b, m = SETTINGS.betting, SETTINGS.model
    return {
        "sport": "nhl", "sport_label": "NHL", "generated_at": datetime.now(ET).isoformat(timespec="minutes"),
        "season_label": f"{ctx['season'] - 1}-{str(ctx['season'])[2:]}", "n_cur_games": ctx["n_cur_games"], "n_prior_games": ctx["n_prior_games"],
        "stat_names": STAT_NAMES, "stats_order": STATS_ORDER, "minutes_key": "toi", "minutes_label": "TOI",
        "card_stats": [["g", "G"], ["a", "A"], ["p", "P"], ["sog", "SOG"]],
        "rotation_cols": [["toi", "TOI"], ["g", "G"], ["a", "A"], ["p", "P"], ["sog", "SOG"], ["blk", "BLK"], ["hits", "HIT"]],
        "goalie_cols": [["toi", "TOI"], ["sa", "SA"], ["sv", "SV"], ["ga", "GA"], ["sv_pct", "SV%"]],
        "prev_cols": [["toi", "TOI"], ["g", "G"], ["a", "A"], ["p", "P"], ["sog", "SOG"], ["blk", "BLK"], ["hits", "HIT"], ["pp_toi", "PP TOI"]],
        "prev_cols_goalie": [["toi", "TOI"], ["sa", "SA"], ["sv", "SV"], ["ga", "GA"]],
        "score_label": "goals", "no_props_note": "Polymarket lists no NHL player props; floors here are our fair prices for standard lines. Compare them with your book's odds.",
        "config": {"bankroll": b.bankroll, "edge_threshold": b.edge_threshold, "prob_floor": b.prob_floor, "min_liquidity": b.min_liquidity,
                   "kelly_fraction": b.kelly_fraction, "max_stake_pct": b.max_stake_pct, "anchor_cap": m.anchor_cap, "anchor_games": m.anchor_games},
        "league": {k: _num(v) for k, v in ctx["lg"].items()}, "games": games, "teams": teams, "team_names": TEAM_NAMES,
        "picks_history": store.pick_history(), "perf": {k: (_num(v) if isinstance(v, float) else v) for k, v in performance(store).items()},
    }


def export_json(store: Store, path, days_ahead=None):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(build_payload(store, days_ahead), f)
    return path
