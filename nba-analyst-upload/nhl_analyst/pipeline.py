"""update (ingest) -> analyze (ratings, model, scan) -> grade."""

import math
import time
from datetime import date, datetime, timedelta

from .analyst import players as pl
from .analyst.teams import (build_game_lines, flag_issues, league_averages, rest_info, strength_ratings,
                            team_profile)
from .config import SETTINGS
from .ingest import espn, polymarket
from .model import american_to_prob, anchor, devig, expected_goals, summarize
from .scanner import PROP_TYPES, kelly_stake, scan_game, select_daily
from .storage import Store
from nba_analyst.odds.ladders import build_prop_items, game_lines_table, scan_quotes
from nba_analyst.odds.polymarket_adapter import polymarket_quotes
from nba_analyst.pipeline import fetch_books
from .analyst.players import STAT_LABEL


def current_season_year(today=None):
    today = today or datetime.now(espn.ET).date()
    return today.year + 1 if today.month >= 8 else today.year


def _season_of(row):
    y, m = int(row["date_et"][:4]), int(row["date_et"][5:7])
    return y + 1 if m >= 8 else y


# ---------------------------------------------------------------------------
def update(store: Store, with_prior=True, max_summaries=600, log=print):
    start, end = espn.date_range(SETTINGS.lookback_days, SETTINGS.lookahead_days)
    log(f"[update] schedule {start} .. {end}")
    games, ref = espn.fetch_scoreboard(start, end)
    for g in games:
        store.upsert_game(g)
    for o in ref:
        store.add_reference_odds(o["game_id"], o["provider"], o["spread_home"], o["total"], o["ml_home"], o["ml_away"])
    store.commit()
    if with_prior:
        prior_year = current_season_year() - 1
        have = store.conn.execute("SELECT COUNT(*) FROM games WHERE season_year=? AND completed=1", (prior_year,)).fetchone()[0]
        if have < 200:
            pg, _ = espn.fetch_scoreboard(date(prior_year, 3, 1), date(prior_year, 4, 18))
            for g in pg:
                if g["season_type"] == 2:
                    store.upsert_game(g)
            store.commit()
            log(f"[update] {len(pg)} prior-season games")
    missing = store.games_missing_stats()
    log(f"[update] box scores to fetch: {len(missing)}")
    for i, g in enumerate(missing[:max_summaries]):
        try:
            s = espn.fetch_summary(g["game_id"])
        except Exception as e:  # noqa: BLE001
            log(f"  ! {g['game_id']}: {e}")
            continue
        for r in s["team"]:
            r["goals"] = g["home_score"] if r["is_home"] else g["away_score"]
            store.upsert_team_stats(r)
        for r in s["skaters"]:
            store.upsert_skater(r)
        for r in s["goalies"]:
            store.upsert_goalie(r)
        if s.get("finish") and not g.get("finish"):
            store.conn.execute("UPDATE games SET finish=? WHERE game_id=?", (s["finish"], g["game_id"]))
        if (i + 1) % 25 == 0:
            store.commit()
        time.sleep(0.2)
    store.commit()
    today = datetime.now(espn.ET).date()
    for g in store.games_upcoming(str(today), str(today + timedelta(days=SETTINGS.lookahead_days))):
        try:
            s = espn.fetch_summary(g["game_id"])
            store.replace_injuries(g["game_id"], s["injuries"])
        except Exception as e:  # noqa: BLE001
            log(f"  ! injuries {g['game_id']}: {e}")
        try:
            slug, markets = polymarket.fetch_game_markets(g["away"], g["home"], g["date_et"])
            if slug:
                store.add_pm_markets(g["game_id"], slug, markets)
                log(f"  {g['away']} @ {g['home']}: {len(markets)} markets ({slug})")
        except Exception as e:  # noqa: BLE001
            log(f"  ! polymarket {g['game_id']}: {e}")
        time.sleep(0.2)
    store.commit()
    fetch_books(store, store.games_upcoming(str(today), str(today + timedelta(days=SETTINGS.lookahead_days))), "nhl", espn.TEAM_NAMES, log)


# ---------------------------------------------------------------------------
def build_context(store: Store):
    season = current_season_year()
    rows = store.team_stats()
    cur = build_game_lines([r for r in rows if _season_of(r) == season])
    prior = build_game_lines([r for r in rows if _season_of(r) == season - 1])
    prior_r = strength_ratings(prior) if prior else {}
    ratings = strength_ratings(cur, prior=prior_r) if cur else {t: dict(v, n=0) for t, v in prior_r.items()}
    lg = league_averages(cur) if len(cur) >= 80 else league_averages(prior)
    games_played = {}
    for l in cur:
        games_played[l["team"]] = games_played.get(l["team"], 0) + 1
    sk_rows = store.skater_stats()
    gl_rows = store.goalie_stats()
    return {"season": season, "ratings": ratings, "lg": lg, "cur_lines": cur, "prior_lines": prior,
            "profile_lines": cur or prior, "games_played": games_played,
            "by_skater": pl.skater_games(sk_rows), "by_goalie": pl.goalie_games(gl_rows),
            "n_cur_games": len(cur) // 2, "n_prior_games": len(prior) // 2}


def analyze_game(store: Store, g, ctx, record=True):
    home, away, lg, ratings = g["home"], g["away"], ctx["lg"], ctx["ratings"]
    rest_h, rest_a = rest_info(ctx["cur_lines"], home, g["date_et"]), rest_info(ctx["cur_lines"], away, g["date_et"])
    lh, la, notes = expected_goals(home, away, ratings, lg.get("gf", SETTINGS.model.league_gpg), rest_h, rest_a)
    raw = summarize(lh, la)
    ref = store.latest_reference_odds(g["game_id"])
    ref_probs = None
    if ref and ref.get("ml_home") is not None:
        ph, pa = devig(american_to_prob(ref["ml_home"]), american_to_prob(ref["ml_away"]))
        ref_probs = {"p_home": ph, "p_away": pa}
    gp = min(ctx["games_played"].get(home, 0), ctx["games_played"].get(away, 0))
    lh2, la2, w = anchor(lh, la, ref_probs["p_home"] if ref_probs else None, ref["total"] if ref else None, gp)
    S = summarize(lh2, la2)
    pred = {"exp_home": S["exp_home"], "exp_away": S["exp_away"], "exp_total": S["exp_total"], "p_home_win": S["p_home_win"],
            "p_tie_reg": S["p_tie"], "anchor_weight": w, "notes": notes,
            "model_only": {"exp_home": raw["exp_home"], "exp_away": raw["exp_away"], "exp_total": raw["exp_total"],
                           "p_home_win": raw["p_home_win"]}}
    if record:
        store.add_prediction(g["game_id"], pred, {"ratings": {home: ratings.get(home), away: ratings.get(away)},
                                                  "reference": ref, "anchor_weight": w})
    profiles = {t: team_profile(ctx["profile_lines"], t) for t in (home, away)}
    issues = {t: flag_issues(profiles[t], lg) for t in (home, away)}
    injuries = store.latest_injuries(g["game_id"])
    out_names = {t: [i["player"] for i in injuries if i["team"] == t and (i.get("status") or "").lower() == "out"] for t in (home, away)}
    rotations = {t: pl.team_rotation(ctx["by_skater"], t) for t in (home, away)}
    goalies = {t: pl.team_goalies(ctx["by_goalie"], t) for t in (home, away)}
    ladders = build_ladders(g, ctx, S, rotations, goalies, profiles, out_names)
    player_ctx = {"ladders": ladders}
    markets = store.latest_pm_markets(g["game_id"])
    priced, _pm_picks = scan_game(g, S, markets, player_ctx=player_ctx)
    # ---- every book on one footing -------------------------------------------------
    quotes = polymarket_quotes(g, markets, espn.NICK_TO_ESPN, PROP_TYPES) + store.latest_quotes(g["game_id"])
    fns = {}
    for team, lad in ladders.items():
        for (player, stat), e in lad.items():
            fns[(player.lower(), stat)] = {
                "p_fn": e["p_fn"], "hit_fn": (lambda line, n, pid=e["player_id"], st=stat, tm=team: _hit(ctx, pid, st, tm, line, n)),
                "player_id": e["player_id"], "player": player, "team": team, "opp": away if team == home else home,
                "position": e["position"], "proj": e["proj"], "proj_raw": e["proj"]["mean"] / (e["factor"] or 1), "recent": e["recent"],
                "l10": e["l10"], "l5": e["l5"], "season": e["season"], "l10_sd": e["l10_sd"], "matchup": e["factor"],
                "rate60": e.get("rate60"), "starter_note": e.get("starter_note"), "p10": None}
    dtd = {i["player"].lower() for i in injuries if (i.get("status") or "").lower() not in ("out", "")}
    prop_items = build_prop_items(g, fns, quotes, SETTINGS.standard_lines, SETTINGS.betting, lambda st: STAT_LABEL.get(st, st), dtd,
                                  flags_for=lambda e: (["starter unconfirmed"] if e.get("starter_note") else []))
    from .model import p_cover as _pc, p_total_over as _pt
    def prob_for(q):
        if q.market == "moneyline":
            return S["p_home_win"] if q.side == home else (1 - S["p_home_win"] if q.side == away else None)
        if q.market == "spreads" and q.line is not None and q.side in (home, away):
            return _pc(S, q.side == home, q.line)
        if q.market == "totals" and q.line is not None:
            po = _pt(S, q.line)
            return po if q.side == "over" else 1 - po
        if q.player and q.line is not None:
            key = next((k for k in fns if k[1] == q.market and (k[0] == q.player.lower() or k[0].split()[-1] == q.player.lower().split()[-1])), None)
            if key:
                p_o = fns[key]["p_fn"](q.line)
                return p_o if q.side == "over" else 1 - p_o
        return None
    lines_table = game_lines_table(g, quotes, prob_for)
    _c, picks = scan_quotes(g, quotes, prob_for, SETTINGS.betting, set(PROP_TYPES.values()), kelly_stake)
    return {"game": g, "pred": pred, "profiles": profiles, "issues": issues, "rest": {home: rest_h, away: rest_a},
            "prop_items": prop_items, "lines_table": lines_table, "books": sorted({q.book for q in quotes}), "quotes": quotes,
            "injuries": injuries, "rotations": rotations, "goalies": goalies, "ladders": ladders,
            "ref": ref, "ref_probs": ref_probs, "markets": priced, "picks": picks,
            "ratings": {home: ratings.get(home), away: ratings.get(away)},
            "profile_source": "current season" if ctx["cur_lines"] else "last season"}


def build_ladders(g, ctx, S, rotations, goalies, profiles, out_names):
    """{team: {(player, stat): {entries: [...], p_fn, meta}}} - standard lines priced
    for every rotation skater and the likely starting goalie."""
    home, away, lg = g["home"], g["away"], ctx["lg"]
    out = {}
    cfg = SETTINGS
    for team in (home, away):
        opp = away if team == home else home
        opp_def = math.exp(ctx["ratings"].get(opp, {}).get("def", 0.0))  # >1 = leaky defence
        exp_team_goals = S["exp_home"] if team == home else S["exp_away"]
        team_shots_rel = (profiles[team]["last10"]["sf"] or lg["sf"]) / lg["sf"] if lg.get("sf") else 1.0
        opp_sa_rel = (profiles[opp]["last10"]["sa"] or lg["sa"]) / lg["sa"] if lg.get("sa") else 1.0
        shot_factor = max(0.85, min(1.15, (team_shots_rel * opp_sa_rel) ** 0.5))
        lad = {}
        outs = {n.lower() for n in out_names.get(team, [])}
        for p in rotations[team]:
            if p["player"].lower() in outs:
                continue
            games = [x for x in ctx["by_skater"].get(p["player_id"], []) if x["team"] == team]
            if len(games) < cfg.betting.prop_min_games:
                continue
            for stat, lines in cfg.standard_lines.items():
                if stat == "sv" or not lines:
                    continue
                factor = max(0.85, min(1.15, opp_def ** 0.5)) if stat in ("g", "a", "p") else (shot_factor if stat == "sog" else 1.0)
                proj = pl.projection(games, stat, factor)
                if not proj:
                    continue
                entries = []
                for line in lines:
                    pb, pm, hr, hn = pl.blended_p_over(games, stat, line, proj)
                    hr5, _ = pl.hit_rate(games, stat, line, 5)
                    hrs, ns = pl.hit_rate(games, stat, line, 0)
                    entries.append({"line": line, "p_over": pb, "p_model": pm, "hit10": hr, "hit5": hr5, "hit_season": hrs,
                                    "hit_n": hn, "hit_season_n": ns})
                lad[(p["player"], stat)] = {
                    "entries": entries, "proj": proj, "factor": factor, "player_id": p["player_id"], "position": p["position"],
                    "p_fn": (lambda line, gm=games, st=stat, pj=proj: pl.blended_p_over(gm, st, line, pj)[0]),
                    "recent": [x.get(stat) or 0 for x in p["recent"]], "l10": p["last10"].get(stat), "l5": p["last5"].get(stat),
                    "season": p["season"].get(stat), "l10_sd": p["last10"].get(stat + "_sd"), "rate60": proj["rate60"],
                }
        # goalie saves
        starter = pl.likely_starter(goalies[team], g["date_et"])
        if starter:
            exp_sa = lg.get("sa", 29.0) * ((profiles[team]["last10"]["sa"] or lg["sa"]) / lg["sa"]) ** 0.5 * \
                     ((profiles[opp]["last10"]["sf"] or lg["sf"]) / lg["sf"]) ** 0.5 if lg.get("sa") else 29.0
            ggames = [x for x in ctx["by_goalie"].get(starter["player_id"], []) if x["team"] == team]
            proj = pl.saves_projection(ggames, exp_sa)
            from nba_analyst.model.game_model import norm_cdf
            base = round(proj["mean"])
            lines = [base + d + 0.5 for d in (-6, -4, -2, 0, 2)]
            entries = []
            for line in lines:
                pm = 1 - norm_cdf((line - proj["mean"]) / proj["sd"])
                hr, hn = pl.hit_rate(ggames, "sv", line, 10)
                w = 0.65 if hn >= 8 else (0.8 if hn >= 5 else 1.0)
                p = w * pm + (1 - w) * hr if hr is not None else pm
                hr5, _ = pl.hit_rate(ggames, "sv", line, 5)
                hrs, ns = pl.hit_rate(ggames, "sv", line, 0)
                entries.append({"line": line, "p_over": p, "p_model": pm, "hit10": hr, "hit5": hr5, "hit_season": hrs,
                                "hit_n": hn, "hit_season_n": ns})
            lad[(starter["player"], "sv")] = {
                "entries": entries, "proj": proj, "factor": 1.0, "player_id": starter["player_id"], "position": "G",
                "p_fn": (lambda line, pj=proj: 1 - norm_cdf((line - pj["mean"]) / pj["sd"])),
                "recent": [x["sv"] for x in starter["recent"]], "l10": starter["last10"]["sv"], "l5": starter["last5"]["sv"],
                "season": starter["season"]["sv"], "l10_sd": starter["last10"]["sv_sd"], "rate60": None,
                "starter_note": "likely starter (more starts, did not play yesterday); confirm before betting",
            }
        out[team] = lad
    return out


def analyze(store: Store, days_ahead=None, log=print):
    ctx = build_context(store)
    log(f"[analyze] season {ctx['season']}: {ctx['n_cur_games']} current games, {ctx['n_prior_games']} prior, {len(ctx['ratings'])} teams")
    today = datetime.now(espn.ET).date()
    end = today + timedelta(days=days_ahead if days_ahead is not None else SETTINGS.lookahead_days)
    results = [analyze_game(store, g, ctx) for g in store.games_upcoming(str(today), str(end))]
    existing = {p["market_id"] for p in store.open_picks()}
    chosen = select_daily([p for r in results for p in r["picks"]])
    for p in chosen:
        if p["market_id"] not in existing:
            store.add_pick(dict(p, mode="paper"))
    store.commit()
    ids = {p["market_id"] for p in chosen}
    for r in results:
        r["picks"] = [p for p in r["picks"] if p["market_id"] in ids]
    return ctx, results


def grade(store: Store, log=print):
    graded = 0
    for p in store.open_picks():
        if not p["completed"] or p["home_score"] is None:
            continue
        hs, as_, fin = p["home_score"], p["away_score"], (p.get("finish") or "").upper()
        mt, line, out = p["market_type"], p["line"], p["outcome"]
        team = espn.NICK_TO_ESPN.get(out)
        won = None
        if mt == "moneyline" and team:
            won = (hs > as_) if team == p["home"] else (as_ > hs)
        elif mt == "spreads" and team and line is not None:
            mk = [m for m in store.latest_pm_markets(p["game_id"]) if m["market_id"] == p["market_id"]]
            q = mk[0]["question"] if mk else ""
            line_team = next((a for n, a in espn.NICK_TO_ESPN.items() if n in q), team)
            eff = line if line_team == team else -line
            margin = (hs - as_) if team == p["home"] else (as_ - hs)
            won = margin + eff > 0
        elif mt == "totals" and line is not None:
            won = (hs + as_ > line) if out.lower().startswith("over") else (hs + as_ < line)
        if won is None:
            continue
        pnl = round(p["stake"] * (1 - p["price"]) / p["price"], 2) if won else -p["stake"]
        mk = store.latest_pm_markets(p["game_id"])
        closing = next((m["prices"][m["outcomes"].index(out)] for m in mk if m["market_id"] == p["market_id"] and out in m["outcomes"]), None)
        store.grade_pick(p["pick_id"], "win" if won else "loss", pnl, closing)
        graded += 1
    store.commit()
    log(f"[grade] graded {graded} picks")
    return graded


def performance(store: Store):
    hist = [p for p in store.pick_history() if p["result"]]
    if not hist:
        return {"n": 0}
    n, wins = len(hist), sum(1 for p in hist if p["result"] == "win")
    pnl, staked = sum(p["pnl"] or 0 for p in hist), sum(p["stake"] for p in hist)
    clv = [(p["closing_price"] - p["price"]) for p in hist if p["closing_price"] is not None]
    return {"n": n, "wins": wins, "hit_rate": wins / n, "pnl": pnl, "roi": pnl / staked if staked else 0,
            "avg_clv": sum(clv) / len(clv) if clv else None, "n_clv": len(clv)}


def _hit(ctx, player_id, stat, team, line, n):
    if stat == "sv":
        games = [x for x in ctx["by_goalie"].get(player_id, []) if x["team"] == team]
    else:
        games = [x for x in ctx["by_skater"].get(player_id, []) if x["team"] == team]
    return pl.hit_rate(games, stat, line, n)
