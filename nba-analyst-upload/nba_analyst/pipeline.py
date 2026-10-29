"""Orchestration: update (ingest) -> analyze (ratings, model, scan, picks)
-> report -> grade."""

import os
import time
from datetime import date, datetime, timedelta

from .analyst import players as pl
from .analyst.ratings import (build_game_lines, flag_issues, league_averages, rest_info,
                              strength_ratings, team_profile)
from .config import SETTINGS
from .ingest import espn, polymarket
from .ingest.polymarket import NICK_TO_ESPN
from .model.game_model import american_to_prob, blend_with_reference, devig_pair, expected_score, p_cover, p_over
from .odds.ladders import build_prop_items, game_lines_table, scan_quotes
from .odds.polymarket_adapter import polymarket_quotes
from .odds.the_odds_api import OddsAPI, api_key
from .odds.csv_import import load_csv
from .scanner import PROP_TYPES, STAT_LABEL, kelly_stake, scan_game, select_daily
from .storage import Store


def current_season_year(today=None):
    """ESPN season.year for the season in progress: 2026-27 -> 2027."""
    today = today or datetime.now(espn.ET).date()
    return today.year + 1 if today.month >= 8 else today.year


# ---------------------------------------------------------------------------
# update
# ---------------------------------------------------------------------------

def update(store: Store, with_prior=True, max_summaries=600, log=print):
    start, end = espn.date_range(SETTINGS.lookback_days, SETTINGS.lookahead_days)
    log(f"[update] schedule {start} .. {end}")
    games, ref = espn.fetch_scoreboard(start, end)
    for g in games:
        store.upsert_game(g)
    for o in ref:
        store.add_reference_odds(o["game_id"], o["provider"], o["spread_home"], o["total"],
                                 o["ml_home"], o["ml_away"])
    store.commit()
    log(f"[update] {len(games)} games in window ({sum(g['completed'] for g in games)} completed)")

    if with_prior:
        prior_year = current_season_year() - 1
        have = store.conn.execute(
            "SELECT COUNT(*) FROM games WHERE season_year=? AND completed=1", (prior_year,)).fetchone()[0]
        if have < 200:
            p_start, p_end = date(prior_year, 2, 1), date(prior_year, 4, 20)
            log(f"[update] pulling last-season prior {p_start} .. {p_end}")
            pg, _ = espn.fetch_scoreboard(p_start, p_end)
            for g in pg:
                if g["season_type"] == 2:  # regular season only
                    store.upsert_game(g)
            store.commit()
            log(f"[update] {len(pg)} prior-season games")

    missing = store.games_missing_stats()
    log(f"[update] box scores to fetch: {len(missing)} (max {max_summaries})")
    n = 0
    for g in missing[:max_summaries]:
        try:
            rows, _, players = espn.fetch_summary(g["game_id"], with_players=True)
        except Exception as e:  # noqa: BLE001
            log(f"  ! summary {g['game_id']} failed: {e}")
            continue
        for r in rows:
            r.setdefault("pts", g["home_score"] if r["is_home"] else g["away_score"])
            store.upsert_team_stats(r)
        for p in players:
            store.upsert_player_stats(p)
        n += 1
        if n % 25 == 0:
            store.commit()
            log(f"  .. {n} box scores")
        time.sleep(0.2)
    store.commit()

    # backfill player box scores for games stored before the player layer existed
    missing_p = [g for g in store.games_missing_player_stats() if g["game_id"] not in {m["game_id"] for m in missing[:max_summaries]}]
    if missing_p:
        log(f"[update] player box scores to backfill: {len(missing_p)}")
        for i, g in enumerate(missing_p[:max_summaries]):
            try:
                _, _, players = espn.fetch_summary(g["game_id"], with_players=True)
                for p in players:
                    store.upsert_player_stats(p)
            except Exception as e:  # noqa: BLE001
                log(f"  ! players {g['game_id']} failed: {e}")
            if (i + 1) % 25 == 0:
                store.commit()
            time.sleep(0.2)
        store.commit()

    today = datetime.now(espn.ET).date()
    upcoming = store.games_upcoming(str(today), str(today + timedelta(days=SETTINGS.lookahead_days)))
    log(f"[update] upcoming games: {len(upcoming)}")
    for g in upcoming:
        try:
            _, inj = espn.fetch_summary(g["game_id"])
            store.replace_injuries(g["game_id"], inj)
        except Exception as e:  # noqa: BLE001
            log(f"  ! injuries {g['game_id']} failed: {e}")
        try:
            slug, markets = polymarket.fetch_game_markets(g["away"], g["home"], g["date_et"])
            if slug:
                store.add_pm_markets(g["game_id"], slug, markets)
                log(f"  {g['away']} @ {g['home']}: {len(markets)} Polymarket markets ({slug})")
            else:
                log(f"  {g['away']} @ {g['home']}: no Polymarket event yet")
        except Exception as e:  # noqa: BLE001
            log(f"  ! polymarket {g['game_id']} failed: {e}")
        time.sleep(0.2)
    store.commit()
    fetch_books(store, upcoming, "nba", espn.TEAM_NAMES, log)


def fetch_books(store, upcoming, sport, team_names, log=print):
    """Other books: The Odds API (if ODDS_API_KEY is set) and manual CSV files."""
    if not upcoming:
        return
    n = 0
    if api_key():
        try:
            api = OddsAPI(sport, log=log)
            qs = api.game_quotes(upcoming, team_names) + api.prop_quotes(upcoming, team_names)
            store.add_quotes(qs)
            n += len(qs)
        except Exception as e:  # noqa: BLE001
            log(f"  ! odds api: {e}")
    else:
        log("[books] ODDS_API_KEY not set - skipping Bet365/Pinnacle/DraftKings feed (see README)")
    import glob
    for path in sorted(glob.glob(os.path.join("data", "odds", f"{sport}_*.csv"))):
        qs = load_csv(path, upcoming)
        store.add_quotes(qs)
        n += len(qs)
        log(f"[books] {path}: {len(qs)} manual quotes")
    store.commit()
    log(f"[books] {n} quotes stored")


# ---------------------------------------------------------------------------
# analyze
# ---------------------------------------------------------------------------

def build_context(store: Store):
    """Ratings, league averages, profiles from everything in the store."""
    season = current_season_year()
    rows = store.team_stats()
    cur_rows = [r for r in rows if _season_of(r) == season]
    prior_rows = [r for r in rows if _season_of(r) == season - 1]
    cur_lines = build_game_lines(cur_rows)
    prior_lines = build_game_lines(prior_rows)

    prior_ratings = strength_ratings(prior_lines, prior=None) if prior_lines else {}
    # preseason games count half
    gtype = {g["game_id"]: g["season_type"] for g in store.games_completed()}
    weights = {gid: (0.5 if t == 1 else 1.0) for gid, t in gtype.items()}
    ratings = strength_ratings(cur_lines, prior=prior_ratings, weights=weights) if cur_lines \
        else {t: dict(v, n=0) for t, v in prior_ratings.items()}
    cur_regular = [l for l in cur_lines if l.get("season_type") == 2]
    lg = league_averages(cur_regular) if len(cur_regular) >= 60 else league_averages(prior_lines)
    games_played = {}
    for l in cur_regular:
        games_played[l["team"]] = games_played.get(l["team"], 0) + 1
    # profiles: current season if the team has played, else last season's tail.
    # Issues are flagged against the averages of the SAME pool of games, so a
    # preseason profile is compared with preseason norms, not regular-season ones.
    profile_lines = cur_lines if cur_lines else prior_lines
    lg_profile = league_averages(profile_lines, regular_only=bool(cur_regular) or not cur_lines)
    # player layer: all stored player games (recency weighting handles the rest)
    all_player_rows = store.player_stats()
    by_player = pl.player_games(all_player_rows)
    cur_player_rows = [r for r in all_player_rows if _season_of(r) == season and r.get("season_type") == 2]
    dvp = pl.defense_vs_position(cur_player_rows) if len(cur_player_rows) > 200 else pl.defense_vs_position(all_player_rows)
    return {"season": season, "ratings": ratings, "prior_ratings": prior_ratings, "lg": lg,
            "lg_profile": lg_profile, "by_player": by_player, "dvp": dvp,
            "cur_lines": cur_lines, "prior_lines": prior_lines, "profile_lines": profile_lines,
            "games_played": games_played,
            "n_cur_games": len(cur_lines) // 2, "n_prior_games": len(prior_lines) // 2}


def _season_of(stat_row):
    # season year derived from date: Aug-Dec -> year+1, Jan-Jul -> year
    y, m = int(stat_row["date_et"][:4]), int(stat_row["date_et"][5:7])
    return y + 1 if m >= 8 else y


def analyze(store: Store, days_ahead=None, log=print):
    ctx = build_context(store)
    log(f"[analyze] season {ctx['season']}: {ctx['n_cur_games']} current games, "
        f"{ctx['n_prior_games']} prior-season games, {len(ctx['ratings'])} teams rated")
    today = datetime.now(espn.ET).date()
    end = today + timedelta(days=days_ahead if days_ahead is not None else SETTINGS.lookahead_days)
    upcoming = store.games_upcoming(str(today), str(end))
    existing = {p["market_id"] for p in store.open_picks()}
    results, day_picks = [], []
    for g in upcoming:
        res = analyze_game(store, g, ctx)
        for p in res["picks"]:
            if p["market_id"] not in existing:
                store.add_pick(dict(p, mode="paper"))
                existing.add(p["market_id"])
        day_picks.extend(res["picks"])
        results.append(res)
    store.commit()
    chosen = select_daily(list(day_picks))
    chosen_ids = {p["market_id"] for p in chosen}
    for r in results:
        r["picks"] = [p for p in r["picks"] if p["market_id"] in chosen_ids]
    return ctx, results


def analyze_game(store: Store, g, ctx, record=True):
    lines, lg, ratings = ctx["profile_lines"], ctx["lg"], ctx["ratings"]
    home, away = g["home"], g["away"]
    rest_h = rest_info(ctx["cur_lines"], home, g["date_et"])
    rest_a = rest_info(ctx["cur_lines"], away, g["date_et"])
    raw = expected_score(home, away, ratings, lg, rest_h, rest_a)
    ref = store.latest_reference_odds(g["game_id"])
    ref_probs = None
    if ref and ref.get("ml_home") is not None:
        ph, pa = devig_pair(american_to_prob(ref["ml_home"]), american_to_prob(ref["ml_away"]))
        ref_probs = {"p_home": ph, "p_away": pa}
    gp = min(ctx["games_played"].get(home, 0), ctx["games_played"].get(away, 0))
    pred = blend_with_reference(raw, ref["spread_home"] if ref else None,
                                ref["total"] if ref else None, gp)
    if record:
        store.add_prediction(g["game_id"], pred, {
            "ratings": {home: ratings.get(home), away: ratings.get(away)},
            "rest": {home: rest_h, away: rest_a}, "lg": lg, "model_only": pred["model_only"],
            "anchor_weight": pred["anchor_weight"], "reference": ref})
    profiles = {t: team_profile(lines, t) for t in (home, away)}
    issues = {t: flag_issues(profiles[t], ctx["lg_profile"]) for t in (home, away)}
    markets = store.latest_pm_markets(g["game_id"])
    injuries = store.latest_injuries(g["game_id"])
    out_names = {t: [i["player"] for i in injuries if i["team"] == t and (i.get("status") or "").lower() == "out"]
                 for t in (home, away)}
    rotations = {t: pl.team_rotation(ctx["by_player"], t) for t in (home, away)}
    player_ctx = {"profiles": rotations, "games": ctx["by_player"], "out": out_names,
                  "dvp": ctx["dvp"], "pace": pred.get("pace"), "lg_pace": lg.get("poss")}
    priced, _pm_picks = scan_game(g, pred, markets, player_ctx=player_ctx)
    # ---- every book on one footing -------------------------------------------------
    quotes = polymarket_quotes(g, markets, NICK_TO_ESPN, PROP_TYPES) + store.latest_quotes(g["game_id"])
    fns = prop_fns(g, ctx, rotations, out_names, player_ctx)
    dtd = {i["player"].lower() for i in injuries if (i.get("status") or "").lower() not in ("out", "")}
    prop_items = build_prop_items(g, fns, quotes, {}, SETTINGS.betting, lambda st: STAT_LABEL.get(st, st), dtd)
    def prob_for(q):
        if q.market == "moneyline":
            return pred["p_home_win"] if q.side == home else (1 - pred["p_home_win"] if q.side == away else None)
        if q.market == "spreads" and q.line is not None and q.side in (home, away):
            return p_cover(pred["exp_margin"] if q.side == home else -pred["exp_margin"], q.line)
        if q.market == "totals" and q.line is not None:
            po = p_over(pred["exp_total"], q.line)
            return po if q.side == "over" else 1 - po
        if q.market in PROP_TYPES.values() and q.player and q.line is not None:
            key = _resolve_player(fns, q.player, q.market)
            if key:
                p_o = fns[key]["p_fn"](q.line)
                return p_o if q.side == "over" else 1 - p_o
        return None
    lines_table = game_lines_table(g, quotes, prob_for)
    _cands, picks = scan_quotes(g, quotes, prob_for, SETTINGS.betting, set(PROP_TYPES.values()), kelly_stake)
    return {"game": g, "pred": pred, "profiles": profiles, "issues": issues, "quotes": quotes,
            "prop_items": prop_items, "lines_table": lines_table, "books": sorted({q.book for q in quotes}),
            "rest": {home: rest_h, away: rest_a}, "injuries": injuries,
            "rotations": rotations, "out_names": out_names, "dvp": {t: ctx["dvp"].get(t) for t in (home, away)},
            "dvp_league": ctx["dvp"].get("_league"), "dvp_games": {t: ctx["dvp"].get("_games", {}).get(t, 0) for t in (home, away)},
            "ref": ref, "ref_probs": ref_probs, "markets": priced, "picks": picks,
            "ratings": {home: ratings.get(home), away: ratings.get(away)},
            "profile_source": ("current season" if any(l.get("season_type") == 2 for l in ctx["cur_lines"])
                               else "preseason" if ctx["cur_lines"] else "last season (no current games yet)")}


# ---------------------------------------------------------------------------
# grade
# ---------------------------------------------------------------------------

def grade(store: Store, log=print):
    graded = 0
    for p in store.open_picks():
        if not p["completed"] or p["home_score"] is None:
            continue
        hs, as_ = p["home_score"], p["away_score"]
        mt, line, out = p["market_type"], p["line"], p["outcome"]
        from .ingest.polymarket import NICK_TO_ESPN  # local import to avoid cycle
        team = NICK_TO_ESPN.get(out)
        won = None
        if mt == "moneyline" and team:
            won = (hs > as_) if team == p["home"] else (as_ > hs)
        elif mt == "spreads" and team and line is not None:
            margin = (hs - as_) if team == p["home"] else (as_ - hs)
            # line stored is for the outcome named in the question; for the other side invert
            mk = [m for m in store.latest_pm_markets(p["game_id"]) if m["market_id"] == p["market_id"]]
            q = mk[0]["question"] if mk else ""
            line_team = next((a for n, a in NICK_TO_ESPN.items() if n in (q or "")), team)
            eff_line = line if line_team == team else -line
            won = margin + eff_line > 0
        elif mt == "totals" and line is not None:
            won = (hs + as_ > line) if out.lower().startswith("over") else (hs + as_ < line)
        elif mt in PROP_TYPES and line is not None:
            mk = [m for m in store.latest_pm_markets(p["game_id"]) if m["market_id"] == p["market_id"]]
            q = mk[0]["question"] if mk else ""
            name = q.split(":")[0].strip().lower()
            stat = PROP_TYPES[mt]
            rows = [r for r in store.player_stats() if r["game_id"] == p["game_id"] and r["player"].lower() == name]
            if not rows or rows[0].get("dnp"):
                continue  # not graded until the player's line is in the store (or void if DNP)
            r0 = rows[0]
            combo = {"pra": ("pts", "reb", "ast"), "pr": ("pts", "reb"), "pa": ("pts", "ast"), "ra": ("reb", "ast")}
            actual = sum(r0.get(k) or 0 for k in combo[stat]) if stat in combo else (r0.get(stat) or 0)
            won = (actual > line) if out.lower() in ("yes", "over") else (actual < line)
        if won is None:
            continue
        buy = p["price"]
        pnl = round(p["stake"] * (1 - buy) / buy, 2) if won else -p["stake"]
        mk = store.latest_pm_markets(p["game_id"])
        closing = next((m["prices"][m["outcomes"].index(out)] for m in mk
                        if m["market_id"] == p["market_id"] and out in m["outcomes"]), None)
        store.grade_pick(p["pick_id"], "win" if won else "loss", pnl, closing)
        graded += 1
    store.commit()
    log(f"[grade] graded {graded} picks")
    return graded


def performance(store: Store):
    hist = [p for p in store.pick_history() if p["result"]]
    if not hist:
        return {"n": 0}
    n = len(hist)
    wins = sum(1 for p in hist if p["result"] == "win")
    pnl = sum(p["pnl"] or 0 for p in hist)
    staked = sum(p["stake"] for p in hist)
    clv = [(p["closing_price"] - p["price"]) for p in hist if p["closing_price"] is not None]
    return {"n": n, "wins": wins, "hit_rate": wins / n, "pnl": pnl, "roi": pnl / staked if staked else 0,
            "avg_clv": sum(clv) / len(clv) if clv else None, "n_clv": len(clv)}


# ---------------------------------------------------------------------------
# prop probability functions (one closure per player x stat), shared by the
# ladders, the multi-book scanner and the dashboard
# ---------------------------------------------------------------------------

def _resolve_player(fns, name, stat):
    n = (name or "").lower().strip()
    if (n, stat) in fns:
        return (n, stat)
    last = n.split()[-1] if n else ""
    c = [k for k in fns if k[1] == stat and k[0].split()[-1] == last]
    return c[0] if len(c) == 1 else None


def prop_fns(g, ctx, rotations, out_names, player_ctx):
    home, away = g["home"], g["away"]
    cfg = SETTINGS.betting
    out = {}
    for team in (home, away):
        opp = away if team == home else home
        outs = {n.lower() for n in out_names.get(team, [])}
        profs = rotations[team]
        for prof in profs:
            if prof["player"].lower() in outs:
                continue
            games = [x for x in ctx["by_player"].get(prof["player_id"], []) if x["team"] == team]
            if len(games) < cfg.prop_min_games:
                continue
            bump = pl.usage_bump_for(profs, out_names.get(team, []), prof["player"]) or 0.0
            for stat in ("pts", "reb", "ast", "tpm", "stl", "blk", "pra", "pr", "pa", "ra"):
                proj = pl.projection(games, stat, minutes_bump=bump)
                if not proj or proj.get("preseason_only"):
                    continue
                mf, minfo = pl.matchup_factor(player_ctx.get("dvp"), opp, prof.get("position"), stat)
                pf = pl.pace_factor(player_ctx.get("pace"), player_ctx.get("lg_pace"))
                adj = dict(proj, mean=proj["mean"] * mf * pf)

                def p_fn(line, gm=games, st=stat, pj=adj):
                    pm = pl.p_over(pj, line)
                    hr, hn = pl.hit_rate(gm, st, line, 10)
                    w = 0.65 if hn >= 8 else (0.8 if hn >= 5 else 1.0)
                    return w * pm + (1 - w) * hr if hr is not None else pm

                def hit_fn(line, n, gm=games, st=stat):
                    return pl.hit_rate(gm, st, line, n)
                recent = [x.get(stat) or 0 for x in prof["recent"]]
                out[(prof["player"].lower(), stat)] = {
                    "p_fn": p_fn, "hit_fn": hit_fn, "player_id": prof["player_id"], "player": prof["player"], "team": team, "opp": opp,
                    "position": prof.get("position"), "proj": adj, "proj_raw": proj["mean"], "recent": recent,
                    "l10": prof["last10"].get(stat), "l5": prof["last5"].get(stat), "season": prof["season"].get(stat),
                    "l10_sd": prof["last10"].get(stat + "_sd"), "matchup": mf, "matchup_info": minfo, "pace_factor": pf, "bump": bump,
                    "p10": round(max(0.0, adj["mean"] - 1.28 * adj["sd"]), 1),
                }
    return out
