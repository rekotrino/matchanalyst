"""Market scanner: price every Polymarket market on a game from our model,
compare with the market price, keep only markets where our edge clears the
threshold, and THEN prefer the highest-probability bet.

Selection rule (in order):
  1. market is active, modelled, and liquid enough
  2. edge = our_prob - effective buy price >= edge_threshold
  3. our_prob >= prob_floor (we are a favourites-first system)
  4. rank remaining by our_prob desc, then edge desc
  5. correlation guard: at most one side market (moneyline/spread) and one
     total market per game; max picks per game and per day
"""

import re

from .config import SETTINGS
from .ingest.polymarket import NICK_TO_ESPN
from .model.game_model import p_cover, p_over, p_team_over

from .analyst import players as pl

SIDE_TYPES = {"moneyline", "spreads"}
TOTAL_TYPES = {"totals"}
PROP_TYPES = {"points": "pts", "rebounds": "reb", "assists": "ast", "threes": "tpm", "steals": "stl",
              "blocks": "blk", "pra": "pra", "points_rebounds_assists": "pra", "points_rebounds": "pr",
              "points_assists": "pa", "rebounds_assists": "ra"}
STAT_LABEL = {"pts": "Points", "reb": "Rebounds", "ast": "Assists", "tpm": "3-pointers", "stl": "Steals",
              "blk": "Blocks", "pra": "Pts+Reb+Ast", "pr": "Pts+Reb", "pa": "Pts+Ast", "ra": "Reb+Ast"}
MODELLED = SIDE_TYPES | TOTAL_TYPES | {"team_totals"} | set(PROP_TYPES)

_SPREAD_RE = re.compile(r"Spread:\s*(.+?)\s*\(([-+]?\d+(?:\.\d+)?)\)")
_PROP_RE = re.compile(r"^(.+?):\s*(.+?)\s*O/U\s*([\d.]+)", re.I)


def price_prop(m, player_ctx, home, away):
    """Price a player prop from the player projection. player_ctx:
    {"profiles": {team: [profile]}, "games": {player_id: [rows]}, "out": {team: [names]},
     "dvp": defense-vs-position table, "pace": expected game pace, "lg_pace": league pace}
    Returns rows for Yes/No (over/under) with projection and context attached."""
    if not player_ctx:
        return None
    stat = PROP_TYPES.get(m["market_type"])
    mm = _PROP_RE.match(m.get("question") or "")
    if not stat or not mm:
        return None
    name, line = mm.group(1).strip(), m.get("line")
    line = float(mm.group(3)) if line is None else line
    for team in (home, away):
        profs = player_ctx["profiles"].get(team, [])
        prof = pl.find_player(profs, name)
        if not prof:
            continue
        opp = away if team == home else home
        outs = player_ctx["out"].get(team, [])
        bump = pl.usage_bump_for(profs, outs, prof["player"])
        if bump is None:
            return None  # player listed Out
        games = player_ctx["games"].get(prof["player_id"], [])
        games = [g for g in games if g["team"] == team]
        if len(games) < SETTINGS.betting.prop_min_games:
            return None
        proj = pl.projection(games, stat, minutes_bump=bump)
        if not proj or proj.get("preseason_only"):
            return None  # no regular-season minutes to project from yet
        mf, minfo = pl.matchup_factor(player_ctx.get("dvp"), opp, prof.get("position"), stat)
        pf = pl.pace_factor(player_ctx.get("pace"), player_ctx.get("lg_pace"))
        adj = dict(proj, mean=proj["mean"] * mf * pf)
        p_model = pl.p_over(adj, line)
        hr10, hn = pl.hit_rate(games, stat, line, 10)
        hr5, _ = pl.hit_rate(games, stat, line, 5)
        hrs, ns = pl.hit_rate(games, stat, line, 0)
        # counting stats are skewed; blend the model with what actually happened
        # over the last 10 games so a hot projection can't run away from reality
        w = 0.65 if hn >= 8 else (0.8 if hn >= 5 else 1.0)
        p = w * p_model + (1 - w) * hr10 if hr10 is not None else p_model
        outs_, prices = m["outcomes"], m["prices"]
        half = (m.get("bid_ask") or 0.0) / 2
        rows = []
        for o, mid in zip(outs_, prices):
            is_over = o.lower() in ("yes", "over")
            prob = p if is_over else 1 - p
            buy = min(0.999, mid + half)
            rows.append({"outcome": o, "our_prob": prob, "price": mid, "buy_price": buy, "edge": prob - buy,
                         "proj": adj["mean"], "proj_raw": proj["mean"], "proj_sd": adj["sd"], "proj_min": adj["minutes"],
                         "hit_rate": hr10, "hit_n": hn, "hit5": hr5, "hit_season": hrs, "hit_season_n": ns,
                         "player": prof["player"], "player_id": prof["player_id"], "team": team, "opp": opp,
                         "position": prof.get("position"), "stat": stat, "stat_label": STAT_LABEL.get(stat, stat),
                         "line": line, "bump": bump, "matchup": mf, "matchup_info": minfo, "pace_factor": pf,
                         "p_model": p_model, "l10_avg": prof["last10"].get(stat), "l5_avg": prof["last5"].get(stat),
                         "season_avg": prof["season"].get(stat), "l10_sd": prof["last10"].get(stat + "_sd")})
        return rows
    return None


def _team_of(outcome, home, away):
    abbr = NICK_TO_ESPN.get(outcome)
    if abbr in (home, away):
        return abbr
    if outcome.upper() in (home, away):
        return outcome.upper()
    return None


def price_market(m, pred, home, away, player_ctx=None):
    """Returns list of {outcome, our_prob, price, buy_price} for each outcome,
    or None if the market type isn't modelled."""
    mtype = m["market_type"]
    outs, prices = m["outcomes"], m["prices"]
    half_spread = (m.get("bid_ask") or 0.0) / 2
    probs = None
    if mtype in PROP_TYPES:
        return price_prop(m, player_ctx, home, away)
    if mtype == "moneyline":
        t0 = _team_of(outs[0], home, away)
        if t0 is None:
            return None
        p_home = pred["p_home_win"]
        probs = [p_home if t0 == home else 1 - p_home]
        probs.append(1 - probs[0])
    elif mtype == "spreads":
        line = m.get("line")
        mm = _SPREAD_RE.search(m.get("question") or "")
        if mm:
            team_nick, qline = mm.group(1), float(mm.group(2))
            line = qline if line is None else line
        else:
            team_nick = outs[0]
        t_line = _team_of(team_nick, home, away)
        if t_line is None or line is None:
            return None
        exp_margin_t = pred["exp_margin"] if t_line == home else -pred["exp_margin"]
        p_t = p_cover(exp_margin_t, line)
        t0 = _team_of(outs[0], home, away)
        probs = [p_t, 1 - p_t] if t0 == t_line else [1 - p_t, p_t]
    elif mtype == "totals":
        line = m.get("line")
        if line is None:
            return None
        po = p_over(pred["exp_total"], line)
        probs = [po, 1 - po] if outs[0].lower().startswith("over") else [1 - po, po]
    elif mtype == "team_totals":
        line = m.get("line")
        q = m.get("question") or ""
        team = next((a for n, a in NICK_TO_ESPN.items() if n in q and a in (home, away)), None)
        if line is None or team is None:
            return None
        exp = pred["exp_home_pts"] if team == home else pred["exp_away_pts"]
        po = p_team_over(exp, line)
        probs = [po, 1 - po] if outs[0].lower().startswith("over") else [1 - po, po]
    else:
        return None
    rows = []
    for o, p, mid in zip(outs, probs, prices):
        buy = min(0.999, mid + half_spread)
        rows.append({"outcome": o, "our_prob": p, "price": mid, "buy_price": buy,
                     "edge": p - buy})
    return rows


def kelly_stake(p, buy_price, cfg=SETTINGS.betting):
    b = (1 - buy_price) / buy_price          # net odds on a binary share
    f = (p * b - (1 - p)) / b if b > 0 else 0.0
    f = max(0.0, f) * cfg.kelly_fraction
    return round(min(f, cfg.max_stake_pct) * cfg.bankroll, 2)


def scan_game(game, pred, markets, cfg=SETTINGS.betting, player_ctx=None):
    """Returns (all_priced_rows, candidate_picks) for one game."""
    home, away = game["home"], game["away"]
    priced, candidates = [], []
    for m in markets:
        rows = price_market(m, pred, home, away, player_ctx)
        if rows is None:
            priced.append({"market": m, "rows": None})
            continue
        priced.append({"market": m, "rows": rows})
        if not m.get("active", True) and m.get("active") is not None:
            continue
        for r in rows:
            if (m.get("liquidity") or 0) < cfg.min_liquidity:
                continue
            if r["edge"] < cfg.edge_threshold or r["our_prob"] < cfg.prob_floor:
                continue
            candidates.append({
                "game_id": game["game_id"], "matchup": f"{away} @ {home}",
                "market_id": m["market_id"], "market_type": m["market_type"],
                "line": m.get("line"), "question": m.get("question"),
                "outcome": r["outcome"], "our_prob": r["our_prob"], "price": r["price"],
                "buy_price": r["buy_price"], "edge": r["edge"],
                "liquidity": m.get("liquidity"), "stake": kelly_stake(r["our_prob"], r["buy_price"], cfg),
                "proj": r.get("proj"), "hit_rate": r.get("hit_rate"), "player": r.get("player"),
            })
    candidates.sort(key=lambda c: (-c["our_prob"], -c["edge"]))
    picks, used_side, used_total, used_players = [], False, False, set()
    for c in candidates:
        if len(picks) >= cfg.max_picks_per_game:
            break
        if c["market_type"] in SIDE_TYPES:
            if used_side:
                continue
            used_side = True
        elif c["market_type"] in TOTAL_TYPES | {"team_totals"}:
            if used_total:
                continue
            used_total = True
        elif c["market_type"] in PROP_TYPES:
            key = c["question"].split(":")[0]
            if key in used_players:
                continue  # one prop per player
            used_players.add(key)
        picks.append(c)
    return priced, picks


def select_daily(all_picks, cfg=SETTINGS.betting):
    all_picks.sort(key=lambda c: (-c["our_prob"], -c["edge"]))
    return all_picks[:cfg.max_picks_per_day]
