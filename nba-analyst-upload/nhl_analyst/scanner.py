"""Price Polymarket NHL markets from the score distribution and select paper
picks (same rules as the NBA scanner). Player-prop market types are handled
if Polymarket ever lists them (none observed as of 2026-10-29)."""

import re

from .analyst import players as pl
from .config import SETTINGS
from .ingest.espn import NICK_TO_ESPN
from .model import p_cover, p_total_over

SIDE_TYPES = {"moneyline", "spreads"}
TOTAL_TYPES = {"totals"}
PROP_TYPES = {"goals": "g", "assists": "a", "points": "p", "shots_on_goal": "sog", "shots": "sog",
              "blocked_shots": "blk", "hits": "hits", "saves": "sv"}
_SPREAD_RE = re.compile(r"Spread:\s*(.+?)\s*\(([-+]?\d+(?:\.\d+)?)\)")
_PROP_RE = re.compile(r"^(.+?):\s*(.+?)\s*O/U\s*([\d.]+)", re.I)


def _team_of(outcome, home, away):
    abbr = NICK_TO_ESPN.get(outcome)
    if abbr in (home, away):
        return abbr
    return outcome.upper() if outcome.upper() in (home, away) else None


def price_market(m, S, home, away, player_ctx=None):
    mtype, outs, prices = m["market_type"], m["outcomes"], m["prices"]
    half = (m.get("bid_ask") or 0.0) / 2
    probs = None
    if mtype == "moneyline":
        t0 = _team_of(outs[0], home, away)
        if t0 is None:
            return None
        p = S["p_home_win"] if t0 == home else 1 - S["p_home_win"]
        probs = [p, 1 - p]
    elif mtype == "spreads":
        mm = _SPREAD_RE.search(m.get("question") or "")
        line = m.get("line")
        if mm:
            line = float(mm.group(2)) if line is None else line
            t_line = _team_of(mm.group(1), home, away)
        else:
            t_line = _team_of(outs[0], home, away)
        if t_line is None or line is None:
            return None
        p_t = p_cover(S, t_line == home, line)
        probs = [p_t, 1 - p_t] if _team_of(outs[0], home, away) == t_line else [1 - p_t, p_t]
    elif mtype == "totals":
        if m.get("line") is None:
            return None
        po = p_total_over(S, m["line"])
        probs = [po, 1 - po] if outs[0].lower().startswith("over") else [1 - po, po]
    elif mtype in PROP_TYPES and player_ctx:
        return price_prop(m, player_ctx, home, away)
    else:
        return None
    rows = []
    for o, p, mid in zip(outs, probs, prices):
        buy = min(0.999, mid + half)
        rows.append({"outcome": o, "our_prob": p, "price": mid, "buy_price": buy, "edge": p - buy})
    return rows


def price_prop(m, ctx, home, away):
    stat = PROP_TYPES.get(m["market_type"])
    mm = _PROP_RE.match(m.get("question") or "")
    if not stat or not mm:
        return None
    name, line = mm.group(1).strip(), m.get("line") if m.get("line") is not None else float(mm.group(3))
    for team in (home, away):
        lad = ctx["ladders"].get(team, {})
        key = next((k for k in lad if k[0].lower() == name.lower() and k[1] == stat), None)
        if not key:
            continue
        entry = lad[key]
        p = entry["p_fn"](line)
        half = (m.get("bid_ask") or 0.0) / 2
        rows = []
        for o, mid in zip(m["outcomes"], m["prices"]):
            is_over = o.lower() in ("yes", "over")
            prob = p if is_over else 1 - p
            buy = min(0.999, mid + half)
            rows.append({"outcome": o, "our_prob": prob, "price": mid, "buy_price": buy, "edge": prob - buy,
                         "player": key[0], "stat": stat, "line": line})
        return rows
    return None


def kelly_stake(p, buy, cfg=SETTINGS.betting):
    b = (1 - buy) / buy
    f = max(0.0, (p * b - (1 - p)) / b) * cfg.kelly_fraction if b > 0 else 0.0
    return round(min(f, cfg.max_stake_pct) * cfg.bankroll, 2)


def scan_game(game, S, markets, cfg=SETTINGS.betting, player_ctx=None):
    home, away = game["home"], game["away"]
    priced, cands = [], []
    for m in markets:
        rows = price_market(m, S, home, away, player_ctx)
        priced.append({"market": m, "rows": rows})
        if not rows:
            continue
        for r in rows:
            if (m.get("liquidity") or 0) < cfg.min_liquidity or r["edge"] < cfg.edge_threshold or r["our_prob"] < cfg.prob_floor:
                continue
            cands.append({"game_id": game["game_id"], "matchup": f"{away} @ {home}", "market_id": m["market_id"],
                          "market_type": m["market_type"], "line": m.get("line"), "question": m.get("question"),
                          "outcome": r["outcome"], "our_prob": r["our_prob"], "price": r["price"], "buy_price": r["buy_price"],
                          "edge": r["edge"], "liquidity": m.get("liquidity"), "stake": kelly_stake(r["our_prob"], r["buy_price"], cfg)})
    cands.sort(key=lambda c: (-c["our_prob"], -c["edge"]))
    picks, used_side, used_total = [], False, False
    for c in cands:
        if len(picks) >= cfg.max_picks_per_game:
            break
        if c["market_type"] in SIDE_TYPES:
            if used_side:
                continue
            used_side = True
        elif c["market_type"] in TOTAL_TYPES:
            if used_total:
                continue
            used_total = True
        picks.append(c)
    return priced, picks


def select_daily(picks, cfg=SETTINGS.betting):
    picks.sort(key=lambda c: (-c["our_prob"], -c["edge"]))
    return picks[:cfg.max_picks_per_day]
