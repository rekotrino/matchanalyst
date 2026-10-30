"""Turn stored Polymarket markets into Quote rows (decimal price = 1 / ask)."""

import re

from .base import Quote

_SPREAD_RE = re.compile(r"Spread:\s*(.+?)\s*\(([-+]?\d+(?:\.\d+)?)\)")
_PROP_RE = re.compile(r"^(.+?):\s*(.+?)\s*O/U\s*([\d.]+)", re.I)


def polymarket_quotes(game, markets, nick_to_abbr, prop_types):
    """markets: store.latest_pm_markets(game_id). prop_types: {market_type: stat}."""
    out = []
    home, away = game["home"], game["away"]
    for m in markets:
        ts = m.get("fetched_at")
        half = (m.get("bid_ask") or 0.0) / 2
        asks = [min(0.999, p + half) for p in m["prices"]]
        mt = m["market_type"]
        liq = m.get("liquidity") or 0
        if mt == "moneyline":
            for o, ask in zip(m["outcomes"], asks):
                t = nick_to_abbr.get(o)
                if t in (home, away) and ask > 0:
                    out.append(Quote(game["game_id"], "polymarket", "moneyline", t, 1 / ask, liquidity=liq, ts=ts))
        elif mt == "spreads":
            mm = _SPREAD_RE.search(m.get("question") or "")
            if not mm:
                continue
            t_line, line = nick_to_abbr.get(mm.group(1)), float(mm.group(2))
            for o, ask in zip(m["outcomes"], asks):
                t = nick_to_abbr.get(o)
                if t in (home, away) and ask > 0:
                    out.append(Quote(game["game_id"], "polymarket", "spreads", t, 1 / ask, line=line if t == t_line else -line, team=t, liquidity=liq, ts=ts))
        elif mt == "totals" and m.get("line") is not None:
            for o, ask in zip(m["outcomes"], asks):
                if ask > 0:
                    out.append(Quote(game["game_id"], "polymarket", "totals", o.lower(), 1 / ask, line=m["line"], liquidity=liq, ts=ts))
        elif mt in prop_types:
            mm = _PROP_RE.match(m.get("question") or "")
            if not mm:
                continue
            player, line = mm.group(1).strip(), m.get("line") if m.get("line") is not None else float(mm.group(3))
            for o, ask in zip(m["outcomes"], asks):
                side = "over" if o.lower() in ("yes", "over") else "under"
                if ask > 0:
                    out.append(Quote(game["game_id"], "polymarket", prop_types[mt], side, 1 / ask, line=line, player=player, liquidity=liq, ts=ts))
    return out
