"""Polymarket NHL events: slug nhl-{away}-{home}-{YYYY-MM-DD} with codes such as
tb, utah, mon, cal, lak, sj. Verified 2026-10-29: markets are moneyline, totals
(ladder) and spreads (puck lines); no player props listed for any NHL event."""

from nba_analyst.ingest.http import get_json
from nba_analyst.ingest.polymarket import BASE, _loads

from .espn import ESPN_TO_PM


def slug_candidates(away, home, date_et):
    aw = ESPN_TO_PM.get(away, [away.lower()])
    hm = ESPN_TO_PM.get(home, [home.lower()])
    return [f"nhl-{a}-{h}-{date_et}" for a in aw for h in hm]


def fetch_game_markets(away, home, date_et):
    for slug in slug_candidates(away, home, date_et):
        try:
            events = get_json(f"{BASE}/events", {"slug": slug})
        except Exception:  # noqa: BLE001
            events = []
        if not events:
            continue
        out = []
        for m in events[0].get("markets", []) or []:
            outcomes = _loads(m.get("outcomes"), [])
            prices = [float(p) for p in _loads(m.get("outcomePrices"), [])]
            if len(outcomes) != 2 or len(prices) != 2:
                continue
            line = m.get("line")
            out.append({"market_id": str(m.get("id")), "question": m.get("question"),
                        "market_type": m.get("sportsMarketType") or "other", "line": float(line) if line is not None else None,
                        "outcomes": outcomes, "prices": prices, "token_ids": _loads(m.get("clobTokenIds"), None),
                        "liquidity": float(m.get("liquidity") or 0.0), "bid_ask": float(m.get("spread") or 0.0),
                        "active": bool(m.get("active", True)) and not bool(m.get("closed", False))})
        return slug, out
    return None, []
