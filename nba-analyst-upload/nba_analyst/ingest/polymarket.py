"""Polymarket Gamma API (public, no key). Game events live at
  https://gamma-api.polymarket.com/events?slug=nba-{away}-{home}-{YYYY-MM-DD}
with standard 3-letter team codes (was, uta, nop, gsw, nyk, sas, bkn, phx...).

Each market: question, sportsMarketType (moneyline | spreads | totals | points |
rebounds | assists | basketball_*), line, outcomes (JSON string), outcomePrices
(JSON string, midpoint prices), clobTokenIds (JSON string), liquidity.
Verified 2026-10-19.
"""

import json

from .http import get_json

BASE = "https://gamma-api.polymarket.com"

# ESPN abbreviation -> Polymarket slug code candidates (first is the expected one)
ESPN_TO_PM = {
    "GS": ["gsw", "gs"], "NO": ["nop", "no"], "NY": ["nyk", "ny"], "SA": ["sas", "sa"],
    "UTAH": ["uta", "utah"], "WSH": ["was", "wsh"], "PHX": ["phx", "pho"],
    "BKN": ["bkn", "brk"], "CHA": ["cha", "cho"],
}

# Polymarket moneyline/spread outcomes are nicknames ("Hawks"); map nickname -> ESPN abbr
NICK_TO_ESPN = {
    "Hawks": "ATL", "Celtics": "BOS", "Nets": "BKN", "Hornets": "CHA", "Bulls": "CHI",
    "Cavaliers": "CLE", "Mavericks": "DAL", "Nuggets": "DEN", "Pistons": "DET",
    "Warriors": "GS", "Rockets": "HOU", "Pacers": "IND", "Clippers": "LAC", "Lakers": "LAL",
    "Grizzlies": "MEM", "Heat": "MIA", "Bucks": "MIL", "Timberwolves": "MIN",
    "Pelicans": "NO", "Knicks": "NY", "Thunder": "OKC", "Magic": "ORL", "76ers": "PHI",
    "Suns": "PHX", "Trail Blazers": "POR", "Kings": "SAC", "Spurs": "SA", "Raptors": "TOR",
    "Jazz": "UTAH", "Wizards": "WSH",
}


def slug_candidates(away, home, date_et):
    aw = ESPN_TO_PM.get(away, [away.lower()])
    hm = ESPN_TO_PM.get(home, [home.lower()])
    return [f"nba-{a}-{h}-{date_et}" for a in aw for h in hm]


def _loads(v, default):
    if v is None:
        return default
    if isinstance(v, (list, dict)):
        return v
    try:
        return json.loads(v)
    except (TypeError, ValueError):
        return default


def fetch_game_markets(away, home, date_et):
    """Returns (slug, markets) or (None, []) when Polymarket has no event."""
    for slug in slug_candidates(away, home, date_et):
        try:
            events = get_json(f"{BASE}/events", {"slug": slug})
        except Exception:  # noqa: BLE001
            events = []
        if not events:
            continue
        ev = events[0]
        out = []
        for m in ev.get("markets", []) or []:
            outcomes = _loads(m.get("outcomes"), [])
            prices = [float(p) for p in _loads(m.get("outcomePrices"), [])]
            if len(outcomes) != 2 or len(prices) != 2:
                continue
            mtype = m.get("sportsMarketType") or "other"
            line = m.get("line")
            out.append({
                "market_id": str(m.get("id")), "question": m.get("question"),
                "market_type": mtype, "line": float(line) if line is not None else None,
                "outcomes": outcomes, "prices": prices,
                "token_ids": _loads(m.get("clobTokenIds"), None),
                "liquidity": float(m.get("liquidity") or 0.0),
                "bid_ask": float(m.get("spread") or 0.0),
                "active": bool(m.get("active", True)) and not bool(m.get("closed", False)),
            })
        return slug, out
    return None, []
