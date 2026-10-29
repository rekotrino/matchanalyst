"""The Odds API provider (https://the-odds-api.com/liveapi/guides/v4/).

Endpoints used:
  GET /v4/sports/{sport}/odds?regions=..&markets=h2h,spreads,totals&bookmakers=..&oddsFormat=decimal
  GET /v4/sports/{sport}/events?dateFormat=iso
  GET /v4/sports/{sport}/events/{eventId}/odds?regions=..&markets=player_points,...&oddsFormat=decimal

Bet365 is served under the uk / au / eu regions (not us). Pinnacle under eu.
Response shape (verified against the public docs):
  [{id, sport_key, commence_time, home_team, away_team,
    bookmakers: [{key, title, last_update, markets: [{key, outcomes: [{name, price, point?, description?}]}]}]}]
For player props, outcome.description is the player name and outcome.name is Over/Under.
"""

import os
import time
from datetime import datetime, timezone

from ..ingest.http import get_json
from .base import Quote

BASE = "https://api.the-odds-api.com/v4"
SPORT_KEYS = {"nba": "basketball_nba", "nhl": "icehockey_nhl"}
DEFAULT_BOOKS = ["bet365", "pinnacle", "draftkings", "fanduel", "betmgm", "betfair_ex_uk", "williamhill", "unibet_eu", "caesars"]
DEFAULT_REGIONS = "uk,eu,us,au"

PROP_MARKETS = {
    "nba": {"player_points": "pts", "player_rebounds": "reb", "player_assists": "ast", "player_threes": "tpm",
            "player_blocks": "blk", "player_steals": "stl", "player_points_rebounds_assists": "pra",
            "player_points_rebounds": "pr", "player_points_assists": "pa", "player_rebounds_assists": "ra"},
    "nhl": {"player_points": "p", "player_assists": "a", "player_shots_on_goal": "sog", "player_blocked_shots": "blk",
            "player_total_saves": "sv", "player_goals": "g"},  # player_goal_scorer_anytime is a yes/no market, handled separately
}


def api_key():
    return os.environ.get("ODDS_API_KEY") or ""


class OddsAPI:
    def __init__(self, sport, key=None, books=None, regions=DEFAULT_REGIONS, log=print):
        self.sport, self.key, self.books, self.regions, self.log = sport, key or api_key(), books or DEFAULT_BOOKS, regions, log
        self.sport_key = SPORT_KEYS[sport]
        self.remaining = None

    def _get(self, path, params):
        params = dict(params, apiKey=self.key)
        data = get_json(f"{BASE}{path}", params)
        return data

    # ---- matching API events to our ESPN games -------------------------------------
    def match_events(self, games, team_names):
        """games: our upcoming games; team_names: {abbr: full name}. Returns {event_id: game}."""
        events = self._get(f"/sports/{self.sport_key}/events", {"dateFormat": "iso"})
        name_to_abbr = {v.lower(): k for k, v in team_names.items()}
        out = {}
        for ev in events:
            h, a = name_to_abbr.get(ev["home_team"].lower()), name_to_abbr.get(ev["away_team"].lower())
            if not h or not a:
                continue
            day = ev["commence_time"][:10]
            for g in games:
                if g["home"] == h and g["away"] == a and abs((datetime.fromisoformat(ev["commence_time"].replace("Z", "+00:00")) -
                                                              datetime.strptime(g["date_utc"], "%Y-%m-%dT%H:%MZ").replace(tzinfo=timezone.utc)).total_seconds()) < 6 * 3600:
                    out[ev["id"]] = g
        self.log(f"[oddsapi] matched {len(out)} of {len(games)} games")
        return out

    # ---- game odds -----------------------------------------------------------------
    def game_quotes(self, games, team_names):
        matched = self.match_events(games, team_names)
        if not matched:
            return []
        data = self._get(f"/sports/{self.sport_key}/odds", {"regions": self.regions, "markets": "h2h,spreads,totals",
                                                             "oddsFormat": "decimal", "bookmakers": ",".join(self.books)})
        name_to_abbr = {v.lower(): k for k, v in team_names.items()}
        quotes = []
        for ev in data:
            g = matched.get(ev["id"])
            if not g:
                continue
            for bk in ev.get("bookmakers", []):
                for m in bk.get("markets", []):
                    for o in m.get("outcomes", []):
                        if m["key"] == "h2h":
                            t = name_to_abbr.get(o["name"].lower())
                            if t:
                                quotes.append(Quote(g["game_id"], bk["key"], "moneyline", t, float(o["price"])))
                        elif m["key"] == "spreads":
                            t = name_to_abbr.get(o["name"].lower())
                            if t:
                                quotes.append(Quote(g["game_id"], bk["key"], "spreads", t, float(o["price"]), line=float(o.get("point")), team=t))
                        elif m["key"] == "totals":
                            quotes.append(Quote(g["game_id"], bk["key"], "totals", o["name"].lower(), float(o["price"]), line=float(o.get("point"))))
        self.log(f"[oddsapi] {len(quotes)} game quotes")
        return quotes

    # ---- player props (one request per event) --------------------------------------
    def prop_quotes(self, games, team_names, max_events=20):
        matched = self.match_events(games, team_names)
        markets = ",".join(PROP_MARKETS[self.sport])
        quotes = []
        for i, (eid, g) in enumerate(list(matched.items())[:max_events]):
            try:
                ev = self._get(f"/sports/{self.sport_key}/events/{eid}/odds",
                               {"regions": self.regions, "markets": markets, "oddsFormat": "decimal", "bookmakers": ",".join(self.books)})
            except Exception as e:  # noqa: BLE001
                self.log(f"  ! props {g['away']}@{g['home']}: {e}")
                continue
            quotes.extend(parse_prop_event(ev, g, self.sport))
            time.sleep(0.2)
        self.log(f"[oddsapi] {len(quotes)} prop quotes")
        return quotes


def parse_prop_event(ev, g, sport):
    quotes = []
    for bk in ev.get("bookmakers", []):
        for m in bk.get("markets", []):
            stat = PROP_MARKETS[sport].get(m["key"])
            if not stat:
                continue
            for o in m.get("outcomes", []):
                side = (o.get("name") or "").lower()
                if side not in ("over", "under") or o.get("point") is None:
                    continue
                quotes.append(Quote(g["game_id"], bk["key"], stat, side, float(o["price"]), line=float(o["point"]),
                                    player=o.get("description")))
    return quotes
