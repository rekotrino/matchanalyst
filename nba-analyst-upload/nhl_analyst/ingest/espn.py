"""ESPN public NHL API (verified 2026-10-29).
  scoreboard?dates=YYYYMMDD   events[].competitions[0].competitors, odds, status.type.altDetail (OT / SO)
  summary?event=<id>          boxscore.teams[].statistics (per game), boxscore.players[].statistics[]
                              blocks forwards / defenses / goalies with keys[] aligned to athletes[].stats[]
"""

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from nba_analyst.ingest.http import get_json

BASE = "https://site.api.espn.com/apis/site/v2/sports/hockey/nhl"
ET = ZoneInfo("America/New_York")

TEAM_NAMES = {
    "ANA": "Anaheim Ducks", "BOS": "Boston Bruins", "BUF": "Buffalo Sabres", "CGY": "Calgary Flames",
    "CAR": "Carolina Hurricanes", "CHI": "Chicago Blackhawks", "COL": "Colorado Avalanche",
    "CBJ": "Columbus Blue Jackets", "DAL": "Dallas Stars", "DET": "Detroit Red Wings", "EDM": "Edmonton Oilers",
    "FLA": "Florida Panthers", "LA": "Los Angeles Kings", "MIN": "Minnesota Wild", "MTL": "Montreal Canadiens",
    "NSH": "Nashville Predators", "NJ": "New Jersey Devils", "NYI": "New York Islanders", "NYR": "New York Rangers",
    "OTT": "Ottawa Senators", "PHI": "Philadelphia Flyers", "PIT": "Pittsburgh Penguins", "SJ": "San Jose Sharks",
    "SEA": "Seattle Kraken", "STL": "St. Louis Blues", "TB": "Tampa Bay Lightning", "TOR": "Toronto Maple Leafs",
    "UTA": "Utah Mammoth", "VAN": "Vancouver Canucks", "VGK": "Vegas Golden Knights", "WSH": "Washington Capitals",
    "WPG": "Winnipeg Jets",
}
# Polymarket nickname -> ESPN abbreviation (moneyline / spread outcomes)
NICK_TO_ESPN = {
    "Ducks": "ANA", "Bruins": "BOS", "Sabres": "BUF", "Flames": "CGY", "Hurricanes": "CAR", "Blackhawks": "CHI",
    "Avalanche": "COL", "Blue Jackets": "CBJ", "Stars": "DAL", "Red Wings": "DET", "Oilers": "EDM", "Panthers": "FLA",
    "Kings": "LA", "Wild": "MIN", "Canadiens": "MTL", "Predators": "NSH", "Devils": "NJ", "Islanders": "NYI",
    "Rangers": "NYR", "Senators": "OTT", "Flyers": "PHI", "Penguins": "PIT", "Sharks": "SJ", "Kraken": "SEA",
    "Blues": "STL", "Lightning": "TB", "Maple Leafs": "TOR", "Mammoth": "UTA", "Canucks": "VAN",
    "Golden Knights": "VGK", "Capitals": "WSH", "Jets": "WPG",
}
# ESPN abbreviation -> Polymarket slug codes (observed: tb, utah, mon, cal, lak, sj, wsh, nj?)
ESPN_TO_PM = {"TB": ["tb", "tbl"], "UTA": ["utah", "uta"], "MTL": ["mon", "mtl"], "CGY": ["cal", "cgy"],
              "LA": ["lak", "la"], "SJ": ["sj", "sjs"], "NJ": ["nj", "njd"], "WSH": ["wsh", "was"]}


def _et_date(iso_utc):
    return datetime.strptime(iso_utc, "%Y-%m-%dT%H:%MZ").replace(tzinfo=ZoneInfo("UTC")).astimezone(ET).strftime("%Y-%m-%d")


def _toi(v):
    if not v or ":" not in str(v):
        return None
    m, s = str(v).split(":")
    return round(int(m) + int(s) / 60, 2)


def parse_events(events):
    games, ref = [], []
    for ev in events:
        comp = ev["competitions"][0]
        home = away = None
        for c in comp["competitors"]:
            d = {"abbr": c["team"]["abbreviation"], "score": int(c["score"]) if str(c.get("score", "")).isdigit() else None}
            if c["homeAway"] == "home":
                home = d
            else:
                away = d
        if not home or not away:
            continue
        st = ev["status"]["type"]
        done = bool(st.get("completed"))
        games.append({"game_id": ev["id"], "date_utc": ev["date"], "date_et": _et_date(ev["date"]),
                      "season_year": (ev.get("season") or {}).get("year"), "season_type": (ev.get("season") or {}).get("type"),
                      "home": home["abbr"], "away": away["abbr"], "home_score": home["score"] if done else None,
                      "away_score": away["score"] if done else None, "completed": int(done), "status": st.get("name"),
                      "finish": (st.get("altDetail") or "") if done else None})
        for o in comp.get("odds", []) or []:
            ml = o.get("moneyline") or {}
            def _ml(side):
                try:
                    return int(str(ml[side]["close"]["odds"]).replace("+", "")) if ml.get(side) else None
                except (KeyError, ValueError, TypeError):
                    return None
            ref.append({"game_id": ev["id"], "provider": (o.get("provider") or {}).get("name", "?"),
                        "spread_home": o.get("spread"), "total": o.get("overUnder"), "ml_home": _ml("home"), "ml_away": _ml("away")})
    return games, ref


def fetch_scoreboard(start, end):
    events = []
    d = start
    while d <= end:
        try:
            events.extend(get_json(f"{BASE}/scoreboard", {"dates": f"{d:%Y%m%d}", "limit": 100}).get("events", []) or [])
        except Exception:  # noqa: BLE001
            pass
        d += timedelta(days=1)
    return parse_events(events)


def fetch_summary(game_id):
    """Returns dict(team=[...], skaters=[...], goalies=[...], injuries=[...], odds=..., finish=str)."""
    j = get_json(f"{BASE}/summary", {"event": game_id})
    team = []
    for t in (j.get("boxscore") or {}).get("teams", []) or []:
        o = {s.get("name"): s.get("displayValue") for s in t.get("statistics", []) or []}
        if not o:
            continue
        def n(k):
            try:
                return int(float(o.get(k, 0) or 0))
            except ValueError:
                return 0
        team.append({"game_id": game_id, "team": t["team"]["abbreviation"], "is_home": int(t.get("homeAway") == "home"),
                     "shots": n("shotsTotal"), "ppg": n("powerPlayGoals"), "ppo": n("powerPlayOpportunities"), "hits": n("hits"),
                     "blocked": n("blockedShots"), "fow": n("faceoffsWon"), "giveaways": n("giveaways"),
                     "takeaways": n("takeaways"), "pim": n("penaltyMinutes")})
    skaters, goalies = [], []
    for t in (j.get("boxscore") or {}).get("players", []) or []:
        abbr = (t.get("team") or {}).get("abbreviation")
        for b in t.get("statistics", []) or []:
            keys, blk = b.get("keys") or [], (b.get("name") or b.get("type") or "").lower()
            for a in b.get("athletes", []) or []:
                ath = a.get("athlete") or {}
                if not ath.get("id") or not ath.get("displayName") or a.get("didNotPlay"):
                    continue
                o = dict(zip(keys, a.get("stats") or []))
                toi = _toi(o.get("timeOnIce"))
                if toi is None:
                    continue
                def n(k):
                    try:
                        return int(float(o.get(k, 0) or 0))
                    except ValueError:
                        return 0
                if blk == "goalies":
                    goalies.append({"game_id": game_id, "team": abbr, "player_id": str(ath["id"]), "player": ath["displayName"],
                                    "toi": toi, "ga": n("goalsAgainst"), "sa": n("shotsAgainst"), "sv": n("saves")})
                elif blk in ("forwards", "defenses"):
                    skaters.append({"game_id": game_id, "team": abbr, "player_id": str(ath["id"]), "player": ath["displayName"],
                                    "position": (ath.get("position") or {}).get("abbreviation") or ("D" if blk == "defenses" else "F"),
                                    "toi": toi, "g": n("goals"), "a": n("assists"), "sog": n("shotsTotal"), "missed": n("shotsMissed"),
                                    "blk": n("blockedShots"), "hits": n("hits"), "pp_toi": _toi(o.get("powerPlayTimeOnIce")) or 0.0,
                                    "plus_minus": n("plusMinus"), "pim": n("penaltyMinutes")})
    inj = []
    for t in j.get("injuries", []) or []:
        for i in t.get("injuries", []) or []:
            inj.append({"team": (t.get("team") or {}).get("abbreviation"), "player": (i.get("athlete") or {}).get("displayName", "?"),
                        "status": i.get("status"), "details": ((i.get("details") or {}).get("type") or "")})
    st = (((j.get("header") or {}).get("competitions") or [{}])[0].get("status") or {}).get("type") or {}
    return {"team": team, "skaters": skaters, "goalies": goalies, "injuries": inj, "finish": st.get("altDetail") or ""}


def date_range(lookback_days, lookahead_days):
    today = datetime.now(ET).date()
    return today - timedelta(days=lookback_days), today + timedelta(days=lookahead_days)
