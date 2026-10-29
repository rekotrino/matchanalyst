"""ESPN public site API: schedule, scores, team box scores, injuries, and a
DraftKings reference line. No API key needed.

Endpoints (verified 2026-10-19):
  scoreboard?dates=YYYYMMDD or YYYYMMDD-YYYYMMDD
  summary?event=<id>   -> boxscore.teams[].statistics, injuries[], odds
"""

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from .http import get_json

BASE = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba"
ET = ZoneInfo("America/New_York")

# ESPN abbreviations used as canonical team keys throughout the project.
TEAM_NAMES = {
    "ATL": "Atlanta Hawks", "BOS": "Boston Celtics", "BKN": "Brooklyn Nets",
    "CHA": "Charlotte Hornets", "CHI": "Chicago Bulls", "CLE": "Cleveland Cavaliers",
    "DAL": "Dallas Mavericks", "DEN": "Denver Nuggets", "DET": "Detroit Pistons",
    "GS": "Golden State Warriors", "HOU": "Houston Rockets", "IND": "Indiana Pacers",
    "LAC": "LA Clippers", "LAL": "Los Angeles Lakers", "MEM": "Memphis Grizzlies",
    "MIA": "Miami Heat", "MIL": "Milwaukee Bucks", "MIN": "Minnesota Timberwolves",
    "NO": "New Orleans Pelicans", "NY": "New York Knicks", "OKC": "Oklahoma City Thunder",
    "ORL": "Orlando Magic", "PHI": "Philadelphia 76ers", "PHX": "Phoenix Suns",
    "POR": "Portland Trail Blazers", "SAC": "Sacramento Kings", "SA": "San Antonio Spurs",
    "TOR": "Toronto Raptors", "UTAH": "Utah Jazz", "WSH": "Washington Wizards",
}


def _et_date(iso_utc):
    dt = datetime.strptime(iso_utc, "%Y-%m-%dT%H:%MZ").replace(tzinfo=ZoneInfo("UTC"))
    return dt.astimezone(ET).strftime("%Y-%m-%d")


def fetch_scoreboard(start, end, per_day_fallback=True):
    """start/end: date objects (ET calendar days). Tries a single range request
    first; if ESPN rejects it or returns nothing, falls back to one request per
    day."""
    dates = f"{start:%Y%m%d}-{end:%Y%m%d}" if start != end else f"{start:%Y%m%d}"
    events = []
    try:
        data = get_json(f"{BASE}/scoreboard", {"dates": dates, "limit": 1000})
        events = data.get("events", []) or []
    except Exception:  # noqa: BLE001
        events = []
    if not events and per_day_fallback and start != end:
        d = start
        while d <= end:
            try:
                data = get_json(f"{BASE}/scoreboard", {"dates": f"{d:%Y%m%d}", "limit": 100})
                events.extend(data.get("events", []) or [])
            except Exception:  # noqa: BLE001
                pass
            d += timedelta(days=1)
    return parse_events(events)


def parse_events(events):
    games, ref_odds = [], []
    for ev in events:
        comp = ev["competitions"][0]
        home = away = None
        for c in comp["competitors"]:
            d = {"abbr": c["team"]["abbreviation"],
                 "score": int(c["score"]) if str(c.get("score", "")).isdigit() else None}
            if c["homeAway"] == "home":
                home = d
            else:
                away = d
        if not home or not away:
            continue
        completed = bool(ev["status"]["type"].get("completed"))
        season = ev.get("season", {})
        games.append({
            "game_id": ev["id"], "date_utc": ev["date"], "date_et": _et_date(ev["date"]),
            "season_year": season.get("year"), "season_type": season.get("type"),
            "home": home["abbr"], "away": away["abbr"],
            "home_score": home["score"] if completed else None,
            "away_score": away["score"] if completed else None,
            "completed": int(completed), "status": ev["status"]["type"].get("name"),
        })
        for o in comp.get("odds", []) or []:
            ml = o.get("moneyline") or {}
            def _ml(side):
                try:
                    return int(str(ml[side]["close"]["odds"]).replace("+", "")) if ml.get(side) else None
                except (KeyError, ValueError, TypeError):
                    return None
            ref_odds.append({
                "game_id": ev["id"], "provider": (o.get("provider") or {}).get("name", "?"),
                "spread_home": o.get("spread"), "total": o.get("overUnder"),
                "ml_home": _ml("home"), "ml_away": _ml("away"),
            })
    return games, ref_odds


STAT_MAP = {
    "fieldGoalsMade-fieldGoalsAttempted": ("fgm", "fga"),
    "threePointFieldGoalsMade-threePointFieldGoalsAttempted": ("tpm", "tpa"),
    "freeThrowsMade-freeThrowsAttempted": ("ftm", "fta"),
    "totalRebounds": "reb", "offensiveRebounds": "oreb", "defensiveRebounds": "dreb",
    "assists": "ast", "steals": "stl", "blocks": "blk", "totalTurnovers": "tov",
    "pointsInPaint": "pts_paint", "fastBreakPoints": "fast_break",
    "turnoverPoints": "pts_off_tov", "fouls": "fouls",
}


PLAYER_KEYS = {"minutes": "min", "points": "pts", "rebounds": "reb", "assists": "ast", "turnovers": "tov",
               "steals": "stl", "blocks": "blk", "offensiveRebounds": "oreb", "defensiveRebounds": "dreb",
               "fouls": "pf", "plusMinus": "plus_minus",
               "fieldGoalsMade-fieldGoalsAttempted": ("fgm", "fga"),
               "threePointFieldGoalsMade-threePointFieldGoalsAttempted": ("tpm", "tpa"),
               "freeThrowsMade-freeThrowsAttempted": ("ftm", "fta")}


def parse_player_box(data, game_id):
    """boxscore.players[].statistics[0]: keys[] aligned with athletes[].stats[].
    Verified 2026-10-19. Returns list of player rows."""
    rows = []
    for t in (data.get("boxscore") or {}).get("players", []) or []:
        abbr = (t.get("team") or {}).get("abbreviation")
        for block in t.get("statistics", []) or []:
            keys = block.get("keys") or []
            for a in block.get("athletes", []) or []:
                ath = a.get("athlete") or {}
                if not ath.get("id") or not ath.get("displayName"):
                    continue  # placeholder lines
                row = {"game_id": game_id, "team": abbr, "player_id": str(ath.get("id")),
                       "player": ath.get("displayName", "?"),
                       "position": ((ath.get("position") or {}).get("abbreviation")),
                       "starter": int(bool(a.get("starter"))), "dnp": int(bool(a.get("didNotPlay")))}
                stats = a.get("stats") or []
                if row["dnp"] or not stats:
                    row["dnp"] = 1
                    rows.append(row)
                    continue
                for k, v in zip(keys, stats):
                    col = PLAYER_KEYS.get(k)
                    if not col:
                        continue
                    v = str(v).strip()
                    try:
                        if isinstance(col, tuple):
                            m, n = v.split("-")
                            row[col[0]], row[col[1]] = int(m), int(n)
                        elif col == "min":
                            row["min"] = float(v.split(":")[0]) + (float(v.split(":")[1]) / 60 if ":" in v else 0)
                        else:
                            row[col] = int(v.replace("+", ""))
                    except ValueError:
                        pass
                rows.append(row)
    return rows


def fetch_summary(game_id, with_players=False):
    """Returns (team_stats_rows, injuries_rows[, player_rows]).
    Stats rows empty for games not yet played."""
    data = get_json(f"{BASE}/summary", {"event": game_id})
    rows = []
    for t in (data.get("boxscore") or {}).get("teams", []):
        if not t.get("statistics"):
            continue
        row = {"game_id": game_id, "team": t["team"]["abbreviation"],
               "is_home": int(t.get("homeAway") == "home")}
        for s in t["statistics"]:
            key = STAT_MAP.get(s.get("name"))
            if not key:
                continue
            val = str(s.get("displayValue", "")).strip()
            if isinstance(key, tuple):
                try:
                    a, b = val.split("-")
                    row[key[0]], row[key[1]] = int(a), int(b)
                except ValueError:
                    pass
            else:
                try:
                    row[key] = int(float(val))
                except ValueError:
                    pass
        rows.append(row)
    # points: from header competitors scores if present
    comps = ((data.get("header") or {}).get("competitions") or [{}])[0].get("competitors", [])
    scores = {c["team"]["abbreviation"]: c.get("score") for c in comps if c.get("team")}
    for row in rows:
        sc = scores.get(row["team"])
        if sc is not None and str(sc).isdigit():
            row["pts"] = int(sc)
        elif "fgm" in row and "tpm" in row and "ftm" in row:
            row["pts"] = 2 * (row["fgm"] - row["tpm"]) + 3 * row["tpm"] + row["ftm"]
    inj = []
    for t in data.get("injuries", []) or []:
        abbr = (t.get("team") or {}).get("abbreviation")
        for i in t.get("injuries", []) or []:
            inj.append({"team": abbr, "player": (i.get("athlete") or {}).get("displayName", "?"),
                        "status": i.get("status"),
                        "details": ((i.get("details") or {}).get("type") or "") + " " +
                                   ((i.get("details") or {}).get("returnDate") or "")})
    if with_players:
        return rows, inj, parse_player_box(data, game_id)
    return rows, inj


def date_range(lookback_days, lookahead_days):
    today = datetime.now(ET).date()
    return today - timedelta(days=lookback_days), today + timedelta(days=lookahead_days)
