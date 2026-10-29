"""Analyst engine: turns raw team box scores into per-game efficiency numbers,
rolling-window team profiles (last 5 / last 10 / season), opponent-adjusted
strength ratings, and a list of flagged strengths and issues per team.

Everything here is deterministic. No LLM, no opinions - just the numbers the
report and the model are built on.
"""

from collections import defaultdict
from datetime import date
from statistics import mean

from ..config import SETTINGS
from ..ingest.espn import TEAM_NAMES


# ---------------------------------------------------------------------------
# Per-game efficiency lines
# ---------------------------------------------------------------------------

def _poss(s):
    return s["fga"] - s["oreb"] + s["tov"] + 0.44 * s["fta"]


def build_game_lines(stat_rows):
    """stat_rows: team_game_stats joined with games. Returns list of per-team
    per-game dicts with efficiency metrics, in chronological order."""
    by_game = defaultdict(dict)
    for r in stat_rows:
        by_game[r["game_id"]][r["team"]] = r
    lines = []
    for gid, teams in by_game.items():
        if len(teams) != 2 or any(t not in TEAM_NAMES for t in teams):
            continue  # skip All-Star, Rising Stars, exhibition opponents
        (ta, a), (tb, b) = teams.items()
        if any(a.get(k) is None or b.get(k) is None
               for k in ("fga", "oreb", "tov", "fta", "fgm", "tpm", "tpa", "dreb", "pts")):
            continue
        poss = (_poss(a) + _poss(b)) / 2
        if poss <= 0:
            continue
        for me, opp in ((a, b), (b, a)):
            lines.append({
                "game_id": gid, "team": me["team"], "opp": opp["team"],
                "date": me["date_et"], "is_home": me["is_home"],
                "season_type": me.get("season_type"),
                "pts": me["pts"], "opp_pts": opp["pts"], "margin": me["pts"] - opp["pts"],
                "poss": poss,
                "ortg": 100 * me["pts"] / poss, "drtg": 100 * opp["pts"] / poss,
                "efg": (me["fgm"] + 0.5 * me["tpm"]) / max(me["fga"], 1),
                "opp_efg": (opp["fgm"] + 0.5 * opp["tpm"]) / max(opp["fga"], 1),
                "tov_pct": me["tov"] / poss, "opp_tov_pct": opp["tov"] / poss,
                "oreb_pct": me["oreb"] / max(me["oreb"] + opp["dreb"], 1),
                "dreb_pct": me["dreb"] / max(me["dreb"] + opp["oreb"], 1),
                "ftr": me["fta"] / max(me["fga"], 1), "opp_ftr": opp["fta"] / max(opp["fga"], 1),
                "tp_rate": me["tpa"] / max(me["fga"], 1),
                "tp_pct": me["tpm"] / max(me["tpa"], 1), "opp_tp_pct": opp["tpm"] / max(opp["tpa"], 1),
                "paint": me.get("pts_paint"), "opp_paint": opp.get("pts_paint"),
                "fast_break": me.get("fast_break"), "opp_fast_break": opp.get("fast_break"),
            })
    lines.sort(key=lambda x: (x["date"], x["game_id"]))
    return lines


# ---------------------------------------------------------------------------
# Team profiles (rolling windows) and league averages
# ---------------------------------------------------------------------------

METRICS = ["ortg", "drtg", "poss", "efg", "opp_efg", "tov_pct", "opp_tov_pct", "oreb_pct",
           "dreb_pct", "ftr", "opp_ftr", "tp_rate", "tp_pct", "opp_tp_pct", "margin"]


def league_averages(lines, regular_only=True):
    use = [l for l in lines if l.get("season_type") in (2, None)] if regular_only else lines
    use = use or lines
    return {m: mean(l[m] for l in use) for m in METRICS} if use else {}


def _avg(ls, m):
    vals = [l[m] for l in ls if l.get(m) is not None]
    return mean(vals) if vals else None


def team_profile(lines, team, cfg=SETTINGS.model):
    mine = [l for l in lines if l["team"] == team]
    prof = {"team": team, "games": len(mine)}
    for name, n in (("last5", cfg.recent_window), ("last10", cfg.form_window), ("season", None)):
        window = mine[-n:] if n else mine
        prof[name] = {m: _avg(window, m) for m in METRICS}
        prof[name]["n"] = len(window)
        prof[name]["record"] = f"{sum(1 for l in window if l['margin'] > 0)}-{sum(1 for l in window if l['margin'] < 0)}"
    home = [l for l in mine if l["is_home"]]
    away = [l for l in mine if not l["is_home"]]
    prof["home_net"] = _avg(home, "margin")
    prof["away_net"] = _avg(away, "margin")
    prof["last_games"] = [{"date": l["date"], "opp": l["opp"], "home": l["is_home"],
                           "pts": l["pts"], "opp_pts": l["opp_pts"], "margin": l["margin"]}
                          for l in mine[-cfg.form_window:]]
    return prof


# ---------------------------------------------------------------------------
# Opponent-adjusted strength ratings with recency weighting + shrinkage
# ---------------------------------------------------------------------------

def strength_ratings(lines, prior=None, cfg=SETTINGS.model, half_life=10.0, prior_weight=8.0,
                     weights=None):
    """Returns {team: {off, def, pace, n}} where off/def are points per 100
    possessions relative to league average (def: negative = better), opponent
    adjusted, recency weighted, shrunk toward the prior (last season) and then
    toward league average.

    prior: {team: {off, def, pace}} from an earlier period (e.g. last season).
    weights: optional {game_id: weight multiplier} (e.g. 0.5 for preseason).
    """
    if not lines:
        return {t: dict(v, n=0) for t, v in (prior or {}).items()}
    lg = league_averages(lines)
    teams = sorted({l["team"] for l in lines})
    # recency weights per team
    per_team = defaultdict(list)
    for l in lines:
        per_team[l["team"]].append(l)
    w = {}
    for t, ls in per_team.items():
        n = len(ls)
        for i, l in enumerate(ls):
            games_ago = n - 1 - i
            w[(t, l["game_id"])] = (0.5 ** (games_ago / half_life)) * (weights or {}).get(l["game_id"], 1.0)

    rating = {t: {"off": 0.0, "def": 0.0, "pace": 0.0} for t in teams}
    for t in teams:
        if prior and t in prior:
            rating[t].update({k: prior[t].get(k, 0.0) for k in ("off", "def", "pace")})
    shrink0 = 3.0  # extra pull toward league average
    for _ in range(4):  # iterate opponent adjustment
        new = {}
        for t in teams:
            ls = per_team[t]
            sw = sum(w[(t, l["game_id"])] for l in ls)
            p = prior.get(t) if prior else None
            pw = prior_weight if p else 0.0
            off_num = sum(w[(t, l["game_id"])] * (l["ortg"] - lg["ortg"] - rating[l["opp"]]["def"]) for l in ls)
            def_num = sum(w[(t, l["game_id"])] * (l["drtg"] - lg["drtg"] - rating[l["opp"]]["off"]) for l in ls)
            pace_num = sum(w[(t, l["game_id"])] * (l["poss"] - lg["poss"] - rating[l["opp"]]["pace"]) for l in ls)
            denom = sw + pw + shrink0
            new[t] = {
                "off": (off_num + pw * (p["off"] if p else 0.0)) / denom,
                "def": (def_num + pw * (p["def"] if p else 0.0)) / denom,
                "pace": (pace_num + pw * (p["pace"] if p else 0.0)) / denom,
                "n": len(ls), "weight": sw,
            }
        rating = new
    return rating


# ---------------------------------------------------------------------------
# Rest / schedule
# ---------------------------------------------------------------------------

def rest_info(lines, team, game_date_et):
    mine = [l for l in lines if l["team"] == team and l["date"] < game_date_et]
    if not mine:
        return {"rest_days": None, "b2b": False, "three_in_four": False, "games_last7": 0}
    gd = date.fromisoformat(game_date_et)
    dates = [date.fromisoformat(l["date"]) for l in mine]
    last = max(dates)
    rest = (gd - last).days
    recent4 = sum(1 for d in dates if 0 < (gd - d).days <= 3)
    recent7 = sum(1 for d in dates if 0 < (gd - d).days <= 6)
    return {"rest_days": rest, "b2b": rest == 1, "three_in_four": recent4 >= 2,
            "games_last7": recent7}


# ---------------------------------------------------------------------------
# Strengths and issues (what the report highlights)
# ---------------------------------------------------------------------------

def flag_issues(prof, lg):
    """Compare a team's last-10 profile to league averages and return
    [(kind, severity, text)] where kind is 'strength' or 'issue'."""
    out = []
    w = prof["last10"]
    r = prof["last5"]
    if not w["n"] or not lg:
        return [("note", 0, "No completed games in the window yet; profile falls back to prior ratings.")]

    def d(metric, scale=1.0):
        return (w[metric] - lg[metric]) * scale

    checks = [
        # metric, scale, good_if_positive, label_good, label_bad, threshold
        ("ortg", 1, True, "Elite offense ({v:+.1f} pts/100 vs league)", "Offense struggling ({v:+.1f} pts/100 vs league)", 3.0),
        ("drtg", 1, False, "Stingy defense ({v:+.1f} pts/100 allowed vs league)", "Defense leaking ({v:+.1f} pts/100 allowed vs league)", 3.0),
        ("efg", 100, True, "Shooting hot (eFG {v:+.1f} pts vs league)", "Shooting cold (eFG {v:+.1f} pts vs league)", 2.0),
        ("opp_efg", 100, False, "Holding opponents to poor shots (opp eFG {v:+.1f})", "Giving up easy shots (opp eFG {v:+.1f})", 2.0),
        ("tov_pct", 100, False, "Takes care of the ball (TOV% {v:+.1f})", "Turnover-prone (TOV% {v:+.1f})", 1.5),
        ("opp_tov_pct", 100, True, "Forces turnovers (opp TOV% {v:+.1f})", "Doesn't force turnovers (opp TOV% {v:+.1f})", 1.5),
        ("oreb_pct", 100, True, "Strong on the offensive glass (OREB% {v:+.1f})", "Weak on the offensive glass (OREB% {v:+.1f})", 2.5),
        ("dreb_pct", 100, True, "Closes possessions on the defensive glass (DREB% {v:+.1f})", "Gives up second chances (DREB% {v:+.1f})", 2.5),
        ("opp_tp_pct", 100, False, "Opponents cold from three (opp 3P% {v:+.1f}) - may regress", "Opponents hot from three (opp 3P% {v:+.1f}) - may regress", 3.0),
        ("tp_pct", 100, True, "Hot from three (3P% {v:+.1f}) - may regress", "Cold from three (3P% {v:+.1f}) - may regress", 3.0),
        ("ftr", 100, True, "Gets to the line a lot (FT rate {v:+.1f})", "Rarely gets to the line (FT rate {v:+.1f})", 4.0),
    ]
    for metric, scale, good_pos, lab_good, lab_bad, thr in checks:
        v = d(metric, scale)
        good = v > 0 if good_pos else v < 0
        if abs(v) >= thr:
            sev = min(3, int(abs(v) / thr))
            out.append(("strength" if good else "issue", sev, (lab_good if good else lab_bad).format(v=v)))
    # momentum: last 5 vs last 10
    if r["n"] >= 3 and w["n"] >= 6:
        delta = r["margin"] - w["margin"]
        if delta >= 4:
            out.append(("strength", 1, f"Trending up: last-5 margin {r['margin']:+.1f} vs last-10 {w['margin']:+.1f}"))
        elif delta <= -4:
            out.append(("issue", 1, f"Trending down: last-5 margin {r['margin']:+.1f} vs last-10 {w['margin']:+.1f}"))
    # pace style
    pv = w["poss"] - lg["poss"]
    if abs(pv) >= 2.5:
        out.append(("note", 0, f"{'Fast' if pv > 0 else 'Slow'}-paced ({pv:+.1f} possessions vs league)"))
    # home/away gap
    if prof["home_net"] is not None and prof["away_net"] is not None:
        gap = prof["home_net"] - prof["away_net"]
        if gap >= 8:
            out.append(("note", 0, f"Much better at home ({prof['home_net']:+.1f}) than away ({prof['away_net']:+.1f})"))
        elif gap <= -5:
            out.append(("note", 0, f"Travels well: away margin {prof['away_net']:+.1f} vs home {prof['home_net']:+.1f}"))
    out.sort(key=lambda x: (-x[1], x[0]))
    return out
