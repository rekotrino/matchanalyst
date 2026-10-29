"""Skater and goalie analytics: per-60 rates, projected ice time, Poisson
projections for goals / assists / points / shots / blocks / hits, and a saves
projection for goalies. Probabilities are blended with the player's real hit
rate on the line over the last 10 games."""

import math
from collections import defaultdict
from statistics import mean, pstdev

from ..config import SETTINGS

SKATER_STATS = ("g", "a", "p", "sog", "blk", "hits")
STAT_LABEL = {"g": "Goals", "a": "Assists", "p": "Points", "sog": "Shots on goal", "blk": "Blocked shots",
              "hits": "Hits", "sv": "Saves"}
HALF_LIFE = 10.0
SHRINK = 8.0


def _w(n, i):
    return 0.5 ** ((n - 1 - i) / HALF_LIFE)


def skater_games(rows):
    by = defaultdict(list)
    for r in rows:
        if not r.get("toi") or r["toi"] <= 0:
            continue
        r = dict(r)
        r["p"] = (r.get("g") or 0) + (r.get("a") or 0)
        by[r["player_id"]].append(r)
    return by


def goalie_games(rows):
    by = defaultdict(list)
    for r in rows:
        if not r.get("toi") or r["toi"] <= 0:
            continue
        by[r["player_id"]].append(dict(r))
    return by


def profile(games):
    if not games:
        return None
    g = games[-1]
    out = {"player_id": g["player_id"], "player": g["player"], "team": g["team"], "position": g.get("position"),
           "games": len(games), "last_date": g["date_et"]}
    for name, n in (("last5", 5), ("last10", 10), ("season", None)):
        win = games[-n:] if n else games
        b = {"n": len(win)}
        for s in ("toi", "pp_toi") + SKATER_STATS:
            vals = [x.get(s) or 0 for x in win]
            b[s] = mean(vals)
            b[s + "_sd"] = pstdev(vals) if len(vals) > 1 else None
        toi = sum(x.get("toi") or 0 for x in win)
        for s in SKATER_STATS:
            b[s + "_60"] = 60 * sum(x.get(s) or 0 for x in win) / toi if toi else 0
        b["sh_pct"] = sum(x.get("g") or 0 for x in win) / max(1, sum(x.get("sog") or 0 for x in win))
        out[name] = b
    out["trend"] = {k: out["last5"][k] - out["last10"][k] for k in ("toi", "g", "a", "p", "sog")}
    out["recent"] = [{"date": x["date_et"], "opp": x["away"] if x["team"] == x["home"] else x["home"],
                      "home": x["team"] == x["home"], "toi": x.get("toi"), "g": x.get("g"), "a": x.get("a"), "p": x["p"],
                      "sog": x.get("sog"), "blk": x.get("blk"), "hits": x.get("hits"), "pp_toi": x.get("pp_toi")}
                     for x in games[-10:]]
    return out


def hit_rate(games, stat, line, n=10):
    win = games[-n:] if n else games
    if not win:
        return None, 0
    vals = [x.get(stat) or 0 for x in win]
    return sum(1 for v in vals if v > line) / len(vals), len(vals)


def projection(games, stat, factor=1.0):
    """Poisson mean for a skater stat in the next game."""
    if not games:
        return None
    n = len(games)
    ws = [_w(n, i) for i in range(n)]
    toi = [x.get("toi") or 0 for x in games]
    vals = [x.get(stat) or 0 for x in games]
    rate_recent = sum(w * v for w, v in zip(ws, vals)) / max(1e-6, sum(w * t for w, t in zip(ws, toi)))  # per minute
    rate_season = sum(vals) / max(1e-6, sum(toi))
    eff = sum(ws)
    rate = (rate_recent * eff + rate_season * SHRINK) / (eff + SHRINK)
    toi_recent = sum(w * t for w, t in zip(ws, toi)) / eff
    toi_proj = (toi_recent * eff + mean(toi) * SHRINK) / (eff + SHRINK)
    lam = rate * toi_proj * factor
    return {"mean": lam, "minutes": toi_proj, "rate60": rate * 60, "n": n,
            "sd": math.sqrt(lam) if stat != "sog" else max(1.2, math.sqrt(lam) * 1.1)}


def poisson_cdf(lam, k):
    if k < 0:
        return 0.0
    s, term = 0.0, math.exp(-lam)
    for i in range(int(k) + 1):
        s += term
        term *= lam / (i + 1)
    return min(1.0, s)


def p_over(proj, line):
    """P(X > line) for Poisson X. Lines are x.5 so floor(line) is the max failing count."""
    return 1.0 - poisson_cdf(proj["mean"], math.floor(line))


def blended_p_over(games, stat, line, proj):
    pm = p_over(proj, line)
    hr, hn = hit_rate(games, stat, line, 10)
    w = 0.65 if hn >= 8 else (0.8 if hn >= 5 else 1.0)
    return (w * pm + (1 - w) * hr if hr is not None else pm), pm, hr, hn


def goalie_profile(games):
    if not games:
        return None
    g = games[-1]
    out = {"player_id": g["player_id"], "player": g["player"], "team": g["team"], "position": "G", "games": len(games),
           "last_date": g["date_et"]}
    for name, n in (("last5", 5), ("last10", 10), ("season", None)):
        win = games[-n:] if n else games
        sa, sv, ga = sum(x["sa"] for x in win), sum(x["sv"] for x in win), sum(x["ga"] for x in win)
        out[name] = {"n": len(win), "sa": sa / len(win), "sv": sv / len(win), "ga": ga / len(win),
                     "sv_pct": sv / sa if sa else None, "toi": mean(x["toi"] for x in win),
                     "sv_sd": pstdev([x["sv"] for x in win]) if len(win) > 1 else None}
    out["recent"] = [{"date": x["date_et"], "opp": x["away"] if x["team"] == x["home"] else x["home"],
                      "home": x["team"] == x["home"], "toi": x["toi"], "sa": x["sa"], "sv": x["sv"], "ga": x["ga"]}
                     for x in games[-10:]]
    return out


def saves_projection(goalie_games_, exp_shots_against, league_sv=0.905):
    """Saves ~ Normal(mean = expected shots against * save%, sd ~ 6). Save% is the
    goalie's season rate shrunk toward league."""
    sa = sum(x["sa"] for x in goalie_games_)
    sv = sum(x["sv"] for x in goalie_games_)
    svp = (sv + league_sv * 300) / (sa + 300) if sa else league_sv
    mean_sv = exp_shots_against * svp
    return {"mean": mean_sv, "sd": max(5.0, math.sqrt(exp_shots_against) * 1.1), "minutes": 60, "sv_pct": svp,
            "exp_sa": exp_shots_against, "n": len(goalie_games_)}


def team_rotation(by_player, team, top_n=14, min_games=1):
    rows = []
    for games in by_player.values():
        if games[-1]["team"] != team:
            continue  # moved teams
        g = [x for x in games if x["team"] == team]
        if len(g) >= min_games:
            p = profile(g)
            if p:
                rows.append(p)
    rows.sort(key=lambda p: (p["games"] < 3, -p["last5"]["toi"]))
    return rows[:top_n]


def team_goalies(by_goalie, team):
    rows = []
    for games in by_goalie.values():
        if games[-1]["team"] != team:
            continue
        g = [x for x in games if x["team"] == team]
        if g:
            rows.append(goalie_profile(g))
    rows.sort(key=lambda p: -p["games"])
    return rows


def likely_starter(goalies, game_date_et):
    """Heuristic: the goalie with more starts who did not play yesterday."""
    from datetime import date
    if not goalies:
        return None
    gd = date.fromisoformat(game_date_et)
    cands = []
    for g in goalies:
        last = date.fromisoformat(g["last_date"])
        played_yesterday = (gd - last).days == 1
        cands.append((played_yesterday, -g["games"], g))
    cands.sort(key=lambda x: (x[0], x[1]))
    return cands[0][2]


def find_player(profiles, name):
    n = name.lower().strip()
    for p in profiles:
        if p["player"].lower() == n:
            return p
    last = n.split()[-1] if n else ""
    c = [p for p in profiles if p["player"].lower().split()[-1] == last]
    return c[0] if len(c) == 1 else None
