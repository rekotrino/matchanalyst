"""Team analytics for hockey: per-game lines (goals for/against in regulation+OT,
shots, power play), rolling profiles, and multiplicative attack/defence
ratings (Poisson-style, opponent-adjusted, recency-weighted, shrunk to prior).

Shootout goals are removed: a game that ends in a shootout counts as a tie in
regulation+OT for the ratings, and the official score's extra goal is ignored.
"""

import math
from collections import defaultdict
from datetime import date
from statistics import mean

from ..config import SETTINGS
from ..ingest.espn import TEAM_NAMES

METRICS = ["gf", "ga", "sf", "sa", "ppg", "ppo", "pk_goals", "pk_opp", "hits", "blocked", "pim", "margin"]


def build_game_lines(team_rows):
    by_game = defaultdict(dict)
    for r in team_rows:
        by_game[r["game_id"]][r["team"]] = r
    lines = []
    for gid, teams in by_game.items():
        if len(teams) != 2 or any(t not in TEAM_NAMES for t in teams):
            continue
        (ta, a), (tb, b) = teams.items()
        finish = (a.get("finish") or "").upper()
        for me, opp in ((a, b), (b, a)):
            gf = me["home_score"] if me["is_home"] else me["away_score"]
            ga = me["away_score"] if me["is_home"] else me["home_score"]
            if gf is None or ga is None:
                continue
            if finish == "SO":  # strip the shootout "goal" from the winner
                if gf > ga:
                    gf -= 1
                else:
                    ga -= 1
            lines.append({"game_id": gid, "team": me["team"], "opp": opp["team"], "date": me["date_et"],
                          "is_home": me["is_home"], "season_type": me.get("season_type"), "finish": finish,
                          "gf": gf, "ga": ga, "margin": gf - ga, "sf": me["shots"], "sa": opp["shots"],
                          "ppg": me["ppg"], "ppo": me["ppo"], "pk_goals": opp["ppg"], "pk_opp": opp["ppo"],
                          "hits": me["hits"], "blocked": me["blocked"], "pim": me["pim"],
                          "won": gf > ga, "ot": finish in ("OT", "SO")})
    lines.sort(key=lambda x: (x["date"], x["game_id"]))
    return lines


def league_averages(lines, regular_only=True):
    use = [l for l in lines if l.get("season_type") in (2, None)] if regular_only else lines
    use = use or lines
    if not use:
        return {}
    out = {m: mean(l[m] for l in use) for m in METRICS}
    out["pp_pct"] = sum(l["ppg"] for l in use) / max(1, sum(l["ppo"] for l in use))
    return out


def _avg(ls, m):
    v = [l[m] for l in ls if l.get(m) is not None]
    return mean(v) if v else None


def team_profile(lines, team, cfg=SETTINGS.model):
    mine = [l for l in lines if l["team"] == team]
    prof = {"team": team, "games": len(mine)}
    for name, n in (("last5", 5), ("last10", 10), ("season", None)):
        w = mine[-n:] if n else mine
        b = {m: _avg(w, m) for m in METRICS}
        b["n"] = len(w)
        wins = sum(1 for l in w if l["won"])
        otl = sum(1 for l in w if not l["won"] and l["ot"])
        b["record"] = f"{wins}-{len(w) - wins - otl}-{otl}"
        b["pp_pct"] = sum(l["ppg"] for l in w) / max(1, sum(l["ppo"] for l in w)) if w else None
        b["pk_pct"] = 1 - sum(l["pk_goals"] for l in w) / max(1, sum(l["pk_opp"] for l in w)) if w else None
        b["sh_pct"] = sum(l["gf"] for l in w) / max(1, sum(l["sf"] for l in w)) if w else None
        b["sv_pct"] = 1 - sum(l["ga"] for l in w) / max(1, sum(l["sa"] for l in w)) if w else None
        prof[name] = b
    home = [l for l in mine if l["is_home"]]
    away = [l for l in mine if not l["is_home"]]
    prof["home_net"] = _avg(home, "margin")
    prof["away_net"] = _avg(away, "margin")
    prof["last_games"] = [{"date": l["date"], "opp": l["opp"], "home": l["is_home"], "pts": l["gf"], "opp_pts": l["ga"],
                           "margin": l["margin"], "finish": l["finish"]} for l in mine[-10:]]
    return prof


def strength_ratings(lines, prior=None, cfg=SETTINGS.model, weights=None, prior_weight=8.0):
    """Multiplicative ratings: expected goals for team i vs j =
    lg_gpg * att_i * def_j (* home factor). att/def are log-ratings here (0 = average)."""
    if not lines:
        return {t: dict(v, n=0) for t, v in (prior or {}).items()}
    lg = league_averages(lines)
    lg_g = lg["gf"]
    teams = sorted({l["team"] for l in lines})
    per_team = defaultdict(list)
    for l in lines:
        per_team[l["team"]].append(l)
    w = {}
    for t, ls in per_team.items():
        n = len(ls)
        for i, l in enumerate(ls):
            w[(t, l["game_id"])] = (0.5 ** ((n - 1 - i) / cfg.half_life)) * (weights or {}).get(l["game_id"], 1.0)
    rating = {t: {"att": 0.0, "def": 0.0} for t in teams}
    for t in teams:
        if prior and t in prior:
            rating[t].update({k: prior[t].get(k, 0.0) for k in ("att", "def")})
    for _ in range(6):
        new = {}
        for t in teams:
            ls = per_team[t]
            p = prior.get(t) if prior else None
            pw = prior_weight if p else 0.0
            sw = sum(w[(t, l["game_id"])] for l in ls)
            # observed log-rate adjusted for opponent: log(gf/lg) - def_opp
            att_num = sum(w[(t, l["game_id"])] * (math.log((l["gf"] + 0.5) / (lg_g + 0.5)) - rating[l["opp"]]["def"]) for l in ls)
            def_num = sum(w[(t, l["game_id"])] * (math.log((l["ga"] + 0.5) / (lg_g + 0.5)) - rating[l["opp"]]["att"]) for l in ls)
            denom = sw + pw + cfg.shrink_games
            new[t] = {"att": (att_num + pw * (p["att"] if p else 0.0)) / denom,
                      "def": (def_num + pw * (p["def"] if p else 0.0)) / denom, "n": len(ls), "weight": sw}
        rating = new
    return rating


def rest_info(lines, team, game_date_et):
    mine = [l for l in lines if l["team"] == team and l["date"] < game_date_et]
    if not mine:
        return {"rest_days": None, "b2b": False, "games_last7": 0}
    gd = date.fromisoformat(game_date_et)
    dates = [date.fromisoformat(l["date"]) for l in mine]
    rest = (gd - max(dates)).days
    return {"rest_days": rest, "b2b": rest == 1, "games_last7": sum(1 for d in dates if 0 < (gd - d).days <= 6)}


def flag_issues(prof, lg):
    out = []
    w = prof["last10"]
    if not w["n"] or not lg:
        return [("note", 0, "No completed games in the window yet.")]
    checks = [
        ("gf", 1, True, "Scoring well ({v:+.2f} goals/game vs league)", "Struggling to score ({v:+.2f} goals/game vs league)", 0.4),
        ("ga", 1, False, "Tight defence ({v:+.2f} goals against vs league)", "Leaking goals ({v:+.2f} goals against vs league)", 0.4),
        ("sf", 1, True, "Generating shots ({v:+.1f} per game vs league)", "Not generating shots ({v:+.1f} per game vs league)", 3.0),
        ("sa", 1, False, "Suppressing shots ({v:+.1f} against vs league)", "Giving up shots ({v:+.1f} against vs league)", 3.0),
    ]
    for metric, scale, good_pos, lab_good, lab_bad, thr in checks:
        v = (w[metric] - lg[metric]) * scale
        good = v > 0 if good_pos else v < 0
        if abs(v) >= thr:
            out.append(("strength" if good else "issue", min(3, int(abs(v) / thr)), (lab_good if good else lab_bad).format(v=v)))
    if w.get("pp_pct") is not None and lg.get("pp_pct"):
        d = w["pp_pct"] - lg["pp_pct"]
        if abs(d) >= 0.05:
            out.append(("strength" if d > 0 else "issue", 1, f"Power play {'hot' if d > 0 else 'cold'} ({100 * w['pp_pct']:.0f}% vs league {100 * lg['pp_pct']:.0f}%)"))
    if w.get("sh_pct") is not None:
        if w["sh_pct"] >= 0.125:
            out.append(("note", 0, f"Shooting {100 * w['sh_pct']:.1f}% over the last 10 - likely to cool off"))
        elif w["sh_pct"] <= 0.075:
            out.append(("note", 0, f"Shooting {100 * w['sh_pct']:.1f}% over the last 10 - likely to bounce back"))
    if w.get("sv_pct") is not None:
        if w["sv_pct"] >= 0.925:
            out.append(("note", 0, f"Goaltending at {1000 * w['sv_pct']:.0f} save pct over the last 10 - may regress"))
        elif w["sv_pct"] <= 0.885:
            out.append(("issue", 1, f"Goaltending at {1000 * w['sv_pct']:.0f} save pct over the last 10"))
    out.sort(key=lambda x: (-x[1], x[0]))
    return out
