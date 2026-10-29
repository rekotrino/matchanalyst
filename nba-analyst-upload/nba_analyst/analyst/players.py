"""Player analytics: rolling production (points, rebounds, assists, minutes),
trends, consistency, hit rates against a line, and a projection used to price
player-prop markets (points / rebounds / assists over-under).

Projection logic (deterministic, no LLM):
  rate      = recency-weighted per-minute rate, shrunk toward the season rate
  minutes   = recency-weighted minutes, shrunk toward season minutes, bumped
              when a high-usage teammate is listed Out
  mean      = rate * minutes
  sd        = empirical game-to-game sd (floored), inflated when the sample is small
  P(over L) = 1 - Phi((L - mean) / sd)   with a continuity bump for integer-ish stats
Everything here is a Phase 1 starting point; calibration comes from grading.
"""

import math
from collections import defaultdict
from statistics import mean, pstdev

from ..model.game_model import norm_cdf

STATS = ("pts", "reb", "ast")
PRA = ("pts", "reb", "ast")

# noise floors: a player's game-to-game sd is rarely below these
SD_FLOOR = {"pts": 4.5, "reb": 2.0, "ast": 1.6, "pra": 6.0, "min": 4.0, "tpm": 1.1, "stl": 0.9, "blk": 0.8, "pr": 5.0, "pa": 5.0, "ra": 3.0}
# how many games of history count as "enough"; fewer -> wider distribution
FULL_SAMPLE = 10
HALF_LIFE = 8.0        # games; recency weight = 0.5 ** (games_ago / HALF_LIFE)
SHRINK_GAMES = 6.0     # recent rate pulled toward season rate with this pseudo-count


def _w(n, i):
    return 0.5 ** ((n - 1 - i) / HALF_LIFE)


def _wavg(vals, weights):
    s = sum(weights)
    return sum(v * w for v, w in zip(vals, weights)) / s if s else None


def player_games(rows):
    """Group player stat rows (already chronological) by player_id, dropping DNPs
    and zero-minute lines."""
    by = defaultdict(list)
    for r in rows:
        if r.get("dnp") or r.get("min") is None or r["min"] <= 0:
            continue
        r = dict(r)
        r["pra"] = (r.get("pts") or 0) + (r.get("reb") or 0) + (r.get("ast") or 0)
        r["pr"] = (r.get("pts") or 0) + (r.get("reb") or 0)
        r["pa"] = (r.get("pts") or 0) + (r.get("ast") or 0)
        r["ra"] = (r.get("reb") or 0) + (r.get("ast") or 0)
        for k in ("tpm", "stl", "blk"):
            r[k] = r.get(k) or 0
        by[r["player_id"]].append(r)
    return by


def profile(games, windows=(5, 10)):
    """Rolling averages, per-minute rates, trends and consistency for one player."""
    if not games:
        return None
    g = games[-1]
    out = {"player_id": g["player_id"], "player": g["player"], "team": g["team"],
           "position": g.get("position"), "games": len(games),
           "starter_rate": mean(x.get("starter") or 0 for x in games[-10:]),
           "last_date": g["date_et"]}
    for name, n in [(f"last{w}", w) for w in windows] + [("season", None)]:
        win = games[-n:] if n else games
        block = {"n": len(win)}
        for s in ("min", "pts", "reb", "ast", "pra", "tpm", "stl", "blk"):
            vals = [x.get(s) or 0 for x in win]
            block[s] = mean(vals)
            block[s + "_sd"] = pstdev(vals) if len(vals) > 1 else None
        block["pts_per_min"] = block["pts"] / block["min"] if block["min"] else 0
        block["reb_per_min"] = block["reb"] / block["min"] if block["min"] else 0
        block["ast_per_min"] = block["ast"] / block["min"] if block["min"] else 0
        fga = sum(x.get("fga") or 0 for x in win)
        fta = sum(x.get("fta") or 0 for x in win)
        tov = sum(x.get("tov") or 0 for x in win)
        mins = sum(x.get("min") or 0 for x in win)
        block["usage_proxy"] = (fga + 0.44 * fta + tov) / mins * 36 if mins else 0  # per-36 possessions used
        block["ts_pct"] = (sum(x.get("pts") or 0 for x in win) / (2 * (fga + 0.44 * fta))) if (fga + 0.44 * fta) else None
        out[name] = block
    l5, l10, s = out["last5"], out["last10"], out["season"]
    out["trend"] = {k: l5[k] - l10[k] for k in ("min", "pts", "reb", "ast")}
    out["recent"] = [{"date": x["date_et"], "opp": x["away"] if x["team"] == x["home"] else x["home"],
                      "home": x["team"] == x["home"], "min": x.get("min"), "pts": x.get("pts"),
                      "reb": x.get("reb"), "ast": x.get("ast"), "tpm": x.get("tpm"), "stl": x.get("stl"),
                      "blk": x.get("blk"), "pra": x.get("pra"), "starter": x.get("starter"),
                      "season_type": x.get("season_type")}
                     for x in games[-10:]]
    return out


def hit_rate(games, stat, line, n=10):
    win = games[-n:] if n else games
    if not win:
        return None, 0
    vals = [x.get(stat) or 0 for x in win]
    return sum(1 for v in vals if v > line) / len(vals), len(vals)


def projection(games, stat, minutes_bump=0.0, min_override=None):
    """Projected mean and sd for one stat in the next game."""
    if not games:
        return None
    n = len(games)
    ws = [_w(n, i) for i in range(n)]
    mins = [x.get("min") or 0 for x in games]
    vals = [x.get(stat) or 0 for x in games]
    rates = [v / m if m else 0 for v, m in zip(vals, mins)]
    recent_rate = _wavg(rates, ws)
    season_rate = sum(vals) / sum(mins) if sum(mins) else 0
    eff_n = sum(ws)
    rate = (recent_rate * eff_n + season_rate * SHRINK_GAMES) / (eff_n + SHRINK_GAMES)
    # minutes: preseason minutes are not representative (stars ramp, benches play),
    # so when regular-season games exist, project minutes from those only.
    reg = [x for x in games if x.get("season_type") in (2, None)]
    preseason_only = not reg
    mins_src = [x.get("min") or 0 for x in (reg or games)]
    wr = [_w(len(mins_src), i) for i in range(len(mins_src))]
    recent_min = _wavg(mins_src, wr)
    season_min = mean(mins_src)
    eff_m = sum(wr)
    proj_min = (recent_min * eff_m + season_min * SHRINK_GAMES) / (eff_m + SHRINK_GAMES)
    if min_override is not None:
        proj_min = min_override
    proj_min = max(0.0, min(42.0, proj_min + minutes_bump))
    mu = rate * proj_min * (1 + 0.25 * minutes_bump / max(proj_min, 1))  # usage rises a little when minutes do
    emp_sd = pstdev(vals) if n > 1 else SD_FLOOR[stat] * 1.5
    sd = max(SD_FLOOR[stat], emp_sd)
    if n < FULL_SAMPLE:
        sd *= 1 + 0.08 * (FULL_SAMPLE - n)   # small sample -> wider
    if preseason_only:
        sd *= 1.3
    return {"mean": mu, "sd": sd, "minutes": proj_min, "rate": rate, "n": n,
            "preseason_only": preseason_only}


def p_over(proj, line):
    """P(stat > line). Lines are x.5 so no pushes; add a half-point continuity
    correction because counts are integers."""
    if not proj:
        return None
    z = (line + 0.0 - proj["mean"]) / proj["sd"]
    return 1.0 - norm_cdf(z)


def team_rotation(by_player, team, top_n=9, min_games=1):
    """Players on a team ranked by recent minutes."""
    rows = []
    for pid, games in by_player.items():
        if games[-1]["team"] != team:
            continue  # player has since moved to another team
        g = [x for x in games if x["team"] == team]
        if len(g) < min_games:
            continue
        prof = profile(g)
        if prof:
            rows.append(prof)
    # rank by recent minutes, but players with fewer than 3 games go to the back
    rows.sort(key=lambda p: (p["games"] < 3, -p["last5"]["min"]))
    return rows[:top_n]


def usage_bump_for(team_profiles, injured_out_names, player_name):
    """If a top-usage teammate is Out, bump minutes/usage for the others.
    Returns extra minutes (0-6)."""
    out_set = {n.lower() for n in injured_out_names}
    if player_name.lower() in out_set:
        return None  # the player themself is out
    bump = 0.0
    for p in team_profiles[:7]:
        # only a real rotation player being out moves the needle: 25+ minutes, 5+ games
        if p["player"].lower() in out_set and p["games"] >= 5 and p["last10"]["min"] >= 25:
            bump += min(2.0, (p["last10"]["min"] - 20) / 8.0)
    return min(4.0, bump)


def find_player(team_profiles, name):
    """Loose name match (Polymarket uses display names like ESPN, but be lenient)."""
    n = name.lower().strip()
    for p in team_profiles:
        if p["player"].lower() == n:
            return p
    last = n.split()[-1] if n else ""
    cands = [p for p in team_profiles if p["player"].lower().split()[-1] == last]
    return cands[0] if len(cands) == 1 else None


def player_table(profiles):
    """Plain-text table for the CLI."""
    hdr = f"{'Player':<24}{'Pos':<5}{'G':>3}{'MIN':>6}{'PTS':>6}{'REB':>6}{'AST':>6}{'PRA':>6}{'L5 PTS':>8}{'L5 REB':>8}{'L5 AST':>8} {'Trend':>12}"
    lines = [hdr, "-" * len(hdr)]
    for p in profiles:
        l10, l5, t = p["last10"], p["last5"], p["trend"]
        trend = f"{t['pts']:+.1f}/{t['reb']:+.1f}/{t['ast']:+.1f}"
        lines.append(f"{p['player']:<24}{(p['position'] or ''):<5}{p['games']:>3}{l10['min']:>6.1f}{l10['pts']:>6.1f}"
                     f"{l10['reb']:>6.1f}{l10['ast']:>6.1f}{l10['pra']:>6.1f}{l5['pts']:>8.1f}{l5['reb']:>8.1f}{l5['ast']:>8.1f} {trend:>12}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Matchup context: what each defense allows to each position
# ---------------------------------------------------------------------------

POS_GROUP = {"PG": "G", "SG": "G", "G": "G", "SF": "F", "PF": "F", "F": "F", "C": "C", "GF": "F", "FC": "C"}
MATCHUP_STATS = ("pts", "reb", "ast", "tpm", "stl", "blk", "pra")


def position_group(pos):
    return POS_GROUP.get((pos or "").upper(), "F")


def defense_vs_position(rows, regular_only=True):
    """rows: player stat rows joined with games (all players). Returns
    {team: {pos: {stat: per_game_allowed}}, "_league": {pos: {stat: avg}}, "_games": {team: n}}.
    'Allowed' = what opposing players at that position produced, per game."""
    by_game = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))  # (game, defending team) -> pos -> stat
    teams_in_game = defaultdict(set)
    for r in rows:
        if r.get("dnp") or not r.get("min"):
            continue
        if regular_only and r.get("season_type") not in (2, None):
            continue
        teams_in_game[r["game_id"]].add(r["team"])
    for r in rows:
        if r.get("dnp") or not r.get("min"):
            continue
        if regular_only and r.get("season_type") not in (2, None):
            continue
        opp = [t for t in teams_in_game[r["game_id"]] if t != r["team"]]
        if len(opp) != 1:
            continue
        pos = position_group(r.get("position"))
        bucket = by_game[(r["game_id"], opp[0])][pos]
        bucket["pts"] += r.get("pts") or 0
        bucket["reb"] += r.get("reb") or 0
        bucket["ast"] += r.get("ast") or 0
        bucket["tpm"] += r.get("tpm") or 0
        bucket["stl"] += r.get("stl") or 0
        bucket["blk"] += r.get("blk") or 0
        bucket["pra"] += (r.get("pts") or 0) + (r.get("reb") or 0) + (r.get("ast") or 0)
    per_team = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for (gid, team), posmap in by_game.items():
        for pos in ("G", "F", "C"):
            for s in MATCHUP_STATS:
                per_team[team][pos][s].append(posmap[pos][s] if pos in posmap else 0.0)
    out, league = {}, {p: {s: [] for s in MATCHUP_STATS} for p in ("G", "F", "C")}
    games_n = {}
    for team, posmap in per_team.items():
        out[team] = {}
        games_n[team] = max(len(v) for p in posmap.values() for v in p.values()) if posmap else 0
        for pos, stats in posmap.items():
            out[team][pos] = {s: (sum(v) / len(v) if v else 0.0) for s, v in stats.items()}
            for s, v in stats.items():
                league[pos][s].extend(v)
    out["_league"] = {p: {s: (sum(v) / len(v) if v else 0.0) for s, v in stats.items()} for p, stats in league.items()}
    out["_games"] = games_n
    return out


def matchup_factor(dvp, opp_team, pos, stat, cap=0.08, shrink_games=8.0):
    """Multiplier on a projection from what the opponent allows to this
    position vs league, shrunk toward 1 for small samples and capped."""
    if not dvp or opp_team not in dvp:
        return 1.0, None
    key = "pra" if stat in ("pra", "pr", "pa", "ra") else stat
    if key not in MATCHUP_STATS:
        return 1.0, None
    pos = position_group(pos)
    allowed = dvp[opp_team].get(pos, {}).get(key)
    lg = dvp["_league"].get(pos, {}).get(key)
    if not allowed or not lg:
        return 1.0, None
    n = dvp["_games"].get(opp_team, 0)
    rel = (allowed - lg) / lg
    rel *= n / (n + shrink_games)
    f = 1.0 + max(-cap, min(cap, 0.5 * rel))
    return f, {"allowed": allowed, "league": lg, "rel": rel, "games": n, "pos": pos}


def pace_factor(game_pace, league_pace, cap=0.06):
    if not game_pace or not league_pace:
        return 1.0
    return 1.0 + max(-cap, min(cap, 0.5 * (game_pace - league_pace) / league_pace))
