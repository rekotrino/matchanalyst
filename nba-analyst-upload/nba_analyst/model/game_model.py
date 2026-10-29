"""Probability model. Produces an expected score distribution for a game and
prices every game-level market from it (moneyline, any spread, any total,
team totals).

Phase 0 uses a normal approximation on margin and total with fixed standard
deviations from the literature. Phase 1 backtesting will calibrate margin_sd,
total_sd, home advantage and the shrinkage parameters against two seasons of
closing lines.
"""

import math

from ..config import SETTINGS


def norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def expected_score(home, away, ratings, lg, rest_home, rest_away, cfg=SETTINGS.model):
    """ratings: {team: {off, def, pace}} relative to league; lg: league averages."""
    rh = ratings.get(home, {"off": 0.0, "def": 0.0, "pace": 0.0})
    ra = ratings.get(away, {"off": 0.0, "def": 0.0, "pace": 0.0})
    lg_ortg = lg.get("ortg", cfg.league_ortg)
    lg_pace = lg.get("poss", cfg.league_pace)
    pace = lg_pace + rh["pace"] + ra["pace"]
    home_pts = pace / 100 * (lg_ortg + rh["off"] + ra["def"]) + cfg.home_advantage_pts / 2
    away_pts = pace / 100 * (lg_ortg + ra["off"] + rh["def"]) - cfg.home_advantage_pts / 2

    # rest / schedule adjustments (in points of margin, split across both teams)
    adj = 0.0
    notes = []
    for team, rest, sign in ((home, rest_home, +1), (away, rest_away, -1)):
        if rest.get("b2b"):
            adj -= sign * cfg.back_to_back_penalty
            notes.append(f"{team} on a back-to-back ({-cfg.back_to_back_penalty:+.1f} pts)")
        elif rest.get("three_in_four"):
            adj -= sign * cfg.three_in_four_penalty
            notes.append(f"{team} playing 3rd game in 4 nights ({-cfg.three_in_four_penalty:+.1f} pts)")
    rd_h, rd_a = rest_home.get("rest_days"), rest_away.get("rest_days")
    if rd_h is not None and rd_a is not None and not (rest_home.get("b2b") or rest_away.get("b2b")):
        diff = max(-cfg.rest_advantage_cap, min(cfg.rest_advantage_cap, 0.5 * (min(rd_h, 4) - min(rd_a, 4))))
        if abs(diff) >= 0.5:
            adj += diff
            notes.append(f"Rest edge {diff:+.1f} pts ({home} {rd_h}d vs {away} {rd_a}d)")
    home_pts += adj / 2
    away_pts -= adj / 2

    margin = home_pts - away_pts
    total = home_pts + away_pts
    return {
        "exp_home_pts": home_pts, "exp_away_pts": away_pts,
        "exp_margin": margin, "exp_total": total,
        "p_home_win": norm_cdf(margin / cfg.margin_sd),
        "pace": pace, "notes": notes,
    }


def blend_with_reference(pred, ref_spread_home, ref_total, games_played, cfg=SETTINGS.model):
    """Anchor the model to the reference line while the model is uncalibrated.
    ref_spread_home: home handicap (negative = home favoured). Returns a new
    prediction dict; the pure-model numbers are kept under 'model_only'."""
    out = dict(pred)
    out["model_only"] = {k: pred[k] for k in ("exp_home_pts", "exp_away_pts", "exp_margin", "exp_total", "p_home_win")}
    if ref_spread_home is None and ref_total is None:
        out["anchor_weight"] = 1.0
        return out
    w = min(cfg.anchor_cap, games_played / (games_played + cfg.anchor_games))
    margin = pred["exp_margin"] if ref_spread_home is None else w * pred["exp_margin"] + (1 - w) * (-ref_spread_home)
    total = pred["exp_total"] if ref_total is None else w * pred["exp_total"] + (1 - w) * ref_total
    out.update({
        "exp_margin": margin, "exp_total": total,
        "exp_home_pts": (total + margin) / 2, "exp_away_pts": (total - margin) / 2,
        "p_home_win": norm_cdf(margin / cfg.margin_sd), "anchor_weight": w,
    })
    return out


# ---------------------------------------------------------------------------
# Market pricing from the distribution
# ---------------------------------------------------------------------------

def p_cover(exp_margin_team, line, cfg=SETTINGS.model):
    """P(team's margin + line > 0). line is the handicap applied to the team
    (negative for favourites). Half-point lines only (no pushes)."""
    return norm_cdf((exp_margin_team + line) / cfg.margin_sd)


def p_over(exp_total, line, cfg=SETTINGS.model):
    return 1.0 - norm_cdf((line - exp_total) / cfg.total_sd)


def p_team_over(exp_pts, line, cfg=SETTINGS.model):
    return 1.0 - norm_cdf((line - exp_pts) / cfg.team_pts_sd)


def fair_markets(pred, home, away, cfg=SETTINGS.model):
    """A grid of fair prices for the report: spreads and totals around the
    expectation so you can see the whole shape, not just one line."""
    m = pred["exp_margin"]
    rows = []
    base = round(m * 2) / 2
    for off in (-6, -4, -2, 0, 2, 4, 6):
        line = -(base + off)  # home handicap
        if abs(line) < 0.5:
            continue
        line = math.floor(line) + 0.5 if line > 0 else math.ceil(line) - 0.5
        rows.append({"type": "spread", "label": f"{home} {line:+.1f}", "p": p_cover(m, line)})
    tb = round(pred["exp_total"])
    for off in (-8, -4, 0, 4, 8):
        line = tb + off + 0.5
        rows.append({"type": "total", "label": f"Over {line}", "p": p_over(pred["exp_total"], line)})
    return rows


def american_to_prob(odds):
    if odds is None:
        return None
    return 100 / (odds + 100) if odds > 0 else -odds / (-odds + 100)


def devig_pair(p1, p2):
    if p1 is None or p2 is None:
        return None, None
    s = p1 + p2
    return p1 / s, p2 / s
