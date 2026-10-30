"""Monte Carlo player-performance simulation.

Each simulated game is a plausible version of the real one, not a draw from a
bell curve around the average:

  minutes   ~ truncated normal around projected minutes (regular-season games
              only), with blowout, foul-trouble and back-to-back disruptions
  form      ~ a shared log-normal "usage" factor per game (hot / cold nights)
              that moves shot volume, free throws, assists and turnovers together
  shots     ~ Poisson volume (FGA, 3PA share, FTA) scaled by minutes x form x
              pace x matchup, converted with Beta-distributed efficiency around
              the player's own percentages (shrunk toward league norms)
  points    = 3*3PM + 2*2PM + FTM          (so PTS, 3PM, FGA are correlated)
  rebounds, assists, steals, blocks, turnovers ~ overdispersed Poisson
              (rate x minutes x own between-game factor), assists partly tied
              to the usage factor
  combos    = sums of the same simulated game (PRA, PR, PA, RA keep their
              natural correlation)

Outputs per stat: mean, median, p10/p25/p75/p90, P(over) for any line, and the
same under alternative scenarios (more/fewer minutes, blowout, tough or
favourable matchup, teammate out) so a pick can be judged on robustness.

Everything is vectorised with numpy; 10,000 games per player take a few ms.
"""

import math
from statistics import mean, pstdev

import numpy as np

N_SIMS = 10_000
HALF_LIFE = 8.0
SHRINK_GAMES = 6.0
PRIOR_GAMES = 4.0          # weight of last season's rates when this season is thin
LEAGUE = {"p2": 0.54, "p3": 0.36, "ft": 0.78}
EFF_PSEUDO = {"p2": 60.0, "p3": 60.0, "ft": 30.0}   # pseudo-attempts shrinking efficiency to league
SIG_FLOOR, SIG_CAP = 0.12, 0.60                      # between-game log-normal factor bounds
MIN_SD_FLOOR = 3.0
BLOWOUT_MARGIN = 15.0

STATS = ("pts", "reb", "ast", "tpm", "stl", "blk", "tov", "pra", "pr", "pa", "ra")
COUNT_STATS = ("reb", "ast", "stl", "blk", "tov")

SCENARIOS = {
    "base": {},
    "more_minutes": {"min_mult": 1.12, "usage_mult": 1.05},
    "fewer_minutes": {"min_mult": 0.85},
    "blowout": {"force_blowout": True},
    "tough_matchup": {"rate_mult": 0.92},
    "favorable_matchup": {"rate_mult": 1.08},
}


def _w(n, i, hl=HALF_LIFE):
    return 0.5 ** ((n - 1 - i) / hl)


def _rw_rate(games, num_key, mins, prior=None):
    """Recency-weighted per-minute rate shrunk toward the season rate (and a
    prior-season rate when the sample is thin)."""
    n = len(games)
    ws = np.array([_w(n, i) for i in range(n)])
    vals = np.array([g.get(num_key) or 0 for g in games], dtype=float)
    m = np.array(mins, dtype=float)
    rates = np.where(m > 0, vals / np.maximum(m, 1e-9), 0.0)
    recent = float((rates * ws).sum() / ws.sum()) if ws.sum() else 0.0
    season = float(vals.sum() / m.sum()) if m.sum() else 0.0
    eff = float(ws.sum())
    rate = (recent * eff + season * SHRINK_GAMES) / (eff + SHRINK_GAMES)
    if prior is not None and n < 10:
        rate = (rate * n + prior * PRIOR_GAMES) / (n + PRIOR_GAMES)
    return rate


def _overdispersion(games, key, mins, rate):
    """Log-normal sigma for the between-game factor of a count stat: observed
    variance beyond Poisson, expressed as a relative sd, floored and capped."""
    x = np.array([g.get(key) or 0 for g in games], dtype=float)
    lam = rate * np.array(mins, dtype=float)
    if len(x) < 3 or lam.mean() <= 0:
        return 0.35
    v = float(((x - lam) ** 2).mean())
    lbar = float(lam.mean())
    phi = max(0.0, (v - lbar) / (lbar ** 2))
    sig = math.sqrt(math.log(1.0 + phi))
    if len(x) < 10:
        sig *= 1 + 0.05 * (10 - len(x))
    return float(min(SIG_CAP, max(SIG_FLOOR, sig)))


def _eff(games, made, att, key):
    """Beta parameters for a shooting percentage shrunk toward league."""
    m = sum(g.get(made) or 0 for g in games)
    a = sum(g.get(att) or 0 for g in games)
    p0, k = LEAGUE[key], EFF_PSEUDO[key]
    alpha = m + p0 * k
    beta = (a - m) + (1 - p0) * k
    return max(0.5, alpha), max(0.5, beta)


def player_params(games, prior_games=None, minutes_bump=0.0, min_override=None):
    """Everything the simulator needs for one player, estimated from his game
    log with the current team (chronological). prior_games: last season's log,
    used as a weak prior when this season is thin."""
    if not games:
        return None
    reg = [g for g in games if g.get("season_type") in (2, None)]
    src = reg or games
    mins = [g.get("min") or 0 for g in games]
    # --- minutes ---------------------------------------------------------------
    m_src = [g.get("min") or 0 for g in src]
    wr = np.array([_w(len(m_src), i) for i in range(len(m_src))])
    recent_min = float((np.array(m_src) * wr).sum() / wr.sum())
    season_min = float(np.mean(m_src))
    proj_min = (recent_min * wr.sum() + season_min * SHRINK_GAMES) / (wr.sum() + SHRINK_GAMES)
    if min_override is not None:
        proj_min = min_override
    proj_min = max(0.0, min(42.0, proj_min + minutes_bump))
    last10 = m_src[-10:]
    min_sd = max(MIN_SD_FLOOR, pstdev(last10) if len(last10) > 1 else 6.0)
    if len(src) < 8:
        min_sd *= 1 + 0.08 * (8 - len(src))
    # --- rates -----------------------------------------------------------------
    prior = {}
    if prior_games:
        pm = [g.get("min") or 0 for g in prior_games]
        tot = sum(pm) or 1.0
        for k in ("fga", "fta", "tpa", "reb", "ast", "stl", "blk", "tov"):
            prior[k] = sum(g.get(k) or 0 for g in prior_games) / tot
    rates = {k: _rw_rate(games, k, mins, prior.get(k)) for k in ("fga", "fta", "reb", "ast", "stl", "blk", "tov")}
    tpa_rate = _rw_rate(games, "tpa", mins, prior.get("tpa"))
    tpa_share = min(0.95, tpa_rate / rates["fga"]) if rates["fga"] > 0 else 0.0
    usage_vals = [((g.get("fga") or 0) + 0.44 * (g.get("fta") or 0) + (g.get("tov") or 0)) / (g.get("min") or 1) for g in games]
    u_mean = mean(usage_vals) if usage_vals else 0.0
    u_sig = (pstdev(usage_vals) / u_mean) if len(usage_vals) > 2 and u_mean > 0 else 0.3
    u_sig = float(min(0.45, max(0.10, u_sig)))
    sig = {k: _overdispersion(games, k, mins, rates[k]) for k in COUNT_STATS}
    pf_hist = [g.get("pf") or 0 for g in games[-20:]]
    p_foul = (sum(1 for x in pf_hist if x >= 5) + 0.5) / (len(pf_hist) + 5)   # shrunk share of 5+ foul games
    eff = {k: _eff(games, m, a, k) for k, m, a in (("p3", "tpm", "tpa"), ("ft", "ftm", "fta"))}
    two_m = [dict(fgm2=(g.get("fgm") or 0) - (g.get("tpm") or 0), fga2=(g.get("fga") or 0) - (g.get("tpa") or 0)) for g in games]
    eff["p2"] = _eff(two_m, "fgm2", "fga2", "p2")
    return {
        "n": len(games), "n_reg": len(reg), "preseason_only": not reg,
        "proj_min": proj_min, "min_sd": min_sd, "starter_rate": mean(g.get("starter") or 0 for g in games[-10:]),
        "rates": rates, "tpa_share": tpa_share, "u_sig": u_sig, "sig": sig, "p_foul": float(p_foul), "eff": eff,
    }


def game_draws(game_id, n=N_SIMS, p_blowout=0.0):
    """Game-level random events shared by every player in the game (a blowout
    hits all of them at once), so parlays across teammates and opponents can
    be evaluated on jointly simulated games."""
    rng = np.random.default_rng(abs(hash(("game", game_id))) % (2 ** 32))
    return {"blow": rng.random(n) < p_blowout}


def simulate(params, n=N_SIMS, matchup=None, pace=1.0, p_blowout=0.0, b2b=False, scenario=None, seed=None, game=None):
    """Simulate n games. matchup: {stat: multiplier} from what the opponent allows.
    Returns {stat: np.ndarray} including minutes."""
    sc = SCENARIOS.get(scenario or "base", {}) if isinstance(scenario, str) else (scenario or {})
    rng = np.random.default_rng(seed)
    matchup = matchup or {}
    pm = params["proj_min"] * sc.get("min_mult", 1.0)
    if b2b and pm >= 28:
        pm -= 1.0
    minutes = rng.normal(pm, params["min_sd"], n)
    # disruptions: blowout (starters sit, bench plays), foul trouble, random early exit
    pb = 1.0 if sc.get("force_blowout") else p_blowout
    if game is not None and len(game["blow"]) == n and not sc.get("force_blowout"):
        blow = game["blow"]
    else:
        blow = rng.random(n) < pb
    if pm >= 26:
        minutes = np.where(blow, minutes * rng.uniform(0.72, 0.9, n), minutes)
    else:
        minutes = np.where(blow, minutes * rng.uniform(1.0, 1.2, n), minutes)
    foul = rng.random(n) < params["p_foul"]
    minutes = np.where(foul, minutes * rng.uniform(0.7, 0.92, n), minutes)
    minutes = np.clip(minutes, 0.0, 46.0)
    rate_mult = sc.get("rate_mult", 1.0) * (0.98 if b2b else 1.0)
    # shared form / usage factor and per-stat between-game factors
    z_u = rng.normal(0.0, 1.0, n)
    usage = np.exp(params["u_sig"] * z_u - 0.5 * params["u_sig"] ** 2) * sc.get("usage_mult", 1.0)
    r = params["rates"]
    mf = lambda k: matchup.get(k, 1.0)  # noqa: E731
    vol = minutes * pace * rate_mult
    fga = rng.poisson(np.maximum(0.0, vol * r["fga"] * usage * math.sqrt(mf("pts"))))
    tpa = rng.binomial(fga, params["tpa_share"]) if params["tpa_share"] > 0 else np.zeros(n, dtype=int)
    fta = rng.poisson(np.maximum(0.0, vol * r["fta"] * usage * math.sqrt(mf("pts"))))
    e = params["eff"]
    p3 = np.clip(rng.beta(*e["p3"], n) * math.sqrt(mf("tpm")), 0.02, 0.8)
    p2 = np.clip(rng.beta(*e["p2"], n) * math.sqrt(mf("pts")), 0.1, 0.9)
    pft = rng.beta(*e["ft"], n)
    tpm = rng.binomial(tpa, p3)
    two = rng.binomial(np.maximum(fga - tpa, 0), p2)
    ftm = rng.binomial(fta, pft)
    pts = 3 * tpm + 2 * two + ftm

    def count(key, corr=0.0):
        s = params["sig"][key]
        z = corr * z_u + math.sqrt(max(0.0, 1 - corr ** 2)) * rng.normal(0.0, 1.0, n)
        f = np.exp(s * z - 0.5 * s ** 2)
        return rng.poisson(np.maximum(0.0, vol * r[key] * f * mf(key)))
    reb = count("reb", 0.0)
    ast = count("ast", 0.35)
    stl = count("stl", 0.0)
    blk = count("blk", 0.0)
    tov = rng.poisson(np.maximum(0.0, vol * r["tov"] * usage))
    return {"min": minutes, "pts": pts, "reb": reb, "ast": ast, "tpm": tpm, "stl": stl, "blk": blk, "tov": tov,
            "fga": fga, "fta": fta, "pra": pts + reb + ast, "pr": pts + reb, "pa": pts + ast, "ra": reb + ast}


def summarize(samples):
    """Distribution summary for one stat's sample array."""
    s = np.asarray(samples, dtype=float)
    q = np.percentile(s, [10, 25, 50, 75, 90])
    return {"mean": round(float(s.mean()), 2), "median": round(float(q[2]), 1), "p10": round(float(q[0]), 1),
            "p25": round(float(q[1]), 1), "p75": round(float(q[3]), 1), "p90": round(float(q[4]), 1), "sd": round(float(s.std()), 2)}


def p_over(samples, line):
    """P(stat > line); pushes (integer lines) count as neither."""
    s = np.asarray(samples)
    return float((s > line).mean())


def p_under(samples, line):
    s = np.asarray(samples)
    return float((s < line).mean())


def ladder(samples, lines):
    return {float(L): p_over(samples, L) for L in lines}


def standard_lines(samples, stat):
    """Alternative lines worth quoting around the simulated median (more below
    than above: the board is about the lowest line a player clears)."""
    med = float(np.median(samples))
    if stat in ("pts", "pra", "pr", "pa"):
        lo, hi = 5, 3
    elif stat in ("reb", "ast", "ra"):
        lo, hi = 3, 2
    else:
        lo, hi = 1, 1
    base = math.floor(med) + 0.5
    return [round(base + k, 1) for k in range(-lo, hi + 1) if base + k > 0]


def blowout_probability(pred, margin_sd):
    """P(|final margin| >= 15) from the game model's expected margin."""
    from ..model.game_model import norm_cdf
    m = pred.get("exp_margin") or 0.0
    return float(1 - norm_cdf((BLOWOUT_MARGIN - m) / margin_sd) + norm_cdf((-BLOWOUT_MARGIN - m) / margin_sd))
