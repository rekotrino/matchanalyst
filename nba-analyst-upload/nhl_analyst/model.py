"""Poisson goals model for hockey. Expected goals per team come from the
multiplicative ratings; the joint score distribution prices moneyline (with
overtime / shootout), totals (OT included, shootout counts as one goal, as on
Polymarket) and puck lines. While uncalibrated the expectations are anchored
to the reference line the same way the NBA model is."""

import math

from .config import SETTINGS

MAXG = 12


def poisson_pmf(lam, k):
    return math.exp(-lam) * lam ** k / math.factorial(k)


def score_matrix(lh, la):
    ph = [poisson_pmf(lh, k) for k in range(MAXG + 1)]
    pa = [poisson_pmf(la, k) for k in range(MAXG + 1)]
    return [[ph[h] * pa[a] for a in range(MAXG + 1)] for h in range(MAXG + 1)]


def summarize(lh, la, cfg=SETTINGS.model):
    M = score_matrix(lh, la)
    p_home_reg = sum(M[h][a] for h in range(MAXG + 1) for a in range(MAXG + 1) if h > a)
    p_tie = sum(M[h][h] for h in range(MAXG + 1))
    p_away_reg = 1 - p_home_reg - p_tie
    edge = max(-0.1, min(0.1, 0.06 * (lh - la)))
    p_home_ot = 0.5 + edge
    return {"M": M, "p_home_reg": p_home_reg, "p_tie": p_tie, "p_away_reg": p_away_reg, "p_home_ot": p_home_ot,
            "p_home_win": p_home_reg + p_tie * p_home_ot, "exp_home": lh, "exp_away": la,
            "exp_total": lh + la + p_tie}  # tied games add exactly one more goal (OT or shootout credit)


def p_total_over(S, line):
    """P(total goals incl. OT/SO credit > line)."""
    M, tot = S["M"], 0.0
    for h in range(MAXG + 1):
        for a in range(MAXG + 1):
            t = h + a + (1 if h == a else 0)
            if t > line:
                tot += M[h][a]
    return tot


def p_cover(S, team_is_home, line):
    """P(team covers handicap `line`). Margin after OT/SO is +-1 when tied in regulation."""
    M, p = S["M"], 0.0
    for h in range(MAXG + 1):
        for a in range(MAXG + 1):
            pr = M[h][a]
            if h == a:
                # tied: OT winner by exactly 1
                m_home = 1
                p_h = S["p_home_ot"]
                for mh, w in ((m_home, p_h), (-m_home, 1 - p_h)):
                    margin = mh if team_is_home else -mh
                    if margin + line > 0:
                        p += pr * w
            else:
                margin = (h - a) if team_is_home else (a - h)
                if margin + line > 0:
                    p += pr
    return p


def expected_goals(home, away, ratings, lg_gpg, rest_home, rest_away, cfg=SETTINGS.model):
    rh = ratings.get(home, {"att": 0.0, "def": 0.0})
    ra = ratings.get(away, {"att": 0.0, "def": 0.0})
    lh = lg_gpg * math.exp(rh["att"] + ra["def"]) * (1 + cfg.home_advantage / 2)
    la = lg_gpg * math.exp(ra["att"] + rh["def"]) / (1 + cfg.home_advantage / 2)
    notes = []
    if rest_home.get("b2b"):
        lh *= 0.96; la *= 1.03; notes.append(f"{home} on a back-to-back")
    if rest_away.get("b2b"):
        la *= 0.96; lh *= 1.03; notes.append(f"{away} on a back-to-back")
    return lh, la, notes


def american_to_prob(o):
    if o is None:
        return None
    return 100 / (o + 100) if o > 0 else -o / (-o + 100)


def devig(p1, p2):
    if p1 is None or p2 is None:
        return None, None
    return p1 / (p1 + p2), p2 / (p1 + p2)


def anchor(lh, la, ref_p_home, ref_total, games_played, cfg=SETTINGS.model):
    """Rescale expected goals so the win probability and total move toward the
    reference line with weight (1-w)."""
    w = min(cfg.anchor_cap, games_played / (games_played + cfg.anchor_games))
    S = summarize(lh, la)
    target_p = S["p_home_win"] if ref_p_home is None else w * S["p_home_win"] + (1 - w) * ref_p_home
    target_t = S["exp_total"] if ref_total is None else w * S["exp_total"] + (1 - w) * ref_total
    # bisection on log ratio r = log(lh/la), with sum chosen to hit the total
    lo, hi = -2.0, 2.0
    for _ in range(40):
        r = (lo + hi) / 2
        s = target_t - 0.0
        # choose lh + la = s - p_tie (unknown) -> iterate twice
        tie = S["p_tie"]
        for _ in range(3):
            tot = max(1.0, s - tie)
            lh2 = tot * math.exp(r) / (1 + math.exp(r))
            la2 = tot - lh2
            tie = summarize(lh2, la2)["p_tie"]
        p = summarize(lh2, la2)["p_home_win"]
        if p < target_p:
            lo = r
        else:
            hi = r
    return lh2, la2, w
