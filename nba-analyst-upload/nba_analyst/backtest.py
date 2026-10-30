"""Walk-forward backtest of the player-prop simulator.

For every completed regular-season game in the database, each player is
simulated using only what was known before that date (his earlier games with
the team, last season as a weak prior, the opponent's defence-by-position from
earlier games) and the simulated P(over) at a ladder of lines is compared with
what he actually did.

Reports: Brier score and log loss, a calibration table (predicted vs observed
by probability bucket), the hit rate of selections at the qualifying threshold
for overs and unders separately, and flat-stake ROI at a standard -110 price.
Fits a Platt recalibration (p' = sigmoid(a + b*logit(p))) and writes it to
data/calibration.json, which the live pipeline picks up automatically.

  python -m nba_analyst backtest                # all completed games with enough history
  python -m nba_analyst backtest --sims 2000 --min-history 8 --no-write
"""

import json
import math
import os
from collections import defaultdict
from datetime import datetime

import numpy as np

from .analyst import players as pl
from .analyst import sim
from .config import SETTINGS

EVAL_STATS = ("pts", "reb", "ast", "tpm", "pra", "stl", "blk")


def _logit(p):
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def fit_platt(ps, ys, iters=200, lr=0.5):
    """Logistic regression of outcome on logit(p): returns (a, b)."""
    x = np.array([_logit(p) for p in ps])
    y = np.array(ys, dtype=float)
    a, b = 0.0, 1.0
    for _ in range(iters):
        z = a + b * x
        q = 1 / (1 + np.exp(-z))
        ga, gb = (q - y).mean(), ((q - y) * x).mean()
        a -= lr * ga
        b -= lr * gb
    return float(a), float(b)


def run(store, n_sims=3000, min_history=8, hit_weight=None, log=print, seasons=None):
    cfg = SETTINGS.betting
    hw = cfg.hit_rate_weight if hit_weight is None else hit_weight
    rows = store.player_stats()
    rows = [r for r in rows if r.get("season_type") in (2, None)]
    by = pl.player_games(rows)
    games = {}
    for r in rows:
        games.setdefault(r["game_id"], {"game_id": r["game_id"], "date_et": r["date_et"], "home": r["home"], "away": r["away"]})
    glist = sorted(games.values(), key=lambda g: g["date_et"])
    if seasons:
        glist = [g for g in glist if any(g["date_et"].startswith(s) for s in seasons)]
    dates = sorted({g["date_et"] for g in glist})
    log(f"[backtest] {len(glist)} games on {len(dates)} dates, {len(by)} players; {n_sims} sims per player")
    # defence-by-position as of each date (only earlier games)
    dvp_cache, rows_sorted = {}, sorted(rows, key=lambda r: r["date_et"])
    records = []
    team_dates = defaultdict(set)
    for g in glist:
        team_dates[g["home"]].add(g["date_et"])
        team_dates[g["away"]].add(g["date_et"])
    for gi, g in enumerate(glist):
        d = g["date_et"]
        if d not in dvp_cache:
            prior_rows = [r for r in rows_sorted if r["date_et"] < d]
            dvp_cache[d] = pl.defense_vs_position(prior_rows) if prior_rows else None
        dvp = dvp_cache[d]
        season_start = f"{int(d[:4]) - (1 if d[5:7] < '09' else 0)}-09-01"
        for team in (g["home"], g["away"]):
            opp = g["away"] if team == g["home"] else g["home"]
            prev = [x for x in team_dates[team] if x < d]
            b2b = bool(prev) and (datetime.fromisoformat(d) - datetime.fromisoformat(max(prev))).days == 1
            for pid, pg in by.items():
                today_rows = [x for x in pg if x["game_id"] == g["game_id"] and x["team"] == team]
                if not today_rows:
                    continue
                actual = today_rows[0]
                hist = [x for x in pg if x["date_et"] < d and x["team"] == team]
                cur = [x for x in hist if x["date_et"] >= season_start]
                prior = [x for x in pg if x["date_et"] < season_start]
                if len(cur) < min_history:
                    continue
                params = sim.player_params(cur, prior_games=prior)
                if not params:
                    continue
                mfs = {}
                for st in ("pts", "reb", "ast", "tpm", "stl", "blk"):
                    mfs[st], _ = pl.matchup_factor(dvp, opp, actual.get("position"), st) if dvp else (1.0, None)
                S = sim.simulate(params, n_sims, mfs, 1.0, 0.12, b2b, seed=gi * 1000 + hash(pid) % 1000)
                for st in EVAL_STATS:
                    a = actual.get(st) or 0
                    for line in sim.standard_lines(S[st], st):
                        p_s = sim.p_over(S[st], line)
                        hr, hn = pl.hit_rate(cur, st, line, 10)
                        w = hw if hn >= 8 else (hw / 2 if hn >= 5 else 0.0)
                        p_b = (1 - w) * p_s + w * hr if hr is not None else p_s
                        records.append({"date": d, "player": actual["player"], "stat": st, "line": line,
                                        "p_sim": p_s, "p_blend": p_b, "hit": 1 if a > line else 0, "push": a == line,
                                        "min_pred": params["proj_min"], "min_act": actual.get("min") or 0})
        if (gi + 1) % 50 == 0:
            log(f"  .. {gi + 1} games, {len(records)} prop evaluations")
    return records


def evaluate(records, cfg=SETTINGS.betting, price=1.909):
    recs = [r for r in records if not r["push"]]
    n = len(recs)
    if not n:
        return {"n": 0}
    out = {"n": n}
    for key in ("p_sim", "p_blend"):
        ps = np.array([r[key] for r in recs])
        ys = np.array([r["hit"] for r in recs])
        out[key] = {"brier": float(((ps - ys) ** 2).mean()),
                    "logloss": float(-(ys * np.log(np.clip(ps, 1e-6, 1)) + (1 - ys) * np.log(np.clip(1 - ps, 1e-6, 1))).mean())}
        # calibration by bucket (both sides folded to the favoured side)
        buckets = defaultdict(list)
        for p, y in zip(ps, ys):
            side_p, side_y = (p, y) if p >= 0.5 else (1 - p, 1 - y)
            buckets[min(int(side_p * 20) / 20, 0.95)].append((side_p, side_y))
        out[key]["calibration"] = [{"bucket": b, "n": len(v), "pred": float(np.mean([x[0] for x in v])),
                                    "obs": float(np.mean([x[1] for x in v]))} for b, v in sorted(buckets.items())]
        # selections at the qualifying threshold
        sel = {}
        for side in ("over", "under"):
            picks = [(p, y) if side == "over" else (1 - p, 1 - y) for p, y in zip(ps, ys)]
            picks = [(p, y) for p, y in picks if p >= cfg.qualify_prob]
            if picks:
                hits = sum(y for _, y in picks)
                sel[side] = {"n": len(picks), "hit_rate": hits / len(picks), "avg_pred": float(np.mean([p for p, _ in picks])),
                             "roi_at_-110": (hits * (price - 1) - (len(picks) - hits)) / len(picks)}
            else:
                sel[side] = {"n": 0}
        out[key]["selections"] = sel
        by_stat = defaultdict(list)
        for r, p, y in zip(recs, ps, ys):
            by_stat[r["stat"]].append((p, y))
        out[key]["by_stat"] = {st: {"n": len(v), "brier": float(np.mean([(p - y) ** 2 for p, y in v]))} for st, v in by_stat.items()}
    a, b = fit_platt([r["p_blend"] for r in recs], [r["hit"] for r in recs])
    out["platt"] = {"a": a, "b": b}
    ps = np.array([1 / (1 + math.exp(-(a + b * _logit(r["p_blend"])))) for r in recs])
    ys = np.array([r["hit"] for r in recs])
    out["p_calibrated"] = {"brier": float(((ps - ys) ** 2).mean())}
    mp = np.array([r["min_pred"] for r in recs]); ma = np.array([r["min_act"] for r in recs])
    out["minutes"] = {"mae": float(np.abs(mp - ma).mean()), "bias": float((mp - ma).mean())}
    return out


def report_md(res, path):
    L = [f"# Prop simulator backtest - {datetime.now():%Y-%m-%d %H:%M}", "",
         f"{res['n']} player-prop evaluations (regular-season games, walk-forward, standard ladders).", ""]
    for key, label in (("p_sim", "Simulation only"), ("p_blend", "Simulation blended with L10 hit rate")):
        r = res[key]
        L += [f"## {label}", "", f"Brier {r['brier']:.4f} · log loss {r['logloss']:.4f}", "",
              "| bucket | n | predicted | observed |", "|---|---|---|---|"]
        L += [f"| {c['bucket']:.2f}+ | {c['n']} | {c['pred']:.3f} | {c['obs']:.3f} |" for c in r["calibration"]]
        L += ["", "Selections at the qualifying threshold (ROI is at a hypothetical -110 price on our own alternative",
              "lines, i.e. a calibration check, not a market result - real books price those lines differently):", ""]
        for side, s in r["selections"].items():
            if s["n"]:
                L.append(f"- {side}: n={s['n']}, hit rate {s['hit_rate']:.3f} (predicted {s['avg_pred']:.3f}), ROI at -110 {s['roi_at_-110']:+.3f}")
            else:
                L.append(f"- {side}: no selections")
        L += ["", "By stat (Brier): " + ", ".join(f"{st} {v['brier']:.3f} (n={v['n']})" for st, v in sorted(r["by_stat"].items())), ""]
    L += ["## Recalibration", "", f"Platt a={res['platt']['a']:+.3f}, b={res['platt']['b']:.3f} -> Brier {res['p_calibrated']['brier']:.4f}",
          f"Minutes projection: MAE {res['minutes']['mae']:.2f}, bias {res['minutes']['bias']:+.2f}", ""]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    return path


def write_calibration(res, path=None):
    path = path or os.path.join(os.path.dirname(SETTINGS.db_path) or "data", "calibration.json")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"calibration_a": res["platt"]["a"], "calibration_b": res["platt"]["b"],
                   "fitted_at": datetime.now().isoformat(timespec="minutes"), "n": res["n"],
                   "brier_before": res["p_blend"]["brier"], "brier_after": res["p_calibrated"]["brier"]}, f, indent=1)
    return path
