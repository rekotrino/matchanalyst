"""Profitability & Value Intelligence: the second-stage filter.

The engine answers "what is likely to happen". This module answers "which of
those opportunities are worth betting on" using the prices that are actually
available:

  singles   every priced player-prop rung and game line, best price per side
            across books, with devigged implied probability, EV and ROI per
            unit staked, and a risk class
  parlays   built per sportsbook (a parlay lives at one book; Polymarket has
            none) from the simulated hit indicators, so legs in the same game
            are evaluated on the same simulated games (true joint probability,
            correlation reported), legs in different games multiply
  summary   the numbers the page headlines

Nothing here re-simulates; it reuses the engine's samples and quotes. Odds are
never invented: a selection without a real price has no EV.

  EV  = p * (price - 1) - (1 - p)      per 1 unit staked
  ROI = EV * 100 (%)
"""

from datetime import datetime
from itertools import combinations

import numpy as np

import os

STALE_HOURS = float(os.environ.get("VALUE_STALE_HOURS", "6"))
PARLAY_CATEGORIES = {
    "conservative": {"legs": (2, 3), "min_p": 0.72, "min_joint": 0.40, "objective": "joint", "min_ev": 0.0},
    "balanced": {"legs": (3, 3), "min_p": 0.62, "min_joint": 0.22, "objective": "ev", "min_ev": 0.0},
    "higher_return": {"legs": (4, 5), "min_p": 0.55, "min_joint": 0.06, "objective": "ev", "min_ev": 0.0, "min_odds": 6.0},
    "probability_optimized": {"legs": (2, 3), "min_p": 0.60, "min_joint": 0.0, "objective": "joint", "min_ev": -0.05},
    "value_optimized": {"legs": (2, 4), "min_p": 0.55, "min_joint": 0.15, "objective": "ev", "min_ev": 0.0},
}
CATEGORY_LABEL = {"conservative": "Conservative", "balanced": "Balanced", "higher_return": "Higher return",
                  "probability_optimized": "Probability-optimized", "value_optimized": "Value-optimized"}


def _age_hours(ts, now):
    if not ts:
        return None
    try:
        t = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        if t.tzinfo is None:
            t = t.replace(tzinfo=now.tzinfo)
        return max(0.0, (now - t).total_seconds() / 3600)
    except ValueError:
        return None


MAX_EDGE = 0.25   # our probability minus devigged implied above this means a bad line, not a good bet


def _risk_single(p, robust, stale, flagged, implied=None, small_sample=False):
    if flagged:
        return "unverified"
    if implied is not None and p - implied > MAX_EDGE:
        return "suspect"
    if stale:
        return "stale"
    if small_sample:
        return "medium" if p >= 0.65 else "high"
    if p >= 0.70 and (robust is None or robust >= 0.60):
        return "low"
    if p >= 0.60:
        return "medium"
    return "high"


def _risk_parlay(jp, n_legs):
    if jp >= 0.45:
        return "low"
    if jp >= 0.25:
        return "medium"
    return "high" if n_legs <= 4 else "very high"


SOFT_FLAGS = {"small sample"}


def hard_flags(flags):
    return [f for f in (flags or []) if f not in SOFT_FLAGS]


def build_singles(results, now, stale_hours=STALE_HOURS):
    out = []
    for r in results:
        g = r["game"]
        if g.get("completed"):
            continue
        label = f"{g['away']} @ {g['home']}"
        for it in r["prop_items"]:
            flagged = bool(hard_flags(it.get("flags")))
            for side in ("over", "under"):
                best = None
                for x in it["ladder"]:
                    if not x["real"]:
                        continue
                    price = x["over_price"] if side == "over" else x["under_price"]
                    if not price:
                        continue
                    p = x["p_over"] if side == "over" else 1 - x["p_over"]
                    ev = p * price - 1
                    cand = {"line": x["line"], "p": round(p, 3), "price": price, "ev": round(ev, 3),
                            "book": x["best_over_book"] if side == "over" else x["best_under_book"],
                            "implied": x.get("implied_over") if side == "over" else x.get("implied_under"),
                            "robust": x.get("robust_over") if side == "over" else x.get("robust_under"),
                            "hit10": x.get("hit10") if side == "over" else (None if x.get("hit10") is None else round(1 - x["hit10"], 2)),
                            "quoted_at": x.get("quoted_at"), "books": {b: v.get(side) for b, v in (x.get("books") or {}).items() if v.get(side)},
                            "qualifying": bool(it.get("qualifying") and it["qualifying"]["line"] == x["line"] and it["qualifying"]["side"] == side)}
                    if best is None or cand["ev"] > best["ev"]:
                        best = cand
                if not best or flagged or best["p"] < 0.5 or (best["ev"] <= 0 and best["p"] < 0.65):
                    continue   # unverified, a long shot, or neither profitable nor high-probability: nothing to say
                age = _age_hours(best["quoted_at"], now)
                stale = age is not None and age > stale_hours
                sim = it.get("sim") or {}
                out.append({
                    "kind": "prop", "game_id": g["game_id"], "game": label, "date_et": g["date_et"], "tip": g["date_utc"],
                    "player": it["player"], "player_id": it["player_id"], "team": it["team"], "opp": it["opp"],
                    "market": it["stat_label"], "stat": it["stat"], "side": side, "line": best["line"],
                    "proj": sim.get("median"), "proj_mean": sim.get("mean"), "p10": sim.get("p10"), "p25": sim.get("p25"),
                    "p75": sim.get("p75"), "p90": sim.get("p90"),
                    "p": best["p"], "price": best["price"], "book": best["book"], "implied": best["implied"], "implied_devig": best["implied"] is not None,
                    "ev": best["ev"], "roi": round(100 * best["ev"], 1), "robust": best["robust"], "hit10": best["hit10"],
                    "quoted_at": best["quoted_at"], "age_h": None if age is None else round(age, 1), "stale": stale,
                    "flags": it.get("flags") or [], "qualifying": best["qualifying"], "books": best["books"],
                    "matchup": it.get("matchup"), "l10_avg": it.get("l10_avg"), "season_avg": it.get("season_avg"),
                    "proj_min": it.get("proj_min"), "n_games": it.get("n_games"),
                    "risk": _risk_single(best["p"], best["robust"], stale, flagged, best["implied"], "small sample" in (it.get("flags") or [])),
                })
        for row in r.get("lines_table") or []:
            p, price = row["our_prob"], row["best_price"]
            ev = p * price - 1
            if p < 0.5 or (ev <= 0 and p < 0.65):
                continue
            age = _age_hours(row.get("quoted_at"), now)
            stale = age is not None and age > stale_hours
            out.append({
                "kind": "game", "game_id": g["game_id"], "game": label, "date_et": g["date_et"], "tip": g["date_utc"],
                "player": None, "team": row["side"] if row["market"] != "totals" else None, "market": row["market"],
                "side": row["side"], "line": row.get("line"), "proj": None,
                "p": p, "price": price, "book": row["best_book"], "implied": round(1 / price, 3), "implied_devig": False,
                "ev": round(ev, 3), "roi": round(100 * ev, 1), "robust": None, "hit10": None,
                "quoted_at": row.get("quoted_at"), "age_h": None if age is None else round(age, 1), "stale": stale,
                "flags": [], "qualifying": False, "books": row.get("books") or {},
                "risk": _risk_single(p, None, stale, False),
            })
    out.sort(key=lambda x: (-x["ev"], -x["p"]))
    return out


# ---------------------------------------------------------------------------
# parlays on jointly simulated games
# ---------------------------------------------------------------------------

def _unpack(leg):
    return np.unpackbits(leg["hit"])[: leg["n"]].astype(bool)


def joint_probability(legs, cache):
    """Legs in the same game: fraction of simulated games where every leg hit
    (same game-level draws, same player draws). Different games: independent."""
    by_game = {}
    for i, leg in enumerate(legs):
        by_game.setdefault(leg["game_id"], []).append(i)
    jp = 1.0
    for idxs in by_game.values():
        mask = None
        for i in idxs:
            h = cache[i]
            mask = h if mask is None else (mask & h)
        jp *= float(mask.mean())
    return jp


def build_parlays(results, now, stale_hours=STALE_HOURS, per_category=3, pool_size=14):
    """Per sportsbook, the best parlays in each category. Returns (parlays, notes)."""
    legs = [l for r in results if not r["game"].get("completed") for l in (r.get("legs") or [])]
    notes = []
    if not legs:
        return {}, ["No priced prop legs with simulation samples; parlays need at least two +EV legs at one sportsbook."]
    books = sorted({l["book"] for l in legs if l["book"] != "polymarket"})
    if "polymarket" in {l["book"] for l in legs}:
        notes.append("Polymarket prices are singles only (no parlay product); parlays are built per sportsbook.")
    out = {k: [] for k in PARLAY_CATEGORIES}
    for book in books:
        pool = {}
        for l in legs:
            if l["book"] != book:
                continue
            age = _age_hours(l.get("quoted_at"), now)
            if age is not None and age > stale_hours:
                continue
            key = l["player_id"]
            if key not in pool or (l["qualifying"], l["ev"]) > (pool[key]["qualifying"], pool[key]["ev"]):
                pool[key] = l       # one leg per player: his stats rise and fall together
        pool = list(pool.values())
        if len(pool) < 2:
            continue
        for cat, spec in PARLAY_CATEGORIES.items():
            cand = [l for l in pool if l["p"] >= spec["min_p"]]
            cand.sort(key=lambda l: -(l["p"] if spec["objective"] == "joint" else l["ev"]))
            cand = cand[:pool_size]
            if len(cand) < spec["legs"][0]:
                continue
            cache = {i: _unpack(l) for i, l in enumerate(cand)}
            found = []
            for k in range(spec["legs"][0], min(spec["legs"][1], len(cand)) + 1):
                for combo in combinations(range(len(cand)), k):
                    sel = [cand[i] for i in combo]
                    jp = joint_probability(sel, {j: cache[i] for j, i in enumerate(combo)})
                    odds = float(np.prod([l["price"] for l in sel]))
                    ev = jp * odds - 1
                    if jp < spec["min_joint"] or ev < spec["min_ev"] or odds < spec.get("min_odds", 1.0):
                        continue
                    indep = float(np.prod([l["p"] for l in sel]))
                    found.append({"legs": sel, "jp": jp, "odds": odds, "ev": ev, "indep": indep})
            found.sort(key=lambda x: -(x["jp"] if spec["objective"] == "joint" else x["ev"]))
            seen_sets = []
            for f in found:
                ids = {l["player_id"] for l in f["legs"]}
                if any(len(ids & s) >= len(ids) - 0 for s in seen_sets):
                    continue
                seen_sets.append(ids)
                games = {l["game_id"] for l in f["legs"]}
                ratio = f["jp"] / f["indep"] if f["indep"] > 0 else 1.0
                corr = "independent games" if len(games) == len(f["legs"]) else (
                    f"positively correlated (+{100 * (ratio - 1):.0f}% vs independent)" if ratio > 1.03 else
                    f"negatively correlated ({100 * (ratio - 1):.0f}% vs independent)" if ratio < 0.97 else "same game, near-independent")
                out[cat].append({
                    "category": cat, "label": CATEGORY_LABEL[cat], "book": book,
                    "legs": [{k: v for k, v in l.items() if k not in ("hit", "n")} for l in f["legs"]],
                    "joint_p": round(f["jp"], 4), "independent_p": round(f["indep"], 4), "odds": round(f["odds"], 3),
                    "ev": round(f["ev"], 3), "roi": round(100 * f["ev"], 1), "payout_per_10": round(10 * f["odds"], 2),
                    "correlation": corr, "corr_ratio": round(ratio, 3), "n_games": len(games),
                    "risk": _risk_parlay(f["jp"], len(f["legs"])),
                })
                if sum(1 for x in out[cat] if x["book"] == book) >= per_category:
                    break
    for cat in out:
        out[cat].sort(key=lambda x: -(x["joint_p"] if PARLAY_CATEGORIES[cat]["objective"] == "joint" else x["ev"]))
        out[cat] = out[cat][:per_category * 2]
    if not books:
        notes.append("No sportsbook prop prices in the window (ODDS_API_KEY missing or quota exhausted); only Polymarket singles are available.")
    return out, notes


def build_value(results, now, generated_at):
    singles = build_singles(results, now)
    parlays, notes = build_parlays(results, now)
    pos = [s for s in singles if s["ev"] > 0 and not s["stale"] and s["risk"] not in ("unverified", "suspect")]
    qual = [s for s in singles if s["qualifying"]]
    all_parlays = [p for v in parlays.values() for p in v]
    best_parlay = max(all_parlays, key=lambda p: p["ev"]) if all_parlays else None
    risk = {}
    for s in pos:
        risk[s["risk"]] = risk.get(s["risk"], 0) + 1
    missing = [
        "Existing parlay recommendations: the engine had none; parlays are built here from the engine's simulated games.",
        "Game-line implied probabilities are raw (1/price): both sides at one book are not always stored, so the margin is not removed there.",
        "Defensive schemes, primary-defender assignments and confirmed starting lineups are not in the data; matchup uses what defences allow by position.",
    ] + notes
    stale_n = sum(1 for s in singles if s["stale"])
    if stale_n:
        missing.append(f"{stale_n} selections carry prices older than {STALE_HOURS:.0f} hours and are excluded from parlays and the positive-EV counts.")
    sus = sum(1 for s in singles if s["risk"] == "suspect")
    if sus:
        missing.append(f"{sus} selections show an edge above {100 * MAX_EDGE:.0f} points over the devigged implied probability; treated as suspect lines, not value.")
    return {
        "generated_at": generated_at, "stale_hours": STALE_HOURS,
        "summary": {
            "n_qualifying": len(qual), "n_positive_ev": len(pos), "n_singles": len(singles),
            "highest_prob": max(pos, key=lambda s: s["p"]) if pos else (max(singles, key=lambda s: s["p"]) if singles else None),
            "best_value": max(pos, key=lambda s: s["ev"]) if pos else None,
            "best_parlay": best_parlay,
            "avg_roi": round(float(np.mean([s["roi"] for s in pos])), 1) if pos else None,
            "risk": risk, "n_parlays": len(all_parlays), "books": sorted({s["book"] for s in singles if s["book"]}),
        },
        "singles": singles, "parlays": parlays, "missing": missing,
    }
