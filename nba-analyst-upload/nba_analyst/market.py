"""Market Intelligence & Edge Detection.

An independent module that looks at the *market* rather than the prediction:
where books disagree with each other, how lines and prices have moved since
they opened (and whether an injury announcement sits behind a move), which
alternative lines pair probability with payout best, where the engine's
projection and the market consensus disagree and whether that gap looks like
an opportunity or like missing information, how reliable each projection is,
and which of it all deserves a place on a ranked, de-correlated list.

It reuses the engine's results (simulated distributions, ladders with every
book's price) and the price history the pipeline stores on every run. It
never re-simulates and never invents a price.
"""

import math
from collections import defaultdict
from datetime import datetime
from statistics import median

SIGNIFICANT_LINE = {"pts": 1.0, "pra": 1.5, "pr": 1.0, "pa": 1.0, "reb": 0.5, "ast": 0.5, "ra": 1.0, "tpm": 0.5, "stl": 0.5, "blk": 0.5, "tov": 0.5}
SIGNIFICANT_PRICE = 0.04        # implied-probability points
INJURY_WINDOW_H = 6.0
RECOMMEND_SCORE = 60.0
RECOMMEND_MIN_EV = 0.03
RECOMMEND_MIN_CONF = 50.0


def _ts(s):
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return None


def _hours(a, b):
    ta, tb = _ts(a), _ts(b)
    if not ta or not tb:
        return None
    if ta.tzinfo is None:
        ta = ta.replace(tzinfo=tb.tzinfo)
    if tb.tzinfo is None:
        tb = tb.replace(tzinfo=ta.tzinfo)
    return (tb - ta).total_seconds() / 3600


def _devig(over, under):
    if not over or not under:
        return None, None
    io, iu = 1 / over, 1 / under
    return io / (io + iu), iu / (io + iu)


# ---------------------------------------------------------------------------
# price history -> opening / current / movement per book and consensus
# ---------------------------------------------------------------------------

def _main_line(rows):
    """Among a book's lines for one player/stat at one fetch, the main line is the
    one whose over and under prices are closest (the book's centre)."""
    by_line = defaultdict(dict)
    for r in rows:
        by_line[r["line"]][r["side"]] = r["price"]
    best, gap = None, 9e9
    for line, sides in by_line.items():
        if "over" in sides and "under" in sides:
            g = abs(sides["over"] - sides["under"])
            if g < gap:
                best, gap = (line, sides["over"], sides["under"]), g
    return best


def line_history(hist_rows):
    """hist_rows: book_odds rows for one game ordered by fetched_at.
    Returns {(player_lower, stat): {book: [snapshots...]}} where a snapshot is
    (fetched_at, main_line, over_price, under_price)."""
    by = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))   # (player, stat) -> book -> fetched_at -> rows
    for r in hist_rows:
        if not r.get("player") or r["market"] in ("moneyline", "spreads", "totals"):
            continue
        by[(r["player"].lower(), r["market"])][r["book"]][r["fetched_at"]].append(r)
    out = {}
    for key, books in by.items():
        out[key] = {}
        for book, fetches in books.items():
            snaps = []
            for at in sorted(fetches):
                ml = _main_line(fetches[at])
                if ml:
                    snaps.append((at, ml[0], ml[1], ml[2]))
            if snaps:
                out[key][book] = snaps
    return out


def movement(snaps, stat):
    """Opening vs current for one book's snapshots."""
    o, c = snaps[0], snaps[-1]
    io_o, _ = _devig(o[2], o[3])
    io_c, _ = _devig(c[2], c[3])
    d_line = round(c[1] - o[1], 1)
    d_over = None if io_o is None or io_c is None else round(io_c - io_o, 3)
    sig = abs(d_line) >= SIGNIFICANT_LINE.get(stat, 1.0) or (d_over is not None and abs(d_over) >= SIGNIFICANT_PRICE)
    return {"opened_at": o[0], "open_line": o[1], "open_over": o[2], "open_under": o[3], "current_at": c[0],
            "line": c[1], "over": c[2], "under": c[3], "d_line": d_line, "d_over_implied": d_over,
            "n_snapshots": len(snaps), "significant": sig,
            "steps": [{"at": s[0], "line": s[1], "over": s[2], "under": s[3]} for s in snaps[-8:]]}


def injury_events(inj_rows):
    """Status changes per player from the injury history: [(at, player_lower, team, old, new)]."""
    last = {}
    events = []
    for r in sorted(inj_rows, key=lambda x: x["fetched_at"]):
        k = (r["team"], r["player"].lower())
        new = (r.get("status") or "").lower()
        if k not in last:
            events.append((r["fetched_at"], r["player"].lower(), r["team"], None, new))
        elif last[k] != new:
            events.append((r["fetched_at"], r["player"].lower(), r["team"], last[k], new))
        last[k] = new
    return events


# ---------------------------------------------------------------------------
# confidence (reliability of the projection, distinct from its probability)
# ---------------------------------------------------------------------------

def confidence(item):
    score, reasons = 100.0, []
    ms = item.get("min_sd")
    if ms is not None and ms > 4:
        pen = min(25.0, (ms - 4) * 4)
        score -= pen
        reasons.append(f"minutes vary (sd {ms:.1f}) -{pen:.0f}")
    n = item.get("n_games") or 0
    if n < 8:
        score -= (8 - n) * 4
        reasons.append(f"only {n} games with team -{(8 - n) * 4}")
    cv = item.get("consistency")
    if cv is not None:
        if cv > 0.5:
            score -= 15
            reasons.append(f"high variance (sd/mean {cv}) -15")
        elif cv > 0.35:
            score -= 8
            reasons.append(f"moderate variance ({cv}) -8")
    for f in item.get("flags") or []:
        pen = {"day-to-day": 25, "market doubt": 20, "unpredictable minutes": 10, "role in flux": 15, "small sample": 5}.get(f, 10)
        score -= pen
        reasons.append(f"{f} -{pen}")
    mi = item.get("matchup_info") or {}
    if mi and (mi.get("games") or 0) < 5:
        score -= 5
        reasons.append("thin matchup sample -5")
    if item.get("b2b"):
        score -= 3
        reasons.append("back-to-back -3")
    return max(0.0, min(100.0, score)), reasons


# ---------------------------------------------------------------------------
# the page
# ---------------------------------------------------------------------------

def build_market(results, store, now, generated_at, snapshot=True):
    """snapshot=True (the daily `analyze`) stores this run's monitoring state; the
    read-only payload build passes False and compares against the run before it."""
    comparisons, movements, alts, disagreements, opportunities, alerts, reliability = [], [], [], [], [], [], []
    book_best = defaultdict(int)
    book_quotes = defaultdict(int)
    prev_run, prev = store.market_state_last(skip_latest=not snapshot)
    state_rows = []
    injury_alerts = []
    for r in results:
        g = r["game"]
        if g.get("completed"):
            continue
        label = f"{g['away']} @ {g['home']}"
        hist = store.odds_history(g["game_id"])
        lh = line_history(hist)
        inj_ev = injury_events(store.injuries_history(g["game_id"]))
        inj_latest = {i["player"].lower(): (i.get("status") or "") for i in r.get("injuries") or []}
        for it in r["prop_items"]:
            key = (it["player"].lower(), it["stat"])
            sim = it.get("sim") or {}
            conf, conf_why = confidence(it)
            if it["stat"] == "pts":
                reliability.append({"game_id": g["game_id"], "game": label, "player": it["player"], "player_id": it["player_id"], "team": it["team"],
                                    "position": it.get("position"), "confidence": round(conf), "reasons": conf_why, "min_sd": it.get("min_sd"),
                                    "proj_min": it.get("proj_min"), "n_games": it.get("n_games"), "consistency": it.get("consistency"),
                                    "flags": it.get("flags") or [], "injury": inj_latest.get(key[0]) or None, "p_blowout": it.get("p_blowout"),
                                    "matchup_games": (it.get("matchup_info") or {}).get("games")})
            # ---- sportsbook comparison on every real rung ----------------------------
            real = [x for x in it["ladder"] if x.get("real") and x.get("books")]
            for x in real:
                books = x["books"]
                overs = {b: v["over"] for b, v in books.items() if v.get("over")}
                unders = {b: v["under"] for b, v in books.items() if v.get("under")}
                if not overs and not unders:
                    continue
                bo = max(overs, key=overs.get) if overs else None
                bu = max(unders, key=unders.get) if unders else None
                for b in overs:
                    book_quotes[b] += 1
                for b in unders:
                    book_quotes[b] += 1
                if bo:
                    book_best[bo] += 1
                if bu:
                    book_best[bu] += 1
                spread_o = (max(1 / v for v in overs.values()) - min(1 / v for v in overs.values())) if len(overs) > 1 else 0.0
                spread_u = (max(1 / v for v in unders.values()) - min(1 / v for v in unders.values())) if len(unders) > 1 else 0.0
                comparisons.append({
                    "game_id": g["game_id"], "game": label, "player": it["player"], "player_id": it["player_id"], "team": it["team"],
                    "stat": it["stat"], "market": it["stat_label"], "line": x["line"], "p_over": x["p_over"],
                    "best_over": {"book": bo, "price": overs.get(bo)} if bo else None,
                    "best_under": {"book": bu, "price": unders.get(bu)} if bu else None,
                    "n_books": len(set(overs) | set(unders)), "books": books,
                    "price_spread_over": round(spread_o, 3), "price_spread_under": round(spread_u, 3),
                    "quoted_at": x.get("quoted_at"),
                })
            # lines differ across books? (each book's main line)
            book_lines = {}
            for b, snaps in (lh.get(key) or {}).items():
                book_lines[b] = snaps[-1][1]
            line_disp = (max(book_lines.values()) - min(book_lines.values())) if len(book_lines) > 1 else 0.0
            # ---- line movement per book + consensus -------------------------------------
            moves = {}
            for b, snaps in (lh.get(key) or {}).items():
                m = movement(snaps, it["stat"])
                # injury news within the window before the latest significant step?
                linked = None
                if m["significant"]:
                    for at, pl, team, old, new in inj_ev:
                        if team != it["team"]:
                            continue
                        h = _hours(at, m["current_at"])
                        if h is not None and -0.5 <= h <= INJURY_WINDOW_H:
                            linked = f"{pl.title()} {old or 'listed'} → {new or 'cleared'} ({h:.1f} h before the move)"
                            break
                m["injury_link"] = linked
                moves[b] = m
            cons_now = median(book_lines.values()) if book_lines else None
            cons_open = median(m["open_line"] for m in moves.values()) if moves else None
            if moves:
                movements.append({
                    "game_id": g["game_id"], "game": label, "player": it["player"], "player_id": it["player_id"], "team": it["team"],
                    "stat": it["stat"], "market": it["stat_label"], "consensus_open": cons_open, "consensus_now": cons_now,
                    "consensus_move": None if cons_open is None or cons_now is None else round(cons_now - cons_open, 1),
                    "line_dispersion": round(line_disp, 1), "books": moves, "model_median": sim.get("median"),
                    "significant": any(m["significant"] for m in moves.values()),
                    "injury_linked": next((m["injury_link"] for m in moves.values() if m.get("injury_link")), None),
                })
            # ---- alternative lines ---------------------------------------------------
            rungs = []
            for x in real:
                for side in ("over", "under"):
                    price = x["over_price"] if side == "over" else x["under_price"]
                    if not price:
                        continue
                    p = x["p_over"] if side == "over" else 1 - x["p_over"]
                    ev = p * price - 1
                    kelly = (p * price - 1) / (price - 1) if price > 1 else 0.0
                    rungs.append({"line": x["line"], "side": side, "p": round(p, 3), "price": price,
                                  "book": x["best_over_book"] if side == "over" else x["best_under_book"],
                                  "implied": x.get("implied_over") if side == "over" else x.get("implied_under"),
                                  "ev": round(ev, 3), "kelly": round(max(0.0, kelly), 3),
                                  "profile": "safe" if p >= 0.75 else "balanced" if p >= 0.6 else "aggressive" if p >= 0.45 else "long shot"})
            if rungs:
                best = max((z for z in rungs if z["ev"] > 0), key=lambda z: z["kelly"], default=None)
                alts.append({"game_id": g["game_id"], "game": label, "player": it["player"], "player_id": it["player_id"], "team": it["team"],
                             "stat": it["stat"], "market": it["stat_label"], "model_median": sim.get("median"), "model_sd": sim.get("sd"),
                             "rungs": sorted(rungs, key=lambda z: (z["line"], z["side"])), "best": best})
            # ---- market disagreement -------------------------------------------------
            if cons_now is not None and sim.get("median") is not None and sim.get("sd"):
                gap = sim["median"] - cons_now
                gap_sd = gap / max(sim["sd"], 0.75)
                side = "over" if gap > 0 else "under"
                hr10 = None
                rung = next((x for x in it["ladder"] if x["line"] == cons_now), None)
                if rung and rung.get("hit10") is not None:
                    hr10 = rung["hit10"] if side == "over" else 1 - rung["hit10"]
                l10 = it.get("l10_avg")
                agree_hist = hr10 is not None and hr10 >= 0.6
                flags = it.get("flags") or []
                inj = inj_latest.get(key[0], "")
                if abs(gap_sd) >= 0.5:
                    if flags or inj:
                        verdict, why = "likely missing information", f"market is pricing against the model while the player carries {', '.join(flags) or inj}"
                    elif conf < 50:
                        verdict, why = "model uncertainty", "projection reliability is low (minutes / sample / variance)"
                    elif agree_hist and abs(gap_sd) >= 0.75:
                        verdict, why = "genuine opportunity", f"recent results agree: hit {100 * hr10:.0f}% of last 10 at this line"
                    elif agree_hist:
                        verdict, why = "lean", "history agrees but the gap is modest"
                    else:
                        verdict, why = "unsupported", f"recent hit rate ({'n/a' if hr10 is None else f'{100 * hr10:.0f}%'}) does not back the model"
                    disagreements.append({
                        "game_id": g["game_id"], "game": label, "player": it["player"], "player_id": it["player_id"], "team": it["team"],
                        "stat": it["stat"], "market": it["stat_label"], "model_median": sim["median"], "model_mean": sim.get("mean"),
                        "consensus_line": cons_now, "gap": round(gap, 1), "gap_sd": round(gap_sd, 2), "side": side,
                        "l10_avg": l10, "season_avg": it.get("season_avg"), "hit10_side": None if hr10 is None else round(hr10, 2),
                        "confidence": round(conf), "flags": flags, "injury": inj or None, "verdict": verdict, "why": why,
                        "consensus_move": None if cons_open is None else round(cons_now - cons_open, 1),
                    })
            # ---- opportunity scoring (per side, best rung by EV) ---------------------
            for side in ("over", "under"):
                cands = [z for z in rungs if z["side"] == side and z["ev"] > 0 and z["p"] >= 0.55]
                if not cands:
                    continue
                z = max(cands, key=lambda q: q["ev"])
                gap_sd = 0.0
                if cons_now is not None and sim.get("sd"):
                    gap_sd = (sim["median"] - z["line"]) / max(sim["sd"], 0.75) * (1 if side == "over" else -1)
                disc = max(0.0, min(1.0, gap_sd / 1.0))
                # odds advantage: best book vs the consensus of books at this rung
                rung = next((x for x in it["ladder"] if x["line"] == z["line"]), None)
                prices = [v.get(side) for v in (rung.get("books") if rung else {}).values() if v.get(side)]
                adv = 0.0
                if len(prices) > 1:
                    cons_imp = sum(1 / v for v in prices) / len(prices)
                    adv = max(0.0, min(1.0, (cons_imp - 1 / z["price"]) * 10))
                mv = 0.0
                cm = None if cons_open is None or cons_now is None else cons_now - cons_open
                if cm:
                    mv = max(-1.0, min(1.0, (-cm if side == "over" else cm) / SIGNIFICANT_LINE.get(it["stat"], 1.0)))
                hr = None
                if rung and rung.get("hit10") is not None:
                    hr = rung["hit10"] if side == "over" else 1 - rung["hit10"]
                evid = 0.0 if hr is None else max(0.0, min(1.0, (hr - 0.5) / 0.4))
                score = 35 * disc + 25 * adv + 20 * (conf / 100) + 10 * evid + 10 * max(0.0, mv)
                if disc < 0.15 and adv < 0.1:
                    continue        # movement alone is never a reason
                if z["implied"] is not None and z["p"] - z["implied"] > 0.25:
                    continue        # an edge that large is a bad line, not an opportunity
                recommended = score >= RECOMMEND_SCORE and z["ev"] >= RECOMMEND_MIN_EV and conf >= RECOMMEND_MIN_CONF and not (it.get("flags"))
                skey = f"{g['game_id']}|{key[0]}|{it['stat']}|{side}"
                opp = {
                    "key": skey, "game_id": g["game_id"], "game": label, "date_et": g["date_et"], "player": it["player"], "player_id": it["player_id"],
                    "team": it["team"], "opp": it["opp"], "stat": it["stat"], "market": it["stat_label"], "side": side,
                    "line": z["line"], "price": z["price"], "book": z["book"], "implied": z["implied"], "p": z["p"], "ev": z["ev"],
                    "score": round(score, 1), "components": {"discrepancy": round(disc, 2), "odds_advantage": round(adv, 2),
                                                             "reliability": round(conf / 100, 2), "evidence": round(evid, 2), "movement": round(mv, 2)},
                    "gap_sd": round(gap_sd, 2), "consensus_line": cons_now, "consensus_move": None if cm is None else round(cm, 1),
                    "confidence": round(conf), "confidence_why": conf_why, "hit10": None if hr is None else round(hr, 2),
                    "model_median": sim.get("median"), "flags": it.get("flags") or [], "recommended": recommended,
                    "injury": inj_latest.get(key[0]) or None,
                }
                opportunities.append(opp)
                # ---- monitoring: compare with the previous run -------------------------
                st = {"key": skey, "game_id": g["game_id"], "line": z["line"], "price": z["price"], "book": z["book"],
                      "score": round(score, 1), "gap_sd": round(gap_sd, 2), "recommended": recommended, "injury": inj_latest.get(key[0]) or None}
                state_rows.append(st)
                pv = prev.get(skey)
                if pv:
                    if pv["line"] != z["line"]:
                        alerts.append({"kind": "line moved", "key": skey, "text": f"{it['player']} {side} {it['stat_label']}: best line {pv['line']} → {z['line']} ({pv['book']} → {z['book']})", "opp": opp})
                    elif z["price"] > (pv["price"] or 0) + 0.02:
                        alerts.append({"kind": "better odds", "key": skey, "text": f"{it['player']} {side} {z['line']} {it['stat_label']}: {pv['price']:.2f}× → {z['price']:.2f}× at {z['book']}", "opp": opp})
                    if (pv.get("injury") or None) != (inj_latest.get(key[0]) or None):
                        alerts.append({"kind": "injury change", "key": skey, "text": f"{it['player']}: injury status {pv.get('injury') or 'none'} → {inj_latest.get(key[0]) or 'none'}", "opp": opp})
                    if recommended and not pv.get("recommended"):
                        alerts.append({"kind": "became valuable", "key": skey, "text": f"{it['player']} {side} {z['line']} {it['stat_label']} now scores {score:.0f} (was {pv.get('score') or 0:.0f})", "opp": opp})
                    if abs(gap_sd) >= 1.0 and abs(pv.get("gap_sd") or 0) < 1.0:
                        alerts.append({"kind": "discrepancy significant", "key": skey, "text": f"{it['player']} {it['stat_label']}: model vs market gap now {gap_sd:+.2f} sd", "opp": opp})
    run_at = None
    if snapshot:
        run_at = store.market_state_save(state_rows)
        store.commit()
    # ---- ranking + portfolio ------------------------------------------------------
    opportunities.sort(key=lambda o: (-o["score"], -o["ev"]))
    seen = set()
    ranked = []
    for o in opportunities:
        k = (o["game_id"], o["player_id"])
        if k in seen:
            continue      # one side/stat per player on the ranked list
        seen.add(k)
        ranked.append(o)
    recs = [o for o in ranked if o["recommended"]]
    by_player, by_team, by_game = defaultdict(list), defaultdict(list), defaultdict(list)
    for o in recs:
        by_player[o["player"]].append(o)
        by_team[(o["game_id"], o["team"])].append(o)
        by_game[o["game_id"]].append(o)
    portfolio = {
        "n": len(recs),
        "same_game": [{"game": v[0]["game"], "n": len(v), "players": [o["player"] + " " + o["side"] + " " + o["market"] for o in v]} for v in by_game.values() if len(v) > 1],
        "same_team": [{"team": k[1], "game": v[0]["game"], "n": len(v), "players": [o["player"] for o in v]} for k, v in by_team.items() if len(v) > 1],
        "note": "Legs in the same game share the pace and blowout environment; legs on the same team share minutes and usage. Treat each cluster as one exposure.",
    }
    books_table = sorted(({"book": b, "quotes": book_quotes[b], "best": book_best[b], "best_share": round(book_best[b] / book_quotes[b], 3) if book_quotes[b] else 0}
                          for b in book_quotes), key=lambda x: -x["best_share"])
    # most interesting comparisons first: biggest price spread
    comparisons.sort(key=lambda c: -(c["price_spread_over"] + c["price_spread_under"]))
    movements = [m for m in movements if m["significant"] or any(b["n_snapshots"] > 1 for b in m["books"].values()) or m["line_dispersion"]]
    movements.sort(key=lambda m: (not m["significant"], -(abs(m["consensus_move"] or 0)), -m["line_dispersion"]))
    for m in movements[150:]:
        for b in m["books"].values():
            b.pop("steps", None)
    disagreements.sort(key=lambda d: -abs(d["gap_sd"]))
    alts.sort(key=lambda a: -(a["best"]["kelly"] if a["best"] else -1))
    return {
        "generated_at": generated_at, "previous_run": prev_run, "monitor_run": run_at,
        "books": books_table, "comparisons": comparisons[:400], "movements": movements[:400], "alternatives": alts[:400],
        "disagreements": disagreements[:300], "opportunities": ranked[:80], "recommended": recs[:60], "alerts": alerts[:80],
        "reliability": sorted(reliability, key=lambda x: x["confidence"]),
        "portfolio": portfolio,
        "rules": {"recommend_score": RECOMMEND_SCORE, "min_ev": RECOMMEND_MIN_EV, "min_confidence": RECOMMEND_MIN_CONF,
                  "weights": {"discrepancy": 35, "odds_advantage": 25, "reliability": 20, "evidence": 10, "movement": 10}},
        "notes": [
            "Opening line = the first price this system stored for that book (the pipeline runs every 30 minutes), not the book's true opener.",
            "Consensus line = median of each book's main line (the line whose over and under prices sit closest together).",
            "Movement never earns a recommendation on its own: a selection needs a model–market discrepancy or an odds advantage first.",
            "Confidence measures how reliable the projection is (minutes, role, sample, variance, injury), separately from how likely the outcome is.",
        ],
    }


# ---------------------------------------------------------------------------
# settlement shared by every system (recs log)
# ---------------------------------------------------------------------------

def settle(actual, line, side):
    if actual is None:
        return "void"
    if actual == line:
        return "push"
    return "win" if ((actual > line) == (side == "over")) else "loss"


def system_stats(rows):
    """Performance of one system from its graded recs (flat 1-unit stakes)."""
    graded = [r for r in rows if r.get("result") in ("win", "loss")]
    out = {"n_identified": len(rows), "n_graded": len(graded), "n_open": sum(1 for r in rows if not r.get("result"))}
    if not graded:
        return out
    wins = sum(1 for r in graded if r["result"] == "win")
    rets = [(r["price"] - 1) if r["result"] == "win" else -1.0 for r in graded]
    pnl = sum(rets)
    mean = pnl / len(rets)
    sd = math.sqrt(sum((x - mean) ** 2 for x in rets) / len(rets)) if len(rets) > 1 else 0.0
    clv = []
    beat = 0
    line_moves = []
    for r in graded:
        if r.get("closing_price"):
            c = 1 / r["closing_price"] - 1 / r["price"]      # +: the market moved toward our side after we took it
            clv.append(c)
            if c > 0:
                beat += 1
        if r.get("closing_line") is not None:
            mv = (r["closing_line"] - r["line"]) * (1 if r["side"] == "over" else -1)
            line_moves.append(mv)
    by_cat = defaultdict(list)
    for r, x in zip(graded, rets):
        by_cat[r["stat"]].append((r["result"] == "win", x))
    cum, series = 0.0, []
    for r, x in sorted(zip(graded, rets), key=lambda t: (t[0]["date_et"], t[0]["created_at"])):
        cum += x
        series.append({"date": r["date_et"], "cum": round(cum, 2)})
    out.update({
        "wins": wins, "hit_rate": round(wins / len(graded), 3), "avg_odds": round(sum(r["price"] for r in graded) / len(graded), 3),
        "avg_prob": round(sum(r["prob"] for r in graded) / len(graded), 3),
        "expected_roi": round(100 * sum(r["prob"] * r["price"] - 1 for r in graded) / len(graded), 1),
        "actual_roi": round(100 * mean, 1), "pnl": round(pnl, 2),
        "clv_avg": None if not clv else round(100 * sum(clv) / len(clv), 2), "beat_close": None if not clv else round(beat / len(clv), 3), "n_clv": len(clv),
        "avg_line_move": None if not line_moves else round(sum(line_moves) / len(line_moves), 2),
        "sharpe": None if sd == 0 else round(mean / sd * math.sqrt(len(rets)), 2),
        "by_category": {k: {"n": len(v), "hit_rate": round(sum(1 for w, _ in v if w) / len(v), 3), "roi": round(100 * sum(x for _, x in v) / len(v), 1)} for k, v in by_cat.items()},
        "series": series,
    })
    return out
