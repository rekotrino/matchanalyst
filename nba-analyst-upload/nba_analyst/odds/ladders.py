"""Shared, sport-agnostic pieces that work on Quote rows:

  build_prop_items   - per player·stat ladders (union of every book's lines and the
                       sport's standard lines) with our probability, the best price
                       per side and which book offers it
  scan_quotes        - paper-pick candidates across all books (best price per side)
  game_lines_table   - moneyline / spread / total comparison rows for the game view
"""

from collections import defaultdict

from .base import Quote, best_quote


def _name_resolver(fns):
    """fns keys are (player_lower, stat). Returns resolve(name) -> player_lower or None."""
    names = {k[0] for k in fns}
    by_last = defaultdict(set)
    for n in names:
        by_last[n.split()[-1]].add(n)

    def resolve(name):
        n = (name or "").lower().strip()
        if n in names:
            return n
        c = by_last.get(n.split()[-1] if n else "", set())
        return next(iter(c)) if len(c) == 1 else None
    return resolve


def build_prop_items(game, fns, quotes, standard_lines, cfg, stat_label, dtd_names, flags_for=None):
    """fns: {(player_lower, stat): {p_fn, player_id, player, team, opp, position, proj{mean,sd,minutes}, recent[],
    l10, l5, season, l10_sd, hit_fn(line, n)->(rate, n), extra...}}.
    quotes: Quote rows for this game. standard_lines: {stat: [lines]} or {}.
    Returns the list the dashboard renders (same shape for every sport)."""
    resolve = _name_resolver(fns)
    qmap = defaultdict(list)   # (player_lower, stat, line) -> quotes
    for q in quotes:
        if q.player is None or q.line is None:
            continue
        key = resolve(q.player)
        if key and (key, q.market) in fns:
            qmap[(key, q.market, q.line)].append(q)
    lines_for = defaultdict(set)
    for (pl, stat, line) in qmap:
        lines_for[(pl, stat)].add(line)
    for (pl, stat) in fns:
        for line in (standard_lines or {}).get(stat) or []:
            lines_for[(pl, stat)].add(line)

    out = []
    for (pl, stat), e in fns.items():
        lines = sorted(lines_for.get((pl, stat), set()))
        if not lines:
            continue
        ladder = []
        for line in lines:
            p = e["p_fn"](line)
            qs = qmap.get((pl, stat, line), [])
            overs = [q for q in qs if q.side == "over"]
            unders = [q for q in qs if q.side == "under"]
            bo, bu = best_quote(overs), best_quote(unders)
            books = {}
            for q in qs:
                books.setdefault(q.book, {})[q.side] = round(q.price, 3)
            hr10, hn = e["hit_fn"](line, 10)
            hr5, _ = e["hit_fn"](line, 5)
            hrs, ns = e["hit_fn"](line, 0)
            pm_liq = next((q.liquidity for q in qs if q.book == "polymarket"), None)
            has_market = bool(qs) and (bo is not None or bu is not None)
            # a Polymarket-only quote with thin liquidity is not a real market
            real = has_market and any(q.book != "polymarket" or (q.liquidity or 0) >= cfg.min_liquidity for q in qs)
            ladder.append({
                "market_id": None, "line": line, "p_over": round(p, 3), "p_model": None,
                "hit10": None if hr10 is None else round(hr10, 2), "hit5": None if hr5 is None else round(hr5, 2),
                "hit_season": None if hrs is None else round(hrs, 2), "hit_n": hn, "hit_season_n": ns,
                "yes": None if bo is None else round(1 / bo.price, 3), "no": None if bu is None else round(1 / bu.price, 3),
                "yes_buy": None if bo is None else round(1 / bo.price, 3), "no_buy": None if bu is None else round(1 / bu.price, 3),
                "best_over_book": bo.book if bo else None, "best_under_book": bu.book if bu else None,
                "over_price": None if bo is None else round(bo.price, 3), "under_price": None if bu is None else round(bu.price, 3),
                "edge_over": None if bo is None else round(p - 1 / bo.price, 3),
                "edge_under": None if bu is None else round((1 - p) - 1 / bu.price, 3),
                "books": books, "liquidity": pm_liq, "has_market": has_market, "real": real,
            })
        flags = list(flags_for(e) if flags_for else [])
        if e["player"].lower() in dtd_names:
            flags.append("day-to-day")
        if any(x["real"] and x["edge_over"] is not None and x["edge_over"] >= 0.25 for x in ladder):
            flags.append("market doubt")
        pool = [] if ("day-to-day" in flags or "market doubt" in flags) else [x for x in ladder if x["real"]]
        fair_pool = [x for x in ladder if not x["has_market"]]
        safe = [x for x in pool if x["p_over"] >= 0.60 and x["edge_over"] is not None and x["edge_over"] >= -0.01]
        if not safe and not pool and "day-to-day" not in flags:   # no market anywhere: fair-price floor
            safe = [x for x in fair_pool if x["p_over"] >= 0.60]
        value = [x for x in pool if x["p_over"] >= cfg.prob_floor and x["edge_over"] is not None and x["edge_over"] >= cfg.edge_threshold]
        under = [x for x in pool if (1 - x["p_over"]) >= 0.60 and x["edge_under"] is not None and x["edge_under"] >= 0.0]
        vals = e.get("recent") or []
        proj = e["proj"]
        books_seen = sorted({b for x in ladder for b in x["books"]})
        out.append({
            "player_id": e["player_id"], "player": e["player"], "team": e["team"], "opp": e["opp"], "position": e.get("position"),
            "stat": stat, "stat_label": stat_label(stat), "proj": round(proj["mean"], 2), "proj_raw": round(e.get("proj_raw", proj["mean"]), 2),
            "proj_sd": round(proj["sd"], 2), "proj_min": round(proj["minutes"], 1), "p10": e.get("p10"),
            "l10_avg": e.get("l10"), "l5_avg": e.get("l5"), "season_avg": e.get("season"), "l10_sd": e.get("l10_sd"),
            "recent": vals, "l10_min": min(vals) if vals else None, "matchup": e.get("matchup", 1.0), "matchup_info": e.get("matchup_info"),
            "pace_factor": e.get("pace_factor", 1.0), "bump": e.get("bump", 0), "rate60": e.get("rate60"), "starter_note": e.get("starter_note"),
            "ladder": ladder, "floor_line": ladder[0]["line"], "floor_liquid": (pool[0]["line"] if pool else None),
            "flags": flags, "liquid": bool(pool), "fair_only": not any(x["has_market"] for x in ladder), "books": books_seen,
            "safe_pick": max(safe, key=lambda x: (x["p_over"], x["edge_over"] or 0)) if safe else None,
            "value_pick": max(value, key=lambda x: x["edge_over"]) if value else None,
            "under_pick": max(under, key=lambda x: (1 - x["p_over"], x["edge_under"])) if under else None,
        })
    out.sort(key=lambda x: (x["team"], -(x["l10_avg"] or 0)))
    return out


def game_lines_table(game, quotes, prob_fn):
    """Rows for moneyline / spreads / totals across books.
    prob_fn(quote) -> our probability for that quote's side, or None."""
    groups = defaultdict(list)
    for q in quotes:
        if q.market in ("moneyline", "spreads", "totals"):
            groups[(q.market, q.side, q.line)].append(q)
    rows = []
    for (market, side, line), qs in groups.items():
        p = prob_fn(qs[0])
        if p is None:
            continue
        b = best_quote(qs)
        rows.append({"market": market, "side": side, "line": line, "our_prob": round(p, 3),
                     "best_price": round(b.price, 3), "best_book": b.book, "edge": round(p - 1 / b.price, 3),
                     "books": {q.book: round(q.price, 3) for q in sorted(qs, key=lambda x: -x.price)},
                     "liquidity": next((q.liquidity for q in qs if q.book == "polymarket"), None)})
    order = {"moneyline": 0, "spreads": 1, "totals": 2}
    rows.sort(key=lambda r: (order[r["market"]], -(r["edge"] or 0)))
    return rows


def scan_quotes(game, quotes, prob_fn, cfg, prop_stats, kelly):
    """Paper-pick candidates across all books: best price per (market, side, line, player)."""
    groups = defaultdict(list)
    for q in quotes:
        groups[(q.market, q.side, q.line, (q.player or "").lower())].append(q)
    cands = []
    for key, qs in groups.items():
        qs = [q for q in qs if q.book != "polymarket" or (q.liquidity or 0) >= cfg.min_liquidity]
        if not qs:
            continue
        b = best_quote(qs)
        p = prob_fn(b)
        if p is None:
            continue
        buy = 1 / b.price
        edge = p - buy
        if edge < cfg.edge_threshold or p < cfg.prob_floor:
            continue
        label = f"{b.player}: {b.market} O/U {b.line}" if b.player else (f"{b.market} {b.side} {b.line if b.line is not None else ''}".strip())
        cands.append({"game_id": game["game_id"], "matchup": f"{game['away']} @ {game['home']}", "market_id": f"{b.book}:{b.market}:{b.side}:{b.line}:{(b.player or '').lower()}",
                      "market_type": b.market, "line": b.line, "question": label, "outcome": b.side if not b.player else b.side,
                      "player": b.player, "book": b.book, "our_prob": p, "price": buy, "buy_price": buy, "decimal": b.price, "edge": edge,
                      "liquidity": b.liquidity, "stake": kelly(p, buy), "books": {q.book: round(q.price, 3) for q in qs}})
    cands.sort(key=lambda c: (-c["our_prob"], -c["edge"]))
    picks, used_side, used_total, used_players = [], False, False, set()
    for c in cands:
        if len(picks) >= cfg.max_picks_per_game:
            break
        if c["market_type"] in ("moneyline", "spreads"):
            if used_side:
                continue
            used_side = True
        elif c["market_type"] == "totals":
            if used_total:
                continue
            used_total = True
        elif c["market_type"] in prop_stats:
            if (c["player"] or "").lower() in used_players:
                continue
            used_players.add((c["player"] or "").lower())
        picks.append(c)
    return cands, picks
