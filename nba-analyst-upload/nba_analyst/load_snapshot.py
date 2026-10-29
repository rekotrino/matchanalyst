"""Load a data snapshot (JSON files pulled from ESPN / Polymarket elsewhere)
into the store. Used when the machine running the pipeline can't reach the
APIs directly. Formats match the compact arrays produced by the browser
collector in tools/collect_browser.js."""

import json
import sys

from .ingest.espn import _et_date
from .storage import Store

BOX_COLS = ["game_id", "team", "is_home", "fgm", "fga", "tpm", "tpa", "ftm", "fta", "oreb", "dreb",
            "ast", "stl", "blk", "tov", "pts_paint", "fast_break", "pts_off_tov", "fouls"]


def _loads(v):
    if isinstance(v, str):
        try:
            return json.loads(v)
        except ValueError:
            return v
    return v


PLAYER_SNAPSHOT_COLS = ["game_id", "team", "player_id", "player", "position", "starter", "min", "pts", "reb",
                        "oreb", "dreb", "ast", "stl", "blk", "tov", "fg", "tp", "ft", "pf", "plus_minus"]


def _player_row(arr):
    row = dict(zip(PLAYER_SNAPSHOT_COLS, arr))
    row["dnp"] = 0
    for k in ("fg", "tp", "ft"):
        v = row.pop(k, None)
        m, a = (k + "m", k + "a") if k != "fg" else ("fgm", "fga")
        if k == "tp":
            m, a = "tpm", "tpa"
        if k == "ft":
            m, a = "ftm", "fta"
        try:
            x, y = str(v).split("-")
            row[m], row[a] = int(x), int(y)
        except (ValueError, AttributeError):
            row[m] = row[a] = None
    for k in ("min", "pts", "reb", "oreb", "dreb", "ast", "stl", "blk", "tov", "pf", "plus_minus"):
        try:
            row[k] = float(row[k]) if k == "min" else int(str(row[k]).replace("+", ""))
        except (ValueError, TypeError):
            row[k] = None
    if row["min"] is None or row["min"] <= 0:
        row["dnp"] = 1
    return row


def load(store: Store, sched=None, box=None, inj=None, odds=None, pm=None, players=None):
    scores = {}
    for g in sched or []:
        gid, date_utc, year, stype, home, away, hs, as_, done, status = g[:10]
        hs = int(hs) if done and str(hs).isdigit() else None
        as_ = int(as_) if done and str(as_).isdigit() else None
        scores[gid] = (hs, as_)
        store.upsert_game({"game_id": gid, "date_utc": date_utc, "date_et": _et_date(date_utc),
                           "season_year": year, "season_type": stype, "home": home, "away": away,
                           "home_score": hs, "away_score": as_, "completed": int(bool(done)),
                           "status": status})
        if len(g) > 10 and g[10] is not None:
            store.add_reference_odds(gid, g[14] or "ESPN", g[10], g[11], _odds(g[12]), _odds(g[13]))
    for b in box or []:
        row = dict(zip(BOX_COLS, b))
        hs, as_ = scores.get(row["game_id"], (None, None))
        if hs is None or row.get("fgm") is None:
            continue  # game not completed yet (empty box score)
        row["pts"] = hs if row["is_home"] else as_
        if row["pts"] is None:
            row["pts"] = 2 * (row["fgm"] - row["tpm"]) + 3 * row["tpm"] + row["ftm"]
        row["reb"] = (row["oreb"] or 0) + (row["dreb"] or 0)
        store.upsert_team_stats(row)
    by_game = {}
    for i in inj or []:
        by_game.setdefault(i[0], []).append({"team": i[1], "player": i[2], "status": i[3],
                                             "details": f"{i[4]} {i[5]}".strip()})
    for gid, rows in by_game.items():
        store.replace_injuries(gid, rows)
    for o in odds or []:
        store.add_reference_odds(o[0], o[1], o[2], o[3], _odds(o[4]), _odds(o[5]))
    by_slug = {}
    for r in pm or []:
        gid, slug, mid, q, mtype, line, outs, prices, toks, liq, spread, active = r
        by_slug.setdefault((gid, slug), []).append({
            "market_id": mid, "question": q, "market_type": mtype, "line": line,
            "outcomes": _loads(outs), "prices": [float(p) for p in _loads(prices)],
            "token_ids": _loads(toks), "liquidity": liq, "bid_ask": spread, "active": active})
    for (gid, slug), ms in by_slug.items():
        store.add_pm_markets(gid, slug, [m for m in ms if len(m["outcomes"]) == 2])
    for arr in players or []:
        if not arr[3] or arr[2] in (None, "undefined"):
            continue  # team-total / placeholder lines
        store.upsert_player_stats(_player_row(arr))
    store.commit()


def _odds(v):
    if v is None:
        return None
    try:
        return int(str(v).replace("+", ""))
    except ValueError:
        return None


if __name__ == "__main__":
    db, folder = sys.argv[1], sys.argv[2]
    def rd(name):
        try:
            return json.load(open(f"{folder}/{name}.json"))
        except FileNotFoundError:
            return None
    load(Store(db), rd("sched"), rd("box"), rd("inj"), rd("odds"), rd("pm"), rd("players"))
    print("loaded")
