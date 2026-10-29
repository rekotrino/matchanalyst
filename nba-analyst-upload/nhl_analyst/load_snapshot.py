"""Load a browser-collected NHL snapshot (sched, box, sk, gl, inj, odds, ot, pm JSON) into the store."""

import json
import sys

from .ingest.espn import _et_date
from .storage import Store

BOX = ["game_id", "team", "is_home", "shots", "ppg", "ppo", "hits", "blocked", "fow", "giveaways", "takeaways", "pim"]
SK = ["game_id", "team", "player_id", "player", "position", "toi", "g", "a", "sog", "missed", "blk", "hits", "pp_toi", "plus_minus", "pim"]
GL = ["game_id", "team", "player_id", "player", "position", "toi", "ga", "sa", "sv"]


def _odds(v):
    try:
        return int(str(v).replace("+", "")) if v is not None else None
    except ValueError:
        return None


def _loads(v):
    if isinstance(v, str):
        try:
            return json.loads(v)
        except ValueError:
            return v
    return v


def load(store: Store, sched, box=None, sk=None, gl=None, inj=None, odds=None, ot=None, pm=None):
    finish = {o[0]: o[1] for o in (ot or [])}
    scores = {}
    for g in sched:
        gid, date_utc, year, stype, home, away, hs, as_, done, status = g[:10]
        hs = int(hs) if done and str(hs).isdigit() else None
        as_ = int(as_) if done and str(as_).isdigit() else None
        scores[gid] = (hs, as_, int(bool(done)))
        store.upsert_game({"game_id": gid, "date_utc": date_utc, "date_et": _et_date(date_utc), "season_year": year,
                           "season_type": stype, "home": home, "away": away, "home_score": hs, "away_score": as_,
                           "completed": int(bool(done)), "status": status, "finish": finish.get(gid) or (g[15] if len(g) > 15 and done else None)})
        if len(g) > 10 and g[10] is not None:
            store.add_reference_odds(gid, g[14] or "ESPN", g[10], g[11], _odds(g[12]), _odds(g[13]))
    for b in box or []:
        row = dict(zip(BOX, b))
        hs, as_, done = scores.get(row["game_id"], (None, None, 0))
        if not done:
            continue
        row["goals"] = hs if row["is_home"] else as_
        store.upsert_team_stats(row)
    for s in sk or []:
        row = dict(zip(SK, s))
        if scores.get(row["game_id"], (None, None, 0))[2]:
            store.upsert_skater(row)
    for s in gl or []:
        row = dict(zip(GL, s))
        row.pop("position", None)
        if scores.get(row["game_id"], (None, None, 0))[2]:
            store.upsert_goalie(row)
    by = {}
    for i in inj or []:
        by.setdefault(i[0], []).append({"team": i[1], "player": i[2], "status": i[3], "details": f"{i[4]} {i[5]}".strip()})
    for gid, rows in by.items():
        store.replace_injuries(gid, rows)
    for o in odds or []:
        store.add_reference_odds(o[0], o[1], o[2], o[3], _odds(o[4]), _odds(o[5]))
    by_slug = {}
    for r in pm or []:
        gid, slug, mid, q, mtype, line, outs, prices, toks, liq, spread, active = r
        by_slug.setdefault((gid, slug), []).append({"market_id": mid, "question": q, "market_type": mtype, "line": line,
                                                     "outcomes": _loads(outs), "prices": [float(p) for p in _loads(prices)],
                                                     "token_ids": _loads(toks), "liquidity": liq, "bid_ask": spread, "active": active})
    for (gid, slug), ms in by_slug.items():
        store.add_pm_markets(gid, slug, [m for m in ms if len(m["outcomes"]) == 2])
    store.commit()


if __name__ == "__main__":
    db, folder = sys.argv[1], sys.argv[2]
    def rd(name):
        try:
            return json.load(open(f"{folder}/{name}.json"))
        except FileNotFoundError:
            return None
    load(Store(db), rd("sched"), rd("box"), rd("sk"), rd("gl"), rd("inj"), rd("odds"), rd("ot"), rd("pm"))
    print("loaded")
