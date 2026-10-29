"""Synthetic data generator so the full pipeline can be exercised offline.
Teams get hidden true strengths; box scores are simulated from them; a few
upcoming games get synthetic Polymarket markets priced with deliberate noise
so the scanner has something to find. Nothing here touches the network."""

import random
from datetime import date, timedelta

from .ingest.espn import TEAM_NAMES
from .ingest.polymarket import NICK_TO_ESPN
from .storage import Store

ESPN_TO_NICK = {v: k for k, v in NICK_TO_ESPN.items()}


def _box(pts, poss, rng):
    fga = int(rng.gauss(88, 5))
    tpa = int(fga * rng.uniform(0.35, 0.48))
    tpm = int(tpa * rng.uniform(0.30, 0.42))
    fta = int(rng.gauss(22, 5))
    ftm = int(fta * 0.78)
    two_pts = pts - 3 * tpm - ftm
    fgm_two = max(0, two_pts // 2)
    fgm = fgm_two + tpm
    oreb = int(rng.gauss(10, 3))
    tov = max(5, int(poss - fga + oreb - 0.44 * fta))
    return {"pts": pts, "fgm": fgm, "fga": fga, "tpm": tpm, "tpa": tpa, "ftm": ftm, "fta": fta,
            "oreb": oreb, "dreb": int(rng.gauss(33, 4)), "reb": 0, "ast": int(rng.gauss(25, 4)),
            "stl": int(rng.gauss(7, 2)), "blk": int(rng.gauss(5, 2)), "tov": tov,
            "pts_paint": int(rng.gauss(48, 8)), "fast_break": int(rng.gauss(14, 5)),
            "pts_off_tov": int(rng.gauss(17, 5)), "fouls": int(rng.gauss(19, 3))}


def seed_demo(store: Store, seed=7, prior_games_per_team=20, cur_games_per_team=6, upcoming=5):
    rng = random.Random(seed)
    teams = list(TEAM_NAMES)
    true = {t: {"off": rng.gauss(0, 4), "def": rng.gauss(0, 4), "pace": rng.gauss(0, 2.5)} for t in teams}
    # ten players per team with latent minutes and per-minute rates
    roster = {}
    for t in teams:
        roster[t] = []
        for k in range(10):
            mins = max(8, 34 - 3 * k + rng.gauss(0, 2))
            roster[t].append({"pid": f"{t}-{k}", "name": f"{t} Player {k + 1}", "pos": rng.choice(["G", "F", "C"]),
                              "min": mins, "pts_r": rng.uniform(0.3, 0.9) if k < 3 else rng.uniform(0.2, 0.5),
                              "reb_r": rng.uniform(0.1, 0.35), "ast_r": rng.uniform(0.05, 0.3)})
    today = date(2026, 10, 19)
    gid = 900000

    def sim_players(game_id, team):
        for p in roster[team]:
            m = max(0, rng.gauss(p["min"], 4))
            row = {"game_id": game_id, "team": team, "player_id": p["pid"], "player": p["name"],
                   "position": p["pos"], "starter": int(p["min"] > 24), "dnp": 0, "min": round(m, 1),
                   "pts": max(0, int(round(rng.gauss(p["pts_r"] * m, 5)))),
                   "reb": max(0, int(round(rng.gauss(p["reb_r"] * m, 2.5)))),
                   "ast": max(0, int(round(rng.gauss(p["ast_r"] * m, 2)))),
                   "fga": int(p["pts_r"] * m / 1.1), "fta": int(m / 8), "tov": int(m / 15)}
            store.upsert_player_stats(row)

    def sim(d, home, away, season_type):
        nonlocal gid
        gid += 1
        pace = 99 + true[home]["pace"] + true[away]["pace"] + rng.gauss(0, 3)
        hp = pace / 100 * (114.5 + true[home]["off"] + true[away]["def"]) + 1.2 + rng.gauss(0, 8)
        ap = pace / 100 * (114.5 + true[away]["off"] + true[home]["def"]) - 1.2 + rng.gauss(0, 8)
        hp, ap = int(round(hp)), int(round(ap))
        if hp == ap:
            hp += 1
        g = {"game_id": str(gid), "date_utc": f"{d}T23:30Z", "date_et": str(d),
             "season_year": 2026 if d.year == 2026 and d.month < 8 else 2027, "season_type": season_type,
             "home": home, "away": away, "home_score": hp, "away_score": ap, "completed": 1, "status": "STATUS_FINAL"}
        store.upsert_game(g)
        for t, pts, is_home in ((home, hp, 1), (away, ap, 0)):
            row = {"game_id": str(gid), "team": t, "is_home": is_home}
            row.update(_box(pts, pace, rng))
            store.upsert_team_stats(row)
            sim_players(str(gid), t)

    # prior season: Feb-Apr 2026
    d = date(2026, 2, 1)
    for _ in range(prior_games_per_team * 15):
        a, b = rng.sample(teams, 2)
        sim(d, a, b, 2)
        d += timedelta(days=rng.choice([0, 0, 1]))
        if d > date(2026, 4, 12):
            d = date(2026, 2, 1)
    # current season: preseason + a few regular games
    d = date(2026, 10, 3)
    for i in range(cur_games_per_team * 15):
        a, b = rng.sample(teams, 2)
        sim(d, a, b, 1 if d < date(2026, 10, 14) else 2)
        d += timedelta(days=rng.choice([0, 1]))
        if d >= today:
            d = date(2026, 10, 14)

    # upcoming games with synthetic Polymarket markets
    for i in range(upcoming):
        home, away = rng.sample(teams, 2)
        gid += 1
        gd = today + timedelta(days=i // 3)
        g = {"game_id": str(gid), "date_utc": f"{gd}T23:30Z", "date_et": str(gd), "season_year": 2027,
             "season_type": 2, "home": home, "away": away, "home_score": None, "away_score": None,
             "completed": 0, "status": "STATUS_SCHEDULED"}
        store.upsert_game(g)
        true_margin = (true[home]["off"] + true[away]["def"] - true[away]["off"] - true[home]["def"]) + 2.3
        p_home = 0.5 * (1 + __import__("math").erf(true_margin / 13 / 2 ** 0.5))
        noise = rng.gauss(0, 0.06)
        mkt_p = min(0.95, max(0.05, p_home + noise))
        hn, an = ESPN_TO_NICK[home], ESPN_TO_NICK[away]
        fav, dog = (hn, an) if true_margin > 0 else (an, hn)
        spread = -(abs(round(true_margin * 2) / 2) + rng.choice([-1, 0, 1]))
        spread = spread if spread % 1 else spread - 0.5
        total = round(2 * (99 + true[home]["pace"] + true[away]["pace"]) / 100 * 114.5 + rng.gauss(0, 4)) + 0.5
        markets = [
            {"market_id": f"{gid}-ml", "question": f"{an} vs. {hn}", "market_type": "moneyline",
             "line": None, "outcomes": [an, hn], "prices": [round(1 - mkt_p, 3), round(mkt_p, 3)],
             "liquidity": rng.uniform(2000, 12000), "bid_ask": 0.02, "active": True},
            {"market_id": f"{gid}-sp", "question": f"Spread: {fav} ({spread})", "market_type": "spreads",
             "line": spread, "outcomes": [fav, dog], "prices": [round(0.5 + rng.gauss(0, 0.05), 3), 0],
             "liquidity": rng.uniform(1000, 8000), "bid_ask": 0.03, "active": True},
            {"market_id": f"{gid}-tot", "question": f"{an} vs. {hn}: O/U {total}", "market_type": "totals",
             "line": total, "outcomes": ["Over", "Under"], "prices": [round(0.5 + rng.gauss(0, 0.05), 3), 0],
             "liquidity": rng.uniform(1000, 6000), "bid_ask": 0.02, "active": True},
            {"market_id": f"{gid}-pp", "question": f"{roster[home][0]['name']}: Points O/U {round(roster[home][0]['pts_r'] * roster[home][0]['min']) - 2.5}",
             "market_type": "points", "line": round(roster[home][0]['pts_r'] * roster[home][0]['min']) - 2.5,
             "outcomes": ["Yes", "No"], "prices": [0.5, 0.5], "liquidity": 900.0, "bid_ask": 0.04, "active": True},
            {"market_id": f"{gid}-pr", "question": f"{roster[away][1]['name']}: Rebounds O/U 6.5",
             "market_type": "rebounds", "line": 6.5, "outcomes": ["Yes", "No"], "prices": [0.5, 0.5],
             "liquidity": 8.0, "bid_ask": 0.1, "active": True},
        ]
        for m in markets:
            if m["prices"][1] == 0:
                m["prices"][1] = round(1 - m["prices"][0], 3)
        store.add_pm_markets(str(gid), f"nba-{away.lower()}-{home.lower()}-{gd}", markets)
        store.add_reference_odds(str(gid), "DemoBook", spread if true_margin > 0 else -spread, total,
                                 -150 if mkt_p > 0.5 else 130, 130 if mkt_p > 0.5 else -150)
        store.replace_injuries(str(gid), [{"team": home, "player": "Demo Player", "status": "Day-To-Day",
                                           "details": "Ankle"}])
    store.commit()
    return true
