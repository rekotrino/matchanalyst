"""Offline checks for the odds layer: The Odds API parsing (documented response
shape), CSV import, Polymarket adapter, and best-price ladders."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nba_analyst.odds.the_odds_api import parse_prop_event
from nba_analyst.odds.csv_import import load_csv
from nba_analyst.odds.base import Quote, best_quote, american_to_decimal
from nba_analyst.odds.ladders import build_prop_items

game = {"game_id": "401909893", "home": "ATL", "away": "CLE", "date_utc": "2026-10-30T00:00Z"}
ev = {"id": "abc", "home_team": "Atlanta Hawks", "away_team": "Cleveland Cavaliers", "bookmakers": [
    {"key": "bet365", "title": "Bet365", "markets": [{"key": "player_points", "outcomes": [
        {"name": "Over", "description": "Donovan Mitchell", "price": 1.91, "point": 24.5},
        {"name": "Under", "description": "Donovan Mitchell", "price": 1.91, "point": 24.5}]}]},
    {"key": "pinnacle", "title": "Pinnacle", "markets": [{"key": "player_points", "outcomes": [
        {"name": "Over", "description": "Donovan Mitchell", "price": 1.95, "point": 24.5},
        {"name": "Under", "description": "Donovan Mitchell", "price": 1.88, "point": 24.5}]}]}]}
qs = parse_prop_event(ev, game, "nba")
assert len(qs) == 4 and qs[0].market == "pts" and qs[0].player == "Donovan Mitchell" and qs[0].line == 24.5
assert best_quote([q for q in qs if q.side == "over"]).book == "pinnacle"
csvq = load_csv(os.path.join(os.path.dirname(__file__), "..", "data", "odds", "nba_example.csv.txt"), [game])
assert len(csvq) == 3 and abs(csvq[2].price - american_to_decimal(-115)) < 1e-9 and csvq[0].side == "CLE"

class Cfg: min_liquidity = 500; prob_floor = 0.55; edge_threshold = 0.045
fns = {("donovan mitchell", "pts"): {"p_fn": lambda line: 0.62 if line == 24.5 else 0.5, "hit_fn": lambda line, n: (0.6, 10),
       "player_id": "1", "player": "Donovan Mitchell", "team": "CLE", "opp": "ATL", "position": "G",
       "proj": {"mean": 26.1, "sd": 6.0, "minutes": 34}, "recent": [25, 30, 22], "l10": 26.0, "l5": 27.0, "season": 25.5, "l10_sd": 5.0}}
items = build_prop_items(game, fns, qs + csvq, {}, Cfg, lambda s: "Points", set())
lad = items[0]["ladder"][0]
assert lad["best_over_book"] == "pinnacle" and lad["over_price"] == 1.95 and set(lad["books"]) == {"bet365", "pinnacle", "stake"} and lad["books"]["stake"]["over"] == 1.87
assert abs(lad["edge_over"] - (0.62 - 1 / 1.95)) < 1e-3 and items[0]["safe_pick"]["line"] == 24.5
print("odds layer OK:", len(qs), "api quotes,", len(csvq), "csv quotes, best over", lad["best_over_book"], lad["over_price"])
