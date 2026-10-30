"""SQLite store for the NHL engine. Timestamped like the NBA store."""

import json
import os
import sqlite3
from datetime import datetime, timezone

SCHEMA = """
CREATE TABLE IF NOT EXISTS games (
    game_id TEXT PRIMARY KEY, date_utc TEXT NOT NULL, date_et TEXT NOT NULL,
    season_year INTEGER, season_type INTEGER, home TEXT NOT NULL, away TEXT NOT NULL,
    home_score INTEGER, away_score INTEGER, completed INTEGER NOT NULL DEFAULT 0,
    status TEXT, finish TEXT, fetched_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS team_game_stats (
    game_id TEXT NOT NULL, team TEXT NOT NULL, is_home INTEGER NOT NULL,
    goals INTEGER, shots INTEGER, ppg INTEGER, ppo INTEGER, hits INTEGER, blocked INTEGER,
    fow INTEGER, giveaways INTEGER, takeaways INTEGER, pim INTEGER,
    fetched_at TEXT NOT NULL, PRIMARY KEY (game_id, team)
);
CREATE TABLE IF NOT EXISTS skater_game_stats (
    game_id TEXT NOT NULL, team TEXT NOT NULL, player_id TEXT NOT NULL, player TEXT NOT NULL,
    position TEXT, toi REAL, g INTEGER, a INTEGER, sog INTEGER, missed INTEGER, blk INTEGER,
    hits INTEGER, pp_toi REAL, plus_minus INTEGER, pim INTEGER,
    fetched_at TEXT NOT NULL, PRIMARY KEY (game_id, player_id)
);
CREATE TABLE IF NOT EXISTS goalie_game_stats (
    game_id TEXT NOT NULL, team TEXT NOT NULL, player_id TEXT NOT NULL, player TEXT NOT NULL,
    toi REAL, ga INTEGER, sa INTEGER, sv INTEGER,
    fetched_at TEXT NOT NULL, PRIMARY KEY (game_id, player_id)
);
CREATE TABLE IF NOT EXISTS injuries (
    game_id TEXT NOT NULL, team TEXT NOT NULL, player TEXT NOT NULL, status TEXT, details TEXT,
    fetched_at TEXT NOT NULL, PRIMARY KEY (game_id, team, player, fetched_at)
);
CREATE TABLE IF NOT EXISTS reference_odds (
    game_id TEXT NOT NULL, provider TEXT NOT NULL, spread_home REAL, total REAL, ml_home INTEGER, ml_away INTEGER,
    fetched_at TEXT NOT NULL, PRIMARY KEY (game_id, provider, fetched_at)
);
CREATE TABLE IF NOT EXISTS pm_markets (
    game_id TEXT NOT NULL, event_slug TEXT NOT NULL, market_id TEXT NOT NULL, market_type TEXT NOT NULL,
    line REAL, outcomes TEXT NOT NULL, prices TEXT NOT NULL, token_ids TEXT, liquidity REAL, bid_ask REAL,
    question TEXT, fetched_at TEXT NOT NULL, PRIMARY KEY (market_id, fetched_at)
);
CREATE TABLE IF NOT EXISTS book_odds (
    game_id TEXT NOT NULL, book TEXT NOT NULL, market TEXT NOT NULL, side TEXT NOT NULL,
    line REAL, player TEXT, team TEXT, price REAL NOT NULL, liquidity REAL, note TEXT,
    fetched_at TEXT NOT NULL,
    PRIMARY KEY (game_id, book, market, side, line, player, fetched_at)
);
CREATE TABLE IF NOT EXISTS predictions (
    game_id TEXT NOT NULL, exp_home REAL, exp_away REAL, p_home_win REAL, inputs TEXT,
    created_at TEXT NOT NULL, PRIMARY KEY (game_id, created_at)
);
CREATE TABLE IF NOT EXISTS picks (
    pick_id INTEGER PRIMARY KEY AUTOINCREMENT, game_id TEXT NOT NULL, market_id TEXT NOT NULL,
    market_type TEXT NOT NULL, line REAL, outcome TEXT NOT NULL, our_prob REAL NOT NULL, price REAL NOT NULL,
    edge REAL NOT NULL, stake REAL NOT NULL, mode TEXT NOT NULL DEFAULT 'paper', created_at TEXT NOT NULL,
    closing_price REAL, result TEXT, pnl REAL, graded_at TEXT, book TEXT
);
"""


def now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Store:
    def __init__(self, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        cols = {r[1] for r in self.conn.execute("PRAGMA table_info(picks)")}
        if "book" not in cols:
            self.conn.execute("ALTER TABLE picks ADD COLUMN book TEXT")
            self.conn.commit()

    def _ins(self, table, row):
        cols = list(row) + ["fetched_at"]
        self.conn.execute(f"INSERT OR REPLACE INTO {table} ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})",
                          list(row.values()) + [now_iso()])

    def upsert_game(self, g):
        self._ins("games", g)

    def upsert_team_stats(self, r):
        self._ins("team_game_stats", r)

    def upsert_skater(self, r):
        self._ins("skater_game_stats", r)

    def upsert_goalie(self, r):
        self._ins("goalie_game_stats", r)

    def replace_injuries(self, game_id, rows):
        for r in rows:
            self._ins("injuries", {"game_id": game_id, "team": r["team"], "player": r["player"],
                                   "status": r.get("status"), "details": r.get("details")})

    def add_reference_odds(self, game_id, provider, spread_home, total, ml_home, ml_away):
        self._ins("reference_odds", {"game_id": game_id, "provider": provider, "spread_home": spread_home,
                                     "total": total, "ml_home": ml_home, "ml_away": ml_away})

    def add_pm_markets(self, game_id, slug, markets):
        for m in markets:
            self._ins("pm_markets", {"game_id": game_id, "event_slug": slug, "market_id": m["market_id"],
                                     "market_type": m["market_type"], "line": m.get("line"),
                                     "outcomes": json.dumps(m["outcomes"]), "prices": json.dumps(m["prices"]),
                                     "token_ids": json.dumps(m.get("token_ids")), "liquidity": m.get("liquidity"),
                                     "bid_ask": m.get("bid_ask"), "question": m.get("question")})

    def games_completed(self):
        return [dict(r) for r in self.conn.execute("SELECT * FROM games WHERE completed=1 ORDER BY date_utc")]

    def games_upcoming(self, from_et, to_et):
        return [dict(r) for r in self.conn.execute(
            "SELECT * FROM games WHERE completed=0 AND date_et>=? AND date_et<=? ORDER BY date_utc", (from_et, to_et))]

    def games_missing_stats(self):
        return [dict(r) for r in self.conn.execute(
            "SELECT g.* FROM games g WHERE g.completed=1 AND NOT EXISTS (SELECT 1 FROM skater_game_stats s WHERE s.game_id=g.game_id)")]

    def team_stats(self):
        return [dict(r) for r in self.conn.execute(
            """SELECT s.*, g.date_et, g.home, g.away, g.home_score, g.away_score, g.finish, g.season_type
               FROM team_game_stats s JOIN games g ON g.game_id=s.game_id WHERE g.completed=1 ORDER BY g.date_utc""")]

    def skater_stats(self):
        return [dict(r) for r in self.conn.execute(
            """SELECT s.*, g.date_et, g.home, g.away, g.season_type FROM skater_game_stats s
               JOIN games g ON g.game_id=s.game_id WHERE g.completed=1 ORDER BY g.date_utc""")]

    def goalie_stats(self):
        return [dict(r) for r in self.conn.execute(
            """SELECT s.*, g.date_et, g.home, g.away, g.season_type FROM goalie_game_stats s
               JOIN games g ON g.game_id=s.game_id WHERE g.completed=1 ORDER BY g.date_utc""")]

    def latest_injuries(self, game_id):
        rows = [dict(r) for r in self.conn.execute("SELECT * FROM injuries WHERE game_id=? ORDER BY fetched_at DESC", (game_id,))]
        return [r for r in rows if r["fetched_at"] == rows[0]["fetched_at"]] if rows else []

    def latest_reference_odds(self, game_id):
        r = self.conn.execute("SELECT * FROM reference_odds WHERE game_id=? ORDER BY fetched_at DESC LIMIT 1", (game_id,)).fetchone()
        return dict(r) if r else None

    def latest_pm_markets(self, game_id):
        rows = [dict(r) for r in self.conn.execute("SELECT * FROM pm_markets WHERE game_id=? ORDER BY fetched_at DESC", (game_id,))]
        seen, out = set(), []
        for r in rows:
            if r["market_id"] in seen:
                continue
            seen.add(r["market_id"])
            r["outcomes"] = json.loads(r["outcomes"])
            r["prices"] = [float(p) for p in json.loads(r["prices"])]
            r["token_ids"] = json.loads(r["token_ids"]) if r["token_ids"] else None
            out.append(r)
        return out

    def add_prediction(self, game_id, pred, inputs):
        self.conn.execute("INSERT OR REPLACE INTO predictions VALUES (?,?,?,?,?,?)",
                          (game_id, pred["exp_home"], pred["exp_away"], pred["p_home_win"], json.dumps(inputs), now_iso()))

    def add_pick(self, p):
        self.conn.execute(
            "INSERT INTO picks (game_id,market_id,market_type,line,outcome,our_prob,price,edge,stake,mode,created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (p["game_id"], p["market_id"], p["market_type"], p.get("line"), p["outcome"], p["our_prob"], p["price"],
             p["edge"], p["stake"], p.get("mode", "paper"), now_iso()))
        if p.get("book"):
            self.conn.execute("UPDATE picks SET book=? WHERE pick_id=last_insert_rowid()", (p["book"],))

    def open_picks(self):
        return [dict(r) for r in self.conn.execute(
            "SELECT p.*, g.home, g.away, g.home_score, g.away_score, g.completed, g.finish FROM picks p JOIN games g ON g.game_id=p.game_id WHERE p.result IS NULL")]

    def grade_pick(self, pick_id, result, pnl, closing_price):
        self.conn.execute("UPDATE picks SET result=?, pnl=?, closing_price=?, graded_at=? WHERE pick_id=?",
                          (result, pnl, closing_price, now_iso(), pick_id))

    def pick_history(self):
        return [dict(r) for r in self.conn.execute("SELECT * FROM picks ORDER BY created_at")]

    # -- multi-book odds ---------------------------------------------------
    def add_quotes(self, quotes):
        ts = now_iso()
        for q in quotes:
            self.conn.execute(
                "INSERT OR REPLACE INTO book_odds VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (q.game_id, q.book, q.market, q.side, q.line, q.player or "", q.team, q.price, q.liquidity, q.note, ts))

    def latest_quotes(self, game_id):
        """Latest price per (book, market, side, line, player) for a game."""
        from nba_analyst.odds.base import Quote as _Q
        rows = self.conn.execute("SELECT * FROM book_odds WHERE game_id=? ORDER BY fetched_at DESC", (game_id,)).fetchall()
        seen, out = set(), []
        for r in rows:
            k = (r["book"], r["market"], r["side"], r["line"], r["player"])
            if k in seen:
                continue
            seen.add(k)
            out.append(_Q(r["game_id"], r["book"], r["market"], r["side"], r["price"], line=r["line"],
                          player=r["player"] or None, team=r["team"], liquidity=r["liquidity"], note=r["note"]))
        return out

    def commit(self):
        self.conn.commit()
