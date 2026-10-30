"""SQLite store. Every row carries fetched_at so we always know *when* we knew
something - that is what makes later backtests and CLV grading honest."""

import json
import os
import sqlite3
from datetime import datetime, timedelta, timezone

SCHEMA = """
CREATE TABLE IF NOT EXISTS games (
    game_id TEXT PRIMARY KEY,
    date_utc TEXT NOT NULL,
    date_et TEXT NOT NULL,
    season_year INTEGER,
    season_type INTEGER,
    home TEXT NOT NULL,
    away TEXT NOT NULL,
    home_score INTEGER,
    away_score INTEGER,
    completed INTEGER NOT NULL DEFAULT 0,
    status TEXT,
    fetched_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS team_game_stats (
    game_id TEXT NOT NULL,
    team TEXT NOT NULL,
    is_home INTEGER NOT NULL,
    pts INTEGER, fgm INTEGER, fga INTEGER, tpm INTEGER, tpa INTEGER,
    ftm INTEGER, fta INTEGER, oreb INTEGER, dreb INTEGER, reb INTEGER,
    ast INTEGER, stl INTEGER, blk INTEGER, tov INTEGER,
    pts_paint INTEGER, fast_break INTEGER, pts_off_tov INTEGER, fouls INTEGER,
    fetched_at TEXT NOT NULL,
    PRIMARY KEY (game_id, team)
);
CREATE TABLE IF NOT EXISTS player_game_stats (
    game_id TEXT NOT NULL,
    team TEXT NOT NULL,
    player_id TEXT NOT NULL,
    player TEXT NOT NULL,
    position TEXT,
    starter INTEGER,
    dnp INTEGER NOT NULL DEFAULT 0,
    min REAL, pts INTEGER, reb INTEGER, oreb INTEGER, dreb INTEGER, ast INTEGER,
    stl INTEGER, blk INTEGER, tov INTEGER, fgm INTEGER, fga INTEGER, tpm INTEGER, tpa INTEGER,
    ftm INTEGER, fta INTEGER, pf INTEGER, plus_minus INTEGER,
    fetched_at TEXT NOT NULL,
    PRIMARY KEY (game_id, player_id)
);
CREATE TABLE IF NOT EXISTS injuries (
    game_id TEXT NOT NULL,
    team TEXT NOT NULL,
    player TEXT NOT NULL,
    status TEXT,
    details TEXT,
    fetched_at TEXT NOT NULL,
    PRIMARY KEY (game_id, team, player, fetched_at)
);
CREATE TABLE IF NOT EXISTS reference_odds (
    game_id TEXT NOT NULL,
    provider TEXT NOT NULL,
    spread_home REAL,
    total REAL,
    ml_home INTEGER,
    ml_away INTEGER,
    fetched_at TEXT NOT NULL,
    PRIMARY KEY (game_id, provider, fetched_at)
);
CREATE TABLE IF NOT EXISTS pm_markets (
    game_id TEXT NOT NULL,
    event_slug TEXT NOT NULL,
    market_id TEXT NOT NULL,
    market_type TEXT NOT NULL,
    line REAL,
    outcomes TEXT NOT NULL,
    prices TEXT NOT NULL,
    token_ids TEXT,
    liquidity REAL,
    bid_ask REAL,
    question TEXT,
    fetched_at TEXT NOT NULL,
    PRIMARY KEY (market_id, fetched_at)
);
CREATE TABLE IF NOT EXISTS book_odds (
    game_id TEXT NOT NULL, book TEXT NOT NULL, market TEXT NOT NULL, side TEXT NOT NULL,
    line REAL, player TEXT, team TEXT, price REAL NOT NULL, liquidity REAL, note TEXT,
    fetched_at TEXT NOT NULL,
    PRIMARY KEY (game_id, book, market, side, line, player, fetched_at)
);
CREATE TABLE IF NOT EXISTS predictions (
    game_id TEXT NOT NULL,
    exp_home_pts REAL, exp_away_pts REAL, exp_margin REAL, exp_total REAL,
    p_home_win REAL,
    inputs TEXT,
    created_at TEXT NOT NULL,
    PRIMARY KEY (game_id, created_at)
);
CREATE TABLE IF NOT EXISTS qualifying_log (
    game_id TEXT NOT NULL, date_et TEXT NOT NULL, player TEXT NOT NULL, team TEXT, opp TEXT,
    stat TEXT NOT NULL, line REAL NOT NULL, side TEXT NOT NULL,
    prob REAL NOT NULL, ev REAL, price REAL, book TEXT, implied REAL, robust REAL, hit10 REAL,
    created_at TEXT NOT NULL, actual REAL, result TEXT, graded_at TEXT,
    PRIMARY KEY (game_id, player, stat, line, side)
);
CREATE TABLE IF NOT EXISTS recs (
    system TEXT NOT NULL, game_id TEXT NOT NULL, date_et TEXT NOT NULL, tip_utc TEXT,
    player TEXT NOT NULL, team TEXT, opp TEXT, stat TEXT NOT NULL, side TEXT NOT NULL, line REAL NOT NULL,
    price REAL NOT NULL, book TEXT NOT NULL, prob REAL NOT NULL, ev REAL, implied REAL, meta TEXT,
    created_at TEXT NOT NULL,
    closing_price REAL, closing_line REAL, closing_at TEXT, actual REAL, result TEXT, graded_at TEXT,
    PRIMARY KEY (system, game_id, player, stat, side, line, book)
);
CREATE TABLE IF NOT EXISTS market_state (
    run_at TEXT NOT NULL, key TEXT NOT NULL, game_id TEXT NOT NULL, line REAL, price REAL, book TEXT,
    score REAL, gap_sd REAL, recommended INTEGER, injury TEXT,
    PRIMARY KEY (run_at, key)
);
CREATE TABLE IF NOT EXISTS picks (
    pick_id INTEGER PRIMARY KEY AUTOINCREMENT,
    game_id TEXT NOT NULL,
    market_id TEXT NOT NULL,
    market_type TEXT NOT NULL,
    line REAL,
    outcome TEXT NOT NULL,
    our_prob REAL NOT NULL,
    price REAL NOT NULL,
    edge REAL NOT NULL,
    stake REAL NOT NULL,
    mode TEXT NOT NULL DEFAULT 'paper',
    created_at TEXT NOT NULL,
    closing_price REAL,
    result TEXT,
    pnl REAL,
    graded_at TEXT,
    book TEXT
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
        self._migrate()

    def _migrate(self):
        """Add columns introduced after a database was first created."""
        cols = {r[1] for r in self.conn.execute("PRAGMA table_info(picks)")}
        if "book" not in cols:
            self.conn.execute("ALTER TABLE picks ADD COLUMN book TEXT")
            self.conn.commit()

    # -- games -------------------------------------------------------------
    def upsert_game(self, g):
        self.conn.execute(
            """INSERT INTO games (game_id,date_utc,date_et,season_year,season_type,home,away,
               home_score,away_score,completed,status,fetched_at)
               VALUES (:game_id,:date_utc,:date_et,:season_year,:season_type,:home,:away,
               :home_score,:away_score,:completed,:status,:fetched_at)
               ON CONFLICT(game_id) DO UPDATE SET
               date_utc=excluded.date_utc, date_et=excluded.date_et,
               home_score=excluded.home_score, away_score=excluded.away_score,
               completed=excluded.completed, status=excluded.status,
               fetched_at=excluded.fetched_at""",
            dict(g, fetched_at=now_iso()),
        )

    def games_completed(self, since_et=None):
        q = "SELECT * FROM games WHERE completed=1"
        args = []
        if since_et:
            q += " AND date_et>=?"
            args.append(since_et)
        return [dict(r) for r in self.conn.execute(q + " ORDER BY date_utc", args)]

    def games_upcoming(self, from_et, to_et):
        return [dict(r) for r in self.conn.execute(
            "SELECT * FROM games WHERE completed=0 AND date_et>=? AND date_et<=? ORDER BY date_utc",
            (from_et, to_et))]

    def games_between(self, from_et, to_et):
        """Every game in the window, finished or not."""
        return [dict(r) for r in self.conn.execute(
            "SELECT * FROM games WHERE date_et>=? AND date_et<=? ORDER BY date_utc", (from_et, to_et))]

    def game_player_stats(self, game_id):
        return [dict(r) for r in self.conn.execute("SELECT * FROM player_game_stats WHERE game_id=?", (game_id,))]

    def add_qualifying(self, game, rows):
        ts = now_iso()
        for q in rows:
            self.conn.execute(
                """INSERT OR IGNORE INTO qualifying_log (game_id,date_et,player,team,opp,stat,line,side,prob,ev,price,book,
                   implied,robust,hit10,created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (game["game_id"], game["date_et"], q["player"], q["team"], q["opp"], q["stat"], q["line"], q["side"],
                 q["p"], q["ev"], q["price"], q["book"], q["implied"], q["robust"], q["hit10"], ts))

    def ungraded_qualifying(self):
        return [dict(r) for r in self.conn.execute(
            """SELECT q.* FROM qualifying_log q JOIN games g ON g.game_id=q.game_id
               WHERE q.result IS NULL AND g.completed=1""")]

    def grade_qualifying(self, game_id, player, stat, line, side, actual, result):
        self.conn.execute(
            "UPDATE qualifying_log SET actual=?, result=?, graded_at=? WHERE game_id=? AND player=? AND stat=? AND line=? AND side=?",
            (actual, result, now_iso(), game_id, player, stat, line, side))

    def qualifying_history(self):
        return [dict(r) for r in self.conn.execute("SELECT * FROM qualifying_log ORDER BY date_et, created_at")]

    # -- recommendation log shared by the three systems ---------------------
    def add_recs(self, system, game, rows):
        """One paper bet per player per game per system: the first time a player is
        recommended his line and price are locked; later runs that move the line or
        switch the stat do not add a second exposure."""
        ts = now_iso()
        have = {r[0].lower() for r in self.conn.execute("SELECT player FROM recs WHERE system=? AND game_id=?", (system, game["game_id"]))}
        for q in rows:
            if q["player"].lower() in have:
                continue
            have.add(q["player"].lower())
            self.conn.execute(
                """INSERT OR IGNORE INTO recs (system,game_id,date_et,tip_utc,player,team,opp,stat,side,line,price,book,prob,ev,implied,meta,created_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (system, game["game_id"], game["date_et"], game.get("date_utc"), q["player"], q.get("team"), q.get("opp"),
                 q["stat"], q["side"], q["line"], q["price"], q["book"], q["p"], q.get("ev"), q.get("implied"),
                 json.dumps(q.get("meta") or {}), ts))

    def ungraded_recs(self):
        return [dict(r) for r in self.conn.execute(
            """SELECT r.*, g.completed, g.date_utc FROM recs r JOIN games g ON g.game_id=r.game_id
               WHERE r.result IS NULL AND g.completed=1""")]

    def grade_rec(self, r, actual, result, closing_price, closing_line, closing_at):
        self.conn.execute(
            """UPDATE recs SET actual=?, result=?, closing_price=?, closing_line=?, closing_at=?, graded_at=?
               WHERE system=? AND game_id=? AND player=? AND stat=? AND side=? AND line=? AND book=?""",
            (actual, result, closing_price, closing_line, closing_at, now_iso(),
             r["system"], r["game_id"], r["player"], r["stat"], r["side"], r["line"], r["book"]))

    def recs_history(self, system=None):
        q = "SELECT * FROM recs" + (" WHERE system=?" if system else "") + " ORDER BY date_et, created_at"
        return [dict(r) for r in self.conn.execute(q, (system,) if system else ())]

    def recs_open_keys(self, system):
        return {(r["game_id"], r["player"].lower(), r["stat"]) for r in self.conn.execute(
            "SELECT game_id, player, stat FROM recs WHERE system=?", (system,))}

    # -- odds history / closing prices -----------------------------------------
    def odds_history(self, game_id):
        return [dict(r) for r in self.conn.execute(
            "SELECT * FROM book_odds WHERE game_id=? ORDER BY fetched_at", (game_id,))]

    def closing_quote(self, game_id, book, market, side, player, line, tip_utc):
        """Last stored price at that book before tip-off: (price at the same line, line, price at the
        last line seen, fetched_at). Either price may be None."""
        rows = [dict(r) for r in self.conn.execute(
            """SELECT * FROM book_odds WHERE game_id=? AND book=? AND market=? AND side=? AND player=?
               AND fetched_at<=? ORDER BY fetched_at DESC""", (game_id, book, market, side, player or "", tip_utc or "9"))]
        if not rows:
            return None, None, None, None
        last_at = rows[0]["fetched_at"]
        last = [r for r in rows if r["fetched_at"] == last_at]
        same = next((r for r in last if r["line"] == line), None)
        main = min(last, key=lambda r: abs((r["line"] or 0) - (line or 0)))
        return (same["price"] if same else None), main["line"], main["price"], last_at

    def injuries_history(self, game_id):
        return [dict(r) for r in self.conn.execute(
            "SELECT * FROM injuries WHERE game_id=? ORDER BY fetched_at", (game_id,))]

    # -- market monitoring snapshots ---------------------------------------------
    def market_state_last(self, skip_latest=False):
        runs = [x[0] for x in self.conn.execute("SELECT DISTINCT run_at FROM market_state ORDER BY run_at DESC")]
        if skip_latest:
            runs = runs[1:]
        if not runs:
            return None, {}
        rows = [dict(r) for r in self.conn.execute("SELECT * FROM market_state WHERE run_at=?", (runs[0],))]
        return runs[0], {r["key"]: r for r in rows}

    def market_state_save(self, rows):
        ts = now_iso()
        for r in rows:
            self.conn.execute("INSERT OR REPLACE INTO market_state VALUES (?,?,?,?,?,?,?,?,?,?)",
                              (ts, r["key"], r["game_id"], r.get("line"), r.get("price"), r.get("book"), r.get("score"),
                               r.get("gap_sd"), 1 if r.get("recommended") else 0, r.get("injury")))
        # keep only the two most recent snapshots
        runs = [x[0] for x in self.conn.execute("SELECT DISTINCT run_at FROM market_state ORDER BY run_at DESC")]
        for old in runs[2:]:
            self.conn.execute("DELETE FROM market_state WHERE run_at=?", (old,))
        return ts

    def prune_odds(self, keep_days=14):
        """Price history for games that finished more than keep_days ago is no longer needed."""
        cutoff = (datetime.now(timezone.utc) - timedelta(days=keep_days)).strftime("%Y-%m-%d")
        self.conn.execute("DELETE FROM book_odds WHERE game_id IN (SELECT game_id FROM games WHERE completed=1 AND date_et<?)", (cutoff,))
        self.conn.execute("DELETE FROM pm_markets WHERE game_id IN (SELECT game_id FROM games WHERE completed=1 AND date_et<?)", (cutoff,))

    def games_missing_stats(self):
        return [dict(r) for r in self.conn.execute(
            """SELECT g.* FROM games g WHERE g.completed=1 AND NOT EXISTS
               (SELECT 1 FROM team_game_stats s WHERE s.game_id=g.game_id)""")]

    # -- stats -------------------------------------------------------------
    def upsert_team_stats(self, row):
        cols = list(row.keys()) + ["fetched_at"]
        vals = list(row.values()) + [now_iso()]
        self.conn.execute(
            f"INSERT OR REPLACE INTO team_game_stats ({','.join(cols)}) VALUES ({','.join('?'*len(cols))})",
            vals)

    def team_stats(self, since_et=None):
        q = """SELECT s.*, g.date_et, g.home, g.away, g.home_score, g.away_score, g.season_type
               FROM team_game_stats s JOIN games g ON g.game_id=s.game_id WHERE g.completed=1"""
        args = []
        if since_et:
            q += " AND g.date_et>=?"
            args.append(since_et)
        return [dict(r) for r in self.conn.execute(q + " ORDER BY g.date_utc", args)]

    # -- players -----------------------------------------------------------
    PLAYER_COLS = ["game_id", "team", "player_id", "player", "position", "starter", "dnp", "min", "pts",
                   "reb", "oreb", "dreb", "ast", "stl", "blk", "tov", "fgm", "fga", "tpm", "tpa", "ftm",
                   "fta", "pf", "plus_minus"]

    def upsert_player_stats(self, row):
        cols = self.PLAYER_COLS + ["fetched_at"]
        vals = [row.get(c) for c in self.PLAYER_COLS] + [now_iso()]
        self.conn.execute(
            f"INSERT OR REPLACE INTO player_game_stats ({','.join(cols)}) VALUES ({','.join('?'*len(cols))})",
            vals)

    def player_stats(self, team=None, since_et=None):
        q = """SELECT p.*, g.date_et, g.home, g.away, g.season_type, g.season_year, g.home_score, g.away_score
               FROM player_game_stats p JOIN games g ON g.game_id=p.game_id WHERE g.completed=1"""
        args = []
        if team:
            q += " AND p.team=?"
            args.append(team)
        if since_et:
            q += " AND g.date_et>=?"
            args.append(since_et)
        return [dict(r) for r in self.conn.execute(q + " ORDER BY g.date_utc", args)]

    def games_missing_player_stats(self):
        return [dict(r) for r in self.conn.execute(
            """SELECT g.* FROM games g WHERE g.completed=1 AND NOT EXISTS
               (SELECT 1 FROM player_game_stats p WHERE p.game_id=g.game_id)""")]

    # -- injuries / odds / markets ---------------------------------------
    def replace_injuries(self, game_id, rows):
        ts = now_iso()
        for r in rows:
            self.conn.execute(
                "INSERT OR REPLACE INTO injuries VALUES (?,?,?,?,?,?)",
                (game_id, r["team"], r["player"], r.get("status"), r.get("details"), ts))

    def latest_injuries(self, game_id):
        rows = [dict(r) for r in self.conn.execute(
            "SELECT * FROM injuries WHERE game_id=? ORDER BY fetched_at DESC", (game_id,))]
        if not rows:
            return []
        latest = rows[0]["fetched_at"]
        return [r for r in rows if r["fetched_at"] == latest]

    def add_reference_odds(self, game_id, provider, spread_home, total, ml_home, ml_away):
        self.conn.execute("INSERT OR REPLACE INTO reference_odds VALUES (?,?,?,?,?,?,?)",
                          (game_id, provider, spread_home, total, ml_home, ml_away, now_iso()))

    def latest_reference_odds(self, game_id):
        r = self.conn.execute(
            "SELECT * FROM reference_odds WHERE game_id=? ORDER BY fetched_at DESC LIMIT 1",
            (game_id,)).fetchone()
        return dict(r) if r else None

    def add_pm_markets(self, game_id, slug, markets):
        ts = now_iso()
        for m in markets:
            self.conn.execute(
                "INSERT OR REPLACE INTO pm_markets VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (game_id, slug, m["market_id"], m["market_type"], m.get("line"),
                 json.dumps(m["outcomes"]), json.dumps(m["prices"]),
                 json.dumps(m.get("token_ids")), m.get("liquidity"), m.get("bid_ask"),
                 m.get("question"), ts))

    def latest_pm_markets(self, game_id):
        rows = [dict(r) for r in self.conn.execute(
            "SELECT * FROM pm_markets WHERE game_id=? ORDER BY fetched_at DESC", (game_id,))]
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

    # -- predictions / picks ---------------------------------------------
    def add_prediction(self, game_id, pred, inputs):
        self.conn.execute(
            "INSERT OR REPLACE INTO predictions VALUES (?,?,?,?,?,?,?,?)",
            (game_id, pred["exp_home_pts"], pred["exp_away_pts"], pred["exp_margin"],
             pred["exp_total"], pred["p_home_win"], json.dumps(inputs), now_iso()))

    def add_pick(self, p):
        self.conn.execute(
            """INSERT INTO picks (game_id,market_id,market_type,line,outcome,our_prob,price,edge,stake,mode,created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (p["game_id"], p["market_id"], p["market_type"], p.get("line"), p["outcome"],
             p["our_prob"], p["price"], p["edge"], p["stake"], p.get("mode", "paper"), now_iso()))
        if p.get("book"):
            self.conn.execute("UPDATE picks SET book=? WHERE pick_id=last_insert_rowid()", (p["book"],))

    def open_picks(self):
        return [dict(r) for r in self.conn.execute(
            "SELECT p.*, g.home, g.away, g.home_score, g.away_score, g.completed FROM picks p "
            "JOIN games g ON g.game_id=p.game_id WHERE p.result IS NULL")]

    def grade_pick(self, pick_id, result, pnl, closing_price):
        self.conn.execute(
            "UPDATE picks SET result=?, pnl=?, closing_price=?, graded_at=? WHERE pick_id=?",
            (result, pnl, closing_price, now_iso(), pick_id))

    def pick_history(self):
        return [dict(r) for r in self.conn.execute(
            "SELECT * FROM picks ORDER BY created_at")]

    # -- multi-book odds ---------------------------------------------------
    def add_quotes(self, quotes):
        ts = now_iso()
        for q in quotes:
            self.conn.execute(
                "INSERT OR REPLACE INTO book_odds VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (q.game_id, q.book, q.market, q.side, q.line, q.player or "", q.team, q.price, q.liquidity, q.note, ts))

    def latest_quotes(self, game_id):
        """Latest price per (book, market, side, line, player) for a game."""
        from .odds.base import Quote as _Q
        rows = self.conn.execute("SELECT * FROM book_odds WHERE game_id=? ORDER BY fetched_at DESC", (game_id,)).fetchall()
        seen, out = set(), []
        for r in rows:
            k = (r["book"], r["market"], r["side"], r["line"], r["player"])
            if k in seen:
                continue
            seen.add(k)
            out.append(_Q(r["game_id"], r["book"], r["market"], r["side"], r["price"], line=r["line"],
                          player=r["player"] or None, team=r["team"], liquidity=r["liquidity"], note=r["note"], ts=r["fetched_at"]))
        return out

    def commit(self):
        self.conn.commit()
