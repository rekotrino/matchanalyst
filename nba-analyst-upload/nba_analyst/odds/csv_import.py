"""Manual odds import for books without a feed (Stake, Rollbit, a local bookie).

File: data/odds/<sport>_<YYYY-MM-DD>.csv with a header row:
  game,book,market,side,line,player,price
    game    : AWAY@HOME using ESPN codes, e.g. CLE@ATL
    book    : stake | rollbit | bet365 | ... (any label)
    market  : moneyline | spreads | totals | pts | reb | ast | tpm | stl | blk | pra | g | a | p | sog | sv | hits
    side    : team code for moneyline/spreads (the team the price is on); over | under for totals and props
    line    : handicap / total / prop line (blank for moneyline)
    player  : player name for props (blank otherwise)
    price   : decimal odds (1.91) or American (-110 / +120)

Example rows:
  CLE@ATL,stake,moneyline,CLE,,,2.10
  CLE@ATL,stake,pts,over,24.5,Donovan Mitchell,1.87
  WPG@TB,rollbit,sog,over,2.5,Nikita Kucherov,1.75
"""

import csv
import os

from .base import Quote, american_to_decimal


def load_csv(path, games):
    """games: upcoming games (to map AWAY@HOME -> game_id). Returns Quote list."""
    if not os.path.exists(path):
        return []
    key = {f"{g['away']}@{g['home']}": g["game_id"] for g in games}
    out, skipped = [], 0
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            gid = key.get((r.get("game") or "").strip().upper())
            price = (r.get("price") or "").strip()
            if not gid or not price:
                skipped += 1
                continue
            try:
                dec = float(price)
                if dec >= 100 or dec <= -100:
                    dec = american_to_decimal(dec)
            except ValueError:
                skipped += 1
                continue
            line = (r.get("line") or "").strip()
            out.append(Quote(gid, (r.get("book") or "manual").strip().lower(), (r.get("market") or "").strip().lower(),
                             (r.get("side") or "").strip().lower() if (r.get("market") or "").strip().lower() not in ("moneyline", "spreads") else (r.get("side") or "").strip().upper(),
                             dec, line=float(line) if line else None, player=(r.get("player") or "").strip() or None, note="manual"))
    return out
