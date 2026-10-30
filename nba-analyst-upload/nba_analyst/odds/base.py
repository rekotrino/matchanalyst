"""Quote model + conversions shared by all providers and both sports."""

from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class Quote:
    game_id: str            # ESPN game id
    book: str               # 'bet365', 'pinnacle', 'polymarket', 'stake', ...
    market: str             # 'moneyline' | 'spreads' | 'totals' | prop stat key ('pts','reb','ast','tpm','stl','blk','pra','g','a','p','sog','sv','blk','hits')
    side: str               # team abbr for moneyline/spreads; 'over'/'under' for totals and props
    price: float            # decimal odds (e.g. 1.91)
    line: Optional[float] = None     # handicap / total / prop line
    player: Optional[str] = None     # prop player display name
    team: Optional[str] = None       # team abbr the line applies to (spreads) or player's team
    liquidity: Optional[float] = None
    note: Optional[str] = None
    ts: Optional[str] = None         # when this price was fetched (ISO); None = unknown

    @property
    def implied(self):
        return 1.0 / self.price if self.price and self.price > 1.0 else None

    def row(self):
        return asdict(self)


def american_to_decimal(a):
    a = float(a)
    return 1 + a / 100 if a > 0 else 1 + 100 / abs(a)


def decimal_to_american(d):
    return round((d - 1) * 100) if d >= 2 else round(-100 / (d - 1))


def prob_to_decimal(p):
    return 1.0 / p if p and p > 0 else None


def devig_two_way(p1, p2):
    s = p1 + p2
    return (p1 / s, p2 / s) if s else (p1, p2)


def best_quote(quotes):
    """Highest decimal price among quotes (best for the bettor)."""
    quotes = [q for q in quotes if q.price and q.price > 1.0]
    return max(quotes, key=lambda q: q.price) if quotes else None
