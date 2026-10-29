"""Multi-book odds layer. Every venue is normalised to Quote rows so the
scanner and the dashboard compare our probability against the best available
price, whichever book offers it.

Providers:
  polymarket   - the markets already ingested (ask price, from mid + half spread)
  oddsapi      - The Odds API (https://the-odds-api.com): Bet365, Pinnacle, DraftKings,
                 FanDuel, Betfair exchange, BetMGM, Caesars, Unibet, William Hill, ... incl. player props.
                 Needs ODDS_API_KEY (free tier: 500 requests / month).
  csv          - manual import for books without a feed (Stake, Rollbit, local books):
                 data/odds/<sport>_<date>.csv
"""
