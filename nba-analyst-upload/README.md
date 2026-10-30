# Sports AI Analyst — NBA + NHL (Phase 0)

Pre-match analysis engine for NBA games. It pulls schedule, results, team box
scores, injuries and a reference line from ESPN's public API, pulls every
market Polymarket lists for each game, builds opponent-adjusted team ratings,
prices every game-level market from an expected-score distribution, and writes
a pre-match report per game plus a daily digest with paper picks.

Scope so far: NBA; game-level markets (moneyline, spreads, totals) and player
props (points, rebounds, assists) priced from player box scores; paper picks
only. No real money is placed anywhere.

## Setup (Windows)

1. Install Python 3.11 or newer from python.org (tick "Add Python to PATH").
2. Open PowerShell in this folder and run:

       pip install -r requirements.txt

3. Daily run (ingest, analyse, write reports):

       python -m nba_analyst run

   The first run pulls last season's Feb–Apr games as the rating baseline
   (about 500 box scores, 5–10 minutes). Later runs only fetch what is new.

4. Reports land in `reports\<date>\` (one file per game) and
   `reports\<date>_digest.md`. The database is `data\nba_analyst.sqlite`.

Other commands:

    python -m nba_analyst update      # ingest only
    python -m nba_analyst analyze     # re-run analysis on stored data
    python -m nba_analyst grade       # grade finished paper picks, print the record
    python -m nba_analyst demo        # offline synthetic run (no network)

Dashboard (reads the same database, no extra installs):

The Props tab is the landing view and follows one flow: matches list → open a
match → its players → each player's previous games → the lowest line they are
very likely to clear ("floor we take") with the price and payout, plus a value
line and an under where one exists. Only lines with real liquidity are
suggested; thin rungs are shown greyed. A player who is day-to-day, or whose
liquid line is priced far below our number ("market doubt": usually a late
scratch or minutes limit the stats don't know yet), is flagged and gets no
suggestion until you check the news.


    python -m nba_analyst serve --with-nhl
    # then open http://127.0.0.1:8765 in your browser; the NBA | NHL switch is in the header

Props (above), Board (picks, model-vs-market chart, games), Games (model view, team
profiles, rotations, full market scan), Players (rotation tables and game
logs per team), Picks (history, P&L, CLV). The Refresh button recomputes from
the database; run `python -m nba_analyst run` first to fetch new data.
`python -m nba_analyst export-html reports\board.html` writes a standalone
copy with the data embedded, for sharing.

Player analytics from the command line:

    python -m nba_analyst players --team IND                  # rotation table, last-10 averages and trends
    python -m nba_analyst players --name "Pascal Siakam"      # game log, last-5/10/season splits, usage, TS%
    python -m nba_analyst players --name "Pascal Siakam" --stat pts --line 20.5   # projection and P(over)

If this machine can't reach the APIs, load a snapshot collected elsewhere:

    python -m nba_analyst.load_snapshot data\nba_analyst.sqlite data\snapshot_2026-10-18

## Simulation engine (player props)

Every rotation player in every game is simulated 10,000 times
(`nba_analyst/analyst/sim.py`). A simulated game draws minutes (projected from
regular-season games, with blowout, foul-trouble and back-to-back
disruptions), a shared hot/cold usage factor, Poisson shot volume (FGA, 3PA
share, FTA) scaled by minutes, pace and the opponent's defence against the
player's position, and Beta-distributed shooting efficiency around the
player's own percentages. Points are built from the made shots, so points,
threes and combo props stay correlated; rebounds, assists, steals, blocks and
turnovers are overdispersed Poisson counts. Last season is a weak prior when
the current season is thin. Every ladder shows the mean, median, 25th/75th
percentiles, floor (10th) and ceiling (90th), P(over)/P(under) for every book
line and for alternative lines, and the same probabilities under stress
scenarios (more/fewer minutes, blowout, tough or favourable matchup, a
questionable teammate sitting).

A bet is **qualifying** only when every check passes on one rung: simulated
probability ≥ 65%, EV ≥ +3% at the best price across books (implied
probability is taken with the bookmaker margin removed), the player's real
last-10 hit rate on that exact line ≥ 60%, the worst stress scenario still
≥ 55%, the matchup is not adverse, availability is verified (no injury flag,
stable role, no unpredictable minutes, 8+ games with the team), and the edge
is plausible (an edge above 25 points means a bad line, not a good bet).
Overs and unders are tested on equal terms. When nothing passes the board
says **NO QUALIFYING BETS**. Qualifying bets are logged
(`qualifying_log`) and graded after the game, and the Props tab has a day
switcher so yesterday's results sit next to today's slate. All thresholds are
in `BettingConfig` (`config.py`).

### Backtest and calibration

    python -m nba_analyst backtest

replays every completed regular-season game walk-forward (only earlier games
are used for each prediction), scores the simulated probabilities (Brier,
log loss, calibration by bucket, hit rate of selections at the 65%
threshold for overs and unders) and fits a Platt recalibration that the live
pipeline reads from `data/calibration.json`. On 236k evaluations from the
2025-26 and early 2026-27 seasons the raw simulator was within 1 point of
observed frequencies up to 85% and slightly over-confident above that; the
fitted correction (a=+0.13, b=0.88) is applied automatically. Re-run after
each month of games. The ROI figure in the report is at a hypothetical -110
on our own alternative lines; it is a calibration check, not a market result.

Not in the data (and therefore not in the model): defensive schemes and
primary-defender assignments, confirmed starting lineups before tip-off, and
minutes restrictions that are not on ESPN's injury report. Treat "availability
verified" as "no red flag in the data", and still check the news for a late
scratch.

## Profitability & Value Intelligence (the Value tab)

The engine answers "what is likely to happen"; the Value tab answers "which of
those is worth betting on at the prices actually available"
(`nba_analyst/value.py`, reusing the engine's samples and quotes, nothing is
re-simulated). For every priced selection (player-prop rungs and game lines):
best price per side across books, implied probability with the margin removed
where both sides of a line are quoted at one book, EV = P(win) × profit −
P(loss) × stake per unit, ROI = EV × 100, and a risk class (low ≥ 70% and
robust to stress scenarios; medium ≥ 60%; high below; stale = price older
than 6 hours; suspect = edge above 25 points, i.e. a bad line; unverified =
injury/role flag). The table sorts and filters by probability, EV, ROI,
player, game, market, side, book and risk; a row expands to the full
simulation evidence and a "Track this bet" button.

Parlays are built per sportsbook (a parlay lives at one book; Polymarket has
no parlay product) from the simulated hit indicators, one leg per player.
Legs in the same game are evaluated on the same 10,000 simulated games, with
game-level events (a blowout) shared by every player in that game, so the
joint probability already carries their correlation and the card reports how
far it sits from the independent product. Five categories: conservative,
balanced, higher return, probability-optimized, value-optimized; each shows
every leg, combined odds, joint probability, EV, ROI, payout and risk.

The betting tracker records bets you choose to take (stake, odds, book),
grades them automatically when the game is in the board's window, lets you
set results by hand otherwise, and charts cumulative P&L. It lives in the
browser's local storage, separate from simulations and backtests; export it
to JSON from time to time.

## How to read a report

- **Pure model** is what the ratings alone say. Early in the season it is
  mostly last season's ratings, so it does not know about trades, signings or
  long-term injuries.
- **Reference line** is DraftKings via ESPN, with the vig removed. A warning
  fires when the pure model disagrees with it by more than 10 points: that is
  usually a roster change the ratings haven't seen.
- **Working numbers** blend the two. The model's weight starts at 0% and grows
  with regular-season games played, capped at 60% until Phase 1 backtests
  earn it more. All scan probabilities use the working numbers.
- **Strengths / Issues** compare the team's last 10 games with the league
  norms of the same pool of games (preseason vs preseason, regular vs regular).
- **Player analytics** lists each team's rotation by recent minutes with
  last-10 points / rebounds / assists / PRA, last-5 averages, the trend
  (last-5 minus last-10) and notes: OUT, starter, minutes up or down, volatile
  scorer.
- **Player props** cover points, rebounds, assists, 3-pointers, steals and
  blocks (and combo markets when listed). They are priced per player: a recency-weighted per-minute rate
  (shrunk toward the season rate) times projected minutes (regular-season
  minutes only; preseason minutes are ignored, and a player with preseason
  games only is not priced), with a game-to-game standard deviation floored
  per stat and widened for small samples. A rotation teammate listed Out adds
  a small minutes and usage bump. The resulting probability is blended with
  the player's actual hit rate on that line over the last 10 games, so a hot
  projection can't run away from what has really been happening. Two context
  factors adjust the projection, each capped: what the opponent allows to the
  player's position (guards / forwards / centers) relative to league, shrunk
  for small samples, and the expected pace of the game. Props need 5+ games
  with the current team.
- **Polymarket scan** prices every listed market. A market becomes a
  *candidate* only when edge ≥ 4.5 pts, our probability ≥ 55%, and the market
  has ≥ $500 liquidity. Candidates are ranked by probability (favourites
  first), with at most one side market and one total per game and six picks
  per day. Stake is quarter-Kelly capped at 3% of bankroll.
- **Picks (paper)** are logged to the database with the price at the time.
  `grade` fills in the result, P&L and the last price seen before tip-off
  (closing price), so CLV can be tracked from day one.

## Knobs

Everything tunable is in `nba_analyst/config.py`: bankroll, edge threshold,
probability floor, liquidity floor, Kelly fraction, model standard deviations,
home advantage, rest penalties, rating shrinkage and the market-anchor cap.

## Layout

    nba_analyst/
      config.py            knobs
      storage.py           SQLite schema and queries (everything timestamped)
      ingest/espn.py       schedule, scores, box scores, injuries, reference odds
      ingest/polymarket.py game events and markets (slug nba-{away}-{home}-{date})
      analyst/ratings.py   per-game efficiency, rolling profiles, adjusted ratings, issue flags
      analyst/players.py   player game logs, rolling production, projections, hit rates
      model/game_model.py  expected score, market pricing, reference anchoring
      scanner.py           edge filter, favourites-first ranking, correlation guard, Kelly
      pipeline.py          update / analyze / grade orchestration
      report.py            markdown reports
      payload.py           JSON for the dashboard (read-only)
      web.py               local dashboard server + standalone export
      ui/dashboard.html    the dashboard page (single file, inline SVG charts)
      demo.py              synthetic data for offline testing
      load_snapshot.py     load a JSON data snapshot collected elsewhere

## What Phase 1 adds

Backtest against two seasons of closing lines to calibrate the game model and
the prop projections (matchup and pace adjustments for props come here too);
the bounded LLM news layer; the champion/challenger learning loop; and the
EPL engine.


## NHL engine

`nhl_analyst` is the hockey version of the same pipeline and shares the
dashboard page. Run it from this folder:

    python -m nhl_analyst run            # ingest (ESPN + Polymarket) + analyze
    python -m nhl_analyst serve          # dashboard at http://127.0.0.1:8766
    python -m nhl_analyst export-html reports\nhl_board.html
    python -m nhl_analyst.load_snapshot data\nhl_analyst.sqlite data\nhl_snapshot_2026-10-29

Differences from the NBA engine: team ratings are multiplicative attack /
defence log-ratings (shootout goals stripped), the game model is a Poisson
score matrix that prices moneyline with OT/shootout, totals (OT included,
shootout counts one goal) and puck lines; skater projections are per-60 rates
times projected ice time with Poisson probabilities for goals / assists /
points / shots / blocks / hits; the likely starting goalie (more starts, did
not play yesterday) gets a saves ladder. Polymarket lists no NHL player props
(checked 2026-10-29), so the Props view shows fair prices for standard lines:
a book paying more than the fair payout is value.


## Odds from more than Polymarket

Every venue is normalised into one quote table, and the scanner, the ladders
and the dashboard compare our probability against the **best price across
all books**, naming the book. Three sources:

1. **The Odds API** (https://the-odds-api.com, free key, 500 requests/month):
   Bet365 (under its uk / au / eu regions), Pinnacle, DraftKings, FanDuel,
   BetMGM, Betfair exchange, William Hill, Unibet, Caesars and more, including
   player props for NBA and NHL. Set the key once and the daily run fetches
   game lines and props for every upcoming game:

       set ODDS_API_KEY=your_key        (PowerShell: $env:ODDS_API_KEY="your_key")
       python -m nba_analyst run
       python -m nhl_analyst run

   Books and regions are configurable in `nba_analyst/odds/the_odds_api.py`.
   Props cost one request per game, so with the free tier keep `lookahead_days`
   small.
2. **Manual CSV** for books with no feed (Stake, Rollbit, a local book):
   drop `data/odds/nba_<date>.csv` or `nhl_<date>.csv` with the columns
   `game,book,market,side,line,player,price` (decimal or American prices).
   See `data/odds/nba_example.csv.txt`. They are merged on the next run.
3. **Polymarket**, as before, at the ask price (mid + half the spread).

Bet365, Stake and Rollbit do not publish APIs and block scraping, which is why
the aggregator and the CSV route exist. Paper picks record the book they were
priced at, so grading tracks each venue's closing price.

## Deploying the board as a website

    python -m nba_analyst export-site site

writes `site\index.html` plus `site\data.json` (NBA + NHL). Drag the `site`
folder onto Netlify's deploy page or point any static host at it. The page
reads `data.json` next to it and re-reads it every 3 minutes, so replacing
`data.json` is enough to update a deployed board. `--embed` writes the old
single-file version instead (data inside index.html, nothing else needed).

## Keeping it live (auto-refresh every 30 minutes)

Netlify only hosts files; the pipeline itself (ESPN, Polymarket, the odds
API, the model) has to run somewhere with internet. `.github/workflows/refresh.yml`
runs it on GitHub's free runners every 30 minutes and pushes the new
`site/` to Netlify, so the published link always shows fresh data without
anything running on your PC. Setup, once:

1. Put this folder in a GitHub repository (private is fine). The snapshot
   folders under `data\` are committed so the first run has a baseline;
   the SQLite databases are not (they are rebuilt and cached by the workflow).
2. Netlify: create the site once (drag the `site` folder), then copy
   **Site configuration → Site details → Site ID**, and create a personal
   access token under **User settings → Applications → Personal access tokens**.
3. GitHub: **Settings → Secrets and variables → Actions → New repository secret**
   for `NETLIFY_SITE_ID`, `NETLIFY_AUTH_TOKEN`, and optionally `ODDS_API_KEY`
   (without it the board uses Polymarket + ESPN only).
4. **Actions** tab → *refresh-board* → *Run workflow* for the first build
   (3–8 minutes). After that it runs itself every 30 minutes; the
   `updated …` stamp in the dashboard header shows the last build.

Notes: GitHub pauses scheduled workflows on repositories with no commits
for 60 days (one commit re-enables them); the schedule can drift a few minutes
at busy times; the free tier gives 2,000 Actions minutes per month and this
uses roughly 1,500 at every-30-minutes, so switch the cron to `*/45` or
hourly if you also run other workflows. To stop the board: disable the
workflow in the Actions tab.

Alternative for a faster board: run `python -m nba_analyst serve --with-nhl`
on any always-on machine (a $5 VPS) with a cron job calling `run` for both
sports; the page polls `/api/dashboard` the same way.
