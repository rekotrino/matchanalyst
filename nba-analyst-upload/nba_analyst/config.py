"""All tunable knobs live here. Phase 1 backtesting will calibrate the model
parameters; the betting parameters are deliberately conservative."""

from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    # League baselines (2025-26 approximate; refreshed from data when available)
    league_ortg: float = 114.5          # points per 100 possessions
    league_pace: float = 99.0           # possessions per 48 minutes
    home_advantage_pts: float = 2.3     # home margin edge in points
    margin_sd: float = 13.0             # std dev of actual - expected margin
    total_sd: float = 19.0              # std dev of actual - expected total
    team_pts_sd: float = 11.5           # std dev for a single team's points
    # How much we trust recent form vs. the season-long baseline.
    # Effective "prior games": a team with N games gets weight N/(N+shrink).
    shrink_games: float = 12.0
    # Rating windows
    recent_window: int = 5
    form_window: int = 10
    # Market anchoring while uncalibrated: final line = w*model + (1-w)*reference.
    # w = min(anchor_cap, games/(games+anchor_games)) using current-season
    # regular-season games played by the less-played team. Phase 1 raises the cap
    # only if backtests show the pure model beats the reference.
    anchor_cap: float = 0.6
    anchor_games: float = 12.0
    # Rest effects (points of margin)
    back_to_back_penalty: float = 1.5
    three_in_four_penalty: float = 0.8
    rest_advantage_cap: float = 2.0


@dataclass
class BettingConfig:
    bankroll: float = 500.0
    edge_threshold: float = 0.045       # min (our prob - price) to consider
    prob_floor: float = 0.55            # we prefer favourites; below this we skip
    min_liquidity: float = 500.0        # USD on Polymarket market
    kelly_fraction: float = 0.25
    max_stake_pct: float = 0.03
    max_picks_per_game: int = 3         # one side market + one total + props (one per player)
    prop_min_games: int = 5             # player needs this many games with the team to price a prop
    max_picks_per_day: int = 6


@dataclass
class Settings:
    model: ModelConfig = field(default_factory=ModelConfig)
    betting: BettingConfig = field(default_factory=BettingConfig)
    db_path: str = "data/nba_analyst.sqlite"
    reports_dir: str = "reports"
    lookback_days: int = 45             # how far back to pull completed games
    lookahead_days: int = 3             # how far ahead to pull fixtures
    request_timeout: int = 25
    user_agent: str = "Mozilla/5.0 (nba-analyst/0.1)"


SETTINGS = Settings()
