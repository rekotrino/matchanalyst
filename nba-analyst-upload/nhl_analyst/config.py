from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    league_gpg: float = 3.05            # goals per team per game (regulation + OT), refreshed from data
    home_advantage: float = 0.08        # multiplicative on home goals (~ +0.1 goals)
    ot_prob_better_team: float = 0.53   # share of tied-after-regulation games the stronger team wins
    shrink_games: float = 10.0          # pull toward last-season prior / league average
    half_life: float = 12.0             # recency weight half-life in games
    anchor_cap: float = 0.6             # max model weight while uncalibrated (rest is the reference line)
    anchor_games: float = 15.0


@dataclass
class BettingConfig:
    bankroll: float = 500.0
    edge_threshold: float = 0.045
    prob_floor: float = 0.55
    min_liquidity: float = 500.0
    kelly_fraction: float = 0.25
    max_stake_pct: float = 0.03
    max_picks_per_game: int = 2
    max_picks_per_day: int = 6
    prop_min_games: int = 5


@dataclass
class Settings:
    model: ModelConfig = field(default_factory=ModelConfig)
    betting: BettingConfig = field(default_factory=BettingConfig)
    db_path: str = "data/nhl_analyst.sqlite"
    reports_dir: str = "reports"
    lookback_days: int = 45
    lookahead_days: int = 2
    # standard prop lines priced even when no market is listed
    standard_lines: dict = field(default_factory=lambda: {
        "g": [0.5, 1.5], "a": [0.5, 1.5], "p": [0.5, 1.5, 2.5], "sog": [1.5, 2.5, 3.5, 4.5, 5.5],
        "blk": [1.5, 2.5, 3.5], "hits": [1.5, 2.5, 3.5], "sv": None,  # saves: ladder built around the projection
    })


SETTINGS = Settings()
