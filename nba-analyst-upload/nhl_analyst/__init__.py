"""NHL AI Analyst. Same architecture as nba_analyst: ESPN + Polymarket ingest ->
SQLite -> team ratings -> Poisson goals model -> player projections -> scanner
-> dashboard. Depends on nba_analyst for shared helpers (HTTP, Polymarket
fetch, normal CDF) and the dashboard page."""
__version__ = "0.1.0"
