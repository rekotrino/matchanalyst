"""NBA AI Analyst - Phase 0.

Pipeline: ingest (ESPN + Polymarket) -> analyst ratings -> probability model
-> market scanner -> pre-match reports. Everything is timestamped in SQLite so
later phases can backtest honestly and grade picks against closing prices.
"""

__version__ = "0.1.0"
