"""
config.py — project-specific configuration for the Germany electricity demand project.

All domain constants live here so that:
- util/ classes remain fully generic (no hardcoded country/city defaults)
- etl.py and fetch_prepare_data.py import from one place instead of each defining their own
"""

import pandas as pd

# ---------------------------------------------------------------------------
# SMARD API
# ---------------------------------------------------------------------------
SMARD_BASE              = "https://www.smard.de/app/chart_data"
SMARD_HEADERS           = {"User-Agent": "Mozilla/5.0 (compatible; smard-fetcher/1.0)"}
SMARD_REGION            = "DE"
SMARD_RESOLUTION        = "hour"
SMARD_FILTER_NETZLAST   = 410   # Realisierter Stromverbrauch – Netzlast (actual)
SMARD_FILTER_FORECAST   = 411   # Prognostizierter Stromverbrauch – Netzlast (SMARD official forecast)

# ---------------------------------------------------------------------------
# Dataset date boundaries
# ---------------------------------------------------------------------------
KAGGLE_END_DATE  = "2025-09-30"   # last date fully covered by the Kaggle dataset
SMARD_START_DATE = "2025-10-01"   # first date fetched from SMARD actual

# ---------------------------------------------------------------------------
# Calendar / holiday features
# ---------------------------------------------------------------------------
DE_STATE_CODES = [
    'BB', 'BE', 'BW', 'BY', 'HB', 'HE',
    'HH', 'MV', 'NI', 'NW', 'RP', 'SH',
    'SL', 'SN', 'ST', 'TH',
]

# Population weights by federal state (Destatis population level, rounded).
# Absolute values are used only as relative weights and should be refreshed
# together when a newer reference year is adopted.
DE_STATE_POPULATION = {
    'BW': 11_339_000,
    'BY': 13_435_000,
    'BE':  3_878_000,
    'BB':  2_582_000,
    'HB':    692_000,
    'HE':  6_391_000,
    'HH':  1_910_000,
    'MV':  1_628_000,
    'NI':  8_162_000,
    'NW': 18_190_000,
    'RP':  4_174_000,
    'SH':  2_966_000,
    'SL':    994_000,
    'SN':  4_086_000,
    'ST':  2_145_000,
    'TH':  2_120_000,
}

PANDEMIC_START = pd.Timestamp('2020-03-01', tz='Europe/Berlin')
PANDEMIC_END   = pd.Timestamp('2021-12-31', tz='Europe/Berlin')

# ---------------------------------------------------------------------------
# Weather — Open-Meteo
# ---------------------------------------------------------------------------
WEATHER_VARIABLES = [
    'apparent_temperature',
    'rain',
    'snowfall',
    'wind_speed_10m',
    'shortwave_radiation',
    # 'temperature_2m' dropped: high correlation with apparent_temperature (see notebook 02 EDA)
]

# Population-weighted average over 5 major German cities
SELECTED_CITIES = {
    'Berlin':    {'latitude': 52.5200, 'longitude': 13.4050},
    'Hamburg':   {'latitude': 53.5511, 'longitude':  9.9937},
    'München':   {'latitude': 48.1351, 'longitude': 11.5820},
    'Köln':      {'latitude': 50.9375, 'longitude':  6.9603},
    'Frankfurt': {'latitude': 50.1109, 'longitude':  8.6821},
}

CITY_POPULATION = {
    'Berlin':    3_644_826,
    'Hamburg':   1_841_179,
    'München':   1_471_508,
    'Köln':      1_085_664,
    'Frankfurt':   753_056,
}

# ---------------------------------------------------------------------------
# Persisted ETL models
# ---------------------------------------------------------------------------
MODEL_FILENAMES = {
    'LGBM': 'best_lgbm_model_bayesian_etl.pkl',
    'XGBoost': 'best_xgb_model_bayesian_etl.pkl',
}

# ---------------------------------------------------------------------------
# Weather feature engineering thresholds
# ---------------------------------------------------------------------------
BASE_TEMPERATURE_HEATING = 18   # °C — heating degree days threshold
BASE_TEMPERATURE_COOLING = 25   # °C — cooling degree days threshold
