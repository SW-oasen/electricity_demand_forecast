"""
ETL pipeline: create and maintain a SQLite database for the electricity demand forecasting project.

Tables
------
energy_demand
    Kaggle historical (2019-01-01 – 2025-09-30) + SMARD actual (2025-10-01 onwards),
    with pre-computed time & energy lag features.
    smard_forecast_mwh: SMARD official forecast (filter_id=411) for the full range where
    available — NULL where the API has no data.

weather
    Open-Meteo historical weather for Germany (population-weighted, 5 cities),
    with pre-computed lag & rolling features.

View
----
energy_weather_combined
    JOIN of the two tables on time — ready for model training queries.

Entry point
-----------
update_database(db_path)
    Idempotent: creates the DB, seeds empty tables on first run,
    or fills any gaps up to yesterday on subsequent runs.
"""

import sqlite3
import sys
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd

# ---------------------------------------------------------------------------
# Path setup — works regardless of cwd
# ---------------------------------------------------------------------------
SRC_DIR  = Path(__file__).parent
ROOT_DIR = SRC_DIR.parent

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from prediction_store import ensure_prediction_table

from fetch_prepare_data import (
    fetch_smard_netzlast,
    prepare_weather_data,
    fetch_weather_data_for_cities,
    merge_weather_data_with_city_weights,
    create_time_based_features,
    create_energy_features,
    create_weather_features,
    rename_time_column,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
from config import (
    SMARD_FILTER_NETZLAST  as FILTER_NETZLAST_ACTUAL,
    SMARD_FILTER_FORECAST  as FILTER_NETZLAST_FORECAST,
    KAGGLE_END_DATE,
    SMARD_START_DATE,
)

KAGGLE_RAW_PATH  = ROOT_DIR / "data" / "raw" / "MHLV_2019_2025_combined.csv"


DB_DIR          = ROOT_DIR / "db"
DEFAULT_DB_PATH = DB_DIR / "energy_demand.db"
CALENDAR_FEATURE_VERSION = "2"

# Context rows needed to compute lag/rolling features at the seam during incremental updates
ENERGY_CONTEXT_ROWS  = 168   # lag_168h is the deepest lookback
WEATHER_CONTEXT_ROWS = 24    # lag_24h / rolling_24h is the deepest lookback

# ---------------------------------------------------------------------------
# DDL
# ---------------------------------------------------------------------------
_CREATE_ENERGY_TABLE = """
CREATE TABLE IF NOT EXISTS energy_demand (
    time                            TEXT    PRIMARY KEY,
    energy_demand_mwh               REAL,
    smard_forecast_mwh              REAL,
    data_source                     TEXT,
    year                            INTEGER,
    hour                            INTEGER,
    weekday                         INTEGER,
    month                           INTEGER,
    is_weekend                      INTEGER,
    is_holiday                      INTEGER,
    holiday_ratio                   REAL,
    is_school_holiday               INTEGER,
    school_holiday_ratio            REAL,
    is_workday                      INTEGER,
    is_bridge_day                   INTEGER,
    holiday_weight                  REAL,
    is_pandemic_time                INTEGER,
    energy_demand_lag_24h           REAL,
    energy_demand_lag_168h          REAL,
    energy_demand_rolling_mean_24h  REAL,
    energy_demand_rolling_mean_168h REAL
)
"""

_CREATE_WEATHER_TABLE = """
CREATE TABLE IF NOT EXISTS weather (
    time                                    TEXT    PRIMARY KEY,
    apparent_temperature                    REAL,
    rain                                    REAL,
    snowfall                                REAL,
    wind_speed_10m                          REAL,
    shortwave_radiation                     REAL,
    apparent_temperature_lag_24h            REAL,
    apparent_temperature_rolling_mean_24h   REAL,
    shortwave_radiation_0m_lag_24h          REAL,
    shortwave_radiation_0m_rolling_mean_24h REAL,
    heating_degree                          REAL,
    cooling_degree                          REAL
)
"""

_CREATE_ETL_METADATA_TABLE = """
CREATE TABLE IF NOT EXISTS etl_metadata (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
)
"""

_CREATE_COMBINED_VIEW = """
CREATE VIEW energy_weather_combined AS
SELECT
    e.time,
    e.energy_demand_mwh,
    e.smard_forecast_mwh,
    e.data_source,
    e.year, e.hour, e.weekday, e.month,
    e.is_weekend, e.is_holiday, e.holiday_ratio,
    e.is_school_holiday, e.school_holiday_ratio,
    e.is_workday, e.is_bridge_day, e.holiday_weight, e.is_pandemic_time,
    e.energy_demand_lag_24h, e.energy_demand_lag_168h,
    e.energy_demand_rolling_mean_24h, e.energy_demand_rolling_mean_168h,
    w.apparent_temperature, w.rain, w.snowfall, w.wind_speed_10m, w.shortwave_radiation,
    w.apparent_temperature_lag_24h, w.apparent_temperature_rolling_mean_24h,
    w.shortwave_radiation_0m_lag_24h, w.shortwave_radiation_0m_rolling_mean_24h,
    w.heating_degree, w.cooling_degree
FROM energy_demand e
JOIN weather w ON e.time = w.time
"""

_ENERGY_DB_COLS = [
    'time', 'energy_demand_mwh', 'smard_forecast_mwh', 'data_source',
    'year', 'hour', 'weekday', 'month',
    'is_weekend', 'is_holiday', 'holiday_ratio',
    'is_school_holiday', 'school_holiday_ratio',
    'is_workday', 'is_bridge_day', 'holiday_weight', 'is_pandemic_time',
    'energy_demand_lag_24h', 'energy_demand_lag_168h',
    'energy_demand_rolling_mean_24h', 'energy_demand_rolling_mean_168h',
]

_WEATHER_DB_COLS = [
    'time', 'apparent_temperature', 'rain', 'snowfall', 'wind_speed_10m', 'shortwave_radiation',
    'apparent_temperature_lag_24h', 'apparent_temperature_rolling_mean_24h',
    'shortwave_radiation_0m_lag_24h', 'shortwave_radiation_0m_rolling_mean_24h',
    'heating_degree', 'cooling_degree',
]


# ---------------------------------------------------------------------------
# Phase 1 — DB Setup
# ---------------------------------------------------------------------------

def create_database(db_path: Path = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """
    Create the SQLite database file, tables, and combined view.
    Safe to call multiple times (uses IF NOT EXISTS). Returns open connection.
    """
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(_CREATE_ENERGY_TABLE)
    cur.execute(_CREATE_WEATHER_TABLE)
    cur.execute(_CREATE_ETL_METADATA_TABLE)
    existing_energy_cols = {
        row[1] for row in cur.execute("PRAGMA table_info(energy_demand)")
    }
    for column, sql_type in (
        ('is_school_holiday', 'INTEGER'),
        ('school_holiday_ratio', 'REAL'),
    ):
        if column not in existing_energy_cols:
            cur.execute(
                f'ALTER TABLE energy_demand ADD COLUMN {column} {sql_type}'
            )
    ensure_prediction_table(conn)
    cur.execute("DROP VIEW IF EXISTS energy_weather_combined")
    try:
        cur.execute(_CREATE_COMBINED_VIEW)
    except sqlite3.OperationalError:
        pass  # view already exists — DROP/RECREATE not needed for idempotency
    conn.commit()
    print(f"Database ready: {db_path}")
    return conn


def backfill_calendar_features(
    conn: sqlite3.Connection,
    school_holiday_dates: dict | None = None,
    force: bool = False,
) -> int:
    """Backfill population-weighted holiday features once per feature version."""
    version_row = conn.execute(
        "SELECT value FROM etl_metadata WHERE key = 'calendar_feature_version'"
    ).fetchone()
    if not force and version_row and version_row[0] == CALENDAR_FEATURE_VERSION:
        return 0

    times = pd.read_sql("SELECT time FROM energy_demand ORDER BY time", conn)
    if times.empty:
        return 0

    db_times = times['time'].copy()
    times = _parse_time_col(times)
    parsed_years = times['time'].dt.year
    features = create_time_based_features(
        times,
        in_year=int(parsed_years.max()),
        in_school_holiday_dates=school_holiday_dates,
    )
    updates = list(
        features.assign(db_time=db_times)[[
            'holiday_ratio',
            'is_school_holiday',
            'school_holiday_ratio',
            'db_time',
        ]].itertuples(index=False, name=None)
    )
    with conn:
        conn.executemany(
            """
            UPDATE energy_demand
            SET holiday_ratio = ?,
                is_school_holiday = ?,
                school_holiday_ratio = ?
            WHERE time = ?
            """,
            updates,
        )
        conn.execute(
            """
            INSERT INTO etl_metadata (key, value)
            VALUES ('calendar_feature_version', ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """,
            (CALENDAR_FEATURE_VERSION,),
        )
    print(f"Backfilled calendar features: {len(updates)} rows")
    return len(updates)


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def _yesterday() -> str:
    """Yesterday's date as 'YYYY-MM-DD'."""
    return (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")


def _time_to_str(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert tz-aware 'time' column to ISO 8601 string for SQLite storage.
    Format: '2019-01-01T00:00:00+0100'  (sortable, unambiguous).
    """
    df = df.copy()
    df['time'] = df['time'].dt.strftime('%Y-%m-%dT%H:%M:%S%z')
    return df


def _parse_time_col(df: pd.DataFrame) -> pd.DataFrame:
    """Parse stored ISO 8601 time strings back to tz-aware Europe/Berlin timestamps."""
    df = df.copy()
    df['time'] = (pd.to_datetime(df['time'], utc=True)
                  .dt.tz_convert("Europe/Berlin")
                  .dt.as_unit('s'))
    return df


def _rename_energy_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Rename pandas energy feature columns to match the DB schema."""
    return df.rename(columns={
        'EnergyDemand':                   'energy_demand_mwh',
        'EnergyDemand_lag_24h':           'energy_demand_lag_24h',
        'EnergyDemand_lag_168h':          'energy_demand_lag_168h',
        'EnergyDemand_rolling_mean_24h':  'energy_demand_rolling_mean_24h',
        'EnergyDemand_rolling_mean_168h': 'energy_demand_rolling_mean_168h',
    })


def _select_db_cols(df: pd.DataFrame, col_list: list) -> pd.DataFrame:
    """Return only the columns present in both df and col_list, in col_list order."""
    return df[[c for c in col_list if c in df.columns]]


# ---------------------------------------------------------------------------
# Phase 2 — Seeding helpers (private)
# ---------------------------------------------------------------------------

def _load_raw_kaggle_energy() -> pd.DataFrame:
    """
    Load the Kaggle raw CSV, filter Germany, normalise column names and timezone.
    Returns raw DataFrame with columns ['time', 'EnergyDemand'] — no features yet.
    """
    df = pd.read_csv(KAGGLE_RAW_PATH)
    df = rename_time_column(df)
    df['time'] = (pd.to_datetime(df['time'])
                  .dt.tz_localize("UTC")
                  .dt.tz_convert("Europe/Berlin")
                  .dt.as_unit('s'))
    df = df[df['CountryCode'] == 'DE'].copy()
    df = df.rename(columns={'Value': 'EnergyDemand'})
    df = df[['time', 'EnergyDemand']].sort_values('time').reset_index(drop=True)
    # Clip to the Kaggle era boundary (defensive guard)
    cutoff = pd.Timestamp(KAGGLE_END_DATE, tz="Europe/Berlin") + pd.Timedelta(hours=23)
    return df[df['time'] <= cutoff].reset_index(drop=True)


def _fetch_smard_forecast(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch SMARD official hourly forecast (filter_id=411) for [start_date, end_date].
    Returns DataFrame with columns ['time', 'smard_forecast_mwh'].
    Returns empty DataFrame if SMARD has no data for the requested range.
    """
    df = fetch_smard_netzlast(start_date, end_date, filter_id=FILTER_NETZLAST_FORECAST)
    return df.rename(columns={'EnergyDemand': 'smard_forecast_mwh'})


def _build_full_energy_df(yesterday_str: str) -> tuple:
    """
    Build the complete energy DataFrame with features, ready for initial seeding.

    Steps
    -----
    1. Load Kaggle raw CSV  (2019-01-01 – 2025-09-30)
    2. Fetch SMARD actual   (2025-10-01 – yesterday)  filter_id=410
    3. Concatenate the two series; feature-engineer in one pass
       so lag/rolling features are correct across the 2025-09-30 → 2025-10-01 boundary.
    4. Fetch SMARD forecast (2019-01-01 – yesterday)  filter_id=411
       for the full historical range (used for Actual vs. Forecast comparison).

    Returns
    -------
    (energy_df_with_features, smard_forecast_df)
    """
    print("Loading Kaggle energy data...")
    df_kaggle = _load_raw_kaggle_energy()
    print(f"  {len(df_kaggle)} rows ({df_kaggle['time'].min().date()} – {df_kaggle['time'].max().date()})")

    print(f"Fetching SMARD actual data ({SMARD_START_DATE} → {yesterday_str})...")
    df_smard = fetch_smard_netzlast(SMARD_START_DATE, yesterday_str, filter_id=FILTER_NETZLAST_ACTUAL)
    df_smard = df_smard[['time', 'EnergyDemand']]
    print(f"  {len(df_smard)} rows fetched")

    # Combine; SMARD wins for any overlap at the boundary
    smard_start_ts = pd.Timestamp(SMARD_START_DATE, tz="Europe/Berlin")
    df_combined = (pd.concat([df_kaggle, df_smard], ignore_index=True)
                   .sort_values('time')
                   .drop_duplicates(subset=['time'], keep='last')   # SMARD wins
                   .reset_index(drop=True))
    df_combined['data_source'] = df_combined['time'].apply(
        lambda t: 'smard' if t >= smard_start_ts else 'kaggle'
    )

    # Single-pass feature engineering — handles boundary lags correctly
    print("Applying time-based and energy feature engineering...")
    df_combined = create_time_based_features(df_combined, in_year=pd.to_datetime(yesterday_str).year)
    df_combined = create_energy_features(df_combined)   # dropna() removes the leading NaN rows
    print(f"  {len(df_combined)} rows with features ({df_combined['time'].min().date()} – {df_combined['time'].max().date()})")

    # SMARD forecast — full historical range for comparison charts
    print(f"Fetching SMARD forecast (filter_id=411, 2019-01-01 → {yesterday_str})...")
    try:
        df_forecast = _fetch_smard_forecast("2019-01-01", yesterday_str)
        print(f"  {len(df_forecast)} forecast rows")
    except Exception as exc:
        print(f"  Full range unavailable ({exc}), retrying from {SMARD_START_DATE}...")
        try:
            df_forecast = _fetch_smard_forecast(SMARD_START_DATE, yesterday_str)
            print(f"  {len(df_forecast)} forecast rows (from {SMARD_START_DATE})")
        except Exception as exc2:
            print(f"  SMARD forecast fetch failed: {exc2}. smard_forecast_mwh will be NULL.")
            df_forecast = pd.DataFrame(columns=['time', 'smard_forecast_mwh'])

    return df_combined, df_forecast


# ---------------------------------------------------------------------------
# Phase 2 — Seeding (public)
# ---------------------------------------------------------------------------

def seed_energy_table(conn: sqlite3.Connection) -> None:
    """
    Seed the energy_demand table from scratch (Kaggle CSV + SMARD API).
    Assumes the table is currently empty.
    """
    yesterday_str = _yesterday()
    df_energy, df_forecast = _build_full_energy_df(yesterday_str)
    df_energy = _rename_energy_cols(df_energy)

    # Convert time to string before merge so both sides use the same key format
    df_energy = _time_to_str(df_energy)

    if not df_forecast.empty:
        df_forecast = _time_to_str(df_forecast)
        df_energy = df_energy.merge(df_forecast, on='time', how='left')
    else:
        df_energy['smard_forecast_mwh'] = None

    df_energy = _select_db_cols(df_energy, _ENERGY_DB_COLS)
    df_energy.to_sql('energy_demand', conn, if_exists='append', index=False)
    conn.commit()
    print(f"Seeded energy_demand: {len(df_energy)} rows")


def seed_weather_table(conn: sqlite3.Connection) -> None:
    """
    Seed the weather table from scratch (Open-Meteo archive API, full range).
    Assumes the table is currently empty. Fetches 5 cities for ~7 years — takes a few minutes.
    """
    yesterday_str = _yesterday()
    print(f"Fetching weather data (2019-01-01 → {yesterday_str}) — this may take a few minutes...")
    df_weather = prepare_weather_data('2019-01-01', yesterday_str)
    df_weather = df_weather.dropna()   # remove leading NaN rows from lag/rolling
    df_weather = _time_to_str(df_weather)
    df_weather = _select_db_cols(df_weather, _WEATHER_DB_COLS)
    df_weather.to_sql('weather', conn, if_exists='append', index=False)
    conn.commit()
    print(f"Seeded weather: {len(df_weather)} rows")


# ---------------------------------------------------------------------------
# Phase 3 — Status check
# ---------------------------------------------------------------------------

def check_data_status(conn: sqlite3.Connection) -> dict:
    """
    Query each table for its max timestamp and row count.

    Returns
    -------
    dict with keys 'energy' and 'weather', each containing:
        max_time  : str | None  — most recent stored timestamp
        rows      : int         — total row count
        is_current: bool        — True if max_time >= yesterday
    """
    yesterday_date = (datetime.now() - timedelta(days=1)).date()
    cur = conn.cursor()

    cur.execute("SELECT MAX(time), COUNT(*) FROM energy_demand")
    energy_max, energy_rows = cur.fetchone()

    cur.execute("SELECT MAX(time), COUNT(*) FROM weather")
    weather_max, weather_rows = cur.fetchone()

    def _is_current(max_time_str: str | None) -> bool:
        if not max_time_str:
            return False
        return pd.to_datetime(max_time_str).date() >= yesterday_date

    return {
        'energy': {
            'max_time':   energy_max,
            'rows':       energy_rows,
            'is_current': _is_current(energy_max),
        },
        'weather': {
            'max_time':   weather_max,
            'rows':       weather_rows,
            'is_current': _is_current(weather_max),
        },
    }


# ---------------------------------------------------------------------------
# Phase 3 — Incremental updates
# ---------------------------------------------------------------------------

def update_energy_table(conn: sqlite3.Connection) -> int:
    """
    Fetch missing energy data from SMARD and insert into energy_demand.

    Queries the last ENERGY_CONTEXT_ROWS (168) rows from the DB as a lag context
    so that energy_demand_lag_168h and rolling features are computed correctly
    at the seam between existing and new data.

    Returns the number of newly inserted rows.
    """
    status = check_data_status(conn)['energy']
    if status['rows'] == 0:
        raise RuntimeError("energy_demand table is empty — run seed_energy_table() first.")
    if status['is_current']:
        print("Energy table is already up to date.")
        return 0

    max_time     = pd.to_datetime(status['max_time']).tz_convert("Europe/Berlin").as_unit('s')
    update_start = (max_time + pd.Timedelta(hours=1)).strftime("%Y-%m-%d")
    yesterday_str = _yesterday()
    print(f"Updating energy data ({update_start} → {yesterday_str})...")

    df_new_actual = fetch_smard_netzlast(update_start, yesterday_str, filter_id=FILTER_NETZLAST_ACTUAL)
    if df_new_actual.empty:
        print("No new energy data available from SMARD.")
        return 0
    df_new_actual = df_new_actual[['time', 'EnergyDemand']]

    df_new_forecast = _fetch_smard_forecast(update_start, yesterday_str)

    # Load context rows for correct lag computation across the seam
    df_context = pd.read_sql(
        f"SELECT time, energy_demand_mwh FROM energy_demand ORDER BY time DESC LIMIT {ENERGY_CONTEXT_ROWS}",
        conn
    )
    df_context = _parse_time_col(df_context.rename(columns={'energy_demand_mwh': 'EnergyDemand'}))
    df_context = df_context.sort_values('time').reset_index(drop=True)

    # Single-pass feature engineering; dropna() in create_energy_features removes context rows
    df_combined = (pd.concat([df_context, df_new_actual], ignore_index=True)
                   .sort_values('time')
                   .drop_duplicates(subset=['time'])
                   .reset_index(drop=True))
    df_combined = create_time_based_features(df_combined, in_year=pd.to_datetime(yesterday_str).year)
    df_combined = create_energy_features(df_combined)

    df_new = df_combined[df_combined['time'] > max_time].copy()
    df_new['data_source'] = 'smard'
    df_new = _rename_energy_cols(df_new)
    df_new = _time_to_str(df_new)

    if not df_new_forecast.empty:
        df_new_forecast = _time_to_str(df_new_forecast)
        df_new = df_new.merge(df_new_forecast, on='time', how='left')
    else:
        df_new['smard_forecast_mwh'] = None

    df_new = _select_db_cols(df_new, _ENERGY_DB_COLS)
    df_new.to_sql('energy_demand', conn, if_exists='append', index=False)
    conn.commit()
    print(f"Inserted {len(df_new)} new rows into energy_demand.")
    return len(df_new)


def update_weather_table(conn: sqlite3.Connection) -> int:
    """
    Fetch missing weather data from Open-Meteo and insert into weather.

    Queries the last WEATHER_CONTEXT_ROWS (24) rows from the DB as a rolling context
    so that lag_24h and rolling_mean_24h are computed correctly at the seam.

    Returns the number of newly inserted rows.
    """
    status = check_data_status(conn)['weather']
    if status['rows'] == 0:
        raise RuntimeError("weather table is empty — run seed_weather_table() first.")
    if status['is_current']:
        print("Weather table is already up to date.")
        return 0

    max_time     = pd.to_datetime(status['max_time']).tz_convert("Europe/Berlin").as_unit('s')
    update_start = (max_time + pd.Timedelta(hours=1)).strftime("%Y-%m-%d")
    yesterday_str = _yesterday()
    print(f"Updating weather data ({update_start} → {yesterday_str})...")

    # Fetch raw new weather (city-by-city, then population-weight merge)
    weather_city_dict = fetch_weather_data_for_cities(update_start, yesterday_str)
    df_new_raw = rename_time_column(
        merge_weather_data_with_city_weights(weather_city_dict)
    ).sort_values('time').reset_index(drop=True)

    # Load context rows (raw weather variables only — no derived columns needed)
    raw_cols = 'time, apparent_temperature, rain, snowfall, wind_speed_10m, shortwave_radiation'
    df_context = _parse_time_col(
        pd.read_sql(
            f"SELECT {raw_cols} FROM weather ORDER BY time DESC LIMIT {WEATHER_CONTEXT_ROWS}",
            conn
        )
    ).sort_values('time').reset_index(drop=True)

    df_combined = (pd.concat([df_context, df_new_raw], ignore_index=True)
                   .sort_values('time')
                   .drop_duplicates(subset=['time'])
                   .reset_index(drop=True))
    df_combined = create_weather_features(df_combined)

    df_new = df_combined[df_combined['time'] > max_time].dropna().copy()
    df_new = _time_to_str(df_new)
    df_new = _select_db_cols(df_new, _WEATHER_DB_COLS)
    df_new.to_sql('weather', conn, if_exists='append', index=False)
    conn.commit()
    print(f"Inserted {len(df_new)} new rows into weather.")
    return len(df_new)


# ---------------------------------------------------------------------------
# Phase 3 — Orchestrator (main entry point)
# ---------------------------------------------------------------------------

def update_database(db_path: Path = DEFAULT_DB_PATH) -> None:
    """
    Main entry point for the ETL pipeline.

    - Creates the database and tables if they do not exist.
    - Seeds each table on first run.
    - Fills any gap since the last run on subsequent calls.
    - Safe to run repeatedly (idempotent).
    """
    conn = create_database(db_path)
    try:
        status = check_data_status(conn)
        print("\nCurrent data status:")
        for tbl, s in status.items():
            print(f"  {tbl:15s}: {s['rows']:>6} rows | max: {s['max_time']} | up-to-date: {s['is_current']}")

        # --- Energy ---
        if status['energy']['rows'] == 0:
            print("\n[Energy] First run — seeding table...")
            seed_energy_table(conn)
        elif not status['energy']['is_current']:
            print("\n[Energy] Filling gap...")
            update_energy_table(conn)
        else:
            print("\n[Energy] Up to date — nothing to do.")

        backfill_calendar_features(conn)

        # --- Weather ---
        if status['weather']['rows'] == 0:
            print("\n[Weather] First run — seeding table...")
            seed_weather_table(conn)
        elif not status['weather']['is_current']:
            print("\n[Weather] Filling gap...")
            update_weather_table(conn)
        else:
            print("\n[Weather] Up to date — nothing to do.")

        final = check_data_status(conn)
        print(f"\nDone.")
        print(f"  energy_demand : {final['energy']['rows']} rows | max: {final['energy']['max_time']}")
        print(f"  weather       : {final['weather']['rows']} rows | max: {final['weather']['max_time']}")

    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Read helpers (for use by other modules / notebooks)
# ---------------------------------------------------------------------------

def get_connection(db_path: Path = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """Return an open connection to the database."""
    return sqlite3.connect(db_path)


def load_energy_data(conn: sqlite3.Connection,
                     start_date: str = None,
                     end_date: str = None) -> pd.DataFrame:
    """
    Load energy_demand rows with optional date filter (inclusive, 'YYYY-MM-DD').
    Returns DataFrame with tz-aware 'time' column (Europe/Berlin).
    """
    query, params = _build_query("SELECT * FROM energy_demand", start_date, end_date)
    return _parse_time_col(pd.read_sql(query, conn, params=params or None))


def load_weather_data(conn: sqlite3.Connection,
                      start_date: str = None,
                      end_date: str = None) -> pd.DataFrame:
    """
    Load weather rows with optional date filter (inclusive, 'YYYY-MM-DD').
    Returns DataFrame with tz-aware 'time' column (Europe/Berlin).
    """
    query, params = _build_query("SELECT * FROM weather", start_date, end_date)
    return _parse_time_col(pd.read_sql(query, conn, params=params or None))


def load_combined_data(conn: sqlite3.Connection,
                       start_date: str = None,
                       end_date: str = None) -> pd.DataFrame:
    """
    Load from the energy_weather_combined VIEW with optional date filter.
    Returns DataFrame with tz-aware 'time' column (Europe/Berlin).
    """
    query, params = _build_query("SELECT * FROM energy_weather_combined", start_date, end_date)
    return _parse_time_col(pd.read_sql(query, conn, params=params or None))


def _build_query(base: str, start_date: str, end_date: str) -> tuple:
    """Build a parameterised SELECT query with optional date range filters."""
    conditions, params = [], []
    if start_date:
        conditions.append("time >= ?")
        params.append(start_date)
    if end_date:
        conditions.append("time <= ?")
        params.append(end_date + "T23:59:59")
    if conditions:
        base += " WHERE " + " AND ".join(conditions)
    return base + " ORDER BY time", params


def prepare_for_prediction_tomorrow_etl(
    prediction_date: str,
    model,
    db_path: Path = DEFAULT_DB_PATH,
) -> pd.DataFrame:
    """Build tomorrow's features after recursively forecasting the current day.

    Actual demand is loaded only from before D-1. The selected model predicts
    D-1 hour by hour; these intermediate predictions are then used to construct
    the energy features for target day D.
    """
    from fetch_prepare_data import prepare_weather_for_prediction
    from walk_forward import build_recursive_forecast_features

    target_start = pd.Timestamp(prediction_date, tz='Europe/Berlin')
    target_end = target_start + pd.DateOffset(days=1)
    forecast_origin = target_start - pd.DateOffset(days=1)

    conn = sqlite3.connect(db_path)
    try:
        df_ctx = pd.read_sql(
            """
            SELECT time, energy_demand_mwh FROM energy_demand
            WHERE time < ? ORDER BY time DESC LIMIT 168
            """,
            conn,
            params=[forecast_origin.strftime('%Y-%m-%dT%H:%M:%S%z')],
        )
    finally:
        conn.close()

    history = (
        _parse_time_col(df_ctx).sort_values('time')
        .set_index('time')['energy_demand_mwh']
    )
    if len(history.dropna()) < 168:
        raise ValueError('Insufficient demand history for recursive prediction')

    horizon_time = pd.date_range(
        forecast_origin, target_end, freq='h', inclusive='left'
    )
    df_time = create_time_based_features(
        pd.DataFrame({'time': horizon_time}), in_year=target_start.year
    )

    df_weather = prepare_weather_for_prediction(prediction_date)
    df_weather = df_weather[
        (df_weather['time'] >= forecast_origin) & (df_weather['time'] < target_end)
    ].copy()
    source = pd.merge(df_time, df_weather, on='time', how='inner')
    if len(source) != len(horizon_time):
        raise ValueError(
            f'Incomplete weather horizon: expected {len(horizon_time)} rows, '
            f'got {len(source)}'
        )

    return build_recursive_forecast_features(
        model, source, history, target_start
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    update_database()
