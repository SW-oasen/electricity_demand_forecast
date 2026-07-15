"""SQLite persistence for walk-forward prediction results."""

from __future__ import annotations

import sqlite3

import pandas as pd


CREATE_PREDICTIONS_TABLE = """
CREATE TABLE IF NOT EXISTS walk_forward_predictions (
    target_time      TEXT NOT NULL,
    forecast_origin  TEXT NOT NULL,
    model_name       TEXT NOT NULL,
    evaluation_mode  TEXT NOT NULL,
    prediction_mwh   REAL NOT NULL,
    actual_mwh       REAL,
    created_at       TEXT NOT NULL,
    PRIMARY KEY (target_time, model_name, evaluation_mode)
)
"""

PREDICTION_COLUMNS = [
    "target_time",
    "forecast_origin",
    "model_name",
    "evaluation_mode",
    "prediction_mwh",
    "actual_mwh",
    "created_at",
]


def ensure_prediction_table(connection: sqlite3.Connection) -> None:
    """Create the prediction table if it does not exist."""
    connection.execute(CREATE_PREDICTIONS_TABLE)
    connection.commit()


def upsert_predictions(
    connection: sqlite3.Connection,
    predictions: pd.DataFrame,
) -> int:
    """Insert or update prediction rows without deleting other results."""
    missing = [column for column in PREDICTION_COLUMNS if column not in predictions]
    if missing:
        raise ValueError(f"Missing walk-forward columns: {missing}")
    if predictions.empty:
        return 0

    rows = predictions.loc[:, PREDICTION_COLUMNS].copy()
    for column in ("target_time", "forecast_origin"):
        parsed = pd.to_datetime(rows[column], utc=True, errors="raise")
        rows[column] = (
            parsed.dt.tz_convert("Europe/Berlin")
            .dt.strftime("%Y-%m-%dT%H:%M:%S%z")
        )
    required = [
        "target_time",
        "forecast_origin",
        "model_name",
        "evaluation_mode",
        "prediction_mwh",
        "created_at",
    ]
    if rows[required].isna().any().any():
        raise ValueError("Required walk-forward values must not be null")

    sql = """
        INSERT INTO walk_forward_predictions (
            target_time, forecast_origin, model_name, evaluation_mode,
            prediction_mwh, actual_mwh, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(target_time, model_name, evaluation_mode) DO UPDATE SET
            forecast_origin = excluded.forecast_origin,
            prediction_mwh = excluded.prediction_mwh,
            actual_mwh = excluded.actual_mwh,
            created_at = excluded.created_at
    """
    connection.executemany(sql, rows.itertuples(index=False, name=None))
    connection.commit()
    return len(rows)


def load_predictions(
    connection: sqlite3.Connection,
    model_name: str,
    evaluation_mode: str,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    """Load predictions for one model, evaluation mode and optional date range."""
    query = """
        SELECT target_time, forecast_origin, model_name, evaluation_mode,
               prediction_mwh, actual_mwh, created_at
        FROM walk_forward_predictions
        WHERE model_name = ? AND evaluation_mode = ?
    """
    params: list[str] = [model_name, evaluation_mode]
    if start_date:
        query += " AND target_time >= ?"
        params.append(start_date)
    if end_date:
        query += " AND target_time <= ?"
        params.append(end_date + "T23:59:59+9999")
    query += " ORDER BY target_time"
    result = pd.read_sql(query, connection, params=params)
    if not result.empty:
        for column in ("target_time", "forecast_origin"):
            result[column] = (
                pd.to_datetime(result[column], utc=True)
                .dt.tz_convert("Europe/Berlin")
                .dt.as_unit("s")
            )
    return result
