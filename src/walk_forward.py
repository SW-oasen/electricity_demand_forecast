"""Leakage-safe walk-forward evaluation for the ETL demand models.

The evaluator simulates a day-ahead forecast made at the start of the day before
the target day.  At that point actual demand is known only up to the end of D-2.
It therefore predicts D-1 first and feeds those predictions into the energy
history before predicting target day D.

Historical weather features currently come from the ETL database and are thus
observed weather, not archived weather forecasts.  This is made explicit by the
evaluation mode stored alongside every prediction.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd


EVALUATION_MODE = "walk_forward_48h_actual_weather"
RECURSIVE_PREDICTION_COLUMN = "_recursive_prediction_mwh"
NON_FEATURE_COLUMNS = {
    "time",
    "energy_demand_mwh",
    "smard_forecast_mwh",
    "data_source",
}
CSV_COLUMNS = (
    "target_time",
    "forecast_origin",
    "model_name",
    "evaluation_mode",
    "prediction_mwh",
    "actual_mwh",
    "created_at",
)


class PredictionModel(Protocol):
    def predict(self, features: pd.DataFrame) -> np.ndarray: ...


def build_energy_features(history: pd.Series) -> dict[str, float]:
    """Build the next row's energy features from strictly earlier load values.

    Position-based windows deliberately match ``shift``/``rolling`` used during
    model training.  They also handle Europe/Berlin DST days with 23/25 rows.
    ``history`` may contain both actual observations and earlier predictions.
    """
    values = pd.to_numeric(history, errors="coerce").dropna()
    if len(values) < 168:
        raise ValueError(
            f"At least 168 preceding load values are required, got {len(values)}."
        )
    return {
        "energy_demand_lag_24h": float(values.iloc[-24]),
        "energy_demand_lag_168h": float(values.iloc[-168]),
        "energy_demand_rolling_mean_24h": float(values.iloc[-24:].mean()),
        "energy_demand_rolling_mean_168h": float(values.iloc[-168:].mean()),
    }


def build_prediction_features(
    source_row: pd.Series,
    history: pd.Series,
    model: PredictionModel,
) -> pd.DataFrame:
    row = source_row.drop(labels=list(NON_FEATURE_COLUMNS), errors="ignore").copy()
    for name, value in build_energy_features(history).items():
        row[name] = value
    features = row.to_frame().T
    if hasattr(model, "feature_names_in_"):
        features = features.reindex(columns=list(model.feature_names_in_))
    features = features.apply(pd.to_numeric, errors="coerce")
    if features.isna().any().any():
        missing = features.columns[features.isna().any()].tolist()
        raise ValueError(f"Missing prediction features: {missing}")
    return features


def build_recursive_forecast_features(
    model: PredictionModel,
    source_data: pd.DataFrame,
    history: pd.Series,
    target_start: pd.Timestamp,
) -> pd.DataFrame:
    """Recursively predict the horizon and return features from target_start."""
    recursive_history = history.dropna().sort_index().copy()
    output: list[pd.DataFrame] = []
    for _, source_row in source_data.sort_values("time").iterrows():
        timestamp = source_row["time"]
        features = build_prediction_features(source_row, recursive_history, model)
        prediction = float(np.asarray(model.predict(features)).reshape(-1)[0])
        recursive_history.loc[timestamp] = prediction
        if timestamp >= target_start:
            result_row = features.copy()
            result_row.insert(0, "time", timestamp)
            result_row[RECURSIVE_PREDICTION_COLUMN] = prediction
            output.append(result_row)
    if not output:
        names = list(getattr(model, "feature_names_in_", []))
        return pd.DataFrame(
            columns=["time", *names, RECURSIVE_PREDICTION_COLUMN]
        )
    return pd.concat(output, ignore_index=True)


def predict_target_day(
    model: PredictionModel,
    combined_data: pd.DataFrame,
    target_date: str,
    model_name: str,
) -> pd.DataFrame:
    """Predict one target day after recursively predicting the preceding day.

    ``combined_data`` must include at least 168 actual hourly rows before the
    forecast origin plus feature rows for D-1 and D.  Actual demand at and after
    the origin is never read into the prediction history.
    """
    if combined_data.empty:
        raise ValueError("combined_data is empty")
    data = combined_data.sort_values("time").drop_duplicates("time").copy()
    if not isinstance(data["time"].dtype, pd.DatetimeTZDtype):
        raise ValueError("combined_data['time'] must be timezone-aware")

    tz = data["time"].dt.tz
    target_start = pd.Timestamp(target_date, tz=tz)
    target_end = target_start + pd.DateOffset(days=1)
    forecast_origin = target_start - pd.DateOffset(days=1)

    known = data.loc[data["time"] < forecast_origin, ["time", "energy_demand_mwh"]]
    history = known.set_index("time")["energy_demand_mwh"].dropna().sort_index()
    if len(history) < 168:
        raise ValueError("Insufficient actual history before forecast origin")

    horizon = data.loc[
        (data["time"] >= forecast_origin) & (data["time"] < target_end)
    ].copy()
    if horizon.empty or horizon["time"].min() != forecast_origin:
        raise ValueError("Forecast horizon does not start at the expected origin")

    predictions: list[tuple[pd.Timestamp, float]] = []
    for _, source_row in horizon.iterrows():
        timestamp = source_row["time"]
        features = build_prediction_features(source_row, history, model)
        prediction = float(np.asarray(model.predict(features)).reshape(-1)[0])
        history.loc[timestamp] = prediction
        if timestamp >= target_start:
            predictions.append((timestamp, prediction))

    actual = data.set_index("time")["energy_demand_mwh"]
    created_at = datetime.now(timezone.utc).isoformat()
    return pd.DataFrame(
        {
            "target_time": [timestamp for timestamp, _ in predictions],
            "forecast_origin": forecast_origin,
            "model_name": model_name,
            "evaluation_mode": EVALUATION_MODE,
            "prediction_mwh": [value for _, value in predictions],
            "actual_mwh": [actual.get(timestamp, np.nan) for timestamp, _ in predictions],
            "created_at": created_at,
        },
        columns=CSV_COLUMNS,
    )


def save_predictions_csv(predictions: pd.DataFrame, csv_path: Path) -> pd.DataFrame:
    """Atomically merge predictions into a resumable CSV cache."""
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    incoming = predictions.loc[:, CSV_COLUMNS].copy()
    if csv_path.exists():
        existing = pd.read_csv(csv_path)
        combined = pd.concat([existing, incoming], ignore_index=True)
    else:
        combined = incoming
    # A freshly generated tz-aware Timestamp and the same value read back from
    # CSV have different Python representations.  Canonical UTC strings keep
    # the resume/deduplication key stable across process restarts and DST.
    for column in ("target_time", "forecast_origin"):
        parsed = pd.to_datetime(combined[column], utc=True, errors="raise")
        combined[column] = parsed.dt.strftime("%Y-%m-%dT%H:%M:%S%z")
    key = ["target_time", "model_name", "evaluation_mode"]
    combined = (
        combined.drop_duplicates(key, keep="last")
        .sort_values(key)
        .reset_index(drop=True)
    )
    temporary_path = csv_path.with_suffix(csv_path.suffix + ".tmp")
    combined.to_csv(temporary_path, index=False)
    temporary_path.replace(csv_path)
    return combined


def load_predictions_csv(csv_path: Path) -> pd.DataFrame:
    """Load a prediction cache with timezone-aware UTC timestamp columns."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        return pd.DataFrame(columns=CSV_COLUMNS)
    result = pd.read_csv(csv_path)
    for column in ("target_time", "forecast_origin"):
        result[column] = pd.to_datetime(result[column], utc=True)
    return result


def predict_date_range(
    model: PredictionModel,
    combined_data: pd.DataFrame,
    start_date: str,
    end_date: str,
    model_name: str,
    csv_path: Path,
) -> pd.DataFrame:
    """Predict and checkpoint every missing target day in an inclusive range."""
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    if end < start:
        raise ValueError("end_date must not be before start_date")

    data = combined_data.sort_values("time").copy()
    if data.empty or not isinstance(data["time"].dtype, pd.DatetimeTZDtype):
        raise ValueError("combined_data must contain timezone-aware rows")
    timezone_name = data["time"].dt.tz
    cached = load_predictions_csv(csv_path)

    for day in pd.date_range(start, end, freq="D"):
        target_start = pd.Timestamp(day.date(), tz=timezone_name)
        target_end = target_start + pd.DateOffset(days=1)
        expected = data.loc[
            (data["time"] >= target_start) & (data["time"] < target_end), "time"
        ]
        if expected.empty:
            raise ValueError(f"No source rows available for target day {day.date()}")

        if not cached.empty:
            matching = cached.loc[
                (cached["model_name"] == model_name)
                & (cached["evaluation_mode"] == EVALUATION_MODE),
                "target_time",
            ]
            cached_keys = set(matching.dt.tz_convert("UTC"))
            expected_keys = set(expected.dt.tz_convert("UTC"))
            if expected_keys.issubset(cached_keys):
                continue

        daily = predict_target_day(model, data, str(day.date()), model_name)
        cached = save_predictions_csv(daily, csv_path)
        for column in ("target_time", "forecast_origin"):
            cached[column] = pd.to_datetime(cached[column], utc=True)

    if cached.empty:
        return cached
    range_start = pd.Timestamp(start.date(), tz=timezone_name).tz_convert("UTC")
    range_end = (pd.Timestamp(end.date(), tz=timezone_name) + pd.DateOffset(days=1)).tz_convert("UTC")
    return cached.loc[
        (cached["model_name"] == model_name)
        & (cached["evaluation_mode"] == EVALUATION_MODE)
        & (cached["target_time"] >= range_start)
        & (cached["target_time"] < range_end)
    ].sort_values("target_time").reset_index(drop=True)


def predict_date_range_in_memory(
    model: PredictionModel,
    combined_data: pd.DataFrame,
    start_date: str,
    end_date: str,
    model_name: str,
) -> pd.DataFrame:
    """Walk forward over an inclusive date range without persistence.

    This variant is intended for model-training notebooks and tests. Each target
    day is evaluated under the same D-2 information boundary as
    :func:`predict_target_day`, but no CSV checkpoint or database row is written.
    """
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    if end < start:
        raise ValueError("end_date must not be before start_date")

    daily_predictions = [
        predict_target_day(model, combined_data, str(day.date()), model_name)
        for day in pd.date_range(start, end, freq="D")
    ]
    if not daily_predictions:
        return pd.DataFrame(columns=CSV_COLUMNS)
    return pd.concat(daily_predictions, ignore_index=True)


def sync_predictions_csv_to_database(
    csv_path: Path,
    connection,
    model_name: str,
    evaluation_mode: str = EVALUATION_MODE,
) -> pd.DataFrame:
    """Upsert a CSV cache and verify that every cached key exists in SQLite.

    The CSV is intentionally retained after a successful synchronization.
    """
    from prediction_store import load_predictions, upsert_predictions

    cached = load_predictions_csv(csv_path)
    cached = cached.loc[
        (cached["model_name"] == model_name)
        & (cached["evaluation_mode"] == evaluation_mode)
    ].copy()
    if cached.empty:
        return cached

    upsert_predictions(connection, cached)
    persisted = load_predictions(
        connection, model_name=model_name, evaluation_mode=evaluation_mode
    )

    def _keys(frame: pd.DataFrame) -> set[tuple[str, str, str]]:
        target = pd.to_datetime(frame["target_time"], utc=True)
        return set(zip(target.astype(str), frame["model_name"], frame["evaluation_mode"]))

    missing = _keys(cached) - _keys(persisted)
    if missing:
        raise RuntimeError(f"CSV-to-database verification failed for {len(missing)} rows")
    return persisted
