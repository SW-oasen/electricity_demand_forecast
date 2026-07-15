"""Reusable application services for Streamlit and interactive notebooks."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import pandas as pd

from config import MODEL_FILENAMES
from etl import (
    DEFAULT_DB_PATH,
    get_connection,
    load_combined_data,
    prepare_for_prediction_tomorrow_etl,
)
from prediction_store import load_predictions
from train_model_predict import load_model_from_pickle
from walk_forward import (
    EVALUATION_MODE,
    RECURSIVE_PREDICTION_COLUMN,
    predict_date_range,
    sync_predictions_csv_to_database,
)


def load_project_models(
    model_dir: Path,
    loader: Callable = load_model_from_pickle,
) -> dict:
    """Load every configured ETL model or report all missing artifacts."""
    model_dir = Path(model_dir)
    missing = [name for name in MODEL_FILENAMES.values() if not (model_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            "Fehlende Modellartefakte in models/: " + ", ".join(missing)
            + ". Bitte zuerst Notebook 06 ausführen oder die Modelle bereitstellen."
        )
    return {
        name: loader(model_dir / filename)
        for name, filename in MODEL_FILENAMES.items()
    }


def forecast_tomorrow(
    model,
    prediction_date: str,
    db_path: Path = DEFAULT_DB_PATH,
) -> tuple[pd.DataFrame, pd.Series]:
    """Return target features and their already-recursive demand predictions."""
    features = prepare_for_prediction_tomorrow_etl(prediction_date, model, db_path)
    predictions = features[RECURSIVE_PREDICTION_COLUMN].rename("ML Prediction")
    return features, predictions


def evaluate_historical_range(
    model,
    model_name: str,
    start_date: str,
    end_date: str,
    csv_dir: Path,
    db_path: Path = DEFAULT_DB_PATH,
) -> pd.DataFrame:
    """Run/resume walk-forward evaluation and return aligned comparison series."""
    context_start = str(pd.Timestamp(start_date).date() - pd.Timedelta(days=9))
    connection = get_connection(db_path)
    try:
        source = load_combined_data(connection, context_start, end_date)
    finally:
        connection.close()
    if source.empty:
        return pd.DataFrame(columns=["Actual", "SMARD Forecast", "ML Prediction"])

    csv_path = Path(csv_dir) / f"walk_forward_{model_name.lower()}.csv"
    predict_date_range(model, source, start_date, end_date, model_name, csv_path)

    connection = get_connection(db_path)
    try:
        sync_predictions_csv_to_database(csv_path, connection, model_name)
        predictions = load_predictions(
            connection, model_name, EVALUATION_MODE, start_date, end_date
        )
    finally:
        connection.close()

    target_start = pd.Timestamp(start_date, tz="Europe/Berlin")
    target_end = pd.Timestamp(end_date, tz="Europe/Berlin") + pd.DateOffset(days=1)
    selected = source.loc[
        (source["time"] >= target_start) & (source["time"] < target_end)
    ]
    actual = selected.set_index("time")["energy_demand_mwh"].rename("Actual")
    smard = selected.set_index("time")["smard_forecast_mwh"].rename("SMARD Forecast")
    prediction_time = predictions["target_time"].dt.tz_convert("Europe/Berlin")
    ml = pd.Series(
        predictions["prediction_mwh"].to_numpy(),
        index=prediction_time,
        name="ML Prediction",
    )
    return pd.concat([actual, smard, ml], axis=1)


def error_metrics(actual: pd.Series, prediction: pd.Series) -> dict[str, float | int]:
    """Calculate MAE, RMSE and aligned point count."""
    aligned = pd.concat([actual, prediction], axis=1).dropna()
    if aligned.empty:
        return {"mae": float("nan"), "rmse": float("nan"), "points": 0}
    residual = aligned.iloc[:, 0] - aligned.iloc[:, 1]
    return {
        "mae": float(residual.abs().mean()),
        "rmse": float((residual.pow(2).mean()) ** 0.5),
        "points": len(aligned),
    }
