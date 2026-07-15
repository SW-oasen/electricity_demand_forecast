"""As-of historical weather features from archived Open-Meteo model runs."""

from __future__ import annotations

from pathlib import Path
from functools import partial

import pandas as pd
import requests

from config import CITY_POPULATION, SELECTED_CITIES, WEATHER_VARIABLES
from fetch_prepare_data import create_weather_features
from util.openmeteo_client import OpenMeteoClient


ROOT_DIR = Path(__file__).resolve().parents[1]
SINGLE_RUN_CACHE_DIR = ROOT_DIR / "data" / "cache" / "openmeteo_single_runs"
FORECAST_WEATHER_EVALUATION_MODE = "walk_forward_48h_archived_weather"

WEATHER_FEATURE_COLUMNS = [
    *WEATHER_VARIABLES,
    "apparent_temperature_lag_24h",
    "apparent_temperature_rolling_mean_24h",
    "shortwave_radiation_0m_lag_24h",
    "shortwave_radiation_0m_rolling_mean_24h",
    "heating_degree",
    "cooling_degree",
]


def create_single_run_client() -> OpenMeteoClient:
    """Create the project-configured population-weighted weather client."""
    return OpenMeteoClient(
        cities=SELECTED_CITIES,
        city_population=CITY_POPULATION,
        weather_variables=WEATHER_VARIABLES,
    )


def inject_forecast_weather_horizon(
    combined_data: pd.DataFrame,
    target_date: str,
    client: OpenMeteoClient | None = None,
    cache_dir: Path = SINGLE_RUN_CACHE_DIR,
) -> tuple[pd.DataFrame, str]:
    """Replace D-1/D observed weather with archived, leakage-safe forecasts.

    The 24 raw weather rows immediately before the forecast origin are observed
    and therefore known. They provide the context for lag/rolling weather
    features. Every raw weather value at and after the origin comes from the
    same archived model run. If that run has an archive gap, fixed 48-hour
    Best-Match forecasts from Previous Runs provide a leakage-safe fallback.
    """
    if combined_data.empty:
        raise ValueError("combined_data is empty")
    data = combined_data.sort_values("time").drop_duplicates("time").copy()
    if not isinstance(data["time"].dtype, pd.DatetimeTZDtype):
        raise ValueError("combined_data['time'] must be timezone-aware")

    missing_raw = set(WEATHER_VARIABLES) - set(data.columns)
    if missing_raw:
        raise ValueError(f"combined_data lacks weather columns: {sorted(missing_raw)}")

    timezone_name = data["time"].dt.tz
    target_start = pd.Timestamp(target_date, tz=timezone_name)
    target_end = target_start + pd.DateOffset(days=1)
    forecast_origin = target_start - pd.DateOffset(days=1)

    expected = pd.Series(
        pd.date_range(
            forecast_origin,
            target_end,
            freq="h",
            inclusive="left",
        ),
        name="time",
    )
    available_source_times = set(
        data.loc[
            (data["time"] >= forecast_origin) & (data["time"] < target_end),
            "time",
        ]
    )
    missing_source = sorted(set(expected) - available_source_times)
    if missing_source:
        raise ValueError(
            "Combined source data do not fully cover D-1/D; "
            f"missing {len(missing_source)} hourly rows. "
            "Select a target date with complete demand data."
        )

    client = client or create_single_run_client()
    run = client.latest_available_run(forecast_origin)
    try:
        archived_run = client.fetch_single_run(
            run,
            forecast_days=4,
            model="ecmwf_ifs",
            cache_dir=cache_dir,
        )
    except requests.exceptions.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else None
        if status not in {400, 404}:
            raise
        forecast = pd.DataFrame(columns=["time", *WEATHER_VARIABLES])
    else:
        forecast = archived_run.loc[
            (archived_run["time"] >= forecast_origin)
            & (archived_run["time"] < target_end),
            ["time", *WEATHER_VARIABLES],
        ].copy()

    source = f"single_run_ecmwf_ifs:{run.isoformat()}"
    complete_single_run = (
        set(forecast["time"]) == set(expected)
        and not forecast[WEATHER_VARIABLES].isna().any().any()
    )
    if not complete_single_run:
        fallback = client.fetch_previous_runs(
            forecast_origin.tz_convert("UTC").strftime("%Y-%m-%d"),
            (target_end - pd.Timedelta(hours=1))
            .tz_convert("UTC")
            .strftime("%Y-%m-%d"),
            lead_days=2,
            cache_dir=cache_dir,
        )
        forecast = fallback.loc[
            (fallback["time"] >= forecast_origin)
            & (fallback["time"] < target_end),
            ["time", *WEATHER_VARIABLES],
        ].copy()
        source = "previous_runs_best_match:day2"

    missing = sorted(set(expected) - set(forecast["time"]))
    if missing:
        raise ValueError(
            "Archived weather forecasts do not fully cover D-1/D; "
            f"missing {len(missing)} hourly rows"
        )
    forecast = (
        forecast.drop_duplicates("time", keep="last")
        .set_index("time")
        .loc[expected]
        .reset_index()
    )
    if forecast[WEATHER_VARIABLES].isna().any().any():
        missing_columns = forecast.columns[forecast.isna().any()].tolist()
        raise ValueError(
            "Archived weather forecasts contain missing values after fallback: "
            f"{missing_columns}"
        )

    context = data.loc[
        data["time"] < forecast_origin,
        ["time", *WEATHER_VARIABLES],
    ].tail(24)
    if len(context) < 24:
        raise ValueError("At least 24 observed weather rows are required")

    engineered = create_weather_features(
        pd.concat([context, forecast], ignore_index=True)
        .sort_values("time")
        .drop_duplicates("time", keep="last")
        .reset_index(drop=True)
    )
    engineered = engineered.loc[
        engineered["time"] >= forecast_origin,
        ["time", *WEATHER_FEATURE_COLUMNS],
    ]
    if engineered[WEATHER_FEATURE_COLUMNS].isna().any().any():
        raise ValueError("Weather feature engineering produced missing values")

    indexed_features = engineered.set_index("time")
    horizon_mask = (data["time"] >= forecast_origin) & (data["time"] < target_end)
    horizon_times = data.loc[horizon_mask, "time"]
    data.loc[horizon_mask, WEATHER_FEATURE_COLUMNS] = (
        indexed_features.loc[horizon_times, WEATHER_FEATURE_COLUMNS].to_numpy()
    )
    return data, source


def predict_target_day_with_forecast_weather(
    model,
    combined_data: pd.DataFrame,
    target_date: str,
    model_name: str,
    client: OpenMeteoClient | None = None,
    cache_dir: Path = SINGLE_RUN_CACHE_DIR,
) -> pd.DataFrame:
    """Predict one day with recursive demand and as-of archived weather."""
    from walk_forward import predict_target_day

    prepared, _weather_source = inject_forecast_weather_horizon(
        combined_data,
        target_date,
        client=client,
        cache_dir=cache_dir,
    )
    return predict_target_day(
        model,
        prepared,
        target_date,
        model_name,
        evaluation_mode=FORECAST_WEATHER_EVALUATION_MODE,
    )


def predict_date_range_with_forecast_weather(
    model,
    combined_data: pd.DataFrame,
    start_date: str,
    end_date: str,
    model_name: str,
    csv_path: Path,
    client: OpenMeteoClient | None = None,
    cache_dir: Path = SINGLE_RUN_CACHE_DIR,
) -> pd.DataFrame:
    """Checkpoint a date range using leakage-safe archived weather."""
    from walk_forward import predict_date_range

    shared_client = client or create_single_run_client()
    predictor = partial(
        predict_target_day_with_forecast_weather,
        client=shared_client,
        cache_dir=cache_dir,
    )
    return predict_date_range(
        model,
        combined_data,
        start_date,
        end_date,
        model_name,
        csv_path,
        evaluation_mode=FORECAST_WEATHER_EVALUATION_MODE,
        predict_day=predictor,
    )
