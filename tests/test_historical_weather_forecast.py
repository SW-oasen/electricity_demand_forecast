import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from config import WEATHER_VARIABLES
from fetch_prepare_data import create_weather_features
from historical_weather_forecast import (
    FORECAST_WEATHER_EVALUATION_MODE,
    inject_forecast_weather_horizon,
    predict_date_range_with_forecast_weather,
    predict_target_day_with_forecast_weather,
)


class ApparentTemperatureModel:
    feature_names_in_ = np.array(["apparent_temperature"])

    def predict(self, features):
        return features["apparent_temperature"].to_numpy(dtype=float)


class HistoricalWeatherForecastTests(unittest.TestCase):
    def _source(self):
        timestamps = pd.date_range(
            "2025-01-01", "2025-01-11", freq="h",
            tz="Europe/Berlin", inclusive="left",
        )
        source = pd.DataFrame({"time": timestamps})
        for offset, column in enumerate(WEATHER_VARIABLES):
            source[column] = np.arange(len(source), dtype=float) + offset
        source = create_weather_features(source)
        source["energy_demand_mwh"] = np.arange(len(source), dtype=float)
        return source

    def test_injects_one_run_and_builds_features_from_known_context(self):
        source = self._source()
        origin = pd.Timestamp("2025-01-09", tz="Europe/Berlin")
        target_end = pd.Timestamp("2025-01-11", tz="Europe/Berlin")
        run_weather = pd.DataFrame({
            "time": pd.date_range(
                origin, target_end, freq="h", inclusive="left"
            )
        })
        for offset, column in enumerate(WEATHER_VARIABLES):
            run_weather[column] = 1000.0 + offset

        client = Mock()
        client.latest_available_run.return_value = pd.Timestamp(
            "2025-01-08 12:00", tz="UTC"
        )
        client.fetch_single_run.return_value = run_weather
        with tempfile.TemporaryDirectory() as directory:
            result, run = inject_forecast_weather_horizon(
                source,
                "2025-01-10",
                client=client,
                cache_dir=Path(directory),
            )

        self.assertEqual(
            run,
            "single_run_ecmwf_ifs:2025-01-08T12:00:00+00:00",
        )
        horizon = result[result["time"] >= origin]
        self.assertTrue((horizon["apparent_temperature"] == 1000.0).all())

        prior = source[source["time"] < origin].tail(24)
        first = horizon.iloc[0]
        self.assertEqual(
            first["apparent_temperature_lag_24h"],
            prior.iloc[0]["apparent_temperature"],
        )
        self.assertAlmostEqual(
            first["apparent_temperature_rolling_mean_24h"],
            prior["apparent_temperature"].mean(),
        )
        client.fetch_single_run.assert_called_once()

    def test_incomplete_single_run_uses_previous_runs_fallback(self):
        source = self._source()
        origin = pd.Timestamp("2025-01-09", tz="Europe/Berlin")
        run_weather = pd.DataFrame({
            "time": pd.date_range(
                origin,
                pd.Timestamp("2025-01-11", tz="Europe/Berlin"),
                freq="h",
                inclusive="left",
            ).delete(5)
        })
        for column in WEATHER_VARIABLES:
            run_weather[column] = 1.0
        client = Mock()
        client.latest_available_run.return_value = pd.Timestamp(
            "2025-01-08 12:00", tz="UTC"
        )
        client.fetch_single_run.return_value = run_weather
        fallback = pd.DataFrame({
            "time": pd.date_range(
                origin,
                pd.Timestamp("2025-01-11", tz="Europe/Berlin"),
                freq="h",
                inclusive="left",
            )
        })
        for column in WEATHER_VARIABLES:
            fallback[column] = 77.0
        client.fetch_previous_runs.return_value = fallback
        with tempfile.TemporaryDirectory() as directory:
            result, weather_source = inject_forecast_weather_horizon(
                source,
                "2025-01-10",
                client=client,
                cache_dir=Path(directory),
            )

        self.assertEqual(weather_source, "previous_runs_best_match:day2")
        horizon = result[result["time"] >= origin]
        self.assertTrue((horizon["apparent_temperature"] == 77.0).all())
        client.fetch_previous_runs.assert_called_once()

    def test_rejects_incomplete_fallback_horizon(self):
        source = self._source()
        origin = pd.Timestamp("2025-01-09", tz="Europe/Berlin")
        incomplete = pd.DataFrame({
            "time": pd.date_range(
                origin,
                pd.Timestamp("2025-01-11", tz="Europe/Berlin"),
                freq="h",
                inclusive="left",
            ).delete(5)
        })
        for column in WEATHER_VARIABLES:
            incomplete[column] = 1.0
        client = Mock()
        client.latest_available_run.return_value = pd.Timestamp(
            "2025-01-08 12:00", tz="UTC"
        )
        client.fetch_single_run.return_value = incomplete
        client.fetch_previous_runs.return_value = incomplete

        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "do not fully cover"):
                inject_forecast_weather_horizon(
                    source,
                    "2025-01-10",
                    client=client,
                    cache_dir=Path(directory),
                )

    def test_rejects_incomplete_combined_source_with_clear_message(self):
        source = self._source().iloc[:-1].copy()
        client = Mock()

        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(
                ValueError, "Select a target date with complete demand data"
            ):
                inject_forecast_weather_horizon(
                    source,
                    "2025-01-10",
                    client=client,
                    cache_dir=Path(directory),
                )

        client.fetch_single_run.assert_not_called()

    def test_prediction_uses_separate_mode_and_not_observed_horizon_weather(self):
        source = self._source()
        origin = pd.Timestamp("2025-01-09", tz="Europe/Berlin")
        target_end = pd.Timestamp("2025-01-11", tz="Europe/Berlin")
        run_weather = pd.DataFrame({
            "time": pd.date_range(origin, target_end, freq="h", inclusive="left")
        })
        for column in WEATHER_VARIABLES:
            run_weather[column] = 500.0

        client = Mock()
        client.latest_available_run.return_value = pd.Timestamp(
            "2025-01-08 12:00", tz="UTC"
        )
        client.fetch_single_run.return_value = run_weather
        changed = source.copy()
        changed.loc[changed["time"] >= origin, WEATHER_VARIABLES] += 1_000_000

        with tempfile.TemporaryDirectory() as directory:
            first = predict_target_day_with_forecast_weather(
                ApparentTemperatureModel(), source, "2025-01-10", "test",
                client=client, cache_dir=Path(directory),
            )
            second = predict_target_day_with_forecast_weather(
                ApparentTemperatureModel(), changed, "2025-01-10", "test",
                client=client, cache_dir=Path(directory),
            )

        np.testing.assert_allclose(first["prediction_mwh"], 500.0)
        np.testing.assert_allclose(
            first["prediction_mwh"], second["prediction_mwh"]
        )
        self.assertEqual(
            set(first["evaluation_mode"]),
            {FORECAST_WEATHER_EVALUATION_MODE},
        )

    def test_date_range_uses_separate_resumable_cache(self):
        source = self._source()
        origin = pd.Timestamp("2025-01-09", tz="Europe/Berlin")
        target_end = pd.Timestamp("2025-01-11", tz="Europe/Berlin")
        run_weather = pd.DataFrame({
            "time": pd.date_range(origin, target_end, freq="h", inclusive="left")
        })
        for column in WEATHER_VARIABLES:
            run_weather[column] = 500.0
        client = Mock()
        client.latest_available_run.return_value = pd.Timestamp(
            "2025-01-08 12:00", tz="UTC"
        )
        client.fetch_single_run.return_value = run_weather

        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / "forecast_weather.csv"
            first = predict_date_range_with_forecast_weather(
                ApparentTemperatureModel(), source,
                "2025-01-10", "2025-01-10", "test", csv_path,
                client=client, cache_dir=Path(directory) / "weather",
            )
            call_count = client.fetch_single_run.call_count
            second = predict_date_range_with_forecast_weather(
                ApparentTemperatureModel(), source,
                "2025-01-10", "2025-01-10", "test", csv_path,
                client=client, cache_dir=Path(directory) / "weather",
            )

        self.assertEqual(len(first), 24)
        self.assertEqual(len(second), 24)
        self.assertEqual(client.fetch_single_run.call_count, call_count)
        self.assertEqual(
            set(second["evaluation_mode"]),
            {FORECAST_WEATHER_EVALUATION_MODE},
        )


if __name__ == "__main__":
    unittest.main()
