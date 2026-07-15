import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from walk_forward import (
    RECURSIVE_PREDICTION_COLUMN,
    build_energy_features,
    predict_date_range,
    predict_date_range_in_memory,
    predict_target_day,
    save_predictions_csv,
)
from etl import (
    create_database,
    prepare_for_prediction_tomorrow_etl,
)
from config import DE_STATE_CODES


EMPTY_SCHOOL_HOLIDAYS = {code: set() for code in DE_STATE_CODES}


class Lag24Model:
    feature_names_in_ = np.array(["energy_demand_lag_24h"])

    def predict(self, features):
        return features["energy_demand_lag_24h"].to_numpy(dtype=float)


class CountingLag24Model(Lag24Model):
    def __init__(self):
        self.calls = 0

    def predict(self, features):
        self.calls += 1
        return super().predict(features)


class WalkForwardTests(unittest.TestCase):
    def test_energy_features_match_shift_and_rolling(self):
        history = pd.Series(np.arange(1.0, 201.0))
        result = build_energy_features(history)
        self.assertEqual(result["energy_demand_lag_24h"], 177.0)
        self.assertEqual(result["energy_demand_lag_168h"], 33.0)
        self.assertAlmostEqual(result["energy_demand_rolling_mean_24h"], history.iloc[-24:].mean())
        self.assertAlmostEqual(result["energy_demand_rolling_mean_168h"], history.iloc[-168:].mean())

    def test_target_day_uses_predictions_instead_of_hidden_actuals(self):
        time = pd.date_range("2025-01-01", "2025-01-12", freq="h", tz="Europe/Berlin", inclusive="left")
        data = pd.DataFrame({"time": time, "energy_demand_mwh": np.arange(len(time), dtype=float)})
        first = predict_target_day(Lag24Model(), data, "2025-01-10", "test")

        changed = data.copy()
        hidden = changed["time"] >= pd.Timestamp("2025-01-09", tz="Europe/Berlin")
        changed.loc[hidden, "energy_demand_mwh"] += 1_000_000
        second = predict_target_day(Lag24Model(), changed, "2025-01-10", "test")

        np.testing.assert_allclose(first["prediction_mwh"], second["prediction_mwh"])
        self.assertFalse(np.allclose(first["actual_mwh"], second["actual_mwh"]))
        self.assertEqual(len(first), 24)

    def test_dst_target_day_has_23_hours(self):
        time = pd.date_range("2025-03-20", "2025-04-02", freq="h", tz="Europe/Berlin", inclusive="left")
        data = pd.DataFrame({"time": time, "energy_demand_mwh": np.arange(len(time), dtype=float)})
        result = predict_target_day(Lag24Model(), data, "2025-03-30", "test")
        self.assertEqual(len(result), 23)

    def test_dst_target_day_has_25_hours(self):
        time = pd.date_range("2025-10-15", "2025-11-02", freq="h", tz="Europe/Berlin", inclusive="left")
        data = pd.DataFrame({"time": time, "energy_demand_mwh": np.arange(len(time), dtype=float)})
        result = predict_target_day(Lag24Model(), data, "2025-10-26", "test")
        self.assertEqual(len(result), 25)

    def test_csv_merge_is_resumable(self):
        time = pd.date_range("2025-01-01", periods=2, freq="h", tz="Europe/Berlin")
        rows = pd.DataFrame({
            "target_time": time,
            "forecast_origin": time[0] - pd.DateOffset(days=1),
            "model_name": "test",
            "evaluation_mode": "mode",
            "prediction_mwh": [1.0, 2.0],
            "actual_mwh": [1.5, 2.5],
            "created_at": "now",
        })
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.csv"
            save_predictions_csv(rows.iloc[:1], path)
            saved = save_predictions_csv(rows, path)
            self.assertEqual(len(saved), 2)
            self.assertTrue(path.exists())
            self.assertFalse(path.with_suffix(".csv.tmp").exists())

    def test_date_range_resumes_complete_days_from_csv(self):
        time = pd.date_range("2025-01-01", "2025-01-13", freq="h", tz="Europe/Berlin", inclusive="left")
        data = pd.DataFrame({"time": time, "energy_demand_mwh": np.arange(len(time), dtype=float)})
        model = CountingLag24Model()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.csv"
            first = predict_date_range(model, data, "2025-01-10", "2025-01-11", "test", path)
            calls_after_first_run = model.calls
            second = predict_date_range(model, data, "2025-01-10", "2025-01-11", "test", path)
            self.assertEqual(len(first), 48)
            self.assertEqual(len(second), 48)
            self.assertEqual(model.calls, calls_after_first_run)

    def test_in_memory_date_range_has_no_persistence_and_all_days(self):
        time = pd.date_range(
            "2025-01-01", "2025-01-13", freq="h",
            tz="Europe/Berlin", inclusive="left",
        )
        data = pd.DataFrame({
            "time": time,
            "energy_demand_mwh": np.arange(len(time), dtype=float),
        })
        result = predict_date_range_in_memory(
            Lag24Model(), data, "2025-01-10", "2025-01-11", "test"
        )
        self.assertEqual(len(result), 48)
        self.assertEqual(result["target_time"].dt.date.nunique(), 2)

    def test_operational_tomorrow_features_recursively_predict_current_day(self):
        model = Lag24Model()
        history_time = pd.date_range(
            "2025-01-01", "2025-01-10", freq="h",
            tz="Europe/Berlin", inclusive="left",
        )
        values = np.arange(len(history_time), dtype=float)
        hidden_today = history_time >= pd.Timestamp("2025-01-09", tz="Europe/Berlin")
        values[hidden_today] += 1_000_000
        weather_time = pd.date_range(
            "2025-01-09", "2025-01-11", freq="h",
            tz="Europe/Berlin", inclusive="left",
        )
        weather = pd.DataFrame({"time": weather_time})

        with tempfile.TemporaryDirectory() as directory:
            db_path = Path(directory) / "test.db"
            connection = create_database(db_path)
            try:
                rows = [
                    (timestamp.strftime("%Y-%m-%dT%H:%M:%S%z"), float(value))
                    for timestamp, value in zip(history_time, values)
                ]
                connection.executemany(
                    "INSERT INTO energy_demand (time, energy_demand_mwh) VALUES (?, ?)",
                    rows,
                )
                connection.commit()
            finally:
                connection.close()

            with patch(
                "fetch_prepare_data.prepare_weather_for_prediction",
                return_value=weather,
            ), patch(
                "fetch_prepare_data.load_school_holiday_dates",
                return_value=EMPTY_SCHOOL_HOLIDAYS,
            ):
                result = prepare_for_prediction_tomorrow_etl(
                    "2025-01-10", model, db_path
                )

        expected = float(
            values[history_time.get_loc(pd.Timestamp("2025-01-08", tz="Europe/Berlin"))]
        )
        self.assertEqual(len(result), 24)
        self.assertTrue(result[RECURSIVE_PREDICTION_COLUMN].notna().all())
        self.assertEqual(result.iloc[0]["energy_demand_lag_24h"], expected)
        self.assertLess(result.iloc[0]["energy_demand_lag_24h"], 1_000_000)

    def test_operational_tomorrow_rejects_incomplete_weather(self):
        model = Lag24Model()
        history_time = pd.date_range(
            "2025-01-01", "2025-01-09", freq="h",
            tz="Europe/Berlin", inclusive="left",
        )
        weather_time = pd.date_range(
            "2025-01-09", "2025-01-11", freq="h",
            tz="Europe/Berlin", inclusive="left",
        ).delete(10)
        weather = pd.DataFrame({"time": weather_time})
        with tempfile.TemporaryDirectory() as directory:
            db_path = Path(directory) / "test.db"
            connection = create_database(db_path)
            try:
                connection.executemany(
                    "INSERT INTO energy_demand (time, energy_demand_mwh) VALUES (?, ?)",
                    [
                        (timestamp.strftime("%Y-%m-%dT%H:%M:%S%z"), float(index))
                        for index, timestamp in enumerate(history_time)
                    ],
                )
                connection.commit()
            finally:
                connection.close()
            with patch(
                "fetch_prepare_data.prepare_weather_for_prediction",
                return_value=weather,
            ), patch(
                "fetch_prepare_data.load_school_holiday_dates",
                return_value=EMPTY_SCHOOL_HOLIDAYS,
            ):
                with self.assertRaisesRegex(ValueError, "Incomplete weather horizon"):
                    prepare_for_prediction_tomorrow_etl("2025-01-10", model, db_path)


if __name__ == "__main__":
    unittest.main()
