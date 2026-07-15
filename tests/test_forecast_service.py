import sys
import tempfile
import unittest
import sqlite3
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from forecast_service import (
    error_metrics,
    evaluate_historical_range,
    latest_complete_actual_date,
    load_project_models,
)


class ForecastServiceTests(unittest.TestCase):
    def test_latest_complete_actual_date_skips_partial_latest_day(self):
        with tempfile.TemporaryDirectory() as directory:
            db_path = Path(directory) / "energy.db"
            connection = sqlite3.connect(db_path)
            connection.execute("CREATE TABLE energy_demand (time TEXT)")
            complete = pd.date_range(
                "2025-07-13", "2025-07-14", freq="h",
                tz="Europe/Berlin", inclusive="left",
            )
            partial = pd.date_range(
                "2025-07-14", periods=2, freq="h", tz="Europe/Berlin"
            )
            connection.executemany(
                "INSERT INTO energy_demand VALUES (?)",
                [(timestamp.isoformat(),) for timestamp in [*complete, *partial]],
            )
            connection.commit()
            connection.close()

            result = latest_complete_actual_date(db_path)

        self.assertEqual(result.isoformat(), "2025-07-13")

    def test_historical_evaluation_rejects_unknown_weather_mode(self):
        with self.assertRaisesRegex(ValueError, "Unsupported evaluation mode"):
            evaluate_historical_range(
                model=None,
                model_name="test",
                start_date="2025-01-01",
                end_date="2025-01-02",
                csv_dir=Path("unused"),
                evaluation_mode="unknown",
            )

    def test_error_metrics_align_and_ignore_missing_values(self):
        actual = pd.Series([1.0, 3.0, 5.0], index=[0, 1, 2])
        prediction = pd.Series([2.0, None, 1.0], index=[0, 1, 2])
        result = error_metrics(actual, prediction)
        self.assertEqual(result["points"], 2)
        self.assertEqual(result["mae"], 2.5)
        self.assertAlmostEqual(result["rmse"], (17 / 2) ** 0.5)

    def test_model_loader_reports_all_missing_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(
                FileNotFoundError, "best_lgbm_model_bayesian_etl.pkl"
            ):
                load_project_models(Path(directory))

    def test_model_loader_exposes_only_standard_models(self):
        with tempfile.TemporaryDirectory() as directory:
            model_dir = Path(directory)
            filenames = {
                "best_lgbm_model_bayesian_etl.pkl",
                "best_xgb_model_bayesian_etl.pkl",
            }
            for filename in filenames:
                (model_dir / filename).touch()

            loaded = load_project_models(
                model_dir,
                loader=lambda path: path.name,
            )
            self.assertEqual(set(loaded), {"LGBM", "XGBoost"})
            self.assertEqual(set(loaded.values()), filenames)


if __name__ == "__main__":
    unittest.main()
