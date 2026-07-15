import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from forecast_service import error_metrics, load_project_models


class ForecastServiceTests(unittest.TestCase):
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
