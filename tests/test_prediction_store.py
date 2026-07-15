import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from etl import create_database
from prediction_store import load_predictions, upsert_predictions
from walk_forward import save_predictions_csv, sync_predictions_csv_to_database


def prediction_rows() -> pd.DataFrame:
    time = pd.date_range("2025-01-10", periods=2, freq="h", tz="Europe/Berlin")
    return pd.DataFrame({
        "target_time": time,
        "forecast_origin": pd.Timestamp("2025-01-09", tz="Europe/Berlin"),
        "model_name": "test",
        "evaluation_mode": "mode",
        "prediction_mwh": [1.0, 2.0],
        "actual_mwh": [1.5, 2.5],
        "created_at": "2025-01-01T00:00:00+00:00",
    })


class PredictionStoreTests(unittest.TestCase):
    def test_database_upsert_is_idempotent(self):
        rows = prediction_rows()
        with tempfile.TemporaryDirectory() as directory:
            connection = create_database(Path(directory) / "test.db")
            try:
                upsert_predictions(connection, rows)
                changed = rows.copy()
                changed.loc[0, "prediction_mwh"] = 9.0
                upsert_predictions(connection, changed)
                loaded = load_predictions(connection, "test", "mode")
            finally:
                connection.close()
            self.assertEqual(len(loaded), 2)
            self.assertEqual(loaded.iloc[0]["prediction_mwh"], 9.0)

    def test_csv_database_sync_keeps_csv_and_verifies_rows(self):
        rows = prediction_rows()
        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / "predictions.csv"
            save_predictions_csv(rows, csv_path)
            connection = create_database(Path(directory) / "test.db")
            try:
                persisted = sync_predictions_csv_to_database(
                    csv_path, connection, "test", "mode"
                )
            finally:
                connection.close()
            self.assertEqual(len(persisted), 2)
            self.assertTrue(csv_path.exists())


if __name__ == "__main__":
    unittest.main()
