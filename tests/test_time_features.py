import sys
import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from util.school_holidays import load_school_holiday_dates
from util.time_features import TimeFeatureCreator
from src.fetch_prepare_data import create_time_based_features


class TimeFeatureTests(unittest.TestCase):
    def test_population_weighted_school_holiday_ratio(self):
        creator = TimeFeatureCreator(
            country="DE",
            state_codes=["BE", "BB"],
            state_weights={"BE": 3.0, "BB": 1.0},
            school_holiday_dates={"BE": {date(2025, 7, 1)}, "BB": set()},
            include_features=["is_school_holiday", "school_holiday_ratio"],
        )
        frame = pd.DataFrame({
            "time": pd.DatetimeIndex([pd.Timestamp("2025-07-01", tz="Europe/Berlin")])
        })
        result = creator.create(frame, 2025)
        self.assertEqual(result.loc[0, "is_school_holiday"], 1)
        self.assertEqual(result.loc[0, "school_holiday_ratio"], 0.75)

    def test_school_holiday_loader_includes_interval_boundaries_and_caches(self):
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = [{"start": "2025-07-01", "end": "2025-07-03"}]
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "school.json"
            with patch("util.school_holidays.requests.get", return_value=response) as get:
                first = load_school_holiday_dates(["BE"], [2025], cache)
                second = load_school_holiday_dates(["BE"], [2025], cache)
            self.assertEqual(get.call_count, 1)
            self.assertEqual(first, second)
            self.assertEqual(
                first["BE"],
                {date(2025, 7, 1), date(2025, 7, 2), date(2025, 7, 3)},
            )

    def test_project_wrapper_injects_population_weighted_school_features(self):
        frame = pd.DataFrame({
            "time": pd.DatetimeIndex([pd.Timestamp("2025-07-01", tz="Europe/Berlin")])
        })
        result = create_time_based_features(
            frame,
            2025,
            in_state_codes=["BE", "BB"],
            in_state_weights={"BE": 3.0, "BB": 1.0},
            in_school_holiday_dates={"BE": {date(2025, 7, 1)}, "BB": set()},
        )
        self.assertEqual(result.loc[0, "is_school_holiday"], 1)
        self.assertEqual(result.loc[0, "school_holiday_ratio"], 0.75)


if __name__ == "__main__":
    unittest.main()
