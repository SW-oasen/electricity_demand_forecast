import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from util.openmeteo_client import OpenMeteoClient


class OpenMeteoSingleRunTests(unittest.TestCase):
    def setUp(self):
        self.client = OpenMeteoClient(
            cities={
                "large": {"latitude": 1.0, "longitude": 2.0},
                "small": {"latitude": 3.0, "longitude": 4.0},
            },
            city_population={"large": 3, "small": 1},
            weather_variables=["apparent_temperature", "rain"],
            city_sleep=0,
        )

    def test_latest_available_run_uses_six_hour_delay_across_dst(self):
        summer = pd.Timestamp("2025-07-09 00:00", tz="Europe/Berlin")
        winter = pd.Timestamp("2025-01-09 00:00", tz="Europe/Berlin")
        expected_summer = pd.Timestamp("2025-07-08 12:00", tz="UTC")
        expected_winter = pd.Timestamp("2025-01-08 12:00", tz="UTC")
        self.assertEqual(self.client.latest_available_run(summer), expected_summer)
        self.assertEqual(self.client.latest_available_run(winter), expected_winter)

    def test_single_run_is_weighted_cached_and_reused(self):
        responses = []
        for temperature, rain in (([10.0, 14.0], [0.0, 2.0]), ([2.0, 6.0], [4.0, 6.0])):
            response = Mock()
            response.raise_for_status.return_value = None
            response.json.return_value = {
                "hourly": {
                    "time": ["2025-10-01T12:00", "2025-10-01T13:00"],
                    "apparent_temperature": temperature,
                    "rain": rain,
                }
            }
            responses.append(response)

        with tempfile.TemporaryDirectory() as directory:
            cache_dir = Path(directory)
            with patch(
                "util.openmeteo_client.requests.get", side_effect=responses
            ) as request_get:
                first = self.client.fetch_single_run(
                    pd.Timestamp("2025-10-01 12:00", tz="UTC"),
                    cache_dir=cache_dir,
                )
            self.assertEqual(request_get.call_count, 2)
            params = request_get.call_args.kwargs["params"]
            self.assertEqual(params["models"], "ecmwf_ifs")
            self.assertEqual(params["run"], "2025-10-01T12:00")
            self.assertAlmostEqual(first.loc[0, "apparent_temperature"], 8.0)
            self.assertAlmostEqual(first.loc[0, "rain"], 1.0)

            with patch(
                "util.openmeteo_client.requests.get",
                side_effect=AssertionError("cache was not used"),
            ):
                second = self.client.fetch_single_run(
                    pd.Timestamp("2025-10-01 12:00", tz="UTC"),
                    cache_dir=cache_dir,
                )
            pd.testing.assert_frame_equal(first, second)

    def test_single_run_rejects_invalid_cycle(self):
        with self.assertRaisesRegex(ValueError, "00/06/12/18"):
            self.client.fetch_single_run(
                pd.Timestamp("2025-10-01 13:00", tz="UTC")
            )

    def test_single_run_does_not_retry_bad_request(self):
        response = Mock(status_code=400)
        response.raise_for_status.side_effect = requests.HTTPError(
            "unavailable run", response=response
        )
        with patch(
            "util.openmeteo_client.requests.get", return_value=response
        ) as request_get:
            with self.assertRaises(requests.HTTPError):
                self.client.fetch_single_run(
                    pd.Timestamp("2025-12-30 12:00", tz="UTC")
                )
        self.assertEqual(request_get.call_count, 1)

    def test_previous_runs_renames_fixed_lead_variables(self):
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "hourly": {
                "time": ["2025-10-01T00:00"],
                "apparent_temperature_previous_day2": [8.0],
                "rain_previous_day2": [0.5],
            }
        }
        one_city = OpenMeteoClient(
            cities={"city": {"latitude": 1.0, "longitude": 2.0}},
            city_population={"city": 1},
            weather_variables=["apparent_temperature", "rain"],
            city_sleep=0,
        )
        with patch("util.openmeteo_client.requests.get", return_value=response) as get:
            result = one_city.fetch_previous_runs(
                "2025-10-01", "2025-10-01", lead_days=2
            )
        self.assertEqual(
            list(result.columns), ["time", "apparent_temperature", "rain"]
        )
        self.assertEqual(result.loc[0, "rain"], 0.5)
        self.assertIn("apparent_temperature_previous_day2", get.call_args.kwargs["params"]["hourly"])

    def test_previous_runs_renormalizes_available_city_weights(self):
        responses = []
        for temperature, rain in ((10.0, 2.0), (2.0, None)):
            response = Mock()
            response.raise_for_status.return_value = None
            response.json.return_value = {
                "hourly": {
                    "time": ["2025-10-01T00:00"],
                    "apparent_temperature_previous_day2": [temperature],
                    "rain_previous_day2": [rain],
                }
            }
            responses.append(response)

        with patch(
            "util.openmeteo_client.requests.get", side_effect=responses
        ):
            result = self.client.fetch_previous_runs(
                "2025-10-01", "2025-10-01", lead_days=2
            )

        self.assertAlmostEqual(result.loc[0, "apparent_temperature"], 8.0)
        self.assertAlmostEqual(result.loc[0, "rain"], 2.0)


if __name__ == "__main__":
    unittest.main()
