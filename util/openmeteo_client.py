"""
openmeteo_client.py — generic client for the Open-Meteo archive and forecast APIs.

No city list or variable defaults — all domain values must be supplied by the caller
(typically from src/config.py for this project).

Usage example (Germany):
    from util.openmeteo_client import OpenMeteoClient
    from config import SELECTED_CITIES, CITY_POPULATION, WEATHER_VARIABLES

    client = OpenMeteoClient(
        cities=SELECTED_CITIES,
        city_population=CITY_POPULATION,
        weather_variables=WEATHER_VARIABLES,
    )

    df_hist    = client.fetch_archive('2024-01-01', '2024-12-31')
    df_fc      = client.fetch_forecast(forecast_days=3)
    df_pred    = client.prepare_for_prediction('2026-05-27')

API reference: documents/open-meteo_api.md

Returned DataFrames always have:
    time — tz-aware datetime64[s] (Europe/Berlin)
    one column per weather variable (population-weighted aggregate)
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import requests
import truststore


# Requests otherwise uses only certifi. On managed Windows installations the
# required issuer may live in the OS trust store; this keeps verification active.
truststore.inject_into_ssl()


_ARCHIVE_URL  = "https://archive-api.open-meteo.com/v1/archive"
_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
_SINGLE_RUNS_URL = "https://single-runs-api.open-meteo.com/v1/forecast"
_PREVIOUS_RUNS_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"


class OpenMeteoClient:
    """
    Fetch and population-weight hourly weather data from Open-Meteo for a set
    of cities.

    Parameters
    ----------
    cities : dict
        Mapping of city name → {'latitude': float, 'longitude': float}.
    city_population : dict
        Mapping of city name → population (int).  Used to compute weights.
    weather_variables : list[str]
        Open-Meteo variable names to request (e.g. 'apparent_temperature').
    city_sleep : float
        Seconds to sleep between per-city requests (default 1 s).
    timeout : int
        HTTP request timeout in seconds (default 30).
    """

    def __init__(
        self,
        cities: dict,
        city_population: dict,
        weather_variables: list[str],
        city_sleep: float = 1.0,
        timeout: int = 30,
    ) -> None:
        if set(cities) != set(city_population):
            raise ValueError("cities and city_population must have the same keys.")
        self.cities            = cities
        self.city_population   = city_population
        self.weather_variables = list(weather_variables)
        self.city_sleep        = city_sleep
        self.timeout           = timeout

        total = sum(city_population.values())
        self._weights = {city: pop / total for city, pop in city_population.items()}

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _merge_cities(
        self,
        city_dict: dict[str, pd.DataFrame],
        renormalize_available: bool = False,
    ) -> pd.DataFrame:
        """
        Population-weight the per-city DataFrames into one aggregate DataFrame.
        The 'time' column is taken from the first city; numeric columns are
        weighted-summed across cities. With ``renormalize_available``, missing
        values at individual locations are excluded and the remaining
        population weights are normalized per variable and timestamp.
        """
        if renormalize_available:
            first = next(iter(city_dict.values()))
            out = first[["time"]].copy()
            for variable in self.weather_variables:
                numerator = pd.Series(0.0, index=out.index)
                denominator = pd.Series(0.0, index=out.index)
                for city, city_data in city_dict.items():
                    values = city_data[variable].reset_index(drop=True)
                    available = values.notna()
                    weight = self._weights[city]
                    numerator = numerator.add(values.fillna(0.0) * weight)
                    denominator = denominator.add(available.astype(float) * weight)
                out[variable] = numerator.div(denominator.where(denominator > 0))
            return out

        out = pd.DataFrame()
        for city, df_city in city_dict.items():
            w = self._weights[city]
            df_w = df_city.copy()
            for var in self.weather_variables:
                df_w[var] = df_city[var] * w
            if out.empty:
                out = df_w
            else:
                out[self.weather_variables] = (
                    out[self.weather_variables].values
                    + df_w[self.weather_variables].values
                )
        return out

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @staticmethod
    def latest_available_run(
        forecast_origin: pd.Timestamp,
        availability_delay_hours: int = 6,
    ) -> pd.Timestamp:
        """Return the latest six-hourly run safely available at an origin.

        Open-Meteo documents a typical 4–6 hour publication delay for global
        models. Using the upper bound prevents selecting a run that had been
        initialised but was not yet publicly available at ``forecast_origin``.
        """
        origin = pd.Timestamp(forecast_origin)
        if origin.tzinfo is None:
            raise ValueError("forecast_origin must be timezone-aware")
        cutoff = origin.tz_convert("UTC") - pd.Timedelta(
            hours=availability_delay_hours
        )
        run_hour = (cutoff.hour // 6) * 6
        return cutoff.normalize() + pd.Timedelta(hours=run_hour)

    def fetch_single_run(
        self,
        run: pd.Timestamp,
        forecast_days: int = 4,
        model: str = "ecmwf_ifs",
        cache_dir: Path | None = None,
    ) -> pd.DataFrame:
        """Fetch and population-weight one archived Open-Meteo model run.

        ``run`` is the model initialisation time in UTC. When ``cache_dir`` is
        supplied, the aggregated run is stored atomically as CSV and reused on
        subsequent calls. No TLS verification is disabled by this client.
        """
        run = pd.Timestamp(run)
        if run.tzinfo is None:
            raise ValueError("run must be timezone-aware")
        run_utc = run.tz_convert("UTC")
        if run_utc.minute or run_utc.second or run_utc.hour % 6:
            raise ValueError("run must be a 00/06/12/18 UTC model cycle")
        if not 1 <= forecast_days <= 16:
            raise ValueError("forecast_days must be between 1 and 16")

        cache_path = None
        if cache_dir is not None:
            safe_model = "".join(
                char for char in model if char.isalnum() or char in ("-", "_")
            )
            cache_path = (
                Path(cache_dir)
                / safe_model
                / f"{run_utc.strftime('%Y%m%dT%H%MZ')}_{forecast_days}d.csv"
            )
            if cache_path.exists():
                cached = pd.read_csv(cache_path)
                required = {"time", *self.weather_variables}
                if required.issubset(cached.columns):
                    cached["time"] = (
                        pd.to_datetime(cached["time"], utc=True)
                        .dt.tz_convert("Europe/Berlin")
                        .dt.as_unit("s")
                    )
                    return cached.sort_values("time").reset_index(drop=True)

        city_dict: dict[str, pd.DataFrame] = {}
        params_base = {
            "hourly": ",".join(self.weather_variables),
            "models": model,
            "run": run_utc.strftime("%Y-%m-%dT%H:%M"),
            "timezone": "UTC",
            "forecast_days": forecast_days,
        }
        for city, coords in self.cities.items():
            params = {
                **params_base,
                "latitude": coords["latitude"],
                "longitude": coords["longitude"],
            }
            for attempt in range(3):
                try:
                    response = requests.get(
                        _SINGLE_RUNS_URL,
                        params=params,
                        timeout=self.timeout,
                    )
                    response.raise_for_status()
                    data = response.json()
                    break
                except requests.exceptions.RequestException:
                    if attempt == 2:
                        raise
                    time.sleep(5)

            df_city = pd.DataFrame(data["hourly"])
            missing = set(self.weather_variables) - set(df_city.columns)
            if missing:
                raise ValueError(
                    f"Single Runs response for {city} lacks variables: "
                    f"{sorted(missing)}"
                )
            df_city["time"] = (
                pd.to_datetime(df_city["time"], utc=True)
                .dt.tz_convert("Europe/Berlin")
                .dt.as_unit("s")
            )
            city_dict[city] = df_city[["time", *self.weather_variables]]
            time.sleep(self.city_sleep)

        result = self._merge_cities(city_dict).sort_values("time").reset_index(drop=True)
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = cache_path.with_suffix(cache_path.suffix + ".tmp")
            to_cache = result.copy()
            to_cache["time"] = to_cache["time"].dt.strftime(
                "%Y-%m-%dT%H:%M:%S%z"
            )
            to_cache.to_csv(temporary, index=False)
            temporary.replace(cache_path)
        return result

    def fetch_previous_runs(
        self,
        start_date: str,
        end_date: str,
        lead_days: int = 2,
        cache_dir: Path | None = None,
    ) -> pd.DataFrame:
        """Fetch fixed-lead archived forecasts, using Open-Meteo Best Match."""
        if not 1 <= lead_days <= 7:
            raise ValueError("lead_days must be between 1 and 7")
        suffix = f"_previous_day{lead_days}"
        requested = [f"{name}{suffix}" for name in self.weather_variables]
        cache_path = None
        if cache_dir is not None:
            cache_path = (
                Path(cache_dir)
                / "previous_runs_best_match"
                / f"{start_date}_{end_date}_{lead_days}d_v2.csv"
            )
            if cache_path.exists():
                cached = pd.read_csv(cache_path)
                required = {"time", *self.weather_variables}
                if required.issubset(cached.columns):
                    cached["time"] = (
                        pd.to_datetime(cached["time"], utc=True)
                        .dt.tz_convert("Europe/Berlin")
                        .dt.as_unit("s")
                    )
                    return cached.sort_values("time").reset_index(drop=True)

        city_dict: dict[str, pd.DataFrame] = {}
        for city, coords in self.cities.items():
            params = {
                "latitude": coords["latitude"],
                "longitude": coords["longitude"],
                "start_date": start_date,
                "end_date": end_date,
                "hourly": ",".join(requested),
                "timezone": "UTC",
            }
            response = requests.get(
                _PREVIOUS_RUNS_URL, params=params, timeout=self.timeout
            )
            response.raise_for_status()
            df_city = pd.DataFrame(response.json()["hourly"]).rename(
                columns={forecast: actual for forecast, actual in zip(requested, self.weather_variables)}
            )
            df_city["time"] = (
                pd.to_datetime(df_city["time"], utc=True)
                .dt.tz_convert("Europe/Berlin")
                .dt.as_unit("s")
            )
            city_dict[city] = df_city[["time", *self.weather_variables]]
            time.sleep(self.city_sleep)

        result = (
            self._merge_cities(city_dict, renormalize_available=True)
            .sort_values("time")
            .reset_index(drop=True)
        )
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = cache_path.with_suffix(cache_path.suffix + ".tmp")
            to_cache = result.copy()
            to_cache["time"] = to_cache["time"].dt.strftime("%Y-%m-%dT%H:%M:%S%z")
            to_cache.to_csv(temporary, index=False)
            temporary.replace(cache_path)
        return result

    def fetch_archive(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch historical (archive) weather from Open-Meteo for all cities,
        population-weight the result, and return a single DataFrame.

        Fetches one extra day before start_date in UTC to ensure Berlin midnight
        (= UTC-2 the prior day) is included, then clips to start_date Berlin time.

        Parameters
        ----------
        start_date : str  'YYYY-MM-DD' (Europe/Berlin local)
        end_date   : str  'YYYY-MM-DD' (Europe/Berlin local)

        Returns
        -------
        DataFrame with columns: ['time'] + weather_variables
        """
        # Fetch 1 extra day in UTC so Berlin 00:00+02:00 is always included.
        api_start = (pd.to_datetime(start_date) - pd.Timedelta(days=1)).strftime('%Y-%m-%d')
        clip_start = pd.Timestamp(start_date, tz='Europe/Berlin')
        vars_str = ','.join(self.weather_variables)

        city_dict: dict[str, pd.DataFrame] = {}
        for city, coords in self.cities.items():
            url = (
                f"{_ARCHIVE_URL}"
                f"?latitude={coords['latitude']}"
                f"&longitude={coords['longitude']}"
                f"&start_date={api_start}"
                f"&end_date={end_date}"
                f"&hourly={vars_str}"
                f"&timezone=UTC"
            )
            for attempt in range(3):
                try:
                    r = requests.get(url, timeout=self.timeout)
                    r.raise_for_status()
                    data = r.json()
                    break
                except requests.exceptions.RequestException:
                    if attempt == 2:
                        raise
                    time.sleep(5)

            df_city = pd.DataFrame(data['hourly'])
            df_city['time'] = (
                pd.to_datetime(df_city['time'], utc=True)
                .dt.tz_convert('Europe/Berlin')
                .dt.as_unit('s')
            )
            df_city = df_city[df_city['time'] >= clip_start].reset_index(drop=True)
            city_dict[city] = df_city
            time.sleep(self.city_sleep)

        df = self._merge_cities(city_dict)
        return df.sort_values('time').reset_index(drop=True)

    def fetch_forecast(self, forecast_days: int = 2) -> pd.DataFrame:
        """
        Fetch hourly weather forecast from Open-Meteo for all cities,
        population-weight the result, and return a single DataFrame.

        Uses past_days=1 to ensure Berlin midnight is included, then clips to
        today's Berlin midnight so the series always starts at 00:00.

        Parameters
        ----------
        forecast_days : int  1–16 (Open-Meteo free tier max is 16).

        Returns
        -------
        DataFrame with columns: ['time'] + weather_variables
        """
        vars_str   = ','.join(self.weather_variables)
        today_midnight = pd.Timestamp.now(tz='Europe/Berlin').normalize()

        city_dict: dict[str, pd.DataFrame] = {}
        for city, coords in self.cities.items():
            url = (
                f"{_FORECAST_URL}"
                f"?latitude={coords['latitude']}"
                f"&longitude={coords['longitude']}"
                f"&hourly={vars_str}"
                f"&forecast_days={forecast_days}"
                f"&past_days=1"
                f"&timezone=UTC"
            )
            for attempt in range(3):
                try:
                    r = requests.get(url, timeout=self.timeout)
                    r.raise_for_status()
                    data = r.json()
                    break
                except requests.exceptions.RequestException:
                    if attempt == 2:
                        raise
                    time.sleep(5)

            df_city = pd.DataFrame(data['hourly'])
            df_city['time'] = (
                pd.to_datetime(df_city['time'], utc=True)
                .dt.tz_convert('Europe/Berlin')
                .dt.as_unit('s')
            )
            df_city = df_city[df_city['time'] >= today_midnight].reset_index(drop=True)
            city_dict[city] = df_city
            time.sleep(self.city_sleep)

        df = self._merge_cities(city_dict)
        return df.sort_values('time').reset_index(drop=True)

    def prepare_for_prediction(
        self,
        prediction_date: str,
        lookback_days: int = 2,
        forecast_days: int = 3,
    ) -> pd.DataFrame:
        """
        Combine archive (lookback context) and forecast data for a prediction date.

        The archive lookback provides the lag/rolling context rows that the
        feature engineering step (create_weather_features in fetch_prepare_data.py)
        needs.  Without it, lag/rolling values for the prediction day would be NaN.

        Parameters
        ----------
        prediction_date : str  'YYYY-MM-DD'
        lookback_days   : int  Days of archive history before prediction_date.
        forecast_days   : int  Days of forecast to include (must cover prediction_date).

        Returns
        -------
        DataFrame covering [prediction_date - lookback_days, prediction_date + (forecast_days-1)]
        with columns: ['time'] + weather_variables
        """
        archive_start = (
            pd.to_datetime(prediction_date) - pd.Timedelta(days=lookback_days)
        ).strftime('%Y-%m-%d')
        archive_end = (
            pd.to_datetime(prediction_date) - pd.Timedelta(days=1)
        ).strftime('%Y-%m-%d')

        df_archive = self.fetch_archive(archive_start, archive_end)

        df_forecast = self.fetch_forecast(forecast_days=forecast_days)

        pred_start = pd.Timestamp(prediction_date, tz='Europe/Berlin')
        df_archive = df_archive[df_archive['time'] < pred_start].copy()

        df_combined = (
            pd.concat([df_archive, df_forecast], ignore_index=True)
            .sort_values('time')
            .drop_duplicates(subset=['time'])   # forecast wins on overlap
            .reset_index(drop=True)
        )
        return df_combined
