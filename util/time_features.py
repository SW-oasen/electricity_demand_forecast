"""
time_features.py — generic calendar and holiday feature creator.

No country-specific defaults — all domain values must be supplied by the caller
(typically from src/config.py for this project).

Usage example (Germany):
    from util.time_features import TimeFeatureCreator
    from config import DE_STATE_CODES, PANDEMIC_START, PANDEMIC_END

    tfc = TimeFeatureCreator(
        country='DE',
        state_codes=DE_STATE_CODES,
        pandemic_start=PANDEMIC_START,
        pandemic_end=PANDEMIC_END,
    )
    df_with_features = tfc.create(df, year=2024)

Feature selection example:
    tfc = TimeFeatureCreator(
        country='DE',
        state_codes=DE_STATE_CODES,
        include_features=['hour', 'weekday', 'month', 'is_holiday'],
    )

Custom feature extension example:
    tfc = TimeFeatureCreator(
        country='DE',
        state_codes=DE_STATE_CODES,
        extra_feature_fns=[
            lambda df, col: df.assign(quarter=df[col].dt.quarter),
        ],
    )
"""

from __future__ import annotations

from functools import lru_cache
from typing import Callable

import holidays
import pandas as pd


# ---------------------------------------------------------------------------
# Module-level cache — survives across TimeFeatureCreator instances.
# Keyed on (country, state_code, year) so it is safe for multi-country use.
# ---------------------------------------------------------------------------
@lru_cache(maxsize=None)
def _cached_state_holidays(country: str, state_code: str, year: int):
    """Return a holidays.HolidayBase object for one state and one year."""
    return holidays.country_holidays(country, subdiv=state_code, years=[year])


# ---------------------------------------------------------------------------
# Built-in feature catalogue
# ---------------------------------------------------------------------------
# Dependency map: some features need others to already be present in the df.
_FEATURE_DEPS: dict[str, list[str]] = {
    'year':            [],
    'hour':            [],
    'weekday':         [],
    'month':           [],
    'is_weekend':      [],
    'is_holiday':      [],
    'holiday_ratio':   [],
    'is_school_holiday': ['school_holiday_ratio'],
    'school_holiday_ratio': [],
    'is_workday':      ['is_weekend', 'is_holiday'],
    'is_bridge_day':   ['is_holiday'],
    'holiday_weight':  ['holiday_ratio', 'is_weekend'],
    'is_pandemic_time': [],
}

ALL_FEATURES: list[str] = list(_FEATURE_DEPS.keys())


def _resolve_features(requested: list[str]) -> list[str]:
    """Return the requested features plus any dependencies, in topological order."""
    needed: set[str] = set()

    def _add(name: str) -> None:
        if name in needed:
            return
        for dep in _FEATURE_DEPS.get(name, []):
            _add(dep)
        needed.add(name)

    for f in requested:
        if f not in _FEATURE_DEPS:
            raise ValueError(f"Unknown feature '{f}'. Available: {ALL_FEATURES}")
        _add(f)

    # Return in catalogue order so dependencies are always computed first.
    return [f for f in ALL_FEATURES if f in needed]


# ---------------------------------------------------------------------------
# TimeFeatureCreator
# ---------------------------------------------------------------------------
class TimeFeatureCreator:
    """
    Add calendar and public-holiday features to a DataFrame that contains a
    tz-aware datetime column.

    Parameters
    ----------
    country : str
        ISO 3166-1 alpha-2 country code used by the `holidays` library (e.g. 'DE').
    state_codes : list[str]
        Subdivision codes for the country (e.g. German Bundesland codes).
        Used to compute holiday ratios as a share of the selected states' population.
    pandemic_start : pd.Timestamp | None
        Start of the pandemic flag period (inclusive, tz-aware).  None disables
        the is_pandemic_time feature even if it is in include_features.
    pandemic_end : pd.Timestamp | None
        End of the pandemic flag period (inclusive, tz-aware).
    time_column : str
        Name of the datetime column in the input DataFrame (default 'time').
    include_features : list[str] | None
        Subset of built-in features to produce.  None means all features.
        Dependencies are resolved automatically.
        Available: year, hour, weekday, month, is_weekend, is_holiday,
                   holiday_ratio, is_school_holiday, school_holiday_ratio,
                   is_workday, is_bridge_day, holiday_weight, is_pandemic_time
    extra_feature_fns : list[Callable[[pd.DataFrame, str], pd.DataFrame]] | None
        Optional list of user-defined feature functions.  Each receives
        (df, time_column) and must return a DataFrame with new columns added.
        Applied in order after all built-in features.
    """

    def __init__(
        self,
        country: str,
        state_codes: list[str],
        state_weights: dict[str, float] | None = None,
        school_holiday_dates: dict[str, set] | None = None,
        pandemic_start: pd.Timestamp | None = None,
        pandemic_end:   pd.Timestamp | None = None,
        time_column: str = 'time',
        include_features: list[str] | None = None,
        extra_feature_fns: list[Callable] | None = None,
    ) -> None:
        self.country    = country
        self.state_codes = list(state_codes)
        self.state_weights = state_weights or {code: 1.0 for code in state_codes}
        missing_weights = set(self.state_codes) - set(self.state_weights)
        if missing_weights:
            raise ValueError(f"Missing state weights: {sorted(missing_weights)}")
        self.school_holiday_dates = school_holiday_dates or {
            code: set() for code in state_codes
        }
        self.pandemic_start = pandemic_start
        self.pandemic_end   = pandemic_end
        self.time_column    = time_column
        self.extra_feature_fns = extra_feature_fns or []

        if include_features is None:
            self._features = list(ALL_FEATURES)
        else:
            self._features = _resolve_features(include_features)

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def holiday_ratio(self, date) -> float:
        """Fraction of state_codes that observe a public holiday on *date*."""
        holiday_population = sum(
            self.state_weights[code] for code in self.state_codes
            if date in _cached_state_holidays(self.country, code, date.year)
        )
        total_population = sum(
            self.state_weights[code] for code in self.state_codes
        )
        return holiday_population / total_population

    def school_holiday_ratio(self, date) -> float:
        """Population share living in states with school holidays on *date*."""
        holiday_population = sum(
            self.state_weights[code] for code in self.state_codes
            if date in self.school_holiday_dates.get(code, set())
        )
        total_population = sum(
            self.state_weights[code] for code in self.state_codes
        )
        return holiday_population / total_population

    def available_features(self) -> list[str]:
        """Return the list of built-in feature names this instance will produce."""
        return list(self._features)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def create(self, df: pd.DataFrame, year: int) -> pd.DataFrame:
        """
        Add the configured calendar features to *df* and return a new DataFrame.

        Parameters
        ----------
        df   : DataFrame with at least the time_column present.
        year : Maximum year in the dataset — used to pre-build the national
               holiday calendar (must cover all years in df).
        """
        out = df.copy()
        col = self.time_column

        # National holiday calendar spanning 2019 to `year`
        min_year = out[col].dt.year.min()
        national_holidays = holidays.country_holidays(
            self.country, years=range(min_year, year + 1)
        )

        feat = set(self._features)

        if 'year' in feat:
            out['year'] = out[col].dt.year.astype(int)

        if 'hour' in feat:
            out['hour'] = out[col].dt.hour.astype(int)

        if 'weekday' in feat:
            out['weekday'] = out[col].dt.dayofweek.astype(int)

        if 'month' in feat:
            out['month'] = out[col].dt.month.astype(int)

        if 'is_weekend' in feat:
            out['is_weekend'] = (out[col].dt.dayofweek >= 5).astype(int)

        if 'is_holiday' in feat:
            out['is_holiday'] = out[col].dt.date.apply(
                lambda d: 1 if d in national_holidays else 0
            ).astype(int)

        if 'holiday_ratio' in feat:
            out['holiday_ratio'] = out[col].dt.date.apply(self.holiday_ratio).astype(float)

        if 'school_holiday_ratio' in feat:
            out['school_holiday_ratio'] = (
                out[col].dt.date.apply(self.school_holiday_ratio).astype(float)
            )

        if 'is_school_holiday' in feat:
            out['is_school_holiday'] = (
                out['school_holiday_ratio'] > 0
            ).astype(int)

        if 'is_workday' in feat:
            out['is_workday'] = (
                (out['is_weekend'] == 0) & (out['is_holiday'] == 0)
            ).astype(int)

        if 'is_bridge_day' in feat:
            dates = out[col].dt.date
            out['is_bridge_day'] = dates.apply(
                lambda d: 1 if (
                    d.weekday() not in (5, 6)
                    and d not in national_holidays
                    and (
                        (
                            (pd.Timestamp(d) - pd.Timedelta(days=1)).date() in national_holidays
                            or (pd.Timestamp(d) - pd.Timedelta(days=1)).date().weekday() >= 5
                        )
                        and (
                            (pd.Timestamp(d) + pd.Timedelta(days=1)).date() in national_holidays
                            or (pd.Timestamp(d) + pd.Timedelta(days=1)).date().weekday() >= 5
                        )
                    )
                ) else 0
            ).astype(int)

        if 'holiday_weight' in feat:
            out['holiday_weight'] = out[['holiday_ratio', 'is_weekend']].apply(
                lambda row: max(row['holiday_ratio'], row['is_weekend'] * 0.5), axis=1
            ).astype(float)

        if 'is_pandemic_time' in feat:
            if self.pandemic_start is not None and self.pandemic_end is not None:
                out['is_pandemic_time'] = out[col].apply(
                    lambda x: 1 if self.pandemic_start <= x <= self.pandemic_end else 0
                ).astype(int)
            else:
                out['is_pandemic_time'] = 0

        # User-defined extensions
        for fn in self.extra_feature_fns:
            out = fn(out, col)

        return out
