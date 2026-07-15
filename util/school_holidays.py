"""School-holiday loader with a persistent JSON cache."""

from __future__ import annotations

import json
import time
from datetime import date, timedelta
from pathlib import Path

import requests


FERIEN_API_URL = "https://ferien-api.de/api/v1/holidays/{state}/{year}"


def _read_cache(cache_path: Path) -> dict[str, list[dict]]:
    if not cache_path.exists():
        return {}
    return json.loads(cache_path.read_text(encoding="utf-8"))


def _write_cache(cache_path: Path, cache: dict[str, list[dict]]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_suffix(cache_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(cache, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(cache_path)


def load_school_holiday_dates(
    state_codes: list[str],
    years: range | list[int],
    cache_path: Path,
    timeout: int = 20,
) -> dict[str, set[date]]:
    """Return holiday dates by state, fetching only missing state/year pairs."""
    cache_path = Path(cache_path)
    cache = _read_cache(cache_path)
    changed = False
    for state in state_codes:
        for year in years:
            key = f"{state}:{year}"
            if key not in cache:
                print(f"Fetching holidays for {state} in {year}...")
                
                # Retry loop for 429 errors
                max_retries = 3
                for attempt in range(max_retries):
                    response = requests.get(
                        FERIEN_API_URL.format(state=state, year=year),
                        timeout=timeout,
                    )
                    
                    if response.status_code == 429:
                        wait_time = (attempt + 1) * 5
                        print(f"Rate limited (429). Waiting {wait_time}s before retry {attempt + 1}/{max_retries}...")
                        time.sleep(wait_time)
                        continue
                    
                    response.raise_for_status()
                    cache[key] = response.json()
                    changed = True
                    # Small courtesy delay between successful requests to avoid triggering rate limit again
                    time.sleep(0.5)
                    break
    if changed:
        _write_cache(cache_path, cache)

    result = {state: set() for state in state_codes}
    for state in state_codes:
        for year in years:
            for interval in cache.get(f"{state}:{year}", []):
                start = date.fromisoformat(interval["start"])
                end = date.fromisoformat(interval["end"])
                current = start
                while current <= end:
                    result[state].add(current)
                    current += timedelta(days=1)
    return result
