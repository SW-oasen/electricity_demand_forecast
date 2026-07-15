# SMARD Chart Data API

## Overview

The **SMARD** (Strommarktdaten) platform is operated by the **Bundesnetzagentur** (Federal Network Agency of Germany) and publishes electricity market data including realized grid load, generation, and prices.

- Website: [https://www.smard.de](https://www.smard.de/home)
- API type: Unofficial JSON REST API (undocumented, reverse-engineered from the SMARD web app)
- Authentication: None required
- Rate limiting: Polite usage recommended (`0.3 s` sleep between requests)

---

## API Structure

### Base URL

```
https://www.smard.de/app/chart_data
```

### Filter IDs

Each data series is identified by a numeric filter ID.

#### Electricity Consumption (`Stromverbrauch`)

| Filter ID | Series (DE) | Series (EN) | Unit |
|-----------|-------------|-------------|------|
| `410` | Stromverbrauch: Gesamt (Netzlast) | **Realized grid load** ← used in this project | MWh |
| `411` | Prognostizierter Stromverbrauch: Netzlast | **Forecasted grid load** (SMARD official forecast) | MWh |
| `4359` | Stromverbrauch: Residuallast | Residual load (total minus wind + PV) | MWh |
| `4387` | Stromverbrauch: Pumpspeicher | Pumped storage consumption | MWh |

> **Filter 411 note:** Confirmed via `chart_data` API: returns hourly consumption forecasts for Germany (~40,000–63,000 MWh/h range), available from 2014-12-29 onward. Corresponds to module ID `6000411` on the SMARD website. Useful for benchmarking ML forecasts against SMARD's official day-ahead consumption forecast.
> Example: `fetch_smard_netzlast(start, end, filter_id=411)` returns the same `['time', 'EnergyDemand']` schema.

#### Electricity Generation (`Stromerzeugung`) — Realized

| Filter ID | Series (DE) | Series (EN) | Unit |
|-----------|-------------|-------------|------|
| `4067` | Stromerzeugung: Wind Onshore | Onshore wind generation | MWh |
| `1225` | Stromerzeugung: Wind Offshore | Offshore wind generation | MWh |
| `4068` | Stromerzeugung: Photovoltaik | Solar PV generation | MWh |
| `4066` | Stromerzeugung: Biomasse | Biomass generation | MWh |
| `1226` | Stromerzeugung: Wasserkraft | Run-of-river hydropower | MWh |
| `4070` | Stromerzeugung: Pumpspeicher | Pumped storage generation | MWh |
| `4071` | Stromerzeugung: Erdgas | Natural gas generation | MWh |
| `1223` | Stromerzeugung: Braunkohle | Lignite (brown coal) generation | MWh |
| `4069` | Stromerzeugung: Steinkohle | Hard coal generation | MWh |
| `1224` | Stromerzeugung: Kernenergie | Nuclear generation | MWh |
| `1227` | Stromerzeugung: Sonstige Konventionelle | Other conventional generation | MWh |
| `1228` | Stromerzeugung: Sonstige Erneuerbare | Other renewable generation | MWh |

#### Generation Forecasts (`Prognostizierte Erzeugung`)

| Filter ID | Series (DE) | Series (EN) | Unit |
|-----------|-------------|-------------|------|
| `122`  | Prognostizierte Erzeugung: Gesamt | Total generation forecast | MWh |
| `3791` | Prognostizierte Erzeugung: Wind Offshore | Offshore wind forecast | MWh |
| `123`  | Prognostizierte Erzeugung: Wind Onshore | Onshore wind forecast | MWh |
| `125`  | Prognostizierte Erzeugung: Photovoltaik | Solar PV forecast | MWh |
| `126`  | Prognostizierte Erzeugung: Photovoltaik (alt.) | Solar PV forecast (alternate) | MWh |
| `715`  | Prognostizierte Erzeugung: Sonstige | Other forecast | MWh |
| `5097` | Prognostizierte Erzeugung: Wind und Photovoltaik | Wind + solar combined forecast | MWh |

> **Note:** Both `125` and `126` appear in different sources for solar PV forecasts — verify availability for your target region and date range.

#### Day-Ahead Market Prices (`Marktpreis`)

| Filter ID | Series (DE) | Series (EN) | Unit |
|-----------|-------------|-------------|------|
| `4169` | Marktpreis: Deutschland/Luxemburg | Day-ahead price DE/LU | EUR/MWh |
| `5078` | Marktpreis: Anrainer DE/LU | Neighboring countries avg. | EUR/MWh |
| `4170` | Marktpreis: Österreich | Austria | EUR/MWh |
| `4996` | Marktpreis: Belgien | Belgium | EUR/MWh |
| `4997` | Marktpreis: Norwegen 2 | Norway 2 | EUR/MWh |
| `252`  | Marktpreis: Dänemark 1 | Denmark 1 | EUR/MWh |
| `253`  | Marktpreis: Dänemark 2 | Denmark 2 | EUR/MWh |
| `254`  | Marktpreis: Frankreich | France | EUR/MWh |
| `255`  | Marktpreis: Italien (Nord) | Italy North | EUR/MWh |
| `256`  | Marktpreis: Niederlande | Netherlands | EUR/MWh |
| `257`  | Marktpreis: Polen | Poland | EUR/MWh |
| `258`  | Marktpreis: Polen (alt.) | Poland (alternate) | EUR/MWh |
| `259`  | Marktpreis: Schweiz | Switzerland | EUR/MWh |
| `260`  | Marktpreis: Slowenien | Slovenia | EUR/MWh |
| `261`  | Marktpreis: Tschechien | Czech Republic | EUR/MWh |
| `262`  | Marktpreis: Ungarn | Hungary | EUR/MWh |

---

### Region Codes

| Code | Type | Description |
|------|------|-------------|
| `DE` | Country | Germany (default) |
| `AT` | Country | Austria |
| `LU` | Country | Luxembourg |
| `DE-LU` | Market area | DE/LU market area (from 01.10.2018) |
| `DE-AT-LU` | Market area | DE/AT/LU market area (until 30.09.2018) |
| `50Hertz` | Control area (DE) | 50Hertz TSO zone (NE Germany) |
| `Amprion` | Control area (DE) | Amprion TSO zone (NW/W Germany) |
| `TenneT` | Control area (DE) | TenneT TSO zone (N/SE Germany) |
| `TransnetBW` | Control area (DE) | TransnetBW TSO zone (SW Germany) |
| `APG` | Control area (AT) | Austrian Power Grid |
| `Creos` | Control area (LU) | Creos Luxembourg |

### Resolutions

| Value | Description | Availability |
|-------|-------------|--------------|
| `hour` | Hourly data (default) | Most filter IDs |
| `quarterhour` | 15-minute data | Some filter IDs |
| `day` | Daily aggregates | Most filter IDs |
| `week` | Weekly aggregates | Most filter IDs |
| `month` | Monthly aggregates | Most filter IDs |
| `year` | Yearly aggregates | Most filter IDs |

---

## Endpoints

### 1. Index Endpoint — List Available Weekly Buckets

```
GET {base}/{filter_id}/{region}/index_{resolution}.json
```

**Example:**
```
GET https://www.smard.de/app/chart_data/410/DE/index_hour.json
```

**Response:**
```json
{
  "timestamps": [1546300800000, 1546905600000, ...]
}
```

- Returns a list of Unix timestamps in **milliseconds** (UTC).
- Each timestamp marks the start of a weekly data bucket.

---

### 2. Weekly Data Endpoint — Fetch One Week of Data

```
GET {base}/{filter_id}/{region}/{filter_id}_{region}_{resolution}_{timestamp_ms}.json
```

**Example:**
```
GET https://www.smard.de/app/chart_data/410/DE/410_DE_hour_1546300800000.json
```

**Response:**
```json
{
  "series": [
    [1546300800000, 58234.0],
    [1546304400000, 55100.0],
    ...
  ]
}
```

- Each entry is `[timestamp_ms, value]`.
- `value` is in **MWh** for energy series, **EUR/MWh** for price series.
- Entries with `null` values (missing data) are dropped during processing.

### 3. Table Data Endpoint — 15-Minute Versioned Data

```
GET https://www.smard.de/app/table_data/{filter_id}/{region}/{filter_id}_{region}_quarterhour_{timestamp_ms}.json
```

**Example:**
```
GET https://www.smard.de/app/table_data/410/DE/410_DE_quarterhour_1546300800000.json
```

**Response:**
```json
{
  "series": [
    { "timestamp": 1546300800000, "versions": [{ "value": 14500.0, "name": null }] },
    ...
  ]
}
```

- Returns **versioned** 15-minute data — each row may contain multiple publication versions.
- Useful when you need the most recently published revision of a value.
- Only `quarterhour` resolution is supported for this endpoint.

---

## Implementation in This Project

### Purpose

The Kaggle dataset (ENTSO-E based) covers **2019-01-01 to 2025-09-30**. SMARD is used to extend the energy demand data from **2025-10-01 onward** for:

- SMARD benchmark series in Notebook 11 and `streamlit_app_etl.py`
- Building lag features (`lag_24h`, `lag_168h`) that require recent historical load values

### Key Functions — `src/fetch_prepare_data.py`

#### `_get_index(filter_id, region, resolution) → list[int]`
Fetches the list of available weekly bucket timestamps from the index endpoint.

#### `_fetch_week(filter_id, timestamp_ms, region, resolution) → list`
Fetches the raw `[[ts_ms, value], ...]` series for a single weekly bucket.

#### `fetch_smard_netzlast(in_start_date, in_end_date, ...) → pd.DataFrame`
Main public function. Orchestrates the full data fetch:

1. Converts `YYYY-MM-DD` date strings to UTC millisecond boundaries.
2. Calls `_get_index()` to get all available bucket timestamps.
3. Filters buckets that overlap with the requested date range.
4. Iterates over relevant buckets, calling `_fetch_week()` for each with a configurable sleep between requests.
5. Concatenates all rows into a single DataFrame.
6. Clips to the exact requested time range.
7. Renames columns to match the training data schema (`timestamp → time`, `load_MWh → EnergyDemand`).
8. Optionally saves the result to a CSV file.

**Signature:**
```python
fetch_smard_netzlast(
    in_start_date: str,           # "YYYY-MM-DD", inclusive
    in_end_date: str,             # "YYYY-MM-DD", inclusive
    output_file: str | None = None,
    region: str = "DE",
    resolution: str = "hour",
    filter_id: int = 410,
    sleep: float = 0.3
) -> pd.DataFrame
```

**Returns:** `pd.DataFrame` with columns `['time', 'EnergyDemand']`

### Usage Example

```python
from src.fetch_prepare_data import fetch_smard_netzlast

df = fetch_smard_netzlast(
    in_start_date="2025-10-01",
    in_end_date="2026-05-10",
    output_file="../data/raw/netzlast_2025-10-01_to_2026-05-10.csv"
)
```

### Output Schema

| Column | Type | Description |
|--------|------|-------------|
| `time` | `datetime64[ns, UTC]` | Hourly timestamp (UTC) |
| `EnergyDemand` | `float64` | Realized grid load in MWh |

---

## Notes

- The API is **unofficial** and undocumented. The URL structure may change without notice.
- A `User-Agent` header is set to identify requests: `Mozilla/5.0 (compatible; smard-fetcher/1.0)`.
- Data for the current week may be incomplete (the last bucket is updated in near-real-time).
- SMARD displays times in **CET/CEST** on the web interface, but the stored timestamps are UTC milliseconds — this is handled correctly by the implementation.
- The processed output file is stored at `data/raw/netzlast_2025-10-01_to_2026-05-10_20.csv`.

---

## Ideas for Future Projects

The SMARD API provides a rich set of time series that can be combined with weather data and other signals for further modelling work.

### Electricity Price Forecasting
- **Target:** Day-ahead price DE/LU (`4169`) in EUR/MWh
- **Features:** Wind + solar generation (`4067`, `1225`, `4068`), gas generation (`4071`), total load (`410`), generation forecasts (`122`, `5097`), neighboring country prices (`4170`, `254`, `256`, ...)
- **Notes:** Prices show strong negative correlation with renewable generation (merit-order effect). Negative prices occur with high renewable surplus.

### Renewable Generation Forecasting
- **Target:** Wind onshore (`4067`) or solar PV (`4068`) in MWh
- **Features:** Open-Meteo wind speed / irradiance forecasts, installed capacity trends, season/hour features
- **Notes:** Use control-area regions (`50Hertz`, `Amprion`, `TenneT`, `TransnetBW`) to build regional models and aggregate.

### Residual Load Forecasting
- **Target:** Residual load (`4359`) = total demand minus wind and solar
- **Use case:** Signals how much dispatchable (gas, coal, hydro) capacity needs to be online — directly linked to price spikes.

### Generation Mix / CO₂ Intensity
- Combine all generation filter IDs (`1223`, `1224`, `1225`, `1226`, `4066`, `4067`, `4068`, `4069`, `4070`, `4071`) to reconstruct the hourly generation mix.
- Map each source to an emission factor to estimate grid CO₂ intensity (gCO₂/kWh).

### Cross-Border Price Arbitrage Analysis
- Compare DE/LU price (`4169`) against neighboring country prices (`252`–`262`, `4170`, ...).
- Identify hours where price spreads exceed interconnector costs → signals for import/export modelling.

### Fetching Multiple Series in Parallel

```python
from fetch_prepare_data import fetch_smard_netzlast  # reuse the same pattern

GENERATION_IDS = {
    'wind_onshore':  4067,
    'wind_offshore': 1225,
    'solar':         4068,
    'biomass':       4066,
    'hydro':         1226,
    'gas':           4071,
    'lignite':       1223,
    'hard_coal':     4069,
    'nuclear':       1224,
    'pumped_gen':    4070,
}

dfs = {}
for name, fid in GENERATION_IDS.items():
    df = fetch_smard_netzlast('2024-01-01', '2024-12-31', filter_id=fid)
    df = df.rename(columns={'EnergyDemand': name})
    dfs[name] = df.set_index('time')

df_generation = pd.concat(dfs.values(), axis=1)
```

### Fetching Day-Ahead Prices

```python
df_price = fetch_smard_netzlast('2024-01-01', '2024-12-31', filter_id=4169)
df_price = df_price.rename(columns={'EnergyDemand': 'price_eur_per_mwh'})
```

### Fetching Regional (Control Area) Data

```python
# Load for the 50Hertz control area only (north-east Germany)
df_50hertz = fetch_smard_netzlast('2024-01-01', '2024-12-31', region='50Hertz')
```

