# PROJECT_CONTEXT — Electricity Demand Forecasting

> Projektübersicht, App-Nutzung und Business-Erkenntnisse: [README.md](README.md)

**Energy Analytics + Time Series + Wetter + Kalenderfeatures**

---

## ETL-Pipeline (`src/etl.py`)

### Überblick

`update_database()` ist idempotent: beim ersten Aufruf wird die DB erstellt und aus der Kaggle-CSV + SMARD-API + Open-Meteo-API befüllt; bei späteren Aufrufen werden nur fehlende Tage ergänzt.

```
db/energy_demand.db
├── energy_demand   (64 576 Zeilen, max: 2026-05-21)
├── weather         (64 729 Zeilen, max: 2026-05-22)
├── walk_forward_predictions  (persistierte historische ML-Prognosen)
└── energy_weather_combined  (VIEW — JOIN beider Tabellen)
```

### DB-Spalten (View `energy_weather_combined`)

```
time, energy_demand_mwh, smard_forecast_mwh, data_source,
year, hour, weekday, month, is_weekend, is_holiday, holiday_ratio,
is_workday, is_bridge_day, holiday_weight, is_pandemic_time,
energy_demand_lag_24h, energy_demand_lag_168h,
energy_demand_rolling_mean_24h, energy_demand_rolling_mean_168h,
apparent_temperature, rain, snowfall, wind_speed_10m, shortwave_radiation,
apparent_temperature_lag_24h, apparent_temperature_rolling_mean_24h,
shortwave_radiation_0m_lag_24h, shortwave_radiation_0m_rolling_mean_24h,
heating_degree, cooling_degree
```

### Wichtige Konstanten

| Konstante | Wert | Bedeutung |
|---|---|---|
| `ENERGY_CONTEXT_ROWS` | 168 | Kontext-Zeilen für korrekte Lag-Berechnung an der Naht |
| `WEATHER_CONTEXT_ROWS` | 24 | Kontext-Zeilen für Wetter-Lags |
| `KAGGLE_END_DATE` | 2025-09-30 | Letzter Kaggle-Datentag |
| `SMARD_START_DATE` | 2025-10-01 | Erster SMARD-API-Datentag |

### Spaltenumbenennung: Legacy → ETL

Die DB verwendet snake_case statt PascalCase:

| Legacy (`fetch_prepare_data.py`) | ETL DB-Schema |
|---|---|
| `EnergyDemand` | `energy_demand_mwh` |
| `EnergyDemand_lag_24h` | `energy_demand_lag_24h` |
| `EnergyDemand_lag_168h` | `energy_demand_lag_168h` |
| `EnergyDemand_rolling_mean_24h` | `energy_demand_rolling_mean_24h` |
| `EnergyDemand_rolling_mean_168h` | `energy_demand_rolling_mean_168h` |

### Öffentliche Read-Helfer

| Funktion | Beschreibung |
|---|---|
| `get_connection(db_path)` | SQLite-Verbindung |
| `load_energy_data(conn)` | Energietabelle als DataFrame |
| `load_weather_data(conn)` | Wettertabelle als DataFrame |
| `load_combined_data(conn, start_date, end_date)` | View mit optionalem Datumsfilter |
| `prepare_for_prediction_tomorrow_etl(date, model, db_path)` | Prognostiziert zuerst rekursiv den fehlenden heutigen Tag und erzeugt daraus die Feature-Matrix für morgen |
| `prediction_store.upsert_predictions(conn, predictions)` | CSV-Ergebnisse idempotent in SQLite speichern |
| `prediction_store.load_predictions(...)` | Persistierte Walk-Forward-Prognosen nach Modell, Modus und Zeitraum laden |

---

## Datenquellen

### 1. Stromverbrauch

**Europe Electricity Load (Hourly, 2019–2025)**  
Quelle: Kaggle, basierend auf ENTSO-E Transparency Platform.  
([Kaggle](https://www.kaggle.com/datasets/dsersun/europe-electricity-load-hourly-20192025))

Verwendete Spalten:
- `DateUTC`
- `CountryCode` (gefiltert auf `DE`)
- `Value` → umbenannt in `EnergyDemand`

Lizenzhinweis:
- ENTSO-E attribution
- CC BY-SA 4.0

---

### 2. Aktuelle Stromverbrauchsdaten (ab 2025-10-01)

**SMARD Chart Data API** (Bundesnetzagentur)  
([SMARD](https://www.smard.de/home))

Filter-ID 410: Realisierter Stromverbrauch – Netzlast  
Filter-ID 411: Prognostizierter Stromverbrauch – Netzlast (offizielle SMARD-Tagesvorhersage)  
Programmatisch abgerufen über `fetch_smard_netzlast(filter_id=...)` in `src/fetch_prepare_data.py`.  
Filter 411 wird in Streamlit und Notebook 08 als Referenz-Benchmark verwendet.

> **Hinweis Timezone**: SMARD liefert Timestamps in CET/CEST. Die Kaggle-Quelldaten enthalten ebenfalls UTC-Zeitstempel (`DateUTC`). Open-Meteo gibt mit `&timezone=auto` lokale Zeit zurück (CEST im Sommer +2h, CET im Winter +1h) — potenzielle 1h-Verschiebung zwischen Wetter- und Verbrauchsdaten im Sommer.


> SMARD JSON API Dokumentation: /documents/smard_api.md

---

### 3. Historische Wetterdaten

**Open-Meteo Historical Weather API**  
([Open Meteo](https://open-meteo.com/en/docs/historical-weather-api))

API-Endpunkt:
```
https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}&start_date=2019-01-01&end_date=2025-09-30&hourly={variables}&timezone=auto
```

Verwendete Variablen:
- `apparent_temperature`
- `rain`
- `snowfall`
- `wind_speed_10m`
- `shortwave_radiation`

Aggregation über Top-5-Städte Deutschland (gewichtet nach Stadtbevölkerung):

| Stadt | Einwohner |
|---|---|
| Berlin | 3,69 Mio |
| Hamburg | 1,86 Mio |
| München | 1,51 Mio |
| Köln | 1,02 Mio |
| Frankfurt a.M. | 0,76 Mio |

> open-meteo API Dokumentation: /documents/open-meteo_api.md

---

### 4. Feiertage

**python-holidays**  
([holidays.readthedocs.io](https://holidays.readthedocs.io/))

Schulferien werden bundeslandweise über die Ferien-API
(`https://ferien-api.de/api/v1/holidays/{state}/{year}`) geladen und dauerhaft in
`data/cache/school_holidays.json` zwischengespeichert. Die Bevölkerungsgewichte
stehen zentral in `src/config.py` und müssen gemeinsam aktualisiert werden.

Features:
- `is_holiday` — nationaler/regionaler Feiertag (0/1)
- `holiday_ratio` — Bevölkerungsanteil der Bundesländer mit Feiertag (0–1)
- `is_school_holiday` — 1, wenn in mindestens einem Bundesland Schulferien sind
- `school_holiday_ratio` — Bevölkerungsanteil der Bundesländer mit Schulferien (0–1)

---

## Feature Engineering

### Zeitfeatures
| Feature | Beschreibung |
|---|---|
| `hour` | Stunde des Tages (0–23) |
| `weekday` | Wochentag (0=Mo, 6=So) |
| `month` | Monat (1–12) |
| `is_weekend` | 1 wenn Sa/So |

> Empfehlung für Weiterentwicklung: zyklische Kodierung (`sin_hour`, `cos_hour`, `sin_month`, `cos_month`) statt Integer, um Periodizität korrekt abzubilden.

### Kalenderfeatures
| Feature | Beschreibung |
|---|---|
| `is_holiday` | Feiertag ja/nein |
| `holiday_ratio` | Bevölkerungsanteil der Bundesländer mit Feiertag (0–1) |
| `is_school_holiday` | 1, wenn mindestens ein Bundesland Schulferien hat |
| `school_holiday_ratio` | Bevölkerungsanteil der Bundesländer mit Schulferien (0–1) |
| `is_workday` | 1 wenn Werktag und kein Feiertag (direktes Signal für Hochlasttage) |
| `is_bridge_day` | 1 wenn Werktag eingeklemmt zwischen Feiertag und Wochenende |
| `holiday_weight` | kombiniertes Signal: `max(holiday_ratio, is_weekend × 0.5)` |
| `is_pandemic_time` | 2020-03-01 bis 2021-12-31 |

### Wetterfeatures
| Feature | Beschreibung |
|---|---|
| `apparent_temperature` | gefühlte Temperatur |
| `rain`, `snowfall` | Niederschlag |
| `wind_speed_10m` | Windgeschwindigkeit |
| `shortwave_radiation` | Solarstrahlung |
| `apparent_temperature_lag_24h` | Temperatur vor 24h |
| `apparent_temperature_rolling_mean_24h` | 24h-Rollmittel Temperatur |
| `shortwave_radiation_0m_lag_24h` | Solarstrahlung vor 24h |
| `shortwave_radiation_0m_rolling_mean_24h` | 24h-Rollmittel Solarstrahlung |
| `heating_degree` | `max(0, 18 - apparent_temperature)` |
| `cooling_degree` | `max(0, apparent_temperature - 25)` |

* Gewichtete Wetteraggregation nach Stadtbevölkerung

### Lag-Features Stromverbrauch (entscheidend für Saisonalität)

**Legacy-Benennung** (in älteren EDA-Notebooks):

| Feature | Beschreibung |
|---|---|
| `EnergyDemand_lag_24h` | Verbrauch vor 24h (selbe Stunde gestern) |
| `EnergyDemand_lag_168h` | Verbrauch vor 168h (selbe Stunde letzte Woche) |
| `EnergyDemand_rolling_mean_24h` | 24h-Rollmittel Verbrauch (shift(24)) |
| `EnergyDemand_rolling_mean_168h` | 168h-Rollmittel Verbrauch (shift(24)) |

**ETL-Benennung** (in DB-Schema, `etl.py`, Notebooks 06–08, `streamlit_app_etl.py`):

| Feature | Beschreibung |
|---|---|
| `energy_demand_lag_24h` | identisch, DB snake_case |
| `energy_demand_lag_168h` | identisch, DB snake_case |
| `energy_demand_rolling_mean_24h` | identisch, DB snake_case |
| `energy_demand_rolling_mean_168h` | identisch, DB snake_case |

> `EnergyDemand_lag_8760h` und `EnergyDemand_rolling_mean_8760h` wurden nach Feature-Importance-Analyse entfernt (geringer Beitrag, erzwang Wegfall des Jahres 2019).

> **Hinweis Rolling-Features**: `shift(24)` stellt sicher, dass das Fenster auf gestern ausgerichtet ist (T-24h bis T-(24+n-1)h). Kein Datenleck durch unmittelbar vorangehende Stunden; konsistente temporale Ausrichtung mit `lag_24h`.

---

## Train/Test Split

### Walk-Forward-Evaluation

Die historische Auswertung der ETL-App verwendet eine rekursive 48-Stunden-
Simulation. Für einen Zieltag `D` gelten nur Verbrauchswerte vor `D-1` als
bekannt. Die Stunden von `D-1` werden zuerst prognostiziert und anschließend als
Lastverlauf für die Prognose von `D` verwendet. Bewertet wird ausschließlich der
Zieltag `D`.

Vollständig berechnete Tage werden je Modell unter
`data/walk_forward_predictions/` als CSV zwischengespeichert. Ein erneuter Lauf
überspringt bereits vollständige Tage. Die CSV-Dateien bleiben erhalten, bis die
Ergebnisse erfolgreich in `walk_forward_predictions` geschrieben und abgeglichen
wurden. Auch nach erfolgreichem Abgleich werden die CSV-Dateien nicht automatisch
gelöscht.

Der Evaluationsmodus heißt `walk_forward_48h_actual_weather`: Das Last-Leakage
ist ausgeschlossen, historische Wetterfeatures stammen derzeit jedoch aus
beobachtetem Wetter und nicht aus archivierten Wetterprognosen.

| Split | Zeitraum | Verwendung |
|---|---|---|
| Training | 2019–2024 | Modelltraining |
| Test | 2025 | Finale Evaluation |

Zeitbasierter Split — kein zufälliges Mischen. Cross-Validation mit `TimeSeriesSplit` (kein Standard-k-Fold, da Datenleck durch Lag-Features).

---

## Modelle

### Preprocessing
Für distanzbasierte Modelle (Linear Regression, SVR): `StandardScaler` + `OneHotEncoder` über `ColumnTransformer`.  
Für baumbasierte Modelle (Random Forest, XGBoost, LightGBM): kein Preprocessing nötig.

### Evaluierte Modelle

| Modell | Preprocessing | Anmerkung |
|---|---|---|
| Linear Regression | StandardScaler + OHE | Schwache Baseline |
| Random Forest | keines | Beste Performance mit Lag-Features |
| SVR (rbf) | StandardScaler + OHE | Nicht geeignet für ~46k Zeilen; nur auf 10k-Subset getestet |
| XGBoost | keines | Gute Performance |
| LightGBM | keines | Vergleichbar mit XGBoost, schneller |
| SARIMAX | — | Auf täglicher Frequenz getestet (zu langsam auf Stundenbasis) |

### Hyperparameter-Tuning

Schritt 1: `RandomizedSearchCV` mit `TimeSeriesSplit(n_splits=5)` — respektiert zeitliche Reihenfolge.  
Schritt 2: Bayesian Optimization mit **Optuna** (`TPESampler`, 100 Trials) 
- bestes LightGBM-Modell gespeichert als `best_lgbm_model_bayesian.pkl`.  
- bestes XGBoost-Modell gespeichert als `best_xgb_model_bayesian.pkl`.  

Scoring: `neg_mean_absolute_error` (MAE praxisrelevanter als R² für Lastvorhersage).

### Bewertungsmetriken

- **MAE** — mittlerer absoluter Fehler (primäre Metrik)
- **RMSE** — Wurzel mittlerer quadratischer Fehler (gleiche Skala wie MAE, stärker gewichtete Ausreißer)
- **R²** — Erklärte Varianz

---

## Technische Erkenntnisse & Limitierungen

- **Demand-Lag-Features** (`lag_168h`, `lag_24h`) sind die stärksten Prädiktoren — deutlich wirksamer als Kalender-Integer-Features allein
- Baumbasierte Modelle übertreffen lineare Modelle deutlich; **SVR** skaliert schlecht ($O(n^2)$–$O(n^3)$) auf den ~46k-Zeilen-Datensatz
- Standard-k-Fold CV führt bei Lag-Features zu Datenleck → `TimeSeriesSplit` verwenden
- Zyklische Kodierung (`sin`/`cos`) für `hour` und `month` empfohlen (Integer bilden keine Periodikität ab)
- Industrieller Verbrauch (~40% der Netzlast) wird durch Wetterdaten nicht abgebildet — größte verbleibende Fehlerquelle
- **Timezone-Problem (behoben in ETL-App)**: Matplotlib konvertiert tz-aware Timestamps intern nach UTC beim Plotten. In `streamlit_app_etl.py` werden beide Serien (ML + SMARD) über `_strip_tz()` zu tz-naive Europe/Berlin normiert, bevor sie an matplotlib übergeben werden.
- **pandas 3.0 Mixed-Timezone-Bug (behoben)**: `pd.to_datetime(col)` wirft `ValueError: Mixed timezones` bei Spalten mit gemischten UTC-Offsets (`+0100`/`+0200`). Fix: `pd.to_datetime(col, utc=True)` in `_parse_time_col` in `etl.py`.
- **Operative Morgenprognose**: Istwerte werden strikt vor D-1 abgeschnitten. D-1 wird stündlich rekursiv prognostiziert und als Kontext für Zieltag D verwendet.

---

## Notebook-Übersicht

| Notebook | Inhalt |
|---|---|
| `01_eda_energy.ipynb` | EDA Stromverbrauch, Zeitreihenzerlegung, Saisonalität |
| `02_eda_weather.ipynb` | EDA Wetterdaten je Stadt |
| `03_eda_energy_weather.ipynb` | Feature Engineering, kombinierter Datensatz, Korrelationsanalyse |
| `04_base_models_eval.ipynb` | Modelltraining, Tuning, Lernkurven, Prediction vs. Actual |
| `05_scrape_eda_smard.ipynb` | Historische SMARD-Analyse und Untersuchung des Prognoseverhaltens |
| `06_ml_pipeline_etl.ipynb` | ETL-Training und Walk-Forward-Evaluation von LightGBM und XGBoost |
| `07_feature_importances.ipynb` | Feature Importances, Ferienfeature-Analyse und Befunde zu konservativen Prognosen |
| `08_interactive_prediction_etl.ipynb` | Rekursive Morgenprognose und historischer Walk-Forward-Vergleich mit CSV-/DB-Persistenz |

### Notebook 08 — Implementierungsdetails

**Teil 1 — Tagesvorhersage (morgen)**

- `prepare_for_prediction_tomorrow_etl(tomorrow_str, model)` baut die Feature-Matrix:
  - Energie-Lag-Kontext: letzte 168 DB-Zeilen vor Beginn des heutigen Tages
  - der ausgewählte Modellstand prognostiziert den heutigen Tag stündlich rekursiv
  - die prognostizierten heutigen Lastwerte bilden Lags und Rolling Features für morgen
  - Wetter-Forecast für heute und morgen: live von Open-Meteo API
  - Spaltennamen entsprechen direkt dem ETL-DB-Schema — kein Umbenennen nötig
- SMARD-Tagesprognose (Filter 411) wird parallel per API abgerufen und als Vergleichslinie eingeblendet (sofern bereits veröffentlicht)
- `_render_future`: Liniengrafik (2.5-Anteile) + stündliche Wertetabelle (1-Anteil) nebeneinander
- `_strip_tz(series)`: konvertiert tz-aware Timestamps nach tz-naiver Europe/Berlin-Zeit, damit matplotlib keine UTC-Verschiebung erzeugt

**Teil 2 — Historischer Vergleich**

- Quelldaten inklusive Warm-up-Kontext werden aus SQLite geladen; fehlende Zieltage werden per Walk-Forward berechnet
- Ergebnisse werden tageweise als CSV gesichert, in SQLite persistiert und beim nächsten Abruf wiederverwendet
- Zeitraum frei wählbar (min. 2019-01-08, max. 1 Jahr); Live-Validierung über `_validate_range()` sperrt den Compare-Button bei ungültiger Auswahl
- X-Achsen-Format passt sich automatisch an den gewählten Zeitraum an (≤3 Tage: `%m-%d %H:%M`, ≤31 Tage: `%Y-%m-%d`, sonst: `%Y-%m`)
- Metriktabelle (MAE, RMSE, Datenpunkte) für ML-Prognose **und** SMARD-Prognose nebeneinander

## Source Code (`/src`)

| Datei | Inhalt |
|---|---|
| `fetch_prepare_data.py` | Kaggle/SMARD/Open-Meteo Datenabruf und gemeinsames Feature Engineering |
| `train_model_predict.py` | Modelltraining, Hyperparameter-Tuning, Modell-Persistenz |
| `etl.py` | ETL-Pipeline: SQLite-DB erstellen/aktualisieren; Read-Helfer (`load_combined_data`, `prepare_for_prediction_tomorrow_etl`) |
| `walk_forward.py` | Leakage-sichere rekursive Feature- und Evaluationslogik sowie CSV-Checkpoint |
| `prediction_store.py` | SQLite-Schema, Upsert und Abfragen für persistierte Prognosen |
| `forecast_service.py` | Gemeinsame Orchestrierung für Modelle, Morgenprognose, historische Evaluation und Metriken |
| `streamlit_app_etl.py` | Rekursive Morgenprognose und historischer Walk-Forward-Vergleich |

---

## Links

- [Europe Electricity Load (Hourly, 2019–2025) – Kaggle](https://www.kaggle.com/datasets/dsersun/europe-electricity-load-hourly-20192025)
- [SMARD Marktdaten - Bundesnetzagentur](https://www.smard.de/page/home/marktdaten/)
- [Open-Meteo Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api)
- [Open-Meteo Forecast API](https://open-meteo.com/en/docs)
- [python-holidays](https://holidays.readthedocs.io/)
- [Optuna – Hyperparameter Optimization Framework](https://optuna.readthedocs.io/)
- [Deutsche Schulferien API](https://ferien-api.maxleistner.de/)

## GitHub

- https://github.com/SW-oasen/electricity_demand_forecast


## Implementierungshinweie

### Zeitdiskrepanzen aus verschiedenen Quellen

- SMARD ts_ms
pd.to_datetime(df["ts_ms"], unit="ms", utc=True).dt.tz_convert("Europe/Berlin")

- Open-Meteo archive mit timezone=UTC
pd.to_datetime(df["time"], utc=True).dt.tz_convert("Europe/Berlin")

- Open-Meteo forecast mit timezone=Europe/Berlin
pd.to_datetime(df["time"]).dt.tz_localize("Europe/Berlin")

### Vorhersage-Lag-Features

Historische Evaluation und operative Morgenprognose verwenden denselben positionsbasierten rekursiven Feature-Builder aus `walk_forward.py`. Dadurch stimmen Lag- und Rolling-Definitionen mit dem Training überein und DST-Tage mit 23 beziehungsweise 25 Stunden bleiben korrekt.

---

## Projektstatus

### Abgeschlossen

- [x] EDA Stromverbrauch Deutschland (Notebook 01)
- [x] EDA Wetterdaten (Notebook 02)
- [x] Feature Engineering & EDA kombinierter Datensatz (Notebook 03)
- [x] Baseline- und ML-Modell-Evaluation (Notebook 04)
- [x] SMARD-Exploration und Prognoseanalyse (Notebook 05)
- [x] ETL-Modelltraining und Walk-Forward-Evaluation von LightGBM und XGBoost (Notebook 06)
- [x] Feature Importances und Ferienfeature-Analyse (Notebook 07)
- [x] Python Source Refactoring /src (`fetch_prepare_data.py`, `train_model_predict.py`)
- [x] Vollständige ML-Pipeline: Training, Tuning, Persistenz (Notebook 06)
- [x] Bayesian Hyperparameter-Optimierung mit Optuna auf AI PC
- [x] Rolling-Features auf `shift(1).rolling(...)` vereinheitlicht
- [x] **ETL-Pipeline** (`src/etl.py`): SQLite-DB mit inkrementellem Update; alle Features vorberechnet; Kaggle-CSV + SMARD-API + Open-Meteo-API als Quellen
- [x] **ETL ML-Pipeline** (Notebook 06): Training von LightGBM und XGBoost auf DB-Daten; Modelle mit `_etl`-Suffix gespeichert
- [x] Konservative Quantilmodelle analysiert und aus der produktiven Pipeline entfernt: geringere Unterschätzungsrate, aber deutlich schlechtere MAE und starker positiver Bias; SMARD zeigte im untersuchten Zeitraum ebenfalls keine konservative Coverage
- [x] **ETL Interaktive Vorhersage** (Notebook 08): rekursive Morgenprognose und historischer Walk-Forward-Vergleich
- [x] **ETL Streamlit App** (`src/streamlit_app_etl.py`): DB-basierte Vorhersage-App; SMARD-Zeitversatz-Bug behoben (`_strip_tz` auf beide Serien)
- [x] Bug behoben: `_parse_time_col` in `etl.py` — `pd.to_datetime(..., utc=True)` für gemischte UTC-Offsets (pandas 3.0)

### Offen

- [ ] Strompreis-Vorhersage — separates Folgeprojekt
