# Electricity Demand Forecasting — Deutschland

Portfolio-Projekt zur stündlichen Vorhersage des deutschen Stromverbrauchs auf Basis von Wetter-, Kalender- und historischen Verbrauchsdaten.

> **Technische Details** (Feature-Engineering, Implementierung, Modellparameter, Projektstatus): [PROJECT_CONTEXT.md](PROJECT_CONTEXT.md)

---

## Was macht dieses Projekt?

- **Tagesvorhersage**: Stündliche ML-Prognose für den nächsten Tag (00:00–23:00 Europe/Berlin), verglichen mit der offiziellen SMARD-Prognose
- **Historischer Vergleich**: Tatsächlicher Verbrauch vs. SMARD-Prognose vs. ML-Vorhersage für einen frei wählbaren Zeitraum (bis 1 Jahr) — inkl. MAE und RMSE
- **ETL-Pipeline**: Inkrementelles SQLite-Datenbank-Update (`etl.py`), das Kaggle-CSV, SMARD-API und Open-Meteo-API kombiniert und alle Features vorberechnet — Basis für schnelle historische Abfragen ohne Live-API-Aufrufe
- **Modelle**: LightGBM und XGBoost — DB-basiert bis 2025-09-30 trainiert und ab 2025-10-01 per Walk-forward evaluiert

---

## Quick Start – Streamlit App

Voraussetzung: virtuelle Umgebung aktiviert, trainierte Modelle liegen unter `models/`.

### ETL App

```powershell
cd C:\Pfad\zum\Projekt\workspace_energy_demand
.venv\Scripts\Activate.ps1
python -m streamlit run src/streamlit_app_etl.py
```

Browser öffnet sich unter **http://localhost:8501** (oder Port 8502 bei gleichzeitigem Betrieb).

Beim ersten Start führt die App `update_database()` aus und befüllt/aktualisiert die SQLite-DB — danach idempotent (Sekunden).

| Tab | Funktion |
|---|---|
| Vorhersage (morgen) | Energie-Lag-Kontext aus DB + Open-Meteo Wetter-Forecast → ML-Prognose für morgen inkl. SMARD-Vergleichslinie |
| Historischer Vergleich | DB-Lastdaten + archivierte 48h-Wetterprognose → Walk-forward → CSV-Checkpoint; max. 1 Jahr, mit MAE + RMSE |

---

## Interaktive Notebooks

Als interaktive Oberfläche steht zusätzlich Notebook 08 in `notebook/` bereit:

| Notebook | Art | Beschreibung |
|---|---|---|
| `08_interactive_prediction_etl.ipynb` | ETL | Rekursive Morgenprognose und historischer Walk-Forward-Vergleich mit CSV-Checkpoints |

### Notebook 08 — Aufbau

**Teil 1 — Tagesvorhersage (morgen)**
- Energie-Lag-Kontext wird bis vor Beginn des heutigen Tages aus SQLite geladen
- Der heutige Tag wird rekursiv prognostiziert und als Kontext für morgen verwendet
- Wetter-Forecast wird live von der Open-Meteo API abgerufen
- Spaltennamen entsprechen bereits dem ETL-DB-Schema — kein Umbenennen nötig
- Ergebnis: Liniengrafik + stündliche Wertetabelle nebeneinander; SMARD-Tagesprognose als Vergleichslinie (sofern veröffentlicht)

**Teil 2 — Historischer Vergleich (Actual vs. SMARD vs. ML)**
- Quelldaten werden aus SQLite geladen; fehlende Walk-Forward-Tage werden berechnet
- Die Datumsauswahl endet automatisch am letzten vollständigen Isttag der Lasttabelle; angebrochene Tage sind nicht auswählbar.
- Der kombinierte DB-View verwendet einen `LEFT JOIN`: Lastzeilen bleiben erhalten, auch wenn beobachtetes Wetter am D-1/D-Horizont noch fehlt.
- Lastwerte werden rekursiv ohne Zukunftswissen erzeugt. Primär verwenden D-1 und Zieltag denselben, am Prognosezeitpunkt sicher verfügbaren archivierten ECMWF-Lauf.
- Bei unvollständigen oder von der Single-Runs-API nicht archivierten ECMWF-Läufen (`400/404`) wird auf die ebenfalls leakage-sichere Open-Meteo-Best-Match-Prognose mit festem 48-Stunden-Vorlauf zurückgegriffen. Fehlt dabei nur ein Standort, werden die verfügbaren Städte populationsgewichtet neu normiert; vollständig fehlende Stunden werden nicht imputiert und führen zum Abbruch.
- Wetterprognosen werden unter `data/cache/openmeteo_single_runs/` dauerhaft zwischengespeichert; der erste Abruf eines Zeitraums benötigt API-Zugriff
- Jeder vollständige Tag wird sofort als CSV gesichert. Solange die Walk-forward-Logik validiert wird, findet bewusst kein automatischer Import der Prognosen nach SQLite statt.
- Auswählbarer Zeitraum bis maximal 1 Jahr; Live-Validierung verhindert ungültige Auswahl
- Metriktabelle (MAE, RMSE, Datenpunkte) für ML-Prognose **und** SMARD-Prognose im Vergleich

### Reproduzierbarer Projektstart

Da Rohdaten, Datenbank und Modellartefakte nicht im Git-Repository liegen:

Nach Änderungen an Kalenderfeatures zuerst die ETL-Pipeline ausführen. Sie migriert
das Schema und führt den versionierten historischen Backfill aus. Anschließend die
Modelle mit `06_ml_pipeline_etl.ipynb` neu trainieren. Bereits vorhandene
Walk-Forward-CSV-Checkpoints stammen noch vom alten Modell und müssen vor einer
neuen Evaluation bewusst entfernt werden.

1. Rohdaten unter `data/raw/` bereitstellen.
2. `python -m src.etl` beziehungsweise `python src/etl.py` ausführen, um die DB aufzubauen.
3. Notebook 06 ausführen, um LightGBM und XGBoost unter `models/` zu erzeugen.
4. Die Streamlit-App mit `python -m streamlit run src/streamlit_app_etl.py` starten.

### Code-Architektur

| Modul | Verantwortung |
|---|---|
| `etl.py` | Quelldaten und Feature-DB |
| `walk_forward.py` | Rekursive Prognose- und Evaluationslogik |
| `historical_weather_forecast.py` | Archivierte, leakage-sichere Wetterprognosen für historische D-1/D-Horizonte |
| `util/openmeteo_client.py` | Open-Meteo Archive-, Single-Run- und Previous-Runs-Client mit lokalem Cache |
| `prediction_store.py` | Persistenz der Walk-Forward-Ergebnisse |
| `forecast_service.py` | Gemeinsame Anwendungslogik für Streamlit und Notebook 08 |
| `streamlit_app_etl.py` | Darstellung und Benutzerinteraktion |


---

## Datenquellen

| Quelle | Inhalt | Lizenz |
|---|---|---|
| [Kaggle / ENTSO-E](https://www.kaggle.com/datasets/dsersun/europe-electricity-load-hourly-20192025) | Stündlicher Stromverbrauch Europa 2019–2025 | CC BY-SA 4.0 |
| [SMARD (Bundesnetzagentur)](https://www.smard.de/home) | Realisierter + prognostizierter Verbrauch (Filter 410 / 411) | — |
| [Open-Meteo](https://open-meteo.com/en/docs) | Stündliche Wetterdaten 5 Städte DE (Archiv + Forecast) | CC BY 4.0 |
| [python-holidays](https://holidays.readthedocs.io/) | Deutsche Feiertage, alle 16 Bundesländer | — |
| [Ferien-API](https://ferien-api.de/) | Schulferien aller Bundesländer; lokal gecacht | — |


### Electricity Market Data
Quelle: Bundesnetzagentur | SMARD.de  
https://www.smard.de/

SMARD Daten ist lizensiert unter CC BY 4.0.

### Weather Data
Quelle: Open-Meteo  
https://open-meteo.com/

Wetterdaten ist lizensiert unter CC BY 4.0.

Die orginal Daten sind bereinigt, aggregiert und transformiert für Machine Learning und Visualisierung. 

---

## Erkenntnisse

- **Demand-Lag-Features** (`lag_168h`, `lag_24h`) sind die stärksten Prädiktoren — deutlich wirksamer als Kalender-Integer-Features allein
- Baumbasierte Modelle (LightGBM, XGBoost) übertreffen lineare Modelle klar
- Industrieller Verbrauch (~40% der Netzlast) wird durch Wetterdaten nicht abgebildet — größte verbleibende Fehlerquelle
- Feiertags- und Schulferienquoten werden nach der Bevölkerung der betroffenen Bundesländer gewichtet; Brückentage und `holiday_weight` ergänzen diese Ausnahmetage
- Die offizielle SMARD-Prognose (Filter 411) dient als starker Benchmark. Eine Analyse in Notebook 07 zeigte jedoch keine konservative Prognosestrategie: SMARD unterschätzte im untersuchten Zeitraum häufiger, als es überschätzte. Separate Quantilmodelle wurden deshalb aus der produktiven Pipeline entfernt, da ihre geringere Unterschätzungsrate mit deutlich schlechterer MAE und starkem positivem Bias erkauft wurde.

---

## Potenzielle Erweiterungen

- ENTSO-E Day-Ahead-Preise als Feature
- Industrieproduktionsindex (Destatis, monatlich)
- Operative Morgenprognose und historischer Backtest auf dasselbe explizite Wettermodell vereinheitlichen
- Mehrere Länder wegen besonderem Klima (FI – Finnland, ES – Spanien)
- 7-Tage-Forecast (iterative/rekursive Vorhersage)

### Folgeprojekt

- Strompreis-Vorhersage: `Abhängig von PV und Wind-Energie Produktion, Gas- und Kohlepreise (für konventionelle Erzeugung als Ergänzung zu erneuerbarer Energie, und Stromverbrauch. Parallelle und stacked Prognose.`

---

## Links

- [Europe Electricity Load (Hourly, 2019–2025) – Kaggle](https://www.kaggle.com/datasets/dsersun/europe-electricity-load-hourly-20192025)
- [SMARD Marktdaten - Bundesnetzagentur](https://www.smard.de/page/home/marktdaten/)
- [Open-Meteo Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api)
- [python-holidays](https://holidays.readthedocs.io/)
- [Deutsche Schulferien API](https://ferien-api.de/)

## GitHub

- https://github.com/SW-oasen/electricity_demand_forecast
