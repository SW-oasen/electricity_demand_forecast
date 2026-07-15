"""
Streamlit web app — Germany hourly energy demand forecast (ETL pipeline).

Two sections:
  1. Vorhersage (morgen)  — predict the full next day using energy context
     loaded from the SQLite DB + live Open-Meteo weather forecast.
  2. Historischer Vergleich — features, actuals and SMARD forecast are all
     read from the pre-computed DB view (single SQL query, no live API calls).

Uses the ETL-trained LightGBM and XGBoost models from notebook 06.

Run with (from workspace root):
    streamlit run src/streamlit_app_etl.py
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datetime import date, timedelta, datetime, timezone

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import streamlit as st

FILTER_SMARD_FORECAST = 411  # Prognostizierter Stromverbrauch: Netzlast
MAX_RANGE_DAYS        = 365

from fetch_prepare_data import fetch_smard_netzlast
from etl import (
    update_database,
)
from forecast_service import (
    error_metrics,
    evaluate_historical_range,
    forecast_tomorrow,
    load_project_models,
)

ROOT_DIR = Path(__file__).resolve().parent.parent
WALK_FORWARD_CSV_DIR = ROOT_DIR / "data" / "walk_forward_predictions"

# ── page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Stromverbrauchsprognose DE — ETL",
    page_icon="⚡",
    layout="wide",
)

# ── ensure DB is current (runs once per process, ~seconds if already up to date)
@st.cache_resource(show_spinner="Datenbank wird aktualisiert …")
def _init_db():
    update_database()
    return True

_init_db()

# ── load ETL-trained models once (cached across sessions) ─────────────────────
@st.cache_resource
def load_models():
    return load_project_models(ROOT_DIR / "models")


models = load_models()


def _strip_tz(series: pd.Series) -> pd.Series:
    """Convert tz-aware Europe/Berlin timestamps to tz-naive for matplotlib."""
    return series.dt.tz_convert("Europe/Berlin").dt.tz_localize(None)


def _set_padded_ylim(ax: plt.Axes, df_plot: pd.DataFrame) -> None:
    plotted_values = pd.Series(
        df_plot[["Actual", "ML Prediction", "SMARD Forecast"]].to_numpy().ravel()
    ).dropna()
    if plotted_values.empty:
        return

    y_min = float(plotted_values.min())
    y_max = float(plotted_values.max())
    padding = (y_max - y_min) * 0.10
    if padding == 0:
        padding = max(abs(y_max) * 0.10, 1.0)
    ax.set_ylim(y_min - padding, y_max + padding)


def _render_metric_comparison(
    model_name: str,
    mae_ml: float,
    rmse_ml: float,
    ml_points: int,
    mae_smard: float | None = None,
    rmse_smard: float | None = None,
    smard_points: int | None = None,
) -> None:
    rows = [
        {
            "Series": f"ML Prediction ({model_name})",
            "MAE (MWh)": f"{mae_ml:,.0f}",
            "RMSE (MWh)": f"{rmse_ml:,.0f}",
            "Points": f"{ml_points:,}",
        }
    ]
    if mae_smard is not None and rmse_smard is not None and smard_points is not None:
        rows.append(
            {
                "Series": "SMARD official forecast",
                "MAE (MWh)": f"{mae_smard:,.0f}",
                "RMSE (MWh)": f"{rmse_smard:,.0f}",
                "Points": f"{smard_points:,}",
            }
        )

    st.markdown("### **Metrikvergleich**")
    df_metrics = pd.DataFrame(rows)
    html = df_metrics.to_html(index=False, border=0)
    st.markdown(
        """
        <style>
        .metric-comparison-table table {
            width: 100%;
            border-collapse: collapse;
            font-size: 16px;
        }
        .metric-comparison-table th,
        .metric-comparison-table td {
            padding: 10px 12px;
            text-align: left;
            border: 1px solid rgba(49, 51, 63, 0.2);
        }
        .metric-comparison-table th {
            background: rgba(240, 242, 246, 0.9);
            font-weight: 700;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(f'<div class="metric-comparison-table">{html}</div>', unsafe_allow_html=True)


# ── page header ────────────────────────────────────────────────────────────────
st.title("⚡ Stromverbrauchsprognose Deutschland — ETL Pipeline")
st.markdown(
    "Stündliche Vorhersage und Vergleich der deutschen Netzlast. "
    "Modelle und Features basieren auf der **SQLite-Datenbank** (`etl.py`)."
)

tab_future, tab_hist = st.tabs(["🔮 Vorhersage (morgen)", "📊 Historischer Vergleich"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Future Prediction (tomorrow, full day)
# Demand context → actual DB history before today, then recursive today forecast
# Weather forecast   → fetched live from Open-Meteo API
# ══════════════════════════════════════════════════════════════════════════════
with tab_future:
    st.markdown(
        "Vorhersage des Stromverbrauchs für den **nächsten Tag** (00:00–23:00 Europe/Berlin).  \n"
        "**Energie-Lag-Kontext** wird aus der DB geladen — kein SMARD-API-Abruf nötig."
    )

    now_berlin = datetime.now(tz=datetime.now(timezone.utc).astimezone().tzinfo)
    tomorrow   = date.today() + timedelta(days=1)

    col_info, col_ctrl = st.columns([2, 1])
    with col_info:
        st.markdown(
            f"**Aktuell (Berlin):** "
            f"{pd.Timestamp.now(tz='Europe/Berlin').strftime('%Y-%m-%d %H:%M')}"
        )
        st.markdown(f"**Vorhersagetag:** {tomorrow.isoformat()}")
    with col_ctrl:
        future_model = st.selectbox(
            "Modell", options=list(models.keys()), key="future_model"
        )

    if st.button("Predict for Tomorrow", type="primary", key="btn_future"):
        tomorrow_str = tomorrow.isoformat()

        # 1. SMARD official day-ahead forecast (filter 411) ────────────────────
        with st.spinner("SMARD-Prognose wird abgerufen …"):
            try:
                df_smard_fc = fetch_smard_netzlast(
                    tomorrow_str, tomorrow_str, filter_id=FILTER_SMARD_FORECAST
                )
            except Exception:
                df_smard_fc = pd.DataFrame(columns=["time", "EnergyDemand"])

        if df_smard_fc.empty:
            st.info("SMARD-Tagesprognose noch nicht veröffentlicht — nur ML-Vorhersage.")
        else:
            st.success(f"SMARD-Prognose: {len(df_smard_fc)} Stunden abgerufen.")

        # 2. Build features via ETL approach ───────────────────────────────────
        #    Demand: DB cutoff before today + recursive predictions for today
        #    Weather forecast: fetched live from Open-Meteo
        with st.spinner(f"Features werden aus DB vorbereitet für {tomorrow_str} …"):
            try:
                df_future, prediction_series = forecast_tomorrow(
                    models[future_model], tomorrow_str
                )
            except Exception as exc:
                st.error(f"Feature-Vorbereitung fehlgeschlagen: {exc}")
                st.stop()

        if df_future.empty:
            st.error("Keine Features zurückgegeben — API-Verbindung prüfen.")
            st.stop()

        # 3. Predict ───────────────────────────────────────────────────────────
        with st.spinner(f"{future_model} wird ausgeführt …"):
            preds = prediction_series.to_numpy()

        st.success(f"Vorhersage abgeschlossen — {tomorrow_str} ({future_model})")

        col_chart, col_table = st.columns([2.5, 1])

        with col_chart:
            fig, ax = plt.subplots(figsize=(10, 4))
            if not df_smard_fc.empty:
                ax.plot(
                    _strip_tz(df_smard_fc["time"]), df_smard_fc["EnergyDemand"],
                    color="mediumseagreen", linewidth=1.5, linestyle="-.",
                    label="SMARD offizielle Prognose",
                )
            ax.plot(
                _strip_tz(df_future["time"]), preds,
                linewidth=2, color="darkorange", linestyle="--",
                label=f"{future_model} Vorhersage",
            )
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
            ax.set_xlabel("Stunde (Europe/Berlin)")
            ax.set_ylabel("Netzlast (MWh)")
            ax.set_title(f"Stromverbrauchsprognose — {tomorrow_str}  [{future_model}]")
            ax.legend()
            ax.grid(True, alpha=0.3)
            fig.autofmt_xdate()
            plt.tight_layout()
            st.pyplot(fig)

        with col_table:
            df_result = df_future[["time"]].copy()
            df_result["ML (MWh)"] = preds.round(0).astype(int)
            if not df_smard_fc.empty:
                smard_idx = df_smard_fc.set_index("time")["EnergyDemand"]
                df_result["SMARD (MWh)"] = (
                    df_result["time"].map(smard_idx).round(0).astype("Int64")
                )
            df_result["Stunde (Berlin)"] = _strip_tz(df_result["time"]).dt.strftime(
                "%H:%M"
            )
            display_cols = ["Stunde (Berlin)", "ML (MWh)"]
            if "SMARD (MWh)" in df_result.columns:
                display_cols.append("SMARD (MWh)")
            st.dataframe(
                df_result[display_cols].reset_index(drop=True),
                use_container_width=True,
                height=600,
            )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Historical Comparison
# Source data comes from the DB; walk-forward predictions are cached as CSV.
# ══════════════════════════════════════════════════════════════════════════════
with tab_hist:
    st.info(
        "ML-Prognosen werden jetzt als Walk-Forward-Evaluation berechnet: "
        "Zuerst wird der fehlende Vortag prognostiziert, danach der Zieltag. "
        "Vollständige Tage werden als CSV zwischengespeichert; historische "
        "Wetterfeatures stammen weiterhin aus beobachtetem Wetter."
    )
    st.markdown(
        "Vergleich von tatsächlichem Verbrauch, SMARD-Prognose und ML-Vorhersage.  \n"
        "Quelldaten kommen aus der DB; ML-Prognosen aus dem CSV-Zwischenspeicher.  \n"
        "Maximaler Zeitraum: **1 Jahr**."
    )

    _default_to   = date.today() - timedelta(days=1)
    _default_from = _default_to - timedelta(days=6)
    _min_date     = date(2019, 1, 17)  # 168 actual hours required before D-1
    _max_date     = date.today() - timedelta(days=1)

    col1, col2, col3 = st.columns(3)
    with col1:
        date_from = st.date_input(
            "Von:",
            value=_default_from,
            min_value=_min_date,
            max_value=_max_date,
            key="hist_from",
        )
    with col2:
        date_to = st.date_input(
            "Bis:",
            value=_default_to,
            min_value=_min_date,
            max_value=_max_date,
            key="hist_to",
        )
    with col3:
        hist_model = st.selectbox(
            "Modell", options=list(models.keys()), key="hist_model"
        )

    # ── range validation ───────────────────────────────────────────────────────
    delta_days = (date_to - date_from).days

    if delta_days < 0:
        st.error('⚠ „Bis"-Datum muss nach dem „Von"-Datum liegen.')
    elif delta_days > MAX_RANGE_DAYS:
        st.warning(
            f"⚠ Gewählter Zeitraum: **{delta_days} Tage** — "
            f"Maximum sind **{MAX_RANGE_DAYS} Tage**. "
            "Bitte Auswahl einschränken."
        )
    else:
        st.success(f"Zeitraum: {delta_days + 1} Tag(e)  ✓")

        if st.button("Compare Prediction vs Actual", type="primary", key="btn_compare"):
            from_str = str(date_from)
            to_str   = str(date_to)
            with st.spinner(f"Walk-Forward wird geladen/berechnet: {from_str} → {to_str} …"):
                df_plot = evaluate_historical_range(
                    models[hist_model], hist_model, from_str, to_str,
                    WALK_FORWARD_CSV_DIR,
                )

            if df_plot.empty:
                st.error(f"Keine Daten in der DB für {from_str} → {to_str}.")
                st.stop()

            st.success(f"Vergleich abgeschlossen — {from_str} → {to_str} ({hist_model})")

            # 4. Plot ──────────────────────────────────────────────────────────
            df_plot.index = df_plot.index.tz_convert("Europe/Berlin").tz_localize(None)
            fig, ax = plt.subplots(figsize=(14, 5))
            ax.plot(
                df_plot.index, df_plot["Actual"],
                color="steelblue", linewidth=1.5,
                label="Tatsächlicher Verbrauch (DB)",
            )
            if df_plot["SMARD Forecast"].notna().any():
                ax.plot(
                    df_plot.index, df_plot["SMARD Forecast"],
                    color="mediumseagreen", linewidth=1.5, linestyle="-.",
                    label="SMARD offizielle Prognose (DB)",
                )
            ax.plot(
                df_plot.index, df_plot["ML Prediction"],
                color="darkorange", linewidth=1.5, linestyle="--",
                label=f"ML Vorhersage ({hist_model})",
            )
            locator = mdates.AutoDateLocator(minticks=6, maxticks=10)
            ax.xaxis.set_major_locator(locator)
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
            _set_padded_ylim(ax, df_plot)
            ax.set_xlabel("Datum / Uhrzeit (Europe/Berlin)")
            ax.set_ylabel("Netzlast (MWh)")
            ax.set_title(
                f"Tatsächlicher vs. vorhergesagter Verbrauch — "
                f"{from_str} bis {to_str}  [{hist_model}]"
            )
            ax.legend()
            ax.grid(True, alpha=0.3)
            fig.autofmt_xdate()
            plt.tight_layout()
            st.pyplot(fig)

            # 5. Metrics ───────────────────────────────────────────────────────
            ml_metrics = error_metrics(df_plot["Actual"], df_plot["ML Prediction"])

            if df_plot["SMARD Forecast"].notna().any():
                smard_metrics = error_metrics(df_plot["Actual"], df_plot["SMARD Forecast"])
                _render_metric_comparison(
                    model_name=hist_model,
                    mae_ml=ml_metrics["mae"],
                    rmse_ml=ml_metrics["rmse"],
                    ml_points=ml_metrics["points"],
                    mae_smard=smard_metrics["mae"],
                    rmse_smard=smard_metrics["rmse"],
                    smard_points=smard_metrics["points"],
                )
            else:
                _render_metric_comparison(
                    model_name=hist_model,
                    mae_ml=ml_metrics["mae"],
                    rmse_ml=ml_metrics["rmse"],
                    ml_points=ml_metrics["points"],
                )
