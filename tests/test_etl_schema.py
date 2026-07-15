import sqlite3
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.config import DE_STATE_CODES
from src.etl import backfill_calendar_features, create_database


class EtlSchemaTests(unittest.TestCase):
    def test_existing_database_is_migrated_and_view_is_refreshed(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "energy.db"
            conn = sqlite3.connect(db_path)
            conn.execute("CREATE TABLE energy_demand (time TEXT PRIMARY KEY)")
            conn.execute("CREATE TABLE weather (time TEXT PRIMARY KEY)")
            conn.execute(
                "CREATE VIEW energy_weather_combined AS "
                "SELECT e.time FROM energy_demand e JOIN weather w ON e.time = w.time"
            )
            conn.close()

            migrated = create_database(db_path)
            try:
                columns = {
                    row[1]
                    for row in migrated.execute("PRAGMA table_info(energy_demand)")
                }
                self.assertIn("is_school_holiday", columns)
                self.assertIn("school_holiday_ratio", columns)

                view_sql = migrated.execute(
                    "SELECT sql FROM sqlite_master "
                    "WHERE type = 'view' AND name = 'energy_weather_combined'"
                ).fetchone()[0]
                self.assertIn("school_holiday_ratio", view_sql)
                self.assertIn("LEFT JOIN weather", view_sql)
            finally:
                migrated.close()

    def test_combined_view_preserves_energy_without_weather(self):
        with tempfile.TemporaryDirectory() as tmp:
            conn = create_database(Path(tmp) / "energy.db")
            try:
                conn.execute(
                    "INSERT INTO energy_demand (time, energy_demand_mwh) "
                    "VALUES (?, ?)",
                    ("2025-07-14T23:00:00+0200", 42.0),
                )
                conn.commit()

                row = conn.execute(
                    "SELECT energy_demand_mwh, apparent_temperature "
                    "FROM energy_weather_combined"
                ).fetchone()

                self.assertEqual(row[0], 42.0)
                self.assertIsNone(row[1])
            finally:
                conn.close()

    def test_calendar_backfill_is_versioned_and_updates_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            conn = create_database(Path(tmp) / "energy.db")
            try:
                conn.execute(
                    """
                    INSERT INTO energy_demand
                        (time, holiday_ratio, is_school_holiday, school_holiday_ratio)
                    VALUES (?, ?, ?, ?)
                    """,
                    ("2024-07-15 12:00:00+02:00", 0.0, None, None),
                )
                conn.commit()
                school_dates = {
                    code: ({pd.Timestamp("2024-07-15").date()} if code == "BE" else set())
                    for code in DE_STATE_CODES
                }

                self.assertEqual(
                    backfill_calendar_features(conn, school_dates),
                    1,
                )
                row = conn.execute(
                    """
                    SELECT is_school_holiday, school_holiday_ratio
                    FROM energy_demand
                    """
                ).fetchone()
                self.assertEqual(row[0], 1)
                self.assertGreater(row[1], 0.0)
                self.assertEqual(backfill_calendar_features(conn, school_dates), 0)
            finally:
                conn.close()


if __name__ == "__main__":
    unittest.main()
