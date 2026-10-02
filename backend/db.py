"""SQLite persistence for inspections, calibration settings, and alerts."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from . import config

DB_PATH = config.DATA_DIR / "visionqc.db"


def connect() -> sqlite3.Connection:
    """Return a configured SQLite connection and ensure the schema exists."""
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DB_PATH, timeout=10)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL")
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS inspections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            source TEXT NOT NULL,
            verdict TEXT NOT NULL,
            reason TEXT NOT NULL,
            score REAL NOT NULL,
            confidence REAL NOT NULL,
            hotspot_x REAL,
            hotspot_y REAL,
            frame_path TEXT,
            heatmap_path TEXT,
            feedback TEXT
        );
        CREATE TABLE IF NOT EXISTS settings (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            message TEXT NOT NULL,
            cell_x INTEGER NOT NULL,
            cell_y INTEGER NOT NULL
        );
        CREATE INDEX IF NOT EXISTS inspections_ts_idx ON inspections(ts);
        CREATE INDEX IF NOT EXISTS inspections_verdict_idx ON inspections(verdict);
        CREATE INDEX IF NOT EXISTS alerts_cell_ts_idx ON alerts(cell_x, cell_y, ts);
        """
    )
    columns = {row["name"] for row in connection.execute("PRAGMA table_info(inspections)").fetchall()}
    for name in ("brightness", "sharpness", "alignment_shift"):
        if name not in columns:
            connection.execute(f"ALTER TABLE inspections ADD COLUMN {name} REAL")
    connection.execute(
        "INSERT OR IGNORE INTO settings(key, value) VALUES('threshold', ?)",
        (str(config.DEFAULT_THRESHOLD),),
    )
    connection.executemany(
        "INSERT OR IGNORE INTO settings(key, value) VALUES(?, ?)",
        (
            ("window_n", json.dumps(config.REPEAT_WINDOW)),
            ("cluster_k", json.dumps(config.REPEAT_COUNT)),
            ("show_seed", json.dumps(config.SHOW_SEED_DATA)),
        ),
    )
    connection.commit()
    return connection


@contextmanager
def session() -> Iterator[sqlite3.Connection]:
    """Yield a database connection and always close it after the transaction."""
    connection = connect()
    try:
        with connection:
            yield connection
    finally:
        connection.close()


def add_inspection(
    source: str,
    verdict: str,
    reason: str,
    score: float,
    confidence: float,
    hotspot: tuple[float, float] | None,
    frame_path: str | None,
    heatmap_path: str | None,
    brightness: float | None = None,
    sharpness: float | None = None,
    alignment_shift: float | None = None,
) -> int:
    """Insert one inspection and return its database id."""
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with session() as connection:
        cursor = connection.execute(
            """INSERT INTO inspections
               (ts, source, verdict, reason, score, confidence, hotspot_x, hotspot_y, frame_path, heatmap_path,
                brightness, sharpness, alignment_shift)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (timestamp, source, verdict, reason, score, confidence,
             hotspot[0] if hotspot else None, hotspot[1] if hotspot else None,
             frame_path, heatmap_path, brightness, sharpness, alignment_shift),
        )
        return int(cursor.lastrowid)


def get_history(limit: int = 20) -> list[dict[str, Any]]:
    """Return recent inspections, newest first."""
    show_seed = bool(get_setting("show_seed", config.SHOW_SEED_DATA))
    with session() as connection:
        query = "SELECT * FROM inspections"
        if not show_seed:
            query += " WHERE source != 'seed'"
        rows = connection.execute(f"{query} ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    return [dict(row) for row in rows]


def get_inspection(inspection_id: int) -> dict[str, Any] | None:
    """Return one inspection by id."""
    with session() as connection:
        row = connection.execute("SELECT * FROM inspections WHERE id = ?", (inspection_id,)).fetchone()
    return dict(row) if row else None


def mark_false_alarm(inspection_id: int) -> None:
    """Mark an inspection as a confirmed false alarm."""
    with session() as connection:
        cursor = connection.execute(
            "UPDATE inspections SET feedback = 'false_alarm_confirmed' WHERE id = ?",
            (inspection_id,),
        )
        if cursor.rowcount == 0:
            raise KeyError(f"Inspection {inspection_id} not found")


def get_threshold() -> float:
    """Read the supervisor decision threshold."""
    with session() as connection:
        row = connection.execute("SELECT value FROM settings WHERE key = 'threshold'").fetchone()
    return float(row["value"]) if row else config.DEFAULT_THRESHOLD


def set_threshold(value: float) -> None:
    """Persist a supervisor decision threshold."""
    with session() as connection:
        connection.execute(
            "INSERT INTO settings(key, value) VALUES('threshold', ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (str(value),),
        )


def set_setting(key: str, value: Any) -> None:
    """Persist a JSON-compatible setting."""
    with session() as connection:
        connection.execute(
            "INSERT INTO settings(key, value) VALUES(?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, json.dumps(value)),
        )


def get_setting(key: str, default: Any = None) -> Any:
    """Read a JSON-compatible setting or return the supplied default."""
    with session() as connection:
        row = connection.execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
    if row is None:
        return default
    try:
        return json.loads(row["value"])
    except json.JSONDecodeError:
        return row["value"]


def get_app_settings() -> dict[str, int | bool]:
    """Return persisted Settings controls using the configured defaults."""
    return {
        "window_n": int(get_setting("window_n", config.REPEAT_WINDOW)),
        "cluster_k": int(get_setting("cluster_k", config.REPEAT_COUNT)),
        "show_seed": bool(get_setting("show_seed", config.SHOW_SEED_DATA)),
    }


def add_alert(message: str, cell_x: int, cell_y: int) -> int:
    """Record a repeated-defect alert."""
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with session() as connection:
        cursor = connection.execute(
            "INSERT INTO alerts(ts, message, cell_x, cell_y) VALUES(?, ?, ?, ?)",
            (timestamp, message, cell_x, cell_y),
        )
        return int(cursor.lastrowid)


def get_alerts(limit: int = 50) -> list[dict[str, Any]]:
    """Return recent alerts, newest first."""
    with session() as connection:
        rows = connection.execute("SELECT * FROM alerts ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    return [dict(row) for row in rows]
