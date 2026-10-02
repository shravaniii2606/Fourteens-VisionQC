"""Inspection metrics, recurring-location alerts, and drift monitoring."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from typing import Any

from . import config, db


def _cell_for(row: dict[str, Any]) -> tuple[int, int] | None:
    """Map a normalized hotspot into the configured dashboard grid."""
    x, y = row.get("hotspot_x"), row.get("hotspot_y")
    if x is None or y is None:
        return None
    return (
        min(config.GRID_SIZE - 1, max(0, int(float(x) * config.GRID_SIZE))),
        min(config.GRID_SIZE - 1, max(0, int(float(y) * config.GRID_SIZE))),
    )


def stats() -> dict[str, Any]:
    """Summarize counts, rejection rates, score trend, and recapture reasons."""
    today = datetime.now(timezone.utc).date().isoformat()
    with db.session() as connection:
        counts = connection.execute(
            """SELECT COUNT(*) AS total,
                      SUM(verdict = 'PASS') AS pass_count,
                      SUM(verdict = 'FAIL') AS fail_count,
                      SUM(verdict = 'RECAPTURE') AS recapture_count,
                      SUM(verdict = 'PASS' AND ts LIKE ?) AS today_pass,
                      SUM(verdict = 'FAIL' AND ts LIKE ?) AS today_fail
               FROM inspections""",
            (f"{today}%", f"{today}%"),
        ).fetchone()
        reason_rows = connection.execute(
            "SELECT reason, COUNT(*) AS count FROM inspections WHERE verdict = 'RECAPTURE' GROUP BY reason ORDER BY count DESC"
        ).fetchall()
    total = int(counts["total"] or 0)
    pass_count = int(counts["pass_count"] or 0)
    fail_count = int(counts["fail_count"] or 0)
    recapture_count = int(counts["recapture_count"] or 0)
    today_pass = int(counts["today_pass"] or 0)
    today_fail = int(counts["today_fail"] or 0)
    recent = db.get_history(config.DASHBOARD_TREND_POINTS)
    decided = today_pass + today_fail
    trend = [
        {"ts": row["ts"], "score": row["score"], "verdict": row["verdict"]}
        for row in reversed(recent[:100])
    ]
    return {
        "total": total,
        "pass": pass_count,
        "fail": fail_count,
        "recapture": recapture_count,
        "rejection_rate_today": today_fail / decided if decided else 0.0,
        "recapture_rate": recapture_count / total if total else 0.0,
        "score_trend": trend,
        "recapture_reasons": [{"reason": row["reason"], "count": row["count"]} for row in reason_rows],
    }


def repeat_defect_check(window: int = config.REPEAT_WINDOW, required: int = config.REPEAT_COUNT) -> list[dict[str, Any]]:
    """Create alerts when a grid location repeats across recent failures."""
    failures = [row for row in db.get_history(config.ANALYTICS_SCAN_LIMIT) if row["verdict"] == "FAIL"][:window]
    cells = Counter(cell for row in failures if (cell := _cell_for(row)) is not None)
    if not failures:
        return []
    oldest_timestamp = min(row["ts"] for row in failures)
    created: list[dict[str, Any]] = []
    for (cell_x, cell_y), count in cells.items():
        if count < required:
            continue
        with db.session() as connection:
            duplicate = connection.execute(
                "SELECT id FROM alerts WHERE cell_x = ? AND cell_y = ? AND ts >= ? LIMIT 1",
                (cell_x, cell_y, oldest_timestamp),
            ).fetchone()
        if duplicate:
            continue
        message = f"Same spot failing {count} of last {window} units. Check the machine."
        alert_id = db.add_alert(message, cell_x, cell_y)
        created.append({"id": alert_id, "message": message, "cell_x": cell_x, "cell_y": cell_y})
    return created


def cumulative_heatmap(window: int = config.REPEAT_WINDOW) -> dict[str, Any]:
    """Return a square grid of failure counts from recent inspections."""
    recent = db.get_history(window)
    grid = [[0 for _ in range(config.GRID_SIZE)] for _ in range(config.GRID_SIZE)]
    for row in recent:
        if row["verdict"] != "FAIL":
            continue
        cell = _cell_for(row)
        if cell:
            grid[cell[1]][cell[0]] += 1
    return {"grid": grid, "size": config.GRID_SIZE, "window": window}


def drift(calibration_mean: float | None = None) -> dict[str, Any]:
    """Compare the recent decision-score mean with its calibrated baseline."""
    rows = [row for row in db.get_history(config.ANALYTICS_SCAN_LIMIT) if row["verdict"] in ("PASS", "FAIL")][:config.DRIFT_WINDOW]
    baseline = calibration_mean
    if baseline is None:
        baseline = db.get_setting("calibration", {}).get("mean")
    if not rows or baseline is None:
        return {"warn": False, "moving_average": None, "calibration_mean": baseline}
    moving_average = sum(float(row["score"]) for row in rows) / len(rows)
    warn = moving_average > float(baseline) * (1.0 + config.DRIFT_PERCENT / 100.0)
    return {"warn": warn, "moving_average": moving_average, "calibration_mean": baseline}


def top_failing_cells(limit: int = 5) -> list[dict[str, int]]:
    """Return the most common failed hotspot cells from recent history."""
    failures = [row for row in db.get_history(config.ANALYTICS_SCAN_LIMIT) if row["verdict"] == "FAIL"]
    counts = Counter(cell for row in failures if (cell := _cell_for(row)) is not None)
    return [{"cell_x": cell[0], "cell_y": cell[1], "count": count} for cell, count in counts.most_common(limit)]
