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
    source_filter = "" if db.get_app_settings()["show_seed"] else " WHERE source != 'seed'"
    reason_filter = "" if db.get_app_settings()["show_seed"] else " AND source != 'seed'"
    with db.session() as connection:
        counts = connection.execute(
            """SELECT COUNT(*) AS total,
                      SUM(verdict = 'PASS') AS pass_count,
                      SUM(verdict = 'FAIL') AS fail_count,
                      SUM(verdict = 'RECAPTURE') AS recapture_count,
                      SUM(verdict = 'PASS' AND ts LIKE ?) AS today_pass,
                      SUM(verdict = 'FAIL' AND ts LIKE ?) AS today_fail
               FROM inspections""" + source_filter,
            (f"{today}%", f"{today}%"),
        ).fetchone()
        reason_rows = connection.execute(
            "SELECT reason, COUNT(*) AS count FROM inspections WHERE verdict = 'RECAPTURE'"
            + reason_filter + " GROUP BY reason ORDER BY count DESC"
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


def repeat_defect_check(window: int | None = None, required: int | None = None) -> list[dict[str, Any]]:
    """Create alerts when a grid location repeats across recent failures."""
    settings = db.get_app_settings()
    window = int(settings["window_n"]) if window is None else window
    required = int(settings["cluster_k"]) if required is None else required
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


def cumulative_heatmap(window: int | None = None) -> dict[str, Any]:
    """Return failure counts and unit metadata for the recent defect grid."""
    settings = db.get_app_settings()
    window = int(settings["window_n"]) if window is None else window
    required = int(settings["cluster_k"])
    recent = db.get_history(window)
    grid = [[0 for _ in range(config.GRID_SIZE)] for _ in range(config.GRID_SIZE)]
    cells = [[[] for _ in range(config.GRID_SIZE)] for _ in range(config.GRID_SIZE)]
    for row in recent:
        if row["verdict"] != "FAIL":
            continue
        cell = _cell_for(row)
        if cell:
            grid[cell[1]][cell[0]] += 1
            cells[cell[1]][cell[0]].append(row)

    cell_data = []
    hottest = None
    for row_index, row_cells in enumerate(cells):
        data_row = []
        for column_index, rows in enumerate(row_cells):
            rows = sorted(rows, key=lambda item: item["ts"], reverse=True)
            item = {
                "col": column_index + 1,
                "row": row_index + 1,
                "count": len(rows),
                "unit_ids": [int(unit["id"]) for unit in rows[:3]],
                "last_seen": rows[0]["ts"] if rows else None,
            }
            data_row.append(item)
            if rows and (hottest is None or item["count"] > hottest["count"]):
                hottest = item
        cell_data.append(data_row)

    active_alert = None
    if hottest and hottest["count"] >= required:
        active_alert = {
            **hottest,
            "message": f"SAME SPOT FAILING {hottest['count']} OF LAST {window} UNITS. CHECK THE MACHINE.",
        }
    return {
        "grid": grid,
        "cells": cell_data,
        "hottest": hottest,
        "active_alert": active_alert,
        "size": config.GRID_SIZE,
        "window": window,
        "alert_threshold": required,
    }


def drift(calibration_mean: float | None = None) -> dict[str, Any]:
    """Return PASS-only score/capture drift metrics for the active rolling window."""
    calibration = db.get_setting("calibration", {})
    capture_baseline = db.get_setting("capture_calibration", {})
    baseline = calibration_mean if calibration_mean is not None else calibration.get("mean")
    reset_at = db.get_setting("drift_reset_at")
    rows = db.get_history(config.ANALYTICS_SCAN_LIMIT)
    if reset_at:
        rows = [row for row in rows if row.get("ts", "") >= reset_at]
    passed = [row for row in rows if row["verdict"] == "PASS"][: config.DRIFT_WINDOW]
    score_values = [float(row["score"]) for row in passed]
    score_std = float(calibration.get("std", 0.0)) if calibration else 0.0
    missing = [field for field in ("brightness", "sharpness") if not passed or any(row.get(field) is None for row in passed)]
    camera_cause = None
    product_cause = None
    for row in passed:
        brightness = row.get("brightness")
        sharpness = row.get("sharpness")
        if brightness is not None and (
            brightness < capture_baseline.get("brightness_min", -float("inf"))
            or brightness > capture_baseline.get("brightness_max", float("inf"))
        ):
            camera_cause = "Lighting changed"
            break
        if sharpness is not None and sharpness < capture_baseline.get("sharpness_min", -float("inf")):
            camera_cause = "Image getting blurry, clean the lens"
            break
    if camera_cause is None and any(
        row.get("alignment_shift") is not None
        and float(row["alignment_shift"]) > max(8.0, config.INPUT_SIZE * config.ALIGN_MAX_SHIFT_FRACTION)
        for row in passed
    ):
        product_cause = "Product position changed"
    if product_cause is None and baseline is not None and score_values:
        if max(score_values) > float(baseline) + 2.0 * score_std:
            product_cause = "Product appearance changed"
    if len(passed) < config.DRIFT_MIN_PASSED or baseline is None:
        status = "warming_up"
        cause = None
    elif camera_cause:
        status = "camera_drift"
        cause = camera_cause
    elif product_cause:
        status = "product_drift"
        cause = product_cause
    else:
        status = "stable"
        cause = None
    return {
        "warn": status in ("camera_drift", "product_drift"),
        "status": status,
        "cause": cause,
        "message": cause or ("Collecting baseline data" if status == "warming_up" else "Inspection conditions are stable"),
        "moving_average": float(sum(score_values) / len(score_values)) if score_values else None,
        "calibration_mean": baseline,
        "calibration_std": score_std,
        "score_trend": [{"ts": row["ts"], "score": row["score"], "verdict": row["verdict"]} for row in reversed(passed)],
        "capture_health": [
            {
                "ts": row["ts"],
                "brightness": row.get("brightness"),
                "sharpness": row.get("sharpness"),
                "alignment_shift": row.get("alignment_shift"),
            }
            for row in reversed(passed)
        ],
        "capture_baseline": {
            "brightness": capture_baseline.get(
                "brightness_mean",
                (float(capture_baseline["brightness_min"]) + float(capture_baseline["brightness_max"])) / 2.0
                if "brightness_min" in capture_baseline and "brightness_max" in capture_baseline
                else None,
            ),
            "sharpness": capture_baseline.get(
                "sharpness_mean",
                float(capture_baseline["sharpness_min"]) * 2.0 if "sharpness_min" in capture_baseline else None,
            ),
        },
        "missing_fields": missing,
        "units_used": len(passed),
        "minimum_needed": config.DRIFT_MIN_PASSED,
        "window_size": config.DRIFT_WINDOW,
        "reset_at": reset_at,
    }


def top_failing_cells(limit: int = 5) -> list[dict[str, int]]:
    """Return the most common failed hotspot cells from recent history."""
    failures = [row for row in db.get_history(config.ANALYTICS_SCAN_LIMIT) if row["verdict"] == "FAIL"]
    counts = Counter(cell for row in failures if (cell := _cell_for(row)) is not None)
    return [{"cell_x": cell[0], "cell_y": cell[1], "count": count} for cell, count in counts.most_common(limit)]
