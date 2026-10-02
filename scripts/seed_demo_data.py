"""Insert clearly marked sample inspections for dashboard demonstration."""

from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone

from backend import db


def main() -> None:
    """Create 60 synthetic rows over the previous day and a repeated fail cell."""
    random.seed(23)
    now = datetime.now(timezone.utc)
    for index in range(60):
        timestamp = (now - timedelta(minutes=(59 - index) * 24)).isoformat(timespec="seconds")
        clustered = 32 <= index <= 37
        if clustered:
            verdict, reason = "FAIL", "Anomaly score exceeds threshold"
            hotspot = (0.45 + random.uniform(-0.02, 0.02), 0.55 + random.uniform(-0.02, 0.02))
            score = random.uniform(1.15, 1.8)
        else:
            kind = random.choices(("PASS", "FAIL", "RECAPTURE"), weights=(0.78, 0.12, 0.10))[0]
            verdict = kind
            reason = {
                "PASS": "Within threshold",
                "FAIL": "Anomaly score exceeds threshold",
                "RECAPTURE": random.choice(("Too dark", "Blurry, hold steady", "No product in frame")),
            }[kind]
            hotspot = (random.random(), random.random()) if kind == "FAIL" else None
            score = random.uniform(0.1, 0.8) if kind != "FAIL" else random.uniform(1.05, 1.5)
        with db.session() as connection:
            connection.execute(
                """INSERT INTO inspections
                   (ts, source, verdict, reason, score, confidence, hotspot_x, hotspot_y, frame_path, heatmap_path)
                   VALUES (?, 'seed', ?, ?, ?, ?, ?, ?, NULL, NULL)""",
                (timestamp, verdict, reason, score, random.uniform(55, 99),
                 hotspot[0] if hotspot else None, hotspot[1] if hotspot else None),
            )
    print("Inserted 60 source='seed' inspections across the previous day.")


if __name__ == "__main__":
    main()
