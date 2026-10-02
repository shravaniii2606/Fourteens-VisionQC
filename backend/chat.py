"""Answer questions using only aggregate and row data in SQLite."""

from __future__ import annotations

import json
import os
from typing import Any

from . import analytics, config, db

SYSTEM_PROMPT = "Answer only from the provided inspection data. If the data does not contain the answer, say so."


def _context() -> dict[str, Any]:
    """Collect the approved SQL-backed context for one chat request."""
    recent_fails = [row for row in db.get_history(config.ANALYTICS_SCAN_LIMIT) if row["verdict"] == "FAIL"][:config.CHAT_FAIL_LIMIT]
    return {
        "stats": analytics.stats(),
        "top_failing_grid_cells": analytics.top_failing_cells(),
        "last_10_fails": [
            {"ts": row["ts"], "score": row["score"], "reason": row["reason"],
             "hotspot_x": row["hotspot_x"], "hotspot_y": row["hotspot_y"]}
            for row in recent_fails
        ],
        "recent_alerts": db.get_alerts(config.CHAT_ALERT_LIMIT),
        "recapture_reasons": analytics.stats()["recapture_reasons"],
    }


def answer(question: str) -> dict[str, str]:
    """Query Anthropic with a strict SQLite-only inspection context."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return {"answer": "Chat unavailable: ANTHROPIC_API_KEY is not set."}
    try:
        from anthropic import Anthropic

        client = Anthropic(api_key=api_key)
        response = client.messages.create(
            model=config.ANTHROPIC_MODEL,
            max_tokens=config.ANTHROPIC_MAX_TOKENS,
            system=SYSTEM_PROMPT,
            messages=[{
                "role": "user",
                "content": "Inspection data (SQLite):\n"
                + json.dumps(_context(), ensure_ascii=True)
                + "\n\nQuestion: " + question,
            }],
        )
        text = "\n".join(block.text for block in response.content if getattr(block, "type", "") == "text")
        return {"answer": text or "The inspection data does not contain an answer."}
    except Exception as error:
        return {"answer": f"Chat request failed: {error}"}
