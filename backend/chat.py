"""Answer questions using only aggregate and row data in SQLite."""

from __future__ import annotations

import json
from typing import Any

from dotenv import load_dotenv

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
    """Query OpenRouter with a strict SQLite-only inspection context."""
    load_dotenv(config.ROOT_DIR / ".env")
    import os

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        return {"answer": "Chat unavailable: add OPENROUTER_API_KEY to the project .env file."}
    try:
        from openai import OpenAI

        client = OpenAI(
            api_key=api_key,
            base_url=config.OPENROUTER_BASE_URL,
            default_headers={
                "HTTP-Referer": "http://localhost:5173",
                "X-Title": "VisionQC",
            },
        )
        response = client.chat.completions.create(
            model=config.OPENROUTER_MODEL,
            max_tokens=config.CHAT_MAX_TOKENS,
            messages=[{
                "role": "system",
                "content": SYSTEM_PROMPT,
            }, {
                "role": "user",
                "content": "Inspection data (SQLite):\n"
                + json.dumps(_context(), ensure_ascii=True)
                + "\n\nQuestion: " + question,
            }],
        )
        text = response.choices[0].message.content or ""
        return {"answer": text or "The inspection data does not contain an answer."}
    except Exception as error:
        return {"answer": f"Chat request failed: {error}"}
