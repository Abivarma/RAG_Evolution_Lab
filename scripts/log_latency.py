from __future__ import annotations
"""Append a session-end entry to results/session_log.jsonl on Stop hook."""

import json
import os
from datetime import datetime, timezone
from pathlib import Path

session_id = os.environ.get("CLAUDE_SESSION_ID", "unknown")
log_path = Path("results/session_log.jsonl")
log_path.parent.mkdir(parents=True, exist_ok=True)

entry = {
    "session_id": session_id,
    "stopped_at": datetime.now(timezone.utc).isoformat(),
}
with log_path.open("a") as fh:
    fh.write(json.dumps(entry) + "\n")
