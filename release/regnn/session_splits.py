from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Optional


def load_session_allowlist(
    manifest_path: Optional[Path],
    split_name: Optional[str],
) -> tuple[Optional[tuple[str, ...]], Optional[str]]:
    """Load a named session-disjoint split and hash its source manifest."""
    if manifest_path is None and split_name is None:
        return None, None
    if manifest_path is None or split_name is None:
        raise ValueError(
            "session-split-manifest and session-split-name must be set together"
        )
    manifest_path = manifest_path.resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    splits = payload.get("splits")
    if not isinstance(splits, dict) or split_name not in splits:
        raise ValueError(f"Unknown session split {split_name!r}")
    raw_sessions = splits[split_name]
    if not isinstance(raw_sessions, list) or not raw_sessions:
        raise ValueError(f"Session split {split_name!r} must be a non-empty list")
    sessions = tuple(sorted({str(value) for value in raw_sessions}))
    if len(sessions) != len(raw_sessions):
        raise ValueError(f"Session split {split_name!r} contains duplicates")
    return sessions, hashlib.sha256(manifest_path.read_bytes()).hexdigest()
