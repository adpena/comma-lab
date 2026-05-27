# SPDX-License-Identifier: MIT
"""Stable JSON identity helpers for scheduler custody surfaces."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any


def stable_json_sha256(payload: Mapping[str, Any]) -> str:
    """Return the stable SHA-256 used for queue and policy identity."""

    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


__all__ = ["stable_json_sha256"]
