# SPDX-License-Identifier: MIT
"""decode_cache — content-addressed memoization of the (expensive) decoded witness verdict (task #350).

The payload-TTO before/after validation runs the level-set byte-close -> inflate -> frozen-CPU-torch
SegNet/PoseNet realized d_seg/d_pose verdict, which is the costly step (600 EfficientNet-B2 forwards).
It is a PURE FUNCTION of (payload bytes, decode config): the SAME code table + SAME decode config
always inflates to the SAME frames -> the SAME realized d_seg/d_pose. So it is safely memoizable by
content address.

Key = sha256(payload_sha256 || decode_config_sha256). Value = the decoded verdict dict
(d_seg, d_pose, n_pairs, ...). Store = an fcntl-locked append-only JSONL (last-write-wins on key),
mirroring the repo's canonical state-ledger discipline (atomic append under LOCK_EX). This lets the
TTO loop skip re-decoding an already-measured payload and lets run-2 reuse run-1's decoded verdicts
for free (the ILC error term stays cheap).

AUTHORITY-PRESERVING: the cache stores whatever axis label the verdict carried; it NEVER upgrades an
advisory verdict to a score. Reading a cached value returns the same `[macOS-CPU advisory]` (or
whatever) authority it was stored with.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
from pathlib import Path
from typing import Any

from tac.jsonl_store import append_locked_jsonl

_DEFAULT_STORE = Path(__file__).resolve().parents[3] / ".omx" / "state" / "witness_decode_cache.jsonl"


def payload_sha256(code_table) -> str:
    """Content address of a code payload (numpy or mlx array-like) — fp32 bytes."""
    import numpy as np

    a = np.ascontiguousarray(np.asarray(code_table, np.float32))
    return hashlib.sha256(a.tobytes()).hexdigest()


def config_sha256(decode_config: dict[str, Any]) -> str:
    """Content address of the decode config (sorted-key canonical JSON)."""
    blob = json.dumps(decode_config, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()


def cache_key(payload_sha: str, config_sha: str) -> str:
    return hashlib.sha256((payload_sha + "||" + config_sha).encode()).hexdigest()[:32]


def _resolve(store: str | Path | None) -> Path:
    return Path(store) if store is not None else _DEFAULT_STORE


def get(payload_sha: str, config_sha: str, *, store: str | Path | None = None) -> dict | None:
    """Return the cached decoded verdict for this content address, or None. Last-write-wins."""
    path = _resolve(store)
    if not path.exists():
        return None
    key = cache_key(payload_sha, config_sha)
    hit = None
    with open(path) as fh:
        fcntl.flock(fh.fileno(), fcntl.LOCK_SH)
        try:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if row.get("key") == key:
                    hit = row  # keep scanning: last write wins
        finally:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
    return hit


def put(payload_sha: str, config_sha: str, verdict: dict, *,
        store: str | Path | None = None) -> str:
    """Append a decoded verdict under its content-address key (fcntl-locked atomic append).

    Uses the canonical .omx/state helper (tac.jsonl_store.append_locked_jsonl); see
    .omx/research/fcntl_lock_canonicalization_plan_20260710.md Batch 1. ``get()`` above keeps
    its own ``LOCK_SH`` read-lock (untouched — a distinct, read-only operation).
    """
    path = _resolve(store)
    key = cache_key(payload_sha, config_sha)
    row = {"key": key, "payload_sha": payload_sha, "config_sha": config_sha, "verdict": verdict}
    append_locked_jsonl(path, row)
    return key


def memoized_decode(code_table, decode_config: dict[str, Any], decode_fn,
                    *, store: str | Path | None = None) -> tuple[dict, bool]:
    """Return (verdict, was_cached). Computes ``decode_fn()`` only on a cache miss, then stores it.

    ``decode_fn`` is a 0-arg callable returning the decoded verdict dict (the caller closes over the
    real byte-close -> inflate -> CPU-torch SegNet/PoseNet path). This is the memoization hook the
    TTO driver wraps its before/after decoded verdict in."""
    p_sha = payload_sha256(code_table)
    c_sha = config_sha256(decode_config)
    hit = get(p_sha, c_sha, store=store)
    if hit is not None:
        return hit["verdict"], True
    verdict = decode_fn()
    put(p_sha, c_sha, verdict, store=store)
    return verdict, False
