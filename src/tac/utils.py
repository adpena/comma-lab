"""Shared training utilities for CPU and GPU lanes.

Provides signal handling, JSONL telemetry, formatted epoch logging,
and canonical research log appenders so durable state stays in sync.
"""

from __future__ import annotations

import atexit
import json
import signal
import sys
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
TRACKER_DB = REPO_ROOT / ".omx" / "state" / "review_tracker.duckdb"


def find_project_root(start: Path | None = None) -> Path:
    """Walk up from *start* to find the project root (contains src/ and upstream/).

    Args:
        start: starting directory (defaults to caller's file location via REPO_ROOT).

    Returns:
        Path to the project root.

    Raises:
        RuntimeError: if no directory with both src/ and upstream/ is found.
    """
    if start is None:
        return REPO_ROOT
    p = Path(start).resolve()
    while p != p.parent:
        if (p / "src").is_dir() and (p / "upstream").is_dir():
            return p
        p = p.parent
    raise RuntimeError("Cannot find project root (expected src/ and upstream/ dirs)")


def setup_signal_handlers(save_fn: Callable[[], None]) -> None:
    """Register SIGTERM/SIGINT/SIGHUP + atexit to call *save_fn* before exit.

    Args:
        save_fn: Zero-arg callable that persists the current training state.
    """

    def _signal_handler(signum, frame):
        try:
            print(f"\n[train] EMERGENCY SAVE (signal {signum})")
            save_fn()
            print("[train] Emergency save complete.")
        except Exception as e:
            print(f"[train] Emergency save FAILED: {e}")
        sys.exit(1)

    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        try:
            signal.signal(sig, _signal_handler)
        except (OSError, ValueError):
            pass  # some signals unavailable in threads

    def _atexit_save():
        try:
            save_fn()
        except Exception:
            pass

    atexit.register(_atexit_save)


def write_telemetry(path: str | Path, data: dict) -> None:
    """Append a single JSON object as one line to *path*."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(data) + "\n")


def log_epoch(
    epoch: int,
    total: int,
    loss: float,
    metrics: dict[str, float],
    lr: float,
    elapsed: float,
    tag: str = "",
    extra: str = "",
) -> None:
    """Print a standardised one-line epoch summary.

    Args:
        epoch: Current epoch (0-indexed).
        total: Total number of epochs.
        loss: Average loss this epoch.
        metrics: Dict of metric_name -> value (e.g. {"pose": 0.001, "seg": 0.02}).
        lr: Current learning rate.
        elapsed: Seconds this epoch took.
        tag: Optional run tag for prefix.
        extra: Optional suffix (e.g. " *BEST*").
    """
    eta_hours = elapsed * (total - epoch - 1) / 3600 if elapsed > 0 else 0.0
    parts = [f"[ep {epoch:4d}/{total}]", f"loss={loss:.4f}"]
    for k, v in metrics.items():
        parts.append(f"{k}={v:.6f}")
    parts.append(f"lr={lr:.6f}")
    parts.append(f"{elapsed:.1f}s/ep ETA={eta_hours:.1f}h")
    if extra:
        parts.append(extra)
    print(" ".join(parts))


# ---------------------------------------------------------------------------
# Canonical research log — DuckDB + markdown dual-write
# ---------------------------------------------------------------------------

def _get_research_db():
    """Get DuckDB connection with research tables (findings, runs, council)."""
    try:
        import duckdb
    except ImportError:
        return None
    TRACKER_DB.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(TRACKER_DB))
    # Sequences must exist before tables that reference them
    for seq in ("finding_seq", "run_seq", "council_seq"):
        try:
            con.execute(f"CREATE SEQUENCE IF NOT EXISTS {seq} START 1")
        except Exception:
            pass
    con.execute("""
        CREATE TABLE IF NOT EXISTS findings (
            id INTEGER DEFAULT nextval('finding_seq'),
            timestamp VARCHAR NOT NULL,
            category VARCHAR NOT NULL,
            title VARCHAR NOT NULL,
            body VARCHAR NOT NULL,
            score DOUBLE,
            variant VARCHAR DEFAULT '',
            tags VARCHAR DEFAULT ''
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS runs (
            id INTEGER DEFAULT nextval('run_seq'),
            timestamp VARCHAR NOT NULL,
            variant VARCHAR NOT NULL,
            platform VARCHAR NOT NULL,
            epoch INTEGER,
            proxy_score DOUBLE,
            auth_score DOUBLE,
            posenet DOUBLE,
            segnet DOUBLE,
            rate DOUBLE,
            tag VARCHAR DEFAULT '',
            notes VARCHAR DEFAULT ''
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS council_outputs (
            id INTEGER DEFAULT nextval('council_seq'),
            timestamp VARCHAR NOT NULL,
            session_type VARCHAR NOT NULL,
            title VARCHAR NOT NULL,
            body VARCHAR NOT NULL,
            file_path VARCHAR DEFAULT ''
        )
    """)
    return con


def record_finding(category: str, title: str, body: str,
                   score: float | None = None, variant: str = "",
                   tags: str = "") -> None:
    """Record a research finding to DuckDB + findings.md.

    Args:
        category: discovery | negative | decision | promotion | technique
        title: One-line summary
        body: Full finding text
        score: Associated score if any
        variant: Model variant if applicable
        tags: Comma-separated tags
    """
    now = datetime.now(timezone.utc).isoformat()

    # DuckDB
    con = _get_research_db()
    if con:
        try:
            con.execute("""
                INSERT INTO findings (timestamp, category, title, body, score, variant, tags)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, [now, category, title, body, score, variant, tags])
        except Exception as e:
            print(f"[utils] WARN: DuckDB finding insert failed: {e}", file=sys.stderr)
        finally:
            con.close()

    # Append to findings.md
    findings_path = REPO_ROOT / ".omx" / "research" / "findings.md"
    findings_path.parent.mkdir(parents=True, exist_ok=True)
    with open(findings_path, "a") as f:
        f.write(f"\n## {now[:10]} [{category}] {title}\n\n{body}\n")
        if score is not None:
            f.write(f"- Score: {score}\n")
        if variant:
            f.write(f"- Variant: {variant}\n")


def record_run(variant: str, platform: str, epoch: int,
               proxy_score: float | None = None, auth_score: float | None = None,
               posenet: float | None = None, segnet: float | None = None,
               rate: float | None = None, tag: str = "", notes: str = "") -> None:
    """Record a training run result to DuckDB + run_log.md.

    Args:
        variant: Architecture variant (e.g. dilated_h64)
        platform: Where it ran (local, modal_a10g, kaggle_p100)
        epoch: Epoch number at recording time
        proxy_score: Proxy composite score
        auth_score: Authoritative composite score
        posenet: PoseNet distortion
        segnet: SegNet distortion
        rate: Rate in bytes
        tag: Run tag
        notes: Free-text notes
    """
    now = datetime.now(timezone.utc).isoformat()

    # DuckDB
    con = _get_research_db()
    if con:
        try:
            con.execute("""
                INSERT INTO runs (timestamp, variant, platform, epoch,
                    proxy_score, auth_score, posenet, segnet, rate, tag, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [now, variant, platform, epoch,
                  proxy_score, auth_score, posenet, segnet, rate, tag, notes])
        except Exception as e:
            print(f"[utils] WARN: DuckDB run insert failed: {e}", file=sys.stderr)
        finally:
            con.close()

    # Append to run_log.md
    run_log = REPO_ROOT / ".ralph" / "run_log.md"
    run_log.parent.mkdir(parents=True, exist_ok=True)
    score_str = f"auth={auth_score}" if auth_score is not None else f"proxy={proxy_score}"
    with open(run_log, "a") as f:
        f.write(f"\n## {now[:19]} — {variant} on {platform} ep{epoch}\n\n")
        f.write(f"- {score_str}\n")
        if posenet is not None:
            f.write(f"- PoseNet: {posenet:.8f}\n")
        if segnet is not None:
            f.write(f"- SegNet: {segnet:.8f}\n")
        if rate is not None:
            f.write(f"- Rate: {rate}\n")
        if notes:
            f.write(f"- Notes: {notes}\n")


def record_council(session_type: str, title: str, body: str,
                   file_path: str = "") -> None:
    """Record a council output to DuckDB.

    Args:
        session_type: greenup | strategic | review | signal_loss | dx_audit
        title: Session title
        body: Full council output
        file_path: Path where output was also saved (if any)
    """
    now = datetime.now(timezone.utc).isoformat()
    con = _get_research_db()
    if con:
        try:
            con.execute("""
                INSERT INTO council_outputs (timestamp, session_type, title, body, file_path)
                VALUES (?, ?, ?, ?, ?)
            """, [now, session_type, title, body, file_path])
        except Exception as e:
            print(f"[utils] WARN: DuckDB council insert failed: {e}", file=sys.stderr)
        finally:
            con.close()
