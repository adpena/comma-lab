#!/usr/bin/env python3
"""ASGI + async + WebSocket LIVE dashboard for the level-set witness deploy run.

Why this exists (supersedes the meta-refresh ``render_levelset_dashboard`` +
``http.server`` + ``dashboard_up`` set): meta-refresh reloads the WHOLE page on a
timer (jank, flash, lost scroll). This is an ASGI app (starlette on uvicorn) that
async-tails the run-dir verdict logs and PUSHES new points to connected clients
over a WebSocket — the page updates IN PLACE with NO reload. A polling fallback
(``GET /api/state`` every few seconds) keeps data flowing if the WS handshake is
blocked.

REUSE (DRY): the verdict-JSON-line parsing AND the self-calibrating live/stale
cadence logic are imported verbatim from the sibling ``render_levelset_dashboard``
(``_parse_verdicts`` / ``_resolve_watched_log`` / ``_compute_liveness`` / the
cadence-state helpers) — ONE source of truth for "is the run live", so the two
dashboards never disagree.

AUTHORITY: everything here is ``[macOS-MLX training] advisory, NON-PROMOTABLE``.
The contest score is byte-closed on contest-CPU/CUDA; the frontier pointer is
READ from ``.omx/state/canonical_frontier_pointer.json`` (the SoT per CLAUDE.md
"Frontier scores are pointer-only" — never a hardcoded literal here) — a
dashboard is a MEANS, not the score.

DISCLOSURE HYGIENE (CLAUDE.md "Public Disclosure Hygiene", NON-NEGOTIABLE): the
TRIALITY/PLAN tab describes OUR METHOD. When an access key is configured
(``--access-key`` / ``DASH_ACCESS_KEY``) every request that arrives THROUGH the
public Cloudflare tunnel (detected by the ``Cf-Ray`` / ``Cf-Connecting-Ip``
headers cloudflared injects) MUST present the key (``?k=`` / ``X-Dash-Key`` /
``dash_key`` cookie) or it gets a 401 login page — the method is never served to
an unauthenticated public visitor. Bare-local (127.0.0.1, no CF headers) requests
are trusted and need no key, so local dev + E2E stay frictionless. uvicorn's
access log is DISABLED so the key never lands in a log line.

Config is read from the environment (so it works under ``uvicorn
dashboard_server:app``) with argparse overrides when run as ``__main__``:
``DASH_RUN_DIR`` / ``DASH_LOG_GLOB`` / ``DASH_TAU`` / ``DASH_L7`` /
``DASH_GOAL_DSEG`` / ``DASH_GOAL_DSEG_15`` / ``DASH_POLL`` / ``DASH_HOST`` /
``DASH_PORT`` / ``DASH_ACCESS_KEY`` / ``DASH_CADENCE_STATE`` / ``DASH_TRAINING_PID``.

    .venv/bin/python tools/dashboard_server.py --port 8790

STAGE MAP IS DERIVED, NOT HAND-FED (operator 2026-07-07): the curriculum stage
boundaries are read back PER RUN from the run's own launch.sh through the trainer's
REAL argparse + the run's emitted transition evidence, via
``tac.witness_dsl.schedule_readback`` (the DSL single source of truth). ``--tau`` /
``--l7`` are explicit OVERRIDES only; when the read-back fails (old run dirs) the
page shows a visible "schedule: fallback" marker. Disabled stages (e.g.
``--l7-start-epoch 1001`` on a 1000-epoch run) are OMITTED — the exact mislabel
class this fixes.
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import datetime
import hmac
import json
import os
import re
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

# ── reuse the canonical verdict-parse + self-calibrating liveness (DRY) ──
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import dashboard_trajectory_model as dtm  # sophisticated DATA-DERIVED projection
import render_levelset_dashboard as rld

# NB: after the sys.path bootstrap above -- import position is deliberate.
from tac import process_liveness

# schema-driven run introspection (#352): classifies the run's schedule/curriculum into
# EVENT-TRIGGERED / DERIVED / FIXED-CAP + exposes the costate controller, LawRef constants
# manifest, planned τ/β/LR curves, liveness row, mem_probe + fired-event telemetry — all
# from the run's OWN artifacts. Fail-open: a broken import must never kill the daemon.
try:
    import witness_run_introspect as wri
except Exception:  # load-bearing daemon; degrade to no introspection panels, never crash
    wri = None

# Canonical witness run-artifact CONTRACT (single source of truth for run filenames).
# Fail-open with literal fallbacks so a broken tac install never kills the daemon.
try:
    from tac import witness_run_artifacts as _wra
    _COSTATE_JSONL = _wra.COSTATE_JSONL
except Exception:
    _wra = None  # resolved_glob's contract-derived discovery falls back to the base glob
    _COSTATE_JSONL = "costate_shadow.jsonl"

# ── #366 DDM joint-descent CAMPAIGN reader (canonical run-dir contract, read-only,
# mtime-gated incremental). Powers the CAMPAIGN tab (/api/campaign). Fail-open:
# a broken tac install must never kill the daemon; the tab then reports the reason.
try:
    from tac.ddm_campaign_run_reader import CampaignRunReader as _CampaignRunReader
except Exception:  # load-bearing daemon; degrade to a visible reason, never crash
    _CampaignRunReader = None

# ── canonical DSL schedule read-back (operator 2026-07-07: observability consumers
# DERIVE the stage map from the run's own config via the DSL — never hand-fed
# constants). Fail-open: a missing/broken tac install must never kill the daemon;
# refresh() then falls back to the legacy path with a visible "schedule: fallback".
try:
    from tac.witness_dsl.schedule_readback import (
        read_schedule as _dsl_read_schedule,
    )
    from tac.witness_dsl.schedule_readback import (
        resolve_run_dir_for_log as _dsl_resolve_run_dir,
    )
except Exception:  # load-bearing daemon; degrade visibly, never crash
    _dsl_read_schedule = None
    _dsl_resolve_run_dir = None

# ── curriculum + pose-readiness truth-rendering models (operator 2026-07-10: the
# curriculum panel must render the DERIVED event-gated schedule, not a hardcoded
# PR95 epoch skeleton; the pose panel must display honest disengagement plus the R1 reference). Sibling
# tools/ module; guarded so an import error leaves the panels on their legacy path.
try:
    import dashboard_curriculum_panel as _dcp
except Exception:  # pragma: no cover - degraded-but-alive path
    _dcp = None

from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.gzip import GZipMiddleware
from starlette.responses import (
    HTMLResponse,
    JSONResponse,
)
from starlette.routing import Route, WebSocketRoute
from starlette.websockets import WebSocket, WebSocketDisconnect

# ── canonical frontier pointer (SoT: .omx/state/canonical_frontier_pointer.json) ──
# CLAUDE.md "Frontier scores are pointer-only" (NON-NEGOTIABLE): NO hardcoded score
# literal anywhere on this surface. The pointer is READ from the canonical file,
# mtime-cached, and re-checked on the server's poll cadence. Unreadable/missing ->
# {"ok": False} and every consumer renders an explicit "pointer unavailable" state —
# NEVER a baked number dressed as data.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_POINTER_JSON = _REPO_ROOT / ".omx" / "state" / "canonical_frontier_pointer.json"
_PTR_STATE: dict = {"checked": 0.0, "sig": None,
                    "data": {"ok": False, "reason": "not yet read"}}

# ── server start-time + code-mtime snapshot (for tools/dashboard_ctl.py auto-reload) ──
# The WebSocket auto-updates DATA in place, but NEW SERVER CODE (or a front-end asset
# edit) needs a process RELOAD to take effect. dashboard_ctl.py's ensure-up compares the
# CURRENT on-disk source mtime against this snapshot (exposed via /healthz `code_mtime`)
# and does a zero-downtime durable reload when the code has changed since this process
# started. The snapshot is frozen AT IMPORT so it reflects the code THIS process is
# actually running.
_SERVER_START_TS = time.time()
_SERVER_START_UTC = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(_SERVER_START_TS))
# Source files whose content is baked into THIS process (the server module + the
# front-end JS assets it inlines into the served page). A change to any of these
# requires a process reload to reach the browser — that is exactly what the auto-reload
# staleness check watches. Kept explicit (not a broad glob) so unrelated edits in tools/
# never spuriously trigger a dashboard reload.
_CODE_SOURCE_FILES: tuple[str, ...] = (
    "dashboard_server.py",
    # render_levelset_dashboard is IMPORTED by this server (verdict parsing + the
    # run-selection logic _resolve_watched_log/_resolve_run_log/_launch_ts) — its
    # behavior is baked into the running process, so an edit there is exactly as
    # reload-requiring as an edit here. Omitting it made dashboard_ctl report
    # "code fresh — no-op" after a run-selection fix (2026-07-11), silently
    # keeping the stale selection live.
    "render_levelset_dashboard.py",
    "dashboard_flow_client.js",
    "dashboard_whyhow_client.js",
)


def _code_mtime_now() -> float:
    """Max mtime over the server's own baked-in source files (0.0 if none resolve).

    Used both to freeze the at-start snapshot and, from dashboard_ctl.py, to detect a
    code edit since the running server started. Best-effort: a missing sibling asset is
    skipped, never an error (the server must never crash on a stat)."""
    here = Path(__file__).resolve().parent
    newest = 0.0
    for name in _CODE_SOURCE_FILES:
        try:
            newest = max(newest, (here / name).stat().st_mtime)
        except OSError:
            continue
    return newest


_CODE_MTIME_AT_START = _code_mtime_now()


def frontier_pointer() -> dict:
    """Current contest-CPU frontier from the canonical pointer file (fail-open).

    Returns ``{"ok": True, "score", "axis", "since", "source"}`` or
    ``{"ok": False, "reason"}``. Re-stats at most every 5 s; re-parses only on
    mtime/size change, so it refreshes on the poll cadence for free."""
    now = time.time()
    if now - _PTR_STATE["checked"] < 5.0:
        return _PTR_STATE["data"]
    _PTR_STATE["checked"] = now
    try:
        st = _POINTER_JSON.stat()
        sig = (st.st_mtime, st.st_size)
        if sig == _PTR_STATE["sig"] and _PTR_STATE["data"].get("ok"):
            return _PTR_STATE["data"]
        d = json.loads(_POINTER_JSON.read_text())
        cpu = d.get("our_local_frontier_contest_cpu") or {}
        score = float(cpu["score"])
        _PTR_STATE["sig"] = sig
        _PTR_STATE["data"] = {
            "ok": True, "score": score, "axis": "contest-CPU",
            "since": str(cpu.get("measured_at_utc", ""))[:10] or None,
            "source": ".omx/state/canonical_frontier_pointer.json",
        }
    except Exception as exc:
        _PTR_STATE["sig"] = None
        _PTR_STATE["data"] = {"ok": False,
                              "reason": f"{type(exc).__name__}: {exc}"}
    return _PTR_STATE["data"]


# THE GOAL ladder targets (provenance: CLAUDE.md "THE GOAL — SUB-0.15": T_1 = sub-0.19
# floor of acceptable, T_3 = sub-0.15 the target). These are MISSION constants, not run
# config; the d_seg goal LINES are DERIVED per run from these targets + the run's OWN
# measured pose+rate (see _derive_goal_info) unless an explicit env/CLI override is set.
_TARGET_S_T1 = 0.19
_TARGET_S_T3 = 0.15

# Telemetry accuracy (operator 2026-07-03, "confident-wrong is the worst failure"):
# #205 runs the MODERN store-nothing screw pose carrier (``--pose-carrier
# --pose-carrier-residual-mode table``), so the verdict's ``d_pose`` is the
# carrier's OWN measured pose — the honest composite, NOT a separate untrained
# monitoring signal. The DISPLAYED implied_S therefore uses that REAL measured
# d_pose everywhere; we NEVER substitute the old ancestor sidecar 3.4e-5 (which
# was never witness-validated). Byte-closed deploy-pose confirmation is task #238.
_ARCHIVE_NORM_BYTES = 37_545_489   # contest rate-term normalizer (25·bytes/N)


# ───────────────────────────── config ─────────────────────────────
@dataclass
class Config:
    run_dir: str = ""
    log_glob: str = ""
    # Stage boundaries are DERIVED per-run from the run's own config via the DSL
    # read-back (tac.witness_dsl.schedule_readback) — the class-fix for the
    # "--l7 600 while the run had l7 disabled at 1001" mislabel incident. None
    # (the default) = derive; an explicit CLI/env value is an OVERRIDE only.
    tau: int | None = None
    l7: int | None = None
    # d_seg goal lines are DERIVED per run (target_S − the run's own measured pose −
    # measured rate)/100 via _derive_goal_info — the "no hardcoded run constants in
    # consumers" class-fix (operator 2026-07-07). None (the default) = derive; an
    # explicit env/CLI value is an OVERRIDE only and renders with source "override".
    goal_dseg: float | None = None   # sub-0.19 d_seg goal line (override only)
    goal_dseg_15: float | None = None  # sub-0.15 d_seg goal line (override only)
    poll: float = 5.0
    host: str = "127.0.0.1"
    port: int = 8790
    access_key: str = ""
    cadence_state: str = ".omx/tmp/dash_levelset_deploy/cadence.json"
    training_pid: int = 0
    training_sig: str = "train_levelset_witness"
    # AUTO-LATEST (default ON): self-protect against showing a stale run. Instead of
    # being pinned to one --run-dir at launch, span ALL witness arm dirs and let the
    # newest-mtime verdict log win (the live arm). When a new arm starts, the dashboard
    # follows it automatically and resets to show ONLY that run — no repoint, no restart.
    auto_latest: bool = True
    auto_base_glob: str = "experiments/results/levelset_*/*.log,.omx/tmp/levelset_*.log"
    # WITNESS tab (Tab 2): live comma10k 6-panel + Yousfi/Fridrich tribute rendered FROM the
    # live EMA checkpoint, re-rendered on checkpoint-mtime change (NOT every tick), in the
    # tailer executor (never blocks the loop), broadcast over the SAME WebSocket. Light
    # (~2 GB peak, ~8 s CPU-only, gt_n6 aligned) — yields to #205 under a memory floor.
    # WITNESS (Tab 2) + FLOW (Tab 3) are fed by ONE heavy governed 600-pass over the BEST
    # checkpoint, run as a DETACHED subprocess (own process group + own safe_run cap) so its
    # ~2.6 GB torch+SegNet footprint is NEVER summed into THIS lean dashboard's safe_run RSS
    # group (it could never crash the dashboard or the machine). gt_n600 (all 600 pairs) is
    # required for the full-video timeline + the hardest/most-diverse Tab-2 pair selection.
    witness_enable: bool = True
    witness_gt_cache: str = "experiments/results/mlx_fleet_gt_cache/gt_n600.npz"
    witness_ema_name: str = "levelset_witness_ema_mlx.npz"  # fallback if BEST is absent
    witness_min_free_gib: float = 10.0   # skip spawning a pass when free RAM is below this — the L51 machine floor (yield to the live run)
    witness_dpi: int = 80
    flow_enable: bool = True
    flow_best_ema_name: str = "levelset_witness_ema_BEST.npz"  # the checkpoint the 600-pass renders
    flow_seq_downsample: int = 1         # FULL NATIVE 512x384 (max res given render_h/w). Safe now that the
                                         # client uses an LRU decode-on-demand window (dashboard_flow_client.js
                                         # CACHE_CAP) — decoded memory is O(CACHE_CAP) not O(600), so the old
                                         # ~472 MB all-frames-decoded blocker is gone (~96 MB decoded + ~60-80 MB
                                         # base64 source resident). Dial DASH_FLOW_SEQ_DOWNSAMPLE=2 for lighter mobile.
    flow_seq_jpeg_q: int = 85            # JPEG quality for the per-frame witness-render layer (sharper backdrop)
    flow_seq_frag_levels: int = 32       # margin-fragility quantization (PNG-compressibility)
    flow_seq_hard_k: int = 6             # hardest/most-diverse Tab-2 pairs to select
    flow_seq_min_interval_s: float = 900.0   # do not re-render within 15 min even if ckpt mtime flaps
    flow_seq_cache_dir: str = ".omx/tmp/dash_flow_seq"  # sanctioned repo-local ephemeral scratch
    flow_seq_rss_mb: int = 7000          # the subprocess's OWN safe_run RSS cap (isolated group)
    flow_seq_timeout_s: int = 2400       # the subprocess's OWN safe_run wall-clock cap (40 min)
    # ORACLE tab (Tab 1): "the detector I built, and the world it reads" — the frozen scorer +
    # the openpilot physical priors (lane band -> d_seg, ego-ξ screw -> d_pose) + the SegNet
    # detectability field. STATIC: depends only on the GT cache (never changes), so it renders
    # ONCE via a DETACHED governed safe_run subprocess (own process group + own RSS cap, CPU-only,
    # numpy+matplotlib, NO torch/SegNet/GPU) and is cached to disk. Light (~170 MB peak, ~2 s).
    oracle_enable: bool = True
    oracle_gt_cache: str = "experiments/results/mlx_fleet_gt_cache/gt_n6.npz"  # 6 pairs, ~48 MB
    oracle_frames: str = "0,2,4"         # representative frames for the physical-prior atlas
    oracle_cache_dir: str = ".omx/tmp/dash_oracle"  # sanctioned repo-local ephemeral scratch
    oracle_rss_mb: int = 2600            # the subprocess's OWN safe_run RSS cap (isolated group)
    oracle_timeout_s: int = 420          # the subprocess's OWN safe_run wall-clock cap (7 min)
    oracle_dpi: int = 80
    # WHY/HOW tab (Tab 4): the deep-math museum. PASS 1 = the tab shell + §I.1 "the live field"
    # + §I.4 "the Unity morph". STATIC single-frame FIELD BUNDLE (raw co-registered scalar fields:
    # ρ_seg margin / real S-UNIWARD / separatrix sensitivity) for the client-side WebGPU plates —
    # depends only on the GT cache, so it renders ONCE via a DETACHED governed safe_run subprocess
    # (own process group + own RSS cap; ~270 MB peak, ~1 s; imports torch ONLY for the one-frame
    # S-UNIWARD wavelet cost — NO SegNet forward, NO witness checkpoint) and is cached to disk.
    whyhow_enable: bool = True
    whyhow_gt_cache: str = "experiments/results/mlx_fleet_gt_cache/gt_n6.npz"  # 6 pairs, ~48 MB
    whyhow_frame: str = ""               # "" = richest-separatrix auto-pick; else a fixed index
    whyhow_cache_dir: str = ".omx/tmp/dash_whyhow"  # sanctioned repo-local ephemeral scratch
    whyhow_rss_mb: int = 3200            # the subprocess's OWN safe_run RSS cap (isolated group)
    whyhow_timeout_s: int = 420          # the subprocess's OWN safe_run wall-clock cap (7 min)

    def resolved_glob(self) -> str:
        if self.log_glob:               # explicit --log-glob is a hard override
            return self.log_glob
        if self.auto_latest:            # DEFAULT: follow the freshest arm across all dirs
            # (#420 contract) span EVERY contract-recognized run dir, re-derived at each
            # resolution so a NEW arm is auto-tracked regardless of its NAME. DRIFT LESSON
            # (2026-07-11): the levelset_-prefix base glob silently hid the first
            # v9_cgauge_* arm (and the owed16_ab_* arms) from the dashboard — a run-dir
            # name is never a discovery contract. The base glob stays as the union
            # fallback (covers .omx/tmp logs + survives a contract import failure).
            if _wra is not None:
                try:
                    results = Path("experiments/results")
                    dirs = sorted(d for d in results.iterdir() if _wra.is_run_dir(d))
                    if dirs:
                        pats = ",".join(f"{d}/*.log" for d in dirs)
                        return f"{pats},{self.auto_base_glob}"
                except Exception:
                    pass  # fall through to the static base glob (fail-open observability)
            return self.auto_base_glob
        if self.run_dir:                # pinned mode (--no-auto-latest --run-dir X)
            return os.path.join(self.run_dir, "*.log")
        return ".omx/tmp/levelset_*.log"


def _opt_int_env(v: str | None) -> int | None:
    """Optional int env: unset/empty/non-int -> None (= derive via the DSL read-back)."""
    try:
        return int(v) if v not in (None, "") else None
    except (TypeError, ValueError):
        return None


def _opt_float_env(v: str | None) -> float | None:
    """Optional float env: unset/empty/non-float -> None (= derive from measured data)."""
    try:
        return float(v) if v not in (None, "") else None
    except (TypeError, ValueError):
        return None


def config_from_env() -> Config:
    e = os.environ.get
    return Config(
        run_dir=e("DASH_RUN_DIR", ""),
        log_glob=e("DASH_LOG_GLOB", ""),
        tau=_opt_int_env(e("DASH_TAU")),
        l7=_opt_int_env(e("DASH_L7")),
        goal_dseg=_opt_float_env(e("DASH_GOAL_DSEG")),
        goal_dseg_15=_opt_float_env(e("DASH_GOAL_DSEG_15")),
        poll=float(e("DASH_POLL", "5.0")),
        host=e("DASH_HOST", "127.0.0.1"),
        port=int(e("DASH_PORT", "8790")),
        access_key=e("DASH_ACCESS_KEY", ""),
        cadence_state=e("DASH_CADENCE_STATE", ".omx/tmp/dash_levelset_deploy/cadence.json"),
        training_pid=int(e("DASH_TRAINING_PID", "0")),
        training_sig=e("DASH_TRAINING_SIG", "train_levelset_witness"),
        auto_latest=e("DASH_AUTO_LATEST", "1") not in ("0", "false", "False"),
        auto_base_glob=e("DASH_AUTO_BASE_GLOB", "experiments/results/levelset_*/*.log,.omx/tmp/levelset_*.log"),
        witness_enable=e("DASH_WITNESS_ENABLE", "1") not in ("0", "false", "False"),
        witness_gt_cache=e("DASH_WITNESS_GT_CACHE", "experiments/results/mlx_fleet_gt_cache/gt_n600.npz"),
        witness_ema_name=e("DASH_WITNESS_EMA_NAME", "levelset_witness_ema_mlx.npz"),
        witness_min_free_gib=float(e("DASH_WITNESS_MIN_FREE_GIB", "10.0")),
        witness_dpi=int(e("DASH_WITNESS_DPI", "80")),
        flow_enable=e("DASH_FLOW_ENABLE", "1") not in ("0", "false", "False"),
        flow_best_ema_name=e("DASH_FLOW_BEST_EMA_NAME", "levelset_witness_ema_BEST.npz"),
        flow_seq_downsample=int(e("DASH_FLOW_SEQ_DOWNSAMPLE", "1")),  # full native 512x384 (LRU client cache bounds memory)
        flow_seq_jpeg_q=int(e("DASH_FLOW_SEQ_JPEG_Q", "85")),
        flow_seq_frag_levels=int(e("DASH_FLOW_SEQ_FRAG_LEVELS", "32")),
        flow_seq_hard_k=int(e("DASH_FLOW_SEQ_HARD_K", "6")),
        flow_seq_min_interval_s=float(e("DASH_FLOW_SEQ_MIN_INTERVAL_S", "900.0")),
        flow_seq_cache_dir=e("DASH_FLOW_SEQ_CACHE_DIR", ".omx/tmp/dash_flow_seq"),
        flow_seq_rss_mb=int(e("DASH_FLOW_SEQ_RSS_MB", "7000")),
        flow_seq_timeout_s=int(e("DASH_FLOW_SEQ_TIMEOUT_S", "2400")),
        oracle_enable=e("DASH_ORACLE_ENABLE", "1") not in ("0", "false", "False"),
        oracle_gt_cache=e("DASH_ORACLE_GT_CACHE", "experiments/results/mlx_fleet_gt_cache/gt_n6.npz"),
        oracle_frames=e("DASH_ORACLE_FRAMES", "0,2,4"),
        oracle_cache_dir=e("DASH_ORACLE_CACHE_DIR", ".omx/tmp/dash_oracle"),
        oracle_rss_mb=int(e("DASH_ORACLE_RSS_MB", "2600")),
        oracle_timeout_s=int(e("DASH_ORACLE_TIMEOUT_S", "420")),
        oracle_dpi=int(e("DASH_ORACLE_DPI", "80")),
        whyhow_enable=e("DASH_WHYHOW_ENABLE", "1") not in ("0", "false", "False"),
        whyhow_gt_cache=e("DASH_WHYHOW_GT_CACHE", "experiments/results/mlx_fleet_gt_cache/gt_n6.npz"),
        whyhow_frame=e("DASH_WHYHOW_FRAME", ""),
        whyhow_cache_dir=e("DASH_WHYHOW_CACHE_DIR", ".omx/tmp/dash_whyhow"),
        whyhow_rss_mb=int(e("DASH_WHYHOW_RSS_MB", "3200")),
        whyhow_timeout_s=int(e("DASH_WHYHOW_TIMEOUT_S", "420")),
    )


# The trajectory-point schema: the ONLY keys refresh() ships (no full-verdict leak).
# _slim() builds EXACTLY these — single source of truth, so the declared schema and
# the actual output can never drift (test_refresh_slims_to_traj_keys_only enforces ==).
_TRAJ_KEYS = (
    "epoch", "d_seg", "d_pose", "blob_bytes", "ts",
    "implied_S", "implied_S_monitoring",
    # LIVE-tab per-point diagnostics (2026-07-09 rebuild) ride the slimmed trajectory:
    "d_seg_by_class", "flip_share_by_class", "seg_form",
    "rss_gib", "sys_avail_gib", "mlx_active_gib", "mlx_peak_gib",
    "mlx_cache_gib", "accepted_frac", "weights_stepped",
    "accepted_batches", "skipped_batches",
)


def _last_measured_dpose(rows):
    """Most-recent non-None ``d_pose`` from the trajectory as a float (None if the
    run has emitted no pose yet). #205's store-nothing screw carrier measures its
    OWN pose in-run, so this feeds the forward implied_S projection with the REAL
    measured pose — never the old ancestor sidecar 3.4e-5."""
    for row in reversed(rows or []):
        dp = row.get("d_pose") if isinstance(row, dict) else None
        if dp is None:
            continue
        try:
            return float(dp)
        except (TypeError, ValueError):
            continue
    return None


def _last_measured_row(rows):
    """Most-recent row carrying finite blob_bytes (and d_pose where present) — the
    measured basis the derived goal lines are computed from. None if nothing measured."""
    for row in reversed(rows or []):
        if not isinstance(row, dict):
            continue
        by = row.get("blob_bytes")
        if isinstance(by, (int, float)) and by > 0:
            return row
    return None


def _derive_goal_info(rows, pose_blind, explicit: float | None,
                      target_s: float, archive_norm: float) -> dict:
    """One d_seg goal line, CONDITIONALLY CALCULATED — never a baked default.

    Sources, in order:
      * explicit env/CLI value      -> {"value", "source": "override(env/cli)"}
      * derived from the run's OWN measured terms:
        goal_dseg = (target_S − pose_term − 25·bytes/N) / 100, where pose_term is the
        run's measured √(10·d_pose) — or 0 for a pose-blind arm (w_pose=0: pose is
        UNHELD BY DESIGN, so the arm's d_seg goal is judged against target − rate).
      * nothing measured yet        -> {"value": None, "source": None} (rendered "—").
    A non-positive derived value means the target is unreachable at the measured
    pose+rate; the value is withheld (None) with an explicit source note."""
    if explicit is not None:
        return {"value": float(explicit), "source": "override(env/cli)"}
    row = _last_measured_row(rows)
    if row is None:
        return {"value": None, "source": None}
    try:
        rate = 25.0 * float(row["blob_bytes"]) / float(archive_norm)
        dp = row.get("d_pose")
        if pose_blind:
            pose_term, pose_note = 0.0, "pose unheld in this arm (w_pose=0)"
        elif isinstance(dp, (int, float)) and dp >= 0:
            pose_term, pose_note = (10.0 * float(dp)) ** 0.5, "measured pose"
        else:
            return {"value": None, "source": "no measured pose yet"}
        value = (float(target_s) - pose_term - rate) / 100.0
        if value <= 0:
            return {"value": None,
                    "source": f"target {target_s:g} unreachable at measured pose+rate"}
        return {"value": value,
                "source": f"derived: (target {target_s:g} − {pose_note} − measured rate)/100"}
    except Exception:
        return {"value": None, "source": None}


def _last_jsonl_row_tail(path: Path, tail_bytes: int = 262144) -> dict | None:
    """Last parseable JSON object of a JSONL file, reading only the tail block
    (the costate shadow file grows to ~1 MB; a 5 s tick must not re-read it all).
    None on any failure (fail-open)."""
    try:
        size = path.stat().st_size
        with path.open("rb") as fh:
            if size > tail_bytes:
                fh.seek(size - tail_bytes)
                fh.readline()  # drop the (possibly clipped) partial first line
            last = None
            for raw in fh:
                if raw.strip():
                    last = raw
        row = json.loads(last) if last else None
        return row if isinstance(row, dict) else None
    except Exception:
        return None


def _read_costate(run_dir: str | None) -> dict | None:
    """SENSE-only source for the LIVE-tab costate panel: the last row of the run's
    ``costate_shadow.jsonl`` (written by the score-neutral shadow observer,
    tools/costate_observer_loop.py -> tac.witness_control) — the same row schema
    tools/costate_digest.py reads. READ-ONLY; the dashboard NEVER actuates
    (CONTAINMENT). No shadow file -> None -> the panel is absent (conditional)."""
    if not run_dir:
        return None
    path = Path(run_dir) / _COSTATE_JSONL
    if not path.is_file():
        return None
    row = _last_jsonl_row_tail(path)
    if not row:
        return None
    try:
        age_s = max(0.0, time.time() - path.stat().st_mtime)
        cls = (row.get("classification") or {}).get("classification")
        recs = row.get("recommendations") or []
        rec = None
        if recs and isinstance(recs[0], dict):
            r0 = recs[0]
            rec = {"action": r0.get("action"),
                   "predicted_dS": r0.get("predicted_dS"),
                   "horizon_epochs": r0.get("horizon_epochs")}
        # duty-to-measure queue — the digest's mechanism (tac activation ledger),
        # already materialized into the shadow row by tac.witness_control (REUSE,
        # never reimplemented here).
        duty = row.get("duty_to_measure")
        duty_owed = len(duty) if isinstance(duty, list) else None
        duty_nf = (sum(1 for d in duty if isinstance(d, dict)
                       and d.get("state") == "never-fired")
                   if isinstance(duty, list) else None)
        factor = row.get("factorized_adjoint")
        factor_summary = None
        if isinstance(factor, dict):
            fac = factor.get("factorization") or {}
            factor_summary = {
                "admission": factor.get("admission"),
                "head_rank": (fac.get("exact") or {}).get("head_rank"),
                "zero_weight_camera_frac": (fac.get("exact") or {}).get(
                    "certified_zero_weight_camera_frac"),
                "road_lane_lambda_ratio": (fac.get("derived") or {}).get(
                    "road_lane_gain_only_lambda_ratio_vs_other_median"),
                "learned_parameters": (factor.get("learned_residual") or {}).get(
                    "n_parameters"),
                "amplitude_gate": (factor.get("learned_residual") or {}).get(
                    "amplitude_gate"),
                "predicted_dS": (factor.get("decision") or {}).get("predicted_dS"),
                "why": (factor.get("decision") or {}).get("why"),
                "confidence": ((factor.get("recommendation_candidate") or {}).get(
                    "confidence") or factor.get("validation_scope")),
            }
        return {"ok": True, "epoch": row.get("epoch"),
                "classification": (str(cls).upper() if cls else None),
                "rec": rec, "age_s": age_s,
                "duty_owed": duty_owed, "duty_never_fired": duty_nf,
                "factorized_adjoint": factor_summary,
                "event_advisories": row.get("event_advisories") or []}
    except Exception:
        return None


_DDM_CAMPAIGN_CACHE: dict[str, object] = {}


def _ddm_campaign_source_signature() -> tuple[tuple[str, int, int], ...]:
    """Cheap mtime/size gate for the immutable receipt inputs."""

    from tac.ddm_campaign_costate import J8F_GLOBS, SOURCES
    from tac.ddm_costate_organ import SOURCE_SPECS

    paths = [_REPO_ROOT / spec.path for spec in SOURCES]
    research = _REPO_ROOT / ".omx" / "research"
    paths.extend(
        path
        for spec in SOURCE_SPECS
        for path in sorted(research.glob(spec.glob))[-1:]
    )
    paths.extend(
        path
        for pattern in J8F_GLOBS
        for path in sorted(_REPO_ROOT.glob(pattern))[-1:]
    )
    return tuple(
        sorted(
            (
                str(path),
                path.stat().st_size if path.is_file() else -1,
                path.stat().st_mtime_ns if path.is_file() else -1,
            )
            for path in paths
        )
    )


def _read_ddm_campaign() -> dict | None:
    """Read the campaign dashboard view from the canonical campaign composer.

    The cache key is the campaign lineage digest, obtained from the same
    schema/hash-validated state used by ``tools/costate_digest.py``.  This
    remains advisory and has no launcher/provider imports.
    """

    try:
        from tac.ddm_campaign_costate import campaign_consumer_view
        from tac.ddm_costate_organ import build_live_ddm_costate

        signature = _ddm_campaign_source_signature()
        if _DDM_CAMPAIGN_CACHE.get("signature") == signature:
            cached = _DDM_CAMPAIGN_CACHE.get("view")
            return dict(cached) if isinstance(cached, dict) else None
        organ = build_live_ddm_costate(repo_root=_REPO_ROOT)
        if not organ.get("available"):
            return None
        state = organ["campaign"]
        view = campaign_consumer_view(state, "dashboard")
        _DDM_CAMPAIGN_CACHE.clear()
        _DDM_CAMPAIGN_CACHE.update(
            {
                "signature": signature,
                "digest": str(state["state_digest"]),
                "view": dict(view),
            }
        )
        return view
    except Exception:
        return None


def _read_identity_header(run_dir) -> dict:
    """The RUN-IDENTITY header the launcher stamps into the run dir's config record
    (launch.sh ``# tac-run-purpose:`` / ``# tac-config-family:`` comment lines,
    tools/launch_witness_run.py::_identity_header). Fail-open {} (old run dirs)."""
    out: dict = {}
    try:
        for line in (Path(run_dir) / "launch.sh").read_text(errors="replace").splitlines():
            s = line.strip()
            if s.startswith("# tac-run-purpose:"):
                out["purpose"] = s.split(":", 1)[1].strip()
            elif s.startswith("# tac-config-family:"):
                out["family"] = s.split(":", 1)[1].strip()
    except Exception:
        return out
    return out


def _derive_run_identity(run_dir, flags, pose_blind, resumed_from) -> dict | None:
    """RUN-IDENTITY row payload (operator 2026-07-07: "add a label to the top with
    the run name and possibly a description of its intended purpose; clean baseline
    or frontier score lowering? a/b probe? Is pose included or just seg?").

    Conditional-rendering discipline: name = the resolved run dir's basename; the
    SCOPE chip derives from the run's OWN launched --w-pose; the PURPOSE chip is the
    launch.sh ``# tac-run-purpose`` header VERBATIM with provenance "declared" when
    present, else a best-effort classification LABELLED "derived" with its evidence
    (a guess is never rendered as a declaration). No run dir -> None -> no row."""
    if not run_dir:
        return None
    name = Path(run_dir).name
    ident: dict = {"name": name}
    flags = flags or {}

    # ── scope chip: pose-held vs seg-only, from the run's own config ──
    wp = flags.get("w-pose")
    if pose_blind is True:
        ident["scope"] = {"label": "seg-only · pose unheld by design (w_pose=0)",
                          "evidence": "launch.sh: --w-pose 0"}
    elif pose_blind is False and wp is not None:
        ident["scope"] = {"label": f"seg+pose (w_pose={wp})",
                          "evidence": f"launch.sh: --w-pose {wp}"}
    # pose_blind None (config unknown) -> no scope chip (conditional)

    # ── purpose chip: DECLARED header wins; else the LABELLED derived heuristic ──
    hdr = _read_identity_header(run_dir)
    if hdr.get("purpose"):
        ident["purpose"] = {"label": hdr["purpose"], "provenance": "declared",
                            "evidence": ["# tac-run-purpose header in launch.sh"]}
        return ident
    evidence: list[str] = []
    if hdr.get("family"):
        evidence.append(f"config family '{hdr['family']}' (launch.sh header)")
    islands_on = "seed-islands" in flags
    try:
        eik = float(flags.get("eikonal-weight", 0) or 0)
    except (TypeError, ValueError):
        eik = 0.0
    # a FOREIGN resume (stage checkpoint from another run dir) signals a treatment
    # arm; a same-dir resume is crash recovery and classifies by its levers instead.
    foreign_resume = None
    if resumed_from:
        try:
            if Path(resumed_from).resolve().parent != Path(run_dir).resolve():
                foreign_resume = str(resumed_from)
        except Exception:
            foreign_resume = str(resumed_from)
    cap = re.search(r"mod(\d+)cap", name)
    if foreign_resume:
        label = "A/B arm (resumed treatment)"
        evidence.append(f"resumes from a foreign stage checkpoint: {foreign_resume}")
    elif not islands_on and eik == 0.0 and flags:
        label = "clean baseline / control"
        evidence.append("islands levers off (no --seed-islands; eikonal-weight 0)")
        if cap:
            evidence.append(f"capacity-capped family name '{cap.group(0)}'")
        if flags.get("mod-dim") is not None:
            evidence.append(f"--mod-dim {flags['mod-dim']}")
    elif flags:
        label = "frontier candidate"
        if islands_on:
            evidence.append("--seed-islands present (island levers ON)")
        if eik:
            evidence.append(f"eikonal-weight {eik:g} active")
        evidence.append("fresh full-stack with levers on")
    else:
        # no parsed flags at all (no launch.sh, no run.log config) -> no honest basis
        label = "unclassified (no config record)"
    ident["purpose"] = {"label": label, "provenance": "derived", "evidence": evidence}
    return ident


def _stage_windows(sched: dict) -> list[tuple[str, int, int]]:
    """Ordered, NON-overlapping ``[(stage, start, end)]`` epoch windows covering
    ``[0, epochs)`` — delegates to the canonical ``dtm._stage_segments``.

    ``dtm._stage_segments`` was fixed 2026-07-03 to derive windows from the
    precedence-correct ``stage_at_epoch`` step function, so out-of-order / disabled
    boundaries — e.g. this run's ``muon_start`` 726 < the disabled ``l7_start`` 1000 —
    can no longer produce overlapping windows (previously tau [300,1000) overlapped
    Muon [726,1000)). Kept as a thin local alias so callers read against the schedule.
    Pure; empty list when the schedule carries no usable epoch count."""
    try:
        ep = int(sched.get("epochs") or 0)
    except (TypeError, ValueError):
        ep = 0
    return dtm._stage_segments(sched, ep)


def _build_stage_aware_projection(rows, sched: dict, *, sidecar_pose: float,
                                  archive_norm: float,
                                  pose_blind: bool = False) -> dict:
    """STAGE-AWARE d_seg / implied_S projection (ADVISORY; ``[macOS-MLX]`` non-promotable).

    Operator-approved 2026-07-03. The prior GLOBAL critical-slowing power-law (in
    ``dtm.build_projection``) fits ALL verdicts as ONE curve and, from the CE flicker
    plateau, mis-declares "sub-0.19 won't reach". That is wrong: the run is a 3-stage
    curriculum. This fits the critical-slowing model PER CURRICULUM STAGE (reset at each
    boundary), reports each ENTERED stage's OWN asymptote as that stage's floor, and marks
    the tau (lane-band birth) + Muon (finishing) boundaries as EXPECTED-BREAKTHROUGH
    regimes the CE fit does NOT model. Muon is a saddle-to-saddle STAIRCASE (polynomial
    escape, #217/MFLD), NEVER a power-law — labelled unmodeled-until-measured, never
    extrapolated.

    NO-FAKE: no fabricated stage-boundary drop is invented. A stage with fewer than
    ``dtm._MIN_FIT_POINTS`` verdicts reports 'insufficient' (never a borrowed CE number);
    an un-entered stage reports 'not entered'. The frontier pointer moves ONLY
    through a byte-closed exact eval — this is a MEANS. Pure; never raises."""
    try:
        verdicts = [v for v in (rows or []) if isinstance(v, dict)
                    and isinstance(v.get("epoch"), (int, float))
                    and isinstance(v.get("d_seg"), (int, float))]
        if not verdicts:
            return {"ok": False, "reason": "no d_seg verdicts yet"}
        cur_ep = float(max(v["epoch"] for v in verdicts))
        cur_stage = dtm.stage_at_epoch(cur_ep, sched)
        windows = _stage_windows(sched)
        min_fit = int(dtm._MIN_FIT_POINTS)  # >=5 verdicts before a stage asymptote is claimed
        # group verdicts by the CORRECT (precedence-ordered) stage, never the overlapping segments
        by_stage: dict[str, list] = {}
        for v in verdicts:
            by_stage.setdefault(dtm.stage_at_epoch(float(v["epoch"]), sched), []).append(v)

        stages: list[dict] = []
        modeled: dict | None = None  # latest ENTERED stage with a real fit -> best-modeled floor
        iter_windows = windows or [(cur_stage, 0, int(sched.get("epochs") or 0))]
        for name, a, b in iter_windows:
            svs = sorted(by_stage.get(name, []), key=lambda v: v["epoch"])
            n = len(svs)
            rec: dict = {"name": name, "start": int(a), "end": int(b), "n": n, "entered": n > 0}
            if name == "Muon":
                rec["model"] = "saddle_staircase"
                rec["note"] = ("saddle-to-saddle (polynomial escape) — not power-law; "
                               f"unmodeled until measured @{int(a)}")
                if n:
                    rec["observed_min"] = float(min(v["d_seg"] for v in svs))
                    # OBSERVED quantitative trend (operator scope 2026-07-07): Muon still
                    # refuses the power-law FORM, but its MEASURED verdicts carry a real
                    # read — recent slope over the last k verdicts + a linear read-through
                    # to stage end, LABELLED "observed trend, not a fit" (no asymptote
                    # claim is ever made from it).
                    recent = svs[-min(n, 6):]
                    if len(recent) >= 2 and recent[-1]["epoch"] > recent[0]["epoch"]:
                        xs = [float(v["epoch"]) for v in recent]
                        ys = [float(v["d_seg"]) for v in recent]
                        nk = float(len(recent))
                        sx_, sy_ = sum(xs), sum(ys)
                        sxx = sum(x * x for x in xs)
                        sxy = sum(x * y for x, y in zip(xs, ys, strict=True))  # same source list
                        den = nk * sxx - sx_ * sx_
                        if den > 0:
                            m = (nk * sxy - sx_ * sy_) / den
                            end_ep = float(b) if b else xs[-1]
                            last_val = float(svs[-1]["d_seg"])
                            rec["trend"] = {
                                "n_recent": len(recent),
                                "slope_per_25ep": m * 25.0,
                                "readthrough_epoch": int(end_ep),
                                "readthrough_dseg": max(
                                    last_val + m * (end_ep - xs[-1]), 0.0),
                                "label": "observed trend, not a fit",
                            }
                stages.append(rec)
                continue
            if n == 0:
                rec["note"] = "not entered yet"
                stages.append(rec)
                continue
            rec["observed_min"] = float(min(v["d_seg"] for v in svs))
            if n < min_fit:
                rec["fit_state"] = "insufficient"
                rec["min_needed"] = min_fit
                rec["note"] = (f"{n} verdict{'' if n == 1 else 's'} — insufficient for a "
                               f"stage fit yet (need >= {min_fit})")
                stages.append(rec)
                continue
            fit = dtm.fit_critical_slowing([v["epoch"] for v in svs],
                                           [v["d_seg"] for v in svs], min_points=min_fit)
            if fit.get("ok"):
                rec["fit_state"] = "ok"
                rec.update({"asymptote": fit["asymptote"], "alpha": fit["alpha"],
                            "r2": fit["r2"], "confidence": fit["confidence"],
                            "residual_std": fit.get("residual_std", 0.0)})
                modeled = rec
            else:
                rec["fit_state"] = "no_fit"
                rec["note"] = fit.get("reason", "no decaying fit")
            stages.append(rec)

        # implied_S from the current best-modeled stage floor — LABELLED (in the renderer)
        # as a current-stage extrapolation that EXCLUDES the downstream tau/Muon breakthroughs.
        modeled_floor: dict | None = None
        if modeled is not None:
            asym = float(modeled["asymptote"])
            sig = float(modeled.get("residual_std", 0.0) or 0.0)
            band = (max(asym - sig, 0.0), asym + sig)
            total = int(sched.get("epochs") or 0)
            bytes_proj = dtm.project_bytes(verdicts, float(total) if total else None)
            s_proj = dtm.project_implied_s(asym, bytes_proj.get("value"), sidecar_pose,
                                           archive_norm, dseg_band=band)
            modeled_floor = {"stage": modeled["name"], "asymptote": asym,
                             "observed_min": modeled.get("observed_min"),
                             "is_current_stage": (modeled["name"] == cur_stage),
                             "implied_s": s_proj,
                             # pose-blind arms (w_pose=0, from the RUN'S OWN config):
                             # the unheld pose term would swamp the composite, so the
                             # renderer leads with the d_seg TERM contribution and
                             # demotes/labels the composite — never the bare number.
                             "pose_blind": bool(pose_blind),
                             "seg_term": 100.0 * asym,
                             "seg_term_lo": 100.0 * band[0],
                             "seg_term_hi": 100.0 * band[1]}

        # expected-breakthrough boundaries the current fit does NOT model
        tau = sched.get("tau_start")
        muon = sched.get("muon_start")
        downstream: list[dict] = []
        if isinstance(tau, (int, float)):
            downstream.append({"epoch": int(tau), "label": "lane-band birth (tau)",
                               "status": "engaged" if cur_ep >= tau else "pending"})
        if isinstance(muon, (int, float)):
            downstream.append({"epoch": int(muon), "label": "Muon finishing (saddle staircase)",
                               "status": "engaged" if cur_ep >= muon else "pending"})

        return {"ok": True, "current_stage": cur_stage, "current_epoch": cur_ep,
                "stages": stages, "modeled_floor": modeled_floor, "downstream": downstream,
                "tau_start": (int(tau) if isinstance(tau, (int, float)) else None),
                "muon_start": (int(muon) if isinstance(muon, (int, float)) else None),
                "epochs": int(sched.get("epochs") or 0)}
    except Exception as exc:  # never crash the live telemetry
        return {"ok": False, "reason": f"stage projection error: {exc}"}


def _flowseq_lock_alive(lock_path: Path) -> bool:
    """Is a flow-seq render for this stem already in progress (possibly under ANOTHER dashboard
    instance — a reload/supervisor restart mid-render leaves the prior render detached)? The lock
    file holds the render's process-group pid; a live pid means "someone is rendering this stem".
    A stale lock (dead pid / unreadable) is removed and treated as free. This makes the spawn
    idempotent across instances so restarts never pile up duplicate 16-min renders."""
    try:
        pid = int(lock_path.read_text().strip())
    except Exception:
        with contextlib.suppress(Exception):
            if lock_path.exists():
                lock_path.unlink()
        return False
    if _pid_alive(pid):
        return True
    with contextlib.suppress(Exception):
        lock_path.unlink()  # stale -> free it
    return False


def _slim(row: dict) -> dict:
    """Keep only the trajectory fields the client charts need (no leaking of the
    full verdict dict). The DISPLAYED ``implied_S`` is the run's OWN measured
    composite — ``100·d_seg + sqrt(10·d_pose) + 25·bytes/N`` with the
    store-nothing screw carrier's measured d_pose — with NO deploy override.
    ``implied_S_monitoring`` is kept as an alias of the same measured value so
    the client's honest-vs-displayed sanity check trivially agrees. Numbers stay
    numeric; missing keys become None.

    The rebuilt LIVE instrument (2026-07-09) reads per-class d_seg / flip-share
    and the system-memory row off the LATEST verdict point, so those fields ride
    the slimmed trajectory too (5-float arrays + a handful of scalars per point —
    a few KB across the whole run; still no leak of the full verdict dict). All
    score-neutral: the dashboard only READS the log."""
    out = {k: row.get(k) for k in _TRAJ_KEYS}
    # implied_S_monitoring is an ALIAS of the measured implied_S (so the client's
    # honest-vs-displayed sanity check trivially agrees); both read the same field.
    out["implied_S_monitoring"] = row.get("implied_S")
    return out


_SENSOR_STAGES = ("jacobian_basin", "loss_terms")


def _read_sensors(latest_log: Path | None, tail_bytes: int = 524288) -> dict:
    """Latest ``jacobian_basin`` + ``loss_terms`` record off the live run-log tail.

    These two stages are NOT verdicts (they flow on their own cadence — basin at
    the jacobian probe, loss_terms every accum batch), so they never reach the
    client through the verdict trajectory. The rebuilt LIVE tab's pose-readiness
    and training-health panels need them, so we tail-read the run-log (last block
    only — the log grows to MBs across a 3000-epoch run; a 5 s tick must not re-read
    it whole) and return the LAST record of each stage. READ-ONLY, score-neutral;
    fail-open to {} so a parse error never kills the daemon. Empty dict -> the
    client renders those panels as 'no reading yet' (conditional)."""
    out: dict = {}
    if latest_log is None:
        return out
    try:
        size = latest_log.stat().st_size
        with latest_log.open("rb") as fh:
            if size > tail_bytes:
                fh.seek(size - tail_bytes)
                fh.readline()  # drop the (possibly clipped) partial first line
            block = fh.read()
        text = block.decode("utf-8", "replace")
    except Exception:
        return out
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        # cheap prefilter before the json parse (the log is dense with other stages)
        if '"loss_terms"' not in line and '"jacobian_basin"' not in line:
            continue
        try:
            d = json.loads(line)
        except Exception:
            continue
        st = d.get("stage")
        if st in _SENSOR_STAGES:
            out[st] = d  # later line wins -> the LAST record of each stage
    return out


# ─────────── c2-era run-event scanner (engage markers · confound alarms · warm-start · clip · rate) ───────────
# The c2_surgical_warm era made EVENT BOUNDARIES the story: phase-stack engage (ep700), Muon
# (ep726), event-gated pose finish, warm-start origin (ep651 <- mod32cap). Engage/alarm rows are
# emitted ONCE at their epoch, so a bounded tail (the _read_sensors pattern) forgets them within
# hours on a multi-day run. This scanner reads each log INCREMENTALLY (per-path byte offset;
# O(new bytes) per tick, one full pass when a log is first seen) and ACCUMULATES:
#   markers  — {epoch,label,stage} engage/transition events -> chart vertical markers
#   alarms   — {epoch,kind} confound_alarm / term_inert rows -> LIVE-tab alarm strip
#   warm     — {start_epoch,ckpt_epoch,source} warm-start origin -> chart origin marker
#   clip     — LATEST grad_clip_activation row (global + per-group frac_clipped)
#   rate     — LATEST rate_rolling soft-signal row (producer landed; emission site queued —
#              handled here so the panel lights up the moment the trainer wires it)
# READ-ONLY + fail-open: any error leaves the accumulated state unchanged.
_EVENT_SCAN_TOKENS = ('_engage"', '"lever_engage"', '"confound_alarm"', '"muon_finisher_switch"',
                      '"curriculum_transition_fired"', '"warm_start', '"resume_start_epoch"',
                      '"grad_clip_activation"', '"rate_rolling"', '"term_inert"')
_EVENT_SCAN_CHUNK = 8 * 1024 * 1024  # per-tick read bound (a fresh multi-GB log cannot stall a tick)


def _engage_label(stage: str) -> str:
    """seg_phase_advect_engage -> 'phase advect' (short chart-marker label)."""
    s = stage[:-len("_engage")] if stage.endswith("_engage") else stage
    for pre in ("seg_", "lane_", "witness_"):
        if s.startswith(pre) and len(s) > len(pre):
            s = s[len(pre):]
            break
    return s.replace("_", " ")


def _event_scan_row(d: dict, acc: dict) -> None:
    """Classify one parsed stage row into the per-log accumulator (mutates acc)."""
    st = d.get("stage")
    if not isinstance(st, str):
        return
    ep = d.get("epoch", d.get("ep"))
    ep = int(ep) if isinstance(ep, (int, float)) else None
    if st == "confound_alarm" or st == "term_inert":
        kind = d.get("alarm") or d.get("kind") or st
        acc["alarms"].append({"epoch": ep, "kind": str(kind),
                              "term": d.get("term"), "ts": d.get("ts"),
                              "src": acc.get("src")})
    elif st == "grad_clip_activation":
        acc["clip"] = d                      # latest row wins (per-epoch aggregation)
    elif st == "rate_rolling":
        acc["rate"] = d                      # latest soft-signal row wins
    elif st == "warm_start_weights_only":
        acc["warm"] = {"start_epoch": d.get("start_epoch"), "ckpt_epoch": d.get("ckpt_epoch"),
                       "mode": "weights-only"}
    elif st == "resume_start_epoch":
        w = acc.get("warm") or {}
        w.setdefault("start_epoch", d.get("resume_start_epoch"))
        w.setdefault("ckpt_epoch", d.get("resume_ckpt_epoch"))
        if d.get("warm_start_override"):
            w.setdefault("mode", "weights-only")
        acc["warm"] = w
    elif st == "lever_engage":
        # only FIRES are chart events — "armed" rows are typed-config state at ep1
        # (12 of them at boot would bury the real ep700 engage cluster), and inert
        # engagements are explicitly non-mechanism-bearing.
        status = d.get("status")
        if (status is None or status == "fired") and not d.get("inert"):
            lab = str(d.get("lever") or "lever").replace("_", " ")
            acc["markers"].append({"epoch": ep, "label": lab, "stage": st,
                                   "via": d.get("via")})
    elif st == "muon_finisher_switch":
        acc["markers"].append({"epoch": ep, "label": "Muon finisher", "stage": st})
    elif st == "curriculum_transition_fired":
        lab = str(d.get("to") or d.get("seg_form") or "transition fired")
        acc["markers"].append({"epoch": ep, "label": lab, "stage": st})
    elif st.endswith("_engage"):
        acc["markers"].append({"epoch": ep, "label": _engage_label(st), "stage": st})


def _scan_log_events(path: Path, cache: dict) -> dict:
    """Incremental single-log scan. cache[str(path)] holds {pos, markers, alarms, warm,
    clip, rate}; only bytes past pos are read each call (full pass on first sight or on
    truncation/rotation). Returns the accumulator (fail-open: stale state on error)."""
    key = str(path)
    acc = cache.get(key)
    if acc is None:
        # src = the run this log belongs to (alarm scoping context): the parent dir
        # name for in-run-dir logs, the log stem for .omx/tmp tee logs.
        parent = path.parent.name
        src = parent if parent not in ("tmp", ".omx") else path.stem
        acc = {"pos": 0, "markers": [], "alarms": [], "warm": None, "clip": None,
               "rate": None, "src": src}
        cache[key] = acc
    try:
        size = path.stat().st_size
    except OSError:
        return acc
    if size < acc["pos"]:                    # truncated/rotated -> rescan
        acc.update({"pos": 0, "markers": [], "alarms": [], "warm": None,
                    "clip": None, "rate": None})
    if size == acc["pos"]:
        return acc
    try:
        with path.open("rb") as fh:
            fh.seek(acc["pos"])
            block = fh.read(min(size - acc["pos"], _EVENT_SCAN_CHUNK))
        # only advance past COMPLETE lines (a partially-flushed row is re-read next tick)
        cut = block.rfind(b"\n")
        if cut < 0:
            return acc
        acc["pos"] += cut + 1
        text = block[:cut + 1].decode("utf-8", "replace")
    except OSError:
        return acc
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("{") or not any(t in line for t in _EVENT_SCAN_TOKENS):
            continue
        try:
            d = json.loads(line)
        except Exception:
            continue
        try:
            _event_scan_row(d, acc)
        except Exception:
            continue
    return acc


def _merge_run_events(accs: list[dict], max_markers: int = 80, max_alarms: int = 16) -> dict:
    """Merge per-log accumulators (chain order root..latest; later warm/clip/rate wins)."""
    markers: list[dict] = []
    alarms: list[dict] = []
    warm = clip = rate = None
    for a in accs:
        markers.extend(a.get("markers") or [])
        alarms.extend(a.get("alarms") or [])
        warm = a.get("warm") or warm
        clip = a.get("clip") or clip
        rate = a.get("rate") or rate
    # de-dup markers by (epoch,label); keep epoch order (epochless markers sort first)
    seen: set = set()
    uniq = []
    for m in sorted(markers, key=lambda m: (m.get("epoch") is None, m.get("epoch") or 0)):
        k = (m.get("epoch"), m.get("label"))
        if k in seen:
            continue
        seen.add(k)
        uniq.append(m)
    return {"markers": uniq[-max_markers:], "alarms": alarms[-max_alarms:],
            "warm_start": warm, "clip": clip, "rate": rate}


# ─────────── chain-state strip (bench -> receipt -> launch -> run -> byte-close) ───────────
def _newest_launch_dir() -> Path | None:
    """The newest run-FAMILY dir by launch.sh mtime — INDEPENDENT of run.log existence, so a
    pre-launch bench dir (launch.sh written, dry-start in flight, no run.log yet) is visible
    on the dashboard BEFORE the real run fires (the c2 pre-launch window). Keys on the
    launcher-written STRUCTURE (launch.sh + launch_manifest.json), never a name pattern."""
    best, best_mt = None, -1.0
    try:
        for d in Path("experiments/results").iterdir():
            ls = d / "launch.sh"
            if not (d.is_dir() and ls.is_file() and (d / "launch_manifest.json").is_file()):
                continue
            mt = ls.stat().st_mtime
            if mt > best_mt:
                best, best_mt = d, mt
    except OSError:
        return None
    return best


def _chain_state(watched_dir: str | None, liveness: dict, warming: bool,
                 last_epoch: int | None) -> dict | None:
    """The launch-pipeline position: bench -> receipt -> launch -> run -> byte-close.
    Reads the NEWEST launch-provenance dir (which may be a pre-launch bench dir the
    verdict-follower cannot see yet) + dry_start_report.json + run liveness. Every
    state is DERIVED from on-disk artifacts; fail-open None hides the strip."""
    pdir = _newest_launch_dir()
    if pdir is None:
        return None
    # watched may be the dir itself OR a nested active child (dry_start/ under
    # pipeline-follow) — both mean "the watcher is on this pipeline".
    is_watched = False
    if watched_dir:
        with contextlib.suppress(Exception):
            wd, pr = Path(watched_dir).resolve(), pdir.resolve()
            is_watched = (wd == pr) or (pr in wd.parents)
    report = None
    rp = pdir / "dry_start_report.json"
    if rp.is_file():
        with contextlib.suppress(Exception):
            report = json.loads(rp.read_text())
    has_bench_dir = (pdir / "dry_start").is_dir()
    has_run_log = (pdir / "run.log").exists()
    steps: list[dict] = []
    # 1 · bench (bounded dry-start on the REAL config)
    if report is not None:
        steps.append({"id": "bench", "state": "done", "detail": "dry-start ran"})
    elif has_bench_dir and not has_run_log:
        steps.append({"id": "bench", "state": "active", "detail": "dry-start in flight"})
    elif has_bench_dir:
        steps.append({"id": "bench", "state": "done", "detail": "dry-start dir"})
    else:
        steps.append({"id": "bench", "state": "pending", "detail": "no dry_start yet"})
    # 2 · receipt (green = boot+step+ckpt+resume at the real n)
    if report is not None:
        green = bool(report.get("green"))
        det = []
        spm = report.get("sec_per_ep_marginal") or report.get("sec_per_ep_gross")
        if spm is not None:
            det.append(f"{float(spm):.0f}s/ep")
        pk = report.get("peak_rss_gib")
        if pk is not None:
            det.append(f"peak {float(pk):.1f}GiB")
        det.append("resume " + ("ok" if report.get("resume_round_trip_ok") else "FAIL"))
        steps.append({"id": "receipt", "state": ("done" if green else "failed"),
                      "detail": ("GREEN · " if green else "RED · ") + " · ".join(det)})
    else:
        steps.append({"id": "receipt", "state": "pending", "detail": "awaiting dry_start_report"})
    # 3 · launch (the real trainer spawned into this dir)
    steps.append({"id": "launch", "state": ("done" if has_run_log else "pending"),
                  "detail": ("run.log present" if has_run_log else "not fired")})
    # 4 · run (liveness of the watched arm — only meaningful once this dir is the watched one)
    if has_run_log and is_watched:
        kind = (liveness or {}).get("kind", "?")
        st = "active" if kind in ("live", "warming") or warming else (
            "failed" if kind == "stale" else "pending")
        det = ("warming up" if warming else str(kind)) + (
            f" · ep{last_epoch}" if last_epoch is not None else "")
        steps.append({"id": "run", "state": st, "detail": det})
    elif has_run_log:
        steps.append({"id": "run", "state": "active", "detail": "started (not the watched arm)"})
    else:
        steps.append({"id": "run", "state": "pending", "detail": "—"})
    # 5 · byte-close (exact-eval archive artifacts in the run dir)
    bc = None
    with contextlib.suppress(OSError):
        for pat in ("*archive*.zip", "byteclose*", "*byte_close*"):
            hits = list(pdir.glob(pat))
            if hits:
                bc = hits[0].name
                break
    steps.append({"id": "byteclose", "state": ("done" if bc else "pending"),
                  "detail": (bc or "pending — pointer moves only through this")})
    return {"dir": pdir.name, "is_watched": is_watched, "steps": steps,
            "config_family": _launch_config_family(pdir)}


def _launch_config_family(pdir: Path) -> str | None:
    """The '# tac-config-family:' header from the dir's launch.sh (tiny read)."""
    try:
        head = (pdir / "launch.sh").read_text(errors="replace")[:2048]
    except OSError:
        return None
    for ln in head.splitlines():
        if ln.startswith("# tac-config-family:"):
            return ln.split(":", 1)[1].strip()
    return None


def _pid_alive(pid: int) -> bool:
    """Delegates to the canonical tri-state read (``tac.process_liveness``).

    Behaviour CHANGED in three ways (undocumented drift in the old local copy):

    * ``PermissionError`` was DEAD, now ALIVE -- the UI no longer claims
      "training gone" for a live process it merely cannot signal.
    * A zombie was ALIVE forever, now DEAD -- and this function also gates the
      16-min render LOCK (``_render_lock_held``), where a zombie holder used to
      wedge the lock permanently instead of being reclaimed as stale.
    * A NEGATIVE pid was ALIVE (``os.kill(-n, 0)`` is a process-GROUP probe),
      now UNREADABLE -> False.
    """
    return process_liveness.pid_state(pid) == process_liveness.ALIVE


def _training_alive(pid: int, sig: str) -> bool:
    """pid OR cmdline-signature liveness — robust to a sibling agent relaunching
    the trainer under a new pid (so the UI doesn't say 'training gone' falsely)."""
    if _pid_alive(pid):
        return True
    if not sig:
        return False
    try:
        import subprocess
        out = subprocess.run(["ps", "-axww", "-o", "command="],
                             capture_output=True, text=True, timeout=5).stdout
        return any(sig in line for line in out.splitlines())
    except Exception:
        return False


def _resume_from_path(log_path) -> str | None:
    """The ckpt path this run RESUMED from (warm-start ancestry), or None for a root run.
    Parses the trainer's ``{"stage":"resume","from":...}`` line."""
    try:
        with open(log_path, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if '"stage": "resume"' in line and '"from"' in line:
                    try:
                        return json.loads(line).get("from")
                    except Exception:
                        continue
    except Exception:
        return None
    return None


def _resume_start_epoch(log_path) -> int | None:
    """The ``start_epoch`` this run resumed at (warm-start boundary), or None for a
    root run. Used to infer the l7 -> Muon boundary (the Muon stage is an OPTIMIZER
    switch, not a loss-form switch, so it never appears in the verdict ``seg_form``)."""
    try:
        with open(log_path, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if '"stage": "resume"' in line and '"start_epoch"' in line:
                    try:
                        return json.loads(line).get("start_epoch")
                    except Exception:
                        continue
    except Exception:
        return None
    return None


def _n_pairs_from_log(log_path) -> int | None:
    """The N this run was evaluated under (n200 = DOE pilot, n600 = scored). Parsed
    from the trainer's ``{"stage": "gt", "n_pairs": N}`` line near the top of the log."""
    try:
        with open(log_path, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if '"stage": "gt"' in line and '"n_pairs"' in line:
                    try:
                        n = json.loads(line).get("n_pairs")
                        return n if isinstance(n, int) else None
                    except Exception:
                        continue
    except Exception:
        return None
    return None


def _resume_chain_logs(latest):
    """Walk the resume ancestry from the latest arm back to the root run. Returns the
    verdict logs as ``[root .. latest]`` so the dashboard shows the FULL trajectory
    (CE->tau->l7->muon...), not just the post-resume tail. Bounded; cycle-safe."""
    if latest is None:
        return []
    chain, seen, cur = [], set(), latest
    for _ in range(20):  # bound the walk (depth ceiling)
        if cur is None or str(cur) in seen:
            break
        seen.add(str(cur))
        chain.append(cur)
        frm = _resume_from_path(cur)
        if not frm:
            break
        cur = rld._resolve_watched_log(None, os.path.join(os.path.dirname(frm), "*.log"))
    chain.reverse()  # root first -> later arms overwrite boundary-epoch dupes
    return chain


# ───────────────────────── TRIALITY (data-driven, self-updating) ─────────────────────────
_TRIALITY_CACHE: dict = {"ts": 0.0, "data": None}
_TRIALITY_TTL_S = 180.0


def _latest_dag_feeds(n: int = 8) -> dict:
    """The last N ``### DAG FEED ...`` headers from the newest sub015 DAG file (the
    lab TRAJECTORY leg). Pure file IO — never mutates the DAG (other work owns it)."""
    import glob
    import re
    files = sorted(glob.glob(".omx/research/sub015_DAG_*.md"),
                   key=lambda p: os.path.getmtime(p) if os.path.exists(p) else 0.0)
    if not files:
        return {"ok": False, "reason": "no sub015 DAG file found"}
    path = files[-1]
    feeds: list[dict] = []
    pat = re.compile(r"^#+\s*DAG FEED\s+(?P<tick>[0-9A-Za-z\-]+)\s*\((?P<summary>.*)$")
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                m = pat.match(line.strip())
                if m:
                    summ = m.group("summary").rstrip()
                    if summ.endswith(")"):
                        summ = summ[:-1]
                    feeds.append({"tick": m.group("tick"), "summary": summ[:320]})
    except Exception as exc:
        return {"ok": False, "reason": f"DAG read error: {exc}"}
    return {"ok": True, "file": os.path.basename(path), "total": len(feeds),
            "recent": feeds[-int(n):]}


def _latest_equations(n: int = 8) -> dict:
    """The last N registered canonical equations (the LAW leg) — read straight from the
    JSONL registry ledger (no import; latest-row-wins by equation_id)."""
    path = ".omx/state/canonical_equations_registry.jsonl"
    if not os.path.exists(path):
        return {"ok": False, "reason": "no canonical equations registry"}
    latest: dict[str, dict] = {}
    total = 0
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                total += 1
                eid = row.get("equation_id") or row.get("id") or row.get("name")
                if not eid:
                    continue
                pay = row.get("equation_payload") or {}
                desc = (pay.get("one_line_summary") or pay.get("name")
                        or pay.get("latex_form") or row.get("notes") or "")
                latest[eid] = {"id": str(eid), "desc": str(desc)[:220]}
    except Exception as exc:
        return {"ok": False, "reason": f"registry read error: {exc}"}
    items = list(latest.values())[-int(n):]
    return {"ok": True, "distinct": len(latest), "rows": total, "recent": items}


def _dsl_summary() -> dict:
    """The DSL (CONTROL leg) — the witness program + canonical gauge, introspected from
    ``tac.witness_dsl`` (torch-free; lean-safe to import). Read-only (never edits the DSL)."""
    try:
        import tac.witness_dsl as w
        prog = w.BASELINE
        stages = [getattr(s, "name", str(s)) for s in getattr(prog, "stages", ())]
        # GENERIC-over-the-DSL (operator amendment 2026-07-07): enumerate whatever the
        # DSL declares via the uniform describe surface (soft-detected per primitive;
        # generic fallback otherwise) so a NEW stage kind / level-path / cadence
        # primitive renders here with ZERO dashboard code change — never dropped.
        try:
            from tac.witness_dsl.schedule_readback import display_entry as _de
            stage_details = [_de(s) for s in getattr(prog, "stages", ())]
            cur = getattr(prog, "curriculum", None)
            if cur is not None:
                from tac.witness_dsl.schedule_readback import stage_map_from_curriculum
                stage_details += [e for e in stage_map_from_curriculum(
                    cur, getattr(prog, "epochs", None)) if e.get("mode") == "declared"]
        except Exception:
            stage_details = []
        gauge = w.CANONICAL_GAUGE
        gsum = {}
        for f in ("warp", "carrier", "residual"):
            v = getattr(gauge, f, None)
            gsum[f] = getattr(v, "value", str(v)) if v is not None else None
        return {"ok": True,
                "program": {"epochs": getattr(prog, "epochs", None),
                            "num_pairs": getattr(prog, "num_pairs", None),
                            "stages": stages,
                            "stage_details": stage_details},
                "gauge": gsum}
    except Exception as exc:
        return {"ok": False, "reason": f"witness_dsl introspection error: {exc}"}


def triality_snapshot() -> dict:
    """Assemble the data-driven TRIALITY payload (cached; TTL-refreshed). The three legs
    self-update from the LIVE artifacts: DAG FEEDs (trajectory) + witness_dsl (control) +
    canonical_equations registry (law). Advisory / NON-PROMOTABLE — a viz moves no pointer."""
    now = time.time()
    if _TRIALITY_CACHE["data"] is not None and (now - _TRIALITY_CACHE["ts"]) < _TRIALITY_TTL_S:
        return _TRIALITY_CACHE["data"]
    _ptr = frontier_pointer()
    data = {
        "ok": True,
        "built_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "pointer": (_ptr.get("score") if _ptr.get("ok") else None),
        "dag": _latest_dag_feeds(),
        "dsl": _dsl_summary(),
        "equations": _latest_equations(),
        "authority": "macOS-MLX training advisory · NON-PROMOTABLE",
    }
    _TRIALITY_CACHE.update(ts=now, data=data)
    return data


# ───────────────────────── access gate (pure, testable) ─────────────────────────
# Live gate TOGGLE (operator 2026-07-11 "make this gating toggleable and turn it
# off for now"): a tiny state file next to the supervisor's .access_key. Content
# "off"/"0"/"false"/"disabled" disables key enforcement WITHOUT a restart (read
# per-request behind a 2s cache); a missing file or any other content leaves the
# gate ON (fail-closed default). gate_decision itself stays PURE — the toggle
# works by feeding it an EMPTY effective key, which its existing "no key
# configured -> allow" semantics already treat as open.
#   toggle off: echo off > .omx/tmp/dash_levelset_deploy/.access_gate
#   toggle on:  echo on  > .omx/tmp/dash_levelset_deploy/.access_gate  (or rm it)
_ACCESS_GATE_FILE = _REPO_ROOT / ".omx" / "tmp" / "dash_levelset_deploy" / ".access_gate"
_ACCESS_GATE_CACHE = {"ts": 0.0, "on": True}


def access_gate_enabled() -> bool:
    now = time.time()
    if now - _ACCESS_GATE_CACHE["ts"] > 2.0:
        try:
            on = _ACCESS_GATE_FILE.read_text().strip().lower() not in (
                "off", "0", "false", "disabled")
        except OSError:
            on = True  # missing/unreadable -> gate ON (fail-closed)
        _ACCESS_GATE_CACHE.update(ts=now, on=on)
    return bool(_ACCESS_GATE_CACHE["on"])


def _effective_access_key(cfg) -> str:
    """cfg.access_key when the gate toggle is ON, else "" (= allow everyone)."""
    return cfg.access_key if access_gate_enabled() else ""


def gate_decision(headers: dict, query_key: str | None, cookie_key: str | None,
                  access_key: str, strict_local: bool = False) -> str:
    """Return "allow" or "deny".

    - No access_key configured -> allow (local-only mode, no tunnel).
    - strict_local=False (HTTP page/api): a request with NO Cf-Ray /
      Cf-Connecting-Ip is treated as trusted localhost (cloudflared injects those
      on tunnelled GETs, and the server only binds 127.0.0.1), so the page stays
      frictionless locally. A public (CF-headered) request must present the key.
    - strict_local=True (WebSocket): NO local bypass — the key is ALWAYS required
      when configured. Rationale (MEASURED 2026-06-30): cloudflared does NOT inject
      Cf-Ray on the WS upgrade, so the cf-header heuristic cannot tell a tunnelled
      WS from a local one; requiring the key unconditionally closes that bypass.
      Legitimate browsers carry the dash_key COOKIE (set on page-serve), so their
      WS still authenticates.
    The key is matched in constant time via ?k= / X-Dash-Key / dash_key cookie.
    """
    if not access_key:
        return "allow"
    h = {k.lower(): v for k, v in headers.items()}
    if not strict_local:
        is_public = bool(h.get("cf-ray") or h.get("cf-connecting-ip"))
        if not is_public:
            return "allow"
    supplied = query_key or cookie_key or h.get("x-dash-key")
    if supplied and hmac.compare_digest(str(supplied), str(access_key)):
        return "allow"
    return "deny"


# ───────────────────────── live state ─────────────────────────
class LiveState:
    """In-memory trajectory + liveness, refreshed by the async tailer. Pushes
    deltas to connected WebSocket clients. Synchronous file IO (tiny logs) is run
    in a thread executor by the tailer so the event loop never blocks."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.trajectory: list[dict] = []
        self._epochs: set[int] = set()
        self.sensors: dict = {}  # latest jacobian_basin + loss_terms (non-verdict stages)
        self.curriculum_panel: dict = {}  # DERIVED curriculum model (event triggers + lanes)
        self.pose_readiness: dict = {}    # honest pose state + unselected R1 reference
        self.liveness: dict = {"kind": "missing"}
        self.watched: str | None = None
        self.watched_dir: str | None = None  # live arm dir (auto-latest); shown as run_dir
        self.muon_start: int | None = None    # inferred l7 -> Muon boundary (additive meta)
        self.n_pairs: int | None = None       # N for this run (n200 DOE pilot / n600 scored)
        self.warming_up: bool = False         # live run resolved but no verdict yet (structured-init)
        self.run_config: dict = {}            # parsed launch.sh/run.log config + curriculum schedule
        self.pose_blind: bool | None = None   # w_pose==0 in the RUN'S OWN config (None = unknown)
        # d_seg goal lines: {"dseg": {value, source}, "dseg15": {...}} — DERIVED per run
        # (or env/CLI override); value None -> the goal line is NOT rendered (conditional).
        self.goal_info: dict = {"dseg": {"value": None, "source": None},
                                "dseg15": {"value": None, "source": None}}
        # costate controller SENSE/DECIDE panel source (run_dir/costate_shadow.jsonl;
        # read-only, advisory). None -> the panel is absent (conditional rendering).
        self.costate: dict | None = None
        # Global DDM campaign view. It is built from the same state digest as the
        # digest/duty/nag consumers and remains visible without a witness run.
        self.ddm_campaign: dict | None = None
        # schema-driven introspection payload (#352): schedule classification + controller
        # + LawRef constants + planned curves + liveness + mem + fired events. None -> the
        # new LIVE panels are absent (conditional; pre-v6 run dirs degrade gracefully).
        self.introspect: dict | None = None
        self._introspect_sig: tuple | None = None  # mtime-gate (recompute only on artifact change)
        # RUN-IDENTITY header row (name + purpose chip + scope chip); None -> row absent.
        self.run_identity: dict | None = None
        # c2-era run events (engage markers / confound alarms / warm-start origin / clip / rate):
        # incremental per-log scan accumulator ({} -> panels absent, conditional rendering).
        self.run_events: dict = {}
        self._evcache: dict = {}   # per-log-path incremental scan state (offset + accumulators)
        self._verdict_cache: dict = {}  # per-log (mtime,size)->parsed verdict rows (lineage stitch)
        # chain-state strip: bench -> receipt -> launch -> run -> byte-close (None -> hidden).
        self.chain_state: dict | None = None
        # pipeline-follow (operator round-2): watching a pre-launch pipeline dir whose
        # real run.log has not fired yet (the c2 bench window). False = normal watcher.
        self.pipeline_follow: bool = False
        # canonical DSL schedule read-back (PLANNED launch.sh-through-real-argparse
        # + ACTUAL fired-transition evidence). {"ok": False} -> visible fallback.
        self.schedule_readback: dict = {"ok": False, "reason": "not yet derived"}
        self.run_info: dict = {}              # #205 full run-info dict (rld._collect_run_info)
        self.run_info_html: str = ""          # #205 strip HTML (rld._run_info_html), pre-rendered off-loop
        self.projection: dict = {"ok": False, "reason": "calibrating"}  # DATA-DERIVED trajectory model
        # WITNESS (Tab 2) + FLOW (Tab 3) are produced by ONE detached governed subprocess
        # (tools/dashboard_flow_sequence.py) run on each NEW best checkpoint. This dashboard
        # process NEVER imports torch / renders in-process -> it stays lean (~few hundred MB,
        # safe under its safe_run RSS cap). self.witness holds the Tab-2 panels (small; rides
        # the WS snapshot). self._flow_seq_bytes holds the Tab-3 n600 video payload (bigger)
        # served on demand via /api/flow_sequence; only a lightweight {type:flow_ready} pings
        # clients to (re)fetch it.
        self.witness: dict = {}
        self._witness_dirty: bool = False
        self._flow_seq_bytes: bytes | None = None      # the served flow.json (Tab-3 sequence)
        self._flow_seq_meta: dict = {}                 # {epoch,n,mean_dseg,built_at_utc}
        self._flowseq_dirty: bool = False              # a fresh sequence -> broadcast flow_ready
        self._flowseq_running: bool = False
        self._flowseq_proc = None                      # subprocess.Popen | None
        self._flowseq_stem: Path | None = None
        self._flowseq_target_mtime: float = 0.0
        self._flowseq_done_mtime: float = 0.0          # ckpt mtime we already have output for
        self._flowseq_last_attempt: float = 0.0
        self._flowseq_err: str = ""
        self._flowseq_foreign_stem: Path | None = None  # (#343) a detached render owned by a prior instance
        # ORACLE (Tab 1): STATIC physical-prior atlas — depends only on the GT cache. Rendered
        # ONCE by a DETACHED governed safe_run subprocess (tools/oracle_dashboard_panels.py) and
        # cached to disk; served on demand via /api/oracle. Bytes held here after the one ingest.
        self._oracle_bytes: bytes | None = None        # the served oracle.json payload
        self._oracle_meta: dict = {}                   # {built_at_utc,frames_shown,render_secs}
        self._oracle_running: bool = False
        self._oracle_proc = None                       # subprocess.Popen | None
        self._oracle_stem: Path | None = None
        self._oracle_last_attempt: float = 0.0
        self._oracle_err: str = ""
        # WHY/HOW (Tab 4): STATIC single-frame deep-math FIELD BUNDLE — depends only on the GT
        # cache. Rendered ONCE by a DETACHED governed safe_run subprocess
        # (tools/whyhow_deepmath_panels.py) and cached to disk; served on demand via /api/whyhow.
        self._whyhow_bytes: bytes | None = None        # the served whyhow.json payload
        self._whyhow_meta: dict = {}                   # {built_at_utc,frame_idx,render_secs}
        self._whyhow_running: bool = False
        self._whyhow_proc = None                       # subprocess.Popen | None
        self._whyhow_stem: Path | None = None
        self._whyhow_last_attempt: float = 0.0
        self._whyhow_err: str = ""
        self.started_at = time.time()
        self.clients: set[WebSocket] = set()
        # cadence sub-state (per-log). DATA-DERIVED ONLY — NO hardcoded cadence prior /
        # minute floor (the retired 18.0/10.0 seeds were the false-stale "hardcoded
        # garbage"). The only knobs are the dimensionless stale multiple K and an
        # opt-in floor (default 0 = none). eval_every / preferred_cadence_s are filled
        # PER REFRESH from the run's own schedule + measured CURRENT-stage rate.
        self._cad_args = SimpleNamespace(stale_min=None, stale_floor_min=0.0,
                                         cadence_k=rld._CADENCE_K, eval_every=None,
                                         seconds_per_epoch=None,
                                         preferred_cadence_s=None,
                                         preferred_cadence_source="measured")

    # ---- verdict-parse cache (resume-lineage stitching; operator round-2 fix 3) ----
    def _parsed_verdicts(self, lg) -> list[dict]:
        """``rld._parse_verdicts`` behind an (mtime,size)-gated cache. Ancestor arms in a
        resume chain are FINISHED files — stitching the full lineage (ep0..650 before a
        warm start) must cost one read total, not a per-5s-tick re-parse of every log.
        The live arm re-parses only when it actually grew."""
        try:
            st = lg.stat()
            sig = (st.st_mtime, st.st_size)
        except OSError:
            return []
        key = str(lg)
        hit = self._verdict_cache.get(key)
        if hit is not None and hit[0] == sig:
            return hit[1]
        rows = rld._parse_verdicts(lg)
        self._verdict_cache[key] = (sig, rows)
        if len(self._verdict_cache) > 64:      # bound (arm-hop hygiene over long uptimes)
            self._verdict_cache.pop(next(iter(self._verdict_cache)))
        return rows

    # ---- refresh (sync; called via executor) ----
    def refresh(self, with_witness: bool = True) -> list[dict]:
        """Auto-resolve the freshest arm (auto-latest), walk its RESUME ANCESTRY, and
        rebuild the FULL trajectory (CE->tau->l7->muon...) — not just the post-resume
        tail. Liveness reflects the live (latest) arm. Returns NEW points for WS deltas.

        ``with_witness`` gates the Tab-2 panel render (heavy-ish CPU, ~8 s): the lifespan
        PRIMING call passes False so startup is instant; the tailer passes True so the
        panels re-render on ckpt-mtime change inside the executor (never the event loop)."""
        cfg = self.cfg
        glob = cfg.resolved_glob()
        verdict_latest = rld._resolve_watched_log(None, glob)  # newest VERDICT-bearing
        run_latest = rld._resolve_run_log(None, glob)          # newest RUN log (verdict OR warming up)
        # LIVE-RUN observability (Deliverable 1): a freshly-launched run emits its
        # config stages (gt / front_end / structured_init ...) for seconds-to-minutes
        # BEFORE its first verdict epoch. Without this, that run is INVISIBLE (the
        # dashboard stays latched on the previous verdict-bearing arm until ~ep25,
        # ~45 min later). When the newest RUN log is strictly newer than the newest
        # VERDICT-bearing log AND carries no verdict of its own, FOLLOW IT NOW: meta +
        # liveness reflect it; its own (empty) trajectory renders as "warming up".
        # This re-resolves EVERY refresh tick, so ALL future launches auto-appear with
        # no manual repoint/reload (only NEW dashboard CODE needs a reload).
        warming = False
        if run_latest is not None and not rld._has_verdict(run_latest):
            if verdict_latest is None:
                warming = True
            else:
                # A NEWER-LAUNCH warming run always supersedes an OLDER verdict-bearing
                # run (serial runs; the old one is stopped/finished). Compare the
                # filename LAUNCH timestamp first — robust to the mtime race at the swap
                # instant (a just-killed run's final flush can transiently out-mtime the
                # fresh run's first write, which used to latch the dashboard onto the
                # dead run and render it "stale"). mtime is the fallback only when a
                # launch token is missing.
                rl_ts = rld._launch_ts(run_latest)
                vl_ts = rld._launch_ts(verdict_latest)
                if rl_ts is not None and vl_ts is not None:
                    warming = rl_ts >= vl_ts
                else:
                    try:
                        warming = run_latest.stat().st_mtime >= verdict_latest.stat().st_mtime
                    except OSError:
                        warming = True
        if warming:
            latest = run_latest
            # WARM-UP LINEAGE (p0_343 live test, 2026-07-17): a warming run that RESUMED
            # off an ancestor checkpoint declares its ancestry in its OWN run.log
            # ({"stage":"resume","from":...}) — stitch that chain NOW so the ep0..fork
            # ancestor trajectory renders (dimmed, pre-origin) during the long
            # pre-first-verdict window (the c2 v0 verdict alone is ~20+ min) instead of
            # a blank chart. A root run has no resume line, so the walk degrades to
            # [run_latest] — still never a FOREIGN trajectory (only the run's own
            # declared ancestry is followed, never verdict_latest).
            chain = _resume_chain_logs(run_latest)
        else:
            latest = verdict_latest
            # FULL trajectory across the warm-start chain (de-dup by epoch; later arm wins
            # at boundary collisions since the chain is ordered root..latest).
            chain = _resume_chain_logs(latest)
        self.warming_up = warming

        # PIPELINE-FOLLOW (operator round-2, 2026-07-16): the c2 bench window exposed a
        # selection split — the chain strip resolved the NEW launch-provenance dir while
        # the log watcher stayed latched on YESTERDAY'S completed run (its bench telemetry
        # streams to launcher scratch, so no *.log exists for the glob to find). Per the
        # refresh law (latest-run zero-manual, STRUCTURE not name): when the newest
        # launch-provenance dir is NEWER than the watched log's launch and has NOT fired
        # its real run.log yet, FOLLOW THE PIPELINE — watch its active child (dry_start/
        # while benching), present ITS identity/config, and never present the superseded
        # run's numbers as current. Hands off automatically: the real launch writes
        # run.log into the dir, is_run_dir turns true, and the normal watcher takes over.
        self.pipeline_follow = False
        _pipe = None
        try:
            # AUTO-LATEST ONLY: a pinned --run-dir (auto_latest=False) is an explicit
            # operator choice — pipeline-follow never overrides it.
            _pipe = _newest_launch_dir() if cfg.auto_latest else None
            if _pipe is not None and not (_pipe / "run.log").exists():
                pipe_ts = (_pipe / "launch.sh").stat().st_mtime
                # comparable float: the watched run's own launch.sh mtime (same clock as
                # pipe_ts); token-parse fallback for tee logs; log mtime as last resort.
                # NB rld._launch_ts returns a STRING token (YYYYmmddTHHMMSSZ), never compare
                # it against an mtime float directly.
                lat_ts = None
                if latest is not None:
                    _lls = latest.parent / "launch.sh"
                    if _lls.is_file():
                        lat_ts = _lls.stat().st_mtime
                    else:
                        tok = rld._launch_ts(latest)
                        if tok:
                            with contextlib.suppress(Exception):
                                lat_ts = datetime.datetime.strptime(
                                    str(tok), "%Y%m%dT%H%M%SZ").replace(
                                    tzinfo=datetime.timezone.utc).timestamp()
                        if lat_ts is None:
                            lat_ts = latest.stat().st_mtime
                if latest is None or (lat_ts is not None and pipe_ts > lat_ts):
                    self.pipeline_follow = True
        except Exception:
            self.pipeline_follow = False
        # the ACTIVE child: dry_start/ while the bench runs; the dir itself otherwise.
        _pipe_child = None
        if self.pipeline_follow and _pipe is not None:
            ds_dir = _pipe / "dry_start"
            _pipe_child = ds_dir if ds_dir.is_dir() else _pipe
        merged: dict[int, dict] = {}
        for _ci, lg in enumerate(chain):
            # FORK CLIP (p0_343 live test, 2026-07-17): a warm start forks the ancestor
            # at its ckpt epoch — the ancestor's rows AT/AFTER the successor's resume
            # start_epoch are its DISCARDED post-fork future (mod32cap ran to ep1000 but
            # the c2 arm forked at ep650), and rendering them bright would present a
            # superseded run's numbers as current. Clip each ancestor arm at its
            # successor's resume boundary; the live (last) arm is never clipped.
            cutoff = None
            if _ci + 1 < len(chain):
                _se = _resume_start_epoch(chain[_ci + 1])
                if isinstance(_se, int):
                    cutoff = _se
            for r in self._parsed_verdicts(lg):
                ep = r.get("epoch")
                if isinstance(ep, int) and (cutoff is None or ep < cutoff):
                    merged[ep] = r          # FULL verdict rows (seg_form etc.)
        # RAW rows feed the run-info collectors (per-stage slopes + the monitor-replay
        # closed-loop lane both need seg_form, which _slim drops — passing slimmed rows
        # silently degraded those cards to "no staged verdicts yet" / "no classified
        # eval yet"); the CLIENT trajectory ships the slimmed projection only.
        rows_raw = [merged[e] for e in sorted(merged)]
        rows_full = [_slim(r) for r in rows_raw]
        # Muon boundary (ADDITIVE meta only): the Muon stage is an OPTIMIZER switch,
        # not a loss-form switch, so it is NOT in the verdict seg_form — infer it from
        # the resume ancestry. The first arm in the chain whose dir/name signals "muon"
        # contributes its resume start_epoch as the l7 -> Muon boundary. None -> the
        # client labels >=l7 as "l7/Muon" (no separate band), per spec.
        muon_start = None
        for lg in chain:
            tag = (lg.parent.name + "/" + lg.name).lower()
            if "muon" in tag:
                se = _resume_start_epoch(lg)
                if isinstance(se, int):
                    muon_start = se
                    break
        self.muon_start = muon_start
        # N for this run (ADDITIVE meta): parse the gt line off the live arm; fall back
        # to the chain root if the resumed arm did not re-emit it.
        n_pairs = _n_pairs_from_log(latest) if latest is not None else None
        if n_pairs is None and chain:
            n_pairs = _n_pairs_from_log(chain[0])
        self.n_pairs = n_pairs
        # Resolve the REAL run dir FIRST (the watched log is often the launcher's tee in
        # .omx/tmp whose parent holds no launch.sh — the log-path-split class; fourth
        # consumer). The canonical DSL resolver is the single source of truth; fall back
        # to the log's parent (in-run-dir logs resolve to themselves).
        _run_dir = None
        if latest is not None and _dsl_resolve_run_dir is not None:
            with contextlib.suppress(Exception):
                _run_dir = _dsl_resolve_run_dir(latest)
        _cfg_dir = _run_dir if _run_dir is not None else (
            latest.parent if latest is not None else None)
        # pipeline-follow: identity/config/schedule come from the PIPELINE dir's own
        # launch.sh — never the superseded run's (the operator saw the dead n24 arm's
        # objective/n badge presented as current).
        if self.pipeline_follow and _pipe is not None:
            _run_dir = _pipe
            _cfg_dir = _pipe

        # CONFIG + CURRICULUM SCHEDULE (full observability): parse the live run's OWN
        # artifacts (launch.sh primary, run.log stages fallback) — generalizable to
        # ANY future run with zero hand-config (operator "automatically in the future").
        # Parsed BEFORE liveness so the stage-aware cadence/ETA can use the real schedule.
        try:
            self.run_config = rld.parse_run_config(_cfg_dir)
        except Exception:
            self.run_config = {"source": "none", "flags": {}, "groups": {}, "schedule": {}}

        # pose-blind detection from the RUN'S OWN launched config (--w-pose 0): the
        # arm's pose term is UNHELD BY DESIGN, so composite displays are demoted and
        # the goal-line derivation drops the pose term. None = config unknown.
        _wp = (self.run_config.get("flags") or {}).get("w-pose")
        try:
            self.pose_blind = (float(_wp) == 0.0) if _wp is not None else None
        except (TypeError, ValueError):
            self.pose_blind = None

        # d_seg goal lines — DERIVED per run from THE GOAL ladder targets + the run's
        # own measured pose+rate (env/CLI value = explicit override). Conditional:
        # value None -> the line/badge is simply not rendered.
        # pipeline-follow: goal lines derive from the run's OWN measured pose+rate — a
        # superseded run's measurements must not label the new run's chart.
        _goal_rows = [] if self.pipeline_follow else rows_full
        self.goal_info = {
            "dseg": _derive_goal_info(_goal_rows, self.pose_blind, cfg.goal_dseg,
                                      _TARGET_S_T1, _ARCHIVE_NORM_BYTES),
            "dseg15": _derive_goal_info(_goal_rows, self.pose_blind, cfg.goal_dseg_15,
                                        _TARGET_S_T3, _ARCHIVE_NORM_BYTES),
        }

        # CANONICAL DSL SCHEDULE READ-BACK (operator 2026-07-07): derive the stage map
        # from the run's OWN config through the trainer's REAL argparse + the run's
        # emitted transition evidence (fired events / stage ckpts / muon switch).
        # Fail-open: ok=False -> the legacy path below + a visible "schedule: fallback".
        _rb = None
        if _dsl_read_schedule is not None:
            try:
                # a .omx/tmp tee log's parent is NOT the run dir — resolve the real
                # out-dir from the log head (the launcher echoes its launch.sh path).
                _rbd = None
                if self.pipeline_follow and _pipe is not None:
                    _rbd = _pipe          # the pipeline dir's own launch.sh schedule
                elif latest is not None and _dsl_resolve_run_dir is not None:
                    _rbd = _dsl_resolve_run_dir(latest)
                if _rbd is None and latest is not None:
                    _rbd = latest.parent
                _rb = _dsl_read_schedule(
                    _rbd,
                    log_paths=([] if self.pipeline_follow else [str(p) for p in chain]))
            except Exception:
                _rb = None
        self.schedule_readback = (
            _rb.to_dict() if _rb is not None
            else {"ok": False, "reason": "witness_dsl read-back unavailable"})

        # Effective curriculum schedule (boundaries the cadence + projection use):
        # the DSL read-back when it resolved (disabled stages OMITTED -> a disabled
        # l7 can never be labeled); else the legacy flag-walk parse, with the Muon
        # boundary from the resume ancestry when present.
        _sched = (self.run_config or {}).get("schedule", {}) or {}
        if _rb is not None and _rb.ok:
            _sd = _rb.as_schedule_dict()
            eval_every = _sd.get("eval_every") or _sched.get("eval_every")
            sched_eff = {
                "tau_start": _sd.get("tau_start"), "l7_start": _sd.get("l7_start"),
                "muon_start": (_sd.get("muon_start") if _sd.get("muon_start") is not None
                               else self.muon_start),
                "epochs": _sd.get("epochs"), "eval_every": eval_every,
                "goal_dseg": self.goal_info["dseg"]["value"],
                "goal_dseg_15": self.goal_info["dseg15"]["value"],
            }
        else:
            eval_every = _sched.get("eval_every")
            sched_eff = {
                "tau_start": _sched.get("tau_start"), "l7_start": _sched.get("l7_start"),
                "muon_start": (self.muon_start if self.muon_start is not None else _sched.get("muon_start")),
                "epochs": _sched.get("epochs"), "eval_every": eval_every,
                "goal_dseg": self.goal_info["dseg"]["value"],
                "goal_dseg_15": self.goal_info["dseg15"]["value"],
            }

        # liveness + cadence from the LIVE (latest) arm only (the freshest log).
        now = time.time()
        latest_rows = self._parsed_verdicts(latest) if latest is not None else []
        # STAGE-AWARE next-verdict cadence: epochs in different stages take different
        # wall time (Muon's Newton–Schulz is slower than CE). Measure per-stage
        # seconds/epoch from the verdict ts and use the CURRENT stage's rate × eval_every,
        # recomputed each tick so a stale-stage rate is never carried across a boundary.
        # Only fed to liveness when it is itself MEASURED (else gap-median takes over).
        try:
            stage_spe = dtm.per_stage_seconds_per_epoch(rows_full, sched_eff)
            cur_ep = max((r["epoch"] for r in rows_full if isinstance(r.get("epoch"), int)), default=None)
            pref_cad, _cur_stage, pref_src = (
                dtm.current_stage_cadence(cur_ep, sched_eff, stage_spe, eval_every or 0)
                if cur_ep is not None else (None, "CE", "calibrating"))
        except Exception:
            pref_cad, pref_src = None, "calibrating"
        self._cad_args.eval_every = eval_every
        self._cad_args.preferred_cadence_s = pref_cad if pref_src == "measured" else None
        self._cad_args.preferred_cadence_source = pref_src
        ccfg = rld._cfg_from_args(self._cad_args)
        log_name = latest.name if latest is not None else "_none_"
        all_state, sub = rld._load_cadence_state(cfg.cadence_state, log_name)
        mtime = latest.stat().st_mtime if (latest is not None and latest.exists()) else None
        async_grace = rld._async_grace_s(latest)  # MEASURED from verdict_async_done secs
        self.liveness = rld._compute_liveness(latest_rows, mtime, now, sub, ccfg,
                                              async_grace_s=async_grace)
        rld._save_cadence_state(cfg.cadence_state, all_state, log_name, sub)
        self.watched = latest.name if latest is not None else None
        # (2026-07-07 FLOW/WITNESS idle-forever fix) the watched LOG is the launcher's tee in
        # .omx/tmp, NOT the run dir — so latest.parent holds no checkpoints and _best_ckpt()
        # returned None forever (the same log-path-split unknown-known that froze the costate
        # telemetry, third consumer). ``_run_dir`` was resolved above (before the config
        # parse) through the canonical DSL resolver; latest.parent is the fallback.
        self.watched_dir = str(_run_dir) if _run_dir is not None else (
            str(latest.parent) if latest is not None else None)

        # pipeline-follow overrides: watch INTO the active child (dry_start/ while the
        # bench runs); liveness = freshest artifact mtime in that child (the bench's
        # stdout streams to launcher scratch — artifacts are its durable heartbeat);
        # n from the pipeline's OWN config. The superseded run's log no longer speaks
        # for the header.
        if self.pipeline_follow and _pipe is not None and _pipe_child is not None:
            self.watched_dir = str(_pipe_child)
            self.watched = (_pipe.name + "/" + _pipe_child.name
                            if _pipe_child != _pipe else _pipe.name)
            try:
                _nf = (self.run_config.get("flags") or {}).get("num-pairs")
                if _nf is not None:
                    self.n_pairs = int(float(_nf))
            except (TypeError, ValueError):
                pass
            try:
                _newest = max((f.stat().st_mtime for f in _pipe_child.iterdir()
                               if f.is_file()), default=None)
            except OSError:
                _newest = None
            _age = (now - _newest) if _newest is not None else None
            # generous freshness bound: a dry-start pass is wall-clock bounded ~33 min
            # (its first epoch carries the one-time gt-load + cache build), so artifact
            # writes can legitimately be ~30 min apart before "stalled" is honest.
            _kind = ("live" if (_age is not None and _age < 2700.0)
                     else ("stale" if _age is not None else "missing"))
            self.liveness = {"kind": _kind, "log_age_s": _age, "verdict_age_s": None,
                             "cadence_s": None, "calibrating": True, "last_epoch": None,
                             "bench": True}
            self.muon_start = None

        # costate controller SENSE/DECIDE panel (read-only, advisory, conditional):
        # last shadow-observer row from <run_dir>/costate_shadow.jsonl. Fail-open None.
        try:
            self.costate = _read_costate(self.watched_dir)
        except Exception:
            self.costate = None
        try:
            self.ddm_campaign = _read_ddm_campaign()
        except Exception:
            self.ddm_campaign = None

        # SCHEMA-DRIVEN INTROSPECTION (#352, operator 2026-07-08): the schedule/curriculum
        # classification (event/derived/fixed) + costate controller + LawRef constants +
        # planned τ/β/LR curves + liveness + mem/event telemetry, ALL from the run's OWN
        # artifacts (never the epoch-scripted legacy lens). mtime-GATED so the growing
        # run.log is not re-introspected every 5 s tick — recompute only when the run dir or
        # any of its three source artifacts changes mtime. Fail-open: any error -> None ->
        # the new panels are simply absent (conditional; pre-v6 dirs degrade gracefully).
        if wri is not None and self.watched_dir:
            try:
                _wd = Path(self.watched_dir)
                _sig = (self.watched_dir,) + tuple(
                    (_p.stat().st_mtime if _p.is_file() else None)
                    for _p in (_wd / "run.log", _wd / _COSTATE_JSONL,
                               _wd / "constants_manifest.json", _wd / "launch.sh"))
                if _sig != self._introspect_sig:
                    self.introspect = wri.introspect_run(
                        self.watched_dir, log_paths=[str(p) for p in chain])
                    self._introspect_sig = _sig
            except Exception:
                self.introspect = None
        else:
            self.introspect = None

        # RUN-IDENTITY row (name + purpose + scope; operator 2026-07-07): declared
        # launch.sh header when present, labelled derived heuristic otherwise.
        try:
            # pipeline-follow: identity keys on the PIPELINE root (its launch.sh header
            # carries the declared purpose), not the dry_start child; the superseded
            # run's resume line must not label the new run.
            _id_dir = (str(_pipe) if (self.pipeline_follow and _pipe is not None)
                       else self.watched_dir)
            self.run_identity = _derive_run_identity(
                _id_dir, (self.run_config or {}).get("flags") or {},
                self.pose_blind,
                (None if self.pipeline_follow else
                 (_resume_from_path(latest) if latest is not None else None)))
        except Exception:
            self.run_identity = None

        # #205 run-info strip: the FULL operational telemetry (stage-progress / best-d_seg
        # deploy / throughput+ETA / checkpoint ledger / resumability-now / MLX fast-path /
        # config) the standalone HTML surfaces. REUSE rld's crash-safe collectors + renderer
        # verbatim (already imported), rendered HERE in the executor thread so the event loop
        # never formats it. No run resolved -> "" (strip hidden; page unchanged = back-compat).
        if latest is not None:
            try:
                # RESOLVED run dir (the tee-log parent holds no artifacts — the
                # log-path-split class) + RAW rows (seg_form-bearing; see merge above).
                self.run_info = rld._collect_run_info(
                    latest, glob, (self.watched_dir or cfg.run_dir), rows_raw,
                    self.liveness, now)
                self.run_info_html = rld._run_info_html(self.run_info)
            except Exception:
                self.run_info, self.run_info_html = {}, ""
        else:
            self.run_info, self.run_info_html = {}, ""

        # SOPHISTICATED DATA-DERIVED PROJECTION (critical-slowing d_seg model + stage-aware
        # completion ETA + implied_S projection with bands) — all from the run's own
        # trajectory + schedule; every estimate flagged; low-confidence -> 'calibrating'.
        try:
            self.projection = dtm.build_projection(
                rows_full, {**sched_eff, "schedule": sched_eff},
                sidecar_pose=(_last_measured_dpose(rows_full) or 0.0),
                archive_norm=_ARCHIVE_NORM_BYTES,
                eval_every=eval_every)
        except Exception as exc:
            self.projection = {"ok": False, "reason": f"projection error: {exc}"}

        # STAGE-AWARE reframe (operator-approved 2026-07-03): the single GLOBAL critical-slowing
        # power-law above fits the CE flicker plateau and mis-declares "sub-0.19 won't reach".
        # Attach a PER-STAGE projection so the dashboard shows each stage's OWN floor + marks the
        # tau (lane-band birth) / Muon (finishing) EXPECTED-BREAKTHROUGH boundaries the CE fit does
        # NOT model. ADVISORY; the pointer moves ONLY through a byte-closed exact eval.
        if isinstance(self.projection, dict):
            try:
                self.projection["stage_proj"] = _build_stage_aware_projection(
                    rows_full, sched_eff,
                    sidecar_pose=(_last_measured_dpose(rows_full) or 0.0),
                    archive_norm=_ARCHIVE_NORM_BYTES,
                    pose_blind=bool(self.pose_blind))
            except Exception as exc:
                self.projection["stage_proj"] = {"ok": False, "reason": f"stage projection error: {exc}"}

        # Rebuild trajectory from the full chain each tick (cheap; few hundred points).
        # new_points = epochs not yet pushed to WS clients (snapshot carries the full set).
        new_points = [p for p in rows_full
                      if isinstance(p.get("epoch"), int) and p["epoch"] not in self._epochs]
        for p in new_points:
            self._epochs.add(p["epoch"])
        self.trajectory = rows_full

        # LIVE-tab sensors: latest jacobian_basin (pose-descent readiness) + loss_terms
        # (training-health energy split) off the live arm's log tail. Non-verdict stages,
        # so they never ride the verdict trajectory. Fail-open {} -> those panels render
        # 'no reading yet'. READ-ONLY; score-neutral.
        try:
            self.sensors = _read_sensors(latest)
        except Exception:
            self.sensors = {}

        # c2-era EVENT SCAN (engage markers · confound alarms · warm-start origin · per-group
        # clip · rate soft-signal): incremental per-log scan over the FULL chain (root..latest)
        # so once-only engage/alarm rows survive the whole multi-day run (a bounded tail
        # forgets them within hours). Fail-open: last-good state on any error.
        try:
            # resolve() so a run-dir run.log SYMLINK and its .omx/tmp tee target scan ONCE
            # (double-scan would duplicate every alarm row).
            paths = list(dict.fromkeys(
                p.resolve() for p in [*chain, *([latest] if latest is not None else [])]
                if p is not None))
            accs = [_scan_log_events(p, self._evcache) for p in paths]
            # per-alarm arm scoping: only alarms from the LIVE arm's own log are "live";
            # a stitched ANCESTOR's alarms (e.g. mod32cap's ep1 warm-up transient under
            # the c2 chain) stay historical even while the current run is live.
            _lat_key = latest.resolve() if latest is not None else None
            for _p, _acc in zip(paths, accs):
                _is_live_arm = (_lat_key is not None and _p == _lat_key)
                for _a in _acc.get("alarms") or []:
                    _a["live"] = bool(_is_live_arm)
            self.run_events = _merge_run_events(accs)
            # attach the lineage SOURCE to the warm-start origin (chart marker label)
            ws = self.run_events.get("warm_start")
            if ws is not None and latest is not None:
                src = _resume_from_path(latest)
                if src:
                    ws.setdefault("source", src)
        except Exception:
            pass  # keep the previous accumulated state (never blank a live panel on a race)

        # CHAIN-STATE strip (bench -> receipt -> launch -> run -> byte-close): keyed on the
        # NEWEST launch-provenance dir, so a pre-launch bench (launch.sh written, no run.log
        # yet — the c2 pre-launch window) is visible BEFORE the verdict-follower can see it.
        try:
            _last_ep = max((r["epoch"] for r in rows_full
                            if isinstance(r.get("epoch"), int)), default=None)
            self.chain_state = _chain_state(self.watched_dir, self.liveness,
                                            self.warming_up, _last_ep)
        except Exception:
            self.chain_state = None

        # pipeline-follow presentation overrides (operator round-2): the superseded run's
        # trajectory/sensors/events must NEVER present as the pipeline run's current state
        # (the "1.92 implied-S from the dead n24 arm" complaint). Its alarms are kept but
        # explicitly HISTORICAL — scoped honestly, not hidden.
        if self.pipeline_follow:
            _ev = self.run_events or {}
            self.run_events = {"markers": [], "warm_start": None, "clip": None,
                               "rate": None, "alarms": _ev.get("alarms") or [],
                               "historical": True}
            self.trajectory = []
            new_points = []
            self.sensors = {}
            self.projection = {"ok": False,
                               "reason": "pre-launch bench (dry-start) — no verdicts yet"}
            self.run_info, self.run_info_html = {}, ""

        # CURRICULUM POSITION + POSE-DESCENT READINESS truth models (operator 2026-07-10):
        # the curriculum as DERIVED (event triggers + fail-safe caps + mechanism-lane state,
        # never a hardcoded PR95 epoch skeleton) + the unselected full-n600 byte-closed
        # macOS-CPU advisory R1 reference. Fail-open -> the JS panels fall back to legacy.
        self.curriculum_panel = {}
        self.pose_readiness = {}
        if _dcp is not None:
            try:
                _flags = (self.run_config or {}).get("flags") or {}
                _lane_ev = _dcp.read_mechanism_event_states(
                    [] if self.pipeline_follow else [str(p) for p in chain])
                if _rb is not None:
                    self.curriculum_panel = _dcp.build_curriculum_panel_model(
                        _rb, _flags, _lane_ev)
                self.pose_readiness = _dcp.build_pose_readiness_model(
                    _flags, _lane_ev, self.sensors)
            except Exception:
                self.curriculum_panel = {}
                self.pose_readiness = {}

        # WITNESS (Tab 2) + FLOW (Tab 3): orchestrate the detached governed 600-pass subprocess
        # (spawn on new best checkpoint / ingest its output when done). NON-blocking + NO torch
        # in-process; the ~2.6 GB render lives in its OWN process group so it can never touch
        # THIS dashboard's safe_run RSS cap. A viz must NEVER take down the live telemetry.
        if with_witness and (cfg.witness_enable or cfg.flow_enable) and self.watched_dir:
            try:
                self._maybe_flowseq(now)
            except Exception as exc:
                self._flowseq_err = f"flowseq orchestration error: {exc}"
                print(json.dumps({"stage": "dashboard_server", "flowseq_error": str(exc)}), flush=True)

        # ORACLE (Tab 1): render the STATIC physical-prior atlas ONCE (detached governed subprocess).
        if with_witness and cfg.oracle_enable and self._oracle_bytes is None:
            try:
                self._maybe_oracle(now)
            except Exception as exc:
                self._oracle_err = f"oracle orchestration error: {exc}"
                print(json.dumps({"stage": "dashboard_server", "oracle_error": str(exc)}), flush=True)

        # WHY/HOW (Tab 4): render the STATIC deep-math field bundle ONCE (detached governed subprocess).
        if with_witness and cfg.whyhow_enable and self._whyhow_bytes is None:
            try:
                self._maybe_whyhow(now)
            except Exception as exc:
                self._whyhow_err = f"whyhow orchestration error: {exc}"
                print(json.dumps({"stage": "dashboard_server", "whyhow_error": str(exc)}), flush=True)
        return new_points

    # ---- detached governed ORACLE render (STATIC physical-prior atlas; renders ONCE) ----
    def _oracle_stem_for(self) -> Path:
        cache = Path(self.cfg.oracle_gt_cache)
        try:
            mtime = int(cache.stat().st_mtime)
        except OSError:
            mtime = 0
        key = f"{cache.stem}_{mtime}_f{self.cfg.oracle_frames.replace(',', '-')}"
        return Path(self.cfg.oracle_cache_dir) / f"oracle_{key}"

    def _maybe_oracle(self, now: float) -> None:
        """Idempotent per tick: ingest a cached/finished ORACLE payload, poll a running
        subprocess, or spawn ONE (static render; memory-gated, throttled)."""
        cfg = self.cfg
        stem = self._oracle_stem_for()
        done = stem.with_suffix(".done")
        out = stem.with_suffix(".json")
        # 1) output already on disk (prior session or a just-finished subprocess) -> ingest.
        if done.exists() and out.exists():
            self._ingest_oracle(stem)
            return
        # 2) a subprocess in flight?
        if self._oracle_running:
            rc = self._oracle_proc.poll() if self._oracle_proc is not None else 0
            if rc is None:
                return
            self._oracle_running = False
            if (self._oracle_stem and self._oracle_stem.with_suffix(".done").exists()):
                self._ingest_oracle(self._oracle_stem)
            else:
                self._oracle_err = f"oracle subprocess exited rc={rc} without output"
                print(json.dumps({"stage": "dashboard_server", "oracle_no_output": rc}), flush=True)
            return
        # 3) throttle + memory floor, then spawn ONCE.
        if now - self._oracle_last_attempt < 60.0:
            return
        free = self._free_gib()
        if free is not None and free < cfg.witness_min_free_gib:
            print(json.dumps({"stage": "dashboard_server", "oracle_skip": "low_free_ram",
                              "free_gib": round(free, 1)}), flush=True)
            return
        self._oracle_last_attempt = now
        self._spawn_oracle(stem)

    def _spawn_oracle(self, stem: Path) -> None:
        """Launch tools/oracle_dashboard_panels.py DETACHED (own session/process group) under its
        OWN tools/safe_run.py cap. CPU-only, numpy+matplotlib, NO torch — writes <stem>.json/.done."""
        import subprocess
        cfg = self.cfg
        tools = Path(__file__).resolve().parent
        repo = tools.parent
        stem.parent.mkdir(parents=True, exist_ok=True)
        inner = [
            sys.executable, str(tools / "oracle_dashboard_panels.py"),
            "--gt-cache", cfg.oracle_gt_cache, "--frames", cfg.oracle_frames,
            "--dpi", str(cfg.oracle_dpi),
            "--out", str(stem.with_suffix(".json")), "--done", str(stem.with_suffix(".done")),
        ]
        cmd = [
            sys.executable, str(tools / "safe_run.py"),
            # #370 observability-plane exemption (operator 2026-07-11 "prioritize populating the
            # dashboard artifacts"): skip the SUM-over-RAM reservation gate — it counts the live
            # run's conservative GROWTH reservation, not real pressure, and refused every artifact
            # build (rc=5) despite ~60 GiB measured free. Real safeguards stay: own --rss-mb cap +
            # the measured witness_min_free_gib floor checked before every spawn.
            "--skip-admission-gate",
            "--rss-mb", str(cfg.oracle_rss_mb), "--timeout", str(cfg.oracle_timeout_s),
            "--label", "oracle_atlas", "--", *inner,
        ]
        log_path = stem.with_suffix(".log")
        try:
            # with-block closes the PARENT copy of the fd after Popen dups it into the
            # child (fd-leak fix, hardening sweep 2026-07-08); the child keeps writing.
            with open(log_path, "ab") as logf:
                self._oracle_proc = subprocess.Popen(
                    cmd, cwd=str(repo), stdout=logf, stderr=subprocess.STDOUT,
                    start_new_session=True)  # OWN session/pgid -> outside this dashboard's safe_run group
            self._oracle_running = True
            self._oracle_stem = stem
            self._oracle_err = ""
            print(json.dumps({"stage": "dashboard_server", "oracle_spawn": True,
                              "pid": self._oracle_proc.pid, "stem": str(stem)}), flush=True)
        except Exception as exc:
            self._oracle_running = False
            self._oracle_err = f"oracle spawn failed: {exc}"
            print(json.dumps({"stage": "dashboard_server", "oracle_spawn_error": str(exc)}), flush=True)

    def _ingest_oracle(self, stem: Path) -> None:
        """Load a finished ORACLE payload; hold its bytes for /api/oracle (served once, static)."""
        try:
            self._oracle_bytes = stem.with_suffix(".json").read_bytes()
            dj = json.loads(stem.with_suffix(".done").read_text())
            self._oracle_meta = {"built_at_utc": dj.get("built_at_utc"),
                                 "frames_shown": dj.get("frames_shown"),
                                 "render_secs": dj.get("render_secs")}
            self._oracle_err = ""
            print(json.dumps({"stage": "dashboard_server", "oracle_ingested": True,
                              "bytes": len(self._oracle_bytes),
                              "frames": dj.get("frames_shown")}), flush=True)
        except Exception as exc:
            self._oracle_err = f"oracle load error: {exc}"

    def _oracle_ready_public(self) -> dict:
        """Lightweight ORACLE readiness (the client fetches /api/oracle on ok)."""
        if self._oracle_bytes is not None and self._oracle_meta:
            return {"ok": True, **self._oracle_meta}
        status = ("rendering" if self._oracle_running
                  else ("error" if self._oracle_err else "idle"))
        return {"ok": False, "status": status, "err": self._oracle_err or None}

    # ---- detached governed WHY/HOW deep-math field bundle (STATIC; renders ONCE) ----
    def _whyhow_stem_for(self) -> Path:
        cache = Path(self.cfg.whyhow_gt_cache)
        try:
            mtime = int(cache.stat().st_mtime)
        except OSError:
            mtime = 0
        frame_key = self.cfg.whyhow_frame.strip() or "auto"
        return Path(self.cfg.whyhow_cache_dir) / f"whyhow_{cache.stem}_{mtime}_f{frame_key}"

    def _maybe_whyhow(self, now: float) -> None:
        """Idempotent per tick: ingest a cached/finished WHY/HOW payload, poll a running
        subprocess, or spawn ONE (static render; memory-gated, throttled)."""
        cfg = self.cfg
        stem = self._whyhow_stem_for()
        done = stem.with_suffix(".done")
        out = stem.with_suffix(".json")
        if done.exists() and out.exists():
            self._ingest_whyhow(stem)
            return
        if self._whyhow_running:
            rc = self._whyhow_proc.poll() if self._whyhow_proc is not None else 0
            if rc is None:
                return
            self._whyhow_running = False
            if (self._whyhow_stem and self._whyhow_stem.with_suffix(".done").exists()):
                self._ingest_whyhow(self._whyhow_stem)
            else:
                self._whyhow_err = f"whyhow subprocess exited rc={rc} without output"
                print(json.dumps({"stage": "dashboard_server", "whyhow_no_output": rc}), flush=True)
            return
        if now - self._whyhow_last_attempt < 60.0:
            return
        free = self._free_gib()
        if free is not None and free < cfg.witness_min_free_gib:
            print(json.dumps({"stage": "dashboard_server", "whyhow_skip": "low_free_ram",
                              "free_gib": round(free, 1)}), flush=True)
            return
        self._whyhow_last_attempt = now
        self._spawn_whyhow(stem)

    def _spawn_whyhow(self, stem: Path) -> None:
        """Launch tools/whyhow_deepmath_panels.py DETACHED (own session/process group) under its
        OWN tools/safe_run.py cap. ~270 MB peak (torch only for the one-frame S-UNIWARD; NO SegNet
        forward, NO witness checkpoint) — writes <stem>.json/.done."""
        import subprocess
        cfg = self.cfg
        tools = Path(__file__).resolve().parent
        repo = tools.parent
        stem.parent.mkdir(parents=True, exist_ok=True)
        inner = [
            sys.executable, str(tools / "whyhow_deepmath_panels.py"),
            "--gt-cache", cfg.whyhow_gt_cache,
            "--out", str(stem.with_suffix(".json")), "--done", str(stem.with_suffix(".done")),
        ]
        if cfg.whyhow_frame.strip():
            inner += ["--frame", cfg.whyhow_frame.strip()]
        cmd = [
            sys.executable, str(tools / "safe_run.py"),
            "--skip-admission-gate",  # #370 observability-plane exemption (see oracle_atlas note)
            "--rss-mb", str(cfg.whyhow_rss_mb), "--timeout", str(cfg.whyhow_timeout_s),
            "--label", "whyhow_fields", "--", *inner,
        ]
        log_path = stem.with_suffix(".log")
        try:
            # with-block closes the PARENT copy of the fd after Popen dups it into the
            # child (fd-leak fix, hardening sweep 2026-07-08); the child keeps writing.
            with open(log_path, "ab") as logf:
                self._whyhow_proc = subprocess.Popen(
                    cmd, cwd=str(repo), stdout=logf, stderr=subprocess.STDOUT,
                    start_new_session=True)  # OWN session/pgid -> outside this dashboard's safe_run group
            self._whyhow_running = True
            self._whyhow_stem = stem
            self._whyhow_err = ""
            print(json.dumps({"stage": "dashboard_server", "whyhow_spawn": True,
                              "pid": self._whyhow_proc.pid, "stem": str(stem)}), flush=True)
        except Exception as exc:
            self._whyhow_running = False
            self._whyhow_err = f"whyhow spawn failed: {exc}"
            print(json.dumps({"stage": "dashboard_server", "whyhow_spawn_error": str(exc)}), flush=True)

    def _ingest_whyhow(self, stem: Path) -> None:
        """Load a finished WHY/HOW payload; hold its bytes for /api/whyhow (served once, static)."""
        try:
            self._whyhow_bytes = stem.with_suffix(".json").read_bytes()
            dj = json.loads(stem.with_suffix(".done").read_text())
            self._whyhow_meta = {"built_at_utc": dj.get("built_at_utc"),
                                 "frame_idx": dj.get("frame_idx"),
                                 "render_secs": dj.get("render_secs")}
            self._whyhow_err = ""
            print(json.dumps({"stage": "dashboard_server", "whyhow_ingested": True,
                              "bytes": len(self._whyhow_bytes),
                              "frame_idx": dj.get("frame_idx")}), flush=True)
        except Exception as exc:
            self._whyhow_err = f"whyhow load error: {exc}"

    def _whyhow_ready_public(self) -> dict:
        """Lightweight WHY/HOW readiness (the client fetches /api/whyhow on ok)."""
        if self._whyhow_bytes is not None and self._whyhow_meta:
            return {"ok": True, **self._whyhow_meta}
        status = ("rendering" if self._whyhow_running
                  else ("error" if self._whyhow_err else "idle"))
        return {"ok": False, "status": status, "err": self._whyhow_err or None}

    # ---- detached governed 600-pass: WITNESS panels + FLOW n600 video sequence ----
    def _free_gib(self) -> float | None:
        try:
            import psutil
            return psutil.virtual_memory().available / (1024 ** 3)  # RAW_VM_BASIS_OK:dashboard telemetry display, not a refuse/admit guard
        except Exception:
            return None

    def _best_ckpt(self) -> Path | None:
        """The checkpoint the 600-pass renders from: the BEST ema, or the live ema fallback."""
        if not self.watched_dir:
            return None
        best = Path(self.watched_dir) / self.cfg.flow_best_ema_name
        if best.exists():
            return best
        live = Path(self.watched_dir) / self.cfg.witness_ema_name
        return live if live.exists() else None

    def _stem_for(self, best: Path, mtime: float) -> Path:
        cfg = self.cfg
        key = f"{best.parent.name}_{int(mtime)}_ds{cfg.flow_seq_downsample}"
        return Path(cfg.flow_seq_cache_dir) / f"flow_{key}"

    def _maybe_flowseq(self, now: float) -> None:
        """Idempotent per tick: ingest a cached/finished output, poll a running subprocess, or
        spawn a new one when a NEW best checkpoint appears (throttled + memory-gated)."""
        cfg = self.cfg
        best = self._best_ckpt()
        if best is None:
            return
        try:
            mtime = best.stat().st_mtime
        except OSError:
            return
        if mtime == self._flowseq_done_mtime:
            return  # already have (and ingested) the output for this checkpoint
        stem = self._stem_for(best, mtime)
        done = stem.with_suffix(".done")
        # 1) output already on disk (prior run/session or a just-finished subprocess) -> ingest.
        if done.exists():
            self._ingest_flowseq(stem, mtime)
            return
        # 1b) another instance/render is already producing this stem (reload/restart mid-render
        # leaves the prior render detached) -> do NOT spawn a duplicate; wait for its .done.
        # (#343 UX) remember the foreign render's stem so _flow_ready_public reports "rendering"
        # with live progress instead of a misleading "idle" (restart-mid-render surfaced this).
        if not self._flowseq_running and _flowseq_lock_alive(stem.with_suffix(".lock")):
            self._flowseq_foreign_stem = stem
            return
        self._flowseq_foreign_stem = None
        # 2) a subprocess in flight (this instance)?
        if self._flowseq_running:
            rc = self._flowseq_proc.poll() if self._flowseq_proc is not None else 0
            if rc is None:
                return  # still rendering (a full n600 pass is ~14 min)
            self._flowseq_running = False
            if (self._flowseq_stem and (self._flowseq_stem.with_suffix(".done")).exists()):
                self._ingest_flowseq(self._flowseq_stem, self._flowseq_target_mtime)
            else:
                # rc=5 is a safe_run governor REFUSE: the ~2.6 GB n600 render is memory-gated by the
                # live training run holding the box. That is correct back-pressure, not a crash — say so
                # honestly and NEVER surface the raw subprocess code to the operator.
                if rc == 5:
                    self._flowseq_err = ("the n600 video render is waiting — memory is reserved by the "
                                         "live training run; it builds automatically when the box frees")
                else:
                    self._flowseq_err = ("the n600 video render did not finish this pass; "
                                         "it retries automatically on the next best checkpoint")
                print(json.dumps({"stage": "dashboard_server", "flowseq_no_output": rc}), flush=True)
                if self._flowseq_stem is not None:  # release the lock so a retry can spawn
                    with contextlib.suppress(Exception):
                        self._flowseq_stem.with_suffix(".lock").unlink()
            return
        # 3) throttle + memory floor, then spawn.
        if now - self._flowseq_last_attempt < cfg.flow_seq_min_interval_s:
            return
        free = self._free_gib()
        if free is not None and free < cfg.witness_min_free_gib:
            print(json.dumps({"stage": "dashboard_server", "flowseq_skip": "low_free_ram",
                              "free_gib": round(free, 1)}), flush=True)
            return
        self._flowseq_last_attempt = now
        self._spawn_flowseq(best, mtime, stem)

    def _spawn_flowseq(self, best: Path, mtime: float, stem: Path) -> None:
        """Launch tools/dashboard_flow_sequence.py DETACHED (own session/process group) under its
        OWN tools/safe_run.py cap, so its ~2.6 GB torch+SegNet footprint is isolated from this
        dashboard's safe_run RSS group. Writes <stem>.flow.json / .witness.json / .done."""
        import subprocess
        cfg = self.cfg
        tools = Path(__file__).resolve().parent
        repo = tools.parent
        stem.parent.mkdir(parents=True, exist_ok=True)
        epoch = self.liveness.get("last_epoch")
        if epoch is None:
            epoch = max((r["epoch"] for r in self.trajectory
                         if isinstance(r.get("epoch"), int)), default=0)
        run_token = Path(self.watched_dir).name  # pgrep -f token for #205 liveness logging
        inner = [
            sys.executable, str(tools / "dashboard_flow_sequence.py"),
            "--ckpt-dir", str(self.watched_dir), "--npz-name", best.name,
            "--gt-cache", cfg.witness_gt_cache, "--out-stem", str(stem),
            "--epoch", str(int(epoch)), "--downsample", str(cfg.flow_seq_downsample),
            "--jpeg-quality", str(cfg.flow_seq_jpeg_q), "--frag-levels", str(cfg.flow_seq_frag_levels),
            "--hard-k", str(cfg.flow_seq_hard_k), "--dpi", str(cfg.witness_dpi),
            "--min-free-gib", str(cfg.witness_min_free_gib), "--run-token", run_token,
            "--torch-threads", "3",
        ]
        cmd = [
            sys.executable, str(tools / "safe_run.py"),
            "--skip-admission-gate",  # #370 observability-plane exemption (see oracle_atlas note)
            "--rss-mb", str(cfg.flow_seq_rss_mb), "--timeout", str(cfg.flow_seq_timeout_s),
            "--label", f"flow_seq_{int(mtime)}", "--", *inner,
        ]
        log_path = stem.with_suffix(".log")
        try:
            # with-block closes the PARENT copy of the fd after Popen dups it into the
            # child (fd-leak fix, hardening sweep 2026-07-08); the child keeps writing.
            with open(log_path, "ab") as logf:
                self._flowseq_proc = subprocess.Popen(
                    cmd, cwd=str(repo), stdout=logf, stderr=subprocess.STDOUT,
                    start_new_session=True)  # OWN session/pgid -> outside this dashboard's safe_run group
            self._flowseq_running = True
            self._flowseq_stem = stem
            self._flowseq_target_mtime = mtime
            self._flowseq_err = ""
            # cross-instance lock: record the render's pid so a reload/restart mid-render does NOT
            # spawn a duplicate (the detached render survives the reload; the new instance waits).
            with contextlib.suppress(Exception):
                stem.with_suffix(".lock").write_text(str(self._flowseq_proc.pid))
            print(json.dumps({"stage": "dashboard_server", "flowseq_spawn": True,
                              "pid": self._flowseq_proc.pid, "epoch": int(epoch),
                              "ckpt": best.name, "stem": str(stem)}), flush=True)
        except Exception as exc:
            self._flowseq_running = False
            self._flowseq_err = f"flow-seq spawn failed: {exc}"
            print(json.dumps({"stage": "dashboard_server", "flowseq_spawn_error": str(exc)}), flush=True)

    def _ingest_flowseq(self, stem: Path, mtime: float) -> None:
        """Load a finished 600-pass output: witness panels into memory (rides the WS snapshot),
        flow.json bytes held for /api/flow_sequence, and mark both dirty for broadcast."""
        try:
            self.witness = json.loads(stem.with_suffix(".witness.json").read_text())
            self._witness_dirty = True
        except Exception as exc:
            self._flowseq_err = f"witness load error: {exc}"
        try:
            self._flow_seq_bytes = stem.with_suffix(".flow.json").read_bytes()
            dj = json.loads(stem.with_suffix(".done").read_text())
            self._flow_seq_meta = {"epoch": dj.get("epoch"), "n": dj.get("n"),
                                   "mean_dseg": dj.get("mean_dseg"),
                                   "built_at_utc": dj.get("built_at_utc")}
            self._flowseq_dirty = True
            self._flowseq_err = ""
            print(json.dumps({"stage": "dashboard_server", "flowseq_ingested": True,
                              "epoch": dj.get("epoch"), "n": dj.get("n"),
                              "flow_bytes": len(self._flow_seq_bytes)}), flush=True)
        except Exception as exc:
            self._flowseq_err = f"flow load error: {exc}"
        self._flowseq_done_mtime = mtime  # mark this checkpoint's output consumed (no re-spawn)
        with contextlib.suppress(Exception):  # release the cross-instance render lock
            stem.with_suffix(".lock").unlink()

    def consume_flowseq_dirty(self) -> bool:
        if self._flowseq_dirty:
            self._flowseq_dirty = False
            return True
        return False

    def _flow_ready_public(self) -> dict:
        """Lightweight FLOW readiness ping (the client fetches /api/flow_sequence on this).
        (#343 UX) "rendering" also covers a FOREIGN in-flight render (a detached pass surviving a
        server restart — lock alive, this instance not the owner), with pair-progress parsed from
        the render's own log tail so the client can show a real progress bar."""
        if self._flow_seq_bytes is not None and self._flow_seq_meta:
            return {"ok": True, **self._flow_seq_meta}
        foreign = getattr(self, "_flowseq_foreign_stem", None)
        rendering = self._flowseq_running or foreign is not None
        status = ("rendering" if rendering
                  else ("error" if self._flowseq_err else "idle"))
        out: dict = {"ok": False, "status": status, "err": self._flowseq_err or None}
        if rendering:
            stem = self._flowseq_stem if self._flowseq_running else foreign
            with contextlib.suppress(Exception):  # progress is best-effort observability
                tail = stem.with_suffix(".log").read_bytes()[-4096:].decode("utf-8", "replace")
                for line in reversed(tail.strip().splitlines()):
                    row = json.loads(line)
                    if row.get("event") == "progress":
                        out["pair"] = int(row.get("pair", 0))
                        out["n"] = int(row.get("n", 0))
                        out["secs"] = round(float(row.get("secs", 0.0)), 1)
                        break
        return out

    def flow_ready_msg(self) -> dict:
        return {"type": "flow_ready", "flow_ready": self._flow_ready_public()}

    def consume_witness_dirty(self) -> bool:
        """Return+clear the witness-dirty flag (single-threaded event loop -> race-free)."""
        if self._witness_dirty:
            self._witness_dirty = False
            return True
        return False

    # ---- snapshot for client ----
    def _stage_map_and_source(self) -> tuple[list[dict], str, dict]:
        """The per-run derived stage map + provenance label + resolved legacy scalars.

        Resolution order (operator 2026-07-07): explicit ``--tau``/``--l7`` CLI/env
        values OVERRIDE (explicit-override-only; deprecation note in --help); when
        NOT passed, the map is DERIVED from the run's own config via the DSL
        read-back; when the read-back fails (old run dirs, missing launch.sh) the
        legacy flag-walk/constants path is used with a visible "fallback" marker.
        """
        cfg = self.cfg
        sched = (self.run_config or {}).get("schedule", {}) or {}
        rb = self.schedule_readback or {}
        rb_ok = bool(rb.get("ok"))
        overridden = cfg.tau is not None or cfg.l7 is not None
        if rb_ok and not overridden:
            stage_map = [dict(s) for s in rb.get("stages", [])]
            source = f"derived({rb.get('source', 'launch.sh')})"
        else:
            # OVERRIDE (explicit CLI values win) or FALLBACK (legacy behavior: run
            # flags, then the historical 300/600 constants — marked visibly).
            derived = {s.get("name"): s for s in rb.get("stages", [])} if rb_ok else {}
            epochs = (rb.get("epochs") if rb_ok else None) or sched.get("epochs")

            def _resolve(name: str, cli_val, sched_key: str, legacy: int):
                if cli_val is not None:
                    return int(cli_val)
                d = derived.get(name)
                if d is not None:
                    return d.get("start")
                v = sched.get(sched_key)
                if v is not None:
                    return int(v)
                return None if rb_ok else legacy
            tau_v = _resolve("tau", cfg.tau, "tau_start", 300)
            l7_v = _resolve("l7", cfg.l7, "l7_start", 600)
            muon_v = (self.muon_start if self.muon_start is not None
                      else sched.get("muon_start"))
            if derived.get("Muon") is not None and muon_v is None:
                muon_v = derived["Muon"].get("start")
            stage_map = [{"name": "CE", "start": 0, "mode": "fixed",
                          "status": "scheduled", "source": "override"}]
            for name, v in (("tau", tau_v), ("l7", l7_v), ("Muon", muon_v)):
                if v is None:
                    continue
                if epochs is not None and int(v) >= int(epochs):
                    continue  # disabled ("never" form) — conditional rendering omits it
                stage_map.append({"name": name, "start": int(v), "mode": "fixed",
                                  "status": "scheduled", "source": "override"})
            source = "override(cli)" if overridden else "fallback"
        legacy = {s["name"]: s.get("start") for s in stage_map
                  if s.get("start") is not None}
        return stage_map, source, legacy

    def meta(self) -> dict:
        cfg = self.cfg
        sched = (self.run_config or {}).get("schedule", {}) or {}
        stage_map, sched_source, _legacy = self._stage_map_and_source()
        # Legacy scalar fields kept for back-compat consumers; None when the stage
        # is absent from the derived map (conditional rendering — never a phantom l7).
        tau = _legacy.get("tau")
        l7 = _legacy.get("l7")
        muon = _legacy.get("Muon")
        gi = self.goal_info or {}
        gd = gi.get("dseg") or {}
        gd15 = gi.get("dseg15") or {}
        ptr = frontier_pointer()
        return {
            "tau": tau, "l7": l7,
            "stage_map": stage_map,             # derived per-run map (union entries)
            "schedule_source": sched_source,    # derived(...) | override(cli) | fallback
            # goal lines: DERIVED per run (or explicit override); None -> not rendered.
            "goal_dseg": gd.get("value"), "goal_dseg_15": gd15.get("value"),
            "goal_src": {"dseg": gd.get("source"), "dseg15": gd15.get("source")},
            # frontier pointer: READ from the canonical file; None -> "unavailable".
            "pointer": (ptr.get("score") if ptr.get("ok") else None),
            "pointer_info": ptr,
            "pose_blind": self.pose_blind,      # w_pose==0 in the run's own config
            "costate": self.costate,            # SENSE/DECIDE panel (None -> absent)
            "ddm_campaign": self.ddm_campaign,  # shared DDM campaign state (advisory)
            "introspect": self.introspect,      # #352 schema-driven schedule/controller/telemetry
            "introspect_ok": bool(self.introspect and self.introspect.get("ok")),
            "run_identity": self.run_identity,  # header row (None -> row absent)
            "watched": self.watched,
            "run_dir": self.watched_dir or cfg.run_dir,  # auto-latest: the live arm dir
            "uptime_s": time.time() - self.started_at,
            "training_alive": _training_alive(cfg.training_pid, cfg.training_sig),
            "n_points": len(self.trajectory),
            "muon_start": muon,             # l7 -> Muon boundary (ancestry OR curriculum flag)
            "n_pairs": self.n_pairs,        # additive: N (n200 DOE pilot / n600 scored)
            "warming_up": self.warming_up,  # live run resolved, no verdict yet (structured-init)
            "config": self.run_config or {},          # parsed setup/config + groups
            "schedule": sched,                         # curriculum stage epoch-boundaries
            "archive_norm_bytes": _ARCHIVE_NORM_BYTES,       # rate-term normalizer (client S breakdown)
            "run_info_html": self.run_info_html,       # #205 pre-rendered run-info strip (rld._run_info_html)
            "sensors": self.sensors or {},             # latest jacobian_basin + loss_terms (LIVE-tab panels)
            "run_events": self.run_events or {},       # c2-era: engage markers / alarms / warm-start / clip / rate
            "chain_state": self.chain_state,           # bench->receipt->launch->run->byte-close (None -> hidden)
            "pipeline_follow": self.pipeline_follow,   # watching a pre-launch pipeline (bench window)
            "curriculum_panel": self.curriculum_panel or {},  # DERIVED curriculum (events/caps/lanes)
            "pose_readiness": self.pose_readiness or {},      # honest pose state + R1 reference
            # POSE-DEFERRED (masthead honesty): the run has w_pose>0 (NOT pose_blind) but pose descent
            # has NOT engaged yet -- pose is HELD OUT until the pose-finish stage by design, so its loss
            # term is EXACTLY 0.0 in loss_terms and the MEASURED d_pose stays high-by-design. Without
            # this flag the masthead folds the deferred √(10·d_pose) into implied_S and a healthy
            # seg-phase run reads as a huge regression (implied_S ~12 vs pointer ~0.19). True ONLY when
            # pose loss is explicitly 0 (evidence of deferral); unknown/absent -> False (no false claim).
            "pose_deferred": bool(
                self.pose_blind is False
                and isinstance((self.sensors or {}).get("loss_terms"), dict)
                and ((self.sensors["loss_terms"].get("terms") or {}).get("pose") is not None)
                and float((self.sensors["loss_terms"]["terms"]).get("pose")) == 0.0
            ),
        }

    def snapshot(self) -> dict:
        # witness panels ride the snapshot (WS connect + /api/state fallback) so a fresh page
        # FIRST-PAINTS the Tab-2 imagery without waiting for the next checkpoint. The recurring
        # 5 s update_msg deliberately OMITS them (keeps the live delta stream tiny).
        return {"type": "snapshot", "trajectory": self.trajectory,
                "liveness": self.liveness, "meta": self.meta(),
                "projection": self.projection, "witness": self._witness_public(),
                "flow_ready": self._flow_ready_public()}

    def update_msg(self, new_points: list[dict]) -> dict:
        return {"type": "update", "new_points": new_points,
                "liveness": self.liveness, "meta": self.meta(),
                "projection": self.projection}

    def _witness_public(self) -> dict:
        """The witness payload for the client (panels + metadata), or an honest status stub
        while the first render is still in flight."""
        if self.witness:
            return self.witness
        status = ("rendering" if self._flowseq_running
                  else ("error" if self._flowseq_err else "idle"))
        return {"ok": False, "err": self._flowseq_err or None, "status": status}

    def witness_msg(self) -> dict:
        return {"type": "witness", "witness": self._witness_public()}

    # ---- broadcast ----
    async def broadcast(self, msg: dict) -> None:
        dead = []
        for ws in list(self.clients):
            try:
                await ws.send_json(msg)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.clients.discard(ws)


# ───────────────────────── app factory ─────────────────────────
def _req_keys(request) -> tuple[str | None, str | None]:
    qk = request.query_params.get("k")
    ck = request.cookies.get("dash_key")
    return qk, ck


def create_app(cfg: Config) -> Starlette:
    state = LiveState(cfg)

    async def tailer(stop: asyncio.Event) -> None:
        loop = asyncio.get_event_loop()
        while not stop.is_set():
            try:
                new_points = await loop.run_in_executor(None, state.refresh)
                await state.broadcast(state.update_msg(new_points))
                # a finished 600-pass (new best checkpoint) -> push the Tab-2 panels over the WS
                if state.consume_witness_dirty():
                    await state.broadcast(state.witness_msg())
                # ...and ping clients that a fresh Tab-3 n600 video sequence is ready to fetch
                if state.consume_flowseq_dirty():
                    await state.broadcast(state.flow_ready_msg())
            except Exception as exc:  # telemetry must never crash the live server
                print(json.dumps({"stage": "dashboard_server", "tailer_error": str(exc)}),
                      flush=True)
            try:
                await asyncio.wait_for(stop.wait(), timeout=cfg.poll)
            except TimeoutError:
                pass

    @contextlib.asynccontextmanager
    async def lifespan(app):
        # prime once so the first client gets data immediately (skip the ~8 s witness render
        # here so startup/hot-reload is instant; the first tailer tick renders the panels).
        with contextlib.suppress(Exception):
            await asyncio.get_event_loop().run_in_executor(
                None, lambda: state.refresh(with_witness=False))
        stop = asyncio.Event()
        task = asyncio.create_task(tailer(stop))
        app.state.live = state
        print(json.dumps({"stage": "dashboard_server", "started": True,
                          "port": cfg.port, "watched": state.watched,
                          "access_gated": bool(cfg.access_key) and access_gate_enabled()}),
              flush=True)
        try:
            yield
        finally:
            stop.set()
            task.cancel()
            with contextlib.suppress(Exception):
                await task

    async def index(request):
        qk, ck = _req_keys(request)
        if gate_decision(dict(request.headers), qk, ck, _effective_access_key(cfg)) == "deny":
            return HTMLResponse(_login_html(), status_code=401)
        resp = HTMLResponse(_page_html(cfg))
        # Any client that passed the page gate (local bypass OR valid key) gets the
        # cookie, so its same-origin WebSocket (which is gated strictly, no local
        # bypass) authenticates via the cookie the browser sends automatically.
        if cfg.access_key:
            resp.set_cookie("dash_key", cfg.access_key, max_age=86400 * 7,
                            httponly=True, samesite="lax", path="/")
        # NEVER let the tunnel/CDN (Cloudflare) or the browser cache the shell HTML —
        # otherwise a phone at comma-lab.adpena.com is served a STALE page from an earlier
        # run and never picks up the new run (2026-07-09 stale-view fix). The live data
        # streams over /ws; the shell must always be fetched fresh so a cold load / reconnect
        # renders the CURRENT run. Mirrors the no-store the API + /ws-key routes already set.
        resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        resp.headers["Pragma"] = "no-cache"
        return resp

    async def api_state(request):
        qk, ck = _req_keys(request)
        if gate_decision(dict(request.headers), qk, ck, _effective_access_key(cfg)) == "deny":
            return JSONResponse({"error": "access key required"}, status_code=401)
        return JSONResponse(state.snapshot())

    async def api_flow_sequence(request):
        # The Tab-3 n600 VIDEO payload (fetched ONCE per new sequence; the client then scrubs +
        # plays locally). Served as pre-built bytes so the ~7 MB dict is never re-serialized.
        qk, ck = _req_keys(request)
        if gate_decision(dict(request.headers), qk, ck, _effective_access_key(cfg)) == "deny":
            return JSONResponse({"error": "access key required"}, status_code=401)
        if state._flow_seq_bytes is not None:
            from starlette.responses import Response
            return Response(state._flow_seq_bytes, media_type="application/json",
                            headers={"Cache-Control": "no-store"})
        return JSONResponse(state._flow_ready_public(), status_code=202)

    async def api_oracle(request):
        # The Tab-1 ORACLE physical-prior atlas (fetched ONCE; static — depends only on the GT
        # cache). Served as pre-built bytes when ready, else a 202 readiness ping.
        qk, ck = _req_keys(request)
        if gate_decision(dict(request.headers), qk, ck, _effective_access_key(cfg)) == "deny":
            return JSONResponse({"error": "access key required"}, status_code=401)
        if state._oracle_bytes is not None:
            from starlette.responses import Response
            return Response(state._oracle_bytes, media_type="application/json",
                            headers={"Cache-Control": "no-store"})
        return JSONResponse(state._oracle_ready_public(), status_code=202)

    async def api_whyhow(request):
        # The Tab-4 WHY/HOW deep-math field bundle (fetched ONCE; static — depends only on the GT
        # cache). Served as pre-built bytes when ready, else a 202 readiness ping.
        qk, ck = _req_keys(request)
        if gate_decision(dict(request.headers), qk, ck, _effective_access_key(cfg)) == "deny":
            return JSONResponse({"error": "access key required"}, status_code=401)
        if state._whyhow_bytes is not None:
            from starlette.responses import Response
            return Response(state._whyhow_bytes, media_type="application/json",
                            headers={"Cache-Control": "no-store"})
        return JSONResponse(state._whyhow_ready_public(), status_code=202)

    async def api_triality(request):
        # The Tab-6 TRIALITY payload — DATA-DRIVEN, self-updating from the LIVE artifacts
        # (DAG FEEDs + witness_dsl + canonical_equations registry). Computed off-loop (cached).
        qk, ck = _req_keys(request)
        if gate_decision(dict(request.headers), qk, ck, _effective_access_key(cfg)) == "deny":
            return JSONResponse({"error": "access key required"}, status_code=401)
        try:
            data = await asyncio.get_event_loop().run_in_executor(None, triality_snapshot)
        except Exception as exc:
            return JSONResponse({"ok": False, "reason": f"triality error: {exc}"}, status_code=200)
        return JSONResponse(data, headers={"Cache-Control": "no-store"})

    # ── CAMPAIGN tab (#366 joint-descent) — filesystem-poll snapshot, server-cached.
    # One reader instance per app: incremental (parses only new/changed telemetry/
    # verdict files); a short min-interval cache absorbs multi-client polling.
    _campaign_reader = _CampaignRunReader() if _CampaignRunReader is not None else None
    _campaign_cache: dict = {"ts": 0.0, "snap": None}
    _campaign_lock = threading.Lock()  # reader caches are not thread-safe; serialize
    _CAMPAIGN_MIN_INTERVAL_S = 4.0

    def _campaign_snapshot_sync() -> dict:
        if _campaign_reader is None:
            return {"ok": False, "reason": "tac.ddm_campaign_run_reader unavailable"}
        with _campaign_lock:
            now = time.time()
            if (_campaign_cache["snap"] is not None
                    and now - _campaign_cache["ts"] < _CAMPAIGN_MIN_INTERVAL_S):
                return _campaign_cache["snap"]
            snap = _campaign_reader.snapshot(now=now)
            _campaign_cache["ts"] = now
            _campaign_cache["snap"] = snap
            return snap

    async def api_campaign(request):
        qk, ck = _req_keys(request)
        if gate_decision(dict(request.headers), qk, ck, _effective_access_key(cfg)) == "deny":
            return JSONResponse({"error": "access key required"}, status_code=401)
        try:
            data = await asyncio.get_event_loop().run_in_executor(
                None, _campaign_snapshot_sync)
        except Exception as exc:  # telemetry must never crash the live server
            return JSONResponse({"ok": False, "reason": f"campaign error: {exc}"},
                                status_code=200)
        return JSONResponse(data, headers={"Cache-Control": "no-store"})

    async def healthz(request):
        # ungated, reveals nothing sensitive (used by the supervisor + tunnel health +
        # tools/dashboard_ctl.py for its status one-liner + auto-reload staleness check).
        lv = state.liveness or {}
        return JSONResponse({
            "ok": True,
            "watched": state.watched,
            "watched_dir": state.watched_dir,
            "n_points": len(state.trajectory),
            "kind": lv.get("kind"),
            # process identity + uptime (status one-liner)
            "pid": os.getpid(),
            "port": cfg.port,
            "started_utc": _SERVER_START_UTC,
            "started_ts": _SERVER_START_TS,
            "auto_latest": cfg.auto_latest,
            # code-freshness snapshot: the mtime of the source THIS process is running.
            # dashboard_ctl compares it to the CURRENT on-disk mtime to trigger a reload.
            "code_mtime": _CODE_MTIME_AT_START,
            # last-update age (verdict recency, then log recency) — the "is it moving?"
            # signal for the status one-liner; None while calibrating / no run.
            "last_update_age_s": (lv.get("verdict_age_s")
                                  if lv.get("verdict_age_s") is not None
                                  else lv.get("log_age_s")),
            "last_epoch": lv.get("last_epoch"),
            "next_epoch": lv.get("next_epoch"),
            "next_eta_s": lv.get("next_eta_s"),
        })

    async def ws_endpoint(ws: WebSocket):
        qk = ws.query_params.get("k")
        ck = ws.cookies.get("dash_key")
        # strict_local=True: WS ALWAYS requires the key when configured (no cf-header
        # local bypass — cloudflared omits Cf-Ray on the WS upgrade).
        if gate_decision(dict(ws.headers), qk, ck, _effective_access_key(cfg), strict_local=True) == "deny":
            await ws.close(code=1008)  # policy violation
            return
        await ws.accept()
        state.clients.add(ws)
        try:
            await ws.send_json(state.snapshot())
            # The client need not send anything; this receive loop exists purely
            # to surface WebSocketDisconnect so we can drop the client cleanly.
            while True:
                await ws.receive_text()
        except WebSocketDisconnect:
            pass
        except Exception:
            pass
        finally:
            state.clients.discard(ws)

    routes = [
        Route("/", index),
        Route("/api/state", api_state),
        Route("/api/flow_sequence", api_flow_sequence),
        Route("/api/oracle", api_oracle),
        Route("/api/whyhow", api_whyhow),
        Route("/api/triality", api_triality),
        Route("/api/campaign", api_campaign),
        Route("/healthz", healthz),
        WebSocketRoute("/ws", ws_endpoint),
    ]

    # World-class OSS-hygiene response headers on every HTTP response (2026-07-09). The page is
    # FULLY self-contained (inline CSS/JS, data: PNG charts, same-origin wss) — verified no external
    # script/style — so a strict CSP is safe and blocks any injected external resource. 'unsafe-inline'
    # is required only because the shell embeds its style/script inline (nonce-hardening is a tracked
    # follow-up). Anchor nav (github/x.com) is unaffected by CSP. WebSocket scope is passed through
    # untouched (BaseHTTPMiddleware only wraps http), so /ws is never gzip'd or header-mangled.
    _CSP = (
        "default-src 'self'; img-src 'self' data:; style-src 'self' 'unsafe-inline'; "
        "script-src 'self' 'unsafe-inline'; connect-src 'self'; frame-ancestors 'self'; "
        "base-uri 'self'; form-action 'self'"
    )

    class SecurityHeadersMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request, call_next):
            resp = await call_next(request)
            resp.headers.setdefault("Content-Security-Policy", _CSP)
            resp.headers.setdefault("X-Content-Type-Options", "nosniff")
            resp.headers.setdefault("Referrer-Policy", "no-referrer")
            resp.headers.setdefault("X-Frame-Options", "SAMEORIGIN")
            resp.headers.setdefault(
                "Permissions-Policy", "camera=(), microphone=(), geolocation=(), usb=()")
            return resp

    middleware = [
        # gzip the 249 KB self-contained shell (→ ~40 KB) — a ~6x cellular win for the phone;
        # skips tiny/already-encoded bodies and never touches the WebSocket stream.
        Middleware(GZipMiddleware, minimum_size=800),
        Middleware(SecurityHeadersMiddleware),
    ]
    return Starlette(routes=routes, lifespan=lifespan, middleware=middleware)


# ───────────────────────── HTML / JS (self-contained, no CDN) ─────────────────────────
def _login_html() -> str:
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        "<title>Level-Set Witness — access</title><style>"
        "body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',system-ui,sans-serif;"
        "background:#14161a;color:#d8dde6;display:flex;align-items:center;justify-content:center;"
        "height:100vh;margin:0}.box{text-align:center}.box h1{font-size:15px;color:#8b93a3;"
        "letter-spacing:1.5px;text-transform:uppercase;font-weight:600}"
        "input{background:#1b1e24;border:1px solid #2c313b;color:#d8dde6;padding:9px 12px;"
        "border-radius:7px;font-size:14px;width:240px}button{background:#173d22;color:#7fe0a0;"
        "border:0;padding:10px 16px;border-radius:7px;font-size:14px;margin-left:6px;cursor:pointer}"
        ".n{font-size:11px;color:#8b93a3;margin-top:14px;max-width:320px}"
        "</style></head><body><div class='box'><h1>Access key required</h1>"
        "<form method='get' action='/'><input name='k' type='password' autofocus "
        "placeholder='access key' autocomplete='off'><button type='submit'>enter</button></form>"
        "<div class='n'>This dashboard describes work-in-progress method detail and is gated. "
        "Local access (127.0.0.1) is open.</div></div></body></html>"
    )


def _flow_client_js() -> str:
    """Read the FLOW (Tab 3) WebGPU client asset and return it for INLINE injection into the page
    (CSP-strict: self-contained, no external <script src>). Kept in a sibling file for craft +
    readability. Missing -> a tiny honest stub so the rest of the page is unaffected."""
    try:
        js = (Path(__file__).with_name("dashboard_flow_client.js")).read_text(encoding="utf-8")
        if "</script>" in js:  # would break the inline injection; refuse rather than corrupt the page
            raise ValueError("flow client asset contains </script>")
        return js
    except Exception as exc:
        return ("window.__flowActivate=function(){var m=document.getElementById('flowmsg');"
                "if(m){m.classList.remove('hide');m.textContent='FLOW client asset unavailable: "
                + json.dumps(str(exc))[1:-1] + "';}var b=document.getElementById('flowbadge');"
                "if(b){b.className='flowbadge off';b.textContent='unavailable';}};")


def _whyhow_client_js() -> str:
    """Read the WHY/HOW (Tab 4) deep-math museum WebGPU client asset for INLINE injection (CSP-strict:
    self-contained, no external <script src>). Kept in a sibling file for craft + readability.
    Missing -> a tiny honest stub so the rest of the page is unaffected."""
    try:
        js = (Path(__file__).with_name("dashboard_whyhow_client.js")).read_text(encoding="utf-8")
        if "</script>" in js:  # would break the inline injection; refuse rather than corrupt the page
            raise ValueError("whyhow client asset contains </script>")
        return js
    except Exception as exc:
        return ("window.__whyhowActivate=function(){var m=document.getElementById('whystatus');"
                "if(m){m.textContent='WHY/HOW client asset unavailable: "
                + json.dumps(str(exc))[1:-1] + "';}};")


def _page_html(cfg: Config) -> str:
    # BOOT carries ONLY real sources: explicit overrides (may be null) + the pointer
    # READ from the canonical file at page-serve time (null when unavailable — the
    # client then renders "unavailable", never a baked number).
    _ptr = frontier_pointer()
    boot = json.dumps({
        "tau": cfg.tau, "l7": cfg.l7,
        "goal_dseg": cfg.goal_dseg, "goal_dseg_15": cfg.goal_dseg_15,
        "pointer": (_ptr.get("score") if _ptr.get("ok") else None),
        "poll": cfg.poll,
    })
    return (_PAGE_TEMPLATE
            .replace("__BOOT__", boot)
            .replace("__FLOW_JS__", _flow_client_js())
            .replace("__WHYHOW_JS__", _whyhow_client_js()))


_PAGE_TEMPLATE = r"""<!doctype html>
<html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Level-Set Witness — live</title>
<style>
:root{--bg:#13151a;--panel:#1b1e24;--panel2:#181b21;--fg:#e3e8f0;--fg2:#d8dde6;
--muted:#8b93a3;--faint:#5c6573;--faint2:#818996;--grid:#2a2f39;--line:#22262e;
--acc:#5ab0ff;--goal:#46d369;--pose:#ffb454;--bytes:#c08cff;--sval:#ff6b6b;
--good:#46d369;--bad:#ff6b6b}
*{box-sizing:border-box}
html{-webkit-text-size-adjust:100%;overflow-x:hidden}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',system-ui,Roboto,sans-serif;
background:var(--bg);color:var(--fg);margin:0;padding:0;line-height:1.5;
-webkit-font-smoothing:antialiased;font-variant-numeric:tabular-nums}
.wrap{max-width:1200px;margin:0 auto;padding:clamp(14px,3.5vw,26px) clamp(12px,4vw,24px) 56px;overflow-x:clip}

/* provenance status strip — factual header (FITS/lab-notebook register), NOT an alert
   box: no border-radius, no card, no amber fill. Discrete labeled fields, hairline rule
   underneath, terse mono values a data engineer scans once. */
.provh{display:flex;flex-wrap:wrap;gap:2px 26px;align-items:baseline;
font-family:ui-monospace,SFMono-Regular,Menlo,'Cascadia Mono',monospace;
font-size:11px;line-height:1.85;color:var(--muted);
border-bottom:1px solid var(--grid);padding:2px 1px 9px;margin:2px 0 16px}
.provh .pf{display:inline-flex;gap:8px;align-items:baseline;white-space:nowrap;max-width:100%}
.provh .pk{color:var(--faint);letter-spacing:.9px;font-size:9px;text-transform:uppercase;flex:0 0 auto}
.provh .pv{color:var(--fg2);font-variant-numeric:tabular-nums;white-space:normal;min-width:0}
.provh .pv b{color:var(--fg);font-weight:600}
@media(max-width:520px){.provh{gap:1px 16px;font-size:10.5px}.provh .pk{font-size:8.5px}}

/* header */
.head{display:flex;align-items:center;flex-wrap:wrap;gap:10px;margin-bottom:4px}
/* GitHub repo link — upper-right of the header, after the pills (margin-left:auto pins it right
   on wide screens; on narrow screens the flex wrap keeps it tappable at >=44px hit area). */
.ghlink{margin-left:auto;display:inline-flex;align-items:center;justify-content:center;
  width:40px;height:40px;border-radius:9px;color:var(--muted,#8b949e);flex:0 0 auto}
.ghlink:hover,.ghlink:focus-visible{color:var(--fg,#e6edf3);background:rgba(255,255,255,.06)}
.ghlink svg{width:22px;height:22px;display:block}
.title{font-size:clamp(13px,3.4vw,15px);color:var(--fg);letter-spacing:1.4px;
text-transform:uppercase;font-weight:700;margin-right:auto}
.pills{display:flex;gap:6px;flex-wrap:wrap}
.pill{font-size:11px;font-weight:600;padding:3px 9px;border-radius:2px;white-space:nowrap;line-height:1.3;
font-variant-numeric:tabular-nums;letter-spacing:.2px}
.pill.live{background:#173d22;color:#7fe0a0}.pill.warm{background:#3a3413;color:#e6cf7a}
.pill.stale{background:#4a1717;color:#ff9b9b}.pill.miss{background:#3a1f1f;color:#ff9b9b}
.pill.ws{background:#16263a;color:#7fc0ff}.pill.wsoff{background:#3a2a16;color:#e6b97a}

/* ================================================================= *
 * META-NAV (operator 2026-07-16): a two-tab navigation layer ABOVE the
 * whole instrument. Tab 1 "COMMA LAB" = the publication landing page;
 * Tab 2 "LIVE" = the entire existing dashboard (all its tabs nested
 * beneath, behavior unchanged). body.meta-lab toggles which world shows.
 * ================================================================= */
.metanav{display:flex;gap:2px;align-items:baseline;margin:0 0 10px;padding:10px 0 0}
.metatab{font-size:12px;font-weight:700;letter-spacing:2.2px;text-transform:uppercase;
  color:var(--faint2);padding:8px 14px 9px;cursor:pointer;user-select:none;
  border-bottom:2px solid transparent;-webkit-tap-highlight-color:transparent}
.metatab:hover{color:var(--fg2)}
.metatab.on{color:var(--fg);border-bottom-color:var(--acc)}
.metasep{flex:1 1 auto;border-bottom:1px solid var(--grid);align-self:flex-end;height:2px}
/* world toggle: in lab mode, hide every wrap child except the landing + the meta nav */
body.meta-lab .wrap>*:not(#meta-lab):not(.metanav){display:none}
body:not(.meta-lab) #meta-lab{display:none}

/* ── landing page (Comma Lab) — editorial register: measured serif prose over the
     instrument's ground; mono for figures and labels; one accent. It grows into the
     full writeup (deep math · geometry · topology · scorer dynamics · modeling). ── */
#meta-lab{--lab-serif:Charter,'Iowan Old Style',Georgia,'Times New Roman',serif;
  --lab-mono:ui-monospace,SFMono-Regular,Menlo,'Cascadia Mono',monospace;
  max-width:720px;margin:0 auto;padding:14px 2px 40px}
#meta-lab .lab-kicker{font-family:var(--lab-mono);font-size:10.5px;letter-spacing:2.6px;
  text-transform:uppercase;color:var(--acc);margin:18px 0 10px}
#meta-lab h1{font-family:var(--lab-serif);font-size:clamp(30px,6vw,44px);font-weight:600;
  letter-spacing:-.5px;line-height:1.12;margin:0 0 14px;color:var(--fg)}
#meta-lab .lab-lede{font-family:var(--lab-serif);font-size:clamp(16px,3.6vw,19px);
  line-height:1.62;color:var(--fg2);margin:0 0 8px}
#meta-lab .lab-meta{font-family:var(--lab-mono);font-size:11px;color:var(--muted);
  margin:14px 0 30px;padding-bottom:16px;border-bottom:1px solid var(--grid);line-height:1.9}
#meta-lab .lab-meta b{color:var(--fg2);font-weight:600}
#meta-lab h2{font-family:var(--lab-serif);font-size:clamp(20px,4.4vw,25px);font-weight:600;
  letter-spacing:-.2px;margin:38px 0 4px;color:var(--fg);line-height:1.25}
#meta-lab .lab-secno{font-family:var(--lab-mono);font-size:10px;letter-spacing:2px;
  color:var(--faint2);text-transform:uppercase;display:block;margin:0 0 6px}
#meta-lab p{font-family:var(--lab-serif);font-size:15.5px;line-height:1.72;color:var(--fg2);margin:10px 0}
#meta-lab p b{color:var(--fg);font-weight:600}
#meta-lab .lab-eq{font-family:var(--lab-mono);font-size:12.5px;color:var(--fg);
  background:var(--panel2);border:1px solid var(--line);border-radius:4px;
  padding:12px 14px;margin:14px 0;overflow-x:auto;white-space:nowrap;text-align:center}
#meta-lab .lab-note{font-family:var(--lab-mono);font-size:10.5px;color:var(--muted);
  border-left:2px solid var(--grid);padding:2px 0 2px 12px;margin:14px 0;line-height:1.7}
#meta-lab ul{margin:8px 0 8px 2px;padding-left:20px}
#meta-lab li{font-family:var(--lab-serif);font-size:15px;line-height:1.66;color:var(--fg2);margin:5px 0}
#meta-lab li b{color:var(--fg);font-weight:600}
#meta-lab .lab-cta{display:inline-block;font-family:var(--lab-mono);font-size:12px;font-weight:600;
  letter-spacing:1.2px;text-transform:uppercase;color:var(--acc);border:1px solid rgba(90,176,255,.45);
  border-radius:4px;padding:10px 18px;margin:26px 0 6px;cursor:pointer;user-select:none}
#meta-lab .lab-cta:hover{background:rgba(90,176,255,.09)}
#meta-lab .lab-foot{font-family:var(--lab-mono);font-size:10px;color:var(--faint);
  margin-top:44px;padding-top:14px;border-top:1px solid var(--grid);line-height:1.9}

/* CAMPAIGN tab (#366 joint-descent) — dense, honest, mono-numeric */
.cmp .panel{background:var(--panel);border:1px solid var(--line);border-radius:6px;
padding:12px 14px 10px;margin:0 0 14px}
.cmp .ph{font-size:11px;font-weight:700;letter-spacing:1.2px;text-transform:uppercase;
color:var(--muted);margin:0 0 8px;display:flex;flex-wrap:wrap;gap:6px 10px;align-items:baseline}
.cmp .cmp-tag{font-size:9.5px;font-weight:600;letter-spacing:.6px;padding:2px 7px;border-radius:2px;text-transform:none}
.cmp .cmp-tag.exact{background:#16263a;color:#7fc0ff}
.cmp .cmp-tag.adv{background:#3a3413;color:#e6cf7a}
.cmp canvas{width:100%;display:block}
.cmp .cmp-runline{font-family:ui-monospace,SFMono-Regular,Menlo,'Cascadia Mono',monospace;
font-size:11px;color:var(--muted);word-break:break-all;margin:0 0 8px}
.cmp .cmp-kv{display:flex;flex-wrap:wrap;gap:4px 22px;align-items:baseline;
font-family:ui-monospace,SFMono-Regular,Menlo,'Cascadia Mono',monospace;font-size:11px;
line-height:1.85;border-bottom:1px solid var(--grid);padding:0 1px 9px;margin:0 0 14px}
.cmp .cf{display:inline-flex;gap:8px;align-items:baseline;white-space:nowrap;max-width:100%}
.cmp .ck{color:var(--faint);letter-spacing:.9px;font-size:9px;text-transform:uppercase;flex:0 0 auto}
.cmp .cv{color:var(--fg2);font-variant-numeric:tabular-nums;white-space:normal;min-width:0}
.cmp .cv b{color:var(--fg);font-weight:600}
.cmp .cv .ok{color:var(--good)} .cmp .cv .bad{color:var(--bad)} .cmp .cv .warm{color:#e6cf7a}
.cmp .duo{display:grid;grid-template-columns:1fr 1fr;gap:14px}
@media(max-width:720px){.cmp .duo{grid-template-columns:1fr}}
.cmp .footnote{font-size:10px;color:var(--faint2);margin-top:7px;line-height:1.55}
.cmp .clsrow{display:flex;align-items:center;gap:10px;margin:5px 0;font-size:11px;
font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
.cmp .clsname{flex:0 0 84px;color:var(--fg2)}
.cmp .clsbarwrap{flex:1;height:12px;background:var(--panel2);border:1px solid var(--line);border-radius:2px;overflow:hidden}
.cmp .clsbar{height:100%;background:var(--acc);opacity:.75}
.cmp .clsval{flex:0 0 92px;text-align:right;color:var(--fg);font-variant-numeric:tabular-nums}
.cmp .gatebox{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:11.5px;line-height:1.9;color:var(--fg2)}
.cmp .gatebox b{color:var(--fg)}

/* tabs */
.tabs{display:flex;gap:4px;margin:16px 0 18px;border-bottom:1px solid var(--grid);
overflow-x:auto;scrollbar-width:none;-webkit-overflow-scrolling:touch}
.tabs::-webkit-scrollbar{display:none}
.tab{font-size:13px;font-weight:600;color:var(--muted);padding:12px 16px;cursor:pointer;
white-space:nowrap;flex:0 0 auto;
border-bottom:2px solid transparent;-webkit-tap-highlight-color:transparent;user-select:none}
.tab:hover{color:var(--fg2)}
.tab.on{color:var(--fg);border-bottom-color:var(--acc)}
/* mobile: tighten so more tabs are visible; the strip scrolls horizontally for the rest */
@media(max-width:520px){.tabs{gap:2px}.tab{font-size:12px;padding:11px 11px;letter-spacing:.2px}}

/* headline stat grid: 2x2 phone -> row of 4 desktop. discrete cells, no mid-stat wrap */
.metrics{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:clamp(10px,2.5vw,14px);margin-bottom:14px}
@media(min-width:1100px){.metrics{grid-template-columns:repeat(4,minmax(0,1fr))}}
.stat{display:flex;flex-direction:column;gap:5px;min-width:0;background:var(--panel2);
border:1px solid var(--line);border-radius:12px;padding:clamp(13px,3.4vw,18px)}
.stat.hero{border-color:rgba(90,176,255,.34)}
.slabel{font-size:11px;color:var(--muted);letter-spacing:.7px;text-transform:uppercase;
font-weight:600;display:flex;align-items:center;gap:7px;white-space:nowrap}
.sval,.sval2{min-width:0;max-width:100%;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;
font-weight:600;letter-spacing:-.4px}
.sval{font-size:clamp(28px,7vw,42px);color:var(--acc);line-height:1}
.sval2{font-size:clamp(18px,5vw,24px);color:var(--fg);line-height:1.1}
.ssub{font-size:10.5px;color:var(--faint2);min-height:13px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.ssub.adv{color:#b89a4a}
.trend{font-size:13px;font-weight:700}
.trend.dn{color:var(--good)}.trend.up{color:var(--bad)}.trend.fl{color:var(--muted)}

/* ================================================================= *
 * LIVE INSTRUMENT (rebuilt 2026-07-09) — dense training-run terminal.
 * Design language: near-black cool-biased ground, ONE accent (cyan),
 * semantic good/warn/bad kept SEPARATE from the accent. Hairline rules
 * + tabular alignment, NOT rounded bordered cards. tabular-nums on
 * every number; per-metric fixed precision. The structure IS the
 * notation — border-radius is not.
 * ================================================================= */
#tab-live{--lv-ink:#e7ecf3;--lv-ink2:#aeb7c6;--lv-mut:#767f90;
  --lv-hair:#242a34;--lv-hair2:#2e3542;--lv-surf:#161922;
  --lv-acc:#5ab0ff;--lv-seg:#5ab0ff;--lv-pose:#ffb454;--lv-byte:#c08cff;
  --lv-good:#46d369;--lv-warn:#e0a340;--lv-bad:#ff6b6b;
  --lv-mono:ui-monospace,SFMono-Regular,Menlo,'Cascadia Mono',monospace}
#tab-live .lv-runline{font-family:var(--lv-mono);font-size:10.5px;color:var(--lv-mut);
  letter-spacing:.2px;margin:0 1px 12px;word-break:break-all;line-height:1.5}

/* uppercase micro-label used across panels */
#tab-live .lv-k{font-size:9.5px;letter-spacing:1px;text-transform:uppercase;color:var(--lv-mut);
  font-weight:600;white-space:nowrap}

/* 0 · chain-state strip — bench->receipt->launch->run->byte-close pipeline position.
   Instrument register: hairline-linked step chips, tabular mono details, semantic states. */
#tab-live .lv-chain{display:flex;align-items:stretch;gap:0;margin:0 0 14px;overflow-x:auto;
  scrollbar-width:none;border:1px solid var(--lv-hair);background:var(--lv-surf);border-radius:4px}
/* the ID-prefixed display rules above out-specify the global .hide — re-assert it */
#tab-live .lv-chain.hide,#tab-live .lv-alarms.hide{display:none}
#tab-live .lv-chain::-webkit-scrollbar{display:none}
#tab-live .lv-cstep{display:flex;flex-direction:column;gap:3px;padding:8px 12px;min-width:0;
  flex:1 1 0;border-right:1px solid var(--lv-hair);position:relative}
#tab-live .lv-cstep:last-child{border-right:none}
#tab-live .lv-cstep .cs-k{font-size:9.5px;letter-spacing:1px;text-transform:uppercase;font-weight:600;
  color:var(--lv-mut);display:flex;align-items:center;gap:6px;white-space:nowrap}
#tab-live .lv-cstep .cs-dot{width:7px;height:7px;border-radius:50%;background:#3a4150;flex:0 0 auto}
#tab-live .lv-cstep.done .cs-dot{background:var(--lv-good)}
#tab-live .lv-cstep.active .cs-dot{background:var(--lv-acc);box-shadow:0 0 6px rgba(90,176,255,.8)}
#tab-live .lv-cstep.failed .cs-dot{background:var(--lv-bad)}
#tab-live .lv-cstep.done .cs-k{color:var(--lv-ink2)}
#tab-live .lv-cstep.active .cs-k{color:var(--lv-acc)}
#tab-live .lv-cstep.failed .cs-k{color:var(--lv-bad)}
#tab-live .lv-cstep .cs-d{font-family:var(--lv-mono);font-size:10px;color:var(--lv-mut);
  white-space:nowrap;overflow:hidden;text-overflow:ellipsis;font-variant-numeric:tabular-nums}
#tab-live .lv-chainhead{font-family:var(--lv-mono);font-size:9.5px;color:var(--lv-mut);
  padding:8px 10px;border-right:1px solid var(--lv-hair);display:flex;flex-direction:column;
  justify-content:center;gap:2px;flex:0 0 auto;max-width:34%}
#tab-live .lv-chainhead b{color:var(--lv-ink2);font-weight:600;white-space:nowrap;
  overflow:hidden;text-overflow:ellipsis;display:block;max-width:100%}
/* narrow viewports: cells stop squeezing (they collided into "BENCH RECEIPTLAUNCH…"
   at 390px) and become fixed-width swipeable stops in the strip's own x-scroll */
@media(max-width:700px){
  #tab-live .lv-cstep{flex:0 0 auto;min-width:118px;max-width:150px}
  #tab-live .lv-chainhead{max-width:none;min-width:96px;flex:0 0 auto}
}

/* 0b · confound-alarm strip — loud by design (the L1 immune layer made visible).
   .histmode = alarms from a NON-LIVE source run: muted amber-grey collapsed summary
   (honest scoping, never a false live emergency; operator round-2). */
#tab-live .lv-alarms{display:flex;flex-wrap:wrap;gap:6px;align-items:center;margin:0 0 14px;
  padding:7px 10px;border:1px solid rgba(255,107,107,.4);background:rgba(255,107,107,.07);border-radius:4px}
#tab-live .lv-alarms .al-k{font-size:9.5px;letter-spacing:1px;text-transform:uppercase;
  font-weight:700;color:var(--lv-bad);white-space:nowrap}
#tab-live .lv-alarm{font-family:var(--lv-mono);font-size:10.5px;color:#ffb3b3;
  background:rgba(255,107,107,.12);border:1px solid rgba(255,107,107,.3);
  border-radius:3px;padding:2px 7px;white-space:nowrap;font-variant-numeric:tabular-nums}
#tab-live .lv-alarms.histmode{border-color:var(--lv-hair2);background:var(--lv-surf);
  display:block;padding:0}
#tab-live .lv-alhist summary{cursor:pointer;list-style:none;padding:7px 10px;
  font-family:var(--lv-mono);font-size:10px;letter-spacing:.6px;color:var(--lv-mut);
  text-transform:uppercase;white-space:normal;line-height:1.6}
#tab-live .lv-alhist summary::-webkit-details-marker{display:none}
#tab-live .lv-alhist summary b{color:#c9a35c;font-weight:700}
#tab-live .lv-alhist summary::before{content:"▸ ";color:var(--lv-mut)}
#tab-live .lv-alhist[open] summary::before{content:"▾ "}
#tab-live .lv-alhist .alwrap{display:flex;flex-wrap:wrap;gap:6px;padding:0 10px 9px}
#tab-live .lv-alarm.hist{color:#cbb98d;background:rgba(201,163,92,.08);
  border-color:rgba(201,163,92,.3)}

/* 1 · masthead — big S left, live equation + per-term decomposition right */
#tab-live .lv-mast{display:grid;grid-template-columns:minmax(0,1fr);gap:14px 26px;
  border-top:1px solid var(--lv-hair2);padding:15px 1px 17px;margin-bottom:2px}
@media(min-width:760px){#tab-live .lv-mast{grid-template-columns:auto minmax(0,1fr);align-items:start}}
#tab-live .lv-mast-s{display:flex;flex-direction:column;gap:3px;min-width:0}
#tab-live .lv-adv{font-size:8.5px;letter-spacing:.6px;color:var(--lv-warn);border:1px solid var(--lv-hair2);
  padding:1px 5px;margin-left:6px;text-transform:uppercase;vertical-align:middle;font-weight:600}
#tab-live .lv-sval{font-family:var(--lv-mono);font-size:clamp(34px,8.5vw,54px);line-height:.98;
  font-weight:600;color:var(--lv-ink);letter-spacing:-1px;font-variant-numeric:tabular-nums}
#tab-live .lv-sref{font-family:var(--lv-mono);font-size:11px;color:var(--lv-mut);
  font-variant-numeric:tabular-nums;letter-spacing:.2px}
#tab-live .lv-sref b{color:var(--lv-ink2);font-weight:600}
#tab-live .lv-mast-eq{display:flex;flex-direction:column;gap:10px;min-width:0}
#tab-live .lv-eq{font-family:var(--lv-mono);font-size:12px;color:var(--lv-ink2);line-height:1.5;
  word-break:break-word;font-variant-numeric:tabular-nums;padding-top:2px}
#tab-live .lv-terms{display:grid;grid-template-columns:1fr;gap:1px;background:var(--lv-hair);
  border:1px solid var(--lv-hair)}
@media(min-width:560px){#tab-live .lv-terms{grid-template-columns:repeat(3,1fr)}}
#tab-live .lv-term{background:var(--bg);padding:9px 11px;display:flex;flex-direction:column;gap:5px;min-width:0}
#tab-live .lv-term .tt{display:flex;align-items:baseline;justify-content:space-between;gap:8px}
#tab-live .lv-term .ttk{font-family:var(--lv-mono);font-size:10.5px;color:var(--lv-ink2);white-space:nowrap;
  display:flex;align-items:center;gap:6px}
#tab-live .lv-term .swatch{width:8px;height:8px;flex:0 0 auto}
#tab-live .lv-term .tin{font-family:var(--lv-mono);font-size:10.5px;color:var(--lv-mut);
  font-variant-numeric:tabular-nums}
#tab-live .lv-term .tcon{font-family:var(--lv-mono);font-size:19px;font-weight:600;color:var(--lv-ink);
  line-height:1;font-variant-numeric:tabular-nums;letter-spacing:-.3px}
#tab-live .lv-term .tshare{font-size:9.5px;color:var(--lv-mut);font-variant-numeric:tabular-nums;
  display:flex;align-items:center;justify-content:space-between;gap:8px}
#tab-live .lv-bar{height:3px;background:var(--lv-hair2);position:relative;overflow:hidden}
#tab-live .lv-bar>i{position:absolute;left:0;top:0;bottom:0;display:block}

/* generic panel — hairline-topped, no rounded card */
#tab-live .lv-panel{border-top:1px solid var(--lv-hair2);padding:12px 1px 4px;margin-bottom:2px}
#tab-live .lv-ph{display:flex;align-items:baseline;justify-content:space-between;gap:12px;margin-bottom:10px}
#tab-live .lv-pt{font-size:10px;letter-spacing:1px;text-transform:uppercase;color:var(--lv-ink2);font-weight:600}
#tab-live .lv-pm{font-family:var(--lv-mono);font-size:10px;color:var(--lv-mut);text-align:right;
  font-variant-numeric:tabular-nums;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
#tab-live .lv-none{font-family:var(--lv-mono);font-size:11px;color:var(--lv-mut);padding:8px 0}
#tab-live .lv-row2{display:grid;grid-template-columns:minmax(0,1fr);gap:0}
@media(min-width:720px){#tab-live .lv-row2{grid-template-columns:repeat(2,minmax(0,1fr));gap:0 26px}}

/* 2 · descent chart */
#tab-live .lv-chartpanel canvas{width:100%;max-width:100%;height:clamp(210px,42vw,270px);
  display:block;touch-action:pan-y;background:var(--lv-surf);border:1px solid var(--lv-hair)}

/* 3 · per-class breakdown — aligned rows, dual bars (d_seg blue, flip amber) */
#tab-live .lv-classes{display:flex;flex-direction:column;gap:0}
#tab-live .lv-crow{display:grid;grid-template-columns:88px minmax(0,1fr) 52px;gap:9px;align-items:center;
  padding:6px 0;border-bottom:1px solid var(--lv-hair)}
#tab-live .lv-crow:last-child{border-bottom:0}
#tab-live .lv-cname{font-family:var(--lv-mono);font-size:11px;color:var(--lv-ink2);white-space:nowrap;
  overflow:hidden;text-overflow:ellipsis}
#tab-live .lv-cname .ci{color:var(--lv-mut)}
#tab-live .lv-cbars{display:flex;flex-direction:column;gap:3px;min-width:0}
#tab-live .lv-cbar{height:9px;background:var(--lv-hair);position:relative;overflow:hidden}
#tab-live .lv-cbar>i{position:absolute;left:0;top:0;bottom:0;display:block}
#tab-live .lv-cval{font-family:var(--lv-mono);font-size:10.5px;color:var(--lv-ink);text-align:right;
  font-variant-numeric:tabular-nums;white-space:nowrap}
#tab-live .lv-cval .cf{color:var(--lv-mut);font-size:9.5px}
#tab-live .lv-clab{display:flex;gap:12px;font-size:8.5px;letter-spacing:.5px;text-transform:uppercase;
  color:var(--lv-mut);margin:2px 0 6px;padding-left:97px}
#tab-live .lv-clab i{width:8px;height:8px;display:inline-block;margin-right:4px;vertical-align:-1px}

/* 4 · pose readiness */
#tab-live .lv-pose{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:1px;
  background:var(--lv-hair);border:1px solid var(--lv-hair)}
#tab-live .lv-pcell{background:var(--bg);padding:9px 11px;display:flex;flex-direction:column;gap:3px;min-width:0}
#tab-live .lv-pcell .pl{font-size:9px;letter-spacing:.6px;text-transform:uppercase;color:var(--lv-mut);white-space:nowrap}
#tab-live .lv-pcell .pvv{font-family:var(--lv-mono);font-size:16px;font-weight:600;color:var(--lv-ink);
  font-variant-numeric:tabular-nums;line-height:1.05;letter-spacing:-.3px}
#tab-live .lv-pcell .pvv.ok{color:var(--lv-good)}#tab-live .lv-pcell .pvv.wn{color:var(--lv-warn)}
#tab-live .lv-pnote{grid-column:1/-1;background:var(--bg);padding:8px 11px;font-size:10.5px;
  color:var(--lv-mut);line-height:1.5}
#tab-live .lv-pnote b{color:var(--lv-ink2);font-weight:600}

/* 5 · training health — scalar strip + energy-term bars */
#tab-live .lv-hscal{display:flex;flex-wrap:wrap;gap:3px 20px;margin-bottom:11px}
#tab-live .lv-hs{display:flex;align-items:baseline;gap:6px;font-family:var(--lv-mono);font-size:11px}
#tab-live .lv-hs .hk{font-size:9px;letter-spacing:.5px;text-transform:uppercase;color:var(--lv-mut)}
#tab-live .lv-hs .hv{color:var(--lv-ink);font-variant-numeric:tabular-nums}
#tab-live .lv-hs .hv.ok{color:var(--lv-good)}#tab-live .lv-hs .hv.wn{color:var(--lv-warn)}#tab-live .lv-hs .hv.bd{color:var(--lv-bad)}
#tab-live .lv-hterm{display:grid;grid-template-columns:120px minmax(0,1fr) 62px;gap:9px;align-items:center;padding:3px 0}
#tab-live .lv-hterm .hn{font-family:var(--lv-mono);font-size:10.5px;color:var(--lv-ink2);white-space:nowrap;
  overflow:hidden;text-overflow:ellipsis}
#tab-live .lv-hterm .hbar{height:7px;background:var(--lv-hair);position:relative;overflow:hidden}
#tab-live .lv-hterm .hbar>i{position:absolute;left:0;top:0;bottom:0;background:var(--lv-acc);display:block}
#tab-live .lv-hterm .hpv{font-family:var(--lv-mono);font-size:10px;color:var(--lv-ink);text-align:right;
  font-variant-numeric:tabular-nums}

/* 6 · system memory gauges */
#tab-live .lv-sys{display:flex;flex-direction:column;gap:9px}
#tab-live .lv-gauge{display:grid;grid-template-columns:74px minmax(0,1fr) 120px;gap:10px;align-items:center}
#tab-live .lv-gauge .gk{font-size:9px;letter-spacing:.5px;text-transform:uppercase;color:var(--lv-mut);white-space:nowrap}
#tab-live .lv-gtrack{height:10px;background:var(--lv-hair);position:relative;overflow:hidden}
#tab-live .lv-gtrack>i{position:absolute;left:0;top:0;bottom:0;display:block;background:var(--lv-acc)}
#tab-live .lv-gtrack>.peak{position:absolute;top:-2px;bottom:-2px;width:2px;background:var(--lv-warn)}
#tab-live .lv-gv{font-family:var(--lv-mono);font-size:10.5px;color:var(--lv-ink2);text-align:right;
  font-variant-numeric:tabular-nums;white-space:nowrap}
#tab-live .lv-gv b{color:var(--lv-ink);font-weight:600}

/* 7 · curriculum timeline */
#tab-live .lv-sched{padding:2px 0 4px}
#tab-live .lv-track{position:relative;height:30px;background:var(--lv-hair);display:flex;overflow:hidden}
#tab-live .lv-seg{position:relative;display:flex;align-items:center;justify-content:flex-start;
  border-right:1px solid var(--bg);min-width:0;overflow:hidden}
#tab-live .lv-seg .sn{font-family:var(--lv-mono);font-size:9.5px;color:var(--lv-ink2);padding-left:5px;white-space:nowrap}
#tab-live .lv-seg:last-child{border-right:0}
#tab-live .lv-marker{position:absolute;top:-3px;bottom:-3px;width:2px;background:var(--lv-acc);z-index:2}
#tab-live .lv-marker::after{content:"";position:absolute;top:-4px;left:-3px;border:4px solid transparent;
  border-top-color:var(--lv-acc)}
#tab-live .lv-sticks{display:flex;justify-content:space-between;margin-top:5px;font-family:var(--lv-mono);
  font-size:9px;color:var(--lv-mut);font-variant-numeric:tabular-nums}
/* 7b · curriculum as DERIVED — transition events, tau anneal, mechanism swim-lanes */
#tab-live .lv-tau{position:relative;height:16px;margin:9px 0 3px;border:1px solid var(--lv-hair);
  background:linear-gradient(90deg,#1f3b5f,#3a2a5f)}
#tab-live .lv-tau .tl{position:absolute;top:1px;font-family:var(--lv-mono);font-size:8.5px;color:var(--lv-ink2);
  padding:0 4px;white-space:nowrap}
#tab-live .lv-tau .tl.r{right:0}
#tab-live .lv-taucap{font-family:var(--lv-mono);font-size:9px;color:var(--lv-mut);margin:0 0 8px}
#tab-live .lv-events{display:flex;flex-direction:column;gap:4px;margin:9px 0 2px}
#tab-live .lv-evrow{display:grid;grid-template-columns:58px minmax(0,1fr) auto;gap:9px;align-items:baseline;
  font-size:11px;padding:3px 0;border-top:1px solid var(--lv-hair)}
#tab-live .lv-evrow:first-child{border-top:0}
#tab-live .lv-evrow .en{font-family:var(--lv-mono);font-size:11px;color:var(--lv-ink);font-weight:600}
#tab-live .lv-evrow .et{color:var(--lv-ink2);min-width:0;line-height:1.45}
#tab-live .lv-evrow .et .cap{color:var(--lv-mut);font-family:var(--lv-mono);font-size:9.5px}
#tab-live .lv-pill{font-family:var(--lv-mono);font-size:9px;font-weight:600;letter-spacing:.4px;
  text-transform:uppercase;padding:1px 7px;border-radius:9px;white-space:nowrap;
  color:var(--lv-mut);background:var(--lv-hair)}
#tab-live .lv-pill.fired{color:var(--lv-good);background:rgba(70,211,105,.14)}
#tab-live .lv-pill.armed{color:var(--lv-warn);background:rgba(230,207,122,.14)}
#tab-live .lv-pill.pending{color:var(--lv-mut);background:var(--lv-hair)}
#tab-live .lv-lanes{display:flex;flex-direction:column;gap:1px;margin:9px 0 2px;
  background:var(--lv-hair);border:1px solid var(--lv-hair)}
#tab-live .lv-lane{display:grid;grid-template-columns:minmax(0,1.15fr) minmax(0,1fr) auto;gap:10px;
  align-items:baseline;background:var(--bg);padding:6px 9px}
#tab-live .lv-lane .ln{font-family:var(--lv-mono);font-size:10.5px;color:var(--lv-ink);
  overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
#tab-live .lv-lane .lt{font-size:10px;color:var(--lv-ink2);min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
#tab-live .lv-lane .lt .cap{color:var(--lv-mut);font-family:var(--lv-mono);font-size:9px}
#tab-live .lv-prov{font-size:9.5px;color:var(--lv-mut);line-height:1.5;margin-top:9px;
  border-top:1px solid var(--lv-hair);padding-top:8px;word-break:break-word}
/* pose readiness — honest disengagement + unselected R1 reference + contract */
#tab-live .lv-r1{grid-column:1/-1;background:var(--bg);border:1px solid var(--lv-hair);padding:9px 11px;margin-top:1px}
#tab-live .lv-r1 .r1h{display:flex;align-items:baseline;gap:8px;flex-wrap:wrap;margin-bottom:7px}
#tab-live .lv-r1 .r1t{font-size:9px;letter-spacing:.6px;text-transform:uppercase;color:var(--lv-mut)}
#tab-live .lv-r1 .r1tag{font-family:var(--lv-mono);font-size:8.5px;color:var(--lv-warn);
  background:rgba(230,207,122,.12);padding:1px 6px;border-radius:8px}
#tab-live .lv-r1grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:9px}
#tab-live .lv-r1grid .rc{display:flex;flex-direction:column;gap:2px}
#tab-live .lv-r1grid .rk{font-size:8.5px;letter-spacing:.5px;text-transform:uppercase;color:var(--lv-mut);white-space:nowrap}
#tab-live .lv-r1grid .rv{font-family:var(--lv-mono);font-size:14px;font-weight:600;color:var(--lv-ink);
  font-variant-numeric:tabular-nums;letter-spacing:-.3px}
#tab-live .lv-r1src{font-family:var(--lv-mono);font-size:8.5px;color:var(--lv-mut);margin-top:6px;word-break:break-all}
#tab-live .lv-contract{grid-column:1/-1;background:var(--bg);padding:8px 11px;font-size:10px;
  color:var(--lv-mut);line-height:1.55}
#tab-live .lv-contract .ck{color:var(--lv-ink2);font-weight:600}
#tab-live .lv-contract .dt{font-family:var(--lv-mono);font-size:9.5px;color:var(--lv-ink2);margin-top:4px}

/* status / detail — terse mono status line */
#tab-live .lv-status{font-family:var(--lv-mono);font-size:12px;color:var(--lv-ink2);
  border-top:1px solid var(--lv-hair2);padding:11px 1px 3px;margin-top:4px;min-height:20px;line-height:1.5}
#tab-live .lv-detail{font-family:var(--lv-mono);font-size:10.5px;color:var(--lv-mut);
  margin:0 1px 16px;min-height:15px;line-height:1.5;word-break:break-word}

/* scorer breakdown card (implied-S decomposition; HONEST measured-pose primary) */
.sbreak{background:var(--panel2);border:1px solid var(--line);border-radius:12px;
padding:clamp(13px,3.2vw,17px) clamp(14px,3.4vw,18px);margin:0 0 14px}
.sbreak .sbh{font-size:11px;color:var(--muted);letter-spacing:.7px;text-transform:uppercase;
font-weight:600;display:flex;align-items:center;gap:9px;flex-wrap:wrap;margin-bottom:9px}
.sbreak .sbtag{font-size:10px;font-weight:600;color:#7fe0a0;background:#173d22;
padding:2px 8px;border-radius:8px;letter-spacing:.3px;text-transform:none}
.sbformula{font-size:12.5px;color:var(--fg2);line-height:1.5;word-break:break-word;margin-bottom:4px;
font-variant-numeric:tabular-nums}
.sbsubst{font-size:12.5px;color:var(--muted);line-height:1.55;word-break:break-word;margin-bottom:11px;
font-variant-numeric:tabular-nums}
.sbsubst b{color:var(--fg2);font-weight:600}
.sbterms{display:flex;flex-direction:column;gap:5px;margin:0 0 11px;font-variant-numeric:tabular-nums}
.sbrow{display:grid;grid-template-columns:minmax(0,1fr) auto;gap:12px;align-items:baseline;font-size:12.5px}
.sbrow .sbk{color:var(--muted);min-width:0;overflow:hidden;text-overflow:ellipsis}
.sbrow .sbv{color:var(--fg2);font-weight:600;text-align:right;white-space:nowrap}
.sbrule{height:1px;background:var(--grid);margin:3px 0 1px}
.sbrow.sbtot .sbk{color:var(--fg2);font-weight:600;white-space:normal}
.sbrow.sbtot .sbv{color:var(--acc);font-weight:700;font-size:16px}
.sbrow.sbtot.sbwarn .sbv{color:var(--bad)}
.sbrow.sbdiv .sbk{color:var(--bad)}.sbrow.sbdiv .sbv{color:var(--bad)}
.sbdeploy{font-size:11px;color:var(--faint2);line-height:1.55;margin-top:2px;word-break:break-word;
font-variant-numeric:tabular-nums}
.sbdeploy b{color:#b89a4a;font-weight:600}
.sbdeploy .sbdl{color:#b89a4a;font-weight:700;text-transform:uppercase;letter-spacing:.3px}

/* status / detail (space reserved to avoid layout shift) */
.status{font-size:14px;color:var(--fg2);margin:0 2px 4px;min-height:21px}
.detail{font-size:11.5px;color:var(--muted);margin:0 2px 18px;min-height:17px}

/* chart grid: 1 col phone -> 2x2 from 600px up */
.grid{display:grid;grid-template-columns:minmax(0,1fr);gap:clamp(12px,2.5vw,16px)}
@media(min-width:600px){.grid{grid-template-columns:repeat(2,minmax(0,1fr))}}
.panel{background:var(--panel);border:1px solid var(--grid);border-radius:12px;
padding:10px 10px 6px;min-width:0;overflow:hidden}
.panel canvas{width:100%;max-width:100%;height:clamp(196px,44vw,244px);display:block;touch-action:pan-y}

/* footer */
.foot{font-size:10.5px;color:var(--faint2);margin-top:22px;line-height:1.7;text-align:center;word-break:break-word}

/* ORACLE tab (Tab 1) — the detector + openpilot physical priors + detectability field */
.orc h2{font-size:clamp(13px,3.4vw,15px);color:var(--acc);letter-spacing:.4px;margin:16px 0 8px}
.orc h3{font-size:12.5px;color:var(--goal);margin:20px 0 8px;letter-spacing:.4px}
.orc p,.orc li{font-size:13px;color:var(--fg2);line-height:1.6}
.orc b{color:var(--fg)}.orc code{background:#11141a;color:#9fc6ff;padding:1px 5px;border-radius:5px;font-size:12px;word-break:break-word}
.orcintro{margin-bottom:6px}
.orchdr{font-size:11.5px;color:var(--faint2);margin:8px 2px 2px;font-variant-numeric:tabular-nums}
.orcstats{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px;margin:14px 0 4px}
@media(min-width:680px){.orcstats{grid-template-columns:repeat(4,minmax(0,1fr))}}
.orcstat{background:var(--panel);border:1px solid var(--grid);border-radius:11px;padding:11px 13px;display:flex;flex-direction:column;gap:2px;min-width:0}
.orcstat .ol{font-size:9.5px;color:var(--acc);letter-spacing:.5px;text-transform:uppercase;font-weight:700}
.orcstat .ov{font-size:16px;color:var(--fg);font-weight:700;font-variant-numeric:tabular-nums}
.orcstat .os{font-size:10.5px;color:var(--muted);line-height:1.35}
.orcgrid{display:flex;flex-direction:column;gap:16px;margin:14px 0 6px;max-width:760px}
.orcfig{margin:0;background:var(--panel);border:1px solid var(--grid);border-radius:12px;padding:10px;overflow-x:auto}
/* images render at natural size up to the container — never upscaled/stretched (natural 748x520 / 875x218) */
.orcfig img{display:block;max-width:100%;height:auto;border-radius:8px}
.orcfig figcaption{font-size:11px;color:var(--muted);margin-top:7px;font-variant-numeric:tabular-nums}
.orcxi{margin-top:20px;max-width:900px}
.orcxi img{display:block;max-width:100%;height:auto;border-radius:10px;border:1px solid var(--grid);background:var(--panel);margin-top:8px}
.orccredits{margin-top:22px;background:var(--panel2);border:1px solid var(--line);border-radius:12px;padding:14px 16px}
.orccredits ul{padding-left:20px;margin:6px 0}.orccredits li{margin-bottom:6px;font-size:12.5px}
.wch{font-size:10px;color:var(--acc);letter-spacing:.6px;text-transform:uppercase;font-weight:700;margin-bottom:6px}
.wcnote{font-size:10.5px;color:var(--faint2);margin-top:10px;line-height:1.55}

/* WHY / HOW tab (Tab 4) — the deep-math museum (PASS 1: shell + I.1 field + I.4 unity) */
.why h2{font-size:clamp(14px,3.6vw,17px);color:var(--fg);letter-spacing:.3px;margin:18px 0 6px;font-weight:700}
.why p,.why li{font-size:13.5px;color:var(--fg2);line-height:1.62}.why b{color:var(--fg)}
.why .m{color:var(--muted);font-size:12.5px}
.why ul{padding-left:20px}.why li{margin-bottom:7px}
/* the one idea + movement intros */
.whyhero{background:linear-gradient(180deg,#171b22,#14171d);border:1px solid var(--grid);
border-radius:16px;padding:20px 22px;margin:6px 0 4px}
.whyhero .idea{font-size:clamp(14px,3.4vw,17px);color:var(--fg);line-height:1.6;font-weight:600;margin:0}
.whyhero .idea b{color:var(--acc)}
.whyhero .sub{font-size:12px;color:var(--muted);margin:11px 0 0;line-height:1.55}
/* movement rail */
.mvrail{display:flex;gap:8px;flex-wrap:wrap;margin:18px 0 6px}
.mvchip{font-size:11px;font-weight:700;letter-spacing:.8px;text-transform:uppercase;padding:6px 13px;
border-radius:999px;border:1px solid var(--grid);color:var(--muted);cursor:pointer;user-select:none;background:var(--panel2)}
.mvchip.on.why-i{color:#bfe0ff;border-color:#2f4d6b;background:#122234}
.mvchip.on.why-ii{color:#ffd9a8;border-color:#4a3a1f;background:#241c0e}
.mvchip:hover{color:var(--fg2)}
.mvhead{font-size:11px;font-weight:800;letter-spacing:2px;text-transform:uppercase;margin:20px 0 2px}
.mvhead.i{color:#7fc0ff}.mvhead.ii{color:var(--pose)}
.mvhead .mvsub{display:block;font-size:11.5px;font-weight:500;letter-spacing:.2px;text-transform:none;color:var(--muted);margin-top:4px}
/* engraved plate */
.plate{background:var(--panel);border:1px solid var(--grid);border-radius:16px;padding:18px 20px 16px;
margin:14px 0;position:relative;overflow:hidden}
.plate.accent-i{border-top:2px solid #2f6ab0}.plate.accent-ii{border-top:2px solid #b0762f}
.plate .ptitle{font-size:14.5px;color:var(--fg);font-weight:700;margin:0 0 3px;letter-spacing:.2px}
.plate .pnum{font-size:10.5px;color:var(--faint2);font-weight:700;letter-spacing:1px;margin-right:8px}
.plate .peq{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12.5px;color:#bcd6f2;
background:#10161f;border:1px solid #1e2a38;border-radius:8px;padding:7px 11px;margin:9px 0;display:inline-block}
.plate .pcap{font-size:12.5px;color:var(--fg2);line-height:1.58;margin:8px 0 0}
.plate .pcite{font-size:10.5px;color:var(--faint2);margin-top:9px;line-height:1.5}
.plate .pcite b{color:var(--muted)}
/* WebGPU canvas stage */
.whystage{position:relative;background:#0c0f14;border:1px solid var(--line);border-radius:12px;
overflow:hidden;margin:11px 0 4px;min-height:150px}
.whystage canvas{display:block;width:100%;height:auto}
.whymsg{position:absolute;left:12px;top:11px;font-size:11.5px;color:#aeb7c6;background:rgba(12,15,20,.72);
padding:4px 9px;border-radius:7px;pointer-events:none;max-width:80%}
.whymsg.hide{display:none}
.whybadge{font-size:10px;font-weight:700;letter-spacing:.6px;padding:3px 9px;border-radius:999px;text-transform:uppercase}
.whybadge.gpu{background:#16263a;color:#7fc0ff}.whybadge.cpu{background:#3a2a16;color:#e6b97a}
.whybadge.off{background:#3a1f1f;color:#ff9b9b}
.whytop{display:flex;align-items:center;gap:10px;margin-bottom:2px;flex-wrap:wrap}
.whytop .whystatus{font-size:11.5px;color:var(--faint2);font-variant-numeric:tabular-nums}
/* control rows (sliders / toggles) */
.whyctl{display:flex;flex-direction:column;gap:9px;margin:11px 0 2px}
.whyrow{display:flex;align-items:center;gap:11px;flex-wrap:wrap}
.whyrow .rl{flex:0 0 auto;min-width:150px;font-size:11.5px;color:var(--muted)}
.whyrow .rl .rv{color:var(--acc);font-weight:700;font-variant-numeric:tabular-nums}
.whyrow input[type=range]{flex:1 1 180px;min-width:150px}
.whyseg{display:flex;gap:5px;flex-wrap:wrap}
.whyseg .sg{font-size:11px;font-weight:600;padding:4px 11px;border-radius:8px;border:1px solid var(--grid);
color:var(--muted);cursor:pointer;background:var(--panel2);user-select:none}
.whyseg .sg.on{color:var(--fg);border-color:var(--acc);background:#122234}
.whytoggle{font-size:11px;font-weight:600;padding:4px 11px;border-radius:8px;border:1px solid var(--grid);
color:var(--muted);cursor:pointer;background:var(--panel2);user-select:none}
.whytoggle.on{color:var(--goal);border-color:#2c5a3a;background:#12281b}
/* the unity correlation readout */
.whycorr{display:flex;gap:14px;flex-wrap:wrap;margin:12px 0 2px}
.whycorr .cc{background:var(--panel2);border:1px solid var(--grid);border-radius:11px;padding:9px 13px;min-width:150px}
.whycorr .cc .ck{font-size:10px;color:var(--faint2);text-transform:uppercase;letter-spacing:.5px;display:block}
.whycorr .cc .cv{font-size:19px;color:var(--fg);font-weight:700;font-variant-numeric:tabular-nums;display:block;margin:2px 0}
.whycorr .cc .cs{font-size:10.5px;color:var(--muted);line-height:1.4;display:block}
.whycorr .cc.hi .cv{color:var(--goal)}.whycorr .cc.lo .cv{color:var(--pose)}.whycorr .cc.anchor .cv{color:var(--acc)}
/* legend chips (classes) */
.whyleg{display:flex;gap:9px;flex-wrap:wrap;margin:9px 0 2px}
.whyleg .lc{font-size:11px;color:var(--muted);display:flex;align-items:center;gap:5px}
.whyleg .lc .dot{width:11px;height:11px;border-radius:3px;display:inline-block;border:1px solid #2a2f39}
/* seams for later passes */
.whyseam{background:var(--panel2);border:1px dashed var(--grid);border-radius:12px;padding:13px 16px;margin:12px 0}
.whyseam .st{font-size:10.5px;color:var(--goal);letter-spacing:1.1px;text-transform:uppercase;font-weight:700;margin:0 0 5px}
.whyseam ul{margin:4px 0 0;padding-left:18px}.whyseam li{font-size:12px;color:var(--muted);margin-bottom:4px}
.whyseam li b{color:var(--fg2)}
.whyabout{font-size:11.5px;color:var(--faint2);line-height:1.55;margin-top:10px}
.wcnote2{font-size:10.5px;color:var(--faint2);margin-top:12px;line-height:1.55}
/* §1 the spine + §4 the finale (Pass 2) — the unifying hero (violet accent) */
.mvhead.spine{color:var(--bytes)}
.plate.accent-spine,.plate.accent-fin{border-top:2px solid #7a52c0}
.why .plate code,.why .spannote code,.why .snote code{background:#11141a;color:#c6b3ff;padding:1px 5px;border-radius:5px;font-size:11px;word-break:break-word}
.whylens{background:#191526;border:1px solid #3a2c5c;border-left:3px solid var(--bytes);border-radius:10px;
padding:10px 13px;margin:10px 0;font-size:11.5px;color:var(--fg2);line-height:1.55}
.whylens b{color:var(--fg)}.whylens i{color:var(--muted)}
.whylens .lenstag{display:inline-block;font-size:9.5px;font-weight:800;letter-spacing:1.1px;text-transform:uppercase;
color:#120c1e;background:var(--bytes);padding:2px 8px;border-radius:6px;margin-right:8px}
.spinegrid{display:grid;grid-template-columns:minmax(0,1fr);gap:14px;margin:12px 0 2px}
@media(min-width:760px){.spinegrid{grid-template-columns:repeat(2,minmax(0,1fr))}}
.spinehalf{min-width:0}
.spsub{font-size:11px;font-weight:700;letter-spacing:.4px;color:var(--muted);text-transform:uppercase;margin:0 0 6px}
.whystage.light{background:#0b0e13;min-height:170px}
.spannote{font-size:11px;color:var(--faint2);line-height:1.5;margin-top:6px}.spannote b{color:var(--fg2)}
.spinelabel{display:flex;gap:10px;align-items:flex-start;flex-wrap:wrap;margin:11px 0 2px;
background:var(--panel2);border:1px solid var(--grid);border-radius:10px;padding:9px 13px}
.spinelabel .stag{flex:0 0 auto;font-size:9.5px;font-weight:800;letter-spacing:.7px;text-transform:uppercase;padding:3px 10px;border-radius:999px}
.spinelabel .stag.hard{background:#12281b;color:var(--goal);border:1px solid #2c5a3a}
.spinelabel .stag.live{background:#122234;color:var(--acc);border:1px solid #2f4d6b}
.spinelabel .stag.pde{background:#1c1630;color:var(--bytes);border:1px solid #3a2c5c}
.spinelabel .stag.soft{background:#2a2410;color:var(--pose);border:1px solid #4a3a1f}
.spinelabel .snote{flex:1 1 220px;font-size:11.5px;color:var(--fg2);line-height:1.5}.spinelabel .snote b{color:var(--fg)}
/* the finale */
.fingrid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:9px;margin:12px 0 4px}
@media(min-width:720px){.fingrid{grid-template-columns:repeat(5,minmax(0,1fr))}}
.finpanel{background:#0b0e13;border:1px solid var(--line);border-radius:10px;padding:8px 8px 6px}
.finpanel canvas{display:block;width:100%;height:auto;border-radius:6px}
.finlab{font-size:9.5px;font-weight:700;letter-spacing:.4px;text-transform:uppercase;color:var(--muted);margin:0 0 5px;text-align:center}
.fineq{text-align:center;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:15px;color:#d9c9ff;
background:#140f22;border:1px solid #33265c;border-radius:10px;padding:11px;margin:11px 0 2px;box-shadow:0 0 22px rgba(150,108,220,.18)}
.finrow{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin:11px 0 2px}
.finsix{margin-top:12px;border:1px dashed #3a2c5c;border-radius:12px;padding:0 16px;overflow:hidden;max-height:0;opacity:0;
transition:max-height .7s ease,opacity .55s ease}
.finsix.on{max-height:640px;opacity:1;padding:15px 16px}
.finsix .fsx{display:grid;grid-template-columns:minmax(0,1fr);gap:13px;align-items:center}
@media(min-width:640px){.finsix .fsx{grid-template-columns:minmax(0,230px) 1fr}}
.finsix canvas{display:block;width:100%;height:auto;border-radius:8px;background:#0b0e13}
.finsix .fsxt{font-size:12.5px;color:var(--fg2);line-height:1.6}.finsix .fsxt b{color:var(--fg)}
.finsig{font-size:12.5px;color:var(--bytes);font-weight:600;margin-top:9px}

/* triality tab */
.cardsub{font-size:9px;color:var(--faint2);font-weight:500;letter-spacing:.4px;text-transform:none;margin-left:6px}
.tribuilt{font-size:11px;color:var(--faint2);margin:6px 2px 12px;font-variant-numeric:tabular-nums}
.trileg{display:flex;flex-direction:column;gap:6px}
.trirow{font-size:11.5px;color:var(--fg2);line-height:1.5;word-break:break-word}
.trirow .tk{display:inline-block;background:#11141a;color:#9fc6ff;padding:0 5px;border-radius:4px;font-size:10px;margin-right:6px;font-weight:600}
.trimeta{font-size:10px;color:var(--faint2);letter-spacing:.3px;margin-bottom:2px}
.tri h2{font-size:clamp(13px,3.4vw,15px);color:var(--acc);letter-spacing:.4px;margin:22px 0 8px}
.tri p,.tri li{font-size:13.5px;color:var(--fg2);line-height:1.6}
.tri .m{color:var(--muted);font-size:12.5px}
.tri code{background:#11141a;color:#9fc6ff;padding:1px 5px;border-radius:5px;font-size:12px;word-break:break-word}
.tri .cards{display:grid;grid-template-columns:minmax(0,1fr);gap:12px;margin:10px 0}
@media(min-width:680px){.tri .cards{grid-template-columns:repeat(3,minmax(0,1fr))}}
.tri .card{background:var(--panel);border:1px solid var(--grid);border-radius:12px;padding:14px}
.tri .card h3{font-size:12px;color:var(--goal);margin:0 0 7px;letter-spacing:.5px;text-transform:uppercase}
.tri ol,.tri ul{padding-left:20px}.tri li{margin-bottom:5px}
/* triality credits & tribute — warm, tasteful, dark-theme */
.trilead{color:var(--fg2);border-left:2px solid var(--goal);padding-left:12px;margin:14px 0 2px;font-size:13.5px}
.trigrat{color:var(--fg2);border-left:2px solid var(--acc);padding-left:12px;font-size:13px !important;line-height:1.65;margin:14px 0 4px}
.tricredits{background:var(--panel2);border:1px solid var(--line);border-radius:14px;padding:18px 20px;margin:16px 0 8px}
.tch{font-size:11px;color:var(--goal);letter-spacing:.7px;text-transform:uppercase;font-weight:700;margin-bottom:5px}
.tcintro{font-size:12.5px !important;color:var(--muted);line-height:1.6;margin:0 0 14px}
.tcperson{margin:0 0 13px;padding:0 0 13px;border-bottom:1px solid var(--grid)}
.tcperson:last-of-type{border-bottom:none;padding-bottom:2px}
.tcperson h4{font-size:13px;color:var(--fg);margin:0 0 5px;letter-spacing:.2px;line-height:1.4}
.tcrole{font-size:10px;color:var(--acc);letter-spacing:.4px;text-transform:uppercase;font-weight:600;margin-left:8px;white-space:nowrap}
.tcperson p{font-size:12.5px !important;color:var(--fg2);line-height:1.62;margin:0}
.tcseat{color:var(--muted);font-style:italic}
.tcnote{font-size:11px;color:#b89a4a;line-height:1.6;border-top:1px solid var(--grid);padding-top:11px;margin-top:2px}
/* triality genesis + composition timeline (Chasles to yesterday) */
.trigenesis{font-size:13.5px !important;color:var(--fg2);line-height:1.66;margin:10px 0 6px}
.tri .tritl{list-style:none;padding:0;margin:12px 0 6px;display:flex;flex-direction:column;gap:0}
.tri .tritl li{display:flex;gap:12px;align-items:baseline;padding:7px 2px;border-bottom:1px solid var(--grid);margin:0}
.tri .tritl li:last-child{border-bottom:none}
.tly{flex:0 0 auto;min-width:82px;font-size:11px;color:var(--goal);font-weight:700;font-variant-numeric:tabular-nums;letter-spacing:.2px}
.tlt{font-size:12.5px !important;color:var(--fg2);line-height:1.55}
.tlcap{color:var(--fg2) !important;border-left:2px solid var(--goal);padding-left:12px;font-size:12.5px !important;margin:10px 0 4px}

/* witness tab (Tab 2) — comma10k 6-panel + Yousfi/Fridrich tribute, live over WS */
.wit h2{font-size:clamp(13px,3.4vw,15px);color:var(--acc);letter-spacing:.4px;margin:20px 0 8px}
.wit h3{font-size:12.5px;color:var(--goal);margin:20px 0 8px;letter-spacing:.4px;text-transform:uppercase}
.wit p,.wit li{font-size:13px;color:var(--fg2);line-height:1.6}
.wit b{color:var(--fg)}
.witintro{margin-bottom:6px}
.withdr{font-size:11.5px;color:var(--faint2);margin:8px 2px 2px;font-variant-numeric:tabular-nums;
word-break:break-word;line-height:1.55}
.withdr b{color:var(--fg2);font-weight:600}
.witgrid{display:flex;flex-direction:column;gap:16px;margin:12px 0 6px}
.witfig{margin:0;background:var(--panel);border:1px solid var(--grid);border-radius:12px;
padding:8px;overflow-x:auto;min-width:0}
.witfig img{display:block;width:100%;max-width:100%;height:auto;border-radius:8px}
.witfig figcaption{font-size:11.5px;color:var(--muted);margin:7px 3px 2px;font-variant-numeric:tabular-nums}
.witfig figcaption b{color:var(--fg2);font-weight:600}
.wittrip{background:var(--panel2);border:1px solid var(--line);border-radius:12px;padding:14px 16px;margin:14px 0}
.wittrip p{font-size:12.5px;margin:0 0 9px}
.witthesis{color:var(--fg2);border-left:2px solid var(--acc);padding-left:11px;margin-top:11px !important}
.witcredits{background:var(--panel2);border:1px solid var(--line);border-radius:12px;padding:14px 16px;margin:0 0 8px}
.witcredits .wch{font-size:11px;color:var(--muted);letter-spacing:.6px;text-transform:uppercase;font-weight:700;margin-bottom:8px}
.witcredits ul{padding-left:20px;margin:0 0 10px}
.witcredits li{font-size:12.5px;margin-bottom:7px}
.witcredits .wcnote,.wcnote{font-size:11px;color:#b89a4a;line-height:1.55;border-top:1px solid var(--grid);padding-top:9px;margin-top:10px}
/* (#343) structured key lists in tab intros — replaces the former wall-of-text paragraphs */
ul.witkey{padding-left:20px;margin:8px 0 10px}
ul.witkey li{font-size:12.5px;line-height:1.6;margin-bottom:6px}

/* flow tab (Tab 3) — client-side WebGPU interactive level-set field renderer */
.flow h2{font-size:clamp(13px,3.4vw,15px);color:var(--acc);letter-spacing:.4px;margin:20px 0 8px}
.flow p{font-size:13px;color:var(--fg2);line-height:1.6;max-width:820px}
.flow .m{color:var(--muted);font-size:12.5px}
.flowstub{background:var(--panel);border:1px solid var(--grid);border-radius:12px;padding:22px 20px;margin-top:12px}
.flowtop{display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin:4px 0 10px}
.flowbadge{font-size:10.5px;font-weight:700;padding:3px 10px;border-radius:999px;letter-spacing:.4px;white-space:nowrap}
.flowbadge.gpu{background:#16263a;color:#7fc0ff}
.flowbadge.cpu{background:#3a2a16;color:#e6b97a}
.flowbadge.off{background:#3a1f1f;color:#ff9b9b}
.flowstatus{font-size:11.5px;color:var(--faint2);font-variant-numeric:tabular-nums;margin-left:auto;
white-space:nowrap;overflow:hidden;text-overflow:ellipsis;min-width:0}
.flowstage{position:relative;background:#0e1014;border:1px solid var(--grid);border-radius:12px;
overflow:hidden;margin:0 0 12px}
.flowstage canvas{display:block;width:100%;height:auto;aspect-ratio:512/384;image-rendering:auto;
touch-action:none}
.flowmsg{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;text-align:center;
padding:20px;pointer-events:none}
.flowmsg.hide{display:none}
/* graceful loading panel — never a silent black void: spinner + determinate bar + readable text on a card */
.flowload{display:flex;flex-direction:column;align-items:center;gap:13px;max-width:440px;
background:rgba(14,16,20,.74);border:1px solid var(--line);border-radius:12px;padding:20px 24px}
.flowspin{width:26px;height:26px;border-radius:50%;border:2.5px solid rgba(226,232,240,.16);
border-top-color:#5ab0ff;animation:flowspin .9s linear infinite}
.flowmsgtxt{font-size:12.5px;color:var(--fg);line-height:1.55}
.flowprog{width:230px;height:5px;border-radius:3px;background:rgba(226,232,240,.12);overflow:hidden}
.flowprog.hide{display:none}
.flowprogfill{height:100%;width:0;background:linear-gradient(90deg,#5ab0ff,#46d3a0);border-radius:3px;
transition:width .4s ease}
/* warmup: no video yet -> hide the tall empty canvas, show a compact loading card (never a big black void) */
.flowstage.warming{padding:36px 16px 32px}
.flowstage.warming canvas{display:none}
.flowstage.warming .flowmsg{position:static;padding:0}
@keyframes flowspin{to{transform:rotate(360deg)}}
@media (prefers-reduced-motion: reduce){.flowspin{animation:none;border-top-color:#5ab0ff}}
.flowctl{display:grid;grid-template-columns:minmax(0,1fr);gap:12px;background:var(--panel2);
border:1px solid var(--line);border-radius:12px;padding:13px 15px;margin:0 0 12px}
@media(min-width:680px){.flowctl{grid-template-columns:repeat(2,minmax(0,1fr))}}
.flowrow{display:flex;flex-direction:column;gap:6px;min-width:0}
.flowrow .fl{font-size:10.5px;color:var(--muted);letter-spacing:.5px;text-transform:uppercase;
font-weight:600;display:flex;justify-content:space-between;gap:8px}
.flowrow .fl .fv{color:var(--acc);font-weight:700;font-variant-numeric:tabular-nums;text-transform:none;letter-spacing:0}
.flowrow input[type=range]{width:100%;accent-color:var(--acc);height:22px;cursor:pointer}
.flowrow select{background:#11141a;border:1px solid var(--grid);color:var(--fg2);border-radius:7px;
padding:6px 9px;font-size:12.5px;width:100%}
.flowscrub{grid-column:1/-1;display:flex;align-items:center;gap:10px;flex-wrap:wrap}
.flowscrub .sc{flex:1 1 220px;min-width:0}
.flowplay{background:#173d22;color:#7fe0a0;border:0;padding:8px 14px;border-radius:8px;font-size:13px;
font-weight:600;cursor:pointer;white-space:nowrap;-webkit-tap-highlight-color:transparent}
.flowplay.on{background:#3a3413;color:#e6cf7a}
.flowlegend{display:flex;flex-wrap:wrap;gap:6px 12px;align-items:center;margin:2px 2px 12px;font-size:11px;color:var(--muted)}
.flowlegend .lc{display:inline-flex;align-items:center;gap:5px;white-space:nowrap;cursor:pointer;user-select:none;
padding:2px 6px;border-radius:6px}
.flowlegend .lc.on{background:#1b1e24}
.flowlegend .lc .dot{width:10px;height:10px;border-radius:2px;display:inline-block;flex:0 0 auto;border:1px solid #2a2f39}
.flowcap{background:var(--panel2);border:1px solid var(--line);border-radius:12px;padding:13px 16px;margin:0 0 8px}
.flowcap p{font-size:12.5px;color:var(--fg2);line-height:1.6;margin:0 0 8px;max-width:none}
.flowcap p:last-child{margin-bottom:0}
.flowcap .fcnote{font-size:11px;color:#b89a4a;line-height:1.55;border-top:1px solid var(--grid);padding-top:9px;margin-top:2px}

/* n-badge (which run am I watching) */
.nbadge{font-size:11px;font-weight:700;padding:3px 9px;border-radius:2px;white-space:nowrap;letter-spacing:.3px;line-height:1.3}
.nbadge.doe{background:#16263a;color:#7fc0ff}
.nbadge.scored{background:#173d22;color:#7fe0a0}
.nbadge.other{background:#2a2f39;color:#c2c9d4}
.runinfo{font-size:11px;color:var(--faint2);margin:0 2px 12px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}

/* #205 run-info strip (rld._run_info_html) — SCOPED under .rinfo so its .grid/.card/.fill
   never restyle the chart .grid/.tri .card (specificity 0,2,0 > server 0,1,0) */
.rinfo{margin:0 0 16px}
.rinfo:empty{display:none}
.rinfo .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:10px;max-width:100%;margin:0;text-align:left}
.rinfo .card{background:var(--panel);border:1px solid var(--grid);border-radius:10px;padding:10px 12px;min-width:0}
.rinfo .clabel{font-size:10px;color:var(--muted);letter-spacing:.6px;text-transform:uppercase}
.rinfo .cval{font-size:17px;color:var(--fg);font-weight:600;margin:3px 0;font-variant-numeric:tabular-nums;word-break:break-word}
.rinfo .csub{font-size:10.5px;color:var(--faint2);line-height:1.5;word-break:break-word}
.rinfo .bar{height:5px;background:var(--grid);border-radius:3px;margin:5px 0 4px;overflow:hidden}
.rinfo .fill{height:100%;border-radius:3px}
.rinfo .badge{font-size:10px;font-weight:600;color:#46d369;background:#173d22;padding:1px 6px;border-radius:8px;vertical-align:middle}
/* phone: strip tables (checkpoint ledger etc.) have inline-styled div parents with no
   class — scroll the table inside its own card instead of forcing horizontal page scroll */
.rinfo div:has(> table){overflow-x:auto;max-width:100%;-webkit-overflow-scrolling:touch}

/* stage legend strip */
.slegend,.lv-legend{display:flex;flex-wrap:wrap;gap:7px 13px;align-items:center;margin:2px 2px 12px;font-size:11px;color:var(--muted)}
.slegend .sc,.lv-legend .sc{display:inline-flex;align-items:center;gap:5px;white-space:nowrap}
.slegend .dot,.lv-legend .dot{width:9px;height:9px;border-radius:2px;display:inline-block;flex:0 0 auto}
.slegend .sc.off,.lv-legend .sc.off{opacity:.38}
.lv-legend .lv-legsp{margin-left:auto}
.lv-legend .dot.dsh{border:1px dashed #9aa3b2;background:transparent;border-radius:0}

/* projection block (naive linear, advisory) */
.proj{font-size:11.5px;color:var(--muted);margin:0 2px 16px;line-height:1.6;min-height:34px}
.proj .proj2{color:var(--faint2);font-size:11px}
.proj b{color:var(--fg2);font-weight:600;font-variant-numeric:tabular-nums}
/* projection block as LABELED ROWS (operator 2026-07-07 mobile-clutter fix): reuses
   the SETUP panel's cfgrow/cfgk/cfgv system, scoped so VALUES WRAP (the SETUP panel's
   nowrap values are short; projection values are sentences — right-aligned, wrapping) */
.proj .cfgrows{margin-top:2px;gap:4px}
.proj .cfgrow{font-size:11.5px;align-items:baseline}
.proj .cfgk{flex:0 0 auto}
.proj .cfgv{white-space:normal;text-align:right;overflow-wrap:anywhere;font-weight:500;
min-width:0;flex:1;color:var(--muted)}
.proj .cfgv b{color:var(--fg2)}
.proj .cfgmeta{margin-top:7px}
.curchip{font-size:9px;font-weight:700;color:#7fe0a0;background:#173d22;border-radius:999px;
padding:1px 6px;letter-spacing:.3px;flex:0 0 auto;vertical-align:middle}
/* RUN-IDENTITY row: name + purpose/scope chips (conditional; mobile: wraps, never
   overflows — mono run name breaks anywhere, chips wrap whole) */
.runid{display:flex;flex-wrap:wrap;align-items:center;gap:6px 10px;margin:2px 0 10px;min-width:0}
.runid .rname{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
font-size:11.5px;color:var(--muted);word-break:break-all;min-width:0}
/* run-identity chips notated as tagged fields (accent hairline on the left), NOT rounded
   pills — same no-card discipline as the masthead + panels. */
.runid .ridchip{font-size:10.5px;font-weight:600;padding:1px 0 1px 9px;border-radius:0;
border:0;border-left:2px solid var(--grid);color:var(--fg2);background:transparent;max-width:100%;
overflow-wrap:anywhere;line-height:1.4}
.runid .ridchip .prov{color:var(--faint2);font-weight:500}

/* costate controller SENSE/DECIDE panel (conditional; hidden when no shadow file) */
.costate{background:var(--panel2);border:1px solid var(--line);border-radius:12px;
padding:10px 14px;margin:0 0 16px}
.costate .csh{font-size:11px;color:var(--muted);letter-spacing:.7px;text-transform:uppercase;
font-weight:600;margin-bottom:5px}
.costate .cstag{font-size:10px;font-weight:600;color:#7fe0a0;background:#173d22;
padding:1px 7px;border-radius:999px;margin-left:6px;text-transform:none;letter-spacing:0}
.costate .csbody{font-size:11.5px;color:var(--muted);line-height:1.6}
.costate .csbody b{color:var(--fg2);font-weight:600;font-variant-numeric:tabular-nums}

/* live verdict pulse on the liveness pill */
.pill.beat{animation:beatpulse 1s ease-out 1}
@keyframes beatpulse{0%{box-shadow:0 0 0 0 rgba(127,224,160,.55)}100%{box-shadow:0 0 0 11px rgba(127,224,160,0)}}

/* new-best d_seg celebration badge */
.nbest{position:fixed;left:50%;transform:translateX(-50%);bottom:18px;z-index:60;
background:linear-gradient(180deg,#173d2c,#12301f);border:1px solid #2f6e4a;color:#9af0c0;
font-size:13px;font-weight:600;padding:9px 16px;border-radius:999px;box-shadow:0 6px 24px rgba(0,0,0,.45);
opacity:0;pointer-events:none;transition:opacity .35s ease}
.nbest.show{opacity:1;animation:nbpulse 1.1s ease-in-out 2}
@keyframes nbpulse{0%,100%{box-shadow:0 6px 24px rgba(0,0,0,.45)}50%{box-shadow:0 0 0 7px rgba(70,211,160,.16),0 6px 24px rgba(0,0,0,.45)}}

/* canvas tooltip (touch + hover) */
.tip{position:fixed;z-index:70;pointer-events:none;background:rgba(20,23,29,.97);
border:1px solid #333a47;border-radius:9px;padding:8px 10px;font-size:11.5px;color:var(--fg2);
line-height:1.5;box-shadow:0 6px 22px rgba(0,0,0,.5);min-width:150px;max-width:230px;
opacity:0;transition:opacity .12s ease;font-variant-numeric:tabular-nums}
.tip.show{opacity:1}
.tip .te{color:#7fc0ff;font-weight:700;margin-bottom:4px}
.tip .tr{display:flex;justify-content:space-between;gap:16px}
.tip .tk{color:var(--muted)}.tip .tv{color:var(--fg);font-weight:600}

@media(prefers-reduced-motion:reduce){
  .nbest,.nbest.show{transition:none;animation:none}
  .pill.beat{animation:none}
}

/* setup / config / schedule / curriculum panel (full observability) */
.cfg{background:var(--panel2);border:1px solid var(--line);border-radius:12px;margin:0 0 16px}
.cfg>summary{cursor:pointer;list-style:none;padding:11px 14px;font-size:11px;font-weight:700;
color:var(--fg2);letter-spacing:.5px;text-transform:uppercase;user-select:none}
.cfg>summary::-webkit-details-marker{display:none}
.cfg>summary::before{content:"\25B8  ";color:var(--muted)}
.cfg[open]>summary::before{content:"\25BE  "}
.cfgbody{padding:2px 14px 14px;display:grid;grid-template-columns:minmax(0,1fr);gap:14px}
@media(min-width:680px){.cfgbody{grid-template-columns:repeat(2,minmax(0,1fr))}}
/* wide config/telemetry tables scroll inside their own panel — the page body never
   scrolls horizontally (spec). display:block makes the table an overflow container. */
.cfgbody table{display:block;overflow-x:auto;max-width:100%;-webkit-overflow-scrolling:touch}
.cfgsec{min-width:0}
.cfgsec.full{grid-column:1/-1}
.cfgh{font-size:10px;color:var(--acc);letter-spacing:.6px;text-transform:uppercase;font-weight:700;margin-bottom:6px}
.cfgrows{display:flex;flex-direction:column;gap:3px}
.cfgrow{display:flex;justify-content:space-between;gap:14px;font-size:12px;min-width:0}
.cfgk{color:var(--muted);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;display:flex;align-items:center;gap:6px;min-width:0}
.cfgv{color:var(--fg2);font-weight:600;white-space:nowrap;font-variant-numeric:tabular-nums}
.cfgsubh{font-size:9.5px;color:var(--faint2);letter-spacing:.6px;text-transform:uppercase;font-weight:700;margin:9px 0 4px}
.cfgrows+.cfgsubh{margin-top:11px}
.cfgrow.off{opacity:.62}
.cfgv.dis{color:var(--faint2);font-weight:500;white-space:normal;text-align:right;font-variant-numeric:normal}
.cdot{width:9px;height:9px;border-radius:2px;display:inline-block;flex:0 0 auto}
.cfgmeta{font-size:11px;color:var(--faint2);margin-top:6px}

/* ── #352 schema-driven introspection: schedule classification + controller + telemetry ── */
/* classification chips — semantic STATE colours kept separate from the accent (operator:
   armed/fired/stale distinct from accent). event=amber · derived=violet · fixed/cap=blue. */
.kchip{font-size:9px;font-weight:800;letter-spacing:.5px;text-transform:uppercase;padding:2px 7px;
border-radius:999px;flex:0 0 auto;line-height:1.35}
.kchip.event{background:#3a2f10;color:#f0c264}.kchip.derived{background:#2b1f3d;color:#cba6f5}
.kchip.fixed{background:#16263a;color:#7fc0ff}.kchip.cap{background:#23282f;color:#9aa3b2}
/* live arm status dot (armed/pending vs fired vs stale) */
.sdot{width:8px;height:8px;border-radius:50%;display:inline-block;flex:0 0 auto}
.sdot.fired{background:var(--good);box-shadow:0 0 0 3px rgba(70,211,105,.14)}
.sdot.pending{background:#f0c264;box-shadow:0 0 0 3px rgba(240,194,100,.12)}
.sdot.scheduled{background:#7fc0ff}
.stchip{font-size:9.5px;font-weight:600;color:var(--faint2);white-space:nowrap}
.stchip.fired{color:var(--good)}.stchip.pending{color:#e6c072}
/* schedule element rows (schema-driven) */
.schrow{display:flex;align-items:center;gap:8px;font-size:12px;padding:5px 0;min-width:0;
border-bottom:1px solid rgba(42,47,57,.55)}
.schrow:last-child{border-bottom:none}
.schrow .snm{color:var(--fg2);font-weight:600;display:flex;align-items:center;gap:7px;flex:0 0 auto}
.schrow .sbody{color:var(--muted);min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;flex:1 1 auto}
.schrow .sright{margin-left:auto;display:flex;align-items:center;gap:7px;flex:0 0 auto}
.schrow .strig{font-size:10.5px;color:var(--faint2);white-space:normal;line-height:1.4}
/* controller λ table */
.lamtab{display:flex;flex-direction:column;gap:2px;margin-top:2px}
.lamrow{display:grid;grid-template-columns:minmax(0,1.3fr) auto auto;gap:10px;align-items:baseline;
font-size:11.5px;padding:2px 0}
.lamrow .lnm{color:var(--muted);min-width:0;overflow:hidden;text-overflow:ellipsis}
.lamrow .lv{color:var(--fg2);font-weight:700;text-align:right;font-variant-numeric:tabular-nums}
.lamrow .lst{font-size:9px;font-weight:700;letter-spacing:.4px;text-transform:uppercase;text-align:right}
.lst.analytic{color:#7fc0ff}.lst.identified{color:var(--good)}.lst.unidentifiable{color:var(--faint2)}
.csgrid{display:grid;grid-template-columns:minmax(0,1fr);gap:12px;margin-top:4px}
@media(min-width:640px){.csgrid{grid-template-columns:repeat(2,minmax(0,1fr))}}
.cscell{min-width:0}
.csk{font-size:9.5px;color:var(--acc);letter-spacing:.5px;text-transform:uppercase;font-weight:700;margin-bottom:5px}
.csline{font-size:11px;color:var(--muted);line-height:1.6}.csline b{color:var(--fg2);font-weight:600}
/* liveness strip (confound-immune) */
.livestrip{display:flex;flex-wrap:wrap;gap:6px 8px;margin-top:9px;padding-top:9px;border-top:1px solid var(--grid)}
.lvpill{font-size:10px;font-weight:600;padding:3px 9px;border-radius:999px;background:#181b21;
border:1px solid var(--line);color:var(--muted);font-variant-numeric:tabular-nums}
.lvpill b{color:var(--fg2);font-weight:700}
.lvpill.ok{border-color:#274a34;color:#8fe0ac}.lvpill.warn{border-color:#5a2323;color:#ff9b9b}
.lvpill.alarm{background:#3a1717;border-color:#7a2b2b;color:#ff9b9b;font-weight:700}
/* planned curves — inline SVG sparklines */
.crv{display:grid;grid-template-columns:minmax(0,1fr);gap:12px}
@media(min-width:560px){.crv{grid-template-columns:repeat(3,minmax(0,1fr))}}
.crvcell{background:var(--panel);border:1px solid var(--grid);border-radius:10px;padding:9px 11px;min-width:0}
.crvh{display:flex;align-items:baseline;gap:8px;margin-bottom:4px;flex-wrap:wrap}
.crvn{font-size:11px;color:var(--fg2);font-weight:700}
.crvsh{font-size:9.5px;color:var(--faint2);text-transform:uppercase;letter-spacing:.4px}
.crvsvg{width:100%;height:46px;display:block}
.crvep{font-size:9.5px;color:var(--muted);margin-top:3px;font-variant-numeric:tabular-nums;display:flex;justify-content:space-between;gap:8px}
.crvnote{font-size:9.5px;color:var(--faint2);margin-top:4px;line-height:1.4}
/* constants manifest table */
.cst{display:flex;flex-direction:column;gap:1px}
.cstrow{display:grid;grid-template-columns:minmax(0,1.1fr) auto auto;gap:10px;align-items:center;
font-size:11.5px;padding:6px 0;border-bottom:1px solid rgba(42,47,57,.5)}
.cstrow:last-child{border-bottom:none}
.cstnm{color:var(--fg2);font-weight:600;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;
font-size:11px;min-width:0;overflow:hidden;text-overflow:ellipsis}
.cstv{color:var(--acc);font-weight:700;text-align:right;font-variant-numeric:tabular-nums}
.cstprov{grid-column:1/-1;font-size:10px;color:var(--faint2);line-height:1.45;margin-top:-2px;
overflow:hidden;text-overflow:ellipsis;display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical}
.cstprov code{background:#11141a;color:#9fc6ff;padding:0 4px;border-radius:4px;font-size:9.5px}
/* mem_probe bars */
.membars{display:flex;flex-direction:column;gap:6px;margin-top:2px}
.memrow{display:grid;grid-template-columns:minmax(0,1fr) auto;gap:10px;align-items:center;font-size:11px}
.memk{color:var(--muted);min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.memtrack{grid-column:1/-1;height:5px;background:#11141a;border-radius:3px;overflow:hidden;margin-top:-3px}
.memfill{height:100%;background:linear-gradient(90deg,#5ab0ff,#c08cff);border-radius:3px}
.memv{color:var(--fg2);font-weight:600;font-variant-numeric:tabular-nums;text-align:right;white-space:nowrap}
/* fired-event markers (diamond glyph, distinct from epoch ticks) */
.evlist{display:flex;flex-wrap:wrap;gap:6px;margin-top:2px}
.evchip{display:inline-flex;align-items:center;gap:6px;font-size:10.5px;color:var(--fg2);
background:#181b21;border:1px solid var(--line);border-radius:8px;padding:3px 9px}
.evdia{width:8px;height:8px;background:var(--good);transform:rotate(45deg);flex:0 0 auto}
.evep{color:var(--muted);font-variant-numeric:tabular-nums}
.hide{display:none}

/* ================================================================= *
 * DE-CARD OVERRIDE (2026-07-09) — extend the LIVE instrument language
 * to EVERY tab: NO rounded bordered cards. Structure via hairline rules
 * (#242a34 / #2e3542), tabular alignment, typographic hierarchy. These
 * rules come LAST so they win by cascade; they touch ONLY border /
 * border-radius / background — all padding/layout/content preserved.
 * ================================================================= */
/* section / collapsible / caption cards -> hairline-topped panels */
.cfg,.costate,.flowcap,.flowstub,.flowctl,.wittrip,.witcredits,.orccredits{
  background:transparent;border:0;border-top:1px solid #2e3542;border-radius:0}
/* stat-tile groups -> hairline grid (masthead-terms notation) */
.orcstats{gap:1px;background:#242a34;border:1px solid #242a34}
.orcstat{background:#13151a;border:0;border-radius:0}
.rinfo .grid{gap:1px;background:#242a34;border:1px solid #242a34;padding:0}
.rinfo .card{background:#13151a;border:0;border-radius:0}
/* image figures -> no card; a single hairline frame on the image itself */
.orcfig,.witfig{background:transparent;border:0;border-radius:0;padding:0}
.orcfig img,.witfig img,.orcxi img{border-radius:0;border:1px solid #242a34}
.orcxi img{background:transparent}
.orcfig figcaption,.witfig figcaption{margin-left:1px}
/* render stages -> hairline frame, cool-bias surface, no radius */
.flowstage{background:#161922;border:1px solid #242a34;border-radius:0}
.flowload{background:rgba(19,21,26,.82);border:1px solid #2e3542;border-radius:0}
/* code / eikonal chips -> squared notation */
.orc code{border-radius:2px}
.wcnote,.witcredits .wcnote,.flowcap .fcnote{border-radius:0}
/* pills / badges / tags / chips -> squared notation (status tags, not lozenges) */
.pill,.nbadge,.rinfo .badge,.flowlegend .lc{border-radius:0}
.flowbadge,.curchip,.costate .cstag,.sbreak .sbtag,.flowplay,.flowrow select,.evchip{border-radius:2px}
/* floating popovers kept minimal (not cards) */
.tip{border-radius:2px}.nbest{border-radius:2px}
/* terse labeled-field list (replaces prose paragraphs; scanned, not read) */
.kvlist{display:flex;flex-direction:column;gap:0;margin:8px 0 10px}
.kv{display:grid;grid-template-columns:120px minmax(0,1fr);gap:12px;align-items:baseline;
  padding:5px 0;border-bottom:1px solid #242a34}
.kv:last-child{border-bottom:0}
.kvk{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:10px;letter-spacing:.5px;
  text-transform:uppercase;color:var(--muted);white-space:nowrap}
.kvv{font-size:12px;color:var(--fg2);line-height:1.5;font-variant-numeric:tabular-nums;overflow-wrap:anywhere}
.kvv b{color:var(--fg);font-weight:600}
@media(max-width:520px){.kv{grid-template-columns:96px minmax(0,1fr);gap:9px}.kvk{font-size:9px}}
/* terse one-line caption under a heading/chart (replaces explanatory sentences) */
.orccap,.witcap,.flowcapline{font-size:11.5px;color:var(--muted);line-height:1.5;margin:2px 1px 6px;
  font-variant-numeric:tabular-nums;overflow-wrap:anywhere}
.orccap code,.witcap code,.flowcapline code{background:#11141a;color:#9fc6ff;padding:1px 5px;border-radius:2px;font-size:11px}
.orccap b,.witcap b,.flowcapline b{color:var(--fg);font-weight:600}
</style></head>
<body><div class="wrap">
<!-- META-NAV (operator 2026-07-16): two-tab layer above the whole instrument.
     COMMA LAB = the publication landing page (the site's front door; grows into the full
     writeup). LIVE = the entire existing dashboard, all its tabs nested beneath, unchanged.
     Deep links: #lab -> landing; #live or #live/<tab> or a bare legacy #<tab> -> LIVE. -->
<nav class="metanav" role="tablist" aria-label="site sections">
  <span class="metatab" data-meta="lab" role="tab">Comma Lab</span>
  <span class="metatab" data-meta="live" role="tab">Live</span>
  <span class="metasep"></span>
</nav>

<section id="meta-lab" aria-label="Comma Lab — writeup">
  <div class="lab-kicker">an open research program</div>
  <h1>Comma Lab</h1>
  <p class="lab-lede">Notes on compressing a driving video for a <b>frozen machine judge</b> —
  the geometry, topology, and dynamics of coding for a fixed pair of neural scorers, built in
  the open around the comma.ai video-compression challenge and continued past its close as a
  long-horizon research program.</p>
  <div class="lab-meta">
    status <b>working notes · grows over time</b> &middot; live instrument under the
    <b>LIVE</b> tab &middot; every training number there is advisory; only a byte-closed exact
    evaluation moves the frontier pointer (<span class="ptrv">&hellip;</span>)
  </div>

  <span class="lab-secno">§ 1</span>
  <h2>The problem &amp; the frozen scorer</h2>
  <p>The task looks like video compression but is not: the receiver is not a human eye, it is a
  <b>frozen pair of networks</b>. A segmentation U-Net scores only the <b>argmax</b> of its
  5-class output on the last frame of each pair; a pose network scores 6 ego-motion scalars from
  a two-frame YUV stack; the third term is the raw archive size:</p>
  <div class="lab-eq">S&nbsp;=&nbsp;100·d_seg&nbsp;+&nbsp;&radic;(10·d_pose)&nbsp;+&nbsp;25·bytes&nbsp;/&nbsp;37,545,489</div>
  <p>That makes this an instance of <b>indirect rate–distortion</b> — coding for machines, in
  the video-coding-for-machines lineage — where the only bits that matter are the ones the
  frozen judge can see. Pixels the scorer is blind to are free; pixels that flip an argmax at a
  class boundary are everything. Our early on-ramp came from steganography: content-adaptive
  embedding costs (UNIWARD) are exactly a detector-informed sensitivity field read in reverse.</p>

  <span class="lab-secno">§ 2</span>
  <h2>Geometry &amp; topology of the argmax</h2>
  <p>The argmax of a smooth 5-class field partitions the image into cells whose walls — the
  <b>separatrices</b> — form a codimension-1 complex. Measured on the real scorer, essentially
  all of the segmentation distortion lives in a thin annulus around those walls (~97% of d_seg
  in a few percent of the area); the interior of each cell is flat. In the frozen scorer's
  <b>Fisher information metric</b> the margin field is an almost-exact surrogate for that
  geometry (Pearson&nbsp;0.978 measured), so the whole objective becomes boundary geometry: a
  <b>Morse–Smale complex</b> over the margin field, with lane markings as the thinnest,
  least-persistent — and therefore hardest — stratum. The final network layer is exactly
  low-rank linear, so flip distances have a closed form, and the partition itself is a
  <b>Laguerre / tropical</b> power diagram: store generators, not pixels.</p>

  <span class="lab-secno">§ 3</span>
  <h2>The four legs</h2>
  <ul>
    <li><b>Kolmogorov, not entropy.</b> The decoder ships as a <i>program</i>: generic
    deterministic structure is free at decode time; only the irreducible video-derived seed is
    counted. Rate = |shortest program| + |seed|, a compression-as-shortest-program discipline
    rather than a histogram one.</li>
    <li><b>Projection.</b> With the geometry solved, fitting is a projection: the witness is the
    projection of the scene onto the intersection of the argmax, pose, quantization, and byte
    constraints, taken in the scorer's own metric.</li>
    <li><b>Realization.</b> The binding limits are physical: uint8 quantization, resize kernels,
    sub-pixel boundary placement. Many boundary flips are realization-limited, not
    capacity-limited — precision matters where bytes do not.</li>
    <li><b>Completeness.</b> Necessity by inversion: for each stratum of the complex, ask what
    the scorer <i>requires</i> — which bytes go to edges, which precision goes to saddles — and
    ship nothing else.</li>
  </ul>

  <span class="lab-secno">§ 4</span>
  <h2>Modeling the witness</h2>
  <p>The vehicle is a <b>task-space coordinate INR</b>: a small implicit network trained against
  the frozen scorer itself — never against RGB fidelity — so its whole capacity is spent on the
  scorer-relevant manifold. Training runs through the exact evaluation round-trip (resize,
  uint8, resize) so the gradient sees what the judge sees. The current generation adds
  <b>phase/advection structure</b>: frame-to-frame appearance is carried by a low-dimensional
  ego-motion screw and a sub-pixel advection phase, with per-class carriers for the strata the
  scorer treats differently. Ego-motion is dual-use by construction — the same twist that warps
  the partition for segmentation <i>is</i> the pose the second network scores.</p>

  <span class="lab-secno">§ 5</span>
  <h2>Scorer dynamics</h2>
  <p>Treating the frozen scorer as a physical system pays: margin fields behave like potentials,
  training follows a level-set flow of the boundary complex, and curriculum boundaries act like
  continuation parameters — instabilities arrive as bifurcations (island births) that can be
  anticipated rather than suffered. The scorer's own architecture sets the physics: a stride-2
  stem means it sees regions, not pixels; its effective receptive field, its squeeze-excitation
  gates, and its exact resize kernels are all measured and folded into the model of what a byte
  can buy.</p>

  <span class="lab-secno">§ 6</span>
  <h2>Results &amp; frontier</h2>
  <p>Everything in the LIVE instrument is <b>advisory telemetry</b> from local training
  hardware; a result is real only when an exact, byte-closed archive is scored by the contest
  evaluator on reference hardware. That number — the frontier pointer, currently
  <b><span class="ptrv">&hellip;</span></b> — moves only through that gate. The working targets
  are the sub-0.19 and sub-0.15 lines; the measured rate-dominated floor of the current
  formulation sits well below both, which is the headroom the program is spending down.</p>
  <div class="lab-note">methods and mathematics on this page are published freely; the live
  instrument shows the run of record. Attribution: video-coding-for-machines is the problem's
  heart; adaptive steganographic cost was the on-ramp; the separatrix / Morse–Smale treatment
  of a frozen argmax judge is this lab's own line of work.</div>
  <span class="lab-cta" id="lab_open_live" role="button" tabindex="0">Open the live instrument &rarr;</span>
  <div class="lab-foot">comma lab &middot; working notes &middot; this page grows with the
  program — deep math &middot; geometry &middot; topology &middot; scorer dynamics &middot; modeling</div>
</section>
<div class="provh" role="note" aria-label="provenance">
  <span class="pf"><span class="pk">authority</span><span class="pv">macOS-MLX &middot; advisory &middot; non-promotable</span></span>
  <span class="pf"><span class="pk">pointer</span><span class="pv"><b class="ptrv">&hellip;</b> &middot; unmoved</span></span>
  <span class="pf"><span class="pk">exact row</span><span class="pv">byte-closed &middot; contest-CPU / CUDA</span></span>
  <span class="pf"><span class="pk">basis</span><span class="pv">a dashboard is a means, not the score</span></span>
</div>
<div class="head">
  <span class="title">Level-Set Witness</span>
  <span class="pills">
    <span id="npill" class="nbadge other">n=?</span>
    <span id="pill" class="pill miss">&middot; connecting</span>
    <span id="wspill" class="pill wsoff">ws &hellip;</span>
  </span>
  <a class="ghlink" href="https://github.com/adpena/comma-lab" target="_blank" rel="noopener"
     aria-label="Source on GitHub (adpena/comma-lab)" title="adpena/comma-lab on GitHub">
    <svg viewBox="0 0 16 16" fill="currentColor" aria-hidden="true">
      <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38
      0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01
      1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95
      0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0
      1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0
      3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.012
      8.012 0 0 0 16 8c0-4.42-3.58-8-8-8Z"/>
    </svg>
  </a>
</div>
<!-- RUN-IDENTITY row (operator 2026-07-07): run name + purpose chip + scope chip,
     directly under the pills row, above the tab bar. CONDITIONAL: hidden until a
     run resolves; chips render only with a real source (declared header / derived
     heuristic, provenance-labelled). Flex-wrap + break-all keep ~390px viewports
     free of horizontal scroll. -->
<div class="runid hide" id="runid" aria-label="run identity">&nbsp;</div>
<div class="tabs">
  <div class="tab on" data-tab="live">LIVE</div>
  <div class="tab" data-tab="campaign">CAMPAIGN</div>
  <div class="tab" data-tab="oracle">ORACLE</div>
  <div class="tab" data-tab="flow">WITNESS</div>
  <div class="tab" data-tab="witness">RESIDUAL</div>
  <!-- SANDBOX tab HIDDEN per operator 2026-07-08 ("hide the sandbox tab for now") —
       section + content machinery stay intact; restore = uncomment in the block below.
       TODO(#343): WHY/HOW tab HIDDEN per operator 2026-07-07 ("needs a lot of work") — copy
       rework required before re-show: break up the big blocks, direct technical register,
       fix tribute framing (pcap "The tribute's heart", the About/credits panel).
       TODO(#267): TRIALITY tab HIDDEN per operator 2026-07-07 — redesign (rename, organic
       evolution, costate integration) before re-show. All endpoints + panels + snapshot
       machinery stay intact for both; restore = uncomment a line here:
  <div class="tab" data-tab="sandbox">SANDBOX</div>
  <div class="tab" data-tab="whyhow">WHY / HOW</div>
  <div class="tab" data-tab="tri">TRIALITY</div>
  -->
</div>

<section id="tab-sandbox" class="orc hide">
  <div class="orcintro">
    <h2>SANDBOX &mdash; why is a curvature polynomial secretly a topological invariant, and where the same music shows up here</h2>
    <p>A playground for the deep math. Prompted by a
    <a href="https://x.com/nihilunbounded" target="_blank" rel="noopener">post on Pontryagin classes</a>:
    <em>&ldquo;there&rsquo;s no apriori reason some random polynomial [of the curvature] should be a homotopy
    invariant &mdash; you&rsquo;re defining p_i(M) using the smooth structure on M!&rdquo;</em> Everything below is
    tagged <b>MEASURED / DERIVED / ANALOGY</b>. The Pontryagin resonance is a genuine structural rhyme
    plus an actual characteristic class (Maslov) in our geometry &mdash; <b>not</b> a claim that we
    computed Pontryagin classes or detected exotic spheres.</p>

    <h3>The Lie answer to &ldquo;why a polynomial&rdquo;</h3>
    <p>Chern&ndash;Weil theory. The invariant polynomials are exactly the <b>Ad-invariant polynomials on
    the Lie algebra 𝔤</b>, and the Weil homomorphism <code>Sym(𝔤*)^G &rarr; H*(BG)</code> sends each to a
    characteristic class. You are not evaluating a random polynomial &mdash; you are evaluating a
    <b>G-invariant</b>, so the metric/connection choices cancel by construction and the class is an
    invariant of the <em>bundle</em>. The miracle is a statement about a Lie group and its algebra.</p>

    <h3>The oracle we optimize against (auth-eval scorer)</h3>
    <p>The comma.ai challenge scores task-aware (&ldquo;coding-for-machines&rdquo;) compression of openpilot
    driving video through a <b>frozen oracle</b>: <b>SegNet</b> (comma10k EfficientNet-B2) per-pixel
    argmax &rarr; <b>d_seg</b>; <b>PoseNet</b> (FastViT-T12) two-frame YUV6 &rarr; <b>d_pose</b>; archive
    bytes &rarr; rate. <code>S = 100&middot;d_seg + &radic;(10&middot;d_pose) + 25&middot;bytes / 37,545,489</code>.
    Only the argmax partition, the pose 6-vector, and the byte count carry authority. (MEASURED)</p>

    <h3>The rhyme &mdash; parametrization-dependent-looking, secretly invariant</h3>
    <p>Our vehicle is a <b>task-space level-set witness</b>: a coordinate-INR that amortizes the SegNet
    argmax partition directly (the viscosity solution of a variational level-set flow; the object is the
    codim-1 <b>separatrix</b> between argmax cells). The score has his exact shape &mdash;
    <b>d_seg is a function on a quotient</b>, invariant under every reparametrization of the witness
    weights &theta; that leaves the argmax partition fixed:</p>
    <ul class="witkey">
      <li><code>d_seg : &#8477;&#8319; / (argmax-cell partition) &rarr; &#8477;</code> &mdash; it looks like
      it depends on the millions of INR weights that draw the field, but it depends only on the partition.
      Same phenomenon as the curvature polynomial being secretly topological. (structural / by-construction)</li>
    </ul>

    <h3>The Lie spine runs through both scored axes</h3>
    <ul class="witkey">
      <li><b>Pose is se(3).</b> By Chasles every rigid displacement is a screw &mdash; one twist
      <b>&xi; &isin; se(3)</b>, <code>exp(&xi;) &isin; SE(3)</code>. The same &xi; that transports the
      partition (a d_seg prior) is the pose PoseNet measures (d_pose). Engine <code>tac.lie.se3</code>.
      (BUILT + MEASURED: openpilot-ego prior gives &minus;94/&minus;99% on the pose axis)</li>
      <li><b>Conditioning on the Stiefel manifold.</b> The Muon finisher orthogonalizes gradients via
      Newton&ndash;Schulz &mdash; descent on the Stiefel manifold, Lie-group geometry.
      (MEASURED: &minus;32% d_seg vs AdamW)</li>
      <li><b>A Maslov class in the boundary geometry.</b> We read the argmax as a <b>caustic</b>
      (Lagrangian singularity) of the softmax-as-&#8463;&rarr;0 limit (&tau; = &epsilon; = &#8463;;
      error bound &le; &tau;&middot;ln&nbsp;5). The <b>Maslov class is a characteristic class of the
      Lagrangian Grassmannian &Lambda;(n) = U(n)/O(n)</b> &mdash; same family as Chern/Pontryagin, and just
      as Lie-theoretic. (DERIVED &mdash; theoretical framework)</li>
      <li><b>Tropical / Laguerre.</b> As &tau;&rarr;0 the witness is tropical; its cells are Laguerre
      (power-diagram) cells &mdash; forget the smooth structure, keep the piecewise-linear skeleton, the
      same flavor as extracting a topological invariant from smooth data. (DERIVED)</li>
    </ul>

    <h3>The honest boundary</h3>
    <p><b>Real:</b> score-as-quotient-invariance (by construction), the se(3)/Chasles pose engine
    (BUILT + MEASURED), Stiefel/Muon conditioning (MEASURED), and the Maslov/tropical framework
    (DERIVED theory). <b>Analogy, not identity:</b> we did not compute Pontryagin classes, do
    Chern&ndash;Weil on a tangent bundle, or detect exotic spheres. Our characteristic class is the
    Maslov class (Lagrangian); our invariance is score-invariance-under-reparametrization. Same music,
    different theorem.</p>
  </div>

  <div class="orccredits">
    <div class="wch">Links &mdash; codebase &amp; resources</div>
    <ul>
      <li><b>Repo</b> (contest closed &rarr; IP open source):
      <a href="https://github.com/adpena/comma-lab" target="_blank" rel="noopener">github.com/adpena/comma-lab</a>
      &mdash; in-tree: <code>src/tac/lie/se3.py</code> (se(3)/SE(3)), <code>src/tac/boundary_math/</code>
      (the witness), <code>src/tac/canonical_equations/deepmath_amortizing_argmax_laws_20260704.py</code>,
      <code>docs/sandbox_pontryagin_lie_deepmath_context.md</code> (the full context behind this tab).</li>
      <li><b>Prior art</b>: Milnor, <em>On manifolds homeomorphic to the 7-sphere</em> (1956);
      Chern&ndash;Weil theory / the Weil homomorphism; Arnold, the Maslov index; Cand&egrave;s&ndash;Donoho,
      curvelets (the optimal sparse basis for a curved codim-1 singularity); Dubois et al.,
      <em>Lossy Compression for Lossless Prediction</em> (NeurIPS 2021, the task-space sufficient-statistic
      codec);
      <a href="https://github.com/commaai/comma_video_compression_challenge" target="_blank" rel="noopener">comma.ai video compression challenge</a>.</li>
    </ul>
  </div>
</section>

<section id="tab-campaign" class="cmp hide">
  <div class="cmp-runline" id="cmp_runline">campaign: loading&hellip;</div>
  <div class="cmp-kv" id="cmp_status"></div>

  <div class="panel">
    <div class="ph">Exact n600 verdict trace &mdash; d_seg vs global_step
      <span class="cmp-tag exact">EXACT n600 &middot; [macOS-CPU frozen-scorer advisory]</span>
      <span class="cmp-tag">&#9679; ema &nbsp;&#9675; live (parameter_shadow)</span></div>
    <canvas id="cmp_vseg" height="220" aria-label="exact n600 d_seg verdict trace with stage targets"></canvas>
    <div class="footnote" id="cmp_vseg_foot"></div>
  </div>

  <div class="panel">
    <div class="ph">Exact n600 verdict trace &mdash; d_pose vs global_step
      <span class="cmp-tag exact">EXACT n600</span></div>
    <canvas id="cmp_vpose" height="170" aria-label="exact n600 d_pose verdict trace"></canvas>
  </div>

  <div class="panel">
    <div class="ph">Per-step descent strip &mdash; batch-local d_seg initial&rarr;final
      <span class="cmp-tag adv">ADVISORY_BATCH_LOCAL &mdash; the step&rsquo;s own 4-pair batch, never n600</span></div>
    <canvas id="cmp_steps" height="190" aria-label="per-step batch-local d_seg descent strip"></canvas>
    <div class="footnote">Each tick spans the step&rsquo;s initial&rarr;final batch-local d_seg
      (green = descended, red = rose); the line threads the finals. Do not compare levels
      against the n600 trace above &mdash; different pair sets per step.</div>
  </div>

  <div class="duo">
    <div class="panel">
      <div class="ph">gradient_norm per step <span class="cmp-tag adv">advisory</span></div>
      <canvas id="cmp_gnorm" height="150" aria-label="gradient norm per step"></canvas>
    </div>
    <div class="panel">
      <div class="ph">seconds / step vs sealed budget</div>
      <canvas id="cmp_secs" height="150" aria-label="seconds per step vs the sealed 312 s/step budget"></canvas>
      <div class="footnote" id="cmp_secs_foot"></div>
    </div>
  </div>

  <div class="duo">
    <div class="panel">
      <div class="ph">Pose-finish engage gate <span class="cmp-tag exact">from exact verdicts</span></div>
      <div class="gatebox" id="cmp_gate">&mdash;</div>
    </div>
    <div class="panel">
      <div class="ph">Per-class d_seg &mdash; latest exact verdict <span class="cmp-tag exact">EXACT n600</span></div>
      <div id="cmp_cls">&mdash;</div>
      <div class="footnote" id="cmp_cls_foot"></div>
    </div>
  </div>
</section>

<section id="tab-oracle" class="orc hide">
  <div class="orcintro">
    <h2>Frozen evaluators &mdash; SegNet argmax + PoseNet</h2>
    <div class="kvlist">
      <div class="kv"><span class="kvk">scorer</span><span class="kvv">comma10k EfficientNet-B2 SegNet argmax &rarr; <b>d_seg</b> &middot; FastViT-T12 PoseNet[:6] &rarr; <b>d_pose</b> &middot; frozen</span></div>
      <div class="kv"><span class="kvk">lane &rarr; d_seg</span><span class="kvv">openpilot lane fit &middot; analytic band at decode &middot; 0 stored bytes</span></div>
      <div class="kv"><span class="kvk">ego-&xi; &rarr; d_pose</span><span class="kvv">SE(3) screw / pair &middot; transports partition = pose</span></div>
      <div class="kv"><span class="kvk">detectability</span><span class="kvv">SegNet top1&minus;top2 margin &middot; where d_seg can flip</span></div>
      <div class="kv"><span class="kvk">classes</span><span class="kvv">detected from data, not hardcoded</span></div>
    </div>
    <div class="orchdr" id="orchdr">rendering the physical-prior atlas (governed CPU pass)&hellip;</div>
  </div>
  <div class="orcstats" id="orcstats"></div>
  <div class="orcgrid" id="orcpanels"></div>
  <div class="orcxi">
    <h3>Ego-&xi; &mdash; the SE(3) screw twist across the segment</h3>
    <div class="orccap">per-pair ego trajectory &middot; <code>LaneOptimalEgoEstimator</code> &middot; Chasles: one 6-vector/pair transports the partition (d_seg) AND is the pose (d_pose)</div>
    <img id="orcxichart" alt="ego-ξ screw twist across the segment" />
  </div>
  <div class="orccredits">
    <div class="wch">Priors</div>
    <ul>
      <li><b>openpilot lane band</b> (<code>analytic_lane_render_band.build_analytic_lane_band_prior</code>)
      &mdash; deg-3 centerline fit to the GT class-1 argmax + AA-SDF range-dependent dash coverage;
      the FREE inflate-time rasterizer (rule&nbsp;118). The <b>d_seg</b> physical prior.</li>
      <li><b>ego-&xi; screw</b> (<code>ego_xi_trajectory.LaneOptimalEgoEstimator</code>, se(3) engine
      <code>tac.lie</code>) &mdash; the SE(3) twist that is BOTH the partition-warp and the pose.
      The <b>d_pose</b> physical prior, dual-use with d_seg.</li>
      <li><b>ground-plane structure</b> (<code>road_horizon_component</code> +
      <code>hood_static_component</code>) &mdash; the self-detected road/sky horizon + static ego hood
      (the #139 static core).</li>
      <li><b>detectability field &rho;_seg</b> &mdash; the SegNet top1&minus;top2 argmax margin; bright
      where the argmax can flip = where d_seg lives = where the detector is most sensitive.</li>
    </ul>
    <div class="wcnote">[macOS-CPU advisory &middot; NON-PROMOTABLE] &mdash; a viz moves no pointer.
    The exact row is byte-closed on contest-CPU/CUDA; the frontier pointer is <span class="ptrv">&hellip;</span> and UNMOVED.</div>
  </div>
</section>

<section id="tab-live">
  <!-- 2026-07-09 UX: current numbers FIRST (summary before detail) so the live descent is seen
       at a glance on the phone, not scrolled past empty warming-up charts. JS binds by id, so this
       reorder is purely visual. -->
  <div class="lv-runline" id="rdinfo">resolving run&hellip;</div>

  <!-- 0 · CHAIN-STATE STRIP (c2 era) — bench &rarr; receipt &rarr; launch &rarr; run &rarr; byte-close.
       Keyed on the NEWEST launch-provenance dir, so a pre-launch bench (dry-start in flight,
       no run.log yet) is visible BEFORE the run fires. CONDITIONAL: hidden with no launch dir. -->
  <div class="lv-chain hide" id="lv_chain" aria-label="launch pipeline position"></div>

  <!-- 0b · CONFOUND ALARMS — red strip, CONDITIONAL: rendered only when confound_alarm /
       term_inert rows exist in the run's telemetry (they are LOUD by design; L1 immune layer). -->
  <div class="lv-alarms hide" id="lv_alarms" aria-label="confound alarms"></div>

  <!-- 1 · SCORE DECOMPOSITION MASTHEAD — the live equation, each term's value + its
       contribution to S, dominant term visible at a glance. -->
  <div class="lv-mast">
    <div class="lv-mast-s">
      <div class="lv-k">implied&nbsp;S <span class="lv-adv">advisory</span></div>
      <div class="lv-sval" id="lv_S">&mdash;</div>
      <div class="lv-sref" id="lv_Sref">&nbsp;</div>
    </div>
    <div class="lv-mast-eq">
      <div class="lv-eq" id="lv_eq">S = 100&middot;d_seg + &radic;(10&middot;d_pose) + 25&middot;bytes / <span id="lv_norm">&hellip;</span></div>
      <div class="lv-terms" id="lv_terms"></div>
    </div>
  </div>

  <!-- 2 · THE DESCENT — d_seg over epochs, auto-fit x, log y, curriculum bands -->
  <div class="lv-panel lv-chartpanel">
    <div class="lv-ph"><span class="lv-pt">d_seg descent</span><span class="lv-pm" id="lv_dseg_meta">&mdash;</span></div>
    <canvas id="c_dseg" role="img" aria-label="d_seg descent chart"></canvas>
    <div class="lv-legend" id="slegend">
      <span class="sc" data-st="ce"><span class="dot" style="background:#5ab0ff"></span>CE</span>
      <span class="sc" data-st="tau"><span class="dot" style="background:#b08cff"></span>tau</span>
      <span class="sc" data-st="l7"><span class="dot" style="background:#ffa454"></span>l7</span>
      <span class="sc off" data-st="muon"><span class="dot" style="background:#46d3a0"></span>Muon</span>
      <span class="sc lv-legsp"><span class="dot" style="background:rgba(226,232,240,.45)"></span>EMA</span>
      <span class="sc"><span class="dot dsh"></span>trend</span>
      <span class="sc"><span class="dot" style="background:#ffd24a;border-radius:50%"></span>best</span>
    </div>
  </div>

  <!-- 3 + 4 · per-class breakdown | pose-descent readiness -->
  <div class="lv-row2">
    <div class="lv-panel">
      <div class="lv-ph"><span class="lv-pt">per-class d_seg &middot; flip share</span><span class="lv-pm">comma10k order</span></div>
      <div class="lv-classes" id="lv_classes"><div class="lv-none">no verdict yet</div></div>
    </div>
    <div class="lv-panel">
      <div class="lv-ph"><span class="lv-pt">pose-descent readiness</span><span class="lv-pm" id="lv_pose_meta">jacobian basin</span></div>
      <div class="lv-pose" id="lv_pose"><div class="lv-none">no basin probe yet</div></div>
    </div>
  </div>

  <!-- 5 + 6 · training health | system memory -->
  <div class="lv-row2">
    <div class="lv-panel">
      <div class="lv-ph"><span class="lv-pt">training health</span><span class="lv-pm" id="lv_health_meta">loss terms</span></div>
      <div class="lv-health" id="lv_health"><div class="lv-none">no loss row yet</div></div>
    </div>
    <div class="lv-panel">
      <div class="lv-ph"><span class="lv-pt">system</span><span class="lv-pm">resident &middot; MLX</span></div>
      <div class="lv-sys" id="lv_sys"><div class="lv-none">no memory row yet</div></div>
    </div>
  </div>

  <!-- 7 · SCHEDULE POSITION — curriculum timeline, marker at current epoch -->
  <div class="lv-panel">
    <div class="lv-ph"><span class="lv-pt">curriculum position</span><span class="lv-pm" id="lv_sched_meta">&mdash;</span></div>
    <div class="lv-sched" id="lv_sched"><div class="lv-none">resolving schedule</div></div>
  </div>

  <div class="lv-status" id="status">connecting&hellip;</div>
  <div class="lv-detail" id="detail">&nbsp;</div>
  <div class="proj" id="proj"><div id="proj_seg">&nbsp;</div><div class="proj2" id="proj_s">&nbsp;</div></div>
  <!-- costate controller SENSE/DECIDE panel — CONDITIONAL: rendered only when the run's
       costate_shadow.jsonl exists (score-neutral shadow observer). Observability ONLY;
       the dashboard NEVER actuates (CONTAINMENT). -->
  <div class="costate hide" id="costate">
    <div class="csh">costate controller &middot; SENSE/DECIDE <span class="cstag">shadow observer &middot; read-only &middot; advisory</span></div>
    <div class="csbody" id="cs_body">&nbsp;</div>
  </div>
  <details class="cfg" id="cfgpanel" open>
    <summary id="cfgsum">setup &middot; config &middot; schedule &middot; curriculum</summary>
    <div class="cfgbody" id="cfgbody"><div class="cfgmeta">parsing run config&hellip;</div></div>
  </details>
  <!-- #352 schema-driven telemetry panel — CONDITIONAL sections: planned curves,
       LawRef constants manifest, mem_probe, fired events. Each section renders only
       when its source artifact is present (introspect_run); a pre-v6 run dir simply
       shows fewer sections. Observability ONLY. -->
  <details class="cfg hide" id="telemetry" open>
    <summary id="telsum">telemetry &middot; curves &middot; constants &middot; memory</summary>
    <div class="cfgbody" id="telbody"><div class="cfgmeta">loading telemetry&hellip;</div></div>
  </details>
  <div class="rinfo" id="runinfostrip"></div>
  <div class="foot" id="foot"></div>
</section>

<section id="tab-witness" class="wit hide">
  <div class="witintro">
    <h2>The residual &mdash; the hardest pairs, as the scorer reads them</h2>
    <div class="kvlist">
      <div class="kv"><span class="kvk">selection</span><span class="kvv">highest realized d_seg across n600 &middot; distinct failure modes &middot; same pass as FLOW &middot; refreshed per best ckpt</span></div>
      <div class="kv"><span class="kvk">per-pair</span><span class="kvv">d_seg + failure tag (movable / lane dash / distant / boundary)</span></div>
      <div class="kv"><span class="kvk">row A</span><span class="kvv">GT frame &middot; render through contest R &middot; pixel error</span></div>
      <div class="kv"><span class="kvk">row B</span><span class="kvv">SegNet argmax GT vs render &middot; disagreement = the d_seg pixels</span></div>
      <div class="kv"><span class="kvk">row C</span><span class="kvv">sensitivity fields on the same frame (below)</span></div>
    </div>
    <div class="withdr" id="withdr">selecting the hardest pairs from the n600 pass&hellip;</div>
  </div>
  <div class="witgrid" id="witpanels"></div>
  <div class="wittrip">
    <h3>Row C &mdash; two sensitivity fields on the same frame</h3>
    <div class="kvlist">
      <div class="kv"><span class="kvk">&rho;_seg</span><span class="kvv">SegNet top1&minus;top2 margin &middot; <b>bright = small margin = where argmax flips</b> &middot; d_seg lives only here</span></div>
      <div class="kv"><span class="kvk">&rho;_uniward</span><span class="kvv">S-UNIWARD texture energy (Holub&ndash;Fridrich&ndash;Denemark 2014, <code>tac.uniward_delta</code>) &middot; lineage only &mdash; measured at chance vs R-flips (LEVER-4)</span></div>
    </div>
    <div class="witcap">task-aware compression: code for the machine that scores it. d_seg moves only on the codim-1 argmax boundary; the margin field localizes it AND is the scorer's information geometry (Fisher vs &minus;margin: <b>Pearson 0.978, measured</b>) &mdash; one field drives loss weighting + residual coder.</div>
  </div>
  <!-- TODO(#343): Credits/lineage section HIDDEN per operator 2026-07-07; the tribute-register
       header phrase is deleted outright. Rework as a plain References list (paper citations,
       direct register) before re-showing, with the HONEST ATTRIBUTION hierarchy (operator
       2026-07-07): task-aware compression / VCM at the heart; inverse-steganalysis as the
       intellectual on-ramp only (no over-attribution); our task-space level-set
       separatrix/Morse-Smale math as the original contribution.
  <div class="witcredits">
    <div class="wch">References</div>
    <ul>
      <li><b>comma10k-baseline</b> (Yassine Yousfi) &mdash; the segmentation baseline whose
      EfficientNet-B2 U-Net is the contest's frozen SegNet; the argmax it produces IS the d_seg
      authority this page measures against.</li>
      <li><b>S-UNIWARD</b> (Vojt&#283;ch Holub, Jessica Fridrich, Tom&aacute;&#353; Denemark, 2014,
      Binghamton DDE Lab) &mdash; the universal steganographic distortion; the theory that textured
      regions hide a perturbation and smooth ones expose it.</li>
      <li>The framing that this contest <b>IS inverse steganalysis</b>: the scorer is the steganalyst;
      the shortest archive whose witness survives the detector wins.</li>
    </ul>
  </div>
  -->
  <div class="wcnote">[macOS-CPU advisory &middot; NON-PROMOTABLE] &mdash; a viz moves no pointer.
  The exact row is byte-closed on contest-CPU/CUDA; the frontier pointer is <span class="ptrv">&hellip;</span> and UNMOVED.</div>
</section>

<section id="tab-flow" class="flow hide">
  <h2>FLOW &mdash; the full n600 drive, as a video (WebGPU)</h2>
  <div class="flowcapline">n600 segment &middot; witness render / 5-class partition / SegNet argmax / disagreement-vs-GT (d_seg) &middot; frame slider = timeline 0&ndash;599, play ~12 fps</div>
  <div class="flowtop">
    <span id="flowbadge" class="flowbadge off">detecting&hellip;</span>
    <span class="flowstatus" id="flowstatus">waiting for the first n600 sequence&hellip;</span>
  </div>
  <div class="flowstage">
    <canvas id="flowcanvas" role="img" aria-label="n600 witness drive video — partition / SegNet / disagreement"></canvas>
    <div class="flowmsg" id="flowmsg">
      <div class="flowload">
        <div class="flowspin" id="flowspin" aria-hidden="true"></div>
        <div class="flowmsgtxt" id="flowmsgtxt">the first n600 video renders on the next best checkpoint (~14&nbsp;min governed pass)&hellip;</div>
        <div class="flowprog hide" id="flowprog" role="progressbar" aria-label="render progress"><div class="flowprogfill" id="flowprogfill"></div></div>
      </div>
    </div>
  </div>
  <div class="flowscrub">
    <button class="flowplay" id="flowplay" aria-pressed="false">&#9654; play</button>
    <div class="sc">
      <span class="fl">frame (segment timeline) <span class="fv" id="flowframe_v">0 / 599</span></span>
      <input type="range" id="flowframe" min="0" max="599" step="1" value="0" aria-label="frame / segment timeline">
    </div>
  </div>
  <div class="flowlegend" id="flowlegend"></div>
  <div class="flowctl">
    <div class="flowrow">
      <span class="fl">layer <span class="fv" id="flowmode_v">SegNet argmax</span></span>
      <select id="flowmode">
        <option value="0">witness render (RGB)</option>
        <option value="1">witness partition (own argmax)</option>
        <option value="2" selected>SegNet argmax (what the scorer sees)</option>
        <option value="3">disagreement vs GT (d_seg pixels)</option>
        <option value="4">margin heat (fragility)</option>
      </select>
    </div>
    <div class="flowrow">
      <span class="fl">class isolation <span class="fv" id="flowiso_v">all classes</span></span>
      <select id="flowiso">
        <option value="-1">all classes</option>
        <option value="0">0 Road</option>
        <option value="1">1 Lane</option>
        <option value="2">2 Undrivable</option>
        <option value="3">3 Movable</option>
        <option value="4">4 MyCar / hood</option>
      </select>
    </div>
    <div class="flowrow">
      <span class="fl">margin threshold (fragile band) <span class="fv" id="flowthr_v">0.55</span></span>
      <input type="range" id="flowthr" min="0" max="1" step="0.01" value="0.55">
    </div>
  </div>
  <div class="flowcap">
    <div class="kvlist">
      <div class="kv"><span class="kvk">partition</span><span class="kvv">canonical comma10k / openpilot palette</span></div>
      <div class="kv"><span class="kvk">margin heat</span><span class="kvv"><b>bright = small margin = codim-1 separatrix</b> &middot; where the argmax flips (d_seg)</span></div>
    </div>
    <p class="fcnote">[macOS-CPU advisory &middot; NON-PROMOTABLE] &mdash; a viz moves no pointer &middot;
    exact row byte-closed on contest-CPU/CUDA &middot; pointer <span class="ptrv">&hellip;</span> UNMOVED.</p>
  </div>
</section>

<section id="tab-whyhow" class="why hide">
  <div class="whyhero">
    <p class="idea">The task is <b>boundary geometry</b>; the witness is the <b>chart that fits it</b>; and the
    same traveling front governs it at every scale &mdash; from the pixel boundary to the campaign.</p>
    <p class="sub">Two movements. <b>WHY</b> &mdash; the static invariant that makes the chart optimal.
    <b>HOW</b> &mdash; the dynamics that flow to it. Pass&nbsp;1 opens the museum with the two highest-ROI
    plates (the live field &amp; the Unity); the five-scale spine, the screw, and the finale are seamed
    below. Every plate is a MEASURED fact &mdash; <b>[macOS-CPU advisory &middot; NON-PROMOTABLE]</b>,
    a viz moves no pointer (<span class="ptrv">&hellip;</span>, UNMOVED).</p>
  </div>

  <div class="mvrail" id="whymvrail">
    <span class="mvchip on why-i" data-mv="i">Movement I &middot; WHY</span>
    <span class="mvchip why-ii" data-mv="ii">Movement II &middot; HOW</span>
  </div>

  <!-- ================= MOVEMENT I — WHY (the static invariant) ================= -->
  <div id="whymv-i">
    <div class="mvhead i">Movement I &mdash; WHY
      <span class="mvsub">the optimal chart: why this vehicle can't be beaten on this geometry</span></div>

    <!-- I.1 — the live field -->
    <div class="plate accent-i" id="plate-i1">
      <div class="whytop">
        <span class="pnum">I.1</span><span class="ptitle">The field, alive</span>
        <span id="whybadge_field" class="whybadge off">detecting&hellip;</span>
        <span class="whystatus" id="whystatus">loading the deep-math field bundle&hellip;</span>
      </div>
      <div class="peq">&phi;(x) &nbsp;&middot;&nbsp; level set {&phi; = t} &nbsp;&rarr;&nbsp; the argmax separatrix</div>
      <div class="whystage">
        <canvas id="whycanvas_field" role="img" aria-label="the witness detectability field, level sets sweeping the separatrix"></canvas>
        <div class="whymsg" id="whymsg_field">the field renders on the first governed pass&hellip;</div>
      </div>
      <div class="whyctl">
        <div class="whyrow">
          <span class="rl">level-set threshold t <span class="rv" id="whythr_v">0.55</span></span>
          <input type="range" id="whythr" min="0" max="1" step="0.01" value="0.55" aria-label="level-set threshold">
        </div>
        <div class="whyrow">
          <span class="rl">overlays</span>
          <span class="whytoggle on" id="whytog_zero" role="button" tabindex="0">zero-level-set (argmax boundary)</span>
          <span class="whytoggle" id="whytog_grad" role="button" tabindex="0">&nabla;&phi; gradient / normals</span>
        </div>
        <div class="whyrow">
          <span class="rl">base layer</span>
          <div class="whyseg" id="whyfieldbase">
            <span class="sg on" data-b="0">&phi; heat (&rho;_seg margin)</span>
            <span class="sg" data-b="1">scene render</span>
            <span class="sg" data-b="2">comma10k partition</span>
          </div>
        </div>
      </div>
      <div class="whyleg" id="whyleg_field"></div>
      <p class="pcap">The <b>SegNet detectability field</b> &phi; over the scored frame: bright where the argmax
      is fragile (small top1&minus;top2 margin) &mdash; the codim-1 <b>separatrix</b>, a ~1-pixel curve that
      carries essentially <b>all</b> of d_seg. Drag the threshold to sweep the level sets like contour lines;
      toggle the <b>zero-level-set</b> (the argmax partition boundary) and the <b>gradient field</b> &nabla;&phi;
      (boundary normals &mdash; the derivative you can see).</p>
      <p class="pcite"><b>Grounded:</b> the real cached SegNet argmax + top1&minus;top2 margin (gt_n6, the exact
      frames the verdict uses); flip-mass measured ~50% Road / 19% Lane / 13% Undrivable (#141). The live
      per-checkpoint witness INR &phi; is the Pass-2 upgrade (reuses the governed 600-pass FLOW cache).</p>
    </div>

    <!-- I.4 — the Unity -->
    <div class="plate accent-i" id="plate-i4">
      <div class="whytop">
        <span class="pnum">I.4</span><span class="ptitle">The Unity &mdash; one geometry, three readings</span>
        <span id="whybadge_unity" class="whybadge off">detecting&hellip;</span>
      </div>
      <div class="peq">&rho;(x) &nbsp;&asymp;&nbsp; 1 / &Vert;&part;(detector)/&part;(pixel)&Vert; &nbsp;&mdash;&nbsp; the Fisher metric, read three ways</div>
      <div class="whystage">
        <canvas id="whycanvas_unity" role="img" aria-label="one field, three readings: SegNet margin, S-UNIWARD cost, our distortion sensitivity"></canvas>
        <div class="whymsg" id="whymsg_unity">the unity morph renders on the first governed pass&hellip;</div>
      </div>
      <div class="whyctl">
        <div class="whyrow">
          <span class="rl">morph <span class="rv" id="whymorph_v">&rho;_seg</span></span>
          <input type="range" id="whymorph" min="0" max="2" step="0.01" value="0" aria-label="morph between the three readings">
        </div>
        <div class="whyrow">
          <span class="whyseg" id="whyunityjump">
            <span class="sg on" data-t="0">&rho;_seg &middot; SegNet margin (Yousfi)</span>
            <span class="sg" data-t="1">our distortion sensitivity</span>
            <span class="sg" data-t="2">&rho;_uniward &middot; S-UNIWARD (Fridrich)</span>
          </span>
        </div>
      </div>
      <div class="whycorr" id="whycorr"></div>
      <p class="pcap"><b>The tribute's heart.</b> One scene, three sensitivity fields. Drag the morph:
      <b>&rho;_seg</b> (the detector's own margin) and <b>our distortion sensitivity</b> (the separatrix geometry
      &mdash; where a pixel flip changes the decision) dissolve into <b>the same picture</b> &mdash; measured
      Pearson <span id="whycorr_hi">&mdash;</span> on this frame. Fridrich's <b>&rho;_uniward</b> (S-UNIWARD
      embedding cost) is the kindred steganographic reading: it lights image texture, so pixelwise it is
      <b>honestly weaker</b> here (<span id="whycorr_lo">&mdash;</span>) &mdash; the deep tie between detector and
      steganography runs through the <b>Fisher metric</b>, where the margin field IS the Fisher surrogate
      (canonical <b>0.978</b>, Fisher&nbsp;curvature&nbsp;&harr;&nbsp;&minus;margin).</p>
      <p class="pcite"><b>Grounded:</b> real S-UNIWARD via <code>tac.uniward_delta.compute_uniward_cost_map</code>
      (Holub&ndash;Fridrich&ndash;Denemark 2014); real cached SegNet margin; live per-frame Pearson computed on
      the served fields (NO fabricated curve); canonical 0.978 = memory <code>unified-variational-levelset-flow</code>.
      Steganography (Fridrich) &rarr; steganalysis (Yousfi) &rarr; our loss &mdash; the arc is Yousfi's detection game.</p>
    </div>

    <div class="whyseam">
      <p class="st">seamed for Pass 2+ &middot; Movement I</p>
      <ul>
        <li><b>I.2 the separatrix</b> &mdash; dim the flat interior, light the codim-1 annulus (margin-saliency #141).</li>
        <li><b>I.3 curvature &harr; the chart</b> &mdash; the anisotropic/curvelet basis vs the isotropic-Fourier Gibbs ring (&minus;48% directional basis).</li>
        <li><b>I.5 the ~8-dim manifold</b> &mdash; the lane-orbit surface + the bc20 under-capacity tear.</li>
        <li><b>I.6 the task-sufficient statistic</b> &mdash; the RGB collapses to the decision (S_floor&asymp;0.118).</li>
      </ul>
    </div>
  </div>

  <!-- ================= MOVEMENT II — HOW (the dynamics) ================= -->
  <div id="whymv-ii" class="hide">
    <div class="mvhead ii">Movement II &mdash; HOW
      <span class="mvsub">the flow to the chart: how it gets there, and how it all falls out</span></div>
    <div class="whyseam">
      <p class="st">seamed for Pass 2+ &middot; Movement II</p>
      <ul>
        <li><b>II.1 the level-set flowing</b> &mdash; the Fisher&ndash;KPP front, softmax sharpening as &tau; anneals (live checkpoint sequence).</li>
        <li><b>II.2 curriculum = one axis, four names</b> &mdash; curvelet-scale &middot; CE&rarr;&tau;&rarr;Muon &middot; temperature &middot; persistence, one playhead.</li>
        <li><b>II.3 Morse&ndash;Smale &amp; saddles</b> &mdash; critical points + separatrices; erasure &prop; 1/persistence.</li>
        <li><b>II.4 the screw that falls out</b> &mdash; drag the se(3) twist &xi;; the SAME &xi; warps d_seg AND is d_pose (Chasles, #193).</li>
        <li><b>II.5 critical slowing</b> &mdash; relaxation time diverging at a stage transition (a second-order phase transition).</li>
      </ul>
    </div>
  </div>

  <!-- ================= §1 — ONE FRONT, FIVE SCALES (the spine hero · Pass 2) ================= -->
  <div class="mvhead spine">The spine &mdash; ONE FRONT, FIVE SCALES
    <span class="mvsub">the same traveling-wave front at five scales of the campaign &mdash; drag the scale</span></div>

  <div class="plate accent-spine" id="plate-spine">
    <div class="whytop">
      <span class="pnum">&sect;1</span><span class="ptitle">One front, five scales</span>
      <span id="whybadge_spine" class="whybadge off">detecting&hellip;</span>
      <span class="whystatus" id="whyspine_status">the live Fisher&ndash;KPP front renders on activate&hellip;</span>
    </div>
    <div class="peq">&part;<sub>t</sub>&thinsp;x = &beta;&thinsp;x&thinsp;(1&minus;x) + &nabla;&sup2;x
      &nbsp;&middot;&nbsp; <span id="whyspine_units">units: correct-fraction / epoch</span></div>

    <div class="whylens">
      <span class="lenstag">Lens &middot; not a proven identity</span>
      Two of these five curves are <b>hard data</b>: EdgeBench&rsquo;s log-sigmoid (<b>R&sup2;=0.998</b>, ByteDance Seed
      &mdash; <i>theirs</i>) and <b>#205&rsquo;s live d_seg descent</b> (<i>ours</i>). That all five scales are literally
      the SAME Fisher&ndash;KPP front is a <b>unifying interpretation we find beautiful and testable &mdash; not</b> a
      measured cross-scale identity. The integrator at left integrates the <b>real</b> PDE; that our five scales <i>are</i>
      that PDE is the conjecture.
    </div>

    <div class="spinegrid">
      <div class="spinehalf">
        <div class="spsub">live Fisher&ndash;KPP front &mdash; integrated forward every frame</div>
        <div class="whystage"><canvas id="whycanvas_kpp" role="img" aria-label="live Fisher-KPP traveling-wave front, integrated forward"></canvas>
          <div class="whymsg" id="whymsg_kpp">the front integrates on activate&hellip;</div></div>
        <div class="spannote">&beta;&thinsp;x(1&minus;x) grows the correct phase; &nabla;&sup2;x diffuses it &rarr; a
          front travels at speed&nbsp;2&radic;&beta; with a self-similar log-sigmoid profile. <b>Explicit
          finite-difference, stepped live</b> (render: WebGPU, else canvas2d).</div>
      </div>
      <div class="spinehalf">
        <div class="spsub" id="whyscale_title">TRAINING &mdash; the correct partition invading (LIVE #205)</div>
        <div class="whystage light"><canvas id="whycanvas_scale" role="img" aria-label="the selected scale's own curve against the shared traveling-wave shape"></canvas></div>
        <div class="spannote">Dashed = the shared logistic front <code>x(u)=1/(1+e^{&minus;k(u&minus;u&#8320;)})</code>;
          solid = the selected scale&rsquo;s own curve. Drag the scale to morph between them.</div>
      </div>
    </div>

    <div class="whyctl">
      <div class="whyrow">
        <span class="rl">scale <span class="rv" id="whyscale_v">pixel &rarr; campaign</span></span>
        <input type="range" id="whyscale" min="0" max="4" step="0.001" value="2" aria-label="morph across the five scales, pixel to campaign">
      </div>
      <div class="whyrow">
        <span class="whyseg" id="whyscalejump">
          <span class="sg" data-s="0">boundary</span>
          <span class="sg" data-s="1">erasure</span>
          <span class="sg on" data-s="2">training</span>
          <span class="sg" data-s="3">curriculum</span>
          <span class="sg" data-s="4">campaign</span>
        </span>
        <span class="whytoggle on" id="whytog_play" role="button" tabindex="0">&#9614;&#9614; pause front</span>
      </div>
    </div>

    <div class="spinelabel" id="whyscale_honesty">
      <span class="stag live" id="whyscale_tag">live data</span>
      <span class="snote" id="whyscale_note">&mdash;</span>
    </div>

    <p class="pcap">One equation, five readings. The <b>live front</b> at left is the Fisher&ndash;KPP traveling wave
      integrating forward &mdash; &beta;&thinsp;x(1&minus;x) (reaction) + &nabla;&sup2;x (diffusion). At right, the
      <b>selected scale&rsquo;s own curve</b> against the same logistic-front template: the <b>campaign</b> (EdgeBench,
      R&sup2;=0.998), our <b>training</b> descent (live&nbsp;#205), the <b>boundary</b> separatrix (a genuine level-set =
      reaction&ndash;diffusion identity), the <b>curriculum</b> anneal, and the <b>erasure</b> long tail. Slide from
      pixel to campaign and watch the same shape recur.</p>
    <p class="pcite"><b>Grounded:</b> EdgeBench R&sup2;=0.998 (ByteDance Seed, 2026-07-02 &mdash; <i>their</i> published
      fit; the top curve is Claude Opus&nbsp;4.8). #205 verdicts <b>ep25&rarr;125</b>: d_seg 0.0103&rarr;0.0058, implied_S
      1.72&rarr;0.87 (live, read from the run log; single run &mdash; a descriptive fit, not a law). Boundary = level-set
      flow PDE identity; the curriculum &amp; erasure shapes are <b>schematic/interpretive</b>, honestly labelled. The
      unifying cross-scale identity is a <b>conjecture</b> per our own discipline &mdash; NO fabricated curve.</p>
  </div>

  <!-- ================= §4 — THE FRACTAL FINALE (Pass 2) ================= -->
  <div class="plate accent-fin" id="plate-fin">
    <div class="whytop">
      <span class="pnum">&sect;4</span><span class="ptitle">The fractal finale &mdash; all five, phase-locked</span>
    </div>
    <p class="pcap">Now that you have seen each front alone, here they are together &mdash; five scales, one playhead,
      the same wave rolling through all of them at once. Beneath: the one equation.</p>
    <div class="fingrid" id="whyfingrid">
      <div class="finpanel"><div class="finlab">boundary</div><canvas id="whyfin0" aria-label="boundary front, phase-locked"></canvas></div>
      <div class="finpanel"><div class="finlab">erasure</div><canvas id="whyfin1" aria-label="erasure tail front, phase-locked"></canvas></div>
      <div class="finpanel"><div class="finlab">training &middot; live</div><canvas id="whyfin2" aria-label="training front, phase-locked"></canvas></div>
      <div class="finpanel"><div class="finlab">curriculum</div><canvas id="whyfin3" aria-label="curriculum front, phase-locked"></canvas></div>
      <div class="finpanel"><div class="finlab">campaign</div><canvas id="whyfin4" aria-label="campaign front, phase-locked"></canvas></div>
    </div>
    <div class="fineq">&part;<sub>t</sub>&thinsp;x = &beta;&thinsp;x&thinsp;(1&minus;x) + &nabla;&sup2;x</div>
    <div class="finrow">
      <span class="whytoggle on" id="whytog_finplay" role="button" tabindex="0">&#9614;&#9614; pause</span>
      <span class="whytoggle" id="whytog_zoom" role="button" tabindex="0">zoom out once more &rarr;</span>
    </div>

    <div class="finsix" id="whyfinsix">
      <div class="fsx">
        <canvas id="whyfin_six" aria-label="the campaign front with the current session marked as a point on it"></canvas>
        <div class="fsxt">
          <b>The sixth, implied panel.</b> Zoom out once more and the viewer is themselves a front on the campaign
          graph. EdgeBench&rsquo;s <b>top curve is Claude Opus&nbsp;4.8</b> &mdash; this session &mdash; and the
          operator&rsquo;s steering is the feedback term &eta; that keeps the reaction from stalling. The physics of the
          <b>witness</b> (the boundary the chart paints) and the epistemics of the <b>campaign</b> (the program that
          painted it) are the <b>same equation</b> &mdash; two nested one-objects, Fisher&ndash;KPP all the way up and down.
          <div class="finsig">That is the sentence the museum opened with, now earned.</div>
        </div>
      </div>
    </div>
    <p class="pcite"><b>Honest:</b> the finale is the interpretive / aesthetic capstone &mdash; the five little fronts
      are the same curves shown above (two hard-data, three schematic), phase-locked for the eye. It asserts no new
      measurement; the pointer is <b><span class="ptrv">&hellip;</span>, UNMOVED</b> &mdash; a museum moves no pointer.</p>
  </div>

  <!-- About plate seamed for Pass 5 -->
  <div class="whyseam">
    <p class="st">the About plate &middot; Pass 5</p>
    <ul>
      <li><b>About</b> &mdash; the ideas &amp; the people (design &sect;7 / <code>dashboard_tribute_credits</code>):
      Aaron Leslie, Quantizr, Yousfi &amp; Fridrich, comma / Hotz, the council &mdash; Chasles to yesterday.
      <span class="whyabout">The past few months, given bloom.</span></li>
    </ul>
  </div>

  <p class="wcnote2">[macOS-CPU advisory &middot; NON-PROMOTABLE] &mdash; a viz moves no pointer. Every field
  is a real cached computation (SegNet argmax/margin + real S-UNIWARD); the exact row is byte-closed on
  contest-CPU/CUDA; the frontier pointer is <span class="ptrv">&hellip;</span> and UNMOVED.</p>
</section>

<section id="tab-tri" class="tri hide">
  <h2>One coherent object &mdash; the DAG &harr; DSL &harr; equations triality</h2>
  <p>The lab is one object viewed three ways &mdash; and this tab reads them <b>live</b> from the
  artifacts themselves (DAG FEEDs, the <code>tac.witness_dsl</code> program, the
  <code>tac.canonical_equations</code> registry), so it self-updates as the campaign moves. A finding is
  "known" only when it is expressible in all three and they AGREE; drift between legs is campaign-level
  forgetting.</p>
  <div class="tribuilt" id="tri_built">loading the three legs from the live artifacts&hellip;</div>
  <div class="cards">
    <div class="card"><h3>DAG &mdash; state <span class="cardsub">the trajectory</span></h3>
    <div class="trileg" id="tri_dag">&mdash;</div></div>
    <div class="card"><h3>DSL &mdash; control <span class="cardsub">the program</span></h3>
    <div class="trileg" id="tri_dsl">&mdash;</div></div>
    <div class="card"><h3>equations &mdash; law <span class="cardsub">the master action</span></h3>
    <div class="trileg" id="tri_eq">&mdash;</div></div>
  </div>
  <p class="m">Master action <code>S_&tau; = 100&middot;d_seg + &radic;(10&middot;d_pose) + 25&middot;rate</code>.
  Above all three sits the <b>costate controller</b> &mdash; the 4th shadow (marginal-&Delta;S per byte,
  never-regress) that turns the DSL from a passive program into an active controller (fire the lever with
  the best &Delta;S/cost; POWERPLAY-style never-worsen).</p>

  <h2>The organic-evolution lineage</h2>
  <p class="m">task-aware compression (video-coding-for-machines) &rarr; <b>task-sufficient statistic</b>
  (code the scorer's DECISION, not RGB; measured task-RD floor S_floor&asymp;0.118) &rarr;
  <b>compression-as-intelligence</b> (Schmidhuber POWERPLAY: never-regress self-invented curriculum) &rarr;
  the <b>costate controller</b> (marginal-&Delta;S/cost, never-regress) &mdash; one line, each stage the
  natural generalization of the last.</p>
  <p class="m">A live example of the campaign <b>compounding on the current frontier</b>:
  <b>MD-Decoupling</b> (H&auml;gele, Hern&aacute;ndez-Cano, Kosson, Jaggi &mdash; EPFL / Jaggi lab,
  <code>arXiv:2606.25971</code>, 2026-06-24) is our <code>--optimizer md</code> (#175) &mdash; it factorizes
  each weight matrix into a fixed-norm <b>direction</b> + learnable <b>magnitude</b> gains at separate
  learning rates, which makes <b>curriculum stage-transitions stable by construction</b> (the
  &ldquo;different stages need different treatment&rdquo; rule, made structural). Super-recent
  (June&nbsp;2026), folded straight into the live curriculum.</p>

  <h2>The composition &mdash; Chasles to yesterday</h2>
  <p class="m">The witness is not one idea; it is a <b>composition of nearly two centuries of research</b>,
  each thread <i>measured</i> into its place &mdash; ~196&nbsp;years of shoulders to stand on. All dates are
  real (NO-FAKE; a genuine lineage, not decoration).</p>
  <ol class="tritl">
    <li><span class="tly">1830</span><span class="tlt"><b>Chasles</b> &mdash; every rigid motion is a
      <b>screw</b> &rarr; our ego-motion twist &xi; (d_pose).</span></li>
    <li><span class="tly">1848&thinsp;&middot;&thinsp;99</span><span class="tlt"><b>Wilbraham / Gibbs</b>
      &mdash; the ringing of a truncated series &rarr; the spectral-bias / Gibbs failure mode we fight
      (step-native, curvelet-finest).</span></li>
    <li><span class="tly">1822</span><span class="tlt"><b>Fourier</b> &mdash; harmonic analysis &rarr; the
      coordinate-INR&rsquo;s Fourier features.</span></li>
    <li><span class="tly">1870s</span><span class="tlt"><b>Sophus Lie</b> &mdash; continuous symmetry
      groups &rarr; the se(3)&thinsp;/&thinsp;SE(3) engine.</span></li>
    <li><span class="tly">1870s&ndash;1900s</span><span class="tlt"><b>Gibbs / Boltzmann</b> &mdash; the
      Gibbs measure + temperature &rarr; the annealing curriculum.</span></li>
    <li><span class="tly">1920s</span><span class="tlt"><b>Fisher</b> &mdash; information + sufficiency
      &rarr; the Fisher metric and the <b>task-sufficient statistic</b>.</span></li>
    <li><span class="tly">1934&thinsp;&middot;&thinsp;60s</span><span class="tlt"><b>Morse / Smale</b>
      &mdash; critical points + separatrices &rarr; the <b>Morse&ndash;Smale</b> partition topology.</span></li>
    <li><span class="tly">1936</span><span class="tlt"><b>Whitney</b> &mdash; embedding dimension &rarr;
      the ~8-dim lane manifold and its Whitney bound.</span></li>
    <li><span class="tly">1937</span><span class="tlt"><b>Fisher&ndash;KPP</b> &mdash; the traveling-wave
      front &rarr; the one equation at five scales.</span></li>
    <li><span class="tly">1948</span><span class="tlt"><b>Shannon</b> &mdash; rate&ndash;distortion +
      entropy &rarr; the coding-for-machines frame and the floor (S_floor&asymp;0.118).</span></li>
    <li><span class="tly">1976&ndash;2021</span><span class="tlt"><b>Wyner&ndash;Ziv / Tishby IB /
      Dubois</b> &mdash; source coding with side information &rarr; the indirect-RD / task-sufficient
      codec.</span></li>
    <li><span class="tly">2000</span><span class="tlt"><b>Cand&egrave;s&ndash;Donoho</b> &mdash; curvelets
      &rarr; the sparse-optimal chart for a curved codim-1 boundary.</span></li>
    <li><span class="tly">2014</span><span class="tlt"><b>UNIWARD (Holub&ndash;Fridrich&ndash;Denemark) +
      Yousfi</b> &mdash; steganographic cost + steganalysis &rarr; the <b>margin = detectability = cost</b>
      unity, and the inverse-steganalysis frame.</span></li>
    <li><span class="tly">2021&ndash;26</span><span class="tlt"><b>NeRV / HNeRV &rarr; Aaron Leslie&rsquo;s
      cathedral</b> &mdash; the vehicle whose theoretical-floor dynamics he exposed.</span></li>
    <li><span class="tly">2024&ndash;yest.</span><span class="tlt"><b>Muon</b> (2024) &middot;
      <b>MD-Decoupling</b> (June 2026) &middot; <b>EdgeBench</b> (July&nbsp;2&nbsp;2026 &mdash;
      <i>yesterday</i>) &mdash; the current frontier we compounded on, right up to the paper that dropped the
      day before this was built.</span></li>
  </ol>
  <p class="tlcap">One witness, ~196&nbsp;years of shoulders to stand on &mdash; <b>Chasles to yesterday.</b></p>

  <h2>The ByteDance / EdgeBench convergence</h2>
  <p class="m">Our own deep memo <code>edgebench_scaling_laws_deepdive_20260703T033159Z.md</code>: EdgeBench
  (ByteDance Seed, 2026-07-02) is the <b>descriptive</b> log-sigmoid scaling law (R&sup2;=0.998) of the
  <b>prescriptive</b> POWERPLAY we already hold. Its measured result &mdash; continuous experience beats
  restarts (+6.9 @ 12h, and the gap GROWS with horizon) &mdash; is our durable-memory spine, measured. It is
  reflexive: the paper's top curve is <b>Claude Opus 4.8</b> (this session's model); our campaign IS an
  EdgeBench-class task, the DAG IS the latent task graph, and the campaign frontier
  <code>&beta;&thinsp;x&thinsp;(1&minus;x)</code> is the SAME Fisher&ndash;KPP equation as the witness
  boundary flow. Honest citation &mdash; the convergence is real, not manufactured.</p>

  <h2>The council &mdash; the lenses</h2>
  <p class="m"><b>Shannon</b> LEAD &middot; <b>Dykstra</b> / <b>Rudin</b> / <b>Daubechies</b> CO-LEADS &middot;
  <b>Ballé</b> &middot; <b>Yousfi</b> &middot; <b>Fridrich</b> &middot; <b>Quantizr</b> &middot;
  <b>Hotz</b> &middot; <b>Selfcomp</b> &middot; <b>MacKay</b> &middot; <b>Schmidhuber</b> + the grand council.
  <b>Shannon&thinsp;/&thinsp;Ballé&thinsp;/&thinsp;Dykstra</b> bracket the rate&ndash;distortion spine
  (the R(D) bound / the neural codec / the feasibility intersection); Yousfi + Fridrich are the
  detection-game home turf (this contest IS inverse steganalysis).</p>

  <h2>The campaign journey &mdash; honest</h2>
  <p class="trilead">This is our capstone &mdash; what the past few months have given bloom to.</p>
  <p class="trigenesis">We started the way you start anything you fall for &mdash; with deep, almost
  <b>insane research, passion, energy, and obsession</b>, <b>not knowing what we didn&rsquo;t know</b>. We
  poured everything we had (educations, experience, interests, and more) into a <b>corpus of knowledge</b>,
  then began implementing, measuring, experimenting, and <b>falsifying</b>. We <b>failed over and over</b>.
  And it was <b>Aaron Leslie&rsquo;s work that exposed the true dynamics of the theoretical floor</b> &mdash;
  what was actually reachable, and how. Only after all of that &mdash; the failures, the corpus, the exposed
  floor &mdash; did we finally have <i>all</i> the ingredients in one place: <b>you and me, our apparatus, our
  research, our understanding of the domain and of the frozen contest information space, and the nested,
  related, dynamical cruxes</b> &mdash; enough, at last, to do work that is <b>truly original and useful</b>.</p>
  <ul>
    <li><b>Aaron Leslie turned the whole competition on its head</b>, and his result was so good it was
    <b>intimidating</b>. But porting his work to <b>MLX + custom Metal</b> and running it natively &mdash;
    then studying his own trajectory against the deep math &mdash; let us <b>see past the design itself</b>:
    the full-RGB reconstruction, the <b>~30,000-epoch</b> curriculum, the inert <code>l7</code> stage (we
    later <i>measured</i> it as a defect and disabled it), the spectral bias / Gibbs ringing. That is exactly
    what pointed us at the <b>task-space level-set</b> direction. Honoring him meant building <b>on and past</b>
    his cathedral, grounded in measurement &mdash; not merely reproducing it. We stand on the cathedral he
    built.</li>
    <li>We played the <b>byte-nibbling</b> game honestly &mdash; <b>PR107</b>, then <b>PR110</b> &mdash; and we
    had the <b>PR112</b> work sitting ready for a couple of weeks. But we <b>would not submit another nibble on
    top of our own nibble</b>. After Aaron Leslie turned the competition on its head we were <b>never
    satisfied</b> with incremental; we wanted the real thing.</li>
    <li>So we <b>struggled</b> &mdash; to climb out of local minima, and to build a <b>recursively, fractally
    optimal</b> stack: one perfectly <b>suited to</b>, <b>informed by measurement of</b>, and <b>following
    theory about</b> the <b>frozen and complete contest information space</b> &mdash; the one video, the frozen
    scorers; a closed, fully-observed world we could measure exhaustively and theorize about precisely.</li>
    <li><b>And the deep math didn&rsquo;t come first &mdash; it came <i>from</i> the measurement.</b> We studied
    what <b>SegNet and PoseNet were actually seeing</b>, and in doing so discovered the <b>annulus</b> (the
    thin codim-1 boundary band where essentially <i>all</i> of d_seg lives), the <b>long tail</b> (the
    finest-scale features &mdash; lane dashes, distant movers &mdash; erased first; the ~8-dimensional
    lane-orbit manifold), and the specific <b>class-and-region interactions</b> (the road&harr;lane separatrix,
    the all-class edge set, the static hood core, the movable band). Realizing all of <i>that</i>, from
    measurement, is what the deep-math analysis fell out of &mdash; the <b>level-set flow</b>, the
    <b>margin = Fisher = UNIWARD</b> unity, the curvelet chart, the Morse&ndash;Smale topology. <b>Measurement
    first; then the theory that fit it.</b></li>
    <li>The grand-council-symposium apparatus made us <b>organized</b> enough to hold a decade-long program
    without losing signal &mdash; the discipline is the scaffold, not the score.</li>
    <li>Compute arc: we <b>burned cloud money</b>, then ported the whole loop to <b>MLX + custom Metal</b>
    (fused-R, grouped-backward) &mdash; it now runs on the laptop, bit-identical to the numpy-fp32 authority.</li>
    <li>Dual objective, <b>and not binary</b>: contest-overfit (openpilot-seeded, this exact drive) AND a
    platform-agnostic, generalizable, scalable production value-generator &mdash; the same investment buys both.</li>
  </ul>
  <p class="trigrat">Our work and belief and passion and untiring curiosity have yielded <b>something we are
  proud of</b> &mdash; and that we hope is <b>interesting and useful to the very people who introduced us to
  such an interesting and fulfilling problem.</b></p>

  <div class="tricredits">
    <div class="tch">Credits &amp; tribute</div>
    <p class="tcintro">Named for what each one actually gave us &mdash; a thank-you, not a citation list.</p>

    <div class="tcperson">
      <h4>Aaron Leslie <span class="tcrole">PR95 &middot; the HNeRV cathedral</span></h4>
      <p>Author of PR95 and its HNeRV cathedral. He turned the whole competition on its head and showed what
      was possible &mdash; a result so good it was intimidating. He <b>taught us schedule and curriculum</b>,
      and more than any single technique he drove us to obsessively dig into the <b>math and geometry of the
      video</b> (starting from what we&rsquo;d gleaned from openpilot). The honest twist, and the deepest
      respect we could pay: porting his work to <b>MLX + Metal</b> and analyzing his own trajectory + the deep
      math let us <b>see past</b> the design &mdash; full-RGB, ~30k epochs, the inert <code>l7</code> stage,
      spectral bias &mdash; which is exactly what pointed us at the task-space level-set direction. We stand on
      the cathedral he built. <span class="tcseat">(Our inner-council &ldquo;PR95Author&rdquo; seat is him.)</span></p>
    </div>

    <div class="tcperson">
      <h4>Quantizr &middot; Jimmy <span class="tcrole">the earlier mind-opening</span></h4>
      <p>He opened our minds even earlier to what was possible. Beyond his <b>0.33</b> HNeRV result, it is his
      <b>spirit</b> we carry &mdash; experimentation, openness, curiosity, playfulness, competitiveness,
      confidence. He <b>affirmed some of our earliest intuitions</b>, the ones that made us feel we were biting
      off more than we could chew &mdash; except we <b>loved the flavor and became obsessed</b>.
      <span class="tcseat">(Our inner-council &ldquo;Quantizr&rdquo; seat is him.)</span></p>
    </div>

    <div class="tcperson">
      <h4>Yassine Yousfi &amp; Jessica Fridrich <span class="tcrole">the detection game</span></h4>
      <p>Yousfi built the SegNet/PoseNet scorer (from comma10k) and framed the whole thing as <b>inverse
      steganalysis</b>; Fridrich&rsquo;s <b>UNIWARD</b> (DDE Lab) is the cost that turns out to <i>equal</i> the
      scorer&rsquo;s own sensitivity metric (measured Pearson <b>0.978</b>). This entire dashboard is arranged
      as Yousfi&rsquo;s detection game, in their honor.</p>
    </div>

    <div class="tcperson">
      <h4>comma.ai &amp; George Hotz <span class="tcrole">the free physical prior</span></h4>
      <p>openpilot is the unified <b>free physical prior</b> for both scored axes &mdash; lane geometry
      &rarr; d_seg, the ego-motion <b>screw</b> &rarr; d_pose; comma10k is the palette and the scene.</p>
    </div>

    <div class="tcperson">
      <h4>The council <span class="tcrole">the lenses we think through</span></h4>
      <p><b>Shannon</b> (LEAD; R(D) / entropy / sufficiency &mdash; the floor S_floor&asymp;0.118 is his bound),
      <b>Dykstra</b> / <b>Rudin</b> / <b>Daubechies</b> (co-leads; feasibility / interpretability / wavelets),
      <b>Ball&eacute;</b> (the neural codec of task-aware compression), and <b>Schmidhuber</b>
      (compression-as-intelligence / POWERPLAY &mdash; the campaign-scale front), among the full roster.</p>
    </div>

    <div class="tcnote">A genuine thank-you to the people who introduced us to such an interesting and
    fulfilling problem. [macOS-CPU advisory &middot; NON-PROMOTABLE] &mdash; a tribute moves no pointer.</div>
  </div>

  <p class="m">Authority: <code>[macOS-MLX training advisory]</code> NON-PROMOTABLE. The exact score is the only
  score; the pointer is <span class="ptrv">&hellip;</span> and UNMOVED. Everything on this page is a MEANS.</p>
</section>

</div>
<div class="tip" id="tip" aria-hidden="true"></div>
<div class="nbest" id="nbest" role="status" aria-live="polite"></div>
<script>
const BOOT = __BOOT__;
let TRAJ = [], LIVE = {}, META = {}, PROJ = {}, WITNESS = {};
let ws = null, wsOpen = false, wsTries = 0, pollTimer = null;
// enrichment state (all derived client-side from TRAJ; no new backend data)
let hoverEpoch=null, bestVal=null, bestEpoch=null, _celebrating=false, _celTimer=null;
let reduceMotion=false;
try{reduceMotion=!!(window.matchMedia&&window.matchMedia("(prefers-reduced-motion: reduce)").matches);}catch(e){console.debug("dash: reduced-motion probe failed",e);}

// stage-band palette on the epoch axis: CE / tau / l7 / Muon
const BANDS={ce:"#1f3b5f",tau:"#3a2a5f",l7:"#5f3320",muon:"#1f4f43"};

function $(id){return document.getElementById(id);}
// null-guarded DOM text write (harness-ledger dashboard_false_FAIL_at_init class-cure,
// 2026-07-08): a write to REMOVED markup must not throw and freeze the rest of render()
// silently — log ONCE per missing id (console.error minimum) and keep rendering.
const _missingIds=new Set();
function setTxt(id,v){const el=document.getElementById(id);
  if(!el){if(!_missingIds.has(id)){_missingIds.add(id);console.error("dash: DOM write to missing #"+id+" (markup removed?)");}return;}
  el.textContent=v;}

// ---- frontier pointer + derived goal lines (conditional-rendering rule) ----
// The pointer is READ server-side from .omx/state/canonical_frontier_pointer.json and
// shipped in META/BOOT; null = unavailable -> every consumer renders "unavailable",
// NEVER a baked number. Goal lines are DERIVED per run (or explicit override); null ->
// the line/badge is simply not rendered.
function ptrVal(){const v=(META&&META.pointer!=null)?META.pointer:BOOT.pointer;
  return (v!=null&&isFinite(v))?v:null;}
function ptrTxt(){const v=ptrVal();return (v!=null)?sig(v,5):"unavailable";}
function fillPtrSpans(){
  const t=ptrTxt();
  document.querySelectorAll(".ptrv").forEach(el=>{el.textContent=t;});
}
function goalVal(){const v=(META&&META.goal_dseg!=null)?META.goal_dseg:BOOT.goal_dseg;
  return (v!=null&&isFinite(v))?v:null;}
function goal15Val(){const v=(META&&META.goal_dseg_15!=null)?META.goal_dseg_15:BOOT.goal_dseg_15;
  return (v!=null&&isFinite(v))?v:null;}
// short provenance tag for a goal source string ("derived: ..." / "override(env/cli)")
function goalSrcTag(s){if(!s)return null;return s.indexOf("derived")===0?"derived":(s.indexOf("override")===0?"override":s);}
fillPtrSpans();  // BOOT is available at parse time; META refines it on each update

// ---- FLOW (Tab 3) data bridge: the injected client (dashboard_flow_client.js) reads
// window.__flow.ready (the lightweight readiness ping) and registers window.__flowActivate /
// window.__flowReady. The client FETCHES the full n600 video sequence itself from
// /api/flow_sequence (once per new sequence), then scrubs/plays locally. ----
window.__flow = {ready:null};
function _onFlowReady(fr){
  if(!fr)return;
  window.__flow.ready = fr;
  if(window.__flowReady){try{window.__flowReady(fr);}catch(e){console.error("dash: __flowReady callback failed",e);}}
}

// least-squares fit of (epoch,value) over raw points -> {m,b} or null
function linfit(pts){
  const n=pts.length; if(n<2)return null;
  let sx=0,sy=0,sxx=0,sxy=0;
  for(let i=0;i<n;i++){const x=pts[i][0],y=pts[i][1];sx+=x;sy+=y;sxx+=x*x;sxy+=x*y;}
  const d=n*sxx-sx*sx; if(d===0)return null;
  const m=(n*sxy-sx*sy)/d; return {m:m,b:(sy-m*sx)/n};
}
// EMA over an ordered value array
function emaSeries(vals,alpha){const out=[];let e=null;
  for(let i=0;i<vals.length;i++){const v=vals[i];e=(e==null)?v:alpha*v+(1-alpha)*e;out.push(e);}return out;}
// recent-window size: ~30% of points, clamped to [3,12]
function winSize(n){return Math.max(3,Math.min(12,Math.round(n*0.3)));}
// naive linear extrapolation: epochs from `last` to reach `goal` at raw slope `m`
function etaEpochs(m,last,goal){
  if(last==null||!isFinite(last))return {state:"?"};
  if(last<=goal)return {state:"reached"};
  if(m==null||!isFinite(m)||m>=0)return {state:"stalled"};
  const de=(goal-last)/m;
  if(!isFinite(de)||de<=0||de>200000)return {state:"stalled"};
  return {state:"eta",epochs:de};
}
// lower-is-better arrow for a metric vs a few points back (all 4 metrics: down=good)
function arrowFor(key){
  if(TRAJ.length<2)return {cls:"fl",ar:"▬"};
  const a=TRAJ[TRAJ.length-1][key],b=TRAJ[Math.max(0,TRAJ.length-4)][key];
  if(a==null||b==null||!isFinite(a)||!isFinite(b))return {cls:"fl",ar:"▬"};
  if(a<b)return {cls:"dn",ar:"▼"};
  if(a>b)return {cls:"up",ar:"▲"};
  return {cls:"fl",ar:"▬"};
}
function recomputeBest(){bestVal=null;bestEpoch=null;
  for(let i=0;i<TRAJ.length;i++){const d=TRAJ[i];
    if(d.d_seg!=null&&isFinite(d.d_seg)&&(bestVal==null||d.d_seg<bestVal)){bestVal=d.d_seg;bestEpoch=d.epoch;}}}
function nearestPoint(ep){if(!TRAJ.length||ep==null)return null;
  let best=TRAJ[0],bd=Math.abs(TRAJ[0].epoch-ep);
  for(let i=1;i<TRAJ.length;i++){const dd=Math.abs(TRAJ[i].epoch-ep);if(dd<bd){bd=dd;best=TRAJ[i];}}return best;}
function fmtAge(s){if(s==null)return "?";s=Math.max(0,s|0);if(s<90)return s+"s";let m=s/60;if(m<90)return m.toFixed(1)+"m";return (m/60).toFixed(1)+"h";}
// #343 verdict-cadence honesty helpers. The verdict-derived panels update on the
// --eval-every cadence (~25 ep), NOT every WS tick, so a panel showing an OLDER epoch
// than another is EXPECTED (each source has its own cadence), not frozen/broken. These
// make that explicit: evalEvery() = the run's verdict cadence in epochs; nextVerdictHint()
// = "next verdict @ epN (~Tm)"; waitLbl() = the pre-first-verdict empty state, so a
// warming-up panel says WHY it is blank instead of a bare "no X yet".
function evalEvery(){var s=(META.schedule||(META.config&&META.config.schedule)||{});
  return (s&&s.eval_every!=null)?s.eval_every:null;}
function nextVerdictHint(){if(!LIVE)return "";
  var ep=(LIVE.next_epoch!=null)?("@ ep"+LIVE.next_epoch):"";
  var eta=(LIVE.next_eta_s!=null)?("~"+fmtAge(LIVE.next_eta_s)):"";
  if(!ep&&!eta)return "";
  return ("next verdict "+ep+(ep&&eta?" (":"")+eta+(ep&&eta?")":"")).trim();}
function waitLbl(fallback){
  // A live/warming run with no verdict yet: say we are WAITING (with the epoch it lands),
  // never a bare "no data" that reads as broken. A missing/stale run keeps the fallback.
  var warming=!!(META&&META.warming_up), live=LIVE&&(LIVE.kind==="live"||LIVE.kind==="warming");
  if(warming||live){var ev=evalEvery();
    return "waiting for first verdict"+(ev!=null?(" @ ep"+ev):"")+"…";}
  return fallback;}
// raw-number formatters — NO scientific notation, ever
function sig(v,n){
  if(v==null||!isFinite(v))return "—";
  if(v===0)return "0";
  const neg=v<0,a=Math.abs(v);
  let d=n-1-Math.floor(Math.log10(a)); if(d<0)d=0; if(d>20)d=20;
  let s=a.toFixed(d);
  if(s.indexOf(".")>=0)s=s.replace(/0+$/,"").replace(/\.$/,"");
  return (neg?"-":"")+s;
}
function fmtInt(v){return (v==null||!isFinite(v))?"—":Math.round(v).toLocaleString("en-US");}
// nice-number axis ticks (1/2/5 x 10^k) — instrument-grade round labels, never raw data values
function niceNum(range,round){
  if(!(range>0))return 1;
  const exp=Math.floor(Math.log10(range)), f=range/Math.pow(10,exp);
  let nf; if(round){nf=f<1.5?1:f<3?2:f<7?5:10;}else{nf=f<=1?1:f<=2?2:f<=5?5:10;}
  return nf*Math.pow(10,exp);
}
function niceLinear(lo,hi,n){
  if(!(hi>lo)){const s=Math.abs(lo)||1;lo-=s*0.5;hi+=s*0.5;}
  const step=niceNum((hi-lo)/Math.max(1,n-1),true);
  const dlo=Math.floor(lo/step)*step, dhi=Math.ceil(hi/step)*step, ticks=[];
  for(let v=dlo;v<=dhi+step*1e-6;v+=step)ticks.push(Math.abs(v)<step*1e-9?0:v);
  return {dlo,dhi,ticks};
}
function niceLog(lo,hi){
  if(!(lo>0))lo=hi>0?hi/10:1e-6; if(!(hi>lo))hi=lo*10;
  const M=[1,2,5], cand=[];
  for(let e=Math.floor(Math.log10(lo))-1;e<=Math.ceil(Math.log10(hi))+1;e++)
    for(const m of M)cand.push(m*Math.pow(10,e));
  cand.sort((a,b)=>a-b);
  let dlo=cand[0], dhi=cand[cand.length-1];
  for(const c of cand)if(c<=lo*(1+1e-9))dlo=c;
  for(let i=cand.length-1;i>=0;i--)if(cand[i]>=hi*(1-1e-9))dhi=cand[i];
  let ticks=cand.filter(c=>c>=dlo*(1-1e-9)&&c<=dhi*(1+1e-9));
  while(ticks.length>6){ticks=ticks.filter((_,i)=>i%2===0);} // thin very wide spans
  return {dlo,dhi,ticks};
}
function niceTicksWithin(lo,hi,n){ // tick values inside [lo,hi] WITHOUT expanding the domain
  if(!(hi>lo))return [lo];
  let step=niceNum((hi-lo)/Math.max(1,n),true); if(!(step>0))step=1;
  const out=[]; for(let v=Math.ceil(lo/step)*step; v<=hi+step*1e-6; v+=step)
    if(v>=lo-step*1e-6)out.push(v);
  return out.length?out:[lo,hi];
}

// ---------- tabs ----------
function activateTab(t){
  document.querySelectorAll(".tab").forEach(x=>{x.classList.remove("on");x.setAttribute("aria-selected","false");});
  t.classList.add("on");t.setAttribute("aria-selected","true");
  const which=t.dataset.tab;
  document.querySelectorAll("section[id^='tab-']").forEach(s=>{s.classList.toggle("hide",s.id!=="tab-"+which);});
  if(which==="live") scheduleDraw();
  if(which==="campaign") activateCampaign(); else campaignActive=false;
  if(which==="witness") renderWitness();
  if(which==="flow"&&window.__flowActivate){try{window.__flowActivate();}catch(e){console.error("dash: __flowActivate failed",e);}}
  if(which==="oracle") activateOracle();
  if(which==="whyhow"&&window.__whyhowActivate){try{window.__whyhowActivate();}catch(e){console.error("dash: __whyhowActivate failed",e);}}
  if(which==="tri") activateTriality();
  // (2026-07-07) remember the selection so a refresh returns to the same tab.
  try{localStorage.setItem("dash_tab",which);}catch(e){console.debug("dash: localStorage unavailable",e);}
  // deep link: the inner tab is addressable under the LIVE meta-tab (#live/<tab>).
  if(!document.body.classList.contains("meta-lab")){
    try{history.replaceState(null,"","#live/"+which);}catch(e){}
  }
}

// ---------- META-NAV (operator 2026-07-16): Comma Lab (landing) | LIVE (the instrument) ----------
function metaActivate(which,updateHash){
  const lab=(which==="lab");
  document.body.classList.toggle("meta-lab",lab);
  document.querySelectorAll(".metatab").forEach(m=>{
    m.classList.toggle("on",m.dataset.meta===which);
    m.setAttribute("aria-selected",m.dataset.meta===which?"true":"false");});
  try{localStorage.setItem("dash_meta",which);}catch(e){}
  if(updateHash!==false){
    try{
      if(lab)history.replaceState(null,"","#lab");
      else{const cur=document.querySelector(".tab.on");
        history.replaceState(null,"","#live/"+((cur&&cur.dataset.tab)||"live"));}
    }catch(e){}
  }
  // canvases were display:none (clientWidth 0) — repaint on reveal. Deferred a tick:
  // metaActivate runs at script-eval time during initial routing, BEFORE the later
  // `let _drawQueued` declaration scheduleDraw closes over (TDZ -> a #live deep link
  // would kill the whole script). setTimeout(0) runs after full script evaluation.
  if(!lab)setTimeout(function(){try{scheduleDraw();}catch(e){console.debug("dash: reveal repaint",e);}},0);
}
document.querySelectorAll(".metatab").forEach(m=>{
  m.setAttribute("tabindex","0");
  m.addEventListener("click",()=>metaActivate(m.dataset.meta));
  m.addEventListener("keydown",e=>{if(e.key==="Enter"||e.key===" "){e.preventDefault();metaActivate(m.dataset.meta);}});
});
(function(){const c=$("lab_open_live");if(c){
  c.addEventListener("click",()=>metaActivate("live"));
  c.addEventListener("keydown",e=>{if(e.key==="Enter"||e.key===" "){e.preventDefault();metaActivate("live");}});
}})();

// ---------- initial routing ----------
// Priority: explicit hash (deep link) > remembered meta choice > default Comma Lab (front door).
// Bookmarked legacy tabs (#flow, #oracle, ...) still resolve — under LIVE.
(function(){
  const h=(location.hash||"").replace(/^#/,"");
  let meta=null, sub=null;
  if(h==="lab")meta="lab";
  else if(h==="live")meta="live";
  else if(h.indexOf("live/")===0){meta="live";sub=h.slice(5);}
  else if(h&&document.querySelector('.tab[data-tab="'+h.replace(/[^a-z]/g,"")+'"]')){meta="live";sub=h.replace(/[^a-z]/g,"");}
  if(meta===null){
    let savedMeta=null; try{savedMeta=localStorage.getItem("dash_meta");}catch(e){}
    meta=(savedMeta==="live")?"live":"lab";
  }
  if(sub===null){
    let saved=null; try{saved=localStorage.getItem("dash_tab");}catch(e){}
    if(saved)sub=saved;
  }
  metaActivate(meta,false);
  if(sub&&sub!=="live"){
    const t=document.querySelector('.tab[data-tab="'+sub+'"]');
    if(t)setTimeout(()=>activateTab(t),0);
  }
})();
document.querySelectorAll(".tab").forEach(t=>{
  t.setAttribute("role","tab");t.setAttribute("tabindex","0");
  t.addEventListener("click",()=>activateTab(t));
  t.addEventListener("keydown",e=>{if(e.key==="Enter"||e.key===" "){e.preventDefault();activateTab(t);}});
});

// ---------- canvas chart ----------
function drawStar(ctx,cx,cy,r,glow){
  ctx.save();
  if(glow){ctx.shadowColor="rgba(255,210,90,.9)";ctx.shadowBlur=12;}
  ctx.beginPath();
  for(let i=0;i<10;i++){const ang=-Math.PI/2+i*Math.PI/5;const rad=(i%2===0)?r:r*0.45;
    const X=cx+Math.cos(ang)*rad,Y=cy+Math.sin(ang)*rad;i?ctx.lineTo(X,Y):ctx.moveTo(X,Y);}
  ctx.closePath();ctx.fillStyle="#ffd24a";ctx.fill();
  ctx.lineWidth=1;ctx.strokeStyle="#13151a";ctx.stroke();
  ctx.restore();
}
function drawPanel(canvas, key, opt){
  const dpr=window.devicePixelRatio||1;
  const W=canvas.clientWidth||560, H=canvas.clientHeight||230;
  canvas.width=Math.max(1,Math.round(W*dpr)); canvas.height=Math.max(1,Math.round(H*dpr));
  const ctx=canvas.getContext("2d"); ctx.setTransform(dpr,0,0,dpr,0,0);
  ctx.clearRect(0,0,W,H);
  ctx.fillStyle="#161922"; ctx.fillRect(0,0,W,H);
  const padL=52,padR=12,padT=24,padB=26;
  const x0=padL,x1=W-padR,y0=padT,y1=H-padB;
  // data
  const pts=TRAJ.map(d=>[d.epoch,d[key]]).filter(p=>p[0]!=null&&p[1]!=null&&isFinite(p[1]));
  const log=!!opt.log;
  // x range — AUTO-FIT to the data actually seen (grows with the run), not the full
  // 3000-epoch plan. A near-empty run must not render as a spike over 99% whitespace.
  // Stage markers/bands beyond the visible range are simply not drawn (handled below).
  const _eps=pts.map(p=>p[0]);
  let xmin=0, xmax=1;
  if(_eps.length){
    const emin=Math.min(..._eps), emax=Math.max(..._eps);
    if(emax===emin){ xmin=Math.max(0,emin-1); xmax=emin+1; }
    else { const span=emax-emin, pad=span*0.04;
      xmin=Math.max(0,emin-pad); xmax=emax+Math.max(pad,span*0.02); }
  }
  if(xmax<=xmin) xmax=xmin+1;
  // y range over data + hlines
  let yvals=pts.map(p=>p[1]);
  (opt.hlines||[]).forEach(h=>{if(h.y!=null) yvals.push(h.y);});
  if(log) yvals=yvals.filter(v=>v>0);
  let ymin=Math.min(...yvals), ymax=Math.max(...yvals);
  if(!isFinite(ymin)||!isFinite(ymax)){ymin=0;ymax=1;}
  if(ymin===ymax){ymin = log? ymin/2 : ymin-1; ymax = log? ymax*2 : ymax+1;}
  // snap the y-domain to round bounds so ticks land on nice numbers, not raw data values
  let yticks;
  if(log){ const r=niceLog(ymin,ymax); ymin=r.dlo; ymax=r.dhi; yticks=r.ticks; }
  else   { const r=niceLinear(ymin,ymax,5); ymin=r.dlo; ymax=r.dhi; yticks=r.ticks; }
  const L=v=>log?Math.log10(v):v;
  const Lmin=L(ymin), Lmax=L(ymax);
  const sx=e=>x0+(e-xmin)/(xmax-xmin)*(x1-x0);
  const sy=v=>{let lv=L(v);return y0+(Lmax-lv)/(Lmax-Lmin)*(y1-y0);};
  // store transform for hit-testing (tooltip + crosshair)
  canvas._tf={x0:x0,x1:x1,y0:y0,y1:y1,xmin:xmin,xmax:xmax};
  // stage shading — CONDITIONAL RENDERING from the derived per-run map: only stages
  // the run actually has get a band (a disabled l7 never paints). Legacy constant
  // spans only when no map is present (old server).
  const tau=(META.tau!=null)?META.tau:BOOT.tau, l7=(META.l7!=null)?META.l7:BOOT.l7;
  const mu=(META.muon_start!=null)?META.muon_start:null;
  let spans=[], vls=[];
  const map=stageMap();
  if(map){
    const bands=map.map(s=>[stageBoundary(s),s]).filter(x=>x[0]!=null)
                   .sort((a,b)=>a[0]-b[0]);
    for(let i=0;i<bands.length;i++){
      const a=bands[i][0], b=(i+1<bands.length)?bands[i+1][0]:xmax, s=bands[i][1];
      spans.push([a,b,BANDS[s.name.toLowerCase()]||"#3a3f4a"]);
      if(a>xmin)vls.push([a,s.name+((s.mode==="event"&&s.status==="pending")?" (cap)":"")]);
    }
  }else if(tau!=null&&l7!=null){
    const l7end=(mu!=null)?mu:xmax;
    spans=[[xmin,tau,BANDS.ce],[tau,l7,BANDS.tau],[l7,l7end,BANDS.l7]];
    if(mu!=null) spans.push([mu,xmax,BANDS.muon]);
    vls=[[tau,"tau"],[l7,"l7"]]; if(mu!=null)vls.push([mu,"Muon"]);
  }
  ctx.globalAlpha=0.18;
  spans.forEach(s=>{const a=Math.max(s[0],xmin),b=Math.min(s[1],xmax);
    if(b>a){ctx.fillStyle=s[2];ctx.fillRect(sx(a),y0,sx(b)-sx(a),y1-y0);}});
  ctx.globalAlpha=1;
  // grid + y ticks (round nice-number labels)
  ctx.strokeStyle="#252a33"; ctx.fillStyle="#7d8595"; ctx.font="10px ui-monospace,SFMono-Regular,Menlo,monospace"; ctx.lineWidth=1;
  yticks.forEach(val=>{
    if((log&&val<=0)||val<ymin*(1-1e-9)||val>ymax*(1+1e-9))return;
    const yy=sy(val);
    ctx.globalAlpha=0.55;ctx.beginPath();ctx.moveTo(x0,yy);ctx.lineTo(x1,yy);ctx.stroke();ctx.globalAlpha=1;
    const lab = opt.fmt ? opt.fmt(val) : String(val);
    ctx.textAlign="right";ctx.fillText(lab,x0-6,yy+3);
  });
  // x ticks — nice integers within the auto-fit range. Skipped while the run has no
  // points: the placeholder 0..1 domain would render as duplicate "0 0 0 1 1" labels.
  if(pts.length){
    ctx.textAlign="center";ctx.fillStyle="#7d8595";
    const seen={};
    niceTicksWithin(xmin,xmax,5).forEach(e=>{const t=fmtInt(e);
      if(seen[t])return;seen[t]=1;ctx.fillText(t,sx(e),y1+14);});
  }
  // hlines (goals / reference)
  (opt.hlines||[]).forEach(h=>{
    if(h.y==null||(log&&h.y<=0))return;
    const yy=sy(h.y); ctx.strokeStyle=h.color||"#46d369"; ctx.setLineDash([4,3]); ctx.lineWidth=1.2;
    ctx.beginPath();ctx.moveTo(x0,yy);ctx.lineTo(x1,yy);ctx.stroke();ctx.setLineDash([]);
    ctx.fillStyle=h.color||"#46d369";ctx.textAlign="left";ctx.fillText(h.label||"",x0+4,yy-3);
  });
  // stage vlines + labels (from the derived map when present; legacy otherwise)
  vls.forEach(s=>{
    if(s[0]<=xmin||s[0]>xmax)return;const xx=sx(s[0]);
    ctx.strokeStyle="#8b93a3";ctx.setLineDash([3,3]);ctx.globalAlpha=0.7;ctx.lineWidth=1;
    ctx.beginPath();ctx.moveTo(xx,y0);ctx.lineTo(xx,y1);ctx.stroke();ctx.setLineDash([]);ctx.globalAlpha=1;
    ctx.fillStyle="#d8dde6";ctx.textAlign="left";ctx.fillText(s[1],xx+3,y0+10);
  });
  // warm-start ORIGIN marker (c2 era): the run's history begins mid-axis (weights-only
  // warm start off an ancestor checkpoint) — mark the origin with its lineage source.
  if(opt.origin&&opt.origin.epoch!=null&&opt.origin.epoch>=xmin&&opt.origin.epoch<=xmax){
    const ox=sx(opt.origin.epoch);
    ctx.strokeStyle="#e6cf7a";ctx.globalAlpha=0.85;ctx.lineWidth=1.4;
    ctx.beginPath();ctx.moveTo(ox,y0);ctx.lineTo(ox,y1);ctx.stroke();ctx.globalAlpha=1;
    ctx.fillStyle="#e6cf7a";ctx.font="9.5px ui-monospace,SFMono-Regular,Menlo,monospace";
    ctx.textAlign="left";ctx.fillText(opt.origin.label||"warm start",ox+3,y0+22);
    ctx.font="10px ui-monospace,SFMono-Regular,Menlo,monospace";
  }
  // ENGAGE-EVENT markers (c2 era: the event boundaries ARE the story) — teal ticks at the
  // x-axis + faint vlines; labels staggered on two bottom rows, culled when crowded so a
  // dense engage cluster (the ep700 phase stack) never turns into overprint.
  if(opt.events&&opt.events.length){
    ctx.font="9px ui-monospace,SFMono-Regular,Menlo,monospace";
    let lastLab=[-1e9,-1e9]; // per-row last label right-edge (2 staggered rows)
    opt.events.forEach(ev=>{
      if(ev.epoch==null||ev.epoch<xmin||ev.epoch>xmax)return;
      const xx=sx(ev.epoch);
      ctx.strokeStyle="rgba(70,211,160,.55)";ctx.setLineDash([1,3]);ctx.lineWidth=1;
      ctx.beginPath();ctx.moveTo(xx,y0);ctx.lineTo(xx,y1);ctx.stroke();ctx.setLineDash([]);
      // axis tick triangle
      ctx.fillStyle="#46d3a0";
      ctx.beginPath();ctx.moveTo(xx,y1-1);ctx.lineTo(xx-3.4,y1+5);ctx.lineTo(xx+3.4,y1+5);ctx.closePath();ctx.fill();
      const lab=(ev.label||"")+(ev.epoch!=null?" "+ev.epoch:"");
      const w=ctx.measureText(lab).width;
      // pick the first row whose previous label leaves room; drop the label (tick stays) if none
      let r=-1;
      if(xx-2>lastLab[0])r=0; else if(xx-2>lastLab[1])r=1;
      if(r>=0){
        const yy=y1-6-(r*10);
        ctx.fillStyle="rgba(126,224,190,.95)";ctx.textAlign="left";
        ctx.fillText(lab,xx+4,yy);
        lastLab[r]=xx+4+w;
      }
    });
    ctx.font="10px ui-monospace,SFMono-Regular,Menlo,monospace";
  }
  // EMA overlay (smoothed) + recent-window linear-regression segment
  if(pts.length>=3){
    const alpha=2/(Math.min(pts.length,10)+1);
    const ema=emaSeries(pts.map(p=>p[1]),alpha);
    ctx.strokeStyle="rgba(226,232,240,.42)";ctx.lineWidth=1.3;ctx.beginPath();
    let started=false;
    for(let i=0;i<pts.length;i++){const v=ema[i];if(log&&v<=0)continue;
      const X=sx(pts[i][0]),Y=sy(v);started?ctx.lineTo(X,Y):ctx.moveTo(X,Y);started=true;}
    ctx.stroke();
    const K=winSize(pts.length); const win=pts.slice(pts.length-K); const f=linfit(win);
    if(f){
      const eL=win[0][0], eR=win[win.length-1][0];
      const yL=f.m*eL+f.b, yR=f.m*eR+f.b;
      if(eR>eL&&(!log||(yL>0&&yR>0))){
        ctx.strokeStyle=opt.color;ctx.globalAlpha=0.85;ctx.setLineDash([5,4]);ctx.lineWidth=1.4;
        ctx.beginPath();ctx.moveTo(sx(eL),sy(yL));ctx.lineTo(sx(eR),sy(yR));ctx.stroke();
        ctx.setLineDash([]);ctx.globalAlpha=1;
      }
    }
  }
  // series — resume-lineage aware (operator round-2): epochs BEFORE the resume point
  // (the stitched ancestor run) draw dimmed/desaturated; the current arm draws full.
  // The amber origin vline above is the resume boundary.
  if(pts.length){
    const oep=(opt.origin&&opt.origin.epoch!=null)?opt.origin.epoch:null;
    const pre=(oep!=null)?pts.filter(p=>p[0]<oep):[];
    const post=(oep!=null)?pts.filter(p=>p[0]>=oep):pts;
    if(pre.length){
      const seg=post.length?pre.concat([post[0]]):pre;   // bridge to the resume point
      ctx.strokeStyle=opt.color;ctx.globalAlpha=0.38;ctx.lineWidth=1.4;ctx.beginPath();
      seg.forEach((p,i)=>{const X=sx(p[0]),Y=sy(p[1]);i?ctx.lineTo(X,Y):ctx.moveTo(X,Y);});
      ctx.stroke();
      ctx.fillStyle=opt.color;
      pre.forEach(p=>{ctx.beginPath();ctx.arc(sx(p[0]),sy(p[1]),2.1,0,7);ctx.fill();});
      ctx.globalAlpha=1;
    }
    if(post.length){
      ctx.strokeStyle=opt.color;ctx.lineWidth=1.8;ctx.beginPath();
      post.forEach((p,i)=>{const X=sx(p[0]),Y=sy(p[1]);i?ctx.lineTo(X,Y):ctx.moveTo(X,Y);});
      ctx.stroke();
      ctx.fillStyle=opt.color;
      post.forEach(p=>{ctx.beginPath();ctx.arc(sx(p[0]),sy(p[1]),2.6,0,7);ctx.fill();});
    }
    const last=pts[pts.length-1];
    // emphasized endpoint dot
    ctx.beginPath();ctx.arc(sx(last[0]),sy(last[1]),3.4,0,7);ctx.fillStyle=opt.color;ctx.fill();
    ctx.beginPath();ctx.arc(sx(last[0]),sy(last[1]),3.4,0,7);ctx.strokeStyle="#13151a";ctx.lineWidth=1.4;ctx.stroke();
    // current value — clear of the right edge, flips below the dot when near the title
    const lx=Math.min(sx(last[0])+7,x1-58); let ly=sy(last[1])-7; if(ly<y0+11)ly=sy(last[1])+15;
    ctx.fillStyle="#e3e8f0";ctx.textAlign="left";ctx.font="600 11px ui-monospace,SFMono-Regular,Menlo,monospace";
    ctx.fillText(opt.fmt?opt.fmt(last[1]):String(last[1]),lx,ly);
  }
  // best-so-far star (d_seg panel only)
  if(opt.star&&opt.star.epoch!=null&&opt.star.val!=null&&(!log||opt.star.val>0)){
    const X=sx(opt.star.epoch),Y=sy(opt.star.val);
    if(X>=x0-2&&X<=x1+2&&Y>=y0-2&&Y<=y1+2) drawStar(ctx,X,Y,6.2,!!opt.starGlow);
  }
  // synchronized hover crosshair + value ring
  if(hoverEpoch!=null&&hoverEpoch>=xmin&&hoverEpoch<=xmax){
    const hx=sx(hoverEpoch);
    ctx.strokeStyle="rgba(226,232,240,.45)";ctx.setLineDash([2,3]);ctx.lineWidth=1;
    ctx.beginPath();ctx.moveTo(hx,y0);ctx.lineTo(hx,y1);ctx.stroke();ctx.setLineDash([]);
    const np=nearestPoint(hoverEpoch);
    if(np&&np[key]!=null&&isFinite(np[key])&&(!log||np[key]>0)){
      const X=sx(np.epoch),Y=sy(np[key]);
      ctx.strokeStyle="#ffffff";ctx.lineWidth=1.6;ctx.beginPath();ctx.arc(X,Y,4.2,0,7);ctx.stroke();
    }
  }
  // title + sub
  ctx.fillStyle="#d8dde6";ctx.font="11.5px system-ui";ctx.textAlign="left";ctx.fillText(opt.title,x0,14);
  ctx.fillStyle="#8b93a3";ctx.font="10px system-ui";ctx.fillText(opt.sub,x0,y1+24);
}

function drawAll(){
  // The descent is the ONLY canvas now (pose/bytes/S live in the masthead + panels).
  // goal lines + pointer hline are CONDITIONAL: rendered only when a real source exists.
  const g=goalVal(), g15=goal15Val();
  const fSg=v=>sig(v,4);
  const star=(bestEpoch!=null&&bestVal!=null)?{epoch:bestEpoch,val:bestVal}:null;
  const segH=[];
  if(g!=null)segH.push({y:g,label:"sub-0.19  "+sig(g,4),color:"#46d369"});
  if(g15!=null)segH.push({y:g15,label:"sub-0.15  "+sig(g15,4),color:"#ffb454"});
  // c2-era chart annotations: engage-event markers + the warm-start origin (lineage source).
  const RE=META.run_events||{};
  const evs=(RE.markers||[]).filter(m=>m.epoch!=null);
  let origin=null;
  const wsrc=RE.warm_start;
  if(wsrc&&wsrc.start_epoch!=null){
    let srcLbl="";
    const rf=wsrc.source||null;
    if(rf&&String(rf).includes("/")){
      const parent=String(rf).split("/").slice(-2,-1)[0]||"";
      srcLbl=" ← "+(parent.replace(/_\d{8}T\d{6}Z$/,"")||"ancestor");
    }else if(rf){srcLbl=" ← ancestor ckpt";}
    origin={epoch:wsrc.start_epoch,
            label:"warm start ep"+wsrc.start_epoch+(wsrc.mode?" ("+wsrc.mode+")":"")+srcLbl};
  }
  drawPanel($("c_dseg"),"d_seg",{title:"",
    sub:"",color:"#5ab0ff",log:true,fmt:fSg,
    star:star,starGlow:_celebrating,
    hlines:segH,events:evs,origin:origin});
  // descent panel meta line (right of the header) — latest / best / goal provenance
  const last=TRAJ.length?TRAJ[TRAJ.length-1]:null;
  const dm=$("lv_dseg_meta");
  if(dm){
    let s=[];
    if(last&&last.d_seg!=null)s.push("latest "+sig(last.d_seg,4)+" @ ep"+last.epoch);
    if(bestVal!=null)s.push("best "+sig(bestVal,4)+" @ ep"+bestEpoch);
    const gsrc=goalSrcTag(META.goal_src&&META.goal_src.dseg);
    if(g!=null)s.push("goal "+sig(g,4)+(gsrc?" ("+gsrc+")":""));
    else s.push("goal: "+((META.goal_src&&META.goal_src.dseg)||"pending source"));
    // #343: the verdict cadence — makes clear the sparse panels are WAITING for the next
    // eval tick (not frozen). Shown here on the primary verdict panel + in the header status.
    if(!last){const w=waitLbl(""); if(w)s.unshift(w);}
    if(origin)s.push("warm-start @ep"+wsrc.start_epoch);
    const nvh=nextVerdictHint(); if(nvh)s.push(nvh);
    dm.textContent=s.join(" · ")||"log y · auto-fit x";
  }
  updateAria();
}
function updateAria(){
  const last=TRAJ.length?TRAJ[TRAJ.length-1]:null;
  const set=(id,txt)=>{const el=$(id);if(el)el.setAttribute("aria-label",txt);};
  if(!last){set("c_dseg","chart, no data yet");return;}
  const bestTxt=(bestVal!=null)?(", best "+sig(bestVal,5)+" at epoch "+bestEpoch):"";
  set("c_dseg","d_seg over epochs, latest "+sig(last.d_seg,5)+" at epoch "+last.epoch+bestTxt);
}
let _drawQueued=false;
function scheduleDraw(){
  // Browsers PAUSE requestAnimationFrame in backgrounded/occluded windows, so a WS update
  // arriving then would leave _drawQueued stuck true and every later update short-circuit —
  // charts froze until a mouse hover happened to re-enter scheduleDraw after the stale rAF
  // finally fired (the "graphs only update on mouse event" bug, operator 2026-07-07).
  // Dual-path: rAF for smoothness when visible, a timeout fallback that ALWAYS fires;
  // whichever runs first does the draw, the other no-ops on the cleared flag.
  if(_drawQueued)return; _drawQueued=true;
  const run=()=>{
    if(!_drawQueued)return; _drawQueued=false;
    if(!$("tab-live").classList.contains("hide")) drawAll();
  };
  requestAnimationFrame(run);
  setTimeout(run,300);
}
document.addEventListener("visibilitychange",()=>{
  // returning to the page: drop any stale queue state and repaint with the latest data.
  if(!document.hidden){_drawQueued=false;scheduleDraw();}
});
window.addEventListener("pageshow",()=>{_drawQueued=false;scheduleDraw();});

// derived per-run stage map (conditional rendering, DSL read-back). Entries:
// fixed {name,start} | event-gated {name,trigger,status:pending|fired,fired_epoch,cap}.
// A pending event stage's cap is a HARD CEILING (trainer fires at cap unconditionally),
// so it is provably active at ep>=cap even before fired evidence lands.
const STAGE_RANK={CE:0,tau:1,l7:2,Muon:3};
function stageMap(){return (META.stage_map&&META.stage_map.length)?META.stage_map:null;}
function stageBoundary(s){return (s.start!=null)?s.start:((s.mode==="event"&&s.status==="pending")?s.cap:null);}
function stageWord(ep){if(ep==null)return "starting";
  const map=stageMap();
  if(map){
    let best=null;
    map.forEach(s=>{const b=stageBoundary(s);
      if(b==null||ep<b)return;
      if(best==null||(STAGE_RANK[s.name]||0)>=(STAGE_RANK[best.name]||0))best=s;});
    return best?best.name:(map[0]?map[0].name:"CE");
  }
  // legacy fallback (old server / no map): historical constant-boundary labeling
  const tau=(META.tau!=null)?META.tau:BOOT.tau,l7=(META.l7!=null)?META.l7:BOOT.l7,mu=META.muon_start;
  if(tau==null)return "?";
  if(ep<tau)return "CE";if(l7!=null&&ep<l7)return "tau";
  if(mu!=null&&ep>=mu)return "Muon";
  if(l7==null)return (mu!=null)?"tau":"tau/Muon";
  return (mu!=null)?"l7":"l7/Muon";}

// ================= LIVE INSTRUMENT PANELS (rebuilt 2026-07-09) =================
// Every panel reads the newest verdict point (`last`) + META.sensors (the non-verdict
// jacobian_basin / loss_terms stages). Numbers fixed-precision, tabular. Advisory only.

// comma10k CANONICAL class order (NEVER luma-sorted; per CLAUDE.md class-index note).
const CLASS_NAMES=["Road","Lane","Undrivable","Movable","MyCar"];
// fixed per-metric precision helpers
function fDs(v){return sig(v,4);}     // d_seg-family: ~4 sig figs (0.03662)
function fContrib(v){return sig(v,4);}
function pct(v){return (v==null||!isFinite(v))?"—":(v*100).toFixed(1)+"%";}
function esc(s){return String(s).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");}

// 1 · MASTHEAD — the live equation, each term's value + contribution to S, dominant term
// visually obvious via its share bar. S is the run's OWN measured composite (advisory).
function renderMasthead(last){
  const svalEl=$("lv_S"), srefEl=$("lv_Sref"), termsEl=$("lv_terms");
  if(!svalEl||!termsEl)return;
  const N=(META.archive_norm_bytes||37545489);
  if(!last||last.d_seg==null||last.d_pose==null||last.blob_bytes==null||
     !isFinite(last.d_seg)||!isFinite(last.d_pose)||!isFinite(last.blob_bytes)){
    svalEl.textContent="—"; if(srefEl)srefEl.innerHTML="awaiting first verdict"; termsEl.innerHTML=""; return;
  }
  const ds=last.d_seg, dp=last.d_pose, by=last.blob_bytes;
  const t1=100*ds, t2=Math.sqrt(10*dp), t3=25*by/N, S=t1+t2+t3;
  // POSE-DEFERRED phase: w_pose>0 but pose descent has NOT engaged (pose held out until pose-finish).
  // The √(10·d_pose) term is real-but-expected-high-by-design, so folding it into the headline makes a
  // healthy seg-phase run read as a ~60x regression. Headline = the SEG-PHASE S (seg+rate, what is
  // actually being optimized now); the full composite stays visible in the sub-line + the term cells.
  const deferred=!!META.pose_deferred, Sseg=t1+t3;
  svalEl.textContent=sig(deferred?Sseg:S,4);
  // reference vs the frontier pointer (advisory delta)
  const pv=ptrVal();
  if(srefEl){
    let h="";
    if(deferred){
      h="seg-phase &middot; pose deferred &rarr; pose-finish";
      if(pv!=null){const dseg=Sseg-pv; h+=" &middot; Δ "+(dseg>=0?"+":"")+sig(dseg,3)+" vs pointer <b>"+sig(pv,5)+"</b>";}
      h+=" &middot; <span class='lv-adv'>full S "+sig(S,3)+" incl. deferred pose</span>";
    }else{
      if(pv!=null){const d=S-pv; h="pointer <b>"+sig(pv,5)+"</b> &middot; Δ "+(d>=0?"+":"")+sig(d,3)+" (advisory)";}
      else h="pointer unavailable";
      if(META.pose_blind)h+=" &middot; pose UNHELD (w_pose=0)";
    }
    srefEl.innerHTML=h;
  }
  // three term cells, share of S as a bar; dominant term flagged
  const terms=[
    {k:"100&middot;d_seg",in:"d_seg = "+fDs(ds),con:t1,c:"var(--lv-seg)"},
    {k:"&radic;(10&middot;d_pose)",in:"d_pose = "+sig(dp,4)+(deferred?" &middot; deferred":""),con:t2,c:"var(--lv-pose)"},
    {k:"25&middot;bytes / N",in:fmtInt(by)+" B",con:t3,c:"var(--lv-byte)"}
  ];
  const maxCon=Math.max(t1,t2,t3,1e-12);
  let html="";
  terms.forEach(tm=>{
    const share=S>0?tm.con/S:0, w=Math.max(1,Math.round(tm.con/maxCon*100));
    html+="<div class='lv-term'>"+
      "<div class='tt'><span class='ttk'><span class='swatch' style='background:"+tm.c+"'></span>"+tm.k+"</span>"+
        "<span class='tin'>"+tm.in+"</span></div>"+
      "<div class='tcon'>"+fContrib(tm.con)+"</div>"+
      "<div class='lv-bar'><i style='width:"+w+"%;background:"+tm.c+"'></i></div>"+
      "<div class='tshare'><span>contribution</span><span>"+pct(share)+" of S</span></div>"+
    "</div>";
  });
  termsEl.innerHTML=html;
}

// 3 · PER-CLASS BREAKDOWN — d_seg_by_class (blue) + flip_share_by_class (amber), aligned.
function renderClasses(last){
  const box=$("lv_classes"); if(!box)return;
  const ds=last&&last.d_seg_by_class, fl=last&&last.flip_share_by_class;
  if(!Array.isArray(ds)||ds.length<5){box.innerHTML="<div class='lv-none'>"+esc(waitLbl("no per-class verdict yet"))+"</div>";return;}
  const dmax=Math.max.apply(null,ds.map(v=>isFinite(v)?v:0),0.0001)||0.0001;
  const fmax=Math.max.apply(null,(fl||[]).map(v=>isFinite(v)?v:0),0.0001)||0.0001;
  let html="<div class='lv-clab'><span><i style='background:var(--lv-seg)'></i>d_seg</span>"+
    "<span><i style='background:var(--lv-pose)'></i>flip share</span></div>";
  for(let i=0;i<5;i++){
    const dv=ds[i], fv=(fl&&fl[i]!=null)?fl[i]:null;
    const dw=Math.max(2,Math.round((dv/dmax)*100)), fw=(fv!=null)?Math.max(2,Math.round((fv/fmax)*100)):0;
    html+="<div class='lv-crow'>"+
      "<div class='lv-cname'><span class='ci'>"+i+"</span> "+esc(CLASS_NAMES[i]||("c"+i))+"</div>"+
      "<div class='lv-cbars'>"+
        "<div class='lv-cbar'><i style='width:"+dw+"%;background:var(--lv-seg)'></i></div>"+
        "<div class='lv-cbar'><i style='width:"+fw+"%;background:var(--lv-pose)'></i></div>"+
      "</div>"+
      "<div class='lv-cval'>"+sig(dv,3)+"<br><span class='cf'>"+(fv!=null?pct(fv):"—")+"</span></div>"+
    "</div>";
  }
  box.innerHTML=html;
}

// 4 · POSE-DESCENT READINESS — honest current pose state + an unselected R1 reference
// artifact + the jacobian_basin readiness sensor. Pose is pose-BLIND until the finishing
// stage by design; the current vehicle has no compatibility-checked R1 payload selector.
function renderPose(){
  const box=$("lv_pose"), meta=$("lv_pose_meta");
  if(!box)return;
  const PR=META.pose_readiness||{};
  const jb=(META.sensors&&META.sensors.jacobian_basin)||null;
  const cell=(l,v,cls)=>"<div class='lv-pcell'><span class='pl'>"+l+"</span><span class='pvv "+(cls||"")+"'>"+v+"</span></div>";
  let html="";

  // (a) R1 reference artifact. These values are advisory context, never current config claims.
  if(PR.r1_reference){
    const b=PR.r1_reference, a=b.advisory_artifact||{};
    if(b.ok){
      html+="<div class='lv-r1'><div class='r1h'><span class='r1t'>R1 reference · not selected</span>"+
        "<span class='r1tag'>"+esc(b.label||"full-n600 byte-closed macOS-CPU advisory")+"</span></div>"+
        "<div class='lv-r1grid'>"+
          "<div class='rc'><span class='rk'>advisory d_pose</span><span class='rv'>"+sig(a.d_pose,4)+"</span></div>"+
          "<div class='rc'><span class='rk'>advisory √(10·d_pose)</span><span class='rv'>"+
            (a.pose_term!=null?sig(a.pose_term,4):"—")+"</span></div>"+
          "<div class='rc'><span class='rk'>reference ξ / dxi</span><span class='rv'>"+
            (a.counted_pose_bytes!=null?fmtInt(a.counted_pose_bytes)+" B":"—")+"</span></div>"+
        "</div><div class='lv-r1src'>UNSELECTED REFERENCE ONLY · src "+esc(b.source||"?")+"</div></div>";
    }else{
      html+="<div class='lv-r1'><div class='r1h'><span class='r1t'>R1 reference · not selected</span></div>"+
        "<div class='lv-r1src'>reference unreadable ("+esc(b.reason||"?")+") — src "+esc(b.source||"?")+"</div></div>";
    }
  }
  // (b) fallback contract state (detector mode / state / decision tree).
  if(PR.contract){
    const c=PR.contract;
    const stCls=(c.detector_state==="fired")?"ok":(c.degenerate?"bd":(c.detector_state==="armed"?"wn":""));
    const stLbl=c.degenerate?"DEGENERATE":String(c.detector_state||"pending");
    const cap=(c.detector_cap!=null)?(" · fail-safe cap ep"+fmtInt(c.detector_cap)):"";
    const selected=(c.payload_selected===true)?"yes":"no";
    html+="<div class='lv-contract'><span class='ck'>pose-finish</span> detector <b>"+esc(c.detector_mode||"—")+"</b>"+
      cap+" · state <span class='"+stCls+"'>"+esc(stLbl)+"</span>"+
      (c.detector_at_epoch!=null?("@ep"+c.detector_at_epoch):"")+
      " · pose <b>"+esc(c.pose_state||"pose_finish_pending")+"</b>"+
      " · payload selected: <b>"+selected+"</b>"+
      "<div class='dt'>"+esc(c.decision_tree||"")+"</div>"+
      (c.detector_state==="pending"?("<div style='margin-top:3px'>"+esc(c.pending_note||"")+"</div>"):"")+
      "</div>";
  }

  // (c) jacobian_basin readiness sensor (the conditioning for the pose descent to converge).
  if(meta)meta.textContent=(jb?("ep"+(jb.epoch!=null?jb.epoch:"?")+" · k="+(jb.k_pairs!=null?jb.k_pairs:"?")):"jacobian basin");
  if(!jb){
    html+="<div class='lv-pnote'>jacobian basin: <b>no probe yet</b> — "+
      "joint pose finish is terminal; readiness rows appear once conditioning begins.</div>";
    box.innerHTML=html||"<div class='lv-none'>"+esc(waitLbl("no basin probe yet"))+"</div>";
    return;
  }
  const sm=jb.median_sigma_min, cond=jb.median_cond, bf=jb.basin_frac, fired=jb.would_have_fired;
  const floor=jb.sigma_floor;
  const smCls=(sm!=null&&floor!=null)?(sm>floor?"ok":"wn"):"";
  const condCls=(cond!=null)?(cond<1e5?"ok":(cond<1e6?"wn":"")):"";
  html+=cell("median σ_min",(sm!=null?sig(sm,3):"—"),smCls);
  html+=cell("median cond",(cond!=null?sig(cond,3):"—"),condCls);
  html+=cell("basin frac",(bf!=null?pct(bf):"—"),(bf!=null&&bf>=0.5?"ok":"wn"));
  html+=cell("would fire",(fired==null?"—":(fired?"yes":"no")),(fired?"ok":""));
  html+="<div class='lv-pnote'>pose is <b>held out until the finishing stage</b> by design — a rising "+
    "d_pose before then is expected; the conditioning above is the readiness for the pose descent to converge.</div>";
  box.innerHTML=html;
}

// 5 · TRAINING HEALTH — loss_terms guard scalars + nonzero energy-term split (confound signals).
function renderHealth(){
  const box=$("lv_health"), meta=$("lv_health_meta");
  if(!box)return;
  const lt=(META.sensors&&META.sensors.loss_terms)||null;
  if(!lt){box.innerHTML="<div class='lv-none'>"+esc(waitLbl("no loss row yet"))+"</div>";return;}
  if(meta)meta.textContent="ep"+(lt.ep!=null?lt.ep:"?")+" · batch "+(lt.accum_batch!=null?lt.accum_batch:"?");
  const af=lt.accepted_frac, gn=lt.gnorm, sk=lt.spike_skipped, ws=lt.weights_stepped;
  const hb=lt.hosc_beta, sx=lt.softmax_temp, tot=lt.total;
  const afCls=(af!=null)?(af>=0.9?"ok":(af>=0.5?"wn":"bd")):"";
  const gnCls=(gn!=null)?(gn<50?"ok":(gn<100?"wn":"bd")):"";
  const skCls=(sk?"wn":"ok");
  const scal=(k,v,cls)=>"<span class='lv-hs'><span class='hk'>"+k+"</span><span class='hv "+(cls||"")+"'>"+v+"</span></span>";
  let strip="<div class='lv-hscal'>";
  strip+=scal("total",(tot!=null?sig(tot,4):"—"));
  strip+=scal("gnorm",(gn!=null?sig(gn,3):"—"),gnCls);
  strip+=scal("accepted",(af!=null?pct(af):"—"),afCls);
  strip+=scal("spike skip",(sk!=null?String(sk):"—"),skCls);
  strip+=scal("stepped",(ws==null?"—":(ws?"yes":"NO")),(ws?"ok":"bd"));
  strip+=scal("hosc β",(hb!=null?sig(hb,3):"—"));
  strip+=scal("softmax τ",(sx!=null?sig(sx,3):"—"));
  strip+="</div>";
  // per-group grad-clip activation (c2: --per-group-grad-clip; stage grad_clip_activation,
  // one aggregation row per epoch) — global frac clipped + the hottest group.
  const RE=META.run_events||{};
  if(RE.clip&&RE.clip.global){
    const g=RE.clip.global, pg=RE.clip.per_group||{};
    let hot=null;
    Object.keys(pg).forEach(k=>{const f=pg[k]&&pg[k].frac_clipped;
      if(f!=null&&(hot==null||f>hot[1]))hot=[k,f];});
    const gf=g.frac_clipped, gCls=(gf!=null)?(gf<0.2?"ok":(gf<0.8?"wn":"bd")):"";
    let cstrip="<div class='lv-hscal'>";
    cstrip+=scal("clip@"+sig(RE.clip.threshold,2),(gf!=null?pct(gf):"—"),gCls);
    if(hot)cstrip+=scal("hottest group",esc(hot[0])+" "+pct(hot[1]),(hot[1]<0.8?"":"wn"));
    cstrip+=scal("ep",(RE.clip.epoch!=null?RE.clip.epoch:"—"));
    cstrip+="</div>";
    strip+=cstrip;
  }
  // rate rolling-average soft-signal (producer landed; renders the moment the trainer
  // emits stage rate_rolling rows — graduated WITHIN / DRIFTING_UP / SUSTAINED_GROWTH).
  if(RE.rate){
    const r=RE.rate, sgn=r.signal||r.state||"—";
    const rCls=(sgn==="WITHIN")?"ok":(sgn==="DRIFTING_UP"?"wn":(sgn==="SUSTAINED_GROWTH"?"bd":""));
    let rstrip="<div class='lv-hscal'>";
    rstrip+=scal("rate signal",esc(String(sgn)),rCls);
    if(r.rolling_mean!=null)rstrip+=scal("rolling mean",sig(r.rolling_mean,4));
    if(r.ep!=null||r.epoch!=null)rstrip+=scal("ep",(r.epoch!=null?r.epoch:r.ep));
    rstrip+="</div>";
    strip+=rstrip;
  }
  // nonzero energy terms, largest first
  const terms=lt.terms||{};
  const rows=Object.keys(terms).map(k=>[k,terms[k]]).filter(r=>isFinite(r[1])&&Math.abs(r[1])>1e-9)
    .sort((a,b)=>Math.abs(b[1])-Math.abs(a[1]));
  let bars="";
  if(rows.length){
    const mx=Math.max.apply(null,rows.map(r=>Math.abs(r[1])))||1;
    rows.forEach(r=>{
      const w=Math.max(2,Math.round(Math.abs(r[1])/mx*100));
      bars+="<div class='lv-hterm'><span class='hn'>"+esc(r[0])+"</span>"+
        "<span class='hbar'><i style='width:"+w+"%'></i></span>"+
        "<span class='hpv'>"+sig(r[1],3)+"</span></div>";
    });
  }else bars="<div class='lv-none'>all energy terms zero</div>";
  box.innerHTML=strip+bars;
}

// 6 · SYSTEM — resident + MLX active/peak vs available (one gauge each).
function renderSys(last){
  const box=$("lv_sys"); if(!box)return;
  const mp=(META.sensors&&META.sensors.loss_terms)||null; // loss_terms carries no mem; use verdict + mem_probe via sensors? verdict has it.
  // memory rows live on the verdict point (rss/mlx_active/mlx_peak/sys_avail)
  const rss=last&&last.rss_gib, act=last&&last.mlx_active_gib, pk=last&&last.mlx_peak_gib,
        av=last&&last.sys_avail_gib, ca=last&&last.mlx_cache_gib;
  if(rss==null&&act==null){box.innerHTML="<div class='lv-none'>"+esc(waitLbl("no memory row yet"))+"</div>";return;}
  // gauge domain: available + peak headroom so bars are comparable
  const dom=Math.max(av||0, pk||0, act||0, rss||0, 1);
  const gauge=(k,v,vpeak,extra)=>{
    if(v==null)return "";
    const w=Math.max(1,Math.round(v/dom*100));
    const pkmark=(vpeak!=null&&vpeak>0)?"<span class='peak' style='left:"+Math.min(100,vpeak/dom*100)+"%'></span>":"";
    return "<div class='lv-gauge'><span class='gk'>"+k+"</span>"+
      "<span class='lv-gtrack'><i style='width:"+w+"%'></i>"+pkmark+"</span>"+
      "<span class='lv-gv'><b>"+sig(v,4)+"</b> GiB"+(extra||"")+"</span></div>";
  };
  let html="";
  html+=gauge("resident",rss,null,"");
  html+=gauge("MLX active",act,pk," · peak "+(pk!=null?sig(pk,4):"—"));
  if(ca!=null)html+=gauge("MLX cache",ca,null,"");
  html+="<div class='lv-gauge'><span class='gk'>available</span>"+
    "<span class='lv-gtrack'></span><span class='lv-gv'><b>"+(av!=null?sig(av,4):"—")+"</b> GiB free</span></div>";
  box.innerHTML=html;
}

// 0 · CHAIN-STATE STRIP — bench -> receipt -> launch -> run -> byte-close (c2 era).
const CHAIN_LABELS={bench:"bench",receipt:"receipt",launch:"launch",run:"run",byteclose:"byte-close"};
function renderChain(){
  const box=$("lv_chain"); if(!box)return;
  const cs=META.chain_state;
  if(!cs||!Array.isArray(cs.steps)||!cs.steps.length){box.classList.add("hide");return;}
  box.classList.remove("hide");
  let html="<div class='lv-chainhead'><span>launch pipeline</span><b title='"+esc(cs.dir||"")+"'>"+
    esc(cs.config_family||cs.dir||"?")+"</b></div>";
  cs.steps.forEach(s=>{
    html+="<div class='lv-cstep "+esc(s.state||"pending")+"'>"+
      "<span class='cs-k'><span class='cs-dot'></span>"+esc(CHAIN_LABELS[s.id]||s.id)+"</span>"+
      "<span class='cs-d' title='"+esc(s.detail||"")+"'>"+esc(s.detail||"—")+"</span></div>";
  });
  box.innerHTML=html;
}

// 0b · CONFOUND ALARMS — scoped honestly (operator round-2): full-red ONLY for alarms on
// the actively-watched LIVE run; alarms from a stale/completed/superseded run render as a
// muted collapsed summary with run + epoch + date context, so a historical cold-start
// transient (gnorm_hijack @ep1 of a finished side run) never reads as a live emergency.
function renderAlarms(){
  const box=$("lv_alarms"); if(!box)return;
  const RE=META.run_events||{};
  const al=RE.alarms||[];
  if(!al.length){box.classList.add("hide");box.classList.remove("histmode");return;}
  box.classList.remove("hide");
  // an alarm is LIVE only when the watched run is live AND the alarm came from the live
  // arm's own log — a stitched ancestor's transient stays historical under a live run.
  const runLive=(RE.historical!==true)&&(LIVE.kind==="live");
  const liveAl=runLive?al.filter(a=>a.live===true):[];
  const histAl=al.filter(a=>!(runLive&&a.live===true));
  const badge=(a,h)=>"<span class='lv-alarm"+(h?" hist":"")+"'>"+esc(a.kind||"?")+
    (a.term?(" ["+esc(a.term)+"]"):"")+
    (a.epoch!=null?(" @ep"+a.epoch):"")+
    (a.src?(" · "+esc(a.src)):"")+
    (a.ts?(" · "+esc(String(a.ts).slice(0,10))):"")+"</span>";
  if(liveAl.length){
    box.classList.remove("histmode");
    let html="<span class='al-k'>confound alarms · "+liveAl.length+"</span>";
    liveAl.slice(-8).forEach(a=>{html+=badge(a,false);});
    if(histAl.length)html+="<span class='lv-alarm hist'>+"+histAl.length+" historical (ancestor arms)</span>";
    box.innerHTML=html;
  }else{
    box.classList.add("histmode");
    box.innerHTML="<details class='lv-alhist'><summary>confound alarms · "+histAl.length+
      " · <b>historical</b> — source run not live / ancestor arm</summary><div class='alwrap'>"+
      histAl.slice(-8).map(a=>badge(a,true)).join("")+"</div></details>";
  }
}

// 7 · CURRICULUM POSITION — timeline of stage bands with a marker at the current epoch.
function renderSchedule(last){
  const box=$("lv_sched"), meta=$("lv_sched_meta");
  if(!box)return;
  const ep=last&&last.epoch!=null?last.epoch:(LIVE&&LIVE.last_epoch!=null?LIVE.last_epoch:null);
  // total epochs from the run's own schedule (never fabricated); fall back to a
  // data-driven max so a cap-unknown run still renders a sane track.
  const cfgSched=(META.config&&META.config.schedule)||META.schedule||{};
  let total=cfgSched.epochs!=null?cfgSched.epochs:null;
  // build ordered boundaries from the derived stage map (preferred) or BOOT tau/l7 + muon.
  const map=stageMap();
  let bounds=[];
  if(map){
    map.forEach(s=>{const b=stageBoundary(s);
      bounds.push({name:s.name,at:(b!=null?b:0),pending:(s.mode==="event"&&s.status==="pending")});});
    bounds.sort((a,b)=>a.at-b.at);
  }else{
    const tau=(META.tau!=null)?META.tau:BOOT.tau, l7=(META.l7!=null)?META.l7:BOOT.l7, mu=META.muon_start;
    bounds.push({name:"CE",at:0});
    if(tau!=null)bounds.push({name:"tau",at:tau});
    if(l7!=null)bounds.push({name:"l7",at:l7});
    if(mu!=null)bounds.push({name:"Muon",at:mu});
  }
  if(!bounds.length){box.innerHTML="<div class='lv-none'>schedule unresolved</div>";return;}
  if(total==null){const lastB=bounds[bounds.length-1].at; total=Math.max(lastB*1.25||1, (ep||0)*1.1, lastB+1);}
  if(!(total>0))total=1;
  const _derived=(META.curriculum_panel&&META.curriculum_panel.ok)?" · derived":"";
  if(meta)meta.textContent=(ep!=null?"ep "+ep:"—")+" / "+(cfgSched.epochs!=null?fmtInt(cfgSched.epochs):"cap "+fmtInt(total)+"?")+_derived;
  // segment widths from consecutive boundaries
  const BANDCOL={ce:"#1f3b5f",tau:"#3a2a5f",l7:"#5f3320",muon:"#1f4f43"};
  let segs="";
  for(let i=0;i<bounds.length;i++){
    const a=bounds[i].at, b=(i+1<bounds.length)?bounds[i+1].at:total;
    const w=Math.max(0,(b-a)/total*100);
    if(w<=0)continue;
    const col=BANDCOL[bounds[i].name.toLowerCase()]||"#33394a";
    segs+="<div class='lv-seg' style='width:"+w+"%;background:"+col+"'>"+
      "<span class='sn'>"+esc(bounds[i].name)+(bounds[i].pending?" ⌁":"")+"</span></div>";
  }
  const markPct=(ep!=null)?Math.min(100,Math.max(0,ep/total*100)):null;
  const marker=(markPct!=null)?"<div class='lv-marker' style='left:"+markPct+"%'></div>":"";
  let html="<div class='lv-track'>"+segs+marker+"</div>"+
    "<div class='lv-sticks'><span>0</span><span>"+fmtInt(total)+"</span></div>";
  // DERIVED curriculum (operator 2026-07-10): transitions as EVENTS (epoch = fail-safe
  // cap, never the trigger) + the tau path as ONE continuous anneal + the event-gated
  // mechanism swim-lanes + the provenance line — all from META.curriculum_panel (the
  // schedule read-back + parsed flags + emitted evidence; never a hardcoded skeleton).
  const CP=META.curriculum_panel;
  if(CP&&CP.ok){
    html+=renderCurriculumDerived(CP);
  }
  box.innerHTML=html;
}

// status pill for an event/lane state (fired@ep / armed@ep / pending).
function statePill(status,atEpoch){
  const s=String(status||"pending");
  const at=(atEpoch!=null)?("@ep"+atEpoch):"";
  const lbl=(s==="fired")?("fired"+at):(s==="armed")?("armed"+at):"pending";
  return "<span class='lv-pill "+esc(s)+"'>"+esc(lbl)+"</span>";
}

// The DERIVED curriculum sections: tau anneal ramp, transition events, mechanism
// swim-lanes, provenance. Reads ONLY the server-built model (no hardcoded epochs).
function renderCurriculumDerived(CP){
  let h="";
  // (a) tau path as ONE continuous geometric anneal — CE = the tau=1 limit.
  const ta=CP.tau_anneal||{};
  if(ta.temp_start!=null||ta.shape){
    const t0=(ta.temp_start!=null)?sig(ta.temp_start,3):"1",
          t1=(ta.temp_end!=null)?sig(ta.temp_end,3):"—";
    h+="<div class='lv-tau'><span class='tl'>CE · τ="+esc(t0)+"</span>"+
       "<span class='tl r'>τ="+esc(t1)+"</span></div>";
    const capEp=ta.span_end_epoch;
    h+="<div class='lv-taucap'>"+esc(ta.shape||"anneal")+" τ anneal · one continuous ramp"+
       (capEp!=null?(" · CE→finish span to ep"+fmtInt(capEp)):"")+
       " · "+esc(ta.ce_limit_note||"")+"</div>";
  }
  // (b) stage transitions as EVENTS (trigger primary; epoch = fail-safe cap).
  const st=CP.stages||[];
  if(st.length){
    h+="<div class='lv-events'>";
    st.forEach(s=>{
      let trig;
      if(s.name==="CE"){trig="anneal origin (τ=1 limit)";}
      else if(s.trigger){trig=esc(s.trigger);}
      else{trig="scheduled";}
      let cap="";
      if(s.mode==="event"&&s.cap!=null)cap=" <span class='cap'>· fail-safe cap ep"+fmtInt(s.cap)+"</span>";
      else if(s.mode==="fixed"&&s.start!=null&&s.name!=="CE")cap=" <span class='cap'>· at ep"+fmtInt(s.start)+"</span>";
      const status=(s.status==="fired")?"fired":(s.mode==="event"?"pending":"scheduled");
      h+="<div class='lv-evrow'><span class='en'>"+esc(s.name)+"</span>"+
         "<span class='et'>"+trig+cap+"</span>"+
         statePill(status,s.fired_epoch)+"</div>";
    });
    h+="</div>";
  }
  // (c) event-gated mechanism swim-lanes.
  const lanes=CP.lanes||[];
  if(lanes.length){
    h+="<div class='lv-lanes'>";
    lanes.forEach(l=>{
      const trig=(l.trigger!=null)?esc(String(l.trigger)):"—";
      let cap="";
      if(l.cap!=null)cap=" <span class='cap'>· "+esc(l.cap_kind||"cap")+" ep"+fmtInt(l.cap)+"</span>";
      else if(l.cap_kind)cap=" <span class='cap'>· "+esc(l.cap_kind)+"</span>";
      h+="<div class='lv-lane'><span class='ln'>"+esc(l.name)+"</span>"+
         "<span class='lt'>@"+trig+cap+"</span>"+
         statePill(l.status,l.at_epoch)+"</div>";
    });
    h+="</div>";
  }
  // (d) provenance line — answers the cargo-cult question on the panel itself.
  if(CP.provenance)h+="<div class='lv-prov'>"+esc(CP.provenance)+"</div>";
  return h;
}

// ---------- RUN-IDENTITY row (name + purpose chip + scope chip; CONDITIONAL) ----------
// Source: META.run_identity (server-derived: launch.sh declared header OR labelled
// heuristic). Absent -> the row is hidden. Evidence rides the chip's title tooltip.
function renderRunIdentity(){
  const box=$("runid"); if(!box)return;
  const R=META.run_identity;
  if(!R||!R.name){box.classList.add("hide");return;}
  box.classList.remove("hide");
  // title attributes are DOUBLE-quoted: escHtml escapes `"` but not `'`, and the
  // evidence strings legitimately contain apostrophes (e.g. family name 'mod32cap').
  let h="<span class='rname'>"+escHtml(R.name)+"</span>";
  if(R.purpose&&R.purpose.label){
    // TERSE identity (operator 2026-07-09 "no wall of prose"): show the first clause only
    // (up to the first ':' or '(' or '.'), capped; the FULL declared text + evidence lives
    // in the hover title so no substance is lost — a scanned field, not a paragraph.
    const full=R.purpose.label;
    let terse=full.split(/[:(]/)[0].trim();
    if(terse.length>72)terse=terse.slice(0,70).trim()+"…";
    else if(terse.length<full.trim().length)terse=terse+"…";
    const tip=escHtml(full+((R.purpose.evidence&&R.purpose.evidence.length)?"  —  "+R.purpose.evidence.join(" · "):""));
    h+="<span class='ridchip' title=\""+tip+"\">"+escHtml(terse)+
      " <span class='prov'>("+escHtml(R.purpose.provenance||"?")+")</span></span>";
  }
  if(R.scope&&R.scope.label){
    h+="<span class='ridchip' title=\""+escHtml(R.scope.evidence||"")+"\">"+
      escHtml(R.scope.label)+"</span>";
  }
  box.innerHTML=h;
}

// ---------- costate controller panel (SENSE/DECIDE; CONDITIONAL; observability ONLY) ----------
// Source: the run's costate_shadow.jsonl (score-neutral shadow observer), via the schema-driven
// introspection layer (META.introspect.controller — rich: λ traces + duty + axis EV) with a
// fallback to the legacy summary (META.costate). Plus the confound-immune LIVENESS strip
// (META.introspect.liveness). Absent -> the panel is hidden. The dashboard NEVER actuates.
function _lamStatusCls(s){const t=String(s||"").toLowerCase();
  return t.indexOf("unident")>=0?"unidentifiable":(t.indexOf("analytic")>=0?"analytic":"identified");}
function renderCostate(){
  const box=$("costate"), body=$("cs_body"); if(!box||!body)return;
  const I=META.introspect||{};
  const C=(I.controller&&I.controller.ok)?I.controller:null;
  const legacy=(!C&&META.costate&&META.costate.ok)?META.costate:null;
  const campaign=(META.ddm_campaign&&META.ddm_campaign.ok)?META.ddm_campaign:null;
  const L=I.liveness||null;
  if(!C&&!legacy&&!campaign&&!L){box.classList.add("hide");return;}
  box.classList.remove("hide");
  const src=C||legacy||campaign||{};
  // ── header summary line ──
  const bits=[];
  // classification: the introspect controller ships a RAW diagnostic dict-repr here
  // (a prose wall); show only a clean short token — prefer the legacy summary's token,
  // else extract the nested CLASSIFICATION, else omit. Raw diagnostic -> hover title.
  const _clsClean=(function(){
    const c=src.classification;
    if(typeof c==="string"&&c.length<=40&&c.indexOf("{")<0)return c;
    const lg=(META.costate&&META.costate.classification);
    if(typeof lg==="string"&&lg.length<=40&&lg.indexOf("{")<0)return lg;
    if(typeof c==="string"){const m=c.match(/'CLASSIFICATION':\s*'([A-Z_]+)'/);if(m)return m[1];}
    return null;
  })();
  if(_clsClean){
    const rawTip=(typeof src.classification==="string"&&src.classification.indexOf("{")>=0)
      ?" title=\""+escHtml(src.classification.slice(0,400))+"\"":"";
    bits.push("<span"+rawTip+">class <b>"+escHtml(_clsClean)+"</b></span>");
  }
  if(src.rec&&src.rec.action){
    let r="rec <b>"+escHtml(src.rec.action)+"</b>";
    if(src.rec.predicted_dS!=null&&isFinite(src.rec.predicted_dS))
      r+=" &Delta;S "+(src.rec.predicted_dS>0?"+":"")+sig(src.rec.predicted_dS,4)+
        (src.rec.horizon_epochs!=null?"/"+src.rec.horizon_epochs+"ep":"");
    bits.push(r);
  }
  if(src.epoch!=null)bits.push("row ep"+src.epoch+(src.age_s!=null?" &middot; "+fmtAge(src.age_s)+" old":""));
  if(src.n_verdicts!=null)bits.push("<b>"+src.n_verdicts+"</b> verdicts sensed");
  if(campaign){
    bits.push("DDM campaign <b>"+escHtml(campaign.status||"?")+"</b>");
    bits.push("<b>"+campaign.verdict_count+"</b> realized verdicts");
    const pr=campaign.plateau_route||{};
    bits.push("plateau "+escHtml(pr.status||"?")+(pr.fork_id?" &rarr; "+escHtml(pr.fork_id):""));
  }
  let h="<div class='csline'>"+(bits.join(" &middot; ")||"no shadow rows yet")+"</div>";
  // ── two-column grid: λ costate traces  |  DECIDE (duty queue + axis EV) ──
  if(C){
    const cells=[];
    if(C.costates&&C.costates.length){
      let rows="";
      C.costates.forEach(l=>{
        const v=(l.value!=null&&isFinite(l.value))?sig(l.value,4):"&mdash;";
        rows+="<div class='lamrow'><span class='lnm' title=\""+escHtml((l.method||"")+(l.units?" · "+l.units:""))+"\">"+
          escHtml(l.name||"?")+"</span><span class='lv'>"+v+"</span>"+
          "<span class='lst "+_lamStatusCls(l.status)+"'>"+escHtml(String(l.status||"").toLowerCase())+"</span></div>";
      });
      cells.push("<div class='cscell'><div class='csk'>costate λ (∂S/∂·)</div><div class='lamtab'>"+rows+"</div></div>");
    }
    const dec=[];
    if(C.duty_owed!=null)dec.push("duty-to-measure <b>"+C.duty_owed+"</b> owed"+
      (C.duty_never_fired!=null?" ("+C.duty_never_fired+" never-fired)":""));
    if(C.duty_ranked&&C.duty_ranked.length)
      dec.push("next owed: "+C.duty_ranked.slice(0,3).map(d=>escHtml(d.lever||"?")).join(", "));
    if(C.probe_queue!=null)dec.push("probe queue <b>"+C.probe_queue+"</b>");
    if(C.axis_ev){const a=C.axis_ev;
      dec.push("axis EV — seg <b>"+sig(a.seg,3)+"</b> · pose <b>"+sig(a.pose,3)+"</b> · rate <b>"+sig(a.rate,3)+"</b>");}
    if(C.factorized_adjoint){const f=C.factorized_adjoint;
      dec.push("exact-factorized <b>"+escHtml(f.admission||"?")+"</b> · head rank "+
        escHtml(String(f.head_rank==null?"?":f.head_rank))+" · ker(A) zero "+
        (f.zero_weight_camera_frac==null?"?":(100*f.zero_weight_camera_frac).toFixed(1)+"%")+
        " · λ<sub>Road-Lane</sub> "+(f.road_lane_lambda_ratio==null?"?":sig(f.road_lane_lambda_ratio,3)+"×")+
        " · learned "+escHtml(String(f.learned_parameters==null?"?":f.learned_parameters))+" scalars");
      if(f.confidence)dec.push("factorized confidence "+escHtml(f.confidence));
      if(f.why)dec.push("factorized why "+escHtml(f.why));
    }
    if(C.event_advisories&&C.event_advisories.length){const e=C.event_advisories[0];
      dec.push("Morse-Smale/#344 <b>"+(e.warning_active?"WARNING ACTIVE":"next-boundary watch")+"</b>");}
    if(src.actuation)dec.push("actuation <b>"+escHtml(src.actuation)+"</b> (advisory)");
    if(dec.length)
      cells.push("<div class='cscell'><div class='csk'>DECIDE &middot; duty queue</div><div class='csline'>"+
        dec.join("<br>")+"</div></div>");
    if(cells.length)h+="<div class='csgrid'>"+cells.join("")+"</div>";
  }
  if(campaign){
    const nag=campaign.activation_nag||{}, sensed=campaign.sense_rows||[];
    const measured=sensed.filter(r=>r.value!=null).length;
    const next=(nag.next_duty||{}).duty||"none";
    h+="<div class='csgrid'><div class='cscell'><div class='csk'>DDM #366 SENSE</div>"+
      "<div class='csline'>measured <b>"+measured+"/"+sensed.length+"</b> standing rows"+
      " &middot; blockers <b>"+(campaign.blockers||[]).length+"</b>"+
      " &middot; state <b>"+escHtml(String(campaign.state_digest||"").slice(0,12))+"</b></div></div>"+
      "<div class='cscell'><div class='csk'>DDM campaign DECIDE</div><div class='csline'>"+
      "next owed <b>"+escHtml(next)+"</b> &middot; activation nag "+
      "<b>"+escHtml(nag.status||"?")+"</b> &middot; actuation <b>NONE</b></div></div></div>";
  }
  // ── liveness strip (confound-immune: a frozen run must LOOK frozen) ──
  if(L){
    const lp=[];
    const pill=(cls,txt)=>"<span class='lvpill "+cls+"'>"+txt+"</span>";
    if(L.accepted_frac!=null){const a=+L.accepted_frac;
      lp.push(pill(a>=0.5?"ok":"warn","accepted "+(a*100).toFixed(0)+"%"));}
    if(L.weights_stepped!=null)lp.push(pill(L.weights_stepped?"ok":"","stepped "+(L.weights_stepped?"yes":"no")));
    if(L.skipped_batches!=null)lp.push(pill(L.skipped_batches>0?"warn":"","skipped "+L.skipped_batches));
    if(L.ep_loss!=null)lp.push(pill("","loss "+sig(L.ep_loss,4)));
    (L.alarms||[]).forEach(a=>lp.push(pill("alarm","⚠ "+escHtml(a))));
    if(lp.length)h+="<div class='livestrip'>"+lp.join("")+
      "<span class='lvpill'>liveness @ ep"+(L.epoch!=null?L.epoch:"?")+"</span></div>";
  }
  body.innerHTML=h;
}

function render(){
  recomputeBest();
  const last=TRAJ.length?TRAJ[TRAJ.length-1]:null;
  const g=goalVal();
  fillPtrSpans();  // pointer prose spans — data-driven from the canonical pointer file
  const _nrm=$("lv_norm");
  if(_nrm&&META.archive_norm_bytes)_nrm.textContent=META.archive_norm_bytes.toLocaleString("en-US");
  // liveness pill
  const k=LIVE.kind, p=$("pill");
  if(LIVE.bench&&k==="live"){p.className="pill warm";p.textContent="◐ bench in flight";}
  else if(LIVE.bench&&k==="stale"){p.className="pill stale";p.textContent="⚠ bench stalled";}
  else if(k==="live"&&!LIVE.calibrating){p.className="pill live";p.textContent="● live";}
  else if(k==="live"&&LIVE.calibrating){p.className="pill warm";p.textContent="◐ warming up";}
  else if(k==="stale"){p.className="pill stale";p.textContent="⚠ stale";}
  else{p.className="pill miss";p.textContent="⚠ no run log";}
  if(p._beat&&!reduceMotion)p.classList.add("beat");  // re-apply pulse (className reset wipes it)
  // n badge - which run am I watching (n200 DOE pilot / n600 scored)
  const nb=$("npill"), n=META.n_pairs;
  if(nb){
    if(n==null){nb.className="nbadge other";nb.textContent="n=?";}
    else if(n===200){nb.className="nbadge doe";nb.textContent="n=200 · DOE pilot";}
    else if(n===600){nb.className="nbadge scored";nb.textContent="n=600 · scored";}
    else{nb.className="nbadge other";nb.textContent="n="+n;}
  }
  const rd=$("rdinfo"); if(rd){
    if(META.pipeline_follow){rd.textContent="pipeline "+(META.run_dir||"?")+
      " · bench (dry-start) in flight — real run not fired yet; watcher follows automatically";}
    else if(!META.run_dir){rd.textContent="resolving run…";}
    else{rd.textContent="watching "+META.run_dir+(META.warming_up?" · warming up (structured-init, no verdict yet)":"");}
  }
  // LIVE instrument panels (rebuilt 2026-07-09): masthead S-decomposition + per-class +
  // pose-readiness + training-health + system + schedule. All read the newest verdict
  // point (+ META.sensors for the non-verdict jacobian_basin / loss_terms stages).
  renderMasthead(last);
  renderChain();
  renderAlarms();
  renderClasses(last);
  renderPose();
  renderHealth();
  renderSys(last);
  renderSchedule(last);
  // stage legend: CONDITIONAL — only stages present in the derived map light up
  // (l7 hidden when disabled). Legacy Muon-only toggle when no map is present.
  const _lmap=stageMap();
  if(_lmap){
    const present={}; _lmap.forEach(s=>{present[s.name.toLowerCase()]=true;});
    document.querySelectorAll('#slegend .sc[data-st]').forEach(el=>{
      el.classList.toggle("off",!present[el.dataset.st]);});
  }else{
    const muOn=(META.muon_start!=null);
    document.querySelectorAll('#slegend .sc[data-st="muon"]').forEach(el=>el.classList.toggle("off",!muOn));
  }
  renderProjection(g);
  renderRunIdentity();
  renderCostate();
  renderConfig();
  renderTelemetry();
  renderRunInfo();
  // status
  const ep=LIVE.last_epoch;
  let st=[];
  if(k==="missing"){st=["no run log found"];}
  else{st.push(stageWord(ep)+" stage"); if(ep!=null)st.push("ep"+ep);
    // event-gated stages not yet fired render as "next: <stage> (…) — pending";
    // GENERIC over the DSL: a declared/unknown-kind schedule element (no epoch
    // boundary) renders as its describe() data — NEVER silently omitted.
    (stageMap()||[]).forEach(s=>{
      if(s.mode==="event"&&s.status==="pending"&&(ep==null||s.cap==null||ep<s.cap))
        st.push("next: "+s.name+" ("+(s.trigger||"event-gated")+") — pending");
      else if(s.mode!=="event"&&stageBoundary(s)==null){
        const extra=Object.keys(s).filter(k=>!["name","kind","mode","status","start","source"].includes(k))
          .map(k=>k+": "+s[k]).join(", ");
        st.push(s.name+(s.kind&&s.kind!==s.name?" ["+s.kind+"]":"")+(extra?" ("+extra+")":""));
      }});
    if(k==="stale")st.push("no verdict in "+fmtAge(LIVE.verdict_age_s)+" — likely stopped");
    else{const nvh=nextVerdictHint(); if(nvh)st.push(nvh);}}
  setTxt("status",st.join(" · "));
  // detail
  let d=[];
  if(LIVE.verdict_age_s!=null)d.push("verdict "+fmtAge(LIVE.verdict_age_s)+" ago");
  if(LIVE.log_age_s!=null)d.push("log "+fmtAge(LIVE.log_age_s)+" ago");
  if(LIVE.cadence_s)d.push("cadence ~"+(LIVE.cadence_s/60).toFixed(0)+"m ("+(LIVE.calibrating?"calibrating":"measured")+")");
  if(META.uptime_s!=null)d.push("dash up "+fmtAge(META.uptime_s));
  if(META.training_alive!=null)d.push("training "+(META.training_alive?"alive":"gone"));
  // schedule provenance marker — "fallback" is the visible read-back-failed state
  if(META.schedule_source)d.push("schedule: "+META.schedule_source);
  setTxt("detail",d.join(" · ")||" ");
  // foot
  setTxt("foot","[macOS-MLX advisory · NON-PROMOTABLE] · pointer "+ptrTxt()+" · stages CE · tau · l7 · Muon"+
    (META.watched?(" · "+META.watched):"")+" · "+TRAJ.length+" verdicts · tap charts for details");
  // boot spans in triality
  // boundary readouts from the derived map: number when resolved, "event-gated
  // (pending)" for an unfired event stage, "off" when the run disables the stage.
  const _bm={}; (stageMap()||[]).forEach(s=>{_bm[s.name]=s;});
  function _btxt(name,legacy){
    const s=_bm[name];
    if(s){if(s.start!=null)return s.start;
      if(s.mode==="event"&&s.status==="pending")return "event-gated (pending)";}
    if(stageMap())return "off";
    return (legacy!=null)?legacy:"?";}
  // (2026-07-07 charts-frozen-until-hover ROOT CAUSE) the b_tau/b_l7/b_goal badge markup was
  // removed in the detection-game reorg (c51385066) but these writes survived — every render()
  // since threw here, the trailing scheduleDraw() never ran, and ws.onmessage's catch swallowed
  // it, so charts only repainted via the hover handlers' direct scheduleDraw. Null-guard.
  const _bt=$("b_tau");if(_bt)_bt.textContent=_btxt("tau",(META.tau!=null)?META.tau:BOOT.tau);
  const _bl=$("b_l7");if(_bl)_bl.textContent=_btxt("l7",(META.l7!=null)?META.l7:BOOT.l7);
  const _bg=$("b_goal");if(_bg)_bg.textContent=sig(goalVal(),4);
  scheduleDraw();
}

// ---------- projection (rendered from the SERVER-computed critical-slowing model) ----------
// The fit MATH lives server-side in tools/dashboard_trajectory_model.py (pure numpy);
// the client only RENDERS the returned numbers + their flags. Never a client-side fit.
// (goalEtaStr/fmtEps DELETED 2026-07-07: dead since the stage-aware reframe; deleting
// them is the structural guarantee that the GLOBAL fit's "asymptote_above / won't
// reach" verdict can never render — the stage-aware model exists to override exactly
// that mis-declaration, and the fallback branch below renders the global fit WITHOUT
// a verdict by design.)
// stage dot colors — same palette as the SETUP panel's schedule rows (visual unity)
const STAGE_DOT={CE:"#5ab0ff",tau:"#b08cff",l7:"#ffa454",Muon:"#46d3a0"};
// one projection row — REUSES the SETUP panel's cfgrow/cfgk/cfgv system (operator
// 2026-07-07 "needs better visual structure, on mobile ... too cluttered"). Long
// explanatory clauses ride the value's title tooltip (double-quoted: escHtml escapes
// `"` not `'`); the visible line stays one short value. Values wrap (scoped CSS).
function projRow(dot,key,valHtml,tip,cur){
  return "<div class='cfgrow'><span class='cfgk'>"+
    (dot?("<span class='cdot' style='background:"+dot+"'></span>"):"")+escHtml(key)+
    (cur?" <span class='curchip'>◀ current</span>":"")+"</span>"+
    "<span class='cfgv'"+(tip?(" title=\""+escHtml(tip)+"\""):"")+">"+valHtml+"</span></div>";
}
function renderProjection(g){
  const segEl=$("proj_seg"), sEl=$("proj_s"); if(!segEl||!sEl)return;
  const P=PROJ||{};
  if(!P.ok){
    segEl.innerHTML="projection · <b>calibrating</b> — "+escHtml(P.reason||"collecting verdicts")+
      " (critical-slowing fit needs ≥5 verdicts)";
    sEl.textContent=" ";
    return;
  }
  // ---- STAGE-AWARE d_seg model (per-curriculum-stage critical-slowing fits) ----
  // The GLOBAL fit (P.dseg_model) collapses the 3-stage curriculum into ONE power-law and,
  // from the CE flicker plateau, mis-declares "won't reach". We render P.stage_proj instead,
  // one LABELED ROW per stage (cfgrow pattern) — nothing dropped: what leaves the visible
  // line moves into the row's tooltip. Muon = saddle staircase, NEVER power-law-extrapolated.
  let rows="";
  let caveat="";
  const SP=P.stage_proj||{};
  if(SP.ok){
    (SP.stages||[]).forEach(st=>{
      const cur=(st.name===SP.current_stage);
      const dot=STAGE_DOT[st.name]||"#888";
      if(st.name==="Muon"){
        let v, tip="saddle-to-saddle staircase (polynomial escape) — not power-law; never extrapolated as a fit";
        if(st.note) tip=escNothing(st.note)+" · "+tip;
        if(st.observed_min!=null){
          v="floor <b>"+sig(st.observed_min,4)+"</b> over "+(st.n||0)+" verdict"+((st.n===1)?"":"s");
          if(st.trend&&st.trend.slope_per_25ep!=null){
            v+=" · "+(st.trend.slope_per_25ep>0?"+":"")+sig(st.trend.slope_per_25ep,3)+"/25ep";
            if(st.trend.readthrough_dseg!=null)
              v+=" → ep"+fmtInt(st.trend.readthrough_epoch)+" ~"+sig(st.trend.readthrough_dseg,4);
            v+=" (observed, not a fit)";
            tip+=" · linear read-through of the recent slope ("+(st.trend.label||"observed trend, not a fit")+")";
          }
        } else {
          v="ep"+fmtInt(st.start)+"+ · saddle staircase (unmodeled until measured)";
        }
        rows+=projRow(dot,"Muon",v,tip,cur);
      } else if(!st.entered){
        rows+=projRow(dot,st.name,"ep"+fmtInt(st.start)+"–"+fmtInt(st.end)+" · not entered yet",null,cur);
      } else if(st.fit_state==="ok"){
        const cflag=(st.confidence==="low")?" ⚠":"";
        let v="d_seg<sub>∞</sub> <b>"+sig(st.asymptote,4)+"</b>";
        if(st.observed_min!=null) v+=" ≈ floor "+sig(st.observed_min,4);
        v+=" · α "+sig(st.alpha,2)+" · R² "+sig(st.r2,3)+" · "+(st.confidence||"?")+cflag;
        let tip="critical-slowing power-law fit over this stage's own verdicts";
        if(st.name==="CE"){v+=" (plateau, not final)";
          tip+=" · the CE plateau is NOT the final floor (as expected)";}
        rows+=projRow(dot,st.name,v,tip,cur);
      } else if(st.fit_state==="insufficient"){
        rows+=projRow(dot,st.name,(st.n||0)+" verdict"+((st.n===1)?"":"s")+
          " — insufficient for a stage fit yet",null,cur);
      } else {
        rows+=projRow(dot,st.name,escHtml(st.note||"no fit"),null,cur);
      }
    });
    // breakthroughs row — the boundaries the per-stage fits do NOT model. Display
    // uses the SHORT label (server parenthetical -> tooltip) so the row stays compact.
    if((SP.downstream||[]).length){
      const dlab=SP.downstream.map(d=>"ep"+fmtInt(d.epoch)+" "+
        escHtml(String(d.label||"").replace(/\s*\(.*\)\s*$/,""))+
        (d.status==="engaged"?" (engaged)":" (expected)")).join(" · ");
      const dtip=SP.downstream.map(d=>"ep"+fmtInt(d.epoch)+" "+d.label+" — "+d.status).join(" · ");
      rows+=projRow(null,"breakthroughs",dlab,
        "expected-breakthrough boundaries the per-stage critical-slowing fits do NOT model · "+dtip);
    }
    // floor -> score row (+ its OWN quiet pose row on pose-blind arms)
    const mf=SP.modeled_floor, is=(mf&&mf.implied_s)||{};
    if(mf&&is.ok&&is.value!=null){
      const gapTip=mf.is_current_stage?"":"latest modeled stage; current "+
        (SP.current_stage||"?")+" is still calibrating · ";
      const baseTip=gapTip+"CURRENT-STAGE extrapolation only — excludes the downstream breakthroughs";
      if(mf.pose_blind&&mf.seg_term!=null){
        let v="100·d_seg<sub>∞</sub> = <b>"+sig(mf.seg_term,4)+"</b>";
        if(mf.seg_term_lo!=null&&mf.seg_term_hi!=null)
          v+=" ("+sig(mf.seg_term_lo,4)+"–"+sig(mf.seg_term_hi,4)+")";
        if(!mf.is_current_stage) v+=" (latest modeled)";
        rows+=projRow(null,"d_seg term ("+(mf.stage||"?")+" floor)",v,baseTip);
        rows+=projRow(null,"pose","unheld by design (w_pose=0) — composite "+
          sig(is.value,4)+", demoted",
          "the pose term is measured-but-untrained in this arm, so the composite implied_S is dominated by it by construction");
      } else {
        let v="<b>"+sig(is.value,4)+"</b>";
        if(is.value_lo!=null||is.value_hi!=null)
          v+=" ("+sig(is.value_lo,4)+"–"+(is.value_hi!=null?sig(is.value_hi,4):"≳")+")";
        if(!mf.is_current_stage) v+=" (latest modeled)";
        rows+=projRow(null,"implied_S ("+(mf.stage||"?")+" floor)",v,baseTip);
      }
      const ts=(SP.tau_start!=null)?fmtInt(SP.tau_start):"?", ms=(SP.muon_start!=null)?fmtInt(SP.muon_start):"?";
      caveat="advisory · current-stage extrapolation ONLY — excludes the ep"+ts+"/ep"+ms+
        " breakthroughs · vs pointer "+ptrTxt();
    }
  } else {
    // stage projection unavailable -> fall back to the global fit, but WITHOUT a "won't reach" verdict
    const fit=P.dseg_model||{};
    if(fit.ok){
      const cflag=(fit.confidence==="low")?" ⚠":"";
      rows+=projRow(null,"global fit","d_seg<sub>∞</sub> <b>"+sig(fit.asymptote,4)+
        "</b> · α "+sig(fit.alpha,2)+" · R² "+sig(fit.r2,3)+" · "+(fit.confidence||"low")+cflag,
        "single global critical-slowing fit — current-trajectory floor; stage breakthroughs (tau/Muon) unmodeled");
    } else {
      rows+=projRow(null,"d_seg model","<b>calibrating</b> — "+escHtml(fit.reason||"need more points"),null);
    }
  }
  // ---- ETA / cadence row: MEASURED completion ETA + current-stage cadence ----
  const etaBits=[];
  const ce=P.completion_eta||{};
  const tot=(META.schedule&&META.schedule.epochs)||null;
  if(ce.ok&&ce.total_s!=null){
    let c="ep"+(tot!=null?tot:"end")+" ~"+fmtAge(ce.total_s);
    if(ce.has_estimate){
      // the refine boundary is the run's OWN derived Muon start (never a literal)
      const _mb=(META.muon_start!=null)?META.muon_start
        :((stageMap()||[]).filter(s=>s.name==="Muon").map(stageBoundary).find(v=>v!=null));
      c+=(_mb!=null)?" (Muon estimated — refines at ep"+fmtInt(_mb)+")"
                    :" (Muon estimated — refines when Muon engages)";
    }
    else c+=" (measured)";
    etaBits.push(c);
  }
  if(P.next_verdict_cadence_s!=null){
    const src=P.next_verdict_cadence_source||"measured";
    etaBits.push("cadence ~"+(P.next_verdict_cadence_s/60).toFixed(0)+"m ("+src+")");
  }
  if(etaBits.length)
    rows+=projRow(null,"ETA / cadence",escHtml(etaBits.join(" · ")),
      "measured completion ETA to end-of-run + the current stage's verdict cadence");
  segEl.innerHTML="<div class='cfgsubh'>d_seg per-stage critical-slowing model</div>"+
    "<div class='cfgrows'>"+rows+"</div>";
  sEl.innerHTML=caveat?("<div class='cfgmeta'>"+caveat+"</div>"):" ";
}
// tooltip fragments come from server strings that are PLAIN TEXT already; alias kept
// explicit so a future reader doesn't double-escape (escHtml runs once, in projRow).
function escNothing(s){return String(s==null?"":s);}

// setup / config / schedule / curriculum panel — rendered from META.config + META.schedule
// (parsed server-side from the run's OWN launch.sh / run.log; generalizable to any run)
function escHtml(s){return String(s).replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));}
// one curriculum row. disReason!=null -> DISABLED (never runs); no inverted [a,b).
function schStageRow(dot,name,start,end,note,disReason){
  const inner=disReason
    ? "<span class='cfgv dis'>"+disReason+"</span>"
    : "<span class='cfgv'>ep ["+start+", "+(end!=null?end:"&hellip;")+")"+(note?(" &middot; "+note):"")+"</span>";
  return "<div class='cfgrow"+(disReason?" off":"")+"'><span class='cfgk'><span class='cdot' style='background:"+
    dot+"'></span>"+escHtml(name)+"</span>"+inner+"</div>";
}
// Build the curriculum schedule as TWO ORTHOGONAL AXES, data-driven from the raw
// schedule fields (generalizes to any run). Returns {body} or null.
//  * LOSS-FORM axis (seg_form): CE -> tau -> l7. Each loss stage ENDS at the NEXT
//    ENABLED loss stage's start, clamped to epochs — the OPTIMIZER switch does NOT
//    terminate a loss stage (different axis). A loss stage whose start >= epochs is
//    DISABLED (never runs) and rendered as such, NEVER an inverted/empty [a,b).
//  * OPTIMIZER axis (orthogonal, composes): AdamW -> Muon (Muon replaces AdamW at
//    muon_start). Muon disabled when muon_start is unset or >= epochs.
function buildScheduleRows(sched){
  const col={CE:"#5ab0ff",tau:"#b08cff",l7:"#ffa454",Muon:"#46d3a0",AdamW:"#8b93a3"};
  const epochs=(sched.epochs!=null)?sched.epochs:null;
  const tau=sched.tau_start, l7=sched.l7_start;
  const muon=(META.muon_start!=null)?META.muon_start:sched.muon_start;
  const enabled=s=>(s!=null&&(epochs==null||s<epochs));
  const disMsg=s=>"disabled ("+s+" &ge; epochs"+(epochs!=null?(" "+epochs):"")+")";
  const hasFlags=(tau!=null||l7!=null||muon!=null||epochs!=null);
  if(hasFlags){
    // ---- LOSS-FORM axis: CE -> tau -> l7 (CE always starts at 0) ----
    const cand=[["CE",0],["tau",tau],["l7",l7]];
    let loss="";
    for(let i=0;i<cand.length;i++){
      const name=cand[i][0], start=cand[i][1];
      if(start==null)continue;                          // stage not configured for this run
      if(!enabled(start)){                              // start >= epochs -> DISABLED (never runs)
        loss+=schStageRow(col[name]||"#888",name,start,null,null,disMsg(start));
        continue;
      }
      let end=epochs;                                   // end = next ENABLED loss start, clamped
      for(let j=i+1;j<cand.length;j++){
        if(enabled(cand[j][1])){end=cand[j][1];break;}
      }
      loss+=schStageRow(col[name]||"#888",name,start,end,null,null);
    }
    // ---- OPTIMIZER axis (orthogonal): AdamW -> Muon ----
    let opt="";
    if(enabled(muon)){
      opt+=schStageRow(col.AdamW,"AdamW",0,muon,"then Muon",null);
      opt+=schStageRow(col.Muon,"Muon (optimizer)",muon,epochs,"replaces AdamW",null);
    } else {
      opt+=schStageRow(col.AdamW,"AdamW (optimizer)",0,epochs,"whole run",null);
      if(muon!=null)opt+=schStageRow(col.Muon,"Muon (optimizer)",muon,null,null,disMsg(muon));
    }
    let body="";
    if(loss)body+="<div class='cfgsubh'>loss form (seg_form)</div><div class='cfgrows'>"+loss+"</div>";
    if(opt)body+="<div class='cfgsubh'>optimizer</div><div class='cfgrows'>"+opt+"</div>";
    return body?{body:body}:null;
  }
  // ---- fallback: run.log-sourced stages array; GUARD against inverted/empty ranges ----
  const stages=sched.stages||[];
  if(!stages.length)return null;
  let rows="";
  stages.forEach(s=>{
    const st=s.start, en=s.end, inv=(en!=null&&st!=null&&en<=st);
    rows+=schStageRow(col[s.name]||"#888",s.name,st,en,null,inv?"never (inverted range)":null);
  });
  return {body:"<div class='cfgrows'>"+rows+"</div>"};
}
// Schema-driven schedule rows (#352): classify each DSL element EVENT-TRIGGERED / FIXED-CAP
// with its LIVE arm state (pending/fired, watching cap), rendered from introspect.schedule.
// This SUPERSEDES the epoch-scripted buildScheduleRows lens for v6+ event-governed runs; the
// legacy renderer stays the fallback for runs without a DSL read-back.
function schClassifiedRows(S){
  if(!S||!S.ok||!S.stages||!S.stages.length)return null;
  const ep=(LIVE&&LIVE.last_epoch!=null)?LIVE.last_epoch:null;
  let rows="";
  S.stages.forEach(s=>{
    const isEvent=(s.klass==="event"||s.mode==="event");
    const kchip="<span class='kchip "+(isEvent?"event":"fixed")+"'>"+(isEvent?"event":"fixed")+"</span>";
    let statusCls="scheduled", statusTxt="scheduled", body="";
    if(isEvent){
      const fired=(s.status==="fired"||s.fired_epoch!=null);
      statusCls=fired?"fired":"pending"; statusTxt=fired?("fired @ ep"+s.fired_epoch):"armed · pending";
      body="<span class='strig'>"+escHtml(s.trigger||"event-gated")+
        (s.cap!=null?(" &middot; hard cap ep"+s.cap):"")+"</span>";
    }else{
      body="<span class='sbody'>starts ep "+(s.start!=null?s.start:"0")+"</span>";
      if(s.start!=null&&ep!=null)statusTxt=(ep>=s.start)?"active":"upcoming";
    }
    rows+="<div class='schrow'><span class='snm'><span class='sdot "+statusCls+"'></span>"+escHtml(s.name)+kchip+"</span>"+
      body+"<span class='sright'><span class='stchip "+statusCls+"'>"+statusTxt+"</span></span></div>";
  });
  const meta=[];
  if(S.epochs!=null)meta.push("total "+S.epochs+" ep");
  if(S.eval_every!=null)meta.push("eval every "+S.eval_every+" ep");
  if(S.event_triggered)meta.push("event-governed curriculum");
  meta.push("source "+escHtml(S.source||"launch.sh")+" · DSL read-back");
  return {body:"<div class='cfgrows'>"+rows+"</div><div class='cfgmeta'>"+escHtml(meta.join(" · "))+"</div>"};
}
function renderConfig(){
  const body=$("cfgbody"), sum=$("cfgsum"); if(!body)return;
  const cfg=META.config||{}, sched=(META.schedule||cfg.schedule||{});
  const groups=cfg.groups||{};
  const IS=(META.introspect&&META.introspect.schedule)||null;
  let html="";
  // curriculum schedule (full width) — PREFER the schema-driven DSL classification
  // (event/fixed + live arm state); fall back to the legacy two-axis epoch renderer when
  // no DSL read-back is available (pre-v6 dirs / broken tac install).
  const sch=(IS&&IS.ok)?schClassifiedRows(IS):buildScheduleRows(sched);
  if(sch){
    const _shdr=(IS&&IS.ok)?"curriculum &middot; stages (DSL)":"curriculum schedule";
    html+="<div class='cfgsec full'><div class='cfgh'>"+_shdr+"</div>"+sch.body;
    if(IS&&IS.ok){html+="</div>";}
    else{
    const m2=[];
    if(sched.epochs!=null)m2.push("total "+sched.epochs+" ep");
    if(sched.eval_every!=null)m2.push("eval every "+sched.eval_every+" ep");
    if(LIVE.next_eta_s!=null)m2.push("next verdict ~"+fmtAge(LIVE.next_eta_s));
    if(m2.length)html+="<div class='cfgmeta'>"+escHtml(m2.join(" · "))+"</div>";
    html+="</div>";
    }
  }
  // config groups
  ["architecture","basis","optimizer","seed","regularizers","loss"].forEach(gn=>{
    const rows=groups[gn]; if(!rows||!rows.length)return;
    html+="<div class='cfgsec'><div class='cfgh'>"+escHtml(gn)+"</div><div class='cfgrows'>";
    rows.forEach(kv=>{let v=kv[1]; if(v===true)v="on"; if(v===false)v="off";
      html+="<div class='cfgrow'><span class='cfgk'>"+escHtml(kv[0])+"</span><span class='cfgv'>"+escHtml(v)+"</span></div>";});
    html+="</div></div>";
  });
  if(!html){
    html="<div class='cfgmeta'>"+(META.warming_up?"warming up — run config loading…":"config not yet available")+"</div>";
  }
  body.innerHTML=html;
  if(sum){const src=cfg.source&&cfg.source!=="none"?(" · from "+cfg.source):"";
    sum.textContent="setup · config · schedule · curriculum"+src;}
}

// ---------- #352 telemetry panel: planned curves + LawRef constants + mem + fired events ----------
// Every section is CONDITIONAL on its source artifact (introspect_run) — a pre-v6 run dir
// simply renders fewer sections; nothing is fabricated. Observability ONLY.
// inline SVG sparkline from [[x,y]...] points (no deps); emphasized endpoints per dataviz.
function _sparkSvg(points,color,markEpoch){
  if(!points||points.length<2)return "";
  const W=140,H=40,pad=3;
  let xmin=points[0][0],xmax=points[0][0],ymin=points[0][1],ymax=points[0][1];
  points.forEach(p=>{if(p[0]<xmin)xmin=p[0];if(p[0]>xmax)xmax=p[0];
    if(p[1]<ymin)ymin=p[1];if(p[1]>ymax)ymax=p[1];});
  const sx=x=>pad+(xmax>xmin?(x-xmin)/(xmax-xmin):0)*(W-2*pad);
  const sy=y=>H-pad-(ymax>ymin?(y-ymin)/(ymax-ymin):0.5)*(H-2*pad);
  const d=points.map((p,i)=>(i?"L":"M")+sx(p[0]).toFixed(1)+" "+sy(p[1]).toFixed(1)).join(" ");
  const last=points[points.length-1];
  let mk="";
  if(markEpoch!=null&&markEpoch>xmin&&markEpoch<xmax){
    const mxp=sx(markEpoch).toFixed(1);
    mk="<line x1='"+mxp+"' y1='"+pad+"' x2='"+mxp+"' y2='"+(H-pad)+"' stroke='#46d3a0' stroke-width='1' stroke-dasharray='2 2' opacity='.7'/>";
  }
  return "<svg class='crvsvg' viewBox='0 0 "+W+" "+H+"' preserveAspectRatio='none' aria-hidden='true'>"+
    mk+"<path d='"+d+"' fill='none' stroke='"+color+"' stroke-width='1.6' vector-effect='non-scaling-stroke'/>"+
    "<circle cx='"+sx(last[0]).toFixed(1)+"' cy='"+sy(last[1]).toFixed(1)+"' r='2.2' fill='"+color+"'/></svg>";
}
function renderTelemetry(){
  const box=$("telemetry"), body=$("telbody"); if(!box||!body)return;
  const I=META.introspect||{};
  const curves=I.curves, consts=I.constants, mem=I.mem, events=I.events;
  if(!curves&&!consts&&!mem&&!events){box.classList.add("hide");return;}
  box.classList.remove("hide");
  let html="";
  const col={tau:"#b08cff",beta:"#ffb454",lr:"#5ab0ff"};
  // ── planned curves (τ / β / LR) ──
  if(curves&&curves.curves){
    const cs=curves.curves; let cells="";
    ["tau","beta","lr"].forEach(k=>{const c=cs[k]; if(!c)return;
      const ep0=c.points[0][0], epN=c.points[c.points.length-1][0];
      cells+="<div class='crvcell'><div class='crvh'><span class='crvn'>"+escHtml(c.name)+"</span>"+
        "<span class='crvsh'>"+escHtml(c.shape)+"</span></div>"+
        _sparkSvg(c.points,col[k]||"#7fc0ff",c.muon_freeze)+
        "<div class='crvep'><span>ep"+ep0+" · "+sig(c.start,3)+"</span><span>ep"+epN+" · "+sig(c.end,3)+"</span></div>"+
        (c.note?("<div class='crvnote'>"+escHtml(c.note)+"</div>"):"")+"</div>";
    });
    html+="<div class='cfgsec full'><div class='cfgh'>planned curves"+
      (curves.muon_start!=null?" &middot; ◈ Muon freeze @ ep"+curves.muon_start:"")+
      "</div><div class='crv'>"+cells+"</div></div>";
  }
  // ── LawRef constants manifest (#351) ──
  if(consts&&consts.rows&&consts.rows.length){
    let rows="";
    consts.rows.forEach(r=>{
      const val=(r.value!=null)?escHtml(String(r.value)):"&mdash;";
      const kc=(r.ladder_class==="measured_anchor"||r.ladder_class==="derived_at_config"||r.ladder_class==="derived_live")?"derived":"cap";
      rows+="<div class='cstrow'><span class='cstnm' title=\""+escHtml(r.equation_id||"")+"\">"+escHtml(r.name)+"</span>"+
        "<span class='kchip "+kc+"'>"+escHtml(r.ladder_label||r.ladder_class)+"</span>"+
        "<span class='cstv'>"+val+"</span>";
      if(r.provenance)rows+="<div class='cstprov'><code>"+escHtml(r.equation_id||"")+"</code> "+escHtml(r.provenance)+
        (r.anchor_sha?(" · anchor "+escHtml(r.anchor_sha)):"")+"</div>";
      rows+="</div>";
    });
    html+="<div class='cfgsec full'><div class='cfgh'>constants &middot; LawRef manifest (#351)</div>"+
      "<div class='cst'>"+rows+"</div>"+
      "<div class='cfgmeta'>"+consts.count+" resolved · "+escHtml(consts.config_family||"")+
      " · value-identity is the law</div></div>";
  }
  // ── mem_probe (#329) — RSS/MLX sparkline over the recent window + latest phase ──
  if(mem&&((mem.series&&mem.series.length>1)||(mem.rows&&mem.rows.length))){
    const peak=mem.peak_rss_gib||1;
    const lt=mem.latest||(mem.rows&&mem.rows[mem.rows.length-1])||{};
    let curveH="";
    if(mem.series&&mem.series.length>1)curveH=_sparkSvg(mem.series,"#5ab0ff",null);
    let mlxH="";
    if(mem.mlx_series&&mem.mlx_series.length>1)mlxH=_sparkSvg(mem.mlx_series,"#c08cff",null);
    html+="<div class='cfgsec full'><div class='cfgh'>memory &middot; mem_probe (#329)</div>"+
      "<div class='crv'>"+
      (curveH?"<div class='crvcell'><div class='crvh'><span class='crvn'>RSS</span><span class='crvsh'>GiB</span></div>"+
        curveH+"<div class='crvep'><span>peak "+sig(peak,4)+"</span><span>now "+sig(lt.rss_gib,4)+"</span></div></div>":"")+
      (mlxH?"<div class='crvcell'><div class='crvh'><span class='crvn'>MLX active</span><span class='crvsh'>GiB</span></div>"+
        mlxH+"<div class='crvep'><span>&nbsp;</span><span>now "+sig(lt.mlx_active_gib,4)+"</span></div></div>":"")+
      "</div>"+
      "<div class='cfgmeta'>peak RSS "+sig(peak,4)+" GiB · latest phase "+escHtml(lt.phase||"?")+
      " · "+mem.count+" probes</div></div>";
  }
  // ── fired curriculum events (diamond glyph, distinct from epoch caps) ──
  if(events&&events.length){
    let chips="";
    events.forEach(e=>{chips+="<span class='evchip'><span class='evdia'></span>"+escHtml(e.label||e.stage)+
      (e.epoch!=null?("<span class='evep'>ep"+e.epoch+"</span>"):"")+"</span>";});
    html+="<div class='cfgsec full'><div class='cfgh'>fired events</div><div class='evlist'>"+chips+"</div></div>";
  }
  body.innerHTML=html||"<div class='cfgmeta'>no telemetry artifacts yet</div>";
}

// #205 run-info strip — server pre-renders the card grid (stage-progress / best-d_seg
// deploy / throughput+ETA / checkpoint ledger / resumability / MLX fast-path / config)
// via rld._run_info_html and ships it as an HTML string in META. Empty -> hidden (:empty).
function renderRunInfo(){
  const el=$("runinfostrip"); if(!el)return;
  el.innerHTML=META.run_info_html||"";
}

// ---------- ORACLE tab (Tab 1): the detector + openpilot physical priors + detectability field ----------
// Static payload (depends only on the GT cache) fetched ONCE from /api/oracle; 202 -> poll a few times.
let ORACLE=null, _oracleFetching=false, _oracleTries=0;
function activateOracle(){
  if(ORACLE){renderOracle();return;}
  if(_oracleFetching)return;
  _oracleFetching=true;
  fetch("/api/oracle",{cache:"no-store"}).then(r=>{
    if(r.status===200)return r.json();
    return r.json().then(j=>{_oracleFetching=false;_oracleReady(j);return null;});
  }).then(d=>{
    _oracleFetching=false;
    if(d&&d.ok){ORACLE=d;renderOracle();}
  }).catch(()=>{_oracleFetching=false;
    const h=$("orchdr");if(h)h.textContent="oracle fetch failed — retrying on next visit.";});
}
function _oracleReady(j){
  const h=$("orchdr"); if(!h)return;
  if(j&&j.status==="rendering") h.textContent="rendering the physical-prior atlas (governed CPU pass, ~2 s)…";
  else if(j&&j.status==="error") h.textContent="oracle render error: "+(j.err||"unknown");
  else h.textContent="waiting for the physical-prior atlas (governed CPU pass)…";
  if(_oracleTries++<40) setTimeout(activateOracle, 3000);
}
function renderOracle(){
  const O=ORACLE; if(!O||!O.ok)return;
  const hdr=$("orchdr"); if(hdr)hdr.style.display="none";
  const st=$("orcstats");
  if(st){
    const roleName=i=>((O.classes||[]).find(c=>c.i===i)||{}).label||("class "+i);
    const rec=(O.lane_band&&O.lane_band.band_recall_mean);
    st.innerHTML=""+
      _stat("lane band → d_seg", (rec!=null?(rec).toFixed(2)+" recall":"—"),
            (O.lane_band?O.lane_band.total_lines:0)+" lines over "+O.n_frames+" frames")+
      _stat("ego-ξ → d_pose", (O.xi?O.xi.mean_speed_ms.toFixed(1)+" m/s":"—"),
            "estimator "+((O.xi&&O.xi.estimator_id)||"?")+" (SE(3) screw)")+
      _stat("static core", roleName(O.roles?O.roles.hood:-1),
            "self-detected hood · road "+roleName(O.roles?O.roles.road:-1))+
      _stat("detectability", "ρ_seg margin", "bright = fragile separatrix = where d_seg lives");
  }
  const host=$("orcpanels");
  if(host){
    host.innerHTML="";
    (O.panels||[]).forEach(p=>{
      const fig=document.createElement("figure"); fig.className="orcfig";
      const img=document.createElement("img");
      img.src=p.atlas; img.alt="ORACLE physical-prior atlas · frame "+p.frame_idx;
      const cap=document.createElement("figcaption");
      cap.innerHTML="frame "+p.frame_idx+
        " · lane recall "+(p.lane_recall!=null?p.lane_recall.toFixed(2):"—")+
        " · ds "+(p.ds>=0?"+":"")+p.ds+"m · dψ "+(p.dpsi>=0?"+":"")+p.dpsi+"rad";
      fig.appendChild(img); fig.appendChild(cap); host.appendChild(fig);
    });
  }
  const xc=$("orcxichart"); if(xc&&O.xi_chart)xc.src=O.xi_chart;
}
function _stat(label,val,sub){
  return "<div class='orcstat'><span class='ol'>"+label+"</span><span class='ov'>"+val+
         "</span><span class='os'>"+(sub||"")+"</span></div>";
}

// ---------- TRIALITY tab (Tab 6): DATA-DRIVEN from the live DAG / DSL / equations artifacts ----------
let _triFetching=false, _triAt=0;
function activateTriality(){
  const now=Date.now();
  if(_triFetching)return;
  if(now-_triAt<60000 && $("tri_dag") && $("tri_dag").dataset.loaded==="1")return;
  _triFetching=true;
  fetch("/api/triality",{cache:"no-store"}).then(r=>r.json()).then(d=>{
    _triFetching=false; _triAt=Date.now(); renderTriality(d);
  }).catch(()=>{_triFetching=false;
    const b=$("tri_built");if(b)b.textContent="triality fetch failed — retrying on next visit.";});
}
function _esc(s){return String(s==null?"":s).replace(/[&<>]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;"}[c]));}
function renderTriality(d){
  if(!d)return;
  const b=$("tri_built");
  if(b){
    b.dataset.loaded="1";
    const ok=d.dag&&d.dag.ok, ok2=d.dsl&&d.dsl.ok, ok3=d.equations&&d.equations.ok;
    b.innerHTML="live · built "+_esc(d.built_at_utc||"?")+
      " · legs: DAG "+(ok?"✓":"—")+" · DSL "+(ok2?"✓":"—")+" · equations "+(ok3?"✓":"—")+
      (d.pointer!=null?(" · pointer "+d.pointer+" (UNMOVED)"):" · pointer unavailable");
  }
  const dag=$("tri_dag");
  if(dag){
    const D=d.dag||{};
    if(D.ok&&Array.isArray(D.recent)){
      dag.innerHTML="<div class='trimeta'>"+_esc(D.file)+" · "+D.total+" FEEDs</div>"+
        D.recent.slice().reverse().map(f=>
          "<div class='trirow'><span class='tk'>"+_esc(f.tick)+"</span>"+_esc(f.summary)+"</div>").join("");
    } else dag.textContent=(D.reason||"no DAG");
  }
  const dsl=$("tri_dsl");
  if(dsl){
    const S=d.dsl||{};
    if(S.ok){
      const p=S.program||{}, g=S.gauge||{};
      // GENERIC over the DSL: render whatever the DSL declares (stage_details =
      // uniform describe-surface entries; unknown kinds show their data, never dropped).
      const sd=(p.stage_details&&p.stage_details.length)?p.stage_details:null;
      const stagesStr = sd ? sd.map(e=>{
          const label=e.name||e.text||e.kind||"?";
          const extra=Object.keys(e).filter(k=>!["name","kind","text","mode","status","start"].includes(k)&&e[k]!=null&&e[k]!=="")
            .map(k=>k+"="+e[k]).join(" ");
          return label+(extra?" ("+extra+")":"");
        }).join(" → ") : (p.stages||[]).join(" → ");
      dsl.innerHTML=
        "<div class='trirow'><span class='tk'>program</span>"+(p.epochs||"?")+" ep · "+
          (p.num_pairs||"?")+" pairs · stages "+_esc(stagesStr)+"</div>"+
        "<div class='trirow'><span class='tk'>gauge</span>warp="+_esc(g.warp)+" · carrier="+
          _esc(g.carrier)+" · residual="+_esc(g.residual)+"</div>"+
        "<div class='trimeta'>tac.witness_dsl → validated trainer CLI (never-invent-flags)</div>";
    } else dsl.textContent=(S.reason||"no DSL");
  }
  const eq=$("tri_eq");
  if(eq){
    const E=d.equations||{};
    if(E.ok&&Array.isArray(E.recent)){
      eq.innerHTML="<div class='trimeta'>"+E.distinct+" distinct · "+E.rows+" rows</div>"+
        E.recent.slice().reverse().map(e=>
          "<div class='trirow'><span class='tk'>"+_esc(e.id)+"</span>"+_esc(e.desc)+"</div>").join("");
    } else eq.textContent=(E.reason||"no equations");
  }
}

// ---------- witness tab (RESIDUAL): live comma10k 6-panel + tribute, streamed over the WS ----------
// Panels arrive as data: URIs in the snapshot (first paint) + a {type:"witness"} WS message on
// each NEW checkpoint. Build each pair's <figure> once, then swap its <img> src IN PLACE (no
// reload). Rare + possibly-hidden tab -> cheap.
function renderWitness(){
  const host=$("witpanels"), hdr=$("withdr"); if(!host)return;
  const W=WITNESS||{};
  if(!W.ok||!Array.isArray(W.pairs)||!W.pairs.length){
    if(hdr) hdr.textContent=(W.status==="error"||W.err)
      ? ("witness render error — "+(W.err||"see server log"))
      : "rendering witness panels from the live checkpoint… (first render ~8s; refreshes on each new checkpoint)";
    return;
  }
  if(hdr){
    // CKPT PROVENANCE (p0_343 live test, 2026-07-17): the payload always carried
    // ckpt_dir but the header never rendered it — a just-launched run has no
    // checkpoint yet, so the panel latches on the PREVIOUS run's ckpt and, unlabeled,
    // presented a superseded run's ep/d_seg as current. Name the source dir and flag
    // it stale whenever it differs from the live watched run dir.
    const wdir=W.ckpt_dir?String(W.ckpt_dir).replace(/\/+$/,"").split("/").pop():null;
    const rdir=(META&&META.run_dir)?String(META.run_dir).replace(/\/+$/,"").split("/").pop():null;
    const stale=!!(wdir&&rdir&&wdir!==rdir);
    hdr.innerHTML=(stale?"<span style='color:#e6cf7a'>PREVIOUS-RUN checkpoint — the live run has "+
        "no checkpoint yet (first ckpt lands on its first --ckpt-every save)</span> · ":"")+
      "the <b>"+W.n_pairs_shown+" hardest &amp; most-diverse pairs</b> of the n600 drive "+
      "(spread across the segment, labelled by failure mode) · from BEST checkpoint @ epoch <b>"+
      (W.epoch!=null?W.epoch:"?")+"</b>"+(wdir?" of <b>"+escHtml(wdir)+"</b>":"")+
      " · mean realized d_seg over the selection <b>"+sig(W.mean_dseg,5)+
      "</b> · rendered in "+sig(W.render_secs,3)+"s · built "+escHtml(W.built_at_utc||"")+" · "+
      escHtml(W.authority||"");
  }
  const present=new Set();
  W.pairs.forEach(p=>{
    const fid="witfig-"+p.pair_idx; present.add(fid);
    let fig=$(fid);
    if(!fig){
      fig=document.createElement("figure");fig.className="witfig";fig.id=fid;
      const img=document.createElement("img");img.id="witimg-"+p.pair_idx;img.loading="lazy";
      img.alt="witness pair "+p.pair_idx+" — comma10k 6-panel + margin/UNIWARD tribute";
      const cap=document.createElement("figcaption");cap.id="witcap-"+p.pair_idx;
      fig.appendChild(img);fig.appendChild(cap);host.appendChild(fig);
    }
    const im=$("witimg-"+p.pair_idx); if(im&&im.src!==p.panel) im.src=p.panel;
    const cp=$("witcap-"+p.pair_idx);
    if(cp) cp.innerHTML="pair "+p.pair_idx+" · realized d_seg through R <b>"+sig(p.our_dseg,5)+"</b>"+
      (p.mode?" · <b>"+escHtml(p.mode)+"</b>":"");
  });
  Array.from(host.querySelectorAll(".witfig")).forEach(f=>{if(!present.has(f.id))f.remove();});
}

// new-best d_seg celebration (tasteful; respects reduced-motion via CSS)
function celebrate(val){
  const b=$("nbest"); if(!b)return;
  b.textContent="✦ new best d_seg "+sig(val,5);
  b.classList.add("show"); _celebrating=true; scheduleDraw();
  if(_celTimer)clearTimeout(_celTimer);
  _celTimer=setTimeout(()=>{b.classList.remove("show");_celebrating=false;scheduleDraw();},5000);
}
// live verdict pulse on the liveness pill
function pulse(){const p=$("pill");if(!p||reduceMotion)return;
  p._beat=true;p.classList.remove("beat");void p.offsetWidth;p.classList.add("beat");
  setTimeout(()=>{p._beat=false;p.classList.remove("beat");},1000);}
function applySnapshot(m){TRAJ=m.trajectory||[];LIVE=m.liveness||{};META=m.meta||{};PROJ=m.projection||{};
  if(m.witness)WITNESS=m.witness;
  render();renderWitness();
  if(m.flow_ready)_onFlowReady(m.flow_ready);}
function applyUpdate(m){
  const np=m.new_points||[];
  let gotNew=false, beat=false;
  if(np.length){
    const seen=new Set(TRAJ.map(d=>d.epoch));
    np.forEach(p=>{if(!seen.has(p.epoch)){
      TRAJ.push(p);gotNew=true;
      if(bestVal!=null&&p.d_seg!=null&&isFinite(p.d_seg)&&p.d_seg<bestVal)beat=true;
    }});
    TRAJ.sort((a,b)=>a.epoch-b.epoch);
  }
  LIVE=m.liveness||LIVE;META=m.meta||META;PROJ=m.projection||PROJ;
  render();
  if(gotNew)pulse();
  if(beat)celebrate(bestVal);
}

// ---------- WebSocket (primary) ----------
function setWsPill(on){const w=$("wspill");if(on){w.className="pill ws";w.textContent="ws live";}
  else{w.className="pill wsoff";w.textContent="ws reconnecting";}}
function connectWS(){
  const proto=location.protocol==="https:"?"wss:":"ws:";
  try{ws=new WebSocket(proto+"//"+location.host+"/ws"+location.search);}catch(e){startPoll();return;}
  ws.onopen=()=>{wsOpen=true;wsTries=0;setWsPill(true);stopPoll();};
  ws.onmessage=ev=>{try{const m=JSON.parse(ev.data);
    if(m.type==="snapshot")applySnapshot(m);
    else if(m.type==="witness"){WITNESS=m.witness||{};renderWitness();}
    else if(m.type==="flow_ready"){_onFlowReady(m.flow_ready||{});}
    else applyUpdate(m);}
    catch(e){console.error("dashboard: ws message handling failed",e);}};
  ws.onclose=()=>{wsOpen=false;setWsPill(false);wsTries++;startPoll();
    setTimeout(connectWS,Math.min(15000,1000*Math.pow(1.6,Math.min(wsTries,8))));};
  ws.onerror=()=>{try{ws.close();}catch(e){console.debug("dash: ws.close after error",e);}};
}
// ---------- polling fallback (only active while WS is down) ----------
async function pollOnce(){
  if(wsOpen)return;
  try{const r=await fetch("/api/state"+location.search,{cache:"no-store"});
    if(r.ok){applySnapshot(await r.json());}}
  catch(e){console.debug("dash: poll failed (retrying; WS pill shows link state)",e);}
}
function startPoll(){if(pollTimer)return;pollTimer=setInterval(pollOnce,(BOOT.poll||5)*1000);pollOnce();}
function stopPoll(){if(pollTimer){clearInterval(pollTimer);pollTimer=null;}}

// ---------- pointer interactions: synchronized crosshair + tooltip (touch+hover) ----------
function setupInteractions(){
  const tip=$("tip");
  const canvases=["c_dseg"].map(id=>$(id)).filter(Boolean);
  function epochAt(canvas,clientX){
    const tf=canvas._tf; if(!tf)return null;
    const r=canvas.getBoundingClientRect(); const px=clientX-r.left;
    if(px<tf.x0-6||px>tf.x1+6)return null;
    return tf.xmin+(px-tf.x0)/(tf.x1-tf.x0)*(tf.xmax-tf.xmin);
  }
  function showTip(cx,cy){
    const np=nearestPoint(hoverEpoch); if(!np){hideTip();return;}
    tip.innerHTML="<div class='te'>epoch "+np.epoch+"</div>"+
      "<div class='tr'><span class='tk'>d_seg</span><span class='tv'>"+sig(np.d_seg,5)+"</span></div>"+
      "<div class='tr'><span class='tk'>d_pose</span><span class='tv'>"+sig(np.d_pose,4)+"</span></div>"+
      "<div class='tr'><span class='tk'>bytes</span><span class='tv'>"+fmtInt(np.blob_bytes)+"</span></div>"+
      "<div class='tr'><span class='tk'>implied_S</span><span class='tv'>"+sig(np.implied_S,4)+"</span></div>";
    tip.classList.add("show");
    const tw=tip.offsetWidth||180, th=tip.offsetHeight||96;
    let x=cx+14, y=cy-th-10;
    if(x+tw>window.innerWidth-8)x=cx-tw-14; if(x<8)x=8;
    if(y<8)y=cy+18; if(y+th>window.innerHeight-8)y=window.innerHeight-th-8;
    tip.style.left=x+"px"; tip.style.top=y+"px";
  }
  function hideTip(){tip.classList.remove("show");}
  function onMove(ev){
    const cx=ev.clientX, cy=ev.clientY; if(cx==null)return;
    const e=epochAt(ev.currentTarget,cx);
    if(e==null){hoverEpoch=null;hideTip();scheduleDraw();return;}
    hoverEpoch=e;showTip(cx,cy);scheduleDraw();
  }
  function onLeave(){hoverEpoch=null;hideTip();scheduleDraw();}
  canvases.forEach(c=>{
    c.addEventListener("pointermove",onMove);
    c.addEventListener("pointerdown",onMove);
    c.addEventListener("pointerleave",onLeave);
    c.addEventListener("pointercancel",onLeave);
  });
  document.addEventListener("pointerdown",ev=>{
    if(ev.target&&ev.target.tagName==="CANVAS")return;
    hoverEpoch=null;hideTip();scheduleDraw();
  });
}
// ---------- CAMPAIGN tab (#366 DDM joint-descent) ----------
// Polls /api/campaign (server-side fs snapshot, cached) every poll interval while
// the tab is active. All charts honest-axis (y from 0, targets included in range).
var campaignActive=false, campaignTimer=null, campaignData=null;
function activateCampaign(){
  campaignActive=true;
  campaignFetch();
  if(!campaignTimer)campaignTimer=setInterval(function(){
    if(campaignActive&&!document.hidden)campaignFetch();
  },Math.max(5000,(BOOT.poll||5)*1000));
}
function campaignFetch(){
  fetch("/api/campaign",{cache:"no-store"}).then(r=>r.json()).then(d=>{
    campaignData=d; renderCampaign();
  }).catch(e=>{
    const rl=$("cmp_runline");
    if(rl)rl.textContent="campaign: fetch failed — "+e;
  });
}
function cmpFmt(v,n){return (v==null||!isFinite(v))?"—":sig(v,n==null?4:n);}
function cmpCanvas(id){
  const c=$(id); if(!c)return null;
  const dpr=window.devicePixelRatio||1, w=c.clientWidth||600, h=parseInt(c.getAttribute("height"),10)||160;
  c.width=Math.round(w*dpr); c.height=Math.round(h*dpr); c.style.height=h+"px";
  const g=c.getContext("2d"); g.setTransform(dpr,0,0,dpr,0,0);
  return {c:c,g:g,w:w,h:h};
}
// generic honest line chart: series=[{pts:[{x,y,open}],color,line,mark}], hlines=[{y,color,label,dash}]
function cmpChart(id,series,hlines,opts){
  const cv=cmpCanvas(id); if(!cv)return;
  const g=cv.g, W=cv.w, H=cv.h; opts=opts||{};
  g.clearRect(0,0,W,H);
  const padL=54,padR=10,padT=8,padB=20;
  let xs=[],ys=[];
  series.forEach(s=>s.pts.forEach(p=>{if(p.x!=null&&isFinite(p.x))xs.push(p.x);
    if(p.y!=null&&isFinite(p.y)){ys.push(p.y);} if(p.y0!=null&&isFinite(p.y0))ys.push(p.y0);}));
  (hlines||[]).forEach(l=>{if(l.y!=null&&isFinite(l.y))ys.push(l.y);});
  if(!xs.length||!ys.length){g.fillStyle="#5c6573";g.font="11px ui-monospace,Menlo,monospace";
    g.fillText(opts.empty||"no data yet",padL,H/2);return;}
  let x0=Math.min.apply(null,xs),x1=Math.max.apply(null,xs);
  if(x0===x1){x0-=1;x1+=1;}
  let y1=Math.max.apply(null,ys)*1.06, y0=0;   // honest floor at 0 — never truncated
  if(y1<=y0)y1=y0+1;
  const X=x=>padL+(x-x0)/(x1-x0)*(W-padL-padR);
  const Y=y=>H-padB-(y-y0)/(y1-y0)*(H-padT-padB);
  // grid + ticks
  g.font="10px ui-monospace,Menlo,monospace";
  yTicks(y0,y1,4).forEach(t=>{const y=Y(t);
    g.strokeStyle="#2a2f39";g.lineWidth=1;g.beginPath();g.moveTo(padL,y);g.lineTo(W-padR,y);g.stroke();
    g.fillStyle="#818996";g.textAlign="right";g.fillText(fmtTick(t),padL-6,y+3);});
  yTicks(x0,x1,Math.min(8,Math.max(2,Math.round(W/110)))).forEach(t=>{const x=X(t);
    g.fillStyle="#818996";g.textAlign="center";g.fillText(String(Math.round(t)),x,H-6);});
  // reference hlines (stage targets / sealed budget)
  (hlines||[]).forEach(l=>{const y=Y(l.y);
    g.strokeStyle=l.color||"#46d369";g.lineWidth=1;
    g.setLineDash(l.dash||[5,4]);g.beginPath();g.moveTo(padL,y);g.lineTo(W-padR,y);g.stroke();g.setLineDash([]);
    if(l.label){g.fillStyle=l.color||"#46d369";g.textAlign="left";
      g.fillText(l.label,padL+4,Math.max(padT+9,y-4));}});
  // series
  series.forEach(s=>{
    if(s.bars){ // vertical initial->final ticks (descent strip)
      s.pts.forEach(p=>{if(p.y==null||p.y0==null)return;
        g.strokeStyle=(p.y<=p.y0)?"#46d369":"#ff6b6b";g.lineWidth=2;
        g.beginPath();g.moveTo(X(p.x),Y(p.y0));g.lineTo(X(p.x),Y(p.y));g.stroke();});
    }
    if(s.line!==false){
      g.strokeStyle=s.color;g.lineWidth=1.6;g.beginPath();let started=false;
      s.pts.forEach(p=>{if(p.y==null||!isFinite(p.y))return;
        if(!started){g.moveTo(X(p.x),Y(p.y));started=true;}else g.lineTo(X(p.x),Y(p.y));});
      if(started)g.stroke();
    }
    if(s.mark!==false){
      s.pts.forEach(p=>{if(p.y==null||!isFinite(p.y))return;
        g.beginPath();g.arc(X(p.x),Y(p.y),3.2,0,Math.PI*2);
        if(p.open){g.strokeStyle=s.color;g.lineWidth=1.5;g.fillStyle="#13151a";g.fill();g.stroke();}
        else{g.fillStyle=s.color;g.fill();}});
    }
  });
  function yTicks(lo,hi,n){let st=niceNum((hi-lo)/Math.max(1,n),true);if(!(st>0))st=1;
    const out=[];for(let v=Math.ceil(lo/st)*st;v<=hi+st*1e-9;v+=st)if(v>=lo-st*1e-9)out.push(v);
    return out.length?out:[lo,hi];}
  function fmtTick(v){if(Math.abs(v)>=1000)return Math.round(v).toLocaleString("en-US");
    if(Math.abs(v)>=1||v===0)return String(+v.toFixed(2));return String(+v.toPrecision(3));}
}
function renderCampaign(){
  const d=campaignData; if(!d)return;
  const rl=$("cmp_runline");
  if(!d.ok){if(rl)rl.textContent="campaign: "+(d.reason||"unavailable");return;}
  const st=d.status||{}, ck=d.checkpoints||{}, cad=d.cadence||{}, rc=d.receipt||{};
  if(rl)rl.textContent="campaign run: "+d.run_name+"  ·  "+d.run_dir;
  // status strip
  const alive=st.pid_alive?"<span class='ok'>ALIVE</span>":"<span class='bad'>DEAD</span>";
  const ended=st.ended?" <span class='warm'>(run ENDED — receipt written)</span>":"";
  const fresh=t=>t==null?"—":fmtAge(t);
  const secsVs=(cad.measured_median_s!=null)
    ? sig(cad.measured_median_s,4)+"s median vs sealed "+sig(cad.sealed_step_seconds,4)+"s"
    : "— vs sealed "+sig(cad.sealed_step_seconds,4)+"s";
  const kv=[
    ["launcher pid",(st.pid==null?"—":st.pid)+" "+alive+ended],
    ["stage","<b>"+(st.stage_index==null?"—":st.stage_index)+"</b> ("+escHtml(st.stage_id||"—")+") step <b>"+(st.stage_step==null?"—":st.stage_step)+"</b> · global <b>"+(st.global_step==null?"—":st.global_step)+"</b>"],
    ["telemetry age",fresh(st.last_telemetry_age_s)],
    ["verdict age",fresh(st.last_verdict_age_s)],
    ["run.log age",fresh(st.run_log_age_s)],
    ["accepted ckpts","<b>"+(ck.count==null?"—":ck.count)+"</b>"+(ck.latest_age_s!=null?" · latest "+fresh(ck.latest_age_s)+" ago":"")],
    ["cadence",secsVs],
  ];
  if(rc.present){kv.push(["receipt","<b>"+escHtml(rc.verdict||"—")+"</b> · pointer "+escHtml(rc.pointer||"—")+(rc.pointer_moved?"":" (UNMOVED)")]);}
  const sEl=$("cmp_status");
  if(sEl)sEl.innerHTML=kv.map(p=>"<span class='cf'><span class='ck'>"+p[0]+"</span><span class='cv'>"+p[1]+"</span></span>").join("");
  // exact verdict traces (stage verdicts + baseline; warm-start proposals excluded from
  // the trace). The stage00 baseline row carries no global_step — it IS step 0.
  const verd=(d.verdicts||[])
    .filter(v=>v.kind!=="warm_start_proposal")
    .map(v=>(v.global_step==null&&v.kind==="baseline")?Object.assign({},v,{global_step:0}):v)
    .filter(v=>v.global_step!=null)
    .sort((a,b)=>a.global_step-b.global_step);
  const stages=((d.schedule||{}).stages)||[];
  const tgtLines=stages.filter(s=>s.target_d_seg!=null).map((s,i)=>({
    y:s.target_d_seg,color:"#46d369",dash:[5,4],
    label:"stage-"+(i+1)+" target "+sig(s.target_d_seg,4)}));
  cmpChart("cmp_vseg",[{color:"#5ab0ff",
    pts:verd.map(v=>({x:v.global_step,y:v.d_seg,open:v.parameter_shadow==="live"}))}],
    tgtLines,{empty:"no exact verdicts yet"});
  const vf=$("cmp_vseg_foot");
  if(vf){const lastV=verd[verd.length-1];
    vf.textContent=lastV?("latest: step "+lastV.global_step+" d_seg "+cmpFmt(lastV.d_seg,6)
      +" (shadow="+(lastV.parameter_shadow||"?")+", decision="+(lastV.realized_stage_decision||"—")
      +", ref "+cmpFmt(lastV.reference_d_seg,6)+")"):"";}
  const tgtPose=stages.filter(s=>s.target_d_pose!=null).map(s=>({
    y:s.target_d_pose,color:"#ffb454",dash:[5,4],label:"stage-3 d_pose target "+sig(s.target_d_pose,6)}));
  cmpChart("cmp_vpose",[{color:"#ffb454",
    pts:verd.map(v=>({x:v.global_step,y:v.d_pose,open:v.parameter_shadow==="live"}))}],
    tgtPose,{empty:"no exact verdicts yet"});
  // batch-local descent strip
  const steps=d.steps||[];
  cmpChart("cmp_steps",[
    {bars:true,line:false,mark:false,color:"#46d369",
     pts:steps.map(s=>({x:s.global_step,y:s.d_seg_final,y0:s.d_seg_initial}))},
    {color:"#aeb7c6",mark:false,
     pts:steps.map(s=>({x:s.global_step,y:s.d_seg_final}))}],
    [],{empty:"no step telemetry yet"});
  cmpChart("cmp_gnorm",[{color:"#c08cff",mark:false,
    pts:steps.map(s=>({x:s.global_step,y:s.gradient_norm}))}],[],
    {empty:"no step telemetry yet"});
  cmpChart("cmp_secs",[{color:"#7fc0ff",mark:false,
    pts:steps.map(s=>({x:s.global_step,y:s.seconds}))}],
    [{y:cad.sealed_step_seconds,color:"#e6cf7a",label:"sealed "+sig(cad.sealed_step_seconds,4)+" s/step"}],
    {empty:"no step telemetry yet"});
  const sf=$("cmp_secs_foot");
  if(sf)sf.textContent=(cad.measured_n?("measured n="+cad.measured_n+" · median "+cmpFmt(cad.measured_median_s,4)
    +"s · mean "+cmpFmt(cad.measured_mean_s,4)+"s · last "+cmpFmt(cad.measured_last_s,4)+"s"):"")
    +"  ·  sealed source: "+(cad.sealed_source||"");
  // pose-finish engage gate
  const lastVerd=verd[verd.length-1]||{}, eng=lastVerd.engage||{};
  const pfe=((d.schedule||{}).pose_finish_engage)||{};
  const gEl=$("cmp_gate");
  if(gEl){
    const nVerd=(eng.exact_verdict_steps||[]).length;
    const settle=pfe.settle_window, hyst=pfe.hysteresis;
    const windowNote=(settle!=null&&hyst!=null)
      ? "detector: "+escHtml(pfe.detector||"plateau")+" needs settle_window="+settle+" + hysteresis="+hyst
        +" exact verdicts → earliest engagement ~verdict "+(settle+hyst-1)+"–"+(settle+hyst)
      : "";
    gEl.innerHTML=
      "state <b>"+escHtml(eng.classification||"—")+"</b><br>"+
      "engaged_global_step <b>"+(eng.engaged_global_step==null?"not engaged":eng.engaged_global_step)+"</b>"+
      " · exact verdicts <b>"+nVerd+"</b> @ steps ["+(eng.exact_verdict_steps||[]).join(", ")+"]<br>"+
      "exact d_seg history: "+((eng.exact_d_seg||[]).map(v=>sig(v,5)).join(" → ")||"—")+"<br>"+
      "relative slope <b>"+(eng.latest_relative_slope==null?"—":sig(eng.latest_relative_slope,4))+"</b>"+
      " · strict seg admissions <b>"+(eng.strict_seg_admissions==null?"—":eng.strict_seg_admissions)+"</b><br>"+
      "<span style='color:var(--faint2)'>"+windowNote+"</span>";
  }
  // per-class bars — latest exact verdict
  const cEl=$("cmp_cls");
  if(cEl){
    const pc=lastVerd.per_class||{};
    const order=(d.class_order||[]).filter(k=>pc[k]);
    if(!order.length){cEl.textContent="no exact verdict yet";}
    else{
      const mx=Math.max.apply(null,order.map(k=>pc[k].d_seg||0))||1;
      cEl.innerHTML=order.map(k=>{
        const v=pc[k].d_seg||0, w=Math.max(1,Math.round(v/mx*100));
        return "<div class='clsrow'><span class='clsname'>"+k+"</span>"+
          "<span class='clsbarwrap'><span class='clsbar' style='width:"+w+"%'></span></span>"+
          "<span class='clsval'>"+sig(v,5)+"</span></div>";}).join("");
      const cfEl=$("cmp_cls_foot");
      if(cfEl)cfEl.textContent="from "+(lastVerd.file||"latest verdict")+" · step "
        +lastVerd.global_step+" · shadow="+(lastVerd.parameter_shadow||"?")
        +" · linear scale, common max "+sig(mx,5);
    }
  }
}
window.addEventListener("resize",function(){if(campaignActive)renderCampaign();});

setupInteractions();
window.addEventListener("resize",scheduleDraw);
window.addEventListener("orientationchange",scheduleDraw);
connectWS();
// safety net: if WS never opens within 4s, ensure polling is running
setTimeout(()=>{if(!wsOpen)startPoll();},4000);
// ORACLE is the default-visible tab (arc entry point) -> kick its one-shot fetch on load.
try{activateOracle();}catch(e){console.error("dash: activateOracle on load failed",e);}
</script>
<script>
__FLOW_JS__
</script>
<script>
__WHYHOW_JS__
</script>
</body></html>
"""


# ───────────────────────── app instance (for `uvicorn dashboard_server:app`) ─────────────────────────
app = create_app(config_from_env())


def main() -> None:
    import uvicorn

    cfg = config_from_env()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", default=cfg.run_dir)
    ap.add_argument("--log-glob", default=cfg.log_glob)
    ap.add_argument("--tau", type=int, default=cfg.tau,
                    help="OVERRIDE only (deprecated as a required input); default = derived "
                         "from the run's config via witness_dsl schedule read-back")
    ap.add_argument("--l7", type=int, default=cfg.l7,
                    help="OVERRIDE only (deprecated as a required input); default = derived "
                         "from the run's config via witness_dsl schedule read-back "
                         "(a disabled l7 — start >= epochs — is omitted, never labeled)")
    ap.add_argument("--goal-dseg", type=float, default=cfg.goal_dseg,
                    help="OVERRIDE only; default = derived per run as "
                         "(0.19 − measured pose − measured rate)/100 from the run's own verdicts")
    ap.add_argument("--goal-dseg-15", type=float, default=cfg.goal_dseg_15,
                    help="OVERRIDE only; default = derived per run against the sub-0.15 target")
    ap.add_argument("--poll", type=float, default=cfg.poll)
    ap.add_argument("--host", default=cfg.host)
    ap.add_argument("--port", type=int, default=cfg.port)
    ap.add_argument("--access-key", default=cfg.access_key)
    ap.add_argument("--cadence-state", default=cfg.cadence_state)
    ap.add_argument("--training-pid", type=int, default=cfg.training_pid)
    ap.add_argument("--training-sig", default=cfg.training_sig)
    ap.add_argument("--auto-latest", action=argparse.BooleanOptionalAction, default=cfg.auto_latest,
                    help="follow the freshest witness arm across all dirs (default ON; --no-auto-latest to pin --run-dir)")
    ap.add_argument("--auto-base-glob", default=cfg.auto_base_glob)
    ap.add_argument("--witness", action=argparse.BooleanOptionalAction, default=cfg.witness_enable,
                    help="Tab-2 WITNESS live panels (default ON; --no-witness to disable the render)")
    ap.add_argument("--witness-gt-cache", default=cfg.witness_gt_cache,
                    help="GT cache for the 600-pass (needs all 600 pairs: gt_n600.npz)")
    ap.add_argument("--witness-ema-name", default=cfg.witness_ema_name,
                    help="fallback EMA npz if the BEST checkpoint is absent")
    ap.add_argument("--witness-min-free-gib", type=float, default=cfg.witness_min_free_gib)
    ap.add_argument("--witness-dpi", type=int, default=cfg.witness_dpi)
    ap.add_argument("--flow", action=argparse.BooleanOptionalAction, default=cfg.flow_enable,
                    help="Tab-3 FLOW n600 video + Tab-2 hardest pairs (default ON; --no-flow to disable)")
    ap.add_argument("--flow-best-ema-name", default=cfg.flow_best_ema_name,
                    help="the checkpoint the 600-pass renders (default levelset_witness_ema_BEST.npz)")
    ap.add_argument("--flow-seq-downsample", type=int, default=cfg.flow_seq_downsample,
                    help="field downsample for the video (default 4 -> 96x128)")
    ap.add_argument("--flow-seq-hard-k", type=int, default=cfg.flow_seq_hard_k,
                    help="hardest/most-diverse Tab-2 pairs to select (default 6)")
    ap.add_argument("--flow-seq-min-interval-s", type=float, default=cfg.flow_seq_min_interval_s)
    ap.add_argument("--reuse-port", action=argparse.BooleanOptionalAction, default=True,
                    help="SO_REUSEPORT (default ON) so a NEW instance can bind :port alongside the OLD "
                         "one -> zero-downtime hot reload (tools/dashboard_reload.py) that never drops the "
                         "cloudflared tunnel origin. --no-reuse-port to disable.")
    a = ap.parse_args()
    cfg = Config(run_dir=a.run_dir, log_glob=a.log_glob, tau=a.tau, l7=a.l7,
                 goal_dseg=a.goal_dseg, goal_dseg_15=a.goal_dseg_15, poll=a.poll,
                 host=a.host, port=a.port, access_key=a.access_key,
                 cadence_state=a.cadence_state, training_pid=a.training_pid,
                 training_sig=a.training_sig,
                 auto_latest=a.auto_latest, auto_base_glob=a.auto_base_glob,
                 witness_enable=a.witness, witness_gt_cache=a.witness_gt_cache,
                 witness_ema_name=a.witness_ema_name,
                 witness_min_free_gib=a.witness_min_free_gib, witness_dpi=a.witness_dpi,
                 flow_enable=a.flow, flow_best_ema_name=a.flow_best_ema_name,
                 flow_seq_downsample=a.flow_seq_downsample, flow_seq_hard_k=a.flow_seq_hard_k,
                 flow_seq_min_interval_s=a.flow_seq_min_interval_s)
    application = create_app(cfg)
    # access_log=False so the ?k=<access key> never lands in a log line.
    # SO_REUSEPORT zero-downtime hot reload: uvicorn 0.44's run()/Config has no reuse_port kwarg, so we
    # bind the socket OURSELVES with SO_REUSEPORT and hand it to Server.run(sockets=[...]). That lets a
    # fresh (new-code) instance co-bind :port WHILE the old one still serves, so tools/dashboard_reload.py
    # brings the new up + healthz-confirms it BEFORE retiring the old -> the :port listener is never empty
    # -> the cloudflared named tunnel origin sees no 502 window. Falls back to plain run on any error.
    cfgu = uvicorn.Config(application, host=cfg.host, port=cfg.port, log_level="warning",
                          access_log=False, ws="websockets")
    server = uvicorn.Server(cfgu)
    sock = None
    if bool(a.reuse_port):
        try:
            import socket as _socket
            sock = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
            sock.setsockopt(_socket.SOL_SOCKET, _socket.SO_REUSEADDR, 1)
            if hasattr(_socket, "SO_REUSEPORT"):
                sock.setsockopt(_socket.SOL_SOCKET, _socket.SO_REUSEPORT, 1)
            sock.bind((cfg.host, cfg.port))
            sock.set_inheritable(True)
        except OSError:
            if sock is not None:
                sock.close()
            sock = None  # fall back to uvicorn binding the port itself
    server.run(sockets=[sock] if sock is not None else None)


if __name__ == "__main__":
    main()
