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
0.19110 and UNMOVED — a dashboard is a MEANS, not the score.

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

    .venv/bin/python tools/dashboard_server.py \
        --run-dir experiments/results/levelset_openpilot_seeded_n200_DEPLOY \
        --port 8790 --tau 300 --l7 600
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import hmac
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

# ── reuse the canonical verdict-parse + self-calibrating liveness (DRY) ──
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import render_levelset_dashboard as rld  # noqa: E402
import dashboard_trajectory_model as dtm  # noqa: E402  (sophisticated DATA-DERIVED projection)

from starlette.applications import Starlette  # noqa: E402
from starlette.responses import (  # noqa: E402
    HTMLResponse,
    JSONResponse,
    PlainTextResponse,
)
from starlette.routing import Route, WebSocketRoute  # noqa: E402
from starlette.websockets import WebSocket, WebSocketDisconnect  # noqa: E402

POINTER = 0.19110  # frontier pointer (contest-CPU), UNMOVED — advisory only here

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
    tau: int = 300
    l7: int = 600
    goal_dseg: float = 0.00092  # sub-0.19 d_seg goal line
    goal_dseg_15: float = 0.00032  # sub-0.15 d_seg goal line
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
    auto_base_glob: str = "experiments/results/levelset_*/*.log"
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
    witness_min_free_gib: float = 5.0    # skip spawning a pass when free RAM is below this (yield to #205)
    witness_dpi: int = 80
    flow_enable: bool = True
    flow_best_ema_name: str = "levelset_witness_ema_BEST.npz"  # the checkpoint the 600-pass renders
    flow_seq_downsample: int = 4         # field downsample (384x512 -> 96x128) for the video payload
    flow_seq_jpeg_q: int = 62            # JPEG quality for the per-frame witness-render layer
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
            return self.auto_base_glob
        if self.run_dir:                # pinned mode (--no-auto-latest --run-dir X)
            return os.path.join(self.run_dir, "*.log")
        return ".omx/tmp/levelset_*.log"


def config_from_env() -> Config:
    e = os.environ.get
    return Config(
        run_dir=e("DASH_RUN_DIR", ""),
        log_glob=e("DASH_LOG_GLOB", ""),
        tau=int(e("DASH_TAU", "300")),
        l7=int(e("DASH_L7", "600")),
        goal_dseg=float(e("DASH_GOAL_DSEG", "0.00092")),
        goal_dseg_15=float(e("DASH_GOAL_DSEG_15", "0.00032")),
        poll=float(e("DASH_POLL", "5.0")),
        host=e("DASH_HOST", "127.0.0.1"),
        port=int(e("DASH_PORT", "8790")),
        access_key=e("DASH_ACCESS_KEY", ""),
        cadence_state=e("DASH_CADENCE_STATE", ".omx/tmp/dash_levelset_deploy/cadence.json"),
        training_pid=int(e("DASH_TRAINING_PID", "0")),
        training_sig=e("DASH_TRAINING_SIG", "train_levelset_witness"),
        auto_latest=e("DASH_AUTO_LATEST", "1") not in ("0", "false", "False"),
        auto_base_glob=e("DASH_AUTO_BASE_GLOB", "experiments/results/levelset_*/*.log"),
        witness_enable=e("DASH_WITNESS_ENABLE", "1") not in ("0", "false", "False"),
        witness_gt_cache=e("DASH_WITNESS_GT_CACHE", "experiments/results/mlx_fleet_gt_cache/gt_n600.npz"),
        witness_ema_name=e("DASH_WITNESS_EMA_NAME", "levelset_witness_ema_mlx.npz"),
        witness_min_free_gib=float(e("DASH_WITNESS_MIN_FREE_GIB", "5.0")),
        witness_dpi=int(e("DASH_WITNESS_DPI", "80")),
        flow_enable=e("DASH_FLOW_ENABLE", "1") not in ("0", "false", "False"),
        flow_best_ema_name=e("DASH_FLOW_BEST_EMA_NAME", "levelset_witness_ema_BEST.npz"),
        flow_seq_downsample=int(e("DASH_FLOW_SEQ_DOWNSAMPLE", "4")),
        flow_seq_jpeg_q=int(e("DASH_FLOW_SEQ_JPEG_Q", "62")),
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


_TRAJ_KEYS = ("epoch", "d_seg", "d_pose", "blob_bytes", "implied_S",
              "implied_S_monitoring", "ts")


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
                                  archive_norm: float) -> dict:
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
    an un-entered stage reports 'not entered'. The frontier pointer (0.19110) moves ONLY
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
                             "implied_s": s_proj}

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
    numeric; missing keys become None."""
    out = {k: row.get(k) for k in ("epoch", "d_seg", "d_pose", "blob_bytes", "ts")}
    out["implied_S"] = row.get("implied_S")
    out["implied_S_monitoring"] = row.get("implied_S")
    return out


def _pid_alive(pid: int) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
        return True
    except Exception:
        return False


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
        for line in open(log_path, encoding="utf-8", errors="replace"):
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
        for line in open(log_path, encoding="utf-8", errors="replace"):
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
        for line in open(log_path, encoding="utf-8", errors="replace"):
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
        for line in open(path, encoding="utf-8", errors="replace"):
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
        for line in open(path, encoding="utf-8", errors="replace"):
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
        gauge = w.CANONICAL_GAUGE
        gsum = {}
        for f in ("warp", "carrier", "residual"):
            v = getattr(gauge, f, None)
            gsum[f] = getattr(v, "value", str(v)) if v is not None else None
        return {"ok": True,
                "program": {"epochs": getattr(prog, "epochs", None),
                            "num_pairs": getattr(prog, "num_pairs", None),
                            "stages": stages},
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
    data = {
        "ok": True,
        "built_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "pointer": POINTER,
        "dag": _latest_dag_feeds(),
        "dsl": _dsl_summary(),
        "equations": _latest_equations(),
        "authority": "macOS-MLX training advisory · NON-PROMOTABLE",
    }
    _TRIALITY_CACHE.update(ts=now, data=data)
    return data


# ───────────────────────── access gate (pure, testable) ─────────────────────────
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
        self.liveness: dict = {"kind": "missing"}
        self.watched: str | None = None
        self.watched_dir: str | None = None  # live arm dir (auto-latest); shown as run_dir
        self.muon_start: int | None = None    # inferred l7 -> Muon boundary (additive meta)
        self.n_pairs: int | None = None       # N for this run (n200 DOE pilot / n600 scored)
        self.warming_up: bool = False         # live run resolved but no verdict yet (structured-init)
        self.run_config: dict = {}            # parsed launch.sh/run.log config + curriculum schedule
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
                try:
                    warming = run_latest.stat().st_mtime >= verdict_latest.stat().st_mtime
                except OSError:
                    warming = True
        if warming:
            latest = run_latest
            chain = [run_latest]              # its own (0) verdicts — never a foreign trajectory
        else:
            latest = verdict_latest
            # FULL trajectory across the warm-start chain (de-dup by epoch; later arm wins
            # at boundary collisions since the chain is ordered root..latest).
            chain = _resume_chain_logs(latest)
        self.warming_up = warming
        merged: dict[int, dict] = {}
        for lg in chain:
            for r in rld._parse_verdicts(lg):
                ep = r.get("epoch")
                if isinstance(ep, int):
                    merged[ep] = _slim(r)
        rows_full = [merged[e] for e in sorted(merged)]
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
        # CONFIG + CURRICULUM SCHEDULE (full observability): parse the live run's OWN
        # artifacts (launch.sh primary, run.log stages fallback) — generalizable to
        # ANY future run with zero hand-config (operator "automatically in the future").
        # Parsed BEFORE liveness so the stage-aware cadence/ETA can use the real schedule.
        try:
            self.run_config = rld.parse_run_config(latest.parent if latest is not None else None)
        except Exception:
            self.run_config = {"source": "none", "flags": {}, "groups": {}, "schedule": {}}

        # Effective curriculum schedule (boundaries the cadence + projection use): the
        # run's OWN flags, with the Muon boundary from the resume ancestry when present.
        _sched = (self.run_config or {}).get("schedule", {}) or {}
        eval_every = _sched.get("eval_every")
        sched_eff = {
            "tau_start": _sched.get("tau_start"), "l7_start": _sched.get("l7_start"),
            "muon_start": (self.muon_start if self.muon_start is not None else _sched.get("muon_start")),
            "epochs": _sched.get("epochs"), "eval_every": eval_every,
            "goal_dseg": cfg.goal_dseg, "goal_dseg_15": cfg.goal_dseg_15,
        }

        # liveness + cadence from the LIVE (latest) arm only (the freshest log).
        now = time.time()
        latest_rows = rld._parse_verdicts(latest) if latest is not None else []
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
        self.watched_dir = str(latest.parent) if latest is not None else None

        # #205 run-info strip: the FULL operational telemetry (stage-progress / best-d_seg
        # deploy / throughput+ETA / checkpoint ledger / resumability-now / MLX fast-path /
        # config) the standalone HTML surfaces. REUSE rld's crash-safe collectors + renderer
        # verbatim (already imported), rendered HERE in the executor thread so the event loop
        # never formats it. No run resolved -> "" (strip hidden; page unchanged = back-compat).
        if latest is not None:
            try:
                self.run_info = rld._collect_run_info(
                    latest, glob, cfg.run_dir, rows_full, self.liveness, now)
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
        # NOT model. ADVISORY; the pointer (0.19110) moves ONLY through a byte-closed exact eval.
        if isinstance(self.projection, dict):
            try:
                self.projection["stage_proj"] = _build_stage_aware_projection(
                    rows_full, sched_eff,
                    sidecar_pose=(_last_measured_dpose(rows_full) or 0.0),
                    archive_norm=_ARCHIVE_NORM_BYTES)
            except Exception as exc:
                self.projection["stage_proj"] = {"ok": False, "reason": f"stage projection error: {exc}"}

        # Rebuild trajectory from the full chain each tick (cheap; few hundred points).
        # new_points = epochs not yet pushed to WS clients (snapshot carries the full set).
        new_points = [p for p in rows_full
                      if isinstance(p.get("epoch"), int) and p["epoch"] not in self._epochs]
        for p in new_points:
            self._epochs.add(p["epoch"])
        self.trajectory = rows_full

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
            "--rss-mb", str(cfg.oracle_rss_mb), "--timeout", str(cfg.oracle_timeout_s),
            "--label", "oracle_atlas", "--", *inner,
        ]
        log_path = stem.with_suffix(".log")
        try:
            logf = open(log_path, "ab")
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
            "--rss-mb", str(cfg.whyhow_rss_mb), "--timeout", str(cfg.whyhow_timeout_s),
            "--label", "whyhow_fields", "--", *inner,
        ]
        log_path = stem.with_suffix(".log")
        try:
            logf = open(log_path, "ab")
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
            return psutil.virtual_memory().available / (1024 ** 3)
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
        if not self._flowseq_running and _flowseq_lock_alive(stem.with_suffix(".lock")):
            return
        # 2) a subprocess in flight (this instance)?
        if self._flowseq_running:
            rc = self._flowseq_proc.poll() if self._flowseq_proc is not None else 0
            if rc is None:
                return  # still rendering (a full n600 pass is ~14 min)
            self._flowseq_running = False
            if (self._flowseq_stem and (self._flowseq_stem.with_suffix(".done")).exists()):
                self._ingest_flowseq(self._flowseq_stem, self._flowseq_target_mtime)
            else:
                self._flowseq_err = f"flow-seq subprocess exited rc={rc} without output"
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
            "--rss-mb", str(cfg.flow_seq_rss_mb), "--timeout", str(cfg.flow_seq_timeout_s),
            "--label", f"flow_seq_{int(mtime)}", "--", *inner,
        ]
        log_path = stem.with_suffix(".log")
        try:
            logf = open(log_path, "ab")
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
        """Lightweight FLOW readiness ping (the client fetches /api/flow_sequence on this)."""
        if self._flow_seq_bytes is not None and self._flow_seq_meta:
            return {"ok": True, **self._flow_seq_meta}
        status = ("rendering" if self._flowseq_running
                  else ("error" if self._flowseq_err else "idle"))
        return {"ok": False, "status": status, "err": self._flowseq_err or None}

    def flow_ready_msg(self) -> dict:
        return {"type": "flow_ready", "flow_ready": self._flow_ready_public()}

    def consume_witness_dirty(self) -> bool:
        """Return+clear the witness-dirty flag (single-threaded event loop -> race-free)."""
        if self._witness_dirty:
            self._witness_dirty = False
            return True
        return False

    # ---- snapshot for client ----
    def meta(self) -> dict:
        cfg = self.cfg
        sched = (self.run_config or {}).get("schedule", {}) or {}
        # Curriculum boundaries: prefer the run's OWN flags (the actual schedule) over
        # the dashboard launch defaults; Muon boundary from the resume ancestry OR the
        # --muon-start-epoch flag (from-scratch curriculum runs have no resume).
        tau = sched.get("tau_start") if sched.get("tau_start") is not None else cfg.tau
        l7 = sched.get("l7_start") if sched.get("l7_start") is not None else cfg.l7
        muon = self.muon_start if self.muon_start is not None else sched.get("muon_start")
        return {
            "tau": tau, "l7": l7,
            "goal_dseg": cfg.goal_dseg, "goal_dseg_15": cfg.goal_dseg_15,
            "pointer": POINTER, "watched": self.watched,
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
            except asyncio.TimeoutError:
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
                          "access_gated": bool(cfg.access_key)}), flush=True)
        try:
            yield
        finally:
            stop.set()
            task.cancel()
            with contextlib.suppress(Exception):
                await task

    async def index(request):
        qk, ck = _req_keys(request)
        if gate_decision(dict(request.headers), qk, ck, cfg.access_key) == "deny":
            return HTMLResponse(_login_html(), status_code=401)
        resp = HTMLResponse(_page_html(cfg))
        # Any client that passed the page gate (local bypass OR valid key) gets the
        # cookie, so its same-origin WebSocket (which is gated strictly, no local
        # bypass) authenticates via the cookie the browser sends automatically.
        if cfg.access_key:
            resp.set_cookie("dash_key", cfg.access_key, max_age=86400 * 7,
                            httponly=True, samesite="lax", path="/")
        return resp

    async def api_state(request):
        qk, ck = _req_keys(request)
        if gate_decision(dict(request.headers), qk, ck, cfg.access_key) == "deny":
            return JSONResponse({"error": "access key required"}, status_code=401)
        return JSONResponse(state.snapshot())

    async def api_flow_sequence(request):
        # The Tab-3 n600 VIDEO payload (fetched ONCE per new sequence; the client then scrubs +
        # plays locally). Served as pre-built bytes so the ~7 MB dict is never re-serialized.
        qk, ck = _req_keys(request)
        if gate_decision(dict(request.headers), qk, ck, cfg.access_key) == "deny":
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
        if gate_decision(dict(request.headers), qk, ck, cfg.access_key) == "deny":
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
        if gate_decision(dict(request.headers), qk, ck, cfg.access_key) == "deny":
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
        if gate_decision(dict(request.headers), qk, ck, cfg.access_key) == "deny":
            return JSONResponse({"error": "access key required"}, status_code=401)
        try:
            data = await asyncio.get_event_loop().run_in_executor(None, triality_snapshot)
        except Exception as exc:
            return JSONResponse({"ok": False, "reason": f"triality error: {exc}"}, status_code=200)
        return JSONResponse(data, headers={"Cache-Control": "no-store"})

    async def healthz(request):
        # ungated, reveals nothing sensitive (used by the supervisor + tunnel health)
        return JSONResponse({"ok": True, "watched": state.watched,
                             "n_points": len(state.trajectory),
                             "kind": state.liveness.get("kind")})

    async def ws_endpoint(ws: WebSocket):
        qk = ws.query_params.get("k")
        ck = ws.cookies.get("dash_key")
        # strict_local=True: WS ALWAYS requires the key when configured (no cf-header
        # local bypass — cloudflared omits Cf-Ray on the WS upgrade).
        if gate_decision(dict(ws.headers), qk, ck, cfg.access_key, strict_local=True) == "deny":
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
        Route("/healthz", healthz),
        WebSocketRoute("/ws", ws_endpoint),
    ]
    return Starlette(routes=routes, lifespan=lifespan)


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
    boot = json.dumps({
        "tau": cfg.tau, "l7": cfg.l7,
        "goal_dseg": cfg.goal_dseg, "goal_dseg_15": cfg.goal_dseg_15,
        "pointer": POINTER, "poll": cfg.poll,
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
.wrap{max-width:1200px;margin:0 auto;padding:clamp(14px,3.5vw,26px) clamp(12px,4vw,24px) 56px}

/* advisory banner */
.auth{background:linear-gradient(180deg,#241b0d,#1f180c);border:1px solid #4a3a12;color:#e6cf7a;
font-size:clamp(11px,3vw,12.5px);padding:10px 14px;border-radius:10px;margin-bottom:16px;
text-align:center;line-height:1.55}
.auth b{color:#f4dd8a}

/* header */
.head{display:flex;align-items:center;flex-wrap:wrap;gap:10px;margin-bottom:4px}
.title{font-size:clamp(13px,3.4vw,15px);color:var(--fg);letter-spacing:1.4px;
text-transform:uppercase;font-weight:700;margin-right:auto}
.pills{display:flex;gap:6px;flex-wrap:wrap}
.pill{font-size:11.5px;font-weight:600;padding:4px 11px;border-radius:999px;white-space:nowrap;line-height:1.3}
.pill.live{background:#173d22;color:#7fe0a0}.pill.warm{background:#3a3413;color:#e6cf7a}
.pill.stale{background:#4a1717;color:#ff9b9b}.pill.miss{background:#3a1f1f;color:#ff9b9b}
.pill.ws{background:#16263a;color:#7fc0ff}.pill.wsoff{background:#3a2a16;color:#e6b97a}

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
.orcgrid{display:flex;flex-direction:column;gap:16px;margin:14px 0 6px}
.orcfig{margin:0;background:var(--panel);border:1px solid var(--grid);border-radius:12px;padding:10px;overflow-x:auto}
.orcfig img{display:block;width:100%;max-width:100%;height:auto;border-radius:8px}
.orcfig figcaption{font-size:11px;color:var(--muted);margin-top:7px;font-variant-numeric:tabular-nums}
.orcxi{margin-top:20px}
.orcxi img{display:block;width:100%;max-width:100%;height:auto;border-radius:10px;border:1px solid var(--grid);background:var(--panel);margin-top:8px}
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
.witcredits .wcnote{font-size:11px;color:#b89a4a;line-height:1.55;border-top:1px solid var(--grid);padding-top:9px}

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
padding:20px;font-size:12.5px;color:var(--muted);line-height:1.6;pointer-events:none}
.flowmsg.hide{display:none}
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
.nbadge{font-size:11.5px;font-weight:700;padding:4px 11px;border-radius:999px;white-space:nowrap;letter-spacing:.3px;line-height:1.3}
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

/* stage legend strip */
.slegend{display:flex;flex-wrap:wrap;gap:7px 13px;align-items:center;margin:2px 2px 12px;font-size:11px;color:var(--muted)}
.slegend .sc{display:inline-flex;align-items:center;gap:5px;white-space:nowrap}
.slegend .dot{width:9px;height:9px;border-radius:2px;display:inline-block;flex:0 0 auto}
.slegend .sc.off{opacity:.38}

/* projection block (naive linear, advisory) */
.proj{font-size:11.5px;color:var(--muted);margin:0 2px 16px;line-height:1.6;min-height:34px}
.proj .proj2{color:var(--faint2);font-size:11px}
.proj b{color:var(--fg2);font-weight:600;font-variant-numeric:tabular-nums}

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
.hide{display:none}
</style></head>
<body><div class="wrap">
<div class="auth">[macOS-MLX training] advisory &middot; <b>NON-PROMOTABLE</b> &mdash; the exact contest row is byte-closed on contest-CPU/CUDA &middot; frontier pointer <b>0.19110</b> (UNMOVED). A dashboard is a MEANS, not the score.</div>
<div class="head">
  <span class="title">Level-Set Witness</span>
  <span class="pills">
    <span id="npill" class="nbadge other">n=?</span>
    <span id="pill" class="pill miss">&middot; connecting</span>
    <span id="wspill" class="pill wsoff">ws &hellip;</span>
  </span>
</div>
<div class="tabs">
  <div class="tab on" data-tab="oracle">ORACLE</div>
  <div class="tab" data-tab="flow">WITNESS</div>
  <div class="tab" data-tab="witness">RESIDUAL</div>
  <div class="tab" data-tab="whyhow">WHY / HOW</div>
  <div class="tab" data-tab="live">DESCENT</div>
  <div class="tab" data-tab="tri">TRIALITY</div>
</div>

<section id="tab-oracle" class="orc">
  <div class="orcintro">
    <h2>The detector I built &mdash; and the world it reads</h2>
    <p>Before the witness, the <b>oracle</b>: the frozen contest scorer (Yousfi's comma10k
    EfficientNet-B2 SegNet, whose argmax IS the d_seg authority), the <b>openpilot physical
    priors</b> that structure the scene, and the scorer's own <b>detectability field</b>. This is
    the world the witness must survive in. The thesis in one line:
    <b>openpilot is the unified FREE physical prior for BOTH scored axes</b> &mdash; the lane
    polynomial &rarr; the analytic lane band (<b>d_seg</b>), and the ego-motion screw &xi; &rarr;
    the pose (<b>d_pose</b>). Every prior below is a REAL in-tree module, self-detecting its
    classes (no hardcoded indices), fit to the EXACT frames the scorer's verdict uses.</p>
    <div class="orchdr" id="orchdr">rendering the physical-prior atlas (governed CPU pass)&hellip;</div>
  </div>
  <div class="orcstats" id="orcstats"></div>
  <div class="orcgrid" id="orcpanels"></div>
  <div class="orcxi">
    <h3>Ego-&xi; &mdash; the SE(3) screw twist across the segment</h3>
    <p>The lane-optimal per-pair ego trajectory (<code>LaneOptimalEgoEstimator</code> over the
    openpilot lane fit). The <b>same twist &xi;</b> that warps the partition for d_seg IS the pose
    for d_pose (Chasles' theorem) &mdash; encode it once, spend it twice.</p>
    <img id="orcxichart" alt="ego-ξ screw twist across the segment" />
  </div>
  <div class="orccredits">
    <div class="wch">Physical priors wired &mdash; de-orphaned, not rebuilt</div>
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
    The exact row is byte-closed on contest-CPU/CUDA; the frontier pointer is 0.19110 and UNMOVED.</div>
  </div>
</section>

<section id="tab-live" class="hide">
  <div class="runinfo" id="rdinfo">resolving run&hellip;</div>
  <div class="metrics" id="headline">
    <div class="stat hero">
      <span class="slabel">d_seg <span class="trend fl" id="m_trend">&middot;</span></span>
      <span class="sval" id="d_seg_val">&mdash;</span>
      <span class="ssub" id="m_goal">&nbsp;</span>
      <span class="ssub" id="m_best">&nbsp;</span>
    </div>
    <div class="stat">
      <span class="slabel">d_pose <span class="trend fl" id="p_trend">&middot;</span></span>
      <span class="sval2" id="d_pose_val">&mdash;</span>
      <span class="ssub">PoseNet MSE</span>
    </div>
    <div class="stat">
      <span class="slabel">bytes <span class="trend fl" id="b_trend">&middot;</span></span>
      <span class="sval2" id="bytes_val">&mdash;</span>
      <span class="ssub">learned payload</span>
    </div>
    <div class="stat">
      <span class="slabel">implied_S <span class="trend fl" id="s_trend">&middot;</span></span>
      <span class="sval2" id="s_val">&mdash;</span>
      <span class="ssub adv">advisory &middot; measured d_seg + store-nothing-carrier d_pose + rate &middot; NOT the 0.19110 pointer</span>
    </div>
  </div>
  <div class="sbreak" id="sbreak">
    <div class="sbh">Scorer breakdown <span class="sbtag">honest &middot; measured pose</span></div>
    <div class="sbformula">S = 100&middot;d_seg + &radic;(10&middot;d_pose) + 25&middot;(bytes / 37,545,489)</div>
    <div class="sbsubst" id="sb_subst">&nbsp;</div>
    <div class="sbterms" id="sb_terms"></div>
    <div class="sbdeploy" id="sb_deploy">&nbsp;</div>
  </div>
  <div class="status" id="status">connecting&hellip;</div>
  <div class="detail" id="detail">&nbsp;</div>
  <div class="slegend" id="slegend">
    <span class="sc" data-st="ce"><span class="dot" style="background:#5ab0ff"></span>CE</span>
    <span class="sc" data-st="tau"><span class="dot" style="background:#b08cff"></span>tau</span>
    <span class="sc" data-st="l7"><span class="dot" style="background:#ffa454"></span>l7</span>
    <span class="sc off" data-st="muon"><span class="dot" style="background:#46d3a0"></span>Muon</span>
    <span class="sc" style="margin-left:auto"><span class="dot" style="background:rgba(226,232,240,.45)"></span>EMA</span>
    <span class="sc"><span class="dot" style="border:1px dashed #9aa3b2;background:transparent;border-radius:0"></span>trend</span>
    <span class="sc"><span class="dot" style="background:#ffd24a;border-radius:50%"></span>best</span>
  </div>
  <div class="proj" id="proj"><div id="proj_seg">&nbsp;</div><div class="proj2" id="proj_s">&nbsp;</div></div>
  <details class="cfg" id="cfgpanel" open>
    <summary id="cfgsum">setup &middot; config &middot; schedule &middot; curriculum</summary>
    <div class="cfgbody" id="cfgbody"><div class="cfgmeta">parsing run config&hellip;</div></div>
  </details>
  <div class="rinfo" id="runinfostrip"></div>
  <div class="grid">
    <div class="panel"><canvas id="c_dseg" role="img" aria-label="d_seg chart"></canvas></div>
    <div class="panel"><canvas id="c_dpose" role="img" aria-label="d_pose chart"></canvas></div>
    <div class="panel"><canvas id="c_bytes" role="img" aria-label="blob bytes chart"></canvas></div>
    <div class="panel"><canvas id="c_s" role="img" aria-label="implied S chart"></canvas></div>
  </div>
  <div class="foot" id="foot"></div>
</section>

<section id="tab-witness" class="wit hide">
  <div class="witintro">
    <h2>The scorer's eye &mdash; the hardest &amp; most-diverse pairs</h2>
    <p>The pairs with the <b>highest realized d_seg</b> across the whole n600 drive, chosen to be
    <b>spread across the segment</b> (not clustered) and to cover distinct failure modes &mdash; each
    labelled with its <b>per-pair d_seg + failure-mode tag</b> (adjacent/movable vehicle, lane-marking
    dash, distant object, or boundary flips, read honestly from the disagreement composition).
    Selected from the SAME governed 600-pass that builds the FLOW video, refreshed when a new best
    checkpoint lands. Each pair is the canonical comma multipane: <b>Row A</b> GT &middot; our witness
    render (through the contest R operator) &middot; pixel error; <b>Row B</b> the SegNet argmax the
    scorer sees for GT vs our render, and their disagreement (the d_seg pixels); <b>Row C</b> the
    tribute triptych.</p>
    <div class="withdr" id="withdr">selecting the hardest pairs from the n600 pass&hellip;</div>
  </div>
  <div class="witgrid" id="witpanels"></div>
  <div class="wittrip">
    <h3>Row C &mdash; the tribute triptych</h3>
    <p><b>&rho;_seg &mdash; SegNet margin field.</b> Top1&minus;top2 logit margin per pixel;
    <b>bright = small margin = fragile</b>, the codim-1 <i>separatrix</i> where the argmax can flip
    &mdash; i.e. where d_seg lives. Yassine <b>Yousfi</b>'s comma10k segmentation baseline IS the
    contest SegNet; this panel is the detector's own sensitivity map.</p>
    <p><b>&rho;_uniward &mdash; S-UNIWARD cost.</b> The real Holub&ndash;Fridrich&ndash;Denemark 2014
    universal distortion (directional-Haar wavelet residual reciprocal-energy &mdash; the same
    approximation Yousfi 2017 uses), computed on our render. High (textured) = <b>undetectable</b> to
    embed into; smooth regions are where a perturbation would be caught.</p>
    <p class="witthesis">Read together: <b>UNIWARD undetectability</b> (Fridrich) &rarr;
    <b>SegNet detection</b> (Yousfi) &rarr; our <b>inverse-steganalysis witness</b> &mdash; a
    task-space carrier that spends its bytes on the small-margin separatrix the detector is most
    sensitive to, and nowhere else.</p>
  </div>
  <div class="witcredits">
    <div class="wch">Credits &amp; lineage &mdash; a genuine tribute</div>
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
    <div class="wcnote">[macOS-CPU advisory &middot; NON-PROMOTABLE] &mdash; a viz moves no pointer.
    The exact row is byte-closed on contest-CPU/CUDA; the frontier pointer is 0.19110 and UNMOVED.</div>
  </div>
</section>

<section id="tab-flow" class="flow hide">
  <h2>FLOW &mdash; the full n600 drive, as a video (WebGPU)</h2>
  <p>Scrub or play the <b>entire 600-pair segment</b> the scorer evaluates &mdash; the witness's own
  render, its 5-class partition, the <b>SegNet argmax the scorer actually sees</b>, and their
  <b>disagreement vs GT</b> (the d_seg pixels), flowing across the drive. The <b>frame slider is the
  timeline</b> (0&nbsp;&rarr;&nbsp;599); <b>play</b> animates it at ~12&nbsp;fps. Switch the layer to
  read the partition (canonical comma10k colors) or the fragility heat where the argmax can flip.</p>
  <div class="flowtop">
    <span id="flowbadge" class="flowbadge off">detecting&hellip;</span>
    <span class="flowstatus" id="flowstatus">waiting for the first n600 sequence&hellip;</span>
  </div>
  <div class="flowstage">
    <canvas id="flowcanvas" role="img" aria-label="n600 witness drive video — partition / SegNet / disagreement"></canvas>
    <div class="flowmsg" id="flowmsg">the first n600 video renders on the next best checkpoint (~14&nbsp;min governed pass)&hellip;</div>
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
    <p>The partition layer uses the <b>canonical comma10k / openpilot segmentation palette</b> (the
    colors Yousfi + comma read instantly). The margin-heat layer is the scorer's own <b>sensitivity /
    UNIWARD geometry</b> read as a field: <b>bright = small margin = the codim-1 separatrix</b> where
    the argmax can flip (where d_seg lives). Playing the timeline is the dynamical-system view &mdash;
    the witness partition flowing toward the SegNet argmax across the whole drive.</p>
    <p class="fcnote">[macOS-CPU advisory &middot; NON-PROMOTABLE] &mdash; a viz moves no pointer.
    The exact row is byte-closed on contest-CPU/CUDA; the frontier pointer is 0.19110 and UNMOVED.</p>
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
    a viz moves no pointer (0.19110, UNMOVED).</p>
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
      measurement; the pointer is <b>0.19110, UNMOVED</b> &mdash; a museum moves no pointer.</p>
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
  contest-CPU/CUDA; the frontier pointer is 0.19110 and UNMOVED.</p>
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
  score; the pointer is 0.19110 and UNMOVED. Everything on this page is a MEANS.</p>
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
try{reduceMotion=!!(window.matchMedia&&window.matchMedia("(prefers-reduced-motion: reduce)").matches);}catch(e){}

// stage-band palette on the epoch axis: CE / tau / l7 / Muon
const BANDS={ce:"#1f3b5f",tau:"#3a2a5f",l7:"#5f3320",muon:"#1f4f43"};

function $(id){return document.getElementById(id);}

// ---- FLOW (Tab 3) data bridge: the injected client (dashboard_flow_client.js) reads
// window.__flow.ready (the lightweight readiness ping) and registers window.__flowActivate /
// window.__flowReady. The client FETCHES the full n600 video sequence itself from
// /api/flow_sequence (once per new sequence), then scrubs/plays locally. ----
window.__flow = {ready:null};
function _onFlowReady(fr){
  if(!fr)return;
  window.__flow.ready = fr;
  if(window.__flowReady){try{window.__flowReady(fr);}catch(e){}}
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

// ---------- tabs ----------
function activateTab(t){
  document.querySelectorAll(".tab").forEach(x=>{x.classList.remove("on");x.setAttribute("aria-selected","false");});
  t.classList.add("on");t.setAttribute("aria-selected","true");
  const which=t.dataset.tab;
  document.querySelectorAll("section[id^='tab-']").forEach(s=>{s.classList.toggle("hide",s.id!=="tab-"+which);});
  if(which==="live") scheduleDraw();
  if(which==="witness") renderWitness();
  if(which==="flow"&&window.__flowActivate){try{window.__flowActivate();}catch(e){}}
  if(which==="oracle") activateOracle();
  if(which==="whyhow"&&window.__whyhowActivate){try{window.__whyhowActivate();}catch(e){}}
  if(which==="tri") activateTriality();
}
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
  ctx.fillStyle="#1b1e24"; ctx.fillRect(0,0,W,H);
  const padL=52,padR=12,padT=24,padB=26;
  const x0=padL,x1=W-padR,y0=padT,y1=H-padB;
  // data
  const pts=TRAJ.map(d=>[d.epoch,d[key]]).filter(p=>p[0]!=null&&p[1]!=null&&isFinite(p[1]));
  const log=!!opt.log;
  // x range — extend to the FULL curriculum horizon (schedule.epochs) so all 4
  // stage bands (incl. the FUTURE Muon band) are visible up front, not just up to l7.
  const schedEnd=(META.schedule&&META.schedule.epochs)||0;
  let xmin=0, xmax=Math.max(META.l7||BOOT.l7, schedEnd, ...(pts.map(p=>p[0])), 1);
  if(xmax<=xmin) xmax=xmin+1;
  // y range over data + hlines
  let yvals=pts.map(p=>p[1]);
  (opt.hlines||[]).forEach(h=>{if(h.y!=null) yvals.push(h.y);});
  if(log) yvals=yvals.filter(v=>v>0);
  let ymin=Math.min(...yvals), ymax=Math.max(...yvals);
  if(!isFinite(ymin)||!isFinite(ymax)){ymin=0;ymax=1;}
  if(ymin===ymax){ymin = log? ymin/2 : ymin-1; ymax = log? ymax*2 : ymax+1;}
  const L=v=>log?Math.log10(v):v;
  const Lmin=L(ymin), Lmax=L(ymax);
  const sx=e=>x0+(e-xmin)/(xmax-xmin)*(x1-x0);
  const sy=v=>{let lv=L(v);return y0+(Lmax-lv)/(Lmax-Lmin)*(y1-y0);};
  // store transform for hit-testing (tooltip + crosshair)
  canvas._tf={x0:x0,x1:x1,y0:y0,y1:y1,xmin:xmin,xmax:xmax};
  // stage shading (CE / tau / l7 / Muon — Muon band only when the boundary is known)
  const tau=META.tau||BOOT.tau, l7=META.l7||BOOT.l7;
  const mu=(META.muon_start!=null)?META.muon_start:null;
  const l7end=(mu!=null)?mu:xmax;
  const spans=[[xmin,tau,BANDS.ce],[tau,l7,BANDS.tau],[l7,l7end,BANDS.l7]];
  if(mu!=null) spans.push([mu,xmax,BANDS.muon]);
  ctx.globalAlpha=0.18;
  spans.forEach(s=>{const a=Math.max(s[0],xmin),b=Math.min(s[1],xmax);
    if(b>a){ctx.fillStyle=s[2];ctx.fillRect(sx(a),y0,sx(b)-sx(a),y1-y0);}});
  ctx.globalAlpha=1;
  // grid + y ticks
  ctx.strokeStyle="#2c313b"; ctx.fillStyle="#8b93a3"; ctx.font="10px system-ui"; ctx.lineWidth=1;
  const nT=4;
  for(let i=0;i<=nT;i++){
    const lv=Lmin+(Lmax-Lmin)*i/nT; const val=log?Math.pow(10,lv):lv; const yy=sy(val);
    ctx.globalAlpha=0.5;ctx.beginPath();ctx.moveTo(x0,yy);ctx.lineTo(x1,yy);ctx.stroke();ctx.globalAlpha=1;
    let lab = opt.fmt ? opt.fmt(val) : String(val);
    ctx.textAlign="right";ctx.fillText(lab,x0-5,yy+3);
  }
  // x ticks
  ctx.textAlign="center";ctx.fillStyle="#8b93a3";
  for(let i=0;i<=4;i++){const e=xmin+(xmax-xmin)*i/4;ctx.fillText(Math.round(e),sx(e),y1+14);}
  // hlines (goals / reference)
  (opt.hlines||[]).forEach(h=>{
    if(h.y==null||(log&&h.y<=0))return;
    const yy=sy(h.y); ctx.strokeStyle=h.color||"#46d369"; ctx.setLineDash([4,3]); ctx.lineWidth=1.2;
    ctx.beginPath();ctx.moveTo(x0,yy);ctx.lineTo(x1,yy);ctx.stroke();ctx.setLineDash([]);
    ctx.fillStyle=h.color||"#46d369";ctx.textAlign="left";ctx.fillText(h.label||"",x0+4,yy-3);
  });
  // stage vlines + labels (tau / l7 / Muon)
  const vls=[[tau,"tau"],[l7,"l7"]]; if(mu!=null)vls.push([mu,"Muon"]);
  vls.forEach(s=>{
    if(s[0]<=xmin||s[0]>xmax)return;const xx=sx(s[0]);
    ctx.strokeStyle="#8b93a3";ctx.setLineDash([3,3]);ctx.globalAlpha=0.7;ctx.lineWidth=1;
    ctx.beginPath();ctx.moveTo(xx,y0);ctx.lineTo(xx,y1);ctx.stroke();ctx.setLineDash([]);ctx.globalAlpha=1;
    ctx.fillStyle="#d8dde6";ctx.textAlign="left";ctx.fillText(s[1],xx+3,y0+10);
  });
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
  // series
  if(pts.length){
    ctx.strokeStyle=opt.color;ctx.lineWidth=1.8;ctx.beginPath();
    pts.forEach((p,i)=>{const X=sx(p[0]),Y=sy(p[1]);i?ctx.lineTo(X,Y):ctx.moveTo(X,Y);});
    ctx.stroke();
    ctx.fillStyle=opt.color;
    pts.forEach(p=>{ctx.beginPath();ctx.arc(sx(p[0]),sy(p[1]),2.6,0,7);ctx.fill();});
    const last=pts[pts.length-1];
    ctx.fillStyle="#d8dde6";ctx.textAlign="left";ctx.font="11px system-ui";
    ctx.fillText(opt.fmt?opt.fmt(last[1]):String(last[1]),Math.min(sx(last[0])+5,x1-72),sy(last[1])-5);
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
  const g=META.goal_dseg||BOOT.goal_dseg, g15=META.goal_dseg_15||BOOT.goal_dseg_15;
  const fS=v=>sig(v,5), fP=v=>sig(v,4), fI=v=>fmtInt(v), fSv=v=>sig(v,4);
  const star=(bestEpoch!=null&&bestVal!=null)?{epoch:bestEpoch,val:bestVal}:null;
  drawPanel($("c_dseg"),"d_seg",{title:"d_seg — realized SegNet-argmax disagreement (lower better)",
    sub:"epoch · log scale · ★ best · goal lines = sub-0.19 / sub-0.15",color:"#5ab0ff",log:true,fmt:fS,
    star:star,starGlow:_celebrating,
    hlines:[{y:g,label:"sub-0.19  "+sig(g,4),color:"#46d369"},{y:g15,label:"sub-0.15  "+sig(g15,4),color:"#ffb454"}]});
  drawPanel($("c_dpose"),"d_pose",{title:"d_pose — store-nothing screw carrier · in-run measured (byte-closed confirm pending #238)",
    sub:"epoch · log · the carrier's own measured pose",color:"#ffb454",log:true,fmt:fP,
    hlines:[]});
  drawPanel($("c_bytes"),"blob_bytes",{title:"blob_bytes — LEARNED payload (counted in archive)",
    sub:"epoch · smaller payload = lower rate term",color:"#c08cff",log:false,fmt:fI,hlines:[]});
  drawPanel($("c_s"),"implied_S",{title:"implied_S — advisory (measured d_seg + store-nothing-carrier d_pose + rate)",
    sub:"epoch · log · run's own measured composite · NOT the exact 0.19110 pointer",color:"#ff6b6b",log:true,fmt:fSv,
    hlines:[{y:META.pointer||BOOT.pointer,label:"pointer 0.19110",color:"#46d369"}]});
  updateAria();
}
function updateAria(){
  const last=TRAJ.length?TRAJ[TRAJ.length-1]:null;
  const set=(id,txt)=>{const el=$(id);if(el)el.setAttribute("aria-label",txt);};
  if(!last){["c_dseg","c_dpose","c_bytes","c_s"].forEach(i=>set(i,"chart, no data yet"));return;}
  const bestTxt=(bestVal!=null)?(", best "+sig(bestVal,5)+" at epoch "+bestEpoch):"";
  set("c_dseg","d_seg over epochs, latest "+sig(last.d_seg,5)+" at epoch "+last.epoch+bestTxt);
  set("c_dpose","d_pose over epochs, latest "+sig(last.d_pose,4)+" at epoch "+last.epoch);
  set("c_bytes","blob bytes over epochs, latest "+fmtInt(last.blob_bytes)+" at epoch "+last.epoch);
  set("c_s","implied S advisory over epochs, latest "+sig(last.implied_S,4)+" at epoch "+last.epoch+", pointer 0.19110");
}
let _drawQueued=false;
function scheduleDraw(){
  if(_drawQueued)return; _drawQueued=true;
  requestAnimationFrame(()=>{_drawQueued=false;
    if(!$("tab-live").classList.contains("hide")) drawAll();});
}

function stageWord(ep){if(ep==null)return "starting";
  const tau=META.tau||BOOT.tau,l7=META.l7||BOOT.l7,mu=META.muon_start;
  if(ep<tau)return "CE";if(ep<l7)return "tau";
  if(mu!=null&&ep>=mu)return "Muon";
  return (mu!=null)?"l7":"l7/Muon";}

// ---------- scorer breakdown (implied-S decomposition; HONEST measured terms) ----------
// S = 100·d_seg + √(10·d_pose) + 25·(bytes/N), all three from the last measured row.
// d_pose is the store-nothing screw carrier's OWN measured pose — the honest composite.
// This is advisory (NOT the exact 0.19110 pointer); byte-closed pose confirmation is #238.
function renderScorerBreakdown(last){
  const subEl=$("sb_subst"), termsEl=$("sb_terms"), depEl=$("sb_deploy");
  if(!subEl||!termsEl||!depEl)return;
  if(!last||last.d_seg==null||last.d_pose==null||last.blob_bytes==null||
     !isFinite(last.d_seg)||!isFinite(last.d_pose)||!isFinite(last.blob_bytes)){
    subEl.innerHTML="&nbsp;"; termsEl.innerHTML=""; depEl.innerHTML="&nbsp;"; return;
  }
  const N=META.archive_norm_bytes||37545489, Nstr=N.toLocaleString("en-US");
  const ds=last.d_seg, dp=last.d_pose, by=last.blob_bytes;
  const t1=100.0*ds, t2=Math.sqrt(10.0*dp), t3=25.0*by/N;
  const measuredS=t1+t2+t3;
  // (2) substituted formula with the latest row's actual MEASURED values
  subEl.innerHTML="with our values (measured): S = 100&middot;(<b>"+sig(ds,6)+
    "</b>) + &radic;(10&middot;<b>"+sig(dp,6)+"</b>) + 25&middot;(<b>"+fmtInt(by)+"</b> / "+Nstr+")";
  // (3) weighted terms -> S (all three measured)
  let html="";
  [["100&middot;d_seg",t1],["&radic;(10&middot;d_pose)",t2],["25&middot;bytes / "+Nstr,t3]].forEach(r=>{
    html+="<div class='sbrow'><span class='sbk'>"+r[0]+"</span><span class='sbv'>= "+sig(r[1],4)+"</span></div>";
  });
  html+="<div class='sbrule'></div>";
  html+="<div class='sbrow sbtot'><span class='sbk'>S (measured) = implied_S</span>"+
    "<span class='sbv'>= "+sig(measuredS,4)+"</span></div>";
  termsEl.innerHTML=html;
  // (4) one-line advisory note — this is the run's own measured composite, not the pointer
  depEl.innerHTML="advisory &mdash; the run's own measured composite (store-nothing-carrier d_pose), "+
    "NOT the exact 0.19110 pointer &middot; byte-closed pose confirmation is #238.";
}

function render(){
  recomputeBest();
  const last=TRAJ.length?TRAJ[TRAJ.length-1]:null;
  const g=META.goal_dseg||BOOT.goal_dseg;
  // liveness pill
  const k=LIVE.kind, p=$("pill");
  if(k==="live"&&!LIVE.calibrating){p.className="pill live";p.textContent="● live";}
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
    if(!META.run_dir){rd.textContent="resolving run…";}
    else{rd.textContent="watching "+META.run_dir+(META.warming_up?" · warming up (structured-init, no verdict yet)":"");}
  }
  // headline stat cells - raw numbers in discrete cells; down=good arrows on all four
  if(last){
    $("d_seg_val").textContent=sig(last.d_seg,5);
    const setArr=(id,key)=>{const a=arrowFor(key),el=$(id);if(el){el.className="trend "+a.cls;el.textContent=a.ar;}};
    setArr("m_trend","d_seg");setArr("p_trend","d_pose");setArr("b_trend","blob_bytes");setArr("s_trend","implied_S");
    $("m_goal").textContent="goal < "+sig(g,4);
    $("m_best").textContent=(bestVal!=null)?("best "+sig(bestVal,5)+" @ ep"+bestEpoch):" ";
    $("d_pose_val").textContent=sig(last.d_pose,4);
    $("bytes_val").textContent=fmtInt(last.blob_bytes);
    $("s_val").textContent=sig(last.implied_S,4);
  }
  renderScorerBreakdown(last);
  // stage legend: light up Muon only when the boundary is known
  const muOn=(META.muon_start!=null);
  document.querySelectorAll('#slegend .sc[data-st="muon"]').forEach(el=>el.classList.toggle("off",!muOn));
  renderProjection(g);
  renderConfig();
  renderRunInfo();
  // status
  const ep=LIVE.last_epoch;
  let st=[];
  if(k==="missing"){st=["no run log found"];}
  else{st.push(stageWord(ep)+" stage"); if(ep!=null)st.push("ep"+ep);
    if(k==="stale")st.push("no verdict in "+fmtAge(LIVE.verdict_age_s)+" — likely stopped");
    else if(LIVE.next_eta_s!=null)st.push("next verdict ~"+fmtAge(LIVE.next_eta_s));}
  $("status").textContent=st.join(" · ");
  // detail
  let d=[];
  if(LIVE.verdict_age_s!=null)d.push("verdict "+fmtAge(LIVE.verdict_age_s)+" ago");
  if(LIVE.log_age_s!=null)d.push("log "+fmtAge(LIVE.log_age_s)+" ago");
  if(LIVE.cadence_s)d.push("cadence ~"+(LIVE.cadence_s/60).toFixed(0)+"m ("+(LIVE.calibrating?"calibrating":"measured")+")");
  if(META.uptime_s!=null)d.push("dash up "+fmtAge(META.uptime_s));
  if(META.training_alive!=null)d.push("training "+(META.training_alive?"alive":"gone"));
  $("detail").textContent=d.join(" · ")||" ";
  // foot
  $("foot").textContent="[macOS-MLX advisory · NON-PROMOTABLE] · pointer 0.19110 · stages CE · tau · l7 · Muon"+
    (META.watched?(" · "+META.watched):"")+" · "+TRAJ.length+" verdicts · tap charts for details";
  // boot spans in triality
  $("b_tau").textContent=META.tau||BOOT.tau; $("b_l7").textContent=META.l7||BOOT.l7;
  $("b_goal").textContent=sig(META.goal_dseg||BOOT.goal_dseg,4);
  scheduleDraw();
}

// ---------- projection (rendered from the SERVER-computed critical-slowing model) ----------
// The fit MATH lives server-side in tools/dashboard_trajectory_model.py (pure numpy);
// the client only RENDERS the returned numbers + their flags. Never a client-side fit.
function fmtEps(n){return (n==null||!isFinite(n))?"?":fmtInt(n)+" ep";}
function goalEtaStr(ge){
  if(!ge||ge.state==="calibrating")return "calibrating";
  if(ge.state==="reached")return "reached ✓";
  if(ge.state==="not_descending")return "not descending";
  if(ge.state==="asymptote_above")return "won't reach — asymptote "+sig(ge.asymptote,3)+" &gt; target";
  if(ge.state==="eta"){
    let s="~"+fmtEps(ge.eta_epochs);
    if(ge.epoch_lo!=null||ge.epoch_hi!=null){
      const lo=(ge.epoch_lo!=null)?fmtInt(ge.epoch_lo):"?";
      const hi=(ge.epoch_hi!=null)?fmtInt(ge.epoch_hi):"≳";  // open upper band = may not reach in 1σ
      s+=" (ep "+lo+"–"+hi+")";
    }
    return s;
  }
  return "?";
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
  // from the CE flicker plateau, mis-declares "won't reach". We render P.stage_proj instead:
  // each stage's OWN asymptote/floor + the ep-tau (lane-band birth) / ep-Muon (finishing)
  // EXPECTED-BREAKTHROUGH boundaries the CE fit does NOT model. Muon = saddle-to-saddle
  // staircase (polynomial escape), NEVER power-law-extrapolated. Advisory.
  const bits=[];
  const SP=P.stage_proj||{};
  if(SP.ok){
    const parts=[];
    (SP.stages||[]).forEach(st=>{
      const cur=(st.name===SP.current_stage)?" <b>◀ current</b>":"";
      if(st.name==="Muon"){
        parts.push("<b>Muon</b> ep"+fmtInt(st.start)+"+ · "+escHtml(st.note||"saddle staircase — not power-law")+cur);
      } else if(!st.entered){
        parts.push("<b>"+escHtml(st.name)+"</b> ep"+fmtInt(st.start)+"–"+fmtInt(st.end)+" · not entered yet"+cur);
      } else if(st.fit_state==="ok"){
        const cflag=(st.confidence==="low")?" ⚠low-conf":"";
        let s="<b>"+escHtml(st.name)+"</b> asymptote d_seg<sub>∞</sub> "+sig(st.asymptote,4)+
          " (α "+sig(st.alpha,2)+" · R² "+sig(st.r2,3)+" · "+(st.confidence||"?")+cflag+")";
        if(st.observed_min!=null) s+=" ≈ observed floor "+sig(st.observed_min,4);
        if(st.name==="CE") s+=" — the CE plateau, NOT the final floor (as expected)";
        parts.push(s+cur);
      } else if(st.fit_state==="insufficient"){
        parts.push("<b>"+escHtml(st.name)+"</b> "+(st.n||0)+" verdict"+((st.n===1)?"":"s")+
          " — insufficient for a stage fit yet"+cur);
      } else {
        parts.push("<b>"+escHtml(st.name)+"</b> "+escHtml(st.note||"no fit")+cur);
      }
    });
    segEl.innerHTML="d_seg per-stage critical-slowing model · "+parts.join(" &nbsp;&middot;&nbsp; ");
    // downstream EXPECTED-BREAKTHROUGH boundaries (the CE fit does NOT model these)
    if((SP.downstream||[]).length){
      const dlab=SP.downstream.map(d=>"ep"+fmtInt(d.epoch)+" "+escHtml(d.label)+
        (d.status==="engaged"?" (engaged)":" (expected)")).join(" · ");
      bits.push("EXPECTED-BREAKTHROUGH boundaries the CE fit does NOT model: "+dlab);
    }
    // implied_S from the best-modeled stage floor — LABELLED current-stage-only (excludes breakthroughs)
    const mf=SP.modeled_floor, is=(mf&&mf.implied_s)||{};
    if(mf&&is.ok&&is.value!=null){
      const gap=mf.is_current_stage?"":" (latest modeled; current "+escHtml(SP.current_stage||"?")+" still calibrating)";
      let s=escHtml(mf.stage)+"-stage floor"+gap+" → implied_S <b>"+sig(is.value,4)+"</b>";
      if(is.value_lo!=null||is.value_hi!=null)
        s+=" ("+sig(is.value_lo,4)+"–"+(is.value_hi!=null?sig(is.value_hi,4):"≳")+")";
      const ts=(SP.tau_start!=null)?fmtInt(SP.tau_start):"?", ms=(SP.muon_start!=null)?fmtInt(SP.muon_start):"?";
      s+=" — CURRENT-STAGE extrapolation ONLY, EXCLUDES the ep"+ts+"/ep"+ms+" breakthroughs · vs pointer 0.19110";
      bits.push(s);
    }
  } else {
    // stage projection unavailable -> fall back to the global fit, but WITHOUT a "won't reach" verdict
    const fit=P.dseg_model||{};
    if(fit.ok){
      const cflag=(fit.confidence==="low")?" ⚠low-confidence":"";
      segEl.innerHTML="d_seg critical-slowing model (global) · asymptote d_seg<sub>∞</sub> <b>"+sig(fit.asymptote,4)+
        "</b> · α "+sig(fit.alpha,2)+" · R² "+sig(fit.r2,3)+" · "+(fit.confidence||"low")+cflag+
        " — current-trajectory floor; stage breakthroughs (tau/Muon) unmodeled";
    } else {
      segEl.innerHTML="d_seg model · <b>calibrating</b> — "+escHtml(fit.reason||"need more points");
    }
  }
  // ---- keep the good parts: MEASURED completion ETA + current-stage cadence ----
  const ce=P.completion_eta||{};
  const tot=(META.schedule&&META.schedule.epochs)||null;
  if(ce.ok&&ce.total_s!=null){
    let c="ETA to ep"+(tot!=null?tot:"end")+" ~"+fmtAge(ce.total_s);
    if(ce.has_estimate)c+=" (Muon estimated — refines at ep726)";
    else c+=" (measured)";
    bits.push(c);
  }
  if(P.next_verdict_cadence_s!=null){
    const src=P.next_verdict_cadence_source||"measured";
    bits.push((P.stage||"")+"-stage cadence ~"+(P.next_verdict_cadence_s/60).toFixed(0)+"m ("+src+")");
  }
  sEl.innerHTML=bits.join(" · ")||" ";
}

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
function renderConfig(){
  const body=$("cfgbody"), sum=$("cfgsum"); if(!body)return;
  const cfg=META.config||{}, sched=(META.schedule||cfg.schedule||{});
  const groups=cfg.groups||{};
  let html="";
  // curriculum schedule (full width) — TWO ORTHOGONAL AXES, data-driven from the raw
  // schedule fields (NOT the naive [start,next_start) bounds, which INVERT when the
  // optimizer switch muon_start precedes a disabled l7_start -> e.g. l7 [1000,726)).
  const sch=buildScheduleRows(sched);
  if(sch){
    html+="<div class='cfgsec full'><div class='cfgh'>curriculum schedule</div>"+sch.body;
    const m2=[];
    if(sched.epochs!=null)m2.push("total "+sched.epochs+" ep");
    if(sched.eval_every!=null)m2.push("eval every "+sched.eval_every+" ep");
    if(LIVE.next_eta_s!=null)m2.push("next verdict ~"+fmtAge(LIVE.next_eta_s));
    if(m2.length)html+="<div class='cfgmeta'>"+escHtml(m2.join(" · "))+"</div>";
    html+="</div>";
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
      " · pointer "+(d.pointer!=null?d.pointer:"0.19110")+" (UNMOVED)";
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
      dsl.innerHTML=
        "<div class='trirow'><span class='tk'>program</span>"+(p.epochs||"?")+" ep · "+
          (p.num_pairs||"?")+" pairs · stages "+_esc((p.stages||[]).join(" → "))+"</div>"+
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
    hdr.innerHTML="the <b>"+W.n_pairs_shown+" hardest &amp; most-diverse pairs</b> of the n600 drive "+
      "(spread across the segment, labelled by failure mode) · from BEST checkpoint @ epoch <b>"+
      (W.epoch!=null?W.epoch:"?")+"</b> · mean realized d_seg over the selection <b>"+sig(W.mean_dseg,5)+
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
    else applyUpdate(m);}catch(e){}};
  ws.onclose=()=>{wsOpen=false;setWsPill(false);wsTries++;startPoll();
    setTimeout(connectWS,Math.min(15000,1000*Math.pow(1.6,Math.min(wsTries,8))));};
  ws.onerror=()=>{try{ws.close();}catch(e){}};
}
// ---------- polling fallback (only active while WS is down) ----------
async function pollOnce(){
  if(wsOpen)return;
  try{const r=await fetch("/api/state"+location.search,{cache:"no-store"});
    if(r.ok){applySnapshot(await r.json());}}catch(e){}
}
function startPoll(){if(pollTimer)return;pollTimer=setInterval(pollOnce,(BOOT.poll||5)*1000);pollOnce();}
function stopPoll(){if(pollTimer){clearInterval(pollTimer);pollTimer=null;}}

// ---------- pointer interactions: synchronized crosshair + tooltip (touch+hover) ----------
function setupInteractions(){
  const tip=$("tip");
  const canvases=["c_dseg","c_dpose","c_bytes","c_s"].map(id=>$(id)).filter(Boolean);
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
setupInteractions();
window.addEventListener("resize",scheduleDraw);
window.addEventListener("orientationchange",scheduleDraw);
connectWS();
// safety net: if WS never opens within 4s, ensure polling is running
setTimeout(()=>{if(!wsOpen)startPoll();},4000);
// ORACLE is the default-visible tab (arc entry point) -> kick its one-shot fetch on load.
try{activateOracle();}catch(e){}
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
    ap.add_argument("--tau", type=int, default=cfg.tau)
    ap.add_argument("--l7", type=int, default=cfg.l7)
    ap.add_argument("--goal-dseg", type=float, default=cfg.goal_dseg)
    ap.add_argument("--goal-dseg-15", type=float, default=cfg.goal_dseg_15)
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
