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

from starlette.applications import Starlette  # noqa: E402
from starlette.responses import (  # noqa: E402
    HTMLResponse,
    JSONResponse,
    PlainTextResponse,
)
from starlette.routing import Route, WebSocketRoute  # noqa: E402
from starlette.websockets import WebSocket, WebSocketDisconnect  # noqa: E402

POINTER = 0.19110  # frontier pointer (contest-CPU), UNMOVED — advisory only here


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
    )


_TRAJ_KEYS = ("epoch", "d_seg", "d_pose", "blob_bytes", "implied_S", "ts")


def _slim(row: dict) -> dict:
    """Keep only the trajectory fields the client charts need (no leaking of the
    full verdict dict). Numbers stay numeric; missing keys become None."""
    return {k: row.get(k) for k in _TRAJ_KEYS}


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
        self.started_at = time.time()
        self.clients: set[WebSocket] = set()
        # cadence sub-state (per-log), persisted to disk like render_levelset_dashboard
        self._cad_args = SimpleNamespace(stale_min=None, stale_floor_min=10.0,
                                         cadence_k=2.5, cadence_prior_min=18.0)

    # ---- refresh (sync; called via executor) ----
    def refresh(self) -> list[dict]:
        """Auto-resolve the freshest arm (auto-latest), walk its RESUME ANCESTRY, and
        rebuild the FULL trajectory (CE->tau->l7->muon...) — not just the post-resume
        tail. Liveness reflects the live (latest) arm. Returns NEW points for WS deltas."""
        cfg = self.cfg
        latest = rld._resolve_watched_log(None, cfg.resolved_glob())
        # FULL trajectory across the warm-start chain (de-dup by epoch; later arm wins
        # at boundary collisions since the chain is ordered root..latest).
        chain = _resume_chain_logs(latest)
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
        # liveness + cadence from the LIVE (latest) arm only (the freshest log).
        now = time.time()
        ccfg = rld._cfg_from_args(self._cad_args)
        latest_rows = rld._parse_verdicts(latest) if latest is not None else []
        log_name = latest.name if latest is not None else "_none_"
        all_state, sub = rld._load_cadence_state(cfg.cadence_state, log_name)
        mtime = latest.stat().st_mtime if (latest is not None and latest.exists()) else None
        self.liveness = rld._compute_liveness(latest_rows, mtime, now, sub, ccfg)
        rld._save_cadence_state(cfg.cadence_state, all_state, log_name, sub)
        self.watched = latest.name if latest is not None else None
        self.watched_dir = str(latest.parent) if latest is not None else None

        # Rebuild trajectory from the full chain each tick (cheap; few hundred points).
        # new_points = epochs not yet pushed to WS clients (snapshot carries the full set).
        new_points = [p for p in rows_full
                      if isinstance(p.get("epoch"), int) and p["epoch"] not in self._epochs]
        for p in new_points:
            self._epochs.add(p["epoch"])
        self.trajectory = rows_full
        return new_points

    # ---- snapshot for client ----
    def meta(self) -> dict:
        cfg = self.cfg
        return {
            "tau": cfg.tau, "l7": cfg.l7,
            "goal_dseg": cfg.goal_dseg, "goal_dseg_15": cfg.goal_dseg_15,
            "pointer": POINTER, "watched": self.watched,
            "run_dir": self.watched_dir or cfg.run_dir,  # auto-latest: the live arm dir
            "uptime_s": time.time() - self.started_at,
            "training_alive": _training_alive(cfg.training_pid, cfg.training_sig),
            "n_points": len(self.trajectory),
            "muon_start": self.muon_start,  # additive: inferred l7 -> Muon boundary epoch
            "n_pairs": self.n_pairs,        # additive: N (n200 DOE pilot / n600 scored)
        }

    def snapshot(self) -> dict:
        return {"type": "snapshot", "trajectory": self.trajectory,
                "liveness": self.liveness, "meta": self.meta()}

    def update_msg(self, new_points: list[dict]) -> dict:
        return {"type": "update", "new_points": new_points,
                "liveness": self.liveness, "meta": self.meta()}

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
            except Exception as exc:  # telemetry must never crash the live server
                print(json.dumps({"stage": "dashboard_server", "tailer_error": str(exc)}),
                      flush=True)
            try:
                await asyncio.wait_for(stop.wait(), timeout=cfg.poll)
            except asyncio.TimeoutError:
                pass

    @contextlib.asynccontextmanager
    async def lifespan(app):
        # prime once so the first client gets data immediately
        with contextlib.suppress(Exception):
            await asyncio.get_event_loop().run_in_executor(None, state.refresh)
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


def _page_html(cfg: Config) -> str:
    boot = json.dumps({
        "tau": cfg.tau, "l7": cfg.l7,
        "goal_dseg": cfg.goal_dseg, "goal_dseg_15": cfg.goal_dseg_15,
        "pointer": POINTER, "poll": cfg.poll,
    })
    return _PAGE_TEMPLATE.replace("__BOOT__", boot)


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
.tabs{display:flex;gap:4px;margin:16px 0 18px;border-bottom:1px solid var(--grid)}
.tab{font-size:13px;font-weight:600;color:var(--muted);padding:12px 16px;cursor:pointer;
border-bottom:2px solid transparent;-webkit-tap-highlight-color:transparent;user-select:none}
.tab:hover{color:var(--fg2)}
.tab.on{color:var(--fg);border-bottom-color:var(--acc)}

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

/* triality tab */
.tri h2{font-size:clamp(13px,3.4vw,15px);color:var(--acc);letter-spacing:.4px;margin:22px 0 8px}
.tri p,.tri li{font-size:13.5px;color:var(--fg2);line-height:1.6}
.tri .m{color:var(--muted);font-size:12.5px}
.tri code{background:#11141a;color:#9fc6ff;padding:1px 5px;border-radius:5px;font-size:12px;word-break:break-word}
.tri .cards{display:grid;grid-template-columns:minmax(0,1fr);gap:12px;margin:10px 0}
@media(min-width:680px){.tri .cards{grid-template-columns:repeat(3,minmax(0,1fr))}}
.tri .card{background:var(--panel);border:1px solid var(--grid);border-radius:12px;padding:14px}
.tri .card h3{font-size:12px;color:var(--goal);margin:0 0 7px;letter-spacing:.5px;text-transform:uppercase}
.tri ol,.tri ul{padding-left:20px}.tri li{margin-bottom:5px}

/* n-badge (which run am I watching) */
.nbadge{font-size:11.5px;font-weight:700;padding:4px 11px;border-radius:999px;white-space:nowrap;letter-spacing:.3px;line-height:1.3}
.nbadge.doe{background:#16263a;color:#7fc0ff}
.nbadge.scored{background:#173d22;color:#7fe0a0}
.nbadge.other{background:#2a2f39;color:#c2c9d4}
.runinfo{font-size:11px;color:var(--faint2);margin:0 2px 12px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}

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
  <div class="tab on" data-tab="live">LIVE</div>
  <div class="tab" data-tab="tri">TRIALITY &middot; PLAN</div>
</div>

<section id="tab-live">
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
      <span class="ssub adv">advisory est.</span>
    </div>
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
  <div class="grid">
    <div class="panel"><canvas id="c_dseg" role="img" aria-label="d_seg chart"></canvas></div>
    <div class="panel"><canvas id="c_dpose" role="img" aria-label="d_pose chart"></canvas></div>
    <div class="panel"><canvas id="c_bytes" role="img" aria-label="blob bytes chart"></canvas></div>
    <div class="panel"><canvas id="c_s" role="img" aria-label="implied S chart"></canvas></div>
  </div>
  <div class="foot" id="foot"></div>
</section>

<section id="tab-tri" class="tri hide">
  <h2>One coherent object &mdash; the DAG &harr; DSL &harr; equations triality</h2>
  <p>The lab is one object viewed three ways. The witness is a gauge-invariant target (the
  frozen-scorer equivalence class); the three views below are the math, the program, and the trajectory of running it.</p>
  <div class="cards">
    <div class="card"><h3>System of equations</h3><p class="m">The MATH / grammar. ONE master action
    <code>S_&tau; = 100&middot;d_seg + &radic;(10&middot;d_pose) + 25&middot;rate</code>, with terms E0&ndash;E12 in
    <code>tac.canonical_equations</code>. Every lever is a term / relaxation of this same action &mdash;
    which is why composing levers is principled, not a sweep.</p></div>
    <div class="card"><h3>DSL &mdash; the witness program</h3><p class="m">A declarative recursion+math front-end
    (<code>tac.witness_dsl</code>) that COMPILES to the proven trainer CLI and VALIDATES every emitted flag
    against the real argparse (never-invent-flags). Arms = <code>BASELINE.with_lever(&hellip;)</code>; the composed
    optimum &theta;* = <code>compose</code> of the measured-positive levers.</p></div>
    <div class="card"><h3>DAG &mdash; the lab trajectory</h3><p class="m">The dependency graph of
    findings / experiments / decisions (the append-only FEEDs in
    <code>sub015_DAG_&hellip;.md</code>). The campaign engine literally IS a DAG: nodes = arms,
    edges = warm-start chains, sink = the &theta;* compose.</p></div>
  </div>
  <p class="m">Above all three sits the GAUGE meta-layer: equivalent expressions (screw-twist vs per-class
  homography warp; single-SDF vs MSDF carrier; stored vs learned residual) are different gauges with
  gauge-dependent cost (counted bytes / d_seg-through-R), selected hard-gates-first &rarr; min-S &rarr; synergy.</p>

  <h2>This run &mdash; from-scratch openpilot-seeded level-set witness</h2>
  <ul>
    <li>A <b>nonlinear coordinate-INR witness</b> that amortizes the SegNet argmax partition directly
    (scorer-only-trained, no full-RGB reconstruction) &mdash; OUR carrier, NOT a PR95/HNeRV reskin.</li>
    <li><b>From scratch</b> with an <b>openpilot seeding prior</b>: <code>--structured-init</code> builds the
    static-core partition SDFs (road/sky/hood/lane, self-detected) and <code>--lane-prior-phi1</code> injects the
    openpilot deg-3 centerline SDF into the &phi;1 lane channel (the measured Road&harr;Lane separatrix; a
    rule-118 FREE training-time init that ships 0 archive bytes).</li>
    <li>Carrier: <code>hosc</code> activation + SIREN init, hidden 96 / mod 32, curvelet front-end
    (<code>--self-orient</code>, directional Fourier bank) + <b>chroma</b> (a genuine d_seg actuator &mdash; SegNet reads RGB).</li>
    <li>Curriculum: CE &rarr; tau-softplus (&tau;=0.3, the measured reachability floor) at ep <span id="b_tau">300</span>,
    l7 / Muon stacking at ep <span id="b_l7">600</span>. Pose rides the stored-target sidecar
    (<code>--w-pose 0</code>); the witness's sole binding controllable job is <b>d_seg</b>.</li>
    <li><b>Binding unknown:</b> the realized-through-R d_seg as the seg-surrogate engages &mdash; does the
    seeded basin + directional basis + chroma drive d_seg toward the sub-0.19 budget
    (&lt;<span id="b_goal">0.00092</span>) and onward to sub-0.15 (&lt;0.00032)?</li>
  </ul>

  <h2>Next steps &mdash; toward a byte-closed exact row</h2>
  <ol>
    <li>Watch realized-through-R d_seg through CE &rarr; tau; adaptively stack l7 / Muon (extend / advance+reheat /
    rollback-to-best) per the campaign policy.</li>
    <li>Compose the measured-positive levers into &theta;* (the DSL <code>compose</code>); map the RD curve
    (deterministic &harr; hybrid &harr; neural Pareto set), not one point.</li>
    <li>Byte-close in the L13 task-space format + the stored pose sidecar; then run the dual exact eval
    (contest-CPU AND contest-CUDA) on the EXACT archive bytes.</li>
    <li>Only a measured exact row below <b>0.19110</b> moves the pointer &mdash; toward sub-0.19, then the
    sub-0.15 target. Everything on this page is advisory until that row lands.</li>
  </ol>
  <p class="m">Authority: <code>[macOS-MLX training advisory]</code> NON-PROMOTABLE. The exact score is the only
  score; the pointer is 0.19110 and UNMOVED.</p>
</section>

</div>
<div class="tip" id="tip" aria-hidden="true"></div>
<div class="nbest" id="nbest" role="status" aria-live="polite"></div>
<script>
const BOOT = __BOOT__;
let TRAJ = [], LIVE = {}, META = {};
let ws = null, wsOpen = false, wsTries = 0, pollTimer = null;
// enrichment state (all derived client-side from TRAJ; no new backend data)
let hoverEpoch=null, bestVal=null, bestEpoch=null, _celebrating=false, _celTimer=null;
let reduceMotion=false;
try{reduceMotion=!!(window.matchMedia&&window.matchMedia("(prefers-reduced-motion: reduce)").matches);}catch(e){}

// stage-band palette on the epoch axis: CE / tau / l7 / Muon
const BANDS={ce:"#1f3b5f",tau:"#3a2a5f",l7:"#5f3320",muon:"#1f4f43"};

function $(id){return document.getElementById(id);}

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
  $("tab-live").classList.toggle("hide",which!=="live");
  $("tab-tri").classList.toggle("hide",which!=="tri");
  if(which==="live") scheduleDraw();
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
  // x range
  let xmin=0, xmax=Math.max(META.l7||BOOT.l7, ...(pts.map(p=>p[0])), 1);
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
  drawPanel($("c_dpose"),"d_pose",{title:"d_pose — realized PoseNet MSE (pose on sidecar)",
    sub:"epoch · log scale · existence-proof ~0.0009",color:"#ffb454",log:true,fmt:fP,
    hlines:[{y:0.0009,label:"~0.0009",color:"#46d369"}]});
  drawPanel($("c_bytes"),"blob_bytes",{title:"blob_bytes — LEARNED payload (counted in archive)",
    sub:"epoch · smaller payload = lower rate term",color:"#c08cff",log:false,fmt:fI,hlines:[]});
  drawPanel($("c_s"),"implied_S",{title:"implied_S — ADVISORY mid-training estimate (NOT the contest score)",
    sub:"epoch · log scale · frontier pointer = 0.19110",color:"#ff6b6b",log:true,fmt:fSv,
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
  const rd=$("rdinfo"); if(rd)rd.textContent=META.run_dir?("watching "+META.run_dir):"resolving run…";
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
  // stage legend: light up Muon only when the boundary is known
  const muOn=(META.muon_start!=null);
  document.querySelectorAll('#slegend .sc[data-st="muon"]').forEach(el=>el.classList.toggle("off",!muOn));
  renderProjection(g);
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

function renderProjection(g){
  const segEl=$("proj_seg"), sEl=$("proj_s"); if(!segEl||!sEl)return;
  const g15=META.goal_dseg_15||BOOT.goal_dseg_15;
  const dpts=TRAJ.map(d=>[d.epoch,d.d_seg]).filter(p=>p[0]!=null&&p[1]!=null&&isFinite(p[1])&&p[1]>0);
  if(dpts.length<3){segEl.textContent="projection · collecting points…";sEl.textContent=" ";return;}
  const f=linfit(dpts.slice(dpts.length-winSize(dpts.length)));
  const last=dpts[dpts.length-1][1];
  const slope=f?f.m:null, per100=(slope!=null)?slope*100:null;
  const etaStr=goal=>{const e=etaEpochs(slope,last,goal);
    if(e.state==="reached")return "reached ✓";if(e.state==="stalled")return "stalled";
    if(e.state==="eta")return "~"+fmtInt(e.epochs)+" ep";return "?";};
  const slopeStr=(per100!=null)?((per100>=0?"+":"")+sig(per100,2)+"/100ep"):"n/a";
  segEl.innerHTML="projection · naive linear · advisory: d_seg <b>"+slopeStr+
    "</b> · sub-0.19 <b>"+etaStr(g)+"</b> · sub-0.15 <b>"+etaStr(g15)+"</b>";
  const spts=TRAJ.map(d=>[d.epoch,d.implied_S]).filter(p=>p[0]!=null&&p[1]!=null&&isFinite(p[1])&&p[1]>0);
  const ptr=META.pointer||BOOT.pointer;
  if(spts.length>=3){
    const fs=linfit(spts.slice(spts.length-winSize(spts.length)));
    const ls=spts[spts.length-1][1]; const es=etaEpochs(fs?fs.m:null,ls,ptr);
    const t=(es.state==="reached")?"below pointer ✓":(es.state==="stalled")?"stalled":
      (es.state==="eta")?("~"+fmtInt(es.epochs)+" ep"):"?";
    sEl.innerHTML="implied_S (advisory) → pointer 0.19110: <b>"+t+"</b> · current "+sig(ls,4);
  } else { sEl.textContent=" "; }
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
function applySnapshot(m){TRAJ=m.trajectory||[];LIVE=m.liveness||{};META=m.meta||{};render();}
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
  LIVE=m.liveness||LIVE;META=m.meta||META;
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
  ws.onmessage=ev=>{try{const m=JSON.parse(ev.data);if(m.type==="snapshot")applySnapshot(m);else applyUpdate(m);}catch(e){}};
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
    a = ap.parse_args()
    cfg = Config(run_dir=a.run_dir, log_glob=a.log_glob, tau=a.tau, l7=a.l7,
                 goal_dseg=a.goal_dseg, goal_dseg_15=a.goal_dseg_15, poll=a.poll,
                 host=a.host, port=a.port, access_key=a.access_key,
                 cadence_state=a.cadence_state, training_pid=a.training_pid,
                 training_sig=a.training_sig,
                 auto_latest=a.auto_latest, auto_base_glob=a.auto_base_glob)
    application = create_app(cfg)
    # access_log=False so the ?k=<access key> never lands in a log line.
    uvicorn.run(application, host=cfg.host, port=cfg.port, log_level="warning",
                access_log=False, ws="websockets")


if __name__ == "__main__":
    main()
