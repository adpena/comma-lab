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

    def resolved_glob(self) -> str:
        if self.log_glob:
            return self.log_glob
        if self.run_dir:
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
        self.started_at = time.time()
        self.clients: set[WebSocket] = set()
        # cadence sub-state (per-log), persisted to disk like render_levelset_dashboard
        self._cad_args = SimpleNamespace(stale_min=None, stale_floor_min=10.0,
                                         cadence_k=2.5, cadence_prior_min=18.0)

    # ---- refresh (sync; called via executor) ----
    def refresh(self) -> list[dict]:
        """Re-resolve the newest verdict log, parse verdicts, update liveness.
        Returns the list of NEW trajectory points (epochs not seen before)."""
        cfg = self.cfg
        watched = rld._resolve_watched_log(None, cfg.resolved_glob())
        rows = rld._parse_verdicts(watched) if watched is not None else []
        now = time.time()
        ccfg = rld._cfg_from_args(self._cad_args)
        log_name = watched.name if watched is not None else "_none_"
        all_state, sub = rld._load_cadence_state(cfg.cadence_state, log_name)
        mtime = watched.stat().st_mtime if (watched is not None and watched.exists()) else None
        self.liveness = rld._compute_liveness(rows, mtime, now, sub, ccfg)
        rld._save_cadence_state(cfg.cadence_state, all_state, log_name, sub)
        self.watched = watched.name if watched is not None else None

        new_points: list[dict] = []
        for r in rows:
            ep = r.get("epoch")
            if isinstance(ep, int) and ep not in self._epochs:
                self._epochs.add(ep)
                slim = _slim(r)
                self.trajectory.append(slim)
                new_points.append(slim)
        self.trajectory.sort(key=lambda d: (d["epoch"] if d["epoch"] is not None else 0))
        return new_points

    # ---- snapshot for client ----
    def meta(self) -> dict:
        cfg = self.cfg
        return {
            "tau": cfg.tau, "l7": cfg.l7,
            "goal_dseg": cfg.goal_dseg, "goal_dseg_15": cfg.goal_dseg_15,
            "pointer": POINTER, "watched": self.watched, "run_dir": cfg.run_dir,
            "uptime_s": time.time() - self.started_at,
            "training_alive": _training_alive(cfg.training_pid, cfg.training_sig),
            "n_points": len(self.trajectory),
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
:root{--bg:#14161a;--panel:#1b1e24;--fg:#d8dde6;--muted:#8b93a3;--grid:#2c313b;
--acc:#5ab0ff;--goal:#46d369;--pose:#ffb454;--bytes:#c08cff;--sval:#ff6b6b}
*{box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',system-ui,sans-serif;
background:var(--bg);color:var(--fg);margin:0;padding:0;line-height:1.5}
.wrap{max-width:1180px;margin:0 auto;padding:18px 16px 40px}
.auth{background:#221a0e;border:1px solid #4a3a12;color:#e6cf7a;font-size:12px;
padding:8px 12px;border-radius:7px;margin-bottom:14px;text-align:center}
.head{display:flex;align-items:center;flex-wrap:wrap;gap:10px;margin-bottom:6px}
.title{font-size:13px;color:var(--muted);letter-spacing:1.6px;text-transform:uppercase;font-weight:600}
.pill{font-size:11.5px;font-weight:600;padding:3px 10px;border-radius:11px}
.pill.live{background:#173d22;color:#7fe0a0}.pill.warm{background:#3a3413;color:#e6cf7a}
.pill.stale{background:#4a1717;color:#ff9b9b}.pill.miss{background:#3a1f1f;color:#ff9b9b}
.pill.ws{background:#16263a;color:#7fc0ff}.pill.wsoff{background:#3a2a16;color:#e6b97a}
.tabs{display:flex;gap:6px;margin:14px 0 16px;border-bottom:1px solid var(--grid)}
.tab{font-size:13px;color:var(--muted);padding:8px 14px;cursor:pointer;border-bottom:2px solid transparent}
.tab.on{color:var(--fg);border-bottom-color:var(--acc)}
.hl{font-size:20px;font-weight:300;margin:2px 0}
.hl b{color:var(--acc);font-size:34px;font-weight:600;font-variant-numeric:tabular-nums;margin:0 4px}
.arr{font-size:15px;color:var(--muted)}.goal{font-size:12px;color:var(--muted);margin-left:12px}
.status{font-size:14px;margin:8px 0 2px}.detail{font-size:11px;color:var(--muted);
margin-bottom:14px;font-variant-numeric:tabular-nums}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:14px}
@media(max-width:760px){.grid{grid-template-columns:1fr}}
.panel{background:var(--panel);border:1px solid var(--grid);border-radius:9px;padding:8px 8px 4px}
.panel canvas{width:100%;height:230px;display:block}
.foot{font-size:10.5px;color:var(--muted);opacity:.8;margin-top:18px;line-height:1.7;text-align:center}
.tri h2{font-size:13px;color:var(--acc);letter-spacing:.5px;margin:20px 0 6px}
.tri p,.tri li{font-size:13px;color:var(--fg)}
.tri .m{color:var(--muted);font-size:12px}
.tri code{background:#11141a;color:#9fc6ff;padding:1px 5px;border-radius:4px;font-size:12px}
.tri .cards{display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px;margin:8px 0}
@media(max-width:760px){.tri .cards{grid-template-columns:1fr}}
.tri .card{background:var(--panel);border:1px solid var(--grid);border-radius:9px;padding:12px}
.tri .card h3{font-size:12px;color:var(--goal);margin:0 0 6px;letter-spacing:.5px;text-transform:uppercase}
.tri ol{padding-left:18px}.tri ul{padding-left:18px}
.hide{display:none}
</style></head>
<body><div class="wrap">
<div class="auth">[macOS-MLX training] advisory &middot; <b>NON-PROMOTABLE</b> &mdash; the exact contest row is byte-closed on contest-CPU/CUDA &middot; frontier pointer <b>0.19110</b> (UNMOVED). A dashboard is a MEANS, not the score.</div>
<div class="head">
  <span class="title">Level-Set Witness</span>
  <span id="pill" class="pill miss">&middot; connecting</span>
  <span id="wspill" class="pill wsoff">ws &hellip;</span>
</div>
<div class="tabs">
  <div class="tab on" data-tab="live">LIVE</div>
  <div class="tab" data-tab="tri">TRIALITY &middot; PLAN</div>
</div>

<section id="tab-live">
  <div class="hl" id="headline">waiting for ep0 verdict&hellip;</div>
  <div class="status" id="status">connecting&hellip;</div>
  <div class="detail" id="detail"></div>
  <div class="grid">
    <div class="panel"><canvas id="c_dseg"></canvas></div>
    <div class="panel"><canvas id="c_dpose"></canvas></div>
    <div class="panel"><canvas id="c_bytes"></canvas></div>
    <div class="panel"><canvas id="c_s"></canvas></div>
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
<script>
const BOOT = __BOOT__;
let TRAJ = [], LIVE = {}, META = {};
let ws = null, wsOpen = false, wsTries = 0, pollTimer = null;

function $(id){return document.getElementById(id);}
function fmtAge(s){if(s==null)return "?";s=Math.max(0,s|0);if(s<90)return s+"s";let m=s/60;if(m<90)return m.toFixed(1)+"m";return (m/60).toFixed(1)+"h";}
function fmtNum(v,d){return (v==null||isNaN(v))?"—":Number(v).toFixed(d);}

// ---------- tabs ----------
document.querySelectorAll(".tab").forEach(t=>t.onclick=()=>{
  document.querySelectorAll(".tab").forEach(x=>x.classList.remove("on"));
  t.classList.add("on");
  const which=t.dataset.tab;
  $("tab-live").classList.toggle("hide",which!=="live");
  $("tab-tri").classList.toggle("hide",which!=="tri");
  if(which==="live") drawAll();
});

// ---------- canvas chart ----------
function drawPanel(canvas, key, opt){
  const dpr=window.devicePixelRatio||1;
  const W=canvas.clientWidth||560, H=canvas.clientHeight||230;
  canvas.width=W*dpr; canvas.height=H*dpr;
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
  // stage shading
  const tau=META.tau||BOOT.tau, l7=META.l7||BOOT.l7;
  const spans=[[xmin,Math.min(tau,xmax),"#1f3b5f"],[Math.max(tau,xmin),Math.min(l7,xmax),"#3a2a5f"],[Math.max(l7,xmin),xmax,"#5f3320"]];
  ctx.globalAlpha=0.18;
  spans.forEach(s=>{if(s[1]>s[0]){ctx.fillStyle=s[2];ctx.fillRect(sx(s[0]),y0,sx(s[1])-sx(s[0]),y1-y0);}});
  ctx.globalAlpha=1;
  // grid + y ticks
  ctx.strokeStyle="#2c313b"; ctx.fillStyle="#8b93a3"; ctx.font="10px system-ui"; ctx.lineWidth=1;
  const nT=4;
  for(let i=0;i<=nT;i++){
    const lv=Lmin+(Lmax-Lmin)*i/nT; const val=log?Math.pow(10,lv):lv; const yy=sy(val);
    ctx.globalAlpha=0.5;ctx.beginPath();ctx.moveTo(x0,yy);ctx.lineTo(x1,yy);ctx.stroke();ctx.globalAlpha=1;
    let lab = (log? val.toExponential(1) : (Math.abs(val)>=1000? (val/1000).toFixed(0)+"k" : val.toPrecision(3)));
    ctx.textAlign="right";ctx.fillText(lab,x0-5,yy+3);
  }
  // x ticks
  ctx.textAlign="center";
  for(let i=0;i<=4;i++){const e=xmin+(xmax-xmin)*i/4;ctx.fillText(Math.round(e),sx(e),y1+14);}
  // hlines (goals)
  (opt.hlines||[]).forEach(h=>{
    if(h.y==null||(log&&h.y<=0))return;
    const yy=sy(h.y); ctx.strokeStyle=h.color||"#46d369"; ctx.setLineDash([4,3]); ctx.lineWidth=1.2;
    ctx.beginPath();ctx.moveTo(x0,yy);ctx.lineTo(x1,yy);ctx.stroke();ctx.setLineDash([]);
    ctx.fillStyle=h.color||"#46d369";ctx.textAlign="left";ctx.fillText(h.label||"",x0+4,yy-3);
  });
  // stage vlines
  [[tau,"tau"],[l7,"l7"]].forEach(s=>{
    if(s[0]<xmin||s[0]>xmax)return;const xx=sx(s[0]);
    ctx.strokeStyle="#8b93a3";ctx.setLineDash([3,3]);ctx.globalAlpha=0.7;ctx.lineWidth=1;
    ctx.beginPath();ctx.moveTo(xx,y0);ctx.lineTo(xx,y1);ctx.stroke();ctx.setLineDash([]);ctx.globalAlpha=1;
    ctx.fillStyle="#d8dde6";ctx.textAlign="left";ctx.fillText(s[1],xx+3,y0+10);
  });
  // series
  if(pts.length){
    ctx.strokeStyle=opt.color;ctx.lineWidth=1.8;ctx.beginPath();
    pts.forEach((p,i)=>{const X=sx(p[0]),Y=sy(p[1]);i?ctx.lineTo(X,Y):ctx.moveTo(X,Y);});
    ctx.stroke();
    ctx.fillStyle=opt.color;
    pts.forEach(p=>{ctx.beginPath();ctx.arc(sx(p[0]),sy(p[1]),2.6,0,7);ctx.fill();});
    const last=pts[pts.length-1];
    ctx.fillStyle="#d8dde6";ctx.textAlign="left";ctx.font="11px system-ui";
    ctx.fillText((log?last[1].toExponential(2):last[1].toPrecision(4)),Math.min(sx(last[0])+5,x1-60),sy(last[1])-5);
  }
  // title + sub
  ctx.fillStyle="#d8dde6";ctx.font="11.5px system-ui";ctx.textAlign="left";ctx.fillText(opt.title,x0,14);
  ctx.fillStyle="#8b93a3";ctx.font="10px system-ui";ctx.fillText(opt.sub,x0,y1+24);
}

function drawAll(){
  const g=META.goal_dseg||BOOT.goal_dseg, g15=META.goal_dseg_15||BOOT.goal_dseg_15;
  drawPanel($("c_dseg"),"d_seg",{title:"d_seg — realized SegNet-argmax disagreement (lower better)",
    sub:"epoch · log scale · goal lines = sub-0.19 / sub-0.15",color:"#5ab0ff",log:true,
    hlines:[{y:g,label:"sub-0.19 ("+g+")",color:"#46d369"},{y:g15,label:"sub-0.15 ("+g15+")",color:"#ffb454"}]});
  drawPanel($("c_dpose"),"d_pose",{title:"d_pose — realized PoseNet MSE (pose on sidecar)",
    sub:"epoch · log scale · existence-proof ~9e-4",color:"#ffb454",log:true,
    hlines:[{y:9e-4,label:"~9e-4",color:"#46d369"}]});
  drawPanel($("c_bytes"),"blob_bytes",{title:"blob_bytes — LEARNED payload (counted in archive)",
    sub:"epoch · smaller payload = lower rate term",color:"#c08cff",log:false,hlines:[]});
  drawPanel($("c_s"),"implied_S",{title:"implied_S — ADVISORY mid-training estimate (NOT the contest score)",
    sub:"epoch · log scale · frontier pointer = 0.19110",color:"#ff6b6b",log:true,
    hlines:[{y:META.pointer||BOOT.pointer,label:"pointer 0.19110",color:"#46d369"}]});
}

function stageWord(ep){if(ep==null)return "starting";const tau=META.tau||BOOT.tau,l7=META.l7||BOOT.l7;
  if(ep<tau)return "CE";if(ep<l7)return "tau";return "l7/Muon";}

function render(){
  const last=TRAJ.length?TRAJ[TRAJ.length-1]:null;
  const g=META.goal_dseg||BOOT.goal_dseg;
  // pill
  const k=LIVE.kind, p=$("pill");
  if(k==="live"&&!LIVE.calibrating){p.className="pill live";p.innerHTML="● live";}
  else if(k==="live"&&LIVE.calibrating){p.className="pill warm";p.innerHTML="◐ warming up";}
  else if(k==="stale"){p.className="pill stale";p.innerHTML="⚠ stale";}
  else{p.className="pill miss";p.innerHTML="⚠ no run log";}
  // headline
  if(last){
    let arrow="·";
    if(TRAJ.length>=2){const a=TRAJ[TRAJ.length-1].d_seg,b=TRAJ[Math.max(0,TRAJ.length-4)].d_seg;
      if(a!=null&&b!=null)arrow=a<b?"▼":(a>b?"▲":"▬");}
    $("headline").innerHTML="d_seg <b>"+fmtNum(last.d_seg,5)+"</b><span class='arr'>"+arrow+
      "</span><span class='goal'>goal &lt;"+g+"</span>";
  }
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
  $("detail").textContent=d.join(" · ");
  // foot
  $("foot").textContent="[macOS-MLX advisory · NON-PROMOTABLE] · pointer 0.19110 · stages CE · tau · l7 · Muon"+
    (META.watched?(" · "+META.watched):"")+" · "+TRAJ.length+" verdicts";
  // boot spans in triality
  $("b_tau").textContent=META.tau||BOOT.tau; $("b_l7").textContent=META.l7||BOOT.l7;
  $("b_goal").textContent=META.goal_dseg||BOOT.goal_dseg;
  if(!$("tab-live").classList.contains("hide")) drawAll();
}

function applySnapshot(m){TRAJ=m.trajectory||[];LIVE=m.liveness||{};META=m.meta||{};render();}
function applyUpdate(m){
  const np=m.new_points||[];
  if(np.length){const seen=new Set(TRAJ.map(d=>d.epoch));np.forEach(p=>{if(!seen.has(p.epoch))TRAJ.push(p);});
    TRAJ.sort((a,b)=>a.epoch-b.epoch);}
  LIVE=m.liveness||LIVE;META=m.meta||META;render();
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

window.addEventListener("resize",()=>{if(!$("tab-live").classList.contains("hide"))drawAll();});
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
    a = ap.parse_args()
    cfg = Config(run_dir=a.run_dir, log_glob=a.log_glob, tau=a.tau, l7=a.l7,
                 goal_dseg=a.goal_dseg, goal_dseg_15=a.goal_dseg_15, poll=a.poll,
                 host=a.host, port=a.port, access_key=a.access_key,
                 cadence_state=a.cadence_state, training_pid=a.training_pid,
                 training_sig=a.training_sig)
    application = create_app(cfg)
    # access_log=False so the ?k=<access key> never lands in a log line.
    uvicorn.run(application, host=cfg.host, port=cfg.port, log_level="warning",
                access_log=False, ws="websockets")


if __name__ == "__main__":
    main()
