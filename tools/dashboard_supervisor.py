#!/usr/bin/env python3
"""Lifecycle supervisor for the level-set witness ASGI dashboard (supersedes dashboard_up.py).

Operator asks (2026-06-30): a production-grade dashboard that is a SINGLETON with
AUTO-CLEANUP, CONTEXT-MANAGER teardown, and AUTO-TEARDOWN-ON-INACTIVITY, fronted by
a STABLE named Cloudflare tunnel (comma-lab.adpena.com) instead of a churning quick
tunnel.

What this manages (the dashboard set):
    {dashboard_server.py (uvicorn ASGI+WS)}  +  {cloudflared tunnel run (named, token)}

Modes:
  (default LAUNCH)  cleanup ALL existing dashboard procs (singleton), then spawn the
                    supervisor as a durable daemon (survives this process exit) which
                    brings up the server + tunnel and monitors them. Prints the URL.
  --run             the daemon body (internal): a DashboardSession context manager
                    (spawns + tears down server+tunnel, no orphan) wrapped around a
                    monitor loop (auto-teardown on inactivity + server-health restart).
  --status          list dashboard procs + the URL, change nothing.
  --down            tear everything down (stop server+tunnel+supervisor daemons).

SINGLETON + AUTO-CLEANUP: kill signatures match ONLY dashboard process kinds
(dashboard_server / render_levelset_dashboard / http.server:port / cloudflared
tunnel / other dashboard_supervisor) and ALWAYS exclude the training daemon
(``train_levelset_witness``) and this process's own group. Each managed daemon is
launched/stopped via the canonical ``tools/spawn_durable_daemon.py`` (killpg, no
orphan — CLAUDE.md daemon discipline).

NEVER touches the training daemon.
"""
from __future__ import annotations

import argparse
import atexit
import contextlib
import json
import os
import secrets
import signal
import subprocess
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)  # import sibling tools (cloudflare_named_tunnel)

TRAINING_SIG = "train_levelset_witness"  # NEVER kill
SERVER_LABEL = "levelset_dash_server"
TUNNEL_LABEL = "levelset_dash_tunnel"
SUPERVISOR_LABEL = "levelset_dash_supervisor"


# ───────────────────────── process discovery (pure-ish) ─────────────────────────
def _ps_rows() -> list[tuple[int, int, str]]:
    """Return [(pid, pgid, cmd)] for all processes."""
    out = subprocess.run(["ps", "-axww", "-o", "pid=,pgid=,command="],
                         capture_output=True, text=True).stdout
    rows: list[tuple[int, int, str]] = []
    for line in out.splitlines():
        parts = line.strip().split(None, 2)
        if len(parts) < 3:
            continue
        try:
            rows.append((int(parts[0]), int(parts[1]), parts[2]))
        except ValueError:
            continue
    return rows


def classify_proc(cmd: str, port: int) -> str | None:
    """Classify a command line into a dashboard kind, or None. NEVER classifies the
    training daemon (returns None for it)."""
    if TRAINING_SIG in cmd:
        return None
    if "dashboard_server" in cmd:
        return "server"
    if "render_levelset_dashboard" in cmd:
        return "render"
    if "dashboard_supervisor" in cmd:
        return "supervisor"
    if "http.server" in cmd and (str(port) in cmd or "dash_levelset" in cmd):
        return "httpd"
    if "cloudflared" in cmd and "tunnel" in cmd:
        return "tunnel"
    return None


def select_kill_pids(rows: list[tuple[int, int, str]], port: int,
                     exclude_pgids: set[int], kinds: set[str]) -> list[tuple[int, int, str]]:
    """Return [(pid, pgid, kind)] of dashboard procs to kill: classified in `kinds`,
    not in an excluded process group, never the training daemon."""
    out = []
    for pid, pgid, cmd in rows:
        if pgid in exclude_pgids:
            continue
        kind = classify_proc(cmd, port)
        if kind in kinds:
            out.append((pid, pgid, kind))
    return out


def _killpg(pgid: int, sig: int) -> None:
    with contextlib.suppress(Exception):
        os.killpg(pgid, sig)


def cleanup(port: int, exclude_pgids: set[int], kinds: set[str]) -> list[tuple[int, int, str]]:
    rows = _ps_rows()
    targets = select_kill_pids(rows, port, exclude_pgids, kinds)
    for _, pgid, _ in targets:
        _killpg(pgid, signal.SIGTERM)
    if targets:
        time.sleep(2)
        for pid, pgid, _ in targets:
            try:
                os.kill(pid, 0)
                _killpg(pgid, signal.SIGKILL)
            except Exception:
                pass
    return targets


# ───────────────────────── daemon spawn / stop via canonical helper ─────────────────────────
def _spawn_daemon(label: str, log: str, cmd: list[str], env: dict | None = None,
                  projected_gb: float = 1.5, rss_mb: int = 2000) -> int:
    argv = [sys.executable, os.path.join(HERE, "spawn_durable_daemon.py"),
            "--label", label, "--log", log,
            "--projected-gb", str(projected_gb), "--min-free-gb", "10",
            "--rss-cap-mb", str(rss_mb), "--walltime-cap-s", "288000", "--"] + cmd
    full_env = dict(os.environ)
    if env:
        full_env.update(env)
    return subprocess.run(argv, capture_output=True, text=True, env=full_env).returncode


def _stop_daemon(label: str) -> None:
    with contextlib.suppress(Exception):
        subprocess.run([sys.executable, os.path.join(HERE, "spawn_durable_daemon.py"),
                        "--stop", label], capture_output=True, text=True, timeout=30)


# ───────────────────────── health / activity ─────────────────────────
def healthz_ok(port: int, timeout: float = 4.0) -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/healthz", timeout=timeout) as r:
            return r.getcode() == 200
    except Exception:
        return False


def _parse_quick_url(log_path: str) -> str | None:
    import re
    pat = re.compile(r"https://[a-z0-9.-]+\.trycloudflare\.com")
    with contextlib.suppress(Exception):
        m = pat.search(open(log_path, encoding="utf-8", errors="replace").read())
        if m:
            return m.group(0)
    return None


def public_ok(hostname: str, timeout: float = 6.0) -> bool:
    try:
        with urllib.request.urlopen(f"https://{hostname}/healthz", timeout=timeout) as r:
            return r.getcode() == 200
    except Exception:
        return False


def newest_log_age_s(run_dir: str, now: float | None = None) -> float | None:
    import glob as _glob
    now = now if now is not None else time.time()
    logs = _glob.glob(os.path.join(run_dir, "*.log"))
    mtimes = []
    for p in logs:
        with contextlib.suppress(Exception):
            mtimes.append(os.path.getmtime(p))
    if not mtimes:
        return None
    return now - max(mtimes)


def _pid_alive(pid: int) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
        return True
    except Exception:
        return False


def training_alive(pid: int, sig: str) -> bool:
    """Robust training-liveness: the given pid OR any process whose cmdline carries
    the training signature. Signature-based so it survives a sibling agent
    relaunching the trainer under a NEW pid (the original --training-pid going
    stale must NOT trick the inactivity-teardown into killing a live dashboard)."""
    if _pid_alive(pid):
        return True
    if sig:
        for _, _, cmd in _ps_rows():
            if sig in cmd:
                return True
    return False


def should_teardown_inactive(newest_age_s: float | None, training_alive: bool,
                             threshold_s: float) -> bool:
    """Auto-teardown only when BOTH: the run logs have been quiet longer than the
    threshold AND the training daemon is gone. A live-but-quiet run (training
    alive) is never torn down. No logs at all + training gone => teardown."""
    if training_alive:
        return False
    if newest_age_s is None:
        return True
    return newest_age_s > threshold_s


# ───────────────────────── access key (stable, gitignored) ─────────────────────────
def load_or_make_access_key(dash_dir: str) -> str:
    path = os.path.join(dash_dir, ".access_key")
    with contextlib.suppress(Exception):
        if os.path.exists(path):
            k = open(path).read().strip()
            if k:
                return k
    k = secrets.token_urlsafe(24)
    with contextlib.suppress(Exception):
        os.makedirs(dash_dir, exist_ok=True)
        with open(path, "w") as f:
            f.write(k)
        os.chmod(path, 0o600)
    return k


# ───────────────────────── DashboardSession (context-manager teardown) ─────────────────────────
class DashboardSession:
    """Owns the server + tunnel daemons. __enter__ cleans stale dashboards, brings up
    a fresh singleton set; __exit__ (and atexit/signal) tears them down with no orphan."""

    def __init__(self, args):
        self.a = args
        self._down = False
        self.url = None
        self.mode = "none"  # "named" (stable) | "quick" (interim trycloudflare)
        self.hostname_effective = ""
        self.access_info: dict = {}
        self.access_key = ""
        self.own_pgid = os.getpgid(0)

    def __enter__(self) -> "DashboardSession":
        a = self.a
        os.makedirs(a.dash_dir, exist_ok=True)
        # 1. singleton: kill stale server/tunnel/render/httpd (NOT supervisors — LAUNCH did that; never self group)
        killed = cleanup(a.port, {self.own_pgid}, {"server", "render", "httpd", "tunnel"})
        _stop_daemon(SERVER_LABEL)
        _stop_daemon(TUNNEL_LABEL)
        print(json.dumps({"stage": "supervisor", "cleanup_killed": len(killed)}), flush=True)

        # 2. access key (stable, app-layer gate; uphold disclosure hygiene)
        self.access_key = load_or_make_access_key(a.dash_dir)

        # 3. spawn the ASGI server daemon (access key + config via env)
        rc_s = self._spawn_server()

        # 4. tunnel: named (stable) if the token allows, else QUICK fallback (app-key-gated)
        rc_t = self._bring_up_tunnel()

        # 5. wait for local server health
        local_ok = False
        for _ in range(30):
            time.sleep(1)
            if healthz_ok(a.port):
                local_ok = True
                break

        # 6. tunnel liveness = the cloudflared connector PROCESS is up. We do NOT use
        #    an HTTPS public_ok() probe here because this host's resolver may not
        #    resolve *.trycloudflare.com (Tailscale MagicDNS NXDOMAIN), which would
        #    false-negative even though the tunnel is publicly reachable.
        pub_ok = self._tunnel_alive() if not a.no_tunnel else False

        if self.url is None:
            self.url = f"https://{a.hostname}"
        self._write_url_file(local_ok, pub_ok, rc_s, rc_t)
        print(json.dumps({"stage": "supervisor", "server_rc": rc_s, "tunnel_rc": rc_t,
                          "local_health": local_ok, "public_health": pub_ok,
                          "url": self.url,
                          "access_enabled": self.access_info.get("access_enabled")}),
              flush=True)
        # register teardown safety nets
        atexit.register(self.teardown)
        for s in (signal.SIGTERM, signal.SIGINT):
            with contextlib.suppress(Exception):
                signal.signal(s, lambda *_: (self.teardown(), os._exit(0)))
        return self

    # ---- reusable bring-up (shared by __enter__ + monitor self-heal) ----
    def _server_env(self) -> dict:
        a = self.a
        return {
            "DASH_RUN_DIR": a.run_dir, "DASH_TAU": str(a.tau), "DASH_L7": str(a.l7),
            "DASH_PORT": str(a.port), "DASH_HOST": "127.0.0.1",
            "DASH_ACCESS_KEY": self.access_key, "DASH_TRAINING_PID": str(a.training_pid),
            "DASH_TRAINING_SIG": a.training_sig,
            "DASH_CADENCE_STATE": os.path.join(a.dash_dir, "cadence.json"),
            "DASH_GOAL_DSEG": str(a.goal_dseg), "DASH_GOAL_DSEG_15": str(a.goal_dseg_15),
        }

    def _tunnel_alive(self) -> bool:
        """The cloudflared connector process is up. Resolver-independent liveness
        (the local resolver may not resolve *.trycloudflare.com)."""
        return any(classify_proc(cmd, self.a.port) == "tunnel"
                   for _, pgid, cmd in _ps_rows() if pgid != self.own_pgid)

    def _spawn_server(self) -> int:
        a = self.a
        _stop_daemon(SERVER_LABEL)
        return _spawn_daemon(SERVER_LABEL, os.path.join(a.dash_dir, "server.log"),
                             [sys.executable, os.path.join(HERE, "dashboard_server.py"),
                              "--run-dir", a.run_dir, "--port", str(a.port),
                              "--tau", str(a.tau), "--l7", str(a.l7),
                              "--training-pid", str(a.training_pid)],
                             env=self._server_env(), projected_gb=1.5, rss_mb=2500)

    def _bring_up_tunnel(self) -> int:
        """Idempotently bring up the tunnel: named (stable) when the API token
        allows it, else a QUICK trycloudflare tunnel (app-key-gated). Sets
        self.mode/url/hostname_effective. Reused by __enter__ and monitor self-heal."""
        a = self.a
        if a.no_tunnel:
            self.mode = "none"
            return -1
        _stop_daemon(TUNNEL_LABEL)
        tunnel_log = os.path.join(a.dash_dir, "tunnel.log")
        tunnel_token = None
        import cloudflare_named_tunnel as cnt
        tun = cnt.ensure_named_tunnel(a.hostname, a.port, name=a.tunnel_name,
                                      allow_email=(a.allow_email or None))
        self.access_info = {k: v for k, v in tun.items() if k != "tunnel_token"}
        tunnel_token = tun.get("tunnel_token")
        if tunnel_token:
            self.mode = "named"
            self.hostname_effective = a.hostname
            self.url = f"https://{a.hostname}"
            return _spawn_daemon(TUNNEL_LABEL, tunnel_log,
                                 ["cloudflared", "tunnel", "--no-autoupdate", "run"],
                                 env={"TUNNEL_TOKEN": tunnel_token}, projected_gb=0.5, rss_mb=800)
        # quick fallback
        self.mode = "quick"
        with contextlib.suppress(Exception):
            open(tunnel_log, "w").close()  # truncate stale URL so the parse is fresh
        rc_t = _spawn_daemon(TUNNEL_LABEL, tunnel_log,
                             ["cloudflared", "tunnel", "--no-autoupdate", "--url",
                              f"http://127.0.0.1:{a.port}"], projected_gb=0.5, rss_mb=800)
        for _ in range(20):
            time.sleep(2)
            u = _parse_quick_url(tunnel_log)
            if u:
                self.url = u
                self.hostname_effective = u.split("//", 1)[-1].strip("/")
                break
        print(json.dumps({"stage": "supervisor", "named_tunnel_blocked": True,
                          "reason": self.access_info.get("error"),
                          "fallback": "quick_tunnel", "url": self.url}), flush=True)
        return rc_t

    def _write_url_file(self, local_ok: bool, pub_ok: bool, rc_s: int, rc_t: int) -> None:
        a = self.a
        path = os.path.join(a.dash_dir, "URL.txt")
        access_on = bool(self.access_info.get("access_enabled"))
        gate = ("Cloudflare Access (email one-time-PIN)" if access_on
                else "app-layer access key (public/tunnel requests need ?k=; local is open)")
        lines = [
            f"stable_url: {self.url}",
            (f"authed_url: {self.url}/?k={self.access_key}" if not access_on else f"authed_url: {self.url}"),
            f"local_url: http://127.0.0.1:{a.port}",
            f"tunnel_mode: {self.mode}",
            f"gate: {gate}",
            f"cloudflare_access_enabled: {access_on}",
            f"named_tunnel_blocked_reason: {self.access_info.get('error')}",
            f"local_health: {local_ok}",
            f"public_health: {pub_ok}",
            f"server_spawn_rc: {rc_s}",
            f"tunnel_spawn_rc: {rc_t}",
            f"written_utc: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}",
            "NOTE: this file holds a secret access key; it lives under .omx/tmp (gitignored).",
        ]
        with contextlib.suppress(Exception):
            with open(path, "w") as f:
                f.write("\n".join(lines) + "\n")
            os.chmod(path, 0o600)

    def teardown(self) -> None:
        if self._down:
            return
        self._down = True
        _stop_daemon(SERVER_LABEL)
        _stop_daemon(TUNNEL_LABEL)
        # belt-and-suspenders: kill any stragglers in our managed kinds (not self, not training)
        with contextlib.suppress(Exception):
            cleanup(self.a.port, {self.own_pgid}, {"server", "tunnel"})
        print(json.dumps({"stage": "supervisor", "torn_down": True}), flush=True)

    def __exit__(self, *exc) -> None:
        self.teardown()

    # ---- monitor loop ----
    def monitor(self) -> None:
        a = self.a
        threshold_s = a.inactivity_min * 60.0
        pub_fails = 0
        while True:
            time.sleep(a.check_interval)
            age = newest_log_age_s(a.run_dir)
            train_alive = training_alive(a.training_pid, a.training_sig)
            if should_teardown_inactive(age, train_alive, threshold_s):
                print(json.dumps({"stage": "supervisor", "auto_teardown": True,
                                  "reason": "inactive", "newest_log_age_s": age,
                                  "training_alive": train_alive}), flush=True)
                return
            # server-health self-heal
            if not healthz_ok(a.port):
                print(json.dumps({"stage": "supervisor", "server_unhealthy": True,
                                  "action": "restart"}), flush=True)
                self._spawn_server()
            # tunnel self-heal: restart only when the connector PROCESS has died
            # (resolver-independent; avoids false restarts from local DNS quirks).
            if not a.no_tunnel:
                if self._tunnel_alive():
                    pub_fails = 0
                else:
                    pub_fails += 1
                    if pub_fails >= 2:  # ~2 checks to ride out a brief reconnect
                        print(json.dumps({"stage": "supervisor", "tunnel_unhealthy": True,
                                          "action": "restart"}), flush=True)
                        rc_t = self._bring_up_tunnel()
                        self._write_url_file(healthz_ok(a.port), self._tunnel_alive(), 0, rc_t)
                        pub_fails = 0


# ───────────────────────── modes ─────────────────────────
def _add_common_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--run-dir", default="experiments/results/levelset_openpilot_seeded_n200_DEPLOY")
    ap.add_argument("--dash-dir", default=".omx/tmp/dash_levelset_deploy")
    ap.add_argument("--hostname", default="comma-lab.adpena.com")
    ap.add_argument("--tunnel-name", default="comma-lab")
    ap.add_argument("--allow-email", default="batdurston@gmail.com")
    ap.add_argument("--port", type=int, default=8790)
    ap.add_argument("--tau", type=int, default=300)
    ap.add_argument("--l7", type=int, default=600)
    ap.add_argument("--goal-dseg", type=float, default=0.00092)
    ap.add_argument("--goal-dseg-15", type=float, default=0.00032)
    ap.add_argument("--training-pid", type=int, default=26124,
                    help="hint pid; liveness also checks --training-sig (pid may go stale)")
    ap.add_argument("--training-sig", default="train_levelset_witness",
                    help="cmdline signature for robust training-liveness (survives relaunch)")
    ap.add_argument("--inactivity-min", type=float, default=30.0)
    ap.add_argument("--check-interval", type=float, default=60.0)
    ap.add_argument("--no-tunnel", action="store_true",
                    help="local-only: bring up the ASGI server, skip the tunnel")


def _url_from_file(dash_dir: str) -> str | None:
    path = os.path.join(dash_dir, "URL.txt")
    with contextlib.suppress(Exception):
        for line in open(path):
            if line.startswith("stable_url:"):
                return line.split(":", 1)[1].strip()
    return None


def cmd_status(a) -> int:
    rows = _ps_rows()
    found = select_kill_pids(rows, a.port, set(), {"server", "render", "httpd", "tunnel", "supervisor"})
    print(f"dashboard procs -> {len(found)}")
    for pid, pgid, kind in found:
        print(f"  {kind:10s} pid={pid} pgid={pgid}")
    print("local_health:", healthz_ok(a.port))
    print("public_health:", public_ok(a.hostname))
    print("url:", _url_from_file(a.dash_dir))
    return 0


def cmd_down(a) -> int:
    _stop_daemon(SUPERVISOR_LABEL)
    _stop_daemon(SERVER_LABEL)
    _stop_daemon(TUNNEL_LABEL)
    own = {os.getpgid(0)}
    killed = cleanup(a.port, own, {"server", "render", "httpd", "tunnel", "supervisor"})
    print(json.dumps({"stage": "supervisor", "down": True, "killed": len(killed)}))
    return 0


def cmd_run(a) -> int:
    """Daemon body: context-managed session + monitor loop."""
    with DashboardSession(a) as sess:
        sess.monitor()
    return 0


def cmd_launch(a) -> int:
    """Default: enforce singleton, spawn the supervisor daemon, report the URL."""
    own = {os.getpgid(0)}
    killed = cleanup(a.port, own, {"server", "render", "httpd", "tunnel", "supervisor"})
    _stop_daemon(SUPERVISOR_LABEL)
    print(json.dumps({"stage": "supervisor_launch", "cleanup_killed": len(killed)}))
    os.makedirs(a.dash_dir, exist_ok=True)
    # delete the stale URL.txt so the launch-poll below reports the FRESH url the
    # new daemon writes (not the prior run's, which would be a dead tunnel).
    with contextlib.suppress(Exception):
        os.remove(os.path.join(a.dash_dir, "URL.txt"))
    cmd = [sys.executable, os.path.join(HERE, "dashboard_supervisor.py"), "--run",
           "--run-dir", a.run_dir, "--dash-dir", a.dash_dir, "--hostname", a.hostname,
           "--tunnel-name", a.tunnel_name, "--allow-email", a.allow_email,
           "--port", str(a.port), "--tau", str(a.tau), "--l7", str(a.l7),
           "--goal-dseg", str(a.goal_dseg), "--goal-dseg-15", str(a.goal_dseg_15),
           "--training-pid", str(a.training_pid), "--training-sig", a.training_sig,
           "--inactivity-min", str(a.inactivity_min),
           "--check-interval", str(a.check_interval)]
    if a.no_tunnel:
        cmd.append("--no-tunnel")
    rc = _spawn_daemon(SUPERVISOR_LABEL, os.path.join(a.dash_dir, "supervisor.log"),
                       cmd, projected_gb=1.0, rss_mb=1500)
    print(json.dumps({"stage": "supervisor_launch", "daemon_rc": rc}))
    # wait for the daemon to write URL.txt + come up
    url = None
    for _ in range(45):
        time.sleep(2)
        if healthz_ok(a.port) and os.path.exists(os.path.join(a.dash_dir, "URL.txt")):
            url = _url_from_file(a.dash_dir)
            break
    print("STABLE_URL:", url or f"https://{a.hostname} (pending — check --status)")
    print("URL_FILE:", os.path.join(a.dash_dir, "URL.txt"))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--run", action="store_true", help="(internal) daemon body")
    g.add_argument("--status", action="store_true", help="list dashboard procs + URL")
    g.add_argument("--down", action="store_true", help="tear everything down")
    _add_common_args(ap)
    a = ap.parse_args()
    if a.status:
        return cmd_status(a)
    if a.down:
        return cmd_down(a)
    if a.run:
        return cmd_run(a)
    return cmd_launch(a)


if __name__ == "__main__":
    raise SystemExit(main())
