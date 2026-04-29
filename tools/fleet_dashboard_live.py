#!/usr/bin/env python3
"""Live fleet telemetry dashboard with WebSocket push.

Real-time experiment monitoring: live epoch counters, score trajectories,
process health, competition context, and action-needed alerts.

Usage:
    .venv/bin/python tools/fleet_dashboard_live.py [--port 8780]
"""
from __future__ import annotations

import asyncio
import json
import os
import subprocess
import time
import webbrowser
from datetime import datetime, timezone, timedelta
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from threading import Thread

import websockets

REPO = Path(__file__).parent.parent
WEIGHTS_DIR = REPO / "experiments" / "postfilter_weights"

# Competition constants
CONTEST_DEADLINE = datetime(2026, 5, 3, 23, 59, tzinfo=timezone(timedelta(hours=-7)))  # PT
PROMOTED_SCORE = 1.727
PROMOTED_DATE = datetime(2026, 4, 9, 10, 0, tzinfo=timezone(timedelta(hours=-5)))  # CT
LEADERBOARD = [
    {"rank": 1, "name": "PACT (ours)", "score": 1.727},
    {"rank": 2, "name": "neural_inflate", "score": 1.89},
    {"rank": 3, "name": "roi_v2", "score": 1.94},
    {"rank": 4, "name": "av1_roi_lanczos_unsharp", "score": 1.95},
]
PRACTICAL_FLOOR = 1.50
THEORETICAL_FLOOR = 1.20
NEXT_TARGET = 1.60

# Track score history per experiment
SCORE_HISTORIES: dict[str, list[dict]] = {}


def get_processes() -> list[dict]:
    result = subprocess.run(["ps", "aux"], capture_output=True, text=True)
    procs = []
    seen_pids = set()
    for line in result.stdout.splitlines():
        if "python" not in line or "grep" in line or "dashboard" in line or "http.server" in line:
            continue
        if not any(k in line for k in ["train", "segnet", "proxy", "evaluate"]):
            continue
        parts = line.split()
        pid = parts[1]
        if pid in seen_pids:
            continue
        # Skip wrapper processes (zsh, uv run)
        cmd = " ".join(parts[10:])
        if cmd.startswith("/bin/zsh") or cmd.startswith("uv run"):
            continue
        seen_pids.add(pid)
        tag = ""
        if "--tag" in cmd:
            # IndexError if --tag is the last token (no value follows);
            # other exceptions would indicate a real bug, so don't swallow.
            try:
                tag = cmd.split("--tag")[1].strip().split()[0]
            except IndexError:
                tag = ""
        kind = "trainer"
        if "proxy" in cmd or "evaluate" in cmd:
            kind = "proxy"
        elif "segnet_boundary" in cmd:
            kind = "boundary"
        procs.append({
            "pid": pid, "cpu": float(parts[2]), "mem_gb": round(int(parts[5]) / 1048576, 1),
            "tag": tag, "kind": kind, "cmd": cmd[:100],
            "runtime_min": round(float(parts[9].split(":")[0]) if ":" in parts[9] else 0),
        })
    return procs


def get_checkpoints() -> list[dict]:
    checkpoints = []
    if not WEIGHTS_DIR.exists():
        return checkpoints
    for mf in sorted(WEIGHTS_DIR.glob("*_best_meta.json"), key=os.path.getmtime, reverse=True):
        try:
            d = json.loads(mf.read_text())
            tag = mf.stem.replace("postfilter_", "").replace("_best_meta", "")
            age = int(time.time() - mf.stat().st_mtime)
            d["tag"] = tag
            d["age_s"] = age
            d["active"] = age < 600  # checkpoint updated in last 10 min = active experiment

            # Track history
            if tag not in SCORE_HISTORIES:
                SCORE_HISTORIES[tag] = []
            hist = SCORE_HISTORIES[tag]
            if not hist or hist[-1]["epoch"] != d.get("epoch"):
                hist.append({"t": time.time(), "epoch": d.get("epoch", 0), "scorer": d.get("scorer", 0)})
            # Keep last 200 points
            SCORE_HISTORIES[tag] = hist[-200:]

            # Compute trend (score change per epoch over last 5 checkpoints)
            if len(hist) >= 2:
                recent = hist[-5:]
                if recent[-1]["epoch"] > recent[0]["epoch"]:
                    d["trend"] = (recent[-1]["scorer"] - recent[0]["scorer"]) / (recent[-1]["epoch"] - recent[0]["epoch"])
                    epochs_to_target = (d["scorer"] - 3.55) / abs(d["trend"]) if d["trend"] < 0 else float("inf")
                    d["eta_epochs"] = max(0, int(epochs_to_target))
                else:
                    d["trend"] = 0
                    d["eta_epochs"] = None
            else:
                d["trend"] = 0
                d["eta_epochs"] = None

            checkpoints.append(d)
        except Exception:
            pass
    return checkpoints


def get_alerts(procs, ckpts) -> list[dict]:
    alerts = []
    now = datetime.now(timezone.utc)
    deadline = CONTEST_DEADLINE.astimezone(timezone.utc)
    days_left = (deadline - now).total_seconds() / 86400

    trainers = [p for p in procs if p["kind"] == "trainer"]
    if len(trainers) < 2:
        alerts.append({"level": "warn", "msg": f"Only {len(trainers)} trainer(s) running — should be 2+"})
    if len(trainers) == 0:
        alerts.append({"level": "error", "msg": "No trainers running! Restart needed."})

    active_ckpts = [c for c in ckpts if c.get("active")]
    if not active_ckpts:
        alerts.append({"level": "warn", "msg": "No checkpoints updated in 10 min — training may be stuck"})

    if days_left < 1:
        alerts.append({"level": "error", "msg": f"FINAL DAY — {days_left*24:.1f} hours left"})
    elif days_left < 5:
        alerts.append({"level": "warn", "msg": f"Under 5 days left ({days_left:.1f}d)"})

    best = min(ckpts, key=lambda c: c.get("scorer", 999)) if ckpts else None
    if best and best.get("eta_epochs") and best["eta_epochs"] > 2000:
        alerts.append({"level": "info", "msg": f"Best experiment needs ~{best['eta_epochs']} more epochs to proxy threshold"})

    if not alerts:
        alerts.append({"level": "ok", "msg": "All systems nominal — no action needed"})

    return alerts


def collect_state() -> dict:
    procs = get_processes()
    ckpts = get_checkpoints()
    alerts = get_alerts(procs, ckpts)

    now = datetime.now(timezone.utc)
    deadline = CONTEST_DEADLINE.astimezone(timezone.utc)
    days_left = (deadline - now).total_seconds() / 86400
    since_last = (now - PROMOTED_DATE.astimezone(timezone.utc)).total_seconds() / 3600

    return {
        "ts": time.strftime("%H:%M:%S"),
        "competition": {
            "score": PROMOTED_SCORE,
            "lead": round(LEADERBOARD[1]["score"] - PROMOTED_SCORE, 3),
            "rank": 1,
            "leaderboard": LEADERBOARD,
            "days_left": round(days_left, 1),
            "hours_since_breakthrough": round(since_last, 1),
            "next_target": NEXT_TARGET,
            "practical_floor": PRACTICAL_FLOOR,
            "theoretical_floor": THEORETICAL_FLOOR,
            "headroom": round(PROMOTED_SCORE - PRACTICAL_FLOOR, 3),
        },
        "procs": procs,
        "experiments": [c for c in ckpts if c.get("active")],
        "all_checkpoints": ckpts[:15],
        "histories": {k: v[-50:] for k, v in SCORE_HISTORIES.items() if v},
        "alerts": alerts,
    }


HTML = r"""<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<title>comma-lab telemetry</title>
<script src="https://d3js.org/d3.v7.min.js"></script>
<style>
:root { --bg: #0f1724; --surface: #161b22; --border: #30363d; --text: #c9d1d9; --muted: #8b949e; --accent: #8ed1c0; --warn: #f0b35f; --error: #f85149; --ok: #3fb950; }
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font: 13px/1.5 -apple-system, SF Mono, monospace; background: var(--bg); color: var(--text); padding: 16px 20px; }
h1 { font-size: 15px; color: var(--accent); }
.meta { font-size: 11px; color: var(--muted); }
.grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin: 12px 0; }
.card { background: var(--surface); border: 1px solid var(--border); border-radius: 6px; padding: 10px; }
.card h3 { font-size: 10px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.06em; }
.metric { font-size: 26px; font-weight: 700; color: var(--accent); letter-spacing: -0.03em; }
.metric.small { font-size: 18px; }
.alert { padding: 6px 10px; border-radius: 4px; font-size: 12px; margin: 2px 0; }
.alert.ok { background: rgba(63,185,80,0.1); color: var(--ok); border-left: 3px solid var(--ok); }
.alert.warn { background: rgba(240,179,95,0.1); color: var(--warn); border-left: 3px solid var(--warn); }
.alert.error { background: rgba(248,81,73,0.1); color: var(--error); border-left: 3px solid var(--error); }
.alert.info { background: rgba(142,209,192,0.1); color: var(--accent); border-left: 3px solid var(--accent); }
h2 { font-size: 11px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.06em; margin: 14px 0 6px; }
table { width: 100%; border-collapse: collapse; }
th { text-align: left; color: var(--muted); font-size: 10px; text-transform: uppercase; padding: 3px 6px; border-bottom: 1px solid var(--border); }
td { padding: 3px 6px; border-bottom: 1px solid #1c2333; font-size: 12px; }
.tag { color: var(--warn); }
.score { color: var(--accent); font-weight: 700; }
.trend-down { color: var(--ok); }
.trend-up { color: var(--error); }
.fresh { color: var(--ok); }
.stale { color: var(--error); }
.chart-container { background: var(--surface); border: 1px solid var(--border); border-radius: 6px; padding: 10px; margin: 8px 0; }
.chart-container svg { width: 100%; height: 180px; }
.axis text { fill: var(--muted); font-size: 10px; }
.axis line, .axis path { stroke: var(--border); }
.dot-label { font-size: 9px; fill: var(--muted); }
</style>
</head><body>
<h1>comma-lab telemetry</h1>
<div class="meta" id="status">connecting...</div>

<div id="alerts"></div>

<div class="grid">
  <div class="card"><h3>Score</h3><div class="metric" id="score">--</div><div class="meta" id="scoreMeta">--</div></div>
  <div class="card"><h3>Days Left</h3><div class="metric" id="days">--</div><div class="meta" id="daysMeta">--</div></div>
  <div class="card"><h3>Since Breakthrough</h3><div class="metric small" id="since">--</div><div class="meta">last score improvement</div></div>
  <div class="card"><h3>Headroom</h3><div class="metric small" id="headroom">--</div><div class="meta" id="headroomMeta">--</div></div>
</div>

<h2>Live Experiments</h2>
<table id="expTable"><tr><th>Tag</th><th>Epoch</th><th>Scorer</th><th>Trend</th><th>ETA to proxy</th><th>Updated</th></tr></table>

<h2>Score Trajectories</h2>
<div class="chart-container"><svg id="chart"></svg></div>

<h2>Fleet Processes</h2>
<table id="procTable"><tr><th>PID</th><th>Kind</th><th>CPU</th><th>RAM</th><th>Tag</th></tr></table>

<h2>Leaderboard</h2>
<table id="lbTable"><tr><th>#</th><th>Score</th><th>Name</th></tr></table>

<script>
let ws;
function connect() {
  ws = new WebSocket('ws://localhost:8781');
  ws.onopen = () => document.getElementById('status').textContent = 'live — push every 3s';
  ws.onclose = () => { document.getElementById('status').textContent = 'disconnected'; setTimeout(connect, 2000); };
  ws.onmessage = (e) => update(JSON.parse(e.data));
}

function update(d) {
  const c = d.competition;
  document.getElementById('score').textContent = c.score.toFixed(3);
  document.getElementById('scoreMeta').textContent = `#${c.rank} · +${c.lead.toFixed(3)} lead`;
  document.getElementById('days').textContent = c.days_left.toFixed(1);
  document.getElementById('daysMeta').textContent = c.days_left < 5 ? '⚠️ UNDER 5 DAYS' : 'until deadline';
  const hrs = c.hours_since_breakthrough;
  document.getElementById('since').textContent = hrs < 24 ? hrs.toFixed(1)+'h' : (hrs/24).toFixed(1)+'d';
  document.getElementById('headroom').textContent = c.headroom.toFixed(3);
  document.getElementById('headroomMeta').textContent = `target ${c.next_target} · floor ${c.practical_floor}`;

  // Alerts
  const al = document.getElementById('alerts');
  al.innerHTML = d.alerts.map(a => `<div class="alert ${a.level}">${a.msg}</div>`).join('');

  // Experiments
  const et = document.getElementById('expTable');
  et.innerHTML = '<tr><th>Tag</th><th>Epoch</th><th>Scorer</th><th>Trend/ep</th><th>ETA to proxy</th><th>Updated</th></tr>';
  d.experiments.forEach(e => {
    const age = e.age_s < 60 ? e.age_s+'s' : Math.floor(e.age_s/60)+'m';
    const trend = e.trend ? (e.trend < 0 ? `<span class="trend-down">${e.trend.toFixed(4)}</span>` : `<span class="trend-up">+${e.trend.toFixed(4)}</span>`) : '--';
    const eta = e.eta_epochs != null ? (e.eta_epochs < 9999 ? `~${e.eta_epochs} ep` : 'far') : '--';
    et.innerHTML += `<tr><td class="tag">${e.tag}</td><td>${e.epoch}</td><td class="score">${e.scorer.toFixed(4)}</td><td>${trend}</td><td>${eta}</td><td class="${e.age_s<120?'fresh':'stale'}">${age}</td></tr>`;
  });

  // Processes
  const pt = document.getElementById('procTable');
  pt.innerHTML = '<tr><th>PID</th><th>Kind</th><th>CPU</th><th>RAM</th><th>Tag</th></tr>';
  d.procs.forEach(p => {
    pt.innerHTML += `<tr><td>${p.pid}</td><td>${p.kind}</td><td>${p.cpu}%</td><td>${p.mem_gb}GB</td><td class="tag">${p.tag||p.cmd.slice(0,40)}</td></tr>`;
  });

  // Leaderboard
  const lb = document.getElementById('lbTable');
  lb.innerHTML = '<tr><th>#</th><th>Score</th><th>Name</th></tr>';
  c.leaderboard.forEach(e => {
    const us = e.name.includes('ours') ? ' ← us' : '';
    lb.innerHTML += `<tr><td>${e.rank}</td><td class="score">${e.score.toFixed(3)}</td><td>${e.name}${us}</td></tr>`;
  });

  // Chart
  drawChart(d.histories);
}

const colors = ['#8ed1c0','#f0b35f','#bc8cff','#f85149','#8ab4ff','#f0a0c0'];
function drawChart(histories) {
  const svg = d3.select('#chart');
  svg.selectAll('*').remove();
  const entries = Object.entries(histories).filter(([,v]) => v.length > 1);
  if (!entries.length) return;

  const w = svg.node().parentElement.clientWidth - 20;
  const h = 170;
  const m = {top:8, right:60, bottom:22, left:50};

  const allPts = entries.flatMap(([,v]) => v);
  const x = d3.scaleLinear().domain(d3.extent(allPts, d=>d.epoch)).range([m.left, w-m.right]);
  const yMin = Math.min(3.5, d3.min(allPts, d=>d.scorer)-0.1);
  const yMax = d3.max(allPts, d=>d.scorer)+0.1;
  const y = d3.scaleLinear().domain([yMin, yMax]).range([h-m.bottom, m.top]);

  svg.attr('viewBox', `0 0 ${w} ${h}`);
  svg.append('g').attr('class','axis').attr('transform',`translate(0,${h-m.bottom})`).call(d3.axisBottom(x).ticks(6));
  svg.append('g').attr('class','axis').attr('transform',`translate(${m.left},0)`).call(d3.axisLeft(y).ticks(5));

  // Proxy threshold line
  svg.append('line').attr('x1',m.left).attr('x2',w-m.right).attr('y1',y(3.55)).attr('y2',y(3.55))
    .attr('stroke','#3fb950').attr('stroke-dasharray','4 2').attr('opacity',0.5);
  svg.append('text').attr('x',w-m.right+4).attr('y',y(3.55)+3).attr('fill','#3fb950').attr('font-size',9).text('proxy');

  entries.forEach(([tag, pts], i) => {
    const color = colors[i % colors.length];
    const line = d3.line().x(d=>x(d.epoch)).y(d=>y(d.scorer));
    svg.append('path').datum(pts).attr('d',line).attr('fill','none').attr('stroke',color).attr('stroke-width',1.5);
    svg.selectAll(null).data(pts.slice(-1)).join('circle')
      .attr('cx',d=>x(d.epoch)).attr('cy',d=>y(d.scorer)).attr('r',3).attr('fill',color);
    svg.selectAll(null).data(pts.slice(-1)).join('text')
      .attr('class','dot-label').attr('x',d=>x(d.epoch)+6).attr('y',d=>y(d.scorer)+3)
      .attr('fill',color).text(tag.slice(0,20));
  });
}

connect();
</script>
</body></html>"""


class PageHandler(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.end_headers()
        self.wfile.write(HTML.encode())


async def ws_handler(websocket):
    try:
        while True:
            state = collect_state()
            await websocket.send(json.dumps(state))
            await asyncio.sleep(3)
    except websockets.exceptions.ConnectionClosed:
        pass


async def start_ws():
    async with websockets.serve(ws_handler, "127.0.0.1", 8781):
        await asyncio.Future()


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8780)
    args = p.parse_args()

    http = HTTPServer(("127.0.0.1", args.port), PageHandler)
    Thread(target=http.serve_forever, daemon=True).start()

    url = f"http://localhost:{args.port}"
    print(f"Fleet telemetry at {url}")
    webbrowser.open(url)
    asyncio.run(start_ws())


if __name__ == "__main__":
    main()
