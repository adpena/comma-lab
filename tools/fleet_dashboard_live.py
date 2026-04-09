#!/usr/bin/env python3
"""Live fleet dashboard with WebSocket push + D3 charts.

Real-time training metrics via WebSocket. No polling, no refresh.
Score trajectory chart updates live as checkpoints improve.

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
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from threading import Thread

import websockets

REPO = Path(__file__).parent.parent
WEIGHTS_DIR = REPO / "experiments" / "postfilter_weights"
SCORE_HISTORY: list[dict] = []


def collect_state() -> dict:
    """Collect full fleet state."""
    # Training processes
    result = subprocess.run(["ps", "aux"], capture_output=True, text=True)
    procs = []
    for line in result.stdout.splitlines():
        if "python" in line and ("train" in line or "segnet" in line or "proxy" in line) and "grep" not in line and "dashboard" not in line:
            parts = line.split()
            if len(parts) < 11:
                continue
            cmd = " ".join(parts[10:])
            tag = ""
            if "--tag" in cmd:
                try:
                    tag = cmd.split("--tag")[1].strip().split()[0]
                except IndexError:
                    pass
            procs.append({
                "pid": parts[1],
                "cpu": float(parts[2]),
                "mem_gb": round(int(parts[5]) / 1048576, 1) if parts[5].isdigit() else 0,
                "tag": tag,
                "cmd": cmd[:100],
            })

    # Checkpoints
    checkpoints = []
    if WEIGHTS_DIR.exists():
        for mf in sorted(WEIGHTS_DIR.glob("*_best_meta.json"), key=os.path.getmtime, reverse=True):
            try:
                d = json.loads(mf.read_text())
                d["age_s"] = int(time.time() - mf.stat().st_mtime)
                d["tag"] = mf.stem.replace("postfilter_", "").replace("_best_meta", "")
                checkpoints.append(d)
            except Exception:
                pass

    # Log tails
    logs = {}
    for lf in Path("/tmp").glob("*.log"):
        if any(k in lf.name for k in ["standard", "segnet", "h48", "h64", "h32", "h96", "dilated", "psd", "boundary"]):
            try:
                lines = lf.read_text().strip().splitlines()
                logs[lf.name] = lines[-5:] if lines else []
            except Exception:
                pass

    # Best overall
    best = min(checkpoints, key=lambda c: c.get("scorer", 999)) if checkpoints else {}

    # Track history
    if best:
        entry = {"t": time.time(), "scorer": best.get("scorer", 0), "epoch": best.get("epoch", 0), "tag": best.get("tag", "")}
        if not SCORE_HISTORY or SCORE_HISTORY[-1]["scorer"] != entry["scorer"]:
            SCORE_HISTORY.append(entry)

    return {
        "ts": time.strftime("%H:%M:%S"),
        "procs": procs,
        "checkpoints": checkpoints[:10],
        "logs": logs,
        "best": best,
        "history": SCORE_HISTORY[-100:],
    }


HTML = """<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<title>Fleet Dashboard</title>
<script src="https://d3js.org/d3.v7.min.js"></script>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font: 13px/1.5 -apple-system, SF Mono, monospace; background: #0f1724; color: #c9d1d9; padding: 20px; }
h1 { font-size: 16px; color: #8ed1c0; margin-bottom: 4px; }
.status { font-size: 11px; color: #3fb950; margin-bottom: 16px; }
.status.disconnected { color: #f85149; }
.grid { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 12px; margin-bottom: 16px; }
.card { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 12px; }
.card h3 { font-size: 11px; color: #8b949e; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 4px; }
.metric { font-size: 32px; font-weight: 700; color: #8ed1c0; letter-spacing: -0.03em; }
.metric.improving { color: #3fb950; }
.meta { font-size: 11px; color: #8b949e; }
h2 { font-size: 12px; color: #8b949e; text-transform: uppercase; letter-spacing: 0.06em; margin: 16px 0 8px; }
table { width: 100%; border-collapse: collapse; }
th { text-align: left; color: #6b7280; font-size: 10px; text-transform: uppercase; letter-spacing: 0.06em; padding: 4px 8px; border-bottom: 1px solid #30363d; }
td { padding: 4px 8px; border-bottom: 1px solid #21262d; font-size: 12px; }
.tag { color: #f0b35f; }
.score { color: #8ed1c0; font-weight: 700; }
.fresh { color: #3fb950; }
.stale { color: #f85149; }
.chart { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 12px; margin-bottom: 16px; }
.chart svg { width: 100%; height: 200px; }
.chart .axis text { fill: #6b7280; font-size: 10px; }
.chart .axis line, .chart .axis path { stroke: #30363d; }
.chart .line { fill: none; stroke: #8ed1c0; stroke-width: 2; }
.chart .dot { fill: #8ed1c0; }
pre { background: #0d1117; padding: 8px; border-radius: 4px; font-size: 11px; color: #8b949e; overflow-x: auto; margin-bottom: 6px; white-space: pre-wrap; }
.log-name { font-size: 11px; color: #f0b35f; margin: 6px 0 2px; }
</style>
</head><body>
<h1>comma-lab fleet</h1>
<div class="status" id="status">connecting...</div>

<div class="grid">
  <div class="card"><h3>Promoted Floor</h3><div class="metric">1.727</div><div class="meta">h=64 standard int8</div></div>
  <div class="card"><h3>Best Training</h3><div class="metric" id="bestScore">--</div><div class="meta" id="bestMeta">--</div></div>
  <div class="card"><h3>Active Trainers</h3><div class="metric" id="trainerCount">--</div><div class="meta" id="trainerMeta">--</div></div>
</div>

<div class="chart">
  <h2>Score trajectory (live)</h2>
  <svg id="chart"></svg>
</div>

<h2>Processes</h2>
<table id="procsTable"><tr><th>PID</th><th>CPU</th><th>RAM</th><th>Tag</th><th>Command</th></tr></table>

<h2>Checkpoints</h2>
<table id="ckptTable"><tr><th>Tag</th><th>Epoch</th><th>Scorer</th><th>Size</th><th>Age</th></tr></table>

<h2>Logs</h2>
<div id="logs"></div>

<script>
let ws;
let history = [];

function connect() {
  ws = new WebSocket('ws://localhost:8781');
  ws.onopen = () => { document.getElementById('status').textContent = 'live'; document.getElementById('status').className = 'status'; };
  ws.onclose = () => { document.getElementById('status').textContent = 'disconnected — reconnecting...'; document.getElementById('status').className = 'status disconnected'; setTimeout(connect, 2000); };
  ws.onmessage = (e) => update(JSON.parse(e.data));
}

function update(d) {
  // Best score
  if (d.best && d.best.scorer) {
    document.getElementById('bestScore').textContent = d.best.scorer.toFixed(4);
    document.getElementById('bestMeta').textContent = d.best.tag + ' ep ' + d.best.epoch;
  }

  // Trainer count
  document.getElementById('trainerCount').textContent = d.procs.length;
  const totalCpu = d.procs.reduce((s,p) => s + p.cpu, 0);
  const totalMem = d.procs.reduce((s,p) => s + p.mem_gb, 0);
  document.getElementById('trainerMeta').textContent = totalCpu.toFixed(0) + '% CPU, ' + totalMem.toFixed(1) + 'GB RAM';

  // Processes table
  const pt = document.getElementById('procsTable');
  pt.innerHTML = '<tr><th>PID</th><th>CPU</th><th>RAM</th><th>Tag</th><th>Command</th></tr>';
  d.procs.forEach(p => {
    pt.innerHTML += `<tr><td>${p.pid}</td><td>${p.cpu}%</td><td>${p.mem_gb}GB</td><td class="tag">${p.tag}</td><td>${p.cmd}</td></tr>`;
  });

  // Checkpoints table
  const ct = document.getElementById('ckptTable');
  ct.innerHTML = '<tr><th>Tag</th><th>Epoch</th><th>Scorer</th><th>Size</th><th>Age</th></tr>';
  d.checkpoints.forEach(c => {
    const age = c.age_s < 60 ? c.age_s+'s' : c.age_s < 3600 ? Math.floor(c.age_s/60)+'m' : Math.floor(c.age_s/3600)+'h';
    const cls = c.age_s < 120 ? 'fresh' : c.age_s > 600 ? 'stale' : '';
    ct.innerHTML += `<tr><td class="tag">${c.tag}</td><td>${c.epoch}</td><td class="score">${c.scorer.toFixed(4)}</td><td>${c.int8_size||'?'}B</td><td class="${cls}">${age}</td></tr>`;
  });

  // Logs
  const lg = document.getElementById('logs');
  lg.innerHTML = '';
  Object.entries(d.logs).sort().forEach(([name, lines]) => {
    lg.innerHTML += `<div class="log-name">${name}</div><pre>${lines.join('\\n')}</pre>`;
  });

  // Chart
  history = d.history || [];
  drawChart();
}

function drawChart() {
  if (!history.length) return;
  const svg = d3.select('#chart');
  svg.selectAll('*').remove();
  const w = svg.node().parentElement.clientWidth - 24;
  const h = 180;
  const m = {top:10, right:10, bottom:25, left:50};

  const x = d3.scaleLinear().domain(d3.extent(history, d=>d.epoch)).range([m.left, w-m.right]);
  const y = d3.scaleLinear().domain([d3.min(history,d=>d.scorer)-0.1, d3.max(history,d=>d.scorer)+0.1]).range([h-m.bottom, m.top]);

  svg.attr('viewBox', `0 0 ${w} ${h}`);
  svg.append('g').attr('class','axis').attr('transform',`translate(0,${h-m.bottom})`).call(d3.axisBottom(x).ticks(6));
  svg.append('g').attr('class','axis').attr('transform',`translate(${m.left},0)`).call(d3.axisLeft(y).ticks(5));

  svg.append('path').datum(history).attr('class','line')
    .attr('d', d3.line().x(d=>x(d.epoch)).y(d=>y(d.scorer)));
  svg.selectAll('.dot').data(history).join('circle').attr('class','dot')
    .attr('cx',d=>x(d.epoch)).attr('cy',d=>y(d.scorer)).attr('r',3);
}

connect();
</script>
</body></html>"""


class PageHandler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.end_headers()
        self.wfile.write(HTML.encode())


async def ws_handler(websocket):
    """Push state updates every 3 seconds."""
    try:
        while True:
            state = collect_state()
            await websocket.send(json.dumps(state))
            await asyncio.sleep(3)
    except websockets.exceptions.ConnectionClosed:
        pass


async def start_ws_server():
    async with websockets.serve(ws_handler, "127.0.0.1", 8781):
        await asyncio.Future()  # run forever


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8780)
    args = parser.parse_args()

    # HTTP server in thread
    http = HTTPServer(("127.0.0.1", args.port), PageHandler)
    Thread(target=http.serve_forever, daemon=True).start()

    url = f"http://localhost:{args.port}"
    print(f"Fleet dashboard at {url} (WebSocket on :8781)")
    webbrowser.open(url)

    # WebSocket server in main thread
    asyncio.run(start_ws_server())


if __name__ == "__main__":
    main()
