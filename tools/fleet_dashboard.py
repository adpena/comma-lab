#!/usr/bin/env python3
"""Live fleet dashboard — monitors all training processes and checkpoints.

Opens a browser with auto-refreshing stats. No dependencies beyond stdlib.

Usage:
    python tools/fleet_dashboard.py [--port 8780]
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path

REPO = Path(__file__).parent.parent
WEIGHTS_DIR = REPO / "experiments" / "postfilter_weights"


def get_training_processes() -> list[dict]:
    """Get all active Python training processes."""
    result = subprocess.run(
        ["ps", "aux"], capture_output=True, text=True
    )
    procs = []
    for line in result.stdout.splitlines():
        if "python" in line and "train" in line and "grep" not in line and "http.server" not in line:
            parts = line.split()
            pid = parts[1]
            cpu = parts[2]
            mem_kb = int(parts[5]) if parts[5].isdigit() else 0
            cmd = " ".join(parts[10:])
            # Extract tag from command
            tag = ""
            if "--tag" in cmd:
                idx = cmd.index("--tag") + 6
                tag = cmd[idx:].split()[0] if idx < len(cmd) else ""
            elif "segnet_boundary" in cmd:
                tag = "segnet_boundary"
            procs.append({
                "pid": pid,
                "cpu": cpu,
                "mem_gb": round(mem_kb / 1048576, 1),
                "tag": tag,
                "cmd": cmd[:120],
            })
    return procs


def get_checkpoints() -> list[dict]:
    """Get all best checkpoint metadata files."""
    checkpoints = []
    if not WEIGHTS_DIR.exists():
        return checkpoints
    for meta_file in sorted(WEIGHTS_DIR.glob("*_best_meta.json"), key=os.path.getmtime, reverse=True):
        try:
            data = json.loads(meta_file.read_text())
            age = int(time.time() - meta_file.stat().st_mtime)
            data["age_s"] = age
            data["tag"] = meta_file.stem.replace("postfilter_", "").replace("_best_meta", "")
            checkpoints.append(data)
        except (json.JSONDecodeError, KeyError):
            continue
    return checkpoints


def get_log_tails() -> dict[str, str]:
    """Get last few lines of training logs."""
    logs = {}
    for log_file in Path("/tmp").glob("*.log"):
        if any(k in log_file.name for k in ["standard", "segnet", "h48", "h64", "h32", "h96", "dilated", "psd"]):
            try:
                lines = log_file.read_text().strip().splitlines()
                logs[log_file.name] = "\n".join(lines[-5:])
            except Exception:
                pass
    return logs


PAGE = """<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<title>Fleet Dashboard</title>
<meta http-equiv="refresh" content="10">
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font: 14px/1.6 -apple-system, SF Mono, monospace; background: #0f1724; color: #c9d1d9; padding: 24px; }
h1 { font-size: 18px; color: #8ed1c0; margin-bottom: 16px; }
h2 { font-size: 14px; color: #a8b1bc; margin: 20px 0 8px; text-transform: uppercase; letter-spacing: 0.08em; }
table { width: 100%%; border-collapse: collapse; margin-bottom: 16px; }
th { text-align: left; color: #8b949e; font-size: 11px; text-transform: uppercase; letter-spacing: 0.06em; padding: 6px 10px; border-bottom: 1px solid #30363d; }
td { padding: 6px 10px; border-bottom: 1px solid #21262d; }
.score { color: #8ed1c0; font-weight: 700; font-size: 16px; }
.stale { color: #f85149; }
.fresh { color: #3fb950; }
.tag { color: #f0b35f; }
pre { background: #161b22; padding: 10px; border-radius: 6px; font-size: 12px; overflow-x: auto; margin-bottom: 8px; color: #8b949e; }
.grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
.card { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 14px; }
.card h3 { font-size: 13px; color: #c9d1d9; margin-bottom: 8px; }
.metric { font-size: 28px; font-weight: 700; color: #8ed1c0; }
.meta { font-size: 11px; color: #8b949e; }
</style>
</head><body>
<h1>comma-lab fleet dashboard</h1>
<p class="meta">Auto-refreshes every 10s &middot; %(timestamp)s</p>

<div class="grid">
<div class="card">
<h3>Promoted Floor</h3>
<div class="metric">1.727</div>
<div class="meta">h=64 standard, 45.6KB int8</div>
</div>
<div class="card">
<h3>Best Training Scorer</h3>
<div class="metric">%(best_scorer)s</div>
<div class="meta">%(best_tag)s &middot; ep %(best_epoch)s</div>
</div>
</div>

<h2>Active Training Processes</h2>
<table>
<tr><th>PID</th><th>CPU</th><th>RAM</th><th>Tag</th><th>Command</th></tr>
%(procs_rows)s
</table>

<h2>Best Checkpoints (by recency)</h2>
<table>
<tr><th>Tag</th><th>Epoch</th><th>Scorer</th><th>Int8 Size</th><th>Age</th></tr>
%(ckpt_rows)s
</table>

<h2>Training Logs</h2>
%(log_blocks)s

</body></html>"""


class DashHandler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        procs = get_training_processes()
        ckpts = get_checkpoints()
        logs = get_log_tails()

        procs_rows = ""
        for p in procs:
            procs_rows += f"<tr><td>{p['pid']}</td><td>{p['cpu']}%%</td><td>{p['mem_gb']}GB</td><td class='tag'>{p['tag']}</td><td>{p['cmd']}</td></tr>\n"
        if not procs:
            procs_rows = "<tr><td colspan='5' style='color:#f85149'>No training processes running</td></tr>"

        ckpt_rows = ""
        best_scorer = "--"
        best_tag = "--"
        best_epoch = "--"
        for c in ckpts:
            age = c.get("age_s", 0)
            age_str = f"{age}s" if age < 60 else f"{age//60}m" if age < 3600 else f"{age//3600}h"
            freshness = "fresh" if age < 120 else "stale" if age > 600 else ""
            scorer = c.get("scorer", 0)
            ckpt_rows += f"<tr><td class='tag'>{c.get('tag','?')}</td><td>{c.get('epoch','?')}</td><td class='score'>{scorer:.4f}</td><td>{c.get('int8_size', '?')}B</td><td class='{freshness}'>{age_str} ago</td></tr>\n"
            if best_scorer == "--" or (isinstance(scorer, (int, float)) and scorer < float(best_scorer)):
                best_scorer = f"{scorer:.4f}"
                best_tag = c.get("tag", "?")
                best_epoch = str(c.get("epoch", "?"))

        log_blocks = ""
        for name, content in sorted(logs.items()):
            log_blocks += f"<h3 style='font-size:12px;color:#f0b35f;margin:8px 0 4px'>{name}</h3><pre>{content}</pre>\n"
        if not logs:
            log_blocks = "<p class='meta'>No training logs in /tmp/</p>"

        html = PAGE % {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "procs_rows": procs_rows,
            "ckpt_rows": ckpt_rows,
            "log_blocks": log_blocks,
            "best_scorer": best_scorer,
            "best_tag": best_tag,
            "best_epoch": best_epoch,
        }
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.end_headers()
        self.wfile.write(html.encode())


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8780)
    args = parser.parse_args()

    server = HTTPServer(("127.0.0.1", args.port), DashHandler)
    url = f"http://localhost:{args.port}"
    print(f"Fleet dashboard at {url}")
    import webbrowser
    webbrowser.open(url)
    server.serve_forever()


if __name__ == "__main__":
    main()
