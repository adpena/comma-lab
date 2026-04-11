#!/usr/bin/env python3
"""Markdown Timeline Viewer — git-powered diff scrubber.

A local web tool that lets you:
  - Browse all markdown files tracked in git
  - Scrub a slider across commits to see how a file evolved
  - View word-level diffs between any two versions
  - Search commits by date range or content
  - Jump to any commit with one click

Usage:
    python tools/md_timeline.py [--port 8765] [--repo .]

Opens a browser to http://localhost:8765
"""
from __future__ import annotations

import argparse
import html
import json
import os
import re
import subprocess
import sys
import webbrowser
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlparse


def git(args: list[str], cwd: str) -> str:
    """Run a git command and return stdout."""
    result = subprocess.run(
        ["git"] + args,
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=10,
    )
    return result.stdout.strip()


def get_commits(cwd: str) -> list[dict]:
    """Get all commits with metadata."""
    raw = git(
        ["log", "--all", "--format=%H|%ai|%an|%s", "--date=iso"],
        cwd,
    )
    commits = []
    for line in raw.splitlines():
        if not line.strip():
            continue
        parts = line.split("|", 3)
        if len(parts) < 4:
            continue
        commits.append({
            "hash": parts[0],
            "date": parts[1],
            "author": parts[2],
            "message": parts[3],
        })
    return commits


def get_md_files_at_commit(cwd: str, commit_hash: str) -> list[str]:
    """List all .md files present at a given commit."""
    raw = git(["ls-tree", "-r", "--name-only", commit_hash], cwd)
    return [f for f in raw.splitlines() if f.endswith(".md")]


def get_file_at_commit(cwd: str, commit_hash: str, filepath: str) -> str | None:
    """Get file contents at a specific commit. Returns None if not found."""
    result = subprocess.run(
        ["git", "show", f"{commit_hash}:{filepath}"],
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=10,
    )
    if result.returncode != 0:
        return None
    return result.stdout


def get_word_diff(cwd: str, hash_a: str, hash_b: str, filepath: str) -> str:
    """Get word-level diff between two commits for a file."""
    result = subprocess.run(
        ["git", "diff", "--word-diff=porcelain", f"{hash_a}..{hash_b}", "--", filepath],
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=10,
    )
    return result.stdout


def word_diff_to_html(raw_diff: str) -> str:
    """Convert git word-diff porcelain output to styled HTML."""
    if not raw_diff.strip():
        return "<p class='no-change'>No changes in this file between these commits.</p>"

    lines = raw_diff.split("\n")
    html_parts = []
    in_hunk = False

    for line in lines:
        if line.startswith("diff ") or line.startswith("index ") or line.startswith("---") or line.startswith("+++"):
            continue
        if line.startswith("@@"):
            if html_parts:
                html_parts.append("<hr class='hunk-sep'>")
            in_hunk = True
            # Extract line numbers
            match = re.search(r"@@ -(\d+)", line)
            if match:
                html_parts.append(f"<span class='hunk-header'>Line {match.group(1)}</span>")
            continue
        if not in_hunk:
            continue

        if line.startswith("+"):
            html_parts.append(f"<ins>{html.escape(line[1:])}</ins>")
        elif line.startswith("-"):
            html_parts.append(f"<del>{html.escape(line[1:])}</del>")
        elif line.startswith("~"):
            html_parts.append("<br>")
        elif line.startswith(" "):
            html_parts.append(html.escape(line[1:]))
        else:
            html_parts.append(html.escape(line))

    return "".join(html_parts) if html_parts else "<p class='no-change'>No changes.</p>"


def get_file_history(cwd: str, filepath: str) -> list[dict]:
    """Get commits that touched a specific file."""
    raw = git(
        ["log", "--all", "--follow", "--format=%H|%ai|%an|%s", "--", filepath],
        cwd,
    )
    commits = []
    for line in raw.splitlines():
        if not line.strip():
            continue
        parts = line.split("|", 3)
        if len(parts) < 4:
            continue
        commits.append({
            "hash": parts[0],
            "date": parts[1],
            "author": parts[2],
            "message": parts[3],
        })
    return commits


FRONTEND_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Markdown Timeline</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
    font-family: -apple-system, BlinkMacSystemFont, 'SF Mono', 'Fira Code', monospace;
    background: #0d1117; color: #c9d1d9;
    display: flex; flex-direction: column; height: 100vh;
}
header {
    background: #161b22; border-bottom: 1px solid #30363d;
    padding: 12px 20px; display: flex; align-items: center; gap: 16px;
    flex-wrap: wrap;
}
header h1 { font-size: 16px; color: #58a6ff; white-space: nowrap; }
select, input[type="text"] {
    background: #0d1117; color: #c9d1d9; border: 1px solid #30363d;
    padding: 6px 10px; border-radius: 6px; font-size: 13px;
}
select:focus, input:focus { border-color: #58a6ff; outline: none; }
.slider-wrap {
    flex: 1; min-width: 200px; display: flex; align-items: center; gap: 8px;
}
.slider-wrap input[type="range"] {
    flex: 1; accent-color: #58a6ff;
}
.slider-wrap .commit-label {
    font-size: 11px; color: #8b949e; min-width: 80px; text-align: right;
}
.controls { display: flex; gap: 8px; align-items: center; }
button {
    background: #21262d; color: #c9d1d9; border: 1px solid #30363d;
    padding: 6px 12px; border-radius: 6px; cursor: pointer; font-size: 12px;
}
button:hover { background: #30363d; }
button.active { background: #1f6feb; border-color: #1f6feb; color: #fff; }
.main { display: flex; flex: 1; overflow: hidden; }
.sidebar {
    width: 280px; background: #161b22; border-right: 1px solid #30363d;
    overflow-y: auto; padding: 8px;
}
.sidebar .file-item {
    padding: 6px 10px; border-radius: 4px; cursor: pointer;
    font-size: 12px; color: #8b949e; white-space: nowrap;
    overflow: hidden; text-overflow: ellipsis;
}
.sidebar .file-item:hover { background: #21262d; }
.sidebar .file-item.selected { background: #1f6feb33; color: #58a6ff; }
.sidebar .commit-item {
    padding: 8px 10px; border-radius: 4px; cursor: pointer;
    font-size: 12px; border-left: 3px solid transparent; margin: 2px 0;
}
.sidebar .commit-item:hover { background: #21262d; }
.sidebar .commit-item.selected { border-left-color: #58a6ff; background: #1f6feb22; }
.sidebar .commit-item .hash { color: #58a6ff; font-family: monospace; }
.sidebar .commit-item .date { color: #8b949e; font-size: 11px; }
.sidebar .commit-item .msg { color: #c9d1d9; margin-top: 2px; }
.sidebar h3 {
    color: #8b949e; font-size: 11px; text-transform: uppercase;
    letter-spacing: 1px; padding: 8px 10px 4px;
}
.content { flex: 1; overflow-y: auto; padding: 20px 32px; }
.content pre {
    white-space: pre-wrap; word-wrap: break-word; font-size: 13px;
    line-height: 1.6; color: #c9d1d9;
}
.content ins {
    background: #2ea04333; color: #3fb950; text-decoration: none;
    padding: 1px 2px; border-radius: 2px;
}
.content del {
    background: #f8514933; color: #f85149; text-decoration: line-through;
    padding: 1px 2px; border-radius: 2px;
}
.content .hunk-sep { border-color: #30363d; margin: 12px 0; }
.content .hunk-header {
    display: inline-block; background: #1f6feb33; color: #58a6ff;
    padding: 2px 8px; border-radius: 4px; font-size: 11px; margin: 4px 0;
}
.content .no-change { color: #8b949e; font-style: italic; }
.info-bar {
    background: #161b22; border-top: 1px solid #30363d;
    padding: 6px 20px; font-size: 11px; color: #8b949e;
    display: flex; justify-content: space-between;
}
.mode-tabs { display: flex; gap: 4px; }
.search-box { min-width: 160px; }
#loading {
    position: fixed; top: 50%; left: 50%; transform: translate(-50%, -50%);
    background: #21262d; padding: 20px 32px; border-radius: 8px;
    border: 1px solid #30363d; display: none; z-index: 100;
}
</style>
</head>
<body>

<header>
    <h1>md timeline</h1>
    <select id="filePicker"><option value="">-- select file --</option></select>
    <div class="slider-wrap">
        <input type="range" id="slider" min="0" max="0" value="0" disabled>
        <span class="commit-label" id="sliderLabel">--</span>
    </div>
    <div class="controls">
        <div class="mode-tabs">
            <button id="btnContent" class="active" onclick="setMode('content')">Content</button>
            <button id="btnDiff" onclick="setMode('diff')">Diff</button>
        </div>
        <input type="text" class="search-box" id="searchBox" placeholder="Search content...">
    </div>
</header>

<div class="main">
    <div class="sidebar" id="sidebar"></div>
    <div class="content"><pre id="viewer">Select a markdown file to begin.</pre></div>
</div>

<div class="info-bar">
    <span id="infoLeft">No file selected</span>
    <span id="infoRight"></span>
</div>

<div id="loading">Loading...</div>

<script>
let commits = [];
let fileCommits = [];
let allFiles = [];
let currentFile = '';
let currentMode = 'content';
let currentIdx = 0;

async function api(endpoint, params = {}) {
    const url = new URL(endpoint, location.href);
    Object.entries(params).forEach(([k,v]) => url.searchParams.set(k, v));
    const res = await fetch(url);
    return res.json();
}

async function init() {
    const data = await api('/api/commits');
    commits = data.commits;
    allFiles = data.files;

    const picker = document.getElementById('filePicker');
    allFiles.forEach(f => {
        const opt = document.createElement('option');
        opt.value = f; opt.textContent = f;
        picker.appendChild(opt);
    });
    picker.onchange = () => selectFile(picker.value);

    document.getElementById('searchBox').oninput = debounce(doSearch, 300);
}

async function selectFile(filepath) {
    if (!filepath) return;
    currentFile = filepath;
    const data = await api('/api/file_history', { file: filepath });
    fileCommits = data.commits;

    const slider = document.getElementById('slider');
    slider.max = Math.max(0, fileCommits.length - 1);
    slider.value = 0;
    slider.disabled = fileCommits.length === 0;
    slider.oninput = () => {
        currentIdx = parseInt(slider.value);
        renderCurrent();
    };

    renderSidebar();
    currentIdx = 0;
    renderCurrent();
}

function renderSidebar() {
    const sb = document.getElementById('sidebar');
    sb.innerHTML = '<h3>Commits (' + fileCommits.length + ')</h3>';
    fileCommits.forEach((c, i) => {
        const div = document.createElement('div');
        div.className = 'commit-item' + (i === currentIdx ? ' selected' : '');
        div.innerHTML = `
            <span class="hash">${c.hash.slice(0,7)}</span>
            <span class="date">${c.date.slice(0,16)}</span>
            <div class="msg">${escHtml(c.message)}</div>
        `;
        div.onclick = () => {
            currentIdx = i;
            document.getElementById('slider').value = i;
            renderCurrent();
        };
        sb.appendChild(div);
    });
}

async function renderCurrent() {
    if (!fileCommits.length) return;
    const c = fileCommits[currentIdx];

    // Update sidebar selection
    document.querySelectorAll('.commit-item').forEach((el, i) => {
        el.className = 'commit-item' + (i === currentIdx ? ' selected' : '');
    });

    document.getElementById('sliderLabel').textContent = c.hash.slice(0,7);
    document.getElementById('infoLeft').textContent =
        `${currentFile} @ ${c.hash.slice(0,7)} (${c.date.slice(0,16)})`;
    document.getElementById('infoRight').textContent =
        `Commit ${currentIdx + 1} of ${fileCommits.length}`;

    if (currentMode === 'content') {
        const data = await api('/api/file_content', {
            file: currentFile, hash: c.hash
        });
        const viewer = document.getElementById('viewer');
        const search = document.getElementById('searchBox').value;
        let text = escHtml(data.content || '(file not found at this commit)');
        if (search) {
            const re = new RegExp('(' + escRegex(search) + ')', 'gi');
            text = text.replace(re, '<mark style="background:#e3b341;color:#0d1117">$1</mark>');
        }
        viewer.innerHTML = text;
    } else {
        // Diff mode: diff against previous commit
        const prevIdx = Math.min(currentIdx + 1, fileCommits.length - 1);
        if (prevIdx === currentIdx) {
            document.getElementById('viewer').innerHTML =
                '<span class="no-change">This is the earliest commit for this file.</span>';
            return;
        }
        const prev = fileCommits[prevIdx];
        const data = await api('/api/diff', {
            file: currentFile, hash_a: prev.hash, hash_b: c.hash
        });
        document.getElementById('viewer').innerHTML = data.html;
    }
}

function setMode(mode) {
    currentMode = mode;
    document.getElementById('btnContent').className = mode === 'content' ? 'active' : '';
    document.getElementById('btnDiff').className = mode === 'diff' ? 'active' : '';
    renderCurrent();
}

async function doSearch() {
    if (currentMode === 'content') renderCurrent();
}

function escHtml(s) {
    return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}
function escRegex(s) {
    return s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}
function debounce(fn, ms) {
    let t; return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); };
}

init();
</script>
</body>
</html>"""


class TimelineHandler(BaseHTTPRequestHandler):
    repo_dir: str = "."

    def log_message(self, fmt, *args):
        pass  # Suppress default logging

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        params = parse_qs(parsed.query)

        if path == "/" or path == "/index.html":
            self._respond_html(FRONTEND_HTML)
        elif path == "/api/commits":
            commits = get_commits(self.repo_dir)
            # Get all md files from HEAD
            head = commits[0]["hash"] if commits else "HEAD"
            files = get_md_files_at_commit(self.repo_dir, head)
            self._respond_json({"commits": commits, "files": sorted(files)})
        elif path == "/api/file_history":
            filepath = self._sanitize_path(params.get("file", [""])[0])
            history = get_file_history(self.repo_dir, filepath)
            self._respond_json({"commits": history})
        elif path == "/api/file_content":
            filepath = self._sanitize_path(params.get("file", [""])[0])
            commit_hash = self._sanitize_hash(params.get("hash", ["HEAD"])[0])
            content = get_file_at_commit(self.repo_dir, commit_hash, filepath)
            self._respond_json({"content": content or ""})
        elif path == "/api/diff":
            filepath = self._sanitize_path(params.get("file", [""])[0])
            hash_a = self._sanitize_hash(params.get("hash_a", [""])[0])
            hash_b = self._sanitize_hash(params.get("hash_b", [""])[0])
            raw = get_word_diff(self.repo_dir, hash_a, hash_b, filepath)
            diff_html = word_diff_to_html(raw)
            self._respond_json({"html": diff_html})
        else:
            self.send_error(404)

    @staticmethod
    def _sanitize_hash(h: str) -> str:
        """Reject git args that don't look like commit hashes or refs."""
        import re as _re
        if not h or not _re.match(r'^[a-zA-Z0-9_.~^/@{}\-]+$', h):
            return "HEAD"
        return h

    @staticmethod
    def _sanitize_path(p: str) -> str:
        """Reject paths with suspicious characters."""
        import re as _re
        if not p or not _re.match(r'^[a-zA-Z0-9_./ \-]+$', p):
            return ""
        return p

    def _respond_json(self, data):
        body = json.dumps(data).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _respond_html(self, html_str):
        body = html_str.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main():
    parser = argparse.ArgumentParser(description="Markdown Timeline Viewer")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--repo", default=".", help="Path to git repo")
    parser.add_argument("--no-open", action="store_true", help="Don't open browser")
    args = parser.parse_args()

    repo = os.path.abspath(args.repo)
    TimelineHandler.repo_dir = repo

    server = HTTPServer(("127.0.0.1", args.port), TimelineHandler)
    url = f"http://localhost:{args.port}"
    print(f"Markdown Timeline Viewer running at {url}")
    print(f"Repo: {repo}")
    print("Press Ctrl+C to stop")

    if not args.no_open:
        webbrowser.open(url)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
        server.server_close()


if __name__ == "__main__":
    main()
