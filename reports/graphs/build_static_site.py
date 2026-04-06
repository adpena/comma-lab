#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parent
SITE = ROOT / 'site'
FILES = [
    ('index.html', 'index.html'),
    ('dashboard_data.json', 'dashboard_data.json'),
    ('score_timeline.json', 'score_timeline.json'),
    ('score_timeline.mmd', 'score_timeline.mmd'),
    ('experiment_graph.json', 'experiment_graph.json'),
    ('promotion_accounting.md', 'promotion_accounting.md'),
    ('promotion_review_latest.md', 'promotion_review_latest.md'),
    ('writeup_outline.md', 'writeup_outline.md'),
    ('final_writeup_draft.md', 'final_writeup_draft.md'),
    ('experiment_journal.md', 'experiment_journal.md'),
    ('ffmpeg_path_review.md', 'ffmpeg_path_review.md'),
    ('legend.md', 'legend.md'),
    ('release_checklist.md', 'release_checklist.md'),
    ('submission_packet.md', 'submission_packet.md'),
    ('final_submission_notes.md', 'final_submission_notes.md'),
    ('judges_one_pager.md', 'judges_one_pager.md'),
    ('evidence_index.md', 'evidence_index.md'),
    ('asset_manifest.json', 'asset_manifest.json'),
    ('frontend_audit.md', 'frontend_audit.md'),
]

SITE.mkdir(parents=True, exist_ok=True)
for src_name, dest_name in FILES:
    src = ROOT / src_name
    if src.exists():
        shutil.copy2(src, SITE / dest_name)

(SITE / '_headers').write_text('''
/*
  X-Content-Type-Options: nosniff
  X-Frame-Options: DENY
  Referrer-Policy: strict-origin-when-cross-origin
  Permissions-Policy: camera=(), microphone=(), geolocation=()
  Cross-Origin-Opener-Policy: same-origin
  Cross-Origin-Resource-Policy: same-origin

/*.json
  Content-Type: application/json; charset=utf-8
  Cache-Control: public, max-age=300

/*.md
  Content-Type: text/markdown; charset=utf-8
  Cache-Control: public, max-age=300

/index.html
  Cache-Control: public, max-age=60
  Content-Security-Policy: default-src 'self'; img-src 'self' data:; style-src 'self' 'unsafe-inline'; script-src 'self'; connect-src 'self'; object-src 'none'; base-uri 'self'; frame-ancestors 'none'
'''.strip() + '\n')

(SITE / '_redirects').write_text('/ /index.html 200\n')
print('site_ready', SITE)
