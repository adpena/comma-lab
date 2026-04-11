#!/usr/bin/env python3
from __future__ import annotations

import argparse
import filecmp
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parent
SITE = ROOT / 'site'
MEDIA = ROOT / 'media'
REPO_ROOT = ROOT.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.comma_lab.state_sync import doctor_repo
FILES = [
    ('index.html', 'index.html'),
    ('report_history.html', 'report_history.html'),
    ('report_history.json', 'report_history.json'),
    ('dashboard_data.json', 'dashboard_data.json'),
    ('score_timeline.json', 'score_timeline.json'),
    ('score_timeline.mmd', 'score_timeline.mmd'),
    ('experiment_graph.json', 'experiment_graph.json'),
    ('promotion_accounting.md', 'promotion_accounting.md'),
    ('promotion_review_latest.md', 'promotion_review_latest.md'),
    ('code_callouts.md', 'code_callouts.md'),
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
    ('experiment_manifest.json', 'experiment_manifest.json'),
    ('frontend_audit.md', 'frontend_audit.md'),
]

HEADERS_TEXT = '''
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

/*.mp4
  Content-Type: video/mp4
  Cache-Control: public, max-age=300

/index.html
  Cache-Control: public, max-age=60
  Content-Security-Policy: default-src 'self'; img-src 'self' data:; style-src 'self' 'unsafe-inline'; script-src 'self'; connect-src 'self'; object-src 'none'; base-uri 'self'; frame-ancestors 'none'
'''.strip() + '\n'

REDIRECTS_TEXT = '/ /index.html 200\n'


def copy_site() -> None:
    SITE.mkdir(parents=True, exist_ok=True)
    for src_name, dest_name in FILES:
        src = ROOT / src_name
        if src.exists():
            shutil.copy2(src, SITE / dest_name)

    if MEDIA.exists():
        media_dst = SITE / 'media'
        if media_dst.exists():
            shutil.rmtree(media_dst)
        shutil.copytree(MEDIA, media_dst)

    (SITE / '_headers').write_text(HEADERS_TEXT)
    (SITE / '_redirects').write_text(REDIRECTS_TEXT)


def check_site_parity() -> int:
    problems: list[str] = []
    for src_name, dest_name in FILES:
        src = ROOT / src_name
        dst = SITE / dest_name
        if not src.exists():
            continue
        if not dst.exists():
            problems.append(f'missing site copy: {dst}')
            continue
        if not filecmp.cmp(src, dst, shallow=False):
            problems.append(f'drift detected: {src} != {dst}')

    if MEDIA.exists():
        media_dst = SITE / 'media'
        if not media_dst.exists():
            problems.append(f'missing media dir: {media_dst}')

    headers = SITE / '_headers'
    redirects = SITE / '_redirects'
    if not headers.exists() or headers.read_text() != HEADERS_TEXT:
        problems.append(f'drift detected: {headers}')
    if not redirects.exists() or redirects.read_text() != REDIRECTS_TEXT:
        problems.append(f'drift detected: {redirects}')

    if problems:
        print('Drift detected between reports/graphs and reports/graphs/site:', file=sys.stderr)
        for item in problems:
            print(f'- {item}', file=sys.stderr)
        return 1
    print('site_in_sync', SITE)
    return 0


def check_promoted_state() -> int:
    report = doctor_repo(REPO_ROOT)
    if not report.findings:
        return 0
    print("Promoted state drift detected:", file=sys.stderr)
    for finding in report.findings:
        print(f"- [{finding.severity}] {finding.code}: {finding.path} -> {finding.message}", file=sys.stderr)
    return 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--check', action='store_true', help='verify that site copies match source artifacts')
    args = parser.parse_args()

    drift_rc = check_promoted_state()
    if drift_rc != 0:
        return drift_rc

    if args.check:
        return check_site_parity()

    copy_site()
    print('site_ready', SITE)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
