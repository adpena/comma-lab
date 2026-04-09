#!/usr/bin/env python3
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / 'reports' / 'results.jsonl'
TIMELINE = ROOT / 'reports' / 'timeline.jsonl'
OUT_JSON = ROOT / 'reports' / 'graphs' / 'dashboard_data.json'
OUT_HTML = ROOT / 'reports' / 'graphs' / 'index.html'
OUT_TIMELINE = ROOT / 'reports' / 'graphs' / 'score_timeline.json'
OUT_GRAPH = ROOT / 'reports' / 'graphs' / 'experiment_graph.json'
MEDIA_MANIFEST = ROOT / 'reports' / 'graphs' / 'media' / 'comparison_manifest.json'
CHALLENGE_URL = 'https://github.com/commaai/comma_video_compression_challenge'
GITHUB_URL = 'https://github.com/adpena/comma-lab'
LOCAL_TZ = ZoneInfo('America/Chicago')


def load_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_json(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def dedupe_by(items: list[dict], key_fn):
    seen_order = []
    latest_by_key = {}
    for item in items:
        key = key_fn(item)
        if key not in latest_by_key:
            seen_order.append(key)
        latest_by_key[key] = item
    return [latest_by_key[key] for key in seen_order]


def parse_utc_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace('Z', '+00:00'))
    except ValueError:
        return None


def format_local_datetime(value: datetime | str | None) -> str:
    if value is None:
        return ''
    dt = parse_utc_timestamp(value) if isinstance(value, str) else value
    if dt is None:
        return ''
    local = dt.astimezone(LOCAL_TZ)
    hour = local.strftime('%I').lstrip('0') or '12'
    return f"{local.strftime('%b')} {local.day}, {local.year}, {hour}:{local.strftime('%M %p %Z')}"


def derive_promoted_run_ids(robust_runs: list[dict], timeline: list[dict]) -> list[str]:
    promoted_ids: list[str] = []

    def score_match(run: dict, summary: str) -> int:
        cfg = run.get('config', {})
        s = summary.lower()
        total = 0
        scale = f"{cfg.get('scale_w')}x{cfg.get('scale_h')}"
        if cfg.get('scale_w') and scale in s:
            total += 2
        gop = cfg.get('gop')
        if gop is not None and (f'keyint {gop}' in s or f'keyint{gop}' in s):
            total += 2
        preset = cfg.get('preset')
        if preset and preset in s:
            total += 1
        filters = str(cfg.get('filters', ''))
        if filters and (filters in s or filters.replace('/', '+') in s):
            total += 2
        return total

    for event in timeline:
        if event.get('type') != 'promotion':
            continue
        score = event.get('current_workflow_score')
        if score is None:
            continue
        summary = event.get('summary', '')
        candidates = [
            r for r in robust_runs
            if r['run_id'] not in promoted_ids and abs(r['current_workflow_score'] - score) < 0.011
        ]
        if not candidates:
            continue
        best_score = max(score_match(r, summary) for r in candidates)
        filtered = [r for r in candidates if score_match(r, summary) == best_score]
        promoted_ids.append(filtered[0]['run_id'])
    return promoted_ids


def fmt_int(n: int | None) -> str:
    return '' if n is None else f'{n:,}'


def score_parts(run: dict) -> dict[str, float]:
    seg = float(run.get('segnet_distortion', 0.0))
    pose = float(run.get('posenet_distortion', 0.0))
    rate = float(run.get('rate', 0.0))
    return {
        'seg_component': 100.0 * seg,
        'pose_component': math.sqrt(10.0 * pose),
        'rate_component': 25.0 * rate,
    }


def describe_config(cfg: dict) -> str:
    if cfg.get('video_codec') == 'libsvtav1':
        return (
            f"{cfg.get('scale_w')}x{cfg.get('scale_h')}, "
            f"{cfg.get('preset')}, crf {cfg.get('crf')}, "
            f"keyint {cfg.get('gop')}, film-grain {cfg.get('film_grain')}, "
            f"{cfg.get('filters')}"
        )
    return (
        f"{cfg.get('scale_w')}x{cfg.get('scale_h')}, "
        f"{cfg.get('filters')}, crf {cfg.get('crf')}, "
        f"keyint {cfg.get('gop')}, bframes {cfg.get('bframes')}, ref {cfg.get('ref')}"
    )


def short_label(run: dict) -> str:
    cfg = run.get('config', {})
    if cfg.get('video_codec') == 'libsvtav1':
        return f"AV1 {cfg.get('scale_w')}x{cfg.get('scale_h')} / crf {cfg.get('crf')}"
    return f"x265 {cfg.get('scale_w')}x{cfg.get('scale_h')} / crf {cfg.get('crf')}"


def promoted_run_note(run: dict) -> str:
    run_id = run.get('run_id', '')
    cfg = run.get('config', {})
    filters = str(cfg.get('filters', ''))
    inflate = str(cfg.get('inflate_postfilter', ''))

    if 'ensemble-h32-mc75-25' in run_id or 'ensemble-postfilter-h32-mc75-25' in filters:
        return 'weighted ensemble h32 + MC 75/25'
    if 'long1000-h64' in run_id or 'postfilter-h64' in filters or ' h64' in inflate:
        return 'long1000 QAT+EMA learned int8 post-filter h64'
    if 'long1000-h32' in run_id or 'postfilter-h32' in filters or ' h32' in inflate:
        return 'long1000 QAT+EMA learned int8 post-filter h32'
    if 'long1000-h16' in run_id or 'postfilter-h16' in filters or ' h16' in inflate:
        return 'long1000 QAT+EMA learned int8 post-filter h16'
    if 'learned-postfilter' in filters:
        return 'tiny learned int8 post-filter'
    if inflate:
        return inflate
    return short_label(run)


def clean_public_notes(notes: list[str]) -> list[str]:
    blocked = {
        'ties public leaderboard lead',
        'beats public leaderboard lead',
    }
    return [note for note in notes if note not in blocked]


def main() -> int:
    results = dedupe_by(load_jsonl(RESULTS), lambda row: row['run_id'])
    timeline = dedupe_by(
        load_jsonl(TIMELINE),
        lambda row: (row.get('event_id'), row.get('ts_utc'), row.get('type')),
    )
    media_manifest = load_json(MEDIA_MANIFEST)
    build_time_utc = datetime.now(timezone.utc)
    results.sort(key=lambda r: r['ts_utc'])
    timeline.sort(key=lambda r: r['ts_utc'])

    exact_runs = [r for r in results if r['track'] == 'exact_current']
    robust_runs = [r for r in results if r['track'] == 'robust_current']
    run_index = {r['run_id']: r for r in robust_runs}
    promoted_ids = derive_promoted_run_ids(robust_runs, timeline)
    promoted = [r for r in robust_runs if r['run_id'] in promoted_ids]
    promoted_id_set = set(promoted_ids)
    rejected = [r for r in robust_runs if any('reject' in n.lower() for n in r.get('notes', []))]

    best = min(robust_runs, key=lambda r: r['current_workflow_score'])
    latest = robust_runs[-1]

    promotion_ladder = [
        {
            'run_id': r['run_id'],
            'score': r['current_workflow_score'],
            'archive_bytes': r['archive_bytes'],
            'config': r.get('config', {}),
            'ts_utc': r['ts_utc'],
        }
        for r in promoted
    ]

    promotion_accounting = [
        {
            'run_id': r['run_id'],
            'current_workflow_score': r['current_workflow_score'],
            'current_workflow_bytes': r['archive_bytes'],
            'rule_faithful_score': r.get('rule_faithful_score'),
            'rule_faithful_bytes': r.get('rule_faithful_bundle_bytes'),
            'config': r.get('config', {}),
        }
        for r in promoted
    ]

    negative_priority_ids = [
        'robust_current-av1-524x394-cpu-2026-04-05',
        'robust_current-av1-524x394-filmgrain0-rejected-cpu-2026-04-06',
        'robust_current-dynamic-main-roi-cpu-2026-04-05',
        'robust_current-roi-two-pass-cpu-2026-04-04',
        'robust_current-av1-524x394-crf35-rejected-cpu-2026-04-06',
        'robust_current-av1-522x392-rejected-cpu-2026-04-06',
        'robust_current-av1-524x394-unsharp030-rejected-cpu-2026-04-06',
        'robust_current-cand-428x320-crf23-g48-b4-r4-cpu-2026-04-05',
    ]
    negative_results = []
    for rid in negative_priority_ids:
        r = run_index.get(rid)
        if not r:
            continue
        negative_results.append({
            'run_id': r['run_id'],
            'score': r['current_workflow_score'],
            'archive_bytes': r['archive_bytes'],
            'notes': r.get('notes', []),
            'config': r.get('config', {}),
        })

    notable_events = [
        e for e in timeline
        if e.get('type') in {'promotion', 'research', 'verification', 'analysis', 'infra', 'decision'}
    ]

    key_turning_points = [
        {'label': 'Initial honest floor', 'run_id': 'robust_current-baseline-cpu-2026-04-03', 'score': 4.06},
        {'label': '512x384 / crf23', 'run_id': 'robust_current-medium23-cpu-2026-04-03', 'score': 3.62},
        {'label': 'ROI branch rejection', 'run_id': 'robust_current-dynamic-main-roi-cpu-2026-04-05', 'score': 4.47},
        {'label': 'Current floor', 'run_id': best['run_id'], 'score': best['current_workflow_score']},
    ]

    local_frontier_run_ids = [
        'robust_current-av1-524x394-rgb24-promoted-cpu-2026-04-05',
        'robust_current-av1-524x394-crf34-promoted-cpu-2026-04-06',
        'robust_current-av1-524x394-crf35-rejected-cpu-2026-04-06',
        'robust_current-av1-524x394-unsharp030-rejected-cpu-2026-04-06',
        'robust_current-av1-524x394-filmgrain0-rejected-cpu-2026-04-06',
        'robust_current-av1-522x392-rejected-cpu-2026-04-06',
        'robust_current-av1-524x394-upscale-lanczos-promoted-cpu-2026-04-06',
        'robust_current-sharpness1-promoted-cpu-2026-04-06',
        'robust_current-exp-h-sharpness1-consensus-rejected-cpu-2026-04-07',
        'robust_current-exp-j-preprocess-rejected-cpu-2026-04-07',
        'robust_current-grain-mask-final-rejected-cpu-2026-04-07',
        'robust_current-av1-522x392-postfilter-promoted-cpu-2026-04-07',
    ]
    local_frontier = [run_index[rid] for rid in local_frontier_run_ids if rid in run_index]

    data = {
        'summary': {
            'best_track_b_score': best['current_workflow_score'],
            'best_track_b_bytes': best['archive_bytes'],
            'best_track_b_run_id': best['run_id'],
            'latest_track_b_score': latest['current_workflow_score'],
            'track_a_current_workflow_score': exact_runs[-1]['current_workflow_score'] if exact_runs else None,
            'robust_run_count': len(robust_runs),
            'promotion_count': len(promoted),
            'rejection_count': len(rejected),
        },
        'promotion_ladder': promotion_ladder,
        'promotion_accounting': promotion_accounting,
        'negative_results': negative_results,
        'key_turning_points': key_turning_points,
        'local_frontier': [
            {
                'run_id': r['run_id'],
                'score': r['current_workflow_score'],
                'archive_bytes': r['archive_bytes'],
                'notes': r.get('notes', []),
                'config': r.get('config', {}),
                'parts': score_parts(r),
            }
            for r in local_frontier
        ],
        'results': [
            {
                'run_id': r['run_id'],
                'track': r['track'],
                'score': r['current_workflow_score'],
                'archive_bytes': r['archive_bytes'],
                'ts_utc': r['ts_utc'],
                'notes': clean_public_notes(r.get('notes', [])),
                'config': r.get('config', {}),
                'is_promotion': r['run_id'] in promoted_id_set,
            }
            for r in results
        ],
        'timeline': notable_events,
        'theses': [
            'Strong standard-codec AV1 was already viable here; the real blocker was a rawvideo byte-format bug in the inflator.',
            'BAT00 became useful as a research-only ranking lane, not as a score authority.',
            'ROI-style multi-stream ideas were informative negative results but too expensive in the tested forms.',
            'The x265 ladder still matters as evidence: it established a 3.25 local floor before the repaired AV1 branch advanced through 2.20, 2.19, 2.18, 2.12, 2.08, 2.05, 1.99, 1.95, 1.92, 1.85, 1.84, and now 1.73.',
        ],
        'media_manifest': media_manifest,
        'site_meta': {
            'challenge_url': CHALLENGE_URL,
            'github_url': GITHUB_URL,
            'repo_slug': 'adpena/comma-lab',
            'maintainer': 'Alejandro Pena',
            'updated_at_utc': build_time_utc.isoformat().replace('+00:00', 'Z'),
            'updated_at_local': format_local_datetime(build_time_utc),
        },
    }
    OUT_JSON.write_text(json.dumps(data, indent=2))
    OUT_TIMELINE.write_text(json.dumps([
        {
            'run_id': r['run_id'],
            'ts_utc': r['ts_utc'],
            'score': r['current_workflow_score'],
            'archive_bytes': r['archive_bytes'],
            'config': r.get('config', {}),
            'notes': r.get('notes', []),
        }
        for r in robust_runs
    ], indent=2))

    param_nodes: dict[str, dict] = {}
    graph_nodes: list[dict] = []
    graph_edges: list[dict] = []
    for r in results:
        graph_nodes.append({
            'id': r['run_id'],
            'type': 'run',
            'score': r['current_workflow_score'],
            'bytes': r['archive_bytes'],
            'track': r['track'],
        })
        cfg = r.get('config', {})
        for key, value in (
            ('track', r['track']),
            ('codec', cfg.get('video_codec') or 'libx265'),
            ('scale', f"{cfg.get('scale_w')}x{cfg.get('scale_h')}" if cfg.get('scale_w') else None),
            ('filters', cfg.get('filters')),
            ('crf', cfg.get('crf')),
            ('gop', cfg.get('gop')),
        ):
            if value is None:
                continue
            node_id = f'{key}:{value}'
            if node_id not in param_nodes:
                param_nodes[node_id] = {'id': node_id, 'type': key}
            graph_edges.append({'source': node_id, 'target': r['run_id']})
    OUT_GRAPH.write_text(json.dumps({
        'nodes': list(param_nodes.values()) + graph_nodes,
        'edges': graph_edges,
    }, indent=2))

    av1_bug = next((r for r in robust_runs if r.get('config', {}).get('video_codec') == 'libsvtav1' and r['current_workflow_score'] > 50), None)
    roi_fail = next((r for r in robust_runs if 'roi-two-pass' in r['run_id']), None)
    dynamic_roi_fail = next((r for r in robust_runs if 'dynamic-main-roi' in r['run_id']), None)
    nearby_failures = [
        r for r in robust_runs
        if any(tag in r['run_id'] for tag in ('428x320', '426x320', '416x312'))
    ][:3]
    first_av1_promo = next((r for r in promoted if r.get('config', {}).get('video_codec') == 'libsvtav1'), None)
    last_x265_promo = None
    if first_av1_promo:
        idx = promoted.index(first_av1_promo)
        if idx > 0:
            last_x265_promo = promoted[idx - 1]
    promotion_nodes = promoted[-7:]
    main_positions = []
    if promotion_nodes:
        step = 780 / max(1, len(promotion_nodes) - 1)
        for idx, run in enumerate(promotion_nodes):
            main_positions.append({
                'run': run,
                'x': 70 + idx * step,
                'y': 220,
            })
    position_by_id = {item['run']['run_id']: item for item in main_positions}
    lineage_nodes: list[dict] = []
    lineage_edges: list[tuple[str, str, str]] = []
    for item in main_positions:
        run = item['run']
        family = 'promotion-av1' if run.get('config', {}).get('video_codec') == 'libsvtav1' else 'promotion-x265'
        lineage_nodes.append({
            'id': run['run_id'],
            'x': item['x'],
            'y': item['y'],
            'score': run['current_workflow_score'],
            'label': short_label(run),
            'family': family,
        })
    for left, right in zip(main_positions, main_positions[1:]):
        lineage_edges.append((left['run']['run_id'], right['run']['run_id'], 'mainline'))

    def add_branch(run: dict | None, parent: dict | None, *, y: float, family: str) -> None:
        if not run or not parent:
            return
        lineage_nodes.append({
            'id': run['run_id'],
            'x': parent['x'] + 60,
            'y': y,
            'score': run['current_workflow_score'],
            'label': short_label(run),
            'family': family,
        })
        lineage_edges.append((parent['run']['run_id'], run['run_id'], 'branch'))

    if last_x265_promo and last_x265_promo['run_id'] in position_by_id:
        parent = position_by_id[last_x265_promo['run_id']]
        add_branch(roi_fail, parent, y=82, family='failure')
        add_branch(dynamic_roi_fail, parent, y=358, family='failure')
        for idx, run in enumerate(nearby_failures):
            add_branch(run, parent, y=120 + idx * 68, family='failure')
        add_branch(av1_bug, parent, y=92, family='failure-av1')
    if av1_bug and first_av1_promo and first_av1_promo['run_id'] in position_by_id:
        lineage_edges.append((av1_bug['run_id'], first_av1_promo['run_id'], 'repair'))

    prev_best = promotion_nodes[-2] if len(promotion_nodes) > 1 else best
    best_delta_score = best['current_workflow_score'] - prev_best['current_workflow_score']
    best_delta_bytes = best['archive_bytes'] - prev_best['archive_bytes']
    best_delta_rate = best.get('rate', 0.0) - prev_best.get('rate', 0.0)
    best_delta_pose = best.get('posenet_distortion', 0.0) - prev_best.get('posenet_distortion', 0.0)
    best_delta_seg = best.get('segnet_distortion', 0.0) - prev_best.get('segnet_distortion', 0.0)
    best_note = promoted_run_note(best)
    prev_best_note = promoted_run_note(prev_best)
    best_cfg = best.get('config', {})
    config_line = (
        f"robust_current · {best_cfg.get('video_codec', 'unknown')} · "
        f"{best_cfg.get('scale_w')}x{best_cfg.get('scale_h')} · "
        f"film-grain={best_cfg.get('film_grain')} · "
        f"{best_cfg.get('filters')} · {best_note}"
    )

    robust_only = [r for r in data['results'] if r['track'] == 'robust_current']
    scatter_runs = [r for r in robust_only if r['score'] <= 10.0]
    omitted_outliers = [r for r in robust_only if r['score'] > 10.0]
    width = 920
    height = 420
    pad = 68

    def fmt_axis_bytes(value: int) -> str:
        if value >= 1_000_000:
            return f'{value / 1_000_000:.1f}M'
        return f'{value / 1_000:.0f}k'

    def scatter_color(run: dict) -> str:
        return '#2ad4a0' if run['is_promotion'] else '#ff6b6b' if any('reject' in n.lower() for n in run['notes']) else '#8ab4ff'

    def scatter_status(run: dict) -> str:
        if run['is_promotion']:
            return 'promotion'
        if any('reject' in n.lower() for n in run['notes']):
            return 'explicit rejection'
        return 'measured run'

    def render_scatter_svg(
        runs: list[dict],
        *,
        x_min: int | None = None,
        x_max: int | None = None,
        y_min: float | None = None,
        y_max: float | None = None,
    ) -> str:
        plot_runs = [
            r for r in runs
            if (x_min is None or r['archive_bytes'] >= x_min)
            and (x_max is None or r['archive_bytes'] <= x_max)
            and (y_min is None or r['score'] >= y_min)
            and (y_max is None or r['score'] <= y_max)
        ] or runs

        min_score = y_min if y_min is not None else min(r['score'] for r in plot_runs)
        max_score = y_max if y_max is not None else min(max(r['score'] for r in plot_runs), 6.0)
        min_bytes = x_min if x_min is not None else min(r['archive_bytes'] for r in plot_runs)
        max_bytes = x_max if x_max is not None else max(r['archive_bytes'] for r in plot_runs)
        log_min_bytes = math.log10(min_bytes)
        log_max_bytes = math.log10(max_bytes)

        def sx(x: float) -> float:
            if log_max_bytes == log_min_bytes:
                return width / 2
            return pad + (math.log10(x) - log_min_bytes) * (width - 2 * pad) / (log_max_bytes - log_min_bytes)

        def sy(y: float) -> float:
            if max_score == min_score:
                return height / 2
            return height - pad - (y - min_score) * (height - 2 * pad) / (max_score - min_score)

        x_ticks = [v for v in [700_000, 900_000, 1_200_000, 1_700_000, 2_500_000, 4_000_000, 5_000_000] if min_bytes <= v <= max_bytes]
        y_ticks = [v for v in [2.0, 2.5, 3.0, 4.0, 5.0, 6.0] if min_score <= v <= max_score]
        grid_lines = []
        for tick in x_ticks:
            x = sx(tick)
            grid_lines.append(f'<line x1="{x:.1f}" y1="{pad}" x2="{x:.1f}" y2="{height-pad}" stroke="rgba(255,255,255,0.08)" />')
            grid_lines.append(f'<text x="{x:.1f}" y="{height-pad+20}" text-anchor="middle" class="axis-tick">{fmt_axis_bytes(tick)}</text>')
        for tick in y_ticks:
            y = sy(tick)
            grid_lines.append(f'<line x1="{pad}" y1="{y:.1f}" x2="{width-pad}" y2="{y:.1f}" stroke="rgba(255,255,255,0.08)" />')
            grid_lines.append(f'<text x="{pad-10}" y="{y+4:.1f}" text-anchor="end" class="axis-tick">{tick:g}</text>')

        points = []
        for r in plot_runs:
            tip_label = short_label(r)
            tip_meta = f'{scatter_status(r)} · score {r["score"]:.2f} · {r["archive_bytes"]:,} bytes'
            tip_detail = describe_config(r.get('config', {}))
            aria = escape(f'{tip_label}. {tip_meta}. {tip_detail}.')
            points.append(
                f'<circle class="scatter-point" data-scatter-point tabindex="0" role="button" '
                f'data-tip-label="{escape(tip_label)}" data-tip-meta="{escape(tip_meta)}" data-tip-detail="{escape(tip_detail)}" '
                f'aria-label="{aria}" cx="{sx(r["archive_bytes"]):.1f}" cy="{sy(r["score"]):.1f}" r="4.25" fill="{scatter_color(r)}"></circle>'
            )

        return (
            f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="Score versus bytes scatterplot for Track B runs">'
            f'{"".join(grid_lines)}'
            f'<line x1="{pad}" y1="{height-pad}" x2="{width-pad}" y2="{height-pad}" stroke="#475569" />'
            f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{height-pad}" stroke="#475569" />'
            f'{"".join(points)}'
            f'<text x="{width/2:.1f}" y="{height-10}" text-anchor="middle" class="axis-label">archive size (bytes, log scale)</text>'
            f'<text x="16" y="{height/2:.1f}" transform="rotate(-90 16 {height/2:.1f})" text-anchor="middle" class="axis-label">current_workflow score (linear, lower is better)</text>'
            f'</svg>'
        )

    full_scatter_svg = render_scatter_svg(scatter_runs)
    focus_scatter_svg = render_scatter_svg(scatter_runs, x_min=780_000, x_max=2_900_000, y_min=2.0, y_max=4.8)

    timeline_rows = []
    for e in notable_events[-14:]:
        timeline_rows.append(
            f"<tr><td>{escape(format_local_datetime(e.get('ts_utc')))}</td><td>{escape(e.get('type',''))}</td><td>{escape(e.get('summary',''))}</td></tr>"
        )

    ladder_items = []
    for r in reversed(promotion_ladder):
        cfg = r['config']
        ladder_items.append(
            f"<li><strong>{r['score']:.2f}</strong> — {escape(describe_config(cfg))}</li>"
        )

    accounting_rows = []
    for r in promotion_accounting:
        cfg = r['config']
        rf_score = '' if r['rule_faithful_score'] is None else f"{r['rule_faithful_score']:.3f}"
        accounting_rows.append(
            '<tr>'
            f"<td>{escape(r['run_id'])}</td>"
            f"<td>{escape(str(cfg.get('scale_w')))}x{escape(str(cfg.get('scale_h')))}</td>"
            f"<td>{escape(str(cfg.get('filters')))}</td>"
            f"<td>{r['current_workflow_score']:.2f}</td>"
            f"<td>{fmt_int(r['current_workflow_bytes'])}</td>"
            f"<td>{rf_score}</td>"
            f"<td>{fmt_int(r['rule_faithful_bytes'])}</td>"
            '</tr>'
        )

    negative_items = []
    for r in negative_results[:8]:
        negative_items.append(
            f"<li><strong>{r['score']:.2f}</strong> — {escape(r['run_id'])}</li>"
        )

    turning_items = []
    for t in key_turning_points:
        turning_items.append(f"<li><strong>{escape(t['label'])}</strong> — {escape(t['run_id'])} ({t['score']:.2f})</li>")

    frontier_rows = []
    for idx, r in enumerate(data['local_frontier']):
        prev = data['local_frontier'][idx - 1] if idx > 0 else None
        baseline = prev
        if 'upscale-lanczos' in r['run_id'] or '522x392' in r['run_id']:
            baseline = next((x for x in data['local_frontier'] if x['run_id'] == 'robust_current-av1-524x394-crf34-promoted-cpu-2026-04-06'), prev)
        if 'colorspace-hardening' in r['run_id']:
            baseline = next((x for x in data['local_frontier'] if x['run_id'] == 'robust_current-av1-524x394-upscale-lanczos-promoted-cpu-2026-04-06'), prev)
        delta_score = '' if prev is None else f"{r['score'] - prev['score']:+.2f}"
        delta_bytes = '' if prev is None else f"{r['archive_bytes'] - prev['archive_bytes']:+,}"
        if baseline is not None:
            delta_score = f"{r['score'] - baseline['score']:+.2f}"
            delta_bytes = f"{r['archive_bytes'] - baseline['archive_bytes']:+,}"
        cfg = r['config']
        run_id = r['run_id']
        if '522x392' in run_id:
            label = '522x392'
            axis = 'geometry 522x392'
        elif 'upscale-lanczos' in run_id:
            label = 'lanczos upscale'
            axis = 'upscale lanczos'
        elif 'postfilter' in run_id:
            label = 'learned post-filter'
            axis = 'decode learned int8 post-filter'
        elif 'grain-mask' in run_id:
            label = 'grain mask'
            axis = 'decode saliency-masked grain synthesis'
        elif 'exp-j' in run_id:
            label = 'ROI preprocess'
            axis = 'corridor preprocessing'
        elif 'exp-h' in run_id:
            label = 'consensus stack'
            axis = 'crf33 + scd0 + hqdn3d'
        elif 'filmgrain0' in run_id or cfg.get('film_grain') == 0:
            label = 'film-grain 0'
            axis = f"film-grain {cfg.get('film_grain')}"
        elif 'unsharp030' in run_id:
            label = 'unsharp 0.30'
            axis = f"postfilter {cfg['inflate_postfilter']}"
        elif cfg.get('crf') is not None:
            label = f"crf {cfg['crf']}"
            axis = f"crf {cfg['crf']}"
        elif cfg.get('inflate_postfilter'):
            label = 'postfilter'
            axis = f"postfilter {cfg['inflate_postfilter']}"
        else:
            label = short_label(r)
            axis = describe_config(cfg)
        verdict = 'promoted' if 'promoted' in r['run_id'] else 'rejected'
        frontier_rows.append(
            '<tr>'
            f'<td><span title="{escape(r["run_id"])}">{escape(label)}</span></td>'
            f"<td>{escape(axis)}</td>"
            f"<td>{r['score']:.2f}</td>"
            f"<td>{r['archive_bytes']:,}</td>"
            f"<td>{escape(delta_score)}</td>"
            f"<td>{escape(delta_bytes)}</td>"
            f"<td>{verdict}</td>"
            '</tr>'
        )

    decomposition_rows = []
    for r in data['local_frontier']:
        parts = r['parts']
        total = parts['seg_component'] + parts['pose_component'] + parts['rate_component']
        decomposition_rows.append(
            '<tr>'
            f"<td>{escape(r['run_id'])}</td>"
            f"<td>{parts['seg_component']:.3f}</td>"
            f"<td>{parts['pose_component']:.3f}</td>"
            f"<td>{parts['rate_component']:.3f}</td>"
            f"<td>{total:.3f}</td>"
            '</tr>'
        )

    current_best_id = best['run_id']
    x265_floor_id = 'robust_current-cand-424x318-crf23-g48-b4-r4-cpu-2026-04-05'
    av1_bug_id = 'robust_current-av1-524x394-cpu-2026-04-05'
    roi_fail_id = 'robust_current-dynamic-main-roi-cpu-2026-04-05'
    nearby_reject_id = 'robust_current-cand-428x320-crf23-g48-b4-r4-cpu-2026-04-05'
    h16_floor_id = 'robust_current-long500-h16-promoted-cpu-2026-04-08'

    trajectory_mainline = [
        ('robust_current-baseline-cpu-2026-04-03', 'baseline'),
        (x265_floor_id, 'x265 floor'),
        ('robust_current-av1-524x394-rgb24-promoted-cpu-2026-04-05', 'AV1 repair'),
        ('robust_current-sharpness1-promoted-cpu-2026-04-06', 'sharpness=1'),
        ('robust_current-av1-522x392-postfilter-promoted-cpu-2026-04-07', 'first post-filter'),
        (h16_floor_id, 'long500 h16'),
        (current_best_id, 'current floor'),
    ]

    trajectory_branches = [
        (roi_fail_id, 'ROI fail', 'branch', 'robust_current-432x324-cpu-2026-04-04'),
        (nearby_reject_id, 'nearby scale', 'branch', x265_floor_id),
        (av1_bug_id, 'AV1 failure', 'diagnostic', 'robust_current-av1-524x394-rgb24-promoted-cpu-2026-04-05'),
    ]

    trajectory_width = 1180
    trajectory_height = 360
    traj_pad_left = 78
    traj_pad_right = 28
    traj_pad_top = 24
    traj_pad_bottom = 46
    traj_visible_min = 1.8
    traj_visible_max = 5.0

    mainline_nodes = []
    for idx, (run_id, label) in enumerate(trajectory_mainline):
        run = run_index.get(run_id)
        if not run:
            continue
        x = traj_pad_left + idx * ((trajectory_width - traj_pad_left - traj_pad_right) / max(1, len(trajectory_mainline) - 1))
        score = run['current_workflow_score']
        y = traj_pad_top + (traj_visible_max - score) * (trajectory_height - traj_pad_top - traj_pad_bottom) / (traj_visible_max - traj_visible_min)
        mainline_nodes.append({
            'run_id': run_id,
            'label': label,
            'score': score,
            'bytes': run['archive_bytes'],
            'config': run.get('config', {}),
            'x': x,
            'y': y,
            'kind': 'mainline',
        })

    mainline_lookup = {node['run_id']: node for node in mainline_nodes}
    branch_nodes = []
    for run_id, label, kind, parent_id in trajectory_branches:
        run = run_index.get(run_id)
        parent = mainline_lookup.get(parent_id)
        if not run or not parent:
            continue
        score = run['current_workflow_score']
        visible_score = min(score, traj_visible_max)
        y = traj_pad_top + (traj_visible_max - visible_score) * (trajectory_height - traj_pad_top - traj_pad_bottom) / (traj_visible_max - traj_visible_min)
        branch_nodes.append({
            'run_id': run_id,
            'label': label,
            'score': score,
            'bytes': run['archive_bytes'],
            'config': run.get('config', {}),
            'x': parent['x'],
            'y': y,
            'kind': kind,
            'parent_id': parent_id,
            'offscale': score > traj_visible_max,
        })

    def trajectory_axis_y(value: float) -> float:
        return traj_pad_top + (traj_visible_max - value) * (trajectory_height - traj_pad_top - traj_pad_bottom) / (traj_visible_max - traj_visible_min)

    trajectory_grid_svg = []
    for tick in [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]:
        y = trajectory_axis_y(tick)
        trajectory_grid_svg.append(f'<line x1="{traj_pad_left}" y1="{y:.1f}" x2="{trajectory_width - traj_pad_right}" y2="{y:.1f}" stroke="rgba(255,255,255,0.08)" />')
        trajectory_grid_svg.append(f'<text x="{traj_pad_left - 10}" y="{y + 4:.1f}" text-anchor="end" class="axis-tick">{tick:.1f}</text>')

    trajectory_path = ' '.join(
        [f'M {mainline_nodes[0]["x"]:.1f} {mainline_nodes[0]["y"]:.1f}'] +
        [f'L {node["x"]:.1f} {node["y"]:.1f}' for node in mainline_nodes[1:]]
    ) if mainline_nodes else ''

    def trajectory_short_detail(node: dict) -> str:
        cfg = node.get('config', {})
        if node['kind'] == 'diagnostic':
            return f'{node["score"]:.2f} off-scale'
        if cfg.get('video_codec') == 'libsvtav1':
            return f'{node["score"]:.2f} · {cfg.get("scale_w")}x{cfg.get("scale_h")}'
        return f'{node["score"]:.2f} · x265'

    trajectory_nodes_svg = []
    for idx, node in enumerate(mainline_nodes):
        label_y = node['y'] - 18 if idx % 2 == 0 else node['y'] + 28
        anchor = 'middle'
        title = escape(f'{node["run_id"]} | score={node["score"]:.2f} | bytes={node["bytes"]:,}')
        trajectory_nodes_svg.append(
            f'<path class="trajectory-stem" d="M {node["x"]:.1f} {node["y"]:.1f} L {node["x"]:.1f} {label_y - 8:.1f}" />'
            f'<circle class="trajectory-node trajectory-node-main" cx="{node["x"]:.1f}" cy="{node["y"]:.1f}" r="6.5"><title>{title}</title></circle>'
            f'<text class="trajectory-label" x="{node["x"]:.1f}" y="{label_y:.1f}" text-anchor="{anchor}">{escape(node["label"])}</text>'
            f'<text class="trajectory-meta" x="{node["x"]:.1f}" y="{label_y + 13:.1f}" text-anchor="{anchor}">{escape(trajectory_short_detail(node))}</text>'
        )

    trajectory_branch_svg = []
    for node in branch_nodes:
        parent = mainline_lookup[node['parent_id']]
        branch_x = parent['x'] + 46
        label_y = node['y'] - 16 if node['kind'] != 'branch' else node['y'] + 28
        color_cls = 'trajectory-node-diagnostic' if node['kind'] == 'diagnostic' else 'trajectory-node-branch'
        edge_cls = 'trajectory-edge-diagnostic' if node['kind'] == 'diagnostic' else 'trajectory-edge-branch'
        title = escape(f'{node["run_id"]} | score={node["score"]:.2f} | bytes={node["bytes"]:,}')
        branch_y = max(traj_pad_top + 8, node['y'])
        trajectory_branch_svg.append(
            f'<path class="trajectory-edge {edge_cls}" d="M {parent["x"]:.1f} {parent["y"]:.1f} C {parent["x"] + 18:.1f} {parent["y"]:.1f}, {branch_x - 18:.1f} {branch_y:.1f}, {branch_x:.1f} {branch_y:.1f}" />'
            f'<circle class="trajectory-node {color_cls}" cx="{branch_x:.1f}" cy="{branch_y:.1f}" r="5.5"><title>{title}</title></circle>'
            f'<text class="trajectory-label" x="{branch_x:.1f}" y="{label_y:.1f}" text-anchor="middle">{escape(node["label"])}</text>'
            f'<text class="trajectory-meta" x="{branch_x:.1f}" y="{label_y + 13:.1f}" text-anchor="middle">{escape(trajectory_short_detail(node))}</text>'
        )
        if node['offscale']:
            trajectory_branch_svg.append(
                f'<text class="trajectory-offscale" x="{branch_x:.1f}" y="{traj_pad_top - 4:.1f}" text-anchor="middle">off-scale diagnostic spike</text>'
            )

    baseline_run = run_index.get('robust_current-baseline-cpu-2026-04-03')
    baseline_score = baseline_run['current_workflow_score'] if baseline_run else None
    baseline_bytes = baseline_run['archive_bytes'] if baseline_run else None
    published_baseline_score = best.get('published_baseline_score')
    published_delta_score = (
        best['current_workflow_score'] - published_baseline_score
        if published_baseline_score is not None else None
    )
    fixed_av1_score = first_av1_promo['current_workflow_score'] if first_av1_promo else best['current_workflow_score']
    hypothesis_cards = [
        {
            'title': 'Evaluator path',
            'summary': 'The scorer resizes frames and measures task distortion. It is sensitive to pipeline details, not just visual appearance.',
            'detail': 'PoseNet sees both frames. SegNet sees only the last frame in each pair. That asymmetry is why small encoding or decode-path changes can move the score.',
        },
        {
            'title': 'Critical bug',
            'summary': 'The main AV1 failure was a byte-layout error, not a codec-limit problem.',
            'detail': 'The failed path emitted rawvideo as yuv444p bytes. The corrected path forces rgb24, which matches the evaluator’s raw-frame expectation.',
        },
        {
            'title': 'Current operating point',
            'summary': f'The current honest floor is {best["current_workflow_score"]:.2f} at {best["archive_bytes"]:,} bytes.',
            'detail': (
                f'Against the first honest baseline'
                f'{f" ({baseline_score:.2f} at {baseline_bytes:,} bytes)" if baseline_score is not None and baseline_bytes is not None else ""}, '
                'the lab reduced bytes while preserving task signal well enough to keep the score moving down.'
            ),
        },
    ]

    compare_manifest = media_manifest or {}
    compare_variants = compare_manifest.get('variants') or [
        {
            'id': 'floor_173',
            'label': '1.73 current floor',
            'note': 'long1000 h64',
            'score': 1.73,
            'preview': 'inflated_preview.mp4',
            'poster': 'inflated_poster.jpg',
        }
    ]
    leaderboard_refs = compare_manifest.get('leaderboard_refs') or []
    current_variant = compare_variants[0]
    compare_buttons_html = ''.join(
        f'<button type="button" data-run-button data-run-id="{escape(v["id"])}" '
        f'data-run-label="{escape(v["label"])}" data-run-note="{escape(v["note"])}" '
        f'data-run-score="{v["score"]:.2f}" class="{"is-active" if idx == 0 else ""}">'
        f'<span>{escape(v["label"])}</span><small>{escape(v["note"])}</small></button>'
        for idx, v in enumerate(compare_variants)
    )
    compare_stage_html = ''.join(
        f'<div class="run-panel {"is-active" if idx == 0 else ""}" data-run-panel="{escape(v["id"])}">'
        f'<video data-compare-video data-run-id="{escape(v["id"])}" preload="metadata" muted playsinline '
        f'poster="./media/{escape(v["poster"])}" src="./media/{escape(v["preview"])}"></video>'
        f'</div>'
        for idx, v in enumerate(compare_variants)
    )
    compare_zoom_html = ''.join(
        f'<div class="run-panel {"is-active" if idx == 0 else ""}" data-run-zoom-panel="{escape(v["id"])}">'
        f'<div class="zoom-viewport" data-zoom-viewport>'
        f'<video data-compare-video data-run-id="{escape(v["id"])}" preload="metadata" muted playsinline '
        f'poster="./media/{escape(v["poster"])}" src="./media/{escape(v["preview"])}"></video>'
        f'</div>'
        f'</div>'
        for idx, v in enumerate(compare_variants)
    )
    leaderboard_refs_html = ''.join(
        f'<span class="leader-pill"><strong>{escape(ref["label"])}</strong><span>{ref["score"]:.2f}</span><span>{escape(ref["note"])}</span></span>'
        for ref in leaderboard_refs
    )
    compare_updated = compare_manifest.get('updated_at_utc')
    compare_updated_label = format_local_datetime(compare_updated) if compare_updated else format_local_datetime(build_time_utc)

    concept_cards = [
        {
            'title': 'Archive bytes',
            'summary': 'Smaller is better, but only if the reconstructed video still preserves the task signal that the scorer measures.',
        },
        {
            'title': 'PoseNet distortion',
            'summary': 'This term measures how much the compressed video changes driving-related pose outputs. Lower is better.',
        },
        {
            'title': 'SegNet distortion',
            'summary': 'This term measures disagreement on segmentation labels. At the current operating point, tiny SegNet regressions matter a lot.',
        },
    ]

    delta_metric_rows = [
        ('current_workflow score', f'{prev_best["current_workflow_score"]:.2f}', f'{best["current_workflow_score"]:.2f}', f'{best_delta_score:+.2f}'),
        ('archive bytes', f'{prev_best["archive_bytes"]:,}', f'{best["archive_bytes"]:,}', f'{best_delta_bytes:+,}'),
        ('PoseNet distortion', f'{prev_best.get("posenet_distortion", 0.0):.8f}', f'{best.get("posenet_distortion", 0.0):.8f}', f'{best_delta_pose:+.8f}'),
        ('SegNet distortion', f'{prev_best.get("segnet_distortion", 0.0):.8f}', f'{best.get("segnet_distortion", 0.0):.8f}', f'{best_delta_seg:+.8f}'),
    ]
    delta_rows_html = []
    for label, old, new, delta in delta_metric_rows:
        delta_class = 'delta-negative' if delta.startswith('-') else 'delta-positive'
        delta_rows_html.append(
            '<tr>'
            f'<td>{escape(label)}</td>'
            f'<td>{escape(old)}</td>'
            f'<td>{escape(new)}</td>'
            f'<td class="{delta_class}">{escape(delta)}</td>'
            '</tr>'
        )
    published_delta_display = '' if published_delta_score is None else f'{published_delta_score:+.2f}'
    published_baseline_display = '' if published_baseline_score is None else f'{published_baseline_score:.2f}'

    html = f'''<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>comma-lab</title>
  <meta name="description" content="Scorer-backed compression experiments, evidence, and reporting for the comma video challenge." />
  <meta name="robots" content="noindex,nofollow" />
  <meta name="theme-color" content="#0f1724" />
  <meta name="color-scheme" content="dark" />
  <meta property="og:title" content="comma-lab" />
  <meta property="og:description" content="Scorer-backed compression experiments and evidence for the comma video challenge." />
  <meta property="og:type" content="website" />
  <meta name="twitter:card" content="summary_large_image" />
  <script type="application/ld+json">{{"@context":"https://schema.org","@type":"WebSite","name":"comma-lab"}}</script>
  <style>
    :root {{
      --bg: #0f1724;
      --bg-2: #101d2f;
      --panel: rgba(255,255,255,0.055);
      --panel-strong: rgba(255,255,255,0.045);
      --stroke: rgba(255,255,255,0.10);
      --text: #f4f1e8;
      --muted: #a8b1bc;
      --soft: #d9e0e8;
      --accent: #8ed1c0;
      --accent-2: #f0b35f;
      --danger: #ff6b6b;
      --shadow: none;
      --radius: 20px;
      --radius-sm: 14px;
      --max: 1200px;
    }}
    * {{ box-sizing: border-box; }}
    html {{ scroll-behavior: smooth; }}
    body {{
      margin: 0;
      background:
        radial-gradient(circle at top left, rgba(142,209,192,0.14), transparent 32%),
        radial-gradient(circle at top right, rgba(240,179,95,0.12), transparent 30%),
        linear-gradient(180deg, #111b2c 0%, #0c1320 48%, #0a1018 100%);
      color: var(--text);
      font: 500 16px/1.6 ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "SF Pro Display", sans-serif;
      text-rendering: optimizeLegibility;
    }}
    a {{ color: inherit; text-underline-offset: 0.2em; }}
    a:focus-visible, button:focus-visible {{ outline: 3px solid var(--accent-2); outline-offset: 3px; border-radius: 6px; }}
    .skip {{ position: absolute; left: 12px; top: -48px; background: #fff; color: #000; padding: 10px 14px; border-radius: 10px; z-index: 99; }}
    .skip:focus {{ top: 12px; }}
    .wrap {{ max-width: var(--max); margin: 0 auto; padding: 28px 24px 72px; }}
    .hero {{ padding: 28px 0 10px; display: grid; gap: 18px; }}
    .eyebrow {{ color: var(--muted); font-size: 11px; letter-spacing: 0.12em; text-transform: uppercase; font-weight: 700; }}
    h1 {{ margin: 0; font-family: "Iowan Old Style", "Palatino Linotype", "Book Antiqua", Georgia, ui-serif, serif; font-size: clamp(42px, 6vw, 78px); line-height: 0.94; letter-spacing: -0.05em; font-weight: 700; max-width: 11ch; }}
    h2, h3 {{ font-family: "Iowan Old Style", "Palatino Linotype", "Book Antiqua", Georgia, ui-serif, serif; }}
    .lede {{ max-width: 760px; color: var(--soft); font-size: 17px; margin: 0; }}
    .hero-grid {{ display:grid; grid-template-columns: 1.1fr 0.9fr; gap: 18px; align-items: start; }}
    .hero-stack {{ display:grid; gap: 14px; }}
    .pill-row {{ display:flex; gap:10px; flex-wrap:wrap; }}
    .pill {{ border: 1px solid var(--stroke); background: rgba(255,255,255,0.035); padding: 8px 12px; border-radius: 999px; color: var(--soft); font-size: 12px; }}
    .hero-card-grid {{ display:grid; gap: 12px; grid-template-columns: repeat(3, minmax(0, 1fr)); }}
    .hero-card {{
      background: linear-gradient(180deg, rgba(255,255,255,0.05), rgba(255,255,255,0.025));
      border: 1px solid var(--stroke);
      border-radius: 16px;
      padding: 16px;
      min-height: 100%;
    }}
    .hero-card h2 {{ margin: 0 0 8px; font-size: 18px; letter-spacing: -0.03em; }}
    .hero-card p {{ margin: 0; color: var(--soft); font-size: 14px; line-height: 1.6; }}
    .hero-links {{ display:flex; flex-wrap:wrap; gap: 10px 14px; margin-top: 10px; font-size: 13px; color: var(--muted); }}
    .hero-links a {{ text-decoration: none; }}
    .two {{ display: grid; gap: 16px; }}
    .two {{ grid-template-columns: 1.25fr 0.75fr; }}
    .panel {{ background: linear-gradient(180deg, rgba(16,23,36,0.92), rgba(10,14,20,0.92)); border: 1px solid var(--stroke); border-radius: 14px; box-shadow: none; padding: 20px; margin-top: 18px; }}
    .panel h2 {{ margin: 0 0 10px; font-size: 20px; letter-spacing: -0.03em; }}
    .muted {{ color: var(--muted); }}
    .subnav {{ display:flex; gap:14px; flex-wrap:wrap; font-size: 13px; color: var(--muted); }}
    .subnav a {{ text-decoration: none; }}
    .context-strip {{ display:grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 0; margin-top: 12px; border-top: 1px solid var(--stroke); border-bottom: 1px solid var(--stroke); }}
    .context-item {{ padding: 16px 16px 18px; }}
    .context-item + .context-item {{ border-left: 1px solid var(--stroke); }}
    .context-label {{ color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: 0.08em; }}
    .context-copy {{ margin: 8px 0 0; color: var(--soft); font-size: 13px; line-height: 1.6; max-width: 38ch; }}
    .context-meta {{ margin: 8px 0 0; color: var(--muted); font-size: 12px; }}
    .context-links {{ display:flex; flex-wrap:wrap; gap: 10px 14px; margin-top: 8px; color: var(--muted); font-size: 12px; }}
    .context-links a {{ text-decoration: none; }}
    .summary-shell {{ margin-top: 18px; padding: 4px 0 0; }}
    .summary-grid {{ display:grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 0; border-top: 1px solid var(--stroke); border-bottom: 1px solid var(--stroke); }}
    .summary-stat {{ padding: 14px 16px; }}
    .summary-stat + .summary-stat {{ border-left: 1px solid var(--stroke); }}
    .summary-label {{ color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: 0.08em; }}
    .summary-value {{ font-size: 28px; line-height: 1; margin-top: 8px; font-weight: 720; letter-spacing: -0.04em; }}
    .summary-meta {{ color: var(--muted); font-size: 12px; margin-top: 8px; }}
    .concept-grid {{ display:grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 14px; }}
    .concept-card {{ border: 1px solid var(--stroke); border-radius: 14px; padding: 16px; background: rgba(255,255,255,0.03); }}
    .concept-card h3 {{ margin: 0 0 8px; font-size: 18px; letter-spacing: -0.02em; }}
    .concept-card p {{ margin: 0; color: var(--soft); font-size: 14px; line-height: 1.6; }}
    .walkthrough {{ list-style: none; margin: 0; padding: 0; counter-reset: step; }}
    .walkthrough li {{ counter-increment: step; display: grid; grid-template-columns: 28px 1fr; gap: 14px; padding: 14px 0; border-top: 1px solid var(--stroke); }}
    .walkthrough li:first-child {{ border-top: 0; }}
    .step-num {{ color: var(--muted); font: 700 12px/1 ui-monospace, SFMono-Regular, Menlo, monospace; padding-top: 4px; }}
    .step-num::before {{ content: counter(step, decimal-leading-zero); }}
    .step-body h3 {{ margin: 0 0 6px; font-size: 14px; letter-spacing: -0.02em; }}
    .step-body p {{ margin: 0; color: var(--soft); font-size: 13px; line-height: 1.55; }}
    .step-body p + p {{ margin-top: 6px; color: var(--muted); }}
    table {{ width: 100%; border-collapse: collapse; }}
    th, td {{ border-bottom: 1px solid var(--stroke); padding: 10px 8px; text-align: left; vertical-align: top; }}
    th {{ color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: 0.08em; }}
    .table-wrap {{ overflow-x: auto; -webkit-overflow-scrolling: touch; }}
    .table-wrap table {{ min-width: 760px; }}
    ul {{ margin: 0; padding-left: 18px; }}
    li + li {{ margin-top: 8px; }}
    svg {{ width: 100%; height: auto; background: #0b0f14; border: 1px solid var(--stroke); border-radius: 10px; }}
    .lineage-wrap {{ overflow-x: auto; }}
    .lineage-edge {{ fill: none; stroke-width: 2.2; opacity: 0.92; stroke-linecap: round; }}
    .lineage-edge-promotion {{ stroke: #2ad4a0; }}
    .lineage-edge-rejection {{ stroke: #ff6b6b; stroke-dasharray: 6 7; }}
    .lineage-edge-diagnostic {{ stroke: #f6c14b; stroke-dasharray: 2 10; }}
    .lineage-edge-repair {{ stroke: #8ab4ff; stroke-width: 3.2; }}
    .lineage-node {{ stroke: rgba(255,255,255,0.18); stroke-width: 1.2; }}
    .lineage-node-x265 {{ fill: #2ad4a0; }}
    .lineage-node-best {{ fill: #8ab4ff; }}
    .lineage-node-bug {{ fill: #f6c14b; }}
    .lineage-node-roi, .lineage-node-rejection {{ fill: #ff6b6b; }}
    .lineage-note-box {{ fill: rgba(12,16,22,0.94); stroke: rgba(255,255,255,0.09); stroke-width: 1; }}
    .lineage-note-title {{ font-size: 12px; font-weight: 650; fill: var(--text); letter-spacing: -0.01em; }}
    .lineage-note-meta {{ font-size: 10px; fill: var(--muted); }}
    .trajectory-edge {{ fill: none; stroke-linecap: round; stroke-width: 2.4; }}
    .trajectory-edge-main {{ stroke: #8ed1c0; }}
    .trajectory-edge-branch {{ stroke: #ff8f8f; stroke-dasharray: 5 6; }}
    .trajectory-edge-diagnostic {{ stroke: #f0b35f; stroke-dasharray: 2 8; }}
    .trajectory-node {{ stroke: rgba(255,255,255,0.16); stroke-width: 1.4; }}
    .trajectory-node-main {{ fill: #8ed1c0; }}
    .trajectory-node-branch {{ fill: #ff8f8f; }}
    .trajectory-node-diagnostic {{ fill: #f0b35f; }}
    .trajectory-label {{ fill: var(--text); font-size: 11px; font-weight: 650; letter-spacing: -0.01em; }}
    .trajectory-meta {{ fill: var(--muted); font-size: 10px; }}
    .trajectory-stem {{ stroke: rgba(255,255,255,0.10); stroke-width: 1; }}
    .trajectory-offscale {{ fill: #f0b35f; font-size: 10px; letter-spacing: 0.04em; text-transform: uppercase; }}
    .legend-row {{ display:flex; gap:10px; flex-wrap:wrap; margin: 12px 0 0; }}
    .legend-key {{ display:inline-flex; align-items:center; gap:8px; color: var(--muted); font-size: 12px; }}
    .legend-dot {{ width:10px; height:10px; border-radius:999px; display:inline-block; }}
    .legend-line {{ width:22px; height:0; border-top:2px solid currentColor; display:inline-block; }}
    .scatter-shell {{ position: relative; }}
    .scatter-point {{ cursor: pointer; transition: transform 120ms ease; }}
    .scatter-point:hover, .scatter-point:focus-visible {{ transform: scale(1.18); stroke: rgba(255,255,255,0.95); stroke-width: 1.8; outline: none; }}
    .scatter-tooltip {{ position: absolute; top: 0; left: 0; max-width: min(320px, calc(100% - 28px)); padding: 10px 12px; border: 1px solid var(--stroke); border-radius: 10px; background: rgba(8,11,15,0.96); box-shadow: 0 12px 28px rgba(0,0,0,0.28); opacity: 0; transform: translate(var(--tip-x, 14px), calc(var(--tip-y, 14px) + 4px)); transition: opacity 120ms ease, transform 120ms ease; pointer-events: none; }}
    .scatter-tooltip.is-visible {{ opacity: 1; transform: translate(var(--tip-x, 14px), var(--tip-y, 14px)); }}
    .scatter-tooltip strong {{ display:block; font-size: 13px; color: var(--text); }}
    .scatter-tooltip span {{ display:block; color: var(--muted); font-size: 12px; margin-top: 4px; }}
    .bug-box {{ display:grid; grid-template-columns: 1fr auto 1fr; gap: 10px; align-items:center; margin-top: 12px; }}
    .bug-side {{ border: 1px solid var(--stroke); border-radius: var(--radius-sm); padding: 12px; background: rgba(255,255,255,0.03); }}
    .bug-side strong {{ display:block; margin-bottom: 6px; }}
    .metric-delta {{ color: var(--muted); font-size: 12px; margin-top: 6px; }}
    .frontier-callout {{ color: var(--soft); font-size: 14px; line-height: 1.6; margin: 0 0 14px; }}
    .footnote {{ color: var(--muted); font-size: 13px; }}
    .table-hint {{ margin-top: 8px; color: var(--muted); font-size: 12px; }}
    .refs-grid {{ align-items: start; }}
    .refs-group + .refs-group {{ border-left: 1px solid var(--stroke); padding-left: 18px; }}
    .refs-group-title {{ margin: 0 0 10px; color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: 0.08em; }}
    .refs-note {{ margin-top: 12px; }}
    .axis-label {{ fill: var(--muted); font-size: 12px; letter-spacing: 0.02em; }}
    .axis-tick {{ fill: var(--muted); font-size: 11px; }}
    .config-line {{ margin: 10px 0 0; color: var(--muted); font: 500 12px/1.6 ui-monospace, SFMono-Regular, Menlo, monospace; }}
    .list-tight {{ margin: 0; padding-left: 18px; }}
    .list-tight li + li {{ margin-top: 6px; }}
    .compare-head {{ display:flex; justify-content:space-between; align-items:flex-end; gap: 16px; flex-wrap: wrap; }}
    .compare-controls {{ display:flex; gap:12px; row-gap:10px; flex-wrap:wrap; align-items:center; margin-top: 12px; }}
    .compare-controls button {{
      min-height: 36px;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      background: rgba(255,255,255,0.025);
      color: var(--muted);
      border: 1px solid rgba(255,255,255,0.10);
      border-radius: 999px;
      padding: 0 14px;
      font: 600 13px/1 inherit;
      letter-spacing: -0.01em;
      cursor: pointer;
    }}
    .compare-controls button:hover {{ background: rgba(255,255,255,0.045); color: var(--soft); }}
    .compare-controls button.is-active {{
      background: rgba(142,209,192,0.10);
      border-color: rgba(142,209,192,0.42);
      color: var(--text);
      box-shadow: inset 0 0 0 1px rgba(142,209,192,0.14);
    }}
    .compare-controls input[type="range"] {{
      width: min(280px, 42vw);
      margin: 0 2px;
      accent-color: var(--accent);
    }}
    .explorer-toolbar {{ display:grid; gap: 14px; margin-top: 14px; }}
    .run-picker {{ display:flex; gap:10px; flex-wrap:wrap; }}
    .run-picker button {{
      min-width: 156px;
      min-height: 48px;
      display:grid;
      gap:4px;
      text-align:left;
      align-content:center;
      justify-items:start;
      background: rgba(255,255,255,0.028);
      color: var(--soft);
      border: 1px solid rgba(255,255,255,0.09);
      border-radius: 16px;
      padding: 10px 14px;
      font: inherit;
      cursor: pointer;
    }}
    .run-picker button small {{ color: var(--muted); font-size: 11px; }}
    .run-picker button.is-active {{ border-color: rgba(142,209,192,0.42); background: rgba(142,209,192,0.09); }}
    .leaderboard-band {{ display:flex; gap:10px; flex-wrap:wrap; align-items:center; }}
    .leaderboard-label {{ color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: 0.08em; }}
    .leader-pill {{
      display:inline-grid;
      gap:2px;
      padding: 9px 12px;
      border:1px solid rgba(255,255,255,0.08);
      border-radius: 14px;
      background: rgba(255,255,255,0.025);
      color: var(--soft);
      font-size: 12px;
    }}
    .leader-pill strong {{ font-size: 11px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.08em; }}
    .explorer-grid {{ display:grid; grid-template-columns: 1fr 1fr; gap:16px; margin-top: 16px; }}
    .explorer-stage {{
      border:1px solid var(--stroke);
      border-radius:12px;
      background:#0b0f14;
      padding:12px;
    }}
    .explorer-stage h3 {{ margin:0 0 8px; font-size: 13px; }}
    .zoom-stage {{
      position:relative;
      overflow:hidden;
      border-radius:10px;
      background:#000;
    }}
    .zoom-stage video {{ width:100%; display:block; }}
    .zoom-window {{
      position:absolute;
      inset:20% auto auto 35%;
      width:22%;
      aspect-ratio: 420 / 316;
      border: 2px solid rgba(240,179,95,0.95);
      border-radius: 14px;
      box-shadow: 0 0 0 9999px rgba(0,0,0,0.22);
      cursor: grab;
      touch-action: none;
    }}
    .zoom-window.is-dragging {{ cursor: grabbing; }}
    .zoom-grid {{ display:grid; grid-template-columns: 1fr 1fr; gap:16px; margin-top:16px; }}
    .zoom-pane {{
      border:1px solid var(--stroke);
      border-radius:12px;
      background:#0b0f14;
      padding:12px;
    }}
    .zoom-pane h3 {{ margin:0 0 8px; font-size:13px; }}
    .zoom-viewport {{
      position:relative;
      overflow:hidden;
      border-radius:10px;
      background:#000;
      aspect-ratio: 420 / 316;
    }}
    .zoom-viewport video {{
      position:absolute;
      inset:0;
      width:100%;
      height:100%;
      object-fit: fill;
      transform-origin: top left;
      will-change: transform;
    }}
    .run-panel {{ display:none; }}
    .run-panel.is-active {{ display:block; }}
    .explorer-meta {{ margin-top: 12px; color: var(--muted); font-size: 12px; line-height: 1.6; }}
    .compare-grid {{ display:grid; grid-template-columns: 1fr 1fr; gap:16px; margin-top: 16px; }}
    .compare-pane {{ border:1px solid var(--stroke); border-radius:10px; padding:10px; background:#0b0f14; }}
    .compare-pane h3 {{ margin: 0 0 8px; font-size: 13px; letter-spacing: -0.01em; }}
    .compare-pane video {{ width:100%; display:block; border-radius:8px; background:#000; }}
    .compare-pane p {{ margin: 8px 0 0; color: var(--muted); font-size: 12px; }}
    .compare-mode {{ display:none; }}
    .compare-mode.is-active {{ display:block; }}
    .delta-compare {{ display:grid; gap: 16px; }}
    .delta-head {{ display:grid; grid-template-columns: 1fr auto 1fr; gap: 14px; align-items: stretch; }}
    .delta-card {{ border: 1px solid var(--stroke); border-radius: 10px; padding: 14px 16px; background: #0b0f14; }}
    .delta-kicker {{ color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: 0.08em; }}
    .delta-scoreline {{ display:flex; align-items:baseline; gap: 12px; margin-top: 8px; flex-wrap: wrap; }}
    .delta-score {{ font-size: 38px; line-height: 0.95; font-weight: 760; letter-spacing: -0.05em; }}
    .delta-bytes {{ color: var(--muted); font-size: 13px; }}
    .delta-note {{ margin-top: 10px; color: var(--muted); font-size: 12px; line-height: 1.55; }}
    .delta-arrow {{ display:grid; place-items:center; color: var(--accent); font-size: 28px; }}
    .delta-table td:nth-child(3) {{ color: var(--accent); }}
    .delta-negative {{ color: #2ad4a0; }}
    .delta-positive {{ color: #ff6b6b; }}
    .footer-meta {{ display:flex; justify-content:space-between; gap: 16px; flex-wrap:wrap; margin-top: 18px; padding-top: 16px; border-top: 1px solid var(--stroke); color: var(--muted); font-size: 12px; }}
    .footer-meta a {{ text-decoration: none; }}
    @media (max-width: 980px) {{ .hero-grid, .hero-card-grid, .concept-grid, .two {{ grid-template-columns: 1fr; }} .context-strip, .summary-grid {{ grid-template-columns: 1fr; }} .context-item + .context-item, .summary-stat + .summary-stat {{ border-left: 0; border-top: 1px solid var(--stroke); }} }}
    @media (max-width: 720px) {{ .hero-grid, .hero-card-grid, .concept-grid, .two, .context-strip, .summary-grid, .compare-grid, .delta-head, .explorer-grid, .zoom-grid {{ grid-template-columns: 1fr; }} .context-item + .context-item, .summary-stat + .summary-stat {{ border-left: 0; border-top: 1px solid var(--stroke); }} .wrap {{ padding-inline: 16px; }} h1 {{ font-size: clamp(36px, 15vw, 64px); max-width: none; }} .panel {{ padding: 16px; }} .panel h2 {{ font-size: 18px; }} .walkthrough li {{ grid-template-columns: 24px 1fr; gap: 12px; padding: 12px 0; }} .compare-controls input[type="range"] {{ width: 100%; }} .delta-arrow {{ display:none; }} .refs-group + .refs-group {{ border-left: 0; border-top: 1px solid var(--stroke); padding-left: 0; padding-top: 16px; }} }}
    @media (prefers-reduced-motion: reduce) {{ html {{ scroll-behavior: auto; }} * {{ animation: none !important; transition: none !important; }} }}
  </style>
</head>
<body>
<a class="skip" href="#content">Skip to content</a>
<div class="wrap">
    <header class="hero" aria-labelledby="site-title">
      <div class="eyebrow">comma-lab</div>
      <div class="hero-grid">
        <div class="hero-stack">
          <h1 id="site-title">A lab notebook for compression that machines can trust and people can follow.</h1>
          <p class="lede">This site tracks scorer-backed experiments for the comma.ai video compression challenge. It is designed to work at two levels: a plain-English executive layer for judges and newcomers, and a deeper evidence trail for researchers who want the exact runs, artifacts, and failure cases.</p>
          <div class="pill-row" aria-label="Current highlights">
            <span class="pill">Current floor: {data['summary']['best_track_b_score']:.2f}</span>
            <span class="pill">Rule-faithful: {best.get('rule_faithful_score', 0):.3f}</span>
            <span class="pill">{data['summary']['robust_run_count']} measured runs</span>
          </div>
          <nav class="subnav" aria-label="Primary resources">
            <a href="./judges_one_pager.md">Judges one-pager</a>
            <a href="./submission_packet.md">Submission packet</a>
            <a href="./lab_notebook.md">Lab notebook</a>
            <a href="./promotion_review_latest.md">Promotion review</a>
            <a href="./evidence_index.md">Evidence index</a>
          </nav>
        </div>
        <div class="hero-card-grid" aria-label="Entry points">
          <article class="hero-card">
            <h2>Start here</h2>
            <p>Read this page first if you want the outcome, the score, and the plain-English explanation of what is being optimized.</p>
            <div class="hero-links">
              <a href="#concepts">What the score means</a>
              <a href="#compare-title">Visual proof</a>
            </div>
          </article>
          <article class="hero-card">
            <h2>For judges</h2>
            <p>Jump to the compact packet and the promotion review if you want the measured result, the evidence root, and the short argument for why it matters.</p>
            <div class="hero-links">
              <a href="./submission_packet.md">Submission packet</a>
              <a href="./judges_one_pager.md">Judges one-pager</a>
            </div>
          </article>
          <article class="hero-card">
            <h2>For researchers</h2>
            <p>Use the notebook, manifests, and raw evidence if you want the experimental path, the exact artifacts, and the rejected branches too.</p>
            <div class="hero-links">
              <a href="./lab_notebook.md">Lab notebook</a>
              <a href="./experiment_manifest.json">Experiment manifest</a>
            </div>
          </article>
        </div>
      </div>
    </header>

  <main id="content">
    <section class="context-strip" aria-label="Project context">
      <div class="context-item">
        <div class="context-label">Contest</div>
        <p class="context-copy">comma.ai’s public challenge asks entrants to ship an <code>archive.zip</code> that inflates to video. The published score combines archive bytes, SegNet distortion, and PoseNet distortion on the public test clip.</p>
        <div class="context-links">
          <a href="{data['site_meta']['challenge_url']}">Official challenge repo</a>
        </div>
      </div>
      <div class="context-item">
        <div class="context-label">Who we are</div>
        <p class="context-copy">comma-lab is a public experiment log and submission repo maintained by {escape(data['site_meta']['maintainer'])}. It publishes measured runs, rejected branches, and the current promoted operating point in one place.</p>
        <div class="context-links">
          <a href="{data['site_meta']['github_url']}">{escape(data['site_meta']['repo_slug'])}</a>
          <a href="./experiment_manifest.json">Experiment manifest</a>
        </div>
      </div>
      <div class="context-item">
        <div class="context-label">Last updated</div>
        <p class="context-copy">{escape(data['site_meta']['updated_at_local'])}</p>
        <p class="context-meta">Generated from repository state plus scorer-backed artifacts stored in this repo.</p>
      </div>
    </section>

    <section class="summary-shell" aria-label="Current state">
      <div class="summary-grid">
        <div class="summary-stat"><div class="summary-label">Track B current_workflow</div><div class="summary-value">{data['summary']['best_track_b_score']:.2f}</div><div class="summary-meta">{data['summary']['best_track_b_bytes']:,} bytes</div></div>
        <div class="summary-stat"><div class="summary-label">Track B rule_faithful</div><div class="summary-value">{best.get('rule_faithful_score', 0):.3f}</div><div class="summary-meta">{fmt_int(best.get('rule_faithful_bundle_bytes'))} bytes</div></div>
        <div class="summary-stat"><div class="summary-label">Delta vs published baseline</div><div class="summary-value">{published_delta_display}</div><div class="summary-meta">{published_baseline_display} → {best['current_workflow_score']:.2f}</div></div>
        <div class="summary-stat"><div class="summary-label">Delta vs prior floor</div><div class="summary-value">{best_delta_score:+.2f}</div><div class="summary-meta">{best_delta_bytes:+,} bytes</div></div>
      </div>
      <p class="footnote">Track A remains separate: `current_workflow` 0.00 at 167 bytes. The robust run ledger currently contains {data['summary']['robust_run_count']} measured runs, {data['summary']['promotion_count']} promotions, and {data['summary']['rejection_count']} explicit rejections.</p>
      <p class="config-line" aria-label="Current promoted method">{escape(config_line)}</p>
    </section>

    <section class="panel" id="concepts" aria-labelledby="concepts-title">
      <h2 id="concepts-title">What the score means</h2>
      <p class="muted">The challenge score mixes filesize and task distortion. Lower is better, but the score is not a generic visual metric. It rewards preserving the signals that the scorer models actually use.</p>
      <div class="concept-grid">
        <article class="concept-card">
          <h3>{escape(concept_cards[0]['title'])}</h3>
          <p>{escape(concept_cards[0]['summary'])}</p>
        </article>
        <article class="concept-card">
          <h3>{escape(concept_cards[1]['title'])}</h3>
          <p>{escape(concept_cards[1]['summary'])}</p>
        </article>
        <article class="concept-card">
          <h3>{escape(concept_cards[2]['title'])}</h3>
          <p>{escape(concept_cards[2]['summary'])}</p>
        </article>
      </div>
    </section>

    <section class="panel" aria-labelledby="scatter-title">
      <h2 id="scatter-title">Score vs bytes</h2>
      <p class="muted">Track B runs only. Better runs move toward the lower-left. The x-axis uses log scaling. The y-axis is linear. The severe AV1 bug run at 97.45 is omitted here so the operating range stays legible; it remains documented in the search-path section.</p>
      <div class="compare-controls" aria-label="Scatter view controls">
        <button type="button" class="is-active" data-scatter-mode="full">full range</button>
        <button type="button" data-scatter-mode="focus">operating range</button>
      </div>
      <div class="compare-mode is-active" data-scatter-panel="full">
        <div class="scatter-shell" data-scatter-shell>
          {full_scatter_svg}
          <div class="scatter-tooltip" data-scatter-tooltip><strong></strong><span></span><span></span></div>
        </div>
      </div>
      <div class="compare-mode" data-scatter-panel="focus">
        <div class="scatter-shell" data-scatter-shell>
          {focus_scatter_svg}
          <div class="scatter-tooltip" data-scatter-tooltip><strong></strong><span></span><span></span></div>
        </div>
      </div>
      <div class="legend-row" aria-label="Scatter legend">
        <span class="legend-key"><span class="legend-dot" style="background:#2ad4a0"></span>promotion</span>
        <span class="legend-key"><span class="legend-dot" style="background:#ff6b6b"></span>explicit rejection</span>
        <span class="legend-key"><span class="legend-dot" style="background:#8ab4ff"></span>measured run</span>
      </div>
      <p class="footnote">Lower is better. The plot is generated directly from scorer-backed artifacts in the repository.</p>
    </section>

    <section class="panel" aria-labelledby="lineage-title">
      <h2 id="lineage-title">Trajectory and branch points</h2>
      <p class="muted">The x-axis is turning-point order. The y-axis is actual current_workflow score. Lower is better. The AV1 failure is called out as an off-scale diagnostic spike rather than squeezed into the honest operating range.</p>
      <div class="legend-row" aria-label="Lineage legend">
        <span class="legend-key"><span class="legend-line" style="color:#8ed1c0"></span>mainline improvements</span>
        <span class="legend-key"><span class="legend-line" style="color:#ff8f8f"></span>rejected side branches</span>
        <span class="legend-key"><span class="legend-line" style="color:#f0b35f"></span>diagnostic off-scale event</span>
      </div>
      <div class="lineage-wrap">
        <svg viewBox="0 0 1180 360" role="img" aria-label="Scaled milestone trajectory with branch points and an off-scale AV1 diagnostic failure">
          {''.join(trajectory_grid_svg)}
          <line x1="{traj_pad_left}" y1="{trajectory_height - traj_pad_bottom}" x2="{trajectory_width - traj_pad_right}" y2="{trajectory_height - traj_pad_bottom}" stroke="#475569" />
          <line x1="{traj_pad_left}" y1="{traj_pad_top}" x2="{traj_pad_left}" y2="{trajectory_height - traj_pad_bottom}" stroke="#475569" />
          <path class="trajectory-edge trajectory-edge-main" d="{trajectory_path}" />
          {''.join(trajectory_branch_svg)}
          {''.join(trajectory_nodes_svg)}
          <text x="{trajectory_width/2:.1f}" y="{trajectory_height - 10}" text-anchor="middle" class="axis-label">measured turning-point order</text>
          <text x="16" y="{trajectory_height/2:.1f}" transform="rotate(-90 16 {trajectory_height/2:.1f})" text-anchor="middle" class="axis-label">actual current_workflow score</text>
        </svg>
      </div>
      <p class="footnote">This chart is selective by design. It shows the milestones that changed the operating point or changed the lab’s understanding of the evaluator.</p>
    </section>

    <section class="panel" aria-labelledby="compare-title">
      <div class="compare-head">
        <div>
          <h2 id="compare-title">Top runs explorer</h2>
          <p class="muted">Compare the strongest scorer-backed internal runs against the source clip. drag to move the zoom window, change its size, and keep every pane frame-synced.</p>
          <p class="explorer-meta">Internal media explorer last refreshed {escape(compare_updated_label)}. Public entries below are reference metadata only because we do not have their artifacts or synced media. Official reference: <a href="https://comma.ai/leaderboard">comma.ai leaderboard</a> and the public PR stream in the official challenge repo.</p>
        </div>
        <div class="leaderboard-band" aria-label="Unofficial community leaderboard snapshot">
          <div class="leaderboard-label">Unofficial community leaderboard snapshot</div>
          {leaderboard_refs_html}
        </div>
      </div>
      <div class="explorer-toolbar">
        <div class="run-picker" data-run-picker aria-label="Top run picker">
          {compare_buttons_html}
        </div>
        <div class="compare-controls" aria-label="Explorer controls">
          <button type="button" data-compare-toggle>play / pause</button>
          <input type="range" min="0" max="100" value="0" step="0.1" data-compare-scrub aria-label="Comparison scrubber" />
          <input type="range" min="16" max="36" value="22" step="1" data-zoom-size aria-label="Zoom window size" />
        </div>
      </div>
      <div class="explorer-grid">
        <figure class="explorer-stage">
          <h3>Original</h3>
          <div class="zoom-stage" data-zoom-stage>
            <video data-compare-video data-run-id="original" preload="metadata" muted playsinline poster="./media/original_poster.jpg" src="./media/original_preview.mp4"></video>
            <div class="zoom-window" data-zoom-window></div>
          </div>
          <p class="explorer-meta">Drag to move the zoom window.</p>
        </figure>
        <figure class="explorer-stage">
          <h3 data-selected-run-title>{escape(current_variant["label"])}</h3>
          <div class="zoom-stage">
            {compare_stage_html}
          </div>
          <p class="explorer-meta" data-selected-run-note>{escape(current_variant["note"])}</p>
        </figure>
      </div>
      <div class="zoom-grid">
        <figure class="zoom-pane">
          <h3>Original zoom</h3>
          <div class="zoom-viewport" data-zoom-viewport>
            <video data-compare-video data-run-id="original" preload="metadata" muted playsinline poster="./media/original_poster.jpg" src="./media/original_preview.mp4"></video>
          </div>
        </figure>
        <figure class="zoom-pane">
          <h3 data-selected-run-title>{escape(current_variant["label"])} zoom</h3>
          {compare_zoom_html}
        </figure>
      </div>
      <p class="footnote" data-selected-run-metrics>Selected run: {current_variant["label"]} · score {current_variant["score"]:.2f} · {current_variant["note"]}.</p>
    </section>

    <section class="panel" aria-labelledby="mechanism-title">
      <h2 id="mechanism-title">How to read this lab</h2>
      <ol class="walkthrough">
        <li>
          <div class="step-num"></div>
          <div class="step-body">
            <h3>Evaluator path</h3>
            <p>{escape(hypothesis_cards[0]['summary'])}</p>
            <p>{escape(hypothesis_cards[0]['detail'])}</p>
          </div>
        </li>
        <li>
          <div class="step-num"></div>
          <div class="step-body">
            <h3>Critical bug</h3>
            <p>{escape(hypothesis_cards[1]['summary'])}</p>
            <p>{escape(hypothesis_cards[1]['detail'])}</p>
          </div>
        </li>
        <li>
          <div class="step-num"></div>
          <div class="step-body">
            <h3>Current operating point</h3>
            <p>{escape(hypothesis_cards[2]['summary'])}</p>
            <p>{escape(hypothesis_cards[2]['detail'])}</p>
          </div>
        </li>
      </ol>
    </section>

    <section class="panel" aria-labelledby="explain-title">
      <h2 id="explain-title">Why {best["current_workflow_score"]:.2f} beat {prev_best["current_workflow_score"]:.2f}</h2>
      <div class="delta-compare">
        <div class="delta-head" aria-label="Comparison between the prior floor and the current promoted floor">
          <div class="delta-card">
            <div class="delta-kicker">prior floor</div>
            <div class="delta-scoreline">
              <span class="delta-score">{prev_best["current_workflow_score"]:.2f}</span>
              <span class="delta-bytes">{prev_best["archive_bytes"]:,} bytes</span>
            </div>
            <div class="delta-note">{escape(prev_best_note)}</div>
          </div>
          <div class="delta-arrow" aria-hidden="true">→</div>
          <div class="delta-card">
            <div class="delta-kicker">current floor</div>
            <div class="delta-scoreline">
              <span class="delta-score">{best["current_workflow_score"]:.2f}</span>
              <span class="delta-bytes">{best["archive_bytes"]:,} bytes</span>
            </div>
            <div class="delta-note">{escape(best_note)}</div>
          </div>
        </div>
        <div class="table-wrap">
          <table class="delta-table">
            <thead><tr><th scope="col">Metric</th><th scope="col">Prior</th><th scope="col">Current</th><th scope="col">Delta</th></tr></thead>
            <tbody>{''.join(delta_rows_html)}</tbody>
          </table>
        </div>
      </div>
      <p class="footnote">The new floor kept bytes in the same regime while lowering the distortion that mattered most in this comparison.</p>
    </section>

    <section class="panel" aria-labelledby="frontier-title">
      <h2 id="frontier-title">Local neighborhood</h2>
      <p class="frontier-callout">This table isolates the local AV1 neighborhood around the promoted floor. It shows which nearby changes improved the result and which did not.</p>
      <div class="table-wrap">
        <table>
          <thead><tr><th scope="col">Variant</th><th scope="col">Changed axis</th><th scope="col">Score</th><th scope="col">Bytes</th><th scope="col">Δ score</th><th scope="col">Δ bytes</th><th scope="col">Verdict</th></tr></thead>
          <tbody>{''.join(frontier_rows)}</tbody>
        </table>
      </div>
      <p class="table-hint">On narrow screens, swipe horizontally to inspect the full table.</p>
    </section>

    <section class="panel" aria-labelledby="references-title">
      <h2 id="references-title">References</h2>
      <div class="two refs-grid">
        <div class="refs-group">
          <p class="refs-group-title">Primary artifacts</p>
          <ul class="list-tight">
            <li><a href="./judges_one_pager.md">Judges one-pager</a></li>
            <li><a href="./submission_packet.md">Submission packet</a></li>
            <li><a href="./promotion_review_latest.md">Promotion review</a></li>
            <li><a href="./promotion_accounting.md">Promotion accounting</a></li>
            <li><a href="./code_callouts.md">Code callouts</a></li>
            <li><a href="./experiment_manifest.json">Experiment manifest</a></li>
            <li><a href="./ffmpeg_path_review.md">ffmpeg path review</a></li>
            <li><a href="./experiment_journal.md">Experiment journal</a></li>
            <li><a href="./evidence_index.md">Evidence index</a></li>
          </ul>
        </div>
        <div class="refs-group">
          <p class="refs-group-title">Turning points</p>
          <ul class="list-tight">{''.join(turning_items)}</ul>
        </div>
      </div>
      <p class="footnote refs-note">The landing page is the guided entry point. Full depth lives in the linked notebook, packet, and raw evidence artifacts.</p>
    </section>
    <footer class="footer-meta" aria-label="Site metadata">
      <span>Last updated {escape(data['site_meta']['updated_at_local'])}</span>
      <span><a href="{data['site_meta']['github_url']}">GitHub</a> · <a href="{data['site_meta']['challenge_url']}">Challenge</a></span>
    </footer>
  </main>
</div>
<script>
(() => {{
  const scatterModeButtons = Array.from(document.querySelectorAll('[data-scatter-mode]'));
  const scatterPanels = Array.from(document.querySelectorAll('[data-scatter-panel]'));
  const videos = Array.from(document.querySelectorAll('[data-compare-video]'));
  const toggle = document.querySelector('[data-compare-toggle]');
  const scrub = document.querySelector('[data-compare-scrub]');
  const zoomSize = document.querySelector('[data-zoom-size]');
  const runButtons = Array.from(document.querySelectorAll('[data-run-button]'));
  const runPanels = Array.from(document.querySelectorAll('[data-run-panel]'));
  const runZoomPanels = Array.from(document.querySelectorAll('[data-run-zoom-panel]'));
  const zoomStage = document.querySelector('[data-zoom-stage]');
  const zoomWindow = document.querySelector('[data-zoom-window]');
  const selectedTitles = Array.from(document.querySelectorAll('[data-selected-run-title]'));
  const selectedNote = document.querySelector('[data-selected-run-note]');
  const selectedMetrics = document.querySelector('[data-selected-run-metrics]');
  const cropMeta = {json.dumps(compare_manifest.get("zoom_crop") or {"x": ZOOM_X, "y": ZOOM_Y, "w": ZOOM_W, "h": ZOOM_H})};
  const sourceMeta = {json.dumps(compare_manifest.get("source_size") or {"w": 1164, "h": 874})};
  const previewMeta = {json.dumps(compare_manifest.get("preview_size") or {"w": 960, "h": 720})};
  const variantMeta = {json.dumps(compare_variants)};
  let activeRun = runButtons[0]?.getAttribute('data-run-id') || variantMeta[0]?.id || null;
  let zoomWidth = Number(zoomSize?.value || 22) / 100;
  const zoomAspect = cropMeta.w / cropMeta.h;
  let zoomHeight = zoomWidth / zoomAspect;
  let zoomX = cropMeta.x / sourceMeta.w;
  let zoomY = cropMeta.y / sourceMeta.h;
  let dragState = null;

  const activeVideos = () => videos.filter(v => {{
    const runId = v.getAttribute('data-run-id');
    return runId === 'original' || runId === activeRun;
  }});
  const currentDuration = () => activeVideos()[0]?.duration || 0;
  const anyActivePlaying = () => activeVideos().some(v => !v.paused && !v.ended);

  const setScrubFromTime = (time) => {{
    if (!scrub) return;
    const duration = currentDuration();
    scrub.value = duration > 0 ? String((Math.min(time, duration) / duration) * 100) : '0';
  }};

  const pauseVideos = (list) => {{
    list.forEach(v => v.pause());
  }};

  const syncCurrentTime = (time, list = videos) => {{
    list.forEach(v => {{
      if (!Number.isNaN(v.duration) && v.duration > 0) {{
        v.currentTime = Math.min(time, v.duration);
      }}
    }});
  }};

  const updateToggleLabel = () => {{
    if (!toggle) return;
    const playing = anyActivePlaying();
    toggle.textContent = playing ? 'pause' : 'play';
    toggle.setAttribute('aria-pressed', playing ? 'true' : 'false');
  }};

  const playVideos = async (list) => {{
    await Promise.all(list.map(v => v.play().catch(() => null)));
  }};

  const clamp = (value, min, max) => Math.max(min, Math.min(max, value));

  const updateZoomGeometry = () => {{
    if (!zoomWindow || !zoomStage) return;
    zoomHeight = clamp(zoomWidth / zoomAspect, 0.12, 0.72);
    zoomX = clamp(zoomX, 0, 1 - zoomWidth);
    zoomY = clamp(zoomY, 0, 1 - zoomHeight);
    zoomWindow.style.left = `${{zoomX * 100}}%`;
    zoomWindow.style.top = `${{zoomY * 100}}%`;
    zoomWindow.style.width = `${{zoomWidth * 100}}%`;
    zoomWindow.style.height = `${{zoomHeight * 100}}%`;
    document.querySelectorAll('[data-zoom-viewport] video').forEach(video => {{
      video.style.width = `${{100 / zoomWidth}}%`;
      video.style.height = `${{100 / zoomHeight}}%`;
      video.style.transform = `translate(-${{(zoomX * 100) / zoomWidth}}%, -${{(zoomY * 100) / zoomHeight}}%)`;
    }});
  }};

  const activateRun = (runId) => {{
    activeRun = runId;
    const variant = variantMeta.find(v => v.id === runId);
    runButtons.forEach(btn => btn.classList.toggle('is-active', btn.getAttribute('data-run-id') === runId));
    runPanels.forEach(panel => panel.classList.toggle('is-active', panel.getAttribute('data-run-panel') === runId));
    runZoomPanels.forEach(panel => panel.classList.toggle('is-active', panel.getAttribute('data-run-zoom-panel') === runId));
    if (variant) {{
      selectedTitles.forEach(el => {{
        el.textContent = el.textContent?.includes('zoom') ? `${{variant.label}} zoom` : variant.label;
      }});
      if (selectedNote) selectedNote.textContent = variant.note;
      if (selectedMetrics) selectedMetrics.textContent = `Selected run: ${{variant.label}} · score ${{variant.score.toFixed(2)}} · ${{variant.note}}.`;
    }}
  }};

  runButtons.forEach(btn => {{
    btn.addEventListener('click', () => activateRun(btn.getAttribute('data-run-id')));
  }});

  scatterModeButtons.forEach(btn => {{
    btn.addEventListener('click', () => {{
      const mode = btn.getAttribute('data-scatter-mode');
      scatterModeButtons.forEach(b => b.classList.toggle('is-active', b === btn));
      scatterPanels.forEach(panel => panel.classList.toggle('is-active', panel.getAttribute('data-scatter-panel') === mode));
    }});
  }});

  if (toggle) {{
    toggle.addEventListener('click', async () => {{
      const active = activeVideos();
      if (!active.length) return;
      const shouldPlay = active.every(v => v.paused || v.ended);
      if (shouldPlay) {{
        await playVideos(active);
      }} else {{
        pauseVideos(active);
      }}
      updateToggleLabel();
    }});
  }}

  if (scrub) {{
    scrub.addEventListener('input', async () => {{
      const duration = currentDuration();
      if (!duration) return;
      const shouldResume = anyActivePlaying();
      const t = duration * (Number(scrub.value) / 100);
      pauseVideos(videos);
      syncCurrentTime(t, videos);
      if (shouldResume) {{
        await playVideos(activeVideos());
      }}
      updateToggleLabel();
    }});
  }}

  if (zoomSize) {{
    zoomSize.addEventListener('input', () => {{
      zoomWidth = Number(zoomSize.value) / 100;
      updateZoomGeometry();
    }});
  }}

  if (zoomStage && zoomWindow) {{
    const updateFromPointer = (event) => {{
      const rect = zoomStage.getBoundingClientRect();
      const x = clamp((event.clientX - rect.left) / rect.width - zoomWidth / 2, 0, 1 - zoomWidth);
      const y = clamp((event.clientY - rect.top) / rect.height - zoomHeight / 2, 0, 1 - zoomHeight);
      zoomX = x;
      zoomY = y;
      updateZoomGeometry();
    }};
    zoomWindow.addEventListener('pointerdown', (event) => {{
      dragState = true;
      zoomWindow.classList.add('is-dragging');
      zoomWindow.setPointerCapture(event.pointerId);
      updateFromPointer(event);
    }});
    zoomStage.addEventListener('pointermove', (event) => {{
      if (!dragState) return;
      updateFromPointer(event);
    }});
    zoomStage.addEventListener('pointerup', () => {{
      dragState = null;
      zoomWindow.classList.remove('is-dragging');
    }});
    zoomStage.addEventListener('pointerleave', () => {{
      dragState = null;
      zoomWindow.classList.remove('is-dragging');
    }});
  }}

  videos.forEach(video => {{
    video.addEventListener('timeupdate', () => {{
      const active = activeVideos();
      if (!active.includes(video) || !scrub || !video.duration) return;
      setScrubFromTime(video.currentTime);
      videos.forEach(other => {{
        if (other !== video && Math.abs(other.currentTime - video.currentTime) > 0.08) {{
          other.currentTime = video.currentTime;
        }}
      }});
    }});
    video.addEventListener('play', updateToggleLabel);
    video.addEventListener('pause', updateToggleLabel);
    video.addEventListener('ended', updateToggleLabel);
  }});

  document.querySelectorAll('[data-scatter-shell]').forEach(shell => {{
    const tooltip = shell.querySelector('[data-scatter-tooltip]');
    if (!tooltip) return;
    const [title, meta, detail] = tooltip.querySelectorAll('strong, span');
    const hide = () => tooltip.classList.remove('is-visible');
    const show = (point) => {{
      title.textContent = point.getAttribute('data-tip-label') || '';
      meta.textContent = point.getAttribute('data-tip-meta') || '';
      detail.textContent = point.getAttribute('data-tip-detail') || '';
      const shellRect = shell.getBoundingClientRect();
      const pointRect = point.getBoundingClientRect();
      const tooltipWidth = Math.min(320, Math.max(220, shellRect.width - 28));
      const left = Math.min(
        shellRect.width - tooltipWidth - 14,
        Math.max(14, pointRect.left - shellRect.left - tooltipWidth / 2)
      );
      const preferAbove = pointRect.top - shellRect.top > 82;
      const top = preferAbove
        ? pointRect.top - shellRect.top - 74
        : pointRect.bottom - shellRect.top + 12;
      tooltip.style.setProperty('--tip-x', `${{left}}px`);
      tooltip.style.setProperty('--tip-y', `${{Math.max(14, top)}}px`);
      tooltip.classList.add('is-visible');
    }};

    shell.querySelectorAll('[data-scatter-point]').forEach(point => {{
      point.addEventListener('mouseenter', () => show(point));
      point.addEventListener('focus', () => show(point));
      point.addEventListener('mouseleave', hide);
      point.addEventListener('blur', hide);
    }});
  }});

  activateRun(activeRun);
  updateZoomGeometry();
  updateToggleLabel();
}})();
</script>
</body>
</html>
'''
    OUT_HTML.write_text(html)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
