#!/usr/bin/env python3
from __future__ import annotations

import json
import math
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / 'reports' / 'results.jsonl'
TIMELINE = ROOT / 'reports' / 'timeline.jsonl'
OUT_JSON = ROOT / 'reports' / 'graphs' / 'dashboard_data.json'
OUT_HTML = ROOT / 'reports' / 'graphs' / 'index.html'
OUT_TIMELINE = ROOT / 'reports' / 'graphs' / 'score_timeline.json'
OUT_GRAPH = ROOT / 'reports' / 'graphs' / 'experiment_graph.json'
CANONICAL_URL = 'https://comma-lab.pages.dev/'


def load_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


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


def main() -> int:
    results = load_jsonl(RESULTS)
    timeline = load_jsonl(TIMELINE)
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
        {'label': 'First honest floor', 'run_id': 'robust_current-baseline-cpu-2026-04-03', 'score': 4.06},
        {'label': 'First big win', 'run_id': 'robust_current-medium23-cpu-2026-04-03', 'score': 3.62},
        {'label': 'ROI failure', 'run_id': 'robust_current-dynamic-main-roi-cpu-2026-04-05', 'score': 4.47},
        {'label': 'Current best floor', 'run_id': best['run_id'], 'score': best['current_workflow_score']},
    ]

    local_frontier_run_ids = [
        'robust_current-av1-524x394-rgb24-promoted-cpu-2026-04-05',
        'robust_current-av1-524x394-crf34-promoted-cpu-2026-04-06',
        'robust_current-av1-524x394-crf35-rejected-cpu-2026-04-06',
        'robust_current-av1-524x394-unsharp030-rejected-cpu-2026-04-06',
        'robust_current-av1-524x394-filmgrain0-rejected-cpu-2026-04-06',
        'robust_current-av1-522x392-rejected-cpu-2026-04-06',
        'robust_current-av1-524x394-upscale-lanczos-promoted-cpu-2026-04-06',
        'robust_current-av1-524x394-colorspace-hardening-promoted-cpu-2026-04-06',
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
                'notes': r.get('notes', []),
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
            'The x265 ladder still matters as evidence: it established a 3.25 local floor before the repaired AV1 branch advanced through 2.20, 2.19, 2.18, and 2.12.',
        ],
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

    if last_x265_promo:
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

    robust_only = [r for r in data['results'] if r['track'] == 'robust_current']
    scatter_runs = [r for r in robust_only if r['score'] <= 10.0]
    omitted_outliers = [r for r in robust_only if r['score'] > 10.0]
    min_score = min(r['score'] for r in scatter_runs)
    max_score = max(r['score'] for r in scatter_runs)
    min_bytes = min(r['archive_bytes'] for r in scatter_runs)
    max_bytes = max(r['archive_bytes'] for r in scatter_runs)
    log_min_score = math.log10(min_score)
    log_max_score = math.log10(max_score)
    log_min_bytes = math.log10(min_bytes)
    log_max_bytes = math.log10(max_bytes)

    width = 920
    height = 420
    pad = 52

    def sx(x: float) -> float:
        if log_max_bytes == log_min_bytes:
            return width / 2
        return pad + (math.log10(x) - log_min_bytes) * (width - 2 * pad) / (log_max_bytes - log_min_bytes)

    plot_max_score = min(max_score, 6.0)

    def sy(y: float) -> float:
        if plot_max_score == min_score:
            return height / 2
        return height - pad - (y - min_score) * (height - 2 * pad) / (plot_max_score - min_score)

    def fmt_axis_bytes(value: int) -> str:
        if value >= 1_000_000:
            return f'{value / 1_000_000:.1f}M'
        return f'{value / 1_000:.0f}k'

    x_ticks = [v for v in [700_000, 900_000, 1_200_000, 1_700_000, 2_500_000, 4_000_000, 5_000_000] if min_bytes <= v <= max_bytes]
    y_ticks = [v for v in [2.0, 2.5, 3.0, 4.0, 5.0, 6.0] if min_score <= v <= plot_max_score]
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
    for r in scatter_runs:
        color = '#2ad4a0' if r['is_promotion'] else '#ff6b6b' if any('reject' in n.lower() for n in r['notes']) else '#8ab4ff'
        label = escape(f"{r['run_id']} | score={r['score']} | bytes={r['archive_bytes']}")
        points.append(f'<circle cx="{sx(r["archive_bytes"]):.1f}" cy="{sy(r["score"]):.1f}" r="5.5" fill="{color}"><title>{label}</title></circle>')

    timeline_rows = []
    for e in notable_events[-14:]:
        timeline_rows.append(
            f"<tr><td>{escape(e.get('ts_utc',''))}</td><td>{escape(e.get('type',''))}</td><td>{escape(e.get('summary',''))}</td></tr>"
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
    for t in reversed(key_turning_points):
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
            axis = 'geometry 522x392'
        elif 'upscale-lanczos' in run_id:
            axis = 'upscale lanczos'
        elif 'colorspace-hardening' in run_id:
            axis = 'explicit bt709/tv -> rgb24(pc)'
        elif 'filmgrain0' in run_id or cfg.get('film_grain') == 0:
            axis = f"film-grain {cfg.get('film_grain')}"
        elif 'unsharp030' in run_id:
            axis = f"postfilter {cfg['inflate_postfilter']}"
        elif cfg.get('crf') is not None:
            axis = f"crf {cfg['crf']}"
        elif cfg.get('inflate_postfilter'):
            axis = f"postfilter {cfg['inflate_postfilter']}"
        else:
            axis = describe_config(cfg)
        verdict = 'promoted' if 'promoted' in r['run_id'] else 'rejected'
        frontier_rows.append(
            '<tr>'
            f"<td>{escape(r['run_id'])}</td>"
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

    lineage_nodes = []
    lineage_edges = []

    def add_lineage_node(run_id: str, label: str, family: str, x: int, y: int) -> None:
        run = run_index.get(run_id)
        if not run:
            return
        lineage_nodes.append({
            'run_id': run_id,
            'label': label,
            'family': family,
            'x': x,
            'y': y,
            'score': run['current_workflow_score'],
            'bytes': run['archive_bytes'],
            'config': run.get('config', {}),
        })

    def add_lineage_edge(source: str, target: str, kind: str) -> None:
        if source in run_index and target in run_index:
            lineage_edges.append({'source': source, 'target': target, 'kind': kind})

    current_best_id = best['run_id']
    x265_floor_id = 'robust_current-cand-424x318-crf23-g48-b4-r4-cpu-2026-04-05'
    av1_bug_id = 'robust_current-av1-524x394-cpu-2026-04-05'
    roi_fail_id = 'robust_current-dynamic-main-roi-cpu-2026-04-05'
    nearby_reject_id = 'robust_current-cand-428x320-crf23-g48-b4-r4-cpu-2026-04-05'

    add_lineage_node('robust_current-baseline-cpu-2026-04-03', 'baseline', 'x265', 90, 270)
    add_lineage_node('robust_current-medium23-cpu-2026-04-03', '512x384 / crf23', 'x265', 240, 238)
    add_lineage_node('robust_current-448x336-medium23-cpu-2026-04-03', '448x336', 'x265', 390, 218)
    add_lineage_node('robust_current-lanczos-lanczos-cpu-2026-04-04', 'lanczos/lanczos', 'x265', 535, 205)
    add_lineage_node('robust_current-432x324-cpu-2026-04-04', '432x324', 'x265', 675, 184)
    add_lineage_node(x265_floor_id, 'x265 floor', 'x265', 820, 165)
    add_lineage_node(roi_fail_id, 'dynamic ROI', 'roi', 675, 78)
    add_lineage_node(nearby_reject_id, 'nearby scale', 'rejection', 820, 285)
    add_lineage_node(av1_bug_id, 'AV1 failure', 'bug', 965, 78)
    if current_best_id not in {n['run_id'] for n in lineage_nodes}:
        add_lineage_node(current_best_id, 'current floor', 'best', 1085, 160 if current_best_id != av1_bug_id else 118)

    add_lineage_edge('robust_current-baseline-cpu-2026-04-03', 'robust_current-medium23-cpu-2026-04-03', 'promotion')
    add_lineage_edge('robust_current-medium23-cpu-2026-04-03', 'robust_current-448x336-medium23-cpu-2026-04-03', 'promotion')
    add_lineage_edge('robust_current-448x336-medium23-cpu-2026-04-03', 'robust_current-lanczos-lanczos-cpu-2026-04-04', 'promotion')
    add_lineage_edge('robust_current-lanczos-lanczos-cpu-2026-04-04', 'robust_current-432x324-cpu-2026-04-04', 'promotion')
    add_lineage_edge('robust_current-432x324-cpu-2026-04-04', x265_floor_id, 'promotion')
    add_lineage_edge('robust_current-432x324-cpu-2026-04-04', roi_fail_id, 'rejection')
    add_lineage_edge(x265_floor_id, nearby_reject_id, 'rejection')
    add_lineage_edge(x265_floor_id, av1_bug_id, 'diagnostic')
    add_lineage_edge(av1_bug_id, current_best_id, 'repair')

    def edge_path(source: dict, target: dict) -> str:
        mx = (source['x'] + target['x']) / 2
        my = min(source['y'], target['y']) - 46
        return f"M {source['x']} {source['y']} Q {mx:.1f} {my:.1f} {target['x']} {target['y']}"

    node_lookup = {n['run_id']: n for n in lineage_nodes}
    lineage_edge_svg = []
    for edge in lineage_edges:
        source = node_lookup.get(edge['source'])
        target = node_lookup.get(edge['target'])
        if not source or not target:
            continue
        lineage_edge_svg.append(
            f'<path class="lineage-edge lineage-edge-{edge["kind"]}" d="{edge_path(source, target)}" />'
        )

    def lineage_detail_lines(node: dict) -> tuple[str, str]:
        cfg = node['config']
        score_prefix = f"{node['score']:.2f}"
        if cfg.get('video_codec') == 'libsvtav1':
            line1 = f"{score_prefix} · {cfg.get('scale_w')}x{cfg.get('scale_h')} · crf {cfg.get('crf')}"
            filters = str(cfg.get('filters', ''))
            preset = cfg.get('preset')
            if filters and 'explicit' in filters:
                line2 = f"{preset} · explicit color"
            else:
                line2 = f"{preset} · {filters}"
            return line1, line2
        line1 = f"{score_prefix} · {cfg.get('scale_w')}x{cfg.get('scale_h')} · crf {cfg.get('crf')}"
        line2 = f"k{cfg.get('gop')} · {cfg.get('filters')}"
        return line1, line2

    lineage_node_svg = []
    svg_width = 1180
    for idx, node in enumerate(lineage_nodes):
        color_cls = f'lineage-node-{node["family"]}'
        title = escape(f'{node["run_id"]} | score={node["score"]:.2f} | bytes={node["bytes"]:,}')
        detail_line_1, detail_line_2 = lineage_detail_lines(node)
        box_w = 178
        box_h = 54
        if node['family'] in {'roi', 'bug'}:
            box_y = node['y'] - 76
        elif node['family'] == 'rejection':
            box_y = node['y'] + 18
        else:
            box_y = node['y'] - 76 if idx % 2 else node['y'] + 18
        box_x = max(10, min(node['x'] - box_w / 2, svg_width - box_w - 10))
        text_x = box_x + 10
        title_y = box_y + 16
        meta_y_1 = box_y + 30
        meta_y_2 = box_y + 42
        lineage_node_svg.append(
            f'<circle class="lineage-node {color_cls}" cx="{node["x"]}" cy="{node["y"]}" r="9"><title>{title}</title></circle>'
            f'<g class="lineage-note">'
            f'<rect class="lineage-note-box" x="{box_x:.1f}" y="{box_y:.1f}" width="{box_w}" height="{box_h}" rx="10" ry="10"></rect>'
            f'<text class="lineage-note-title" x="{text_x:.1f}" y="{title_y:.1f}">{escape(node["label"])}</text>'
            f'<text class="lineage-note-meta" x="{text_x:.1f}" y="{meta_y_1:.1f}">{escape(detail_line_1)}</text>'
            f'<text class="lineage-note-meta" x="{text_x:.1f}" y="{meta_y_2:.1f}">{escape(detail_line_2)}</text>'
            f'</g>'
        )

    baseline_run = run_index.get('robust_current-baseline-cpu-2026-04-03')
    baseline_score = baseline_run['current_workflow_score'] if baseline_run else None
    baseline_bytes = baseline_run['archive_bytes'] if baseline_run else None
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

    html = f'''<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>comma-lab</title>
  <meta name="description" content="Scorer-backed compression experiments, evidence, and reporting for the comma video challenge." />
  <meta name="robots" content="index,follow" />
  <meta name="theme-color" content="#0b1020" />
  <meta name="color-scheme" content="dark" />
  <link rel="canonical" href="{CANONICAL_URL}" />
  <meta property="og:title" content="comma-lab" />
  <meta property="og:description" content="Scorer-backed compression experiments and evidence for the comma video challenge." />
  <meta property="og:type" content="website" />
  <meta property="og:url" content="{CANONICAL_URL}" />
  <meta name="twitter:card" content="summary_large_image" />
  <script type="application/ld+json">{{"@context":"https://schema.org","@type":"WebSite","name":"comma-lab","url":"{CANONICAL_URL}"}}</script>
  <style>
    :root {{
      --bg: #0a0d12;
      --panel: rgba(255,255,255,0.045);
      --panel-strong: rgba(255,255,255,0.035);
      --stroke: rgba(255,255,255,0.09);
      --text: #f5f7fb;
      --muted: #9aa6b2;
      --soft: #cbd5df;
      --accent: #8ab4ff;
      --accent-2: #8ab4ff;
      --danger: #ff6b6b;
      --shadow: none;
      --radius: 20px;
      --radius-sm: 14px;
      --max: 1200px;
    }}
    * {{ box-sizing: border-box; }}
    html {{ scroll-behavior: smooth; }}
    body {{ margin: 0; background: #0b0d10; color: var(--text); font: 500 16px/1.6 Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "SF Pro Display", sans-serif; text-rendering: optimizeLegibility; }}
    a {{ color: inherit; text-underline-offset: 0.2em; }}
    a:focus-visible, button:focus-visible {{ outline: 3px solid var(--accent-2); outline-offset: 3px; border-radius: 6px; }}
    .skip {{ position: absolute; left: 12px; top: -48px; background: #fff; color: #000; padding: 10px 14px; border-radius: 10px; z-index: 99; }}
    .skip:focus {{ top: 12px; }}
    .wrap {{ max-width: var(--max); margin: 0 auto; padding: 28px 24px 72px; }}
    .hero {{ padding: 28px 0 10px; display: grid; gap: 14px; }}
    .eyebrow {{ color: var(--muted); font-size: 11px; letter-spacing: 0.12em; text-transform: uppercase; font-weight: 700; }}
    h1 {{ margin: 0; font-size: clamp(40px, 6vw, 72px); line-height: 0.96; letter-spacing: -0.045em; font-weight: 760; }}
    .lede {{ max-width: 760px; color: var(--soft); font-size: 16px; margin: 0; }}
    .two {{ display: grid; gap: 16px; }}
    .two {{ grid-template-columns: 1.25fr 0.75fr; }}
    .panel {{ background: #0f1318; border: 1px solid var(--stroke); border-radius: 12px; box-shadow: none; padding: 20px; margin-top: 18px; }}
    .panel h2 {{ margin: 0 0 10px; font-size: 20px; letter-spacing: -0.03em; }}
    .muted {{ color: var(--muted); }}
    .subnav {{ display:flex; gap:14px; flex-wrap:wrap; font-size: 13px; color: var(--muted); }}
    .subnav a {{ text-decoration: none; }}
    .summary-grid {{ display:grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 0; border-top: 1px solid var(--stroke); border-bottom: 1px solid var(--stroke); }}
    .summary-stat {{ padding: 14px 16px; }}
    .summary-stat + .summary-stat {{ border-left: 1px solid var(--stroke); }}
    .summary-label {{ color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: 0.08em; }}
    .summary-value {{ font-size: 28px; line-height: 1; margin-top: 8px; font-weight: 720; letter-spacing: -0.04em; }}
    .summary-meta {{ color: var(--muted); font-size: 12px; margin-top: 8px; }}
    .note-grid {{ display:grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 18px; }}
    .note {{ padding: 0 14px 0 0; }}
    .note + .note {{ border-left: 1px solid var(--stroke); padding-left: 18px; }}
    .note h3 {{ margin: 0 0 8px; font-size: 14px; letter-spacing: -0.02em; }}
    .note p {{ margin: 0; color: var(--soft); font-size: 13px; line-height: 1.55; }}
    .note p + p {{ margin-top: 8px; color: var(--muted); }}
    table {{ width: 100%; border-collapse: collapse; }}
    th, td {{ border-bottom: 1px solid var(--stroke); padding: 10px 8px; text-align: left; vertical-align: top; }}
    th {{ color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: 0.08em; }}
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
    .legend-row {{ display:flex; gap:10px; flex-wrap:wrap; margin: 12px 0 0; }}
    .legend-key {{ display:inline-flex; align-items:center; gap:8px; color: var(--muted); font-size: 12px; }}
    .legend-dot {{ width:10px; height:10px; border-radius:999px; display:inline-block; }}
    .legend-line {{ width:22px; height:0; border-top:2px solid currentColor; display:inline-block; }}
    .bug-box {{ display:grid; grid-template-columns: 1fr auto 1fr; gap: 10px; align-items:center; margin-top: 12px; }}
    .bug-side {{ border: 1px solid var(--stroke); border-radius: var(--radius-sm); padding: 12px; background: rgba(255,255,255,0.03); }}
    .bug-side strong {{ display:block; margin-bottom: 6px; }}
    .metric-delta {{ color: var(--muted); font-size: 12px; margin-top: 6px; }}
    .frontier-callout {{ color: var(--soft); font-size: 14px; line-height: 1.6; margin: 0 0 14px; }}
    .footnote {{ color: var(--muted); font-size: 13px; }}
    .axis-label {{ fill: var(--muted); font-size: 12px; letter-spacing: 0.02em; }}
    .axis-tick {{ fill: var(--muted); font-size: 11px; }}
    .mono-block {{ margin: 12px 0 0; padding: 14px; border: 1px solid var(--stroke); border-radius: 10px; background: #0b0f14; color: var(--soft); font: 500 13px/1.6 ui-monospace, SFMono-Regular, Menlo, monospace; white-space: pre-wrap; }}
    .list-tight {{ margin: 0; padding-left: 18px; }}
    .list-tight li + li {{ margin-top: 6px; }}
    @media (max-width: 980px) {{ .two, .note-grid {{ grid-template-columns: 1fr 1fr; }} .summary-grid {{ grid-template-columns: 1fr; }} .summary-stat + .summary-stat {{ border-left: 0; border-top: 1px solid var(--stroke); }} .note:nth-child(3) {{ border-left: 0; padding-left: 0; }} }}
    @media (max-width: 720px) {{ .two, .note-grid, .summary-grid {{ grid-template-columns: 1fr; }} .summary-stat + .summary-stat {{ border-left: 0; border-top: 1px solid var(--stroke); }} .note + .note {{ border-left: 0; border-top: 1px solid var(--stroke); padding-left: 0; padding-top: 14px; }} .wrap {{ padding-inline: 16px; }} h1 {{ font-size: clamp(36px, 15vw, 64px); }} }}
    @media (prefers-reduced-motion: reduce) {{ html {{ scroll-behavior: auto; }} * {{ animation: none !important; transition: none !important; }} }}
  </style>
</head>
<body>
<a class="skip" href="#content">Skip to content</a>
<div class="wrap">
    <header class="hero" aria-labelledby="site-title">
      <div class="eyebrow">comma-lab</div>
      <h1 id="site-title">comma-lab</h1>
      <p class="lede">Scorer-backed compression experiments for the comma video challenge. This page is the brief: current state, search path, and supporting references.</p>
      <nav class="subnav" aria-label="Primary resources">
        <a href="./judges_one_pager.md">Judges one-pager</a>
        <a href="./submission_packet.md">Submission packet</a>
        <a href="./promotion_review_latest.md">Promotion review</a>
        <a href="./evidence_index.md">Evidence index</a>
      </nav>
    </header>

  <main id="content">
    <section class="panel" aria-label="Current state">
      <div class="summary-grid">
        <div class="summary-stat"><div class="summary-label">Track B current_workflow</div><div class="summary-value">{data['summary']['best_track_b_score']:.2f}</div><div class="summary-meta">{data['summary']['best_track_b_bytes']:,} bytes</div></div>
        <div class="summary-stat"><div class="summary-label">Track B rule_faithful</div><div class="summary-value">{best.get('rule_faithful_score', 0):.3f}</div><div class="summary-meta">{fmt_int(best.get('rule_faithful_bundle_bytes'))} bytes</div></div>
        <div class="summary-stat"><div class="summary-label">Run ledger</div><div class="summary-value">{data['summary']['robust_run_count']}</div><div class="summary-meta">{data['summary']['promotion_count']} promotions / {data['summary']['rejection_count']} rejections</div></div>
      </div>
      <p class="footnote">Track A remains separate: `current_workflow` 0.00 at 167 bytes. It is preserved for transparency and is not part of the honest promotion path.</p>
      <div class="mono-block" aria-label="Current promoted method">track: robust_current
codec: libsvtav1
scale: 524x394
decode: explicit rgb24(pc)
archive: 864,486 bytes
score: 2.12</div>
    </section>

    <section class="panel" aria-labelledby="scatter-title">
      <h2 id="scatter-title">Score vs bytes</h2>
      <p class="muted">Track B runs only. The x-axis uses log scaling. The y-axis is linear. The severe AV1 bug run at 97.45 is omitted here so the operating range stays legible; it remains documented in the search-path section.</p>
      <svg viewBox="0 0 {width} {height}" role="img" aria-label="Score versus bytes scatterplot for Track B runs">
        {''.join(grid_lines)}
        <line x1="{pad}" y1="{height-pad}" x2="{width-pad}" y2="{height-pad}" stroke="#475569" />
        <line x1="{pad}" y1="{pad}" x2="{pad}" y2="{height-pad}" stroke="#475569" />
        {''.join(points)}
        <text x="{width/2:.1f}" y="{height-10}" text-anchor="middle" class="axis-label">archive size (bytes, log scale)</text>
        <text x="16" y="{height/2:.1f}" transform="rotate(-90 16 {height/2:.1f})" text-anchor="middle" class="axis-label">current_workflow score (linear, lower is better)</text>
      </svg>
      <div class="legend-row" aria-label="Scatter legend">
        <span class="legend-key"><span class="legend-dot" style="background:#2ad4a0"></span>promotion</span>
        <span class="legend-key"><span class="legend-dot" style="background:#ff6b6b"></span>explicit rejection</span>
        <span class="legend-key"><span class="legend-dot" style="background:#8ab4ff"></span>measured run</span>
      </div>
      <p class="footnote">Lower is better. The plot is generated directly from scorer-backed artifacts in the repository.</p>
    </section>

    <section class="panel" aria-labelledby="lineage-title">
      <h2 id="lineage-title">Search path</h2>
      <p class="muted">A branch view of the measured search path: x265 reductions, ROI failures, the AV1 byte-layout failure, and the later hardening step that moved the floor to 2.12.</p>
      <div class="legend-row" aria-label="Lineage legend">
        <span class="legend-key"><span class="legend-dot" style="background:#2ad4a0"></span>x265 / earlier promotions</span>
        <span class="legend-key"><span class="legend-dot" style="background:#8ab4ff"></span>current best floor</span>
        <span class="legend-key"><span class="legend-dot" style="background:#f6c14b"></span>diagnostic bug node</span>
        <span class="legend-key"><span class="legend-dot" style="background:#ff6b6b"></span>rejection / failed branch</span>
      </div>
      <div class="lineage-wrap">
        <svg viewBox="0 0 1180 360" role="img" aria-label="Experimental lineage graph showing promotions, failures, and repaired AV1 branch">
          {''.join(lineage_edge_svg)}
          {''.join(lineage_node_svg)}
        </svg>
      </div>
      <p class="footnote">This graph is selective by design. It shows the runs that changed the operating point or changed the lab’s understanding of the evaluator.</p>
    </section>

    <section class="panel" aria-labelledby="mechanism-title">
      <h2 id="mechanism-title">Operational notes</h2>
      <div class="note-grid">
        <article class="note">
          <h3>Evaluator path</h3>
          <p>{escape(hypothesis_cards[0]['summary'])}</p>
          <p>{escape(hypothesis_cards[0]['detail'])}</p>
        </article>
        <article class="note">
          <h3>Current operating point</h3>
          <p>{escape(hypothesis_cards[2]['summary'])}</p>
          <p>{escape(hypothesis_cards[2]['detail'])}</p>
        </article>
        <article class="note">
          <h3>Critical bug</h3>
          <p>{escape(hypothesis_cards[1]['summary'])}</p>
          <p>{escape(hypothesis_cards[1]['detail'])}</p>
        </article>
      </div>
    </section>

    <section class="panel" aria-labelledby="frontier-title">
      <h2 id="frontier-title">Local neighborhood</h2>
      <p class="frontier-callout">This table isolates the local AV1 neighborhood around the promoted floor. It shows which nearby changes improved the result and which did not.</p>
      <table>
        <thead><tr><th scope="col">Run</th><th scope="col">Changed axis</th><th scope="col">Score</th><th scope="col">Bytes</th><th scope="col">Δ score</th><th scope="col">Δ bytes</th><th scope="col">Verdict</th></tr></thead>
        <tbody>{''.join(frontier_rows)}</tbody>
      </table>
    </section>

    <section class="panel">
      <h2>References</h2>
      <div class="two">
        <div>
          <ul class="list-tight">
            <li><a href="./judges_one_pager.md">Judges one-pager</a></li>
            <li><a href="./submission_packet.md">Submission packet</a></li>
            <li><a href="./promotion_review_latest.md">Promotion review</a></li>
            <li><a href="./promotion_accounting.md">Promotion accounting</a></li>
            <li><a href="./ffmpeg_path_review.md">ffmpeg path review</a></li>
            <li><a href="./experiment_journal.md">Experiment journal</a></li>
            <li><a href="./evidence_index.md">Evidence index</a></li>
          </ul>
        </div>
        <div>
          <ul class="list-tight">{''.join(turning_items)}</ul>
        </div>
      </div>
    </section>
  </main>
</div>
</body>
</html>
'''
    OUT_HTML.write_text(html)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
