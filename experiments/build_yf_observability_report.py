#!/usr/bin/env python3
"""Build deterministic observability artifacts for Yousfi-Fridrich optimization.

The report is a human-facing control-plane artifact. It summarizes exact eval
component breakdowns and planning-only atom profiles, then emits JSON,
Markdown, HTML, and SVG figures. It never creates a score claim by itself.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping


SCHEMA = "yousfi_fridrich_observability_report_v1"
TOOL = "experiments/build_yf_observability_report.py"
SCORE_DENOMINATOR = 37_545_489  # [contest-defined: original video bytes]
DEFAULT_TARGET_SCORE = 0.300  # [heuristic: aspirational frontier-floor target]


class ReportError(RuntimeError):
    """Raised when report inputs are malformed."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ReportError(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ReportError(f"{path} must contain a JSON object")
    return payload


def _json_text(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"


def _finite_number(payload: Mapping[str, Any], key: str, *, default: float | None = None) -> float:
    value = payload.get(key, default)
    if value is None:
        raise ReportError(f"missing numeric field {key!r}")
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ReportError(f"field {key!r} must be finite numeric")
    return float(value)


def _optional_number(payload: Mapping[str, Any], key: str) -> float | None:
    value = payload.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ReportError(f"field {key!r} must be finite numeric when present")
    return float(value)


def _fmt(value: float | int | None, *, digits: int = 6) -> str:
    if value is None:
        return ""
    if isinstance(value, int):
        return f"{value:,}"
    if abs(value) >= 10:
        return f"{value:.6f}"
    return f"{value:.{digits}f}"


def _parse_eval_spec(raw: str) -> tuple[str, Path]:
    if "=" in raw:
        label, path = raw.split("=", 1)
        label = label.strip()
        if not label:
            raise argparse.ArgumentTypeError("--eval label cannot be empty")
        return label, Path(path)
    path = Path(raw)
    return path.parent.name or path.stem, path


def _eval_record(label: str, path: Path) -> dict[str, Any]:
    path = path.resolve()
    payload = _read_json(path)
    archive_bytes = int(_finite_number(payload, "archive_size_bytes"))
    seg = _finite_number(payload, "avg_segnet_dist")
    pose = _finite_number(payload, "avg_posenet_dist")
    recomputed = _finite_number(payload, "score_recomputed_from_components")
    seg_part = _optional_number(payload, "score_seg_contribution")
    pose_part = _optional_number(payload, "score_pose_contribution")
    rate_part = _optional_number(payload, "score_rate_contribution")
    if seg_part is None:
        seg_part = 100.0 * seg
    if pose_part is None:
        pose_part = math.sqrt(10.0 * pose)
    if rate_part is None:
        rate_part = 25.0 * archive_bytes / SCORE_DENOMINATOR
    formula = 100.0 * seg + math.sqrt(10.0 * pose) + 25.0 * archive_bytes / SCORE_DENOMINATOR
    provenance = payload.get("provenance") if isinstance(payload.get("provenance"), Mapping) else {}
    return {
        "archive_size_bytes": archive_bytes,
        "archive_sha256": provenance.get("archive_sha256") or payload.get("archive_sha256"),
        "avg_posenet_dist": pose,
        "avg_segnet_dist": seg,
        "device": provenance.get("device"),
        "final_score": _optional_number(payload, "final_score"),
        "gpu_model": provenance.get("gpu_model"),
        "input": {
            "path": str(path),
            "sha256": _sha256_file(path),
            "size_bytes": path.stat().st_size,
        },
        "label": label,
        "n_samples": int(_finite_number(payload, "n_samples")),
        "rate_component": rate_part,
        "score_claim_source": "input_exact_eval_json",
        "score_pose_contribution": pose_part,
        "score_recomputed_from_components": recomputed,
        "score_seg_contribution": seg_part,
        "score_formula_cross_check": {
            "absolute_difference": abs(formula - recomputed),
            "formula_score": formula,
            "matches_within_1e_5": abs(formula - recomputed) <= 1e-5,
        },
    }


def _add_eval_reference_signals(evals: list[dict[str, Any]]) -> None:
    if not evals:
        return
    reference = min(evals, key=lambda item: float(item["score_recomputed_from_components"]))
    ref_score = max(float(reference["score_recomputed_from_components"]), 1e-12)
    ref_pose = max(float(reference["avg_posenet_dist"]), 1e-12)
    ref_seg = max(float(reference["avg_segnet_dist"]), 1e-12)
    ref_bytes = int(reference["archive_size_bytes"])
    for item in evals:
        pose_ratio = float(item["avg_posenet_dist"]) / ref_pose
        seg_ratio = float(item["avg_segnet_dist"]) / ref_seg
        score_delta = float(item["score_recomputed_from_components"]) - ref_score
        byte_delta = int(item["archive_size_bytes"]) - ref_bytes
        component_cliff = pose_ratio >= 4.0 or seg_ratio >= 4.0
        if score_delta > 0 and byte_delta < 0:
            wasted_rate_gain = score_delta / max(abs(byte_delta), 1)
        else:
            wasted_rate_gain = 0.0
        item["fridrich_yousfi_signals"] = {
            "byte_delta_vs_best": byte_delta,
            "component_cliff": component_cliff,
            "pose_ratio_vs_best": pose_ratio,
            "score_delta_vs_best": score_delta,
            "seg_ratio_vs_best": seg_ratio,
            "wasted_score_per_saved_byte": wasted_rate_gain,
            "reference_label": reference["label"],
        }


def _target_gap_analysis(evals: list[Mapping[str, Any]], *, target_score: float) -> dict[str, Any]:
    best = min(evals, key=lambda item: float(item["score_recomputed_from_components"]))
    best_score = float(best["score_recomputed_from_components"])
    distortion_score = float(best["score_seg_contribution"]) + float(best["score_pose_contribution"])
    current_rate_score = float(best["rate_component"])
    target_rate_score = target_score - distortion_score
    score_gap = best_score - target_score
    bytes_at_target_if_distortion_unchanged = target_rate_score * SCORE_DENOMINATOR / 25.0
    bytes_to_remove_if_distortion_unchanged = int(
        math.ceil(
            max(
                0.0,
                float(best["archive_size_bytes"]) - bytes_at_target_if_distortion_unchanged,
            )
        )
    )
    distortion_room = target_score - current_rate_score
    return {
        "best_label": best["label"],
        "best_score": best_score,
        "bytes_at_target_if_distortion_unchanged": max(0.0, bytes_at_target_if_distortion_unchanged),
        "bytes_to_remove_if_distortion_unchanged": bytes_to_remove_if_distortion_unchanged,
        "current_archive_size_bytes": int(best["archive_size_bytes"]),
        "current_distortion_score": distortion_score,
        "current_rate_score": current_rate_score,
        "distortion_score_room_at_current_bytes": distortion_room,
        "score_gap_to_target": score_gap,
        "target_score": target_score,
    }


def _action_recommendations(
    *,
    evals: list[Mapping[str, Any]],
    profile_summary: Mapping[str, Any] | None,
    target_gap: Mapping[str, Any],
) -> list[dict[str, Any]]:
    cliff_evals = [
        item
        for item in evals
        if item.get("fridrich_yousfi_signals", {}).get("component_cliff")
    ]
    top_pairs = [] if profile_summary is None else list(profile_summary.get("top_pairs", []))[:8]
    top_confusions = [] if profile_summary is None else list(profile_summary.get("top_confusions", []))[:4]
    recommendations: list[dict[str, Any]] = []
    if cliff_evals and top_pairs:
        recommendations.append(
            {
                "action": "build_repaired_or_multimask_candidates",
                "evidence": "byte-saving exact negatives plus repeated active-subspace hotspots",
                "first_pairs": [str(row["key"]) for row in top_pairs],
                "first_confusions": [str(row["key"]) for row in top_confusions],
                "why": (
                    "Raw row-span replacement saves bytes but creates a component cliff. "
                    "The next mask candidate should charge only consensus repair/fusion atoms "
                    "around repeated PoseNet-sensitive regions."
                ),
            }
        )
    if int(target_gap["bytes_to_remove_if_distortion_unchanged"]) > 0:
        recommendations.append(
            {
                "action": "prioritize_large_rate_levers_before_micro_pose_polish",
                "bytes_needed_at_fixed_distortion": int(
                    target_gap["bytes_to_remove_if_distortion_unchanged"]
                ),
                "evidence": "target gap decomposition",
                "why": (
                    "At the current distortion level, sub-0.300 requires a material byte "
                    "drop; pose-only micro-gains cannot close the full gap."
                ),
            }
        )
    recommendations.append(
        {
            "action": "promote_only_exact_archive_bytes",
            "evidence": "AGENTS exact CUDA contract",
            "why": (
                "Profiler, visualization, and byte-screen output are control-plane signals only. "
                "Every recommendation must resolve to a deterministic archive and exact CUDA auth eval."
            ),
        }
    )
    return recommendations


def _table_rows(rows: Iterable[Iterable[str]]) -> str:
    return "\n".join("| " + " | ".join(row) + " |" for row in rows)


def _markdown(payload: Mapping[str, Any]) -> str:
    lines: list[str] = [
        "# Yousfi-Fridrich Observability Report",
        "",
        f"Schema: `{payload['schema']}`",
        "",
        "This is a control-plane observability artifact. It does not promote, rank, or claim a new score.",
        "",
        "## Target Gap",
        "",
    ]
    gap = payload["target_gap_analysis"]
    lines.extend(
        [
            f"- Best exact anchor: `{gap['best_label']}` at `{_fmt(float(gap['best_score']), digits=9)}`.",
            f"- Target score: `{_fmt(float(gap['target_score']), digits=9)}`.",
            "- Bytes to remove at unchanged distortion: "
            f"`{_fmt(int(gap['bytes_to_remove_if_distortion_unchanged']))}`.",
            "",
            "## Action Recommendations",
            "",
        ]
    )
    for item in payload["action_recommendations"]:
        lines.append(f"- `{item['action']}`: {item['why']}")
    lines.extend(
        [
            "",
        "## Exact Eval Component Breakdown",
        "",
        ]
    )
    evals = payload["exact_evals"]
    rows = [
        ["label", "score", "bytes", "dbytes", "seg", "pose", "rate", "pose ratio", "seg ratio", "cliff"],
        ["---", "---:", "---:", "---:", "---:", "---:", "---:", "---:", "---:", "---"],
    ]
    for item in evals:
        signals = item.get("fridrich_yousfi_signals", {})
        rows.append(
            [
                str(item["label"]),
                _fmt(float(item["score_recomputed_from_components"]), digits=9),
                _fmt(int(item["archive_size_bytes"])),
                _fmt(int(signals.get("byte_delta_vs_best", 0))),
                _fmt(float(item["score_seg_contribution"])),
                _fmt(float(item["score_pose_contribution"])),
                _fmt(float(item["rate_component"])),
                _fmt(float(signals.get("pose_ratio_vs_best", 1.0)), digits=3),
                _fmt(float(signals.get("seg_ratio_vs_best", 1.0)), digits=3),
                "yes" if signals.get("component_cliff") else "no",
            ]
        )
    lines.append(_table_rows(rows))
    lines.extend(["", "## Atom Subspace Summary", ""])
    profile = payload.get("atom_profile_summary")
    if profile:
        lines.append(f"Profile: `{profile['input']['path']}`")
        lines.append("")
        for section, title in (
            ("top_pairs", "Top Pairs"),
            ("top_classes", "Top Classes"),
            ("top_confusions", "Top Class Confusions"),
        ):
            lines.append(f"### {title}")
            table = [["key", "hits", "score proxy", "bytes proxy", "score per byte"], ["---", "---:", "---:", "---:", "---:"]]
            for row in profile[section]:
                table.append(
                    [
                        str(row["key"]),
                        str(row["hit_count"]),
                        _fmt(float(row["estimated_marginal_score_saved_proxy_sum"]), digits=9),
                        _fmt(float(row["estimated_charged_bytes_sum"]), digits=3),
                        _fmt(float(row["estimated_score_saved_per_charged_byte"]), digits=12),
                    ]
                )
            lines.append(_table_rows(table))
            lines.append("")
    lines.extend(
        [
            "## Generated Figures",
            "",
            "- `score_breakdown.svg`",
            "- `top_pairs.svg`",
            "- `target_gap.svg`",
        ]
    )
    mpl = payload.get("matplotlib")
    if isinstance(mpl, Mapping) and mpl.get("generated_files"):
        for filename in mpl["generated_files"]:
            lines.append(f"- `{filename}`")
    lines.append("")
    return "\n".join(lines)


def _html_doc(payload: Mapping[str, Any], markdown: str) -> str:
    rows = []
    for item in payload["exact_evals"]:
        signals = item.get("fridrich_yousfi_signals", {})
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(item['label']))}</td>"
            f"<td>{_fmt(float(item['score_recomputed_from_components']), digits=9)}</td>"
            f"<td>{_fmt(int(item['archive_size_bytes']))}</td>"
            f"<td>{_fmt(int(signals.get('byte_delta_vs_best', 0)))}</td>"
            f"<td>{_fmt(float(item['score_seg_contribution']))}</td>"
            f"<td>{_fmt(float(item['score_pose_contribution']))}</td>"
            f"<td>{_fmt(float(item['rate_component']))}</td>"
            f"<td>{_fmt(float(signals.get('pose_ratio_vs_best', 1.0)), digits=3)}</td>"
            f"<td>{_fmt(float(signals.get('seg_ratio_vs_best', 1.0)), digits=3)}</td>"
            f"<td>{html.escape(str(item.get('gpu_model') or ''))}</td>"
            "</tr>"
        )
    png_files = []
    figure_files = payload.get("figure_files")
    if isinstance(figure_files, Mapping):
        generated = figure_files.get("matplotlib_png")
        if isinstance(generated, list):
            png_files = [str(item) for item in generated]
    button_html = "".join(
        f'<button type="button" data-visual-picker="{html.escape(filename)}">{html.escape(filename)}</button>'
        for filename in png_files
    )
    first_png = html.escape(png_files[0]) if png_files else ""
    png_gallery = ""
    if png_files:
        png_gallery = f"""
  <div class="picker">{button_html}</div>
  <img id="active-figure" src="{first_png}" alt="selected observability figure">
"""
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Yousfi-Fridrich Observability</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 32px; color: #17202a; }}
    table {{ border-collapse: collapse; width: 100%; margin: 16px 0 28px; }}
    th, td {{ border-bottom: 1px solid #d9e0e7; padding: 8px 10px; text-align: right; }}
    th:first-child, td:first-child {{ text-align: left; }}
    .figures {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr)); gap: 24px; }}
    .picker {{ display: flex; flex-wrap: wrap; gap: 8px; margin: 18px 0 10px; }}
    .picker button {{ border: 1px solid #bcc6d0; background: #f7f9fb; padding: 6px 10px; cursor: pointer; }}
    #active-figure {{ max-width: 100%; border: 1px solid #d9e0e7; }}
    .note {{ color: #566573; max-width: 980px; line-height: 1.45; }}
    pre {{ white-space: pre-wrap; background: #f6f8fa; padding: 16px; overflow-x: auto; }}
    th[data-sort-key] {{ cursor: pointer; }}
  </style>
</head>
<body>
  <h1>Yousfi-Fridrich Observability</h1>
  <p class="note">Control-plane report only. Score truth remains exact CUDA auth eval of exact archive bytes.</p>
  {png_gallery}
  <section class="figures">
    <object data="target_gap.svg" type="image/svg+xml"></object>
    <object data="score_breakdown.svg" type="image/svg+xml"></object>
    <object data="top_pairs.svg" type="image/svg+xml"></object>
  </section>
  <h2>Exact Eval Component Breakdown</h2>
  <table id="exact-eval-table">
    <thead><tr><th data-sort-key="label">label</th><th data-sort-key="number">score</th><th data-sort-key="number">bytes</th><th data-sort-key="number">dbytes</th><th data-sort-key="number">seg</th><th data-sort-key="number">pose</th><th data-sort-key="number">rate</th><th data-sort-key="number">pose ratio</th><th data-sort-key="number">seg ratio</th><th data-sort-key="label">GPU</th></tr></thead>
    <tbody>{''.join(rows)}</tbody>
  </table>
  <h2>Markdown Source</h2>
  <pre>{html.escape(markdown)}</pre>
  <script>
    for (const button of document.querySelectorAll('[data-visual-picker]')) {{
      button.addEventListener('click', () => {{
        const img = document.getElementById('active-figure');
        if (img) img.src = button.dataset.visualPicker;
      }});
    }}
    const table = document.getElementById('exact-eval-table');
    if (table) {{
      for (const [index, th] of table.querySelectorAll('th').entries()) {{
        th.addEventListener('click', () => {{
          const rows = Array.from(table.tBodies[0].rows);
          const numeric = th.dataset.sortKey === 'number';
          rows.sort((a, b) => {{
            const av = a.cells[index].textContent.trim().replace(/,/g, '');
            const bv = b.cells[index].textContent.trim().replace(/,/g, '');
            return numeric ? (parseFloat(av) - parseFloat(bv)) : av.localeCompare(bv);
          }});
          for (const row of rows) table.tBodies[0].appendChild(row);
        }});
      }}
    }}
  </script>
</body>
</html>
"""


def _svg_score_breakdown(evals: list[Mapping[str, Any]]) -> str:
    width = 980
    row_h = 44
    top = 46
    height = top + row_h * max(1, len(evals)) + 40
    max_score = max(float(item["score_recomputed_from_components"]) for item in evals) if evals else 1.0
    max_score = max(max_score, 1e-9)
    colors = {"seg": "#3b82f6", "pose": "#ef4444", "rate": "#22c55e"}
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="20" y="28" font-family="Arial, sans-serif" font-size="18" font-weight="700">Score component breakdown</text>',
    ]
    x0 = 220
    bar_w = 640
    for idx, item in enumerate(evals):
        y = top + idx * row_h
        label = html.escape(str(item["label"]))
        seg = float(item["score_seg_contribution"])
        pose = float(item["score_pose_contribution"])
        rate = float(item["rate_component"])
        score = float(item["score_recomputed_from_components"])
        lines.append(f'<text x="20" y="{y + 20}" font-family="Arial, sans-serif" font-size="12">{label}</text>')
        cursor = x0
        for key, value in (("seg", seg), ("pose", pose), ("rate", rate)):
            w = 0 if score <= 0 else bar_w * value / max_score
            lines.append(f'<rect x="{cursor:.3f}" y="{y}" width="{w:.3f}" height="24" fill="{colors[key]}"/>')
            cursor += w
        lines.append(f'<text x="{x0 + bar_w + 12}" y="{y + 17}" font-family="Arial, sans-serif" font-size="12">{_fmt(score, digits=6)}</text>')
    legend_y = height - 18
    lx = 20
    for key in ("seg", "pose", "rate"):
        lines.append(f'<rect x="{lx}" y="{legend_y - 10}" width="12" height="12" fill="{colors[key]}"/>')
        lines.append(f'<text x="{lx + 18}" y="{legend_y}" font-family="Arial, sans-serif" font-size="12">{key}</text>')
        lx += 78
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def _svg_target_gap(target_gap: Mapping[str, Any]) -> str:
    width = 980
    height = 220
    score_gap = max(float(target_gap["score_gap_to_target"]), 0.0)
    current_rate = float(target_gap["current_rate_score"])
    current_distortion = float(target_gap["current_distortion_score"])
    target_score = float(target_gap["target_score"])
    best_score = float(target_gap["best_score"])
    scale_max = max(best_score, target_score, 1e-9)
    x0 = 170
    bar_w = 620
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="20" y="30" font-family="Arial, sans-serif" font-size="18" font-weight="700">Sub-0.300 target gap</text>',
        f'<text x="20" y="72" font-family="Arial, sans-serif" font-size="12">anchor {html.escape(str(target_gap["best_label"]))}</text>',
    ]
    y = 58
    rate_w = bar_w * current_rate / scale_max
    distortion_w = bar_w * current_distortion / scale_max
    lines.append(f'<rect x="{x0}" y="{y}" width="{distortion_w:.3f}" height="28" fill="#ef4444"/>')
    lines.append(f'<rect x="{x0 + distortion_w:.3f}" y="{y}" width="{rate_w:.3f}" height="28" fill="#22c55e"/>')
    target_x = x0 + bar_w * target_score / scale_max
    lines.append(f'<line x1="{target_x:.3f}" x2="{target_x:.3f}" y1="{y - 10}" y2="{y + 44}" stroke="#111827" stroke-width="2"/>')
    lines.append(f'<text x="{target_x + 6:.3f}" y="{y - 2}" font-family="Arial, sans-serif" font-size="11">target {_fmt(target_score, digits=3)}</text>')
    lines.append(f'<text x="{x0 + bar_w + 12}" y="{y + 18}" font-family="Arial, sans-serif" font-size="12">{_fmt(best_score, digits=9)}</text>')
    bytes_needed = int(target_gap["bytes_to_remove_if_distortion_unchanged"])
    lines.append(
        f'<text x="20" y="130" font-family="Arial, sans-serif" font-size="14">'
        f'Need {bytes_needed:,} bytes removed at unchanged distortion; score gap {_fmt(score_gap, digits=9)}.</text>'
    )
    lines.append('<rect x="20" y="158" width="12" height="12" fill="#ef4444"/>')
    lines.append('<text x="38" y="169" font-family="Arial, sans-serif" font-size="12">SegNet + PoseNet distortion</text>')
    lines.append('<rect x="245" y="158" width="12" height="12" fill="#22c55e"/>')
    lines.append('<text x="263" y="169" font-family="Arial, sans-serif" font-size="12">Rate</text>')
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def _svg_top_pairs(profile_summary: Mapping[str, Any] | None) -> str:
    width = 980
    rows = [] if profile_summary is None else list(profile_summary.get("top_pairs", []))[:16]
    row_h = 30
    top = 46
    height = top + max(1, len(rows)) * row_h + 28
    max_value = max((float(row["estimated_lagrangian_net_proxy_sum"]) for row in rows), default=1.0)
    max_value = max(max_value, 1e-12)
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="20" y="28" font-family="Arial, sans-serif" font-size="18" font-weight="700">Top atom subspace pairs</text>',
    ]
    x0 = 130
    bar_w = 690
    for idx, row in enumerate(rows):
        y = top + idx * row_h
        value = float(row["estimated_lagrangian_net_proxy_sum"])
        w = bar_w * value / max_value
        label = html.escape(str(row["key"]))
        lines.append(f'<text x="20" y="{y + 17}" font-family="Arial, sans-serif" font-size="12">pair {label}</text>')
        lines.append(f'<rect x="{x0}" y="{y}" width="{w:.3f}" height="18" fill="#7c3aed"/>')
        lines.append(f'<text x="{x0 + w + 8:.3f}" y="{y + 14}" font-family="Arial, sans-serif" font-size="11">{_fmt(value, digits=9)} / hits {row["hit_count"]}</text>')
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def _write_matplotlib_figures(
    *,
    evals: list[Mapping[str, Any]],
    profile_summary: Mapping[str, Any] | None,
    output_dir: Path,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "available": False,
        "error": None,
        "generated_files": [],
        "purpose": (
            "Optional report-quality PNG figures. These are observability only; "
            "score truth remains exact CUDA JSON."
        ),
    }
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - depends on optional extra
        record["error"] = f"{type(exc).__name__}: {exc}"
        return record

    record["available"] = True

    labels = [str(item["label"]) for item in evals]
    seg = [float(item["score_seg_contribution"]) for item in evals]
    pose = [float(item["score_pose_contribution"]) for item in evals]
    rate = [float(item["rate_component"]) for item in evals]
    y = list(range(len(evals)))
    fig_height = max(3.2, 0.46 * len(evals) + 1.4)
    fig, ax = plt.subplots(figsize=(11, fig_height))
    ax.barh(y, seg, color="#3b82f6", label="SegNet")
    ax.barh(y, pose, left=seg, color="#ef4444", label="PoseNet")
    left_rate = [a + b for a, b in zip(seg, pose)]
    ax.barh(y, rate, left=left_rate, color="#22c55e", label="Rate")
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Official score contribution")
    ax.set_title("Exact CUDA component breakdown")
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "score_breakdown.png", dpi=180)
    plt.close(fig)
    record["generated_files"].append("score_breakdown.png")

    if profile_summary is not None:
        pairs = list(profile_summary.get("top_pairs", []))[:20]
        if pairs:
            labels = [f"pair {row['key']}" for row in pairs]
            values = [float(row["estimated_lagrangian_net_proxy_sum"]) for row in pairs]
            hits = [int(row["hit_count"]) for row in pairs]
            fig, ax = plt.subplots(figsize=(11, max(3.2, 0.38 * len(pairs) + 1.2)))
            y = list(range(len(pairs)))
            ax.barh(y, values, color="#7c3aed")
            ax.set_yticks(y, labels)
            ax.invert_yaxis()
            ax.set_xlabel("Lagrangian net proxy")
            ax.set_title("Fridrich/Yousfi active pairs: score-density hotspots")
            for idx, (value, hit_count) in enumerate(zip(values, hits)):
                ax.text(value, idx, f"  hits {hit_count}", va="center", fontsize=8)
            ax.grid(axis="x", alpha=0.25)
            fig.tight_layout()
            fig.savefig(output_dir / "top_pairs.png", dpi=180)
            plt.close(fig)
            record["generated_files"].append("top_pairs.png")

            centroid_rows = [
                row for row in pairs
                if isinstance(row.get("weighted_centroid"), Mapping)
                and row["weighted_centroid"].get("x") is not None
                and row["weighted_centroid"].get("y") is not None
            ]
            if centroid_rows:
                xs = [float(row["weighted_centroid"]["x"]) for row in centroid_rows]
                ys = [float(row["weighted_centroid"]["y"]) for row in centroid_rows]
                masses = [
                    max(float(row["estimated_marginal_score_saved_proxy_sum"]), 1e-12)
                    for row in centroid_rows
                ]
                max_mass = max(masses)
                sizes = [80.0 + 520.0 * mass / max_mass for mass in masses]
                colors = list(range(len(centroid_rows)))
                fig, ax = plt.subplots(figsize=(8.4, 6.2))
                scatter = ax.scatter(xs, ys, s=sizes, c=colors, cmap="viridis", alpha=0.72, edgecolors="#111827")
                for row, x, y0 in zip(centroid_rows[:10], xs[:10], ys[:10]):
                    ax.text(x + 3, y0 + 2, f"p{row['key']}", fontsize=8)
                ax.set_xlim(0, 512)
                ax.set_ylim(384, 0)
                ax.set_xlabel("mask x")
                ax.set_ylabel("mask y")
                ax.set_title("Spatial hotspot map: weighted atom centroids")
                ax.grid(alpha=0.18)
                fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04, label="rank order")
                fig.tight_layout()
                fig.savefig(output_dir / "pair_centroid_hotspots.png", dpi=180)
                plt.close(fig)
                record["generated_files"].append("pair_centroid_hotspots.png")

        confusions = list(profile_summary.get("top_confusions", []))[:16]
        if confusions:
            labels = [str(row["key"]) for row in confusions]
            score_density = [float(row["estimated_score_saved_per_charged_byte"]) for row in confusions]
            pixels = [int(row["residual_pixels_sum"]) for row in confusions]
            fig, ax1 = plt.subplots(figsize=(11, 5.2))
            x = list(range(len(confusions)))
            ax1.bar(x, score_density, color="#0ea5e9", label="score proxy per byte")
            ax1.set_ylabel("score proxy per charged byte")
            ax1.set_xticks(x, labels, rotation=45, ha="right")
            ax1.grid(axis="y", alpha=0.25)
            ax2 = ax1.twinx()
            ax2.plot(x, pixels, color="#f97316", marker="o", label="residual pixels")
            ax2.set_ylabel("residual pixels")
            ax1.set_title("Class-confusion steganalysis surface")
            fig.tight_layout()
            fig.savefig(output_dir / "class_confusions.png", dpi=180)
            plt.close(fig)
            record["generated_files"].append("class_confusions.png")

    return record


def _profile_summary(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    path = path.resolve()
    payload = _read_json(path)
    profiles = payload.get("profiles")
    if not isinstance(profiles, list) or not profiles:
        raise ReportError(f"{path} has no profiles")
    aggregate = payload.get("aggregate_subspaces")
    subspaces: Mapping[str, Any] | None = aggregate if isinstance(aggregate, Mapping) else None
    summary_source = "aggregate_subspaces" if subspaces is not None else "profiles[0].subspaces"
    if subspaces is None:
        first = profiles[0]
        subspaces = first.get("subspaces") if isinstance(first, Mapping) else None
    if not isinstance(subspaces, Mapping):
        raise ReportError(f"{path} first profile has no subspaces")
    return {
        "evidence_grade": payload.get("evidence_grade"),
        "input": {"path": str(path), "sha256": _sha256_file(path), "size_bytes": path.stat().st_size},
        "score_claim": bool(payload.get("score_claim", False)),
        "signal_surface": payload.get("fridrich_yousfi_signal_surface"),
        "summary_source": summary_source,
        "top_classes": list(subspaces.get("classes", []))[:12],
        "top_confusions": list(subspaces.get("source_to_candidate", []))[:12],
        "top_pairs": list(subspaces.get("pairs", []))[:16],
    }


def build_report(
    *,
    eval_specs: Iterable[tuple[str, Path]],
    output_dir: Path,
    atom_profile_json: Path | None = None,
    title: str = "Yousfi-Fridrich Observability Report",
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    evals = [_eval_record(label, path) for label, path in eval_specs]
    if not evals:
        raise ReportError("at least one --eval is required")
    evals.sort(key=lambda item: (float(item["score_recomputed_from_components"]), str(item["label"])))
    _add_eval_reference_signals(evals)
    profile = _profile_summary(atom_profile_json)
    target_gap = _target_gap_analysis(evals, target_score=DEFAULT_TARGET_SCORE)
    recommendations = _action_recommendations(
        evals=evals,
        profile_summary=profile,
        target_gap=target_gap,
    )
    matplotlib_record = _write_matplotlib_figures(
        evals=evals,
        profile_summary=profile,
        output_dir=output_dir,
    )
    figure_files: dict[str, Any] = {
        "html": "index.html",
        "markdown": "observability_report.md",
        "score_breakdown_svg": "score_breakdown.svg",
        "target_gap_svg": "target_gap.svg",
        "top_pairs_svg": "top_pairs.svg",
    }
    if matplotlib_record.get("generated_files"):
        figure_files["matplotlib_png"] = list(matplotlib_record["generated_files"])
    payload: dict[str, Any] = {
        "action_recommendations": recommendations,
        "atom_profile_summary": profile,
        "canonical_score_source_required": "archive.zip -> inflate.sh -> upstream/evaluate.py on CUDA",
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "exact_evals": evals,
        "evidence_grade": "observability",
        "figure_files": figure_files,
        "matplotlib": matplotlib_record,
        "no_score_claim": True,
        "promotion_eligible": False,
        "schema": SCHEMA,
        "score_claim": False,
        "target_gap_analysis": target_gap,
        "title": title,
        "tool": TOOL,
    }
    markdown = _markdown(payload)
    (output_dir / "observability_report.json").write_text(_json_text(payload), encoding="utf-8")
    (output_dir / "observability_report.md").write_text(markdown + "\n", encoding="utf-8")
    (output_dir / "index.html").write_text(_html_doc(payload, markdown), encoding="utf-8")
    (output_dir / "score_breakdown.svg").write_text(_svg_score_breakdown(evals), encoding="utf-8")
    (output_dir / "target_gap.svg").write_text(_svg_target_gap(target_gap), encoding="utf-8")
    (output_dir / "top_pairs.svg").write_text(_svg_top_pairs(profile), encoding="utf-8")
    return payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval", action="append", type=_parse_eval_spec, required=True)
    parser.add_argument("--atom-profile-json", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--title", default="Yousfi-Fridrich Observability Report")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    payload = build_report(
        eval_specs=args.eval,
        output_dir=args.output_dir,
        atom_profile_json=args.atom_profile_json,
        title=args.title,
    )
    print(
        _json_text(
            {
                "eval_count": len(payload["exact_evals"]),
                "output_dir": str(args.output_dir),
                "schema": payload["schema"],
                "score_claim": payload["score_claim"],
            }
        ),
        end="",
    )


if __name__ == "__main__":
    main()
