#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""WAVE-3-NULL-BYTE-PROBE-MATRIX: extend null-byte probe across all master-gradient
anchors per task #1111 + operator NULL-EXPLOIT framing 2026-05-20.

Reads ``.omx/state/master_gradient_anchors.jsonl`` and runs the canonical
single-anchor probe (``tools.probe_null_byte_master_gradient.probe_null_bytes``)
on every row, then emits:

* a per-anchor JSON matrix at ``<output_dir>/null_byte_matrix.json``
* a per-archive markdown comparison table at ``<output_dir>/null_byte_matrix.md``
* codec-family rollups (HNeRV-family vs non-HNeRV + per-substrate-class mean+stddev)
* cross-hardware drift detection across {macOS-CPU advisory, contest-CPU, contest-CUDA}
* a seed-budget x substrate predicted-savings matrix
  (``25 * (null_bytes - K) / 37545489`` per the contest rate formula)

The tool is OBSERVABILITY-ONLY per Catalog #318 + #323 (every emitted row carries
``score_claim=False`` + ``promotion_eligible=False`` + ``promotable=False``
+ ``axis_tag="[predicted]"``).
It does NOT propose mutations and does NOT bypass typed
``CandidateModificationSpec`` discipline.

Sister of:
  * ``tools/probe_null_byte_master_gradient.py`` (canonical single-anchor probe)
  * ``src/tac/canonical_equations/null_space_byte_fraction.py`` (canonical equation)
  * ``tac.cathedral_consumers.null_byte_codebook_candidate_consumer`` (Tier A consumer)
  * ``tac.procedural_codebook_generator`` (PROPOSED Q5 follow-on consumer)

Usage::

    .venv/bin/python tools/probe_null_byte_master_gradient_matrix.py \\
        --anchors-jsonl .omx/state/master_gradient_anchors.jsonl \\
        --epsilon 1e-9 \\
        --output-dir experiments/results/null_byte_probe_matrix_<utc>
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import statistics
import sys
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

# Delegate single-anchor probe to canonical helper. Make the sibling-tool
# import robust whether the file runs from repo root (``python tools/...``),
# from inside ``tools/`` itself, or as a package via ``python -m tools.X``.
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))
try:
    from tools.probe_null_byte_master_gradient import (  # type: ignore[import-not-found]
        probe_null_bytes,
        parse_fec_grammar_from_inner_bytes,
    )
except ModuleNotFoundError:
    from probe_null_byte_master_gradient import (  # type: ignore[no-redef]
        probe_null_bytes,
        parse_fec_grammar_from_inner_bytes,
    )

# Per CLAUDE.md "Frontier scores are pointer-only" + canonical contest formula
_CANONICAL_RATE_DENOM_BYTES = 37_545_489
_CANONICAL_RATE_MULTIPLIER = 25.0
_DEFAULT_SEED_BUDGETS_K = (16, 32, 64, 128, 256)

# Codec-family classification heuristics — sister of the substrate-id-substring
# sets in Catalog #220 / #272 / #298 / #315.
_HNERV_FAMILY_SHA_PREFIXES = {
    "b83bf34886",  # pr101_lc_v2 (HNeRV-family)
    "6bae0201fb",  # pr101_fec6_frontier (HNeRV-family + fec6 selector)
    "87ec7ca5f2",  # a1_finetuned (HNeRV-family A1)
    "f174192aea",  # fec6 scored archive (HNeRV-family + fec6)
}
_PR106_FAMILY_SHA_PREFIXES = {
    "9cb989cef5",  # pr106_format0d
}
_PR107_FAMILY_SHA_PREFIXES = {
    "7ecb0df1c4",  # pr107_apogee
}


def _classify_codec_family(archive_sha256: str) -> str:
    prefix = archive_sha256[:10]
    if prefix in _HNERV_FAMILY_SHA_PREFIXES:
        return "hnerv_family"
    if prefix in _PR106_FAMILY_SHA_PREFIXES:
        return "pr106_format0d_family"
    if prefix in _PR107_FAMILY_SHA_PREFIXES:
        return "pr107_apogee_family"
    return "unknown_family"


def _classify_substrate_label(archive_sha256: str) -> str:
    prefix = archive_sha256[:10]
    if prefix == "6bae0201fb":
        return "pr101_fec6_frontier"
    if prefix == "b83bf34886":
        return "pr101_lc_v2"
    if prefix == "87ec7ca5f2":
        return "a1_finetuned"
    if prefix == "f174192aea":
        return "fec6_subject_sha"
    if prefix == "9cb989cef5":
        return "pr106_format0d"
    if prefix == "7ecb0df1c4":
        return "pr107_apogee"
    return f"unknown_{prefix}"


def _load_inner_member_bytes_if_available(scored_archive_sha256: str | None) -> bytes | None:
    """Best-effort: scan canonical experiments/results dirs for a matching archive.zip.

    We only need this for section bucketing on fec6 archives; non-fec6
    archives skip gracefully via `parse_fec_grammar_from_inner_bytes` -> None.
    """
    if not scored_archive_sha256:
        return None
    repo_root = Path(__file__).resolve().parent.parent
    results_root = repo_root / "experiments" / "results"
    if not results_root.is_dir():
        return None
    # Bounded scan: try the known fec6 frontier path first (most common anchor)
    candidate_dirs = []
    for p in results_root.glob("pr101_frame_exploit_selector_fec6_*"):
        if p.is_dir():
            candidate_dirs.append(p)
    for p in results_root.glob("*pr101_lc_v2*"):
        if p.is_dir():
            candidate_dirs.append(p)
    for cand_dir in candidate_dirs[:10]:  # bound: <=10 candidates per anchor
        archive_zip = cand_dir / "archive.zip"
        if not archive_zip.is_file():
            archive_zip = cand_dir / "submission_dir" / "archive.zip"
        if not archive_zip.is_file():
            continue
        try:
            with zipfile.ZipFile(archive_zip) as zf:
                # Verify match by sha256 of full zip
                sha = hashlib.sha256(archive_zip.read_bytes()).hexdigest()
                if sha != scored_archive_sha256:
                    continue
                names = [info.filename for info in zf.infolist()]
                if len(names) == 1:
                    with zf.open(names[0]) as f:
                        return f.read()
        except (zipfile.BadZipFile, OSError):
            continue
    return None


def _build_canonical_provenance_dict(
    *,
    anchors_jsonl_path: str,
    anchors_jsonl_sha256: str,
) -> dict[str, Any]:
    """Per Catalog #323 canonical Provenance — predicted/observability-only."""
    return {
        "artifact_kind": "PREDICTED_FROM_MODEL",
        "model_id": "null_byte_master_gradient_probe_matrix.v1",
        "inputs_sha256": anchors_jsonl_sha256,
        "measurement_axis": "[predicted]",
        "hardware_substrate": "unknown",
        "evidence_grade": "predicted",
        "captured_at_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "canonical_helper_invocation": "tools/probe_null_byte_master_gradient_matrix.py",
        "source_anchors_jsonl": anchors_jsonl_path,
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "promotable": False,
        "axis_tag": "[predicted]",
    }


def _per_anchor_replacement_priority(null_bytes: int, archive_bytes: int) -> float:
    """Proxy ranking score = null_fraction * archive_bytes.

    Higher = more replacement budget available for procedural-codebook generation.
    Pure observability metric (not a score signal per Catalog #341).
    """
    if archive_bytes <= 0:
        return 0.0
    return (null_bytes / archive_bytes) * archive_bytes  # = null_bytes (intentional)


def _predicted_delta_s_per_seed_budget(
    null_bytes: int,
    seed_budgets_k: tuple[int, ...] = _DEFAULT_SEED_BUDGETS_K,
) -> dict[str, float]:
    """Per CLAUDE.md contest formula: rate term = 25 * archive_bytes / 37_545_489.

    If we replace N null bytes with a K-byte PRNG seed, score reduction is
    ``25 * (null_bytes - K) / 37_545_489`` (negative ΔS = lower score = better).
    """
    out: dict[str, float] = {}
    for k in seed_budgets_k:
        savings_bytes = null_bytes - k
        if savings_bytes <= 0:
            out[f"K={k}"] = 0.0
        else:
            out[f"K={k}"] = -float(_CANONICAL_RATE_MULTIPLIER * savings_bytes) / _CANONICAL_RATE_DENOM_BYTES
    return out


def probe_all_anchors(
    *,
    anchors_jsonl_path: Path,
    epsilon: float = 1e-9,
) -> dict[str, Any]:
    """Probe every anchor in the canonical ledger; return matrix dict."""
    if not anchors_jsonl_path.is_file():
        raise FileNotFoundError(f"anchors ledger not found: {anchors_jsonl_path}")
    rows: list[dict[str, Any]] = []
    with anchors_jsonl_path.open() as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            rows.append(json.loads(raw))

    per_anchor_results: list[dict[str, Any]] = []
    seen_paths: dict[str, dict[str, Any]] = {}  # cache probe results per unique npy
    for idx, row in enumerate(rows, start=1):
        npy_path = Path(row["gradient_array_path"])
        archive_sha = row.get("archive_sha256", "")
        scored_archive_sha = row.get("scored_archive_sha256", archive_sha) or archive_sha
        axis = row.get("measurement_axis", "[unknown]")
        hardware = row.get("measurement_hardware", "unknown")
        n_pairs_used = row.get("n_pairs_used", row.get("n_pairs", None))
        n_bytes_declared = row.get("n_bytes", None)

        if not npy_path.is_file():
            per_anchor_results.append({
                "anchor_index": idx,
                "archive_sha256": archive_sha,
                "scored_archive_sha256": scored_archive_sha,
                "axis": axis,
                "hardware": hardware,
                "n_pairs_used": n_pairs_used,
                "n_bytes_declared": n_bytes_declared,
                "npy_path": str(npy_path),
                "status": "MISSING_NPY",
                "codec_family": _classify_codec_family(scored_archive_sha or ""),
                "substrate_label": _classify_substrate_label(scored_archive_sha or ""),
            })
            continue

        # Cache by (npy_path, epsilon) to avoid redundant probes; rows 1+2 share file
        cache_key = str(npy_path.resolve())
        if cache_key in seen_paths:
            cached = seen_paths[cache_key]
            probe_summary = cached
        else:
            grad = np.load(npy_path)
            inner_bytes = _load_inner_member_bytes_if_available(scored_archive_sha)
            probe_summary = probe_null_bytes(
                grad=grad, epsilon=epsilon, inner_bytes=inner_bytes,
            )
            # Drop the raw indices array from cache (very large) — keep summary only
            probe_summary = {k: v for k, v in probe_summary.items() if k != "null_indices"}
            seen_paths[cache_key] = probe_summary

        codec_family = _classify_codec_family(scored_archive_sha or "")
        substrate_label = _classify_substrate_label(scored_archive_sha or "")
        n_total = probe_summary["n_total_bytes"]
        n_null = probe_summary["n_null_bytes"]
        per_axis = probe_summary["per_axis_zero_counts"]
        per_anchor_results.append({
            "anchor_index": idx,
            "archive_sha256": archive_sha,
            "scored_archive_sha256": scored_archive_sha,
            "axis": axis,
            "hardware": hardware,
            "n_pairs_used": n_pairs_used,
            "n_bytes_declared": n_bytes_declared,
            "npy_path": str(npy_path),
            "status": "OK",
            "codec_family": codec_family,
            "substrate_label": substrate_label,
            "n_total_bytes": n_total,
            "n_null_bytes": n_null,
            "null_fraction": probe_summary["null_fraction"],
            "epsilon": probe_summary["epsilon"],
            "per_axis_zero_counts": per_axis,
            "section_breakdown": probe_summary["section_breakdown"],
            "grammar_detected_label": (
                probe_summary["grammar_detected"]["selector_magic"]
                if probe_summary.get("grammar_detected") else None
            ),
            "replacement_priority_proxy_null_bytes": n_null,
            "predicted_delta_s_per_seed_budget": _predicted_delta_s_per_seed_budget(n_null),
        })

    # --- Codec-family rollups
    family_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in per_anchor_results:
        if r["status"] != "OK":
            continue
        family_groups[r["codec_family"]].append(r)
    family_rollups: dict[str, dict[str, Any]] = {}
    for fam, members in family_groups.items():
        fracs = [m["null_fraction"] for m in members]
        bytes_ = [m["n_null_bytes"] for m in members]
        family_rollups[fam] = {
            "anchor_count": len(members),
            "null_fraction_mean": statistics.fmean(fracs),
            "null_fraction_stddev": (
                statistics.stdev(fracs) if len(fracs) >= 2 else 0.0
            ),
            "null_fraction_min": min(fracs),
            "null_fraction_max": max(fracs),
            "null_bytes_mean": statistics.fmean(bytes_),
            "null_bytes_total": sum(bytes_),
        }

    # --- Cross-hardware drift detection
    # Group by scored_archive_sha256 -> compare across axes
    sha_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in per_anchor_results:
        if r["status"] != "OK":
            continue
        sha_groups[r["scored_archive_sha256"]].append(r)
    cross_hardware_drift: list[dict[str, Any]] = []
    for sha, members in sha_groups.items():
        axes_present = sorted({m["axis"] for m in members})
        if len(axes_present) < 2:
            continue  # No cross-axis comparison possible
        per_axis_fractions: dict[str, list[float]] = defaultdict(list)
        for m in members:
            per_axis_fractions[m["axis"]].append(m["null_fraction"])
        per_axis_mean = {axis: statistics.fmean(vals) for axis, vals in per_axis_fractions.items()}
        fractions = list(per_axis_mean.values())
        spread = max(fractions) - min(fractions)
        cross_hardware_drift.append({
            "scored_archive_sha256": sha,
            "substrate_label": _classify_substrate_label(sha),
            "axes_present": axes_present,
            "per_axis_mean_null_fraction": per_axis_mean,
            "absolute_spread": spread,
            "relative_spread": (
                spread / max(fractions) if max(fractions) > 0 else 0.0
            ),
        })

    # --- Top-5 candidates ranked by null_bytes (replacement budget proxy)
    candidates = [r for r in per_anchor_results if r["status"] == "OK"]
    candidates_sorted = sorted(
        candidates, key=lambda r: r["n_null_bytes"], reverse=True
    )
    top5 = []
    seen_substrates: set[str] = set()
    for r in candidates_sorted:
        if r["substrate_label"] in seen_substrates:
            continue  # Deduplicate per-substrate (multi-anchor same archive)
        seen_substrates.add(r["substrate_label"])
        top5.append({
            "rank": len(top5) + 1,
            "substrate_label": r["substrate_label"],
            "codec_family": r["codec_family"],
            "scored_archive_sha256": r["scored_archive_sha256"],
            "axis": r["axis"],
            "n_null_bytes": r["n_null_bytes"],
            "null_fraction": r["null_fraction"],
            "predicted_delta_s_per_seed_budget": r["predicted_delta_s_per_seed_budget"],
            "anchor_index": r["anchor_index"],
        })
        if len(top5) >= 5:
            break

    return {
        "schema": "null_byte_master_gradient_probe_matrix_v1",
        "n_anchors_scanned": len(rows),
        "n_anchors_probed_ok": sum(1 for r in per_anchor_results if r["status"] == "OK"),
        "n_anchors_missing_npy": sum(
            1 for r in per_anchor_results if r["status"] == "MISSING_NPY"
        ),
        "epsilon": float(epsilon),
        "seed_budgets_k": list(_DEFAULT_SEED_BUDGETS_K),
        "per_anchor": per_anchor_results,
        "codec_family_rollups": family_rollups,
        "cross_hardware_drift_per_archive": cross_hardware_drift,
        "top5_replacement_candidates": top5,
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "promotable": False,
        "axis_tag": "[predicted]",
    }


def _render_markdown_matrix(matrix: dict[str, Any]) -> str:
    """Operator-readable per-archive table + rollups + cross-hardware + top-5."""
    lines: list[str] = []
    lines.append("# Null-byte master-gradient probe matrix (11 anchors)")
    lines.append("")
    lines.append(f"- Generated UTC: {_dt.datetime.now(_dt.timezone.utc).isoformat()}")
    lines.append(f"- Anchors scanned: {matrix['n_anchors_scanned']}")
    lines.append(f"- Probed OK: {matrix['n_anchors_probed_ok']}")
    lines.append(f"- Epsilon: {matrix['epsilon']}")
    lines.append(f"- Axis tag: {matrix['axis_tag']} (observability-only per Catalog #323)")
    lines.append("")
    lines.append("## Per-anchor results")
    lines.append("")
    lines.append(
        "| # | substrate | codec_family | axis | hardware | n_pairs | n_bytes | n_null | null_frac | seg_zero | pose_zero | rate_zero |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in matrix["per_anchor"]:
        if r["status"] != "OK":
            lines.append(
                f"| {r['anchor_index']} | {r['substrate_label']} | {r['codec_family']} | "
                f"{r['axis']} | {r['hardware']} | n/a | n/a | MISSING_NPY | - | - | - | - |"
            )
            continue
        per_axis = r["per_axis_zero_counts"]
        lines.append(
            f"| {r['anchor_index']} | {r['substrate_label']} | {r['codec_family']} | "
            f"{r['axis']} | {r['hardware']} | {r['n_pairs_used']} | {r['n_total_bytes']} | "
            f"{r['n_null_bytes']} | {r['null_fraction']*100:.2f}% | "
            f"{per_axis['seg_axis_zero_count']} | {per_axis['pose_axis_zero_count']} | "
            f"{per_axis['rate_axis_zero_count']} |"
        )
    lines.append("")
    lines.append("## Codec-family rollups")
    lines.append("")
    lines.append("| family | n_anchors | null_frac_mean | null_frac_stddev | null_frac_min | null_frac_max | null_bytes_total |")
    lines.append("|---|---|---|---|---|---|---|")
    for fam, stats in sorted(matrix["codec_family_rollups"].items()):
        lines.append(
            f"| {fam} | {stats['anchor_count']} | "
            f"{stats['null_fraction_mean']*100:.2f}% | "
            f"{stats['null_fraction_stddev']*100:.4f}% | "
            f"{stats['null_fraction_min']*100:.2f}% | "
            f"{stats['null_fraction_max']*100:.2f}% | "
            f"{stats['null_bytes_total']} |"
        )
    lines.append("")
    lines.append("## Cross-hardware drift per archive")
    lines.append("")
    if not matrix["cross_hardware_drift_per_archive"]:
        lines.append("(No archive has multi-axis anchors; cross-hardware drift not computable.)")
    else:
        lines.append("| substrate | axes_present | per_axis_mean_null_fraction | abs_spread | rel_spread |")
        lines.append("|---|---|---|---|---|")
        for drift in matrix["cross_hardware_drift_per_archive"]:
            axes_str = ", ".join(drift["axes_present"])
            per_axis_str = ", ".join(
                f"{axis}={frac*100:.2f}%"
                for axis, frac in drift["per_axis_mean_null_fraction"].items()
            )
            lines.append(
                f"| {drift['substrate_label']} | {axes_str} | {per_axis_str} | "
                f"{drift['absolute_spread']*100:.4f}pp | "
                f"{drift['relative_spread']*100:.2f}% |"
            )
    lines.append("")
    lines.append("## Top-5 replacement candidates (deduplicated per substrate)")
    lines.append("")
    lines.append("| rank | substrate | family | n_null_bytes | null_frac | ΔS@K=16 | ΔS@K=32 | ΔS@K=64 | ΔS@K=128 | ΔS@K=256 |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for c in matrix["top5_replacement_candidates"]:
        ds = c["predicted_delta_s_per_seed_budget"]
        lines.append(
            f"| {c['rank']} | {c['substrate_label']} | {c['codec_family']} | "
            f"{c['n_null_bytes']} | {c['null_fraction']*100:.2f}% | "
            f"{ds.get('K=16', 0):+.6f} | {ds.get('K=32', 0):+.6f} | "
            f"{ds.get('K=64', 0):+.6f} | {ds.get('K=128', 0):+.6f} | "
            f"{ds.get('K=256', 0):+.6f} |"
        )
    lines.append("")
    lines.append("## Provenance (per Catalog #323)")
    lines.append("")
    lines.append("- `score_claim`: False")
    lines.append("- `promotion_eligible`: False")
    lines.append("- `rank_or_kill_eligible`: False")
    lines.append("- `promotable`: False")
    lines.append("- `axis_tag`: `[predicted]`")
    lines.append("- All ΔS predictions assume substituting null bytes with a K-byte PRNG seed")
    lines.append("  + the inflate-side deterministic codebook regeneration; actual contest-CUDA")
    lines.append("  measurement still required per CLAUDE.md \"Apples-to-apples evidence discipline\".")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Probe master-gradient null-space bytes across the full anchors ledger"
    )
    parser.add_argument(
        "--anchors-jsonl",
        type=Path,
        default=Path(".omx/state/master_gradient_anchors.jsonl"),
        help="canonical anchors ledger",
    )
    parser.add_argument("--epsilon", type=float, default=1e-9, help="null-byte threshold")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    if not args.anchors_jsonl.is_file():
        print(f"[null-byte-matrix] FATAL: anchors ledger not found: {args.anchors_jsonl}", file=sys.stderr)
        return 1

    matrix = probe_all_anchors(anchors_jsonl_path=args.anchors_jsonl, epsilon=args.epsilon)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_out = args.output_dir / "null_byte_matrix.json"
    md_out = args.output_dir / "null_byte_matrix.md"
    anchors_sha = hashlib.sha256(args.anchors_jsonl.read_bytes()).hexdigest()
    matrix_with_provenance = dict(matrix)
    matrix_with_provenance["provenance"] = _build_canonical_provenance_dict(
        anchors_jsonl_path=str(args.anchors_jsonl),
        anchors_jsonl_sha256=anchors_sha,
    )
    json_out.write_text(
        json.dumps(matrix_with_provenance, indent=2, sort_keys=True), encoding="utf-8"
    )
    md_out.write_text(_render_markdown_matrix(matrix), encoding="utf-8")

    print(
        f"[null-byte-matrix] [predicted] scanned={matrix['n_anchors_scanned']} "
        f"probed_ok={matrix['n_anchors_probed_ok']} families={len(matrix['codec_family_rollups'])} "
        f"cross_hw_archives={len(matrix['cross_hardware_drift_per_archive'])} "
        f"top5_dedup_substrates={len(matrix['top5_replacement_candidates'])}",
        file=sys.stderr,
    )
    print(f"[null-byte-matrix] wrote {json_out}", file=sys.stderr)
    print(f"[null-byte-matrix] wrote {md_out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
