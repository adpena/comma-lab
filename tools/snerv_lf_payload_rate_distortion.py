#!/usr/bin/env python3
"""CLI: evaluator-conditioned reverse-waterfill plan over an SNeRV LF/source-state payload.

THE LAW (computed per candidate action by the reusable core):

    keep payload component c  iff  -ΔS_distortion(c) > 25·Δbytes(c) / 37,545,489
    where ΔS_distortion = 100·Δd_seg + Δsqrt(10·d_pose)

This thin CLI delegates to ``tac.optimization.lf_payload_rate_distortion``. It
consumes THREE inputs and emits ONE ranked plan JSON:

  (a) --g1b-verdict : the G1b export-binding section-bytes JSON
      (``snerv_g1b_export_binding_verdict.v1`` from g1b_export_binding_measure.py).
      We read ``path_a_advisory.byte_decomposition`` for the per-section bytes and
      ``path_a_advisory.archive_surface_distortion`` for the receiver baseline
      d_seg/d_pose (apples-to-apples archive surface, per the G1b reframe). Read-only.

  (b) --atlas : the scorer spectral atlas JSON
      (``scorer_spectral_sensitivity.v2`` from measure_scorer_spectral_sensitivity.py).
      We read ``grid`` (the measurement scope envelope) + ``cells`` (H_seg/H_pose).

  (c) --section-map : (optional) JSON mapping each byte-decomposition section name to
      its atlas-grid CoefficientGroup (band_indices / channel_basis / channel /
      orientation / frame_incidence / amplitude_lsb / recodeable_floor_bytes /
      droppable). When omitted, a conservative DEFAULT map is used (see below) and
      every section that the default cannot place is emitted scope-invalid (the only
      honest move is an exact re-measure — never a fabricated estimate).

Output: a ``snerv_lf_payload_rd_plan.v1`` ranked-plan JSON. Authority:
``[macOS-CPU advisory]`` / planning-control false authority. PROPOSAL ONLY — every
ranked row is a PREDICTION requiring an exact receiver re-measure before admission.

NEVER /tmp for persisted output. NOT a score claim; ``promotable=False``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from tac.optimization.lf_payload_rate_distortion import (  # noqa: E402
    AtlasScope,
    BaselineScoreTerms,
    CoefficientGroup,
    LfPayloadRateDistortionError,
    PayloadSection,
    atlas_scope_from_grid,
    atlas_sensitivities_from_cells,
    plan_lf_payload_actions,
)

# The G1b byte-decomposition section names (g1b_export_binding_measure.py:177-195).
# Mapping each to whether it is a SCORE-RELEVANT coefficient-bearing section the
# atlas could place, vs a contest-required / metadata section that is not droppable
# and not atlas-placeable by default.
_DEFAULT_SECTION_BYTE_KEYS = (
    "lf_payload_bytes",
    "linf_steps_payload_bytes",
    "decoder_bytes",
    "metadata_bytes",
    "receiver_archive_header_bytes",
)


def _default_section_map() -> dict[str, dict[str, Any]]:
    """Conservative DEFAULT section -> CoefficientGroup placement.

    By design this places ONLY what we can defend, and leaves everything else
    UNPLACED so the core fails closed (scope-invalid -> needs_exact_remeasure)
    rather than fabricating a spectral identity. The LF payload is the low-band
    (band 0) wavelet content — the most defensible placement — but even that is
    declared agnostic on channel/orientation/incidence (empty strings match any
    measured cell) so the operator's real --section-map can refine it. The step-map
    + decoder + metadata + header sections are intentionally LEFT UNPLACED here:
    placing them requires real measurement, not a default guess.
    """
    return {
        # LF wavelet payload: band 0 (lowest), agnostic on the other axes until a
        # real --section-map declares them. amplitude_lsb omitted (matches any swept).
        "lf_payload_bytes": {
            "band_indices": [0],
            "droppable": True,
        },
    }


def _coefficient_group_from_spec(spec: dict[str, Any]) -> CoefficientGroup:
    bands = spec.get("band_indices")
    if not bands:
        raise LfPayloadRateDistortionError(
            "section-map entry must declare non-empty 'band_indices'"
        )
    amp = spec.get("amplitude_lsb")
    return CoefficientGroup(
        band_indices=tuple(int(b) for b in bands),
        channel_basis=str(spec.get("channel_basis", "")),
        channel=str(spec.get("channel", "")),
        orientation=str(spec.get("orientation", "")),
        frame_incidence=str(spec.get("frame_incidence", "")),
        amplitude_lsb=(float(amp) if amp is not None else None),
    )


def _sections_from_g1b(
    byte_decomposition: dict[str, Any],
    section_map: dict[str, dict[str, Any]],
) -> list[PayloadSection]:
    """Build PayloadSection rows from the G1b byte decomposition + the section map.

    Only sections that (a) have positive bytes AND (b) appear in the section_map are
    built as atlas-placeable PayloadSections. Sections present in the byte
    decomposition but absent from the map are reported in the plan's
    ``unmapped_sections`` (the operator must supply a --section-map to place them;
    we do NOT guess).
    """
    sections: list[PayloadSection] = []
    for key in _DEFAULT_SECTION_BYTE_KEYS + tuple(
        k for k in byte_decomposition if k.endswith("_bytes")
    ):
        if key not in byte_decomposition:
            continue
        try:
            section_bytes = int(byte_decomposition[key])
        except (TypeError, ValueError):
            continue
        if section_bytes <= 0:
            continue
        spec = section_map.get(key)
        if spec is None:
            continue
        group = _coefficient_group_from_spec(spec)
        floor = spec.get("recodeable_floor_bytes")
        sections.append(
            PayloadSection(
                name=key,
                bytes=section_bytes,
                coefficient_group=group,
                recodeable_floor_bytes=(int(floor) if floor is not None else None),
                droppable=bool(spec.get("droppable", True)),
            )
        )
    # De-dup (the key tuple may repeat keys); keep first occurrence per name.
    seen: set[str] = set()
    unique: list[PayloadSection] = []
    for s in sections:
        if s.name in seen:
            continue
        seen.add(s.name)
        unique.append(s)
    return unique


def _baseline_from_g1b(
    g1b: dict[str, Any], byte_decomposition: dict[str, Any]
) -> BaselineScoreTerms:
    """Extract the receiver-surface baseline (archive surface d_seg/d_pose + bytes)."""
    advisory = g1b.get("path_a_advisory") or {}
    surface = advisory.get("archive_surface_distortion") or {}
    # Prefer the L-inf archive surface (the carrier's byte-closed advisory surface).
    d_seg = surface.get("d_seg_mean_linf")
    d_pose = surface.get("d_pose_mean_linf")
    if d_seg is None or d_pose is None:
        # Fall back to the independent inflate confirm if the advisory surface is absent.
        confirm = g1b.get("independent_inflate_confirm") or {}
        d_seg = confirm.get("independent_inflate_d_seg", d_seg)
        d_pose = confirm.get("independent_inflate_d_pose", d_pose)
    if d_seg is None or d_pose is None:
        raise LfPayloadRateDistortionError(
            "G1b verdict has no archive-surface d_seg/d_pose "
            "(path_a_advisory.archive_surface_distortion or "
            "independent_inflate_confirm). Cannot baseline; supply --baseline-d-seg/"
            "--baseline-d-pose explicitly."
        )
    archive_bytes = byte_decomposition.get("archive_bytes_total_linf")
    if archive_bytes is None:
        archive_bytes = byte_decomposition.get("archive_bytes_total_l2", 0)
    return BaselineScoreTerms(
        d_seg=float(d_seg),
        d_pose=float(d_pose),
        archive_bytes=int(archive_bytes or 0),
        axis_tag=str(g1b.get("axis_tag", "[macOS-CPU advisory]")),
    )


def build_plan_from_files(
    *,
    g1b_verdict_path: str,
    atlas_path: str,
    section_map_path: str | None = None,
    baseline_d_seg: float | None = None,
    baseline_d_pose: float | None = None,
    baseline_archive_bytes: int | None = None,
    quantize_steps: tuple[float, ...] = (0.5, 1.0, 2.0),
) -> dict[str, Any]:
    """Read the three inputs, build typed inputs, and return the ranked plan dict."""
    g1b = json.loads(Path(g1b_verdict_path).read_text())
    atlas = json.loads(Path(atlas_path).read_text())

    scope: AtlasScope = atlas_scope_from_grid(atlas)
    sensitivities = atlas_sensitivities_from_cells(atlas)

    advisory = g1b.get("path_a_advisory") or {}
    byte_decomposition = dict(advisory.get("byte_decomposition") or {})
    if not byte_decomposition:
        raise LfPayloadRateDistortionError(
            "G1b verdict has no path_a_advisory.byte_decomposition; cannot plan."
        )

    if section_map_path is not None:
        section_map = json.loads(Path(section_map_path).read_text())
        if not isinstance(section_map, dict):
            raise LfPayloadRateDistortionError(
                "--section-map JSON must be an object mapping section name -> spec"
            )
    else:
        section_map = _default_section_map()

    sections = _sections_from_g1b(byte_decomposition, section_map)

    # Baseline: explicit CLI overrides win; else derive from the G1b verdict.
    if baseline_d_seg is not None and baseline_d_pose is not None:
        baseline = BaselineScoreTerms(
            d_seg=float(baseline_d_seg),
            d_pose=float(baseline_d_pose),
            archive_bytes=int(
                baseline_archive_bytes
                if baseline_archive_bytes is not None
                else (
                    byte_decomposition.get("archive_bytes_total_linf")
                    or byte_decomposition.get("archive_bytes_total_l2", 0)
                )
            ),
            axis_tag=str(g1b.get("axis_tag", "[macOS-CPU advisory]")),
        )
    else:
        baseline = _baseline_from_g1b(g1b, byte_decomposition)

    plan = plan_lf_payload_actions(
        sections, sensitivities, scope, baseline, quantize_steps=quantize_steps
    )
    # Provenance + the unmapped sections (operator must supply a --section-map).
    mapped_names = {s.name for s in sections}
    unmapped = sorted(
        k
        for k in byte_decomposition
        if k.endswith("_bytes")
        and isinstance(byte_decomposition.get(k), (int, float))
        and int(byte_decomposition[k] or 0) > 0
        and k not in mapped_names
        and not k.startswith("archive_bytes_total")
    )
    plan["inputs"] = {
        "g1b_verdict_path": str(g1b_verdict_path),
        "g1b_schema": g1b.get("schema"),
        "atlas_path": str(atlas_path),
        "atlas_schema": atlas.get("schema"),
        "atlas_authority_tier": atlas.get("authority_tier"),
        "section_map_path": section_map_path,
        "section_map_is_default": section_map_path is None,
        "n_sections_mapped": len(sections),
        "unmapped_sections": unmapped,
        "unmapped_note": (
            "Sections present in the G1b byte decomposition but NOT placed in the "
            "atlas grid. Supply --section-map to place them; the planner refuses to "
            "guess a spectral identity (fail-closed scope rule)."
        ),
    }
    return plan


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluator-conditioned reverse-waterfill plan over an SNeRV LF payload. "
            "PROPOSAL ONLY: ranked actions are PREDICTIONS requiring exact re-measure."
        )
    )
    parser.add_argument(
        "--g1b-verdict",
        required=True,
        help="Path to snerv_g1b_export_binding_verdict.v1 JSON (read-only).",
    )
    parser.add_argument(
        "--atlas",
        required=True,
        help="Path to scorer_spectral_sensitivity.v2 atlas JSON (read-only).",
    )
    parser.add_argument(
        "--section-map",
        default=None,
        help=(
            "Optional JSON mapping section name -> CoefficientGroup spec "
            "(band_indices/channel_basis/channel/orientation/frame_incidence/"
            "amplitude_lsb/recodeable_floor_bytes/droppable). Omit for the "
            "conservative default (only the LF payload is placed; rest scope-invalid)."
        ),
    )
    parser.add_argument("--baseline-d-seg", type=float, default=None)
    parser.add_argument("--baseline-d-pose", type=float, default=None)
    parser.add_argument("--baseline-archive-bytes", type=int, default=None)
    parser.add_argument(
        "--quantize-steps",
        default="0.5,1.0,2.0",
        help="Comma-separated quantization steps to propose per section.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to write the ranked plan JSON (durable, NEVER /tmp). "
        "If omitted, prints to stdout.",
    )
    args = parser.parse_args(argv)

    if args.output is not None and str(args.output).startswith("/tmp"):
        parser.error("--output must be durable (not /tmp) per CLAUDE.md disk hygiene")

    steps = tuple(
        float(x) for x in str(args.quantize_steps).split(",") if x.strip() != ""
    )

    try:
        plan = build_plan_from_files(
            g1b_verdict_path=args.g1b_verdict,
            atlas_path=args.atlas,
            section_map_path=args.section_map,
            baseline_d_seg=args.baseline_d_seg,
            baseline_d_pose=args.baseline_d_pose,
            baseline_archive_bytes=args.baseline_archive_bytes,
            quantize_steps=steps,
        )
    except (LfPayloadRateDistortionError, FileNotFoundError, json.JSONDecodeError) as exc:
        print(f"[snerv_lf_rd] FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    text = json.dumps(plan, indent=2, sort_keys=True, default=str)
    if args.output is not None:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text)
        print(
            f"[snerv_lf_rd] WROTE {out_path} "
            f"(n_ranked={plan['n_ranked']} best={plan['best_action_id']} "
            f"[macOS-CPU advisory] PROPOSAL-ONLY)"
        )
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
