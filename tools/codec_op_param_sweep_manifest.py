#!/usr/bin/env python3
"""CodecOp parameter-sweep manifest builder.

The cathedral has CodecOps (e.g. ``Op_KLPoseStream``, ``Op_RAFTPoseStream``)
each with one or two free knobs (``n_components``, ``brotli_quality``). To
fire a parallel dispatch on a sweep of those knobs, the operator needs:

  1. Per-knob-setting bytes-out from the CodecOp on a real substrate
     (so predicted score bands are calibrated, not hand-waved)
  2. Per-knob-setting predicted contest-score band, anchored on a
     reference (d_seg, d_pose) and the cathedral's canonical formula
  3. A JSON manifest in the shape ``parallel_dispatch_top_k.py`` consumes

This tool emits that manifest. It does NOT perform archive surgery
(substituting the CodecOp's encoded blob into a contest archive); that
remains the next blocking actuator step before the operator can run
``parallel_dispatch_top_k.py`` on the manifest.

The manifest's candidates are flagged
``ready_for_exact_eval_dispatch=False`` and ``evidence_semantics=
"cpu_substrate_predicted_band"`` because predicted-band evidence is
NOT a contest-CUDA result. To actually dispatch, the operator must:

  (a) perform archive-substitution surgery (substrate-specific, not
      this tool's scope)
  (b) flip ``ready_for_exact_eval_dispatch=True`` with an explicit
      operator override note

Usage::

    .venv/bin/python tools/codec_op_param_sweep_manifest.py \\
        --module tac.codec_pipeline_kl_pose \\
        --class Op_KLPoseStream \\
        --state-dict-path experiments/.../poses.pt \\
        --state-dict-key poses_se3 \\
        --param-grid '{"n_components": [2, 3, 4]}' \\
        --anchor-d-seg 0.000671 \\
        --anchor-d-pose 3.36e-5 \\
        --anchor-archive-bytes 185578 \\
        --baseline-substream-bytes 3600 \\
        --output reports/kl_pose_sweep_manifest.json

The ``--baseline-substream-bytes`` is the bytes the substrate currently
contributes to the archive (e.g. the existing pose blob's compressed
size). Predicted Δ-rate per candidate is
``(candidate_bytes - baseline_bytes) / RAW_VIDEO_BYTES * 25``.
"""
from __future__ import annotations

import argparse
import importlib
import itertools
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tac.contest_rate_distortion_system import (  # noqa: E402
    CONTEST_RATE_WEIGHT,
    CONTEST_RAW_VIDEO_BYTES,
    contest_score,
    contest_score_decomposition,
)


@dataclass
class SweepCandidate:
    candidate_id: str
    op_module: str
    op_class: str
    op_params: dict[str, Any]
    candidate_substream_bytes: int
    bytes_in: int
    bytes_out: int
    predicted_archive_bytes: int
    predicted_score: float
    predicted_decomposition: dict[str, float]
    predicted_band: list[float]  # [low, high] = [score, score+ uncertainty]
    score_delta_vs_anchor: float
    rate_delta_vs_anchor: float
    evidence_semantics: str = "cpu_substrate_predicted_band"
    ready_for_exact_eval_dispatch: bool = False
    score_claim: bool = False
    blockers: list[str] = field(default_factory=lambda: [
        "archive_substitution_surgery_pending",
        "operator_override_to_flip_ready_for_dispatch_required",
    ])


def _import_codec_op(module: str, class_name: str):
    """Dynamic import of a CodecOp class.

    Validates the class exposes encode/decode/validate per the CodecOp
    Protocol; relies on duck-typing rather than a runtime
    ``isinstance(cls, CodecOp)`` check (Protocol-typed classes don't
    always register cleanly under isinstance unless decorated).
    """
    try:
        mod = importlib.import_module(module)
    except ImportError as exc:
        raise SystemExit(f"could not import module {module!r}: {exc}") from None
    if not hasattr(mod, class_name):
        raise SystemExit(f"module {module!r} has no class {class_name!r}")
    cls = getattr(mod, class_name)
    for required in ("encode", "decode", "validate"):
        if not hasattr(cls, required):
            raise SystemExit(
                f"{module}.{class_name} missing CodecOp method {required!r}"
            )
    return cls


def _load_state_dict(path: Path, key: str | None) -> dict[str, torch.Tensor]:
    """Load a state_dict (or a single tensor under a key) from disk."""
    if not path.is_file():
        raise SystemExit(f"state_dict path does not exist: {path}")
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, dict):
        return obj
    if key is None:
        raise SystemExit(
            f"loaded {path} is a tensor not a dict; pass --state-dict-key to "
            f"specify the dict key to wrap it under"
        )
    if not isinstance(obj, torch.Tensor):
        raise SystemExit(
            f"loaded {path} is neither dict nor Tensor (got {type(obj).__name__})"
        )
    return {key: obj}


def _expand_param_grid(grid: dict[str, list]) -> list[dict[str, Any]]:
    """Cartesian product of the parameter grid → list of param dicts."""
    if not grid:
        return [{}]
    keys = sorted(grid.keys())
    value_lists = [grid[k] for k in keys]
    return [dict(zip(keys, combo)) for combo in itertools.product(*value_lists)]


def build_sweep_candidates(
    *,
    op_cls,
    op_module: str,
    op_class_name: str,
    state_dict: dict[str, torch.Tensor],
    param_grid: dict[str, list],
    anchor_d_seg: float,
    anchor_d_pose: float,
    anchor_archive_bytes: int,
    baseline_substream_bytes: int,
    label_prefix: str = "codec_op_sweep",
    band_uncertainty_pct: float = 0.005,
) -> list[SweepCandidate]:
    """Build a list of SweepCandidates by running the CodecOp at each
    parameter setting and computing predicted contest-score impact
    relative to the anchor.

    Args:
        op_cls: the CodecOp class (already imported).
        op_module: the module path (for manifest attribution).
        op_class_name: the class name (for manifest attribution).
        state_dict: the substrate state_dict to apply the CodecOp to.
        param_grid: dict of {param_name: [values, ...]} cartesian-product'd.
        anchor_d_seg / anchor_d_pose: the substrate's authoritative
            distortion components from a contest-CUDA anchor (e.g.
            PR103-on-PR106's 0.000671 / 3.36e-5).
        anchor_archive_bytes: the substrate's full archive bytes
            BEFORE substitution (e.g. PR103-on-PR106's 185,578).
        baseline_substream_bytes: the bytes the substrate's existing
            target stream contributes (e.g. PR101 pose blob ~3,600).
        label_prefix: candidate_id prefix.
        band_uncertainty_pct: predicted band half-width as fraction of
            the predicted score (0.005 = ±0.5%).

    Returns: list of SweepCandidate with predicted contest scores.
    """
    candidates: list[SweepCandidate] = []
    bytes_in = sum(t.numel() * t.element_size() for t in state_dict.values())
    for i, params in enumerate(_expand_param_grid(param_grid)):
        op = op_cls(**params)
        try:
            result = op.encode(state_dict, context={})
        except Exception as exc:  # noqa: BLE001 — surface op-side errors clearly
            raise SystemExit(
                f"CodecOp {op_class_name}({params}) encode failed: {exc}"
            ) from exc
        candidate_substream_bytes = result.bytes_out
        # Predicted archive bytes after substituting the substream.
        predicted_archive_bytes = (
            anchor_archive_bytes - baseline_substream_bytes + candidate_substream_bytes
        )
        # Predicted score using cathedral formula. Distortion components
        # are HELD CONSTANT at the anchor; only the rate term changes.
        # This is the cleanest predicted-band — assumes the codec
        # substitution preserves d_seg + d_pose (true at quantization
        # noise floor, false in general; flagged in evidence_semantics).
        predicted_score = float(contest_score(
            seg_distortion=anchor_d_seg,
            pose_distortion=anchor_d_pose,
            archive_bytes=predicted_archive_bytes,
        ))
        predicted_decomposition = contest_score_decomposition(
            seg_distortion=anchor_d_seg,
            pose_distortion=anchor_d_pose,
            archive_bytes=predicted_archive_bytes,
        )
        anchor_score = float(contest_score(
            seg_distortion=anchor_d_seg,
            pose_distortion=anchor_d_pose,
            archive_bytes=anchor_archive_bytes,
        ))
        rate_delta = (
            CONTEST_RATE_WEIGHT
            * (predicted_archive_bytes - anchor_archive_bytes)
            / CONTEST_RAW_VIDEO_BYTES
        )
        score_delta = predicted_score - anchor_score
        half_band = band_uncertainty_pct * abs(predicted_score)
        candidate_id = (
            f"{label_prefix}_"
            + "_".join(f"{k}{v}" for k, v in sorted(params.items()))
            or f"{label_prefix}_{i}"
        )
        candidates.append(SweepCandidate(
            candidate_id=candidate_id,
            op_module=op_module,
            op_class=op_class_name,
            op_params=params,
            candidate_substream_bytes=candidate_substream_bytes,
            bytes_in=bytes_in,
            bytes_out=result.bytes_out,
            predicted_archive_bytes=predicted_archive_bytes,
            predicted_score=predicted_score,
            predicted_decomposition=predicted_decomposition,
            predicted_band=[predicted_score - half_band, predicted_score + half_band],
            score_delta_vs_anchor=score_delta,
            rate_delta_vs_anchor=rate_delta,
        ))
    return candidates


def to_meta_lagrangian_candidates(
    candidates: list[SweepCandidate], *, lane_class: str | None = None
) -> list[dict[str, Any]]:
    """Convert sweep candidates to the strict schema consumed by
    :func:`tac.optimizer.meta_lagrangian.MetaLagrangianSearch.evaluate_all`
    (via ``tools/meta_lagrangian_search_cli.py --candidates-json``).

    The search engine expects each candidate dict to carry
    ``candidate_id``, ``archive_bytes``, ``rel_err_pct``, ``n_layers``,
    ``lane_class``, and optional ``archive_path`` keys. It forwards each
    dict directly into ``evaluate_candidate(**candidate)``; do not emit
    extra metadata here or the bridge will fail at runtime. The full sweep
    manifest already preserves op params, predicted bands, and provenance.

    Mapping rationale:
      - ``archive_bytes`` ← ``predicted_archive_bytes`` (the substituted size)
      - ``rel_err_pct`` ← 0.0 (CodecOp substitution doesn't change weight
        precision; the legacy field's semantics don't apply to stream-codec
        substitutions, so the deterministic-error case is 0)
      - ``n_layers`` ← 0 (stream codecs target poses/latents/masks, not
        decoder layers; the legacy field's semantics don't apply)
      - ``lane_class`` ← caller override, else derived from ``op_class``
        (lower-cased), so calibration anchors at
        ``.omx/calibration/anchors_<lane_class>.json`` are looked up
      - ``archive_path`` ← None (substitution surgery hasn't run yet;
        the cathedral discipline for this is documented in the manifest's
        ``next_blocking_step``)
    """
    out: list[dict[str, Any]] = []
    for c in candidates:
        derived_lane_class = lane_class or c.op_class.lower()
        out.append({
            "candidate_id": c.candidate_id,
            "archive_bytes": int(c.predicted_archive_bytes),
            "rel_err_pct": 0.0,
            "n_layers": 0,
            "lane_class": derived_lane_class,
            "archive_path": None,
        })
    return out


def emit_manifest(
    candidates: list[SweepCandidate],
    *,
    anchor_d_seg: float,
    anchor_d_pose: float,
    anchor_archive_bytes: int,
    baseline_substream_bytes: int,
) -> dict[str, Any]:
    """Emit the manifest dict in the shape parallel_dispatch_top_k consumes.

    Sorts candidates by ``predicted_score`` ascending (best first) so the
    consumer's ``--top-k`` flag picks the lowest-score candidates.
    """
    sorted_cands = sorted(candidates, key=lambda c: c.predicted_score)
    return {
        "schema_version": "codec_op_param_sweep_manifest.v1",
        "anchor": {
            "d_seg": anchor_d_seg,
            "d_pose": anchor_d_pose,
            "archive_bytes": anchor_archive_bytes,
            "baseline_substream_bytes": baseline_substream_bytes,
        },
        "n_candidates": len(sorted_cands),
        "tool": "tools/codec_op_param_sweep_manifest.py",
        "generated_at_unix_seconds": int(time.time()),
        "ready_for_exact_eval_dispatch_count": sum(
            1 for c in sorted_cands if c.ready_for_exact_eval_dispatch
        ),
        "evidence_semantics": "cpu_substrate_predicted_band",
        "next_blocking_step": (
            "archive_substitution_surgery: substrate-specific CodecOp blob "
            "→ archive zip member replacement, then operator override to "
            "flip ready_for_exact_eval_dispatch=True per-candidate"
        ),
        "candidates": [asdict(c) for c in sorted_cands],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--module", required=True,
                        help="Python module path of the CodecOp (e.g. tac.codec_pipeline_kl_pose)")
    parser.add_argument("--class", dest="class_name", required=True,
                        help="CodecOp class name within the module")
    parser.add_argument("--state-dict-path", type=Path, required=True,
                        help="Path to a .pt file containing the substrate state_dict or tensor")
    parser.add_argument("--state-dict-key", default=None,
                        help="If the .pt file is a single tensor, wrap it under this key")
    parser.add_argument("--param-grid", default="{}",
                        help="JSON dict mapping CodecOp init param name → list of values to sweep")
    parser.add_argument("--anchor-d-seg", type=float, required=True,
                        help="Substrate's authoritative d_seg (from contest-CUDA)")
    parser.add_argument("--anchor-d-pose", type=float, required=True,
                        help="Substrate's authoritative d_pose (from contest-CUDA)")
    parser.add_argument("--anchor-archive-bytes", type=int, required=True,
                        help="Substrate's full archive bytes pre-substitution")
    parser.add_argument("--baseline-substream-bytes", type=int, required=True,
                        help="Bytes the substrate's existing target stream contributes")
    parser.add_argument("--label-prefix", default="codec_op_sweep")
    parser.add_argument("--band-uncertainty-pct", type=float, default=0.005)
    parser.add_argument("--output", type=Path, required=True,
                        help="Output path for the full sweep manifest (JSON)")
    parser.add_argument(
        "--meta-lagrangian-output", type=Path, default=None,
        help="If set, ALSO emit a candidate-list JSON in the schema "
        "consumed by tools/meta_lagrangian_search_cli.py --candidates-json. "
        "Bridges the sweep adapter into the search engine; with this file "
        "in hand, run: meta_lagrangian_search_cli.py --candidates-json "
        "<path> --lane-class <op_class.lower()>",
    )
    parser.add_argument(
        "--meta-lagrangian-lane-class", default=None,
        help="Override the lane_class field in the meta-Lagrangian "
        "candidate dicts (anchors_<lane_class>.json must exist). "
        "Default: derived from op_class.lower().",
    )
    args = parser.parse_args(argv)

    op_cls = _import_codec_op(args.module, args.class_name)
    state_dict = _load_state_dict(args.state_dict_path, args.state_dict_key)
    try:
        param_grid = json.loads(args.param_grid)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"--param-grid is not valid JSON: {exc}") from None
    if not isinstance(param_grid, dict):
        raise SystemExit("--param-grid must be a JSON dict (got non-dict)")
    for k, v in param_grid.items():
        if not isinstance(v, list):
            raise SystemExit(f"--param-grid value for {k!r} must be a list")

    candidates = build_sweep_candidates(
        op_cls=op_cls,
        op_module=args.module,
        op_class_name=args.class_name,
        state_dict=state_dict,
        param_grid=param_grid,
        anchor_d_seg=args.anchor_d_seg,
        anchor_d_pose=args.anchor_d_pose,
        anchor_archive_bytes=args.anchor_archive_bytes,
        baseline_substream_bytes=args.baseline_substream_bytes,
        label_prefix=args.label_prefix,
        band_uncertainty_pct=args.band_uncertainty_pct,
    )
    manifest = emit_manifest(
        candidates,
        anchor_d_seg=args.anchor_d_seg,
        anchor_d_pose=args.anchor_d_pose,
        anchor_archive_bytes=args.anchor_archive_bytes,
        baseline_substream_bytes=args.baseline_substream_bytes,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    print(f"wrote {args.output} (n_candidates={manifest['n_candidates']})")
    print(
        f"top candidate: {manifest['candidates'][0]['candidate_id']} "
        f"predicted_score={manifest['candidates'][0]['predicted_score']:.5f}"
    )
    if args.meta_lagrangian_output:
        ml_candidates = to_meta_lagrangian_candidates(
            candidates, lane_class=args.meta_lagrangian_lane_class,
        )
        args.meta_lagrangian_output.parent.mkdir(parents=True, exist_ok=True)
        args.meta_lagrangian_output.write_text(
            json.dumps(ml_candidates, indent=2, sort_keys=True)
        )
        print(
            f"wrote {args.meta_lagrangian_output} ({len(ml_candidates)} candidates) — "
            f"feed into: meta_lagrangian_search_cli.py --candidates-json "
            f"{args.meta_lagrangian_output} --lane-class "
            f"{args.meta_lagrangian_lane_class or args.class_name.lower()}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
