#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""CodecOp parameter-sweep manifest builder.

The cathedral has CodecOps (e.g. ``Op_KLPoseStream``, ``Op_RAFTPoseStream``)
each with one or two free knobs (``n_components``, ``brotli_quality``). To
fire a parallel dispatch on a sweep of those knobs, the operator needs:

  1. Per-knob-setting bytes-out from the CodecOp on a real substrate
     (so predicted score bands are calibrated, not hand-waved)
  2. Per-knob-setting predicted contest-score band, anchored on a
     reference (d_seg, d_pose) and the cathedral's canonical formula
  3. A JSON manifest in the shape ``parallel_dispatch_top_k.py`` consumes

This tool emits that manifest. By default it is planning-only. With
``--substrate-archive-pr101`` it can also perform the reviewed PR101
decoder-blob substitution path for ``Op1_PR101SplitBrotli`` candidates and
write archive custody fields back into the manifest.

The manifest's candidates are flagged
``ready_for_exact_eval_dispatch=False`` and ``evidence_semantics=
"cpu_substrate_predicted_band"`` because predicted-band evidence is
NOT a contest-CUDA result. To actually dispatch, the operator must:

  (a) perform archive-substitution surgery, either with the PR101 actuator
      here or a substrate-specific actuator elsewhere
  (b) prove runtime parity + active lane claim, then flip
      ``ready_for_exact_eval_dispatch=True`` with an explicit
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
        --baseline-substream-bytes 162164 \\
        --baseline-substream-role decoder_blob \\
        --output reports/kl_pose_sweep_manifest.json

The ``--baseline-substream-bytes`` is the bytes the substrate currently
contributes to the archive. For PR101 archive substitution this must name a
parser-proven PR101 section such as ``decoder_blob``, ``latent_blob``, or
``sidecar_blob``; PR101 has no separate pose blob. Predicted Δ-rate per
candidate is
``(candidate_bytes - baseline_bytes) / RAW_VIDEO_BYTES * 25``.
"""
from __future__ import annotations

import argparse
import hashlib
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

PR101_SUBSTITUTABLE_BASELINE_ROLES = frozenset({
    "decoder_blob",
    "latent_blob",
    "sidecar_blob",
})


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
    materialized_payload_path: str | None = None
    materialized_payload_bytes: int | None = None
    materialized_payload_sha256: str | None = None
    materialized_payload_contract: str | None = None
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


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_state_dict_from_pr101_archive(
    archive_path: Path,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Decode PR101's fixed decoder slice directly from ``archive.zip``.

    This avoids creating a loose intermediate ``.pt`` file when the substrate
    is the public PR101 archive itself, and records the source member/slice
    custody in the sweep manifest.
    """
    if not archive_path.is_file():
        raise SystemExit(f"PR101 archive path does not exist: {archive_path}")
    sys.path.insert(0, str(REPO_ROOT / "tools"))
    from pr101_archive_substitution_surgery import (
        PR101_INNER_MEMBER_NAME,
        _read_inner_blob,
        _sha256,
        _split_pr101_inner_blob,
    )

    from tac.pr101_split_brotli_codec import decode_decoder_compact

    inner_blob = _read_inner_blob(archive_path)
    decoder_blob, latent_blob, sidecar_blob = _split_pr101_inner_blob(inner_blob)
    metadata = {
        "kind": "pr101_archive_decoder_blob",
        "archive_path": archive_path.as_posix(),
        "archive_size_bytes": archive_path.stat().st_size,
        "archive_sha256": _sha256_file(archive_path),
        "inner_member_name": PR101_INNER_MEMBER_NAME,
        "inner_member_bytes": len(inner_blob),
        "inner_member_sha256": _sha256(inner_blob),
        "decoder_blob_offset": 0,
        "decoder_blob_bytes": len(decoder_blob),
        "decoder_blob_sha256": _sha256(decoder_blob),
        "latent_blob_offset": len(decoder_blob),
        "latent_blob_bytes": len(latent_blob),
        "latent_blob_sha256": _sha256(latent_blob),
        "sidecar_blob_offset": len(decoder_blob) + len(latent_blob),
        "sidecar_blob_bytes": len(sidecar_blob),
        "sidecar_blob_sha256": _sha256(sidecar_blob),
    }
    return decode_decoder_compact(decoder_blob), metadata


def _expand_param_grid(grid: dict[str, list]) -> list[dict[str, Any]]:
    """Cartesian product of the parameter grid → list of param dicts."""
    if not grid:
        return [{}]
    keys = sorted(grid.keys())
    value_lists = [grid[k] for k in keys]
    return [dict(zip(keys, combo, strict=True)) for combo in itertools.product(*value_lists)]


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
    materialized_payload_output_dir: Path | None = None,
    materialized_payload_contract: str | None = None,
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
        except Exception as exc:
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
        materialized = _maybe_materialize_payload(
            result,
            candidate_id=candidate_id,
            output_dir=materialized_payload_output_dir,
            expected_bytes=candidate_substream_bytes,
            payload_contract=materialized_payload_contract,
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
            materialized_payload_path=materialized.get("path"),
            materialized_payload_bytes=materialized.get("bytes"),
            materialized_payload_sha256=materialized.get("sha256"),
            materialized_payload_contract=materialized.get("contract"),
        ))
    return candidates


def _maybe_materialize_payload(
    result: Any,
    *,
    candidate_id: str,
    output_dir: Path | None,
    expected_bytes: int,
    payload_contract: str | None,
) -> dict[str, Any]:
    if output_dir is None:
        return {}
    if not hasattr(result, "blob"):
        raise SystemExit(
            "--materialized-payload-output-dir requires CodecOp encode result.blob"
        )
    blob = result.blob
    if not isinstance(blob, bytes | bytearray):
        raise SystemExit(
            "CodecOp encode result.blob must be bytes when materializing payloads"
        )
    payload = bytes(blob)
    if len(payload) != int(expected_bytes):
        raise SystemExit(
            f"CodecOp result bytes_out mismatch: bytes_out={expected_bytes} "
            f"len(blob)={len(payload)}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    payload_path = output_dir / f"{_safe_candidate_stem(candidate_id)}.section"
    payload_path.write_bytes(payload)
    return {
        "path": payload_path.as_posix(),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "contract": payload_contract or "raw_codecop_encode_blob",
    }


def _safe_candidate_stem(candidate_id: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in candidate_id)
    safe = safe.strip("._-")
    return safe or "codec_op_candidate"


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
    baseline_substream_role: str | None = None,
    state_dict_source: dict[str, Any] | None = None,
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
            "baseline_substream_role": baseline_substream_role,
        },
        "state_dict_source": state_dict_source,
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
    parser.add_argument("--state-dict-path", type=Path,
                        help="Path to a .pt file containing the substrate state_dict or tensor")
    parser.add_argument("--state-dict-key", default=None,
                        help="If the .pt file is a single tensor, wrap it under this key")
    parser.add_argument(
        "--state-dict-from-pr101-archive", type=Path, default=None,
        help="Decode the state_dict from the decoder_blob inside a PR101-shaped "
        "archive.zip and record source/member custody in the manifest. Mutually "
        "exclusive with --state-dict-path.",
    )
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
    parser.add_argument(
        "--baseline-substream-role",
        default=None,
        help="Optional parser-proven source section role. Required for PR101 "
        "archive-backed runs and must be one of decoder_blob, latent_blob, "
        "or sidecar_blob. PR101 does not ship a separate pose stream.",
    )
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
    parser.add_argument(
        "--substrate-archive-pr101", type=Path, default=None,
        help="Path to a PR101-shaped archive.zip. When set AND the CodecOp "
        "produces a wire-format-compatible decoder_blob (Op1_PR101SplitBrotli), "
        "the tool calls tools.pr101_archive_substitution_surgery to produce a "
        "real substituted archive per candidate at "
        "experiments/results/<label_prefix>_<candidate_id>/archive.zip and "
        "writes the path into archive_path. For non-compatible ops "
        "(Op_KLPoseStream, etc.) this flag is rejected with a clear error.",
    )
    parser.add_argument(
        "--substituted-archive-output-dir", type=Path, default=None,
        help="Directory under which per-candidate substituted archives are "
        "written (default: experiments/results/<label_prefix>_<candidate_id>/).",
    )
    parser.add_argument(
        "--atom-ledger-output", type=Path, default=None,
        help="Optional path to a JSONL atom ledger; one row appended per "
        "candidate so each sweep entry becomes a meta-Lagrangian atom. "
        "Conventionally this is "
        "experiments/results/bilevel_atom_ledger.jsonl (the cathedral's "
        "shared blackboard). Each row carries the candidate's predicted "
        "band, op identity, params, and evidence_grade=[CPU-prep]. Closes "
        "the manifest -> ledger -> meta-Lagrangian loop.",
    )
    parser.add_argument(
        "--materialized-payload-output-dir", type=Path, default=None,
        help=(
            "Optional directory where each CodecOp encode result.blob is written "
            "as a raw section payload. This is byte custody for downstream "
            "tools/build_monolithic_codec_op_replacement_manifest.py; it does "
            "not make candidates dispatchable."
        ),
    )
    parser.add_argument(
        "--materialized-payload-contract",
        default=None,
        help=(
            "Optional contract label attached to materialized payload files, "
            "for example raw_codecop_encode_blob, pr106_decoder_packed_brotli, "
            "or pr101_decoder_blob. The downstream monolithic bridge still "
            "validates the real section contract."
        ),
    )
    args = parser.parse_args(argv)

    op_cls = _import_codec_op(args.module, args.class_name)
    if (args.state_dict_path is None) == (args.state_dict_from_pr101_archive is None):
        raise SystemExit(
            "pass exactly one of --state-dict-path or --state-dict-from-pr101-archive"
        )
    pr101_archive_backed = (
        args.state_dict_from_pr101_archive is not None
        or args.substrate_archive_pr101 is not None
    )
    if (
        pr101_archive_backed
        and args.baseline_substream_role not in PR101_SUBSTITUTABLE_BASELINE_ROLES
    ):
        allowed = ", ".join(sorted(PR101_SUBSTITUTABLE_BASELINE_ROLES))
        raise SystemExit(
            "PR101 archive-backed CodecOp sweeps require "
            f"--baseline-substream-role in {{{allowed}}}; got "
            f"{args.baseline_substream_role!r}. PR101 has no separate pose "
            "or mask ZIP member, so pose/blob budget shortcuts are invalid."
        )
    if args.state_dict_from_pr101_archive is not None:
        if args.state_dict_key is not None:
            raise SystemExit(
                "--state-dict-key is only valid with --state-dict-path"
            )
        state_dict, state_dict_source = _load_state_dict_from_pr101_archive(
            args.state_dict_from_pr101_archive
        )
    else:
        state_dict = _load_state_dict(args.state_dict_path, args.state_dict_key)
        state_dict_source = {
            "kind": "torch_state_dict_file",
            "path": args.state_dict_path.as_posix(),
            "sha256": _sha256_file(args.state_dict_path),
            "state_dict_key": args.state_dict_key,
        }
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
        materialized_payload_output_dir=args.materialized_payload_output_dir,
        materialized_payload_contract=args.materialized_payload_contract,
    )
    manifest = emit_manifest(
        candidates,
        anchor_d_seg=args.anchor_d_seg,
        anchor_d_pose=args.anchor_d_pose,
        anchor_archive_bytes=args.anchor_archive_bytes,
        baseline_substream_bytes=args.baseline_substream_bytes,
        state_dict_source=state_dict_source,
        baseline_substream_role=args.baseline_substream_role,
    )
    manifest_rows_by_id = {row["candidate_id"]: row for row in manifest["candidates"]}
    # PR101 archive-substitution surgery wire-up (huge tranche #4):
    # for wire-format-compatible CodecOps + a PR101 substrate archive,
    # produce a real substituted archive per candidate so dispatch is
    # ready-to-fire (modulo operator GPU authorization).
    if args.substrate_archive_pr101 is not None:
        if args.class_name not in {"Op1_PR101SplitBrotli"}:
            raise SystemExit(
                f"--substrate-archive-pr101 requires --class Op1_PR101SplitBrotli "
                f"(wire-format-compatible with PR101). Got --class={args.class_name}; "
                f"this op produces a stream PR101 doesn't ship or has incompatible "
                f"wire format. Substitution would corrupt the archive."
            )
        # Lazy-import the surgery tool to keep top-level imports light
        sys.path.insert(0, str(REPO_ROOT / "tools"))
        from pr101_archive_substitution_surgery import substitute_decoder_blob

        out_root = (
            args.substituted_archive_output_dir
            or REPO_ROOT / "experiments" / "results" / "codec_op_substituted"
        )
        out_root.mkdir(parents=True, exist_ok=True)
        for candidate in candidates:
            cand_dir = out_root / f"{args.label_prefix}_{candidate.candidate_id}"
            cand_dir.mkdir(parents=True, exist_ok=True)
            archive_out = cand_dir / "archive.zip"
            # Re-run the encode to get the actual blob bytes (not just bytes_out).
            op_instance = op_cls(**candidate.op_params)
            blob_result = op_instance.encode(state_dict, context={})
            # PR101 surgery requires exact 162,164-byte decoder_blob length.
            # If the cathedral re-encoded with auto_select_byte_maps the bytes
            # MAY be smaller — in which case we cannot substitute into a stock
            # PR101 archive (offsets shift) and must skip + log.
            if len(blob_result.blob) != 162_164:
                print(
                    f"  candidate {candidate.candidate_id}: bytes_out="
                    f"{len(blob_result.blob)} != 162164 (PR101 fixed offset); "
                    "substitution skipped (would corrupt latent_blob extraction). "
                    "This candidate produces savings but requires a forked inflate."
                )
                continue
            report = substitute_decoder_blob(
                input_archive=args.substrate_archive_pr101,
                replacement_decoder_blob=blob_result.blob,
                output_archive=archive_out,
            )
            report_path = cand_dir / "substitution_report.json"
            report_path.write_text(
                json.dumps(asdict(report), indent=2, sort_keys=True)
            )
            manifest_row = manifest_rows_by_id[candidate.candidate_id]
            manifest_row.update({
                "archive_path": archive_out.as_posix(),
                "archive_size_bytes": archive_out.stat().st_size,
                "expected_archive_size_bytes": archive_out.stat().st_size,
                "archive_sha256": report.sha256_output_archive,
                "expected_archive_sha256": report.sha256_output_archive,
                "archive_substitution_report_path": report_path.as_posix(),
                "archive_substitution_schema": "pr101_decoder_blob_surgery_v1",
                "archive_member_name": report.inner_member_name,
                "source_archive_path": report.input_archive,
                "source_archive_size_bytes": report.input_size_bytes,
                "source_archive_sha256": report.sha256_input_archive,
                "source_inner_member_sha256": report.sha256_input_inner_member,
                "output_inner_member_sha256": report.sha256_output_inner_member,
                "input_decoder_blob_sha256": report.sha256_input_decoder_blob,
                "replacement_decoder_blob_sha256": report.sha256_replacement_decoder_blob,
                "materialized_payload_path": manifest_row.get("materialized_payload_path"),
                "materialized_payload_bytes": manifest_row.get("materialized_payload_bytes"),
                "materialized_payload_sha256": manifest_row.get("materialized_payload_sha256"),
                "materialized_payload_contract": manifest_row.get("materialized_payload_contract"),
                "input_latent_blob_sha256": report.sha256_input_latent_blob,
                "output_latent_blob_sha256": report.sha256_output_latent_blob,
                "input_sidecar_blob_sha256": report.sha256_input_sidecar_blob,
                "output_sidecar_blob_sha256": report.sha256_output_sidecar_blob,
            })
            manifest_row["blockers"] = [
                blocker for blocker in manifest_row["blockers"]
                if blocker != "archive_substitution_surgery_pending"
            ]
            manifest_row["blockers"].extend([
                "exact_runtime_parity_not_supplied",
                "matching_lane_dispatch_claim_not_supplied",
            ])
            print(
                f"  candidate {candidate.candidate_id}: substituted "
                f"archive_bytes={archive_out.stat().st_size} "
                f"(input={report.input_size_bytes}, delta={report.bytes_delta:+}) "
                f"-> {archive_out}"
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
        for row in ml_candidates:
            manifest_row = manifest_rows_by_id.get(row["candidate_id"])
            if manifest_row is None or not manifest_row.get("archive_path"):
                continue
            row["archive_path"] = manifest_row["archive_path"]
            row["archive_bytes"] = int(
                manifest_row.get("archive_size_bytes")
                or manifest_row["predicted_archive_bytes"]
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

    # Atom-ledger wire-up (tranche item #6): each sweep candidate becomes
    # a row in the meta-Lagrangian atom ledger, closing the
    # manifest -> ledger -> meta-Lagrangian -> Pareto loop.
    if args.atom_ledger_output:
        timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        args.atom_ledger_output.parent.mkdir(parents=True, exist_ok=True)
        with args.atom_ledger_output.open("a") as f:
            for candidate in candidates:
                # Look up archive metadata from the manifest if surgery ran
                manifest_row = next(
                    (r for r in manifest["candidates"]
                     if r["candidate_id"] == candidate.candidate_id),
                    None,
                )
                archive_path = (
                    manifest_row.get("archive_path")
                    if manifest_row else None
                )
                archive_sha = (
                    manifest_row.get("archive_sha256")
                    if manifest_row else None
                )
                evidence_grade = (
                    "[CPU-prep]" if archive_path is None
                    else "[CPU-prep+substituted-archive]"
                )
                substrate_path = (
                    state_dict_source.get("archive_path")
                    or state_dict_source.get("path")
                )
                row = {
                    "timestamp_utc": timestamp,
                    "phase": None,
                    "substrate_label": f"{args.label_prefix}/{candidate.candidate_id}",
                    "substrate_path": substrate_path,
                    "substrate_score_anchor": None,
                    "contest_cuda_score": None,
                    "evidence_grade": evidence_grade,
                    "predicted_score": candidate.predicted_score,
                    "predicted_band": list(candidate.predicted_band),
                    "score_delta_vs_anchor": candidate.score_delta_vs_anchor,
                    "rate_delta_vs_anchor": candidate.rate_delta_vs_anchor,
                    "archive_bytes": (
                        int(manifest_row["archive_size_bytes"])
                        if manifest_row and manifest_row.get("archive_size_bytes")
                        else int(candidate.predicted_archive_bytes)
                    ),
                    "archive_sha256": archive_sha,
                    "archive_path": archive_path,
                    "materialized_payload_path": (
                        manifest_row.get("materialized_payload_path")
                        if manifest_row else None
                    ),
                    "materialized_payload_sha256": (
                        manifest_row.get("materialized_payload_sha256")
                        if manifest_row else None
                    ),
                    "materialized_payload_contract": (
                        manifest_row.get("materialized_payload_contract")
                        if manifest_row else None
                    ),
                    "cathedral_op": (
                        f"{args.module}.{args.class_name}"
                    ),
                    "op_params": dict(candidate.op_params),
                    "notes": (
                        f"sweep candidate from "
                        f"tools/codec_op_param_sweep_manifest.py; "
                        f"{candidate.evidence_semantics}"
                    ),
                }
                f.write(json.dumps(row, sort_keys=True) + "\n")
        print(
            f"appended {len(candidates)} atom rows to {args.atom_ledger_output} "
            f"(evidence_grade={evidence_grade})"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
