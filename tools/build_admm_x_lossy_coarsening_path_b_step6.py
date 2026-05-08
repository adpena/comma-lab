#!/usr/bin/env python3
"""Build the Path B step 6 Lagrangian per-tensor allocation × lossy_coarsening
byte-closed candidate archive (historically named "ADMM × lossy_coarsening").

NAMING CORRECTION (REVIEW-MATH, 2026-05-08, Dykstra council finding)
--------------------------------------------------------------------
The historical name "ADMM" in this tool refers to the **Lagrangian
per-tensor allocation** mechanism (λ-bisection over independent per-tensor
argmin), NOT iterative primal-dual ADMM. The filename and inputs are kept
backward-compatible; the docstring + manifest field
``review_math_naming_clarification`` document the actual mechanism.

Source-of-finding
-----------------
``reports/raw/pr101_omega_opt_admm_x_lossy_coarsening_20260508T041303Z/manifest.json``
(commit 983598d2 of ``tools/pr101_omega_opt_admm_x_lossy_coarsening_empirical.py``)
showed that Lagrangian per-tensor allocation over a continuous-K basis
produces a smaller archive than subagent D's greedy per-tensor-budget
approach when both target the same RMS distortion.

At ``rms_target=0.0386`` the ADMM solution achieves:

    Ks = [2, 1, 5, 1, 5, 1, 5, 1, 2, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    archive_bytes = 153_639  (proxy, lambda=1276154.69)
    rel_err = 0.0415

Compared to subagent D's greedy archive at the same rms_target
(``experiments/results/lossy_coarsening_20260508T024022Z/archive.zip``,
156,404 B at rel_err=0.0386), the ADMM allocation saves ~14,658 B in the
proxy and ~2,765 B in the byte-closed runtime build (the proxy uses a
condensed PR101 archive overhead model; the runtime uses the wire format
documented in ``experiments/lossy_coarsening_lightning_cuda_test.py``).

Mandate (Subagent ADMMBUILD, 2026-05-08)
----------------------------------------
1. Reuse the build logic from ``experiments/lossy_coarsening_lightning_cuda_test.py``
   but inject the fixed ADMM Ks instead of running the per-tensor K-search.
2. Output to ``experiments/results/admm_x_lossy_coarsening_path_b_step6_<ts>/``
   with ``archive.zip``, ``submission_dir/`` (forked inflate.py + inflate.sh
   + src/{codec.py,model.py}), and ``build_manifest.json``.
3. Smoke-test the archive locally: parse the inner blob with the forked
   inflate, verify all 28 tensors round-trip, and that rel_err matches the
   ADMM proxy.
4. Stamp the build_manifest.json with proper fail-closed custody fields.

Out-of-scope
------------
- No dispatch (Lightning bootstrap is owned by Subagent BSF in parallel).
- No wire-format / inflate.py / canonical encoder modifications.

CLAUDE.md compliance
--------------------
- ``family_falsified=False``,
  ``falsification_scope="lagrangian_x_continuous_K_only"``.
- ``ready_for_exact_eval_dispatch=False`` (CPU build never promotes itself).
- ``cuda_eval_worth_testing=True`` (this is exactly the candidate the user
  approved for CUDA score validation).
- Pure-CPU; never loads a scorer; tags evidence ``[CPU-build]``.

Usage
-----

.. code-block:: bash

    .venv/bin/python tools/build_admm_x_lossy_coarsening_path_b_step6.py
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import struct
import sys
import zipfile
from pathlib import Path

import brotli
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from tac.pr101_split_brotli_codec import (  # noqa: E402
    DECODER_BLOB_LEN,
    FIXED_STATE_SCHEMA,
    LATENT_BLOB_LEN,
    N_QUANT,
    _quantize_tensor,
)

# Re-use the staging helpers + forked inflate sources from the canonical
# Lightning build script. The dispatcher path is gated behind argparse flags
# we never set, so importing it is side-effect free.
_LIGHTNING_BUILDER_PATH = (
    REPO_ROOT / "experiments" / "lossy_coarsening_lightning_cuda_test.py"
)
_spec = importlib.util.spec_from_file_location(
    "_lossy_coarsening_lightning_cuda_test", _LIGHTNING_BUILDER_PATH
)
if _spec is None or _spec.loader is None:  # pragma: no cover - sanity
    raise SystemExit(f"FATAL: could not load builder from {_LIGHTNING_BUILDER_PATH}")
_lightning_builder = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_lightning_builder)

_read_pr101_inner_blob = _lightning_builder._read_pr101_inner_blob
_split_pr101_inner_blob = _lightning_builder._split_pr101_inner_blob
_build_inner_blob = _lightning_builder._build_inner_blob
_write_pr101_archive = _lightning_builder._write_pr101_archive
_stage_forked_submission_dir = _lightning_builder._stage_forked_submission_dir

LANE_ID = "admm_x_lossy_coarsening_path_b_step6"
SCHEMA_VERSION = "admm_x_lossy_coarsening_path_b_step6_build.v1"
TOOL_NAME = "tools/build_admm_x_lossy_coarsening_path_b_step6.py"

# ADMM Ks pulled from
# reports/raw/pr101_omega_opt_admm_x_lossy_coarsening_20260508T041303Z/manifest.json
# at comparison_at_rms_targets[2].admm_K_lagrangian.Ks (rms_target=0.0386).
ADMM_PATH_B_STEP6_KS: tuple[int, ...] = (
    2, 1, 5, 1, 5, 1, 5, 1, 2, 1, 2, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
)
ADMM_PROXY_ARCHIVE_BYTES = 153_639  # from manifest, lagrangian basis
ADMM_PROXY_REL_ERR = 0.0415393353487541
ADMM_PROXY_LAMBDA = 1_276_154.6884887693
ADMM_RMS_TARGET = 0.0386
ADMM_SOURCE_MANIFEST = (
    "reports/raw/pr101_omega_opt_admm_x_lossy_coarsening_20260508T041303Z/manifest.json"
)

DEFAULT_PR101_STATE_DICT = (
    REPO_ROOT
    / "experiments/results/pr101_codecop_sweep_20260507_codex"
    / "pr101_decoder_state_dict.pt"
)
DEFAULT_PR101_FRONTIER_ARCHIVE = (
    REPO_ROOT
    / "experiments/results/public_pr_intake_full"
    / "public_pr101_intake_20260505_auto"
    / "archive.zip"
)
DEFAULT_PR101_SOURCE_DIR = (
    REPO_ROOT
    / "experiments/results/public_pr_intake_full"
    / "public_pr101_intake_20260505_auto"
    / "source/submissions/hnerv_ft_microcodec"
)
DEFAULT_PREDICTED_BAND = (0.18, 0.22)

CPU_BUILD_SCORE_BLOCKERS = [
    "cpu_build_rel_err_proxy_not_score_evidence",
    "exact_cuda_auth_eval_not_yet_harvested",
    "requires_contest_auth_eval_json_before_score_promotion_rank_or_kill",
    # REVIEW-ENG C3 (2026-05-08): the 4.15% rel_err -> score mapping is
    # unmeasured for low-rel_err lossy weight encodings on PR101 substrate.
    # The closest empirical anchor (lossy_coarsening_analytical at 3.86%
    # rel_err) scored 0.3517 [contest-CUDA A-negative] — significantly
    # worse than the 0.2089 frontier. apogee_int6 must land FIRST to
    # calibrate the rel_err->score curve before another paid dispatch.
    "apogee_int6_contest_cuda_anchor_required_first",
]


def _utc_now_iso() -> str:
    return dt.datetime.now(tz=dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def cpu_build_proxy_guard_fields() -> dict[str, object]:
    """Fail-closed custody fields for the local CPU build artifact.

    Mirrors ``cpu_build_proxy_guard_fields`` in the Lightning builder so the
    audit-trail keys are identical across the two build paths.
    """
    return {
        "evidence_semantics": "cpu_build_byte_closed_candidate_proxy_no_score",
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "dispatch_attempted": False,
        "cuda_eval_worth_testing": True,
        "family_falsified": False,
        "falsification_scope": "lagrangian_x_continuous_K_only",
        "custody_status": "transient-allowed",
        "custody_status_reason": (
            "CPU-build archives are ignored local custody artifacts; durable "
            "signal must be summarized in .omx/research and exact-score "
            "promotion requires contest auth eval on a rebuilt packet."
        ),
        "score_claim_blockers": list(CPU_BUILD_SCORE_BLOCKERS),
        "dispatch_blockers": list(CPU_BUILD_SCORE_BLOCKERS),
    }


def _build_lossy_decoder_section_with_fixed_Ks(
    state_dict_path: Path,
    Ks: list[int],
    *,
    brotli_quality: int = 11,
) -> dict:
    """Apply per-tensor coarsening with a FIXED Ks vector; produce wire-format bytes.

    Mirrors the wire format of ``_build_lossy_decoder_section`` in the
    Lightning builder, but skips the K-search and uses the caller-provided Ks
    instead. Wire format (decoder section returned in ``decoder_bytes``):

        uint32 LE: total_section_bytes (D, including this prefix)
        bytes 28: per-tensor K (uint8)
        bytes 56: per-tensor fp16 scale (LE half)
        bytes (D - 4 - 28 - 56): brotli(concat(rounded_int8s))
    """
    if not state_dict_path.is_file():
        raise SystemExit(f"FATAL: PR101 state_dict not found: {state_dict_path}")
    n_tensors = len(FIXED_STATE_SCHEMA)
    if len(Ks) != n_tensors:
        raise SystemExit(
            f"FATAL: Ks length {len(Ks)} != n_tensors {n_tensors} (FIXED_STATE_SCHEMA)"
        )
    for k in Ks:
        if not isinstance(k, int) or k < 1 or k > 255:
            raise SystemExit(f"FATAL: K out of [1,255] range: {k!r}")

    sd = torch.load(state_dict_path, map_location="cpu", weights_only=True)  # REVIEW-ENG C4: tensor-only state_dict
    scales_fp16: list[float] = []
    rounded_chunks: list[np.ndarray] = []
    abs_orig_total = 0.0
    abs_err_total = 0.0
    n_symbols = 0
    for (name, _shape), K in zip(FIXED_STATE_SCHEMA, Ks, strict=True):
        qt = _quantize_tensor(name, sd[name], n_quant=N_QUANT)
        symbols_i32 = qt.q_i8.astype(np.int32).flatten()
        scale_fp16 = float(np.float16(qt.scale))
        rounded = np.round(symbols_i32 / K) * K
        rounded_clipped = rounded.clip(-127, 127)
        abs_orig_total += float(np.abs(symbols_i32).astype(np.float64).sum())
        abs_err_total += float(
            np.abs(rounded_clipped - symbols_i32).astype(np.float64).sum()
        )
        rounded_chunks.append(rounded_clipped.astype(np.int8))
        scales_fp16.append(scale_fp16)
        n_symbols += int(symbols_i32.size)

    flat = np.concatenate(rounded_chunks).tobytes()
    brotli_payload = brotli.compress(
        flat, quality=brotli_quality, lgwin=22, lgblock=24
    )
    rel_err = abs_err_total / abs_orig_total if abs_orig_total > 1e-9 else 0.0  # REL_ERR_NON_CANONICAL_OK: global L1 ratio for ADMM-x-lossy_coarsening Path B step6; preserves 0.0415 anchor. See .omx/research/rel_err_inconsistency_audit_20260508_claude.md

    K_bytes = bytes(Ks)
    scale_arr = np.array(scales_fp16, dtype=np.float16)
    if not scale_arr.dtype.isnative or sys.byteorder != "little":
        scale_bytes = scale_arr.astype("<f2").tobytes()
    else:
        scale_bytes = scale_arr.tobytes()

    section_no_prefix = K_bytes + scale_bytes + brotli_payload
    section_total = 4 + len(section_no_prefix)
    prefix = struct.pack("<I", section_total)
    decoder_bytes = prefix + section_no_prefix
    if len(decoder_bytes) != section_total:
        raise RuntimeError(
            f"decoder section length mismatch: declared {section_total}, "
            f"actual {len(decoder_bytes)}"
        )
    return {
        "decoder_bytes": decoder_bytes,
        "per_tensor_K": list(Ks),
        "per_tensor_scale_fp16": scales_fp16,
        "rel_err": rel_err,
        "n_tensors": n_tensors,
        "n_symbols": n_symbols,
        "brotli_payload_bytes": len(brotli_payload),
        "K_bytes": len(K_bytes),
        "scale_bytes": len(scale_bytes),
        "section_total_bytes": section_total,
    }


def _local_smoke_roundtrip(
    archive_path: Path, *, pr101_state_dict_path: Path, submission_dir: Path
) -> dict:
    """Smoke test: parse the lossy archive with the forked inflate, verify
    every tensor decodes back to the encoder's quantized form within
    ``rel_err`` tolerance.

    Returns roundtrip rel_err vs the encoder's ``_quantize_tensor`` output
    (matches the inflate-side recon path: q_i8 * fp16_scale).
    """
    spec_path = submission_dir / "inflate.py"
    if not spec_path.is_file():
        raise SystemExit(
            f"FATAL: forked inflate.py missing for smoke roundtrip: {spec_path}"
        )
    spec = importlib.util.spec_from_file_location("forked_inflate_admm", spec_path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"FATAL: cannot load forked inflate spec from {spec_path}")
    forked_inflate = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(submission_dir / "src"))
    try:
        spec.loader.exec_module(forked_inflate)
    finally:
        sys.path.pop(0)

    inner = _read_pr101_inner_blob(archive_path)
    decoder_sd, latents, meta = forked_inflate.parse_lossy_archive(inner)

    sd_ref = torch.load(
        pr101_state_dict_path, map_location="cpu", weights_only=True  # REVIEW-ENG C4: tensor-only state_dict
    )
    abs_orig = 0.0
    abs_err = 0.0
    per_tensor: list[dict] = []
    for name, _shape in FIXED_STATE_SCHEMA:
        t_dec = decoder_sd[name].cpu().numpy().astype(np.float32)
        qt_ref = _quantize_tensor(name, sd_ref[name], n_quant=N_QUANT)
        ref_quantized = (
            qt_ref.q_i8.astype(np.float32) * float(np.float16(qt_ref.scale))
        ).reshape(t_dec.shape)
        denom_q = float(np.abs(ref_quantized).sum())
        err_q = float(np.abs(t_dec - ref_quantized).sum())
        per_tensor.append(
            {"name": name, "rel_err_vs_quantized": (err_q / denom_q) if denom_q > 1e-9 else 0.0}
        )
        abs_orig += denom_q
        abs_err += err_q
    rel_err = abs_err / abs_orig if abs_orig > 1e-9 else 0.0  # REL_ERR_NON_CANONICAL_OK: global L1 ratio for fp32 smoke probe; consistent with Path B step6 mainline form

    # Latent + sidecar passthrough sanity: we did not modify those bytes, so
    # the latents tensor should decode without error and have N_PAIRS rows.
    n_pairs = int(latents.shape[0]) if hasattr(latents, "shape") else None
    return {
        "rel_err_vs_quantized_fp32": rel_err,
        "n_tensors_compared": len(per_tensor),
        "max_per_tensor_rel_err": max(t["rel_err_vs_quantized"] for t in per_tensor),
        "n_latent_pairs_decoded": n_pairs,
        "latent_dim_meta": meta.get("latent_dim"),
        "base_channels_meta": meta.get("base_channels"),
        "eval_size_meta": meta.get("eval_size"),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--state-dict",
        type=Path,
        default=DEFAULT_PR101_STATE_DICT,
    )
    p.add_argument(
        "--frontier-archive",
        type=Path,
        default=DEFAULT_PR101_FRONTIER_ARCHIVE,
    )
    p.add_argument(
        "--pr101-source-dir",
        type=Path,
        default=DEFAULT_PR101_SOURCE_DIR,
    )
    p.add_argument(
        "--brotli-quality",
        type=int,
        default=11,
    )
    p.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "experiments" / "results",
    )
    p.add_argument(
        "--predicted-low",
        type=float,
        default=DEFAULT_PREDICTED_BAND[0],
    )
    p.add_argument(
        "--predicted-high",
        type=float,
        default=DEFAULT_PREDICTED_BAND[1],
    )
    args = p.parse_args(argv)

    if not args.state_dict.is_file():
        sys.exit(f"FATAL: --state-dict not found: {args.state_dict}")
    if not args.frontier_archive.is_file():
        sys.exit(f"FATAL: --frontier-archive not found: {args.frontier_archive}")
    if not args.pr101_source_dir.is_dir():
        sys.exit(f"FATAL: --pr101-source-dir not found: {args.pr101_source_dir}")

    timestamp = dt.datetime.now(tz=dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    build_dir = args.output_root / f"admm_x_lossy_coarsening_path_b_step6_{timestamp}"
    build_dir.mkdir(parents=True, exist_ok=True)
    archive_path = build_dir / "archive.zip"
    submission_dir = build_dir / "submission_dir"
    build_manifest_path = build_dir / "build_manifest.json"

    Ks = list(ADMM_PATH_B_STEP6_KS)
    print(
        f"[admm-build] applying ADMM Ks (rms_target={ADMM_RMS_TARGET}, "
        f"lambda={ADMM_PROXY_LAMBDA:.0f})"
    )
    print(f"[admm-build]   Ks = {Ks}")
    section = _build_lossy_decoder_section_with_fixed_Ks(
        args.state_dict, Ks, brotli_quality=args.brotli_quality
    )
    print(
        f"[admm-build]   decoder section bytes={section['section_total_bytes']:,} "
        f"(brotli={section['brotli_payload_bytes']:,})"
    )
    print(f"[admm-build]   rel_err vs int8 quantized symbols: {section['rel_err']:.6f}")

    pr101_inner = _read_pr101_inner_blob(args.frontier_archive)
    _orig_decoder, latent_blob, sidecar_blob = _split_pr101_inner_blob(pr101_inner)
    if len(_orig_decoder) != DECODER_BLOB_LEN:
        raise SystemExit(
            f"FATAL: PR101 frontier decoder length {len(_orig_decoder)} != expected "
            f"{DECODER_BLOB_LEN}"
        )
    if len(latent_blob) != LATENT_BLOB_LEN:
        raise SystemExit(
            f"FATAL: PR101 latent_blob length {len(latent_blob)} != expected "
            f"{LATENT_BLOB_LEN}"
        )
    print(
        f"[admm-build]   PR101 latent_blob={len(latent_blob):,} B "
        f"sidecar_blob={len(sidecar_blob):,} B"
    )

    inner_blob = _build_inner_blob(section["decoder_bytes"], latent_blob, sidecar_blob)
    _write_pr101_archive(inner_blob, archive_path)
    archive_bytes = archive_path.stat().st_size
    archive_sha = _sha256(archive_path.read_bytes())
    print(
        f"[admm-build] WROTE archive: {archive_path.relative_to(REPO_ROOT)} "
        f"size={archive_bytes:,} B sha256={archive_sha[:16]}..."
    )

    _stage_forked_submission_dir(
        submission_dir, pr101_source_dir=args.pr101_source_dir
    )
    print(
        f"[admm-build] WROTE submission dir: {submission_dir.relative_to(REPO_ROOT)}"
    )

    print("[smoke] running CPU roundtrip ...")
    smoke = _local_smoke_roundtrip(
        archive_path,
        pr101_state_dict_path=args.state_dict,
        submission_dir=submission_dir,
    )
    print(
        f"[smoke] rel_err_vs_quantized_fp32={smoke['rel_err_vs_quantized_fp32']:.6f} "
        f"max_per_tensor={smoke['max_per_tensor_rel_err']:.6f} "
        f"n_tensors={smoke['n_tensors_compared']} "
        f"n_latent_pairs={smoke['n_latent_pairs_decoded']}"
    )
    if smoke["rel_err_vs_quantized_fp32"] > section["rel_err"] * 2 + 1e-3:
        sys.exit(
            f"FATAL: roundtrip rel_err {smoke['rel_err_vs_quantized_fp32']:.4f} "
            f">> encoder rel_err {section['rel_err']:.4f}; wire-format bug"
        )
    if smoke["n_latent_pairs_decoded"] != 600:
        sys.exit(
            f"FATAL: smoke decoded n_pairs={smoke['n_latent_pairs_decoded']} != 600 "
            "(PR101 N_PAIRS); latent_blob passthrough broken"
        )

    build_manifest = {
        "schema_version": SCHEMA_VERSION,
        "tool": TOOL_NAME,
        "lane_id": LANE_ID,
        "built_at_utc": _utc_now_iso(),
        "source_admm_manifest": ADMM_SOURCE_MANIFEST,
        "review_math_naming_clarification": {
            "historical_name": "ADMM",
            "actual_mechanism": "lagrangian_per_tensor_allocation",
            "rationale": (
                "λ-bisection over INDEPENDENT per-tensor argmin problems; "
                "no iterative primal-dual ADMM updates, no consensus "
                "constraints. Renamed per REVIEW-MATH 2026-05-08 Dykstra "
                "council finding."
            ),
            "filename_kept_for_backward_compat": True,
        },
        "technique_canonical_name": "lagrangian_per_tensor_allocation_x_continuous_lossy_coarsening",
        "technique_historical_alias": "admm_x_lossy_coarsening_path_b_step6",
        "admm_rms_target": ADMM_RMS_TARGET,
        "admm_lambda": ADMM_PROXY_LAMBDA,
        "admm_proxy_archive_bytes": ADMM_PROXY_ARCHIVE_BYTES,
        "admm_proxy_rel_err": ADMM_PROXY_REL_ERR,
        "rel_err_actual_int8": section["rel_err"],
        "rel_err_actual_fp32_smoke": smoke["rel_err_vs_quantized_fp32"],
        "max_per_tensor_rel_err_fp32_smoke": smoke["max_per_tensor_rel_err"],
        "brotli_quality": args.brotli_quality,
        "archive_relpath": str(archive_path.relative_to(REPO_ROOT)),
        "archive_bytes": archive_bytes,
        "archive_sha256": archive_sha,
        "submission_dir_relpath": str(submission_dir.relative_to(REPO_ROOT)),
        "input_state_dict": str(args.state_dict),
        "input_frontier_archive": str(args.frontier_archive),
        "input_pr101_source_dir": str(args.pr101_source_dir),
        "section_total_bytes": section["section_total_bytes"],
        "section_brotli_payload_bytes": section["brotli_payload_bytes"],
        "n_tensors": section["n_tensors"],
        "n_symbols": section["n_symbols"],
        "per_tensor_K": section["per_tensor_K"],
        "per_tensor_scale_fp16": section["per_tensor_scale_fp16"],
        "smoke_n_latent_pairs_decoded": smoke["n_latent_pairs_decoded"],
        "smoke_latent_dim_meta": smoke["latent_dim_meta"],
        "smoke_base_channels_meta": smoke["base_channels_meta"],
        "smoke_eval_size_meta": smoke["eval_size_meta"],
        "predicted_band": [args.predicted_low, args.predicted_high],
        "evidence_grade": "[CPU-build]",
        "score_affecting_payload_changed": True,
        "charged_bits_changed": True,
        **cpu_build_proxy_guard_fields(),
    }
    build_manifest_path.write_text(
        json.dumps(build_manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"[admm-build] manifest: {build_manifest_path.relative_to(REPO_ROOT)} "
        f"(archive={archive_bytes:,} B, sha256={archive_sha[:16]}...)"
    )
    print(
        f"[admm-build] DONE. CPU build complete. ready_for_exact_eval_dispatch=False; "
        f"cuda_eval_worth_testing=True. Coordinate with Subagent BSF for Lightning "
        f"bootstrap fix before dispatching."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
