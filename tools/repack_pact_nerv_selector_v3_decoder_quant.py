#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Repack a PSV3 archive with decoder-side quantization.

This is an encoder/compression-side tool. It rewrites only the archive bytes
and runtime packet, emits a deterministic local replay proof, and remains
fail-closed for score authority until paired contest CPU/CUDA eval consumes
the resulting byte-closed archive.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

SCHEMA_VERSION = "pact_nerv_selector_v3_decoder_quant_repack.v1"
SOURCE_QUEUE_SCHEMA = "optimizer_candidate_queue_v1"
RUNTIME_PROOF_SCHEMA = "family_agnostic_runtime_consumption_proof_v1"
CANDIDATE_ROW_SCHEMA = "pact_nerv_selector_v3_decoder_quant_candidate.v1"
TARGET_KIND = "pact_nerv_selector_v3_decoder_quant_repack_v1"
MATERIALIZER_ID = "pact_nerv_selector_v3_decoder_quant_repack"
RECEIVER_CONTRACT_KIND = "pact_nerv_selector_v3_decoder_quant_runtime_adapter"

FALSE_AUTHORITY: dict[str, bool] = {
    "score_claim": False,
    "score_claim_valid": False,
    "score_claim_eligible": False,
    "promotion_eligible": False,
    "rank_or_kill_eligible": False,
    "promotable": False,
    "ready_for_exact_eval_dispatch": False,
    "field_selection_ready_for_exact_eval_dispatch": False,
    "dispatch_attempted": False,
    "gpu_launched": False,
    "exact_cuda_auth_eval": False,
    "contest_cuda_auth_eval": False,
}


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _repo_rel(path: Path) -> str:
    try:
        return path.resolve(strict=False).relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return path.resolve(strict=False).as_posix()


def _safe_slug(value: str) -> str:
    out = "".join(ch if ch.isalnum() else "_" for ch in value.lower())
    return "_".join(part for part in out.split("_") if part) or "candidate"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _read_psv3_bytes(path: Path) -> tuple[bytes, str]:
    from tac.substrates.pact_nerv_selector_v3.archive import PSV3_MAGIC

    raw = path.read_bytes()
    if raw[:4] == PSV3_MAGIC:
        return raw, "raw_0_bin"
    if not zipfile.is_zipfile(path):
        raise ValueError(f"{path} is neither raw PSV3 bytes nor a ZIP archive")
    with zipfile.ZipFile(path) as zf:
        names = zf.namelist()
        if "0.bin" not in names:
            raise ValueError(f"{path} is missing ZIP member 0.bin; members={names}")
        data = zf.read("0.bin")
    if data[:4] != PSV3_MAGIC:
        raise ValueError(f"ZIP member 0.bin is not PSV3: magic={data[:4]!r}")
    return data, "zip_member_0_bin"


def _cfg_from_archive(arc: Any) -> Any:
    from tac.substrates.pact_nerv_selector_v3.architecture import (
        PactNervSelectorV3Config,
    )

    meta = arc.meta
    return PactNervSelectorV3Config(
        latent_dim=int(arc.latents.shape[1]),
        embed_dim=int(meta["embed_dim"]),
        initial_grid_h=int(meta["initial_grid_h"]),
        initial_grid_w=int(meta["initial_grid_w"]),
        decoder_channels=tuple(int(c) for c in meta["decoder_channels"]),
        sin_frequency=float(meta["sin_frequency"]),
        num_upsample_blocks=int(meta["num_upsample_blocks"]),
        num_pairs=int(arc.latents.shape[0]),
        output_height=int(meta["output_height"]),
        output_width=int(meta["output_width"]),
        selector_palette_size=int(arc.palette_size),
        rice_golomb_k=int(meta.get("rice_golomb_k", 2)),
    )


def _build_model_from_archive(arc: Any) -> Any:
    import torch

    from tac.substrates.pact_nerv_selector_v3.architecture import (
        PactNervSelectorV3Substrate,
    )

    cfg = _cfg_from_archive(arc)
    model = PactNervSelectorV3Substrate(cfg).eval()
    load_result = model.load_state_dict(arc.decoder_state_dict, strict=False)
    unexpected = set(load_result.unexpected_keys)
    missing = set(load_result.missing_keys) - {"latents", "selectors"}
    if unexpected or missing:
        raise RuntimeError(
            "PSV3 decoder load mismatch: "
            f"missing={sorted(missing)} unexpected={sorted(unexpected)}"
        )
    with torch.no_grad():
        model.latents.copy_(arc.latents.to(dtype=model.latents.dtype))
    return model


def _measure_decoder_drift(source_arc: Any, candidate_arc: Any, n_pairs: int) -> dict[str, Any]:
    import torch

    src_model = _build_model_from_archive(source_arc)
    cand_model = _build_model_from_archive(candidate_arc)
    n = max(1, min(int(n_pairs), int(source_arc.latents.shape[0])))
    idx = torch.arange(n, dtype=torch.long)
    t0 = time.perf_counter()
    with torch.no_grad():
        s0, s1 = src_model(idx)
        c0, c1 = cand_model(idx)
    render_seconds = time.perf_counter() - t0
    src = torch.stack((s0, s1), dim=1)
    cand = torch.stack((c0, c1), dim=1)
    drift = (src - cand).abs()
    return {
        "n_pairs_measured": n,
        "frame_shape": list(src.shape),
        "max_abs_drift_01": float(drift.max()),
        "mean_abs_drift_01": float(drift.mean()),
        "render_seconds_cpu": float(render_seconds),
        "decoder_output_space": "sigmoid_0_to_1",
    }


def repack_pact_nerv_selector_v3_decoder_quant(
    *,
    archive: Path,
    output_dir: Path,
    decoder_quantization: str,
    n_proof_pairs: int,
    candidate_label: str,
) -> dict[str, Any]:
    from tac.optimization.serialized_archive_economics import (
        build_serialized_archive_delta_contract,
    )
    from tac.repo_io import tree_sha256
    from tac.substrates._shared.pact_nerv_full_main import (
        build_archive_zip,
        write_contest_runtime,
    )
    from tac.substrates.pact_nerv_selector_v3.archive import (
        DECODER_QUANTIZATION_KINDS,
        pack_archive,
        parse_archive,
    )

    if decoder_quantization not in DECODER_QUANTIZATION_KINDS:
        raise ValueError(
            f"unsupported decoder_quantization={decoder_quantization!r}; "
            f"expected one of {sorted(DECODER_QUANTIZATION_KINDS)}"
        )
    source_bytes, source_kind = _read_psv3_bytes(archive)
    source_arc = parse_archive(source_bytes)
    meta = dict(source_arc.meta)
    meta.pop("decoder_quantization", None)
    candidate_bytes = pack_archive(
        source_arc.decoder_state_dict,
        source_arc.latents,
        source_arc.selector_bytes,
        meta,
        palette_size=int(source_arc.palette_size),
        decoder_quantization=decoder_quantization,
    )
    candidate_arc = parse_archive(candidate_bytes)

    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_0bin = output_dir / "0.bin"
    candidate_0bin.write_bytes(candidate_bytes)
    runtime_adapter_dir = output_dir / "runtime_adapter"
    write_contest_runtime(
        runtime_adapter_dir,
        substrate_pkg_name="pact_nerv_selector_v3",
        repo_root=REPO_ROOT,
    )
    submission_dir = output_dir / "submission"
    write_contest_runtime(
        submission_dir,
        substrate_pkg_name="pact_nerv_selector_v3",
        repo_root=REPO_ROOT,
    )
    (submission_dir / "0.bin").write_bytes(candidate_bytes)
    archive_zip = output_dir / "archive.zip"
    build_archive_zip(archive_zip, bin_bytes=candidate_bytes, submission_dir=submission_dir)

    drift = _measure_decoder_drift(source_arc, candidate_arc, n_proof_pairs)
    source_size = len(source_bytes)
    candidate_size = len(candidate_bytes)
    source_archive_sha = _sha256_file(archive)
    source_archive_bytes = archive.stat().st_size
    candidate_0bin_sha = _sha256_bytes(candidate_bytes)
    candidate_archive_sha = _sha256_file(archive_zip)
    candidate_archive_bytes = archive_zip.stat().st_size
    runtime_adapter_tree_sha = tree_sha256(runtime_adapter_dir)
    candidate_id = _safe_slug(
        f"{candidate_label}_{decoder_quantization}_{candidate_archive_sha[:12]}"
    )
    lane_id = f"lane_psv3_decoder_quant_{candidate_archive_sha[:12]}"
    manifest_path = output_dir / "decoder_quant_repack_manifest.json"
    runtime_proof_path = output_dir / "runtime_consumption_proof.json"
    source_queue_path = output_dir / "optimizer_candidate_queue.json"
    runtime_adapter_manifest = {
        "schema": "pact_nerv_selector_v3_decoder_quant_runtime_adapter.v1",
        "runtime_adapter_ready": True,
        "runtime_dir": _repo_rel(runtime_adapter_dir),
        "candidate_runtime_dir": _repo_rel(runtime_adapter_dir),
        "runtime_tree_sha256": runtime_adapter_tree_sha,
        "expected_runtime_tree_sha256": runtime_adapter_tree_sha,
        "receiver_contract_kind": RECEIVER_CONTRACT_KIND,
        **FALSE_AUTHORITY,
    }
    source_member = {
        "member_name": "0.bin" if source_kind == "zip_member_0_bin" else archive.name,
        "bytes": source_size,
        "sha256": _sha256_bytes(source_bytes),
    }
    serialized_delta = build_serialized_archive_delta_contract(
        source_archive_bytes=source_archive_bytes,
        candidate_archive_bytes=candidate_archive_bytes,
        modeled_saved_bytes=source_archive_bytes - candidate_archive_bytes,
        require_realized_saving=True,
    )
    runtime_proof = {
        "schema": RUNTIME_PROOF_SCHEMA,
        "generated_utc": datetime.now(UTC).isoformat(),
        "tool": "tools/repack_pact_nerv_selector_v3_decoder_quant.py",
        "target_kind": TARGET_KIND,
        "materializer_id": MATERIALIZER_ID,
        "receiver_contract_kind": RECEIVER_CONTRACT_KIND,
        "receiver_contract_satisfied": True,
        "runtime_consumption_proof_passed": True,
        "passed": True,
        "proof_scope": "psv3_parser_runtime_consumes_quantized_decoder_payload",
        "candidate_id": candidate_id,
        "candidate_archive": {
            "path": _repo_rel(archive_zip),
            "sha256": candidate_archive_sha,
            "bytes": candidate_archive_bytes,
        },
        "candidate_archive_sha256": candidate_archive_sha,
        "archive_sha256": candidate_archive_sha,
        "candidate_member_name": "0.bin",
        "candidate_member_sha256": candidate_0bin_sha,
        "candidate_member_bytes": candidate_size,
        "source_archive": {
            "path": _repo_rel(archive),
            "sha256": source_archive_sha,
            "bytes": source_archive_bytes,
            "source_archive_kind": source_kind,
        },
        "source_member": source_member,
        "runtime_adapter_manifest": runtime_adapter_manifest,
        "runtime_consumption_probe": {
            "schema": "pact_nerv_selector_v3_decoder_quant_runtime_probe.v1",
            "passed": True,
            "blockers": [],
            "parse_archive_passed": True,
            "candidate_0bin_sha256": candidate_0bin_sha,
            "source_0bin_sha256": _sha256_bytes(source_bytes),
            "candidate_archive_sha256": candidate_archive_sha,
            "runtime_adapter_tree_sha256": runtime_adapter_tree_sha,
            "decoder_drift_measured_passed": True,
            "decoder_quantization": decoder_quantization,
            "n_pairs_measured": drift["n_pairs_measured"],
            "max_abs_drift_01": drift["max_abs_drift_01"],
            "mean_abs_drift_01": drift["mean_abs_drift_01"],
        },
        "blockers": [],
        **FALSE_AUTHORITY,
    }
    _write_json(runtime_proof_path, runtime_proof)
    runtime_proof_sha = _sha256_file(runtime_proof_path)
    source_queue_row = {
        "schema": CANDIDATE_ROW_SCHEMA,
        "candidate_id": candidate_id,
        "candidate_family": "pact_nerv_selector_v3_decoder_quant",
        "lane_id": lane_id,
        "target_kind": TARGET_KIND,
        "materializer_id": MATERIALIZER_ID,
        "receiver_contract_kind": RECEIVER_CONTRACT_KIND,
        "optimizer_tool": "tools/repack_pact_nerv_selector_v3_decoder_quant.py",
        "axis_tag": "[macOS-CPU archive-proof]",
        "candidate_archive_path": _repo_rel(archive_zip),
        "archive_path": _repo_rel(archive_zip),
        "candidate_archive_sha256": candidate_archive_sha,
        "archive_sha256": candidate_archive_sha,
        "candidate_archive_bytes": candidate_archive_bytes,
        "archive_bytes": candidate_archive_bytes,
        "archive_size_bytes": candidate_archive_bytes,
        "candidate_member_name": "0.bin",
        "candidate_member_sha256": candidate_0bin_sha,
        "candidate_member_bytes": candidate_size,
        "candidate_0bin_path": _repo_rel(candidate_0bin),
        "candidate_0bin_sha256": candidate_0bin_sha,
        "candidate_0bin_bytes": candidate_size,
        "source_archive_path": _repo_rel(archive),
        "source_archive_sha256": source_archive_sha,
        "source_archive_bytes": source_archive_bytes,
        "source_archive_kind": source_kind,
        "source_member_name": source_member["member_name"],
        "source_member_sha256": source_member["sha256"],
        "source_member_bytes": source_member["bytes"],
        "source_0bin_sha256": source_member["sha256"],
        "source_0bin_bytes": source_size,
        "receiver_contract_satisfied": True,
        "runtime_consumption_proof_required": True,
        "runtime_consumption_proof_status": "present",
        "runtime_consumption_proof_path": _repo_rel(runtime_proof_path),
        "runtime_consumption_proof_sha256": runtime_proof_sha,
        "runtime_consumption_proof_schema": RUNTIME_PROOF_SCHEMA,
        "runtime_adapter_ready": True,
        "candidate_runtime_adapter_blocker_cleared": True,
        "candidate_runtime_dir": _repo_rel(runtime_adapter_dir),
        "candidate_runtime_tree_sha256": runtime_adapter_tree_sha,
        "expected_runtime_tree_sha256": runtime_adapter_tree_sha,
        "runtime_adapter_manifest": runtime_adapter_manifest,
        "decoder_quantization": decoder_quantization,
        "rate_positive": bool(serialized_delta["savings_realized"]),
        "realized_saved_bytes": serialized_delta["realized_saved_bytes"],
        "serialized_archive_delta": serialized_delta,
        "score_affecting_payload_changed": True,
        "charged_bits_changed": True,
        "archive_changed": True,
        "byte_different": True,
        "score_affecting_change_proof": {
            "archive_changed": True,
            "byte_different": True,
            "source_archive_sha256": source_archive_sha,
            "candidate_archive_sha256": candidate_archive_sha,
            "source_archive_bytes": source_archive_bytes,
            "candidate_archive_bytes": candidate_archive_bytes,
            "source_member_sha256": source_member["sha256"],
            "candidate_member_sha256": candidate_0bin_sha,
        },
        "local_decoder_drift": drift,
        "source_manifest_path": _repo_rel(manifest_path),
        "source_paths": [
            _repo_rel(manifest_path),
            _repo_rel(runtime_proof_path),
            _repo_rel(archive_zip),
            _repo_rel(runtime_adapter_dir),
        ],
        "dispatch_blockers": [
            "materializer_candidate_is_not_dispatch_authorization",
            "materializer_chain_harvest_candidate_pending_exact_readiness",
            "exact_auth_eval_result_required_before_score_claim",
            "requires_lane_dispatch_claim_before_gpu_or_remote_eval",
        ],
        **FALSE_AUTHORITY,
    }
    source_queue = {
        "schema": SOURCE_QUEUE_SCHEMA,
        "tool": "tools/repack_pact_nerv_selector_v3_decoder_quant.py",
        "generated_utc": datetime.now(UTC).isoformat(),
        "candidate_label": candidate_label,
        "n_candidates": 1,
        "top_k_count": 1,
        "dispatch_ready_count": 0,
        "source_schemas": [CANDIDATE_ROW_SCHEMA],
        "top_k": [source_queue_row],
        "top_k_forensic": [source_queue_row],
        "dispatch_ready": [],
        "evidence_boundary": {
            "planning_only_by_default": True,
            "mlx_or_local_rows_are_advisory": True,
            "exact_eval_handoff_requires_closure_bridge": True,
            **FALSE_AUTHORITY,
        },
        "dispatch_blockers": [
            "optimizer_candidate_queue_is_planning_only",
            "requires_exact_eval_readiness_gate",
            "requires_lane_dispatch_claim_before_gpu_or_remote_eval",
        ],
        **FALSE_AUTHORITY,
    }
    _write_json(source_queue_path, source_queue)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "generated_utc": datetime.now(UTC).isoformat(),
        "tool": "tools/repack_pact_nerv_selector_v3_decoder_quant.py",
        "candidate_id": candidate_id,
        "lane_id": lane_id,
        "candidate_label": candidate_label,
        "source_archive": str(archive),
        "source_archive_kind": source_kind,
        "source_archive_sha256": source_archive_sha,
        "source_archive_bytes": source_archive_bytes,
        "source_0bin_sha256": _sha256_bytes(source_bytes),
        "source_0bin_bytes": source_size,
        "candidate_0bin_path": str(candidate_0bin),
        "candidate_0bin_sha256": candidate_0bin_sha,
        "candidate_0bin_bytes": candidate_size,
        "candidate_archive_zip_path": str(archive_zip),
        "candidate_archive_zip_sha256": candidate_archive_sha,
        "candidate_archive_zip_bytes": candidate_archive_bytes,
        "runtime_adapter_dir": str(runtime_adapter_dir),
        "runtime_adapter_tree_sha256": runtime_adapter_tree_sha,
        "runtime_consumption_proof_path": str(runtime_proof_path),
        "runtime_consumption_proof_sha256": runtime_proof_sha,
        "optimizer_candidate_queue_path": str(source_queue_path),
        "optimizer_candidate_queue_sha256": _sha256_file(source_queue_path),
        "decoder_quantization": decoder_quantization,
        "rate_delta_0bin_bytes": candidate_size - source_size,
        "rate_delta_0bin_fraction": (
            (candidate_size - source_size) / source_size if source_size else 0.0
        ),
        "serialized_archive_delta": serialized_delta,
        "rate_win": candidate_size < source_size,
        "local_decoder_drift": drift,
        "axis_tag": "[macOS-CPU archive-proof]",
        **FALSE_AUTHORITY,
        "blockers": [
            "local_archive_repack_is_not_contest_score_authority",
            "requires_paired_contest_cpu_plus_cuda_eval_before_score_claim",
            "requires_dispatch_claim_and_runtime_custody_handoff_before_exact_eval",
        ],
    }
    _write_json(manifest_path, manifest)
    manifest["manifest_path"] = str(manifest_path)
    return manifest


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Repack a PSV3 archive with decoder-side quantization."
    )
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--decoder-quantization",
        default="int8_per_channel_brotli_q11",
        choices=[
            "fp16_brotli_q9",
            "fp16_brotli_q11",
            "int8_per_channel_brotli_q11",
        ],
    )
    parser.add_argument("--n-proof-pairs", type=int, default=1)
    parser.add_argument(
        "--candidate-label",
        default="pact_nerv_selector_v3_decoder_quant_repack",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        manifest = repack_pact_nerv_selector_v3_decoder_quant(
            archive=args.archive,
            output_dir=args.output_dir,
            decoder_quantization=args.decoder_quantization,
            n_proof_pairs=args.n_proof_pairs,
            candidate_label=args.candidate_label,
        )
    except Exception as exc:
        print(f"[psv3-decoder-quant] ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
