#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build local PR81/QMA9 + PR82/Henosis stack candidates.

This is a local custody and byte-accounting tool.  It never invokes the scorer,
never dispatches remote/GPU work, and every emitted archive is fail-closed until
the runtime can actually consume the PR81 QMA9 payload and PR82 QRM1/frame-1
randmulti semantics in the scored inflate path.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import zipfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tac.henosis_pr82_transfer import (
    QPOST_STREAM_NAMES,
    decode_randmulti_groups,
    decode_randmulti_qrm1,
    encode_qpost,
    encode_randmulti_nm2,
    encode_randmulti_qrm1,
    parse_pr82_bundle,
    parse_replay_contract,
    randmulti_group_qps1_nm2_compatible,
    randmulti_group_summary,
    randmulti_qrm1_parity_profile,
    sha256_bytes,
    sha256_path,
)
from tac.qma9_range_mask_contract import (
    read_single_member_zip,
    split_qma9_pr81_payload,
)


TOOL = "experiments/build_pr81_pr82_henosis_stack_candidate.py"
SCHEMA = "pr81_pr82_henosis_stack_candidate_v1"
ORIGINAL_VIDEO_BYTES = 37_545_489
RATE_SCORE_PER_BYTE = 25.0 / ORIGINAL_VIDEO_BYTES
PR81_EXACT_T4_SCORE = 0.2812078926981712  # [external: PR-81 contest-CUDA T4 exact-eval]
PR81_EXACT_T4_BYTES = 215_960
PR82_EXACT_T4_SCORE = 0.2983246102939779  # [external: PR-82 contest-CUDA T4 exact-eval]
PR82_EXACT_T4_BYTES = 296_789
PR82_SEGNET_DIST = 0.00057185
PR82_POSENET_DIST = 0.0001894
EXPECTED_PR81_SHA256 = "cd01378a52688fe00ee1bfb898c67695aed89a7b3ded602b597eb7fc3031d7fc"
EXPECTED_PR82_SHA256 = "a0e07c360223c1dd3d3b92263225d38d542e218e83d095ad9b91bf872f94c6e4"
DEFAULT_PR81_DIR = REPO_ROOT / "experiments/results/public_pr81_qzs3_range_mask_intake_20260503_codex"
DEFAULT_PR82_DIR = REPO_ROOT / "experiments/results/public_pr82_henosis_frontier_intake_20260503_codex"
DEFAULT_PR81_ARCHIVE = DEFAULT_PR81_DIR / "archive.zip"
DEFAULT_PR81_PROFILE = DEFAULT_PR81_DIR / "pr81_qma9_semantic_range_mask_profile.json"
DEFAULT_PR82_ARCHIVE = DEFAULT_PR82_DIR / "archive.zip"
DEFAULT_REPLAY_INFLATE = DEFAULT_PR82_DIR / "replay_submission/inflate.py"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "experiments/results/pr81_pr82_henosis_stack_20260503_codex"
RUNTIME_APPLY_PATH = REPO_ROOT / "submissions/robust_current/apply_qzs3_postprocess.py"
RUNTIME_INFLATE_RENDERER_PATH = REPO_ROOT / "submissions/robust_current/inflate_renderer.py"
FIXED_ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)


class Pr81Pr82StackError(ValueError):
    """Raised when the local stack candidate cannot be built safely."""


def _json_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_json_bytes(payload))


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise Pr81Pr82StackError(f"expected JSON object: {path}")
    return payload


def _repo_rel(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, FIXED_ZIP_TIMESTAMP)
    info.compress_type = zipfile.ZIP_STORED
    info.external_attr = 0o644 << 16
    info.create_system = 3
    return info


def _write_archive(path: Path, members: Mapping[str, bytes]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as zf:
        for name in sorted(members):
            member_path = Path(name)
            if not name or name.startswith("/") or ".." in member_path.parts or len(member_path.parts) != 1:
                raise Pr81Pr82StackError(f"unsafe member name: {name!r}")
            zf.writestr(_zip_info(name), members[name])


def _load_runtime_apply() -> Any:
    spec = importlib.util.spec_from_file_location("robust_current_apply_qzs3_postprocess", RUNTIME_APPLY_PATH)
    if spec is None or spec.loader is None:
        raise Pr81Pr82StackError(f"cannot load runtime qpost helper: {RUNTIME_APPLY_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_runtime_inflate_renderer() -> Any:
    spec = importlib.util.spec_from_file_location("robust_current_inflate_renderer", RUNTIME_INFLATE_RENDERER_PATH)
    if spec is None or spec.loader is None:
        raise Pr81Pr82StackError(f"cannot load runtime inflate renderer: {RUNTIME_INFLATE_RENDERER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _pr81_reordered_model_restore_preflight(model_payload: bytes) -> dict[str, Any]:
    """Prove robust_current can restore PR81's reordered QZS3 model bytes."""

    runtime = _load_runtime_inflate_renderer()
    try:
        restored = runtime._restore_pr81_reordered_qzs3_model_payload(model_payload)
    except Exception as exc:
        raise Pr81Pr82StackError(
            f"PR81 reordered QZS3 restore preflight failed: {type(exc).__name__}: {exc}"
        ) from exc
    if not isinstance(restored, (bytes, bytearray)):
        raise Pr81Pr82StackError("PR81 reordered QZS3 restore preflight returned non-bytes")
    restored_bytes = bytes(restored)
    if len(restored_bytes) < 6 or not restored_bytes.startswith(b"QZS3"):
        raise Pr81Pr82StackError("PR81 reordered QZS3 restore preflight did not emit QZS3 bytes")
    block_size = int.from_bytes(restored_bytes[4:6], "little")
    if block_size <= 0:
        raise Pr81Pr82StackError(f"PR81 reordered QZS3 restore emitted invalid block size {block_size}")
    return {
        "input_model_payload_bytes": len(model_payload),
        "input_model_payload_sha256": sha256_bytes(model_payload),
        "runtime_inflate_renderer": _repo_rel(RUNTIME_INFLATE_RENDERER_PATH),
        "runtime_inflate_renderer_sha256": sha256_path(RUNTIME_INFLATE_RENDERER_PATH),
        "restored_block_size": block_size,
        "restored_model_bytes": len(restored_bytes),
        "restored_model_sha256": sha256_bytes(restored_bytes),
        "status": "passed",
    }


def _read_archive_member(path: Path, name: str) -> bytes:
    with zipfile.ZipFile(path, "r") as zf:
        names = [info.filename for info in zf.infolist() if not info.is_dir()]
        if name not in names:
            raise Pr81Pr82StackError(f"{path} missing required member {name!r}; members={names!r}")
        return zf.read(name)


def _synthetic_raw_pair(pair_index: int, *, height: int, width: int, archive_sha256: str) -> np.ndarray:
    seed = int(archive_sha256[:16], 16) ^ (int(pair_index) * 0x9E3779B185EBCA87)
    rng = np.random.default_rng(seed & 0xFFFFFFFFFFFFFFFF)
    # Keep values away from 0/255 so small PR82 integer biases and shifts are visible after clamp.
    return rng.integers(32, 224, size=(1, 2, height, width, 3), dtype=np.uint8)


def _synthetic_source_mask_pair(pair_index: int, *, height: int = 384, width: int = 512) -> torch.Tensor:
    yy, xx = np.indices((height, width), dtype=np.int64)
    classes = ((yy // 32) + (xx // 32) + int(pair_index)) % 5
    return torch.from_numpy(classes.reshape(1, height, width).astype(np.int64, copy=False))


def _qpost_active_pair_order(runtime: Any, qpost_path: Path, *, device: torch.device) -> list[int]:
    try:
        state = runtime.read_qpost(qpost_path, device)
    except Exception as exc:
        raise Pr81Pr82StackError(f"runtime qpost parse/apply failed closed: {type(exc).__name__}: {exc}") from exc
    counts = np.zeros(600, dtype=np.int32)
    if state.postprocess is not None:
        for _gains, _biases, choices in state.postprocess:
            arr = choices.detach().cpu().numpy()
            counts[: arr.shape[0]] += (arr[:600] != 0).astype(np.int32)
    for choices, default in (
        (state.f0_shift, 40),
        (state.f1_frac, 4),
        (state.f1_frac2, 4),
        (state.f1_frac3, 4),
        (state.f1_bias, 13),
        (state.f1_region, 0),
    ):
        if choices is None:
            continue
        arr = choices.detach().cpu().numpy()
        counts[: arr.shape[0]] += (arr[:600] != default).astype(np.int32)
    if state.f1_randmulti is not None:
        for arr_choices, _lh, _lw, _amp in state.f1_randmulti:
            arr = arr_choices.detach().cpu().numpy()
            counts += np.count_nonzero(arr, axis=0).astype(np.int32)
    active = np.flatnonzero(counts)
    if active.size:
        return [int(idx) for idx in active[np.argsort(-counts[active], kind="stable")]]
    return list(range(600))


def _synthetic_raw_delta_proof(
    *,
    archive_path: Path,
    candidate_dir: Path,
    qpost: bytes,
    height: int = 874,
    width: int = 1164,
    device_name: str = "cpu",
) -> dict[str, Any]:
    """Attach a deterministic no-op guard for the exact archive qpost bytes.

    This does not claim PR81/PR82 component carryover.  It proves the charged
    sidecar is parsed by the contest runtime helper and changes raw RGB values
    for at least one pair under deterministic, nonsaturated input.
    """

    qpost_path = candidate_dir / "qpost_for_raw_delta_proof.bin"
    qpost_path.write_bytes(qpost)
    archive_sha = sha256_path(archive_path)
    runtime = _load_runtime_apply()
    device = torch.device(device_name)
    try:
        state = runtime.read_qpost(qpost_path, device)
    except Exception as exc:
        raise Pr81Pr82StackError(f"runtime qpost parse/apply failed closed: {type(exc).__name__}: {exc}") from exc
    grid_cache: dict[tuple[float, int, int, int], torch.Tensor] = {}
    randpat_cache: dict[tuple[int, int, int, int], torch.Tensor] = {}
    errors: list[str] = []
    for proof_pair_index in _qpost_active_pair_order(runtime, qpost_path, device=device):
        before = _synthetic_raw_pair(proof_pair_index, height=height, width=width, archive_sha256=archive_sha)
        batch = torch.from_numpy(before).to(device=device, dtype=torch.float32)
        try:
            source_masks = None
            if hasattr(runtime, "qpost_requires_source_masks") and runtime.qpost_requires_source_masks(state):
                source_masks = _synthetic_source_mask_pair(proof_pair_index).to(device)
            after_tensor = runtime.apply_qpost_batch(
                batch,
                pair_start=proof_pair_index,
                state=state,
                grid_cache=grid_cache,
                randpat_cache=randpat_cache,
                source_masks=source_masks,
            )
        except Exception as exc:  # pragma: no cover - surfaced as fail-closed manifest blocker.
            errors.append(f"pair {proof_pair_index}: {type(exc).__name__}: {exc}")
            continue
        after = after_tensor.clamp(0, 255).round().to(torch.uint8).cpu().numpy()
        diff = after.astype(np.int16) - before.astype(np.int16)
        changed = int(np.count_nonzero(diff))
        if changed <= 0:
            continue
        proof = {
            "archive_bytes": archive_path.stat().st_size,
            "archive_path": _repo_rel(archive_path),
            "archive_sha256": archive_sha,
            "after_bytes": int(after.nbytes),
            "after_sha256": sha256_bytes(after.tobytes()),
            "before_bytes": int(before.nbytes),
            "before_sha256": sha256_bytes(before.tobytes()),
            "changed_values": changed,
            "compared_values": int(diff.size),
            "device": str(device),
            "exact_equal": False,
            "height": height,
            "max_abs_delta": int(np.abs(diff).max()),
            "mean_abs_delta": float(np.abs(diff).mean()),
            "proof_pair_index": int(proof_pair_index),
            "proof_scope": "synthetic_raw_noop_guard_only_not_component_parity",
            "proof_type": "runtime_apply_qzs3_postprocess_synthetic_raw_pair",
            "qpost_path": _repo_rel(qpost_path),
            "qpost_sha256": sha256_path(qpost_path),
            "runtime_helper": _repo_rel(RUNTIME_APPLY_PATH),
            "runtime_helper_sha256": sha256_path(RUNTIME_APPLY_PATH),
            "width": width,
        }
        _write_json(candidate_dir / "raw_output_delta_proof.json", proof)
        return proof
    raise Pr81Pr82StackError(
        "qpost sidecar was a raw-output no-op under deterministic synthetic proof"
        + (f"; runtime errors: {errors[:3]}" if errors else "")
    )


def _classify_qrm1_stream(qrm1: bytes) -> dict[str, Any]:
    runtime = _load_runtime_apply()
    try:
        report = runtime.classify_qrm1_randmulti_stream(qrm1)
    except Exception as exc:
        raise Pr81Pr82StackError(f"runtime QRM1 support classification failed: {type(exc).__name__}: {exc}") from exc
    if not isinstance(report, dict):
        raise Pr81Pr82StackError("runtime QRM1 support classifier returned a non-object report")
    return report


def _qrm1_exclusion_report(
    groups: Sequence[Any],
    *,
    support_report: Mapping[str, Any],
) -> dict[str, Any]:
    active_unsupported = sorted(int(value) for value in support_report.get("active_unsupported_group_ids", []))
    group_by_id = {int(group.group_index): group for group in groups}
    rows_by_id = {
        int(row["group_id"]): row
        for row in support_report.get("group_rows", [])
        if isinstance(row, dict) and "group_id" in row
    }
    excluded = []
    for group_id in active_unsupported:
        group = group_by_id[group_id]
        row = rows_by_id.get(group_id, {})
        excluded.append(
            {
                "amplitude": int(group.amplitude),
                "group_index": group_id,
                "height": int(group.height),
                "nonzero_choice_total": int(np.count_nonzero(group.rows)),
                "reason": str(row.get("reason", "runtime does not support this active QRM1 group")),
                "scount": int(group.scount),
                "width": int(group.width),
            }
        )
    return {
        "dispatchable_before_exclusion": bool(support_report.get("dispatchable_qrm1")),
        "excluded_active_unsupported_group_count": len(excluded),
        "excluded_active_unsupported_group_ids": active_unsupported,
        "excluded_active_unsupported_groups": excluded,
        "policy": "drop only active PR82 QRM1 groups rejected by robust_current runtime support classifier",
        "support_classifier": _repo_rel(RUNTIME_APPLY_PATH),
        "support_classifier_sha256": sha256_path(RUNTIME_APPLY_PATH),
    }


def contest_score_from_components(archive_bytes: int, *, segnet_dist: float, posenet_dist: float) -> float:
    return 100.0 * float(segnet_dist) + (10.0 * float(posenet_dist)) ** 0.5 + 25.0 * int(archive_bytes) / ORIGINAL_VIDEO_BYTES


def _load_pr81_source(path: Path, profile_json: Path, *, expected_sha256: str | None) -> dict[str, Any]:
    path = path.resolve()
    if expected_sha256 is not None:
        actual = sha256_path(path)
        if actual != expected_sha256:
            raise Pr81Pr82StackError(f"PR81 archive SHA mismatch: expected {expected_sha256}, got {actual}")
    payload, zip_profile = read_single_member_zip(path, expected_member="p")
    profile = _read_json(profile_json)
    constants = profile.get("split_constants")
    if not isinstance(constants, dict):
        raise Pr81Pr82StackError(f"PR81 profile lacks split_constants: {profile_json}")
    split = split_qma9_pr81_payload(
        payload,
        range_mask_bytes=int(constants["RANGE_MASK_BYTES"]),
        model_bytes=int(constants["SPLIT_MODEL_REORDERED_BYTES"]),
        pose_bytes=int(constants["POSE_STREAM_BYTES"]),
        router_bytes=int(constants["ROUTER_ACTION_BYTES"]),
    )
    restore_preflight = _pr81_reordered_model_restore_preflight(split.model)
    return {
        "archive": zip_profile,
        "payload": payload,
        "profile_json": profile_json,
        "qma9": profile.get("qma9", {}),
        "runtime_restore_preflight": restore_preflight,
        "split": split,
    }


def _load_pr82_source(path: Path, replay_inflate: Path, *, expected_sha256: str | None) -> dict[str, Any]:
    path = path.resolve()
    if expected_sha256 is not None:
        actual = sha256_path(path)
        if actual != expected_sha256:
            raise Pr81Pr82StackError(f"PR82 archive SHA mismatch: expected {expected_sha256}, got {actual}")
    with zipfile.ZipFile(path, "r") as zf:
        infos = [info for info in zf.infolist() if not info.is_dir()]
        names = [info.filename for info in infos]
        if names != ["x"]:
            raise Pr81Pr82StackError(f"{path}: expected single member 'x', got {names!r}")
        raw = zf.read("x")
    contract = parse_replay_contract(replay_inflate)
    bundle = parse_pr82_bundle(raw, contract)
    return {
        "archive_bytes": path.stat().st_size,
        "archive_path": path,
        "archive_sha256": sha256_path(path),
        "bundle": bundle,
        "contract": contract,
        "payload_bytes": len(raw),
        "payload_sha256": sha256_bytes(raw),
    }


def _qpost_archive_manifest(
    *,
    archive: Path,
    candidate_id: str,
    candidate_kind: str,
    pr81: Mapping[str, Any],
    qpost: bytes,
    qpost_contract: str,
    changed_atoms: int,
    runtime_blockers: Sequence[str],
    raw_output_delta_proof: Mapping[str, Any] | None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    archive_bytes = archive.stat().st_size
    candidate_score_if_pr82_components_carry = contest_score_from_components(
        archive_bytes,
        segnet_dist=PR82_SEGNET_DIST,
        posenet_dist=PR82_POSENET_DIST,
    )
    blockers = list(runtime_blockers)
    if changed_atoms <= 0:
        blockers.append("candidate is a no-op after stream filtering")
    if raw_output_delta_proof is None:
        blockers.append("missing positive raw-output delta proof")
    dispatch_ready = len(blockers) == 0
    manifest = {
        "archive_bytes": archive_bytes,
        "archive_path": _repo_rel(archive),
        "archive_sha256": sha256_path(archive),
        "candidate_id": candidate_id,
        "candidate_kind": candidate_kind,
        "dispatch_gate": {
            "dispatch_ready_now": dispatch_ready,
            "lane_claim_required_before_any_exact_eval": True,
            "no_remote_dispatch": True,
            "reason": "ready_for_exact_cuda_eval_after_lane_claim" if dispatch_ready else "; ".join(blockers),
            "remote_dispatch_performed": False,
            "score_claim": False,
        },
        "evidence_grade": "empirical_local_archive_build_and_static_lower_bound",
        "manifest_schema": SCHEMA,
        "no_op_detection": {
            "changed_atom_count": int(changed_atoms),
            "is_noop": int(changed_atoms) == 0,
        },
        "output_archive": {
            "bytes": archive_bytes,
            "path": _repo_rel(archive),
            "sha256": sha256_path(archive),
        },
        "qpost": {
            "bytes": len(qpost),
            "runtime_contract": qpost_contract,
            "sha256": sha256_bytes(qpost),
        },
        "raw_output_delta_proof": dict(raw_output_delta_proof) if raw_output_delta_proof is not None else None,
        "score_claim": False,
        "source_pr81_archive": {
            "bytes": int(pr81["archive"].archive_bytes),
            "path": _repo_rel(Path(pr81["archive"].archive_path)),
            "sha256": pr81["archive"].archive_sha256,
        },
        "source_pr81_runtime_restore_preflight": dict(pr81["runtime_restore_preflight"]),
        "static_score_band_if_pr82_components_carried": {
            "archive_bytes": archive_bytes,
            "bytes_delta_vs_pr81": archive_bytes - int(pr81["archive"].archive_bytes),
            "component_assumption": "PR82 exact T4 components copied unchanged; planning-only lower bound, not evidence",
            "expected_score": candidate_score_if_pr82_components_carry,
            "pr82_posenet_dist": PR82_POSENET_DIST,
            "pr82_segnet_dist": PR82_SEGNET_DIST,
        },
        "stream_delta": {
            "archive_delta_bytes_vs_pr81": archive_bytes - int(pr81["archive"].archive_bytes),
            "qpost_charged_member_bytes": len(qpost),
        },
        "tool": TOOL,
    }
    if extra:
        manifest.update(extra)
    return manifest


def _changed_qpost_atoms_from_pr82(bundle: Any, *, include_randmulti: bool) -> int:
    from tac.henosis_pr82_transfer import decode_control_arrays

    arrays = decode_control_arrays(bundle.encoded_segments)
    total = 0
    defaults = {"shift": 40, "frac": 4, "frac2": 4, "frac3": 4, "bias": 13, "region": 0}
    for name, arr in arrays.items():
        if name == "post":
            total += int(np.count_nonzero(arr != 0))
        else:
            total += int(np.count_nonzero(arr != defaults[name]))
    if include_randmulti:
        # Filled by callers that already decoded the exact group representation.
        return total
    return total


def _build_qpost_candidate(
    *,
    pr81: Mapping[str, Any],
    streams: Mapping[str, bytes],
    candidate_id: str,
    candidate_kind: str,
    qpost_contract: str,
    changed_atoms: int,
    runtime_blockers: Sequence[str],
    output_dir: Path,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    qpost = encode_qpost(streams)
    archive = output_dir / candidate_id / "archive.zip"
    _write_archive(archive, {"p": pr81["payload"], "qpost.bin": qpost})
    candidate_dir = output_dir / candidate_id
    raw_output_delta_proof = None
    try:
        raw_output_delta_proof = _synthetic_raw_delta_proof(
            archive_path=archive,
            candidate_dir=candidate_dir,
            qpost=qpost,
        )
    except Pr81Pr82StackError as exc:
        runtime_blockers = [*runtime_blockers, str(exc)]
    manifest = _qpost_archive_manifest(
        archive=archive,
        candidate_id=candidate_id,
        candidate_kind=candidate_kind,
        pr81=pr81,
        qpost=qpost,
        qpost_contract=qpost_contract,
        changed_atoms=changed_atoms,
        runtime_blockers=runtime_blockers,
        raw_output_delta_proof=raw_output_delta_proof,
        extra=extra,
    )
    _write_json(output_dir / candidate_id / "manifest.json", manifest)
    return {
        "archive_bytes": manifest["output_archive"]["bytes"],
        "archive_path": manifest["output_archive"]["path"],
        "candidate_id": candidate_id,
        "candidate_kind": candidate_kind,
        "dispatch_gate": manifest["dispatch_gate"],
        "manifest_path": _repo_rel(output_dir / candidate_id / "manifest.json"),
        "qpost_bytes": len(qpost),
        "score_if_pr82_components_carried": manifest["static_score_band_if_pr82_components_carried"]["expected_score"],
        "stream_delta": manifest["stream_delta"],
    }


def static_lower_bounds(*, pr81: Mapping[str, Any], pr82: Mapping[str, Any]) -> list[dict[str, Any]]:
    split = pr81["split"]
    bundle = pr82["bundle"]
    pr82_segments = bundle.encoded_segments
    header_bytes = 24
    zip_overhead = 100

    def score(bytes_: int) -> float:
        return contest_score_from_components(bytes_, segnet_dist=PR82_SEGNET_DIST, posenet_dist=PR82_POSENET_DIST)

    compact_model_pose = (
        zip_overhead
        + header_bytes
        + len(split.range_mask)
        + len(pr82_segments["model"])
        + len(pr82_segments["pose"])
    )
    compact_controls = compact_model_pose + sum(
        len(pr82_segments[name])
        for name in ("post", "shift", "frac", "frac2", "frac3", "bias", "region")
    )
    compact_full = compact_controls + len(pr82_segments["randmulti"])
    pr82_with_pr81_mask = int(pr82["archive_bytes"]) - len(pr82_segments["mask"]) + len(split.range_mask)
    rows = [
        {
            "label": "ideal_pr81_archive_bytes_with_pr82_components",
            "archive_bytes_lower_bound": int(pr81["archive"].archive_bytes),
            "expected_score_if_pr82_components_carry": score(int(pr81["archive"].archive_bytes)),
            "runtime_status": "not_a_build; component carryover assumption only",
        },
        {
            "label": "compact_x_pr81_mask_pr82_model_pose_only",
            "archive_bytes_lower_bound": compact_model_pose,
            "expected_score_if_pr82_components_carry": score(compact_model_pose),
            "runtime_status": "unsupported: PR82 replay contract requires control tails and PR81 QMA9 mask loader bridge",
        },
        {
            "label": "compact_x_pr81_mask_pr82_model_pose_post_controls",
            "archive_bytes_lower_bound": compact_controls,
            "expected_score_if_pr82_components_carry": score(compact_controls),
            "runtime_status": "unsupported: PR82 replay contract requires randmulti tail handling and QMA9 mask loader bridge",
        },
        {
            "label": "compact_x_pr81_mask_pr82_model_pose_post_randmulti",
            "archive_bytes_lower_bound": compact_full,
            "expected_score_if_pr82_components_carry": score(compact_full),
            "runtime_status": "unsupported: exact PR82 compact semantics with PR81 QMA9 mask not implemented",
        },
        {
            "label": "pr82_archive_with_pr81_mask_byte_substitution",
            "archive_bytes_lower_bound": pr82_with_pr81_mask,
            "expected_score_if_pr82_components_carry": score(pr82_with_pr81_mask),
            "runtime_status": "unsupported: static byte substitution lower bound only",
        },
    ]
    return rows


def build_candidates(
    *,
    pr81_archive: Path = DEFAULT_PR81_ARCHIVE,
    pr81_profile_json: Path = DEFAULT_PR81_PROFILE,
    pr82_archive: Path = DEFAULT_PR82_ARCHIVE,
    replay_inflate: Path = DEFAULT_REPLAY_INFLATE,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    expected_pr81_sha256: str | None = EXPECTED_PR81_SHA256,
    expected_pr82_sha256: str | None = EXPECTED_PR82_SHA256,
) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    pr81 = _load_pr81_source(pr81_archive, pr81_profile_json, expected_sha256=expected_pr81_sha256)
    pr82 = _load_pr82_source(pr82_archive, replay_inflate, expected_sha256=expected_pr82_sha256)
    bundle = pr82["bundle"]
    groups = decode_randmulti_groups(bundle.encoded_segments["randmulti"], pr82["contract"].randmulti_specs)
    qrm1 = encode_randmulti_qrm1(groups)
    qrm1_support_report = _classify_qrm1_stream(qrm1)
    qrm1_exclusion = _qrm1_exclusion_report(groups, support_report=qrm1_support_report)
    qrm1_excluded_group_ids = set(qrm1_exclusion["excluded_active_unsupported_group_ids"])
    qrm1_supported_groups = [
        group for group in groups if int(group.group_index) not in qrm1_excluded_group_ids
    ]
    qrm1_supported_subset = encode_randmulti_qrm1(qrm1_supported_groups)
    qrm1_supported_subset_report = _classify_qrm1_stream(qrm1_supported_subset)
    if not bool(qrm1_supported_subset_report.get("dispatchable_qrm1")):
        raise Pr81Pr82StackError(
            "QRM1 subset/exclusion policy did not produce a runtime-dispatchable stream: "
            f"{qrm1_supported_subset_report}"
        )
    qrm1_decoded = decode_randmulti_qrm1(qrm1, pr82["contract"].randmulti_specs)
    qrm1_profile = randmulti_qrm1_parity_profile(
        groups,
        qrm1_decoded,
        encoded=qrm1,
        source_encoded=bundle.encoded_segments["randmulti"],
    )
    qrm1_supported_decoded = decode_randmulti_qrm1(qrm1_supported_subset, pr82["contract"].randmulti_specs)
    qrm1_supported_profile = randmulti_qrm1_parity_profile(
        qrm1_supported_groups,
        qrm1_supported_decoded,
        encoded=qrm1_supported_subset,
        source_encoded=bundle.encoded_segments["randmulti"],
    )
    compatible_groups = [group for group in groups if randmulti_group_qps1_nm2_compatible(group)]
    compatible_groups = sorted(
        compatible_groups,
        key=lambda group: (-int(np.count_nonzero(group.rows)), int(group.group_index)),
    )
    controls_streams = {name: bundle.encoded_segments[name] for name in QPOST_STREAM_NAMES if name != "randmulti"}
    controls_streams["randmulti"] = b""
    control_atoms = _changed_qpost_atoms_from_pr82(bundle, include_randmulti=False)
    common_pr81_runtime_blocker: list[str] = []
    qps1_controls = _build_qpost_candidate(
        pr81=pr81,
        streams=controls_streams,
        candidate_id="pr81_qma9_pr82_qps1_controls_all600",
        candidate_kind="pr81_qma9_payload_plus_pr82_qps1_control_sidecar",
        qpost_contract="QPS1/post_shift_frac_bias_region_without_randmulti",
        changed_atoms=control_atoms,
        runtime_blockers=common_pr81_runtime_blocker,
        output_dir=output_dir,
    )

    qps1_nm2 = None
    if compatible_groups:
        nm2 = encode_randmulti_nm2(compatible_groups)
        nm2_streams = {name: b"" for name in QPOST_STREAM_NAMES}
        nm2_streams["randmulti"] = nm2
        nm2_atoms = int(sum(np.count_nonzero(group.rows) for group in compatible_groups))
        qps1_nm2 = _build_qpost_candidate(
            pr81=pr81,
            streams=nm2_streams,
            candidate_id="pr81_qma9_pr82_qps1_nm2_generic_randmulti",
            candidate_kind="pr81_qma9_payload_plus_pr82_nm2_generic_randmulti_sidecar",
            qpost_contract="QPS1/NM2_generic_frame0_randmulti_subset",
            changed_atoms=nm2_atoms,
            runtime_blockers=common_pr81_runtime_blocker,
            output_dir=output_dir,
            extra={
                "randmulti": {
                    "groups": [randmulti_group_summary(group) for group in compatible_groups],
                    "semantic_scope": "generic frame-0 nearest random-pattern groups only",
                }
            },
        )

    qrm1_streams = {name: b"" for name in QPOST_STREAM_NAMES}
    qrm1_streams["randmulti"] = qrm1
    qrm1_atoms = int(sum(np.count_nonzero(group.rows) for group in groups))
    qps1_qrm1 = _build_qpost_candidate(
        pr81=pr81,
        streams=qrm1_streams,
        candidate_id="pr81_qma9_pr82_qps1_qrm1_all072_randmulti",
        candidate_kind="pr81_qma9_payload_plus_pr82_qrm1_all72_randmulti_sidecar",
        qpost_contract="QPS1/QRM1_native_sparse_group_id_randmulti",
        changed_atoms=qrm1_atoms,
        runtime_blockers=common_pr81_runtime_blocker,
        output_dir=output_dir,
        extra={
            "randmulti": {
                "groups": [randmulti_group_summary(group) for group in groups],
                "local_decode_profile": qrm1_profile,
                "runtime_support_report": qrm1_support_report,
                "semantic_scope": "all 72 PR82 randmulti groups represented exactly at sparse row level",
            }
        },
    )

    qrm1_supported_streams = {name: b"" for name in QPOST_STREAM_NAMES}
    qrm1_supported_streams["randmulti"] = qrm1_supported_subset
    qrm1_supported_atoms = int(sum(np.count_nonzero(group.rows) for group in qrm1_supported_groups))
    qps1_qrm1_supported = _build_qpost_candidate(
        pr81=pr81,
        streams=qrm1_supported_streams,
        candidate_id="pr81_qma9_pr82_qps1_qrm1_supported_subset_randmulti",
        candidate_kind="pr81_qma9_payload_plus_pr82_qrm1_runtime_supported_randmulti_sidecar",
        qpost_contract="QPS1/QRM1_native_sparse_group_id_runtime_supported_subset",
        changed_atoms=qrm1_supported_atoms,
        runtime_blockers=common_pr81_runtime_blocker,
        output_dir=output_dir,
        extra={
            "randmulti": {
                "excluded_group_policy": qrm1_exclusion,
                "groups": [randmulti_group_summary(group) for group in qrm1_supported_groups],
                "local_decode_profile": qrm1_supported_profile,
                "runtime_support_report_after_exclusion": qrm1_supported_subset_report,
                "runtime_support_report_before_exclusion": qrm1_support_report,
                "semantic_scope": (
                    "PR82 QRM1 groups accepted by robust_current raw-frame postprocess runtime; "
                    "active unsupported mask-dependent groups are excluded"
                ),
            }
        },
    )

    full_supported_streams = dict(controls_streams)
    full_supported_streams["randmulti"] = qrm1_supported_subset
    qps1_full_supported = _build_qpost_candidate(
        pr81=pr81,
        streams=full_supported_streams,
        candidate_id="pr81_qma9_pr82_qps1_controls_qrm1_supported_subset",
        candidate_kind="pr81_qma9_payload_plus_pr82_controls_and_qrm1_runtime_supported_subset_sidecar",
        qpost_contract="QPS1/full_controls_plus_QRM1_runtime_supported_subset",
        changed_atoms=control_atoms + qrm1_supported_atoms,
        runtime_blockers=common_pr81_runtime_blocker,
        output_dir=output_dir,
        extra={
            "randmulti": {
                "excluded_group_policy": qrm1_exclusion,
                "groups": [randmulti_group_summary(group) for group in qrm1_supported_groups],
                "local_decode_profile": qrm1_supported_profile,
                "runtime_support_report_after_exclusion": qrm1_supported_subset_report,
                "runtime_support_report_before_exclusion": qrm1_support_report,
                "semantic_scope": (
                    "all PR82 controls plus PR82 QRM1 groups accepted by robust_current raw-frame "
                    "postprocess runtime; active unsupported mask-dependent groups are excluded"
                ),
            }
        },
    )

    full_streams = dict(controls_streams)
    full_streams["randmulti"] = qrm1
    qps1_full = _build_qpost_candidate(
        pr81=pr81,
        streams=full_streams,
        candidate_id="pr81_qma9_pr82_qps1_controls_qrm1_all072",
        candidate_kind="pr81_qma9_payload_plus_pr82_controls_and_qrm1_all72_sidecar",
        qpost_contract="QPS1/full_controls_plus_QRM1_all72",
        changed_atoms=control_atoms + qrm1_atoms,
        runtime_blockers=common_pr81_runtime_blocker,
        output_dir=output_dir,
        extra={
            "randmulti": {
                "local_decode_profile": qrm1_profile,
                "runtime_support_report": qrm1_support_report,
                "semantic_scope": "all PR82 controls plus all 72 randmulti groups represented as charged sidecar bytes",
            }
        },
    )

    lower_bounds = static_lower_bounds(pr81=pr81, pr82=pr82)
    candidate_rows = [qps1_controls, qps1_qrm1, qps1_qrm1_supported, qps1_full_supported, qps1_full]
    if qps1_nm2 is not None:
        candidate_rows.insert(1, qps1_nm2)
    smallest_runtime_compatible_stack_candidate = min(
        [row for row in candidate_rows if "QRM1" not in str(row["candidate_kind"])],
        key=lambda row: int(row["archive_bytes"]),
    )
    summary = {
        "candidate_count": len(candidate_rows),
        "candidates": candidate_rows,
        "evidence_grade": "empirical_local_archive_build_and_static_lower_bound",
        "exact_scores_used_for_planning": {
            "pr81": {"archive_bytes": PR81_EXACT_T4_BYTES, "score": PR81_EXACT_T4_SCORE},
            "pr82": {
                "archive_bytes": PR82_EXACT_T4_BYTES,
                "posenet_dist": PR82_POSENET_DIST,
                "score": PR82_EXACT_T4_SCORE,
                "segnet_dist": PR82_SEGNET_DIST,
            },
        },
        "highest_ev_local_candidate": qps1_full_supported,
        "highest_ev_qrm1_compatible_candidate": qps1_full_supported,
        "no_remote_dispatch": True,
        "pr81_profile": {
            "archive_bytes": int(pr81["archive"].archive_bytes),
            "archive_sha256": pr81["archive"].archive_sha256,
            "payload_bytes": int(pr81["archive"].member_bytes),
            "payload_sha256": pr81["archive"].member_sha256,
            "qma9": pr81["qma9"],
            "segments": [
                {
                    "bytes": segment.size_bytes,
                    "codec": segment.codec,
                    "name": segment.name,
                    "offset": segment.offset,
                    "sha256": segment.sha256,
                }
                for segment in pr81["split"].segments
            ],
        },
        "pr82_profile": {
            "archive_bytes": pr82["archive_bytes"],
            "archive_sha256": pr82["archive_sha256"],
            "contract": {
                "fixed_bias_bytes": pr82["contract"].fixed_bias_bytes,
                "fixed_region_bytes": pr82["contract"].fixed_region_bytes,
                "randmulti_group_count": len(pr82["contract"].randmulti_specs),
                "replay_inflate": _repo_rel(replay_inflate),
                "replay_inflate_sha256": pr82["contract"].source_sha256,
            },
            "payload_bytes": pr82["payload_bytes"],
            "payload_sha256": pr82["payload_sha256"],
            "segments": {
                name: {
                    "bytes": len(data),
                    "sha256": sha256_bytes(data),
                }
                for name, data in bundle.encoded_segments.items()
            },
        },
        "schema": SCHEMA,
        "score_claim": False,
        "smallest_runtime_compatible_stack_candidate": smallest_runtime_compatible_stack_candidate,
        "static_lower_bounds": lower_bounds,
        "tool": TOOL,
        "unsupported_semantics_before_t4_dispatch": [
            "all-72 PR82 QRM1 is runtime-dispatchable only when parse/apply and raw-output delta proof pass",
            "the supported-subset QRM1 candidate remains available as a narrower control",
            "a lane dispatch claim is required before any future exact T4 run",
        ],
    }
    _write_json(output_dir / "candidate_summary.json", summary)
    (output_dir / "DESIGN_NOTE.md").write_text(
        "# PR81 QMA9 + PR82 Henosis Stack\n\n"
        "No remote GPU dispatch was performed.\n\n"
        "The highest-EV local byte candidate is the PR81 QMA9 payload with a "
        "charged PR82 QRM1 supported-subset sidecar.  The archive is "
        "deterministic and byte-closed, and active unsupported mask-dependent "
        "QRM1 groups are excluded by the robust-current runtime classifier.  "
        "No exact CUDA eval or remote dispatch was performed.\n",
        encoding="utf-8",
    )
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pr81-archive", type=Path, default=DEFAULT_PR81_ARCHIVE)
    parser.add_argument("--pr81-profile-json", type=Path, default=DEFAULT_PR81_PROFILE)
    parser.add_argument("--pr82-archive", type=Path, default=DEFAULT_PR82_ARCHIVE)
    parser.add_argument("--replay-inflate", type=Path, default=DEFAULT_REPLAY_INFLATE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--no-pr81-sha-check", action="store_true")
    parser.add_argument("--no-pr82-sha-check", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    summary = build_candidates(
        pr81_archive=args.pr81_archive,
        pr81_profile_json=args.pr81_profile_json,
        pr82_archive=args.pr82_archive,
        replay_inflate=args.replay_inflate,
        output_dir=args.output_dir,
        expected_pr81_sha256=None if args.no_pr81_sha_check else EXPECTED_PR81_SHA256,
        expected_pr82_sha256=None if args.no_pr82_sha_check else EXPECTED_PR82_SHA256,
    )
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
