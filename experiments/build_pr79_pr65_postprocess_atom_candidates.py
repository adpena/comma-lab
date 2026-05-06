#!/usr/bin/env python3
"""Build local PR79/S2-compatible PR65 postprocess atom candidates.

This is a local-only builder.  It mines typed PR65/Henosis qpost atoms, writes
archive-closed candidates by charging ``qpost.bin`` inside ``archive.zip``, and
proves that selected atoms change raw RGB output through the same runtime helper
used by ``inflate.sh``.  It records exact-eval command templates only; it never
dispatches remote jobs.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

CORE_BUILDER_PATH = REPO_ROOT / "experiments/build_pr65_qpost_atom_candidates.py"
RUNTIME_APPLY_PATH = REPO_ROOT / "submissions/robust_current/apply_qzs3_postprocess.py"
UNPACKER_PATH = REPO_ROOT / "submissions/robust_current/unpack_renderer_payload.py"
PRODUCER = "experiments/build_pr79_pr65_postprocess_atom_candidates.py"
SCHEMA_VERSION = 1
ORIGINAL_VIDEO_BYTES = 37_545_489
DEFAULT_OUTPUT_DIR = REPO_ROOT / "experiments/results/pr79_pr65_postprocess_atoms_20260503_worker"
DEFAULT_PR65_ARCHIVE = (
    REPO_ROOT
    / "experiments/results/top_submission_delta_reverse_engineering_20260503/"
    "sources/pr65_henosis_archive.zip"
)
DEFAULT_PR79_S2_ARCHIVE = (
    REPO_ROOT
    / "experiments/results/pr79_action_dictionary_repack_v2_20260503_codex/"
    "pr79_s2_fixed_adaptive_actions/archive.zip"
)
DEFAULT_PR79_SELECTED_RAW = (
    REPO_ROOT
    / "experiments/results/top_submission_reverse_engineering_20260503_pr79/"
    "raw_output_parity_pairs_cpu/selected_raw/robust_current_selected.raw"
)
DEFAULT_PR79_PARITY_JSON = DEFAULT_PR79_SELECTED_RAW.parents[1] / "pr75_raw_output_parity.json"
DEFAULT_FRONTIER_EVAL_JSON = (
    REPO_ROOT
    / "experiments/results/lightning_batch/"
    "exact_eval_pr75_minp_public_replay_t4_20260503T1049Z/contest_auth_eval.json"
)
DEFAULT_EXACT_NEGATIVE_GLOBS = (
    "experiments/results/lightning_batch/exact_eval_pr65_qpost_*",
    "experiments/results/lightning_batch/exact_eval_pr75_qpost_*",
)
EXPECTED_PR65_SHA256 = "b331cb4f6df9d8929db966b943b8c73624cdf3b6db71acbde361570852e59e68"
EXPECTED_PR79_S2_SHA256 = "5740aca7e255b00093154eb1823b5b6207d8795f8eb287d35758c4cda438ec68"
CURRENT_FRONTIER_SCORE = 0.31516575028285976  # [external: PR-65 contest-CUDA T4 frontier (== ANCHOR_SCORE)]
SUB314_TARGET = 0.314
_CANDIDATE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_]{2,140}$")


class PostprocessAtomBuildError(ValueError):
    """Raised when postprocess atom candidate building fails a guard."""


@dataclass(frozen=True)
class CandidateSpec:
    candidate_id: str
    include_streams: tuple[str, ...]
    pair_indices: tuple[int, ...]
    risk_family: str


DEFAULT_SPECS: tuple[CandidateSpec, ...] = (
    CandidateSpec(
        "pr79_s2_pr65_qpost_bias_pair598",
        ("bias",),
        (598,),
        "single_pair_bias_raw_delta_probe",
    ),
    CandidateSpec(
        "pr79_s2_pr65_qpost_bias_region_pair598",
        ("bias", "region"),
        (598,),
        "single_pair_bias_region_diagnostic",
    ),
    CandidateSpec(
        "pr79_s2_pr65_qpost_post_pair104",
        ("post",),
        (104,),
        "single_pair_post_diagnostic_exact_negative_family",
    ),
)


def _load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise PostprocessAtomBuildError(f"cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_core() -> Any:
    return _load_module(CORE_BUILDER_PATH, "pact_pr79_pr65_qpost_core")


def _load_runtime_apply() -> Any:
    return _load_module(RUNTIME_APPLY_PATH, "pact_pr79_pr65_qpost_runtime_apply")


def _load_unpacker() -> Any:
    return _load_module(UNPACKER_PATH, "pact_pr79_pr65_payload_unpacker")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_json_bytes(payload))


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise PostprocessAtomBuildError(f"invalid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise PostprocessAtomBuildError(f"expected JSON object: {path}")
    return payload


def _repo_rel(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _single_level_members(archive: Path) -> dict[str, bytes]:
    members: dict[str, bytes] = {}
    with zipfile.ZipFile(archive, "r") as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            name = info.filename
            path = Path(name)
            if not name or name.startswith("/") or ".." in path.parts or len(path.parts) != 1:
                raise PostprocessAtomBuildError(f"unsafe archive member path: {name!r}")
            if name in members:
                raise PostprocessAtomBuildError(f"duplicate archive member: {name!r}")
            members[name] = zf.read(info)
    return members


def _validate_spec(core: Any, spec: CandidateSpec) -> None:
    if not _CANDIDATE_ID_RE.match(spec.candidate_id):
        raise PostprocessAtomBuildError(f"invalid candidate id: {spec.candidate_id!r}")
    if tuple(sorted(set(spec.pair_indices))) != spec.pair_indices:
        raise PostprocessAtomBuildError(f"{spec.candidate_id}: pair indices must be unique and sorted")
    if not spec.pair_indices:
        raise PostprocessAtomBuildError(f"{spec.candidate_id}: no selected pairs")
    if any(pair < 0 or pair >= 600 for pair in spec.pair_indices):
        raise PostprocessAtomBuildError(f"{spec.candidate_id}: pair index out of range")
    supported = set(core.SUPPORTED_PAIR_FILTER_STREAMS)
    unknown = sorted(set(spec.include_streams) - supported)
    if unknown:
        raise PostprocessAtomBuildError(f"{spec.candidate_id}: unsupported qpost streams: {unknown}")


def _verify_custody(
    *,
    source_archive: Path,
    pr65_archive: Path,
    expected_source_sha256: str | None,
    expected_pr65_sha256: str | None,
) -> dict[str, Any]:
    if not source_archive.is_file():
        raise PostprocessAtomBuildError(f"missing source archive: {source_archive}")
    if not pr65_archive.is_file():
        raise PostprocessAtomBuildError(f"missing PR65 archive: {pr65_archive}")
    source_sha = _sha256_path(source_archive)
    pr65_sha = _sha256_path(pr65_archive)
    if expected_source_sha256 and source_sha != expected_source_sha256:
        raise PostprocessAtomBuildError(
            f"source archive SHA mismatch: expected {expected_source_sha256}, got {source_sha}"
        )
    if expected_pr65_sha256 and pr65_sha != expected_pr65_sha256:
        raise PostprocessAtomBuildError(
            f"PR65 archive SHA mismatch: expected {expected_pr65_sha256}, got {pr65_sha}"
        )
    return {
        "source_archive": _repo_rel(source_archive),
        "source_archive_bytes": source_archive.stat().st_size,
        "source_archive_sha256": source_sha,
        "pr65_archive": _repo_rel(pr65_archive),
        "pr65_archive_bytes": pr65_archive.stat().st_size,
        "pr65_archive_sha256": pr65_sha,
    }


def _probe_source_runtime(source_archive: Path, *, require_probe: bool) -> dict[str, Any]:
    members = _single_level_members(source_archive)
    if "p" not in members:
        raise PostprocessAtomBuildError(f"{source_archive} is missing single payload member 'p'")
    probe: dict[str, Any] = {
        "member_names": sorted(members),
        "payload_member": "p",
        "payload_bytes": len(members["p"]),
        "payload_sha256": _sha256_bytes(members["p"]),
        "runtime_parse_supported": False,
    }
    try:
        unpacker = _load_unpacker()
        header, parsed_members = unpacker._parse_payload(members["p"])  # noqa: SLF001
    except Exception as exc:
        if require_probe:
            raise PostprocessAtomBuildError(f"source payload is not runtime-parseable: {exc}") from exc
        probe["runtime_probe_error"] = str(exc)
        return probe
    required = {"masks.mkv", "renderer.bin"}
    if not required.issubset(parsed_members):
        raise PostprocessAtomBuildError(f"source payload parse missing required members: {required - set(parsed_members)}")
    action_member = parsed_members.get("seg_tile_actions.bin")
    pose_member = parsed_members.get("optimized_poses.qp1") or parsed_members.get("optimized_poses.bin")
    probe.update(
        {
            "runtime_parse_supported": True,
            "payload_format": header.get("payload_format") or header.get("schema"),
            "runtime_members": {
                name: {
                    "bytes": len(data),
                    "sha256": _sha256_bytes(data),
                }
                for name, data in sorted(parsed_members.items())
            },
            "seg_tile_actions_present": action_member is not None,
            "seg_tile_actions_decoded_sha256": _sha256_bytes(action_member) if action_member is not None else None,
            "pose_present": pose_member is not None,
        }
    )
    return probe


def _summarize_stream_activity(core: Any, arrays: Mapping[str, np.ndarray], spec: CandidateSpec) -> dict[str, Any]:
    by_stream: dict[str, Any] = {}
    total = 0
    active_pairs: set[int] = set()
    selected = list(spec.pair_indices)
    for name in spec.include_streams:
        counts = core._nondefault_mask(name, arrays[name])  # noqa: SLF001
        selected_counts = counts[selected]
        active = [int(pair) for pair, count in zip(selected, selected_counts) if int(count) > 0]
        stream_total = int(selected_counts.sum())
        total += stream_total
        active_pairs.update(active)
        by_stream[name] = {
            "selected_active_atoms": stream_total,
            "selected_active_pairs": active,
            "selected_pair_atom_counts": {str(pair): int(count) for pair, count in zip(selected, selected_counts)},
        }
    return {
        "is_noop": total == 0,
        "selected_pairs": selected,
        "selected_active_atoms_total": total,
        "selected_active_pair_count": len(active_pairs),
        "selected_active_pairs_any_stream": sorted(active_pairs),
        "by_stream": by_stream,
        "nonselected_pairs_default_to_identity": True,
    }


def _parse_parity_pair_indices(path: Path | None) -> tuple[int, ...]:
    if path is None or not path.exists():
        return ()
    payload = _read_json(path)
    raw = payload.get("pair_indices")
    if not isinstance(raw, list):
        return ()
    return tuple(int(value) for value in raw)


def _read_qpost_from_archive(archive_path: Path, candidate_dir: Path) -> Path:
    members = _single_level_members(archive_path)
    qpost = members.get("qpost.bin")
    if qpost is None:
        raise PostprocessAtomBuildError(f"{archive_path} missing qpost.bin after build")
    qpost_path = candidate_dir / "qpost_for_raw_delta_proof.bin"
    qpost_path.write_bytes(qpost)
    return qpost_path


def _raw_pair_from_selected_raw(
    raw_path: Path,
    *,
    pair_indices: Sequence[int],
    pair_index: int,
    height: int,
    width: int,
) -> np.ndarray:
    if pair_index not in pair_indices:
        raise PostprocessAtomBuildError(
            f"pair {pair_index} is not present in selected raw proof set {list(pair_indices)}"
        )
    frame_bytes = height * width * 3
    pair_bytes = 2 * frame_bytes
    expected = pair_bytes * len(pair_indices)
    actual = raw_path.stat().st_size
    if actual != expected:
        raise PostprocessAtomBuildError(
            f"selected raw size mismatch: expected {expected} bytes for {len(pair_indices)} pairs, got {actual}"
        )
    offset = pair_indices.index(pair_index) * pair_bytes
    with raw_path.open("rb") as handle:
        handle.seek(offset)
        data = handle.read(pair_bytes)
    if len(data) != pair_bytes:
        raise PostprocessAtomBuildError(f"short selected raw read for pair {pair_index}")
    return np.frombuffer(data, dtype=np.uint8).copy().reshape(1, 2, height, width, 3)


def _raw_delta_proof(
    *,
    archive_path: Path,
    candidate_dir: Path,
    selected_raw_path: Path | None,
    selected_raw_pair_indices: Sequence[int],
    proof_pair_index: int,
    height: int,
    width: int,
    device_name: str,
) -> dict[str, Any]:
    if selected_raw_path is None or not selected_raw_path.exists() or not selected_raw_pair_indices:
        raise PostprocessAtomBuildError("raw-output proof requires selected raw bytes and pair indices")
    qpost_path = _read_qpost_from_archive(archive_path, candidate_dir)
    runtime = _load_runtime_apply()
    device = torch.device(device_name)
    before = _raw_pair_from_selected_raw(
        selected_raw_path,
        pair_indices=selected_raw_pair_indices,
        pair_index=proof_pair_index,
        height=height,
        width=width,
    )
    state = runtime.read_qpost(qpost_path, device)
    batch = torch.from_numpy(before).to(device=device, dtype=torch.float32)
    after_tensor = runtime.apply_qpost_batch(
        batch,
        pair_start=proof_pair_index,
        state=state,
        grid_cache={},
        randpat_cache={},
    )
    after = after_tensor.clamp(0, 255).round().to(torch.uint8).cpu().numpy()
    diff = after.astype(np.int16) - before.astype(np.int16)
    changed = int(np.count_nonzero(diff))
    if changed <= 0:
        raise PostprocessAtomBuildError(f"qpost atom candidate is raw-output no-op for pair {proof_pair_index}")
    proof = {
        "proof_type": "runtime_apply_qzs3_postprocess_selected_raw_pair",
        "runtime_helper": _repo_rel(RUNTIME_APPLY_PATH),
        "runtime_helper_sha256": _sha256_path(RUNTIME_APPLY_PATH),
        "qpost_path": _repo_rel(qpost_path),
        "qpost_sha256": _sha256_path(qpost_path),
        "selected_raw_path": _repo_rel(selected_raw_path),
        "selected_raw_sha256": _sha256_path(selected_raw_path),
        "selected_raw_pair_indices": list(selected_raw_pair_indices),
        "proof_pair_index": proof_pair_index,
        "height": height,
        "width": width,
        "device": str(device),
        "before_bytes": int(before.nbytes),
        "before_sha256": _sha256_bytes(before.tobytes()),
        "after_bytes": int(after.nbytes),
        "after_sha256": _sha256_bytes(after.tobytes()),
        "exact_equal": False,
        "changed_values": changed,
        "compared_values": int(diff.size),
        "max_abs_delta": int(np.abs(diff).max()),
        "mean_abs_delta": float(np.abs(diff).mean()),
    }
    _write_json(candidate_dir / "raw_output_delta_proof.json", proof)
    return proof


def _load_frontier(frontier_eval_json: Path | None) -> dict[str, Any]:
    if frontier_eval_json is None or not frontier_eval_json.exists():
        return {
            "score": CURRENT_FRONTIER_SCORE,
            "target_score": SUB314_TARGET,
            "source": "hardcoded_current_frontier_from_research_note",
        }
    payload = _read_json(frontier_eval_json)
    return {
        "score": float(payload.get("canonical_score", CURRENT_FRONTIER_SCORE)),
        "target_score": SUB314_TARGET,
        "eval_json": _repo_rel(frontier_eval_json),
        "archive_sha256": (payload.get("provenance") or {}).get("archive_sha256"),
        "n_samples": payload.get("n_samples"),
        "avg_posenet_dist": payload.get("avg_posenet_dist"),
        "avg_segnet_dist": payload.get("avg_segnet_dist"),
        "score_pose_contribution": payload.get("score_pose_contribution"),
        "score_seg_contribution": payload.get("score_seg_contribution"),
        "score_rate_contribution": payload.get("score_rate_contribution"),
    }


def _summarize_exact_negatives(globs: Sequence[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for pattern in globs:
        for directory in sorted(REPO_ROOT.glob(pattern)):
            eval_json = directory / "contest_auth_eval.json"
            if not eval_json.exists():
                continue
            payload = _read_json(eval_json)
            rows.append(
                {
                    "run_dir": _repo_rel(directory),
                    "canonical_score": payload.get("canonical_score"),
                    "avg_posenet_dist": payload.get("avg_posenet_dist"),
                    "avg_segnet_dist": payload.get("avg_segnet_dist"),
                    "n_samples": payload.get("n_samples"),
                    "score_pose_contribution": payload.get("score_pose_contribution"),
                    "score_seg_contribution": payload.get("score_seg_contribution"),
                    "score_rate_contribution": payload.get("score_rate_contribution"),
                    "evidence_grade": "A++_exact_t4_negative_for_qpost_family",
                }
            )
    return rows


def _dispatch_screen(
    *,
    candidate: Mapping[str, Any],
    spec: CandidateSpec,
    frontier: Mapping[str, Any],
    exact_negatives: Sequence[Mapping[str, Any]],
    source_runtime_probe: Mapping[str, Any],
) -> dict[str, Any]:
    blockers: list[str] = []
    if candidate["no_op_proof"]["is_noop"]:
        blockers.append("selected qpost atoms are identity after pair filtering")
    if not candidate["raw_output_delta_proof"] or candidate["raw_output_delta_proof"].get("exact_equal"):
        blockers.append("missing positive raw-output delta proof")
    if not source_runtime_probe.get("runtime_parse_supported"):
        blockers.append("source payload is not supported by robust runtime unpacker")
    if "post" in spec.include_streams:
        blockers.append("post stream belongs to already-negative qpost exact family")
    if "region" in spec.include_streams:
        blockers.append("region stream belongs to already-negative qpost exact family")
    if any(name in spec.include_streams for name in ("shift", "frac", "frac2", "frac3")):
        blockers.append("motion/fractional streams belong to already-negative qpost exact family")
    if exact_negatives:
        blockers.append("existing exact T4 qpost screens are all worse than current exact frontier")

    rate_delta = float(candidate["formula_rate_score_delta_vs_source"])
    frontier_score = float(frontier.get("score", CURRENT_FRONTIER_SCORE))
    break_even = (frontier_score - SUB314_TARGET) + max(rate_delta, 0.0)
    if break_even > 0.001:
        blockers.append(
            f"no component evidence for required sub-0.314 gain {break_even:.9f}"
        )
    return {
        "remote_dispatched": False,
        "score_claim": False,
        "dispatch_ready_now": False,
        "recommendation": "do_not_dispatch",
        "lane_claim_required_before_any_future_exact_eval": True,
        "reason": "; ".join(blockers) if blockers else "local candidate only; exact CUDA evidence still required",
        "blockers": blockers,
        "break_even_component_gain_for_sub314_vs_frontier": break_even,
        "rate_score_delta_vs_source": rate_delta,
        "qpost_exact_negative_count": len(exact_negatives),
    }


def _exact_eval_command(archive: Path, output_dir: Path, candidate_id: str) -> str:
    work_dir = output_dir / "exact_eval_work" / candidate_id
    return (
        ".venv/bin/python -u experiments/contest_auth_eval.py "
        f"--archive {archive} "
        "--inflate-sh submissions/robust_current/inflate.sh "
        "--upstream-dir upstream --device cuda --keep-work-dir "
        f"--work-dir {work_dir}"
    )


def build_candidates(
    *,
    source_archive: Path = DEFAULT_PR79_S2_ARCHIVE,
    pr65_archive: Path = DEFAULT_PR65_ARCHIVE,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    specs: Sequence[CandidateSpec] = DEFAULT_SPECS,
    selected_raw_path: Path | None = DEFAULT_PR79_SELECTED_RAW,
    parity_json: Path | None = DEFAULT_PR79_PARITY_JSON,
    raw_height: int = 874,
    raw_width: int = 1164,
    proof_device: str = "cpu",
    frontier_eval_json: Path | None = DEFAULT_FRONTIER_EVAL_JSON,
    exact_negative_globs: Sequence[str] = DEFAULT_EXACT_NEGATIVE_GLOBS,
    expected_source_sha256: str | None = EXPECTED_PR79_S2_SHA256,
    expected_pr65_sha256: str | None = EXPECTED_PR65_SHA256,
    require_runtime_probe: bool = True,
) -> dict[str, Any]:
    core = _load_core()
    source_archive = source_archive.resolve()
    pr65_archive = pr65_archive.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    for spec in specs:
        _validate_spec(core, spec)

    custody = _verify_custody(
        source_archive=source_archive,
        pr65_archive=pr65_archive,
        expected_source_sha256=expected_source_sha256,
        expected_pr65_sha256=expected_pr65_sha256,
    )
    source_runtime_probe = _probe_source_runtime(source_archive, require_probe=require_runtime_probe)
    qpost_builder = core._load_qpost_builder()  # noqa: SLF001
    streams = qpost_builder.extract_pr65_qpost_streams(pr65_archive)
    arrays = core._decode_stream_arrays(qpost_builder, streams)  # noqa: SLF001
    selected_raw_pair_indices = _parse_parity_pair_indices(parity_json)
    frontier = _load_frontier(frontier_eval_json)
    exact_negatives = _summarize_exact_negatives(exact_negative_globs)

    candidates: list[dict[str, Any]] = []
    for spec in specs:
        no_op = _summarize_stream_activity(core, arrays, spec)
        if no_op["is_noop"]:
            raise PostprocessAtomBuildError(f"{spec.candidate_id}: selected atom set is no-op")
        proof_pairs = [
            pair
            for pair in spec.pair_indices
            if pair in selected_raw_pair_indices
            and pair in no_op["selected_active_pairs_any_stream"]
        ]
        if not proof_pairs:
            raise PostprocessAtomBuildError(
                f"{spec.candidate_id}: no selected active pair is available in raw proof set"
            )
        proof_pair = int(proof_pairs[0])
        candidate_dir = output_dir / spec.candidate_id
        archive_path = candidate_dir / "archive.zip"
        meta = qpost_builder.build_candidate(
            source_archive,
            pr65_archive,
            archive_path,
            include_streams=spec.include_streams,
            pair_indices=spec.pair_indices,
        )
        raw_proof = _raw_delta_proof(
            archive_path=archive_path,
            candidate_dir=candidate_dir,
            selected_raw_path=selected_raw_path,
            selected_raw_pair_indices=selected_raw_pair_indices,
            proof_pair_index=proof_pair,
            height=raw_height,
            width=raw_width,
            device_name=proof_device,
        )
        candidate = {
            "candidate_id": spec.candidate_id,
            "schema_version": SCHEMA_VERSION,
            "tool": PRODUCER,
            "score_claim": False,
            "evidence_grade": "empirical_archive_closed_raw_delta_proof_only",
            "archive": _repo_rel(archive_path),
            "archive_bytes": meta["output_archive_bytes"],
            "archive_sha256": meta["output_archive_sha256"],
            "source_archive_bytes": meta["source_archive_bytes"],
            "source_archive_sha256": meta["source_archive_sha256"],
            "archive_byte_delta_vs_source": meta["archive_byte_delta"],
            "formula_rate_score_delta_vs_source": meta["formula_rate_score_delta"],
            "qpost_bytes": meta["members"]["qpost.bin"]["bytes"],
            "qpost_sha256": meta["members"]["qpost.bin"]["sha256"],
            "include_streams": list(spec.include_streams),
            "selected_pairs": list(spec.pair_indices),
            "risk_family": spec.risk_family,
            "qpost_streams": meta["qpost_streams"],
            "no_op_proof": no_op,
            "raw_output_delta_proof": raw_proof,
            "exact_eval_command_template": _exact_eval_command(archive_path, output_dir, spec.candidate_id),
        }
        candidate["dispatch_recommendation"] = _dispatch_screen(
            candidate=candidate,
            spec=spec,
            frontier=frontier,
            exact_negatives=exact_negatives,
            source_runtime_probe=source_runtime_probe,
        )
        _write_json(candidate_dir / "manifest.json", candidate)
        candidates.append(candidate)

    summary = {
        "schema_version": SCHEMA_VERSION,
        "tool": PRODUCER,
        "score_claim": False,
        "evidence_grade": "empirical_archive_closed_raw_delta_proof_only",
        "custody": custody,
        "source_runtime_probe": source_runtime_probe,
        "frontier_reference": frontier,
        "exact_qpost_negative_evidence": exact_negatives,
        "candidate_count": len(candidates),
        "candidates": [
            {
                "candidate_id": row["candidate_id"],
                "archive": row["archive"],
                "archive_bytes": row["archive_bytes"],
                "archive_sha256": row["archive_sha256"],
                "archive_byte_delta_vs_source": row["archive_byte_delta_vs_source"],
                "qpost_bytes": row["qpost_bytes"],
                "selected_pairs": row["selected_pairs"],
                "include_streams": row["include_streams"],
                "raw_changed_values": row["raw_output_delta_proof"]["changed_values"],
                "dispatch_recommendation": row["dispatch_recommendation"],
            }
            for row in candidates
        ],
        "dispatch_summary": {
            "remote_dispatched": False,
            "recommendation": "do_not_dispatch",
            "reason": (
                "Candidates are archive-closed and raw-output-changing, but the "
                "known qpost exact T4 screens are negative and no local component "
                "evidence supports spending another exact-eval slot."
            ),
        },
    }
    _write_json(output_dir / "candidate_summary.json", summary)
    design_note = (
        "# PR79/S2 PR65 Postprocess Atom Local Builder\n\n"
        "No remote jobs were dispatched.\n\n"
        "This builder produced archive-closed qpost atom candidates and proved "
        "that selected atoms change raw RGB output through "
        "`submissions/robust_current/apply_qzs3_postprocess.py`. The dispatch "
        "recommendation is fail-closed: do not exact-eval these candidates now, "
        "because existing PR65/PR75 qpost exact T4 screens are negative versus "
        "the current exact frontier and these artifacts have no component-trace "
        "evidence that clears the sub-0.314 break-even requirement.\n"
    )
    (output_dir / "DESIGN_NOTE.md").write_text(design_note)
    return summary


def _parse_spec(raw: str) -> CandidateSpec:
    parts = [part.strip() for part in raw.split(":")]
    if len(parts) != 3 or not all(parts):
        raise PostprocessAtomBuildError(
            "custom spec must be candidate_id:stream1,stream2:pair1,pair2"
        )
    streams = tuple(part.strip() for part in parts[1].split(",") if part.strip())
    pairs = tuple(sorted(set(int(part.strip()) for part in parts[2].split(",") if part.strip())))
    return CandidateSpec(parts[0], streams, pairs, "custom")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-archive", type=Path, default=DEFAULT_PR79_S2_ARCHIVE)
    parser.add_argument("--pr65-archive", type=Path, default=DEFAULT_PR65_ARCHIVE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--selected-raw", type=Path, default=DEFAULT_PR79_SELECTED_RAW)
    parser.add_argument("--parity-json", type=Path, default=DEFAULT_PR79_PARITY_JSON)
    parser.add_argument("--raw-height", type=int, default=874)
    parser.add_argument("--raw-width", type=int, default=1164)
    parser.add_argument("--proof-device", default="cpu")
    parser.add_argument("--frontier-eval-json", type=Path, default=DEFAULT_FRONTIER_EVAL_JSON)
    parser.add_argument("--spec", action="append", default=None, help="candidate_id:stream1,stream2:pair1,pair2")
    parser.add_argument("--allow-source-sha-mismatch", action="store_true")
    parser.add_argument("--allow-pr65-sha-mismatch", action="store_true")
    parser.add_argument("--skip-runtime-probe", action="store_true")
    args = parser.parse_args(argv)
    specs = tuple(_parse_spec(raw) for raw in args.spec) if args.spec else DEFAULT_SPECS
    summary = build_candidates(
        source_archive=args.source_archive,
        pr65_archive=args.pr65_archive,
        output_dir=args.output_dir,
        specs=specs,
        selected_raw_path=args.selected_raw,
        parity_json=args.parity_json,
        raw_height=args.raw_height,
        raw_width=args.raw_width,
        proof_device=args.proof_device,
        frontier_eval_json=args.frontier_eval_json,
        expected_source_sha256=None if args.allow_source_sha_mismatch else EXPECTED_PR79_S2_SHA256,
        expected_pr65_sha256=None if args.allow_pr65_sha_mismatch else EXPECTED_PR65_SHA256,
        require_runtime_probe=not args.skip_runtime_probe,
    )
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
