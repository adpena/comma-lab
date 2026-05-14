#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build a local PR75/P6 lossless-resweep plus qpost bias micro-stack.

This builder is local-only. It preserves the decoded members of the current
C-089 PR75/QP1/P6 source archive, then adds a counted PR65 bias-only qpost
sidecar selected from positive public-trace pair opportunities. It emits
custody manifests and exact-eval command drafts, but it does not dispatch any
remote GPU work and it does not make a score claim.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import struct
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import brotli


REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (REPO_ROOT, REPO_ROOT / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

TOOL = "experiments/build_pr75_qpost_microstack_candidate.py"
SCHEMA_VERSION = 1
FIXED_ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
ORIGINAL_VIDEO_BYTES = 37_545_489
RATE_SCORE_PER_BYTE = 25.0 / ORIGINAL_VIDEO_BYTES
SOURCE_SCORE = 0.3154707273953505  # [external: PR-75 contest-CUDA T4 anchor (== PR-65 frontier)]
SUB314_TARGET = 0.314
EXPECTED_SOURCE_SHA256 = "0ec53e5b871149ed6eea56c0b9bcca3baec998d5bfad4f371979e0c90e62cea8"
EXPECTED_PR65_SHA256 = "b331cb4f6df9d8929db966b943b8c73624cdf3b6db71acbde361570852e59e68"
RUN_ID = "pr75_qpost_microstack_worker_20260503"

DEFAULT_SOURCE_ARCHIVE = (
    REPO_ROOT
    / "experiments/results/lightning_batch/"
    "exact_eval_c067_pr75_qp1_top40_p6_t4_awsfix1_20260503T0630Z/archive.zip"
)
DEFAULT_PR65_ARCHIVE = (
    REPO_ROOT
    / "experiments/results/top_submission_delta_reverse_engineering_20260503/"
    "sources/pr65_henosis_archive.zip"
)
DEFAULT_C089_TRACE = (
    REPO_ROOT
    / "experiments/results/lightning_batch/"
    "exact_eval_c067_pr75_qp1_top40_p6_t4_awsfix1_20260503T0630Z/component_trace.json"
)
DEFAULT_PR65_TRACE = (
    REPO_ROOT
    / "experiments/results/vast_harvest/public_external_component_trace_20260502T0642Z/"
    "pr65_torch25_compat_adapter/component_trace.json"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / f"experiments/results/{RUN_ID}"
INFLATE_SH = REPO_ROOT / "submissions/robust_current/inflate.sh"
UNPACKER_PATH = REPO_ROOT / "submissions/robust_current/unpack_renderer_payload.py"
QPOST_BUILDER_PATH = REPO_ROOT / "experiments/build_qzs3_postprocess_candidate.py"
QPOST_ATOM_PATH = REPO_ROOT / "experiments/build_pr65_qpost_atom_candidates.py"


class MicrostackBuildError(ValueError):
    """Raised when the local micro-stack cannot be built safely."""


@dataclass(frozen=True)
class P6Slices:
    mask_br: bytes
    model_br: bytes
    actions_br: bytes
    pose_br: bytes
    record_count: int


@dataclass(frozen=True)
class BrotliChoice:
    data: bytes
    params: tuple[int, int, int, int] | None

    @property
    def params_json(self) -> dict[str, int] | str:
        if self.params is None:
            return "source"
        quality, mode, lgwin, lgblock = self.params
        return {
            "quality": quality,
            "mode": mode,
            "lgwin": lgwin,
            "lgblock": lgblock,
        }


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_json_bytes(payload))


def _load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise MicrostackBuildError(f"cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_unpacker() -> Any:
    return _load_module(UNPACKER_PATH, "pr75_qpost_microstack_unpacker")


def _load_qpost_builder() -> Any:
    return _load_module(QPOST_BUILDER_PATH, "pr75_qpost_microstack_qpost_builder")


def _load_qpost_atoms() -> Any:
    return _load_module(QPOST_ATOM_PATH, "pr75_qpost_microstack_qpost_atoms")


def _zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, FIXED_ZIP_TIMESTAMP)
    info.compress_type = zipfile.ZIP_STORED
    info.external_attr = 0o644 << 16
    info.create_system = 3
    return info


def _write_single_member_archive(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(_zip_info("p"), payload)


def _read_single_p_member(path: Path) -> bytes:
    with zipfile.ZipFile(path, "r") as zf:
        infos = [info for info in zf.infolist() if not info.is_dir()]
        names = [info.filename for info in infos]
        if names != ["p"]:
            raise MicrostackBuildError(f"{path} must contain exactly member 'p'; got {names!r}")
        return zf.read(infos[0])


def _parse_p6_payload(payload: bytes) -> P6Slices:
    if not payload.startswith(b"P6"):
        raise MicrostackBuildError(f"source payload must be PR75 P6; got {payload[:2]!r}")
    header_size = 2 + struct.calcsize("<IHHH")
    if len(payload) <= header_size:
        raise MicrostackBuildError("P6 payload is too short")
    mask_len, model_len, actions_len, record_count = struct.unpack_from("<IHHH", payload, 2)
    if min(mask_len, model_len, actions_len, record_count) <= 0:
        raise MicrostackBuildError("P6 payload contains an empty required slice")
    cursor = header_size
    mask_end = cursor + mask_len
    model_end = mask_end + model_len
    actions_end = model_end + actions_len
    if actions_end >= len(payload):
        raise MicrostackBuildError("P6 slice lengths leave no pose stream")
    return P6Slices(
        mask_br=payload[cursor:mask_end],
        model_br=payload[mask_end:model_end],
        actions_br=payload[model_end:actions_end],
        pose_br=payload[actions_end:],
        record_count=record_count,
    )


def _build_p6_payload(slices: P6Slices) -> bytes:
    return (
        b"P6"
        + struct.pack(
            "<IHHH",
            len(slices.mask_br),
            len(slices.model_br),
            len(slices.actions_br),
            int(slices.record_count),
        )
        + slices.mask_br
        + slices.model_br
        + slices.actions_br
        + slices.pose_br
    )


def micro_brotli_param_grid() -> list[tuple[int, int, int, int]]:
    """Return a small deterministic grid for known C-089 P6 stream wins."""
    params = [
        (11, 0, 19, 17),  # mask stream win observed in PR75/C089 basin
        (9, 0, 10, 0),  # tiny delta-varint action stream win
        (11, 0, 16, 0),  # QP1 pose stream win
        (11, 0, 24, 0),
        (11, 0, 18, 17),
        (11, 0, 20, 17),
        (10, 0, 19, 17),
        (8, 0, 19, 17),
    ]
    out: list[tuple[int, int, int, int]] = []
    for param in params:
        if param not in out:
            out.append(param)
    return out


def exhaustive_brotli_param_grid() -> list[tuple[int, int, int, int]]:
    params: list[tuple[int, int, int, int]] = []
    for quality in range(11, -1, -1):
        for mode in (0, 1, 2):
            for lgwin in range(10, 25):
                for lgblock in (0, 16, 17, 18, 19, 20, 21, 22, 23, 24):
                    if lgblock and lgblock > lgwin:
                        continue
                    params.append((quality, mode, lgwin, lgblock))
    return params


def _best_brotli(
    raw: bytes,
    source_br: bytes,
    params: Iterable[tuple[int, int, int, int]],
) -> BrotliChoice:
    best = BrotliChoice(source_br, None)
    for param in params:
        quality, mode, lgwin, lgblock = param
        candidate = brotli.compress(
            raw,
            quality=quality,
            mode=mode,
            lgwin=lgwin,
            lgblock=lgblock,
        )
        if len(candidate) < len(best.data):
            best = BrotliChoice(candidate, param)
    if brotli.decompress(best.data) != raw:
        raise MicrostackBuildError("selected Brotli stream failed round-trip")
    return best


def _decoded_summary(decoded: Mapping[str, bytes]) -> dict[str, dict[str, Any]]:
    return {
        name: {"bytes": len(data), "sha256": _sha256_bytes(data)}
        for name, data in sorted(decoded.items())
    }


def _runtime_member_summary(header: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for item in header.get("members", []):
        out[str(item["name"])] = {
            "bytes": int(item["bytes"]),
            "sha256": str(item["sha256"]),
            "codec": str(item["codec"]),
            "decoded_bytes": int(item["decoded_bytes"]),
            "decoded_sha256": str(item["decoded_sha256"]),
        }
    return out


def _validate_decoded_parity(
    *,
    payload: bytes,
    expected_decoded: Mapping[str, bytes],
    unpacker: Any,
) -> tuple[dict[str, Any], dict[str, bytes]]:
    header, decoded = unpacker._parse_payload(payload)  # noqa: SLF001
    missing = sorted(set(expected_decoded) - set(decoded))
    extra = sorted(set(decoded) - set(expected_decoded))
    if missing or extra:
        raise MicrostackBuildError(f"decoded member mismatch: missing={missing} extra={extra}")
    for name, expected in expected_decoded.items():
        actual = decoded[name]
        if actual != expected:
            raise MicrostackBuildError(
                f"decoded member {name} changed: "
                f"expected={_sha256_bytes(expected)} actual={_sha256_bytes(actual)}"
            )
    return header, decoded


def _verify_runtime_hook() -> dict[str, Any]:
    text = INFLATE_SH.read_text()
    present = "qpost.bin detected" in text and "apply_qzs3_postprocess.py" in text
    return {
        "inflate_sh": str(INFLATE_SH),
        "inflate_sh_sha256": _sha256_file(INFLATE_SH),
        "qpost_runtime_hook_present": present,
    }


def _verify_source_sha(path: Path, expected: str | None, label: str) -> str:
    actual = _sha256_file(path)
    if expected and actual != expected:
        raise MicrostackBuildError(f"{label} SHA mismatch: expected {expected}, got {actual}")
    return actual


def _build_reswept_p6_base(
    *,
    source_archive: Path,
    output_archive: Path,
    brotli_params: Sequence[tuple[int, int, int, int]],
    unpacker: Any,
) -> tuple[dict[str, Any], dict[str, bytes]]:
    source_payload = _read_single_p_member(source_archive)
    source_slices = _parse_p6_payload(source_payload)
    try:
        source_header, source_decoded = unpacker._parse_payload(source_payload)  # noqa: SLF001
    except Exception as exc:
        raise MicrostackBuildError(f"source P6 payload failed runtime parse: {exc}") from exc
    if source_header.get("payload_format") != "public_pr75_qzs3_qp1_segactions_p6_delta_varint":
        raise MicrostackBuildError(
            "source must parse as public PR75 P6 delta-varint payload; "
            f"got {source_header.get('payload_format')!r}"
        )
    raw_streams = {
        "masks.mkv": brotli.decompress(source_slices.mask_br),
        "renderer.bin": brotli.decompress(source_slices.model_br),
        "seg_tile_actions.delta_varint": brotli.decompress(source_slices.actions_br),
        "optimized_poses.qp1": brotli.decompress(source_slices.pose_br),
    }
    choices = {
        "masks.mkv": _best_brotli(raw_streams["masks.mkv"], source_slices.mask_br, brotli_params),
        "renderer.bin": _best_brotli(raw_streams["renderer.bin"], source_slices.model_br, brotli_params),
        "seg_tile_actions.delta_varint": _best_brotli(
            raw_streams["seg_tile_actions.delta_varint"],
            source_slices.actions_br,
            brotli_params,
        ),
        "optimized_poses.qp1": _best_brotli(
            raw_streams["optimized_poses.qp1"],
            source_slices.pose_br,
            brotli_params,
        ),
    }
    candidate_payload = _build_p6_payload(
        P6Slices(
            mask_br=choices["masks.mkv"].data,
            model_br=choices["renderer.bin"].data,
            actions_br=choices["seg_tile_actions.delta_varint"].data,
            pose_br=choices["optimized_poses.qp1"].data,
            record_count=source_slices.record_count,
        )
    )
    candidate_header, _candidate_decoded = _validate_decoded_parity(
        payload=candidate_payload,
        expected_decoded=source_decoded,
        unpacker=unpacker,
    )
    _write_single_member_archive(output_archive, candidate_payload)

    source_bytes = source_archive.stat().st_size
    output_bytes = output_archive.stat().st_size
    source_payload_sha = _sha256_bytes(source_payload)
    candidate_payload_sha = _sha256_bytes(candidate_payload)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "tool": TOOL,
        "candidate_id": "c089_p6_lossless_resweep",
        "score_claim": False,
        "evidence_grade": "empirical_lossless_byte_transform",
        "source_archive": {
            "path": str(source_archive.resolve()),
            "bytes": source_bytes,
            "sha256": _sha256_file(source_archive),
        },
        "output_archive": {
            "path": str(output_archive.resolve()),
            "bytes": output_bytes,
            "sha256": _sha256_file(output_archive),
        },
        "archive_delta_bytes_vs_source": output_bytes - source_bytes,
        "formula_only_rate_score_delta_vs_source": (output_bytes - source_bytes) * RATE_SCORE_PER_BYTE,
        "source_payload": {
            "bytes": len(source_payload),
            "sha256": source_payload_sha,
            "format": source_header.get("payload_format"),
        },
        "payload": {
            "bytes": len(candidate_payload),
            "sha256": candidate_payload_sha,
            "format": candidate_header.get("payload_format"),
            "member": "p",
        },
        "source_preserving": True,
        "decoded_stream_parity": True,
        "decoded_stream_parity_detail": {
            "status": "passed",
            "members_compared": sorted(source_decoded),
            "source_decoded_members": _decoded_summary(source_decoded),
            "candidate_runtime_members": _runtime_member_summary(candidate_header),
        },
        "noop": source_payload == candidate_payload and source_bytes == output_bytes,
        "source_preservation": {
            "status": (
                "byte_identical_noop"
                if source_payload == candidate_payload and source_bytes == output_bytes
                else "lossless_decoded_stream_preserving_repack"
            ),
            "decoded_streams_byte_identical": True,
            "payload_byte_identical_to_source": source_payload == candidate_payload,
            "source_payload_sha256": source_payload_sha,
            "candidate_payload_sha256": candidate_payload_sha,
        },
        "stream_choices": {
            name: {
                "bytes": len(choice.data),
                "sha256": _sha256_bytes(choice.data),
                "params": choice.params_json,
            }
            for name, choice in choices.items()
        },
        "runtime_parse_validation": {
            "parser": str(UNPACKER_PATH),
            "payload_format": candidate_header.get("payload_format"),
            "members": _runtime_member_summary(candidate_header),
        },
    }
    _write_json(output_archive.parent / "manifest.json", manifest)
    return manifest, source_decoded


def _select_qpost_pairs(
    *,
    qpost_builder: Any,
    qpost_atoms: Any,
    pr65_archive: Path,
    c089_trace: Path,
    pr65_trace: Path,
    include_streams: tuple[str, ...],
    top_pairs: int,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, bytes]]:
    streams = qpost_builder.extract_pr65_qpost_streams(pr65_archive)
    arrays = qpost_atoms._decode_stream_arrays(qpost_builder, streams)  # noqa: SLF001
    c089_trace_records = qpost_atoms._load_trace(c089_trace)  # noqa: SLF001
    pr65_trace_records = qpost_atoms._load_trace(pr65_trace)  # noqa: SLF001
    ranked = qpost_atoms._rank_pairs(  # noqa: SLF001
        arrays=arrays,
        include_streams=include_streams,
        c089_trace=c089_trace_records,
        pr65_trace=pr65_trace_records,
        positive_trace_only=True,
    )
    if len(ranked) < top_pairs:
        raise MicrostackBuildError(
            f"only {len(ranked)} qpost-active positive-trace pairs available; "
            f"need {top_pairs}"
        )
    selected = ranked[:top_pairs]
    selected_pairs = [int(row["pair_index"]) for row in selected]
    no_op = qpost_atoms._no_op_proof(  # noqa: SLF001
        arrays=arrays,
        include_streams=include_streams,
        selected_pairs=selected_pairs,
    )
    if bool(no_op["is_noop"]):
        raise MicrostackBuildError("selected qpost atom subset is a no-op")
    return selected, no_op, streams


def _exact_eval_command(archive: Path, output_dir: Path, candidate_id: str) -> str:
    work_dir = output_dir / "exact_eval_work" / candidate_id
    return (
        ".venv/bin/python -u experiments/contest_auth_eval.py "
        f"--archive {archive.resolve()} "
        "--inflate-sh submissions/robust_current/inflate.sh "
        "--upstream-dir upstream --device cuda --keep-work-dir "
        f"--work-dir {work_dir.resolve()}"
    )


def _claim_command(archive_sha256: str) -> str:
    return (
        'ETA_UTC="$(date -u -v+45M +%Y-%m-%dT%H:%MZ)"\n'
        'JOB_ID="exact_eval_pr75_qpost_microstack_bias032_t4_$(date -u +%Y%m%dT%H%MZ)"\n'
        "tools/claim_lane_dispatch.py claim "
        "--lane-id pr75_qpost_microstack_bias032_c089p6 "
        "--platform lightning "
        '--instance-job-id "$JOB_ID" '
        "--agent codex:gpt-5.5 "
        '--predicted-eta-utc "$ETA_UTC" '
        "--status eval "
        f'--notes "C089 P6 lossless resweep + PR65 bias top32 qpost; archive_sha256={archive_sha256}"'
    )


def build_microstack(
    *,
    source_archive: Path = DEFAULT_SOURCE_ARCHIVE,
    pr65_archive: Path = DEFAULT_PR65_ARCHIVE,
    c089_trace: Path = DEFAULT_C089_TRACE,
    pr65_trace: Path = DEFAULT_PR65_TRACE,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    top_pairs: int = 32,
    include_streams: tuple[str, ...] = ("bias",),
    expected_source_sha256: str | None = EXPECTED_SOURCE_SHA256,
    expected_pr65_sha256: str | None = EXPECTED_PR65_SHA256,
    brotli_params: Sequence[tuple[int, int, int, int]] | None = None,
) -> dict[str, Any]:
    if top_pairs <= 0 or top_pairs > 600:
        raise MicrostackBuildError(f"top_pairs must be in [1, 600], got {top_pairs}")
    if include_streams != ("bias",):
        raise MicrostackBuildError(
            "this micro-stack is intentionally bias-only; add a reviewed risk guard "
            "before enabling post/region/motion qpost streams"
        )
    source_archive = source_archive.resolve()
    pr65_archive = pr65_archive.resolve()
    c089_trace = c089_trace.resolve()
    pr65_trace = pr65_trace.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if brotli_params is None:
        brotli_params = micro_brotli_param_grid()

    source_sha = _verify_source_sha(source_archive, expected_source_sha256, "source archive")
    pr65_sha = _verify_source_sha(pr65_archive, expected_pr65_sha256, "PR65 archive")
    runtime_hook = _verify_runtime_hook()
    if not runtime_hook["qpost_runtime_hook_present"]:
        raise MicrostackBuildError("inflate runtime does not apply qpost.bin; refusing sidecar build")

    unpacker = _load_unpacker()
    qpost_builder = _load_qpost_builder()
    qpost_atoms = _load_qpost_atoms()

    resweep_archive = output_dir / "c089_p6_lossless_resweep" / "archive.zip"
    resweep_manifest, _source_decoded = _build_reswept_p6_base(
        source_archive=source_archive,
        output_archive=resweep_archive,
        brotli_params=brotli_params,
        unpacker=unpacker,
    )
    selected_rank_records, no_op, original_streams = _select_qpost_pairs(
        qpost_builder=qpost_builder,
        qpost_atoms=qpost_atoms,
        pr65_archive=pr65_archive,
        c089_trace=c089_trace,
        pr65_trace=pr65_trace,
        include_streams=include_streams,
        top_pairs=top_pairs,
    )

    candidate_id = f"c089_p6_resweep_pr65_qpost_bias_top{top_pairs:03d}"
    candidate_dir = output_dir / candidate_id
    archive = candidate_dir / "archive.zip"
    pair_indices = tuple(int(row["pair_index"]) for row in selected_rank_records)
    qpost_meta = qpost_builder.build_candidate(
        resweep_archive,
        pr65_archive,
        archive,
        include_streams=include_streams,
        pair_indices=pair_indices,
    )
    archive_bytes = int(qpost_meta["output_archive_bytes"])
    source_bytes = source_archive.stat().st_size
    archive_delta_vs_source = archive_bytes - source_bytes
    rate_delta_vs_source = archive_delta_vs_source * RATE_SCORE_PER_BYTE
    break_even = (SOURCE_SCORE - SUB314_TARGET) + rate_delta_vs_source
    trace_bound = float(
        sum(float(row.get("public_trace_opportunity", 0.0)) for row in selected_rank_records)
    )
    dispatch_blockers: list[str] = []
    if trace_bound <= 0.0:
        dispatch_blockers.append("no positive public-trace opportunity bound")
    if trace_bound < break_even:
        dispatch_blockers.append(
            f"public-trace opportunity bound {trace_bound:.9f} below "
            f"sub-0.314 break-even {break_even:.9f}"
        )
    if bool(no_op["is_noop"]):
        dispatch_blockers.append("qpost subset is no-op after identity-default filtering")
    if not bool(resweep_manifest["decoded_stream_parity"]):
        dispatch_blockers.append("lossless P6 resweep did not prove decoded-stream parity")
    dispatch_recommendation = (
        "exact_cuda_eval_candidate_after_lane_claim"
        if not dispatch_blockers
        else "do_not_dispatch"
    )
    archive_sha = str(qpost_meta["output_archive_sha256"])
    exact_eval_command = _exact_eval_command(archive, output_dir, candidate_id)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "tool": TOOL,
        "candidate_id": candidate_id,
        "score_claim": False,
        "evidence_grade": "byte_trace_planning_only_until_exact_cuda",
        "source_archive": {
            "path": str(source_archive),
            "bytes": source_bytes,
            "sha256": source_sha,
            "score": SOURCE_SCORE,
        },
        "pr65_archive": {
            "path": str(pr65_archive),
            "bytes": pr65_archive.stat().st_size,
            "sha256": pr65_sha,
        },
        "resweep_base": {
            "archive": resweep_manifest["output_archive"],
            "archive_delta_bytes_vs_source": resweep_manifest["archive_delta_bytes_vs_source"],
            "decoded_stream_parity": resweep_manifest["decoded_stream_parity"],
            "source_preservation": resweep_manifest["source_preservation"],
            "stream_choices": resweep_manifest["stream_choices"],
            "manifest": str((resweep_archive.parent / "manifest.json").resolve()),
        },
        "qpost": {
            "include_streams": list(include_streams),
            "selected_top_pairs": top_pairs,
            "selected_pairs": list(pair_indices),
            "selected_rank_records": selected_rank_records,
            "no_op_proof": no_op,
            "members": qpost_meta["members"],
            "qpost_streams": qpost_meta["qpost_streams"],
            "original_stream_shas": {
                name: _sha256_bytes(blob)
                for name, blob in sorted(original_streams.items())
            },
            "nonselected_pairs_default_to_identity": True,
            "randmulti_supported": False,
        },
        "output_archive": {
            "path": str(archive.resolve()),
            "bytes": archive_bytes,
            "sha256": archive_sha,
        },
        "archive_delta_bytes_vs_source": archive_delta_vs_source,
        "formula_rate_score_delta_vs_source": rate_delta_vs_source,
        "sub314_break_even_component_gain": break_even,
        "public_trace_opportunity_bound": trace_bound,
        "estimated_score_if_trace_bound_realized": SOURCE_SCORE - trace_bound + rate_delta_vs_source,
        "safety_preflight": {
            "status": "local_preflight_passed" if not dispatch_blockers else "local_preflight_warn",
            "runtime_hook": runtime_hook,
            "decoded_stream_parity": True,
            "qpost_non_noop": not bool(no_op["is_noop"]),
            "archive_members": ["p", "qpost.bin"],
            "remote_dispatched": False,
            "requires_lane_claim": True,
            "dispatch_recommendation": dispatch_recommendation,
            "dispatch_blockers": dispatch_blockers,
        },
        "dispatch": {
            "remote_dispatched": False,
            "claim_command_draft": _claim_command(archive_sha),
            "exact_eval_command_draft": exact_eval_command,
        },
    }
    if not math.isfinite(float(manifest["estimated_score_if_trace_bound_realized"])):
        raise MicrostackBuildError("non-finite estimated score")
    _write_json(candidate_dir / "manifest.json", manifest)

    summary = {
        "schema_version": SCHEMA_VERSION,
        "tool": TOOL,
        "run_id": RUN_ID,
        "score_claim": False,
        "evidence_grade": "byte_trace_planning_only_until_exact_cuda",
        "source_archive_sha256": source_sha,
        "pr65_archive_sha256": pr65_sha,
        "best_candidate": {
            "candidate_id": candidate_id,
            "archive": str(archive.resolve()),
            "bytes": archive_bytes,
            "sha256": archive_sha,
            "dispatch_recommendation": dispatch_recommendation,
            "public_trace_opportunity_bound": trace_bound,
            "sub314_break_even_component_gain": break_even,
        },
        "artifacts": {
            "resweep_manifest": str((resweep_archive.parent / "manifest.json").resolve()),
            "candidate_manifest": str((candidate_dir / "manifest.json").resolve()),
        },
        "remote_dispatched": False,
    }
    _write_json(output_dir / "candidate_summary.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-archive", type=Path, default=DEFAULT_SOURCE_ARCHIVE)
    parser.add_argument("--pr65-archive", type=Path, default=DEFAULT_PR65_ARCHIVE)
    parser.add_argument("--c089-trace", type=Path, default=DEFAULT_C089_TRACE)
    parser.add_argument("--pr65-trace", type=Path, default=DEFAULT_PR65_TRACE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--top-pairs", type=int, default=32)
    parser.add_argument(
        "--allow-source-sha-mismatch",
        action="store_true",
        help="Planning/test override; default fails closed on C-089/PR65 SHA drift.",
    )
    parser.add_argument(
        "--exhaustive-brotli-grid",
        action="store_true",
        help="Use the full Brotli grid instead of the small C-089 micro grid.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    summary = build_microstack(
        source_archive=args.source_archive,
        pr65_archive=args.pr65_archive,
        c089_trace=args.c089_trace,
        pr65_trace=args.pr65_trace,
        output_dir=args.output_dir,
        top_pairs=args.top_pairs,
        expected_source_sha256=None if args.allow_source_sha_mismatch else EXPECTED_SOURCE_SHA256,
        expected_pr65_sha256=None if args.allow_source_sha_mismatch else EXPECTED_PR65_SHA256,
        brotli_params=(
            exhaustive_brotli_param_grid()
            if args.exhaustive_brotli_grid
            else micro_brotli_param_grid()
        ),
    )
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
