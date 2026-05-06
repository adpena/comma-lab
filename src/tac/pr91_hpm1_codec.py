"""Local fail-closed PR91 HPM1 mask replay and re-encode helpers.

The functions in this module are forensic/preflight tooling only. They never
load contest scorers, run exact eval, dispatch GPU work, or claim a contest
score. They validate the public PR91 archive byte custody, slice ``HPM1``
mask segments, and run probability-contract probe matrices against the
parallel PR86 HPAC source contract.

REHYDRATED 2026-05-05 from .recovery_spec.json (preserved at
.recovery_quarantine_20260505T004735Z/src/tac/pr91_hpm1_codec.recovery_spec.json).
Spec source: bytecode disassembly of compiled .pyc; whitespace + inline comments lost.

PARTIAL REHYDRATION: All ``EXPECTED_*`` constants, default paths, error class,
and ``Hpm1MaskPayload`` dataclass are reconstructed exactly. The HPM1 byte
grammar, static custody checks, local-only residual helpers, and fail-closed
probability matrix/preflight surfaces are implemented. The byte-exact torch /
HPAC arithmetic/range decode loop remains blocked until the PR91 probability
contract is recovered from source-level evidence.
"""
from __future__ import annotations

import hashlib
import json
import time
import struct
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

import numpy as np

try:  # pragma: no cover - optional in lite environments
    import torch
    import torch.nn.functional as F
    from torch import nn
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]

from tac.pr85_bundle import (
    HPM1_HEADER_BYTES,
    HPM1_MAGIC,
    Pr85BundleError,
    Pr85SegmentContract,
    SEGMENT_ORDER,
    pack_pr85_bundle,
    parse_hpm1_mask_segment,
    parse_pr85_bundle,
)
from tac.pr86_hpac_codec import (
    DEFAULT_HPAC_PROBABILITY_VARIANT,
    DEFAULT_PR86_ARCHIVE,
    EXPECTED_PR86_TOKENS_SHA256,
    HPACMini,
    Pr86HpacReplayError,
    _categorical_from_probs,
    _group_masks,
    _normalize_probability_row,
    collect_dependency_report,
    decode_tokens_hpac,
    encode_symbols_hpac_with_prev_context,
    encode_tokens_hpac,
    load_hpac_model_from_ppmd,
    read_pr86_archive,
    resolve_hpac_probability_variant,
    sha256_bytes,
    supported_hpac_probability_variant_names,
)

REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_PR91_INTAKE_DIR = (
    REPO_ROOT / "experiments/results/public_pr91_intake_20260504_codex"
)
DEFAULT_PR91_RUNTIME_SOURCE_DIR = (
    DEFAULT_PR91_INTAKE_DIR / "replay_submission/hpac_coder_hybrid"
)
DEFAULT_PR91_ARCHIVE = (
    REPO_ROOT
    / "experiments/results/public_pr91_intake_20260504_codex/archive.zip"
)
DEFAULT_PR85_STBM_EXACT_DIR = (
    REPO_ROOT
    / "experiments/results/lightning_batch/exact_eval_pr85_stbm1br_stbm_runtime_t4_g4dn2x_20260504T0613Z"
)
DEFAULT_PR85_STBM_ARCHIVE = DEFAULT_PR85_STBM_EXACT_DIR / "archive.zip"
DEFAULT_PR85_STBM_ADJUDICATED_JSON = (
    DEFAULT_PR85_STBM_EXACT_DIR / "contest_auth_eval.adjudicated.json"
)
DEFAULT_PR85_QMA9_TOKEN_SOURCE = (
    REPO_ROOT
    / "experiments/results/public_pr85_intake_20260503_codex/qma9_token_source/pr85_qma9_tokens_u8_storage_order.bin"
)

CONTEST_ARCHIVE_BYTE_DENOMINATOR = 37545489
EXPECTED_PR91_ARCHIVE_BYTES = 222404
EXPECTED_PR91_ARCHIVE_SHA256 = (
    "4c16d04c746c981feb902e4dd508ffadaf3615e532d351993c3d2f6eccda1b4f"
)
EXPECTED_PR91_MEMBER_X_BYTES = 222304
EXPECTED_PR91_MEMBER_X_SHA256 = (
    "5c213c61cc4d29b62286063bfdcb97e812af6b06c0021aeaecc8bc46644e17bf"
)
EXPECTED_PR91_HPM1_MASK_BYTES = 145087
EXPECTED_PR91_HPM1_MASK_SHA256 = (
    "a4ed57ff0af1d8c914f004de165aeead50ec8dd61e99b0afdfbfa2d1e7fd9fcc"
)
EXPECTED_PR91_HPM1_TOKENS_SHA256 = (
    "541016d83852a5bb3e0738caa3b44d7b2b0f7372f1841085cf9554f039c6cf6b"
)
EXPECTED_PR91_HPM1_HPAC_SHA256 = (
    "de7638c531c9dafa06148602cf784bf3ae9997f326f85cc25b9f3646b536abdd"
)
EXPECTED_PR85_QMA9_TOKEN_SOURCE_SHA256 = (
    "c1c47434fd1e6c876cb3e44910f5ab2e124285d9dba2f300bcf322d03fb8bb5a"
)
EXPECTED_PR85_STBM_ARCHIVE_BYTES = 229756
EXPECTED_PR85_STBM_ARCHIVE_SHA256 = (
    "c6f004d444ed32c628611a2f21f567c666af6bcbcceba618cc089ec024a0cda6"
)
EXPECTED_PR85_STBM_MEMBER_X_SHA256 = (
    "c7586795bb29fb0ef611ad44715aec77e0e815370e19674d4c89ef2a54b417b5"
)
EXPECTED_PR85_STBM_HPM1_PROJECTION_SCORE = 0.248795

DEFAULT_PR91_HPM1_CONTEXT_WINDOWS = ((33, 8), (5948, 8))
PR91_HPM1_CONTEXT_MODES = ("decoded_context", "reference_context")

_QUARANTINE_SPEC = (
    ".recovery_quarantine_20260505T004735Z/src/tac/pr91_hpm1_codec.recovery_spec.json"
)


class Pr91Hpm1Error(RuntimeError):
    """Raised on PR91 HPM1 replay or byte-parity failure."""

    def __init__(self, contract: str, code: str, **fields: Any) -> None:
        message_parts = [f"contract={contract}", f"code={code}"]
        for k, v in fields.items():
            message_parts.append(f"{k}={v!r}")
        super().__init__("; ".join(message_parts))
        self.contract = contract
        self.code = code
        self.fields = dict(fields)


@dataclass(frozen=True)
class Hpm1MaskPayload:
    """Parsed PR91 HPM1 mask segment: header config + token + HPAC bytes."""

    n_frames: int
    height: int
    width: int
    predictor_count: int
    delta: int
    channels: int
    use_spm: int
    hpac_d_film: int
    tokens_len: int
    hpac_len: int
    ppmd_order: int
    tokens: bytes
    hpac: bytes
    extra: Mapping[str, Any] = field(default_factory=dict)

    def config(self) -> dict[str, int]:
        """Return only the integer header config (no payload bytes)."""
        return {
            "n_frames": int(self.n_frames),
            "height": int(self.height),
            "width": int(self.width),
            "predictor_count": int(self.predictor_count),
            "delta": int(self.delta),
            "channels": int(self.channels),
            "use_spm": int(self.use_spm),
            "hpac_d_film": int(self.hpac_d_film),
            "tokens_len": int(self.tokens_len),
            "hpac_len": int(self.hpac_len),
            "ppmd_order": int(self.ppmd_order),
        }


def repo_rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def sha256_path(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return repo_rel(value)
    if isinstance(value, np.generic):
        return value.item()
    if torch is not None and isinstance(value, torch.Tensor):
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }
    if isinstance(value, bytes):
        return {"bytes": len(value), "sha256": sha256_bytes(value)}
    return value


def _validate_safe_single_x_archive(
    archive: Path,
) -> tuple[int, str, bytes]:
    if not archive.is_file():
        raise Pr91Hpm1Error("archive_contract", "archive_missing", archive=archive)
    archive_size = archive.stat().st_size
    archive_sha = sha256_path(archive)
    with zipfile.ZipFile(archive, "r") as zf:
        names = [info.filename for info in zf.infolist()]
        if names != ["x"]:
            raise Pr91Hpm1Error(
                "archive_contract",
                "expected_single_x_member",
                archive=archive,
                got_members=names,
            )
        body = zf.read("x")
    return archive_size, archive_sha, body


def split_hpm1_mask_segment(segment: bytes) -> Hpm1MaskPayload:
    """Split an HPM1 mask segment into its typed payload."""
    contract = parse_hpm1_mask_segment(segment)
    metadata = dict(contract.metadata)
    token_start = HPM1_HEADER_BYTES
    token_end = token_start + int(metadata["tokens_len"])
    hpac_end = token_end + int(metadata["hpac_len"])
    return Hpm1MaskPayload(
        n_frames=int(metadata["n_frames"]),
        height=int(metadata["height"]),
        width=int(metadata["width"]),
        predictor_count=int(metadata["predictor_count"]),
        delta=int(metadata["delta"]),
        channels=int(metadata["channels"]),
        use_spm=int(metadata["use_spm"]),
        hpac_d_film=int(metadata["hpac_d_film"]),
        tokens_len=int(metadata["tokens_len"]),
        hpac_len=int(metadata["hpac_len"]),
        ppmd_order=int(metadata["ppmd_order"]),
        tokens=segment[token_start:token_end],
        hpac=segment[token_end:hpac_end],
        extra={
            "tokens_sha256": metadata["tokens_sha256"],
            "hpac_sha256": metadata["hpac_sha256"],
            "tokens_uint32_aligned": metadata["tokens_uint32_aligned"],
        },
    )


def extract_pr91_hpm1_payload(archive: Path) -> Hpm1MaskPayload:
    """Extract the HPM1 mask payload from a PR91 archive."""
    _, _, body = _validate_safe_single_x_archive(archive)
    bundle = parse_pr85_bundle(body)
    mask_segment = bytes(bundle.segments["mask"])
    if not mask_segment.startswith(HPM1_MAGIC):
        raise Pr91Hpm1Error(
            "mask_segment_contract",
            "expected_hpm1_magic",
            magic=mask_segment[:4],
        )
    return split_hpm1_mask_segment(mask_segment)


def _rehydration_failure(symbol: str) -> NotImplementedError:
    return NotImplementedError(
        f"rehydration incomplete: {symbol} requires intricate torch / HPAC "
        f"compose chain that pycdc cannot fully decompile; original bytecode "
        f"preserved in {_QUARANTINE_SPEC}"
    )


def _validate_dependency_report(report: Mapping[str, Any]) -> None:
    """Fail closed when an optional replay dependency is explicitly missing.

    Dependency collection lives in the PR86 HPAC module.  That module is still
    partially rehydrated, so this validator accepts lightweight reports while
    preserving a concrete failure shape for callers that do have dependency
    facts.
    """

    missing = list(report.get("missing", []) or report.get("missing_dependencies", []) or [])
    if missing:
        raise Pr91Hpm1Error(
            "dependency_contract",
            "missing_hpm1_replay_dependencies",
            missing=missing,
        )


def _common_prefix_bytes(a: bytes, b: bytes) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def _extract_call_argument(*args: Any, **kwargs: Any) -> Any:
    raise _rehydration_failure("_extract_call_argument")


def _extract_first_call_body(*args: Any, **kwargs: Any) -> Any:
    raise _rehydration_failure("_extract_first_call_body")


def analyze_pr91_hpm1_runtime_sources(*args: Any, **kwargs: Any) -> Any:
    source_dir = Path(kwargs.get("source_dir") or DEFAULT_PR91_RUNTIME_SOURCE_DIR)
    if not source_dir.is_dir():
        return {
            "status": "failed_closed_missing_sources",
            "source_dir": repo_rel(source_dir),
            "score_claim": False,
        }
    files = []
    for path in sorted(source_dir.rglob("*")):
        if path.is_file():
            rel = path.relative_to(source_dir).as_posix()
            data = path.read_bytes()
            files.append({"path": rel, "bytes": len(data), "sha256": sha256_bytes(data)})
    return {
        "status": "passed_static_source_inventory",
        "source_dir": repo_rel(source_dir),
        "score_claim": False,
        "file_count": len(files),
        "files": files,
        "contains_range_codec_cpp": any(row["path"].endswith("range_mask_codec.cpp") for row in files),
    }


def compare_hpm1_to_pr86_hpac_contract(
    archive: Path | Hpm1MaskPayload, *, reference: Path | None = None
) -> dict[str, Any]:
    payload = archive if isinstance(archive, Hpm1MaskPayload) else extract_pr91_hpm1_payload(Path(archive))
    reference_archive = Path(reference or DEFAULT_PR86_ARCHIVE)
    pr86_available = reference_archive.is_file()
    relationship = {
        "status": "passed_static_hpac_relationship",
        "score_claim": False,
        "dispatch_allowed": False,
        "pr91_tokens_sha256": sha256_bytes(payload.tokens),
        "pr91_hpac_sha256": sha256_bytes(payload.hpac),
        "expected_pr86_tokens_sha256": EXPECTED_PR86_TOKENS_SHA256,
        "expected_pr91_hpm1_hpac_sha256": EXPECTED_PR91_HPM1_HPAC_SHA256,
        "hpac_model_matches_expected_pr91": sha256_bytes(payload.hpac) == EXPECTED_PR91_HPM1_HPAC_SHA256,
        "tokens_match_pr86_expected": sha256_bytes(payload.tokens) == EXPECTED_PR86_TOKENS_SHA256,
        "pr86_archive_available": pr86_available,
        "pr86_archive": repo_rel(reference_archive),
    }
    if not pr86_available:
        relationship["status"] = "failed_closed_pr86_archive_unavailable"
    return relationship


def validate_hpm1_static_contract(archive: Path | Hpm1MaskPayload) -> dict[str, Any]:
    payload = archive if isinstance(archive, Hpm1MaskPayload) else extract_pr91_hpm1_payload(Path(archive))
    failures: list[str] = []
    config = payload.config()
    if payload.tokens_len != len(payload.tokens):
        failures.append("tokens_len_mismatch")
    if payload.hpac_len != len(payload.hpac):
        failures.append("hpac_len_mismatch")
    if payload.tokens_len <= 0:
        failures.append("tokens_len_nonpositive")
    if payload.hpac_len <= 0:
        failures.append("hpac_len_nonpositive")
    if payload.tokens_len % 4:
        failures.append("tokens_not_uint32_aligned")
    for key in ("n_frames", "height", "width", "predictor_count", "channels", "hpac_d_film"):
        if int(config[key]) <= 0:
            failures.append(f"{key}_nonpositive")
    if int(config["delta"]) < 0:
        failures.append("delta_negative")
    if int(config["use_spm"]) not in (0, 1):
        failures.append("use_spm_not_boolean")
    return {
        "schema": "pr91_hpm1_static_contract_v1",
        "status": "failed" if failures else "passed",
        "score_claim": False,
        "dispatch_allowed": False,
        "config": config,
        "tokens": {
            "bytes": len(payload.tokens),
            "sha256": sha256_bytes(payload.tokens),
            "uint32_aligned": payload.tokens_len % 4 == 0,
        },
        "hpac": {"bytes": len(payload.hpac), "sha256": sha256_bytes(payload.hpac)},
        "failures": failures,
    }


def load_hpm1_hpac_model(payload: Hpm1MaskPayload, *, device: str = "cpu") -> Any:
    raise _rehydration_failure("load_hpm1_hpac_model")


def run_pr91_hpm1_preflight(
    archive: Path,
    *,
    max_frames: int | None = 1,
    attempt_reencode: bool = False,
    probability_variant: str = DEFAULT_HPAC_PROBABILITY_VARIANT,
    device: str = "cpu",
    output_dir: Path | None = None,
    strict: bool = False,
    summary: bool = True,
    write_json: bool = True,
) -> dict[str, Any]:
    started_at = time.time()
    payload = extract_pr91_hpm1_payload(Path(archive))
    static_report = validate_hpm1_static_contract(payload)
    relationship = compare_hpm1_to_pr86_hpac_contract(payload)
    report: dict[str, Any] = {
        "schema_version": 1,
        "tool": "tac.pr91_hpm1_codec.run_pr91_hpm1_preflight",
        "recorded_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "status": "blocked_hpm1_probability_range_contract_mismatch",
        "score_claim": False,
        "dispatch_performed": False,
        "gpu_or_remote_work": False,
        "local_only": True,
        "device": device,
        "max_frames": max_frames,
        "attempt_reencode": attempt_reencode,
        "probability_variant": probability_variant,
        "archive": _archive_report(Path(archive)),
        "hpm1_static_contract": static_report,
        "pr86_hpac_relationship": relationship,
        "prefix_or_full_decode": {
            "status": "failed_closed",
            "failure_stage": "hpac_probability_range_decode",
            "failure_reason": (
                "Full HPAC arithmetic/range replay is not rehydrated in tac; "
                "public PR91 remains non-dispatchable until a byte-exact "
                "probability/range contract decodes the HPM1 token stream."
            ),
            "max_frames": max_frames,
        },
        "hpac_reencode": {
            "attempted": bool(attempt_reencode),
            "status": "not_attempted" if not attempt_reencode else "blocked_by_decode_contract",
        },
        "failure_stage": "hpac_probability_range_decode",
        "failure_reason": "hpm1_probability_range_contract_not_byte_closed",
        "failure_context": {
            "known_public_failure": "frame=0 group=10 symbol=191 after 5951 decoded symbols",
            "evidence_grade": "byte_static_plus_prior_local_prefix_diagnostic",
        },
        "blocker_class": "hpm1_probability_range_contract_mismatch",
        "dispatch_unlocked": False,
        "elapsed_sec": round(time.time() - started_at, 3),
    }
    if static_report["status"] != "passed":
        report["status"] = "failed_static_hpm1_contract"
        report["blocker_class"] = "hpm1_static_contract"
    if write_json and output_dir is not None:
        write_json_report(report, Path(output_dir) / "pr91_hpm1_preflight.json")
    if strict and report["status"] != "passed":
        raise Pr91Hpm1Error("hpm1_preflight", str(report["status"]), report=report)
    return _jsonable(report)


def run_pr91_hpm1_probability_variant_matrix(
    archive: Path,
    *,
    output_dir: Path | None = None,
    variants: tuple[str, ...] | None = None,
    max_frames: int | None = 1,
    attempt_reencode: bool = False,
    strict: bool = False,
    summary: bool = True,
    write_json: bool = True,
) -> dict[str, Any]:
    """Run a fail-closed PR91 HPM1 probability-variant probe matrix.

    This implementation is intentionally conservative: it validates the archive
    and enumerates the known HPAC probability contracts, but it does not claim
    decode success without a rehydrated byte-exact range decoder.
    """

    started_at = time.time()
    payload = extract_pr91_hpm1_payload(Path(archive))
    variant_names = variants or supported_hpac_probability_variant_names()
    rows: list[dict[str, Any]] = []
    for name in variant_names:
        variant = resolve_hpac_probability_variant(name)
        rows.append(
            {
                "variant": variant.name,
                "probability_dtype": variant.probability_dtype,
                "categorical_perfect": variant.categorical_perfect,
                "source_contract": variant.source_contract,
                "status": "failed_closed",
                "decoded_frame0": False,
                "byte_exact_reencode": False,
                "failure_stage": "hpac_probability_range_decode",
                "failure_reason": "hpm1_probability_range_contract_not_byte_closed",
                "known_public_failure": (
                    "source contract fails at frame=0 group=10 symbol=191 after "
                    "5951 decoded symbols"
                    if variant.source_contract
                    else None
                ),
            }
        )
    report = {
        "schema": "pr91_hpm1_probability_variant_matrix_v1",
        "tool": "tac.pr91_hpm1_codec.run_pr91_hpm1_probability_variant_matrix",
        "recorded_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "status": "blocked_hpm1_probability_range_contract_mismatch",
        "score_claim": False,
        "dispatch_allowed": False,
        "dispatch_performed": False,
        "max_frames": max_frames,
        "attempt_reencode": attempt_reencode,
        "archive": _archive_report(Path(archive)),
        "hpm1_static_contract": validate_hpm1_static_contract(payload),
        "payload": {
            "config": payload.config(),
            "tokens_sha256": sha256_bytes(payload.tokens),
            "hpac_sha256": sha256_bytes(payload.hpac),
        },
        "variant_results": rows,
        "failed_variants": [row["variant"] for row in rows if row["status"] != "passed"],
        "elapsed_sec": round(time.time() - started_at, 3),
    }
    if write_json and output_dir is not None:
        write_json_report(report, Path(output_dir) / "pr91_hpm1_probability_variant_matrix.json")
    if strict:
        raise Pr91Hpm1Error("hpm1_probability_variant_matrix", str(report["status"]), report=report)
    return _jsonable(report)


def _hpm1_token_stream_transform_candidates(*args: Any, **kwargs: Any) -> Any:
    raise _rehydration_failure("_hpm1_token_stream_transform_candidates")


def run_pr91_hpm1_stream_transform_probe(*args: Any, **kwargs: Any) -> Any:
    raise _rehydration_failure("run_pr91_hpm1_stream_transform_probe")


def _load_reference_tokens(*args: Any, **kwargs: Any) -> Any:
    raise _rehydration_failure("_load_reference_tokens")


def _symbol_position(*args: Any, **kwargs: Any) -> Any:
    raise _rehydration_failure("_symbol_position")


def _archive_report(archive: Path) -> dict[str, Any]:
    archive_size, archive_sha, body = _validate_safe_single_x_archive(archive)
    bundle = parse_pr85_bundle(body)
    return {
        "path": repo_rel(archive),
        "bytes": archive_size,
        "sha256": archive_sha,
        "member_x_bytes": len(body),
        "member_x_sha256": sha256_bytes(body),
        "bundle_format": bundle.format,
        "segment_bytes": {name: len(bundle.segments[name]) for name in SEGMENT_ORDER},
        "segment_sha256": {name: sha256_bytes(bundle.segments[name]) for name in SEGMENT_ORDER},
    }


def build_hpm1_mask_segment(
    tokens_blob: bytes,
    hpac_ppmd_blob: bytes,
    *,
    N: int,
    H: int,
    W: int,
    P: int,
    delta: int,
    ch: int,
    use_spm: bool,
    hpac_d_film: int,
    ppmd_order: int = 4,
) -> bytes:
    """Build a deterministic HPM1 mask segment from typed byte payloads."""

    if len(tokens_blob) <= 0 or len(tokens_blob) % 4:
        raise Pr91Hpm1Error(
            "hpm1_segment_builder",
            "tokens_blob_must_be_nonempty_uint32_aligned",
            tokens_bytes=len(tokens_blob),
        )
    if len(hpac_ppmd_blob) <= 0:
        raise Pr91Hpm1Error(
            "hpm1_segment_builder",
            "hpac_ppmd_blob_must_be_nonempty",
            hpac_bytes=len(hpac_ppmd_blob),
        )
    fields = (N, H, W, P, delta, ch, int(bool(use_spm)), hpac_d_film, len(tokens_blob), len(hpac_ppmd_blob), ppmd_order)
    for name, value in zip(
        ("N", "H", "W", "P", "delta", "ch", "use_spm", "hpac_d_film", "tokens_len", "hpac_len", "ppmd_order"),
        fields,
    ):
        if int(value) < 0:
            raise Pr91Hpm1Error("hpm1_segment_builder", "negative_header_field", field=name, value=value)
    required_positive = {
        "N",
        "H",
        "W",
        "P",
        "ch",
        "hpac_d_film",
        "tokens_len",
        "hpac_len",
        "ppmd_order",
    }
    for name, value in zip(
        ("N", "H", "W", "P", "delta", "ch", "use_spm", "hpac_d_film", "tokens_len", "hpac_len", "ppmd_order"),
        fields,
    ):
        if name not in required_positive:
            continue
        if int(value) <= 0:
            raise Pr91Hpm1Error("hpm1_segment_builder", "nonpositive_header_field", field=name, value=value)
    return HPM1_MAGIC + struct.pack("<IIIIIIIIIII", *map(int, fields)) + bytes(tokens_blob) + bytes(hpac_ppmd_blob)


def raw_tokens_to_mod5_residual_symbols(raw_tokens: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert raw 0..4 HPM1 tokens into previous-frame mod-5 residual symbols."""

    raw = np.asarray(raw_tokens, dtype=np.uint8)
    if raw.ndim != 3:
        raise Pr91Hpm1Error("hpm1_residual_symbols", "raw_tokens_must_be_nhw", shape=list(raw.shape))
    if raw.size and (int(raw.min()) < 0 or int(raw.max()) > 4):
        raise Pr91Hpm1Error("hpm1_residual_symbols", "raw_tokens_out_of_mod5_range")
    prev = np.zeros_like(raw, dtype=np.uint8)
    if raw.shape[0] > 1:
        prev[1:] = raw[:-1]
    residual = ((raw.astype(np.int16) - prev.astype(np.int16)) % 5).astype(np.uint8)
    return residual, prev


def reconstruct_raw_tokens_from_mod5_residual_symbols(
    symbols: np.ndarray, prev_context_tokens: np.ndarray
) -> np.ndarray:
    symbols_arr = np.asarray(symbols, dtype=np.uint8)
    prev = np.asarray(prev_context_tokens, dtype=np.uint8)
    if symbols_arr.shape != prev.shape:
        raise Pr91Hpm1Error(
            "hpm1_residual_symbols",
            "symbol_prev_shape_mismatch",
            symbols_shape=list(symbols_arr.shape),
            prev_shape=list(prev.shape),
        )
    return ((symbols_arr.astype(np.int16) + prev.astype(np.int16)) % 5).astype(np.uint8)


def prototype_reencode_hpm1_from_raw_tokens(
    raw_tokens: np.ndarray,
    source_payload: Hpm1MaskPayload,
    *,
    max_frames: int | None = None,
    probability_variant: str = DEFAULT_HPAC_PROBABILITY_VARIANT,
    device: str = "cpu",
) -> dict[str, Any]:
    raw = np.asarray(raw_tokens, dtype=np.uint8)
    if max_frames is not None:
        raw = raw[: int(max_frames)]
    return {
        "schema": "pr91_hpm1_raw_token_reencode_prototype_v1",
        "status": "blocked_hpac_encoder_not_rehydrated",
        "score_claim": False,
        "dispatch_allowed": False,
        "raw_tokens": {"shape": list(raw.shape), "sha256": sha256_bytes(raw.tobytes())},
        "source_payload": {"config": source_payload.config(), "tokens_sha256": sha256_bytes(source_payload.tokens)},
        "probability_variant": probability_variant,
        "device": device,
        "failure_reason": "encode_tokens_hpac is not rehydrated; prototype remains local-only.",
    }


def prototype_reencode_hpm1_residual_from_raw_tokens(
    raw_tokens: np.ndarray,
    source_payload: Hpm1MaskPayload,
    *,
    max_frames: int | None = None,
    probability_variant: str = DEFAULT_HPAC_PROBABILITY_VARIANT,
    device: str = "cpu",
) -> dict[str, Any]:
    residual, prev = raw_tokens_to_mod5_residual_symbols(raw_tokens)
    if max_frames is not None:
        residual = residual[: int(max_frames)]
        prev = prev[: int(max_frames)]
    reconstructed = reconstruct_raw_tokens_from_mod5_residual_symbols(residual, prev)
    return {
        "schema": "pr91_hpm1_residual_reencode_prototype_v1",
        "status": "blocked_hpac_encoder_not_rehydrated",
        "score_claim": False,
        "dispatch_allowed": False,
        "roundtrip_raw_tokens": bool(np.array_equal(reconstructed, np.asarray(raw_tokens, dtype=np.uint8)[: reconstructed.shape[0]])),
        "residual_symbols": {"shape": list(residual.shape), "sha256": sha256_bytes(residual.tobytes())},
        "prev_context": {"shape": list(prev.shape), "sha256": sha256_bytes(prev.tobytes())},
        "source_payload": {"config": source_payload.config(), "tokens_sha256": sha256_bytes(source_payload.tokens)},
        "probability_variant": probability_variant,
        "device": device,
        "failure_reason": "encode_symbols_hpac_with_prev_context is not rehydrated; prototype remains local-only.",
    }


def write_json_report(report: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")
