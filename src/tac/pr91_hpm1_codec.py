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
and ``Hpm1MaskPayload`` dataclass are reconstructed exactly. The probability
variant probe matrix entry point ``run_pr91_hpm1_probability_variant_matrix``
is exposed but defers to ``NotImplementedError`` because the bytecode
disassembly contains intricate torch / HPAC compose chains that pycdc
cannot fully decompile. Production replay path is the upstream PR91
``replay_submission`` directory.
"""
from __future__ import annotations

import hashlib
import json
import time
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
    raise _rehydration_failure("_validate_dependency_report")


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
    raise _rehydration_failure("analyze_pr91_hpm1_runtime_sources")


def compare_hpm1_to_pr86_hpac_contract(
    archive: Path, *, reference: Path | None = None
) -> dict[str, Any]:
    raise _rehydration_failure("compare_hpm1_to_pr86_hpac_contract")


def validate_hpm1_static_contract(archive: Path) -> dict[str, Any]:
    raise _rehydration_failure("validate_hpm1_static_contract")


def load_hpm1_hpac_model(payload: Hpm1MaskPayload, *, device: str = "cpu") -> Any:
    raise _rehydration_failure("load_hpm1_hpac_model")


def run_pr91_hpm1_preflight(
    archive: Path,
    *,
    output_dir: Path | None = None,
    strict: bool = False,
    summary: bool = True,
    write_json: bool = True,
) -> dict[str, Any]:
    raise _rehydration_failure("run_pr91_hpm1_preflight")


def run_pr91_hpm1_probability_variant_matrix(
    archive: Path,
    *,
    output_dir: Path | None = None,
    variants: tuple[str, ...] | None = None,
    strict: bool = False,
    summary: bool = True,
    write_json: bool = True,
) -> dict[str, Any]:
    """Run the PR91 HPM1 probability-variant probe matrix.

    DEFERRED: requires the full HPAC encode/decode replay loop from
    ``tac.pr86_hpac_codec`` which is itself rehydrated as a stub.
    """
    raise _rehydration_failure("run_pr91_hpm1_probability_variant_matrix")


def _hpm1_token_stream_transform_candidates(*args: Any, **kwargs: Any) -> Any:
    raise _rehydration_failure("_hpm1_token_stream_transform_candidates")


def run_pr91_hpm1_stream_transform_probe(*args: Any, **kwargs: Any) -> Any:
    raise _rehydration_failure("run_pr91_hpm1_stream_transform_probe")


def _load_reference_tokens(*args: Any, **kwargs: Any) -> Any:
    raise _rehydration_failure("_load_reference_tokens")


def _symbol_position(*args: Any, **kwargs: Any) -> Any:
    raise _rehydration_failure("_symbol_position")
