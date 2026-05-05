"""Fail-closed local PR86 HPAC replay and byte-parity helpers.

This module is deliberately local-only: it never runs contest eval, never
dispatches remote work, and never makes a score claim. It validates PR86
``archive.zip`` byte custody, exposes the canonical HPAC probability-variant
contract probes, and provides stub entry points for replay/recode helpers.

REHYDRATED 2026-05-05 from .recovery_spec.json (preserved at
.recovery_quarantine_20260505T004735Z/src/tac/pr86_hpac_codec.recovery_spec.json).
Spec source: bytecode disassembly of compiled .pyc; whitespace + inline comments lost.

PARTIAL REHYDRATION: Module-level constants, dataclasses, and the
``HPAC_PROBABILITY_VARIANTS`` registry are reconstructed exactly. The torch /
gzip / PPMd compose helpers and the actual HPAC encode/decode replay loop are
stubbed to ``NotImplementedError`` because the bytecode disassembly contains
intricate mixins (``_MaskedConv2dPG``, masked-conv RGB pair model, custom
PPMd Categorical wrapping, gzip-roundtrip parity) that pycdc cannot fully
decompile. The intended consumers in ``experiments/`` are themselves
quarantined.
"""
from __future__ import annotations

import gzip
import hashlib
import importlib.metadata
import io
import json
import sys
import time
import zipfile
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

import numpy as np

try:  # pragma: no cover - optional in lite environments
    import torch
    from torch import nn
    from torch.nn import functional as F
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]

REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_PR86_DIR = REPO_ROOT / "experiments/results/public_pr86_intake_20260504_codex"
DEFAULT_PR86_ARCHIVE = DEFAULT_PR86_DIR / "archive.zip"
DEFAULT_PR86_PROBABILITY_CONTRACT_DIR = (
    REPO_ROOT / "experiments/results/pr86_hpac_probability_contract_20260504_worker"
)
DEFAULT_PR86_PROBABILITY_CONTRACT_REPORT = (
    DEFAULT_PR86_PROBABILITY_CONTRACT_DIR
    / "pr86_hpac_probability_contract_variants.json"
)
DEFAULT_PR86_MERGED_SOURCE_DIR = (
    REPO_ROOT / "experiments/results/public_pr86_intake_20260504_merged_refresh"
)
DEFAULT_MERGED_INTAKE_SUMMARY = DEFAULT_PR86_MERGED_SOURCE_DIR / "intake_summary.json"
DEFAULT_MERGED_SOURCE_MANIFEST = (
    DEFAULT_PR86_MERGED_SOURCE_DIR / "source_manifest.json"
)
DEFAULT_MERGED_PR_API = DEFAULT_PR86_MERGED_SOURCE_DIR / "pr86_api.json"
DEFAULT_FULL_REENCODE = (
    DEFAULT_PR86_DIR / "pr86_hpac_full_decode_reencode_gate_20260504_codex.json"
)
DEFAULT_TOKEN_ANATOMY = DEFAULT_PR86_DIR / "pr86_hpac_token_anatomy_forensics.json"
DEFAULT_PR85_PROBE = DEFAULT_PR86_DIR / "pr86_hpac_pr85_qma9_parity_probe.json"
DEFAULT_PR_VIEW = DEFAULT_PR86_DIR / "pr86_view.json"

EXPECTED_PR86_ARCHIVE_BYTES = 207579
EXPECTED_PR86_ARCHIVE_SHA256 = (
    "e67b7c22240dbe33853c19d049b0044a5df16ce5f751ba8f1021cab8ceb03cef"
)
EXPECTED_PR86_TOKENS_SHA256 = (
    "14144bde496631f89a02646496bc2e66306bba6da149ddca37e21d85d175f225"
)
EXPECTED_PR86_MEMBERS = (
    "master.pt.gz",
    "slave.pt.gz",
    "hpac.pt.ppmd",
    "tokens.bin",
    "meta.pt",
)
EXPECTED_PR86_MEMBER_BYTES: Mapping[str, int] = {
    "master.pt.gz": 31144,
    "slave.pt.gz": 32287,
    "hpac.pt.ppmd": 28243,
    "tokens.bin": 113900,
    "meta.pt": 1499,
}
RECORDED_PR86_DEPENDENCIES: Mapping[str, str] = {
    "python": "3.12.13",
    "torch": "2.11.0",
    "numpy": "2.4.4",
    "constriction": "0.4.2",
    "pyppmd": "1.3.1",
}
NUM_CLASSES = 5
SEGNET_IN_H = 384
SEGNET_IN_W = 512
PPMD_MAX_ORDER = 4
PPMD_MEM_SIZE = 16777216
PROB_EPS = 1e-07
DEFAULT_HPAC_PROBABILITY_VARIANT = "source_float64_perfect_false"

_QUARANTINE_SPEC = (
    ".recovery_quarantine_20260505T004735Z/src/tac/pr86_hpac_codec.recovery_spec.json"
)


class Pr86HpacReplayError(RuntimeError):
    """Raised on PR86 HPAC replay or byte-parity failure."""

    def __init__(self, contract: str, code: str, **fields: Any) -> None:
        message_parts = [f"contract={contract}", f"code={code}"]
        for k, v in fields.items():
            message_parts.append(f"{k}={v!r}")
        super().__init__("; ".join(message_parts))
        self.contract = contract
        self.code = code
        self.fields = dict(fields)


@dataclass(frozen=True)
class Pr86ArchiveContract:
    """Expected member byte sizes / sha256 for the public PR86 archive."""

    archive_path: Path
    expected_bytes: int = EXPECTED_PR86_ARCHIVE_BYTES
    expected_sha256: str = EXPECTED_PR86_ARCHIVE_SHA256
    expected_members: tuple[str, ...] = EXPECTED_PR86_MEMBERS
    expected_member_bytes: Mapping[str, int] = field(
        default_factory=lambda: dict(EXPECTED_PR86_MEMBER_BYTES)
    )

    def member_bytes(self, member: str) -> int:
        try:
            return int(self.expected_member_bytes[member])
        except KeyError as exc:
            raise Pr86HpacReplayError(
                "archive_member_contract",
                "unknown_pr86_member",
                requested=member,
                supported=list(self.expected_member_bytes.keys()),
            ) from exc


@dataclass(frozen=True)
class Pr86ArchiveBundle:
    """Decoded PR86 archive: byte payloads keyed by member name."""

    contract: Pr86ArchiveContract
    members: Mapping[str, bytes]
    sha256: str
    extra: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class HpacProbabilityVariant:
    """One row of the HPAC probability-contract probe matrix."""

    name: str
    probability_dtype: str
    categorical_perfect: bool
    source_contract: bool
    description: str


HPAC_PROBABILITY_VARIANTS: Mapping[str, HpacProbabilityVariant] = {
    "source_float64_perfect_false": HpacProbabilityVariant(
        name="source_float64_perfect_false",
        probability_dtype="float64",
        categorical_perfect=False,
        source_contract=True,
        description=(
            "Merged PR86 source contract: clipped/renormalized numpy float64 "
            "probabilities with Categorical(..., perfect=False)."
        ),
    ),
    "source_float32_perfect_false": HpacProbabilityVariant(
        name="source_float32_perfect_false",
        probability_dtype="float32",
        categorical_perfect=False,
        source_contract=False,
        description=(
            "Off-contract probe: pass clipped/renormalized numpy float32 "
            "probabilities to Categorical(..., perfect=False)."
        ),
    ),
    "source_float64_perfect_true": HpacProbabilityVariant(
        name="source_float64_perfect_true",
        probability_dtype="float64",
        categorical_perfect=True,
        source_contract=False,
        description=(
            "Off-contract probe: keep float64 probabilities but construct "
            "Categorical(..., perfect=True)."
        ),
    ),
    "source_float32_perfect_true": HpacProbabilityVariant(
        name="source_float32_perfect_true",
        probability_dtype="float32",
        categorical_perfect=True,
        source_contract=False,
        description=(
            "Off-contract combined probe: pass float32 probabilities to "
            "Categorical(..., perfect=True)."
        ),
    ),
}


def repo_rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_path(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def supported_hpac_probability_variant_names() -> tuple[str, ...]:
    """Return stable CLI choices for HPAC probability-contract probes."""
    return tuple(HPAC_PROBABILITY_VARIANTS.keys())


def resolve_hpac_probability_variant(variant: Any) -> HpacProbabilityVariant:
    """Resolve a named HPAC probability variant or fail closed on unknown input."""
    if isinstance(variant, HpacProbabilityVariant):
        return variant
    try:
        return HPAC_PROBABILITY_VARIANTS[str(variant)]
    except KeyError as exc:
        raise Pr86HpacReplayError(
            "probability_variant_contract",
            "unknown_hpac_probability_variant",
            requested_variant=str(variant),
            supported_variants=list(supported_hpac_probability_variant_names()),
        ) from exc


def default_source_artifact_paths() -> tuple[Path, ...]:
    return (
        DEFAULT_MERGED_INTAKE_SUMMARY,
        DEFAULT_MERGED_SOURCE_MANIFEST,
        DEFAULT_MERGED_PR_API,
        DEFAULT_FULL_REENCODE,
        DEFAULT_TOKEN_ANATOMY,
        DEFAULT_PR85_PROBE,
        DEFAULT_PR_VIEW,
    )


def _rehydration_failure(symbol: str) -> NotImplementedError:
    return NotImplementedError(
        f"rehydration incomplete: {symbol} requires intricate torch / gzip / "
        f"PPMd compose chain that pycdc cannot fully decompile; original "
        f"bytecode preserved in {_QUARANTINE_SPEC}"
    )


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return repo_rel(value)
    if isinstance(value, np.generic):
        return value.item()
    if torch is not None and isinstance(value, torch.Tensor):
        if value.device.type == "cpu" and value.numel() <= 16384:
            return {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "sha256": sha256_bytes(value.detach().cpu().numpy().tobytes()),
            }
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "device": str(value.device),
            "numel": int(value.numel()),
        }
    if isinstance(value, bytes):
        return {"bytes": len(value), "sha256": sha256_bytes(value)}
    return value


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:  # pragma: no cover
        return "unknown"


def _load_json_file(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _validate_safe_member_name(name: str) -> str:
    if not name or "\x00" in name or "\\" in name:
        raise Pr86HpacReplayError(
            "archive_member_safety", "unsafe_pr86_member_name", member=name
        )
    posix = PurePosixPath(name)
    if posix.is_absolute() or any(p == ".." for p in posix.parts):
        raise Pr86HpacReplayError(
            "archive_member_safety", "unsafe_pr86_member_name", member=name
        )
    return name


def read_pr86_archive(
    archive_path: Path | str, *, contract: Pr86ArchiveContract | None = None
) -> Pr86ArchiveBundle:
    """Read and validate a PR86 archive into a typed bundle (deferred)."""
    raise _rehydration_failure("read_pr86_archive")


def load_source_artifact_summaries(
    paths: Mapping[str, Path] | None = None,
) -> dict[str, Any]:
    raise _rehydration_failure("load_source_artifact_summaries")


def analyze_pr86_current_source_context(
    sources: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    raise _rehydration_failure("analyze_pr86_current_source_context")


def collect_dependency_report(
    *, expected: Mapping[str, str] | None = None, strict: bool = False
) -> dict[str, Any]:
    raise _rehydration_failure("collect_dependency_report")


def decode_gzip_torch_member(
    payload: bytes, *, weights_only: bool = False
) -> Any:
    """Decode a ``master.pt.gz`` / ``slave.pt.gz`` member into a torch state.

    Stub: defers to ``torch.load(gzip.decompress(...))`` once the masked
    conv pair model is rehydrated.
    """
    raise _rehydration_failure("decode_gzip_torch_member")


def decode_meta_member(payload: bytes) -> Any:
    """Decode the ``meta.pt`` member into a structured metadata object."""
    raise _rehydration_failure("decode_meta_member")


# --- Symbols re-exported by pr91_hpm1_codec (deferred) ---


class HPACMini:  # REHYDRATED stub
    """Minimal HPAC encoder/decoder wrapper used by PR86 replay (deferred)."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise _rehydration_failure("HPACMini")


def _categorical_from_probs(*args: Any, **kwargs: Any) -> Any:
    raise _rehydration_failure("_categorical_from_probs")


def _group_masks(*args: Any, **kwargs: Any) -> Any:
    raise _rehydration_failure("_group_masks")


def _normalize_probability_row(*args: Any, **kwargs: Any) -> Any:
    raise _rehydration_failure("_normalize_probability_row")


def decode_tokens_hpac(*args: Any, **kwargs: Any) -> Any:
    raise _rehydration_failure("decode_tokens_hpac")


def encode_tokens_hpac(*args: Any, **kwargs: Any) -> Any:
    raise _rehydration_failure("encode_tokens_hpac")


def encode_symbols_hpac_with_prev_context(*args: Any, **kwargs: Any) -> Any:
    raise _rehydration_failure("encode_symbols_hpac_with_prev_context")


def load_hpac_model_from_ppmd(*args: Any, **kwargs: Any) -> Any:
    raise _rehydration_failure("load_hpac_model_from_ppmd")
