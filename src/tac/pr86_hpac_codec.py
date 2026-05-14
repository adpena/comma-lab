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
    import constriction
except ImportError:  # pragma: no cover
    constriction = None  # type: ignore[assignment]

try:  # pragma: no cover - optional in lite environments
    import pyppmd  # PYPPMD_LGPL_OK:public-PR86-archive-replay-decode-only-no-permissive-PPMd-binding-on-PyPI
except ImportError:  # pragma: no cover
    # pyppmd is LGPL-2.1-or-later and is now an OPTIONAL dep (pip install
    # tac[pr86_replay]) per OSS v0.2.0-rc1 BLOCKER B1 (lane
    # lane_pyppmd_to_constriction_migrate_20260514). Default `pip install
    # tac` does NOT pull pyppmd, so this ImportError is the EXPECTED path
    # outside the PR86/PR91 third-party-archive-replay forensic flow. The
    # PR86 codec sets `pyppmd = None` here and `_require_pyppmd()` raises
    # Pr86HpacReplayError("dependency_contract", "pyppmd_missing") at the
    # call site for any code path that actually needs to decode an
    # hpac.pt.ppmd wire-format member. Per CLAUDE.md "MPS auth eval is
    # NOISE" / "Apples-to-apples evidence discipline" tag-discipline: the
    # bytes pyppmd decodes are public-PR-author-emitted wire bytes; we have
    # no permissive replacement (constriction lacks PPMd context modeling;
    # all PyPI PPMd bindings are LGPL).
    pyppmd = None  # type: ignore[assignment]

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
DEFAULT_PR86_EXACT_EVAL_LOGS = (
    REPO_ROOT
    / "experiments/results/lightning_batch/exact_eval_public_pr86_hpac_t4_hedge_20260504T0152Z/auth_eval.log",
    REPO_ROOT
    / "experiments/results/lightning_batch/exact_eval_public_pr86_hpac_t4_retry1_20260504T0213Z/auth_eval.log",
)
DEFAULT_PR86_RELEASE_SOURCE_ROOT = (
    REPO_ROOT
    / "experiments/results/public_pr_archive_release_view/public_pr86_intake_20260505_auto/source/submissions/jas0xf_adversarial_neural_representation"
)
DEFAULT_PR86_TRAINING_ARCHIVE_SOURCE = DEFAULT_PR86_RELEASE_SOURCE_ROOT / "training/archive.py"
DEFAULT_PR86_INFLATE_SOURCE = DEFAULT_PR86_RELEASE_SOURCE_ROOT / "inflate.py"

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
    """Read and validate a PR86 archive into a typed, zip-safe bundle."""

    archive = Path(archive_path)
    resolved_contract = contract or Pr86ArchiveContract(archive)
    if not archive.is_file():
        raise Pr86HpacReplayError(
            "archive_contract", "pr86_archive_missing", archive=archive
        )
    archive_bytes = archive.stat().st_size
    archive_sha = sha256_path(archive)
    if archive_bytes != int(resolved_contract.expected_bytes):
        raise Pr86HpacReplayError(
            "archive_contract",
            "pr86_archive_bytes_mismatch",
            archive=archive,
            expected=resolved_contract.expected_bytes,
            got=archive_bytes,
        )
    if archive_sha != resolved_contract.expected_sha256:
        raise Pr86HpacReplayError(
            "archive_contract",
            "pr86_archive_sha256_mismatch",
            archive=archive,
            expected=resolved_contract.expected_sha256,
            got=archive_sha,
        )
    members: dict[str, bytes] = {}
    member_reports: dict[str, Any] = {}
    with zipfile.ZipFile(archive, "r") as zf:
        infos = zf.infolist()
        names = [_validate_safe_member_name(info.filename) for info in infos]
        duplicate_names = [name for name, count in Counter(names).items() if count > 1]
        if duplicate_names:
            raise Pr86HpacReplayError(
                "archive_member_contract",
                "duplicate_pr86_members",
                duplicates=duplicate_names,
            )
        if tuple(names) != tuple(resolved_contract.expected_members):
            raise Pr86HpacReplayError(
                "archive_member_contract",
                "pr86_archive_member_order_mismatch",
                expected=list(resolved_contract.expected_members),
                got=names,
            )
        for info in infos:
            payload = zf.read(info.filename)
            expected_size = resolved_contract.member_bytes(info.filename)
            if len(payload) != expected_size or int(info.file_size) != expected_size:
                raise Pr86HpacReplayError(
                    "archive_member_contract",
                    "pr86_member_bytes_mismatch",
                    member=info.filename,
                    expected=expected_size,
                    got=len(payload),
                    zip_file_size=int(info.file_size),
                )
            members[info.filename] = payload
            member_reports[info.filename] = {
                "bytes": len(payload),
                "zip_file_size": int(info.file_size),
                "zip_compress_size": int(info.compress_size),
                "sha256": sha256_bytes(payload),
            }
    return Pr86ArchiveBundle(
        contract=resolved_contract,
        members=members,
        sha256=archive_sha,
        extra={
            "archive_bytes": archive_bytes,
            "member_reports": member_reports,
        },
    )


def load_source_artifact_summaries(
    paths: Mapping[str, Path] | None = None,
) -> dict[str, Any]:
    requested = dict(paths or {path.name: path for path in default_source_artifact_paths()})
    rows: dict[str, Any] = {}
    for name, path_like in requested.items():
        path = Path(path_like)
        row: dict[str, Any] = {"path": repo_rel(path), "exists": path.is_file()}
        if path.is_file():
            data = path.read_bytes()
            row.update({"bytes": len(data), "sha256": sha256_bytes(data)})
            if path.suffix == ".json":
                try:
                    row["json"] = json.loads(data.decode("utf-8"))
                    row["json_parse_status"] = "passed"
                except Exception as exc:
                    row["json_parse_status"] = "failed"
                    row["json_parse_error"] = type(exc).__name__
        rows[str(name)] = row
    return {"status": "passed_source_artifact_inventory", "artifacts": rows}


def analyze_pr86_current_source_context(
    sources: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if isinstance(sources, (str, Path)):
        source_root = Path(sources)
        files = []
        if source_root.is_dir():
            for path in sorted(source_root.rglob("*")):
                if path.is_file():
                    data = path.read_bytes()
                    files.append(
                        {
                            "path": path.relative_to(source_root).as_posix(),
                            "bytes": len(data),
                            "sha256": sha256_bytes(data),
                        }
                    )
        return {
            "status": "passed_static_pr86_source_context" if files else "missing_source_context",
            "score_claim": False,
            "source_root": repo_rel(source_root),
            "file_count": len(files),
            "files": files,
        }
    artifact_report = dict(sources or load_source_artifact_summaries())
    artifacts = artifact_report.get("artifacts", {})
    present = [name for name, row in artifacts.items() if row.get("exists")]
    missing = [name for name, row in artifacts.items() if not row.get("exists")]
    return {
        "status": "passed_static_pr86_source_context",
        "score_claim": False,
        "artifact_count": len(artifacts),
        "present_artifacts": present,
        "missing_artifacts": missing,
        "artifact_report": artifact_report,
    }


def collect_dependency_report(
    *, expected: Mapping[str, str] | None = None, strict: bool = False
) -> dict[str, Any]:
    expected_versions = dict(expected or RECORDED_PR86_DEPENDENCIES)
    observed = {
        "python": ".".join(map(str, sys.version_info[:3])),
        "torch": getattr(torch, "__version__", "missing") if torch is not None else "missing",
        "numpy": np.__version__,
        "constriction": _package_version("constriction"),
        "pyppmd": _package_version("pyppmd"),
    }
    mismatches = {
        name: {"expected": expected_version, "observed": observed.get(name, "missing")}
        for name, expected_version in expected_versions.items()
        if observed.get(name, "missing") != expected_version
    }
    missing = [name for name, value in observed.items() if value in ("missing", "unknown")]
    report = {
        "status": "passed" if not mismatches and not missing else "mismatch",
        "expected": expected_versions,
        "observed": observed,
        "mismatches": mismatches,
        "missing": missing,
        "strict": bool(strict),
    }
    if strict and (mismatches or missing):
        raise Pr86HpacReplayError(
            "dependency_contract",
            "pr86_dependency_mismatch",
            mismatches=mismatches,
            missing=missing,
        )
    return report


def decode_gzip_torch_member(
    payload: bytes, *, weights_only: bool = False
) -> Any:
    """Decode a ``master.pt.gz`` / ``slave.pt.gz`` member into a torch state.

    Stub: defers to ``torch.load(gzip.decompress(...))`` once the masked
    conv pair model is rehydrated.
    """
    _require_torch()
    try:
        raw = gzip.decompress(payload)
    except Exception as exc:
        raise Pr86HpacReplayError(
            "torch_member_contract", "gzip_decompress_failed"
        ) from exc
    return torch.load(io.BytesIO(raw), map_location="cpu", weights_only=weights_only)


def decode_meta_member(payload: bytes) -> Any:
    """Decode the ``meta.pt`` member into a structured metadata object."""
    _require_torch()
    meta = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=False)
    if not isinstance(meta, Mapping):
        raise Pr86HpacReplayError(
            "meta_member_contract",
            "expected_meta_mapping",
            loaded_type=type(meta).__name__,
        )
    return dict(meta)


# --- Symbols re-exported by pr91_hpm1_codec ---


def _require_torch() -> None:
    if torch is None or nn is None or F is None:  # pragma: no cover
        raise Pr86HpacReplayError("dependency_contract", "torch_missing")


def _require_constriction() -> None:
    if constriction is None:  # pragma: no cover
        raise Pr86HpacReplayError("dependency_contract", "constriction_missing")


def _require_pyppmd() -> None:
    if pyppmd is None:  # pragma: no cover
        raise Pr86HpacReplayError("dependency_contract", "pyppmd_missing")


def _patch_group_mask(k: int, delta: int, type_: str) -> Any:
    _require_torch()
    mask = torch.zeros(k, k, dtype=torch.float32)
    center = (k - 1) // 2
    for dr_idx in range(k):
        for dc_idx in range(k):
            dr = dr_idx - center
            dc = dc_idx - center
            val = dc + int(delta) * dr
            if type_ == "A":
                if val < 0:
                    mask[dr_idx, dc_idx] = 1.0
            elif type_ == "B":
                if val <= 0:
                    mask[dr_idx, dc_idx] = 1.0
            else:
                raise Pr86HpacReplayError(
                    "masked_conv_contract", "unknown_patch_mask_type", type=type_
                )
    return mask


class _MaskedConv2dPG(nn.Module if nn is not None else object):  # type: ignore[misc]
    """Plain masked conv from the PR86/PR91 public inflate source."""

    def __init__(
        self,
        c_in: int,
        c_out: int,
        k: int,
        *,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        type_: str = "B",
        delta: int = 2,
        bias: bool = True,
    ) -> None:
        _require_torch()
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(c_out, c_in // groups, k, k))
        self.bias = nn.Parameter(torch.zeros(c_out)) if bias else None
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        mask = _patch_group_mask(k, delta, type_)
        self.register_buffer("mask", mask.view(1, 1, k, k), persistent=False)

    def forward(self, x: Any) -> Any:
        return F.conv2d(
            x,
            self.weight * self.mask,
            self.bias,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )


class _ChannelNorm2d(nn.Module if nn is not None else object):  # type: ignore[misc]
    def __init__(self, num_channels: int, eps: float = 1e-5) -> None:
        _require_torch()
        super().__init__()
        self.scale = nn.Parameter(torch.ones(num_channels))
        self.shift = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: Any) -> Any:
        mu = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True, unbiased=False)
        x = (x - mu) / torch.sqrt(var + self.eps)
        return x * self.scale.view(1, -1, 1, 1) + self.shift.view(1, -1, 1, 1)


class _CausalSPM(nn.Module if nn is not None else object):  # type: ignore[misc]
    """Decode-time causal spatial prior module from the public HPAC source."""

    def __init__(self, ch: int, P: int = 32) -> None:
        _require_torch()
        super().__init__()
        self.P = P
        self.norm = _ChannelNorm2d(ch)
        self.dw = nn.Conv2d(ch, ch, kernel_size=3, padding=1, groups=ch)
        self.pw = nn.Conv2d(ch, ch, kernel_size=1)

    def forward(self, h_past: Any) -> Any:
        B, C, H, W = h_past.shape
        P = self.P
        NRp, NCp = H // P, W // P
        x_p = h_past.view(B, C, NRp, P, NCp, P).mean(dim=(3, 5))
        x_p = self.norm(x_p)
        x_p = self.dw(x_p)
        x_p = F.gelu(x_p)
        x_p = self.pw(x_p)
        x_full = (
            x_p.unsqueeze(3)
            .unsqueeze(5)
            .expand(B, C, NRp, P, NCp, P)
            .contiguous()
        )
        return x_full.view(B, C, NRp * P, NCp * P)


class HPACMini(nn.Module if nn is not None else object):  # type: ignore[misc]
    """Minimal HPAC token probability model used by PR86/PR91 replay."""

    def __init__(
        self,
        num_pairs: int = 600,
        num_classes: int = NUM_CLASSES,
        P: int = 32,
        delta: int = 2,
        d_film: int = 32,
        ch: int = 64,
        use_spm: bool = False,
    ) -> None:
        _require_torch()
        super().__init__()
        self.num_classes = num_classes
        self.P = int(P)
        self.delta = int(delta)
        self.ch = int(ch)
        self.use_spm = bool(use_spm)
        self.frame_embed = nn.Embedding(int(num_pairs), int(d_film))
        self.film_gen = nn.Linear(int(d_film), int(ch) * 2)
        self.conv_a = _MaskedConv2dPG(
            int(num_classes) + 2,
            int(ch),
            7,
            padding=3,
            type_="A",
            delta=int(delta),
        )
        self.gn_a = _ChannelNorm2d(int(ch))
        self.conv_b1 = _MaskedConv2dPG(
            int(ch),
            int(ch),
            5,
            padding=4,
            dilation=2,
            groups=int(ch),
            type_="B",
            delta=int(delta),
        )
        self.gn_b1 = _ChannelNorm2d(int(ch))
        self.conv_b2 = _MaskedConv2dPG(
            int(ch),
            int(ch),
            3,
            padding=4,
            dilation=4,
            groups=int(ch),
            type_="B",
            delta=int(delta),
        )
        self.gn_b2 = _ChannelNorm2d(int(ch))
        self.conv_past = nn.Conv2d(int(num_classes), int(ch), kernel_size=3, padding=1)
        self.spm = _CausalSPM(int(ch), P=int(P)) if use_spm else None
        self.head = nn.Conv2d(int(ch), int(num_classes), kernel_size=1, padding=0)
        self.register_buffer("_coord_cache", torch.zeros(0), persistent=False)
        self._cached_P = -1

    def _patch_coord_grid(self, B: int, device: Any) -> Any:
        if self._cached_P != self.P or self._coord_cache.numel() == 0:
            P = self.P
            ys = (
                torch.linspace(-1.0, 1.0, P, device=device)
                .view(1, 1, P, 1)
                .expand(1, 1, P, P)
            )
            xs = (
                torch.linspace(-1.0, 1.0, P, device=device)
                .view(1, 1, 1, P)
                .expand(1, 1, P, P)
            )
            self._coord_cache = torch.cat([ys, xs], dim=1)
            self._cached_P = self.P
        return self._coord_cache.expand(B, -1, -1, -1)

    def _to_patches(self, x: Any) -> Any:
        B, C, H, W = x.shape
        P = self.P
        NRp, NCp = H // P, W // P
        x = x.view(B, C, NRp, P, NCp, P).permute(0, 2, 4, 1, 3, 5).contiguous()
        return x.view(B * NRp * NCp, C, P, P)

    def _from_patches(self, x_p: Any, B: int, NRp: int, NCp: int) -> Any:
        P = self.P
        C = x_p.shape[1]
        x_p = (
            x_p.view(B, NRp, NCp, C, P, P)
            .permute(0, 3, 1, 4, 2, 5)
            .contiguous()
        )
        return x_p.view(B, C, NRp * P, NCp * P)

    def forward(self, tokens: Any, idx: Any, prev_tokens: Any) -> Any:
        B, H, W = tokens.shape
        P = self.P
        NRp, NCp = H // P, W // P
        Np = NRp * NCp
        x = F.one_hot(tokens, num_classes=self.num_classes).permute(0, 3, 1, 2).float()
        x_p = self._to_patches(x)
        coord_p = self._patch_coord_grid(B * Np, x.device)
        x_in_p = torch.cat([x_p, coord_p], dim=1)
        h_p = self.gn_a(self.conv_a(x_in_p))
        emb = self.frame_embed(idx)
        film = self.film_gen(emb)
        scale, shift = film.chunk(2, dim=1)
        scale_p = (
            scale.view(B, 1, self.ch, 1, 1)
            .expand(B, Np, self.ch, 1, 1)
            .reshape(B * Np, self.ch, 1, 1)
        )
        shift_p = (
            shift.view(B, 1, self.ch, 1, 1)
            .expand(B, Np, self.ch, 1, 1)
            .reshape(B * Np, self.ch, 1, 1)
        )
        h_p = h_p * (1.0 + scale_p) + shift_p
        h_p = F.gelu(h_p)
        x_prev = (
            F.one_hot(prev_tokens, num_classes=self.num_classes)
            .permute(0, 3, 1, 2)
            .float()
        )
        h_past_full = self.conv_past(x_prev)
        h_p = h_p + self._to_patches(h_past_full)
        if self.spm is not None:
            h_p = h_p + self._to_patches(self.spm(h_past_full))
        h_p = F.gelu(self.gn_b1(self.conv_b1(h_p)))
        h_p = F.gelu(self.gn_b2(self.conv_b2(h_p)))
        logits_p = self.head(h_p)
        return self._from_patches(logits_p, B, NRp, NCp)


def _reconstruct_hpac_state_dict(packed_sd: Mapping[str, Any], device: str) -> dict[str, Any]:
    _require_torch()
    out: dict[str, Any] = {}
    bases = sorted({k[: -len(".weight_q")] for k in packed_sd if k.endswith(".weight_q")})
    for base in bases:
        q = packed_sd[base + ".weight_q"].to(device).float()
        scale = packed_sd[base + ".weight_scale"].to(device).float()
        shape = [1] * q.ndim
        shape[0] = -1
        out[base + ".weight"] = (q * scale.view(*shape)).to(torch.float32)
    skip = {f"{base}.weight_q" for base in bases} | {
        f"{base}.weight_scale" for base in bases
    }
    for key, value in packed_sd.items():
        if key in skip:
            continue
        if torch.is_tensor(value):
            out[key] = value.to(device).float() if torch.is_floating_point(value) else value.to(device)
        else:
            out[key] = value
    return out


def _normalize_probability_row(
    probs: Any,
    *,
    prob_eps: float = PROB_EPS,
    variant: str | HpacProbabilityVariant = DEFAULT_HPAC_PROBABILITY_VARIANT,
) -> np.ndarray:
    resolved = resolve_hpac_probability_variant(variant)
    dtype = np.float32 if resolved.probability_dtype == "float32" else np.float64
    arr = np.asarray(probs).astype(dtype, copy=False)
    if arr.ndim != 1 or arr.size != NUM_CLASSES:
        raise Pr86HpacReplayError(
            "probability_row_contract",
            "expected_single_num_classes_row",
            shape=list(arr.shape),
            expected_classes=NUM_CLASSES,
        )
    if not np.all(np.isfinite(arr)):
        raise Pr86HpacReplayError("probability_row_contract", "nonfinite_probability")
    arr = np.clip(arr, dtype(prob_eps), dtype(1.0))
    denom = float(arr.sum())
    if denom <= 0.0:
        raise Pr86HpacReplayError("probability_row_contract", "zero_probability_mass")
    arr = arr / denom
    return arr.astype(dtype, copy=False)


def _categorical_from_probs(
    probs: Any,
    *,
    prob_eps: float = PROB_EPS,
    variant: str | HpacProbabilityVariant = DEFAULT_HPAC_PROBABILITY_VARIANT,
) -> Any:
    _require_constriction()
    resolved = resolve_hpac_probability_variant(variant)
    row = _normalize_probability_row(probs, prob_eps=prob_eps, variant=resolved)
    return constriction.stream.model.Categorical(
        probabilities=row,
        perfect=bool(resolved.categorical_perfect),
    )


def _group_masks(H: int, W: int, *, P: int, delta: int, device: str | Any = "cpu") -> list[Any | None]:
    _require_torch()
    if H <= 0 or W <= 0 or P <= 0:
        raise Pr86HpacReplayError(
            "group_mask_contract", "nonpositive_geometry", H=H, W=W, P=P
        )
    if H % P or W % P:
        raise Pr86HpacReplayError(
            "group_mask_contract", "geometry_not_divisible_by_patch", H=H, W=W, P=P
        )
    if delta < 0:
        raise Pr86HpacReplayError("group_mask_contract", "negative_delta", delta=delta)
    NRp, NCp = H // P, W // P
    rs = torch.arange(P, device=device).view(P, 1).expand(P, P)
    cs = torch.arange(P, device=device).view(1, P).expand(P, P)
    s_grid = cs + int(delta) * rs
    n_groups = int((1 + int(delta)) * P - int(delta))
    masks: list[Any | None] = []
    for s in range(n_groups):
        mp = s_grid == s
        if not bool(mp.any()):
            masks.append(None)
            continue
        full = (
            mp.unsqueeze(0)
            .unsqueeze(0)
            .expand(NRp, NCp, P, P)
            .permute(0, 2, 1, 3)
            .reshape(NRp * P, NCp * P)
        )
        masks.append(full)
    return masks


def decode_tokens_hpac(
    gen: HPACMini,
    token_blob: bytes,
    N: int,
    H: int,
    W: int,
    P: int,
    delta: int,
    *,
    device: str = "cpu",
    prob_eps: float = PROB_EPS,
    probability_variant: str | HpacProbabilityVariant = DEFAULT_HPAC_PROBABILITY_VARIANT,
    max_frames: int | None = None,
    return_report: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, Any]]:
    """Decode HPAC arithmetic-coded token frames with the public PR86 loop.

    PR86/PR91 source-level evidence points at CPU replay as the deterministic
    entropy contract.  GPU devices fail closed here instead of generating
    misleading prefix evidence.
    """

    _require_torch()
    _require_constriction()
    if str(device) != "cpu":
        raise Pr86HpacReplayError(
            "device_contract",
            "pr86_hpac_replay_is_cpu_only",
            requested_device=device,
        )
    if len(token_blob) % 4:
        raise Pr86HpacReplayError(
            "tokens_bin_contract",
            "tokens_bin_not_uint32_aligned",
            tokens_bytes=len(token_blob),
        )
    resolved = resolve_hpac_probability_variant(probability_variant)
    frame_count = int(N if max_frames is None else min(int(max_frames), int(N)))
    if frame_count < 0:
        raise Pr86HpacReplayError("decode_contract", "negative_max_frames", max_frames=max_frames)
    dev = torch.device(device)
    gen = gen.to(dev).eval()
    masks = _group_masks(int(H), int(W), P=int(P), delta=int(delta), device=dev)
    words = np.frombuffer(token_blob, dtype="<u4").astype(np.uint32, copy=False)
    decoder = constriction.stream.queue.RangeDecoder(words)
    tokens = np.empty((frame_count, int(H), int(W)), dtype=np.uint8)
    decoded_prev = torch.zeros((1, int(H), int(W)), dtype=torch.long, device=dev)
    decoded_symbols = 0
    started_at = time.time()
    with torch.no_grad():
        for frame in range(frame_count):
            idx = torch.tensor([frame], dtype=torch.long, device=dev)
            cur = torch.zeros((1, int(H), int(W)), dtype=torch.long, device=dev)
            frame_start_symbols = decoded_symbols
            for group, mask in enumerate(masks):
                if mask is None:
                    continue
                group_start_symbols = decoded_symbols
                logits = gen(cur, idx, decoded_prev)
                probs = F.softmax(logits.float(), dim=1)
                probs_at_group = probs[0][:, mask].permute(1, 0).contiguous()
                decoded = np.empty(int(probs_at_group.shape[0]), dtype=np.int64)
                probs_np = probs_at_group.cpu().numpy()
                for symbol_in_group, row in enumerate(probs_np):
                    cat = _categorical_from_probs(row, prob_eps=prob_eps, variant=resolved)
                    try:
                        decoded[symbol_in_group] = decoder.decode(cat)
                    except Exception as exc:
                        raise Pr86HpacReplayError(
                            "submitted_tokens_decode",
                            "hpac_entropy_decode_contract_mismatch",
                            frame=frame,
                            group=group,
                            symbol_in_group=symbol_in_group,
                            decoded_symbol_count_before_failure=decoded_symbols,
                            group_start_decoded_symbols=group_start_symbols,
                            frame_start_decoded_symbols=frame_start_symbols,
                            probability_variant=resolved.name,
                        ) from exc
                    decoded_symbols += 1
                cur[0, mask] = torch.from_numpy(decoded).to(dev)
            tokens[frame] = cur[0].cpu().numpy().astype(np.uint8)
            decoded_prev = cur.clone()
    report = {
        "status": "passed_prefix_decode" if max_frames is not None else "passed_decode",
        "score_claim": False,
        "device": str(device),
        "probability_variant": resolved.name,
        "prob_eps": prob_eps,
        "decoded_frames": int(tokens.shape[0]),
        "decoded_symbols": decoded_symbols,
        "elapsed_sec": round(time.time() - started_at, 3),
        "tokens": {"shape": list(tokens.shape), "sha256": sha256_bytes(tokens.tobytes())},
    }
    return (tokens, report) if return_report else tokens


def _decode_tokens_hpac_legacy(
    blob: bytes,
    *,
    model: HPACMini,
    N: int,
    H: int,
    W: int,
    P: int,
    delta: int,
    device: str = "cpu",
    probability_variant: str = DEFAULT_HPAC_PROBABILITY_VARIANT,
    max_frames: int | None = None,
) -> np.ndarray:
    """Compatibility wrapper for early recovery tests; new code uses source signature."""

    return decode_tokens_hpac(
        model,
        blob,
        N,
        H,
        W,
        P,
        delta,
        device=device,
        probability_variant=probability_variant,
        max_frames=max_frames,
    )


def encode_tokens_hpac(*args: Any, **kwargs: Any) -> Any:
    raise _rehydration_failure("encode_tokens_hpac")


def encode_symbols_hpac_with_prev_context(*args: Any, **kwargs: Any) -> Any:
    raise _rehydration_failure("encode_symbols_hpac_with_prev_context")


def _load_packed_hpac_state(payload: bytes | Path | str) -> Mapping[str, Any]:
    _require_torch()
    if isinstance(payload, (str, Path)):
        path = Path(payload)
        raw = path.read_bytes()
        if path.suffix == ".ppmd":
            _require_pyppmd()
            raw = pyppmd.decompress(raw, max_order=PPMD_MAX_ORDER, mem_size=PPMD_MEM_SIZE)
        elif path.suffix == ".gz":
            raw = gzip.decompress(raw)
    else:
        _require_pyppmd()
        raw = pyppmd.decompress(bytes(payload), max_order=PPMD_MAX_ORDER, mem_size=PPMD_MEM_SIZE)
    loaded = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=False)
    if not isinstance(loaded, Mapping):
        raise Pr86HpacReplayError(
            "hpac_state_contract",
            "expected_state_dict_mapping",
            loaded_type=type(loaded).__name__,
        )
    return loaded


def summarize_hpac_packed_state_schema(payload: bytes | Path | str) -> dict[str, Any]:
    """Summarize HPAC packed-state keys without treating it as score evidence."""

    packed_sd = _load_packed_hpac_state(payload)
    keys = list(packed_sd.keys())
    suffix_counts = {
        ".weight_q": sum(key.endswith(".weight_q") for key in keys),
        ".weight_scale": sum(key.endswith(".weight_scale") for key in keys),
        ".weight": sum(key.endswith(".weight") for key in keys),
        ".b": sum(key.endswith(".b") for key in keys),
        ".e": sum(key.endswith(".e") for key in keys),
        ".bias": sum(key.endswith(".bias") for key in keys),
    }
    packed_bases = sorted(
        key[: -len(".weight_q")]
        for key in keys
        if key.endswith(".weight_q")
    )
    missing_scales = [
        base for base in packed_bases if f"{base}.weight_scale" not in packed_sd
    ]
    tensor_rows = []
    for key in keys:
        value = packed_sd[key]
        if torch is not None and torch.is_tensor(value):
            tensor_rows.append(
                {
                    "key": key,
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                    "numel": int(value.numel()),
                }
            )
        else:
            tensor_rows.append({"key": key, "type": type(value).__name__})
    raw_weight_keys = [
        key for key in keys
        if key.endswith(".weight") and not key.endswith(".weight_q")
    ]
    return _jsonable(
        {
            "schema": "hpac_packed_state_schema_v1",
            "score_claim": False,
            "key_count": len(keys),
            "suffix_counts": suffix_counts,
            "packed_weight_bases": packed_bases,
            "missing_weight_scales": missing_scales,
            "raw_weight_keys": raw_weight_keys,
            "has_packed_weights": bool(packed_bases),
            "has_raw_float_weights": bool(raw_weight_keys),
            "has_scn_runtime_parameters": bool(suffix_counts[".b"] or suffix_counts[".e"]),
            "keys_preview": keys[:40],
            "tensors": tensor_rows,
        }
    )


def _source_pattern_report(path: Path, patterns: Mapping[str, str]) -> dict[str, Any]:
    row: dict[str, Any] = {
        "path": repo_rel(path),
        "exists": path.is_file(),
        "patterns": {},
    }
    if not path.is_file():
        return row
    text = path.read_text(encoding="utf-8", errors="replace")
    row.update({"bytes": path.stat().st_size, "sha256": sha256_bytes(text.encode("utf-8"))})
    for name, pattern in patterns.items():
        row["patterns"][name] = pattern in text
    return row


def analyze_pr86_hpac_scn_contract(
    archive: Path = DEFAULT_PR86_ARCHIVE,
    *,
    training_archive_source: Path = DEFAULT_PR86_TRAINING_ARCHIVE_SOURCE,
    inflate_source: Path = DEFAULT_PR86_INFLATE_SOURCE,
) -> dict[str, Any]:
    """Check whether PR86 HPAC encode/decode SCN modes are byte-contract aligned.

    This is source-static plus payload-schema evidence. It is intentionally not
    a proof of leaderboard invalidity and does not unlock dispatch.
    """

    started_at = time.time()
    report: dict[str, Any] = {
        "schema": "pr86_hpac_scn_contract_analysis_v1",
        "tool": "tac.pr86_hpac_codec.analyze_pr86_hpac_scn_contract",
        "recorded_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "score_claim": False,
        "dispatch_allowed": False,
        "evidence_grade": "source_static_plus_payload_schema",
        "archive": repo_rel(Path(archive)),
        "status": "running",
        "source_static": {},
        "packed_state_schema": {},
        "candidate_root_causes": [],
        "limitations": [
            "This does not include the original uncompressed encoder workspace checkpoint.",
            "This does not prove the public leaderboard score is impossible.",
            "Full promotion still requires archive.zip -> inflate.sh -> upstream/evaluate.py CUDA success.",
        ],
    }
    try:
        bundle = read_pr86_archive(Path(archive))
        packed_schema = summarize_hpac_packed_state_schema(bundle.members["hpac.pt.ppmd"])
        training_report = _source_pattern_report(
            Path(training_archive_source),
            {
                "encoder_loads_packed_hpac": "load_hpac_from_ppmd",
                "encoder_calls_set_scn_false": "gen.set_scn(False)",
                "encoder_writes_tokens_with_loaded_gen": "write_tokens(gen, gt, arch_dir / \"tokens.bin\")",
                "encoder_reconstructs_packed_state_dict": "reconstruct_hpac_state_dict(packed_sd)",
            },
        )
        inflate_report = _source_pattern_report(
            Path(inflate_source),
            {
                "inflate_has_hpac_plain_runtime": "Plain masked conv (no SCN - quantization pre-applied at build time)",
                "inflate_loads_hpac_ppmd": "hpac.pt.ppmd",
                "inflate_has_set_scn_call": ".set_scn(",
                "inflate_reconstructs_weight_q": ".weight_q",
            },
        )
        report["source_static"] = {
            "training_archive_py": training_report,
            "inflate_py": inflate_report,
        }
        report["packed_state_schema"] = packed_schema
        encoder_patterns = training_report.get("patterns", {})
        inflate_patterns = inflate_report.get("patterns", {})
        encoder_scn_off = bool(encoder_patterns.get("encoder_calls_set_scn_false"))
        inflate_no_scn_call = inflate_report.get("exists") and not bool(
            inflate_patterns.get("inflate_has_set_scn_call")
        )
        packed_only = (
            packed_schema.get("suffix_counts", {}).get(".weight_q", 0) > 0
            and packed_schema.get("suffix_counts", {}).get(".weight", 0) == 0
            and packed_schema.get("suffix_counts", {}).get(".b", 0) == 0
            and packed_schema.get("suffix_counts", {}).get(".e", 0) == 0
        )
        if encoder_scn_off and inflate_no_scn_call and packed_only:
            report["candidate_root_causes"].append(
                {
                    "id": "encoder_decoder_scn_mode_divergence",
                    "confidence": "strong_candidate_not_proof",
                    "summary": (
                        "The archive builder encodes tokens after gen.set_scn(False), "
                        "while the submitted HPAC member contains packed weight_q/"
                        "weight_scale tensors and no raw weights or SCN b/e runtime "
                        "parameters; inflate reconstructs a plain pre-applied model."
                    ),
                    "expected_symptom": (
                        "range decoder can match early symbols but fails once the "
                        "probability stream diverges enough to violate constriction"
                    ),
                    "dispatch_allowed": False,
                }
            )
        report["status"] = (
            "candidate_encoder_decoder_scn_mode_divergence"
            if report["candidate_root_causes"]
            else "no_scn_mode_divergence_candidate_found"
        )
    except Pr86HpacReplayError as exc:
        report["status"] = "failed_closed"
        report["failure_stage"] = exc.contract
        report["failure_reason"] = exc.code
        report["failure_context"] = dict(exc.fields)
    report["elapsed_sec"] = round(time.time() - started_at, 3)
    return _jsonable(report)


def load_hpac_model_from_ppmd(
    payload: bytes | Path | str,
    config: Mapping[str, Any] | None = None,
    *,
    num_pairs: int | None = None,
    P: int | None = None,
    delta: int | None = None,
    ch: int | None = None,
    d_film: int | None = None,
    use_spm: bool | None = None,
    device: str = "cpu",
    strict: bool = False,
) -> HPACMini:
    """Load a PR86 HPACMini from a PPMd/gzip/raw torch state payload."""

    cfg = dict(config or {})
    resolved_num_pairs = num_pairs if num_pairs is not None else cfg.get("N", cfg.get("num_pairs"))
    resolved_P = P if P is not None else cfg.get("P")
    resolved_delta = delta if delta is not None else cfg.get("delta")
    resolved_ch = ch if ch is not None else cfg.get("ch", cfg.get("channels"))
    resolved_d_film = d_film if d_film is not None else cfg.get("hpac_d_film", cfg.get("d_film", 32))
    resolved_use_spm = use_spm if use_spm is not None else cfg.get("use_spm", False)
    missing = [
        name
        for name, value in (
            ("num_pairs", resolved_num_pairs),
            ("P", resolved_P),
            ("delta", resolved_delta),
            ("ch", resolved_ch),
        )
        if value is None
    ]
    if missing:
        raise Pr86HpacReplayError(
            "hpac_model_config_contract",
            "missing_hpac_model_config",
            missing=missing,
        )
    packed_sd = _load_packed_hpac_state(payload)
    sd = _reconstruct_hpac_state_dict(packed_sd, device)
    model = HPACMini(
        num_pairs=int(resolved_num_pairs),
        num_classes=NUM_CLASSES,
        P=int(resolved_P),
        delta=int(resolved_delta),
        ch=int(resolved_ch),
        d_film=int(resolved_d_film),
        use_spm=bool(resolved_use_spm),
    ).to(device).eval()
    model.load_state_dict(sd, strict=bool(strict))
    return model


def _payload_report(payload: Any) -> dict[str, Any]:
    if isinstance(payload, Mapping):
        keys = list(payload.keys())
        return {"type": "mapping", "key_count": len(keys), "keys_preview": keys[:12]}
    if torch is not None and torch.is_tensor(payload):
        return {
            "type": "tensor",
            "shape": list(payload.shape),
            "dtype": str(payload.dtype),
            "numel": int(payload.numel()),
        }
    return {"type": type(payload).__name__}


def _decode_required_members(
    bundle: Pr86ArchiveBundle, *, device: str = "cpu"
) -> tuple[dict[str, Any], HPACMini, dict[str, Any]]:
    meta = decode_meta_member(bundle.members["meta.pt"])
    master = decode_gzip_torch_member(bundle.members["master.pt.gz"])
    slave = decode_gzip_torch_member(bundle.members["slave.pt.gz"])
    hpac_model = load_hpac_model_from_ppmd(
        bundle.members["hpac.pt.ppmd"],
        meta,
        device=device,
        strict=False,
    )
    decoded_members = {
        "meta.pt": _payload_report(meta),
        "master.pt.gz": _payload_report(master),
        "slave.pt.gz": _payload_report(slave),
        "hpac.pt.ppmd": {
            "type": "HPACMini",
            "P": hpac_model.P,
            "delta": hpac_model.delta,
            "ch": hpac_model.ch,
            "use_spm": hpac_model.use_spm,
        },
    }
    return decoded_members, hpac_model, meta


def _archive_bundle_report(bundle: Pr86ArchiveBundle) -> dict[str, Any]:
    return {
        "path": repo_rel(bundle.contract.archive_path),
        "bytes": bundle.extra.get("archive_bytes"),
        "sha256": bundle.sha256,
        "expected_bytes": bundle.contract.expected_bytes,
        "expected_sha256": bundle.contract.expected_sha256,
        "member_reports": bundle.extra.get("member_reports", {}),
    }


def _finalize_replay_report(report: dict[str, Any], started_at: float) -> dict[str, Any]:
    report["elapsed_sec"] = round(time.time() - started_at, 3)
    return _jsonable(report)


def run_pr86_hpac_replay(
    archive: Path = DEFAULT_PR86_ARCHIVE,
    *,
    contract: Pr86ArchiveContract | None = None,
    source_dir: Path | None = DEFAULT_PR86_MERGED_SOURCE_DIR,
    source_artifacts: tuple[Path, ...] = (),
    device: str = "cpu",
    max_frames: int | None = 1,
    attempt_reencode: bool = False,
    probability_variant: str | HpacProbabilityVariant = DEFAULT_HPAC_PROBABILITY_VARIANT,
) -> dict[str, Any]:
    """Run local-only PR86 HPAC archive replay/custody diagnostics.

    This emits structured forensic evidence. It never runs contest eval and
    never unlocks dispatch; HPAC re-encode remains intentionally fail-loud.
    """

    started_at = time.time()
    variant = resolve_hpac_probability_variant(probability_variant)
    report: dict[str, Any] = {
        "schema_version": 1,
        "tool": "tac.pr86_hpac_codec.run_pr86_hpac_replay",
        "recorded_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "status": "running",
        "score_claim": False,
        "dispatch_performed": False,
        "gpu_or_remote_work": False,
        "local_only": True,
        "device": device,
        "max_frames": max_frames,
        "attempt_reencode": bool(attempt_reencode),
        "probability_variant": _jsonable(variant.__dict__),
        "byte_parity_achieved": False,
        "dispatch_unlocked": False,
        "archive": None,
        "current_source_context": {},
        "source_artifacts": {},
        "dependency_contract": None,
        "decoded_members": {},
        "tokens_bin": {},
        "hpac_decode": {},
        "hpac_reencode": {},
        "failure_stage": None,
        "failure_reason": None,
        "failure_context": {},
    }
    try:
        if source_dir is not None:
            report["current_source_context"] = analyze_pr86_current_source_context(source_dir)
        if source_artifacts:
            report["source_artifacts"] = load_source_artifact_summaries(
                {path.name: path for path in source_artifacts}
            )
        dependency_report = collect_dependency_report(strict=False)
        report["dependency_contract"] = dependency_report

        resolved_contract = contract or Pr86ArchiveContract(Path(archive))
        bundle = read_pr86_archive(Path(archive), contract=resolved_contract)
        report["archive"] = _archive_bundle_report(bundle)
        decoded_members, hpac_model, meta = _decode_required_members(bundle, device=device)
        report["decoded_members"] = decoded_members

        token_blob = bundle.members["tokens.bin"]
        token_sha = sha256_bytes(token_blob)
        report["tokens_bin"] = {
            "bytes": len(token_blob),
            "sha256": token_sha,
            "expected_sha256": EXPECTED_PR86_TOKENS_SHA256,
            "sha256_matches_expected": token_sha == EXPECTED_PR86_TOKENS_SHA256,
            "uint32_word_count": len(token_blob) // 4 if len(token_blob) % 4 == 0 else None,
            "encoding": "little_endian_uint32_words_for_constriction_queue",
        }
        if token_sha != EXPECTED_PR86_TOKENS_SHA256:
            raise Pr86HpacReplayError(
                "tokens_bin_contract",
                "tokens_sha256_mismatch",
                expected_tokens_sha256=EXPECTED_PR86_TOKENS_SHA256,
                actual_tokens_sha256=token_sha,
                tokens_bytes=len(token_blob),
            )

        decoded_tokens, decode_report = decode_tokens_hpac(
            hpac_model,
            token_blob,
            int(meta["N"]),
            int(meta.get("H", SEGNET_IN_H)),
            int(meta.get("W", SEGNET_IN_W)),
            int(meta["P"]),
            int(meta["delta"]),
            device=device,
            max_frames=max_frames,
            probability_variant=variant,
            return_report=True,
        )
        report["hpac_decode"] = decode_report
        if not attempt_reencode:
            report["status"] = "passed_prefix_decode_reencode_not_attempted"
            report["failure_stage"] = "decode_then_reencode_byte_parity"
            report["failure_reason"] = "reencode_disabled"
            report["failure_context"] = {"decoded_frames": int(decoded_tokens.shape[0])}
            return _finalize_replay_report(report, started_at)

        report["hpac_reencode"] = {
            "status": "blocked_hpac_encoder_not_rehydrated",
            "failure_reason": "encode_tokens_hpac remains fail-loud",
        }
        report["status"] = "blocked_hpac_encoder_not_rehydrated"
        report["failure_stage"] = "decode_then_reencode_byte_parity"
        report["failure_reason"] = "encode_tokens_hpac_not_rehydrated"
        return _finalize_replay_report(report, started_at)
    except Pr86HpacReplayError as exc:
        report["status"] = "failed_closed"
        report["failure_stage"] = exc.contract
        report["failure_reason"] = exc.code
        report["failure_context"] = dict(exc.fields)
        return _finalize_replay_report(report, started_at)


def run_pr86_hpac_probability_variant_matrix(
    archive: Path = DEFAULT_PR86_ARCHIVE,
    *,
    variants: tuple[str, ...] = supported_hpac_probability_variant_names(),
    contract: Pr86ArchiveContract | None = None,
    source_dir: Path | None = DEFAULT_PR86_MERGED_SOURCE_DIR,
    source_artifacts: tuple[Path, ...] = (),
    device: str = "cpu",
    max_frames: int | None = 1,
    attempt_reencode: bool = False,
) -> dict[str, Any]:
    """Run local-only PR86 HPAC probability-variant diagnostics."""

    started_at = time.time()
    rows = []
    for name in variants:
        result = run_pr86_hpac_replay(
            archive,
            contract=contract,
            source_dir=source_dir,
            source_artifacts=source_artifacts,
            device=device,
            max_frames=max_frames,
            attempt_reencode=attempt_reencode,
            probability_variant=name,
        )
        rows.append(
            {
                "variant": name,
                "status": result["status"],
                "failure_stage": result.get("failure_stage"),
                "failure_reason": result.get("failure_reason"),
                "failure_context": result.get("failure_context", {}),
                "decoded_frames": result.get("hpac_decode", {}).get("decoded_frames"),
                "decoded_tokens_sha256": result.get("hpac_decode", {})
                .get("tokens", {})
                .get("sha256"),
                "dispatch_unlocked": False,
                "score_claim": False,
            }
        )
    passed = [row["variant"] for row in rows if str(row["status"]).startswith("passed")]
    return _jsonable(
        {
            "schema": "pr86_hpac_probability_variant_matrix_v1",
            "tool": "tac.pr86_hpac_codec.run_pr86_hpac_probability_variant_matrix",
            "recorded_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            "status": "passed_local_prefix_matrix" if passed else "failed_closed",
            "score_claim": False,
            "dispatch_allowed": False,
            "dispatch_performed": False,
            "variants": list(variants),
            "passed_variants": passed,
            "variant_results": rows,
            "elapsed_sec": round(time.time() - started_at, 3),
        }
    )


def _summarize_pr86_auth_eval_log(path: Path) -> dict[str, Any]:
    row: dict[str, Any] = {"path": repo_rel(path), "exists": path.is_file()}
    if not path.is_file():
        return row
    text = path.read_text(encoding="utf-8", errors="replace")
    row.update(
        {
            "bytes": path.stat().st_size,
            "sha256": sha256_bytes(text.encode("utf-8", errors="replace")),
            "archive_sha256": None,
            "inflate_returncode": None,
            "failure_kind": None,
            "failure_summary": None,
            "leaderboard_score_claim": False,
        }
    )
    for line in text.splitlines():
        if "archive sha256:" in line:
            row["archive_sha256"] = line.rsplit(":", 1)[-1].strip()
        if "[inflate] returncode=" in line:
            row["inflate_returncode"] = line.split("[inflate] returncode=", 1)[1].split()[0]
        if "UNKNOWN file types in archive" in line:
            row["failure_kind"] = "archive_member_whitelist_failure"
            row["failure_summary"] = line.strip()
        if "Tried to decode from compressed data that is invalid" in line:
            row["failure_kind"] = "hpac_entropy_decode_contract_mismatch"
            row["failure_summary"] = line.strip()
        if "Final score:" in line:
            row["leaderboard_score_claim"] = True
    if row["failure_kind"] is None and "RuntimeError: [inflate] FAILED" in text:
        row["failure_kind"] = "inflate_failed"
        row["failure_summary"] = "RuntimeError: [inflate] FAILED"
    row["contest_auth_eval_passed"] = row["failure_kind"] is None and "RESULT_JSON:" in text
    return row


def analyze_pr86_hpac_entropy_contract(
    archive: Path = DEFAULT_PR86_ARCHIVE,
    *,
    variants: tuple[str, ...] = ("source_float64_perfect_false",),
    exact_eval_logs: tuple[Path, ...] = DEFAULT_PR86_EXACT_EVAL_LOGS,
    max_frames: int | None = 1,
) -> dict[str, Any]:
    """Classify the PR86 HPAC entropy-contract blocker from local evidence."""

    started_at = time.time()
    matrix = run_pr86_hpac_probability_variant_matrix(
        archive,
        variants=variants,
        source_dir=None,
        max_frames=max_frames,
        attempt_reencode=False,
    )
    log_rows = [_summarize_pr86_auth_eval_log(Path(path)) for path in exact_eval_logs]
    scn_contract = analyze_pr86_hpac_scn_contract(archive)
    entropy_failures = [
        row for row in matrix["variant_results"]
        if row.get("failure_reason") == "hpac_entropy_decode_contract_mismatch"
    ]
    auth_failures = [row for row in log_rows if row.get("failure_kind")]
    local_exact_validated = any(row.get("contest_auth_eval_passed") for row in log_rows)
    if local_exact_validated:
        classification = "contest_auth_eval_validated"
    elif entropy_failures and any(
        row.get("failure_kind") == "hpac_entropy_decode_contract_mismatch"
        for row in auth_failures
    ):
        classification = "not_locally_contest_validated_entropy_contract_mismatch"
    elif auth_failures:
        classification = "not_locally_contest_validated_infra_or_archive_failure"
    elif entropy_failures:
        classification = "local_prefix_entropy_contract_mismatch"
    else:
        classification = "indeterminate"
    return _jsonable(
        {
            "schema": "pr86_hpac_entropy_contract_analysis_v1",
            "tool": "tac.pr86_hpac_codec.analyze_pr86_hpac_entropy_contract",
            "recorded_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            "status": classification,
            "score_claim": False,
            "dispatch_allowed": False,
            "archive": repo_rel(Path(archive)),
            "probability_matrix": matrix,
            "exact_eval_logs": log_rows,
            "scn_contract": scn_contract,
            "classification": {
                "local_exact_validated": local_exact_validated,
                "entropy_failure_variants": [row["variant"] for row in entropy_failures],
                "auth_failure_kinds": [row.get("failure_kind") for row in auth_failures],
                "likely_root_cause_candidates": [
                    row["id"] for row in scn_contract.get("candidate_root_causes", [])
                ],
                "contest_compliance_position": (
                    "external_leaderboard_claim_not_promotable_until_auth_eval_passes"
                    if not local_exact_validated
                    else "local_auth_eval_passed"
                ),
                "next_actions": [
                    "Diff the submitted runtime/dependency path against the environment that produced the PR body score.",
                    "Recover a token source or encoder contract that crosses frame 0 group 10 symbol 191.",
                    "Do not port PR86/PR91 HPAC into dispatchable stacks until full decode and reencode parity pass.",
                ],
            },
            "elapsed_sec": round(time.time() - started_at, 3),
        }
    )
