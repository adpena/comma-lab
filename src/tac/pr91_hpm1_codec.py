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
import io
import json
import struct
import time
import zipfile
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from itertools import permutations
from pathlib import Path
from typing import Any

import numpy as np

from tac import pr86_hpac_codec as _pr86_hpac_codec

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
    SEGMENT_ORDER,
    parse_hpm1_mask_segment,
    parse_pr85_bundle,
)
from tac.pr86_hpac_codec import (
    DEFAULT_HPAC_PROBABILITY_VARIANT,
    DEFAULT_PR86_ARCHIVE,
    EXPECTED_PR86_TOKENS_SHA256,
    PPMD_MEM_SIZE,
    PROB_EPS,
    Pr86HpacReplayError,
    _categorical_from_probs,
    _group_masks,
    _normalize_probability_row,
    collect_dependency_report,
    decode_tokens_hpac,
    load_hpac_model_from_ppmd,
    resolve_hpac_probability_variant,
    sha256_bytes,
    supported_hpac_probability_variant_names,
)

REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_PR91_INTAKE_DIR = (
    REPO_ROOT / "experiments/results/public_pr91_intake_20260504_codex"
)
DEFAULT_PR91_RELEASE_RUNTIME_SOURCE_DIR = (
    REPO_ROOT
    / "experiments/results/public_pr_archive_release_view/"
    "public_pr91_intake_20260505_auto/source/submissions/hpac_coder_hybrid"
)
DEFAULT_PR91_RUNTIME_SOURCE_DIR = DEFAULT_PR91_RELEASE_RUNTIME_SOURCE_DIR
PR91_REQUIRED_RUNTIME_SOURCE_FILES = (
    "inflate.py",
    "pr86_hpac.py",
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
DEFAULT_PR85_QMA9_DECODED_REFERENCE_TOKEN_SOURCE = (
    REPO_ROOT
    / "experiments/results/pr85_qma9_mode_sweep_20260504_codex/adaptive6pr.decoded.raw"
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
PR91_HPM1_TOKEN_WORD_ORDER_CANDIDATES = (
    "source_little_uint32",
    "source_native_uint32",
    "big_endian_uint32_words",
    "reversed_little_uint32_words",
    "reversed_big_endian_uint32_words",
)
PR91_HPM1_SPATIAL_ORDER_CANDIDATES = (
    "source_mask_row_major",
    "full_col_major",
    "tile_major_row_major",
    "phase_major_row_major",
)
PR91_HPM1_SOURCE_FAILURE_FRAME = 0
PR91_HPM1_SOURCE_FAILURE_GROUP = 10
PR91_HPM1_SOURCE_FAILURE_SYMBOL_IN_GROUP = 191
PR91_HPM1_SOURCE_FAILURE_DECODED_BEFORE = 5951
DEFAULT_PR91_HPM1_RANGE_PREFIX_WINDOW_SYMBOLS = (1024, 256, 64, 16, 1)
DEFAULT_PR91_HPM1_RANGE_PREFIX_SEED_SYMBOLS = (1, 2, 4, 8, 16, 32, 64, 128)
DEFAULT_PR91_HPM1_RANGE_PREFIX_REPLAY_SYMBOL_LIMIT = 2048
DEFAULT_PR91_HPM1_RANGE_PREFIX_MAX_TARGET_DECODED_BEFORE = 4096
DEFAULT_PR91_HPM1_SYMBOL_BRIDGE_PREFIX_SYMBOLS = 32
DEFAULT_PR91_HPM1_SYMBOL_BRIDGE_MAX_SYMBOLS = 4096

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
    for x, y in zip(a, b, strict=False):
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
            "required_source_files_present": False,
            "required_source_files": list(PR91_REQUIRED_RUNTIME_SOURCE_FILES),
            "missing_required_source_files": list(PR91_REQUIRED_RUNTIME_SOURCE_FILES),
        }
    files = []
    source_files = []
    pycache_files = []
    for path in sorted(source_dir.rglob("*")):
        if path.is_file():
            rel = path.relative_to(source_dir).as_posix()
            data = path.read_bytes()
            record = {"path": rel, "bytes": len(data), "sha256": sha256_bytes(data)}
            files.append(record)
            if rel.startswith("__pycache__/") or rel.endswith(".pyc"):
                pycache_files.append(record)
            else:
                source_files.append(record)
    present_paths = {row["path"] for row in files}
    missing_required = [
        name for name in PR91_REQUIRED_RUNTIME_SOURCE_FILES if name not in present_paths
    ]
    required_present = not missing_required
    pycache_only = bool(files) and not source_files
    status = (
        "passed_static_source_inventory"
        if required_present and not pycache_only
        else "failed_closed_missing_required_runtime_sources"
    )
    return {
        "status": status,
        "source_dir": repo_rel(source_dir),
        "score_claim": False,
        "file_count": len(files),
        "source_file_count": len(source_files),
        "pycache_file_count": len(pycache_files),
        "files": files,
        "source_files": source_files,
        "pycache_files": pycache_files,
        "required_source_files": list(PR91_REQUIRED_RUNTIME_SOURCE_FILES),
        "missing_required_source_files": missing_required,
        "required_source_files_present": required_present,
        "pycache_only": pycache_only,
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
    """Load the HPACMini model embedded in an HPM1 mask payload.

    This is a local forensic helper only.  It reconstructs the model object
    needed for bounded token-prefix probes, but it does not dispatch eval or
    imply byte-closed PR91 replay.
    """

    return load_hpac_model_from_ppmd(
        payload.hpac,
        num_pairs=payload.n_frames,
        P=payload.predictor_count,
        delta=payload.delta,
        ch=payload.channels,
        d_film=payload.hpac_d_film,
        use_spm=bool(payload.use_spm),
        device=device,
        strict=False,
    )


def _dependency_observed(report: Mapping[str, Any], name: str) -> str:
    observed = report.get("observed", {})
    if not isinstance(observed, Mapping):
        return "unknown"
    value = observed.get(name, "unknown")
    return value if isinstance(value, str) else str(value)


def _dependency_available(report: Mapping[str, Any], name: str) -> bool:
    return _dependency_observed(report, name) not in {"missing", "unknown"}


def _load_hpm1_packed_state_bytes(payload: Hpm1MaskPayload) -> tuple[bytes, Mapping[str, Any]]:
    if torch is None:  # pragma: no cover - optional dependency path
        raise Pr91Hpm1Error("dependency_contract", "torch_missing")
    try:
        import pyppmd
    except ImportError as exc:  # pragma: no cover - optional dependency path
        raise Pr91Hpm1Error("dependency_contract", "pyppmd_missing") from exc
    try:
        raw = pyppmd.decompress(
            payload.hpac,
            max_order=int(payload.ppmd_order),
            mem_size=PPMD_MEM_SIZE,
        )
    except Exception as exc:
        raise Pr91Hpm1Error(
            "hpac_model_contract",
            "ppmd_decompress_failed",
            ppmd_order=payload.ppmd_order,
            hpac_bytes=len(payload.hpac),
        ) from exc
    loaded = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=False)
    if not isinstance(loaded, Mapping):
        raise Pr91Hpm1Error(
            "hpac_model_contract",
            "expected_state_dict_mapping",
            loaded_type=type(loaded).__name__,
        )
    return raw, loaded


def _tensor_bytes_sha256(tensor: Any) -> str:
    cpu_tensor = tensor.detach().cpu().contiguous()
    return sha256_bytes(cpu_tensor.numpy().tobytes())


def _state_dict_tensor_inventory(state: Mapping[str, Any], *, state_kind: str) -> dict[str, Any]:
    tensor_rows: list[dict[str, Any]] = []
    non_tensor_rows: list[dict[str, Any]] = []
    dtype_counts: dict[str, int] = {}
    total_numel = 0
    sorted_keys = sorted(str(key) for key in state)
    for key in sorted_keys:
        value = state[key]
        if torch is not None and torch.is_tensor(value):
            dtype = str(value.dtype)
            numel = int(value.numel())
            dtype_counts[dtype] = dtype_counts.get(dtype, 0) + 1
            total_numel += numel
            tensor_rows.append(
                {
                    "key": key,
                    "shape": list(value.shape),
                    "dtype": dtype,
                    "numel": numel,
                    "sha256": _tensor_bytes_sha256(value),
                }
            )
        else:
            non_tensor_rows.append({"key": key, "type": type(value).__name__})
    return {
        "state_kind": state_kind,
        "key_count": len(sorted_keys),
        "tensor_count": len(tensor_rows),
        "non_tensor_count": len(non_tensor_rows),
        "total_numel": total_numel,
        "dtype_counts": dtype_counts,
        "keys_sha256": sha256_bytes("\n".join(sorted_keys).encode("utf-8")),
        "tensors": tensor_rows,
        "non_tensors": non_tensor_rows,
    }


def _resolve_probe_variants(variants: tuple[str, ...] | list[str] | None) -> tuple[str, ...]:
    if variants is None:
        return (DEFAULT_HPAC_PROBABILITY_VARIANT,)
    return _validate_probe_variants(tuple(variants))


def _hpm1_probability_row_probe(
    model: Any,
    payload: Hpm1MaskPayload,
    *,
    variants: tuple[str, ...],
    row_count: int,
    prob_eps: float,
    device: str,
) -> dict[str, Any]:
    if torch is None or F is None:  # pragma: no cover - optional dependency path
        raise Pr91Hpm1Error("dependency_contract", "torch_missing")
    if int(row_count) <= 0:
        raise Pr91Hpm1Error(
            "probability_row_probe_contract",
            "row_count_must_be_positive",
            row_count=row_count,
        )
    dev = torch.device(device)
    masks = _group_masks(
        payload.height,
        payload.width,
        P=payload.predictor_count,
        delta=payload.delta,
        device=dev,
    )
    first_group = None
    first_mask = None
    for group_index, mask in enumerate(masks):
        if mask is not None:
            first_group = group_index
            first_mask = mask
            break
    if first_mask is None or first_group is None:
        raise Pr91Hpm1Error("probability_row_probe_contract", "no_nonempty_hpac_group")

    model = model.to(dev).eval()
    with torch.no_grad():
        idx = torch.tensor([0], dtype=torch.long, device=dev)
        cur = torch.zeros(
            (1, payload.height, payload.width),
            dtype=torch.long,
            device=dev,
        )
        prev = torch.zeros_like(cur)
        logits = model(cur, idx, prev)
        probs = F.softmax(logits.float(), dim=1)
        rows = probs[0][:, first_mask].permute(1, 0).contiguous()
        rows = rows[: min(int(row_count), int(rows.shape[0]))].cpu().numpy()

    variant_rows: list[dict[str, Any]] = []
    for variant_name in variants:
        resolved = resolve_hpac_probability_variant(variant_name)
        normalized = np.stack(
            [
                _normalize_probability_row(row, prob_eps=prob_eps, variant=resolved)
                for row in rows
            ],
            axis=0,
        )
        categorical_constructed = False
        categorical_error = ""
        try:
            _categorical_from_probs(rows[0], prob_eps=prob_eps, variant=resolved)
            categorical_constructed = True
        except Exception as exc:  # pragma: no cover - dependency-specific shape
            categorical_error = f"{type(exc).__name__}: {exc}"
        row_sums = normalized.sum(axis=1)
        variant_rows.append(
            {
                "variant": resolved.name,
                "probability_dtype": resolved.probability_dtype,
                "categorical_perfect": resolved.categorical_perfect,
                "source_contract": resolved.source_contract,
                "normalized_rows_shape": list(normalized.shape),
                "normalized_rows_sha256": sha256_bytes(normalized.tobytes()),
                "first_normalized_row": [
                    round(float(value), 10) for value in normalized[0].tolist()
                ],
                "row_sum_min": round(float(row_sums.min()), 10),
                "row_sum_max": round(float(row_sums.max()), 10),
                "categorical_constructed_first_row": categorical_constructed,
                "categorical_error": categorical_error,
            }
        )

    return {
        "status": "passed_probability_row_inventory",
        "passed": True,
        "score_claim": False,
        "dispatch_allowed": False,
        "device": device,
        "prob_eps": float(prob_eps),
        "first_nonempty_group": first_group,
        "symbols_in_group": int(first_mask.sum().item()),
        "requested_rows": int(row_count),
        "recorded_rows": int(rows.shape[0]),
        "raw_softmax_rows": {
            "shape": list(rows.shape),
            "dtype": str(rows.dtype),
            "sha256": sha256_bytes(rows.tobytes()),
            "first_row": [round(float(value), 10) for value in rows[0].tolist()],
        },
        "variant_rows": variant_rows,
        "full_decode_proven": False,
        "byte_exact_reencode_proven": False,
    }


def _small_int_preview(values: np.ndarray, *, count: int = 16) -> dict[str, Any]:
    arr = np.asarray(values).reshape(-1)
    return {
        "count": int(arr.size),
        "first": [int(value) for value in arr[:count].tolist()],
        "last": [int(value) for value in arr[-count:].tolist()],
    }


def _context_tensor_summary(tensor: Any, *, label: str) -> dict[str, Any]:
    arr = tensor.detach().cpu().numpy().astype(np.uint8, copy=False)
    return {
        "label": label,
        "shape": list(arr.shape),
        "sha256": sha256_bytes(arr.tobytes()),
        "nonzero_count": int(np.count_nonzero(arr)),
        "min": int(arr.min()) if arr.size else None,
        "max": int(arr.max()) if arr.size else None,
    }


def _hpm1_token_words_for_candidate(
    payload: Hpm1MaskPayload,
    candidate: str,
) -> np.ndarray:
    """Return a deterministic uint32 queue view for a token-order hypothesis."""

    if payload.tokens_len % 4:
        raise Pr91Hpm1Error(
            "hpm1_token_word_order_probe",
            "tokens_not_uint32_aligned",
            tokens_bytes=payload.tokens_len,
        )
    little = np.frombuffer(payload.tokens, dtype="<u4").astype(np.uint32, copy=False)
    if candidate == "source_little_uint32":
        return np.ascontiguousarray(little, dtype=np.uint32)
    if candidate == "source_native_uint32":
        return np.ascontiguousarray(
            np.frombuffer(payload.tokens, dtype=np.uint32).astype(np.uint32, copy=False),
            dtype=np.uint32,
        )
    if candidate == "big_endian_uint32_words":
        return np.ascontiguousarray(
            np.frombuffer(payload.tokens, dtype=">u4").astype(np.uint32, copy=True),
            dtype=np.uint32,
        )
    if candidate == "reversed_little_uint32_words":
        return np.ascontiguousarray(little[::-1], dtype=np.uint32)
    if candidate == "reversed_big_endian_uint32_words":
        big = np.frombuffer(payload.tokens, dtype=">u4").astype(np.uint32, copy=True)
        return np.ascontiguousarray(big[::-1], dtype=np.uint32)
    raise Pr91Hpm1Error(
        "hpm1_token_word_order_probe",
        "unsupported_token_word_order_candidate",
        candidate=candidate,
        supported=list(PR91_HPM1_TOKEN_WORD_ORDER_CANDIDATES),
    )


def _token_words_summary(words: np.ndarray) -> dict[str, Any]:
    arr = np.ascontiguousarray(words, dtype=np.uint32)
    return {
        "uint32_word_count": int(arr.size),
        "decoder_words_sha256": sha256_bytes(arr.tobytes()),
        "first_words_hex": [f"0x{int(word):08x}" for word in arr[:8]],
        "last_words_hex": [f"0x{int(word):08x}" for word in arr[-8:]],
    }


def _failure_row_signature(trace: Mapping[str, Any]) -> dict[str, int] | None:
    failure = trace.get("failure")
    if not isinstance(failure, Mapping):
        return None
    keys = (
        "frame",
        "group",
        "symbol_in_group",
        "decoded_symbol_count_before_failure",
    )
    if not all(key in failure for key in keys):
        return None
    return {key: int(failure[key]) for key in keys}


def _decoded_symbols_or_before_failure(trace: Mapping[str, Any]) -> int:
    signature = _failure_row_signature(trace)
    if signature is not None:
        return int(signature["decoded_symbol_count_before_failure"])
    return int(trace.get("decoded_symbols", 0) or 0)


def _spatial_order_description(candidate: str) -> str:
    if candidate == "source_mask_row_major":
        return (
            "Submitted PR91 source order: boolean-mask indexing over the full "
            "H x W grid, which visits masked positions in row-major order."
        )
    if candidate == "full_col_major":
        return (
            "Off-contract probe: visit the same full-grid masked positions in "
            "column-major order."
        )
    if candidate == "tile_major_row_major":
        return (
            "Off-contract probe: visit patch tiles row-major, then row-major "
            "within each P x P tile for positions belonging to the group."
        )
    if candidate == "phase_major_row_major":
        return (
            "Off-contract probe: visit each P x P phase for the group first, "
            "then scan tiles row-major for that phase."
        )
    raise Pr91Hpm1Error(
        "hpm1_spatial_order_probe",
        "unsupported_spatial_order_candidate",
        candidate=candidate,
        supported=list(PR91_HPM1_SPATIAL_ORDER_CANDIDATES),
    )


def _validate_spatial_order_candidates(candidates: tuple[str, ...]) -> tuple[str, ...]:
    requested = tuple(dict.fromkeys(str(name) for name in candidates if str(name)))
    if not requested:
        raise Pr91Hpm1Error(
            "hpm1_spatial_order_probe",
            "at_least_one_spatial_order_candidate_required",
        )
    for name in requested:
        _spatial_order_description(name)
    return requested


def _spatial_candidate_scope_label(candidates: tuple[str, ...]) -> str:
    labels_by_candidate = {
        "source_mask_row_major": "source-mask row-major",
        "full_col_major": "full-column-major",
        "tile_major_row_major": "tile-major row-major",
        "phase_major_row_major": "phase-major row-major",
    }
    labels = [labels_by_candidate.get(candidate, candidate) for candidate in candidates]
    if len(labels) == 1:
        return labels[0]
    if len(labels) == 2:
        return f"{labels[0]} and {labels[1]}"
    return f"{', '.join(labels[:-1])}, and {labels[-1]}"


def _hpm1_group_coords_for_spatial_order(
    payload: Hpm1MaskPayload,
    *,
    group: int,
    mask: Any,
    candidate: str,
    device: Any,
) -> Any:
    """Return [row, col] coordinates for one HPM1 group-order hypothesis."""

    _spatial_order_description(candidate)
    source_coords = mask.nonzero(as_tuple=False).to(dtype=torch.long, device=device)
    if candidate == "source_mask_row_major":
        return source_coords.contiguous()

    h = int(payload.height)
    w = int(payload.width)
    p = int(payload.predictor_count)
    delta = int(payload.delta)
    if candidate == "full_col_major":
        order = torch.argsort(source_coords[:, 1] * h + source_coords[:, 0])
        return source_coords[order].contiguous()

    nrp = h // p
    ncp = w // p
    rows: list[tuple[int, int]] = []
    if candidate == "tile_major_row_major":
        for tile_r in range(nrp):
            row_base = tile_r * p
            for tile_c in range(ncp):
                col_base = tile_c * p
                for pr in range(p):
                    for pc in range(p):
                        if pc + delta * pr == int(group):
                            rows.append((row_base + pr, col_base + pc))
    elif candidate == "phase_major_row_major":
        for pr in range(p):
            for pc in range(p):
                if pc + delta * pr != int(group):
                    continue
                for tile_r in range(nrp):
                    row = tile_r * p + pr
                    for tile_c in range(ncp):
                        rows.append((row, tile_c * p + pc))
    else:  # pragma: no cover - guarded by _spatial_order_description above
        raise AssertionError(candidate)

    coords = torch.tensor(rows, dtype=torch.long, device=device)
    if int(coords.shape[0]) != int(source_coords.shape[0]):
        raise Pr91Hpm1Error(
            "hpm1_spatial_order_probe",
            "candidate_coord_count_mismatch",
            candidate=candidate,
            group=group,
            expected=int(source_coords.shape[0]),
            actual=int(coords.shape[0]),
        )
    source_set = {
        (int(row), int(col))
        for row, col in source_coords.detach().cpu().numpy().tolist()
    }
    candidate_set = {
        (int(row), int(col))
        for row, col in coords.detach().cpu().numpy().tolist()
    }
    if candidate_set != source_set:
        raise Pr91Hpm1Error(
            "hpm1_spatial_order_probe",
            "candidate_coord_set_mismatch",
            candidate=candidate,
            group=group,
        )
    return coords.contiguous()


def _coords_summary(coords: Any) -> dict[str, Any]:
    arr = coords.detach().cpu().numpy().astype(np.uint16, copy=False)
    return {
        "count": int(arr.shape[0]),
        "sha256": sha256_bytes(np.ascontiguousarray(arr).tobytes()),
        "first": [
            [int(row), int(col)]
            for row, col in arr[:8].astype(np.int64, copy=False).tolist()
        ],
        "last": [
            [int(row), int(col)]
            for row, col in arr[-8:].astype(np.int64, copy=False).tolist()
        ],
    }


def _range_decoder_state_summary(decoder: Any, *, label: str) -> dict[str, Any]:
    """Record the limited public RangeDecoder state without mutating it."""

    state: dict[str, Any] = {
        "label": label,
        "api": "constriction.stream.queue.RangeDecoder",
        "public_state_surface": "maybe_exhausted_only",
    }
    maybe_exhausted = getattr(decoder, "maybe_exhausted", None)
    if callable(maybe_exhausted):
        try:
            state["maybe_exhausted"] = bool(maybe_exhausted())
        except Exception as exc:  # pragma: no cover - dependency-specific
            state["maybe_exhausted_error"] = f"{type(exc).__name__}: {exc}"
    else:
        state["maybe_exhausted_available"] = False
    return state


def _call_noarg_public_method(obj: Any, name: str) -> Any:
    member = getattr(obj, name, None)
    if not callable(member):
        return None
    return member()


def _uint32_words_digest(words: Any, *, preview: int = 8) -> dict[str, Any]:
    arr = np.ascontiguousarray(np.asarray(words, dtype=np.uint32), dtype=np.uint32)
    return {
        "uint32_word_count": int(arr.size),
        "sha256": sha256_bytes(arr.tobytes()),
        "first_words_hex": [f"0x{int(word):08x}" for word in arr[:preview]],
        "last_words_hex": [f"0x{int(word):08x}" for word in arr[-preview:]],
    }


def _common_prefix_uint32_words(a: np.ndarray, b: np.ndarray) -> int:
    left = np.ascontiguousarray(a, dtype=np.uint32)
    right = np.ascontiguousarray(b, dtype=np.uint32)
    limit = min(int(left.size), int(right.size))
    if limit <= 0:
        return 0
    mismatch = np.flatnonzero(left[:limit] != right[:limit])
    if int(mismatch.size) == 0:
        return limit
    return int(mismatch[0])


def _range_encoder_state_summary(encoder: Any, *, label: str) -> dict[str, Any]:
    """Record deterministic public RangeEncoder state without mutating it."""

    state: dict[str, Any] = {
        "label": label,
        "api": "constriction.stream.queue.RangeEncoder",
    }
    for name in ("num_bits", "num_words", "is_empty"):
        try:
            value = _call_noarg_public_method(encoder, name)
        except Exception as exc:  # pragma: no cover - dependency-specific
            state[f"{name}_error"] = f"{type(exc).__name__}: {exc}"
            continue
        if value is not None:
            state[name] = int(value) if isinstance(value, (bool, int, np.integer)) else value
    try:
        pos = _call_noarg_public_method(encoder, "pos")
    except Exception as exc:  # pragma: no cover - dependency-specific
        state["pos_error"] = f"{type(exc).__name__}: {exc}"
    else:
        if isinstance(pos, tuple) and len(pos) == 2:
            queue_index, interval = pos
            state["pos"] = {
                "queue_index": int(queue_index),
                "interval": [int(value) for value in interval],
            }
        elif pos is not None:
            state["pos"] = str(pos)
    try:
        clone = encoder.clone()
        state["compressed_words_snapshot"] = _uint32_words_digest(
            clone.get_compressed()
        )
    except Exception as exc:  # pragma: no cover - dependency-specific
        state["compressed_words_snapshot_error"] = f"{type(exc).__name__}: {exc}"
    return state


def _single_row_range_model_roundtrip(
    raw_row: np.ndarray,
    *,
    probability_variant: str | Any,
    prob_eps: float,
) -> dict[str, Any]:
    """Check local constriction single-row admissibility without touching PR91 bytes."""

    if _pr86_hpac_codec.constriction is None:  # pragma: no cover
        return {
            "schema": "pr91_hpm1_single_row_range_model_roundtrip_v1",
            "status": "not_attempted_constriction_missing",
            "passed": False,
            "score_claim": False,
            "dispatch_allowed": False,
        }
    resolved = resolve_hpac_probability_variant(probability_variant)
    row = np.ascontiguousarray(raw_row)
    cat = _categorical_from_probs(row, prob_eps=prob_eps, variant=resolved)
    symbol_results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for symbol in range(int(row.size)):
        try:
            encoder = _pr86_hpac_codec.constriction.stream.queue.RangeEncoder()
            encoder.encode(int(symbol), cat)
            words = np.ascontiguousarray(encoder.get_compressed(), dtype=np.uint32)
            decoder = _pr86_hpac_codec.constriction.stream.queue.RangeDecoder(words)
            decoded = int(decoder.decode(cat))
            passed = decoded == int(symbol)
            result = {
                "symbol": int(symbol),
                "decoded_symbol": decoded,
                "passed": passed,
                "compressed_word_count": int(words.size),
                "compressed_words_sha256": sha256_bytes(words.tobytes()),
                "decoder_maybe_exhausted_after_decode": (
                    bool(decoder.maybe_exhausted())
                    if callable(getattr(decoder, "maybe_exhausted", None))
                    else None
                ),
            }
            if not passed:
                failures.append(result)
            symbol_results.append(result)
        except Exception as exc:  # pragma: no cover - dependency-specific
            failure = {
                "symbol": int(symbol),
                "passed": False,
                "exception_type": type(exc).__name__,
                "exception_text": str(exc),
            }
            failures.append(failure)
            symbol_results.append(failure)
    return {
        "schema": "pr91_hpm1_single_row_range_model_roundtrip_v1",
        "status": (
            "passed_all_symbols_roundtrip"
            if not failures
            else "failed_single_row_roundtrip"
        ),
        "passed": not failures,
        "score_claim": False,
        "dispatch_allowed": False,
        "probability_variant": resolved.name,
        "prob_eps": float(prob_eps),
        "symbol_count": int(row.size),
        "all_symbols_roundtrip": not failures,
        "failed_symbols": [int(row["symbol"]) for row in failures],
        "symbol_results": symbol_results,
    }


def _failure_row_interpretation(
    raw_row: np.ndarray,
    norm_row: np.ndarray,
    *,
    probability_variant: str | Any,
    prob_eps: float,
    decoder_state_before: Mapping[str, Any],
    context_mode: str,
    reference_symbol: int | None = None,
) -> dict[str, Any]:
    """Classify one failed entropy row without claiming stream parity."""

    resolved = resolve_hpac_probability_variant(probability_variant)
    normalized = np.ascontiguousarray(norm_row)
    order = np.argsort(-normalized, kind="stable")
    argmax_symbol = int(order[0]) if int(order.size) else None
    single_row_roundtrip = _single_row_range_model_roundtrip(
        np.ascontiguousarray(raw_row),
        probability_variant=resolved,
        prob_eps=prob_eps,
    )
    maybe_exhausted = decoder_state_before.get("maybe_exhausted")
    diagnostic: dict[str, Any] = {
        "schema": "pr91_hpm1_failure_row_interpretation_v1",
        "status": "no_reference_symbol_available_for_semantic_row_classification",
        "score_claim": False,
        "dispatch_allowed": False,
        "context_mode": context_mode,
        "probability_variant": resolved.name,
        "prob_eps": float(prob_eps),
        "normalized_row_sha256": sha256_bytes(normalized.tobytes()),
        "argmax_symbol": argmax_symbol,
        "argmax_probability": (
            round(float(normalized[argmax_symbol]), 10)
            if argmax_symbol is not None
            else None
        ),
        "probability_order_desc": [int(symbol) for symbol in order.tolist()],
        "decoder_not_stream_exhaustion": bool(maybe_exhausted is False),
        "single_row_range_model_roundtrip": single_row_roundtrip,
        "hypothesis_implications": [
            "single-row categorical admissibility is local-only and does not prove the submitted range state",
        ],
    }
    if reference_symbol is None:
        return diagnostic

    ref = int(reference_symbol)
    if ref < 0 or ref >= int(normalized.size):
        raise Pr91Hpm1Error(
            "hpm1_failure_row_interpretation",
            "reference_symbol_out_of_probability_row_range",
            reference_symbol=ref,
            row_size=int(normalized.size),
        )
    ref_rank = int(np.where(order == ref)[0][0]) + 1
    ref_probability = float(normalized[ref])
    ref_is_argmax = bool(argmax_symbol == ref)
    immediate_semantic_row_mismatch = not ref_is_argmax
    if (
        ref_is_argmax
        and ref_probability >= 0.5
        and single_row_roundtrip.get("all_symbols_roundtrip") is True
        and maybe_exhausted is False
    ):
        status = "not_explained_by_current_row_reference_symbol_probability"
    elif immediate_semantic_row_mismatch:
        status = "semantic_reference_symbol_probability_mismatch_still_open"
    else:
        status = "current_row_semantic_contract_inconclusive"
    diagnostic.update(
        {
            "status": status,
            "reference_symbol": ref,
            "reference_symbol_probability": round(ref_probability, 10),
            "reference_symbol_rank": ref_rank,
            "reference_symbol_is_argmax": ref_is_argmax,
            "immediate_semantic_row_mismatch": immediate_semantic_row_mismatch,
            "model_row_supports_reference_symbol": bool(ref_probability > prob_eps),
            "hypothesis_implications": [
                *diagnostic["hypothesis_implications"],
                (
                    "available reference symbol is the failing-row argmax; "
                    "this row is not explained by a bad PR85/QMA9 label"
                    if ref_is_argmax
                    else "available reference symbol is not the failing-row argmax"
                ),
                (
                    "submitted stream failure is not RangeDecoder exhaustion"
                    if maybe_exhausted is False
                    else "RangeDecoder exhaustion status is unavailable or not false"
                ),
                (
                    "local constriction can encode/decode every symbol for this categorical row"
                    if single_row_roundtrip.get("all_symbols_roundtrip") is True
                    else "local single-row categorical roundtrip did not pass all symbols"
                ),
            ],
        }
    )
    return diagnostic


def _reference_group_symbol_window(
    reference_symbols: np.ndarray,
    coords: Any,
    *,
    frame: int,
    group: int,
    failure_symbol_in_group: int,
    window_before: int,
    window_after: int,
) -> dict[str, Any]:
    """Summarize canonical reference tokens around a failed group symbol."""

    before = int(window_before)
    after = int(window_after)
    if before < 0 or after < 0:
        raise Pr91Hpm1Error(
            "reference_teacher_forcing_probe",
            "reference_window_counts_must_be_nonnegative",
            window_before=window_before,
            window_after=window_after,
        )
    ref = np.asarray(reference_symbols, dtype=np.uint8)
    coord_arr = coords.detach().cpu().numpy().astype(np.int64, copy=False)
    failure_symbol = int(failure_symbol_in_group)
    if failure_symbol < 0 or failure_symbol >= int(ref.shape[0]):
        raise Pr91Hpm1Error(
            "reference_teacher_forcing_probe",
            "failure_symbol_out_of_reference_group",
            group=group,
            failure_symbol_in_group=failure_symbol,
            symbols_in_group=int(ref.shape[0]),
        )
    start = max(0, failure_symbol - before)
    end = min(int(ref.shape[0]), failure_symbol + after + 1)
    window_symbols = np.ascontiguousarray(ref[start:end], dtype=np.uint8)
    next_symbols = np.ascontiguousarray(
        ref[failure_symbol + 1 : end],
        dtype=np.uint8,
    )
    rows = []
    for symbol_in_group in range(start, end):
        row, col = coord_arr[symbol_in_group]
        rows.append(
            {
                "symbol_in_group": int(symbol_in_group),
                "relative_to_failure": int(symbol_in_group - failure_symbol),
                "pixel_yx": {"y": int(row), "x": int(col)},
                "reference_symbol": int(ref[symbol_in_group]),
            }
        )
    fail_row, fail_col = coord_arr[failure_symbol]
    return {
        "schema": "pr91_hpm1_reference_group_symbol_window_v1",
        "frame": int(frame),
        "group": int(group),
        "failure_symbol_in_group": failure_symbol,
        "window_before": before,
        "window_after": after,
        "start_symbol_in_group": int(start),
        "end_symbol_in_group_exclusive": int(end),
        "includes_failure_symbol": True,
        "window_symbol_count": int(window_symbols.size),
        "reference_symbols_sha256": sha256_bytes(window_symbols.tobytes()),
        "next_reference_symbol_count": int(next_symbols.size),
        "next_reference_symbols_sha256": sha256_bytes(next_symbols.tobytes()),
        "failed_reference_symbol": int(ref[failure_symbol]),
        "failed_reference_pixel_yx": {"y": int(fail_row), "x": int(fail_col)},
        "rows": rows,
        "reference_scope_note": (
            "These are canonical PR85/QMA9 reference tokens under the requested "
            "layout, not proven PR91 encoder semantic tokens."
        ),
    }


def _prefix_checkpoint_symbol_counts(
    *,
    target_decoded_before: int,
    target_group_start: int,
    total_collected_symbols: int,
    range_prefix_window_symbols: tuple[int, ...],
    range_prefix_seed_symbol_counts: tuple[int, ...],
) -> list[dict[str, Any]]:
    """Build deterministic prefix checkpoints around one failed range row."""

    windows = sorted(
        {
            int(window)
            for window in range_prefix_window_symbols
            if int(window) > 0
        },
        reverse=True,
    )
    labels: dict[int, str] = {
        0: "empty_stream",
        int(target_group_start): "target_group_start",
        int(target_decoded_before): "before_failure_row",
        int(target_decoded_before) + 1: "including_failure_row",
    }
    for count in sorted({int(value) for value in range_prefix_seed_symbol_counts if int(value) > 0}):
        labels.setdefault(count, f"first_{count}_symbols")
    for window in windows:
        count = max(0, int(target_decoded_before) - window)
        labels.setdefault(count, f"failure_minus_{window}_symbols")

    rows: list[dict[str, Any]] = []
    for count in sorted(labels):
        if count < 0 or count > int(total_collected_symbols):
            continue
        rows.append(
            {
                "label": labels[count],
                "symbol_count": int(count),
                "symbols_before_failure_remaining": int(
                    int(target_decoded_before) - int(count)
                ),
                "includes_failure_symbol": bool(
                    int(count) > int(target_decoded_before)
                ),
            }
        )
    return rows


def _validate_range_prefix_window_symbols(
    range_prefix_window_symbols: tuple[int, ...] | list[int],
) -> tuple[int, ...]:
    windows = tuple(int(value) for value in range_prefix_window_symbols)
    bad = [int(value) for value in windows if int(value) <= 0]
    if bad:
        raise Pr91Hpm1Error(
            "hpm1_range_state_prefix_probe",
            "range_prefix_window_symbols_must_be_positive",
            bad_values=bad,
        )
    return windows


def _validate_range_prefix_seed_symbol_counts(
    range_prefix_seed_symbol_counts: tuple[int, ...] | list[int],
) -> tuple[int, ...]:
    seed_counts = tuple(int(value) for value in range_prefix_seed_symbol_counts)
    bad = [int(value) for value in seed_counts if int(value) <= 0]
    if bad:
        raise Pr91Hpm1Error(
            "hpm1_range_state_prefix_probe",
            "range_prefix_seed_symbol_counts_must_be_positive",
            bad_values=bad,
        )
    return seed_counts


def _replay_prefix_against_words(
    words: np.ndarray,
    raw_rows: list[np.ndarray],
    reference_symbols: list[int],
    *,
    symbol_count: int,
    probability_variant: str | Any,
    prob_eps: float,
) -> dict[str, Any]:
    """Decode a bounded prefix against a supplied word array."""

    if int(symbol_count) == 0:
        return {
            "status": "decoded_empty_prefix",
            "passed": True,
            "decoded_symbol_count": 0,
        }
    arr = np.ascontiguousarray(words, dtype=np.uint32)
    if int(arr.size) == 0:
        return {
            "status": "empty_word_stream_for_nonempty_prefix",
            "passed": False,
            "decoded_symbol_count": 0,
        }

    decoder = _pr86_hpac_codec.constriction.stream.queue.RangeDecoder(arr)
    decoded: list[int] = []
    for index in range(int(symbol_count)):
        cat = _categorical_from_probs(
            raw_rows[index],
            prob_eps=prob_eps,
            variant=probability_variant,
        )
        try:
            symbol = int(decoder.decode(cat))
        except Exception as exc:
            return {
                "status": "range_decode_exception",
                "passed": False,
                "decoded_symbol_count": int(index),
                "exception_type": type(exc).__name__,
                "exception_text": str(exc),
                "decoder_state_after_exception": _range_decoder_state_summary(
                    decoder,
                    label="after_prefix_replay_exception",
                ),
            }
        decoded.append(symbol)
        expected = int(reference_symbols[index])
        if symbol != expected:
            return {
                "status": "decoded_symbol_mismatch",
                "passed": False,
                "decoded_symbol_count": int(index + 1),
                "first_mismatch": {
                    "symbol_index": int(index),
                    "decoded_symbol": int(symbol),
                    "reference_symbol": expected,
                },
                "decoded_prefix_sha256": sha256_bytes(
                    np.asarray(decoded, dtype=np.uint8).tobytes()
                ),
                "decoder_state_after_mismatch": _range_decoder_state_summary(
                    decoder,
                    label="after_prefix_replay_mismatch",
                ),
            }

    decoded_arr = np.asarray(decoded, dtype=np.uint8)
    return {
        "status": "decoded_reference_prefix",
        "passed": True,
        "decoded_symbol_count": int(symbol_count),
        "decoded_prefix_sha256": sha256_bytes(decoded_arr.tobytes()),
        "decoder_state_after_replay": _range_decoder_state_summary(
            decoder,
            label="after_prefix_replay",
        ),
    }


def _build_range_prefix_checkpoint_report(
    raw_rows: list[np.ndarray],
    reference_symbols: list[int],
    submitted_words: np.ndarray,
    checkpoint: Mapping[str, Any],
    *,
    probability_variant: str | Any,
    prob_eps: float,
    replay_symbol_limit: int,
) -> dict[str, Any]:
    """Encode/replay one local reference prefix and compare word summaries."""

    symbol_count = int(checkpoint["symbol_count"])
    encoder = _pr86_hpac_codec.constriction.stream.queue.RangeEncoder()
    for index in range(symbol_count):
        cat = _categorical_from_probs(
            raw_rows[index],
            prob_eps=prob_eps,
            variant=probability_variant,
        )
        encoder.encode(int(reference_symbols[index]), cat)

    local_words = np.ascontiguousarray(
        np.asarray(encoder.clone().get_compressed(), dtype=np.uint32),
        dtype=np.uint32,
    )
    submitted_same_count = np.ascontiguousarray(
        submitted_words[: int(local_words.size)],
        dtype=np.uint32,
    )
    full_prefix_match = bool(
        int(local_words.size) == int(submitted_same_count.size)
        and np.array_equal(local_words, submitted_same_count)
    )
    local_replay: dict[str, Any]
    submitted_replay: dict[str, Any]
    submitted_full_replay: dict[str, Any]
    if symbol_count > int(replay_symbol_limit):
        local_replay = {
            "status": "not_attempted_symbol_limit",
            "passed": False,
            "symbol_count": symbol_count,
            "limit": int(replay_symbol_limit),
        }
        submitted_replay = dict(local_replay)
        submitted_full_replay = dict(local_replay)
    else:
        local_replay = _replay_prefix_against_words(
            local_words,
            raw_rows,
            reference_symbols,
            symbol_count=symbol_count,
            probability_variant=probability_variant,
            prob_eps=prob_eps,
        )
        submitted_full_replay = _replay_prefix_against_words(
            submitted_words,
            raw_rows,
            reference_symbols,
            symbol_count=symbol_count,
            probability_variant=probability_variant,
            prob_eps=prob_eps,
        )
        submitted_replay = _replay_prefix_against_words(
            submitted_same_count,
            raw_rows,
            reference_symbols,
            symbol_count=symbol_count,
            probability_variant=probability_variant,
            prob_eps=prob_eps,
        )

    return {
        "label": str(checkpoint["label"]),
        "symbol_count": symbol_count,
        "symbols_before_failure_remaining": int(
            checkpoint["symbols_before_failure_remaining"]
        ),
        "includes_failure_symbol": bool(checkpoint["includes_failure_symbol"]),
        "range_encoder_state": _range_encoder_state_summary(
            encoder,
            label=f"after_encoding_{symbol_count}_reference_symbols",
        ),
        "local_reference_prefix_words": _uint32_words_digest(local_words),
        "submitted_same_word_count_prefix": _uint32_words_digest(submitted_same_count),
        "submitted_word_comparison": {
            "same_word_count_prefix_matches": full_prefix_match,
            "same_word_count_prefix_scope": (
                "submitted stream truncated to the independently finalized "
                "local prefix word count"
            ),
            "common_prefix_word_count": _common_prefix_uint32_words(
                local_words,
                submitted_same_count,
            ),
            "local_word_count": int(local_words.size),
            "submitted_total_word_count": int(submitted_words.size),
            "submitted_full_stream_prefix_replay_passed": bool(
                submitted_full_replay.get("passed") is True
            ),
        },
        "local_reference_prefix_replay": local_replay,
        "submitted_same_word_count_replay": submitted_replay,
        "submitted_full_stream_prefix_replay": submitted_full_replay,
        "scope_note": (
            "Local prefix words are independently finalized reference-context "
            "range streams; mismatch with the submitted full stream is a "
            "range-state diagnostic, not a score or parity claim. The full "
            "submitted stream replay is the prefix-decode fidelity check; the "
            "same-word-count replay is only a truncation sensitivity diagnostic."
        ),
    }


def _classify_range_prefix_reconstruction(
    checkpoint_reports: list[dict[str, Any]],
) -> dict[str, Any]:
    """Summarize local prefix re-encode evidence without promoting parity."""

    first_reference_failure: dict[str, Any] | None = None
    first_word_mismatch: dict[str, Any] | None = None
    attempted_nonempty = 0
    for checkpoint in sorted(
        checkpoint_reports,
        key=lambda row: int(row.get("symbol_count", 0)),
    ):
        symbol_count = int(checkpoint.get("symbol_count", 0))
        if symbol_count <= 0:
            continue
        full_replay = checkpoint.get("submitted_full_stream_prefix_replay", {})
        if isinstance(full_replay, Mapping) and full_replay.get("status") != "not_attempted_symbol_limit":
            attempted_nonempty += 1
            if full_replay.get("passed") is not True and first_reference_failure is None:
                first_reference_failure = {
                    "checkpoint_label": checkpoint.get("label"),
                    "symbol_count": symbol_count,
                    "status": full_replay.get("status"),
                    "decoded_symbol_count": full_replay.get("decoded_symbol_count"),
                    "first_mismatch": full_replay.get("first_mismatch"),
                }
        word_comparison = checkpoint.get("submitted_word_comparison", {})
        if (
            isinstance(word_comparison, Mapping)
            and word_comparison.get("same_word_count_prefix_matches") is False
            and first_word_mismatch is None
        ):
            first_word_mismatch = {
                "checkpoint_label": checkpoint.get("label"),
                "symbol_count": symbol_count,
                "common_prefix_word_count": word_comparison.get("common_prefix_word_count"),
                "local_word_count": word_comparison.get("local_word_count"),
                "submitted_total_word_count": word_comparison.get("submitted_total_word_count"),
            }

    if first_reference_failure is not None:
        status = "submitted_stream_reference_symbol_mismatch_in_seed_prefix"
        next_patch = (
            "recover true PR91 encoder semantic tokens or a bridge from the "
            "phase-major PR85/QMA9 context prior to submitted PR91 symbols"
        )
    elif first_word_mismatch is not None:
        status = "local_reference_prefix_words_diverge_from_submitted_stream"
        next_patch = (
            "recover encoder-side range construction/finalization or probability "
            "numeric grammar before asserting byte-exact reencode"
        )
    elif attempted_nonempty:
        status = "attempted_seed_prefixes_did_not_find_reference_mismatch"
        next_patch = (
            "extend bounded seed-prefix replay or recover full encoder trace"
        )
    else:
        status = "no_nonempty_seed_prefix_replay_attempted"
        next_patch = "raise range_prefix_replay_symbol_limit for local forensic replay"

    return {
        "schema": "pr91_hpm1_range_prefix_reconstruction_classification_v1",
        "status": status,
        "score_claim": False,
        "dispatch_allowed": False,
        "full_decode_proven": False,
        "byte_exact_reencode_proven": False,
        "attempted_nonempty_prefix_replays": attempted_nonempty,
        "first_submitted_reference_prefix_failure": first_reference_failure,
        "first_local_reference_word_mismatch": first_word_mismatch,
        "next_patch_target": next_patch,
        "scope_note": (
            "Seed-prefix checks are local CPU byte-grammar diagnostics only; "
            "they do not prove full HPM1 decode, byte-exact reencode, score "
            "validity, or dispatch readiness."
        ),
    }


def _first_symbol_mismatch(
    expected_symbols: np.ndarray,
    actual_symbols: np.ndarray,
    *,
    expected_label: str,
    actual_label: str,
) -> dict[str, int] | None:
    expected = np.asarray(expected_symbols, dtype=np.uint8)
    actual = np.asarray(actual_symbols, dtype=np.uint8)
    for index, (expected_value, actual_value) in enumerate(zip(expected, actual, strict=True)):
        if int(expected_value) != int(actual_value):
            return {
                "symbol_index": int(index),
                expected_label: int(expected_value),
                actual_label: int(actual_value),
            }
    return None


def _symbol_histogram(values: np.ndarray) -> dict[str, int]:
    arr = np.asarray(values, dtype=np.uint8)
    return {str(symbol): int(np.count_nonzero(arr == symbol)) for symbol in range(5)}


def _candidate_symbol_match_summary(
    candidate_symbols: np.ndarray,
    submitted_symbols: np.ndarray,
    *,
    name: str,
) -> dict[str, Any]:
    candidate = np.asarray(candidate_symbols, dtype=np.uint8)
    submitted = np.asarray(submitted_symbols, dtype=np.uint8)
    if candidate.shape != submitted.shape:
        raise Pr91Hpm1Error(
            "hpm1_symbol_bridge_probe",
            "candidate_submitted_shape_mismatch",
            candidate=name,
            candidate_shape=list(candidate.shape),
            submitted_shape=list(submitted.shape),
        )
    matches = candidate == submitted
    match_count = int(np.count_nonzero(matches))
    return {
        "candidate": name,
        "symbol_count": int(candidate.size),
        "match_count": match_count,
        "mismatch_count": int(candidate.size - match_count),
        "all_symbols_match": bool(candidate.size > 0 and match_count == int(candidate.size)),
        "first_mismatch": _first_symbol_mismatch(
            candidate,
            submitted,
            expected_label="candidate_symbol",
            actual_label="submitted_symbol",
        ),
        "candidate_symbols_sha256": sha256_bytes(np.ascontiguousarray(candidate).tobytes()),
    }


def _summarize_reference_to_submitted_symbol_bridge(
    reference_symbols: list[int] | np.ndarray,
    submitted_symbols: list[int] | np.ndarray,
    previous_reference_symbols: list[int] | np.ndarray,
) -> dict[str, Any]:
    """Classify simple PR85/QMA9 -> PR91 semantic-symbol bridge hypotheses."""

    reference = np.asarray(reference_symbols, dtype=np.uint8)
    submitted = np.asarray(submitted_symbols, dtype=np.uint8)
    previous = np.asarray(previous_reference_symbols, dtype=np.uint8)
    if reference.shape != submitted.shape or reference.shape != previous.shape:
        raise Pr91Hpm1Error(
            "hpm1_symbol_bridge_probe",
            "symbol_sequence_shape_mismatch",
            reference_shape=list(reference.shape),
            submitted_shape=list(submitted.shape),
            previous_shape=list(previous.shape),
        )
    if reference.ndim != 1:
        raise Pr91Hpm1Error(
            "hpm1_symbol_bridge_probe",
            "symbol_sequences_must_be_flat",
            shape=list(reference.shape),
        )
    if int(reference.size) == 0:
        return {
            "schema": "pr91_hpm1_reference_to_submitted_symbol_bridge_v1",
            "status": "no_symbols_available_for_bridge_probe",
            "score_claim": False,
            "dispatch_allowed": False,
            "symbol_count": 0,
            "bridge_found": False,
        }

    identity = _candidate_symbol_match_summary(
        reference,
        submitted,
        name="identity_reference_symbol",
    )
    mod5_offset_rows = [
        _candidate_symbol_match_summary(
            ((reference.astype(np.int16) + offset) % 5).astype(np.uint8),
            submitted,
            name=f"reference_plus_{offset}_mod5",
        )
        for offset in range(5)
    ]
    best_mod5_offset = max(
        mod5_offset_rows,
        key=lambda row: (int(row["match_count"]), -int(row["mismatch_count"])),
    )
    residual_minus = _candidate_symbol_match_summary(
        ((reference.astype(np.int16) - previous.astype(np.int16)) % 5).astype(np.uint8),
        submitted,
        name="reference_minus_previous_mod5",
    )
    residual_plus = _candidate_symbol_match_summary(
        ((reference.astype(np.int16) + previous.astype(np.int16)) % 5).astype(np.uint8),
        submitted,
        name="reference_plus_previous_mod5",
    )

    observed_counts: dict[str, dict[str, int]] = {
        str(symbol): {str(other): 0 for other in range(5)} for symbol in range(5)
    }
    observed_values_by_reference: dict[str, list[int]] = {}
    constraints: dict[int, int] = {}
    conflicts: list[dict[str, Any]] = []
    for ref_value, submitted_value in zip(reference, submitted, strict=True):
        observed_counts[str(int(ref_value))][str(int(submitted_value))] += 1
    for ref_symbol in range(5):
        observed_values = [
            submitted_symbol
            for submitted_symbol, count in observed_counts[str(ref_symbol)].items()
            if int(count) > 0
        ]
        observed_values_int = sorted(int(value) for value in observed_values)
        observed_values_by_reference[str(ref_symbol)] = observed_values_int
        if len(observed_values_int) == 1:
            constraints[ref_symbol] = observed_values_int[0]
        elif len(observed_values_int) > 1:
            conflicts.append(
                {
                    "reference_symbol": ref_symbol,
                    "observed_submitted_symbols": observed_values_int,
                }
            )

    consistent_permutations: list[list[int]] = []
    if not conflicts:
        for perm in permutations(range(5)):
            if all(int(perm[ref_symbol]) == int(submitted_symbol) for ref_symbol, submitted_symbol in constraints.items()):
                consistent_permutations.append([int(value) for value in perm])
    permutation_bridge_found = bool(consistent_permutations)

    if identity["all_symbols_match"]:
        status = "identity_reference_symbols_match_submitted_prefix"
        next_patch = "extend the bounded prefix or attempt byte-exact encoder reconstruction"
        bridge_found = True
    elif permutation_bridge_found:
        status = "global_label_permutation_bridge_matches_submitted_prefix"
        next_patch = "test the surviving label permutation on a deeper prefix before reencode work"
        bridge_found = True
    elif best_mod5_offset["all_symbols_match"]:
        status = "constant_mod5_offset_bridge_matches_submitted_prefix"
        next_patch = "test the mod5 offset on a deeper prefix and then recover encoder range finalization"
        bridge_found = True
    elif residual_minus["all_symbols_match"] or residual_plus["all_symbols_match"]:
        status = "previous_frame_mod5_residual_bridge_matches_submitted_prefix"
        next_patch = "test residual bridge beyond frame-zero seed symbols"
        bridge_found = True
    else:
        status = "no_simple_reference_to_submitted_symbol_bridge_for_prefix"
        next_patch = (
            "recover true PR91 encoder semantic tokens or encoder-side "
            "probability/range trace; the available PR85/QMA9 prefix is not "
            "a simple identity, label-permutation, offset, or residual bridge"
        )
        bridge_found = False

    return {
        "schema": "pr91_hpm1_reference_to_submitted_symbol_bridge_v1",
        "status": status,
        "score_claim": False,
        "dispatch_allowed": False,
        "symbol_count": int(reference.size),
        "bridge_found": bool(bridge_found),
        "reference_symbols": {
            "sha256": sha256_bytes(np.ascontiguousarray(reference).tobytes()),
            "histogram": _symbol_histogram(reference),
            "first_symbols": [int(value) for value in reference[:16].tolist()],
        },
        "submitted_symbols": {
            "sha256": sha256_bytes(np.ascontiguousarray(submitted).tobytes()),
            "histogram": _symbol_histogram(submitted),
            "first_symbols": [int(value) for value in submitted[:16].tolist()],
        },
        "previous_reference_symbols": {
            "sha256": sha256_bytes(np.ascontiguousarray(previous).tobytes()),
            "histogram": _symbol_histogram(previous),
            "first_symbols": [int(value) for value in previous[:16].tolist()],
        },
        "identity_bridge": identity,
        "best_mod5_offset_bridge": best_mod5_offset,
        "mod5_offset_bridges": mod5_offset_rows,
        "previous_frame_residual_bridges": [residual_minus, residual_plus],
        "global_label_permutation_bridge": {
            "status": (
                "conflicting_reference_symbol_mappings"
                if conflicts
                else "consistent_with_observed_prefix"
                if permutation_bridge_found
                else "no_consistent_permutation"
            ),
            "perfect_candidate_found": permutation_bridge_found,
            "consistent_permutation_count": len(consistent_permutations),
            "first_consistent_permutations": consistent_permutations[:5],
            "observed_submitted_symbols_by_reference_symbol": observed_values_by_reference,
            "observed_pair_counts": observed_counts,
            "conflicting_reference_symbols": conflicts,
        },
        "first_identity_mismatch": identity["first_mismatch"],
        "next_patch_target": next_patch,
        "scope_note": (
            "This prefix bridge is a local CPU diagnostic only. It compares "
            "submitted range-decoded seed symbols against available PR85/QMA9 "
            "reference symbols under a requested context/order hypothesis; it "
            "does not prove full HPM1 decode, byte-exact reencode, score "
            "validity, or dispatch readiness."
        ),
    }


def _trace_hpm1_reference_forced_symbol_bridge_prefix(
    model: Any,
    payload: Hpm1MaskPayload,
    reference_tokens: np.ndarray,
    *,
    probability_variant: str | Any,
    prob_eps: float,
    device: str,
    spatial_order_candidate: str,
    symbol_count: int,
    row_preview_limit: int,
    mismatch_limit: int,
) -> dict[str, Any]:
    """Decode a bounded submitted prefix against reference-context rows."""

    if torch is None or F is None:  # pragma: no cover - optional dependency path
        raise Pr91Hpm1Error("dependency_contract", "torch_missing")
    if _pr86_hpac_codec.constriction is None:  # pragma: no cover
        raise Pr91Hpm1Error("dependency_contract", "constriction_missing")
    if str(device) != "cpu":
        raise Pr91Hpm1Error(
            "device_contract",
            "hpm1_symbol_bridge_probe_is_cpu_only",
            requested_device=device,
        )
    _spatial_order_description(spatial_order_candidate)
    if list(reference_tokens.shape) != [payload.n_frames, payload.height, payload.width]:
        raise Pr91Hpm1Error(
            "reference_token_contract",
            "reference_token_shape_mismatch",
            expected=[payload.n_frames, payload.height, payload.width],
            actual=list(reference_tokens.shape),
        )
    if int(symbol_count) <= 0:
        raise Pr91Hpm1Error(
            "hpm1_symbol_bridge_probe",
            "symbol_count_must_be_positive",
            symbol_count=symbol_count,
        )
    if int(symbol_count) > DEFAULT_PR91_HPM1_SYMBOL_BRIDGE_MAX_SYMBOLS:
        raise Pr91Hpm1Error(
            "hpm1_symbol_bridge_probe",
            "symbol_count_exceeds_local_probe_limit",
            symbol_count=symbol_count,
            max_symbols=DEFAULT_PR91_HPM1_SYMBOL_BRIDGE_MAX_SYMBOLS,
        )
    if int(row_preview_limit) < 0 or int(mismatch_limit) < 0:
        raise Pr91Hpm1Error(
            "hpm1_symbol_bridge_probe",
            "row_limits_must_be_nonnegative",
            row_preview_limit=row_preview_limit,
            mismatch_limit=mismatch_limit,
        )

    started_at = time.time()
    resolved = resolve_hpac_probability_variant(probability_variant)
    dev = torch.device(device)
    model = model.to(dev).eval()
    masks = _group_masks(
        payload.height,
        payload.width,
        P=payload.predictor_count,
        delta=payload.delta,
        device=dev,
    )
    submitted_words = _hpm1_token_words_for_candidate(payload, "source_little_uint32")
    decoder = _pr86_hpac_codec.constriction.stream.queue.RangeDecoder(submitted_words)
    decoded_prev = torch.zeros(
        (1, payload.height, payload.width),
        dtype=torch.long,
        device=dev,
    )
    decoded_symbols = 0
    reference_symbols: list[int] = []
    submitted_symbols: list[int] = []
    previous_reference_symbols: list[int] = []
    row_preview: list[dict[str, Any]] = []
    mismatch_rows: list[dict[str, Any]] = []
    failure_report: dict[str, Any] | None = None

    with torch.no_grad():
        for frame in range(payload.n_frames):
            idx = torch.tensor([frame], dtype=torch.long, device=dev)
            cur = torch.zeros(
                (1, payload.height, payload.width),
                dtype=torch.long,
                device=dev,
            )
            for group, mask in enumerate(masks):
                if mask is None:
                    continue
                coords = _hpm1_group_coords_for_spatial_order(
                    payload,
                    group=group,
                    mask=mask,
                    candidate=spatial_order_candidate,
                    device=dev,
                )
                logits = model(cur, idx, decoded_prev)
                probs = F.softmax(logits.float(), dim=1)
                probs_at_group = (
                    probs[0][:, coords[:, 0], coords[:, 1]]
                    .permute(1, 0)
                    .contiguous()
                )
                probs_np = probs_at_group.cpu().numpy()
                coord_rows = coords[:, 0].detach().cpu().numpy()
                coord_cols = coords[:, 1].detach().cpu().numpy()
                ref_at_group = reference_tokens[
                    frame,
                    coord_rows,
                    coord_cols,
                ].astype(np.int64, copy=False)
                prev_at_group = (
                    decoded_prev[0, coords[:, 0], coords[:, 1]]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.int64, copy=False)
                )
                for symbol_in_group, row in enumerate(probs_np):
                    if decoded_symbols >= int(symbol_count):
                        break
                    normalized = _normalize_probability_row(
                        row,
                        prob_eps=prob_eps,
                        variant=resolved,
                    )
                    cat = _categorical_from_probs(
                        row,
                        prob_eps=prob_eps,
                        variant=resolved,
                    )
                    decoder_state_before = _range_decoder_state_summary(
                        decoder,
                        label="before_symbol_bridge_decode",
                    )
                    reference_symbol = int(ref_at_group[symbol_in_group])
                    previous_symbol = int(prev_at_group[symbol_in_group])
                    y = int(coord_rows[symbol_in_group])
                    x = int(coord_cols[symbol_in_group])
                    try:
                        submitted_symbol = int(decoder.decode(cat))
                    except Exception as exc:
                        failure_report = {
                            "status": "range_decode_exception_before_prefix_limit",
                            "passed": False,
                            "exception_type": type(exc).__name__,
                            "exception_text": str(exc),
                            "global_symbol": int(decoded_symbols),
                            "frame": int(frame),
                            "group": int(group),
                            "symbol_in_group": int(symbol_in_group),
                            "pixel_yx": {"y": y, "x": x},
                            "reference_symbol": reference_symbol,
                            "previous_reference_symbol": previous_symbol,
                            "decoder_state_before_exception": decoder_state_before,
                            "decoder_state_after_exception": _range_decoder_state_summary(
                                decoder,
                                label="after_symbol_bridge_decode_exception",
                            ),
                        }
                        break

                    reference_symbols.append(reference_symbol)
                    submitted_symbols.append(submitted_symbol)
                    previous_reference_symbols.append(previous_symbol)
                    order = np.argsort(-normalized)
                    reference_rank = int(np.where(order == reference_symbol)[0][0] + 1)
                    submitted_rank = int(np.where(order == submitted_symbol)[0][0] + 1)
                    row_record = {
                        "global_symbol": int(decoded_symbols),
                        "frame": int(frame),
                        "group": int(group),
                        "symbol_in_group": int(symbol_in_group),
                        "pixel_yx": {"y": y, "x": x},
                        "reference_symbol": reference_symbol,
                        "submitted_symbol": submitted_symbol,
                        "previous_reference_symbol": previous_symbol,
                        "reference_minus_previous_mod5_symbol": int(
                            (reference_symbol - previous_symbol) % 5
                        ),
                        "reference_plus_previous_mod5_symbol": int(
                            (reference_symbol + previous_symbol) % 5
                        ),
                        "matches_reference_symbol": bool(submitted_symbol == reference_symbol),
                        "reference_symbol_probability": round(float(normalized[reference_symbol]), 10),
                        "submitted_symbol_probability": round(float(normalized[submitted_symbol]), 10),
                        "reference_symbol_rank": reference_rank,
                        "submitted_symbol_rank": submitted_rank,
                        "normalized_probability_row_sha256": sha256_bytes(
                            np.ascontiguousarray(normalized).tobytes()
                        ),
                    }
                    if len(row_preview) < int(row_preview_limit):
                        row_preview.append(row_record)
                    if (
                        submitted_symbol != reference_symbol
                        and len(mismatch_rows) < int(mismatch_limit)
                    ):
                        mismatch_rows.append(row_record)
                    decoded_symbols += 1
                if failure_report is not None or decoded_symbols >= int(symbol_count):
                    break
                cur[
                    0,
                    coords[:, 0],
                    coords[:, 1],
                ] = torch.from_numpy(ref_at_group).to(dev)
            if failure_report is not None or decoded_symbols >= int(symbol_count):
                break
            decoded_prev = torch.from_numpy(
                reference_tokens[frame : frame + 1].astype(np.int64, copy=False)
            ).to(dev)

    bridge_summary = _summarize_reference_to_submitted_symbol_bridge(
        reference_symbols,
        submitted_symbols,
        previous_reference_symbols,
    )
    prefix_completed = int(decoded_symbols) == int(symbol_count)
    if failure_report is not None:
        status = "range_decode_exception_before_symbol_bridge_prefix_complete"
    elif bridge_summary.get("bridge_found") is True:
        status = "submitted_prefix_has_simple_reference_symbol_bridge_candidate"
    else:
        status = "submitted_prefix_has_no_simple_reference_symbol_bridge"

    return {
        "schema": "pr91_hpm1_semantic_symbol_bridge_prefix_trace_v1",
        "status": status,
        "passed": bool(prefix_completed and bridge_summary.get("bridge_found") is False),
        "score_claim": False,
        "dispatch_allowed": False,
        "full_decode_proven": False,
        "byte_exact_reencode_proven": False,
        "device": device,
        "probability_variant": resolved.name,
        "prob_eps": float(prob_eps),
        "spatial_order_candidate": spatial_order_candidate,
        "spatial_order_description": _spatial_order_description(spatial_order_candidate),
        "requested_symbol_count": int(symbol_count),
        "decoded_symbol_count": int(decoded_symbols),
        "prefix_completed": bool(prefix_completed),
        "submitted_token_stream": {
            "bytes": len(payload.tokens),
            "sha256": sha256_bytes(payload.tokens),
            **_token_words_summary(submitted_words),
            "word_order_candidate": "source_little_uint32",
        },
        "symbol_sequences": {
            "reference_symbols_sha256": sha256_bytes(
                np.asarray(reference_symbols, dtype=np.uint8).tobytes()
            ),
            "submitted_symbols_sha256": sha256_bytes(
                np.asarray(submitted_symbols, dtype=np.uint8).tobytes()
            ),
            "previous_reference_symbols_sha256": sha256_bytes(
                np.asarray(previous_reference_symbols, dtype=np.uint8).tobytes()
            ),
            "first_reference_symbols": [int(value) for value in reference_symbols[:16]],
            "first_submitted_symbols": [int(value) for value in submitted_symbols[:16]],
            "first_previous_reference_symbols": [
                int(value) for value in previous_reference_symbols[:16]
            ],
            "first_reference_submitted_mismatch": _first_symbol_mismatch(
                np.asarray(reference_symbols, dtype=np.uint8),
                np.asarray(submitted_symbols, dtype=np.uint8),
                expected_label="reference_symbol",
                actual_label="submitted_symbol",
            ),
        },
        "symbol_rows_preview": row_preview,
        "mismatch_rows": mismatch_rows,
        "bridge_summary": bridge_summary,
        "failure": failure_report,
        "scope_note": (
            "This trace uses the submitted PR91 range stream with reference "
            "teacher-forced context rows. It only narrows seed symbol bridge "
            "hypotheses and is not a full decode, reencode, score, or dispatch "
            "artifact."
        ),
        "elapsed_sec": round(time.time() - started_at, 3),
    }


def _hash_probability_row_sequences(
    raw_rows: list[np.ndarray],
    normalized_rows: list[np.ndarray],
    reference_symbols: list[int],
    decoded_symbols: list[int],
    *,
    target_decoded_before: int,
    range_prefix_window_symbols: tuple[int, ...],
) -> dict[str, Any]:
    target = int(target_decoded_before)
    raw_before = np.ascontiguousarray(np.vstack(raw_rows[:target]))
    norm_before = np.ascontiguousarray(np.vstack(normalized_rows[:target]))
    raw_including = np.ascontiguousarray(np.vstack(raw_rows[: target + 1]))
    norm_including = np.ascontiguousarray(np.vstack(normalized_rows[: target + 1]))
    reference_before = np.asarray(reference_symbols[:target], dtype=np.uint8)
    reference_including = np.asarray(reference_symbols[: target + 1], dtype=np.uint8)
    decoded_before = np.asarray(decoded_symbols[:target], dtype=np.uint8)

    windows = []
    for window in sorted({int(v) for v in range_prefix_window_symbols if int(v) > 0}):
        start = max(0, target - window)
        end = min(len(normalized_rows), target + 1)
        norm_window = np.ascontiguousarray(np.vstack(normalized_rows[start:end]))
        ref_window = np.asarray(reference_symbols[start:end], dtype=np.uint8)
        windows.append(
            {
                "symbols_before_failure_window": int(window),
                "start_global_symbol": int(start),
                "end_global_symbol_exclusive": int(end),
                "includes_failure_row": bool(end > target),
                "normalized_rows_sha256": sha256_bytes(norm_window.tobytes()),
                "reference_symbols_sha256": sha256_bytes(ref_window.tobytes()),
            }
        )

    return {
        "schema": "pr91_hpm1_probability_row_sequence_hashes_v1",
        "rows_before_failure": {
            "count": int(raw_before.shape[0]),
            "raw_softmax_sha256": sha256_bytes(raw_before.tobytes()),
            "normalized_for_categorical_sha256": sha256_bytes(norm_before.tobytes()),
            "reference_symbols_sha256": sha256_bytes(reference_before.tobytes()),
            "submitted_decoded_symbols_sha256": sha256_bytes(decoded_before.tobytes()),
        },
        "rows_including_failure": {
            "count": int(raw_including.shape[0]),
            "raw_softmax_sha256": sha256_bytes(raw_including.tobytes()),
            "normalized_for_categorical_sha256": sha256_bytes(
                norm_including.tobytes()
            ),
            "reference_symbols_sha256": sha256_bytes(reference_including.tobytes()),
        },
        "failure_window_hashes": windows,
    }


def _trace_hpm1_reference_forced_range_state_prefix_probe(
    model: Any,
    payload: Hpm1MaskPayload,
    reference_tokens: np.ndarray,
    *,
    probability_variant: str | Any,
    prob_eps: float,
    device: str,
    spatial_order_candidate: str,
    target_failure: Mapping[str, Any],
    range_prefix_window_symbols: tuple[int, ...],
    range_prefix_seed_symbol_counts: tuple[int, ...],
    replay_symbol_limit: int,
) -> dict[str, Any]:
    """Probe probability-row and range-prefix state before one failed row."""

    if torch is None or F is None:  # pragma: no cover - optional dependency path
        raise Pr91Hpm1Error("dependency_contract", "torch_missing")
    if _pr86_hpac_codec.constriction is None:  # pragma: no cover
        raise Pr91Hpm1Error("dependency_contract", "constriction_missing")
    if str(device) != "cpu":
        raise Pr91Hpm1Error(
            "device_contract",
            "hpm1_range_state_prefix_probe_is_cpu_only",
            requested_device=device,
        )
    if int(replay_symbol_limit) < 0:
        raise Pr91Hpm1Error(
            "hpm1_range_state_prefix_probe",
            "replay_symbol_limit_must_be_nonnegative",
            replay_symbol_limit=replay_symbol_limit,
        )

    target_frame = int(target_failure["frame"])
    target_group = int(target_failure["group"])
    target_symbol = int(target_failure["symbol_in_group"])
    target_decoded_before = int(target_failure["decoded_symbol_count_before_failure"])
    resolved = resolve_hpac_probability_variant(probability_variant)
    dev = torch.device(device)
    model = model.to(dev).eval()
    masks = _group_masks(
        payload.height,
        payload.width,
        P=payload.predictor_count,
        delta=payload.delta,
        device=dev,
    )
    submitted_words = _hpm1_token_words_for_candidate(payload, "source_little_uint32")
    decoder = _pr86_hpac_codec.constriction.stream.queue.RangeDecoder(submitted_words)
    decoded_prev = torch.zeros((1, payload.height, payload.width), dtype=torch.long, device=dev)
    decoded_symbols = 0
    target_group_start: int | None = None
    raw_rows: list[np.ndarray] = []
    normalized_rows: list[np.ndarray] = []
    reference_symbols: list[int] = []
    submitted_decoded_symbols: list[int] = []
    reference_mismatch_count = 0
    first_reference_mismatch: dict[str, Any] | None = None
    failure_report: dict[str, Any] | None = None
    started_at = time.time()

    with torch.no_grad():
        for frame in range(target_frame + 1):
            idx = torch.tensor([frame], dtype=torch.long, device=dev)
            cur = torch.zeros((1, payload.height, payload.width), dtype=torch.long, device=dev)
            frame_start_symbols = decoded_symbols
            for group, mask in enumerate(masks):
                if mask is None:
                    continue
                group_start_symbols = decoded_symbols
                coords = _hpm1_group_coords_for_spatial_order(
                    payload,
                    group=group,
                    mask=mask,
                    candidate=spatial_order_candidate,
                    device=dev,
                )
                logits = model(cur, idx, decoded_prev)
                probs = F.softmax(logits.float(), dim=1)
                probs_at_group = (
                    probs[0][:, coords[:, 0], coords[:, 1]]
                    .permute(1, 0)
                    .contiguous()
                )
                probs_np = probs_at_group.cpu().numpy()
                ref_at_group = reference_tokens[
                    frame,
                    coords[:, 0].detach().cpu().numpy(),
                    coords[:, 1].detach().cpu().numpy(),
                ].astype(np.int64, copy=False)
                decoded = np.empty(int(probs_at_group.shape[0]), dtype=np.int64)
                for symbol_in_group, row in enumerate(probs_np):
                    if frame == target_frame and group == target_group:
                        target_group_start = int(group_start_symbols)
                    if (
                        frame > target_frame
                        or (frame == target_frame and group > target_group)
                        or (
                            frame == target_frame
                            and group == target_group
                            and symbol_in_group > target_symbol
                        )
                    ):
                        break
                    normalized = _normalize_probability_row(
                        row,
                        prob_eps=prob_eps,
                        variant=resolved,
                    )
                    raw_rows.append(np.ascontiguousarray(row))
                    normalized_rows.append(np.ascontiguousarray(normalized))
                    reference_symbol = int(ref_at_group[symbol_in_group])
                    reference_symbols.append(reference_symbol)
                    cat = _categorical_from_probs(row, prob_eps=prob_eps, variant=resolved)
                    decoder_state_before = _range_decoder_state_summary(
                        decoder,
                        label="before_decode_attempt",
                    )
                    try:
                        decoded_symbol = int(decoder.decode(cat))
                    except Exception as exc:
                        decoder_state_after = _range_decoder_state_summary(
                            decoder,
                            label="after_failed_decode_exception",
                        )
                        raw_row = np.ascontiguousarray(row)
                        norm_row = np.ascontiguousarray(normalized)
                        failure_report = {
                            "stage": "submitted_tokens_decode",
                            "reason": "hpac_entropy_decode_contract_mismatch",
                            "exception_type": type(exc).__name__,
                            "exception_text": str(exc),
                            "frame": int(frame),
                            "group": int(group),
                            "symbol_in_group": int(symbol_in_group),
                            "decoded_symbol_count_before_failure": int(decoded_symbols),
                            "group_start_decoded_symbols": int(group_start_symbols),
                            "frame_start_decoded_symbols": int(frame_start_symbols),
                            "range_decoder_diagnostic": {
                                "state_before_decode": decoder_state_before,
                                "state_after_failed_decode": decoder_state_after,
                                "not_stream_exhaustion": bool(
                                    decoder_state_before.get("maybe_exhausted") is False
                                ),
                            },
                            "failing_probability_row": {
                                "raw_softmax": {
                                    "dtype": str(raw_row.dtype),
                                    "sha256": sha256_bytes(raw_row.tobytes()),
                                    "values": [
                                        round(float(value), 10)
                                        for value in raw_row.tolist()
                                    ],
                                    "sum": round(float(raw_row.sum()), 10),
                                },
                                "normalized_for_categorical": {
                                    "dtype": str(norm_row.dtype),
                                    "sha256": sha256_bytes(norm_row.tobytes()),
                                    "values": [
                                        round(float(value), 10)
                                        for value in norm_row.tolist()
                                    ],
                                    "sum": round(float(norm_row.sum()), 10),
                                    "argmax_symbol": int(norm_row.argmax()),
                                    "min": round(float(norm_row.min()), 10),
                                    "max": round(float(norm_row.max()), 10),
                                },
                            },
                            "failure_row_interpretation": _failure_row_interpretation(
                                raw_row,
                                norm_row,
                                probability_variant=resolved,
                                prob_eps=prob_eps,
                                decoder_state_before=decoder_state_before,
                                context_mode="reference_teacher_forced_range_state_prefix",
                                reference_symbol=reference_symbol,
                            ),
                        }
                        break
                    submitted_decoded_symbols.append(decoded_symbol)
                    decoded[symbol_in_group] = decoded_symbol
                    if decoded_symbol != reference_symbol:
                        reference_mismatch_count += 1
                        if first_reference_mismatch is None:
                            yx = coords[symbol_in_group].detach().cpu().numpy()
                            first_reference_mismatch = {
                                "global_symbol": int(decoded_symbols),
                                "frame": int(frame),
                                "group": int(group),
                                "symbol_in_group": int(symbol_in_group),
                                "pixel_yx": {"y": int(yx[0]), "x": int(yx[1])},
                                "decoded_symbol": decoded_symbol,
                                "reference_symbol": reference_symbol,
                            }
                    decoded_symbols += 1
                if failure_report is not None:
                    break
                if frame == target_frame and group == target_group:
                    break
                cur[0, coords[:, 0], coords[:, 1]] = torch.from_numpy(
                    ref_at_group
                ).to(dev)
            if failure_report is not None:
                break
            decoded_prev = torch.from_numpy(
                reference_tokens[frame : frame + 1].astype(np.int64, copy=False)
            ).to(dev)

    if failure_report is None:
        return {
            "schema": "pr91_hpm1_range_state_prefix_probe_v1",
            "status": "target_failure_not_reproduced",
            "passed": False,
            "score_claim": False,
            "dispatch_allowed": False,
            "target_failure": dict(target_failure),
            "decoded_symbols_reached": int(decoded_symbols),
            "collected_symbol_count": len(reference_symbols),
            "scope_note": (
                "This probe is only valid when the submitted range failure is "
                "reproduced at the requested target row."
            ),
        }

    observed_failure = {
        key: int(failure_report[key])
        for key in (
            "frame",
            "group",
            "symbol_in_group",
            "decoded_symbol_count_before_failure",
        )
    }
    target_signature = {
        "frame": target_frame,
        "group": target_group,
        "symbol_in_group": target_symbol,
        "decoded_symbol_count_before_failure": target_decoded_before,
    }
    target_reproduced = observed_failure == target_signature
    if target_group_start is None:
        target_group_start = int(failure_report["group_start_decoded_symbols"])
    checkpoints = _prefix_checkpoint_symbol_counts(
        target_decoded_before=target_decoded_before,
        target_group_start=target_group_start,
        total_collected_symbols=len(reference_symbols),
        range_prefix_window_symbols=range_prefix_window_symbols,
        range_prefix_seed_symbol_counts=range_prefix_seed_symbol_counts,
    )
    checkpoint_reports = [
        _build_range_prefix_checkpoint_report(
            raw_rows,
            reference_symbols,
            submitted_words,
            checkpoint,
            probability_variant=resolved,
            prob_eps=prob_eps,
            replay_symbol_limit=replay_symbol_limit,
        )
        for checkpoint in checkpoints
    ]

    return {
        "schema": "pr91_hpm1_range_state_prefix_probe_v1",
        "status": (
            "target_failure_reproduced_with_reference_context"
            if target_reproduced
            else "different_failure_row_observed"
        ),
        "passed": False,
        "score_claim": False,
        "dispatch_allowed": False,
        "full_decode_proven": False,
        "byte_exact_reencode_proven": False,
        "device": device,
        "probability_variant": resolved.name,
        "prob_eps": float(prob_eps),
        "spatial_order_candidate": spatial_order_candidate,
        "target_failure": target_signature,
        "observed_failure": observed_failure,
        "target_failure_reproduced": bool(target_reproduced),
        "target_group_start_decoded_symbols": int(target_group_start),
        "submitted_token_stream": {
            "bytes": len(payload.tokens),
            "sha256": sha256_bytes(payload.tokens),
            **_token_words_summary(submitted_words),
            "word_order_candidate": "source_little_uint32",
        },
        "reference_teacher_forcing": {
            "reference_mismatch_count_before_failure": int(reference_mismatch_count),
            "first_decoded_reference_mismatch": first_reference_mismatch,
        },
        "probability_row_sequence_hashes": _hash_probability_row_sequences(
            raw_rows,
            normalized_rows,
            reference_symbols,
            submitted_decoded_symbols,
            target_decoded_before=target_decoded_before,
            range_prefix_window_symbols=range_prefix_window_symbols,
        ),
        "range_prefix_reconstruction": {
            "schema": "pr91_hpm1_range_prefix_reconstruction_windows_v1",
            "window_symbols_before_failure": [
                int(value)
                for value in range_prefix_window_symbols
                if int(value) > 0
            ],
            "seed_symbol_counts": [
                int(value)
                for value in range_prefix_seed_symbol_counts
                if int(value) > 0
            ],
            "checkpoint_symbol_counts": checkpoints,
            "checkpoint_results": checkpoint_reports,
            "classification": _classify_range_prefix_reconstruction(
                checkpoint_reports
            ),
        },
        "failure": failure_report,
        "scope_note": (
            "Local CPU forensic probe only. It narrows prior context/range-state "
            "hypotheses but does not prove HPM1 decode parity, byte-exact "
            "re-encoding, score validity, or dispatch readiness."
        ),
        "elapsed_sec": round(time.time() - started_at, 3),
    }


def _trace_hpm1_spatial_order_decode_failure(
    model: Any,
    payload: Hpm1MaskPayload,
    *,
    probability_variant: str | Any,
    prob_eps: float,
    max_frames: int | None,
    device: str,
    spatial_order_candidate: str,
) -> dict[str, Any]:
    """Replay HPM1 with one source-context spatial traversal hypothesis."""

    if torch is None or F is None:  # pragma: no cover - optional dependency path
        raise Pr91Hpm1Error("dependency_contract", "torch_missing")
    if _pr86_hpac_codec.constriction is None:  # pragma: no cover
        raise Pr91Hpm1Error("dependency_contract", "constriction_missing")
    if str(device) != "cpu":
        raise Pr91Hpm1Error(
            "device_contract",
            "hpm1_spatial_order_probe_is_cpu_only",
            requested_device=device,
        )

    _spatial_order_description(spatial_order_candidate)
    resolved = resolve_hpac_probability_variant(probability_variant)
    frame_count = int(payload.n_frames if max_frames is None else min(int(max_frames), payload.n_frames))
    if frame_count <= 0:
        raise Pr91Hpm1Error(
            "hpm1_spatial_order_probe",
            "frame_count_must_be_positive",
            max_frames=max_frames,
        )
    dev = torch.device(device)
    model = model.to(dev).eval()
    masks = _group_masks(
        payload.height,
        payload.width,
        P=payload.predictor_count,
        delta=payload.delta,
        device=dev,
    )
    words = _hpm1_token_words_for_candidate(payload, "source_little_uint32")
    decoder = _pr86_hpac_codec.constriction.stream.queue.RangeDecoder(words)
    decoded_prev = torch.zeros((1, payload.height, payload.width), dtype=torch.long, device=dev)
    decoded_symbols = 0
    decoded_frames = np.empty((frame_count, payload.height, payload.width), dtype=np.uint8)
    started_at = time.time()

    with torch.no_grad():
        for frame in range(frame_count):
            idx = torch.tensor([frame], dtype=torch.long, device=dev)
            cur = torch.zeros((1, payload.height, payload.width), dtype=torch.long, device=dev)
            frame_start_symbols = decoded_symbols
            for group, mask in enumerate(masks):
                if mask is None:
                    continue
                group_start_symbols = decoded_symbols
                coords = _hpm1_group_coords_for_spatial_order(
                    payload,
                    group=group,
                    mask=mask,
                    candidate=spatial_order_candidate,
                    device=dev,
                )
                logits = model(cur, idx, decoded_prev)
                probs = F.softmax(logits.float(), dim=1)
                probs_at_group = (
                    probs[0][:, coords[:, 0], coords[:, 1]]
                    .permute(1, 0)
                    .contiguous()
                )
                probs_np = probs_at_group.cpu().numpy()
                decoded = np.empty(int(probs_at_group.shape[0]), dtype=np.int64)
                for symbol_in_group, row in enumerate(probs_np):
                    normalized = _normalize_probability_row(
                        row,
                        prob_eps=prob_eps,
                        variant=resolved,
                    )
                    cat = _categorical_from_probs(row, prob_eps=prob_eps, variant=resolved)
                    decoder_state_before = _range_decoder_state_summary(
                        decoder,
                        label="before_decode_attempt",
                    )
                    try:
                        decoded[symbol_in_group] = decoder.decode(cat)
                    except Exception as exc:
                        decoder_state_after = _range_decoder_state_summary(
                            decoder,
                            label="after_failed_decode_exception",
                        )
                        prefix = decoded[:symbol_in_group].astype(np.uint8, copy=False)
                        raw_row = np.ascontiguousarray(row)
                        norm_row = np.ascontiguousarray(normalized)
                        failure_row_interpretation = _failure_row_interpretation(
                            raw_row,
                            norm_row,
                            probability_variant=resolved,
                            prob_eps=prob_eps,
                            decoder_state_before=decoder_state_before,
                            context_mode="decoded_spatial_order",
                        )
                        return {
                            "status": "failed_at_first_entropy_mismatch",
                            "passed": False,
                            "score_claim": False,
                            "dispatch_allowed": False,
                            "full_decode_proven": False,
                            "byte_exact_reencode_proven": False,
                            "device": device,
                            "probability_variant": resolved.name,
                            "prob_eps": float(prob_eps),
                            "spatial_order_candidate": spatial_order_candidate,
                            "spatial_order_description": _spatial_order_description(
                                spatial_order_candidate
                            ),
                            "failure": {
                                "stage": "submitted_tokens_decode",
                                "reason": "hpac_entropy_decode_contract_mismatch",
                                "exception_type": type(exc).__name__,
                                "frame": int(frame),
                                "group": int(group),
                                "symbol_in_group": int(symbol_in_group),
                                "decoded_symbol_count_before_failure": int(decoded_symbols),
                                "group_start_decoded_symbols": int(group_start_symbols),
                                "frame_start_decoded_symbols": int(frame_start_symbols),
                            },
                            "range_decoder_diagnostic": {
                                "state_before_decode": decoder_state_before,
                                "state_after_failed_decode": decoder_state_after,
                                "exception_text": str(exc),
                                "not_stream_exhaustion": bool(
                                    decoder_state_before.get("maybe_exhausted") is False
                                ),
                            },
                            "token_stream": {
                                "bytes": len(payload.tokens),
                                "sha256": sha256_bytes(payload.tokens),
                                **_token_words_summary(words),
                                "word_order_candidate": "source_little_uint32",
                            },
                            "group_geometry": {
                                "symbols_in_group": int(probs_at_group.shape[0]),
                                "mask_true_count": int(mask.sum().item()),
                                "mask_sha256": sha256_bytes(
                                    mask.cpu().numpy().astype(np.uint8).tobytes()
                                ),
                                "coords": _coords_summary(coords),
                            },
                            "failing_probability_row": {
                                "raw_softmax": {
                                    "dtype": str(raw_row.dtype),
                                    "sha256": sha256_bytes(raw_row.tobytes()),
                                    "values": [
                                        round(float(value), 10)
                                        for value in raw_row.tolist()
                                    ],
                                    "sum": round(float(raw_row.sum()), 10),
                                },
                                "normalized_for_categorical": {
                                    "dtype": str(norm_row.dtype),
                                    "sha256": sha256_bytes(norm_row.tobytes()),
                                    "values": [
                                        round(float(value), 10)
                                        for value in norm_row.tolist()
                                    ],
                                    "sum": round(float(norm_row.sum()), 10),
                                    "argmax_symbol": int(norm_row.argmax()),
                                    "min": round(float(norm_row.min()), 10),
                                    "max": round(float(norm_row.max()), 10),
                                },
                            },
                            "failure_row_interpretation": failure_row_interpretation,
                            "decoded_prefix_in_failing_group": {
                                "symbol_count": int(prefix.size),
                                "sha256": sha256_bytes(prefix.tobytes()),
                                **_small_int_preview(prefix),
                            },
                            "context_before_failing_group": {
                                "current_frame": _context_tensor_summary(
                                    cur[0],
                                    label="current_frame_tokens_before_group_assignment",
                                ),
                                "previous_frame": _context_tensor_summary(
                                    decoded_prev[0],
                                    label="previous_frame_tokens",
                                ),
                            },
                            "source_context_contract": {
                                "source_loop": (
                                    "group mask -> probabilities at selected "
                                    "positions -> decode symbols -> assign the "
                                    "group into cur"
                                ),
                                "candidate_changes": (
                                    "only the selected-position traversal order "
                                    "within each PR91 patch group"
                                ),
                            },
                            "elapsed_sec": round(time.time() - started_at, 3),
                        }
                    decoded_symbols += 1
                cur[0, coords[:, 0], coords[:, 1]] = torch.from_numpy(decoded).to(dev)
            decoded_frames[frame] = cur[0].cpu().numpy().astype(np.uint8)
            decoded_prev = cur.clone()

    return {
        "status": "passed_requested_prefix_decode",
        "passed": True,
        "score_claim": False,
        "dispatch_allowed": False,
        "device": device,
        "probability_variant": resolved.name,
        "prob_eps": float(prob_eps),
        "spatial_order_candidate": spatial_order_candidate,
        "spatial_order_description": _spatial_order_description(spatial_order_candidate),
        "decoded_frames": int(decoded_frames.shape[0]),
        "decoded_symbols": int(decoded_symbols),
        "decoded_tokens": {
            "shape": list(decoded_frames.shape),
            "sha256": sha256_bytes(decoded_frames.tobytes()),
        },
        "full_decode_proven": max_frames is None and int(decoded_frames.shape[0]) == payload.n_frames,
        "byte_exact_reencode_proven": False,
        "elapsed_sec": round(time.time() - started_at, 3),
    }


def _trace_hpm1_submitted_prefix_token_recovery(
    model: Any,
    payload: Hpm1MaskPayload,
    reference_tokens: np.ndarray | None,
    *,
    probability_variant: str | Any,
    prob_eps: float,
    device: str,
    spatial_order_candidate: str,
    max_symbols: int | None,
    row_preview_limit: int,
    mismatch_limit: int,
) -> dict[str, Any]:
    """Recover deterministic submitted symbols until the first range failure."""

    if torch is None or F is None:  # pragma: no cover - optional dependency path
        raise Pr91Hpm1Error("dependency_contract", "torch_missing")
    if _pr86_hpac_codec.constriction is None:  # pragma: no cover
        raise Pr91Hpm1Error("dependency_contract", "constriction_missing")
    if str(device) != "cpu":
        raise Pr91Hpm1Error(
            "device_contract",
            "hpm1_submitted_prefix_token_recovery_is_cpu_only",
            requested_device=device,
        )
    if max_symbols is not None and int(max_symbols) <= 0:
        raise Pr91Hpm1Error(
            "hpm1_submitted_prefix_token_recovery",
            "max_symbols_must_be_positive",
            max_symbols=max_symbols,
        )
    if int(row_preview_limit) < 0 or int(mismatch_limit) < 0:
        raise Pr91Hpm1Error(
            "hpm1_submitted_prefix_token_recovery",
            "row_limits_must_be_nonnegative",
            row_preview_limit=row_preview_limit,
            mismatch_limit=mismatch_limit,
        )
    if reference_tokens is not None and list(reference_tokens.shape) != [
        payload.n_frames,
        payload.height,
        payload.width,
    ]:
        raise Pr91Hpm1Error(
            "reference_token_contract",
            "reference_token_shape_mismatch",
            expected=[payload.n_frames, payload.height, payload.width],
            actual=list(reference_tokens.shape),
        )
    _spatial_order_description(spatial_order_candidate)

    started_at = time.time()
    resolved = resolve_hpac_probability_variant(probability_variant)
    dev = torch.device(device)
    model = model.to(dev).eval()
    masks = _group_masks(
        payload.height,
        payload.width,
        P=payload.predictor_count,
        delta=payload.delta,
        device=dev,
    )
    words = _hpm1_token_words_for_candidate(payload, "source_little_uint32")
    decoder = _pr86_hpac_codec.constriction.stream.queue.RangeDecoder(words)
    decoded_prev = torch.zeros((1, payload.height, payload.width), dtype=torch.long, device=dev)
    decoded_symbols = 0
    recovered_symbols: list[int] = []
    reference_symbols: list[int] = []
    row_preview: list[dict[str, Any]] = []
    mismatch_rows: list[dict[str, Any]] = []
    reference_mismatch_count = 0
    raw_rows_hasher = hashlib.sha256()
    normalized_rows_hasher = hashlib.sha256()
    coords_hasher = hashlib.sha256()
    failure_report: dict[str, Any] | None = None
    hit_symbol_limit = False

    with torch.no_grad():
        for frame in range(payload.n_frames):
            idx = torch.tensor([frame], dtype=torch.long, device=dev)
            cur = torch.zeros((1, payload.height, payload.width), dtype=torch.long, device=dev)
            frame_start_symbols = decoded_symbols
            for group, mask in enumerate(masks):
                if mask is None:
                    continue
                group_start_symbols = decoded_symbols
                coords = _hpm1_group_coords_for_spatial_order(
                    payload,
                    group=group,
                    mask=mask,
                    candidate=spatial_order_candidate,
                    device=dev,
                )
                coord_rows = coords[:, 0].detach().cpu().numpy()
                coord_cols = coords[:, 1].detach().cpu().numpy()
                logits = model(cur, idx, decoded_prev)
                probs = F.softmax(logits.float(), dim=1)
                probs_at_group = (
                    probs[0][:, coords[:, 0], coords[:, 1]]
                    .permute(1, 0)
                    .contiguous()
                )
                probs_np = probs_at_group.cpu().numpy()
                decoded = np.empty(int(probs_at_group.shape[0]), dtype=np.int64)
                for symbol_in_group, row in enumerate(probs_np):
                    if max_symbols is not None and decoded_symbols >= int(max_symbols):
                        hit_symbol_limit = True
                        break
                    raw_row = np.ascontiguousarray(row)
                    normalized = np.ascontiguousarray(
                        _normalize_probability_row(
                            raw_row,
                            prob_eps=prob_eps,
                            variant=resolved,
                        )
                    )
                    raw_rows_hasher.update(raw_row.tobytes())
                    normalized_rows_hasher.update(normalized.tobytes())
                    y = int(coord_rows[symbol_in_group])
                    x = int(coord_cols[symbol_in_group])
                    coords_hasher.update(
                        np.asarray([frame, group, symbol_in_group, y, x], dtype="<i4").tobytes()
                    )
                    cat = _categorical_from_probs(raw_row, prob_eps=prob_eps, variant=resolved)
                    decoder_state_before = _range_decoder_state_summary(
                        decoder,
                        label="before_submitted_prefix_decode",
                    )
                    try:
                        decoded_symbol = int(decoder.decode(cat))
                    except Exception as exc:
                        failure_report = {
                            "stage": "submitted_tokens_decode",
                            "reason": "hpac_entropy_decode_contract_mismatch",
                            "exception_type": type(exc).__name__,
                            "exception_text": str(exc),
                            "frame": int(frame),
                            "group": int(group),
                            "symbol_in_group": int(symbol_in_group),
                            "decoded_symbol_count_before_failure": int(decoded_symbols),
                            "group_start_decoded_symbols": int(group_start_symbols),
                            "frame_start_decoded_symbols": int(frame_start_symbols),
                            "pixel_yx": {"y": y, "x": x},
                            "range_decoder_diagnostic": {
                                "state_before_decode": decoder_state_before,
                                "state_after_failed_decode": _range_decoder_state_summary(
                                    decoder,
                                    label="after_submitted_prefix_decode_exception",
                                ),
                                "not_stream_exhaustion": bool(
                                    decoder_state_before.get("maybe_exhausted") is False
                                ),
                            },
                            "failing_probability_row": {
                                "raw_softmax_sha256": sha256_bytes(raw_row.tobytes()),
                                "normalized_sha256": sha256_bytes(normalized.tobytes()),
                                "normalized_values": [
                                    round(float(value), 10)
                                    for value in normalized.tolist()
                                ],
                                "argmax_symbol": int(normalized.argmax()),
                            },
                        }
                        break

                    decoded[symbol_in_group] = decoded_symbol
                    recovered_symbols.append(decoded_symbol)
                    reference_symbol: int | None = None
                    if reference_tokens is not None:
                        reference_symbol = int(reference_tokens[frame, y, x])
                        reference_symbols.append(reference_symbol)
                        if decoded_symbol != reference_symbol:
                            reference_mismatch_count += 1
                    order = np.argsort(-normalized, kind="stable")
                    record = {
                        "global_symbol": int(decoded_symbols),
                        "frame": int(frame),
                        "group": int(group),
                        "symbol_in_group": int(symbol_in_group),
                        "pixel_yx": {"y": y, "x": x},
                        "submitted_symbol": int(decoded_symbol),
                        "submitted_symbol_probability": round(
                            float(normalized[decoded_symbol]),
                            10,
                        ),
                        "submitted_symbol_rank": int(
                            np.where(order == decoded_symbol)[0][0] + 1
                        ),
                        "argmax_symbol": int(order[0]),
                        "normalized_probability_row_sha256": sha256_bytes(
                            normalized.tobytes()
                        ),
                    }
                    if reference_symbol is not None:
                        record.update(
                            {
                                "reference_symbol": reference_symbol,
                                "matches_reference_symbol": bool(
                                    decoded_symbol == reference_symbol
                                ),
                            }
                        )
                    if len(row_preview) < int(row_preview_limit):
                        row_preview.append(record)
                    if (
                        reference_symbol is not None
                        and decoded_symbol != reference_symbol
                        and len(mismatch_rows) < int(mismatch_limit)
                    ):
                        mismatch_rows.append(record)
                    decoded_symbols += 1
                if failure_report is not None or hit_symbol_limit:
                    break
                cur[0, coords[:, 0], coords[:, 1]] = torch.from_numpy(decoded).to(dev)
            if failure_report is not None or hit_symbol_limit:
                break
            decoded_prev = cur.clone()

    recovered_arr = np.asarray(recovered_symbols, dtype=np.uint8)
    reference_arr = np.asarray(reference_symbols, dtype=np.uint8)
    status = (
        "recovered_requested_submitted_prefix"
        if hit_symbol_limit
        else "recovered_prefix_until_first_entropy_failure"
        if failure_report is not None
        else "recovered_full_submitted_stream"
    )
    return {
        "schema": "pr91_hpm1_submitted_prefix_token_recovery_trace_v1",
        "status": status,
        "passed": bool(recovered_arr.size > 0),
        "score_claim": False,
        "dispatch_allowed": False,
        "full_decode_proven": bool(failure_report is None and not hit_symbol_limit),
        "byte_exact_reencode_proven": False,
        "device": device,
        "probability_variant": resolved.name,
        "prob_eps": float(prob_eps),
        "spatial_order_candidate": spatial_order_candidate,
        "spatial_order_description": _spatial_order_description(spatial_order_candidate),
        "source_loop": (
            "public PR91 source mask row-major decoded-context HPAC loop"
            if spatial_order_candidate == "source_mask_row_major"
            else "off-contract spatial-order hypothesis over submitted PR91 range words"
        ),
        "max_symbols": None if max_symbols is None else int(max_symbols),
        "decoded_symbol_count": int(recovered_arr.size),
        "submitted_symbols": {
            "sha256": sha256_bytes(recovered_arr.tobytes()),
            **_small_int_preview(recovered_arr),
        },
        "probability_row_trace": {
            "raw_softmax_rows_sha256": raw_rows_hasher.hexdigest(),
            "normalized_rows_sha256": normalized_rows_hasher.hexdigest(),
            "row_count": int(recovered_arr.size + (1 if failure_report else 0)),
        },
        "coordinate_trace": {
            "schema": "frame_group_symbol_yx_int32_sequence_v1",
            "sha256": coords_hasher.hexdigest(),
        },
        "reference_comparison": {
            "attempted": reference_tokens is not None,
            "reference_symbols_sha256": (
                sha256_bytes(reference_arr.tobytes())
                if reference_tokens is not None
                else ""
            ),
            "reference_mismatch_count": int(reference_mismatch_count),
            "first_mismatch": (
                _first_symbol_mismatch(
                    reference_arr,
                    recovered_arr,
                    expected_label="reference_symbol",
                    actual_label="submitted_symbol",
                )
                if reference_tokens is not None
                else None
            ),
            "mismatch_rows": mismatch_rows,
        },
        "row_preview": row_preview,
        "failure": failure_report,
        "scope_note": (
            "Recovered symbols are the deterministic local CPU prefix decoded "
            "from submitted PR91 range words under the public source HPAC "
            "probability loop. This is forensic token/trace evidence only, not "
            "a full decode, byte-exact reencode, score, or dispatch artifact."
        ),
        "elapsed_sec": round(time.time() - started_at, 3),
    }


def _trace_hpm1_reference_teacher_forced_spatial_order_failure(
    model: Any,
    payload: Hpm1MaskPayload,
    reference_tokens: np.ndarray,
    *,
    probability_variant: str | Any,
    prob_eps: float,
    max_frames: int | None,
    device: str,
    spatial_order_candidate: str,
    reference_window_before: int = 2,
    reference_window_after: int = 5,
) -> dict[str, Any]:
    """Replay HPM1 while forcing HPAC context from reference mask tokens."""

    if torch is None or F is None:  # pragma: no cover - optional dependency path
        raise Pr91Hpm1Error("dependency_contract", "torch_missing")
    if _pr86_hpac_codec.constriction is None:  # pragma: no cover
        raise Pr91Hpm1Error("dependency_contract", "constriction_missing")
    if str(device) != "cpu":
        raise Pr91Hpm1Error(
            "device_contract",
            "hpm1_reference_teacher_forcing_probe_is_cpu_only",
            requested_device=device,
        )

    _spatial_order_description(spatial_order_candidate)
    if list(reference_tokens.shape) != [payload.n_frames, payload.height, payload.width]:
        raise Pr91Hpm1Error(
            "reference_token_contract",
            "reference_token_shape_mismatch",
            expected=[payload.n_frames, payload.height, payload.width],
            actual=list(reference_tokens.shape),
        )
    if int(reference_window_before) < 0 or int(reference_window_after) < 0:
        raise Pr91Hpm1Error(
            "reference_teacher_forcing_probe",
            "reference_window_counts_must_be_nonnegative",
            reference_window_before=reference_window_before,
            reference_window_after=reference_window_after,
        )
    resolved = resolve_hpac_probability_variant(probability_variant)
    frame_count = int(payload.n_frames if max_frames is None else min(int(max_frames), payload.n_frames))
    if frame_count <= 0:
        raise Pr91Hpm1Error(
            "hpm1_reference_teacher_forcing_probe",
            "frame_count_must_be_positive",
            max_frames=max_frames,
        )
    dev = torch.device(device)
    model = model.to(dev).eval()
    masks = _group_masks(
        payload.height,
        payload.width,
        P=payload.predictor_count,
        delta=payload.delta,
        device=dev,
    )
    words = _hpm1_token_words_for_candidate(payload, "source_little_uint32")
    decoder = _pr86_hpac_codec.constriction.stream.queue.RangeDecoder(words)
    decoded_prev = torch.zeros((1, payload.height, payload.width), dtype=torch.long, device=dev)
    decoded_symbols = 0
    first_reference_mismatch: dict[str, Any] | None = None
    reference_mismatch_count = 0
    started_at = time.time()

    with torch.no_grad():
        for frame in range(frame_count):
            idx = torch.tensor([frame], dtype=torch.long, device=dev)
            cur = torch.zeros((1, payload.height, payload.width), dtype=torch.long, device=dev)
            frame_start_symbols = decoded_symbols
            for group, mask in enumerate(masks):
                if mask is None:
                    continue
                group_start_symbols = decoded_symbols
                coords = _hpm1_group_coords_for_spatial_order(
                    payload,
                    group=group,
                    mask=mask,
                    candidate=spatial_order_candidate,
                    device=dev,
                )
                logits = model(cur, idx, decoded_prev)
                probs = F.softmax(logits.float(), dim=1)
                probs_at_group = (
                    probs[0][:, coords[:, 0], coords[:, 1]]
                    .permute(1, 0)
                    .contiguous()
                )
                probs_np = probs_at_group.cpu().numpy()
                ref_at_group = reference_tokens[
                    frame,
                    coords[:, 0].detach().cpu().numpy(),
                    coords[:, 1].detach().cpu().numpy(),
                ].astype(np.int64, copy=False)
                decoded = np.empty(int(probs_at_group.shape[0]), dtype=np.int64)
                for symbol_in_group, row in enumerate(probs_np):
                    normalized = _normalize_probability_row(
                        row,
                        prob_eps=prob_eps,
                        variant=resolved,
                    )
                    cat = _categorical_from_probs(row, prob_eps=prob_eps, variant=resolved)
                    decoder_state_before = _range_decoder_state_summary(
                        decoder,
                        label="before_decode_attempt",
                    )
                    try:
                        decoded_symbol = int(decoder.decode(cat))
                    except Exception as exc:
                        decoder_state_after = _range_decoder_state_summary(
                            decoder,
                            label="after_failed_decode_exception",
                        )
                        prefix = decoded[:symbol_in_group].astype(np.uint8, copy=False)
                        raw_row = np.ascontiguousarray(row)
                        norm_row = np.ascontiguousarray(normalized)
                        reference_symbol = int(ref_at_group[symbol_in_group])
                        failure_row_interpretation = _failure_row_interpretation(
                            raw_row,
                            norm_row,
                            probability_variant=resolved,
                            prob_eps=prob_eps,
                            decoder_state_before=decoder_state_before,
                            context_mode="reference_teacher_forced",
                            reference_symbol=reference_symbol,
                        )
                        return {
                            "status": "failed_at_first_entropy_mismatch",
                            "passed": False,
                            "score_claim": False,
                            "dispatch_allowed": False,
                            "full_decode_proven": False,
                            "byte_exact_reencode_proven": False,
                            "device": device,
                            "probability_variant": resolved.name,
                            "prob_eps": float(prob_eps),
                            "context_mode": "reference_teacher_forced",
                            "spatial_order_candidate": spatial_order_candidate,
                            "spatial_order_description": _spatial_order_description(
                                spatial_order_candidate
                            ),
                            "failure": {
                                "stage": "submitted_tokens_decode",
                                "reason": "hpac_entropy_decode_contract_mismatch",
                                "exception_type": type(exc).__name__,
                                "frame": int(frame),
                                "group": int(group),
                                "symbol_in_group": int(symbol_in_group),
                                "decoded_symbol_count_before_failure": int(decoded_symbols),
                                "group_start_decoded_symbols": int(group_start_symbols),
                                "frame_start_decoded_symbols": int(frame_start_symbols),
                            },
                            "range_decoder_diagnostic": {
                                "state_before_decode": decoder_state_before,
                                "state_after_failed_decode": decoder_state_after,
                                "exception_text": str(exc),
                                "not_stream_exhaustion": bool(
                                    decoder_state_before.get("maybe_exhausted") is False
                                ),
                            },
                            "token_stream": {
                                "bytes": len(payload.tokens),
                                "sha256": sha256_bytes(payload.tokens),
                                **_token_words_summary(words),
                                "word_order_candidate": "source_little_uint32",
                            },
                            "group_geometry": {
                                "symbols_in_group": int(probs_at_group.shape[0]),
                                "mask_true_count": int(mask.sum().item()),
                                "mask_sha256": sha256_bytes(
                                    mask.cpu().numpy().astype(np.uint8).tobytes()
                                ),
                                "coords": _coords_summary(coords),
                            },
                            "failing_probability_row": {
                                "raw_softmax": {
                                    "dtype": str(raw_row.dtype),
                                    "sha256": sha256_bytes(raw_row.tobytes()),
                                    "values": [
                                        round(float(value), 10)
                                        for value in raw_row.tolist()
                                    ],
                                    "sum": round(float(raw_row.sum()), 10),
                                },
                                "normalized_for_categorical": {
                                    "dtype": str(norm_row.dtype),
                                    "sha256": sha256_bytes(norm_row.tobytes()),
                                    "values": [
                                        round(float(value), 10)
                                        for value in norm_row.tolist()
                                    ],
                                    "sum": round(float(norm_row.sum()), 10),
                                    "argmax_symbol": int(norm_row.argmax()),
                                    "min": round(float(norm_row.min()), 10),
                                    "max": round(float(norm_row.max()), 10),
                                },
                            },
                            "failure_row_interpretation": failure_row_interpretation,
                            "decoded_prefix_in_failing_group": {
                                "symbol_count": int(prefix.size),
                                "sha256": sha256_bytes(prefix.tobytes()),
                                **_small_int_preview(prefix),
                            },
                            "canonical_reference_symbol_window": _reference_group_symbol_window(
                                ref_at_group,
                                coords,
                                frame=frame,
                                group=group,
                                failure_symbol_in_group=symbol_in_group,
                                window_before=reference_window_before,
                                window_after=reference_window_after,
                            ),
                            "reference_teacher_forcing": {
                                "policy": (
                                    "after each complete HPAC group, current-frame "
                                    "context is assigned from reference tokens; "
                                    "previous-frame context is reference frame f-1"
                                ),
                                "reference_mismatch_count_before_failure": int(
                                    reference_mismatch_count
                                ),
                                "first_decoded_reference_mismatch": first_reference_mismatch,
                            },
                            "context_before_failing_group": {
                                "current_frame": _context_tensor_summary(
                                    cur[0],
                                    label="current_frame_reference_context_before_group",
                                ),
                                "previous_frame": _context_tensor_summary(
                                    decoded_prev[0],
                                    label="previous_frame_reference_context",
                                ),
                            },
                            "elapsed_sec": round(time.time() - started_at, 3),
                        }
                    decoded[symbol_in_group] = decoded_symbol
                    reference_symbol = int(ref_at_group[symbol_in_group])
                    if decoded_symbol != reference_symbol:
                        reference_mismatch_count += 1
                        if first_reference_mismatch is None:
                            yx = coords[symbol_in_group].detach().cpu().numpy()
                            first_reference_mismatch = {
                                "global_symbol": int(decoded_symbols),
                                "frame": int(frame),
                                "group": int(group),
                                "symbol_in_group": int(symbol_in_group),
                                "pixel_yx": {"y": int(yx[0]), "x": int(yx[1])},
                                "decoded_symbol": decoded_symbol,
                                "reference_symbol": reference_symbol,
                            }
                    decoded_symbols += 1
                cur[
                    0,
                    coords[:, 0],
                    coords[:, 1],
                ] = torch.from_numpy(ref_at_group).to(dev)
            decoded_prev = torch.from_numpy(
                reference_tokens[frame : frame + 1].astype(np.int64, copy=False)
            ).to(dev)

    return {
        "status": "completed_requested_reference_forced_prefix_decode",
        "passed": True,
        "score_claim": False,
        "dispatch_allowed": False,
        "device": device,
        "probability_variant": resolved.name,
        "prob_eps": float(prob_eps),
        "context_mode": "reference_teacher_forced",
        "spatial_order_candidate": spatial_order_candidate,
        "spatial_order_description": _spatial_order_description(spatial_order_candidate),
        "decoded_frames": int(frame_count),
        "decoded_symbols": int(decoded_symbols),
        "requested_frame_prefix_completed": True,
        "all_frames_requested": max_frames is None and int(frame_count) == payload.n_frames,
        "reference_teacher_forcing": {
            "reference_mismatch_count_before_failure": int(reference_mismatch_count),
            "first_decoded_reference_mismatch": first_reference_mismatch,
        },
        "full_decode_proven": False,
        "full_decode_note": (
            "Reference teacher-forced context is an off-contract semantic probe; "
            "even a completed requested prefix is not standalone HPM1 full decode proof."
        ),
        "byte_exact_reencode_proven": False,
        "elapsed_sec": round(time.time() - started_at, 3),
    }


def _run_hpm1_spatial_group_order_probe(
    model: Any,
    payload: Hpm1MaskPayload,
    *,
    probability_variant: str | Any,
    prob_eps: float,
    max_frames: int | None,
    device: str,
    candidates: tuple[str, ...] = PR91_HPM1_SPATIAL_ORDER_CANDIDATES,
) -> dict[str, Any]:
    """Classify whether the mismatch is a simple spatial traversal mismatch."""

    started_at = time.time()
    requested = _validate_spatial_order_candidates(candidates)
    expected_source_signature = {
        "frame": PR91_HPM1_SOURCE_FAILURE_FRAME,
        "group": PR91_HPM1_SOURCE_FAILURE_GROUP,
        "symbol_in_group": PR91_HPM1_SOURCE_FAILURE_SYMBOL_IN_GROUP,
        "decoded_symbol_count_before_failure": PR91_HPM1_SOURCE_FAILURE_DECODED_BEFORE,
    }
    candidate_results: list[dict[str, Any]] = []
    non_source_past_source_failure: list[str] = []
    source_order_reproduces = False
    source_decoded_before = PR91_HPM1_SOURCE_FAILURE_DECODED_BEFORE
    best_non_source_decoded = -1

    for candidate in requested:
        trace = _trace_hpm1_spatial_order_decode_failure(
            model,
            payload,
            probability_variant=probability_variant,
            prob_eps=prob_eps,
            max_frames=max_frames,
            device=device,
            spatial_order_candidate=candidate,
        )
        signature = _failure_row_signature(trace)
        decoded_before = (
            int(signature["decoded_symbol_count_before_failure"])
            if signature is not None
            else int(trace.get("decoded_symbols", 0))
        )
        if candidate == "source_mask_row_major":
            source_order_reproduces = signature == expected_source_signature
            if signature is not None:
                source_decoded_before = int(signature["decoded_symbol_count_before_failure"])
        elif trace.get("passed") is True or decoded_before > source_decoded_before:
            non_source_past_source_failure.append(candidate)
        if candidate != "source_mask_row_major":
            best_non_source_decoded = max(best_non_source_decoded, decoded_before)

        candidate_results.append(
            {
                "candidate": candidate,
                "description": _spatial_order_description(candidate),
                "status": trace.get("status"),
                "passed": bool(trace.get("passed") is True),
                "failure_signature": signature,
                "decoded_symbols_or_before_failure": decoded_before,
                "passes_source_failure_row": (
                    bool(trace.get("passed") is True)
                    or decoded_before > source_decoded_before
                ),
                "exception_type": (
                    trace.get("failure", {}).get("exception_type")
                    if isinstance(trace.get("failure"), Mapping)
                    else None
                ),
                "failing_probability_row_sha256": (
                    trace.get("failing_probability_row", {})
                    .get("normalized_for_categorical", {})
                    .get("sha256")
                    if isinstance(trace.get("failing_probability_row"), Mapping)
                    else None
                ),
                "group_coords_sha256": (
                    trace.get("group_geometry", {})
                    .get("coords", {})
                    .get("sha256")
                    if isinstance(trace.get("group_geometry"), Mapping)
                    else None
                ),
                "elapsed_sec": trace.get("elapsed_sec"),
            }
        )

    narrowed = source_order_reproduces and not non_source_past_source_failure
    status = (
        "not_explained_by_spatial_group_traversal_order"
        if narrowed
        else "spatial_group_order_hypothesis_still_open"
    )
    return {
        "schema": "pr91_hpm1_spatial_group_order_probe_v1",
        "status": status,
        "passed": bool(narrowed),
        "score_claim": False,
        "dispatch_allowed": False,
        "device": device,
        "probability_variant": resolve_hpac_probability_variant(probability_variant).name,
        "prob_eps": float(prob_eps),
        "max_frames": max_frames,
        "expected_source_failure_signature": expected_source_signature,
        "source_order_reproduces_exact_failure_row": source_order_reproduces,
        "non_source_candidates_passing_source_failure_row": non_source_past_source_failure,
        "best_non_source_decoded_symbols_or_before_failure": (
            None if best_non_source_decoded < 0 else best_non_source_decoded
        ),
        "candidate_results": candidate_results,
        "narrowed_hypothesis": (
            "The PR91 HPM1 mismatch is not explained by source-adjacent "
            "full-grid, tile-major, or phase-major spatial traversal order "
            "for HPAC patch-group symbols."
            if narrowed
            else "At least one non-source spatial traversal candidate decoded past the source failure row; inspect that candidate before testing broader probability grammar."
        ),
        "elapsed_sec": round(time.time() - started_at, 3),
    }


def _classify_reference_teacher_forcing_hypotheses(
    candidate_results: list[dict[str, Any]],
    *,
    advanced_candidates: list[str],
    source_decoded_before: int,
) -> dict[str, Any]:
    """Summarize which HPM1 failure classes remain after teacher forcing."""

    best_row: dict[str, Any] | None = None
    best_reference_before = -1
    for row in candidate_results:
        reference_context = row.get("reference_teacher_forced_context")
        if not isinstance(reference_context, Mapping):
            continue
        decoded_before = int(
            reference_context.get("decoded_symbols_or_before_failure") or -1
        )
        if decoded_before > best_reference_before:
            best_reference_before = decoded_before
            best_row = row

    best_candidate = best_row.get("candidate") if best_row is not None else None
    best_reference_context = (
        best_row.get("reference_teacher_forced_context", {})
        if best_row is not None
        else {}
    )
    best_failure = (
        best_reference_context.get("failure_signature")
        if isinstance(best_reference_context, Mapping)
        else None
    )
    best_interpretation = (
        best_reference_context.get("failure_row_interpretation")
        if isinstance(best_reference_context, Mapping)
        and isinstance(best_reference_context.get("failure_row_interpretation"), Mapping)
        else {}
    )
    single_row_roundtrip = (
        best_interpretation.get("single_row_range_model_roundtrip", {})
        if isinstance(best_interpretation, Mapping)
        else {}
    )
    current_row_not_semantic = (
        best_interpretation.get("status")
        == "not_explained_by_current_row_reference_symbol_probability"
    )
    local_row_roundtrip_passed = (
        single_row_roundtrip.get("all_symbols_roundtrip") is True
    )
    decoder_not_exhausted = (
        best_interpretation.get("decoder_not_stream_exhaustion") is True
        if isinstance(best_interpretation, Mapping)
        else False
    )

    if advanced_candidates:
        row_ordering_status = "phase_major_advances_but_does_not_decode_full_prefix"
        row_ordering_note = (
            "At least one off-source spatial order advances beyond source order, "
            "but the best advanced candidate still fails before completing frame 0."
        )
    else:
        row_ordering_status = "not_explained_by_requested_row_ordering_scope"
        row_ordering_note = (
            "No requested row-order candidate advanced under available reference "
            "teacher forcing."
        )
    semantic_status = (
        "not_explained_by_current_row_available_reference_symbol"
        if current_row_not_semantic
        else "semantic_token_interpretation_still_open"
    )
    range_status = (
        "still_open_prior_range_state_or_encoder_finalization_contract"
        if local_row_roundtrip_passed and decoder_not_exhausted
        else "still_open_range_contract_inconclusive"
    )
    probability_status = "still_open_prior_context_or_probability_numeric_contract"
    overall_status = (
        "narrowed_to_range_or_probability_context_grammar_after_phase_major_reference_row"
        if advanced_candidates and current_row_not_semantic and local_row_roundtrip_passed
        else "hpm1_entropy_grammar_classes_still_open"
    )

    return {
        "schema": "pr91_hpm1_decode_failure_hypothesis_classification_v1",
        "status": overall_status,
        "score_claim": False,
        "dispatch_allowed": False,
        "best_reference_teacher_forced_candidate": best_candidate,
        "best_reference_teacher_forced_decoded_symbols_or_before_failure": (
            None if best_reference_before < 0 else best_reference_before
        ),
        "best_reference_teacher_forced_failure_signature": best_failure,
        "source_decoded_symbols_or_before_failure": int(source_decoded_before),
        "row_ordering": {
            "status": row_ordering_status,
            "advanced_candidates": list(advanced_candidates),
            "note": row_ordering_note,
        },
        "semantic_token_interpretation": {
            "status": semantic_status,
            "available_reference_symbol": (
                best_interpretation.get("reference_symbol")
                if isinstance(best_interpretation, Mapping)
                else None
            ),
            "available_reference_symbol_probability": (
                best_interpretation.get("reference_symbol_probability")
                if isinstance(best_interpretation, Mapping)
                else None
            ),
            "available_reference_symbol_rank": (
                best_interpretation.get("reference_symbol_rank")
                if isinstance(best_interpretation, Mapping)
                else None
            ),
            "true_pr91_encoder_tokens_still_open": True,
        },
        "range_coder_contract": {
            "status": range_status,
            "decoder_not_stream_exhaustion": bool(decoder_not_exhausted),
            "single_row_local_roundtrip_all_symbols": bool(
                local_row_roundtrip_passed
            ),
            "note": (
                "The failed submitted stream state is not reproduced by a "
                "single-row local model failure; prior range state, finalization, "
                "or encoder-side construction remains open."
            ),
        },
        "probability_context_grammar": {
            "status": probability_status,
            "note": (
                "The public runtime gives no encoder-side probability trace, so "
                "context drift or numeric probability grammar before the failing "
                "row remains a fail-closed blocker."
            ),
        },
        "fail_closed_blockers": [
            "full_600_frame_hpm1_decode_not_proven",
            "byte_exact_hpm1_reencode_not_proven",
            "true_pr91_encoder_semantic_tokens_not_recovered",
            "range_coder_full_stream_contract_not_recovered",
        ],
    }


def _summarize_reference_teacher_forcing_candidate_progress(
    candidate_results: list[dict[str, Any]],
    *,
    source_decoded_before: int,
) -> dict[str, Any]:
    """Contrast decoded-context and reference-forced progress by candidate."""

    rows: list[dict[str, Any]] = []
    advancing: list[str] = []
    regressing: list[str] = []
    for row in candidate_results:
        candidate = str(row.get("candidate", ""))
        decoded_context = row.get("decoded_context", {})
        reference_context = row.get("reference_teacher_forced_context", {})
        if not isinstance(decoded_context, Mapping) or not isinstance(
            reference_context,
            Mapping,
        ):
            continue
        decoded_before = int(
            decoded_context.get("decoded_symbols_or_before_failure") or 0
        )
        reference_before = int(
            reference_context.get("decoded_symbols_or_before_failure") or 0
        )
        delta_vs_decoded = reference_before - decoded_before
        delta_vs_source = reference_before - int(source_decoded_before)
        if delta_vs_decoded > 0:
            progress_status = "reference_forcing_advances_candidate"
            advancing.append(candidate)
        elif delta_vs_decoded < 0:
            progress_status = "reference_forcing_regresses_candidate"
            regressing.append(candidate)
        else:
            progress_status = "reference_forcing_same_prefix_as_decoded_context"
        rows.append(
            {
                "candidate": candidate,
                "status": progress_status,
                "decoded_context_symbols_before_failure": decoded_before,
                "reference_forced_symbols_before_failure": reference_before,
                "delta_reference_vs_decoded": int(delta_vs_decoded),
                "delta_reference_vs_source": int(delta_vs_source),
                "decoded_context_failure_signature": decoded_context.get(
                    "failure_signature"
                ),
                "reference_forced_failure_signature": reference_context.get(
                    "failure_signature"
                ),
                "first_decoded_reference_mismatch": reference_context.get(
                    "first_decoded_reference_mismatch"
                ),
            }
        )

    if advancing:
        status = "phase_major_reference_forcing_remains_live_if_present"
        next_patch = (
            "prioritize the advancing candidate; do not use regressing "
            "teacher-forced rows as byte-reencode evidence"
        )
    elif regressing:
        status = "reference_forcing_only_regresses_requested_candidates"
        next_patch = (
            "leave this reference context path fail-closed and recover true "
            "PR91 encoder grammar"
        )
    else:
        status = "reference_forcing_candidate_progress_inconclusive"
        next_patch = "add narrower candidate rows or true PR91 encoder-token evidence"

    return {
        "schema": "pr91_hpm1_reference_teacher_forcing_candidate_progress_v1",
        "status": status,
        "score_claim": False,
        "dispatch_allowed": False,
        "source_decoded_symbols_or_before_failure": int(source_decoded_before),
        "advancing_candidates": advancing,
        "regressing_candidates": regressing,
        "candidate_rows": rows,
        "next_patch_target": next_patch,
        "scope_note": (
            "This is a local CPU candidate-progress diagnostic only. It does "
            "not prove full decode, byte-exact reencode, score validity, or "
            "dispatch readiness."
        ),
    }


def _trace_hpm1_entropy_decode_failure(
    model: Any,
    payload: Hpm1MaskPayload,
    *,
    probability_variant: str | Any,
    prob_eps: float,
    max_frames: int | None,
    device: str,
    token_word_order_candidate: str = "source_little_uint32",
    token_words: np.ndarray | None = None,
) -> dict[str, Any]:
    """Replay HPM1 entropy decode until the first exact grammar failure."""

    if torch is None or F is None:  # pragma: no cover - optional dependency path
        raise Pr91Hpm1Error("dependency_contract", "torch_missing")
    if _pr86_hpac_codec.constriction is None:  # pragma: no cover
        raise Pr91Hpm1Error("dependency_contract", "constriction_missing")
    if str(device) != "cpu":
        raise Pr91Hpm1Error(
            "device_contract",
            "hpm1_entropy_failure_probe_is_cpu_only",
            requested_device=device,
        )

    resolved = resolve_hpac_probability_variant(probability_variant)
    frame_count = int(payload.n_frames if max_frames is None else min(int(max_frames), payload.n_frames))
    if frame_count <= 0:
        raise Pr91Hpm1Error(
            "hpm1_entropy_failure_probe",
            "frame_count_must_be_positive",
            max_frames=max_frames,
        )
    dev = torch.device(device)
    model = model.to(dev).eval()
    masks = _group_masks(
        payload.height,
        payload.width,
        P=payload.predictor_count,
        delta=payload.delta,
        device=dev,
    )
    words = (
        _hpm1_token_words_for_candidate(payload, token_word_order_candidate)
        if token_words is None
        else np.ascontiguousarray(token_words, dtype=np.uint32)
    )
    decoder = _pr86_hpac_codec.constriction.stream.queue.RangeDecoder(words)
    decoded_prev = torch.zeros((1, payload.height, payload.width), dtype=torch.long, device=dev)
    decoded_symbols = 0
    decoded_frames = np.empty((frame_count, payload.height, payload.width), dtype=np.uint8)
    started_at = time.time()

    with torch.no_grad():
        for frame in range(frame_count):
            idx = torch.tensor([frame], dtype=torch.long, device=dev)
            cur = torch.zeros((1, payload.height, payload.width), dtype=torch.long, device=dev)
            frame_start_symbols = decoded_symbols
            for group, mask in enumerate(masks):
                if mask is None:
                    continue
                group_start_symbols = decoded_symbols
                logits = model(cur, idx, decoded_prev)
                probs = F.softmax(logits.float(), dim=1)
                probs_at_group = probs[0][:, mask].permute(1, 0).contiguous()
                probs_np = probs_at_group.cpu().numpy()
                decoded = np.empty(int(probs_at_group.shape[0]), dtype=np.int64)
                for symbol_in_group, row in enumerate(probs_np):
                    normalized = _normalize_probability_row(
                        row,
                        prob_eps=prob_eps,
                        variant=resolved,
                    )
                    cat = _categorical_from_probs(row, prob_eps=prob_eps, variant=resolved)
                    decoder_state_before = _range_decoder_state_summary(
                        decoder,
                        label="before_decode_attempt",
                    )
                    try:
                        decoded[symbol_in_group] = decoder.decode(cat)
                    except Exception as exc:
                        decoder_state_after = _range_decoder_state_summary(
                            decoder,
                            label="after_failed_decode_exception",
                        )
                        prefix = decoded[:symbol_in_group].astype(np.uint8, copy=False)
                        raw_row = np.ascontiguousarray(row)
                        norm_row = np.ascontiguousarray(normalized)
                        failure_row_interpretation = _failure_row_interpretation(
                            raw_row,
                            norm_row,
                            probability_variant=resolved,
                            prob_eps=prob_eps,
                            decoder_state_before=decoder_state_before,
                            context_mode="source_entropy_decode",
                        )
                        return {
                            "status": "failed_at_first_entropy_mismatch",
                            "passed": False,
                            "score_claim": False,
                            "dispatch_allowed": False,
                            "device": device,
                            "probability_variant": resolved.name,
                            "prob_eps": float(prob_eps),
                            "token_word_order_candidate": token_word_order_candidate,
                            "failure": {
                                "stage": "submitted_tokens_decode",
                                "reason": "hpac_entropy_decode_contract_mismatch",
                                "exception_type": type(exc).__name__,
                                "frame": int(frame),
                                "group": int(group),
                                "symbol_in_group": int(symbol_in_group),
                                "decoded_symbol_count_before_failure": int(decoded_symbols),
                                "group_start_decoded_symbols": int(group_start_symbols),
                                "frame_start_decoded_symbols": int(frame_start_symbols),
                            },
                            "range_decoder_diagnostic": {
                                "state_before_decode": decoder_state_before,
                                "state_after_failed_decode": decoder_state_after,
                                "exception_text": str(exc),
                                "not_stream_exhaustion": bool(
                                    decoder_state_before.get("maybe_exhausted") is False
                                ),
                            },
                            "token_stream": {
                                "bytes": len(payload.tokens),
                                "sha256": sha256_bytes(payload.tokens),
                                **_token_words_summary(words),
                                "word_order_candidate": token_word_order_candidate,
                            },
                            "group_geometry": {
                                "symbols_in_group": int(probs_at_group.shape[0]),
                                "mask_true_count": int(mask.sum().item()),
                                "mask_sha256": sha256_bytes(mask.cpu().numpy().astype(np.uint8).tobytes()),
                            },
                            "failing_probability_row": {
                                "raw_softmax": {
                                    "dtype": str(raw_row.dtype),
                                    "sha256": sha256_bytes(raw_row.tobytes()),
                                    "values": [
                                        round(float(value), 10) for value in raw_row.tolist()
                                    ],
                                    "sum": round(float(raw_row.sum()), 10),
                                },
                                "normalized_for_categorical": {
                                    "dtype": str(norm_row.dtype),
                                    "sha256": sha256_bytes(norm_row.tobytes()),
                                    "values": [
                                        round(float(value), 10) for value in norm_row.tolist()
                                    ],
                                    "sum": round(float(norm_row.sum()), 10),
                                    "argmax_symbol": int(norm_row.argmax()),
                                    "min": round(float(norm_row.min()), 10),
                                    "max": round(float(norm_row.max()), 10),
                                },
                                "categorical_perfect": bool(resolved.categorical_perfect),
                                "source_contract": bool(resolved.source_contract),
                            },
                            "failure_row_interpretation": failure_row_interpretation,
                            "decoded_prefix_in_failing_group": {
                                "symbol_count": int(prefix.size),
                                "sha256": sha256_bytes(prefix.tobytes()),
                                **_small_int_preview(prefix),
                            },
                            "context_before_failing_group": {
                                "current_frame": _context_tensor_summary(
                                    cur[0],
                                    label="current_frame_tokens_before_group_assignment",
                                ),
                                "previous_frame": _context_tensor_summary(
                                    decoded_prev[0],
                                    label="previous_frame_tokens",
                                ),
                            },
                            "source_equivalent_decoder_contract": {
                                "range_decoder": "constriction.stream.queue.RangeDecoder(uint32_words)",
                                "probability_rows": (
                                    "F.softmax(logits.float(), dim=1) -> numpy float64 "
                                    "clip/renormalize -> Categorical(perfect=False)"
                                ),
                                "group_assignment": (
                                    "decode complete patch group, then assign cur[mask]; "
                                    "no in-group context mutation"
                                ),
                            },
                            "elapsed_sec": round(time.time() - started_at, 3),
                        }
                    decoded_symbols += 1
                cur[0, mask] = torch.from_numpy(decoded).to(dev)
            decoded_frames[frame] = cur[0].cpu().numpy().astype(np.uint8)
            decoded_prev = cur.clone()

    return {
        "status": "passed_requested_prefix_decode",
        "passed": True,
        "score_claim": False,
        "dispatch_allowed": False,
        "device": device,
        "probability_variant": resolved.name,
        "prob_eps": float(prob_eps),
        "token_word_order_candidate": token_word_order_candidate,
        "decoded_frames": int(decoded_frames.shape[0]),
        "decoded_symbols": int(decoded_symbols),
        "decoded_tokens": {
            "shape": list(decoded_frames.shape),
            "sha256": sha256_bytes(decoded_frames.tobytes()),
        },
        "full_decode_proven": max_frames is None and int(decoded_frames.shape[0]) == payload.n_frames,
        "byte_exact_reencode_proven": False,
        "elapsed_sec": round(time.time() - started_at, 3),
    }


def _run_hpm1_token_word_order_probe(
    model: Any,
    payload: Hpm1MaskPayload,
    *,
    probability_variant: str | Any,
    prob_eps: float,
    max_frames: int | None,
    device: str,
    source_trace: Mapping[str, Any] | None = None,
    candidates: tuple[str, ...] = PR91_HPM1_TOKEN_WORD_ORDER_CANDIDATES,
) -> dict[str, Any]:
    """Classify whether the first mismatch is a simple uint32 queue transform."""

    started_at = time.time()
    seen_word_shas: dict[str, str] = {}
    source_signature = _failure_row_signature(source_trace or {})
    candidate_results: list[dict[str, Any]] = []
    matching_source_failure_rows: list[str] = []
    non_source_matching_source_failure_rows: list[str] = []

    for candidate in candidates:
        words = _hpm1_token_words_for_candidate(payload, candidate)
        word_summary = _token_words_summary(words)
        duplicate_of = seen_word_shas.get(word_summary["decoder_words_sha256"])
        if duplicate_of is not None:
            row = {
                "candidate": candidate,
                "status": "not_run_duplicate_decoder_words",
                "duplicate_of": duplicate_of,
                "passed": False,
                "word_summary": word_summary,
            }
            if source_signature is not None and duplicate_of == "source_little_uint32":
                row["matches_source_failure_row"] = True
                matching_source_failure_rows.append(candidate)
            candidate_results.append(row)
            continue
        seen_word_shas[word_summary["decoder_words_sha256"]] = candidate

        if candidate == "source_little_uint32" and source_trace is not None:
            trace = dict(source_trace)
        else:
            trace = _trace_hpm1_entropy_decode_failure(
                model,
                payload,
                probability_variant=probability_variant,
                prob_eps=prob_eps,
                max_frames=max_frames,
                device=device,
                token_word_order_candidate=candidate,
                token_words=words,
            )
        signature = _failure_row_signature(trace)
        matches_source = bool(source_signature is not None and signature == source_signature)
        if matches_source:
            matching_source_failure_rows.append(candidate)
            if candidate != "source_little_uint32":
                non_source_matching_source_failure_rows.append(candidate)
        candidate_results.append(
            {
                "candidate": candidate,
                "status": trace.get("status"),
                "passed": bool(trace.get("passed") is True),
                "word_summary": word_summary,
                "failure_signature": signature,
                "matches_source_failure_row": matches_source,
                "exception_type": (
                    trace.get("failure", {}).get("exception_type")
                    if isinstance(trace.get("failure"), Mapping)
                    else None
                ),
                "decoded_symbols": trace.get("decoded_symbols"),
                "elapsed_sec": trace.get("elapsed_sec"),
            }
        )

    source_reproduced = "source_little_uint32" in matching_source_failure_rows
    narrowed = source_reproduced and not non_source_matching_source_failure_rows
    return {
        "schema": "pr91_hpm1_token_word_order_probe_v1",
        "status": (
            "not_explained_by_uint32_endian_or_word_reversal"
            if narrowed
            else "word_order_hypothesis_still_open"
        ),
        "passed": bool(narrowed),
        "score_claim": False,
        "dispatch_allowed": False,
        "device": device,
        "probability_variant": resolve_hpac_probability_variant(probability_variant).name,
        "prob_eps": float(prob_eps),
        "max_frames": max_frames,
        "source_failure_signature": source_signature,
        "source_little_reproduces_exact_failure_row": source_reproduced,
        "non_source_candidates_matching_source_failure_row": (
            non_source_matching_source_failure_rows
        ),
        "matching_source_failure_row_candidates": matching_source_failure_rows,
        "candidate_results": candidate_results,
        "narrowed_hypothesis": (
            "The PR91 HPM1 mismatch is not explained by simple uint32 endian "
            "swapping, native-vs-little dtype selection, or whole-word queue "
            "reversal; the remaining gap is semantic probability/range grammar "
            "or encoder-side context construction."
            if narrowed
            else "At least one non-source uint32 queue transform matched the source failure row; investigate token queue orientation further."
        ),
        "elapsed_sec": round(time.time() - started_at, 3),
    }


def _trace_hpm1_in_group_prefix_context_at_source_failure(
    model: Any,
    payload: Hpm1MaskPayload,
    *,
    probability_variant: str | Any,
    prob_eps: float,
    device: str,
    source_trace: Mapping[str, Any],
) -> dict[str, Any]:
    """Test whether same-group serial context clears the PR91 source failure.

    The public source decodes a whole patch group against one probability
    tensor, then writes the decoded symbols into ``cur``.  This bounded probe
    replays the source loop only until the known first failure, then recomputes
    the failing probability row after assigning the already-decoded symbols in
    that same group into ``cur``.  It does not attempt a full serial decode.
    """

    started_at = time.time()
    if torch is None or F is None:  # pragma: no cover - optional dependency path
        raise Pr91Hpm1Error("dependency_contract", "torch_missing")
    if _pr86_hpac_codec.constriction is None:  # pragma: no cover
        raise Pr91Hpm1Error("dependency_contract", "constriction_missing")
    if str(device) != "cpu":
        raise Pr91Hpm1Error(
            "device_contract",
            "hpm1_in_group_context_probe_is_cpu_only",
            requested_device=device,
        )

    source_failure = source_trace.get("failure")
    if not isinstance(source_failure, Mapping):
        return {
            "schema": "pr91_hpm1_in_group_context_update_probe_v1",
            "status": "not_attempted_missing_source_failure_trace",
            "passed": False,
            "score_claim": False,
            "dispatch_allowed": False,
            "reason": "source_trace_did_not_record_a_failure",
        }

    target_frame = int(source_failure.get("frame", PR91_HPM1_SOURCE_FAILURE_FRAME))
    target_group = int(source_failure.get("group", PR91_HPM1_SOURCE_FAILURE_GROUP))
    target_symbol = int(
        source_failure.get(
            "symbol_in_group",
            PR91_HPM1_SOURCE_FAILURE_SYMBOL_IN_GROUP,
        )
    )
    target_decoded_before = int(
        source_failure.get(
            "decoded_symbol_count_before_failure",
            PR91_HPM1_SOURCE_FAILURE_DECODED_BEFORE,
        )
    )
    resolved = resolve_hpac_probability_variant(probability_variant)
    dev = torch.device(device)
    model = model.to(dev).eval()
    masks = _group_masks(
        payload.height,
        payload.width,
        P=payload.predictor_count,
        delta=payload.delta,
        device=dev,
    )
    words = _hpm1_token_words_for_candidate(payload, "source_little_uint32")
    decoder = _pr86_hpac_codec.constriction.stream.queue.RangeDecoder(words)
    decoded_prev = torch.zeros(
        (1, payload.height, payload.width),
        dtype=torch.long,
        device=dev,
    )
    decoded_symbols = 0

    with torch.no_grad():
        for frame in range(target_frame + 1):
            idx = torch.tensor([frame], dtype=torch.long, device=dev)
            cur = torch.zeros(
                (1, payload.height, payload.width),
                dtype=torch.long,
                device=dev,
            )
            for group, mask in enumerate(masks):
                if mask is None:
                    continue
                cur_before_group = cur.clone()
                logits = model(cur, idx, decoded_prev)
                probs = F.softmax(logits.float(), dim=1)
                probs_at_group = probs[0][:, mask].permute(1, 0).contiguous()
                coords = mask.nonzero(as_tuple=False)
                decoded = np.empty(int(probs_at_group.shape[0]), dtype=np.int64)
                for symbol_in_group, row_tensor in enumerate(probs_at_group):
                    row = row_tensor.cpu().numpy()
                    if (
                        frame == target_frame
                        and group == target_group
                        and symbol_in_group == target_symbol
                    ):
                        prefix = decoded[:symbol_in_group].astype(np.uint8, copy=False)
                        serial_cur = cur_before_group.clone()
                        if symbol_in_group:
                            prefix_tensor = torch.from_numpy(
                                decoded[:symbol_in_group]
                            ).to(dev)
                            serial_cur[
                                0,
                                coords[:symbol_in_group, 0],
                                coords[:symbol_in_group, 1],
                            ] = prefix_tensor
                        serial_logits = model(serial_cur, idx, decoded_prev)
                        serial_raw_row = (
                            F.softmax(serial_logits.float(), dim=1)[
                                0,
                                :,
                                coords[symbol_in_group, 0],
                                coords[symbol_in_group, 1],
                            ]
                            .cpu()
                            .numpy()
                        )
                        source_norm = _normalize_probability_row(
                            row,
                            prob_eps=prob_eps,
                            variant=resolved,
                        )
                        serial_norm = _normalize_probability_row(
                            serial_raw_row,
                            prob_eps=prob_eps,
                            variant=resolved,
                        )
                        serial_decode: dict[str, Any]
                        try:
                            serial_symbol = decoder.decode(
                                _categorical_from_probs(
                                    serial_raw_row,
                                    prob_eps=prob_eps,
                                    variant=resolved,
                                )
                            )
                            serial_decode = {
                                "status": "passed_source_failure_symbol",
                                "passed": True,
                                "decoded_symbol": int(serial_symbol),
                                "exception_type": "",
                            }
                        except Exception as exc:
                            serial_decode = {
                                "status": "failed_at_source_failure_symbol",
                                "passed": False,
                                "exception_type": type(exc).__name__,
                                "reason": "serial_prefix_context_still_hits_range_decoder_assertion",
                            }
                        source_sha = sha256_bytes(
                            np.ascontiguousarray(source_norm).tobytes()
                        )
                        serial_sha = sha256_bytes(
                            np.ascontiguousarray(serial_norm).tobytes()
                        )
                        serial_cleared_failure = serial_decode["passed"] is True
                        return {
                            "schema": "pr91_hpm1_in_group_context_update_probe_v1",
                            "status": (
                                "serial_prefix_context_changes_failure_symbol"
                                if serial_cleared_failure
                                else "not_explained_by_serial_in_group_prefix_context"
                            ),
                            "passed": bool(not serial_cleared_failure),
                            "score_claim": False,
                            "dispatch_allowed": False,
                            "device": device,
                            "probability_variant": resolved.name,
                            "prob_eps": float(prob_eps),
                            "target_failure": {
                                "frame": target_frame,
                                "group": target_group,
                                "symbol_in_group": target_symbol,
                                "decoded_symbol_count_before_failure": target_decoded_before,
                            },
                            "replayed_to_target": {
                                "decoded_symbol_count_before_target": int(decoded_symbols),
                                "matches_source_decoded_before": (
                                    int(decoded_symbols) == target_decoded_before
                                ),
                            },
                            "source_batch_row": {
                                "sha256": source_sha,
                                "values": [
                                    round(float(value), 10)
                                    for value in source_norm.tolist()
                                ],
                            },
                            "serial_prefix_row": {
                                "sha256": serial_sha,
                                "values": [
                                    round(float(value), 10)
                                    for value in serial_norm.tolist()
                                ],
                            },
                            "row_comparison": {
                                "rows_equal": bool(np.array_equal(source_norm, serial_norm)),
                                "max_abs_probability_delta": round(
                                    float(np.max(np.abs(source_norm - serial_norm))),
                                    12,
                                ),
                                "source_argmax_symbol": int(source_norm.argmax()),
                                "serial_argmax_symbol": int(serial_norm.argmax()),
                            },
                            "serial_prefix_context": {
                                "assigned_prior_symbols_in_group": int(prefix.size),
                                "prefix_sha256": sha256_bytes(prefix.tobytes()),
                                **_small_int_preview(prefix),
                                "current_frame_before_group": _context_tensor_summary(
                                    cur_before_group[0],
                                    label="current_frame_before_failing_group_assignment",
                                ),
                                "current_frame_with_serial_prefix": _context_tensor_summary(
                                    serial_cur[0],
                                    label="current_frame_with_prior_symbols_in_failing_group",
                                ),
                            },
                            "serial_prefix_decode": serial_decode,
                            "narrowed_hypothesis": (
                                "Assigning prior decoded symbols from the failing "
                                "patch group into the current-frame context changes "
                                "the failing probability row only slightly and does "
                                "not clear the RangeDecoder assertion."
                                if not serial_cleared_failure
                                else "Same-group serial prefix context clears the exact source failure symbol; investigate full serial context replay before further probability variants."
                            ),
                            "elapsed_sec": round(time.time() - started_at, 3),
                        }

                    cat = _categorical_from_probs(row, prob_eps=prob_eps, variant=resolved)
                    try:
                        decoded[symbol_in_group] = decoder.decode(cat)
                    except Exception as exc:
                        return {
                            "schema": "pr91_hpm1_in_group_context_update_probe_v1",
                            "status": "failed_before_target_failure",
                            "passed": False,
                            "score_claim": False,
                            "dispatch_allowed": False,
                            "device": device,
                            "probability_variant": resolved.name,
                            "prob_eps": float(prob_eps),
                            "unexpected_failure": {
                                "frame": int(frame),
                                "group": int(group),
                                "symbol_in_group": int(symbol_in_group),
                                "decoded_symbol_count_before_failure": int(decoded_symbols),
                                "exception_type": type(exc).__name__,
                            },
                            "target_failure": {
                                "frame": target_frame,
                                "group": target_group,
                                "symbol_in_group": target_symbol,
                                "decoded_symbol_count_before_failure": target_decoded_before,
                            },
                            "elapsed_sec": round(time.time() - started_at, 3),
                        }
                    decoded_symbols += 1
                cur[0, mask] = torch.from_numpy(decoded).to(dev)
            decoded_prev = cur.clone()

    return {
        "schema": "pr91_hpm1_in_group_context_update_probe_v1",
        "status": "target_failure_not_reached",
        "passed": False,
        "score_claim": False,
        "dispatch_allowed": False,
        "device": device,
        "probability_variant": resolved.name,
        "prob_eps": float(prob_eps),
        "target_failure": {
            "frame": target_frame,
            "group": target_group,
            "symbol_in_group": target_symbol,
            "decoded_symbol_count_before_failure": target_decoded_before,
        },
        "decoded_symbol_count": int(decoded_symbols),
        "elapsed_sec": round(time.time() - started_at, 3),
    }


def run_pr91_hpm1_entropy_failure_grammar_probe(
    archive: Path = DEFAULT_PR91_ARCHIVE,
    *,
    device: str = "cpu",
    probability_variant: str = DEFAULT_HPAC_PROBABILITY_VARIANT,
    prob_eps: float = PROB_EPS,
    max_frames: int | None = 1,
    include_word_order_probe: bool = True,
    output_dir: Path | None = None,
    strict: bool = False,
    write_json: bool = True,
) -> dict[str, Any]:
    """Pin the exact PR91/HPM1 range-decoder grammar blocker.

    The probe is local-only. It reproduces the submitted runtime entropy loop
    until the first constriction mismatch and records enough context to keep
    future fixes from guessing at the missing grammar.
    """

    started_at = time.time()
    if str(device) != "cpu":
        raise Pr91Hpm1Error(
            "device_contract",
            "pr91_hpm1_entropy_failure_grammar_probe_is_cpu_only",
            requested_device=device,
        )
    archive_path = Path(archive)
    payload = extract_pr91_hpm1_payload(archive_path)
    dependency_report = collect_dependency_report(strict=False)
    static_report = validate_hpm1_static_contract(payload)
    relationship = compare_hpm1_to_pr86_hpac_contract(payload)
    runtime_sources = analyze_pr91_hpm1_runtime_sources()
    model = load_hpm1_hpac_model(payload, device=device)
    trace = _trace_hpm1_entropy_decode_failure(
        model,
        payload,
        probability_variant=probability_variant,
        prob_eps=prob_eps,
        max_frames=max_frames,
        device=device,
    )
    blocked = trace.get("passed") is not True
    word_order_probe = (
        _run_hpm1_token_word_order_probe(
            model,
            payload,
            probability_variant=probability_variant,
            prob_eps=prob_eps,
            max_frames=max_frames,
            device=device,
            source_trace=trace,
        )
        if include_word_order_probe
        else {
            "schema": "pr91_hpm1_token_word_order_probe_v1",
            "status": "not_attempted_by_request",
            "passed": False,
            "score_claim": False,
            "dispatch_allowed": False,
        }
    )
    report: dict[str, Any] = {
        "schema": "pr91_hpm1_entropy_failure_grammar_probe_v1",
        "tool": "tac.pr91_hpm1_codec.run_pr91_hpm1_entropy_failure_grammar_probe",
        "recorded_at_utc": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "status": (
            "blocked_hpm1_entropy_decoder_grammar_mismatch"
            if blocked
            else "passed_requested_prefix_decode_reencode_still_missing"
        ),
        "score_claim": False,
        "dispatch_allowed": False,
        "dispatch_performed": False,
        "gpu_or_remote_work": False,
        "local_only": True,
        "ready_for_exact_eval_dispatch": False,
        "promotion_eligible": False,
        "evidence_grade": "local_entropy_failure_grammar_probe",
        "device": device,
        "archive": _archive_report(archive_path),
        "hpm1_static_contract": static_report,
        "pr86_hpac_relationship": relationship,
        "dependency_report": dependency_report,
        "runtime_source_contract": runtime_sources,
        "payload": {
            "config": payload.config(),
            "tokens_bytes": len(payload.tokens),
            "tokens_sha256": sha256_bytes(payload.tokens),
            "hpac_bytes": len(payload.hpac),
            "hpac_sha256": sha256_bytes(payload.hpac),
        },
        "entropy_failure_trace": trace,
        "token_word_order_probe": word_order_probe,
        "exact_missing_grammar": {
            "status": "identified_as_entropy_decode_contract_gap" if blocked else "not_triggered_in_requested_prefix",
            "missing_wire_contract": (
                "semantic HPAC probability/range grammar that maps the exact "
                "HPM1 uint32 token queue to 600x384x512 class tokens and back "
                "to the same token bytes"
            ),
            "not_missing": [
                "archive/member byte custody",
                "HPM1 header/token/model section split",
                "PPMd HPAC torch state load",
                "submitted runtime source dependency versions",
                "first probability-row construction",
            ],
            "first_blocked_operation": (
                trace.get("failure", {}).get("stage", "")
                if isinstance(trace.get("failure"), dict)
                else ""
            ),
        },
        "full_decode": {
            "passed": False,
            "frame_count": 0,
            "decoded_masks_sha256": "",
            "refusal_reason": "full_600_frame_hpm1_decode_not_proven",
        },
        "byte_exact_semantic_reencode": {
            "passed": False,
            "byte_exact": False,
            "reencoded_hpm1_sha256": "",
            "refusal_reason": "range_encoder_uint32_reemit_not_proven",
        },
        "next_required_proofs": [
            "repair or recover the encoder-side HPAC probability/range grammar at the traced failure row",
            "decode all 600 HPM1 frames from the exact PR91 token stream on CPU",
            "range-encode decoded symbols back to the exact token stream SHA-256",
            "prove byte-exact HPM1 segment reencode from semantic tokens",
        ],
        "elapsed_sec": round(time.time() - started_at, 3),
    }
    if write_json and output_dir is not None:
        write_json_report(report, Path(output_dir) / "hpm1_entropy_failure_grammar_probe.json")
    if strict and report["ready_for_exact_eval_dispatch"] is not True:
        raise Pr91Hpm1Error("hpm1_entropy_failure_grammar_probe", str(report["status"]), report=report)
    return _jsonable(report)


def run_pr91_hpm1_in_group_context_update_probe(
    archive: Path = DEFAULT_PR91_ARCHIVE,
    *,
    device: str = "cpu",
    probability_variant: str = DEFAULT_HPAC_PROBABILITY_VARIANT,
    prob_eps: float = PROB_EPS,
    output_dir: Path | None = None,
    strict: bool = False,
    write_json: bool = True,
) -> dict[str, Any]:
    """Bounded PR91/HPM1 probe for the in-group serial-context hypothesis."""

    started_at = time.time()
    if str(device) != "cpu":
        raise Pr91Hpm1Error(
            "device_contract",
            "pr91_hpm1_in_group_context_update_probe_is_cpu_only",
            requested_device=device,
        )
    archive_path = Path(archive)
    payload = extract_pr91_hpm1_payload(archive_path)
    dependency_report = collect_dependency_report(strict=False)
    static_report = validate_hpm1_static_contract(payload)
    relationship = compare_hpm1_to_pr86_hpac_contract(payload)
    runtime_sources = analyze_pr91_hpm1_runtime_sources()
    model = load_hpm1_hpac_model(payload, device=device)
    source_trace = _trace_hpm1_entropy_decode_failure(
        model,
        payload,
        probability_variant=probability_variant,
        prob_eps=prob_eps,
        max_frames=1,
        device=device,
    )
    context_probe = _trace_hpm1_in_group_prefix_context_at_source_failure(
        model,
        payload,
        probability_variant=probability_variant,
        prob_eps=prob_eps,
        device=device,
        source_trace=source_trace,
    )
    narrowed = (
        context_probe.get("status")
        == "not_explained_by_serial_in_group_prefix_context"
    )
    report: dict[str, Any] = {
        "schema": "pr91_hpm1_in_group_context_update_probe_v1",
        "tool": "tac.pr91_hpm1_codec.run_pr91_hpm1_in_group_context_update_probe",
        "recorded_at_utc": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "status": (
            "narrowed_serial_in_group_context_false_lead"
            if narrowed
            else "serial_in_group_context_hypothesis_still_open"
        ),
        "score_claim": False,
        "dispatch_allowed": False,
        "dispatch_performed": False,
        "gpu_or_remote_work": False,
        "local_only": True,
        "ready_for_exact_eval_dispatch": False,
        "promotion_eligible": False,
        "evidence_grade": "local_hpm1_context_hypothesis_probe",
        "device": device,
        "archive": _archive_report(archive_path),
        "hpm1_static_contract": static_report,
        "pr86_hpac_relationship": relationship,
        "dependency_report": dependency_report,
        "runtime_source_contract": runtime_sources,
        "payload": {
            "config": payload.config(),
            "tokens_bytes": len(payload.tokens),
            "tokens_sha256": sha256_bytes(payload.tokens),
            "hpac_bytes": len(payload.hpac),
            "hpac_sha256": sha256_bytes(payload.hpac),
        },
        "source_entropy_failure_trace": source_trace,
        "in_group_context_update_probe": context_probe,
        "exact_missing_grammar": {
            "status": "still_open_after_serial_in_group_context_probe",
            "not_explained_by_this_probe": (
                "same-group serial prefix assignment at the exact source failure"
                if narrowed
                else "",
            ),
            "remaining_open_classes": [
                "encoder-side probability numeric contract",
                "range-coder construction/finalization contract",
                "context drift before the failing group",
                "training-time token semantics not represented by the submitted decode source",
            ],
        },
        "next_required_proofs": [
            "recover an encoder-side HPAC token generator or full source archive",
            "test probability quantization/range-coder construction against the traced failing row",
            "obtain or derive PR91 semantic mask tokens for teacher-forced context windows",
            "decode all 600 HPM1 frames and re-encode exact token bytes before any dispatch",
        ],
        "elapsed_sec": round(time.time() - started_at, 3),
    }
    if write_json and output_dir is not None:
        write_json_report(report, Path(output_dir) / "in_group_context_update_probe.json")
    if strict and report["ready_for_exact_eval_dispatch"] is not True:
        raise Pr91Hpm1Error(
            "hpm1_in_group_context_update_probe",
            str(report["status"]),
            report=report,
        )
    return _jsonable(report)


def run_pr91_hpm1_spatial_group_order_probe(
    archive: Path = DEFAULT_PR91_ARCHIVE,
    *,
    device: str = "cpu",
    probability_variant: str = DEFAULT_HPAC_PROBABILITY_VARIANT,
    prob_eps: float = PROB_EPS,
    max_frames: int | None = 1,
    candidates: tuple[str, ...] = PR91_HPM1_SPATIAL_ORDER_CANDIDATES,
    output_dir: Path | None = None,
    strict: bool = False,
    write_json: bool = True,
) -> dict[str, Any]:
    """Bounded PR91/HPM1 probe for spatial patch-group traversal order."""

    started_at = time.time()
    if str(device) != "cpu":
        raise Pr91Hpm1Error(
            "device_contract",
            "pr91_hpm1_spatial_group_order_probe_is_cpu_only",
            requested_device=device,
        )
    requested_candidates = _validate_spatial_order_candidates(candidates)
    archive_path = Path(archive)
    payload = extract_pr91_hpm1_payload(archive_path)
    dependency_report = collect_dependency_report(strict=False)
    static_report = validate_hpm1_static_contract(payload)
    relationship = compare_hpm1_to_pr86_hpac_contract(payload)
    runtime_sources = analyze_pr91_hpm1_runtime_sources()
    model = load_hpm1_hpac_model(payload, device=device)
    spatial_probe = _run_hpm1_spatial_group_order_probe(
        model,
        payload,
        probability_variant=probability_variant,
        prob_eps=prob_eps,
        max_frames=max_frames,
        device=device,
        candidates=requested_candidates,
    )
    narrowed = (
        spatial_probe.get("status")
        == "not_explained_by_spatial_group_traversal_order"
    )
    report: dict[str, Any] = {
        "schema": "pr91_hpm1_spatial_group_order_probe_v1",
        "tool": "tac.pr91_hpm1_codec.run_pr91_hpm1_spatial_group_order_probe",
        "recorded_at_utc": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "status": (
            "narrowed_spatial_group_order_false_lead"
            if narrowed
            else "spatial_group_order_hypothesis_still_open"
        ),
        "score_claim": False,
        "dispatch_allowed": False,
        "dispatch_performed": False,
        "gpu_or_remote_work": False,
        "local_only": True,
        "ready_for_exact_eval_dispatch": False,
        "promotion_eligible": False,
        "evidence_grade": "local_hpm1_source_context_hypothesis_probe",
        "device": device,
        "archive": _archive_report(archive_path),
        "hpm1_static_contract": static_report,
        "pr86_hpac_relationship": relationship,
        "dependency_report": dependency_report,
        "runtime_source_contract": runtime_sources,
        "payload": {
            "config": payload.config(),
            "tokens_bytes": len(payload.tokens),
            "tokens_sha256": sha256_bytes(payload.tokens),
            "hpac_bytes": len(payload.hpac),
            "hpac_sha256": sha256_bytes(payload.hpac),
        },
        "spatial_group_order_probe": spatial_probe,
        "exact_missing_grammar": {
            "status": "still_open_after_spatial_group_order_probe",
            "not_explained_by_this_probe": (
                "source-adjacent spatial traversal order for HPAC patch-group symbols"
                if narrowed
                else "not_applicable_spatial_group_order_hypothesis_still_open"
            ),
            "remaining_open_classes": [
                "encoder-side probability numeric contract",
                "range-coder construction/finalization contract",
                "context drift from semantics not visible in submitted decode source",
                "training-time token semantics not represented by the submitted decode source",
            ],
        },
        "next_required_proofs": [
            "recover an encoder-side HPAC token generator or full source archive",
            "test range-coder construction/finalization against the traced failing row",
            "test probability quantization beyond the existing float32/float64/perfect matrix",
            "obtain semantic mask tokens for teacher-forced context-window probes",
            "decode all 600 HPM1 frames and re-encode exact token bytes before any dispatch",
        ],
        "elapsed_sec": round(time.time() - started_at, 3),
    }
    if write_json and output_dir is not None:
        write_json_report(report, Path(output_dir) / "spatial_group_order_probe.json")
    if strict and report["ready_for_exact_eval_dispatch"] is not True:
        raise Pr91Hpm1Error(
            "hpm1_spatial_group_order_probe",
            str(report["status"]),
            report=report,
        )
    return _jsonable(report)


def run_pr91_hpm1_reference_teacher_forcing_probe(
    archive: Path = DEFAULT_PR91_ARCHIVE,
    *,
    reference_tokens_path: Path = DEFAULT_PR85_QMA9_DECODED_REFERENCE_TOKEN_SOURCE,
    reference_layout: str = "legacy_assume_nhw",
    device: str = "cpu",
    probability_variant: str = DEFAULT_HPAC_PROBABILITY_VARIANT,
    prob_eps: float = PROB_EPS,
    max_frames: int | None = 1,
    candidates: tuple[str, ...] = (
        "tile_major_row_major",
        "phase_major_row_major",
    ),
    output_dir: Path | None = None,
    strict: bool = False,
    write_json: bool = True,
    require_expected_reference_sha: bool = True,
    reference_window_before: int = 2,
    reference_window_after: int = 5,
    run_range_prefix_probe: bool = False,
    range_prefix_window_symbols: tuple[int, ...] | list[int] = (
        DEFAULT_PR91_HPM1_RANGE_PREFIX_WINDOW_SYMBOLS
    ),
    range_prefix_seed_symbol_counts: tuple[int, ...] | list[int] = (
        DEFAULT_PR91_HPM1_RANGE_PREFIX_SEED_SYMBOLS
    ),
    range_prefix_replay_symbol_limit: int = (
        DEFAULT_PR91_HPM1_RANGE_PREFIX_REPLAY_SYMBOL_LIMIT
    ),
    range_prefix_max_target_decoded_before: int = (
        DEFAULT_PR91_HPM1_RANGE_PREFIX_MAX_TARGET_DECODED_BEFORE
    ),
) -> dict[str, Any]:
    """Test whether PR85/QMA9 semantic teacher-forcing advances HPM1 replay."""

    started_at = time.time()
    if str(device) != "cpu":
        raise Pr91Hpm1Error(
            "device_contract",
            "pr91_hpm1_reference_teacher_forcing_probe_is_cpu_only",
            requested_device=device,
        )
    requested_candidates = _validate_spatial_order_candidates(candidates)
    if int(reference_window_before) < 0 or int(reference_window_after) < 0:
        raise Pr91Hpm1Error(
            "reference_teacher_forcing_probe",
            "reference_window_counts_must_be_nonnegative",
            reference_window_before=reference_window_before,
            reference_window_after=reference_window_after,
        )
    range_prefix_windows = _validate_range_prefix_window_symbols(
        range_prefix_window_symbols
    )
    range_prefix_seed_counts = _validate_range_prefix_seed_symbol_counts(
        range_prefix_seed_symbol_counts
    )
    if int(range_prefix_replay_symbol_limit) < 0:
        raise Pr91Hpm1Error(
            "hpm1_range_state_prefix_probe",
            "replay_symbol_limit_must_be_nonnegative",
            replay_symbol_limit=range_prefix_replay_symbol_limit,
        )
    if int(range_prefix_max_target_decoded_before) < 0:
        raise Pr91Hpm1Error(
            "hpm1_range_state_prefix_probe",
            "range_prefix_max_target_decoded_before_must_be_nonnegative",
            range_prefix_max_target_decoded_before=range_prefix_max_target_decoded_before,
        )
    archive_path = Path(archive)
    payload = extract_pr91_hpm1_payload(archive_path)
    reference_tokens, reference_report = _load_reference_tokens(
        Path(reference_tokens_path),
        payload.n_frames,
        payload.height,
        payload.width,
        reference_layout,
    )
    reference_sha_matches_expected = bool(
        reference_report["matches_expected_pr85_qma9_token_source"]
    )
    if require_expected_reference_sha and not reference_sha_matches_expected:
        raise Pr91Hpm1Error(
            "reference_token_contract",
            "unexpected_pr85_qma9_reference_token_sha256",
            path=reference_report["path"],
            expected_sha256=reference_report["expected_sha256"],
            actual_sha256=reference_report["sha256"],
            layout=reference_layout,
        )
    dependency_report = collect_dependency_report(strict=False)
    static_report = validate_hpm1_static_contract(payload)
    relationship = compare_hpm1_to_pr86_hpac_contract(payload)
    runtime_sources = analyze_pr91_hpm1_runtime_sources()
    model = load_hpm1_hpac_model(payload, device=device)
    source_trace = _trace_hpm1_spatial_order_decode_failure(
        model,
        payload,
        probability_variant=probability_variant,
        prob_eps=prob_eps,
        max_frames=max_frames,
        device=device,
        spatial_order_candidate="source_mask_row_major",
    )
    source_decoded_before = _decoded_symbols_or_before_failure(source_trace)
    candidate_results: list[dict[str, Any]] = []
    advanced_candidates: list[str] = []

    for candidate in requested_candidates:
        decoded_trace = _trace_hpm1_spatial_order_decode_failure(
            model,
            payload,
            probability_variant=probability_variant,
            prob_eps=prob_eps,
            max_frames=max_frames,
            device=device,
            spatial_order_candidate=candidate,
        )
        reference_trace = _trace_hpm1_reference_teacher_forced_spatial_order_failure(
            model,
            payload,
            reference_tokens,
            probability_variant=probability_variant,
            prob_eps=prob_eps,
            max_frames=max_frames,
            device=device,
            spatial_order_candidate=candidate,
            reference_window_before=reference_window_before,
            reference_window_after=reference_window_after,
        )
        range_prefix_trace: dict[str, Any] | None = None
        if run_range_prefix_probe:
            reference_signature = _failure_row_signature(reference_trace)
            if reference_signature is None:
                range_prefix_trace = {
                    "schema": "pr91_hpm1_range_state_prefix_probe_v1",
                    "status": "not_attempted_no_reference_forced_failure_row",
                    "passed": False,
                    "score_claim": False,
                    "dispatch_allowed": False,
                    "full_decode_proven": False,
                    "byte_exact_reencode_proven": False,
                    "reason": (
                        "range-prefix reconstruction requires a reproduced "
                        "reference-forced entropy failure row"
                    ),
                }
            elif (
                int(reference_signature["decoded_symbol_count_before_failure"])
                > int(range_prefix_max_target_decoded_before)
            ):
                range_prefix_trace = {
                    "schema": "pr91_hpm1_range_state_prefix_probe_v1",
                    "status": "not_attempted_target_exceeds_symbol_budget",
                    "passed": False,
                    "score_claim": False,
                    "dispatch_allowed": False,
                    "full_decode_proven": False,
                    "byte_exact_reencode_proven": False,
                    "target_failure": dict(reference_signature),
                    "target_decoded_symbols_before_failure": int(
                        reference_signature["decoded_symbol_count_before_failure"]
                    ),
                    "range_prefix_max_target_decoded_before": int(
                        range_prefix_max_target_decoded_before
                    ),
                    "reason": (
                        "range-prefix reconstruction replays model context to "
                        "the target row; raise the explicit budget for slow "
                        "forensic runs."
                    ),
                }
            else:
                range_prefix_trace = (
                    _trace_hpm1_reference_forced_range_state_prefix_probe(
                        model,
                        payload,
                        reference_tokens,
                        probability_variant=probability_variant,
                        prob_eps=prob_eps,
                        device=device,
                        spatial_order_candidate=candidate,
                        target_failure=reference_signature,
                        range_prefix_window_symbols=range_prefix_windows,
                        range_prefix_seed_symbol_counts=range_prefix_seed_counts,
                        replay_symbol_limit=range_prefix_replay_symbol_limit,
                    )
                )
        decoded_before = _decoded_symbols_or_before_failure(decoded_trace)
        reference_before = _decoded_symbols_or_before_failure(reference_trace)
        advances_beyond_decoded = (
            reference_trace.get("passed") is True or reference_before > decoded_before
        )
        if advances_beyond_decoded:
            advanced_candidates.append(candidate)
        candidate_results.append(
            {
                "candidate": candidate,
                "description": _spatial_order_description(candidate),
                "decoded_context": {
                    "status": decoded_trace.get("status"),
                    "passed": bool(decoded_trace.get("passed") is True),
                    "failure_signature": _failure_row_signature(decoded_trace),
                    "decoded_symbols_or_before_failure": decoded_before,
                    "full_decode_proven": bool(
                        decoded_trace.get("full_decode_proven") is True
                    ),
                    "byte_exact_reencode_proven": bool(
                        decoded_trace.get("byte_exact_reencode_proven") is True
                    ),
                    "passes_source_failure_row": (
                        decoded_trace.get("passed") is True
                        or decoded_before > source_decoded_before
                    ),
                    "exception_type": (
                        decoded_trace.get("failure", {}).get("exception_type")
                        if isinstance(decoded_trace.get("failure"), Mapping)
                        else None
                    ),
                    "failing_probability_row_sha256": (
                        decoded_trace.get("failing_probability_row", {})
                        .get("normalized_for_categorical", {})
                        .get("sha256")
                        if isinstance(decoded_trace.get("failing_probability_row"), Mapping)
                        else None
                    ),
                    "range_decoder_diagnostic": (
                        decoded_trace.get("range_decoder_diagnostic")
                        if isinstance(decoded_trace.get("range_decoder_diagnostic"), Mapping)
                        else None
                    ),
                    "failure_row_interpretation": (
                        decoded_trace.get("failure_row_interpretation")
                        if isinstance(decoded_trace.get("failure_row_interpretation"), Mapping)
                        else None
                    ),
                },
                "reference_teacher_forced_context": {
                    "status": reference_trace.get("status"),
                    "passed": bool(reference_trace.get("passed") is True),
                    "failure_signature": _failure_row_signature(reference_trace),
                    "decoded_symbols_or_before_failure": reference_before,
                    "full_decode_proven": bool(
                        reference_trace.get("full_decode_proven") is True
                    ),
                    "byte_exact_reencode_proven": bool(
                        reference_trace.get("byte_exact_reencode_proven") is True
                    ),
                    "advances_beyond_decoded_context": bool(advances_beyond_decoded),
                    "advances_beyond_source_failure_row": (
                        reference_trace.get("passed") is True
                        or reference_before > source_decoded_before
                    ),
                    "exception_type": (
                        reference_trace.get("failure", {}).get("exception_type")
                        if isinstance(reference_trace.get("failure"), Mapping)
                        else None
                    ),
                    "failing_probability_row_sha256": (
                        reference_trace.get("failing_probability_row", {})
                        .get("normalized_for_categorical", {})
                        .get("sha256")
                        if isinstance(reference_trace.get("failing_probability_row"), Mapping)
                        else None
                    ),
                    "range_decoder_diagnostic": (
                        reference_trace.get("range_decoder_diagnostic")
                        if isinstance(reference_trace.get("range_decoder_diagnostic"), Mapping)
                        else None
                    ),
                    "failure_row_interpretation": (
                        reference_trace.get("failure_row_interpretation")
                        if isinstance(
                            reference_trace.get("failure_row_interpretation"),
                            Mapping,
                        )
                        else None
                    ),
                    "canonical_reference_symbol_window": (
                        reference_trace.get("canonical_reference_symbol_window")
                        if isinstance(
                            reference_trace.get("canonical_reference_symbol_window"),
                            Mapping,
                        )
                        else None
                    ),
                    "reference_mismatch_count_before_failure": (
                        reference_trace.get("reference_teacher_forcing", {}).get(
                            "reference_mismatch_count_before_failure"
                        )
                        if isinstance(
                            reference_trace.get("reference_teacher_forcing"),
                            Mapping,
                        )
                        else None
                    ),
                    "first_decoded_reference_mismatch": (
                        reference_trace.get("reference_teacher_forcing", {}).get(
                            "first_decoded_reference_mismatch"
                        )
                        if isinstance(
                            reference_trace.get("reference_teacher_forcing"),
                            Mapping,
                        )
                        else None
                    ),
                    **(
                        {"range_state_prefix_probe": range_prefix_trace}
                        if run_range_prefix_probe
                        else {}
                    ),
                },
            }
        )

    narrowed = not advanced_candidates
    candidate_scope_label = _spatial_candidate_scope_label(requested_candidates)
    hypothesis_classification = _classify_reference_teacher_forcing_hypotheses(
        candidate_results,
        advanced_candidates=advanced_candidates,
        source_decoded_before=source_decoded_before,
    )
    report: dict[str, Any] = {
        "schema": "pr91_hpm1_reference_teacher_forcing_probe_v1",
        "tool": "tac.pr91_hpm1_codec.run_pr91_hpm1_reference_teacher_forcing_probe",
        "recorded_at_utc": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "status": (
            "narrowed_pr85_qma9_reference_teacher_forcing_false_lead"
            if narrowed
            else "reference_teacher_forcing_hypothesis_still_open"
        ),
        "score_claim": False,
        "dispatch_allowed": False,
        "dispatch_attempted": False,
        "dispatch_performed": False,
        "gpu_or_remote_work": False,
        "local_only": True,
        "ready_for_exact_eval_dispatch": False,
        "promotion_eligible": False,
        "evidence_grade": "empirical",
        "evidence_scope": "local_cpu_hpm1_reference_teacher_forcing_hypothesis_probe",
        "evidence_limitations": [
            "not score evidence",
            "not full standalone HPM1 decode proof",
            "not byte-exact re-encode proof",
            "not dispatch-eligible",
        ],
        "device": device,
        "archive": _archive_report(archive_path),
        "reference_tokens": reference_report,
        "reference_token_sha256_contract": {
            "required": bool(require_expected_reference_sha),
            "expected_sha256": reference_report["expected_sha256"],
            "actual_sha256": reference_report["sha256"],
            "matches_expected": reference_sha_matches_expected,
        },
        "hpm1_static_contract": static_report,
        "pr86_hpac_relationship": relationship,
        "dependency_report": dependency_report,
        "runtime_source_contract": runtime_sources,
        "payload": {
            "config": payload.config(),
            "tokens_bytes": len(payload.tokens),
            "tokens_sha256": sha256_bytes(payload.tokens),
            "hpac_bytes": len(payload.hpac),
            "hpac_sha256": sha256_bytes(payload.hpac),
        },
        "source_order_baseline": {
            "status": source_trace.get("status"),
            "failure_signature": _failure_row_signature(source_trace),
            "decoded_symbols_or_before_failure": source_decoded_before,
            "range_decoder_diagnostic": (
                source_trace.get("range_decoder_diagnostic")
                if isinstance(source_trace.get("range_decoder_diagnostic"), Mapping)
                else None
            ),
        },
        "reference_teacher_forcing_probe": {
            "schema": "pr91_hpm1_reference_teacher_forcing_probe_v1",
            "status": (
                "not_explained_by_pr85_qma9_reference_teacher_forcing"
                if narrowed
                else "at_least_one_candidate_advanced_under_reference_teacher_forcing"
            ),
            "passed": bool(narrowed),
            "score_claim": False,
            "dispatch_allowed": False,
            "probability_variant": resolve_hpac_probability_variant(
                probability_variant
            ).name,
            "prob_eps": float(prob_eps),
            "max_frames": max_frames,
            "reference_symbol_window": {
                "window_before": int(reference_window_before),
                "window_after": int(reference_window_after),
                "scope": "canonical_reference_tokens_at_failure_row",
            },
            "range_prefix_probe": {
                "enabled": bool(run_range_prefix_probe),
                "window_symbols_before_failure": [
                    int(value) for value in range_prefix_windows
                ],
                "seed_symbol_counts": [
                    int(value) for value in range_prefix_seed_counts
                ],
                "replay_symbol_limit": int(range_prefix_replay_symbol_limit),
                "max_target_decoded_before": int(
                    range_prefix_max_target_decoded_before
                ),
                "scope": (
                    "local CPU reference-forced range-state prefix "
                    "reconstruction; not score or dispatch evidence"
                ),
            },
            "candidate_scope": {
                "requested_candidates": list(requested_candidates),
                "label": candidate_scope_label,
            },
            "candidate_results": candidate_results,
            "advanced_candidates": advanced_candidates,
            "candidate_progress_summary": (
                _summarize_reference_teacher_forcing_candidate_progress(
                    candidate_results,
                    source_decoded_before=source_decoded_before,
                )
            ),
            "hypothesis_classification": hypothesis_classification,
            "narrowed_hypothesis": (
                "Forcing the HPAC current/previous context from the available "
                "PR85/QMA9 reference token tensor does not advance the requested "
                f"spatial candidate scope ({candidate_scope_label}) beyond the "
                "decoded-context failure rows."
                if narrowed
                else "At least one spatial candidate advances under PR85/QMA9 "
                "reference teacher-forcing; recover true PR91 semantic tokens "
                "or encoder context before dismissing this class."
            ),
        },
        "hypothesis_classification": hypothesis_classification,
        "exact_missing_grammar": {
            "status": "still_open_after_reference_teacher_forcing_probe",
            "not_explained_by_this_probe": (
                "PR85/QMA9 render-order semantic teacher-forcing for the "
                f"requested spatial candidate scope ({candidate_scope_label})"
                if narrowed
                else ""
            ),
            "remaining_open_classes": [
                "encoder-side probability numeric contract",
                "range-coder construction/finalization contract",
                "true PR91 encoder semantic tokens if they differ from PR85/QMA9 reference",
                "training-time token semantics not represented by submitted decode source",
                "row ordering is only partial evidence until full decode/reencode parity exists",
            ],
        },
        "next_required_proofs": [
            "recover an encoder-side HPAC token generator or full source archive",
            "test range-coder construction/finalization against the traced failing row",
            "test probability quantization beyond the existing float32/float64/perfect matrix",
            "decode all 600 HPM1 frames and re-encode exact token bytes before any dispatch",
        ],
        "elapsed_sec": round(time.time() - started_at, 3),
    }
    if write_json and output_dir is not None:
        write_json_report(
            report,
            Path(output_dir) / "reference_teacher_forcing_probe.json",
        )
    if strict and report["ready_for_exact_eval_dispatch"] is not True:
        raise Pr91Hpm1Error(
            "hpm1_reference_teacher_forcing_probe",
            str(report["status"]),
            report=report,
        )
    return _jsonable(report)


def run_pr91_hpm1_semantic_symbol_bridge_probe(
    archive: Path = DEFAULT_PR91_ARCHIVE,
    *,
    reference_tokens_path: Path = DEFAULT_PR85_QMA9_DECODED_REFERENCE_TOKEN_SOURCE,
    reference_layout: str = "legacy_assume_nhw",
    device: str = "cpu",
    probability_variant: str = DEFAULT_HPAC_PROBABILITY_VARIANT,
    prob_eps: float = PROB_EPS,
    spatial_order_candidate: str = "phase_major_row_major",
    symbol_count: int = DEFAULT_PR91_HPM1_SYMBOL_BRIDGE_PREFIX_SYMBOLS,
    row_preview_limit: int = 16,
    mismatch_limit: int = 16,
    output_dir: Path | None = None,
    strict: bool = False,
    write_json: bool = True,
    require_expected_reference_sha: bool = True,
) -> dict[str, Any]:
    """Probe simple bridges from PR85/QMA9 reference tokens to PR91 symbols."""

    started_at = time.time()
    if str(device) != "cpu":
        raise Pr91Hpm1Error(
            "device_contract",
            "pr91_hpm1_semantic_symbol_bridge_probe_is_cpu_only",
            requested_device=device,
        )
    archive_path = Path(archive)
    payload = extract_pr91_hpm1_payload(archive_path)
    reference_tokens, reference_report = _load_reference_tokens(
        Path(reference_tokens_path),
        payload.n_frames,
        payload.height,
        payload.width,
        reference_layout,
    )
    reference_sha_matches_expected = bool(
        reference_report["matches_expected_pr85_qma9_token_source"]
    )
    if require_expected_reference_sha and not reference_sha_matches_expected:
        raise Pr91Hpm1Error(
            "reference_token_contract",
            "unexpected_pr85_qma9_reference_token_sha256",
            path=reference_report["path"],
            expected_sha256=reference_report["expected_sha256"],
            actual_sha256=reference_report["sha256"],
            layout=reference_layout,
        )

    dependency_report = collect_dependency_report(strict=False)
    static_report = validate_hpm1_static_contract(payload)
    relationship = compare_hpm1_to_pr86_hpac_contract(payload)
    runtime_sources = analyze_pr91_hpm1_runtime_sources()
    model = load_hpm1_hpac_model(payload, device=device)
    bridge_trace = _trace_hpm1_reference_forced_symbol_bridge_prefix(
        model,
        payload,
        reference_tokens,
        probability_variant=probability_variant,
        prob_eps=prob_eps,
        device=device,
        spatial_order_candidate=spatial_order_candidate,
        symbol_count=symbol_count,
        row_preview_limit=row_preview_limit,
        mismatch_limit=mismatch_limit,
    )
    bridge_summary = bridge_trace.get("bridge_summary", {})
    prefix_completed = bridge_trace.get("prefix_completed") is True
    bridge_found = (
        isinstance(bridge_summary, Mapping)
        and bridge_summary.get("bridge_found") is True
    )
    if prefix_completed and not bridge_found:
        status = "narrowed_no_simple_pr85_qma9_to_pr91_symbol_bridge"
        not_explained = (
            "identity, global label permutation, constant mod5 offset, or "
            "previous-frame mod5 residual bridge for the requested prefix"
        )
    elif bridge_found:
        status = "simple_symbol_bridge_hypothesis_still_open"
        not_explained = ""
    else:
        status = "blocked_before_symbol_bridge_prefix_complete"
        not_explained = ""

    report: dict[str, Any] = {
        "schema": "pr91_hpm1_semantic_symbol_bridge_probe_v1",
        "tool": "tac.pr91_hpm1_codec.run_pr91_hpm1_semantic_symbol_bridge_probe",
        "recorded_at_utc": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "status": status,
        "score_claim": False,
        "dispatch_allowed": False,
        "dispatch_attempted": False,
        "dispatch_performed": False,
        "gpu_or_remote_work": False,
        "local_only": True,
        "ready_for_exact_eval_dispatch": False,
        "promotion_eligible": False,
        "evidence_grade": "empirical",
        "evidence_scope": "local_cpu_hpm1_semantic_symbol_bridge_prefix_probe",
        "evidence_limitations": [
            "not score evidence",
            "not full standalone HPM1 decode proof",
            "not byte-exact re-encode proof",
            "not dispatch-eligible",
        ],
        "device": device,
        "archive": _archive_report(archive_path),
        "reference_tokens": reference_report,
        "reference_token_sha256_contract": {
            "required": bool(require_expected_reference_sha),
            "expected_sha256": reference_report["expected_sha256"],
            "actual_sha256": reference_report["sha256"],
            "matches_expected": reference_sha_matches_expected,
        },
        "hpm1_static_contract": static_report,
        "pr86_hpac_relationship": relationship,
        "dependency_report": dependency_report,
        "runtime_source_contract": runtime_sources,
        "payload": {
            "config": payload.config(),
            "tokens_bytes": len(payload.tokens),
            "tokens_sha256": sha256_bytes(payload.tokens),
            "hpac_bytes": len(payload.hpac),
            "hpac_sha256": sha256_bytes(payload.hpac),
        },
        "symbol_bridge_probe": {
            "schema": "pr91_hpm1_semantic_symbol_bridge_probe_v1",
            "status": bridge_trace.get("status"),
            "passed": bool(bridge_trace.get("passed") is True),
            "score_claim": False,
            "dispatch_allowed": False,
            "probability_variant": resolve_hpac_probability_variant(
                probability_variant
            ).name,
            "prob_eps": float(prob_eps),
            "spatial_order_candidate": spatial_order_candidate,
            "requested_symbol_count": int(symbol_count),
            "row_preview_limit": int(row_preview_limit),
            "mismatch_limit": int(mismatch_limit),
            "prefix_trace": bridge_trace,
        },
        "exact_missing_grammar": {
            "status": "still_open_after_symbol_bridge_probe",
            "not_explained_by_this_probe": not_explained,
            "remaining_open_classes": [
                "true PR91 encoder semantic tokens not visible in public runtime",
                "encoder-side probability numeric contract",
                "range-coder construction/finalization contract",
                "phase-major reference context remains only a prior until full decode/reencode parity",
            ],
        },
        "next_required_proofs": [
            "recover true PR91 encoder semantic tokens or source encoder trace",
            "test any surviving bridge on a deeper bounded prefix before reencode work",
            "decode all 600 HPM1 frames and re-encode exact token bytes before any dispatch",
        ],
        "elapsed_sec": round(time.time() - started_at, 3),
    }
    if write_json and output_dir is not None:
        write_json_report(
            report,
            Path(output_dir) / "semantic_symbol_bridge_probe.json",
        )
    if strict and report["ready_for_exact_eval_dispatch"] is not True:
        raise Pr91Hpm1Error(
            "hpm1_semantic_symbol_bridge_probe",
            str(report["status"]),
            report=report,
        )
    return _jsonable(report)


def run_pr91_hpm1_submitted_prefix_token_recovery_probe(
    archive: Path = DEFAULT_PR91_ARCHIVE,
    *,
    reference_tokens_path: Path | None = DEFAULT_PR85_QMA9_DECODED_REFERENCE_TOKEN_SOURCE,
    reference_layout: str = "legacy_assume_nhw",
    device: str = "cpu",
    probability_variant: str = DEFAULT_HPAC_PROBABILITY_VARIANT,
    prob_eps: float = PROB_EPS,
    spatial_order_candidate: str = "source_mask_row_major",
    max_symbols: int | None = None,
    row_preview_limit: int = 16,
    mismatch_limit: int = 16,
    output_dir: Path | None = None,
    strict: bool = False,
    write_json: bool = True,
    require_expected_reference_sha: bool = True,
) -> dict[str, Any]:
    """Recover the deterministic submitted PR91 HPM1 prefix before failure."""

    started_at = time.time()
    if str(device) != "cpu":
        raise Pr91Hpm1Error(
            "device_contract",
            "pr91_hpm1_submitted_prefix_token_recovery_probe_is_cpu_only",
            requested_device=device,
        )
    _spatial_order_description(spatial_order_candidate)
    archive_path = Path(archive)
    payload = extract_pr91_hpm1_payload(archive_path)
    reference_tokens: np.ndarray | None = None
    reference_report: dict[str, Any] = {
        "attempted": False,
        "path": None,
        "layout": reference_layout,
    }
    reference_sha_matches_expected = False
    if reference_tokens_path is not None:
        reference_tokens, loaded_reference_report = _load_reference_tokens(
            Path(reference_tokens_path),
            payload.n_frames,
            payload.height,
            payload.width,
            reference_layout,
        )
        reference_report = loaded_reference_report
        reference_sha_matches_expected = bool(
            reference_report["matches_expected_pr85_qma9_token_source"]
        )
        if require_expected_reference_sha and not reference_sha_matches_expected:
            raise Pr91Hpm1Error(
                "reference_token_contract",
                "unexpected_pr85_qma9_reference_token_sha256",
                path=reference_report["path"],
                expected_sha256=reference_report["expected_sha256"],
                actual_sha256=reference_report["sha256"],
                layout=reference_layout,
            )

    dependency_report = collect_dependency_report(strict=False)
    static_report = validate_hpm1_static_contract(payload)
    relationship = compare_hpm1_to_pr86_hpac_contract(payload)
    runtime_sources = analyze_pr91_hpm1_runtime_sources()
    model = load_hpm1_hpac_model(payload, device=device)
    trace = _trace_hpm1_submitted_prefix_token_recovery(
        model,
        payload,
        reference_tokens,
        probability_variant=probability_variant,
        prob_eps=prob_eps,
        device=device,
        spatial_order_candidate=spatial_order_candidate,
        max_symbols=max_symbols,
        row_preview_limit=row_preview_limit,
        mismatch_limit=mismatch_limit,
    )
    recovered_prefix = (
        trace.get("decoded_symbol_count", 0) > 0
        and trace.get("status") == "recovered_prefix_until_first_entropy_failure"
    )
    report: dict[str, Any] = {
        "schema": "pr91_hpm1_submitted_prefix_token_recovery_probe_v1",
        "tool": "tac.pr91_hpm1_codec.run_pr91_hpm1_submitted_prefix_token_recovery_probe",
        "recorded_at_utc": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "status": (
            "recovered_submitted_prefix_until_first_entropy_failure"
            if recovered_prefix
            else str(trace.get("status"))
        ),
        "score_claim": False,
        "dispatch_allowed": False,
        "dispatch_attempted": False,
        "dispatch_performed": False,
        "gpu_or_remote_work": False,
        "local_only": True,
        "ready_for_exact_eval_dispatch": False,
        "promotion_eligible": False,
        "evidence_grade": "empirical",
        "evidence_scope": "local_cpu_hpm1_submitted_prefix_token_and_probability_trace",
        "spatial_order_probe": {
            "candidate": spatial_order_candidate,
            "description": _spatial_order_description(spatial_order_candidate),
            "source_contract": spatial_order_candidate == "source_mask_row_major",
            "dispatch_allowed": False,
            "scope_note": (
                "Non-source spatial orders are off-contract local forensic "
                "hypotheses. They may recover deeper submitted range prefixes "
                "but do not prove PR91 decode parity or byte-exact reencode."
            ),
        },
        "evidence_limitations": [
            "not score evidence",
            "not full standalone HPM1 decode proof",
            "not byte-exact re-encode proof",
            "not dispatch-eligible",
        ],
        "device": device,
        "archive": _archive_report(archive_path),
        "reference_tokens": reference_report,
        "reference_token_sha256_contract": {
            "required": bool(
                reference_tokens_path is not None and require_expected_reference_sha
            ),
            "expected_sha256": reference_report.get("expected_sha256"),
            "actual_sha256": reference_report.get("sha256"),
            "matches_expected": reference_sha_matches_expected,
        },
        "hpm1_static_contract": static_report,
        "pr86_hpac_relationship": relationship,
        "dependency_report": dependency_report,
        "runtime_source_contract": runtime_sources,
        "payload": {
            "config": payload.config(),
            "tokens_bytes": len(payload.tokens),
            "tokens_sha256": sha256_bytes(payload.tokens),
            "hpac_bytes": len(payload.hpac),
            "hpac_sha256": sha256_bytes(payload.hpac),
        },
        "submitted_prefix_token_recovery": trace,
        "exact_missing_grammar": {
            "status": "still_open_after_submitted_prefix_token_recovery",
            "recovered": (
                "deterministic submitted semantic-symbol prefix before the "
                "first public HPAC range failure"
                if spatial_order_candidate == "source_mask_row_major"
                else (
                    "deterministic submitted semantic-symbol prefix before the "
                    f"first off-contract {spatial_order_candidate} HPAC range failure"
                )
            ),
            "remaining_open_classes": [
                "encoder-side probability numeric contract at the failure row",
                "range-coder construction/finalization contract",
                "true PR91 symbols beyond the first entropy failure",
                "byte-exact full-stream reencode",
            ],
        },
        "next_required_proofs": [
            "use the recovered prefix and probability-row hashes to isolate the first non-public encoder contract difference",
            "recover an encoder-side HPAC token generator or full source archive",
            "decode all 600 HPM1 frames and re-encode exact token bytes before any dispatch",
        ],
        "elapsed_sec": round(time.time() - started_at, 3),
    }
    if write_json and output_dir is not None:
        write_json_report(
            report,
            Path(output_dir) / "submitted_prefix_token_recovery_probe.json",
        )
    if strict and report["ready_for_exact_eval_dispatch"] is not True:
        raise Pr91Hpm1Error(
            "hpm1_submitted_prefix_token_recovery_probe",
            str(report["status"]),
            report=report,
        )
    return _jsonable(report)


def run_pr91_hpm1_semantic_decode_trench(
    archive: Path = DEFAULT_PR91_ARCHIVE,
    *,
    device: str = "cpu",
    probability_variants: tuple[str, ...] | list[str] | None = None,
    probability_row_count: int = 8,
    prob_eps: float = PROB_EPS,
    prefix_max_frames: int | None = 1,
    attempt_prefix_decode: bool = True,
    output_dir: Path | None = None,
    strict: bool = False,
    write_json: bool = True,
) -> dict[str, Any]:
    """Inventory the embedded PR91 HPM1 HPAC model, then fail closed.

    This is the semantic decode trench: it proves the charged HPAC model bytes
    can be decompressed, loaded, and used to emit deterministic probability
    rows on CPU. It does not claim decoded masks or byte-exact re-encode parity.
    """

    started_at = time.time()
    if str(device) != "cpu":
        raise Pr91Hpm1Error(
            "device_contract",
            "pr91_hpm1_semantic_decode_trench_is_cpu_only",
            requested_device=device,
        )
    requested_variants = _resolve_probe_variants(probability_variants)
    archive_path = Path(archive)
    payload = extract_pr91_hpm1_payload(archive_path)
    mask_segment_sha = sha256_bytes(
        build_hpm1_mask_segment(
            payload.tokens,
            payload.hpac,
            N=payload.n_frames,
            H=payload.height,
            W=payload.width,
            P=payload.predictor_count,
            delta=payload.delta,
            ch=payload.channels,
            use_spm=bool(payload.use_spm),
            hpac_d_film=payload.hpac_d_film,
            ppmd_order=payload.ppmd_order,
        )
    )
    dependency_report = collect_dependency_report(strict=False)
    static_report = validate_hpm1_static_contract(payload)
    relationship = compare_hpm1_to_pr86_hpac_contract(payload)
    blockers: list[str] = [
        "full_hpm1_decode_600_frames_not_proven",
        "byte_exact_hpm1_reencode_not_proven",
        "runtime_hpm1_loader_sidecar_free_not_proven",
        "exact_cuda_auth_eval_not_allowed_without_local_parity",
    ]
    warnings: list[str] = []
    model: Any | None = None
    raw_state_sha = ""
    packed_state_inventory: dict[str, Any] = {
        "state_kind": "packed_hpac_state_dict",
        "loaded": False,
        "blocker": "",
    }
    reconstructed_state_inventory: dict[str, Any] = {
        "state_kind": "reconstructed_hpacmini_state_dict",
        "loaded": False,
        "blocker": "",
    }
    model_load: dict[str, Any] = {
        "status": "not_attempted",
        "loaded": False,
        "device": device,
        "compressed_hpac": {
            "bytes": len(payload.hpac),
            "sha256": sha256_bytes(payload.hpac),
            "ppmd_order": payload.ppmd_order,
            "ppmd_mem_size": PPMD_MEM_SIZE,
        },
        "decompressed_torch_state": {
            "bytes": None,
            "sha256": "",
        },
        "blockers": [],
    }

    missing_model_deps = [
        name
        for name in ("torch", "pyppmd")
        if not _dependency_available(dependency_report, name)
    ]
    if missing_model_deps:
        blockers.extend(f"decoder_dependency_missing:{name}" for name in missing_model_deps)
        model_load.update(
            {
                "status": "failed_closed_missing_model_dependencies",
                "blockers": [f"missing:{name}" for name in missing_model_deps],
            }
        )
    else:
        try:
            raw_state, packed_state = _load_hpm1_packed_state_bytes(payload)
            raw_state_sha = sha256_bytes(raw_state)
            packed_state_inventory = _state_dict_tensor_inventory(
                packed_state,
                state_kind="packed_hpac_state_dict",
            )
            packed_state_inventory["loaded"] = True
            model = load_hpm1_hpac_model(payload, device=device)
            reconstructed_state_inventory = _state_dict_tensor_inventory(
                model.state_dict(),
                state_kind="reconstructed_hpacmini_state_dict",
            )
            reconstructed_state_inventory["loaded"] = True
            model_load.update(
                {
                    "status": "passed_hpac_model_load",
                    "loaded": True,
                    "decompressed_torch_state": {
                        "bytes": len(raw_state),
                        "sha256": raw_state_sha,
                    },
                    "model": {
                        "type": type(model).__name__,
                        "P": int(model.P),
                        "delta": int(model.delta),
                        "ch": int(model.ch),
                        "use_spm": bool(model.use_spm),
                        "num_classes": int(model.num_classes),
                        "frame_embed_num_embeddings": int(model.frame_embed.num_embeddings),
                        "frame_embed_dim": int(model.frame_embed.embedding_dim),
                    },
                    "blockers": [],
                }
            )
        except (Pr91Hpm1Error, Pr86HpacReplayError) as exc:
            blockers.append(f"hpac_model_load_failed:{exc.code}")
            model_load.update(
                {
                    "status": "failed_closed_hpac_model_load",
                    "blockers": [exc.code],
                    "failure_contract": exc.contract,
                    "failure_context": dict(exc.fields),
                }
            )

    probability_row_probe: dict[str, Any] = {
        "status": "not_attempted_model_not_loaded",
        "passed": False,
        "blockers": ["hpac_model_not_loaded"],
    }
    if model is not None:
        try:
            probability_row_probe = _hpm1_probability_row_probe(
                model,
                payload,
                variants=requested_variants,
                row_count=probability_row_count,
                prob_eps=prob_eps,
                device=device,
            )
        except (Pr91Hpm1Error, Pr86HpacReplayError) as exc:
            blockers.append(f"probability_row_probe_failed:{exc.code}")
            probability_row_probe = {
                "status": "failed_closed_probability_row_probe",
                "passed": False,
                "failure_contract": exc.contract,
                "failure_reason": exc.code,
                "failure_context": dict(exc.fields),
                "blockers": [exc.code],
            }

    prefix_decode: dict[str, Any] = {
        "attempted": bool(attempt_prefix_decode),
        "status": "not_attempted",
        "passed": False,
        "max_frames": prefix_max_frames,
    }
    if model is None:
        prefix_decode["status"] = "not_attempted_model_not_loaded"
    elif not attempt_prefix_decode:
        prefix_decode["status"] = "not_attempted_by_request"
        blockers.append("prefix_decode_not_attempted")
    else:
        try:
            decoded, decode_report = decode_tokens_hpac(
                model,
                payload.tokens,
                payload.n_frames,
                payload.height,
                payload.width,
                payload.predictor_count,
                payload.delta,
                device=device,
                prob_eps=prob_eps,
                probability_variant=DEFAULT_HPAC_PROBABILITY_VARIANT,
                max_frames=prefix_max_frames,
                return_report=True,
            )
            prefix_decode = {
                "attempted": True,
                "status": "passed_prefix_decode",
                "passed": True,
                "max_frames": prefix_max_frames,
                "decoded_tokens": {
                    "shape": list(decoded.shape),
                    "sha256": sha256_bytes(decoded.tobytes()),
                },
                "decode_report": decode_report,
            }
            blockers.append("full_decode_not_attempted_after_prefix")
        except Pr86HpacReplayError as exc:
            blockers.append(f"prefix_decode_failed:{exc.code}")
            prefix_decode = {
                "attempted": True,
                "status": "failed_closed",
                "passed": False,
                "max_frames": prefix_max_frames,
                "failure_stage": exc.contract,
                "failure_reason": exc.code,
                "failure_context": dict(exc.fields),
            }

    if model_load["loaded"] and prefix_decode["status"] == "failed_closed":
        status = "blocked_prefix_decode_entropy_contract_mismatch_after_model_load"
    elif model_load["loaded"] and probability_row_probe.get("passed") is True:
        status = "blocked_full_decode_reencode_not_proven_after_model_row_inventory"
    else:
        status = "blocked_hpac_model_load_or_probability_row_inventory"

    report: dict[str, Any] = {
        "schema": "pr91_hpm1_semantic_decode_trench_v1",
        "tool": "tac.pr91_hpm1_codec.run_pr91_hpm1_semantic_decode_trench",
        "recorded_at_utc": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "status": status,
        "score_claim": False,
        "dispatch_allowed": False,
        "dispatch_performed": False,
        "gpu_or_remote_work": False,
        "local_only": True,
        "ready_for_exact_eval_dispatch": False,
        "promotion_eligible": False,
        "evidence_grade": "local_hpac_model_load_probability_row_inventory",
        "device": device,
        "archive": _archive_report(archive_path),
        "hpm1_static_contract": static_report,
        "pr86_hpac_relationship": relationship,
        "dependency_report": dependency_report,
        "payload": {
            "config": payload.config(),
            "mask_segment_sha256": mask_segment_sha,
            "expected_pr91_mask_segment_sha256": EXPECTED_PR91_HPM1_MASK_SHA256,
            "tokens_bytes": len(payload.tokens),
            "tokens_sha256": sha256_bytes(payload.tokens),
            "hpac_bytes": len(payload.hpac),
            "hpac_sha256": sha256_bytes(payload.hpac),
        },
        "hpac_model_load": model_load,
        "packed_state_inventory": packed_state_inventory,
        "reconstructed_state_inventory": reconstructed_state_inventory,
        "probability_row_probe": probability_row_probe,
        "prefix_decode": prefix_decode,
        "full_decode": {
            "passed": False,
            "frame_count": 0,
            "decoded_masks_sha256": "",
            "refusal_reason": "full_600_frame_hpm1_decode_not_proven",
        },
        "byte_exact_semantic_reencode": {
            "passed": False,
            "byte_exact": False,
            "reencoded_hpm1_sha256": "",
            "refusal_reason": "semantic_decode_or_range_reencode_parity_not_proven",
        },
        "semantic_decode_blockers": sorted(set(blockers)),
        "warnings": warnings,
        "next_required_proofs": [
            "repair the HPAC probability/range contract past the local prefix failure",
            "decode all 600 HPM1 frames from the exact PR91 token stream on CPU",
            "record decoded mask tensor SHA-256 and context-window traces",
            "range-encode decoded symbols back to the exact token stream SHA-256",
            "prove charged runtime HPM1 loading without sidecars or fallback",
        ],
        "elapsed_sec": round(time.time() - started_at, 3),
    }
    if raw_state_sha:
        report["hpac_model_load"]["decompressed_torch_state"]["sha256"] = raw_state_sha
    if write_json and output_dir is not None:
        write_json_report(report, Path(output_dir) / "semantic_decode_trench.json")
    if strict and report["ready_for_exact_eval_dispatch"] is not True:
        raise Pr91Hpm1Error("hpm1_semantic_decode_trench", str(report["status"]), report=report)
    return _jsonable(report)


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
    canonical_payload = (
        sha256_bytes(payload.tokens) == EXPECTED_PR91_HPM1_TOKENS_SHA256
        and sha256_bytes(payload.hpac) == EXPECTED_PR91_HPM1_HPAC_SHA256
    )
    prefix_decode: dict[str, Any] = {
        "status": "not_attempted_noncanonical_payload",
        "max_frames": max_frames,
        "canonical_pr91_payload": canonical_payload,
    }
    if canonical_payload and max_frames is not None and int(max_frames) > 0:
        try:
            model = load_hpm1_hpac_model(payload, device=device)
            decoded, decode_report = decode_tokens_hpac(
                model,
                payload.tokens,
                payload.n_frames,
                payload.height,
                payload.width,
                payload.predictor_count,
                payload.delta,
                device=device,
                probability_variant=probability_variant,
                max_frames=max_frames,
                return_report=True,
            )
            prefix_decode = {
                "status": "passed_prefix_decode",
                "max_frames": max_frames,
                "canonical_pr91_payload": True,
                "decoded_tokens": {
                    "shape": list(decoded.shape),
                    "sha256": sha256_bytes(decoded.tobytes()),
                },
                "decode_report": decode_report,
            }
        except Pr86HpacReplayError as exc:
            prefix_decode = {
                "status": "failed_closed",
                "failure_stage": "hpac_probability_range_decode",
                "failure_reason": exc.code,
                "failure_context": dict(exc.fields),
                "max_frames": max_frames,
                "canonical_pr91_payload": True,
            }
    report: dict[str, Any] = {
        "schema_version": 1,
        "tool": "tac.pr91_hpm1_codec.run_pr91_hpm1_preflight",
        "recorded_at_utc": datetime.now(UTC).replace(microsecond=0).isoformat(),
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
        "prefix_or_full_decode": prefix_decode,
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
    canonical_payload = (
        sha256_bytes(payload.tokens) == EXPECTED_PR91_HPM1_TOKENS_SHA256
        and sha256_bytes(payload.hpac) == EXPECTED_PR91_HPM1_HPAC_SHA256
    )
    variant_names = variants or supported_hpac_probability_variant_names()
    model: Any | None = None
    if canonical_payload and max_frames is not None and int(max_frames) > 0:
        model = load_hpm1_hpac_model(payload, device="cpu")
    rows: list[dict[str, Any]] = []
    for name in variant_names:
        variant = resolve_hpac_probability_variant(name)
        row = {
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
            "prefix_attempted": bool(model is not None),
            "canonical_pr91_payload": canonical_payload,
        }
        if model is not None:
            try:
                decoded, decode_report = decode_tokens_hpac(
                    model,
                    payload.tokens,
                    payload.n_frames,
                    payload.height,
                    payload.width,
                    payload.predictor_count,
                    payload.delta,
                    device="cpu",
                    probability_variant=variant.name,
                    max_frames=max_frames,
                    return_report=True,
                )
                row.update(
                    {
                        "status": "passed_prefix_decode",
                        "decoded_frame0": bool(decoded.shape[0] >= 1),
                        "decoded_tokens_sha256": sha256_bytes(decoded.tobytes()),
                        "decode_report": decode_report,
                        "failure_stage": None,
                        "failure_reason": None,
                    }
                )
            except Pr86HpacReplayError as exc:
                row.update(
                    {
                        "failure_reason": exc.code,
                        "failure_context": dict(exc.fields),
                    }
                )
        rows.append(row)
    report = {
        "schema": "pr91_hpm1_probability_variant_matrix_v1",
        "tool": "tac.pr91_hpm1_codec.run_pr91_hpm1_probability_variant_matrix",
        "recorded_at_utc": datetime.now(UTC).replace(microsecond=0).isoformat(),
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


def _load_reference_tokens(
    path: Path,
    N: int,
    H: int,
    W: int,
    layout: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Load uint8 reference mask tokens into render-order ``N,H,W`` shape."""

    path = Path(path)
    if not path.is_file():
        raise Pr91Hpm1Error(
            "reference_token_contract",
            "reference_tokens_missing",
            path=path,
        )
    raw = path.read_bytes()
    expected_bytes = int(N) * int(H) * int(W)
    if len(raw) != expected_bytes:
        raise Pr91Hpm1Error(
            "reference_token_contract",
            "reference_token_size_mismatch",
            path=path,
            expected_bytes=expected_bytes,
            actual_bytes=len(raw),
            actual_sha256=sha256_bytes(raw),
        )
    if layout == "qma9_storage_wh_to_render_hw":
        arr = (
            np.frombuffer(raw, dtype=np.uint8)
            .reshape(int(N), int(W), int(H))
            .transpose(0, 2, 1)
            .copy()
        )
        storage_shape = [int(N), int(W), int(H)]
        storage_order = "frame_major_header_width_by_header_height"
        render_transform = "reshape_NWH_transpose_to_NHW"
    elif layout in {"legacy_assume_nhw", "nhw_render_order"}:
        arr = np.ascontiguousarray(
            np.frombuffer(raw, dtype=np.uint8).reshape(int(N), int(H), int(W))
        )
        storage_shape = [int(N), int(H), int(W)]
        storage_order = "frame_major_header_height_by_header_width"
        render_transform = "none"
    else:
        raise Pr91Hpm1Error(
            "reference_token_contract",
            "unsupported_reference_token_layout",
            requested_layout=layout,
            supported_layouts=[
                "qma9_storage_wh_to_render_hw",
                "legacy_assume_nhw",
                "nhw_render_order",
            ],
        )
    observed_min = int(arr.min()) if arr.size else None
    observed_max = int(arr.max()) if arr.size else None
    if (
        (observed_min is not None and observed_min < 0)
        or (observed_max is not None and observed_max > 4)
    ):
        raise Pr91Hpm1Error(
            "reference_token_contract",
            "reference_token_values_out_of_range",
            path=path,
            observed_min=observed_min,
            observed_max=observed_max,
        )
    digest = sha256_bytes(raw)
    render_digest = sha256_bytes(arr.tobytes(order="C"))
    return arr, {
        "path": repo_rel(path),
        "bytes": len(raw),
        "sha256": digest,
        "expected_sha256": EXPECTED_PR85_QMA9_TOKEN_SOURCE_SHA256,
        "matches_expected_pr85_qma9_token_source": (
            digest == EXPECTED_PR85_QMA9_TOKEN_SOURCE_SHA256
        ),
        "layout": layout,
        "storage_shape": storage_shape,
        "returned_shape": [int(N), int(H), int(W)],
        "storage_order": storage_order,
        "render_transform": render_transform,
        "render_order_sha256": render_digest,
        "observed_range": {"min": observed_min, "max": observed_max},
    }


def _symbol_position(*args: Any, **kwargs: Any) -> Any:
    raise _rehydration_failure("_symbol_position")


def _normalize_symbol_windows(
    windows: tuple[tuple[int, int], ...] | list[tuple[int, int]],
) -> tuple[tuple[int, int], ...]:
    normalized: list[tuple[int, int]] = []
    for item in windows:
        if len(item) != 2:
            raise Pr91Hpm1Error(
                "context_window_probe_contract",
                "window_must_be_start_and_count",
                window=list(item),
            )
        start, count = int(item[0]), int(item[1])
        if start < 0 or count <= 0:
            raise Pr91Hpm1Error(
                "context_window_probe_contract",
                "window_start_must_be_nonnegative_and_count_positive",
                window={"start_global_symbol": start, "count": count},
            )
        normalized.append((start, count))
    if not normalized:
        raise Pr91Hpm1Error(
            "context_window_probe_contract",
            "at_least_one_symbol_window_required",
        )
    return tuple(sorted(dict.fromkeys(normalized)))


def _context_mode_description(context_mode: str) -> str:
    if context_mode == "decoded_context":
        return (
            "Submitted stream is replayed normally: decoded symbols update the "
            "current-frame and previous-frame HPAC contexts."
        )
    if context_mode == "reference_context":
        return (
            "Submitted stream is still consumed by RangeDecoder, but after each "
            "group the HPAC model context is teacher-forced from the reference "
            "token tensor. This separates accumulated decoded-context drift "
            "from range/probability numeric mismatch."
        )
    raise Pr91Hpm1Error(
        "context_window_probe_contract",
        "unsupported_context_mode",
        context_mode=context_mode,
        supported_context_modes=list(PR91_HPM1_CONTEXT_MODES),
    )


def _reference_token_record(path: Path | None, layout: str) -> dict[str, Any]:
    if path is None:
        return {"path": None, "exists": False, "layout": layout, "loaded": False}
    resolved = Path(path)
    row: dict[str, Any] = {
        "path": repo_rel(resolved),
        "exists": resolved.is_file(),
        "layout": layout,
        "loaded": False,
    }
    if resolved.is_file():
        row.update({"bytes": resolved.stat().st_size, "sha256": sha256_path(resolved)})
    return row


def _validate_probe_variants(variants: tuple[str, ...]) -> tuple[str, ...]:
    requested = tuple(dict.fromkeys(str(name) for name in variants if str(name)))
    if not requested:
        raise Pr91Hpm1Error("probability_variant_contract", "at_least_one_variant_required")
    for name in requested:
        resolve_hpac_probability_variant(name)
    return requested


def run_pr91_hpm1_first_symbol_state_probe(
    archive: Path = DEFAULT_PR91_ARCHIVE,
    *,
    reference_tokens_path: Path | None = DEFAULT_PR85_QMA9_TOKEN_SOURCE,
    reference_layout: str = "qma9_storage_wh_to_render_hw",
    variants: tuple[str, ...] = (DEFAULT_HPAC_PROBABILITY_VARIANT,),
    symbol_count: int = 16,
    symbol_offset: int = 0,
    device: str = "cpu",
    prob_eps: float = 1e-7,
) -> dict[str, Any]:
    """Return a local-only PR91 first-symbol probe blocker report.

    The recovered pyc described a richer probability-row trace, but the live
    byte-exact HPAC stream is still not closed. This function keeps the CLI and
    downstream tooling usable without pretending the trace is promotable.
    """

    started_at = time.time()
    if device != "cpu":
        raise Pr91Hpm1Error(
            "device_contract",
            "pr91_hpm1_first_symbol_probe_is_cpu_only",
            requested_device=device,
        )
    if int(symbol_count) <= 0:
        raise Pr91Hpm1Error(
            "first_symbol_probe_contract",
            "symbol_count_must_be_positive",
            symbol_count=symbol_count,
        )
    if int(symbol_offset) < 0:
        raise Pr91Hpm1Error(
            "first_symbol_probe_contract",
            "symbol_offset_must_be_nonnegative",
            symbol_offset=symbol_offset,
        )
    requested_variants = _validate_probe_variants(variants)
    payload = extract_pr91_hpm1_payload(Path(archive))
    return _jsonable(
        {
            "schema_version": 1,
            "tool": "tac.pr91_hpm1_codec.run_pr91_hpm1_first_symbol_state_probe",
            "recorded_at_utc": datetime.now(UTC).replace(microsecond=0).isoformat(),
            "status": "blocked_hpm1_probability_range_contract_mismatch",
            "score_claim": False,
            "dispatch_performed": False,
            "dispatch_allowed": False,
            "gpu_or_remote_work": False,
            "local_only": True,
            "device": device,
            "symbol_count": int(symbol_count),
            "symbol_offset": int(symbol_offset),
            "symbol_window": {
                "start_global_symbol": int(symbol_offset),
                "requested_count": int(symbol_count),
                "end_global_symbol_exclusive": int(symbol_offset) + int(symbol_count),
            },
            "prob_eps": float(prob_eps),
            "variants": list(requested_variants),
            "archive": _archive_report(Path(archive)),
            "reference_tokens": _reference_token_record(reference_tokens_path, reference_layout),
            "hpm1_static_contract": validate_hpm1_static_contract(payload),
            "pr86_hpac_relationship": compare_hpm1_to_pr86_hpac_contract(payload),
            "payload": {
                "config": payload.config(),
                "tokens_sha256": sha256_bytes(payload.tokens),
                "hpac_sha256": sha256_bytes(payload.hpac),
            },
            "variant_results": [
                {
                    "variant": name,
                    "status": "blocked",
                    "failure_stage": "hpac_probability_range_decode",
                    "failure_reason": "first_symbol_trace_requires_byte_closed_hpm1_replay",
                    "byte_exact_replay_required": True,
                }
                for name in requested_variants
            ],
            "source_contract_summary": {
                "known_public_failure": "frame=0 group=10 symbol=191 after 5951 decoded symbols",
                "current_live_decode_surface": "bounded prefix replay fails closed before trace promotion",
            },
            "dispatch_unlocked": False,
            "pr91_ready_for_exact_eval": False,
            "failure_reason": "hpm1_probability_range_contract_not_byte_closed",
            "blocker_class": "hpm1_probability_range_contract_mismatch",
            "elapsed_sec": round(time.time() - started_at, 3),
        }
    )


def run_pr91_hpm1_context_window_probe(
    archive: Path = DEFAULT_PR91_ARCHIVE,
    *,
    reference_tokens_path: Path = DEFAULT_PR85_QMA9_TOKEN_SOURCE,
    reference_layout: str = "qma9_storage_wh_to_render_hw",
    windows: tuple[tuple[int, int], ...] = DEFAULT_PR91_HPM1_CONTEXT_WINDOWS,
    variants: tuple[str, ...] = (DEFAULT_HPAC_PROBABILITY_VARIANT,),
    context_modes: tuple[str, ...] = PR91_HPM1_CONTEXT_MODES,
    prob_eps_values: tuple[float, ...] = (1e-7,),
    device: str = "cpu",
    require_expected_pr91_identity: bool = True,
) -> dict[str, Any]:
    """Return a local-only context-window probe blocker report."""

    started_at = time.time()
    if device != "cpu":
        raise Pr91Hpm1Error(
            "device_contract",
            "pr91_hpm1_context_window_probe_is_cpu_only",
            requested_device=device,
        )
    normalized_windows = _normalize_symbol_windows(windows)
    requested_variants = _validate_probe_variants(variants)
    requested_modes = tuple(dict.fromkeys(str(mode) for mode in context_modes if str(mode)))
    if not requested_modes:
        raise Pr91Hpm1Error(
            "context_window_probe_contract",
            "at_least_one_context_mode_required",
        )
    mode_descriptions = {
        mode: _context_mode_description(mode)
        for mode in requested_modes
    }
    eps_values = tuple(float(value) for value in prob_eps_values)
    if not eps_values or any(value <= 0.0 for value in eps_values):
        raise Pr91Hpm1Error(
            "context_window_probe_contract",
            "prob_eps_values_must_be_positive",
            prob_eps_values=list(prob_eps_values),
        )
    archive_report = _archive_report(Path(archive))
    if require_expected_pr91_identity and (
        archive_report["bytes"] != EXPECTED_PR91_ARCHIVE_BYTES
        or archive_report["sha256"] != EXPECTED_PR91_ARCHIVE_SHA256
    ):
        raise Pr91Hpm1Error(
            "archive_contract",
            "expected_canonical_pr91_archive_for_context_window_probe",
            expected_bytes=EXPECTED_PR91_ARCHIVE_BYTES,
            actual_bytes=archive_report["bytes"],
            expected_sha256=EXPECTED_PR91_ARCHIVE_SHA256,
            actual_sha256=archive_report["sha256"],
        )
    payload = extract_pr91_hpm1_payload(Path(archive))
    return _jsonable(
        {
            "schema_version": 1,
            "tool": "tac.pr91_hpm1_codec.run_pr91_hpm1_context_window_probe",
            "recorded_at_utc": datetime.now(UTC).replace(microsecond=0).isoformat(),
            "status": "blocked_hpm1_probability_range_contract_mismatch",
            "score_claim": False,
            "dispatch_performed": False,
            "dispatch_allowed": False,
            "gpu_or_remote_work": False,
            "local_only": True,
            "device": device,
            "archive": archive_report,
            "require_expected_pr91_identity": bool(require_expected_pr91_identity),
            "reference_tokens": _reference_token_record(reference_tokens_path, reference_layout),
            "hpm1_static_contract": validate_hpm1_static_contract(payload),
            "pr86_hpac_relationship": compare_hpm1_to_pr86_hpac_contract(payload),
            "windows": [
                {
                    "start_global_symbol": start,
                    "count": count,
                    "end_global_symbol_exclusive": start + count,
                }
                for start, count in normalized_windows
            ],
            "variants": list(requested_variants),
            "context_modes": [
                {"mode": mode, "description": mode_descriptions[mode]}
                for mode in requested_modes
            ],
            "prob_eps_values": list(eps_values),
            "window_results": [
                {
                    "start_global_symbol": start,
                    "count": count,
                    "context_mode": mode,
                    "variant": variant,
                    "prob_eps": eps,
                    "status": "blocked",
                    "failure_stage": "hpac_probability_range_decode",
                    "failure_reason": "context_window_trace_requires_byte_closed_hpm1_replay",
                }
                for start, count in normalized_windows
                for mode in requested_modes
                for variant in requested_variants
                for eps in eps_values
            ],
            "source_contract_summary": {
                "known_public_failure": "frame=0 group=10 symbol=191 after 5951 decoded symbols",
                "current_live_decode_surface": "bounded prefix replay fails closed before context-window trace promotion",
            },
            "dispatch_unlocked": False,
            "pr91_ready_for_exact_eval": False,
            "failure_reason": "hpm1_probability_range_contract_not_byte_closed",
            "blocker_class": "hpm1_probability_range_contract_mismatch",
            "elapsed_sec": round(time.time() - started_at, 3),
        }
    )


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
        strict=True,
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
        strict=True,
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
