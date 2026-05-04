"""Local fail-closed PR91 HPM1 mask replay and re-encode helpers.

The functions in this module are forensic/preflight tooling only. They never
load contest scorers, run exact eval, dispatch GPU work, or make score claims.
"""

from __future__ import annotations

import hashlib
import json
import time
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn.functional as F

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
    DEFAULT_PR86_ARCHIVE,
    DEFAULT_HPAC_PROBABILITY_VARIANT,
    EXPECTED_PR86_TOKENS_SHA256,
    HPACMini,
    Pr86HpacReplayError,
    _categorical_from_probs,
    _group_masks,
    _normalize_probability_row,
    collect_dependency_report,
    decode_tokens_hpac,
    encode_tokens_hpac,
    encode_symbols_hpac_with_prev_context,
    load_hpac_model_from_ppmd,
    read_pr86_archive,
    resolve_hpac_probability_variant,
    sha256_bytes,
    supported_hpac_probability_variant_names,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PR91_INTAKE_DIR = REPO_ROOT / "experiments/results/public_pr91_intake_20260504_codex"
DEFAULT_PR91_RUNTIME_SOURCE_DIR = DEFAULT_PR91_INTAKE_DIR / "replay_submission/hpac_coder_hybrid"
DEFAULT_PR91_ARCHIVE = REPO_ROOT / "experiments/results/public_pr91_intake_20260504_codex/archive.zip"
DEFAULT_PR85_STBM_EXACT_DIR = (
    REPO_ROOT
    / "experiments/results/lightning_batch/"
    "exact_eval_pr85_stbm1br_stbm_runtime_t4_g4dn2x_20260504T0613Z"
)
DEFAULT_PR85_STBM_ARCHIVE = DEFAULT_PR85_STBM_EXACT_DIR / "archive.zip"
DEFAULT_PR85_STBM_ADJUDICATED_JSON = DEFAULT_PR85_STBM_EXACT_DIR / "contest_auth_eval.adjudicated.json"
DEFAULT_PR85_QMA9_TOKEN_SOURCE = (
    REPO_ROOT
    / "experiments/results/public_pr85_intake_20260503_codex/qma9_token_source/pr85_qma9_tokens_u8_storage_order.bin"
)
CONTEST_ARCHIVE_BYTE_DENOMINATOR = 37_545_489
EXPECTED_PR91_ARCHIVE_BYTES = 222_404
EXPECTED_PR91_ARCHIVE_SHA256 = "4c16d04c746c981feb902e4dd508ffadaf3615e532d351993c3d2f6eccda1b4f"
EXPECTED_PR91_MEMBER_X_BYTES = 222_304
EXPECTED_PR91_MEMBER_X_SHA256 = "5c213c61cc4d29b62286063bfdcb97e812af6b06c0021aeaecc8bc46644e17bf"
EXPECTED_PR91_HPM1_MASK_BYTES = 145_087
EXPECTED_PR91_HPM1_MASK_SHA256 = "a4ed57ff0af1d8c914f004de165aeead50ec8dd61e99b0afdfbfa2d1e7fd9fcc"
EXPECTED_PR91_HPM1_TOKENS_SHA256 = "541016d83852a5bb3e0738caa3b44d7b2b0f7372f1841085cf9554f039c6cf6b"
EXPECTED_PR91_HPM1_HPAC_SHA256 = "de7638c531c9dafa06148602cf784bf3ae9997f326f85cc25b9f3646b536abdd"
EXPECTED_PR85_QMA9_TOKEN_SOURCE_SHA256 = "c1c47434fd1e6c876cb3e44910f5ab2e124285d9dba2f300bcf322d03fb8bb5a"
EXPECTED_PR85_STBM_ARCHIVE_BYTES = 229_756
EXPECTED_PR85_STBM_ARCHIVE_SHA256 = "c6f004d444ed32c628611a2f21f567c666af6bcbcceba618cc089ec024a0cda6"
EXPECTED_PR85_STBM_MEMBER_X_SHA256 = "c7586795bb29fb0ef611ad44715aec77e0e815370e19674d4c89ef2a54b417b5"
EXPECTED_PR85_STBM_HPM1_PROJECTION_SCORE = 0.24879480490416128
DEFAULT_PR91_HPM1_CONTEXT_WINDOWS = ((33, 8), (5948, 8))
PR91_HPM1_CONTEXT_MODES = ("decoded_context", "reference_context")


class Pr91Hpm1Error(RuntimeError):
    """Raised when PR91 HPM1 preflight violates a local contract."""

    def __init__(self, stage: str, reason: str, **context: Any) -> None:
        super().__init__(reason)
        self.stage = stage
        self.reason = reason
        self.context = context


@dataclass(frozen=True)
class Hpm1MaskPayload:
    """Typed PR91 HPM1 mask payload split into token and model blobs."""

    archive_path: Path | None
    segment: bytes
    contract: Pr85SegmentContract
    tokens_blob: bytes
    hpac_ppmd_blob: bytes
    archive_report: Mapping[str, Any]
    bundle_report: Mapping[str, Any]

    @property
    def config(self) -> dict[str, Any]:
        meta = self.contract.metadata
        return {
            "N": int(meta["N"]),
            "H": int(meta["H"]),
            "W": int(meta["W"]),
            "P": int(meta["P"]),
            "delta": int(meta["delta"]),
            "ch": int(meta["ch"]),
            "use_spm": bool(meta["use_spm"]),
            "hpac_d_film": int(meta["hpac_d_film"]),
        }


def repo_rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return repo_rel(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _validate_safe_single_x_archive(archive: Path) -> tuple[bytes, dict[str, Any]]:
    if not archive.is_file():
        raise Pr91Hpm1Error("archive_contract", "archive_missing", archive=archive)
    archive_size = archive.stat().st_size
    archive_sha = sha256_path(archive)
    try:
        with zipfile.ZipFile(archive, "r") as zf:
            infos = [info for info in zf.infolist() if not info.is_dir()]
            names = [info.filename for info in infos]
            duplicate_names = sorted({name for name in names if names.count(name) > 1})
            if duplicate_names:
                raise Pr91Hpm1Error(
                    "archive_member_contract",
                    "duplicate_zip_members",
                    duplicate_member_names=duplicate_names,
                )
            if names != ["x"]:
                raise Pr91Hpm1Error(
                    "archive_member_contract",
                    "expected_single_member_x",
                    member_names=names,
                )
            member = PurePosixPath(names[0])
            if member.is_absolute() or ".." in member.parts:
                raise Pr91Hpm1Error("archive_member_contract", "unsafe_member_name", member=names[0])
            info = infos[0]
            raw = zf.read("x")
    except Pr91Hpm1Error:
        raise
    except zipfile.BadZipFile as exc:
        raise Pr91Hpm1Error("archive_contract", "bad_zip_file", error=str(exc)) from exc

    member_sha = sha256_bytes(raw)
    return raw, {
        "path": repo_rel(archive),
        "bytes": archive_size,
        "sha256": archive_sha,
        "expected_bytes": EXPECTED_PR91_ARCHIVE_BYTES,
        "expected_sha256": EXPECTED_PR91_ARCHIVE_SHA256,
        "matches_expected_pr91_archive": (
            archive_size == EXPECTED_PR91_ARCHIVE_BYTES
            and archive_sha == EXPECTED_PR91_ARCHIVE_SHA256
        ),
        "member_count": 1,
        "duplicate_member_names": [],
        "member_x": {
            "bytes": len(raw),
            "sha256": member_sha,
            "expected_bytes": EXPECTED_PR91_MEMBER_X_BYTES,
            "expected_sha256": EXPECTED_PR91_MEMBER_X_SHA256,
            "matches_expected_pr91_member": (
                len(raw) == EXPECTED_PR91_MEMBER_X_BYTES
                and member_sha == EXPECTED_PR91_MEMBER_X_SHA256
            ),
            "compress_type": int(info.compress_type),
            "compress_size": int(info.compress_size),
            "file_size": int(info.file_size),
        },
    }


def split_hpm1_mask_segment(segment: bytes) -> tuple[Pr85SegmentContract, bytes, bytes]:
    """Parse HPM1 bytes and return the typed contract plus token/model slices."""

    contract = parse_hpm1_mask_segment(segment)
    meta = contract.metadata
    token_start = HPM1_HEADER_BYTES
    token_end = token_start + int(meta["tokens_len"])
    hpac_end = token_end + int(meta["hpac_len"])
    tokens_blob = segment[token_start:token_end]
    hpac_ppmd_blob = segment[token_end:hpac_end]
    if len(tokens_blob) % 4 != 0:
        raise Pr91Hpm1Error(
            "hpm1_token_stream_contract",
            "tokens_blob_not_uint32_aligned",
            tokens_bytes=len(tokens_blob),
        )
    if sha256_bytes(tokens_blob) != meta["tokens_sha256"]:
        raise Pr91Hpm1Error("hpm1_token_stream_contract", "tokens_sha256_metadata_mismatch")
    if sha256_bytes(hpac_ppmd_blob) != meta["hpac_ppmd_sha256"]:
        raise Pr91Hpm1Error("hpm1_hpac_model_contract", "hpac_sha256_metadata_mismatch")
    return contract, tokens_blob, hpac_ppmd_blob


def extract_pr91_hpm1_payload(archive: Path = DEFAULT_PR91_ARCHIVE) -> Hpm1MaskPayload:
    """Extract the HPM1 mask payload from PR91's single-member archive."""

    raw, archive_report = _validate_safe_single_x_archive(Path(archive))
    try:
        bundle = parse_pr85_bundle(raw)
    except Pr85BundleError as exc:
        raise Pr91Hpm1Error("bundle_contract", "pr85_family_bundle_parse_failed", error=str(exc)) from exc
    segment = bytes(bundle.segments["mask"])
    if not segment.startswith(HPM1_MAGIC):
        raise Pr91Hpm1Error(
            "hpm1_mask_contract",
            "mask_segment_is_not_hpm1",
            magic=segment[:4].hex(),
        )
    contract, tokens_blob, hpac_ppmd_blob = split_hpm1_mask_segment(segment)
    bundle_report = {
        "format": bundle.format,
        "header_bytes": bundle.header_bytes,
        "segment_lengths": bundle.segment_lengths,
        "mask_sha256": contract.sha256,
        "mask_expected_bytes": EXPECTED_PR91_HPM1_MASK_BYTES,
        "mask_expected_sha256": EXPECTED_PR91_HPM1_MASK_SHA256,
        "mask_matches_expected_pr91_hpm1": (
            contract.bytes == EXPECTED_PR91_HPM1_MASK_BYTES
            and contract.sha256 == EXPECTED_PR91_HPM1_MASK_SHA256
        ),
        "fixed_length_segments": dict(bundle.fixed_length_segments),
    }
    return Hpm1MaskPayload(
        archive_path=Path(archive),
        segment=segment,
        contract=contract,
        tokens_blob=tokens_blob,
        hpac_ppmd_blob=hpac_ppmd_blob,
        archive_report=archive_report,
        bundle_report=bundle_report,
    )


def _validate_dependency_report(report: Mapping[str, Any]) -> None:
    if str(report.get("status", "")).startswith("failed_closed"):
        raise Pr91Hpm1Error("dependency_contract", str(report.get("status")), dependency_report=report)


def _common_prefix_bytes(left: bytes, right: bytes) -> int:
    count = 0
    for left_byte, right_byte in zip(left, right, strict=False):
        if left_byte != right_byte:
            break
        count += 1
    return count


def _extract_call_argument(call_text: str, argument_index: int) -> str | None:
    """Best-effort static extraction for simple source-contract reporting."""

    depth = 0
    current: list[str] = []
    args: list[str] = []
    for char in call_text:
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        if char == "," and depth == 0:
            args.append("".join(current).strip())
            current = []
            continue
        current.append(char)
    if current:
        args.append("".join(current).strip())
    if 0 <= argument_index < len(args):
        return args[argument_index]
    return None


def _extract_first_call_body(source_text: str, call_name: str) -> str:
    marker = f"{call_name}("
    start = source_text.find(marker)
    if start < 0:
        return ""
    pos = start + len(marker)
    depth = 0
    body: list[str] = []
    for char in source_text[pos:]:
        if char == "(":
            depth += 1
        elif char == ")":
            if depth == 0:
                return "".join(body)
            depth -= 1
        body.append(char)
    return ""


def analyze_pr91_hpm1_runtime_sources(
    source_dir: Path = DEFAULT_PR91_RUNTIME_SOURCE_DIR,
) -> dict[str, Any]:
    """Summarize PR91's submitted HPM1 decode contract from downloaded sources."""

    source_dir = Path(source_dir)
    report: dict[str, Any] = {
        "status": "missing",
        "source_dir": repo_rel(source_dir),
        "score_claim": False,
        "dispatch_performed": False,
        "local_only": True,
        "files": {},
        "hpm1_runtime_contract": {},
        "probability_model_contract": {},
    }
    inflate_path = source_dir / "inflate.py"
    hpac_path = source_dir / "pr86_hpac.py"
    missing = [path for path in (inflate_path, hpac_path) if not path.is_file()]
    if missing:
        report.update(
            {
                "status": "failed_closed_missing_sources",
                "missing_files": [repo_rel(path) for path in missing],
            }
        )
        return report

    inflate_text = inflate_path.read_text(encoding="utf-8")
    hpac_text = hpac_path.read_text(encoding="utf-8")
    hpm1_block = (
        inflate_text.split("bundle[\"mask\"][:4] == b\"HPM1\"", 1)[-1].split(
            "elif bundle is not None", 1
        )[0]
        if "bundle[\"mask\"][:4] == b\"HPM1\"" in inflate_text
        else ""
    )
    report["files"] = {
        "inflate.py": {
            "path": repo_rel(inflate_path),
            "bytes": inflate_path.stat().st_size,
            "sha256": sha256_path(inflate_path),
        },
        "pr86_hpac.py": {
            "path": repo_rel(hpac_path),
            "bytes": hpac_path.stat().st_size,
            "sha256": sha256_path(hpac_path),
        },
    }
    hpm1_branch_present = "bundle[\"mask\"][:4] == b\"HPM1\"" in inflate_text
    hpm1_call_body = _extract_first_call_body(hpm1_block, "decompress_pr86_hpac_tokens")
    hpm1_decode_device_argument = _extract_call_argument(hpm1_call_body, 8)
    hpm1_decode_passes_main_device = (
        "decompress_pr86_hpac_tokens(" in inflate_text
        and "str(device)" in inflate_text
    )
    explicit_cpu_force = hpm1_decode_device_argument in {"str(\"cpu\")", "str('cpu')", "\"cpu\"", "'cpu'"}
    cpu_force_comment_detected = (
        "Force HPAC decode onto CPU" in hpac_text
        or "Force HPAC decode onto CPU" in inflate_text
    )
    report.update(
        {
            "status": "passed",
            "hpm1_runtime_contract": {
                "hpm1_branch_present": hpm1_branch_present,
                "decode_function": "pr86_hpac.decompress_tokens_hpac",
                "decode_passes_main_runtime_device": hpm1_decode_passes_main_device,
                "decode_device_argument": hpm1_decode_device_argument,
                "main_device_expression_selects_cuda_when_available": (
                    "torch.device('cuda' if torch.cuda.is_available() else 'cpu')" in inflate_text
                    or "\"cuda\" if torch.cuda.is_available() else \"cpu\"" in hpac_text
                ),
                "explicit_hpac_cpu_force_detected": explicit_cpu_force,
                "hpac_cpu_force_comment_detected": cpu_force_comment_detected,
                "hpac_cpu_force_comment_matches_hpm1_call": (
                    cpu_force_comment_detected and explicit_cpu_force
                ),
                "fallback_on_hpm1_entropy_failure_detected": (
                    (
                        "except Exception" in hpm1_block
                        or "except RuntimeError" in hpm1_block
                        or "except BaseException" in hpm1_block
                    )
                    and (
                        "load_range_mask" in hpm1_block
                        or "load_encoded_mask_video" in hpm1_block
                        or "mask_frames_all" in hpm1_block.split("except", 1)[-1]
                    )
                ),
            },
            "probability_model_contract": {
                "probability_numpy_dtype": "float64" if ".astype(np.float64)" in hpac_text else "unknown",
                "categorical_perfect_false": (
                    "Categorical(probabilities=probs_np[i], perfect=False)" in hpac_text
                ),
                "probability_clip_eps": "1e-7" if "1e-7" in hpac_text else "unknown",
                "range_decoder_uint32_words": "RangeDecoder(np.frombuffer(blob, dtype=np.uint32))" in hpac_text,
                "explicit_16384_probability_grid": "16384" in hpac_text,
            },
        }
    )
    return report


def compare_hpm1_to_pr86_hpac_contract(
    payload: Hpm1MaskPayload,
    *,
    pr86_archive: Path = DEFAULT_PR86_ARCHIVE,
) -> dict[str, Any]:
    """Compare PR91's embedded HPM1 HPAC blobs against the PR86 source archive.

    This is a static custody comparison. It does not decode tokens, load
    scorers, dispatch, or imply that either HPAC stream is locally replayable.
    """

    report: dict[str, Any] = {
        "status": "unknown",
        "score_claim": False,
        "dispatch_performed": False,
        "local_only": True,
        "pr86_archive": repo_rel(Path(pr86_archive)),
        "relationship": "unknown",
        "tokens": {},
        "hpac_model": {},
    }
    try:
        pr86_bundle = read_pr86_archive(Path(pr86_archive))
    except Pr86HpacReplayError as exc:
        report.update(
            {
                "status": "failed_closed_pr86_archive_unavailable",
                "failure_stage": exc.stage,
                "failure_reason": exc.reason,
                "failure_context": _jsonable(exc.context),
            }
        )
        return report

    pr86_tokens = bytes(pr86_bundle.members["tokens.bin"])
    pr86_hpac = bytes(pr86_bundle.members["hpac.pt.ppmd"])
    prefix_bytes = _common_prefix_bytes(payload.tokens_blob, pr86_tokens)
    first_mismatch_byte = prefix_bytes if prefix_bytes < min(len(payload.tokens_blob), len(pr86_tokens)) else None
    first_mismatch_word = (
        prefix_bytes // 4
        if first_mismatch_byte is not None and prefix_bytes % 4 == 0
        else None
    )
    report.update(
        {
            "status": "passed",
            "relationship": (
                "pr91_reuses_pr86_hpac_model_with_distinct_hpm1_token_stream"
                if payload.hpac_ppmd_blob == pr86_hpac and payload.tokens_blob != pr86_tokens
                else "unexpected_pr91_pr86_hpac_relationship"
            ),
            "pr86_archive_report": pr86_bundle.report,
            "tokens": {
                "pr91_bytes": len(payload.tokens_blob),
                "pr91_sha256": sha256_bytes(payload.tokens_blob),
                "pr86_bytes": len(pr86_tokens),
                "pr86_sha256": sha256_bytes(pr86_tokens),
                "expected_pr86_sha256": EXPECTED_PR86_TOKENS_SHA256,
                "same_as_pr86_tokens": payload.tokens_blob == pr86_tokens,
                "pr91_minus_pr86_bytes": len(payload.tokens_blob) - len(pr86_tokens),
                "common_prefix_bytes": prefix_bytes,
                "common_prefix_uint32_words": prefix_bytes // 4,
                "first_mismatch_byte": first_mismatch_byte,
                "first_mismatch_uint32_word": first_mismatch_word,
            },
            "hpac_model": {
                "pr91_bytes": len(payload.hpac_ppmd_blob),
                "pr91_sha256": sha256_bytes(payload.hpac_ppmd_blob),
                "pr86_bytes": len(pr86_hpac),
                "pr86_sha256": sha256_bytes(pr86_hpac),
                "same_as_pr86_hpac_ppmd": payload.hpac_ppmd_blob == pr86_hpac,
            },
        }
    )
    return report


def validate_hpm1_static_contract(payload: Hpm1MaskPayload) -> dict[str, Any]:
    """Validate header/token/model facts without loading scorers or decoding frames."""

    meta = dict(payload.contract.metadata)
    failures: list[str] = []
    if payload.contract.bytes != EXPECTED_PR91_HPM1_MASK_BYTES:
        failures.append("mask_bytes_mismatch")
    if payload.contract.sha256 != EXPECTED_PR91_HPM1_MASK_SHA256:
        failures.append("mask_sha256_mismatch")
    if meta.get("tokens_sha256") != EXPECTED_PR91_HPM1_TOKENS_SHA256:
        failures.append("tokens_sha256_mismatch")
    if meta.get("hpac_ppmd_sha256") != EXPECTED_PR91_HPM1_HPAC_SHA256:
        failures.append("hpac_ppmd_sha256_mismatch")
    if not meta.get("tokens_uint32_aligned"):
        failures.append("tokens_not_uint32_aligned")
    if (meta.get("N"), meta.get("H"), meta.get("W")) != (600, 384, 512):
        failures.append("unexpected_hpm1_geometry")
    return {
        "status": "passed" if not failures else "failed_closed",
        "failures": failures,
        "mask": {
            "bytes": payload.contract.bytes,
            "sha256": payload.contract.sha256,
            "magic": payload.contract.magic,
            "metadata": meta,
        },
        "tokens": {
            "bytes": len(payload.tokens_blob),
            "sha256": sha256_bytes(payload.tokens_blob),
            "uint32_words": len(payload.tokens_blob) // 4,
        },
        "hpac_ppmd": {
            "bytes": len(payload.hpac_ppmd_blob),
            "sha256": sha256_bytes(payload.hpac_ppmd_blob),
        },
    }


def load_hpm1_hpac_model(payload: Hpm1MaskPayload, *, device: str = "cpu") -> tuple[HPACMini, dict[str, Any]]:
    """Load the HPM1 HPAC model blob with the PR86-compatible model contract."""

    return load_hpac_model_from_ppmd(payload.hpac_ppmd_blob, config=payload.config, device=device)


def run_pr91_hpm1_preflight(
    archive: Path = DEFAULT_PR91_ARCHIVE,
    *,
    max_frames: int | None = 1,
    attempt_reencode: bool = False,
    probability_variant: str = DEFAULT_HPAC_PROBABILITY_VARIANT,
    device: str = "cpu",
) -> dict[str, Any]:
    """Run local HPM1 preflight and return a JSON-safe fail-closed report."""

    started_at = time.time()
    report: dict[str, Any] = {
        "schema_version": 1,
        "tool": "tac.pr91_hpm1_codec.run_pr91_hpm1_preflight",
        "recorded_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "status": "running",
        "score_claim": False,
        "dispatch_performed": False,
        "gpu_or_remote_work": False,
        "local_only": True,
        "device": device,
        "max_frames": max_frames,
        "attempt_reencode": attempt_reencode,
        "probability_variant": probability_variant,
        "archive": {},
        "bundle": {},
        "hpm1_static_contract": {},
        "pr86_hpac_relationship": {},
        "dependency_contract": {},
        "hpac_model": {},
        "prefix_or_full_decode": {},
        "hpac_reencode": {},
        "failure_stage": None,
        "failure_reason": None,
        "failure_context": {},
        "blocker_class": None,
        "dispatch_unlocked": False,
    }
    try:
        dependency_report = collect_dependency_report()
        report["dependency_contract"] = dependency_report
        _validate_dependency_report(dependency_report)

        payload = extract_pr91_hpm1_payload(Path(archive))
        report["archive"] = payload.archive_report
        report["bundle"] = payload.bundle_report
        static_report = validate_hpm1_static_contract(payload)
        report["hpm1_static_contract"] = static_report
        report["pr86_hpac_relationship"] = compare_hpm1_to_pr86_hpac_contract(payload)
        if static_report["status"] != "passed":
            raise Pr91Hpm1Error(
                "hpm1_static_contract",
                "hpm1_static_contract_failed",
                failures=static_report["failures"],
            )

        model, model_report = load_hpm1_hpac_model(payload, device=device)
        report["hpac_model"] = model_report
        tokens, decode_report = decode_tokens_hpac(
            model,
            payload.tokens_blob,
            N=payload.config["N"],
            H=payload.config["H"],
            W=payload.config["W"],
            P=payload.config["P"],
            delta=payload.config["delta"],
            device=device,
            max_frames=max_frames,
            probability_variant=probability_variant,
        )
        report["prefix_or_full_decode"] = decode_report

        if attempt_reencode:
            encoded_blob, encode_report = encode_tokens_hpac(
                model,
                tokens,
                P=payload.config["P"],
                delta=payload.config["delta"],
                device=device,
                probability_variant=probability_variant,
            )
            encode_report["byte_exact_reencode"] = (
                max_frames is None and encoded_blob == payload.tokens_blob
            )
            encode_report["byte_parity_scope"] = "full_stream" if max_frames is None else "prefix_only_not_comparable"
            report["hpac_reencode"] = encode_report

        report["status"] = "passed"
        report["blocker_class"] = None
    except Pr86HpacReplayError as exc:
        report["status"] = "failed_closed"
        report["failure_stage"] = exc.stage
        report["failure_reason"] = exc.reason
        report["failure_context"] = _jsonable(exc.context)
        report["blocker_class"] = (
            "real_invalid_entropy_or_probability_model_contract_mismatch"
            if exc.stage == "submitted_tokens_decode"
            else "hpm1_hpac_replay_contract"
        )
    except Pr91Hpm1Error as exc:
        report["status"] = "failed_closed"
        report["failure_stage"] = exc.stage
        report["failure_reason"] = exc.reason
        report["failure_context"] = _jsonable(exc.context)
        report["blocker_class"] = exc.stage
    finally:
        report["elapsed_sec"] = round(time.time() - started_at, 3)
    return _jsonable(report)


def run_pr91_hpm1_probability_variant_matrix(
    archive: Path = DEFAULT_PR91_ARCHIVE,
    *,
    variants: tuple[str, ...] | None = None,
    max_frames: int | None = 1,
    attempt_reencode: bool = False,
    require_expected_pr91_identity: bool = True,
    device: str = "cpu",
) -> dict[str, Any]:
    """Probe PR91 HPM1 decode under explicit HPAC probability contracts.

    The matrix is fail-closed by construction: it is local-only, never unlocks
    dispatch, and treats prefix decode as diagnostic unless a full-stream
    decode plus byte-exact re-encode is proven.
    """

    started_at = time.time()
    requested_variants = tuple(variants or supported_hpac_probability_variant_names())
    report: dict[str, Any] = {
        "schema_version": 1,
        "tool": "tac.pr91_hpm1_codec.run_pr91_hpm1_probability_variant_matrix",
        "recorded_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "status": "running",
        "score_claim": False,
        "dispatch_performed": False,
        "gpu_or_remote_work": False,
        "local_only": True,
        "archive": repo_rel(Path(archive)),
        "device": device,
        "max_frames": max_frames,
        "attempt_reencode": attempt_reencode,
        "require_expected_pr91_identity": require_expected_pr91_identity,
        "full_stream_required_for_dispatch": True,
        "byte_exact_reencode_required_for_dispatch": True,
        "dispatch_unlocked": False,
        "pr91_ready_for_exact_eval": False,
        "local_decode_byte_parity_proven": False,
        "dependency_contract": {},
        "runtime_source_contract": {},
        "hpm1_static_contract": {},
        "pr86_hpac_relationship": {},
        "variant_results": [],
        "local_decode_variants": [],
        "byte_parity_variants": [],
        "source_contract_byte_parity_variants": [],
        "failure_reason": None,
        "blocker_class": None,
    }
    try:
        dependency_report = collect_dependency_report()
        report["dependency_contract"] = dependency_report
        _validate_dependency_report(dependency_report)

        payload = extract_pr91_hpm1_payload(Path(archive))
        report["archive"] = payload.archive_report
        report["bundle"] = payload.bundle_report
        static_report = validate_hpm1_static_contract(payload)
        report["hpm1_static_contract"] = static_report
        report["pr86_hpac_relationship"] = compare_hpm1_to_pr86_hpac_contract(payload)
        report["runtime_source_contract"] = analyze_pr91_hpm1_runtime_sources()
        if require_expected_pr91_identity and static_report["status"] != "passed":
            raise Pr91Hpm1Error(
                "hpm1_static_contract",
                "hpm1_static_contract_failed",
                failures=static_report["failures"],
            )

        model, model_report = load_hpm1_hpac_model(payload, device=device)
        report["hpac_model"] = model_report
        for variant in requested_variants:
            variant_result: dict[str, Any] = {
                "variant": variant,
                "status": "running",
                "score_claim": False,
                "dispatch_performed": False,
                "local_only": True,
                "decode": {},
                "reencode": {},
                "byte_parity_achieved": False,
                "full_stream_decode": max_frames is None,
            }
            try:
                decoded_tokens, decode_report = decode_tokens_hpac(
                    model,
                    payload.tokens_blob,
                    N=payload.config["N"],
                    H=payload.config["H"],
                    W=payload.config["W"],
                    P=payload.config["P"],
                    delta=payload.config["delta"],
                    device=device,
                    max_frames=max_frames,
                    probability_variant=variant,
                )
                variant_result["decode"] = decode_report
                variant_result["decoded_tokens"] = {
                    "shape": list(decoded_tokens.shape),
                    "dtype": str(decoded_tokens.dtype),
                    "sha256": sha256_bytes(decoded_tokens.tobytes(order="C")),
                    "min": int(decoded_tokens.min()) if decoded_tokens.size else None,
                    "max": int(decoded_tokens.max()) if decoded_tokens.size else None,
                }
                variant_result["status"] = "passed"
                report["local_decode_variants"].append(variant)

                if attempt_reencode:
                    encoded_blob, encode_report = encode_tokens_hpac(
                        model,
                        decoded_tokens,
                        P=payload.config["P"],
                        delta=payload.config["delta"],
                        device=device,
                        probability_variant=variant,
                    )
                    byte_exact = max_frames is None and encoded_blob == payload.tokens_blob
                    encode_report.update(
                        {
                            "byte_exact_reencode": byte_exact,
                            "byte_parity_scope": (
                                "full_stream" if max_frames is None else "prefix_only_not_comparable"
                            ),
                            "source_tokens_sha256": sha256_bytes(payload.tokens_blob),
                        }
                    )
                    variant_result["reencode"] = encode_report
                    variant_result["byte_parity_achieved"] = bool(byte_exact)
                    if byte_exact:
                        report["byte_parity_variants"].append(variant)
                        probability = encode_report.get("probability_variant", {})
                        if isinstance(probability, Mapping) and probability.get("source_contract") is True:
                            report["source_contract_byte_parity_variants"].append(variant)
            except Pr86HpacReplayError as exc:
                variant_result.update(
                    {
                        "status": "failed_closed",
                        "failure_stage": exc.stage,
                        "failure_reason": exc.reason,
                        "failure_context": _jsonable(exc.context),
                    }
                )
            report["variant_results"].append(variant_result)

        if report["source_contract_byte_parity_variants"] and max_frames is None:
            report["status"] = "passed"
            report["failure_reason"] = None
            report["blocker_class"] = None
            report["local_decode_byte_parity_proven"] = True
            report["pr91_ready_for_exact_eval"] = bool(require_expected_pr91_identity)
        elif report["local_decode_variants"]:
            report["status"] = "failed_closed"
            report["failure_reason"] = (
                "prefix_decode_only_without_full_source_contract_byte_parity"
                if max_frames is not None
                else "no_source_contract_variant_full_decode_byte_exact_reencode"
            )
            report["blocker_class"] = "hpm1_byte_parity_not_proven"
        else:
            report["status"] = "failed_closed"
            report["failure_reason"] = "no_probability_variant_decodes_pr91_hpm1_prefix"
            report["blocker_class"] = "real_invalid_entropy_or_probability_model_contract_mismatch"
    except Pr91Hpm1Error as exc:
        report["status"] = "failed_closed"
        report["failure_reason"] = exc.reason
        report["failure_stage"] = exc.stage
        report["failure_context"] = _jsonable(exc.context)
        report["blocker_class"] = exc.stage
    finally:
        report["elapsed_sec"] = round(time.time() - started_at, 3)
    return _jsonable(report)


def _hpm1_token_stream_transform_candidates(token_blob: bytes) -> tuple[tuple[str, str, bytes], ...]:
    if len(token_blob) % 4 != 0:
        raise Pr91Hpm1Error(
            "hpm1_token_stream_contract",
            "tokens_blob_not_uint32_aligned",
            tokens_bytes=len(token_blob),
        )
    word_chunks = [token_blob[index : index + 4] for index in range(0, len(token_blob), 4)]
    return (
        (
            "raw_le_u32",
            "Submitted HPM1 bytes interpreted as native/little-endian uint32 range-coder words.",
            token_blob,
        ),
        (
            "word_byteswap",
            "Each uint32 word byte-swapped before RangeDecoder construction.",
            b"".join(chunk[::-1] for chunk in word_chunks),
        ),
        (
            "word_reverse",
            "Uint32 word order reversed before RangeDecoder construction.",
            b"".join(reversed(word_chunks)),
        ),
        (
            "byte_reverse",
            "Entire submitted token byte stream reversed before RangeDecoder construction.",
            token_blob[::-1],
        ),
    )


def run_pr91_hpm1_stream_transform_probe(
    archive: Path = DEFAULT_PR91_ARCHIVE,
    *,
    max_frames: int | None = 1,
    probability_variant: str = DEFAULT_HPAC_PROBABILITY_VARIANT,
    device: str = "cpu",
) -> dict[str, Any]:
    """Probe low-level token stream byte/word contracts without score claims.

    This is a local-only forensic probe. It helps distinguish a genuine HPAC
    probability/model mismatch from simpler range-coder byte-order, word-order,
    or queue-orientation mistakes.
    """

    started_at = time.time()
    report: dict[str, Any] = {
        "schema_version": 1,
        "tool": "tac.pr91_hpm1_codec.run_pr91_hpm1_stream_transform_probe",
        "recorded_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "status": "running",
        "score_claim": False,
        "dispatch_performed": False,
        "gpu_or_remote_work": False,
        "local_only": True,
        "archive": repo_rel(Path(archive)),
        "device": device,
        "max_frames": max_frames,
        "probability_variant": probability_variant,
        "dispatch_unlocked": False,
        "pr91_ready_for_exact_eval": False,
        "dependency_contract": {},
        "hpm1_static_contract": {},
        "transform_results": [],
        "decode_variants": [],
        "failure_reason": None,
        "blocker_class": None,
    }
    try:
        dependency_report = collect_dependency_report()
        report["dependency_contract"] = dependency_report
        _validate_dependency_report(dependency_report)

        payload = extract_pr91_hpm1_payload(Path(archive))
        report["archive"] = payload.archive_report
        report["bundle"] = payload.bundle_report
        static_report = validate_hpm1_static_contract(payload)
        report["hpm1_static_contract"] = static_report
        if static_report["status"] != "passed":
            raise Pr91Hpm1Error(
                "hpm1_static_contract",
                "hpm1_static_contract_failed",
                failures=static_report["failures"],
            )

        model, model_report = load_hpm1_hpac_model(payload, device=device)
        report["hpac_model"] = model_report
        for name, description, token_blob in _hpm1_token_stream_transform_candidates(payload.tokens_blob):
            result: dict[str, Any] = {
                "variant": name,
                "description": description,
                "status": "running",
                "tokens_bytes": len(token_blob),
                "tokens_sha256": sha256_bytes(token_blob),
                "score_claim": False,
                "dispatch_performed": False,
                "local_only": True,
                "decode": {},
            }
            try:
                decoded_tokens, decode_report = decode_tokens_hpac(
                    model,
                    token_blob,
                    N=payload.config["N"],
                    H=payload.config["H"],
                    W=payload.config["W"],
                    P=payload.config["P"],
                    delta=payload.config["delta"],
                    device=device,
                    max_frames=max_frames,
                    probability_variant=probability_variant,
                )
                result["status"] = "passed"
                result["decode"] = decode_report
                result["decoded_tokens"] = {
                    "shape": list(decoded_tokens.shape),
                    "dtype": str(decoded_tokens.dtype),
                    "sha256": sha256_bytes(decoded_tokens.tobytes(order="C")),
                    "min": int(decoded_tokens.min()) if decoded_tokens.size else None,
                    "max": int(decoded_tokens.max()) if decoded_tokens.size else None,
                }
                report["decode_variants"].append(name)
            except Pr86HpacReplayError as exc:
                result.update(
                    {
                        "status": "failed_closed",
                        "failure_stage": exc.stage,
                        "failure_reason": exc.reason,
                        "failure_context": _jsonable(exc.context),
                    }
                )
            report["transform_results"].append(result)

        if report["decode_variants"]:
            report["status"] = "passed"
            report["failure_reason"] = None
            report["blocker_class"] = None
        else:
            report["status"] = "failed_closed"
            report["failure_reason"] = "no_token_stream_transform_decodes_pr91_hpm1_prefix"
            report["blocker_class"] = "not_byte_or_word_order_contract_mismatch"
    except Pr91Hpm1Error as exc:
        report["status"] = "failed_closed"
        report["failure_reason"] = exc.reason
        report["failure_stage"] = exc.stage
        report["failure_context"] = _jsonable(exc.context)
        report["blocker_class"] = exc.stage
    finally:
        report["elapsed_sec"] = round(time.time() - started_at, 3)
    return _jsonable(report)


def _load_reference_tokens(
    path: Path,
    *,
    N: int,
    H: int,
    W: int,
    layout: str = "qma9_storage_wh_to_render_hw",
) -> tuple[np.ndarray, dict[str, Any]]:
    path = Path(path)
    if not path.is_file():
        raise Pr91Hpm1Error("reference_token_contract", "reference_tokens_missing", path=path)
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
        arr = np.frombuffer(raw, dtype=np.uint8).reshape(int(N), int(W), int(H)).transpose(0, 2, 1)
        returned_shape = [int(N), int(H), int(W)]
        storage_shape = [int(N), int(W), int(H)]
        storage_order = "frame_major_header_width_by_header_height"
        render_transform = "reshape_NWH_transpose_to_NHW"
    elif layout == "legacy_assume_nhw":
        arr = np.frombuffer(raw, dtype=np.uint8).reshape(int(N), int(H), int(W))
        returned_shape = [int(N), int(H), int(W)]
        storage_shape = [int(N), int(H), int(W)]
        storage_order = "legacy_frame_major_header_height_by_header_width_assumption"
        render_transform = "none"
    else:
        raise Pr91Hpm1Error(
            "reference_token_contract",
            "unsupported_reference_token_layout",
            requested_layout=layout,
            supported_layouts=["qma9_storage_wh_to_render_hw", "legacy_assume_nhw"],
        )
    observed_min = int(arr.min()) if arr.size else None
    observed_max = int(arr.max()) if arr.size else None
    if observed_min is None or observed_min < 0 or observed_max is None or observed_max > 4:
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
        "matches_expected_pr85_qma9_token_source": digest == EXPECTED_PR85_QMA9_TOKEN_SOURCE_SHA256,
        "layout": layout,
        "storage_shape": storage_shape,
        "returned_shape": returned_shape,
        "storage_order": storage_order,
        "render_transform": render_transform,
        "render_order_sha256": render_digest,
        "observed_range": {"min": observed_min, "max": observed_max},
    }


def _symbol_position(mask: torch.Tensor, symbol_in_group: int) -> dict[str, int] | None:
    coords = torch.nonzero(mask, as_tuple=False)
    if symbol_in_group < 0 or symbol_in_group >= int(coords.shape[0]):
        return None
    row, col = coords[int(symbol_in_group)].detach().cpu().tolist()
    return {"y": int(row), "x": int(col)}


def _probability_row_profile(row: np.ndarray, *, reference_symbol: int, variant_name: str, prob_eps: float) -> dict[str, Any]:
    variant = resolve_hpac_probability_variant(variant_name)
    dtype = np.float32 if variant.probability_dtype == "float32" else np.float64
    clipped = np.clip(row.astype(dtype, copy=False), dtype(prob_eps), dtype(1.0))
    clipped = clipped / clipped.sum()
    return {
        "variant": variant_name,
        "probability_dtype": variant.probability_dtype,
        "categorical_perfect": variant.categorical_perfect,
        "argmax": int(clipped.argmax()),
        "min": float(clipped.min()),
        "max": float(clipped.max()),
        "reference_symbol": int(reference_symbol),
        "reference_probability": float(clipped[int(reference_symbol)]),
        "values": [float(value) for value in clipped],
    }


def _torch_scalar_at(tensor: torch.Tensor, y: int, x: int) -> int | None:
    if y < 0 or x < 0 or y >= int(tensor.shape[-2]) or x >= int(tensor.shape[-1]):
        return None
    return int(tensor[0, y, x].detach().cpu().item())


def _symbol_context_profile(cur: torch.Tensor, prev: torch.Tensor, pixel_yx: Mapping[str, int] | None) -> dict[str, int | None]:
    if pixel_yx is None:
        return {}
    y = int(pixel_yx["y"])
    x = int(pixel_yx["x"])
    return {
        "current_left": _torch_scalar_at(cur, y, x - 1),
        "current_up": _torch_scalar_at(cur, y - 1, x),
        "current_up_left": _torch_scalar_at(cur, y - 1, x - 1),
        "current_up_right": _torch_scalar_at(cur, y - 1, x + 1),
        "previous_same_pixel": _torch_scalar_at(prev, y, x),
        "previous_left": _torch_scalar_at(prev, y, x - 1),
        "previous_right": _torch_scalar_at(prev, y, x + 1),
        "previous_up": _torch_scalar_at(prev, y - 1, x),
        "previous_down": _torch_scalar_at(prev, y + 1, x),
    }


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
        start, count = (int(item[0]), int(item[1]))
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


def _window_for_symbol(
    global_symbol: int,
    windows: tuple[tuple[int, int], ...],
) -> tuple[int, int] | None:
    for start, count in windows:
        if start <= global_symbol < start + count:
            return start, count
    return None


def _window_trace_report(
    windows: tuple[tuple[int, int], ...],
    traces: Mapping[tuple[int, int], list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for start, count in windows:
        trace = traces[(start, count)]
        rows.append(
            {
                "start_global_symbol": start,
                "requested_count": count,
                "end_global_symbol_exclusive": start + count,
                "recorded_count": len(trace),
                "trace": trace,
                "trace_sha256": sha256_bytes(
                    json.dumps(_jsonable(trace), sort_keys=True, separators=(",", ":")).encode(
                        "utf-8"
                    )
                ),
            }
        )
    return rows


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


def _teacher_forced_reference_probability_windows(
    model: HPACMini,
    reference_tokens: np.ndarray,
    masks: tuple[torch.Tensor | None, ...],
    *,
    config: Mapping[str, int],
    windows: tuple[tuple[int, int], ...],
    variant_names: tuple[str, ...],
    prob_eps_values: tuple[float, ...],
    device: torch.device,
) -> list[dict[str, Any]]:
    """Record reference-context probability rows without consuming RangeDecoder."""

    max_requested_symbol = max(start + count for start, count in windows)
    results: list[dict[str, Any]] = []
    for prob_eps in prob_eps_values:
        traces: dict[tuple[int, int], list[dict[str, Any]]] = {window: [] for window in windows}
        decoded_prev = torch.zeros((1, config["H"], config["W"]), dtype=torch.long, device=device)
        global_symbol = 0
        stop = False
        for frame in range(int(config["N"])):
            idx = torch.tensor([frame], dtype=torch.long, device=device)
            cur = torch.zeros((1, config["H"], config["W"]), dtype=torch.long, device=device)
            for group, mask in enumerate(masks):
                if mask is None:
                    continue
                logits = model(cur, idx, decoded_prev)
                probs = F.softmax(logits.float(), dim=1)
                logits_at_group = logits[0][:, mask].permute(1, 0).contiguous().detach().cpu().numpy()
                probs_np = probs[0][:, mask].permute(1, 0).contiguous().detach().cpu().numpy()
                mask_np = mask.detach().cpu().numpy()
                ref_at_group = reference_tokens[frame][mask_np].astype(np.int64, copy=False)
                for symbol_in_group, row in enumerate(probs_np):
                    if global_symbol >= max_requested_symbol:
                        stop = True
                        break
                    containing_window = _window_for_symbol(global_symbol, windows)
                    if containing_window is not None:
                        pixel_yx = _symbol_position(mask, symbol_in_group)
                        reference_symbol = int(ref_at_group[symbol_in_group])
                        traces[containing_window].append(
                            {
                                "global_symbol": global_symbol,
                                "frame": frame,
                                "group": group,
                                "symbol_in_group": symbol_in_group,
                                "pixel_yx": pixel_yx,
                                "reference_symbol": reference_symbol,
                                "context_before_probability": _symbol_context_profile(
                                    cur,
                                    decoded_prev,
                                    pixel_yx,
                                ),
                                "probability_state": _probability_state_profile(
                                    row,
                                    logits_at_group[symbol_in_group],
                                    reference_symbol=reference_symbol,
                                    decoded_symbol=None,
                                    variant_names=variant_names,
                                    prob_eps=prob_eps,
                                ),
                            }
                        )
                    global_symbol += 1
                if stop:
                    break
                cur[0, mask] = torch.from_numpy(ref_at_group).to(device)
            if stop:
                break
            decoded_prev = cur.clone()
        window_results = _window_trace_report(windows, traces)
        results.append(
            {
                "mode": "reference_context_probability_only",
                "prob_eps": float(prob_eps),
                "range_decoder_consumed": False,
                "score_claim": False,
                "dispatch": False,
                "local_only": True,
                "status": "passed"
                if all(int(row["recorded_count"]) == int(row["requested_count"]) for row in window_results)
                else "failed_closed",
                "decoded_symbol_count": None,
                "global_symbols_profiled": global_symbol,
                "window_results": window_results,
                "trace_set_sha256": sha256_bytes(
                    json.dumps(
                        _jsonable(window_results),
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode("utf-8")
                ),
                "limitation": (
                    "Probability-only teacher forcing does not validate entropy bytes; "
                    "it records the HPAC model state at requested reference symbols when "
                    "RangeDecoder cannot seek through a diverged prefix."
                ),
            }
        )
    return results


def _probability_state_profile(
    probability_row: np.ndarray,
    logits_row: np.ndarray,
    *,
    reference_symbol: int | None,
    decoded_symbol: int | None,
    variant_names: tuple[str, ...],
    prob_eps: float,
) -> dict[str, Any]:
    raw_probs = probability_row.astype(np.float64, copy=False)
    logits = logits_row.astype(np.float64, copy=False)
    variants: dict[str, Any] = {}
    for variant_name in variant_names:
        variant = resolve_hpac_probability_variant(variant_name)
        normalized = _normalize_probability_row(
            probability_row,
            prob_eps=prob_eps,
            variant=variant,
        )
        order = [int(index) for index in np.argsort(-normalized)]
        row: dict[str, Any] = {
            "probability_dtype": variant.probability_dtype,
            "categorical_perfect": variant.categorical_perfect,
            "values": [float(value) for value in normalized],
            "sum": float(normalized.sum()),
            "argmax": int(normalized.argmax()),
            "rank_desc": order,
        }
        if decoded_symbol is not None:
            row["decoded_symbol_probability"] = float(normalized[int(decoded_symbol)])
            row["decoded_symbol_rank"] = int(order.index(int(decoded_symbol)))
        if reference_symbol is not None:
            row["reference_symbol_probability"] = float(normalized[int(reference_symbol)])
            row["reference_symbol_rank"] = int(order.index(int(reference_symbol)))
        variants[variant_name] = row
    return {
        "raw_softmax_float32_values": [float(value) for value in probability_row],
        "raw_softmax_float64_sum": float(raw_probs.sum()),
        "logits_float32_values": [float(value) for value in logits_row],
        "logits_argmax": int(logits.argmax()),
        "variant_rows": variants,
    }


@torch.no_grad()
def run_pr91_hpm1_reference_prefix_probe(
    archive: Path = DEFAULT_PR91_ARCHIVE,
    *,
    reference_tokens_path: Path = DEFAULT_PR85_QMA9_TOKEN_SOURCE,
    reference_layout: str = "qma9_storage_wh_to_render_hw",
    variants: tuple[str, ...] = (DEFAULT_HPAC_PROBABILITY_VARIANT,),
    max_frames: int = 1,
    device: str = "cpu",
    prob_eps: float = 1e-7,
) -> dict[str, Any]:
    """Compare locally decoded PR91 HPM1 prefixes against the PR85 QMA9 token source.

    This profiler does not prove PR91 correctness. It shrinks the unknown
    contract by checking whether a local HPAC prefix is even byte-identical to
    the recorded PR85 mask-token source before entropy decode fails.
    """

    started_at = time.time()
    report: dict[str, Any] = {
        "schema_version": 1,
        "tool": "tac.pr91_hpm1_codec.run_pr91_hpm1_reference_prefix_probe",
        "recorded_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "status": "running",
        "score_claim": False,
        "dispatch_performed": False,
        "gpu_or_remote_work": False,
        "local_only": True,
        "archive": repo_rel(Path(archive)),
        "reference_tokens_path": repo_rel(Path(reference_tokens_path)),
        "reference_layout": reference_layout,
        "device": device,
        "max_frames": max_frames,
        "variants": list(variants),
        "dependency_contract": {},
        "hpm1_static_contract": {},
        "reference_tokens": {},
        "variant_results": [],
        "source_contract_summary": {},
        "dispatch_unlocked": False,
        "pr91_ready_for_exact_eval": False,
        "failure_reason": None,
        "blocker_class": None,
    }
    try:
        dependency_report = collect_dependency_report()
        report["dependency_contract"] = dependency_report
        _validate_dependency_report(dependency_report)

        payload = extract_pr91_hpm1_payload(Path(archive))
        report["archive"] = payload.archive_report
        report["bundle"] = payload.bundle_report
        static_report = validate_hpm1_static_contract(payload)
        report["hpm1_static_contract"] = static_report
        if static_report["status"] != "passed":
            raise Pr91Hpm1Error(
                "hpm1_static_contract",
                "hpm1_static_contract_failed",
                failures=static_report["failures"],
            )

        config = payload.config
        reference_tokens, reference_report = _load_reference_tokens(
            Path(reference_tokens_path),
            N=config["N"],
            H=config["H"],
            W=config["W"],
            layout=reference_layout,
        )
        report["reference_tokens"] = reference_report
        frame_count = min(int(max_frames), int(config["N"]))
        dev = torch.device(device)
        model, model_report = load_hpm1_hpac_model(payload, device=device)
        report["hpac_model"] = model_report
        model = model.to(dev).eval()
        masks = _group_masks(config["H"], config["W"], config["P"], config["delta"], dev)

        try:
            import constriction
        except ImportError as exc:  # pragma: no cover - dependency gate catches this first.
            raise Pr86HpacReplayError("dependency_contract", "missing_constriction") from exc

        for variant_name in variants:
            variant = resolve_hpac_probability_variant(variant_name)
            variant_result: dict[str, Any] = {
                "variant": variant_name,
                "status": "running",
                "score_claim": False,
                "dispatch_performed": False,
                "local_only": True,
                "decoded_symbol_count_before_failure": None,
                "prefix_matches_reference_until_failure": True,
                "first_reference_mismatch": None,
                "decoded_prefix_sha256": None,
                "reference_prefix_sha256": None,
                "failure_context": {},
            }
            decoded_prefix = bytearray()
            reference_prefix = bytearray()
            decoded_symbols = 0
            first_reference_mismatch: dict[str, Any] | None = None
            decoded_prev = torch.zeros((1, config["H"], config["W"]), dtype=torch.long, device=dev)
            decoder = constriction.stream.queue.RangeDecoder(
                np.frombuffer(payload.tokens_blob, dtype="<u4").astype(np.uint32, copy=False)
            )

            try:
                for frame in range(frame_count):
                    idx = torch.tensor([frame], dtype=torch.long, device=dev)
                    cur = torch.zeros((1, config["H"], config["W"]), dtype=torch.long, device=dev)
                    for group, mask in enumerate(masks):
                        if mask is None:
                            continue
                        logits = model(cur, idx, decoded_prev)
                        probs = F.softmax(logits.float(), dim=1)
                        probs_at_group = probs[0][:, mask].permute(1, 0).contiguous()
                        probs_np = probs_at_group.cpu().numpy()
                        ref_at_group = reference_tokens[frame][mask.detach().cpu().numpy()]
                        decoded = np.empty(probs_np.shape[0], dtype=np.int64)
                        for symbol_in_group, row in enumerate(probs_np):
                            try:
                                symbol = int(
                                    decoder.decode(
                                        _categorical_from_probs(
                                            row,
                                            prob_eps=prob_eps,
                                            variant=variant,
                                        )
                                    )
                                )
                            except Exception as exc:
                                reference_symbol = int(ref_at_group[symbol_in_group])
                                variant_result.update(
                                    {
                                        "status": "failed_closed",
                                        "failure_stage": "submitted_tokens_decode",
                                        "failure_reason": "hpac_entropy_decode_contract_mismatch",
                                        "error_type": type(exc).__name__,
                                        "error": str(exc),
                                        "decoded_symbol_count_before_failure": decoded_symbols,
                                        "prefix_matches_reference_until_failure": (
                                            first_reference_mismatch is None
                                        ),
                                        "first_reference_mismatch": first_reference_mismatch,
                                        "decoded_prefix_sha256": sha256_bytes(bytes(decoded_prefix)),
                                        "reference_prefix_sha256": sha256_bytes(bytes(reference_prefix)),
                                        "failure_context": {
                                            "failed_at": {
                                                "frame": frame,
                                                "group": group,
                                                "symbol_in_group": symbol_in_group,
                                            },
                                            "pixel_yx": _symbol_position(mask, symbol_in_group),
                                            "reference_symbol_at_failure": reference_symbol,
                                            "probability_row_at_failure": _probability_row_profile(
                                                row,
                                                reference_symbol=reference_symbol,
                                                variant_name=variant_name,
                                                prob_eps=prob_eps,
                                            ),
                                        },
                                    }
                                )
                                raise StopIteration
                            decoded[symbol_in_group] = symbol
                            reference_symbol = int(ref_at_group[symbol_in_group])
                            if first_reference_mismatch is None and symbol != reference_symbol:
                                first_reference_mismatch = {
                                    "global_symbol": decoded_symbols,
                                    "frame": frame,
                                    "group": group,
                                    "symbol_in_group": symbol_in_group,
                                    "pixel_yx": _symbol_position(mask, symbol_in_group),
                                    "decoded": symbol,
                                    "reference": reference_symbol,
                                }
                            decoded_prefix.append(symbol)
                            reference_prefix.append(reference_symbol)
                            decoded_symbols += 1
                        cur[0, mask] = torch.from_numpy(decoded).to(dev)
                    decoded_prev = cur.clone()
                variant_result.update(
                    {
                        "status": "passed",
                        "decoded_symbol_count": decoded_symbols,
                        "prefix_matches_reference_until_failure": first_reference_mismatch is None,
                        "first_reference_mismatch": first_reference_mismatch,
                        "decoded_prefix_sha256": sha256_bytes(bytes(decoded_prefix)),
                        "reference_prefix_sha256": sha256_bytes(bytes(reference_prefix)),
                    }
                )
            except StopIteration:
                pass
            report["variant_results"].append(variant_result)

        source_result = next(
            (
                row
                for row in report["variant_results"]
                if row.get("variant") == DEFAULT_HPAC_PROBABILITY_VARIANT
            ),
            None,
        )
        if source_result is not None:
            report["source_contract_summary"] = {
                "status": source_result.get("status"),
                "decoded_symbol_count_before_failure": source_result.get(
                    "decoded_symbol_count_before_failure"
                ),
                "prefix_matches_reference_until_failure": source_result.get(
                    "prefix_matches_reference_until_failure"
                ),
                "first_reference_mismatch": source_result.get("first_reference_mismatch"),
                "failure_context": source_result.get("failure_context"),
            }

        any_reference_match = any(
            row.get("status") == "passed" and row.get("prefix_matches_reference_until_failure") is True
            for row in report["variant_results"]
        )
        if any_reference_match:
            report["status"] = "passed"
            report["failure_reason"] = None
            report["blocker_class"] = None
        else:
            report["status"] = "failed_closed"
            report["failure_reason"] = "no_local_probability_variant_proves_pr91_hpm1_pr85_reference_prefix"
            report["blocker_class"] = "hpm1_reference_identity_unproven"
    except (Pr91Hpm1Error, Pr86HpacReplayError) as exc:
        report["status"] = "failed_closed"
        report["failure_stage"] = exc.stage
        report["failure_reason"] = exc.reason
        report["failure_context"] = _jsonable(exc.context)
        report["blocker_class"] = exc.stage
    finally:
        report["elapsed_sec"] = round(time.time() - started_at, 3)
    return _jsonable(report)


@torch.no_grad()
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
    """Trace a submitted HPM1 symbol window and probability rows.

    The probe is intentionally local-only and CPU-only. It does not claim that
    the submitted stream is valid; it exposes a bounded probability/token
    window so contract hypotheses can be compared deterministically without
    dumping the full prefix.
    """

    started_at = time.time()
    report: dict[str, Any] = {
        "schema_version": 1,
        "tool": "tac.pr91_hpm1_codec.run_pr91_hpm1_first_symbol_state_probe",
        "recorded_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "status": "running",
        "score_claim": False,
        "dispatch_performed": False,
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
        "variants": list(variants),
        "archive": repo_rel(Path(archive)),
        "reference_tokens_path": repo_rel(Path(reference_tokens_path)) if reference_tokens_path else None,
        "reference_layout": reference_layout,
        "dependency_contract": {},
        "runtime_source_contract": {},
        "hpm1_static_contract": {},
        "pr86_hpac_relationship": {},
        "reference_tokens": {},
        "hpac_model": {},
        "token_stream": {},
        "variant_results": [],
        "source_contract_summary": {},
        "dispatch_unlocked": False,
        "pr91_ready_for_exact_eval": False,
        "failure_reason": None,
        "blocker_class": None,
    }
    try:
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
        window_start = int(symbol_offset)
        window_end = window_start + int(symbol_count)
        requested_variants = tuple(dict.fromkeys(str(name) for name in variants))
        for variant_name in requested_variants:
            resolve_hpac_probability_variant(variant_name)

        dependency_report = collect_dependency_report()
        report["dependency_contract"] = dependency_report
        _validate_dependency_report(dependency_report)

        payload = extract_pr91_hpm1_payload(Path(archive))
        report["archive"] = payload.archive_report
        report["bundle"] = payload.bundle_report
        static_report = validate_hpm1_static_contract(payload)
        report["hpm1_static_contract"] = static_report
        report["runtime_source_contract"] = analyze_pr91_hpm1_runtime_sources()
        report["pr86_hpac_relationship"] = compare_hpm1_to_pr86_hpac_contract(payload)
        if static_report["status"] != "passed":
            raise Pr91Hpm1Error(
                "hpm1_static_contract",
                "hpm1_static_contract_failed",
                failures=static_report["failures"],
            )

        config = payload.config
        reference_tokens: np.ndarray | None = None
        if reference_tokens_path is not None:
            reference_tokens, reference_report = _load_reference_tokens(
                Path(reference_tokens_path),
                N=config["N"],
                H=config["H"],
                W=config["W"],
                layout=reference_layout,
            )
            report["reference_tokens"] = reference_report

        model, model_report = load_hpm1_hpac_model(payload, device=device)
        report["hpac_model"] = model_report
        dev = torch.device(device)
        model = model.to(dev).eval()
        masks = _group_masks(config["H"], config["W"], config["P"], config["delta"], dev)
        token_words = np.frombuffer(payload.tokens_blob, dtype="<u4").astype(np.uint32, copy=False)
        report["token_stream"] = {
            "bytes": len(payload.tokens_blob),
            "sha256": sha256_bytes(payload.tokens_blob),
            "uint32_words": int(token_words.size),
            "first_16_uint32_words_hex": [f"{int(word):08x}" for word in token_words[:16]],
        }

        try:
            import constriction
        except ImportError as exc:  # pragma: no cover - dependency gate catches this first.
            raise Pr86HpacReplayError("dependency_contract", "missing_constriction") from exc

        for variant_name in requested_variants:
            variant = resolve_hpac_probability_variant(variant_name)
            decoder = constriction.stream.queue.RangeDecoder(token_words)
            decoded_prev = torch.zeros((1, config["H"], config["W"]), dtype=torch.long, device=dev)
            trace: list[dict[str, Any]] = []
            decoded_symbols = 0
            first_reference_mismatch: dict[str, Any] | None = None
            result: dict[str, Any] = {
                "variant": variant_name,
                "status": "running",
                "score_claim": False,
                "dispatch_performed": False,
                "local_only": True,
                "trace": trace,
                "first_reference_mismatch": None,
                "emitted_symbol_count": 0,
                "decoded_symbol_count": 0,
                "failure_context": {},
            }
            failure: dict[str, Any] | None = None
            stop = False
            for frame in range(config["N"]):
                idx = torch.tensor([frame], dtype=torch.long, device=dev)
                cur = torch.zeros((1, config["H"], config["W"]), dtype=torch.long, device=dev)
                for group, mask in enumerate(masks):
                    if mask is None:
                        continue
                    logits = model(cur, idx, decoded_prev)
                    probs = F.softmax(logits.float(), dim=1)
                    logits_at_group = logits[0][:, mask].permute(1, 0).contiguous().detach().cpu().numpy()
                    probs_at_group = probs[0][:, mask].permute(1, 0).contiguous()
                    probs_np = probs_at_group.detach().cpu().numpy()
                    decoded_group_values: list[int] = []
                    ref_at_group = (
                        reference_tokens[frame][mask.detach().cpu().numpy()]
                        if reference_tokens is not None
                        else None
                    )
                    decoded = np.empty(probs_np.shape[0], dtype=np.int64)
                    for symbol_in_group, row in enumerate(probs_np):
                        if decoded_symbols >= window_end:
                            stop = True
                            break
                        pixel_yx = _symbol_position(mask, symbol_in_group)
                        reference_symbol = (
                            int(ref_at_group[symbol_in_group])
                            if ref_at_group is not None
                            else None
                        )
                        try:
                            symbol = int(
                                decoder.decode(
                                    _categorical_from_probs(
                                        row,
                                        prob_eps=prob_eps,
                                        variant=variant,
                                    )
                                )
                            )
                        except Exception as exc:
                            failure = {
                                "failure_stage": "submitted_tokens_decode",
                                "failure_reason": "hpac_entropy_decode_contract_mismatch",
                                "error_type": type(exc).__name__,
                                "error": str(exc),
                                "decoded_symbol_count_before_failure": decoded_symbols,
                                "window_recorded_symbol_count_before_failure": len(trace),
                                "failed_at": {
                                    "frame": frame,
                                    "group": group,
                                    "symbol_in_group": symbol_in_group,
                                },
                                "pixel_yx": pixel_yx,
                                "reference_symbol_at_failure": reference_symbol,
                                "probability_state_at_failure": _probability_state_profile(
                                    row,
                                    logits_at_group[symbol_in_group],
                                    reference_symbol=reference_symbol,
                                    decoded_symbol=None,
                                    variant_names=requested_variants,
                                    prob_eps=prob_eps,
                                ),
                            }
                            stop = True
                            break

                        decoded[symbol_in_group] = symbol
                        decoded_group_values.append(symbol)
                        row_record = {
                            "global_symbol": decoded_symbols,
                            "frame": frame,
                            "group": group,
                            "symbol_in_group": symbol_in_group,
                            "pixel_yx": pixel_yx,
                            "decoded_symbol": symbol,
                            "reference_symbol": reference_symbol,
                            "matches_reference": (
                                None if reference_symbol is None else symbol == reference_symbol
                            ),
                            "context_before_decode": _symbol_context_profile(cur, decoded_prev, pixel_yx),
                            "probability_state": _probability_state_profile(
                                row,
                                logits_at_group[symbol_in_group],
                                reference_symbol=reference_symbol,
                                decoded_symbol=symbol,
                                variant_names=requested_variants,
                                prob_eps=prob_eps,
                            ),
                        }
                        if (
                            reference_symbol is not None
                            and first_reference_mismatch is None
                            and symbol != reference_symbol
                        ):
                            first_reference_mismatch = {
                                "global_symbol": decoded_symbols,
                                "frame": frame,
                                "group": group,
                                "symbol_in_group": symbol_in_group,
                                "pixel_yx": pixel_yx,
                                "decoded": symbol,
                                "reference": reference_symbol,
                            }
                        if decoded_symbols >= window_start:
                            trace.append(row_record)
                        decoded_symbols += 1
                    if stop:
                        break
                    if len(decoded_group_values) != int(probs_np.shape[0]):
                        stop = True
                        break
                    cur[0, mask] = torch.from_numpy(decoded).to(dev)
                if stop:
                    break
                decoded_prev = cur.clone()

            result["emitted_symbol_count"] = len(trace)
            result["decoded_symbol_count"] = decoded_symbols
            result["symbol_window"] = {
                "start_global_symbol": window_start,
                "requested_count": int(symbol_count),
                "end_global_symbol_exclusive": window_end,
                "recorded_count": len(trace),
            }
            result["first_reference_mismatch"] = first_reference_mismatch
            result["trace_sha256"] = sha256_bytes(
                json.dumps(_jsonable(trace), sort_keys=True, separators=(",", ":")).encode("utf-8")
            )
            if failure is not None:
                result.update(
                    {
                        "status": "failed_closed",
                        "failure_stage": failure["failure_stage"],
                        "failure_reason": failure["failure_reason"],
                        "failure_context": _jsonable(failure),
                    }
                )
            elif len(trace) == int(symbol_count):
                result["status"] = "passed"
            else:
                result.update(
                    {
                        "status": "failed_closed",
                        "failure_stage": "first_symbol_probe_contract",
                        "failure_reason": "stream_ended_before_requested_symbol_count",
                        "failure_context": {
                            "emitted_symbol_count": len(trace),
                            "decoded_symbol_count": decoded_symbols,
                            "symbol_window": result["symbol_window"],
                        },
                    }
                )
            report["variant_results"].append(result)

        source_result = next(
            (
                row
                for row in report["variant_results"]
                if row.get("variant") == DEFAULT_HPAC_PROBABILITY_VARIANT
            ),
            None,
        )
        if source_result is not None:
            first_trace = (source_result.get("trace") or [])[: int(symbol_count)]
            report["source_contract_summary"] = {
                "status": source_result.get("status"),
                "emitted_symbol_count": source_result.get("emitted_symbol_count"),
                "decoded_symbol_count": source_result.get("decoded_symbol_count"),
                "symbol_window": source_result.get("symbol_window"),
                "first_reference_mismatch": source_result.get("first_reference_mismatch"),
                "decoded_symbols": [row.get("decoded_symbol") for row in first_trace],
                "reference_symbols": [row.get("reference_symbol") for row in first_trace],
                "trace_sha256": source_result.get("trace_sha256"),
            }

        if any(row.get("status") == "passed" for row in report["variant_results"]):
            report["status"] = "passed"
            report["failure_reason"] = None
            report["blocker_class"] = None
        else:
            report["status"] = "failed_closed"
            report["failure_reason"] = "no_variant_emitted_requested_first_symbol_trace"
            report["blocker_class"] = "hpm1_first_symbol_trace_unavailable"
    except (Pr91Hpm1Error, Pr86HpacReplayError) as exc:
        report["status"] = "failed_closed"
        report["failure_stage"] = exc.stage
        report["failure_reason"] = exc.reason
        report["failure_context"] = _jsonable(exc.context)
        report["blocker_class"] = exc.stage
    finally:
        report["elapsed_sec"] = round(time.time() - started_at, 3)
    return _jsonable(report)


@torch.no_grad()
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
    """Replay bounded PR91 HPM1 symbol windows under decoded/reference context.

    The arithmetic stream is always consumed from the submitted token bytes. In
    ``reference_context`` mode only the HPAC model state is teacher-forced from
    the supplied reference token tensor after each group, which separates
    earlier decoded-context drift from probability/range-coder contract drift.
    """

    started_at = time.time()
    report: dict[str, Any] = {
        "schema_version": 1,
        "tool": "tac.pr91_hpm1_codec.run_pr91_hpm1_context_window_probe",
        "recorded_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "status": "running",
        "score_claim": False,
        "dispatch": False,
        "dispatch_performed": False,
        "gpu_or_remote_work": False,
        "local_only": True,
        "device": device,
        "archive": repo_rel(Path(archive)),
        "reference_tokens_path": repo_rel(Path(reference_tokens_path)),
        "reference_layout": reference_layout,
        "windows": [],
        "variants": list(variants),
        "context_modes": list(context_modes),
        "prob_eps_values": [float(value) for value in prob_eps_values],
        "require_expected_pr91_identity": bool(require_expected_pr91_identity),
        "dependency_contract": {},
        "runtime_source_contract": {},
        "hpm1_static_contract": {},
        "pr86_hpac_relationship": {},
        "reference_tokens": {},
        "hpac_model": {},
        "token_stream": {},
        "context_results": [],
        "teacher_forced_reference_probability_windows": [],
        "range_decoder_seek_limitation": (
            "constriction RangeDecoder is prefix-stateful; after a wrong decoded symbol "
            "is consumed, later requested windows cannot be decoded independently without "
            "the missing encoder/range state"
        ),
        "source_contract_summary": {},
        "narrowed_blocker": None,
        "candidate_next_mutation": None,
        "dispatch_unlocked": False,
        "pr91_ready_for_exact_eval": False,
        "failure_reason": None,
        "blocker_class": None,
    }
    try:
        if device != "cpu":
            raise Pr91Hpm1Error(
                "device_contract",
                "pr91_hpm1_context_window_probe_is_cpu_only",
                requested_device=device,
            )
        symbol_windows = _normalize_symbol_windows(windows)
        report["windows"] = [
            {
                "start_global_symbol": start,
                "requested_count": count,
                "end_global_symbol_exclusive": start + count,
            }
            for start, count in symbol_windows
        ]
        max_requested_symbol = max(start + count for start, count in symbol_windows)
        requested_variants = tuple(dict.fromkeys(str(name) for name in variants))
        for variant_name in requested_variants:
            resolve_hpac_probability_variant(variant_name)
        requested_modes = tuple(dict.fromkeys(str(name) for name in context_modes))
        for mode in requested_modes:
            _context_mode_description(mode)
        eps_values = tuple(float(value) for value in prob_eps_values)
        if not eps_values or any(value <= 0.0 or value >= 1.0 for value in eps_values):
            raise Pr91Hpm1Error(
                "context_window_probe_contract",
                "prob_eps_values_must_be_between_zero_and_one",
                prob_eps_values=[float(value) for value in prob_eps_values],
            )

        dependency_report = collect_dependency_report()
        report["dependency_contract"] = dependency_report
        _validate_dependency_report(dependency_report)

        payload = extract_pr91_hpm1_payload(Path(archive))
        report["archive"] = payload.archive_report
        report["bundle"] = payload.bundle_report
        static_report = validate_hpm1_static_contract(payload)
        report["hpm1_static_contract"] = static_report
        report["runtime_source_contract"] = analyze_pr91_hpm1_runtime_sources()
        report["pr86_hpac_relationship"] = compare_hpm1_to_pr86_hpac_contract(payload)
        if require_expected_pr91_identity and static_report["status"] != "passed":
            raise Pr91Hpm1Error(
                "hpm1_static_contract",
                "hpm1_static_contract_failed",
                failures=static_report["failures"],
            )

        config = payload.config
        reference_tokens, reference_report = _load_reference_tokens(
            Path(reference_tokens_path),
            N=config["N"],
            H=config["H"],
            W=config["W"],
            layout=reference_layout,
        )
        report["reference_tokens"] = reference_report

        model, model_report = load_hpm1_hpac_model(payload, device=device)
        report["hpac_model"] = model_report
        dev = torch.device(device)
        model = model.to(dev).eval()
        masks = _group_masks(config["H"], config["W"], config["P"], config["delta"], dev)
        token_words = np.frombuffer(payload.tokens_blob, dtype="<u4").astype(np.uint32, copy=False)
        report["token_stream"] = {
            "bytes": len(payload.tokens_blob),
            "sha256": sha256_bytes(payload.tokens_blob),
            "uint32_words": int(token_words.size),
            "first_16_uint32_words_hex": [f"{int(word):08x}" for word in token_words[:16]],
        }
        report["teacher_forced_reference_probability_windows"] = (
            _teacher_forced_reference_probability_windows(
                model,
                reference_tokens,
                masks,
                config=config,
                windows=symbol_windows,
                variant_names=requested_variants,
                prob_eps_values=eps_values,
                device=dev,
            )
        )

        try:
            import constriction
        except ImportError as exc:  # pragma: no cover - dependency gate catches this first.
            raise Pr86HpacReplayError("dependency_contract", "missing_constriction") from exc

        for variant_name in requested_variants:
            variant = resolve_hpac_probability_variant(variant_name)
            for context_mode in requested_modes:
                for prob_eps in eps_values:
                    decoder = constriction.stream.queue.RangeDecoder(token_words)
                    decoded_prev = torch.zeros(
                        (1, config["H"], config["W"]),
                        dtype=torch.long,
                        device=dev,
                    )
                    traces: dict[tuple[int, int], list[dict[str, Any]]] = {
                        window: [] for window in symbol_windows
                    }
                    decoded_symbols = 0
                    first_reference_mismatch: dict[str, Any] | None = None
                    result: dict[str, Any] = {
                        "variant": variant_name,
                        "context_mode": context_mode,
                        "context_mode_description": _context_mode_description(context_mode),
                        "prob_eps": float(prob_eps),
                        "status": "running",
                        "score_claim": False,
                        "dispatch": False,
                        "dispatch_performed": False,
                        "local_only": True,
                        "decoded_symbol_count": 0,
                        "emitted_window_symbol_count": 0,
                        "first_reference_mismatch": None,
                        "window_results": [],
                        "failure_context": {},
                    }
                    failure: dict[str, Any] | None = None
                    stop = False
                    for frame in range(config["N"]):
                        idx = torch.tensor([frame], dtype=torch.long, device=dev)
                        cur = torch.zeros((1, config["H"], config["W"]), dtype=torch.long, device=dev)
                        for group, mask in enumerate(masks):
                            if mask is None:
                                continue
                            logits = model(cur, idx, decoded_prev)
                            probs = F.softmax(logits.float(), dim=1)
                            logits_at_group = (
                                logits[0][:, mask].permute(1, 0).contiguous().detach().cpu().numpy()
                            )
                            probs_at_group = probs[0][:, mask].permute(1, 0).contiguous()
                            probs_np = probs_at_group.detach().cpu().numpy()
                            mask_np = mask.detach().cpu().numpy()
                            ref_at_group = reference_tokens[frame][mask_np].astype(np.int64, copy=False)
                            decoded = np.empty(probs_np.shape[0], dtype=np.int64)
                            for symbol_in_group, row in enumerate(probs_np):
                                if decoded_symbols >= max_requested_symbol:
                                    stop = True
                                    break
                                pixel_yx = _symbol_position(mask, symbol_in_group)
                                reference_symbol = int(ref_at_group[symbol_in_group])
                                try:
                                    symbol = int(
                                        decoder.decode(
                                            _categorical_from_probs(
                                                row,
                                                prob_eps=prob_eps,
                                                variant=variant,
                                            )
                                        )
                                    )
                                except Exception as exc:
                                    failure = {
                                        "failure_stage": "submitted_tokens_decode",
                                        "failure_reason": "hpac_entropy_decode_contract_mismatch",
                                        "error_type": type(exc).__name__,
                                        "error": str(exc),
                                        "decoded_symbol_count_before_failure": decoded_symbols,
                                        "failed_at": {
                                            "frame": frame,
                                            "group": group,
                                            "symbol_in_group": symbol_in_group,
                                            "global_symbol": decoded_symbols,
                                        },
                                        "pixel_yx": pixel_yx,
                                        "reference_symbol_at_failure": reference_symbol,
                                        "context_before_decode": _symbol_context_profile(
                                            cur,
                                            decoded_prev,
                                            pixel_yx,
                                        ),
                                        "probability_state_at_failure": _probability_state_profile(
                                            row,
                                            logits_at_group[symbol_in_group],
                                            reference_symbol=reference_symbol,
                                            decoded_symbol=None,
                                            variant_names=requested_variants,
                                            prob_eps=prob_eps,
                                        ),
                                    }
                                    stop = True
                                    break

                                decoded[symbol_in_group] = symbol
                                if first_reference_mismatch is None and symbol != reference_symbol:
                                    first_reference_mismatch = {
                                        "global_symbol": decoded_symbols,
                                        "frame": frame,
                                        "group": group,
                                        "symbol_in_group": symbol_in_group,
                                        "pixel_yx": pixel_yx,
                                        "decoded": symbol,
                                        "reference": reference_symbol,
                                    }
                                containing_window = _window_for_symbol(decoded_symbols, symbol_windows)
                                if containing_window is not None:
                                    traces[containing_window].append(
                                        {
                                            "global_symbol": decoded_symbols,
                                            "frame": frame,
                                            "group": group,
                                            "symbol_in_group": symbol_in_group,
                                            "pixel_yx": pixel_yx,
                                            "decoded_symbol": symbol,
                                            "reference_symbol": reference_symbol,
                                            "matches_reference": symbol == reference_symbol,
                                            "context_before_decode": _symbol_context_profile(
                                                cur,
                                                decoded_prev,
                                                pixel_yx,
                                            ),
                                            "probability_state": _probability_state_profile(
                                                row,
                                                logits_at_group[symbol_in_group],
                                                reference_symbol=reference_symbol,
                                                decoded_symbol=symbol,
                                                variant_names=requested_variants,
                                                prob_eps=prob_eps,
                                            ),
                                        }
                                    )
                                decoded_symbols += 1
                            if stop:
                                break
                            if context_mode == "reference_context":
                                context_values = ref_at_group
                            else:
                                context_values = decoded
                            cur[0, mask] = torch.from_numpy(context_values).to(dev)
                        if stop:
                            break
                        decoded_prev = cur.clone()

                    window_results = _window_trace_report(symbol_windows, traces)
                    result["window_results"] = window_results
                    result["emitted_window_symbol_count"] = sum(
                        int(row["recorded_count"]) for row in window_results
                    )
                    result["decoded_symbol_count"] = decoded_symbols
                    result["first_reference_mismatch"] = first_reference_mismatch
                    result["all_windows_complete"] = all(
                        int(row["recorded_count"]) == int(row["requested_count"])
                        for row in window_results
                    )
                    result["trace_set_sha256"] = sha256_bytes(
                        json.dumps(
                            _jsonable(window_results),
                            sort_keys=True,
                            separators=(",", ":"),
                        ).encode("utf-8")
                    )
                    if failure is not None:
                        result.update(
                            {
                                "status": "failed_closed",
                                "failure_stage": failure["failure_stage"],
                                "failure_reason": failure["failure_reason"],
                                "failure_context": _jsonable(failure),
                            }
                        )
                    elif result["all_windows_complete"]:
                        result["status"] = "passed"
                    else:
                        result.update(
                            {
                                "status": "failed_closed",
                                "failure_stage": "context_window_probe_contract",
                                "failure_reason": "stream_ended_before_requested_windows",
                                "failure_context": {
                                    "decoded_symbol_count": decoded_symbols,
                                    "max_requested_symbol_exclusive": max_requested_symbol,
                                    "window_results": [
                                        {
                                            "start_global_symbol": row["start_global_symbol"],
                                            "requested_count": row["requested_count"],
                                            "recorded_count": row["recorded_count"],
                                        }
                                        for row in window_results
                                    ],
                                },
                            }
                        )
                    report["context_results"].append(result)

        source_results = [
            row
            for row in report["context_results"]
            if row.get("variant") == DEFAULT_HPAC_PROBABILITY_VARIANT
            and float(row.get("prob_eps", 0.0)) == float(eps_values[0])
        ]
        if source_results:
            by_mode = {str(row.get("context_mode")): row for row in source_results}
            decoded_result = by_mode.get("decoded_context")
            reference_result = by_mode.get("reference_context")
            decoded_failure_symbol = (
                ((decoded_result or {}).get("failure_context") or {})
                .get("failed_at", {})
                .get("global_symbol")
            )
            reference_failure_symbol = (
                ((reference_result or {}).get("failure_context") or {})
                .get("failed_at", {})
                .get("global_symbol")
            )
            if (
                decoded_result is not None
                and reference_result is not None
                and decoded_result.get("status") == "failed_closed"
                and reference_result.get("status") == "failed_closed"
                and decoded_failure_symbol == reference_failure_symbol
            ):
                classification = "reference_context_does_not_move_entropy_failure"
            elif (
                decoded_result is not None
                and reference_result is not None
                and decoded_result.get("status") == "failed_closed"
                and reference_result.get("status") == "failed_closed"
                and isinstance(decoded_failure_symbol, int)
                and isinstance(reference_failure_symbol, int)
                and reference_failure_symbol < decoded_failure_symbol
            ):
                classification = "reference_context_fails_earlier_after_first_range_divergence"
            elif reference_result is not None and reference_result.get("status") == "passed":
                classification = "earlier_decoded_context_drift_remains_plausible"
            else:
                classification = "context_window_probe_inconclusive"
            report["source_contract_summary"] = {
                "classification": classification,
                "decoded_context": {
                    "status": (decoded_result or {}).get("status"),
                    "decoded_symbol_count": (decoded_result or {}).get("decoded_symbol_count"),
                    "first_reference_mismatch": (decoded_result or {}).get(
                        "first_reference_mismatch"
                    ),
                    "failure_context": (decoded_result or {}).get("failure_context"),
                },
                "reference_context": {
                    "status": (reference_result or {}).get("status"),
                    "decoded_symbol_count": (reference_result or {}).get("decoded_symbol_count"),
                    "first_reference_mismatch": (reference_result or {}).get(
                        "first_reference_mismatch"
                    ),
                    "failure_context": (reference_result or {}).get("failure_context"),
                },
            }

        if all(row.get("status") == "passed" for row in report["context_results"]):
            report["status"] = "passed"
            report["failure_reason"] = None
            report["blocker_class"] = None
            report["narrowed_blocker"] = (
                "requested windows replay under all requested context/probability variants; "
                "this is diagnostic only and does not prove full HPM1 parity"
            )
            report["candidate_next_mutation"] = (
                "extend recovered context/probability hypothesis to full-stream decode and "
                "byte-exact re-encode before any exact-eval readiness"
            )
        else:
            report["status"] = "failed_closed"
            report["failure_reason"] = "context_window_probe_failed_before_all_requested_windows"
            summary = report.get("source_contract_summary") or {}
            classification = summary.get("classification")
            if classification == "reference_context_does_not_move_entropy_failure":
                report["blocker_class"] = "range_probability_numeric_contract_mismatch_not_decoded_context_drift"
                report["narrowed_blocker"] = (
                    "teacher-forced reference context reaches the same submitted-token "
                    "entropy assertion as decoded context"
                )
                report["candidate_next_mutation"] = (
                    "probe the range-coder numeric contract itself: constriction queue state, "
                    "Categorical normalization/grid precision, and CPU-vs-CUDA probability traces"
                )
            elif classification == "reference_context_fails_earlier_after_first_range_divergence":
                report["blocker_class"] = "range_probability_numeric_contract_at_first_divergence"
                report["narrowed_blocker"] = (
                    "teacher-forced PR85 reference context cannot rescue PR91 after the "
                    "first local range divergence at global symbol 33; it fails before "
                    "the later 5948..5951 decoded-context window"
                )
                report["candidate_next_mutation"] = (
                    "instrument or reproduce the encoder-side range/Categorical state at "
                    "global symbol 33, then replay the submitted stream with that exact "
                    "numeric contract before extending to the 5951 failure"
                )
            elif classification == "earlier_decoded_context_drift_remains_plausible":
                report["blocker_class"] = "earlier_decoded_context_drift_changes_later_entropy_state"
                report["narrowed_blocker"] = (
                    "reference-context teacher forcing completes requested windows where "
                    "decoded-context replay fails"
                )
                report["candidate_next_mutation"] = (
                    "recover the intended source token tensor around the first divergence and "
                    "replay with that state update contract"
                )
            else:
                report["blocker_class"] = "hpm1_context_window_probe_unresolved"
                report["narrowed_blocker"] = "context-window variants did not complete deterministically"
                report["candidate_next_mutation"] = (
                    "add the smallest missing numeric/state variant shown by the failed window artifact"
                )
    except (Pr91Hpm1Error, Pr86HpacReplayError) as exc:
        report["status"] = "failed_closed"
        report["failure_stage"] = exc.stage
        report["failure_reason"] = exc.reason
        report["failure_context"] = _jsonable(exc.context)
        report["blocker_class"] = exc.stage
        report["narrowed_blocker"] = exc.reason
        report["candidate_next_mutation"] = "fix the fail-closed input contract before replaying PR91 HPM1"
    finally:
        report["elapsed_sec"] = round(time.time() - started_at, 3)
    return _jsonable(report)


def _read_single_x_archive_for_fusion(archive: Path, *, label: str) -> tuple[bytes, dict[str, Any]]:
    archive = Path(archive)
    if not archive.is_file():
        raise Pr91Hpm1Error("fusion_archive_contract", "archive_missing", label=label, archive=archive)
    archive_bytes = archive.stat().st_size
    archive_sha = sha256_path(archive)
    try:
        with zipfile.ZipFile(archive, "r") as zf:
            infos = [info for info in zf.infolist() if not info.is_dir()]
            names = [info.filename for info in infos]
            duplicate_names = sorted({name for name in names if names.count(name) > 1})
            if duplicate_names:
                raise Pr91Hpm1Error(
                    "fusion_archive_contract",
                    "duplicate_zip_members",
                    label=label,
                    duplicate_member_names=duplicate_names,
                )
            if names != ["x"]:
                raise Pr91Hpm1Error(
                    "fusion_archive_contract",
                    "expected_single_member_x",
                    label=label,
                    member_names=names,
                )
            member = PurePosixPath(names[0])
            if member.is_absolute() or ".." in member.parts:
                raise Pr91Hpm1Error(
                    "fusion_archive_contract",
                    "unsafe_member_name",
                    label=label,
                    member=names[0],
                )
            info = infos[0]
            raw = zf.read("x")
    except Pr91Hpm1Error:
        raise
    except zipfile.BadZipFile as exc:
        raise Pr91Hpm1Error(
            "fusion_archive_contract",
            "bad_zip_file",
            label=label,
            error=str(exc),
        ) from exc
    return raw, {
        "label": label,
        "path": repo_rel(archive),
        "archive_bytes": archive_bytes,
        "archive_sha256": archive_sha,
        "member_name": "x",
        "member_bytes": len(raw),
        "member_sha256": sha256_bytes(raw),
        "zip_overhead_bytes": archive_bytes - len(raw),
        "zip_compress_type": int(info.compress_type),
        "zip_file_size": int(info.file_size),
        "zip_compress_size": int(info.compress_size),
    }


def _segment_codec_label(name: str, segment: bytes) -> str:
    if name == "mask" and segment.startswith(b"STBM1BR\0"):
        return "STBM1BR"
    if name == "mask" and segment.startswith(HPM1_MAGIC):
        return "HPM1"
    if name == "mask" and segment.startswith(b"QMA"):
        return segment[:4].decode("ascii", errors="replace")
    return "opaque"


def _segment_digest_rows(left: Mapping[str, bytes], right: Mapping[str, bytes]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name in SEGMENT_ORDER:
        left_segment = bytes(left[name])
        right_segment = bytes(right[name])
        rows.append(
            {
                "segment": name,
                "same_bytes": left_segment == right_segment,
                "left_bytes": len(left_segment),
                "right_bytes": len(right_segment),
                "byte_delta_right_minus_left": len(right_segment) - len(left_segment),
                "left_sha256": sha256_bytes(left_segment),
                "right_sha256": sha256_bytes(right_segment),
                "left_codec": _segment_codec_label(name, left_segment),
                "right_codec": _segment_codec_label(name, right_segment),
            }
        )
    return rows


def _load_pr85_stbm_score_anchor(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {"status": "not_provided"}
    path = Path(path)
    if not path.is_file():
        return {"status": "missing", "path": repo_rel(path)}
    payload = json.loads(path.read_text(encoding="utf-8"))
    canonical_score = payload.get("canonical_score", payload.get("score_recomputed_from_components"))
    archive_bytes = payload.get("archive_size_bytes")
    archive_sha = (payload.get("provenance") or {}).get("archive_sha256")
    return {
        "status": "loaded",
        "path": repo_rel(path),
        "canonical_score": canonical_score,
        "score_recomputed_from_components": payload.get("score_recomputed_from_components"),
        "avg_segnet_dist": payload.get("avg_segnet_dist"),
        "avg_posenet_dist": payload.get("avg_posenet_dist"),
        "archive_size_bytes": archive_bytes,
        "archive_sha256": archive_sha,
        "n_samples": payload.get("n_samples"),
        "evidence_note": "A++ score anchor is used only for a rate-delta projection, not as an HPM1 score claim.",
    }


def plan_pr91_hpm1_pr85_stbm_fusion(
    *,
    pr85_stbm_archive: Path = DEFAULT_PR85_STBM_ARCHIVE,
    pr91_archive: Path = DEFAULT_PR91_ARCHIVE,
    pr85_stbm_adjudicated_json: Path | None = DEFAULT_PR85_STBM_ADJUDICATED_JSON,
    include_hpm1_prefix_probe: bool = True,
    hpm1_prefix_probe_max_frames: int | None = 1,
) -> dict[str, Any]:
    """Prove the byte-level PR85+STBM -> PR91/HPM1 fusion relation.

    This planner intentionally never unlocks dispatch. A smaller HPM1 mask can
    beat STBM only after the HPM1 stream proves exact decode parity under the
    submitted runtime contract. Falling back to STBM/QMA9 after HPM1 failure is
    not a valid rate win unless that fallback payload is charged in the archive.
    """

    started_at = time.time()
    report: dict[str, Any] = {
        "schema_version": 1,
        "tool": "tac.pr91_hpm1_codec.plan_pr91_hpm1_pr85_stbm_fusion",
        "recorded_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "status": "running",
        "score_claim": False,
        "dispatch_performed": False,
        "gpu_or_remote_work": False,
        "local_only": True,
        "dispatch_unlocked": False,
        "pr85_stbm_archive": repo_rel(Path(pr85_stbm_archive)),
        "pr91_archive": repo_rel(Path(pr91_archive)),
        "source_frontier_score_anchor": {},
        "segment_comparison": {},
        "hpm1_static_contract": {},
        "byte_faithful_fusion": {},
        "score_projection_if_hpm1_mask_is_semantically_identical": {},
        "hpm1_replay_gate": {},
        "fallback_semantics": {},
        "next_exact_eval_dispatch_plan_after_parity": [],
        "do_not_dispatch": [],
        "failure_reason": None,
        "blocker_class": None,
    }
    try:
        stbm_raw, stbm_archive_report = _read_single_x_archive_for_fusion(
            Path(pr85_stbm_archive),
            label="pr85_stbm_frontier",
        )
        pr91_raw, pr91_archive_report = _read_single_x_archive_for_fusion(
            Path(pr91_archive),
            label="pr91_hpm1_source",
        )
        try:
            stbm_bundle = parse_pr85_bundle(stbm_raw)
            pr91_bundle = parse_pr85_bundle(pr91_raw)
        except Pr85BundleError as exc:
            raise Pr91Hpm1Error("fusion_bundle_contract", "pr85_family_parse_failed", error=str(exc)) from exc

        stbm_segments = dict(stbm_bundle.segments)
        pr91_segments = dict(pr91_bundle.segments)
        stbm_mask = bytes(stbm_segments["mask"])
        hpm1_mask = bytes(pr91_segments["mask"])
        if not stbm_mask.startswith(b"STBM1BR\0"):
            raise Pr91Hpm1Error(
                "fusion_source_contract",
                "source_frontier_mask_is_not_stbm1br",
                source_mask_magic=stbm_mask[:8].hex(),
            )
        if not hpm1_mask.startswith(HPM1_MAGIC):
            raise Pr91Hpm1Error(
                "fusion_source_contract",
                "pr91_mask_is_not_hpm1",
                pr91_mask_magic=hpm1_mask[:8].hex(),
            )
        hpm1_contract, hpm1_tokens_blob, hpm1_hpac_blob = split_hpm1_mask_segment(hpm1_mask)
        report["hpm1_static_contract"] = {
            "status": "passed",
            "mask": {
                "bytes": hpm1_contract.bytes,
                "sha256": hpm1_contract.sha256,
                "metadata": dict(hpm1_contract.metadata),
            },
            "tokens": {
                "bytes": len(hpm1_tokens_blob),
                "sha256": sha256_bytes(hpm1_tokens_blob),
            },
            "hpac_ppmd": {
                "bytes": len(hpm1_hpac_blob),
                "sha256": sha256_bytes(hpm1_hpac_blob),
            },
        }

        rows = _segment_digest_rows(stbm_segments, pr91_segments)
        non_mask_rows = [row for row in rows if row["segment"] != "mask"]
        all_non_mask_identical = all(bool(row["same_bytes"]) for row in non_mask_rows)
        changed_segments = [row["segment"] for row in rows if not row["same_bytes"]]
        report["segment_comparison"] = {
            "status": "passed" if changed_segments == ["mask"] and all_non_mask_identical else "failed",
            "stbm_archive": stbm_archive_report,
            "pr91_archive": pr91_archive_report,
            "stbm_bundle": {
                "format": stbm_bundle.format,
                "header_bytes": stbm_bundle.header_bytes,
                "segment_lengths": stbm_bundle.segment_lengths,
            },
            "pr91_bundle": {
                "format": pr91_bundle.format,
                "header_bytes": pr91_bundle.header_bytes,
                "segment_lengths": pr91_bundle.segment_lengths,
            },
            "all_non_mask_segments_identical": all_non_mask_identical,
            "changed_segments": changed_segments,
            "segments": rows,
        }

        header_mode = "v5" if stbm_bundle.header_bytes == 24 else "explicit_30"
        fused_segments = dict(stbm_segments)
        fused_segments["mask"] = hpm1_mask
        fused_member = pack_pr85_bundle(fused_segments, header_mode=header_mode)
        fusion_member_equals_pr91 = fused_member == pr91_raw
        byte_delta = int(pr91_archive_report["archive_bytes"]) - int(stbm_archive_report["archive_bytes"])
        mask_byte_delta = len(hpm1_mask) - len(stbm_mask)
        rate_score_delta = 25.0 * byte_delta / float(CONTEST_ARCHIVE_BYTE_DENOMINATOR)

        score_anchor = _load_pr85_stbm_score_anchor(pr85_stbm_adjudicated_json)
        report["source_frontier_score_anchor"] = score_anchor
        base_score = score_anchor.get("canonical_score")
        projected_score = None
        if isinstance(base_score, (int, float)):
            projected_score = float(base_score) + rate_score_delta
        report["byte_faithful_fusion"] = {
            "exists": bool(changed_segments == ["mask"] and all_non_mask_identical and fusion_member_equals_pr91),
            "header_mode": header_mode,
            "fusion_member_equals_existing_pr91_member": fusion_member_equals_pr91,
            "fusion_member_bytes": len(fused_member),
            "fusion_member_sha256": sha256_bytes(fused_member),
            "existing_pr91_member_sha256": pr91_archive_report["member_sha256"],
            "archive_bytes_delta_vs_pr85_stbm": byte_delta,
            "mask_bytes_delta_hpm1_vs_stbm": mask_byte_delta,
            "rate_score_delta_if_archive_is_valid": rate_score_delta,
            "component_identity_required": True,
            "semantic_identity_evidence_required": "full HPM1 decode plus byte/semantic parity against PR85/STBM masks",
        }
        report["score_projection_if_hpm1_mask_is_semantically_identical"] = {
            "evidence_grade": "prediction",
            "score_claim": False,
            "base_pr85_stbm_score": base_score,
            "base_score_source": repo_rel(Path(pr85_stbm_adjudicated_json))
            if pr85_stbm_adjudicated_json is not None
            else None,
            "rate_score_delta": rate_score_delta,
            "projected_score": projected_score,
            "expected_public_pr91_self_report": EXPECTED_PR85_STBM_HPM1_PROJECTION_SCORE,
            "expected_public_pr91_self_report_abs_delta": (
                None
                if projected_score is None
                else abs(projected_score - EXPECTED_PR85_STBM_HPM1_PROJECTION_SCORE)
            ),
            "matches_expected_public_pr91_self_report": (
                projected_score is not None
                and abs(projected_score - EXPECTED_PR85_STBM_HPM1_PROJECTION_SCORE) < 1e-6
            ),
            "projection_formula": "base_score + 25 * archive_byte_delta / 37545489",
        }

        if include_hpm1_prefix_probe:
            report["hpm1_replay_gate"] = run_pr91_hpm1_preflight(
                Path(pr91_archive),
                max_frames=hpm1_prefix_probe_max_frames,
                attempt_reencode=False,
            )
        else:
            report["hpm1_replay_gate"] = {
                "status": "not_run",
                "reason": "include_hpm1_prefix_probe_false",
                "full_decode_byte_parity_proven": False,
            }

        report["fallback_semantics"] = {
            "status": "fail_closed_required",
            "hpm1_entropy_failure_must_abort": True,
            "fallback_to_stbm_or_qma9_after_hpm1_failure_allowed": False,
            "reason": (
                "A fallback to the STBM/QMA9 mask stream would consume bytes not present in the "
                "HPM1 archive unless the fallback payload is also charged. That is not a valid "
                "way to claim the 7352-byte rate reduction."
            ),
            "submitted_pr91_runtime_source_contract": analyze_pr91_hpm1_runtime_sources(),
        }

        fusion_exists = bool(report["byte_faithful_fusion"]["exists"])
        prefix_status = str(report["hpm1_replay_gate"].get("status"))
        report["next_exact_eval_dispatch_plan_after_parity"] = [
            "Prove full HPM1 decode under the submitted runtime contract and record decoded mask SHA/class counts.",
            "Prove PR91/HPM1 decoded masks are byte-identical or semantically identical to the PR85+STBM render-order mask tensor.",
            "Prove full HPM1 decode plus byte-exact re-encode or an equivalent reviewed entropy trace for the submitted token stream.",
            "Claim a lane with tools/claim_lane_dispatch.py before any exact-eval job.",
            "Run archive.zip -> inflate.sh -> upstream/evaluate.py on T4/equivalent with no STBM/QMA9 fallback path.",
            "Adjudicate contest_auth_eval.json and compare archive SHA plus inflate runtime tree SHA before promotion.",
        ]
        report["do_not_dispatch"] = [
            "PR91/HPM1 or HPM1+QRGB candidates while HPM1 local/exact replay fails entropy decode.",
            "Any HPM1 archive whose inflate runtime falls back to STBM/QMA9 after entropy failure.",
            "Any score projection that assumes PR85 components without decoded-mask parity evidence.",
        ]
        if not fusion_exists:
            report["status"] = "failed_closed"
            report["failure_reason"] = "byte_faithful_pr85_stbm_to_pr91_fusion_not_proven"
            report["blocker_class"] = "fusion_byte_contract"
        elif prefix_status == "failed_closed":
            report["status"] = "blocked_hpm1_replay_failed"
            report["failure_reason"] = report["hpm1_replay_gate"].get("failure_reason")
            report["blocker_class"] = report["hpm1_replay_gate"].get("blocker_class")
        else:
            report["status"] = "blocked_pending_hpm1_full_decode_byte_parity"
            report["failure_reason"] = "hpm1_full_decode_byte_parity_not_proven"
            report["blocker_class"] = "hpm1_parity_gate"
    except (Pr91Hpm1Error, Pr86HpacReplayError) as exc:
        report["status"] = "failed_closed"
        report["failure_stage"] = exc.stage
        report["failure_reason"] = exc.reason
        report["failure_context"] = _jsonable(exc.context)
        report["blocker_class"] = exc.stage
    finally:
        report["elapsed_sec"] = round(time.time() - started_at, 3)
    return _jsonable(report)


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
    """Build an HPM1 segment from explicit charged token/model bytes."""

    if len(tokens_blob) <= 0 or len(tokens_blob) % 4 != 0:
        raise Pr91Hpm1Error(
            "hpm1_encoder_contract",
            "tokens_blob_must_be_nonempty_uint32_words",
            tokens_bytes=len(tokens_blob),
        )
    if len(hpac_ppmd_blob) <= 0:
        raise Pr91Hpm1Error("hpm1_encoder_contract", "hpac_ppmd_blob_must_be_nonempty")
    header = HPM1_MAGIC + b"".join(
        int(value).to_bytes(4, "little")
        for value in (
            N,
            H,
            W,
            P,
            delta,
            ch,
            1 if use_spm else 0,
            hpac_d_film,
            len(tokens_blob),
            len(hpac_ppmd_blob),
            ppmd_order,
        )
    )
    segment = header + tokens_blob + hpac_ppmd_blob
    parse_hpm1_mask_segment(segment)
    return segment


def prototype_reencode_hpm1_from_raw_tokens(
    raw_tokens: np.ndarray,
    source_payload: Hpm1MaskPayload,
    *,
    max_frames: int | None = None,
    probability_variant: str = DEFAULT_HPAC_PROBABILITY_VARIANT,
    device: str = "cpu",
) -> dict[str, Any]:
    """Local-only HPM1 re-encode prototype from decoded mask tokens.

    This is a byte-construction prototype, not a dispatchable candidate. It is
    useful only after decoded PR85/PR91 mask-token custody is already proven.
    """

    started_at = time.time()
    if raw_tokens.ndim != 3:
        raise Pr91Hpm1Error("hpm1_encoder_contract", "raw_tokens_must_be_nhw", shape=list(raw_tokens.shape))
    config = source_payload.config
    frame_count = int(raw_tokens.shape[0]) if max_frames is None else min(int(max_frames), int(raw_tokens.shape[0]))
    if tuple(raw_tokens.shape[1:]) != (config["H"], config["W"]):
        raise Pr91Hpm1Error(
            "hpm1_encoder_contract",
            "raw_token_geometry_mismatch",
            raw_shape=list(raw_tokens.shape),
            expected=[config["N"], config["H"], config["W"]],
        )
    model, model_report = load_hpm1_hpac_model(source_payload, device=device)
    encoded_blob, encode_report = encode_tokens_hpac(
        model,
        raw_tokens[:frame_count].astype(np.uint8, copy=False),
        P=config["P"],
        delta=config["delta"],
        device=device,
        probability_variant=probability_variant,
    )
    segment = build_hpm1_mask_segment(
        encoded_blob,
        source_payload.hpac_ppmd_blob,
        N=frame_count,
        H=config["H"],
        W=config["W"],
        P=config["P"],
        delta=config["delta"],
        ch=config["ch"],
        use_spm=config["use_spm"],
        hpac_d_film=config["hpac_d_film"],
        ppmd_order=int(source_payload.contract.metadata["ppmd_order"]),
    )
    contract = parse_hpm1_mask_segment(segment)
    return {
        "schema_version": 1,
        "tool": "tac.pr91_hpm1_codec.prototype_reencode_hpm1_from_raw_tokens",
        "recorded_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "status": "passed",
        "score_claim": False,
        "dispatch_performed": False,
        "gpu_or_remote_work": False,
        "local_only": True,
        "prototype_only": True,
        "source_archive": repo_rel(source_payload.archive_path) if source_payload.archive_path else None,
        "frames_encoded": frame_count,
        "input_tokens_sha256": sha256_bytes(raw_tokens[:frame_count].tobytes(order="C")),
        "hpac_model": model_report,
        "hpm1_encode": encode_report,
        "candidate_hpm1_segment": {
            "bytes": contract.bytes,
            "sha256": contract.sha256,
            "tokens_len": contract.metadata["tokens_len"],
            "hpac_len": contract.metadata["hpac_len"],
            "mask_bytes_vs_pr91_hpm1": contract.bytes - EXPECTED_PR91_HPM1_MASK_BYTES,
        },
        "elapsed_sec": round(time.time() - started_at, 3),
    }


def raw_tokens_to_mod5_residual_symbols(raw_tokens: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return mod-5 residual symbols plus raw previous-frame context tokens."""

    if raw_tokens.ndim != 3:
        raise Pr91Hpm1Error("hpm1_residual_contract", "raw_tokens_must_be_nhw", shape=list(raw_tokens.shape))
    raw = raw_tokens.astype(np.uint8, copy=False)
    if raw.size and (int(raw.min()) < 0 or int(raw.max()) > 4):
        raise Pr91Hpm1Error(
            "hpm1_residual_contract",
            "raw_token_class_value_out_of_range",
            min_value=int(raw.min()),
            max_value=int(raw.max()),
        )
    prev = np.zeros_like(raw, dtype=np.uint8)
    if raw.shape[0] > 1:
        prev[1:] = raw[:-1]
    symbols = ((raw.astype(np.int16) - prev.astype(np.int16)) % 5).astype(np.uint8)
    return symbols, prev


def reconstruct_raw_tokens_from_mod5_residual_symbols(
    symbols: np.ndarray,
    prev_context_tokens: np.ndarray,
) -> np.ndarray:
    """Reconstruct raw token maps from mod-5 residual symbols and previous context."""

    if symbols.shape != prev_context_tokens.shape:
        raise Pr91Hpm1Error(
            "hpm1_residual_contract",
            "residual_prev_shape_mismatch",
            symbols_shape=list(symbols.shape),
            prev_context_shape=list(prev_context_tokens.shape),
        )
    return ((symbols.astype(np.int16) + prev_context_tokens.astype(np.int16)) % 5).astype(np.uint8)


def prototype_reencode_hpm1_residual_from_raw_tokens(
    raw_tokens: np.ndarray,
    source_payload: Hpm1MaskPayload,
    *,
    max_frames: int | None = None,
    probability_variant: str = DEFAULT_HPAC_PROBABILITY_VARIANT,
    device: str = "cpu",
) -> dict[str, Any]:
    """Local-only HPM1 residual-symbol prototype from decoded mask tokens."""

    started_at = time.time()
    if raw_tokens.ndim != 3:
        raise Pr91Hpm1Error("hpm1_residual_contract", "raw_tokens_must_be_nhw", shape=list(raw_tokens.shape))
    config = source_payload.config
    frame_count = int(raw_tokens.shape[0]) if max_frames is None else min(int(max_frames), int(raw_tokens.shape[0]))
    if tuple(raw_tokens.shape[1:]) != (config["H"], config["W"]):
        raise Pr91Hpm1Error(
            "hpm1_residual_contract",
            "raw_token_geometry_mismatch",
            raw_shape=list(raw_tokens.shape),
            expected=[config["N"], config["H"], config["W"]],
        )
    raw_prefix = raw_tokens[:frame_count].astype(np.uint8, copy=False)
    residual_symbols, prev_context = raw_tokens_to_mod5_residual_symbols(raw_prefix)
    reconstructed = reconstruct_raw_tokens_from_mod5_residual_symbols(residual_symbols, prev_context)
    if not np.array_equal(reconstructed, raw_prefix):
        raise Pr91Hpm1Error("hpm1_residual_contract", "residual_roundtrip_failed")

    model, model_report = load_hpm1_hpac_model(source_payload, device=device)
    encoded_blob, encode_report = encode_symbols_hpac_with_prev_context(
        model,
        residual_symbols,
        prev_context,
        P=config["P"],
        delta=config["delta"],
        device=device,
        probability_variant=probability_variant,
    )
    segment = build_hpm1_mask_segment(
        encoded_blob,
        source_payload.hpac_ppmd_blob,
        N=frame_count,
        H=config["H"],
        W=config["W"],
        P=config["P"],
        delta=config["delta"],
        ch=config["ch"],
        use_spm=config["use_spm"],
        hpac_d_film=config["hpac_d_film"],
        ppmd_order=int(source_payload.contract.metadata["ppmd_order"]),
    )
    contract = parse_hpm1_mask_segment(segment)
    return {
        "schema_version": 1,
        "tool": "tac.pr91_hpm1_codec.prototype_reencode_hpm1_residual_from_raw_tokens",
        "recorded_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "status": "passed",
        "score_claim": False,
        "dispatch_performed": False,
        "gpu_or_remote_work": False,
        "local_only": True,
        "prototype_only": True,
        "source_archive": repo_rel(source_payload.archive_path) if source_payload.archive_path else None,
        "frames_encoded": frame_count,
        "input_tokens_sha256": sha256_bytes(raw_prefix.tobytes(order="C")),
        "residual_symbols_sha256": sha256_bytes(residual_symbols.tobytes(order="C")),
        "residual_roundtrip_raw_tokens_sha256": sha256_bytes(reconstructed.tobytes(order="C")),
        "hpac_model": model_report,
        "hpm1_encode": encode_report,
        "candidate_hpm1_segment": {
            "bytes": contract.bytes,
            "sha256": contract.sha256,
            "tokens_len": contract.metadata["tokens_len"],
            "hpac_len": contract.metadata["hpac_len"],
            "mask_bytes_vs_pr91_hpm1": contract.bytes - EXPECTED_PR91_HPM1_MASK_BYTES,
        },
        "elapsed_sec": round(time.time() - started_at, 3),
    }


def write_json_report(report: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")
