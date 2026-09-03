#!/usr/bin/env python3
"""Fit the DDM CL1 model-byte/token-byte exchange from serialized rows.

The fitter has two admissible input shapes.  The interim shape contains the
two byte-identical lambda-1 interruption controls and lambda 0.5; it emits only
the exact adjacent secant and either stops the formulation or requests the
preregistered lambda-0.25 rung.  The final shape adds lambda 0.25 and fits the
three unique lambda representatives.  A duplicated control never becomes a
pseudo-replicate in the regression.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import platform
import subprocess
import sys
from collections import defaultdict
from itertools import pairwise
from pathlib import Path
from types import MappingProxyType
from typing import Any, cast

import numpy as np
import torch

INPUT_SCHEMA = "ddm_cl1_capacity_measurement_set.v1"
OUTPUT_SCHEMA = "ddm_cl1_capacity_fit.v2"
ATTESTATION_SCHEMA = "ddm_cl1_hpac_artifact_attestation.v1"
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tac.admission_guard import assert_governed_admission  # noqa: E402

TRAINER_PATH = REPO_ROOT / "tools/train_ddm_cl1_hpac_capacity.py"
MEASUREMENT_ROOT = Path("/Volumes/VertigoDataTier/pact/ddm_cl1_capacity_20260809")
INTAKE_CODE_ROOT = Path("/Volumes/VertigoDataTier/pact/pr130_eureka_intake_20260806/repro_repo/code")
EXPECTED_CAUSAL_SOURCE_SHA256 = MappingProxyType(
    {
        "pack_hpac_self_compress.py": ("e796d9249926f8c7dcc45a7cdf1f39e33d0b4409ffee275fbb9cd481a6f5f099"),
        "codec_hpac_integer.py": ("70632168250cbecc40b9d6de5da5b167adeb56031368311ff936404a1ceba7e0"),
        "codec_hpac_residual.py": ("3a879556de59569fb382ee0bb7d1330bdc637feeaed62ca7167148e81512d90e"),
        "hpac_integer.py": ("6e6b4f4d0b293fb60cc1b751958756a4cd6c2ce7bcff68c6f03e20277856803f"),
        "hpac_integer_sparse.py": ("2240ee32c53fe949b560d316d349e0bbdccc0ceb78787307cd4d530623d42a0c"),
        "pack_hpac_quantized.py": ("c72b90e17ae1268c12118492df3dfaadc8a63abfefe17f2cabd296d8d879a7c1"),
        "hpac_self_compress.py": ("d63d67945a0719ebbe72da36e6c99909557219360d50e74f20577d68d678beec"),
    }
)
PACK_CAUSAL_SOURCES = (
    "pack_hpac_self_compress.py",
    "hpac_integer.py",
    "hpac_self_compress.py",
)
CODEC_CAUSAL_SOURCES = (
    "codec_hpac_integer.py",
    "codec_hpac_residual.py",
    "hpac_integer.py",
    "hpac_integer_sparse.py",
    "pack_hpac_quantized.py",
    "hpac_self_compress.py",
)
PYTHON_PATH = REPO_ROOT / ".venv/bin/python"
CANONICAL_CACHE_PATH = Path(
    "/Volumes/VertigoDataTier/pact/ddm_op1r_20260809/authority_cache/gt_cache_600_official_ada.pt"  # GT_LINEAGE_OK: canonical bytes are registry-classified DALI_NVDEC sha256 382d7dfe38b37c0c
)
CANONICAL_INIT_PATH = Path(
    "/Volumes/VertigoDataTier/pact/ddm_hb1_20260806/checkpoints/gt/hpac_p64_exact_from_archive.pt"
)
CANONICAL_TOPOLOGY = (64, 64, 2, 8)
PIXELS = 600 * 384 * 512
EXPECTED_CACHE_SHA256 = "382d7dfe38b37c0cc5017e5645032faa045af6924db66e0b67549cc96c840195"
EXPECTED_INIT_SHA256 = "0e6c30cef6b36c4e530779c92c56e9128c1d86c62e85e9fc5358a7e9f40ec985"
EXPECTED_RAW_TOKEN_SHA256 = "c5c7671d037b6912980c57929a5b6d789d250ee6a93e3b0a6018cf9f63e32ece"
TOKEN_CODER = "queue.RangeEncoder(perfect=False)"
RUNG_LAMBDAS = {
    "lambda_1p0_resume_control": 1.0,
    "lambda_1p0_uninterrupted_twin": 1.0,
    "lambda_0p5": 0.5,
    "lambda_0p25": 0.25,
}
RUNG_ORDER = (1.0, 0.5, 0.25)
PR130_NAME = "PR130_CPR1_immutable_reference"
PR130_PACKED_MODEL_BYTES = 15_164
PR130_IDEAL_TOKEN_BYTES = 114_852
PR130_RANGE_TOKEN_BYTES = 116_980
PR130_IDEAL_JOINT_BYTES = 130_016
PR130_RANGE_JOINT_BYTES = 132_144
PR130_REFERENCE = MappingProxyType(
    {
        "name": PR130_NAME,
        "packed_model_bytes": PR130_PACKED_MODEL_BYTES,
        "packed_model_sha256": ("ef8bb9d59bdd3916fb77713c11cdcb85e029f01d80b82472a40ab28f7e56a9ee"),
        "ideal_token_bytes": PR130_IDEAL_TOKEN_BYTES,
        "range_token_bytes": PR130_RANGE_TOKEN_BYTES,
        "range_token_sha256": ("948379872ff81a4e5d948ec301c143be00ebd0033544c8abdfb4af0f4c4a15eb"),
        "ideal_joint_bytes": PR130_IDEAL_JOINT_BYTES,
        "range_joint_bytes": PR130_RANGE_JOINT_BYTES,
    }
)
ARTIFACT_PATH_FIELDS = (
    "selected_checkpoint_path",
    "packed_model_path",
    "range_token_path",
    "decoded_raw_token_path",
    "pack_report_path",
    "encode_report_path",
    "decode_report_path",
    "pack_attestation_path",
    "encode_attestation_path",
    "decode_attestation_path",
    "pack_receipt_path",
    "encode_receipt_path",
    "decode_receipt_path",
)
RECEIPT_HASH_FIELDS = (
    "pack_receipt_sha256",
    "encode_receipt_sha256",
    "decode_receipt_sha256",
)
CONTROL_EQUAL_FIELDS = (
    "causal_state_sha256",
    "packed_model_bytes",
    "packed_model_sha256",
    "ideal_token_bytes",
    "range_token_bytes",
    "range_token_sha256",
    "encoded_raw_token_sha256",
    "decoded_raw_token_sha256",
)


class CL1FitError(RuntimeError):
    """Fail-closed error for incomplete or incomparable CL1 rows."""


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise CL1FitError(f"JSON root must be an object: {path}")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _file_record(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise CL1FitError(f"artifact is absent: {resolved}")
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": _sha256_file(resolved),
    }


def _verified_causal_sources(operation: str) -> dict[str, str]:
    names = PACK_CAUSAL_SOURCES if operation == "pack" else CODEC_CAUSAL_SOURCES
    observed: dict[str, str] = {}
    for name in names:
        path = INTAKE_CODE_ROOT / name
        if not path.is_file():
            raise CL1FitError(f"pinned HPAC causal source is absent: {path}")
        digest = _sha256_file(path)
        expected = EXPECTED_CAUSAL_SOURCE_SHA256[name]
        if digest != expected:
            raise CL1FitError(f"pinned HPAC causal source changed for {name}: expected {expected}, observed {digest}")
        observed[name] = digest
    return observed


def _verify_canonical_cache() -> None:
    if not CANONICAL_CACHE_PATH.is_file():
        raise CL1FitError(f"canonical DALI cache is absent: {CANONICAL_CACHE_PATH}")
    observed = _sha256_file(CANONICAL_CACHE_PATH)
    if observed != EXPECTED_CACHE_SHA256:
        raise CL1FitError(f"canonical DALI cache bytes changed: expected {EXPECTED_CACHE_SHA256}, observed {observed}")


def _artifact_runtime_identity() -> dict[str, Any]:
    relevant_env = (
        "PYTHONHASHSEED",
        "PYTORCH_ENABLE_MPS_FALLBACK",
        "PYTORCH_MPS_FAST_MATH",
        "PYTORCH_MPS_HIGH_WATERMARK_RATIO",
        "PYTORCH_MPS_LOW_WATERMARK_RATIO",
        "PYTORCH_MPS_PREFER_METAL",
        "TAC_ADMISSION_ENFORCE",
    )
    try:
        constriction_version = importlib.metadata.version("constriction")
    except importlib.metadata.PackageNotFoundError:
        constriction_version = "absent"
    return {
        "python": sys.version,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "constriction": constriction_version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "mps_built": torch.backends.mps.is_built(),
        "mps_available": torch.backends.mps.is_available(),
        "environment": {name: os.environ.get(name) for name in relevant_env},
    }


def _canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_trainer_tool() -> Any:
    spec = importlib.util.spec_from_file_location("train_ddm_cl1_hpac_capacity_for_fit", TRAINER_PATH)
    if spec is None or spec.loader is None:
        raise CL1FitError(f"cannot load the pinned trainer: {TRAINER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _artifact_path(row: dict[str, Any], field: str, index: int) -> Path:
    value = row.get(field)
    if not isinstance(value, str) or not value:
        raise CL1FitError(f"row {index} lacks {field}")
    path = Path(value).expanduser().resolve()
    try:
        path.relative_to(MEASUREMENT_ROOT.resolve())
    except ValueError as exc:
        raise CL1FitError(f"row {index} {field} is outside the CL1 SSD custody root") from exc
    if not path.is_file():
        raise CL1FitError(f"row {index} artifact is absent: {path}")
    return path


def _tree_equal(left: Any, right: Any) -> bool:
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left, right)
    if isinstance(left, dict):
        return (
            isinstance(right, dict)
            and left.keys() == right.keys()
            and all(_tree_equal(left[key], right[key]) for key in left)
        )
    if isinstance(left, (list, tuple)):
        return (
            isinstance(right, type(left))
            and len(left) == len(right)
            and all(_tree_equal(a, b) for a, b in zip(left, right, strict=True))
        )
    if isinstance(left, np.ndarray):
        return isinstance(right, np.ndarray) and np.array_equal(left, right)
    return left == right


def _normalize_argv(argv: list[Any]) -> list[str]:
    if not all(isinstance(item, str) for item in argv):
        raise CL1FitError("safe-run receipt argv is malformed")
    output = list(argv)
    path_flags = {
        "--checkpoint",
        "--cache",
        "--init",
        "--save",
        "--out",
        "--resume-from",
        "--intake-code",
        "--blob",
        "--report",
        "--tokens-out",
        "--decode-from",
        "--raw-out",
        "--tokens",
        "--raw",
        "--attestation",
    }
    if output:
        # abspath, NOT resolve(): `.venv/bin/python` is a symlink to `python3.13`
        # (rebuilt 2026-08-31); following it breaks the `PYTHON_PATH` identity that
        # `_expected_training_argv` deliberately leaves unresolved (site-packages).
        output[0] = str(Path(os.path.abspath(Path(output[0]).expanduser())))
    if len(output) > 1:
        output[1] = str(Path(output[1]).expanduser().resolve())
    for position, item in enumerate(output[:-1]):
        if item in path_flags:
            output[position + 1] = str(Path(output[position + 1]).expanduser().resolve())
    return output


def _safe_run_receipt(
    path: Path,
    *,
    expected_argv: list[str],
    label: str,
    expected_exit: int = 0,
    require_observed_peak: bool = False,
) -> tuple[dict[str, Any], str]:
    payload = _read_json(path)
    if payload.get("schema") != "safe_run_status_receipt.v1":
        raise CL1FitError(f"{label} is not a safe-run receipt")
    if payload.get("status") != "ok" or payload.get("exit") != expected_exit:
        raise CL1FitError(f"{label} has the wrong terminal status/exit")
    argv = payload.get("argv")
    if not isinstance(argv, list) or _normalize_argv(argv) != expected_argv:
        raise CL1FitError(f"{label} argv differs from the pinned command")
    child_pid = payload.get("child_pid")
    pgid = payload.get("pgid")
    elapsed = payload.get("elapsed_s")
    rss_limit = payload.get("rss_limit_mib")
    timeout = payload.get("timeout_s")
    if (
        type(child_pid) is not int
        or child_pid < 1
        or pgid != child_pid
        or not isinstance(payload.get("start_utc"), str)
        or not isinstance(payload.get("generated_utc"), str)
        or isinstance(elapsed, bool)
        or not isinstance(elapsed, (int, float))
        or not math.isfinite(float(elapsed))
        or float(elapsed) < 0.0
        or type(rss_limit) is not int
        or rss_limit < 1
        or isinstance(timeout, bool)
        or not isinstance(timeout, (int, float))
        or float(timeout) <= 0.0
        or payload.get("kill_action") is not None
    ):
        raise CL1FitError(f"{label} lacks complete successful process custody")
    if require_observed_peak and (
        payload.get("peak_rss_observed") is not True
        or isinstance(payload.get("peak_rss_mib"), bool)
        or not isinstance(payload.get("peak_rss_mib"), (int, float))
        or float(payload["peak_rss_mib"]) <= 0.0
    ):
        raise CL1FitError(f"{label} lacks an observed positive RSS peak")
    return payload, _sha256_file(path)


def _common_codec_argv(checkpoint: Path) -> list[str]:
    return [
        str(PYTHON_PATH),  # no .resolve(): following the venv symlink strips site-packages (numpy gone)
        str((INTAKE_CODE_ROOT / "codec_hpac_integer.py").resolve()),
        "--checkpoint",
        str(checkpoint),
        "--cache",
        str(CANONICAL_CACHE_PATH.resolve()),
        "--channels",
        "64",
        "--patch",
        "64",
        "--delta",
        "2",
        "--frame-dim",
        "8",
        "--norm-mode",
        "none",
        "--activation",
        "relu",
        "--frame-scale",
        "--weight-bound",
        "127",
        "--activation-bound",
        "127",
        "--weight-scales",
        "--weight-exponent-min",
        "-6",
        "--spm",
        "--sparse",
        "--self-compress",
        "--target-mode",
        "raw",
        "--frames",
        "600",
        "--device",
        "mps",
    ]


def _expected_pack_argv(checkpoint: Path, blob: Path, report: Path) -> list[str]:
    return [
        str(PYTHON_PATH),  # no .resolve(): following the venv symlink strips site-packages (numpy gone)
        str((INTAKE_CODE_ROOT / "pack_hpac_self_compress.py").resolve()),
        "--checkpoint",
        str(checkpoint),
        "--channels",
        "64",
        "--patch",
        "64",
        "--delta",
        "2",
        "--frame-dim",
        "8",
        "--weight-bound",
        "127",
        "--activation-bound",
        "127",
        "--weight-exponent-min",
        "-6",
        "--device",
        "cpu",
        "--blob",
        str(blob),
        "--report",
        str(report),
    ]


def _expected_encode_argv(checkpoint: Path, tokens: Path, report: Path) -> list[str]:
    return [
        *_common_codec_argv(checkpoint),
        "--tokens-out",
        str(tokens),
        "--report",
        str(report),
    ]


def _expected_decode_argv(checkpoint: Path, tokens: Path, raw: Path, report: Path) -> list[str]:
    return [
        *_common_codec_argv(checkpoint),
        "--decode-from",
        str(tokens),
        "--raw-out",
        str(raw),
        "--require-exact",
        "--report",
        str(report),
    ]


def _expected_training_argv(
    *,
    rate_lambda: float,
    save: Path,
    out: Path,
    resume_from: Path | None = None,
) -> list[str]:
    argv = [
        str(PYTHON_PATH),  # no .resolve(): following the venv symlink strips site-packages (numpy gone)
        str(TRAINER_PATH.resolve()),
        "--cache",
        str(CANONICAL_CACHE_PATH.resolve()),
        "--init",
        str(CANONICAL_INIT_PATH.resolve()),
        "--epochs",
        "60",
        "--batch-size",
        "8",
        "--eval-batch-size",
        "4",
        "--eval-every",
        "2",
        "--lr",
        "0.003",
        "--lr-exponent",
        "0.0002",
        "--lr-bits",
        "0.01",
        "--bit-eps",
        "1e-6",
        "--rate-lambda",
        str(rate_lambda),
        "--qat-fraction",
        "0.5",
        "--init-bits",
        "8.0",
        "--channels",
        "64",
        "--patch",
        "64",
        "--delta",
        "2",
        "--frame-dim",
        "8",
        "--norm-mode",
        "none",
        "--activation",
        "relu",
        "--frame-scale",
        "--weight-bound",
        "127",
        "--activation-bound",
        "127",
        "--weight-scales",
        "--weight-exponent-min",
        "-6",
        "--spm",
        "--target-mode",
        "raw",
        "--seed",
        "20260716",
        "--ema-target-seed-fraction",
        "0.01",
        "--device",
        "mps",
        "--save",
        str(save.resolve()),
        "--out",
        str(out.resolve()),
    ]
    if resume_from is not None:
        argv.extend(["--resume-from", str(resume_from.resolve())])
    return argv


def _terminal_checkpoint_path(save: Path) -> Path:
    checkpoint_root = save.with_name(save.stem + ".checkpoints")
    return checkpoint_root / "qat_stage_end_epoch_0060.pt"


def _epoch_one_checkpoint_path(save: Path) -> Path:
    checkpoint_root = save.with_name(save.stem + ".checkpoints")
    return checkpoint_root / "periodic" / "epoch_0001.pt"


def _expected_runner_argv(
    operation: str,
    *,
    checkpoint: Path,
    report: Path,
    attestation: Path,
    blob: Path | None = None,
    tokens: Path | None = None,
    raw: Path | None = None,
) -> list[str]:
    output = [
        str(PYTHON_PATH),  # no .resolve(): following the venv symlink strips site-packages (numpy gone)
        str(Path(__file__).resolve()),
        operation,
        "--checkpoint",
        str(checkpoint.resolve()),
    ]
    if operation == "pack":
        if blob is None:
            raise CL1FitError("pack runner argv requires a blob path")
        output.extend(["--blob", str(blob.resolve())])
    elif operation == "encode":
        if tokens is None:
            raise CL1FitError("encode runner argv requires a token path")
        output.extend(["--tokens", str(tokens.resolve())])
    elif operation == "decode":
        if tokens is None or raw is None:
            raise CL1FitError("decode runner argv requires token and raw paths")
        output.extend(["--tokens", str(tokens.resolve()), "--raw", str(raw.resolve())])
    else:
        raise CL1FitError(f"unknown artifact operation: {operation}")
    output.extend(
        [
            "--report",
            str(report.resolve()),
            "--attestation",
            str(attestation.resolve()),
        ]
    )
    return output


def _atomic_bytes(path: Path, encoded: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with tmp.open("wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if tmp.exists():
            tmp.unlink()


def _atomic_json(path: Path, value: Any) -> None:
    _atomic_bytes(
        path,
        (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )


def _require_measurement_path(path: Path, *, label: str) -> Path:
    resolved = path.expanduser().resolve(strict=False)
    try:
        resolved.relative_to(MEASUREMENT_ROOT.resolve())
    except ValueError as exc:
        raise CL1FitError(f"{label} is outside the CL1 SSD custody root") from exc
    return resolved


def _validate_native_report(operation: str, report: dict[str, Any], output: Path) -> None:
    if operation == "pack":
        if (
            report.get("verified_exact") is not True
            or report.get("max_logit_diff") != 0.0
            or report.get("compressed_model_bytes") != output.stat().st_size
        ):
            raise CL1FitError("packer did not emit an exact size-matched model")
        return
    if operation == "encode":
        if report.get("frames") != 600 or report.get("token_bytes") != output.stat().st_size:
            raise CL1FitError("encoder did not emit a complete n600 Range stream")
        return
    if (
        report.get("frames") != 600
        or report.get("verified_exact") is not True
        or report.get("raw_token_sha256") != EXPECTED_RAW_TOKEN_SHA256
        or output.stat().st_size != PIXELS
        or _sha256_file(output) != EXPECTED_RAW_TOKEN_SHA256
    ):
        raise CL1FitError("decoder did not emit the exact canonical n600 token tensor")


def _run_artifact_operation(args: argparse.Namespace) -> dict[str, Any]:
    assert_governed_admission("fit_ddm_cl1_hpac_capacity_artifact_runner")
    operation = str(args.command)
    if os.environ.get("TAC_ADMISSION_ENFORCE") != "1":
        raise CL1FitError("set TAC_ADMISSION_ENFORCE=1 for a hard governor gate")
    if os.environ.get("PYTHONHASHSEED") != "0":
        raise CL1FitError("set PYTHONHASHSEED=0 before artifact execution")
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") != "0":
        raise CL1FitError("set PYTORCH_ENABLE_MPS_FALLBACK=0; CPU fallback is forbidden")
    if operation in {"encode", "decode"} and (
        platform.system() != "Darwin" or not torch.backends.mps.is_built() or not torch.backends.mps.is_available()
    ):
        raise CL1FitError("codec operation requires live local Metal")
    checkpoint = _require_measurement_path(args.checkpoint, label="checkpoint")
    if not checkpoint.is_file():
        raise CL1FitError(f"checkpoint is absent: {checkpoint}")
    report = _require_measurement_path(args.report, label="report")
    attestation = _require_measurement_path(args.attestation, label="attestation")
    if operation == "pack":
        output = _require_measurement_path(args.blob, label="packed model")
        child_argv = _expected_pack_argv(checkpoint, output, report)
        inputs = {"checkpoint": _file_record(checkpoint)}
        outputs = {"packed_model": output, "report": report}
    elif operation == "encode":
        _verify_canonical_cache()
        output = _require_measurement_path(args.tokens, label="Range stream")
        child_argv = _expected_encode_argv(checkpoint, output, report)
        inputs = {
            "checkpoint": _file_record(checkpoint),
            "cache": _file_record(CANONICAL_CACHE_PATH),
        }
        outputs = {"range_tokens": output, "report": report}
    elif operation == "decode":
        _verify_canonical_cache()
        token_path = _require_measurement_path(args.tokens, label="Range stream")
        output = _require_measurement_path(args.raw, label="decoded token tensor")
        child_argv = _expected_decode_argv(checkpoint, token_path, output, report)
        inputs = {
            "checkpoint": _file_record(checkpoint),
            "cache": _file_record(CANONICAL_CACHE_PATH),
            "range_tokens": _file_record(token_path),
        }
        outputs = {"decoded_raw_tokens": output, "report": report}
    else:
        raise CL1FitError(f"unknown artifact operation: {operation}")

    output_paths = [*outputs.values(), attestation]
    if len(set(output_paths)) != len(output_paths):
        raise CL1FitError("artifact output paths collide")
    occupied = [str(path) for path in output_paths if path.exists()]
    if occupied:
        raise CL1FitError("artifact outputs already exist; preserve or choose fresh paths: " + json.dumps(occupied))
    for path in output_paths:
        path.parent.mkdir(parents=True, exist_ok=True)

    causal_sources = _verified_causal_sources(operation)
    completed = subprocess.run(child_argv, check=False)
    if completed.returncode != 0:
        raise CL1FitError(f"{operation} child exited nonzero: {completed.returncode}")
    if not output.is_file() or not report.is_file():
        raise CL1FitError(f"{operation} child omitted an output or report")
    native_report = _read_json(report)
    _validate_native_report(operation, native_report, output)
    output_records = {name: _file_record(path) for name, path in outputs.items()}
    payload = {
        "schema": ATTESTATION_SCHEMA,
        "operation": operation,
        "runner_sha256": _sha256_file(Path(__file__).resolve()),
        "runtime_identity": _artifact_runtime_identity(),
        "causal_source_sha256": causal_sources,
        "child_argv": child_argv,
        "inputs": inputs,
        "outputs": output_records,
    }
    _atomic_json(attestation, payload)
    return payload


def _verify_attestation(
    path: Path,
    *,
    operation: str,
    child_argv: list[str],
    inputs: dict[str, Path],
    outputs: dict[str, Path],
) -> tuple[dict[str, Any], str]:
    payload = _read_json(path)
    if payload.get("schema") != ATTESTATION_SCHEMA or payload.get("operation") != operation:
        raise CL1FitError(f"{operation} attestation has the wrong identity")
    if payload.get("runner_sha256") != _sha256_file(Path(__file__).resolve()):
        raise CL1FitError(f"{operation} attestation was made by different runner bytes")
    runtime_identity = payload.get("runtime_identity")
    if runtime_identity != _artifact_runtime_identity():
        raise CL1FitError(f"{operation} attestation runtime identity changed")
    runtime_environment = runtime_identity.get("environment") if isinstance(runtime_identity, dict) else None
    if not isinstance(runtime_environment, dict) or (
        runtime_environment.get("TAC_ADMISSION_ENFORCE") != "1"
        or runtime_environment.get("PYTHONHASHSEED") != "0"
        or runtime_environment.get("PYTORCH_ENABLE_MPS_FALLBACK") != "0"
    ):
        raise CL1FitError(f"{operation} attestation lacks hard deterministic admission")
    if payload.get("causal_source_sha256") != _verified_causal_sources(operation):
        raise CL1FitError(f"{operation} attestation changes causal source bytes")
    if payload.get("child_argv") != child_argv:
        raise CL1FitError(f"{operation} attestation child argv differs")
    expected_inputs = {name: _file_record(value) for name, value in inputs.items()}
    expected_outputs = {name: _file_record(value) for name, value in outputs.items()}
    if payload.get("inputs") != expected_inputs:
        raise CL1FitError(f"{operation} attestation input bytes changed")
    if payload.get("outputs") != expected_outputs:
        raise CL1FitError(f"{operation} attestation output bytes changed")
    return payload, _sha256_file(path)


def _require_sha256(row: dict[str, Any], field: str, index: int) -> str:
    value = row.get(field)
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise CL1FitError(f"row {index} has malformed {field}")
    return value


def _fit_line(x_values: list[float], y_values: list[float]) -> dict[str, Any]:
    if len(x_values) != 3 or len(y_values) != 3:
        raise CL1FitError("the preregistered regression requires exactly three unique lambdas")
    x = np.asarray(x_values, dtype=np.float64)
    y = np.asarray(y_values, dtype=np.float64)
    centered = x - x.mean()
    sxx = float(np.dot(centered, centered))
    if not math.isfinite(sxx) or sxx <= 0.0:
        raise CL1FitError("model-byte coordinates have zero or invalid variance")
    slope = float(np.dot(centered, y - y.mean()) / sxx)
    intercept = float(y.mean() - slope * x.mean())
    residual = y - (intercept + slope * x)
    residual_ss = float(np.dot(residual, residual))
    degrees_freedom = 1
    residual_standard_error = math.sqrt(residual_ss / degrees_freedom)
    slope_standard_error = residual_standard_error / math.sqrt(sxx)
    critical = 12.706  # Student-t 97.5th percentile, df=1.
    ci95 = [
        slope - critical * slope_standard_error,
        slope + critical * slope_standard_error,
    ]
    if ci95[1] < -1.0:
        break_even_status = "PAYS"
    elif ci95[0] >= -1.0:
        break_even_status = "DOES_NOT_PAY"
    else:
        break_even_status = "INCONCLUSIVE"
    return {
        "n_unique_lambdas": 3,
        "degrees_freedom": degrees_freedom,
        "slope_token_bytes_per_model_byte": slope,
        "intercept_token_bytes": intercept,
        "residual_standard_error_token_bytes": residual_standard_error,
        "slope_standard_error": slope_standard_error,
        "ci95": ci95,
        "ci95_break_even_status": break_even_status,
        "uncertainty_method": (
            "conditional descriptive OLS across the three unique lambda rows, "
            "with a two-sided Student-t 95% interval at df=1; this reflects "
            "linearity residual/selection effects, not replicated training variance"
        ),
    }


def _validate_run_identity(
    trainer: Any,
    checkpoint: dict[str, Any],
    *,
    rate_lambda: float,
    index: int,
) -> dict[str, Any]:
    identity = checkpoint.get("run_identity")
    if not isinstance(identity, dict):
        raise CL1FitError(f"row {index} checkpoint lacks run identity")
    expected_config = dict(trainer.PREREGISTERED_CONFIG)
    expected_config["rate_lambda"] = rate_lambda
    if identity.get("training_config") != expected_config:
        raise CL1FitError(f"row {index} changes the pinned training config")
    if (
        identity.get("cache_sha256") != EXPECTED_CACHE_SHA256
        or Path(str(identity.get("cache_path", ""))).resolve() != CANONICAL_CACHE_PATH.resolve()
    ):
        raise CL1FitError(f"row {index} changes the pinned DALI cache")
    if (
        identity.get("init_sha256") != EXPECTED_INIT_SHA256
        or Path(str(identity.get("init_path", ""))).resolve() != CANONICAL_INIT_PATH.resolve()
    ):
        raise CL1FitError(f"row {index} changes the pinned initialization")
    trainer_sha256 = _sha256_file(TRAINER_PATH)
    local_causal = trainer._local_causal_sha256()
    if identity.get("trainer_sha256") != trainer_sha256:
        raise CL1FitError(f"row {index} trainer bytes differ from this fitter")
    if identity.get("intake_source_sha256") != trainer.EXPECTED_INTAKE_SHA256:
        raise CL1FitError(f"row {index} changes pinned intake source bytes")
    if identity.get("local_causal_source_sha256") != local_causal:
        raise CL1FitError(f"row {index} changes local causal source bytes")
    ema_policy = identity.get("ema_policy")
    if not isinstance(ema_policy, dict) or checkpoint.get("ema_policy") != ema_policy:
        raise CL1FitError(f"row {index} EMA policy identity is absent or changed")
    schedule_config = {key: value for key, value in expected_config.items() if key != "rate_lambda"}
    expected_schedule_sha256 = _canonical_json_sha256({"schedule_config": schedule_config, "ema_policy": ema_policy})
    source_identity = {
        "trainer_sha256": trainer_sha256,
        "intake_source_sha256": trainer.EXPECTED_INTAKE_SHA256,
        "local_causal_source_sha256": local_causal,
    }
    expected_source_sha256 = _canonical_json_sha256(source_identity)
    if identity.get("seed_schedule_identity_sha256") != expected_schedule_sha256:
        raise CL1FitError(f"row {index} seed/schedule identity does not verify")
    if identity.get("trainer_source_identity_sha256") != expected_source_sha256:
        raise CL1FitError(f"row {index} trainer/source identity does not verify")
    hardware = identity.get("hardware")
    if not isinstance(hardware, dict) or not (
        hardware.get("system") == "Darwin"
        and hardware.get("mps_built") is True
        and hardware.get("mps_available") is True
    ):
        raise CL1FitError(f"row {index} is not a live macOS-MPS training identity")
    launch_git_sha = identity.get("launch_git_sha")
    if (
        not isinstance(launch_git_sha, str)
        or len(launch_git_sha) != 40
        or any(character not in "0123456789abcdef" for character in launch_git_sha)
    ):
        raise CL1FitError(f"row {index} launch git identity is malformed")
    if checkpoint.get("run_identity_sha256") != _canonical_json_sha256(identity):
        raise CL1FitError(f"row {index} run identity hash does not verify")
    return identity


def _comparison_identity_sha256(identity: dict[str, Any]) -> str:
    comparable = json.loads(json.dumps(identity, sort_keys=True))
    training_config = comparable.get("training_config")
    if not isinstance(training_config, dict) or "rate_lambda" not in training_config:
        raise CL1FitError("run identity lacks the lambda comparison coordinate")
    training_config.pop("rate_lambda")
    return _canonical_json_sha256(comparable)


def _parse_training_receipt(
    trainer: Any,
    path: Path,
    *,
    rate_lambda: float,
    index: int,
    expected_exit: int,
    require_observed_peak: bool,
) -> tuple[argparse.Namespace, dict[str, Any], str]:
    payload = _read_json(path)
    raw_argv = payload.get("argv")
    if not isinstance(raw_argv, list):
        raise CL1FitError(f"row {index} training receipt argv is malformed")
    argv = _normalize_argv(raw_argv)
    if len(argv) < 2 or argv[:2] != [
        str(PYTHON_PATH),  # no .resolve(): following the venv symlink strips site-packages (numpy gone)
        str(TRAINER_PATH.resolve()),
    ]:
        raise CL1FitError(f"row {index} training receipt names the wrong trainer")
    try:
        args = trainer._build_parser().parse_args(argv[2:])
        trainer._assert_preregistered_config(args)
    except (SystemExit, Exception) as exc:
        if isinstance(exc, CL1FitError):
            raise
        raise CL1FitError(f"row {index} training argv does not parse exactly") from exc
    save = _require_measurement_path(args.save, label=f"row {index} training save")
    out = _require_measurement_path(args.out, label=f"row {index} training out")
    resume_from = None
    if args.resume_from is not None:
        resume_from = _require_measurement_path(args.resume_from, label=f"row {index} resume parent")
    expected_argv = _expected_training_argv(
        rate_lambda=rate_lambda,
        save=save,
        out=out,
        resume_from=resume_from,
    )
    receipt, receipt_sha256 = _safe_run_receipt(
        path,
        expected_argv=expected_argv,
        label=f"row {index} training receipt",
        expected_exit=expected_exit,
        require_observed_peak=require_observed_peak,
    )
    args.save = save
    args.out = out
    args.resume_from = resume_from
    return args, receipt, receipt_sha256


def _validate_training_custody(
    trainer: Any,
    row: dict[str, Any],
    *,
    rung_id: str,
    rate_lambda: float,
    index: int,
    selected_checkpoint: Path,
    checkpoint: dict[str, Any],
) -> tuple[dict[str, Any], list[Path]]:
    raw_paths = row.get("training_receipt_paths")
    expected_count = 2 if rung_id == "lambda_1p0_resume_control" else 1
    if not isinstance(raw_paths, list) or len(raw_paths) != expected_count:
        raise CL1FitError(f"row {index} requires {expected_count} ordered training receipt(s)")
    receipt_paths: list[Path] = []
    for value in raw_paths:
        if not isinstance(value, str) or not value:
            raise CL1FitError(f"row {index} training receipt path is malformed")
        path = _require_measurement_path(Path(value), label=f"row {index} training receipt")
        if not path.is_file():
            raise CL1FitError(f"row {index} training receipt is absent: {path}")
        receipt_paths.append(path)
    if len(set(receipt_paths)) != len(receipt_paths):
        raise CL1FitError(f"row {index} repeats a training receipt")

    parsed: list[argparse.Namespace] = []
    receipt_hashes: list[str] = []
    for position, path in enumerate(receipt_paths):
        interrupted = rung_id == "lambda_1p0_resume_control" and position == 0
        args, _, digest = _parse_training_receipt(
            trainer,
            path,
            rate_lambda=rate_lambda,
            index=index,
            expected_exit=-9 if interrupted else 0,
            require_observed_peak=interrupted,
        )
        parsed.append(args)
        receipt_hashes.append(digest)

    final_args = parsed[-1]
    if selected_checkpoint != _terminal_checkpoint_path(final_args.save).resolve():
        raise CL1FitError(f"row {index} did not select its terminal epoch-60 QAT checkpoint")
    if rung_id == "lambda_1p0_resume_control":
        interrupted_args, resumed_args = parsed
        if interrupted_args.resume_from is not None:
            raise CL1FitError("resume control's interrupted launch is not fresh")
        expected_parent = _epoch_one_checkpoint_path(interrupted_args.save).resolve()
        if resumed_args.resume_from != expected_parent:
            raise CL1FitError("resume control did not resume the interrupted epoch-1 parent")
        if interrupted_args.save == resumed_args.save or interrupted_args.out == resumed_args.out:
            raise CL1FitError("resume control must continue into a fresh output root")
        if not expected_parent.is_file():
            raise CL1FitError("resume control epoch-1 parent is absent")
        parent_checkpoint = torch.load(expected_parent, map_location="cpu", weights_only=False)
        if (
            not isinstance(parent_checkpoint, dict)
            or parent_checkpoint.get("schema") != trainer.CHECKPOINT_SCHEMA
            or parent_checkpoint.get("epoch") != 1
            or parent_checkpoint.get("phase") != "continuous"
            or parent_checkpoint.get("run_identity") != checkpoint.get("run_identity")
            or parent_checkpoint.get("causal_state_sha256") != trainer._causal_state_sha256(parent_checkpoint)
        ):
            raise CL1FitError("resume control epoch-1 parent state does not verify")
        parent_sha256 = _sha256_file(expected_parent)
        parent_bytes = expected_parent.stat().st_size
        lineage = checkpoint.get("resume_lineage")
        if not isinstance(lineage, list):
            raise CL1FitError("resume control checkpoint has no embedded lineage")
        matching = [
            entry
            for entry in lineage
            if isinstance(entry, dict)
            and Path(str(entry.get("source_path", ""))).resolve() == expected_parent
            and entry.get("sha256") == parent_sha256
            and entry.get("bytes") == parent_bytes
        ]
        if len(matching) != 1:
            raise CL1FitError("resume control lineage does not bind the epoch-1 parent")
        preserved = Path(str(matching[0].get("preserved_path", ""))).resolve()
        if (
            not preserved.is_file()
            or preserved.stat().st_size != parent_bytes
            or _sha256_file(preserved) != parent_sha256
        ):
            raise CL1FitError("resume control preserved parent bytes do not verify")
        parent_record: dict[str, Any] | None = _file_record(expected_parent)
        preserved_parent_record: dict[str, Any] | None = _file_record(preserved)
    else:
        if final_args.resume_from is not None or checkpoint.get("resume_lineage") != []:
            raise CL1FitError(f"row {index} non-resume rung is not a fresh run")
        parent_record = None
        preserved_parent_record = None

    final_path = final_args.save
    result_path = final_args.out
    manifest_path = final_path.with_suffix(".artifacts.json")
    for label, path in (
        ("surrogate-best file", final_path),
        ("trainer result", result_path),
        ("trainer manifest", manifest_path),
    ):
        if not path.is_file():
            raise CL1FitError(f"row {index} {label} is absent: {path}")
    manifest = _read_json(manifest_path)
    if (
        manifest.get("schema") != trainer.MANIFEST_SCHEMA
        or manifest.get("score_claim") is not False
        or manifest.get("run_identity") != checkpoint.get("run_identity")
        or not isinstance(manifest.get("argv"), list)
        or _normalize_argv(manifest["argv"]) != _normalize_argv(_read_json(receipt_paths[-1])["argv"])
    ):
        raise CL1FitError(f"row {index} trainer success manifest does not verify")
    selected_record = _file_record(selected_checkpoint)
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or not any(
        isinstance(entry, dict)
        and entry.get("path") == selected_record["path"]
        and entry.get("bytes") == selected_record["bytes"]
        and entry.get("sha256") == selected_record["sha256"]
        for entry in artifacts
    ):
        raise CL1FitError(f"row {index} manifest omits the selected terminal checkpoint")
    result_payload = _read_json(result_path)
    if (
        result_payload.get("schema") != "ddm_cl1_hpac_capacity_trainer_result.v1"
        or result_payload.get("run_identity") != checkpoint.get("run_identity")
        or result_payload.get("score_claim") is not False
    ):
        raise CL1FitError(f"row {index} trainer result does not verify")

    custody = {
        "training_receipt_paths": [str(path) for path in receipt_paths],
        "training_receipt_sha256": receipt_hashes,
        "trainer_final_path": str(final_path),
        "trainer_final_sha256": _sha256_file(final_path),
        "trainer_result_path": str(result_path),
        "trainer_result_sha256": _sha256_file(result_path),
        "trainer_manifest_path": str(manifest_path),
        "trainer_manifest_sha256": _sha256_file(manifest_path),
        "resume_parent": parent_record,
        "preserved_resume_parent": preserved_parent_record,
    }
    derived_paths = [*receipt_paths, final_path, result_path, manifest_path]
    if parent_record is not None and preserved_parent_record is not None:
        derived_paths.extend([expected_parent, preserved])
    return custody, derived_paths


def _validated_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    if payload.get("schema") != INPUT_SCHEMA:
        raise CL1FitError(f"expected input schema {INPUT_SCHEMA}")
    raw_rows = payload.get("rows")
    if not isinstance(raw_rows, list) or len(raw_rows) not in (3, 4):
        raise CL1FitError(
            "measurement set must contain the two controls plus lambda 0.5, optionally followed by lambda 0.25"
        )
    trainer = _load_trainer_tool()
    _verify_canonical_cache()
    _verified_causal_sources("pack")
    _verified_causal_sources("encode")
    rows: list[dict[str, Any]] = []
    by_rung: dict[str, dict[str, Any]] = {}
    common_schedule: set[str] = set()
    common_source: set[str] = set()
    common_comparison_identity: set[str] = set()
    seen_paths: set[Path] = set()
    seen_training_receipt_hashes: set[str] = set()
    for index, raw_value in enumerate(raw_rows):
        if not isinstance(raw_value, dict):
            raise CL1FitError(f"row {index} is not an object")
        raw = cast("dict[str, Any]", raw_value)
        rung_value = raw.get("rung_id")
        if not isinstance(rung_value, str) or (rung_value not in RUNG_LAMBDAS or rung_value in by_rung):
            raise CL1FitError(f"row {index} has unknown or duplicate rung_id")
        rung_id = rung_value
        rate_lambda = RUNG_LAMBDAS[rung_id]
        paths = {field: _artifact_path(raw, field, index) for field in ARTIFACT_PATH_FIELDS}
        duplicates = seen_paths.intersection(paths.values())
        if duplicates:
            raise CL1FitError(f"row {index} reuses an artifact path")
        seen_paths.update(paths.values())

        checkpoint = torch.load(
            paths["selected_checkpoint_path"],
            map_location="cpu",
            weights_only=False,
        )
        if not isinstance(checkpoint, dict) or checkpoint.get("schema") != (trainer.CHECKPOINT_SCHEMA):
            raise CL1FitError(f"row {index} selected artifact is not a CL1 checkpoint")
        required_checkpoint_fields = {
            "state_dict",
            "live_state_dict",
            "ema",
            "optimizer_state_dict",
            "scheduler_state_dict",
            "rng",
            "best",
            "history",
            "resume_lineage",
            "causal_state_sha256",
        }
        if not required_checkpoint_fields.issubset(checkpoint):
            raise CL1FitError(f"row {index} checkpoint lacks complete resume state")
        epoch = checkpoint.get("epoch")
        if (
            type(epoch) is not int
            or epoch != 60
            or checkpoint.get("phase") != "discrete_qat"
            or checkpoint.get("qat_start") != 31
        ):
            raise CL1FitError(f"row {index} is not the terminal epoch-60 discrete-QAT candidate")
        ema_state = checkpoint.get("ema")
        if (
            checkpoint.get("deployment_weights") != "ema_shadow"
            or not isinstance(ema_state, dict)
            or not _tree_equal(checkpoint["state_dict"], ema_state.get("shadow"))
        ):
            raise CL1FitError(f"row {index} does not deploy its EMA shadow")
        causal_state_sha256 = checkpoint.get("causal_state_sha256")
        if causal_state_sha256 != trainer._causal_state_sha256(checkpoint):
            raise CL1FitError(f"row {index} causal state hash does not verify")
        identity = _validate_run_identity(trainer, checkpoint, rate_lambda=rate_lambda, index=index)
        comparison_identity_sha256 = _comparison_identity_sha256(identity)
        training_custody, training_paths = _validate_training_custody(
            trainer,
            raw,
            rung_id=rung_id,
            rate_lambda=rate_lambda,
            index=index,
            selected_checkpoint=paths["selected_checkpoint_path"],
            checkpoint=checkpoint,
        )
        training_duplicates = seen_paths.intersection(training_paths)
        if training_duplicates:
            raise CL1FitError(f"row {index} reuses training custody paths")
        seen_paths.update(training_paths)
        training_hashes = training_custody["training_receipt_sha256"]
        if not isinstance(training_hashes, list) or any(
            digest in seen_training_receipt_hashes for digest in training_hashes
        ):
            raise CL1FitError(f"row {index} reuses training receipt bytes")
        seen_training_receipt_hashes.update(training_hashes)

        pack_report = _read_json(paths["pack_report_path"])
        model_bytes = paths["packed_model_path"].stat().st_size
        if type(pack_report.get("compressed_model_bytes")) is not int or (
            pack_report["compressed_model_bytes"] != model_bytes
        ):
            raise CL1FitError(f"row {index} packed-model bytes do not verify")
        if type(pack_report.get("raw_model_bytes")) is not int or (pack_report["raw_model_bytes"] < 1):
            raise CL1FitError(f"row {index} packed raw-model bytes are malformed")
        if pack_report.get("verified_exact") is not True or (pack_report.get("max_logit_diff") != 0.0):
            raise CL1FitError(f"row {index} pack logit round-trip is not bit exact")

        encode_report = _read_json(paths["encode_report_path"])
        range_bytes = paths["range_token_path"].stat().st_size
        if (
            encode_report.get("frames") != 600
            or type(encode_report.get("token_bytes")) is not int
            or encode_report["token_bytes"] != range_bytes
        ):
            raise CL1FitError(f"row {index} Range stream bytes do not verify")
        ideal_bpp = encode_report.get("ideal_bpp")
        token_bpp = encode_report.get("token_bpp")
        if isinstance(ideal_bpp, bool) or not isinstance(ideal_bpp, (int, float)):
            raise CL1FitError(f"row {index} encode ideal_bpp is malformed")
        ideal_bpp = float(ideal_bpp)
        if not math.isfinite(ideal_bpp) or ideal_bpp <= 0.0:
            raise CL1FitError(f"row {index} encode ideal_bpp is malformed")
        ideal_bytes = math.ceil(ideal_bpp * PIXELS / 8.0)
        expected_token_bpp = range_bytes * 8.0 / PIXELS
        if not isinstance(token_bpp, (int, float)) or not math.isclose(
            float(token_bpp), expected_token_bpp, rel_tol=0.0, abs_tol=1e-15
        ):
            raise CL1FitError(f"row {index} Range token_bpp does not verify")
        if range_bytes < ideal_bytes:
            raise CL1FitError(f"row {index} claims a Range stream smaller than its own ideal bytes")
        encode_logit_hash = _require_sha256(encode_report, "logit_hash_encode", index)

        decode_report = _read_json(paths["decode_report_path"])
        if decode_report.get("frames") != 600 or (decode_report.get("verified_exact") is not True):
            raise CL1FitError(f"row {index} lacks an exact n600 decode")
        decode_logit_hash = _require_sha256(decode_report, "logit_hash_decode", index)
        if decode_logit_hash != encode_logit_hash:
            raise CL1FitError(f"row {index} encode/decode logit hashes differ")
        if decode_report.get("raw_token_sha256") != EXPECTED_RAW_TOKEN_SHA256:
            raise CL1FitError(f"row {index} decoded the wrong raw token tensor")
        if paths["decoded_raw_token_path"].stat().st_size != PIXELS or (
            _sha256_file(paths["decoded_raw_token_path"]) != EXPECTED_RAW_TOKEN_SHA256
        ):
            raise CL1FitError(f"row {index} decoded raw artifact does not verify")

        _, pack_attestation_sha256 = _verify_attestation(
            paths["pack_attestation_path"],
            operation="pack",
            child_argv=_expected_pack_argv(
                paths["selected_checkpoint_path"],
                paths["packed_model_path"],
                paths["pack_report_path"],
            ),
            inputs={"checkpoint": paths["selected_checkpoint_path"]},
            outputs={
                "packed_model": paths["packed_model_path"],
                "report": paths["pack_report_path"],
            },
        )
        _, encode_attestation_sha256 = _verify_attestation(
            paths["encode_attestation_path"],
            operation="encode",
            child_argv=_expected_encode_argv(
                paths["selected_checkpoint_path"],
                paths["range_token_path"],
                paths["encode_report_path"],
            ),
            inputs={
                "checkpoint": paths["selected_checkpoint_path"],
                "cache": CANONICAL_CACHE_PATH,
            },
            outputs={
                "range_tokens": paths["range_token_path"],
                "report": paths["encode_report_path"],
            },
        )
        _, decode_attestation_sha256 = _verify_attestation(
            paths["decode_attestation_path"],
            operation="decode",
            child_argv=_expected_decode_argv(
                paths["selected_checkpoint_path"],
                paths["range_token_path"],
                paths["decoded_raw_token_path"],
                paths["decode_report_path"],
            ),
            inputs={
                "checkpoint": paths["selected_checkpoint_path"],
                "cache": CANONICAL_CACHE_PATH,
                "range_tokens": paths["range_token_path"],
            },
            outputs={
                "decoded_raw_tokens": paths["decoded_raw_token_path"],
                "report": paths["decode_report_path"],
            },
        )
        _, pack_receipt_sha256 = _safe_run_receipt(
            paths["pack_receipt_path"],
            expected_argv=_expected_runner_argv(
                "pack",
                checkpoint=paths["selected_checkpoint_path"],
                blob=paths["packed_model_path"],
                report=paths["pack_report_path"],
                attestation=paths["pack_attestation_path"],
            ),
            label=f"row {index} pack receipt",
        )
        _, encode_receipt_sha256 = _safe_run_receipt(
            paths["encode_receipt_path"],
            expected_argv=_expected_runner_argv(
                "encode",
                checkpoint=paths["selected_checkpoint_path"],
                tokens=paths["range_token_path"],
                report=paths["encode_report_path"],
                attestation=paths["encode_attestation_path"],
            ),
            label=f"row {index} encode receipt",
        )
        decode_receipt, decode_receipt_sha256 = _safe_run_receipt(
            paths["decode_receipt_path"],
            expected_argv=_expected_runner_argv(
                "decode",
                checkpoint=paths["selected_checkpoint_path"],
                tokens=paths["range_token_path"],
                raw=paths["decoded_raw_token_path"],
                report=paths["decode_report_path"],
                attestation=paths["decode_attestation_path"],
            ),
            label=f"row {index} decode receipt",
        )
        decode_seconds = decode_receipt.get("elapsed_s")
        if (
            isinstance(decode_seconds, bool)
            or not isinstance(decode_seconds, (int, float))
            or not math.isfinite(float(decode_seconds))
            or not (0.0 <= float(decode_seconds) <= 1800.0)
        ):
            raise CL1FitError(f"row {index} exceeds the 30-minute decode bound")

        row = {
            "rung_id": rung_id,
            "rate_lambda": rate_lambda,
            "channels": 64,
            "patch": 64,
            "delta": 2,
            "frame_dim": 8,
            "selected_epoch": epoch,
            "selected_checkpoint_path": str(paths["selected_checkpoint_path"]),
            "selected_checkpoint_sha256": _sha256_file(paths["selected_checkpoint_path"]),
            "causal_state_sha256": causal_state_sha256,
            "packed_model_path": str(paths["packed_model_path"]),
            "packed_model_bytes": model_bytes,
            "packed_model_sha256": _sha256_file(paths["packed_model_path"]),
            "ideal_bpp": ideal_bpp,
            "ideal_token_bytes": ideal_bytes,
            "range_token_path": str(paths["range_token_path"]),
            "range_token_bytes": range_bytes,
            "range_token_sha256": _sha256_file(paths["range_token_path"]),
            "decoded_raw_token_path": str(paths["decoded_raw_token_path"]),
            "decoded_raw_token_bytes": paths["decoded_raw_token_path"].stat().st_size,
            "pack_verified_exact": True,
            "pack_max_logit_diff": 0.0,
            "decode_verified_exact": True,
            "same_cache_verified": True,
            "frames": 600,
            "decode_seconds": float(decode_seconds),
            "token_coder": TOKEN_CODER,
            "cache_sha256": EXPECTED_CACHE_SHA256,
            "encoded_raw_token_sha256": EXPECTED_RAW_TOKEN_SHA256,
            "decoded_raw_token_sha256": EXPECTED_RAW_TOKEN_SHA256,
            "pack_report_path": str(paths["pack_report_path"]),
            "pack_report_sha256": _sha256_file(paths["pack_report_path"]),
            "encode_report_path": str(paths["encode_report_path"]),
            "encode_report_sha256": _sha256_file(paths["encode_report_path"]),
            "decode_report_path": str(paths["decode_report_path"]),
            "decode_report_sha256": _sha256_file(paths["decode_report_path"]),
            "pack_attestation_path": str(paths["pack_attestation_path"]),
            "pack_attestation_sha256": pack_attestation_sha256,
            "encode_attestation_path": str(paths["encode_attestation_path"]),
            "encode_attestation_sha256": encode_attestation_sha256,
            "decode_attestation_path": str(paths["decode_attestation_path"]),
            "decode_attestation_sha256": decode_attestation_sha256,
            "pack_receipt_path": str(paths["pack_receipt_path"]),
            "pack_receipt_sha256": pack_receipt_sha256,
            "encode_receipt_path": str(paths["encode_receipt_path"]),
            "encode_receipt_sha256": encode_receipt_sha256,
            "decode_receipt_path": str(paths["decode_receipt_path"]),
            "decode_receipt_sha256": decode_receipt_sha256,
            "seed_schedule_identity_sha256": identity["seed_schedule_identity_sha256"],
            "trainer_source_identity_sha256": identity["trainer_source_identity_sha256"],
            "comparison_identity_sha256": comparison_identity_sha256,
            **training_custody,
        }
        rows.append(row)
        by_rung[rung_id] = row
        common_schedule.add(row["seed_schedule_identity_sha256"])
        common_source.add(row["trainer_source_identity_sha256"])
        common_comparison_identity.add(row["comparison_identity_sha256"])
    required = {
        "lambda_1p0_resume_control",
        "lambda_1p0_uninterrupted_twin",
        "lambda_0p5",
    }
    observed = set(by_rung)
    if not required.issubset(observed):
        raise CL1FitError("both named controls and lambda 0.5 are required")
    if len(rows) == 4 and observed != set(RUNG_LAMBDAS):
        raise CL1FitError("the fourth row must be the preregistered lambda 0.25")
    if len(rows) == 3 and observed != required:
        raise CL1FitError("lambda 0.25 cannot replace an earlier rung")
    if len(common_schedule) != 1:
        raise CL1FitError("rows do not share one pinned seed/schedule identity")
    if len(common_source) != 1:
        raise CL1FitError("rows do not share one trainer/source identity")
    if len(common_comparison_identity) != 1:
        raise CL1FitError("rows change launch/software/hardware/input identity beyond rate lambda")
    for receipt_field in RECEIPT_HASH_FIELDS:
        identities = [row[receipt_field] for row in rows]
        if len(identities) != len(set(identities)):
            raise CL1FitError(f"rows reuse a {receipt_field} identity")
    resume_control = by_rung["lambda_1p0_resume_control"]
    uninterrupted = by_rung["lambda_1p0_uninterrupted_twin"]
    differences = {
        field: [resume_control[field], uninterrupted[field]]
        for field in CONTROL_EQUAL_FIELDS
        if resume_control[field] != uninterrupted[field]
    }
    if differences:
        raise CL1FitError(
            "lambda-1 controls diverge; MPS resume equivalence is not proved: "
            + json.dumps(differences, sort_keys=True)
        )
    return rows


def _aggregate_by_lambda(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[float, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["rate_lambda"]].append(row)
    result: list[dict[str, Any]] = []
    for rate_lambda in RUNG_ORDER:
        members = grouped.get(rate_lambda)
        if not members:
            continue
        model = float(np.mean([row["packed_model_bytes"] for row in members]))
        ideal = float(np.mean([row["ideal_token_bytes"] for row in members]))
        real = float(np.mean([row["range_token_bytes"] for row in members]))
        result.append(
            {
                "rate_lambda": rate_lambda,
                "replicates": len(members),
                "mean_packed_model_bytes": model,
                "mean_ideal_token_bytes": ideal,
                "mean_range_token_bytes": real,
                "mean_ideal_joint_bytes": model + ideal,
                "mean_range_joint_bytes": model + real,
                "delta_packed_model_bytes_vs_pr130": (model - PR130_PACKED_MODEL_BYTES),
                "delta_ideal_token_bytes_vs_pr130": (ideal - PR130_IDEAL_TOKEN_BYTES),
                "delta_range_token_bytes_vs_pr130": (real - PR130_RANGE_TOKEN_BYTES),
                "delta_ideal_joint_bytes_vs_pr130": (model + ideal - PR130_IDEAL_JOINT_BYTES),
                "delta_range_joint_bytes_vs_pr130": (model + real - PR130_RANGE_JOINT_BYTES),
            }
        )
    return result


def _adjacent_slopes(aggregates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for left, right in pairwise(aggregates):
        delta_model = right["mean_packed_model_bytes"] - left["mean_packed_model_bytes"]
        if delta_model <= 0.0:
            ideal_slope = None
            range_slope = None
            status = "NON_MONOTONE_MODEL_CAPACITY"
        else:
            ideal_slope = (right["mean_ideal_token_bytes"] - left["mean_ideal_token_bytes"]) / delta_model
            range_slope = (right["mean_range_token_bytes"] - left["mean_range_token_bytes"]) / delta_model
            status = "PAYS" if range_slope < -1.0 else "DOES_NOT_PAY"
        output.append(
            {
                "left_lambda": left["rate_lambda"],
                "right_lambda": right["rate_lambda"],
                "delta_model_bytes": delta_model,
                "ideal_token_bytes_per_model_byte": ideal_slope,
                "range_token_bytes_per_model_byte": range_slope,
                "range_break_even_status": status,
                "real_break_even_pass": range_slope is not None and range_slope < -1.0,
            }
        )
    return output


def _knee_status(adjacent: list[dict[str, Any]], *, final: bool) -> dict[str, Any]:
    if any(row["range_token_bytes_per_model_byte"] is None for row in adjacent):
        return {
            "status": "NON_MONOTONE_MODEL_CAPACITY",
            "knee_lambda": None,
            "best_observed_lambda": None,
            "knee_bracket_lambda": None,
            "interpretation": (
                "lower lambda did not strictly grow the serialized prior; the "
                "registered capacity coordinate was not realized"
            ),
        }
    first_pays = adjacent[0]["real_break_even_pass"]
    if not first_pays:
        return {
            "status": "REFERENCE_BOUNDARY",
            "knee_lambda": None,
            "best_observed_lambda": 1.0,
            "knee_bracket_lambda": [0.5, 1.0],
            "interpretation": (
                "the first prior-growth interval does not pay; lambda 1.0 is "
                "the best observed boundary, not a measured continuous optimum"
            ),
        }
    if not final:
        return {
            "status": "UNRESOLVED_FIRE_LAMBDA_0P25",
            "knee_lambda": None,
            "best_observed_lambda": 0.5,
            "knee_bracket_lambda": None,
            "interpretation": (
                "the first interval pays; the preregistered lambda-0.25 rung is "
                "required before locating or bounding a knee"
            ),
        }
    if not adjacent[1]["real_break_even_pass"]:
        return {
            "status": "BRACKETED_BY_ADJACENT_SECANTS",
            "knee_lambda": None,
            "best_observed_lambda": 0.5,
            "knee_bracket_lambda": [0.25, 1.0],
            "interpretation": (
                "the 1.0-to-0.5 secant pays and the 0.5-to-0.25 secant does not; "
                "lambda 0.5 is the best observed rung, not an exact optimum"
            ),
        }
    return {
        "status": "UNBRACKETED_LOWER_LAMBDA",
        "knee_lambda": None,
        "best_observed_lambda": 0.25,
        "knee_bracket_lambda": None,
        "best_observed_boundary_lambda": 0.25,
        "interpretation": (
            "both observed prior-growth intervals pay; the knee lies beyond the "
            "lower-lambda boundary if the trend continues"
        ),
    }


def fit(payload: dict[str, Any]) -> dict[str, Any]:
    rows = _validated_rows(payload)
    aggregates = _aggregate_by_lambda(rows)
    adjacent = _adjacent_slopes(aggregates)
    final = len(aggregates) == 3
    if final and (
        not adjacent
        or adjacent[0]["range_token_bytes_per_model_byte"] is None
        or not adjacent[0]["real_break_even_pass"]
    ):
        raise CL1FitError(
            "lambda 0.25 violates the conditional fire order because the "
            "lambda 1.0 to 0.5 Range secant was non-monotone or did not pay"
        )
    monotone = all(row["delta_model_bytes"] > 0.0 for row in adjacent)
    if final and monotone:
        x = [row["mean_packed_model_bytes"] for row in aggregates]
        ideal_fit = _fit_line(x, [row["mean_ideal_token_bytes"] for row in aggregates])
        range_fit = _fit_line(x, [row["mean_range_token_bytes"] for row in aggregates])
    else:
        ideal_fit = None
        range_fit = None
    knee = _knee_status(adjacent, final=final)
    best_experimental = min(aggregates, key=lambda row: row["mean_range_joint_bytes"])
    beats_pr130 = best_experimental["mean_range_joint_bytes"] < PR130_RANGE_JOINT_BYTES
    control_rows = [row for row in rows if row["rate_lambda"] == 1.0]
    repeat_floor = {
        "status": "EXACT_MATCH",
        "causal_state_equal": True,
        "packed_model_spread_bytes": max(row["packed_model_bytes"] for row in control_rows)
        - min(row["packed_model_bytes"] for row in control_rows),
        "range_token_spread_bytes": max(row["range_token_bytes"] for row in control_rows)
        - min(row["range_token_bytes"] for row in control_rows),
        "range_joint_spread_bytes": max(row["packed_model_bytes"] + row["range_token_bytes"] for row in control_rows)
        - min(row["packed_model_bytes"] + row["range_token_bytes"] for row in control_rows),
    }
    return {
        "schema": OUTPUT_SCHEMA,
        "score_claim": False,
        "axis": "[macOS-MPS research-signal; scorer-free real serialized bytes]",
        "fit_scope": (
            "FORMULATION: fixed-topology C64/P64/delta2/D8 learned bit allocation "
            "under the preregistered 60-epoch recipe"
        ),
        "measurement_shape": ("FINAL_THREE_UNIQUE_LAMBDAS" if final else "INTERIM_EXACT_SECANT"),
        "immutable_pr130_reference": dict(PR130_REFERENCE),
        "rows": rows,
        "aggregated_by_lambda": aggregates,
        "ideal_fit": ideal_fit,
        "range_fit": range_fit,
        "adjacent_slopes": adjacent,
        "control_repeat_floor": repeat_floor,
        "knee": knee,
        "verdict": {
            "capacity_order_status": ("MONOTONE_MODEL_GROWTH" if monotone else "NON_MONOTONE_MODEL_CAPACITY"),
            "fitted_range_break_even_status": (
                range_fit["ci95_break_even_status"] if range_fit is not None else "NOT_AVAILABLE"
            ),
            "all_observed_intervals_pay": bool(adjacent) and all(row["real_break_even_pass"] for row in adjacent),
            "knee_status": knee["status"],
            "knee_lambda": knee.get("knee_lambda"),
            "knee_bracket_lambda": knee.get("knee_bracket_lambda"),
            "best_observed_lambda": knee.get("best_observed_lambda"),
            "best_experimental_lambda": best_experimental["rate_lambda"],
            "best_experimental_range_joint_bytes": best_experimental["mean_range_joint_bytes"],
            "best_experimental_delta_range_joint_bytes_vs_pr130": (
                best_experimental["delta_range_joint_bytes_vs_pr130"]
            ),
            "best_experimental_beats_pr130": beats_pr130,
            "selected_section_candidate": (
                f"lambda_{best_experimental['rate_lambda']:.6g}" if beats_pr130 else PR130_NAME
            ),
            "break_even_rule": "real Range token bytes per model byte < -1",
        },
        "uncertainty_boundary": (
            "a fitted interval is emitted only for the final three unique lambda "
            "representatives; it is a conditional descriptive regression interval, "
            "not a training-population or contest-score interval"
        ),
    }


def _render_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# DDM CL1 serialized capacity fit",
        "",
        "`score_claim=false`; scorer-free byte measurement.",
        "",
        "| lambda | replicates | packed model B | ideal token B | Range token B | Range joint B | delta vs PR130 B |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result["aggregated_by_lambda"]:
        lines.append(
            f"| {row['rate_lambda']:.6g} | {row['replicates']} | "
            f"{row['mean_packed_model_bytes']:.3f} | "
            f"{row['mean_ideal_token_bytes']:.3f} | "
            f"{row['mean_range_token_bytes']:.3f} | "
            f"{row['mean_range_joint_bytes']:.3f} | "
            f"{row['delta_range_joint_bytes_vs_pr130']:+.3f} |"
        )
    verdict = result["verdict"]
    lines.extend(["", "## Exchange verdict", ""])
    if result["range_fit"] is None:
        lines.append("- Fitted Range slope: not emitted for this measurement shape.")
    else:
        real = result["range_fit"]
        lines.append(
            f"- Real Range slope: {real['slope_token_bytes_per_model_byte']:.6f} "
            f"(conditional 95% CI {real['ci95'][0]:.6f}, "
            f"{real['ci95'][1]:.6f}; "
            f"{real['ci95_break_even_status']})."
        )
    lines.extend(
        [
            f"- Knee status: {verdict['knee_status']}; "
            f"exact lambda={verdict['knee_lambda']}; "
            f"observed best={verdict['best_observed_lambda']}; "
            f"bracket={verdict['knee_bracket_lambda']}.",
            f"- All observed adjacent intervals pay: {str(verdict['all_observed_intervals_pay']).lower()}.",
            f"- Best experimental delta from PR130 Range joint: "
            f"{verdict['best_experimental_delta_range_joint_bytes_vs_pr130']:+.3f} B.",
            f"- Selected section candidate: {verdict['selected_section_candidate']}.",
            "",
            f"Uncertainty boundary: {result['uncertainty_boundary']}",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    fit_parser = subparsers.add_parser("fit")
    fit_parser.add_argument("--measurements", type=Path, required=True)
    fit_parser.add_argument("--out-json", type=Path, required=True)
    fit_parser.add_argument("--out-md", type=Path, required=True)
    for operation in ("pack", "encode", "decode"):
        operation_parser = subparsers.add_parser(operation)
        operation_parser.add_argument("--checkpoint", type=Path, required=True)
        if operation == "pack":
            operation_parser.add_argument("--blob", type=Path, required=True)
        else:
            operation_parser.add_argument("--tokens", type=Path, required=True)
        if operation == "decode":
            operation_parser.add_argument("--raw", type=Path, required=True)
        operation_parser.add_argument("--report", type=Path, required=True)
        operation_parser.add_argument("--attestation", type=Path, required=True)
    args = parser.parse_args()
    if args.command != "fit":
        result = _run_artifact_operation(args)
        print(json.dumps(result, sort_keys=True))
        return
    result = fit(_read_json(args.measurements))
    _atomic_json(args.out_json, result)
    _atomic_bytes(args.out_md, _render_markdown(result).encode("utf-8"))
    print(json.dumps(result["verdict"], sort_keys=True))


if __name__ == "__main__":
    main()
