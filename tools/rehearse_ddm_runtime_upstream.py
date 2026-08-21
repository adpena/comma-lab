#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Run the actual frozen upstream harness on an exported DDM E1/E2/E3/E4 packet."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Literal

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr  # noqa: E402

from tac.optimization.ddm_runtime_exporter import (  # noqa: E402
    _publish_or_verify,
    _sha256_file,
    load_config,
)
from tac.optimization.direct_description_measurement_ladder import (  # noqa: E402
    rfc8785_canonicalize,
)
from tac.process_group_kill import run_in_process_group  # noqa: E402


class HarnessError(ValueError):
    """The frozen upstream rehearsal failed closed."""


class DDME1UpstreamHarnessConfigV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_: Literal[
        "DDME1UpstreamHarnessConfigV1",
        "DDME2UpstreamHarnessConfigV1",
        "DDME3UpstreamHarnessConfigV1",
        "DDME4UpstreamHarnessConfigV1",
        "DDME4WS1UpstreamHarnessConfigV1",
        "DDMIC1UpstreamHarnessConfigV1",
        "DDMIC2UpstreamHarnessConfigV1",
        "DDMIC2UpstreamHarnessRecheckConfigV1",
        "DDMIC2UpstreamHarnessCleanPassConfigV1",
        "DDMIC2UpstreamHarnessFinalConfigV1",
    ] = Field(alias="schema")
    run_id: Literal[
        "ddm_e1_upstream_harness_20260723",
        "ddm_e2_upstream_harness_20260723",
        "ddm_e3_upstream_harness_20260723",
        "ddm_e4_brotli_upstream_harness_20260724",
        "ddm_e4_lzma1_fallback_upstream_harness_20260724",
        "ddm_e5_e4_ws1_brotli_upstream_harness_20260724",
        "ddm_e5_e4_ws1_lzma1_fallback_upstream_harness_20260724",
        "ddm_ic1_incumbent_brotli_upstream_harness_20260724",
        "ddm_ic2_optimal_incumbent_brotli_upstream_harness_20260724",
        "ddm_ic2_optimal_incumbent_brotli_upstream_harness_recheck_20260724",
        "ddm_ic2_optimal_incumbent_brotli_upstream_harness_clean_pass_20260724",
        "ddm_ic2_optimal_incumbent_brotli_upstream_harness_final_20260724",
    ]
    export_config_path: StrictStr
    upstream_root: StrictStr
    submission_directory: StrictStr
    python_executable: StrictStr
    device: Literal["cpu"]
    timeout_seconds: StrictInt
    research_only: Literal[True]
    execution_allowed: Literal[False]
    score_claim: Literal[False]


def _load_config(path: Path) -> DDME1UpstreamHarnessConfigV1:
    payload = path.read_bytes()
    value = json.loads(payload)
    if rfc8785_canonicalize(value) + b"\n" != payload:
        raise HarnessError("harness config is not canonical JSON plus one newline")
    config = DDME1UpstreamHarnessConfigV1.model_validate(value)
    if config.timeout_seconds != 1800:
        raise HarnessError("harness timeout must equal the contest decode budget")
    return config


def _parse_report(payload: str) -> dict[str, str | int]:
    patterns = {
        "d_pose": r"Average PoseNet Distortion: ([0-9.]+)",
        "d_seg": r"Average SegNet Distortion: ([0-9.]+)",
        "archive_bytes": r"Submission file size: ([0-9,]+) bytes",
        "score_rounded": r"Final score: .* = ([0-9.]+)",
    }
    result: dict[str, str | int] = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, payload)
        if match is None:
            raise HarnessError(f"upstream report lacks {key}")
        value = match.group(1)
        result[key] = int(value.replace(",", "")) if key == "archive_bytes" else value
    return result


def _failure_reasons(
    *,
    timeout_hit: bool,
    returncode: int,
    parse_error: str | None,
    parsed: dict[str, str | int],
    expected_archive_bytes: int,
    raw_identity: tuple[int, str],
    expected_raw: tuple[int, str],
    wallclock: float,
    timeout_seconds: int,
) -> list[str]:
    reasons = []
    if timeout_hit:
        reasons.append("harness_timeout")
    if returncode != 0:
        reasons.append(f"exit_code_{returncode}")
    if parse_error is not None:
        reasons.append(parse_error)
    if parsed and parsed["archive_bytes"] != expected_archive_bytes:
        reasons.append("upstream_archive_size_mismatch")
    if raw_identity != expected_raw:
        reasons.append("upstream_raw_identity_mismatch")
    if wallclock >= timeout_seconds:
        reasons.append("wallclock_budget_exceeded")
    return reasons


def rehearse(config_path: Path) -> tuple[dict, Path]:
    started = time.monotonic()
    config = _load_config(config_path)
    export_config_path = (REPO_ROOT / config.export_config_path).resolve()
    export_config_path.relative_to(REPO_ROOT)
    export_config = load_config(export_config_path)
    is_e2 = config.run_id == "ddm_e2_upstream_harness_20260723"
    is_e3 = config.run_id == "ddm_e3_upstream_harness_20260723"
    is_e4 = config.run_id in {
        "ddm_e4_brotli_upstream_harness_20260724",
        "ddm_e4_lzma1_fallback_upstream_harness_20260724",
    }
    is_e4_ws1 = config.run_id in {
        "ddm_e5_e4_ws1_brotli_upstream_harness_20260724",
        "ddm_e5_e4_ws1_lzma1_fallback_upstream_harness_20260724",
    }
    is_ic1 = config.run_id == "ddm_ic1_incumbent_brotli_upstream_harness_20260724"
    is_ic2 = config.run_id in {
        "ddm_ic2_optimal_incumbent_brotli_upstream_harness_20260724",
        "ddm_ic2_optimal_incumbent_brotli_upstream_harness_recheck_20260724",
        "ddm_ic2_optimal_incumbent_brotli_upstream_harness_clean_pass_20260724",
        "ddm_ic2_optimal_incumbent_brotli_upstream_harness_final_20260724",
    }
    is_ic2_recheck = config.run_id == "ddm_ic2_optimal_incumbent_brotli_upstream_harness_recheck_20260724"
    is_ic2_clean_pass = (
        config.run_id == "ddm_ic2_optimal_incumbent_brotli_upstream_harness_clean_pass_20260724"
    )
    is_ic2_final = config.run_id == "ddm_ic2_optimal_incumbent_brotli_upstream_harness_final_20260724"
    packet = (REPO_ROOT / export_config.output_directory).resolve()
    output_root = packet.parent
    export_receipt_payload = (
        output_root
        / (
            "ddm_e4_runtime_export_receipt.json"
            if is_e4
            else "ddm_ic2_runtime_export_receipt.json"
            if is_ic2
            else "ddm_ic1_runtime_export_receipt.json"
            if is_ic1
            else "ddm_e4_ws1_runtime_export_receipt.json"
            if is_e4_ws1
            else "ddm_e3_runtime_export_receipt.json"
            if is_e3
            else "ddm_e2_runtime_export_receipt.json"
            if is_e2
            else "ddm_e1_runtime_export_receipt.json"
        )
    ).read_bytes()
    export_receipt = json.loads(export_receipt_payload)
    if rfc8785_canonicalize(export_receipt) + b"\n" != export_receipt_payload:
        raise HarnessError("export receipt is not canonical JSON plus one newline")

    upstream_root = Path(config.upstream_root).resolve()
    submission = Path(config.submission_directory).resolve()
    proof_root = Path(export_config.proof_root).resolve()
    try:
        submission.relative_to(proof_root)
    except ValueError as exc:
        raise HarnessError("submission directory escaped governed proof root") from exc
    python_executable = Path(config.python_executable)
    if not python_executable.is_absolute() or not python_executable.is_file():
        raise HarnessError("configured Python executable is absent")

    upstream_files = {
        "evaluate.py": upstream_root / "evaluate.py",
        "evaluate.sh": upstream_root / "evaluate.sh",
        "video_names": upstream_root / "public_test_video_names.txt",
    }
    if any(not path.is_file() for path in upstream_files.values()):
        raise HarnessError("frozen upstream harness files are incomplete")
    if not (upstream_root / "videos").is_dir():
        raise HarnessError("frozen upstream videos are absent")

    submission.mkdir(parents=True, exist_ok=True)
    packet_identity = {}
    for name in ("archive.zip", "inflate.py", "inflate.sh"):
        source = packet / name
        payload = source.read_bytes()
        _publish_or_verify(submission / name, payload, executable=name.endswith((".py", ".sh")))
        packet_identity[name] = {
            "bytes": len(payload),
            "sha256": _sha256_file(source)[1],
        }

    argv = [
        "/usr/bin/env",
        f"PYTHON={python_executable}",
        "bash",
        str(upstream_files["evaluate.sh"]),
        "--submission-dir",
        str(submission),
        "--video-names-file",
        str(upstream_files["video_names"]),
        "--device",
        config.device,
    ]
    environment = dict(os.environ)
    environment["PYTHON"] = str(python_executable)
    environment["PATH"] = str(python_executable.parent) + os.pathsep + environment.get("PATH", "")
    timeout_hit = False
    try:
        completed = run_in_process_group(
            argv,
            cwd=upstream_root,
            env=environment,
            capture_output=True,
            text=True,
            timeout=config.timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        timeout_hit = True
        completed = subprocess.CompletedProcess(
            argv,
            124,
            stdout=(exc.stdout.decode("utf-8", "replace") if isinstance(exc.stdout, bytes) else exc.stdout or ""),
            stderr=(exc.stderr.decode("utf-8", "replace") if isinstance(exc.stderr, bytes) else exc.stderr or ""),
        )
    wallclock = time.monotonic() - started
    log_root = (
        submission.parent
        / "logs"
        / ("final" if is_ic2_final else "clean_pass" if is_ic2_clean_pass else "recheck")
        if is_ic2_recheck
        or is_ic2_clean_pass
        or is_ic2_final
        else submission.parent / "logs"
    )
    stdout_path = _publish_or_verify(log_root / "evaluate.stdout.txt", completed.stdout.encode("utf-8"))
    stderr_path = _publish_or_verify(log_root / "evaluate.stderr.txt", completed.stderr.encode("utf-8"))
    report_path = submission / "report.txt"
    report = report_path.read_text(encoding="utf-8") if report_path.is_file() else ""
    parse_error = None
    parsed = {}
    if completed.returncode == 0:
        try:
            parsed = _parse_report(report)
        except HarnessError as exc:
            parse_error = str(exc)
    raw_path = submission / "inflated" / "0.raw"
    raw_identity = _sha256_file(raw_path) if raw_path.is_file() else (0, "")
    upstream_git = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=upstream_root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    expected_raw = (
        int(export_receipt["output_identity"]["bytes"]),
        str(export_receipt["output_identity"]["sha256"]),
    )
    failure_reasons = _failure_reasons(
        timeout_hit=timeout_hit,
        returncode=completed.returncode,
        parse_error=parse_error,
        parsed=parsed,
        expected_archive_bytes=packet_identity["archive.zip"]["bytes"],
        raw_identity=raw_identity,
        expected_raw=expected_raw,
        wallclock=wallclock,
        timeout_seconds=config.timeout_seconds,
    )
    result = {
        "argv": argv,
        "device": config.device,
        "evidence_axis": "[macOS-CPU upstream frozen-harness advisory]",
        "execution_allowed": False,
        "exit_code": completed.returncode,
        "failure_reasons": failure_reasons,
        "interface": {
            "archive_path": str(submission / "archive.zip"),
            "inflate_argv_contract": ("inflate.sh <archive_dir> <inflated_dir> <video_names_file>"),
            "report_path": str(report_path),
            "video_names_path": str(upstream_files["video_names"]),
        },
        "packet": packet_identity,
        "parsed_report": parsed,
        "raw": {"bytes": raw_identity[0], "sha256": raw_identity[1]},
        "research_only": True,
        "runtime_coder": export_receipt["runtime"].get("coder"),
        "schema": (
            "ddm_e4_upstream_harness_receipt.v1"
            if is_e4
            else "ddm_ic2_upstream_harness_final_receipt.v1"
            if is_ic2_final
            else "ddm_ic2_upstream_harness_clean_pass_receipt.v1"
            if is_ic2_clean_pass
            else "ddm_ic2_upstream_harness_recheck_receipt.v1"
            if is_ic2_recheck
            else "ddm_ic2_upstream_harness_receipt.v1"
            if is_ic2
            else "ddm_ic1_upstream_harness_receipt.v1"
            if is_ic1
            else "ddm_e4_ws1_upstream_harness_receipt.v1"
            if is_e4_ws1
            else "ddm_e3_upstream_harness_receipt.v1"
            if is_e3
            else "ddm_e2_upstream_harness_receipt.v1"
            if is_e2
            else "ddm_e1_upstream_harness_receipt.v1"
        ),
        "score_claim": False,
        "status": "PASS" if not failure_reasons else "FAIL",
        "stderr": {
            "bytes": _sha256_file(stderr_path)[0],
            "path": str(stderr_path),
            "sha256": _sha256_file(stderr_path)[1],
        },
        "stdout": {
            "bytes": _sha256_file(stdout_path)[0],
            "path": str(stdout_path),
            "sha256": _sha256_file(stdout_path)[1],
        },
        "timeout_seconds": config.timeout_seconds,
        "under_1800_seconds": wallclock < config.timeout_seconds,
        "upstream": {
            "evaluate_py": {
                "bytes": _sha256_file(upstream_files["evaluate.py"])[0],
                "sha256": _sha256_file(upstream_files["evaluate.py"])[1],
            },
            "evaluate_sh": {
                "bytes": _sha256_file(upstream_files["evaluate.sh"])[0],
                "sha256": _sha256_file(upstream_files["evaluate.sh"])[1],
            },
            "git_sha": upstream_git,
            "root": str(upstream_root),
        },
        "wallclock_seconds": format(wallclock, ".6f"),
    }
    receipt_path = _publish_or_verify(
        output_root
        / (
            (
                "ddm_e4_brotli_upstream_harness_receipt.json"
                if config.run_id == "ddm_e4_brotli_upstream_harness_20260724"
                else "ddm_e4_lzma1_fallback_upstream_harness_receipt.json"
            )
            if is_e4
            else "ddm_ic2_upstream_harness_final_receipt.json"
            if is_ic2_final
            else "ddm_ic2_upstream_harness_clean_pass_receipt.json"
            if is_ic2_clean_pass
            else "ddm_ic2_upstream_harness_recheck_receipt.json"
            if is_ic2_recheck
            else "ddm_ic2_upstream_harness_receipt.json"
            if is_ic2
            else "ddm_ic1_upstream_harness_receipt.json"
            if is_ic1
            else "ddm_e4_ws1_upstream_harness_receipt.json"
            if is_e4_ws1
            else "ddm_e3_upstream_harness_receipt.json"
            if is_e3
            else "ddm_e2_upstream_harness_receipt.json"
            if is_e2
            else "ddm_e1_upstream_harness_receipt.json"
        ),
        rfc8785_canonicalize(result) + b"\n",
    )
    if failure_reasons:
        raise HarnessError(f"upstream harness failed: {failure_reasons}; receipt={receipt_path}")
    return result, receipt_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    config_path = Path(args.config).resolve()
    config_path.relative_to(REPO_ROOT)
    result, receipt_path = rehearse(config_path)
    print(
        json.dumps(
            {
                "parsed_report": result["parsed_report"],
                "receipt_path": str(receipt_path),
                "status": result["status"],
                "wallclock_seconds": result["wallclock_seconds"],
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
