#!/usr/bin/env python3
"""Resumably decode and locally score the sealed Einstein--Kolmogorov packet.

This is deliberately a measurement harness, not an optimizer.  It consumes one
already-sealed packet, checkpoints the full raw decode before invoking the hard
CPU-Torch oracle, and records enough custody to certify deletion of the 3.66 GB
rebuildable raw after a successful score receipt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from tac import contest_score  # noqa: E402
from tools import levelset_byte_close_and_eval as byte_close  # noqa: E402

CONFIG_SCHEMA = "einstein_kolmogorov_crux_v3_measurement_config.v1"
INFLATE_STAGE_SCHEMA = "einstein_kolmogorov_crux_v3_inflate_stage.v1"
SCORE_STAGE_SCHEMA = "einstein_kolmogorov_crux_v3_score_stage.v1"
RECEIPT_SCHEMA = "einstein_kolmogorov_crux_v3_receipt.v1"
AUTHORITY = "[macOS-CPU advisory] NON-PROMOTABLE"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".partial", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, sort_keys=True, indent=2, ensure_ascii=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object at {path}")
    return payload


def _require_file(path: Path, expected_sha256: str | None = None) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    if not resolved.is_file():
        raise ValueError(f"not a regular file: {resolved}")
    digest = _sha256(resolved)
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError(
            f"SHA-256 mismatch for {resolved}: measured {digest}, expected {expected_sha256}"
        )
    return {"path": str(resolved), "bytes": resolved.stat().st_size, "sha256": digest}


def _load_config(path: Path) -> dict[str, Any]:
    config = _read_json(path)
    if config.get("schema") != CONFIG_SCHEMA:
        raise ValueError(f"config schema must be {CONFIG_SCHEMA!r}")
    required = {
        "packet_dir",
        "expected_archive_bytes",
        "expected_archive_sha256",
        "expected_inflate_py_sha256",
        "strict_n24_receipt_path",
        "strict_n24_receipt_sha256",
        "gt_cache_path",
        "gt_cache_sha256",
        "inflate_stage_receipt_path",
        "score_stage_receipt_path",
        "result_receipt_path",
        "comparison_bank_score",
        "comparison_byte_box",
        "cleanup_raw_after_success",
    }
    missing = sorted(required - config.keys())
    if missing:
        raise ValueError(f"config missing required keys: {missing}")
    if isinstance(config["expected_archive_bytes"], bool) or int(config["expected_archive_bytes"]) <= 0:
        raise ValueError("expected_archive_bytes must be a positive integer")
    if float(config["comparison_bank_score"]) <= 0.0:
        raise ValueError("comparison_bank_score must be positive")
    if int(config["comparison_byte_box"]) <= 0:
        raise ValueError("comparison_byte_box must be positive")
    if not isinstance(config["cleanup_raw_after_success"], bool):
        raise ValueError("cleanup_raw_after_success must be boolean")
    return config


def _git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True, encoding="utf-8"
    ).strip()


def _validate_packet(config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    packet_dir = Path(config["packet_dir"]).resolve(strict=True)
    archive = _require_file(
        packet_dir / "archive.zip", str(config["expected_archive_sha256"])
    )
    if archive["bytes"] != int(config["expected_archive_bytes"]):
        raise ValueError(
            f"archive bytes mismatch: measured {archive['bytes']}, "
            f"expected {config['expected_archive_bytes']}"
        )
    inflate = _require_file(
        packet_dir / "inflate.py", str(config["expected_inflate_py_sha256"])
    )
    with zipfile.ZipFile(archive["path"]) as packet:
        if packet.namelist() != ["0.bin"]:
            raise ValueError(f"archive members must be exactly ['0.bin'], got {packet.namelist()}")
        blob = packet.read("0.bin")
    manifest = byte_close._read_blob_bytes(blob)[0]
    if int(manifest.get("n_pairs", -1)) != 600:
        raise ValueError(f"sealed packet n_pairs must be 600, got {manifest.get('n_pairs')}")
    archive["member_0bin_bytes"] = len(blob)
    archive["member_0bin_sha256"] = hashlib.sha256(blob).hexdigest()
    return {"archive": archive, "inflate_py": inflate}, manifest


def _validate_n24_receipt(config: dict[str, Any], archive_sha256: str) -> dict[str, Any]:
    custody = _require_file(
        Path(config["strict_n24_receipt_path"]), str(config["strict_n24_receipt_sha256"])
    )
    receipt = _read_json(Path(custody["path"]))
    nested = receipt.get("nested_levelset_report") or {}
    gate = nested.get("bit_exact_roundtrip_gate") or {}
    packet = nested.get("byte_close") or {}
    if not (gate.get("checked") is True and gate.get("bit_exact") is True):
        raise ValueError("strict n24 receipt does not prove a checked bit-exact gate")
    if int(gate.get("max_abs_uint8_diff", -1)) != 0:
        raise ValueError("strict n24 receipt has nonzero max_abs_uint8_diff")
    if int(gate.get("gate_pairs", -1)) != 24 or int(gate.get("frames_compared", -1)) != 48:
        raise ValueError(
            "strict n24 receipt must prove all 24 pairs / 48 frames; "
            f"got gate_pairs={gate.get('gate_pairs')} frames_compared={gate.get('frames_compared')}"
        )
    if packet.get("archive_zip_sha256") != archive_sha256:
        raise ValueError("strict n24 receipt archive SHA does not match the sealed packet")
    custody["bit_exact_gate"] = gate
    custody["pose_carrier_confirmation"] = nested.get("pose_carrier_confirmation")
    return custody


def _inflate_stage_valid(
    stage_path: Path, raw_path: Path, archive_sha256: str, inflate_py_sha256: str
) -> tuple[dict[str, Any], bool]:
    if not stage_path.is_file() or not raw_path.is_file():
        return {}, False
    stage = _read_json(stage_path)
    if stage.get("schema") != INFLATE_STAGE_SCHEMA:
        return stage, False
    if stage.get("archive_sha256") != archive_sha256:
        return stage, False
    if stage.get("inflate_py_sha256") != inflate_py_sha256:
        return stage, False
    expected_bytes = 1200 * byte_close.CAMERA_H * byte_close.CAMERA_W * 3
    if raw_path.stat().st_size != expected_bytes or stage.get("raw_bytes") != expected_bytes:
        return stage, False
    measured_sha = _sha256(raw_path)
    return stage, measured_sha == stage.get("raw_sha256")


def _run_or_resume_inflate(
    config: dict[str, Any], packet: dict[str, Any], command_argv: list[str]
) -> dict[str, Any]:
    packet_dir = Path(config["packet_dir"]).resolve(strict=True)
    raw_path = packet_dir / "inflated" / "0.raw"
    stage_path = Path(config["inflate_stage_receipt_path"])
    stage, valid = _inflate_stage_valid(
        stage_path,
        raw_path,
        packet["archive"]["sha256"],
        packet["inflate_py"]["sha256"],
    )
    if valid:
        print(f"[resume] verified full raw stage {stage['raw_sha256']}", flush=True)
        return stage

    print("[stage inflate] decoding the exact sealed n600 packet", flush=True)
    info = byte_close.run_inflate(packet_dir, n_pairs_total=600, max_pairs=None)
    raw_path = Path(info["raw_path"])
    raw_sha256 = _sha256(raw_path)
    stage = {
        "schema": INFLATE_STAGE_SCHEMA,
        "utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "authority": AUTHORITY,
        "archive_sha256": packet["archive"]["sha256"],
        "inflate_py_sha256": packet["inflate_py"]["sha256"],
        "raw_path": str(raw_path),
        "raw_bytes": raw_path.stat().st_size,
        "raw_sha256": raw_sha256,
        "run_inflate": info,
        "command_argv": command_argv,
        "rebuildable_from": {
            "archive": packet["archive"],
            "inflate_py": packet["inflate_py"],
            "function": "tools.levelset_byte_close_and_eval.run_inflate(max_pairs=None)",
        },
    }
    _atomic_json(stage_path, stage)
    print(f"[stage inflate] checkpointed raw sha256={raw_sha256}", flush=True)
    return stage


def _oracle_custody(config: dict[str, Any]) -> dict[str, Any]:
    """Content bindings required before a hard-oracle stage may be reused."""

    return {
        "gt_cache": _require_file(
            Path(config["gt_cache_path"]), str(config["gt_cache_sha256"])
        ),
        "byte_close_tool": _require_file(Path(byte_close.__file__)),
        "hard_oracle_module": _require_file(Path(byte_close.twr.__file__)),
        "contest_score": _require_file(Path(contest_score.__file__)),
    }


def _score_stage_valid(
    stage: dict[str, Any],
    packet: dict[str, Any],
    inflate_stage: dict[str, Any],
    oracle_custody: dict[str, Any],
) -> bool:
    scorer = stage.get("scorer") or {}
    bindings = scorer.get("content_bindings") or {}
    return bool(
        stage.get("schema") == SCORE_STAGE_SCHEMA
        and stage.get("archive_sha256") == packet["archive"]["sha256"]
        and stage.get("raw_sha256") == inflate_stage["raw_sha256"]
        and (stage.get("parity") or {}).get("pairs_scored") == 600
        and bindings == oracle_custody
    )


def _run_or_resume_score(
    config: dict[str, Any], packet: dict[str, Any], inflate_stage: dict[str, Any]
) -> dict[str, Any]:
    oracle_custody = _oracle_custody(config)
    stage_path = Path(config["score_stage_receipt_path"])
    if stage_path.is_file():
        stage = _read_json(stage_path)
        if _score_stage_valid(stage, packet, inflate_stage, oracle_custody):
            print("[resume] verified full hard-oracle score stage", flush=True)
            return stage

    print("[stage score] running frozen hard CPU-Torch SegNet/PoseNet over 600 pairs", flush=True)
    parity = byte_close.parity_on_inflated(
        Path(inflate_stage["raw_path"]), 600, str(config["gt_cache_path"]), 600
    )
    archive_bytes = int(packet["archive"]["bytes"])
    d_seg = float(parity["d_seg_realized_on_inflated"])
    d_pose = float(parity["d_pose_realized_on_inflated"])
    score = float(contest_score.compute_contest_score(d_seg, d_pose, archive_bytes))
    stage = {
        "schema": SCORE_STAGE_SCHEMA,
        "utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "authority": AUTHORITY,
        "archive_sha256": packet["archive"]["sha256"],
        "raw_sha256": inflate_stage["raw_sha256"],
        "parity": parity,
        "canonical_terms": {
            "seg_term": float(contest_score.seg_term(d_seg)),
            "pose_term": float(contest_score.pose_term(d_pose)),
            "rate_term": float(contest_score.rate_term(archive_bytes)),
            "projected_total_S": score,
            "formula": "100*d_seg + sqrt(10*d_pose) + 25*archive_bytes/37545489",
            "source": "tac.contest_score",
        },
        "scorer": {
            "surface": "tools.levelset_byte_close_and_eval.parity_on_inflated",
            "device": "cpu",
            "torch_version": torch.__version__,
            "torch_num_threads": torch.get_num_threads(),
            "content_bindings": oracle_custody,
        },
    }
    _atomic_json(stage_path, stage)
    print(f"[stage score] checkpointed projected S={score:.12f}", flush=True)
    return stage


def _build_final_receipt(
    config_path: Path,
    config: dict[str, Any],
    packet: dict[str, Any],
    manifest: dict[str, Any],
    n24: dict[str, Any],
    inflate_stage: dict[str, Any],
    score_stage: dict[str, Any],
    command_argv: list[str],
) -> dict[str, Any]:
    parity = score_stage["parity"]
    terms = score_stage["canonical_terms"]
    archive_bytes = int(packet["archive"]["bytes"])
    bank = float(config["comparison_bank_score"])
    byte_box = int(config["comparison_byte_box"])
    source_files = {
        "measurement_tool": _require_file(Path(__file__)),
        "byte_close_tool": _require_file(Path(byte_close.__file__)),
        "contest_score": _require_file(Path(contest_score.__file__)),
        "hard_oracle_module": _require_file(Path(byte_close.twr.__file__)),
    }
    return {
        "schema": RECEIPT_SCHEMA,
        "utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "classification": "A",
        "authority": AUTHORITY,
        "promotion_claim": False,
        "pointer_moved": False,
        "research_only": True,
        "git_head": _git_head(),
        "command_argv": command_argv,
        "config": _require_file(config_path),
        "runtime": {
            "platform": platform.platform(),
            "python": sys.version,
            "python_executable": sys.executable,
            "numpy_version": np.__version__,
            "torch_version": torch.__version__,
        },
        "source_files": source_files,
        "packet": packet,
        "packet_manifest": {
            "n_pairs": manifest["n_pairs"],
            "pose_carrier": manifest.get("pose_carrier"),
            "code_shape": manifest.get("code_shape"),
        },
        "strict_n24_bit_identity": n24,
        "full_n600_decode": inflate_stage,
        "full_n600_hard_cpu_torch_oracle": score_stage,
        "verdict": {
            "archive_bytes_measured": archive_bytes,
            "archive_sha256_measured": packet["archive"]["sha256"],
            "d_seg_measured": parity["d_seg_realized_on_inflated"],
            "d_pose_measured": parity["d_pose_realized_on_inflated"],
            "pairs_scored_measured": parity["pairs_scored"],
            "projected_total_S_derived": terms["projected_total_S"],
            "comparison_bank_score_settled_input": bank,
            "delta_S_vs_bank_derived": float(terms["projected_total_S"]) - bank,
            "comparison_byte_box_settled_input": byte_box,
            "byte_box_headroom_derived": byte_box - archive_bytes,
            "inside_byte_box_derived": archive_bytes <= byte_box,
        },
        "measured_derived_assumed": {
            "MEASURED": [
                "archive bytes and SHA-256 on the exact retained packet",
                "strict n24 shipped-receiver bit identity",
                "full n600 decoded raw bytes and SHA-256",
                "d_seg and d_pose from the hard CPU-Torch oracle over all 600 pairs",
            ],
            "DERIVED": [
                "canonical score terms and projected total S via tac.contest_score",
                "delta versus the settled 0.19108 bank and headroom versus the 264320-byte box",
            ],
            "SETTLED_INPUT": ["comparison bank score", "comparison byte box"],
            "ASSUMED": [],
        },
        "verdict_scope": (
            "Same-packet byte-closed local macOS CPU measurement only. It is not contest-CPU "
            "Linux x86_64 or contest-CUDA evidence and cannot move the frontier pointer."
        ),
        "cleanup": {
            "performed": False,
            "target": inflate_stage["raw_path"],
            "bytes": inflate_stage["raw_bytes"],
            "sha256": inflate_stage["raw_sha256"],
            "reason": "full raw is rebuildable scratch after the score and custody receipts land",
            "rebuild_command": command_argv,
        },
    }


def _resume_existing_result(
    config_path: Path,
    config: dict[str, Any],
    packet: dict[str, Any],
    n24: dict[str, Any],
    result_path: Path,
) -> dict[str, Any]:
    """Revalidate every content binding and finish interrupted raw cleanup."""

    receipt = _read_json(result_path)
    if receipt.get("schema") != RECEIPT_SCHEMA or receipt.get("classification") != "A":
        raise ValueError(f"existing result is not a completed {RECEIPT_SCHEMA} A receipt")
    if receipt.get("packet") != packet:
        raise ValueError("existing result packet custody does not match live sealed bytes")
    if receipt.get("config") != _require_file(config_path):
        raise ValueError("existing result config content binding does not match live config")
    if (receipt.get("strict_n24_bit_identity") or {}).get("sha256") != n24["sha256"]:
        raise ValueError("existing result strict-n24 receipt binding does not match live evidence")

    inflate_stage = receipt.get("full_n600_decode") or {}
    if not (
        inflate_stage.get("schema") == INFLATE_STAGE_SCHEMA
        and inflate_stage.get("archive_sha256") == packet["archive"]["sha256"]
        and inflate_stage.get("inflate_py_sha256") == packet["inflate_py"]["sha256"]
    ):
        raise ValueError("existing result inflate stage is not bound to the live packet/receiver")
    if _read_json(Path(config["inflate_stage_receipt_path"])) != inflate_stage:
        raise ValueError("existing result inflate stage differs from its durable stage receipt")

    oracle_custody = _oracle_custody(config)
    score_stage = receipt.get("full_n600_hard_cpu_torch_oracle") or {}
    if not _score_stage_valid(score_stage, packet, inflate_stage, oracle_custody):
        raise ValueError("existing result score stage is not bound to the live GT/oracle sources")
    if _read_json(Path(config["score_stage_receipt_path"])) != score_stage:
        raise ValueError("existing result score stage differs from its durable stage receipt")

    source_files = {
        "measurement_tool": _require_file(Path(__file__)),
        "byte_close_tool": oracle_custody["byte_close_tool"],
        "contest_score": oracle_custody["contest_score"],
        "hard_oracle_module": oracle_custody["hard_oracle_module"],
    }
    if receipt.get("source_files") != source_files:
        raise ValueError("existing result source-file binding does not match live measurement code")

    cleanup = receipt.get("cleanup") or {}
    raw_path = Path(str(cleanup.get("target", "")))
    if raw_path.is_file():
        if raw_path.stat().st_size != int(cleanup.get("bytes", -1)):
            raise ValueError("cleanup target size differs from the certified raw")
        if _sha256(raw_path) != cleanup.get("sha256"):
            raise ValueError("cleanup target hash differs from the certified raw")
        if config["cleanup_raw_after_success"]:
            raw_path.unlink()
            cleanup["performed"] = True
            cleanup["performed_utc"] = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
            cleanup["target_exists_after_cleanup"] = raw_path.exists()
            receipt["cleanup"] = cleanup
            _atomic_json(result_path, receipt)
    elif config["cleanup_raw_after_success"] and cleanup.get("performed") is not True:
        raise ValueError("raw is absent but existing receipt does not certify completed cleanup")
    return receipt


def execute(config_path: Path, command_argv: list[str]) -> dict[str, Any]:
    config = _load_config(config_path)
    result_path = Path(config["result_receipt_path"])
    packet, manifest = _validate_packet(config)
    n24 = _validate_n24_receipt(config, packet["archive"]["sha256"])
    if result_path.is_file():
        receipt = _resume_existing_result(
            config_path, config, packet, n24, result_path
        )
        print(json.dumps(receipt["verdict"], sort_keys=True), flush=True)
        return receipt

    inflate_stage = _run_or_resume_inflate(config, packet, command_argv)
    score_stage = _run_or_resume_score(config, packet, inflate_stage)
    receipt = _build_final_receipt(
        config_path,
        config,
        packet,
        manifest,
        n24,
        inflate_stage,
        score_stage,
        command_argv,
    )
    _atomic_json(result_path, receipt)

    raw_path = Path(inflate_stage["raw_path"])
    if config["cleanup_raw_after_success"] and raw_path.is_file():
        raw_path.unlink()
        receipt["cleanup"]["performed"] = True
        receipt["cleanup"]["performed_utc"] = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        receipt["cleanup"]["target_exists_after_cleanup"] = raw_path.exists()
        _atomic_json(result_path, receipt)
    print(json.dumps(receipt["verdict"], sort_keys=True), flush=True)
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)
    command_argv = [sys.executable, str(Path(__file__).resolve()), "--config", str(args.config.resolve())]
    execute(args.config.resolve(strict=True), command_argv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
