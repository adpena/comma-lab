#!/usr/bin/env python3
"""Execute the F26R direct-context and conv-A-delta native performance rung."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import experiments.ddm_f26q_rc64_native_lowering as parent

WORK_DIR = Path("/Volumes/VertigoDataTier/pact/ddm_f26r_hpac_final_rung_20260814")
PARENT_DIR = Path("/Volumes/VertigoDataTier/pact/ddm_f26q_rc64_native_20260814")
ARCHIVE_SHA256 = "f0ba4bb41d55fff85542f2a17dfe682508aa4f9ab50ef51cda573d79f0c4b1de"
ARCHIVE_BYTES = 186_269
TOKEN_SHA256 = "9ba2e52b3096585895970066b389bf1261ebc203d5b828cdea056c13858aea52"
TOKEN_BYTES = 117_964_800
DECODER_BIT_POSITION = 921_964
CORRECTED_SHA256 = "617e9fcfc967c200f1ecc8bea93dd45a22f7af2a050092f982169b5f5e5a3523"
CDF_SHA256 = "ba0d529b7eaf6e16da1f62fc1cc7ca43ccc1b989356a68b8d37988088cb7c7ff"
PARENT_M5_SECONDS = 203.8433591669891
PARENT_M5_HIDDEN_SECONDS = 140.57528018951416
PARENT_M5_CONTEXT_SECONDS = 30.169883078080602
PARENT_M5_CONV_UPDATE_SECONDS = 22.297621250152588
MODAL_TOTAL_SECONDS = 2933.204220146
MODAL_TOKEN_SECONDS = 2613.9199947839998
M5_REFERENCE_TOKEN_SECONDS = 383.35438545793295
MODAL_STAGE_RATIO = MODAL_TOKEN_SECONDS / M5_REFERENCE_TOKEN_SECONDS
FIXED_NON_TOKEN_SECONDS = MODAL_TOTAL_SECONDS - MODAL_TOKEN_SECONDS
FIRE_THRESHOLD_SECONDS = 1600.0
REQUIRED_M5_SECONDS = (FIRE_THRESHOLD_SECONDS - FIXED_NON_TOKEN_SECONDS) / MODAL_STAGE_RATIO


class F26RFailure(RuntimeError):
    """A storage, build, identity, retention, or fire-gate invariant failed."""


def _sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _file_fact(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _storage_preflight(required_bytes: int = 1_073_741_824) -> dict[str, int]:
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(WORK_DIR)
    if usage.free < required_bytes:
        raise F26RFailure(
            f"Vertigo has {usage.free} free bytes; F26R requires {required_bytes}"
        )
    return {"free_bytes": usage.free, "required_bytes": required_bytes}


def _configure_parent(rung: str) -> Path:
    if not rung or any(character not in "abcdefghijklmnopqrstuvwxyz0123456789_" for character in rung):
        raise F26RFailure("rung must contain only lowercase letters, digits, and underscores")
    rung_dir = WORK_DIR / "rungs" / rung
    parent.WORK_DIR = rung_dir
    parent.CANDIDATE_STAGE = rung_dir / "submission_candidate_direct_context_delta_python_default"
    parent.SEALED_STAGE = rung_dir / "submission_native_sealed"
    parent.RETAINED_DIR = rung_dir / "retained"
    parent.RECEIPT_DIR = rung_dir / "receipts"
    parent.BUILD_DIR = parent.RETAINED_DIR / "native_build_direct_context_delta"
    parent.PROFILE_DIR = PARENT_DIR / "retained" / "profiles"
    return rung_dir


def _compile_command(source: Path, output: Path, *, scalar: bool) -> tuple[list[str], list[str]]:
    if sys.platform == "darwin":
        import torch

        torch_lib = Path(torch.__file__).resolve().parent / "lib"
        linker_flags = [
            "-Xpreprocessor",
            "-fopenmp",
            "-I/opt/homebrew/opt/libomp/include",
            f"-L{torch_lib}",
            "-lomp",
            f"-Wl,-rpath,{torch_lib}",
            "-Wl,-install_name,@rpath/f26_hpac_native.so",
        ]
    else:
        linker_flags = ["-fopenmp", "-Wl,-soname,f26_hpac_native.so"]
    command = [
        os.environ.get("CC", "cc"),
        "-O3",
        "-mcpu=native" if sys.platform == "darwin" else "-march=native",
        "-std=c11",
        "-shared",
        "-fPIC",
        "-ffp-contract=off",
        "-fno-fast-math",
        *(["-DF26_FORCE_SCALAR=1"] if scalar else []),
        *linker_flags,
        str(source),
        "-lm",
        "-o",
        str(output),
    ]
    return command, linker_flags


def _normalize_darwin_binary(path: Path) -> list[list[str]]:
    if sys.platform != "darwin":
        return []
    commands = [
        [
            "install_name_tool",
            "-change",
            "/opt/llvm-openmp/lib/libomp.dylib",
            "@rpath/libomp.dylib",
            str(path),
        ],
        ["codesign", "--remove-signature", str(path)],
        [
            "codesign",
            "-s",
            "-",
            "--force",
            "--timestamp=none",
            "-i",
            "f26_hpac_native",
            str(path),
        ],
    ]
    for command in commands:
        subprocess.run(command, check=True)
    return commands


def build_scalar_twin(repo_root: Path, rung: str) -> dict[str, Any]:
    """Build and retain two deterministic portable-scalar twin binaries."""
    rung_dir = _configure_parent(rung)
    parent.build(repo_root)
    build_dir = rung_dir / "retained" / "native_build_scalar_twin"
    receipt_path = rung_dir / "receipts" / "native_build_scalar_twin.json"
    outputs = [build_dir / "f26_hpac_scalar_a.so", build_dir / "f26_hpac_scalar_b.so"]
    if receipt_path.is_file():
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if not receipt.get("deterministic_binary"):
            raise F26RFailure("retained scalar-twin build is not deterministic")
        for fact in receipt["repeat_builds"]:
            path = Path(fact["path"])
            if not path.is_file() or _file_fact(path) != fact:
                raise F26RFailure("retained scalar-twin binary changed")
        return receipt
    if any(path.exists() for path in outputs):
        raise F26RFailure("scalar-twin binary exists without its build receipt")
    build_dir.mkdir(parents=True, exist_ok=True)
    source = parent.CANDIDATE_STAGE / "runtime" / "f26_hpac_native.c"
    commands: list[list[str]] = []
    facts = []
    linker_flags: list[str] = []
    for output in outputs:
        command, linker_flags = _compile_command(source, output, scalar=True)
        subprocess.run(command, check=True, cwd=repo_root)
        commands.append(command)
        commands.extend(_normalize_darwin_binary(output))
        facts.append(_file_fact(output))
    deterministic = facts[0]["sha256"] == facts[1]["sha256"]
    if not deterministic:
        raise F26RFailure("repeat scalar-twin builds are not byte-identical")
    receipt = {
        "schema": "ddm_f26r_scalar_twin_build.v1",
        "complete": True,
        "compiler": subprocess.run(
            [os.environ.get("CC", "cc"), "--version"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()[0],
        "source": _file_fact(source),
        "commands": commands,
        "linker_flags": linker_flags,
        "compile_definition": "F26_FORCE_SCALAR=1",
        "repeat_builds": facts,
        "deterministic_binary": deterministic,
        "role": "portable scalar equality twin for the NEON/AVX2-or-scalar source",
    }
    _atomic_json(receipt_path, receipt)
    return receipt


def _exact_gates(report: dict[str, Any]) -> dict[str, bool]:
    return {
        "decoded_token_sha256": report["decoded_token_sha256"] == TOKEN_SHA256,
        "decoded_token_bytes": report["decoded_token_bytes"] == TOKEN_BYTES,
        "decoder_bit_position": report["decoder_bit_position"] == DECODER_BIT_POSITION,
        "corrected_quantized_logit_sha256": (
            report["corrected_quantized_logit_sha256"] == CORRECTED_SHA256
        ),
        "corrected_cdf_input_sha256": report["corrected_cdf_input_sha256"] == CDF_SHA256,
        "digest_scope": report["digest_scope"] == "full_field",
    }


def run_variant(
    repo_root: Path,
    rung: str,
    *,
    variant: str,
    frames: int,
) -> dict[str, Any]:
    """Run and retain a fresh optimized-repeat or scalar-twin token field."""
    if variant not in {"optimized_repeat", "scalar_twin"}:
        raise F26RFailure(f"unsupported retained variant: {variant}")
    if frames <= 0 or frames > 600:
        raise F26RFailure("frames must be within the real n600 field")
    rung_dir = _configure_parent(rung)
    optimized_build = parent.build(repo_root)
    scalar_build = build_scalar_twin(repo_root, rung)
    binary_fact = (
        optimized_build["repeat_builds"][1]
        if variant == "optimized_repeat"
        else scalar_build["repeat_builds"][0]
    )
    binary = Path(binary_fact["path"])
    receipt_path = rung_dir / "receipts" / f"native_run_{variant}_n{frames}.json"
    if receipt_path.is_file():
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        payload = Path(receipt["native_report"]["decoded_token_path"])
        if not payload.is_file() or _file_fact(payload)["sha256"] != receipt["native_report"]["decoded_token_sha256"]:
            raise F26RFailure(f"retained {variant} payload changed")
        return receipt

    renderer, parts, _model, _sparse, torch = parent._load_runtime()
    from runtime.f26_hpac_native import decode_native_tokens

    torch.set_num_threads(4)
    output_path = rung_dir / "retained" / f"native_tokens_{variant}_n{frames}.u8"
    checkpoint_dir = rung_dir / "retained" / f"native_tokens_{variant}_n{frames}_checkpoints"
    os.environ["F26_HPAC_NATIVE_LIBRARY"] = str(binary)
    started = time.perf_counter()
    _tokens, report = decode_native_tokens(
        parts,
        renderer,
        parent.CANDIDATE_STAGE / "cpr1",
        torch.device("cpu"),
        frame_limit=frames,
        output_path=output_path,
        checkpoint_dir=checkpoint_dir,
    )
    wall_seconds = time.perf_counter() - started
    if frames == 600:
        gates = _exact_gates(report)
        if not all(gates.values()):
            raise F26RFailure(f"{variant} full-field identity failed: {gates}")
    else:
        reference = PARENT_DIR / "retained" / "profiles" / f"reference_prefix_n{frames}.u8"
        if not reference.is_file():
            raise F26RFailure(f"missing parent Python prefix oracle: {reference}")
        gates = {"prefix_token_bytes": reference.read_bytes() == output_path.read_bytes()}
        if not all(gates.values()):
            raise F26RFailure(f"{variant} prefix differs from the Python oracle")
    receipt = {
        "schema": "ddm_f26r_native_variant_run.v1",
        "complete": True,
        "axis_label": "[M5-CPU 4-thread scorer-free native token decode]",
        "variant": variant,
        "frames": frames,
        "wall_seconds": wall_seconds,
        "native_report": report,
        "gates": gates,
        "native_build": binary_fact,
    }
    _atomic_json(receipt_path, receipt)
    return receipt


def run_optimized(repo_root: Path, rung: str, frames: int) -> dict[str, Any]:
    """Run the adopted NEON/AVX2-or-scalar build through the parent six-gate harness."""
    _configure_parent(rung)
    return parent.run_native(repo_root, frames)


def summarize(repo_root: Path, rung: str) -> dict[str, Any]:
    """Re-prove all gates, compute the projection, and seal the fire order if eligible."""
    rung_dir = _configure_parent(rung)
    primary = run_optimized(repo_root, rung, 600)
    repeat = run_variant(repo_root, rung, variant="optimized_repeat", frames=600)
    scalar = run_variant(repo_root, rung, variant="scalar_twin", frames=600)
    primary_report = primary["native_report"]
    repeat_report = repeat["native_report"]
    scalar_report = scalar["native_report"]
    repeat_determinism = {
        "token_sha256": primary_report["decoded_token_sha256"]
        == repeat_report["decoded_token_sha256"],
        "corrected_logit_sha256": primary_report["corrected_quantized_logit_sha256"]
        == repeat_report["corrected_quantized_logit_sha256"],
        "cdf_input_sha256": primary_report["corrected_cdf_input_sha256"]
        == repeat_report["corrected_cdf_input_sha256"],
        "decoder_bit_position": primary_report["decoder_bit_position"]
        == repeat_report["decoder_bit_position"],
    }
    scalar_twin_parity = {
        "token_sha256": primary_report["decoded_token_sha256"]
        == scalar_report["decoded_token_sha256"],
        "corrected_logit_sha256": primary_report["corrected_quantized_logit_sha256"]
        == scalar_report["corrected_quantized_logit_sha256"],
        "cdf_input_sha256": primary_report["corrected_cdf_input_sha256"]
        == scalar_report["corrected_cdf_input_sha256"],
        "decoder_bit_position": primary_report["decoder_bit_position"]
        == scalar_report["decoder_bit_position"],
    }
    if not all(repeat_determinism.values()) or not all(scalar_twin_parity.values()):
        raise F26RFailure("repeat determinism or scalar-twin full-field parity failed")

    measured_m5_seconds = float(primary_report["decode_runtime_seconds"])
    projected_token_seconds = measured_m5_seconds * MODAL_STAGE_RATIO
    projected_total_seconds = FIXED_NON_TOKEN_SECONDS + projected_token_seconds
    fire_ready = projected_total_seconds <= FIRE_THRESHOLD_SECONDS
    sealed = parent.seal(repo_root) if fire_ready else None
    fire_order = None
    if fire_ready:
        sealed_root = str(sealed["sealed_stage"]["root"])
        run_id = "ddm_f26r_mc36_contest_cpu_20260814"
        command = [
            "env",
            "PYTHONPATH=src:upstream:$PWD",
            ".venv/bin/modal",
            "run",
            "--detach",
            "experiments/modal_auth_eval_cpu.py::main",
            "--archive",
            f"{sealed_root}/archive.zip",
            "--submission-dir",
            sealed_root,
            "--inflate-sh",
            "inflate.sh",
            "--expected-archive-sha256",
            ARCHIVE_SHA256,
            "--expected-runtime-tree-sha256",
            "auto",
            "--output-dir",
            f"experiments/results/{run_id}",
            "--inflate-timeout",
            "1800",
            "--evaluate-timeout",
            "5400",
            "--scorer-input-cache-tensor-volume-run-id",
            run_id,
            "--detach",
            "--provider-detach-ack",
            "--lane-id",
            f"lane_{run_id}",
            "--instance-job-id",
            run_id,
            "--claim-agent",
            "MAIN",
            "--claim-policy",
            "require_active",
            "--pair-group-id",
            "ddm_mc36_promotion_paired_modal_auth_20260814T182512Z",
        ]
        fire_order = {
            "schema": "ddm_f26r_sealed_fire_order.v1",
            "disposition": "SEALED_FIRE_ORDER",
            "owner": "MAIN",
            "consumer_store": ".omx/state/main_hot_state.md and the Modal returned-artifact store",
            "fire_trigger": "this file exists after all F26R n600 identity gates passed and derived Modal total is at most 1600 seconds",
            "axis": "[contest-CPU Modal x86_64, n600]",
            "canonical_chain": "experiments/modal_auth_eval_cpu.py",
            "command_argv": command,
            "lane_preclaim_required": True,
            "archive": {"bytes": ARCHIVE_BYTES, "sha256": ARCHIVE_SHA256},
            "runtime_stage": sealed["sealed_stage"],
            "token_decoder": "native-hpac",
            "cpu_threads": 4,
            "payload_retention": {
                "required": True,
                "volume": "comma-auth-eval-cache-artifacts",
                "must_retain": "the canonical wrapper now commits inflated_outputs_manifest.json, including per-frame identity and aggregate SHA-256, to the volume before returning",
                "container_ephemeral_raw_forbidden": True,
            },
            "returned_bundle_required": [
                "archive and uploaded runtime-tree hashes",
                "stdout and stderr",
                "contest_auth_eval.json and recomputed score components",
                "inflated_outputs_manifest.json",
                "inflated_outputs_volume_manifest.json with volume path and manifest SHA-256",
                "command and config",
                "hardware axis",
                "failure sentinel",
            ],
            "derived_projection": {
                "label": "DERIVED from the measured f26q M5-to-Modal token-stage ratio; not a Modal measurement",
                "ratio": MODAL_STAGE_RATIO,
                "measured_m5_token_seconds": measured_m5_seconds,
                "projected_modal_token_seconds": projected_token_seconds,
                "fixed_non_token_seconds": FIXED_NON_TOKEN_SECONDS,
                "projected_modal_total_seconds": projected_total_seconds,
                "gate_seconds": FIRE_THRESHOLD_SECONDS,
            },
        }
        _atomic_json(WORK_DIR / "SEALED_FIRE_ORDER.json", fire_order)

    result = {
        "schema": "ddm_f26r_result.v1",
        "complete": True,
        "rung": rung,
        "score_claim": False,
        "frontier_moved": False,
        "exact_row_fired": False,
        "measured": {
            "axis_label": "[M5-CPU 4-thread scorer-free native token decode]",
            "parent_m5_token_seconds": PARENT_M5_SECONDS,
            "f26r_m5_token_seconds": measured_m5_seconds,
            "m5_seconds_removed": PARENT_M5_SECONDS - measured_m5_seconds,
            "m5_speedup_vs_parent": PARENT_M5_SECONDS / measured_m5_seconds,
            "required_m5_seconds": REQUIRED_M5_SECONDS,
            "required_parent_seconds_to_remove": PARENT_M5_SECONDS - REQUIRED_M5_SECONDS,
            "primary_payload": _file_fact(Path(primary_report["decoded_token_path"])),
            "repeat_payload": _file_fact(Path(repeat_report["decoded_token_path"])),
            "scalar_twin_payload": _file_fact(Path(scalar_report["decoded_token_path"])),
            "primary_stage_seconds": primary_report["stage_seconds"],
            "repeat_stage_seconds": repeat_report["stage_seconds"],
            "scalar_twin_stage_seconds": scalar_report["stage_seconds"],
        },
        "identity_gates": {
            "primary_full_field": primary["gates"],
            "repeat_full_field": repeat["gates"],
            "scalar_twin_full_field": scalar["gates"],
            "repeat_determinism": repeat_determinism,
            "scalar_twin_parity": scalar_twin_parity,
        },
        "derived_modal_projection": {
            "label": "DERIVED from the measured f26q M5-to-Modal token-stage ratio; not a Modal measurement",
            "measured_stage_ratio": MODAL_STAGE_RATIO,
            "projected_token_seconds": projected_token_seconds,
            "fixed_non_token_seconds_from_failed_modal_run": FIXED_NON_TOKEN_SECONDS,
            "projected_total_seconds": projected_total_seconds,
            "fire_threshold_seconds": FIRE_THRESHOLD_SECONDS,
            "verdict": "SEALED_FIRE_ORDER" if fire_ready else "DECODE_ENGINEERING_GATED",
        },
        "fire_ready": fire_ready,
        "sealed_stage": None if sealed is None else sealed["sealed_stage"],
        "sealed_fire_order": fire_order,
        "decode_engineering_residual": None
        if fire_ready
        else {
            "disposition": "QUEUED_WITH_A_FIRE_ORDER",
            "verdict_scope": "INSTANCE(F26R direct-int16-context plus conv-A-delta on M5 and the measured f26q Modal stage ratio)",
            "remaining_measured_m5_seconds_to_fire_gate": measured_m5_seconds - REQUIRED_M5_SECONDS,
            "dominant_element": max(
                (
                    (name, float(value))
                    for name, value in primary_report["stage_seconds"].items()
                    if name.startswith("native_")
                ),
                key=lambda item: item[1],
            )[0],
            "owner": "successor native-runtime arm",
            "consumer_store": str(WORK_DIR / "receipts" / "result.json"),
            "fire_trigger": "a new full-n600 byte-identical receipt projects Modal total at or below 1600 seconds",
        },
        "custody": {
            "work_dir": str(WORK_DIR),
            "rung_dir": str(rung_dir),
            "cleanup_disposition": "keep every binary, token payload, checkpoint, and receipt; no deletion authorized",
            "storage_after_runs": {
                "free_bytes": shutil.disk_usage(WORK_DIR).free,
                "heavy_launch_required_bytes": 1_073_741_824,
                "note": "the heavy-launch preflight passed before materialization; this is the post-retention free-space fact",
            },
        },
    }
    _atomic_json(WORK_DIR / "receipts" / "result.json", result)
    return result


def run_command(repo_root: Path, command: str, rung: str, frames: int) -> dict[str, Any]:
    _storage_preflight(67_108_864 if command == "summarize" else 1_073_741_824)
    _configure_parent(rung)
    if command == "prepare":
        return parent.prepare(repo_root)
    if command == "build":
        optimized = parent.build(repo_root)
        scalar = build_scalar_twin(repo_root, rung)
        return {"optimized": optimized, "scalar_twin": scalar}
    if command == "run-optimized":
        return run_optimized(repo_root, rung, frames)
    if command == "run-optimized-repeat":
        return run_variant(repo_root, rung, variant="optimized_repeat", frames=frames)
    if command == "run-scalar-twin":
        return run_variant(repo_root, rung, variant="scalar_twin", frames=frames)
    if command == "summarize":
        return summarize(repo_root, rung)
    if command == "all":
        parent.prepare(repo_root)
        return summarize(repo_root, rung)
    raise F26RFailure(f"unsupported command: {command}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=(
            "prepare",
            "build",
            "run-optimized",
            "run-optimized-repeat",
            "run-scalar-twin",
            "summarize",
            "all",
        ),
    )
    parser.add_argument("--rung", default="direct_context_delta_v1")
    parser.add_argument("--frames", type=int, default=600)
    args = parser.parse_args()
    for name, value in parent.THREAD_ENV.items():
        os.environ[name] = value
    result = run_command(Path(__file__).resolve().parents[1], args.command, args.rung, args.frames)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
