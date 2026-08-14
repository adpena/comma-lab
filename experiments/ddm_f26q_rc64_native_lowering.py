#!/usr/bin/env python3
"""Profile, lower, verify, and seal the F26 native HPAC/RC64 CPU path."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

WORK_DIR = Path("/Volumes/VertigoDataTier/pact/ddm_f26q_rc64_native_20260814")
SOURCE_STAGE = Path(
    "/Volumes/VertigoDataTier/pact/ddm_f26p_runtime_cpu_lift_20260814/lifted_submission_cpu"
)
CANDIDATE_STAGE = WORK_DIR / "submission_candidate_v13_persistent_vector_python_default"
SEALED_STAGE = WORK_DIR / "submission_native_sealed"
RETAINED_DIR = WORK_DIR / "retained"
RECEIPT_DIR = WORK_DIR / "receipts"
BUILD_DIR = RETAINED_DIR / "native_build_v13_persistent_vector"
PROFILE_DIR = RETAINED_DIR / "profiles"

ARCHIVE_SHA256 = "f0ba4bb41d55fff85542f2a17dfe682508aa4f9ab50ef51cda573d79f0c4b1de"
ARCHIVE_BYTES = 186_269
TOKEN_SHA256 = "9ba2e52b3096585895970066b389bf1261ebc203d5b828cdea056c13858aea52"
TOKEN_BYTES = 117_964_800
DECODER_BIT_POSITION = 921_964
CORRECTED_SHA256 = "617e9fcfc967c200f1ecc8bea93dd45a22f7af2a050092f982169b5f5e5a3523"
CDF_SHA256 = "ba0d529b7eaf6e16da1f62fc1cc7ca43ccc1b989356a68b8d37988088cb7c7ff"
MODAL_TOTAL_SECONDS = 2933.204220146
MODAL_TOKEN_SECONDS = 2613.9199947839998
M5_REFERENCE_TOKEN_SECONDS = 383.35438545793295
MODAL_STAGE_RATIO = MODAL_TOKEN_SECONDS / M5_REFERENCE_TOKEN_SECONDS
THREAD_ENV = {
    "OMP_NUM_THREADS": "4",
    "MKL_NUM_THREADS": "4",
    "OPENBLAS_NUM_THREADS": "4",
    "VECLIB_MAXIMUM_THREADS": "4",
    "NUMEXPR_NUM_THREADS": "4",
}


class NativeLoweringError(RuntimeError):
    """A custody, parity, storage, or native-build invariant failed."""


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


def _manifest(root: Path) -> list[dict[str, Any]]:
    return [
        {
            "relative_path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
        for path in sorted(root.rglob("*"))
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
    ]


def _manifest_sha256(records: list[dict[str, Any]]) -> str:
    return hashlib.sha256(
        json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _replace_exact(path: Path, old: str, new: str, *, count: int = 1) -> None:
    source = path.read_text(encoding="utf-8")
    observed = source.count(old)
    if observed != count:
        raise NativeLoweringError(
            f"expected {count} transform sites in {path}, observed {observed}"
        )
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(source.replace(old, new), encoding="utf-8")
    os.replace(temporary, path)


def _storage_preflight(required_bytes: int = 536_870_912) -> dict[str, int]:
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(WORK_DIR)
    if usage.free < required_bytes:
        raise NativeLoweringError(
            f"Vertigo has {usage.free} bytes free; native lowering requires {required_bytes}"
        )
    return {"free_bytes": usage.free, "required_bytes": required_bytes}


def prepare(repo_root: Path) -> dict[str, Any]:
    """Build a fresh candidate tree whose runtime default remains Python."""
    storage = _storage_preflight()
    failed_prefix = RETAINED_DIR / "native_tokens_n4.u8"
    failed_prefix_receipt = RECEIPT_DIR / "native_prefix_failed_dlopen.json"
    if failed_prefix.is_file() and not failed_prefix_receipt.exists():
        _atomic_json(
            failed_prefix_receipt,
            {
                "schema": "ddm_f26q_native_prefix_failure.v1",
                "complete": False,
                "reason": "Darwin rejected the no-UUID dylib before native decoding began",
                "materialized_token_field": _file_fact(failed_prefix),
                "decoded_frames": 0,
                "retention": "preserved; excluded from parity and timing evidence",
            },
        )
    archive = SOURCE_STAGE / "archive.zip"
    if not archive.is_file() or archive.stat().st_size != ARCHIVE_BYTES:
        raise NativeLoweringError("source F26 archive is missing or has the wrong size")
    if _sha256_file(archive) != ARCHIVE_SHA256:
        raise NativeLoweringError("source F26 archive differs from the charter SHA-256")
    source_c = repo_root / "runtime-rs" / "native" / "f26-hpac" / "f26_hpac_native.c"
    source_py = repo_root / "experiments" / "ddm_f26q_f26_hpac_native.py"
    if not source_c.is_file() or not source_py.is_file():
        raise NativeLoweringError("native runtime sources are incomplete")

    failed_v2 = RETAINED_DIR / "native_tokens_v2_n4.u8"
    failed_v2_receipt = RECEIPT_DIR / "native_prefix_v2_context_mismatch.json"
    if failed_v2.is_file() and not failed_v2_receipt.exists():
        reference_v2 = PROFILE_DIR / "reference_prefix_n4.u8"
        changed = None
        if reference_v2.is_file() and reference_v2.stat().st_size == failed_v2.stat().st_size:
            changed = int(
                np.count_nonzero(
                    np.fromfile(reference_v2, dtype=np.uint8)
                    != np.fromfile(failed_v2, dtype=np.uint8)
                )
            )
        _atomic_json(
            failed_v2_receipt,
            {
                "schema": "ddm_f26q_native_prefix_failure.v1",
                "complete": False,
                "reason": "native v2 omitted the reference past-plus-SPM clamp before hidden addition",
                "verdict_scope": "INSTANCE(native v2 context composition)",
                "materialized_token_field": _file_fact(failed_v2),
                "reference_token_field": None if not reference_v2.is_file() else _file_fact(reference_v2),
                "changed_token_bytes": changed,
                "retention": "preserved; excluded from parity and timing evidence",
            },
        )
    v3_receipt = RECEIPT_DIR / "native_run_n4.json"
    v3_fold_receipt = RECEIPT_DIR / "native_v3_parity_but_slow.json"
    if v3_receipt.is_file() and not v3_fold_receipt.exists():
        v3 = json.loads(v3_receipt.read_text(encoding="utf-8"))
        _atomic_json(
            v3_fold_receipt,
            {
                "schema": "ddm_f26q_native_performance_rung.v1",
                "complete": True,
                "disposition": "FOLDED_PERFORMANCE_ONLY",
                "verdict_scope": "INSTANCE(single-thread native v3 on M5)",
                "reason": "four-frame token/logit/CDF parity passed but single-thread native wall exceeded Python",
                "receipt": _file_fact(v3_receipt),
                "measured_wall_seconds": v3["wall_seconds"],
                "parity": v3["gates"],
                "next_rung": "same arithmetic with four-thread patch parallelism",
            },
        )
    v4_failure_receipt = RECEIPT_DIR / "native_build_v4_openmp_compile_failure.json"
    if not v4_failure_receipt.exists() and (WORK_DIR / "submission_candidate_v4_openmp_python_default").exists():
        _atomic_json(
            v4_failure_receipt,
            {
                "schema": "ddm_f26q_native_build_failure.v1",
                "complete": False,
                "reason": "OpenMP structured blocks forbid the v4 loop's goto-based invalid-class guard",
                "payload_materialized": False,
                "cure": "record the impossible class as a status and exit after the parallel region",
            },
        )
    v5_failure_receipt = RECEIPT_DIR / "native_build_v5_duplicate_openmp_failure.json"
    if not v5_failure_receipt.exists() and (RETAINED_DIR / "native_build_v5_openmp").exists():
        _atomic_json(
            v5_failure_receipt,
            {
                "schema": "ddm_f26q_native_build_failure.v1",
                "complete": False,
                "reason": "M5 dlopen refused simultaneous PyTorch-bundled and Homebrew libomp runtimes",
                "payload_materialized": False,
                "retained_build_dir": str(RETAINED_DIR / "native_build_v5_openmp"),
                "cure": "link local measurement binary to PyTorch libomp and use one @rpath identity",
            },
        )
    v6_failure_receipt = RECEIPT_DIR / "native_build_v6_nondeterministic_signature_failure.json"
    v6_build_dir = RETAINED_DIR / "native_build_v6_torch_openmp"
    if not v6_failure_receipt.exists() and v6_build_dir.exists():
        retained_v6 = [
            _file_fact(path)
            for path in sorted(v6_build_dir.glob("f26_hpac_native_*.so"))
            if path.is_file()
        ]
        _atomic_json(
            v6_failure_receipt,
            {
                "schema": "ddm_f26q_native_build_failure.v1",
                "complete": False,
                "reason": "install_name_tool ad-hoc signed each loadable dylib with an output-name-dependent identifier",
                "payload_materialized": False,
                "retained_binaries": retained_v6,
                "cure": "remove the output-specific signature and ad-hoc sign both builds with the fixed f26_hpac_native identifier",
            },
        )
    v7_failure_receipt = RECEIPT_DIR / "native_prefix_v7_openmp_shared_index_failure.json"
    v7_output = RETAINED_DIR / "native_tokens_v7_n4.u8"
    if not v7_failure_receipt.exists() and v7_output.is_file():
        reference_v7 = PROFILE_DIR / "reference_prefix_n4.u8"
        changed_per_frame = None
        if reference_v7.is_file() and reference_v7.stat().st_size == v7_output.stat().st_size:
            reference_array = np.fromfile(reference_v7, dtype=np.uint8).reshape(4, -1)
            native_array = np.fromfile(v7_output, dtype=np.uint8).reshape(4, -1)
            changed_per_frame = [
                int(np.count_nonzero(reference_array[index] != native_array[index]))
                for index in range(4)
            ]
        _atomic_json(
            v7_failure_receipt,
            {
                "schema": "ddm_f26q_native_prefix_failure.v1",
                "complete": False,
                "reason": "the OpenMP patch loop shared its nested local_index induction variable across workers",
                "verdict_scope": "INSTANCE(native v7 OpenMP data sharing)",
                "materialized_token_field": _file_fact(v7_output),
                "reference_token_field": None if not reference_v7.is_file() else _file_fact(reference_v7),
                "changed_token_bytes_per_frame": changed_per_frame,
                "retention": "preserved; excluded from parity and timing evidence",
                "cure": "declare local_index private in the OpenMP patch region",
            },
        )
    v8_run_receipt = RECEIPT_DIR / "native_run_v8_n32.json"
    v8_fold_receipt = RECEIPT_DIR / "native_v8_parallel_parity_performance_rung.json"
    if v8_run_receipt.is_file() and not v8_fold_receipt.exists():
        v8 = json.loads(v8_run_receipt.read_text(encoding="utf-8"))
        _atomic_json(
            v8_fold_receipt,
            {
                "schema": "ddm_f26q_native_performance_rung.v1",
                "complete": True,
                "disposition": "FOLDED_PERFORMANCE_ONLY",
                "verdict_scope": "INSTANCE(OpenMP v8 n32 on M5)",
                "reason": "full prefix parity passed but the stable native kernel rate remained above the charter fire-gate budget",
                "receipt": _file_fact(v8_run_receipt),
                "measured_wall_seconds": v8["wall_seconds"],
                "measured_native_kernel_seconds": v8["native_report"]["stage_seconds"]["native_fused_hpac_probability_rc64"],
                "parity": v8["gates"],
                "next_rung": "compile the same generic native source for the actual CPU architecture",
            },
        )
    v9_run_receipt = RECEIPT_DIR / "native_run_v9_n32.json"
    v9_fold_receipt = RECEIPT_DIR / "native_v9_cpu_tuned_performance_rung.json"
    if v9_run_receipt.is_file() and not v9_fold_receipt.exists():
        v9 = json.loads(v9_run_receipt.read_text(encoding="utf-8"))
        _atomic_json(
            v9_fold_receipt,
            {
                "schema": "ddm_f26q_native_performance_rung.v1",
                "complete": True,
                "disposition": "FOLDED_PERFORMANCE_ONLY",
                "verdict_scope": "INSTANCE(cpu-tuned v9 n32 on M5)",
                "reason": "the architecture-native compiler flag produced the same binary and no kernel speed gain on M5",
                "receipt": _file_fact(v9_run_receipt),
                "measured_native_kernel_seconds": v9["native_report"]["stage_seconds"]["native_fused_hpac_probability_rc64"],
                "parity": v9["gates"],
                "next_rung": "incrementally maintain conv-A accumulators as causal symbols become known",
            },
        )
    v10_run_receipt = RECEIPT_DIR / "native_run_v10_n4.json"
    v10_fold_receipt = RECEIPT_DIR / "native_v10_incremental_conv_probe.json"
    if v10_run_receipt.is_file() and not v10_fold_receipt.exists():
        v10 = json.loads(v10_run_receipt.read_text(encoding="utf-8"))
        _atomic_json(
            v10_fold_receipt,
            {
                "schema": "ddm_f26q_native_performance_rung.v1",
                "complete": True,
                "disposition": "FOLDED_PENDING_ELEMENT_PROFILE",
                "verdict_scope": "INSTANCE(incremental conv-A v10 n4 on M5)",
                "reason": "prefix parity passed but the four-frame native kernel wall was unchanged within the observed rung spread",
                "receipt": _file_fact(v10_run_receipt),
                "measured_native_kernel_seconds": v10["native_report"]["stage_seconds"]["native_fused_hpac_probability_rc64"],
                "parity": v10["gates"],
                "next_rung": "measure native initialization, hidden/logit, probability/RC64, and incremental-update elements separately",
            },
        )
    v11_run_receipt = RECEIPT_DIR / "native_run_v11_n32.json"
    v11_profile_receipt = RECEIPT_DIR / "native_v11_element_profile.json"
    if v11_run_receipt.is_file() and not v11_profile_receipt.exists():
        v11 = json.loads(v11_run_receipt.read_text(encoding="utf-8"))
        stages = v11["native_report"]["stage_seconds"]
        _atomic_json(
            v11_profile_receipt,
            {
                "schema": "ddm_f26q_native_element_profile.v1",
                "complete": True,
                "disposition": "CONSUMED_BY_NEXT_RUNG",
                "verdict_scope": "INSTANCE(profiled incremental v11 n32 on M5)",
                "receipt": _file_fact(v11_run_receipt),
                "measured_native_elements_seconds": {
                    "conv_state_initialization": stages["native_conv_state_initialization"],
                    "sparse_hidden_and_logits": stages["native_sparse_hidden_and_logits"],
                    "probability_and_rc64": stages["native_probability_and_rc64"],
                    "incremental_conv_update": stages["native_incremental_conv_update"],
                },
                "finding": "sparse hidden/logit arithmetic dominates; entropy is already subdominant",
                "next_rung": "pack the quantized frame context to int16 once per frame instead of converting it at every hidden activation",
            },
        )
    v12_run_receipt = RECEIPT_DIR / "native_run_v12_n32.json"
    v12_fold_receipt = RECEIPT_DIR / "native_v12_integer_context_rung.json"
    if v12_run_receipt.is_file() and not v12_fold_receipt.exists():
        v12 = json.loads(v12_run_receipt.read_text(encoding="utf-8"))
        _atomic_json(
            v12_fold_receipt,
            {
                "schema": "ddm_f26q_native_performance_rung.v1",
                "complete": True,
                "disposition": "CONSUMED_BY_NEXT_RUNG",
                "verdict_scope": "INSTANCE(integer-context v12 n32 on M5)",
                "reason": "integer context preserved parity and reduced native-kernel time, but allocation and scalar depthwise work remained",
                "receipt": _file_fact(v12_run_receipt),
                "measured_native_kernel_seconds": v12["native_report"]["stage_seconds"]["native_fused_hpac_probability_rc64"],
                "parity": v12["gates"],
                "next_rung": "reuse native workspaces across frames and vectorize the depthwise channel reductions",
            },
        )
    v13_run_receipt = RECEIPT_DIR / "native_run_v13_n32.json"
    v13_fold_receipt = RECEIPT_DIR / "native_v13_persistent_vector_rung.json"
    if v13_run_receipt.is_file() and not v13_fold_receipt.exists():
        v13 = json.loads(v13_run_receipt.read_text(encoding="utf-8"))
        _atomic_json(
            v13_fold_receipt,
            {
                "schema": "ddm_f26q_native_performance_rung.v1",
                "complete": True,
                "disposition": "CONSUMED_BY_NEXT_RUNG",
                "verdict_scope": "INSTANCE(persistent-vector v13 n32 on M5)",
                "reason": "workspace reuse and vectorized depthwise reductions preserved parity and halved wall time, but per-group OpenMP team creation remained",
                "receipt": _file_fact(v13_run_receipt),
                "measured_wall_seconds": v13["wall_seconds"],
                "measured_native_kernel_seconds": v13["native_report"]["stage_seconds"]["native_fused_hpac_probability_rc64"],
                "parity": v13["gates"],
                "next_rung": "hold one OpenMP team across all 190 causal groups in each frame",
            },
        )
    v14_run_receipt = RECEIPT_DIR / "native_run_v14_n32.json"
    v14_fold_receipt = RECEIPT_DIR / "native_v14_persistent_team_dead_end.json"
    if v14_run_receipt.is_file() and not v14_fold_receipt.exists():
        v14 = json.loads(v14_run_receipt.read_text(encoding="utf-8"))
        _atomic_json(
            v14_fold_receipt,
            {
                "schema": "ddm_f26q_native_performance_rung.v1",
                "complete": True,
                "disposition": "FOLDED_PERFORMANCE_ONLY",
                "verdict_scope": "INSTANCE(persistent-team v14 n32 on M5)",
                "reason": "one persistent OpenMP team preserved parity but measured slower than the retained v13 per-group team baseline",
                "receipt": _file_fact(v14_run_receipt),
                "measured_wall_seconds": v14["wall_seconds"],
                "measured_native_kernel_seconds": v14["native_report"]["stage_seconds"]["native_fused_hpac_probability_rc64"],
                "adopted_baseline": _file_fact(v13_run_receipt),
                "next_rung": "revert to v13 and measure the full n600 field",
            },
        )
    receipt_path = RECEIPT_DIR / "prepare_v13.json"
    if CANDIDATE_STAGE.exists():
        if not receipt_path.is_file():
            raise NativeLoweringError("candidate stage exists without a preparation receipt")
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        current = _manifest(CANDIDATE_STAGE)
        if _manifest_sha256(current) != receipt["candidate_stage"]["manifest_sha256"]:
            raise NativeLoweringError("candidate stage differs from its preparation receipt")
        return receipt

    temporary = WORK_DIR / f".submission_candidate.{os.getpid()}.tmp"
    if temporary.exists():
        raise NativeLoweringError(f"refusing stale temporary stage: {temporary}")
    shutil.copytree(SOURCE_STAGE, temporary)
    shutil.copy2(source_c, temporary / "runtime" / "f26_hpac_native.c")
    shutil.copy2(source_py, temporary / "runtime" / "f26_hpac_native.py")

    residual = temporary / "runtime" / "residual_archive.py"
    _replace_exact(
        residual,
        '    """Decode the F26 RC64 stream with the fixed boundary residual model."""\n'
        "    import torch\n\n",
        '    """Decode the F26 RC64 stream with the fixed boundary residual model."""\n'
        '    if os.environ.get("F26_TOKEN_DECODER", "python") == "native-hpac":\n'
        "        from .f26_hpac_native import decode_production_tokens_native\n\n"
        "        return decode_production_tokens_native(parts, runtime, code_dir, device)\n"
        "    import torch\n\n",
    )
    f26 = temporary / "runtime" / "f26_inflate.py"
    _replace_exact(
        f26,
        "    if loaded is None:\n"
        "        tokens, token_report = decode_production_tokens(parts, renderer, renderer_dir, device)\n",
        "    if loaded is None:\n"
        '        if os.environ.get("F26_TOKEN_DECODER", "python") == "native-hpac":\n'
        "            if checkpoint_dir is None:\n"
        '                raise InflationError("native token decode requires checkpoint_dir")\n'
        '            native_progress = checkpoint_dir.resolve() / "native_hpac_progress"\n'
        '            os.environ["F26_NATIVE_CHECKPOINT_DIR"] = str(native_progress / "checkpoints")\n'
        '            os.environ["F26_NATIVE_TOKEN_OUTPUT"] = str(native_progress / "tokens_partial.u8")\n'
        "        tokens, token_report = decode_production_tokens(parts, renderer, renderer_dir, device)\n",
    )
    inflate_sh = temporary / "inflate.sh"
    _replace_exact(
        inflate_sh,
        'export CPR1_RC64_LIBRARY="$BUILD_DIR/rc64_backend.so"\n',
        'export CPR1_RC64_LIBRARY="$BUILD_DIR/rc64_backend.so"\n'
        'case "$(uname -m)" in arm64|aarch64) F26_CPU_FLAG=-mcpu=native ;; *) F26_CPU_FLAG=-march=native ;; esac\n'
        '"${CC:-cc}" -O3 "$F26_CPU_FLAG" -std=c11 -shared -fPIC -ffp-contract=off -fno-fast-math -fopenmp \\\n'
        '  "$HERE/runtime/f26_hpac_native.c" -lm -o "$BUILD_DIR/f26_hpac_native.so"\n'
        'export F26_HPAC_NATIVE_LIBRARY="$BUILD_DIR/f26_hpac_native.so"\n'
        'export F26_TOKEN_DECODER="${F26_TOKEN_DECODER:-python}"\n',
    )
    os.replace(temporary, CANDIDATE_STAGE)
    candidate_manifest = _manifest(CANDIDATE_STAGE)
    receipt = {
        "schema": "ddm_f26q_prepare.v1",
        "complete": True,
        "storage_preflight": storage,
        "source_stage": {
            "root": str(SOURCE_STAGE),
            "archive": _file_fact(archive),
            "manifest_sha256": _manifest_sha256(_manifest(SOURCE_STAGE)),
        },
        "native_sources": [_file_fact(source_c), _file_fact(source_py)],
        "candidate_stage": {
            "root": str(CANDIDATE_STAGE),
            "manifest_sha256": _manifest_sha256(candidate_manifest),
            "files": candidate_manifest,
            "default_token_decoder": "python",
        },
        "transforms": [
            "copy native generic source and Python binding into a new lifted tree",
            "route F26_TOKEN_DECODER=native-hpac to the native path",
            "bind native progress to the lifted CPU checkpoint directory",
            "compile the native binary inside inflate while preserving Python as default",
        ],
    }
    _atomic_json(receipt_path, receipt)
    return receipt


def build(repo_root: Path) -> dict[str, Any]:
    """Compile and retain two deterministic native binaries."""
    prepare(repo_root)
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    source = CANDIDATE_STAGE / "runtime" / "f26_hpac_native.c"
    prior_outputs = [BUILD_DIR / name for name in ("f26_hpac_native_a.so", "f26_hpac_native_b.so")]
    build_receipt_path = RECEIPT_DIR / "native_build_v13.json"
    if any(path.exists() for path in prior_outputs) and not build_receipt_path.is_file():
        failure_index = 1
        while (BUILD_DIR / f"failed_nondeterministic_build_{failure_index}").exists():
            failure_index += 1
        failure_dir = BUILD_DIR / f"failed_nondeterministic_build_{failure_index}"
        failure_dir.mkdir(parents=True, exist_ok=True)
        retained = []
        for path in prior_outputs:
            if not path.exists():
                continue
            destination = failure_dir / path.name
            if destination.exists():
                raise NativeLoweringError(f"refusing to overwrite retained failed build {destination}")
            os.replace(path, destination)
            retained.append(_file_fact(destination))
        _atomic_json(
            RECEIPT_DIR / f"native_build_failed_nondeterministic_{failure_index}.json",
            {
                "schema": "ddm_f26q_native_build_failure.v1",
                "complete": False,
                "reason": "platform linker metadata made two otherwise identical builds byte-different",
                "retained_binaries": retained,
                "cure": "rebuild with deterministic UUID and install-name or SONAME linker metadata",
            },
        )
    commands = []
    outputs = []
    if sys.platform == "darwin":
        import torch

        torch_lib = Path(torch.__file__).resolve().parent / "lib"
        deterministic_linker_flags = [
            "-Xpreprocessor",
            "-fopenmp",
            "-I/opt/homebrew/opt/libomp/include",
            f"-L{torch_lib}",
            "-lomp",
            f"-Wl,-rpath,{torch_lib}",
            "-Wl,-install_name,@rpath/f26_hpac_native.so",
        ]
    else:
        deterministic_linker_flags = ["-fopenmp", "-Wl,-soname,f26_hpac_native.so"]
    for name in ("f26_hpac_native_a.so", "f26_hpac_native_b.so"):
        output = BUILD_DIR / name
        command = [
            os.environ.get("CC", "cc"),
            "-O3",
            "-mcpu=native" if sys.platform == "darwin" else "-march=native",
            "-std=c11",
            "-shared",
            "-fPIC",
            "-ffp-contract=off",
            "-fno-fast-math",
            *deterministic_linker_flags,
            str(source),
            "-lm",
            "-o",
            str(output),
        ]
        subprocess.run(command, check=True, cwd=repo_root)
        commands.append(command)
        if sys.platform == "darwin":
            install_name = [
                "install_name_tool",
                "-change",
                "/opt/llvm-openmp/lib/libomp.dylib",
                "@rpath/libomp.dylib",
                str(output),
            ]
            subprocess.run(install_name, check=True, cwd=repo_root)
            commands.append(install_name)
            remove_signature = ["codesign", "--remove-signature", str(output)]
            subprocess.run(remove_signature, check=True, cwd=repo_root)
            commands.append(remove_signature)
            fixed_signature = [
                "codesign",
                "-s",
                "-",
                "--force",
                "--timestamp=none",
                "-i",
                "f26_hpac_native",
                str(output),
            ]
            subprocess.run(fixed_signature, check=True, cwd=repo_root)
            commands.append(fixed_signature)
        outputs.append(_file_fact(output))
    deterministic = outputs[0]["sha256"] == outputs[1]["sha256"]
    if not deterministic:
        raise NativeLoweringError("repeat native builds are not byte-identical")

    rc64_source = CANDIDATE_STAGE / "runtime" / "entropy" / "rc64_backend.c"
    rc64_output = BUILD_DIR / "rc64_reference.so"
    rc64_command = [
        os.environ.get("CC", "cc"),
        "-O3",
        "-std=c11",
        "-shared",
        "-fPIC",
        str(rc64_source),
        "-o",
        str(rc64_output),
    ]
    subprocess.run(rc64_command, check=True, cwd=repo_root)
    receipt = {
        "schema": "ddm_f26q_native_build.v1",
        "complete": True,
        "compiler": subprocess.run(
            [os.environ.get("CC", "cc"), "--version"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()[0],
        "source": _file_fact(source),
        "deterministic_linker_flags": deterministic_linker_flags,
        "commands": commands,
        "repeat_builds": outputs,
        "deterministic_binary": deterministic,
        "rc64_reference": {"command": rc64_command, "binary": _file_fact(rc64_output)},
        "rebuild_instructions": (
            "cc -O3 -std=c11 -shared -fPIC -ffp-contract=off -fno-fast-math -fopenmp "
            "runtime-rs/native/f26-hpac/f26_hpac_native.c -lm -o f26_hpac_native.so"
        ),
    }
    _atomic_json(build_receipt_path, receipt)
    return receipt


def _load_runtime() -> tuple[Any, Any, Any, Any, Any]:
    import torch

    renderer_dir = CANDIDATE_STAGE / "cpr1"
    sys.path.insert(0, str(CANDIDATE_STAGE))
    sys.path.insert(0, str(renderer_dir))
    spec = importlib.util.spec_from_file_location("_ddm_f26q_renderer", renderer_dir / "inflate.py")
    if spec is None or spec.loader is None:
        raise NativeLoweringError("cannot load F26 renderer")
    renderer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(renderer)
    from runtime.hpac_inference import optimize_sparse_evaluator
    from runtime.ihs2 import materialize_ihs1
    from runtime.residual_archive import _sparse_class, read_residual_archive

    parts = read_residual_archive(CANDIDATE_STAGE / "archive.zip")
    model = renderer.load_hpac(materialize_ihs1(parts.hpac_blob, renderer), torch.device("cpu"))
    sparse = _sparse_class(renderer_dir)(model, renderer.EVAL_H, renderer.EVAL_W)
    optimize_sparse_evaluator(sparse)
    return renderer, parts, model, sparse, torch


def profile_reference(repo_root: Path, frames: int) -> dict[str, Any]:
    """Element-profile the real Python path on a retained archive prefix."""
    build(repo_root)
    if frames <= 0 or frames > 600:
        raise NativeLoweringError("profile frames must be within the real n600 field")
    renderer, parts, model, sparse, torch = _load_runtime()
    from runtime.residual_archive import _boundary_buckets, _probability_table
    from runtime.entropy.rc64 import NativeDecoder

    torch.set_num_threads(4)
    group_plans = []
    for mask in renderer.group_masks(torch.device("cpu")):
        flat = np.flatnonzero(mask.numpy().reshape(-1))
        group_plans.append((torch.from_numpy(flat), flat))
    decoder = NativeDecoder(BUILD_DIR / "rc64_reference.so", parts.token_stream)
    output_path = PROFILE_DIR / f"reference_prefix_n{frames}.u8"
    receipt_path = RECEIPT_DIR / f"reference_profile_n{frames}.json"
    PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    if output_path.exists() or receipt_path.exists():
        raise NativeLoweringError("reference profile output already exists; refusing overwrite")
    tokens = np.memmap(
        output_path,
        mode="w+",
        dtype=np.uint8,
        shape=(frames, renderer.EVAL_H, renderer.EVAL_W),
    )
    previous = torch.zeros((1, renderer.EVAL_H, renderer.EVAL_W), dtype=torch.long)
    corrected_digest = hashlib.sha256()
    cdf_digest = hashlib.sha256()
    timing = {
        "prepare_frame_context": 0.0,
        "boundary_buckets": 0.0,
        "selected_logits": 0.0,
        "correction_and_probability": 0.0,
        "native_rc64_entropy_call": 0.0,
        "scatter_and_frame_copy": 0.0,
    }
    started = time.perf_counter()
    with torch.inference_mode():
        for frame in range(frames):
            tick = time.perf_counter()
            context = model.prepare_frame_context(torch.tensor([frame]), previous)
            timing["prepare_frame_context"] += time.perf_counter() - tick
            tick = time.perf_counter()
            boundary = (
                _boundary_buckets(previous[0].to(torch.uint8).numpy()).reshape(-1)
                if frame
                else np.full(renderer.EVAL_H * renderer.EVAL_W, 4, dtype=np.uint8)
            )
            timing["boundary_buckets"] += time.perf_counter() - tick
            current = torch.zeros_like(previous)
            for group, (device_positions, flat_positions) in enumerate(group_plans):
                tick = time.perf_counter()
                selected = sparse.selected_logits(current, context, group)
                base_logits = selected.numpy()
                timing["selected_logits"] += time.perf_counter() - tick
                tick = time.perf_counter()
                predicted = base_logits.argmax(axis=1).astype(np.int64)
                feature = boundary[flat_positions].astype(np.int64) * 5 + predicted
                corrected = base_logits + parts.table.values[feature]
                corrected_digest.update(memoryview(np.ascontiguousarray(corrected, dtype="<f4")).cast("B"))
                probability = _probability_table(corrected, renderer.HPAC_LOGIT_PRECISION)
                cdf_digest.update(memoryview(np.ascontiguousarray(probability, dtype="<f4")).cast("B"))
                timing["correction_and_probability"] += time.perf_counter() - tick
                tick = time.perf_counter()
                symbols = decoder.decode(probability).astype(np.int64)
                timing["native_rc64_entropy_call"] += time.perf_counter() - tick
                tick = time.perf_counter()
                current.reshape(-1)[device_positions] = torch.from_numpy(symbols)
                timing["scatter_and_frame_copy"] += time.perf_counter() - tick
            tick = time.perf_counter()
            tokens[frame] = current[0].to(torch.uint8).numpy()
            previous = current
            timing["scatter_and_frame_copy"] += time.perf_counter() - tick
    tokens.flush()
    wall = time.perf_counter() - started
    census = json.loads(
        (
            Path("/Volumes/VertigoDataTier/pact/ddm_f26p_runtime_cpu_lift_20260814")
            / "receipts"
            / "runtime_analysis.json"
        ).read_text(encoding="utf-8")
    )["hpac_operation_census"]
    receipt = {
        "schema": "ddm_f26q_reference_profile.v1",
        "complete": True,
        "axis_label": "[M5-CPU 4-thread scorer-free real-stream profile]",
        "scope": {
            "real_archive_prefix_frames": frames,
            "full_field_frames": 600,
            "selection": "contiguous prefix of the real pinned archive; timing is scope-only",
            "verdict_authority": False,
        },
        "wall_seconds": wall,
        "stage_seconds": timing,
        "stage_fraction": {name: value / wall for name, value in timing.items()},
        "measured_prefix": {
            "decoded_tokens": _file_fact(output_path),
            "corrected_quantized_logit_sha256": corrected_digest.hexdigest(),
            "corrected_cdf_input_sha256": cdf_digest.hexdigest(),
            "decoder_bit_position": decoder.bit_position,
        },
        "element_counts": {
            "prefix_group_calls": frames * census["groups_per_frame"],
            "prefix_output_symbols": frames * census["per_frame_output_symbols"],
            "full_field": census,
        },
        "derived_full_field_extrapolation": {
            "wall_seconds_linear_from_prefix": wall * 600 / frames,
            "label": "DERIVED, not a measured full-field wall",
        },
    }
    _atomic_json(receipt_path, receipt)
    return receipt


def run_native(repo_root: Path, frames: int) -> dict[str, Any]:
    """Run the fused native path on a real prefix or the full n600 field."""
    if frames <= 0 or frames > 600:
        raise NativeLoweringError("native frames must be within the real n600 field")
    receipt_path = RECEIPT_DIR / f"native_run_v13_n{frames}.json"
    if receipt_path.is_file():
        retained = json.loads(receipt_path.read_text(encoding="utf-8"))
        output_fact = retained.get("native_report", {})
        output_path = Path(str(output_fact.get("decoded_token_path", "")))
        if not retained.get("complete") or not output_path.is_file():
            raise NativeLoweringError("retained native receipt is incomplete or lost its payload")
        if output_path.stat().st_size != int(output_fact["decoded_token_bytes"]):
            raise NativeLoweringError("retained native token payload changed size")
        if _sha256_file(output_path) != output_fact["decoded_token_sha256"]:
            raise NativeLoweringError("retained native token payload changed SHA-256")
        return retained
    build_receipt = build(repo_root)
    renderer, parts, _model, _sparse, torch = _load_runtime()
    from runtime.f26_hpac_native import decode_native_tokens

    torch.set_num_threads(4)
    output_path = RETAINED_DIR / f"native_tokens_v13_n{frames}.u8"
    checkpoint_dir = RETAINED_DIR / f"native_tokens_v13_n{frames}_checkpoints"
    os.environ["F26_HPAC_NATIVE_LIBRARY"] = str(BUILD_DIR / "f26_hpac_native_a.so")
    started = time.perf_counter()
    _tokens, report = decode_native_tokens(
        parts,
        renderer,
        CANDIDATE_STAGE / "cpr1",
        torch.device("cpu"),
        frame_limit=frames,
        output_path=output_path,
        checkpoint_dir=checkpoint_dir,
    )
    wall = time.perf_counter() - started
    reference_path = PROFILE_DIR / f"reference_prefix_n{frames}.u8"
    if frames < 600 and not reference_path.is_file():
        raise NativeLoweringError(
            f"prefix parity requires retained reference profile {reference_path}"
        )
    parity = {
        "token_sha_match": None,
        "token_bytes_match": None,
        "reference": None,
    }
    if reference_path.is_file():
        parity = {
            "token_sha_match": _sha256_file(reference_path) == _sha256_file(output_path),
            "token_bytes_match": reference_path.read_bytes() == output_path.read_bytes(),
            "reference": _file_fact(reference_path),
        }
        if not parity["token_bytes_match"]:
            raise NativeLoweringError("native prefix tokens differ from Python reference")
    if frames == 600:
        gates = {
            "decoded_token_sha256": report["decoded_token_sha256"] == TOKEN_SHA256,
            "decoded_token_bytes": report["decoded_token_bytes"] == TOKEN_BYTES,
            "decoder_bit_position": report["decoder_bit_position"] == DECODER_BIT_POSITION,
            "corrected_quantized_logit_sha256": (
                report["corrected_quantized_logit_sha256"] == CORRECTED_SHA256
            ),
            "corrected_cdf_input_sha256": report["corrected_cdf_input_sha256"] == CDF_SHA256,
            "digest_scope": report["digest_scope"] == "full_field",
        }
        if not all(gates.values()):
            raise NativeLoweringError(f"full native parity gate failed: {gates}")
    else:
        gates = {"prefix_token_bytes": bool(parity["token_bytes_match"])}
    receipt = {
        "schema": "ddm_f26q_native_run.v1",
        "complete": True,
        "axis_label": "[M5-CPU 4-thread scorer-free native token decode]",
        "frames": frames,
        "wall_seconds": wall,
        "native_report": report,
        "parity": parity,
        "gates": gates,
        "native_build": build_receipt["repeat_builds"][0],
    }
    _atomic_json(receipt_path, receipt)
    return receipt


def seal(repo_root: Path) -> dict[str, Any]:
    """Activate native-by-default only after all full-field identity gates pass."""
    full_path = RECEIPT_DIR / "native_run_v13_n600.json"
    if not full_path.is_file():
        raise NativeLoweringError("seal requires a full n600 native run receipt")
    full = json.loads(full_path.read_text(encoding="utf-8"))
    if not full.get("complete") or not all(full.get("gates", {}).values()):
        raise NativeLoweringError("seal requires every full-field native parity gate")
    prepare(repo_root)
    receipt_path = RECEIPT_DIR / "seal.json"
    if SEALED_STAGE.exists():
        if not receipt_path.is_file():
            raise NativeLoweringError("sealed stage exists without receipt")
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if _manifest_sha256(_manifest(SEALED_STAGE)) != receipt["sealed_stage"]["manifest_sha256"]:
            raise NativeLoweringError("sealed stage differs from its receipt")
        return receipt
    temporary = WORK_DIR / f".submission_native_sealed.{os.getpid()}.tmp"
    shutil.copytree(CANDIDATE_STAGE, temporary)
    _replace_exact(
        temporary / "inflate.sh",
        'export F26_TOKEN_DECODER="${F26_TOKEN_DECODER:-python}"',
        'export F26_TOKEN_DECODER="${F26_TOKEN_DECODER:-native-hpac}"',
    )
    os.replace(temporary, SEALED_STAGE)
    records = _manifest(SEALED_STAGE)
    receipt = {
        "schema": "ddm_f26q_native_seal.v1",
        "complete": True,
        "activation_gate": _file_fact(full_path),
        "archive": _file_fact(SEALED_STAGE / "archive.zip"),
        "sealed_stage": {
            "root": str(SEALED_STAGE),
            "manifest_sha256": _manifest_sha256(records),
            "files": records,
            "default_token_decoder": "native-hpac",
            "fallback": "set F26_TOKEN_DECODER=python explicitly",
        },
    }
    _atomic_json(receipt_path, receipt)
    return receipt


def summarize(repo_root: Path) -> dict[str, Any]:
    """Write the measured M5 wall and the explicitly derived Modal projection."""
    run = json.loads((RECEIPT_DIR / "native_run_v13_n600.json").read_text(encoding="utf-8"))
    if not run.get("complete") or not all(run.get("gates", {}).values()):
        raise NativeLoweringError("summary requires every full-field native parity gate")
    native_m5 = float(run["native_report"]["decode_runtime_seconds"])
    modal_fixed = MODAL_TOTAL_SECONDS - MODAL_TOKEN_SECONDS
    projected_modal_token = native_m5 * MODAL_STAGE_RATIO
    projected_modal_total = modal_fixed + projected_modal_token
    fire_ready = projected_modal_total <= 1600.0
    seal_receipt = seal(repo_root) if fire_ready else None
    prepared = prepare(repo_root)
    result = {
        "schema": "ddm_f26q_result.v1",
        "complete": True,
        "score_claim": False,
        "measured": {
            "axis_label": "[M5-CPU 4-thread scorer-free native token decode]",
            "native_token_stage_seconds": native_m5,
            "reference_token_stage_seconds": M5_REFERENCE_TOKEN_SECONDS,
            "native_speedup": M5_REFERENCE_TOKEN_SECONDS / native_m5,
            "decoded_tokens": _file_fact(RETAINED_DIR / "native_tokens_v13_n600.u8"),
            "decoder_bit_position": run["native_report"]["decoder_bit_position"],
        },
        "derived_modal_projection": {
            "label": "DERIVED from the measured 6.8x F26 stage ratio; not a Modal measurement",
            "measured_stage_ratio": MODAL_STAGE_RATIO,
            "projected_token_seconds": projected_modal_token,
            "fixed_non_token_seconds_from_failed_modal_run": modal_fixed,
            "projected_total_seconds": projected_modal_total,
            "fire_threshold_seconds": 1600.0,
            "verdict": "SEALED_FIRE_ORDER" if fire_ready else "DECODE_ENGINEERING_GATED",
        },
        "full_field_parity": run["gates"],
        "measured_native_stage_seconds": run["native_report"]["stage_seconds"],
        "sealed_stage": None if seal_receipt is None else seal_receipt["sealed_stage"],
        "unsealed_candidate_stage": prepared["candidate_stage"],
        "decode_engineering_residual": None
        if fire_ready
        else {
            "disposition": "QUEUED_WITH_A_FIRE_ORDER",
            "verdict_scope": "INSTANCE(v13 native F26 on M5 plus measured Modal stage ratio)",
            "remaining_measured_m5_seconds_to_fire_gate": native_m5
            - (1600.0 - modal_fixed) / MODAL_STAGE_RATIO,
            "dominant_element": "native_sparse_hidden_and_logits",
            "owner": "MAIN or successor native-runtime arm",
            "consumer_store": ".omx/research/ddm_f26q_rc64_native_lowering_20260814.md",
            "fire_trigger": "a new full-n600 byte-identical native token receipt projects Modal total at or below 1600 seconds",
        },
        "fire_ready": fire_ready,
        "exact_row_fired": False,
        "frontier_moved": False,
    }
    _atomic_json(RECEIPT_DIR / "result.json", result)
    return result


def run_command(repo_root: Path, command: str, frames: int) -> dict[str, Any]:
    if command == "prepare":
        return prepare(repo_root)
    if command == "build":
        return build(repo_root)
    if command == "profile-reference":
        return profile_reference(repo_root, frames)
    if command == "run-native":
        return run_native(repo_root, frames)
    if command == "seal":
        return seal(repo_root)
    if command in {"summarize", "all"}:
        return summarize(repo_root)
    raise NativeLoweringError(f"unsupported command: {command}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=("prepare", "build", "profile-reference", "run-native", "seal", "summarize", "all"),
    )
    parser.add_argument("--frames", type=int, default=600)
    args = parser.parse_args()
    for name, value in THREAD_ENV.items():
        os.environ[name] = value
    result = run_command(Path(__file__).resolve().parents[1], args.command, args.frames)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
