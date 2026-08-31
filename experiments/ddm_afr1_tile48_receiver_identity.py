#!/usr/bin/env python3
"""Port AFC1's tile48 x groupbin8 interaction to the native receiver.

The stages are intentionally ordered and fail closed:

1. re-derive AFC1's retained -81 B full-n600 control;
2. stage the exact candidate archive and add only the generation-23 C rule;
3. compare the Python authority and the C function at all 196,608 positions;
4. run one retained full-n600 receiver identity on CPU;
5. retain a complete manifest and a typed MAIN fire order.

No scorer and no remote dispatch are used by this apparatus.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments.ddm_fcd1_field_for_coder_diagonal import (
    NATIVE_CORRECTOR_BUILD,
    PYTHON_CORRECTOR_SELECTION,
    stage_runtime,
)
from tac.candidate_seal import CONSISTENT, check_pin_consistency, repin_receiver

VERTIGO_ROOT = Path("/Volumes/VertigoDataTier/pact")
AP_ROOT = Path("/Volumes/APDataStore/pact")
STORE = VERTIGO_ROOT / "ddm_afr1_tile48_receiver_identity"
MEASUREMENT = STORE / "measurement_v1"
CONTROL_RETAINED = STORE / "retained" / "afc1_control"
BUILD_ROOT = STORE / "retained" / "build"
PARITY_ROOT = STORE / "retained" / "parity"
RUNTIME = STORE / "runtime_candidate_native"
IDENTITY = STORE / "identity_v1"
IDENTITY_DATA = IDENTITY / "data"
IDENTITY_OUT = IDENTITY / "out"

AFC1_ROOT = AP_ROOT / "ddm_afc1_address_free_census" / "tile48_groupbin8"
AFC1_MEASUREMENT = AFC1_ROOT / "measurement_v1"
AFC1_RUN = AFC1_ROOT / "physical_v1"
AFC1_ADJUDICATION = AFC1_MEASUREMENT / "ADJUDICATION.json"
AFC1_MANIFEST = AFC1_MEASUREMENT / "MANIFEST.json"
AFC1_PATCHED_PYTHON = AFC1_ROOT / "retained" / "runtime_surface" / "fx2_model_axis_corrector.py"
AFC1_CANDIDATE_ARCHIVE = AFC1_RUN / "retained" / "candidate_afc1_tile48_groupbin8.zip"
AFC1_REPEAT_ARCHIVE = AFC1_RUN / "retained" / "candidate_afc1_tile48_groupbin8_repeat.zip"
AFC1_CANDIDATE_STREAM = AFC1_RUN / "work" / "tail_afc1_tile48_groupbin8.bin"
AFC1_REPEAT_STREAM = AFC1_RUN / "work" / "tail_afc1_tile48_groupbin8_repeat.bin"
AFC1_CONTROL_STREAM = AFC1_RUN / "work" / "tail_control_600.bin"

LB1_ROOT = AP_ROOT / "ddm_lb1_banked_lossless_joint_collect"
LB1_RUNTIME = LB1_ROOT / "runtime_candidate_native"
LB1_NATIVE_SOURCE = LB1_RUNTIME / "runtime" / "f26_corrector_native.c"
LB1_ARCHIVE = LB1_ROOT / "retained" / "candidate_lb1_joint22_patch192.zip"
LB1_IDENTITY_RECEIPT = LB1_ROOT / "measurement_v1" / "NATIVE_IDENTITY.json"
LB1_REFERENCE_RAW = LB1_ROOT / "identity_native" / "out" / "0.raw"
LB1_CUDA_RESULT = LB1_ROOT / "fire_cuda" / "MODAL_REMOTE_RESULT.json"
PYTHON_REFERENCE_ROOT = AP_ROOT / "ddm_jg5" / "advisory_final" / "work"
PYTHON_REFERENCE_RAW = PYTHON_REFERENCE_ROOT / "inflated" / "0.raw"
PYTHON_REFERENCE_MANIFEST = PYTHON_REFERENCE_ROOT / "inflated_outputs_manifest.json"

PUBLIC_FILE_LIST = REPO / "upstream" / "public_test_video_names.txt"

WIDTH = 512
HEIGHT = 384
PAIR_COUNT = 600
TOKEN_BYTES = 117_964_800
RAW_BYTES = 3_662_409_600
RESERVE_BYTES = 4 << 30
SCRATCH_BYTES = 64 << 20
IDENTITY_REQUIRED_FREE_BYTES = RAW_BYTES + TOKEN_BYTES + RESERVE_BYTES + SCRATCH_BYTES

LB1_ARCHIVE_BYTES = 180_083
LB1_ARCHIVE_SHA256 = "5b856e667961dd9ab68ddd7166384662bfb5912fabc8c9270098ea63a8ad28c9"
LB1_STREAM_BYTES = 113_492
LB1_STREAM_SHA256 = "8838e44f6498cd9b94f480ae04d9ea12d89b7020ff3c6f215ff83de177a3eac2"
CANDIDATE_ARCHIVE_BYTES = 180_002
CANDIDATE_ARCHIVE_SHA256 = "cbb8d928a8ccdd3f5103da1d4a8d38d0662a5e5615266b923b5f8350d405bf25"
CANDIDATE_STREAM_BYTES = 113_411
CANDIDATE_STREAM_SHA256 = "5601d6fd792c60c176e7cb7478e6033c4ed9a7e87404582340ed3f50ed60cfe3"
TOKEN_SHA256 = "cc10a7b09353c0af1ebe4e52a1640df1fadac4d245a27f41aff8cf0992636efb"
PYTHON_REFERENCE_RAW_SHA256 = "7246a4ff8f79b03ab14b3a72f6a6e2fff18b567fcb61f12a7fe311d48f5f2de7"
AFC1_ADJUDICATION_SHA256 = "9bda316e278e6bf37e762c6c1308cc014db2f76703ce327eef0bad064b6ed841"
AFC1_MANIFEST_SHA256 = "1e8a111e8f5d010d67ac34e212a81370341743c4e9ab148c14b2ceb22425a425"

AXIS = "[macOS-CPU full receiver identity / no score claim]"


class Afr1Error(RuntimeError):
    """A custody, storage, port, parity, identity, or retention gate refused."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_fact(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise Afr1Error(f"required file is absent: {path}")
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def require_fact(path: Path, expected_bytes: int, expected_sha256: str) -> dict[str, Any]:
    observed = file_fact(path)
    if observed["bytes"] != expected_bytes or observed["sha256"] != expected_sha256:
        raise Afr1Error(
            f"pin mismatch at {path}: {observed['bytes']}/{observed['sha256']} != "
            f"{expected_bytes}/{expected_sha256}"
        )
    return observed


def require_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise Afr1Error(f"expected a JSON object at {path}")
    return value


def copy_retained(source: Path, destination: Path) -> dict[str, Any]:
    source_fact = file_fact(source)
    if destination.exists():
        destination_fact = file_fact(destination)
        if (
            destination_fact["bytes"] != source_fact["bytes"]
            or destination_fact["sha256"] != source_fact["sha256"]
        ):
            raise Afr1Error(f"retained copy differs from source: {destination}")
        return destination_fact
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.partial")
    shutil.copy2(source, temporary)
    os.replace(temporary, destination)
    destination_fact = file_fact(destination)
    if destination_fact["sha256"] != source_fact["sha256"]:
        raise Afr1Error(f"retained copy failed byte identity: {destination}")
    return destination_fact


def require_control() -> dict[str, Any]:
    path = MEASUREMENT / "CONTROL_REPRODUCTION.json"
    if not path.is_file():
        raise Afr1Error("run the AFC1 control reproduction before the native port")
    control = require_json(path)
    if (
        control.get("status") != "PASS"
        or control.get("archive_delta_bytes") != -81
        or control.get("token_stream_delta_bytes") != -81
        or control.get("changed_decoded_symbols") != 0
    ):
        raise Afr1Error("the retained AFC1 control no longer proves the -81 B identity row")
    return control


def stage_control() -> dict[str, Any]:
    """Re-derive AFC1's retained physical row before touching the native port."""
    STORE.mkdir(parents=True, exist_ok=True)
    free_before = shutil.disk_usage(STORE).free
    if free_before < IDENTITY_REQUIRED_FREE_BYTES:
        raise Afr1Error(
            f"Vertigo identity floor failed before control: {free_before} < "
            f"{IDENTITY_REQUIRED_FREE_BYTES} B"
        )
    adjudication_fact = require_fact(
        AFC1_ADJUDICATION,
        AFC1_ADJUDICATION.stat().st_size,
        AFC1_ADJUDICATION_SHA256,
    )
    manifest_fact = require_fact(
        AFC1_MANIFEST,
        AFC1_MANIFEST.stat().st_size,
        AFC1_MANIFEST_SHA256,
    )
    adjudication = require_json(AFC1_ADJUDICATION)
    base_archive = require_fact(LB1_ARCHIVE, LB1_ARCHIVE_BYTES, LB1_ARCHIVE_SHA256)
    candidate_archive = require_fact(
        AFC1_CANDIDATE_ARCHIVE,
        CANDIDATE_ARCHIVE_BYTES,
        CANDIDATE_ARCHIVE_SHA256,
    )
    repeat_archive = require_fact(
        AFC1_REPEAT_ARCHIVE,
        CANDIDATE_ARCHIVE_BYTES,
        CANDIDATE_ARCHIVE_SHA256,
    )
    candidate_stream = require_fact(
        AFC1_CANDIDATE_STREAM,
        CANDIDATE_STREAM_BYTES,
        CANDIDATE_STREAM_SHA256,
    )
    repeat_stream = require_fact(
        AFC1_REPEAT_STREAM,
        CANDIDATE_STREAM_BYTES,
        CANDIDATE_STREAM_SHA256,
    )
    control_stream = require_fact(
        AFC1_CONTROL_STREAM,
        LB1_STREAM_BYTES,
        LB1_STREAM_SHA256,
    )

    archive_delta = int(candidate_archive["bytes"]) - int(base_archive["bytes"])
    stream_delta = int(candidate_stream["bytes"]) - int(control_stream["bytes"])
    checks = {
        "adjudication_delta": adjudication.get("archive_saving_bytes") == 81,
        "candidate_repeat_archive_identity": candidate_archive["sha256"] == repeat_archive["sha256"],
        "candidate_repeat_stream_identity": candidate_stream["sha256"] == repeat_stream["sha256"],
        "archive_delta_is_minus_81": archive_delta == -81,
        "stream_delta_is_minus_81": stream_delta == -81,
        "tokens_unchanged": adjudication.get("tokens_changed") == 0,
        "deterministic_repeat": adjudication.get("deterministic_repeat") is True,
    }
    if not all(checks.values()):
        raise Afr1Error(f"AFC1 control reproduction failed: {checks}")

    retained_sources = (
        AFC1_ADJUDICATION,
        AFC1_MANIFEST,
        AFC1_CANDIDATE_ARCHIVE,
        AFC1_REPEAT_ARCHIVE,
        AFC1_CANDIDATE_STREAM,
        AFC1_REPEAT_STREAM,
        AFC1_CONTROL_STREAM,
        AFC1_RUN / "retained" / "S1_control_600.json",
        AFC1_RUN / "retained" / "S1_encode_afc1_tile48_groupbin8.json",
        AFC1_RUN / "retained" / "S1_encode_afc1_tile48_groupbin8_repeat.json",
        AFC1_RUN / "work" / "bits_per_frame_control_600.npy",
        AFC1_RUN / "work" / "bits_per_frame_afc1_tile48_groupbin8.npy",
        AFC1_RUN / "work" / "bits_per_frame_afc1_tile48_groupbin8_repeat.npy",
        AFC1_RUN / "work" / "encode_control_600.checkpoint.npz",
        AFC1_RUN / "work" / "encode_afc1_tile48_groupbin8.checkpoint.npz",
        AFC1_RUN / "work" / "encode_afc1_tile48_groupbin8_repeat.checkpoint.npz",
        AFC1_RUN / "work" / "encode_control_600.encoder.bin",
        AFC1_RUN / "work" / "encode_afc1_tile48_groupbin8.encoder.bin",
        AFC1_RUN / "work" / "encode_afc1_tile48_groupbin8_repeat.encoder.bin",
    )
    retained = [
        copy_retained(source, CONTROL_RETAINED / source.name) for source in retained_sources
    ]
    free_after = shutil.disk_usage(STORE).free
    payload = {
        "schema": "ddm_afr1_afc1_control_reproduction.v1",
        "axis": "[macOS-CPU scorer-free retained physical byte control]",
        "score_claim": False,
        "status": "PASS",
        "source_receipts": {
            "adjudication": adjudication_fact,
            "manifest": manifest_fact,
        },
        "base_archive": base_archive,
        "candidate_archive": candidate_archive,
        "repeat_archive": repeat_archive,
        "control_stream": control_stream,
        "candidate_stream": candidate_stream,
        "repeat_stream": repeat_stream,
        "archive_delta_bytes": archive_delta,
        "token_stream_delta_bytes": stream_delta,
        "delta_S_rate": archive_delta * 25.0 / 37_545_489.0,
        "changed_decoded_symbols": 0,
        "full_n600_tokens": TOKEN_BYTES,
        "checks": checks,
        "retained_copies": retained,
        "storage": {
            "root": str(STORE),
            "free_before_bytes": free_before,
            "free_after_bytes": free_after,
            "identity_floor_bytes": IDENTITY_REQUIRED_FREE_BYTES,
            "status": "PASS" if free_after >= IDENTITY_REQUIRED_FREE_BYTES else "THIN_AFTER_CONTROL",
        },
    }
    atomic_json(MEASUREMENT / "CONTROL_REPRODUCTION.json", payload)
    return payload


def replace_once(text: str, before: str, after: str, *, label: str) -> str:
    count = text.count(before)
    if count != 1:
        raise Afr1Error(f"native port anchor {label!r} occurs {count} times")
    return text.replace(before, after, 1)


def patch_native_source(c_path: Path) -> dict[str, Any]:
    before = file_fact(c_path)
    text = c_path.read_text(encoding="utf-8")
    if "RULE_TILE48_GROUPBIN8" not in text:
        replacements = (
            (
                "#define N_FAMILIES 22 /* ddm_lb1: +cls_groupbin8 +patch192_only */",
                "#define N_FAMILIES 23 /* ddm_afr1: +tile48_groupbin8 */",
                "family count",
                1,
            ),
            (
                "    RULE_PATCH192_ONLY            /* patch192_only, ddm_lb1 */\n};",
                "    RULE_PATCH192_ONLY,           /* patch192_only, ddm_lb1 */\n"
                "    RULE_TILE48_GROUPBIN8          /* tile48_groupbin8, ddm_afr1 */\n};",
                "rule enum and family rule table",
                2,
            ),
            (
                "    192                                                          /* ddm_lb1 */\n};",
                "    192,                                                         /* ddm_lb1 */\n"
                "    48 * GROUP_BINS                                               /* 384, ddm_afr1 */\n};",
                "family size table",
                1,
            ),
            (
                "    /* ddm_gb1 + ddm_jt21 + ddm_lb1 */\n    0, 0, 0\n};",
                "    /* ddm_gb1 + ddm_jt21 + ddm_lb1 + ddm_afr1 */\n    0, 0, 0, 0\n};",
                "count limits and initial weights",
                2,
            ),
            (
                "                                        int64_t groupbin8, int64_t patch192)",
                "                                        int64_t groupbin8, int64_t patch192,\n"
                "                                        int64_t tile48_groupbin8)",
                "rule arguments",
                1,
            ),
            (
                "    case RULE_PATCH192_ONLY:\n        return patch192;\n    default:",
                "    case RULE_PATCH192_ONLY:\n        return patch192;\n"
                "    case RULE_TILE48_GROUPBIN8:\n        return tile48_groupbin8;\n    default:",
                "rule body",
                1,
            ),
            (
                "int f26_corrector_group_state(void *handle, const float *probability,",
                "int64_t f26_tile48_groupbin8_context(int64_t x, int64_t y)\n"
                "{\n"
                "    if (x < 0 || x >= WIDTH || y < 0 || y >= HEIGHT) return -1;\n"
                "    int64_t tile48 = (y / 64) * (WIDTH / 64) + (x / 64);\n"
                "    int64_t groupbin8 = (((x % 64) + 2 * (y % 64)) * GROUP_BINS) / 190;\n"
                "    return tile48 * GROUP_BINS + groupbin8;\n"
                "}\n\n"
                "int f26_corrector_group_state(void *handle, const float *probability,",
                "parity function",
                1,
            ),
            (
                "        int64_t patch192 = (y / 32) * (WIDTH / 32) + (x / 32);\n"
                "        for (int pos = 0; pos < N_FAMILIES; ++pos) {",
                "        int64_t patch192 = (y / 32) * (WIDTH / 32) + (x / 32);\n"
                "        int64_t tile48_groupbin8 = f26_tile48_groupbin8_context(x, y);\n"
                "        for (int pos = 0; pos < N_FAMILIES; ++pos) {",
                "context computation",
                1,
            ),
            (
                "                                  patch192);",
                "                                  patch192, tile48_groupbin8);",
                "rule call",
                1,
            ),
        )
        for old, new, label, expected_count in replacements:
            observed_count = text.count(old)
            if observed_count != expected_count:
                raise Afr1Error(
                    f"native port anchor {label!r} occurs {observed_count} times; "
                    f"expected {expected_count}"
                )
            text = text.replace(old, new)
        temporary = c_path.with_name(f".{c_path.name}.{os.getpid()}.partial")
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, c_path)
    required = (
        "#define N_FAMILIES 23",
        "RULE_TILE48_GROUPBIN8",
        "f26_tile48_groupbin8_context",
        "48 * GROUP_BINS",
    )
    if not all(item in c_path.read_text(encoding="utf-8") for item in required):
        raise Afr1Error("generation-23 C source is incomplete")
    return {"before": before, "after": file_fact(c_path), "added_rule": "tile48_groupbin8"}


def patch_python_gate(py_path: Path) -> dict[str, Any]:
    before = file_fact(py_path)
    text = py_path.read_text(encoding="utf-8")
    if '        "tile48_groupbin8",' not in text:
        text = replace_once(
            text,
            '        "patch192_only",\n    ),',
            '        "patch192_only",\n        "tile48_groupbin8",\n    ),',
            label="expected Python family order",
        )
        temporary = py_path.with_name(f".{py_path.name}.{os.getpid()}.partial")
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, py_path)
    return {"before": before, "after": file_fact(py_path)}


def patch_native_selection(inflate_path: Path) -> dict[str, Any]:
    before = file_fact(inflate_path)
    text = inflate_path.read_text(encoding="utf-8")
    changed = False
    if PYTHON_CORRECTOR_SELECTION in text:
        text = replace_once(
            text,
            PYTHON_CORRECTOR_SELECTION,
            NATIVE_CORRECTOR_BUILD,
            label="native corrector selection",
        )
        temporary = inflate_path.with_name(f".{inflate_path.name}.{os.getpid()}.partial")
        temporary.write_text(text, encoding="utf-8")
        temporary.chmod(inflate_path.stat().st_mode)
        os.replace(temporary, inflate_path)
        changed = True
    if NATIVE_CORRECTOR_BUILD not in inflate_path.read_text(encoding="utf-8"):
        raise Afr1Error("staged inflate.sh does not select the native corrector")
    return {"before": before, "after": file_fact(inflate_path), "changed": changed}


def compile_native(c_path: Path, output: Path, log_stem: str) -> dict[str, Any]:
    output.parent.mkdir(parents=True, exist_ok=True)
    argv = [
        os.environ.get("CC", "cc"),
        "-O3",
        "-std=c11",
        "-shared",
        "-fPIC",
        "-ffp-contract=off",
        "-fno-fast-math",
        "-Wall",
        "-Wextra",
    ]
    if platform.system() == "Darwin":
        argv.append("-Wl,-install_name,f26_corrector_native_afr1.dylib")
    argv.extend((str(c_path), "-lm", "-o", str(output)))
    run = subprocess.run(argv, capture_output=True, text=True, check=False)
    atomic_bytes(BUILD_ROOT / f"{log_stem}.stdout.log", run.stdout.encode())
    atomic_bytes(BUILD_ROOT / f"{log_stem}.stderr.log", run.stderr.encode())
    if run.returncode != 0 or run.stderr.strip():
        raise Afr1Error(f"native compile refused rc={run.returncode}: {run.stderr}")
    return {"argv": argv, "returncode": run.returncode, "library": file_fact(output)}


def stage_port() -> dict[str, Any]:
    control = require_control()
    receipt_path = MEASUREMENT / "NATIVE_PORT.json"
    if receipt_path.is_file():
        prior = require_json(receipt_path)
        source = file_fact(RUNTIME / "runtime" / "f26_corrector_native.c")
        parent = file_fact(LB1_NATIVE_SOURCE)
        build_a = file_fact(
            BUILD_ROOT / "loadable_a" / "f26_corrector_native_afr1.dylib"
        )
        build_b = file_fact(
            BUILD_ROOT / "loadable_b" / "f26_corrector_native_afr1.dylib"
        )
        verdict = check_pin_consistency(RUNTIME)
        if (
            prior.get("status") != "PASS"
            or source["sha256"]
            != prior.get("c_port", {}).get("after", {}).get("sha256")
            or source["sha256"] == parent["sha256"]
            or build_a["sha256"]
            != prior.get("build_a", {}).get("library", {}).get("sha256")
            or build_b["sha256"]
            != prior.get("build_b", {}).get("library", {}).get("sha256")
            or build_a["sha256"] != build_b["sha256"]
            or verdict.verdict != CONSISTENT
        ):
            raise Afr1Error("existing native port no longer passes its custody gates")
        prior["parent_generation22_source"] = parent
        prior["source_changed_vs_parent"] = True
        atomic_json(receipt_path, prior)
        return prior
    free_before = shutil.disk_usage(STORE).free
    candidate = require_fact(
        AFC1_CANDIDATE_ARCHIVE,
        CANDIDATE_ARCHIVE_BYTES,
        CANDIDATE_ARCHIVE_SHA256,
    )
    staged = stage_runtime(LB1_RUNTIME, AFC1_CANDIDATE_ARCHIVE, RUNTIME)

    python_target = RUNTIME / "runtime" / "fx2_model_axis_corrector.py"
    python_before = file_fact(python_target)
    python_temporary = python_target.with_name(
        f".{python_target.name}.{os.getpid()}.partial"
    )
    shutil.copy2(AFC1_PATCHED_PYTHON, python_temporary)
    os.replace(python_temporary, python_target)
    python_copy = {"before": python_before, "after": file_fact(python_target)}
    if python_copy["after"]["sha256"] != sha256_file(AFC1_PATCHED_PYTHON):
        raise Afr1Error("staged Python authority is not byte-identical to AFC1")
    c_port = patch_native_source(RUNTIME / "runtime" / "f26_corrector_native.c")
    py_gate = patch_python_gate(RUNTIME / "runtime" / "native_free_corrector.py")
    selection = patch_native_selection(RUNTIME / "inflate.sh")
    repin = repin_receiver(RUNTIME)
    verdict = check_pin_consistency(RUNTIME)
    if verdict.verdict != CONSISTENT:
        raise Afr1Error(f"candidate runtime pin refusal: {verdict.summary()}")

    build_a = compile_native(
        RUNTIME / "runtime" / "f26_corrector_native.c",
        BUILD_ROOT / "loadable_a" / "f26_corrector_native_afr1.dylib",
        "loadable_a/build",
    )
    build_b = compile_native(
        RUNTIME / "runtime" / "f26_corrector_native.c",
        BUILD_ROOT / "loadable_b" / "f26_corrector_native_afr1.dylib",
        "loadable_b/build",
    )
    deterministic_build = build_a["library"]["sha256"] == build_b["library"]["sha256"]
    if not deterministic_build:
        raise Afr1Error("repeat native builds are not byte-identical")

    config_probe = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                f"sys.path.insert(0,{str(RUNTIME)!r}); "
                "from runtime.native_free_corrector import assert_config_matches; "
                "assert_config_matches(); print('CONFIG_MATCH')"
            ),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if config_probe.returncode != 0 or config_probe.stdout.strip() != "CONFIG_MATCH":
        raise Afr1Error(f"native config gate refused:\n{config_probe.stdout}\n{config_probe.stderr}")

    payload = {
        "schema": "ddm_afr1_native_port.v1",
        "axis": "[macOS-CPU native corrector build / no score claim]",
        "score_claim": False,
        "status": "PASS",
        "control_receipt_sha256": sha256_file(MEASUREMENT / "CONTROL_REPRODUCTION.json"),
        "control_archive_delta_bytes": control["archive_delta_bytes"],
        "candidate_archive": candidate,
        "staged_runtime": staged,
        "python_authority_source": python_copy,
        "parent_generation22_source": file_fact(LB1_NATIVE_SOURCE),
        "source_changed_vs_parent": c_port["after"]["sha256"]
        != sha256_file(LB1_NATIVE_SOURCE),
        "c_port": c_port,
        "python_config_gate": py_gate,
        "inflate_selection": selection,
        "repin_changed": repin.changed,
        "pin_consistency": verdict.verdict,
        "config_gate": "PASS",
        "build_a": build_a,
        "build_b": build_b,
        "deterministic_repeat_build": deterministic_build,
        "scope": "only tile48_groupbin8 was added to the generation-22 native corrector",
        "storage": {
            "free_before_bytes": free_before,
            "free_after_bytes": shutil.disk_usage(STORE).free,
        },
    }
    atomic_json(receipt_path, payload)
    return payload


def stage_parity() -> dict[str, Any]:
    require_control()
    port = require_json(MEASUREMENT / "NATIVE_PORT.json")
    if port.get("status") != "PASS" or port.get("config_gate") != "PASS":
        raise Afr1Error("native port has not passed its configuration gate")

    runtime_text = str(RUNTIME)
    sys.path.insert(0, runtime_text)
    try:
        from runtime.fx2_model_axis_corrector import GROUP_BINS, fx2_family_specs

        specs = fx2_family_specs()
        cells, reference_function = specs["tile48_groupbin8"]
        if int(cells) != 384:
            raise Afr1Error(f"Python authority reports {cells} cells, expected 384")
        y, x = np.indices((HEIGHT, WIDTH), dtype=np.int64)
        features = {
            "tile48": (y // 64) * (WIDTH // 64) + x // 64,
            "groupbin8": (((x % 64) + 2 * (y % 64)) * int(GROUP_BINS)) // 190,
        }
        python_contexts = np.asarray(reference_function(features), dtype=np.int64)
    finally:
        sys.path.pop(0)

    library_path = BUILD_ROOT / "loadable_a" / "f26_corrector_native_afr1.dylib"
    library = ctypes.CDLL(str(library_path))
    function = library.f26_tile48_groupbin8_context
    function.argtypes = (ctypes.c_int64, ctypes.c_int64)
    function.restype = ctypes.c_int64
    native_contexts = np.fromiter(
        (function(column, row) for row in range(HEIGHT) for column in range(WIDTH)),
        dtype=np.int64,
        count=HEIGHT * WIDTH,
    ).reshape(HEIGHT, WIDTH)
    mismatch = python_contexts != native_contexts
    mismatch_count = int(np.count_nonzero(mismatch))

    python_i16 = np.ascontiguousarray(python_contexts, dtype="<i2")
    native_i16 = np.ascontiguousarray(native_contexts, dtype="<i2")
    atomic_bytes(PARITY_ROOT / "python_contexts.i16le", python_i16.tobytes())
    atomic_bytes(PARITY_ROOT / "native_contexts.i16le", native_i16.tobytes())
    if mismatch_count:
        coords = np.argwhere(mismatch)
        atomic_bytes(
            PARITY_ROOT / "mismatch_coordinates.i32le",
            np.ascontiguousarray(coords, dtype="<i4").tobytes(),
        )
        raise Afr1Error(f"Python/C tile48_groupbin8 parity failed at {mismatch_count} positions")
    if int(python_contexts.min()) != 0 or int(python_contexts.max()) != 383:
        raise Afr1Error("tile48_groupbin8 contexts do not span exactly [0, 383]")

    payload = {
        "schema": "ddm_afr1_python_c_configuration_parity.v1",
        "axis": "[macOS-CPU exhaustive context parity / no score claim]",
        "score_claim": False,
        "status": "PASS",
        "python_authority": file_fact(RUNTIME / "runtime" / "fx2_model_axis_corrector.py"),
        "native_source": file_fact(RUNTIME / "runtime" / "f26_corrector_native.c"),
        "native_library": file_fact(library_path),
        "denominator_positions": HEIGHT * WIDTH,
        "mismatch_positions": mismatch_count,
        "parity_fraction": 1.0,
        "context_min": int(python_contexts.min()),
        "context_max": int(python_contexts.max()),
        "context_cells": int(np.unique(python_contexts).size),
        "python_payload": file_fact(PARITY_ROOT / "python_contexts.i16le"),
        "native_payload": file_fact(PARITY_ROOT / "native_contexts.i16le"),
        "expression": (
            "tile48=((y//64)*8+(x//64)); "
            "groupbin8=(((x%64)+2*(y%64))*8)//190; context=tile48*8+groupbin8"
        ),
    }
    atomic_json(MEASUREMENT / "CONFIGURATION_PARITY.json", payload)
    return payload


def stage_identity_preflight() -> dict[str, Any]:
    require_control()
    parity = require_json(MEASUREMENT / "CONFIGURATION_PARITY.json")
    if parity.get("status") != "PASS" or parity.get("mismatch_positions") != 0:
        raise Afr1Error("full identity requires a parity-clean native build")
    free = shutil.disk_usage(STORE).free
    enough = free >= IDENTITY_REQUIRED_FREE_BYTES
    payload = {
        "schema": "ddm_afr1_identity_preflight.v1",
        "axis": "[Vertigo storage/full receiver preflight / no score claim]",
        "score_claim": False,
        "candidate_archive": require_fact(
            RUNTIME / "archive.zip",
            CANDIDATE_ARCHIVE_BYTES,
            CANDIDATE_ARCHIVE_SHA256,
        ),
        "storage": {
            "root": str(STORE),
            "observed_free_bytes": free,
            "required_free_bytes": IDENTITY_REQUIRED_FREE_BYTES,
            "shortfall_bytes": max(IDENTITY_REQUIRED_FREE_BYTES - free, 0),
            "raw_payload_bytes": RAW_BYTES,
            "token_checkpoint_bytes": TOKEN_BYTES,
            "scratch_bytes": SCRATCH_BYTES,
            "post_run_reserve_bytes": RESERVE_BYTES,
            "status": "PASS" if enough else "BLOCKED_STORAGE",
        },
        "disposition": "FIRE_ONE_FULL_N600_IDENTITY" if enough else "BLOCKED_VERTIGO_SHORTFALL",
    }
    atomic_json(MEASUREMENT / "IDENTITY_PREFLIGHT.json", payload)
    if not enough:
        raise Afr1Error(
            f"Vertigo identity floor failed: {free} < {IDENTITY_REQUIRED_FREE_BYTES} B"
        )
    return payload


def stage_identity() -> dict[str, Any]:
    preflight = stage_identity_preflight()
    output = IDENTITY_OUT / "0.raw"
    receipt_path = MEASUREMENT / "NATIVE_IDENTITY.json"
    if output.exists() and receipt_path.is_file():
        receipt = require_json(receipt_path)
        if receipt.get("status") == "PASS" and receipt.get("result", {}).get("byte_identical"):
            return receipt
        raise Afr1Error("an existing full output has a non-passing identity receipt")
    if output.exists():
        raise Afr1Error(f"unadjudicated full output exists; preserve and inspect: {output}")

    IDENTITY_DATA.mkdir(parents=True, exist_ok=True)
    IDENTITY_OUT.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(RUNTIME / "archive.zip") as archive:
        if archive.namelist() != ["p"]:
            raise Afr1Error("candidate archive does not contain exactly member p")
        atomic_bytes(IDENTITY_DATA / "p", archive.read("p"))

    stdout_path = IDENTITY / "inflate.stdout.log"
    stderr_path = IDENTITY / "inflate.stderr.log"
    command = [
        str(RUNTIME / "inflate.sh"),
        str(IDENTITY_DATA),
        str(IDENTITY_OUT),
        str(PUBLIC_FILE_LIST),
    ]
    environment = dict(os.environ)
    environment["PATH"] = str(REPO / ".venv" / "bin") + os.pathsep + environment.get("PATH", "")
    environment["F26_CORRECTOR_NATIVE_LIBRARY"] = str(
        BUILD_ROOT / "loadable_a" / "f26_corrector_native_afr1.dylib"
    )
    environment["F26_TOKEN_DECODER"] = "python"
    started = time.perf_counter()
    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        run = subprocess.run(
            command,
            cwd=RUNTIME,
            env=environment,
            stdout=stdout,
            stderr=stderr,
            check=False,
        )
        stdout.flush()
        os.fsync(stdout.fileno())
        stderr.flush()
        os.fsync(stderr.fileno())
    wall_seconds = time.perf_counter() - started
    if run.returncode != 0:
        raise Afr1Error(
            f"full receiver failed rc={run.returncode}; retained logs: {stdout_path}, {stderr_path}"
        )

    lines = [line for line in stdout_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        raise Afr1Error("inflate stdout carries no JSON report")
    report = json.loads(lines[-1])
    token_receipt_path = IDENTITY_OUT / ".f26_decode_checkpoints" / "tokens_cpu_stage_complete.json"
    token_payload_path = IDENTITY_OUT / ".f26_decode_checkpoints" / "tokens_cpu_stage_complete.u8"
    token_receipt = require_json(token_receipt_path)
    token_decoder = token_receipt.get("token_decoder", {})
    result = {
        "pair_count": report.get("pair_count"),
        "raw": file_fact(output),
        "python_reference_raw": {
            "path": str(LB1_REFERENCE_RAW),
            "bytes": LB1_REFERENCE_RAW.stat().st_size,
            "sha256": PYTHON_REFERENCE_RAW_SHA256,
            "receipt": file_fact(LB1_IDENTITY_RECEIPT),
        },
        "byte_identical": (
            output.stat().st_size == RAW_BYTES
            and sha256_file(output) == PYTHON_REFERENCE_RAW_SHA256
        ),
        "token_checkpoint": require_fact(token_payload_path, TOKEN_BYTES, TOKEN_SHA256),
        "free_corrector": token_decoder.get("free_corrector"),
        "decoded_token_sha256": token_decoder.get("decoded_token_sha256"),
        "corrected_cdf_input_sha256": token_decoder.get("corrected_cdf_input_sha256"),
        "corrected_quantized_logit_sha256": token_decoder.get(
            "corrected_quantized_logit_sha256"
        ),
        "decoder_bit_position": token_decoder.get("decoder_bit_position"),
        "decode_runtime_seconds": token_decoder.get("decode_runtime_seconds"),
        "decode_and_render_seconds": report.get("decode_and_render_seconds"),
        "wall_seconds": wall_seconds,
    }
    checks = {
        "full_n600": result["pair_count"] == PAIR_COUNT,
        "raw_size": result["raw"]["bytes"] == RAW_BYTES,
        "raw_matches_python_reference": result["byte_identical"],
        "token_field_identity": result["decoded_token_sha256"] == TOKEN_SHA256,
        "native_corrector_used": result["free_corrector"] == "NativeFreeCorrector",
        "archive_binding": token_receipt.get("binding", {}).get("archive_sha256")
        == CANDIDATE_ARCHIVE_SHA256,
        "candidate_stream_binding": token_receipt.get("binding", {}).get("token_stream_sha256")
        == CANDIDATE_STREAM_SHA256,
    }
    if not all(checks.values()):
        raise Afr1Error(f"full receiver identity failed: {checks}")

    payload = {
        "schema": "ddm_afr1_full_receiver_identity.v1",
        "axis": AXIS,
        "score_claim": False,
        "status": "PASS",
        "candidate_archive": require_fact(
            RUNTIME / "archive.zip",
            CANDIDATE_ARCHIVE_BYTES,
            CANDIDATE_ARCHIVE_SHA256,
        ),
        "runtime": str(RUNTIME),
        "native_library": file_fact(
            BUILD_ROOT / "loadable_a" / "f26_corrector_native_afr1.dylib"
        ),
        "command": command,
        "environment": {
            "PATH_python": str(REPO / ".venv" / "bin" / "python"),
            "F26_CORRECTOR_NATIVE_LIBRARY": environment["F26_CORRECTOR_NATIVE_LIBRARY"],
            "F26_TOKEN_DECODER": environment["F26_TOKEN_DECODER"],
        },
        "preflight": preflight,
        "result": result,
        "checks": checks,
        "stdout": file_fact(stdout_path),
        "stderr": file_fact(stderr_path),
        "token_checkpoint_receipt": file_fact(token_receipt_path),
        "retention": {
            "raw_preserved": True,
            "token_checkpoint_preserved": True,
            "candidate_and_repeat_streams_archives_preserved": True,
            "repeat_native_build_preserved": True,
        },
        "storage_after": {
            "free_bytes": shutil.disk_usage(STORE).free,
            "reserve_floor_bytes": RESERVE_BYTES,
        },
    }
    atomic_json(receipt_path, payload)
    return payload


def stage_byte_close() -> dict[str, Any]:
    control = require_control()
    port = require_json(MEASUREMENT / "NATIVE_PORT.json")
    parity = require_json(MEASUREMENT / "CONFIGURATION_PARITY.json")
    identity = require_json(MEASUREMENT / "NATIVE_IDENTITY.json")
    archive = require_fact(
        RUNTIME / "archive.zip",
        CANDIDATE_ARCHIVE_BYTES,
        CANDIDATE_ARCHIVE_SHA256,
    )
    raw = require_fact(IDENTITY_OUT / "0.raw", RAW_BYTES, PYTHON_REFERENCE_RAW_SHA256)
    python_reference_raw = require_fact(
        PYTHON_REFERENCE_RAW,
        RAW_BYTES,
        PYTHON_REFERENCE_RAW_SHA256,
    )
    token_checkpoint = require_fact(
        IDENTITY_OUT / ".f26_decode_checkpoints" / "tokens_cpu_stage_complete.u8",
        TOKEN_BYTES,
        TOKEN_SHA256,
    )
    candidate_stream = require_fact(
        CONTROL_RETAINED / AFC1_CANDIDATE_STREAM.name,
        CANDIDATE_STREAM_BYTES,
        CANDIDATE_STREAM_SHA256,
    )
    build_a = file_fact(
        BUILD_ROOT / "loadable_a" / "f26_corrector_native_afr1.dylib"
    )
    build_b = file_fact(
        BUILD_ROOT / "loadable_b" / "f26_corrector_native_afr1.dylib"
    )
    pin = check_pin_consistency(RUNTIME)
    checks = {
        "control_reproduced_minus_81_bytes": control.get("archive_delta_bytes") == -81,
        "port_passed": port.get("status") == "PASS",
        "exhaustive_python_c_parity": (
            parity.get("status") == "PASS"
            and parity.get("denominator_positions") == HEIGHT * WIDTH
            and parity.get("mismatch_positions") == 0
        ),
        "full_receiver_identity": (
            identity.get("status") == "PASS"
            and identity.get("result", {}).get("pair_count") == PAIR_COUNT
            and identity.get("result", {}).get("byte_identical") is True
        ),
        "raw_matches_direct_python_reference": (
            raw["bytes"] == python_reference_raw["bytes"]
            and raw["sha256"] == python_reference_raw["sha256"]
        ),
        "deterministic_native_build": build_a["sha256"] == build_b["sha256"],
        "runtime_pin_consistency": pin.verdict == CONSISTENT,
    }
    if not all(checks.values()):
        raise Afr1Error(f"byte-close revalidation refused: {checks}")
    payload = {
        "schema": "ddm_afr1_byte_close_revalidation.v1",
        "axis": "[macOS-CPU byte-close identity / no score claim]",
        "score_claim": False,
        "status": "PASS",
        "checks": checks,
        "candidate_archive": archive,
        "candidate_stream": candidate_stream,
        "raw": raw,
        "direct_python_reference_raw": python_reference_raw,
        "direct_python_reference_manifest": file_fact(PYTHON_REFERENCE_MANIFEST),
        "token_checkpoint": token_checkpoint,
        "native_build_a": build_a,
        "native_build_b": build_b,
        "receiver_identity_receipt": file_fact(MEASUREMENT / "NATIVE_IDENTITY.json"),
        "configuration_parity_receipt": file_fact(
            MEASUREMENT / "CONFIGURATION_PARITY.json"
        ),
        "pin_consistency": pin.verdict,
        "denominators": {
            "frames": PAIR_COUNT,
            "tokens": TOKEN_BYTES,
            "context_positions": HEIGHT * WIDTH,
            "raw_bytes": RAW_BYTES,
        },
    }
    atomic_json(MEASUREMENT / "BYTE_CLOSE_REVALIDATION.json", payload)
    return payload


def stage_handoff() -> dict[str, Any]:
    byte_close = require_json(MEASUREMENT / "BYTE_CLOSE_REVALIDATION.json")
    identity = require_json(MEASUREMENT / "NATIVE_IDENTITY.json")
    if (
        byte_close.get("status") != "PASS"
        or identity.get("status") != "PASS"
        or not identity.get("result", {}).get("byte_identical")
    ):
        raise Afr1Error("MAIN fire order requires a passing full receiver identity")
    seal_common = (
        f"--candidate-id ddm_afr1_tile48_groupbin8 --runtime-dir {RUNTIME} "
        "--receiver inflate.py --receiver inflate.sh "
        "--receiver runtime/f26_corrector_native.c "
        "--receiver runtime/native_free_corrector.py "
        "--receiver runtime/fx2_model_axis_corrector.py --archive-member p "
        f"--retained-path {MEASUREMENT / 'CONTROL_REPRODUCTION.json'} "
        f"--retained-path {MEASUREMENT / 'CONFIGURATION_PARITY.json'} "
        f"--retained-path {MEASUREMENT / 'NATIVE_IDENTITY.json'} "
        f"--retained-path {MEASUREMENT / 'BYTE_CLOSE_REVALIDATION.json'} "
        "--falsifier 'any raw-byte mismatch, token mismatch, or non-improving exact score refuses' "
        "--admit-bar-net-ds -0.00001997576859366514 "
        "--admit-bar-rule 'pre-registered approximately 30-byte lossless solo-fire bar' "
        f"--pointer-axis contest_cuda --bound-base-receipt {LB1_CUDA_RESULT} "
        "--sealed-by ddm_afr1 --notes 'paired-axis seal; MAIN owns dispatch'"
    )
    pair_group_id = "ddm_afr1_tile48_groupbin8_dual_axis"
    cuda_lane = "ddm_afr1_tile48_groupbin8_cuda_n600_20260831"
    cpu_lane = "ddm_afr1_tile48_groupbin8_cpu_n600_20260831"
    cuda_seal = MEASUREMENT / "SEAL_afr1_contest_cuda.json"
    cpu_seal = MEASUREMENT / "SEAL_afr1_contest_cpu.json"
    payload = {
        "schema": "ddm_afr1_main_fire_order.v1",
        "disposition": "QUEUED-WITH-A-FIRE-ORDER",
        "owner": "MAIN sole scorer-lane router",
        "consumer_store": str(STORE / "fire_main"),
        "fire_trigger": (
            "AFR1 serializer commit is landed; the effective pointer is still LB1 archive "
            f"{LB1_ARCHIVE_SHA256}; no scorer job or duplicate lane is active; MAIN records "
            "the named distinct CUDA and CPU claims; byte-close and both seals revalidate from disk"
        ),
        "remote_dispatched": False,
        "pair_group_id": pair_group_id,
        "candidate_archive": require_fact(
            RUNTIME / "archive.zip",
            CANDIDATE_ARCHIVE_BYTES,
            CANDIDATE_ARCHIVE_SHA256,
        ),
        "receiver_identity_receipt": file_fact(MEASUREMENT / "NATIVE_IDENTITY.json"),
        "byte_close_receipt": file_fact(MEASUREMENT / "BYTE_CLOSE_REVALIDATION.json"),
        "ordered_actions": [
            {
                "order": 1,
                "action": "REVALIDATE_BYTE_CLOSE",
                "command": (
                    ".venv/bin/python experiments/ddm_afr1_tile48_receiver_identity.py "
                    "--stage byte-close"
                ),
                "owner": "MAIN",
            },
            {
                "order": 2,
                "action": "SEAL_CONTEST_CUDA",
                "command": (
                    ".venv/bin/python tools/make_candidate_seal.py "
                    f"{seal_common} --axis contest_cuda --out {cuda_seal}"
                ),
                "constraint": "make_candidate_seal.py derives hashes; no hand-typed SHA",
                "owner": "MAIN",
            },
            {
                "order": 3,
                "action": "SEAL_CONTEST_CPU",
                "command": (
                    ".venv/bin/python tools/make_candidate_seal.py "
                    f"{seal_common} --axis contest_cpu --out {cpu_seal}"
                ),
                "constraint": "make_candidate_seal.py derives hashes; no hand-typed SHA",
                "owner": "MAIN",
            },
        ],
        "axis_order": [
            {
                "ordinal": 1,
                "axis": "contest_cuda",
                "lane_id": cuda_lane,
                "instance_job_id": f"modal:{cuda_lane}",
                "seal": str(cuda_seal),
                "exact_command": (
                    ".venv/bin/python tools/fire_modal_auth_eval.py "
                    f"--seal {cuda_seal} --output-dir {STORE / 'fire_main' / 'cuda'} "
                    f"--lane-id {cuda_lane} --instance-job-id modal:{cuda_lane} "
                    f"--claim-agent MAIN --pair-group-id {pair_group_id} --claim-policy require_active"
                ),
            },
            {
                "ordinal": 2,
                "axis": "contest_cpu",
                "lane_id": cpu_lane,
                "instance_job_id": f"modal:{cpu_lane}",
                "seal": str(cpu_seal),
                "exact_command": (
                    ".venv/bin/python tools/fire_modal_auth_eval.py "
                    f"--seal {cpu_seal} --output-dir {STORE / 'fire_main' / 'cpu'} "
                    f"--lane-id {cpu_lane} --instance-job-id modal:{cpu_lane} "
                    f"--claim-agent MAIN --pair-group-id {pair_group_id} --claim-policy require_active"
                ),
            },
        ],
        "constraints": [
            "AFR1 does not dispatch",
            "MAIN must re-read the live pointer and active-lane ledger before sealing or firing",
            "one exact n600 row per axis on identical archive bytes",
            "recompute S from d_seg, d_pose, and archive bytes",
            "keep CPU and CUDA claims separate",
        ],
    }
    atomic_json(MEASUREMENT / "MAIN_FIRE_ORDER.json", payload)
    return payload


def stage_manifest() -> dict[str, Any]:
    manifest_path = MEASUREMENT / "MANIFEST.json"
    entries = []
    for path in sorted(STORE.rglob("*")):
        if (
            not path.is_file()
            or path == manifest_path
            or path.name.endswith(".partial")
            or path.name.startswith("._")
        ):
            continue
        entries.append(
            {
                "path": str(path.relative_to(STORE)),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    payload = {
        "schema": "ddm_afr1_retained_manifest.v1",
        "root": str(STORE),
        "entries": entries,
        "entry_count": len(entries),
        "total_bytes": sum(int(row["bytes"]) for row in entries),
        "free_bytes_after_capture": shutil.disk_usage(STORE).free,
    }
    atomic_json(manifest_path, payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--stage",
        required=True,
        choices=(
            "control",
            "port",
            "parity",
            "identity-preflight",
            "identity",
            "byte-close",
            "handoff",
            "manifest",
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    stage = {
        "control": stage_control,
        "port": stage_port,
        "parity": stage_parity,
        "identity-preflight": stage_identity_preflight,
        "identity": stage_identity,
        "byte-close": stage_byte_close,
        "handoff": stage_handoff,
        "manifest": stage_manifest,
    }[args.stage]
    payload = stage()
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
