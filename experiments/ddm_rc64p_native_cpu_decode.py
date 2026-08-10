#!/usr/bin/env python3
"""Reproducible native-ANS adjudication for the custodied lc2 n600 payload.

The experiment keeps the counted payload unchanged.  It compares only the
receiver implementation selected for lc2's existing constriction ANS wire,
retains each complete decoded token field, and separates entropy-call time
from the causal HPAC probability-generation time.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

sys.dont_write_bytecode = True


REPO = Path(__file__).resolve().parents[1]
SOURCE_DIR = Path("/Volumes/VertigoDataTier/pact/ddm_lc2_20260810/submission")
SOURCE_ARCHIVE = SOURCE_DIR / "archive.zip"
SOURCE_MEMBER = SOURCE_DIR / "archive" / "p"
SOURCE_TOKEN = Path(
    "/Volumes/VertigoDataTier/pact/ddm_lc2_20260810/retained/inputs/tokens.ans"
)
GRANTED_RC64_SOURCE = Path(
    "/Volumes/VertigoDataTier/pact/pr135_intake_20260810/experiment_book/"
    "src/cpr1_sub4/entropy/rc64_backend.c"
)
DT1_MANIFEST = Path(
    "/Volumes/VertigoDataTier/pact/ddm_dt1_20260809/retained/chunk_manifest.json"
)
CONTROL = Path("/Volumes/VertigoDataTier/pact/ddm_rc64p_20260810")
BULK = Path("/Volumes/APDataStore/pact/ddm_rc64p_20260810")
IMPLEMENTATION = REPO / "experiments" / "ddm_rc64p_native_cpu_decode"
PYTHON = Path(
    "/Volumes/VertigoDataTier/pact/ddm_pq1_runtime_20260809/venv/bin/python"
)
ARCHIVE_SHA256 = "f154f0abb76980a30715282cf330d611cac7ebce3379c5f8093830dc273e1a45"
ARCHIVE_BYTES = 187_226
TOKEN_PAYLOAD_SHA256 = (
    "85d6c199ffb93ddab0fe1631448882a255e9fea1f6858bab5a04cea2310a7331"
)
TOKEN_PAYLOAD_BYTES = 114_528
DECODED_TOKEN_SHA256 = (
    "c5c7671d037b6912980c57929a5b6d789d250ee6a93e3b0a6018cf9f63e32ece"
)
DECODED_TOKEN_BYTES = 117_964_800
RAW_SHA256 = "a18eb42a8da9399bcc03e795e17597bfbd459412dbb37990117665f48c4c0353"
RAW_BYTES = 3_662_409_600
N = 600
EVAL_H = 384
EVAL_W = 512
EXPECTED_CONSTRICTION = "0.5.0"
AXIS = "[macOS-CPU receiver timing; scorer-free; n600]"


def utc_now() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_fact(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def verified_copy(source: Path, destination: Path) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.is_file() or sha256_file(destination) != sha256_file(source):
        temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
        shutil.copyfile(source, temporary)
        os.replace(temporary, destination)
    if sha256_file(source) != sha256_file(destination):
        raise RuntimeError(f"copy verification failed: {source} -> {destination}")
    return file_fact(destination)


def require_file(path: Path, *, expected_bytes: int, expected_sha256: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.stat().st_size != expected_bytes:
        raise RuntimeError(f"{path} bytes changed")
    if sha256_file(path) != expected_sha256:
        raise RuntimeError(f"{path} SHA-256 changed")


def require_space(path: Path, required_bytes: int) -> dict[str, int]:
    usage = shutil.disk_usage(path)
    if usage.free < required_bytes:
        raise RuntimeError(
            f"storage preflight failed for {path}: need {required_bytes}, "
            f"have {usage.free}"
        )
    return {
        "total_bytes": usage.total,
        "used_bytes": usage.used,
        "free_bytes": usage.free,
        "required_bytes": required_bytes,
    }


def compile_backend(destination: Path) -> dict[str, Any]:
    source = IMPLEMENTATION / "ans_backend.c"
    destination.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "/usr/bin/cc",
        "-O3",
        "-std=c11",
        "-shared",
        "-fPIC",
        "-ffp-contract=off",
        "-fno-fast-math",
    ]
    if sys.platform == "darwin":
        command.append("-Wl,-install_name,@rpath/liblc2_ans.dylib")
    command.extend([str(source), "-o", str(destination)])
    subprocess.run(command, check=True)
    return {"argv": command, "output": file_fact(destination)}


def compile_rc64_backend(source: Path, destination: Path) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "/usr/bin/cc",
        "-O3",
        "-std=c11",
        "-shared",
        "-fPIC",
        "-ffp-contract=off",
        "-fno-fast-math",
    ]
    if sys.platform == "darwin":
        command.append("-Wl,-install_name,@rpath/liblc2_rc64.dylib")
    command.extend([str(source), "-o", str(destination)])
    subprocess.run(command, check=True)
    return {"argv": command, "output": file_fact(destination)}


def patched_receiver(source: str) -> str:
    import_anchor = "import constriction\nimport numpy as np\n"
    import_replacement = (
        "import constriction\nimport numpy as np\n\n"
        "from native_ans import NativeAnsDecoder\n"
        "from route_b_rc64 import NativeRc64Decoder\n"
    )
    if source.count(import_anchor) != 1:
        raise RuntimeError("receiver import anchor changed")
    source = source.replace(import_anchor, import_replacement)
    old = '''    if token_codec == "ans":\n        return constriction.stream.stack.AnsCoder(words)\n'''
    new = '''    if token_codec == "ans":\n        native_mode = os.environ.get("LC2_NATIVE_ANS_MODE", "auto").strip().lower()\n        library_text = os.environ.get("LC2_NATIVE_ANS_LIBRARY", "").strip()\n        if native_mode not in ("auto", "off", "required"):\n            raise ReceiverFormatError(\n                f"unsupported LC2_NATIVE_ANS_MODE: {native_mode!r}"\n            )\n        if native_mode != "off" and library_text:\n            try:\n                return NativeAnsDecoder(Path(library_text), blob)\n            except Exception as error:\n                if native_mode == "required":\n                    raise ReceiverFormatError(\n                        "required native ANS decoder failed to initialize"\n                    ) from error\n                print(\n                    "LC2_NATIVE_ANS_FALLBACK reason="\n                    f"{type(error).__name__}:{error}",\n                    flush=True,\n                )\n        elif native_mode == "required":\n            raise ReceiverFormatError(\n                "required native ANS decoder lacks LC2_NATIVE_ANS_LIBRARY"\n            )\n        return constriction.stream.stack.AnsCoder(words)\n'''
    rc64_prefix = '''    if token_codec == "ans" and blob[:4] in (b"R6D1", b"R6C1"):\n        library_text = os.environ.get("LC2_RC64_LIBRARY", "").strip()\n        if not library_text:\n            raise ReceiverFormatError(\n                "explicit RC64 token payload lacks LC2_RC64_LIBRARY"\n            )\n        try:\n            return NativeRc64Decoder(Path(library_text), blob)\n        except Exception as error:\n            raise ReceiverFormatError(\n                "explicit RC64 token payload failed to initialize"\n            ) from error\n'''
    if source.count(old) != 1:
        raise RuntimeError("receiver ANS anchor changed")
    return source.replace(old, rc64_prefix + new)


def native_inflate_wrapper() -> str:
    return '''#!/bin/sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PYBIN=${PYTHON:-python3}
DEPS_DIR=${PR130_RUNTIME_DEPS_DIR:-"$SCRIPT_DIR/.runtime-deps"}
NATIVE_DIR=${LC2_NATIVE_BUILD_DIR:-"$DEPS_DIR.lc2-native-ans"}
NATIVE_LIBRARY="$NATIVE_DIR/liblc2_ans.dylib"
NATIVE_TMP="$NATIVE_DIR/.liblc2_ans.$$.tmp"
RC64_LIBRARY="$NATIVE_DIR/liblc2_rc64.dylib"
RC64_TMP="$NATIVE_DIR/.liblc2_rc64.$$.tmp"
NATIVE_MODE=${LC2_NATIVE_ANS_MODE:-auto}
CCBIN=${CC:-}
if [ -z "$CCBIN" ]; then
    CCBIN=$(command -v cc || true)
fi
LINK_DETERMINISM=
RC64_LINK_DETERMINISM=
if [ "$(uname -s)" = Darwin ]; then
    LINK_DETERMINISM='-Wl,-install_name,@rpath/liblc2_ans.dylib'
    RC64_LINK_DETERMINISM='-Wl,-install_name,@rpath/liblc2_rc64.dylib'
fi

if [ "$NATIVE_MODE" != off ]; then
    if [ -z "$CCBIN" ] || [ ! -x "$CCBIN" ]; then
        if [ "$NATIVE_MODE" = required ]; then
            echo "LC2 required native ANS compiler cc is unavailable" >&2
            exit 72
        fi
        echo "LC2_NATIVE_ANS_FALLBACK reason=cc_unavailable" >&2
    fi
    mkdir -p -- "$NATIVE_DIR"
    if [ ! -f "$NATIVE_LIBRARY" ] && [ -n "$CCBIN" ] && [ -x "$CCBIN" ]; then
        if "$CCBIN" -O3 -std=c11 -shared -fPIC \
            -ffp-contract=off -fno-fast-math \
            $LINK_DETERMINISM \
            "$SCRIPT_DIR/ans_backend.c" -o "$NATIVE_TMP"; then
            mv -- "$NATIVE_TMP" "$NATIVE_LIBRARY"
        else
            rm -f -- "$NATIVE_TMP"
            if [ "$NATIVE_MODE" = required ]; then
                echo "LC2 required native ANS compile failed" >&2
                exit 71
            fi
            echo "LC2_NATIVE_ANS_FALLBACK reason=compile_failed" >&2
        fi
    fi
    if [ -f "$NATIVE_LIBRARY" ]; then
        export LC2_NATIVE_ANS_LIBRARY="$NATIVE_LIBRARY"
    fi
fi

if [ "${LC2_RC64_MODE:-auto}" != off ]; then
    if [ -z "$CCBIN" ] || [ ! -x "$CCBIN" ]; then
        if [ "${LC2_RC64_MODE:-auto}" = required ]; then
            echo "LC2 required RC64 compiler cc is unavailable" >&2
            exit 75
        fi
    else
        mkdir -p -- "$NATIVE_DIR"
        if [ ! -f "$RC64_LIBRARY" ]; then
            if "$CCBIN" -O3 -std=c11 -shared -fPIC \
                -ffp-contract=off -fno-fast-math \
                $RC64_LINK_DETERMINISM \
                "$SCRIPT_DIR/rc64_backend.c" -o "$RC64_TMP"; then
                mv -- "$RC64_TMP" "$RC64_LIBRARY"
            else
                rm -f -- "$RC64_TMP"
                if [ "${LC2_RC64_MODE:-auto}" = required ]; then
                    echo "LC2 required RC64 compile failed" >&2
                    exit 74
                fi
            fi
        fi
        if [ -f "$RC64_LIBRARY" ]; then
            export LC2_RC64_LIBRARY="$RC64_LIBRARY"
        fi
    fi
fi

if [ "${LC2_NATIVE_COMPILE_SMOKE_ONLY:-0}" = 1 ]; then
    if [ ! -f "$NATIVE_LIBRARY" ]; then
        echo "LC2 native compile smoke lacks a loadable library" >&2
        exit 73
    fi
    export PYTHONNOUSERSITE=1
    export PYTHONDONTWRITEBYTECODE=1
    export PYTHONPATH="$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}"
    exec "$PYBIN" - "$NATIVE_LIBRARY" "$RC64_LIBRARY" <<'PY'
import ctypes
import sys

library = ctypes.CDLL(sys.argv[1])
library.lc2_ans_precision.restype = ctypes.c_uint32
library.lc2_ans_alphabet.restype = ctypes.c_uint32
if library.lc2_ans_precision() != 24 or library.lc2_ans_alphabet() != 5:
    raise SystemExit("LC2 native compile smoke resolved the wrong decoder grammar")
rc64 = ctypes.CDLL(sys.argv[2])
rc64.rc64_total_frequency.restype = ctypes.c_uint64
if rc64.rc64_total_frequency() != 1 << 31:
    raise SystemExit("LC2 RC64 compile smoke resolved the wrong frequency lattice")
print("LC2_NATIVE_COMPILE_READY ans_precision=24 alphabet=5 rc64_total=2147483648")
PY
fi

exec "$SCRIPT_DIR/inflate_lc2.sh" "$@"
'''


def prepare() -> dict[str, Any]:
    require_file(
        SOURCE_ARCHIVE,
        expected_bytes=ARCHIVE_BYTES,
        expected_sha256=ARCHIVE_SHA256,
    )
    require_file(
        SOURCE_TOKEN,
        expected_bytes=TOKEN_PAYLOAD_BYTES,
        expected_sha256=TOKEN_PAYLOAD_SHA256,
    )
    control_space = require_space(CONTROL.parent, 256 * 1024 * 1024)
    bulk_space = require_space(BULK.parent, 6 * 1024 * 1024 * 1024)
    inputs = CONTROL / "retained" / "inputs"
    runtime = CONTROL / "runtime"
    inputs.mkdir(parents=True, exist_ok=True)
    runtime.mkdir(parents=True, exist_ok=True)
    BULK.mkdir(parents=True, exist_ok=True)

    retained = {
        "archive": verified_copy(SOURCE_ARCHIVE, inputs / "archive.zip"),
        "token_payload": verified_copy(SOURCE_TOKEN, inputs / "tokens.ans"),
    }
    with zipfile.ZipFile(SOURCE_ARCHIVE) as archive:
        payload = archive.read("p")
    atomic_bytes(runtime / "archive" / "p", payload)
    if payload != SOURCE_MEMBER.read_bytes():
        raise RuntimeError("archive member p differs from source runtime member")

    runtime_files = (
        "carrier_codec.py",
        "hpac_integer.py",
        "hpac_integer_sparse.py",
        "inflate.py",
        "integer_model_io.py",
        "runtime-dependencies.json",
    )
    for name in runtime_files:
        verified_copy(SOURCE_DIR / name, runtime / name)
    atomic_bytes(
        runtime / "receiver.py",
        patched_receiver((SOURCE_DIR / "receiver.py").read_text()).encode(),
    )
    verified_copy(SOURCE_DIR / "inflate.sh", runtime / "inflate_lc2.sh")
    os.chmod(runtime / "inflate_lc2.sh", 0o755)
    atomic_bytes(runtime / "inflate.sh", native_inflate_wrapper().encode())
    os.chmod(runtime / "inflate.sh", 0o755)
    verified_copy(IMPLEMENTATION / "ans_backend.c", runtime / "ans_backend.c")
    verified_copy(IMPLEMENTATION / "native_ans.py", runtime / "native_ans.py")
    verified_copy(
        IMPLEMENTATION / "route_b_rc64.py", runtime / "route_b_rc64.py"
    )
    route_b_spec = importlib.util.spec_from_file_location(
        "ddm_rc64p_route_b_prepare", IMPLEMENTATION / "route_b_rc64.py"
    )
    if route_b_spec is None or route_b_spec.loader is None:
        raise RuntimeError("cannot import Route-B RC64 implementation")
    route_b_module = importlib.util.module_from_spec(route_b_spec)
    route_b_spec.loader.exec_module(route_b_module)
    rc64_source = GRANTED_RC64_SOURCE.read_bytes() + (
        "\n" + route_b_module.RC64_CHECKPOINT_EXTENSION
    ).encode()
    atomic_bytes(runtime / "rc64_backend.c", rc64_source)
    atomic_bytes(runtime / "video_names.txt", b"0.mp4\n")

    repeat_output = CONTROL / "build" / "repeat" / "liblc2_ans.dylib"
    first_build = compile_backend(repeat_output)
    first_fact = verified_copy(
        repeat_output, CONTROL / "build" / "a" / "liblc2_ans.dylib"
    )
    second_build = compile_backend(repeat_output)
    second_fact = verified_copy(
        repeat_output, CONTROL / "build" / "b" / "liblc2_ans.dylib"
    )
    compile_a = {**first_build, "retained_output": first_fact}
    compile_b = {**second_build, "retained_output": second_fact}
    if first_fact["sha256"] != second_fact["sha256"]:
        raise RuntimeError("native backend compile repeat was not byte-identical")

    rc64_repeat = CONTROL / "build" / "repeat" / "liblc2_rc64.dylib"
    rc64_first_build = compile_rc64_backend(runtime / "rc64_backend.c", rc64_repeat)
    rc64_first_fact = verified_copy(
        rc64_repeat, CONTROL / "build" / "rc64_a" / "liblc2_rc64.dylib"
    )
    rc64_second_build = compile_rc64_backend(runtime / "rc64_backend.c", rc64_repeat)
    rc64_second_fact = verified_copy(
        rc64_repeat, CONTROL / "build" / "rc64_b" / "liblc2_rc64.dylib"
    )
    if rc64_first_fact["sha256"] != rc64_second_fact["sha256"]:
        raise RuntimeError("RC64 backend compile repeat was not byte-identical")

    for cache in runtime.rglob("__pycache__"):
        shutil.rmtree(cache)
    runtime_manifest = []
    for path in sorted(runtime.rglob("*")):
        if path.is_file():
            runtime_manifest.append(file_fact(path))
    result = {
        "schema": "ddm_rc64p_prepare.v1",
        "complete": True,
        "written_at_utc": utc_now(),
        "axis": AXIS,
        "score_claim": False,
        "storage": {"control": control_space, "bulk": bulk_space},
        "retained": retained,
        "runtime_manifest": runtime_manifest,
        "compile_a": compile_a,
        "compile_b": compile_b,
        "compile_repeat_identical": True,
        "rc64_compile_a": {
            **rc64_first_build,
            "retained_output": rc64_first_fact,
        },
        "rc64_compile_b": {
            **rc64_second_build,
            "retained_output": rc64_second_fact,
        },
        "rc64_compile_repeat_identical": True,
        "bulk_policy": (
            "VertigoDataTier lacked room for four token checkpoints plus the "
            "3.66 GB raw; bulky byte-closed outputs are retained on APDataStore"
        ),
    }
    atomic_json(CONTROL / "receipts" / "prepare.json", result)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return result


def import_runtime(*, optimized: bool = False) -> tuple[Any, Any]:
    runtime = CONTROL / ("runtime_optimized" if optimized else "runtime")
    for candidate in (CONTROL / "runtime", CONTROL / "runtime_optimized"):
        while str(candidate) in sys.path:
            sys.path.remove(str(candidate))
    sys.path.insert(0, str(runtime))
    for name in (
        "receiver",
        "inflate",
        "native_ans",
        "route_b_rc64",
        "hpac_integer",
        "hpac_integer_sparse",
        "integer_model_io",
        "carrier_codec",
    ):
        sys.modules.pop(name, None)
    receiver = importlib.import_module("receiver")
    inflate = importlib.import_module("inflate")
    return receiver, inflate


def prepare_optimized_runtime() -> dict[str, Any]:
    source = CONTROL / "runtime"
    destination = CONTROL / "runtime_optimized"
    if not (CONTROL / "receipts" / "prepare.json").is_file():
        raise RuntimeError("base runtime prepare receipt is missing")
    for path in sorted(source.rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts:
            verified_copy(path, destination / path.relative_to(source))
    verified_copy(
        IMPLEMENTATION / "hpac_integer_sparse_optimized.py",
        destination / "hpac_integer_sparse.py",
    )
    os.chmod(destination / "inflate.sh", 0o755)
    os.chmod(destination / "inflate_lc2.sh", 0o755)
    for cache in destination.rglob("__pycache__"):
        shutil.rmtree(cache)
    manifest = [
        file_fact(path) for path in sorted(destination.rglob("*")) if path.is_file()
    ]
    result = {
        "schema": "ddm_rc64p_optimized_runtime_prepare.v1",
        "complete": True,
        "written_at_utc": utc_now(),
        "axis": "[generic receiver runtime build; scorer-free]",
        "score_claim": False,
        "optimization": (
            "cache immutable rounded weights, mask-compacted kernels, exponent "
            "powers, and conv-a gather indices outside the 114000-group loop"
        ),
        "files": manifest,
    }
    atomic_json(CONTROL / "receipts" / "optimized_runtime_prepare.json", result)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return result


def optimized_parity_smoke() -> dict[str, Any]:
    os.environ["PR130_BROTLI_CLI"] = "/opt/homebrew/bin/brotli"
    receiver, inflate = import_runtime(optimized=False)
    import torch

    payload = (CONTROL / "runtime" / "archive" / "p").read_bytes()
    parts = receiver.split_payload(payload)
    decoded_models = receiver.decode_models(parts.models, model_codec=parts.model_codec)
    models_raw, _ = receiver.split_optional_temporal_reversion(decoded_models.raw)
    semantic_bytes = int.from_bytes(models_raw[:4], "little")
    carrier_bytes = int.from_bytes(models_raw[4:8], "little")
    hpac = inflate.load_hpac(
        models_raw[8 + semantic_bytes + carrier_bytes:], torch.device("cpu")
    )
    base_module = importlib.import_module("hpac_integer_sparse")
    spec = importlib.util.spec_from_file_location(
        "ddm_rc64p_hpac_optimized",
        IMPLEMENTATION / "hpac_integer_sparse_optimized.py",
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import optimized sparse evaluator")
    optimized_module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = optimized_module
    spec.loader.exec_module(optimized_module)
    baseline = base_module.SparseIntegerHPAC(hpac, EVAL_H, EVAL_W)
    optimized = optimized_module.SparseIntegerHPAC(hpac, EVAL_H, EVAL_W)
    masks = inflate.group_masks(torch.device("cpu"))
    canonical_path = BULK / "runs" / "native_t4" / "decoded_tokens.u8"
    require_file(
        canonical_path,
        expected_bytes=DECODED_TOKEN_BYTES,
        expected_sha256=DECODED_TOKEN_SHA256,
    )
    canonical = np.memmap(
        canonical_path, mode="r", dtype=np.uint8, shape=(N, EVAL_H, EVAL_W)
    )
    previous = torch.zeros((1, EVAL_H, EVAL_W), dtype=torch.long)
    current = torch.zeros_like(previous)
    context = hpac.prepare_frame_context(torch.tensor([0]), previous)
    started = time.perf_counter()
    checked_values = 0
    for group, mask in enumerate(masks):
        expected = baseline.selected_logits(current, context, group)
        actual = optimized.selected_logits(current, context, group)
        if not torch.equal(expected, actual):
            difference = (expected != actual).nonzero(as_tuple=False)[0].tolist()
            raise RuntimeError(
                f"optimized sparse logits differ at group {group}, index {difference}"
            )
        checked_values += actual.numel()
        current[0, mask] = torch.from_numpy(
            np.asarray(canonical[0][mask.numpy()], dtype=np.int64).copy()
        )
    result = {
        "schema": "ddm_rc64p_optimized_sparse_parity_smoke.v1",
        "complete": True,
        "written_at_utc": utc_now(),
        "axis": "[macOS-CPU actual lc2 model and canonical frame 0; scorer-free]",
        "score_claim": False,
        "frames": 1,
        "groups": len(masks),
        "logit_values_checked": checked_values,
        "bit_identical_logits": True,
        "wall_seconds": time.perf_counter() - started,
        "canonical_tokens": file_fact(canonical_path),
    }
    atomic_json(CONTROL / "receipts" / "optimized_parity_smoke.json", result)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return result


def import_route_b() -> Any:
    name = "ddm_rc64p_route_b"
    sys.modules.pop(name, None)
    spec = importlib.util.spec_from_file_location(
        name, IMPLEMENTATION / "route_b_rc64.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import Route-B RC64 implementation")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def import_tm1() -> Any:
    name = "ddm_rc64p_tm1_reference"
    sys.modules.pop(name, None)
    spec = importlib.util.spec_from_file_location(
        name, REPO / "experiments" / "ddm_tm1_token_model_lever.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import the granted TM1 retained-corpus reader")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def deterministic_zip(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with zipfile.ZipFile(temporary, "w", allowZip64=False) as archive:
        info = zipfile.ZipInfo("p", date_time=(1980, 1, 1, 0, 0, 0))
        info.compress_type = zipfile.ZIP_STORED
        info.create_system = 3
        info.external_attr = 0o100644 << 16
        archive.writestr(info, payload)
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def route_b_encode() -> dict[str, Any]:
    prior_receipt = CONTROL / "route_b" / "encode_receipt.json"
    token_path = CONTROL / "route_b" / "retained" / "tokens.rc64"
    archive_path = CONTROL / "route_b" / "submission" / "archive.zip"
    if prior_receipt.is_file():
        prior = json.loads(prior_receipt.read_text())
        if (
            prior.get("complete") is True
            and token_path.is_file()
            and archive_path.is_file()
            and sha256_file(token_path) == prior["token_payload"]["sha256"]
            and sha256_file(archive_path) == prior["archive"]["sha256"]
        ):
            print(json.dumps(prior, indent=2, sort_keys=True), flush=True)
            return prior

    tm1 = import_tm1()
    route_b = import_route_b()
    corpus = tm1.RetainedCorpus(DT1_MANIFEST)
    corpus_receipt = corpus.validate(deep_hash=True)
    os.environ["PR130_BROTLI_CLI"] = "/opt/homebrew/bin/brotli"
    receiver, _ = import_runtime()
    source_payload = (CONTROL / "runtime" / "archive" / "p").read_bytes()
    parts = receiver.split_payload(source_payload)
    decoded_models = receiver.decode_models(
        parts.models, model_codec=parts.model_codec
    )
    _, temporal = receiver.split_optional_temporal_reversion(decoded_models.raw)
    if temporal is None:
        raise RuntimeError("lc2 payload lost its temporal-reversion model")
    model = tm1.CandidateModel(
        "temporal_reversion", corrections=temporal.corrections.copy()
    )
    model.validate()

    checkpoint_dir = CONTROL / "route_b" / "checkpoints"
    latest_path = checkpoint_dir / "LATEST.json"
    start_frame = 0
    checkpoint: bytes | None = None
    if latest_path.is_file():
        latest = json.loads(latest_path.read_text())
        checkpoint_path = Path(latest["encoder_checkpoint"]["path"])
        if sha256_file(checkpoint_path) != latest["encoder_checkpoint"]["sha256"]:
            raise RuntimeError("Route-B encoder checkpoint SHA-256 changed")
        checkpoint = checkpoint_path.read_bytes()
        start_frame = int(latest["next_frame"])
        if not 0 <= start_frame <= N:
            raise RuntimeError("Route-B encoder checkpoint frame is invalid")

    encoder = route_b.NativeRc64Encoder(
        CONTROL / "build" / "rc64_a" / "liblc2_rc64.dylib",
        checkpoint=checkpoint,
    )
    started = time.perf_counter()
    try:
        for frame in range(start_frame, N):
            symbols_raw, codes_raw = corpus.frame(frame)
            symbols = np.asarray(symbols_raw, dtype=np.int32)
            codes = np.asarray(codes_raw, dtype=np.int16)
            previous_one = (
                np.zeros(tm1.TOKENS_PER_FRAME, dtype=np.uint8)
                if frame < 1
                else np.asarray(corpus.frame(frame - 1)[0], dtype=np.uint8)
            )
            previous_two = (
                np.zeros(tm1.TOKENS_PER_FRAME, dtype=np.uint8)
                if frame < 2
                else np.asarray(corpus.frame(frame - 2)[0], dtype=np.uint8)
            )
            corrected = tm1.apply_candidate(
                model,
                codes,
                frame,
                np.empty(0, dtype=np.int16),
                previous_one=previous_one,
                previous_two=previous_two,
            )
            tables = tm1.probability_tables(corrected)
            encoder.encode(symbols, tables)
            if (frame + 1) % 25 == 0 and frame + 1 < N:
                state_path = checkpoint_dir / f"through_frame_{frame:03d}.encoder"
                atomic_bytes(state_path, encoder.snapshot())
                checkpoint_receipt = {
                    "schema": "ddm_rc64p_route_b_encoder_checkpoint.v1",
                    "complete": True,
                    "written_at_utc": utc_now(),
                    "next_frame": frame + 1,
                    "encoder_checkpoint": file_fact(state_path),
                    "manifest": corpus_receipt,
                }
                atomic_json(
                    checkpoint_dir / f"through_frame_{frame:03d}.json",
                    checkpoint_receipt,
                )
                atomic_json(latest_path, checkpoint_receipt)
        token_payload = encoder.finish()
    finally:
        encoder.close()

    # Retention precedes every byte/rate comparison.
    atomic_bytes(token_path, token_payload)
    member = receiver.pack_payload(
        parts.models,
        token_payload,
        token_codec="ans",
        model_codec=parts.model_codec,
    )
    member_path = CONTROL / "route_b" / "submission" / "archive" / "p"
    atomic_bytes(member_path, member)
    deterministic_zip(archive_path, member)
    repeat_archive = CONTROL / "route_b" / "submission" / "archive.repeat.zip"
    deterministic_zip(repeat_archive, member)
    if archive_path.read_bytes() != repeat_archive.read_bytes():
        raise RuntimeError("Route-B archive repeat is not byte-identical")

    token_fact = file_fact(token_path)
    archive_fact = file_fact(archive_path)
    result = {
        "schema": "ddm_rc64p_route_b_encode.v1",
        "complete": True,
        "written_at_utc": utc_now(),
        "axis": "[macOS-CPU retained-code RC64 encoding; scorer-free; n600]",
        "score_claim": False,
        "frames": N,
        "symbols": DECODED_TOKEN_BYTES,
        "elapsed_seconds_this_invocation": time.perf_counter() - started,
        "resumed_from_frame": start_frame,
        "manifest": corpus_receipt,
        "borrowed_rc64_source": file_fact(GRANTED_RC64_SOURCE),
        "compiled_rc64_library": file_fact(
            CONTROL / "build" / "rc64_a" / "liblc2_rc64.dylib"
        ),
        "token_payload": token_fact,
        "ans_control_payload": file_fact(
            CONTROL / "retained" / "inputs" / "tokens.ans"
        ),
        "token_bytes_saved": TOKEN_PAYLOAD_BYTES - token_fact["bytes"],
        "strict_token_byte_win": token_fact["bytes"] < TOKEN_PAYLOAD_BYTES,
        "member": file_fact(member_path),
        "archive": archive_fact,
        "archive_repeat": file_fact(repeat_archive),
        "archive_repeat_identical": True,
        "archive_bytes_saved": ARCHIVE_BYTES - archive_fact["bytes"],
        "validation_status": (
            "full_receiver_required"
            if token_fact["bytes"] < TOKEN_PAYLOAD_BYTES
            else "not_run_route_b_failed_byte_gate"
        ),
    }
    atomic_json(prior_receipt, result)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return result


def golden() -> dict[str, Any]:
    import constriction

    sys.path.insert(0, str(IMPLEMENTATION))
    native_ans = importlib.import_module("native_ans")
    rng = np.random.default_rng(6464)
    count = 10_003
    logits = rng.normal(size=(count, 5)).astype(np.float32)
    logits -= logits.max(axis=1, keepdims=True)
    tables = np.exp(logits).astype(np.float32)
    tables /= tables.sum(axis=1, keepdims=True)
    symbols = rng.integers(0, 5, count, dtype=np.int32)
    family = constriction.stream.model.Categorical(perfect=False)
    encoder = constriction.stream.stack.AnsCoder()
    encoder.encode_reverse(symbols, family, tables)
    words = encoder.get_compressed().copy()
    payload = words.astype("<u4", copy=False).tobytes(order="C")
    oracle = constriction.stream.stack.AnsCoder(words.copy())
    native = native_ans.NativeAnsDecoder(
        CONTROL / "build" / "a" / "liblc2_ans.dylib", payload
    )
    split = count // 3
    oracle_first = oracle.decode(family, tables[:split])
    native_first = native.decode(family, tables[:split])
    oracle_snapshot = oracle.get_compressed().astype("<u4", copy=False)
    native_snapshot = native.get_compressed().astype("<u4", copy=False)
    oracle_rest = oracle.decode(family, tables[split:])
    native_rest = native.decode(family, tables[split:])
    oracle_symbols = np.concatenate([oracle_first, oracle_rest])
    native_symbols = np.concatenate([native_first, native_rest])
    checks = {
        "oracle_matches_source": bool(np.array_equal(oracle_symbols, symbols)),
        "native_matches_source": bool(np.array_equal(native_symbols, symbols)),
        "native_matches_oracle": bool(np.array_equal(native_symbols, oracle_symbols)),
        "midstream_snapshot_identity": bool(
            np.array_equal(native_snapshot, oracle_snapshot)
        ),
        "oracle_empty": bool(oracle.is_empty()),
        "native_empty": bool(native.is_empty()),
    }
    if not all(checks.values()):
        raise RuntimeError(f"native golden-vector gate failed: {checks}")
    vector_dir = CONTROL / "retained" / "golden"
    atomic_bytes(vector_dir / "payload.ans", payload)
    atomic_bytes(vector_dir / "symbols.i32", symbols.astype("<i4").tobytes())
    atomic_bytes(vector_dir / "tables.f32", tables.astype("<f4").tobytes())
    atomic_bytes(
        vector_dir / "midstream_snapshot.u32", native_snapshot.tobytes(order="C")
    )
    result = {
        "schema": "ddm_rc64p_golden.v1",
        "complete": True,
        "written_at_utc": utc_now(),
        "axis": "[portable native/Python equivalence; scorer-free]",
        "score_claim": False,
        "seed": 6464,
        "symbols": count,
        "checks": checks,
        "artifacts": [file_fact(path) for path in sorted(vector_dir.iterdir())],
        "native_library": file_fact(
            CONTROL / "build" / "a" / "liblc2_ans.dylib"
        ),
    }
    atomic_json(CONTROL / "receipts" / "golden.json", result)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return result


class TimedDecoder:
    def __init__(self, inner: object, counters: dict[str, Any]) -> None:
        self.inner = inner
        self.counters = counters

    def decode(self, model: object, probabilities: np.ndarray) -> np.ndarray:
        started = time.perf_counter()
        result = self.inner.decode(model, probabilities)
        elapsed = time.perf_counter() - started
        self.counters["decode_calls"] += 1
        self.counters["decoded_symbols"] += len(result)
        self.counters["entropy_call_seconds"] += elapsed
        return result

    def is_empty(self) -> bool:
        return bool(self.inner.is_empty())

    def get_compressed(self) -> np.ndarray:
        return self.inner.get_compressed()


def worker(decoder: str, threads: int, run_tag: str = "") -> dict[str, Any]:
    if decoder not in ("constriction", "native", "rc64", "rc64_opt"):
        raise ValueError(decoder)
    if threads not in (1, 4):
        raise ValueError(threads)
    if run_tag and (not run_tag.replace("-", "").isalnum()):
        raise ValueError("run tag must be alphanumeric/hyphen")
    if decoder in ("rc64", "rc64_opt"):
        route_receipt = json.loads(
            (CONTROL / "route_b" / "encode_receipt.json").read_text()
        )
        if not route_receipt.get("strict_token_byte_win"):
            raise RuntimeError("Route B did not pass its strict byte gate")
        token_path = Path(route_receipt["token_payload"]["path"])
        expected_token_bytes = int(route_receipt["token_payload"]["bytes"])
        expected_token_sha = str(route_receipt["token_payload"]["sha256"])
        payload_path = CONTROL / "route_b" / "submission" / "archive" / "p"
        source_archive_path = CONTROL / "route_b" / "submission" / "archive.zip"
    else:
        token_path = CONTROL / "retained" / "inputs" / "tokens.ans"
        expected_token_bytes = TOKEN_PAYLOAD_BYTES
        expected_token_sha = TOKEN_PAYLOAD_SHA256
        payload_path = CONTROL / "runtime" / "archive" / "p"
        source_archive_path = CONTROL / "retained" / "inputs" / "archive.zip"
    require_file(
        token_path,
        expected_bytes=expected_token_bytes,
        expected_sha256=expected_token_sha,
    )
    run_name = f"{decoder}_{run_tag}_t{threads}" if run_tag else f"{decoder}_t{threads}"
    run_control = CONTROL / "runs" / run_name
    run_bulk = BULK / "runs" / run_name
    run_control.mkdir(parents=True, exist_ok=True)
    run_bulk.mkdir(parents=True, exist_ok=True)
    receipt_path = run_control / "timing_receipt.json"
    decoded_path = run_bulk / "decoded_tokens.u8"
    cache_path = run_bulk / "tokens.npz"
    cache_receipt = run_bulk / "tokens_receipt.json"
    if receipt_path.is_file():
        prior = json.loads(receipt_path.read_text())
        if (
            prior.get("complete") is True
            and decoded_path.is_file()
            and sha256_file(decoded_path) == DECODED_TOKEN_SHA256
            and cache_path.is_file()
        ):
            print(json.dumps(prior, indent=2, sort_keys=True), flush=True)
            return prior

    os.environ["LC2_NATIVE_ANS_MODE"] = "required" if decoder == "native" else "off"
    os.environ["LC2_NATIVE_ANS_LIBRARY"] = str(
        CONTROL / "build" / "a" / "liblc2_ans.dylib"
    )
    os.environ["LC2_RC64_LIBRARY"] = str(
        CONTROL / "build" / "rc64_a" / "liblc2_rc64.dylib"
    )
    brotli_cli = Path("/opt/homebrew/bin/brotli")
    brotli_version = subprocess.run(
        [str(brotli_cli), "--version"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if brotli_version != "brotli 1.2.0":
        raise RuntimeError(f"expected brotli 1.2.0, resolved {brotli_version!r}")
    os.environ["PR130_BROTLI_CLI"] = str(brotli_cli)
    receiver, inflate = import_runtime(optimized=decoder == "rc64_opt")
    import torch

    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    counters: dict[str, Any] = {
        "decode_calls": 0,
        "decoded_symbols": 0,
        "entropy_call_seconds": 0.0,
        "decoder_class": None,
    }
    original_new_decoder = receiver.new_token_decoder

    def new_timed_decoder(blob: bytes, token_codec: str) -> TimedDecoder:
        inner = original_new_decoder(blob, token_codec)
        decoder_class = type(inner).__name__
        if counters["decoder_class"] not in (None, decoder_class):
            raise RuntimeError("token decoder class changed during one run")
        counters["decoder_class"] = decoder_class
        return TimedDecoder(inner, counters)

    receiver.new_token_decoder = new_timed_decoder
    inflate.new_token_decoder = new_timed_decoder
    native_ans = importlib.import_module("native_ans")
    native_ans.reset_telemetry()

    payload = payload_path.read_bytes()
    parts = receiver.split_payload(payload)
    if parts.token_codec != "ans":
        raise RuntimeError("lc2 payload no longer selects ANS")
    if sha256_file(token_path) != hashlib.sha256(parts.tokens).hexdigest():
        raise RuntimeError("runtime token section differs from retained payload")
    decoded_models = receiver.decode_models(parts.models, model_codec=parts.model_codec)
    models_raw_wire = decoded_models.raw
    models_raw, temporal_reversion = receiver.split_optional_temporal_reversion(
        models_raw_wire
    )
    semantic_bytes = int.from_bytes(models_raw[:4], byteorder="little")
    carrier_bytes = int.from_bytes(models_raw[4:8], byteorder="little")
    semantic_pose_bytes = 8 + semantic_bytes + carrier_bytes
    device = torch.device("cpu")
    hpac = inflate.load_hpac(models_raw[semantic_pose_bytes:], device)
    progress_path = run_control / "tokens.progress.npz"
    legacy_progress_path = run_bulk / "tokens.progress.npz"
    if not progress_path.is_file() and legacy_progress_path.is_file():
        verified_copy(legacy_progress_path, progress_path)
    progress_start_frame = 0
    if progress_path.is_file():
        with np.load(progress_path, allow_pickle=False) as progress:
            progress_start_frame = int(progress["frames_completed"].item())
    binding = inflate.token_binding(
        payload, models_raw_wire, parts.tokens, parts.token_codec
    )
    started_wall = time.perf_counter()
    started_process = time.process_time()
    tokens, finish_proof = inflate.decode_tokens(
        hpac,
        parts.tokens,
        device,
        token_codec=parts.token_codec,
        return_finish_proof=True,
        progress_cache_path=progress_path,
        progress_binding=binding,
        checkpoint_interval_frames=10,
        temporal_reversion=temporal_reversion,
    )
    process_seconds = time.process_time() - started_process
    wall_seconds = time.perf_counter() - started_wall
    del hpac
    inflate.write_token_checkpoint(
        tokens,
        finish_proof=finish_proof,
        payload=payload,
        models_raw=models_raw_wire,
        token_payload=parts.tokens,
        token_codec=parts.token_codec,
        cache_path=cache_path,
        receipt_path=cache_receipt,
    )
    array = tokens.detach().cpu().contiguous().numpy()
    temporary = decoded_path.with_name(f".{decoded_path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        array.tofile(handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, decoded_path)
    decoded_fact = file_fact(decoded_path)
    if decoded_fact["bytes"] != DECODED_TOKEN_BYTES:
        raise RuntimeError("decoded n600 token geometry changed")
    if decoded_fact["sha256"] != DECODED_TOKEN_SHA256:
        raise RuntimeError("decoded n600 tokens differ from the Python oracle")
    if counters["decoded_symbols"] != DECODED_TOKEN_BYTES:
        raise RuntimeError("timing wrapper did not observe every decoded symbol")
    native_telemetry = native_ans.telemetry_snapshot()
    if decoder == "native":
        if native_telemetry["decoded_symbols"] != DECODED_TOKEN_BYTES:
            raise RuntimeError("required native decoder did not consume every symbol")
        if native_telemetry["decoder_instances"] < 1:
            raise RuntimeError("required native decoder was not instantiated")
    elif native_telemetry["decoded_symbols"]:
        raise RuntimeError("constriction control unexpectedly entered native code")
    if decoder in ("rc64", "rc64_opt") and (
        counters["decoder_class"] != "NativeRc64Decoder"
    ):
        raise RuntimeError("explicit RC64 payload did not use NativeRc64Decoder")

    result = {
        "schema": "ddm_rc64p_n600_timing.v1",
        "complete": True,
        "written_at_utc": utc_now(),
        "axis": AXIS,
        "score_claim": False,
        "decoder": decoder,
        "threads": threads,
        "frames": N,
        "decoded_symbols": DECODED_TOKEN_BYTES,
        "wall_seconds": wall_seconds,
        "timing_scope": (
            "full_n600_uninterrupted"
            if progress_start_frame == 0
            else "resumed_suffix_not_full_n600_timing"
        ),
        "progress_start_frame": progress_start_frame,
        "wall_seconds_is_full_n600": progress_start_frame == 0,
        "process_seconds": process_seconds,
        "entropy_call_seconds": counters["entropy_call_seconds"],
        "non_entropy_wall_lower_bound_seconds": max(
            0.0, wall_seconds - counters["entropy_call_seconds"]
        ),
        "entropy_wall_fraction": counters["entropy_call_seconds"] / wall_seconds,
        "timing_counters": counters,
        "native_telemetry": native_telemetry,
        "native_required_no_fallback": decoder in ("native", "rc64", "rc64_opt"),
        "optimized_hpac_runtime": decoder == "rc64_opt",
        "source_archive": file_fact(source_archive_path),
        "source_token_payload": file_fact(token_path),
        "decoded_tokens": decoded_fact,
        "token_checkpoint": file_fact(cache_path),
        "token_checkpoint_receipt": file_fact(cache_receipt),
        "progress_checkpoint": file_fact(progress_path),
        "expected_decoded_token_sha256": DECODED_TOKEN_SHA256,
        "finish_token_decode_returned": True,
        "entropy_final_state_empty": True,
        "command": [
            str(PYTHON),
            str(Path(__file__).resolve()),
            "worker",
            "--decoder",
            decoder,
            "--threads",
            str(threads),
            *(["--run-tag", run_tag] if run_tag else []),
        ],
    }
    atomic_json(receipt_path, result)
    atomic_json(run_bulk / "timing_receipt.json", result)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return result


def bootstrap_smoke() -> dict[str, Any]:
    prepare_receipt = CONTROL / "receipts" / "prepare.json"
    if not prepare_receipt.is_file():
        raise RuntimeError("prepare receipt missing")
    smoke_root = BULK / "bootstrap_smoke"
    native_build = smoke_root / "fresh-native-build-v4"
    if native_build.exists():
        raise RuntimeError(f"fresh native build target already exists: {native_build}")
    smoke_root.mkdir(parents=True, exist_ok=True)
    log = smoke_root / "bootstrap.log"
    if log.is_file():
        verified_copy(log, smoke_root / "bootstrap_failure_network.log")
    env = os.environ.copy()
    env.update(
        PYTHON="/opt/homebrew/bin/python3",
        LC2_NATIVE_ANS_MODE="required",
        LC2_RC64_MODE="required",
        LC2_NATIVE_BUILD_DIR=str(native_build),
        LC2_NATIVE_COMPILE_SMOKE_ONLY="1",
    )
    command = [str(CONTROL / "runtime" / "inflate.sh")]
    started = time.perf_counter()
    run = subprocess.run(command, env=env, text=True, capture_output=True, check=False)
    elapsed = time.perf_counter() - started
    atomic_bytes(log, (run.stdout + run.stderr).encode())
    library = native_build / "liblc2_ans.dylib"
    rc64_library = native_build / "liblc2_rc64.dylib"
    if run.returncode != 0 or not library.is_file() or not rc64_library.is_file():
        raise RuntimeError(
            f"bare-target runtime bootstrap failed ({run.returncode}); see {log}"
        )
    result = {
        "schema": "ddm_rc64p_bootstrap_smoke.v1",
        "complete": True,
        "written_at_utc": utc_now(),
        "axis": "[macOS-CPU fresh dependency target; scorer-free]",
        "score_claim": False,
        "returncode": run.returncode,
        "wall_seconds": elapsed,
        "command": command,
        "environment": {
            "PYTHON": env["PYTHON"],
            "LC2_NATIVE_ANS_MODE": "required",
            "LC2_RC64_MODE": "required",
            "LC2_NATIVE_BUILD_DIR": str(native_build),
            "LC2_NATIVE_COMPILE_SMOKE_ONLY": "1",
        },
        "scope": (
            "fresh native build and load on the Homebrew Python provider that "
            "has contest-like numpy/torch but no constriction; inherited lc2 "
            "Python dependency bootstrap is outside this new-path smoke"
        ),
        "log": file_fact(log),
        "native_library": file_fact(library),
        "rc64_library": file_fact(rc64_library),
    }
    atomic_json(CONTROL / "receipts" / "bootstrap_smoke.json", result)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return result


def end_to_end() -> dict[str, Any]:
    native_receipt = CONTROL / "runs" / "native_t4" / "timing_receipt.json"
    if not native_receipt.is_file():
        raise RuntimeError("native_t4 token receipt missing")
    native = json.loads(native_receipt.read_text())
    if not native.get("complete"):
        raise RuntimeError("native_t4 token receipt is incomplete")
    output = BULK / "end_to_end" / "inflated"
    output.mkdir(parents=True, exist_ok=True)
    raw = output / "0.raw"
    log = CONTROL / "receipts" / "end_to_end.log"
    env = os.environ.copy()
    env.update(
        PYTHON=str(PYTHON),
        PR130_RUNTIME_DEPS_DIR=str(BULK / "end_to_end" / "runtime-deps"),
        PR130_TOKEN_CACHE=str(BULK / "runs" / "native_t4" / "tokens.npz"),
        PR130_TOKEN_RECEIPT=str(
            BULK / "runs" / "native_t4" / "tokens_receipt.json"
        ),
        LC2_NATIVE_ANS_MODE="required",
        OMP_NUM_THREADS="4",
        MKL_NUM_THREADS="4",
        PR130_BROTLI_CLI="/opt/homebrew/bin/brotli",
    )
    command = [
        str(CONTROL / "runtime" / "inflate.sh"),
        str(CONTROL / "runtime" / "archive"),
        str(output),
        str(CONTROL / "runtime" / "video_names.txt"),
    ]
    started = time.perf_counter()
    run = subprocess.run(command, env=env, text=True, capture_output=True, check=False)
    elapsed = time.perf_counter() - started
    atomic_bytes(log, (run.stdout + run.stderr).encode())
    if run.returncode != 0:
        raise RuntimeError(f"literal receiver runtime failed; see {log}")
    require_file(raw, expected_bytes=RAW_BYTES, expected_sha256=RAW_SHA256)
    result = {
        "schema": "ddm_rc64p_end_to_end.v1",
        "complete": True,
        "written_at_utc": utc_now(),
        "axis": "[macOS-CPU literal inflate.sh parseback; scorer-free; n600]",
        "score_claim": False,
        "wall_seconds": elapsed,
        "command": command,
        "native_decode_proof": file_fact(native_receipt),
        "token_cache_reused": True,
        "raw": file_fact(raw),
        "log": file_fact(log),
    }
    atomic_json(CONTROL / "receipts" / "end_to_end.json", result)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return result


def route_b_end_to_end() -> dict[str, Any]:
    timing_receipt = CONTROL / "runs" / "rc64_opt_t4" / "timing_receipt.json"
    if not timing_receipt.is_file():
        raise RuntimeError("uninterrupted rc64_opt_t4 receipt missing")
    timing = json.loads(timing_receipt.read_text())
    if not timing.get("complete") or not timing.get("wall_seconds_is_full_n600"):
        raise RuntimeError("rc64_opt_t4 receipt is not full-n600 admissible")
    output = BULK / "route_b_end_to_end" / "inflated"
    output.mkdir(parents=True, exist_ok=True)
    raw = output / "0.raw"
    log = CONTROL / "receipts" / "route_b_end_to_end.log"
    route_archive = CONTROL / "route_b" / "runtime_archive"
    verified_copy(
        CONTROL / "route_b" / "submission" / "archive" / "p",
        route_archive / "p",
    )
    native_build = BULK / "route_b_end_to_end" / "native-build"
    env = os.environ.copy()
    env.update(
        PYTHON=str(PYTHON),
        PR130_RUNTIME_DEPS_DIR=str(
            BULK / "route_b_end_to_end" / "runtime-deps"
        ),
        PR130_TOKEN_CACHE=str(
            BULK / "runs" / "rc64_opt_t4" / "tokens.npz"
        ),
        PR130_TOKEN_RECEIPT=str(
            BULK / "runs" / "rc64_opt_t4" / "tokens_receipt.json"
        ),
        LC2_NATIVE_ANS_MODE="off",
        LC2_RC64_MODE="required",
        LC2_NATIVE_BUILD_DIR=str(native_build),
        OMP_NUM_THREADS="4",
        MKL_NUM_THREADS="4",
        PR130_BROTLI_CLI="/opt/homebrew/bin/brotli",
    )
    command = [
        str(CONTROL / "runtime_optimized" / "inflate.sh"),
        str(route_archive),
        str(output),
        str(CONTROL / "runtime_optimized" / "video_names.txt"),
    ]
    started = time.perf_counter()
    run = subprocess.run(command, env=env, text=True, capture_output=True, check=False)
    elapsed = time.perf_counter() - started
    atomic_bytes(log, (run.stdout + run.stderr).encode())
    if run.returncode != 0:
        raise RuntimeError(f"literal Route-B receiver failed; see {log}")
    require_file(raw, expected_bytes=RAW_BYTES, expected_sha256=RAW_SHA256)
    result = {
        "schema": "ddm_rc64p_route_b_end_to_end.v1",
        "complete": True,
        "written_at_utc": utc_now(),
        "axis": "[macOS-CPU literal Route-B inflate.sh parseback; scorer-free; n600]",
        "score_claim": False,
        "wall_seconds": elapsed,
        "command": command,
        "environment_contract": {
            "LC2_NATIVE_ANS_MODE": "off",
            "LC2_RC64_MODE": "required",
            "OMP_NUM_THREADS": "4",
            "MKL_NUM_THREADS": "4",
        },
        "native_decode_proof": file_fact(timing_receipt),
        "token_cache_reused": True,
        "archive": file_fact(
            CONTROL / "route_b" / "submission" / "archive.zip"
        ),
        "archive_member": file_fact(route_archive / "p"),
        "raw": file_fact(raw),
        "native_rc64_library": file_fact(native_build / "liblc2_rc64.dylib"),
        "log": file_fact(log),
    }
    atomic_json(CONTROL / "receipts" / "route_b_end_to_end.json", result)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return result


def summarize() -> dict[str, Any]:
    cells: dict[str, dict[str, Any]] = {}
    for decoder in ("constriction", "native"):
        for threads in (1, 4):
            name = f"{decoder}_t{threads}"
            path = CONTROL / "runs" / name / "timing_receipt.json"
            if not path.is_file():
                raise RuntimeError(f"missing timing cell: {name}")
            cells[name] = json.loads(path.read_text())
            if not cells[name].get("complete"):
                raise RuntimeError(f"incomplete timing cell: {name}")
    comparisons = {}
    for threads in (1, 4):
        baseline = cells[f"constriction_t{threads}"]
        native = cells[f"native_t{threads}"]
        comparisons[f"t{threads}"] = {
            "constriction_wall_seconds": baseline["wall_seconds"],
            "native_wall_seconds": native["wall_seconds"],
            "wall_speedup": baseline["wall_seconds"] / native["wall_seconds"],
            "wall_seconds_saved": baseline["wall_seconds"] - native["wall_seconds"],
            "constriction_entropy_call_seconds": baseline["entropy_call_seconds"],
            "native_entropy_call_seconds": native["entropy_call_seconds"],
            "entropy_call_speedup": (
                baseline["entropy_call_seconds"] / native["entropy_call_seconds"]
            ),
            "native_under_120_seconds": native["wall_seconds"] < 120.0,
            "native_under_900_seconds": native["wall_seconds"] < 900.0,
            "native_under_1500_seconds": native["wall_seconds"] < 1500.0,
        }
    verdict = (
        "adopt"
        if comparisons["t4"]["native_under_1500_seconds"]
        and comparisons["t4"]["wall_seconds_saved"] > 0.0
        else "refuse_native_entropy_as_cpu_cure"
    )
    result = {
        "schema": "ddm_rc64p_summary.v1",
        "complete": True,
        "written_at_utc": utc_now(),
        "axis": AXIS,
        "score_claim": False,
        "cells": cells,
        "comparisons": comparisons,
        "verdict": verdict,
        "charter_predictions": {
            "route_a_expected_seconds": "under 120",
            "route_b_expected_seconds": "under 180",
            "falsifier_seconds": 900,
            "honest_margin_gate_seconds": 1500,
            "contest_budget_seconds": 1800,
        },
        "mechanism_boundary": (
            "entropy_call_seconds measures only ANS symbol recovery; wall_seconds "
            "also includes the serial causal HPAC probability-generation loop"
        ),
    }
    atomic_json(CONTROL / "receipts" / "summary.json", result)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return result


def finalize() -> dict[str, Any]:
    route_a = summarize()
    route_b_encode_receipt = CONTROL / "route_b" / "encode_receipt.json"
    literal_receipt = CONTROL / "receipts" / "route_b_end_to_end.json"
    paths = {
        "rc64_t1": CONTROL / "runs" / "rc64_clean_t1" / "timing_receipt.json",
        "rc64_t4": CONTROL / "runs" / "rc64_clean_t4" / "timing_receipt.json",
        "rc64_opt_t1": CONTROL / "runs" / "rc64_opt_t1" / "timing_receipt.json",
        "rc64_opt_t4": CONTROL / "runs" / "rc64_opt_t4" / "timing_receipt.json",
    }
    for path in (route_b_encode_receipt, literal_receipt, *paths.values()):
        if not path.is_file():
            raise RuntimeError(f"finalize is missing {path}")
    route_b_encode_result = json.loads(route_b_encode_receipt.read_text())
    literal = json.loads(literal_receipt.read_text())
    cells = {name: json.loads(path.read_text()) for name, path in paths.items()}
    for name, cell in cells.items():
        if (
            not cell.get("complete")
            or not cell.get("wall_seconds_is_full_n600")
            or cell["decoded_tokens"]["sha256"] != DECODED_TOKEN_SHA256
            or not cell.get("entropy_final_state_empty")
        ):
            raise RuntimeError(f"inadmissible final cell: {name}")
    component_sum = cells["rc64_t4"]["wall_seconds"] + literal["wall_seconds"]
    rate_delta = 25.0 * route_b_encode_result["archive_bytes_saved"] / 37_545_489
    result = {
        "schema": "ddm_rc64p_final_summary.v1",
        "complete": True,
        "written_at_utc": utc_now(),
        "score_claim": False,
        "axis": AXIS,
        "route_a": {
            "verdict": route_a["verdict"],
            "summary_receipt": file_fact(CONTROL / "receipts" / "summary.json"),
        },
        "route_b": {
            "verdict": "adopt_rate_only_receiver_closed_candidate",
            "archive": route_b_encode_result["archive"],
            "token_payload": route_b_encode_result["token_payload"],
            "archive_bytes_saved": route_b_encode_result["archive_bytes_saved"],
            "cells": cells,
            "literal_parseback": literal,
            "component_sum_projection_seconds": component_sum,
            "component_sum_projection_margin_to_1800_seconds": 1800.0 - component_sum,
            "component_sum_projection_boundary": (
                "sum of separate token and cached-token render cells, not one "
                "literal uncached contest run"
            ),
            "lc2_score_anchor": 0.16959899569230852,
            "rate_only_projected_score": 0.16959899569230852 - rate_delta,
            "rate_only_projected_score_delta": -rate_delta,
            "score_projection_boundary": (
                "raw identity proves semantic equality, but upstream/evaluate.py "
                "was not run on Route-B archive bytes"
            ),
        },
        "cached_hpac_formulation": {
            "verdict": "refuse_instance_no_speed_cure",
            "verdict_scope": "INSTANCE",
            "one_run_per_cell_no_noise_floor": True,
        },
        "mechanism": (
            "entropy recovery is under 0.5 percent of every full token cell; "
            "causal sparse HPAC probability generation dominates"
        ),
        "timing_boundary": (
            "cell order showed host drift; report observations separately and do "
            "not claim stable cross-cell speed ordering without repeats"
        ),
        "contest_cpu_authority": "not_measured_modal_dispatch_forbidden",
        "scorer": "not_run_charter_owned_no_scorer_slot",
    }
    atomic_json(CONTROL / "receipts" / "final_summary.json", result)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("prepare")
    commands.add_parser("golden")
    worker_parser = commands.add_parser("worker")
    worker_parser.add_argument(
        "--decoder",
        choices=("constriction", "native", "rc64", "rc64_opt"),
        required=True,
    )
    worker_parser.add_argument("--threads", type=int, choices=(1, 4), required=True)
    worker_parser.add_argument("--run-tag", default="")
    commands.add_parser("bootstrap-smoke")
    commands.add_parser("end-to-end")
    commands.add_parser("summarize")
    commands.add_parser("route-b-encode")
    commands.add_parser("prepare-optimized-runtime")
    commands.add_parser("optimized-parity-smoke")
    commands.add_parser("route-b-end-to-end")
    commands.add_parser("finalize")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "prepare":
        prepare()
    elif args.command == "golden":
        golden()
    elif args.command == "worker":
        worker(args.decoder, args.threads, args.run_tag)
    elif args.command == "bootstrap-smoke":
        bootstrap_smoke()
    elif args.command == "end-to-end":
        end_to_end()
    elif args.command == "summarize":
        summarize()
    elif args.command == "route-b-encode":
        route_b_encode()
    elif args.command == "prepare-optimized-runtime":
        prepare_optimized_runtime()
    elif args.command == "optimized-parity-smoke":
        optimized_parity_smoke()
    elif args.command == "route-b-end-to-end":
        route_b_end_to_end()
    elif args.command == "finalize":
        finalize()
    else:  # pragma: no cover - argparse enforces the command set.
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
