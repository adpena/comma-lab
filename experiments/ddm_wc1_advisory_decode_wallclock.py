#!/usr/bin/env python3
"""Build and measure the identity-gated WC1 advisory F26 fast path.

This runner never mutates a source generation.  It copies one generation into
an APDataStore work root, installs the default-off native/cache/parallel
advisory components, retains every token/render/raw payload, and emits a
machine-readable receipt.  Full n600 invocations are launched by the canonical
watched detached launcher; this file is the resumable child entrypoint.
"""

from __future__ import annotations

import argparse
import fcntl
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

REPO = Path(__file__).resolve().parents[1]
ROOT = Path("/Volumes/APDataStore/pact/ddm_wc1_advisory_decode_wallclock_20260815")
CACHE_ROOT = ROOT / "cache/tokens"
NATIVE_ROOT = ROOT / "retained/native"
BASE_GENERATION = Path(
    "/Volumes/APDataStore/pact/ddm_mp2_mixed_precision_receiver_close_20260815/"
    "generations/hv1_base_control"
)
REFERENCE_ROOT = Path("/Volumes/APDataStore/pact/ddm_hv1_base_advisory_n600_cpu/work_r2")
REFERENCE_RAW = REFERENCE_ROOT / "inflated/0.raw"
REFERENCE_TOKENS = REFERENCE_ROOT / "inflated/.f26_decode_checkpoints/tokens_cpu_stage_complete.u8"
OPTIMIZED_NATIVE_SOURCE = Path(
    "/Volumes/VertigoDataTier/pact/ddm_f26r_hpac_final_rung_20260814/"
    "rungs/direct_context_delta_v1/retained/native_build_direct_context_delta/"
    "f26_hpac_native_a.so"
)
SCALAR_NATIVE_SOURCE = Path(
    "/Volumes/VertigoDataTier/pact/ddm_f26r_hpac_final_rung_20260814/"
    "rungs/direct_context_delta_v1/retained/native_build_scalar_twin/"
    "f26_hpac_scalar_a.so"
)

BASE_ARCHIVE_SHA256 = "80d9c8c6fdc72caaa3e180a8abb2a859e7f316a484b38f33fe90d5701420178e"
BASE_ARCHIVE_BYTES = 182_759
BASE_RUNTIME_F26_SHA256 = "4718834fe2e589f4be998061a8d9cba552ba4814b584e0efcc77a41aa2cb6680"
REFERENCE_RAW_SHA256 = "e5539653f598a1c31e28900888f450a6de019cb29864674f232ad2f8956b15c9"
REFERENCE_TOKEN_SHA256 = "9ba2e52b3096585895970066b389bf1261ebc203d5b828cdea056c13858aea52"
REFERENCE_CORRECTED_SHA256 = "562ac652b372faa020d0fc5e2ed9b7b61625169e0f5c2041d4fe99196055b8c7"
REFERENCE_CDF_SHA256 = "dd48843b021763e78524caf3dcd01e944045e7bd0ffd93b451dec83548f083b7"
REFERENCE_BIT_POSITION = 896_939
OPTIMIZED_NATIVE_SHA256 = "1cf0e61b53d5b25a2b0cbb6adb47232921ebd442aa461cfcbb8db97d664a6aae"
SCALAR_NATIVE_SHA256 = "64efe1e803aa0d22dbb0e3d02df5e7799a2e76b7ae4298311e78ab96cc86f4a8"
FRAME_BYTES = 874 * 1164 * 3
PAIR_TOKEN_BYTES = 384 * 512
FULL_PAIRS = 600
FULL_RAW_BYTES = FULL_PAIRS * 2 * FRAME_BYTES
FULL_TOKEN_BYTES = FULL_PAIRS * PAIR_TOKEN_BYTES


class WC1RunError(RuntimeError):
    """A WC1 provenance, launch, retention, or identity gate failed."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_fact(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": _sha256(path)}


def _canonical_json(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(_canonical_json(value))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _tree_manifest(root: Path) -> list[dict[str, Any]]:
    return [
        {
            "relative_path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in sorted(root.rglob("*"))
        if path.is_file()
        and not path.name.startswith("._")
        and "__pycache__" not in path.parts
        and path.suffix != ".pyc"
    ]


def _manifest_sha256(records: list[dict[str, Any]]) -> str:
    return hashlib.sha256(
        json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise WC1RunError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise WC1RunError(f"JSON root must be an object: {path}")
    return value


def _replace_exact(path: Path, old: str, new: str) -> None:
    source = path.read_text(encoding="utf-8")
    if source.count(old) != 1:
        raise WC1RunError(f"expected exactly one WC1 patch point in {path}")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(source.replace(old, new), encoding="utf-8")
    os.replace(temporary, path)


def _copy_ignore(_root: str, names: list[str]) -> set[str]:
    return {
        name
        for name in names
        if name.startswith("._") or name == "__pycache__" or name.endswith(".pyc")
    }


def _sweep_appledouble(root: Path) -> list[str]:
    removed: list[str] = []
    for path in sorted(root.rglob("._*")):
        if path.is_file():
            removed.append(path.relative_to(root).as_posix())
            path.unlink()
    return removed


def _verify_reference_payloads() -> None:
    facts = (
        (REFERENCE_RAW, FULL_RAW_BYTES, REFERENCE_RAW_SHA256),
        (REFERENCE_TOKENS, FULL_TOKEN_BYTES, REFERENCE_TOKEN_SHA256),
    )
    for path, expected_bytes, expected_sha in facts:
        if not path.is_file() or path.stat().st_size != expected_bytes or _sha256(path) != expected_sha:
            raise WC1RunError(f"retained hv1 reference payload differs: {path}")


def bootstrap_native_assets() -> dict[str, Any]:
    ROOT.mkdir(parents=True, exist_ok=True)
    NATIVE_ROOT.mkdir(parents=True, exist_ok=True)
    rows = []
    for source, name, expected_sha in (
        (OPTIMIZED_NATIVE_SOURCE, "f26_hpac_native_optimized.so", OPTIMIZED_NATIVE_SHA256),
        (SCALAR_NATIVE_SOURCE, "f26_hpac_native_scalar.so", SCALAR_NATIVE_SHA256),
    ):
        if not source.is_file() or _sha256(source) != expected_sha:
            raise WC1RunError(f"F26R native source asset differs: {source}")
        destination = NATIVE_ROOT / name
        if destination.exists():
            if _sha256(destination) != expected_sha:
                raise WC1RunError(f"WC1 retained native asset differs: {destination}")
        else:
            temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
            with source.open("rb") as src, temporary.open("wb") as dst:
                shutil.copyfileobj(src, dst, length=1 << 20)
                dst.flush()
                os.fsync(dst.fileno())
            os.replace(temporary, destination)
        rows.append({"source": _file_fact(source), "retained": _file_fact(destination)})
    receipt = {
        "schema": "ddm_wc1_native_assets.v1",
        "complete": True,
        "assets": rows,
        "payload_cleanliness_authority": (
            "/Volumes/VertigoDataTier/pact/ddm_f26r_hpac_final_rung_20260814/"
            "rungs/direct_context_delta_v1/submission_native_sealed/runtime/"
            "archive_payload_manifest.json"
        ),
    }
    _atomic_json(ROOT / "receipts/native_assets.json", receipt)
    return receipt


def _archive_fact(generation: Path) -> dict[str, Any]:
    archive = generation / "archive.zip"
    if not archive.is_file():
        raise WC1RunError(f"generation archive is absent: {archive}")
    with zipfile.ZipFile(archive) as handle:
        if handle.namelist() != ["p"] or handle.testzip() is not None:
            raise WC1RunError(f"generation archive is not the exact single-member F26 form: {archive}")
    return _file_fact(archive)


def prepare_advisory_runtime(source_generation: Path, destination: Path) -> dict[str, Any]:
    """Copy one generation and install default-off advisory-only components."""

    source_generation = source_generation.resolve()
    destination = destination.resolve()
    archive = _archive_fact(source_generation)
    source_f26 = source_generation / "runtime/f26_inflate.py"
    if not source_f26.is_file() or _sha256(source_f26) != BASE_RUNTIME_F26_SHA256:
        raise WC1RunError(
            "source generation is not on the identity-pinned F26P runtime; "
            f"observed {None if not source_f26.is_file() else _sha256(source_f26)}"
        )
    receipt_path = destination.parent / f"{destination.name}.prepare.json"
    if destination.exists():
        if not receipt_path.is_file():
            raise WC1RunError(f"advisory runtime exists without receipt: {destination}")
        _sweep_appledouble(destination)
        receipt = _read_json(receipt_path)
        manifest = _tree_manifest(destination)
        if _manifest_sha256(manifest) != receipt["advisory_runtime"]["manifest_sha256"]:
            raise WC1RunError("existing advisory runtime differs from its receipt")
        return receipt

    temporary = destination.parent / f".{destination.name}.{os.getpid()}.building"
    if temporary.exists():
        raise WC1RunError(f"stale advisory runtime partial blocks preparation: {temporary}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source_generation, temporary, ignore=_copy_ignore)
    shutil.copy2(REPO / "experiments/ddm_f26p_f26_inflate_cpu.py", temporary / "runtime/f26_inflate.py")
    shutil.copy2(
        REPO / "experiments/ddm_wc1_advisory_runtime.py",
        temporary / "runtime/ddm_wc1_advisory_runtime.py",
    )
    shutil.copy2(
        REPO / "experiments/ddm_f26q_f26_hpac_native.py",
        temporary / "runtime/f26_hpac_native.py",
    )
    shutil.copy2(
        REPO / "runtime-rs/native/f26-hpac/f26_hpac_native.c",
        temporary / "runtime/f26_hpac_native.c",
    )
    inflate_sh = temporary / "inflate.sh"
    _replace_exact(
        inflate_sh,
        'export CPR1_RC64_LIBRARY="$BUILD_DIR/rc64_backend.so"\n',
        'export CPR1_RC64_LIBRARY="$BUILD_DIR/rc64_backend.so"\n'
        'export F26_TOKEN_DECODER="${F26_TOKEN_DECODER:-python}"\n'
        'if [[ "$F26_TOKEN_DECODER" == "native-hpac" ]]; then\n'
        '  if [[ -n "${F26_HPAC_NATIVE_LIBRARY:-}" ]]; then\n'
        '    [[ -f "$F26_HPAC_NATIVE_LIBRARY" ]] || { echo "missing F26 native library" >&2; exit 69; }\n'
        '  else\n'
        '    case "$(uname -s)" in\n'
        '      Darwin)\n'
        '        LIBOMP_PREFIX="$(brew --prefix libomp)"\n'
        '        "${CC:-cc}" -O3 -mcpu=native -std=c11 -shared -fPIC -ffp-contract=off -fno-fast-math \\\n'
        '          -Xpreprocessor -fopenmp -I"$LIBOMP_PREFIX/include" \\\n'
        '          "$HERE/runtime/f26_hpac_native.c" -L"$LIBOMP_PREFIX/lib" -lomp \\\n'
        '          -Wl,-rpath,"$LIBOMP_PREFIX/lib" -lm -o "$BUILD_DIR/f26_hpac_native.so" ;;\n'
        '      *)\n'
        '        "${CC:-cc}" -O3 -march=native -std=c11 -shared -fPIC -ffp-contract=off -fno-fast-math \\\n'
        '          -fopenmp "$HERE/runtime/f26_hpac_native.c" -lm -o "$BUILD_DIR/f26_hpac_native.so" ;;\n'
        '    esac\n'
        '    export F26_HPAC_NATIVE_LIBRARY="$BUILD_DIR/f26_hpac_native.so"\n'
        '  fi\n'
        'fi\n',
    )
    appledouble_removed = _sweep_appledouble(temporary)
    os.replace(temporary, destination)
    manifest = _tree_manifest(destination)
    receipt = {
        "schema": "ddm_wc1_advisory_runtime_prepare.v1",
        "complete": True,
        "source_generation": {
            "root": str(source_generation),
            "archive": archive,
            "f26_runtime": _file_fact(source_f26),
        },
        "advisory_runtime": {
            "root": str(destination),
            "manifest_sha256": _manifest_sha256(manifest),
            "files": manifest,
        },
        "defaults": {
            "F26_TOKEN_DECODER": "python",
            "F26_ADVISORY_RENDER_WORKERS": "unset serial path",
            "F26_ADVISORY_DECODE_CACHE_ROOT": "unset disabled",
            "shipping_packet_touched": False,
        },
        "appledouble_sweep": {
            "scope": str(destination),
            "removed_from_copy": appledouble_removed,
            "post_sweep_count": len(list(destination.rglob("._*"))),
        },
        "source_pin": {
            "charter_literal": "4718834f",
            "verified_full_sha256": BASE_RUNTIME_F26_SHA256,
            "basis": "canonical MP2 advisory queue generation bytes",
        },
    }
    _atomic_json(receipt_path, receipt)
    return receipt


def _extract_input(stage: Path, input_dir: Path) -> Path:
    payload = input_dir / "p"
    if payload.is_file():
        with zipfile.ZipFile(stage / "archive.zip") as handle:
            member = handle.read("p")
        if payload.stat().st_size != len(member) or _sha256(payload) != hashlib.sha256(member).hexdigest():
            raise WC1RunError("retained extracted input differs from the staged archive")
        return payload
    input_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(stage / "archive.zip") as handle:
        data = handle.read("p")
    temporary = payload.with_name(f".{payload.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, payload)
    return payload


def _parse_inflate_report(log_path: Path) -> dict[str, Any]:
    for line in reversed(log_path.read_text(encoding="utf-8", errors="replace").splitlines()):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and value.get("schema") == "ddm_f26p_inflate_report.v1":
            return value
    raise WC1RunError(f"inflate log has no complete F26P report: {log_path}")


def _sha256_prefix(path: Path, length: int) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        remaining = length
        while remaining:
            chunk = handle.read(min(1 << 20, remaining))
            if not chunk:
                raise WC1RunError(f"payload ended inside requested prefix: {path}")
            digest.update(chunk)
            remaining -= len(chunk)
    return digest.hexdigest()


def _frame_manifest(raw: Path, pair_count: int, destination: Path) -> dict[str, Any]:
    rows = []
    with raw.open("rb") as handle:
        for frame_index in range(pair_count * 2):
            frame = handle.read(FRAME_BYTES)
            if len(frame) != FRAME_BYTES:
                raise WC1RunError(f"raw output ended inside frame {frame_index}")
            rows.append(
                {
                    "frame_index": frame_index,
                    "pair_index": frame_index // 2,
                    "within_pair": frame_index % 2,
                    "bytes": FRAME_BYTES,
                    "sha256": hashlib.sha256(frame).hexdigest(),
                }
            )
        if handle.read(1):
            raise WC1RunError("raw output has bytes beyond the declared pair count")
    receipt = {
        "schema": "ddm_wc1_frame_manifest.v1",
        "pair_count": pair_count,
        "frame_count": pair_count * 2,
        "raw": _file_fact(raw),
        "frames": rows,
    }
    _atomic_json(destination, receipt)
    return receipt


def run_decode(
    *,
    source_generation: Path,
    run_id: str,
    pair_count: int,
    native_variant: str,
    cache_mode: str,
    workers: str,
    reference_raw: Path | None,
    compare_result: Path | None,
) -> dict[str, Any]:
    if not 1 <= pair_count <= FULL_PAIRS:
        raise WC1RunError("pair_count must be within the real n600 field")
    if native_variant not in {"optimized", "scalar"}:
        raise WC1RunError("native_variant must be optimized or scalar")
    if cache_mode not in {"disabled", "read-write"}:
        raise WC1RunError("cache_mode must be disabled or read-write")
    if pair_count != FULL_PAIRS and cache_mode != "disabled":
        raise WC1RunError("prefix runs cannot touch the full-field token cache")
    _verify_reference_payloads()
    native_assets = bootstrap_native_assets()
    run_root = ROOT / "runs" / run_id
    result_path = run_root / "result.json"
    if result_path.is_file():
        result = _read_json(result_path)
        raw = Path(result["raw_output"]["path"])
        if not raw.is_file() or _file_fact(raw) != result["raw_output"]:
            raise WC1RunError(f"retained run result changed: {run_id}")
        return result
    required_bytes = pair_count * 2 * FRAME_BYTES * 2 + 2 * pair_count * PAIR_TOKEN_BYTES + (1 << 30)
    free_bytes = shutil.disk_usage(ROOT).free
    if free_bytes < required_bytes:
        raise WC1RunError(f"APDataStore has {free_bytes} free bytes; run requires {required_bytes}")
    run_root.mkdir(parents=True, exist_ok=True)
    stage = run_root / "advisory_generation"
    prepare_receipt = prepare_advisory_runtime(source_generation, stage)
    input_dir = run_root / "input"
    output_dir = run_root / "output"
    log_path = run_root / "inflate.log"
    file_list = run_root / "file_list.txt"
    _extract_input(stage, input_dir)
    _atomic_text(file_list, "0.mkv\n")
    native_path = NATIVE_ROOT / (
        "f26_hpac_native_optimized.so"
        if native_variant == "optimized"
        else "f26_hpac_native_scalar.so"
    )
    environment = os.environ.copy()
    environment.update(
        {
            "PATH": os.pathsep.join([str(REPO / ".venv/bin"), environment.get("PATH", "")]),
            "PYTHONDONTWRITEBYTECODE": "1",
            "F26_TOKEN_DECODER": "native-hpac",
            "F26_HPAC_NATIVE_LIBRARY": str(native_path),
            "F26_ADVISORY_RENDER_WORKERS": workers,
            "F26_ADVISORY_PAIR_LIMIT": str(pair_count),
        }
    )
    if cache_mode == "read-write":
        environment["F26_ADVISORY_DECODE_CACHE_ROOT"] = str(CACHE_ROOT)
    command = [str(stage / "inflate.sh"), str(input_dir), str(output_dir), str(file_list)]
    launch_appledouble_removed = _sweep_appledouble(stage)
    if list(stage.rglob("._*")):
        raise WC1RunError("AppleDouble sweep did not leave the copied runtime clean")
    if (run_root / "failure.json").exists():
        raise WC1RunError(f"terminal failure receipt requires an explicit new run_id: {run_root}")
    launch_state_path = run_root / "launch_state.json"
    previous_launch = _read_json(launch_state_path) if launch_state_path.is_file() else None
    started_unix = time.time()
    _atomic_json(
        launch_state_path,
        {
            "schema": "ddm_wc1_decode_launch_state.v1",
            "status": "running",
            "started_unix": started_unix,
            "resume_of": previous_launch,
            "command": command,
            "appledouble_removed_before_launch": launch_appledouble_removed,
        },
    )
    started = time.perf_counter()
    log_mode = "a" if log_path.exists() else "w"
    with log_path.open(log_mode, encoding="utf-8") as log:
        log.write(
            _canonical_json(
                {
                    "schema": "ddm_wc1_decode_launch.v1",
                    "command": command,
                    "environment": {
                        key: environment[key]
                        for key in (
                            "PYTHONDONTWRITEBYTECODE",
                            "F26_TOKEN_DECODER",
                            "F26_HPAC_NATIVE_LIBRARY",
                            "F26_ADVISORY_RENDER_WORKERS",
                            "F26_ADVISORY_PAIR_LIMIT",
                            "F26_ADVISORY_DECODE_CACHE_ROOT",
                        )
                        if key in environment
                    },
                }
            )
        )
        log.flush()
        process = subprocess.Popen(
            command,
            cwd=REPO,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log.write(line)
            log.flush()
        returncode = process.wait()
    wall_seconds = time.perf_counter() - started
    if returncode != 0:
        failure = {
            "schema": "ddm_wc1_decode_failure.v1",
            "complete": False,
            "run_id": run_id,
            "returncode": returncode,
            "wall_seconds": wall_seconds,
            "log": _file_fact(log_path),
            "retention": "all token/render/raw partials remain under the run root",
        }
        _atomic_json(run_root / "failure.json", failure)
        raise WC1RunError(f"WC1 inflate exited {returncode}; payloads retained at {run_root}")
    _atomic_json(
        launch_state_path,
        {
            "schema": "ddm_wc1_decode_launch_state.v1",
            "status": "complete",
            "started_unix": started_unix,
            "subprocess_wall_seconds": wall_seconds,
            "returncode": returncode,
            "command": command,
            "appledouble_removed_before_launch": launch_appledouble_removed,
        },
    )
    raw = output_dir / "0.raw"
    expected_raw_bytes = pair_count * 2 * FRAME_BYTES
    if not raw.is_file() or raw.stat().st_size != expected_raw_bytes:
        raise WC1RunError("WC1 inflate did not retain the declared raw payload")
    report = _parse_inflate_report(log_path)
    if report.get("raw_sha256") != _sha256(raw) or report.get("pair_count") != pair_count:
        raise WC1RunError("inflate report differs from the retained raw payload or pair count")
    checkpoint_dir = output_dir / ".f26_decode_checkpoints"
    tokens = checkpoint_dir / "tokens_cpu_stage_complete.u8"
    if not tokens.is_file() or tokens.stat().st_size != pair_count * PAIR_TOKEN_BYTES:
        raise WC1RunError("WC1 run did not retain the complete token-stage payload")
    token_prefix_sha = _sha256_prefix(REFERENCE_TOKENS, tokens.stat().st_size)
    archive_sha = _sha256(stage / "archive.zip")
    raw_reference = (
        REFERENCE_RAW
        if reference_raw is None and archive_sha == BASE_ARCHIVE_SHA256
        else None if reference_raw is None else reference_raw.resolve()
    )
    raw_prefix_sha = (
        None if raw_reference is None else _sha256_prefix(raw_reference, raw.stat().st_size)
    )
    frame_manifest_path = run_root / "frame_manifest.json"
    frame_manifest = _frame_manifest(raw, pair_count, frame_manifest_path)
    comparison = None
    if compare_result is not None:
        other = _read_json(compare_result.resolve())
        comparison = {
            "result": _file_fact(compare_result.resolve()),
            "raw_sha256_equal": other["raw_output"]["sha256"] == _sha256(raw),
            "raw_bytes_equal": other["raw_output"]["bytes"] == raw.stat().st_size,
            "token_sha256_equal": other["token_output"]["sha256"] == _sha256(tokens),
        }
        if not all(comparison[key] for key in comparison if key.endswith("_equal")):
            raise WC1RunError("cached-vs-fresh retained payload parity failed")
    token_report = report.get("token_decoder")
    if not isinstance(token_report, dict):
        raise WC1RunError("inflate report has no token-decoder receipt")
    identity = {
        "raw_prefix_sha256_expected": raw_prefix_sha,
        "raw_sha256_observed": _sha256(raw),
        "raw_identity_pass": None if raw_prefix_sha is None else raw_prefix_sha == _sha256(raw),
        "token_prefix_sha256_expected": token_prefix_sha,
        "token_sha256_observed": _sha256(tokens),
        "token_identity_pass": token_prefix_sha == _sha256(tokens),
    }
    if identity["raw_identity_pass"] is False or not identity["token_identity_pass"]:
        raise WC1RunError("WC1 retained payload differs from the hv1 reference prefix")
    full_token_gates = None
    if pair_count == FULL_PAIRS:
        full_token_gates = {
            "decoded_token_sha256": token_report.get("decoded_token_sha256")
            == REFERENCE_TOKEN_SHA256,
            "corrected_quantized_logit_sha256": token_report.get(
                "corrected_quantized_logit_sha256"
            )
            == REFERENCE_CORRECTED_SHA256,
            "corrected_cdf_input_sha256": token_report.get("corrected_cdf_input_sha256")
            == REFERENCE_CDF_SHA256,
            "decoder_bit_position": token_report.get("decoder_bit_position")
            == REFERENCE_BIT_POSITION,
        }
        if not all(full_token_gates.values()):
            raise WC1RunError(f"full hv1 native token gates failed: {full_token_gates}")
    result = {
        "schema": "ddm_wc1_decode_result.v1",
        "complete": True,
        "run_id": run_id,
        "axis_label": "[M5-CPU scorer-free advisory decode]",
        "score_claim": False,
        "started_unix": started_unix,
        "subprocess_wall_seconds": wall_seconds,
        "pair_count": pair_count,
        "native_variant": native_variant,
        "cache_mode": cache_mode,
        "requested_workers": workers,
        "source_generation": str(source_generation.resolve()),
        "archive": _file_fact(stage / "archive.zip"),
        "advisory_runtime_prepare": prepare_receipt,
        "native_assets": native_assets,
        "environment": {
            key: environment[key]
            for key in (
                "PYTHONDONTWRITEBYTECODE",
                "F26_TOKEN_DECODER",
                "F26_HPAC_NATIVE_LIBRARY",
                "F26_ADVISORY_RENDER_WORKERS",
                "F26_ADVISORY_PAIR_LIMIT",
                "F26_ADVISORY_DECODE_CACHE_ROOT",
            )
            if key in environment
        },
        "host": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "cpu_count": os.cpu_count(),
        },
        "inflate_report": report,
        "raw_output": _file_fact(raw),
        "token_output": _file_fact(tokens),
        "render_stage": _file_fact(checkpoint_dir / "parallel_render/render_stage.raw"),
        "frame_manifest": _file_fact(frame_manifest_path),
        "frame_count": frame_manifest["frame_count"],
        "identity": identity,
        "full_token_gates": full_token_gates,
        "comparison": comparison,
        "log": _file_fact(log_path),
        "storage_preflight": {"free_bytes": free_bytes, "required_bytes": required_bytes},
        "appledouble_removed_before_launch": launch_appledouble_removed,
    }
    _atomic_json(result_path, result)
    return result


def run_decode_locked(**kwargs: Any) -> dict[str, Any]:
    run_id = str(kwargs["run_id"])
    locks = ROOT / "runs"
    locks.mkdir(parents=True, exist_ok=True)
    lock_path = locks / f".{run_id}.singleflight.lock"
    with lock_path.open("a+b") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise WC1RunError(f"run_id already has a live producer: {run_id}") from exc
        return run_decode(**kwargs)


def finalize(
    *,
    prefix_n4: Path,
    prefix_n32: Path,
    base_optimized: Path,
    base_scalar: Path,
    micro_fresh: Path,
    micro_cached: Path,
) -> dict[str, Any]:
    rows = {
        "prefix_n4": _read_json(prefix_n4.resolve()),
        "prefix_n32": _read_json(prefix_n32.resolve()),
        "base_optimized": _read_json(base_optimized.resolve()),
        "base_scalar": _read_json(base_scalar.resolve()),
        "micro_fresh": _read_json(micro_fresh.resolve()),
        "micro_cached": _read_json(micro_cached.resolve()),
    }
    blockers: list[str] = []
    for name in ("base_optimized", "base_scalar", "micro_fresh", "micro_cached"):
        row = rows[name]
        gates = row.get("full_token_gates")
        if row.get("pair_count") != FULL_PAIRS or not isinstance(gates, dict) or not all(gates.values()):
            blockers.append(f"{name}:full_token_identity")
    for name, expected_pairs in (("prefix_n4", 4), ("prefix_n32", 32)):
        row = rows[name]
        if (
            row.get("pair_count") != expected_pairs
            or row.get("identity", {}).get("raw_identity_pass") is not True
            or row.get("identity", {}).get("token_identity_pass") is not True
        ):
            blockers.append(f"{name}:prefix_identity")
    if rows["base_optimized"]["native_variant"] != "optimized":
        blockers.append("base_optimized:variant")
    if rows["base_scalar"]["native_variant"] != "scalar":
        blockers.append("base_scalar:variant")
    if rows["base_optimized"]["raw_output"]["sha256"] != rows["base_scalar"]["raw_output"]["sha256"]:
        blockers.append("optimized_scalar_raw_parity")
    if rows["base_optimized"]["raw_output"]["sha256"] != REFERENCE_RAW_SHA256:
        blockers.append("base_reference_raw_identity")
    if rows["micro_fresh"]["raw_output"]["sha256"] != rows["micro_cached"]["raw_output"]["sha256"]:
        blockers.append("micro_cached_fresh_raw_parity")
    if rows["micro_fresh"]["token_output"]["sha256"] != rows["micro_cached"]["token_output"]["sha256"]:
        blockers.append("micro_cached_fresh_token_parity")
    base_cache = rows["base_optimized"]["inflate_report"]["token_cache"]
    cached_cache = rows["micro_cached"]["inflate_report"]["token_cache"]
    if base_cache.get("status") not in {"POPULATED", "RACE_HIT_IDENTICAL"}:
        blockers.append("base_cache_population")
    if cached_cache.get("status") != "HIT":
        blockers.append("micro_cache_hit")
    if base_cache.get("key") != cached_cache.get("key"):
        blockers.append("content_addressed_key_reuse")
    cache_parity = {
        "schema": "ddm_wc1_cache_parity.v1",
        "complete": not blockers,
        "candidate_archive": rows["micro_fresh"]["archive"],
        "fresh_result": _file_fact(micro_fresh.resolve()),
        "cached_result": _file_fact(micro_cached.resolve()),
        "token_cache_key": cached_cache.get("key"),
        "raw_sha256": rows["micro_cached"]["raw_output"]["sha256"],
        "token_sha256": rows["micro_cached"]["token_output"]["sha256"],
        "parity_pass": not blockers,
        "blockers": blockers,
    }
    _atomic_json(ROOT / "receipts/cache_parity.json", cache_parity)
    optimized_report = rows["base_optimized"]["inflate_report"]
    admission = {
        "schema": "ddm_wc1_advisory_fast_path_admission.v1",
        "complete": not blockers,
        "identity_pass": not blockers,
        "shipping_packet_touched": False,
        "axis_label": "[M5-CPU scorer-free advisory decode]",
        "base_archive": rows["base_optimized"]["archive"],
        "base_result": _file_fact(base_optimized.resolve()),
        "prefix_n4_result": _file_fact(prefix_n4.resolve()),
        "prefix_n32_result": _file_fact(prefix_n32.resolve()),
        "scalar_twin_result": _file_fact(base_scalar.resolve()),
        "cache_parity": _file_fact(ROOT / "receipts/cache_parity.json"),
        "measured": {
            "subprocess_wall_seconds": rows["base_optimized"]["subprocess_wall_seconds"],
            "inflate_internal_seconds": optimized_report["decode_and_render_seconds"],
            "stage_seconds": optimized_report["stage_seconds"],
            "workers": optimized_report["parallel_render"]["resources"]["selected"],
            "worker_peak_rss_bytes": optimized_report["parallel_render"][
                "worker_peak_rss_bytes"
            ],
            "raw_sha256": rows["base_optimized"]["raw_output"]["sha256"],
        },
        "consumer_environment": {
            "F26_TOKEN_DECODER": "native-hpac",
            "F26_HPAC_NATIVE_LIBRARY": str(
                NATIVE_ROOT / "f26_hpac_native_optimized.so"
            ),
            "F26_ADVISORY_RENDER_WORKERS": "auto",
            "F26_ADVISORY_DECODE_CACHE_ROOT": str(CACHE_ROOT),
            "F26_ADVISORY_RENDER_RSS_BYTES": max(
                int(value)
                for value in optimized_report["parallel_render"]["worker_peak_rss_bytes"].values()
            ),
        },
        "consumer_code": {
            "runtime": _file_fact(REPO / "experiments/ddm_wc1_advisory_runtime.py"),
            "inflate_driver": _file_fact(REPO / "experiments/ddm_f26p_f26_inflate_cpu.py"),
            "builder": _file_fact(REPO / "experiments/ddm_wc1_advisory_decode_wallclock.py"),
            "mp2_queue": _file_fact(REPO / "experiments/ddm_mp2_advisory_queue.py"),
        },
        "blockers": blockers,
    }
    _atomic_json(ROOT / "receipts/ADMISSION_GATE.json", admission)
    if blockers:
        raise WC1RunError(f"WC1 admission gates failed: {blockers}")
    return admission


def write_watcher_configs(
    *,
    run_id: str,
    launcher_dir: Path,
    done_receipt_name: str,
) -> dict[str, Any]:
    config_root = ROOT / "watchers/configs"
    state_root = ROOT / "watchers/state" / run_id
    done_receipt = REPO / ".omx/tmp/codex_runs" / f"{done_receipt_name}.done"
    liveness_path = config_root / f"{run_id}.liveness.json"
    quality_path = config_root / f"{run_id}.quality.json"
    liveness = {
        "schema": "pact.run_liveness_watcher.config.v1",
        "pid_file": str((launcher_dir / "run.pid").resolve()),
        "alert_path": str((state_root / "liveness.alert.json").resolve()),
        "poll_s": 10,
        "initial_delay_s": 0,
        "warmup_s": 900,
        "receipt_checks": [],
        "heartbeat_checks": [],
        "artifact_checks": [
            {
                "label": f"{run_id}_launcher_log",
                "path": str((launcher_dir / "run.log").resolve()),
                "max_age_s": 1200,
                "grace_s": 900,
            }
        ],
        "success_receipts": [
            {"label": done_receipt_name, "path": str(done_receipt.resolve())}
        ],
        "success_settle_s": 30,
    }
    quality = {
        "schema": "pact.run_quality_poller.config.v1",
        "log_path": str((launcher_dir / "run.log").resolve()),
        "pid_file": str((launcher_dir / "run.pid").resolve()),
        "telemetry_path": str((state_root / "quality_telemetry.jsonl").resolve()),
        "alert_path": str((state_root / "quality.alert.json").resolve()),
        "poll_s": 10,
        "eval_period_s": 30,
        "stale_periods": 100,
        "startup_grace_s": 900,
        "json_marker": '"subprocess_wall_seconds"',
        "fields": {
            "epoch": "pair_count",
            "value": "subprocess_wall_seconds",
            "phase": "axis_label",
            "finite": ["subprocess_wall_seconds"],
        },
        "bar_value": 700.0,
        "bar_start_epoch": FULL_PAIRS,
        "alert_conditions": {
            "joint_regression": True,
            "qat_knee_shock": False,
            "nan_or_garbage": True,
            "stale_telemetry": False,
        },
        "regression_bands": [],
        "phase_knee": {
            "epoch": FULL_PAIRS,
            "window_epochs": 1,
            "shock_multiplier": 1.0,
            "continuous_phase": "[M5-CPU scorer-free advisory decode]",
        },
        "best_not_latest": {
            "phase": "[M5-CPU scorer-free advisory decode]",
            "min_rows": 999999,
            "lag_epochs": 999999,
        },
    }
    _atomic_json(liveness_path, liveness)
    _atomic_json(quality_path, quality)
    return {
        "schema": "ddm_wc1_watcher_configs.v1",
        "run_id": run_id,
        "liveness": _file_fact(liveness_path),
        "quality": _file_fact(quality_path),
        "success_receipt": str(done_receipt.resolve()),
    }


def write_full_n600_blocker(
    *,
    launcher_dir: Path,
    run_id: str,
    prefix_n4: Path,
    prefix_n32: Path,
) -> dict[str, Any]:
    run_root = ROOT / "runs" / run_id
    launch_manifest = launcher_dir / "launch_manifest.json"
    safe_status = launcher_dir / "resource_safe_run_status.json"
    governor_log = REPO / ".omx/state/memory_blackbox.daemon.log"
    blackbox = REPO / ".omx/state/memory_blackbox.jsonl"
    if not launch_manifest.is_file() or not safe_status.is_file():
        raise WC1RunError("full-n600 blocker requires the retained watched-launch receipts")
    safe_status_value = _read_json(safe_status)
    safe_status_snapshot = ROOT / "receipts/base_optimized_n600_safe_status_at_blocker.json"
    _atomic_json(safe_status_snapshot, safe_status_value)
    blackbox_rows = [
        json.loads(line)
        for line in blackbox.read_text(encoding="utf-8", errors="replace").splitlines()
        if line.strip()
    ]
    if not blackbox_rows:
        raise WC1RunError("memory blackbox has no rows for the governed-run binding")
    latest_blackbox = blackbox_rows[-1]
    child_pid = safe_status_value.get("child_pid")
    labels: list[str] = []
    for blackbox_row in reversed(blackbox_rows):
        tracked = blackbox_row.get("tracked")
        tracked = tracked if isinstance(tracked, list) else []
        row_labels = [
            row.get("label")
            for row in tracked
            if isinstance(row, dict) and row.get("pid") == child_pid
        ]
        if row_labels:
            labels = [label for label in row_labels if isinstance(label, str)]
            break
    if len(labels) != 1:
        raise WC1RunError("cannot bind the governed run to one historical tracked-job label")
    pause_lines = [
        line
        for line in governor_log.read_text(encoding="utf-8", errors="replace").splitlines()
        if "GOVERNOR PAUSE" in line and repr(labels[0]) in line
    ]
    if not pause_lines:
        raise WC1RunError("did not find this run's exact governor-pause event")
    payload_paths = (
        run_root / "output/0.raw",
        run_root / "output/.f26_decode_checkpoints/tokens_cpu_stage_complete.u8",
        run_root
        / "output/.f26_decode_checkpoints/parallel_render/render_stage.raw",
    )
    materialized = [_file_fact(path) for path in payload_paths if path.is_file()]
    done_path = REPO / ".omx/tmp/codex_runs/wc1_base_optimized_n600.done"
    receipt = {
        "schema": "ddm_wc1_full_n600_blocker.v1",
        "complete": True,
        "disposition": "QUEUED-WITH-A-FIRE-ORDER",
        "verdict_scope": "INSTANCE: current host memory-pressure state",
        "axis_label": "[M5-CPU scorer-free advisory decode]",
        "score_claim": False,
        "attempt": {
            "run_id": run_id,
            "status": (
                "PAUSED_BY_SYSTEM_MEMORY_GOVERNOR_THEN_TIMEOUT"
                if safe_status_value.get("status") == "timeout"
                else "PAUSED_BY_SYSTEM_MEMORY_GOVERNOR"
            ),
            "launch_manifest": _file_fact(launch_manifest),
            "safe_run_status_live_path": str(safe_status.resolve()),
            "safe_run_status_snapshot": _file_fact(safe_status_snapshot),
            "done_receipt": None if not done_path.is_file() else _file_fact(done_path),
            "governor_pause_event": pause_lines[0],
            "governor_latest_pause_event": pause_lines[-1],
            "latest_pressure": {
                key: latest_blackbox.get(key)
                for key in (
                    "ts_iso",
                    "available_gib",
                    "pressure",
                    "pressure_level",
                    "load1",
                )
            },
            "safety_floor": latest_blackbox.get("safety_floor"),
        },
        "materialized_decode_payloads": materialized,
        "payload_retention": (
            "no token/render/raw payload existed at the pause boundary"
            if not materialized
            else "all materialized token/render/raw payloads remain under the run root"
        ),
        "prefix_proofs": {
            "n4": _file_fact(prefix_n4.resolve()),
            "n32": _file_fact(prefix_n32.resolve()),
        },
        "unmet_gates": [
            "hv1 n600 optimized native plus parallel raw identity",
            "hv1 n600 forced-scalar twin identity",
            "real micro-edit fresh-versus-cache parity",
            "end-to-end wall-clock bar",
        ],
        "fire_order": {
            "owner": "MAIN",
            "consumer_store": str((ROOT / "receipts").resolve()),
            "fire_trigger": (
                "memory_blackbox latest row reports pressure=normal and available_gib>=64 "
                "for three consecutive polls, and the prior done receipt is terminal"
            ),
            "action": (
                "launch a new watched base optimized n600 run_id, then the scalar and "
                "micro-edit cache-parity rows serially"
            ),
        },
    }
    destination = ROOT / "receipts/FULL_N600_BLOCKED.json"
    _atomic_json(destination, receipt)
    return receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--source-generation", type=Path, default=BASE_GENERATION)
    prepare.add_argument("--destination", type=Path, required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--source-generation", type=Path, default=BASE_GENERATION)
    run.add_argument("--run-id", required=True)
    run.add_argument("--pair-count", type=int, default=FULL_PAIRS)
    run.add_argument("--native-variant", choices=("optimized", "scalar"), default="optimized")
    run.add_argument("--cache-mode", choices=("disabled", "read-write"), default="disabled")
    run.add_argument("--workers", default="auto")
    run.add_argument("--reference-raw", type=Path)
    run.add_argument("--compare-result", type=Path)
    finish = subparsers.add_parser("finalize")
    finish.add_argument("--prefix-n4", type=Path, required=True)
    finish.add_argument("--prefix-n32", type=Path, required=True)
    finish.add_argument("--base-optimized", type=Path, required=True)
    finish.add_argument("--base-scalar", type=Path, required=True)
    finish.add_argument("--micro-fresh", type=Path, required=True)
    finish.add_argument("--micro-cached", type=Path, required=True)
    watchers = subparsers.add_parser("write-watchers")
    watchers.add_argument("--run-id", required=True)
    watchers.add_argument("--launcher-dir", type=Path, required=True)
    watchers.add_argument("--done-receipt-name", required=True)
    blocker = subparsers.add_parser("write-blocker")
    blocker.add_argument("--launcher-dir", type=Path, required=True)
    blocker.add_argument("--run-id", required=True)
    blocker.add_argument("--prefix-n4", type=Path, required=True)
    blocker.add_argument("--prefix-n32", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.command == "prepare":
            result = prepare_advisory_runtime(args.source_generation, args.destination)
        elif args.command == "run":
            result = run_decode_locked(
                source_generation=args.source_generation,
                run_id=args.run_id,
                pair_count=args.pair_count,
                native_variant=args.native_variant,
                cache_mode=args.cache_mode,
                workers=args.workers,
                reference_raw=args.reference_raw,
                compare_result=args.compare_result,
            )
        elif args.command == "finalize":
            result = finalize(
                prefix_n4=args.prefix_n4,
                prefix_n32=args.prefix_n32,
                base_optimized=args.base_optimized,
                base_scalar=args.base_scalar,
                micro_fresh=args.micro_fresh,
                micro_cached=args.micro_cached,
            )
        elif args.command == "write-watchers":
            result = write_watcher_configs(
                run_id=args.run_id,
                launcher_dir=args.launcher_dir,
                done_receipt_name=args.done_receipt_name,
            )
        else:
            result = write_full_n600_blocker(
                launcher_dir=args.launcher_dir,
                run_id=args.run_id,
                prefix_n4=args.prefix_n4,
                prefix_n32=args.prefix_n32,
            )
    except WC1RunError as exc:
        print(json.dumps({"status": "BLOCKED", "reason": str(exc)}), file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
