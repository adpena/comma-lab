#!/usr/bin/env python3
"""Build and certify the evaluator-runnable PR130 ANS control.

The retained TM1 ANS control contains real ANS words but leaves the PR130
outer token selector clear.  This tool changes only that zero-byte selector,
copies the already-landed resumable receiver runtime, and runs two complete
``inflate.sh`` decodes.  Every materialized payload is retained under the
selected SSD output root and every long decode has a receiver-native token
checkpoint that can be reused after interruption.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
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

REPO = Path(__file__).resolve().parents[1]
RUNTIME = REPO / "src" / "tac" / "pr130_runtime" / "dv1_cpu_runtime"
RECEIVER_SPEC = importlib.util.spec_from_file_location("ddm_ai1_receiver", RUNTIME / "receiver.py")
if RECEIVER_SPEC is None or RECEIVER_SPEC.loader is None:
    raise RuntimeError("could not load the landed PR130 receiver")
receiver = importlib.util.module_from_spec(RECEIVER_SPEC)
sys.modules[RECEIVER_SPEC.name] = receiver
RECEIVER_SPEC.loader.exec_module(receiver)

OUTPUT = Path("/Volumes/VertigoDataTier/pact/ddm_ai1_20260809")
BASE_ARCHIVE = Path("/Volumes/VertigoDataTier/pact/ddm_pr130_reproduce_20260809/reproduction/archive.zip")
TM1_ANS_ARCHIVE = Path(
    "/Volumes/VertigoDataTier/pact/ddm_tm1_20260809/measurement_v2/baselines/ans_control/archive.zip"
)
ANS_PAYLOAD = Path("/Volumes/VertigoDataTier/pact/ddm_dt1_20260809/retained/ans_n600.bin")
DT1_RECEIPT = Path("/Volumes/VertigoDataTier/pact/ddm_dt1_20260809/retained/retained_n600_result.json")
VIDEO_NAMES = REPO / "upstream" / "public_test_video_names.txt"
PINNED_PYTHON = Path("/Volumes/VertigoDataTier/pact/ddm_pq1_runtime_20260809/venv/bin/python")

EXPECTED_BASE_BYTES = 191_052
EXPECTED_BASE_SHA256 = "0491d5df84fc70b62b3f7ccf8894f5e1b81c616de46a052e4423fc1e18fdc7cd"
EXPECTED_TM1_ARCHIVE_BYTES = 188_932
EXPECTED_TM1_ARCHIVE_SHA256 = "447d7697f60b86e2d5e26a70f48f497dd852ce19ef2fd78f2901d952a8535b42"
EXPECTED_ANS_BYTES = 114_860
EXPECTED_ANS_SHA256 = "a0b18dc0803ef541d3eb265bba5380f7aa067593f6af584b0891ded5bdd74488"
EXPECTED_RANGE_SHA256 = "948379872ff81a4e5d948ec301c143be00ebd0033544c8abdfb4af0f4c4a15eb"
EXPECTED_MODELS_RAW_SHA256 = "62dd72dfa0858a25ca32bdee1e536627a17883b6fc7efd7cd5b2de7b13b84517"
EXPECTED_DECODED_TOKEN_SHA256 = "c5c7671d037b6912980c57929a5b6d789d250ee6a93e3b0a6018cf9f63e32ece"
EXPECTED_DT1_RECEIPT_SHA256 = "5c15f38ab68df68c09a5859d17d19e4247f90e76457282edccbc8a34d060916c"
EXPECTED_VIDEO_NAMES_SHA256 = "7ff99d08c8351dd8167ec09213b758da5bbb705dedabe361ba881217374029a8"
EXPECTED_RAW_BYTES = 1_200 * 874 * 1_164 * 3
EXPECTED_FRAMES = 600
EXPECTED_TOKENS = 117_964_800
RUNTIME_FILES = (
    "carrier_codec.py",
    "hpac_integer.py",
    "hpac_integer_sparse.py",
    "inflate.py",
    "inflate.sh",
    "integer_model_io.py",
    "receiver.py",
    "runtime-dependencies.json",
)


def utc_now() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def atomic_bytes(path: Path, payload: bytes, *, executable: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    if executable:
        temporary.chmod(0o755)
    os.replace(temporary, path)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    atomic_bytes(
        path,
        (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode(),
    )


def read_stored_member(path: Path) -> bytes:
    with zipfile.ZipFile(path) as archive:
        entries = archive.infolist()
        if len(entries) != 1 or entries[0].filename != "p":
            raise RuntimeError(f"{path} must contain exactly one member named p")
        info = entries[0]
        if (
            info.is_dir()
            or info.compress_type != zipfile.ZIP_STORED
            or info.file_size != info.compress_size
            or info.flag_bits & 0x1
        ):
            raise RuntimeError(f"{path} member p is not an unencrypted stored file")
        payload = archive.read(info)
        if archive.testzip() is not None:
            raise RuntimeError(f"{path} failed ZIP CRC validation")
        return payload


def deterministic_zip(payload: bytes) -> bytes:
    import io

    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", allowZip64=False) as archive:
        info = zipfile.ZipInfo("p", date_time=(1980, 1, 1, 0, 0, 0))
        info.compress_type = zipfile.ZIP_STORED
        info.create_system = 3
        info.external_attr = 0o100644 << 16
        archive.writestr(info, payload)
    return output.getvalue()


def require_pin(path: Path, *, size: int, digest: str, label: str) -> None:
    if not path.is_file():
        raise RuntimeError(f"{label} is absent: {path}")
    if path.stat().st_size != size or sha256_file(path) != digest:
        raise RuntimeError(f"{label} differs from its charter pin: {path}")


def validate_dt1() -> dict[str, Any]:
    require_pin(
        DT1_RECEIPT,
        size=DT1_RECEIPT.stat().st_size,
        digest=EXPECTED_DT1_RECEIPT_SHA256,
        label="DT1 terminal receipt",
    )
    result = json.loads(DT1_RECEIPT.read_text())
    provenance = result.get("provenance", {})
    streams = result.get("streams", {})
    for label in ("range", "ans"):
        decoded = result.get(f"{label}_decode", {})
        if (
            decoded.get("frames") != EXPECTED_FRAMES
            or decoded.get("tokens") != EXPECTED_TOKENS
            or decoded.get("exact_target_equality") is not True
            or decoded.get("all_tokens_reconstructed") is not True
            or decoded.get("decoded_sha256") != EXPECTED_DECODED_TOKEN_SHA256
        ):
            raise RuntimeError(f"DT1 {label} decode proof differs from the pin")
    if (
        result.get("schema") != "ddm_dt1_retained_n600.v1"
        or result.get("complete") is not True
        or provenance.get("archive_sha256") != EXPECTED_BASE_SHA256
        or provenance.get("models_raw_sha256") != EXPECTED_MODELS_RAW_SHA256
        or streams.get("range", {}).get("sha256") != EXPECTED_RANGE_SHA256
        or streams.get("range", {}).get("byte_identical_to_shipped") is not True
        or streams.get("ans", {}).get("bytes") != EXPECTED_ANS_BYTES
        or streams.get("ans", {}).get("sha256") != EXPECTED_ANS_SHA256
        or provenance.get("host", {}).get("constriction") != "0.5.0"
    ):
        raise RuntimeError("DT1 terminal receipt is not the pinned PR130 ANS proof")
    return result


def runtime_commit(path: Path) -> str:
    result = subprocess.run(
        ["git", "log", "-1", "--format=%H", "--", str(path.relative_to(REPO))],
        cwd=REPO,
        check=True,
        capture_output=True,
        text=True,
    )
    commit = result.stdout.strip()
    if len(commit) != 40:
        raise RuntimeError(f"could not resolve source commit for {path}")
    return commit


def build(output: Path, minimum_free_bytes: int) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    receipt_path = output / "ans_control_build_receipt.json"
    if receipt_path.is_file():
        result, _ = validate_build(output)
        return result
    if shutil.disk_usage(output).free < minimum_free_bytes:
        raise RuntimeError(
            f"storage preflight refused: free={shutil.disk_usage(output).free}, required={minimum_free_bytes}"
        )
    require_pin(
        BASE_ARCHIVE,
        size=EXPECTED_BASE_BYTES,
        digest=EXPECTED_BASE_SHA256,
        label="PR130 reproduced archive",
    )
    require_pin(
        TM1_ANS_ARCHIVE,
        size=EXPECTED_TM1_ARCHIVE_BYTES,
        digest=EXPECTED_TM1_ARCHIVE_SHA256,
        label="TM1 untagged ANS control",
    )
    require_pin(
        ANS_PAYLOAD,
        size=EXPECTED_ANS_BYTES,
        digest=EXPECTED_ANS_SHA256,
        label="retained DT1 ANS payload",
    )
    validate_dt1()
    if sha256_file(VIDEO_NAMES) != EXPECTED_VIDEO_NAMES_SHA256:
        raise RuntimeError("public video-names file differs from the n600 pin")

    base_member = read_stored_member(BASE_ARCHIVE)
    tm1_member = read_stored_member(TM1_ANS_ARCHIVE)
    base_parts = receiver.split_payload(base_member)
    tm1_parts = receiver.split_payload(tm1_member)
    ans = ANS_PAYLOAD.read_bytes()
    if (
        base_parts.token_codec != "range"
        or base_parts.model_codec != "legacy_lzma"
        or tm1_parts.token_codec != "range"
        or tm1_parts.model_codec != "legacy_lzma"
        or tm1_parts.models != base_parts.models
        or tm1_parts.tokens != ans
    ):
        raise RuntimeError("TM1 control is not the expected untagged base-model + ANS object")

    legacy_rebuild = deterministic_zip(base_member)
    if legacy_rebuild != BASE_ARCHIVE.read_bytes():
        raise RuntimeError("legacy Range archive did not rebuild byte-identically")
    tagged_member = receiver.pack_payload(
        base_parts.models,
        ans,
        token_codec="ans",
        model_codec="legacy_lzma",
    )
    tagged_parts = receiver.split_payload(tagged_member)
    if (
        len(tagged_member) != len(tm1_member)
        or tagged_member[4:] != tm1_member[4:]
        or tagged_parts.token_codec != "ans"
        or tagged_parts.model_codec != "legacy_lzma"
    ):
        raise RuntimeError("zero-byte ANS selector changed data beyond the outer word")
    archive = deterministic_zip(tagged_member)
    repeat = deterministic_zip(tagged_member)
    if archive != repeat or len(archive) != EXPECTED_TM1_ARCHIVE_BYTES:
        raise RuntimeError("tagged ANS archive is not deterministic and size-neutral")

    retained = output / "retained" / "ans_control"
    atomic_bytes(retained / "models.xz", base_parts.models)
    atomic_bytes(retained / "tokens.ans", ans)
    atomic_bytes(retained / "payload.p", tagged_member)
    atomic_bytes(retained / "archive.zip", archive)
    atomic_bytes(retained / "archive.repeat.zip", repeat)

    submission = output / "submission_ans_control"
    (submission / "archive").mkdir(parents=True, exist_ok=True)
    atomic_bytes(submission / "archive.zip", archive)
    atomic_bytes(submission / "archive" / "p", tagged_member)
    runtime_records: dict[str, Any] = {}
    for name in RUNTIME_FILES:
        source = RUNTIME / name
        destination = submission / name
        atomic_bytes(
            destination,
            source.read_bytes(),
            executable=name == "inflate.sh",
        )
        runtime_records[name] = {
            **file_record(destination),
            "source": str(source.relative_to(REPO)),
            "source_sha256": sha256_file(source),
            "source_commit": runtime_commit(source),
        }

    result = {
        "schema": "ddm_ai1_ans_control_build.v1",
        "complete": True,
        "written_at_utc": utc_now(),
        "axis": "[archive-byte exact; scorer-free]",
        "score_claim": False,
        "receiver_integration_complete": True,
        "evaluator_runnable": False,
        "evaluator_blocker": "full n600 inflate.sh has not yet executed",
        "candidate": "PR130 CPR1 base models + retained n600 ANS tokens",
        "archive": file_record(retained / "archive.zip"),
        "archive_repeat": file_record(retained / "archive.repeat.zip"),
        "repeat_build_byte_identical": True,
        "archive_delta_vs_shipped_range_bytes": len(archive) - EXPECTED_BASE_BYTES,
        "member": file_record(retained / "payload.p"),
        "models": file_record(retained / "models.xz"),
        "tokens": file_record(retained / "tokens.ans"),
        "zero_byte_selector": {
            "legacy_outer_word": int.from_bytes(base_member[:4], "little"),
            "untagged_tm1_outer_word": int.from_bytes(tm1_member[:4], "little"),
            "tagged_ans_outer_word": int.from_bytes(tagged_member[:4], "little"),
            "bytes_changed": [
                index
                for index, (before, after) in enumerate(zip(tm1_member, tagged_member, strict=True))
                if before != after
            ],
        },
        "legacy_range_control": {
            "archive_bytes": len(legacy_rebuild),
            "archive_sha256": sha256_bytes(legacy_rebuild),
            "byte_identical_to_reproduced_archive": True,
        },
        "sources": {
            "base_archive": file_record(BASE_ARCHIVE),
            "tm1_untagged_archive": file_record(TM1_ANS_ARCHIVE),
            "dt1_receipt": file_record(DT1_RECEIPT),
        },
        "submission": str(submission.resolve()),
        "runtime_files": runtime_records,
        "storage_preflight": {
            "path": str(output.resolve()),
            "minimum_free_bytes": minimum_free_bytes,
            "free_bytes_at_build": shutil.disk_usage(output).free,
        },
    }
    atomic_json(receipt_path, result)
    return result


def validate_build(output: Path) -> tuple[dict[str, Any], Path]:
    receipt_path = output / "ans_control_build_receipt.json"
    result = json.loads(receipt_path.read_text())
    if (
        result.get("schema") != "ddm_ai1_ans_control_build.v1"
        or result.get("complete") is not True
        or result.get("receiver_integration_complete") is not True
    ):
        raise RuntimeError("ANS control build receipt is incomplete")
    submission = Path(result["submission"])
    for field, path in (
        ("archive", output / "retained" / "ans_control" / "archive.zip"),
        ("archive_repeat", output / "retained" / "ans_control" / "archive.repeat.zip"),
        ("member", output / "retained" / "ans_control" / "payload.p"),
        ("models", output / "retained" / "ans_control" / "models.xz"),
        ("tokens", output / "retained" / "ans_control" / "tokens.ans"),
    ):
        if file_record(path) != result[field]:
            raise RuntimeError(f"retained build artifact changed: {field}")
    if (submission / "archive.zip").read_bytes() != (output / "retained" / "ans_control" / "archive.zip").read_bytes():
        raise RuntimeError("evaluator-facing archive differs from retained archive")
    if (submission / "archive" / "p").read_bytes() != read_stored_member(submission / "archive.zip"):
        raise RuntimeError("expanded archive member differs from archive.zip")
    for name, record in result["runtime_files"].items():
        if file_record(submission / name) != {key: record[key] for key in ("path", "bytes", "sha256")}:
            raise RuntimeError(f"shipping runtime changed after build: {name}")
    return result, submission


def validate_token_receipt(path: Path, member: bytes, tokens: bytes) -> dict[str, Any]:
    result = json.loads(path.read_text())
    if (
        result.get("schema") != "ddm_cx2_token_checkpoint.v1"
        or result.get("complete") is not True
        or result.get("frames") != EXPECTED_FRAMES
        or result.get("tokens") != EXPECTED_TOKENS
        or result.get("token_codec") != "ans"
        or result.get("finish_token_decode_returned") is not True
        or result.get("ans_final_state_empty") is not True
        or result.get("decoded_token_sha256") != EXPECTED_DECODED_TOKEN_SHA256
        or result.get("archive_member_sha256") != sha256_bytes(member)
        or result.get("models_raw_sha256") != EXPECTED_MODELS_RAW_SHA256
        or result.get("token_payload_sha256") != sha256_bytes(tokens)
    ):
        raise RuntimeError(f"token checkpoint is not the complete ANS proof: {path}")
    cache = Path(result["cache"]["path"])
    if file_record(cache) != result["cache"]:
        raise RuntimeError(f"token checkpoint bytes changed: {cache}")
    return result


def decode(
    output: Path,
    label: str,
    python: Path,
    timeout_seconds: int,
    minimum_free_bytes: int,
) -> dict[str, Any]:
    build_receipt, submission = validate_build(output)
    run_root = output / "decode" / label
    run_root.mkdir(parents=True, exist_ok=True)
    complete_path = run_root / "decode_receipt.json"
    final_raw = run_root / "0.raw"
    token_cache = run_root / "checkpoint" / "tokens.npz"
    token_progress = run_root / "checkpoint" / "tokens.progress.npz"
    token_receipt = run_root / "checkpoint" / "tokens_receipt.json"
    log_path = run_root / "inflate.log"
    if complete_path.is_file():
        result = json.loads(complete_path.read_text())
        member = (submission / "archive" / "p").read_bytes()
        parts = receiver.split_payload(member)
        validate_token_receipt(token_receipt, member, parts.tokens)
        if result.get("complete") is not True or file_record(final_raw) != result.get("raw"):
            raise RuntimeError(f"completed decode receipt changed: {label}")
        return result

    free = shutil.disk_usage(output).free
    if free < minimum_free_bytes:
        raise RuntimeError(f"storage preflight refused decode {label}: free={free}, required={minimum_free_bytes}")
    if not python.is_file():
        raise RuntimeError(f"pinned runtime Python is absent: {python}")
    staging = run_root / "staging"
    staging.mkdir(parents=True, exist_ok=True)
    staging_raw = staging / "0.raw"
    command = [
        str(submission / "inflate.sh"),
        str(submission / "archive"),
        str(staging),
        str(VIDEO_NAMES),
    ]
    state = {
        "schema": "ddm_ai1_decode_state.v1",
        "complete": False,
        "written_at_utc": utc_now(),
        "label": label,
        "command": command,
        "archive": build_receipt["archive"],
        "token_cache": str(token_cache.resolve()),
        "token_progress": str(token_progress.resolve()),
        "token_receipt": str(token_receipt.resolve()),
        "staging_raw": str(staging_raw.resolve()),
        "resumable_from_disk": True,
        "storage_preflight": {
            "path": str(output.resolve()),
            "minimum_free_bytes": minimum_free_bytes,
            "free_bytes_at_launch": free,
        },
    }
    atomic_json(run_root / "decode_state.json", state)
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHON": str(python),
            "PR130_INFLATE_DEVICE": "cpu",
            "PR130_TOKEN_CACHE": str(token_cache),
            "PR130_TOKEN_RECEIPT": str(token_receipt),
            "PR130_RUNTIME_DEPS_DIR": str(run_root / "runtime-deps"),
        }
    )
    started = time.perf_counter()
    with log_path.open("ab") as log:
        log.write(
            (
                f"\nDDM_AI1_DECODE_START label={label} utc={utc_now()} resume_cache_exists={token_cache.is_file()}\n"
            ).encode()
        )
        log.flush()
        os.fsync(log.fileno())
        completed = subprocess.run(
            command,
            cwd=submission,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
            timeout=timeout_seconds,
        )
        log.flush()
        os.fsync(log.fileno())
    wall = time.perf_counter() - started
    if completed.returncode != 0:
        raise RuntimeError(
            f"inflate.sh failed for {label} with rc={completed.returncode}; "
            f"partial payload and log retained at {run_root}"
        )
    if staging_raw.stat().st_size != EXPECTED_RAW_BYTES:
        raise RuntimeError(f"decode {label} produced the wrong raw byte count")
    member = (submission / "archive" / "p").read_bytes()
    parts = receiver.split_payload(member)
    token_result = validate_token_receipt(token_receipt, member, parts.tokens)
    os.replace(staging_raw, final_raw)
    result = {
        "schema": "ddm_ai1_literal_decode.v1",
        "complete": True,
        "written_at_utc": utc_now(),
        "axis": "[macOS-CPU receiver decode; scorer-free]",
        "score_claim": False,
        "label": label,
        "command": command,
        "wall_seconds": wall,
        "within_1800_second_inflate_limit": wall <= 1_800,
        "archive": file_record(submission / "archive.zip"),
        "archive_member": file_record(submission / "archive" / "p"),
        "raw": file_record(final_raw),
        "token_checkpoint": token_result,
        "token_progress_checkpoint": file_record(token_progress),
        "log": file_record(log_path),
        "runtime_files": {name: file_record(submission / name) for name in RUNTIME_FILES},
        "resumable_from_disk": True,
    }
    atomic_json(complete_path, result)
    state["complete"] = True
    state["completed_receipt"] = file_record(complete_path)
    atomic_json(run_root / "decode_state.json", state)
    return result


def certify(output: Path) -> dict[str, Any]:
    build_receipt, submission = validate_build(output)
    runs = {}
    for label in ("a", "b"):
        receipt = output / "decode" / label / "decode_receipt.json"
        if not receipt.is_file():
            raise RuntimeError(f"decode {label} is not complete")
        result = json.loads(receipt.read_text())
        raw = Path(result["raw"]["path"])
        if result.get("complete") is not True or file_record(raw) != result["raw"]:
            raise RuntimeError(f"decode {label} receipt or raw changed")
        token_path = output / "decode" / label / "checkpoint" / "tokens_receipt.json"
        member = (submission / "archive" / "p").read_bytes()
        tokens = receiver.split_payload(member).tokens
        validate_token_receipt(token_path, member, tokens)
        runs[label] = result
    if runs["a"]["raw"]["sha256"] != runs["b"]["raw"]["sha256"]:
        raise RuntimeError("the two complete receiver decodes are not byte-identical")
    inflated = submission / "inflated"
    inflated.mkdir(parents=True, exist_ok=True)
    evaluator_raw = inflated / "0.raw"
    source_raw = Path(runs["a"]["raw"]["path"])
    if evaluator_raw.exists():
        if file_record(evaluator_raw)["sha256"] != runs["a"]["raw"]["sha256"]:
            raise RuntimeError("existing evaluator raw differs from decode a")
    else:
        os.link(source_raw, evaluator_raw)
    result = {
        "schema": "ddm_ai1_determinism_receipt.v1",
        "complete": True,
        "written_at_utc": utc_now(),
        "axis": "[macOS-CPU receiver decode; scorer-free]",
        "score_claim": False,
        "archive": build_receipt["archive"],
        "archive_repeat": build_receipt["archive_repeat"],
        "archive_repeat_byte_identical": True,
        "runs": {label: file_record(output / "decode" / label / "decode_receipt.json") for label in runs},
        "raw_a": runs["a"]["raw"],
        "raw_b": runs["b"]["raw"],
        "frames_bit_identical": True,
        "decoded_token_sha256": EXPECTED_DECODED_TOKEN_SHA256,
        "ans_terminal_state_empty_both": True,
        "inflate_wall_seconds": {label: runs[label]["wall_seconds"] for label in runs},
        "within_1800_second_inflate_limit_both": all(runs[label]["within_1800_second_inflate_limit"] for label in runs),
        "evaluator_raw": file_record(evaluator_raw),
    }
    atomic_json(output / "determinism_receipt.json", result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("build", "decode", "certify"))
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--label", choices=("a", "b"))
    parser.add_argument("--python", type=Path, default=PINNED_PYTHON)
    parser.add_argument("--timeout-seconds", type=int, default=1_800)
    parser.add_argument("--minimum-free-bytes", type=int, default=12 << 30)
    args = parser.parse_args()
    if args.timeout_seconds <= 0 or args.minimum_free_bytes <= 0:
        parser.error("timeout and minimum-free-bytes must be positive")
    if args.command == "decode" and args.label is None:
        parser.error("decode requires --label a|b")
    if args.command != "decode" and args.label is not None:
        parser.error("--label is accepted only for decode")
    return args


def main() -> None:
    args = parse_args()
    if args.command == "build":
        result = build(args.output.resolve(), args.minimum_free_bytes)
    elif args.command == "decode":
        result = decode(
            args.output.resolve(),
            args.label,
            args.python.expanduser().absolute(),
            args.timeout_seconds,
            args.minimum_free_bytes,
        )
    else:
        result = certify(args.output.resolve())
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
