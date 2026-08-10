#!/usr/bin/env python3
"""Run and certify the literal CX2 three-argument receiver on n600.

The decode writes into an archive-specific staging directory.  Only a raw
whose size and SHA-256 are bound to the unchanged archive, runtime, DT1 token
target, and receiver-native token checkpoint is atomically promoted into the
evaluator-facing ``submission/inflated`` directory.  A completed receipt is a
resume checkpoint; an interrupted render can reuse the receiver-native token
cache without repeating the causal ANS decode.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import shutil
import signal
import struct
import subprocess
import time
import zipfile
from pathlib import Path
from typing import Any

OUTPUT = Path("/Volumes/VertigoDataTier/pact/ddm_cx2_20260809")
DT1_RECEIPT = Path(
    "/Volumes/VertigoDataTier/pact/ddm_dt1_20260809/retained/"
    "retained_n600_result.json"
)
VIDEO_NAMES = Path(__file__).resolve().parents[1] / "upstream" / (
    "public_test_video_names.txt"
)
EXPECTED_ARCHIVE_BYTES = 186_698
EXPECTED_ARCHIVE_SHA256 = (
    "2acd09e7a585c12403936d1e8a6dc70a9b35d826fe61ead7dea49ad470c4a996"
)
EXPECTED_DT1_RECEIPT_SHA256 = (
    "5c15f38ab68df68c09a5859d17d19e4247f90e76457282edccbc8a34d060916c"
)
EXPECTED_ANS_SHA256 = (
    "a0b18dc0803ef541d3eb265bba5380f7aa067593f6af584b0891ded5bdd74488"
)
EXPECTED_RANGE_SHA256 = (
    "948379872ff81a4e5d948ec301c143be00ebd0033544c8abdfb4af0f4c4a15eb"
)
EXPECTED_DECODED_TOKEN_SHA256 = (
    "c5c7671d037b6912980c57929a5b6d789d250ee6a93e3b0a6018cf9f63e32ece"
)
EXPECTED_BASE_SHA256 = (
    "0491d5df84fc70b62b3f7ccf8894f5e1b81c616de46a052e4423fc1e18fdc7cd"
)
EXPECTED_BASE_MODELS_RAW_SHA256 = (
    "62dd72dfa0858a25ca32bdee1e536627a17883b6fc7efd7cd5b2de7b13b84517"
)
EXPECTED_CANDIDATE_MODELS_RAW_SHA256 = (
    "4bae76d7878a753ef5a675c5fddafaff6d8987c4fe1611f54ebcd6f5d0fabf21"
)
EXPECTED_RUNTIME_SHA256 = {
    "carrier_codec.py": "d2f14402374b4e622b7f981d736389fb04f0ca0165180e4c75f3a32ffe996bed",
    "hpac_integer.py": "6e6b4f4d0b293fb60cc1b751958756a4cd6c2ce7bcff68c6f03e20277856803f",
    "hpac_integer_sparse.py": (
        "2240ee32c53fe949b560d316d349e0bbdccc0ceb78787307cd4d530623d42a0c"
    ),
    "inflate.py": "105c951edccf58223818fc492e504b39872c0279e29bd4b901b60236acbf7d39",
    "inflate.sh": "bc92880ef9c038c6adfe4968a4b6206b8e565501e839634e1d6762a704421915",
    "integer_model_io.py": (
        "6f91c91ed4785d203aa3570af362fbe9c6a64bb2249599f8554adb31174b80a5"
    ),
    "receiver.py": "6239649cc81e9c5a86273502be0beff19805720854b980f167bb71a0a80c3a42",
    "runtime-dependencies.json": (
        "55397f16f270472e0f0bde1e69d2c5c5a2e015f2cc051a31e19ce2dbfc8cfe07"
    ),
}
EXPECTED_VIDEO_NAMES_SHA256 = (
    "7ff99d08c8351dd8167ec09213b758da5bbb705dedabe361ba881217374029a8"
)
EXPECTED_RAW_BYTES = 1_200 * 874 * 1_164 * 3
EXPECTED_FRAMES = 600
EXPECTED_TOKENS = 117_964_800


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


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def file_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def runtime_records(submission: Path) -> dict[str, dict[str, Any]]:
    manifest = json.loads((submission / "submission_manifest.json").read_text())
    if (
        manifest.get("schema") != "ddm_cx2_submission.v1"
        or manifest.get("complete") is not True
        or manifest.get("evaluator_archive", {}).get("bytes")
        != EXPECTED_ARCHIVE_BYTES
        or manifest.get("evaluator_archive", {}).get("sha256")
        != EXPECTED_ARCHIVE_SHA256
        or set(manifest.get("runtime_files", {})) != set(EXPECTED_RUNTIME_SHA256)
    ):
        raise RuntimeError("submission manifest denominator or archive binding differs")
    records: dict[str, dict[str, Any]] = {}
    for name, declared in manifest["runtime_files"].items():
        path = submission / name
        actual = file_record(path)
        if (
            declared["sha256"] != EXPECTED_RUNTIME_SHA256[name]
            or actual["bytes"] != declared["bytes"]
            or actual["sha256"] != declared["sha256"]
        ):
            raise RuntimeError(f"submission runtime differs from manifest: {name}")
        records[name] = actual
    dependency_manifest = json.loads(
        (submission / "runtime-dependencies.json").read_text()
    )
    expected_copied = {
        name: digest
        for name, digest in EXPECTED_RUNTIME_SHA256.items()
        if name != "runtime-dependencies.json"
    }
    if dependency_manifest.get("source", {}).get("copied_files") != expected_copied:
        raise RuntimeError("runtime dependency manifest shipping hashes differ")
    return records


def token_tail(member: bytes) -> bytes:
    if len(member) < 4:
        raise RuntimeError("archive member is truncated before model word")
    model_word = struct.unpack_from("<I", member)[0]
    model_bytes = model_word & ((1 << 29) - 1)
    token_start = 4 + model_bytes
    if token_start >= len(member):
        raise RuntimeError("archive member is truncated before token payload")
    return member[token_start:]


def read_stored_member(path: Path) -> bytes:
    with zipfile.ZipFile(path) as archive:
        entries = archive.infolist()
        if len(entries) != 1 or entries[0].filename != "p":
            raise RuntimeError("archive.zip must contain exactly one member named p")
        info = entries[0]
        if (
            info.is_dir()
            or info.compress_type != zipfile.ZIP_STORED
            or info.file_size != info.compress_size
            or info.flag_bits & 0x1
        ):
            raise RuntimeError("archive.zip member p is not an unencrypted stored file")
        payload = archive.read(info)
        if archive.testzip() is not None:
            raise RuntimeError("archive.zip failed CRC validation")
        return payload


def validate_token_checkpoint(
    record: dict[str, Any], member: bytes, tokens: bytes
) -> None:
    if (
        record.get("schema") != "ddm_cx2_token_checkpoint.v1"
        or record.get("complete") is not True
        or record.get("frames") != EXPECTED_FRAMES
        or record.get("tokens") != EXPECTED_TOKENS
        or record.get("token_codec") != "ans"
        or record.get("finish_token_decode_returned") is not True
        or record.get("ans_final_state_empty") is not True
        or record.get("decoded_token_sha256")
        != EXPECTED_DECODED_TOKEN_SHA256
        or record.get("archive_member_sha256") != sha256_bytes(member)
        or record.get("models_raw_sha256")
        != EXPECTED_CANDIDATE_MODELS_RAW_SHA256
        or record.get("token_payload_sha256") != sha256_bytes(tokens)
    ):
        raise RuntimeError("literal receiver token checkpoint differs from CX2 pins")


def validate_completed(
    result: dict[str, Any],
    submission: Path,
    runtime: dict[str, Any],
    member: bytes,
    tokens: bytes,
) -> None:
    if result.get("schema") != "ddm_cx2_literal_receiver.v1":
        raise RuntimeError("literal receiver receipt schema changed")
    if result.get("complete") is not True:
        raise RuntimeError("literal receiver receipt is not complete")
    archive = submission / "archive.zip"
    raw = submission / "inflated" / "0.raw"
    if result.get("archive", {}).get("sha256") != sha256_file(archive):
        raise RuntimeError("completed decode receipt names a different archive")
    if result.get("raw", {}).get("sha256") != sha256_file(raw):
        raise RuntimeError("completed decode receipt raw changed")
    if result.get("raw", {}).get("bytes") != EXPECTED_RAW_BYTES:
        raise RuntimeError("completed decode receipt raw geometry changed")
    if result.get("runtime_files") != runtime:
        raise RuntimeError("completed decode receipt runtime changed")
    if result.get("dt1_receipt", {}).get("sha256") != EXPECTED_DT1_RECEIPT_SHA256:
        raise RuntimeError("completed decode receipt DT1 custody changed")
    validate_token_checkpoint(result.get("token_checkpoint", {}), member, tokens)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--submission", type=Path, default=OUTPUT / "submission")
    parser.add_argument("--resume-from", type=Path, required=True)
    parser.add_argument("--dt1-receipt", type=Path, default=DT1_RECEIPT)
    parser.add_argument("--video-names-file", type=Path, default=VIDEO_NAMES)
    parser.add_argument(
        "--python",
        type=Path,
        default=Path(
            "/Volumes/VertigoDataTier/pact/ddm_pq1_runtime_20260809/"
            "venv/bin/python"
        ),
    )
    parser.add_argument(
        "--brotli-cli", type=Path, default=Path("/opt/homebrew/bin/brotli")
    )
    parser.add_argument("--timeout-seconds", type=int, default=1_800)
    parser.add_argument("--minimum-free-bytes", type=int, default=8 << 30)
    args = parser.parse_args()
    if args.timeout_seconds <= 0 or args.minimum_free_bytes <= 0:
        parser.error("timeout and minimum free bytes must be positive")
    return args


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    args.resume_from.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(args.output)
    if usage.free < args.minimum_free_bytes:
        raise RuntimeError(
            f"storage preflight refused: free={usage.free}, "
            f"required={args.minimum_free_bytes}"
        )

    submission = args.submission.resolve()
    archive = submission / "archive.zip"
    member_path = submission / "archive" / "p"
    if archive.stat().st_size != EXPECTED_ARCHIVE_BYTES:
        raise RuntimeError("evaluator-facing archive byte count differs")
    if sha256_file(archive) != EXPECTED_ARCHIVE_SHA256:
        raise RuntimeError("evaluator-facing archive SHA-256 differs")
    if sha256_file(args.video_names_file) != EXPECTED_VIDEO_NAMES_SHA256:
        raise RuntimeError("public video-names file differs")
    member = member_path.read_bytes()
    zipped_member = read_stored_member(archive)
    if member != zipped_member:
        raise RuntimeError("expanded archive/p differs from scored archive.zip member p")
    tokens = token_tail(member)
    runtime_pre = runtime_records(submission)
    if sha256_file(args.dt1_receipt) != EXPECTED_DT1_RECEIPT_SHA256:
        raise RuntimeError("DT1 terminal receipt SHA-256 differs from the CX2 pin")
    dt1 = json.loads(args.dt1_receipt.read_text())
    provenance = dt1.get("provenance", {})
    range_stream = dt1.get("streams", {}).get("range", {})
    ans_stream = dt1.get("streams", {}).get("ans", {})
    range_decode = dt1.get("range_decode", {})
    ans_decode = dt1.get("ans_decode", {})
    if (
        dt1.get("schema") != "ddm_dt1_retained_n600.v1"
        or dt1.get("complete") is not True
        or provenance.get("archive_bytes") != 191_052
        or provenance.get("archive_sha256") != EXPECTED_BASE_SHA256
        or provenance.get("models_raw_sha256") != EXPECTED_BASE_MODELS_RAW_SHA256
        or provenance.get("recorded_range_bytes") != 116_980
        or provenance.get("recorded_range_sha256") != EXPECTED_RANGE_SHA256
        or provenance.get("recorded_range_equals_archive") is not True
        or provenance.get("host", {}).get("constriction") != "0.5.0"
        or range_stream.get("bytes") != 116_980
        or range_stream.get("sha256") != EXPECTED_RANGE_SHA256
        or range_stream.get("byte_identical_to_shipped") is not True
        or ans_stream.get("bytes") != 114_860
        or ans_stream.get("sha256") != EXPECTED_ANS_SHA256
        or sha256_bytes(tokens) != EXPECTED_ANS_SHA256
    ):
        raise RuntimeError("DT1 custody does not bind this archive's n600 ANS tail")
    for label, decoded in (("Range", range_decode), ("ANS", ans_decode)):
        if (
            decoded.get("frames") != EXPECTED_FRAMES
            or decoded.get("tokens") != EXPECTED_TOKENS
            or decoded.get("exact_target_equality") is not True
            or decoded.get("all_tokens_reconstructed") is not True
            or decoded.get("decoded_sha256") != EXPECTED_DECODED_TOKEN_SHA256
        ):
            raise RuntimeError(f"DT1 {label} n600 decode proof differs from the pin")

    final_receipt = args.output / "decode" / "literal_receiver_receipt.json"
    if final_receipt.is_file():
        result = json.loads(final_receipt.read_text())
        validate_completed(result, submission, runtime_pre, member, tokens)
        print(json.dumps(result, indent=2, sort_keys=True), flush=True)
        return

    inflated = submission / "inflated"
    inflated.mkdir(parents=True, exist_ok=True)
    final_raw = inflated / "0.raw"
    promotion_intent = args.resume_from / "stages" / "07_promotion_intent.json"
    if promotion_intent.is_file():
        intent = json.loads(promotion_intent.read_text())
        result = intent.get("result", {})
        if intent.get("schema") != "ddm_cx2_promotion_intent.v1":
            raise RuntimeError("raw-promotion intent schema changed")
        if not final_raw.is_file():
            staging_record = result.get("staging_raw_before_promotion", {})
            staging_path = Path(staging_record.get("path", ""))
            if not staging_path.is_file():
                intent["recovery"] = (
                    "no final or staging raw existed; preserved as an abandoned "
                    "pre-promotion intent and relaunched from the token checkpoint"
                )
                abandoned = promotion_intent.with_name(
                    f"07_promotion_intent_abandoned_{int(time.time())}.json"
                )
                atomic_json(promotion_intent, intent)
                os.replace(promotion_intent, abandoned)
            elif (
                staging_record.get("bytes") != staging_path.stat().st_size
                or staging_record.get("sha256") != sha256_file(staging_path)
            ):
                raise RuntimeError("pre-promotion staging raw changed")
            else:
                os.replace(staging_path, final_raw)
        if final_raw.is_file():
            result["complete"] = True
            result["recovered_after_atomic_promotion"] = True
            result["written_at_utc"] = utc_now()
            validate_completed(result, submission, runtime_pre, member, tokens)
            atomic_json(final_receipt, result)
            atomic_json(args.resume_from / "stages" / "07_literal_receiver.json", result)
            print(json.dumps(result, indent=2, sort_keys=True), flush=True)
            return
    if any(inflated.iterdir()):
        raise RuntimeError(
            "refusing literal decode over unbound evaluator-facing inflated output"
        )

    attempt_id = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    attempt_id += f"-pid{os.getpid()}"
    attempt = args.output / "decode" / "attempts" / attempt_id
    staging = attempt / "inflated"
    staging.mkdir(parents=True)
    stdout_path = attempt / "inflate_stdout.log"
    stderr_path = attempt / "inflate_stderr.log"
    token_cache = args.resume_from / "stages" / "06_tokens.npz"
    token_receipt = args.resume_from / "stages" / "06_tokens.json"
    command = [
        str(submission / "inflate.sh"),
        str(submission / "archive"),
        str(staging),
        str(args.video_names_file),
    ]
    environment = dict(os.environ)
    environment.update(
        {
            "PYTHON": str(args.python),
            "PR130_BROTLI_CLI": str(args.brotli_cli),
            "PR130_INFLATE_DEVICE": "cpu",
            "PR130_RUNTIME_DEPS_DIR": str(args.output / "runtime-deps"),
            "PR130_TOKEN_CACHE": str(token_cache),
            "PR130_TOKEN_RECEIPT": str(token_receipt),
            "PYTHONHASHSEED": "0",
            "OMP_NUM_THREADS": "2",
            "MKL_NUM_THREADS": "2",
        }
    )
    launch = {
        "schema": "ddm_cx2_literal_receiver_launch.v1",
        "complete": False,
        "written_at_utc": utc_now(),
        "axis": "[macOS-CPU literal receiver; scorer-free]",
        "score_claim": False,
        "command": command,
        "environment": {
            key: environment[key]
            for key in (
                "PYTHON",
                "PR130_BROTLI_CLI",
                "PR130_INFLATE_DEVICE",
                "PR130_RUNTIME_DEPS_DIR",
                "PR130_TOKEN_CACHE",
                "PR130_TOKEN_RECEIPT",
                "PYTHONHASHSEED",
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
            )
        },
        "storage_preflight": {
            "free_bytes": usage.free,
            "required_free_bytes": args.minimum_free_bytes,
        },
        "archive": file_record(archive),
        "archive_member": file_record(member_path),
        "archive_member_equals_scored_zip": True,
        "runtime_files": runtime_pre,
        "dt1_receipt": file_record(args.dt1_receipt),
        "expected_decoded_token_sha256": EXPECTED_DECODED_TOKEN_SHA256,
        "attempt_dir": str(attempt),
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
        "timeout_seconds": args.timeout_seconds,
    }
    atomic_json(attempt / "launch.json", launch)
    atomic_json(args.resume_from / "stages" / "06_literal_receiver_launch.json", launch)

    started = time.perf_counter()
    returncode: int | None = None
    failure: str | None = None
    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        process = subprocess.Popen(
            command,
            env=environment,
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
        )
        try:
            returncode = process.wait(timeout=args.timeout_seconds)
            if returncode != 0:
                failure = f"literal inflate.sh exited {returncode}"
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            failure = f"literal inflate.sh exceeded {args.timeout_seconds} seconds"
    wall_s = time.perf_counter() - started
    staging_raw = staging / "0.raw"
    if failure is not None:
        blocked = {
            **launch,
            "schema": "ddm_cx2_literal_receiver_failure.v1",
            "complete": False,
            "finished_at_utc": utc_now(),
            "wall_s": wall_s,
            "returncode": returncode,
            "failure": failure,
            "stdout_record": file_record(stdout_path),
            "stderr_record": file_record(stderr_path),
            "staging_raw": file_record(staging_raw) if staging_raw.is_file() else None,
            "preservation": "failed staging bytes retained; no evaluator raw promoted",
        }
        atomic_json(attempt / "failure.json", blocked)
        raise RuntimeError(failure)

    def refuse_post_decode(reason: str) -> None:
        blocked = {
            **launch,
            "schema": "ddm_cx2_literal_receiver_failure.v1",
            "complete": False,
            "finished_at_utc": utc_now(),
            "wall_s": wall_s,
            "returncode": returncode,
            "failure": reason,
            "stdout_record": file_record(stdout_path),
            "stderr_record": file_record(stderr_path),
            "staging_raw": file_record(staging_raw) if staging_raw.is_file() else None,
            "preservation": "failed staging bytes retained; no evaluator raw promoted",
        }
        atomic_json(attempt / "failure.json", blocked)
        raise RuntimeError(reason)

    if not token_receipt.is_file():
        refuse_post_decode("literal receiver did not emit its token checkpoint")
    token_result = json.loads(token_receipt.read_text())
    try:
        validate_token_checkpoint(token_result, member, tokens)
    except RuntimeError:
        refuse_post_decode("literal receiver token checkpoint differs from DT1 target")
    if not staging_raw.is_file() or staging_raw.stat().st_size != EXPECTED_RAW_BYTES:
        refuse_post_decode("literal receiver raw has the wrong byte count")
    staging_raw_record = file_record(staging_raw)
    runtime_post = runtime_records(submission)
    if runtime_post != runtime_pre:
        refuse_post_decode("submission runtime changed during literal decode")
    if sha256_file(archive) != EXPECTED_ARCHIVE_SHA256:
        refuse_post_decode("submission archive changed during literal decode")

    if final_raw.exists():
        refuse_post_decode("refusing to overwrite an existing evaluator-facing raw")
    final_raw_record = dict(staging_raw_record)
    final_raw_record["path"] = str(final_raw)
    result = {
        "schema": "ddm_cx2_literal_receiver.v1",
        "complete": True,
        "written_at_utc": utc_now(),
        "axis": "[macOS-CPU literal receiver; scorer-free]",
        "score_claim": False,
        "command": command,
        "environment": launch["environment"],
        "wall_s": wall_s,
        "returncode": returncode,
        "archive": file_record(archive),
        "archive_member": file_record(member_path),
        "runtime_files": runtime_post,
        "token_checkpoint": token_result,
        "dt1_receipt": file_record(args.dt1_receipt),
        "raw": final_raw_record,
        "staging_raw_before_promotion": staging_raw_record,
        "atomic_promotion": True,
        "stdout_record": file_record(stdout_path),
        "stderr_record": file_record(stderr_path),
        "resume": {
            "resume_from": str(args.resume_from),
            "token_cache": file_record(token_cache),
            "stage_checkpoint": str(token_receipt),
        },
        "budget": {
            "limit_s": args.timeout_seconds,
            "headroom_s": args.timeout_seconds - wall_s,
            "within_budget": wall_s <= args.timeout_seconds,
        },
    }
    intent = {
        "schema": "ddm_cx2_promotion_intent.v1",
        "complete": False,
        "written_at_utc": utc_now(),
        "result": result,
    }
    atomic_json(promotion_intent, intent)
    os.replace(staging_raw, final_raw)
    if file_record(final_raw) != final_raw_record:
        raise RuntimeError("atomically promoted raw differs from validated staging bytes")
    atomic_json(attempt / "success.json", result)
    atomic_json(final_receipt, result)
    atomic_json(args.resume_from / "stages" / "07_literal_receiver.json", result)
    intent["complete"] = True
    intent["promoted_at_utc"] = utc_now()
    atomic_json(promotion_intent, intent)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
