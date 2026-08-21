#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""Bounded stale-checkpoint rehearsal for the E1 TR1 runtime.

This compiles all counted state but renders at most four explicitly selected
pairs.  It never emits a score claim.  Its evaluator leg copies the locked
public entrypoint byte-for-byte and runs it against a deterministically derived
two-frame, one-sample corpus.  That proves only interface execution; the
resulting score and rate are explicitly non-comparable with an n600 row.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve()
REPO = HERE.parents[1]
for value in (str(REPO), str(REPO / "src")):
    if value not in sys.path:
        sys.path.insert(0, value)

from tac.optimization import ddm_tr1_runtime as runtime  # noqa: E402
from tac.process_group_kill import run_in_process_group  # noqa: E402

DEFAULT_CHECKPOINT = Path(
    "/Volumes/VertigoDataTier/pact/ddm_tb1_20260728/t2_n600_lotto/checkpoints/stage_seg_trunk_tau_final.npz"
)
DEFAULT_UPSTREAM = Path("/Volumes/VertigoDataTier/pact/molab_witness_machine_upstream_20260709")
DEFAULT_LOCKED_PYTHON = Path(
    "/Volumes/VertigoDataTier/pact/evidence/ddm_e4_brotli_declared_dep_20260724/env_brotli/bin/python"
)
DEFAULT_PROOF_ROOT = Path("/Volumes/VertigoDataTier/pact/ddm_eg1_tr1_rehearsal_20260728")
EXPECTED_CHECKPOINT_SHA256 = "67454bc9ac30ea6f971c29c6c6cea7b8af8d98293ed77e44be3a14b8a8667bc2"
EXPECTED_EVALUATE_PY_SHA256 = "7da71a84ce24286bc6b583470f9bbd25c998971da301320d0d4e9d6fd40baa4b"
EXPECTED_EVALUATE_SH_SHA256 = "9612284ce6e9585aefcf636f3027808a56160ffd572edffdf4b8622a65fac917"
PORTABLE_PARITY_MIN = 0.9997
MIN_PROOF_FREE_BYTES = 256 * 1024 * 1024

INFLATE_RUNNER = b"""\
from __future__ import annotations
import sys
from pathlib import Path, PurePosixPath
from ddm_tr1_runtime import parse_extracted_archive, render_pair_camera_uint8

archive_dir, output_dir, names_file = map(Path, sys.argv[1:4])
names = [row.strip() for row in names_file.read_text().splitlines() if row.strip()]
if names != ["0.mkv"]:
    raise SystemExit("bounded E1 inflate requires exactly the custodied 0.mkv")
parsed = parse_extracted_archive(archive_dir)
pair = render_pair_camera_uint8(parsed.packet, 0)
name = PurePosixPath(names[0])
if name.is_absolute() or ".." in name.parts:
    raise SystemExit("unsafe video name")
target = output_dir / name.with_suffix(".raw")
target.parent.mkdir(parents=True, exist_ok=True)
target.write_bytes(pair.tobytes())
"""

# Resolve the interpreter explicitly and fail CLOSED.  A bare ``python`` is
# not portable: on a host that ships only ``python3`` it dies with
# "python: command not found", and a backgrounding wrapper then reports the
# LAUNCHER's rc=0 instead of that 127 -- the failure and the success carry the
# same symbol.  Measured live by ddm_ob1 2026-08-03 (task #929): an inflate of
# the shipped pu2 archive reported exit 0 having produced nothing, caught only
# by reading a zero-byte log.  ``exec`` keeps the runner's own exit status as
# this script's exit status, so no rc can be swallowed here either.
INFLATE_SH = b"""\
#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON:-}"
if [ -z "${PY}" ]; then
  for candidate in python3 python; do
    if command -v "${candidate}" >/dev/null 2>&1; then
      PY="${candidate}"
      break
    fi
  done
fi
if [ -z "${PY}" ]; then
  echo "inflate.sh: FATAL: no Python interpreter (tried PYTHON env var, python3, python)" >&2
  exit 127
fi
exec "${PY}" "$HERE/inflate_runner.py" "$1" "$2" "$3"
"""


def _sha256_file(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    total = 0
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(8 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            total += len(chunk)
    return total, digest.hexdigest()


def _mlx_reference(
    checkpoint: Path,
    pair_ids: list[int],
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], str, dict[str, Any]]:
    import mlx.core as mx
    from mlx.utils import tree_unflatten

    from experiments.train_tr1_partition_renderer_mlx import (
        TR1Config,
        build_module,
    )

    with np.load(checkpoint, allow_pickle=False) as stored:
        metadata = json.loads(bytes(stored["meta::json"]).decode())
        config = TR1Config(**metadata["cfg"])
        model = build_module(config)
        ema_numpy = [
            (key[len("ema::") :], np.asarray(stored[key], dtype=np.float32))
            for key in stored.files
            if key.startswith("ema::")
        ]
    # §3.3(a) lattice-anneal engagement is a PLAIN BOOL on the module (never an mx.array =>
    # deliberately absent from the checkpoint), so a freshly-built module always starts at
    # `cfg.token_quant_anneal != "at_knee"`. The trainer re-engages it on resume
    # (train_tr1_partition_renderer_mlx.py: `if cfg.token_quant_anneal == "at_knee" and
    # stage != "seg_trunk_ce"`); this rehearsal did not, so on an `at_knee` checkpoint the
    # reference rendered UNQUANTIZED float tokens while the receiver
    # (`compile_archive_from_checkpoint` -> `_token_codes`) quantizes UNCONDITIONALLY. The gate
    # then measured token_max_abs == exactly half a lattice step and returned a FALSE
    # BLOCKED_DEPLOYMENT_BACKEND_PARITY (#828; measured 0.0666667 == 1/(levels-1) -> 0.0).
    #
    # The reference must sit in the state the ARCHIVE encodes, and the receiver quantizes for
    # every checkpoint, so engagement is unconditional here — derived from the receiver contract,
    # not from the training schedule. The trainer's own resume rule is still evaluated and
    # emitted so a pre-knee checkpoint (where the two disagree) is visible, never silent.
    stage = str(metadata.get("stage", ""))
    anneal = getattr(config, "token_quant_anneal", "off")
    trainer_resume_rule = (anneal != "at_knee") or (stage != "seg_trunk_ce")
    build_default = model._quant_engaged
    model._quant_engaged = True  # receiver quantizes unconditionally => reference must too
    engagement = {
        "basis": "receiver_contract_compile_archive_always_quantizes",
        "build_default_quant_engaged": bool(build_default),
        "checkpoint_stage": stage,
        "reference_quant_engaged": True,
        "token_quant_anneal": anneal,
        "trainer_resume_rule_would_engage": bool(trainer_resume_rule),
        "trainer_rule_agrees_with_reference": bool(trainer_resume_rule),
    }
    model.update(tree_unflatten([(key, mx.array(value)) for key, value in ema_numpy]))
    mx.eval(model.parameters())
    tokens: list[np.ndarray] = []
    source_frames: list[np.ndarray] = []
    with mx.stream(mx.cpu):
        for pair_id in pair_ids:
            token = model.quantized_tokens(pair_id)
            frame = model.render_frame(pair_id)
            mx.eval(token, frame)
            tokens.append(np.asarray(token, dtype=np.float32))
            source_frames.append(np.asarray(frame, dtype=np.float32)[0])
    deployed_ema = [
        (
            key,
            (value.astype(np.float16).astype(np.float32) if key.startswith(("g_", "b_")) else value),
        )
        for key, value in ema_numpy
    ]
    model.update(tree_unflatten([(key, mx.array(value)) for key, value in deployed_ema]))
    mx.eval(model.parameters())
    deployed_frames: list[np.ndarray] = []
    with mx.stream(mx.cpu):
        for pair_id in pair_ids:
            frame = model.render_frame(pair_id)
            mx.eval(frame)
            deployed_frames.append(np.asarray(frame, dtype=np.float32)[0])
    trainer_path = REPO / "experiments/train_tr1_partition_renderer_mlx.py"
    return (
        tokens,
        source_frames,
        deployed_frames,
        _sha256_file(trainer_path)[1],
        engagement,
    )


def _parity_receipt(
    parsed: runtime.ParsedTR1Packet,
    checkpoint: Path,
    pair_ids: list[int],
) -> dict[str, Any]:
    from experiments.train_witness_realized_through_R_mlx import (
        _torch_R_to_camera_uint8,
    )

    (
        reference_tokens,
        source_frames,
        deployed_frames,
        trainer_sha,
        engagement,
    ) = _mlx_reference(checkpoint, pair_ids)
    rows: list[dict[str, Any]] = []
    for index, pair_id in enumerate(pair_ids):
        token_numpy = runtime.decode_token_grid(parsed, pair_id)
        frame_numpy = runtime.render_frame1_float(parsed, pair_id)
        token_delta = np.abs(token_numpy - reference_tokens[index])
        frame_delta = np.abs(frame_numpy - deployed_frames[index])
        source_deploy_delta = np.abs(frame_numpy - source_frames[index])
        camera_numpy = runtime.render_frame1_camera_uint8(parsed, pair_id)
        camera_reference = _torch_R_to_camera_uint8(deployed_frames[index])
        camera_source = _torch_R_to_camera_uint8(source_frames[index])
        camera_mismatches = int(np.count_nonzero(camera_numpy != camera_reference))
        camera_values = int(camera_numpy.size)
        source_camera_mismatches = int(np.count_nonzero(camera_numpy != camera_source))
        rows.append(
            {
                "camera_byte_agreement": (1.0 - camera_mismatches / camera_values),
                "camera_byte_mismatches": camera_mismatches,
                "camera_values": camera_values,
                "float32_max_abs": float(frame_delta.max()),
                "float32_mean_abs": float(frame_delta.mean()),
                "pair_id": pair_id,
                "source_ema_fp32_camera_byte_agreement": (1.0 - source_camera_mismatches / camera_values),
                "source_ema_fp32_camera_byte_mismatches": (source_camera_mismatches),
                "source_ema_fp32_float32_max_abs": float(source_deploy_delta.max()),
                "source_ema_fp32_float32_mean_abs": float(source_deploy_delta.mean()),
                "token_max_abs": float(token_delta.max()),
                "token_mean_abs": float(token_delta.mean()),
            }
        )
    passed = all(
        row["token_max_abs"] == 0.0
        and row["float32_max_abs"] <= 5e-4
        and row["float32_mean_abs"] <= 5e-5
        and row["camera_byte_agreement"] >= PORTABLE_PARITY_MIN
        for row in rows
    )
    return {
        "backend_a": "numpy_fp32_scipy_erf_receiver",
        "backend_b": "mlx_cpu_fp32_with_deployed_fp16_values_upcast",
        "camera_reference": "torch_cpu_bicubic_align_corners_false_uint8",
        "pair_rows": rows,
        "parity_minimum": PORTABLE_PARITY_MIN,
        "passed": passed,
        "quant_engagement": engagement,
        "source_ema_fp32_deploy_gap_status": ("MEASURED_NOT_BACKEND_PARITY"),
        "source_ema_fp32_reference": "mlx_cpu_fp32_checkpoint_forward",
        "trainer_source_sha256": trainer_sha,
    }


def _write_exact(path: Path, payload: bytes, *, executable: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.is_symlink() or not path.is_file():
            raise runtime.TR1RuntimeError(f"proof member is not regular: {path}")
        if path.read_bytes() != payload:
            raise runtime.TR1RuntimeError(f"existing proof member differs; refusing overwrite: {path}")
    else:
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_bytes(payload)
        os.replace(temporary, path)
    if executable:
        path.chmod(0o755)


def _storage_preflight(proof_root: Path) -> dict[str, Any]:
    if not str(proof_root).startswith("/Volumes/VertigoDataTier/pact/"):
        raise runtime.TR1RuntimeError("bounded proof root must use VertigoDataTier")
    probe = proof_root
    while not probe.exists():
        if probe == probe.parent:
            raise runtime.TR1RuntimeError("cannot resolve proof-root storage tier")
        probe = probe.parent
    stat = os.statvfs(probe)
    free_bytes = stat.f_bavail * stat.f_frsize
    if free_bytes < MIN_PROOF_FREE_BYTES:
        raise runtime.TR1RuntimeError("bounded proof root lacks the 256 MiB storage safety floor")
    return {
        "checked_tier": "/Volumes/VertigoDataTier/pact",
        "minimum_free_bytes": MIN_PROOF_FREE_BYTES,
        "passed": True,
        "scratch_policy": "atomic temporary files removed; durable proof bytes retained",
    }


def _bounded_video(
    *,
    source: Path,
    target: Path,
    ffmpeg: Path,
) -> dict[str, Any]:
    source_bytes, source_sha = _sha256_file(source)
    ffmpeg_bytes, ffmpeg_sha = _sha256_file(ffmpeg)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name("0.bounded.tmp.mkv")
    if temporary.exists():
        temporary.unlink()
    command = [
        str(ffmpeg),
        "-v",
        "error",
        "-nostdin",
        "-i",
        str(source),
        "-map",
        "0:v:0",
        "-frames:v",
        "2",
        "-an",
        "-c:v",
        "ffv1",
        "-level",
        "3",
        "-pix_fmt",
        "yuv420p",
        "-fflags",
        "+bitexact",
        "-flags:v",
        "+bitexact",
        "-map_metadata",
        "-1",
        str(temporary),
    ]
    result = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode != 0 or not temporary.is_file():
        temporary.unlink(missing_ok=True)
        raise runtime.TR1RuntimeError(f"bounded ffmpeg failed rc={result.returncode}: {result.stderr[-1000:]}")
    generated = temporary.read_bytes()
    superseded: dict[str, Any] | None = None
    if target.exists() and target.read_bytes() != generated:
        old_bytes, old_sha = _sha256_file(target)
        cold = target.parent.parent / "superseded" / f"0.{old_sha}.mkv"
        cold.parent.mkdir(parents=True, exist_ok=True)
        if cold.exists() and cold.read_bytes() != target.read_bytes():
            temporary.unlink()
            raise runtime.TR1RuntimeError("bounded-video cold-store collision")
        if not cold.exists():
            os.replace(target, cold)
        else:
            target.unlink()
        superseded = {
            "bytes": old_bytes,
            "path": str(cold),
            "sha256": old_sha,
        }
    if target.exists():
        temporary.unlink()
    else:
        os.replace(temporary, target)
    target_bytes, target_sha = _sha256_file(target)
    return {
        "command": command,
        "ffmpeg_bytes": ffmpeg_bytes,
        "ffmpeg_path": str(ffmpeg),
        "ffmpeg_sha256": ffmpeg_sha,
        "source_bytes": source_bytes,
        "source_path": str(source),
        "source_sha256": source_sha,
        "target_bytes": target_bytes,
        "target_path": str(target),
        "target_sha256": target_sha,
        "temporary_removed": not temporary.exists(),
        "superseded_previous": superseded,
    }


def _run_bounded_upstream_smoke(
    upstream_root: Path,
    locked_python: Path,
    proof_root: Path,
    compiled: runtime.CompiledTR1Archive,
) -> dict[str, Any]:
    evaluate_py = upstream_root / "evaluate.py"
    evaluate_sh = upstream_root / "evaluate.sh"
    py_bytes, py_sha = _sha256_file(evaluate_py)
    sh_bytes, sh_sha = _sha256_file(evaluate_sh)
    locked = py_sha == EXPECTED_EVALUATE_PY_SHA256 and sh_sha == EXPECTED_EVALUATE_SH_SHA256
    if not locked:
        return {
            "attempted": False,
            "blocker": "evaluate.py/evaluate.sh bytes differ from locked custody",
            "evaluate_py_sha256": py_sha,
            "evaluate_sh_sha256": sh_sha,
            "status": "BLOCKED_LOCKED_UPSTREAM_HASH_MISMATCH",
        }
    storage_preflight = _storage_preflight(proof_root)

    proof_root.mkdir(parents=True, exist_ok=True)
    copied: list[dict[str, Any]] = []
    for relative in (
        "evaluate.py",
        "evaluate.sh",
        "frame_utils.py",
        "modules.py",
        "models/segnet.safetensors",
        "models/posenet.safetensors",
    ):
        source_path = upstream_root / relative
        payload = source_path.read_bytes()
        target_path = proof_root / relative
        _write_exact(
            target_path,
            payload,
            executable=relative.endswith((".py", ".sh")),
        )
        copied.append(
            {
                "bytes": len(payload),
                "path": relative,
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    _write_exact(proof_root / "public_test_video_names.txt", b"0.mkv\n")
    video = _bounded_video(
        source=upstream_root / "videos/0.mkv",
        target=proof_root / "videos/0.mkv",
        ffmpeg=Path(os.environ.get("DDM_EG1_FFMPEG", "/opt/homebrew/bin/ffmpeg")),
    )

    submission = proof_root / "submission"
    _write_exact(submission / "archive.zip", compiled.archive_bytes)
    _write_exact(
        submission / "ddm_tr1_runtime.py",
        Path(runtime.__file__).resolve().read_bytes(),
    )
    _write_exact(submission / "inflate_runner.py", INFLATE_RUNNER)
    _write_exact(submission / "inflate.sh", INFLATE_SH, executable=True)

    command = [
        "bash",
        str(proof_root / "evaluate.sh"),
        "--submission-dir",
        str(submission),
        "--video-names-file",
        str(proof_root / "public_test_video_names.txt"),
        "--device",
        "cpu",
    ]
    environment = dict(os.environ)
    environment["PATH"] = str(locked_python.parent) + os.pathsep + environment.get("PATH", "")
    # The locked evaluator's tqdm timing text is nondeterministic even when
    # every scientific result is byte-identical. Suppress progress rendering
    # so preserved process custody can reproduce exactly.
    environment["TQDM_DISABLE"] = "1"
    result = run_in_process_group(
        command,
        cwd=proof_root,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=600,
    )
    stdout_payload = result.stdout.encode()
    stderr_payload = result.stderr.encode()
    stdout_path = proof_root / (f"bounded_evaluate.stdout.{hashlib.sha256(stdout_payload).hexdigest()}.txt")
    stderr_path = proof_root / (f"bounded_evaluate.stderr.{hashlib.sha256(stderr_payload).hexdigest()}.txt")
    _write_exact(stdout_path, stdout_payload)
    _write_exact(stderr_path, stderr_payload)
    report = submission / "report.txt"
    report_text = report.read_text() if report.is_file() else ""
    sample_match = re.search(
        r"Evaluation results over ([0-9]+) samples",
        report_text,
    )
    report_bytes, report_sha = _sha256_file(report) if report.is_file() else (0, None)
    inflated = submission / "inflated/0.raw"
    inflated_bytes, inflated_sha = _sha256_file(inflated) if inflated.is_file() else (0, None)
    status = (
        "PASS_INTERFACE_ONLY_NONCOMPARABLE"
        if result.returncode == 0 and sample_match and sample_match.group(1) == "1"
        else "BLOCKED_BOUNDED_EVALUATOR_RUNTIME_ERROR"
    )
    blocker = (
        None
        if status.startswith("PASS")
        else (f"evaluate.sh rc={result.returncode}; stderr_tail={result.stderr[-1000:]!r}")
    )
    return {
        "attempted": True,
        "blocker": blocker,
        "command": command,
        "copied_locked_assets": copied,
        "evaluate_py_bytes": py_bytes,
        "evaluate_py_sha256": py_sha,
        "evaluate_sh_bytes": sh_bytes,
        "evaluate_sh_sha256": sh_sha,
        "exit_code": result.returncode,
        "inflated_bytes": inflated_bytes,
        "inflated_path": str(inflated),
        "inflated_sha256": inflated_sha,
        "interface_only": True,
        "locked_hashes_match": locked,
        "locked_python_exists": locked_python.exists(),
        "locked_python_path": str(locked_python),
        "locked_python_resolved": str(locked_python.resolve()),
        "proof_root": str(proof_root),
        "rate_and_score_comparable": False,
        "report_bytes": report_bytes,
        "report_path": str(report),
        "report_sha256": report_sha,
        "sample_count": int(sample_match.group(1)) if sample_match else None,
        "storage_preflight": storage_preflight,
        "stderr_path": str(stderr_path),
        "status": status,
        "stdout_path": str(stdout_path),
        "video_derivation": video,
    }


def rehearse(
    *,
    checkpoint: Path,
    max_pairs: int,
    upstream_root: Path,
    locked_python: Path,
    proof_root: Path = DEFAULT_PROOF_ROOT,
) -> dict[str, Any]:
    if max_pairs < 1 or max_pairs > 4:
        raise runtime.TR1RuntimeError("--max-pairs must be in [1,4]")
    first = runtime.compile_archive_from_checkpoint(checkpoint)
    second = runtime.compile_archive_from_checkpoint(checkpoint)
    if checkpoint == DEFAULT_CHECKPOINT and (first.checkpoint.sha256 != EXPECTED_CHECKPOINT_SHA256):
        raise runtime.TR1RuntimeError("stale T2 lotto checkpoint SHA-256 differs")
    if first.archive_bytes != second.archive_bytes:
        raise runtime.TR1RuntimeError("archive compilation is nondeterministic")
    parsed_archive = runtime.parse_archive(first.archive_bytes)
    if runtime.reemit_archive(parsed_archive) != first.archive_bytes:
        raise runtime.TR1RuntimeError("archive parse/re-emit changed bytes")
    pair_ids = list(range(max_pairs))
    parity = _parity_receipt(
        parsed_archive.packet,
        checkpoint,
        pair_ids,
    )
    upstream = _run_bounded_upstream_smoke(
        upstream_root,
        locked_python,
        proof_root,
        first,
    )
    runtime_bytes, runtime_sha = _sha256_file(Path(runtime.__file__).resolve())
    tool_bytes, tool_sha = _sha256_file(HERE)
    return {
        "archive": {
            "bytes": len(first.archive_bytes),
            "deterministic_compile_twice": True,
            "exact_parse_reemit": True,
            "sha256": first.archive_sha256,
            "zip_compression": "stored",
        },
        "catalog_417_receiver_consumption": {
            "catalog_402_correction": ("#402 is telemetry liveness; exact counted consumption is #417"),
            "final_cursor_equal": True,
            "pose_stub_consumed": parsed_archive.packet.pose_stub_consumed,
            "section_ledger": runtime.section_ledger(parsed_archive.packet),
            "strict_order": list(runtime.SECTION_NAMES),
            "trailing_bytes_refused": True,
        },
        "checkpoint": {
            "bytes": first.checkpoint.bytes,
            "config_hash": first.checkpoint.config_hash,
            "parameter_set": "ema",
            "path": str(first.checkpoint.path),
            "sha256": first.checkpoint.sha256,
        },
        "implementation": {
            "runtime_bytes": runtime_bytes,
            "runtime_path": str(Path(runtime.__file__).resolve()),
            "runtime_sha256": runtime_sha,
            "tool_bytes": tool_bytes,
            "tool_path": str(HERE),
            "tool_sha256": tool_sha,
        },
        "packet": {
            "bytes": len(first.packet_bytes),
            "combined_ema_base_plus_delta": True,
            "sha256": first.packet_sha256,
        },
        "parity": parity,
        "pointer_moved": False,
        "pose_claim": False,
        "pose_stub": {
            "frame0_behavior": "generic_zero_frame",
            "inert": True,
            "pose_claim": False,
            "values_count": 0,
        },
        "promotion_eligible": False,
        "research_only": True,
        "run_id": "ddm_eg1_tr1_rehearsal_20260728",
        "schema": "ddm_eg1_tr1_rehearsal.v1",
        "score_claim": False,
        "stale_pairs_rendered": pair_ids,
        "upstream_bounded_smoke": upstream,
        "verdict": (
            "PASS_BOUNDED_RUNTIME_AND_INTERFACE_ONLY_EVALUATOR"
            if parity["passed"] and upstream["status"] == "PASS_INTERFACE_ONLY_NONCOMPARABLE"
            else ("BLOCKED_DEPLOYMENT_BACKEND_PARITY" if not parity["passed"] else "BLOCKED_BOUNDED_EVALUATOR_RUNTIME")
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--max-pairs", type=int, default=1)
    parser.add_argument("--upstream-root", type=Path, default=DEFAULT_UPSTREAM)
    parser.add_argument(
        "--locked-python",
        type=Path,
        default=DEFAULT_LOCKED_PYTHON,
    )
    parser.add_argument("--proof-root", type=Path, default=DEFAULT_PROOF_ROOT)
    args = parser.parse_args()
    receipt = rehearse(
        checkpoint=args.checkpoint,
        max_pairs=args.max_pairs,
        upstream_root=args.upstream_root,
        locked_python=args.locked_python,
        proof_root=args.proof_root,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0 if receipt["parity"]["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
