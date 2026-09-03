#!/usr/bin/env python3
"""Run and publish a fresh Schur carrier solve for one fcd1 candidate body.

``instrument`` repoints the proven jg5 Gauss-Newton/byte-close machinery at the
candidate's own archive and raw decode, including the module globals that jg5
serializes into its receipts.  ``publish`` is the compile-time gate: it accepts
only two deterministic byte-close builds from those fresh rows, asserts that the
resolved pose is no worse than the same-instrument jt21 base within the declared
band, and only then stages a pin-consistent runtime.

An archive emitted by jg5 ``close`` before this gate remains a retained probe.  It
is not a compiled candidate and cannot enter the fire order.  This distinction is
the structural cure for qs4's stale/carried-compensation failure.
"""

from __future__ import annotations

import argparse
import hashlib
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

from tac.candidate_seal import CONSISTENT, check_pin_consistency, repin_receiver

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

N_PAIRS = 600
DEFAULT_BAND = 1e-8
AXIS = "[macOS-CPU advisory, frozen CPU-torch PoseNet, DALI-lineage GT]"
RAW_BYTES = 2 * N_PAIRS * 874 * 1164 * 3
MINIMUM_DECODE_FREE_BYTES = 6 << 30


class Fcd1SchurError(RuntimeError):
    """Fresh-object, repeat, or pose-safety gate refused the compile."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_fact(path: Path) -> dict[str, Any]:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise Fcd1SchurError(f"receipt missing: {path}")
    return json.loads(path.read_text())


def validate_pose_gate(*, d_pose_after: float, d_pose_base: float, band: float) -> bool:
    if not np.isfinite(d_pose_after) or not np.isfinite(d_pose_base):
        raise Fcd1SchurError("pose gate received a non-finite value")
    if band < 0.0:
        raise Fcd1SchurError("pose band must be nonnegative")
    return bool(d_pose_after <= d_pose_base + band)


def run_decode(args: argparse.Namespace) -> int:
    """Retain the candidate receiver output needed by its fresh Schur solve."""
    runtime = Path(args.runtime).resolve()
    archive = runtime / "archive.zip"
    output = Path(args.output).resolve()
    custody_root = Path("/Volumes/APDataStore/pact")
    if not output.is_relative_to(custody_root):
        raise Fcd1SchurError("decode output must remain on the APDataStore custody tier")
    pin = check_pin_consistency(runtime)
    if pin.verdict != CONSISTENT:
        raise Fcd1SchurError(f"runtime pin refused: {pin.summary()}")
    output.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(output).free
    if free < MINIMUM_DECODE_FREE_BYTES:
        raise Fcd1SchurError(f"decode storage preflight: {free} B free < {MINIMUM_DECODE_FREE_BYTES} B")
    receipt_path = output / "DECODE.json"
    raw_path = output / "inflated" / "0.raw"
    if receipt_path.is_file() and raw_path.is_file():
        prior = load_json(receipt_path)
        if prior.get("raw") == file_fact(raw_path) and raw_path.stat().st_size == RAW_BYTES:
            print(json.dumps(prior, indent=2, sort_keys=True))
            return 0
        raise Fcd1SchurError("existing decode receipt or raw payload drifted")

    extracted = output / "archive"
    extracted.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as handle:
        if handle.namelist() != ["p"]:
            raise Fcd1SchurError("archive must contain exactly payload p")
        payload = handle.read("p")
    payload_path = extracted / "p"
    temporary_payload = payload_path.with_suffix(".partial")
    temporary_payload.write_bytes(payload)
    os.replace(temporary_payload, payload_path)
    file_list = output / "public_test_video_names.txt"
    file_list.write_text("0.mkv\n")

    from tools.fire_local_advisory import PATH_TAIL, generate_pyshim

    shim = generate_pyshim(output / "pyshim", Path(sys.executable))
    env = {
        **os.environ,
        "PYTHONDONTWRITEBYTECODE": "1",
        "PATH": f"{shim.parent}:{PATH_TAIL}",
    }
    (output / "inflated").mkdir(parents=True, exist_ok=True)
    stdout_path = output / "inflate.stdout.log"
    stderr_path = output / "inflate.stderr.log"
    started = time.time()
    with stdout_path.open("a") as stdout, stderr_path.open("a") as stderr:
        process = subprocess.run(
            [
                str(Path(sys.executable)),
                "tools/safe_run.py",
                "--rss-mb",
                "32768",
                "--timeout",
                str(args.timeout),
                "--poll",
                "5",
                "--label",
                "fcd1_decode",
                "--projected-gib",
                "6",
                "--child-pidfile",
                str(output / "safe_child.pid"),
                "--status-receipt",
                str(output / "safe_run_status.json"),
                "--",
                "bash",
                str(runtime / "inflate.sh"),
                str(extracted),
                str(output / "inflated"),
                str(file_list),
            ],
            cwd=Path(__file__).resolve().parents[1],
            env=env,
            stdout=stdout,
            stderr=stderr,
            check=False,
        )
    if process.returncode != 0:
        raise Fcd1SchurError(f"receiver failed rc={process.returncode}; retained logs at {output}")
    if not raw_path.is_file() or raw_path.stat().st_size != RAW_BYTES:
        raise Fcd1SchurError(f"receiver raw payload missing or wrong size: {raw_path}")
    result = {
        "schema": "ddm_fcd1_decode.v1",
        "axis": "[macOS-CPU advisory / real public receiver to uint8 payload, no scorer or score]",
        "score_claim": False,
        "runtime": str(runtime),
        "archive": file_fact(archive),
        "extracted_payload": file_fact(payload_path),
        "raw": file_fact(raw_path),
        "stdout": file_fact(stdout_path),
        "stderr": file_fact(stderr_path),
        "elapsed_seconds": time.time() - started,
        "resume_checkpoint_dir": str(output / "inflated" / ".f26_decode_checkpoints"),
    }
    atomic_json(receipt_path, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def run_instrument(args: argparse.Namespace) -> int:
    """Delegate a jg5 mode after binding every candidate identity dynamically."""
    runtime = Path(args.runtime).resolve()
    archive = runtime / "archive.zip"
    raw = Path(args.raw).resolve()
    if not archive.is_file() or not raw.is_file():
        raise Fcd1SchurError("instrument requires a staged archive and its retained raw decode")
    observed_sha = sha256_file(archive)
    observed_bytes = archive.stat().st_size
    if observed_sha != args.archive_sha256:
        raise Fcd1SchurError(f"runtime archive sha {observed_sha} != --archive-sha256 {args.archive_sha256}")
    if args.raw_sha256 and sha256_file(raw) != args.raw_sha256:
        raise Fcd1SchurError("raw decode sha256 drifted")
    pin = check_pin_consistency(runtime)
    if pin.verdict != CONSISTENT:
        raise Fcd1SchurError(f"runtime pin refused: {pin.summary()}")

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import ddm_jg5_pose_resolve_on_edited_renders as jg5

    original = jg5.load_candidate_instrument
    jg5.CANDIDATE_RUNTIME = runtime
    jg5.CANDIDATE_ARCHIVE_SHA256 = observed_sha
    jg5.CANDIDATE_ARCHIVE_BYTES = observed_bytes
    jg5.CANDIDATE_RAW = raw

    def rebound(
        *,
        runtime: Path = runtime,
        expect_archive_sha256: str = observed_sha,
        raw_path: Path = raw,
        expect_raw_sha256: str | None = None,
    ):
        return original(
            runtime=runtime,
            expect_archive_sha256=expect_archive_sha256,
            raw_path=raw_path,
            expect_raw_sha256=expect_raw_sha256,
        )

    jg5.load_candidate_instrument = rebound
    rest = list(args.jg5_argv)
    if rest and rest[0] == "--":
        rest = rest[1:]
    if not rest:
        raise Fcd1SchurError("no jg5 mode supplied after --")
    if "--device" not in rest:
        rest.extend(["--device", args.device])
    else:
        device_index = rest.index("--device")
        if device_index + 1 >= len(rest) or rest[device_index + 1] != args.device:
            raise Fcd1SchurError("FCD1 and delegated JG5 --device values differ")
    return_code = int(jg5.main(rest))
    if return_code:
        return return_code

    def option_value(flag: str) -> str | None:
        try:
            return rest[rest.index(flag) + 1]
        except (ValueError, IndexError):
            return None

    output_value = option_value("--out")
    receipt_names = {
        "control": "CONTROL.json",
        "baseline": "BASELINE.json",
        "gn": "SUMMARY.json",
        "refine": "SUMMARY.json",
        "waterfill": "WATERFILL.json",
        "close": "CLOSE.json",
    }
    receipt_name = receipt_names.get(rest[0])
    if output_value and receipt_name:
        receipt_path = Path(output_value) / receipt_name
        receipt = load_json(receipt_path)
        binding: dict[str, Any] = {
            "runtime_archive": file_fact(archive),
            "candidate_raw": file_fact(raw),
            "instrument_wrapper": file_fact(Path(__file__)),
        }
        base_raw_value = option_value("--base-raw")
        if base_raw_value:
            binding["base_raw"] = file_fact(Path(base_raw_value))
        receipt["fcd1_binding"] = binding
        atomic_json(receipt_path, receipt)
    return 0


def stage_published_runtime(body_runtime: Path, archive: Path, destination: Path) -> dict[str, Any]:
    if destination.exists():
        raise Fcd1SchurError(f"publish destination already exists: {destination}")
    temporary = destination.with_name(destination.name + ".partial")
    if temporary.exists():
        raise Fcd1SchurError(f"partial publish exists: {temporary}")
    shutil.copytree(body_runtime, temporary, copy_function=shutil.copy2)
    shutil.copy2(archive, temporary / "archive.zip")
    repin_receiver(temporary)
    verdict = check_pin_consistency(temporary)
    if verdict.verdict != CONSISTENT:
        raise Fcd1SchurError(f"published runtime pin refused: {verdict.summary()}")
    os.replace(temporary, destination)
    return {
        "runtime": str(destination),
        "archive": file_fact(destination / "archive.zip"),
        "pin_consistency": verdict.verdict,
    }


def run_publish(args: argparse.Namespace) -> int:
    """Assert fresh-object pose safety and repeat identity before publication."""
    body_runtime = Path(args.body_runtime).resolve()
    body_archive = body_runtime / "archive.zip"
    if not body_archive.is_file():
        raise Fcd1SchurError(f"candidate body missing: {body_archive}")
    body_fact = file_fact(body_archive)
    body_pin = check_pin_consistency(body_runtime)
    if body_pin.verdict != CONSISTENT:
        raise Fcd1SchurError(f"candidate body runtime pin refused: {body_pin.summary()}")

    baseline_path = Path(args.baseline)
    close_path = Path(args.close)
    repeat_path = Path(args.repeat_close)
    baseline = load_json(baseline_path)
    close = load_json(close_path)
    repeat = load_json(repeat_path)
    if baseline.get("pairs") != N_PAIRS:
        raise Fcd1SchurError("baseline is not n600")
    if baseline.get("candidate_archive_sha256") != body_fact["sha256"]:
        raise Fcd1SchurError("baseline was measured on a different candidate body")
    binding = baseline.get("fcd1_binding")
    if not isinstance(binding, dict):
        raise Fcd1SchurError("baseline lacks the fcd1 fresh-object binding")
    if binding.get("runtime_archive") != body_fact:
        raise Fcd1SchurError("baseline runtime archive binding drifted")
    candidate_raw_fact = binding.get("candidate_raw")
    base_raw_fact = binding.get("base_raw")
    for label, fact in (("candidate", candidate_raw_fact), ("base", base_raw_fact)):
        if not isinstance(fact, dict):
            raise Fcd1SchurError(f"baseline lacks its {label} raw binding")
        if file_fact(Path(fact["path"])) != fact:
            raise Fcd1SchurError(f"baseline {label} raw payload drifted")
    for label, receipt in (("close", close), ("repeat", repeat)):
        close_binding = receipt.get("fcd1_binding")
        if not isinstance(close_binding, dict):
            raise Fcd1SchurError(f"{label} lacks the fcd1 fresh-object binding")
        if close_binding.get("runtime_archive") != body_fact:
            raise Fcd1SchurError(f"{label} runtime archive binding drifted")
        if close_binding.get("candidate_raw") != candidate_raw_fact:
            raise Fcd1SchurError(f"{label} candidate raw binding drifted")
        before = receipt.get("body_before_resolve", {})
        if before.get("archive_sha256") != body_fact["sha256"]:
            raise Fcd1SchurError(f"{label} compile used a different candidate body")
        if receipt.get("rows_solved") != N_PAIRS:
            raise Fcd1SchurError(f"{label} compile did not solve n600")
        if not receipt.get("control_identity_rebuild_is_byte_identical"):
            raise Fcd1SchurError(f"{label} body rebuild identity failed")
        if not receipt.get("best", {}).get("frame1_sections", {}).get("frame1_sections_all_identical"):
            raise Fcd1SchurError(f"{label} changed a frame-1-producing section")

    best = close["best"]
    repeat_best = repeat["best"]
    archive = Path(best["archive_path"])
    repeat_archive = Path(repeat_best["archive_path"])
    archive_fact = file_fact(archive)
    repeat_fact = file_fact(repeat_archive)
    if archive_fact["sha256"] != best["archive_sha256"]:
        raise Fcd1SchurError("close receipt archive drifted")
    if repeat_fact["sha256"] != repeat_best["archive_sha256"]:
        raise Fcd1SchurError("repeat close receipt archive drifted")
    if archive_fact["sha256"] != repeat_fact["sha256"]:
        raise Fcd1SchurError("determinism repeat produced different archive bytes")
    if float(best["d_pose_final"]) != float(repeat_best["d_pose_final"]):
        raise Fcd1SchurError("determinism repeat produced a different d_pose")

    base_control = baseline.get("base_odd_frames_control")
    if not isinstance(base_control, dict):
        raise Fcd1SchurError("baseline lacks same-carrier jt21 base control")
    d_pose_base = float(base_control["d_pose_mean"])
    d_pose_after = float(best["d_pose_final"])
    pose_gate = validate_pose_gate(d_pose_after=d_pose_after, d_pose_base=d_pose_base, band=args.pose_band)
    result = {
        "schema": "ddm_fcd1_incompile_schur.v1",
        "axis": AXIS,
        "score_claim": False,
        "promotable": False,
        "body": body_fact,
        "candidate_raw": candidate_raw_fact,
        "base_raw": base_raw_fact,
        "baseline_receipt": file_fact(baseline_path),
        "close_receipt": file_fact(close_path),
        "repeat_close_receipt": file_fact(repeat_path),
        "pose_gate": {
            "assertion": "d_pose_after <= d_pose_base + band",
            "d_pose_base": d_pose_base,
            "d_pose_after": d_pose_after,
            "band": args.pose_band,
            "passed": pose_gate,
        },
        "determinism_repeat": {
            "archive": archive_fact,
            "repeat_archive": repeat_fact,
            "byte_identical": True,
            "d_pose_identical": True,
        },
        "disposition": "PUBLISHABLE" if pose_gate else "REFUSED_POSE_GATE",
        "published": None,
    }
    if not pose_gate:
        # A refusal is a measured terminal result, not an exception-shaped absence.
        # Persist the bound inputs, repeat proof, and failed inequality before raising.
        atomic_json(Path(args.receipt), result)
        print(json.dumps(result, indent=2, sort_keys=True))
        raise Fcd1SchurError(
            f"fresh Schur pose gate failed: {d_pose_after} > "
            f"{d_pose_base} + {args.pose_band}"
        )
    # This assertion is inside the compile/publish path by design.  The explicit
    # exception below preserves fail-closed behaviour even under ``python -O``.
    assert d_pose_after <= d_pose_base + args.pose_band, (
        f"fresh Schur pose gate failed: {d_pose_after} > {d_pose_base} + {args.pose_band}"
    )

    published = stage_published_runtime(body_runtime, archive, Path(args.destination).resolve())
    result["disposition"] = "PUBLISHED"
    result["published"] = published
    atomic_json(Path(args.receipt), result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="stage", required=True)

    decode = sub.add_parser("decode")
    decode.add_argument("--runtime", required=True)
    decode.add_argument("--output", required=True)
    decode.add_argument("--timeout", type=int, default=5400)
    decode.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    decode.set_defaults(func=run_decode)

    instrument = sub.add_parser("instrument")
    instrument.add_argument("--runtime", required=True)
    instrument.add_argument("--archive-sha256", required=True)
    instrument.add_argument("--raw", required=True)
    instrument.add_argument("--raw-sha256", default=None)
    instrument.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    instrument.add_argument("jg5_argv", nargs=argparse.REMAINDER)
    instrument.set_defaults(func=run_instrument)

    publish = sub.add_parser("publish")
    publish.add_argument("--body-runtime", required=True)
    publish.add_argument("--baseline", required=True)
    publish.add_argument("--close", required=True)
    publish.add_argument("--repeat-close", required=True)
    publish.add_argument("--pose-band", type=float, default=DEFAULT_BAND)
    publish.add_argument("--destination", required=True)
    publish.add_argument("--receipt", required=True)
    publish.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    publish.set_defaults(func=run_publish)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    from tac.semantic_pipeline.contracts import require_device

    args.device_binding = require_device(args.device).as_dict()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
