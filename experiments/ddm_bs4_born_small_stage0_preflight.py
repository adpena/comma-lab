#!/usr/bin/env python3
"""Run BS4's exact Stage-0 identity and retained-storage fire gate.

This runner is intentionally a preflight, not the born-small carrier solver.  It
hashes every BODY_RESULT custody record, the exact DX2/BO2/scorer/GT inputs, and
the three reference implementations before deciding whether stages 1--4 may
start.  It also prices the *minimum* payload surface the pinned QS5 retention
form can materialize for random n=32.  A failed identity or storage gate writes
the named Stage-0 checkpoint and refuses before any frame or scorer payload is
materialized.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Final

REPO: Final = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tac.candidate_seal import check_pin_consistency, measure_runtime_digest

OUTPUT: Final = Path("/Volumes/APDataStore/pact/ddm_bs3_born_small_resolved")
BODY_RESULT: Final = OUTPUT / "BODY_RESULT.json"
FIRE_ORDER: Final = OUTPUT / "FIRE_ORDER.json"
CHECKPOINT: Final = OUTPUT / "checkpoints/stage_00_source_preflight.json"
DX2_RUNTIME: Final = Path("/Volumes/APDataStore/pact/ddm_dx2/r7/candidate_runtime_dx2")
DX2_SEAL: Final = Path(
    "/Volumes/APDataStore/pact/ddm_dx2/r7/CANDIDATE_SEAL_dx2_fx5_cabac.json"
)
BO2_RAW: Final = Path(
    "/Volumes/APDataStore/pact/ddm_bo2_born_small_distortion/"
    "rows/hg1_generator_field/work/inflated/0.raw"
)
POSE_WEIGHTS: Final = REPO / "upstream/models/posenet.safetensors"
SEG_WEIGHTS: Final = REPO / "upstream/models/segnet.safetensors"
AXIS: Final = (
    "[macOS-CPU advisory, seeded uniform random n=32 from n600] NON-PROMOTABLE"
)
PAIR_COUNT: Final = 32
DIMENSIONS: Final = 12
CAMERA_H: Final = 874
CAMERA_W: Final = 1164
CAMERA_C: Final = 3
STORAGE_RESERVE_BYTES: Final = 8 * 1024**3

DIRECT_PINS: Final = {
    "body_result": (
        BODY_RESULT,
        None,
        "ea3ce5b18ec88d1451c5cd90cd49afc97ee1e52b67cebfe1524aa7abf49f84f3",
    ),
    "fire_order": (
        FIRE_ORDER,
        None,
        "d684c9bc859f825e5d5341c822dcd8c989f91d3a8e7aef1a44316ced3b333db5",
    ),
    "bo2_born_small_raw": (
        BO2_RAW,
        3_662_409_600,
        "43c359eadd7c6e263adf7a1e2732a2b34948b1db8681bcc1be8f7c493b2ac841",
    ),
    "dx2_candidate_seal": (
        DX2_SEAL,
        None,
        "f3e8970cc2168ed904a8944bbffc43823b02a1ea845aaf790642ce1226a1d13d",
    ),
    "posenet_weights": (
        POSE_WEIGHTS,
        55_835_560,
        "0f3a0874c5c387f990d7b88bd1d7e1f6de35d98b45f2a289989db2c77b9b6576",
    ),
    "segnet_weights": (
        SEG_WEIGHTS,
        38_502_892,
        "68956e328d4c5d875389a1a444870e6bac1c052c9986123827af95c07c6991b6",
    ),
}


class BS4PreflightError(RuntimeError):
    """The retained Stage-0 checkpoint or its governing schema is invalid."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_fact(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"path": str(path.resolve()), "present": False}
    before = path.stat()
    digest = sha256_file(path)
    after = path.stat()
    stable = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    ) == (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    )
    return {
        "path": str(path.resolve()),
        "present": True,
        "bytes": after.st_size,
        "sha256": digest,
        "stable_while_hashed": stable,
    }


def checked_file(
    name: str,
    path: Path,
    expected_bytes: int | None,
    expected_sha256: str,
) -> dict[str, Any]:
    observed = file_fact(path)
    problems: list[str] = []
    if not observed["present"]:
        problems.append("required file is absent")
    else:
        if observed["stable_while_hashed"] is not True:
            problems.append("file identity changed while it was hashed")
        if expected_bytes is not None and int(observed["bytes"]) != expected_bytes:
            problems.append(
                f"bytes expected {expected_bytes}, observed {observed['bytes']}"
            )
        if str(observed["sha256"]) != expected_sha256:
            problems.append(
                f"sha256 expected {expected_sha256}, observed {observed['sha256']}"
            )
    return {
        "name": name,
        "expected": {
            "path": str(path.resolve()),
            "bytes": expected_bytes,
            "sha256": expected_sha256,
        },
        "observed": observed,
        "passed": not problems,
        "problems": problems,
    }


def atomic_json_once(path: Path, value: Any) -> dict[str, Any]:
    payload = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    expected = {
        "path": str(path.resolve()),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        observed = file_fact(path)
        comparable = {key: observed[key] for key in ("path", "bytes", "sha256")}
        if comparable != expected:
            raise BS4PreflightError(f"refusing to replace different checkpoint: {path}")
        return expected
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    try:
        with temporary.open("wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return expected


def additive_checkpoint_path(path: Path, value: Any) -> Path:
    payload = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    expected = {
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    revision = 1
    while True:
        candidate = (
            path
            if revision == 1
            else path.with_name(f"{path.stem}_r{revision}{path.suffix}")
        )
        if not candidate.is_file():
            return candidate
        observed = file_fact(candidate)
        if all(observed[key] == expected[key] for key in expected):
            return candidate
        revision += 1


def _run_receipt(argv: list[str]) -> dict[str, Any]:
    completed = subprocess.run(
        argv,
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        "argv": argv,
        "exit_code": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def body_custody_controls(body: dict[str, Any]) -> list[dict[str, Any]]:
    if body.get("schema") != "ddm_bs3_residual_lifted_body.v1":
        raise BS4PreflightError("BODY_RESULT schema differs")
    controls: list[dict[str, Any]] = []
    seen: set[str] = set()
    rows = list(body.get("custody_records", []))
    rows.extend(body.get("source_pins", {}).values())
    for index, record in enumerate(rows):
        path = Path(str(record["path"]))
        key = str(path.resolve())
        if key in seen:
            continue
        seen.add(key)
        controls.append(
            checked_file(
                f"body_custody_{index:03d}",
                path,
                int(record["bytes"]),
                str(record["sha256"]),
            )
        )
    return controls


def rj2_controls(
    fire_order: dict[str, Any], *, seal_valid: bool
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    references = fire_order.get("reference_implementations", {})
    controls: list[dict[str, Any]] = []
    for name, record in sorted(references.items()):
        controls.append(
            checked_file(
                f"reference_{name}",
                Path(str(record["path"])),
                int(record["bytes"]),
                str(record["sha256"]),
            )
        )
    if not all(row["passed"] for row in controls):
        return controls, {"loaded": False, "reason": "reference source pin differs"}

    # Import only after the fire-order source hashes pass.  The imported PINS are
    # therefore tied to the exact RJ2 source named by the charter rather than to
    # a recalled copy of its constants.
    from experiments import ddm_rj2_joint_renderer_object_change as rj2

    for name, (path, expected_bytes, expected_sha256) in sorted(rj2.PINS.items()):
        controls.append(
            checked_file(
                f"rj2_{name}", path, expected_bytes, expected_sha256
            )
        )
    if not seal_valid:
        return controls, {"loaded": False, "reason": "DX2 candidate seal pin differs"}
    seal = json.loads(DX2_SEAL.read_text())
    expected_runtime = seal.get("runtime")
    if not isinstance(expected_runtime, dict):
        raise BS4PreflightError("DX2 candidate seal carries no runtime identity")
    runtime_digest = measure_runtime_digest(DX2_RUNTIME).to_dict()
    runtime_passed = runtime_digest == {
        key: expected_runtime[key] for key in ("sha256", "file_count", "total_bytes")
    }
    controls.append(
        {
            "name": "dx2_sealed_runtime_digest",
            "expected": expected_runtime,
            "observed": runtime_digest,
            "passed": runtime_passed,
            "problems": []
            if runtime_passed
            else ["shippable runtime digest differs from the pinned DX2 candidate seal"],
        }
    )
    diagnostic_tree = rj2.local_tree_snapshot(DX2_RUNTIME)
    return controls, {
        "loaded": True,
        "sealed_shippable_digest": runtime_digest,
        "diagnostic_tree_including_host_residue": diagnostic_tree,
    }


def storage_waterfall(output: Path) -> dict[str, Any]:
    camera_bytes = CAMERA_H * CAMERA_W * CAMERA_C
    # QS5/QS1 evaluate_codes persists both the uint8 slave and the two-frame
    # uint8 PoseNet input.  This deliberately excludes codes, vectors, masters,
    # JSON, later descent passes, Stage 3, and Stage 4, so it is a lower bound.
    bytes_per_candidate = 3 * camera_bytes
    central_difference_candidates = 1 + 2 * DIMENSIONS
    # At a signed-int12 endpoint each of the three active cube dimensions still
    # has at least three valid offsets in radius 2.  The exact current point may
    # duplicate one cube point, so 3**3 is the safe minimum.
    minimum_integer_cube_candidates = 3**3
    # One full coordinate pass has the current point plus at least one legal
    # signed step for each dimension, even if every coordinate is an endpoint.
    minimum_descent_candidates = 1 + DIMENSIONS
    minimum_candidates_per_pair = (
        1
        + 1
        + central_difference_candidates
        + minimum_integer_cube_candidates
        + minimum_descent_candidates
    )
    minimum_payload_bytes = (
        PAIR_COUNT * minimum_candidates_per_pair * bytes_per_candidate
    )
    required_free_bytes = minimum_payload_bytes + STORAGE_RESERVE_BYTES
    tiers = []
    for path, eligible, reason in (
        (
            output,
            True,
            "charter-mandated additive consumer root",
        ),
        (
            Path("/Volumes/VertigoDataTier/pact"),
            False,
            "charter mandates APDataStore shared root for every BS4 payload",
        ),
        (
            REPO,
            False,
            "local bulk retention requires explicit operator opt-in",
        ),
    ):
        usage = shutil.disk_usage(path)
        tiers.append(
            {
                "path": str(path.resolve()),
                "eligible": eligible,
                "eligibility_reason": reason,
                "free_bytes": usage.free,
                "required_free_bytes": required_free_bytes,
                "passed": eligible and usage.free >= required_free_bytes,
            }
        )
    return {
        "schema": "ddm_bs4_qs5_retention_storage_waterfall.v1",
        "reference_form": (
            "QS5 central difference + damped solve + integer neighbourhood + "
            "one full non-improving coordinate pass; QS1 uncompressed retained npy batches"
        ),
        "selected_pairs": PAIR_COUNT,
        "camera_shape": [CAMERA_H, CAMERA_W, CAMERA_C],
        "bytes_per_candidate_lower_bound": bytes_per_candidate,
        "minimum_candidate_evaluations_per_pair": minimum_candidates_per_pair,
        "minimum_materialized_payload_bytes": minimum_payload_bytes,
        "excluded_from_lower_bound": [
            "codes and pose vectors",
            "born-small masters and semantic fingerprints",
            "JSON/checkpoint framing",
            "all improving coordinate-descent passes after the mandatory terminal pass",
            "resolved carrier/container payloads",
            "three-way SegNet/PoseNet measurement payloads",
        ],
        "reserve_bytes": STORAGE_RESERVE_BYTES,
        "required_free_bytes": required_free_bytes,
        "tiers": tiers,
        "passed": any(row["passed"] for row in tiers),
        "cleanup_policy": (
            "certify-or-block; no BS3 retained tree may be deleted or moved; no partial scorer launch"
        ),
    }


def run(output: Path = OUTPUT) -> dict[str, Any]:
    if output.resolve() != OUTPUT.resolve():
        raise BS4PreflightError(
            "BS4 Stage 0 may write only the charter-mandated APDataStore consumer root"
        )
    direct_controls = [
        checked_file(name, path, expected_bytes, expected_sha256)
        for name, (path, expected_bytes, expected_sha256) in DIRECT_PINS.items()
    ]
    direct_by_name = {row["name"]: row for row in direct_controls}
    if not all(row["passed"] for row in direct_controls[:2]):
        body: dict[str, Any] = {}
        fire_order: dict[str, Any] = {}
        body_controls: list[dict[str, Any]] = []
        implementation_controls: list[dict[str, Any]] = []
        runtime_tree = {"loaded": False, "reason": "governing input differs"}
    else:
        body = json.loads(BODY_RESULT.read_text())
        fire_order = json.loads(FIRE_ORDER.read_text())
        if fire_order.get("schema") != "ddm_bs3_resolved_carrier_fire_order.v1":
            raise BS4PreflightError("FIRE_ORDER schema differs")
        body_controls = body_custody_controls(body)
        implementation_controls, runtime_tree = rj2_controls(
            fire_order,
            seal_valid=bool(direct_by_name["dx2_candidate_seal"]["passed"]),
        )

    pin = check_pin_consistency(DX2_RUNTIME)
    queue = _run_receipt(
        [str(REPO / ".venv/bin/python"), "tools/codex_arm_queue.py", "status"]
    )
    lane = _run_receipt(
        [str(REPO / ".venv/bin/python"), "tools/claim_lane_dispatch.py", "summary"]
    )
    storage = storage_waterfall(output)
    all_controls = direct_controls + body_controls + implementation_controls
    identities_passed = all(row["passed"] for row in all_controls)
    pin_passed = pin.ok
    scorer_slot_free = queue["exit_code"] == 0 and "scorer slot: free" in queue["stdout"]
    trigger_passed = identities_passed and pin_passed and scorer_slot_free

    if not identities_passed or not pin_passed:
        status = "REFUSED_IDENTITY_MISMATCH"
        verdict_scope = "INSTANCE: BS4 exact pinned fire object"
        blockers = [
            "Stage 0 source/runtime identity control failed; #1237 requires refusal before scorer/fire."
        ]
    elif not scorer_slot_free:
        status = "REFUSED_SCORER_SLOT_NOT_FREE"
        verdict_scope = "INSTANCE: BS4 fire scheduling state"
        blockers = [
            "The fleet queue did not report a free scorer slot; the one-full-n600-equivalent rule forbids fire."
        ]
    elif not storage["passed"]:
        status = "REFUSED_STORAGE_PREFLIGHT"
        verdict_scope = "INSTANCE: BS4 random-n32 QS5 exact-retention launch"
        blockers = [
            "The charter-mandated APDataStore tier cannot retain even the conservative minimum QS5 payload surface plus reserve.",
            "Stages 1-4 were not started; deleting or moving BS3 custody and local-bulk fallback are not authorized.",
        ]
    else:
        status = "READY_FOR_STAGE_1"
        verdict_scope = "INSTANCE: BS4 exact pinned fire object"
        blockers = []

    result = {
        "schema": "ddm_bs4_stage_00_source_preflight.v1",
        "stage": 0,
        "status": status,
        "axis": AXIS,
        "verdict_scope": verdict_scope,
        "blockers": blockers,
        "identity_controls": {
            "passed": identities_passed,
            "count": len(all_controls),
            "rows": all_controls,
        },
        "dx2_runtime_pin_consistency": pin.to_dict(),
        "dx2_runtime_tree": runtime_tree,
        "harness_controls": {
            "body_and_reference_pins": identities_passed,
            "runtime_archive_sha_and_bytes": pin_passed,
            "scorer_slot_free": scorer_slot_free,
            "queue": queue,
            "lane_summary": lane,
        },
        "storage_waterfall": storage,
        "fire_trigger_passed_before_storage": trigger_passed,
        "stage_1_through_4_fired": False,
        "scorer_forwards": 0,
        "segnet_forwards": 0,
        "posenet_forwards": 0,
        "modal_invocations": 0,
        "upstream_mutated": False,
        "all_materialized_payloads_retained": True,
        "score_claim": False,
        "promotion_eligible": False,
        "provenance": {
            "argv": sys.argv,
            "cwd": str(Path.cwd()),
            "python": sys.version,
            "platform": platform.platform(),
            "git_head": subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=REPO,
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip(),
            "runner": file_fact(Path(__file__).resolve()),
        },
    }
    checkpoint_base = output / CHECKPOINT.relative_to(OUTPUT)
    checkpoint = atomic_json_once(additive_checkpoint_path(checkpoint_base, result), result)
    result["checkpoint"] = checkpoint
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run(args.output.resolve())
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "READY_FOR_STAGE_1" else 2


if __name__ == "__main__":
    raise SystemExit(main())
