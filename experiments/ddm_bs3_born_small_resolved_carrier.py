#!/usr/bin/env python3
"""Build the residual-lifted born-small body and seal its scorer continuation.

This runner is deliberately scorer-free.  The BS3 queue record says that this
arm does not own the shared scorer, so it closes only the exact byte object:
HG1's four generated semantic streams, an actually encoded zero-correction
residual, the inherited DX2 semantic renderer/carrier/compact-residual sections,
and deterministic primary/repeat containers.  Every raw and coded payload is
retained under the governed APDataStore root.

The runner also emits a seeded uniform n=32 pair selection and an executable
stage specification for MAIN.  It does not claim that the inherited carrier is
the requested re-solved member, and it does not transfer BO2 distortion into
the new object.
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

import numpy as np

REPO: Final = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments import ddm_hg1_heterogeneous_analytic_generator_gate as hg1

OUTPUT: Final = Path("/Volumes/APDataStore/pact/ddm_bs3_born_small_resolved")
HG1_ROOT: Final = Path(
    "/Volumes/APDataStore/pact/ddm_hg1_heterogeneous_analytic_generator_gate"
)
HG1_GENERATED: Final = HG1_ROOT / "retained/generators/generated_tokens.u8"
SEED: Final = 20260826
SAMPLE_N: Final = 32
RATE_DENOMINATOR: Final = 37_545_489
SUB012_BODY_CAP: Final = 137_986
GB1_ARCHIVE_BYTES: Final = 180_215
GB1_SCORE: Final = 0.14811799921260607
AXIS: Final = "[macOS-CPU advisory / scorer-free exact byte measurement]"

SOURCE_PINS: Final = {
    "generated_tokens": (
        HG1_GENERATED,
        117_964_800,
        "2884c5701dc2b2059df0e9f8e4ee3ed81809457b127a48ad3fd3fb6f7a17152b",
    ),
    "road_undrivable": (
        HG1_ROOT / "retained/generators/road_undrivable.raw",
        20_447,
        "4188cbd22d4e31a96f685c9faeea96405b83d050560472902abfdfc8d9fd18cb",
    ),
    "lane": (
        HG1_ROOT / "retained/generators/lane.raw",
        159_395,
        "1633194e9dc48b1469fea3247e115474468d9bfc3f99162d8302a7affe83e3d1",
    ),
    "movable": (
        HG1_ROOT / "retained/generators/movable.raw",
        24_599,
        "d75b86e62c5c63ec769555735ea3a200b1839e25dff4e6a87910e814a8de1454",
    ),
    "mycar": (
        HG1_ROOT / "retained/generators/mycar.raw",
        24_589,
        "3efa81c8f6744afb46acb09d0c7a6f382a33237541b02af3dfdb545fa86de032",
    ),
    "semantic_renderer": (
        HG1_ROOT / "retained/source_sections/source_semantic_renderer.bin",
        30_856,
        "39d1be52ba62933498395c48ce4d9482f37db097d504da76c2a321efe3e4a76f",
    ),
    "inherited_pose_carrier": (
        HG1_ROOT / "retained/source_sections/source_pose_carrier.bin",
        22_010,
        "932b979f5181b331a9099162c6f392f558860b7998c62a36f38c2c99629c9b12",
    ),
    "compact_residual": (
        HG1_ROOT / "retained/source_sections/source_compact_residual.bin",
        96,
        "8ab2fe748ab7d69d2102ba2292289e22bd7ea503f8ae29938e0854ec46ca3da1",
    ),
}


class BS3Error(RuntimeError):
    """A source, payload, coder, receiver, or custody invariant failed."""


def sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def file_fact(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise BS3Error(f"required file is absent: {path}")
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def atomic_bytes(path: Path, payload: bytes) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    expected = {"path": str(path.resolve()), "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
    if path.is_file():
        if file_fact(path) != expected:
            raise BS3Error(f"refusing to replace different retained bytes: {path}")
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
    return file_fact(path)


def retain_json(path: Path, value: Any) -> dict[str, Any]:
    payload = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    return atomic_bytes(path, payload)


def retain_npy(path: Path, value: np.ndarray) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    try:
        with temporary.open("wb") as stream:
            np.save(stream, np.asarray(value), allow_pickle=False)
            stream.flush()
            os.fsync(stream.fileno())
        candidate = file_fact(temporary)
        expected = {
            "path": str(path.resolve()),
            "bytes": candidate["bytes"],
            "sha256": candidate["sha256"],
        }
        if path.is_file():
            if file_fact(path) != expected:
                raise BS3Error(f"refusing to replace different retained array: {path}")
        else:
            os.replace(temporary, path)
        return file_fact(path)
    finally:
        temporary.unlink(missing_ok=True)


def retain_copy(source: Path, destination: Path) -> dict[str, Any]:
    source_fact = file_fact(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        observed = file_fact(destination)
        if observed["bytes"] != source_fact["bytes"] or observed["sha256"] != source_fact["sha256"]:
            raise BS3Error(f"retained copy differs from pinned source: {destination}")
        return observed
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.partial")
    try:
        with source.open("rb") as incoming, temporary.open("wb") as outgoing:
            shutil.copyfileobj(incoming, outgoing, length=8 * 1024 * 1024)
            outgoing.flush()
            os.fsync(outgoing.fileno())
        copied = file_fact(temporary)
        if copied["bytes"] != source_fact["bytes"] or copied["sha256"] != source_fact["sha256"]:
            raise BS3Error(f"copy identity differs: {source} -> {destination}")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return file_fact(destination)


def require_sources() -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for name, (path, expected_bytes, expected_sha256) in SOURCE_PINS.items():
        observed = file_fact(path)
        if observed["bytes"] != expected_bytes or observed["sha256"] != expected_sha256:
            raise BS3Error(f"source pin differs for {name}: {observed}")
        rows[name] = observed
    return rows


def storage_preflight(output: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "STORAGE_PREFLIGHT.json"
    if checkpoint.is_file():
        prior = json.loads(checkpoint.read_text())
        if prior.get("schema") != "ddm_bs3_storage_preflight.v1" or prior.get("passed") is not True:
            raise BS3Error("retained storage preflight is invalid")
        if shutil.disk_usage(output).free < int(prior["reserve_bytes"]):
            raise BS3Error("APDataStore free space fell below the retained reserve")
        return prior
    already = sum(path.stat().st_size for path in output.rglob("*") if path.is_file())
    expected_total = 768 * 1024**2
    reserve = 4 * 1024**3
    required = max(0, expected_total - already) + reserve
    free = shutil.disk_usage(output).free
    result = {
        "schema": "ddm_bs3_storage_preflight.v1",
        "tier": str(output.resolve()),
        "already_retained_bytes": already,
        "expected_total_bytes": expected_total,
        "reserve_bytes": reserve,
        "required_free_bytes": required,
        "free_bytes": free,
        "passed": free >= required,
        "cleanup_policy": "certify-or-block; retain every materialized raw/coded/candidate payload",
    }
    retain_json(checkpoint, result)
    if not result["passed"]:
        raise BS3Error("APDataStore storage preflight failed")
    return result


def command_receipt(argv: list[str], output: Path) -> dict[str, Any]:
    completed = subprocess.run(argv, cwd=REPO, capture_output=True, text=True, check=False)
    payload = (completed.stdout + completed.stderr).encode()
    transcript = atomic_bytes(output, payload)
    return {"argv": argv, "exit_code": completed.returncode, "transcript": transcript}


def build_body(output: Path) -> dict[str, Any]:
    completed_path = output / "BODY_RESULT.json"
    if completed_path.is_file():
        prior = json.loads(completed_path.read_text())
        if prior.get("schema") != "ddm_bs3_residual_lifted_body.v1":
            raise BS3Error("retained body result has a different schema")
        for record in prior["custody_records"]:
            if file_fact(Path(record["path"])) != record:
                raise BS3Error(f"retained body payload drifted: {record['path']}")
        return prior

    storage = storage_preflight(output)
    sources = require_sources()
    retained_sources: dict[str, dict[str, Any]] = {}
    for name in ("generated_tokens", "road_undrivable", "lane", "movable", "mycar"):
        retained_sources[name] = retain_copy(
            Path(sources[name]["path"]), output / "retained/source_payloads" / Path(sources[name]["path"]).name
        )
    for name in ("semantic_renderer", "inherited_pose_carrier", "compact_residual"):
        retained_sources[name] = retain_copy(
            Path(sources[name]["path"]), output / "retained/source_sections" / f"{name}.bin"
        )

    generated_path = Path(retained_sources["generated_tokens"]["path"])
    generated = np.memmap(generated_path, mode="r", dtype=np.uint8, shape=hg1.TOKEN_SHAPE)
    residual_raw = output / "retained/generators/residual_zero.raw"
    residual = hg1.encode_residual(
        generated,
        generated,
        residual_raw,
        protected=None,
        order="frame_raster",
    )
    del generated
    if residual["corrections"] != 0:
        raise BS3Error("residual-lifted body unexpectedly encoded corrections")

    raw_paths = {
        "road_undrivable": Path(retained_sources["road_undrivable"]["path"]),
        "lane": Path(retained_sources["lane"]["path"]),
        "movable": Path(retained_sources["movable"]["path"]),
        "mycar": Path(retained_sources["mycar"]["path"]),
        "residual": residual_raw,
    }
    races = [hg1.coder_race(name, raw_paths[name], output) for name in hg1.STREAMS]
    packet_path = output / "retained/body/born_small_zero_residual.packet"
    packet_fact = hg1.build_packet(races, packet_path)
    packet = packet_path.read_bytes()
    direct = hg1.decode_packet_to_file(packet, output / "retained/body/direct_decode_tokens.u8")
    if direct["corrections"] != 0 or direct["sha256"] != retained_sources["generated_tokens"]["sha256"]:
        raise BS3Error("zero-residual packet does not decode to the retained generated field")

    semantic = Path(retained_sources["semantic_renderer"]["path"]).read_bytes()
    carrier = Path(retained_sources["inherited_pose_carrier"]["path"]).read_bytes()
    compact = Path(retained_sources["compact_residual"]["path"]).read_bytes()
    archive_path = output / "retained/body/born_small_inherited_carrier.zip"
    repeat_path = output / "retained/body/born_small_inherited_carrier.repeat.zip"
    hg1.build_complete_archive(archive_path, packet, semantic, carrier, compact)
    hg1.build_complete_archive(repeat_path, packet, semantic, carrier, compact)
    archive_fact = file_fact(archive_path)
    repeat_fact = file_fact(repeat_path)
    if archive_fact["sha256"] != repeat_fact["sha256"]:
        raise BS3Error("primary and repeat born-small containers differ")
    parsed_sections, parsed_packet = hg1.parse_complete_archive(archive_path)
    if parsed_sections != {
        "semantic_renderer": semantic,
        "pose_carrier": carrier,
        "compact_residual": compact,
    } or parsed_packet != packet:
        raise BS3Error("born-small container parse-back differs")
    archive_decode = hg1.decode_packet_to_file(
        parsed_packet, output / "retained/body/archive_parseback_tokens.u8"
    )
    if archive_decode["sha256"] != retained_sources["generated_tokens"]["sha256"]:
        raise BS3Error("archive parse-back field differs from the retained born-small field")

    rng = np.random.default_rng(SEED)
    pair_ids = np.sort(rng.choice(hg1.N_PAIRS, size=SAMPLE_N, replace=False)).astype(np.int32)
    selection_npy = retain_npy(output / "retained/selection/random_pair_ids.int32.npy", pair_ids)
    selection_json = {
        "schema": "ddm_bs3_seeded_uniform_selection.v1",
        "population_pairs": hg1.N_PAIRS,
        "sample_pairs": SAMPLE_N,
        "seed": SEED,
        "selection_mode": "uniform without replacement from all n600 pairs; sorted after draw",
        "prefix": False,
        "pair_ids": pair_ids.tolist(),
        "npy": selection_npy,
    }
    selection_receipt = retain_json(output / "retained/selection/SELECTION.json", selection_json)

    race_rows = []
    for race in races:
        winner = race["winner"]
        race_rows.append(
            {
                "member": race["name"],
                "raw": race["raw"],
                "winner": winner,
                "coded": race["coders"][winner]["coded"],
                "all_coders": race["coders"],
            }
        )
    coded_stream_bytes = sum(int(row["coded"]["bytes"]) for row in race_rows)
    framing_bytes = archive_fact["bytes"] - (
        len(semantic) + len(carrier) + len(compact) + coded_stream_bytes
    )
    if framing_bytes <= 0:
        raise BS3Error("container framing arithmetic is non-positive")

    queue_receipt = command_receipt(
        [str(REPO / ".venv/bin/python"), "tools/codex_arm_queue.py", "status"],
        output / "receipts/codex_arm_queue_status.txt",
    )
    lane_receipt = command_receipt(
        [str(REPO / ".venv/bin/python"), "tools/claim_lane_dispatch.py", "summary"],
        output / "receipts/lane_dispatch_summary.txt",
    )

    body = {
        "schema": "ddm_bs3_residual_lifted_body.v1",
        "status": "BODY_BYTE_CLOSED_SCORER_NOT_OWNED",
        "axis": AXIS,
        "seed": SEED,
        "storage": storage,
        "source_pins": sources,
        "retained_sources": retained_sources,
        "body": {
            "archive": archive_fact,
            "archive_repeat": repeat_fact,
            "repeat_equal": True,
            "packet": packet_fact,
            "direct_decode": direct,
            "archive_parseback_decode": archive_decode,
            "generated_field_exact": True,
            "residual_corrections": 0,
            "real_coder_rows": race_rows,
            "coded_stream_bytes": coded_stream_bytes,
            "semantic_renderer_bytes": len(semantic),
            "inherited_pose_carrier_bytes": len(carrier),
            "compact_residual_bytes": len(compact),
            "container_framing_bytes": framing_bytes,
        },
        "sample_selection": selection_json,
        "rate_arithmetic": {
            "archive_bytes": archive_fact["bytes"],
            "strict_sub012_cap_bytes_at_current_gb1_distortion": SUB012_BODY_CAP,
            "bytes_under_cap_before_carrier_resolve": SUB012_BODY_CAP - archive_fact["bytes"],
            "gb1_archive_bytes": GB1_ARCHIVE_BYTES,
            "bytes_saved_vs_gb1_before_carrier_resolve": GB1_ARCHIVE_BYTES - archive_fact["bytes"],
            "rate_s": 25.0 * archive_fact["bytes"] / RATE_DENOMINATOR,
            "gb1_score": GB1_SCORE,
            "distortion": "UNMEASURED_ON_THIS_OBJECT",
            "net_s": "UNMEASURED_ON_THIS_OBJECT",
        },
        "governance": {
            "queue_status": queue_receipt,
            "lane_status": lane_receipt,
            "scorer_slot_owned_by_bs3": False,
            "scorer_run": False,
            "modal_run": False,
            "upstream_mutated": False,
        },
        "selection_receipt": selection_receipt,
        "all_materialized_payloads_retained": True,
        "score_claim": False,
        "promotion_eligible": False,
        "provenance": {
            "argv": sys.argv,
            "cwd": str(Path.cwd()),
            "python": sys.version,
            "platform": platform.platform(),
            "git_head": subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True, check=True
            ).stdout.strip(),
            "runner": file_fact(Path(__file__).resolve()),
        },
    }
    custody_records = []
    for path in sorted(output.rglob("*")):
        if path.is_file() and path != completed_path:
            custody_records.append(file_fact(path))
    body["custody_records"] = custody_records
    retain_json(completed_path, body)
    return body


def build_fire_order(output: Path, body: dict[str, Any]) -> dict[str, Any]:
    pair_selection = body["sample_selection"]
    order = {
        "schema": "ddm_bs3_resolved_carrier_fire_order.v1",
        "disposition": "QUEUED-WITH-A-FIRE-ORDER",
        "owner": "MAIN sole scorer-lane router",
        "consumer_store": str(output.resolve()),
        "fire_trigger": (
            "MAIN grants this exact object scorer ownership, no other full-n600 scorer job is active, "
            "and every BODY_RESULT source/payload SHA-256 revalidates"
        ),
        "scorer_scope": {
            "axis": "[macOS-CPU advisory, seeded uniform random n=32 from n600] NON-PROMOTABLE",
            "selection": pair_selection,
            "chunk_max_pairs": 32,
            "prefix": False,
            "one_full_n600_scorer_at_a_time": True,
        },
        "exact_stage_order": [
            {
                "stage": 0,
                "action": "revalidate all body, DX2 runtime/raw, BO2 born-small raw, scorer-weight, and GT cache identities",
                "checkpoint": "checkpoints/stage_00_source_preflight.json",
            },
            {
                "stage": 1,
                "action": (
                    "for every selected pair retain the born-small frame-1 master from the exact BO2 receiver output "
                    "and prove its semantic-field fingerprint matches BODY_RESULT"
                ),
                "checkpoint": "checkpoints/stage_10_exact_born_small_masters.json",
            },
            {
                "stage": 2,
                "action": (
                    "decode the exact DX2 600x12 signed-int12 carrier; for each selected pair compute the QS5 "
                    "central-difference 6x12 PoseNet Jacobian against that born-small master, damped solve, integer "
                    "neighbourhood, and exact coordinate descent until one full non-improving pass; retain every "
                    "code batch, frame, scorer input, pose vector, objective, and object fingerprint"
                ),
                "checkpoint": "checkpoints/stage_20_qs5_exact_pair_solves.json",
            },
            {
                "stage": 3,
                "action": (
                    "re-encode the full lattice with RJ2's identity-controlled CPR1->CAP1->DX2->RR5->Brotli q9/w16 "
                    "production chain; replace only the carrier section; build repeat-equal container and receiver-parse codes"
                ),
                "checkpoint": "checkpoints/stage_30_resolved_carrier_container.json",
            },
            {
                "stage": 4,
                "action": (
                    "measure realized d_seg and d_pose on the same selected pairs for GB1/DX2 base, BO2 born-small "
                    "with stale carrier, and born-small with fresh exact solve; recompute S from components and real bytes"
                ),
                "checkpoint": "checkpoints/stage_40_three_way_measurement.json",
            },
            {
                "stage": 5,
                "action": (
                    "screen a deterministic one-hidden-layer nonlinear carrier model on a fixed train/holdout split of "
                    "the exact solved code deltas; retain model weights, real-coded payloads, predictions, rendered frames, "
                    "and holdout scorer outputs; label SCREEN only and never use it for family closure"
                ),
                "checkpoint": "checkpoints/stage_50_learned_implicit_screen.json",
            },
        ],
        "reference_implementations": {
            "qs5_exact_solve": file_fact(REPO / "experiments/ddm_qs5_resolve_compensation.py"),
            "dx2_carrier_surface_and_production_encoder": file_fact(
                REPO / "experiments/ddm_rj2_joint_renderer_object_change.py"
            ),
            "dx2_carrier_decode": file_fact(
                REPO / "experiments/ddm_po1_t4_error_feedback_pose_compensation.py"
            ),
        },
        "refusals": [
            "do not use the CP135 carrier: it is a different byte object from DX2/HG1",
            "do not transfer BO2 aggregate distortion into the resolved member",
            "do not substitute an autograd-only or fitted-linear overlay for the QS5 exact integer solve",
            "do not promote a random-n32 screen or use it to close the learned-carrier family",
        ],
        "expected_exit": (
            "ADMIT only if measured resolved-carrier delta beats both baselines and real-byte arithmetic is negative; "
            "otherwise emit an INSTANCE or FORMULATION-scoped refusal with the Amendment-2 form-grade table"
        ),
    }
    retain_json(output / "FIRE_ORDER.json", order)
    return order


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument(
        "--resume-from",
        type=Path,
        default=OUTPUT / "BODY_RESULT.json",
        help="must name the BODY_RESULT path under --output",
    )
    args = parser.parse_args()
    output = args.output.resolve()
    if args.resume_from.resolve() != output / "BODY_RESULT.json":
        raise BS3Error("--resume-from must be OUTPUT/BODY_RESULT.json")
    body = build_body(output)
    fire_order = build_fire_order(output, body)
    result = {
        "status": body["status"],
        "archive": body["body"]["archive"],
        "rate_arithmetic": body["rate_arithmetic"],
        "fire_order": {
            "disposition": fire_order["disposition"],
            "owner": fire_order["owner"],
            "consumer_store": fire_order["consumer_store"],
            "fire_trigger": fire_order["fire_trigger"],
        },
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
