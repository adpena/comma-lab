#!/usr/bin/env python3
"""Build and locally gate the RFO1 MICRO35 union and its retained successors.

This is a macOS-CPU advisory builder.  It composes the six exact QS2 token
objects with RE1's admitted pair-96 singleton and the sign-verified pair-7
singleton, solves Schur compensation against the final token frames, closes
HP3/RC64 once, applies the receiver-identical HP4 order-0 repack, and recounts
the changed pairs through the frozen CPU scorers.  The successor mode builds
the pose-constrained pair-105 and drop-532 variants, conditionally composes
them, and seals at most one fire order.  It never dispatches a remote job and
never promotes the local results.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import io
import json
import math
import os
import shutil
import struct
import sys
import zipfile
from collections.abc import Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Final

import numpy as np

REPO: Final = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments import ddm_jo1_joint_probability_object as jo1
from experiments import ddm_js6_seg_representation_join as js6
from experiments import ddm_qs1_frame0_schur_coupled_solve as qs1
from experiments import ddm_qs2_compensation_overlay_runtime as overlay_codec
from experiments import ddm_qs2_compensation_rate_rung as qs2
from experiments import ddm_qs3_saturation_compose as qs3
from experiments import ddm_qs5_resolve_compensation as qs5

OUTPUT: Final = Path("/Volumes/VertigoDataTier/pact/ddm_mc35_20260814")
MC36_OUTPUT: Final = Path("/Volumes/VertigoDataTier/pact/ddm_mc36_20260814")
PAIR105_OUTPUT: Final = Path(
    "/Volumes/VertigoDataTier/pact/ddm_mc35_successor_pair105"
)
DROP532_OUTPUT: Final = Path(
    "/Volumes/VertigoDataTier/pact/ddm_mc35_successor_drop532"
)
COMPOSED_OUTPUT: Final = Path(
    "/Volumes/VertigoDataTier/pact/ddm_mc35_successor_drop532_pair105"
)
MC36_BULK: Final = Path("/Volumes/APDataStore/pact/ddm_mc36_20260814")
AXIS: Final = "[macOS-CPU advisory frozen CPU-torch SegNet/PoseNet; eight changed pairs over n600] NON-PROMOTABLE"
CP135_BYTES: Final = 186_252
CP135_FLIPS: Final = 34_970
CP135_DPOSE: Final = 6.885642960696714e-6
PIXELS: Final = 600 * 384 * 512
RATE_DENOMINATOR: Final = 37_545_489
FLIP_GATE: Final = 35
BYTE_GATE: Final = 29
POSE_GATE: Final = 5.9739759814e-10
RFO1_COMMIT: Final = "6fab4cd3fc"

QS1_COMPILE: Final = Path(
    "/Volumes/VertigoDataTier/pact/ddm_qs1_20260813/compile_workspace/retained/"
    "candidates/qs1_combined_unique_pairs/primary/QS1_COMPILED_RESULT.json"
)
QS2_COMPILE: Final = Path(
    "/Volumes/VertigoDataTier/pact/ddm_qs2_20260813/candidate/COMPILE_RESULT.json"
)
RE1_SPATIAL: Final = Path(
    "/Volumes/VertigoDataTier/pact/pr135_joint_solve_20260810/probability_object_race/"
    "ddm_re1_20260813/retained/candidates/round_02_distinct_pair_stack/primary/"
    "receiver_state/decoded_spatial_tokens.shipped.bin"
)
RE1_EVENTS: Final = Path(
    "/Volumes/VertigoDataTier/pact/ddm_vd1_20260812/main_harvest/results/EVENT_RESULTS.jsonl"
)
HP4_WINNER_MODEL: Final = Path(
    "/Volumes/VertigoDataTier/pact/ddm_hp4/retained/candidates/order0/brotli_q11/"
    "models.hp4m.bin"
)
HP4_WINNER_RESULT: Final = HP4_WINNER_MODEL.with_name("RESULT.json")

QS2_IDS: Final = (
    "js6_0000_9fbf75d81c43",
    "js6_0072_f790b6493122",
    "js6_0006_92685b3e3e44",
    "js6_0004_06fc74e20d9e",
    "js6_0001_da319a6b65d0",
    "js6_0118_83f376603d6e",
)
QS2_PAIRS: Final = (105, 176, 178, 517, 523, 532)
RE1_IDS: Final = ("ec1_0164_3a4e239de5b9", "ec1_0004_3bc2b69c706c")
RE1_PAIRS: Final = (96, 7)
PAIR105: Final = 105
DROP_PAIR: Final = 532

HP4_HEADER: Final = struct.Struct("<4sBBBBHHHHH")
HP4_MAGIC: Final = b"HP4M"
HP4_VERSION: Final = 1
FRAME_OFFSET_BODY: Final = 13_373
FRAME_BYTES: Final = 2_400


class MC35Error(RuntimeError):
    """A pin, retained payload, receiver, compensation, or gate invariant failed."""


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def require_record(record: dict[str, Any]) -> Path:
    path = Path(record["path"])
    if qs1.file_record(path) != record:
        raise MC35Error(f"retained record differs: {path}")
    return path


@contextmanager
def arm_lock(output: Path):
    output.mkdir(parents=True, exist_ok=True)
    lock_path = output / "RUN.lock"
    with lock_path.open("a+b") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def checkpoint(output: Path, stage: str, payload: dict[str, Any]) -> dict[str, Any]:
    value = {"schema": "ddm_mc35_checkpoint.v1", "stage": stage, **payload}
    qs1.retain_json(output / "checkpoints" / f"{stage}.json", value)
    return value


def route_variant_bulk(
    *, output: Path, bulk_root: Path, variant: str, expected_bulk_bytes: int
) -> dict[str, Any]:
    """Route large retained stages to APDataStore with an explicit receipt."""
    output.mkdir(parents=True, exist_ok=True)
    bulk_root.mkdir(parents=True, exist_ok=True)
    receipt_path = output / "STORAGE_ROUTING.json"
    if receipt_path.is_file():
        prior = json.loads(receipt_path.read_text())
        if (
            prior.get("schema") != "ddm_mc36_storage_routing.v1"
            or prior.get("variant") != variant
            or prior.get("physical_bulk_store") != str(bulk_root.resolve())
            or int(prior.get("expected_new_bulk_bytes", -1)) != expected_bulk_bytes
            or prior.get("passed") is not True
        ):
            raise MC35Error(f"resumed storage route differs: {receipt_path}")
        retained_link = output / "retained"
        retained_target = (bulk_root / "retained").resolve()
        if (
            not retained_link.is_symlink()
            or retained_link.resolve() != retained_target
        ):
            raise MC35Error(f"resumed storage link differs: {retained_link}")
        addendum_path = output / "STORAGE_ROUTING_ADDENDUM.json"
        if addendum_path.is_file():
            addendum = json.loads(addendum_path.read_text())
            workspace = output / "compile_workspace"
            attempt = output / "compile_workspace_cross_device_attempt"
            expected_attempt = (bulk_root / "compile_workspace").resolve()
            if (
                addendum.get("schema")
                != "ddm_mc36_compile_workspace_same_device_route.v1"
                or addendum.get("variant") != variant
                or workspace.is_symlink()
                or not workspace.is_dir()
                or not attempt.is_symlink()
                or attempt.resolve() != expected_attempt
            ):
                raise MC35Error(f"resumed compile-workspace addendum differs: {output}")
        else:
            workspace_link = output / "compile_workspace"
            workspace_target = (bulk_root / "compile_workspace").resolve()
            if (
                not workspace_link.is_symlink()
                or workspace_link.resolve() != workspace_target
            ):
                raise MC35Error(f"resumed storage link differs: {workspace_link}")
        return prior
    reserve = 8 * 1024**3
    free = shutil.disk_usage(bulk_root).free
    required = expected_bulk_bytes + reserve
    if free < required:
        raise MC35Error(
            f"storage preflight failed for {variant}: {free} < {required} bytes"
        )
    links = {}
    for name in ("retained", "compile_workspace"):
        target = (bulk_root / name).resolve()
        target.mkdir(parents=True, exist_ok=True)
        link = output / name
        if link.is_symlink():
            if link.resolve() != target:
                raise MC35Error(f"{variant} bulk route differs: {link} -> {link.resolve()}")
        elif link.exists():
            raise MC35Error(f"{variant} bulk route is a non-symlink path: {link}")
        else:
            link.symlink_to(target, target_is_directory=True)
        links[name] = {
            "logical_path": str(link.absolute()),
            "physical_path": str(target),
            "kind": "directory_symlink",
        }
    receipt = {
        "schema": "ddm_mc36_storage_routing.v1",
        "variant": variant,
        "logical_store": str(output.resolve()),
        "physical_bulk_store": str(bulk_root.resolve()),
        "links": links,
        "free_bytes_before": free,
        "expected_new_bulk_bytes": expected_bulk_bytes,
        "reserve_bytes": reserve,
        "required_free_bytes": required,
        "passed": True,
        "reason": (
            "Vertigo had insufficient headroom for both fully retained variants; "
            "APDataStore is the governed second SSD tier"
        ),
        "cleanup_policy": "certify-or-block; no payload is deleted or moved",
    }
    qs1.retain_json(receipt_path, receipt)
    return receipt


def migrate_compile_workspace_to_hardlink_device(
    *, output: Path, variant: str, source_payload: Path
) -> dict[str, Any]:
    """Keep JO1's hardlink workspace on the source payload's filesystem."""
    addendum_path = output / "STORAGE_ROUTING_ADDENDUM.json"
    if addendum_path.is_file():
        prior = json.loads(addendum_path.read_text())
        if prior.get("variant") != variant:
            raise MC35Error("compile-workspace route variant differs on resume")
        return prior
    workspace = output / "compile_workspace"
    if not workspace.is_symlink():
        if not workspace.is_dir():
            raise MC35Error(f"compile workspace is absent: {workspace}")
        if workspace.stat().st_dev != source_payload.stat().st_dev:
            raise MC35Error("compile workspace cannot hard-link the JO1 source payload")
        return {
            "schema": "ddm_mc36_compile_workspace_same_device_route.v1",
            "variant": variant,
            "migration_required": False,
            "workspace": str(workspace.resolve()),
            "source_payload": qs1.file_record(source_payload),
        }
    physical_attempt = workspace.resolve()
    if physical_attempt.stat().st_dev == source_payload.stat().st_dev:
        return {
            "schema": "ddm_mc36_compile_workspace_same_device_route.v1",
            "variant": variant,
            "migration_required": False,
            "workspace": str(physical_attempt),
            "source_payload": qs1.file_record(source_payload),
        }
    free = shutil.disk_usage(output).free
    required = 4 * 1024**3 + 8 * 1024**3
    if free < required:
        raise MC35Error(
            f"same-device compile workspace lacks reserve: {free} < {required}"
        )
    attempt_link = output / "compile_workspace_cross_device_attempt"
    if attempt_link.exists() or attempt_link.is_symlink():
        raise MC35Error(f"cross-device attempt custody path already exists: {attempt_link}")
    os.replace(workspace, attempt_link)
    workspace.mkdir()
    if workspace.stat().st_dev != source_payload.stat().st_dev:
        raise MC35Error("migrated compile workspace is still cross-device")
    addendum = {
        "schema": "ddm_mc36_compile_workspace_same_device_route.v1",
        "variant": variant,
        "migration_required": True,
        "source_payload": qs1.file_record(source_payload),
        "source_device": source_payload.stat().st_dev,
        "failed_attempt_physical_store": str(physical_attempt),
        "failed_attempt_logical_link": str(attempt_link.absolute()),
        "replacement_workspace": str(workspace.resolve()),
        "replacement_device": workspace.stat().st_dev,
        "free_bytes_before": free,
        "required_free_bytes": required,
        "failed_attempt_payload_preserved": True,
        "payload_deleted_or_moved": False,
        "reason": "JO1 unchanged probabilities require same-filesystem hard links",
    }
    qs1.retain_json(addendum_path, addendum)
    return addendum


def preflight(output: Path) -> dict[str, Any]:
    prior_path = output / "checkpoints/stage_00_preflight.json"
    if prior_path.is_file():
        prior = json.loads(prior_path.read_text())
        for name, record in prior["sources"].items():
            if name == "runner" and qs1.file_record(Path(__file__).resolve()) != record:
                current_runner = qs1.file_record(Path(__file__).resolve())
                qs1.retain_json(
                    output
                    / f"SOURCE_REVISION_{current_runner['sha256'][:12]}.json",
                    {
                        "schema": "ddm_mc35_source_revision.v1",
                        "prior_runner": record,
                        "current_runner": current_runner,
                        "reason": "tighten fresh-object assertion to admit hash-bound RE1 objects without fabricating JS6-bank entries",
                        "completed_payloads_reused": True,
                    },
                )
                continue
            require_record(record)
        return prior
    free = shutil.disk_usage(output).free
    if free < 8 * 1024**3:
        raise MC35Error(f"storage preflight failed: {free} bytes free")
    sources = {
        "cp135_archive": qs1.require_file(
            qs1.CP135_ARCHIVE,
            expected_bytes=CP135_BYTES,
            expected_sha256=qs1.CP135_ARCHIVE_SHA256,
        ),
        "qs1_compile": qs1.require_file(QS1_COMPILE),
        "qs2_compile": qs1.require_file(QS2_COMPILE),
        "re1_spatial": qs1.require_file(RE1_SPATIAL),
        "re1_events": qs1.require_file(RE1_EVENTS),
        "hp4_winner_model": qs1.require_file(
            HP4_WINNER_MODEL,
            expected_bytes=70_820,
            expected_sha256="47f97fff5463f5409bf39bb881c5dd246e75653e5cb081084e11d6a0976be57d",
        ),
        "hp4_winner_result": qs1.require_file(HP4_WINNER_RESULT),
        "base_argmax": qs1.require_file(
            qs3.QS1_BASE_FIELD, expected_sha256=qs3.EXPECTED_BASE_SHA256
        ),
        "gt_argmax": qs1.require_file(
            qs3.QS1_GT_FIELD, expected_sha256=qs3.EXPECTED_GT_SHA256
        ),
        "cp135_raw": qs1.require_file(
            qs1.CP135_RAW,
            expected_bytes=qs1.RAW_BYTES,
            expected_sha256=qs1.CP135_RAW_SHA256,
        ),
        "base_pose": qs1.require_file(qs1.CP135_BASE_POSE),
        "gt_pose": qs1.require_file(qs1.GT_POSE),
        "runner": qs1.require_file(Path(__file__).resolve()),
    }
    return checkpoint(
        output,
        "stage_00_preflight",
        {
            "axis": AXIS,
            "free_bytes": free,
            "required_free_bytes": 8 * 1024**3,
            "rfo1_commit": RFO1_COMMIT,
            "sources": sources,
            "resumable_from": str(output.resolve()),
            "remote_dispatch": False,
        },
    )


def _event_rows() -> dict[str, dict[str, Any]]:
    wanted = set(RE1_IDS)
    rows: dict[str, dict[str, Any]] = {}
    with RE1_EVENTS.open() as stream:
        for line in stream:
            row = json.loads(line)
            if row.get("proposal_id") in wanted:
                rows[str(row["proposal_id"])] = row
    if set(rows) != wanted:
        raise MC35Error(f"RE1 singleton evidence is incomplete: {sorted(rows)}")
    return rows


def build_supports(output: Path) -> dict[str, Any]:
    result_path = output / "OBJECT_SUPPORTS.json"
    if result_path.is_file():
        prior = json.loads(result_path.read_text())
        for row in prior["rows"]:
            require_record(row["candidate_tokens"])
        return prior
    qs1_compile = json.loads(QS1_COMPILE.read_text())
    if tuple(qs1_compile["selected_proposal_ids"]) != QS2_IDS:
        raise MC35Error("QS1/QS2 selected token objects differ from the charter")
    if tuple(qs1_compile["selected_pairs"]) != QS2_PAIRS:
        raise MC35Error("QS1/QS2 selected pairs differ from the charter")
    rows: list[dict[str, Any]] = []
    for proposal_id, pair in zip(QS2_IDS, QS2_PAIRS, strict=True):
        source = qs1.JS6_BANK / "proposals" / proposal_id
        proposal = json.loads((source / "proposal.json").read_text())
        path = source / "candidate_tokens.uint8.npy"
        rows.append(
            {
                "proposal_id": proposal_id,
                "bank": "QS2",
                "pair": pair,
                "kept_sites": int(proposal["token_site_count"]),
                "token_site_count": int(proposal["token_site_count"]),
                "candidate_tokens": qs1.file_record(path),
                "candidate_tokens_path": str(path.resolve()),
            }
        )
    event_rows = _event_rows()
    re1_spatial = np.memmap(
        RE1_SPATIAL, mode="r", dtype=np.uint8, shape=(600, 384, 512)
    )
    for proposal_id, pair in zip(RE1_IDS, RE1_PAIRS, strict=True):
        evidence = event_rows[proposal_id]
        if int(evidence["pair"]) != pair or int(evidence["site_count"]) != 1:
            raise MC35Error(f"RE1 singleton pin differs: {proposal_id}")
        path = output / "retained/supports" / proposal_id / "candidate_tokens.uint8.npy"
        record = qs1.retain_npy(path, np.asarray(re1_spatial[pair]).copy())
        rows.append(
            {
                "proposal_id": proposal_id,
                "bank": "RE1",
                "pair": pair,
                "kept_sites": 1,
                "token_site_count": 1,
                "candidate_tokens": record,
                "candidate_tokens_path": record["path"],
                "sign_verified_singleton": {
                    "axis": evidence["axis"],
                    "net_flip_gain_base_minus_candidate": int(
                        evidence["net_flip_gain_base_minus_candidate"]
                    ),
                    "delta_d_pose_global_n600": float(
                        evidence["delta_d_pose_global_n600"]
                    ),
                },
            }
        )
    if len({int(row["pair"]) for row in rows}) != len(rows):
        raise MC35Error("MICRO35 requires eight distinct temporal pairs")
    result = {
        "schema": "ddm_mc35_object_supports.v1",
        "selection_mode": "six exact QS2 objects plus RE1 admitted pair-96 singleton and smallest distinct-pair sign-verified singleton",
        "rows": rows,
        "pairs": [int(row["pair"]) for row in rows],
        "token_site_overlap_count": 0,
        "overlap_proof": "all eight support frames have distinct pair indices",
        "all_payloads_retained": True,
    }
    qs1.retain_json(result_path, result)
    checkpoint(output, "stage_10_supports", {"supports": qs1.file_record(result_path)})
    return result


def build_variant_supports(
    *, output: Path, variant: str, drop_pair: int | None
) -> dict[str, Any]:
    """Bind a successor to mc35's retained exact token objects."""
    result_path = output / "OBJECT_SUPPORTS.json"
    if result_path.is_file():
        prior = json.loads(result_path.read_text())
        if prior.get("variant") != variant:
            raise MC35Error(f"resumed support variant differs: {variant}")
        for row in prior["rows"]:
            require_record(row["candidate_tokens"])
        return prior
    source_path = OUTPUT / "OBJECT_SUPPORTS.json"
    source = json.loads(source_path.read_text())
    if source.get("schema") != "ddm_mc35_object_supports.v1":
        raise MC35Error("mc35 support source schema differs")
    rows = [
        dict(row)
        for row in source["rows"]
        if drop_pair is None or int(row["pair"]) != drop_pair
    ]
    for row in rows:
        require_record(row["candidate_tokens"])
    expected = 8 if drop_pair is None else 7
    if len(rows) != expected or len({int(row["pair"]) for row in rows}) != expected:
        raise MC35Error(f"{variant} retained support census differs")
    result = {
        "schema": "ddm_mc36_variant_object_supports.v1",
        "variant": variant,
        "selection_mode": (
            "all eight exact mc35 token objects"
            if drop_pair is None
            else f"seven exact mc35 token objects with pair {drop_pair} removed"
        ),
        "rows": rows,
        "pairs": [int(row["pair"]) for row in rows],
        "dropped_pair": drop_pair,
        "source_mc35_supports": qs1.file_record(source_path),
        "pair_overlap_count": 0,
        "token_site_overlap_count": 0,
        "all_payloads_retained": True,
    }
    qs1.retain_json(result_path, result)
    checkpoint(output, "stage_10_supports", {"supports": qs1.file_record(result_path)})
    return result


def materialize_union(output: Path, supports: dict[str, Any]) -> dict[str, Any]:
    result_path = output / "MATERIALIZED_UNION.json"
    if result_path.is_file():
        prior = json.loads(result_path.read_text())
        require_record(prior["primary"]["spatial_tokens"])
        require_record(prior["repeat_spatial"])
        require_record(prior["entropy_primary"]["token"])
        require_record(prior["entropy_repeat"]["token"])
        return prior
    rows = supports["rows"]
    _jo1, primary_root, primary = qs1._materialize_js6_tokens(
        output=output, name="micro35_union", selected=rows, repeat=False
    )
    _jo1, repeat_root, repeated = qs1._materialize_js6_tokens(
        output=output, name="micro35_union", selected=rows, repeat=True
    )
    if primary["spatial_raw_sha256"] != repeated["spatial_raw_sha256"]:
        raise MC35Error("union token materialization repeat differs")
    primary_entropy = jo1.reclose_candidate(
        output / "compile_workspace", "micro35_union", repeat=False
    )
    repeat_entropy = jo1.reclose_candidate(
        output / "compile_workspace", "micro35_union", repeat=True
    )
    if primary_entropy["token"]["sha256"] != repeat_entropy["token"]["sha256"]:
        raise MC35Error("union HP3/RC64 repeat differs")
    result = {
        "schema": "ddm_mc35_materialized_union.v1",
        "primary_root": str(primary_root.resolve()),
        "repeat_root": str(repeat_root.resolve()),
        "primary": primary,
        "repeat_spatial": repeated["spatial_tokens"],
        "entropy_primary": primary_entropy,
        "entropy_repeat": repeat_entropy,
        "repeat_byte_identical": True,
        "all_payloads_retained": True,
    }
    qs1.retain_json(result_path, result)
    checkpoint(
        output,
        "stage_20_materialized_union",
        {"result": qs1.file_record(result_path)},
    )
    return result


def render_and_solve(
    output: Path, supports: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    # QS1's legacy guard resolves canonical objects only inside JS6.  MICRO35
    # deliberately adds external RE1 objects, so use a stricter union guard:
    # every row must carry the fresh binding and reproduce its content hash.
    qs1.assert_compensation_matches_compile_object = assert_union_compensation
    object_result = {"rows": supports["rows"]}
    rendered, _semantic = qs5._render_exact_masters(output, object_result)
    surface, _carrier = qs1.CP135Surface.load()
    posenet = qs1.load_posenet()
    solver = qs1._load_module("ddm_mc35_joint_pose_solve", qs1.JOINT_SOLVER_SOURCE)
    raw = np.memmap(
        qs1.CP135_RAW,
        mode="r",
        dtype=np.uint8,
        shape=(1_200, qs1.CAMERA_H, qs1.CAMERA_W, 3),
    )
    base_pose = np.load(qs1.CP135_BASE_POSE, allow_pickle=False)
    gt_pose = np.load(qs1.GT_POSE, allow_pickle=False)
    rendered_by_id = {row["proposal_id"]: row for row in rendered}
    solved = []
    for support in supports["rows"]:
        solved.append(
            qs5.solve_exact_object(
                output=output,
                support=support,
                rendered=rendered_by_id[support["proposal_id"]],
                surface=surface,
                posenet=posenet,
                raw=raw,
                base_pose_all=base_pose,
                gt_pose_all=gt_pose,
                solver=solver,
            )
        )
    solved.sort(key=lambda row: int(row["pair"]))
    if any(
        row["solve"]["compensation_object_fingerprint_sha256"]
        != row["compensation_object"]["fingerprint_sha256"]
        for row in solved
    ):
        raise MC35Error("fresh compensation fingerprint assertion failed")
    result = {
        "schema": "ddm_mc35_fresh_compensation.v1",
        "rows": solved,
        "all_eight_pairs_solved_against_final_token_frames": True,
        "stale_compensation_carried": False,
        "all_payloads_retained": True,
    }
    qs1.retain_json(output / "FRESH_COMPENSATION.json", result)
    checkpoint(
        output,
        "stage_30_fresh_compensation",
        {"result": qs1.file_record(output / "FRESH_COMPENSATION.json")},
    )
    return rendered, solved


def load_mc35_solved_rows() -> list[dict[str, Any]]:
    """Load and revalidate mc35's exact-object compensation rows."""
    source = OUTPUT / "FRESH_COMPENSATION.json"
    value = json.loads(source.read_text())
    if value.get("schema") != "ddm_mc35_fresh_compensation.v1":
        raise MC35Error("mc35 compensation source schema differs")
    rows = []
    for source_row in value["rows"]:
        row = dict(source_row)
        assert_union_compensation(row)
        pose_path = (
            OUTPUT
            / "retained/compensation"
            / str(row["proposal_id"])
            / "FINAL_POSE_VECTOR.float32.npy"
        )
        row["variant_final_pose_vector"] = qs1.file_record(pose_path)
        rows.append(row)
    rows.sort(key=lambda row: int(row["pair"]))
    if [int(row["pair"]) for row in rows] != [7, 96, 105, 176, 178, 517, 523, 532]:
        raise MC35Error("mc35 compensation pair census differs")
    return rows


def verify_reused_exact_object(
    *, support: dict[str, Any], rendered: dict[str, Any], solved: dict[str, Any]
) -> dict[str, Any]:
    """Prove a retained solve is for the byte-identical final per-pair object."""
    binding = solved["compensation_object"]
    token = qs1.file_record(Path(support["candidate_tokens"]["path"]))
    master = qs1.file_record(Path(rendered["master_camera"]["path"]))
    expected = qs1.compensation_object_fingerprint(
        pair=int(support["pair"]), semantic_tokens=token, master_camera=master
    )
    if (
        int(solved["pair"]) != int(support["pair"])
        or binding["fingerprint_sha256"] != expected
        or solved["solve"]["compensation_object_fingerprint_sha256"] != expected
    ):
        raise MC35Error(f"reused exact object differs: {support['proposal_id']}")
    return {
        "proposal_id": str(support["proposal_id"]),
        "pair": int(support["pair"]),
        "fingerprint_sha256": expected,
        "semantic_tokens": token,
        "rendered_master": master,
        "source_binding": binding,
        "passed": True,
    }


def _signed_int12_neighbours(codes: np.ndarray) -> tuple[np.ndarray, ...]:
    values = np.asarray(codes, dtype=np.int32)
    candidates = [values.copy()]
    for dimension in range(qs1.DIMENSIONS):
        for delta in (-1, 1):
            candidate = values.copy()
            candidate[dimension] += delta
            if -2048 <= candidate[dimension] <= 2047:
                candidates.append(candidate)
    return tuple(candidates)


def _pose_delta_dpose(
    vector: np.ndarray, baseline: np.ndarray, gt: np.ndarray
) -> float:
    return (
        float(
            np.sum(
                np.square(vector.astype(np.float64) - gt.astype(np.float64))
            )
        )
        - float(
            np.sum(
                np.square(baseline.astype(np.float64) - gt.astype(np.float64))
            )
        )
    ) / (600 * qs1.POSE_DIMENSIONS)


def _best_pose_feasible_index(
    objectives: np.ndarray, composed_deltas: np.ndarray, cap: float
) -> int | None:
    values = np.asarray(objectives, dtype=np.float64)
    deltas = np.asarray(composed_deltas, dtype=np.float64)
    if values.ndim != 1 or deltas.shape != values.shape:
        raise MC35Error("pose-constrained candidate metrics have different geometry")
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(deltas)):
        raise MC35Error("pose-constrained candidate metrics are non-finite")
    feasible = np.flatnonzero(deltas <= cap + 1e-18).tolist()
    if not feasible:
        return None
    return min(
        feasible,
        key=lambda index: (float(values[index]), float(deltas[index]), index),
    )


def _complementary_gate_coverage(
    left: dict[str, bool], right: dict[str, bool]
) -> tuple[set[str], set[str]]:
    if set(left) != set(right):
        raise MC35Error("variant gate names differ")
    left_only = {name for name in left if left[name] and not right[name]}
    right_only = {name for name in left if right[name] and not left[name]}
    return left_only, right_only


def _composition_evidence(
    *,
    pair_result: dict[str, Any],
    drop_result: dict[str, Any],
    pair_recount: dict[str, Any],
    drop_recount: dict[str, Any],
) -> dict[str, Any]:
    """Admit C only when both partial rows expose measured complementary value."""
    pair_gates = pair_result["gates"]
    drop_gates = drop_result["gates"]
    pair_partial = any(pair_gates.values()) and not all(pair_gates.values())
    drop_partial = any(drop_gates.values()) and not all(drop_gates.values())
    evidence = {
        "pair105_supplies_pose_pass": bool(
            pair_gates["delta_dpose_lte_cap"]
            and not drop_gates["delta_dpose_lte_cap"]
        ),
        "drop532_supplies_more_seg_flips": int(
            drop_recount["seg"]["net_flip_gain"]
        )
        > int(pair_recount["seg"]["net_flip_gain"]),
        "drop532_supplies_rate_relief_vs_pair105": int(
            drop_recount["rate"]["candidate_archive_bytes"]
        )
        < int(pair_recount["rate"]["candidate_archive_bytes"]),
    }
    return {
        "pair105_is_partial": pair_partial,
        "drop532_is_partial": drop_partial,
        "measured_complementarity": evidence,
        "justified": pair_partial and drop_partial and all(evidence.values()),
    }


def select_pose_constrained_pair105(
    *,
    output: Path,
    support: dict[str, Any],
    rendered: dict[str, Any],
    fixed_rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Extend the exact-object Schur solve with a GT-relative pose constraint.

    QS5 supplies the real DLS/integer-cube solve.  This extension evaluates the
    signed-int12 neighbors of its final point on the same exact rendered object,
    admits only candidates whose composed n600 delta-d_pose meets the unchanged
    MICRO35 cap, and descends the baseline-return objective inside that feasible
    set.  It is a bounded local constrained solve, not a global optimum claim.
    """
    if int(support["pair"]) != PAIR105:
        raise MC35Error("pose-constrained successor is defined only for pair 105")
    result_root = output / "retained/constrained_compensation" / str(
        support["proposal_id"]
    )
    result_path = result_root / "RESULT.json"
    if result_path.is_file():
        prior = json.loads(result_path.read_text())
        assert_union_compensation(prior)
        require_record(prior["variant_final_pose_vector"])
        return prior
    surface, _carrier = qs1.CP135Surface.load()
    posenet = qs1.load_posenet()
    solver = qs1._load_module("ddm_mc36_joint_pose_solve", qs1.JOINT_SOLVER_SOURCE)
    raw = np.memmap(
        qs1.CP135_RAW,
        mode="r",
        dtype=np.uint8,
        shape=(1_200, qs1.CAMERA_H, qs1.CAMERA_W, 3),
    )
    base_pose_all = np.load(qs1.CP135_BASE_POSE, allow_pickle=False)
    gt_pose_all = np.load(qs1.GT_POSE, allow_pickle=False)
    initial = qs5.solve_exact_object(
        output=output,
        support=support,
        rendered=rendered,
        surface=surface,
        posenet=posenet,
        raw=raw,
        base_pose_all=base_pose_all,
        gt_pose_all=gt_pose_all,
        solver=solver,
    )
    pair = PAIR105
    baseline = np.load(
        output
        / "retained/compensation"
        / str(support["proposal_id"])
        / "stage_10_baseline/ALL_POSE_VECTORS.float32.npy",
        allow_pickle=False,
    )[0]
    gt = gt_pose_all[pair]
    exact_master = np.load(rendered["master_camera"]["path"], allow_pickle=False)
    fixed_delta = sum(
        float(row["pose"]["exact_local_delta_dpose_one_pair_over_n600"])
        for row in fixed_rows
    )
    pair_cap = POSE_GATE - fixed_delta
    current_codes = np.asarray(initial["solve"]["final_codes"], dtype=np.int32)
    current_objective = math.inf
    current_vector = np.full(qs1.POSE_DIMENSIONS, np.nan, dtype=np.float32)
    passes = 0
    entry_from_infeasible_initial = False
    while True:
        candidates = _signed_int12_neighbours(current_codes)
        stage = result_root / f"stage_60_constraint_descent/pass_{passes:04d}"
        vectors = qs1.evaluate_codes(
            surface=surface,
            posenet=posenet,
            codes=candidates,
            master=exact_master,
            pair=pair,
            stage_root=stage,
        )
        objectives = np.mean(
            np.square(vectors.astype(np.float64) - baseline[None].astype(np.float64)),
            axis=1,
        )
        pair_deltas = np.asarray(
            [_pose_delta_dpose(vector, baseline, gt) for vector in vectors],
            dtype=np.float64,
        )
        composed_deltas = pair_deltas + fixed_delta
        feasible = composed_deltas <= POSE_GATE + 1e-18
        qs1.retain_npy(stage / "OBJECTIVES.float64.npy", objectives)
        qs1.retain_npy(stage / "PAIR_DELTA_DPOSE.float64.npy", pair_deltas)
        qs1.retain_npy(stage / "COMPOSED_DELTA_DPOSE.float64.npy", composed_deltas)
        qs1.retain_npy(stage / "FEASIBLE.uint8.npy", feasible.astype(np.uint8))
        best_index = _best_pose_feasible_index(
            objectives, composed_deltas, POSE_GATE
        )
        if best_index is None:
            raise MC35Error("pair-105 exact neighbor set has no pose-feasible candidate")
        best_objective = float(objectives[best_index])
        if math.isinf(current_objective):
            entry_from_infeasible_initial = not bool(feasible[0])
        elif not best_objective < current_objective:
            break
        current_codes = np.asarray(candidates[best_index], dtype=np.int32)
        current_vector = np.asarray(vectors[best_index], dtype=np.float32)
        current_objective = best_objective
        passes += 1
        if passes > 64:
            raise MC35Error("pair-105 constrained descent exceeded 64 strict passes")
    pair_delta = _pose_delta_dpose(current_vector, baseline, gt)
    composed_delta = fixed_delta + pair_delta
    if composed_delta > POSE_GATE + 1e-18:
        raise MC35Error("pair-105 final constrained object violates the pose cap")
    base_codes = np.asarray(initial["solve"]["base_codes"], dtype=np.int32)
    leak = np.asarray(initial["pose"]["leak_vector"], dtype=np.float64)
    residual = current_vector.astype(np.float64) - baseline.astype(np.float64)
    metrics = qs1.cancellation_metrics(leak, residual)
    result = json.loads(json.dumps(initial))
    result.update(
        {
            "schema": "ddm_mc36_pair105_pose_constrained_result.v1",
            "variant": "successor_pair105",
            "solve_scope": (
                "bounded local exact-object Schur plus signed-int12 constrained descent"
            ),
            "verdict_scope": (
                "INSTANCE: pair 105 on the retained mc35 semantic object and CP135 "
                "int12 carrier; macOS-CPU advisory PoseNet"
            ),
        }
    )
    result["solve"].update(
        {
            "unconstrained_initial_final_codes": initial["solve"]["final_codes"],
            "final_codes": current_codes.tolist(),
            "final_code_delta": (current_codes - base_codes).tolist(),
            "final_objective_mse_to_base_pose_vector": current_objective,
            "constraint_descent_full_passes": passes,
            "entry_from_infeasible_initial": entry_from_infeasible_initial,
            "derived_stop": (
                "one complete signed-int12 feasible pass accepted zero strict "
                "baseline-objective improvements"
            ),
        }
    )
    result["pose"] = {
        **metrics,
        "leak_vector": leak.tolist(),
        "residual_vector": residual.tolist(),
        "exact_local_delta_dpose_one_pair_over_n600": pair_delta,
        "fixed_other_pairs_delta_dpose": fixed_delta,
        "pair105_delta_dpose_cap": pair_cap,
        "composed_delta_dpose": composed_delta,
        "composed_pose_gate": POSE_GATE,
        "constraint_passed": True,
    }
    qs1.retain_npy(result_root / "FINAL_CODES.int32.npy", current_codes)
    pose_record = qs1.retain_npy(
        result_root / "FINAL_POSE_VECTOR.float32.npy", current_vector
    )
    result["variant_final_pose_vector"] = pose_record
    qs1.retain_json(result_path, result)
    assert_union_compensation(result)
    checkpoint(
        output,
        "stage_35_pair105_pose_constrained",
        {"result": qs1.file_record(result_path)},
    )
    return result


def assert_union_compensation(row: dict[str, Any]) -> dict[str, Any]:
    """Require a fresh exact-object binding for JS6 and external RE1 rows."""
    proposal_id = str(row["proposal_id"])
    pair = int(row["pair"])
    token_path = Path(str(row.get("candidate_tokens_path", "")))
    token_record = qs1.file_record(token_path)
    binding = row.get("compensation_object")
    if not isinstance(binding, dict):
        raise MC35Error(f"fresh compensation binding is absent: {proposal_id}")
    if binding.get("schema") != "ddm_qs1_compensation_object_binding.v1":
        raise MC35Error(f"fresh compensation schema differs: {proposal_id}")
    if int(binding.get("pair", -1)) != pair:
        raise MC35Error(f"fresh compensation pair differs: {proposal_id}")
    if binding.get("semantic_tokens") != token_record:
        raise MC35Error(f"fresh compensation token record differs: {proposal_id}")
    master_record = binding.get("master_camera")
    if not isinstance(master_record, dict) or qs1.file_record(
        Path(str(master_record.get("path", "")))
    ) != master_record:
        raise MC35Error(f"fresh compensation master record differs: {proposal_id}")
    expected = qs1.compensation_object_fingerprint(
        pair=pair,
        semantic_tokens=token_record,
        master_camera=master_record,
    )
    if binding.get("fingerprint_sha256") != expected:
        raise MC35Error(f"fresh compensation binding fingerprint differs: {proposal_id}")
    if row.get("solve", {}).get("compensation_object_fingerprint_sha256") != expected:
        raise MC35Error(f"fresh compensation solve fingerprint differs: {proposal_id}")
    return {
        "schema": "ddm_qs1_compile_compensation_binding.v1",
        "proposal_id": proposal_id,
        "pair": pair,
        "mode": "EXTERNAL_OR_JS6_EXACT_OBJECT_BOUND_FRESH_SOLVE",
        "semantic_tokens": token_record,
        "master_camera": master_record,
        "fingerprint_sha256": expected,
        "passed": True,
    }


def _hp4_model(final_archive: Path, output: Path) -> tuple[bytes, bytes, dict[str, Any]]:
    streams, suffix = qs2._split_member(qs2._zip_member(final_archive))
    source = HP4_WINNER_MODEL.read_bytes()
    if len(source) < HP4_HEADER.size:
        raise MC35Error("HP4 winner is truncated")
    magic, version, predictor, codec, flags, *lengths = HP4_HEADER.unpack_from(source)
    if (magic, version, predictor, codec, flags) != (HP4_MAGIC, HP4_VERSION, 0, 2, 0):
        raise MC35Error("HP4 winner header differs from order0+Brotli-q11")
    offset = HP4_HEADER.size
    fields = []
    for length in lengths:
        fields.append(source[offset : offset + length])
        offset += length
    if offset != len(source) or fields[1] != b"":
        raise MC35Error("HP4 winner field accounting differs")
    union_fields = (fields[0], fields[1], fields[2], streams[1], streams[2])
    if any(len(field) >= 1 << 16 for field in union_fields):
        raise MC35Error("MICRO35 HP4 field exceeds u16")
    model = HP4_HEADER.pack(
        HP4_MAGIC, HP4_VERSION, 0, 2, 0, *(len(field) for field in union_fields)
    ) + b"".join(union_fields)
    member = model + suffix
    final_root = output / "micro35_candidate/hp4"
    qs1.retain_bytes(final_root / "models.hp4m.bin", model)
    qs1.retain_bytes(final_root / "hpac_remainder.q10.br", fields[0])
    qs1.retain_bytes(final_root / "embedding.q11.br", fields[2])
    qs1.retain_bytes(final_root / "semantic.br", streams[1])
    qs1.retain_bytes(final_root / "carrier.br", streams[2])
    qs1.retain_bytes(final_root / "p", member)
    return model, member, {
        "source_hp4_winner": qs1.file_record(HP4_WINNER_MODEL),
        "model": qs1.file_record(final_root / "models.hp4m.bin"),
        "member": qs1.file_record(final_root / "p"),
        "hp4_receiver_identical_repack": True,
    }


def _patch_hp4_runtime(runtime_root: Path) -> dict[str, Any]:
    path = runtime_root / "runtime/residual_archive.py"
    source = path.read_text()
    if "\n+def _decode_hp4_models" in source:
        repaired = "\n".join(
            line[1:] if line.startswith("+") else line
            for line in source.split("\n")
        )
        if "def _decode_hp4_models" not in repaired:
            raise MC35Error("HP4 runtime repair did not restore its decoder")
        qs2._atomic_replace(path, repaired.encode())
        return qs1.file_record(path)
    constants_old = "SPLIT_MODEL_HEADER = struct.Struct(\"<HHH\")\n"
    constants_new = constants_old + (
        "HP4_MODEL_HEADER = struct.Struct(\"<4sBBBBHHHHH\")\n"
        "HP4_MAGIC = b\"HP4M\"\n"
        "HP4_FRAME_OFFSET_BODY = 13_373\n"
        "HP4_FRAME_BYTES = 2_400\n"
    )
    helper = '''\n\ndef _decode_hp4_models(outer: bytes) -> tuple[bytes, bytes] | None:
    """Restore the exact F24S body from HP4 order0+Brotli-q11 fields."""
    if len(outer) < HP4_MODEL_HEADER.size or not outer.startswith(HP4_MAGIC):
        return None
    magic, version, predictor, codec, flags, *lengths = HP4_MODEL_HEADER.unpack_from(outer)
    model_end = HP4_MODEL_HEADER.size + sum(lengths)
    if (magic, version, predictor, codec, flags) != (HP4_MAGIC, 1, 0, 2, 0):
        raise ResidualArchiveError("unsupported HP4 model")
    if model_end + 96 >= len(outer):
        raise ResidualArchiveError("truncated HP4 model")
    fields = []
    offset = HP4_MODEL_HEADER.size
    for length in lengths:
        fields.append(outer[offset : offset + length])
        offset += length
    remainder = _decompress_brotli(fields[0])
    if fields[1]:
        raise ResidualArchiveError("HP4 order0 predictor parameters must be empty")
    embedding = _decompress_brotli(fields[2])
    if len(embedding) != HP4_FRAME_BYTES:
        raise ResidualArchiveError("HP4 embedding length differs")
    hpac = remainder[:HP4_FRAME_OFFSET_BODY] + embedding + remainder[HP4_FRAME_OFFSET_BODY:]
    semantic = _decompress_brotli(fields[3])
    carrier = _decompress_brotli(fields[4])
    if len(hpac) != IHS2_BODY_BYTES or len(semantic) != WANS_BODY_BYTES:
        raise ResidualArchiveError("HP4 restored fixed section length differs")
    if len(carrier) == PACKED_CAP1_SECTION_BYTES:
        carrier = _restore_packed_cap1_metadata(carrier)
    elif (
        len(carrier) > PACKED_CAP1_SECTION_BYTES
        and carrier[PACKED_CAP1_SECTION_BYTES:].startswith(COMPENSATION_MAGIC)
    ):
        carrier = (
            _restore_packed_cap1_metadata(carrier[:PACKED_CAP1_SECTION_BYTES])
            + carrier[PACKED_CAP1_SECTION_BYTES:]
        )
    elif len(carrier) != CANONICAL_CAP1_SECTION_BYTES:
        raise ResidualArchiveError("HP4 restored carrier length differs")
    return b"F24S" + hpac + semantic + carrier, outer[model_end:]
'''
    helper_anchor = "\n\ndef _decode_split_models(outer: bytes) -> tuple[bytes, bytes] | None:\n"
    read_old = "    split = _decode_split_models(outer)\n    if split is not None:\n        models, section = split\n"
    read_new = (
        "    split = _decode_hp4_models(outer)\n"
        "    if split is None:\n"
        "        split = _decode_split_models(outer)\n"
        "    if split is not None:\n"
        "        models, section = split\n"
    )
    if source.count(constants_old) != 1 or source.count(helper_anchor) != 1:
        raise MC35Error("HP4 runtime patch anchor differs")
    if source.count(read_old) != 1:
        raise MC35Error("HP4 read path patch anchor differs")
    source = source.replace(constants_old, constants_new)
    source = source.replace(helper_anchor, helper + helper_anchor)
    source = source.replace(read_old, read_new)
    qs2._atomic_replace(path, source.encode())
    return qs1.file_record(path)


def build_archive(
    output: Path,
    solved: list[dict[str, Any]],
    *,
    primary_override: dict[str, Any] | None = None,
    primary_repeat_override: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], Any]:
    result_path = output / "COMPILED_ARCHIVE.json"
    if result_path.is_file():
        prior = json.loads(result_path.read_text())
        require_record(prior["archive"])
        require_record(prior["archive_repeat"])
        return prior, None
    if primary_override is None:
        primary = qs1._compile_one(output=output, selected=solved, repeat=False)
        # The primary compile intentionally loads its copied receiver for parse-back.
        # Release that module tree before the repeat re-enters the pinned base runtime.
        for module_name in tuple(sys.modules):
            if module_name == "runtime" or module_name.startswith("runtime."):
                sys.modules.pop(module_name, None)
        repeated = qs1._compile_one(output=output, selected=solved, repeat=True)
        if primary["archive"]["sha256"] != repeated["archive"]["sha256"]:
            raise MC35Error("fresh-compensation closure repeat differs")
        primary_reencoded = True
    else:
        if primary_repeat_override is None:
            raise MC35Error("retained entropy source lacks its deterministic repeat")
        primary = json.loads(json.dumps(primary_override))
        require_record(primary["archive"])
        require_record(primary["token"])
        require_record(primary_repeat_override)
        if primary["archive"]["sha256"] != primary_repeat_override["sha256"]:
            raise MC35Error("retained entropy source repeat differs")
        repeated = {"archive": primary_repeat_override}
        primary_reencoded = False
    overlay = qs5._exact_overlay_candidate(output, primary, solved)
    if overlay is None:
        raise MC35Error("fresh compensation escaped Q2C1 exact overlay domain")
    split_checkpoint = output / "SPLIT_FINAL_CHECKPOINT.json"
    if split_checkpoint.is_file():
        split_final = json.loads(split_checkpoint.read_text())
        require_record(split_final["archive"])
        require_record(split_final["archive_repeat"])
    elif (output / "candidate/archive.zip").is_file():
        # Recover a completed qs5 split stage after an outer-stage interruption.
        # The qs5 helper patches its runtime in place and is not re-entrant.
        from experiments import ddm_cp135_rate_compose as cp135

        split_archive = qs1.file_record(output / "candidate/archive.zip")
        split_repeat = qs1.file_record(output / "candidate/archive.repeat.zip")
        if (split_archive["bytes"], split_archive["sha256"]) != (
            split_repeat["bytes"],
            split_repeat["sha256"],
        ):
            raise MC35Error("recovered split archive repeat differs")
        runtime_root = output / "candidate/adapted_runtime"
        split_final = {
            "schema": "ddm_mc35_recovered_qs5_split.v1",
            "archive": split_archive,
            "archive_repeat": split_repeat,
            "runtime_root": str(runtime_root.resolve()),
            "runtime_tree": cp135.tree_record(runtime_root),
            "candidate_codes": qs1.file_record(
                output / "candidate/candidate_codes.int32.npy"
            ),
            "overlay": overlay["overlay"],
            "receiver_code_lattice_exact": True,
            "recovery_reason": "outer HP4 receipt assertion interrupted after complete qs5 split build",
        }
        qs1.retain_json(split_checkpoint, split_final)
    else:
        split_final = qs5._compile_final_runtime(output, primary, solved, overlay)
        qs1.retain_json(split_checkpoint, split_final)
    split_archive = require_record(split_final["archive"])
    _model, member, hp4 = _hp4_model(split_archive, output)
    archive_payload = jo1.deterministic_zip(member)
    archive_repeat = jo1.deterministic_zip(member)
    archive = qs1.retain_bytes(
        output / "micro35_candidate/archive.zip", archive_payload
    )
    repeat = qs1.retain_bytes(
        output / "micro35_candidate/archive.repeat.zip", archive_repeat
    )
    if (archive["bytes"], archive["sha256"]) != (
        repeat["bytes"],
        repeat["sha256"],
    ):
        raise MC35Error("HP4 union archive repeat differs")
    runtime_root = output / "micro35_candidate/adapted_runtime"
    if runtime_root.exists():
        prior_archive = runtime_root / "archive.zip"
        if prior_archive.is_file() and prior_archive.read_bytes() != archive_payload:
            raise MC35Error("resumed MICRO35 runtime archive differs")
    else:
        shutil.copytree(Path(split_final["runtime_root"]), runtime_root)
    qs2._atomic_replace(runtime_root / "archive.zip", archive_payload)
    inflate = runtime_root / "inflate.py"
    text = inflate.read_text()
    old_archive = split_final["archive"]
    if str(archive["sha256"]) not in text:
        if text.count(str(old_archive["sha256"])) != 1:
            raise MC35Error("runtime inflate archive SHA patch surface differs")
        text = text.replace(str(old_archive["sha256"]), str(archive["sha256"]))
    old_size = f"ARCHIVE_BYTES = {int(old_archive['bytes']):_}"
    new_size = f"ARCHIVE_BYTES = {int(archive['bytes']):_}"
    if new_size not in text:
        if text.count(old_size) != 1:
            raise MC35Error("runtime inflate archive-size patch surface differs")
        text = text.replace(old_size, new_size)
    qs2._atomic_replace(inflate, text.encode(), executable=True)
    parser_patch = _patch_hp4_runtime(runtime_root)
    for module_name in tuple(sys.modules):
        if module_name == "runtime" or module_name.startswith("runtime."):
            sys.modules.pop(module_name, None)
    from experiments import ddm_cp135_rate_compose as cp135

    runtime = cp135.load_runtime(runtime_root)
    parsed = runtime.read_residual_archive(runtime_root / "archive.zip")
    expected_token = Path(primary["token"]["path"]).read_bytes()
    if parsed.token_stream != expected_token:
        raise MC35Error("HP4 runtime token parse-back differs")
    expected_pairs, expected_deltas = qs2.exact_deltas(solved)
    actual_pairs, actual_deltas = overlay_codec.decode_compensation_overlay(
        parsed.compensation_blob
    )
    if not np.array_equal(actual_pairs, expected_pairs) or not np.array_equal(
        actual_deltas, expected_deltas
    ):
        raise MC35Error("HP4 runtime compensation parse-back differs")
    result = {
        "schema": "ddm_mc35_compiled_archive.v1",
        "axis": "[macOS-CPU scorer-free real coder and runtime parse-back]",
        "fresh_compensation_hp3_rc64": primary,
        "fresh_compensation_repeat": repeated["archive"],
        "primary_reencoded_for_this_variant": primary_reencoded,
        "retained_entropy_source_only": not primary_reencoded,
        "q2c1_split_candidate": split_final,
        "hp4": hp4,
        "archive": archive,
        "archive_repeat": repeat,
        "archive_repeat_byte_identical": True,
        "delta_bytes_vs_cp135": int(archive["bytes"]) - CP135_BYTES,
        "runtime_root": str(runtime_root.resolve()),
        "runtime_parser_patch": parser_patch,
        "runtime_tree": cp135.tree_record(runtime_root),
        "runtime_parseback": {
            "hp4_container": True,
            "token_stream_exact": True,
            "fresh_compensation_overlay_exact": True,
            "compensation_pairs": actual_pairs.astype(int).tolist(),
        },
        "all_payloads_retained": True,
        "score_claim": False,
        "promotion_eligible": False,
    }
    qs1.retain_json(output / "COMPILED_ARCHIVE.json", result)
    checkpoint(
        output,
        "stage_40_compiled_archive",
        {"result": qs1.file_record(output / "COMPILED_ARCHIVE.json")},
    )
    return result, parsed


def local_recount(
    output: Path,
    supports: dict[str, Any],
    rendered: list[dict[str, Any]],
    solved: list[dict[str, Any]],
    compiled: dict[str, Any],
    *,
    axis: str = AXIS,
) -> dict[str, Any]:
    result_path = output / "LOCAL_ADVISORY_RECOUNT.json"
    if result_path.is_file():
        return json.loads(result_path.read_text())
    import torch

    from experiments import ddm_js1b_cuda_argmax_field_materializer_worker as worker

    pairs = [int(row["pair"]) for row in supports["rows"]]
    rendered_by_id = {row["proposal_id"]: row for row in rendered}
    candidate_input = np.stack(
        [
            np.load(
                rendered_by_id[row["proposal_id"]]["scorer_input"]["path"],
                allow_pickle=False,
            )
            for row in supports["rows"]
        ]
    )
    raw = np.memmap(
        qs1.CP135_RAW,
        mode="r",
        dtype=np.uint8,
        shape=(1_200, qs1.CAMERA_H, qs1.CAMERA_W, 3),
    )
    base_input = np.stack(
        [js6.base_scorer_input(np.asarray(raw[2 * pair + 1])) for pair in pairs]
    )
    scorer_root = output / "retained/local_scorer"
    candidate_input_record = qs1.retain_npy(
        scorer_root / "candidate_seg_input.float32.npy", candidate_input
    )
    base_input_record = qs1.retain_npy(
        scorer_root / "base_seg_input.float32.npy", base_input
    )
    segnet = worker.load_segnet(torch.device("cpu"))
    with torch.inference_mode():
        candidate_logits = segnet(torch.from_numpy(candidate_input))
        base_logits = segnet(torch.from_numpy(base_input))
    candidate_logits_np = candidate_logits.cpu().numpy()
    base_logits_np = base_logits.cpu().numpy()
    candidate_argmax = candidate_logits.argmax(dim=1).to(torch.uint8).cpu().numpy()
    base_argmax = base_logits.argmax(dim=1).to(torch.uint8).cpu().numpy()
    logits_records = {
        "candidate": qs1.retain_npy(
            scorer_root / "candidate_seg_logits.float32.npy", candidate_logits_np
        ),
        "base": qs1.retain_npy(
            scorer_root / "base_seg_logits.float32.npy", base_logits_np
        ),
        "candidate_argmax": qs1.retain_npy(
            scorer_root / "candidate_argmax.uint8.npy", candidate_argmax
        ),
        "base_argmax": qs1.retain_npy(
            scorer_root / "base_argmax.uint8.npy", base_argmax
        ),
    }
    base_field = np.load(qs3.QS1_BASE_FIELD, allow_pickle=False)
    gt_field = np.load(qs3.QS1_GT_FIELD, allow_pickle=False)
    composed = np.asarray(base_field).copy()
    for ordinal, pair in enumerate(pairs):
        composed[pair] = candidate_argmax[ordinal]
    composed_record = qs1.retain_npy(
        scorer_root / "composed_argmax_n600.uint8.npy", composed
    )
    base_flips = int(np.count_nonzero(base_field != gt_field))
    candidate_flips = int(np.count_nonzero(composed != gt_field))
    if base_flips != CP135_FLIPS:
        raise MC35Error(f"retained base flip count differs: {base_flips}")
    pair_rows = []
    groups = {"QS2": 0, "RE1": 0}
    cpu_t4_base_mismatch = 0
    for ordinal, support in enumerate(supports["rows"]):
        pair = int(support["pair"])
        before = int(np.count_nonzero(base_field[pair] != gt_field[pair]))
        after = int(np.count_nonzero(candidate_argmax[ordinal] != gt_field[pair]))
        gain = before - after
        groups[str(support["bank"])] += gain
        mismatch = int(np.count_nonzero(base_argmax[ordinal] != base_field[pair]))
        cpu_t4_base_mismatch += mismatch
        pair_rows.append(
            {
                "proposal_id": support["proposal_id"],
                "bank": support["bank"],
                "pair": pair,
                "base_flips": before,
                "candidate_flips": after,
                "net_flip_gain": gain,
                "cpu_base_vs_retained_t4_base_mismatch": mismatch,
            }
        )
    solved_by_pair = {int(row["pair"]): row for row in solved}
    base_pose = np.load(qs1.CP135_BASE_POSE, allow_pickle=False)
    gt_pose = np.load(qs1.GT_POSE, allow_pickle=False)
    candidate_pose = np.asarray(base_pose).copy()
    for pair, row in solved_by_pair.items():
        pose_record = row.get("variant_final_pose_vector")
        pose_path = (
            Path(pose_record["path"])
            if isinstance(pose_record, dict)
            else output
            / "retained/compensation"
            / row["proposal_id"]
            / "FINAL_POSE_VECTOR.float32.npy"
        )
        candidate_pose[pair] = np.load(
            pose_path,
            allow_pickle=False,
        )
    candidate_pose_record = qs1.retain_npy(
        scorer_root / "candidate_pose_first6_n600.float32.npy", candidate_pose
    )
    base_dpose = float(
        np.mean(np.square(base_pose.astype(np.float64) - gt_pose.astype(np.float64)))
    )
    candidate_dpose = float(
        np.mean(
            np.square(candidate_pose.astype(np.float64) - gt_pose.astype(np.float64))
        )
    )
    delta_dpose = candidate_dpose - base_dpose
    net_gain = base_flips - candidate_flips
    delta_bytes = int(compiled["delta_bytes_vs_cp135"])
    gates = {
        "net_flips_gte_35": net_gain >= FLIP_GATE,
        "delta_bytes_lte_29": delta_bytes <= BYTE_GATE,
        "delta_dpose_lte_cap": delta_dpose <= POSE_GATE,
        "receiver_parseback": compiled["runtime_parseback"][
            "fresh_compensation_overlay_exact"
        ],
    }
    result = {
        "schema": "ddm_mc35_local_advisory_recount.v1",
        "axis": axis,
        "selection_mode": (
            f"all {len(pairs)} changed pairs after one composed receiver closure; "
            "no prefix"
        ),
        "denominators": {"pairs": 600, "seg_pixels": PIXELS, "pose_scalars": 3_600},
        "retention": {
            "candidate_input": candidate_input_record,
            "base_input": base_input_record,
            "scorer_outputs": logits_records,
            "composed_argmax": composed_record,
            "candidate_pose": candidate_pose_record,
        },
        "seg": {
            "base_flips": base_flips,
            "candidate_flips": candidate_flips,
            "net_flip_gain": net_gain,
            "per_bank_net_flip_gain": groups,
            "support_overlap": {
                "pair_overlap_count": 0,
                "pixel_overlap_count": 0,
                "proof": "QS2 and RE1 occupy disjoint pair indices in the built union",
            },
            "pair_rows": pair_rows,
            "cpu_base_vs_retained_t4_base_mismatch_pixels": cpu_t4_base_mismatch,
        },
        "pose": {
            "base_dpose_recomputed": base_dpose,
            "candidate_dpose": candidate_dpose,
            "delta_dpose": delta_dpose,
            "cap": POSE_GATE,
        },
        "rate": {
            "base_archive_bytes": CP135_BYTES,
            "candidate_archive_bytes": int(compiled["archive"]["bytes"]),
            "delta_bytes": delta_bytes,
            "cap": BYTE_GATE,
        },
        "projected_delta_s": (
            -100.0 * net_gain / PIXELS
            + math.sqrt(10.0 * candidate_dpose)
            - math.sqrt(10.0 * base_dpose)
            + 25.0 * delta_bytes / RATE_DENOMINATOR
        ),
        "gates": gates,
        "all_gates_passed": all(gates.values()),
        "admission_authority": "local advisory gate only; no exact score or promotion claim",
        "all_scorer_payloads_retained": True,
        "score_claim": False,
        "promotion_eligible": False,
    }
    qs1.retain_json(result_path, result)
    checkpoint(output, "stage_50_local_recount", {"result": qs1.file_record(result_path)})
    return result


def deterministic_runtime_zip(runtime_root: Path) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", allowZip64=False) as archive:
        for path in sorted(runtime_root.rglob("*")):
            if not path.is_file() or "__pycache__" in path.parts or path.suffix == ".pyc":
                continue
            relative = path.relative_to(runtime_root).as_posix()
            info = zipfile.ZipInfo(relative, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = (0o100755 if os.access(path, os.X_OK) else 0o100644) << 16
            archive.writestr(info, path.read_bytes())
    return output.getvalue()


def finalize(output: Path, compiled: dict[str, Any], recount: dict[str, Any]) -> dict[str, Any]:
    if recount["all_gates_passed"]:
        runtime_root = Path(compiled["runtime_root"])
        runtime_payload = deterministic_runtime_zip(runtime_root)
        runtime_record = qs1.retain_bytes(
            output / "fire_order/fire_inputs/candidate_runtime.zip", runtime_payload
        )
        archive_payload = Path(compiled["archive"]["path"]).read_bytes()
        archive_record = qs1.retain_bytes(
            output / "fire_order/fire_inputs/candidate_archive.zip", archive_payload
        )
        request = {
            "schema": "ddm_mc35_dual_axis_fire_order.v1",
            "disposition": "QUEUED-WITH-A-FIRE-ORDER",
            "owner": "MAIN dispatcher; self-claim before launch",
            "consumer_store": "/Volumes/VertigoDataTier/pact/ddm_mc35_20260814/dispatch/ddm_mc35_dual_axis_t4_r1",
            "fire_trigger": "MAIN verifies no active full-n600 scorer lane, claims ddm_mc35_dual_axis_t4_r1, and the sealed archive/runtime SHAs still match",
            "worker": "proven RE1T/JS1B dual-axis T4 chain",
            "estimated_cost_usd": 0.16,
            "max_scorer_chunk": 120,
            "candidate_archive": archive_record,
            "candidate_runtime": runtime_record,
            "required_returns": [
                "retained decoded 0.raw",
                "candidate/GT/base argmax fields",
                "candidate/base/GT first-six PoseNet vectors",
                "exact upstream evaluator receipt on the same archive bytes",
            ],
            "arm_preclaimed_lane": False,
            "remote_dispatched": False,
        }
        request_record = qs1.retain_json(output / "fire_order/SEALED_REQUEST.json", request)
        fire_order: dict[str, Any] | None = {
            **request,
            "sealed_request": request_record,
            "request_sha256": request_record["sha256"],
        }
        qs1.retain_json(output / "SEALED_FIRE_ORDER.json", fire_order)
        status = "PASS_QUEUED_WITH_FIRE_ORDER"
    else:
        fire_order = None
        status = "TERMINAL_GATE_FAILURE_NO_FIRE_ORDER"
        qs1.retain_json(
            output / "GATE_FAILURE.json",
            {
                "schema": "ddm_mc35_gate_failure.v1",
                "status": status,
                "failed_gates": [
                    name for name, passed in recount["gates"].items() if not passed
                ],
                "no_fire_order": True,
                "no_redesign_creep": True,
            },
        )
    final = {
        "schema": "ddm_mc35_final_result.v1",
        "status": status,
        "axis": AXIS,
        "archive": compiled["archive"],
        "archive_repeat": compiled["archive_repeat"],
        "runtime_root": compiled["runtime_root"],
        "local_recount": qs1.file_record(output / "LOCAL_ADVISORY_RECOUNT.json"),
        "gates": recount["gates"],
        "all_gates_passed": recount["all_gates_passed"],
        "fire_order": fire_order,
        "remote_dispatched": False,
        "frontier_pointer_moved": False,
        "score_claim": False,
        "promotion_eligible": False,
        "all_materialized_payloads_retained": True,
    }
    qs1.retain_json(output / "FINAL_RESULT.json", final)
    checkpoint(output, "stage_90_final", {"result": qs1.file_record(output / "FINAL_RESULT.json")})
    return final


def _variant_axis(variant: str, pair_count: int) -> str:
    return (
        "[macOS-CPU advisory frozen CPU-torch SegNet/PoseNet; "
        f"{pair_count} changed pairs over n600; {variant}] NON-PROMOTABLE"
    )


def finalize_variant(
    *,
    output: Path,
    variant: str,
    compiled: dict[str, Any],
    recount: dict[str, Any],
) -> dict[str, Any]:
    status = (
        "PASS_ELIGIBLE_FOR_MC36_ADJUDICATION"
        if recount["all_gates_passed"]
        else "TERMINAL_LOCAL_GATE_FAILURE"
    )
    if not recount["all_gates_passed"]:
        qs1.retain_json(
            output / "GATE_FAILURE.json",
            {
                "schema": "ddm_mc36_variant_gate_failure.v1",
                "variant": variant,
                "status": status,
                "failed_gates": [
                    name for name, passed in recount["gates"].items() if not passed
                ],
                "no_fire_order": True,
                "no_gate_relaxation": True,
            },
        )
    final = {
        "schema": "ddm_mc36_variant_final_result.v1",
        "variant": variant,
        "status": status,
        "axis": recount["axis"],
        "archive": compiled["archive"],
        "archive_repeat": compiled["archive_repeat"],
        "runtime_root": compiled["runtime_root"],
        "local_recount": qs1.file_record(output / "LOCAL_ADVISORY_RECOUNT.json"),
        "gates": recount["gates"],
        "all_gates_passed": recount["all_gates_passed"],
        "fire_order": None,
        "remote_dispatched": False,
        "frontier_pointer_moved": False,
        "score_claim": False,
        "promotion_eligible": False,
        "all_materialized_payloads_retained": True,
    }
    qs1.retain_json(output / "FINAL_RESULT.json", final)
    checkpoint(
        output,
        "stage_90_final",
        {"result": qs1.file_record(output / "FINAL_RESULT.json")},
    )
    return final


def _retained_primary_source(source_output: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    compiled_path = source_output / "COMPILED_ARCHIVE.json"
    compiled = json.loads(compiled_path.read_text())
    primary = compiled["fresh_compensation_hp3_rc64"]
    repeated = compiled["fresh_compensation_repeat"]
    require_record(primary["archive"])
    require_record(primary["token"])
    require_record(repeated)
    if primary["archive"]["sha256"] != repeated["sha256"]:
        raise MC35Error(f"retained primary repeat differs: {source_output}")
    return primary, repeated


def _retain_compensation_result(
    *,
    output: Path,
    variant: str,
    rows: Sequence[dict[str, Any]],
    reuse_proofs: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    value = {
        "schema": "ddm_mc36_variant_fresh_compensation.v1",
        "variant": variant,
        "rows": list(rows),
        "reused_exact_object_proofs": list(reuse_proofs),
        "stale_compensation_carried": False,
        "all_final_object_fingerprints_asserted": True,
        "all_payloads_retained": True,
    }
    legacy_path = output / "FRESH_COMPENSATION.json"
    if legacy_path.is_file():
        legacy = json.loads(legacy_path.read_text())
        legacy_rows = [
            {key: item for key, item in row.items() if key != "variant_final_pose_vector"}
            for row in legacy["rows"]
        ]
        variant_rows = [
            {key: item for key, item in row.items() if key != "variant_final_pose_vector"}
            for row in rows
        ]
        if legacy_rows != variant_rows:
            raise MC35Error("legacy fresh-compensation rows differ from variant receipt")
    else:
        qs1.retain_json(legacy_path, value)
    path = output / "VARIANT_COMPENSATION.json"
    qs1.retain_json(path, value)
    checkpoint(
        output,
        "stage_35_variant_compensation_receipt",
        {"result": qs1.file_record(path)},
    )
    return value


def build_successor_pair105(*, bulk_root: Path) -> dict[str, Any]:
    output = PAIR105_OUTPUT
    route_variant_bulk(
        output=output,
        bulk_root=bulk_root,
        variant="successor_pair105",
        expected_bulk_bytes=4 * 1024**3,
    )
    preflight(output)
    supports = build_variant_supports(
        output=output, variant="successor_pair105", drop_pair=None
    )
    source_materialized_path = OUTPUT / "MATERIALIZED_UNION.json"
    source_materialized = json.loads(source_materialized_path.read_text())
    require_record(source_materialized["entropy_primary"]["token"])
    require_record(source_materialized["entropy_repeat"]["token"])
    if (
        source_materialized["entropy_primary"]["token"]["sha256"]
        != source_materialized["entropy_repeat"]["token"]["sha256"]
    ):
        raise MC35Error("mc35 retained token entropy repeat differs")
    qs1.retain_json(
        output / "MATERIALIZED_UNION.json",
        {
            "schema": "ddm_mc36_retained_union_reuse.v1",
            "variant": "successor_pair105",
            "reason": "the eight frame-1 token objects are byte-identical to mc35",
            "source": qs1.file_record(source_materialized_path),
            "token_primary": source_materialized["entropy_primary"]["token"],
            "token_repeat": source_materialized["entropy_repeat"]["token"],
            "repeat_byte_identical": True,
            "all_payloads_retained": True,
        },
    )
    rendered, _semantic = qs5._render_exact_masters(output, {"rows": supports["rows"]})
    rendered_by_id = {row["proposal_id"]: row for row in rendered}
    source_rows = load_mc35_solved_rows()
    fixed_rows = [row for row in source_rows if int(row["pair"]) != PAIR105]
    reuse_proofs = []
    support_by_pair = {int(row["pair"]): row for row in supports["rows"]}
    for row in fixed_rows:
        support = support_by_pair[int(row["pair"])]
        reuse_proofs.append(
            verify_reused_exact_object(
                support=support,
                rendered=rendered_by_id[support["proposal_id"]],
                solved=row,
            )
        )
    pair105_support = support_by_pair[PAIR105]
    constrained = select_pose_constrained_pair105(
        output=output,
        support=pair105_support,
        rendered=rendered_by_id[pair105_support["proposal_id"]],
        fixed_rows=fixed_rows,
    )
    solved = sorted([*fixed_rows, constrained], key=lambda row: int(row["pair"]))
    _retain_compensation_result(
        output=output,
        variant="successor_pair105",
        rows=solved,
        reuse_proofs=reuse_proofs,
    )
    primary, repeated = _retained_primary_source(OUTPUT)
    compiled, _parsed = build_archive(
        output,
        solved,
        primary_override=primary,
        primary_repeat_override=repeated,
    )
    recount = local_recount(
        output,
        supports,
        rendered,
        solved,
        compiled,
        axis=_variant_axis("successor_pair105", len(solved)),
    )
    return finalize_variant(
        output=output,
        variant="successor_pair105",
        compiled=compiled,
        recount=recount,
    )


def build_successor_drop532(*, bulk_root: Path) -> dict[str, Any]:
    output = DROP532_OUTPUT
    route_variant_bulk(
        output=output,
        bulk_root=bulk_root,
        variant="successor_drop532",
        expected_bulk_bytes=32 * 1024**3,
    )
    preflight(output)
    supports = build_variant_supports(
        output=output, variant="successor_drop532", drop_pair=DROP_PAIR
    )
    migrate_compile_workspace_to_hardlink_device(
        output=output,
        variant="successor_drop532",
        source_payload=jo1.BASE_PROBABILITIES / "codes_0000.npy",
    )
    materialize_union(output, supports)
    rendered, solved = render_and_solve(output, supports)
    for row in solved:
        row["variant_final_pose_vector"] = qs1.file_record(
            output
            / "retained/compensation"
            / str(row["proposal_id"])
            / "FINAL_POSE_VECTOR.float32.npy"
        )
    _retain_compensation_result(
        output=output,
        variant="successor_drop532",
        rows=solved,
        reuse_proofs=[],
    )
    compiled, _parsed = build_archive(output, solved)
    recount = local_recount(
        output,
        supports,
        rendered,
        solved,
        compiled,
        axis=_variant_axis("successor_drop532", len(solved)),
    )
    return finalize_variant(
        output=output,
        variant="successor_drop532",
        compiled=compiled,
        recount=recount,
    )


def build_successor_composed(*, bulk_root: Path) -> dict[str, Any]:
    output = COMPOSED_OUTPUT
    route_variant_bulk(
        output=output,
        bulk_root=bulk_root,
        variant="successor_drop532_pair105",
        expected_bulk_bytes=2 * 1024**3,
    )
    preflight(output)
    supports = build_variant_supports(
        output=output, variant="successor_drop532_pair105", drop_pair=DROP_PAIR
    )
    drop_compensation = json.loads(
        (DROP532_OUTPUT / "VARIANT_COMPENSATION.json").read_text()
    )
    pair_compensation = json.loads(
        (PAIR105_OUTPUT / "VARIANT_COMPENSATION.json").read_text()
    )
    pair105 = next(
        row for row in pair_compensation["rows"] if int(row["pair"]) == PAIR105
    )
    rows = [
        row for row in drop_compensation["rows"] if int(row["pair"]) != PAIR105
    ] + [pair105]
    rows.sort(key=lambda row: int(row["pair"]))
    rendered = []
    reuse_proofs = []
    support_by_pair = {int(row["pair"]): row for row in supports["rows"]}
    for support in supports["rows"]:
        source_root = (
            PAIR105_OUTPUT if int(support["pair"]) == PAIR105 else DROP532_OUTPUT
        )
        rendered_row = json.loads(
            (
                source_root
                / "retained/supports"
                / str(support["proposal_id"])
                / "exact_master/RESULT.json"
            ).read_text()
        )
        rendered.append(rendered_row)
    rendered_by_id = {row["proposal_id"]: row for row in rendered}
    for row in rows:
        support = support_by_pair[int(row["pair"])]
        reuse_proofs.append(
            verify_reused_exact_object(
                support=support,
                rendered=rendered_by_id[support["proposal_id"]],
                solved=row,
            )
        )
    _retain_compensation_result(
        output=output,
        variant="successor_drop532_pair105",
        rows=rows,
        reuse_proofs=reuse_proofs,
    )
    primary, repeated = _retained_primary_source(DROP532_OUTPUT)
    compiled, _parsed = build_archive(
        output,
        rows,
        primary_override=primary,
        primary_repeat_override=repeated,
    )
    recount = local_recount(
        output,
        supports,
        rendered,
        rows,
        compiled,
        axis=_variant_axis("successor_drop532_pair105", len(rows)),
    )
    return finalize_variant(
        output=output,
        variant="successor_drop532_pair105",
        compiled=compiled,
        recount=recount,
    )


def seal_mc36_fire_order(
    *, coordinator: Path, winner_output: Path, winner: dict[str, Any]
) -> dict[str, Any]:
    compiled = json.loads((winner_output / "COMPILED_ARCHIVE.json").read_text())
    runtime_root = Path(compiled["runtime_root"])
    runtime_record = qs1.retain_bytes(
        winner_output / "fire_order/fire_inputs/candidate_runtime.zip",
        deterministic_runtime_zip(runtime_root),
    )
    archive_record = qs1.retain_bytes(
        winner_output / "fire_order/fire_inputs/candidate_archive.zip",
        Path(compiled["archive"]["path"]).read_bytes(),
    )
    request = {
        "schema": "ddm_mc36_dual_axis_fire_order.v1",
        "disposition": "QUEUED-WITH-A-FIRE-ORDER",
        "variant": winner["variant"],
        "owner": "MAIN sole scorer-lane router; self-claim before launch",
        "consumer_store": str(
            winner_output / "dispatch/ddm_mc36_dual_axis_t4_r1"
        ),
        "fire_trigger": (
            "MAIN verifies no active full-n600 scorer lane, claims "
            "ddm_mc36_dual_axis_t4_r1, and the sealed archive/runtime SHAs match"
        ),
        "worker": "proven RE1T/JS1B dual-axis T4 chain",
        "estimated_cost_usd": 0.16,
        "max_scorer_chunk": 120,
        "candidate_archive": archive_record,
        "candidate_runtime": runtime_record,
        "required_returns": [
            "retained decoded 0.raw",
            "candidate/GT/base argmax fields",
            "candidate/base/GT first-six PoseNet vectors",
            "exact upstream evaluator receipt on the same archive bytes",
        ],
        "arm_preclaimed_lane": False,
        "remote_dispatched": False,
    }
    request_record = qs1.retain_json(
        winner_output / "fire_order/SEALED_REQUEST.json", request
    )
    sealed = {
        **request,
        "sealed_request": request_record,
        "request_sha256": request_record["sha256"],
    }
    qs1.retain_json(winner_output / "SEALED_FIRE_ORDER.json", sealed)
    winner = dict(winner)
    winner["status"] = "PASS_QUEUED_WITH_FIRE_ORDER"
    winner["fire_order"] = sealed
    qs1.retain_json(winner_output / "FINAL_RESULT_SEALED.json", winner)
    qs1.retain_json(coordinator / "SEALED_FIRE_ORDER.json", sealed)
    return sealed


def run_mc36_successors(*, coordinator: Path, bulk_root: Path) -> dict[str, Any]:
    final_path = coordinator / "FINAL_RESULT_V2.json"
    if final_path.is_file():
        prior = json.loads(final_path.read_text())
        if prior.get("sealed_fire_order") is not None:
            require_record(prior["sealed_fire_order"]["sealed_request"])
        return prior
    superseded_path = coordinator / "FINAL_RESULT.json"
    superseded = (
        qs1.file_record(superseded_path) if superseded_path.is_file() else None
    )
    route_variant_bulk(
        output=coordinator,
        bulk_root=bulk_root / "coordinator",
        variant="mc36_coordinator",
        expected_bulk_bytes=1024**3,
    )
    pair105 = build_successor_pair105(bulk_root=bulk_root / "successor_pair105")
    drop532 = build_successor_drop532(bulk_root=bulk_root / "successor_drop532")
    candidates: list[tuple[Path, dict[str, Any]]] = [
        (PAIR105_OUTPUT, pair105),
        (DROP532_OUTPUT, drop532),
    ]
    pair_only, drop_only = _complementary_gate_coverage(
        pair105["gates"], drop532["gates"]
    )
    pair_recount = json.loads((PAIR105_OUTPUT / "LOCAL_ADVISORY_RECOUNT.json").read_text())
    drop_recount = json.loads((DROP532_OUTPUT / "LOCAL_ADVISORY_RECOUNT.json").read_text())
    composition_evidence = _composition_evidence(
        pair_result=pair105,
        drop_result=drop532,
        pair_recount=pair_recount,
        drop_recount=drop_recount,
    )
    compose_justified = bool(composition_evidence["justified"])
    composed = None
    if compose_justified:
        composed = build_successor_composed(
            bulk_root=bulk_root / "successor_drop532_pair105"
        )
        candidates.append((COMPOSED_OUTPUT, composed))
    passing = [item for item in candidates if item[1]["all_gates_passed"]]
    if passing:
        winner_output, winner = min(
            passing,
            key=lambda item: (
                float(
                    json.loads(
                        (item[0] / "LOCAL_ADVISORY_RECOUNT.json").read_text()
                    )["projected_delta_s"]
                ),
                int(item[1]["archive"]["bytes"]),
                item[1]["variant"],
            ),
        )
        sealed = seal_mc36_fire_order(
            coordinator=coordinator, winner_output=winner_output, winner=winner
        )
        status = "PASS_ONE_FIRE_ORDER_SEALED"
    else:
        winner_output = None
        winner = None
        sealed = None
        status = "TERMINAL_NO_VARIANT_PASSED_ALL_GATES"
    result = {
        "schema": "ddm_mc36_successor_adjudication.v1",
        "status": status,
        "variant_results": {
            "successor_pair105": qs1.file_record(PAIR105_OUTPUT / "FINAL_RESULT.json"),
            "successor_drop532": qs1.file_record(DROP532_OUTPUT / "FINAL_RESULT.json"),
            "successor_drop532_pair105": (
                None
                if composed is None
                else qs1.file_record(COMPOSED_OUTPUT / "FINAL_RESULT.json")
            ),
        },
        "compose_decision": {
            "justified": compose_justified,
            "pair105_only_passed_gates": sorted(pair_only),
            "drop532_only_passed_gates": sorted(drop_only),
            **composition_evidence,
            "measured_pair105_archive_bytes": int(
                pair_recount["rate"]["candidate_archive_bytes"]
            ),
            "measured_drop532_archive_bytes": int(
                drop_recount["rate"]["candidate_archive_bytes"]
            ),
            "measured_pair105_net_flip_gain": int(
                pair_recount["seg"]["net_flip_gain"]
            ),
            "measured_drop532_net_flip_gain": int(
                drop_recount["seg"]["net_flip_gain"]
            ),
            "rule": (
                "compose only when A and B are partial and measured A/B results "
                "supply complementary pose, Seg, and rate value"
            ),
        },
        "winner_variant": None if winner is None else winner["variant"],
        "winner_store": None if winner_output is None else str(winner_output),
        "winner_sealed_result": (
            None
            if winner_output is None
            else qs1.file_record(winner_output / "FINAL_RESULT_SEALED.json")
        ),
        "sealed_fire_order": sealed,
        "remote_dispatched": False,
        "frontier_pointer_moved": False,
        "score_claim": False,
        "all_materialized_payloads_retained": True,
        "superseded_prior_adjudication": superseded,
    }
    qs1.retain_json(final_path, result)
    return result


def run(output: Path) -> dict[str, Any]:
    preflight(output)
    supports = build_supports(output)
    materialize_union(output, supports)
    rendered, solved = render_and_solve(output, supports)
    compiled, _parsed = build_archive(output, solved)
    recount = local_recount(output, supports, rendered, solved, compiled)
    return finalize(output, compiled, recount)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument(
        "--variant", choices=("micro35", "successors"), default="micro35"
    )
    value.add_argument("--output", type=Path)
    value.add_argument("--bulk-root", type=Path, default=MC36_BULK)
    value.add_argument("--resume-from", type=Path, required=True)
    return value


def main() -> None:
    args = parser().parse_args()
    default_output = OUTPUT if args.variant == "micro35" else MC36_OUTPUT
    output = (args.output or default_output).resolve()
    if args.resume_from.resolve() != output:
        raise SystemExit("FATAL: --resume-from must equal --output")
    with arm_lock(output):
        if args.variant == "successors":
            result = run_mc36_successors(
                coordinator=output, bulk_root=args.bulk_root.resolve()
            )
        else:
            result = run(output)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
