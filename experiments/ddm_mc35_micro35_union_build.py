#!/usr/bin/env python3
"""Build and locally gate the RFO1 MICRO35 union as one retained archive.

This is a macOS-CPU advisory builder.  It composes the six exact QS2 token
objects with RE1's admitted pair-96 singleton and the sign-verified pair-7
singleton, solves Schur compensation against the final token frames, closes
HP3/RC64 once, applies the receiver-identical HP4 order-0 repack, and recounts
the eight changed pairs through the frozen CPU scorers.  It never dispatches a
remote job and never promotes the local results.
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
    output: Path, solved: list[dict[str, Any]]
) -> tuple[dict[str, Any], Any]:
    result_path = output / "COMPILED_ARCHIVE.json"
    if result_path.is_file():
        prior = json.loads(result_path.read_text())
        require_record(prior["archive"])
        require_record(prior["archive_repeat"])
        return prior, None
    primary = qs1._compile_one(output=output, selected=solved, repeat=False)
    # The primary compile intentionally loads its copied receiver for parse-back.
    # Release that module tree before the repeat re-enters the pinned base runtime.
    for module_name in tuple(sys.modules):
        if module_name == "runtime" or module_name.startswith("runtime."):
            sys.modules.pop(module_name, None)
    repeated = qs1._compile_one(output=output, selected=solved, repeat=True)
    if primary["archive"]["sha256"] != repeated["archive"]["sha256"]:
        raise MC35Error("fresh-compensation closure repeat differs")
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
        candidate_pose[pair] = np.load(
            output
            / "retained/compensation"
            / row["proposal_id"]
            / "FINAL_POSE_VECTOR.float32.npy",
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
        "axis": AXIS,
        "selection_mode": "all eight changed pairs after one composed receiver closure; no prefix",
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
    value.add_argument("--output", type=Path, default=OUTPUT)
    value.add_argument("--resume-from", type=Path, required=True)
    return value


def main() -> None:
    args = parser().parse_args()
    output = args.output.resolve()
    if args.resume_from.resolve() != output:
        raise SystemExit("FATAL: --resume-from must equal --output")
    with arm_lock(output):
        result = run(output)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
