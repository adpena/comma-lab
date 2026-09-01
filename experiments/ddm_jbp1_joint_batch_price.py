#!/usr/bin/env python3
"""Execute the bounded JBP1 exact-price roster without faking a G/M refit.

The retained RXC1 API is the physical byte instrument.  This runner adds the
candidate-ingestion boundary: it verifies every source pin, materializes and
retains each complete edited field plus a pair-plane overlay, proves the SFP1
null archive byte-identical, and prices only candidates whose declared
mechanism the instrument actually implements.

SFP1's three proposal documents currently declare
``refit_cross_group_causal_schedule`` with ``refit_required=true``.  RXC1/JG2
pins the shipped HPAC section and shipped group plan.  Therefore those rows are
blocked rather than silently priced as fixed-G/M field edits.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import struct
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments import ddm_rxc1_restartable_exact_coder as rxc1  # noqa: I001


STORE = Path("/Volumes/APDataStore/pact/ddm_jbp1_joint_batch_price")
INHERITED_STORE = Path("/Volumes/APDataStore/pact/ddm_jc1/restartable_exact_coder")
SFP1_STORE = Path("/Volumes/APDataStore/pact/ddm_jc1/scmdl_projection")
XOV1_STORE = Path("/Volumes/APDataStore/pact/ddm_xov1_crossover_pass")

FIELD_SHAPE = (600, 384, 512)
FIELD_BYTES = int(np.prod(FIELD_SHAPE))
RESERVE_BYTES = 1 << 30
PROJECTED_NEW_BYTES = 1_500_000_000
DEMAND_BYTES = 39_522.14
BASE_JOINT_POOL_BYTES = 126_926
ALLOWED_JOINT_POOL_BYTES = 87_403.86
AFFINE_ARCHIVE_CEILING_BYTES = 140_479.86
AXIS = "[macOS-CPU advisory / scorer-free EXACT byte measurement]"

BHW_HEADER = struct.Struct(">4sBBHQQQ32s32s32s")
BHW_RECORD = struct.Struct("<IBB")
BHW_RECORDS = XOV1_STORE / "retained/cross_bhw/cross_parent_bhw.records"
BHW_RECORDS_SHA256 = "f75f06fe18f5afe08779a30cd3f6aeeec84720aa8f1f8b923b5f1f9f300819aa"
GF1_FIELD_SHA256 = "4026c4e2c805beb5b79be2879bb4a84311655d0d7d80dbc766654847522a5d19"
GT_SEMANTIC_SHA256 = "a98b90678ca5d4e12b385d2c8596839b368af8d52277eea3c1d3666f7a4c9b3d"

SOURCE_PINS: tuple[tuple[Path, int | None, str], ...] = (
    (
        REPO / ".omx/research/charters/ddm_jbp1_joint_batch_price_20260901.md",
        None,
        "d0c9df554a08276a51f2f08f62dd9a2f6f10612097cb3f274b7b879caa432693",
    ),
    (
        REPO / ".omx/tmp/codex_runs/_common_contract.md",
        None,
        "eeae9e0035582e6bdd65fd837e4aa35a65e064fd09900b9c212d41ac02086771",
    ),
    (
        REPO / ".omx/research/ddm_rxc1_gen3_gate1_verdict_20260901.md",
        None,
        "ec6cb0f5c0260e6aa5d6cab536fe1bfa91732c0d61a7983aef05fcf4301daf52",
    ),
    (
        REPO / ".omx/research/ddm_dds1_ceiling_readjudication_20260901.md",
        None,
        "13f33173c38b8ccdbcca6529976740a3fdb117ca9bc8897997dd69f742a3dc38",
    ),
    (
        REPO / ".omx/research/ddm_sfp1_scmdl_field_proposal_prep_20260901.md",
        None,
        "af70ab65c258b8700851bdf525ab8dc1c58b41cf34b374403e9c8e67ad48538b",
    ),
    (
        REPO / ".omx/research/ddm_xov1_crossover_pass_20260901.md",
        None,
        "ad093da3358996cb30700a2d0976af2436b10ea570eeada276580336b6ce6345",
    ),
    (BHW_RECORDS, 7_965_356, BHW_RECORDS_SHA256),
    (
        INHERITED_STORE / "SCREEN.json",
        60_753,
        "e6a400be9bb140bd220b4d5e77473a36dd692e2794cbd47f48a65b30d9309420",
    ),
    (
        INHERITED_STORE / "MANIFEST.json",
        138_824,
        "954ccada3094df4f22147d47225cab84238782291287ea51c52d1362d0a4ae2c",
    ),
    (
        SFP1_STORE / "HANDOFF.json",
        5_886,
        "83b92462e8deec3686f6ed1b23f93302493f5ed9e8403da66641083f8f682e97",
    ),
    (
        SFP1_STORE / "CANDIDATE_SET.json",
        6_430,
        "00885e9a77d2779df9e68c3daf12869f95b91f28b704d7fd07a4cbd25e9a4787",
    ),
    (
        SFP1_STORE / "CONTROLS.json",
        1_745,
        "4cfa3eb73ae5570a6dc7a6ba55f5fe2de1ec0d8a7c5f7f49a755588b6c364eb5",
    ),
)


class Jbp1Error(RuntimeError):
    """A source, custody, identity, or mechanism gate refused."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_fact(path: Path) -> dict[str, object]:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def atomic_json(path: Path, payload: Mapping[str, Any] | Sequence[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def write_once_json(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path.is_file():
        if path.read_text() != encoded:
            raise Jbp1Error(f"immutable receipt changed on resume: {path}")
        return
    atomic_json(path, payload)


def require_file(path: Path, expected_bytes: int | None, expected_sha256: str) -> dict[str, object]:
    if not path.is_file():
        raise Jbp1Error(f"missing pinned input: {path}")
    fact = file_fact(path)
    if expected_bytes is not None and fact["bytes"] != expected_bytes:
        raise Jbp1Error(f"byte mismatch for {path}: {fact['bytes']} != {expected_bytes}")
    if fact["sha256"] != expected_sha256:
        raise Jbp1Error(f"SHA-256 mismatch for {path}: {fact['sha256']} != {expected_sha256}")
    return fact


def atomic_copy(source: Path, destination: Path, expected_sha256: str) -> dict[str, object]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        return require_file(destination, source.stat().st_size, expected_sha256)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.partial")
    with source.open("rb") as read_handle, temporary.open("xb") as write_handle:
        shutil.copyfileobj(read_handle, write_handle, 8 << 20)
        write_handle.flush()
        os.fsync(write_handle.fileno())
    if sha256_file(temporary) != expected_sha256:
        raise Jbp1Error(f"copied payload changed before admission: {source}")
    os.replace(temporary, destination)
    return file_fact(destination)


def parse_bhw_group0(payload: bytes) -> tuple[dict[str, object], list[tuple[int, int, int]]]:
    if len(payload) < BHW_HEADER.size:
        raise Jbp1Error("B/H/W record payload is shorter than its header")
    magic, version, flags, reserved, benefit, harm, wash, gf1_sha, afr1_sha, gt_sha = (
        BHW_HEADER.unpack_from(payload)
    )
    if (magic, version, flags, reserved) != (b"XBH1", 1, 0, 0):
        raise Jbp1Error("B/H/W record header identity changed")
    if gf1_sha.hex() != GF1_FIELD_SHA256 or afr1_sha.hex() != rxc1.TOKENS_SHA256:
        raise Jbp1Error("B/H/W record parent field pins changed")
    if gt_sha.hex() != GT_SEMANTIC_SHA256:
        raise Jbp1Error("B/H/W record GT semantic pin changed")
    expected = BHW_HEADER.size + (benefit + harm + wash) * BHW_RECORD.size
    if expected != len(payload):
        raise Jbp1Error(f"B/H/W record denominator changed: {len(payload)} != {expected}")
    records: list[tuple[int, int, int]] = []
    offset = BHW_HEADER.size
    for _ in range(benefit):
        records.append(BHW_RECORD.unpack_from(payload, offset))
        offset += BHW_RECORD.size
    return (
        {
            "benefit": benefit,
            "harm": harm,
            "wash": wash,
            "disagreement_denominator": benefit + harm + wash,
        },
        records,
    )


def materialize_bhw_field(destination: Path) -> dict[str, object]:
    payload = BHW_RECORDS.read_bytes()
    denominator, records = parse_bhw_group0(payload)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.is_file():
        temporary = destination.with_name(f".{destination.name}.{os.getpid()}.partial")
        with rxc1.TOKENS.open("rb") as read_handle, temporary.open("xb") as write_handle:
            shutil.copyfileobj(read_handle, write_handle, 8 << 20)
            write_handle.flush()
            os.fsync(write_handle.fileno())
        field = np.memmap(temporary, dtype=np.uint8, mode="r+", shape=(FIELD_BYTES,))
        for index, old, new in records:
            if int(field[index]) != old:
                raise Jbp1Error(f"B/H/W old token mismatch at flat index {index}")
            if new >= 5 or new == old:
                raise Jbp1Error(f"B/H/W invalid replacement at flat index {index}: {old}->{new}")
            field[index] = new
        field.flush()
        del field
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    field = np.memmap(destination, dtype=np.uint8, mode="r", shape=(FIELD_BYTES,))
    base = np.memmap(rxc1.TOKENS, dtype=np.uint8, mode="r", shape=(FIELD_BYTES,))
    changed = int(np.count_nonzero(field != base))
    del field, base
    if changed != len(records) or changed != 5_506:
        raise Jbp1Error(f"materialized B/H/W field has {changed} changes, expected 5,506")
    return {**file_fact(destination), **denominator, "changed_sites": changed}


def atomic_overlay(base_path: Path, field_path: Path, destination: Path) -> dict[str, object]:
    base = np.memmap(base_path, dtype=np.uint8, mode="r", shape=FIELD_SHAPE)
    field = np.memmap(field_path, dtype=np.uint8, mode="r", shape=FIELD_SHAPE)
    pairs = [pair for pair in range(FIELD_SHAPE[0]) if np.any(base[pair] != field[pair])]
    changed = sum(int(np.count_nonzero(base[pair] != field[pair])) for pair in pairs)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.is_file():
        temporary = destination.with_name(f".{destination.name}.{os.getpid()}.partial")
        planes = {str(pair): np.asarray(field[pair], dtype=np.uint8).copy() for pair in pairs}
        with temporary.open("xb") as handle:
            np.savez_compressed(handle, **planes)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    with np.load(destination, allow_pickle=False) as blob:
        observed_pairs = sorted(int(key) for key in blob.files)
        observed_changed = sum(
            int(np.count_nonzero(np.asarray(blob[str(pair)], dtype=np.uint8) != base[pair]))
            for pair in observed_pairs
        )
    del base, field
    if observed_pairs != pairs or observed_changed != changed:
        raise Jbp1Error(f"overlay parse-back changed for {destination}")
    return {
        **file_fact(destination),
        "edited_pairs": pairs,
        "edited_pair_count": len(pairs),
        "changed_sites": changed,
    }


def price_row(archive_delta_bytes: int) -> dict[str, object]:
    cut = -int(archive_delta_bytes)
    remaining = DEMAND_BYTES - cut
    return {
        "archive_delta_bytes": int(archive_delta_bytes),
        "bytes_cut": cut,
        "fraction_of_demand": cut / DEMAND_BYTES,
        "percent_of_demand": 100.0 * cut / DEMAND_BYTES,
        "candidate_joint_pool_bytes": BASE_JOINT_POOL_BYTES + int(archive_delta_bytes),
        "joint_pool_allowance_bytes": ALLOWED_JOINT_POOL_BYTES,
        "affine_archive_ceiling_bytes": AFFINE_ARCHIVE_CEILING_BYTES,
        "verdict": (
            f"BYTE-WIN by {cut - DEMAND_BYTES:.2f} B"
            if remaining <= 0
            else f"REFUSED +{remaining:.2f} B"
        ),
        "prior_law_falsified": cut >= 10_000,
    }


def require_storage_reserve() -> int:
    free = shutil.disk_usage(STORE.parent).free
    if free < RESERVE_BYTES:
        raise Jbp1Error(f"AP reserve gate failed: {free} < {RESERVE_BYTES}")
    return free


def sfp1_refit_audit(candidate_set: Mapping[str, Any], sources: Sequence[Path]) -> dict[str, object]:
    operation = "refit_cross_group_causal_schedule"
    candidates = list(candidate_set.get("candidates", []))
    source_text = "\n".join(path.read_text() for path in sources)
    rows = []
    executable_keys = {
        "implementation",
        "model_payload",
        "decoder_source",
        "schedule_source",
        "trained_model",
        "model_bytes",
    }
    for candidate in candidates:
        g_edit = candidate.get("g_edit") or {}
        rows.append(
            {
                "proposal_id": candidate.get("proposal_id"),
                "refit_required": candidate.get("refit_required") is True,
                "operation": g_edit.get("operation"),
                "declared_executable_keys": sorted(executable_keys.intersection(g_edit)),
            }
        )
    all_require = bool(rows) and all(
        row["refit_required"] and row["operation"] == operation for row in rows
    )
    operation_consumed = operation in source_text
    blocker = all_require and not operation_consumed and all(
        not row["declared_executable_keys"] for row in rows
    )
    return {
        "candidate_denominator": len(rows),
        "rows": rows,
        "all_require_cross_group_refit": all_require,
        "operation_consumed_by_rxc1_or_jg2": operation_consumed,
        "rxc1_pins_shipped_hpac_sha256": rxc1.AFR1_HPAC_SHA256,
        "rxc1_pack_replaces": "token tail only; HPAC section retained unchanged",
        "status": "MISSING_EXECUTABLE_GM_REFIT" if blocker else "EXECUTABLE_BINDING_PRESENT",
        "blocks_fixed_gm_standin": blocker,
    }


def stage_preflight() -> dict[str, object]:
    receipt = STORE / "PREFLIGHT.v2.json"
    if receipt.is_file():
        payload = json.loads(receipt.read_text())
        for fact in payload["source_pins"]:
            require_file(Path(fact["path"]), int(fact["bytes"]), str(fact["sha256"]))
        for key in ("runner_source", "rxc1_source", "jg2_source"):
            fact = payload["inherited_api"][key]
            require_file(Path(fact["path"]), int(fact["bytes"]), str(fact["sha256"]))
        require_storage_reserve()
        return payload
    facts = [require_file(path, size, digest) for path, size, digest in SOURCE_PINS]
    free = shutil.disk_usage(STORE.parent).free
    if free - PROJECTED_NEW_BYTES < RESERVE_BYTES:
        raise Jbp1Error(
            f"AP reserve gate failed: {free} - {PROJECTED_NEW_BYTES} < {RESERVE_BYTES}"
        )
    api = rxc1.RestartableExactCoder()
    if api.sections["hpac"] != rxc1.read_sections()[0]["hpac"]:
        raise Jbp1Error("inherited API HPAC section changed during preflight")
    payload = {
        "schema": "ddm_jbp1_preflight.v1",
        "axis": AXIS,
        "score_claim": False,
        "storage": {
            "root": str(STORE),
            "free_bytes_at_start": free,
            "projected_new_bytes": PROJECTED_NEW_BYTES,
            "mandatory_reserve_bytes": RESERVE_BYTES,
            "post_projection_free_bytes": free - PROJECTED_NEW_BYTES,
            "status": "PASS",
        },
        "source_pins": facts,
        "inherited_api": {
            "runner_source": file_fact(Path(__file__)),
            "rxc1_source": file_fact(Path(rxc1.__file__)),
            "jg2_source": file_fact(Path(rxc1.jg2.__file__)),
            "hpac_bytes": len(api.sections["hpac"]),
            "hpac_sha256": hashlib.sha256(api.sections["hpac"]).hexdigest(),
            "base_stream_bytes": len(api.shipped_stream),
            "base_stream_sha256": hashlib.sha256(api.shipped_stream).hexdigest(),
        },
        "authority_boundaries": {
            "scorer_runs": 0,
            "modal_calls": 0,
            "contest_evaluations": 0,
            "upstream_writes": 0,
        },
    }
    write_once_json(receipt, payload)
    return payload


def stage_prepare() -> dict[str, object]:
    stage_preflight()
    receipt = STORE / "PREPARED.json"
    if receipt.is_file():
        payload = json.loads(receipt.read_text())
        for section in ("fields", "overlays"):
            for fact in payload[section].values():
                require_file(Path(fact["path"]), int(fact["bytes"]), str(fact["sha256"]))
        return payload
    controls = json.loads((SFP1_STORE / "CONTROLS.json").read_text())
    fields: dict[str, dict[str, object]] = {}
    null_source = Path(controls["null"]["path"])
    fields["null"] = atomic_copy(
        null_source,
        STORE / "retained/fields/sfp1_null_empty.u8",
        str(controls["null"]["sha256"]),
    )
    overlays = {
        name: atomic_overlay(
            rxc1.TOKENS,
            Path(str(fact["path"])),
            STORE / f"retained/overlays/{name}.pair_planes.npz",
        )
        for name, fact in fields.items()
    }
    for name, overlay in overlays.items():
        expected = int(fields[name].get("changed_sites", 0))
        if int(overlay["changed_sites"]) != expected:
            raise Jbp1Error(
                f"changed-site denominator drifted for {name}: "
                f"{overlay['changed_sites']} != {expected}"
            )
    payload = {
        "schema": "ddm_jbp1_prepared_null.v1",
        "fields": fields,
        "overlays": overlays,
        "roster": ["null"],
    }
    write_once_json(receipt, payload)
    return payload


def stage_prepare_candidates() -> dict[str, object]:
    null = stage_null()
    if not null["byte_identical"]:
        raise Jbp1Error("candidate payloads cannot be prepared before the null identity gate passes")
    receipt = STORE / "CANDIDATES_PREPARED.json"
    if receipt.is_file():
        payload = json.loads(receipt.read_text())
        for section in ("fields", "overlays"):
            for fact in payload[section].values():
                require_file(Path(fact["path"]), int(fact["bytes"]), str(fact["sha256"]))
        return payload
    candidate_set = json.loads((SFP1_STORE / "CANDIDATE_SET.json").read_text())
    fields: dict[str, dict[str, object]] = {
        "xov1_bhw5506": materialize_bhw_field(STORE / "retained/fields/xov1_bhw5506.u8")
    }
    for candidate in candidate_set["candidates"]:
        materialized = candidate["materialized_field"]
        fields[str(candidate["proposal_id"])] = atomic_copy(
            Path(materialized["path"]),
            STORE / f"retained/fields/{candidate['proposal_id']}.u8",
            str(materialized["sha256"]),
        )
        fields[str(candidate["proposal_id"])]["changed_sites"] = int(
            materialized["changed_sites"]
        )
    overlays = {
        name: atomic_overlay(
            rxc1.TOKENS,
            Path(str(fact["path"])),
            STORE / f"retained/overlays/{name}.pair_planes.npz",
        )
        for name, fact in fields.items()
    }
    for name, overlay in overlays.items():
        expected = int(fields[name]["changed_sites"])
        if int(overlay["changed_sites"]) != expected:
            raise Jbp1Error(
                f"changed-site denominator drifted for {name}: "
                f"{overlay['changed_sites']} != {expected}"
            )
    payload = {
        "schema": "ddm_jbp1_prepared_candidates.v1",
        "fields": fields,
        "overlays": overlays,
        "sfp1_refit_audit": sfp1_refit_audit(
            candidate_set,
            (Path(rxc1.__file__), Path(rxc1.jg2.__file__)),
        ),
        "roster": ["xov1_bhw5506", *[row["proposal_id"] for row in candidate_set["candidates"]]],
        "candidate_denominator": 4,
    }
    write_once_json(receipt, payload)
    return payload


def _validated_run(receipt_path: Path) -> dict[str, Any]:
    payload = rxc1.validate_run_receipt(receipt_path)
    return payload


def stage_null() -> dict[str, object]:
    prepared = stage_prepare()
    receipt = STORE / "NULL_IDENTITY.json"
    if receipt.is_file():
        payload = json.loads(receipt.read_text())
        _validated_run(Path(payload["run_receipt"]["path"]))
        if not payload["archive_comparison"]["byte_identical"]:
            raise Jbp1Error("retained JBP1 null receipt records failed archive identity")
        return payload
    api = rxc1.RestartableExactCoder()
    require_storage_reserve()
    run = api.run(
        edit_path=Path(prepared["overlays"]["null"]["path"]),
        run_dir=STORE / "retained/exact/null",
        resume_frame=None,
    )
    base = json.loads((INHERITED_STORE / "retained/baseline/RESULT.json").read_text())
    stream_comparison = rxc1.compare_bytes(
        Path(run["stream"]["path"]), Path(base["stream"]["path"])
    )
    archive_comparison = rxc1.compare_bytes(
        Path(run["archive"]["path"]), Path(base["archive"]["path"])
    )
    payload = {
        "schema": "ddm_jbp1_null_identity.v1",
        "axis": AXIS,
        "score_claim": False,
        "source_field": prepared["fields"]["null"],
        "overlay": prepared["overlays"]["null"],
        "run_receipt": file_fact(STORE / "retained/exact/null/RESULT.json"),
        "stream_comparison": stream_comparison,
        "archive_comparison": archive_comparison,
        "byte_identical": bool(
            stream_comparison["byte_identical"] and archive_comparison["byte_identical"]
        ),
        "status": "PASS" if archive_comparison["byte_identical"] else "FAILED_BYTE_IDENTITY",
    }
    write_once_json(receipt, payload)
    if not payload["byte_identical"]:
        raise Jbp1Error("identity gate failed: SFP1 null archive differs from AFR1")
    return payload


def stage_candidate_a() -> dict[str, object]:
    null = stage_null()
    if not null["byte_identical"]:
        raise Jbp1Error("candidate A cannot run before the null identity gate passes")
    prepared_candidates = stage_prepare_candidates()
    receipt = STORE / "CANDIDATE_A.json"
    if receipt.is_file():
        payload = json.loads(receipt.read_text())
        _validated_run(Path(payload["run_receipt"]["path"]))
        return payload
    api = rxc1.RestartableExactCoder()
    require_storage_reserve()
    run = api.run(
        edit_path=Path(prepared_candidates["overlays"]["xov1_bhw5506"]["path"]),
        run_dir=STORE / "retained/exact/xov1_bhw5506",
        resume_frame=None,
    )
    if int(run["edit"]["tokens_changed"]) != 5_506:
        raise Jbp1Error("candidate A exact run did not consume all 5,506 edits")
    payload = {
        "schema": "ddm_jbp1_candidate_rate_row.v1",
        "candidate_id": "xov1_bhw5506",
        "axis": AXIS,
        "score_claim": False,
        "mechanism": "changed X under the shipped causal graph/model with exact adaptive-state replay",
        "gm_scope": "no distinct G edit declared by XOV1 candidate 3; shipped G/M used physically",
        "field": prepared_candidates["fields"]["xov1_bhw5506"],
        "overlay": prepared_candidates["overlays"]["xov1_bhw5506"],
        "run_receipt": file_fact(STORE / "retained/exact/xov1_bhw5506/RESULT.json"),
        "stream": run["stream"],
        "archive": run["archive"],
        "terminal_checkpoint": run["terminal_checkpoint"],
        "wall_seconds": run["wall_seconds"],
        "demand_arithmetic": price_row(int(run["archive_delta_bytes"])),
        "distortion": "NOT MEASURED; RATE ROW ONLY",
    }
    write_once_json(receipt, payload)
    return payload


def stage_sfp1_blocker() -> dict[str, object]:
    stage_candidate_a()
    prepared = stage_prepare_candidates()
    audit = prepared["sfp1_refit_audit"]
    if not audit["blocks_fixed_gm_standin"]:
        raise Jbp1Error("SFP1 executable refit binding appeared; extend the runner before pricing")
    receipt = STORE / "SFP1_GM_REFIT_BLOCKER.json"
    payload = {
        "schema": "ddm_jbp1_typed_blocker.v1",
        "status": "BLOCKED-MISSING-EXECUTABLE-GM-REFIT",
        "verdict_scope": "INSTANCE: the three retained SFP1 proposals and the inherited RXC1/JG2 instrument",
        "candidate_denominator": 3,
        "blocked_candidates": [row["proposal_id"] for row in audit["rows"]],
        "reason": (
            "All three proposals require refit_cross_group_causal_schedule, but their handoff "
            "contains no executable schedule/model/decoder binding and RXC1/JG2 pins the shipped "
            "HPAC section and shipped group plan. A fixed-G/M re-encode would be a mechanism "
            "reduction and cannot be called the chartered joint price."
        ),
        "audit": audit,
        "not_run": {
            "exact_reencodes": 3,
            "scorer_runs": 0,
            "modal_calls": 0,
            "contest_evaluations": 0,
        },
        "fire_order": {
            "disposition": "QUEUED-WITH-A-FIRE-ORDER",
            "owner": "task #1374 SCMDL causal-state/model builder assigned by MAIN",
            "consumer_store": str(STORE),
            "fire_trigger": (
                "a receiver-executable cross-group schedule and counted fitted model are retained, "
                "hashed, parse-back bound, and exposed to the exact coder without changing the "
                "three SFP1 field hashes"
            ),
            "action": "resume B1-B3 in rank order with one exact physical re-encode per fitted G/M row",
        },
    }
    write_once_json(receipt, payload)
    return payload


def stage_result() -> dict[str, object]:
    preflight = stage_preflight()
    null = stage_null()
    candidate_a = stage_candidate_a()
    blocker = stage_sfp1_blocker()
    receipt = STORE / "RESULT.json"
    payload = {
        "schema": "ddm_jbp1_joint_batch_price.v1",
        "status": "PARTIAL-BLOCKED",
        "axis": AXIS,
        "score_claim": False,
        "pointer_moved": False,
        "preflight": preflight,
        "null_identity": null,
        "candidate_rows": [candidate_a],
        "measured_candidate_denominator": 1,
        "chartered_candidate_denominator": 4,
        "blocked_candidate_denominator": 3,
        "sfp1_blocker": blocker,
        "routing": (
            "Candidate A is adjudicated on exact bytes. The roster is not EXHAUSTED-MEASURED "
            "because B1-B3 lack the executable G/M mechanism their own schema requires."
        ),
        "authority_boundaries": {
            "scorer_runs": 0,
            "modal_calls": 0,
            "contest_evaluations": 0,
            "upstream_writes": 0,
        },
    }
    write_once_json(receipt, payload)
    return payload


def stage_manifest() -> dict[str, object]:
    stage_result()
    manifest_path = STORE / "MANIFEST.json"
    entries = []
    for path in sorted(STORE.rglob("*")):
        if not path.is_file() or path == manifest_path or path.name.startswith("."):
            continue
        entries.append(
            {
                "path": str(path.relative_to(STORE)),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    payload = {
        "schema": "ddm_jbp1_manifest.v1",
        "root": str(STORE),
        "entry_count": len(entries),
        "total_bytes": sum(int(row["bytes"]) for row in entries),
        "free_bytes_after_capture": shutil.disk_usage(STORE).free,
        "mandatory_reserve_bytes": RESERVE_BYTES,
        "reserve_pass": shutil.disk_usage(STORE).free >= RESERVE_BYTES,
        "entries": entries,
    }
    atomic_json(manifest_path, payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--stage",
        required=True,
        choices=("preflight", "prepare", "null", "candidate-a", "blocker", "result", "manifest", "all"),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    stage = build_parser().parse_args(argv).stage
    stages = {
        "preflight": stage_preflight,
        "prepare": stage_prepare,
        "null": stage_null,
        "candidate-a": stage_candidate_a,
        "blocker": stage_sfp1_blocker,
        "result": stage_result,
        "manifest": stage_manifest,
        "all": stage_manifest,
    }
    payload = stages[stage]()
    print(json.dumps({"stage": stage, "status": payload.get("status", "PASS")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
