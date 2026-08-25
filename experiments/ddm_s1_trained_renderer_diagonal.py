#!/usr/bin/env python3
"""Seal the S1 trained-renderer diagonal without faking missing interfaces.

S1 is a three-stage composition, not a new trainer: WD3 must train a W96
renderer born from the retained RJ1/GB1 object; JG2 must re-encode the token
field on that moved runtime; and a QS5-style Schur solve must be compiled on
the exact resulting object.  This module verifies the retained inputs,
pre-registers the score arithmetic, and emits a resumable MAIN-owned order.

The current source tree does not yet expose all three cross-object interfaces.
The seal therefore records typed blockers and emits no runnable argv.  A
blocked seal is a real result; turning it into READY without the executable
interfaces would be a fake implementation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

REPO: Final = Path(__file__).resolve().parents[1]
OUTPUT_ROOT: Final = Path("/Volumes/APDataStore/pact/ddm_s1_trained_renderer_diagonal")
RJ1_ROOT: Final = Path("/Volumes/VertigoDataTier/pact/ddm_rj1_renderer_joint_move/precompile_r1")
RJ1_INVENTORY: Final = RJ1_ROOT / "CUSTODY_INVENTORY.json"
RJ1_RESULT: Final = RJ1_ROOT / "RESULT.json"
GB1_ARCHIVE: Final = Path(
    "/Volumes/APDataStore/pact/ddm_gb1_groupbin8_conditioning/retained/candidate_gb1_groupbin8_surprise.zip"
)
WD3_SOURCE: Final = REPO / "experiments/ddm_wd3_scorer_aware_width_distillation.py"
JG2_SOURCE: Final = REPO / "experiments/ddm_jg2_tail_reencode.py"
QS5_SOURCE: Final = REPO / "experiments/ddm_qs5_resolve_compensation.py"

GB1_BYTES: Final = 180_215
GB1_SHA256: Final = "ba1f3830cd51b820d7f9b834a1dcc12e8776a0260f9da57a4e8e0944b988e3a4"
RJ1_INVENTORY_SHA256: Final = "dd3b89b7f9d68f11f3d828457316b748b796aa55fea36304d566dd5cd2f8467c"
RJ1_RESULT_SHA256: Final = "405f89fb32a23fded3ded5b715989c2bf6efe7df6cedc79b98bbf89323fa26f0"
RJ1_TREE_SHA256: Final = "576c16b2159cd3262dfa18e2df7bd53b7f8ac80c9c8dc546ccdc7dd5cd17d88a"
RATE_DENOMINATOR: Final = 37_545_489
RATE_S_PER_BYTE: Final = 25.0 / RATE_DENOMINATOR
BASE_DSEG: Final = 0.00020139
BASE_DPOSE: Final = 6.37e-6
BASE_SCORE: Final = 0.14811799921260607
SEEDS: Final = (20260815, 20260816)
WINDOW_END_EPOCHS: Final = (5, 15, 30)
MINIMUM_FREE_BYTES: Final = 1 << 30
SEAL_ROOT_NAME: Final = "seal_v2"


class S1Error(RuntimeError):
    """An S1 custody, arithmetic, interface, or retention gate failed."""


def sha256_file(path: Path) -> str:
    if not path.is_file():
        raise S1Error(f"required file is absent: {path}")
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def atomic_bytes(path: Path, payload: bytes) -> dict[str, Any]:
    """Retain a payload atomically; resume only from byte-identical content."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if not path.is_file() or path.read_bytes() != payload:
            raise S1Error(f"refusing to overwrite differing retained payload: {path}")
        return file_record(path)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    try:
        with temporary.open("wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return file_record(path)


def canonical_json(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def atomic_json(path: Path, value: object) -> dict[str, Any]:
    return atomic_bytes(path, canonical_json(value))


def _require_record(path: Path, *, size: int | None = None, digest: str) -> dict[str, Any]:
    record = file_record(path)
    if record["sha256"] != digest or (size is not None and record["bytes"] != size):
        raise S1Error(f"source custody differs: {path}")
    return record


def verify_rj1_inventory() -> dict[str, Any]:
    """Verify every retained RJ1 file record, not only the manifest headline."""

    inventory_record = _require_record(RJ1_INVENTORY, digest=RJ1_INVENTORY_SHA256)
    payload = json.loads(RJ1_INVENTORY.read_text(encoding="utf-8"))
    if (
        payload.get("schema") != "ddm_rj1_retained_tree.v1"
        or int(payload.get("file_count", -1)) != 192
        or int(payload.get("payload_bytes", -1)) != 5_375_503
        or payload.get("tree_sha256") != RJ1_TREE_SHA256
    ):
        raise S1Error("RJ1 inventory headline differs")
    verified = 0
    failures = []
    for row in payload["files"]:
        relative = Path(str(row["relative_path"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise S1Error(f"RJ1 inventory path escapes custody root: {relative}")
        path = RJ1_ROOT / relative
        if not path.is_file():
            failures.append(
                {
                    "relative_path": relative.as_posix(),
                    "status": "MISSING",
                    "expected_bytes": int(row["bytes"]),
                    "expected_sha256": row["sha256"],
                }
            )
            continue
        observed = file_record(path)
        if observed["bytes"] != int(row["bytes"]) or observed["sha256"] != row["sha256"]:
            failures.append(
                {
                    "relative_path": relative.as_posix(),
                    "status": "DRIFTED",
                    "expected_bytes": int(row["bytes"]),
                    "expected_sha256": row["sha256"],
                    "observed_bytes": observed["bytes"],
                    "observed_sha256": observed["sha256"],
                }
            )
            continue
        verified += 1
    return {
        "inventory": inventory_record,
        "passed": not failures,
        "verified_file_numerator": verified,
        "inventory_file_denominator": 192,
        "failure_count": len(failures),
        "failures": failures,
        "tree_sha256": RJ1_TREE_SHA256,
    }


def source_preflight() -> dict[str, Any]:
    rj1 = verify_rj1_inventory()
    result = _require_record(RJ1_RESULT, digest=RJ1_RESULT_SHA256)
    gb1 = _require_record(GB1_ARCHIVE, size=GB1_BYTES, digest=GB1_SHA256)
    return {
        "schema": "ddm_s1_source_preflight.v1",
        "passed": rj1["passed"],
        "rj1": rj1,
        "rj1_result": result,
        "gb1_pointer_archive": gb1,
        "instruments": {
            "s1": file_record(Path(__file__).resolve()),
            "wd3": file_record(WD3_SOURCE),
            "jg2": file_record(JG2_SOURCE),
            "qs5": file_record(QS5_SOURCE),
        },
        "source_trees_mutated": False,
    }


def _line_hits(source: str, needle: str) -> list[int]:
    return [number for number, line in enumerate(source.splitlines(), start=1) if needle in line]


def audit_interfaces(
    wd3_source: str,
    jg2_source: str,
    qs5_source: str,
) -> dict[str, Any]:
    """Type the actual cross-object interfaces exposed by the current sources."""

    fixed_seed = _line_hits(wd3_source, 'int(config["seed"]) != SEED')
    wd2_body = {
        "source_streams": _line_hits(wd3_source, "wd2_build._source_streams()"),
        "source_residual": _line_hits(wd3_source, "wd2_build.SOURCE_RESIDUAL"),
        "source_token": _line_hits(wd3_source, "wd2_build.SOURCE_TOKEN"),
    }
    random_birth = _line_hits(wd3_source, "model = receiver.StudentSemanticRenderer(ARM_SPECS")
    jg2_edits = _line_hits(jg2_source, "def apply_edits(")
    jg2_edit_flag = _line_hits(jg2_source, 'parser.add_argument("--edits"')
    qs5_cp135 = _line_hits(qs5_source, "CP135_ARCHIVE")
    qs5_fixed_output = _line_hits(qs5_source, 'Path("/Volumes/VertigoDataTier/pact/ddm_qs5_20260813")')
    blockers = []
    if fixed_seed:
        blockers.append("WD3 compiler admits only seed 20260815, so the charter's second seed cannot compile")
    if any(wd2_body.values()) or random_birth:
        blockers.append(
            "WD3 births W96 from a fresh model and byte-closes against WD2 source sections, not the pinned GB1/RJ1 object"
        )
    if jg2_edits and jg2_edit_flag:
        blockers.append(
            "JG2 can encode a supplied edited field but no S1 stage produces a moved-renderer token field; absent edits it only regenerates the unchanged field"
        )
    else:
        blockers.append("JG2 edited-field input surface is absent")
    if qs5_cp135 or qs5_fixed_output:
        blockers.append(
            "QS5 is hard-pinned to the CP135/QS4 object and governed store; it cannot consume an S1 GB1/W96 archive"
        )
    return {
        "schema": "ddm_s1_interface_audit.v1",
        "wd3": {
            "fixed_seed_guard_lines": fixed_seed,
            "wd2_body_binding_lines": wd2_body,
            "fresh_birth_lines": random_birth,
            "s1_multiseed_gb1_birth_ready": False,
        },
        "jg2": {
            "apply_edits_lines": jg2_edits,
            "edits_cli_lines": jg2_edit_flag,
            "real_reencoder_present": bool(jg2_edits and jg2_edit_flag),
            "moved_field_producer_present": False,
        },
        "qs5": {
            "cp135_binding_lines": qs5_cp135,
            "fixed_output_lines": qs5_fixed_output,
            "generic_exact_object_entrypoint_present": False,
        },
        "blockers": blockers,
        "ready": not blockers,
    }


def break_even_row(bytes_shed: int) -> dict[str, Any]:
    if bytes_shed < 0:
        raise S1Error("bytes_shed must be nonnegative")
    credit = bytes_shed * RATE_S_PER_BYTE
    return {
        "bytes_shed_numerator": bytes_shed,
        "gb1_archive_byte_denominator": GB1_BYTES,
        "rate_exchange_s_per_byte": RATE_S_PER_BYTE,
        "rate_credit_s": credit,
        "maximum_combined_seg_plus_pose_damage_s_for_delta_s_lt_zero": credit,
        "verdict_formula": (
            "delta_s = 100*(d_seg_candidate-0.00020139) + sqrt(10*d_pose_candidate)-sqrt(10*6.37e-6) - rate_credit_s"
        ),
    }


def preregistered_table() -> list[dict[str, Any]]:
    """Score-credit windows fixed before any S1 training or scoring run."""

    return [
        {"window": label, **break_even_row(bytes_shed)}
        for label, bytes_shed in (
            ("zero_credit_control", 0),
            ("film_w96_rj1_observed_renderer_cut", 1_078),
            ("five_kilobyte_cut", 5_000),
            ("ten_kilobyte_cut", 10_000),
            ("fifteen_kilobyte_cut", 15_000),
            ("twenty_kilobyte_cut", 20_000),
            ("full_renderer_block_removed_ceiling", 30_856),
            ("gb1_fixed_distortion_sub012_demand", 42_229),
        )
    ]


def seed_window_rows() -> list[dict[str, Any]]:
    rows = []
    for seed in SEEDS:
        for epoch in WINDOW_END_EPOCHS:
            rows.append(
                {
                    "row_id": f"film_w96_seed_{seed}_epoch_{epoch:04d}",
                    "representation": "film_amortized_flat_w96",
                    "mechanism": "TRAINED-not-SVD W96 renderer",
                    "seed": seed,
                    "window_end_epoch": epoch,
                    "scope_reduction": "bounded training epochs; mechanism unchanged",
                    "bytes_shed": None,
                    "d_seg": None,
                    "d_pose": None,
                    "composed_delta_s": None,
                    "receipt_path": str(OUTPUT_ROOT / f"stage_a/seed_{seed}/epoch_{epoch:04d}/EVALUATION_RESULT.json"),
                    "status": "BLOCKED_NOT_RUN",
                }
            )
    return rows


def compile_seal(
    *,
    source_receipt: Mapping[str, Any],
    interface_audit: Mapping[str, Any],
) -> dict[str, Any]:
    blockers = list(interface_audit["blockers"])
    if source_receipt.get("passed") is not True:
        rj1 = source_receipt.get("rj1", {})
        blockers.insert(
            0,
            "RJ1 custody inventory is not coherent: "
            f"{rj1.get('verified_file_numerator', 0)}/{rj1.get('inventory_file_denominator', 192)} "
            "retained file records verified",
        )
    if not blockers:
        blockers.append(
            "S1 stage adapters are absent; readiness cannot be inferred from source markers or command-shaped strings"
        )
    disposition = "BLOCKED_MISSING_COMPOSED_INTERFACES"
    stages = [
        {
            "stage": "A",
            "claim": "train retained film-amortized W96 from the GB1/RJ1 renderer object under WD3",
            "owner": "MAIN",
            "consumer_store": str(OUTPUT_ROOT / "stage_a"),
            "fire_trigger": "WD3 admits both sealed seeds, consumes the verified RJ1 initializer, and byte-closes on the GB1 body",
            "exact_command_argv": None,
            "status": "BLOCKED",
        },
        {
            "stage": "B",
            "claim": "re-encode the retained moved-object token field with experiments/ddm_jg2_tail_reencode.py",
            "owner": "MAIN",
            "consumer_store": str(OUTPUT_ROOT / "stage_b"),
            "fire_trigger": "stage A retains a receiver-consumed moved runtime plus an explicit n600 token-field payload or typed unchanged-field declaration",
            "exact_command_argv": None,
            "status": "BLOCKED",
        },
        {
            "stage": "C",
            "claim": "solve and compile QS5 Schur compensation on the exact stage-B object",
            "owner": "MAIN",
            "consumer_store": str(OUTPUT_ROOT / "stage_c"),
            "fire_trigger": "QS5 exposes a generic exact-object input contract and binds its solve receipt to the stage-B archive, runtime, frame-1 field, and base Pose6",
            "exact_command_argv": None,
            "status": "BLOCKED",
        },
        {
            "stage": "D",
            "claim": "score the exact compensated archive in chunks of at most 120 pairs",
            "owner": "MAIN sole n600 scorer-lane router",
            "consumer_store": str(OUTPUT_ROOT / "admission"),
            "fire_trigger": "stages A-C are receiver-closed, repeat-identical, fully retained, and their exact source SHAs are sealed",
            "exact_command_argv": None,
            "status": "BLOCKED",
        },
    ]
    return {
        "schema": "ddm_s1_trained_renderer_diagonal_seal.v1",
        "disposition": disposition,
        "score_claim": False,
        "promotion_eligible": False,
        "training_launched": False,
        "scorer_invocations": 0,
        "metal_invocations": 0,
        "modal_invocations": 0,
        "frontier_moved": False,
        "base": {
            "score": BASE_SCORE,
            "d_seg": BASE_DSEG,
            "d_pose": BASE_DPOSE,
            "archive_bytes": GB1_BYTES,
            "archive_sha256": GB1_SHA256,
            "axis": "[contest-CUDA T4 n600]",
        },
        "verified_two_missing_halves": [
            "TRAINED-not-SVD W96 renderer",
            "token re-encode on the moved object",
        ],
        "representation_dispositions": {
            "nested_group_dense_w72": "REFERENCE_ONLY_NOT_W96_CLASS",
            "pointwise_svd_w96_r32": "DEAD_AND_FORBIDDEN_SVD_MECHANISM",
            "film_amortized_flat_w96": "SOLE_ADMISSIBLE_RJ1_W96_FORM_FOR_S1_BUILD",
        },
        "source_preflight": dict(source_receipt),
        "interface_audit": dict(interface_audit),
        "preregistered_break_even": preregistered_table(),
        "seed_window_rows": seed_window_rows(),
        "stages": stages,
        "blockers": blockers,
        "resumable_from": str(OUTPUT_ROOT),
        "all_materialized_payloads_retained": True,
    }


def retention_inventory(output: Path) -> dict[str, Any]:
    records = [
        file_record(path)
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name != "RETENTION_INVENTORY.json" and not path.name.startswith("._")
    ]
    return {
        "schema": "ddm_s1_retention_inventory.v1",
        "root": str(output.resolve()),
        "file_count": len(records),
        "total_bytes": sum(int(record["bytes"]) for record in records),
        "files": records,
        "cleanup_disposition": "KEEP; certify-or-block",
    }


def storage_preflight_payload(receipt_root: Path, output: Path, free: int) -> dict[str, Any]:
    path = receipt_root / "STORAGE_PREFLIGHT.json"
    expected = {
        "schema": "ddm_s1_storage_preflight.v1",
        "root": str(output.resolve()),
        "required_free_bytes": MINIMUM_FREE_BYTES,
        "observed_free_bytes": free,
        "passed": True,
        "cleanup_policy": "certify-or-block; no generated payload deleted",
    }
    if not path.exists():
        return expected
    retained = json.loads(path.read_text(encoding="utf-8"))
    stable_expected = {key: value for key, value in expected.items() if key != "observed_free_bytes"}
    stable_retained = {key: value for key, value in retained.items() if key != "observed_free_bytes"}
    if stable_retained != stable_expected or not isinstance(retained.get("observed_free_bytes"), int):
        raise S1Error(f"retained storage preflight differs: {path}")
    return retained


def seal(output: Path) -> dict[str, Any]:
    if output.resolve() != OUTPUT_ROOT.resolve():
        raise S1Error(f"output must be the governed S1 store: {OUTPUT_ROOT}")
    output.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(output).free
    if free < MINIMUM_FREE_BYTES:
        raise S1Error(f"S1 storage preflight needs {MINIMUM_FREE_BYTES} free bytes, observed {free}")
    receipt_root = output / SEAL_ROOT_NAME
    receipt_root.mkdir(parents=True, exist_ok=True)
    source_receipt = source_preflight()
    audit = audit_interfaces(
        WD3_SOURCE.read_text(encoding="utf-8"),
        JG2_SOURCE.read_text(encoding="utf-8"),
        QS5_SOURCE.read_text(encoding="utf-8"),
    )
    result = compile_seal(source_receipt=source_receipt, interface_audit=audit)
    atomic_json(
        receipt_root / "STORAGE_PREFLIGHT.json",
        storage_preflight_payload(receipt_root, output, free),
    )
    atomic_json(receipt_root / "SOURCE_PREFLIGHT.json", source_receipt)
    atomic_json(receipt_root / "INTERFACE_AUDIT.json", audit)
    atomic_json(receipt_root / "BREAK_EVEN_TABLE.json", result["preregistered_break_even"])
    atomic_json(receipt_root / "SEED_WINDOW_LEDGER.json", result["seed_window_rows"])
    atomic_json(
        receipt_root / "FIRE_ORDER.json",
        {key: result[key] for key in ("disposition", "stages", "blockers")},
    )
    primary = atomic_json(receipt_root / "S1_CHAIN_SEAL.json", result)
    repeat = atomic_json(receipt_root / "S1_CHAIN_SEAL.repeat.json", result)
    if primary["sha256"] != repeat["sha256"] or primary["bytes"] != repeat["bytes"]:
        raise S1Error("S1 seal determinism repeat differs")
    inventory = retention_inventory(receipt_root)
    atomic_json(receipt_root / "RETENTION_INVENTORY.json", inventory)
    return {
        **result,
        "seal": primary,
        "seal_repeat": repeat,
        "seal_repeat_byte_identical": True,
        "retention_inventory": file_record(receipt_root / "RETENTION_INVENTORY.json"),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--resume-from", type=Path, default=OUTPUT_ROOT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.resume_from.resolve() != args.output.resolve():
        raise S1Error("--resume-from must equal --output")
    result = seal(args.output)
    print(
        json.dumps(
            {
                "disposition": result["disposition"],
                "blockers": result["blockers"],
                "seal": result["seal"],
                "frontier_moved": result["frontier_moved"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
