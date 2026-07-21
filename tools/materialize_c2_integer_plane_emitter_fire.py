#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Materialize, but never execute, a governed full-n600 C2 trainer argv."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from tac.boundary_math.c2_r1b4_curvelet_binding import (
    BINDING_BASIS_ID,
    C2R1B4CurveletBinding,
    C2R1B4CurveletBindingError,
)
from tac.boundary_math.integer_plane_banded_trainer import (
    DEFAULT_STAGE_PLAN,
    BandArtifact,
    C2BandedTrainerError,
    canonical_json,
    policy_from_args,
    sha256_file,
    storage_preflight,
)
from tac.boundary_math.integer_plane_banded_trainer import (
    build_parser as build_trainer_parser,
)
from tac.boundary_math.integer_plane_emitter_byte_close import C2ByteCloseError
from tac.boundary_math.power_diagram_witness import decode_pdw2, encode_pdw2
from tac.witness_dsl.curriculum_dsl import IntegerPlaneEmitter
from tac.witness_dsl.integer_plane_emitter_policy import (
    BasisMode,
    IntegerPlaneEmitterPolicy,
    PolicyMode,
)

SCHEMA = "c2_integer_plane_governed_fire_config.v2"
SSD_ROOTS = (Path("/Volumes/VertigoDataTier/pact"), Path("/Volumes/APDataStore/pact"))


class FireConfigError(ValueError):
    """Stale, incompatible, or non-governed materialization request."""


class StorageRefusal(RuntimeError):
    """Insufficient governed storage (the sole rc=4 condition)."""


def _beneath_ssd(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    for root in SSD_ROOTS:
        try:
            resolved.relative_to(root)
        except ValueError:
            continue
        return resolved
    raise FireConfigError(f"full-run bulk path is outside the SSD waterfall: {resolved}")


def _base_packet_compatible(path: Path) -> None:
    import zipfile

    with zipfile.ZipFile(path, "r") as archive:
        infos = archive.infolist()
        if len(infos) != 1 or infos[0].filename != "0.bin" or infos[0].is_dir() or infos[0].flag_bits & 1:
            raise FireConfigError("base archive is not the byte-close-compatible one-packet grammar")


def _hash_record(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve(strict=True)
    return {"path": str(resolved), "bytes": resolved.stat().st_size, "sha256": sha256_file(resolved)}


def materialize(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = _beneath_ssd(args.output_dir)
    cold_store = _beneath_ssd(args.cold_store)
    base_archive = args.base_archive.expanduser().resolve(strict=True)
    base_decoder = args.base_decoder.expanduser().resolve(strict=True)
    cache = args.cache.expanduser().resolve(strict=True)
    band = BandArtifact.load(args.band_manifest)
    if band.mode != "positive_anisotropic":
        raise FireConfigError("full fire requires a positive anisotropic band artifact")
    binding = C2R1B4CurveletBinding.load(args.carrier_binding)
    if binding.band_manifest_sha256 != band.manifest_sha256:
        raise FireConfigError("carrier binding does not supersede this exact band manifest")
    if args.basis != BasisMode.R1B4_WINDOWED_CURVELET.value:
        raise FireConfigError("M1 fire requires the receiver-bound r1b4 windowed-curvelet basis")
    pdw2_path = args.pdw2_packet.expanduser().resolve(strict=True)
    pdw2_bytes = pdw2_path.read_bytes()
    try:
        pdw2 = decode_pdw2(pdw2_bytes)
    except ValueError as exc:
        raise FireConfigError("full fire requires strict canonical #553 PDW2 bytes") from exc
    if encode_pdw2(pdw2) != pdw2_bytes:
        raise FireConfigError("full fire PDW2 bytes fail parse/re-encode")
    _base_packet_compatible(base_archive)
    preflights = {
        "output": storage_preflight(output_dir, required_free_bytes=args.required_free_bytes),
        "cold_store": storage_preflight(cold_store, required_free_bytes=args.required_free_bytes),
    }
    if not all(row["ok"] for row in preflights.values()):
        raise StorageRefusal("insufficient SSD storage")
    policy = IntegerPlaneEmitterPolicy(basis=BasisMode(args.basis), mode=PolicyMode.BANDED_TRAINING)
    lever = IntegerPlaneEmitter(policy=policy)
    contract = policy.compile_contract()
    if args.resume_from is not None:
        resume = args.resume_from.expanduser().resolve(strict=True)
        resume_record: dict[str, Any] | None = _hash_record(resume)
    else:
        resume_record = None
    trainer_argv = [
        "python3",
        "experiments/train_c2_integer_plane_emitter_banded.py",
    ]
    for flag, value in lever.overrides.items():
        trainer_argv.extend((flag, str(value)))
    trainer_argv.extend(
        (
            "--base-archive",
            str(base_archive),
            "--base-decoder",
            str(base_decoder),
            "--base-archive-sha256",
            sha256_file(base_archive),
            "--base-decoder-sha256",
            sha256_file(base_decoder),
            "--band-manifest",
            str(band.manifest_path),
            "--r1b4-carrier-binding",
            str(binding.manifest_path),
            "--output-dir",
            str(output_dir),
            "--scratch-root",
            str(output_dir / "scratch"),
            "--required-free-bytes",
            str(args.required_free_bytes),
            "--run-id",
            args.run_id,
            "--seed",
            str(args.seed),
            "--pair-batch-size",
            str(args.pair_batch_size),
            "--checkpoint-every-steps",
            str(args.checkpoint_every_steps),
            "--ema-decay",
            str(args.ema_decay),
            "--stage-plan-json",
            canonical_json([stage.to_dict() for stage in DEFAULT_STAGE_PLAN]).decode("ascii"),
            "--receipt",
            str(output_dir / f"{args.run_id}.training_receipt.json"),
        )
    )
    if args.resume_from is not None:
        trainer_argv.extend(("--resume-from", str(args.resume_from.expanduser().resolve(strict=True))))
    parsed_trainer = build_trainer_parser().parse_args(trainer_argv[2:])
    if policy_from_args(parsed_trainer).compile_contract() != contract:
        raise FireConfigError("compiled DSL is not consumed by the exact C2 trainer parser")
    config = {
        "schema": SCHEMA,
        "authority": "materialized_ready_not_fired",
        "readiness": "READY",
        "verdict_scope": (
            "inputs parse and bind for config-only review; this is not trainer, receiver-byte, "
            "score, dispatch, or promotion authority"
        ),
        "blocking_gates": [],
        "remaining_preflights": [
            "governed_launcher_governor",
            "witness_memory_preflight",
        ],
        "logical_pair_count": 600,
        "policy_contract": contract,
        "dsl_overrides": lever.overrides,
        "runtime_receipt_schemas": lever.runtime_receipt_schemas,
        "base_archive": _hash_record(base_archive),
        "base_decoder": _hash_record(base_decoder),
        "cache": _hash_record(cache),
        "band_manifest": _hash_record(band.manifest_path),
        "band_source_sha256": band.source_sha256,
        "carrier_binding": _hash_record(binding.manifest_path),
        "carrier_topology_sha256": binding.topology_sha256,
        "receiver_binding": {
            "schema": "r1b4_section_receiver.v1",
            "archive_section": "boundary_coordinate.bgj",
            "packet_path_after_training": str(output_dir / f"{args.run_id}.ema.{BINDING_BASIS_ID}.bgj"),
            "train_side_emission": True,
            "receiver_side_consumption": True,
            "byte_accounting": "actual_counted_zip_member_bytes",
            "semantic_frame": 1,
            "frame0_factorization": "single_se3_xi_twist",
            "quantization_strata": {
                "realizable_pixels": binding.selected_pixel_count,
                "substep_dead_pixels": binding.dead_pixel_count,
                "dead_stratum_optimization_weight": 0,
                "shared_packet_bytes_pixel_attribution": "not_decomposable_before_receiver_effect_measurement",
                "dead_stratum_spatial_effect": "not_measured_no_launch",
            },
        },
        "pdw2_packet": _hash_record(pdw2_path),
        "resume_checkpoint": resume_record,
        "seed": args.seed,
        "stage_plan": [stage.to_dict() for stage in DEFAULT_STAGE_PLAN],
        "output_dir": str(output_dir),
        "cold_store": str(cold_store),
        "storage_preflight": preflights,
        "trainer_argv": trainer_argv,
        "launch": False,
        "paid_dispatch": False,
        "score_claim": False,
        "pointer_mutation": False,
    }
    return config


def validate_materialized(path: Path) -> dict[str, Any]:
    raw = path.expanduser().resolve(strict=True).read_bytes()
    try:
        config = json.loads(raw.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FireConfigError("fire config is not ASCII JSON") from exc
    if canonical_json(config) != raw or config.get("schema") != SCHEMA:
        raise FireConfigError("fire config is noncanonical or has wrong schema")
    for name in ("base_archive", "base_decoder", "cache", "band_manifest", "carrier_binding"):
        record = config.get(name)
        if not isinstance(record, dict) or sha256_file(record["path"]) != record["sha256"]:
            raise FireConfigError(f"stale {name} custody")
    pdw2_record = config.get("pdw2_packet")
    if not isinstance(pdw2_record, dict) or sha256_file(pdw2_record["path"]) != pdw2_record["sha256"]:
        raise FireConfigError("stale pdw2_packet custody")
    pdw2_bytes = Path(pdw2_record["path"]).read_bytes()
    try:
        pdw2 = decode_pdw2(pdw2_bytes)
    except ValueError as exc:
        raise FireConfigError("stale PDW2 parse custody") from exc
    if encode_pdw2(pdw2) != pdw2_bytes:
        raise FireConfigError("stale PDW2 canonical custody")
    resume = config.get("resume_checkpoint")
    if resume is not None and sha256_file(resume["path"]) != resume["sha256"]:
        raise FireConfigError("stale resume checkpoint custody")
    policy = IntegerPlaneEmitterPolicy(
        basis=BasisMode(config["policy_contract"]["basis"]),
        mode=PolicyMode.BANDED_TRAINING,
    )
    if policy.compile_contract() != config["policy_contract"]:
        raise FireConfigError("stale policy contract custody")
    band = BandArtifact.load(config["band_manifest"]["path"])
    if band.mode != "positive_anisotropic" or band.source_sha256 != config["band_source_sha256"]:
        raise FireConfigError("stale or nonpositive band custody")
    binding = C2R1B4CurveletBinding.load(config["carrier_binding"]["path"])
    if (
        binding.band_manifest_sha256 != config["band_manifest"]["sha256"]
        or binding.topology_sha256 != config["carrier_topology_sha256"]
    ):
        raise FireConfigError("stale carrier binding custody")
    trainer_argv = config.get("trainer_argv")
    if not isinstance(trainer_argv, list) or trainer_argv[:2] != [
        "python3",
        "experiments/train_c2_integer_plane_emitter_banded.py",
    ]:
        raise FireConfigError("ready config lacks the exact trainer argv")
    parsed_trainer = build_trainer_parser().parse_args(trainer_argv[2:])
    if policy_from_args(parsed_trainer).compile_contract() != config["policy_contract"]:
        raise FireConfigError("ready config trainer argv no longer consumes the policy")
    if (
        config.get("readiness") != "READY"
        or not isinstance(config.get("blocking_gates"), list)
        or config["blocking_gates"]
        or config.get("remaining_preflights")
        != ["governed_launcher_governor", "witness_memory_preflight"]
        or config.get("launch") is not False
    ):
        raise FireConfigError("ready config fail-closed fields mismatch")
    return {
        "schema": SCHEMA,
        "valid": True,
        "readiness": "READY",
        "blocking_gates": config["blocking_gates"],
        "remaining_preflights": config["remaining_preflights"],
        "fire_executed": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    make = subparsers.add_parser("materialize")
    make.add_argument("--base-archive", type=Path, required=True)
    make.add_argument("--base-decoder", type=Path, required=True)
    make.add_argument("--cache", type=Path, required=True)
    make.add_argument("--band-manifest", type=Path, required=True)
    make.add_argument("--carrier-binding", type=Path, required=True)
    make.add_argument("--pdw2-packet", type=Path, required=True)
    make.add_argument("--output-dir", type=Path, required=True)
    make.add_argument("--cold-store", type=Path, required=True)
    make.add_argument("--config", type=Path, required=True)
    make.add_argument(
        "--basis",
        choices=[mode.value for mode in BasisMode],
        default=BasisMode.R1B4_WINDOWED_CURVELET.value,
    )
    make.add_argument("--run-id", required=True)
    make.add_argument("--seed", type=int, default=20260719)
    make.add_argument("--pair-batch-size", type=int, default=2)
    make.add_argument("--checkpoint-every-steps", type=int, default=50)
    make.add_argument("--ema-decay", type=float, default=0.997)
    make.add_argument("--required-free-bytes", type=int, default=8_000_000_000)
    make.add_argument("--resume-from", type=Path)
    check = subparsers.add_parser("check")
    check.add_argument("--config", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "check":
            receipt = validate_materialized(args.config)
            print(canonical_json(receipt).decode("ascii"))
            return 0
        config = materialize(args)
        args.config.parent.mkdir(parents=True, exist_ok=True)
        if args.config.exists():
            raise FireConfigError(f"config overwrite refused: {args.config}")
        args.config.write_bytes(canonical_json(config))
    except StorageRefusal as exc:
        print(f"C2 fire storage refusal: {exc}", file=sys.stderr)
        return 4
    except (
        FireConfigError,
        C2BandedTrainerError,
        C2ByteCloseError,
        C2R1B4CurveletBindingError,
        OSError,
        ValueError,
    ) as exc:
        print(f"C2 fire custody/config refusal: {exc}", file=sys.stderr)
        return 6
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
