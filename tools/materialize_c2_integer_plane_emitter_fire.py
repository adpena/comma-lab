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

from tac.boundary_math.integer_plane_banded_trainer import (
    DEFAULT_STAGE_PLAN,
    BandArtifact,
    C2BandedTrainerError,
    canonical_json,
    sha256_file,
    storage_preflight,
)
from tac.boundary_math.integer_plane_emitter_byte_close import C2ByteCloseError
from tac.boundary_math.power_diagram_witness import decode_pdw2, encode_pdw2
from tac.witness_dsl.curriculum_dsl import IntegerPlaneEmitter
from tac.witness_dsl.integer_plane_emitter_policy import (
    BasisMode,
    IntegerPlaneEmitterPolicy,
    PolicyMode,
)

SCHEMA = "c2_integer_plane_governed_fire_config.v1"
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
    pdw2_path = args.pdw2_packet.expanduser().resolve(strict=True)
    pdw2_bytes = pdw2_path.read_bytes()
    try:
        pdw2 = decode_pdw2(pdw2_bytes)
    except ValueError as exc:
        raise FireConfigError("full fire requires strict canonical #553 PDW2 bytes") from exc
    if encode_pdw2(pdw2) != pdw2_bytes:
        raise FireConfigError("full fire PDW2 bytes fail parse/re-encode")
    raise FireConfigError(
        "full fire blocked: #553 PDW2 remains target-only with no scorer-free spatial/RGB "
        "pullback in #543; the current C2 coordinate topology is a polynomial control, not a "
        "receiver-bound curvelet/shearlet carrier; executable Fisher/secant/QP EV field custody "
        "is required by the positive-band manifest"
    )
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
    stage_json = json.dumps(
        [stage.to_dict() for stage in DEFAULT_STAGE_PLAN],
        sort_keys=True,
        separators=(",", ":"),
    )
    trainer = (Path(__file__).resolve().parents[1] / "experiments/train_c2_integer_plane_emitter_banded.py").resolve(
        strict=True
    )
    argv = [sys.executable, str(trainer)]
    for flag, value in lever.overrides.items():
        argv.extend((flag, str(value).lower() if isinstance(value, bool) else str(value)))
    argv.extend(
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
            "--output-dir",
            str(output_dir),
            "--scratch-root",
            str(cold_store / "scratch"),
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
            stage_json,
            "--receipt",
            str(output_dir / "training_receipt.json"),
        )
    )
    if args.resume_from is not None:
        resume = args.resume_from.expanduser().resolve(strict=True)
        argv.extend(("--resume-from", str(resume)))
        resume_record: dict[str, Any] | None = _hash_record(resume)
    else:
        resume_record = None
    config = {
        "schema": SCHEMA,
        "authority": "materialized_only_not_fired",
        "logical_pair_count": 600,
        "policy_contract": contract,
        "dsl_overrides": lever.overrides,
        "runtime_receipt_schemas": lever.runtime_receipt_schemas,
        "base_archive": _hash_record(base_archive),
        "base_decoder": _hash_record(base_decoder),
        "cache": _hash_record(cache),
        "band_manifest": _hash_record(band.manifest_path),
        "band_source_sha256": band.source_sha256,
        "resume_checkpoint": resume_record,
        "seed": args.seed,
        "stage_plan": [stage.to_dict() for stage in DEFAULT_STAGE_PLAN],
        "output_dir": str(output_dir),
        "cold_store": str(cold_store),
        "storage_preflight": preflights,
        "trainer_argv": argv,
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
    for name in ("base_archive", "base_decoder", "cache", "band_manifest"):
        record = config.get(name)
        if not isinstance(record, dict) or sha256_file(record["path"]) != record["sha256"]:
            raise FireConfigError(f"stale {name} custody")
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
    return {"schema": SCHEMA, "valid": True, "fire_executed": False}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    make = subparsers.add_parser("materialize")
    make.add_argument("--base-archive", type=Path, required=True)
    make.add_argument("--base-decoder", type=Path, required=True)
    make.add_argument("--cache", type=Path, required=True)
    make.add_argument("--band-manifest", type=Path, required=True)
    make.add_argument("--pdw2-packet", type=Path, required=True)
    make.add_argument("--output-dir", type=Path, required=True)
    make.add_argument("--cold-store", type=Path, required=True)
    make.add_argument("--config", type=Path, required=True)
    make.add_argument("--basis", choices=[mode.value for mode in BasisMode], default=BasisMode.RAW_CENTERED.value)
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
    except (FireConfigError, C2BandedTrainerError, C2ByteCloseError, OSError, ValueError) as exc:
        print(f"C2 fire custody/config refusal: {exc}", file=sys.stderr)
        return 6
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
