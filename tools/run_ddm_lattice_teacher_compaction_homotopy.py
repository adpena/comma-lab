#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Governed H0 runner for the G21 lattice-teacher compaction takeoff.

The default command refuses execution and reports the exact H0 command. Only
the dense-free teacher index is executable in this landing. The bounded H1 V2
codec exists as a receiver-tested library surface but is intentionally not
wired to the 291 MB selected packet until root review authorizes materializing
that complete-object proposal.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

REPO_ROOT: Final = Path(__file__).resolve().parents[1]
SRC_ROOT: Final = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tac.optimization.ddm_lattice_teacher_solution_index import (  # noqa: E402
    INDEX_SCHEMA,
    LatticeTeacherIndexError,
    TeacherAssetSpec,
    build_solution_index,
    canonical_json_bytes,
    sha256_file,
)
from tac.witness_dsl.c0b_identity_receiver import (  # noqa: E402
    IdentityReceiverError,
    storage_preflight,
)

CONFIG_SCHEMA: Final = "ddm_lattice_teacher_compaction_homotopy_config.v1"
PREFLIGHT_SCHEMA: Final = "ddm_lattice_teacher_compaction_h0_preflight.v1"
STAGE_SCHEMA: Final = "ddm_lattice_teacher_compaction_h0_stage.v1"
DEFAULT_CONFIG: Final = REPO_ROOT / ".omx/research/configs/ddm_lattice_teacher_compaction_homotopy_h0_20260727.json"


class CompactionRunnerError(ValueError):
    """A config, resume, storage, or immutable stage invariant failed."""


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require_sha(value: object, label: str) -> str:
    if type(value) is not str or len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise CompactionRunnerError(f"{label} must be a lowercase SHA-256")
    return value


def _exact_int(value: object, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise CompactionRunnerError(f"{label} must be an exact integer >= {minimum}")
    return value


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CompactionRunnerError(f"{label} must be a JSON object")
    return value


def _resolve_repo_path(value: object, label: str) -> Path:
    if type(value) is not str or not value:
        raise CompactionRunnerError(f"{label} must be a nonempty path string")
    candidate = Path(value)
    return candidate if candidate.is_absolute() else (REPO_ROOT / candidate).resolve()


def _asset(value: object, label: str) -> TeacherAssetSpec:
    row = _mapping(value, label)
    role = row.get("role")
    if type(role) is not str:
        raise CompactionRunnerError(f"{label}.role must be a string")
    return TeacherAssetSpec(
        path=_resolve_repo_path(row.get("path"), f"{label}.path"),
        sha256=_require_sha(row.get("sha256"), f"{label}.sha256"),
        bytes=_exact_int(row.get("bytes"), f"{label}.bytes"),
        role=role,
    )


def _load_config(path: Path) -> tuple[Mapping[str, Any], str]:
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = (REPO_ROOT / config_path).resolve()
    if not config_path.is_file() or config_path.is_symlink():
        raise CompactionRunnerError("config must be one regular non-symlink file")
    raw = config_path.read_bytes()
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CompactionRunnerError("config is not valid JSON") from exc
    config = _mapping(value, "config")
    if (
        config.get("schema") != CONFIG_SCHEMA
        or config.get("lane_id") != "lane_g100_lattice_teacher_compaction_takeoff_20260727"
        or config.get("research_only") is not True
        or config.get("score_claim") is not False
        or config.get("promotion_eligible") is not False
        or config.get("pair_count") != 600
        or config.get("h1_materialization_authorized") is not False
    ):
        raise CompactionRunnerError("config authority/population contract drift")
    _exact_int(config.get("seed"), "config.seed")
    _exact_int(config.get("required_free_bytes"), "config.required_free_bytes", minimum=1)
    _resolve_repo_path(config.get("output_root"), "config.output_root")
    assets = _mapping(config.get("assets"), "config.assets")
    for name in (
        "ms2r_receipt",
        "selected_packet",
        "ms1_receipt",
        "ms1_sense_rows",
        "ms1_factorization",
    ):
        _asset(assets.get(name), f"config.assets.{name}")
    evidence = config.get("additional_encoder_evidence")
    if not isinstance(evidence, list):
        raise CompactionRunnerError("config.additional_encoder_evidence must be a list")
    for index, row in enumerate(evidence):
        _asset(row, f"config.additional_encoder_evidence[{index}]")
    return config, _sha_bytes(raw)


def _atomic_json(path: Path, value: object) -> None:
    if path.exists() or path.is_symlink():
        raise CompactionRunnerError(f"immutable stage already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    if temporary.exists() or temporary.is_symlink():
        raise CompactionRunnerError(f"atomic temporary stage already exists: {temporary}")
    payload = json.dumps(value, sort_keys=True, indent=2, allow_nan=False).encode("utf-8") + b"\n"
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _stage_envelope(
    *,
    stage_id: str,
    config_sha256: str,
    payload: Mapping[str, Any],
) -> dict[str, object]:
    content_sha = _sha_bytes(canonical_json_bytes(payload))
    return {
        "schema": STAGE_SCHEMA,
        "stage_id": stage_id,
        "status": "complete",
        "config_sha256": config_sha256,
        "content_sha256": content_sha,
        "payload": payload,
    }


def _load_completed_stage(
    path: Path,
    *,
    stage_id: str,
    config_sha256: str,
) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CompactionRunnerError(f"resume stage is unreadable: {path}") from exc
    row = _mapping(value, "resume stage")
    payload = _mapping(row.get("payload"), "resume stage payload")
    if (
        row.get("schema") != STAGE_SCHEMA
        or row.get("stage_id") != stage_id
        or row.get("status") != "complete"
        or row.get("config_sha256") != config_sha256
        or row.get("content_sha256") != _sha_bytes(canonical_json_bytes(payload))
    ):
        raise CompactionRunnerError(f"resume stage custody/identity drift: {path}")
    return payload


def _output_root(config: Mapping[str, Any]) -> Path:
    return _resolve_repo_path(config.get("output_root"), "config.output_root")


def _resume_root(value: Path | None, expected_root: Path) -> bool:
    if value is None:
        return False
    resume = Path(value).resolve()
    if resume != expected_root.resolve():
        raise CompactionRunnerError("--resume-from must name this config's exact output_root")
    return True


def _status(config_path: Path) -> dict[str, object]:
    config, config_sha = _load_config(config_path)
    root = _output_root(config)
    stage_root = root / "stage_checkpoints"
    return {
        "schema": "ddm_lattice_teacher_compaction_homotopy_status.v1",
        "config_sha256": config_sha,
        "output_root": str(root),
        "h0_preflight_exists": (stage_root / "00_storage_preflight.json").is_file(),
        "h0_index_exists": (stage_root / "01_solution_index.json").is_file(),
        "h1_materialization_authorized": False,
        "default_execution": "REFUSE",
        "exact_h0_command": (
            f"{sys.executable} {Path(__file__).resolve()} run-h0 --config {Path(config_path).resolve()}"
        ),
        "h1_blocker": "ROOT_REVIEW_REQUIRED_BEFORE_291MB_SELECTED_PACKET_RECODE",
        "pointer_moved": False,
    }


def _run_h0(config_path: Path, *, resume_from: Path | None) -> dict[str, object]:
    config, config_sha = _load_config(config_path)
    root = _output_root(config)
    resume = _resume_root(resume_from, root)
    stage_root = root / "stage_checkpoints"
    preflight_path = stage_root / "00_storage_preflight.json"
    index_path = stage_root / "01_solution_index.json"
    required = _exact_int(config.get("required_free_bytes"), "required_free_bytes", minimum=1)

    if preflight_path.exists():
        if not resume:
            raise CompactionRunnerError("H0 stage already exists; pass --resume-from with the exact output root")
        preflight = _load_completed_stage(
            preflight_path,
            stage_id="00_storage_preflight",
            config_sha256=config_sha,
        )
    else:
        storage = storage_preflight(root, required, allow_local_spill=False)
        preflight = {
            "schema": PREFLIGHT_SCHEMA,
            "config_sha256": config_sha,
            "storage": dict(storage),
            "teacher_bytes_may_enter_payload": False,
            "dense_teacher_persistence_allowed": False,
            "h1_materialization_authorized": False,
            "stage_checkpoint_policy": "immutable-distinct-stage-files",
        }
        _atomic_json(
            preflight_path,
            _stage_envelope(
                stage_id="00_storage_preflight",
                config_sha256=config_sha,
                payload=preflight,
            ),
        )

    if index_path.exists():
        if not resume:
            raise CompactionRunnerError("H0 index already exists; pass --resume-from with the exact output root")
        index = _load_completed_stage(
            index_path,
            stage_id="01_solution_index",
            config_sha256=config_sha,
        )
        if index.get("schema") != INDEX_SCHEMA:
            raise CompactionRunnerError("resumed H0 index schema drift")
    else:
        assets = _mapping(config.get("assets"), "config.assets")
        additional = config.get("additional_encoder_evidence")
        assert isinstance(additional, list)
        index = build_solution_index(
            ms2r_receipt=_asset(assets["ms2r_receipt"], "assets.ms2r_receipt"),
            selected_packet=_asset(assets["selected_packet"], "assets.selected_packet"),
            ms1_receipt=_asset(assets["ms1_receipt"], "assets.ms1_receipt"),
            ms1_sense_rows=_asset(assets["ms1_sense_rows"], "assets.ms1_sense_rows"),
            ms1_factorization=_asset(
                assets["ms1_factorization"],
                "assets.ms1_factorization",
            ),
            additional_encoder_evidence=tuple(
                _asset(row, f"additional_encoder_evidence[{index}]") for index, row in enumerate(additional)
            ),
        )
        _atomic_json(
            index_path,
            _stage_envelope(
                stage_id="01_solution_index",
                config_sha256=config_sha,
                payload=index,
            ),
        )
    return {
        "schema": "ddm_lattice_teacher_compaction_h0_run.v1",
        "status": "complete",
        "config_sha256": config_sha,
        "output_root": str(root),
        "preflight_path": str(preflight_path),
        "preflight_sha256": sha256_file(preflight_path),
        "index_path": str(index_path),
        "index_sha256": sha256_file(index_path),
        "content_root_sha256": index["content_root_sha256"],
        "pair_count": index["pair_count"],
        "dense_teacher_bytes_persisted": index["dense_teacher_bytes_persisted"],
        "candidate_payload_created": False,
        "h1_materialized": False,
        "pointer_moved": False,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        nargs="?",
        choices=("status", "run-h0"),
        default="status",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--resume-from", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = (
            _status(args.config) if args.command == "status" else _run_h0(args.config, resume_from=args.resume_from)
        )
    except (
        CompactionRunnerError,
        IdentityReceiverError,
        LatticeTeacherIndexError,
        OSError,
    ) as exc:
        print(json.dumps({"status": "REFUSE", "error": str(exc)}, sort_keys=True))
        return 2
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
