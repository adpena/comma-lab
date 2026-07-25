#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build the C0 identity scaffold for the original V9-to-V10 witness codec.

This tool hashes live source/evaluator/receiver bytes and emits canonical
``CoupledWitnessState``, ``WitnessCompileConfig``, and ``codec_object.v1``
envelopes.  It deliberately emits no archive, decoded output, score, or
promotion claim; those require future executable edge receipts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from tac.witness_dsl.coupled_witness_state import (
    CodecObjectManifest,
    ContentAddress,
    CoupledWitnessState,
    FrozenSpaceIdentity,
    WitnessCompileConfig,
    canonical_json_bytes,
    canonical_sha256,
)

SPEC_SCHEMA = "tac.coupled_witness_scaffold_spec.v1"
RECEIPT_SCHEMA = "tac.coupled_witness_scaffold_receipt.v1"


class ScaffoldBuildError(ValueError):
    """Fail-closed scaffold configuration or file-custody error."""


def _strict_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ScaffoldBuildError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _decode_json(payload_bytes: bytes, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(
            payload_bytes.decode("utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ScaffoldBuildError(f"non-finite JSON constant: {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ScaffoldBuildError(f"cannot decode scaffold spec {label}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ScaffoldBuildError("scaffold spec must be an object")
    return payload


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload_bytes = path.read_bytes()
    except OSError as exc:
        raise ScaffoldBuildError(f"cannot read scaffold spec: {exc}") from exc
    return _decode_json(payload_bytes, label=path.name)


def _exact_keys(value: Mapping[str, Any], expected: set[str], name: str) -> None:
    if not isinstance(value, Mapping):
        raise ScaffoldBuildError(f"{name} must be an object")
    if set(value) != expected:
        raise ScaffoldBuildError(
            f"{name} fields differ: missing={sorted(expected - set(value))}, "
            f"extra={sorted(set(value) - expected)}"
        )


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or value.strip() != value or not value:
        raise ScaffoldBuildError(f"{name} must be a non-empty trimmed string")
    return value


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ScaffoldBuildError(f"{name} must be a non-negative exact integer")
    return value


def _positive_int(value: Any, name: str) -> int:
    result = _nonnegative_int(value, name)
    if result == 0:
        raise ScaffoldBuildError(f"{name} must be positive")
    return result


def _relative_input_path(path_value: Any, *, repo_root: Path, name: str) -> Path:
    relative = Path(_text(path_value, name))
    if relative.is_absolute() or ".." in relative.parts:
        raise ScaffoldBuildError(f"{name} must be a repository-relative path")
    root = repo_root.resolve()
    resolved = (root / relative).resolve()
    if resolved != root and root not in resolved.parents:
        raise ScaffoldBuildError(f"{name} escapes repository root")
    if not resolved.is_file():
        raise ScaffoldBuildError(f"{name} is not a file: {relative.as_posix()}")
    return resolved


def _content_address(
    value: Mapping[str, Any],
    *,
    repo_root: Path,
    name: str,
) -> ContentAddress:
    _exact_keys(value, {"path", "artifact_schema"}, name)
    path_text = _text(value["path"], f"{name}.path")
    source = _relative_input_path(path_text, repo_root=repo_root, name=f"{name}.path")
    return ContentAddress.from_payload(
        artifact_id=Path(path_text).as_posix(),
        artifact_schema=_text(value["artifact_schema"], f"{name}.artifact_schema"),
        payload=source.read_bytes(),
    )


def _geometry(value: Mapping[str, Any], name: str) -> tuple[int, int]:
    _exact_keys(value, {"height", "width"}, name)
    return (
        _positive_int(value["height"], f"{name}.height"),
        _positive_int(value["width"], f"{name}.width"),
    )


def build_scaffold(
    spec: Mapping[str, Any],
    *,
    repo_root: Path,
    spec_file_sha256: str,
) -> tuple[CoupledWitnessState, WitnessCompileConfig, CodecObjectManifest, dict[str, Any]]:
    """Build and self-verify the exact C0 identities from one typed spec."""

    expected = {
        "schema",
        "source_video",
        "evaluator_artifacts",
        "receiver_artifacts",
        "pair_count",
        "pair_order_id",
        "scorer_geometry",
        "generation",
        "compile",
        "research_only",
        "score_claim",
        "promotion_eligible",
    }
    _exact_keys(spec, expected, "scaffold spec")
    if spec["schema"] != SPEC_SCHEMA:
        raise ScaffoldBuildError("scaffold spec schema differs")
    if (
        spec["research_only"] is not True
        or spec["score_claim"] is not False
        or spec["promotion_eligible"] is not False
    ):
        raise ScaffoldBuildError("scaffold spec false-authority fields differ")

    source_video = _content_address(
        spec["source_video"],
        repo_root=repo_root,
        name="source_video",
    )
    evaluator_rows = spec["evaluator_artifacts"]
    receiver_rows = spec["receiver_artifacts"]
    if not isinstance(evaluator_rows, list) or not evaluator_rows:
        raise ScaffoldBuildError("evaluator_artifacts must be a non-empty array")
    if not isinstance(receiver_rows, list) or not receiver_rows:
        raise ScaffoldBuildError("receiver_artifacts must be a non-empty array")
    evaluator_artifacts = tuple(
        sorted(
            (
                _content_address(
                    row,
                    repo_root=repo_root,
                    name=f"evaluator_artifacts[{index}]",
                )
                for index, row in enumerate(evaluator_rows)
            ),
            key=lambda item: item.artifact_id,
        )
    )
    receiver_artifacts = tuple(
        sorted(
            (
                _content_address(
                    row,
                    repo_root=repo_root,
                    name=f"receiver_artifacts[{index}]",
                )
                for index, row in enumerate(receiver_rows)
            ),
            key=lambda item: item.artifact_id,
        )
    )
    pair_count = _positive_int(spec["pair_count"], "pair_count")
    scorer_height, scorer_width = _geometry(spec["scorer_geometry"], "scorer_geometry")
    generation = spec["generation"]
    _exact_keys(generation, {"seed", "rng_id"}, "generation")
    frozen_space = FrozenSpaceIdentity(
        source_video=source_video,
        evaluator_artifacts=evaluator_artifacts,
        pair_count=pair_count,
        pair_order_id=_text(spec["pair_order_id"], "pair_order_id"),
        pair_order_sha256=canonical_sha256(list(range(pair_count))),
        scorer_height=scorer_height,
        scorer_width=scorer_width,
    )
    state = CoupledWitnessState.empty(
        frozen_space,
        generation_seed=_nonnegative_int(generation["seed"], "generation.seed"),
        generation_rng_id=_text(generation["rng_id"], "generation.rng_id"),
    )

    compile_row = spec["compile"]
    _exact_keys(
        compile_row,
        {
            "container_id",
            "receiver_contract_id",
            "r_chain_id",
            "tie_policy_id",
            "camera_geometry",
            "decoder_seed",
            "decoder_payload_policy",
        },
        "compile",
    )
    camera_height, camera_width = _geometry(compile_row["camera_geometry"], "camera_geometry")
    config = WitnessCompileConfig(
        container_id=_text(compile_row["container_id"], "compile.container_id"),
        receiver_contract_id=_text(
            compile_row["receiver_contract_id"], "compile.receiver_contract_id"
        ),
        receiver_artifacts=receiver_artifacts,
        r_chain_id=_text(compile_row["r_chain_id"], "compile.r_chain_id"),
        tie_policy_id=_text(compile_row["tie_policy_id"], "compile.tie_policy_id"),
        camera_height=camera_height,
        camera_width=camera_width,
        scorer_height=scorer_height,
        scorer_width=scorer_width,
        decoder_seed=_nonnegative_int(compile_row["decoder_seed"], "compile.decoder_seed"),
        stream_policies=(),
        decoder_payload_policy=_text(
            compile_row["decoder_payload_policy"], "compile.decoder_payload_policy"
        ),
    )
    codec_object = CodecObjectManifest.bind(state, config)

    state_bytes = state.to_bytes()
    config_bytes = config.to_bytes()
    object_bytes = codec_object.to_bytes()
    if CoupledWitnessState.from_bytes(state_bytes) != state:
        raise ScaffoldBuildError("state envelope roundtrip differs")
    if WitnessCompileConfig.from_bytes(config_bytes) != config:
        raise ScaffoldBuildError("compile envelope roundtrip differs")
    if CodecObjectManifest.from_bytes(object_bytes) != codec_object:
        raise ScaffoldBuildError("codec-object envelope roundtrip differs")

    receipt: dict[str, Any] = {
        "schema": RECEIPT_SCHEMA,
        "status": "C0_IDENTITY_SCAFFOLD_LANDED",
        "spec_file_sha256": spec_file_sha256,
        "spec_canonical_sha256": canonical_sha256(spec),
        "state_sha256": state.state_sha256,
        "frozen_space_sha256": frozen_space.identity_sha256,
        "compile_config_sha256": config.config_sha256,
        "receiver_bundle_sha256": config.receiver_bundle_sha256,
        "codec_object_sha256": codec_object.object_sha256,
        "files": {
            "state.json": {
                "bytes": len(state_bytes),
                "sha256": hashlib.sha256(state_bytes).hexdigest(),
            },
            "compile_config.json": {
                "bytes": len(config_bytes),
                "sha256": hashlib.sha256(config_bytes).hexdigest(),
            },
            "codec_object.json": {
                "bytes": len(object_bytes),
                "sha256": hashlib.sha256(object_bytes).hexdigest(),
            },
        },
        "source_only_lineage": True,
        "borrowed_candidate_bytes": 0,
        "archive_emitted": False,
        "decoded_output_emitted": False,
        "score_measured": False,
        "next_owed_edges": [
            "typed two-plane obligation IR",
            "state-to-archive compile receipt",
            "archive-to-decoded-output receiver receipt",
            "decoded-output-to-score receipts per authority axis",
        ],
        "research_only": True,
        "score_claim": False,
        "promotion_eligible": False,
    }
    receipt["content_sha256"] = canonical_sha256(receipt)
    return state, config, codec_object, receipt


def _write_staged_file_once(path: Path, payload: bytes) -> None:
    if path.exists():
        raise ScaffoldBuildError(f"write-once output already exists: {path.name}")
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_bundle_once(output_dir: Path, files: Mapping[str, bytes]) -> None:
    """Publish a complete immutable bundle under an exclusive sibling lock.

    All members are first written and verified in a private sibling directory.
    The destination becomes visible only through the final directory rename.
    The exclusive lock makes concurrent invocations on the same destination
    fail closed instead of interleaving identities.
    """

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    lock_path = output_dir.with_name(f".{output_dir.name}.publish.lock")
    try:
        lock_descriptor = os.open(
            lock_path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
    except FileExistsError as exc:
        raise ScaffoldBuildError("output publication lock already exists") from exc

    staging: Path | None = None
    try:
        with os.fdopen(lock_descriptor, "wb") as handle:
            handle.write(f"pid={os.getpid()}\n".encode("ascii"))
            handle.flush()
            os.fsync(handle.fileno())
        if output_dir.exists() or output_dir.is_symlink():
            raise ScaffoldBuildError("write-once output directory already exists")
        staging = Path(
            tempfile.mkdtemp(
                prefix=f".{output_dir.name}.bundle.",
                dir=output_dir.parent,
            )
        )
        for name, payload in files.items():
            _write_staged_file_once(staging / name, payload)
        for name, payload in files.items():
            if (staging / name).read_bytes() != payload:
                raise ScaffoldBuildError(f"staged output verification differs: {name}")
        _fsync_directory(staging)
        if output_dir.exists() or output_dir.is_symlink():
            raise ScaffoldBuildError("output directory appeared during publication")
        os.rename(staging, output_dir)
        staging = None
        _fsync_directory(output_dir.parent)
    except ScaffoldBuildError:
        raise
    except OSError as exc:
        raise ScaffoldBuildError(f"cannot publish scaffold bundle: {exc}") from exc
    finally:
        if staging is not None:
            shutil.rmtree(staging, ignore_errors=True)
        lock_path.unlink(missing_ok=True)


def run(*, spec_path: Path, output_dir: Path, repo_root: Path) -> dict[str, Any]:
    try:
        spec_bytes = spec_path.read_bytes()
    except OSError as exc:
        raise ScaffoldBuildError(f"cannot read scaffold spec: {exc}") from exc
    spec = _decode_json(spec_bytes, label=spec_path.name)
    state, config, codec_object, receipt = build_scaffold(
        spec,
        repo_root=repo_root,
        spec_file_sha256=hashlib.sha256(spec_bytes).hexdigest(),
    )
    _publish_bundle_once(
        output_dir,
        {
            "state.json": state.to_bytes(),
            "compile_config.json": config.to_bytes(),
            "codec_object.json": codec_object.to_bytes(),
            "scaffold_receipt.json": canonical_json_bytes(receipt) + b"\n",
        },
    )
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    receipt = run(
        spec_path=args.spec,
        output_dir=args.output_dir,
        repo_root=args.repo_root,
    )
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
