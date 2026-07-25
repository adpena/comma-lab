#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Link identity-complete applied actions into a strict v2 packet.

This is a thin caller around :mod:`tac.v2_compose.applied_action_linker`.
It writes a deterministic blocker receipt when source custody is incomplete;
it never emits a partial candidate packet.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from tac.analysis.applied_action_receipt import AppliedActionReceipt
from tac.v2_compose.applied_action_linker import (
    V2PacketLinkAttempt,
    V2Section,
    V2SectionReplacement,
    try_link_v2_applied_actions,
)
from tac.v2_compose.archive_grammar import generate_v2_inflate_py, generate_v2_inflate_sh

LINK_INPUT_SCHEMA = "tac.applied_action_packet_link_input.v1"


def _decode_json_object(payload_bytes: bytes, *, label: str) -> dict[str, Any]:
    def _reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate input manifest key: {key}")
            result[key] = value
        return result

    try:
        payload = json.loads(
            payload_bytes.decode("utf-8"),
            object_pairs_hook=_reject_duplicates,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite input manifest constant: {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"cannot decode JSON manifest {label}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("input manifest must be a JSON object")
    return payload


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        payload_bytes = path.read_bytes()
    except OSError as exc:
        raise ValueError(f"cannot read JSON manifest {path.name}: {exc}") from exc
    return _decode_json_object(payload_bytes, label=path.name)


def _resolve_input(path_text: Any, *, base_dir: Path, field: str) -> Path:
    if not isinstance(path_text, str) or not path_text.strip():
        raise ValueError(f"{field} must be a non-empty path")
    path = Path(path_text)
    if not path.is_absolute():
        path = base_dir / path
    path = path.resolve()
    if not path.is_file():
        raise ValueError(f"{field} is not a file: {path.name}")
    return path


def _canonical_json_sha256(payload: Mapping[str, Any], *, omit: str) -> str:
    canonical = dict(payload)
    canonical.pop(omit, None)
    return hashlib.sha256(
        json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _selected_receipts_from_allocation_plan(
    manifest: Mapping[str, Any], *, base_dir: Path
) -> tuple[str, ...] | None:
    path_text = manifest.get("allocation_plan_path")
    if path_text is None:
        return None
    plan_path = _resolve_input(
        path_text,
        base_dir=base_dir,
        field="allocation_plan_path",
    )
    plan = _read_json_object(plan_path)
    if plan.get("schema") != "tac.seven_home_allocation_plan.v1":
        raise ValueError("allocation plan schema differs")
    declared = plan.get("plan_content_sha256")
    if not isinstance(declared, str) or declared != _canonical_json_sha256(
        plan, omit="plan_content_sha256"
    ):
        raise ValueError("allocation plan content SHA-256 differs")
    if (
        plan.get("research_only") is not True
        or plan.get("promotion_eligible") is not False
        or plan.get("score_claim") is not False
    ):
        raise ValueError("allocation plan false-authority fields differ")
    # EV2 allocation homes and WTNV2 relocation homes describe different base
    # archives and have an empty accepted-home intersection.  Treating a receipt
    # ID list as a relocation plan would erase that object mismatch (and used to
    # mishandle nested aggregate leaves).  The original V9+V10 codec now owes a
    # native compiler; these two historical mechanisms remain separate guards.
    raise ValueError(
        "seven-home allocation plan is object/grammar-incompatible with the WTNV2 linker"
    )


def _load_inputs(
    manifest: Mapping[str, Any], *, base_dir: Path
) -> tuple[bytes, tuple[AppliedActionReceipt, ...], tuple[V2SectionReplacement, ...]]:
    if manifest.get("schema") != LINK_INPUT_SCHEMA:
        raise ValueError("link input manifest schema differs")
    selected_receipt_ids = _selected_receipts_from_allocation_plan(
        manifest,
        base_dir=base_dir,
    )
    base_payload_path = _resolve_input(
        manifest.get("base_payload_path"),
        base_dir=base_dir,
        field="base_payload_path",
    )
    raw_actions = manifest.get("actions")
    if not isinstance(raw_actions, list) or not raw_actions:
        raise ValueError("actions must be a non-empty list")
    receipts: list[AppliedActionReceipt] = []
    replacements: list[V2SectionReplacement] = []
    for index, raw in enumerate(raw_actions):
        if not isinstance(raw, dict):
            raise ValueError(f"actions[{index}] must be an object")
        receipt_payload = raw.get("receipt")
        if not isinstance(receipt_payload, dict):
            raise ValueError(f"actions[{index}].receipt must be an object")
        receipt = AppliedActionReceipt.from_dict(receipt_payload)
        candidate_path = _resolve_input(
            raw.get("candidate_section_path"),
            base_dir=base_dir,
            field=f"actions[{index}].candidate_section_path",
        )
        candidate_bytes = candidate_path.read_bytes()
        replacement = V2SectionReplacement(
            receipt_id=receipt.receipt_id,
            byte_home_id=raw.get("byte_home_id"),
            section=V2Section(raw.get("section")),
            receiver_consumer=raw.get("receiver_consumer"),
            base_section_sha256=raw.get("base_section_sha256"),
            candidate_section_sha256=raw.get("candidate_section_sha256"),
            candidate_section_bytes=candidate_bytes,
        )
        receipts.append(receipt)
        replacements.append(replacement)
    if selected_receipt_ids is not None and set(selected_receipt_ids) != {
        receipt.receipt_id for receipt in receipts
    }:
        raise ValueError("link actions differ from allocation-plan selection")
    return base_payload_path.read_bytes(), tuple(receipts), tuple(replacements)


def _atomic_write(path: Path, data: bytes) -> None:
    """Write one member once inside a private staging directory."""

    with path.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_bundle_once(
    output_dir: Path,
    files: Mapping[str, bytes],
    *,
    executable_names: frozenset[str] = frozenset(),
) -> None:
    """Stage and atomically publish one immutable output bundle.

    The sibling lock serializes concurrent linker invocations.  A destination
    that exists before or appears during staging is never replaced.
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
        raise ValueError("output publication lock already exists") from exc

    staging: Path | None = None
    try:
        with os.fdopen(lock_descriptor, "wb") as handle:
            handle.write(f"pid={os.getpid()}\n".encode("ascii"))
            handle.flush()
            os.fsync(handle.fileno())
        if output_dir.exists() or output_dir.is_symlink():
            raise ValueError("write-once output directory already exists")
        staging = Path(
            tempfile.mkdtemp(
                prefix=f".{output_dir.name}.bundle.",
                dir=output_dir.parent,
            )
        )
        for name, payload in files.items():
            _atomic_write(staging / name, payload)
            if name in executable_names:
                (staging / name).chmod(0o755)
        for name, payload in files.items():
            if (staging / name).read_bytes() != payload:
                raise ValueError(f"staged output verification differs: {name}")
        _fsync_directory(staging)
        if output_dir.exists() or output_dir.is_symlink():
            raise ValueError("output directory appeared during publication")
        os.rename(staging, output_dir)
        staging = None
        _fsync_directory(output_dir.parent)
    finally:
        if staging is not None:
            shutil.rmtree(staging, ignore_errors=True)
        lock_path.unlink(missing_ok=True)


def _best_effort_input_ids(manifest: Mapping[str, Any]) -> tuple[str, ...]:
    values: list[str] = []
    actions = manifest.get("actions")
    if isinstance(actions, list):
        for row in actions:
            if not isinstance(row, Mapping):
                continue
            receipt = row.get("receipt")
            if isinstance(receipt, Mapping):
                value = receipt.get("receipt_id")
                if isinstance(value, str) and value.strip():
                    values.append(value.strip())
    return tuple(sorted(set(values)))


def run(*, input_manifest: Path, output_dir: Path) -> tuple[int, dict[str, Any]]:
    try:
        input_manifest_bytes = input_manifest.read_bytes()
        input_manifest_sha256 = hashlib.sha256(input_manifest_bytes).hexdigest()
    except OSError:
        input_manifest_bytes = None
        input_manifest_sha256 = None
    if output_dir.exists() or output_dir.is_symlink():
        refusal = V2PacketLinkAttempt(
            status="BLOCKED",
            input_receipt_ids=(),
            receipt=None,
            blockers=("OUTPUT_DIR_NONEMPTY_REFUSED",),
            input_manifest_sha256=input_manifest_sha256,
        )
        return 2, refusal.as_dict()

    manifest: dict[str, Any] = {}
    try:
        if input_manifest_bytes is None:
            raise ValueError(f"cannot read JSON manifest {input_manifest.name}")
        manifest = _decode_json_object(
            input_manifest_bytes,
            label=input_manifest.name,
        )
        base_payload, receipts, replacements = _load_inputs(
            manifest,
            base_dir=input_manifest.parent,
        )
        attempt, linked = try_link_v2_applied_actions(
            base_payload=base_payload,
            receipts=receipts,
            replacements=replacements,
        )
    except (OSError, TypeError, ValueError) as exc:
        attempt = V2PacketLinkAttempt(
            status="BLOCKED",
            input_receipt_ids=_best_effort_input_ids(manifest),
            receipt=None,
            blockers=(f"{type(exc).__name__}:{exc}",),
        )
        linked = None

    attempt = replace(
        attempt,
        input_manifest_sha256=input_manifest_sha256,
    )

    payload = attempt.as_dict()
    files: dict[str, bytes] = {}
    if linked is not None:
        files.update(
            {
                "0.bin": linked.payload_bytes,
                "archive.zip": linked.archive_bytes,
                "inflate.py": generate_v2_inflate_py().encode("utf-8"),
                "inflate.sh": generate_v2_inflate_sh().encode("utf-8"),
            }
        )
    # The receipt is the final staged member.  The complete directory becomes
    # visible atomically, so neither crashes nor concurrent linkers can publish
    # a LINKED marker beside foreign or partial bytes.
    files["link_attempt.json"] = (
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    try:
        _publish_bundle_once(
            output_dir,
            files,
            executable_names=frozenset({"inflate.sh"}) if linked is not None else frozenset(),
        )
    except (OSError, ValueError) as exc:
        refusal = V2PacketLinkAttempt(
            status="BLOCKED",
            input_receipt_ids=attempt.input_receipt_ids,
            receipt=None,
            blockers=(f"OUTPUT_BUNDLE_PUBLICATION_REFUSED:{type(exc).__name__}:{exc}",),
            input_manifest_sha256=input_manifest_sha256,
        )
        return 2, refusal.as_dict()
    return (0 if linked is not None else 2), payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    input_manifest = args.input_manifest.resolve()
    output_dir = args.output_dir.resolve()
    returncode, payload = run(
        input_manifest=input_manifest,
        output_dir=output_dir,
    )
    print(json.dumps(payload, sort_keys=True))
    return returncode


if __name__ == "__main__":
    raise SystemExit(main())
