# SPDX-License-Identifier: MIT
"""Behavior tests for exact applied-action packet relocation."""

from __future__ import annotations

import json
import struct
import zlib
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from tac.analysis.action_effect import ActionEffect
from tac.analysis.applied_action_receipt import (
    ApplicationStatus,
    AppliedActionReceipt,
    StreamHomeClaim,
    build_applied_action_receipt,
)
from tac.optimization.ddm_min_description_contract import LayerHome, StreamType
from tac.v2_compose.applied_action_linker import (
    V2_R_CHAIN_ID,
    V2_RECEIVER_SCHEMA,
    AppliedActionLinkError,
    V2PacketLinkAttempt,
    V2PacketLinkReceipt,
    V2Section,
    V2SectionReplacement,
    applied_action_receipt_sha256,
    link_v2_applied_actions,
    try_link_v2_applied_actions,
    v2_archive_marginal_home_id,
    v2_receiver_bundle_sha256,
    v2_receiver_consumer,
    v2_receiver_file_sha256s,
    v2_receiver_sha256,
)
from tac.v2_compose.archive_grammar import (
    build_residual_blob,
    build_store_blob,
    build_v2_archive_zip_bytes,
    generate_v2_inflate_py,
    generate_v2_inflate_sh,
    pack_v2_archive,
    unpack_v2_sections,
)
from tools import link_applied_actions_packet as linker_cli

STATE_SHA = "c" * 64
CHANGED_SHA = "e" * 64


def _sha256(blob: bytes) -> str:
    import hashlib

    return hashlib.sha256(blob).hexdigest()


def test_receiver_bundle_identity_covers_both_runnable_entrypoints() -> None:
    inflate_py = generate_v2_inflate_py().encode("utf-8")
    inflate_sh = generate_v2_inflate_sh().encode("utf-8")
    canonical = v2_receiver_bundle_sha256(inflate_py, inflate_sh)

    assert canonical == v2_receiver_sha256()
    assert canonical != v2_receiver_bundle_sha256(inflate_py + b"\n", inflate_sh)
    assert canonical != v2_receiver_bundle_sha256(inflate_py, inflate_sh + b"\n")
    assert v2_receiver_file_sha256s() == {
        "inflate.py": _sha256(inflate_py),
        "inflate.sh": _sha256(inflate_sh),
    }


def _store_blob(token: float = 0.0) -> bytes:
    labels = np.zeros((1, 8, 8), dtype=np.int64)
    palette = np.zeros((5, 3), dtype=np.float32)
    palette[0, 0] = token
    return build_store_blob(
        [0],
        labels,
        palette,
        (0.0, 0.0, 0.0),
        [0, 0, 2, 0, 1],
        reach_kstar=1,
        n_pairs=1,
    )


def _pose_blob(token: float = 0.0) -> bytes:
    poses = np.full((1, 6), token, dtype=np.float16)
    compressed = zlib.compress(poses.tobytes(), level=9)
    return (
        b"PNTG"
        + struct.pack("<HII", 1, 1, 2)
        + struct.pack("<I", len(compressed))
        + compressed
    )


def _minimal_residual_blob(*, code_rows: int) -> bytes:
    hidden = 2
    mod_dim = 2
    n_hidden = 1
    n_classes = 5
    params = {
        "in_proj.weight": np.zeros((hidden, 2), dtype=np.float32),
        "in_proj.bias": np.zeros((hidden,), dtype=np.float32),
        "film.weight": np.zeros(
            (2 * hidden * n_hidden, mod_dim), dtype=np.float32
        ),
        "film.bias": np.zeros((2 * hidden * n_hidden,), dtype=np.float32),
        "hidden.0.weight": np.zeros((hidden, hidden), dtype=np.float32),
        "hidden.0.bias": np.zeros((hidden,), dtype=np.float32),
        "out_sdf.weight": np.zeros((n_classes, hidden), dtype=np.float32),
        "out_sdf.bias": np.zeros((n_classes,), dtype=np.float32),
        "out_tex.weight": np.zeros((3, hidden), dtype=np.float32),
        "out_tex.bias": np.zeros((3,), dtype=np.float32),
        "palette": np.zeros((n_classes, 3), dtype=np.float32),
        "code": np.zeros((code_rows, mod_dim), dtype=np.float32),
    }
    config = {
        "n_hidden": n_hidden,
        "hidden_dim": hidden,
        "mod_dim": mod_dim,
        "n_classes": n_classes,
        "activation": "hosc",
        "hosc_beta": 4.0,
        "hosc_omega": 1.0,
        "softmax_temp": 0.1,
        "wire_w0": 20.0,
        "wire_s0": 10.0,
        "chroma": True,
        "render_h": 8,
        "render_w": 8,
        "bank_n_scales": 1,
        "bank_n_orient0": 1,
        "bank_f0": 2.0,
        "bank_base": 2.0,
        "bank_n_iso": 0,
        "learn_classes": [1, 3],
        "dilate": 1,
        "mask_mode": "boundary_annulus",
    }
    return build_residual_blob(params, config)


def _base_payload(*, store: bytes | None = None, pose: bytes | None = None) -> bytes:
    return pack_v2_archive(
        _store_blob() if store is None else store,
        b"",
        _pose_blob() if pose is None else pose,
        b'{"format":"v2"}',
    )


def _replace_section(base_payload: bytes, section: V2Section, candidate: bytes) -> bytes:
    parsed = unpack_v2_sections(base_payload)
    sections = {
        V2Section.STORE: parsed.store_blob,
        V2Section.RESIDUAL: parsed.residual_inr_blob,
        V2Section.POSE: parsed.pose_sidecar_blob,
        V2Section.MANIFEST: parsed.manifest_bytes,
    }
    sections[section] = candidate
    return pack_v2_archive(
        sections[V2Section.STORE],
        sections[V2Section.RESIDUAL],
        sections[V2Section.POSE],
        sections[V2Section.MANIFEST],
    )


def _receipt_and_replacement(
    *,
    base_payload: bytes,
    receipt_id: str,
    section: V2Section,
    candidate_section: bytes,
    byte_home_id: str | None = None,
    action_id: str,
) -> tuple[AppliedActionReceipt, V2SectionReplacement]:
    base_archive = build_v2_archive_zip_bytes(base_payload)
    candidate_payload = _replace_section(base_payload, section, candidate_section)
    candidate_archive = build_v2_archive_zip_bytes(candidate_payload)
    byte_home_id = (
        v2_archive_marginal_home_id(section)
        if byte_home_id is None
        else byte_home_id
    )
    effect = ActionEffect.build(
        action_id=action_id,
        family="ddm",
        action_kind="counted_receiver_edge",
        authority="batch_local_receiver_exact",
        producer="fixture",
        pair_ids=(7,),
        old_d_seg=0.002,
        new_d_seg=0.001,
        old_d_pose=0.00002,
        new_d_pose=0.00002,
        old_bytes=len(base_archive),
        new_bytes=len(candidate_archive),
        base_archive_sha256=_sha256(base_archive),
        archive_sha256=_sha256(candidate_archive),
        base_payload_sha256=_sha256(base_payload),
        payload_sha256=_sha256(candidate_payload),
        base_state_sha256=STATE_SHA,
        support_sha256="f" * 64,
    )
    home = StreamHomeClaim(
        stream_type=StreamType.SKELETON,
        layer_home=LayerHome.L3_RASTER,
        byte_home_id=byte_home_id,
        coder_id="v2_zip_deflate",
        coder_owner="v2_packet_linker",
        receiver_consumer=v2_receiver_consumer(section),
        bytes_before=len(base_archive),
        bytes_after=len(candidate_archive),
    )
    receipt = build_applied_action_receipt(
        receipt_id=receipt_id,
        status=ApplicationStatus.DOWNHILL_FINITE,
        action_effect=effect,
        codeword_id=f"codeword:{receipt_id}",
        application_operator_id="fixture.apply_section",
        application_operator_version="v1",
        physical_edge_id=f"edge:{receipt_id}",
        edge_from_state_id="base",
        edge_to_state_id=f"candidate:{receipt_id}",
        integer_quantum=1,
        direction=1,
        validity_radius=1.0,
        receiver_schema=V2_RECEIVER_SCHEMA,
        receiver_sha256=v2_receiver_sha256(),
        r_chain_id=V2_R_CHAIN_ID,
        changed_uint8_count=1,
        changed_uint8_sha256=CHANGED_SHA,
        stream_home=home,
        verdict_scope="INSTANCE:fixture",
        provenance_ref="fixture://applied-action-linker",
    )
    parsed_base = unpack_v2_sections(base_payload)
    base_sections = {
        V2Section.STORE: parsed_base.store_blob,
        V2Section.RESIDUAL: parsed_base.residual_inr_blob,
        V2Section.POSE: parsed_base.pose_sidecar_blob,
        V2Section.MANIFEST: parsed_base.manifest_bytes,
    }
    replacement = V2SectionReplacement(
        receipt_id=receipt_id,
        byte_home_id=byte_home_id,
        section=section,
        receiver_consumer=home.receiver_consumer,
        base_section_sha256=_sha256(base_sections[section]),
        candidate_section_sha256=_sha256(candidate_section),
        candidate_section_bytes=candidate_section,
    )
    return receipt, replacement


def test_single_action_link_reconstructs_exact_scored_candidate() -> None:
    base = _base_payload()
    receipt, replacement = _receipt_and_replacement(
        base_payload=base,
        receipt_id="r-store",
        section=V2Section.STORE,
        candidate_section=_store_blob(1.0),
        action_id="action-store",
    )
    linked = link_v2_applied_actions(
        base_payload=base,
        receipts=[receipt],
        replacements=[replacement],
    )

    assert _sha256(linked.payload_bytes) == receipt.candidate_payload_sha256
    assert _sha256(linked.archive_bytes) == receipt.candidate_archive_sha256
    assert linked.receipt.exact_archive_delta_bytes == receipt.stream_home.delta_bytes
    assert linked.receipt.archive_interaction_bytes == 0
    assert linked.receipt.blockers == ()
    assert linked.receipt.relocations[0].source_receipt_sha256 == (
        applied_action_receipt_sha256(receipt)
    )
    assert linked.receipt.relocations[0].receiver_sha256 == v2_receiver_sha256()
    assert V2PacketLinkReceipt.from_dict(
        json.loads(json.dumps(linked.receipt.as_dict()))
    ) == linked.receipt
    attempt, _ = try_link_v2_applied_actions(
        base_payload=base,
        receipts=[receipt],
        replacements=[replacement],
    )
    assert V2PacketLinkAttempt.from_dict(
        json.loads(json.dumps(attempt.as_dict()))
    ) == attempt

    fractional = linked.receipt.as_dict()
    fractional["individual_archive_delta_sum"] = 3.5
    fractional["archive_interaction_bytes"] = (
        fractional["exact_archive_delta_bytes"] - 3.5
    )
    with pytest.raises(AppliedActionLinkError, match="exact integer"):
        V2PacketLinkReceipt.from_dict(fractional)

    nonstring_blocker = linked.receipt.as_dict()
    nonstring_blocker["blockers"] = [7]
    with pytest.raises(AppliedActionLinkError, match="invalid value"):
        V2PacketLinkReceipt.from_dict(nonstring_blocker)

    shifted_sum = linked.receipt.as_dict()
    shifted_sum["individual_archive_delta_sum"] += 1
    shifted_sum["archive_interaction_bytes"] -= 1
    with pytest.raises(AppliedActionLinkError, match="differs from relocations"):
        V2PacketLinkReceipt.from_dict(shifted_sum)

    relabeled_home = linked.receipt.as_dict()
    relabeled_home["relocations"][0]["byte_home_id"] = "ev2/manifest"
    with pytest.raises(AppliedActionLinkError, match="native v2 marginal"):
        V2PacketLinkReceipt.from_dict(relabeled_home)


def test_multi_action_link_is_permutation_deterministic_and_remeasured() -> None:
    base = _base_payload()
    store = _receipt_and_replacement(
        base_payload=base,
        receipt_id="r-store",
        section=V2Section.STORE,
        candidate_section=_store_blob(1.0),
        action_id="action-store",
    )
    pose = _receipt_and_replacement(
        base_payload=base,
        receipt_id="r-pose",
        section=V2Section.POSE,
        candidate_section=_pose_blob(1.0),
        action_id="action-pose",
    )
    first = link_v2_applied_actions(
        base_payload=base,
        receipts=[pose[0], store[0]],
        replacements=[store[1], pose[1]],
    )
    second = link_v2_applied_actions(
        base_payload=base,
        receipts=[store[0], pose[0]],
        replacements=[pose[1], store[1]],
    )

    assert first.payload_bytes == second.payload_bytes
    assert first.archive_bytes == second.archive_bytes
    assert first.receipt == second.receipt
    assert first.receipt.ordered_receipt_ids == ("r-store", "r-pose")
    assert first.receipt.blockers == ("COMPOSED_SCORE_EFFECT_REMEASUREMENT_REQUIRED",)
    assert first.receipt.archive_interaction_bytes == (
        first.receipt.exact_archive_delta_bytes
        - first.receipt.individual_archive_delta_sum
    )


def test_link_refuses_duplicate_physical_section_or_logical_home() -> None:
    base = _base_payload()
    one = _receipt_and_replacement(
        base_payload=base,
        receipt_id="r1",
        section=V2Section.STORE,
        candidate_section=_store_blob(1.0),
        action_id="action-one",
    )
    two_same_section = _receipt_and_replacement(
        base_payload=base,
        receipt_id="r2",
        section=V2Section.STORE,
        candidate_section=_store_blob(2.0),
        action_id="action-two",
    )
    with pytest.raises(AppliedActionLinkError, match="nested receiver grammar is owed"):
        link_v2_applied_actions(
            base_payload=base,
            receipts=[one[0], two_same_section[0]],
            replacements=[one[1], two_same_section[1]],
        )

    two_same_home = _receipt_and_replacement(
        base_payload=base,
        receipt_id="r3",
        section=V2Section.POSE,
        candidate_section=_pose_blob(3.0),
        byte_home_id="v2.archive_marginal/wrong-pose-home",
        action_id="action-three",
    )
    with pytest.raises(AppliedActionLinkError, match="native v2 archive marginal"):
        link_v2_applied_actions(
            base_payload=base,
            receipts=[one[0], two_same_home[0]],
            replacements=[one[1], two_same_home[1]],
        )


def test_link_refuses_cross_base_and_candidate_identity_mismatch() -> None:
    base = _base_payload()
    other_base = _base_payload(store=_store_blob(9.0))
    one = _receipt_and_replacement(
        base_payload=base,
        receipt_id="r1",
        section=V2Section.STORE,
        candidate_section=_store_blob(1.0),
        action_id="action-one",
    )
    other = _receipt_and_replacement(
        base_payload=other_base,
        receipt_id="r2",
        section=V2Section.POSE,
        candidate_section=_pose_blob(2.0),
        action_id="action-two",
    )
    with pytest.raises(AppliedActionLinkError, match="base archive identity"):
        link_v2_applied_actions(
            base_payload=base,
            receipts=[one[0], other[0]],
            replacements=[one[1], other[1]],
        )

    wrong_candidate = replace(
        one[1],
        candidate_section_sha256=_sha256(_store_blob(4.0)),
        candidate_section_bytes=_store_blob(4.0),
    )
    with pytest.raises(AppliedActionLinkError, match="candidate payload identity differs"):
        link_v2_applied_actions(
            base_payload=base,
            receipts=[one[0]],
            replacements=[wrong_candidate],
        )


def test_link_refuses_outer_valid_but_receiver_invalid_section() -> None:
    base = _base_payload()
    receipt, replacement = _receipt_and_replacement(
        base_payload=base,
        receipt_id="r-inner-parse",
        section=V2Section.STORE,
        candidate_section=_store_blob(1.0),
        action_id="action-inner-parse",
    )
    malformed = b"not-a-valid-store"
    replacement = replace(
        replacement,
        candidate_section_sha256=_sha256(malformed),
        candidate_section_bytes=malformed,
    )

    with pytest.raises(ValueError, match="bad store_blob magic"):
        link_v2_applied_actions(
            base_payload=base,
            receipts=[receipt],
            replacements=[replacement],
        )


def test_link_refuses_parseable_residual_outside_receiver_frame_domain() -> None:
    base = _base_payload()
    receipt, replacement = _receipt_and_replacement(
        base_payload=base,
        receipt_id="r-short-code",
        section=V2Section.RESIDUAL,
        candidate_section=_minimal_residual_blob(code_rows=1),
        action_id="action-short-code",
    )

    with pytest.raises(ValueError, match="receiver frame/modulation domain"):
        link_v2_applied_actions(
            base_payload=base,
            receipts=[receipt],
            replacements=[replacement],
        )


def test_link_refuses_shifted_absolute_archive_byte_endpoints() -> None:
    base = _base_payload()
    receipt, replacement = _receipt_and_replacement(
        base_payload=base,
        receipt_id="r-shifted-bytes",
        section=V2Section.STORE,
        candidate_section=_store_blob(1.0),
        action_id="action-shifted-bytes",
    )
    shifted_effect = replace(
        receipt.action_effect,
        old_bytes=receipt.action_effect.old_bytes + 1_000_000,
        new_bytes=receipt.action_effect.new_bytes + 1_000_000,
    )
    shifted_receipt = replace(receipt, action_effect=shifted_effect)

    with pytest.raises(AppliedActionLinkError, match="base archive byte endpoint differs"):
        link_v2_applied_actions(
            base_payload=base,
            receipts=[shifted_receipt],
            replacements=[replacement],
        )


@pytest.mark.parametrize("blocker_owner", ["receipt", "action_effect"])
def test_link_refuses_unresolved_source_blockers(blocker_owner: str) -> None:
    base = _base_payload()
    receipt, replacement = _receipt_and_replacement(
        base_payload=base,
        receipt_id=f"r-blocked-{blocker_owner}",
        section=V2Section.STORE,
        candidate_section=_store_blob(1.0),
        action_id=f"action-blocked-{blocker_owner}",
    )
    if blocker_owner == "receipt":
        blocked = replace(receipt, blockers=("RECEIVER_CUSTODY_UNRESOLVED",))
    else:
        blocked = replace(
            receipt,
            action_effect=replace(
                receipt.action_effect,
                blockers=("RECEIVER_CUSTODY_UNRESOLVED",),
            ),
        )

    with pytest.raises(AppliedActionLinkError, match="unresolved source blockers"):
        link_v2_applied_actions(
            base_payload=base,
            receipts=[blocked],
            replacements=[replacement],
        )


def test_link_refuses_mixed_or_noncanonical_receiver_identity() -> None:
    base = _base_payload()
    store = _receipt_and_replacement(
        base_payload=base,
        receipt_id="r-store-receiver",
        section=V2Section.STORE,
        candidate_section=_store_blob(1.0),
        action_id="action-store-receiver",
    )
    pose = _receipt_and_replacement(
        base_payload=base,
        receipt_id="r-pose-receiver",
        section=V2Section.POSE,
        candidate_section=_pose_blob(1.0),
        action_id="action-pose-receiver",
    )
    wrong_receiver = replace(pose[0], receiver_sha256="a" * 64)
    with pytest.raises(AppliedActionLinkError, match="one receiver identity"):
        link_v2_applied_actions(
            base_payload=base,
            receipts=[store[0], wrong_receiver],
            replacements=[store[1], pose[1]],
        )

    wrong_consumer = replace(
        store[1],
        receiver_consumer="caller-invented-consumer",
    )
    wrong_consumer_receipt = replace(
        store[0],
        stream_home=replace(
            store[0].stream_home,
            receiver_consumer="caller-invented-consumer",
        ),
    )
    with pytest.raises(AppliedActionLinkError, match="section/receiver consumer"):
        link_v2_applied_actions(
            base_payload=base,
            receipts=[wrong_consumer_receipt],
            replacements=[wrong_consumer],
        )


def test_manifest_only_relocation_is_refused_as_non_receiver_mutation() -> None:
    base = _base_payload()
    parsed = unpack_v2_sections(base)
    with pytest.raises(AppliedActionLinkError, match="manifest-only"):
        V2SectionReplacement(
            receipt_id="r-manifest",
            byte_home_id="home/manifest",
            section=V2Section.MANIFEST,
            receiver_consumer="receiver:manifest",
            base_section_sha256=_sha256(parsed.manifest_bytes),
            candidate_section_sha256=_sha256(b'{"format":"v3"}'),
            candidate_section_bytes=b'{"format":"v3"}',
        )


def test_try_link_emits_deterministic_blocker_instead_of_partial_packet() -> None:
    base = _base_payload()
    one = _receipt_and_replacement(
        base_payload=base,
        receipt_id="r1",
        section=V2Section.STORE,
        candidate_section=_store_blob(1.0),
        action_id="action-one",
    )
    attempt, linked = try_link_v2_applied_actions(
        base_payload=base + b"trailing",
        receipts=[one[0]],
        replacements=[one[1]],
    )
    assert linked is None
    assert attempt.status == "BLOCKED"
    assert attempt.input_receipt_ids == ("r1",)
    assert attempt.blockers[0].startswith("ValueError:trailing bytes after v2 packet")
    assert attempt.score_claim is False


def test_cli_writes_durable_blocker_for_invalid_manifest(tmp_path: Path) -> None:
    manifest_path = tmp_path / "invalid.json"
    manifest_path.write_text('{"actions": [], "actions": []}')
    output_dir = tmp_path / "out"

    returncode, payload = linker_cli.run(
        input_manifest=manifest_path,
        output_dir=output_dir,
    )

    assert returncode == 2
    assert payload["status"] == "BLOCKED"
    assert "duplicate input manifest key" in payload["blockers"][0]
    assert payload["input_manifest_sha256"] == _sha256(manifest_path.read_bytes())
    assert json.loads((output_dir / "link_attempt.json").read_text()) == payload
    assert not (output_dir / "0.bin").exists()
    assert not (output_dir / "archive.zip").exists()


def test_cli_writes_exact_linked_packet(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = _base_payload()
    receipt, replacement = _receipt_and_replacement(
        base_payload=base,
        receipt_id="r-cli",
        section=V2Section.STORE,
        candidate_section=_store_blob(5.0),
        action_id="action-cli",
    )
    base_path = tmp_path / "base.bin"
    candidate_path = tmp_path / "candidate-section.bin"
    base_path.write_bytes(base)
    candidate_path.write_bytes(replacement.candidate_section_bytes)
    manifest = {
        "schema": linker_cli.LINK_INPUT_SCHEMA,
        "base_payload_path": base_path.name,
        "actions": [
            {
                "receipt": receipt.as_dict(),
                "byte_home_id": replacement.byte_home_id,
                "section": replacement.section.value,
                "receiver_consumer": replacement.receiver_consumer,
                "base_section_sha256": replacement.base_section_sha256,
                "candidate_section_sha256": replacement.candidate_section_sha256,
                "candidate_section_path": candidate_path.name,
            }
        ],
    }
    manifest_path = tmp_path / "link.json"
    manifest_path.write_text(json.dumps(manifest))
    output_dir = tmp_path / "linked"

    write_order: list[str] = []
    original_atomic_write = linker_cli._atomic_write

    def _recording_atomic_write(path: Path, data: bytes) -> None:
        write_order.append(path.name)
        original_atomic_write(path, data)

    monkeypatch.setattr(linker_cli, "_atomic_write", _recording_atomic_write)
    returncode, payload = linker_cli.run(
        input_manifest=manifest_path,
        output_dir=output_dir,
    )

    assert returncode == 0
    assert payload["status"] == "LINKED"
    assert payload["input_manifest_sha256"] == _sha256(manifest_path.read_bytes())
    assert write_order == [
        "0.bin",
        "archive.zip",
        "inflate.py",
        "inflate.sh",
        "link_attempt.json",
    ]
    assert _sha256((output_dir / "0.bin").read_bytes()) == receipt.candidate_payload_sha256
    assert _sha256((output_dir / "archive.zip").read_bytes()) == receipt.candidate_archive_sha256
    assert (output_dir / "inflate.py").read_text() == generate_v2_inflate_py()
    assert (output_dir / "inflate.sh").read_text() == generate_v2_inflate_sh()
    assert {
        name: _sha256((output_dir / name).read_bytes())
        for name in ("inflate.py", "inflate.sh")
    } == v2_receiver_file_sha256s()
    assert (output_dir / "inflate.sh").stat().st_mode & 0o111
    assert V2PacketLinkAttempt.from_dict(
        json.loads((output_dir / "link_attempt.json").read_text())
    ).status == "LINKED"


def test_cli_hashes_and_parses_one_input_manifest_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = _base_payload()
    receipt, replacement = _receipt_and_replacement(
        base_payload=base,
        receipt_id="r-one-read",
        section=V2Section.STORE,
        candidate_section=_store_blob(6.0),
        action_id="action-one-read",
    )
    (tmp_path / "base.bin").write_bytes(base)
    (tmp_path / "candidate.bin").write_bytes(replacement.candidate_section_bytes)
    manifest_path = tmp_path / "link.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema": linker_cli.LINK_INPUT_SCHEMA,
                "base_payload_path": "base.bin",
                "actions": [
                    {
                        "receipt": receipt.as_dict(),
                        "byte_home_id": replacement.byte_home_id,
                        "section": replacement.section.value,
                        "receiver_consumer": replacement.receiver_consumer,
                        "base_section_sha256": replacement.base_section_sha256,
                        "candidate_section_sha256": replacement.candidate_section_sha256,
                        "candidate_section_path": "candidate.bin",
                    }
                ],
            }
        )
    )
    snapshot = manifest_path.read_bytes()
    monkeypatch.setattr(
        linker_cli,
        "_read_json_object",
        lambda _path: (_ for _ in ()).throw(AssertionError("manifest reopened")),
    )

    returncode, payload = linker_cli.run(
        input_manifest=manifest_path,
        output_dir=tmp_path / "out",
    )

    assert returncode == 0
    assert payload["input_manifest_sha256"] == _sha256(snapshot)


def test_cli_refuses_nonempty_output_without_overwriting(tmp_path: Path) -> None:
    output_dir = tmp_path / "existing"
    output_dir.mkdir()
    sentinel = output_dir / "operator-owned.bin"
    sentinel.write_bytes(b"preserve-me")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text("{}")

    returncode, payload = linker_cli.run(
        input_manifest=manifest_path,
        output_dir=output_dir,
    )

    assert returncode == 2
    assert payload["blockers"] == ["OUTPUT_DIR_NONEMPTY_REFUSED"]
    assert sentinel.read_bytes() == b"preserve-me"
    assert list(output_dir.iterdir()) == [sentinel]


def test_cli_never_mixes_bundle_when_destination_appears_during_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text("{}")
    output_dir = tmp_path / "contended"
    original_atomic_write = linker_cli._atomic_write
    injected = False

    def inject_competing_bundle(path: Path, data: bytes) -> None:
        nonlocal injected
        original_atomic_write(path, data)
        if not injected:
            injected = True
            output_dir.mkdir()
            (output_dir / "foreign-owner.bin").write_bytes(b"preserve-foreign-bundle")

    monkeypatch.setattr(linker_cli, "_atomic_write", inject_competing_bundle)
    returncode, payload = linker_cli.run(
        input_manifest=manifest_path,
        output_dir=output_dir,
    )

    assert returncode == 2
    assert payload["status"] == "BLOCKED"
    assert payload["blockers"][0].startswith("OUTPUT_BUNDLE_PUBLICATION_REFUSED")
    assert list(output_dir.iterdir()) == [output_dir / "foreign-owner.bin"]
    assert (output_dir / "foreign-owner.bin").read_bytes() == b"preserve-foreign-bundle"
    assert not (tmp_path / ".contended.publish.lock").exists()
    assert not tuple(tmp_path.glob(".contended.bundle.*"))


def test_cli_consumes_blocked_allocation_plan_as_typed_link_blocker(
    tmp_path: Path,
) -> None:
    plan = {
        "schema": "tac.seven_home_allocation_plan.v1",
        "status": "BLOCKED_NO_VALID_APPLIED_TRANSITION",
        "selected_identity": None,
        "research_only": True,
        "promotion_eligible": False,
        "score_claim": False,
    }
    plan["plan_content_sha256"] = linker_cli._canonical_json_sha256(
        plan,
        omit="plan_content_sha256",
    )
    plan_path = tmp_path / "allocation.json"
    plan_path.write_text(json.dumps(plan))
    manifest = {
        "schema": linker_cli.LINK_INPUT_SCHEMA,
        "allocation_plan_path": plan_path.name,
    }
    manifest_path = tmp_path / "link.json"
    manifest_path.write_text(json.dumps(manifest))
    output_dir = tmp_path / "blocked-link"

    returncode, payload = linker_cli.run(
        input_manifest=manifest_path,
        output_dir=output_dir,
    )

    assert returncode == 2
    assert payload["status"] == "BLOCKED"
    assert "object/grammar-incompatible" in payload["blockers"][0]
    assert json.loads((output_dir / "link_attempt.json").read_text()) == payload
    assert not (output_dir / "0.bin").exists()
