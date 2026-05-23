from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from tac.optimization.decoder_q_selective_runtime_materializer import (
    build_selective_inflate_py,
    dqs1_payload_from_packet_plan,
    materialize_selective_runtime_candidate,
    parse_dqs1_payload,
    read_single_stored_member,
    sha256_bytes,
    write_json,
    write_single_stored_member,
)
from tac.optimization.decoder_q_selective_runtime_packet import (
    FALSE_AUTHORITY,
    pack_dqs1_payload,
)
from tac.optimization.decoder_q_selective_runtime_packet import (
    SCHEMA as PACKET_SCHEMA,
)
from tac.pr101_split_brotli_codec import DECODER_BLOB_LEN, LATENT_BLOB_LEN
from tac.repo_io import tree_sha256

FEC6_INFLATE = Path(
    "experiments/results/"
    "pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/"
    "submission_dir/inflate.py"
)
HFV1_INFLATE = Path(
    "experiments/results/"
    "pr110_provisional_hfv1_engineering_20260520_codex/"
    "runtime_hfv1/inflate.py"
)


def _packet_plan(
    *,
    pair_indices: list[int] | None = None,
    frame_policy: str = "pair_all_frames",
    pair_encoding: str = "raw_u16",
    base_archive: dict[str, object] | None = None,
) -> dict[str, object]:
    pairs = [3, 5] if pair_indices is None else pair_indices
    payload = pack_dqs1_payload(
        pair_indices=pairs,
        frame_policy=frame_policy,
        storage_index=26,
        q_offset=0,
        delta=1,
        pair_encoding=pair_encoding,
    )
    plan: dict[str, object] = {
        "schema": PACKET_SCHEMA,
        **FALSE_AUTHORITY,
        "candidate_generation_only": True,
        "evidence_grade": "macOS-MLX-research-signal",
        "selective_packet": {
            "frame_policy": frame_policy,
            "pair_encoding": pair_encoding,
            "payload_bytes": len(payload),
            "payload_sha256": sha256_bytes(payload),
            "selected_pair_indices": pairs,
        },
        "mutation": {
            "tensor_name": "rgb_1.weight",
            "q_offset": 0,
            "delta": 1,
            "tensor": {
                "storage_index": 26,
            },
        },
    }
    if base_archive is not None:
        plan["base_archive"] = base_archive
    return plan


def _synthetic_fec6_member(decoder_fill: bytes = b"d") -> bytes:
    decoder = decoder_fill * DECODER_BLOB_LEN
    latent = b"l" * LATENT_BLOB_LEN
    source_payload = decoder + latent
    selector_payload = b"FEC6"
    return (
        b"FP11"
        + len(source_payload).to_bytes(4, "little")
        + source_payload
        + len(selector_payload).to_bytes(2, "little")
        + selector_payload
    )


def _base_archive_meta(archive_path: Path, member_data: bytes) -> dict[str, object]:
    stored = read_single_stored_member(archive_path)
    return {
        "path": str(archive_path),
        **stored.as_dict(),
        "decoder_sha256": sha256_bytes(member_data[8 : 8 + DECODER_BLOB_LEN]),
    }


def test_dqs1_parsing_and_pair_all_frames_mapping() -> None:
    payload = dqs1_payload_from_packet_plan(_packet_plan(pair_indices=[3, 5]))

    parsed = parse_dqs1_payload(payload)

    assert parsed["frame_policy"] == "pair_all_frames"
    assert parsed["pair_encoding"] == "raw_u16"
    assert parsed["mode_byte"] == 1
    assert parsed["storage_index"] == 26
    assert parsed["q_offset"] == 0
    assert parsed["delta"] == 1
    assert parsed["pair_indices"] == [3, 5]
    assert parsed["affected_frame_indices"] == [6, 7, 10, 11]


def test_dqs1_parsing_and_segnet_last_frame_mapping() -> None:
    payload = dqs1_payload_from_packet_plan(
        _packet_plan(pair_indices=[3, 5], frame_policy="segnet_last_frame_only")
    )

    parsed = parse_dqs1_payload(payload)

    assert parsed["frame_policy"] == "segnet_last_frame_only"
    assert parsed["affected_frame_indices"] == [7, 11]


def test_dqs1_parsing_supports_sorted_gap_uleb_pair_encoding() -> None:
    payload = dqs1_payload_from_packet_plan(
        _packet_plan(pair_indices=[2, 5, 9], pair_encoding="sorted_gap_uleb")
    )

    parsed = parse_dqs1_payload(payload)

    assert len(payload) == 14
    assert parsed["frame_policy"] == "pair_all_frames"
    assert parsed["frame_policy_code"] == 1
    assert parsed["mode_byte"] == 0x11
    assert parsed["pair_encoding"] == "sorted_gap_uleb"
    assert parsed["pair_encoding_code"] == 1
    assert parsed["pair_indices"] == [2, 5, 9]
    assert parsed["affected_frame_indices"] == [4, 5, 10, 11, 18, 19]


def test_dqs1_parser_rejects_duplicate_pairs() -> None:
    payload = b"DQS1" + bytes([1, 26]) + (0).to_bytes(2, "little")
    payload += (1).to_bytes(1, "little", signed=True)
    payload += (2).to_bytes(2, "little")
    payload += (4).to_bytes(2, "little") + (4).to_bytes(2, "little")

    with pytest.raises(ValueError, match="duplicates"):
        parse_dqs1_payload(payload)


@pytest.mark.parametrize("inflate_path", [FEC6_INFLATE, HFV1_INFLATE])
def test_selective_inflate_patch_compiles_and_contains_runtime_hooks(
    inflate_path: Path,
) -> None:
    if not inflate_path.is_file():
        pytest.skip(f"runtime fixture not present in this checkout: {inflate_path}")
    base_text = inflate_path.read_text(encoding="utf-8")

    patched = build_selective_inflate_py(base_text)

    compile(patched, "inflate.py", "exec")
    assert "DQS1_MAGIC = b\"DQS1\"" in patched
    assert "IAS1_MAGIC = b\"IAS1\"" in patched
    assert 'DQS1_PAIR_ENCODING_BY_CODE = {0: "raw_u16", 1: "sorted_gap_uleb"}' in patched
    assert 'pair_encoding == "sorted_gap_uleb"' in patched
    assert "parse_dqs1_payload_prefix" in patched
    assert "parse_ias1_descriptor_payload" in patched
    assert "apply_dqs1_patch_to_decoder_state" in patched
    assert "selector_codes, selector_specs, dqs1_packet, ias1_descriptor" in patched
    assert "unexpected archive tail after FES1/DQS1 selector" in patched
    assert "mutated_decoder = None" in patched
    assert "selected_pairs: set[int] = set()" in patched
    assert "decoded[selected_local_tensor] = mutated_decoded[selected_local_tensor]" in patched
    assert (
        "decoded[selected_local_tensor, 1] = mutated_decoded[selected_local_tensor, 1]"
        in patched
    )


def test_materialize_appends_dqs1_inside_stored_member_and_keeps_authority_false(
    tmp_path: Path,
) -> None:
    base_dir = tmp_path / "base_submission"
    base_dir.mkdir()
    (base_dir / "inflate.py").write_text(FEC6_INFLATE.read_text(encoding="utf-8"), encoding="utf-8")
    base_member = _synthetic_fec6_member()
    base_archive = base_dir / "archive.zip"
    write_single_stored_member(base_archive, member_name="x", data=base_member)
    plan_path = tmp_path / "packet_plan.json"
    write_json(
        plan_path,
        _packet_plan(
            pair_indices=[9],
            base_archive=_base_archive_meta(base_archive, base_member),
        ),
    )

    out_dir = tmp_path / "candidate_submission"
    manifest = materialize_selective_runtime_candidate(
        plan_path=plan_path,
        base_submission_dir=base_dir,
        base_archive=base_archive,
        output_dir=out_dir,
        repo_root=tmp_path,
    )

    assert manifest["score_claim"] is False
    assert manifest["promotion_eligible"] is False
    assert manifest["ready_for_exact_eval_dispatch"] is False
    assert manifest["archive_member_delta"]["all_selective_bytes_inside_member_x"] is True
    assert manifest["archive_member_delta"]["external_sidecars_required_at_inflate_time"] is False

    with zipfile.ZipFile(out_dir / "archive.zip") as zf:
        infos = zf.infolist()
        assert [info.filename for info in infos] == ["x"]
        assert infos[0].compress_type == zipfile.ZIP_STORED
        materialized = zf.read("x")

    dqs1_payload = dqs1_payload_from_packet_plan(_packet_plan(pair_indices=[9]))
    assert materialized == base_member + dqs1_payload
    assert parse_dqs1_payload(materialized[len(base_member) :])["pair_indices"] == [9]
    patched_inflate = (out_dir / "inflate.py").read_text(encoding="utf-8")
    assert "DQS1_MAGIC = b\"DQS1\"" in patched_inflate
    assert (out_dir / "selective_runtime_manifest.json").is_file()
    assert (out_dir / "decoder_q_selective_runtime_packet_plan.json").is_file()


def test_materialize_rejects_packet_plan_base_archive_mismatch(tmp_path: Path) -> None:
    base_dir = tmp_path / "base_submission"
    base_dir.mkdir()
    (base_dir / "inflate.py").write_text(FEC6_INFLATE.read_text(encoding="utf-8"), encoding="utf-8")
    plan_member = _synthetic_fec6_member(b"a")
    actual_member = _synthetic_fec6_member(b"b")
    plan_archive = tmp_path / "planned.zip"
    actual_archive = base_dir / "archive.zip"
    write_single_stored_member(plan_archive, member_name="x", data=plan_member)
    write_single_stored_member(actual_archive, member_name="x", data=actual_member)
    plan_path = tmp_path / "packet_plan.json"
    write_json(
        plan_path,
        _packet_plan(
            pair_indices=[9],
            base_archive=_base_archive_meta(plan_archive, plan_member),
        ),
    )

    with pytest.raises(ValueError, match=r"base_archive\.path mismatch"):
        materialize_selective_runtime_candidate(
            plan_path=plan_path,
            base_submission_dir=base_dir,
            base_archive=actual_archive,
            output_dir=tmp_path / "candidate_submission",
            repo_root=tmp_path,
        )


def test_materialize_refuses_existing_output_dir_without_expected_tree(
    tmp_path: Path,
) -> None:
    base_dir = tmp_path / "base_submission"
    base_dir.mkdir()
    (base_dir / "inflate.py").write_text(FEC6_INFLATE.read_text(encoding="utf-8"), encoding="utf-8")
    base_member = _synthetic_fec6_member()
    base_archive = base_dir / "archive.zip"
    write_single_stored_member(base_archive, member_name="x", data=base_member)
    plan_path = tmp_path / "packet_plan.json"
    write_json(
        plan_path,
        _packet_plan(
            pair_indices=[9],
            base_archive=_base_archive_meta(base_archive, base_member),
        ),
    )
    out_dir = tmp_path / "candidate_submission"
    out_dir.mkdir()
    sentinel = out_dir / "sentinel.txt"
    sentinel.write_text("keep", encoding="utf-8")

    with pytest.raises(ValueError, match="expected_existing_tree_sha256"):
        materialize_selective_runtime_candidate(
            plan_path=plan_path,
            base_submission_dir=base_dir,
            base_archive=base_archive,
            output_dir=out_dir,
            repo_root=tmp_path,
            force=True,
        )

    assert sentinel.read_text(encoding="utf-8") == "keep"


def test_materialize_force_requires_matching_existing_output_tree(
    tmp_path: Path,
) -> None:
    base_dir = tmp_path / "base_submission"
    base_dir.mkdir()
    (base_dir / "inflate.py").write_text(FEC6_INFLATE.read_text(encoding="utf-8"), encoding="utf-8")
    base_member = _synthetic_fec6_member()
    base_archive = base_dir / "archive.zip"
    write_single_stored_member(base_archive, member_name="x", data=base_member)
    plan_path = tmp_path / "packet_plan.json"
    write_json(
        plan_path,
        _packet_plan(
            pair_indices=[9],
            base_archive=_base_archive_meta(base_archive, base_member),
        ),
    )
    out_dir = tmp_path / "candidate_submission"
    out_dir.mkdir()
    (out_dir / "sentinel.txt").write_text("replace me", encoding="utf-8")

    manifest = materialize_selective_runtime_candidate(
        plan_path=plan_path,
        base_submission_dir=base_dir,
        base_archive=base_archive,
        output_dir=out_dir,
        repo_root=tmp_path,
        force=True,
        expected_output_tree_sha256=tree_sha256(out_dir),
    )

    assert manifest["output_submission_dir"] == str(out_dir)
    assert not (out_dir / "sentinel.txt").exists()
    assert (out_dir / "selective_runtime_manifest.json").is_file()
