from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

EXPERIMENTS = Path(__file__).resolve().parents[1]
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

import ddm_qbflow_packet as packet
import ddm_qbflow_rate_first_rung as builder

from tac.payload_retention_gate import check_no_measure_and_discard_payload


def _quantized_object(
    *, num_pairs: int = 3
) -> tuple[dict[str, np.ndarray], dict[str, object], dict[int, tuple[np.ndarray, np.ndarray]]]:
    params = packet.initialize_params(20260827)
    boundary, interior = packet.initialize_latents(20260827, num_pairs)
    model_raw = packet.encode_model(params)
    meta_raw, boundary_codes, interior_codes = packet.encode_latent_meta(boundary, interior)
    latent_raw = packet.encode_latent_table(range(num_pairs), boundary_codes, interior_codes)
    return (
        packet.decode_model(model_raw),
        packet.decode_latent_meta(meta_raw),
        packet.decode_latent_table(latent_raw),
    )


def test_real_receiver_branches_change_real_outputs() -> None:
    params, meta, latents = _quantized_object()
    boundary_codes, interior_codes = latents[1]
    boundary, interior = builder.dequantized_latent(meta, boundary_codes, interior_codes)
    baseline = packet.reference_forward(
        params,
        boundary,
        interior,
        pair_id=1,
        num_pairs=3,
        height=8,
        width=8,
    )
    assert baseline["signed_interfaces"].shape == (8, 8, 10)
    assert baseline["class_logits"].shape == (8, 8, 5)
    assert baseline["rgb_pair"].shape == (8, 8, 6)
    assert baseline["pose12"].shape == (12,)

    interface_params = {name: value.copy() for name, value in params.items()}
    interface_params["flow_head_b"][0] += np.float32(0.25)
    interface_changed = packet.reference_forward(
        interface_params,
        boundary,
        interior,
        pair_id=1,
        num_pairs=3,
        height=8,
        width=8,
    )
    assert not np.array_equal(baseline["signed_interfaces"], interface_changed["signed_interfaces"])
    assert not np.array_equal(baseline["class_logits"], interface_changed["class_logits"])

    renderer_params = {name: value.copy() for name, value in params.items()}
    renderer_params["render_out_b"][0] += np.float32(0.25)
    renderer_changed = packet.reference_forward(
        renderer_params,
        boundary,
        interior,
        pair_id=1,
        num_pairs=3,
        height=8,
        width=8,
    )
    assert not np.array_equal(baseline["rgb_pair"], renderer_changed["rgb_pair"])

    pose_params = {name: value.copy() for name, value in params.items()}
    pose_params["pose_out_b"][0] += np.float32(0.25)
    pose_changed = packet.reference_forward(
        pose_params,
        boundary,
        interior,
        pair_id=1,
        num_pairs=3,
        height=8,
        width=8,
    )
    assert not np.array_equal(baseline["pose12"], pose_changed["pose12"])


def test_packet_repeat_parseback_reset_records_and_mutation_refusal(tmp_path: Path) -> None:
    num_pairs = 3
    params = packet.initialize_params(20260827)
    boundary, interior = packet.initialize_latents(20260827, num_pairs)
    config_raw = packet.canonical_json_bytes(packet.architecture_config(num_pairs=num_pairs, seed=20260827))
    model_raw = packet.encode_model(params)
    meta_raw, boundary_codes, interior_codes = packet.encode_latent_meta(boundary, interior)
    latent_raw = packet.encode_latent_table(range(num_pairs), boundary_codes, interior_codes)
    raw_by_section = {
        packet.SECTION_CONFIG: config_raw,
        packet.SECTION_MODEL: model_raw,
        packet.SECTION_LATENT_META: meta_raw,
        packet.SECTION_LATENTS: latent_raw,
    }
    selected = []
    for section_id, raw in raw_by_section.items():
        candidates = packet.encode_section_candidates(section_id, raw)
        assert set(candidates) == set(packet.CODEC_IDS)
        selected.append(packet.choose_section(candidates))
    primary = packet.pack_packet(selected)
    repeat = packet.pack_packet(selected)
    assert primary == repeat
    retained = tmp_path / "packet.qbf"
    retained.write_bytes(primary)
    decoded = packet.decode_packet(retained.read_bytes())
    assert decoded.sections == raw_by_section

    archive = packet.deterministic_archive(primary)
    archive_repeat = packet.deterministic_archive(repeat)
    assert archive == archive_repeat
    assert packet.read_deterministic_archive(archive) == primary

    raw_record = packet.encode_latent_record(1, boundary_codes[1], interior_codes[1])
    reset_candidates = packet.encode_reset_record(raw_record)
    assert set(reset_candidates) == set(packet.CODEC_IDS)
    for candidate in reset_candidates.values():
        assert packet.decode_reset_record(candidate) == raw_record

    for section_id in raw_by_section:
        mutated = packet.mutate_counted_section(primary, section_id)
        (tmp_path / f"section_{section_id}.mutated.qbf").write_bytes(mutated)
        with pytest.raises(packet.QBFLOWPacketError):
            packet.decode_packet(mutated)


def test_seeded_stratified_selection_matches_frozen_lineage() -> None:
    field = builder.source_field()
    _strata, selected, total_interfaces = builder.selection_rows(field)
    assert [row["pair_id"] for row in selected] == builder.EXPECTED_SELECTED_IDS
    assert total_interfaces == builder.SOURCE_INTERFACE_COUNT


def test_qbflow_sources_pass_payload_retention_gate() -> None:
    findings = check_no_measure_and_discard_payload(
        repo_root=builder.REPO,
        strict=False,
        roots=(
            "experiments/ddm_qbflow_packet.py",
            "experiments/ddm_qbflow_rate_first_rung.py",
            "experiments/tests/test_ddm_qbflow_packet.py",
        ),
    )
    assert findings == []
