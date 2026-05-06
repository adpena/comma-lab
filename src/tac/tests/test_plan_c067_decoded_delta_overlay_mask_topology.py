from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments.plan_c067_decoded_delta_overlay_mask_topology import (
    DonorMaskInput,
    _pair_frame_mask,
    build_plan,
)


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _mask(path: Path, array: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array.astype(np.uint8, copy=False))
    return path


def _geometry_plan(path: Path, catastrophic_pairs: list[int]) -> Path:
    return _write_json(
        path,
        {
            "schema": "c067_geometry_safe_mask_topology_v2",
            "score_claim": False,
            "exact_negative_inputs": [
                {
                    "negative_id": "test_negative",
                    "family_group": "mask_topology_global_replacement",
                    "catastrophic_pair_indices": catastrophic_pairs,
                }
            ],
        },
    )


def _trust_plan(path: Path, *, pairs: list[int], classes: list[int] | None = None) -> Path:
    classes = [1] if classes is None else classes
    return _write_json(
        path,
        {
            "schema": "c067_postdecode_mask_repair_pair_class_waterfill_test",
            "score_claim": False,
            "budget_policies": [
                {
                    "policy_id": "budget4000",
                    "budget_payload_bytes": 4000,
                    "selected_atoms": [
                        {
                            "atom_id": f"pair{pair:04d}_class{classes[0]}",
                            "pair_indices": [pair],
                            "class_id": classes[0],
                            "changed_pixels": 1,
                        }
                        for pair in pairs
                    ],
                }
            ],
        },
    )


def _base() -> np.ndarray:
    array = np.zeros((4, 384, 512), dtype=np.uint8)
    array[0:2, 10:14, 20:30] = 1
    array[2:4, 30:34, 40:50] = 1
    return array


def test_emits_safe_local_payload_spec_inside_trust_region(tmp_path: Path) -> None:
    base = _base()
    donor = base.copy()
    donor[0:2, 10:12, 20:24] = 2
    base_path = _mask(tmp_path / "base.npy", base)
    donor_path = _mask(tmp_path / "donor.npy", donor)

    plan = build_plan(
        repo_root=tmp_path,
        base_mask_array=base_path,
        candidate_masks=[DonorMaskInput("donor", donor_path)],
        geometry_plan_json=_geometry_plan(tmp_path / "geometry.json", catastrophic_pairs=[1]),
        postdecode_trust_plan_json=_trust_plan(tmp_path / "trust.json", pairs=[0]),
        output_dir=tmp_path / "out",
    )

    assert plan["safe_spec_count"] == 1
    assert plan["dispatchable_candidate_count"] == 0
    candidate = plan["safe_specs"][0]
    assert candidate["safe_spec_emitted"] is True
    assert candidate["pixel_disagreement"]["selected_overlay_count"] == 16
    assert candidate["pixel_disagreement"]["outside_trust_region_rejected_count"] == 0
    assert candidate["byte_closure"]["archive_byte_closed"] is False
    spec_path = tmp_path / candidate["payload_spec"]["spec_path"]
    assert spec_path.exists()
    payload_header = candidate["payload_spec"]["payload_header"]
    assert payload_header["run_count"] > 0
    assert payload_header["reconstructed_mask_u8_sha256"] == candidate["pixel_disagreement"]["overlay_tensor_sha256"]


def test_auto_pair_basis_maps_c067_half_frame_rows_directly() -> None:
    mask = _pair_frame_mask(600, {216})

    assert np.flatnonzero(mask).tolist() == [216]


def test_half_frame_pair_basis_does_not_double_nonzero_pair_ids(tmp_path: Path) -> None:
    base = _base()
    donor = base.copy()
    donor[2, 10:12, 20:24] = 2
    donor[3, 10:12, 20:24] = 2
    base_path = _mask(tmp_path / "base.npy", base)
    donor_path = _mask(tmp_path / "donor.npy", donor)

    plan = build_plan(
        repo_root=tmp_path,
        base_mask_array=base_path,
        candidate_masks=[DonorMaskInput("donor", donor_path)],
        geometry_plan_json=_geometry_plan(tmp_path / "geometry.json", catastrophic_pairs=[]),
        postdecode_trust_plan_json=_trust_plan(
            tmp_path / "trust.json",
            pairs=[2],
            classes=[0],
        ),
        pair_index_basis="half_frame_pair_index",
        output_dir=tmp_path / "out",
    )

    candidate = plan["safe_specs"][0]
    assert candidate["trust_region"]["pair_index_basis"] == "half_frame_pair_index"
    assert candidate["trust_region"]["selected_pair_indices"] == [2]
    assert candidate["pixel_disagreement"]["selected_overlay_count"] == 8
    header = candidate["payload_spec"]["payload_header"]
    assert header["pair_index_basis"] == "half_frame_pair_index"
    assert header["selected_pair_indices"] == [2]


def test_vetoes_selected_catastrophic_pair(tmp_path: Path) -> None:
    base = _base()
    donor = base.copy()
    donor[0:2, 10:12, 20:24] = 2
    base_path = _mask(tmp_path / "base.npy", base)
    donor_path = _mask(tmp_path / "donor.npy", donor)

    plan = build_plan(
        repo_root=tmp_path,
        base_mask_array=base_path,
        candidate_masks=[DonorMaskInput("donor", donor_path)],
        geometry_plan_json=_geometry_plan(tmp_path / "geometry.json", catastrophic_pairs=[0]),
        postdecode_trust_plan_json=_trust_plan(tmp_path / "trust.json", pairs=[0]),
        output_dir=tmp_path / "out",
    )

    assert plan["safe_spec_count"] == 0
    blockers = plan["gated_candidates"][0]["safety_gate"]["blockers"]
    assert any("catastrophic exact-negative pairs" in blocker for blocker in blockers)
    assert not (tmp_path / "out" / "safe_specs").exists()


def test_can_scope_catastrophic_veto_by_family_group(tmp_path: Path) -> None:
    base = _base()
    donor = base.copy()
    donor[0:2, 10:12, 20:24] = 2
    base_path = _mask(tmp_path / "base.npy", base)
    donor_path = _mask(tmp_path / "donor.npy", donor)

    plan = build_plan(
        repo_root=tmp_path,
        base_mask_array=base_path,
        candidate_masks=[DonorMaskInput("donor", donor_path)],
        geometry_plan_json=_geometry_plan(tmp_path / "geometry.json", catastrophic_pairs=[0]),
        postdecode_trust_plan_json=_trust_plan(tmp_path / "trust.json", pairs=[0]),
        catastrophic_family_groups=["decoded_delta_overlay"],
        output_dir=tmp_path / "out",
    )

    assert plan["safe_spec_count"] == 1
    assert plan["trust_region"]["catastrophic_family_groups"] == ["decoded_delta_overlay"]
    source = plan["trust_region"]["catastrophic_pair_source"]
    assert source["pair_indices"] == []
    assert source["by_negative"][0]["included_in_veto"] is False


def test_rejects_outside_trust_region_pixels_without_applying_them(tmp_path: Path) -> None:
    base = _base()
    donor = base.copy()
    donor[0:2, 10:12, 20:24] = 2
    donor[2:4, 30:32, 40:44] = 3
    base_path = _mask(tmp_path / "base.npy", base)
    donor_path = _mask(tmp_path / "donor.npy", donor)

    plan = build_plan(
        repo_root=tmp_path,
        base_mask_array=base_path,
        candidate_masks=[DonorMaskInput("donor", donor_path)],
        geometry_plan_json=_geometry_plan(tmp_path / "geometry.json", catastrophic_pairs=[]),
        postdecode_trust_plan_json=_trust_plan(tmp_path / "trust.json", pairs=[0]),
        output_dir=tmp_path / "out",
    )

    candidate = plan["safe_specs"][0]
    assert candidate["pixel_disagreement"]["donor_vs_base_count"] == 32
    assert candidate["pixel_disagreement"]["selected_overlay_count"] == 16
    assert candidate["pixel_disagreement"]["outside_trust_region_rejected_count"] == 16
    assert candidate["trust_region"]["selected_pair_indices"] == [0]


def test_blocks_payload_that_exceeds_byte_cap(tmp_path: Path) -> None:
    base = _base()
    donor = base.copy()
    donor[0:2, 10:14, 20:30] = 2
    base_path = _mask(tmp_path / "base.npy", base)
    donor_path = _mask(tmp_path / "donor.npy", donor)

    plan = build_plan(
        repo_root=tmp_path,
        base_mask_array=base_path,
        candidate_masks=[DonorMaskInput("donor", donor_path)],
        geometry_plan_json=_geometry_plan(tmp_path / "geometry.json", catastrophic_pairs=[]),
        postdecode_trust_plan_json=_trust_plan(tmp_path / "trust.json", pairs=[0]),
        max_compressed_payload_bytes=1,
        output_dir=tmp_path / "out",
    )

    assert plan["safe_spec_count"] == 0
    blockers = plan["gated_candidates"][0]["safety_gate"]["blockers"]
    assert any("best compressed payload bytes" in blocker for blocker in blockers)
