# SPDX-License-Identifier: MIT
from __future__ import annotations

from tac.atom.contest_granularity import (
    ContestSignal,
    build_lattice_report,
    frame_and_pixel_atoms_from_xray_row,
    merge_atoms_by_id,
    pair_atom_from_component_row,
)


def test_pair_component_atoms_preserve_mode_variants_and_pair_overlap() -> None:
    rows = [
        {
            "pair": 7,
            "mode_id": "none",
            "family": "identity",
            "posenet_dist": 0.0001,
            "segnet_dist": 0.0002,
            "component_score_no_rate": 0.05,
        },
        {
            "pair": 7,
            "mode_id": "frame0_rgb_bias_p2_m1_m1",
            "family": "frame0_rgb_bias",
            "posenet_dist": 0.00008,
            "segnet_dist": 0.0002,
            "component_score_no_rate": 0.04,
        },
    ]
    atoms = [
        pair_atom_from_component_row(
            row,
            evidence_axis="[macOS-CPU advisory]",
            evidence_ref="fixture.jsonl",
            selected=True,
        )
        for row in rows
    ]
    atoms.extend(
        frame_and_pixel_atoms_from_xray_row(
            {
                "pair_idx": 7,
                "frame0_l1": 10.0,
                "frame1_l1": 11.0,
                "frame0_changed_fraction": 0.5,
                "frame1_changed_fraction": 0.25,
            },
            evidence_axis="[macOS-CPU advisory]",
            evidence_ref="xray.json",
        )
    )

    report = build_lattice_report(
        merge_atoms_by_id(atoms),
        source="fixture",
        generated_at_utc="2026-05-20T00:00:00+00:00",
    )

    atom_ids = {row["atom_id"] for row in report["atoms"]}
    assert "pair:7:mode:none" in atom_ids
    assert "pair:7:mode:frame0_rgb_bias_p2_m1_m1" in atom_ids
    overlap = report["pair_signal_overlap"]
    assert overlap["pair_count"] == 1
    assert overlap["top_pairs"][0]["venn_signature"] == (
        "pair_component&sidecar_selected&xray_pair&xray_pixel"
    )
    assert overlap["top_pairs"][0]["atom_count"] == 6


def test_pair_atom_selected_signal_is_optional() -> None:
    atom = pair_atom_from_component_row(
        {
            "pair": 1,
            "mode_id": "none",
            "family": "identity",
            "posenet_dist": 0.0001,
            "segnet_dist": 0.0002,
        },
        evidence_axis="[macOS-CPU advisory]",
        evidence_ref="fixture.jsonl",
    )
    assert atom.source_signals == (ContestSignal.PAIR_COMPONENT,)
    assert atom.venn_signature == "pair_component"
