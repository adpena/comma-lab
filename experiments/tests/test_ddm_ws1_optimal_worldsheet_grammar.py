from __future__ import annotations

import numpy as np

from experiments import ddm_ws1_optimal_worldsheet_grammar as ws1


def synthetic_partition() -> np.ndarray:
    labels = np.zeros((ws1.HEIGHT, ws1.WIDTH), dtype=np.uint8)
    labels[:80] = 2
    labels[80:, 40:80] = 1
    labels[120:190, 220:300] = 3
    labels[300:, 180:330] = 4
    return labels


def test_full_dual_curve_receiver_roundtrips_partition() -> None:
    labels = synthetic_partition()
    strata = ws1.frame_boundary_sites(labels)
    decoded = ws1.reconstruct_labels(int(labels[0, 0]), strata)
    assert np.array_equal(decoded, labels)


def test_connected_curves_are_disjoint_and_complete() -> None:
    sites = ws1.frame_boundary_sites(synthetic_partition())[ws1.EDGE_TO_ID[(0, 1)]]
    curves = ws1.decompose_sites(sites)
    rebuilt = np.sort(np.concatenate(curves))
    assert np.array_equal(rebuilt, sites)
    assert len({int(value) for curve in curves for value in curve}) == len(sites)


def test_lineage_and_residual_receiver_are_exact() -> None:
    first = ws1.frame_boundary_sites(synthetic_partition())[ws1.EDGE_TO_ID[(0, 3)]]
    shifted_labels = np.roll(synthetic_partition(), 2, axis=1)
    second = ws1.frame_boundary_sites(shifted_labels)[ws1.EDGE_TO_ID[(0, 3)]]
    prior = tuple(ws1.Curve(index, curve) for index, curve in enumerate(ws1.decompose_sites(first), 1))
    lineage, _ = ws1.match_lineage(ws1.decompose_sites(second), prior, None, len(prior) + 1)
    record = ws1.encode_lineage(lineage)
    entries = ws1.decode_lineage(record)
    prediction = ws1.predictor_from_lineage({curve.identity: curve.sites for curve in prior}, entries, None,
                                            use_identity=True)
    residual = ws1.xor_sites(second, prediction)
    assert np.array_equal(ws1.xor_sites(prediction, residual), second)


def test_horizon_model_is_receiver_readable() -> None:
    labels = np.zeros((ws1.HEIGHT, ws1.WIDTH), dtype=np.uint8)
    yy = np.indices(labels.shape)[0]
    horizon = 100 + np.arange(ws1.WIDTH)[None, :] // 16
    labels[yy >= horizon] = 2
    target = ws1.frame_boundary_sites(labels)[ws1.EDGE_TO_ID[(0, 2)]]
    record, prediction = ws1.horizon_model(target, 32)
    assert np.array_equal(prediction, ws1.decode_horizon(record))


def test_real_coder_envelope_parseback() -> None:
    records = {"fixture": (b"abc", b"abd", b"", b"abc")}
    race = ws1.race_records("fixture", records["fixture"])
    payload = ws1.build_envelope("fixture", "static", {"fixture": race})
    metadata, decoded = ws1.parse_envelope(payload)
    assert metadata["candidate"] == "fixture"
    assert decoded == records
