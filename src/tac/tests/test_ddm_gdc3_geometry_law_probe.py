from __future__ import annotations

import numpy as np

from experiments import ddm_gdc3_geometry_law_probe as probe


def test_horizontal_run_buckets_respect_thresholds_and_rows() -> None:
    mask = np.zeros((3, 24), dtype=bool)
    mask[0, 0] = True
    mask[0, 2:4] = True
    mask[0, 5:13] = True
    mask[1, 0:9] = True
    mask[1, 10:19] = True
    mask[2, 23] = True
    bucket, run_id, run_length = probe.horizontal_run_buckets(mask)
    assert bucket[0, 0] == 0
    assert np.all(bucket[0, 2:4] == 1)
    assert np.all(bucket[0, 5:13] == 1)
    assert np.all(bucket[1, 0:9] == 2)
    assert np.all(bucket[1, 10:19] == 2)
    assert bucket[2, 23] == 0
    assert run_length[0, 0] == 1
    assert np.all(run_length[0, 5:13] == 8)
    assert np.all(run_length[1, 0:9] == 9)
    assert len(np.unique(run_id[mask])) == 6
    assert np.all(bucket[~mask] == -1)


def test_two_sided_boundary_marks_both_transition_endpoints() -> None:
    labels = np.zeros((3, 4), dtype=np.uint8)
    labels[:, 2:] = 1
    labels[2, :] = 2
    boundary = probe.two_sided_boundary(labels)
    assert np.all(boundary[:2, 1:3])
    assert np.all(boundary[1:3, :])
    assert not boundary[0, 0]
    assert not boundary[0, 3]


def test_classify_plane_exclusively_covers_mismatches() -> None:
    target = np.zeros((7, 15), dtype=np.uint8)
    target[:, 7:] = 1
    generated = target.copy()
    generated[0, 0] = 2
    generated[1, 5:8] = 2
    generated[3, 1:12] = 2
    classified = probe.classify_plane(target, generated, run_id_offset=11)
    mismatch = classified["mismatch"]
    geometry = classified["geometry"][mismatch]
    assert np.count_nonzero(mismatch) == 15
    assert np.all((geometry >= 0) & (geometry < 6))
    assert sum(np.count_nonzero(geometry == code) for code in range(6)) == 15
    assert classified["geometry"][0, 0] == 1
    assert classified["geometry"][1, 5] == 2
    assert classified["geometry"][3, 1] == 5
    assert int(np.min(classified["run_id"][mismatch])) == 11


def test_residual_roundtrip_preserves_selected_set_for_both_orders() -> None:
    addresses = np.array([0, 64, probe.PLANE + 5, 3 * probe.PLANE - 1], dtype=np.int64)
    labels = np.array([0, 1, 4, 2], dtype=np.uint8)
    for order in probe.REQUIRED_ORDERS:
        payload = probe.serialize_residual(addresses, labels, order)
        decoded_addresses, decoded_labels, decoded_order = probe.parse_residual(payload)
        assert decoded_order == order
        permutation = probe.residual_sort_order(addresses, labels, order)
        assert np.array_equal(decoded_addresses, addresses[permutation])
        assert np.array_equal(decoded_labels, labels[permutation])
