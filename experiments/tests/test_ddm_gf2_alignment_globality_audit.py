from __future__ import annotations

import hashlib

import numpy as np

from experiments import ddm_gf2_alignment_globality_audit as audit


def test_shifted_histograms_match_brute_force(monkeypatch) -> None:
    monkeypatch.setattr(audit, "N_PAIRS", 2)
    monkeypatch.setattr(audit, "HEIGHT", 5)
    monkeypatch.setattr(audit, "WIDTH", 6)
    monkeypatch.setattr(audit, "NUM_CLASSES", 3)
    monkeypatch.setattr(audit, "FIELD_SHAPE", (2, 5, 6))
    field = np.arange(60, dtype=np.uint8).reshape(2, 5, 6) % 3
    histograms, translations = audit.shifted_interior_histograms(field, radius=1)
    for pair in range(2):
        for shift, (dy, dx) in enumerate(translations.tolist()):
            crop = field[pair, 1 + dy : 4 + dy, 1 + dx : 5 + dx]
            assert histograms[pair, shift].tolist() == np.bincount(
                crop.ravel(), minlength=3
            ).tolist()


def test_minimum_histogram_tv_is_exact() -> None:
    left = np.asarray([[7, 2, 1], [6, 3, 1]], dtype=np.int32)
    right = np.asarray([[2, 6, 2], [4, 5, 1]], dtype=np.int32)
    brute = min(
        sum(abs(int(a) - int(b)) for a, b in zip(x, y, strict=True)) // 2
        for x in left.tolist()
        for y in right.tolist()
    )
    assert audit.minimum_histogram_tv(left, right) == brute == 2


def test_histogram_tv_never_exceeds_shared_template_errors() -> None:
    shared = np.asarray([0, 0, 1, 2, 2, 2], dtype=np.uint8)
    left = np.asarray([0, 1, 1, 2, 0, 2], dtype=np.uint8)
    right = np.asarray([1, 1, 1, 2, 2, 0], dtype=np.uint8)
    left_hist = np.bincount(left, minlength=3)[None, :]
    right_hist = np.bincount(right, minlength=3)[None, :]
    bound = audit.minimum_histogram_tv(left_hist, right_hist)
    observed_errors = int(np.count_nonzero(shared != left)) + int(
        np.count_nonzero(shared != right)
    )
    assert bound <= observed_errors


def test_choose_pairing_returns_a_disjoint_global_bound(monkeypatch) -> None:
    monkeypatch.setattr(audit, "N_PAIRS", 4)
    monkeypatch.setattr(audit, "NUM_CLASSES", 2)
    translations = np.asarray([[0, 0], [0, 1]], dtype=np.int16)
    histograms = np.asarray(
        [
            [[10, 0], [9, 1]],
            [[8, 2], [7, 3]],
            [[3, 7], [2, 8]],
            [[1, 9], [0, 10]],
        ],
        dtype=np.int32,
    )
    result = audit.choose_and_bound_pairing(histograms, translations)
    pairs = np.asarray(result["selected_pairs"])
    assert sorted(pairs.ravel().tolist()) == [0, 1, 2, 3]
    assert result["certified_global_mismatch_lower_bound"] == sum(
        result["per_pair_exact_histogram_tv_lower_bounds"]
    )


def test_tiny_end_to_end_retains_repeat_and_resumes(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(audit, "N_PAIRS", 4)
    monkeypatch.setattr(audit, "HEIGHT", 5)
    monkeypatch.setattr(audit, "WIDTH", 6)
    monkeypatch.setattr(audit, "NUM_CLASSES", 3)
    monkeypatch.setattr(audit, "FIELD_SHAPE", (4, 5, 6))
    monkeypatch.setattr(audit, "FIELD_BYTES", 120)
    monkeypatch.setattr(audit, "SEARCH_RADIUS", 1)
    monkeypatch.setattr(audit, "MINIMUM_FREE_BYTES", 1)
    output = tmp_path / "audit"
    monkeypatch.setattr(audit, "OUTPUT", output)
    field = np.asarray(
        [
            np.zeros((5, 6), dtype=np.uint8),
            np.ones((5, 6), dtype=np.uint8),
            np.full((5, 6), 2, dtype=np.uint8),
            np.tile(np.arange(6, dtype=np.uint8) % 3, (5, 1)),
        ]
    )
    field_path = tmp_path / "field.u8"
    field.tofile(field_path)
    monkeypatch.setattr(
        audit, "FIELD_SHA256", hashlib.sha256(field_path.read_bytes()).hexdigest()
    )

    result = audit.run(output, field_path, output)
    assert result["certified_global_mismatch_lower_bound"] >= 0
    assert result["histogram_payloads"]["byte_identical_repeat"] is True
    assert result["pairing_payloads"]["byte_identical_repeat"] is True
    assert (output / "MANIFEST.json").is_file()
    assert audit.run(output, field_path, output) == result
