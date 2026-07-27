# SPDX-License-Identifier: MIT
"""Behavior tests for the G51 costate/admission firewall.

The generated n600 arrays are implementation fixtures only.  They are not
scientific evidence, score evidence, or an actionable project receipt.
"""

from __future__ import annotations

import hashlib
import json
import zipfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import tac.witness_control.taskspace_g51_costate_admission_guard_v1 as guard

REPO_ROOT = Path(__file__).resolve().parents[4]
G51_RECEIPT = (
    REPO_ROOT
    / ".omx/research/original_taskspace_inverse_witness_codec_20260725"
    / "g51_fulln600_adversarial_interpretation_receipt_20260727.json"
)
FALSE_AUTHORITY = {
    "research_only": True,
    "score_claim": False,
    "candidate_claim": False,
    "promotion_eligible": False,
    "ready_for_exact_eval_dispatch": False,
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_write(path: Path, value: object) -> None:
    path.write_bytes(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    )


def _file_ref(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _write_zip(path: Path, member: str, payload: bytes) -> None:
    info = zipfile.ZipInfo(member)
    info.date_time = (1980, 1, 1, 0, 0, 0)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o100644 << 16
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(info, payload)


def _effect_array_specs() -> dict[str, Any]:
    vector = {
        "dtype": "float32",
        "shape": [guard.PUBLIC_PAIR_COUNT, 2],
        "axes": list(guard.EFFECT_AXES),
    }
    return {
        "pair_ids": {
            "dtype": "int32",
            "shape": [guard.PUBLIC_PAIR_COUNT],
            "semantic": "ordered_public_pair_ids_0_through_599",
        },
        "scorer_term_effect_vectors": dict(vector),
        "realized_scorer_jvp": dict(vector),
        "realized_scorer_vjp": dict(vector),
    }


def _build_evidence(
    root: Path,
    *,
    bad_pair_ids: bool = False,
    identical_archives: bool = False,
    unrelated_derivatives: bool = False,
) -> dict[str, Path]:
    baseline_archive = root / "baseline.zip"
    candidate_archive = root / "candidate.zip"
    _write_zip(baseline_archive, "payload.bin", b"baseline" * 11)
    if identical_archives:
        candidate_archive.write_bytes(baseline_archive.read_bytes())
    else:
        _write_zip(candidate_archive, "payload.bin", b"candidate" * 137)

    receiver_runtime = root / "inflate.py"
    evaluator_source = root / "evaluate.py"
    receiver_runtime.write_text("def inflate():\n    return 'fixture-only'\n")
    evaluator_source.write_text("def evaluate():\n    return 'fixture-only'\n")

    pair_ids = np.arange(guard.PUBLIC_PAIR_COUNT, dtype=np.int32)
    if bad_pair_ids:
        pair_ids[-1] = pair_ids[-2]
    effects = np.ones((guard.PUBLIC_PAIR_COUNT, 2), dtype=np.float32)
    effects[:, 1] = 0.25
    effects[0] = np.asarray([16.0, 8.0], dtype=np.float32)
    if unrelated_derivatives:
        jvp = np.flip(effects, axis=0).copy() * np.float32(-3.0)
        vjp = np.roll(effects, 127, axis=0).copy() * np.float32(7.0)
    else:
        jvp = effects * np.float32(0.75)
        vjp = effects * np.float32(1.25)
    effect_bundle = root / "effects.npz"
    np.savez_compressed(
        effect_bundle,
        pair_ids=pair_ids,
        scorer_term_effect_vectors=effects,
        realized_scorer_jvp=jvp,
        realized_scorer_vjp=vjp,
    )

    baseline_output_sha = hashlib.sha256(b"baseline-output").hexdigest()
    candidate_output_sha = hashlib.sha256(b"candidate-output").hexdigest()
    archive_receipt = root / "archive_delta.json"
    archive_value = {
        "schema": guard.SAME_OBJECT_ARCHIVE_DELTA_SCHEMA,
        "object_id": "fixture_same_object",
        "transition_id": "fixture_transition_001",
        "population_pairs": guard.PUBLIC_PAIR_COUNT,
        "same_object_lifecycle": True,
        "outer_zip_delta_measured": True,
        "baseline_archive": _file_ref(baseline_archive),
        "candidate_archive": _file_ref(candidate_archive),
        "measured_zip_delta_bytes": (candidate_archive.stat().st_size - baseline_archive.stat().st_size),
        "receiver_runtime": _file_ref(receiver_runtime),
        "public_evaluator_source": _file_ref(evaluator_source),
        "population_identity_sha256": hashlib.sha256(b"ordered-public-pairs-0-through-599").hexdigest(),
        "baseline_receiver_output_sha256": baseline_output_sha,
        "candidate_receiver_output_sha256": candidate_output_sha,
        "effect_bundle_sha256": _sha256(effect_bundle),
        "truth": dict(FALSE_AUTHORITY),
    }
    _canonical_write(archive_receipt, archive_value)

    effect_receipt = root / "effect_receipt.json"
    effect_value = {
        "schema": guard.UNTRUSTED_SCORER_EFFECT_RECEIPT_SCHEMA,
        "encoder_diagnostic_sha256": guard.G51_ADVERSARIAL_RECEIPT_SHA256,
        "object_id": "fixture_same_object",
        "transition_id": "fixture_transition_001",
        "population_pairs": guard.PUBLIC_PAIR_COUNT,
        "baseline_archive_sha256": _sha256(baseline_archive),
        "candidate_archive_sha256": _sha256(candidate_archive),
        "baseline_receiver_output_sha256": baseline_output_sha,
        "candidate_receiver_output_sha256": candidate_output_sha,
        "archive_delta_receipt_sha256": _sha256(archive_receipt),
        "effect_bundle": _file_ref(effect_bundle),
        "effect_bundle_format": guard.EFFECT_BUNDLE_FORMAT,
        "effect_arrays": _effect_array_specs(),
        "measurement_claims": {
            "public_receiver_invoked": True,
            "realized_through_R": True,
            "frozen_cpu_torch_segnet_invoked": True,
            "frozen_cpu_torch_posenet_invoked": True,
            "scorer_effect_vectors_measured": True,
            "realized_scorer_jvp_measured": True,
            "realized_scorer_vjp_measured": True,
            "numpy_fp32_reference": True,
            "mps_authority": False,
            "proxy": False,
        },
        "truth": dict(FALSE_AUTHORITY),
    }
    _canonical_write(effect_receipt, effect_value)
    return {
        "baseline_archive": baseline_archive,
        "candidate_archive": candidate_archive,
        "effect_bundle": effect_bundle,
        "archive_receipt": archive_receipt,
        "effect_receipt": effect_receipt,
    }


def _rebind_archive_receipt(
    paths: dict[str, Path],
    mutate: Any,
) -> None:
    archive_value = json.loads(paths["archive_receipt"].read_bytes())
    mutate(archive_value)
    _canonical_write(paths["archive_receipt"], archive_value)
    effect_value = json.loads(paths["effect_receipt"].read_bytes())
    effect_value["archive_delta_receipt_sha256"] = _sha256(paths["archive_receipt"])
    _canonical_write(paths["effect_receipt"], effect_value)


def test_exact_g51_receipt_is_quarantined_to_diagnostic_identity() -> None:
    diagnostic = guard.load_g51_encoder_diagnostic(G51_RECEIPT)

    assert diagnostic.receipt.sha256 == guard.G51_ADVERSARIAL_RECEIPT_SHA256
    assert diagnostic.source_commit == guard.G51_ADVERSARIAL_SOURCE_COMMIT
    assert diagnostic.pair_count == 600
    assert diagnostic.allowed_use == "ENCODER_DIAGNOSTIC_ONLY"
    assert diagnostic.actionable_costate_input is False
    assert diagnostic.actionable_consumers == ()
    assert not hasattr(diagnostic, "payload")
    assert not hasattr(diagnostic, "per_pair_marginals")


@pytest.mark.parametrize(
    "legacy_fragment",
    [
        {"zlib9_marginal_bytes": {"pair_0": 12}},
        {"per_pair_marginals": [{"pair_id": 0}]},
        {"fits_current_batch16_headroom_to_effective_frontier": True},
        {"best_archive_claim": {"bytes": 1}},
        {"status": "READY_ENCODER_DIAGNOSTIC_ONLY"},
        {"ambient_unweighted_group_gram": [[1, 0], [0, 1]]},
    ],
)
def test_historical_v1_fields_are_rejected_before_admission(
    tmp_path: Path,
    legacy_fragment: dict[str, Any],
) -> None:
    paths = _build_evidence(tmp_path)
    effect_value = json.loads(paths["effect_receipt"].read_bytes())
    effect_value["legacy_payload"] = legacy_fragment
    _canonical_write(paths["effect_receipt"], effect_value)

    with pytest.raises(
        guard.G51CostateAdmissionError,
        match=r"forbidden historical G51|declarative READY",
    ):
        guard.admit_g51_actionable_costate_evidence(
            diagnostic_receipt_path=G51_RECEIPT,
            scorer_effect_receipt_path=paths["effect_receipt"],
            archive_delta_receipt_path=paths["archive_receipt"],
        )


def test_self_attested_fixture_is_integrity_diagnostic_not_actionable(
    tmp_path: Path,
) -> None:
    paths = _build_evidence(tmp_path)

    candidate = guard.inspect_g51_costate_evidence_candidate(
        diagnostic_receipt_path=G51_RECEIPT,
        scorer_effect_receipt_path=paths["effect_receipt"],
        archive_delta_receipt_path=paths["archive_receipt"],
    )

    assert candidate.pair_count == 600
    assert candidate.integrity_checks_passed is True
    assert candidate.claimed_measurement_not_proven is True
    assert candidate.canonical_materializer_bound is False
    assert candidate.actionable_costate_input is False
    assert candidate.actionable_consumers == ()
    assert candidate.blocked_consumers == guard.BLOCKED_CONSUMERS
    assert not hasattr(candidate, "scorer_effect_magnitude_per_pair")
    assert candidate.serialized_archive_delta_contract["schema"] == ("serialized_archive_delta_contract.v1")
    assert candidate.score_claim is False
    assert candidate.ready_for_exact_eval_dispatch is False

    with pytest.raises(
        guard.G51CostateAdmissionError,
        match=guard.CANONICAL_MATERIALIZER_BLOCKER,
    ):
        guard.admit_g51_actionable_costate_evidence(
            diagnostic_receipt_path=G51_RECEIPT,
            scorer_effect_receipt_path=paths["effect_receipt"],
            archive_delta_receipt_path=paths["archive_receipt"],
        )


def test_every_actionable_consumer_is_explicitly_refused(
    tmp_path: Path,
) -> None:
    paths = _build_evidence(tmp_path)
    candidate = guard.inspect_g51_costate_evidence_candidate(
        diagnostic_receipt_path=G51_RECEIPT,
        scorer_effect_receipt_path=paths["effect_receipt"],
        archive_delta_receipt_path=paths["archive_receipt"],
    )

    serialized = json.dumps(asdict(candidate), sort_keys=True)
    for forbidden in (
        "zlib9_marginal_bytes",
        "per_pair_marginals",
        "fits_",
        "best_archive",
        "ambient_unweighted_gram",
        "READY",
    ):
        assert forbidden not in serialized
    for consumer_id in guard.BLOCKED_CONSUMERS:
        with pytest.raises(
            guard.G51CostateAdmissionError,
            match=guard.CANONICAL_MATERIALIZER_BLOCKER,
        ):
            candidate.refuse_consumer(consumer_id)


def test_diagnostic_receipt_cannot_be_used_as_effect_receipt(
    tmp_path: Path,
) -> None:
    paths = _build_evidence(tmp_path)
    with pytest.raises(guard.G51CostateAdmissionError):
        guard.request_g51_costate_bit_allocation(
            diagnostic_receipt_path=G51_RECEIPT,
            scorer_effect_receipt_path=G51_RECEIPT,
            archive_delta_receipt_path=paths["archive_receipt"],
        )


def test_unrelated_nonzero_jvp_vjp_cannot_unlock_allocator(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _build_evidence(tmp_path, unrelated_derivatives=True)
    called = False

    def _forbidden_allocator(*args: Any, **kwargs: Any) -> None:
        nonlocal called
        called = True
        raise AssertionError("canonical allocator must not be called")

    import tac.bit_allocator.per_pair_difficulty_weighted as allocator

    monkeypatch.setattr(allocator, "allocate_bits_per_pair", _forbidden_allocator)
    candidate = guard.inspect_g51_costate_evidence_candidate(
        diagnostic_receipt_path=G51_RECEIPT,
        scorer_effect_receipt_path=paths["effect_receipt"],
        archive_delta_receipt_path=paths["archive_receipt"],
    )
    assert candidate.claimed_measurement_not_proven is True
    assert candidate.actionable_costate_input is False
    with pytest.raises(
        guard.G51CostateAdmissionError,
        match=guard.CANONICAL_MATERIALIZER_BLOCKER,
    ):
        guard.request_g51_costate_bit_allocation(
            diagnostic_receipt_path=G51_RECEIPT,
            scorer_effect_receipt_path=paths["effect_receipt"],
            archive_delta_receipt_path=paths["archive_receipt"],
        )
    assert called is False


def test_effect_bundle_byte_tamper_is_rejected(tmp_path: Path) -> None:
    paths = _build_evidence(tmp_path)
    with paths["effect_bundle"].open("ab") as stream:
        stream.write(b"tamper")

    with pytest.raises(
        guard.G51CostateAdmissionError,
        match=r"effect\.effect_bundle byte count differs",
    ):
        guard.admit_g51_actionable_costate_evidence(
            diagnostic_receipt_path=G51_RECEIPT,
            scorer_effect_receipt_path=paths["effect_receipt"],
            archive_delta_receipt_path=paths["archive_receipt"],
        )


def test_same_object_zip_delta_is_recomputed_from_reopened_files(
    tmp_path: Path,
) -> None:
    paths = _build_evidence(tmp_path)

    def _mutate(value: dict[str, Any]) -> None:
        value["measured_zip_delta_bytes"] += 1

    _rebind_archive_receipt(paths, _mutate)
    with pytest.raises(
        guard.G51CostateAdmissionError,
        match="does not match reopened files",
    ):
        guard.admit_g51_actionable_costate_evidence(
            diagnostic_receipt_path=G51_RECEIPT,
            scorer_effect_receipt_path=paths["effect_receipt"],
            archive_delta_receipt_path=paths["archive_receipt"],
        )


def test_full_ordered_n600_effect_population_is_required(
    tmp_path: Path,
) -> None:
    paths = _build_evidence(tmp_path, bad_pair_ids=True)

    with pytest.raises(
        guard.G51CostateAdmissionError,
        match=r"ordered public pair ids 0\.\.599",
    ):
        guard.admit_g51_actionable_costate_evidence(
            diagnostic_receipt_path=G51_RECEIPT,
            scorer_effect_receipt_path=paths["effect_receipt"],
            archive_delta_receipt_path=paths["archive_receipt"],
        )


def test_identical_baseline_and_candidate_archive_is_not_a_transition(
    tmp_path: Path,
) -> None:
    paths = _build_evidence(tmp_path, identical_archives=True)

    with pytest.raises(
        guard.G51CostateAdmissionError,
        match="ZIP identities must differ",
    ):
        guard.admit_g51_actionable_costate_evidence(
            diagnostic_receipt_path=G51_RECEIPT,
            scorer_effect_receipt_path=paths["effect_receipt"],
            archive_delta_receipt_path=paths["archive_receipt"],
        )


def test_g51_diagnostic_byte_drift_is_rejected(tmp_path: Path) -> None:
    copied = tmp_path / "g51.json"
    copied.write_bytes(G51_RECEIPT.read_bytes() + b"\n")

    with pytest.raises(
        guard.G51CostateAdmissionError,
        match="not the exact b84 interpretation",
    ):
        guard.load_g51_encoder_diagnostic(copied)
