# SPDX-License-Identifier: MIT
from __future__ import annotations

import io
import json
import zipfile
from dataclasses import replace

import numpy as np
import pytest

from tac.witness_control.trajectory_transaction_v2 import (
    ATOMIC_OWNERS,
    BEST_STAGE_DOMAIN,
    CANONICAL_DOMAIN_COVERAGE,
    CAUSAL_SELECTION_STATE,
    CURRENT_TRAIN_STATE,
    FRESH_LINEAGE_DOMAIN,
    LINEAGE_ENVELOPE,
    MANIFEST_KEY,
    RESTORABLE_STATE_OWNERS,
    ROLLBACK_SAVEPOINT,
    SCHEDULE_CONTROL_STATE,
    SEMANTIC_DOMAINS,
    VERDICT_TRANSACTION,
    BarrierStateBinding,
    DomainCoverage,
    EntryDescriptor,
    EntrySpec,
    ExpectedOwnerSchema,
    ExpectedTransactionSchema,
    OwnerActivity,
    OwnerClaim,
    ParallelHistorySpec,
    QuiescentBarrierState,
    TransactionManifest,
    TransactionValidationError,
    build_manifest,
    canonical_domain_coverage,
    load_npz_staging,
    manifest_array,
    manifest_from_array,
    require_matching_topology,
    stage_arrays,
    verify_canonical_reserialization,
)
from tac.witness_control.trajectory_transaction_v2 import (
    validate_transaction as _validate_transaction,
)

_BARRIER_PREFIX = "vtx."
_BARRIER_KEYS = (
    "vtx.last_applied_result_id",
    "vtx.last_applied_result_sha256",
    "vtx.next_apply_seq",
    "vtx.next_submit_seq",
    "vtx.pending_count",
    "vtx.schema",
)
_OWNER_KEYS = {
    CURRENT_TRAIN_STATE: ("model",),
    ROLLBACK_SAVEPOINT: ("rollback_model",),
    SCHEDULE_CONTROL_STATE: ("rng_state",),
    VERDICT_TRANSACTION: (
        *_BARRIER_KEYS,
        "verdict_loss",
        "verdict_next",
        "verdict_seq",
    ),
    CAUSAL_SELECTION_STATE: ("best_content_sha",),
    LINEAGE_ENVELOPE: ("seed",),
}
_DERIVED_KEYS = ("complete_state_sha",)
_DEFAULT_BARRIER = object()


def _arrays() -> dict[str, np.ndarray]:
    def utf8(value: str) -> np.ndarray:
        return np.frombuffer(value.encode("utf-8"), dtype=np.uint8).copy()

    return {
        "model": np.arange(4, dtype="<f4").reshape(2, 2),
        "rollback_model": np.arange(4, dtype="<f4").reshape(2, 2),
        "rng_state": np.asarray([2, 3, 5, 7], dtype="<u4"),
        "verdict_loss": np.asarray([0.5, 0.25, 0.125], dtype="<f4"),
        "verdict_next": np.asarray(3, dtype="<u8"),
        "verdict_seq": np.asarray([0, 1, 2], dtype="<u8"),
        "vtx.schema": utf8("tac.g111_verdict_barrier.v1"),
        "vtx.next_submit_seq": np.asarray([3], dtype=np.int64),
        "vtx.next_apply_seq": np.asarray([3], dtype=np.int64),
        "vtx.pending_count": np.asarray([0], dtype=np.int64),
        "vtx.last_applied_result_id": utf8("result-2"),
        "vtx.last_applied_result_sha256": utf8("c" * 64),
        "best_content_sha": np.asarray(b"a" * 64, dtype="S64"),
        "seed": np.asarray(11, dtype="<u8"),
        "complete_state_sha": np.asarray(b"b" * 64, dtype="S64"),
    }


def _entry_spec(key: str, array: np.ndarray) -> EntrySpec:
    return EntrySpec(key, array.dtype.str, tuple(array.shape))


def _expected(
    arrays: dict[str, np.ndarray] | None = None,
    *,
    owner_active: dict[str, bool] | None = None,
) -> ExpectedTransactionSchema:
    values = _arrays() if arrays is None else arrays
    activity = dict.fromkeys(ATOMIC_OWNERS, True)
    if owner_active:
        activity.update(owner_active)
    baseline = _arrays()
    owners = []
    for owner in ATOMIC_OWNERS:
        specs = tuple(
            sorted(
                (_entry_spec(key, values.get(key, baseline[key])) for key in _OWNER_KEYS[owner]),
                key=lambda spec: spec.key,
            )
        )
        owners.append(
            ExpectedOwnerSchema(
                owner=owner,
                active=activity[owner],
                required=specs if activity[owner] else (),
                permitted=specs,
            )
        )
    return ExpectedTransactionSchema(
        owners=tuple(owners),
        domain_coverage=canonical_domain_coverage(),
        derived_lineage=tuple(_entry_spec(key, values[key]) for key in _DERIVED_KEYS),
        histories=(
            (
                ParallelHistorySpec(
                    name="verdict_journal",
                    keys=("verdict_seq", "verdict_loss"),
                    sequence_key="verdict_seq",
                    max_length=3,
                    next_sequence_key="verdict_next",
                    require_contiguous=True,
                    allow_empty=False,
                ),
            )
            if activity[VERDICT_TRANSACTION]
            else ()
        ),
    )


def _manifest(
    arrays: dict[str, np.ndarray] | None = None,
    *,
    claims: dict[str, tuple[str, ...]] | None = None,
    activity: dict[str, bool] | None = None,
):
    values = _arrays() if arrays is None else arrays
    owner_claims = dict(_OWNER_KEYS if claims is None else claims)
    active = dict.fromkeys(ATOMIC_OWNERS, True)
    if activity:
        active.update(activity)
    return build_manifest(
        values,
        owner_claims=owner_claims,
        activity=active,
        domain_coverage=CANONICAL_DOMAIN_COVERAGE,
        derived_lineage_keys=_DERIVED_KEYS,
    )


def _barrier() -> QuiescentBarrierState:
    return QuiescentBarrierState(
        next_submit_seq=3,
        next_apply_seq=3,
        pending_count=0,
        last_applied_result_id="result-2",
        last_applied_result_sha256="c" * 64,
    )


def _barrier_binding() -> BarrierStateBinding:
    return BarrierStateBinding.from_prefix(_BARRIER_PREFIX)


def validate_transaction(
    arrays,
    manifest,
    expected,
    *,
    barrier_binding=_DEFAULT_BARRIER,
    expected_barrier_state=_DEFAULT_BARRIER,
    invariant_validators=(),
):
    """Test helper that makes active O4 barrier supply explicit by default."""

    if barrier_binding is _DEFAULT_BARRIER:
        barrier_binding = _barrier_binding()
    if expected_barrier_state is _DEFAULT_BARRIER:
        expected_barrier_state = _barrier()
    return _validate_transaction(
        arrays,
        manifest,
        expected,
        barrier_binding=barrier_binding,
        expected_barrier_state=expected_barrier_state,
        invariant_validators=invariant_validators,
    )


def test_valid_transaction_stages_private_read_only_arrays_and_six_owner_hashes():
    arrays = _arrays()
    manifest = _manifest(arrays)
    staged = validate_transaction(arrays, manifest, _expected(arrays))

    arrays["model"][0, 0] = np.float32(999.0)
    assert staged.arrays["model"][0, 0] == np.float32(0.0)
    assert not staged.arrays["model"].flags.writeable
    assert not staged.arrays["model"].flags.owndata
    with pytest.raises(ValueError, match="cannot set WRITEABLE flag"):
        staged.arrays["model"].setflags(write=True)
    assert tuple(staged.owner_semantic_hashes) == ATOMIC_OWNERS
    assert len(staged.semantic_hash) == 64


def test_manifest_roundtrip_is_canonical_pickle_free_uint8():
    manifest = _manifest()
    encoded = manifest_array(manifest)
    assert encoded.dtype == np.uint8
    assert encoded.ndim == 1
    assert manifest_from_array(encoded).to_json() == manifest.to_json()
    arrays = _arrays()
    arrays[MANIFEST_KEY] = encoded
    validate_transaction(arrays, manifest, _expected())


@pytest.mark.parametrize(
    ("owner", "key"),
    [(owner, keys[0]) for owner, keys in _OWNER_KEYS.items()],
)
def test_delete_one_required_key_fails_for_every_active_owner(owner: str, key: str):
    del owner  # parameter documents the owner covered by this deletion
    arrays = _arrays()
    manifest = _manifest(arrays)
    del arrays[key]
    with pytest.raises(TransactionValidationError, match="descriptor coverage"):
        validate_transaction(arrays, manifest, _expected())


@pytest.mark.parametrize("key", ["totally_unowned", "__o1_model_prefix_impostor"])
def test_unowned_and_fake_prefix_keys_fail_exact_reverse_coverage(key: str):
    arrays = _arrays()
    manifest = _manifest(arrays)
    arrays[key] = np.asarray(1, dtype="<u8")
    with pytest.raises(TransactionValidationError, match="descriptor coverage"):
        validate_transaction(arrays, manifest, _expected())


def test_duplicate_owner_claim_fails_even_when_prefixes_look_valid():
    manifest = _manifest()
    claims = list(manifest.owner_claims)
    claims[1] = OwnerClaim(
        ROLLBACK_SAVEPOINT,
        tuple(sorted((*claims[1].keys, "model"))),
    )
    corrupt = replace(manifest, owner_claims=tuple(claims))
    with pytest.raises(TransactionValidationError, match="multiply owned"):
        validate_transaction(_arrays(), corrupt, _expected())


def test_unknown_or_missing_owner_and_unknown_domain_fail():
    manifest = _manifest()
    claims = list(manifest.owner_claims)
    claims[-1] = OwnerClaim("not_an_atomic_owner", claims[-1].keys)
    with pytest.raises(TransactionValidationError, match="owner claims"):
        validate_transaction(
            _arrays(),
            replace(manifest, owner_claims=tuple(claims)),
            _expected(),
        )

    domains = list(manifest.domain_coverage)
    domains[-1] = DomainCoverage("not_a_semantic_domain", domains[-1].owners)
    with pytest.raises(TransactionValidationError, match="domain coverage"):
        validate_transaction(
            _arrays(),
            replace(manifest, domain_coverage=tuple(domains)),
            _expected(),
        )


def test_activity_drift_fails_in_both_directions():
    manifest = _manifest()
    activity = list(manifest.activity)
    activity[0] = OwnerActivity(CURRENT_TRAIN_STATE, False)
    with pytest.raises(TransactionValidationError, match="activity drift"):
        validate_transaction(
            _arrays(),
            replace(manifest, activity=tuple(activity)),
            _expected(),
        )

    expected = _expected(owner_active={CURRENT_TRAIN_STATE: False})
    with pytest.raises(TransactionValidationError, match="activity drift"):
        validate_transaction(_arrays(), manifest, expected)


def test_inactive_owner_cannot_claim_keys():
    manifest = _manifest(activity={CURRENT_TRAIN_STATE: False})
    expected = _expected(owner_active={CURRENT_TRAIN_STATE: False})
    with pytest.raises(TransactionValidationError, match="inactive owner"):
        validate_transaction(_arrays(), manifest, expected)


def test_active_verdict_owner_requires_explicit_quiescent_barrier():
    with pytest.raises(TransactionValidationError, match="BarrierStateBinding"):
        _validate_transaction(
            _arrays(),
            _manifest(),
            _expected(),
            expected_barrier_state=_barrier(),
        )


def test_inactive_verdict_owner_has_zero_state_and_forbids_barrier():
    arrays = _arrays()
    for key in _OWNER_KEYS[VERDICT_TRANSACTION]:
        del arrays[key]
    claims = dict(_OWNER_KEYS)
    claims[VERDICT_TRANSACTION] = ()
    manifest = _manifest(
        arrays,
        claims=claims,
        activity={VERDICT_TRANSACTION: False},
    )
    expected = _expected(
        arrays,
        owner_active={VERDICT_TRANSACTION: False},
    )
    staged = _validate_transaction(arrays, manifest, expected)
    activity = {row.owner: row.active for row in staged.manifest.activity}
    assert not activity[VERDICT_TRANSACTION]

    with pytest.raises(TransactionValidationError, match="forbids barrier binding"):
        _validate_transaction(
            arrays,
            manifest,
            expected,
            barrier_binding=BarrierStateBinding.from_prefix(_BARRIER_PREFIX),
            expected_barrier_state=QuiescentBarrierState(0, 0, 0),
        )


def test_inactive_expected_owner_forbids_required_keys_and_o4_histories():
    model_spec = _entry_spec("model", _arrays()["model"])
    with pytest.raises(TransactionValidationError, match="zero required keys"):
        ExpectedOwnerSchema(
            owner=CURRENT_TRAIN_STATE,
            active=False,
            required=(model_spec,),
            permitted=(model_spec,),
        )

    expected = _expected(owner_active={VERDICT_TRANSACTION: False})
    history = ParallelHistorySpec(
        name="impossible_inactive_history",
        keys=("verdict_seq",),
        sequence_key="verdict_seq",
        max_length=3,
    )
    with pytest.raises(TransactionValidationError, match="zero history schemas"):
        replace(expected, histories=(history,))


@pytest.mark.parametrize(
    ("factory", "match"),
    [
        (
            lambda: EntrySpec("x", "<f4", (1,), finite=1),
            "finite must be an exact bool",
        ),
        (
            lambda: EntrySpec("x", "<f4", (1,), allow_empty=np.bool_(True)),
            "allow_empty must be an exact bool",
        ),
        (
            lambda: ParallelHistorySpec(
                "h",
                ("seq",),
                "seq",
                1,
                require_contiguous=1,
            ),
            "require_contiguous must be an exact bool",
        ),
        (
            lambda: ParallelHistorySpec(
                "h",
                ("seq",),
                "seq",
                1,
                allow_empty=np.bool_(False),
            ),
            "allow_empty must be an exact bool",
        ),
    ],
)
def test_schema_boolean_flags_require_exact_bool(factory, match: str):
    with pytest.raises(TransactionValidationError, match=match):
        factory()


@pytest.mark.parametrize(
    "bad_shape",
    [
        (True,),
        (1.5,),
        ({},),
        {},
        np.asarray(1),
    ],
    ids=(
        "bool-dimension",
        "float-dimension",
        "dict-dimension",
        "dict-shape",
        "noniterable-array-shape",
    ),
)
def test_shape_fields_reject_noninteger_values_without_coercion(bad_shape):
    with pytest.raises(TransactionValidationError, match="shape"):
        EntrySpec("x", "<f4", bad_shape)


@pytest.mark.parametrize(
    "bad_nbytes",
    [True, 4.5, {}],
    ids=("bool", "float", "dict"),
)
def test_manifest_nbytes_rejects_noninteger_values_without_coercion(bad_nbytes):
    with pytest.raises(TransactionValidationError, match=r"nbytes.*exact integer"):
        EntryDescriptor(
            key="x",
            owner=CURRENT_TRAIN_STATE,
            dtype="<f4",
            shape=(1,),
            nbytes=bad_nbytes,
            sha256="0" * 64,
        )


@pytest.mark.parametrize(
    "bad_max_length",
    [True, 3.5, {}],
    ids=("bool", "float", "dict"),
)
def test_history_max_length_rejects_noninteger_values_without_coercion(
    bad_max_length,
):
    with pytest.raises(TransactionValidationError, match=r"max_length.*exact integer"):
        ParallelHistorySpec(
            name="history",
            keys=("seq",),
            sequence_key="seq",
            max_length=bad_max_length,
        )


@pytest.mark.parametrize(
    ("result_id", "result_sha", "match"),
    [
        (7, "a" * 64, "ID must have exact str type"),
        ("result-0", b"a" * 64, "SHA-256 must have exact str type"),
        ("result 0", "a" * 64, "ID is not canonical"),
        ("\tresult-0", "a" * 64, "ID is not canonical"),
    ],
)
def test_barrier_identity_requires_exact_canonical_strings(
    result_id,
    result_sha,
    match: str,
):
    with pytest.raises(TransactionValidationError, match=match):
        QuiescentBarrierState(
            next_submit_seq=1,
            next_apply_seq=1,
            pending_count=0,
            last_applied_result_id=result_id,
            last_applied_result_sha256=result_sha,
        ).validate()


def test_barrier_identity_rejects_str_subclasses():
    class StrSubclass(str):
        pass

    with pytest.raises(TransactionValidationError, match="ID must have exact str type"):
        QuiescentBarrierState(
            next_submit_seq=1,
            next_apply_seq=1,
            pending_count=0,
            last_applied_result_id=StrSubclass("result-0"),
            last_applied_result_sha256="a" * 64,
        ).validate()


def test_extra_claimed_key_and_dummy_scalar_substitution_fail():
    extra_arrays = _arrays()
    extra_arrays["model_extra"] = np.asarray([1.0], dtype="<f4")
    claims = dict(_OWNER_KEYS)
    claims[CURRENT_TRAIN_STATE] = ("model", "model_extra")
    extra_manifest = _manifest(extra_arrays, claims=claims)
    with pytest.raises(TransactionValidationError, match="extra="):
        validate_transaction(extra_arrays, extra_manifest, _expected())

    dummy_arrays = _arrays()
    dummy_arrays["model"] = np.asarray(0.0, dtype="<f4")
    dummy_manifest = _manifest(dummy_arrays)
    with pytest.raises(TransactionValidationError, match="shape drift"):
        validate_transaction(dummy_arrays, dummy_manifest, _expected())


@pytest.mark.parametrize("corruption", ["dtype", "shape", "nbytes", "hash"])
def test_dtype_shape_and_hash_corruption_fail(corruption: str):
    arrays = _arrays()
    manifest = _manifest(arrays)
    if corruption == "dtype":
        arrays["model"] = arrays["model"].astype("<f8")
        match = "dtype mismatch"
    elif corruption == "shape":
        arrays["model"] = arrays["model"].reshape(4)
        match = "shape mismatch"
    elif corruption == "nbytes":
        entries = list(manifest.entries)
        index = next(i for i, entry in enumerate(entries) if entry.key == "model")
        entries[index] = replace(entries[index], nbytes=entries[index].nbytes + 1)
        manifest = replace(manifest, entries=tuple(entries))
        match = "byte-length mismatch"
    else:
        entries = list(manifest.entries)
        index = next(i for i, entry in enumerate(entries) if entry.key == "model")
        entries[index] = replace(entries[index], sha256="0" * 64)
        manifest = replace(manifest, entries=tuple(entries))
        match = "content hash mismatch"
    with pytest.raises(TransactionValidationError, match=match):
        validate_transaction(arrays, manifest, _expected())


def test_nonfinite_value_fails_even_with_fresh_matching_manifest_hash():
    arrays = _arrays()
    arrays["model"][0, 0] = np.nan
    manifest = _manifest(arrays)
    with pytest.raises(TransactionValidationError, match="non-finite"):
        validate_transaction(arrays, manifest, _expected())


def test_object_arrays_and_pickle_backed_npz_are_forbidden(tmp_path):
    with pytest.raises(TransactionValidationError, match="object and structured arrays"):
        stage_arrays({"bad": np.asarray([{"x": 1}], dtype=object)})

    path = tmp_path / "object_state.npz"
    np.savez(path, bad=np.asarray([{"x": 1}], dtype=object))
    with pytest.raises(TransactionValidationError, match="requires pickle"):
        load_npz_staging(path)


def test_structured_dtypes_are_rejected_despite_equal_scalar_dtype_strings():
    split_dtype = np.dtype([("left", "<f4"), ("right", "<f4")])
    whole_dtype = np.dtype([("whole", "<f8")])
    assert split_dtype.str == whole_dtype.str == "|V8"

    with pytest.raises(TransactionValidationError, match="structured arrays"):
        stage_arrays({"bad": np.zeros((1,), dtype=split_dtype)})
    with pytest.raises(TransactionValidationError, match="structured arrays"):
        EntrySpec("bad", whole_dtype, (1,))


def test_npz_duplicate_member_names_are_rejected_before_mapping_collapse(tmp_path):
    path = tmp_path / "duplicate_members.npz"
    payload = io.BytesIO()
    np.save(payload, np.asarray([1], dtype=np.int64), allow_pickle=False)
    with (
        pytest.warns(UserWarning, match="Duplicate name"),
        zipfile.ZipFile(path, mode="w") as archive,
    ):
        archive.writestr("duplicate.npy", payload.getvalue())
        archive.writestr("duplicate.npy", payload.getvalue())

    with pytest.raises(TransactionValidationError, match="duplicate member names"):
        load_npz_staging(path)


def test_unequal_parallel_history_lengths_fail_before_restore():
    arrays = _arrays()
    arrays["verdict_loss"] = arrays["verdict_loss"][:-1]
    manifest = _manifest(arrays)
    expected = _expected(arrays)
    with pytest.raises(TransactionValidationError, match="unequal parallel lengths"):
        validate_transaction(arrays, manifest, expected)


@pytest.mark.parametrize(
    "sequence",
    [
        np.asarray([0, 2, 1], dtype="<u8"),
        np.asarray([0, 1, 1], dtype="<u8"),
    ],
)
def test_unordered_and_duplicate_journal_sequences_fail(sequence: np.ndarray):
    arrays = _arrays()
    arrays["verdict_seq"] = sequence
    manifest = _manifest(arrays)
    with pytest.raises(
        TransactionValidationError,
        match="unordered or duplicated",
    ):
        validate_transaction(arrays, manifest, _expected(arrays))


def test_invalid_bounded_history_truncation_fails():
    arrays = _arrays()
    arrays["verdict_next"] = np.asarray(4, dtype="<u8")
    manifest = _manifest(arrays)
    with pytest.raises(TransactionValidationError, match="canonical bounded suffix"):
        validate_transaction(arrays, manifest, _expected(arrays))


def test_serialized_pending_tamper_fails_despite_pristine_expected_object():
    arrays = _arrays()
    arrays["vtx.pending_count"] = np.asarray([1], dtype=np.int64)
    with pytest.raises(TransactionValidationError, match="pending_count"):
        validate_transaction(arrays, _manifest(arrays), _expected(arrays))


def test_serialized_cursor_tamper_fails_against_pristine_expected_object():
    arrays = _arrays()
    arrays["vtx.next_submit_seq"] = np.asarray([4], dtype=np.int64)
    arrays["vtx.next_apply_seq"] = np.asarray([4], dtype=np.int64)
    with pytest.raises(TransactionValidationError, match="differs from supplied"):
        validate_transaction(arrays, _manifest(arrays), _expected(arrays))


def test_serialized_last_identity_tamper_fails_against_pristine_expected_object():
    arrays = _arrays()
    arrays["vtx.last_applied_result_id"] = np.frombuffer(
        b"result-X",
        dtype=np.uint8,
    ).copy()
    arrays["vtx.last_applied_result_sha256"] = np.frombuffer(
        b"d" * 64,
        dtype=np.uint8,
    ).copy()
    with pytest.raises(TransactionValidationError, match="differs from supplied"):
        validate_transaction(arrays, _manifest(arrays), _expected(arrays))


@pytest.mark.parametrize(
    ("key", "value", "match"),
    [
        (
            "vtx.next_submit_seq",
            np.asarray(3, dtype=np.int64),
            "dtype int64 and shape",
        ),
        (
            "vtx.next_submit_seq",
            np.asarray([3], dtype=np.uint64),
            "dtype int64 and shape",
        ),
        (
            "vtx.last_applied_result_id",
            np.asarray(b"result-2", dtype="S8"),
            "one-dimensional uint8",
        ),
        (
            "vtx.last_applied_result_id",
            np.asarray([0xFF], dtype=np.uint8),
            "not valid UTF-8",
        ),
        (
            "vtx.schema",
            np.frombuffer(b"wrong.schema", dtype=np.uint8).copy(),
            "barrier schema",
        ),
    ],
)
def test_serialized_barrier_fields_enforce_capture_dtype_shape_and_utf8(
    key: str,
    value: np.ndarray,
    match: str,
):
    arrays = _arrays()
    arrays[key] = value
    with pytest.raises(TransactionValidationError, match=match):
        validate_transaction(arrays, _manifest(arrays), _expected(arrays))


@pytest.mark.parametrize(
    "pending_key",
    [
        "__cl_pend_shadow",
        "vtx.__cl_pend_shadow",
        "ns/__cl_pend_shadow",
        "ns:__cl_pend_shadow",
        r"ns\__cl_pend_shadow",
        "outer.inner.__cl_pend_shadow.tail",
        "prefix__cl_pend_shadow",
    ],
)
def test_native_v3_pending_verdict_payload_is_rejected(pending_key: str):
    arrays = _arrays()
    manifest = _manifest(arrays)
    arrays[pending_key] = np.asarray(1, dtype="<u8")
    with pytest.raises(TransactionValidationError, match="pending-verdict"):
        validate_transaction(arrays, manifest, _expected())


def test_derived_lineage_keys_are_described_but_excluded_from_reverse_ownership():
    manifest = _manifest()
    claims = {claim.owner: claim.keys for claim in manifest.owner_claims}
    assert "complete_state_sha" not in set().union(*map(set, claims.values()))
    descriptor = next(entry for entry in manifest.entries if entry.key == "complete_state_sha")
    assert descriptor.owner == LINEAGE_ENVELOPE
    validate_transaction(_arrays(), manifest, _expected())

    claims[LINEAGE_ENVELOPE] = ("complete_state_sha", "seed")
    with pytest.raises(TransactionValidationError, match="cannot also be owner-claimed"):
        _manifest(claims=claims)


def test_owner_and_fourteen_domain_matrix_are_exact_and_not_prefix_inferred():
    assert len(ATOMIC_OWNERS) == 6
    assert len(SEMANTIC_DOMAINS) == 14
    assert tuple(CANONICAL_DOMAIN_COVERAGE) == SEMANTIC_DOMAINS
    assert CANONICAL_DOMAIN_COVERAGE[FRESH_LINEAGE_DOMAIN] == (LINEAGE_ENVELOPE,)
    assert CANONICAL_DOMAIN_COVERAGE[BEST_STAGE_DOMAIN] == (CAUSAL_SELECTION_STATE,)

    domains = list(canonical_domain_coverage())
    domains[0] = DomainCoverage(domains[0].domain, (ROLLBACK_SAVEPOINT,))
    with pytest.raises(TransactionValidationError, match="canonical matrix"):
        replace(_expected(), domain_coverage=tuple(domains))


def test_canonical_reserialization_accepts_equal_state_and_rejects_value_drift():
    arrays = _arrays()
    reference = validate_transaction(
        arrays,
        _manifest(arrays),
        _expected(arrays),
    )
    equal = {key: value.copy() for key, value in arrays.items()}
    replay = verify_canonical_reserialization(
        reference,
        equal,
        _expected(equal),
        barrier_binding=_barrier_binding(),
    )
    assert all(
        replay.owner_semantic_hashes[owner] == reference.owner_semantic_hashes[owner]
        for owner in RESTORABLE_STATE_OWNERS
    )

    changed = {key: value.copy() for key, value in arrays.items()}
    changed["model"][0, 0] = np.float32(-1.0)
    with pytest.raises(TransactionValidationError, match="semantic hash"):
        verify_canonical_reserialization(
            reference,
            changed,
            _expected(changed),
            barrier_binding=_barrier_binding(),
        )

    changed_barrier = {key: value.copy() for key, value in arrays.items()}
    changed_barrier["vtx.next_submit_seq"] = np.asarray([4], dtype=np.int64)
    changed_barrier["vtx.next_apply_seq"] = np.asarray([4], dtype=np.int64)
    with pytest.raises(TransactionValidationError, match="differs from supplied"):
        verify_canonical_reserialization(
            reference,
            changed_barrier,
            _expected(changed_barrier),
            barrier_binding=_barrier_binding(),
        )

    with pytest.raises(TypeError, match="unexpected keyword argument 'owners'"):
        verify_canonical_reserialization(
            reference,
            equal,
            _expected(equal),
            owners=(CURRENT_TRAIN_STATE,),
            barrier_binding=_barrier_binding(),
        )


def test_cross_invariants_run_on_staged_arrays_and_topology_helper_fails_closed():
    arrays = _arrays()

    def topology(values):
        require_matching_topology(
            values,
            key_pairs=(("model", "rollback_model"),),
            complete_left_keys=("model",),
            complete_right_keys=("rollback_model",),
            label="rollback versus live",
        )

    validate_transaction(
        arrays,
        _manifest(arrays),
        _expected(arrays),
        invariant_validators=(topology,),
    )
    changed = _arrays()
    changed["rollback_model"] = changed["rollback_model"].reshape(4)
    manifest = _manifest(changed)
    expected = _expected(changed)
    with pytest.raises(TransactionValidationError, match="shape differs"):
        validate_transaction(
            changed,
            manifest,
            expected,
            invariant_validators=(topology,),
        )


def test_topology_matching_uses_explicit_pairs_not_namespace_sort_order():
    arrays = {
        "live.alpha": np.zeros((2,), dtype="<f4"),
        "live.zeta": np.zeros((3,), dtype="<f4"),
        "save.aaa": np.zeros((3,), dtype="<f4"),
        "save.zzz": np.zeros((2,), dtype="<f4"),
    }
    require_matching_topology(
        arrays,
        key_pairs=(
            ("live.zeta", "save.aaa"),
            ("live.alpha", "save.zzz"),
        ),
        complete_left_keys=("live.alpha", "live.zeta"),
        complete_right_keys=("save.aaa", "save.zzz"),
        label="explicit namespace pairing",
    )
    with pytest.raises(TransactionValidationError, match="shape differs"):
        require_matching_topology(
            arrays,
            key_pairs=(
                ("live.alpha", "save.aaa"),
                ("live.zeta", "save.zzz"),
            ),
            complete_left_keys=("live.alpha", "live.zeta"),
            complete_right_keys=("save.aaa", "save.zzz"),
            label="wrong explicit pairing",
        )
    with pytest.raises(TransactionValidationError, match="reuses"):
        require_matching_topology(
            arrays,
            key_pairs=(
                ("live.alpha", "save.zzz"),
                ("live.alpha", "save.aaa"),
            ),
            complete_left_keys=("live.alpha", "live.zeta"),
            complete_right_keys=("save.aaa", "save.zzz"),
            label="duplicate explicit pairing",
        )
    with pytest.raises(TransactionValidationError, match="does not exhaust"):
        require_matching_topology(
            arrays,
            key_pairs=(("live.alpha", "save.zzz"),),
            complete_left_keys=("live.alpha", "live.zeta"),
            complete_right_keys=("save.aaa", "save.zzz"),
            label="subset explicit pairing",
        )
    with pytest.raises(TransactionValidationError, match="does not exhaust"):
        require_matching_topology(
            arrays,
            key_pairs=(
                ("live.zeta", "save.aaa"),
                ("live.alpha", "save.zzz"),
            ),
            complete_left_keys=("live.alpha",),
            complete_right_keys=("save.aaa", "save.zzz"),
            label="underdeclared left inventory",
        )


def test_descriptor_rejects_noncanonical_sha_and_unknown_json_fields():
    with pytest.raises(TransactionValidationError, match="sha256"):
        EntryDescriptor(
            key="x",
            owner=CURRENT_TRAIN_STATE,
            dtype="<f4",
            shape=(1,),
            nbytes=4,
            sha256="NOT-A-HASH",
        )
    payload = _manifest().as_dict()
    payload["unknown"] = True

    with pytest.raises(TransactionValidationError, match="unknown"):
        TransactionManifest.from_json(json.dumps(payload))
