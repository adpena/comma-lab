from __future__ import annotations

import hashlib
import json
import os
import stat
from dataclasses import replace
from pathlib import Path

import pytest

from tac.witness_dsl.activation_ledger import duty_to_measure, known_levers
from tac.witness_dsl.curriculum_dsl import BASELINE, EmaDecayCalibrated, IntegerPlaneEmitter
from tac.witness_dsl.integer_plane_emitter_policy import (
    CHECKPOINT_SCHEMA_SHA256,
    FROZEN_SEGNET_SHA256,
    FUTURE_RESUME_HOOK_PREFIX,
    MEASURED_U4_SINGULAR_VALUES,
    POLICY_CONTRACT_RECEIPT_KEY,
    U4_SOURCE_ARTIFACT,
    U4_SOURCE_SHA256,
    BasisMode,
    IntegerPlaneEmitterCheckpointError,
    IntegerPlaneEmitterPolicy,
    IntegerPlaneEmitterPolicyError,
    IntegerPlaneEmitterStageCheckpoint,
    PolicyMode,
    STEMode,
    u4_lawrefs,
)
from tac.witness_dsl.lawref import lawref_to_declaration, resolve
from tac.witness_dsl.lever_registry import (
    completeness,
    lever_factories,
    name_composable_levers,
    package_lever_factories,
    resolve_composable_lever,
)
from tac.witness_dsl.typed_config import typed_lever_from_dsl

_ZERO_SHA = "0" * 64


def _tensor(shape: list[int], value: float) -> dict[str, object]:
    count = 1
    for dim in shape:
        count *= dim
    return {"dtype": "float32", "shape": shape, "data": [value] * count}


def _residual_state(pair_count: int = 12, value: float = 0.25) -> dict[str, object]:
    return {
        "pair_plane_codes": _tensor([pair_count, 2, 4], value),
        "shared_rgb_head": _tensor([4, 3], value),
    }


def _checkpoint(**changes: object) -> IntegerPlaneEmitterStageCheckpoint:
    basis_id = str(changes.get("basis_id", BasisMode.RAW_CENTERED.value))
    try:
        policy = IntegerPlaneEmitterPolicy(basis=BasisMode(basis_id))
    except ValueError:
        policy = IntegerPlaneEmitterPolicy()
    contract = policy.compile_contract()
    values: dict[str, object] = {
        "policy_contract": contract,
        "config_sha256": contract["policy_sha256"],
        "stage_name": "basis_ab",
        "stage_index": 2,
        "epoch": 17,
        "global_step": 900,
        "next_pair": 11,
        "basis_id": BasisMode.RAW_CENTERED.value,
        "ste_id": STEMode.SATURATION_AWARE_UINT8.value,
        "fixed_capacity_signature": contract["capacity_signature"],
        "live_residual_parameters": _residual_state(value=0.25),
        "ema_shadow": _residual_state(value=0.2),
        "optimizer_state": {"step": 900, "moments": [0.1]},
        "rng_state": {"numpy_pcg64": {"state": 12345}},
        "topology_state_sha256": "2" * 64,
        "discrete_state_sha256": "3" * 64,
        "event_state_sha256": "4" * 64,
        "dual_state_sha256": "5" * 64,
    }
    values.update(changes)
    return IntegerPlaneEmitterStageCheckpoint(**values)  # type: ignore[arg-type]


def test_policy_contract_is_json_safe_default_off_and_pair_parallel() -> None:
    contract = IntegerPlaneEmitterPolicy().compile()
    json.dumps(contract, allow_nan=False)
    assert contract["camera_hw"] == [874, 1164]
    assert contract["scorer_hw"] == [384, 512]
    assert contract["plane_count"] == 2
    assert contract["channels"] == 3
    assert contract["pair_parallel_expansion"] is True
    assert contract["cross_pair_autoregression"] is False
    assert contract["score_affecting_enabled"] is False
    assert contract["research_only"] is True
    assert contract["basis_verdict_state"] == "UNRESOLVED_BUILD_ONLY"
    assert contract["resume_hook_status"] == "FUTURE_ONLY_NOT_REGISTERED"
    assert contract["future_resume_hook_prefix"] == FUTURE_RESUME_HOOK_PREFIX == "__ipe_"


def test_basis_arms_share_exact_capacity_but_have_distinct_policy_identity() -> None:
    raw = IntegerPlaneEmitterPolicy(basis=BasisMode.RAW_CENTERED)
    u4 = IntegerPlaneEmitterPolicy(basis=BasisMode.SIGN_FIXED_U4_PAIR_MARGIN)
    curvelet = IntegerPlaneEmitterPolicy(basis=BasisMode.R1B4_WINDOWED_CURVELET)
    assert raw.capacity_signature() == u4.capacity_signature() == curvelet.capacity_signature()
    assert raw.compile()["policy_sha256"] != u4.compile()["policy_sha256"]
    assert raw.compile()["policy_sha256"] != curvelet.compile()["policy_sha256"]
    assert u4.compile()["basis"] == "sign_fixed_u4_pair_margin"
    assert curvelet.compile()["basis"] == "r1b4_windowed_curvelet"


@pytest.mark.parametrize(
    ("change", "match"),
    [
        ({"basis": "raw_centered"}, "BasisMode"),
        ({"ste": "identity"}, "STE mode"),
        ({"residual_width": 0}, "capacity-locked"),
        ({"residual_width": 8}, "capacity-locked"),
        ({"camera_hw": (875, 1164)}, "camera_hw"),
        ({"scorer_hw": (512, 384)}, "scorer_hw"),
        ({"plane_count": 1}, "plane_count"),
        ({"pair_parallel_expansion": False}, "pair_parallel_expansion"),
        ({"cross_pair_autoregression": True}, "cross_pair_autoregression"),
    ],
)
def test_invalid_or_receiver_serial_policy_variants_fail_closed(change: dict[str, object], match: str) -> None:
    with pytest.raises(IntegerPlaneEmitterPolicyError, match=match):
        IntegerPlaneEmitterPolicy(**change)  # type: ignore[arg-type]


def test_authority_fields_are_sealed_and_compile_requests_refuse() -> None:
    for field in (
        "score_affecting_enabled",
        "trainer_activation",
        "launch",
        "paid_dispatch",
        "score_claim",
        "promotion",
        "pointer_mutation",
    ):
        with pytest.raises(IntegerPlaneEmitterPolicyError, match="sealed field"):
            replace(IntegerPlaneEmitterPolicy(), **{field: True})
        with pytest.raises(IntegerPlaneEmitterPolicyError, match="cannot authorize"):
            IntegerPlaneEmitterPolicy().compile_contract(**{field: True})
    with pytest.raises(IntegerPlaneEmitterPolicyError, match="unknown authority"):
        IntegerPlaneEmitterPolicy().compile_contract(submit=True)


def test_four_lawrefs_resolve_exact_sha_pinned_json_paths() -> None:
    refs = u4_lawrefs()
    assert tuple(sorted(refs)) == ("sigma_1", "sigma_2", "sigma_3", "sigma_4")
    resolved = []
    for index, name in enumerate(sorted(refs)):
        ref = refs[name]
        inp = ref.inputs["value"]
        assert inp.artifact_path == U4_SOURCE_ARTIFACT
        assert inp.extract == f"frozen_target/singular_values/{index}"
        assert inp.expected_sha256 == U4_SOURCE_SHA256
        assert inp.config_tags == {"frozen_segnet_sha256": FROZEN_SEGNET_SHA256}
        row = resolve(
            ref,
            target_config_tags={"frozen_segnet_sha256": FROZEN_SEGNET_SHA256},
        )
        resolved.append(float(row.value))
    assert tuple(round(value, 10) for value in resolved) == MEASURED_U4_SINGULAR_VALUES


def test_policy_hash_is_deterministic_and_uses_declaration_hashes() -> None:
    first = IntegerPlaneEmitterPolicy().compile()
    second = IntegerPlaneEmitterPolicy().compile()
    assert first == second
    assert len(first["policy_sha256"]) == 64
    assert set(first["lawref_declaration_sha256"]) == {
        "sigma_1",
        "sigma_2",
        "sigma_3",
        "sigma_4",
    }
    for name, declaration in first["lawref_declarations"].items():
        blob = json.dumps(
            declaration, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
        ).encode("ascii")
        assert first["lawref_declaration_sha256"][name] == hashlib.sha256(blob).hexdigest()
    assert all("resolved_at" not in row for row in first["lawref_resolution"].values())
    for row in first["lawref_resolution"].values():
        assert row["inputs"][0]["source"] == U4_SOURCE_ARTIFACT
        assert not Path(row["inputs"][0]["source"]).is_absolute()


def test_artifact_tamper_refuses_law_resolution(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import tac.witness_dsl.integer_plane_emitter_policy as module

    tampered = tmp_path / "u4.json"
    tampered.write_text(Path(U4_SOURCE_ARTIFACT).read_text() + "\n", encoding="utf-8")
    monkeypatch.setattr(module, "U4_SOURCE_ARTIFACT", str(tampered))
    with pytest.raises(Exception, match="sha256 mismatch"):
        module.IntegerPlaneEmitterPolicy().compile()


def test_required_policy_factory_is_argv_inert_and_baseline_valid() -> None:
    baseline_argv = BASELINE.compile_trainer_argv()
    lever = IntegerPlaneEmitter(policy=IntegerPlaneEmitterPolicy())
    program = BASELINE.with_lever(lever)
    assert lever.name == "IntegerPlaneEmitter"
    assert lever.overrides == {}
    assert lever.epochs_delta == 0
    assert "policy_sha256=" in lever.notes
    assert lever.runtime_receipt_schemas == {}
    assert set(lever.policy_contracts) == {POLICY_CONTRACT_RECEIPT_KEY}
    receipt_contract = lever.policy_contracts[POLICY_CONTRACT_RECEIPT_KEY]
    assert receipt_contract == IntegerPlaneEmitterPolicy().compile_contract()
    typed = typed_lever_from_dsl(lever)
    assert typed.to_dsl().policy_contracts == lever.policy_contracts
    assert program.validate() == []
    assert program.compile_trainer_argv() == baseline_argv
    compiled_argv, manifest = program.compile_trainer_argv_with_constants()
    assert compiled_argv == baseline_argv
    assert manifest == {}


def test_canonical_typed_lever_adapter_preserves_every_custody_surface() -> None:
    source = replace(
        EmaDecayCalibrated(1000),
        runtime_receipt_schemas={"--ema-decay": "ema_decay_runtime_receipt.v1"},
        policy_contracts={"policy_only": {"nested": ["kept"], "enabled": False}},
    )
    typed = typed_lever_from_dsl(source)
    assert typed.lawrefs == source.lawrefs
    assert typed.lawref_declarations == {flag: lawref_to_declaration(ref) for flag, ref in source.lawrefs.items()}
    assert typed.constant_manifest == source.constant_manifest
    assert typed.runtime_receipt_schemas == source.runtime_receipt_schemas
    assert typed.policy_contracts == source.policy_contracts
    assert typed.policy_contracts is not source.policy_contracts
    assert typed.policy_contracts["policy_only"] is not source.policy_contracts["policy_only"]
    roundtrip = typed.to_dsl()
    assert roundtrip.lawrefs == source.lawrefs
    assert roundtrip.constant_manifest == source.constant_manifest
    assert roundtrip.runtime_receipt_schemas == source.runtime_receipt_schemas
    assert roundtrip.policy_contracts == source.policy_contracts


def test_named_v9_lever_adapters_delegate_to_lossless_canonical_adapter() -> None:
    from tac.witness_dsl.spec_next_launch_all_levers_20260713 import _typed
    from tac.witness_dsl.spec_v9_cgauge import _typed_ideal_lever

    source = replace(
        EmaDecayCalibrated(1000),
        runtime_receipt_schemas={"--ema-decay": "ema_decay_runtime_receipt.v1"},
        policy_contracts={"policy_only": {"schema": "adapter_regression.v1"}},
    )
    for adapter in (_typed, _typed_ideal_lever):
        typed = adapter(source)
        assert typed.lawrefs == source.lawrefs
        assert typed.constant_manifest == source.constant_manifest
        assert typed.runtime_receipt_schemas == source.runtime_receipt_schemas
        assert typed.policy_contracts == source.policy_contracts
        assert typed.to_dsl().policy_contracts == source.policy_contracts


def test_factory_refuses_missing_or_wrong_policy() -> None:
    with pytest.raises(TypeError, match="required keyword-only"):
        IntegerPlaneEmitter()  # type: ignore[call-arg]
    with pytest.raises(TypeError, match="requires an IntegerPlaneEmitterPolicy"):
        IntegerPlaneEmitter(policy=object())  # type: ignore[arg-type]


_C2_EMITTER_FLAGS = frozenset(
    {
        "--integer-plane-emitter-mode",
        "--integer-plane-emitter-basis",
        "--integer-plane-emitter-policy-sha256",
    }
)


def test_registry_and_activation_surfaces_track_required_policy_factory(tmp_path: Path) -> None:
    # The factory is a PACKAGE lever bound to the dedicated C2 trainer, not a ``curriculum_dsl``
    # one (ddm_ql3, 2026-09-04). ``lever_factories()`` ASTs ``curriculum_dsl.py`` alone and grades
    # against the level-set trainer pair, which never declared these three flags; the package-wide
    # surface resolves each module against its OWN declared trainer and is the honest home.
    rows = [f for f in package_lever_factories() if f.factory == "IntegerPlaneEmitter"]
    assert len(rows) == 1, "exactly one module may own the C2 emitter factory"
    (row,) = rows
    assert row.module == "integer_plane_emitter_lever.py"
    assert frozenset(row.flags) == _C2_EMITTER_FLAGS
    assert "IntegerPlaneEmitter" not in lever_factories(), (
        "the C2 lever must NOT be graded as a curriculum_dsl factory — that binding is what "
        "reported its three live flags as completeness().stale drift"
    )
    assert "IntegerPlaneEmitter" in known_levers()
    assert "IntegerPlaneEmitter" in duty_to_measure(path=tmp_path / "activation.jsonl")
    assert "IntegerPlaneEmitter" not in name_composable_levers()
    with pytest.raises(ValueError, match="requires explicit args"):
        resolve_composable_lever("IntegerPlaneEmitter")


def test_c2_lever_module_declares_the_trainer_that_owns_its_flags() -> None:
    """The binding is DECLARED, and the C2 parser really declares all three flags.

    Regression for the ddm_ql3 red state: ``completeness().stale`` reported these three live
    flags as DSL drift purely because the lever was homed in a module bound to the level-set
    trainer pair. Pins the cure at both ends — the module states its trainer, and that trainer's
    argparse actually carries every flag the factory emits (so this is not a relabelling).
    """
    import re

    from tac.witness_dsl import integer_plane_emitter_lever as lever_mod
    from tac.witness_dsl.lever_registry import module_declares_trainer, module_trainer_paths

    source = Path(lever_mod.__file__)
    assert module_declares_trainer(source), "a defaulted binding is indistinguishable from an unconsidered one"
    trainers = module_trainer_paths(source)
    assert len(trainers) == 1
    assert trainers[0].name == "integer_plane_banded_trainer.py"
    assert trainers[0].is_file()
    declared = set(re.findall(r'add_argument\(\s*"(--[a-z0-9-]+)"', trainers[0].read_text()))
    assert declared.issuperset(_C2_EMITTER_FLAGS), "the C2 parser must own every flag the lever emits"

    (row,) = [f for f in package_lever_factories() if f.factory == "IntegerPlaneEmitter"]
    assert row.missing_flags == (), "flags exist on the declared trainer — not a DESIGNED-STUB"
    assert not row.is_stub and row.trainer_declared and not row.label_drift


def test_c2_lever_reexport_from_curriculum_dsl_is_the_same_object() -> None:
    """Every historical import path keeps working — and keeps working on the SAME factory."""
    from tac.witness_dsl import curriculum_dsl as cd
    from tac.witness_dsl.integer_plane_emitter_lever import IntegerPlaneEmitter as canonical

    assert IntegerPlaneEmitter is canonical
    assert cd.IntegerPlaneEmitter is canonical
    assert completeness().stale == [], "no DSL-emitted flag may be absent from its trainer"


def test_active_policy_is_argv_effective_while_legacy_policy_stays_inert() -> None:
    inactive = IntegerPlaneEmitter(policy=IntegerPlaneEmitterPolicy())
    active_policy = IntegerPlaneEmitterPolicy(mode=PolicyMode.BANDED_TRAINING)
    active = IntegerPlaneEmitter(policy=active_policy)
    assert inactive.overrides == {}
    assert inactive.runtime_receipt_schemas == {}
    assert active.overrides == {
        "--integer-plane-emitter-mode": "banded_training",
        "--integer-plane-emitter-basis": "raw_centered",
        "--integer-plane-emitter-policy-sha256": active_policy.compile_contract()["policy_sha256"],
    }
    assert set(active.runtime_receipt_schemas) == {"--integer-plane-emitter-policy-sha256"}


def test_checkpoint_roundtrip_is_canonical_and_complete() -> None:
    checkpoint = _checkpoint()
    encoded = checkpoint.to_bytes()
    reopened = IntegerPlaneEmitterStageCheckpoint.from_bytes(encoded)
    assert reopened == checkpoint
    assert reopened.to_bytes() == encoded
    doc = json.loads(encoded)
    body = doc["body"]
    assert body["schema_sha256"] == CHECKPOINT_SCHEMA_SHA256
    assert body["live_residual_parameters"] == checkpoint.live_residual_parameters
    assert body["ema_shadow"] == checkpoint.ema_shadow
    assert body["optimizer_state"] == checkpoint.optimizer_state
    assert body["rng_state"] == checkpoint.rng_state


def test_checkpoint_body_tamper_is_refused() -> None:
    doc = json.loads(_checkpoint().to_bytes())
    doc["body"]["next_pair"] += 1
    tampered = json.dumps(doc, sort_keys=True, separators=(",", ":")).encode("ascii")
    with pytest.raises(IntegerPlaneEmitterCheckpointError, match="body hash mismatch"):
        IntegerPlaneEmitterStageCheckpoint.from_bytes(tampered)


def test_checkpoint_unknown_field_and_noncanonical_encoding_are_refused() -> None:
    encoded = _checkpoint().to_bytes()
    doc = json.loads(encoded)
    doc["body"]["surprise"] = True
    blob = json.dumps(doc, sort_keys=True, separators=(",", ":")).encode("ascii")
    with pytest.raises(IntegerPlaneEmitterCheckpointError, match="body fields mismatch"):
        IntegerPlaneEmitterStageCheckpoint.from_bytes(blob)
    pretty = json.dumps(json.loads(encoded), indent=2).encode("ascii")
    with pytest.raises(IntegerPlaneEmitterCheckpointError, match="noncanonical"):
        IntegerPlaneEmitterStageCheckpoint.from_bytes(pretty)


@pytest.mark.parametrize(
    ("change", "match"),
    [
        ({"config_sha256": "bad"}, "config_sha256"),
        ({"stage_name": "../escape"}, "stage_name"),
        ({"stage_index": -1}, "stage_index"),
        ({"basis_id": "other"}, "basis_id"),
        ({"ste_id": "identity"}, "ste_id"),
        ({"live_residual_parameters": [1, 2]}, "live_residual_parameters"),
        ({"live_residual_parameters": {}}, "live_residual_parameters"),
        ({"ema_shadow": {}}, "ema_shadow"),
        ({"optimizer_state": {}}, "optimizer_state"),
        ({"rng_state": {}}, "rng_state"),
        ({"rng_state": {"bad": float("nan")}}, "canonical JSON"),
    ],
)
def test_checkpoint_invalid_state_fails_closed(change: dict[str, object], match: str) -> None:
    with pytest.raises(IntegerPlaneEmitterCheckpointError, match=match):
        _checkpoint(**change)


def test_stage_filename_is_distinct_encoded_and_safe() -> None:
    checkpoint = _checkpoint()
    name = checkpoint.filename("c2_run")
    assert name == "c2_run__ipe_stage002_basis_ab_ep000017_step000000000900.json"
    with pytest.raises(IntegerPlaneEmitterCheckpointError, match="run_id"):
        checkpoint.filename("../c2")


def test_checkpoint_write_is_atomic_and_never_overwrites(tmp_path: Path) -> None:
    checkpoint = _checkpoint()
    path = checkpoint.write_new(tmp_path, "c2_run")
    original = path.read_bytes()
    assert original == checkpoint.to_bytes()
    with pytest.raises(IntegerPlaneEmitterCheckpointError, match="overwrite refused"):
        checkpoint.write_new(tmp_path, "c2_run")
    assert path.read_bytes() == original
    assert not tuple(tmp_path.glob("*.tmp.*"))
    assert not tuple(tmp_path.glob("*.reserve"))


def test_checkpoint_publish_resumes_exact_crash_temporary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import tac.witness_dsl.integer_plane_emitter_policy as module

    checkpoint = _checkpoint()
    target = tmp_path / checkpoint.filename("c2_run")
    stale = tmp_path / f".{target.name}.tmp.crashed"
    stale.write_bytes(checkpoint.to_bytes())
    synced_modes: list[int] = []
    real_fsync = os.fsync

    def recording_fsync(fd: int) -> None:
        synced_modes.append(os.fstat(fd).st_mode)
        real_fsync(fd)

    monkeypatch.setattr(module.os, "fsync", recording_fsync)
    assert checkpoint.write_new(tmp_path, "c2_run") == target
    assert target.read_bytes() == checkpoint.to_bytes()
    assert not stale.exists()
    assert len(synced_modes) == 3
    assert stat.S_ISREG(synced_modes[0])
    assert stat.S_ISDIR(synced_modes[1])
    assert stat.S_ISDIR(synced_modes[2])


def test_checkpoint_publish_boundary_target_race_never_clobbers_winner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import tac.witness_dsl.integer_plane_emitter_policy as module

    checkpoint = _checkpoint()
    target = tmp_path / checkpoint.filename("c2_run")
    winner = b"independent publisher bytes"
    real_link = os.link

    def create_winner_then_link(
        source: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        destination: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        *,
        follow_symlinks: bool = True,
    ) -> None:
        Path(destination).write_bytes(winner)
        real_link(source, destination, follow_symlinks=follow_symlinks)

    monkeypatch.setattr(module.os, "link", create_winner_then_link)
    with pytest.raises(IntegerPlaneEmitterCheckpointError, match="overwrite refused"):
        checkpoint.write_new(tmp_path, "c2_run")
    assert target.read_bytes() == winner
    retained = tuple(tmp_path.glob(f".{target.name}.tmp.*"))
    assert len(retained) == 1
    assert retained[0].read_bytes() == checkpoint.to_bytes()


def test_checkpoint_stale_source_swap_at_link_boundary_cannot_publish_attacker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import tac.witness_dsl.integer_plane_emitter_policy as module

    checkpoint = _checkpoint()
    payload = checkpoint.to_bytes()
    target = tmp_path / checkpoint.filename("c2_run")
    stale = tmp_path / f".{target.name}.tmp.crashed"
    validated_backup = tmp_path / "validated-stale-backup.json"
    attacker = b"attacker-controlled checkpoint bytes"
    stale.write_bytes(payload)
    real_link = os.link

    def swap_source_then_link(
        source: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        destination: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        *,
        follow_symlinks: bool = True,
    ) -> None:
        Path(source).rename(validated_backup)
        Path(source).write_bytes(attacker)
        real_link(source, destination, follow_symlinks=follow_symlinks)

    monkeypatch.setattr(module.os, "link", swap_source_then_link)
    with pytest.raises(IntegerPlaneEmitterCheckpointError, match="unexpected identity"):
        checkpoint.write_new(tmp_path, "c2_run")
    assert not target.exists()
    assert stale.read_bytes() == attacker
    assert validated_backup.read_bytes() == payload


def test_checkpoint_post_link_target_swap_is_removed_and_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import tac.witness_dsl.integer_plane_emitter_policy as module

    checkpoint = _checkpoint()
    payload = checkpoint.to_bytes()
    target = tmp_path / checkpoint.filename("c2_run")
    stale = tmp_path / f".{target.name}.tmp.crashed"
    attacker = b"post-link target replacement"
    stale.write_bytes(payload)
    real_link = os.link

    def link_then_swap_target(
        source: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        destination: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        *,
        follow_symlinks: bool = True,
    ) -> None:
        real_link(source, destination, follow_symlinks=follow_symlinks)
        Path(destination).unlink()
        Path(destination).write_bytes(attacker)

    monkeypatch.setattr(module.os, "link", link_then_swap_target)
    with pytest.raises(IntegerPlaneEmitterCheckpointError, match="unexpected identity"):
        checkpoint.write_new(tmp_path, "c2_run")
    assert not target.exists()
    assert stale.read_bytes() == payload


def test_checkpoint_target_swap_after_first_verification_is_refused_before_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import tac.witness_dsl.integer_plane_emitter_policy as module

    checkpoint = _checkpoint()
    payload = checkpoint.to_bytes()
    target = tmp_path / checkpoint.filename("c2_run")
    stale = tmp_path / f".{target.name}.tmp.crashed"
    attacker = b"replacement after first target verification"
    stale.write_bytes(payload)
    real_verify = module._verify_published_checkpoint_target
    swapped = False

    def verify_then_swap_once(
        checked_target: Path,
        *,
        expected_stat: os.stat_result,
        payload: bytes,
    ) -> None:
        nonlocal swapped
        real_verify(
            checked_target,
            expected_stat=expected_stat,
            payload=payload,
        )
        if not swapped:
            checked_target.unlink()
            checked_target.write_bytes(attacker)
            swapped = True

    monkeypatch.setattr(module, "_verify_published_checkpoint_target", verify_then_swap_once)
    with pytest.raises(IntegerPlaneEmitterCheckpointError, match="unexpected identity"):
        checkpoint.write_new(tmp_path, "c2_run")
    assert swapped
    assert not target.exists()
    assert stale.read_bytes() == payload


def test_checkpoint_recovery_refuses_symlink_stale_temporary(tmp_path: Path) -> None:
    checkpoint = _checkpoint()
    target = tmp_path / checkpoint.filename("c2_run")
    stale = tmp_path / f".{target.name}.tmp.crashed"
    source = tmp_path / "attacker-controlled.json"
    source.write_bytes(checkpoint.to_bytes())
    stale.symlink_to(source)
    with pytest.raises(IntegerPlaneEmitterCheckpointError, match="non-symlink regular file"):
        checkpoint.write_new(tmp_path, "c2_run")
    assert stale.is_symlink()
    assert not target.exists()


def test_checkpoint_recovery_refuses_nonregular_stale_temporary(tmp_path: Path) -> None:
    checkpoint = _checkpoint()
    target = tmp_path / checkpoint.filename("c2_run")
    stale = tmp_path / f".{target.name}.tmp.crashed"
    stale.mkdir()
    with pytest.raises(IntegerPlaneEmitterCheckpointError, match="non-symlink regular file"):
        checkpoint.write_new(tmp_path, "c2_run")
    assert stale.is_dir()
    assert not target.exists()


def test_checkpoint_publish_refuses_different_stale_temporary(tmp_path: Path) -> None:
    checkpoint = _checkpoint()
    target = tmp_path / checkpoint.filename("c2_run")
    stale = tmp_path / f".{target.name}.tmp.crashed"
    stale.write_bytes(b"different checkpoint bytes")
    with pytest.raises(IntegerPlaneEmitterCheckpointError, match="stale checkpoint temporary differs"):
        checkpoint.write_new(tmp_path, "c2_run")
    assert not target.exists()
    assert stale.read_bytes() == b"different checkpoint bytes"


def test_checkpoint_hash_inputs_cannot_be_empty_placeholders() -> None:
    for field in (
        "config_sha256",
        "fixed_capacity_signature",
        "topology_state_sha256",
        "discrete_state_sha256",
        "event_state_sha256",
        "dual_state_sha256",
    ):
        with pytest.raises(IntegerPlaneEmitterCheckpointError, match=field):
            _checkpoint(**{field: _ZERO_SHA})


@pytest.mark.parametrize(
    ("field", "replacement", "match"),
    [
        ("basis", BasisMode.SIGN_FIXED_U4_PAIR_MARGIN.value, "sealed compiled"),
        ("ste", "identity", "sealed compiled"),
        ("pair_parallel_expansion", False, "sealed compiled"),
        ("cross_pair_autoregression", True, "sealed compiled"),
        ("research_only", False, "sealed compiled"),
        ("capacity_signature", "a" * 64, "sealed compiled"),
    ],
)
def test_checkpoint_policy_contract_drift_is_refused(field: str, replacement: object, match: str) -> None:
    contract = dict(IntegerPlaneEmitterPolicy().compile_contract())
    contract[field] = replacement
    with pytest.raises(IntegerPlaneEmitterCheckpointError, match=match):
        _checkpoint(policy_contract=contract)


@pytest.mark.parametrize(
    ("change", "match"),
    [
        ({"config_sha256": "a" * 64}, "config_sha256 must equal"),
        ({"fixed_capacity_signature": "a" * 64}, "fixed_capacity_signature must equal"),
        (
            {
                "basis_id": BasisMode.SIGN_FIXED_U4_PAIR_MARGIN.value,
                "policy_contract": IntegerPlaneEmitterPolicy().compile_contract(),
            },
            "basis_id differs",
        ),
    ],
)
def test_checkpoint_config_capacity_and_basis_cross_bindings(change: dict[str, object], match: str) -> None:
    change = dict(change)
    if "policy_contract" in change:
        contract = change["policy_contract"]
        assert isinstance(contract, dict)
        change.setdefault("config_sha256", contract["policy_sha256"])
        change.setdefault("fixed_capacity_signature", contract["capacity_signature"])
    with pytest.raises(IntegerPlaneEmitterCheckpointError, match=match):
        _checkpoint(**change)


@pytest.mark.parametrize(
    ("state_field", "tensor_field", "mutation", "match"),
    [
        ("live_residual_parameters", "pair_plane_codes", {"shape": [12, 1, 4]}, "shape"),
        ("live_residual_parameters", "shared_rgb_head", {"shape": [4, 2]}, "shape"),
        ("ema_shadow", "pair_plane_codes", {"dtype": "float64"}, "float32"),
        ("ema_shadow", "shared_rgb_head", {"data": []}, "data length"),
    ],
)
def test_checkpoint_residual_shapes_and_dtype_are_exact(
    state_field: str,
    tensor_field: str,
    mutation: dict[str, object],
    match: str,
) -> None:
    state = _residual_state()
    tensor = dict(state[tensor_field])  # type: ignore[arg-type]
    tensor.update(mutation)
    state[tensor_field] = tensor
    with pytest.raises(IntegerPlaneEmitterCheckpointError, match=match):
        _checkpoint(**{state_field: state})


def test_checkpoint_defensively_copies_nested_state() -> None:
    contract = IntegerPlaneEmitterPolicy().compile_contract()
    live = _residual_state()
    checkpoint = _checkpoint(policy_contract=contract, live_residual_parameters=live)
    encoded = checkpoint.to_bytes()
    contract["basis"] = BasisMode.SIGN_FIXED_U4_PAIR_MARGIN.value
    live["pair_plane_codes"]["data"][0] = 99.0  # type: ignore[index]
    assert checkpoint.to_bytes() == encoded


def test_checkpoint_optimizer_step_is_required_and_nonnegative() -> None:
    with pytest.raises(IntegerPlaneEmitterCheckpointError, match=r"optimizer_state\.step"):
        _checkpoint(optimizer_state={"moments": [0.1]})
    with pytest.raises(IntegerPlaneEmitterCheckpointError, match=r"optimizer_state\.step"):
        _checkpoint(optimizer_state={"step": -1})
