from __future__ import annotations

import itertools
import json
import sys
from collections import OrderedDict
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import tools.profile_v10_uint8_lattice_n600 as profiler_module
from tac.admission_guard import BYPASS_OVERRIDE_ENV, GOVERNED_MARKER_ENV, mark_admitted_env
from tac.optimization.uint8_lattice_feasibility import DisjointResizeOperator
from tac.optimization.uint8_lattice_profile import (
    LatticeProfileError,
    NoOpPosePlugin,
    PoseFilterDecision,
    ProfileStatus,
    SignedResidualCostModel,
    StreamingProfileAggregator,
    build_rd_row,
    candidate_stream_accounting,
    encode_candidate_stream,
    noncorner_positive_control,
    profile_cache_key,
    profile_integer_block,
    vectorized_source_witness_bounds,
)
from tac.witness_control.segnet_head_feature_cache import atomic_json as _atomic_json
from tools.profile_v10_uint8_lattice_n600 import (
    BOUNDS_MODE,
    CREATION_STAGING_SUFFIX,
    ENUMERATED_MODE,
    FINAL_RECEIPT_SCHEMA,
    IDENTITY_NAME,
    LOWER_BOUND_METHOD,
    OUTPUT_CERTIFICATION_NAME,
    PROGRESS_SCHEMA,
    RECEIPT_NAME,
    REPO_ROOT,
    STAGE_RECEIPT_SCHEMA,
    STAGING_SCRATCH_NAME,
    TIMING_CUSTODY_LABEL,
    ProfilerError,
    _assert_real_governed_admission,
    _decode_frame_candidate_payload,
    _derive_partition_custody,
    _expected_derivation,
    _expected_positive_control,
    _feature_binding,
    _FrameSemanticReplay,
    _identity,
    _parse_args,
    _prepared_stage_path,
    _profile_frame_semantics,
    _receiver_support_union,
    _reconstruct_rd_row,
    _resume_from_stage_chain,
    _score_frame_artifacts,
    _selection_custody,
    _source_block_geometry,
    _source_seed_candidate,
    _stage_payload,
    _stage_timing,
    _terminal_custody,
    _timing_summary,
    _validate_compact_state,
    _validate_final_receipt,
    _validate_output_certification,
    _validate_resume_root,
    _validate_source_seed_receiver,
    _validate_stage_receipt,
    run_profile,
)
from tools.profile_v10_uint8_lattice_n600 import (
    _atomic_stage as _production_atomic_stage,
)
from tools.profile_v10_uint8_lattice_n600 import (
    _initialize_fresh_output as _production_initialize_fresh_output,
)

_PARTITION_OPERATOR = DisjointResizeOperator.build(camera_h=2, camera_w=2, scorer_h=1, scorer_w=1)


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    """Authorize fixture rewrites with the exact consumer state they replace."""
    expected_prior = (path.read_bytes(),) if path.exists() else ()
    _atomic_json(path, payload, expected_prior_payloads=expected_prior)


def _atomic_stage(path: Path, payload: bytes) -> None:
    """Commit fixture stages with the same immutable attempt custody as production."""

    receipt, _candidate_payload = profiler_module._parse_stage_payload(payload)
    identity_sha256 = receipt["identity_sha256"]
    progress_path = path.parent.parent / profiler_module.PROGRESS_NAME
    exact_rebuild_argv = (
        json.loads(progress_path.read_text(encoding="utf-8"))["exact_argv"]
        if progress_path.exists()
        else ["python", "profile.py"]
    )
    _production_atomic_stage(
        path,
        payload,
        identity_sha256=identity_sha256,
        exact_rebuild_argv=exact_rebuild_argv,
    )


def _write_stage_attempt_intent(
    final: Path,
    intended_payload: bytes,
    *,
    identity_sha256: str,
    exact_rebuild_argv: list[str] | None = None,
    attempt: int | None = None,
) -> tuple[Path, str]:
    """Persist the production transaction-shaped intent for crash fixtures."""

    argv = exact_rebuild_argv
    if argv is None:
        progress_path = final.parent.parent / profiler_module.PROGRESS_NAME
        argv = json.loads(progress_path.read_text(encoding="utf-8"))["exact_argv"]
    if attempt is None:
        attempt = profiler_module._next_stage_attempt(final)
    profiler_module._ensure_stage_attempt_directory(
        final.parent,
        identity_sha256=identity_sha256,
        frame=int(final.stem.removeprefix("frame_")),
        attempt=attempt,
        authorize_mutation=lambda: None,
    )
    intent = profiler_module._stage_intent_path(
        final,
        identity_sha256=identity_sha256,
        payload_bytes=len(intended_payload),
        payload_sha256=sha256(intended_payload).hexdigest(),
        attempt=attempt,
    )
    record = profiler_module._stage_attempt_transaction(
        final_path=final,
        intent_path=intent,
        identity_sha256=identity_sha256,
        frame=int(final.stem.removeprefix("frame_")),
        attempt=attempt,
        intended_payload=intended_payload,
        exact_rebuild_argv=argv,
    )
    record_payload = profiler_module.canonical_json_bytes(record) + b"\n"
    _atomic_json(intent, record)
    return intent, sha256(record_payload).hexdigest()


def _initialize_fresh_output(
    output_root: Path,
    *,
    identity_hash: str,
    storage_preflight: dict[str, object],
    exact_argv: list[str],
) -> tuple[Path, Path]:
    return _production_initialize_fresh_output(
        output_root,
        identity_hash=identity_hash,
        storage_preflight=storage_preflight,
        exact_argv=exact_argv,
        allow_uncertified_test_output=True,
    )


def _wrapped_frame_payload(rows: tuple[tuple[tuple[int, ...] | None, ...], ...]) -> bytes:
    payload = bytearray(len(rows).to_bytes(4, "little"))
    for candidates in rows:
        row = encode_candidate_stream(candidates)
        payload.extend(len(row).to_bytes(4, "little"))
        payload.extend(row)
    return bytes(payload)


def _receipt_timing(total_blocks: int = 3) -> dict[str, object]:
    return _stage_timing(
        wall_seconds=2.0,
        total_blocks=total_blocks,
        peak_rss_bytes=0,
    )


def _receipt_selection(
    receipt: dict[str, object],
    *,
    seed_source_witness: bool = False,
    receiver_closed: bool = False,
) -> dict[str, object]:
    counters = receipt["counters"]
    assert isinstance(counters, dict)
    return _selection_custody(
        mode=str(receipt["mode"]),
        counters=counters,
        seed_source_witness=seed_source_witness,
        receiver_closed=receiver_closed,
    )


def _replay_from_receipt(
    receipt: dict[str, object],
    candidate_payload: bytes,
) -> _FrameSemanticReplay:
    return _FrameSemanticReplay(
        partition_custody=json.loads(json.dumps(receipt["partition_custody"])),
        aggregate_state=json.loads(json.dumps(receipt["aggregate_delta_state"])),
        candidate_payload=candidate_payload,
        counters=json.loads(json.dumps(receipt["counters"])),
        selection_custody=json.loads(json.dumps(receipt["selection_custody"])),
    )


def _validate_against_replay(
    receipt: dict[str, object],
    candidate_payload: bytes,
    replay: _FrameSemanticReplay,
) -> StreamingProfileAggregator:
    return _validate_stage_receipt(
        receipt,
        candidate_payload,
        expected_partition_custody=replay.partition_custody,
        expected_aggregate_state=replay.aggregate_state,
        expected_candidate_payload=replay.candidate_payload,
        expected_counters=replay.counters,
        expected_selection_custody=replay.selection_custody,
    )


def _tiny_seed_artifacts(
    *,
    source: np.ndarray | None = None,
    max_nodes: int = 1,
) -> tuple[DisjointResizeOperator, np.ndarray, profiler_module._FrameProfileArtifacts]:
    operator = DisjointResizeOperator.build(camera_h=2, camera_w=2, scorer_h=1, scorer_w=1)
    source_frame = (
        np.arange(12, dtype=np.uint8).reshape(2, 2, 3) if source is None else np.asarray(source, dtype=np.uint8)
    )
    logits = np.zeros((5, 1, 1), dtype=np.float32)
    logits[1, 0, 0] = np.float32(2.0)
    selector = SignedResidualCostModel()
    plugin = NoOpPosePlugin()
    artifacts = _profile_frame_semantics(
        0,
        source_frame=source_frame,
        live_logits=logits,
        operator=operator,
        mode=ENUMERATED_MODE,
        seed_source_witness=True,
        max_nodes=max_nodes,
        time_limit_seconds_per_block=None,
        fragile_margin=0.0,
        selector=selector,
        selector_identity=selector.identity,
        pose_plugin=plugin,
        pose_plugin_identity=plugin.identity,
        reuse_cache_entries=8,
        reuse=OrderedDict(),
        build_selected_frame=True,
    )
    return operator, source_frame, artifacts


def _receipt_from_artifacts(
    identity_hash: str,
    artifacts: profiler_module._FrameProfileArtifacts,
) -> dict[str, object]:
    return {
        "schema": STAGE_RECEIPT_SCHEMA,
        "identity_sha256": identity_hash,
        "previous_stage_sha256": identity_hash,
        "frame": 0,
        "partition_custody": artifacts.replay.partition_custody,
        "mode": ENUMERATED_MODE,
        "lower_bound_method": None,
        "derivation": "KNOWN_SOURCE_WITNESS_SEEDED_CHEAPEST_SEEN",
        "aggregate_delta_state": artifacts.replay.aggregate_state,
        "counters": artifacts.replay.counters,
        "selection_custody": artifacts.replay.selection_custody,
        "candidate_payload_bytes": len(artifacts.candidate_payload),
        "candidate_payload_sha256": sha256(artifacts.candidate_payload).hexdigest(),
        "timing": _receipt_timing(artifacts.counters["total_blocks"]),
        "scope": {
            "frame_indices": [0],
            "rgb_channel_blocks": artifacts.counters["total_blocks"],
            "scorer_pixels": int(artifacts.labels.size),
            "node_cap": 1,
            "selection_label": artifacts.replay.selection_custody["selection_label"],
            "receiver_non_closure": artifacts.replay.selection_custody["receiver_non_closure"],
            "scope_extrapolation": "NONE_EXACT_FRAME_INDICES_ONLY",
        },
    }


def _partition_custody(
    frame: int,
    *,
    target_class: int = 1,
    fragile: bool = False,
) -> dict[str, object]:
    logits = np.zeros((5, 1, 1), dtype=np.float32)
    logits[target_class, 0, 0] = np.float32(2.0)
    return _derive_partition_custody(
        frame,
        live_logits=logits,
        source_frame=np.zeros((2, 2, 3), dtype=np.uint8),
        operator=_PARTITION_OPERATOR,
        fragile_margin=3.0 if fragile else 0.0,
    )


def _progress_pointer(
    identity_hash: str,
    *,
    next_frame: int,
    head: str,
    status: str = "partial",
) -> dict[str, object]:
    return {
        "schema": PROGRESS_SCHEMA,
        "identity_sha256": identity_hash,
        "status": status,
        "next_frame": next_frame,
        "stage_chain_head_sha256": head,
        "storage_preflight": {"PASS": True},
        "exact_argv": ["python", "profile.py"],
    }


def _synthetic_fresh_argv() -> list[str]:
    return [sys.executable, str(Path(profiler_module.__file__).resolve()), "--synthetic-child"]


def _synthetic_parent_attestation(
    *,
    parent_pid: int = 4242,
    resume: bool = False,
) -> dict[str, object]:
    def row(path: Path) -> dict[str, object]:
        resolved = path.resolve()
        return {
            "path": str(resolved),
            "bytes": resolved.stat().st_size,
            "sha256": sha256(resolved.read_bytes()).hexdigest(),
        }

    fresh_argv = _synthetic_fresh_argv()
    child_argv = [*fresh_argv, "--resume"] if resume else fresh_argv
    parent_argv = [
        sys.executable,
        str(REPO_ROOT / "tools/safe_run.py"),
        "--rss-mb",
        "512",
        "--timeout",
        "60",
        "--",
        *child_argv,
    ]
    return {
        "schema": "governed_safe_run_parent_attestation.v1",
        "attestation_scope": "DIRECT_PARENT_COMMAND_AT_CHILD_START_NOT_COMPLETED_STATUS",
        "parent_pid": parent_pid,
        "parent_python_executable": str(Path(sys.executable).resolve()),
        "parent_exact_argv": parent_argv,
        "child_exact_argv": child_argv,
        "outer_resource_caps": {"rss_cap_mb": 512, "timeout_seconds": 60.0},
        "governed_marker_present": True,
        "admission_bypass_present": False,
        "completed_safe_run_status_receipt": None,
        "source_custody": {
            "governed_profile_admission": row(REPO_ROOT / "src/tac/governed_profile_admission.py"),
            "safe_run": row(REPO_ROOT / "tools/safe_run.py"),
            "admission_guard": row(REPO_ROOT / "src/tac/admission_guard.py"),
        },
    }


def _certified_identity(
    *,
    output_root: Path,
    mode: str = BOUNDS_MODE,
    seed_source_witness: bool = False,
    creation_preflight: dict[str, object] | None = None,
) -> dict[str, object]:
    bound_preflight = _certified_preflight(output_root) if creation_preflight is None else creation_preflight
    return {
        "schema": PROGRESS_SCHEMA,
        "repository": {
            "git_head": "a" * 40,
            "worktree_cleanliness_required": False,
            "source_byte_hashes_primary": True,
        },
        "sources": {"synthetic_source": {"path": "/synthetic", "bytes": 1, "sha256": "b" * 64}},
        "feature_cache_binding": {
            "synthetic": True,
            "committed_prefix_sha256": "c" * 64,
        },
        "creation_storage_identity": profiler_module._creation_storage_identity_from_preflight(
            bound_preflight,
            expected_output_root=output_root.resolve(),
        ),
        "config": {
            "profiled_frame_limit": 1,
            "mode": mode,
            "seed_source_witness": seed_source_witness,
            "max_nodes": 1,
            "score_segnet": False,
            "requested_outer_governor_limits": {
                "rss_cap_mb": 512,
                "timeout_seconds": 60,
                "profiler_self_enforced": False,
                "scope": "REQUESTED_OUTER_GOVERNOR_LIMITS_METADATA_ONLY_NOT_ENFORCEMENT_RECEIPT",
                "required_launcher": "tools/safe_run.py process-group/system-memory governor",
            },
        },
    }


def _certified_preflight(root: Path) -> dict[str, object]:
    resolved = root.resolve()
    return {
        "waterfall_order": ["/Volumes/VertigoDataTier/pact", "/Volumes/APDataStore/pact"],
        "existing_approved_roots": [],
        "selection_scope": profiler_module.FRESH_STORAGE_SELECTION_SCOPE,
        "selected_root": str(resolved),
        "filesystem_anchor": str(resolved.parent),
        "free_bytes_before": 10_000_000,
        "required_free_bytes": 1_000_000,
        "allow_local_output_for_tests": True,
        "PASS": True,
    }


def _certified_storage_identity(root: Path) -> dict[str, object]:
    return profiler_module._creation_storage_identity_from_preflight(
        _certified_preflight(root),
        expected_output_root=root.resolve(),
    )


def _create_certified_test_root(
    root: Path,
    *,
    identity: dict[str, object] | None = None,
    exact_argv: list[str] | None = None,
) -> tuple[dict[str, object], str, list[str], dict[str, object], Path, Path]:
    value = _certified_identity(output_root=root) if identity is None else identity
    argv = (
        ["python", "profile.py", "--rss-cap-mb", "512", "--timeout-seconds", "60"] if exact_argv is None else exact_argv
    )
    identity_hash = sha256(profiler_module.canonical_json_bytes(value)).hexdigest()
    preflight = _certified_preflight(root)
    stages, progress = _production_initialize_fresh_output(
        root,
        identity_hash=identity_hash,
        storage_preflight=preflight,
        exact_argv=argv,
        identity=value,
    )
    return value, identity_hash, argv, preflight, stages, progress


def _commit_certified_bounds_stage(
    root: Path,
    *,
    identity_hash: str,
    stages: Path,
    progress: Path,
) -> dict[str, object]:
    receipt = _bounds_stage_receipt(identity_hash, frame=0, previous_hash=identity_hash)
    stage = stages / "frame_0000.bin"
    _atomic_stage(stage, _stage_payload(receipt, b""))
    pointer = json.loads(progress.read_text(encoding="utf-8"))
    pointer.update(
        {
            "next_frame": 1,
            "stage_chain_head_sha256": sha256(stage.read_bytes()).hexdigest(),
        }
    )
    atomic_json(progress, pointer)
    assert root == stages.parent
    return receipt


def _clean_room_bounds_receipt(
    root: Path,
    *,
    identity: dict[str, object],
    exact_argv: list[str],
) -> dict[str, object]:
    custody, stage_receipts = _terminal_custody(
        root,
        expected_identity=identity,
        expected_rebuild_argv=exact_argv,
    )
    aggregate = StreamingProfileAggregator(
        n_classes=5,
        named_strata=("boundary_annulus", "fragile", "degenerate"),
    )
    totals = dict.fromkeys(profiler_module.COUNTER_NAMES, 0)
    for stage_receipt in stage_receipts:
        aggregate.merge(StreamingProfileAggregator.from_state(stage_receipt["aggregate_delta_state"]))
        for name in profiler_module.COUNTER_NAMES:
            totals[name] += stage_receipt["counters"][name]
    mode = identity["config"]["mode"]
    seed_source_witness = identity["config"]["seed_source_witness"]
    selection = _selection_custody(
        mode=mode,
        counters=totals,
        seed_source_witness=seed_source_witness,
        receiver_closed=(
            bool(stage_receipts)
            and all(not stage_receipt["selection_custody"]["receiver_non_closure"] for stage_receipt in stage_receipts)
        ),
    )
    rd_row = _reconstruct_rd_row(
        totals=totals,
        stage_receipts=stage_receipts,
        config=identity["config"],
        stream_accounting=custody["stream_accounting"],
        feature_cache_binding=identity["feature_cache_binding"],
    )
    return {
        "schema": FINAL_RECEIPT_SCHEMA,
        "status": "partial_prefix",
        "scope_label": "HASH_VALID_EXPLICIT_PREFIX",
        "mode": mode,
        "lower_bound_method": LOWER_BOUND_METHOD if mode == BOUNDS_MODE else None,
        "derivation": _expected_derivation(
            mode=mode,
            seed_source_witness=seed_source_witness,
        ),
        "frames_profiled": 1,
        "expected_frames": 600,
        "profiled_frame_indices": [0],
        "scope_extrapolation": "NONE_EXACT_FRAME_INDICES_ONLY",
        "aggregate": aggregate.summary(),
        "rd_row": rd_row,
        "counters_rebuilt_from_hashed_stage_receipts": totals,
        "feature_cache_binding": identity["feature_cache_binding"],
        "claims": {
            "exact_count_claim": selection["exact_count_claim"],
            "min_description_claim": False,
            "selection_globally_exact": selection["selection_globally_exact"],
            "selection_label": selection["selection_label"],
            "d_seg_claim": False,
            "candidate_stream_emitted": mode == ENUMERATED_MODE,
            "receiver_non_closure": selection["receiver_non_closure"],
            "per_block_selector_minimum_proved": selection["per_block_selector_minimum_proved"],
            "global_compressed_stream_minimum_claim": False,
        },
        "positive_control": _expected_positive_control(mode=mode),
        "authority": {
            "score_authority": False,
            "promotion_eligible": False,
            "pose_bank_wired": False,
            "factor10_solved": False,
            "global_compressed_stream_minimum_claim": False,
        },
        "identity_sha256": custody["identity_sha256"],
        "git_head": identity["repository"]["git_head"],
        "exact_rebuild_argv": exact_argv,
        "requested_outer_governor_limits": identity["config"]["requested_outer_governor_limits"],
        "custody": custody,
        "timing_summary": _timing_summary(
            stage_receipts,
            terminal_stage_chain_sha256=custody["terminal_stage_chain_sha256"],
        ),
    }


def _brute_count(coefficients: tuple[int, ...], target: int) -> int:
    return sum(
        sum(c * x for c, x in zip(coefficients, candidate, strict=True)) == target
        for candidate in itertools.product(range(256), repeat=len(coefficients))
    )


def _brute_count_small_target(coefficients: tuple[int, ...], target: int) -> int:
    ranges = [range(min(255, target // coefficient) + 1) for coefficient in coefficients]
    return sum(
        sum(c * x for c, x in zip(coefficients, candidate, strict=True)) == target
        for candidate in itertools.product(*ranges)
    )


def test_exhaustive_cardinality_matches_brute_force() -> None:
    coefficients = (2, 3)
    target = 500
    truth = _brute_count(coefficients, target)
    result = profile_integer_block(
        coefficients,
        denominator=sum(coefficients),
        target_integer=target,
        max_nodes=10_000,
    )
    assert result.status is ProfileStatus.EXACT
    assert result.exhaustive is True
    assert result.exact_cardinality == truth
    assert result.cardinality_lower_bound == truth
    assert result.cardinality_upper_bound == truth


def test_budget_bounds_enclose_truth_and_never_claim_infeasible() -> None:
    coefficients = (2, 3)
    target = 500
    truth = _brute_count(coefficients, target)
    result = profile_integer_block(
        coefficients,
        denominator=sum(coefficients),
        target_integer=target,
        max_nodes=1,
    )
    assert result.status is ProfileStatus.BOUNDED_NODE_CAP
    assert result.exhaustive is False
    assert result.exact_cardinality is None
    assert result.proved_infeasible is False
    assert result.cardinality_lower_bound <= truth <= result.cardinality_upper_bound


def test_exhaustive_gcd_impossibility_can_prove_zero() -> None:
    result = profile_integer_block((2, 4), 6, 3, max_nodes=10)
    assert result.status is ProfileStatus.INFEASIBLE_EXHAUSTIVE
    assert result.exact_cardinality == 0
    assert result.proved_infeasible is True


def test_strictly_cheaper_description_winner_differs_from_minimum_norm() -> None:
    # u0 + u1 = 130.  Minimum norm is (65,65), but under a public zero
    # predictor (0,130) costs three bytes while (65,65) costs four.
    model = SignedResidualCostModel(predictor=0)
    result = profile_integer_block(
        (1, 1),
        2,
        130,
        cost_model=model,
        max_nodes=10_000,
    )
    assert result.exact_cardinality == 131
    assert result.selected_candidate == (0, 130)
    assert result.selected_candidate != (65, 65)
    assert model.cost_bits(result.selected_candidate) < model.cost_bits((65, 65))
    assert sum(result.selected_candidate) == 130
    assert result.selection_globally_exact is True


class _TubePlugin:
    identity = "synthetic_pose_tube.u0_ge_8.v1"

    def evaluate(self, candidate: tuple[int, ...]) -> PoseFilterDecision:
        return PoseFilterDecision(candidate[0] >= 8, 1, {"u0": candidate[0]})


def test_pose_noop_parity_and_tube_shrinks_intersection_deterministically() -> None:
    base = profile_integer_block((1, 1), 2, 10, pose_plugin=NoOpPosePlugin(), max_nodes=1000)
    implicit_noop = profile_integer_block((1, 1), 2, 10, max_nodes=1000)
    tube_first = profile_integer_block((1, 1), 2, 10, pose_plugin=_TubePlugin(), max_nodes=1000)
    tube_second = profile_integer_block((1, 1), 2, 10, pose_plugin=_TubePlugin(), max_nodes=1000)
    assert base == implicit_noop
    assert base.exact_cardinality == 11
    assert tube_first.exact_cardinality == 3
    assert tube_first.selected_candidate == tube_second.selected_candidate
    assert tube_first.selected_cost_bits == tube_second.selected_cost_bits


class _RaisingPlugin:
    identity = "synthetic_pose_raises.v1"

    def evaluate(self, candidate: tuple[int, ...]) -> PoseFilterDecision:
        raise RuntimeError(f"budget at {candidate}")


class _MalformedPlugin:
    identity = "synthetic_pose_nonfinite.v1"

    def evaluate(self, candidate: tuple[int, ...]) -> PoseFilterDecision:
        return PoseFilterDecision(True, 0, {"bad": float("nan")})


@pytest.mark.parametrize("plugin", [_RaisingPlugin(), _MalformedPlugin()])
def test_pose_plugin_error_is_unknown_not_infeasible(plugin: object) -> None:
    result = profile_integer_block((1, 1), 2, 10, pose_plugin=plugin, max_nodes=1000)  # type: ignore[arg-type]
    assert result.status is ProfileStatus.PLUGIN_ERROR_UNKNOWN
    assert result.exhaustive is False
    assert result.proved_infeasible is False
    assert result.exact_cardinality is None
    assert result.cardinality_upper_bound >= result.cardinality_lower_bound
    assert result.plugin_error


def test_noncorner_control_extinguishes_corner_only_false_certificate() -> None:
    fixture = noncorner_positive_control()
    assert fixture["witness_satisfies"] is True
    assert fixture["any_corner_satisfies"] is False
    result = profile_integer_block(
        fixture["coefficients"],
        sum(fixture["coefficients"]),
        fixture["target_integer"],
        max_nodes=1000,
    )
    assert result.exact_cardinality and result.exact_cardinality > 0


def test_reuse_key_contains_all_required_identities() -> None:
    base = {
        "coefficients": (1, 2),
        "denominator": 3,
        "target_integer": 7,
        "selector_identity": "selector.a",
        "pose_plugin_identity": "pose.a",
    }
    keys = {
        profile_cache_key(**base),
        profile_cache_key(**{**base, "coefficients": (2, 1)}),
        profile_cache_key(**{**base, "denominator": 4}),
        profile_cache_key(**{**base, "target_integer": 8}),
        profile_cache_key(**{**base, "selector_identity": "selector.b"}),
        profile_cache_key(**{**base, "pose_plugin_identity": "pose.b"}),
    }
    assert len(keys) == 6


def test_actual_candidate_stream_reports_raw_zlib_brotli_and_headers() -> None:
    payload = encode_candidate_stream(
        ((0, 10), None, (255, 1)),
        cost_model=SignedResidualCostModel(),
    )
    accounting = candidate_stream_accounting(payload)
    assert accounting["raw"]["bytes"] == len(payload)
    assert accounting["zlib_level9"]["bytes"] > 0
    assert accounting["brotli_quality11"]["bytes"] > 0
    assert accounting["headers_and_termination_included"] is True
    assert accounting["order0_entropy"]["label"] == "ORDER0_IID_PLUGIN_IDEAL_LENGTH_ESTIMATE_NOT_UNIVERSAL_LOWER_BOUND"


def test_order0_plugin_estimate_is_not_a_context_compression_lower_bound() -> None:
    payload = encode_candidate_stream(tuple((0, 0, 0, 0) for _ in range(4096)))
    accounting = candidate_stream_accounting(payload)
    assert accounting["zlib_level9"]["bytes"] < accounting["order0_entropy"]["rounded_up_bytes"]
    assert "LOWER_BOUND" not in accounting["order0_entropy"]["label"].removesuffix("NOT_UNIVERSAL_LOWER_BOUND")


def test_aggregation_distinguishes_pixels_from_rgb_blocks_and_class_strata() -> None:
    exact = profile_integer_block((1, 1), 2, 10, max_nodes=1000)
    bounded = profile_integer_block((1, 1), 2, 10, max_nodes=1)
    aggregator = StreamingProfileAggregator(
        n_classes=2,
        named_strata=("fragile", "degenerate"),
    )
    aggregator.add_pixel(
        target_class=1,
        channel_results=(exact, exact, bounded),
        strata=("fragile",),
    )
    summary = aggregator.summary()
    assert summary["global"]["scorer_pixels"] == 1
    assert summary["global"]["rgb_channel_blocks"] == 3
    assert summary["global"]["exact_blocks"] == 2
    assert summary["global"]["bounded_blocks"] == 1
    assert summary["per_class"]["1"]["scorer_pixels"] == 1
    assert summary["named_strata"]["fragile"]["scorer_pixels"] == 1
    assert summary["named_strata"]["degenerate"]["scorer_pixels"] == 0
    restored = StreamingProfileAggregator.from_state(aggregator.state())
    assert restored.summary() == summary


def test_rd_row_refuses_margin_proxy_and_needs_scorer_custody_for_dseg() -> None:
    payload = encode_candidate_stream(((0, 10),))
    accounting = candidate_stream_accounting(payload)
    unscored = build_rd_row(
        selected_block_count=1,
        total_block_count=1,
        stream_accounting=accounting,
        axis="NO_VERDICT_SCORER_CUSTODY",
        cache_scope="synthetic",
        receiver_scope="test",
    )
    assert unscored["scorer_custody"] == "NO_VERDICT_SCORER_CUSTODY"
    assert unscored["d_seg"] is None
    scored = build_rd_row(
        selected_block_count=1,
        total_block_count=1,
        stream_accounting=accounting,
        axis="[macOS-CPU advisory]",
        cache_scope="synthetic",
        receiver_scope="test",
        mismatch_count=1,
        scorer_pixel_count=4,
        rate_scope_frames=(0,),
        scorer_scope_frames=(0,),
    )
    assert scored["d_seg"] == 0.25

    with pytest.raises(LatticeProfileError, match="mixed scorer/rate scope"):
        build_rd_row(
            selected_block_count=1,
            total_block_count=1,
            stream_accounting=accounting,
            axis="[macOS-CPU advisory]",
            cache_scope="synthetic",
            receiver_scope="test",
            mismatch_count=1,
            scorer_pixel_count=4,
            rate_scope_frames=(0, 1),
            scorer_scope_frames=(0,),
        )


def test_profile_refuses_nonfinite_or_coercive_inputs() -> None:
    with pytest.raises(LatticeProfileError):
        profile_integer_block((True, 1), 2, 1)
    with pytest.raises(LatticeProfileError):
        profile_integer_block((1, 1), 2, 1, time_limit_seconds=float("nan"))
    with pytest.raises(LatticeProfileError, match="must be an integer"):
        encode_candidate_stream(((True, 1),))


def test_vectorized_source_bounds_enclose_bruteforce_and_count_once() -> None:
    coefficients = np.array([[2, 3], [4, 6], [1, 1]], dtype=np.int64)
    witnesses = np.array(
        [
            [[1, 2], [3, 4], [5, 6]],
            [[2, 2], [4, 4], [6, 6]],
            [[7, 8], [9, 10], [11, 12]],
        ],
        dtype=np.uint8,
    )
    targets = np.sum(coefficients[:, None, :] * witnesses, axis=-1)
    bounds = vectorized_source_witness_bounds(
        coefficients[:, None, :],
        targets,
        witnesses,
    )
    assert bounds.witness_verified_blocks == targets.size
    assert bounds.lower_bound_method == LOWER_BOUND_METHOD
    for row in range(targets.shape[0]):
        for channel in range(targets.shape[1]):
            truth = _brute_count(tuple(coefficients[row]), int(targets[row, channel]))
            assert 1 <= bounds.cardinality_lower_bound[row, channel] <= truth
            assert truth <= bounds.cardinality_upper_bound[row, channel]

    aggregator = StreamingProfileAggregator(
        n_classes=2,
        named_strata=("fragile",),
    )
    aggregator.add_bounds_batch(
        target_classes=np.array([0, 1, 1]),
        lower_bounds=bounds.cardinality_lower_bound,
        upper_bounds=bounds.cardinality_upper_bound,
        strata={"fragile": np.array([False, True, False])},
    )
    summary = aggregator.summary()
    assert summary["global"]["scorer_pixels"] == 3
    assert summary["global"]["rgb_channel_blocks"] == 9
    assert summary["global"]["bounded_blocks"] == 9
    assert summary["global"]["exact_blocks"] == 0
    tampered = witnesses.copy()
    tampered[0, 0, 0] += 1
    with pytest.raises(LatticeProfileError, match="violates exact integer equation"):
        vectorized_source_witness_bounds(coefficients[:, None, :], targets, tampered)


def test_vectorized_bounds_use_same_operator_targets_on_small_geometry() -> None:
    operator = DisjointResizeOperator.build(
        camera_h=2,
        camera_w=1,
        scorer_h=1,
        scorer_w=1,
    )
    source = np.array([[[7, 13, 19]], [[23, 29, 31]]], dtype=np.uint8)
    coefficients, targets, witnesses = _source_block_geometry(operator, source)
    bounds = vectorized_source_witness_bounds(coefficients, targets, witnesses)
    coefficient_tuple = tuple(int(value) for value in coefficients[0, 0, 0])
    for channel in range(3):
        truth = _brute_count(coefficient_tuple, int(targets[0, 0, channel]))
        assert 1 <= bounds.cardinality_lower_bound[0, 0, channel] <= truth
        assert truth <= bounds.cardinality_upper_bound[0, 0, channel]
    assert bounds.witness_verified_blocks == 3


def test_four_tap_pair_fiber_lower_bound_is_certified_by_small_target_truth() -> None:
    coefficients = np.array([[1, 2, 3, 4]], dtype=np.int64)
    witnesses = np.array([[2, 1, 1, 1]], dtype=np.uint8)
    targets = np.sum(coefficients * witnesses, axis=-1)
    truth = _brute_count_small_target(tuple(int(value) for value in coefficients[0]), int(targets[0]))
    bounds = vectorized_source_witness_bounds(coefficients, targets, witnesses)
    lower = int(bounds.cardinality_lower_bound[0])
    assert lower > 1
    assert lower <= truth <= int(bounds.cardinality_upper_bound[0])


def test_pair_fiber_candidates_preserve_equation_and_product_is_injective() -> None:
    coefficients = (2, 3, 5, 7)
    source = (9, 11, 13, 15)
    target = sum(c * x for c, x in zip(coefficients, source, strict=True))
    matchings = (((0, 1), (2, 3)), ((0, 2), (1, 3)), ((0, 3), (1, 2)))
    matching_counts: list[int] = []
    for matching in matchings:
        pair_moves: list[list[tuple[int, int]]] = []
        for first, second in matching:
            divisor = int(np.gcd(coefficients[first], coefficients[second]))
            first_step = coefficients[second] // divisor
            second_step = coefficients[first] // divisor
            lower = max(
                -(source[first] // first_step),
                -((-(source[second] - 255)) // second_step),
            )
            upper = min(
                (255 - source[first]) // first_step,
                source[second] // second_step,
            )
            moves: list[tuple[int, int]] = []
            for offset in range(lower, upper + 1):
                moved = (
                    source[first] + offset * first_step,
                    source[second] - offset * second_step,
                )
                assert all(0 <= value <= 255 for value in moved)
                assert coefficients[first] * moved[0] + coefficients[second] * moved[1] == (
                    coefficients[first] * source[first] + coefficients[second] * source[second]
                )
                moves.append(moved)
            pair_moves.append(moves)
        emitted: set[tuple[int, ...]] = set()
        for first_move, second_move in itertools.product(*pair_moves):
            candidate = list(source)
            for pair, moved in zip(matching, (first_move, second_move), strict=True):
                candidate[pair[0]], candidate[pair[1]] = moved
            candidate_tuple = tuple(candidate)
            assert sum(c * x for c, x in zip(coefficients, candidate_tuple, strict=True)) == target
            emitted.add(candidate_tuple)
        assert len(emitted) == len(pair_moves[0]) * len(pair_moves[1])
        matching_counts.append(len(emitted))
    bounds = vectorized_source_witness_bounds(
        np.asarray([coefficients], dtype=np.int64),
        np.asarray([target], dtype=np.int64),
        np.asarray([source], dtype=np.uint8),
    )
    assert int(bounds.cardinality_lower_bound[0]) == max(matching_counts)


def test_vectorized_bounds_refuse_uint64_wrap_and_unsafe_accumulation_before_cast() -> None:
    with pytest.raises(LatticeProfileError, match="outside the int64 domain"):
        vectorized_source_witness_bounds(
            np.array([[np.iinfo(np.uint64).max]], dtype=np.uint64),
            np.array([0], dtype=np.uint64),
            np.array([[0]], dtype=np.uint64),
        )
    with pytest.raises(LatticeProfileError, match="safe int64 accumulation"):
        vectorized_source_witness_bounds(
            np.array([[np.iinfo(np.int64).max]], dtype=np.uint64),
            np.array([0], dtype=np.uint64),
            np.array([[0]], dtype=np.uint64),
        )
    with pytest.raises(LatticeProfileError, match="source witnesses must stay inside uint8"):
        vectorized_source_witness_bounds(
            np.array([[1]], dtype=np.uint64),
            np.array([256], dtype=np.uint64),
            np.array([[256]], dtype=np.uint64),
        )


def test_source_block_geometry_preserves_canonical_four_tap_three_channel_order() -> None:
    operator = DisjointResizeOperator.build(camera_h=2, camera_w=2, scorer_h=1, scorer_w=1)
    source = np.array(
        [
            [[1, 11, 21], [2, 12, 22]],
            [[3, 13, 23], [4, 14, 24]],
        ],
        dtype=np.uint8,
    )
    coefficients, targets, witnesses = _source_block_geometry(operator, source)
    assert coefficients.shape == (1, 1, 1, 4)
    assert witnesses.shape == (1, 1, 3, 4)
    np.testing.assert_array_equal(
        witnesses[0, 0],
        np.array([[1, 2, 3, 4], [11, 12, 13, 14], [21, 22, 23, 24]], dtype=np.uint8),
    )
    realized = np.sum(coefficients * witnesses, axis=-1, dtype=np.int64)
    np.testing.assert_array_equal(realized, targets)


def test_resume_rebuilds_only_from_hashed_stage_receipts_and_rejects_tampering(
    tmp_path: Path,
) -> None:
    stages = tmp_path / "stages"
    stages.mkdir()
    identity_hash = "a" * 64
    exact = profile_integer_block((1, 1), 2, 10, max_nodes=1000)
    frame_aggregate = StreamingProfileAggregator(
        n_classes=5,
        named_strata=("boundary_annulus", "fragile", "degenerate"),
    )
    frame_aggregate.add_pixel(target_class=1, channel_results=(exact, exact, exact))
    counters = {
        "selected_blocks": 3,
        "total_blocks": 3,
        "exhaustive_selected_blocks": 3,
        "bounded_selected_blocks": 0,
        "omitted_blocks": 0,
        "segnet_mismatches": 0,
        "segnet_pixels": 0,
    }
    candidate_payload = _wrapped_frame_payload((((0, 0, 0, 0),) * 3,))
    receipt = {
        "schema": STAGE_RECEIPT_SCHEMA,
        "identity_sha256": identity_hash,
        "previous_stage_sha256": identity_hash,
        "frame": 0,
        "partition_custody": _partition_custody(0),
        "mode": ENUMERATED_MODE,
        "lower_bound_method": None,
        "derivation": "EXACT_OR_BOUNDED_ENUMERATED_SUBSET",
        "aggregate_delta_state": frame_aggregate.state(),
        "counters": counters,
        "candidate_payload_bytes": len(candidate_payload),
        "candidate_payload_sha256": sha256(candidate_payload).hexdigest(),
        "timing": _receipt_timing(3),
        "scope": {
            "frame_indices": [0],
            "rgb_channel_blocks": 3,
            "scorer_pixels": 1,
        },
    }
    receipt["selection_custody"] = _receipt_selection(receipt, receiver_closed=True)
    expected_replay = _replay_from_receipt(receipt, candidate_payload)
    stage_path = stages / "frame_0000.bin"
    _atomic_stage(stage_path, _stage_payload(receipt, candidate_payload))
    progress_path = tmp_path / "progress.json"
    pointer = _progress_pointer(
        identity_hash,
        next_frame=1,
        head=sha256(stage_path.read_bytes()).hexdigest(),
    )
    atomic_json(progress_path, pointer)
    rebuilt, rebuilt_receipts, rebuilt_counters = _resume_from_stage_chain(
        stages,
        progress_path,
        identity_hash=identity_hash,
        semantic_replay_provider=lambda _frame: expected_replay,
    )
    assert rebuilt.summary() == frame_aggregate.summary()
    assert len(rebuilt_receipts) == 1
    assert rebuilt_counters == counters

    tampered_pointer = {**pointer, "selected_blocks": 999}
    atomic_json(progress_path, tampered_pointer)
    with pytest.raises(ProfilerError, match="exact pointer schema"):
        _resume_from_stage_chain(
            stages,
            progress_path,
            identity_hash=identity_hash,
            semantic_replay_provider=lambda _frame: expected_replay,
        )

    atomic_json(progress_path, pointer)
    tampered_receipt = {**receipt, "previous_stage_sha256": "b" * 64}
    stage_path.write_bytes(_stage_payload(tampered_receipt, candidate_payload))
    pointer["stage_chain_head_sha256"] = sha256(stage_path.read_bytes()).hexdigest()
    atomic_json(progress_path, pointer)
    with pytest.raises(ProfilerError, match="successful stage bytes differ"):
        _resume_from_stage_chain(
            stages,
            progress_path,
            identity_hash=identity_hash,
            semantic_replay_provider=lambda _frame: expected_replay,
        )


def _bounds_stage_receipt(
    identity_hash: str,
    *,
    frame: int,
    previous_hash: str,
    fragile: bool = False,
) -> dict[str, object]:
    aggregate = StreamingProfileAggregator(
        n_classes=5,
        named_strata=("boundary_annulus", "fragile", "degenerate"),
    )
    aggregate.add_bounds_batch(
        target_classes=np.array([1], dtype=np.int64),
        lower_bounds=np.array([[2, 3, 4]], dtype=np.int64),
        upper_bounds=np.array([[4, 5, 6]], dtype=np.int64),
        strata={"fragile": np.array([fragile], dtype=bool)},
    )
    receipt: dict[str, object] = {
        "schema": STAGE_RECEIPT_SCHEMA,
        "identity_sha256": identity_hash,
        "previous_stage_sha256": previous_hash,
        "frame": frame,
        "partition_custody": _partition_custody(frame, fragile=fragile),
        "mode": BOUNDS_MODE,
        "lower_bound_method": LOWER_BOUND_METHOD,
        "derivation": "DERIVED_BOUNDS_FROM_REAL_N600_SOURCE_WITNESS",
        "aggregate_delta_state": aggregate.state(),
        "counters": {
            "selected_blocks": 0,
            "total_blocks": 3,
            "exhaustive_selected_blocks": 0,
            "bounded_selected_blocks": 0,
            "omitted_blocks": 3,
            "segnet_mismatches": 0,
            "segnet_pixels": 0,
        },
        "candidate_payload_bytes": 0,
        "candidate_payload_sha256": sha256(b"").hexdigest(),
        "timing": _receipt_timing(3),
        "scope": {
            "frame_indices": [frame],
            "rgb_channel_blocks": 3,
            "scorer_pixels": 1,
        },
    }
    receipt["selection_custody"] = _receipt_selection(receipt)
    return receipt


def _persist_test_receipt_authorization(
    root: Path,
    *,
    receipt: dict[str, object],
    identity: dict[str, object],
    exact_argv: list[str],
) -> tuple[tuple[bytes, ...], str]:
    receipt_path = root / RECEIPT_NAME
    prior = (
        profiler_module._read_bound_bytes(receipt_path, name="test receipt authorization prior")
        if receipt_path.exists()
        else None
    )
    custody = receipt["custody"]
    assert isinstance(custody, dict)
    authorization = profiler_module._receipt_transition_authorization(
        desired_receipt=receipt,
        prior_snapshot=prior,
        identity_sha256=sha256(profiler_module.canonical_json_bytes(identity)).hexdigest(),
        exact_rebuild_argv=exact_argv,
        terminal_stage_chain_sha256=str(custody["terminal_stage_chain_sha256"]),
        frame_count=int(custody["ordered_stage_count"]),
    )
    _path, digest = profiler_module._persist_receipt_transition_authorization(
        root,
        authorization=authorization,
        authorize_mutation=lambda: None,
    )
    return (() if prior is None else (prior.payload,)), digest


def _write_authorized_receipt(
    root: Path,
    *,
    receipt: dict[str, object],
    identity: dict[str, object],
    exact_argv: list[str],
) -> None:
    priors, digest = _persist_test_receipt_authorization(
        root,
        receipt=receipt,
        identity=identity,
        exact_argv=exact_argv,
    )
    _atomic_json(
        root / RECEIPT_NAME,
        receipt,
        expected_prior_payloads=priors,
        consumer_authorization_sha256=digest,
    )


def _bounds_semantic_replay(frame: int, *, fragile: bool = False) -> _FrameSemanticReplay:
    receipt = _bounds_stage_receipt(
        "0" * 64,
        frame=frame,
        previous_hash="0" * 64,
        fragile=fragile,
    )
    return _replay_from_receipt(receipt, b"")


def _equal_count_semantic_fixture() -> tuple[dict[str, object], _FrameSemanticReplay]:
    operator = DisjointResizeOperator.build(camera_h=2, camera_w=4, scorer_h=1, scorer_w=2)
    source = np.zeros((2, 4, 3), dtype=np.uint8)
    logits = np.zeros((5, 1, 2), dtype=np.float32)
    logits[1, 0, 0] = np.float32(2.0)
    logits[0, 0, 0] = np.float32(1.95)
    logits[2, 0, 1] = np.float32(2.0)
    degenerate = np.array([[False, True]], dtype=bool)
    custody = _derive_partition_custody(
        0,
        live_logits=logits,
        source_frame=source,
        operator=operator,
        fragile_margin=0.1,
        degenerate_mask=degenerate,
    )
    aggregate = StreamingProfileAggregator(
        n_classes=5,
        named_strata=("boundary_annulus", "fragile", "degenerate"),
    )
    aggregate.add_bounds_batch(
        target_classes=np.array([[1, 2]], dtype=np.int64),
        lower_bounds=np.array([[[2, 3, 4], [16, 17, 18]]], dtype=np.int64),
        upper_bounds=np.array([[[5, 6, 7], [24, 25, 26]]], dtype=np.int64),
        strata={
            "boundary_annulus": np.array([[True, True]], dtype=bool),
            "fragile": np.array([[True, False]], dtype=bool),
            "degenerate": degenerate,
        },
    )
    aggregate_state = aggregate.state()
    receipt: dict[str, object] = {
        "schema": STAGE_RECEIPT_SCHEMA,
        "identity_sha256": "4" * 64,
        "previous_stage_sha256": "4" * 64,
        "frame": 0,
        "partition_custody": custody,
        "mode": BOUNDS_MODE,
        "lower_bound_method": LOWER_BOUND_METHOD,
        "derivation": "DERIVED_BOUNDS_FROM_REAL_N600_SOURCE_WITNESS",
        "aggregate_delta_state": aggregate_state,
        "counters": {
            "selected_blocks": 0,
            "total_blocks": 6,
            "exhaustive_selected_blocks": 0,
            "bounded_selected_blocks": 0,
            "omitted_blocks": 6,
            "segnet_mismatches": 0,
            "segnet_pixels": 0,
        },
        "candidate_payload_bytes": 0,
        "candidate_payload_sha256": sha256(b"").hexdigest(),
        "timing": _receipt_timing(6),
        "scope": {
            "frame_indices": [0],
            "rgb_channel_blocks": 6,
            "scorer_pixels": 2,
        },
    }
    receipt["selection_custody"] = _receipt_selection(receipt)
    replay = _replay_from_receipt(receipt, b"")
    return receipt, replay


def test_stage_validator_refuses_freshly_hashed_internal_inconsistency(tmp_path: Path) -> None:
    identity_hash = "c" * 64
    stages = tmp_path / "stages"
    stages.mkdir()
    receipt = _bounds_stage_receipt(identity_hash, frame=0, previous_hash=identity_hash)
    receipt["counters"] = {**receipt["counters"], "total_blocks": 4}  # type: ignore[arg-type]
    stage = stages / "frame_0000.bin"
    _atomic_stage(stage, _stage_payload(receipt, b""))
    progress = tmp_path / "progress.json"
    atomic_json(
        progress,
        _progress_pointer(
            identity_hash,
            next_frame=1,
            head=sha256(stage.read_bytes()).hexdigest(),
        ),
    )
    with pytest.raises(
        ProfilerError,
        match=r"counters do not match immutable-input semantic replay|counter/aggregate/scope|selection/counter",
    ):
        _resume_from_stage_chain(
            stages,
            progress,
            identity_hash=identity_hash,
            semantic_replay_provider=lambda frame: _bounds_semantic_replay(frame),
        )

    receipt = _bounds_stage_receipt(identity_hash, frame=0, previous_hash=identity_hash)
    receipt["counters"] = {
        **receipt["counters"],  # type: ignore[arg-type]
        "selected_blocks": 1,
        "bounded_selected_blocks": 1,
        "omitted_blocks": 2,
    }
    receipt["selection_custody"] = _receipt_selection(receipt)
    with pytest.raises(ProfilerError, match="mode-specific"):
        _validate_against_replay(receipt, b"", _replay_from_receipt(receipt, b""))


def _move_middle_histogram_bin(stats: dict[str, object]) -> None:
    bins = stats["bins"]
    assert isinstance(bins, list)
    occupied = [index for index, count in enumerate(bins) if count]
    source_bin = occupied[len(occupied) // 2]
    bins[source_bin] -= 1
    bins[source_bin + 1] += 1


def test_stage_validator_refuses_coordinated_histogram_redistribution() -> None:
    identity_hash = "7" * 64
    receipt = _bounds_stage_receipt(identity_hash, frame=0, previous_hash=identity_hash)
    aggregate = receipt["aggregate_delta_state"]
    assert isinstance(aggregate, dict)
    _move_middle_histogram_bin(aggregate["global"]["lower"])
    _move_middle_histogram_bin(aggregate["per_class"]["1"]["lower"])
    with pytest.raises(ProfilerError, match=r"compact (total|extrema).*(inconsistent)"):
        _validate_against_replay(receipt, b"", _replay_from_receipt(receipt, b""))

    receipt = _bounds_stage_receipt(
        identity_hash,
        frame=0,
        previous_hash=identity_hash,
        fragile=True,
    )
    aggregate = receipt["aggregate_delta_state"]
    assert isinstance(aggregate, dict)
    _move_middle_histogram_bin(aggregate["global"]["lower"])
    _move_middle_histogram_bin(aggregate["per_class"]["1"]["lower"])
    _move_middle_histogram_bin(aggregate["strata"]["fragile"]["lower"])
    with pytest.raises(ProfilerError, match=r"compact (total|extrema).*(inconsistent)"):
        _validate_against_replay(receipt, b"", _replay_from_receipt(receipt, b""))


def test_compact_histogram_final_bin_and_same_bin_extrema_are_exactly_bounded() -> None:
    bins = [0] * 129
    bins[128] = 1
    valid = {
        "bin_width": 0.25,
        "bins": bins,
        "count": 1,
        "zero_count": 0,
        "total": 32.0,
        "minimum": 32.0,
        "maximum": 32.0,
    }
    _validate_compact_state(valid, name="synthetic", expected_count=1)
    invalid = json.loads(json.dumps(valid))
    invalid["total"] = 32.125
    invalid["maximum"] = 32.125
    with pytest.raises(ProfilerError, match="extrema are inconsistent"):
        _validate_compact_state(invalid, name="synthetic", expected_count=1)

    same_bin = {
        "bin_width": 0.25,
        "bins": [0] * 4 + [3] + [0] * 124,
        "count": 3,
        "zero_count": 0,
        "total": 3.3,
        "minimum": 1.0,
        "maximum": 1.2,
    }
    _validate_compact_state(same_bin, name="same-bin", expected_count=3)


@pytest.mark.parametrize("tamper", ["equal_count_class_swap", "equal_count_stratum_swap"])
@pytest.mark.parametrize("stage_kind", ["final", "prepared"])
def test_semantic_replay_rejects_equal_count_swaps_after_fresh_rehash(
    tmp_path: Path,
    tamper: str,
    stage_kind: str,
) -> None:
    receipt, expected_replay = _equal_count_semantic_fixture()
    aggregate = receipt["aggregate_delta_state"]
    assert isinstance(aggregate, dict)
    original_custody = json.loads(json.dumps(receipt["partition_custody"]))
    if tamper == "equal_count_class_swap":
        left = aggregate["per_class"]["1"]
        right = aggregate["per_class"]["2"]
        assert left["scorer_pixels"] == right["scorer_pixels"] == 1
        assert left != right
        aggregate["per_class"]["1"], aggregate["per_class"]["2"] = right, left
    else:
        left = aggregate["strata"]["fragile"]
        right = aggregate["strata"]["degenerate"]
        assert left["scorer_pixels"] == right["scorer_pixels"] == 1
        assert left != right
        aggregate["strata"]["fragile"], aggregate["strata"]["degenerate"] = right, left
    assert receipt["partition_custody"] == original_custody

    # All pre-fix5 count, histogram, partition-custody, and byte checks pass.
    # Only the independent scientific replay distinguishes the equal-count swap.
    with pytest.raises(ProfilerError, match=r"semantic aggregate.*immutable-input replay"):
        _validate_against_replay(receipt, b"", expected_replay)

    profile_root = tmp_path / tamper / stage_kind
    stages, progress = _initialize_fresh_output(
        profile_root,
        identity_hash="4" * 64,
        storage_preflight={"PASS": True},
        exact_argv=["python", "profile.py"],
    )
    final = stages / "frame_0000.bin"
    path = final if stage_kind == "final" else _prepared_stage_path(final)
    payload = _stage_payload(receipt, b"")
    if stage_kind == "final":
        _atomic_stage(final, payload)
        atomic_json(
            progress,
            _progress_pointer(
                "4" * 64,
                next_frame=1,
                head=sha256(path.read_bytes()).hexdigest(),
            ),
        )
    else:
        path.write_bytes(payload)
        _write_stage_attempt_intent(
            final,
            payload,
            identity_sha256="4" * 64,
            exact_rebuild_argv=["python", "profile.py"],
        )
    before_stage = path.read_bytes()
    before_progress = progress.read_bytes()
    with pytest.raises(ProfilerError, match=r"semantic aggregate.*immutable-input replay"):
        _resume_from_stage_chain(
            stages,
            progress,
            identity_hash="4" * 64,
            semantic_replay_provider=lambda _frame: expected_replay,
            max_frames=1,
        )
    assert path.read_bytes() == before_stage
    assert progress.read_bytes() == before_progress
    assert (stage_kind == "final") == final.exists()


@pytest.mark.parametrize("tamper", ["class_swap", "global_as_fragile"])
def test_resume_rejects_self_consistent_partition_tamper_after_rehash(
    tmp_path: Path,
    tamper: str,
) -> None:
    identity_hash = "6" * 64
    receipt = _bounds_stage_receipt(identity_hash, frame=0, previous_hash=identity_hash)
    aggregate = receipt["aggregate_delta_state"]
    assert isinstance(aggregate, dict)
    if tamper == "class_swap":
        aggregate["per_class"]["1"], aggregate["per_class"]["2"] = (
            aggregate["per_class"]["2"],
            aggregate["per_class"]["1"],
        )
        receipt["partition_custody"] = _partition_custody(0, target_class=2)
    else:
        aggregate["strata"]["fragile"] = json.loads(json.dumps(aggregate["global"]))
        receipt["partition_custody"] = _partition_custody(0, fragile=True)

    # The tampered receipt is internally self-consistent.  Only the independent
    # provider derived from immutable inputs can reject its semantic relabeling.
    _validate_against_replay(receipt, b"", _replay_from_receipt(receipt, b""))

    stages = tmp_path / tamper / "stages"
    stages.mkdir(parents=True)
    stage = stages / "frame_0000.bin"
    _atomic_stage(stage, _stage_payload(receipt, b""))
    progress = stages.parent / "progress.json"
    atomic_json(
        progress,
        _progress_pointer(
            identity_hash,
            next_frame=1,
            head=sha256(stage.read_bytes()).hexdigest(),
        ),
    )
    with pytest.raises(ProfilerError, match="partition custody does not match immutable inputs"):
        _resume_from_stage_chain(
            stages,
            progress,
            identity_hash=identity_hash,
            semantic_replay_provider=lambda frame: _bounds_semantic_replay(frame),
        )


def test_profiler_progress_complete_requires_canonical_n600_prefix(tmp_path: Path) -> None:
    identity_hash = "5" * 64
    stages, progress = _initialize_fresh_output(
        tmp_path / "profile",
        identity_hash=identity_hash,
        storage_preflight={"PASS": True},
        exact_argv=["python", "profile.py"],
    )
    pointer = json.loads(progress.read_text(encoding="utf-8"))
    pointer["status"] = "complete"
    progress.write_text(json.dumps(pointer), encoding="utf-8")
    with pytest.raises(ProfilerError, match="status/prefix invariant"):
        _resume_from_stage_chain(
            stages,
            progress,
            identity_hash=identity_hash,
            semantic_replay_provider=lambda frame: _bounds_semantic_replay(frame),
            max_frames=1,
        )


def test_stage_validator_wraps_malformed_aggregate_reconstruction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identity_hash = "8" * 64
    receipt = _bounds_stage_receipt(identity_hash, frame=0, previous_hash=identity_hash)

    def fail_reconstruction(_state: object) -> StreamingProfileAggregator:
        raise LatticeProfileError("synthetic malformed reconstruction")

    monkeypatch.setattr(StreamingProfileAggregator, "from_state", staticmethod(fail_reconstruction))
    with pytest.raises(ProfilerError, match="not reconstructable"):
        _validate_against_replay(receipt, b"", _replay_from_receipt(receipt, b""))


def test_initial_pointer_resumes_at_frame_zero_and_adopts_valid_orphan(tmp_path: Path) -> None:
    identity_hash = "d" * 64
    stages, progress = _initialize_fresh_output(
        tmp_path / "profile",
        identity_hash=identity_hash,
        storage_preflight={"PASS": True},
        exact_argv=["python", "profile.py"],
    )
    rebuilt, receipts, counters = _resume_from_stage_chain(
        stages,
        progress,
        identity_hash=identity_hash,
        semantic_replay_provider=lambda frame: _bounds_semantic_replay(frame),
        max_frames=1,
    )
    assert rebuilt.summary()["global"]["rgb_channel_blocks"] == 0
    assert receipts == []
    assert counters["total_blocks"] == 0

    receipt = _bounds_stage_receipt(identity_hash, frame=0, previous_hash=identity_hash)
    stage = stages / "frame_0000.bin"
    _atomic_stage(stage, _stage_payload(receipt, b""))
    _aggregate, adopted, adopted_counters = _resume_from_stage_chain(
        stages,
        progress,
        identity_hash=identity_hash,
        semantic_replay_provider=lambda frame: _bounds_semantic_replay(frame),
        max_frames=1,
    )
    pointer = json.loads(progress.read_text(encoding="utf-8"))
    assert len(adopted) == 1
    assert adopted_counters["total_blocks"] == 3
    assert stage.is_file()
    assert pointer["next_frame"] == 1
    assert pointer["stage_chain_head_sha256"] == sha256(stage.read_bytes()).hexdigest()


def test_malformed_orphan_is_refused_without_pointer_advance(tmp_path: Path) -> None:
    identity_hash = "e" * 64
    stages, progress = _initialize_fresh_output(
        tmp_path / "profile",
        identity_hash=identity_hash,
        storage_preflight={"PASS": True},
        exact_argv=["python", "profile.py"],
    )
    receipt = _bounds_stage_receipt(identity_hash, frame=0, previous_hash=identity_hash)
    receipt["scope"] = {**receipt["scope"], "rgb_channel_blocks": 4}  # type: ignore[arg-type]
    stage = stages / "frame_0000.bin"
    _atomic_stage(stage, _stage_payload(receipt, b""))
    with pytest.raises(ProfilerError, match="counter/aggregate/scope"):
        _resume_from_stage_chain(
            stages,
            progress,
            identity_hash=identity_hash,
            semantic_replay_provider=lambda frame: _bounds_semantic_replay(frame),
            max_frames=1,
        )
    pointer = json.loads(progress.read_text(encoding="utf-8"))
    assert pointer["next_frame"] == 0
    assert stage.is_file()


def test_valid_prepared_stage_is_durably_adopted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identity_hash = "9" * 64
    stages, progress = _initialize_fresh_output(
        tmp_path / "profile",
        identity_hash=identity_hash,
        storage_preflight={"PASS": True},
        exact_argv=["python", "profile.py"],
    )
    receipt = _bounds_stage_receipt(identity_hash, frame=0, previous_hash=identity_hash)
    final = stages / "frame_0000.bin"
    prepared = _prepared_stage_path(final)

    def interrupt_stage_rename(source: Path, destination: Path) -> None:
        if source == prepared and destination == final:
            raise OSError("synthetic SIGKILL window")

    monkeypatch.setattr(
        profiler_module.feature_cache_module,
        "_MOVE_PATH_NOREPLACE_TEST_HOOK",
        interrupt_stage_rename,
    )
    with pytest.raises(ProfilerError, match="no-replace move failed"):
        _atomic_stage(final, _stage_payload(receipt, b""))
    assert prepared.is_file()
    assert not final.exists()
    monkeypatch.setattr(profiler_module.feature_cache_module, "_MOVE_PATH_NOREPLACE_TEST_HOOK", None)

    _aggregate, adopted, counters = _resume_from_stage_chain(
        stages,
        progress,
        identity_hash=identity_hash,
        semantic_replay_provider=lambda frame: _bounds_semantic_replay(frame),
        max_frames=1,
    )
    pointer = json.loads(progress.read_text(encoding="utf-8"))
    assert len(adopted) == 1
    assert counters["total_blocks"] == 3
    assert final.is_file()
    assert not prepared.exists()
    assert pointer["next_frame"] == 1
    assert pointer["stage_chain_head_sha256"] == sha256(final.read_bytes()).hexdigest()


def test_bound_stage_move_refuses_late_destination_without_overwrite(tmp_path: Path) -> None:
    source = tmp_path / ".frame_0000.bin.prepared"
    destination = tmp_path / "frame_0000.bin"
    source_bytes = b"authorized-stage"
    foreign_bytes = b"late-foreign-destination"
    source.write_bytes(source_bytes)
    destination.write_bytes(foreign_bytes)
    snapshot = profiler_module._read_bound_bytes(source, name="test prepared stage")

    with pytest.raises(ProfilerError, match="no-replace move failed"):
        profiler_module._replace_bound(source, destination, snapshot, name="test prepared stage")

    assert source.read_bytes() == source_bytes
    assert destination.read_bytes() == foreign_bytes


def test_bound_stage_move_source_substitution_rolls_back_without_foreign_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / ".frame_0000.bin.prepared"
    destination = tmp_path / "frame_0000.bin"
    displaced = tmp_path / "authorized-displaced.bin"
    foreign = tmp_path / "foreign.bin"
    source_bytes = b"authorized-stage"
    foreign_bytes = b"foreign-stage"
    source.write_bytes(source_bytes)
    foreign.write_bytes(foreign_bytes)
    snapshot = profiler_module._read_bound_bytes(source, name="test prepared stage")

    def substitute_source(move_source: Path, move_destination: Path) -> None:
        assert move_source == source
        assert move_destination == destination
        profiler_module.os.replace(source, displaced)
        profiler_module.os.replace(foreign, source)

    monkeypatch.setattr(
        profiler_module.feature_cache_module,
        "_MOVE_PATH_NOREPLACE_TEST_HOOK",
        substitute_source,
    )
    with pytest.raises(ProfilerError, match="no-replace move failed"):
        profiler_module._replace_bound(source, destination, snapshot, name="test prepared stage")

    assert displaced.read_bytes() == source_bytes
    assert source.read_bytes() == foreign_bytes
    assert not destination.exists()
    assert not foreign.exists()


def test_intent_retirement_substitution_preserves_both_inodes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    intent = tmp_path / f".frame_0000.bin.intent-attempt-00000000-1-{'a' * 64}"
    displaced = tmp_path / "authorized-intent-displaced"
    foreign = tmp_path / "foreign-intent"
    intent.write_bytes(b"")
    foreign_bytes = b"foreign-intent-bytes"
    foreign.write_bytes(foreign_bytes)
    snapshot = profiler_module._read_bound_bytes(intent, name="test stage intent")
    retained_destination: list[Path] = []

    def substitute_intent(move_source: Path, move_destination: Path) -> None:
        assert move_source == intent
        retained_destination.append(move_destination)
        profiler_module.os.replace(intent, displaced)
        profiler_module.os.replace(foreign, intent)

    monkeypatch.setattr(
        profiler_module.feature_cache_module,
        "_MOVE_PATH_NOREPLACE_TEST_HOOK",
        substitute_intent,
    )
    with pytest.raises(ProfilerError, match="retention failed"):
        profiler_module._unlink_bound(intent, snapshot, name="test stage intent")

    assert displaced.read_bytes() == b""
    assert intent.read_bytes() == foreign_bytes
    assert not foreign.exists()
    assert retained_destination and not retained_destination[0].exists()


def test_directory_finalization_refuses_destination_appearing_at_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    staging = tmp_path / ".profile-staging"
    destination = tmp_path / "profile"
    staging.mkdir()
    (staging / "source.bin").write_bytes(b"source")

    def create_late_destination(move_source: Path, move_destination: Path) -> None:
        assert move_source == staging
        assert move_destination == destination
        destination.mkdir()
        (destination / "foreign.bin").write_bytes(b"foreign")

    monkeypatch.setattr(
        profiler_module.feature_cache_module,
        "_MOVE_PATH_NOREPLACE_TEST_HOOK",
        create_late_destination,
    )
    with pytest.raises(ProfilerError, match="no-replace move failed"):
        profiler_module._move_directory_noreplace(staging, destination, name="test staging finalization")

    assert (staging / "source.bin").read_bytes() == b"source"
    assert (destination / "foreign.bin").read_bytes() == b"foreign"


def test_retained_stage_hash_tamper_blocks_before_replay_or_progress_mutation(tmp_path: Path) -> None:
    identity_hash = "8" * 64
    stages, progress = _initialize_fresh_output(
        tmp_path / "retained-tamper",
        identity_hash=identity_hash,
        storage_preflight={"PASS": True},
        exact_argv=["python", "profile.py"],
    )
    scratch = stages / f".frame_0000.bin.intent-attempt-00000000-1-{'a' * 64}"
    scratch.write_bytes(b"")
    snapshot = profiler_module._read_bound_bytes(scratch, name="test retained scratch")
    before_names = {path.name for path in stages.iterdir()}
    profiler_module._unlink_bound(scratch, snapshot, name="test retained scratch")
    retained = [
        path
        for path in stages.iterdir()
        if path.name not in before_names and profiler_module.feature_cache_module.is_retained_name(path.name)
    ]
    assert len(retained) == 1
    retained[0].write_bytes(b"tampered")
    pointer_before = progress.read_bytes()

    with pytest.raises(ProfilerError, match="retained custody is malformed"):
        _resume_from_stage_chain(
            stages,
            progress,
            identity_hash=identity_hash,
            semantic_replay_provider=lambda frame: pytest.fail("retention tamper must block before replay"),
            max_frames=1,
        )
    assert progress.read_bytes() == pointer_before


def test_well_formed_retention_for_unknown_stage_role_is_not_whitelisted(tmp_path: Path) -> None:
    identity_hash = "6" * 64
    stages, progress = _initialize_fresh_output(
        tmp_path / "retained-role",
        identity_hash=identity_hash,
        storage_preflight={"PASS": True},
        exact_argv=["python", "profile.py"],
    )
    unknown = stages / "operator-note.bin"
    unknown.write_bytes(b"preserve-but-never-authorize")
    snapshot = profiler_module._read_bound_bytes(unknown, name="unknown retained role")
    profiler_module._unlink_bound(unknown, snapshot, name="unknown retained role")
    pointer_before = progress.read_bytes()
    receipt = _bounds_stage_receipt(identity_hash, frame=0, previous_hash=identity_hash)

    with pytest.raises(ProfilerError, match="role-unproven custody"):
        _atomic_stage(stages / "frame_0000.bin", _stage_payload(receipt, b""))
    assert not (stages / "frame_0000.bin").exists()
    assert not _prepared_stage_path(stages / "frame_0000.bin").exists()

    with pytest.raises(ProfilerError, match="role-unproven custody"):
        _resume_from_stage_chain(
            stages,
            progress,
            identity_hash=identity_hash,
            semantic_replay_provider=lambda frame: pytest.fail("unknown retained role must block before replay"),
            max_frames=1,
        )
    assert progress.read_bytes() == pointer_before


@pytest.mark.parametrize("case", ["malformed", "conflicting", "duplicate", "unknown"])
def test_invalid_prepared_stage_fails_without_pointer_advance_or_byte_loss(
    tmp_path: Path,
    case: str,
) -> None:
    identity_hash = "f" * 64
    stages, progress = _initialize_fresh_output(
        tmp_path / case / "profile",
        identity_hash=identity_hash,
        storage_preflight={"PASS": True},
        exact_argv=["python", "profile.py"],
    )
    receipt = _bounds_stage_receipt(identity_hash, frame=0, previous_hash=identity_hash)
    final = stages / "frame_0000.bin"
    prepared = _prepared_stage_path(final)
    if case == "malformed":
        prepared.write_bytes(b"truncated")
    elif case == "conflicting":
        _atomic_stage(final, _stage_payload(receipt, b""))
        prepared.write_bytes(_stage_payload(receipt, b""))
    elif case == "duplicate":
        prepared.write_bytes(_stage_payload(receipt, b""))
        second = _bounds_stage_receipt(identity_hash, frame=1, previous_hash="0" * 64)
        _prepared_stage_path(stages / "frame_0001.bin").write_bytes(_stage_payload(second, b""))
    else:
        (stages / ".unknown-stage-scratch").write_bytes(b"preserve me")

    before = {path.name: path.read_bytes() for path in stages.iterdir() if path.is_file()}
    with pytest.raises(ProfilerError):
        _resume_from_stage_chain(
            stages,
            progress,
            identity_hash=identity_hash,
            semantic_replay_provider=lambda frame: _bounds_semantic_replay(frame),
            max_frames=2,
        )
    after = {path.name: path.read_bytes() for path in stages.iterdir() if path.is_file()}
    pointer = json.loads(progress.read_text(encoding="utf-8"))
    assert after == before
    assert pointer["next_frame"] == 0
    assert pointer["stage_chain_head_sha256"] == identity_hash


@pytest.mark.parametrize("mode", [BOUNDS_MODE, ENUMERATED_MODE])
def test_genuine_immutable_input_replay_validates_and_resume_continues(
    tmp_path: Path,
    mode: str,
) -> None:
    operator = DisjointResizeOperator.build(camera_h=2, camera_w=2, scorer_h=1, scorer_w=1)
    source = np.zeros((2, 2, 3), dtype=np.uint8)
    logits = np.zeros((5, 1, 1), dtype=np.float32)
    logits[1, 0, 0] = np.float32(2.0)
    selector = SignedResidualCostModel()
    plugin = NoOpPosePlugin()

    def compute() -> profiler_module._FrameProfileArtifacts:
        return _profile_frame_semantics(
            0,
            source_frame=source,
            live_logits=logits,
            operator=operator,
            mode=mode,
            seed_source_witness=False,
            max_nodes=1000,
            time_limit_seconds_per_block=None,
            fragile_margin=0.0,
            selector=selector,
            selector_identity=selector.identity,
            pose_plugin=plugin,
            pose_plugin_identity=plugin.identity,
            reuse_cache_entries=32,
            reuse=OrderedDict(),
            build_selected_frame=False,
        )

    artifacts = compute()
    identity_hash = "3" * 64
    receipt = {
        "schema": STAGE_RECEIPT_SCHEMA,
        "identity_sha256": identity_hash,
        "previous_stage_sha256": identity_hash,
        "frame": 0,
        "partition_custody": artifacts.replay.partition_custody,
        "mode": mode,
        "lower_bound_method": artifacts.lower_bound_method,
        "derivation": _expected_derivation(mode=mode, seed_source_witness=False),
        "aggregate_delta_state": artifacts.replay.aggregate_state,
        "counters": artifacts.counters,
        "candidate_payload_bytes": len(artifacts.candidate_payload),
        "candidate_payload_sha256": sha256(artifacts.candidate_payload).hexdigest(),
        "timing": _receipt_timing(artifacts.counters["total_blocks"]),
        "scope": {
            "frame_indices": [0],
            "rgb_channel_blocks": artifacts.counters["total_blocks"],
            "scorer_pixels": int(artifacts.labels.size),
        },
    }
    receipt["selection_custody"] = artifacts.replay.selection_custody
    replay = compute().replay
    assert replay.aggregate_state == artifacts.replay.aggregate_state
    _validate_against_replay(receipt, artifacts.candidate_payload, replay)

    stages, progress = _initialize_fresh_output(
        tmp_path / mode,
        identity_hash=identity_hash,
        storage_preflight={"PASS": True},
        exact_argv=["python", "profile.py"],
    )
    stage = stages / "frame_0000.bin"
    _atomic_stage(stage, _stage_payload(receipt, artifacts.candidate_payload))
    atomic_json(
        progress,
        _progress_pointer(
            identity_hash,
            next_frame=1,
            head=sha256(stage.read_bytes()).hexdigest(),
        ),
    )
    rebuilt, receipts, counters = _resume_from_stage_chain(
        stages,
        progress,
        identity_hash=identity_hash,
        semantic_replay_provider=lambda _frame: compute().replay,
        max_frames=1,
    )
    assert rebuilt.summary() == artifacts.aggregate.summary()
    assert len(receipts) == 1
    assert counters == artifacts.counters


def test_profiler_refuses_nondeterministic_wall_clock_cap_before_io(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(GOVERNED_MARKER_ENV, "1")
    args = SimpleNamespace(
        max_frames=1,
        max_nodes=1,
        reuse_cache_entries=1,
        fragile_margin=0.0,
        time_limit_seconds_per_block=0.01,
        seed_source_witness=False,
        score_segnet=False,
        mode=ENUMERATED_MODE,
        rss_cap_mb=512,
        timeout_seconds=60,
        resume=False,
        allow_local_output_for_tests=True,
    )
    with pytest.raises(ProfilerError, match="nondeterministic wall-clock profile caps"):
        run_profile(args, exact_argv=["python", "profile.py"])


def test_profiler_identity_binds_exact_executed_optimization_modules(tmp_path: Path) -> None:
    gt_cache = tmp_path / "gt.npz"
    gt_cache.write_bytes(b"gt")
    feature_root = tmp_path / "feature"
    feature_root.mkdir()
    (feature_root / "manifest.json").write_text("{}\n", encoding="utf-8")
    args = SimpleNamespace(
        mode=BOUNDS_MODE,
        seed_source_witness=False,
        max_frames=1,
        max_nodes=1,
        time_limit_seconds_per_block=None,
        fragile_margin=0.0,
        score_segnet=False,
        rss_cap_mb=512,
        timeout_seconds=60,
    )
    identity = _identity(
        args,
        gt_cache,
        feature_root,
        {"synthetic": True},
        None,
        _synthetic_parent_attestation(),
        _synthetic_fresh_argv(),
        _certified_storage_identity(tmp_path / "profile-output"),
    )
    expected = {
        "executed_uint8_lattice_feasibility_module": REPO_ROOT / "src/tac/optimization/uint8_lattice_feasibility.py",
        "executed_uint8_lattice_profile_module": REPO_ROOT / "src/tac/optimization/uint8_lattice_profile.py",
        "executed_feature_cache_module": REPO_ROOT / "src/tac/witness_control/segnet_head_feature_cache.py",
        "executed_feature_extractor_tool": REPO_ROOT / "tools/extract_segnet_head_features_n600.py",
        "executed_stored_npz_module": REPO_ROOT / "src/tac/boundary_math/power_diagram_witness.py",
        "executed_admission_guard_module": REPO_ROOT / "src/tac/admission_guard.py",
        "executed_tool_bootstrap_module": REPO_ROOT / "tools/tool_bootstrap.py",
        "operational_custody_governed_profile_admission": REPO_ROOT / "src/tac/governed_profile_admission.py",
        "operational_custody_safe_run": REPO_ROOT / "tools/safe_run.py",
        "operational_custody_admission_guard": REPO_ROOT / "src/tac/admission_guard.py",
    }
    for role, path in expected.items():
        row = identity["sources"][role]
        assert row["path"] == str(path.resolve())
        assert row["bytes"] == path.stat().st_size
        assert row["sha256"] == sha256(path.read_bytes()).hexdigest()

    assert identity["config"]["seed_source_witness"] is False
    admission = identity["resource_custody"]["stable_admission_contract"]
    assert admission["canonical_fresh_child_argv"] == _synthetic_fresh_argv()
    assert admission["outer_resource_caps"] == {"rss_cap_mb": 512, "timeout_seconds": 60.0}
    assert admission["volatile_per_invocation_fields_excluded"] == [
        "parent_pid",
        "parent_exact_argv",
        "child_exact_argv",
    ]
    assert "parent_pid" not in admission
    assert identity["resource_custody"]["completed_safe_run_status_receipt"] is None
    assert len(identity["repository"]["git_head"]) in (40, 64)
    assert identity["repository"]["worktree_cleanliness_required"] is False
    assert identity["config"]["requested_outer_governor_limits"] == {
        "rss_cap_mb": 512,
        "timeout_seconds": 60,
        "profiler_self_enforced": False,
        "scope": "REQUESTED_OUTER_GOVERNOR_LIMITS_METADATA_ONLY_NOT_ENFORCEMENT_RECEIPT",
        "required_launcher": "tools/safe_run.py process-group/system-memory governor",
    }
    assert identity["config"]["receiver_stream_codecs"]["zlib"] == {
        "level": 9,
        "method": 8,
        "wbits": 15,
        "mem_level": 8,
        "strategy": 0,
    }
    assert identity["config"]["receiver_stream_codecs"]["brotli"] == {
        "mode": 0,
        "quality": 11,
        "lgwin": 22,
        "lgblock": 0,
    }
    assert identity["config"]["runtime"]["zlib_build"]
    assert identity["config"]["runtime"]["zlib_runtime"]
    assert identity["config"]["runtime"]["brotli"]
    seeded_args = SimpleNamespace(
        mode=ENUMERATED_MODE,
        seed_source_witness=True,
        max_frames=1,
        max_nodes=1,
        time_limit_seconds_per_block=None,
        fragile_margin=0.0,
        score_segnet=False,
        rss_cap_mb=512,
        timeout_seconds=60,
    )
    seeded_identity = _identity(
        seeded_args,
        gt_cache,
        feature_root,
        {"synthetic": True},
        None,
        _synthetic_parent_attestation(),
        _synthetic_fresh_argv(),
        _certified_storage_identity(tmp_path / "profile-output"),
    )
    assert seeded_identity["config"]["seed_source_witness"] is True


def test_profiler_identity_is_stable_across_fresh_and_resume_parent_invocations(
    tmp_path: Path,
) -> None:
    gt_cache = tmp_path / "gt.npz"
    gt_cache.write_bytes(b"gt")
    feature_root = tmp_path / "feature"
    feature_root.mkdir()
    (feature_root / "manifest.json").write_text("{}\n", encoding="utf-8")
    args = SimpleNamespace(
        mode=BOUNDS_MODE,
        seed_source_witness=False,
        max_frames=1,
        max_nodes=1,
        time_limit_seconds_per_block=None,
        fragile_margin=0.0,
        score_segnet=False,
        rss_cap_mb=512,
        timeout_seconds=60,
    )
    fresh_identity = _identity(
        args,
        gt_cache,
        feature_root,
        {"synthetic": True},
        None,
        _synthetic_parent_attestation(parent_pid=111),
        _synthetic_fresh_argv(),
        _certified_storage_identity(tmp_path / "profile-output"),
    )
    resumed_identity = _identity(
        args,
        gt_cache,
        feature_root,
        {"synthetic": True},
        None,
        _synthetic_parent_attestation(parent_pid=999, resume=True),
        _synthetic_fresh_argv(),
        _certified_storage_identity(tmp_path / "profile-output"),
    )
    assert profiler_module.canonical_json_bytes(fresh_identity) == profiler_module.canonical_json_bytes(
        resumed_identity
    )


def test_feature_binding_rejects_stale_cache_and_foreign_scorer_sources(tmp_path: Path) -> None:
    def row(path: Path) -> dict[str, object]:
        resolved = path.resolve()
        return {
            "path": str(resolved),
            "bytes": resolved.stat().st_size,
            "sha256": sha256(resolved.read_bytes()).hexdigest(),
        }

    gt_cache = tmp_path / "gt.npz"
    gt_cache.write_bytes(b"gt")
    scorer_file = tmp_path / "segnet.safetensors"
    scorer_file.write_bytes(b"weights")
    scorer_sources = {"segnet_weights": row(scorer_file)}
    source_files = {
        "gt_n600_npz": row(gt_cache),
        "extractor_tool": row(REPO_ROOT / "tools/extract_segnet_head_features_n600.py"),
        "cache_module": row(REPO_ROOT / "src/tac/witness_control/segnet_head_feature_cache.py"),
        **scorer_sources,
    }
    feature = SimpleNamespace(
        identity={
            "source_files": source_files,
            "config": {
                "authority_mode": "deterministic_cpu_float32_batch_one",
                "batch_size": 1,
                "runtime": {
                    "python": profiler_module.platform.python_version(),
                    "python_implementation": profiler_module.platform.python_implementation(),
                    "python_executable": str(Path(profiler_module.sys.executable).resolve()),
                    "torch": torch.__version__,
                    "numpy": np.__version__,
                    "platform": profiler_module.platform.platform(),
                },
                "determinism": {
                    "torch_deterministic_algorithms": True,
                    "torch_threads_effective": 1,
                    "torch_interop_threads_effective": 1,
                },
            },
        },
        progress={"identity_sha256": "1" * 64, "committed_frames": [{"frame": 0}]},
        next_frame=1,
    )
    binding = _feature_binding(
        feature,
        gt_cache,
        prefix_frames=1,
        scorer_sources=scorer_sources,
    )
    assert len(binding["current_execution_source_binding_sha256"]) == 64

    stale = json.loads(json.dumps(source_files))
    stale["cache_module"]["sha256"] = "0" * 64
    feature.identity = {**feature.identity, "source_files": stale}
    with pytest.raises(ProfilerError, match="cache_module source binding is stale or foreign"):
        _feature_binding(feature, gt_cache, prefix_frames=1, scorer_sources=scorer_sources)

    foreign = json.loads(json.dumps(source_files))
    foreign["segnet_weights"]["sha256"] = "2" * 64
    feature.identity = {**feature.identity, "source_files": foreign}
    with pytest.raises(ProfilerError, match="segnet_weights source binding is stale or foreign"):
        _feature_binding(feature, gt_cache, prefix_frames=1, scorer_sources=scorer_sources)

    feature.identity = {
        **feature.identity,
        "source_files": source_files,
        "config": {**feature.identity["config"], "batch_size": 8},
    }
    with pytest.raises(ProfilerError, match="runtime/batch-one determinism contract is stale or foreign"):
        _feature_binding(feature, gt_cache, prefix_frames=1, scorer_sources=scorer_sources)


def test_source_witness_flag_parses_and_is_refused_outside_enumerated_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parsed = _parse_args(
        [
            "--gt-cache",
            "gt.npz",
            "--feature-cache-root",
            "feature-cache",
            "--output-root",
            "profile-output",
            "--mode",
            ENUMERATED_MODE,
            "--seed-source-witness",
            "--rss-cap-mb",
            "512",
            "--timeout-seconds",
            "60",
        ]
    )
    assert parsed.seed_source_witness is True

    invalid = SimpleNamespace(
        max_frames=1,
        max_nodes=1,
        reuse_cache_entries=1,
        fragile_margin=0.0,
        time_limit_seconds_per_block=None,
        seed_source_witness=True,
        score_segnet=False,
        mode=BOUNDS_MODE,
        rss_cap_mb=512,
        timeout_seconds=60,
        resume=False,
        allow_local_output_for_tests=True,
    )
    monkeypatch.setenv(GOVERNED_MARKER_ENV, "1")
    with pytest.raises(ProfilerError, match="valid only in enumerated_subset mode"):
        run_profile(invalid, exact_argv=["python", "profile.py"])


def test_source_seed_extraction_and_cap_one_parseback_are_byte_exact() -> None:
    source = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)
    assert _source_seed_candidate(
        source,
        row_indices=(0, 1),
        column_indices=(0, 1),
        channel=2,
    ) == (2, 5, 8, 11)

    _operator, source_frame, artifacts = _tiny_seed_artifacts(source=source, max_nodes=1)
    assert artifacts.counters["selected_blocks"] == artifacts.counters["total_blocks"] == 3
    assert artifacts.counters["bounded_selected_blocks"] == 3
    assert artifacts.counters["exhaustive_selected_blocks"] == 0
    assert artifacts.receiver_closed is True
    assert np.array_equal(artifacts.selected_frame, source_frame)
    assert np.array_equal(artifacts.decoded_frame, source_frame)
    assert artifacts.replay.candidate_payload == artifacts.candidate_payload

    custody = artifacts.replay.selection_custody
    assert custody == {
        "selection_label": "KNOWN_SOURCE_WITNESS_SEEDED_CHEAPEST_SEEN_NON_GLOBAL",
        "selection_globally_exact": False,
        "exact_count_claim": False,
        "min_description_claim": False,
        "seed_source_witness": True,
        "receiver_non_closure": False,
        "pose_bank_wired": False,
        "factor10_solved": False,
        "scope_extrapolation": "NONE_EXACT_FRAME_INDICES_ONLY",
        "per_block_selector_minimum_proved": False,
        "global_compressed_stream_minimum_claim": False,
    }


def test_canonical_receiver_closes_exact_support_union_with_zero_fill_and_exact_numerators() -> None:
    operator = DisjointResizeOperator.build(camera_h=874, camera_w=1164, scorer_h=384, scorer_w=512)
    support = _receiver_support_union(operator)
    assert support.shape == (874, 1164, 3)
    assert np.any(support)
    assert np.any(~support)

    source = np.full(support.shape, 73, dtype=np.uint8)
    decoded = np.zeros_like(source)
    decoded[support] = source[support]
    assert not np.array_equal(decoded, source)
    _validate_source_seed_receiver(source, decoded, operator=operator)
    source_numerators, source_denominator = operator.apply_numerators(source)
    decoded_numerators, decoded_denominator = operator.apply_numerators(decoded)
    assert source_denominator == decoded_denominator
    assert np.array_equal(source_numerators, decoded_numerators)

    bad_support = decoded.copy()
    first_support = tuple(np.argwhere(support)[0])
    bad_support[first_support] ^= np.uint8(1)
    with pytest.raises(ProfilerError, match="differs from source on resize support S"):
        _validate_source_seed_receiver(source, bad_support, operator=operator)

    bad_fill = decoded.copy()
    first_complement = tuple(np.argwhere(~support)[0])
    bad_fill[first_complement] = np.uint8(1)
    with pytest.raises(ProfilerError, match="outside support S"):
        _validate_source_seed_receiver(source, bad_fill, operator=operator)


def test_source_seed_mode_bypasses_reuse_cache_even_when_keys_repeat() -> None:
    operator = DisjointResizeOperator.build(camera_h=2, camera_w=2, scorer_h=1, scorer_w=1)
    source = np.zeros((2, 2, 3), dtype=np.uint8)
    logits = np.zeros((5, 1, 1), dtype=np.float32)
    logits[1, 0, 0] = np.float32(2.0)
    selector = SignedResidualCostModel()
    plugin = NoOpPosePlugin()
    reuse: OrderedDict[str, object] = OrderedDict()

    _profile_frame_semantics(
        0,
        source_frame=source,
        live_logits=logits,
        operator=operator,
        mode=ENUMERATED_MODE,
        seed_source_witness=True,
        max_nodes=1,
        time_limit_seconds_per_block=None,
        fragile_margin=0.0,
        selector=selector,
        selector_identity=selector.identity,
        pose_plugin=plugin,
        pose_plugin_identity=plugin.identity,
        reuse_cache_entries=8,
        reuse=reuse,
        build_selected_frame=True,
    )
    assert reuse == OrderedDict()


@pytest.mark.parametrize(
    "payload,match",
    [
        (b"", "row-count header is truncated"),
        ((2).to_bytes(4, "little"), "row count mismatches"),
        ((1).to_bytes(4, "little"), "length header is truncated"),
        (
            (1).to_bytes(4, "little") + (100).to_bytes(4, "little") + b"x",
            "exceeds payload",
        ),
        (_wrapped_frame_payload((((0, 0, 0, 0), (1, 1, 1, 1)),)), "count mismatches"),
        (
            _wrapped_frame_payload((((0, 0, 0), (1, 1, 1, 1), (2, 2, 2, 2)),)),
            "arity mismatches",
        ),
        (_wrapped_frame_payload((((0, 0, 0, 0), (1, 1, 1, 1), (2, 2, 2, 2)),)) + b"x", "trailing bytes"),
    ],
)
def test_frame_candidate_wrapper_parser_refuses_malformed_payloads(
    payload: bytes,
    match: str,
) -> None:
    with pytest.raises(ProfilerError, match=match):
        _decode_frame_candidate_payload(
            payload,
            operator=_PARTITION_OPERATOR,
            cost_model=SignedResidualCostModel(),
        )


def test_frame_candidate_wrapper_reassembles_only_receiver_closed_frames() -> None:
    source = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)
    candidates = tuple(tuple(int(value) for value in source[:, :, channel].reshape(-1)) for channel in range(3))
    decoded = _decode_frame_candidate_payload(
        _wrapped_frame_payload((candidates,)),
        operator=_PARTITION_OPERATOR,
        cost_model=SignedResidualCostModel(),
    )
    assert decoded.receiver_closed is True
    assert decoded.selected_blocks == decoded.total_blocks == 3
    assert np.array_equal(decoded.selected_frame, source)

    incomplete = _decode_frame_candidate_payload(
        _wrapped_frame_payload(((candidates[0], None, candidates[2]),)),
        operator=_PARTITION_OPERATOR,
        cost_model=SignedResidualCostModel(),
    )
    assert incomplete.receiver_closed is False
    assert incomplete.selected_blocks == 2
    assert incomplete.selected_frame is None


def test_segnet_scoring_consumes_receiver_decoded_frame_not_mutable_selection() -> None:
    _operator, source, artifacts = _tiny_seed_artifacts()
    assert artifacts.selected_frame is not None
    artifacts.selected_frame.fill(255)

    class _CapturingScorer(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.captured: torch.Tensor | None = None

        def preprocess_input(self, value: torch.Tensor) -> torch.Tensor:
            self.captured = value.clone()
            return value

        def forward(self, value: torch.Tensor) -> torch.Tensor:
            logits = torch.zeros((value.shape[0], 5, 1, 1), dtype=torch.float32)
            logits[:, 1] = 1.0
            return logits

    scorer = _CapturingScorer()
    counters = _score_frame_artifacts(artifacts, scorer=scorer, source_frame=source)
    assert counters["segnet_mismatches"] == 0
    assert counters["segnet_pixels"] == 1
    assert scorer.captured is not None
    observed = scorer.captured[0, 0].permute(1, 2, 0).byte().numpy()
    assert np.array_equal(observed, source)
    assert not np.array_equal(observed, artifacts.selected_frame)

    artifacts.labels[0, 0] = 2
    with pytest.raises(ProfilerError, match="fresh current frozen-source SegNet argmax"):
        _score_frame_artifacts(artifacts, scorer=scorer, source_frame=source)


@pytest.mark.parametrize("stage_kind", ["final", "prepared"])
def test_resume_rejects_valid_rehashed_candidate_payload_tamper(
    tmp_path: Path,
    stage_kind: str,
) -> None:
    _operator, _source, artifacts = _tiny_seed_artifacts()
    identity_hash = "7" * 64
    receipt = _receipt_from_artifacts(identity_hash, artifacts)
    tampered_payload = _wrapped_frame_payload((((0, 0, 0, 0), (1, 1, 1, 1), (2, 2, 2, 2)),))
    assert tampered_payload != artifacts.candidate_payload
    receipt["candidate_payload_bytes"] = len(tampered_payload)
    receipt["candidate_payload_sha256"] = sha256(tampered_payload).hexdigest()

    stages, progress = _initialize_fresh_output(
        tmp_path / stage_kind,
        identity_hash=identity_hash,
        storage_preflight={"PASS": True},
        exact_argv=["python", "profile.py"],
    )
    final = stages / "frame_0000.bin"
    stage = final if stage_kind == "final" else _prepared_stage_path(final)
    stage_payload = _stage_payload(receipt, tampered_payload)
    if stage_kind == "final":
        _atomic_stage(stage, stage_payload)
    else:
        profiler_module._write_exclusive_bytes(
            stage,
            stage_payload,
            name="test interrupted prepared stage",
        )
        intent, _transaction_sha256 = _write_stage_attempt_intent(
            final,
            stage_payload,
            identity_sha256=identity_hash,
        )
    if stage_kind == "final":
        atomic_json(
            progress,
            _progress_pointer(
                identity_hash,
                next_frame=1,
                head=sha256(stage.read_bytes()).hexdigest(),
            ),
        )
    before_stage = stage.read_bytes()
    before_pointer = progress.read_bytes()

    with pytest.raises(ProfilerError, match="candidate payload does not match immutable-input semantic replay"):
        _resume_from_stage_chain(
            stages,
            progress,
            identity_hash=identity_hash,
            semantic_replay_provider=lambda _frame: artifacts.replay,
            max_frames=1,
        )
    assert stage.read_bytes() == before_stage
    assert progress.read_bytes() == before_pointer


def test_seeded_stage_resume_replays_payload_counters_and_authority(tmp_path: Path) -> None:
    _operator, _source, artifacts = _tiny_seed_artifacts()
    identity_hash = "8" * 64
    receipt = _receipt_from_artifacts(identity_hash, artifacts)
    stages, progress = _initialize_fresh_output(
        tmp_path / "seeded-resume",
        identity_hash=identity_hash,
        storage_preflight={"PASS": True},
        exact_argv=["python", "profile.py"],
    )
    stage = stages / "frame_0000.bin"
    _atomic_stage(stage, _stage_payload(receipt, artifacts.candidate_payload))
    atomic_json(
        progress,
        _progress_pointer(
            identity_hash,
            next_frame=1,
            head=sha256(stage.read_bytes()).hexdigest(),
        ),
    )

    rebuilt, receipts, counters = _resume_from_stage_chain(
        stages,
        progress,
        identity_hash=identity_hash,
        semantic_replay_provider=lambda _frame: artifacts.replay,
        max_frames=1,
    )
    assert rebuilt.summary() == artifacts.aggregate.summary()
    assert counters == artifacts.counters
    assert receipts[0]["selection_custody"] == artifacts.replay.selection_custody
    assert receipts[0]["scope"]["frame_indices"] == [0]
    assert receipts[0]["scope"]["scope_extrapolation"] == "NONE_EXACT_FRAME_INDICES_ONLY"


def test_production_profiler_refuses_raw_bypass_and_imported_calls_before_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "must-not-exist"
    monkeypatch.delenv(GOVERNED_MARKER_ENV, raising=False)
    monkeypatch.delenv(BYPASS_OVERRIDE_ENV, raising=False)
    with pytest.raises(ProfilerError, match="real governed-admission child marker"):
        run_profile(
            SimpleNamespace(allow_local_output_for_tests=False, output_root=output),
            exact_argv=["python", "profile.py"],
        )
    assert not output.exists()

    monkeypatch.setenv(BYPASS_OVERRIDE_ENV, "reviewed raw exception")
    with pytest.raises(ProfilerError, match="real governed-admission child marker"):
        run_profile(
            SimpleNamespace(allow_local_output_for_tests=False, output_root=output),
            exact_argv=["python", "profile.py"],
        )
    assert not output.exists()


def test_profiler_storage_waterfall_requires_first_existing_tier_and_validates_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    high = tmp_path / "VertigoDataTier" / "pact"
    low = tmp_path / "APDataStore" / "pact"
    high.mkdir(parents=True)
    low.mkdir(parents=True)
    monkeypatch.setattr(profiler_module, "SSD_ROOTS", (high, low))

    with pytest.raises(ProfilerError, match="first existing SSD root"):
        profiler_module._safe_output_root(
            low / "profile",
            allow_local_output_for_tests=False,
        )
    selected = profiler_module._safe_output_root(
        high / "profile",
        allow_local_output_for_tests=False,
    )
    receipt = profiler_module._storage_preflight(
        selected,
        max_frames=1,
        allow_local_output_for_tests=False,
    )
    assert receipt["existing_approved_roots"] == [str(high.resolve()), str(low.resolve())]
    profiler_module._validate_storage_preflight(receipt, expected_output_root=selected)

    reordered = json.loads(json.dumps(receipt))
    reordered["existing_approved_roots"].reverse()
    with pytest.raises(ProfilerError, match="out of order"):
        profiler_module._validate_storage_preflight(reordered, expected_output_root=selected)

    lower_receipt = json.loads(json.dumps(receipt))
    lower_output = low / "profile"
    lower_receipt["selected_root"] = str(lower_output)
    with pytest.raises(ProfilerError, match="persisted SSD selection scope"):
        profiler_module._validate_storage_preflight(
            lower_receipt,
            expected_output_root=lower_output,
        )


def test_creation_storage_identity_excludes_only_observations_and_binds_selection(
    tmp_path: Path,
) -> None:
    output = tmp_path / "stable-storage-identity"
    preflight = _certified_preflight(output)
    baseline = profiler_module._creation_storage_identity_from_preflight(
        preflight,
        expected_output_root=output.resolve(),
    )
    observation_drift = json.loads(json.dumps(preflight))
    observation_drift["free_bytes_before"] -= 1
    observation_drift["filesystem_anchor"] = str((tmp_path / "created-parent").resolve())
    assert (
        profiler_module._creation_storage_identity_from_preflight(
            observation_drift,
            expected_output_root=output.resolve(),
        )
        == baseline
    )
    assert baseline["volatile_fields_excluded"] == [
        "filesystem_anchor",
        "free_bytes_before",
    ]

    selection_drift = json.loads(json.dumps(preflight))
    selection_drift["existing_approved_roots"] = [preflight["waterfall_order"][0]]
    assert (
        profiler_module._creation_storage_identity_from_preflight(
            selection_drift,
            expected_output_root=output.resolve(),
        )
        != baseline
    )


def test_profiler_resume_preserves_bound_lower_tier_when_higher_tier_appears(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    high = tmp_path / "VertigoDataTier" / "pact"
    low = tmp_path / "APDataStore" / "pact"
    low.mkdir(parents=True)
    monkeypatch.setattr(profiler_module, "SSD_ROOTS", (high, low))
    output = profiler_module._safe_output_root(
        low / "existing-profile",
        allow_local_output_for_tests=False,
    )
    creation = profiler_module._storage_preflight(
        output,
        max_frames=1,
        allow_local_output_for_tests=False,
    )
    assert creation["selection_scope"] == profiler_module.FRESH_STORAGE_SELECTION_SCOPE
    assert creation["existing_approved_roots"] == [str(low.resolve())]
    identity = _certified_identity(
        output_root=output,
        creation_preflight=creation,
    )
    identity_hash = sha256(profiler_module.canonical_json_bytes(identity)).hexdigest()
    argv = ["python", "profile.py", "--rss-cap-mb", "512", "--timeout-seconds", "60"]
    _production_initialize_fresh_output(
        output,
        identity_hash=identity_hash,
        storage_preflight=creation,
        exact_argv=argv,
        identity=identity,
    )

    high.mkdir(parents=True)
    with pytest.raises(ProfilerError, match="first existing SSD root"):
        profiler_module._safe_output_root(
            output,
            allow_local_output_for_tests=False,
        )
    resumed = profiler_module._safe_output_root(
        output,
        allow_local_output_for_tests=False,
        resume=True,
    )
    current = profiler_module._storage_preflight(
        resumed,
        max_frames=1,
        allow_local_output_for_tests=False,
        resume=True,
    )
    assert current["selection_scope"] == profiler_module.RESUME_STORAGE_SELECTION_SCOPE
    assert current["existing_approved_roots"] == [str(high.resolve()), str(low.resolve())]
    profiler_module._validate_storage_preflight(current, expected_output_root=resumed)
    assert creation["existing_approved_roots"] == [str(low.resolve())]
    assert (
        profiler_module._load_identity_bound_creation_storage_identity(resumed) == identity["creation_storage_identity"]
    )
    _validate_resume_root(
        resumed,
        expected_identity=identity,
        expected_rebuild_argv=argv,
        max_frames=1,
    )


@pytest.mark.parametrize(
    ("kind", "name"),
    [
        ("file", "GT cache"),
        ("directory", "feature cache root"),
        ("directory", "scorer upstream root"),
        ("component", "profiler output"),
    ],
)
def test_profiler_refuses_symlinked_input_and_output_paths_before_resolution(
    tmp_path: Path,
    kind: str,
    name: str,
) -> None:
    target = tmp_path / f"{kind}-target"
    alias = tmp_path / f"{kind}-alias"
    if kind == "file":
        target.write_bytes(b"source")
        alias.symlink_to(target)
        supplied = alias
    else:
        target.mkdir()
        alias.symlink_to(target, target_is_directory=True)
        supplied = alias / "profile" if kind == "component" else alias
    with pytest.raises(ProfilerError, match="may not traverse a symlink"):
        profiler_module._resolve_without_symlink_components(supplied, name=name)


def test_profiler_refuses_hardlinked_gt_cache(tmp_path: Path) -> None:
    gt_cache = tmp_path / "gt.npz"
    alias = tmp_path / "gt-hardlink.npz"
    gt_cache.write_bytes(b"stored-npz")
    profiler_module.os.link(gt_cache, alias)
    resolved = profiler_module._resolve_without_symlink_components(gt_cache, name="GT cache")
    with pytest.raises(ProfilerError, match="link count one"):
        profiler_module._require_local_regular_file(resolved, name="GT cache")


def test_creation_identity_rejects_omitted_higher_tier_in_sibling_custody(
    tmp_path: Path,
) -> None:
    root = tmp_path / "identity-bound-preflight"
    preflight = _certified_preflight(root)
    preflight["existing_approved_roots"] = list(preflight["waterfall_order"])
    identity = _certified_identity(output_root=root)
    identity["creation_storage_identity"] = profiler_module._creation_storage_identity_from_preflight(
        preflight,
        expected_output_root=root.resolve(),
    )
    identity_hash = sha256(profiler_module.canonical_json_bytes(identity)).hexdigest()
    argv = ["python", "profile.py", "--rss-cap-mb", "512", "--timeout-seconds", "60"]
    _production_initialize_fresh_output(
        root,
        identity_hash=identity_hash,
        storage_preflight=preflight,
        exact_argv=argv,
        identity=identity,
    )
    for name in (
        STAGING_SCRATCH_NAME,
        profiler_module.PROGRESS_NAME,
        OUTPUT_CERTIFICATION_NAME,
    ):
        path = root / name
        value = json.loads(path.read_text(encoding="utf-8"))
        value["storage_preflight"]["existing_approved_roots"] = [preflight["waterfall_order"][1]]
        atomic_json(path, value)
    with pytest.raises(ProfilerError, match=r"staging scratch identity|creation storage"):
        _validate_output_certification(
            root,
            final_output_root=root,
            expected_identity=identity,
            expected_rebuild_argv=argv,
        )


def test_rewritten_creation_preflight_changes_identity_and_breaks_stage_chain_root(
    tmp_path: Path,
) -> None:
    root = tmp_path / "rewritten-preflight"
    preflight = _certified_preflight(root)
    preflight["existing_approved_roots"] = list(preflight["waterfall_order"])
    identity = _certified_identity(output_root=root)
    identity["creation_storage_identity"] = profiler_module._creation_storage_identity_from_preflight(
        preflight,
        expected_output_root=root.resolve(),
    )
    identity_hash = sha256(profiler_module.canonical_json_bytes(identity)).hexdigest()
    argv = ["python", "profile.py", "--rss-cap-mb", "512", "--timeout-seconds", "60"]
    stages, progress = _production_initialize_fresh_output(
        root,
        identity_hash=identity_hash,
        storage_preflight=preflight,
        exact_argv=argv,
        identity=identity,
    )
    _commit_certified_bounds_stage(
        root,
        identity_hash=identity_hash,
        stages=stages,
        progress=progress,
    )

    rewritten_identity = json.loads(json.dumps(identity))
    rewritten_identity["creation_storage_identity"]["stable_preflight"]["existing_approved_roots"] = [
        preflight["waterfall_order"][1]
    ]
    rewritten_hash = sha256(profiler_module.canonical_json_bytes(rewritten_identity)).hexdigest()
    assert rewritten_hash != identity_hash
    rewritten_progress = json.loads(progress.read_text(encoding="utf-8"))
    rewritten_progress["identity_sha256"] = rewritten_hash
    atomic_json(progress, rewritten_progress)
    with pytest.raises(ProfilerError, match="identity"):
        _resume_from_stage_chain(
            stages,
            progress,
            identity_hash=rewritten_hash,
            semantic_replay_provider=lambda _frame: pytest.fail("tampered stage replayed"),
            max_frames=1,
        )


def test_local_output_policy_never_bypasses_governed_admission(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _assert_real_governed_admission(
        allow_local_output_for_tests=False,
        env=mark_admitted_env({}),
    )
    _assert_real_governed_admission(
        allow_local_output_for_tests=True,
        env=mark_admitted_env({}),
    )
    with pytest.raises(ProfilerError, match="real governed-admission child marker"):
        _assert_real_governed_admission(
            allow_local_output_for_tests=True,
            env={},
        )
    output = tmp_path / "no-io"
    common = {
        "max_frames": 0,
        "max_nodes": 1,
        "reuse_cache_entries": 1,
        "rss_cap_mb": 512,
        "timeout_seconds": 60,
        "resume": False,
        "allow_local_output_for_tests": False,
    }
    monkeypatch.setenv(GOVERNED_MARKER_ENV, "1")
    with pytest.raises(ProfilerError, match="max_frames must be a positive integer"):
        run_profile(SimpleNamespace(**common), exact_argv=["python", "profile.py"])
    assert not output.exists()

    monkeypatch.delenv(GOVERNED_MARKER_ENV, raising=False)
    common["allow_local_output_for_tests"] = True
    with pytest.raises(ProfilerError, match="real governed-admission child marker"):
        run_profile(SimpleNamespace(**common), exact_argv=["python", "profile.py"])
    assert not output.exists()


def test_raw_governed_marker_cannot_replace_direct_safe_run_parent_attestation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exact_argv = [
        str(Path(profiler_module.sys.executable).resolve()),
        str(Path(profiler_module.__file__).resolve()),
        "--gt-cache",
        str(tmp_path / "gt.npz"),
        "--feature-cache-root",
        str(tmp_path / "features"),
        "--output-root",
        str(tmp_path / "output"),
        "--max-frames",
        "1",
        "--rss-cap-mb",
        "512",
        "--timeout-seconds",
        "60",
        "--allow-local-output-for-tests",
    ]
    args = _parse_args(exact_argv[2:])
    monkeypatch.setenv(GOVERNED_MARKER_ENV, "1")
    with pytest.raises(ProfilerError, match="exact direct-parent safe-run custody"):
        run_profile(args, exact_argv=exact_argv)
    assert not (tmp_path / "output").exists()


def test_local_pytest_output_flag_refuses_full_n600_before_io(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = SimpleNamespace(
        max_frames=600,
        max_nodes=1,
        reuse_cache_entries=1,
        fragile_margin=0.0,
        time_limit_seconds_per_block=None,
        seed_source_witness=False,
        score_segnet=False,
        mode=BOUNDS_MODE,
        rss_cap_mb=512,
        timeout_seconds=60,
        resume=False,
        allow_local_output_for_tests=True,
        output_root=tmp_path / "must-not-exist",
    )
    monkeypatch.setenv(GOVERNED_MARKER_ENV, "1")
    with pytest.raises(ProfilerError, match="tiny prefix and can never complete n600"):
        run_profile(args, exact_argv=["python", "profile.py"])
    assert not args.output_root.exists()


def test_exact_argv_reparses_to_the_effective_request_and_rejects_substitution(tmp_path: Path) -> None:
    argv = [
        str(Path(profiler_module.sys.executable).resolve()),
        str(Path(profiler_module.__file__).resolve()),
        "--gt-cache",
        str(tmp_path / "gt.npz"),
        "--feature-cache-root",
        str(tmp_path / "features"),
        "--output-root",
        str(tmp_path / "output"),
        "--mode",
        ENUMERATED_MODE,
        "--max-frames",
        "1",
        "--max-nodes",
        "17",
        "--seed-source-witness",
        "--reuse-cache-entries",
        "9",
        "--fragile-margin",
        "0.25",
        "--rss-cap-mb",
        "512",
        "--timeout-seconds",
        "60",
        "--allow-local-output-for-tests",
    ]
    args = _parse_args(argv[2:])
    assert profiler_module._attest_exact_argv(args, argv) == argv

    substituted = SimpleNamespace(**vars(args))
    substituted.max_nodes = 18
    with pytest.raises(ProfilerError, match="does not reproduce"):
        profiler_module._attest_exact_argv(substituted, argv)
    with pytest.raises(ProfilerError, match=r"executable/tool custody|does not name"):
        profiler_module._attest_exact_argv(args, ["python", "profile.py", *argv[2:]])


@pytest.mark.parametrize(("field", "value"), [("rss_cap_mb", 0), ("timeout_seconds", 0)])
def test_requested_outer_governor_caps_must_be_positive_before_io(
    field: str,
    value: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(GOVERNED_MARKER_ENV, "1")
    args = {
        "max_frames": 1,
        "max_nodes": 1,
        "reuse_cache_entries": 1,
        "rss_cap_mb": 512,
        "timeout_seconds": 60,
        "resume": False,
        "allow_local_output_for_tests": True,
    }
    args[field] = value
    with pytest.raises(ProfilerError, match=f"{field} must be a positive integer"):
        run_profile(SimpleNamespace(**args), exact_argv=["python", "profile.py"])


def test_exhaustive_selection_claim_is_per_block_and_never_global_compression() -> None:
    custody = _selection_custody(
        mode=ENUMERATED_MODE,
        counters={
            "selected_blocks": 3,
            "total_blocks": 3,
            "exhaustive_selected_blocks": 3,
            "bounded_selected_blocks": 0,
            "omitted_blocks": 0,
            "segnet_mismatches": 0,
            "segnet_pixels": 0,
        },
        seed_source_witness=False,
        receiver_closed=True,
    )
    assert custody["selection_label"] == "PER_BLOCK_RECEIVER_PUBLIC_SELECTOR_MINIMUM_EXACT"
    assert custody["exact_count_claim"] is True
    assert custody["per_block_selector_minimum_proved"] is True
    assert custody["min_description_claim"] is False
    assert custody["global_compressed_stream_minimum_claim"] is False


@pytest.mark.parametrize("claim", ["min_description_claim", "global_compressed_stream_minimum_claim"])
def test_stage_validator_refuses_global_compression_false_authority(claim: str) -> None:
    receipt = _bounds_stage_receipt("1" * 64, frame=0, previous_hash="1" * 64)
    selection = receipt["selection_custody"]
    assert isinstance(selection, dict)
    selection[claim] = True
    replay = _replay_from_receipt(receipt, b"")
    with pytest.raises(ProfilerError, match="global compressed-stream optimality"):
        _validate_against_replay(receipt, b"", replay)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("wall_seconds", 0.0, "positive|rate relation"),
        ("blocks_per_second", 9.0, "rate relation"),
        ("peak_rss_bytes", -1, "nonnegative"),
        ("custody_label", "SEMANTIC_REPLAYED", "custody label"),
    ],
)
def test_stage_timing_schema_and_internal_rate_are_fail_closed(
    field: str,
    value: object,
    match: str,
) -> None:
    receipt = _bounds_stage_receipt("2" * 64, frame=0, previous_hash="2" * 64)
    timing = receipt["timing"]
    assert isinstance(timing, dict)
    timing[field] = value
    with pytest.raises(ProfilerError, match=match):
        _validate_against_replay(receipt, b"", _replay_from_receipt(receipt, b""))


def test_valid_timing_is_explicitly_not_semantically_replayed() -> None:
    receipt = _bounds_stage_receipt("2" * 64, frame=0, previous_hash="2" * 64)
    receipt["timing"] = _stage_timing(wall_seconds=4.0, total_blocks=3, peak_rss_bytes=123)
    replay = _replay_from_receipt(receipt, b"")
    _validate_against_replay(receipt, b"", replay)
    assert receipt["timing"]["custody_label"] == TIMING_CUSTODY_LABEL


def test_certified_creation_staging_recovers_after_final_rename_interruption(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "new-output-parent" / "certified-profile"
    initial_free = 1_000_000_000
    retry_free = 900_000_000

    def changing_disk_usage(anchor: Path) -> SimpleNamespace:
        free = initial_free if Path(anchor) == tmp_path else retry_free
        return SimpleNamespace(free=free)

    monkeypatch.setattr(profiler_module.shutil, "disk_usage", changing_disk_usage)
    preflight = profiler_module._storage_preflight(
        root,
        max_frames=1,
        allow_local_output_for_tests=True,
    )
    identity = _certified_identity(output_root=root)
    identity["creation_storage_identity"] = profiler_module._creation_storage_identity_from_preflight(
        preflight,
        expected_output_root=root.resolve(),
    )
    identity_hash = sha256(profiler_module.canonical_json_bytes(identity)).hexdigest()
    argv = ["python", "profile.py", "--rss-cap-mb", "512", "--timeout-seconds", "60"]
    staging = root.with_name(f".{root.name}{CREATION_STAGING_SUFFIX}")

    def interrupt_final_rename(source: Path, destination: Path) -> None:
        if source == staging and destination == root:
            raise OSError("synthetic final-rename interruption")

    monkeypatch.setattr(
        profiler_module.feature_cache_module,
        "_MOVE_PATH_NOREPLACE_TEST_HOOK",
        interrupt_final_rename,
    )
    with pytest.raises(ProfilerError, match="no-replace move failed"):
        _production_initialize_fresh_output(
            root,
            identity_hash=identity_hash,
            storage_preflight=preflight,
            exact_argv=argv,
            identity=identity,
        )
    assert staging.is_dir()
    assert not root.exists()
    assert (staging / STAGING_SCRATCH_NAME).is_file()
    assert (staging / OUTPUT_CERTIFICATION_NAME).is_file()

    monkeypatch.setattr(profiler_module.feature_cache_module, "_MOVE_PATH_NOREPLACE_TEST_HOOK", None)
    retry_preflight = profiler_module._storage_preflight(
        root,
        max_frames=1,
        allow_local_output_for_tests=True,
    )
    assert retry_preflight["free_bytes_before"] == retry_free
    assert retry_preflight["free_bytes_before"] != preflight["free_bytes_before"]
    assert retry_preflight["filesystem_anchor"] != preflight["filesystem_anchor"]
    assert (
        profiler_module._fresh_creation_storage_identity(
            root.resolve(),
            current_preflight=retry_preflight,
        )
        == identity["creation_storage_identity"]
    )
    _production_initialize_fresh_output(
        root,
        identity_hash=identity_hash,
        storage_preflight=retry_preflight,
        exact_argv=argv,
        identity=identity,
    )
    assert root.is_dir()
    assert not staging.exists()
    _validate_output_certification(
        root,
        final_output_root=root,
        expected_identity=identity,
        expected_rebuild_argv=argv,
    )
    for name in (
        STAGING_SCRATCH_NAME,
        profiler_module.PROGRESS_NAME,
        OUTPUT_CERTIFICATION_NAME,
    ):
        record = json.loads((root / name).read_text(encoding="utf-8"))
        assert record["storage_preflight"] == preflight


@pytest.mark.parametrize("partial_write", [False, True])
def test_certified_creation_recovers_stable_prepared_file_without_pid_orphans(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    partial_write: bool,
) -> None:
    root = tmp_path / f"stable-prepared-{partial_write}"
    identity = _certified_identity(output_root=root)
    identity_hash = sha256(profiler_module.canonical_json_bytes(identity)).hexdigest()
    argv = ["python", "profile.py", "--rss-cap-mb", "512", "--timeout-seconds", "60"]
    preflight = _certified_preflight(root)
    staging = root.with_name(f".{root.name}{CREATION_STAGING_SUFFIX}")
    identity_prepared = profiler_module._creation_prepared_path(staging / IDENTITY_NAME)

    if partial_write:
        real_write = profiler_module._write_exclusive_canonical_json

        def interrupt_write(path: Path, value: dict[str, object]) -> None:
            if path == identity_prepared:
                with path.open("xb") as handle:
                    handle.write(b"{")
                    handle.flush()
                    profiler_module.os.fsync(handle.fileno())
                raise OSError("synthetic partial stable preparation")
            real_write(path, value)

        monkeypatch.setattr(profiler_module, "_write_exclusive_canonical_json", interrupt_write)
        expected_exception = OSError
        expected_error = "synthetic partial stable preparation"
    else:

        def interrupt_replace(source: Path, destination: Path) -> None:
            if source == identity_prepared and destination == staging / IDENTITY_NAME:
                raise OSError("synthetic prepared rename interruption")

        monkeypatch.setattr(
            profiler_module.feature_cache_module,
            "_MOVE_PATH_NOREPLACE_TEST_HOOK",
            interrupt_replace,
        )
        expected_exception = ProfilerError
        expected_error = "no-replace move failed"

    with pytest.raises(expected_exception, match=expected_error):
        _production_initialize_fresh_output(
            root,
            identity_hash=identity_hash,
            storage_preflight=preflight,
            exact_argv=argv,
            identity=identity,
        )
    assert staging.is_dir()
    assert (staging / STAGING_SCRATCH_NAME).is_file()
    assert identity_prepared.is_file()
    assert all(".tmp-" not in path.name for path in staging.iterdir())

    monkeypatch.undo()
    _production_initialize_fresh_output(
        root,
        identity_hash=identity_hash,
        storage_preflight=preflight,
        exact_argv=argv,
        identity=identity,
    )
    assert root.is_dir()
    assert not staging.exists()
    for name in (IDENTITY_NAME, profiler_module.PROGRESS_NAME, OUTPUT_CERTIFICATION_NAME):
        assert not profiler_module._creation_prepared_path(root / name).exists()
    for path in root.iterdir():
        if profiler_module.feature_cache_module.is_retained_name(path.name):
            profiler_module.feature_cache_module.validate_retained_file(path, role="test creation retention")


def test_certified_creation_recovers_complete_prepared_first_scratch_after_repreflight(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "prepared-scratch-parent" / "certified-profile"
    initial_free = 1_000_000_000
    retry_free = 900_000_000

    def changing_disk_usage(anchor: Path) -> SimpleNamespace:
        free = initial_free if Path(anchor) == tmp_path else retry_free
        return SimpleNamespace(free=free)

    monkeypatch.setattr(profiler_module.shutil, "disk_usage", changing_disk_usage)
    preflight = profiler_module._storage_preflight(
        root,
        max_frames=1,
        allow_local_output_for_tests=True,
    )
    identity = _certified_identity(
        output_root=root,
        creation_preflight=preflight,
    )
    identity_hash = sha256(profiler_module.canonical_json_bytes(identity)).hexdigest()
    argv = ["python", "profile.py", "--rss-cap-mb", "512", "--timeout-seconds", "60"]
    staging = root.with_name(f".{root.name}{CREATION_STAGING_SUFFIX}")
    scratch_path = staging / STAGING_SCRATCH_NAME
    scratch_prepared = profiler_module._creation_prepared_path(scratch_path)

    def interrupt_scratch_rename(source: Path, destination: Path) -> None:
        if source == scratch_prepared and destination == scratch_path:
            raise OSError("synthetic complete scratch rename interruption")

    monkeypatch.setattr(
        profiler_module.feature_cache_module,
        "_MOVE_PATH_NOREPLACE_TEST_HOOK",
        interrupt_scratch_rename,
    )
    with pytest.raises(ProfilerError, match="no-replace move failed"):
        _production_initialize_fresh_output(
            root,
            identity_hash=identity_hash,
            storage_preflight=preflight,
            exact_argv=argv,
            identity=identity,
        )
    assert {path.name for path in staging.iterdir()} == {scratch_prepared.name}
    prepared_record = json.loads(scratch_prepared.read_text(encoding="utf-8"))
    assert prepared_record["storage_preflight"] == preflight
    assert not scratch_path.exists()
    assert not root.exists()

    monkeypatch.setattr(profiler_module.feature_cache_module, "_MOVE_PATH_NOREPLACE_TEST_HOOK", None)
    retry_preflight = profiler_module._storage_preflight(
        root,
        max_frames=1,
        allow_local_output_for_tests=True,
    )
    assert retry_preflight["free_bytes_before"] == retry_free
    assert retry_preflight["free_bytes_before"] != preflight["free_bytes_before"]
    assert retry_preflight["filesystem_anchor"] != preflight["filesystem_anchor"]
    assert (
        profiler_module._fresh_creation_storage_identity(
            root.resolve(),
            current_preflight=retry_preflight,
        )
        == identity["creation_storage_identity"]
    )
    _production_initialize_fresh_output(
        root,
        identity_hash=identity_hash,
        storage_preflight=retry_preflight,
        exact_argv=argv,
        identity=identity,
    )
    assert root.is_dir()
    assert not staging.exists()
    assert not scratch_prepared.exists()
    _validate_output_certification(
        root,
        final_output_root=root,
        expected_identity=identity,
        expected_rebuild_argv=argv,
    )
    for name in (
        STAGING_SCRATCH_NAME,
        profiler_module.PROGRESS_NAME,
        OUTPUT_CERTIFICATION_NAME,
    ):
        record = json.loads((root / name).read_text(encoding="utf-8"))
        assert record["storage_preflight"] == preflight


def test_complete_prepared_first_scratch_rejects_stable_storage_drift(
    tmp_path: Path,
) -> None:
    root = tmp_path / "prepared-stable-drift"
    staging = root.with_name(f".{root.name}{CREATION_STAGING_SUFFIX}")
    staging.mkdir()
    current_preflight = _certified_preflight(root)
    drifted_preflight = json.loads(json.dumps(current_preflight))
    drifted_preflight["required_free_bytes"] -= 1
    record = profiler_module._staging_scratch_record(
        output_root=root.resolve(),
        identity_sha256="a" * 64,
        exact_rebuild_argv=["python", "profile.py"],
        storage_preflight=drifted_preflight,
    )
    prepared = profiler_module._creation_prepared_path(staging / STAGING_SCRATCH_NAME)
    prepared.write_bytes(profiler_module.canonical_json_bytes(record) + b"\n")
    before = prepared.read_bytes()
    with pytest.raises(ProfilerError, match="stable storage custody"):
        profiler_module._fresh_creation_storage_identity(
            root.resolve(),
            current_preflight=current_preflight,
        )
    assert prepared.read_bytes() == before
    assert not (staging / STAGING_SCRATCH_NAME).exists()


def test_prepared_first_scratch_read_error_blocks_without_unlink_or_rewrite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "prepared-scratch-read-error"
    staging = root.with_name(f".{root.name}{CREATION_STAGING_SUFFIX}")
    staging.mkdir()
    preflight = _certified_preflight(root)
    identity = _certified_identity(
        output_root=root,
        creation_preflight=preflight,
    )
    identity_hash = sha256(profiler_module.canonical_json_bytes(identity)).hexdigest()
    argv = ["python", "profile.py", "--rss-cap-mb", "512", "--timeout-seconds", "60"]
    record = profiler_module._staging_scratch_record(
        output_root=root.resolve(),
        identity_sha256=identity_hash,
        exact_rebuild_argv=argv,
        storage_preflight=preflight,
    )
    scratch = staging / STAGING_SCRATCH_NAME
    prepared = profiler_module._creation_prepared_path(scratch)
    prepared.write_bytes(profiler_module.canonical_json_bytes(record) + b"\n")
    before = prepared.read_bytes()
    real_read_bound = profiler_module._read_bound_bytes

    def fail_prepared_read(path: Path, *, name: str) -> object:
        if path == prepared:
            raise ProfilerError("synthetic prepared scratch custody read failed")
        return real_read_bound(path, name=name)

    monkeypatch.setattr(profiler_module, "_read_bound_bytes", fail_prepared_read)
    with pytest.raises(ProfilerError, match="custody read failed"):
        _production_initialize_fresh_output(
            root,
            identity_hash=identity_hash,
            storage_preflight=preflight,
            exact_argv=argv,
            identity=identity,
        )
    monkeypatch.undo()
    assert prepared.read_bytes() == before
    assert not scratch.exists()
    assert not root.exists()
    assert {path.name for path in staging.iterdir()} == {prepared.name}


def test_later_prepared_creation_read_error_blocks_without_unlink_or_rewrite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    final_path = tmp_path / "identity.json"
    prepared = profiler_module._creation_prepared_path(final_path)
    expected = {"schema": "synthetic.creation.v1", "value": 1}
    atomic_json(prepared, expected)
    before = prepared.read_bytes()
    real_read_bound = profiler_module._read_bound_bytes

    def fail_prepared_read(path: Path, *, name: str) -> object:
        if path == prepared:
            raise ProfilerError("synthetic later prepared custody read failed")
        return real_read_bound(path, name=name)

    monkeypatch.setattr(profiler_module, "_read_bound_bytes", fail_prepared_read)
    with pytest.raises(ProfilerError, match="custody read failed"):
        profiler_module._materialize_creation_json(final_path, expected)
    monkeypatch.undo()
    assert prepared.read_bytes() == before
    assert not final_path.exists()


def test_certified_creation_recovers_partial_first_scratch_record(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "partial-first-scratch"
    identity = _certified_identity(output_root=root)
    identity_hash = sha256(profiler_module.canonical_json_bytes(identity)).hexdigest()
    argv = ["python", "profile.py", "--rss-cap-mb", "512", "--timeout-seconds", "60"]
    preflight = _certified_preflight(root)
    staging = root.with_name(f".{root.name}{CREATION_STAGING_SUFFIX}")
    scratch_prepared = profiler_module._creation_prepared_path(staging / STAGING_SCRATCH_NAME)
    real_write = profiler_module._write_exclusive_canonical_json

    def interrupt_scratch(path: Path, value: dict[str, object]) -> None:
        if path == scratch_prepared:
            with path.open("xb") as handle:
                handle.write(b"{")
                handle.flush()
                profiler_module.os.fsync(handle.fileno())
            raise OSError("synthetic partial first scratch write")
        real_write(path, value)

    monkeypatch.setattr(profiler_module, "_write_exclusive_canonical_json", interrupt_scratch)
    with pytest.raises(OSError, match="synthetic partial first scratch write"):
        _production_initialize_fresh_output(
            root,
            identity_hash=identity_hash,
            storage_preflight=preflight,
            exact_argv=argv,
            identity=identity,
        )
    assert {path.name for path in staging.iterdir()} == {scratch_prepared.name}
    assert ".tmp-" not in scratch_prepared.name

    monkeypatch.undo()
    _production_initialize_fresh_output(
        root,
        identity_hash=identity_hash,
        storage_preflight=preflight,
        exact_argv=argv,
        identity=identity,
    )
    assert root.is_dir()
    assert not staging.exists()
    assert (root / STAGING_SCRATCH_NAME).is_file()


@pytest.mark.parametrize("case", ["unidentified", "certification_tamper"])
def test_creation_staging_refuses_and_preserves_unidentified_or_drifted_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    case: str,
) -> None:
    root = tmp_path / case
    identity = _certified_identity(output_root=root)
    identity_hash = sha256(profiler_module.canonical_json_bytes(identity)).hexdigest()
    argv = ["python", "profile.py", "--rss-cap-mb", "512", "--timeout-seconds", "60"]
    preflight = _certified_preflight(root)
    staging = root.with_name(f".{root.name}{CREATION_STAGING_SUFFIX}")
    if case == "unidentified":
        staging.mkdir()
        (staging / "unknown.bin").write_bytes(b"preserve")
    else:

        def interrupt_final_rename(source: Path, destination: Path) -> None:
            if source == staging and destination == root:
                raise OSError("synthetic final-rename interruption")

        monkeypatch.setattr(
            profiler_module.feature_cache_module,
            "_MOVE_PATH_NOREPLACE_TEST_HOOK",
            interrupt_final_rename,
        )
        with pytest.raises(ProfilerError, match="no-replace move failed"):
            _production_initialize_fresh_output(
                root,
                identity_hash=identity_hash,
                storage_preflight=preflight,
                exact_argv=argv,
                identity=identity,
            )
        monkeypatch.setattr(profiler_module.feature_cache_module, "_MOVE_PATH_NOREPLACE_TEST_HOOK", None)
        certification = json.loads((staging / OUTPUT_CERTIFICATION_NAME).read_text(encoding="utf-8"))
        certification["false_authority_flags"]["score_authority"] = True
        (staging / OUTPUT_CERTIFICATION_NAME).write_bytes(profiler_module.canonical_json_bytes(certification) + b"\n")
    before = {path.name: path.read_bytes() for path in staging.iterdir() if path.is_file()}
    with pytest.raises(ProfilerError, match=r"unidentified|certification"):
        _production_initialize_fresh_output(
            root,
            identity_hash=identity_hash,
            storage_preflight=preflight,
            exact_argv=argv,
            identity=identity,
        )
    after = {path.name: path.read_bytes() for path in staging.iterdir() if path.is_file()}
    assert after == before
    assert not root.exists()


def test_resume_validates_whole_identity_object_and_rederived_hash(tmp_path: Path) -> None:
    root = tmp_path / "identity-resume"
    identity, _identity_hash, argv, _preflight, _stages, _progress = _create_certified_test_root(root)
    _validate_resume_root(
        root,
        expected_identity=identity,
        expected_rebuild_argv=argv,
        max_frames=1,
    )
    substituted = json.loads(json.dumps(identity))
    substituted["sources"]["synthetic_source"]["sha256"] = "c" * 64
    atomic_json(root / IDENTITY_NAME, substituted)
    with pytest.raises(ProfilerError, match="persisted profile identity"):
        _validate_resume_root(
            root,
            expected_identity=identity,
            expected_rebuild_argv=argv,
            max_frames=1,
        )


def test_clean_room_final_receipt_binds_identity_stage_progress_stream_and_timing(tmp_path: Path) -> None:
    root = tmp_path / "clean-room"
    identity, identity_hash, argv, _preflight, stages, progress = _create_certified_test_root(root)
    committed = _commit_certified_bounds_stage(
        root,
        identity_hash=identity_hash,
        stages=stages,
        progress=progress,
    )
    receipt = _clean_room_bounds_receipt(root, identity=identity, exact_argv=argv)
    _write_authorized_receipt(root, receipt=receipt, identity=identity, exact_argv=argv)
    stored = json.loads((root / RECEIPT_NAME).read_text(encoding="utf-8"))
    _validate_final_receipt(
        stored,
        output_root=root,
        expected_identity=identity,
        expected_rebuild_argv=argv,
        semantic_replay_provider=lambda _frame: _replay_from_receipt(committed, b""),
    )
    assert stored["custody"]["ordered_stage_count"] == 1
    assert stored["custody"]["identity_json_sha256"] == sha256((root / IDENTITY_NAME).read_bytes()).hexdigest()
    assert stored["custody"]["progress_pointer"] == json.loads(progress.read_text(encoding="utf-8"))
    assert stored["custody"]["progress_pointer_sha256"] == sha256(progress.read_bytes()).hexdigest()
    assert stored["custody"]["stream_accounting"] is None
    assert stored["timing_summary"]["terminal_stage_chain_sha256"] == stored["custody"]["terminal_stage_chain_sha256"]
    assert stored["timing_summary"]["semantically_replayable"] is False


def test_terminal_receipt_revalidates_output_root_retention_roles(tmp_path: Path) -> None:
    root = tmp_path / "terminal-retained-role"
    identity, identity_hash, argv, _preflight, stages, progress = _create_certified_test_root(root)
    committed = _commit_certified_bounds_stage(
        root,
        identity_hash=identity_hash,
        stages=stages,
        progress=progress,
    )
    receipt = _clean_room_bounds_receipt(root, identity=identity, exact_argv=argv)
    _write_authorized_receipt(root, receipt=receipt, identity=identity, exact_argv=argv)
    unknown = root / "operator-note.bin"
    unknown.write_bytes(b"preserve-but-never-authorize")
    snapshot = profiler_module._read_bound_bytes(unknown, name="unknown root retention")
    profiler_module._unlink_bound(unknown, snapshot, name="unknown root retention")

    with pytest.raises(ProfilerError, match=r"role-unproven retained custody|terminal output root"):
        _validate_final_receipt(
            receipt,
            output_root=root,
            expected_identity=identity,
            expected_rebuild_argv=argv,
            semantic_replay_provider=lambda _frame: _replay_from_receipt(committed, b""),
        )


def test_hash_valid_retained_creation_role_requires_certified_payload_provenance(tmp_path: Path) -> None:
    root = tmp_path / "retained-creation-foreign"
    identity, _identity_hash, argv, _preflight, _stages, _progress = _create_certified_test_root(root)
    foreign = profiler_module._creation_prepared_path(root / IDENTITY_NAME)
    foreign.write_bytes(b"hash-valid-but-foreign-creation-payload")
    snapshot = profiler_module._read_bound_bytes(foreign, name="foreign creation role")
    profiler_module._unlink_bound(foreign, snapshot, name="foreign creation role")
    tree_before = {path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()}

    with pytest.raises(ProfilerError, match="retained creation payload lacks certified target/prefix provenance"):
        _validate_resume_root(
            root,
            expected_identity=identity,
            expected_rebuild_argv=argv,
            max_frames=1,
        )
    assert {path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()} == tree_before


def test_final_receipt_post_exchange_cut_reconciles_before_strict_terminal_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "receipt-post-exchange"
    identity, identity_hash, argv, _preflight, stages, progress = _create_certified_test_root(root)
    committed = _commit_certified_bounds_stage(
        root,
        identity_hash=identity_hash,
        stages=stages,
        progress=progress,
    )
    receipt = _clean_room_bounds_receipt(root, identity=identity, exact_argv=argv)
    receipt_path = root / RECEIPT_NAME
    priors, authorization_sha256 = _persist_test_receipt_authorization(
        root,
        receipt=receipt,
        identity=identity,
        exact_argv=argv,
    )

    def cut_after_exchange(source: Path, destination: Path) -> None:
        if destination == receipt_path:
            raise OSError("synthetic receipt post-exchange cut")

    monkeypatch.setattr(
        profiler_module.feature_cache_module,
        "_ATOMIC_POST_EXCHANGE_TEST_HOOK",
        cut_after_exchange,
    )
    with pytest.raises(RuntimeError, match="post-linearization interruption"):
        _atomic_json(
            receipt_path,
            receipt,
            expected_prior_payloads=priors,
            consumer_authorization_sha256=authorization_sha256,
        )
    assert receipt_path.exists()
    assert profiler_module._active_atomic_scratch(receipt_path)

    monkeypatch.setattr(profiler_module.feature_cache_module, "_ATOMIC_POST_EXCHANGE_TEST_HOOK", None)
    reconciled = profiler_module._reconcile_committed_atomic_json(
        receipt_path,
        name="test existing final receipt",
        expected_prior_payloads=(),
        expected_consumer_authorization_sha256=authorization_sha256,
    )
    assert reconciled == receipt
    assert not profiler_module._active_atomic_scratch(receipt_path)
    _validate_final_receipt(
        receipt,
        output_root=root,
        expected_identity=identity,
        expected_rebuild_argv=argv,
        semantic_replay_provider=lambda _frame: _replay_from_receipt(committed, b""),
    )


def test_existing_receipt_a_to_b_fresh_process_resume_uses_authorized_a(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "receipt-a-to-b"
    identity = _certified_identity(output_root=root)
    identity["config"]["profiled_frame_limit"] = 2  # type: ignore[index]
    identity, identity_hash, argv, _preflight, stages, progress = _create_certified_test_root(
        root,
        identity=identity,
    )
    first = _commit_certified_bounds_stage(
        root,
        identity_hash=identity_hash,
        stages=stages,
        progress=progress,
    )
    receipt_a = _clean_room_bounds_receipt(root, identity=identity, exact_argv=argv)
    _write_authorized_receipt(root, receipt=receipt_a, identity=identity, exact_argv=argv)

    first_path = stages / "frame_0000.bin"
    second = _bounds_stage_receipt(
        identity_hash,
        frame=1,
        previous_hash=sha256(first_path.read_bytes()).hexdigest(),
    )
    second_path = stages / "frame_0001.bin"
    _atomic_stage(second_path, _stage_payload(second, b""))
    atomic_json(
        progress,
        {
            **json.loads(progress.read_text(encoding="utf-8")),
            "next_frame": 2,
            "stage_chain_head_sha256": sha256(second_path.read_bytes()).hexdigest(),
        },
    )
    receipt_b = _clean_room_bounds_receipt(root, identity=identity, exact_argv=argv)
    priors, authorization_sha256 = _persist_test_receipt_authorization(
        root,
        receipt=receipt_b,
        identity=identity,
        exact_argv=argv,
    )
    assert priors == (profiler_module.canonical_json_bytes(receipt_a) + b"\n",)
    receipt_path = root / RECEIPT_NAME

    def cut_after_exchange(_source: Path, destination: Path) -> None:
        if destination == receipt_path:
            raise OSError("synthetic A-to-B post-exchange cut")

    monkeypatch.setattr(
        profiler_module.feature_cache_module,
        "_ATOMIC_POST_EXCHANGE_TEST_HOOK",
        cut_after_exchange,
    )
    with pytest.raises(RuntimeError, match="post-linearization interruption"):
        _atomic_json(
            receipt_path,
            receipt_b,
            expected_prior_payloads=priors,
            consumer_authorization_sha256=authorization_sha256,
        )
    monkeypatch.setattr(profiler_module.feature_cache_module, "_ATOMIC_POST_EXCHANGE_TEST_HOOK", None)
    tree_before = {path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()}
    with pytest.raises(ProfilerError, match="atomic transaction custody"):
        profiler_module._reconcile_committed_atomic_json(
            receipt_path,
            name="B-as-prior must not self-authorize",
            expected_prior_payloads=(profiler_module.canonical_json_bytes(receipt_b) + b"\n",),
            expected_consumer_authorization_sha256=authorization_sha256,
        )
    with pytest.raises(ProfilerError, match="atomic transaction custody"):
        profiler_module._reconcile_committed_atomic_json(
            receipt_path,
            name="foreign receipt authorization",
            expected_prior_payloads=priors,
            expected_consumer_authorization_sha256="f" * 64,
        )
    assert {path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()} == tree_before

    _validate_resume_root(
        root,
        expected_identity=identity,
        expected_rebuild_argv=argv,
        max_frames=2,
    )
    assert not profiler_module._active_atomic_scratch(receipt_path)
    terminal, terminal_receipts = _terminal_custody(
        root,
        expected_identity=identity,
        expected_rebuild_argv=argv,
    )
    assert terminal["ordered_stage_count"] == 2
    assert terminal_receipts == [first, second]


@pytest.mark.parametrize("cut", [0, 1, "midpoint", "n_minus_1"])
def test_partial_attempt_json_construction_retries_same_ordinal(
    tmp_path: Path,
    cut: int | str,
) -> None:
    identity_hash = "8" * 64
    argv = ["python", "profile.py"]
    stages, _progress = _initialize_fresh_output(
        tmp_path / str(cut),
        identity_hash=identity_hash,
        storage_preflight={"PASS": True},
        exact_argv=argv,
    )
    receipt = _bounds_stage_receipt(identity_hash, frame=0, previous_hash=identity_hash)
    payload = _stage_payload(receipt, b"")
    final = stages / "frame_0000.bin"
    transaction_root = profiler_module._ensure_stage_attempt_directory(
        stages,
        identity_sha256=identity_hash,
        frame=0,
        attempt=0,
        authorize_mutation=lambda: None,
    )
    attempt_path = transaction_root / profiler_module.STAGE_ATTEMPT_NAME
    transaction = profiler_module._stage_attempt_transaction(
        final_path=final,
        intent_path=attempt_path,
        identity_sha256=identity_hash,
        frame=0,
        attempt=0,
        intended_payload=payload,
        exact_rebuild_argv=argv,
    )
    expected = profiler_module.canonical_json_bytes(transaction) + b"\n"
    prefix = {0: 0, 1: 1, "midpoint": len(expected) // 2, "n_minus_1": len(expected) - 1}[cut]
    profiler_module.atomic_prepared_path(attempt_path).write_bytes(expected[:prefix])

    _production_atomic_stage(
        final,
        payload,
        identity_sha256=identity_hash,
        exact_rebuild_argv=argv,
    )
    assert json.loads(attempt_path.read_text(encoding="utf-8"))["attempt"] == 0
    custody = profiler_module._validate_stage_attempt_custody(
        stages.parent,
        identity_sha256=identity_hash,
        exact_rebuild_argv=argv,
        terminal=True,
    )
    assert custody.outcomes[0]["success"]["attempt"] == 0


def test_final_receipt_precommit_prefix_is_reachable_until_exact_convergence(tmp_path: Path) -> None:
    root = tmp_path / "receipt-precommit-prefix"
    identity, identity_hash, argv, _preflight, stages, progress = _create_certified_test_root(root)
    committed = _commit_certified_bounds_stage(
        root,
        identity_hash=identity_hash,
        stages=stages,
        progress=progress,
    )
    receipt = _clean_room_bounds_receipt(root, identity=identity, exact_argv=argv)
    receipt_path = root / RECEIPT_NAME
    priors, authorization_sha256 = _persist_test_receipt_authorization(
        root,
        receipt=receipt,
        identity=identity,
        exact_argv=argv,
    )
    prepared = profiler_module.atomic_prepared_path(receipt_path)
    expected = profiler_module.canonical_json_bytes(receipt) + b"\n"
    prepared.write_bytes(expected[: max(1, len(expected) // 3)])

    with pytest.raises(ProfilerError, match="active receipt scratch"):
        profiler_module._terminal_custody(
            root,
            expected_identity=identity,
            expected_rebuild_argv=argv,
        )
    profiler_module._terminal_custody(
        root,
        expected_identity=identity,
        expected_rebuild_argv=argv,
        allow_receipt_scratch=True,
    )
    _atomic_json(
        receipt_path,
        receipt,
        expected_prior_payloads=priors,
        consumer_authorization_sha256=authorization_sha256,
    )
    assert not profiler_module._active_atomic_scratch(receipt_path)
    _validate_final_receipt(
        receipt,
        output_root=root,
        expected_identity=identity,
        expected_rebuild_argv=argv,
        semantic_replay_provider=lambda _frame: _replay_from_receipt(committed, b""),
    )


def test_profiler_atomic_reconciler_requires_exact_consumer_prior_before_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "consumer-state.json"
    prior = {"schema": "consumer-state.v1", "value": "A"}
    desired = {"schema": "consumer-state.v1", "value": "B"}
    atomic_json(target, prior)

    def cut_after_exchange(_source: Path, destination: Path) -> None:
        if destination == target:
            raise OSError("synthetic existing-target post-exchange cut")

    monkeypatch.setattr(
        profiler_module.feature_cache_module,
        "_ATOMIC_POST_EXCHANGE_TEST_HOOK",
        cut_after_exchange,
    )
    with pytest.raises(RuntimeError, match="post-linearization interruption"):
        atomic_json(target, desired)
    monkeypatch.setattr(profiler_module.feature_cache_module, "_ATOMIC_POST_EXCHANGE_TEST_HOOK", None)
    assert json.loads(target.read_text(encoding="utf-8")) == desired
    assert profiler_module._active_atomic_scratch(target)
    tree_before = {path.name: path.read_bytes() for path in tmp_path.iterdir() if path.is_file()}

    wrong_prior = profiler_module.canonical_json_bytes({"schema": "consumer-state.v1", "value": "foreign"}) + b"\n"
    with pytest.raises(ProfilerError, match="atomic transaction custody is malformed"):
        profiler_module._reconcile_committed_atomic_json(
            target,
            name="test exact-prior consumer state",
            expected_prior_payloads=(wrong_prior,),
        )
    assert {path.name: path.read_bytes() for path in tmp_path.iterdir() if path.is_file()} == tree_before

    reconciled = profiler_module._reconcile_committed_atomic_json(
        target,
        name="test exact-prior consumer state",
        expected_prior_payloads=(profiler_module.canonical_json_bytes(prior) + b"\n",),
    )
    assert reconciled == desired
    assert not profiler_module._active_atomic_scratch(target)


def test_clean_room_enumerated_receipt_binds_measured_raw_zlib_and_brotli_streams(tmp_path: Path) -> None:
    root = tmp_path / "clean-room-enumerated"
    identity = _certified_identity(
        output_root=root,
        mode=ENUMERATED_MODE,
        seed_source_witness=True,
    )
    identity, identity_hash, argv, _preflight, stages, progress = _create_certified_test_root(
        root,
        identity=identity,
    )
    _operator, _source, artifacts = _tiny_seed_artifacts()
    stage_receipt = _receipt_from_artifacts(identity_hash, artifacts)
    stage = stages / "frame_0000.bin"
    _atomic_stage(stage, _stage_payload(stage_receipt, artifacts.candidate_payload))
    pointer = json.loads(progress.read_text(encoding="utf-8"))
    pointer.update(
        {
            "next_frame": 1,
            "stage_chain_head_sha256": sha256(stage.read_bytes()).hexdigest(),
        }
    )
    atomic_json(progress, pointer)
    receipt = _clean_room_bounds_receipt(root, identity=identity, exact_argv=argv)
    _write_authorized_receipt(root, receipt=receipt, identity=identity, exact_argv=argv)
    _validate_final_receipt(
        receipt,
        output_root=root,
        expected_identity=identity,
        expected_rebuild_argv=argv,
        semantic_replay_provider=lambda _frame: artifacts.replay,
    )
    stream = receipt["custody"]["stream_accounting"]
    assert stream["raw"]["bytes"] > 0
    assert stream["zlib_level9"]["bytes"] > 0
    assert stream["brotli_quality11"]["bytes"] > 0
    for codec in ("zlib_level9", "brotli_quality11"):
        assert stream[codec]["label"] == "DETERMINISTIC_ENCODER_BYTE_COUNT_NOT_GLOBAL_COMPRESSED_STREAM_MINIMUM"
        assert stream[codec]["codec_parseback_identical_raw"] is True
        assert stream[codec]["decompressed_raw_bytes"] == stream["raw"]["bytes"]
        assert stream[codec]["decompressed_raw_sha256"] == stream["raw"]["sha256"]
    assert stream["order0_entropy"]["label"] == "ORDER0_IID_PLUGIN_IDEAL_LENGTH_ESTIMATE_NOT_UNIVERSAL_LOWER_BOUND"
    assert receipt["rd_row"]["stream_bytes"] == stream
    assert receipt["claims"]["global_compressed_stream_minimum_claim"] is False

    tampered_rows: list[dict[str, object]] = []
    for mutation in (
        lambda row: row["rd_row"].__setitem__("selected_block_count", 0),
        lambda row: row["rd_row"].__setitem__("axis", "[contest-CPU]"),
        lambda row: row["rd_row"].__setitem__("d_seg", 0.0),
        lambda row: row["rd_row"].__setitem__("rate_scope_frames", []),
        lambda row: row.__setitem__("derivation", "PERSISTED_PROSE_IS_NOT_AUTHORITY"),
        lambda row: row["positive_control"].__setitem__("witness_satisfies", False),
        lambda row: row["claims"].__setitem__("candidate_stream_emitted", False),
    ):
        changed = json.loads(json.dumps(receipt))
        mutation(changed)
        tampered_rows.append(changed)
    for changed in tampered_rows:
        with pytest.raises(ProfilerError, match=r"reconstruction|false-authority"):
            _validate_final_receipt(
                changed,
                output_root=root,
                expected_identity=identity,
                expected_rebuild_argv=argv,
                semantic_replay_provider=lambda _frame: artifacts.replay,
            )


@pytest.mark.parametrize(
    "tamper",
    [
        "identity_sha256",
        "identity_json_sha256",
        "exact_argv",
        "ordered_stage_count",
        "terminal_stage_root",
        "progress_hash",
        "stream_custody",
    ],
)
def test_clean_room_final_receipt_rejects_terminal_custody_substitution(
    tmp_path: Path,
    tamper: str,
) -> None:
    root = tmp_path / tamper
    identity, identity_hash, argv, _preflight, stages, progress = _create_certified_test_root(root)
    committed = _commit_certified_bounds_stage(
        root,
        identity_hash=identity_hash,
        stages=stages,
        progress=progress,
    )
    receipt = _clean_room_bounds_receipt(root, identity=identity, exact_argv=argv)
    changed = json.loads(json.dumps(receipt))
    if tamper == "identity_sha256":
        changed["identity_sha256"] = "0" * 64
    elif tamper == "identity_json_sha256":
        changed["custody"]["identity_json_sha256"] = "0" * 64
    elif tamper == "exact_argv":
        changed["exact_rebuild_argv"].append("--different")
    elif tamper == "ordered_stage_count":
        changed["custody"]["ordered_stage_count"] = 2
    elif tamper == "terminal_stage_root":
        changed["custody"]["terminal_stage_chain_sha256"] = "0" * 64
    elif tamper == "progress_hash":
        changed["custody"]["progress_pointer_sha256"] = "0" * 64
    else:
        changed["custody"]["stream_accounting"] = {"invented": True}
    with pytest.raises(ProfilerError, match=r"custody|identity"):
        _validate_final_receipt(
            changed,
            output_root=root,
            expected_identity=identity,
            expected_rebuild_argv=argv,
            semantic_replay_provider=lambda _frame: _replay_from_receipt(committed, b""),
        )


def test_coordinated_stage_and_progress_rehash_cannot_substitute_stale_final_root(tmp_path: Path) -> None:
    root = tmp_path / "stage-rehash"
    identity, identity_hash, argv, _preflight, stages, progress = _create_certified_test_root(root)
    committed = _commit_certified_bounds_stage(
        root,
        identity_hash=identity_hash,
        stages=stages,
        progress=progress,
    )
    receipt = _clean_room_bounds_receipt(root, identity=identity, exact_argv=argv)
    stage = stages / "frame_0000.bin"
    stage_receipt, candidate = profiler_module._parse_stage_payload(stage.read_bytes())
    stage_receipt["timing"] = _stage_timing(wall_seconds=8.0, total_blocks=3, peak_rss_bytes=456)
    stage.write_bytes(_stage_payload(stage_receipt, candidate))
    pointer = json.loads(progress.read_text(encoding="utf-8"))
    pointer["stage_chain_head_sha256"] = sha256(stage.read_bytes()).hexdigest()
    atomic_json(progress, pointer)
    with pytest.raises(ProfilerError, match="attempt transaction"):
        _validate_final_receipt(
            receipt,
            output_root=root,
            expected_identity=identity,
            expected_rebuild_argv=argv,
            semantic_replay_provider=lambda _frame: _replay_from_receipt(committed, b""),
        )


def test_clean_room_semantic_replay_rejects_fully_rehashed_false_stage(tmp_path: Path) -> None:
    root = tmp_path / "semantic-substitution"
    identity, identity_hash, argv, _preflight, stages, progress = _create_certified_test_root(root)
    _commit_certified_bounds_stage(
        root,
        identity_hash=identity_hash,
        stages=stages,
        progress=progress,
    )
    stage = stages / "frame_0000.bin"
    substituted, candidate = profiler_module._parse_stage_payload(stage.read_bytes())
    substituted["partition_custody"]["class_labels"]["sha256"] = "0" * 64
    stage.write_bytes(_stage_payload(substituted, candidate))
    pointer = json.loads(progress.read_text(encoding="utf-8"))
    pointer["stage_chain_head_sha256"] = sha256(stage.read_bytes()).hexdigest()
    atomic_json(progress, pointer)
    with pytest.raises(ProfilerError, match="attempt transaction"):
        _clean_room_bounds_receipt(root, identity=identity, exact_argv=argv)


@pytest.mark.parametrize("link_kind", ["symlink", "hardlink"])
def test_terminal_custody_rejects_linked_stage_bytes(tmp_path: Path, link_kind: str) -> None:
    root = tmp_path / link_kind
    identity, identity_hash, argv, _preflight, stages, progress = _create_certified_test_root(root)
    _commit_certified_bounds_stage(
        root,
        identity_hash=identity_hash,
        stages=stages,
        progress=progress,
    )
    stage = stages / "frame_0000.bin"
    external = tmp_path / f"external-{link_kind}.bin"
    external.write_bytes(stage.read_bytes())
    stage.unlink()
    if link_kind == "symlink":
        profiler_module.os.symlink(external, stage)
    else:
        profiler_module.os.link(external, stage)

    with pytest.raises(ProfilerError, match=r"custody read failed|local regular file with link count one"):
        _terminal_custody(
            root,
            expected_identity=identity,
            expected_rebuild_argv=argv,
        )


def test_resume_rejects_symlinked_identity_custody(tmp_path: Path) -> None:
    root = tmp_path / "identity-link"
    identity, _identity_hash, argv, _preflight, _stages, _progress = _create_certified_test_root(root)
    identity_path = root / IDENTITY_NAME
    external = tmp_path / "external-identity.json"
    external.write_bytes(identity_path.read_bytes())
    identity_path.unlink()
    profiler_module.os.symlink(external, identity_path)
    with pytest.raises(ProfilerError, match="custody read failed"):
        _validate_resume_root(
            root,
            expected_identity=identity,
            expected_rebuild_argv=argv,
            max_frames=1,
        )


def test_resume_reconciler_reads_progress_once_or_reuses_supplied_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identity_hash = "8" * 64
    stages, progress = _initialize_fresh_output(
        tmp_path / "single-read",
        identity_hash=identity_hash,
        storage_preflight={"PASS": True},
        exact_argv=["python", "profile.py"],
    )
    snapshot = json.loads(progress.read_text(encoding="utf-8"))
    real_load = profiler_module._load_canonical_object
    reads = 0

    def count_progress_load(path: Path, *, name: str) -> dict[str, object]:
        nonlocal reads
        if path == progress:
            reads += 1
        return real_load(path, name=name)

    monkeypatch.setattr(profiler_module, "_load_canonical_object", count_progress_load)
    _resume_from_stage_chain(
        stages,
        progress,
        identity_hash=identity_hash,
        semantic_replay_provider=lambda frame: _bounds_semantic_replay(frame),
        max_frames=1,
    )
    assert reads == 1

    progress.write_bytes(b"\xffsubstituted-after-validation")
    with pytest.raises(ProfilerError, match="exact validated snapshot type"):
        _resume_from_stage_chain(
            stages,
            progress,
            identity_hash=identity_hash,
            semantic_replay_provider=lambda frame: _bounds_semantic_replay(frame),
            max_frames=1,
            progress_snapshot=snapshot,
        )
    assert reads == 1
    assert progress.read_bytes() == b"\xffsubstituted-after-validation"


def test_short_intent_bound_stage_is_quarantined_losslessly_and_recomputable(tmp_path: Path) -> None:
    identity_hash = "b" * 64
    stages, progress = _initialize_fresh_output(
        tmp_path / "short-stage",
        identity_hash=identity_hash,
        storage_preflight={"PASS": True},
        exact_argv=["python", "profile.py"],
    )
    receipt = _bounds_stage_receipt(identity_hash, frame=0, previous_hash=identity_hash)
    payload = _stage_payload(receipt, b"")
    final = stages / "frame_0000.bin"
    prepared = _prepared_stage_path(final)
    interrupted = payload[: max(1, len(payload) // 3)]
    prepared.write_bytes(interrupted)
    intent, _transaction_sha256 = _write_stage_attempt_intent(
        final,
        payload,
        identity_sha256=identity_hash,
    )
    pointer_before = progress.read_bytes()

    _aggregate, receipts, counters = _resume_from_stage_chain(
        stages,
        progress,
        identity_hash=identity_hash,
        semantic_replay_provider=lambda frame: _bounds_semantic_replay(frame),
        max_frames=1,
    )

    assert receipts == []
    assert counters["total_blocks"] == 0
    assert progress.read_bytes() == pointer_before
    assert not prepared.exists()
    assert json.loads(intent.read_text(encoding="utf-8"))["schema"] == profiler_module.STAGE_ATTEMPT_TRANSACTION_SCHEMA
    recovered = profiler_module._validated_recovery_transactions(
        stages.parent,
        identity_sha256=identity_hash,
        exact_rebuild_argv=["python", "profile.py"],
    )
    assert len(recovered) == 1
    assert recovered[0]["payload_bytes"] == len(interrupted)
    assert recovered[0]["payload_sha256"] == sha256(interrupted).hexdigest()
    assert recovered[0]["manifest"]["false_authority_flags"]["score_authority"] is False

    _atomic_stage(final, payload)
    _aggregate, receipts, counters = _resume_from_stage_chain(
        stages,
        progress,
        identity_hash=identity_hash,
        semantic_replay_provider=lambda frame: _bounds_semantic_replay(frame),
        max_frames=1,
    )
    assert len(receipts) == 1
    assert counters["total_blocks"] == 3
    assert json.loads(progress.read_text(encoding="utf-8"))["next_frame"] == 1


def test_same_size_corrupt_intent_bound_stage_blocks_without_mutation(tmp_path: Path) -> None:
    identity_hash = "c" * 64
    stages, progress = _initialize_fresh_output(
        tmp_path / "corrupt-stage",
        identity_hash=identity_hash,
        storage_preflight={"PASS": True},
        exact_argv=["python", "profile.py"],
    )
    receipt = _bounds_stage_receipt(identity_hash, frame=0, previous_hash=identity_hash)
    payload = _stage_payload(receipt, b"")
    corrupt = bytearray(payload)
    corrupt[-1] ^= 1
    final = stages / "frame_0000.bin"
    prepared = _prepared_stage_path(final)
    prepared.write_bytes(corrupt)
    intent, _transaction_sha256 = _write_stage_attempt_intent(
        final,
        payload,
        identity_sha256=identity_hash,
    )
    before = {path.name: path.read_bytes() for path in stages.iterdir()}
    pointer_before = progress.read_bytes()

    with pytest.raises(ProfilerError, match="differs from its durable intent"):
        _resume_from_stage_chain(
            stages,
            progress,
            identity_hash=identity_hash,
            semantic_replay_provider=lambda frame: _bounds_semantic_replay(frame),
            max_frames=1,
        )
    assert {path.name: path.read_bytes() for path in stages.iterdir()} == before
    assert progress.read_bytes() == pointer_before


def test_recovery_manifest_requires_link_count_one_and_exact_retained_prepared_identity(
    tmp_path: Path,
) -> None:
    identity_hash = "5" * 64
    argv = ["python", "profile.py"]
    stages, progress = _initialize_fresh_output(
        tmp_path / "retained-prepared-identity",
        identity_hash=identity_hash,
        storage_preflight={"PASS": True},
        exact_argv=argv,
    )
    receipt = _bounds_stage_receipt(identity_hash, frame=0, previous_hash=identity_hash)
    payload = _stage_payload(receipt, b"")
    final = stages / "frame_0000.bin"
    prepared = _prepared_stage_path(final)
    interrupted = payload[: max(1, len(payload) // 3)]
    prepared.write_bytes(interrupted)
    _write_stage_attempt_intent(
        final,
        payload,
        identity_sha256=identity_hash,
        exact_rebuild_argv=argv,
    )
    _resume_from_stage_chain(
        stages,
        progress,
        identity_hash=identity_hash,
        semantic_replay_provider=lambda frame: _bounds_semantic_replay(frame),
        max_frames=1,
    )
    recovered = profiler_module._validated_recovery_transactions(
        stages.parent,
        identity_sha256=identity_hash,
        exact_rebuild_argv=argv,
    )
    source_identity = recovered[0]["manifest"]["prepared_source_file_identity"]
    assert isinstance(source_identity, list)
    assert len(source_identity) == 5
    assert source_identity[4] == 1

    # Same payload and allowed retained role are not authority for a second inode.
    prepared.write_bytes(interrupted)
    injected_snapshot = profiler_module._read_bound_bytes(prepared, name="injected same-byte prepared source")
    assert list(injected_snapshot.file_identity) != source_identity
    profiler_module._unlink_bound(prepared, injected_snapshot, name="injected same-byte prepared source")
    root = stages.parent
    tree_before = {path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()}
    with pytest.raises(ProfilerError, match="retained prepared-stage custody is orphaned"):
        profiler_module._validate_stage_attempt_custody(
            root,
            identity_sha256=identity_hash,
            exact_rebuild_argv=argv,
            terminal=True,
        )
    assert {path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()} == tree_before

    with pytest.raises(ProfilerError, match="file identity is malformed"):
        profiler_module._recovery_manifest(
            original=prepared,
            destination=tmp_path / "impossible-recovery.bin",
            actual_payload=interrupted,
            intended_bytes=len(payload),
            intended_sha256=sha256(payload).hexdigest(),
            identity_hash=identity_hash,
            frame=0,
            attempt=1,
            stage_attempt_transaction_sha256="a" * 64,
            prepared_source_present=True,
            prepared_source_file_identity=(1, 2, len(interrupted), 4, 2),
            exact_rebuild_argv=argv,
        )


def test_recovery_manifest_post_exchange_crash_is_reachable_and_retryable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identity_hash = "7" * 64
    argv = ["python", "profile.py"]
    stages, progress = _initialize_fresh_output(
        tmp_path / "manifest-post-exchange",
        identity_hash=identity_hash,
        storage_preflight={"PASS": True},
        exact_argv=argv,
    )
    receipt = _bounds_stage_receipt(identity_hash, frame=0, previous_hash=identity_hash)
    payload = _stage_payload(receipt, b"")
    final = stages / "frame_0000.bin"
    prepared = _prepared_stage_path(final)
    prepared.write_bytes(payload[: max(1, len(payload) // 3)])
    intent, _transaction_sha256 = _write_stage_attempt_intent(
        final,
        payload,
        identity_sha256=identity_hash,
        exact_rebuild_argv=argv,
    )
    manifest_path = (
        profiler_module._recovery_transaction_path(
            stages,
            identity_hash=identity_hash,
            frame=0,
            attempt=0,
        )
        / profiler_module.RECOVERY_MANIFEST_NAME
    )

    def cut_manifest_after_exchange(source: Path, destination: Path) -> None:
        if destination == manifest_path:
            raise OSError("synthetic recovery manifest post-exchange cut")

    monkeypatch.setattr(
        profiler_module.feature_cache_module,
        "_ATOMIC_POST_EXCHANGE_TEST_HOOK",
        cut_manifest_after_exchange,
    )
    with pytest.raises(RuntimeError, match="post-linearization interruption"):
        _resume_from_stage_chain(
            stages,
            progress,
            identity_hash=identity_hash,
            semantic_replay_provider=lambda frame: _bounds_semantic_replay(frame),
            max_frames=1,
        )
    assert manifest_path.exists()
    assert profiler_module._active_atomic_scratch(manifest_path)
    pointer_before = progress.read_bytes()

    monkeypatch.setattr(profiler_module.feature_cache_module, "_ATOMIC_POST_EXCHANGE_TEST_HOOK", None)
    _aggregate, receipts, counters = _resume_from_stage_chain(
        stages,
        progress,
        identity_hash=identity_hash,
        semantic_replay_provider=lambda frame: _bounds_semantic_replay(frame),
        max_frames=1,
    )
    assert receipts == []
    assert counters["total_blocks"] == 0
    assert progress.read_bytes() == pointer_before
    assert not profiler_module._active_atomic_scratch(manifest_path)
    assert json.loads(intent.read_text(encoding="utf-8"))["schema"] == profiler_module.STAGE_ATTEMPT_TRANSACTION_SCHEMA
    retained = [
        path
        for parent in (manifest_path.parent, stages)
        for path in parent.iterdir()
        if profiler_module.feature_cache_module.is_retained_name(path.name)
    ]
    assert retained
    for path in retained:
        profiler_module.feature_cache_module.validate_retained_file(path, role="recovery retained custody")


def test_recovery_copy_then_source_retention_crash_is_dual_custody_and_retryable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identity_hash = "9" * 64
    argv = ["python", "profile.py"]
    stages, progress = _initialize_fresh_output(
        tmp_path / "recovery-dual-custody",
        identity_hash=identity_hash,
        storage_preflight={"PASS": True},
        exact_argv=argv,
    )
    receipt = _bounds_stage_receipt(identity_hash, frame=0, previous_hash=identity_hash)
    payload = _stage_payload(receipt, b"")
    final = stages / "frame_0000.bin"
    prepared = _prepared_stage_path(final)
    interrupted = payload[: max(1, len(payload) // 3)]
    prepared.write_bytes(interrupted)
    intent, _transaction_sha256 = _write_stage_attempt_intent(
        final,
        payload,
        identity_sha256=identity_hash,
        exact_rebuild_argv=argv,
    )
    destination = (
        profiler_module._recovery_transaction_path(
            stages,
            identity_hash=identity_hash,
            frame=0,
            attempt=0,
        )
        / profiler_module.RECOVERY_PAYLOAD_NAME
    )
    real_unlink = profiler_module._unlink_bound

    def cut_before_source_retention(path: Path, snapshot: object, *, name: str) -> None:
        if path == prepared:
            raise OSError("synthetic source-retention cut")
        real_unlink(path, snapshot, name=name)  # type: ignore[arg-type]

    monkeypatch.setattr(profiler_module, "_unlink_bound", cut_before_source_retention)
    pointer_before = progress.read_bytes()
    with pytest.raises(OSError, match="source-retention cut"):
        _resume_from_stage_chain(
            stages,
            progress,
            identity_hash=identity_hash,
            semantic_replay_provider=lambda frame: _bounds_semantic_replay(frame),
            max_frames=1,
        )
    assert prepared.read_bytes() == interrupted
    assert destination.read_bytes() == interrupted
    assert json.loads(intent.read_bytes())["schema"] == profiler_module.STAGE_ATTEMPT_TRANSACTION_SCHEMA
    assert progress.read_bytes() == pointer_before

    monkeypatch.setattr(profiler_module, "_unlink_bound", real_unlink)
    _aggregate, receipts, counters = _resume_from_stage_chain(
        stages,
        progress,
        identity_hash=identity_hash,
        semantic_replay_provider=lambda frame: _bounds_semantic_replay(frame),
        max_frames=1,
    )
    assert receipts == []
    assert counters["total_blocks"] == 0
    assert progress.read_bytes() == pointer_before
    assert not prepared.exists()
    assert json.loads(intent.read_text(encoding="utf-8"))["schema"] == profiler_module.STAGE_ATTEMPT_TRANSACTION_SCHEMA
    recovered = profiler_module._validated_recovery_transactions(
        stages.parent,
        identity_sha256=identity_hash,
        exact_rebuild_argv=argv,
    )
    assert len(recovered) == 1
    assert recovered[0]["payload_sha256"] == sha256(interrupted).hexdigest()


def test_validated_progress_snapshot_rejects_same_byte_path_substitution(tmp_path: Path) -> None:
    root = tmp_path / "snapshot-substitution"
    identity, identity_hash, argv, _preflight, _stages, _progress = _create_certified_test_root(root)
    stages, progress, _certification, snapshot = _validate_resume_root(
        root,
        expected_identity=identity,
        expected_rebuild_argv=argv,
        max_frames=1,
    )
    original = progress.read_bytes()
    replacement = tmp_path / "replacement-progress.json"
    replacement.write_bytes(original)
    profiler_module.os.replace(replacement, progress)

    with pytest.raises(ProfilerError, match="path changed after validated snapshot"):
        _resume_from_stage_chain(
            stages,
            progress,
            identity_hash=identity_hash,
            semantic_replay_provider=lambda frame: _bounds_semantic_replay(frame),
            max_frames=1,
            progress_snapshot=snapshot,
        )
    assert progress.read_bytes() == original


def test_profile_progress_post_exchange_crash_retries_through_resume_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "progress-post-exchange"
    identity, identity_hash, argv, _preflight, stages, progress = _create_certified_test_root(root)
    receipt = _bounds_stage_receipt(identity_hash, frame=0, previous_hash=identity_hash)
    stage = stages / "frame_0000.bin"
    _atomic_stage(stage, _stage_payload(receipt, b""))
    updated = json.loads(progress.read_text(encoding="utf-8"))
    updated.update(
        {
            "status": "partial",
            "next_frame": 1,
            "stage_chain_head_sha256": profiler_module.sha256_file(stage),
        }
    )

    def cut_progress_after_exchange(source: Path, destination: Path) -> None:
        if destination == progress:
            raise OSError("synthetic progress post-exchange cut")

    monkeypatch.setattr(
        profiler_module.feature_cache_module,
        "_ATOMIC_POST_EXCHANGE_TEST_HOOK",
        cut_progress_after_exchange,
    )
    with pytest.raises(RuntimeError, match="post-linearization interruption"):
        atomic_json(progress, updated)
    prepared = profiler_module.atomic_prepared_path(progress)
    assert prepared.exists()
    assert json.loads(progress.read_text(encoding="utf-8"))["next_frame"] == 1

    monkeypatch.setattr(profiler_module.feature_cache_module, "_ATOMIC_POST_EXCHANGE_TEST_HOOK", None)
    _stages, _progress, _certification, snapshot = _validate_resume_root(
        root,
        expected_identity=identity,
        expected_rebuild_argv=argv,
        max_frames=1,
    )
    assert snapshot._value()["next_frame"] == 1
    assert not prepared.exists()
    retained = [path for path in root.iterdir() if profiler_module.feature_cache_module.is_retained_name(path.name)]
    assert retained
    for path in retained:
        profiler_module.feature_cache_module.validate_retained_file(path, role="progress retained custody")


def test_profiler_scorer_snapshots_require_complete_byte_equality() -> None:
    stable = {
        role: {"path": f"/source/{role}", "bytes": 1, "sha256": "a" * 64}
        for role in (
            "executed_modules_py",
            "executed_frame_utils_py",
            "segnet_weights",
            "executed_tac_scorer_py",
        )
    }
    assert profiler_module._require_equal_scorer_source_snapshots(stable, stable) == stable
    for role in stable:
        changed = json.loads(json.dumps(stable))
        changed[role]["sha256"] = "b" * 64
        with pytest.raises(ProfilerError, match=role):
            profiler_module._require_equal_scorer_source_snapshots(stable, changed)


class _ProfileTinySegNet(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.segmentation_head = torch.nn.Sequential(torch.nn.Conv2d(3, 5, kernel_size=1, bias=True))

    def preprocess_input(self, value: torch.Tensor) -> torch.Tensor:
        return value[:, 0] if value.ndim == 5 else value

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.segmentation_head(value)


def _profile_tiny_segnet_payload(path: Path, *, offset: float = 0.0) -> tuple[_ProfileTinySegNet, bytes]:
    from safetensors.torch import save

    model = _ProfileTinySegNet()
    with torch.no_grad():
        weight = model.segmentation_head[0].weight
        bias = model.segmentation_head[0].bias
        weight.copy_(torch.arange(weight.numel(), dtype=torch.float32).reshape_as(weight) / 64.0 + offset)
        assert bias is not None
        bias.copy_(torch.arange(bias.numel(), dtype=torch.float32) / 32.0 + offset)
    payload = save(model.state_dict())
    path.write_bytes(payload)
    return model.eval(), payload


def test_profile_byte_loader_executes_admitted_payload_while_weight_path_is_foreign(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from safetensors import torch as safetensors_torch

    upstream = tmp_path / "upstream"
    (upstream / "models").mkdir(parents=True)
    modules_path = upstream / "modules.py"
    frame_utils_path = upstream / "frame_utils.py"
    scorer_path = tmp_path / "scorer.py"
    for path in (modules_path, frame_utils_path, scorer_path):
        path.write_bytes(f"source for {path.name}\n".encode())
    weights = upstream / "models" / "segnet.safetensors"
    admitted_model, admitted_payload = _profile_tiny_segnet_payload(weights)
    frozen = profiler_module._read_bound_bytes(weights, name="test admitted SegNet weights")
    canonical = _ProfileTinySegNet()
    canonical.load_state_dict(safetensors_torch.load_file(weights), strict=True)
    canonical.eval()
    rows = {
        "executed_modules_py": profiler_module.source_file_row(modules_path),
        "executed_frame_utils_py": profiler_module.source_file_row(frame_utils_path),
        "executed_tac_scorer_py": profiler_module.source_file_row(scorer_path),
        "segnet_weights": {
            "path": str(weights),
            "bytes": len(frozen.payload),
            "sha256": sha256(frozen.payload).hexdigest(),
        },
    }
    modules_module = SimpleNamespace(__file__=str(modules_path), SegNet=_ProfileTinySegNet)
    frame_utils_module = SimpleNamespace(__file__=str(frame_utils_path))
    scorer_module = SimpleNamespace(__file__=str(scorer_path))

    def source_module(module_name: str, _row: object, *, role: str) -> object:
        if module_name == "modules":
            return modules_module
        if module_name == "frame_utils":
            return frame_utils_module
        assert role == "tac.scorer"
        return scorer_module

    monkeypatch.setattr(profiler_module, "_load_profile_source_module", source_module)
    monkeypatch.setattr(profiler_module, "_require_exact_module_file", lambda *args, **kwargs: Path("."))
    monkeypatch.setattr(profiler_module, "prepend_paths", lambda _path: None)
    monkeypatch.setattr(profiler_module.torch, "set_num_threads", lambda _value: None)
    monkeypatch.setattr(profiler_module.torch, "set_num_interop_threads", lambda _value: None)
    monkeypatch.setattr(profiler_module.torch, "use_deterministic_algorithms", lambda _value: None)

    displaced = tmp_path / "admitted-weights-displaced.safetensors"
    foreign = tmp_path / "foreign.safetensors"
    foreign_custody = tmp_path / "foreign-weight-custody.safetensors"
    _foreign_model, foreign_payload = _profile_tiny_segnet_payload(foreign, offset=9.0)
    profiler_module.os.replace(weights, displaced)
    profiler_module.os.replace(foreign, weights)
    monkeypatch.setattr(
        safetensors_torch,
        "load_file",
        lambda *_args, **_kwargs: pytest.fail("pathname-based weight loading is forbidden"),
    )

    loaded = profiler_module._load_bound_scorer(upstream, rows, frozen.payload)

    assert weights.read_bytes() == foreign_payload
    assert frozen.payload == admitted_payload
    assert rows["segnet_weights"]["bytes"] == len(frozen.payload)
    assert rows["segnet_weights"]["sha256"] == sha256(frozen.payload).hexdigest()
    assert not loaded.training
    assert all(parameter.device.type == "cpu" for parameter in loaded.parameters())
    assert not any(parameter.requires_grad for parameter in loaded.parameters())
    fixed = torch.arange(12, dtype=torch.float32).reshape(1, 3, 2, 2)
    with torch.inference_mode():
        assert torch.equal(loaded(fixed), canonical(fixed))
        assert torch.equal(loaded(fixed), admitted_model(fixed))
    assert safetensors_torch.save(loaded.state_dict()) == admitted_payload

    profiler_module.os.replace(weights, foreign_custody)
    profiler_module.os.replace(displaced, weights)
    assert weights.read_bytes() == admitted_payload
    assert foreign_custody.read_bytes() == foreign_payload


def test_profile_scorer_snapshot_binds_weight_row_to_exact_payload(tmp_path: Path) -> None:
    upstream = tmp_path / "upstream"
    (upstream / "models").mkdir(parents=True)
    (upstream / "modules.py").write_bytes(b"modules source\n")
    (upstream / "frame_utils.py").write_bytes(b"frame utils source\n")
    weights = upstream / "models" / "segnet.safetensors"
    _model, payload = _profile_tiny_segnet_payload(weights)

    snapshot = profiler_module._scorer_source_bindings(upstream)

    assert snapshot.segnet_payload == payload
    assert snapshot.rows["segnet_weights"] == {
        "path": str(weights),
        "bytes": len(payload),
        "sha256": sha256(payload).hexdigest(),
    }
    _foreign, foreign_payload = _profile_tiny_segnet_payload(weights, offset=13.0)
    assert weights.read_bytes() == foreign_payload
    assert snapshot.segnet_payload == payload


class _ProfileStorageBoundaryReached(RuntimeError):
    pass


@pytest.mark.parametrize(
    "changed_role",
    [
        "executed_modules_py",
        "executed_frame_utils_py",
        "segnet_weights",
        "executed_tac_scorer_py",
        None,
    ],
)
def test_run_profile_scorer_loader_window_refuses_each_role_before_storage_or_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    changed_role: str | None,
) -> None:
    upstream = tmp_path / "upstream"
    (upstream / "models").mkdir(parents=True)
    feature_root = tmp_path / "feature-cache"
    feature_root.mkdir()
    output = tmp_path / "output"
    output.mkdir()
    sentinel = output / "operator-sentinel.bin"
    sentinel.write_bytes(b"unchanged")
    gt_cache = tmp_path / "gt.npz"
    gt_cache.write_bytes(b"gt")
    role_paths = {
        "executed_modules_py": upstream / "modules.py",
        "executed_frame_utils_py": upstream / "frame_utils.py",
        "segnet_weights": upstream / "models" / "segnet.safetensors",
        "executed_tac_scorer_py": tmp_path / "scorer.py",
    }
    for role, path in role_paths.items():
        path.write_bytes(f"stable-{role}\n".encode())

    monkeypatch.setattr(profiler_module, "EXPECTED_PAIRS", 2)
    monkeypatch.setattr(profiler_module, "EXPECTED_CAMERA_HW", (2, 2))
    monkeypatch.setattr(profiler_module, "EXPECTED_SEG_HW", (1, 1))
    monkeypatch.setattr(
        profiler_module,
        "_safe_output_root",
        lambda path, **kwargs: Path(path).resolve(),
    )
    monkeypatch.setattr(
        profiler_module,
        "_validate_feature_for_request",
        lambda root, max_frames: SimpleNamespace(
            live_logits=np.zeros((2, 5, 1, 1), dtype=np.float32),
        ),
    )
    monkeypatch.setattr(
        profiler_module,
        "_scorer_source_bindings",
        lambda _upstream: profiler_module._FrozenScorerSnapshot(
            rows={role: profiler_module.source_file_row(path) for role, path in role_paths.items()},
            segnet_payload=role_paths["segnet_weights"].read_bytes(),
        ),
    )
    monkeypatch.setattr(
        profiler_module,
        "open_stored_npy_memmap",
        lambda path, member: np.zeros((2, 2, 2, 3), dtype=np.uint8),
    )
    monkeypatch.setattr(profiler_module, "_assert_real_governed_admission", lambda **kwargs: None)
    monkeypatch.setattr(
        profiler_module,
        "attest_safe_run_parent",
        lambda **kwargs: {"schema": "test-parent"},
    )
    mutated = False

    def controlled_scorer_loader(
        _upstream: Path,
        source_snapshot: object,
        _segnet_payload: bytes,
    ) -> object:
        nonlocal mutated
        if changed_role is not None and not mutated:
            mutated = True
            role_paths[changed_role].write_bytes(role_paths[changed_role].read_bytes() + b"loader-drift")
        return SimpleNamespace()

    monkeypatch.setattr(profiler_module, "_load_bound_scorer", controlled_scorer_loader)
    storage_calls = 0

    def stop_at_storage(*args: object, **kwargs: object) -> object:
        nonlocal storage_calls
        storage_calls += 1
        raise _ProfileStorageBoundaryReached

    monkeypatch.setattr(profiler_module, "_storage_preflight", stop_at_storage)
    tokens = [
        "--gt-cache",
        str(gt_cache),
        "--feature-cache-root",
        str(feature_root),
        "--output-root",
        str(output),
        "--mode",
        ENUMERATED_MODE,
        "--seed-source-witness",
        "--max-frames",
        "1",
        "--max-nodes",
        "1",
        "--score-segnet",
        "--upstream-root",
        str(upstream),
        "--rss-cap-mb",
        "512",
        "--timeout-seconds",
        "60",
        "--allow-local-output-for-tests",
    ]
    args = profiler_module._parse_args(tokens)
    exact = [sys.executable, str(Path(profiler_module.__file__).resolve()), *tokens]

    if changed_role is None:
        with pytest.raises(_ProfileStorageBoundaryReached):
            run_profile(args, exact_argv=exact)
        assert storage_calls == 1
    else:
        with pytest.raises(ProfilerError, match=changed_role):
            run_profile(args, exact_argv=exact)
        assert storage_calls == 0
    assert {path.name for path in output.iterdir()} == {sentinel.name}
    assert sentinel.read_bytes() == b"unchanged"


def test_validated_progress_change_during_semantic_replay_has_zero_mutation(tmp_path: Path) -> None:
    root = tmp_path / "replay-progress-drift"
    identity, identity_hash, argv, _preflight, stages, progress = _create_certified_test_root(root)
    receipt = _bounds_stage_receipt(identity_hash, frame=0, previous_hash=identity_hash)
    final = stages / "frame_0000.bin"
    _atomic_stage(final, _stage_payload(receipt, b""))
    stages, progress, _certification, snapshot = _validate_resume_root(
        root,
        expected_identity=identity,
        expected_rebuild_argv=argv,
        max_frames=1,
    )
    original_progress = progress.read_bytes()
    original_stage = final.read_bytes()
    recovery_root = root / profiler_module.RECOVERY_ROOT_NAME
    recovery_before = {
        path.relative_to(recovery_root): path.read_bytes() for path in recovery_root.rglob("*") if path.is_file()
    }

    def replace_progress_during_replay(_frame: int) -> profiler_module._FrameSemanticReplay:
        replacement = tmp_path / "replacement-progress.json"
        replacement.write_bytes(original_progress)
        profiler_module.os.replace(replacement, progress)
        return _bounds_semantic_replay(0)

    with pytest.raises(ProfilerError, match="path changed after validated snapshot"):
        _resume_from_stage_chain(
            stages,
            progress,
            identity_hash=identity_hash,
            semantic_replay_provider=replace_progress_during_replay,
            max_frames=1,
            progress_snapshot=snapshot,
        )
    assert progress.read_bytes() == original_progress
    assert final.read_bytes() == original_stage
    active_names = {
        path.name for path in stages.iterdir() if not profiler_module.feature_cache_module.is_retained_name(path.name)
    }
    assert active_names == {"frame_0000.bin"}
    assert {
        path.relative_to(recovery_root): path.read_bytes() for path in recovery_root.rglob("*") if path.is_file()
    } == recovery_before


def test_final_present_without_success_outcome_completes_exact_attempt_before_pointer_adoption(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identity_hash = "3" * 64
    argv = ["python", "profile.py"]
    stages, progress = _initialize_fresh_output(
        tmp_path / "final-before-success-outcome",
        identity_hash=identity_hash,
        storage_preflight={"PASS": True},
        exact_argv=argv,
    )
    receipt = _bounds_stage_receipt(identity_hash, frame=0, previous_hash=identity_hash)
    payload = _stage_payload(receipt, b"")
    final = stages / "frame_0000.bin"

    def cut_before_success_outcome(**_kwargs: object) -> dict[str, object]:
        raise OSError("synthetic final-before-success-outcome cut")

    monkeypatch.setattr(profiler_module, "_finalize_successful_stage_attempt", cut_before_success_outcome)
    with pytest.raises(OSError, match="final-before-success-outcome"):
        _production_atomic_stage(
            final,
            payload,
            identity_sha256=identity_hash,
            exact_rebuild_argv=argv,
        )
    assert final.read_bytes() == payload
    attempt_path = (
        profiler_module._stage_attempt_directory(
            stages,
            identity_sha256=identity_hash,
            frame=0,
            attempt=0,
        )
        / profiler_module.STAGE_ATTEMPT_NAME
    )
    assert attempt_path.exists()
    assert not (attempt_path.parent / profiler_module.STAGE_SUCCESS_NAME).exists()
    assert not [path for path in stages.iterdir() if profiler_module.STAGE_INTENT_RE.fullmatch(path.name)]
    pointer_before = progress.read_bytes()

    monkeypatch.undo()
    _aggregate, committed, counters = _resume_from_stage_chain(
        stages,
        progress,
        identity_hash=identity_hash,
        semantic_replay_provider=lambda frame: _bounds_semantic_replay(frame),
        max_frames=1,
    )
    assert committed == [receipt]
    assert counters["total_blocks"] == 3
    assert progress.read_bytes() != pointer_before
    custody = profiler_module._validate_stage_attempt_custody(
        stages.parent,
        identity_sha256=identity_hash,
        exact_rebuild_argv=argv,
        terminal=True,
    )
    assert [outcome["outcome"] for outcome in custody.outcomes] == ["success"]
    success = custody.outcomes[0]["success"]
    assert success["attempt"] == 0
    assert success["final_bytes"] == len(payload)
    assert success["final_sha256"] == sha256(payload).hexdigest()


@pytest.mark.parametrize(
    ("field", "forged_value"),
    [
        ("stage_attempt_transaction_sha256", "f" * 64),
        ("frame", 1),
        ("attempt", 1),
        ("final_sha256", "e" * 64),
    ],
)
def test_forged_success_outcome_binding_blocks_byte_identically(
    tmp_path: Path,
    field: str,
    forged_value: object,
) -> None:
    identity_hash = "4" * 64
    argv = ["python", "profile.py"]
    stages, _progress = _initialize_fresh_output(
        tmp_path / field,
        identity_hash=identity_hash,
        storage_preflight={"PASS": True},
        exact_argv=argv,
    )
    receipt = _bounds_stage_receipt(identity_hash, frame=0, previous_hash=identity_hash)
    payload = _stage_payload(receipt, b"")
    final = stages / "frame_0000.bin"
    _atomic_stage(final, payload)
    success_path = (
        profiler_module._stage_attempt_directory(
            stages,
            identity_sha256=identity_hash,
            frame=0,
            attempt=0,
        )
        / profiler_module.STAGE_SUCCESS_NAME
    )
    forged = json.loads(success_path.read_text(encoding="utf-8"))
    forged[field] = forged_value
    atomic_json(success_path, forged)
    root = stages.parent
    tree_before = {path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()}

    with pytest.raises(ProfilerError, match="success outcome custody mismatch"):
        profiler_module._validate_stage_attempt_custody(
            root,
            identity_sha256=identity_hash,
            exact_rebuild_argv=argv,
            terminal=True,
        )
    assert {path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()} == tree_before


def test_two_consecutive_short_recoveries_same_frame_are_distinct_and_terminal(
    tmp_path: Path,
) -> None:
    root = tmp_path / "two-attempts"
    identity, identity_hash, argv, _preflight, stages, progress = _create_certified_test_root(root)
    receipt = _bounds_stage_receipt(identity_hash, frame=0, previous_hash=identity_hash)
    payload = _stage_payload(receipt, b"")
    final = stages / "frame_0000.bin"

    for attempt, divisor in enumerate((4, 3)):
        prepared = _prepared_stage_path(final)
        interrupted = payload[: max(1, len(payload) // divisor)]
        prepared.write_bytes(interrupted)
        intent, _transaction_sha256 = _write_stage_attempt_intent(
            final,
            payload,
            identity_sha256=identity_hash,
            exact_rebuild_argv=argv,
        )
        assert f"attempt_{attempt:08d}" in intent.parent.name
        pointer_before = progress.read_bytes()
        _aggregate, recovered_receipts, counters = _resume_from_stage_chain(
            stages,
            progress,
            identity_hash=identity_hash,
            semantic_replay_provider=lambda frame: _bounds_semantic_replay(frame),
            max_frames=1,
        )
        assert recovered_receipts == []
        assert counters["total_blocks"] == 0
        assert progress.read_bytes() == pointer_before

    recovered = profiler_module._validated_recovery_transactions(
        root,
        identity_sha256=identity_hash,
        exact_rebuild_argv=argv,
    )
    assert [row["manifest"]["attempt"] for row in recovered] == [0, 1]
    assert [row["manifest"]["frame"] for row in recovered] == [0, 0]
    assert len({row["manifest"]["transaction"] for row in recovered}) == 2

    _atomic_stage(final, payload)
    _aggregate, committed, counters = _resume_from_stage_chain(
        stages,
        progress,
        identity_hash=identity_hash,
        semantic_replay_provider=lambda frame: _bounds_semantic_replay(frame),
        max_frames=1,
    )
    assert len(committed) == 1
    assert counters["total_blocks"] == 3
    terminal, terminal_receipts = _terminal_custody(
        root,
        expected_identity=identity,
        expected_rebuild_argv=argv,
    )
    assert terminal_receipts == committed
    assert [row["manifest"]["attempt"] for row in terminal["recovery_transactions"]] == [0, 1]
    attempt_custody = profiler_module._validate_stage_attempt_custody(
        root,
        identity_sha256=identity_hash,
        exact_rebuild_argv=argv,
        terminal=True,
    )
    assert [outcome["outcome"] for outcome in attempt_custody.outcomes] == ["recovery", "recovery", "success"]
    success = attempt_custody.outcomes[-1]["success"]
    assert success["attempt"] == 2
    assert success["frame"] == 0
    assert success["final_bytes"] == len(payload)
    assert success["final_sha256"] == sha256(payload).hexdigest()
    assert success["exact_rebuild_argv"] == argv


@pytest.mark.parametrize(
    "cut",
    [
        "before_manifest",
        "during_manifest",
        "before_move",
        "after_move",
    ],
)
def test_second_short_attempt_recovery_crash_windows_are_idempotent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cut: str,
) -> None:
    identity_hash = "e" * 64
    argv = ["python", "profile.py"]
    stages, progress = _initialize_fresh_output(
        tmp_path / cut,
        identity_hash=identity_hash,
        storage_preflight={"PASS": True},
        exact_argv=argv,
    )
    receipt = _bounds_stage_receipt(identity_hash, frame=0, previous_hash=identity_hash)
    payload = _stage_payload(receipt, b"")
    final = stages / "frame_0000.bin"

    # Complete attempt zero through the ordinary recovery path.
    first_prepared = _prepared_stage_path(final)
    first_prepared.write_bytes(payload[: max(1, len(payload) // 5)])
    first_intent, _first_transaction_sha256 = _write_stage_attempt_intent(
        final,
        payload,
        identity_sha256=identity_hash,
        exact_rebuild_argv=argv,
    )
    _resume_from_stage_chain(
        stages,
        progress,
        identity_hash=identity_hash,
        semantic_replay_provider=lambda frame: _bounds_semantic_replay(frame),
        max_frames=1,
    )

    interrupted = payload[: max(1, len(payload) // 3)]
    prepared = _prepared_stage_path(final)
    prepared.write_bytes(interrupted)
    intent, transaction_sha256 = _write_stage_attempt_intent(
        final,
        payload,
        identity_sha256=identity_hash,
        exact_rebuild_argv=argv,
    )
    assert "attempt_00000001" in intent.parent.name
    transaction = profiler_module._recovery_transaction_path(
        stages,
        identity_hash=identity_hash,
        frame=0,
        attempt=1,
    )
    assert transaction == intent.parent
    destination = transaction / profiler_module.RECOVERY_PAYLOAD_NAME
    manifest_path = transaction / profiler_module.RECOVERY_MANIFEST_NAME
    manifest = profiler_module._recovery_manifest(
        original=prepared,
        destination=destination,
        actual_payload=interrupted,
        intended_bytes=len(payload),
        intended_sha256=sha256(payload).hexdigest(),
        identity_hash=identity_hash,
        frame=0,
        attempt=1,
        stage_attempt_transaction_sha256=transaction_sha256,
        prepared_source_present=True,
        prepared_source_file_identity=profiler_module._read_bound_bytes(
            prepared,
            name="test second recovery prepared source",
        ).file_identity,
        exact_rebuild_argv=argv,
    )
    if cut == "during_manifest":
        manifest_prepared = profiler_module.atomic_prepared_path(manifest_path)
        expected = profiler_module.canonical_json_bytes(manifest) + b"\n"
        manifest_prepared.write_bytes(expected[: len(expected) // 2])
    elif cut != "before_manifest":
        atomic_json(manifest_path, manifest)
        if cut == "after_move":
            profiler_module._write_exclusive_bytes(
                destination,
                interrupted,
                name="test recovery payload copy",
            )
            prepared_snapshot = profiler_module._read_bound_bytes(
                prepared,
                name="test recovery prepared source",
            )
            profiler_module._unlink_bound(
                prepared,
                prepared_snapshot,
                name="test recovery prepared source",
            )
    pointer_before = progress.read_bytes()
    _aggregate, receipts, counters = _resume_from_stage_chain(
        stages,
        progress,
        identity_hash=identity_hash,
        semantic_replay_provider=lambda frame: _bounds_semantic_replay(frame),
        max_frames=1,
    )
    assert receipts == []
    assert counters["total_blocks"] == 0
    assert progress.read_bytes() == pointer_before
    recovered = profiler_module._validated_recovery_transactions(
        stages.parent,
        identity_sha256=identity_hash,
        exact_rebuild_argv=argv,
    )
    assert [row["manifest"]["attempt"] for row in recovered] == [0, 1]


def test_unknown_recovery_content_blocks_before_orphan_progress_mutation(tmp_path: Path) -> None:
    identity_hash = "d" * 64
    stages, progress = _initialize_fresh_output(
        tmp_path / "unknown-recovery",
        identity_hash=identity_hash,
        storage_preflight={"PASS": True},
        exact_argv=["python", "profile.py"],
    )
    receipt = _bounds_stage_receipt(identity_hash, frame=0, previous_hash=identity_hash)
    final = stages / "frame_0000.bin"
    _atomic_stage(final, _stage_payload(receipt, b""))
    transaction = stages.parent / profiler_module.RECOVERY_ROOT_NAME / identity_hash / "frame_0000-attempt_00000000"
    unknown = transaction / "operator-unknown.bin"
    unknown.write_bytes(b"preserve")
    pointer_before = progress.read_bytes()
    stage_before = final.read_bytes()

    with pytest.raises(ProfilerError, match=r"unidentified .*bytes"):
        _resume_from_stage_chain(
            stages,
            progress,
            identity_hash=identity_hash,
            semantic_replay_provider=lambda frame: pytest.fail("unknown recovery must block before replay"),
            max_frames=1,
        )
    assert progress.read_bytes() == pointer_before
    assert final.read_bytes() == stage_before
    assert unknown.read_bytes() == b"preserve"


def test_hash_valid_retained_recovery_atomic_role_is_not_self_authorizing(tmp_path: Path) -> None:
    identity_hash = "d" * 64
    argv = ["python", "profile.py"]
    stages, progress = _initialize_fresh_output(
        tmp_path / "retained-recovery-atomic-foreign",
        identity_hash=identity_hash,
        storage_preflight={"PASS": True},
        exact_argv=argv,
    )
    receipt = _bounds_stage_receipt(identity_hash, frame=0, previous_hash=identity_hash)
    final = stages / "frame_0000.bin"
    _atomic_stage(final, _stage_payload(receipt, b""))
    transaction = profiler_module._stage_attempt_directory(
        stages,
        identity_sha256=identity_hash,
        frame=0,
        attempt=0,
    )
    foreign = profiler_module.atomic_prepared_path(transaction / profiler_module.RECOVERY_MANIFEST_NAME)
    foreign.write_bytes(b"hash-valid-but-foreign-recovery-atomic")
    snapshot = profiler_module._read_bound_bytes(foreign, name="foreign recovery atomic role")
    profiler_module._unlink_bound(foreign, snapshot, name="foreign recovery atomic role")
    root = stages.parent
    pointer_before = progress.read_bytes()
    tree_before = {path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()}

    with pytest.raises(ProfilerError, match="both success and recovery outcomes"):
        _resume_from_stage_chain(
            stages,
            progress,
            identity_hash=identity_hash,
            semantic_replay_provider=lambda frame: pytest.fail("foreign retained role must block before replay"),
            max_frames=1,
        )
    assert progress.read_bytes() == pointer_before
    assert {path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()} == tree_before


def test_manifest_only_recovery_without_payload_or_intent_blocks_before_orphan_adoption(tmp_path: Path) -> None:
    identity_hash = "f" * 64
    argv = ["python", "profile.py"]
    stages, progress = _initialize_fresh_output(
        tmp_path / "manifest-only-orphan",
        identity_hash=identity_hash,
        storage_preflight={"PASS": True},
        exact_argv=argv,
    )
    receipt = _bounds_stage_receipt(identity_hash, frame=0, previous_hash=identity_hash)
    final = stages / "frame_0000.bin"
    stage_payload = _stage_payload(receipt, b"")
    _atomic_stage(final, stage_payload)
    prepared = _prepared_stage_path(final)
    transaction = profiler_module._recovery_transaction_path(
        stages,
        identity_hash=identity_hash,
        frame=0,
        attempt=1,
    )
    transaction.mkdir(parents=True)
    manifest = profiler_module._recovery_manifest(
        original=prepared,
        destination=transaction / profiler_module.RECOVERY_PAYLOAD_NAME,
        actual_payload=b"",
        intended_bytes=len(stage_payload),
        intended_sha256=sha256(stage_payload).hexdigest(),
        identity_hash=identity_hash,
        frame=0,
        attempt=1,
        stage_attempt_transaction_sha256="a" * 64,
        prepared_source_present=False,
        prepared_source_file_identity=None,
        exact_rebuild_argv=argv,
    )
    atomic_json(transaction / profiler_module.RECOVERY_MANIFEST_NAME, manifest)
    pointer_before = progress.read_bytes()
    stage_before = final.read_bytes()

    with pytest.raises(ProfilerError, match="lacks its exact immutable transaction"):
        _resume_from_stage_chain(
            stages,
            progress,
            identity_hash=identity_hash,
            semantic_replay_provider=lambda frame: pytest.fail("unreachable recovery must block before replay"),
            max_frames=1,
        )
    assert progress.read_bytes() == pointer_before
    assert final.read_bytes() == stage_before


@pytest.mark.parametrize(
    "state",
    [
        "empty_root",
        "empty_identity",
        "empty_transaction",
        "payload_only",
        "prepared_manifest_with_payload",
    ],
)
def test_unreachable_recovery_state_matrix_blocks_before_replay_or_progress_mutation(
    tmp_path: Path,
    state: str,
) -> None:
    identity_hash = "9" * 64
    argv = ["python", "profile.py"]
    root = tmp_path / state
    stages, progress = _initialize_fresh_output(
        root,
        identity_hash=identity_hash,
        storage_preflight={"PASS": True},
        exact_argv=argv,
    )
    recovery_root = root / profiler_module.RECOVERY_ROOT_NAME
    identity_root = recovery_root / identity_hash
    transaction = identity_root / "frame_0000-attempt_00000000"
    if state == "empty_root":
        recovery_root.mkdir()
    elif state == "empty_identity":
        identity_root.mkdir(parents=True)
    else:
        transaction.mkdir(parents=True)
        if state == "payload_only":
            (transaction / profiler_module.RECOVERY_PAYLOAD_NAME).write_bytes(b"short")
        elif state == "prepared_manifest_with_payload":
            final = stages / "frame_0000.bin"
            intended = _stage_payload(_bounds_stage_receipt(identity_hash, frame=0, previous_hash=identity_hash), b"")
            intent, transaction_sha256 = _write_stage_attempt_intent(
                final,
                intended,
                identity_sha256=identity_hash,
                exact_rebuild_argv=argv,
            )
            manifest = profiler_module._recovery_manifest(
                original=_prepared_stage_path(final),
                destination=transaction / profiler_module.RECOVERY_PAYLOAD_NAME,
                actual_payload=b"short",
                intended_bytes=len(intended),
                intended_sha256=sha256(intended).hexdigest(),
                identity_hash=identity_hash,
                frame=0,
                attempt=0,
                stage_attempt_transaction_sha256=transaction_sha256,
                prepared_source_present=False,
                prepared_source_file_identity=None,
                exact_rebuild_argv=argv,
            )
            prepared_manifest = profiler_module.atomic_prepared_path(
                transaction / profiler_module.RECOVERY_MANIFEST_NAME
            )
            prepared_manifest.write_bytes(profiler_module.canonical_json_bytes(manifest)[:8])
            (transaction / profiler_module.RECOVERY_PAYLOAD_NAME).write_bytes(b"short")
    pointer_before = progress.read_bytes()
    tree_before = {path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()}

    if state in {"empty_root", "empty_identity", "empty_transaction"}:
        _resume_from_stage_chain(
            stages,
            progress,
            identity_hash=identity_hash,
            semantic_replay_provider=lambda frame: pytest.fail("unreachable recovery must block before replay"),
            max_frames=1,
        )
    else:
        with pytest.raises(ProfilerError):
            _resume_from_stage_chain(
                stages,
                progress,
                identity_hash=identity_hash,
                semantic_replay_provider=lambda frame: pytest.fail("unreachable recovery must block before replay"),
                max_frames=1,
            )
    assert progress.read_bytes() == pointer_before
    assert {path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()} == tree_before
