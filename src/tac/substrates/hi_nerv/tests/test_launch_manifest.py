"""NO-FAKE behavioral tests for the B1 HiNeRV launch-manifest builder.

Per CLAUDE.md "NO FAKE IMPLEMENTATIONS" Class 2 (tests-verify-behavior-not-
constants): these tests build the REAL MLX model and read the REAL PR95
factory. Every headline assertion would FAIL if ``build_launch_manifest``
returned hand-baked constants instead of computing them from the real model +
factory. The param-count and partition assertions specifically re-derive the
expected values by an INDEPENDENT path (building the model separately) so a
regression that fudges the manifest's numbers is caught.
"""

from __future__ import annotations

import json

import pytest

mx = pytest.importorskip("mlx.core")

from tac.substrates.hi_nerv.launch_manifest import (  # noqa: E402
    B1_CANONICAL_DECODER_CHANNELS,
    B1_PARAM_COUNT_EXACT,
    B1_PARITY_PARAM_TARGET,
    PR95_STAGE8_MUON_WIRED,
    PR95_STAGE8_STOP_BEFORE_FAILCLOSED,
    PR95_TOTAL_EPOCH_BUDGET,
    B1LaunchManifest,
    build_launch_manifest,
    compute_contest_score,
    count_taper_params_and_partition,
)


def _manifest(**overrides):
    kwargs = {
        "commit_sha": "deadbeef",
        "run_id": "test_run",
        "telemetry_path": "/ssd/run/telemetry.jsonl",
        "best_checkpoint_manifest_path": "/repo/.omx/research/best.json",
        "resume_command": "resume cmd",
        "exact_eval_command_stub": "exact eval stub",
    }
    kwargs.update(overrides)
    return build_launch_manifest(**kwargs)


# --- param count is computed from the REAL model, not a baked constant -------


def test_taper_produces_exactly_228903_params_from_real_model():
    """The canonical taper builds a REAL MLX model with EXACTLY 228,903 params."""
    part = count_taper_params_and_partition(B1_CANONICAL_DECODER_CHANNELS)
    assert part.total_params == B1_PARAM_COUNT_EXACT == 228_903
    assert part.total_params <= B1_PARITY_PARAM_TARGET


def test_param_count_changes_with_channels_proving_real_build():
    """A DIFFERENT taper yields a DIFFERENT (larger) count — proves real build.

    If the builder returned a baked 228,903 constant, the default 340K config
    would also report 228,903. It must not.
    """
    default = count_taper_params_and_partition((48, 40, 32, 24, 20, 16, 12))
    assert default.total_params == 340_802
    assert default.total_params != B1_PARAM_COUNT_EXACT


def test_partition_sums_to_total_no_dropped_params():
    part = count_taper_params_and_partition(B1_CANONICAL_DECODER_CHANNELS)
    assert part.muon_params + part.adamw_params == part.total_params
    assert part.sums_to_total is True


def test_partition_is_selective_not_pure_muon():
    """Muon partition is a STRICT subset — AdamW gets a nonzero share.

    Pure Muon would put ALL params under Muon (adamw_params == 0). The selective
    rule must leave biases/1D/stem/rgb/latents under AdamW.
    """
    part = count_taper_params_and_partition(B1_CANONICAL_DECODER_CHANNELS)
    assert part.adamw_params > 0, "selective partition must keep some params under AdamW"
    assert part.muon_params > 0
    assert part.muon_params < part.total_params, "not all params are Muon (not pure Muon)"
    assert part.adamw_tensor_count > 0
    assert part.muon_tensor_count > 0


def test_manifest_param_count_matches_independent_real_build():
    """Manifest's param count equals an independently-computed real-model count."""
    m = _manifest()
    independent = count_taper_params_and_partition(B1_CANONICAL_DECODER_CHANNELS)
    assert m.param_count == independent.total_params
    assert m.muon_param_count == independent.muon_params
    assert m.adamw_param_count == independent.adamw_params


# --- the 8-stage curriculum is read from the REAL factory --------------------


def test_eight_stages_read_from_factory_and_validated():
    m = _manifest()
    assert len(m.stage_list) == 8
    assert m.stages_all_validated is True
    assert sum(s.epochs_in_stage for s in m.stage_list) == PR95_TOTAL_EPOCH_BUDGET == 29650


def test_muon_only_in_stage8_stages_1_to_7_all_adamw():
    """L15: Muon activates ONLY in stage 8; stages 1-7 are all AdamW."""
    m = _manifest()
    for stage in m.stage_list[:7]:
        assert stage.uses_muon is False, f"stage {stage.stage_index} must be AdamW-only"
    assert m.stage_list[7].uses_muon is True, "stage 8 must use Muon"


def test_qat_lambda_sigma_schedule_matches_pr95_source():
    """L16/L17: QAT from S4+, lambda 0->0.01@S5->0.02@S6, sigma 0.2->0.1@S7."""
    m = _manifest()
    s = {stage.stage_index: stage for stage in m.stage_list}
    # QAT
    assert s[1].uses_qat is False and s[3].uses_qat is False
    assert s[4].uses_qat is True and s[8].uses_qat is True
    # c1a lambda
    assert s[1].c1a_lambda == 0.0 and s[4].c1a_lambda == 0.0
    assert s[5].c1a_lambda == 0.01
    assert s[6].c1a_lambda == 0.02 and s[8].c1a_lambda == 0.02
    # sigma
    assert s[1].sigma == 0.2 and s[6].sigma == 0.2
    assert s[7].sigma == 0.1 and s[8].sigma == 0.1


# --- stage-8 Muon wiring verdict (option A) ----------------------------------


def test_stage8_muon_wired_and_validated_option_a():
    m = _manifest()
    assert m.stage8_muon_status == PR95_STAGE8_MUON_WIRED
    assert m.stage8_use_muon_flag is True


# --- the self-consistency GATE -----------------------------------------------


def test_gate_passes_for_canonical_manifest():
    m = _manifest()
    assert m.manifest_complete_and_self_consistent is True


def test_gate_fails_when_sidecar_exported_without_paying_rent():
    m = _manifest(sidecar_export_enabled=True, pay_rent_gate_active=False)
    assert m.sidecar_export_enabled is True
    assert m.manifest_complete_and_self_consistent is False


def test_gate_passes_when_sidecar_exported_AND_pays_rent():
    m = _manifest(sidecar_export_enabled=True, pay_rent_gate_active=True)
    assert m.manifest_complete_and_self_consistent is True


def test_gate_fails_for_out_of_range_ema_decay():
    m = _manifest(ema_decay=1.5)
    assert m.manifest_complete_and_self_consistent is False


def test_gate_fails_for_empty_exact_eval_stub():
    m = _manifest(exact_eval_command_stub="   ")
    assert m.manifest_complete_and_self_consistent is False


def test_ema_decay_defaults_to_canonical_0997():
    m = _manifest()
    assert m.ema_decay == 0.997


def test_default_taper_340k_fails_param_count_gate():
    """Building the manifest with the 340K default must FAIL the param gate."""
    m = _manifest(decoder_channels=(48, 40, 32, 24, 20, 16, 12))
    assert m.param_count == 340_802
    assert m.param_count_confirmed is False
    assert m.manifest_complete_and_self_consistent is False


# --- bad input is rejected ---------------------------------------------------


def test_non_seven_tuple_channels_rejected():
    with pytest.raises(ValueError):
        _manifest(decoder_channels=(36, 30, 23))


def test_nonpositive_channel_rejected():
    with pytest.raises(ValueError):
        _manifest(decoder_channels=(36, 30, 23, 17, 14, 11, 0))


# --- serialization carries non-promotable markers ----------------------------


def test_as_dict_is_json_serializable_and_non_promotable():
    m = _manifest()
    payload = m.as_dict()
    text = json.dumps(payload)  # must not raise
    assert "[macOS-MLX research-signal]" in text
    assert payload["score_claim"] is False
    assert payload["promotable"] is False
    assert payload["ready_for_exact_eval_dispatch"] is False
    assert payload["manifest_complete_and_self_consistent"] is True
    assert payload["total_curriculum_epochs"] == 29650


def test_as_dict_records_pr95_source_partition_for_comparison():
    """The vehicle-divergence explanation is recorded inline for audit."""
    payload = _manifest().as_dict()
    comp = payload["pr95_source_partition_for_comparison"]
    assert comp["muon_param_count"] == 177_156
    assert comp["adamw_param_count"] == 51_802
    # V1's actual partition differs (different vehicle).
    assert payload["muon_param_count"] != comp["muon_param_count"]


# --- contest score helper uses the EXACT (nonlinear) formula -----------------


def test_compute_contest_score_uses_nonlinear_pose_term():
    # 100*0 + sqrt(10*0.1) + 25*0/X = 1.0
    assert compute_contest_score(0.0, 0.1, 0) == pytest.approx(1.0)
    # rate term: 25 * 37_545_489 / 37_545_489 == 25.0
    assert compute_contest_score(0.0, 0.0, 37_545_489) == pytest.approx(25.0)


def test_stage8_status_constants_distinct():
    assert PR95_STAGE8_MUON_WIRED != PR95_STAGE8_STOP_BEFORE_FAILCLOSED


def test_manifest_dataclass_is_frozen():
    m = _manifest()
    assert isinstance(m, B1LaunchManifest)
    with pytest.raises((AttributeError, Exception)):
        m.param_count = 1  # type: ignore[misc]
