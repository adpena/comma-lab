# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import pytest

from tac.kl_config import (
    KL_POLICY_FORMAT,
    DistillationPolicyError,
    distillation_policy_to_provenance,
    normalize_distillation_policy,
)


def test_segnet_aux_kl_normalizes_current_train_config_vocabulary() -> None:
    policy = normalize_distillation_policy(
        {
            "loss_mode": "kl_distill",
            "kl_distill_scope": "segnet_aux",
            "kl_distill_weight": 0.002,
            "kl_distill_temperature": 2.0,
            "eval_roundtrip": True,
            "class_weights": {
                "source": "lane_ps",
                "values": [1.0, 5.0, 5.0, 1.0, 1.0],
                "num_classes": 5,
                "calibration": "heldout",
            },
            "controller": {
                "enabled": True,
                "kind": "snr_ratio",
                "parameters": {"snr_target": 0.1, "eta": 0.5},
            },
        }
    )

    assert policy.family == "segnet_aux_kl"
    assert policy.scope == "segnet_aux"
    assert policy.weight == 0.002
    assert policy.temperature == 2.0
    assert policy.promotion_capable is True
    assert policy.promotion_blockers() == []
    assert policy.class_weights.values == (1.0, 5.0, 5.0, 1.0, 1.0)
    assert policy.controller.parameters == (("eta", 0.5), ("snr_target", 0.1))

    provenance = policy.to_provenance()
    assert provenance["format"] == KL_POLICY_FORMAT
    assert provenance["promotion_eligible"] is True
    assert provenance["promotion_capable"] is True
    assert provenance["class_weights"]["metadata"] == {"calibration": "heldout"}
    assert provenance["roundtrip_contract"] == {
        "eval_roundtrip": True,
        "student_uses_eval_roundtrip": True,
        "teacher_uses_eval_roundtrip": True,
        "same_as_scorer_input": True,
        "student_source": "renderer_output",
        "teacher_source": "teacher_frames",
    }
    assert json.loads(policy.to_json()) == provenance


def test_standard_config_with_default_kl_weight_stays_inactive() -> None:
    policy = normalize_distillation_policy(
        SimpleNamespace(
            loss_mode="standard",
            kl_distill_scope="none",
            kl_distill_weight=0.002,
            kl_distill_temperature=2.0,
            promotion_eligible=True,
        )
    )

    assert policy.family == "none"
    assert policy.scope == "none"
    assert policy.weight == 0.0
    assert policy.promotion_eligible is True
    assert policy.to_provenance()["promotion_blockers"] == []


def test_scoped_auxiliary_weight_is_detected_outside_kl_loss_mode() -> None:
    policy = normalize_distillation_policy(
        {
            "loss_mode": "focal_ste",
            "kl_distill_scope": "segnet_aux",
            "kl_distill_weight": 0.002,
            "kl_distill_temperature": 2.0,
            "eval_roundtrip": True,
            "promotion_eligible": True,
        }
    )

    assert policy.family == "segnet_aux_kl"
    assert policy.scope == "segnet_aux"
    assert policy.active is True
    assert policy.promotion_capable is True


def test_zero_weight_scoped_auxiliary_normalizes_to_inactive_policy() -> None:
    policy = normalize_distillation_policy(
        {
            "loss_mode": "logit_margin",
            "kl_distill_scope": "segnet_aux",
            "kl_distill_weight": 0.0,
            "kl_distill_temperature": 2.0,
            "eval_roundtrip": True,
            "promotion_eligible": True,
        }
    )

    assert policy.family == "none"
    assert policy.scope == "none"
    assert policy.weight == 0.0
    assert policy.active is False


def test_primary_scorer_kl_requires_banned_primary_ack_and_non_promotable_status() -> None:
    base = {
        "loss_mode": "kl_distill",
        "kl_distill_scope": "primary_scorer",
        "kl_distill_weight": 0.002,
        "kl_distill_temperature": 2.0,
        "forensic_reason": "reproduce historical PoseNet collapse",
    }

    with pytest.raises(DistillationPolicyError, match="allow_banned_primary=True"):
        normalize_distillation_policy({**base, "promotion_eligible": False})

    with pytest.raises(DistillationPolicyError, match="promotion_eligible=False"):
        normalize_distillation_policy({**base, "allow_banned_primary": True})

    policy = normalize_distillation_policy(
        {
            **base,
            "allow_banned_primary_kl_distill": True,
            "promotion_eligible": False,
        }
    )
    provenance = policy.to_provenance()
    assert policy.family == "primary_scorer_kl"
    assert provenance["allow_banned_primary"] is True
    assert provenance["promotion_eligible"] is False
    assert "primary_scorer_kl is forensic-only" in provenance["promotion_blockers"]


@pytest.mark.parametrize(
    ("loss_mode", "family"),
    [
        ("segnet_kl", "segnet_kl_legacy"),
        ("JBL", "jbl"),
    ],
)
def test_legacy_segnet_kl_and_jbl_require_explicit_forensic_representation(loss_mode: str, family: str) -> None:
    base = {
        "loss_mode": loss_mode,
        "kl_distill_scope": "segnet_aux",
        "kl_distill_weight": 0.002,
        "kl_distill_temperature": 2.0,
    }

    with pytest.raises(DistillationPolicyError, match="promotion_eligible=False"):
        normalize_distillation_policy({**base, "forensic_reason": "legacy audit"})

    with pytest.raises(DistillationPolicyError, match="forensic_reason"):
        normalize_distillation_policy({**base, "promotion_eligible": False})

    policy = normalize_distillation_policy(
        {
            **base,
            "promotion_eligible": False,
            "forensic_reason": "legacy soft-label lane audit",
        }
    )
    assert policy.family == family
    assert policy.scope == "segnet_aux"
    assert policy.promotion_capable is False
    assert policy.to_provenance()["forensic_reason"] == "legacy soft-label lane audit"


@pytest.mark.parametrize(
    ("override", "match"),
    [
        ({"kl_distill_weight": 0.0}, "finite positive weight"),
        ({"kl_distill_temperature": 1.5}, "temperature >= 2.0"),
        ({"student_eval_roundtrip": False}, "eval-roundtripped student and teacher"),
        ({"teacher_eval_roundtrip": False}, "eval-roundtripped student and teacher"),
        ({"same_as_scorer_input": False}, "eval-roundtripped student and teacher"),
    ],
)
def test_promotion_capable_segnet_aux_kl_is_fail_closed(override: dict[str, object], match: str) -> None:
    base = {
        "loss_mode": "kl_distill",
        "kl_distill_scope": "segnet_aux",
        "kl_distill_weight": 0.002,
        "kl_distill_temperature": 2.0,
        "eval_roundtrip": True,
        "promotion_eligible": True,
    }

    with pytest.raises(DistillationPolicyError, match=match):
        normalize_distillation_policy({**base, **override})


def test_policy_is_frozen_and_provenance_serializer_accepts_mapping() -> None:
    policy = normalize_distillation_policy(
        {
            "family": "segnet_aux_kl",
            "weight": 0.002,
            "temperature": 2.0,
            "promotion_eligible": True,
        }
    )

    with pytest.raises(FrozenInstanceError):
        policy.weight = 0.01  # type: ignore[misc]

    provenance = distillation_policy_to_provenance(policy)
    assert provenance["family"] == "segnet_aux_kl"
    assert provenance["scope"] == "segnet_aux"
    assert provenance["controller"] == {
        "enabled": False,
        "kind": None,
        "parameters": {},
        "state": {},
    }


def test_preflight_policy_schema_rejects_unfenced_jbl_profile() -> None:
    from tac.preflight import PreflightError, check_distillation_policy_schema_clean

    with pytest.raises(PreflightError, match="distillation policy invalid"):
        check_distillation_policy_schema_clean(
            profiles={
                "bad_jbl": {
                    "loss_mode": "jbl",
                    "kl_distill_scope": "segnet_aux",
                    "kl_distill_weight": 0.002,
                    "kl_distill_temperature": 2.0,
                    "eval_roundtrip": True,
                    "promotion_eligible": True,
                }
            },
            strict=True,
            verbose=False,
        )


def test_preflight_policy_schema_accepts_current_forensic_jbl_contract() -> None:
    from tac.preflight import check_distillation_policy_schema_clean

    violations = check_distillation_policy_schema_clean(
        profiles={
            "forensic_jbl": {
                "loss_mode": "jbl",
                "kl_distill_scope": "segnet_aux",
                "kl_distill_weight": 0.002,
                "kl_distill_temperature": 2.0,
                "eval_roundtrip": True,
                "promotion_eligible": False,
                "forensic_reason": "requires exact CUDA non-collapse evidence",
            }
        },
        strict=True,
        verbose=False,
    )

    assert violations == []
