# SPDX-License-Identifier: MIT
"""Dedicated tests for the PR110-OPT-11 L1 PROMOTION trainer.

Per CLAUDE.md "NO FAKE IMPLEMENTATIONS" Slot EEE 5 forbidden classes: every
test verifies ACTUAL L1 behavior, NOT shape constants. Per Catalog #213 the L1
path consumes real ``upstream/videos/0.mkv`` frames via the canonical
``tac.substrates._shared.trainer_skeleton.decode_real_pairs`` helper; tests that
touch real-frame decode are skipped when the contest video is absent
(clean-clone hermeticity).

The substrate's ACTUAL API (verified against
``src/tac/substrates/pr110_opt11_multi_mode_per_pair_composition/substrate.py``):
- ``CANONICAL_MODE_MENU`` is a tuple of ``(family, mode_id)`` tuples.
- ``_apply_canonical_perturbation(frame, mode_idx: int)`` takes a GLOBAL index.
- ``_compose_two_modes_on_frame(frame, mode_a_idx: int, mode_b_idx: int)``.
- ``_mode_indices_in_family(family)`` returns the global indices for a family.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

# test file path: src/tac/substrates/pr110_opt11_.../tests/test_l1_trainer.py
# parents[0]=tests [1]=pr110_opt11_... [2]=substrates [3]=tac [4]=src [5]=<repo>.
REPO_ROOT = Path(__file__).resolve().parents[5]
_TRAINER_PATH = (
    REPO_ROOT
    / "experiments"
    / "train_substrate_pr110_opt11_multi_mode_per_pair_composition.py"
)


def _load_trainer_module():
    """Load the trainer module by path (it lives under experiments/, not a package)."""
    if str(REPO_ROOT / "src") not in sys.path:
        sys.path.insert(0, str(REPO_ROOT / "src"))
    spec = importlib.util.spec_from_file_location(
        "_pr110_opt11_l1_trainer_under_test", _TRAINER_PATH
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _contest_video_present() -> bool:
    return (REPO_ROOT / "upstream" / "videos" / "0.mkv").is_file()


requires_video = pytest.mark.skipif(
    not _contest_video_present(),
    reason="contest video upstream/videos/0.mkv absent (clean-clone hermeticity)",
)


# --- trainer module surface ---


def test_trainer_module_loads():
    mod = _load_trainer_module()
    assert hasattr(mod, "_l1_main")
    assert hasattr(mod, "_real_frame_base_pairs")
    assert hasattr(mod, "_phase_c_validate")
    assert hasattr(mod, "main")


def test_phase_c_required_axes_canonical_seven():
    mod = _load_trainer_module()
    assert len(mod.PHASE_C_REQUIRED_AXES) == 7
    assert mod.PHASE_C_PASS_THRESHOLD == 7
    assert "real_frame_consumption_catalog_213" in mod.PHASE_C_REQUIRED_AXES
    assert (
        "multi_mode_composition_applied_on_real_frames"
        in mod.PHASE_C_REQUIRED_AXES
    )
    assert (
        "composition_distinct_from_both_single_modes" in mod.PHASE_C_REQUIRED_AXES
    )


def test_real_frame_base_pairs_uses_canonical_catalog_213_helper():
    """The L1 real-frame loader MUST route through the canonical
    decode_real_pairs helper (Catalog #213), NOT a synthetic
    rng.integers / make_synthetic path (Slot EEE Class 1 + Class 3)."""
    src = _TRAINER_PATH.read_text()
    assert "decode_real_pairs" in src
    # NO synthetic-fixture CALL in the L1 real-frame path. The forbidden-pattern
    # NAME may appear in docstrings (citing the CLAUDE.md rule); what matters is
    # there is no actual `make_synthetic_pair_batch(` invocation.
    assert "make_synthetic_pair_batch(" not in src


# --- _phase_c_validate behavior (not constants) ---


def test_phase_c_validate_all_green_is_seven():
    mod = _load_trainer_module()
    verdict = mod._phase_c_validate(
        real_frame_consumed=True,
        composition_applied=True,
        composition_distinct=True,
        archive_bytes=89,
        helpers_invoked=True,
        tier_a_present=True,
        provenance_ok=True,
    )
    assert verdict["n_pass"] == 7
    assert verdict["phase_c_verdict"] == "GREEN"


def test_phase_c_validate_missing_real_frame_is_red():
    mod = _load_trainer_module()
    verdict = mod._phase_c_validate(
        real_frame_consumed=False,  # the discriminating axis
        composition_applied=True,
        composition_distinct=True,
        archive_bytes=89,
        helpers_invoked=True,
        tier_a_present=True,
        provenance_ok=True,
    )
    assert verdict["n_pass"] == 6
    assert verdict["phase_c_verdict"] == "RED"
    assert verdict["axes"]["real_frame_consumption_catalog_213"] is False


def test_phase_c_validate_zero_archive_bytes_is_red():
    mod = _load_trainer_module()
    verdict = mod._phase_c_validate(
        real_frame_consumed=True,
        composition_applied=True,
        composition_distinct=True,
        archive_bytes=0,  # byte-stable archive axis fails
        helpers_invoked=True,
        tier_a_present=True,
        provenance_ok=True,
    )
    assert verdict["axes"]["archive_emitted_byte_stable"] is False
    assert verdict["phase_c_verdict"] == "RED"


def test_phase_c_validate_non_distinct_composition_is_red():
    mod = _load_trainer_module()
    verdict = mod._phase_c_validate(
        real_frame_consumed=True,
        composition_applied=True,
        composition_distinct=False,  # the multi-mode distinguishing axis fails
        archive_bytes=89,
        helpers_invoked=True,
        tier_a_present=True,
        provenance_ok=True,
    )
    assert verdict["axes"]["composition_distinct_from_both_single_modes"] is False
    assert verdict["phase_c_verdict"] == "RED"


# --- multi-mode distinctness invariant (structural, against real API) ---


def test_canonical_mode_menu_families_have_no_identity_in_pair_families():
    """The distinctness invariant is HARD-EARNED: the canonical
    CANONICAL_ORTHOGONAL_FAMILY_PAIRS never include the 'identity' family, so
    composition is ALWAYS distinct from both single modes on any non-constant
    frame."""
    from tac.substrates.pr110_opt11_multi_mode_per_pair_composition.substrate import (
        CANONICAL_ORTHOGONAL_FAMILY_PAIRS,
    )

    for fp in CANONICAL_ORTHOGONAL_FAMILY_PAIRS:
        assert "identity" not in fp[0]
        assert "identity" not in fp[1]


def test_composition_distinct_from_both_single_modes_across_all_family_pairs():
    """ACTUAL composition: across EVERY family pair x EVERY mode combo, the
    composed frame differs from BOTH single-mode outputs (NO FAKE Class 1+2)."""
    from tac.substrates.pr110_opt11_multi_mode_per_pair_composition.substrate import (
        CANONICAL_ORTHOGONAL_FAMILY_PAIRS,
        _apply_canonical_perturbation,
        _compose_two_modes_on_frame,
        _mode_indices_in_family,
    )

    rng = np.random.default_rng(7)
    frame = rng.integers(0, 256, size=(48, 64, 3), dtype=np.uint8)
    for fam_a, fam_b in CANONICAL_ORTHOGONAL_FAMILY_PAIRS:
        a_modes = _mode_indices_in_family(fam_a)
        b_modes = _mode_indices_in_family(fam_b)
        assert a_modes and b_modes
        for ma in a_modes:
            for mb in b_modes:
                out_a = _apply_canonical_perturbation(frame, ma)
                out_b = _apply_canonical_perturbation(frame, mb)
                out_ab = _compose_two_modes_on_frame(frame, ma, mb)
                assert not np.array_equal(out_ab, out_a), (
                    f"composed == single_a for ({fam_a},{fam_b}) modes ({ma},{mb})"
                )
                assert not np.array_equal(out_ab, out_b), (
                    f"composed == single_b for ({fam_a},{fam_b}) modes ({ma},{mb})"
                )


def test_apply_canonical_perturbation_does_not_mutate_input_frame():
    from tac.substrates.pr110_opt11_multi_mode_per_pair_composition.substrate import (
        _compose_two_modes_on_frame,
        _mode_indices_in_family,
    )

    rng = np.random.default_rng(11)
    frame = rng.integers(0, 256, size=(32, 48, 3), dtype=np.uint8)
    frame_copy = frame.copy()
    ma = _mode_indices_in_family("frame0_luma_bias")[0]
    mb = _mode_indices_in_family("frame0_rgb_bias")[0]
    _compose_two_modes_on_frame(frame, ma, mb)
    assert np.array_equal(frame, frame_copy), "composition mutated input frame"


# --- real-frame consumption per Catalog #213 ---


@requires_video
def test_real_frame_base_pairs_decodes_real_contest_frames():
    mod = _load_trainer_module()
    pairs = mod._real_frame_base_pairs(4, downscale=4)
    assert pairs.shape[0] == 4
    assert pairs.ndim == 5  # (N, 2, H, W, 3)
    assert pairs.shape[1] == 2
    assert pairs.shape[-1] == 3
    assert pairs.dtype == np.uint8
    # real contest frames are NOT synthetic uniform ~127.5
    mean = float(pairs[0, 0].mean())
    assert 0 < mean < 255
    assert abs(mean - 127.5) > 5.0, (
        f"frame mean {mean} suspiciously close to synthetic uniform 127.5"
    )


@requires_video
def test_real_frame_downscale_reduces_spatial_dims():
    mod = _load_trainer_module()
    pairs_full = mod._real_frame_base_pairs(2, downscale=1)
    pairs_ds = mod._real_frame_base_pairs(2, downscale=4)
    assert pairs_ds.shape[2] < pairs_full.shape[2]
    assert pairs_ds.shape[3] < pairs_full.shape[3]


@requires_video
def test_l1_main_phase_c_green_on_real_frames(tmp_path):
    mod = _load_trainer_module()

    args = argparse.Namespace(
        output_dir=tmp_path / "l1_out",
        n_pairs=4,
        modes_per_pair=2,
        selector_bits_per_mode=4,
        family_pair_index=1,
        downscale=4,
        rng_seed=42,
        enable_autocast_fp16=False,
        enable_torch_compile=False,
        no_grad_eval=True,
        smoke=True,
    )
    rc = mod._l1_main(args)
    assert rc == 0  # Phase C GREEN

    summary_path = tmp_path / "l1_out" / "pr110_opt11_l1_phase_c_summary.json"
    assert summary_path.is_file()
    summary = json.loads(summary_path.read_text())
    assert summary["phase_c"]["phase_c_verdict"] == "GREEN"
    assert summary["phase_c"]["n_pass"] == 7
    # NO FAKE: real-frame composition evidence is ACTUAL behavior
    ev = summary["l1_real_frame_composition_evidence"]
    assert ev["n_composed_on_real_frames"] == 4
    assert ev["n_distinct_from_single_mode_a"] == 4
    assert ev["n_distinct_from_single_mode_b"] == 4
    assert ev["composition_distinct_from_both"] is True
    assert ev["mean_abs_delta_composed_vs_real_base"] > 0.0
    # Catalog #213: real frame consumed (NOT synthetic)
    assert abs(summary["real_frame_0_mean"] - 127.5) > 5.0


@requires_video
def test_l1_main_emits_non_promotable_provenance(tmp_path):
    mod = _load_trainer_module()

    args = argparse.Namespace(
        output_dir=tmp_path / "l1_out2",
        n_pairs=4,
        modes_per_pair=2,
        selector_bits_per_mode=4,
        family_pair_index=1,
        downscale=4,
        rng_seed=42,
        enable_autocast_fp16=False,
        enable_torch_compile=False,
        no_grad_eval=True,
        smoke=True,
    )
    mod._l1_main(args)
    summary = json.loads(
        (tmp_path / "l1_out2" / "pr110_opt11_l1_phase_c_summary.json").read_text()
    )
    # Catalog #192 / #323 / #341: non-promotable by construction
    assert summary["score_claim"] is False
    assert summary["promotion_eligible"] is False
    assert summary["ready_for_exact_eval_dispatch"] is False
    assert summary["evidence_grade"] == "macOS-CPU-advisory"
    prov = summary["canonical_provenance"]
    assert prov["score_claim_valid"] is False
    assert prov["promotion_eligible"] is False
    # canonical Provenance per Catalog #323 carries measurement_axis +
    # evidence_grade (NOT axis_tag — that is the substrate Result object's field).
    assert prov["measurement_axis"] == "[predicted]"
    assert prov["evidence_grade"] == "predicted"
    # Catalog #324: post-training Tier-C density pending
    assert summary["predicted_band_validation_status"] == "pending_post_training"
    assert any(
        "post_training_tier_c" in b for b in summary["result_review_blockers"]
    )


# --- dispatcher mode resolution per Catalog #326 ---


def test_main_dispatcher_routes_l1_mode(monkeypatch, tmp_path):
    mod = _load_trainer_module()
    monkeypatch.setenv("PR110_OPT11_TRAINER_MODE", "l1")
    called = {}

    def _fake_l1(args):
        called["l1"] = True
        return 0

    monkeypatch.setattr(mod, "_l1_main", _fake_l1)
    rc = mod.main(["--output-dir", str(tmp_path / "x"), "--n-pairs", "2"])
    assert rc == 0
    assert called.get("l1") is True


def test_main_dispatcher_default_routes_smoke_not_l1(monkeypatch, tmp_path):
    mod = _load_trainer_module()
    monkeypatch.delenv("PR110_OPT11_TRAINER_MODE", raising=False)
    monkeypatch.setenv("SMOKE_ONLY", "1")
    called = {}

    def _fake_smoke(args):
        called["smoke"] = True
        return 0

    def _fake_l1(args):
        called["l1"] = True
        return 0

    monkeypatch.setattr(mod, "_smoke_main", _fake_smoke)
    monkeypatch.setattr(mod, "_l1_main", _fake_l1)
    rc = mod.main(["--output-dir", str(tmp_path / "y"), "--n-pairs", "2"])
    assert rc == 0
    # L0 SCAFFOLD evidence surface preserved: default is smoke, NOT l1
    assert called.get("smoke") is True
    assert "l1" not in called


def test_main_dispatcher_full_mode_raises_not_implemented(monkeypatch, tmp_path):
    mod = _load_trainer_module()
    monkeypatch.setenv("PR110_OPT11_TRAINER_MODE", "full")
    with pytest.raises(NotImplementedError):
        mod.main(["--output-dir", str(tmp_path / "z"), "--n-pairs", "2"])


def test_full_main_still_council_gated():
    mod = _load_trainer_module()
    args = argparse.Namespace(output_dir=Path("/tmp/x"))
    with pytest.raises(NotImplementedError, match="council-gated"):
        mod._full_main(args)
