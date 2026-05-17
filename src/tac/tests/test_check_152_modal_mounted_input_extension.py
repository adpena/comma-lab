# SPDX-License-Identifier: MIT
"""Tests for the WAVE-1 APPARATUS HARDENING 2026-05-16 extension to
Catalog #152 (``check_operator_wrapper_validates_required_input_files_pre_dispatch``).

The extension adds modal-mounted-input discipline: when a Modal-dispatch
wrapper has a ``required_input_file=True`` flag whose ``default_path`` falls
under the Modal-IGNORED ``experiments/results/**`` subtree (per
``tac.deploy.modal.mount_manifest.DEFAULT_RESULTS_IGNORE``), the trainer
MUST declare the path in its ``TIER_1_EXTRA_MOUNT_PATHS`` /
``MODAL_EXTRA_MOUNT_PATHS`` tuple OR the wrapper MUST carry a same-line
``# REQUIRED_INPUT_MODAL_STAGED_OK:<rationale>`` waiver.

Bug-class anchor: STC v2 smoke ``fc-01KRSB76H04HM4958V2HX2JZZ4`` rc=25
(2026-05-16) on Modal T4 because the Lane A anchor archive at
``experiments/results/lane_a_landed/archive_lane_a.zip`` lives under the
Modal-IGNORED ``results/**`` subtree; the file existed on the operator
workstation so local validation passed but the Modal worker never received
the file and crashed at the lane driver's ``exit 25`` defense-in-depth.

Lane: ``lane_wave_1_catalog_305_strict_flip_plus_152_anchor_extension_20260516``.
Memory: ``feedback_wave_1_catalog_305_strict_flip_plus_152_anchor_extension_landed_20260516.md``.
Sister of Catalog #153 (canonical Modal mount builder) + Catalog #166
(Modal HEAD-parity ledger) + Catalog #201 (sentinel files under Modal
mount set) + Catalog #244 (Modal NVML env block).
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from tac.preflight import (
    PreflightError,
    _check_152_collect_modal_staged_waivers,
    _check_152_default_path_is_modal_ignored,
    _check_152_driver_has_defensive_path_resolution,
    _check_152_driver_has_path_waiver,
    _check_152_driver_references_required_input,
    _check_152_trainer_declares_extra_mount_path,
    _check_152_wrapper_dispatches_to_modal,
    check_operator_wrapper_validates_required_input_files_pre_dispatch,
)

# ----------------------------------------------------------------------------
# Helper unit tests
# ----------------------------------------------------------------------------


def test_default_path_modal_ignored_positive() -> None:
    """experiments/results/** paths are Modal-IGNORED."""
    assert _check_152_default_path_is_modal_ignored(
        "experiments/results/lane_a_landed/archive_lane_a.zip"
    )
    assert _check_152_default_path_is_modal_ignored(
        "experiments/results/track4_sg_a1_t178000_20260509/submission_dir/archive.zip"
    )
    # Leading ./ normalization
    assert _check_152_default_path_is_modal_ignored(
        "./experiments/results/lane_a_landed/foo.bin"
    )


def test_default_path_modal_ignored_negative() -> None:
    """Paths outside experiments/results/** are NOT Modal-IGNORED."""
    assert not _check_152_default_path_is_modal_ignored("upstream/videos/0.mkv")
    assert not _check_152_default_path_is_modal_ignored("src/tac/foo.py")
    assert not _check_152_default_path_is_modal_ignored("tools/foo.py")
    # Empty / non-string-like
    assert not _check_152_default_path_is_modal_ignored("")


def test_wrapper_dispatches_to_modal_positive() -> None:
    """Modal-specific dispatch tokens detected."""
    assert _check_152_wrapper_dispatches_to_modal("python -m modal run app.py")
    assert _check_152_wrapper_dispatches_to_modal(
        "from experiments.modal_train_lane import dispatch"
    )
    assert _check_152_wrapper_dispatches_to_modal("modal.run(app, detach=True)")


def test_wrapper_dispatches_to_modal_negative() -> None:
    """Non-Modal dispatch tokens NOT flagged as Modal dispatch."""
    assert not _check_152_wrapper_dispatches_to_modal("vastai create instance")
    assert not _check_152_wrapper_dispatches_to_modal("lightning run app")
    assert not _check_152_wrapper_dispatches_to_modal("kaggle kernels push")
    assert not _check_152_wrapper_dispatches_to_modal("# no dispatch here")


def test_modal_staged_waiver_rationale_accepted(tmp_path: Path) -> None:
    """Non-placeholder waiver rationale is accepted."""
    text = (
        "modal run experiments/foo.py  "
        "# REQUIRED_INPUT_MODAL_STAGED_OK:staged-via-modal-volume-from-prior-dispatch"
    )
    waivers = _check_152_collect_modal_staged_waivers(text)
    assert len(waivers) == 1
    assert "staged-via-modal-volume" in next(iter(waivers))


def test_modal_staged_waiver_placeholder_rejected() -> None:
    """Placeholder rationale rejected so docstring example cannot self-waive."""
    assert _check_152_collect_modal_staged_waivers(
        "modal run X  # REQUIRED_INPUT_MODAL_STAGED_OK:<rationale>"
    ) == set()
    assert _check_152_collect_modal_staged_waivers(
        "modal run X  # REQUIRED_INPUT_MODAL_STAGED_OK:<reason>"
    ) == set()
    assert _check_152_collect_modal_staged_waivers(
        "modal run X  # REQUIRED_INPUT_MODAL_STAGED_OK:"
    ) == set()


def test_trainer_declares_extra_mount_path_string_literal(tmp_path: Path) -> None:
    """Trainer with direct string-literal entry in TIER_1_EXTRA_MOUNT_PATHS detected."""
    trainer = tmp_path / "train_substrate_foo.py"
    trainer.write_text(textwrap.dedent("""
        from pathlib import Path

        TIER_1_EXTRA_MOUNT_PATHS = (
            "experiments/results/lane_foo/archive.zip",
        )
    """).strip())
    assert _check_152_trainer_declares_extra_mount_path(
        trainer, "experiments/results/lane_foo/archive.zip"
    )


def test_trainer_declares_extra_mount_path_path_segment_expression(tmp_path: Path) -> None:
    """Canonical Path-segment expression form detected.

    The STC v2 fix uses:
        TIER_1_EXTRA_MOUNT_PATHS = (str(DEFAULT_ANCHOR_ARCHIVE.relative_to(REPO_ROOT)),)
    where DEFAULT_ANCHOR_ARCHIVE is built from REPO_ROOT / 'experiments' / ...
    """
    trainer = tmp_path / "train_substrate_stc_v2_like.py"
    trainer.write_text(textwrap.dedent("""
        from pathlib import Path

        REPO_ROOT = Path(__file__).resolve().parent.parent
        DEFAULT_ANCHOR_ARCHIVE = (
            REPO_ROOT / "experiments" / "results" / "lane_a_landed" / "archive_lane_a.zip"
        )

        TIER_1_EXTRA_MOUNT_PATHS = (
            str(DEFAULT_ANCHOR_ARCHIVE.relative_to(REPO_ROOT)),
        )
    """).strip())
    assert _check_152_trainer_declares_extra_mount_path(
        trainer, "experiments/results/lane_a_landed/archive_lane_a.zip"
    )


def test_trainer_declares_extra_mount_path_modal_extra_alias(tmp_path: Path) -> None:
    """MODAL_EXTRA_MOUNT_PATHS sibling alias also accepted."""
    trainer = tmp_path / "train_substrate_bar.py"
    trainer.write_text(textwrap.dedent("""
        MODAL_EXTRA_MOUNT_PATHS = (
            "experiments/results/lane_bar/foo.bin",
        )
    """).strip())
    assert _check_152_trainer_declares_extra_mount_path(
        trainer, "experiments/results/lane_bar/foo.bin"
    )


def test_trainer_does_not_declare_extra_mount_path(tmp_path: Path) -> None:
    """Trainer without any extra-mount declaration NOT accepted."""
    trainer = tmp_path / "train_substrate_baz.py"
    trainer.write_text(textwrap.dedent("""
        from pathlib import Path

        DEFAULT_ARCHIVE = Path("experiments/results/lane_baz/archive.zip")

        # No TIER_1_EXTRA_MOUNT_PATHS declared!
    """).strip())
    assert not _check_152_trainer_declares_extra_mount_path(
        trainer, "experiments/results/lane_baz/archive.zip"
    )


def test_trainer_declares_extra_mount_list_form(tmp_path: Path) -> None:
    """List form (vs tuple) also accepted."""
    trainer = tmp_path / "train_substrate_listform.py"
    trainer.write_text(textwrap.dedent("""
        TIER_1_EXTRA_MOUNT_PATHS = [
            "experiments/results/lane_l/archive.zip",
        ]
    """).strip())
    assert _check_152_trainer_declares_extra_mount_path(
        trainer, "experiments/results/lane_l/archive.zip"
    )


def test_trainer_declares_extra_mount_annotated_assignment(tmp_path: Path) -> None:
    """Annotated assignment form ``TIER_1_EXTRA_MOUNT_PATHS: tuple = (...)`` accepted."""
    trainer = tmp_path / "train_substrate_annot.py"
    trainer.write_text(textwrap.dedent("""
        TIER_1_EXTRA_MOUNT_PATHS: tuple[str, ...] = (
            "experiments/results/lane_a/archive.zip",
        )
    """).strip())
    assert _check_152_trainer_declares_extra_mount_path(
        trainer, "experiments/results/lane_a/archive.zip"
    )


# ----------------------------------------------------------------------------
# End-to-end recipe-schema sub-scan tests
# ----------------------------------------------------------------------------


def test_recipe_schema_subscan_violation_when_trainer_lacks_extra_mount(
    tmp_path: Path,
) -> None:
    """Modal recipe + trainer without extra-mount = violation."""
    # Set up repo structure
    (tmp_path / ".omx" / "operator_authorize_recipes").mkdir(parents=True)
    (tmp_path / "experiments").mkdir()
    trainer = tmp_path / "experiments" / "train_substrate_foo.py"
    trainer.write_text(textwrap.dedent("""
        from pathlib import Path

        # No TIER_1_EXTRA_MOUNT_PATHS declared
        DEFAULT_ARCHIVE = Path("experiments/results/lane_foo/archive.zip")
    """).strip())
    recipe = tmp_path / ".omx" / "operator_authorize_recipes" / "substrate_foo_modal_t4_dispatch.yaml"
    recipe.write_text(textwrap.dedent("""
        schema_version: 1
        name: substrate_foo_modal_t4_dispatch
        platform: modal
        required_input_files:
          - flag: --anchor-archive
            default_path: experiments/results/lane_foo/archive.zip
        required_input_files_trainer: experiments/train_substrate_foo.py
    """).strip())
    violations = check_operator_wrapper_validates_required_input_files_pre_dispatch(
        repo_root=tmp_path, strict=False, verbose=False
    )
    modal_v = [v for v in violations if "Modal recipe" in v]
    assert len(modal_v) == 1, modal_v
    assert "--anchor-archive" in modal_v[0]
    assert "experiments/results/lane_foo/archive.zip" in modal_v[0]
    assert "TIER_1_EXTRA_MOUNT_PATHS" in modal_v[0]
    assert "STC v2 smoke" in modal_v[0]  # Bug-class anchor cited


def test_recipe_schema_subscan_passes_when_trainer_declares_extra_mount(
    tmp_path: Path,
) -> None:
    """Modal recipe + trainer WITH TIER_1_EXTRA_MOUNT_PATHS = pass."""
    (tmp_path / ".omx" / "operator_authorize_recipes").mkdir(parents=True)
    (tmp_path / "experiments").mkdir()
    trainer = tmp_path / "experiments" / "train_substrate_foo.py"
    trainer.write_text(textwrap.dedent("""
        TIER_1_EXTRA_MOUNT_PATHS = (
            "experiments/results/lane_foo/archive.zip",
        )
    """).strip())
    recipe = tmp_path / ".omx" / "operator_authorize_recipes" / "substrate_foo_modal_t4_dispatch.yaml"
    recipe.write_text(textwrap.dedent("""
        schema_version: 1
        name: substrate_foo_modal_t4_dispatch
        platform: modal
        required_input_files:
          - flag: --anchor-archive
            default_path: experiments/results/lane_foo/archive.zip
        required_input_files_trainer: experiments/train_substrate_foo.py
    """).strip())
    violations = check_operator_wrapper_validates_required_input_files_pre_dispatch(
        repo_root=tmp_path, strict=False, verbose=False
    )
    modal_v = [v for v in violations if "Modal recipe" in v]
    assert len(modal_v) == 0, modal_v


def test_recipe_schema_subscan_skips_non_modal_platform(tmp_path: Path) -> None:
    """Vast.ai / Lightning recipes are out of scope."""
    (tmp_path / ".omx" / "operator_authorize_recipes").mkdir(parents=True)
    (tmp_path / "experiments").mkdir()
    trainer = tmp_path / "experiments" / "train_substrate_foo.py"
    trainer.write_text("# no TIER_1_EXTRA_MOUNT_PATHS")
    recipe = tmp_path / ".omx" / "operator_authorize_recipes" / "substrate_foo_vastai_dispatch.yaml"
    recipe.write_text(textwrap.dedent("""
        platform: vastai
        required_input_files:
          - flag: --anchor-archive
            default_path: experiments/results/lane_foo/archive.zip
        required_input_files_trainer: experiments/train_substrate_foo.py
    """).strip())
    violations = check_operator_wrapper_validates_required_input_files_pre_dispatch(
        repo_root=tmp_path, strict=False, verbose=False
    )
    modal_v = [v for v in violations if "Modal recipe" in v]
    assert len(modal_v) == 0, modal_v


def test_recipe_schema_subscan_skips_paths_not_under_results(tmp_path: Path) -> None:
    """upstream/videos/0.mkv style paths are NOT Modal-IGNORED — skip."""
    (tmp_path / ".omx" / "operator_authorize_recipes").mkdir(parents=True)
    (tmp_path / "experiments").mkdir()
    trainer = tmp_path / "experiments" / "train_substrate_foo.py"
    trainer.write_text("# no extra mount")
    recipe = tmp_path / ".omx" / "operator_authorize_recipes" / "substrate_foo_modal_t4_dispatch.yaml"
    recipe.write_text(textwrap.dedent("""
        platform: modal
        required_input_files:
          - flag: --video-path
            default_path: upstream/videos/0.mkv
        required_input_files_trainer: experiments/train_substrate_foo.py
    """).strip())
    violations = check_operator_wrapper_validates_required_input_files_pre_dispatch(
        repo_root=tmp_path, strict=False, verbose=False
    )
    modal_v = [v for v in violations if "Modal recipe" in v]
    assert len(modal_v) == 0, modal_v


def test_recipe_schema_subscan_waiver_accepts(tmp_path: Path) -> None:
    """Same-line `# REQUIRED_INPUT_MODAL_STAGED_OK:<rationale>` waiver accepted."""
    (tmp_path / ".omx" / "operator_authorize_recipes").mkdir(parents=True)
    (tmp_path / "experiments").mkdir()
    trainer = tmp_path / "experiments" / "train_substrate_foo.py"
    trainer.write_text("# no TIER_1_EXTRA_MOUNT_PATHS")
    recipe = tmp_path / ".omx" / "operator_authorize_recipes" / "substrate_foo_modal_t4_dispatch.yaml"
    recipe.write_text(textwrap.dedent("""
        platform: modal
        # REQUIRED_INPUT_MODAL_STAGED_OK:operator-staged-via-modal-volume-from-prior-dispatch-run
        required_input_files:
          - flag: --anchor-archive
            default_path: experiments/results/lane_foo/archive.zip
        required_input_files_trainer: experiments/train_substrate_foo.py
    """).strip())
    violations = check_operator_wrapper_validates_required_input_files_pre_dispatch(
        repo_root=tmp_path, strict=False, verbose=False
    )
    modal_v = [v for v in violations if "Modal recipe" in v]
    assert len(modal_v) == 0, modal_v


def test_recipe_schema_subscan_placeholder_waiver_rejected(tmp_path: Path) -> None:
    """Placeholder `<rationale>` literal does NOT silently waive."""
    (tmp_path / ".omx" / "operator_authorize_recipes").mkdir(parents=True)
    (tmp_path / "experiments").mkdir()
    trainer = tmp_path / "experiments" / "train_substrate_foo.py"
    trainer.write_text("# no TIER_1_EXTRA_MOUNT_PATHS")
    recipe = tmp_path / ".omx" / "operator_authorize_recipes" / "substrate_foo_modal_t4_dispatch.yaml"
    recipe.write_text(textwrap.dedent("""
        platform: modal
        # REQUIRED_INPUT_MODAL_STAGED_OK:<rationale>
        required_input_files:
          - flag: --anchor-archive
            default_path: experiments/results/lane_foo/archive.zip
        required_input_files_trainer: experiments/train_substrate_foo.py
    """).strip())
    violations = check_operator_wrapper_validates_required_input_files_pre_dispatch(
        repo_root=tmp_path, strict=False, verbose=False
    )
    modal_v = [v for v in violations if "Modal recipe" in v]
    assert len(modal_v) == 1, modal_v


def test_recipe_schema_subscan_fallback_to_modal_cost_band_trainer(
    tmp_path: Path,
) -> None:
    """When `required_input_files_trainer` is missing, fall back to
    `modal.cost_band_trainer` to resolve trainer path."""
    (tmp_path / ".omx" / "operator_authorize_recipes").mkdir(parents=True)
    (tmp_path / "experiments").mkdir()
    trainer = tmp_path / "experiments" / "train_substrate_foo.py"
    trainer.write_text(textwrap.dedent("""
        TIER_1_EXTRA_MOUNT_PATHS = (
            "experiments/results/lane_foo/archive.zip",
        )
    """).strip())
    recipe = tmp_path / ".omx" / "operator_authorize_recipes" / "substrate_foo_modal_t4_dispatch.yaml"
    # No required_input_files_trainer; use modal.cost_band_trainer
    recipe.write_text(textwrap.dedent("""
        platform: modal
        required_input_files:
          - flag: --anchor-archive
            default_path: experiments/results/lane_foo/archive.zip
        modal:
          cost_band_trainer: experiments/train_substrate_foo.py
    """).strip())
    violations = check_operator_wrapper_validates_required_input_files_pre_dispatch(
        repo_root=tmp_path, strict=False, verbose=False
    )
    modal_v = [v for v in violations if "Modal recipe" in v]
    assert len(modal_v) == 0, modal_v


def test_strict_mode_raises_on_modal_recipe_violation(tmp_path: Path) -> None:
    """strict=True raises PreflightError citing Catalog #152 + Modal-IGNORED."""
    (tmp_path / ".omx" / "operator_authorize_recipes").mkdir(parents=True)
    (tmp_path / "experiments").mkdir()
    trainer = tmp_path / "experiments" / "train_substrate_foo.py"
    trainer.write_text("# no TIER_1_EXTRA_MOUNT_PATHS")
    recipe = tmp_path / ".omx" / "operator_authorize_recipes" / "substrate_foo_modal_t4_dispatch.yaml"
    recipe.write_text(textwrap.dedent("""
        platform: modal
        required_input_files:
          - flag: --anchor-archive
            default_path: experiments/results/lane_foo/archive.zip
        required_input_files_trainer: experiments/train_substrate_foo.py
    """).strip())
    with pytest.raises(PreflightError) as exc:
        check_operator_wrapper_validates_required_input_files_pre_dispatch(
            repo_root=tmp_path, strict=True, verbose=False
        )
    assert "check_operator_wrapper_validates_required_input_files_pre_dispatch" in str(exc.value)


# ----------------------------------------------------------------------------
# Live-repo regression guards (the actual STC v2 + a1_plus_* fixes)
# ----------------------------------------------------------------------------


def test_live_repo_stc_v2_trainer_declares_extra_mount_path() -> None:
    """STC v2 trainer (the canonical fix anchor) declares the Lane A archive
    in its TIER_1_EXTRA_MOUNT_PATHS tuple."""
    trainer = Path("experiments/train_substrate_stc_v2.py")
    assert trainer.exists()
    assert _check_152_trainer_declares_extra_mount_path(
        trainer, "experiments/results/lane_a_landed/archive_lane_a.zip"
    )


def test_live_repo_a1_plus_wavelet_residual_trainer_declares_extra_mount_path() -> None:
    """a1_plus_wavelet_residual trainer (sister fix) declares the A1 archive
    in its TIER_1_EXTRA_MOUNT_PATHS tuple."""
    trainer = Path("experiments/train_substrate_a1_plus_wavelet_residual.py")
    assert trainer.exists()
    assert _check_152_trainer_declares_extra_mount_path(
        trainer,
        "experiments/results/track4_sg_a1_t178000_20260509/submission_dir/archive.zip",
    )


def test_live_repo_a1_plus_lapose_trainer_declares_extra_mount_path() -> None:
    """a1_plus_lapose trainer (sister fix) declares the A1 archive
    in its TIER_1_EXTRA_MOUNT_PATHS tuple."""
    trainer = Path("experiments/train_substrate_a1_plus_lapose.py")
    assert trainer.exists()
    assert _check_152_trainer_declares_extra_mount_path(
        trainer,
        "experiments/results/track4_sg_a1_t178000_20260509/submission_dir/archive.zip",
    )


def test_live_repo_wave_1_extension_zero_modal_recipe_violations() -> None:
    """Live-repo regression guard: WAVE-1 extension lands 0 Modal-recipe
    violations after STC v2 + a1_plus_wavelet_residual + a1_plus_lapose fixes."""
    violations = check_operator_wrapper_validates_required_input_files_pre_dispatch(
        strict=False, verbose=False
    )
    modal_v = [v for v in violations if "Modal recipe" in v]
    assert len(modal_v) == 0, (
        f"WAVE-1 extension should land 0 Modal-recipe violations; got "
        f"{len(modal_v)}: {modal_v[:3]}"
    )


def test_live_repo_wave_1_extension_zero_modal_ignored_wrapper_violations() -> None:
    """Live-repo regression guard: WAVE-1 wrapper-scan extension lands 0
    Modal-IGNORED wrapper violations."""
    violations = check_operator_wrapper_validates_required_input_files_pre_dispatch(
        strict=False, verbose=False
    )
    modal_v = [v for v in violations if "Modal-IGNORED" in v]
    assert len(modal_v) == 0, (
        f"WAVE-1 extension should land 0 Modal-IGNORED wrapper violations; "
        f"got {len(modal_v)}: {modal_v[:3]}"
    )


# ----------------------------------------------------------------------------
# Catalog #305 strict-flip regression guards
# ----------------------------------------------------------------------------


def test_catalog_305_strict_flip_live_count_zero() -> None:
    """Catalog #305 (Observability surface) live count = 0 post-WAVE-1
    backfill of 3 sister-landed design memos."""
    from tac.preflight import check_substrate_design_memo_has_observability_surface_section
    violations = check_substrate_design_memo_has_observability_surface_section(
        strict=False, verbose=False
    )
    assert len(violations) == 0, (
        f"Catalog #305 should be at 0 violations post-WAVE-1 backfill; "
        f"got {len(violations)}: {violations[:3]}"
    )


def test_catalog_305_strict_mode_does_not_raise() -> None:
    """Catalog #305 strict=True passes without raising (strict-flip evidence)."""
    from tac.preflight import check_substrate_design_memo_has_observability_surface_section
    # Should not raise
    result = check_substrate_design_memo_has_observability_surface_section(
        strict=True, verbose=False
    )
    assert result == []


def test_catalog_305_three_backfilled_memos_have_observability_surface_section() -> None:
    """The 3 WAVE-1-backfilled design memos contain the literal section header."""
    memo_paths = [
        ".omx/research/rudin_floor_interpretable_ml_substrate_asymptotic_pursuit_scoping_design_20260516.md",
        ".omx/research/tier_1_resurrection_4_pr101_compressai_balle_reformulated_full_stack_design_20260516.md",
        ".omx/research/tier_1_resurrection_5_pr106_lanes_05_06_reformulated_uniward_grayscale_lut_full_stack_design_20260516.md",
    ]
    for rel in memo_paths:
        p = Path(rel)
        assert p.exists(), f"memo missing: {rel}"
        text = p.read_text(encoding="utf-8", errors="replace").lower()
        assert "## observability surface" in text, (
            f"memo {rel} missing '## Observability surface' section header"
        )


# ----------------------------------------------------------------------------
# Sister META-meta gates clean post-edit
# ----------------------------------------------------------------------------


def test_meta_meta_gates_clean_post_wave_1_landing() -> None:
    """Catalog #118 / #159 / #176 / #185 (META-meta gates) all 0 post-WAVE-1."""
    from tac.preflight import (
        check_claude_md_catalog_no_duplicate_numbers,
        check_claude_md_catalog_text_matches_preflight_strict_value,
        check_strict_flipped_catalog_entries_have_live_count_zero,
        check_strict_preflight_callsites_have_claude_md_catalog_row,
    )
    assert (
        len(check_claude_md_catalog_no_duplicate_numbers(strict=False, verbose=False))
        == 0
    )
    assert (
        len(
            check_claude_md_catalog_text_matches_preflight_strict_value(
                strict=False, verbose=False
            )
        )
        == 0
    )
    assert (
        len(
            check_strict_preflight_callsites_have_claude_md_catalog_row(
                strict=False, verbose=False
            )
        )
        == 0
    )
    assert (
        len(
            check_strict_flipped_catalog_entries_have_live_count_zero(
                strict=False, verbose=False
            )
        )
        == 0
    )


# ----------------------------------------------------------------------------
# Wave 2 (2026-05-16) driver-path-expectation extension tests
# ----------------------------------------------------------------------------
# Bug-class anchor: STC v2 Modal T4 dispatch fc-01KRSVKF9VEESQY2FS33FF4WDM
# rc=25 at 1.56s (2026-05-17T02:17:51Z). Wave 1's trainer-side fix is INERT
# for generic Modal dispatchers because experiments/modal_train_lane.py
# passes trainer_module_path=None to build_training_image. The driver
# shell script must defensively resolve required-input file paths across
# multiple candidate locations.
# ----------------------------------------------------------------------------


def test_driver_has_defensive_path_resolution_canonical_helper_positive() -> None:
    """Canonical helper function name triggers defensive-resolution acceptance."""
    text = textwrap.dedent(
        """
        #!/bin/bash
        WORKSPACE="${WORKSPACE:-/workspace/pact}"
        resolve_required_input_modal_aware() {
            local env_var="$1"
            local rel_path="$2"
            echo "$WORKSPACE/$rel_path"
        }
        ANCHOR="$(resolve_required_input_modal_aware FOO experiments/results/x.zip)"
        """
    )
    assert _check_152_driver_has_defensive_path_resolution(text) is True


def test_driver_has_defensive_path_resolution_explicit_candidate_probe_positive() -> None:
    """Explicit /workspace/pact + /tmp/pact probe satisfies defensive-resolution."""
    text = textwrap.dedent(
        """
        #!/bin/bash
        for candidate in \
            "/workspace/pact/experiments/results/x.zip" \
            "/tmp/pact/experiments/results/x.zip"; do
            [ -f "$candidate" ] && ANCHOR="$candidate"
        done
        """
    )
    assert _check_152_driver_has_defensive_path_resolution(text) is True


def test_driver_has_defensive_path_resolution_modal_runtime_conditional_positive() -> None:
    """MODAL_RUNTIME-conditional branch satisfies defensive-resolution."""
    text = textwrap.dedent(
        """
        #!/bin/bash
        if [ "${MODAL_RUNTIME:-0}" = "1" ]; then
            ANCHOR="/tmp/pact/experiments/results/x.zip"
        else
            ANCHOR="$WORKSPACE/experiments/results/x.zip"
        fi
        """
    )
    assert _check_152_driver_has_defensive_path_resolution(text) is True


def test_driver_has_defensive_path_resolution_negative() -> None:
    """Plain $WORKSPACE/path usage without defensive idiom is NOT recognized."""
    text = textwrap.dedent(
        """
        #!/bin/bash
        WORKSPACE="${WORKSPACE:-/workspace/pact}"
        ANCHOR="$WORKSPACE/experiments/results/x.zip"
        if [ ! -f "$ANCHOR" ]; then
            exit 25
        fi
        """
    )
    assert _check_152_driver_has_defensive_path_resolution(text) is False


def test_driver_has_path_waiver_accepts_rationale() -> None:
    """Same-line waiver with substantive rationale is accepted."""
    text = "ANCHOR=...  # DRIVER_PATH_MODAL_AWARE_OK:operator pre-stages via Modal volume\n"
    rationales = _check_152_driver_has_path_waiver(text)
    assert "operator pre-stages via Modal volume" in rationales


def test_driver_has_path_waiver_rejects_placeholder() -> None:
    """Placeholder <rationale> + <reason> + empty literals are REJECTED."""
    for placeholder in ("<rationale>", "<reason>", ""):
        text = f"ANCHOR=...  # DRIVER_PATH_MODAL_AWARE_OK:{placeholder}\n"
        rationales = _check_152_driver_has_path_waiver(text)
        if placeholder == "":
            # Empty waiver text => regex captures empty string => rejected.
            assert rationales == set()
        else:
            assert placeholder not in rationales


def test_driver_has_path_waiver_accepts_sister_required_input_modal_staged_marker() -> None:
    """The Wave 1 REQUIRED_INPUT_MODAL_STAGED_OK marker is ALSO accepted."""
    text = (
        "ANCHOR=...  # REQUIRED_INPUT_MODAL_STAGED_OK:operator stages via volume\n"
    )
    rationales = _check_152_driver_has_path_waiver(text)
    assert "operator stages via volume" in rationales


def test_driver_references_required_input_full_path_positive() -> None:
    """Driver text containing the full relative path is recognized."""
    text = "ANCHOR=$WORKSPACE/experiments/results/lane_a_landed/archive_lane_a.zip"
    assert (
        _check_152_driver_references_required_input(
            text, "experiments/results/lane_a_landed/archive_lane_a.zip"
        )
        is True
    )


def test_driver_references_required_input_basename_with_context_positive() -> None:
    """Driver text with basename + context tokens is recognized."""
    text = """
        # Anchor archive from lane_a_landed
        ANCHOR="/some/path/experiments/results/lane_a_landed/archive_lane_a.zip"
    """
    assert (
        _check_152_driver_references_required_input(
            text, "experiments/results/lane_a_landed/archive_lane_a.zip"
        )
        is True
    )


def test_driver_references_required_input_negative_unrelated_file() -> None:
    """Driver text without any reference to the file does NOT trigger."""
    text = "echo hello\nWORKSPACE=/tmp/pact\n"
    assert (
        _check_152_driver_references_required_input(
            text, "experiments/results/lane_a_landed/archive_lane_a.zip"
        )
        is False
    )


def test_driver_references_required_input_empty_path() -> None:
    """Empty default_path does NOT match anything."""
    assert _check_152_driver_references_required_input("some text", "") is False


def test_live_repo_stc_v2_driver_has_defensive_resolution() -> None:
    """Live regression: scripts/remote_lane_substrate_stc_v2.sh carries
    the canonical Wave 2 defensive resolver after the driver fix lands."""
    repo_root = Path(__file__).resolve().parents[3]
    driver = repo_root / "scripts" / "remote_lane_substrate_stc_v2.sh"
    if not driver.is_file():
        pytest.skip("STC v2 driver not present in this checkout")
    text = driver.read_text(encoding="utf-8")
    assert _check_152_driver_has_defensive_path_resolution(text), (
        "STC v2 driver MUST carry canonical defensive path resolution per "
        "Wave 2 Catalog #152 extension (lane "
        "lane_stc_v2_driver_fix_catalog_152_driver_path_extension_20260516)"
    )


def test_live_repo_a1_plus_lapose_driver_has_defensive_resolution() -> None:
    """Live regression: scripts/remote_lane_substrate_a1_plus_lapose.sh carries
    the canonical Wave 2 defensive resolver."""
    repo_root = Path(__file__).resolve().parents[3]
    driver = repo_root / "scripts" / "remote_lane_substrate_a1_plus_lapose.sh"
    if not driver.is_file():
        pytest.skip("a1_plus_lapose driver not present in this checkout")
    text = driver.read_text(encoding="utf-8")
    assert _check_152_driver_has_defensive_path_resolution(text), (
        "a1_plus_lapose driver MUST carry canonical defensive path resolution "
        "per Wave 2 Catalog #152 extension"
    )


def test_live_repo_a1_plus_wavelet_residual_driver_has_defensive_resolution() -> None:
    """Live regression: scripts/remote_lane_substrate_a1_plus_wavelet_residual.sh
    carries the canonical Wave 2 defensive resolver."""
    repo_root = Path(__file__).resolve().parents[3]
    driver = (
        repo_root
        / "scripts"
        / "remote_lane_substrate_a1_plus_wavelet_residual.sh"
    )
    if not driver.is_file():
        pytest.skip("a1_plus_wavelet_residual driver not present in this checkout")
    text = driver.read_text(encoding="utf-8")
    assert _check_152_driver_has_defensive_path_resolution(text), (
        "a1_plus_wavelet_residual driver MUST carry canonical defensive "
        "path resolution per Wave 2 Catalog #152 extension"
    )


def test_live_repo_driver_path_expectation_extension_zero_violations() -> None:
    """End-to-end live regression: with Wave 2 driver fixes landed, the
    extended Catalog #152 gate has 0 driver-path-expectation violations."""
    violations = check_operator_wrapper_validates_required_input_files_pre_dispatch(
        strict=False, verbose=False
    )
    driver_violations = [v for v in violations if "Modal driver script" in v]
    assert len(driver_violations) == 0, (
        f"Expected 0 driver-path-expectation violations; got "
        f"{len(driver_violations)}: {driver_violations[:3]}"
    )


def test_synthetic_modal_recipe_with_undefensive_driver_flagged(
    tmp_path: Path,
) -> None:
    """End-to-end: synthetic Modal recipe + non-defensive driver triggers
    a driver-path-expectation violation."""
    repo = tmp_path / "repo"
    (repo / ".omx" / "operator_authorize_recipes").mkdir(parents=True)
    (repo / "scripts").mkdir()
    (repo / "experiments").mkdir()
    # Create a synthetic substrate driver that consumes a results/** file
    # without defensive resolution.
    driver = repo / "scripts" / "remote_lane_substrate_synthetic_undefensive.sh"
    driver.write_text(
        textwrap.dedent(
            """
            #!/bin/bash
            set -euo pipefail
            WORKSPACE="${WORKSPACE:-/workspace/pact}"
            ANCHOR="$WORKSPACE/experiments/results/lane_a_landed/archive_lane_a.zip"
            if [ ! -f "$ANCHOR" ]; then
                exit 25
            fi
            """
        )
    )
    # Synthetic recipe.
    (repo / ".omx" / "operator_authorize_recipes" / "synth_modal.yaml").write_text(
        textwrap.dedent(
            """
            platform: modal
            lane_script: scripts/remote_lane_substrate_synthetic_undefensive.sh
            required_input_files:
              - flag: --anchor
                default_path: experiments/results/lane_a_landed/archive_lane_a.zip
            """
        )
    )
    violations = check_operator_wrapper_validates_required_input_files_pre_dispatch(
        repo_root=repo, strict=False, verbose=False
    )
    driver_violations = [v for v in violations if "Modal driver script" in v]
    assert len(driver_violations) >= 1, (
        f"Expected >=1 driver violation for non-defensive driver; got "
        f"{len(driver_violations)}: {violations}"
    )
    assert any(
        "fc-01KRSVKF9VEESQY2FS33FF4WDM" in v for v in driver_violations
    ), "Violation message must cite the bug-class anchor (STC v2 fc-id)"


def test_synthetic_modal_recipe_with_defensive_driver_passes(tmp_path: Path) -> None:
    """End-to-end: synthetic Modal recipe + defensive driver passes."""
    repo = tmp_path / "repo"
    (repo / ".omx" / "operator_authorize_recipes").mkdir(parents=True)
    (repo / "scripts").mkdir()
    (repo / "experiments").mkdir()
    driver = repo / "scripts" / "remote_lane_substrate_synthetic_defensive.sh"
    driver.write_text(
        textwrap.dedent(
            """
            #!/bin/bash
            set -euo pipefail
            WORKSPACE="${WORKSPACE:-/workspace/pact}"
            resolve_required_input_modal_aware() {
                local env_var="$1"
                local rel_path="$2"
                echo "$WORKSPACE/$rel_path"
            }
            ANCHOR="$(resolve_required_input_modal_aware FOO experiments/results/lane_a_landed/archive_lane_a.zip)"
            """
        )
    )
    (repo / ".omx" / "operator_authorize_recipes" / "synth_modal.yaml").write_text(
        textwrap.dedent(
            """
            platform: modal
            lane_script: scripts/remote_lane_substrate_synthetic_defensive.sh
            required_input_files:
              - flag: --anchor
                default_path: experiments/results/lane_a_landed/archive_lane_a.zip
            """
        )
    )
    violations = check_operator_wrapper_validates_required_input_files_pre_dispatch(
        repo_root=repo, strict=False, verbose=False
    )
    driver_violations = [v for v in violations if "Modal driver script" in v]
    assert len(driver_violations) == 0, (
        f"Expected 0 driver violations for defensive driver; got "
        f"{len(driver_violations)}: {driver_violations}"
    )


def test_synthetic_modal_recipe_with_waivered_undefensive_driver_passes(
    tmp_path: Path,
) -> None:
    """End-to-end: undefensive driver with same-line waiver passes."""
    repo = tmp_path / "repo"
    (repo / ".omx" / "operator_authorize_recipes").mkdir(parents=True)
    (repo / "scripts").mkdir()
    driver = repo / "scripts" / "remote_lane_substrate_synthetic_waivered.sh"
    driver.write_text(
        textwrap.dedent(
            """
            #!/bin/bash
            set -euo pipefail
            WORKSPACE="${WORKSPACE:-/workspace/pact}"
            # DRIVER_PATH_MODAL_AWARE_OK:operator pre-stages anchor via Modal volume mechanism
            ANCHOR="$WORKSPACE/experiments/results/lane_a_landed/archive_lane_a.zip"
            """
        )
    )
    (repo / ".omx" / "operator_authorize_recipes" / "synth_modal.yaml").write_text(
        textwrap.dedent(
            """
            platform: modal
            lane_script: scripts/remote_lane_substrate_synthetic_waivered.sh
            required_input_files:
              - flag: --anchor
                default_path: experiments/results/lane_a_landed/archive_lane_a.zip
            """
        )
    )
    violations = check_operator_wrapper_validates_required_input_files_pre_dispatch(
        repo_root=repo, strict=False, verbose=False
    )
    driver_violations = [v for v in violations if "Modal driver script" in v]
    assert len(driver_violations) == 0, (
        f"Expected 0 driver violations after waiver; got: {driver_violations}"
    )


def test_synthetic_modal_recipe_with_placeholder_waiver_rejected(tmp_path: Path) -> None:
    """Placeholder waiver does NOT excuse the driver-path violation."""
    repo = tmp_path / "repo"
    (repo / ".omx" / "operator_authorize_recipes").mkdir(parents=True)
    (repo / "scripts").mkdir()
    driver = repo / "scripts" / "remote_lane_substrate_synthetic_placeholder.sh"
    driver.write_text(
        textwrap.dedent(
            """
            #!/bin/bash
            set -euo pipefail
            WORKSPACE="${WORKSPACE:-/workspace/pact}"
            # DRIVER_PATH_MODAL_AWARE_OK:<rationale>
            ANCHOR="$WORKSPACE/experiments/results/lane_a_landed/archive_lane_a.zip"
            """
        )
    )
    (repo / ".omx" / "operator_authorize_recipes" / "synth_modal.yaml").write_text(
        textwrap.dedent(
            """
            platform: modal
            lane_script: scripts/remote_lane_substrate_synthetic_placeholder.sh
            required_input_files:
              - flag: --anchor
                default_path: experiments/results/lane_a_landed/archive_lane_a.zip
            """
        )
    )
    violations = check_operator_wrapper_validates_required_input_files_pre_dispatch(
        repo_root=repo, strict=False, verbose=False
    )
    driver_violations = [v for v in violations if "Modal driver script" in v]
    assert len(driver_violations) >= 1, (
        f"Placeholder waiver MUST NOT excuse violation; got "
        f"{len(driver_violations)}: {violations}"
    )


def test_synthetic_modal_recipe_non_substrate_driver_skipped(tmp_path: Path) -> None:
    """Non-substrate lane scripts are out of scope (only
    scripts/remote_lane_substrate_*.sh are scanned)."""
    repo = tmp_path / "repo"
    (repo / ".omx" / "operator_authorize_recipes").mkdir(parents=True)
    (repo / "scripts").mkdir()
    # Lane script NOT under remote_lane_substrate_*; out of scope.
    driver = repo / "scripts" / "remote_lane_other_thing.sh"
    driver.write_text(
        textwrap.dedent(
            """
            #!/bin/bash
            WORKSPACE="${WORKSPACE:-/workspace/pact}"
            ANCHOR="$WORKSPACE/experiments/results/lane_a_landed/archive_lane_a.zip"
            """
        )
    )
    (repo / ".omx" / "operator_authorize_recipes" / "synth_modal.yaml").write_text(
        textwrap.dedent(
            """
            platform: modal
            lane_script: scripts/remote_lane_other_thing.sh
            required_input_files:
              - flag: --anchor
                default_path: experiments/results/lane_a_landed/archive_lane_a.zip
            """
        )
    )
    violations = check_operator_wrapper_validates_required_input_files_pre_dispatch(
        repo_root=repo, strict=False, verbose=False
    )
    driver_violations = [v for v in violations if "Modal driver script" in v]
    assert len(driver_violations) == 0, (
        f"Non-substrate driver MUST be out of scope; got "
        f"{len(driver_violations)}: {driver_violations}"
    )


def test_synthetic_modal_recipe_non_modal_platform_skipped(tmp_path: Path) -> None:
    """Non-Modal recipes are out of scope (Vast.ai/Lightning/Kaggle)."""
    repo = tmp_path / "repo"
    (repo / ".omx" / "operator_authorize_recipes").mkdir(parents=True)
    (repo / "scripts").mkdir()
    driver = repo / "scripts" / "remote_lane_substrate_synthetic_vast.sh"
    driver.write_text(
        textwrap.dedent(
            """
            #!/bin/bash
            WORKSPACE="${WORKSPACE:-/workspace/pact}"
            ANCHOR="$WORKSPACE/experiments/results/lane_a_landed/archive_lane_a.zip"
            """
        )
    )
    (repo / ".omx" / "operator_authorize_recipes" / "synth_vast.yaml").write_text(
        textwrap.dedent(
            """
            platform: vastai
            lane_script: scripts/remote_lane_substrate_synthetic_vast.sh
            required_input_files:
              - flag: --anchor
                default_path: experiments/results/lane_a_landed/archive_lane_a.zip
            """
        )
    )
    violations = check_operator_wrapper_validates_required_input_files_pre_dispatch(
        repo_root=repo, strict=False, verbose=False
    )
    driver_violations = [v for v in violations if "Modal driver script" in v]
    assert len(driver_violations) == 0, (
        f"Non-Modal platform recipes MUST be out of scope; got "
        f"{len(driver_violations)}: {driver_violations}"
    )


def test_synthetic_modal_recipe_driver_not_consuming_required_input_skipped(
    tmp_path: Path,
) -> None:
    """Driver that does NOT consume any required-input file is out of scope."""
    repo = tmp_path / "repo"
    (repo / ".omx" / "operator_authorize_recipes").mkdir(parents=True)
    (repo / "scripts").mkdir()
    driver = repo / "scripts" / "remote_lane_substrate_synthetic_noinput.sh"
    driver.write_text(
        textwrap.dedent(
            """
            #!/bin/bash
            WORKSPACE="${WORKSPACE:-/workspace/pact}"
            # Driver does not reference the recipe's required-input file at all.
            echo "Just train; no anchor needed"
            """
        )
    )
    (repo / ".omx" / "operator_authorize_recipes" / "synth_modal.yaml").write_text(
        textwrap.dedent(
            """
            platform: modal
            lane_script: scripts/remote_lane_substrate_synthetic_noinput.sh
            required_input_files:
              - flag: --anchor
                default_path: experiments/results/lane_a_landed/archive_lane_a.zip
            """
        )
    )
    violations = check_operator_wrapper_validates_required_input_files_pre_dispatch(
        repo_root=repo, strict=False, verbose=False
    )
    driver_violations = [v for v in violations if "Modal driver script" in v]
    assert len(driver_violations) == 0, (
        f"Driver not consuming the required input MUST be out of scope; "
        f"got {len(driver_violations)}: {driver_violations}"
    )


def test_strict_mode_raises_with_catalog_152_driver_path_message(tmp_path: Path) -> None:
    """Strict mode raises PreflightError citing Catalog #152."""
    repo = tmp_path / "repo"
    (repo / ".omx" / "operator_authorize_recipes").mkdir(parents=True)
    (repo / "scripts").mkdir()
    driver = repo / "scripts" / "remote_lane_substrate_synthetic_strict.sh"
    driver.write_text(
        textwrap.dedent(
            """
            #!/bin/bash
            WORKSPACE="${WORKSPACE:-/workspace/pact}"
            ANCHOR="$WORKSPACE/experiments/results/lane_a_landed/archive_lane_a.zip"
            """
        )
    )
    (repo / ".omx" / "operator_authorize_recipes" / "synth_modal.yaml").write_text(
        textwrap.dedent(
            """
            platform: modal
            lane_script: scripts/remote_lane_substrate_synthetic_strict.sh
            required_input_files:
              - flag: --anchor
                default_path: experiments/results/lane_a_landed/archive_lane_a.zip
            """
        )
    )
    with pytest.raises(PreflightError) as exc_info:
        check_operator_wrapper_validates_required_input_files_pre_dispatch(
            repo_root=repo, strict=True, verbose=False
        )
    err = str(exc_info.value)
    assert (
        "check_operator_wrapper_validates_required_input_files_pre_dispatch" in err
    )


def test_canonical_defensive_patterns_constant_pinned() -> None:
    """The defensive-pattern set MUST include the canonical helper name +
    other recognized idioms (regression guard)."""
    from tac.preflight import _CHECK_152_DRIVER_DEFENSIVE_PATTERNS

    pattern_strs = [p.pattern for p in _CHECK_152_DRIVER_DEFENSIVE_PATTERNS]
    # Canonical helper name must be present.
    assert any(
        "resolve_required_input_modal_aware" in p for p in pattern_strs
    ), "Canonical resolver name must be in defensive patterns"
    # MODAL_RUNTIME conditional pattern must be present.
    assert any("MODAL_RUNTIME" in p for p in pattern_strs), (
        "MODAL_RUNTIME conditional pattern must be in defensive patterns"
    )


def test_waiver_marker_constants_pinned() -> None:
    """The waiver-marker tuple MUST include BOTH the dedicated marker AND
    the Wave 1 sister marker (regression guard)."""
    from tac.preflight import _CHECK_152_DRIVER_WAIVER_TOKENS

    assert "DRIVER_PATH_MODAL_AWARE_OK" in _CHECK_152_DRIVER_WAIVER_TOKENS
    assert "REQUIRED_INPUT_MODAL_STAGED_OK" in _CHECK_152_DRIVER_WAIVER_TOKENS
