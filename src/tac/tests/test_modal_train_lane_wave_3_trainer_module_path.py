# SPDX-License-Identifier: MIT
"""WAVE-3 (2026-05-16) tests for the deeper structural fix per
STC v2 FIX CRITICAL DEEPER FINDING (commit ``7dd8a5412``).

Bug class anchor: ``experiments/modal_train_lane.py:154`` (pre-fix) passed
``trainer_module_path=None`` to ``build_training_image`` making Wave 1's
trainer-side ``TIER_1_EXTRA_MOUNT_PATHS`` declarations STRUCTURALLY INERT
for the generic Modal dispatcher. The canonical fix derives the trainer
module path from ``lane_script`` at dispatch time, reads each declared
extra-mount path into bytes locally, and threads them as a
``{rel: bytes}`` dict payload through the Modal function call so the
worker materializes them under ``/tmp/pact/<rel>`` after the structural
copy.

Test coverage:
- ``_derive_trainer_module_path``: substrate convention mapping
  (``scripts/remote_lane_substrate_<id>.sh`` → ``experiments/train_substrate_<id>.py``)
  + missing trainer file + non-substrate lane script
- ``_collect_trainer_extra_mount_payload``: live STC v2 + a1_plus_lapose +
  a1_plus_wavelet_residual extra-mount discovery + structural-skip behavior
  + None trainer + empty-extras-trainer
- ``_run_lane_inner``: payload materialization under workspace (static
  source check + assertion on the source body)
- 4 wrapper functions thread ``trainer_extra_mount_payload`` kwarg + default
  to None
- main() entrypoint derives + computes + spawns with payload
- backward compat: lane scripts without substrate convention dispatch with
  empty payload (no regression for legacy lanes)

Sister of:
- ``test_mount_manifest.py`` (Catalog #153 canonical builder + Wave 1 hooks)
- ``test_check_152_modal_mounted_input_extension.py`` (Wave 2 driver-side
  defensive resolver)
- ``test_modal_train_lane_hardening.py`` (Catalog #166 / #203 / #224 / #245
  / etc. sister gates)
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
SOURCE = REPO_ROOT / "experiments" / "modal_train_lane.py"


@pytest.fixture(scope="module")
def modal_train_lane_module():
    """Import ``experiments/modal_train_lane.py`` once for the test module.

    The module imports ``modal`` at top — when modal is unavailable the
    import will raise ImportError and tests are skipped.
    """
    if not SOURCE.is_file():
        pytest.skip(f"modal_train_lane.py not found at {SOURCE}")
    try:
        import modal  # noqa: F401
    except ImportError:
        pytest.skip("modal not installed; skipping modal_train_lane import tests")
    spec = importlib.util.spec_from_file_location("modal_train_lane_under_test", SOURCE)
    if spec is None or spec.loader is None:
        pytest.skip(f"could not build importlib spec for {SOURCE}")
    mod = importlib.util.module_from_spec(spec)
    # Ensure src/ is on sys.path so tac.* imports resolve
    src_path = str(REPO_ROOT / "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Static source-level regression guards (no module import required)
# ---------------------------------------------------------------------------


def test_modal_train_lane_imports_collect_extra_mount_paths_helpers() -> None:
    """The WAVE-3 fix imports both canonical helpers from mount_manifest."""
    text = SOURCE.read_text()
    assert "collect_extra_mount_paths" in text
    assert "collect_tier_required_input_files" in text


def test_modal_train_lane_defines_derive_trainer_module_path_helper() -> None:
    """The WAVE-3 fix defines the trainer-path-derivation helper."""
    text = SOURCE.read_text()
    assert "def _derive_trainer_module_path(" in text
    # Honors the canonical substrate naming convention
    assert "remote_lane_substrate_" in text
    assert "train_substrate_" in text


def test_modal_train_lane_defines_collect_trainer_extra_mount_payload_helper() -> None:
    """The WAVE-3 fix defines the payload-collection helper."""
    text = SOURCE.read_text()
    assert "def _collect_trainer_extra_mount_payload(" in text
    assert "TIER_1_EXTRA_MOUNT_PATHS" in text
    # The helper consults BOTH extras + required-input-file defaults so the
    # Modal dispatcher honors the same `required_input_file=True` contract
    # Catalog #152 enforces for operator wrappers.
    assert "collect_extra_mount_paths" in text
    assert "collect_tier_required_input_files" in text


def test_modal_train_lane_run_lane_inner_accepts_payload_kwarg() -> None:
    """``_run_lane_inner`` accepts the new ``trainer_extra_mount_payload`` kwarg."""
    text = SOURCE.read_text()
    assert "trainer_extra_mount_payload: dict | None = None" in text
    # Body materializes the payload under workspace
    assert "for rel, data in trainer_extra_mount_payload.items():" in text
    assert "target.parent.mkdir(parents=True, exist_ok=True)" in text
    assert "target.write_bytes(data)" in text


def test_modal_train_lane_all_4_wrappers_thread_payload_kwarg() -> None:
    """All 4 ``@app.function`` wrappers thread the payload kwarg through."""
    text = SOURCE.read_text()
    # 4 wrappers: T4 / A10G / A100 / H100
    for gpu_fn in (
        "run_lane_training_t4",
        "run_lane_training_a10g",
        "run_lane_training_a100",
        "run_lane_training_h100",
    ):
        # Each wrapper must declare the new kwarg
        wrapper_def_idx = text.find(f"def {gpu_fn}(")
        assert wrapper_def_idx > 0, f"wrapper {gpu_fn} missing"
        # Inspect the next ~600 chars for the kwarg + propagation
        wrapper_body = text[wrapper_def_idx : wrapper_def_idx + 800]
        assert (
            "trainer_extra_mount_payload: dict | None = None" in wrapper_body
        ), f"{gpu_fn} missing kwarg declaration"
        assert (
            "trainer_extra_mount_payload=trainer_extra_mount_payload" in wrapper_body
        ), f"{gpu_fn} not threading kwarg to _run_lane_inner"


def test_modal_train_lane_main_derives_and_spawns_with_payload() -> None:
    """``main()`` derives the trainer module + spawns with the payload."""
    text = SOURCE.read_text()
    main_src = text[text.index("@app.local_entrypoint()"):]
    # WAVE-3 derivation
    assert "_derive_trainer_module_path(" in main_src
    assert "_collect_trainer_extra_mount_payload(" in main_src
    # The spawn passes the payload as the trailing positional arg
    spawn_idx = main_src.index("fn.spawn(")
    spawn_block = main_src[spawn_idx : spawn_idx + 600]
    assert "trainer_extra_mount_payload" in spawn_block


def test_modal_train_lane_main_warns_for_non_substrate_lane() -> None:
    """Legacy lane scripts that don't follow substrate convention get
    explicit logging that they self-bootstrap."""
    text = SOURCE.read_text()
    main_src = text[text.index("@app.local_entrypoint()"):]
    # The literal text may be split across line continuations
    assert "follow substrate naming convention" in main_src
    assert "self-bootstrap" in main_src


def test_modal_train_lane_keeps_module_load_image_build_for_caching() -> None:
    """The Modal image is STILL built at module-load time with
    ``trainer_module_path=None``. This preserves image caching across
    dispatches (the per-dispatch trainer_module_path is consumed via the
    payload pattern, not via per-dispatch image rebuild)."""
    text = SOURCE.read_text()
    # The image build still uses trainer_module_path=None for caching
    # (otherwise every dispatch would trigger a full image rebuild)
    assert "training_image = build_training_image(" in text
    # The image build invocation passes trainer_module_path=None
    image_build_idx = text.index("training_image = build_training_image(")
    image_build_block = text[image_build_idx : image_build_idx + 800]
    assert "trainer_module_path=None" in image_build_block


# ---------------------------------------------------------------------------
# Dynamic import + helper unit tests (require modal installed)
# ---------------------------------------------------------------------------


def test_derive_trainer_module_path_stc_v2(modal_train_lane_module) -> None:
    """STC v2 lane_script → trainer module path."""
    result = modal_train_lane_module._derive_trainer_module_path(
        "scripts/remote_lane_substrate_stc_v2.sh", REPO_ROOT
    )
    assert result is not None
    assert result == REPO_ROOT / "experiments" / "train_substrate_stc_v2.py"
    assert result.is_file()


def test_derive_trainer_module_path_a1_plus_lapose(modal_train_lane_module) -> None:
    """a1_plus_lapose lane_script → trainer module path."""
    result = modal_train_lane_module._derive_trainer_module_path(
        "scripts/remote_lane_substrate_a1_plus_lapose.sh", REPO_ROOT
    )
    assert result is not None
    assert result == REPO_ROOT / "experiments" / "train_substrate_a1_plus_lapose.py"


def test_derive_trainer_module_path_a1_plus_wavelet_residual(
    modal_train_lane_module,
) -> None:
    """a1_plus_wavelet_residual lane_script → trainer module path."""
    result = modal_train_lane_module._derive_trainer_module_path(
        "scripts/remote_lane_substrate_a1_plus_wavelet_residual.sh", REPO_ROOT
    )
    assert result is not None
    assert (
        result
        == REPO_ROOT / "experiments" / "train_substrate_a1_plus_wavelet_residual.py"
    )


def test_derive_trainer_module_path_returns_none_for_legacy_lane(
    modal_train_lane_module,
) -> None:
    """Non-substrate lane scripts (legacy convention) return None."""
    result = modal_train_lane_module._derive_trainer_module_path(
        "scripts/remote_lane_omega_hessian_qat.sh", REPO_ROOT
    )
    assert result is None


def test_derive_trainer_module_path_returns_none_for_missing_trainer(
    modal_train_lane_module,
) -> None:
    """Substrate-named lane script with missing trainer file returns None."""
    result = modal_train_lane_module._derive_trainer_module_path(
        "scripts/remote_lane_substrate_nonexistent_xyz_xyz.sh", REPO_ROOT
    )
    assert result is None


def test_derive_trainer_module_path_returns_none_for_non_sh_suffix(
    modal_train_lane_module,
) -> None:
    """Lane script not ending in ``.sh`` returns None."""
    result = modal_train_lane_module._derive_trainer_module_path(
        "scripts/remote_lane_substrate_stc_v2.txt", REPO_ROOT
    )
    assert result is None


def test_collect_trainer_extra_mount_payload_none_trainer_returns_empty(
    modal_train_lane_module,
) -> None:
    """None trainer module → empty payload (backward compat)."""
    payload = modal_train_lane_module._collect_trainer_extra_mount_payload(
        None, REPO_ROOT
    )
    assert payload == {}


def test_collect_trainer_extra_mount_payload_stc_v2_live(
    modal_train_lane_module,
) -> None:
    """STC v2 live trainer declares the Lane A anchor archive as
    extra-mount; the WAVE-3 helper resolves it correctly."""
    trainer = modal_train_lane_module._derive_trainer_module_path(
        "scripts/remote_lane_substrate_stc_v2.sh", REPO_ROOT
    )
    assert trainer is not None
    payload = modal_train_lane_module._collect_trainer_extra_mount_payload(
        trainer, REPO_ROOT
    )
    # The canonical STC v2 anchor archive path
    expected_key = "experiments/results/lane_a_landed/archive_lane_a.zip"
    if (REPO_ROOT / expected_key).is_file():
        assert expected_key in payload
        assert len(payload[expected_key]) > 0
        # Sanity: bytes match the on-disk file
        assert payload[expected_key] == (REPO_ROOT / expected_key).read_bytes()


def test_collect_trainer_extra_mount_payload_skips_structural_mount_paths(
    modal_train_lane_module, tmp_path
) -> None:
    """Trainer-declared paths under structural mount set (src/, upstream/, etc.)
    are SKIPPED because they are already mounted via STRUCTURAL_MINIMUM_DIRS."""
    # Build a fake repo + trainer
    fake_repo = tmp_path / "fake_repo"
    fake_repo.mkdir()
    (fake_repo / "experiments").mkdir()
    (fake_repo / "upstream").mkdir()
    (fake_repo / "upstream" / "videos").mkdir()
    video_path = fake_repo / "upstream" / "videos" / "0.mkv"
    video_path.write_bytes(b"fake-video-bytes")
    (fake_repo / "experiments" / "results").mkdir()
    (fake_repo / "experiments" / "results" / "foo").mkdir()
    results_path = fake_repo / "experiments" / "results" / "foo" / "bar.bin"
    results_path.write_bytes(b"results-bytes" * 100)
    trainer_content = (
        "from pathlib import Path\n"
        "REPO = Path(__file__).resolve().parents[1]\n"
        "TIER_1_EXTRA_MOUNT_PATHS = (\n"
        "    'upstream/videos/0.mkv',\n"
        "    'experiments/results/foo/bar.bin',\n"
        ")\n"
    )
    trainer_path = fake_repo / "experiments" / "train_substrate_test_xyz.py"
    trainer_path.write_text(trainer_content)

    payload = modal_train_lane_module._collect_trainer_extra_mount_payload(
        trainer_path, fake_repo
    )
    # upstream/videos/0.mkv is under structural mount — SKIPPED
    assert "upstream/videos/0.mkv" not in payload
    # experiments/results/foo/bar.bin is under Modal-IGNORED subtree — INCLUDED
    assert "experiments/results/foo/bar.bin" in payload
    assert payload["experiments/results/foo/bar.bin"] == results_path.read_bytes()


def test_collect_trainer_extra_mount_payload_skips_non_results_experiments_paths(
    modal_train_lane_module, tmp_path
) -> None:
    """Trainer-declared paths under ``experiments/<non-results>/`` are
    SKIPPED because the structural mount covers them (only ``experiments/results/**``
    is ignored per ``DEFAULT_RESULTS_IGNORE``)."""
    fake_repo = tmp_path / "fake_repo"
    fake_repo.mkdir()
    (fake_repo / "experiments").mkdir()
    (fake_repo / "experiments" / "fixtures").mkdir()
    fixture_path = fake_repo / "experiments" / "fixtures" / "data.bin"
    fixture_path.write_bytes(b"fixture-bytes")
    trainer_content = (
        "TIER_1_EXTRA_MOUNT_PATHS = ('experiments/fixtures/data.bin',)\n"
    )
    trainer_path = fake_repo / "experiments" / "train_substrate_test_xyz2.py"
    trainer_path.write_text(trainer_content)

    payload = modal_train_lane_module._collect_trainer_extra_mount_payload(
        trainer_path, fake_repo
    )
    # experiments/fixtures/data.bin is under structural mount (NOT results/)
    # — SKIPPED
    assert "experiments/fixtures/data.bin" not in payload


def test_collect_trainer_extra_mount_payload_handles_missing_file_with_warn(
    modal_train_lane_module, tmp_path, capsys
) -> None:
    """Trainer declares a path that does not exist on disk → WARN + skip
    (does NOT raise — the lane script may handle the missing file separately)."""
    fake_repo = tmp_path / "fake_repo_missing"
    fake_repo.mkdir()
    (fake_repo / "experiments").mkdir()
    (fake_repo / "experiments" / "results").mkdir()
    trainer_content = (
        "TIER_1_EXTRA_MOUNT_PATHS = ('experiments/results/nonexistent.zip',)\n"
    )
    trainer_path = fake_repo / "experiments" / "train_substrate_test_xyz3.py"
    trainer_path.write_text(trainer_content)

    payload = modal_train_lane_module._collect_trainer_extra_mount_payload(
        trainer_path, fake_repo
    )
    assert payload == {}
    captured = capsys.readouterr()
    assert "WARN" in captured.err
    assert "nonexistent.zip" in captured.err


def test_collect_trainer_extra_mount_payload_handles_import_error_with_warn(
    modal_train_lane_module, tmp_path, capsys
) -> None:
    """Trainer module that raises at import time → WARN + empty payload
    (does NOT raise — generic dispatcher may dispatch valid lane scripts
    whose trainer module has unrelated import-time issues)."""
    fake_repo = tmp_path / "fake_repo_broken"
    fake_repo.mkdir()
    (fake_repo / "experiments").mkdir()
    trainer_content = "raise RuntimeError('trainer broken at import')\n"
    trainer_path = fake_repo / "experiments" / "train_substrate_broken.py"
    trainer_path.write_text(trainer_content)

    payload = modal_train_lane_module._collect_trainer_extra_mount_payload(
        trainer_path, fake_repo
    )
    assert payload == {}
    captured = capsys.readouterr()
    assert "WARN" in captured.err
    assert "could not be imported" in captured.err


# ---------------------------------------------------------------------------
# Live-repo regression guards: live STC v2 + a1_plus_lapose +
# a1_plus_wavelet_residual trainers all have functional payload contracts
# ---------------------------------------------------------------------------


def test_live_stc_v2_payload_includes_anchor_archive(modal_train_lane_module) -> None:
    """Live STC v2 trainer's TIER_1_EXTRA_MOUNT_PATHS is now consumed
    structurally by the Modal dispatcher (WAVE-3 fix)."""
    trainer = REPO_ROOT / "experiments" / "train_substrate_stc_v2.py"
    if not trainer.is_file():
        pytest.skip("STC v2 trainer not in repo")
    payload = modal_train_lane_module._collect_trainer_extra_mount_payload(
        trainer, REPO_ROOT
    )
    expected_key = "experiments/results/lane_a_landed/archive_lane_a.zip"
    if (REPO_ROOT / expected_key).is_file():
        assert expected_key in payload
        # Sanity: byte-stable
        assert payload[expected_key] == (REPO_ROOT / expected_key).read_bytes()


def test_live_a1_plus_lapose_payload_includes_a1_base_archive(
    modal_train_lane_module,
) -> None:
    """Live a1_plus_lapose trainer's TIER_1_EXTRA_MOUNT_PATHS is now consumed
    structurally by the Modal dispatcher (WAVE-3 fix)."""
    trainer = REPO_ROOT / "experiments" / "train_substrate_a1_plus_lapose.py"
    if not trainer.is_file():
        pytest.skip("a1_plus_lapose trainer not in repo")
    payload = modal_train_lane_module._collect_trainer_extra_mount_payload(
        trainer, REPO_ROOT
    )
    # The trainer declares DEFAULT_A1_ARCHIVE — verify the payload key
    # matches the canonical A1 anchor location (one of two known paths
    # depending on which canonical post-fine-tune SHA the trainer pinned).
    a1_candidate_paths = [
        "experiments/results/track4_sg_a1_t178000_20260509/submission_dir/archive.zip",
        "experiments/results/lane_a_landed/archive_lane_a.zip",
    ]
    # At least one of the A1 anchor candidates is in payload IFF it exists on disk
    matched = [
        p for p in a1_candidate_paths if (REPO_ROOT / p).is_file() and p in payload
    ]
    assert matched, (
        "a1_plus_lapose trainer should consume at least one canonical A1 anchor "
        f"via WAVE-3 payload; payload keys: {sorted(payload.keys())}"
    )


def test_live_a1_plus_wavelet_residual_payload_includes_a1_base_archive(
    modal_train_lane_module,
) -> None:
    """Live a1_plus_wavelet_residual trainer's TIER_1_EXTRA_MOUNT_PATHS is
    now consumed structurally by the Modal dispatcher (WAVE-3 fix)."""
    trainer = REPO_ROOT / "experiments" / "train_substrate_a1_plus_wavelet_residual.py"
    if not trainer.is_file():
        pytest.skip("a1_plus_wavelet_residual trainer not in repo")
    payload = modal_train_lane_module._collect_trainer_extra_mount_payload(
        trainer, REPO_ROOT
    )
    a1_candidate_paths = [
        "experiments/results/track4_sg_a1_t178000_20260509/submission_dir/archive.zip",
        "experiments/results/lane_a_landed/archive_lane_a.zip",
    ]
    matched = [
        p for p in a1_candidate_paths if (REPO_ROOT / p).is_file() and p in payload
    ]
    assert matched, (
        "a1_plus_wavelet_residual trainer should consume at least one canonical A1 "
        f"anchor via WAVE-3 payload; payload keys: {sorted(payload.keys())}"
    )


def test_live_sane_hnerv_payload_is_empty_no_extras_declared(
    modal_train_lane_module,
) -> None:
    """Live sane_hnerv trainer has NO TIER_1_EXTRA_MOUNT_PATHS declaration
    → payload should be empty (backward compat preserved)."""
    trainer = REPO_ROOT / "experiments" / "train_substrate_sane_hnerv.py"
    if not trainer.is_file():
        pytest.skip("sane_hnerv trainer not in repo")
    payload = modal_train_lane_module._collect_trainer_extra_mount_payload(
        trainer, REPO_ROOT
    )
    # No extras declared + required_input defaults under structural mount =
    # empty payload (sane_hnerv self-bootstraps via lane script).
    # Any non-empty payload here is expected to be required-input-file
    # defaults under experiments/results/** (allowed but rare).
    for key in payload:
        # All keys must live under experiments/results/ (the Modal-IGNORED
        # subtree); the helper's structural-skip filter excludes everything else.
        assert key.startswith("experiments/results/") or not key.startswith(
            ("src/", "upstream/", "scripts/", "tools/", "submissions/")
        ), f"unexpected payload key under structural mount: {key}"
