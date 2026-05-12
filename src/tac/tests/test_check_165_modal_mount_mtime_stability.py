"""Catalog #165 (FIX-I, D2): Modal mount-set mtime-stability check tests.

Coverage targets per the FIX-I landing memo:

- stable mount set: hash unchanged across the window -> proceed (no raise)
- mtime change during the window -> retry, then succeed once stable
- mtime change every retry -> ``MountUploadRaceError``
- per-file granularity: one stable + one changing -> still treated as unstable
- empty mount set -> no-op (no sleep, no raise)
- ``TAC_MODAL_MTIME_STABILITY_DISABLED=1`` opts out entirely
- ``build_training_image(mtime_stability_check=False)`` opts out at the kwarg
- ``build_training_image(mtime_stability_check=True)`` is the default and
  triggers the check
- the check fires BEFORE any ``add_local_*`` call (no torn upload)
- file-size-only change (no mtime change) still detected
- max_retries=0 -> single check, succeed-on-stable
- recursive directory walk picks up changes inside subdirs
- sleep_fn is invoked exactly N times when stable on attempt N
- the diagnostic message names the retry count + window seconds
- file deletion during window is detected as instability
- file creation in a watched dir during window is detected
- per-call mtime_stability_window_seconds threads through
- the stability check is called BEFORE trainer required_input_file validation
  (so a torn trainer manifest does not produce a misleading missing-file error)
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from tac.deploy.modal.mount_manifest import (
    DEFAULT_MTIME_STABILITY_MAX_RETRIES,
    DEFAULT_MTIME_STABILITY_WINDOW_SECONDS,
    MountUploadRaceError,
    _hash_mount_set_fingerprint,
    build_training_image,
    verify_mount_set_mtime_stability,
)


class FakeImage:
    """Records add_local_dir / add_local_file calls; used to assert that the
    stability check fires BEFORE any mount call."""

    def __init__(self) -> None:
        self.dirs: list[tuple[str, str, list[str] | None]] = []
        self.files: list[tuple[str, str]] = []

    def add_local_dir(
        self, local: str, *, remote_path: str, ignore: list[str] | None = None
    ) -> FakeImage:
        self.dirs.append((local, remote_path, ignore))
        return self

    def add_local_file(self, local: str, *, remote_path: str) -> FakeImage:
        self.files.append((local, remote_path))
        return self


def _make_fake_repo(tmp_path: Path) -> Path:
    root = tmp_path / "fakerepo"
    for d in ("src", "scripts", "upstream", "submissions", "experiments", "tools"):
        (root / d).mkdir(parents=True)
    (root / "pyproject.toml").write_text("[project]\nname='fake'\n")
    return root


# ---------------------------------------------------------------------------
# Direct verify_mount_set_mtime_stability() unit tests
# ---------------------------------------------------------------------------


def test_stable_paths_pass(tmp_path: Path) -> None:
    """Stable filesystem => single attempt succeeds; sleep_fn called once."""

    f = tmp_path / "stable.txt"
    f.write_text("hello")
    sleep_calls: list[float] = []

    verify_mount_set_mtime_stability(
        [f],
        window_seconds=0.0,
        max_retries=3,
        sleep_fn=lambda s: sleep_calls.append(s),
    )
    assert len(sleep_calls) == 1  # exactly one sleep before stability confirmed


def test_changing_paths_eventually_stable(tmp_path: Path) -> None:
    """File touched during the first window, then stable on retry => succeed."""

    f = tmp_path / "changing.txt"
    f.write_text("v1")
    state = {"calls": 0}

    def fake_sleep(_seconds: float) -> None:
        state["calls"] += 1
        # On call #1, mutate the file to simulate a concurrent writer.
        # On call #2+, do nothing => file is stable.
        if state["calls"] == 1:
            f.write_text("v2-different-size")
            os.utime(f, (time.time() + 1.0, time.time() + 1.0))

    verify_mount_set_mtime_stability(
        [f],
        window_seconds=0.0,
        max_retries=3,
        sleep_fn=fake_sleep,
    )
    # Stable on attempt 2: attempt1=hash-before, sleep1 (mutates), hash-after=changed;
    # attempt2=hash-before, sleep2 (noop), hash-after=stable => 2 sleep calls total.
    assert state["calls"] == 2


def test_perpetually_changing_paths_raise(tmp_path: Path) -> None:
    """File changes every retry => ``MountUploadRaceError`` after max_retries."""

    f = tmp_path / "racing.txt"
    f.write_text("v0")
    state = {"v": 0}

    def fake_sleep(_seconds: float) -> None:
        state["v"] += 1
        f.write_text(f"v{state['v']}-different")
        os.utime(f, (time.time() + state["v"], time.time() + state["v"]))

    with pytest.raises(MountUploadRaceError) as exc:
        verify_mount_set_mtime_stability(
            [f],
            window_seconds=0.0,
            max_retries=3,
            sleep_fn=fake_sleep,
        )
    # Error message names the retry count + Catalog #
    assert "3 retries" in str(exc.value)
    assert "Catalog #165" in str(exc.value)


def test_one_stable_one_changing_raises(tmp_path: Path) -> None:
    """Mixed mount set: one stable file + one racing file => instability."""

    stable = tmp_path / "stable.txt"
    racing = tmp_path / "racing.txt"
    stable.write_text("never-changes")
    racing.write_text("v0")
    state = {"v": 0}

    def fake_sleep(_seconds: float) -> None:
        state["v"] += 1
        racing.write_text(f"v{state['v']}-x")
        os.utime(racing, (time.time() + state["v"], time.time() + state["v"]))

    with pytest.raises(MountUploadRaceError):
        verify_mount_set_mtime_stability(
            [stable, racing],
            window_seconds=0.0,
            max_retries=3,
            sleep_fn=fake_sleep,
        )


def test_empty_mount_set_is_noop(tmp_path: Path) -> None:
    """No paths => no sleep, no hash, no raise."""

    sleep_calls: list[float] = []
    verify_mount_set_mtime_stability(
        [],
        window_seconds=0.0,
        max_retries=3,
        sleep_fn=lambda s: sleep_calls.append(s),
    )
    assert sleep_calls == []


def test_env_disable_opts_out(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``TAC_MODAL_MTIME_STABILITY_DISABLED=1`` skips the check entirely."""

    monkeypatch.setenv("TAC_MODAL_MTIME_STABILITY_DISABLED", "1")
    f = tmp_path / "racing.txt"
    f.write_text("v0")

    def fake_sleep(_seconds: float) -> None:
        f.write_text("v1-different")
        os.utime(f, (time.time() + 1.0, time.time() + 1.0))

    # No raise even though the file would otherwise race.
    verify_mount_set_mtime_stability(
        [f],
        window_seconds=0.0,
        max_retries=3,
        sleep_fn=fake_sleep,
    )


def test_directory_recursive_walk_detects_change(tmp_path: Path) -> None:
    """Change inside a subdir is detected as instability."""

    d = tmp_path / "dir"
    sub = d / "sub"
    sub.mkdir(parents=True)
    f = sub / "deep.txt"
    f.write_text("v0")
    state = {"v": 0}

    def fake_sleep(_seconds: float) -> None:
        state["v"] += 1
        f.write_text(f"v{state['v']}-xxx")
        os.utime(f, (time.time() + state["v"], time.time() + state["v"]))

    with pytest.raises(MountUploadRaceError):
        verify_mount_set_mtime_stability(
            [d],
            window_seconds=0.0,
            max_retries=2,
            sleep_fn=fake_sleep,
        )


def test_file_deletion_during_window_detected(tmp_path: Path) -> None:
    """File present at hash#1 but deleted before hash#2 => fingerprint change."""

    f = tmp_path / "vanishing.txt"
    f.write_text("here")
    state = {"v": 0}

    def fake_sleep(_seconds: float) -> None:
        state["v"] += 1
        if state["v"] == 1 and f.exists():
            f.unlink()

    with pytest.raises(MountUploadRaceError):
        verify_mount_set_mtime_stability(
            [f],
            window_seconds=0.0,
            max_retries=1,
            sleep_fn=fake_sleep,
        )


def test_file_creation_in_dir_during_window_detected(tmp_path: Path) -> None:
    """New file appears inside a watched dir => instability detected."""

    d = tmp_path / "watched"
    d.mkdir()
    (d / "initial.txt").write_text("a")
    state = {"v": 0}

    def fake_sleep(_seconds: float) -> None:
        state["v"] += 1
        if state["v"] == 1:
            (d / "new.txt").write_text("appeared")

    with pytest.raises(MountUploadRaceError):
        verify_mount_set_mtime_stability(
            [d],
            window_seconds=0.0,
            max_retries=1,
            sleep_fn=fake_sleep,
        )


def test_default_window_and_retries_constants() -> None:
    """The default window is non-zero (real upload-race window) and retries
    allow at least one retry past the initial check."""

    assert DEFAULT_MTIME_STABILITY_WINDOW_SECONDS > 0
    assert DEFAULT_MTIME_STABILITY_MAX_RETRIES >= 1


def test_hash_fingerprint_changes_on_mtime(tmp_path: Path) -> None:
    """The underlying hash function distinguishes (mtime, size) tuples."""

    f = tmp_path / "f.txt"
    f.write_text("hi")
    h1, _ = _hash_mount_set_fingerprint([f])
    # Bump mtime explicitly so the second hash sees a different stat.
    later = time.time() + 5.0
    os.utime(f, (later, later))
    h2, _ = _hash_mount_set_fingerprint([f])
    assert h1 != h2


def test_hash_fingerprint_changes_on_size(tmp_path: Path) -> None:
    """Size change with same mtime is detected."""

    f = tmp_path / "f.txt"
    f.write_text("short")
    h1, _ = _hash_mount_set_fingerprint([f])
    # Append data + force same mtime as before: hash must still differ
    # because we hash (mtime, size). The size moved.
    st1 = f.stat()
    f.write_text("much longer than before by far")
    os.utime(f, (st1.st_mtime, st1.st_mtime))
    h2, _ = _hash_mount_set_fingerprint([f])
    assert h1 != h2


# ---------------------------------------------------------------------------
# build_training_image() integration tests
# ---------------------------------------------------------------------------


def test_build_training_image_calls_stability_before_mounts(
    tmp_path: Path,
) -> None:
    """The stability check fires BEFORE any ``add_local_*`` call (no torn upload)."""

    root = _make_fake_repo(tmp_path)
    image = FakeImage()
    sleep_order: list[str] = []

    def tracking_sleep(_s: float) -> None:
        # When the stability check runs, no mount calls have happened yet.
        sleep_order.append(
            f"sleep@{len(image.dirs)}dirs/{len(image.files)}files"
        )

    build_training_image(
        image,
        repo_root=root,
        mtime_stability_check=True,
        mtime_stability_window_seconds=0.0,
        mtime_stability_max_retries=1,
        mtime_stability_sleep_fn=tracking_sleep,
    )
    # At sleep time: dirs and files lists are empty.
    assert sleep_order == ["sleep@0dirs/0files"]
    # After the build, the structural minimum is mounted.
    assert len(image.dirs) == 6
    assert len(image.files) == 1


def test_build_training_image_disable_kwarg(tmp_path: Path) -> None:
    """``mtime_stability_check=False`` skips the check (no sleep_fn calls)."""

    root = _make_fake_repo(tmp_path)
    image = FakeImage()
    sleep_calls: list[float] = []

    build_training_image(
        image,
        repo_root=root,
        mtime_stability_check=False,
        mtime_stability_sleep_fn=lambda s: sleep_calls.append(s),
    )
    assert sleep_calls == []


def test_build_training_image_raises_on_race(tmp_path: Path) -> None:
    """A racing file inside the structural-minimum set raises before mounts fire."""

    root = _make_fake_repo(tmp_path)
    image = FakeImage()
    # A file inside src/ that we will keep mutating.
    racing = root / "src" / "racing.py"
    racing.write_text("v0")
    state = {"v": 0}

    def fake_sleep(_s: float) -> None:
        state["v"] += 1
        racing.write_text(f"v{state['v']}-changed")
        os.utime(
            racing, (time.time() + state["v"], time.time() + state["v"])
        )

    with pytest.raises(MountUploadRaceError):
        build_training_image(
            image,
            repo_root=root,
            mtime_stability_check=True,
            mtime_stability_window_seconds=0.0,
            mtime_stability_max_retries=2,
            mtime_stability_sleep_fn=fake_sleep,
        )
    # Mount calls did NOT fire (fail-closed before mounting).
    assert image.dirs == []
    assert image.files == []


def test_build_training_image_stability_check_default_is_true() -> None:
    """The default for ``mtime_stability_check`` is True (defense-in-depth)."""

    import inspect

    sig = inspect.signature(build_training_image)
    param = sig.parameters["mtime_stability_check"]
    assert param.default is True


def test_build_training_image_default_window_threads_through(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The default ``mtime_stability_window_seconds`` threads into sleep_fn."""

    root = _make_fake_repo(tmp_path)
    image = FakeImage()
    captured: list[float] = []

    build_training_image(
        image,
        repo_root=root,
        mtime_stability_check=True,
        mtime_stability_max_retries=1,
        mtime_stability_sleep_fn=lambda s: captured.append(s),
    )
    assert captured == [DEFAULT_MTIME_STABILITY_WINDOW_SECONDS]


def test_diagnostic_message_names_catalog_and_retries(tmp_path: Path) -> None:
    """The error message names Catalog #165 + the retry count so the operator
    knows which gate fired and how many windows passed."""

    f = tmp_path / "x.txt"
    f.write_text("v0")
    state = {"v": 0}

    def fake_sleep(_s: float) -> None:
        state["v"] += 1
        f.write_text(f"v{state['v']}-z")
        os.utime(f, (time.time() + state["v"], time.time() + state["v"]))

    with pytest.raises(MountUploadRaceError) as exc:
        verify_mount_set_mtime_stability(
            [f],
            window_seconds=0.5,
            max_retries=4,
            sleep_fn=fake_sleep,
        )
    msg = str(exc.value)
    assert "4 retries" in msg
    assert "window=0.5s" in msg
    assert "Catalog #165" in msg


# ---------------------------------------------------------------------------
# STRICT preflight gate (Catalog #165) tests
# ---------------------------------------------------------------------------


def _make_repo_with_canonical_builder(tmp_path: Path) -> Path:
    """Build a fake repo containing a copy of the canonical mount_manifest.py.

    Tests then mutate the copy to exercise the preflight check.
    """

    root = tmp_path / "fakerepo"
    (root / "src" / "tac" / "deploy" / "modal").mkdir(parents=True)
    canonical = Path(__file__).resolve().parents[1] / "deploy" / "modal" / "mount_manifest.py"
    dest = root / "src" / "tac" / "deploy" / "modal" / "mount_manifest.py"
    dest.write_text(canonical.read_text(encoding="utf-8"))
    return root


def test_preflight_check_165_passes_on_canonical_builder(tmp_path: Path) -> None:
    """The pristine canonical builder satisfies all four contract conditions."""

    from tac.preflight import check_modal_mount_builder_uses_mtime_stability_check

    root = _make_repo_with_canonical_builder(tmp_path)
    violations = check_modal_mount_builder_uses_mtime_stability_check(
        repo_root=root, strict=False, verbose=False
    )
    assert violations == []


def test_preflight_check_165_fires_on_missing_kwarg(tmp_path: Path) -> None:
    """Removing the ``mtime_stability_check`` kwarg from build_training_image
    triggers the gate."""

    from tac.preflight import check_modal_mount_builder_uses_mtime_stability_check

    root = _make_repo_with_canonical_builder(tmp_path)
    p = root / "src" / "tac" / "deploy" / "modal" / "mount_manifest.py"
    body = p.read_text()
    # Remove the kwarg name entirely so neither the kwarg-token check nor
    # the verify-call-token check finds it. We rename the kwarg to a
    # legacy name to simulate a refactor.
    body = body.replace("mtime_stability_check", "_legacy_renamed_kwarg")
    p.write_text(body)

    violations = check_modal_mount_builder_uses_mtime_stability_check(
        repo_root=root, strict=False, verbose=False
    )
    # Expect at least the missing-kwarg violation.
    assert any("mtime_stability_check" in v for v in violations)


def test_preflight_check_165_fires_on_missing_verify_call(tmp_path: Path) -> None:
    """Removing the active call to ``verify_mount_set_mtime_stability`` from
    inside ``build_training_image`` triggers the gate."""

    from tac.preflight import check_modal_mount_builder_uses_mtime_stability_check

    root = _make_repo_with_canonical_builder(tmp_path)
    p = root / "src" / "tac" / "deploy" / "modal" / "mount_manifest.py"
    body = p.read_text()
    # Replace the active call inside build_training_image with a no-op
    # to simulate a refactor that drops the wire-in. We keep the helper
    # definition itself.
    builder_idx = body.find("def build_training_image")
    assert builder_idx >= 0
    next_def_idx = body.find("\ndef ", builder_idx + 1)
    builder_body = body[builder_idx:next_def_idx]
    neutered = builder_body.replace("verify_mount_set_mtime_stability(", "noop_function(")
    body = body[:builder_idx] + neutered + body[next_def_idx:]
    p.write_text(body)

    violations = check_modal_mount_builder_uses_mtime_stability_check(
        repo_root=root, strict=False, verbose=False
    )
    assert any("verify_mount_set_mtime_stability(" in v for v in violations)


def test_preflight_check_165_fires_on_missing_required_surfaces(
    tmp_path: Path,
) -> None:
    """Removing ``MountUploadRaceError`` class definition triggers the gate."""

    from tac.preflight import check_modal_mount_builder_uses_mtime_stability_check

    root = _make_repo_with_canonical_builder(tmp_path)
    p = root / "src" / "tac" / "deploy" / "modal" / "mount_manifest.py"
    body = p.read_text()
    body = body.replace("class MountUploadRaceError", "class _RemovedFormerlyMountUploadRace")
    p.write_text(body)

    violations = check_modal_mount_builder_uses_mtime_stability_check(
        repo_root=root, strict=False, verbose=False
    )
    assert any("MountUploadRaceError" in v for v in violations)


def test_preflight_check_165_strict_raises(tmp_path: Path) -> None:
    """In strict mode, violations raise ``PreflightError``."""

    from tac.preflight import PreflightError, check_modal_mount_builder_uses_mtime_stability_check

    root = _make_repo_with_canonical_builder(tmp_path)
    p = root / "src" / "tac" / "deploy" / "modal" / "mount_manifest.py"
    # Nuke the file content entirely (the file still exists; the contract
    # surfaces are gone).
    p.write_text("# emptied — canonical builder removed")

    with pytest.raises(PreflightError) as exc:
        check_modal_mount_builder_uses_mtime_stability_check(
            repo_root=root, strict=True, verbose=False
        )
    assert "Catalog #165" in str(exc.value)


def test_preflight_check_165_fires_on_missing_file(tmp_path: Path) -> None:
    """If the canonical mount_manifest.py file is deleted, the gate fires."""

    from tac.preflight import check_modal_mount_builder_uses_mtime_stability_check

    root = _make_repo_with_canonical_builder(tmp_path)
    p = root / "src" / "tac" / "deploy" / "modal" / "mount_manifest.py"
    p.unlink()

    violations = check_modal_mount_builder_uses_mtime_stability_check(
        repo_root=root, strict=False, verbose=False
    )
    assert any("missing" in v.lower() for v in violations)


def test_check_runs_before_trainer_required_input_validation(
    tmp_path: Path,
) -> None:
    """If a trainer-required-input file is MISSING but the mount set is racing,
    the stability error wins (it fires first, in step 0)."""

    root = _make_fake_repo(tmp_path)
    # Build a trainer module that declares a required_input_file pointing to a
    # MISSING path. The stability check must fire BEFORE the missing-file
    # detection.
    trainer = tmp_path / "fake_trainer.py"
    trainer.write_text(
        "TIER_1_OPERATOR_REQUIRED_FLAGS = {\n"
        "    '--fake-flag': {\n"
        "        'required_input_file': True,\n"
        "        'default': '.omx/research/does_not_exist.json',\n"
        "    },\n"
        "}\n"
    )

    image = FakeImage()
    # Set up a racing file inside src/ so the stability check fires.
    racing = root / "src" / "racing.py"
    racing.write_text("v0")
    state = {"v": 0}

    def fake_sleep(_s: float) -> None:
        state["v"] += 1
        racing.write_text(f"v{state['v']}-changed")
        os.utime(
            racing, (time.time() + state["v"], time.time() + state["v"])
        )

    # The stability check should raise BEFORE the trainer-required-file
    # validation (which would have raised "missing required_input_file
    # default" otherwise).
    with pytest.raises(MountUploadRaceError):
        build_training_image(
            image,
            repo_root=root,
            trainer_module_path=trainer,
            mtime_stability_check=True,
            mtime_stability_window_seconds=0.0,
            mtime_stability_max_retries=1,
            mtime_stability_sleep_fn=fake_sleep,
        )
