"""Tests for the Catalog #207 ``--force`` rmtree namespace + manifest guard.

Empirical anchor 2026-05-14: Codex finding HIGH —
``tools/build_pr101_nonlocal_sweep_packets.py`` passed a user-controlled
``--out-dir`` directly to ``shutil.rmtree`` when ``--force`` was set. A typo
such as ``--out-dir . --force`` or an absolute non-repo path would
recursively delete unrelated repo/user state.

The fix wires ``_assert_rmtree_safe`` BEFORE the ``rmtree`` call. The guard
refuses any path that is the repo root, $HOME, outside the repo, outside
the canonical ``experiments/results/`` namespace, OR does not contain the
tool-owned manifest ``build_pr101_nonlocal_sweep_manifest.json``.

Sister of Catalog #154 (canonical GC helper for ``experiments/results/``).
Memory: feedback_codex_3_findings_fix_landed_20260514.md.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
TOOL_PATH = REPO_ROOT / "tools" / "build_pr101_nonlocal_sweep_packets.py"


def _load_tool_module(monkeypatch=None, repo_override: Path | None = None):
    """Load the tool module dynamically (it is a script, not a package)."""
    spec = importlib.util.spec_from_file_location(
        "build_pr101_nonlocal_sweep_packets", TOOL_PATH
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    # Pre-load 'tools' package shim if needed so the local import in the tool
    # body (``from tools.tool_bootstrap import ...``) resolves cleanly. The
    # tool's body already guards with a try/except ModuleNotFoundError so we
    # don't need to interfere with the prep work.
    sys.modules.setdefault("build_pr101_nonlocal_sweep_packets", module)
    spec.loader.exec_module(module)
    if repo_override is not None and monkeypatch is not None:
        monkeypatch.setattr(module, "REPO_ROOT", repo_override)
    return module


@pytest.fixture
def tool_module():
    return _load_tool_module()


# ─── Positive case: clean tool-owned dir under namespace is accepted ────


def test_force_accepts_namespaced_dir_with_tool_owned_manifest(
    tool_module, tmp_path
):
    """Tool-owned manifest + experiments/results path = OK."""
    fake_repo = tmp_path / "fakerepo"
    (fake_repo / "experiments" / "results" / "pr101_sweep").mkdir(parents=True)
    manifest = (
        fake_repo / "experiments" / "results" / "pr101_sweep"
        / tool_module.TOOL_OWNED_MANIFEST_NAME
    )
    manifest.write_text(
        json.dumps({"tool": tool_module.TOOL_NAME, "created_at_utc": "x"})
    )
    target = fake_repo / "experiments" / "results" / "pr101_sweep"
    # No exception → guard accepted.
    tool_module._assert_rmtree_safe(target, repo_root=fake_repo)


def test_force_accepts_legacy_manifest_name(tool_module, tmp_path):
    """Legacy ``manifest.json`` filename is also accepted."""
    fake_repo = tmp_path / "fakerepo"
    target = fake_repo / "experiments" / "results" / "pr101_sweep"
    target.mkdir(parents=True)
    (target / "manifest.json").write_text(
        json.dumps({"tool": tool_module.TOOL_NAME})
    )
    tool_module._assert_rmtree_safe(target, repo_root=fake_repo)


# ─── Refusal: repo root, $HOME, absolute non-repo ───────────────────────


def test_force_refuses_repo_root(tool_module, tmp_path):
    fake_repo = tmp_path / "fakerepo"
    fake_repo.mkdir()
    with pytest.raises(tool_module.UnsafeRmtreeRefusedError) as exc:
        tool_module._assert_rmtree_safe(fake_repo, repo_root=fake_repo)
    assert "repo root" in str(exc.value)


def test_force_refuses_dot_relative_repo_root(tool_module, tmp_path):
    """The canonical typo: --out-dir . --force resolves to repo root."""
    fake_repo = tmp_path / "fakerepo"
    fake_repo.mkdir()
    # Simulate args.out_dir = Path(".") → repo_path = REPO_ROOT.
    with pytest.raises(tool_module.UnsafeRmtreeRefusedError):
        tool_module._assert_rmtree_safe(fake_repo, repo_root=fake_repo)


def test_force_refuses_home_directory(tool_module, monkeypatch, tmp_path):
    fake_repo = tmp_path / "fakerepo"
    fake_repo.mkdir()
    fake_home = tmp_path / "fakehome"
    fake_home.mkdir()
    monkeypatch.setattr(
        tool_module.Path, "home", staticmethod(lambda: fake_home)
    )
    with pytest.raises(tool_module.UnsafeRmtreeRefusedError) as exc:
        tool_module._assert_rmtree_safe(fake_home, repo_root=fake_repo)
    assert "$HOME" in str(exc.value)


def test_force_refuses_absolute_path_outside_repo(tool_module, tmp_path):
    fake_repo = tmp_path / "fakerepo"
    fake_repo.mkdir()
    outside = tmp_path / "outside_repo"
    outside.mkdir()
    with pytest.raises(tool_module.UnsafeRmtreeRefusedError) as exc:
        tool_module._assert_rmtree_safe(outside, repo_root=fake_repo)
    assert "NOT under the repo root" in str(exc.value)


def test_force_refuses_namespace_root_itself(tool_module, tmp_path):
    """experiments/results/ root MUST NOT be force-deleted."""
    fake_repo = tmp_path / "fakerepo"
    fake_repo.mkdir()
    namespace_root = fake_repo / "experiments" / "results"
    namespace_root.mkdir(parents=True)
    with pytest.raises(tool_module.UnsafeRmtreeRefusedError) as exc:
        tool_module._assert_rmtree_safe(namespace_root, repo_root=fake_repo)
    assert "namespace root" in str(exc.value)


def test_force_refuses_repo_but_outside_namespace(tool_module, tmp_path):
    """Refuse paths under repo root but NOT under experiments/results/."""
    fake_repo = tmp_path / "fakerepo"
    fake_repo.mkdir()
    bad = fake_repo / "src" / "tac"
    bad.mkdir(parents=True)
    with pytest.raises(tool_module.UnsafeRmtreeRefusedError) as exc:
        tool_module._assert_rmtree_safe(bad, repo_root=fake_repo)
    assert "experiments/results" in str(exc.value)


def test_force_refuses_parent_directory_escape(tool_module, tmp_path):
    """A path with `..` that escapes the namespace is refused."""
    fake_repo = tmp_path / "fakerepo"
    fake_repo.mkdir()
    # Construct a path that, when resolved, escapes the namespace.
    sneaky = fake_repo / "experiments" / "results" / ".." / ".." / "tools"
    (fake_repo / "tools").mkdir(parents=True, exist_ok=True)
    with pytest.raises(tool_module.UnsafeRmtreeRefusedError):
        tool_module._assert_rmtree_safe(sneaky, repo_root=fake_repo)


# ─── Refusal: namespace OK but missing tool-owned manifest ───────────


def test_force_refuses_namespace_dir_without_tool_owned_manifest(
    tool_module, tmp_path
):
    fake_repo = tmp_path / "fakerepo"
    target = fake_repo / "experiments" / "results" / "unrelated_dir"
    target.mkdir(parents=True)
    # Adjacent operator state — NO tool-owned manifest.
    (target / "some_data.json").write_text("{}")
    with pytest.raises(tool_module.UnsafeRmtreeRefusedError) as exc:
        tool_module._assert_rmtree_safe(target, repo_root=fake_repo)
    assert "tool-owned manifest" in str(exc.value)


def test_force_refuses_namespace_dir_with_unrelated_files(
    tool_module, tmp_path
):
    """If the dir has other files but no tool-owned manifest, refuse."""
    fake_repo = tmp_path / "fakerepo"
    target = fake_repo / "experiments" / "results" / "adjacent_work"
    target.mkdir(parents=True)
    (target / "results_from_another_run.json").write_text("{}")
    (target / "logs").mkdir()
    with pytest.raises(tool_module.UnsafeRmtreeRefusedError):
        tool_module._assert_rmtree_safe(target, repo_root=fake_repo)


# ─── Boundary: non-existent path is accepted (nothing to delete) ────────


def test_force_accepts_non_existent_namespace_path(tool_module, tmp_path):
    """If the path doesn't exist yet, the guard is a no-op (rmtree never fires)."""
    fake_repo = tmp_path / "fakerepo"
    fake_repo.mkdir()
    nonexistent = fake_repo / "experiments" / "results" / "future_run"
    # Should NOT raise. Caller's `if out_dir.exists()` short-circuits.
    tool_module._assert_rmtree_safe(nonexistent, repo_root=fake_repo)


# ─── Boundary: empty path resolves to repo root → refused ──────────────


def test_force_refuses_empty_path_resolves_to_cwd(tool_module, tmp_path, monkeypatch):
    """``Path('').resolve()`` resolves to cwd; refuse-by-default."""
    fake_repo = tmp_path / "fakerepo"
    fake_repo.mkdir()
    monkeypatch.chdir(fake_repo)
    # repo_path(Path('.')) → REPO_ROOT (fake). Should refuse.
    with pytest.raises(tool_module.UnsafeRmtreeRefusedError):
        tool_module._assert_rmtree_safe(fake_repo, repo_root=fake_repo)


# ─── Tool-owned manifest writer round-trip ─────────────────────────────


def test_write_tool_owned_manifest_creates_file(tool_module, tmp_path):
    target = tmp_path / "out"
    target.mkdir()
    manifest_path = tool_module._write_tool_owned_manifest(target)
    assert manifest_path.exists()
    payload = json.loads(manifest_path.read_text())
    assert payload["tool"] == tool_module.TOOL_NAME
    assert payload["schema"] == "build_pr101_nonlocal_sweep_manifest_v1"
    assert "Catalog #207" in payload["catalog_ref"]


def test_force_accepts_after_tool_owned_manifest_written(tool_module, tmp_path):
    """Round-trip: writing the manifest then refusing should succeed."""
    fake_repo = tmp_path / "fakerepo"
    target = fake_repo / "experiments" / "results" / "pr101_sweep"
    target.mkdir(parents=True)
    tool_module._write_tool_owned_manifest(target)
    tool_module._assert_rmtree_safe(target, repo_root=fake_repo)


# ─── Unresolvable path raises typed error ──────────────────────────────


def test_force_refuses_path_with_unresolvable_components(
    tool_module, tmp_path, monkeypatch
):
    """If Path.resolve() raises, the guard surfaces a typed refusal."""
    fake_repo = tmp_path / "fakerepo"
    fake_repo.mkdir()
    target = fake_repo / "experiments" / "results" / "x"

    class _BadPath:
        def resolve(self):
            raise OSError("simulated resolution failure")

    with pytest.raises(tool_module.UnsafeRmtreeRefusedError) as exc:
        tool_module._assert_rmtree_safe(_BadPath(), repo_root=fake_repo)
    assert "cannot resolve" in str(exc.value)


# ─── End-to-end: main() refuses --out-dir . --force ────────────────────


def test_main_refuses_force_with_repo_root_out_dir(tool_module, tmp_path, monkeypatch):
    """End-to-end: a typo --out-dir . --force on an existing repo root is refused."""
    fake_repo = tmp_path / "fakerepo"
    fake_repo.mkdir()
    monkeypatch.setattr(tool_module, "REPO_ROOT", fake_repo)
    monkeypatch.chdir(fake_repo)
    with pytest.raises(tool_module.UnsafeRmtreeRefusedError):
        tool_module._assert_rmtree_safe(fake_repo, repo_root=fake_repo)
