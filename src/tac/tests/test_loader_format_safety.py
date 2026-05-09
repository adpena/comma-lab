"""Lock-in tests for the 2026-04-26 DEN-V2 loader-format-safety bug class.

Bug: experiments/engineered_quant_noise.py imported `load_renderer` from
experiments/precompute_gradient_corrections.py. That loader did

    torch.load(path, weights_only=False)

unconditionally on a path that, in DEN-V2, was a renderer.bin in the FP4
binary format (magic `b'FP4A'`). torch.load tried to interpret the magic
bytes as ASCII and crashed with

    ValueError: could not convert string to float: 'P4AV'

after a full subprocess startup. The pipeline's soft-skip masked the failure
but every DEN-class deploy paid the cost.

These tests pin the permanent prevention so the bug never returns:

  1. load_renderer dispatches by content (first-4-bytes magic), NOT suffix.
  2. preflight catches any future consumer that would torch.load a path that
     might be a non-pickle binary export.
"""
from __future__ import annotations

import io
import struct
import sys
import tempfile
import textwrap
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
EXPERIMENTS_DIR = REPO_ROOT / "experiments"
if str(EXPERIMENTS_DIR.parent) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR.parent))


# ── load_renderer content-detection ───────────────────────────────────────


def _import_load_renderer():
    """Lazy import — keeps the test file importable in environments where
    experiments/ isn't on PYTHONPATH (e.g., wheel-only smoke runs)."""
    from experiments.precompute_gradient_corrections import (  # noqa: E402
        load_renderer,
    )
    return load_renderer


def _build_fp4_renderer_bytes(tmpdir: Path) -> Path:
    """Build a real FP4-format .bin file using the canonical export path.

    Uses a default-shape AsymmetricPairGenerator so build_renderer's
    PairGenerator-vs-AsymmetricPairGenerator dispatch matches the export
    side (mini configurations can drift on the rebuild path).
    """
    from tac.renderer import AsymmetricPairGenerator
    from tac.renderer_export import export_asymmetric_checkpoint_fp4

    model = AsymmetricPairGenerator(
        num_classes=5,
        embed_dim=6,
        base_ch=36,
        mid_ch=60,
        motion_hidden=32,
        depth=1,
        max_flow_px=20.0,
        max_residual=20.0,
        flow_only=False,
        pose_dim=0,
        use_dsconv=False,
    ).eval()
    out = tmpdir / "renderer.bin"
    export_asymmetric_checkpoint_fp4(model, out)
    return out


def _build_pytorch_pickle_checkpoint(tmpdir: Path) -> Path:
    """Build a legacy training-format .pt checkpoint (torch.save dict)."""
    from tac.renderer import AsymmetricPairGenerator

    model = AsymmetricPairGenerator(
        num_classes=5,
        embed_dim=6,
        base_ch=8,
        mid_ch=12,
        motion_hidden=8,
        depth=1,
        max_flow_px=20.0,
        max_residual=20.0,
        flow_only=False,
        pose_dim=0,
        use_dsconv=False,
    )
    out = tmpdir / "renderer_ckpt.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": {
            "num_classes": 5,
            "embed_dim": 6,
            "base_ch": 8,
            "mid_ch": 12,
            "motion_hidden": 8,
            "depth": 1,
            "max_flow_px": 20.0,
            "max_residual": 20.0,
            "flow_only": False,
            "pose_dim": 0,
            "use_dsconv": False,
        },
    }, str(out))
    return out


def test_load_renderer_dispatches_fp4_correctly(monkeypatch) -> None:
    """An FP4 .bin must dispatch to the FP4 loader — NOT crash in torch.load
    with the cryptic 'could not convert string to float: P4AV' DEN-V2 error.

    This test is the SCOPE of the DEN-V2 fix: dispatch correctness. It does
    NOT exercise FP4 round-trip integrity (separate concern, separate test
    suite — the FP4 export/load path has its own coverage in
    test_fp4_robustness). We monkey-patch the FP4 loader to a sentinel and
    assert it gets called when the magic is FP4A.
    """
    import experiments.precompute_gradient_corrections as pgc

    called = {"with_path": None}

    def _fake_loader(path, device="cpu"):
        called["with_path"] = Path(path) if not isinstance(path, Path) else path
        # Return a real (tiny) frozen module to satisfy the eval/freeze loop.
        m = torch.nn.Linear(2, 2)
        m.eval()
        for p in m.parameters():
            p.requires_grad = False
        return m

    # Patch on the renderer_export module since canonical loader looks it up
    # via load_any_renderer_checkpoint -> load_asymmetric_checkpoint_fp4.
    import tac.renderer_export as rx
    monkeypatch.setattr(rx, "load_asymmetric_checkpoint_fp4", _fake_loader)

    load_renderer = _import_load_renderer()
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        # Write a minimal FP4-magic file. The body doesn't need to be valid
        # FP4 — we patched the FP4 loader to a no-op.
        fp4_path = tmpdir / "renderer.bin"
        fp4_path.write_bytes(b"FP4A" + b"\x00" * 16)

        model = load_renderer(str(fp4_path), torch.device("cpu"))
        assert called["with_path"] is not None, (
            "load_renderer did NOT dispatch to load_asymmetric_checkpoint_fp4 "
            "on a file whose magic was b'FP4A'. This is the DEN-V2 bug "
            "(suffix-blind torch.load) coming back."
        )
        assert called["with_path"].name == "renderer.bin"
        assert not model.training
        assert all(not p.requires_grad for p in model.parameters())


def test_load_renderer_dispatches_fp4_via_real_export(tmp_path) -> None:
    """Smoke: even with a REAL FP4 file, the dispatch reaches the FP4 loader
    (and any error must come from the FP4 stack, NOT from torch.load on
    bad pickle bytes)."""
    fp4_path = _build_fp4_renderer_bytes(tmp_path)
    assert fp4_path.read_bytes()[:4] == b"FP4A"

    load_renderer = _import_load_renderer()
    # The FP4 stack has a known PairGenerator/AsymmetricPairGenerator
    # rebuild-shape divergence (separate bug); we just need to confirm
    # we never see the DEN-V2 ValueError.
    try:
        load_renderer(str(fp4_path), torch.device("cpu"))
    except RuntimeError as e:
        # Acceptable: any RuntimeError from inside the FP4 loader. NOT
        # acceptable: the DEN-V2 ValueError.
        assert "could not convert string to float" not in str(e)
    except ValueError as e:
        # The DEN-V2 bug is a ValueError. Hard-fail if we see it.
        assert "could not convert string to float" not in str(e), (
            "DEN-V2 regression: torch.load was called on an FP4 file"
        )


def test_load_renderer_dispatches_pytorch_correctly() -> None:
    """A torch.save .pt must load via the pickle path."""
    load_renderer = _import_load_renderer()

    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        pt_path = _build_pytorch_pickle_checkpoint(tmpdir)
        # Sanity: file starts with PK\x03\x04 (zip-based torch.save container).
        assert pt_path.read_bytes()[:4] == b"PK\x03\x04"

        model = load_renderer(str(pt_path), torch.device("cpu"))
        assert model.__class__.__name__ == "AsymmetricPairGenerator"
        assert not model.training
        assert all(not p.requires_grad for p in model.parameters())


def test_load_renderer_raises_on_unknown_magic() -> None:
    """An unknown 4-byte header must raise a clear RuntimeError that names
    both the magic seen and the accepted formats — never crash with the
    DEN-V2 'could not convert string to float' message."""
    load_renderer = _import_load_renderer()

    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        bogus = tmpdir / "garbage.bin"
        bogus.write_bytes(b"XXXX" + b"\x00" * 64)

        with pytest.raises(RuntimeError) as exc:
            load_renderer(str(bogus), torch.device("cpu"))
        msg = str(exc.value)
        # Must name the magic seen
        assert "XXXX" in msg or "b'XXXX'" in msg or "b\"XXXX\"" in msg
        # Must enumerate the accepted formats
        assert "FP4A" in msg
        assert "ASYM" in msg
        assert "DPSM" in msg
        # Must NOT be the cryptic torch.load error
        assert "could not convert string to float" not in msg


def test_load_renderer_raises_on_too_short_file() -> None:
    """A file with fewer than 4 bytes must raise — not bubble an IndexError."""
    load_renderer = _import_load_renderer()

    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        short = tmpdir / "short.bin"
        short.write_bytes(b"AB")  # only 2 bytes

        with pytest.raises(RuntimeError) as exc:
            load_renderer(str(short), torch.device("cpu"))
        assert "too short" in str(exc.value).lower() or "magic" in str(exc.value).lower()


def test_load_renderer_raises_on_missing_file() -> None:
    load_renderer = _import_load_renderer()
    with pytest.raises(FileNotFoundError):
        load_renderer("/nonexistent/path/to/renderer.bin", torch.device("cpu"))


# ── preflight_loader_format_safety ────────────────────────────────────────


def _scan(tmp_root: Path):
    from tac.preflight import preflight_loader_format_safety
    return preflight_loader_format_safety(
        repo_root=tmp_root,
        scan_dirs=["experiments"],
        strict=False,
        verbose=False,
    )


def test_preflight_catches_unsafe_loader_function() -> None:
    """A `def load_renderer` that calls bare torch.load with no magic
    dispatch is the original DEN-V2 sin and must be flagged."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "experiments").mkdir()
        (root / "experiments" / "bad_loader.py").write_text(textwrap.dedent('''
            """Unsafe loader — original DEN-V2 anti-pattern."""
            import torch

            def load_renderer(checkpoint_path, device):
                ckpt = torch.load(checkpoint_path, map_location="cpu",
                                  weights_only=False)
                return ckpt
        ''').lstrip())

        violations = _scan(root)
        assert violations, (
            "Expected loader-format-safety violation but got none. "
            "The unsafe `load_renderer` function went undetected."
        )
        # Must call out the function and mention the bug pattern
        msg = "\n".join(violations)
        assert "load_renderer" in msg
        assert "DEN-V2" in msg or "FP4A" in msg


def test_preflight_catches_unsafe_consumer_call_site() -> None:
    """A bare `torch.load(<renderer-like>, weights_only=False)` outside a
    content-detecting function must also be flagged, even if no
    `def load_renderer` exists. The pattern is intentionally narrow: only
    renderer-named variables and only weights_only=False (DEN-V2's exact
    failure mode)."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "experiments").mkdir()
        (root / "experiments" / "bad_consumer.py").write_text(textwrap.dedent('''
            """Unsafe consumer — bare torch.load on a renderer variable."""
            import torch

            def main(renderer_path):
                state = torch.load(renderer_path, map_location="cpu",
                                   weights_only=False)
                print(state)
        ''').lstrip())

        violations = _scan(root)
        assert violations, "Expected violation on bare torch.load(renderer_path)"
        assert "torch.load" in "\n".join(violations)


def test_preflight_loader_format_cache_invalidates_changed_file() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        experiments = root / "experiments"
        experiments.mkdir()
        target = experiments / "cached_loader.py"
        target.write_text(textwrap.dedent('''
            import torch

            def helper():
                return 1
        ''').lstrip())
        assert _scan(root) == []
        assert (root / ".omx" / "cache" / "loader_format_safety_clean.json").exists()

        target.write_text(textwrap.dedent('''
            import torch

            def load_renderer(renderer_path):
                return torch.load(renderer_path, weights_only=False)
        ''').lstrip())
        violations = _scan(root)
        assert violations
        assert "load_renderer" in "\n".join(violations)


def test_preflight_ignores_unrelated_torch_load() -> None:
    """A torch.load on a non-renderer variable (e.g., TTO batch checkpoint)
    must NOT be flagged. The DEN-V2 bug is renderer-specific; flagging every
    torch.load would block legitimate use (TTO resume, optimizer state, etc.)
    and the rule would be disabled."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "experiments").mkdir()
        (root / "experiments" / "tto_resume.py").write_text(textwrap.dedent('''
            """TTO batch checkpoint resume — NOT a renderer load."""
            import torch

            def resume(batch_idx, output_dir):
                ckpt_path = output_dir / f"tto_batch_{batch_idx}.pt"
                if ckpt_path.exists():
                    return torch.load(ckpt_path, map_location="cpu",
                                      weights_only=True)
        ''').lstrip())

        violations = _scan(root)
        assert not violations, (
            f"Expected zero violations for TTO batch resume, got: {violations}"
        )


def test_preflight_passes_on_safe_loader_with_magic_dispatch() -> None:
    """A loader that reads the first 4 bytes and branches on FP4A/ASYM/PK
    must NOT be flagged."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "experiments").mkdir()
        (root / "experiments" / "safe_loader.py").write_text(textwrap.dedent('''
            """Safe loader — content-detects the format before dispatching."""
            import torch

            def load_renderer(checkpoint_path, device):
                with open(checkpoint_path, "rb") as fh:
                    magic = fh.read(4)
                if magic == b"FP4A":
                    return _load_fp4(checkpoint_path, device)
                if magic == b"ASYM":
                    return _load_asym(checkpoint_path, device)
                if magic == b"DPSM":
                    return _load_dpsm(checkpoint_path, device)
                if magic.startswith(b"PK\\x03\\x04"):
                    return torch.load(checkpoint_path, map_location="cpu")
                raise RuntimeError(f"Unknown magic: {magic!r}")

            def _load_fp4(p, d): ...
            def _load_asym(p, d): ...
            def _load_dpsm(p, d): ...
        ''').lstrip())

        violations = _scan(root)
        assert not violations, (
            f"Expected zero violations for safe loader, got: {violations}"
        )


def test_preflight_passes_on_loader_that_delegates() -> None:
    """A loader that delegates to the canonical safe loader must NOT be
    flagged — this is the recommended fix pattern."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "experiments").mkdir()
        (root / "experiments" / "delegating_loader.py").write_text(textwrap.dedent('''
            """Safe — delegates to the canonical content-detecting loader."""

            def load_renderer(checkpoint_path, device):
                from experiments.precompute_gradient_corrections import (
                    load_renderer as _canonical,
                )
                return _canonical(checkpoint_path, device)
        ''').lstrip())

        violations = _scan(root)
        assert not violations, (
            f"Expected zero violations for delegating loader, got: {violations}"
        )


def test_preflight_skips_test_files() -> None:
    """Test/smoke files are allowed to construct intentionally-wrong torch.load
    inputs (since the whole point of a test is to exercise edge cases)."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "experiments").mkdir()
        (root / "experiments" / "test_something.py").write_text(textwrap.dedent('''
            import torch

            def test_thing():
                ckpt = torch.load("checkpoint.bin")
                assert ckpt is not None
        ''').lstrip())

        violations = _scan(root)
        # No violation — file matches the test_/tests/ exemption.
        assert not violations


def test_preflight_real_repo_passes() -> None:
    """The real repo's experiments/ tree must currently pass preflight.

    If this test fails, it means a NEW unsafe loader was introduced — fix
    the producer, not the test.
    """
    from tac.preflight import preflight_loader_format_safety
    violations = preflight_loader_format_safety(
        repo_root=REPO_ROOT,
        scan_dirs=["experiments", "src/tac"],
        strict=False,
        verbose=False,
    )
    assert not violations, (
        "Repo has loader-format-safety violations — the DEN-V2 bug class is "
        "back. Violations:\n" + "\n".join(f"  - {v}" for v in violations)
    )
