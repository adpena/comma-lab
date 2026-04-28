"""Tests for preflight checks 44, 45, 46, 47 (assertion-strength meta-bugs).

Reference: Round 22 bit-STE sign-bug post-mortem. Council reviews 12/13/14/18
all dismissed the bug because the only assertion on `bits.grad` was
`assert bits.grad is not None`. Round 21 caught it via a hand-derived
numeric value. CLAUDE.md anti-arbitrariness: gradient / loss / quantizer /
archive tests must pin a number, a sign, a comparison, or a size — never
just finiteness.

Per-check coverage (≥3 tests each):
  Check 44 (gradient-direction-tests-exist):
    - positive: a test asserting `grad is not None` ONLY is FLAGGED
    - positive: a test asserting `pytest.approx(...)` is ACCEPTED
    - waiver: same-line `# GRADIENT_DIRECTION_NOT_REQUIRED:` is ACCEPTED
  Check 45 (loss-convergence-tests):
    - positive: a `test_*_loss.py` with no .step() / approx is FLAGGED
    - positive: a file with `loss_after < loss_before` is ACCEPTED
    - boundary: `test_*lossless*.py` is NOT scanned (token boundary)
    - waiver: in-file `# LOSS_CONVERGENCE_NOT_REQUIRED:` is ACCEPTED
  Check 46 (quantizer-roundtrip-tests):
    - positive: a *quant*.py module with no test file is FLAGGED
    - positive: a *quant*.py with `torch.allclose(decode(encode(x)), x)` test ACCEPTED
    - waiver: in-module `# ROUNDTRIP_NOT_REQUIRED:` is ACCEPTED
  Check 47 (lane-archive-size-assertion):
    - positive: a remote_lane_*.sh that builds an archive + invokes auth eval
      WITHOUT any size assertion is FLAGGED
    - positive: same script with `ARCHIVE_BYTES=$(stat ...) && [ ... -gt 0 ]` ACCEPTED
    - waiver: same-line `# ARCHIVE_SIZE_NOT_REQUIRED:` is ACCEPTED
"""
from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from tac.preflight import (
    MetaBugViolation,
    _scan_quantizer_for_roundtrip_test,
    _scan_remote_lane_for_archive_size_assertion,
    _scan_test_file_for_grad_direction,
    _scan_test_file_for_loss_convergence,
    check_gradient_direction_tests_exist,
    check_lane_deploy_scripts_have_archive_size_assertion,
    check_quantizer_modules_have_round_trip_test,
    check_test_assertion_strength_for_loss_functions,
)


def _write(p: Path, body: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(textwrap.dedent(body).lstrip("\n"))


def _stub_repo(tmp_path: Path) -> Path:
    (tmp_path / "src" / "tac" / "tests").mkdir(parents=True, exist_ok=True)
    (tmp_path / "scripts").mkdir(parents=True, exist_ok=True)
    return tmp_path


# ─── Check 44: gradient-direction tests ───────────────────────────────────────


class TestGradientDirectionTestsExist:
    def _stub_quant_module(self, root: Path) -> None:
        """Drop a fake STE class so the scanner registers the symbol."""
        _write(root / "src" / "tac" / "quantization.py", """
            import torch
            class FakeSTE(torch.autograd.Function):
                @staticmethod
                def forward(ctx, x):
                    return x.round()
                @staticmethod
                def backward(ctx, g):
                    return g
        """)

    def test_isnotnone_only_is_flagged(self, tmp_path: Path) -> None:
        root = _stub_repo(tmp_path)
        self._stub_quant_module(root)
        tf = root / "src" / "tac" / "tests" / "test_bad.py"
        _write(tf, """
            import torch
            from tac.quantization import FakeSTE
            def test_grad_isnotnone():
                w = torch.randn(3, requires_grad=True)
                q = FakeSTE.apply(w)
                q.sum().backward()
                assert w.grad is not None
                assert w.grad.shape == w.shape
        """)
        v = _scan_test_file_for_grad_direction(tf, root)
        assert len(v) == 1, v
        assert "test_grad_isnotnone" in v[0]
        assert "Round 22" in v[0]

    def test_pytest_approx_is_accepted(self, tmp_path: Path) -> None:
        root = _stub_repo(tmp_path)
        self._stub_quant_module(root)
        tf = root / "src" / "tac" / "tests" / "test_good.py"
        _write(tf, """
            import torch
            import pytest
            from tac.quantization import FakeSTE
            def test_grad_value():
                w = torch.tensor([0.5], requires_grad=True)
                q = FakeSTE.apply(w)
                q.sum().backward()
                assert w.grad is not None
                assert w.grad.item() == pytest.approx(1.0)
        """)
        v = _scan_test_file_for_grad_direction(tf, root)
        assert v == [], v

    def test_allclose_is_accepted(self, tmp_path: Path) -> None:
        root = _stub_repo(tmp_path)
        self._stub_quant_module(root)
        tf = root / "src" / "tac" / "tests" / "test_allclose.py"
        _write(tf, """
            import torch
            from tac.quantization import FakeSTE
            def test_grad_allclose():
                w = torch.randn(3, requires_grad=True)
                q = FakeSTE.apply(w)
                q.sum().backward()
                assert torch.allclose(w.grad, torch.ones_like(w))
        """)
        v = _scan_test_file_for_grad_direction(tf, root)
        assert v == [], v

    def test_sign_assertion_is_accepted(self, tmp_path: Path) -> None:
        root = _stub_repo(tmp_path)
        self._stub_quant_module(root)
        tf = root / "src" / "tac" / "tests" / "test_sign.py"
        _write(tf, """
            import torch
            from tac.quantization import FakeSTE
            def test_grad_sign():
                w = torch.tensor([0.5], requires_grad=True)
                q = FakeSTE.apply(w)
                (q * -1.0).sum().backward()
                assert w.grad.item() < 0
        """)
        v = _scan_test_file_for_grad_direction(tf, root)
        assert v == [], v

    def test_indexed_grad_check_is_accepted(self, tmp_path: Path) -> None:
        root = _stub_repo(tmp_path)
        self._stub_quant_module(root)
        tf = root / "src" / "tac" / "tests" / "test_indexed.py"
        _write(tf, """
            import torch
            from tac.quantization import FakeSTE
            def test_grad_indexed():
                w = torch.tensor([0.5, -0.5], requires_grad=True)
                q = FakeSTE.apply(w)
                q.sum().backward()
                assert w.grad[0].item() == 1.0
                assert w.grad[1].item() == 1.0
        """)
        v = _scan_test_file_for_grad_direction(tf, root)
        assert v == [], v

    def test_magnitude_check_is_accepted(self, tmp_path: Path) -> None:
        """`abs(grad.item()) < 1e-3` should satisfy the gate."""
        root = _stub_repo(tmp_path)
        self._stub_quant_module(root)
        tf = root / "src" / "tac" / "tests" / "test_mag.py"
        _write(tf, """
            import torch
            from tac.quantization import FakeSTE
            def test_grad_magnitude_small():
                w = torch.tensor([0.5], requires_grad=True)
                q = FakeSTE.apply(w)
                (q * 0.0).sum().backward()
                assert w.grad is not None
                assert abs(w.grad.item()) < 1e-3
        """)
        v = _scan_test_file_for_grad_direction(tf, root)
        assert v == [], v

    def test_waiver_on_def_line_is_accepted(self, tmp_path: Path) -> None:
        root = _stub_repo(tmp_path)
        self._stub_quant_module(root)
        tf = root / "src" / "tac" / "tests" / "test_waived.py"
        _write(tf, """
            import torch
            from tac.quantization import FakeSTE
            def test_grad_waived():  # GRADIENT_DIRECTION_NOT_REQUIRED: smoke-only
                w = torch.randn(3, requires_grad=True)
                q = FakeSTE.apply(w)
                q.sum().backward()
                assert w.grad is not None
        """)
        v = _scan_test_file_for_grad_direction(tf, root)
        assert v == [], v

    def test_no_autograd_reference_is_skipped(self, tmp_path: Path) -> None:
        """Tests that don't touch autograd.Function are not subject to the gate."""
        root = _stub_repo(tmp_path)
        self._stub_quant_module(root)
        tf = root / "src" / "tac" / "tests" / "test_unrelated.py"
        _write(tf, """
            import torch
            def test_no_grad():
                x = torch.tensor([1.0])
                assert x.shape == (1,)
        """)
        v = _scan_test_file_for_grad_direction(tf, root)
        assert v == [], v

    def test_check_strict_raises(self, tmp_path: Path) -> None:
        root = _stub_repo(tmp_path)
        self._stub_quant_module(root)
        _write(root / "src" / "tac" / "tests" / "test_bad.py", """
            import torch
            from tac.quantization import FakeSTE
            def test_grad_isnotnone():
                w = torch.randn(3, requires_grad=True)
                q = FakeSTE.apply(w)
                q.sum().backward()
                assert w.grad is not None
        """)
        with pytest.raises(MetaBugViolation):
            check_gradient_direction_tests_exist(
                repo_root=root, strict=True, verbose=False
            )

    def test_check_warn_only_returns_list(self, tmp_path: Path) -> None:
        root = _stub_repo(tmp_path)
        self._stub_quant_module(root)
        _write(root / "src" / "tac" / "tests" / "test_bad.py", """
            import torch
            from tac.quantization import FakeSTE
            def test_grad_isnotnone():
                w = torch.randn(3, requires_grad=True)
                q = FakeSTE.apply(w)
                q.sum().backward()
                assert w.grad is not None
        """)
        v = check_gradient_direction_tests_exist(
            repo_root=root, strict=False, verbose=False
        )
        assert len(v) >= 1


# ─── Check 45: loss-convergence tests ─────────────────────────────────────────


class TestTestAssertionStrengthForLossFunctions:
    def test_loss_test_with_no_convergence_check_is_flagged(
        self, tmp_path: Path
    ) -> None:
        root = _stub_repo(tmp_path)
        tf = root / "src" / "tac" / "tests" / "test_my_loss.py"
        _write(tf, """
            import torch
            def test_my_loss_is_finite():
                x = torch.tensor([1.0])
                loss = (x ** 2).mean()
                assert torch.isfinite(loss)
                assert loss.shape == ()
        """)
        v = _scan_test_file_for_loss_convergence(tf, root)
        assert len(v) == 1, v
        assert "convergence" in v[0]

    def test_loss_test_with_loss_decrease_is_accepted(
        self, tmp_path: Path
    ) -> None:
        root = _stub_repo(tmp_path)
        tf = root / "src" / "tac" / "tests" / "test_my_loss.py"
        _write(tf, """
            import torch
            def test_my_loss_decreases():
                x = torch.tensor([2.0], requires_grad=True)
                loss_before = (x ** 2).sum()
                loss_before.backward()
                with torch.no_grad():
                    x.sub_(0.1 * x.grad)
                loss_after = (x ** 2).sum()
                assert loss_after < loss_before
        """)
        v = _scan_test_file_for_loss_convergence(tf, root)
        assert v == [], v

    def test_loss_test_with_optim_step_is_accepted(self, tmp_path: Path) -> None:
        root = _stub_repo(tmp_path)
        tf = root / "src" / "tac" / "tests" / "test_optim_loss.py"
        _write(tf, """
            import torch
            def test_my_loss_via_optim():
                x = torch.tensor([2.0], requires_grad=True)
                opt = torch.optim.SGD([x], lr=0.1)
                opt.step()
        """)
        v = _scan_test_file_for_loss_convergence(tf, root)
        assert v == [], v

    def test_lossless_token_boundary_is_skipped(self, tmp_path: Path) -> None:
        """`test_lossless_*.py` is NOT a loss-function test."""
        root = _stub_repo(tmp_path)
        tf = root / "src" / "tac" / "tests" / "test_lossless_codec.py"
        _write(tf, """
            def test_lossless_round_trip():
                pass
        """)
        v = _scan_test_file_for_loss_convergence(tf, root)
        assert v == [], v

    def test_waiver_in_file_is_accepted(self, tmp_path: Path) -> None:
        root = _stub_repo(tmp_path)
        tf = root / "src" / "tac" / "tests" / "test_my_loss.py"
        _write(tf, """
            # LOSS_CONVERGENCE_NOT_REQUIRED: smoke-only forward sanity test
            import torch
            def test_my_loss_is_finite():
                x = torch.tensor([1.0])
                loss = (x ** 2).mean()
                assert torch.isfinite(loss)
        """)
        v = _scan_test_file_for_loss_convergence(tf, root)
        assert v == [], v

    def test_check_strict_raises(self, tmp_path: Path) -> None:
        root = _stub_repo(tmp_path)
        _write(root / "src" / "tac" / "tests" / "test_my_loss.py", """
            import torch
            def test_my_loss_is_finite():
                loss = torch.tensor(1.0)
                assert torch.isfinite(loss)
        """)
        with pytest.raises(MetaBugViolation):
            check_test_assertion_strength_for_loss_functions(
                repo_root=root, strict=True, verbose=False
            )


# ─── Check 46: quantizer-roundtrip tests ──────────────────────────────────────


class TestQuantizerModulesHaveRoundTripTest:
    def test_quant_module_with_no_test_file_is_flagged(
        self, tmp_path: Path
    ) -> None:
        root = _stub_repo(tmp_path)
        mod = root / "src" / "tac" / "fancy_quant.py"
        _write(mod, """
            def quantize(x):
                return x.round()
            def dequantize(x):
                return x
        """)
        v = _scan_quantizer_for_roundtrip_test(mod, root)
        assert len(v) == 1, v
        assert "no test file" in v[0]

    def test_quant_module_with_roundtrip_test_is_accepted(
        self, tmp_path: Path
    ) -> None:
        root = _stub_repo(tmp_path)
        mod = root / "src" / "tac" / "fancy_quant.py"
        _write(mod, """
            def quantize(x):
                return x.round()
            def dequantize(x):
                return x
        """)
        tf = root / "src" / "tac" / "tests" / "test_fancy_quant.py"
        _write(tf, """
            import torch
            from tac.fancy_quant import quantize, dequantize
            def test_roundtrip():
                x = torch.tensor([0.4, 0.6])
                assert torch.allclose(dequantize(quantize(x)), x.round())
        """)
        v = _scan_quantizer_for_roundtrip_test(mod, root)
        assert v == [], v

    def test_quant_module_with_test_no_roundtrip_is_flagged(
        self, tmp_path: Path
    ) -> None:
        root = _stub_repo(tmp_path)
        mod = root / "src" / "tac" / "fancy_quant.py"
        _write(mod, """
            def quantize(x):
                return x.round()
        """)
        tf = root / "src" / "tac" / "tests" / "test_fancy_quant.py"
        _write(tf, """
            import torch
            from tac.fancy_quant import quantize
            def test_quantize_finite():
                x = torch.tensor([0.4])
                assert torch.isfinite(quantize(x)).all()
        """)
        v = _scan_quantizer_for_roundtrip_test(mod, root)
        assert len(v) == 1, v
        assert "roundtrip" in v[0].lower()

    def test_waiver_in_module_is_accepted(self, tmp_path: Path) -> None:
        root = _stub_repo(tmp_path)
        mod = root / "src" / "tac" / "fancy_quant.py"
        _write(mod, """
            # ROUNDTRIP_NOT_REQUIRED: lossy by design (audit-only module)
            def quantize(x):
                return x.round()
        """)
        v = _scan_quantizer_for_roundtrip_test(mod, root)
        assert v == [], v

    def test_check_warn_only_returns_list(self, tmp_path: Path) -> None:
        root = _stub_repo(tmp_path)
        _write(root / "src" / "tac" / "fancy_quant.py", """
            def quantize(x):
                return x.round()
        """)
        v = check_quantizer_modules_have_round_trip_test(
            repo_root=root, strict=False, verbose=False
        )
        assert len(v) >= 1


# ─── Check 47: lane archive-size assertion ────────────────────────────────────


class TestLaneDeployScriptsHaveArchiveSizeAssertion:
    def test_lane_with_no_size_assertion_is_flagged(
        self, tmp_path: Path
    ) -> None:
        root = _stub_repo(tmp_path)
        sh = root / "scripts" / "remote_lane_x_test.sh"
        _write(sh, """
            #!/usr/bin/env bash
            set -euo pipefail
            python -c "
            import zipfile
            with zipfile.ZipFile('out.zip', 'w') as z:
                z.write('renderer.bin')
            "
            python experiments/contest_auth_eval.py --archive out.zip
        """)
        v = _scan_remote_lane_for_archive_size_assertion(sh, root)
        assert len(v) == 1, v
        assert "size" in v[0].lower()

    def test_lane_with_stat_assertion_is_accepted(self, tmp_path: Path) -> None:
        root = _stub_repo(tmp_path)
        sh = root / "scripts" / "remote_lane_x_test.sh"
        _write(sh, """
            #!/usr/bin/env bash
            set -euo pipefail
            python -c "
            import zipfile
            with zipfile.ZipFile('out.zip', 'w') as z:
                z.write('renderer.bin')
            "
            ARCHIVE_BYTES=$(stat -c '%s' out.zip 2>/dev/null || stat -f '%z' out.zip)
            [ "$ARCHIVE_BYTES" -gt 0 ] || { echo FATAL; exit 2; }
            python experiments/contest_auth_eval.py --archive out.zip
        """)
        v = _scan_remote_lane_for_archive_size_assertion(sh, root)
        assert v == [], v

    def test_lane_with_python_size_assert_is_accepted(
        self, tmp_path: Path
    ) -> None:
        root = _stub_repo(tmp_path)
        sh = root / "scripts" / "remote_lane_x_test.sh"
        _write(sh, """
            #!/usr/bin/env bash
            set -euo pipefail
            python -c "
            import os, zipfile
            with zipfile.ZipFile('out.zip', 'w') as z:
                z.write('renderer.bin')
            assert os.path.getsize('out.zip') > 0, 'empty archive'
            "
            python experiments/contest_auth_eval.py --archive out.zip
        """)
        v = _scan_remote_lane_for_archive_size_assertion(sh, root)
        assert v == [], v

    def test_lane_no_archive_build_is_skipped(self, tmp_path: Path) -> None:
        """Scripts that consume an existing archive but don't build one
        are exempt — the producing script owns the size assertion."""
        root = _stub_repo(tmp_path)
        sh = root / "scripts" / "remote_lane_eval_only.sh"
        _write(sh, """
            #!/usr/bin/env bash
            set -euo pipefail
            python experiments/contest_auth_eval.py --archive existing.zip
        """)
        v = _scan_remote_lane_for_archive_size_assertion(sh, root)
        assert v == [], v

    def test_waiver_in_file_is_accepted(self, tmp_path: Path) -> None:
        root = _stub_repo(tmp_path)
        sh = root / "scripts" / "remote_lane_x_test.sh"
        _write(sh, """
            #!/usr/bin/env bash
            # ARCHIVE_SIZE_NOT_REQUIRED: dev-only smoke script, never submitted
            set -euo pipefail
            python -c "
            import zipfile
            with zipfile.ZipFile('out.zip', 'w') as z:
                z.write('renderer.bin')
            "
            python experiments/contest_auth_eval.py --archive out.zip
        """)
        v = _scan_remote_lane_for_archive_size_assertion(sh, root)
        assert v == [], v

    def test_check_strict_raises(self, tmp_path: Path) -> None:
        root = _stub_repo(tmp_path)
        _write(root / "scripts" / "remote_lane_x_test.sh", """
            #!/usr/bin/env bash
            set -euo pipefail
            python -c "import zipfile; zipfile.ZipFile('out.zip','w').write('x')"
            python experiments/contest_auth_eval.py --archive out.zip
        """)
        with pytest.raises(MetaBugViolation):
            check_lane_deploy_scripts_have_archive_size_assertion(
                repo_root=root, strict=True, verbose=False
            )

    def test_check_warn_only_returns_list(self, tmp_path: Path) -> None:
        root = _stub_repo(tmp_path)
        _write(root / "scripts" / "remote_lane_x_test.sh", """
            #!/usr/bin/env bash
            set -euo pipefail
            python -c "import zipfile; zipfile.ZipFile('out.zip','w').write('x')"
            python experiments/contest_auth_eval.py --archive out.zip
        """)
        v = check_lane_deploy_scripts_have_archive_size_assertion(
            repo_root=root, strict=False, verbose=False
        )
        assert len(v) >= 1
