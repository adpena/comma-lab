# SPDX-License-Identifier: MIT
"""Council R3 regression: verify train_renderer's auth-eval-on-best wiring is REAL.

The previous wiring (R2) was dead code:
  - Invented `--auth-eval-masks` flag for auth_eval_renderer that doesn't exist
  - Skipped --archive-size-bytes → rate computed from renderer-only (~290KB)
    instead of triple-joint archive (~700KB), systematically optimistic
  - Silently no-op'd if --auth-eval-masks wasn't passed (which was every chain
    because the flag was never plumbed through any caller)

These tests verify the R3 fix actually:
  A. Reads auth_eval_renderer.py's REAL argparse args (no invented flags)
  B. Builds a REAL archive when masks + poses provided
  C. Passes --archive-size-bytes from the BUILT archive (not renderer-only)
  D. Doesn't silently skip when masks/poses missing (logs WARN, runs anyway)
"""
from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]


@pytest.fixture(scope="module")
def auth_eval_argparse() -> set[str]:
    """The REAL set of --flag-names that auth_eval_renderer.py accepts.

    Computed by introspecting the file (not by guessing). Any wiring that
    passes a flag NOT in this set is dead code that argparse will reject
    or silently ignore.
    """
    src = (REPO / "experiments" / "auth_eval_renderer.py").read_text()
    # Match `parser.add_argument("--name", ...)` and `parser.add_argument( "--name", ...)`
    flags = set(re.findall(r'parser\.add_argument\(\s*"(--[a-z0-9_-]+)"', src))
    assert flags, "no flags found in auth_eval_renderer.py — regex bug"
    return flags


def test_auth_eval_renderer_has_no_masks_flag(auth_eval_argparse):
    """Documents the bug: there is NO --masks or --auth-eval-masks flag.
    Any wiring that passes either is dead code."""
    assert "--masks" not in auth_eval_argparse
    assert "--auth-eval-masks" not in auth_eval_argparse


def test_auth_eval_renderer_required_flags(auth_eval_argparse):
    """Affirmative test: these are the flags train_renderer MUST use."""
    must_exist = {"--checkpoint", "--upstream-dir", "--device", "--poses",
                  "--archive-size-bytes", "--output-dir"}
    missing = must_exist - auth_eval_argparse
    assert not missing, (
        f"auth_eval_renderer.py is missing flags train_renderer expects: {missing}"
    )


def _load_auth_eval_renderer_module():
    module_name = "_auth_eval_renderer_axis_test"
    sys.modules.pop(module_name, None)
    spec = importlib.util.spec_from_file_location(
        module_name,
        REPO / "experiments" / "auth_eval_renderer.py",
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_auth_eval_renderer_reports_cuda_to_cpu_fallback_axis(monkeypatch):
    """A CUDA request that resolves to CPU must expose advisory evidence."""
    mod = _load_auth_eval_renderer_module()
    monkeypatch.setattr(mod.torch.cuda, "is_available", lambda: False)

    fields = mod.resolve_requested_actual_device("cuda")

    assert fields["requested_device"] == "cuda"
    assert fields["actual_device"] == "cpu"
    assert fields["device_fallback_occurred"] is True
    assert fields["fallback_occurred"] is True
    assert fields["device_fallback_reason"] == "cuda_unavailable"
    assert fields["evidence_axis"] != "contest_cuda"
    assert str(fields["evidence_axis"]).endswith("_cpu_advisory")


def test_train_renderer_rejects_cuda_auth_eval_cpu_fallback_result():
    sys.path.insert(0, str(REPO / "src" / "tac" / "experiments"))
    if "train_renderer" in sys.modules:
        del sys.modules["train_renderer"]
    from train_renderer import _enforce_auth_eval_axis_contract  # noqa: E402

    payload = {
        "requested_device": "cuda",
        "actual_device": "cpu",
        "evidence_axis": "linux_cpu_advisory",
        "device_fallback_occurred": True,
    }

    with pytest.raises(RuntimeError, match="evidence axis rejected"):
        _enforce_auth_eval_axis_contract(
            payload,
            requested_device="cuda",
            allowed_axes={"contest_cuda"},
        )


def test_train_renderer_rejects_inconsistent_contest_cuda_from_cpu_fallback():
    sys.path.insert(0, str(REPO / "src" / "tac" / "experiments"))
    if "train_renderer" in sys.modules:
        del sys.modules["train_renderer"]
    from train_renderer import _enforce_auth_eval_axis_contract  # noqa: E402

    payload = {
        "requested_device": "cuda",
        "actual_device": "cpu",
        "evidence_axis": "contest_cuda",
        "device_fallback_occurred": True,
    }

    with pytest.raises(RuntimeError, match="actual_device='cpu'"):
        _enforce_auth_eval_axis_contract(
            payload,
            requested_device="cuda",
            allowed_axes={"contest_cuda", "linux_cpu_advisory"},
        )


def test_train_renderer_can_explicitly_permit_advisory_axis():
    sys.path.insert(0, str(REPO / "src" / "tac" / "experiments"))
    if "train_renderer" in sys.modules:
        del sys.modules["train_renderer"]
    from train_renderer import _enforce_auth_eval_axis_contract  # noqa: E402

    payload = {
        "requested_device": "cuda",
        "actual_device": "cpu",
        "evidence_axis": "linux_cpu_advisory",
        "device_fallback_occurred": True,
    }

    assert _enforce_auth_eval_axis_contract(
        payload,
        requested_device="cuda",
        allowed_axes={"contest_cuda", "linux_cpu_advisory"},
    ) is payload


def test_train_renderer_does_not_pass_invented_flags():
    """train_renderer.py's auth-eval-on-best block must NOT emit any flag
    that auth_eval_renderer.py doesn't accept. This catches the dead-flag
    wiring that R3 found."""
    src = (REPO / "src" / "tac" / "experiments" / "train_renderer.py").read_text()
    # Find the auth-eval block — between "# Auth eval on best" and "return best_scorer"
    m = re.search(r"# Auth eval on best.*?return best_scorer", src, re.DOTALL)
    assert m, "auth-eval-on-best block not found"
    block = m.group(0)

    # Get every "--flag" used in the block as a string literal
    flags_in_block = set(re.findall(r'"(--[a-z0-9_-]+)"', block))

    # Get the real auth_eval_renderer flags
    auth_src = (REPO / "experiments" / "auth_eval_renderer.py").read_text()
    real_flags = set(re.findall(r'parser\.add_argument\(\s*"(--[a-z0-9_-]+)"', auth_src))

    invented = flags_in_block - real_flags
    assert not invented, (
        f"train_renderer's auth-eval block uses flags that don't exist in "
        f"auth_eval_renderer.py argparse: {invented}. This is dead code that "
        f"argparse would reject."
    )


def test_train_renderer_passes_archive_size_bytes():
    """Council R3-2: the subprocess MUST pass --archive-size-bytes,
    otherwise rate is computed from renderer-only (~290KB) which is
    optimistic vs the real triple-joint archive (~700KB)."""
    src = (REPO / "src" / "tac" / "experiments" / "train_renderer.py").read_text()
    m = re.search(r"# Auth eval on best.*?return best_scorer", src, re.DOTALL)
    assert m
    block = m.group(0)
    assert '"--archive-size-bytes"' in block, (
        "train_renderer must pass --archive-size-bytes to auth_eval_renderer, "
        "otherwise the rate term is systematically optimistic by ~2x. "
        "This is Council R3-2."
    )


def test_train_renderer_builds_real_archive():
    """The fix mandate: when masks + poses are provided, the wiring
    must call build_submission_archive (NOT just pass renderer.bin
    bytes as the 'archive size')."""
    src = (REPO / "src" / "tac" / "experiments" / "train_renderer.py").read_text()
    m = re.search(r"# Auth eval on best.*?return best_scorer", src, re.DOTALL)
    assert m
    block = m.group(0)
    assert "build_submission_archive" in block, (
        "auth-eval-on-best must use build_submission_archive when both "
        "masks + poses are provided. Otherwise the archive size passed "
        "to auth_eval_renderer is renderer-only and the rate is biased."
    )


def test_train_renderer_does_not_silently_skip():
    """Council R3-3: previous R2 wiring SKIPPED the eval entirely if
    --auth-eval-masks wasn't passed, meaning every existing chain (none
    of which pass the flag) silently no-op'd. The fix must run the eval
    anyway, with a WARN about the rate bias."""
    src = (REPO / "src" / "tac" / "experiments" / "train_renderer.py").read_text()
    m = re.search(r"# Auth eval on best.*?return best_scorer", src, re.DOTALL)
    assert m
    block = m.group(0)
    # The bad pattern: an early-out/skip when args.auth_eval_masks is missing.
    # Look for `if not args.auth_eval_masks:` followed by a print/skip with
    # NO subsequent subprocess.run.
    bad = re.search(
        r"if not args\.auth_eval_masks[^\n]*:\s*\n\s*print[^\n]*(?:WARN|Skipping)[^\n]*\n\s*else:",
        block,
    )
    assert bad is None, (
        "auth-eval-on-best must not gate the entire eval on --auth-eval-masks. "
        "Run with renderer-only archive size if masks missing (with WARN), "
        "but DO run. This was Council R3-3."
    )


def test_train_renderer_uses_repo_root_constant():
    """Council R3-4: subprocess path resolution must use the
    established `_repo` constant, not brittle `..` traversal."""
    src = (REPO / "src" / "tac" / "experiments" / "train_renderer.py").read_text()
    m = re.search(r"# Auth eval on best.*?return best_scorer", src, re.DOTALL)
    assert m
    block = m.group(0)
    # Either uses _repo or another robust pathlib idiom. Specifically reject
    # the `parents[N] / ".."` antipattern.
    bad_paths = re.findall(r'parents\[\d+\]\s*/\s*"\.\."', block)
    assert not bad_paths, (
        f"train_renderer auth-eval block uses brittle `parents[N] / '..'` "
        f"path traversal: {bad_paths}. Use the existing `_repo` constant."
    )


def _write_auth_eval_renderer_fixture(root: Path) -> None:
    auth_eval = root / "experiments" / "auth_eval_renderer.py"
    auth_eval.parent.mkdir(parents=True, exist_ok=True)
    auth_eval.write_text(
        """
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint")
    parser.add_argument("--upstream-dir")
    parser.add_argument("--device")
    parser.add_argument("--batch-size")
    parser.add_argument("--archive-size-bytes")
    parser.add_argument("--output-dir")
    parser.add_argument("--poses")
    return parser.parse_args()
""",
        encoding="utf-8",
    )


def _write_train_renderer_fixture(root: Path, block: str) -> None:
    train_renderer = root / "src" / "tac" / "experiments" / "train_renderer.py"
    train_renderer.parent.mkdir(parents=True, exist_ok=True)
    train_renderer.write_text(block, encoding="utf-8")


_GOOD_AUTH_EVAL_BLOCK = """
def train(args):
    # Auth eval on best
    if getattr(args, "auth_eval_on_best", False):
        if not args.auth_eval_masks or not args.auth_eval_poses:
            raise RuntimeError("missing masks/poses")
        from tac.submission_archive import build_submission_archive
        archive_path = out_dir / "auth_eval_on_best_archive.zip"
        build_submission_archive(output_path=archive_path)
        archive_bytes = archive_path.stat().st_size
        auth_eval_script = _repo / "experiments" / "auth_eval_renderer.py"
        cmd = [
            sys.executable, "-u", str(auth_eval_script),
            "--checkpoint", str(best_fp4),
            "--upstream-dir", args.auth_eval_upstream_dir,
            "--device", "cuda",
            "--archive-size-bytes", str(archive_bytes),
            "--output-dir", str(out_dir),
            "--poses", str(args.auth_eval_poses),
        ]
        subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    return best_scorer
"""


def test_train_renderer_auth_eval_preflight_passes_live_repo() -> None:
    from tac.preflight import (
        check_train_renderer_auth_eval_uses_current_auth_eval_renderer_cli,
    )

    assert check_train_renderer_auth_eval_uses_current_auth_eval_renderer_cli(
        repo_root=REPO, strict=True, verbose=False,
    ) == []


def test_train_renderer_auth_eval_preflight_accepts_clean_fixture(
    tmp_path: Path,
) -> None:
    from tac.preflight import (
        check_train_renderer_auth_eval_uses_current_auth_eval_renderer_cli,
    )

    _write_auth_eval_renderer_fixture(tmp_path)
    _write_train_renderer_fixture(tmp_path, _GOOD_AUTH_EVAL_BLOCK)

    assert check_train_renderer_auth_eval_uses_current_auth_eval_renderer_cli(
        repo_root=tmp_path, strict=True, verbose=False,
    ) == []


def test_train_renderer_auth_eval_preflight_rejects_invented_flag(
    tmp_path: Path,
) -> None:
    from tac.preflight import (
        check_train_renderer_auth_eval_uses_current_auth_eval_renderer_cli,
    )

    _write_auth_eval_renderer_fixture(tmp_path)
    block = _GOOD_AUTH_EVAL_BLOCK.replace(
        '"--poses", str(args.auth_eval_poses),',
        '"--auth-eval-masks", str(args.auth_eval_masks),\n'
        '            "--poses", str(args.auth_eval_poses),',
    )
    _write_train_renderer_fixture(tmp_path, block)

    violations = check_train_renderer_auth_eval_uses_current_auth_eval_renderer_cli(
        repo_root=tmp_path, strict=False, verbose=False,
    )
    assert any(
        "not declared by experiments/auth_eval_renderer.py" in v
        for v in violations
    )


def test_train_renderer_auth_eval_preflight_rejects_missing_archive_size(
    tmp_path: Path,
) -> None:
    from tac.preflight import (
        check_train_renderer_auth_eval_uses_current_auth_eval_renderer_cli,
    )

    _write_auth_eval_renderer_fixture(tmp_path)
    block = _GOOD_AUTH_EVAL_BLOCK.replace(
        '            "--archive-size-bytes", str(archive_bytes),\n',
        "",
    )
    _write_train_renderer_fixture(tmp_path, block)

    violations = check_train_renderer_auth_eval_uses_current_auth_eval_renderer_cli(
        repo_root=tmp_path, strict=False, verbose=False,
    )
    assert any("--archive-size-bytes" in v for v in violations)


def test_train_renderer_auth_eval_preflight_rejects_renderer_only_archive(
    tmp_path: Path,
) -> None:
    from tac.preflight import (
        check_train_renderer_auth_eval_uses_current_auth_eval_renderer_cli,
    )

    _write_auth_eval_renderer_fixture(tmp_path)
    block = _GOOD_AUTH_EVAL_BLOCK.replace(
        "build_submission_archive",
        "write_renderer_only_archive",
    )
    _write_train_renderer_fixture(tmp_path, block)

    violations = check_train_renderer_auth_eval_uses_current_auth_eval_renderer_cli(
        repo_root=tmp_path, strict=False, verbose=False,
    )
    assert any("build_submission_archive" in v for v in violations)


def test_train_renderer_auth_eval_preflight_rejects_soft_skip_on_missing_masks(
    tmp_path: Path,
) -> None:
    from tac.preflight import (
        check_train_renderer_auth_eval_uses_current_auth_eval_renderer_cli,
    )

    _write_auth_eval_renderer_fixture(tmp_path)
    block = _GOOD_AUTH_EVAL_BLOCK.replace(
        '        if not args.auth_eval_masks or not args.auth_eval_poses:\n'
        '            raise RuntimeError("missing masks/poses")\n',
        '        if not args.auth_eval_masks:\n'
        '            print("WARN Skipping auth eval")\n'
        '        else:\n'
        '            pass\n',
    )
    _write_train_renderer_fixture(tmp_path, block)

    violations = check_train_renderer_auth_eval_uses_current_auth_eval_renderer_cli(
        repo_root=tmp_path, strict=False, verbose=False,
    )
    assert any("fail loud" in v for v in violations)


def test_train_renderer_auth_eval_preflight_strict_raises(
    tmp_path: Path,
) -> None:
    from tac.preflight import (
        PreflightError,
        check_train_renderer_auth_eval_uses_current_auth_eval_renderer_cli,
    )

    _write_auth_eval_renderer_fixture(tmp_path)
    block = _GOOD_AUTH_EVAL_BLOCK.replace(
        '"--poses", str(args.auth_eval_poses),',
        '"--invented-renderer-eval-flag", "1",\n'
        '            "--poses", str(args.auth_eval_poses),',
    )
    _write_train_renderer_fixture(tmp_path, block)

    with pytest.raises(PreflightError, match="Catalog #226 sister gate"):
        check_train_renderer_auth_eval_uses_current_auth_eval_renderer_cli(
            repo_root=tmp_path, strict=True, verbose=False,
        )


def test_argparse_smoke_default_true():
    """The flag must default to True per CLAUDE.md non-negotiable, and
    --no-auth-eval-on-best must turn it off."""
    sys.path.insert(0, str(REPO / "src" / "tac" / "experiments"))
    if "train_renderer" in sys.modules:
        del sys.modules["train_renderer"]
    from train_renderer import parse_args  # noqa: E402
    a1 = parse_args(["--tag", "smoke"])
    assert a1.auth_eval_on_best is True
    assert a1.auth_eval_device == "cuda"
    assert a1.auth_eval_allow_axis is None
    a2 = parse_args(["--tag", "smoke", "--no-auth-eval-on-best"])
    assert a2.auth_eval_on_best is False
    a3 = parse_args([
        "--tag", "smoke",
        "--auth-eval-device", "cpu",
        "--auth-eval-allow-axis", "linux_cpu_advisory",
    ])
    assert a3.auth_eval_device == "cpu"
    assert a3.auth_eval_allow_axis == ["linux_cpu_advisory"]


def test_kl_distill_positive_weight_requires_explicit_segnet_aux_scope():
    sys.path.insert(0, str(REPO / "src" / "tac" / "experiments"))
    if "train_renderer" in sys.modules:
        del sys.modules["train_renderer"]
    from train_renderer import parse_args  # noqa: E402

    with pytest.raises(SystemExit, match="kl_distill_scope='segnet_aux'"):
        parse_args(["--tag", "smoke", "--kl-distill-weight", "0.1"])

    args = parse_args([
        "--tag", "smoke",
        "--kl-distill-weight", "0.01",
        "--kl-distill-scope", "segnet_aux",
    ])
    assert args.kl_distill_weight == 0.01
    assert args.kl_distill_scope == "segnet_aux"


def test_kl_distill_high_weight_requires_forensic_nonpromotion_opt_in():
    sys.path.insert(0, str(REPO / "src" / "tac" / "experiments"))
    if "train_renderer" in sys.modules:
        del sys.modules["train_renderer"]
    from train_renderer import parse_args  # noqa: E402

    with pytest.raises(SystemExit, match="high-scale KL"):
        parse_args([
            "--tag", "smoke",
            "--kl-distill-weight", "1.0",
            "--kl-distill-scope", "segnet_aux",
        ])

    args = parse_args([
        "--tag", "smoke",
        "--kl-distill-weight", "1.0",
        "--kl-distill-scope", "segnet_aux",
        "--allow-high-kl-weight-forensic",
    ])
    assert args.kl_distill_weight == 1.0
    assert args.promotion_eligible is False


def test_kl_distill_profiles_declare_explicit_scope():
    sys.path.insert(0, str(REPO / "src" / "tac" / "experiments"))
    if "train_renderer" in sys.modules:
        del sys.modules["train_renderer"]
    from train_renderer import parse_args  # noqa: E402

    args = parse_args([
        "--profile",
        "dilated_h64_half_frame_v3_annealed_kldistill",
        "--tag",
        "smoke",
    ])
    assert args.kl_distill_weight == 0.002
    assert args.kl_distill_scope == "segnet_aux"
    assert args.promotion_eligible is True

    with pytest.raises(SystemExit, match="never permits primary/full-scorer KL"):
        parse_args([
            "--tag", "smoke",
            "--kl-distill-weight", "0.1",
            "--kl-distill-scope", "primary_scorer",
        ])


def test_train_renderer_kl_scope_preflight_passes_current_profiles():
    from tac.preflight import check_train_renderer_kl_aux_explicit_scope

    assert check_train_renderer_kl_aux_explicit_scope(strict=True, verbose=False) == []


def test_train_renderer_kl_scope_preflight_rejects_promotable_high_weight_profile(monkeypatch):
    import tac.profiles as profiles
    from tac.preflight import PreflightError, check_train_renderer_kl_aux_explicit_scope

    patched = dict(profiles.PROFILES)
    patched["unsafe_high_weight_kl"] = {
        "kl_distill_weight": 1.0,
        "kl_distill_scope": "segnet_aux",
        "promotion_eligible": True,
    }
    monkeypatch.setattr(profiles, "PROFILES", patched)

    with pytest.raises(PreflightError, match="high-scale KL"):
        check_train_renderer_kl_aux_explicit_scope(strict=True, verbose=False)
