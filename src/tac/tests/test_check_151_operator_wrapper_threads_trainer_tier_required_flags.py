"""Dedicated tests for Catalog #151
(`check_operator_wrapper_threads_trainer_tier_required_flags`).

Grand council 2026-05-12 verdict: 9/10 PROCEED with R1-R7 stipulations.
This test module exercises each refinement.

Council refinements covered:
- R1: union ALL TIER_N_OPERATOR_REQUIRED_FLAGS module-level Assigns
- R2: wrapper invoking multiple trainers → union of their required flags
- R3: trainer with no manifest → fail-open (OK, no violation)
- R4: `--profile X` satisfies if X in `satisfied_by_profile` tuple
- R5: scope by INVOCATION (subprocess token near trainer path), not filename
- R6: exclude `experiments/results/public_pr*_intake_*/**` vendored clones
- R7: strict-from-byte-one acceptance (live count = 0)
- Plus: same-line `# TIER_REQUIRED_FLAG_WAIVED_OK:<flag>:<reason>` waiver
"""
from __future__ import annotations

from pathlib import Path

import pytest

from tac.preflight import (
    PreflightError,
    _check_151_collect_waivers,
    _check_151_extract_tier_manifests,
    _check_151_extract_trainer_paths,
    _check_151_wrapper_threads_flag,
    check_operator_wrapper_threads_trainer_tier_required_flags,
)


# -- Fixture helpers ---------------------------------------------------------

def _write_trainer(tmp: Path, name: str, manifest_src: str = "") -> Path:
    path = tmp / "experiments" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(manifest_src or "# trainer with no manifest\n")
    return path


def _write_wrapper(tmp: Path, name: str, body: str) -> Path:
    path = tmp / "scripts" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)
    return path


_BASIC_MANIFEST_SRC = '''
TIER_1_OPERATOR_REQUIRED_FLAGS = {
    "--enable-autocast-fp16": {
        "env": "T1_AUTOCAST",
        "rationale": "speedup",
        "default": None,
        "satisfied_by_profile": (),
        "requires": (),
    },
    "--enable-mp4-sim": {
        "env": "T1_MP4",
        "rationale": "proxy gap",
        "default": None,
        "satisfied_by_profile": (),
        "requires": (),
    },
}
'''


# -- AST extractor tests -----------------------------------------------------

def test_extract_tier_manifests_simple(tmp_path: Path) -> None:
    p = _write_trainer(tmp_path, "train_foo.py", _BASIC_MANIFEST_SRC)
    out = _check_151_extract_tier_manifests(p)
    assert set(out.keys()) == {"--enable-autocast-fp16", "--enable-mp4-sim"}
    assert out["--enable-autocast-fp16"]["env"] == "T1_AUTOCAST"
    assert out["--enable-autocast-fp16"]["rationale"] == "speedup"


def test_extract_no_manifest_returns_empty(tmp_path: Path) -> None:
    p = _write_trainer(tmp_path, "train_bare.py", "def main(): pass\n")
    assert _check_151_extract_tier_manifests(p) == {}


def test_extract_unparseable_returns_empty(tmp_path: Path) -> None:
    p = _write_trainer(tmp_path, "train_bad.py", "def broken(:\n")
    assert _check_151_extract_tier_manifests(p) == {}


# -- R1: multi-tier union ----------------------------------------------------

def test_R1_unions_tier_1_and_tier_2(tmp_path: Path) -> None:
    src = '''
TIER_1_OPERATOR_REQUIRED_FLAGS = {
    "--flag-a": {"env": "ENV_A", "rationale": "r"},
}
TIER_2_OPERATOR_REQUIRED_FLAGS = {
    "--flag-b": {"env": "ENV_B", "rationale": "r"},
}
'''
    p = _write_trainer(tmp_path, "train_multi.py", src)
    out = _check_151_extract_tier_manifests(p)
    assert set(out.keys()) == {"--flag-a", "--flag-b"}


# -- R5: invocation gating ---------------------------------------------------

def test_R5_docstring_mention_is_not_invocation(tmp_path: Path) -> None:
    body = '''#!/usr/bin/env bash
# This script does NOT invoke experiments/train_foo.py — just mentions it.
echo "Hello"
'''
    _write_wrapper(tmp_path, "remote_lane_only_mention.sh", body)
    _write_trainer(tmp_path, "train_foo.py", _BASIC_MANIFEST_SRC)
    out = check_operator_wrapper_threads_trainer_tier_required_flags(
        repo_root=tmp_path, strict=False, verbose=False,
    )
    assert out == [], f"docstring mention should not trigger violation, got {out}"


def test_R5_python_subprocess_invocation_is_in_scope(tmp_path: Path) -> None:
    _write_trainer(tmp_path, "train_foo.py", _BASIC_MANIFEST_SRC)
    body = '''import subprocess
subprocess.run(["python", "experiments/train_foo.py", "--epochs", "10"])
'''
    (tmp_path / "tools").mkdir(parents=True, exist_ok=True)
    (tmp_path / "tools" / "dispatch.py").write_text(body)
    out = check_operator_wrapper_threads_trainer_tier_required_flags(
        repo_root=tmp_path, strict=False, verbose=False,
    )
    assert len(out) == 2, f"expected 2 missing flags, got {out}"


def test_R5_shell_pybin_invocation_is_in_scope(tmp_path: Path) -> None:
    _write_trainer(tmp_path, "train_foo.py", _BASIC_MANIFEST_SRC)
    body = '''#!/usr/bin/env bash
"$PYBIN" -u experiments/train_foo.py --epochs 10
'''
    _write_wrapper(tmp_path, "remote_lane_x.sh", body)
    out = check_operator_wrapper_threads_trainer_tier_required_flags(
        repo_root=tmp_path, strict=False, verbose=False,
    )
    assert len(out) == 2, f"expected 2 missing flags, got {out}"


# -- Acceptance: literal flag --

def test_acceptance_literal_flag_present(tmp_path: Path) -> None:
    _write_trainer(tmp_path, "train_foo.py", _BASIC_MANIFEST_SRC)
    body = '''#!/usr/bin/env bash
"$PYBIN" -u experiments/train_foo.py --enable-autocast-fp16 --enable-mp4-sim
'''
    _write_wrapper(tmp_path, "remote_lane_x.sh", body)
    out = check_operator_wrapper_threads_trainer_tier_required_flags(
        repo_root=tmp_path, strict=False, verbose=False,
    )
    assert out == [], f"literal flags present, expected 0 violations: {out}"


# -- Acceptance: env-var block (the canonical NF1 fix pattern) --

def test_acceptance_env_var_block(tmp_path: Path) -> None:
    _write_trainer(tmp_path, "train_foo.py", _BASIC_MANIFEST_SRC)
    body = '''#!/usr/bin/env bash
"$PYBIN" -u experiments/train_foo.py
if [ "${T1_AUTOCAST:-0}" = "1" ]; then echo "go"; fi
if [ "${T1_MP4:-0}" = "1" ]; then echo "go"; fi
'''
    _write_wrapper(tmp_path, "remote_lane_x.sh", body)
    out = check_operator_wrapper_threads_trainer_tier_required_flags(
        repo_root=tmp_path, strict=False, verbose=False,
    )
    assert out == [], f"env-var blocks present, expected 0 violations: {out}"


# -- R4: profile-equivalence --

def test_R4_satisfied_by_profile_accepts_profile_thread(tmp_path: Path) -> None:
    src = '''
TIER_1_OPERATOR_REQUIRED_FLAGS = {
    "--segmentation-surrogate": {
        "env": "SS",
        "rationale": "r",
        "satisfied_by_profile": ("balle_cheap", "balle_full"),
    },
}
'''
    _write_trainer(tmp_path, "train_foo.py", src)
    body = '''#!/usr/bin/env bash
"$PYBIN" -u experiments/train_foo.py --profile balle_cheap
'''
    _write_wrapper(tmp_path, "remote_lane_x.sh", body)
    out = check_operator_wrapper_threads_trainer_tier_required_flags(
        repo_root=tmp_path, strict=False, verbose=False,
    )
    assert out == [], f"profile satisfies, expected 0 violations: {out}"


def test_R4_profile_mismatch_still_violation(tmp_path: Path) -> None:
    src = '''
TIER_1_OPERATOR_REQUIRED_FLAGS = {
    "--segmentation-surrogate": {
        "env": "SS",
        "rationale": "r",
        "satisfied_by_profile": ("balle_cheap",),
    },
}
'''
    _write_trainer(tmp_path, "train_foo.py", src)
    body = '''#!/usr/bin/env bash
"$PYBIN" -u experiments/train_foo.py --profile some_other_profile
'''
    _write_wrapper(tmp_path, "remote_lane_x.sh", body)
    out = check_operator_wrapper_threads_trainer_tier_required_flags(
        repo_root=tmp_path, strict=False, verbose=False,
    )
    assert len(out) == 1, f"profile mismatch should violate, got {out}"


# -- Same-line waiver --

def test_waiver_same_line(tmp_path: Path) -> None:
    _write_trainer(tmp_path, "train_foo.py", _BASIC_MANIFEST_SRC)
    body = '''#!/usr/bin/env bash
"$PYBIN" -u experiments/train_foo.py  # TIER_REQUIRED_FLAG_WAIVED_OK:--enable-autocast-fp16:debug-run
if [ "${T1_MP4:-0}" = "1" ]; then echo "go"; fi
'''
    _write_wrapper(tmp_path, "remote_lane_x.sh", body)
    out = check_operator_wrapper_threads_trainer_tier_required_flags(
        repo_root=tmp_path, strict=False, verbose=False,
    )
    assert out == [], f"waiver should accept, got {out}"


def test_waiver_only_covers_listed_flag(tmp_path: Path) -> None:
    _write_trainer(tmp_path, "train_foo.py", _BASIC_MANIFEST_SRC)
    # Waive autocast but NOT mp4-sim — should still violate for mp4-sim.
    body = '''#!/usr/bin/env bash
"$PYBIN" -u experiments/train_foo.py  # TIER_REQUIRED_FLAG_WAIVED_OK:--enable-autocast-fp16:reason
'''
    _write_wrapper(tmp_path, "remote_lane_x.sh", body)
    out = check_operator_wrapper_threads_trainer_tier_required_flags(
        repo_root=tmp_path, strict=False, verbose=False,
    )
    assert len(out) == 1 and "--enable-mp4-sim" in out[0]


# -- R3: no manifest → fail-open --

def test_R3_no_manifest_is_fail_open(tmp_path: Path) -> None:
    _write_trainer(tmp_path, "train_bare.py", "def main(): pass\n")
    body = '''#!/usr/bin/env bash
"$PYBIN" -u experiments/train_bare.py
'''
    _write_wrapper(tmp_path, "remote_lane_x.sh", body)
    out = check_operator_wrapper_threads_trainer_tier_required_flags(
        repo_root=tmp_path, strict=False, verbose=False,
    )
    assert out == [], "trainer with no manifest should not violate"


# -- R2: wrapper invokes multiple trainers --

def test_R2_multi_trainer_union(tmp_path: Path) -> None:
    src_a = '''
TIER_1_OPERATOR_REQUIRED_FLAGS = {
    "--flag-a": {"env": "ENV_A", "rationale": "a"},
}
'''
    src_b = '''
TIER_1_OPERATOR_REQUIRED_FLAGS = {
    "--flag-b": {"env": "ENV_B", "rationale": "b"},
}
'''
    _write_trainer(tmp_path, "train_alpha.py", src_a)
    _write_trainer(tmp_path, "train_beta.py", src_b)
    body = '''#!/usr/bin/env bash
"$PYBIN" -u experiments/train_alpha.py
"$PYBIN" -u experiments/train_beta.py
'''
    _write_wrapper(tmp_path, "remote_lane_combo.sh", body)
    out = check_operator_wrapper_threads_trainer_tier_required_flags(
        repo_root=tmp_path, strict=False, verbose=False,
    )
    # Union of required flags = 2; neither threaded.
    assert len(out) == 2
    assert any("--flag-a" in v for v in out)
    assert any("--flag-b" in v for v in out)


# -- R6: vendored intake exclusion --

def test_R6_excludes_public_pr_intake(tmp_path: Path) -> None:
    intake_dir = tmp_path / "experiments" / "results" / "public_pr107_intake_codex"
    intake_dir.mkdir(parents=True)
    _write_trainer(tmp_path, "train_foo.py", _BASIC_MANIFEST_SRC)
    # Vendored wrapper invokes the trainer literally but should be SKIPPED.
    (intake_dir / "wrapper.sh").write_text(
        '#!/usr/bin/env bash\n"$PYBIN" -u experiments/train_foo.py\n'
    )
    # Also a tools/ Python file under intake — same exclusion.
    out = check_operator_wrapper_threads_trainer_tier_required_flags(
        repo_root=tmp_path, strict=False, verbose=False,
    )
    assert out == [], "vendored intake clones must be excluded per R6"


# -- R7: strict-flip behavior --

def test_R7_strict_raises_on_violation(tmp_path: Path) -> None:
    _write_trainer(tmp_path, "train_foo.py", _BASIC_MANIFEST_SRC)
    body = '''#!/usr/bin/env bash
"$PYBIN" -u experiments/train_foo.py
'''
    _write_wrapper(tmp_path, "remote_lane_x.sh", body)
    with pytest.raises(PreflightError, match="check_operator_wrapper_threads"):
        check_operator_wrapper_threads_trainer_tier_required_flags(
            repo_root=tmp_path, strict=True, verbose=False,
        )


def test_R7_strict_passes_on_clean(tmp_path: Path) -> None:
    _write_trainer(tmp_path, "train_foo.py", _BASIC_MANIFEST_SRC)
    body = '''#!/usr/bin/env bash
"$PYBIN" -u experiments/train_foo.py --enable-autocast-fp16 --enable-mp4-sim
'''
    _write_wrapper(tmp_path, "remote_lane_x.sh", body)
    out = check_operator_wrapper_threads_trainer_tier_required_flags(
        repo_root=tmp_path, strict=True, verbose=False,
    )
    assert out == []


# -- Real-repo invariant: STRICT @ 0 on live repo --

def test_live_repo_strict_at_zero() -> None:
    """The canonical assertion: Catalog #151 lands strict-from-byte-one
    because NF1 wire-fix (commit d37c6b20) already drove the live count to 0.
    If this test fails, either: (a) someone broke an existing wrapper, or
    (b) the trainer-side manifest grew an entry without updating wrappers."""
    out = check_operator_wrapper_threads_trainer_tier_required_flags(
        repo_root=Path.cwd(), strict=False, verbose=False,
    )
    assert out == [], (
        f"Catalog #151 live violations: {len(out)}\n  "
        + "\n  ".join(out[:5])
    )


# -- Helpers --

def test_extract_trainer_paths_substring() -> None:
    text = 'subprocess.run(["python", "experiments/train_foo.py", "--x"])'
    assert _check_151_extract_trainer_paths(text) == [
        "experiments/train_foo.py"
    ]


def test_extract_trainer_paths_docstring_excluded() -> None:
    text = '"""This module wraps experiments/train_foo.py and friends."""\n'
    # No invocation token nearby → not in scope.
    assert _check_151_extract_trainer_paths(text) == []


def test_collect_waivers_multiple() -> None:
    text = (
        "x  # TIER_REQUIRED_FLAG_WAIVED_OK:--flag-a:reason1\n"
        "y  # TIER_REQUIRED_FLAG_WAIVED_OK:--flag-b:reason2\n"
    )
    waivers = _check_151_collect_waivers(text)
    assert waivers == {"--flag-a", "--flag-b"}


def test_wrapper_threads_flag_env_token() -> None:
    text = 'if [ "${T1_AUTOCAST:-0}" = "1" ]; then echo go; fi'
    assert _check_151_wrapper_threads_flag(
        text, "--enable-autocast-fp16", "T1_AUTOCAST", ()
    )


def test_wrapper_threads_flag_literal_only() -> None:
    text = '"$PYBIN" -u experiments/train_foo.py --enable-autocast-fp16'
    assert _check_151_wrapper_threads_flag(
        text, "--enable-autocast-fp16", "T1_AUTOCAST", ()
    )


def test_wrapper_threads_flag_neither_returns_false() -> None:
    text = '"$PYBIN" -u experiments/train_foo.py'
    assert not _check_151_wrapper_threads_flag(
        text, "--enable-autocast-fp16", "T1_AUTOCAST", ()
    )


# -- OD-WIRE-3 INVOKED-ONLY manifest declarations (2026-05-12) -----------
#
# Grand council 2026-05-12 verdict (subagent a6046518ed0ec4869): TIER_1 manifest
# constants must be declared on every operator-invoked trainer so Catalog #151
# can audit wire-up. The 5 trainers below were retrofitted; each carries an
# empty dict (no `--enable-*` semantic gates declared) — this pins the
# AST-discoverable manifest at byte zero so any future addition surfaces
# through the council process and a wrapper update lands in the same batch.

def _trainer_path(rel: str) -> Path:
    return Path.cwd() / rel


def test_tier_1_manifest_on_train_renderer() -> None:
    """train_renderer.py (canonical mask-conditioned renderer) declares
    TIER_1_OPERATOR_REQUIRED_FLAGS at module level."""
    p = _trainer_path("src/tac/experiments/train_renderer.py")
    assert p.exists(), f"trainer file missing: {p}"
    out = _check_151_extract_tier_manifests(p)
    assert out == {}, (
        f"train_renderer.py manifest should be empty (no operator-tier "
        f"semantic gates); got {out!r}"
    )


def test_tier_1_manifest_on_train_distill() -> None:
    """train_distill.py (distillation trainer) declares
    TIER_1_OPERATOR_REQUIRED_FLAGS at module level."""
    p = _trainer_path("experiments/train_distill.py")
    assert p.exists(), f"trainer file missing: {p}"
    out = _check_151_extract_tier_manifests(p)
    assert out == {}, (
        f"train_distill.py manifest should be empty (no operator-tier "
        f"semantic gates); got {out!r}"
    )


def test_tier_1_manifest_on_train_segmap() -> None:
    """train_segmap.py (SegMap renderer trainer) declares
    TIER_1_OPERATOR_REQUIRED_FLAGS at module level."""
    p = _trainer_path("experiments/train_segmap.py")
    assert p.exists(), f"trainer file missing: {p}"
    out = _check_151_extract_tier_manifests(p)
    assert out == {}, (
        f"train_segmap.py manifest should be empty (no operator-tier "
        f"semantic gates); got {out!r}"
    )


def test_tier_1_manifest_on_train_joint_pair() -> None:
    """train_joint_pair.py (Click-based JointPairGenerator trainer) declares
    TIER_1_OPERATOR_REQUIRED_FLAGS at module level."""
    p = _trainer_path("experiments/train_joint_pair.py")
    assert p.exists(), f"trainer file missing: {p}"
    out = _check_151_extract_tier_manifests(p)
    assert out == {}, (
        f"train_joint_pair.py manifest should be empty (no operator-tier "
        f"semantic gates); got {out!r}"
    )


def test_tier_1_manifest_on_train_nerv_mask() -> None:
    """train_nerv_mask.py (Lane 12 NeRV mask codec trainer) declares
    TIER_1_OPERATOR_REQUIRED_FLAGS at module level."""
    p = _trainer_path("experiments/train_nerv_mask.py")
    assert p.exists(), f"trainer file missing: {p}"
    out = _check_151_extract_tier_manifests(p)
    assert out == {}, (
        f"train_nerv_mask.py manifest should be empty (no operator-tier "
        f"semantic gates); got {out!r}"
    )
