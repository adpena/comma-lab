"""Tests for the meta-bug preflight checks (CLAUDE.md FORBIDDEN PATTERNS).

Each pattern below has bitten this project at least once and cost real GPU
money. Tests confirm the scanner catches the bad form and passes the clean
form. NEW FILE — does not extend test_preflight_dead_resolvers.py (that file
is being modified by a parallel subagent).
"""
from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from tac.preflight import (
    MetaBugViolation,
    _scan_inflate_for_scorer_load,
    _scan_python_for_disable_eval_roundtrip_flag,
    _scan_python_for_eval_roundtrip_false,
    _scan_python_for_mps_fallback,
    _scan_shell_for_missing_set_e,
    _scan_shell_for_pipefail_grep_q,
    _scan_shell_for_zip_binary,
    _scan_training_script_for_auth_eval,
    check_no_disable_eval_roundtrip_flag,
    check_no_eval_roundtrip_false,
    check_no_mps_fallback_default,
    check_no_pipefail_grep_q_trap,
    check_no_scorer_load_at_inflate,
    check_no_shell_zip_binary,
    check_shell_set_e_present,
    check_training_scripts_have_auth_eval,
)


def _write(p: Path, body: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(textwrap.dedent(body).lstrip("\n"))


def _stub_repo(tmp_path: Path) -> Path:
    """Build a minimal fake repo root."""
    (tmp_path / "src" / "tac").mkdir(parents=True, exist_ok=True)
    (tmp_path / "experiments").mkdir(parents=True, exist_ok=True)
    (tmp_path / "scripts").mkdir(parents=True, exist_ok=True)
    return tmp_path


# ─── Check 1: MPS-fallback device default ────────────────────────────────────


class TestNoMpsFallbackDefault:
    def test_classic_ternary_chain_is_caught(self, tmp_path: Path) -> None:
        root = _stub_repo(tmp_path)
        script = root / "experiments" / "bad.py"
        _write(script, """
            import torch
            def pick():
                device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
                return device
        """)
        v = _scan_python_for_mps_fallback(script, root)
        assert len(v) >= 1, v
        assert any("MPS-fallback" in s for s in v)

    def test_env_get_default_with_mps_is_caught(self, tmp_path: Path) -> None:
        root = _stub_repo(tmp_path)
        script = root / "experiments" / "bad_env.py"
        _write(script, """
            import os
            import torch
            def pick():
                # env.get default is the IfExp inside a Call — AST should still catch it
                d = os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "mps")
                return d
        """)
        v = _scan_python_for_mps_fallback(script, root)
        assert len(v) >= 1, v

    def test_cuda_required_default_passes(self, tmp_path: Path) -> None:
        root = _stub_repo(tmp_path)
        script = root / "experiments" / "good.py"
        _write(script, """
            import torch
            def pick():
                if not torch.cuda.is_available():
                    raise RuntimeError("CUDA required")
                return "cuda"
        """)
        v = _scan_python_for_mps_fallback(script, root)
        assert v == [], v

    def test_explicit_cpu_optin_passes(self, tmp_path: Path) -> None:
        root = _stub_repo(tmp_path)
        script = root / "experiments" / "good_optin.py"
        _write(script, """
            import argparse
            def pick():
                p = argparse.ArgumentParser()
                p.add_argument("--device", default="cuda")
                args = p.parse_args()
                return args.device
        """)
        v = _scan_python_for_mps_fallback(script, root)
        assert v == [], v

    def test_test_files_are_skipped(self, tmp_path: Path) -> None:
        root = _stub_repo(tmp_path)
        script = root / "src" / "tac" / "tests" / "test_dev.py"
        _write(script, """
            import torch
            d = "cuda" if torch.cuda.is_available() else "mps"
        """)
        v = _scan_python_for_mps_fallback(script, root)
        assert v == [], "test files should be skipped"

    def test_check_strict_raises(self, tmp_path: Path) -> None:
        root = _stub_repo(tmp_path)
        _write(root / "experiments" / "bad.py", """
            import torch
            d = "cuda" if torch.cuda.is_available() else "mps"
        """)
        with pytest.raises(MetaBugViolation):
            check_no_mps_fallback_default(repo_root=root, strict=True, verbose=False)

    def test_check_warn_only_returns_list(self, tmp_path: Path) -> None:
        root = _stub_repo(tmp_path)
        _write(root / "experiments" / "bad.py", """
            import torch
            d = "cuda" if torch.cuda.is_available() else "mps"
        """)
        v = check_no_mps_fallback_default(repo_root=root, strict=False, verbose=False)
        assert len(v) >= 1


# ─── Check 2: shell `set -e` required ────────────────────────────────────────


class TestShellSetEPresent:
    def test_set_uo_pipefail_no_e_is_caught(self, tmp_path: Path) -> None:
        root = _stub_repo(tmp_path)
        sh = root / "scripts" / "bad.sh"
        _write(sh, """
            #!/usr/bin/env bash
            set -uo pipefail
            ARCHIVE=$(zip out.zip in)
        """)
        v = _scan_shell_for_missing_set_e(sh, root)
        assert len(v) == 1, v
        assert "set -e" in v[0] or "without `e`" in v[0]

    def test_set_u_alone_is_caught(self, tmp_path: Path) -> None:
        root = _stub_repo(tmp_path)
        sh = root / "scripts" / "bad_u.sh"
        _write(sh, """
            #!/usr/bin/env bash
            set -u
            X=
        """)
        v = _scan_shell_for_missing_set_e(sh, root)
        assert len(v) == 1, v

    def test_set_euo_pipefail_passes(self, tmp_path: Path) -> None:
        root = _stub_repo(tmp_path)
        sh = root / "scripts" / "good.sh"
        _write(sh, """
            #!/usr/bin/env bash
            set -euo pipefail
            ARCHIVE=$(python -c "import zipfile; ...")
        """)
        v = _scan_shell_for_missing_set_e(sh, root)
        assert v == [], v

    def test_set_e_alone_passes(self, tmp_path: Path) -> None:
        root = _stub_repo(tmp_path)
        sh = root / "scripts" / "good_e.sh"
        _write(sh, """
            #!/bin/bash
            set -e
            echo hi
        """)
        v = _scan_shell_for_missing_set_e(sh, root)
        assert v == [], v

    def test_check_strict_raises(self, tmp_path: Path) -> None:
        root = _stub_repo(tmp_path)
        _write(root / "scripts" / "bad.sh", """
            #!/usr/bin/env bash
            set -uo pipefail
        """)
        with pytest.raises(MetaBugViolation):
            check_shell_set_e_present(repo_root=root, strict=True, verbose=False)


# ─── Check 3: shell `zip` binary ─────────────────────────────────────────────


class TestNoShellZipBinary:
    def test_zip_invocation_is_caught(self, tmp_path: Path) -> None:
        root = _stub_repo(tmp_path)
        sh = root / "scripts" / "bad.sh"
        _write(sh, """
            #!/usr/bin/env bash
            set -euo pipefail
            zip archive.zip renderer.bin masks.mkv poses.pt
        """)
        v = _scan_shell_for_zip_binary(sh, root)
        assert len(v) == 1, v
        assert "zip" in v[0] and "zipfile" in v[0]

    def test_python_zipfile_passes(self, tmp_path: Path) -> None:
        root = _stub_repo(tmp_path)
        sh = root / "scripts" / "good.sh"
        _write(sh, """
            #!/usr/bin/env bash
            set -euo pipefail
            python -c "import zipfile; zipfile.ZipFile('archive.zip','w').write('renderer.bin')"
        """)
        v = _scan_shell_for_zip_binary(sh, root)
        assert v == [], v

    def test_unzip_passes(self, tmp_path: Path) -> None:
        root = _stub_repo(tmp_path)
        sh = root / "scripts" / "good_unzip.sh"
        _write(sh, """
            #!/usr/bin/env bash
            unzip archive.zip -d /tmp
        """)
        v = _scan_shell_for_zip_binary(sh, root)
        assert v == [], v

    def test_zipfile_keyword_passes(self, tmp_path: Path) -> None:
        root = _stub_repo(tmp_path)
        sh = root / "scripts" / "good_keyword.sh"
        _write(sh, """
            #!/bin/bash
            python3 my_zipfile_tool.py
        """)
        v = _scan_shell_for_zip_binary(sh, root)
        assert v == [], v

    def test_comments_are_ignored(self, tmp_path: Path) -> None:
        root = _stub_repo(tmp_path)
        sh = root / "scripts" / "comments.sh"
        _write(sh, """
            #!/bin/bash
            # we used to call zip here, now we use python
            python -c "import zipfile"
        """)
        v = _scan_shell_for_zip_binary(sh, root)
        assert v == [], v

    def test_check_strict_raises(self, tmp_path: Path) -> None:
        root = _stub_repo(tmp_path)
        _write(root / "scripts" / "bad.sh", "zip out.zip in\n")
        with pytest.raises(MetaBugViolation):
            check_no_shell_zip_binary(repo_root=root, strict=True, verbose=False)


# ─── Check 4: pipefail + grep -q SIGPIPE trap ────────────────────────────────


class TestNoPipefailGrepQTrap:
    def test_pipefail_grep_q_is_caught(self, tmp_path: Path) -> None:
        root = _stub_repo(tmp_path)
        sh = root / "scripts" / "bad.sh"
        _write(sh, """
            #!/usr/bin/env bash
            set -euo pipefail
            vastai logs INSTANCE | grep -q "ready"
        """)
        v = _scan_shell_for_pipefail_grep_q(sh, root)
        assert len(v) == 1, v
        assert "SIGPIPE" in v[0] or "grep -q" in v[0]

    def test_capture_first_idiom_passes(self, tmp_path: Path) -> None:
        """codex R5-3 #6: `echo "$VAR" | grep -q PAT` is the prescribed
        remediation for the SIGPIPE bug class. Scanner MUST NOT flag it
        — otherwise it blocks its own fix."""
        root = _stub_repo(tmp_path)
        sh = root / "scripts" / "good.sh"
        _write(sh, """
            #!/usr/bin/env bash
            set -euo pipefail
            OUT=$(vastai logs INSTANCE 2>&1)
            echo "$OUT" | grep -q "ready"
        """)
        v = _scan_shell_for_pipefail_grep_q(sh, root)
        assert v == [], (
            f"echo | grep -q is the safe form (echo is a builtin, no "
            f"meaningful SIGPIPE) — must not be flagged; got {v}"
        )

    def test_printf_capture_idiom_passes(self, tmp_path: Path) -> None:
        """codex R5-3 #6: printf is also a builtin — same exemption as echo."""
        root = _stub_repo(tmp_path)
        sh = root / "scripts" / "good_printf.sh"
        _write(sh, """
            #!/usr/bin/env bash
            set -euo pipefail
            OUT=$(some_cmd)
            printf "%s" "$OUT" | grep -q "ready"
        """)
        v = _scan_shell_for_pipefail_grep_q(sh, root)
        assert v == [], (
            f"printf | grep -q must not be flagged (codex R5-3 #6); got {v}"
        )

    def test_here_string_form_passes(self, tmp_path: Path) -> None:
        """codex R5-3 #6: `grep -q PAT <<< "$VAR"` is the here-string form
        — no pipe at all, so SIGPIPE cannot occur."""
        root = _stub_repo(tmp_path)
        sh = root / "scripts" / "good_here_string.sh"
        _write(sh, """
            #!/usr/bin/env bash
            set -euo pipefail
            OUT=$(some_cmd)
            grep -q "ready" <<< "$OUT"
        """)
        v = _scan_shell_for_pipefail_grep_q(sh, root)
        assert v == [], (
            f"here-string `grep -q PAT <<< \"$OUT\"` has no pipe — must not "
            f"be flagged (codex R5-3 #6); got {v}"
        )

    def test_if_negated_echo_passes(self, tmp_path: Path) -> None:
        """codex R5-3 #6: real-world remote_setup_full.sh form
        `if ! echo "$X" | grep -q PAT` — echo is still a builtin."""
        root = _stub_repo(tmp_path)
        sh = root / "scripts" / "good_if_neg.sh"
        _write(sh, """
            #!/usr/bin/env bash
            set -euo pipefail
            X=$(some_cmd)
            if ! echo "$X" | grep -q "needle"; then
                echo "missing"
            fi
        """)
        v = _scan_shell_for_pipefail_grep_q(sh, root)
        assert v == [], (
            f"`if ! echo ... | grep -q` is safe (echo is a builtin) — must "
            f"not be flagged (codex R5-3 #6); got {v}"
        )

    def test_unsafe_external_cmd_still_flagged(self, tmp_path: Path) -> None:
        """codex R5-3 #6: unsafe form (external cmd LHS) must STILL fire.
        The exemption is narrow — only echo/printf/here-string."""
        root = _stub_repo(tmp_path)
        sh = root / "scripts" / "still_bad.sh"
        _write(sh, """
            #!/usr/bin/env bash
            set -euo pipefail
            unzip -l archive.zip | grep -q postfilter.pt
        """)
        v = _scan_shell_for_pipefail_grep_q(sh, root)
        assert len(v) == 1, (
            f"`unzip | grep -q` is the original bug class — must STILL be "
            f"flagged after the codex R5-3 #6 echo/printf exemption; got {v}"
        )

    def test_no_pipefail_no_violation(self, tmp_path: Path) -> None:
        root = _stub_repo(tmp_path)
        sh = root / "scripts" / "no_pipefail.sh"
        _write(sh, """
            #!/bin/bash
            # No set -e, no pipefail — grep -q is safe here.
            cat foo | grep -q "bar"
        """)
        v = _scan_shell_for_pipefail_grep_q(sh, root)
        assert v == [], v

    def test_grep_without_q_passes(self, tmp_path: Path) -> None:
        root = _stub_repo(tmp_path)
        sh = root / "scripts" / "grep_no_q.sh"
        _write(sh, """
            #!/usr/bin/env bash
            set -euo pipefail
            cmd | grep "pattern" > /dev/null
        """)
        v = _scan_shell_for_pipefail_grep_q(sh, root)
        assert v == [], v

    def test_check_strict_raises(self, tmp_path: Path) -> None:
        root = _stub_repo(tmp_path)
        _write(root / "scripts" / "bad.sh", """
            #!/usr/bin/env bash
            set -euo pipefail
            cmd | grep -q "x"
        """)
        with pytest.raises(MetaBugViolation):
            check_no_pipefail_grep_q_trap(repo_root=root, strict=True, verbose=False)


# ─── Check 5: eval_roundtrip=False ───────────────────────────────────────────


class TestNoEvalRoundtripFalse:
    def test_kwarg_false_is_caught(self, tmp_path: Path) -> None:
        root = _stub_repo(tmp_path)
        script = root / "experiments" / "bad_call.py"
        _write(script, """
            def go():
                train(model, eval_roundtrip=False)
        """)
        v = _scan_python_for_eval_roundtrip_false(script, root)
        assert len(v) == 1, v
        assert "eval_roundtrip=False" in v[0]

    def test_default_false_is_caught(self, tmp_path: Path) -> None:
        root = _stub_repo(tmp_path)
        script = root / "experiments" / "bad_def.py"
        _write(script, """
            def train(model, eval_roundtrip: bool = False):
                pass
        """)
        v = _scan_python_for_eval_roundtrip_false(script, root)
        assert len(v) == 1, v
        assert "default" in v[0].lower() or "defaults" in v[0].lower()

    def test_default_true_passes(self, tmp_path: Path) -> None:
        root = _stub_repo(tmp_path)
        script = root / "experiments" / "good.py"
        _write(script, """
            def train(model, eval_roundtrip: bool = True):
                pass
            def go():
                train(m, eval_roundtrip=True)
        """)
        v = _scan_python_for_eval_roundtrip_false(script, root)
        assert v == [], v

    def test_kwonly_default_false_is_caught(self, tmp_path: Path) -> None:
        root = _stub_repo(tmp_path)
        script = root / "experiments" / "kwonly.py"
        _write(script, """
            def train(model, *, eval_roundtrip=False):
                pass
        """)
        v = _scan_python_for_eval_roundtrip_false(script, root)
        assert len(v) == 1, v

    def test_test_files_skipped(self, tmp_path: Path) -> None:
        root = _stub_repo(tmp_path)
        script = root / "src" / "tac" / "tests" / "test_x.py"
        _write(script, """
            def fn(eval_roundtrip=False): pass
        """)
        v = _scan_python_for_eval_roundtrip_false(script, root)
        assert v == [], v

    def test_check_strict_raises(self, tmp_path: Path) -> None:
        root = _stub_repo(tmp_path)
        _write(root / "experiments" / "bad.py", "go(eval_roundtrip=False)\n")
        with pytest.raises(MetaBugViolation):
            check_no_eval_roundtrip_false(repo_root=root, strict=True, verbose=False)


# ─── Check 6: scorer load at inflate ─────────────────────────────────────────


class TestNoScorerLoadAtInflate:
    def test_load_scorers_at_inflate_is_caught(self, tmp_path: Path) -> None:
        root = _stub_repo(tmp_path)
        inf = root / "submissions" / "robust_current" / "inflate.py"
        _write(inf, """
            from tac.scorer import load_scorers
            def main():
                segnet, posenet = load_scorers()
                return segnet, posenet
        """)
        v = _scan_inflate_for_scorer_load(inf, root)
        assert len(v) >= 1, v
        assert any("scorer" in s.lower() for s in v)

    def test_renderer_only_passes(self, tmp_path: Path) -> None:
        root = _stub_repo(tmp_path)
        inf = root / "submissions" / "robust_current" / "inflate.py"
        _write(inf, """
            import torch
            def main():
                model = torch.load("renderer.bin", map_location="cpu")
                return model
        """)
        v = _scan_inflate_for_scorer_load(inf, root)
        assert v == [], v

    def test_load_posenet_call_is_caught(self, tmp_path: Path) -> None:
        root = _stub_repo(tmp_path)
        inf = root / "submissions" / "robust_current" / "inflate_renderer.py"
        _write(inf, """
            from somewhere import load_posenet
            def main():
                pose = load_posenet()
        """)
        v = _scan_inflate_for_scorer_load(inf, root)
        assert len(v) >= 1, v

    def test_check_strict_raises(self, tmp_path: Path) -> None:
        root = _stub_repo(tmp_path)
        _write(root / "submissions" / "robust_current" / "inflate.py", """
            from tac.scorer import load_segnet
        """)
        with pytest.raises(MetaBugViolation):
            check_no_scorer_load_at_inflate(repo_root=root, strict=True, verbose=False)

    def test_check_no_submissions_dir_passes(self, tmp_path: Path) -> None:
        root = _stub_repo(tmp_path)
        # No submissions dir at all → vacuously OK.
        v = check_no_scorer_load_at_inflate(repo_root=root, strict=True, verbose=False)
        assert v == []


# ─── Check 7: training scripts must auth-eval ────────────────────────────────


class TestTrainingScriptsHaveAuthEval:
    def test_save_without_auth_eval_is_caught(self, tmp_path: Path) -> None:
        root = _stub_repo(tmp_path)
        script = root / "experiments" / "train_thing.py"
        _write(script, """
            import torch
            def go():
                model = build()
                torch.save(model.state_dict(), "renderer_best.pt")
        """)
        v = _scan_training_script_for_auth_eval(script, root)
        assert len(v) == 1, v
        assert "auth" in v[0].lower()

    def test_save_with_subprocess_auth_eval_passes(self, tmp_path: Path) -> None:
        root = _stub_repo(tmp_path)
        script = root / "experiments" / "train_good.py"
        _write(script, """
            import subprocess, torch
            def go():
                torch.save(model.state_dict(), "renderer_best.pt")
                subprocess.run(["python", "auth_eval_renderer.py", "--ckpt", "renderer_best.pt"])
        """)
        v = _scan_training_script_for_auth_eval(script, root)
        assert v == [], v

    def test_save_with_tac_auth_eval_import_passes(self, tmp_path: Path) -> None:
        root = _stub_repo(tmp_path)
        script = root / "experiments" / "train_imp.py"
        _write(script, """
            import torch
            from tac.auth_eval import run_auth_eval
            def go():
                torch.save(model, "renderer.pt")
                run_auth_eval("renderer.pt")
        """)
        v = _scan_training_script_for_auth_eval(script, root)
        assert v == [], v

    def test_no_save_skips_check(self, tmp_path: Path) -> None:
        root = _stub_repo(tmp_path)
        script = root / "experiments" / "train_smoke.py"
        _write(script, """
            def go():
                print("just a test, no save")
        """)
        v = _scan_training_script_for_auth_eval(script, root)
        assert v == [], v

    def test_explicit_optout_flag_satisfies(self, tmp_path: Path) -> None:
        root = _stub_repo(tmp_path)
        script = root / "experiments" / "train_optout.py"
        _write(script, """
            # Operator may pass --no-auth-eval-on-best to skip; the rule is
            # satisfied by the existence of the flag (operator made an
            # explicit choice).
            import torch
            import argparse
            p = argparse.ArgumentParser()
            p.add_argument("--no-auth-eval-on-best", action="store_true")
            torch.save(model.state_dict(), "renderer_best.pt")
        """)
        v = _scan_training_script_for_auth_eval(script, root)
        assert v == [], v

    def test_check_strict_raises(self, tmp_path: Path) -> None:
        root = _stub_repo(tmp_path)
        _write(root / "experiments" / "train_bad.py", """
            import torch
            torch.save(m.state_dict(), "renderer_best.pt")
        """)
        with pytest.raises(MetaBugViolation):
            check_training_scripts_have_auth_eval(repo_root=root, strict=True, verbose=False)


# ─── Check 8: --no-eval-roundtrip CLI flag ──────────────────────────────────


class TestNoDisableEvalRoundtripFlag:
    def test_disable_flag_is_caught(self, tmp_path: Path) -> None:
        root = _stub_repo(tmp_path)
        script = root / "experiments" / "bad_argparse.py"
        _write(script, """
            import argparse
            p = argparse.ArgumentParser()
            p.add_argument("--no-eval-roundtrip", action="store_true")
        """)
        v = _scan_python_for_disable_eval_roundtrip_flag(script, root)
        assert len(v) == 1, v
        assert "--no-eval-roundtrip" in v[0]

    def test_clean_argparse_passes(self, tmp_path: Path) -> None:
        root = _stub_repo(tmp_path)
        script = root / "experiments" / "good_argparse.py"
        _write(script, """
            import argparse
            p = argparse.ArgumentParser()
            p.add_argument("--eval-roundtrip", action="store_true", default=True)
        """)
        v = _scan_python_for_disable_eval_roundtrip_flag(script, root)
        assert v == [], v

    def test_test_files_skipped(self, tmp_path: Path) -> None:
        root = _stub_repo(tmp_path)
        script = root / "src" / "tac" / "tests" / "test_thing.py"
        _write(script, """
            import argparse
            p = argparse.ArgumentParser()
            p.add_argument("--no-eval-roundtrip")
        """)
        v = _scan_python_for_disable_eval_roundtrip_flag(script, root)
        assert v == [], v

    def test_check_strict_raises(self, tmp_path: Path) -> None:
        root = _stub_repo(tmp_path)
        _write(root / "experiments" / "bad.py", """
            import argparse
            p = argparse.ArgumentParser()
            p.add_argument("--no-eval-roundtrip")
        """)
        with pytest.raises(MetaBugViolation):
            check_no_disable_eval_roundtrip_flag(repo_root=root, strict=True, verbose=False)


# ─── codex R5-3 #5: heredoc masking for shell scanners ──────────────────────


class TestHeredocMasking:
    """Pin that bash heredoc bodies are NOT scanned as executable shell.

    Bug class (pre-fix): a heredoc embedding Python, docs, or generated
    shell text with lines like `set -uo pipefail`, `zip out.zip ...`,
    or `| grep -q` would be flagged by the shell scanners as if the
    heredoc body were code. Fix: `_mask_shell_heredocs(text)` zeroes
    out heredoc bodies (preserving line numbers) before regex scan.
    """

    # ----- _scan_shell_for_missing_set_e under heredoc -----

    def test_set_uo_pipefail_inside_heredoc_is_not_flagged(
        self, tmp_path: Path,
    ) -> None:
        """Heredoc body containing the bug pattern must NOT trigger.
        Real-world case: scripts that emit shell snippets via heredoc."""
        root = _stub_repo(tmp_path)
        sh = root / "scripts" / "emit.sh"
        _write(sh, """
            #!/usr/bin/env bash
            set -euo pipefail
            cat > /tmp/inner.sh <<'INNER'
            #!/usr/bin/env bash
            set -uo pipefail
            X=$(echo hi)
            INNER
            echo done
        """)
        v = _scan_shell_for_missing_set_e(sh, root)
        assert v == [], (
            f"`set -uo pipefail` inside a heredoc is documentation, not "
            f"executable shell — must not be flagged; got {v}"
        )

    def test_set_uo_pipefail_outside_heredoc_still_flagged(
        self, tmp_path: Path,
    ) -> None:
        """Control: same pattern OUTSIDE a heredoc must STILL be caught."""
        root = _stub_repo(tmp_path)
        sh = root / "scripts" / "real_bad.sh"
        _write(sh, """
            #!/usr/bin/env bash
            set -uo pipefail
            X=
        """)
        v = _scan_shell_for_missing_set_e(sh, root)
        assert len(v) == 1, (
            f"Control: same pattern OUTSIDE heredoc must still flag; got {v}"
        )

    # ----- _scan_shell_for_zip_binary under heredoc -----

    def test_zip_inside_python_heredoc_is_not_flagged(
        self, tmp_path: Path,
    ) -> None:
        """Common idiom: `python3 <<'PY' ... import zipfile ... PY`. The
        word 'zip' appearing in Python code must not be treated as a
        shell `zip` invocation."""
        root = _stub_repo(tmp_path)
        sh = root / "scripts" / "python_emit.sh"
        _write(sh, """
            #!/usr/bin/env bash
            set -euo pipefail
            python3 <<'PY'
            # generate archive in pure python — no shell zip needed
            zip out.zip in
            print("zip out.zip in")
            PY
        """)
        v = _scan_shell_for_zip_binary(sh, root)
        assert v == [], (
            f"`zip out.zip` inside python heredoc is Python code, not shell "
            f"— must not be flagged; got {v}"
        )

    def test_zip_outside_heredoc_still_flagged(self, tmp_path: Path) -> None:
        """Control: shell-level `zip` must STILL be caught."""
        root = _stub_repo(tmp_path)
        sh = root / "scripts" / "real_zip.sh"
        _write(sh, """
            #!/bin/bash
            zip archive.zip renderer.bin
        """)
        v = _scan_shell_for_zip_binary(sh, root)
        assert len(v) == 1, v

    # ----- _scan_shell_for_pipefail_grep_q under heredoc -----

    def test_grep_q_inside_heredoc_is_not_flagged(self, tmp_path: Path) -> None:
        """A heredoc body containing `cmd | grep -q` (e.g. emitted snippet
        for documentation or downstream script) must not be flagged."""
        root = _stub_repo(tmp_path)
        sh = root / "scripts" / "doc_emit.sh"
        _write(sh, """
            #!/usr/bin/env bash
            set -euo pipefail
            cat > README <<'EOF'
            Example bad pattern:
                some_cmd | grep -q "needle"
            EOF
        """)
        v = _scan_shell_for_pipefail_grep_q(sh, root)
        assert v == [], (
            f"`cmd | grep -q` inside heredoc is documentation, not shell "
            f"— must not be flagged; got {v}"
        )

    def test_grep_q_outside_heredoc_still_flagged(self, tmp_path: Path) -> None:
        """Control: real `cmd | grep -q` outside heredoc must STILL be caught."""
        root = _stub_repo(tmp_path)
        sh = root / "scripts" / "real_grep_q.sh"
        _write(sh, """
            #!/usr/bin/env bash
            set -euo pipefail
            unzip -l archive.zip | grep -q renderer.bin
        """)
        v = _scan_shell_for_pipefail_grep_q(sh, root)
        assert len(v) == 1, v

    # ----- heredoc edge cases -----

    def test_dash_heredoc_terminator_with_leading_tabs(
        self, tmp_path: Path,
    ) -> None:
        """`<<-TOKEN` strips leading tabs from the terminator. Mask must
        recognize tab-indented terminators so the body is correctly bounded."""
        root = _stub_repo(tmp_path)
        sh = root / "scripts" / "dash_hd.sh"
        # Note: tabs are intentional inside the heredoc body for <<- form.
        sh.parent.mkdir(parents=True, exist_ok=True)
        sh.write_text(
            "#!/usr/bin/env bash\n"
            "set -euo pipefail\n"
            "cat <<-EOF\n"
            "\tset -uo pipefail\n"
            "\tEOF\n"
            "echo done\n"
        )
        v = _scan_shell_for_missing_set_e(sh, root)
        assert v == [], (
            f"<<-EOF with tab-stripped terminator: body must be masked; got {v}"
        )

    def test_unterminated_heredoc_masks_to_eof(self, tmp_path: Path) -> None:
        """A heredoc without a terminator (script error) — we mask to EOF
        so we conservatively avoid scanning the unterminated body."""
        root = _stub_repo(tmp_path)
        sh = root / "scripts" / "unterminated.sh"
        _write(sh, """
            #!/usr/bin/env bash
            set -euo pipefail
            cat <<NEVER_CLOSED
            zip out.zip in
            set -uo pipefail
        """)
        v_zip = _scan_shell_for_zip_binary(sh, root)
        v_set = _scan_shell_for_missing_set_e(sh, root)
        assert v_zip == [], v_zip
        assert v_set == [], v_set


# ─── codex R5-3 #7: MPS BoolOp chain detection ──────────────────────────────


class TestMpsBoolOpDetection:
    """Pin that BoolOp (and/or) device-selection chains are caught.

    Bug class: `cuda.is_available() and 'cuda' or mps.is_available() and 'mps'
    or 'cpu'` is the same MPS-fallback pattern as the IfExp ternary, but
    has no IfExp anywhere — must AST-walk BoolOp explicitly.
    """

    def test_classic_and_or_chain_is_caught(self, tmp_path: Path) -> None:
        root = _stub_repo(tmp_path)
        script = root / "experiments" / "boolop_bad.py"
        _write(script, """
            import torch
            def pick():
                return (torch.cuda.is_available() and 'cuda'
                        or torch.backends.mps.is_available() and 'mps'
                        or 'cpu')
        """)
        v = _scan_python_for_mps_fallback(script, root)
        assert any("BoolOp" in s or "MPS-fallback" in s for s in v), v
        assert len(v) >= 1, v

    def test_nested_parenthesized_boolop_is_caught(
        self, tmp_path: Path,
    ) -> None:
        """Inline cuda check + nested parens: the outer `or` is the
        result-position BoolOp containing the cuda check AND 'mps'."""
        root = _stub_repo(tmp_path)
        script = root / "experiments" / "boolop_nested.py"
        _write(script, """
            import torch
            def pick():
                return ((torch.cuda.is_available() and 'cuda')
                        or (torch.backends.mps.is_available() and 'mps')
                        or 'cpu')
        """)
        v = _scan_python_for_mps_fallback(script, root)
        # Nested case: outer `or` BoolOp contains the cuda call subtree
        # AND has 'mps' as a leaf via the middle inner BoolOp's value.
        assert len(v) >= 1, v

    def test_pure_mps_literal_without_cuda_check_passes(
        self, tmp_path: Path,
    ) -> None:
        """Someone explicitly choosing MPS for an MPS-only test (e.g.
        `device = 'mps'` literal) must NOT flag — the rule targets the
        FALLBACK pattern, not deliberate MPS use."""
        root = _stub_repo(tmp_path)
        script = root / "experiments" / "explicit_mps.py"
        _write(script, """
            def make_mps_only_test():
                device = "mps"  # we are deliberately on MPS for this probe
                return device
        """)
        v = _scan_python_for_mps_fallback(script, root)
        assert v == [], (
            f"Explicit `device = 'mps'` literal (no cuda check) must NOT "
            f"flag — only the FALLBACK pattern is forbidden; got {v}"
        )

    def test_compare_with_mps_string_is_not_a_fallback(
        self, tmp_path: Path,
    ) -> None:
        """Real FP from training.py: `use_autocast = (cuda.is_available()
        and 'cuda') or str(self.device) == 'mps'`. The literal 'mps' is
        inside a Compare — never selected as the result value. Must NOT
        flag."""
        root = _stub_repo(tmp_path)
        script = root / "experiments" / "compare_mps.py"
        _write(script, """
            import torch
            class T:
                device = None
                def use_autocast(self):
                    return ((str(self.device).startswith("cuda")
                             and torch.cuda.is_available())
                            or str(self.device) == "mps")
        """)
        v = _scan_python_for_mps_fallback(script, root)
        assert v == [], (
            f"`... or str(dev) == 'mps'` is a Compare, not a fallback "
            f"value — must NOT flag; got {v}"
        )


# ─── codex R5-3 #8: training-script auth-eval AST upgrade ───────────────────


class TestTrainingAuthEvalAstUpgrade:
    """Pin the AST-based auth-eval check (replaces the old token-grep).

    Bug class (pre-fix): the regex form was both too narrow (missed
    multiline saves, variable paths) and too broad (matched any auth_eval
    token in a comment / help string / dead import). The new AST walker:
      1. Saves a renderer (path token: renderer/checkpoint/fp4/model.pt).
      2. Calls auth_eval (subprocess.run, .main(), helper) — or has the
         --no-auth-eval-on-best opt-out flag.
      Failure to satisfy 2 → violation. Imports without calls = dead-code
      violation.
    """

    def test_save_non_renderer_does_not_flag(self, tmp_path: Path) -> None:
        """torch.save(stats, 'stats.pt') — not a renderer, no violation."""
        root = _stub_repo(tmp_path)
        script = root / "experiments" / "train_stats.py"
        _write(script, """
            import torch
            def go():
                torch.save({"loss": 0.5}, "stats.pt")
        """)
        v = _scan_training_script_for_auth_eval(script, root)
        assert v == [], (
            f"`stats.pt` is not a renderer (no renderer/checkpoint/fp4 "
            f"token) — must NOT flag; got {v}"
        )

    def test_save_lora_does_not_flag(self, tmp_path: Path) -> None:
        """train_lora_tto.py-style: `torch.save(state, 'lora_best.pt')`.
        LoRA is not a renderer — must not flag."""
        root = _stub_repo(tmp_path)
        script = root / "experiments" / "train_lora.py"
        _write(script, """
            import torch
            def go():
                torch.save(lora_state, output_dir / "lora_best.pt")
                torch.save(lora_state, output_dir / "lora_final.pt")
        """)
        v = _scan_training_script_for_auth_eval(script, root)
        assert v == [], v

    def test_save_postfilter_does_not_flag(self, tmp_path: Path) -> None:
        """train_postfilter_on_renderer.py-style: postfilter saves are
        not renderer saves."""
        root = _stub_repo(tmp_path)
        script = root / "experiments" / "train_pf.py"
        _write(script, """
            import torch
            def go():
                torch.save(postfilter.state_dict(), output_dir / "postfilter_best.pt")
                torch.save(postfilter_int8, output_dir / "postfilter_int8.pt")
        """)
        v = _scan_training_script_for_auth_eval(script, root)
        assert v == [], v

    def test_save_renderer_with_subprocess_auth_eval_passes(
        self, tmp_path: Path,
    ) -> None:
        """Renderer save FOLLOWED by subprocess auth eval — satisfied."""
        root = _stub_repo(tmp_path)
        script = root / "experiments" / "train_good_subp.py"
        _write(script, """
            import subprocess, torch
            def go():
                torch.save(model.state_dict(), "renderer_best.pt")
                subprocess.run([
                    "python", "auth_eval_renderer.py",
                    "--ckpt", "renderer_best.pt",
                ])
        """)
        v = _scan_training_script_for_auth_eval(script, root)
        assert v == [], v

    def test_save_renderer_with_optout_flag_passes(self, tmp_path: Path) -> None:
        """Renderer save with --no-auth-eval-on-best argparse opt-out: the
        operator made an explicit choice — satisfied."""
        root = _stub_repo(tmp_path)
        script = root / "experiments" / "train_optout2.py"
        _write(script, """
            import argparse, torch
            p = argparse.ArgumentParser()
            p.add_argument("--no-auth-eval-on-best", action="store_true")
            torch.save(model.state_dict(), "renderer_best.pt")
        """)
        v = _scan_training_script_for_auth_eval(script, root)
        assert v == [], v

    def test_save_renderer_with_helper_call_passes(self, tmp_path: Path) -> None:
        """Direct call to `run_auth_eval(...)` satisfies the rule."""
        root = _stub_repo(tmp_path)
        script = root / "experiments" / "train_helper.py"
        _write(script, """
            import torch
            from tac.auth_eval import run_auth_eval
            def go():
                torch.save(model.state_dict(), "renderer_best.pt")
                run_auth_eval("renderer_best.pt")
        """)
        v = _scan_training_script_for_auth_eval(script, root)
        assert v == [], v

    def test_save_renderer_imports_but_never_calls_auth_eval_flags(
        self, tmp_path: Path,
    ) -> None:
        """Dead-import-class: importing auth_eval but never calling it is
        STILL a violation — the import alone does not run the eval."""
        root = _stub_repo(tmp_path)
        script = root / "experiments" / "train_dead_import.py"
        _write(script, """
            import torch
            from tac.auth_eval import run_auth_eval  # dead import
            def go():
                torch.save(model.state_dict(), "renderer_best.pt")
                # forgot to call run_auth_eval(...) — still violates
        """)
        v = _scan_training_script_for_auth_eval(script, root)
        assert len(v) == 1, v
        assert "dead import" in v[0].lower() or "imports" in v[0].lower()

    def test_save_renderer_no_auth_eval_at_all_flags(
        self, tmp_path: Path,
    ) -> None:
        """No reference at all: violation."""
        root = _stub_repo(tmp_path)
        script = root / "experiments" / "train_silent.py"
        _function_body = """
            import torch
            def go():
                torch.save(model.state_dict(), "renderer_best.pt")
        """
        _write(script, _function_body)
        v = _scan_training_script_for_auth_eval(script, root)
        assert len(v) == 1, v
        assert "auth_eval" in v[0].lower() or "auth eval" in v[0].lower()

    def test_save_via_pathlib_join_with_renderer_token_flags(
        self, tmp_path: Path,
    ) -> None:
        """`output_dir / "renderer_fp4.bin"` BinOp — AST walker descends into
        BinOp.right.constant.value. Must catch this real-world form."""
        root = _stub_repo(tmp_path)
        script = root / "experiments" / "train_pathlib.py"
        _write(script, """
            import torch
            from pathlib import Path
            def go():
                output_dir = Path("/tmp")
                torch.save(model, output_dir / "renderer_fp4.bin")
        """)
        v = _scan_training_script_for_auth_eval(script, root)
        assert len(v) == 1, (
            f"`output_dir / 'renderer_fp4.bin'` BinOp must be detected as "
            f"a renderer save — the constant is a child of BinOp.right; "
            f"got {v}"
        )

    def test_fstring_renderer_path_flags(self, tmp_path: Path) -> None:
        """f-string renderer paths: `f'{out}/renderer_ep{epoch}.pt'`. The
        AST walker must descend into JoinedStr to find the renderer token."""
        root = _stub_repo(tmp_path)
        script = root / "experiments" / "train_fstring.py"
        _write(script, """
            import torch
            def go(epoch, out):
                torch.save(model, f"{out}/renderer_ep{epoch}.pt")
        """)
        v = _scan_training_script_for_auth_eval(script, root)
        assert len(v) == 1, v

    def test_save_renderer_with_main_call_passes(self, tmp_path: Path) -> None:
        """`auth_eval_renderer.main()` direct invocation satisfies."""
        root = _stub_repo(tmp_path)
        script = root / "experiments" / "train_main_call.py"
        _write(script, """
            import torch
            import auth_eval_renderer
            def go():
                torch.save(model, "renderer_best.pt")
                auth_eval_renderer.main()
        """)
        v = _scan_training_script_for_auth_eval(script, root)
        assert v == [], v


# ─── codex R5-3 #4: preflight_all wires every meta-bug check ────────────────


class TestPreflightAllInvokesMetaBugChecks:
    """Source-grep regression: pin that preflight_all() actually invokes
    every meta-bug check function. Without this test, a future refactor
    could silently drop a check from preflight_all() — operators would
    get zero protection while believing it was wired.
    """

    def test_preflight_all_invokes_all_meta_bug_checks(self) -> None:
        import inspect
        from tac import preflight as pf

        # Source-grep preflight_all to verify each check is referenced.
        src = inspect.getsource(pf.preflight_all)

        required_checks = [
            "check_no_mps_fallback_default",
            "check_shell_set_e_present",
            "check_no_shell_zip_binary",
            "check_no_pipefail_grep_q_trap",
            "check_no_eval_roundtrip_false",
            "check_no_scorer_load_at_inflate",
            "check_training_scripts_have_auth_eval",
            "check_no_disable_eval_roundtrip_flag",
        ]
        missing = [c for c in required_checks if c not in src]
        assert missing == [], (
            f"preflight_all() does not invoke {missing}. "
            f"codex R5-3 #4: every meta-bug check must be wired into "
            f"preflight_all (warn-only is OK, but must be called)."
        )

    def test_preflight_all_calls_meta_checks_warn_only(self) -> None:
        """Pin that the wiring is currently strict=False (warn-only) — the
        live codebase has 30+ violations across these checks; flipping any
        to strict before fixing them would break Lane A. When an operator
        fixes a check's violations to zero and is ready to flip it to
        strict, this test must be updated alongside that change.
        """
        import inspect
        from tac import preflight as pf

        src = inspect.getsource(pf.preflight_all)
        # Each check must currently be invoked with strict=False.
        # If a check is later promoted to strict, the operator must update
        # this assertion to reflect the partition.
        meta_checks = [
            "check_no_mps_fallback_default",
            "check_shell_set_e_present",
            "check_no_shell_zip_binary",
            "check_no_pipefail_grep_q_trap",
            "check_no_eval_roundtrip_false",
            "check_no_scorer_load_at_inflate",
            "check_training_scripts_have_auth_eval",
            "check_no_disable_eval_roundtrip_flag",
        ]
        for chk in meta_checks:
            # Find the line invoking this check and confirm strict=False.
            for line in src.splitlines():
                if chk + "(" in line and "def " not in line:
                    assert "strict=False" in line, (
                        f"{chk} must be invoked with strict=False in "
                        f"preflight_all (warn-only); found: {line.strip()}. "
                        f"If you intend to flip to strict, update this "
                        f"test to whitelist {chk}."
                    )
                    break
