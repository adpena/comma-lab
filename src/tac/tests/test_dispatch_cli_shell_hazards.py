from __future__ import annotations

import importlib.util
import sys
import textwrap
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
HELPER = REPO / "tools" / "check_dispatch_cli_shell_hazards.py"


def _load_helper():
    spec = importlib.util.spec_from_file_location("dispatch_cli_shell_hazards_test", HELPER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _kinds(hazards):
    return {hazard.kind for hazard in hazards}


def test_scanner_catches_adjudicator_flags_passed_to_lightning_launcher(tmp_path: Path) -> None:
    helper = _load_helper()
    script = tmp_path / "scripts" / "bad.sh"
    script.parent.mkdir()
    script.write_text(
        """
        #!/usr/bin/env bash
        python scripts/launch_lightning_batch_job.py exact-eval \\
          --job-name j \\
          --archive a.zip \\
          --required-device cuda \\
          --required-samples 600
        """.lstrip(),
        encoding="utf-8",
    )
    hazards = helper.scan_paths(tmp_path, scan_paths=("scripts",))
    assert _kinds(hazards) == {"stale_lightning_launcher_flag"}
    assert {hazard.message.split(" passed", 1)[0] for hazard in hazards} == {
        "--required-device",
        "--required-samples",
    }


def test_scanner_does_not_flag_adjudicator_flags_on_adjudicator_cli(tmp_path: Path) -> None:
    helper = _load_helper()
    script = tmp_path / "scripts" / "good.sh"
    script.parent.mkdir()
    script.write_text(
        """
        #!/usr/bin/env bash
        python scripts/adjudicate_contest_auth_eval.py \\
          --contest-json contest_auth_eval.json \\
          --required-device cuda \\
          --required-samples 600
        """.lstrip(),
        encoding="utf-8",
    )
    assert helper.scan_paths(tmp_path, scan_paths=("scripts",)) == []


def test_scanner_catches_known_typo_flag(tmp_path: Path) -> None:
    helper = _load_helper()
    script = tmp_path / "scripts" / "typo.sh"
    script.parent.mkdir()
    script.write_text("python tools/x.py --rmote lightning\n", encoding="utf-8")
    hazards = helper.scan_paths(tmp_path, scan_paths=("scripts",))
    assert len(hazards) == 1
    assert hazards[0].kind == "known_typo_flag"


def test_scanner_catches_zsh_path_variable_without_flagging_python_heredoc(tmp_path: Path) -> None:
    helper = _load_helper()
    snippet = tmp_path / "docs" / "ops.md"
    snippet.parent.mkdir()
    snippet.write_text(
        """
        ```zsh
        for path in artifacts/*.json; do
          dirname "$path"
        done
        ```

        ```bash
        python - <<'PY'
        for path in sorted(root.rglob("*")):
            print(path)
        PY
        ```
        """.lstrip(),
        encoding="utf-8",
    )
    hazards = helper.scan_paths(tmp_path, scan_paths=("docs",))
    assert len(hazards) == 1
    assert hazards[0].kind == "zsh_path_special_variable"


def test_scanner_catches_local_find_printf_but_allows_remote_workspace_find(tmp_path: Path) -> None:
    helper = _load_helper()
    script = tmp_path / "scripts" / "finds.sh"
    script.parent.mkdir()
    script.write_text(
        """
        #!/usr/bin/env bash
        find . -name '*.json' -printf '%p\\n'
        ssh root@host "find /workspace/pact -name heartbeat.log -printf '%T@\\n'"
        """.lstrip(),
        encoding="utf-8",
    )
    hazards = helper.scan_paths(tmp_path, scan_paths=("scripts",))
    assert len(hazards) == 1
    assert hazards[0].kind == "macos_find_printf"


# ── dispatch_local_path_leak metabug class (Lightning catastrophe 2026-05-05) ──


def test_scanner_catches_str_repo_root_passed_to_repo_dir(tmp_path: Path) -> None:
    """The exact bug pattern that caused the Lightning $1.55 catastrophe."""
    helper = _load_helper()
    dispatcher = tmp_path / "tools" / "lightning_dispatch_buggy.py"
    dispatcher.parent.mkdir()
    dispatcher.write_text(textwrap.dedent('''
        import subprocess
        from pathlib import Path
        REPO_ROOT = Path(__file__).resolve().parents[1]
        subprocess.run([
            "python", "scripts/launch_lightning_batch_job.py", "exact-eval",
            "--repo-dir", str(REPO_ROOT),
            "--upstream-dir", str(REPO_ROOT / "upstream"),
            "--archive", str(REPO_ROOT / "experiments" / "results" / "x.zip"),
        ])
    ''').lstrip(), encoding="utf-8")
    hazards = helper.scan_paths(tmp_path, scan_paths=("tools",))
    kinds = _kinds(hazards)
    assert "dispatch_local_path_leak" in kinds
    flagged_flags = sorted(h.message.split(" value", 1)[0] for h in hazards if h.kind == "dispatch_local_path_leak")
    assert flagged_flags == ["--archive", "--repo-dir", "--upstream-dir"]


def test_scanner_allows_args_remote_pact_and_relative_to(tmp_path: Path) -> None:
    """The post-fix shape of tools/lightning_dispatch_pr106_stack.py."""
    helper = _load_helper()
    dispatcher = tmp_path / "tools" / "lightning_dispatch_good.py"
    dispatcher.parent.mkdir()
    dispatcher.write_text(textwrap.dedent('''
        import subprocess
        from pathlib import Path
        REPO_ROOT = Path(__file__).resolve().parents[1]
        DEFAULT_REMOTE_PACT = "/teamspace/studios/this_studio/pact"
        def submit(args, archive, inflate_sh, remote_pact=DEFAULT_REMOTE_PACT):
            subprocess.run([
                "python", "scripts/launch_lightning_batch_job.py", "exact-eval",
                "--repo-dir", remote_pact,
                "--upstream-dir", f"{remote_pact}/upstream",
                "--archive", str(archive.relative_to(REPO_ROOT)),
                "--inflate-sh", str(inflate_sh.relative_to(REPO_ROOT)),
            ])
    ''').lstrip(), encoding="utf-8")
    hazards = [h for h in helper.scan_paths(tmp_path, scan_paths=("tools",))
               if h.kind == "dispatch_local_path_leak"]
    assert hazards == [], f"unexpected leak hazards: {hazards}"


def test_scanner_allows_remote_root_string_literal(tmp_path: Path) -> None:
    helper = _load_helper()
    dispatcher = tmp_path / "tools" / "dispatch_modal_lane.py"
    dispatcher.parent.mkdir()
    dispatcher.write_text(textwrap.dedent('''
        import subprocess
        subprocess.run([
            "python", "remote.py",
            "--repo-dir", "/teamspace/studios/this_studio/pact",
            "--archive", "experiments/results/x.zip",
        ])
    ''').lstrip(), encoding="utf-8")
    hazards = [h for h in helper.scan_paths(tmp_path, scan_paths=("tools",))
               if h.kind == "dispatch_local_path_leak"]
    assert hazards == []


def test_scanner_catches_hardcoded_users_path_literal(tmp_path: Path) -> None:
    helper = _load_helper()
    dispatcher = tmp_path / "tools" / "lightning_dispatch_lit.py"
    dispatcher.parent.mkdir()
    dispatcher.write_text(textwrap.dedent('''
        import subprocess
        subprocess.run([
            "python", "scripts/launch_lightning_batch_job.py", "exact-eval",
            "--repo-dir", "/Users/adpena/Projects/pact",
        ])
    ''').lstrip(), encoding="utf-8")
    hazards = [h for h in helper.scan_paths(tmp_path, scan_paths=("tools",))
               if h.kind == "dispatch_local_path_leak"]
    assert len(hazards) == 1
    assert "/Users/adpena/Projects/pact" in hazards[0].message


def test_scanner_only_inspects_dispatcher_files(tmp_path: Path) -> None:
    """Ordinary tool scripts that pass --archive str(REPO_ROOT) are NOT flagged."""
    helper = _load_helper()
    other = tmp_path / "tools" / "audit_archive.py"
    other.parent.mkdir()
    other.write_text(textwrap.dedent('''
        import subprocess
        from pathlib import Path
        REPO_ROOT = Path(__file__).resolve().parents[1]
        subprocess.run([
            "python", "tools/audit_archive_helper.py",
            "--archive", str(REPO_ROOT / "x.zip"),
        ])
    ''').lstrip(), encoding="utf-8")
    hazards = [h for h in helper.scan_paths(tmp_path, scan_paths=("tools",))
               if h.kind == "dispatch_local_path_leak"]
    assert hazards == []


# ── remote_script_local_pythonpath_leak ──


def test_scanner_catches_local_pythonpath_in_remote_lane_script(tmp_path: Path) -> None:
    helper = _load_helper()
    script = tmp_path / "scripts" / "remote_lane_buggy.sh"
    script.parent.mkdir()
    script.write_text(
        '''
        #!/usr/bin/env bash
        export PYTHONPATH="src:upstream:/Users/adpena/projects/pact:${PYTHONPATH:-}"
        '''.lstrip(),
        encoding="utf-8",
    )
    hazards = [h for h in helper.scan_paths(tmp_path, scan_paths=("scripts",))
               if h.kind == "remote_script_local_pythonpath_leak"]
    assert len(hazards) == 1
    assert "/Users/adpena/projects/pact" in hazards[0].message


def test_scanner_allows_canonical_pythonpath(tmp_path: Path) -> None:
    helper = _load_helper()
    script = tmp_path / "scripts" / "remote_lane_good.sh"
    script.parent.mkdir()
    script.write_text(
        '''
        #!/usr/bin/env bash
        export PYTHONPATH="src:upstream:${PYTHONPATH:-}"
        '''.lstrip(),
        encoding="utf-8",
    )
    hazards = [h for h in helper.scan_paths(tmp_path, scan_paths=("scripts",))
               if h.kind == "remote_script_local_pythonpath_leak"]
    assert hazards == []
