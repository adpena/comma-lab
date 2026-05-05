from __future__ import annotations

import importlib.util
import sys
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
