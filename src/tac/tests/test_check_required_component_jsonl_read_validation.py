from __future__ import annotations

from pathlib import Path

import pytest

from tac.preflight import (
    PreflightError,
    check_no_unvalidated_required_component_jsonl_readers,
)


def _write(root: Path, rel: str, body: str) -> Path:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    return path


def _check(root: Path, **kwargs: object) -> list[str]:
    return check_no_unvalidated_required_component_jsonl_readers(
        repo_root=root, **kwargs
    )


def test_raw_literal_reader_is_reported_and_strict_refuses(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "src/tac/bad_reader.py",
        "from pathlib import Path\nrows = Path('.omx/state/required_component_ledger.jsonl').read_text()\n",
    )

    violations = _check(tmp_path)
    assert len(violations) == 1
    assert "bad_reader.py:2" in violations[0]
    with pytest.raises(PreflightError, match="unvalidated_required_component"):
        _check(tmp_path, strict=True)


def test_canonical_helper_consumer_passes(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "tools/consumer.py",
        "from tac.witness_dsl.activation_ledger import read_required_components\n"
        "rows = read_required_components()\n",
    )
    assert _check(tmp_path) == []


def test_canonical_owner_passes(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "src/tac/witness_dsl/activation_ledger.py",
        "from pathlib import Path\n"
        "REQUIRED_COMPONENT_PATH = Path('required_component_ledger.jsonl')\n"
        "rows = REQUIRED_COMPONENT_PATH.read_text()\n",
    )
    assert _check(tmp_path) == []


def test_substantive_same_line_waiver_passes(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "experiments/waived.py",
        "from pathlib import Path\n"
        "rows = Path('required_component_ledger.jsonl').read_text()  "
        "# REQUIRED_COMPONENT_JSONL_READ_OK:migration comparison retains row validation below\n",
    )
    assert _check(tmp_path) == []


@pytest.mark.parametrize(
    "body",
    [
        "# REQUIRED_COMPONENT_JSONL_READ_OK:valid rationale but wrong line\n"
        "rows = open('required_component_ledger.jsonl').read()\n",
        "rows = open('required_component_ledger.jsonl').read()  "
        "# REQUIRED_COMPONENT_JSONL_READ_OK:<rationale>\n",
        "rows = open('required_component_ledger.jsonl').read()  "
        "# REQUIRED_COMPONENT_JSONL_READ_OK:short\n",
    ],
)
def test_nearby_placeholder_and_short_waivers_do_not_pass(
    tmp_path: Path, body: str
) -> None:
    _write(tmp_path, "tools/bad_waiver.py", body)
    assert len(_check(tmp_path)) == 1


def test_imported_path_symbol_raw_open_is_reported(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "src/tac/imported_path_reader.py",
        "from tac.witness_dsl.activation_ledger import REQUIRED_COMPONENT_PATH\n"
        "with REQUIRED_COMPONENT_PATH.open() as handle:\n"
        "    rows = list(handle)\n",
    )
    violations = _check(tmp_path)
    assert len(violations) == 1
    assert "imported_path_reader.py:2" in violations[0]


def test_verbose_denominator_is_nonvacuous_and_warn_only_returns(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _write(tmp_path, "src/tac/clean.py", "VALUE = 1\n")
    _write(
        tmp_path,
        "tools/raw.py",
        "open('required_component_ledger.jsonl').read()\n",
    )
    violations = _check(tmp_path, verbose=True, strict=False)
    output = capsys.readouterr().out
    assert len(violations) == 1
    assert "scanned 2 production Python file(s)" in output
    assert "parsed 2" in output


def test_results_and_test_files_are_outside_production_scope(tmp_path: Path) -> None:
    body = "open('required_component_ledger.jsonl').read()\n"
    _write(tmp_path, "experiments/results/raw.py", body)
    _write(tmp_path, "src/tac/tests/test_raw.py", body)
    _write(tmp_path, "experiments/clean.py", "VALUE = 1\n")
    assert _check(tmp_path) == []
