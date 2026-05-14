# SPDX-License-Identifier: MIT
from scripts.kill_orphaned_mcp_processes import parse_ps_rows


def test_parse_ps_rows_selects_only_mcp_helpers() -> None:
    rows = [
        "101 /Users/adpena/.cargo/bin/rbx-studio-mcp --stdio",
        "102 npm exec chrome-devtools-mcp@latest --channel stable",
        "105 /bin/zsh -c 'npx chrome-devtools-mcp@latest --channel stable'",
        "106 node /Users/adpena/.cache/chrome-devtools-mcp/dist/index.js",
        "103 /bin/zsh -c ps -axo pid=,command=",
        "104 /usr/libexec/colorsyncd",
    ]

    matches = parse_ps_rows(rows, current_pid=999)

    assert [match.pid for match in matches] == [101, 102, 105, 106]
    assert [match.token for match in matches] == [
        "rbx-studio-mcp",
        "chrome-devtools-mcp",
        "chrome-devtools-mcp",
        "chrome-devtools-mcp",
    ]


def test_parse_ps_rows_skips_current_process() -> None:
    rows = [
        "201 /Users/adpena/.cargo/bin/rbx-studio-mcp --stdio",
        "202 chrome-devtools-mcp",
    ]

    matches = parse_ps_rows(rows, current_pid=201)

    assert [match.pid for match in matches] == [202]


def test_parse_ps_rows_ignores_mcp_audit_commands() -> None:
    rows = [
        "301 find /Users/adpena -iname *mcp* -o -iname *model.context*",
        "302 /bin/zsh -c find /Users/adpena -iname '*chrome-devtools-mcp*'",
        "303 rg -n chrome-devtools-mcp AGENTS.md scripts",
        "304 grep -R model.context /Users/adpena/.claude",
        "305 python -c 'print(\"chrome-devtools-mcp\")'",
    ]

    assert parse_ps_rows(rows, current_pid=999) == []


def test_parse_ps_rows_detects_python_module_helper() -> None:
    rows = [
        "401 python -m model.context --stdio",
        "402 /usr/bin/python3 -m model.context",
    ]

    matches = parse_ps_rows(rows, current_pid=999)

    assert [match.pid for match in matches] == [401, 402]
    assert [match.token for match in matches] == ["model.context", "model.context"]
