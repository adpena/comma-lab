"""Focused tests for the public repository path census and classifier."""

from __future__ import annotations

from tools.public_repo_hygiene_census import _fleet_inventory_tokens, classify_path


def test_class_a_source_path_without_pin() -> None:
    category, reason, consumers = classify_path(
        "src/tac/example.py",
        content_sha256="a" * 64,
        pin_consumers={},
    )
    assert (category, reason, consumers) == (
        "a",
        "source_tool_experiment_or_script",
        [],
    )


def test_sha_pin_takes_precedence_over_source_directory() -> None:
    digest = "b" * 64
    category, reason, consumers = classify_path(
        "tools/example.py",
        content_sha256=digest,
        pin_consumers={digest: ["reports/release_manifest.json"]},
    )
    assert category == "c"
    assert reason == "content_sha256_consumed_by_public_seal_or_receipt"
    assert consumers == ["reports/release_manifest.json"]


def test_protected_surface_is_class_c_even_without_discovered_pin() -> None:
    category, reason, consumers = classify_path(
        "submissions/robust_current/jg5_sub015_runtime/inflate.py",
        content_sha256="c" * 64,
        pin_consumers={},
    )
    assert category == "c"
    assert reason == "protected_no_edit_surface"
    assert consumers == ["charter:ddm_sw1"]


def test_fleet_inventory_reads_only_host_and_ip_coordinates(tmp_path) -> None:
    inventory = tmp_path / "fleet.local.toml"
    inventory.write_text(
        """
[hosts.primary]
host = "private-host"
ip = "192.0.2.15"
role = "training"
notes = "not a coordinate"
""".strip()
        + "\n",
        encoding="utf-8",
    )
    assert _fleet_inventory_tokens(inventory) == [
        ("host", "private-host"),
        ("ip", "192.0.2.15"),
    ]
