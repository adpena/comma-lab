"""Bind the PR130 runtime-closure DECLARATION to the CODE it describes.

Review-round finding (2026-08-09, MAIN, on MAIN's own fx5 landing): after enriching
`runtime-dependencies.json` to enumerate all three third-party packages, a consumer sweep
found the file is read by ZERO executable code. `inflate.sh` independently hardcodes
`EXPECTED_CONSTRICTION_VERSION=0.5.0` and the asserted-provided tuple `("numpy", "torch")`.
Two sources of truth, no binding — the declaration got more complete without getting more
binding, which is the config-orphan genus, not a cure.

These tests are the binding. They deliberately do NOT make the shipping entrypoint parse
JSON at decode time: that would add a failure mode to the decode path for no benefit. The
contract is enforced here instead, where a drifting edit fails loudly and costs nothing at
runtime.

The strongest of the three is `test_imported_by_matches_actual_module_imports`: it derives
the closure from the receiver source with `ast` and compares it to the declaration, so a
future receiver module that adds an import cannot silently escape the manifest. That is the
exact defect FX5's bare-image Linux run surfaced by accident.
"""

from __future__ import annotations

import ast
import json
import pathlib
import re

import pytest

TREE = (
    pathlib.Path(__file__).resolve().parent.parent
    / "src"
    / "tac"
    / "pr130_runtime"
    / "fx1_runtime_tree"
)
MANIFEST_PATH = TREE / "runtime-dependencies.json"
ENTRYPOINT_PATH = TREE / "inflate.sh"

# Anything importable from the CPython standard library is not a declared dependency.
# Kept explicit rather than probed, so the test states what it assumes.
_STDLIB = {
    "__future__",
    "collections",
    "dataclasses",
    "heapq",
    "importlib",
    "lzma",
    "math",
    "pathlib",
    "struct",
    "sys",
    "time",
    "typing",
}


@pytest.fixture(scope="module")
def manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text())


@pytest.fixture(scope="module")
def entrypoint_text() -> str:
    return ENTRYPOINT_PATH.read_text()


def _declared(manifest: dict) -> dict[str, dict]:
    return {dep["name"]: dep for dep in manifest["dependencies"]}


def _local_module_names() -> set[str]:
    return {p.stem for p in TREE.glob("*.py")}


def _third_party_imports() -> dict[str, set[str]]:
    """Derive {package: {modules importing it}} from the receiver source via AST."""
    local = _local_module_names()
    found: dict[str, set[str]] = {}
    for path in sorted(TREE.glob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                # level > 0 is an explicit relative import: local by construction.
                names = [node.module] if node.module and node.level == 0 else []
            else:
                continue
            for name in names:
                root = name.split(".")[0]
                if root in _STDLIB or root in local:
                    continue
                found.setdefault(root, set()).add(path.name)
    return found


def test_manifest_declares_every_third_party_import(manifest: dict) -> None:
    """No import may escape the manifest — the FX5 bare-image defect, structurally."""
    actual = set(_third_party_imports())
    declared = set(_declared(manifest))
    assert actual == declared, (
        f"closure drift: imported-not-declared={sorted(actual - declared)}, "
        f"declared-not-imported={sorted(declared - actual)}"
    )


def test_imported_by_matches_actual_module_imports(manifest: dict) -> None:
    """Each dependency's `imported_by` must be the real, complete importer set."""
    actual = _third_party_imports()
    for name, dep in _declared(manifest).items():
        assert set(dep["imported_by"]) == actual[name], (
            f"{name}.imported_by is stale: declared={sorted(dep['imported_by'])}, "
            f"actual={sorted(actual[name])}"
        )


def test_closure_provenance_denominator_matches_the_tree(manifest: dict) -> None:
    """The recorded denominator must equal what is actually on disk."""
    prov = manifest["closure_provenance"]
    assert prov["denominator"] == len(list(TREE.glob("*.py")))
    assert prov["third_party_packages"] == len(manifest["dependencies"])


def test_entrypoint_version_matches_declared_version(
    manifest: dict, entrypoint_text: str
) -> None:
    """inflate.sh's hardcoded version is the runtime authority; bind it to the manifest."""
    match = re.search(
        r"^EXPECTED_CONSTRICTION_VERSION=(\S+)$", entrypoint_text, re.MULTILINE
    )
    assert match is not None, "inflate.sh no longer declares EXPECTED_CONSTRICTION_VERSION"
    assert match.group(1) == _declared(manifest)["constriction"]["version"]


def test_entrypoint_asserts_exactly_the_provided_dependencies(
    manifest: dict, entrypoint_text: str
) -> None:
    """The asserted-not-installed set in the script must equal the declared set."""
    declared_provided = {
        name
        for name, dep in _declared(manifest).items()
        if dep["provisioning"] == "contest_runtime_provided_asserted_not_installed"
    }
    match = re.search(
        r'missing = \[name for name in \(([^)]*)\)', entrypoint_text
    )
    assert match is not None, "inflate.sh no longer carries the assert_provided_deps tuple"
    asserted = set(re.findall(r'"([^"]+)"', match.group(1)))
    assert asserted == declared_provided, (
        f"script asserts {sorted(asserted)} but manifest declares "
        f"{sorted(declared_provided)} as contest-runtime-provided"
    )


def test_self_installed_dependency_is_never_in_the_assert_set(
    manifest: dict, entrypoint_text: str
) -> None:
    """A self-installed dep must NOT be asserted-absent: that would refuse before installing.

    Positive-control direction: this is the failure mode the exit-68 assert would cause if
    someone moved `constriction` into the provided set — the entrypoint would refuse on a
    clean host instead of bootstrapping, which is precisely what FX1 cured.
    """
    self_installed = {
        name
        for name, dep in _declared(manifest).items()
        if dep["provisioning"] == "self_installed_by_entrypoint"
    }
    match = re.search(r'missing = \[name for name in \(([^)]*)\)', entrypoint_text)
    assert match is not None
    asserted = set(re.findall(r'"([^"]+)"', match.group(1)))
    assert not (asserted & self_installed), (
        f"{sorted(asserted & self_installed)} is self-installed but also asserted-absent; "
        "the entrypoint would refuse on a clean host instead of bootstrapping"
    )
