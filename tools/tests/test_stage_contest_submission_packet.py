"""Tests for tools/stage_contest_submission_packet.py.

The stager shipped untested and load-bearing: it is the tool whose receipt is
the identity proof that a staged packet IS the evaluated tree. Both the
generation-5 freeze checklist and the pq4 handoff named "the stager has no
tests" as an owed item, and only its author had read it.

Covered here: the identity proof (rows re-hashed after copy, tree hash
re-derived from staged rows), the fail-closed paths (content drift, missing
row, pre-existing output dir) including the invariant that a failed stage
leaves NO directory behind, the document-staging surface added for the
licence/notices/compression-script packet layer, and the constant-drift guard
that keeps this tool and packet_census_guard.py agreeing about what "declared"
means.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

_TOOLS = Path(__file__).resolve().parents[1]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, _TOOLS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


stager = _load("stage_contest_submission_packet")
guard = _load("packet_census_guard")


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _source_tree(tmp_path: Path, contents: dict[str, bytes], archive: bytes = b"ARCHIVE") -> Path:
    src = tmp_path / "src"
    src.mkdir()
    for rel, payload in contents.items():
        target = src / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
    (src / "archive.zip").write_bytes(archive)
    return src


def _receipt(tmp_path: Path, contents: dict[str, bytes], *, tree_sha: str | None = None) -> Path:
    """Build an auth-eval receipt whose tree hash is the real derived one."""
    rows = [
        {"relative_path": rel, "bytes": len(payload), "sha256": _sha(payload)}
        for rel, payload in sorted(contents.items())
    ]
    manifest: dict = {
        "files": rows,
        "external_dependency_roots": [],
        "repo_local_tac_import_manifest": {"runtime_root_name": "submission_dir"},
        "upstream_evaluate_py": None,
    }
    manifest["runtime_tree_sha256"] = tree_sha or stager.rederive_tree_sha256(manifest, rows)
    path = tmp_path / "contest_auth_eval.json"
    path.write_text(json.dumps({"provenance": {"inflate_runtime_manifest": manifest}}))
    return path


# --------------------------------------------------------------------------
# the identity proof
# --------------------------------------------------------------------------


def test_stage_proves_tree_identity_and_rehashes_every_row(tmp_path):
    contents = {"inflate.py": b"print(1)\n", "runtime/mod.py": b"X = 2\n"}
    src = _source_tree(tmp_path, contents)
    receipt = _receipt(tmp_path, contents)

    record = stager.stage(
        auth_eval_json=receipt,
        source_runtime_dir=src,
        out_dir=tmp_path / "out",
        expected_archive_sha256=_sha(b"ARCHIVE"),
        expected_archive_size_bytes=len(b"ARCHIVE"),
    )

    assert record["verdict"] == "STAGED_TREE_PROVED_IDENTICAL_TO_EVALUATED_TREE"
    assert record["runtime_files_verified"] == record["runtime_files_declared"] == 2
    assert (
        record["runtime_tree_sha256_rederived_from_staged"] == record["runtime_tree_sha256"]
    )
    assert (tmp_path / "out" / "runtime" / "mod.py").read_bytes() == b"X = 2\n"


def test_receipt_carries_the_enumerated_rows_scope_rule(tmp_path):
    """The pin is over enumerated rows; a re-validator must not re-walk the dir."""
    contents = {"inflate.py": b"a\n"}
    record = stager.stage(
        auth_eval_json=_receipt(tmp_path, contents),
        source_runtime_dir=_source_tree(tmp_path, contents),
        out_dir=tmp_path / "out",
    )
    scope = record["runtime_tree_sha256_scope"]
    assert "ENUMERATED" in scope
    assert "NOT over a fresh recursive walk" in scope


def test_content_drift_refuses_and_leaves_no_directory(tmp_path):
    contents = {"inflate.py": b"a\n"}
    receipt = _receipt(tmp_path, contents)
    src = _source_tree(tmp_path, {"inflate.py": b"DIFFERENT\n"})

    with pytest.raises(stager.StagingError, match="differs from the evaluated manifest"):
        stager.stage(
            auth_eval_json=receipt, source_runtime_dir=src, out_dir=tmp_path / "out"
        )
    assert not (tmp_path / "out").exists()


def test_missing_manifest_row_refuses(tmp_path):
    contents = {"inflate.py": b"a\n", "runtime/mod.py": b"b\n"}
    receipt = _receipt(tmp_path, contents)
    src = _source_tree(tmp_path, {"inflate.py": b"a\n"})

    with pytest.raises(stager.StagingError, match="absent from the source tree"):
        stager.stage(
            auth_eval_json=receipt, source_runtime_dir=src, out_dir=tmp_path / "out"
        )
    assert not (tmp_path / "out").exists()


def test_existing_out_dir_is_never_overwritten(tmp_path):
    contents = {"inflate.py": b"a\n"}
    out = tmp_path / "out"
    out.mkdir()
    with pytest.raises(stager.StagingError, match="Staging never overwrites"):
        stager.stage(
            auth_eval_json=_receipt(tmp_path, contents),
            source_runtime_dir=_source_tree(tmp_path, contents),
            out_dir=out,
        )


def test_archive_sha_mismatch_refuses(tmp_path):
    contents = {"inflate.py": b"a\n"}
    with pytest.raises(stager.StagingError, match="archive sha mismatch"):
        stager.stage(
            auth_eval_json=_receipt(tmp_path, contents),
            source_runtime_dir=_source_tree(tmp_path, contents),
            out_dir=tmp_path / "out",
            expected_archive_sha256="f" * 64,
        )
    assert not (tmp_path / "out").exists()


def test_tree_sha_disagreement_refuses(tmp_path):
    contents = {"inflate.py": b"a\n"}
    receipt = _receipt(tmp_path, contents, tree_sha="c" * 64)
    with pytest.raises(stager.StagingError, match="is NOT the evaluated tree"):
        stager.stage(
            auth_eval_json=receipt,
            source_runtime_dir=_source_tree(tmp_path, contents),
            out_dir=tmp_path / "out",
        )


# --------------------------------------------------------------------------
# document staging
# --------------------------------------------------------------------------


def test_documents_are_staged_renamed_and_receipted(tmp_path):
    contents = {"inflate.py": b"a\n"}
    doc = tmp_path / "README_PUBLIC.md"
    doc.write_bytes(b"# packet\n")

    record = stager.stage(
        auth_eval_json=_receipt(tmp_path, contents),
        source_runtime_dir=_source_tree(tmp_path, contents),
        out_dir=tmp_path / "out",
        docs=[(doc, "README.md")],
    )

    assert (tmp_path / "out" / "README.md").read_bytes() == b"# packet\n"
    assert record["staged_document_count"] == 1
    row = record["staged_documents"][0]
    assert row["relative_path"] == "README.md"
    assert row["sha256"] == _sha(b"# packet\n")
    assert row["bytes"] == len(b"# packet\n")


def test_undeclared_document_destination_refuses(tmp_path):
    """A destination the census guard does not know would census as undeclared."""
    contents = {"inflate.py": b"a\n"}
    doc = tmp_path / "notes.md"
    doc.write_bytes(b"x")
    with pytest.raises(stager.StagingError, match="not in DECLARED_NON_RUNTIME"):
        stager.stage(
            auth_eval_json=_receipt(tmp_path, contents),
            source_runtime_dir=_source_tree(tmp_path, contents),
            out_dir=tmp_path / "out",
            docs=[(doc, "SURPRISE.md")],
        )
    assert not (tmp_path / "out").exists()


def test_document_colliding_with_a_runtime_row_refuses(tmp_path):
    contents = {"compress.py": b"real runtime\n"}
    doc = tmp_path / "other.py"
    doc.write_bytes(b"impostor\n")
    with pytest.raises(stager.StagingError, match="collides with a runtime manifest row"):
        stager.stage(
            auth_eval_json=_receipt(tmp_path, contents),
            source_runtime_dir=_source_tree(tmp_path, contents),
            out_dir=tmp_path / "out",
            docs=[(doc, "compress.py")],
        )


def test_missing_document_source_refuses_before_creating_output(tmp_path):
    contents = {"inflate.py": b"a\n"}
    with pytest.raises(stager.StagingError, match="--doc source does not exist"):
        stager.stage(
            auth_eval_json=_receipt(tmp_path, contents),
            source_runtime_dir=_source_tree(tmp_path, contents),
            out_dir=tmp_path / "out",
            docs=[(tmp_path / "nope.md", "README.md")],
        )
    assert not (tmp_path / "out").exists()


def test_duplicate_document_destination_refuses(tmp_path):
    contents = {"inflate.py": b"a\n"}
    a, b = tmp_path / "a.md", tmp_path / "b.md"
    a.write_bytes(b"a")
    b.write_bytes(b"b")
    with pytest.raises(stager.StagingError, match="given twice"):
        stager.stage(
            auth_eval_json=_receipt(tmp_path, contents),
            source_runtime_dir=_source_tree(tmp_path, contents),
            out_dir=tmp_path / "out",
            docs=[(a, "README.md"), (b, "README.md")],
        )


# --------------------------------------------------------------------------
# --doc spec parsing
# --------------------------------------------------------------------------


def test_doc_spec_splits_on_the_last_equals():
    src, dest = stager.parse_doc_spec("dir/a=b.md=README.md")
    assert src == Path("dir/a=b.md")
    assert dest == "README.md"


@pytest.mark.parametrize(
    "spec",
    ["no-equals-here", "=README.md", "src.md=", "src.md=/abs/README.md", "src.md=../README.md"],
)
def test_doc_spec_rejects_malformed_specs(spec):
    with pytest.raises(stager.StagingError):
        stager.parse_doc_spec(spec)


# --------------------------------------------------------------------------
# the drift guard between the two tools
# --------------------------------------------------------------------------


def test_declared_non_runtime_is_identical_in_both_tools():
    """Both tools must agree about what "declared" means, or a packet that
    stages cleanly censuses as contaminated."""
    assert stager.DECLARED_NON_RUNTIME == guard.DECLARED_NON_RUNTIME


def test_licence_notices_and_compression_script_are_declared():
    for name in ("LICENSE", "THIRD_PARTY_NOTICES.md", "MANIFEST.sha256", "compress.py", "COMPRESS.md"):
        assert name in stager.DECLARED_NON_RUNTIME
        assert name in guard.DECLARED_NON_RUNTIME
