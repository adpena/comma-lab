# SPDX-License-Identifier: MIT
"""Controls for the SR3 carve-out: compress a live tree AROUND its live reads.

The carve-out exists so a PROTECTED tree (a live store other code still reads)
can have its bulk reclaimed without burying the exact paths those readers need.
Every control below is executed in both directions -- a carve-out that silently
failed to protect, or silently over-protected, would be worse than no carve-out.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]


def _load():
    path = REPO / "experiments" / "ddm_sr3_ap_certify_compress_reclaim.py"
    spec = importlib.util.spec_from_file_location("_sr3_carve_out", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def sr3():
    return _load()


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    root = tmp_path / "ddm_example_tree"
    (root / "retained" / "body").mkdir(parents=True)
    (root / "retained" / "body" / "BODY.bin").write_bytes(b"live body payload")
    (root / "retained" / "bodyguard").mkdir()          # prefix-sibling decoy
    (root / "retained" / "bodyguard" / "bulk.npy").write_bytes(b"bulk" * 64)
    (root / "retained" / "other.npy").write_bytes(b"other" * 64)
    (root / "caches").mkdir()
    (root / "caches" / "big.npy").write_bytes(b"cache" * 128)
    (root / "BODY_RESULT.json").write_bytes(b'{"ok": true}')
    return root


# --- the carve-out predicate: exact, descendant, and NOT prefix-sibling ---


def test_is_kept_matches_exact_and_descendants(sr3):
    keeps = ("retained/body",)
    assert sr3._is_kept("retained/body", keeps) is True
    assert sr3._is_kept("retained/body/BODY.bin", keeps) is True


def test_is_kept_does_not_match_a_prefix_sibling(sr3):
    """RED DIRECTION: 'retained/body' must not protect 'retained/bodyguard'.

    A naive ``startswith(keep)`` would silently retain an unrelated sibling --
    the carve-out would under-reclaim while reporting success.
    """
    keeps = ("retained/body",)
    assert sr3._is_kept("retained/bodyguard", keeps) is False
    assert sr3._is_kept("retained/bodyguard/bulk.npy", keeps) is False


# --- scan_tree: carve-outs are never manifested and never descended ---


def test_scan_tree_without_carve_outs_sees_everything(sr3, tree):
    paths = {row.path for row in sr3.scan_tree(tree)}
    assert "retained/body" in paths
    assert "retained/body/BODY.bin" in paths
    assert "BODY_RESULT.json" in paths


def test_scan_tree_excludes_carve_outs_and_their_contents(sr3, tree):
    keeps = ("retained/body", "BODY_RESULT.json")
    paths = {row.path for row in sr3.scan_tree(tree, keeps)}
    assert "retained/body" not in paths
    assert "retained/body/BODY.bin" not in paths
    assert "BODY_RESULT.json" not in paths
    # everything else still archives, including the prefix-sibling decoy
    assert "retained/bodyguard/bulk.npy" in paths
    assert "retained/other.npy" in paths
    assert "caches/big.npy" in paths


# --- validation refuses every unsafe carve-out shape ---


@pytest.mark.parametrize(
    "raw, fragment",
    [
        ("/abs/path", "tree-relative"),
        ("../escape", "unsafe"),
        ("retained/../../escape", "unsafe"),
        ("", "must not be empty"),
        ("no_such_dir", "does not exist"),
    ],
)
def test_validate_keep_paths_refuses_unsafe_shapes(sr3, tree, raw, fragment):
    with pytest.raises(sr3.CertifyError, match=fragment):
        sr3.validate_keep_paths(tree, [raw])


def test_validate_keep_paths_refuses_sr3_custody_names(sr3, tree):
    (tree / sr3.MANIFEST_NAME).write_bytes(b"{}")
    with pytest.raises(sr3.CertifyError, match="may not name SR3 custody"):
        sr3.validate_keep_paths(tree, [sr3.MANIFEST_NAME])


def test_validate_keep_paths_refuses_a_redundant_nested_carve_out(sr3, tree):
    with pytest.raises(sr3.CertifyError, match="already covered by"):
        sr3.validate_keep_paths(tree, ["retained", "retained/body"])


def test_validate_keep_paths_normalises_and_dedupes(sr3, tree):
    assert sr3.validate_keep_paths(tree, ["retained/body/", "retained/body"]) == (
        "retained/body",
    )


# --- the protection lift is gated, not deleted ---


def test_protected_tree_still_refuses_without_a_lift(sr3, tree, monkeypatch):
    monkeypatch.setattr(sr3, "AP_ROOT", tree.parent)
    monkeypatch.setattr(sr3, "PROTECTED_TREES", {tree.resolve()})
    with pytest.raises(sr3.CertifyError, match="explicitly protected live store"):
        sr3.validate_tree(str(tree))


def test_protected_tree_is_reachable_only_with_an_explicit_lift(sr3, tree, monkeypatch):
    monkeypatch.setattr(sr3, "AP_ROOT", tree.parent)
    monkeypatch.setattr(sr3, "PROTECTED_TREES", {tree.resolve()})
    assert sr3.validate_tree(str(tree), lift_protection="rationale") == tree.resolve()


# --- the reference scan: the machine-checked half of a lift ---


def _repo_with_reference(root: Path, tree_name: str, referenced: str) -> Path:
    (root / "experiments").mkdir(parents=True)
    (root / "experiments" / "reader.py").write_text(
        f'BODY = "/Volumes/APDataStore/pact/{tree_name}/{referenced}"\n', encoding="utf-8"
    )
    return root


def test_reference_scan_reports_a_reference_no_carve_out_covers(sr3, tmp_path, tree, monkeypatch):
    monkeypatch.setattr(sr3, "AP_ROOT", Path("/Volumes/APDataStore/pact"))
    repo = _repo_with_reference(tmp_path / "repo", tree.name, "retained/body/BODY.bin")
    violations = sr3.scan_live_references(repo, tree, keep_uncompressed=())
    assert [row["referenced_path"] for row in violations] == ["retained/body/BODY.bin"]


def test_reference_scan_is_clean_when_the_carve_out_covers_the_reference(
    sr3, tmp_path, tree, monkeypatch
):
    monkeypatch.setattr(sr3, "AP_ROOT", Path("/Volumes/APDataStore/pact"))
    repo = _repo_with_reference(tmp_path / "repo", tree.name, "retained/body/BODY.bin")
    assert sr3.scan_live_references(repo, tree, keep_uncompressed=("retained/body",)) == []


def test_reference_scan_is_not_fooled_by_a_prefix_sibling_carve_out(
    sr3, tmp_path, tree, monkeypatch
):
    """RED DIRECTION: carving 'retained/body' must NOT clear a 'bodyguard' read."""
    monkeypatch.setattr(sr3, "AP_ROOT", Path("/Volumes/APDataStore/pact"))
    repo = _repo_with_reference(tmp_path / "repo", tree.name, "retained/bodyguard/bulk.npy")
    violations = sr3.scan_live_references(repo, tree, keep_uncompressed=("retained/body",))
    assert [row["referenced_path"] for row in violations] == ["retained/bodyguard/bulk.npy"]


def _repo_with_split_reference(root: Path, tree_name: str, head: str, tail: str) -> Path:
    """A reference written as two implicitly-concatenated literals -- the raw scan sees `head`."""
    (root / "experiments").mkdir(parents=True)
    (root / "experiments" / "reader.py").write_text(
        "BODY = (\n"
        f'    "/Volumes/APDataStore/pact/{tree_name}/{head}"\n'
        f'    "{tail}"\n'
        ")\n",
        encoding="utf-8",
    )
    return root


def test_reference_scan_resolves_a_split_literal_that_leaves_the_carve_out(
    sr3, tmp_path, tree, monkeypatch
):
    """RED DIRECTION: a truncated prefix that MATCHES a carve-out must not read as covered.

    ``"…/retained/body" "guard/bulk.npy"`` truncates to ``retained/body`` -- which IS the
    carve-out -- while the real read is ``retained/bodyguard/bulk.npy``, outside it.
    Trusting the prefix would archive and remove a file the program still opens.  The scan
    joins the siblings, so it adjudicates the REAL path and refuses.
    """
    monkeypatch.setattr(sr3, "AP_ROOT", Path("/Volumes/APDataStore/pact"))
    repo = _repo_with_split_reference(tmp_path / "repo", tree.name, "retained/body", "guard/bulk.npy")
    scan = sr3.scan_live_reference_detail(repo, tree, keep_uncompressed=("retained/body",))
    assert scan.covered == ()
    assert [row["referenced_path"] for row in scan.violations] == ["retained/bodyguard/bulk.npy"]


def test_reference_scan_resolves_a_split_literal_that_stays_inside_the_carve_out(
    sr3, tmp_path, tree, monkeypatch
):
    """The BENIGN split -- the common idiom -- resolves to the real path and reads as covered.

    ``Path("…/retained/body/" "carrier.zip")`` is simply how a long path fits under the
    line limit; a repo-wide scan measured 416 such sites.  Refusing them all would refuse
    every lift, so the scan reads them instead.
    """
    monkeypatch.setattr(sr3, "AP_ROOT", Path("/Volumes/APDataStore/pact"))
    repo = _repo_with_split_reference(tmp_path / "repo", tree.name, "retained/body/", "carrier.zip")
    scan = sr3.scan_live_reference_detail(repo, tree, keep_uncompressed=("retained/body",))
    assert scan.violations == ()
    assert [row["referenced_path"] for row in scan.covered] == ["retained/body/carrier.zip"]


def test_reference_scan_refuses_a_continuation_it_cannot_read(sr3, tmp_path, tree, monkeypatch):
    """An f-string sibling is composed at RUNTIME, so the prefix is unreadable -- refuse it."""
    monkeypatch.setattr(sr3, "AP_ROOT", Path("/Volumes/APDataStore/pact"))
    root = tmp_path / "repo"
    (root / "experiments").mkdir(parents=True)
    (root / "experiments" / "reader.py").write_text(
        "BODY = (\n"
        f'    "/Volumes/APDataStore/pact/{tree.name}/retained/body"\n'
        '    f"{suffix}/bulk.npy"\n'
        ")\n",
        encoding="utf-8",
    )
    scan = sr3.scan_live_reference_detail(root, tree, keep_uncompressed=("retained/body",))
    assert scan.covered == ()
    assert [row["referenced_path"] for row in scan.violations] == ["retained/body"]
    assert scan.violations[0]["unresolvable_implicit_concatenation"] == "true"


def test_reference_scan_does_not_flag_a_complete_literal_as_truncated(
    sr3, tmp_path, tree, monkeypatch
):
    """NEGATIVE DIRECTION: an unbroken literal is covered normally, with no refusal flag."""
    monkeypatch.setattr(sr3, "AP_ROOT", Path("/Volumes/APDataStore/pact"))
    repo = _repo_with_reference(tmp_path / "repo", tree.name, "retained/body/BODY.bin")
    scan = sr3.scan_live_reference_detail(repo, tree, keep_uncompressed=("retained/body",))
    assert scan.violations == ()
    assert [row["referenced_path"] for row in scan.covered] == ["retained/body/BODY.bin"]
    assert "unresolvable_implicit_concatenation" not in scan.covered[0]


def test_reference_scan_refuses_a_truncated_join_that_would_read_as_covered(
    sr3, tmp_path, tree, monkeypatch
):
    """The residual hole one level down: the JOIN itself can truncate.

    ``"…/retained/body" "{slot}.npy"`` joins to ``retained/body{slot}.npy``; the scan can
    only keep ``retained/body``, which is the carve-out -- the bodyguard shape again, now
    inside the resolver.  Refuse, because the truncation is what makes it look covered.
    """
    monkeypatch.setattr(sr3, "AP_ROOT", Path("/Volumes/APDataStore/pact"))
    repo = _repo_with_split_reference(tmp_path / "repo", tree.name, "retained/body", "{slot}.npy")
    scan = sr3.scan_live_reference_detail(repo, tree, keep_uncompressed=("retained/body",))
    assert scan.covered == ()
    assert scan.violations[0]["truncated_continuation_cannot_confirm_carve_out"] == "true"


def test_reference_scan_also_refuses_a_prose_sibling_it_cannot_rule_out(
    sr3, tmp_path, tree, monkeypatch
):
    """A sibling opening on a non-path character is USUALLY prose -- but not provably.

    ``"…/retained/body/x.bin" " is missing"`` is a message and the path really ended.  But
    ``"…/retained/body/" "x y.npy"`` breaks at the same offset 0 and the path did NOT end,
    so the scan cannot tell them apart from the break position.  It refuses both, which
    over-refuses the prose form: measured cost is 1 site in 64,131 files, against a hole
    that would silently archive a live read.  The typed reason makes the refusal
    diagnosable, so a lift blocked this way is a one-line adjudication, not a mystery.
    """
    monkeypatch.setattr(sr3, "AP_ROOT", Path("/Volumes/APDataStore/pact"))
    repo = _repo_with_split_reference(tmp_path / "repo", tree.name, "retained/body/x.bin", " is missing")
    scan = sr3.scan_live_reference_detail(repo, tree, keep_uncompressed=("retained/body",))
    assert scan.covered == ()
    assert scan.violations[0]["truncated_continuation_cannot_confirm_carve_out"] == "true"


def test_literal_continues_only_fires_on_real_concatenation(sr3):
    """Quote -> whitespace -> quote is concatenation; every other follower is not."""
    assert sr3._literal_continues('"a/b/" "c"', 5) is True          # same line
    assert sr3._literal_continues('"a/b/"\n    "c"', 5) is True      # across lines
    assert sr3._literal_continues('"a/b/", "c"', 5) is False         # list/tuple element
    assert sr3._literal_continues('"a/b/")', 5) is False             # end of call
    assert sr3._literal_continues('"a/b/": "c"', 5) is False         # dict key
    assert sr3._literal_continues('"a/b/"', 5) is False              # end of text
    assert sr3._literal_continues('a/b/"""', 4) is False             # docstring close
    assert sr3._literal_continues("a/b/'''", 4) is False             # docstring close, single
    assert sr3._literal_continues('"a/b/" f"{x}"', 5) is True         # prefixed sibling
    assert sr3._literal_continues('"a/b/" rb"c"', 5) is True          # two-char prefix
    assert sr3._literal_continues('"a/b/" and "c"', 5) is False       # keyword, not a prefix
    assert sr3._literal_continues('"a/b/" if p else "c"', 5) is False  # conditional


def test_reference_scan_reports_its_denominator_and_covered_rows(sr3, tmp_path, tree, monkeypatch):
    """The vacuous-pass cure: a clean scan must show what it actually looked at."""
    monkeypatch.setattr(sr3, "AP_ROOT", Path("/Volumes/APDataStore/pact"))
    repo = _repo_with_reference(tmp_path / "repo", tree.name, "retained/body/BODY.bin")
    scan = sr3.scan_live_reference_detail(repo, tree, keep_uncompressed=("retained/body",))
    assert scan.violations == ()
    assert scan.references_found == 1
    assert [row["referenced_path"] for row in scan.covered] == ["retained/body/BODY.bin"]
    receipt = scan.receipt()
    assert receipt["files_scanned"] >= 1
    assert receipt["vacuous_scan_no_reference_found"] is False


def test_reference_scan_flags_a_vacuous_pass(sr3, tmp_path, tree, monkeypatch):
    """RED DIRECTION: finding nothing is REPORTED, never silently called clean.

    A scan whose roots are empty and a scan over a genuinely unreferenced tree
    produce identical violation lists; only the denominator separates them.
    """
    monkeypatch.setattr(sr3, "AP_ROOT", Path("/Volumes/APDataStore/pact"))
    repo = tmp_path / "repo"
    (repo / "experiments").mkdir(parents=True)
    (repo / "experiments" / "unrelated.py").write_text("X = 1\n", encoding="utf-8")
    scan = sr3.scan_live_reference_detail(repo, tree, keep_uncompressed=("retained/body",))
    assert scan.violations == ()
    assert scan.receipt()["vacuous_scan_no_reference_found"] is True


def test_reference_scan_method_states_its_own_bounds(sr3):
    """The receipt must carry the limits, or it will be read as universal coverage."""
    method = sr3._REFERENCE_SCAN_METHOD
    assert "LITERAL" in method
    assert "composed at runtime" in method
    assert "RESOLVED" in method


# --- the scan is unconditional: an UNDECLARED tree is scanned too ---


def _run_main(sr3, monkeypatch, tree, repo, keeps=()):
    """Drive main() far enough to reach (or pass) the reference scan."""
    argv = [
        "sr3",
        "--tree",
        str(tree),
        "--repo",
        str(repo),
        "--closure-citation",
        "does/not/exist.md",  # checked AFTER the scan, so it marks how far we got
        "--closure-note",
        "a substantive closure note for the test",
    ]
    for keep in keeps:
        argv += ["--keep-uncompressed", keep]
    monkeypatch.setattr(sr3.sys, "argv", argv)
    return sr3.main()


def test_unprotected_tree_with_a_live_read_is_refused(sr3, tmp_path, tree, monkeypatch, capsys):
    """RED DIRECTION: protection is a DECLARATION, not what makes a tree readable.

    Before the cure the scan ran only inside the ``--lift-protection`` branch, so a
    tree nobody had declared was archived and its originals removed with no live-read
    check at all.  Measured over the six un-archived terminal-lane trees -- none of
    them protected -- 61 live references, every tree non-zero.  The refusal must fire
    without any declaration.
    """
    monkeypatch.setattr(sr3, "AP_ROOT", tree.parent)
    monkeypatch.setattr(sr3, "PROTECTED_TREES", set())  # explicitly NOT protected
    repo = tmp_path / "repo"
    (repo / "experiments").mkdir(parents=True)
    (repo / "experiments" / "reader.py").write_text(
        f'BODY = "{tree.parent}/{tree.name}/retained/body/BODY.bin"\n', encoding="utf-8"
    )
    assert _run_main(sr3, monkeypatch, tree, repo) == 2
    assert "are not carved out" in capsys.readouterr().err


def test_the_unconditional_scan_still_passes_a_covered_reference(
    sr3, tmp_path, tree, monkeypatch, capsys
):
    """The cure must not become a blanket refusal: a carved-out read still proceeds.

    Reaching the closure-citation error (the NEXT check after the scan) is the proof
    the scan did not refuse -- a gate that refuses everything protects nothing.
    """
    monkeypatch.setattr(sr3, "AP_ROOT", tree.parent)
    monkeypatch.setattr(sr3, "PROTECTED_TREES", set())
    repo = tmp_path / "repo"
    (repo / "experiments").mkdir(parents=True)
    (repo / "experiments" / "reader.py").write_text(
        f'BODY = "{tree.parent}/{tree.name}/retained/body/BODY.bin"\n', encoding="utf-8"
    )
    assert _run_main(sr3, monkeypatch, tree, repo, keeps=("retained/body",)) == 2
    err = capsys.readouterr().err
    assert "are not carved out" not in err
    assert "closure citation is not a file" in err


# --- a reference to an ABSENT path is recorded, not refused ---


def test_a_reference_to_an_absent_path_does_not_block_the_archive(
    sr3, tmp_path, tree, monkeypatch, capsys
):
    """The deadlock the unconditional scan created, and its cure.

    Measured 2026-08-31 on ddm_fs3: live code references FS3_REOPEN_PRICE.json, a path
    the intact tree does not hold.  The scan refused the uncovered reference while
    ``validate_keep_paths`` refuses a carve-out for a path that does not exist -- so the
    tree was unarchivable in BOTH directions over bytes that are already gone.  Absent
    bytes cannot be read, so this must proceed; reaching the closure-citation error is
    the proof the scan did not refuse.
    """
    monkeypatch.setattr(sr3, "AP_ROOT", tree.parent)
    monkeypatch.setattr(sr3, "PROTECTED_TREES", set())
    repo = tmp_path / "repo"
    (repo / "experiments").mkdir(parents=True)
    (repo / "experiments" / "reader.py").write_text(
        f'PRICE = "{tree.parent}/{tree.name}/GHOST_PRICE.json"\n', encoding="utf-8"
    )
    assert _run_main(sr3, monkeypatch, tree, repo) == 2
    err = capsys.readouterr().err
    assert "are not carved out" not in err
    assert "closure citation is not a file" in err


def test_the_absent_class_is_recorded_never_silently_dropped(sr3, tmp_path, tree, monkeypatch):
    """Non-blocking must not mean invisible -- the receipt carries the row and a count."""
    monkeypatch.setattr(sr3, "AP_ROOT", Path("/Volumes/APDataStore/pact"))
    repo = _repo_with_reference(tmp_path / "repo", tree.name, "GHOST_PRICE.json")
    scan = sr3.scan_live_reference_detail(repo, tree, keep_uncompressed=())
    assert scan.violations == ()
    assert [row["referenced_path"] for row in scan.absent] == ["GHOST_PRICE.json"]
    assert scan.receipt()["references_absent_from_tree"] == 1
    assert scan.receipt()["references_found"] == 1  # the denominator still counts it


def test_an_absent_fstring_PREFIX_is_still_a_violation(sr3, tmp_path, tree, monkeypatch):
    """RED DIRECTION: exactness is what makes the absent class safe.

    A literal that stops on an f-string ``{`` slot yields a PREFIX, not the real path.
    An absent prefix proves nothing -- the path the program actually opens may well
    exist -- so these must keep refusing.  Drop the ``exact_read`` conjunct and this
    test fails while the two green-direction tests still pass; that is the control.
    """
    monkeypatch.setattr(sr3, "AP_ROOT", Path("/Volumes/APDataStore/pact"))
    root = tmp_path / "repo"
    (root / "experiments").mkdir(parents=True)
    (root / "experiments" / "reader.py").write_text(
        f'P = f"/Volumes/APDataStore/pact/{tree.name}/ghost/{{name}}.json"\n', encoding="utf-8"
    )
    scan = sr3.scan_live_reference_detail(root, tree, keep_uncompressed=())
    assert scan.absent == ()
    assert [row["referenced_path"] for row in scan.violations] == ["ghost"]


# --- selective removal keeps the carve-out AND its ancestors standing ---


def test_selective_removal_keeps_carve_outs_and_removes_the_rest(sr3, tree):
    keeps = ("retained/body", "BODY_RESULT.json")
    entries = sr3.scan_tree(tree, keeps)
    sr3.remove_original_top_level(tree, entries, keep_uncompressed=keeps)

    # the carve-outs and the ancestor directory they need survive
    assert (tree / "retained" / "body" / "BODY.bin").read_bytes() == b"live body payload"
    assert (tree / "BODY_RESULT.json").is_file()
    assert (tree / "retained").is_dir()
    # everything archived is gone, including the whole prefix-sibling subtree
    assert not (tree / "retained" / "bodyguard").exists()
    assert not (tree / "retained" / "other.npy").exists()
    assert not (tree / "caches").exists()


def test_selective_removal_satisfies_the_reclaim_gate(sr3, tree):
    """The gate main() enforces must pass, and ONLY the ancestor may stand.

    The second assertion is the behavioural half: a gate that exempted every
    surviving directory would pass the first check while leaving real bulk on
    disk.  Naming the exact survivor pins the exemption to one path.
    """
    keeps = ("retained/body",)
    entries = sr3.scan_tree(tree, keeps)
    sr3.remove_original_top_level(tree, entries, keep_uncompressed=keeps)
    assert sr3.unreclaimed_originals(tree, keeps) == []
    assert [row.path for row in sr3.scan_tree(tree, keeps)] == ["retained"]


def test_reclaim_gate_still_refuses_a_non_ancestor_survivor(sr3, tree):
    """RED DIRECTION: the ancestor exemption must not excuse un-removed bulk.

    Removal is handed a manifest missing ``caches/``, so that subtree survives
    without being an ancestor of any carve-out.  The gate must still see it.
    """
    keeps = ("retained/body",)
    entries = [row for row in sr3.scan_tree(tree, keeps) if not row.path.startswith("caches")]
    sr3.remove_original_top_level(tree, entries, keep_uncompressed=keeps)
    residue = {row.path for row in sr3.unreclaimed_originals(tree, keeps)}
    assert "caches/big.npy" in residue
    assert "retained" not in residue


# --- the carve-out's BYTES are verified after removal, not merely recorded ---


def test_keep_records_verify_clean_across_a_real_selective_removal(sr3, tree):
    """The post-condition: removal must leave every carve-out byte-identical."""
    keeps = ("retained/body", "BODY_RESULT.json")
    records = sr3.build_keep_records(tree, keeps)
    entries = sr3.scan_tree(tree, keeps)
    sr3.remove_original_top_level(tree, entries, keep_uncompressed=keeps)
    result = sr3.verify_keep_records(tree, records)
    assert result["clean"] is True
    assert result["paths_checked"] == 2  # BODY.bin + BODY_RESULT.json
    assert result["vanished"] == [] and result["changed"] == []


def test_keep_records_verify_catches_a_vanished_carve_out(sr3, tree):
    """RED DIRECTION: a deleted carve-out must be reported, never passed."""
    keeps = ("retained/body",)
    records = sr3.build_keep_records(tree, keeps)
    (tree / "retained" / "body" / "BODY.bin").unlink()
    result = sr3.verify_keep_records(tree, records)
    assert result["clean"] is False
    assert result["vanished"] == ["retained/body/BODY.bin"]


def test_keep_records_verify_catches_a_byte_changed_carve_out(sr3, tree):
    """RED DIRECTION: same size, different bytes -- only the sha can see it."""
    keeps = ("BODY_RESULT.json",)
    records = sr3.build_keep_records(tree, keeps)
    victim = tree / "BODY_RESULT.json"
    victim.write_bytes(b'{"ok": FALS}')  # same length, different content
    result = sr3.verify_keep_records(tree, records)
    assert result["clean"] is False
    assert result["changed"] == ["BODY_RESULT.json"]


def test_keep_records_verify_allows_a_live_store_to_gain_files(sr3, tree):
    """A carve-out is LIVE: new files there are expected, not drift."""
    keeps = ("retained/body",)
    records = sr3.build_keep_records(tree, keeps)
    (tree / "retained" / "body" / "NEW_RESULT.json").write_bytes(b"{}")
    assert sr3.verify_keep_records(tree, records)["clean"] is True


def test_removal_without_carve_outs_keeps_the_original_wholesale_path(sr3, tree):
    """No carve-outs declared -> unchanged top-level rmtree behaviour."""
    entries = sr3.scan_tree(tree)
    removed = sr3.remove_original_top_level(tree, entries)
    assert sorted(Path(p).name for p in removed) == ["BODY_RESULT.json", "caches", "retained"]
    assert not (tree / "retained").exists()
    assert list(tree.iterdir()) == []
