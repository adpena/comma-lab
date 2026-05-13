"""Tests for tools/xray_per_pr_archive_layout_compare.py."""
from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "tools"))

import xray_per_pr_archive_layout_compare as xc  # noqa: E402


def _make_zip(path: Path, members: dict[str, bytes]) -> Path:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, data in members.items():
            zf.writestr(name, data)
    return path


def test_archive_layout_single_member(tmp_path):
    zp = _make_zip(tmp_path / "a.zip", {"data.bin": b"hello" * 100})
    lay = xc.archive_layout(zp)
    assert lay["section_count"] == 1
    assert lay["sections"][0]["name"] == "data.bin"
    assert len(lay["archive_sha256"]) == 64


def test_archive_layout_multi_member(tmp_path):
    zp = _make_zip(tmp_path / "a.zip", {
        "x.bin": b"x" * 10,
        "y.bin": b"y" * 20,
    })
    lay = xc.archive_layout(zp)
    assert lay["section_count"] == 2
    names = {s["name"] for s in lay["sections"]}
    assert names == {"x.bin", "y.bin"}


def test_compare_matrix_two_identical_archives(tmp_path):
    members = {"d.bin": b"x" * 100}
    zp1 = _make_zip(tmp_path / "a.zip", members)
    zp2 = _make_zip(tmp_path / "b.zip", members)
    lays = [xc.archive_layout(zp1), xc.archive_layout(zp2)]
    cmp = xc.build_compare_matrix(lays, ["a", "b"])
    assert cmp["total_unique_sections"] == 1
    assert cmp["shared_sha256_across_all_archives"] == 1
    assert cmp["present_in_all_but_diff_content"] == 0


def test_compare_matrix_two_diverged_archives(tmp_path):
    zp1 = _make_zip(tmp_path / "a.zip", {"d.bin": b"AAA"})
    zp2 = _make_zip(tmp_path / "b.zip", {"d.bin": b"BBB"})
    lays = [xc.archive_layout(zp1), xc.archive_layout(zp2)]
    cmp = xc.build_compare_matrix(lays, ["a", "b"])
    assert cmp["total_unique_sections"] == 1
    assert cmp["shared_sha256_across_all_archives"] == 0
    assert cmp["present_in_all_but_diff_content"] == 1


def test_compare_matrix_section_missing_from_one(tmp_path):
    zp1 = _make_zip(tmp_path / "a.zip", {"x.bin": b"x", "y.bin": b"y"})
    zp2 = _make_zip(tmp_path / "b.zip", {"x.bin": b"x"})
    lays = [xc.archive_layout(zp1), xc.archive_layout(zp2)]
    cmp = xc.build_compare_matrix(lays, ["a", "b"])
    assert cmp["total_unique_sections"] == 2
    assert cmp["not_present_in_all_archives"] == 1
    y_row = next(r for r in cmp["rows"] if r["section_name"] == "y.bin")
    assert y_row["missing_from"] == ["b"]
    assert y_row["present_in"] == ["a"]


def test_compare_matrix_three_way(tmp_path):
    zp1 = _make_zip(tmp_path / "1.zip", {"d.bin": b"A"})
    zp2 = _make_zip(tmp_path / "2.zip", {"d.bin": b"A"})
    zp3 = _make_zip(tmp_path / "3.zip", {"d.bin": b"B"})
    lays = [xc.archive_layout(z) for z in (zp1, zp2, zp3)]
    cmp = xc.build_compare_matrix(lays, ["one", "two", "three"])
    row = cmp["rows"][0]
    assert row["distinct_sha256_count"] == 2
    assert row["shared_sha256_across_all"] is False


def test_compare_matrix_preserves_section_insertion_order(tmp_path):
    # First archive introduces a, b; second introduces c, then a
    zp1 = _make_zip(tmp_path / "1.zip", {"a.bin": b"x", "b.bin": b"y"})
    zp2 = _make_zip(tmp_path / "2.zip", {"c.bin": b"z", "a.bin": b"x"})
    lays = [xc.archive_layout(z) for z in (zp1, zp2)]
    cmp = xc.build_compare_matrix(lays, ["one", "two"])
    names = [r["section_name"] for r in cmp["rows"]]
    # Order: a, b (from archive 1), then c (introduced by archive 2)
    assert names.index("a.bin") < names.index("b.bin")
    assert names.index("b.bin") < names.index("c.bin")


def test_main_writes_outputs(tmp_path):
    zp1 = _make_zip(tmp_path / "1.zip", {"d.bin": b"x" * 50})
    zp2 = _make_zip(tmp_path / "2.zip", {"d.bin": b"y" * 50})
    out_dir = tmp_path / "out"
    rc = xc.main([
        "--archive", str(zp1),
        "--archive", str(zp2),
        "--output-dir", str(out_dir),
    ])
    assert rc == 0
    assert (out_dir / "layout_compare.json").exists()
    assert (out_dir / "layout_compare.md").exists()
    assert (out_dir / "rebuild_command.txt").exists()


def test_main_requires_two_archives(tmp_path):
    zp1 = _make_zip(tmp_path / "1.zip", {"d.bin": b"x"})
    rc = xc.main([
        "--archive", str(zp1),
        "--output-dir", str(tmp_path / "out"),
    ])
    assert rc == 2


def test_main_label_count_mismatch(tmp_path):
    zp1 = _make_zip(tmp_path / "1.zip", {"d.bin": b"x"})
    zp2 = _make_zip(tmp_path / "2.zip", {"d.bin": b"y"})
    rc = xc.main([
        "--archive", str(zp1),
        "--archive", str(zp2),
        "--label", "only_one",
        "--output-dir", str(tmp_path / "out"),
    ])
    assert rc == 2


def test_main_missing_archive(tmp_path):
    zp1 = _make_zip(tmp_path / "1.zip", {"d.bin": b"x"})
    rc = xc.main([
        "--archive", str(zp1),
        "--archive", str(tmp_path / "missing.zip"),
        "--output-dir", str(tmp_path / "out"),
    ])
    assert rc == 2


def test_main_default_labels_use_stem(tmp_path):
    zp1 = _make_zip(tmp_path / "alpha.zip", {"d.bin": b"x"})
    zp2 = _make_zip(tmp_path / "beta.zip", {"d.bin": b"y"})
    out = tmp_path / "out"
    xc.main(["--archive", str(zp1), "--archive", str(zp2), "--output-dir", str(out)])
    rep = json.loads((out / "layout_compare.json").read_text())
    assert rep["compare"]["labels"] == ["alpha", "beta"]


def test_markdown_marks_shared_and_diverged(tmp_path):
    zp1 = _make_zip(tmp_path / "1.zip", {"shared.bin": b"x", "diverged.bin": b"A"})
    zp2 = _make_zip(tmp_path / "2.zip", {"shared.bin": b"x", "diverged.bin": b"B"})
    out = tmp_path / "out"
    xc.main(["--archive", str(zp1), "--archive", str(zp2), "--output-dir", str(out)])
    md = (out / "layout_compare.md").read_text()
    assert "(SHARED)" in md
    assert "(DIVERGED)" in md


def test_markdown_marks_missing(tmp_path):
    zp1 = _make_zip(tmp_path / "1.zip", {"x.bin": b"x", "y.bin": b"y"})
    zp2 = _make_zip(tmp_path / "2.zip", {"x.bin": b"x"})
    out = tmp_path / "out"
    xc.main([
        "--archive", str(zp1), "--archive", str(zp2),
        "--label", "first", "--label", "second",
        "--output-dir", str(out),
    ])
    md = (out / "layout_compare.md").read_text()
    assert "missing in second" in md


def test_compare_records_distinct_sha_count(tmp_path):
    zp1 = _make_zip(tmp_path / "1.zip", {"d.bin": b"AA"})
    zp2 = _make_zip(tmp_path / "2.zip", {"d.bin": b"AA"})
    zp3 = _make_zip(tmp_path / "3.zip", {"d.bin": b"BB"})
    zp4 = _make_zip(tmp_path / "4.zip", {"d.bin": b"CC"})
    lays = [xc.archive_layout(z) for z in (zp1, zp2, zp3, zp4)]
    cmp = xc.build_compare_matrix(lays, ["a", "b", "c", "d"])
    row = cmp["rows"][0]
    assert row["distinct_sha256_count"] == 3  # AA, BB, CC


def test_score_claim_always_false(tmp_path):
    zp1 = _make_zip(tmp_path / "1.zip", {"d.bin": b"x"})
    zp2 = _make_zip(tmp_path / "2.zip", {"d.bin": b"y"})
    out = tmp_path / "out"
    xc.main(["--archive", str(zp1), "--archive", str(zp2), "--output-dir", str(out)])
    rep = json.loads((out / "layout_compare.json").read_text())
    assert rep["score_claim"] is False
    assert rep["evidence_grade"] == "diagnostic_only"


def test_aggregate_counts_complete(tmp_path):
    # 4-archive setup with mixed shared/diverged/missing
    zp1 = _make_zip(tmp_path / "1.zip", {"all.bin": b"x", "diverged.bin": b"A", "only_a.bin": b"a"})
    zp2 = _make_zip(tmp_path / "2.zip", {"all.bin": b"x", "diverged.bin": b"B"})
    zp3 = _make_zip(tmp_path / "3.zip", {"all.bin": b"x", "diverged.bin": b"C"})
    lays = [xc.archive_layout(z) for z in (zp1, zp2, zp3)]
    cmp = xc.build_compare_matrix(lays, ["a", "b", "c"])
    assert cmp["total_unique_sections"] == 3
    assert cmp["shared_sha256_across_all_archives"] == 1  # all.bin
    assert cmp["present_in_all_but_diff_content"] == 1  # diverged.bin
    assert cmp["not_present_in_all_archives"] == 1  # only_a.bin


def test_per_archive_compress_method_recorded(tmp_path):
    zp = tmp_path / "test.zip"
    with zipfile.ZipFile(zp, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr("data.bin", b"x" * 10)
    lay = xc.archive_layout(zp)
    assert lay["sections"][0]["compress_method"] == zipfile.ZIP_STORED


def test_state_hash_changes_when_archives_differ(tmp_path):
    zp1 = _make_zip(tmp_path / "1.zip", {"d.bin": b"AAA"})
    zp2 = _make_zip(tmp_path / "2.zip", {"d.bin": b"BBB"})
    zp3 = _make_zip(tmp_path / "3.zip", {"d.bin": b"CCC"})
    out1 = tmp_path / "out1"
    out2 = tmp_path / "out2"
    xc.main(["--archive", str(zp1), "--archive", str(zp2), "--output-dir", str(out1)])
    xc.main(["--archive", str(zp1), "--archive", str(zp3), "--output-dir", str(out2)])
    r1 = json.loads((out1 / "layout_compare.json").read_text())
    r2 = json.loads((out2 / "layout_compare.json").read_text())
    assert r1["from_state_hash"] != r2["from_state_hash"]
