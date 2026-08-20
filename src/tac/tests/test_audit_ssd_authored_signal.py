"""Tests for tools/audit_ssd_authored_signal.py — the standing SSD authored-signal guard.

These verify BEHAVIOUR, not constants. Each test builds a real temp tree, runs the real scan, and
asserts on what the scan found. If the scan body were replaced by `return <canonical markers>`,
every test here fails — that is the bar CLAUDE.md's NO-FAKE class 2 sets for a test suite.

The load-bearing case is the positive control: plant an authored file that git has never seen and
prove the guard flags it. A guard that cannot fail is a vacuous gate.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_TOOL = Path(__file__).resolve().parents[3] / "tools" / "audit_ssd_authored_signal.py"


def _load():
    spec = importlib.util.spec_from_file_location("audit_ssd_authored_signal", _TOOL)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


mod = _load()


# ------------------------------------------------------------------------------------- identity


def test_blob_sha1_matches_git_hash_object(tmp_path):
    """Our sha must be git's sha, or every downstream comparison is meaningless."""
    import subprocess

    f = tmp_path / "sample.py"
    f.write_bytes(b"print('hello')\n")
    expected = subprocess.run(
        ["git", "hash-object", str(f)], capture_output=True, text=True, check=True
    ).stdout.strip()
    assert mod.blob_sha1(f) == expected


def test_blob_sha1_returns_none_on_unreadable(tmp_path):
    assert mod.blob_sha1(tmp_path / "does_not_exist.py") is None


def test_blob_sha1_distinguishes_one_byte(tmp_path):
    a = tmp_path / "a.py"
    b = tmp_path / "b.py"
    a.write_bytes(b"x = 1\n")
    b.write_bytes(b"x = 2\n")
    assert mod.blob_sha1(a) != mod.blob_sha1(b)


def test_empty_file_hashes_to_the_git_empty_blob(tmp_path):
    f = tmp_path / "empty.py"
    f.write_bytes(b"")
    assert mod.blob_sha1(f) == "e69de29bb2d1d6434b8b29ae775ad8c2e48c5391"


# ------------------------------------------------------------------------------ positive control


def test_positive_control_planted_authored_file_is_flagged(tmp_path):
    """THE load-bearing test: an authored file git has never seen must land in bucket C."""
    planted = tmp_path / "arm_workspace" / "solver.py"
    planted.parent.mkdir(parents=True)
    planted.write_text("def unique_marker_ddm_sd1(): return 42\n")
    sha = mod.blob_sha1(planted)

    report = mod.scan(roots=[tmp_path], reachable=set(), odb=set())

    assert report["denominator"]["bucket_C_authored_OWED"] == 1
    assert [r["blob_sha1"] for r in report["owed"]] == [sha]
    assert report["owed"][0]["representative_path"] == str(planted)


def test_negative_control_reachable_file_is_not_flagged(tmp_path):
    """A file whose blob a ref reaches is preserved and must NOT be reported as debt."""
    f = tmp_path / "arm" / "committed.py"
    f.parent.mkdir(parents=True)
    f.write_text("x = 1\n")
    report = mod.scan(roots=[tmp_path], reachable={mod.blob_sha1(f)}, odb=set())
    assert report["denominator"]["bucket_C_authored_OWED"] == 0
    assert report["denominator"]["ssd_code_like_files_scanned"] == 1  # scanned, then cleared


def test_gc_eligible_is_owed_but_labelled_distinctly(tmp_path):
    """In the ODB yet unreachable = recoverable now, gone after `git gc --prune`. Still debt."""
    f = tmp_path / "arm" / "staged_never_committed.py"
    f.parent.mkdir(parents=True)
    f.write_text("y = 2\n")
    sha = mod.blob_sha1(f)
    report = mod.scan(roots=[tmp_path], reachable=set(), odb={sha})
    assert report["denominator"]["bucket_C_authored_OWED"] == 1
    assert report["denominator"]["bucket_C_of_which_gc_eligible"] == 1
    assert report["owed"][0]["durability"] == "gc_eligible"


def test_absent_and_gc_eligible_are_not_conflated(tmp_path):
    a = tmp_path / "arm" / "never_seen.py"
    b = tmp_path / "arm" / "staged_only.py"
    a.parent.mkdir(parents=True)
    a.write_text("a = 1\n")
    b.write_text("b = 2\n")
    report = mod.scan(roots=[tmp_path], reachable=set(), odb={mod.blob_sha1(b)})
    kinds = {r["representative_path"]: r["durability"] for r in report["owed"]}
    assert kinds[str(a)] == "absent"
    assert kinds[str(b)] == "gc_eligible"


# ------------------------------------------------------------------------------------- denominator


def test_unmounted_root_is_reported_not_silently_skipped(tmp_path):
    """A missing root must make the report say PARTIAL. Silence here is the vacuity==pass bug."""
    report = mod.scan(roots=[tmp_path / "not_there"], reachable=set(), odb=set())
    assert report["denominator"]["roots_absent_from_this_machine"] == [str(tmp_path / "not_there")]
    assert "ROOT NOT MOUNTED" in mod.fmt(report)


def test_distinct_blobs_collapse_copies_but_instances_count_them(tmp_path):
    """A file copied per run must count ONCE as debt and N times as instances."""
    body = "def shared(): return 1\n"
    for i in range(4):
        d = tmp_path / f"run_{i}" / "runtime"
        d.mkdir(parents=True)
        (d / "codec.py").write_text(body)
    report = mod.scan(roots=[tmp_path], reachable=set(), odb=set())
    assert report["denominator"]["absent_DISTINCT_blobs"] == 1
    assert report["denominator"]["absent_file_instances"] == 4
    assert report["owed"][0]["instance_count"] == 4


def test_a_moved_file_is_counted_once_not_twice(tmp_path):
    """vr1 relocates bulk between tiers; the same bytes at a new path are the same blob."""
    body = "def moved(): return 7\n"
    for tier in ("tier_a", "tier_b"):
        d = tmp_path / tier / "arm"
        d.mkdir(parents=True)
        (d / "same.py").write_text(body)
    report = mod.scan(roots=[tmp_path / "tier_a", tmp_path / "tier_b"], reachable=set(), odb=set())
    assert report["denominator"]["absent_DISTINCT_blobs"] == 1
    assert report["denominator"]["bucket_C_authored_OWED"] == 1


def test_pruned_dirs_and_appledouble_are_excluded(tmp_path):
    for junk in ("__pycache__", ".venv", "node_modules"):
        d = tmp_path / "arm" / junk
        d.mkdir(parents=True)
        (d / "junk.py").write_text("junk = 1\n")
    (tmp_path / "arm" / "._sidecar.py").write_text("apple = 1\n")
    (tmp_path / "arm" / "real.py").write_text("real = 1\n")
    report = mod.scan(roots=[tmp_path], reachable=set(), odb=set())
    assert report["denominator"]["ssd_code_like_files_scanned"] == 1
    assert report["owed"][0]["representative_path"].endswith("real.py")


def test_non_code_extensions_are_not_scanned(tmp_path):
    d = tmp_path / "arm"
    d.mkdir()
    for name in ("weights.npz", "tokens.bin", "result.json", "archive.zip"):
        (d / name).write_bytes(b"\x00\x01")
    (d / "src.py").write_text("s = 1\n")
    report = mod.scan(roots=[tmp_path], reachable=set(), odb=set())
    assert report["denominator"]["ssd_code_like_files_scanned"] == 1


# --------------------------------------------------------------------------------------- buckets


@pytest.mark.parametrize("segment", ["upstream", "third_party", "public_datasets", "openpilot"])
def test_third_party_segments_bucket_A(tmp_path, segment):
    f = tmp_path / segment / "lib.py"
    f.parent.mkdir(parents=True)
    f.write_text("lib = 1\n")
    report = mod.scan(roots=[tmp_path], reachable=set(), odb=set())
    assert report["denominator"]["bucket_A_third_party_or_clone"] == 1
    assert report["denominator"]["bucket_C_authored_OWED"] == 0


def test_nested_clone_buckets_A(tmp_path):
    """A subtree with its own .git is somebody else's repo, not an arm's authored workspace."""
    clone = tmp_path / "some_clone"
    (clone / ".git").mkdir(parents=True)
    (clone / "deep" / "nest").mkdir(parents=True)
    (clone / "deep" / "nest" / "mod.py").write_text("m = 1\n")
    report = mod.scan(roots=[tmp_path], reachable=set(), odb=set())
    assert report["denominator"]["bucket_A_third_party_or_clone"] == 1


def test_cold_store_and_experiments_results_bucket_B(tmp_path):
    for rel in ("cold_store/out.py", "experiments/results/run/out2.py"):
        f = tmp_path / "arm" / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(f"rel = '{rel}'\n")
    report = mod.scan(roots=[tmp_path], reachable=set(), odb=set())
    assert report["denominator"]["bucket_B_run_output_or_coldstore"] == 2
    assert report["denominator"]["bucket_C_authored_OWED"] == 0


def test_vendored_compression_library_source_buckets_A(tmp_path):
    """`brotli110_source/c/**` unpacked for a repro is somebody else's code, not our debt."""
    f = tmp_path / "ddm_arm" / "repro_before" / "brotli110_source" / "c" / "dec" / "decode.c"
    f.parent.mkdir(parents=True)
    f.write_text("int decode(void) { return 0; }\n")
    report = mod.scan(roots=[tmp_path], reachable=set(), odb=set())
    assert report["denominator"]["bucket_A_third_party_or_clone"] == 1
    assert report["denominator"]["bucket_C_authored_OWED"] == 0


def test_generated_packet_inflate_buckets_B(tmp_path):
    """A packet's inflate.py carries an embedded payload — a build product, not an authored source."""
    f = tmp_path / "ddm_arm" / "packet" / "inflate.py"
    f.parent.mkdir(parents=True)
    f.write_text("PAYLOAD = b'\\x00' * 16\n")
    report = mod.scan(roots=[tmp_path], reachable=set(), odb=set())
    assert report["denominator"]["bucket_B_run_output_or_coldstore"] == 1
    assert report["denominator"]["bucket_C_authored_OWED"] == 0


@pytest.mark.parametrize("d", ["packet", "submission", "submission_dir", "archive", "runtime_tree"])
@pytest.mark.parametrize("name", ["inflate.py", "inflate.sh"])
def test_generated_inflate_in_every_measured_packet_dir_buckets_B(tmp_path, d, name):
    f = tmp_path / "ddm_arm" / d / name
    f.parent.mkdir(parents=True)
    f.write_text("# emitted by the archive builder\n")
    report = mod.scan(roots=[tmp_path], reachable=set(), odb=set())
    assert report["denominator"]["bucket_B_run_output_or_coldstore"] == 1
    assert report["denominator"]["bucket_C_authored_OWED"] == 0


def test_an_authored_inflate_outside_a_packet_dir_stays_owed(tmp_path):
    """Only `<packet-ish>/inflate.py` is a build product. A hand-written one elsewhere is debt.

    This is the boundary the generated-file rule must not cross: widening it to "any file named
    inflate.py" would silently excuse a genuinely new receiver an arm authored on the SSD.
    """
    f = tmp_path / "ddm_arm" / "builders" / "inflate.py"
    f.parent.mkdir(parents=True)
    f.write_text("def build(): return 1\n")
    report = mod.scan(roots=[tmp_path], reachable=set(), odb=set())
    assert report["denominator"]["bucket_C_authored_OWED"] == 1


def test_coldstore_named_directory_buckets_B(tmp_path):
    """`vertigo_coldstore_20260811` is a cold store even though the segment is not exactly it."""
    f = tmp_path / "vertigo_coldstore_20260811" / "arm" / "x.py"
    f.parent.mkdir(parents=True)
    f.write_text("x = 1\n")
    report = mod.scan(roots=[tmp_path], reachable=set(), odb=set())
    assert report["denominator"]["bucket_B_run_output_or_coldstore"] == 1


def test_a_bare_results_dir_stays_owed(tmp_path):
    """`<arm>/results/builder.py` is commonly an authored script. Ambiguity must flag, not excuse."""
    f = tmp_path / "ddm_arm" / "results" / "builder.py"
    f.parent.mkdir(parents=True)
    f.write_text("def build(): return 1\n")
    report = mod.scan(roots=[tmp_path], reachable=set(), odb=set())
    assert report["denominator"]["bucket_C_authored_OWED"] == 1
    assert report["denominator"]["bucket_B_run_output_or_coldstore"] == 0


def test_authored_copy_into_coldstore_cannot_launder_the_debt(tmp_path):
    """Same bytes in an authored dir and in cold_store: the strictest reading must win."""
    body = "def authored(): return 3\n"
    a = tmp_path / "cold_store" / "copy.py"
    b = tmp_path / "ddm_arm" / "builders" / "copy.py"
    a.parent.mkdir(parents=True)
    b.parent.mkdir(parents=True)
    a.write_text(body)
    b.write_text(body)
    report = mod.scan(roots=[tmp_path], reachable=set(), odb=set())
    assert report["denominator"]["bucket_C_authored_OWED"] == 1
    assert report["denominator"]["bucket_B_run_output_or_coldstore"] == 0


# ----------------------------------------------------------------------------------- certification


def test_certification_moves_a_blob_from_owed_to_certified(tmp_path, monkeypatch):
    ledger = tmp_path / "cert.jsonl"
    f = tmp_path / "arm" / "scratch.py"
    f.parent.mkdir(parents=True)
    f.write_text("scratch = 1\n")
    sha = mod.blob_sha1(f)
    mod.append_certification(sha, "one-shot scratch probe; generator committed at tools/x.py",
                            "ddm_sd1", str(f), ledger=ledger)
    monkeypatch.setattr(mod, "CERTIFIED", ledger)
    report = mod.scan(roots=[tmp_path], reachable=set(), odb=set())
    assert report["denominator"]["bucket_C_authored_OWED"] == 0
    assert report["denominator"]["bucket_D_certified_in_place"] == 1
    assert report["certified_in_place"][0]["certification"]["owner"] == "ddm_sd1"


@pytest.mark.parametrize("rationale", ["", "tbd", "<rationale>", "short", "placeholder"])
def test_certification_refuses_placeholder_rationale(tmp_path, rationale):
    """An uncertified blob is a more honest state than a fake certificate."""
    with pytest.raises(mod.AuditError):
        mod.append_certification("a" * 40, rationale, "ddm_sd1", ledger=tmp_path / "c.jsonl")


def test_certification_refuses_missing_owner(tmp_path):
    with pytest.raises(mod.AuditError):
        mod.append_certification("a" * 40, "a perfectly substantive rationale here", "",
                                 ledger=tmp_path / "c.jsonl")


def test_certification_refuses_non_sha(tmp_path):
    with pytest.raises(mod.AuditError):
        mod.append_certification("not-a-sha", "a perfectly substantive rationale here",
                                 "ddm_sd1", ledger=tmp_path / "c.jsonl")


def test_certified_ledger_lives_somewhere_git_actually_tracks():
    """A certification is a durable decision. Storing it under a gitignored path would strand it
    on one machine — precisely the failure this module detects."""
    import subprocess

    rel = mod.CERTIFIED.relative_to(mod.REPO)
    ignored = subprocess.run(
        ["git", "check-ignore", "-q", str(rel)], cwd=mod.REPO, capture_output=True
    ).returncode == 0
    assert not ignored, f"{rel} is gitignored — certifications would never leave this machine"
    # The regenerable sweep cache is the opposite case and SHOULD stay ignored.
    cache_ignored = subprocess.run(
        ["git", "check-ignore", "-q", str(mod.CACHE.relative_to(mod.REPO))],
        cwd=mod.REPO, capture_output=True,
    ).returncode == 0
    assert cache_ignored, "the sweep cache is live state and should not be tracked"


def test_certification_ledger_is_append_only_and_latest_wins(tmp_path):
    ledger = tmp_path / "c.jsonl"
    sha = "b" * 40
    mod.append_certification(sha, "first substantive rationale text", "ddm_a", ledger=ledger)
    mod.append_certification(sha, "second substantive rationale text", "ddm_b", ledger=ledger)
    assert len(ledger.read_text().strip().splitlines()) == 2
    assert mod.load_certified(ledger)[sha]["owner"] == "ddm_b"


def test_certification_invalidates_matching_cache_after_durable_append(tmp_path):
    ledger = tmp_path / "c.jsonl"
    cache = tmp_path / "cache.json"
    cache.write_text('{"owed_authored_blobs": 5}', encoding="utf-8")

    mod.append_certification(
        "d" * 40,
        "rebuildable fixture with committed generator",
        "ddm_rvf1",
        ledger=ledger,
        cache=cache,
    )

    assert ledger.read_text(encoding="utf-8").strip()
    assert not cache.exists()


def test_cache_invalidation_failure_does_not_partially_append_certification(tmp_path):
    ledger = tmp_path / "c.jsonl"
    cache = tmp_path / "cache-dir"
    cache.mkdir()

    with pytest.raises(mod.AuditError, match="not appended"):
        mod.append_certification(
            "f" * 40,
            "rebuildable fixture with committed generator",
            "ddm_rv16",
            ledger=ledger,
            cache=cache,
        )

    assert not ledger.exists(), "a refused call must not commit a hidden ledger row"
    assert cache.is_dir(), "the failed invalidation target must remain visible"


def test_certification_refuses_ledger_as_its_own_cache(tmp_path):
    ledger = tmp_path / "c.jsonl"
    with pytest.raises(mod.AuditError, match="distinct paths"):
        mod.append_certification(
            "f" * 40,
            "rebuildable fixture with committed generator",
            "ddm_rv16",
            ledger=ledger,
            cache=ledger,
        )
    assert not ledger.exists()


def test_alternate_certification_ledger_does_not_invalidate_live_cache(
    tmp_path, monkeypatch
):
    live_cache = tmp_path / "live-cache.json"
    live_cache.write_text("still valid for the unrelated ledger", encoding="utf-8")
    monkeypatch.setattr(mod, "CACHE", live_cache)

    mod.append_certification(
        "e" * 40,
        "rebuildable fixture with committed generator",
        "ddm_rvf1",
        ledger=tmp_path / "alternate.jsonl",
    )

    assert live_cache.read_text(encoding="utf-8") == "still valid for the unrelated ledger"


def test_load_certified_tolerates_malformed_rows(tmp_path):
    ledger = tmp_path / "c.jsonl"
    ledger.write_text('{"blob_sha1": "c' + "c" * 38 + '"}\nnot json at all\n\n')
    assert len(mod.load_certified(ledger)) == 1


# ---------------------------------------------------------------------------------- fail-closed


def test_zero_reachable_blobs_refuses_rather_than_reporting_phantom_debt(monkeypatch):
    """An empty identity set would mark every SSD file as owed. Refuse instead of lying."""
    monkeypatch.setattr(mod, "_git", lambda *a, **k: "")
    with pytest.raises(mod.AuditError):
        mod.reachable_git_blobs()


def test_git_failure_raises_audit_error(monkeypatch):
    def boom(*a, **k):
        raise OSError("git missing")

    monkeypatch.setattr(mod.subprocess, "run", boom)
    with pytest.raises(mod.AuditError):
        mod.odb_git_blobs()


# --------------------------------------------------------------------------------------- surface


def test_summary_is_small_and_carries_the_owed_count(tmp_path):
    f = tmp_path / "arm" / "x.py"
    f.parent.mkdir(parents=True)
    f.write_text("x = 1\n")
    summary = mod.summarize(mod.scan(roots=[tmp_path], reachable=set(), odb=set()))
    assert summary["owed_authored_blobs"] == 1
    assert "owed" not in summary  # per-blob rows must not leak into the cadence cache
    assert len(json.dumps(summary)) < 4000


def test_fmt_names_the_debt_and_the_two_dispositions(tmp_path):
    f = tmp_path / "arm" / "x.py"
    f.parent.mkdir(parents=True)
    f.write_text("x = 1\n")
    text = mod.fmt(mod.scan(roots=[tmp_path], reachable=set(), odb=set()))
    assert "AUTHORED — OWED" in text
    assert "COMMIT them" in text and "--certify" in text


def test_strict_exit_code_is_2_only_when_debt_exists(tmp_path, capsys):
    (tmp_path / "arm").mkdir()
    assert mod.main(["--root", str(tmp_path), "--strict"]) == 0
    (tmp_path / "arm" / "x.py").write_text("x = 1\n")
    assert mod.main(["--root", str(tmp_path), "--strict"]) == 2
    capsys.readouterr()


def test_default_exit_code_is_0_even_with_debt(tmp_path, capsys):
    """Warn-only by default: observability never blocks a turn."""
    (tmp_path / "arm").mkdir()
    (tmp_path / "arm" / "x.py").write_text("x = 1\n")
    assert mod.main(["--root", str(tmp_path)]) == 0
    capsys.readouterr()
