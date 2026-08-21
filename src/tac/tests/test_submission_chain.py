# SPDX-License-Identifier: MIT
"""Tests for the canonical submission chain.

Every guard here has an EXECUTED POSITIVE CONTROL: a deliberately broken input
that the guard must REFUSE.  A guard with no demonstrated failing case is not a
landed guard -- it is a comment (CLAUDE.md 'Comment-only contracts are
FORBIDDEN' + the vacuity-equals-pass class).
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from tac.submission_chain import (
    ChainPaths,
    ChainReceipt,
    SubmissionChainError,
    audit_runtime_tree,
    axis_and_authority,
    build_byte_ledger,
    parse_evaluate_report,
    refuse_transient_path,
    run_inflate,
    stage_submission,
    verify_archive_identity,
)


# --------------------------------------------------------------------------- #
# fixtures: a minimal but REAL ix2 container built with the canonical encoder
# --------------------------------------------------------------------------- #
@pytest.fixture()
def container_payload() -> bytes:
    from tac.optimization.ddm_ix2_archive_container import build_payload

    bulk = bytes(range(256)) * 40
    sections = [b"\x00\x01\x02\x03", b"IX2REN01rest", b'{"activation":1}', b"PFS1WPD1xyz"]
    return build_payload(bulk, sections)


@pytest.fixture()
def archive(tmp_path, container_payload):
    from tac.optimization.ddm_ix2_archive_container import build_single_member_zip

    path = tmp_path / "archive.zip"
    path.write_bytes(build_single_member_zip(container_payload))
    return path


# --------------------------------------------------------------------------- #
# byte ledger
# --------------------------------------------------------------------------- #
def test_byte_ledger_closes_exactly_on_a_real_container(archive, container_payload):
    led = build_byte_ledger(archive)
    assert led.closes()
    assert led.residual_bytes == 0
    assert led.accounted_bytes == led.archive_bytes == archive.stat().st_size
    # The accounting identity the ledger exists to keep honest.
    assert (
        led.payload_header_bytes
        + led.bulk_bytes
        + led.joint_count_byte
        + led.joint_coded_bytes
        == led.payload_bytes
    )
    assert led.archive_bytes == led.payload_bytes + led.zip_framing_bytes
    assert len(led.sections) == 4
    assert [s.name for s in led.sections] == ["config", "renderer", "selector", "pose_warp"]


def test_byte_ledger_separates_raw_section_size_from_counted_coded_size(archive):
    """Raw section bytes are PRE-coder and are not a rate cost; the ledger must
    keep the two numbers distinct so neither can be quoted as the other."""
    led = build_byte_ledger(archive)
    assert led.joint_raw_bytes == sum(s.raw_bytes for s in led.sections)
    assert led.joint_coder_saving_bytes == led.joint_raw_bytes - led.joint_coded_bytes
    # They are genuinely different numbers -- not an alias.
    assert led.joint_raw_bytes != led.joint_coded_bytes


def test_byte_ledger_refuses_missing_archive(tmp_path):
    with pytest.raises(SubmissionChainError, match="archive not found"):
        build_byte_ledger(tmp_path / "nope.zip")


def test_byte_ledger_refuses_archive_without_payload_member(tmp_path):
    """POSITIVE CONTROL: a zip that is well-formed but carries the wrong member
    must be REFUSED, not silently ledgered as empty."""
    path = tmp_path / "archive.zip"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("wrong_name.bin", b"payload")
    with pytest.raises(SubmissionChainError, match=r"has no .0\.bin. member"):
        build_byte_ledger(path)


def test_byte_ledger_refuses_a_corrupted_payload(tmp_path, container_payload):
    """POSITIVE CONTROL: truncating the payload must raise, never return a
    ledger with a swallowed residual."""
    from tac.optimization.ddm_ix2_archive_container import build_single_member_zip

    path = tmp_path / "archive.zip"
    from tac.optimization.ddm_ix2_archive_container import IX2ContainerError

    path.write_bytes(build_single_member_zip(container_payload[: len(container_payload) // 2]))
    with pytest.raises(IX2ContainerError, match="truncated"):
        build_byte_ledger(path)


def test_byte_ledger_closure_can_actually_fire_on_a_zip_comment(tmp_path, container_payload):
    """THE control for the vacuity bug this guard shipped with.

    An earlier draft back-computed the ZIP framing by subtraction, which made
    the residual identically zero BY ALGEBRA -- the guard could not fail on any
    input.  A trailing archive comment adds bytes that no member accounts for,
    so a non-circular ledger MUST refuse it.
    """
    from tac.optimization.ddm_ix2_archive_container import build_single_member_zip

    path = tmp_path / "archive.zip"
    path.write_bytes(build_single_member_zip(container_payload))
    clean = build_byte_ledger(path)
    assert clean.closes()

    with zipfile.ZipFile(path, "a") as zf:
        zf.comment = b"x" * 64
    with pytest.raises(SubmissionChainError, match="did not close"):
        build_byte_ledger(path)


def test_byte_ledger_closure_fires_on_a_non_stored_member(tmp_path, container_payload):
    """A DEFLATE member is a different rate cost than a STORED one; the ledger
    must not silently accept a repack that changed the compression method."""
    path = tmp_path / "archive.zip"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        info = zipfile.ZipInfo("0.bin")
        info.compress_type = zipfile.ZIP_DEFLATED
        info.date_time = (1980, 1, 1, 0, 0, 0)
        zf.writestr(info, container_payload)
    # SECOND vacuity found in this module's own tests: the first version of this
    # test carried an `or compress_type != 0` escape clause, so it passed while
    # the ledger happily ACCEPTED the deflate repack. The guard now refuses.
    with pytest.raises(SubmissionChainError, match="non-STORED member"):
        build_byte_ledger(path)

    # ...and the refusal is opt-out-able for a caller that genuinely wants it.
    led = build_byte_ledger(path, require_stored=False)
    assert led.zip_members[0].compress_type == zipfile.ZIP_DEFLATED


def test_byte_ledger_records_the_independent_reencode_check(archive):
    led = build_byte_ledger(archive)
    assert led.payload_reencodes_identically is True
    # The two closure legs are independent, not aliases of one another.
    assert led.predicted_framing_bytes == led.zip_framing_bytes


# --------------------------------------------------------------------------- #
# runtime custody -- the transitive-closure property
# --------------------------------------------------------------------------- #
def _write_runtime_tree(root):
    """A staged tree whose adapter is reachable ONLY through a second hop.

    This is the shipped topology in miniature: ``inflate_runner`` does not import
    ``adapter`` at all; ``token_coder`` does.
    """
    root.mkdir(parents=True, exist_ok=True)
    (root / "inflate.sh").write_text(
        '#!/usr/bin/env bash\nset -euo pipefail\npython "$HERE/inflate_runner.py" "$1" "$2" "$3"\n'
    )
    (root / "inflate_runner.py").write_text("from token_coder import decode\n")
    (root / "token_coder.py").write_text(
        "from adapter import (\n    thing,\n)\n\ndef decode():\n    return thing\n"
    )
    (root / "adapter.py").write_text("thing = 1\n")
    (root / "orphan.py").write_text("# imported by nobody\n")
    return root


def test_custody_import_closure_is_transitive(tmp_path):
    """THE control that would have caught the real bug: a second-hop dependency
    must be reported REACHED, not as deletable dead weight."""
    tree = _write_runtime_tree(tmp_path / "rt")
    custody = audit_runtime_tree(tree, repo_root=tmp_path)
    reached = {f.staged_name: f.reached for f in custody.files}
    assert reached["adapter.py"] is True, "second-hop dependency must be reached"
    assert reached["token_coder.py"] is True
    assert "adapter.py" not in custody.unreached_files


def test_custody_reports_a_genuinely_unreached_file(tmp_path):
    """POSITIVE CONTROL for the other direction: a file nothing imports MUST be
    reported unreached, or the audit can never find real dead weight."""
    tree = _write_runtime_tree(tmp_path / "rt")
    custody = audit_runtime_tree(tree, repo_root=tmp_path)
    assert "orphan.py" in custody.unreached_files


def test_custody_classifies_identical_diverged_and_unmapped(tmp_path):
    tree = _write_runtime_tree(tmp_path / "rt")
    repo = tmp_path / "repo"
    (repo / "src").mkdir(parents=True)
    # identical counterpart
    (repo / "src" / "adapter_repo.py").write_text("thing = 1\n")
    # diverged counterpart
    (repo / "src" / "token_repo.py").write_text("# HEAD moved on\nfrom adapter import thing\n")
    custody = audit_runtime_tree(
        tree,
        repo_root=repo,
        repo_map={
            "adapter.py": "src/adapter_repo.py",
            "token_coder.py": "src/token_repo.py",
        },
    )
    verdicts = {f.staged_name: f.verdict for f in custody.files}
    assert verdicts["adapter.py"] == "IDENTICAL"
    assert verdicts["token_coder.py"] == "DIVERGED"
    assert verdicts["inflate_runner.py"] == "UNMAPPED"
    assert custody.identical_count == 1
    assert custody.diverged_count == 1
    # A DIVERGED row must carry a reason; a bare verdict is not custody.
    reason = next(f.divergence_reason for f in custody.files if f.verdict == "DIVERGED")
    assert reason and "vendored" in reason


def test_custody_refuses_missing_entry_script(tmp_path):
    tree = tmp_path / "rt"
    tree.mkdir()
    with pytest.raises(SubmissionChainError, match="entry script missing"):
        audit_runtime_tree(tree, repo_root=tmp_path)


# --------------------------------------------------------------------------- #
# path configuration + transient-path refusal
# --------------------------------------------------------------------------- #
def test_refuse_transient_path_rejects_tmp():
    with pytest.raises(SubmissionChainError, match="transient tmp root"):
        refuse_transient_path("/tmp/evidence.json", "receipt_path")


def test_refuse_transient_path_allows_durable(tmp_path):
    # tmp_path itself may live under /private/var on macOS, which is durable
    # enough for a test; use an explicit durable-looking path instead.
    assert refuse_transient_path("/Volumes/Data/pact/x.json", "p").name == "x.json"


def test_chain_paths_precedence_arg_beats_env_beats_default(tmp_path):
    env = {"PACT_UPSTREAM_DIR": str(tmp_path / "from_env")}
    p = ChainPaths.from_env(repo_root=tmp_path, env=env)
    assert p.upstream_dir == tmp_path / "from_env"
    p2 = ChainPaths.from_env(repo_root=tmp_path, upstream_dir=tmp_path / "explicit", env=env)
    assert p2.upstream_dir == tmp_path / "explicit"
    p3 = ChainPaths.from_env(repo_root=tmp_path, env={})
    assert p3.upstream_dir == tmp_path / "upstream"


def test_chain_paths_preflight_reports_every_missing_path_at_once(tmp_path):
    """A preflight that dies on the first miss makes the operator discover the
    set one round-trip at a time; this one returns them all."""
    p = ChainPaths.from_env(repo_root=tmp_path, env={})
    missing = p.preflight()
    assert len(missing) >= 3
    assert any(m.startswith("upstream_dir=") for m in missing)
    assert any(m.startswith("videos_dir=") for m in missing)


# --------------------------------------------------------------------------- #
# axis discipline
# --------------------------------------------------------------------------- #
def test_axis_refuses_mps():
    with pytest.raises(SubmissionChainError, match="NEVER a score authority"):
        axis_and_authority("mps")


def test_axis_cuda_is_authority():
    assert axis_and_authority("cuda") == ("[contest-CUDA]", "authority")


def test_axis_cpu_is_advisory_off_linux_x86_64(monkeypatch):
    monkeypatch.setattr("platform.system", lambda: "Darwin")
    axis, authority = axis_and_authority("cpu")
    assert axis == "[macOS-CPU advisory]"
    assert authority == "advisory"


def test_axis_rejects_unknown_device():
    with pytest.raises(SubmissionChainError, match="unknown eval device"):
        axis_and_authority("tpu")


# --------------------------------------------------------------------------- #
# evaluate report parsing
# --------------------------------------------------------------------------- #
_GOOD_REPORT = """
Evaluation results over 600 samples
Average PoseNet Distortion: 0.0012340000
Average SegNet Distortion: 0.0043117947
Compression Rate: 0.0094235
Final score (100*seg + sqrt(10*pose) + 25*rate) = 0.7910689
"""


def test_parse_evaluate_report_extracts_all_components():
    parsed = parse_evaluate_report(_GOOD_REPORT)
    assert parsed["n_samples"] == 600
    assert parsed["d_seg"] == pytest.approx(0.0043117947)
    assert parsed["final_score"] == pytest.approx(0.7910689)


def test_parse_evaluate_report_refuses_a_missing_component():
    """POSITIVE CONTROL: never fabricate a score component."""
    broken = _GOOD_REPORT.replace("Average SegNet Distortion: 0.0043117947", "")
    with pytest.raises(SubmissionChainError, match="missing 'd_seg'"):
        parse_evaluate_report(broken)


def test_parse_evaluate_report_treats_absent_sample_line_as_unknown():
    """Absence of the sample-count line is UNKNOWN, not a partial-sample claim."""
    no_count = _GOOD_REPORT.replace("Evaluation results over 600 samples", "")
    assert parse_evaluate_report(no_count)["n_samples"] is None


# --------------------------------------------------------------------------- #
# stage + byte identity
# --------------------------------------------------------------------------- #
def test_stage_submission_is_deterministic_and_byte_identical(tmp_path, container_payload):
    src = tmp_path / "src"
    src.mkdir()
    (src / "inflate.sh").write_text("#!/usr/bin/env bash\ntrue\n")
    a = stage_submission(container_payload, dest=tmp_path / "d1", runtime_src=src,
                         runtime_files=("inflate.sh",))
    b = stage_submission(container_payload, dest=tmp_path / "d2", runtime_src=src,
                         runtime_files=("inflate.sh",))
    assert a.read_bytes() == b.read_bytes()
    ident = verify_archive_identity(b, expected_sha256=None, expected_bytes=a.stat().st_size)
    assert ident["bytes_match"] is True


def test_stage_submission_refuses_missing_runtime_file(tmp_path, container_payload):
    src = tmp_path / "src"
    src.mkdir()
    with pytest.raises(SubmissionChainError, match="runtime file missing"):
        stage_submission(container_payload, dest=tmp_path / "d", runtime_src=src,
                         runtime_files=("inflate.sh",))


def test_verify_archive_identity_reports_divergence_rather_than_raising(tmp_path, container_payload):
    from tac.optimization.ddm_ix2_archive_container import build_single_member_zip

    path = tmp_path / "archive.zip"
    path.write_bytes(build_single_member_zip(container_payload))
    res = verify_archive_identity(path, expected_sha256="0" * 64, expected_bytes=1)
    assert res["byte_identical"] is False
    assert res["sha_matches"] is False
    assert res["bytes_match"] is False
    assert res["sha256"] != "0" * 64  # the ACTUAL value is reported, for diagnosis


# --------------------------------------------------------------------------- #
# inflate fail-closed -- the three refused modes
# --------------------------------------------------------------------------- #
def _mk_sub(tmp_path, script_body):
    sub = tmp_path / "sub"
    sub.mkdir(parents=True, exist_ok=True)
    (sub / "inflate.sh").write_text(script_body)
    (tmp_path / "arch").mkdir(exist_ok=True)
    (tmp_path / "names.txt").write_text("0\n")
    return sub


def test_run_inflate_refuses_nonzero_returncode(tmp_path):
    sub = _mk_sub(tmp_path, "#!/usr/bin/env bash\nexit 3\n")
    with pytest.raises(SubmissionChainError, match="inflate FAILED rc=3"):
        run_inflate(sub, archive_dir=tmp_path / "arch", out_dir=tmp_path / "out",
                    video_names_file=tmp_path / "names.txt")


def test_run_inflate_refuses_rc0_with_no_output(tmp_path):
    """POSITIVE CONTROL for vacuity-equals-pass: exit 0 producing nothing is a
    FAILURE, not a decode."""
    sub = _mk_sub(tmp_path, "#!/usr/bin/env bash\nexit 0\n")
    with pytest.raises(SubmissionChainError, match="produced NO files"):
        run_inflate(sub, archive_dir=tmp_path / "arch", out_dir=tmp_path / "out",
                    video_names_file=tmp_path / "names.txt")


def test_run_inflate_refuses_rc0_with_zero_byte_output(tmp_path):
    sub = _mk_sub(tmp_path, '#!/usr/bin/env bash\nmkdir -p "$2"\n: > "$2/0.raw"\nexit 0\n')
    with pytest.raises(SubmissionChainError, match="0 bytes"):
        run_inflate(sub, archive_dir=tmp_path / "arch", out_dir=tmp_path / "out",
                    video_names_file=tmp_path / "names.txt")


def test_run_inflate_accepts_a_real_decode(tmp_path):
    sub = _mk_sub(tmp_path, '#!/usr/bin/env bash\nmkdir -p "$2"\nprintf xyz > "$2/0.raw"\nexit 0\n')
    res = run_inflate(sub, archive_dir=tmp_path / "arch", out_dir=tmp_path / "out",
                      video_names_file=tmp_path / "names.txt")
    assert res.returncode == 0
    assert res.raw_files == 1
    assert res.raw_bytes == 3


def test_run_inflate_refuses_missing_entry_script(tmp_path):
    sub = tmp_path / "sub"
    sub.mkdir()
    with pytest.raises(SubmissionChainError, match="entry script missing"):
        run_inflate(sub, archive_dir=tmp_path, out_dir=tmp_path / "out",
                    video_names_file=tmp_path / "names.txt")


# --------------------------------------------------------------------------- #
# receipt
# --------------------------------------------------------------------------- #
def test_receipt_round_trips_and_defaults_to_non_promotable(tmp_path):
    import json

    r = ChainReceipt(archive_bytes=353805, archive_sha256="c72e", score_axis="[macOS-CPU advisory]")
    out = r.write(tmp_path / "receipt.json")
    loaded = json.loads(out.read_text())
    assert loaded["schema"] == "tac_submission_chain_receipt.v1"
    assert loaded["archive_bytes"] == 353805
    assert loaded["score_claim"] is False
    assert loaded["promotion_eligible"] is False


def test_receipt_refuses_a_tmp_destination():
    r = ChainReceipt()
    with pytest.raises(SubmissionChainError, match="transient tmp root"):
        r.write("/tmp/receipt.json")


# --------------------------------------------------------------------------- #
# report custody -- do not clobber a prior run's evidence
# --------------------------------------------------------------------------- #
def test_evaluate_does_not_clobber_an_existing_report(tmp_path, monkeypatch):
    """POSITIVE CONTROL, from a live near-miss.

    On 2026-08-04 this function was about to overwrite the ddm_pu2 ``report.txt``
    that ddm_si1 had cited hours earlier, because the path was hardcoded. A
    submission dir is DURABLE evidence, not scratch.
    """
    import subprocess as _sp

    from tac.submission_chain import run_upstream_evaluate

    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "archive.zip").write_bytes(b"PK\x05\x06" + b"\x00" * 18)
    original = sub / "report.txt"
    original.write_text("ORIGINAL EVIDENCE -- must survive\n")
    up = tmp_path / "upstream"
    up.mkdir()
    (up / "evaluate.py").write_text("# stub\n")
    (tmp_path / "videos").mkdir()
    (tmp_path / "names.txt").write_text("0\n")

    report_text = (
        "Evaluation results over 600 samples\n"
        "Average PoseNet Distortion: 0.00154519\n"
        "Average SegNet Distortion: 0.00431179\n"
        "Compression Rate: 0.00942337\n"
        "Final score = 0.79\n"
    )
    # Patch the seam run_upstream_evaluate ACTUALLY calls. It routes through
    # tac.process_group_kill.run_in_process_group so the 30-min wall binds the whole
    # tree (evaluate.py spawns DataLoader workers a child-scoped timeout would orphan).
    monkeypatch.setattr(
        "tac.submission_chain.run_in_process_group",
        lambda cmd, **kw: _sp.CompletedProcess(cmd, 0, report_text, ""),
    )
    res = run_upstream_evaluate(
        sub, upstream_dir=up, videos_dir=tmp_path / "videos",
        video_names_file=tmp_path / "names.txt", archive_bytes=353805, device="cpu",
    )
    assert original.read_text() == "ORIGINAL EVIDENCE -- must survive\n"
    assert Path(res.report_path) != original
    # ...and the components are recomputed, reproducing the live frontier row.
    assert res.recomputed_score == pytest.approx(0.7910689, abs=1e-6)


def test_evaluate_honours_an_explicit_report_path(tmp_path, monkeypatch):
    import subprocess as _sp

    from tac.submission_chain import run_upstream_evaluate

    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "archive.zip").write_bytes(b"PK\x05\x06" + b"\x00" * 18)
    up = tmp_path / "upstream"
    up.mkdir()
    (up / "evaluate.py").write_text("# stub\n")
    (tmp_path / "videos").mkdir()
    (tmp_path / "names.txt").write_text("0\n")
    want = tmp_path / "custom_report.txt"
    monkeypatch.setattr(
        "tac.submission_chain.run_in_process_group",
        lambda cmd, **kw: _sp.CompletedProcess(
            cmd, 0,
            "Evaluation results over 600 samples\nAverage PoseNet Distortion: 1e-3\n"
            "Average SegNet Distortion: 1e-3\nCompression Rate: 1e-3\nFinal score = 0.2\n",
            "",
        ),
    )
    res = run_upstream_evaluate(
        sub, upstream_dir=up, videos_dir=tmp_path / "videos",
        video_names_file=tmp_path / "names.txt", archive_bytes=1, device="cpu",
        report_path=want,
    )
    assert Path(res.report_path) == want
