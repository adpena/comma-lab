# SPDX-License-Identifier: MIT
from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
from pathlib import Path

from tac.submission_packet.public_pr_audit import (
    CommandResult,
    PublicSubmissionAuditConfig,
    audit_public_submission_pr,
    audit_zip,
    body_patch_suggestions,
    extract_archive_bytes,
    extract_archive_sha256,
    extract_archive_url,
    extract_submission_name,
    parse_github_release_asset_url,
    self_test_result,
)


def _clean_body(*, head: str = "abcdef0123456789abcdef0123456789abcdef01") -> str:
    return f"""# submission name:
sample_submission

# upload zipped `archive.zip`
[archive.zip](https://github.com/adpena/comma_video_compression_challenge/releases/download/v1/archive.zip)
Runtime tree at PR head [`{head[:7]}`](https://github.com/adpena/comma_video_compression_challenge/tree/{head}/submissions/sample_submission).

# report.txt
Archive SHA-256: {"a" * 64}
Archive size bytes: 123
Final score: 0.19 [contest-CPU]

# does your submission require gpu for evaluation (inflation)?
no

# did you include the compression script? and want it to be merged?
yes

# is this submission competitive or innovative? explain why
Competitive: yes. I built a selector over @SajayR's PR #101 substrate.
"""


def test_extract_pr_body_fields() -> None:
    body = _clean_body()

    assert extract_submission_name(body) == "sample_submission"
    assert extract_archive_url(body).endswith("/releases/download/v1/archive.zip")
    assert extract_archive_sha256(body) == "a" * 64
    assert extract_archive_bytes(body) == 123
    assert parse_github_release_asset_url(extract_archive_url(body) or "") == (
        "adpena",
        "comma_video_compression_challenge",
        "v1",
        "archive.zip",
    )


def test_body_patch_suggestions_catch_stale_runtime_head() -> None:
    body = _clean_body(head="0" * 40)
    suggestions = body_patch_suggestions(
        body,
        head_sha="abcdef0123456789abcdef0123456789abcdef01",
        submission_name="sample_submission",
    )

    assert suggestions
    assert "current head" in suggestions[0]


def test_audit_zip_flags_hidden_macos_members(tmp_path: Path) -> None:
    archive = tmp_path / "archive.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("__MACOSX/._x", b"junk")
        zf.writestr(".hidden", b"junk")

    result = audit_zip(archive)

    assert "zip_contains_macos_resource_fork_dir" in result.blockers
    assert "zip_contains_hidden_member" in result.blockers
    assert not result.ok


def test_self_test_is_no_network_and_clean() -> None:
    result = self_test_result()

    assert result.overall_clean
    assert result.submission_name == "sample_submission"
    assert result.archive_sha256 == "a" * 64


def test_full_audit_uses_runner_and_fails_on_public_hazards(tmp_path: Path, monkeypatch) -> None:
    import tac.submission_packet.public_pr_audit as module

    source = tmp_path / "source"
    submission = source / "submissions" / "sample_submission"
    (submission / "src").mkdir(parents=True)
    (submission / "encoder").mkdir()
    for rel, text in {
        "README.md": "# sample\n",
        "requirements.txt": "torch\n",
        "inflate.py": "print('ok')\n",
        "src/__init__.py": "",
        "encoder/build_pr101_frame_exploit_selector_packet.py": "import argparse\nargparse.ArgumentParser().parse_args()\n",
        "encoder/frame_exploit_segnet_posenet_sweep.py": "import argparse\nargparse.ArgumentParser().parse_args()\n",
        "compress.sh": "#!/usr/bin/env bash\nset -euo pipefail\npython3 encoder/build_pr101_frame_exploit_selector_packet.py \"$@\"\n",
        "inflate.sh": "#!/usr/bin/env bash\nset -euo pipefail\npython3 inflate.py\n",
    }.items():
        path = submission / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    (submission / "README.md").write_text("Contains /Users/adpena/private path\n", encoding="utf-8")
    (submission / "compress.sh").chmod(0o755)
    (submission / "inflate.sh").chmod(0o755)

    archive_bytes = b"payload"
    archive_sha = "b" * 64
    body = _clean_body().replace("a" * 64, archive_sha).replace("123", str(120))

    def fake_download(_url: str, output_path: Path, *, timeout_s: float) -> None:
        with zipfile.ZipFile(output_path, "w") as zf:
            zf.writestr("x", archive_bytes)

    monkeypatch.setattr(module, "_download_archive", fake_download)

    def runner(args, cwd=None, timeout_s=None):
        if args[:3] == ["gh", "pr", "view"]:
            return CommandResult(
                tuple(args),
                0,
                json.dumps(
                    {
                        "state": "OPEN",
                        "isDraft": False,
                        "mergeable": "MERGEABLE",
                        "headRefOid": "abcdef0123456789abcdef0123456789abcdef01",
                        "headRefName": "branch",
                        "headRepository": {"name": "repo"},
                        "headRepositoryOwner": {"login": "owner"},
                        "body": body,
                        "comments": [],
                        "url": "https://github.com/example/repo/pull/1",
                    }
                ),
                "",
            )
        if args[:2] == ["gh", "api"]:
            return CommandResult(
                tuple(args),
                0,
                json.dumps(
                    {
                        "draft": False,
                        "prerelease": False,
                        "assets": [
                            {
                                "name": "archive.zip",
                                "size": 120,
                                "digest": f"sha256:{archive_sha}",
                            }
                        ],
                    }
                ),
                "",
            )
        if args[:2] == ["git", "clone"]:
            shutil.copytree(source, Path(args[-1]))
            return CommandResult(tuple(args), 0, "", "")
        if args[:2] == ["git", "ls-files"]:
            files = [
                str(path.relative_to(cwd))
                for path in sorted((cwd / "submissions" / "sample_submission").rglob("*"))
                if path.is_file()
            ]
            return CommandResult(tuple(args), 0, "\n".join(files), "")
        return CommandResult(tuple(args), 0, "", "")

    result = audit_public_submission_pr(
        PublicSubmissionAuditConfig(
            target_repo="commaai/comma_video_compression_challenge",
            pr_number=110,
            expected_archive_sha256=archive_sha,
            expected_archive_bytes=120,
            work_dir=tmp_path / "work",
        ),
        runner=runner,
    )

    assert not result.overall_clean
    assert any(f.rule == "local_absolute_path_users" for f in result.findings)


def test_inflate_smoke_records_raw_output_sha(tmp_path: Path, monkeypatch) -> None:
    import tac.submission_packet.public_pr_audit as module

    source = tmp_path / "source"
    submission = source / "submissions" / "sample_submission"
    submission.mkdir(parents=True)
    for rel, text in {
        "README.md": "# sample\n",
        "requirements.txt": "torch\n",
        "inflate.py": "print('ok')\n",
        "inflate.sh": "#!/usr/bin/env bash\nset -euo pipefail\npython3 inflate.py\n",
    }.items():
        path = submission / rel
        path.write_text(text, encoding="utf-8")
    (submission / "inflate.sh").chmod(0o755)

    archive_sha = "b" * 64
    body = _clean_body().replace("a" * 64, archive_sha).replace("123", str(120))
    raw_bytes = b"raw-output"
    raw_sha = hashlib.sha256(raw_bytes).hexdigest()

    def fake_download(_url: str, output_path: Path, *, timeout_s: float) -> None:
        with zipfile.ZipFile(output_path, "w") as zf:
            zf.writestr("x", b"payload")

    monkeypatch.setattr(module, "_download_archive", fake_download)

    def runner(args, cwd=None, timeout_s=None):
        if args[:3] == ["gh", "pr", "view"]:
            return CommandResult(
                tuple(args),
                0,
                json.dumps(
                    {
                        "state": "OPEN",
                        "isDraft": False,
                        "mergeable": "MERGEABLE",
                        "headRefOid": "abcdef0123456789abcdef0123456789abcdef01",
                        "headRefName": "branch",
                        "headRepository": {"name": "repo"},
                        "headRepositoryOwner": {"login": "owner"},
                        "body": body,
                        "comments": [],
                        "url": "https://github.com/example/repo/pull/1",
                    }
                ),
                "",
            )
        if args[:2] == ["gh", "api"]:
            return CommandResult(
                tuple(args),
                0,
                json.dumps(
                    {
                        "draft": False,
                        "prerelease": False,
                        "assets": [{"name": "archive.zip", "size": 120, "digest": f"sha256:{archive_sha}"}],
                    }
                ),
                "",
            )
        if args[:2] == ["git", "clone"]:
            shutil.copytree(source, Path(args[-1]))
            return CommandResult(tuple(args), 0, "", "")
        if args[:2] == ["git", "ls-files"]:
            files = [
                str(path.relative_to(cwd))
                for path in sorted((cwd / "submissions" / "sample_submission").rglob("*"))
                if path.is_file()
            ]
            return CommandResult(tuple(args), 0, "\n".join(files), "")
        if len(args) >= 4 and args[0] == "bash" and str(args[1]).endswith("inflate.sh"):
            Path(args[3]).mkdir(parents=True, exist_ok=True)
            (Path(args[3]) / "0.raw").write_bytes(raw_bytes)
            return CommandResult(tuple(args), 0, "", "")
        return CommandResult(tuple(args), 0, "", "")

    result = audit_public_submission_pr(
        PublicSubmissionAuditConfig(
            target_repo="commaai/comma_video_compression_challenge",
            pr_number=110,
            expected_output_sha256=raw_sha,
            run_inflate_smoke=True,
            work_dir=tmp_path / "work",
        ),
        runner=runner,
    )

    assert result.overall_clean
    assert result.inflate_smoke is not None
    assert result.inflate_smoke.ok
    assert result.inflate_smoke.raw_sha256 == raw_sha
    assert result.as_dict()["inflate_smoke"]["raw_sha256"] == raw_sha
