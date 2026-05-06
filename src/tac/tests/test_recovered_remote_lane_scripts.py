from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


def _script(name: str) -> str:
    return (REPO_ROOT / "scripts" / name).read_text(encoding="utf-8")


def test_sjkl_recovered_lane_keeps_exact_eval_custody() -> None:
    text = _script("remote_lane_sjkl_c067.sh")

    assert "lane_sjkl_c067" in text
    assert "claim `lane_sjkl_c067`" in text
    assert "scripts/remote_archive_only_eval.sh" in text
    assert 'export KEEP_EVAL_WORK="${KEEP_EVAL_WORK:-1}"' in text
    assert 'CONTEST_AUTH_EVAL_CUSTODY_FLAGS="--keep-work-dir --work-dir"' in text
    assert "contest_auth_eval.json" in text
    assert "score claim until contest_auth_eval.json" in text


def test_pr79_recovered_lane_is_proxy_only_until_exact_eval() -> None:
    text = _script("remote_lane_pr79_segaction_search.sh")

    assert "lane_pr79_segaction_search" in text
    assert "PUBLIC_COMMIT=" in text
    assert 'git checkout --detach "$PUBLIC_COMMIT"' in text
    assert "CLONED_PUBLIC_COMMIT" in text
    assert "PATCH_DIFF_SHA256" in text
    assert "BROTLI_PACKAGE" in text
    assert "parse_pr79_archive" in text
    assert "write_pr79_single_member_archive" in text
    assert '"score_claim": False' in text
    assert "remote_cuda_proxy_search_only_until_exact_t4_auth_eval" in text
    assert "archive.zip -> inflate.sh -> upstream/evaluate.py" in text


def test_qfaithful_recovered_lane_preserves_auth_eval_workdir() -> None:
    text = _script("remote_lane_q_faithful_jointgen.sh")

    assert "lane_q_faithful_jointgen_88k" in text
    assert "experiments/contest_auth_eval.py" in text
    assert "--keep-work-dir" in text
    assert '--work-dir "$LOG_DIR/eval_work"' in text
    assert "predicted band: [0.40, 0.80] [contest-CUDA]" in text
