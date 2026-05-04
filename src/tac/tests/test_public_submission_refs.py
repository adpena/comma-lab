from __future__ import annotations

import argparse

import pytest

from tac.public_submission_refs import parse_public_pr_refs_csv, public_submission_refs_for_manifest


def test_public_submission_refs_are_canonical_github_pull_urls() -> None:
    refs = public_submission_refs_for_manifest(
        ("85", "pr91", "PR95", "96", "PR97", "98", "PR99", "PR100", "101", "PR106", "PR85")
    )

    assert list(refs) == ["PR85", "PR91", "PR95", "PR96", "PR97", "PR98", "PR99", "PR100", "PR101", "PR106"]
    assert refs["PR85"]["pr_number"] == 85
    assert refs["PR91"]["url"].endswith("/pull/91")
    assert refs["PR95"]["family"] == "hnerv_muon_single_member_codec"
    assert refs["PR96"]["url"].endswith("/pull/96")
    assert refs["PR97"]["family"] == "vibe_coder_final_boss_h3_sidecar"
    assert refs["PR98"]["family"] == "hnerv_muon_finetuned_from_pr95"
    assert refs["PR99"]["url"].endswith("/pull/99")
    assert refs["PR100"]["url"].endswith("/pull/100")
    assert refs["PR101"]["url"].endswith("/pull/101")
    assert refs["PR106"]["url"].endswith("/pull/106")


def test_public_submission_refs_fail_closed_on_unknown_pr() -> None:
    with pytest.raises(ValueError, match="unknown public PR reference PR999"):
        parse_public_pr_refs_csv("PR999")


def test_lightning_queue_metadata_expands_source_pr_urls() -> None:
    from scripts import launch_lightning_batch_job

    args = argparse.Namespace(
        queue_metadata=["lane=pr91_hpm1", "source_prs=PR85,91"],
        allow_skip_remote_preflight_reason=None,
        allow_missing_dispatch_claim_reason=None,
    )

    metadata = launch_lightning_batch_job._queue_metadata_from_args(args)

    assert metadata["lane"] == "pr91_hpm1"
    assert metadata["source_prs"] == "PR85,PR91"
    assert "pull/85" in metadata["source_pr_urls"]
    assert "pull/91" in metadata["source_pr_urls"]
