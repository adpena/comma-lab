from __future__ import annotations

from pathlib import Path

from comma_lab.research_state import classify_relpath, render_markdown

REPO = Path(__file__).resolve().parents[1]


def test_omx_research_markdown_is_trackable_git_state() -> None:
    category, disposition, target, reason = classify_relpath(
        ".omx/research/findings.md",
        4096,
        "untracked",
    )

    assert category == "research_ledger"
    assert disposition == "track_in_git"
    assert target == ".omx/research/findings.md"
    assert "durable lab knowledge" in reason


def test_auto_memory_snapshot_is_canonicalized_not_committed_raw() -> None:
    category, disposition, target, reason = classify_relpath(
        ".omx/auto_memory_snapshot_20260504T230223Z/MEMORY.md",
        64_000,
        "ignored",
    )

    assert category == "memory_snapshot"
    assert disposition == "canonicalize_to_research_ledger"
    assert target == ".omx/research/ or docs/paper/ara/"
    assert "operator backups" in reason


def test_provider_state_is_summarized_not_tracked_raw() -> None:
    category, disposition, target, reason = classify_relpath(
        ".omx/state/lightning_batch_jobs.json",
        200_000,
        "ignored",
    )

    assert category == "provider_or_runtime_state"
    assert disposition == "summarize_to_research_ledger"
    assert target == ".omx/research/<dated summary>.md"
    assert "leak" in reason


def test_large_research_artifact_externalizes_with_manifest() -> None:
    category, disposition, target, reason = classify_relpath(
        ".omx/research/artifacts/run/movie.gif",
        12_000_000,
        "untracked",
    )

    assert category == "research_artifact"
    assert disposition == "externalize_with_manifest"
    assert target == "external artifact store plus committed manifest"
    assert "rebuildable custody outputs" in reason


def test_generated_research_artifact_externalizes_even_when_small_text() -> None:
    category, disposition, target, reason = classify_relpath(
        ".omx/research/artifacts/recovery/audit.json",
        12_000,
        "untracked",
    )

    assert category == "research_artifact"
    assert disposition == "externalize_with_manifest"
    assert target == "external artifact store plus committed manifest"
    assert "rebuildable custody outputs" in reason


def test_generated_public_site_bundle_is_not_tracked_as_source() -> None:
    category, disposition, target, reason = classify_relpath(
        "reports/graphs/public_site/final_writeup_draft.md",
        12_000,
        "ignored",
    )

    assert category == "hosted_supplement_build"
    assert disposition == "externalize_with_manifest"
    assert target == "docs/site source plus external hosted supplement manifest"
    assert "rebuilt from source" in reason


def test_wrangler_cache_stays_private_local() -> None:
    category, disposition, target, reason = classify_relpath(
        "reports/graphs/public_site/.wrangler/cache/wrangler-account.json",
        109,
        "ignored",
    )

    assert category == "hosted_supplement_cache"
    assert disposition == "keep_private_local"
    assert target == "local only"
    assert "account-local" in reason


def test_render_markdown_surfaces_tracking_sections() -> None:
    from comma_lab.research_state import ResearchStateRecord

    md = render_markdown(
        [
            ResearchStateRecord(
                relpath=".omx/research/findings.md",
                bytes=1,
                git_status="untracked",
                category="research_ledger",
                disposition="track_in_git",
                target=".omx/research/findings.md",
                reason="Small research ledgers and structured summaries are durable lab knowledge.",
            )
        ]
    )

    assert "Should Be Tracked In Git" in md
    assert ".omx/research/findings.md" in md


def test_repo_policy_keeps_research_state_boundary_in_comma_lab() -> None:
    gitignore = (REPO / ".gitignore").read_text()
    agents = (REPO / "AGENTS.md").read_text()
    claude = (REPO / "CLAUDE.md").read_text()

    assert "tools/audit_research_state_tracking.py" in gitignore
    assert ".omx/auto_memory_snapshot_*/" in gitignore
    assert ".omx/research/artifacts/" in gitignore
    assert ".omx/research/materializer_exact_eval_dispatch_plan_*.json" in gitignore
    assert ".omx/research/materializer_exact_eval_consumer_report_*.json" in gitignore
    assert ".omx/research/*_artifact_retention_*.json" in gitignore
    assert ".omx/research/*_artifact_retention_*.json.journal.jsonl" in gitignore
    assert "**/.*.tmp-*" in gitignore
    assert "experiments/results/lightning_batch/**/source_manifest.json" in gitignore
    assert "!experiments/results/lightning_batch/**/custody_anchor.json" in gitignore
    assert "Keep `tac` clean" in agents
    assert "Contest-specific public-submission reverse engineering belongs in" in agents
    assert "`tac` stays clean; comma-lab owns research state" in claude
    assert "Use `reverse_engineering/` for clean public-submission deconstruction" in claude
    assert (REPO / "src/comma_lab/research_state.py").is_file()
    assert (REPO / "docs/runbooks/research_state_tracking_policy.md").is_file()
    assert (REPO / "reverse_engineering/README.md").is_file()
