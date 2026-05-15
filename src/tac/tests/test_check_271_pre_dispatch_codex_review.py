# SPDX-License-Identifier: MIT
"""Tests for Catalog #271 - PRE-DISPATCH-CODEX-REVIEW-AUTOMATION.

Coverage matrix:
- Cache hit/miss/TTL behavior
- Concurrent claim safety (4-process spawn pool stress)
- Verdict parsing for all 4 outcomes
- Cost gate (skip codex when estimated_cost <= $1)
- Paired-env bypass discipline (Catalog #199 sister rule)
- Codex companion missing -> needs-attention verdict
- STRICT preflight gate live-repo regression guard
- STRICT preflight gate positive/negative/waiver cases
- Catalog #199 sister bypass rejection (bare intent without rationale)
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
TOOLS_DIR = REPO_ROOT / "tools"
HELPER_PATH = TOOLS_DIR / "run_codex_review_for_dispatch.py"

# Make tools/ importable since the helper lives there as a module-style script.
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import operator_authorize  # noqa: E402
import run_codex_review_for_dispatch as helper  # noqa: E402

# Make src/ importable for tac.preflight
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from tac.preflight import (  # noqa: E402
    PreflightError,
    check_dispatch_runs_codex_adversarial_review_for_paid_dispatch,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def tmp_cache(tmp_path: Path) -> Path:
    """Provide an isolated cache JSONL path."""
    return tmp_path / "codex_pre_dispatch_review_cache.jsonl"


@pytest.fixture()
def tmp_repo(tmp_path: Path) -> Path:
    """Build a minimal fake repo layout with a trainer + recipe."""
    (tmp_path / ".omx" / "operator_authorize_recipes").mkdir(parents=True)
    (tmp_path / "experiments").mkdir()
    trainer = tmp_path / "experiments" / "train_substrate_xx.py"
    trainer.write_text("# fake trainer\n")
    recipe = (
        tmp_path / ".omx" / "operator_authorize_recipes" /
        "substrate_xx_modal_t4_dispatch.yaml"
    )
    recipe.write_text("name: substrate_xx_modal_t4_dispatch\n")
    return tmp_path


# ---------------------------------------------------------------------------
# Verdict parsing tests
# ---------------------------------------------------------------------------


class TestVerdictParsing:
    def test_explicit_approve(self) -> None:
        v, f = helper.parse_verdict_from_codex_output("VERDICT: approve\nClean.")
        assert v == "approve"

    def test_explicit_advisory(self) -> None:
        v, f = helper.parse_verdict_from_codex_output("Verdict: advisory\nLow stuff.")
        assert v == "advisory"

    def test_explicit_needs_attention(self) -> None:
        v, f = helper.parse_verdict_from_codex_output(
            "Verdict: needs-attention\nHIGH: bug X."
        )
        assert v == "needs-attention"

    def test_explicit_no_ship(self) -> None:
        v, f = helper.parse_verdict_from_codex_output("VERDICT: no-ship")
        assert v == "no-ship"

    def test_explicit_underscore_form(self) -> None:
        v, f = helper.parse_verdict_from_codex_output(
            "VERDICT: needs_attention\nHIGH: x"
        )
        assert v == "needs-attention"

    def test_implicit_critical_to_no_ship(self) -> None:
        v, f = helper.parse_verdict_from_codex_output(
            "Finding 1 - CRITICAL: archive grammar drift"
        )
        assert v == "no-ship"
        assert any("CRITICAL" in fnd for fnd in f)

    def test_implicit_blocker_to_no_ship(self) -> None:
        v, f = helper.parse_verdict_from_codex_output("BLOCKER: runtime crash")
        assert v == "no-ship"

    def test_implicit_high_to_needs_attention(self) -> None:
        v, f = helper.parse_verdict_from_codex_output("HIGH: scorer drift")
        assert v == "needs-attention"

    def test_implicit_low_to_advisory(self) -> None:
        v, f = helper.parse_verdict_from_codex_output("LOW: doc nit")
        assert v == "advisory"

    def test_implicit_medium_to_advisory(self) -> None:
        v, f = helper.parse_verdict_from_codex_output("MEDIUM: minor refactor")
        assert v == "advisory"

    def test_implicit_clean_to_approve(self) -> None:
        v, f = helper.parse_verdict_from_codex_output(
            "Code looks fine. No findings."
        )
        assert v == "approve"
        assert f == []

    def test_empty_output_to_approve(self) -> None:
        v, f = helper.parse_verdict_from_codex_output("")
        assert v == "approve"

    def test_findings_truncated_at_50(self) -> None:
        text = "\n".join(f"HIGH: bug {i}" for i in range(100))
        v, f = helper.parse_verdict_from_codex_output(text)
        assert v == "needs-attention"
        assert len(f) == 50

    def test_each_finding_truncated_at_200_chars(self) -> None:
        long = "x" * 500
        v, f = helper.parse_verdict_from_codex_output(f"CRITICAL: {long}")
        assert v == "no-ship"
        assert all(len(x) <= 200 for x in f)

    def test_highlight_word_does_not_trigger_high(self) -> None:
        # 'HIGHLIGHT' should not be misclassified as HIGH severity.
        v, f = helper.parse_verdict_from_codex_output(
            "HIGHLIGHT: emphasis on X"
        )
        # No severity tokens at all -> approve
        assert v == "approve"

    def test_explicit_verdict_overrides_implicit_severity(self) -> None:
        # If there are HIGH tokens but the explicit verdict says approve,
        # explicit wins.
        v, f = helper.parse_verdict_from_codex_output(
            "VERDICT: approve\nNote: HIGH frequency observed in plot"
        )
        assert v == "approve"


# ---------------------------------------------------------------------------
# Cost gate tests
# ---------------------------------------------------------------------------


class TestCostGate:
    def test_below_threshold_skips_codex(self, tmp_repo: Path, tmp_cache: Path) -> None:
        recipe = tmp_repo / ".omx" / "operator_authorize_recipes" / "substrate_xx_modal_t4_dispatch.yaml"
        trainer = tmp_repo / "experiments" / "train_substrate_xx.py"
        result = helper.run_codex_review_for_dispatch(
            trainer_path=trainer,
            recipe_path=recipe,
            repo_root=tmp_repo,
            estimated_cost_usd=0.50,  # below $1.00 default threshold
            cache_path=tmp_cache,
        )
        assert result.verdict == "advisory"
        assert result.cache_key == "cost-gated"
        assert result.cache_hit is False
        assert any("cost-gate" in f for f in result.findings)

    def test_at_threshold_skips_codex(self, tmp_repo: Path, tmp_cache: Path) -> None:
        recipe = tmp_repo / ".omx" / "operator_authorize_recipes" / "substrate_xx_modal_t4_dispatch.yaml"
        trainer = tmp_repo / "experiments" / "train_substrate_xx.py"
        result = helper.run_codex_review_for_dispatch(
            trainer_path=trainer,
            recipe_path=recipe,
            repo_root=tmp_repo,
            estimated_cost_usd=1.00,  # exactly at threshold; gate uses <=
            cache_path=tmp_cache,
        )
        assert result.cache_key == "cost-gated"

    def test_custom_threshold(self, tmp_repo: Path, tmp_cache: Path) -> None:
        recipe = tmp_repo / ".omx" / "operator_authorize_recipes" / "substrate_xx_modal_t4_dispatch.yaml"
        trainer = tmp_repo / "experiments" / "train_substrate_xx.py"
        # Cost 5.00, threshold 10.00 => gated
        result = helper.run_codex_review_for_dispatch(
            trainer_path=trainer,
            recipe_path=recipe,
            repo_root=tmp_repo,
            estimated_cost_usd=5.00,
            cost_gate_threshold_usd=10.00,
            cache_path=tmp_cache,
        )
        assert result.cache_key == "cost-gated"


# ---------------------------------------------------------------------------
# Cache tests
# ---------------------------------------------------------------------------


class TestCache:
    def test_cache_miss_returns_none(self, tmp_cache: Path) -> None:
        r = helper.lookup_cached_review("nonexistent", cache_path=tmp_cache)
        assert r is None

    def test_cache_append_and_lookup_roundtrip(self, tmp_cache: Path) -> None:
        result = helper.CodexReviewResult(
            verdict="approve",
            findings=[],
            cache_hit=False,
            cache_age_sec=0,
            cache_key="abc123",
            raw_output_excerpt="ok",
            invoked_at_utc=helper._utc_iso(),
            elapsed_sec=1.5,
        )
        helper.append_cached_review(result, cache_path=tmp_cache)
        cached = helper.lookup_cached_review("abc123", cache_path=tmp_cache)
        assert cached is not None
        assert cached.cache_hit is True
        assert cached.verdict == "approve"
        assert cached.cache_key == "abc123"

    def test_cache_ttl_expiry(self, tmp_cache: Path) -> None:
        result = helper.CodexReviewResult(
            verdict="approve",
            findings=[],
            cache_hit=False,
            cache_age_sec=0,
            cache_key="ttl_test",
            raw_output_excerpt="ok",
            invoked_at_utc=helper._utc_iso(),
            elapsed_sec=1.5,
        )
        helper.append_cached_review(result, cache_path=tmp_cache)
        # Lookup with TTL=0 -> always expired
        cached = helper.lookup_cached_review(
            "ttl_test", ttl_seconds=0, cache_path=tmp_cache
        )
        assert cached is None

    def test_cache_returns_latest_row_for_key(self, tmp_cache: Path) -> None:
        # First write: approve
        r1 = helper.CodexReviewResult(
            verdict="approve",
            findings=[],
            cache_hit=False,
            cache_age_sec=0,
            cache_key="dup",
            raw_output_excerpt="v1",
            invoked_at_utc=helper._utc_iso(),
            elapsed_sec=1.0,
        )
        helper.append_cached_review(r1, cache_path=tmp_cache)
        time.sleep(1.1)  # ensure newer epoch
        # Second write: needs-attention
        r2 = helper.CodexReviewResult(
            verdict="needs-attention",
            findings=["x"],
            cache_hit=False,
            cache_age_sec=0,
            cache_key="dup",
            raw_output_excerpt="v2",
            invoked_at_utc=helper._utc_iso(),
            elapsed_sec=2.0,
        )
        helper.append_cached_review(r2, cache_path=tmp_cache)
        cached = helper.lookup_cached_review("dup", cache_path=tmp_cache)
        assert cached is not None
        assert cached.verdict == "needs-attention"

    def test_cache_skips_malformed_lines(self, tmp_cache: Path) -> None:
        tmp_cache.parent.mkdir(parents=True, exist_ok=True)
        tmp_cache.write_text(
            "this is not json\n"
            + json.dumps({
                "schema_version": helper.CACHE_SCHEMA_VERSION,
                "cache_key": "good",
                "verdict": "approve",
                "findings": [],
                "cache_hit": False,
                "cache_age_sec": 0,
                "raw_output_excerpt": "",
                "invoked_at_utc": helper._utc_iso(),
                "invoked_at_epoch": helper._utc_epoch(),
                "elapsed_sec": 0.0,
            }) + "\n"
        )
        cached = helper.lookup_cached_review("good", cache_path=tmp_cache)
        assert cached is not None
        assert cached.verdict == "approve"


def _worker_append(args: tuple[str, str]) -> None:
    """Worker for concurrent-append stress test."""
    cache_path_str, key = args
    cp = Path(cache_path_str)
    sys.path.insert(0, str(REPO_ROOT / "tools"))
    import run_codex_review_for_dispatch as h
    r = h.CodexReviewResult(
        verdict="approve",
        findings=[],
        cache_hit=False,
        cache_age_sec=0,
        cache_key=key,
        raw_output_excerpt="",
        invoked_at_utc=h._utc_iso(),
        elapsed_sec=0.0,
    )
    for _ in range(5):
        h.append_cached_review(r, cache_path=cp)


class TestConcurrentClaimSafety:
    def test_4proc_spawn_pool_no_corruption(self, tmp_cache: Path) -> None:
        ctx = mp.get_context("spawn")
        with ctx.Pool(4) as pool:
            args = [(str(tmp_cache), f"k{i}") for i in range(4)]
            pool.map(_worker_append, args)
        # Expect exactly 4 * 5 = 20 rows; every line valid JSON
        text = tmp_cache.read_text(encoding="utf-8")
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
        assert len(rows) == 20
        # Each unique cache key should appear 5 times
        from collections import Counter
        c = Counter(r["cache_key"] for r in rows)
        for k in ("k0", "k1", "k2", "k3"):
            assert c[k] == 5


# ---------------------------------------------------------------------------
# Paired-env bypass discipline tests (Catalog #199 sister rule)
# ---------------------------------------------------------------------------


class TestPairedEnvBypass:
    def test_bypass_inactive_when_no_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(helper.BYPASS_VERDICT_ENV, raising=False)
        monkeypatch.delenv(helper.BYPASS_RATIONALE_ENV, raising=False)
        assert helper.check_paired_bypass_env_or_exit() is False

    def test_bypass_active_when_paired(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(helper.BYPASS_VERDICT_ENV, "1")
        monkeypatch.setenv(helper.BYPASS_RATIONALE_ENV, "operator-reviewed Z3-G1")
        assert helper.check_paired_bypass_env_or_exit() is True

    def test_bare_intent_without_rationale_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(helper.BYPASS_VERDICT_ENV, "1")
        monkeypatch.delenv(helper.BYPASS_RATIONALE_ENV, raising=False)
        with pytest.raises(SystemExit) as exc:
            helper.check_paired_bypass_env_or_exit()
        assert exc.value.code == 13

    def test_intent_with_blank_rationale_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(helper.BYPASS_VERDICT_ENV, "1")
        monkeypatch.setenv(helper.BYPASS_RATIONALE_ENV, "   ")
        with pytest.raises(SystemExit) as exc:
            helper.check_paired_bypass_env_or_exit()
        assert exc.value.code == 13


# ---------------------------------------------------------------------------
# Codex companion availability
# ---------------------------------------------------------------------------


class TestCodexCompanionAvailability:
    def test_returns_false_when_script_missing(self, tmp_path: Path) -> None:
        # Pass a non-existent script path
        assert helper.codex_companion_available(
            script_path=str(tmp_path / "nonexistent.mjs")
        ) is False

    def test_returns_true_when_script_present(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        # Synthesize a "script" file
        fake = tmp_path / "fake.mjs"
        fake.write_text("// fake")
        # Ensure node is actually in PATH for this test machine
        if not __import__("shutil").which("node"):
            pytest.skip("node not in PATH")
        assert helper.codex_companion_available(script_path=str(fake)) is True


class TestOperatorAuthorizeCodexReview:
    def test_missing_codex_review_helper_is_fatal(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(operator_authorize, "REPO_ROOT", tmp_path)
        monkeypatch.delenv("OPERATOR_AUTHORIZE_SKIP_CODEX_PRE_DISPATCH_REVIEW", raising=False)
        monkeypatch.delenv("OPERATOR_AUTHORIZE_CODEX_PRE_DISPATCH_BYPASS_REASON", raising=False)

        with pytest.raises(SystemExit, match="codex pre-dispatch helper not found"):
            operator_authorize._run_codex_pre_dispatch_review(
                "experiments/train_x.py",
                ".omx/operator_authorize_recipes/x.yaml",
                5.0,
            )


# ---------------------------------------------------------------------------
# run_codex_review_for_dispatch end-to-end (with codex stubbed via missing)
# ---------------------------------------------------------------------------


class TestRunCodexReviewForDispatchEndToEnd:
    def test_missing_codex_helper_returns_needs_attention(
        self, tmp_repo: Path, tmp_cache: Path, tmp_path: Path
    ) -> None:
        recipe = tmp_repo / ".omx" / "operator_authorize_recipes" / "substrate_xx_modal_t4_dispatch.yaml"
        trainer = tmp_repo / "experiments" / "train_substrate_xx.py"
        # Force script_path to a missing path
        result = helper.run_codex_review_for_dispatch(
            trainer_path=trainer,
            recipe_path=recipe,
            repo_root=tmp_repo,
            estimated_cost_usd=10.00,
            cache_path=tmp_cache,
            script_path=str(tmp_path / "missing.mjs"),
        )
        assert result.verdict == "needs-attention"
        assert any("codex-companion-missing" in f for f in result.findings)

    def test_cache_hit_short_circuits(self, tmp_repo: Path, tmp_cache: Path) -> None:
        recipe = tmp_repo / ".omx" / "operator_authorize_recipes" / "substrate_xx_modal_t4_dispatch.yaml"
        trainer = tmp_repo / "experiments" / "train_substrate_xx.py"
        # Pre-seed cache with an approve result
        git_sha = helper._git_head_sha(tmp_repo)
        recipe_sha = helper._file_sha256(recipe)
        trainer_sha = helper._file_sha256(trainer)
        cache_key = helper._compute_cache_key(git_sha, recipe_sha, trainer_sha)
        seed = helper.CodexReviewResult(
            verdict="approve",
            findings=[],
            cache_hit=False,
            cache_age_sec=0,
            cache_key=cache_key,
            raw_output_excerpt="cached",
            invoked_at_utc=helper._utc_iso(),
            elapsed_sec=2.0,
        )
        helper.append_cached_review(seed, cache_path=tmp_cache)
        result = helper.run_codex_review_for_dispatch(
            trainer_path=trainer,
            recipe_path=recipe,
            repo_root=tmp_repo,
            estimated_cost_usd=10.00,
            cache_path=tmp_cache,
        )
        assert result.cache_hit is True
        assert result.verdict == "approve"

    def test_skip_cache_bypasses_lookup(self, tmp_repo: Path, tmp_cache: Path, tmp_path: Path) -> None:
        recipe = tmp_repo / ".omx" / "operator_authorize_recipes" / "substrate_xx_modal_t4_dispatch.yaml"
        trainer = tmp_repo / "experiments" / "train_substrate_xx.py"
        # Pre-seed cache + force missing codex script -> still calls invoke path
        git_sha = helper._git_head_sha(tmp_repo)
        recipe_sha = helper._file_sha256(recipe)
        trainer_sha = helper._file_sha256(trainer)
        cache_key = helper._compute_cache_key(git_sha, recipe_sha, trainer_sha)
        seed = helper.CodexReviewResult(
            verdict="approve",
            findings=[],
            cache_hit=False,
            cache_age_sec=0,
            cache_key=cache_key,
            raw_output_excerpt="cached",
            invoked_at_utc=helper._utc_iso(),
            elapsed_sec=2.0,
        )
        helper.append_cached_review(seed, cache_path=tmp_cache)
        result = helper.run_codex_review_for_dispatch(
            trainer_path=trainer,
            recipe_path=recipe,
            repo_root=tmp_repo,
            estimated_cost_usd=10.00,
            cache_path=tmp_cache,
            skip_cache=True,
            script_path=str(tmp_path / "missing.mjs"),
        )
        # cache hit is False because we skipped; codex missing -> needs-attention
        assert result.cache_hit is False
        assert result.verdict == "needs-attention"


# ---------------------------------------------------------------------------
# STRICT preflight gate Catalog #271
# ---------------------------------------------------------------------------


class TestPreflightGate:
    def test_live_repo_warn_only_regression_guard(self) -> None:
        """Bound on live violations - prevents runaway growth."""
        violations = check_dispatch_runs_codex_adversarial_review_for_paid_dispatch(
            strict=False, verbose=False,
        )
        assert isinstance(violations, list)
        # Initial wire-in is warn-only; bound generously above current count
        # to allow incremental migration without breaking the gate.
        assert len(violations) <= 200, (
            f"check_271 live count grew unexpectedly: {len(violations)}"
        )

    def test_dispatch_token_without_routing_flagged(self, tmp_path: Path) -> None:
        """A wrapper with a dispatch token but no canonical routing is flagged."""
        (tmp_path / "tools").mkdir()
        bad = tmp_path / "tools" / "wrapper.py"
        bad.write_text("import subprocess\nsubprocess.run(['modal_train_lane', 'foo'])  # noqa\n")
        # Need to also self-exempt the canonical paths to avoid cross-contamination
        (tmp_path / "tools" / "operator_authorize.py").write_text("# stub\n")
        (tmp_path / "tools" / "run_codex_review_for_dispatch.py").write_text("# stub\n")
        violations = check_dispatch_runs_codex_adversarial_review_for_paid_dispatch(
            repo_root=tmp_path, strict=False, verbose=False,
        )
        assert any("wrapper.py" in v for v in violations)

    def test_canonical_routing_via_operator_authorize_accepted(self, tmp_path: Path) -> None:
        (tmp_path / "tools").mkdir()
        good = tmp_path / "tools" / "wrapper.py"
        good.write_text(
            "# Routes via tools/operator_authorize.py\n"
            "import subprocess\n"
            "subprocess.run(['modal_train_lane', 'foo'])  # noqa\n"
        )
        violations = check_dispatch_runs_codex_adversarial_review_for_paid_dispatch(
            repo_root=tmp_path, strict=False, verbose=False,
        )
        assert not any("wrapper.py" in v for v in violations)

    def test_direct_codex_helper_invocation_accepted(self, tmp_path: Path) -> None:
        (tmp_path / "tools").mkdir()
        good = tmp_path / "tools" / "wrapper.py"
        good.write_text(
            "# Calls tools/run_codex_review_for_dispatch.py directly first\n"
            "import subprocess\n"
            "subprocess.run(['modal_train_lane', 'foo'])  # noqa\n"
        )
        violations = check_dispatch_runs_codex_adversarial_review_for_paid_dispatch(
            repo_root=tmp_path, strict=False, verbose=False,
        )
        assert not any("wrapper.py" in v for v in violations)

    def test_same_line_waiver_accepted(self, tmp_path: Path) -> None:
        (tmp_path / "tools").mkdir()
        bad = tmp_path / "tools" / "wrapper.py"
        bad.write_text(
            "import subprocess\n"
            "subprocess.run(['modal_train_lane', 'foo'])  # noqa  # CODEX_PRE_DISPATCH_REVIEW_BYPASS_OK:debug-only-tool\n"
        )
        violations = check_dispatch_runs_codex_adversarial_review_for_paid_dispatch(
            repo_root=tmp_path, strict=False, verbose=False,
        )
        assert not any("wrapper.py" in v for v in violations)

    def test_placeholder_waiver_rejected(self, tmp_path: Path) -> None:
        (tmp_path / "tools").mkdir()
        bad = tmp_path / "tools" / "wrapper.py"
        bad.write_text(
            "import subprocess\n"
            "subprocess.run(['modal_train_lane', 'foo'])  # noqa  # CODEX_PRE_DISPATCH_REVIEW_BYPASS_OK:<reason>\n"
        )
        violations = check_dispatch_runs_codex_adversarial_review_for_paid_dispatch(
            repo_root=tmp_path, strict=False, verbose=False,
        )
        assert any("wrapper.py" in v for v in violations)

    def test_strict_mode_raises(self, tmp_path: Path) -> None:
        (tmp_path / "tools").mkdir()
        bad = tmp_path / "tools" / "wrapper.py"
        bad.write_text(
            "import subprocess\nsubprocess.run(['modal_train_lane', 'foo'])  # noqa\n"
        )
        with pytest.raises(PreflightError) as exc:
            check_dispatch_runs_codex_adversarial_review_for_paid_dispatch(
                repo_root=tmp_path, strict=True, verbose=False,
            )
        assert "Catalog #271" in str(exc.value) or "271" in str(exc.value)

    def test_strict_mode_silent_on_clean(self, tmp_path: Path) -> None:
        # No tools/ dir at all -> no violations possible
        violations = check_dispatch_runs_codex_adversarial_review_for_paid_dispatch(
            repo_root=tmp_path, strict=True, verbose=False,
        )
        assert violations == []

    def test_test_files_excluded(self, tmp_path: Path) -> None:
        (tmp_path / "tools").mkdir()
        # File in tests/ subdir - should be skipped
        (tmp_path / "tools" / "tests").mkdir()
        skipped = tmp_path / "tools" / "tests" / "test_dispatch.py"
        skipped.write_text(
            "import subprocess\nsubprocess.run(['modal_train_lane', 'foo'])  # noqa\n"
        )
        violations = check_dispatch_runs_codex_adversarial_review_for_paid_dispatch(
            repo_root=tmp_path, strict=False, verbose=False,
        )
        assert not any("test_dispatch.py" in v for v in violations)

    def test_test_filename_prefix_excluded(self, tmp_path: Path) -> None:
        (tmp_path / "tools").mkdir()
        skipped = tmp_path / "tools" / "test_thing.py"
        skipped.write_text(
            "import subprocess\nsubprocess.run(['modal_train_lane', 'foo'])  # noqa\n"
        )
        violations = check_dispatch_runs_codex_adversarial_review_for_paid_dispatch(
            repo_root=tmp_path, strict=False, verbose=False,
        )
        assert not any("test_thing.py" in v for v in violations)

    def test_self_exempt_canonical_helper(self, tmp_path: Path) -> None:
        # The canonical helper itself must NOT be flagged even if it
        # contains 'modal run' literal in docs.
        (tmp_path / "tools").mkdir()
        canonical = tmp_path / "tools" / "run_codex_review_for_dispatch.py"
        canonical.write_text(
            "# Mentions modal run in a docstring example\n"
            "import subprocess\nsubprocess.run(['modal_train_lane', 'foo'])  # noqa\n"
        )
        violations = check_dispatch_runs_codex_adversarial_review_for_paid_dispatch(
            repo_root=tmp_path, strict=False, verbose=False,
        )
        assert not any("run_codex_review_for_dispatch" in v for v in violations)


# ---------------------------------------------------------------------------
# CLI subprocess
# ---------------------------------------------------------------------------


class TestCli:
    def test_cli_cost_gated_returns_0(self, tmp_repo: Path, tmp_path: Path) -> None:
        recipe = tmp_repo / ".omx" / "operator_authorize_recipes" / "substrate_xx_modal_t4_dispatch.yaml"
        trainer = tmp_repo / "experiments" / "train_substrate_xx.py"
        out = tmp_path / "result.json"
        cmd = [
            sys.executable,
            str(HELPER_PATH),
            "--trainer", str(trainer),
            "--recipe", str(recipe),
            "--estimated-cost-usd", "0.50",
            "--json-out", str(out),
            "--cache-path", str(tmp_path / "cache.jsonl"),
        ]
        env = {**os.environ}
        env.pop(helper.BYPASS_VERDICT_ENV, None)
        env.pop(helper.BYPASS_RATIONALE_ENV, None)
        rc = subprocess.run(cmd, capture_output=True, text=True, env=env)
        assert rc.returncode == 0
        payload = json.loads(out.read_text())
        assert payload["verdict"] == "advisory"
        assert payload["cache_key"] == "cost-gated"

    def test_cli_bare_bypass_intent_returns_13(
        self, tmp_repo: Path, tmp_path: Path
    ) -> None:
        recipe = tmp_repo / ".omx" / "operator_authorize_recipes" / "substrate_xx_modal_t4_dispatch.yaml"
        trainer = tmp_repo / "experiments" / "train_substrate_xx.py"
        cmd = [
            sys.executable,
            str(HELPER_PATH),
            "--trainer", str(trainer),
            "--recipe", str(recipe),
            "--estimated-cost-usd", "5.00",
            "--cache-path", str(tmp_path / "cache.jsonl"),
        ]
        env = {**os.environ}
        env[helper.BYPASS_VERDICT_ENV] = "1"
        env.pop(helper.BYPASS_RATIONALE_ENV, None)
        rc = subprocess.run(cmd, capture_output=True, text=True, env=env)
        assert rc.returncode == 13


# ---------------------------------------------------------------------------
# Z3-G1 regression guard: simulate a "needs-attention" cached verdict and
# confirm the gate would have refused dispatch.
# ---------------------------------------------------------------------------


class TestZ3G1RegressionGuard:
    def test_cached_needs_attention_blocks_dispatch_via_cli(
        self, tmp_repo: Path, tmp_path: Path
    ) -> None:
        recipe = (
            tmp_repo / ".omx" / "operator_authorize_recipes" /
            "substrate_xx_modal_t4_dispatch.yaml"
        )
        trainer = tmp_repo / "experiments" / "train_substrate_xx.py"
        cache_path = tmp_path / "cache.jsonl"
        # Pre-seed with a needs-attention verdict (mimics what codex would
        # have produced for Z3-G1's F1+F2 findings).
        git_sha = helper._git_head_sha(tmp_repo)
        recipe_sha = helper._file_sha256(recipe)
        trainer_sha = helper._file_sha256(trainer)
        cache_key = helper._compute_cache_key(git_sha, recipe_sha, trainer_sha)
        seed = helper.CodexReviewResult(
            verdict="needs-attention",
            findings=[
                "HIGH: F1 - empty G1 sidecar slots (Z3-G1 regression)",
                "HIGH: F2 - silent except-Exception with synthetic prior fallback",
            ],
            cache_hit=False,
            cache_age_sec=0,
            cache_key=cache_key,
            raw_output_excerpt="seeded",
            invoked_at_utc=helper._utc_iso(),
            elapsed_sec=30.0,
        )
        helper.append_cached_review(seed, cache_path=cache_path)
        cmd = [
            sys.executable,
            str(HELPER_PATH),
            "--trainer", str(trainer),
            "--recipe", str(recipe),
            "--estimated-cost-usd", "5.00",
            "--cache-path", str(cache_path),
            "--repo-root", str(tmp_repo),
        ]
        env = {**os.environ}
        env.pop(helper.BYPASS_VERDICT_ENV, None)
        env.pop(helper.BYPASS_RATIONALE_ENV, None)
        rc = subprocess.run(cmd, capture_output=True, text=True, env=env)
        assert rc.returncode == 1, (
            "Z3-G1 regression: needs-attention verdict MUST block dispatch"
        )

    def test_cached_needs_attention_with_paired_bypass_allows_dispatch(
        self, tmp_repo: Path, tmp_path: Path
    ) -> None:
        recipe = (
            tmp_repo / ".omx" / "operator_authorize_recipes" /
            "substrate_xx_modal_t4_dispatch.yaml"
        )
        trainer = tmp_repo / "experiments" / "train_substrate_xx.py"
        cache_path = tmp_path / "cache.jsonl"
        git_sha = helper._git_head_sha(tmp_repo)
        recipe_sha = helper._file_sha256(recipe)
        trainer_sha = helper._file_sha256(trainer)
        cache_key = helper._compute_cache_key(git_sha, recipe_sha, trainer_sha)
        seed = helper.CodexReviewResult(
            verdict="needs-attention",
            findings=["HIGH: x"],
            cache_hit=False,
            cache_age_sec=0,
            cache_key=cache_key,
            raw_output_excerpt="seeded",
            invoked_at_utc=helper._utc_iso(),
            elapsed_sec=30.0,
        )
        helper.append_cached_review(seed, cache_path=cache_path)
        cmd = [
            sys.executable,
            str(HELPER_PATH),
            "--trainer", str(trainer),
            "--recipe", str(recipe),
            "--estimated-cost-usd", "5.00",
            "--cache-path", str(cache_path),
            "--repo-root", str(tmp_repo),
        ]
        env = {**os.environ}
        env[helper.BYPASS_VERDICT_ENV] = "1"
        env[helper.BYPASS_RATIONALE_ENV] = "operator-reviewed; council-approved override"
        rc = subprocess.run(cmd, capture_output=True, text=True, env=env)
        assert rc.returncode == 0, (
            "Paired-env bypass should allow dispatch even on needs-attention"
        )
