"""Tests for catalog #127 + #128 (codex round-2 HIGH 2 + MEDIUM fix gates).

Catalog #127 — `check_authoritative_tag_requires_custody_metadata`
Catalog #128 — `check_continual_learning_writes_use_lock`

Both gates protect the unified-custody and locked-write contracts landed
in `tac.continual_learning` (codex round-2 HIGH 2 + MEDIUM, 2026-05-09).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from tac.source_index import source_index_context
from tac.preflight import (
    PreflightError,
    check_authoritative_tag_requires_custody_metadata,
    check_continual_learning_writes_use_lock,
    check_custody_gate_accept_tokens_concrete_only,
    check_no_bare_writes_to_shared_state,
    check_no_tag_only_custody_validation,
    check_remote_dispatch_runbooks_no_local_cuda_probe_default,
)


# ─────────────────────────────────────────────────────────────────────────
# Catalog #127 — authoritative tag must route through validate_custody.
# ─────────────────────────────────────────────────────────────────────────


def _mkrepo(tmp_path: Path, files: dict[str, str]) -> Path:
    """Create a temp repo skeleton (`src/tac/`, `tools/`, `experiments/`)."""
    for d in ("src/tac", "tools", "experiments"):
        (tmp_path / d).mkdir(parents=True, exist_ok=True)
    for rel, content in files.items():
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    return tmp_path


def test_check127_clean_repo_passes(tmp_path: Path) -> None:
    """No tag literals → no violations."""
    root = _mkrepo(tmp_path, {"src/tac/foo.py": "x = 1\n"})
    violations = check_authoritative_tag_requires_custody_metadata(
        repo_root=root, strict=False, verbose=False
    )
    assert violations == []


def test_check136_ignores_parentheses_inside_token_strings(tmp_path: Path) -> None:
    """A concrete token like posterior_update( must not make the scanner
    run past the accept-list close and self-match later blocklist text.
    """
    root = _mkrepo(
        tmp_path,
        {
            "src/tac/preflight_like.py": (
                "_CUSTODY_VALIDATOR_TOKENS = (\n"
                "    \"validate_custody\",\n"
                "    \"posterior_update(\",\n"
                ")\n"
                "_GENERIC_TOKEN_BLOCKLIST = (\n"
                "    \"blockers\",\n"
                "    \"errors\",\n"
                ")\n"
            ),
        },
    )

    violations = check_custody_gate_accept_tokens_concrete_only(
        repo_root=root,
        strict=False,
        verbose=False,
    )

    assert violations == []


def test_check136_rejects_bare_generic_accept_token(tmp_path: Path) -> None:
    root = _mkrepo(
        tmp_path,
        {
            "tools/bad_accept_tokens.py": (
                "_SCORE_VALIDATOR_TOKENS = (\n"
                "    \"validate_custody\",\n"
                "    \"blockers\",\n"
                ")\n"
            ),
        },
    )

    violations = check_custody_gate_accept_tokens_concrete_only(
        repo_root=root,
        strict=False,
        verbose=False,
    )

    assert len(violations) == 1
    assert "[Check 136]" in violations[0]
    assert "blockers" in violations[0]


def test_check137_allows_remote_side_workspace_bootstrap(tmp_path: Path) -> None:
    root = _mkrepo(
        tmp_path,
        {
            "scripts/remote_lane_gpu_payload.sh": (
                "#!/usr/bin/env bash\n"
                "set -euo pipefail\n"
                'WORKSPACE="${WORKSPACE:-/workspace/pact}"\n'
                'cd "$WORKSPACE"\n'
                'GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader)\n'
                'bash "$WORKSPACE/scripts/probe_nvdec.sh"\n'
            ),
        },
    )

    violations = check_remote_dispatch_runbooks_no_local_cuda_probe_default(
        repo_root=root,
        strict=False,
        verbose=False,
    )

    assert violations == []


def test_check137_allows_remote_cuda_host_header(tmp_path: Path) -> None:
    root = _mkrepo(
        tmp_path,
        {
            "scripts/remote_lane_cuda_payload.sh": (
                "#!/usr/bin/env bash\n"
                "# This script is intended to run on an already-claimed remote CUDA host.\n"
                "set -euo pipefail\n"
                'WORKSPACE="${WORKSPACE:-$PWD}"\n'
                'cd "$WORKSPACE"\n'
                "python3 - <<'PY'\n"
                "import torch\n"
                "assert torch.cuda.is_available()\n"
                "PY\n"
                "nvidia-smi\n"
            ),
        },
    )

    violations = check_remote_dispatch_runbooks_no_local_cuda_probe_default(
        repo_root=root,
        strict=False,
        verbose=False,
    )

    assert violations == []


def test_check137_allows_nonblocking_gpu_provenance(tmp_path: Path) -> None:
    root = _mkrepo(
        tmp_path,
        {
            "scripts/remote_lane_provenance_only.sh": (
                "#!/usr/bin/env bash\n"
                "set -euo pipefail\n"
                "nvidia-smi || true\n"
                "GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo unknown)\n"
            ),
        },
    )

    violations = check_remote_dispatch_runbooks_no_local_cuda_probe_default(
        repo_root=root,
        strict=False,
        verbose=False,
    )

    assert violations == []


def test_check137_rejects_local_dispatch_driver_probe(tmp_path: Path) -> None:
    root = _mkrepo(
        tmp_path,
        {
            "scripts/remote_lane_local_dispatcher.sh": (
                "#!/usr/bin/env bash\n"
                "set -euo pipefail\n"
                'REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"\n'
                'cd "$REPO_ROOT"\n'
                "nvidia-smi\n"
            ),
        },
    )

    violations = check_remote_dispatch_runbooks_no_local_cuda_probe_default(
        repo_root=root,
        strict=False,
        verbose=False,
    )

    assert len(violations) == 1
    assert "[Check 137]" in violations[0]


def test_check137_accepts_local_cuda_worker_guard(tmp_path: Path) -> None:
    root = _mkrepo(
        tmp_path,
        {
            "scripts/remote_lane_guarded_dispatcher.sh": (
                "#!/usr/bin/env bash\n"
                "set -euo pipefail\n"
                'LOCAL_CUDA_WORKER="${LOCAL_CUDA_WORKER:-0}"\n'
                'if [ "$LOCAL_CUDA_WORKER" = "1" ]; then\n'
                "  nvidia-smi\n"
                "fi\n"
            ),
        },
    )

    violations = check_remote_dispatch_runbooks_no_local_cuda_probe_default(
        repo_root=root,
        strict=False,
        verbose=False,
    )

    assert violations == []


def test_check127_bypass_pattern_caught(tmp_path: Path) -> None:
    """Tag literal compared against AUTHORITATIVE_TAGS without validator → violation."""
    root = _mkrepo(
        tmp_path,
        {
            "tools/bad_emitter.py": (
                "from tac.continual_learning import AUTHORITATIVE_TAGS\n"
                "tag = '[contest-CPU]'\n"
                "if tag in AUTHORITATIVE_TAGS:\n"
                "    pass\n"
            ),
        },
    )
    violations = check_authoritative_tag_requires_custody_metadata(
        repo_root=root, strict=False, verbose=False
    )
    assert len(violations) == 1
    assert "tools/bad_emitter.py" in violations[0]
    assert "[Check 127]" in violations[0]


def test_check127_direct_tag_literal_predicate_caught(tmp_path: Path) -> None:
    """Direct equality against an authoritative tag literal is also a bypass."""
    root = _mkrepo(
        tmp_path,
        {
            "tools/direct_literal.py": (
                "def is_good(tag):\n"
                "    return tag == '[contest-CPU]'\n"
            ),
        },
    )
    violations = check_authoritative_tag_requires_custody_metadata(
        repo_root=root, strict=False, verbose=False
    )
    assert len(violations) == 1
    assert "tools/direct_literal.py" in violations[0]


def test_check127_validator_elsewhere_in_file_does_not_whitelist_all(tmp_path: Path) -> None:
    """A validator mention in one function must not hide a bypass in another."""
    root = _mkrepo(
        tmp_path,
        {
            "tools/mixed.py": (
                "def safe(result):\n"
                "    return result.validate_custody()\n"
                "\n"
                "\n"
                "\n"
                "\n"
                "\n"
                "def unsafe(tag):\n"
                "    return tag == '[contest-CUDA]'\n"
            ),
        },
    )
    violations = check_authoritative_tag_requires_custody_metadata(
        repo_root=root, strict=False, verbose=False
    )
    assert len(violations) == 1
    assert "tools/mixed.py" in violations[0]


def test_check127_validator_in_window_accepted(tmp_path: Path) -> None:
    """Adjacent validate_custody call → no violation."""
    root = _mkrepo(
        tmp_path,
        {
            "tools/good_emitter.py": (
                "from tac.continual_learning import AUTHORITATIVE_TAGS\n"
                "ok, reason = result.validate_custody()\n"
                "if not ok and result.evidence_tag in AUTHORITATIVE_TAGS:\n"
                "    print('[contest-CPU]', reason)\n"
            ),
        },
    )
    violations = check_authoritative_tag_requires_custody_metadata(
        repo_root=root, strict=False, verbose=False
    )
    assert violations == []


def test_check127_posterior_update_routing_accepted(tmp_path: Path) -> None:
    """Routing through posterior_update is treated as the canonical path."""
    root = _mkrepo(
        tmp_path,
        {
            "tools/router.py": (
                "from tac.continual_learning import (\n"
                "    AUTHORITATIVE_TAGS, posterior_update,\n"
                ")\n"
                "if tag in AUTHORITATIVE_TAGS:  # '[contest-CUDA]' literal here\n"
                "    update = posterior_update(posterior, result)\n"
            ),
        },
    )
    violations = check_authoritative_tag_requires_custody_metadata(
        repo_root=root, strict=False, verbose=False
    )
    assert violations == []


def test_check127_comment_validator_token_does_not_whitelist(tmp_path: Path) -> None:
    """A nearby comment mentioning posterior_update is not executable custody."""
    root = _mkrepo(
        tmp_path,
        {
            "tools/comment_token.py": (
                "from tac.continual_learning import AUTHORITATIVE_TAGS\n"
                "# TODO: route this through posterior_update(posterior, result)\n"
                "if tag in AUTHORITATIVE_TAGS:  # '[contest-CUDA]'\n"
                "    print(tag)\n"
            ),
        },
    )
    violations = check_authoritative_tag_requires_custody_metadata(
        repo_root=root, strict=False, verbose=False
    )
    assert len(violations) == 1
    assert "tools/comment_token.py" in violations[0]


def test_check127_string_validator_token_does_not_whitelist(tmp_path: Path) -> None:
    """A nearby string literal mentioning validate_custody is not custody."""
    root = _mkrepo(
        tmp_path,
        {
            "tools/string_token.py": (
                "from tac.continual_learning import AUTHORITATIVE_TAGS\n"
                "hint = 'call validate_custody before promotion'\n"
                "if tag in AUTHORITATIVE_TAGS:  # '[contest-CPU]'\n"
                "    print(tag)\n"
            ),
        },
    )
    violations = check_authoritative_tag_requires_custody_metadata(
        repo_root=root, strict=False, verbose=False
    )
    assert len(violations) == 1
    assert "tools/string_token.py" in violations[0]


def test_check127_same_line_waiver_accepted(tmp_path: Path) -> None:
    """`# CUSTODY_VALIDATOR_OK:<reason>` waiver → no violation."""
    root = _mkrepo(
        tmp_path,
        {
            "tools/waived.py": (
                "from tac.continual_learning import AUTHORITATIVE_TAGS\n"
                "if tag in AUTHORITATIVE_TAGS and tag == '[contest-CPU]':"
                "  # CUSTODY_VALIDATOR_OK: tag-only print, no promotion\n"
                "    print('hi')\n"
            ),
        },
    )
    violations = check_authoritative_tag_requires_custody_metadata(
        repo_root=root, strict=False, verbose=False
    )
    assert violations == []


def test_check127_strict_raises_preflighterror(tmp_path: Path) -> None:
    """strict=True → PreflightError on bypass."""
    root = _mkrepo(
        tmp_path,
        {
            "tools/bad.py": (
                "from tac.continual_learning import AUTHORITATIVE_TAGS\n"
                "if tag in AUTHORITATIVE_TAGS:  # '[contest-CUDA]'\n"
                "    pass\n"
            ),
        },
    )
    with pytest.raises(PreflightError, match="check_authoritative_tag"):
        check_authoritative_tag_requires_custody_metadata(
            repo_root=root, strict=True, verbose=False
        )


def test_check127_test_files_excluded(tmp_path: Path) -> None:
    """Test files (`/tests/` or `test_*.py`) are excluded from the scan."""
    root = _mkrepo(
        tmp_path,
        {
            "src/tac/tests/test_thing.py": (
                "from tac.continual_learning import AUTHORITATIVE_TAGS\n"
                "if tag in AUTHORITATIVE_TAGS:  # '[contest-CPU]'\n"
                "    pass\n"
            ),
        },
    )
    violations = check_authoritative_tag_requires_custody_metadata(
        repo_root=root, strict=False, verbose=False
    )
    assert violations == []


def test_check127_canonical_continual_learning_excluded(tmp_path: Path) -> None:
    """The canonical `src/tac/continual_learning.py` is excluded (it owns the symbols)."""
    root = _mkrepo(
        tmp_path,
        {
            "src/tac/continual_learning.py": (
                "AUTHORITATIVE_TAGS = frozenset({'[contest-CUDA]', '[contest-CPU]'})\n"
                "if tag in AUTHORITATIVE_TAGS:  # '[contest-CPU]' literal\n"
                "    pass\n"
            ),
        },
    )
    violations = check_authoritative_tag_requires_custody_metadata(
        repo_root=root, strict=False, verbose=False
    )
    assert violations == []


def test_check127_live_repo_under_strict_does_not_explode(tmp_path: Path) -> None:
    """Sanity: scanning the live repo returns a finite list (warn-only acceptable)."""
    # Use the actual repo root (3 levels up from this test file).
    repo_root = Path(__file__).resolve().parents[3]
    violations = check_authoritative_tag_requires_custody_metadata(
        repo_root=repo_root, strict=False, verbose=False
    )
    # The check is warn-only initially; live count is informational here.
    assert isinstance(violations, list)


# ─────────────────────────────────────────────────────────────────────────
# Catalog #128 — continual_learning writes must use the locked path.
# ─────────────────────────────────────────────────────────────────────────


def test_check128_clean_repo_passes(tmp_path: Path) -> None:
    """No save_posterior reference → no violations."""
    root = _mkrepo(tmp_path, {"src/tac/foo.py": "x = 1\n"})
    violations = check_continual_learning_writes_use_lock(
        repo_root=root, strict=False, verbose=False
    )
    assert violations == []


def test_check128_bare_save_caught(tmp_path: Path) -> None:
    """Direct `save_posterior(...)` call without locked-path use → violation."""
    root = _mkrepo(
        tmp_path,
        {
            "tools/bad_writer.py": (
                "from tac.continual_learning import save_posterior\n"
                "def f(p):\n"
                "    save_posterior(p)\n"
            ),
        },
    )
    violations = check_continual_learning_writes_use_lock(
        repo_root=root, strict=False, verbose=False
    )
    assert len(violations) == 1
    assert "tools/bad_writer.py" in violations[0]
    assert "[Check 128]" in violations[0]


def test_check128_posterior_update_locked_elsewhere_does_not_whitelist_save(tmp_path: Path) -> None:
    """A posterior_update_locked mention does not make a later bare save safe."""
    root = _mkrepo(
        tmp_path,
        {
            "tools/canonical.py": (
                "from tac.continual_learning import (\n"
                "    save_posterior, posterior_update_locked,\n"
                ")\n"
                "def write(p):\n"
                "    posterior_update_locked(...)\n"
                "    save_posterior(p)\n"
            ),
        },
    )
    violations = check_continual_learning_writes_use_lock(
        repo_root=root, strict=False, verbose=False
    )
    assert len(violations) == 1
    assert "tools/canonical.py" in violations[0]


def test_check128_lock_context_manager_accepted(tmp_path: Path) -> None:
    """File that uses `_posterior_lock` (the lock CM) is also canonical writer."""
    root = _mkrepo(
        tmp_path,
        {
            "tools/cm_writer.py": (
                "from tac.continual_learning import save_posterior, _posterior_lock\n"
                "def f(p):\n"
                "    with _posterior_lock(p):\n"
                "        save_posterior(p)\n"
            ),
        },
    )
    violations = check_continual_learning_writes_use_lock(
        repo_root=root, strict=False, verbose=False
    )
    assert violations == []


def test_check128_same_line_waiver_accepted(tmp_path: Path) -> None:
    """`# SAVE_POSTERIOR_LOCKED_OK:<reason>` waiver → no violation."""
    root = _mkrepo(
        tmp_path,
        {
            "tools/waived.py": (
                "from tac.continual_learning import save_posterior\n"
                "def f(p):\n"
                "    save_posterior(p)  # SAVE_POSTERIOR_LOCKED_OK: single-writer test\n"
            ),
        },
    )
    violations = check_continual_learning_writes_use_lock(
        repo_root=root, strict=False, verbose=False
    )
    assert violations == []


def test_check128_strict_raises_preflighterror(tmp_path: Path) -> None:
    """strict=True → PreflightError on bare save."""
    root = _mkrepo(
        tmp_path,
        {
            "tools/bad.py": (
                "from tac.continual_learning import save_posterior\n"
                "save_posterior(p)\n"
            ),
        },
    )
    with pytest.raises(PreflightError, match="check_continual_learning_writes_use_lock"):
        check_continual_learning_writes_use_lock(
            repo_root=root, strict=True, verbose=False
        )


def test_check128_test_files_excluded(tmp_path: Path) -> None:
    """Test files are excluded from the scan."""
    root = _mkrepo(
        tmp_path,
        {
            "src/tac/tests/test_writer.py": (
                "from tac.continual_learning import save_posterior\n"
                "save_posterior(p)\n"
            ),
        },
    )
    violations = check_continual_learning_writes_use_lock(
        repo_root=root, strict=False, verbose=False
    )
    assert violations == []


def test_check128_canonical_continual_learning_excluded(tmp_path: Path) -> None:
    """The canonical implementation file is excluded."""
    root = _mkrepo(
        tmp_path,
        {
            "src/tac/continual_learning.py": (
                "def save_posterior(p):\n"
                "    pass\n"
                "save_posterior(p)\n"
            ),
        },
    )
    violations = check_continual_learning_writes_use_lock(
        repo_root=root, strict=False, verbose=False
    )
    assert violations == []


def test_check128_imports_only_accepted(tmp_path: Path) -> None:
    """Importing `save_posterior` without calling it → no violation."""
    root = _mkrepo(
        tmp_path,
        {
            "tools/just_imports.py": (
                "from tac.continual_learning import save_posterior  # re-exported\n"
                "__all__ = ['save_posterior']\n"
            ),
        },
    )
    violations = check_continual_learning_writes_use_lock(
        repo_root=root, strict=False, verbose=False
    )
    assert violations == []


def test_check128_live_repo_under_strict_does_not_explode() -> None:
    """Sanity: scanning the live repo returns a finite list."""
    repo_root = Path(__file__).resolve().parents[3]
    violations = check_continual_learning_writes_use_lock(
        repo_root=repo_root, strict=False, verbose=False
    )
    assert isinstance(violations, list)


# ─────────────────────────────────────────────────────────────────────────
# Catalog #130 — broader tag/grade predicates need local custody context.
# ─────────────────────────────────────────────────────────────────────────


def test_check130_bare_evidence_grade_membership_caught(tmp_path: Path) -> None:
    """Tag/grade membership without local custody context is a violation."""
    root = _mkrepo(
        tmp_path,
        {
            "tools/bad_grade_gate.py": (
                "def promote(evidence_grade):\n"
                "    return evidence_grade in {'A++', 'contest-cuda'}\n"
            ),
        },
    )
    violations = check_no_tag_only_custody_validation(
        repo_root=root, strict=False, verbose=False
    )
    assert len(violations) == 1
    assert "[Check 130]" in violations[0]


def test_check130_validator_elsewhere_does_not_whitelist_all(tmp_path: Path) -> None:
    """A distant validator mention must not hide a later grade-only gate."""
    root = _mkrepo(
        tmp_path,
        {
            "tools/mixed_grade_gate.py": (
                "def safe(result):\n"
                "    return result.validate_custody()\n"
                + "\n" * 10
                + "def unsafe(evidence_grade):\n"
                "    return evidence_grade in {'A++'}\n"
            ),
        },
    )
    violations = check_no_tag_only_custody_validation(
        repo_root=root, strict=False, verbose=False
    )
    assert len(violations) == 1
    assert "tools/mixed_grade_gate.py" in violations[0]


def test_check130_local_blocker_context_accepted(tmp_path: Path) -> None:
    """Fail-closed blocker or archive custody context near the gate is accepted."""
    root = _mkrepo(
        tmp_path,
        {
            "tools/good_grade_gate.py": (
                "def promote(evidence_grade, row):\n"
                "    blockers = []\n"
                "    blockers.extend(promotable_exact_cuda_evidence_blockers(row))\n"
                "    if row.get('archive_sha256') and evidence_grade in {'A++'}:\n"
                "        return not blockers\n"
                "    return False\n"
            ),
        },
    )
    violations = check_no_tag_only_custody_validation(
        repo_root=root, strict=False, verbose=False
    )
    assert violations == []


def test_check130_same_line_waiver_accepted(tmp_path: Path) -> None:
    """Same-line waiver is accepted for read-only filters."""
    root = _mkrepo(
        tmp_path,
        {
            "tools/grade_filter.py": (
                "def filter_row(evidence_grade):\n"
                "    return evidence_grade in {'A++'}  # CUSTODY_VALIDATOR_OK: read-only report filter\n"
            ),
        },
    )
    violations = check_no_tag_only_custody_validation(
        repo_root=root, strict=False, verbose=False
    )
    assert violations == []


# ─────────────────────────────────────────────────────────────────────────
# Catalog #131 — shared-state writes need local lock context.
# ─────────────────────────────────────────────────────────────────────────


def test_check131_bare_shared_state_write_caught(tmp_path: Path) -> None:
    """Bare write to a known shared-state path is a violation."""
    root = _mkrepo(
        tmp_path,
        {
            "tools/bad_state_writer.py": (
                "from pathlib import Path\n"
                "STATE = Path('.omx/state/foo.json')\n"
                "def write():\n"
                "    STATE.write_text('{}')\n"
            ),
        },
    )
    violations = check_no_bare_writes_to_shared_state(
        repo_root=root, strict=False, verbose=False
    )
    assert len(violations) == 1
    assert "[Check 131]" in violations[0]


def test_check131_lock_elsewhere_does_not_whitelist_all(tmp_path: Path) -> None:
    """A distant lock token must not hide a later bare write."""
    root = _mkrepo(
        tmp_path,
        {
            "tools/mixed_state_writer.py": (
                "from pathlib import Path\n"
                "import fcntl\n"
                "STATE = Path('.omx/state/foo.json')\n"
                "def safe(lockfd):\n"
                "    fcntl.flock(lockfd, fcntl.LOCK_EX)\n"
                + "\n" * 24
                + "def unsafe():\n"
                "    STATE.write_text('{}')\n"
            ),
        },
    )
    violations = check_no_bare_writes_to_shared_state(
        repo_root=root, strict=False, verbose=False
    )
    assert len(violations) == 1


def test_check131_canonical_helper_name_after_write_does_not_whitelist(
    tmp_path: Path,
) -> None:
    """A helper call near a bare write is not a lock proof for that write."""
    root = _mkrepo(
        tmp_path,
        {
            "tools/mixed_state_writer.py": (
                "from pathlib import Path\n"
                "STATE = Path('.omx/state/foo.json')\n"
                "from tac.vastai_tracker import register_instance\n"
                "def unsafe():\n"
                "    STATE.write_text('{}')\n"
                "    register_instance('i-1', 'label')\n"
            ),
        },
    )
    violations = check_no_bare_writes_to_shared_state(
        repo_root=root, strict=False, verbose=False
    )
    assert len(violations) == 1
    assert "[Check 131]" in violations[0]


def test_check131_local_lock_context_accepted(tmp_path: Path) -> None:
    """Codex round 8 MEDIUM (in-place harden of #131): lock alone is no
    longer sufficient to waive direct write_text on a shared-state path —
    the canonical helper invocation OR explicit transactional pattern
    (write to ``<path>.tmp`` + ``os.replace``) is required. The example
    below now uses the canonical transactional pattern.
    """
    root = _mkrepo(
        tmp_path,
        {
            "tools/good_state_writer.py": (
                "import os\n"
                "from pathlib import Path\n"
                "STATE = Path('.omx/state/foo.json')\n"
                "def write():\n"
                "    with _active_jobs_lock():\n"
                "        tmp = STATE.with_suffix('.tmp')\n"
                "        tmp.write_text('{}')\n"
                "        os.replace(tmp, STATE)\n"
            ),
        },
    )
    violations = check_no_bare_writes_to_shared_state(
        repo_root=root, strict=False, verbose=False
    )
    assert violations == [], violations


def test_check131_lock_alone_no_atomic_replace_is_violation_round8(
    tmp_path: Path,
) -> None:
    """Codex round 8 MEDIUM (NEW behavior): lock present but NO atomic-replace
    pattern → violation. This was a silent false-green before the harden."""
    root = _mkrepo(
        tmp_path,
        {
            "tools/lock_only_writer.py": (
                "from pathlib import Path\n"
                "STATE = Path('.omx/state/foo.json')\n"
                "def write():\n"
                "    with _active_jobs_lock():\n"
                "        STATE.write_text('{}')\n"
            ),
        },
    )
    violations = check_no_bare_writes_to_shared_state(
        repo_root=root, strict=False, verbose=False
    )
    assert len(violations) == 1, violations
    assert "atomic" in violations[0].lower()


def test_check131_same_line_waiver_accepted(tmp_path: Path) -> None:
    """Same-line waiver is accepted for externally serialized writers."""
    root = _mkrepo(
        tmp_path,
        {
            "tools/waived_state_writer.py": (
                "from pathlib import Path\n"
                "STATE = Path('.omx/state/foo.json')\n"
                "def write():\n"
                "    STATE.write_text('{}')  # BARE_WRITE_OK: externally serialized by scheduler\n"
            ),
        },
    )
    violations = check_no_bare_writes_to_shared_state(
        repo_root=root, strict=False, verbose=False
    )
    assert violations == []


def test_check131_uses_source_index_marker_prefilter(tmp_path: Path) -> None:
    """SourceIndex-backed runs should build substring candidates once."""
    root = _mkrepo(
        tmp_path,
        {
            "tools/no_marker.py": "def noop():\n    return 1\n",
            "tools/bad_state_writer.py": (
                "from pathlib import Path\n"
                "STATE = Path('.omx/state/foo.json')\n"
                "def write():\n"
                "    STATE.write_text('{}')\n"
            ),
        },
    )

    with source_index_context(root) as index:
        violations = check_no_bare_writes_to_shared_state(
            repo_root=root, strict=False, verbose=False
        )
        stats = index.stats()

    assert len(violations) == 1
    assert "tools/bad_state_writer.py" in violations[0]
    assert stats["substring_index_entries"] > 0
