# SPDX-License-Identifier: MIT
"""Tests for Catalog #213 ``check_comma2k19_downloads_route_through_canonical_cache``.

The gate refuses any caller under ``src/tac/`` / ``tools/`` / ``experiments/`` /
``scripts/`` that fetches Comma2k19 chunks via bare urlopen / requests / curl
/ wget / aria2c OUTSIDE the canonical ``Comma2k19LocalCache`` helper.

Coverage:
- Live-repo regression guard (clean at landing).
- Positive (bare urlopen / requests / shell wget+curl+aria2c).
- Negative (canonical-cache token in lookback / .fetch_chunk invocation).
- Waiver semantics (rationale accepted, placeholder rejected, file-level).
- Exempt markers (experiments/results/ + intake clones + reports/raw/).
- Self-exempt (canonical implementation files).
- Strict-mode raises with PreflightError.
- AST/comment false-positive guards.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tac.preflight import (
    PreflightError,
    check_comma2k19_downloads_route_through_canonical_cache,
)


# ---------------------------------------------------------------------------
# Live-repo regression guard
# ---------------------------------------------------------------------------


def test_live_repo_has_zero_violations() -> None:
    """At landing, the live repo MUST be clean (STRICT @ 0)."""
    v = check_comma2k19_downloads_route_through_canonical_cache(
        strict=False, verbose=False
    )
    assert v == []


def test_live_repo_strict_mode_passes() -> None:
    """STRICT mode on the live repo MUST NOT raise."""
    check_comma2k19_downloads_route_through_canonical_cache(
        strict=True, verbose=False
    )


# ---------------------------------------------------------------------------
# Positive cases — bare downloads flagged
# ---------------------------------------------------------------------------


def _write_module(root: Path, rel: str, body: str) -> None:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(body)


def test_bare_urlopen_with_raw_githubusercontent_flagged(tmp_path) -> None:
    _write_module(
        tmp_path,
        "src/tac/foo.py",
        """
import urllib.request
def fetch():
    url = "https://raw.githubusercontent.com/commaai/comma2k19/master/x.hevc"
    return urllib.request.urlopen(url).read()
""",
    )
    v = check_comma2k19_downloads_route_through_canonical_cache(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert len(v) == 1
    assert "src/tac/foo.py" in v[0]


def test_bare_requests_get_flagged(tmp_path) -> None:
    _write_module(
        tmp_path,
        "tools/grab.py",
        """
import requests
def grab():
    url = "https://github.com/commaai/comma2k19/raw/master/Example_1/video.hevc"
    return requests.get(url).content
""",
    )
    v = check_comma2k19_downloads_route_through_canonical_cache(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert len(v) == 1


def test_shell_wget_flagged(tmp_path) -> None:
    _write_module(
        tmp_path,
        "scripts/fetch.sh",
        """#!/bin/bash
set -euo pipefail
wget https://raw.githubusercontent.com/commaai/comma2k19/master/x.hevc
""",
    )
    v = check_comma2k19_downloads_route_through_canonical_cache(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert len(v) == 1


def test_shell_curl_flagged(tmp_path) -> None:
    _write_module(
        tmp_path,
        "scripts/fetch2.sh",
        """#!/bin/bash
curl -O https://raw.githubusercontent.com/commaai/comma2k19/master/x.hevc
""",
    )
    v = check_comma2k19_downloads_route_through_canonical_cache(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert len(v) == 1


def test_aria2c_torrent_flagged(tmp_path) -> None:
    _write_module(
        tmp_path,
        "scripts/torrent.sh",
        """#!/bin/bash
aria2c "magnet:academictorrents.com/details/65a2fbc964078aff62076ff4e103f18b951c5ddb"
""",
    )
    v = check_comma2k19_downloads_route_through_canonical_cache(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert len(v) == 1


def test_urlretrieve_flagged(tmp_path) -> None:
    _write_module(
        tmp_path,
        "experiments/dl.py",
        """
from urllib.request import urlretrieve
url = "https://raw.githubusercontent.com/commaai/comma2k19/master/x.hevc"
urlretrieve(url, "/some/path")
""",
    )
    v = check_comma2k19_downloads_route_through_canonical_cache(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert len(v) == 1


def test_multiple_violations_same_file(tmp_path) -> None:
    _write_module(
        tmp_path,
        "src/tac/multi.py",
        """
import urllib.request
import requests
def fetch_a():
    url = "https://raw.githubusercontent.com/commaai/comma2k19/master/a.hevc"
    return urllib.request.urlopen(url).read()
def fetch_b():
    url = "https://github.com/commaai/comma2k19/raw/master/b.hevc"
    return requests.get(url).content
""",
    )
    v = check_comma2k19_downloads_route_through_canonical_cache(
        repo_root=tmp_path, strict=False, verbose=False
    )
    # Both fetch_a and fetch_b should be flagged.
    assert len(v) >= 2


def test_multiple_violations_across_files(tmp_path) -> None:
    _write_module(
        tmp_path,
        "src/tac/a.py",
        """
import urllib.request
url = "https://raw.githubusercontent.com/commaai/comma2k19/x.hevc"
urllib.request.urlopen(url).read()
""",
    )
    _write_module(
        tmp_path,
        "tools/b.py",
        """
import requests
url = "https://github.com/commaai/comma2k19/raw/master/y.hevc"
requests.get(url).content
""",
    )
    v = check_comma2k19_downloads_route_through_canonical_cache(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert len(v) == 2


def test_academictorrents_url_flagged(tmp_path) -> None:
    _write_module(
        tmp_path,
        "scripts/torrent.sh",
        """#!/bin/bash
transmission-cli "https://academictorrents.com/details/65a2fbc964078aff62076ff4e103f18b951c5ddb"
""",
    )
    v = check_comma2k19_downloads_route_through_canonical_cache(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert len(v) == 1


# ---------------------------------------------------------------------------
# Negative cases — canonical-cache token accepted
# ---------------------------------------------------------------------------


def test_canonical_cache_token_in_window_accepted(tmp_path) -> None:
    _write_module(
        tmp_path,
        "src/tac/good.py",
        """
from tac.substrates.pretrained_driving_prior import Comma2k19LocalCache
import urllib.request
cache = Comma2k19LocalCache()
def fetch():
    # The cache routes the download; this line just shows a URL for reference.
    url = "https://raw.githubusercontent.com/commaai/comma2k19/master/x.hevc"
    return cache.fetch_chunk("example_1")
""",
    )
    v = check_comma2k19_downloads_route_through_canonical_cache(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert v == []


def test_fetch_chunk_call_in_window_accepted(tmp_path) -> None:
    _write_module(
        tmp_path,
        "tools/dl.py",
        """
from tac.substrates.pretrained_driving_prior import Comma2k19LocalCache
cache = Comma2k19LocalCache()
def grab():
    url = "https://raw.githubusercontent.com/commaai/comma2k19/master/x.hevc"
    return cache.fetch_chunk("test")
""",
    )
    v = check_comma2k19_downloads_route_through_canonical_cache(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert v == []


def test_url_mention_without_download_call_accepted(tmp_path) -> None:
    """Files mentioning the URL in a comment/docstring but NOT calling download tokens are accepted."""
    _write_module(
        tmp_path,
        "src/tac/doc.py",
        '''
"""Reference doc.

The canonical source is github.com/commaai/comma2k19/raw/master.
"""
def f():
    return 42
''',
    )
    v = check_comma2k19_downloads_route_through_canonical_cache(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert v == []


def test_download_token_without_comma2k19_url_accepted(tmp_path) -> None:
    """urlopen call against an unrelated URL is NOT in-scope for this gate."""
    _write_module(
        tmp_path,
        "src/tac/unrelated.py",
        """
import urllib.request
def fetch():
    return urllib.request.urlopen("https://example.com/api/v1/foo").read()
""",
    )
    v = check_comma2k19_downloads_route_through_canonical_cache(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert v == []


# ---------------------------------------------------------------------------
# Waiver semantics
# ---------------------------------------------------------------------------


def test_same_line_waiver_with_rationale_accepted(tmp_path) -> None:
    _write_module(
        tmp_path,
        "scripts/legacy.sh",
        """#!/bin/bash
wget https://raw.githubusercontent.com/commaai/comma2k19/master/x.hevc  # COMMA2K19_DOWNLOAD_OK:legacy-operator-reviewed-2026-05-01
""",
    )
    v = check_comma2k19_downloads_route_through_canonical_cache(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert v == []


def test_same_line_waiver_placeholder_rejected(tmp_path) -> None:
    """``COMMA2K19_DOWNLOAD_OK:<reason>`` literal placeholder is rejected."""
    _write_module(
        tmp_path,
        "scripts/bad_waiver.sh",
        """#!/bin/bash
wget https://raw.githubusercontent.com/commaai/comma2k19/master/x.hevc  # COMMA2K19_DOWNLOAD_OK:<reason>
""",
    )
    v = check_comma2k19_downloads_route_through_canonical_cache(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert len(v) == 1


def test_file_level_waiver_accepted(tmp_path) -> None:
    _write_module(
        tmp_path,
        "tools/oneoff.py",
        """# COMMA2K19_DOWNLOAD_OK_FILE:operator-reviewed-2026-05-14
import urllib.request
url = "https://raw.githubusercontent.com/commaai/comma2k19/master/x.hevc"
urllib.request.urlopen(url).read()
""",
    )
    v = check_comma2k19_downloads_route_through_canonical_cache(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert v == []


def test_file_level_waiver_placeholder_rejected(tmp_path) -> None:
    _write_module(
        tmp_path,
        "tools/bad_file_waiver.py",
        """# COMMA2K19_DOWNLOAD_OK_FILE:<reason>
import urllib.request
url = "https://raw.githubusercontent.com/commaai/comma2k19/master/x.hevc"
urllib.request.urlopen(url).read()
""",
    )
    v = check_comma2k19_downloads_route_through_canonical_cache(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert len(v) == 1


# ---------------------------------------------------------------------------
# Exempt marker behavior
# ---------------------------------------------------------------------------


def test_experiments_results_path_excluded(tmp_path) -> None:
    _write_module(
        tmp_path,
        "experiments/results/lane_x/build.py",
        """
import urllib.request
url = "https://raw.githubusercontent.com/commaai/comma2k19/master/x.hevc"
urllib.request.urlopen(url).read()
""",
    )
    v = check_comma2k19_downloads_route_through_canonical_cache(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert v == []


def test_intake_clones_path_excluded(tmp_path) -> None:
    _write_module(
        tmp_path,
        "experiments/results/public_pr107_intake_codex/source/x.py",
        """
import urllib.request
url = "https://raw.githubusercontent.com/commaai/comma2k19/master/x.hevc"
urllib.request.urlopen(url).read()
""",
    )
    v = check_comma2k19_downloads_route_through_canonical_cache(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert v == []


def test_oss_export_path_excluded(tmp_path) -> None:
    _write_module(
        tmp_path,
        "src/tac/.omx/oss_export/mirror/x.py",
        """
import urllib.request
url = "https://raw.githubusercontent.com/commaai/comma2k19/master/x.hevc"
urllib.request.urlopen(url).read()
""",
    )
    # The exempt fragment is ``.omx/oss_export/`` — at a tools/ relative
    # path this won't fire because tools/ is the scope root. We use a path
    # under src/tac/.omx/oss_export/... but the file glob only matches src/
    # subtrees that exist; the file is created so let's verify.
    v = check_comma2k19_downloads_route_through_canonical_cache(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert v == []


def test_reports_raw_path_excluded(tmp_path) -> None:
    """``reports/raw/`` is in the exempt-fragment list but only fires if scanned."""
    # reports/raw/ is not under any of the 4 scope dirs, so it's never scanned.
    # Verify by creating a file under reports/raw/.
    _write_module(
        tmp_path,
        "reports/raw/x.py",
        """
import urllib.request
url = "https://raw.githubusercontent.com/commaai/comma2k19/master/x.hevc"
urllib.request.urlopen(url).read()
""",
    )
    v = check_comma2k19_downloads_route_through_canonical_cache(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert v == []


def test_test_files_excluded(tmp_path) -> None:
    _write_module(
        tmp_path,
        "src/tac/tests/test_thing.py",
        """
import urllib.request
url = "https://raw.githubusercontent.com/commaai/comma2k19/master/x.hevc"
urllib.request.urlopen(url).read()
""",
    )
    v = check_comma2k19_downloads_route_through_canonical_cache(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert v == []


def test_test_named_file_excluded(tmp_path) -> None:
    """Files starting with ``test_`` are skipped."""
    _write_module(
        tmp_path,
        "src/tac/test_thing.py",
        """
import urllib.request
url = "https://raw.githubusercontent.com/commaai/comma2k19/master/x.hevc"
urllib.request.urlopen(url).read()
""",
    )
    v = check_comma2k19_downloads_route_through_canonical_cache(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert v == []


# ---------------------------------------------------------------------------
# Self-exempt canonical files
# ---------------------------------------------------------------------------


def test_canonical_implementation_self_exempt(tmp_path) -> None:
    """The canonical local_chunk_cache.py file is self-exempt."""
    _write_module(
        tmp_path,
        "src/tac/substrates/pretrained_driving_prior/local_chunk_cache.py",
        """
import urllib.request
# This is the canonical helper itself; it IS the download path.
url = "https://raw.githubusercontent.com/commaai/comma2k19/master/x.hevc"
urllib.request.urlopen(url).read()
""",
    )
    v = check_comma2k19_downloads_route_through_canonical_cache(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert v == []


def test_log_incremental_feeder_self_exempt(tmp_path) -> None:
    """The log-incremental feeder is self-exempt (uses the canonical helper)."""
    _write_module(
        tmp_path,
        "src/tac/substrates/pretrained_driving_prior/log_incremental_feeder.py",
        """
import urllib.request
url = "https://raw.githubusercontent.com/commaai/comma2k19/master/x.hevc"
urllib.request.urlopen(url).read()
""",
    )
    v = check_comma2k19_downloads_route_through_canonical_cache(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert v == []


def test_substrate_init_self_exempt(tmp_path) -> None:
    """The substrate __init__.py is self-exempt."""
    _write_module(
        tmp_path,
        "src/tac/substrates/pretrained_driving_prior/__init__.py",
        """
import urllib.request
url = "https://raw.githubusercontent.com/commaai/comma2k19/master/x.hevc"
urllib.request.urlopen(url).read()
""",
    )
    v = check_comma2k19_downloads_route_through_canonical_cache(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert v == []


# ---------------------------------------------------------------------------
# Strict mode
# ---------------------------------------------------------------------------


def test_strict_mode_raises_on_violation(tmp_path) -> None:
    _write_module(
        tmp_path,
        "tools/grab.py",
        """
import requests
url = "https://raw.githubusercontent.com/commaai/comma2k19/master/x.hevc"
requests.get(url).content
""",
    )
    with pytest.raises(PreflightError, match="Catalog #213"):
        check_comma2k19_downloads_route_through_canonical_cache(
            repo_root=tmp_path, strict=True, verbose=False
        )


def test_strict_mode_silent_on_clean(tmp_path) -> None:
    """Clean directory tree → strict mode does NOT raise."""
    _write_module(
        tmp_path,
        "src/tac/good.py",
        """
from tac.substrates.pretrained_driving_prior import Comma2k19LocalCache
cache = Comma2k19LocalCache()
""",
    )
    check_comma2k19_downloads_route_through_canonical_cache(
        repo_root=tmp_path, strict=True, verbose=False
    )


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_no_scope_dirs_present_returns_empty(tmp_path) -> None:
    """A repo_root with no src/tac/tools/experiments/scripts dirs returns []."""
    v = check_comma2k19_downloads_route_through_canonical_cache(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert v == []


def test_non_python_non_shell_files_ignored(tmp_path) -> None:
    """A .md file with the bad pattern is NOT flagged."""
    (tmp_path / "src" / "tac").mkdir(parents=True)
    md = tmp_path / "src" / "tac" / "doc.md"
    md.write_text(
        """
# Doc

```python
import urllib.request
url = "https://raw.githubusercontent.com/commaai/comma2k19/master/x.hevc"
urllib.request.urlopen(url).read()
```
"""
    )
    v = check_comma2k19_downloads_route_through_canonical_cache(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert v == []


def test_url_only_no_download_call_in_line_or_window_accepted(tmp_path) -> None:
    """A pure-comment URL with no download call anywhere is accepted."""
    _write_module(
        tmp_path,
        "src/tac/comments.py",
        """
# We may download from https://raw.githubusercontent.com/commaai/comma2k19/master/x.hevc
# at some point. For now, just a marker.
def noop():
    return 42
""",
    )
    v = check_comma2k19_downloads_route_through_canonical_cache(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert v == []


def test_multi_line_download_call_detected(tmp_path) -> None:
    """A urlopen() call SPANNING multiple lines with the URL is detected."""
    _write_module(
        tmp_path,
        "src/tac/multi_line.py",
        """
import urllib.request
url = "https://raw.githubusercontent.com/commaai/comma2k19/master/x.hevc"
resp = urllib.request.urlopen(
    url,
    timeout=60,
)
""",
    )
    v = check_comma2k19_downloads_route_through_canonical_cache(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert len(v) == 1


def test_preflight_py_self_exempt(tmp_path) -> None:
    """preflight.py is self-exempt (carries the literal tokens for the regex)."""
    _write_module(
        tmp_path,
        "src/tac/preflight.py",
        """
# carries URL literals like 'raw.githubusercontent.com/commaai/comma2k19' for the gate
import urllib.request
url = "https://raw.githubusercontent.com/commaai/comma2k19/master/x.hevc"
urllib.request.urlopen(url).read()
""",
    )
    v = check_comma2k19_downloads_route_through_canonical_cache(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert v == []


def test_violation_capped_at_three_per_file(tmp_path) -> None:
    """A file with many violations caps the report at 3 lines per file."""
    body = "import urllib.request\n"
    for i in range(10):
        body += (
            f'url{i} = "https://raw.githubusercontent.com/commaai/comma2k19/{i}.hevc"\n'
        )
        body += f"urllib.request.urlopen(url{i}).read()\n"
    _write_module(tmp_path, "src/tac/many.py", body)
    v = check_comma2k19_downloads_route_through_canonical_cache(
        repo_root=tmp_path, strict=False, verbose=False
    )
    # Cap at 3 per file.
    assert len(v) == 3


def test_violation_message_mentions_catalog_213(tmp_path) -> None:
    _write_module(
        tmp_path,
        "src/tac/foo.py",
        """
import urllib.request
url = "https://raw.githubusercontent.com/commaai/comma2k19/x.hevc"
urllib.request.urlopen(url).read()
""",
    )
    v = check_comma2k19_downloads_route_through_canonical_cache(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert len(v) == 1
    assert "Catalog #213" in v[0]
    assert "Comma2k19LocalCache" in v[0]


def test_violation_message_mentions_waiver_form(tmp_path) -> None:
    _write_module(
        tmp_path,
        "src/tac/foo.py",
        """
import urllib.request
url = "https://raw.githubusercontent.com/commaai/comma2k19/x.hevc"
urllib.request.urlopen(url).read()
""",
    )
    v = check_comma2k19_downloads_route_through_canonical_cache(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert "COMMA2K19_DOWNLOAD_OK" in v[0]


# ---------------------------------------------------------------------------
# Orchestrator-callsite regression guard
# ---------------------------------------------------------------------------


def test_orchestrator_calls_gate_strict_true() -> None:
    """``preflight_all()`` MUST wire this gate at strict=True per CLAUDE.md."""
    from tac import preflight

    src = Path(preflight.__file__).read_text()
    # Find the canonical wire-in pattern.
    assert (
        "check_comma2k19_downloads_route_through_canonical_cache(\n"
        "            strict=True, verbose=verbose,"
        in src
    ) or (
        "check_comma2k19_downloads_route_through_canonical_cache(\n            strict=True"
        in src
    )
