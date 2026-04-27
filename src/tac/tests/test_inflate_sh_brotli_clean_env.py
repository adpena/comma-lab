"""Codex R5-2 #3 regression: brotli must be available in inflate.sh's
runtime in a CLEAN CONTEST ENVIRONMENT.

The codex finding observed that ``submissions/robust_current/inflate.sh``
invoked ``uv run python inflate_renderer.py`` without ``--with brotli`` or
a runtime extra, while ``brotli`` lived only in the optional ``runtime``
extra of pyproject.toml. In a clean contest evaluator environment (a fresh
T4 with the contest's own pyproject.toml on disk, not ours), the brotli
import inside ``_decompress_brotli_in_archive`` would FATAL-EXIT the
moment it saw a ``.br`` file in the archive — which is the entire point
of the Lane B-alt brotli stack.

The fix is two-pronged and this file pins both:

  1. ``inflate.sh`` MUST pass ``--with brotli`` to ``uv run`` for the
     PYTHON_INFLATE=renderer codepath (the only one that consumes .br
     files via _decompress_brotli_in_archive).
  2. ``brotli`` MUST be in ``[project].dependencies`` of pyproject.toml
     (always-installed) so any environment that already pulled the ``tac``
     wheel — like the canonical ``remote_*_bootstrap.sh`` flows — has it
     even before the inflate path requests it.

A POSITIVE round-trip test validates that the helper actually decompresses
a .br file in a tmp dir using the same library function (`brotli`) the
inflate path ends up importing. We do not invoke the full inflate.sh from
inside the test because it requires `uv`, ffmpeg, and a real archive on
disk; the SHAPE of the test (clean tmp dir + helper round-trip) is what
the codex finding asked for, and we add the contract pins above so the
shell-level fix cannot silently regress.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[3]
INFLATE_SH = REPO / "submissions" / "robust_current" / "inflate.sh"
INFLATE_PY = REPO / "submissions" / "robust_current" / "inflate_renderer.py"
PYPROJECT = REPO / "pyproject.toml"


def test_inflate_sh_renderer_path_passes_with_brotli() -> None:
    """The PYTHON_INFLATE=renderer arm of inflate.sh MUST request
    ``--with brotli`` so a clean contest env (no pre-installed extras) can
    decompress a renderer.bin.br archive.

    Without this flag, an archive built with --use-brotli = silent fatal
    exit at _decompress_brotli_in_archive on a fresh T4. This is the
    Lane B-alt deployment blocker the codex finding flagged.
    """
    assert INFLATE_SH.exists(), f"inflate.sh missing at {INFLATE_SH}"
    text = INFLATE_SH.read_text()

    # Locate the renderer codepath (PYTHON_INFLATE=renderer) by looking
    # for the inflate_renderer.py invocation. Then assert --with brotli
    # appears on or near that line (within the same uv run invocation).
    # We use a regex that allows other --with flags (av/torch/numpy) to
    # appear in any order; we just require brotli to be present.
    renderer_invocation = re.search(
        r'"\$UV_BIN"\s+run\s+([^\n]*?)python[^\n]*inflate_renderer\.py',
        text,
    )
    assert renderer_invocation is not None, (
        "Could not locate the inflate_renderer.py invocation in inflate.sh. "
        "Did the layout of the renderer codepath change?"
    )
    flags = renderer_invocation.group(1)
    assert "--with brotli" in flags, (
        "inflate.sh's PYTHON_INFLATE=renderer codepath does NOT pass "
        "--with brotli to uv run. Captured flags: "
        f"{flags!r}\n"
        "Without --with brotli a clean contest env will fatal-exit at "
        "_decompress_brotli_in_archive when it encounters renderer.bin.br. "
        "Add `--with brotli` to the `uv run` invocation."
    )


def test_brotli_in_mandatory_dependencies() -> None:
    """``brotli`` must live in ``[project].dependencies`` (always-installed),
    NOT just in the ``[project.optional-dependencies].runtime`` extra.

    Belt-and-braces against the inflate.sh fix: any env that already
    installed the tac wheel — which is what the canonical bootstraps do —
    pulls brotli automatically, no --with flag required.
    """
    text = PYPROJECT.read_text()
    # Carve out the [project] dependencies block (between
    # `dependencies = [` and the closing `]` before any next [section]).
    # Using a multi-line regex; the file is small so no need for tomllib.
    deps_match = re.search(
        r'^dependencies\s*=\s*\[(.*?)^\]',
        text,
        flags=re.MULTILINE | re.DOTALL,
    )
    assert deps_match is not None, (
        "Could not find a top-level [project].dependencies block in "
        "pyproject.toml — was it accidentally removed or renamed?"
    )
    deps_block = deps_match.group(1)
    assert "brotli" in deps_block, (
        "`brotli` is missing from [project].dependencies. The codex R5-2 "
        "#3 fix promoted it from the optional `runtime` extra so that any "
        "environment installing the `tac` package (including the implicit "
        "`uv run` env) gets it for free.\n"
        "Found dependencies block:\n" + deps_block
    )


def test_inflate_renderer_brotli_helper_round_trip(tmp_path: Path) -> None:
    """End-to-end sanity in a clean tmp dir: write a .br file, run the
    inflate-side helper, assert the original bytes come back.

    This is the positive contract the codex finding asked for. We use the
    same helper inflate_renderer.py uses (compress with `brotli` lib +
    decompress with same lib) so any future divergence (e.g. accidental
    switch to a different compressor) surfaces here.
    """
    pytest.importorskip(
        "brotli",
        reason="brotli mandatory dep — install with `uv pip install brotli`",
    )
    import brotli  # noqa: E402

    archive_dir = tmp_path / "archive"
    archive_dir.mkdir()
    payload = b"renderer-binary-bytes-" * 4096  # ~88KB of structured data
    compressed = brotli.compress(payload, quality=11)
    (archive_dir / "renderer.bin.br").write_bytes(compressed)

    # Import the inflate-side decompress helper directly — it has no
    # heavy torch/av side-effects beyond the lazy `import brotli` we want
    # to exercise.
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "_inflate_renderer_for_test", INFLATE_PY
    )
    assert spec is not None and spec.loader is not None
    # NB: we do NOT exec the whole module — `import torch` at module top
    # is acceptable for a real env but we want this test to stay light.
    # Instead read the helper's source and exec only the helper symbol.
    src = INFLATE_PY.read_text()
    # Locate `_decompress_brotli_in_archive` as a self-contained function
    # plus its required imports (sys, Path).
    helper_src_match = re.search(
        r'def _decompress_brotli_in_archive\(.*?(?=\n\n# ===|\nclass |\ndef \w)',
        src, flags=re.DOTALL,
    )
    assert helper_src_match is not None, (
        "Could not extract _decompress_brotli_in_archive from "
        f"{INFLATE_PY}. Did the function rename?"
    )
    helper_src = "import sys\nfrom pathlib import Path\n" + helper_src_match.group(0)
    ns: dict = {}
    exec(compile(helper_src, str(INFLATE_PY), "exec"), ns)
    helper = ns["_decompress_brotli_in_archive"]

    # Execute on the tmp archive dir — must NOT raise, must NOT exit.
    helper(str(archive_dir))

    out = archive_dir / "renderer.bin"
    assert out.exists(), (
        "Inflate-side brotli helper did not produce renderer.bin from "
        "renderer.bin.br. Either decompression silently failed or the "
        ".br→.bin suffix-stripping logic changed."
    )
    assert out.read_bytes() == payload, (
        "Brotli round-trip byte mismatch — the inflate path will produce "
        "a corrupt renderer at contest time."
    )
    # The .br must be removed so subsequent inflate steps don't re-process it.
    assert not (archive_dir / "renderer.bin.br").exists(), (
        ".br file was not unlinked after decompression — inflate would "
        "re-attempt decompression on a stale path on every run."
    )


def test_inflate_renderer_brotli_helper_no_op_when_no_br(tmp_path: Path) -> None:
    """If no .br files exist, the helper must be a NO-OP. Critical so the
    Lane A / Lane C codepaths (no brotli) still work in a clean env. If
    the helper hard-failed without checking for files first, every
    non-brotli archive would also break."""
    pytest.importorskip("brotli")
    archive_dir = tmp_path / "archive"
    archive_dir.mkdir()
    (archive_dir / "renderer.bin").write_bytes(b"placeholder-renderer")

    src = INFLATE_PY.read_text()
    helper_src_match = re.search(
        r'def _decompress_brotli_in_archive\(.*?(?=\n\n# ===|\nclass |\ndef \w)',
        src, flags=re.DOTALL,
    )
    assert helper_src_match is not None
    helper_src = "import sys\nfrom pathlib import Path\n" + helper_src_match.group(0)
    ns: dict = {}
    exec(compile(helper_src, str(INFLATE_PY), "exec"), ns)
    helper = ns["_decompress_brotli_in_archive"]
    helper(str(archive_dir))  # must not raise
    # Original bytes untouched.
    assert (archive_dir / "renderer.bin").read_bytes() == b"placeholder-renderer"


def test_inflate_renderer_brotli_helper_lists_missing_files(tmp_path: Path) -> None:
    """Codex R5-2 #3 sub-fix: when brotli is genuinely missing from the
    env, the FATAL message must list the offending .br files so the
    operator can immediately see which artifact triggered the failure
    (rather than a generic 'install brotli'). This is the actionable-
    diagnostic improvement the codex recommendation called for.
    """
    src = INFLATE_PY.read_text()
    # Pin the structure of the FATAL block — we want both the listing
    # construction AND a hint about the project dep.
    assert "Files needing decompression" in src, (
        "FATAL brotli-missing message must enumerate the .br files that "
        "triggered the import — operators can't tell what to fix without it."
    )
    assert "pip install brotli" in src or "uv pip install brotli" in src, (
        "FATAL message must include an actionable install command."
    )
    assert "[project].dependencies" in src or "uv sync" in src, (
        "FATAL message must point at the canonical fix (pyproject dep / "
        "uv sync) rather than generic pip-installation instructions."
    )
