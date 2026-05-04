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
    # for the inflate_renderer.py invocation. Then assert it uses the pinned
    # renderer dependency array, which includes brotli plus av/torch/numpy.
    renderer_invocation = re.search(
        r'"\$UV_BIN"\s+run\s+([^\n]*?)python[^\n]*inflate_renderer\.py',
        text,
    )
    assert renderer_invocation is not None, (
        "Could not locate the inflate_renderer.py invocation in inflate.sh. "
        "Did the layout of the renderer codepath change?"
    )
    flags = renderer_invocation.group(1)
    assert "UV_WITH_RENDERER_DEPS" in flags, (
        "inflate.sh's PYTHON_INFLATE=renderer codepath does NOT pass "
        "the pinned renderer dependency specs to uv run. Captured flags: "
        f"{flags!r}\n"
        "Without the brotli spec a clean contest env will fatal-exit at "
        "_decompress_brotli_in_archive when it encounters renderer.bin.br. "
        "Use UV_WITH_RENDERER_DEPS for renderer inflate."
    )
    assert "INFLATE_BROTLI_SPEC" in text


def test_inflate_sh_auto_selects_grayscale_renderer_for_grayscale_only_archive() -> None:
    text = INFLATE_SH.read_text()
    assert "auto-selecting PYTHON_INFLATE=renderer_grayscale" in text
    assert '[ -f "$ARCHIVE_DIR/grayscale.mkv" ]' in text
    assert '[ ! -f "$ARCHIVE_DIR/masks.mkv" ]' in text


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


# ============================================================
# Codex R5-3 #1 (MEDIUM): brotli .br detection + decompression must
# fire BEFORE PYTHON_INFLATE branch dispatch, not only on the
# renderer arm. Below tests pin the centralized "Stage 0" step.
# ============================================================
def test_inflate_sh_has_centralized_brotli_stage_before_branch_dispatch() -> None:
    """The Stage 0 brotli decompression block in inflate.sh must appear
    BEFORE the `while IFS= read -r rel; do` loop that contains the
    PYTHON_INFLATE branch dispatch.

    Without this ordering, any non-renderer branch (PYTHON_INFLATE=1 /
    postfilter / grain_mask / future variants) that received a .br
    archive would fail later as a missing renderer.bin / masks.mkv
    rather than the actionable Stage 0 diagnostic.
    """
    text = INFLATE_SH.read_text()
    stage0_marker = text.find("Stage 0: brotli decompression")
    assert stage0_marker != -1, (
        "inflate.sh missing 'Stage 0: brotli decompression' centralized "
        "block; codex R5-3 fix regressed."
    )
    while_marker = text.find("while IFS= read -r rel; do")
    assert while_marker != -1, "inflate.sh missing the per-video while loop"
    assert stage0_marker < while_marker, (
        "Stage 0 brotli block must appear BEFORE the PYTHON_INFLATE branch "
        "dispatch (the while loop). Currently Stage 0 is at offset "
        f"{stage0_marker} and the while loop starts at {while_marker}."
    )


def test_inflate_sh_stage0_passes_with_brotli_to_uv_run() -> None:
    """Stage 0 invokes `uv run` to do the decompression; it MUST pass
    `--with brotli` so a clean contest env (no tac wheel installed) can
    still resolve the brotli wheel. Without this flag we'd fall back to
    whatever pyproject `uv run` happened to discover, which in a contest
    evaluator is the contest's pyproject — NOT ours.
    """
    text = INFLATE_SH.read_text()
    # Carve out the Stage 0 block (between the Stage 0 banner comment and
    # the next `upscale_rgb_base_filter()` definition).
    block_match = re.search(
        r"Stage 0: brotli decompression.*?upscale_rgb_base_filter\(\)",
        text, flags=re.DOTALL,
    )
    assert block_match is not None, "Stage 0 block bounds not found"
    block = block_match.group(0)
    assert "UV_WITH_BROTLI" in block and "INFLATE_BROTLI_SPEC" in text, (
        "Stage 0 must pass the pinned brotli dependency spec to `uv run`. "
        "The clean-contest-env guarantee depends on this."
    )


def test_inflate_sh_stage0_runs_for_all_branches_not_just_renderer(tmp_path: Path) -> None:
    """The Stage 0 decompression must NOT be guarded by PYTHON_INFLATE — it
    must execute regardless of which branch the dispatch later picks.

    We assert this via the source structure: the Stage 0 block must NOT
    appear inside any `if [ "$PYTHON_INFLATE" = "..." ]` branch — i.e. it
    is unconditional with respect to PYTHON_INFLATE.
    """
    text = INFLATE_SH.read_text()
    stage0_marker = text.find("Stage 0: brotli decompression")
    assert stage0_marker != -1
    # Find all PYTHON_INFLATE branch openings (the if/elif chain inside
    # the per-video loop). All of them must come AFTER Stage 0.
    branch_offsets = [
        m.start() for m in re.finditer(r'\$PYTHON_INFLATE"\s*=\s*"', text)
    ]
    assert branch_offsets, "PYTHON_INFLATE branch dispatch not found"
    for offset in branch_offsets:
        assert offset > stage0_marker, (
            "Stage 0 must precede every PYTHON_INFLATE branch check. Found "
            f"a branch at offset {offset} before Stage 0 at {stage0_marker}."
        )


def test_inflate_sh_stage0_skips_when_no_br_files(tmp_path: Path) -> None:
    """The Stage 0 block must be guarded so it is a true no-op when no
    .br files exist (the common Lane A path). We assert via the source
    structure: the block opens with a `compgen -G` (or equivalent) test
    on `*.br` and only runs the uv invocation when a match is found.
    """
    text = INFLATE_SH.read_text()
    block_match = re.search(
        r"Stage 0: brotli decompression.*?upscale_rgb_base_filter\(\)",
        text, flags=re.DOTALL,
    )
    assert block_match is not None
    block = block_match.group(0)
    assert "compgen -G" in block and "*.br" in block, (
        "Stage 0 must guard the uv run invocation behind a `compgen -G "
        "\"$ARCHIVE_DIR\"/*.br` (or equivalent) so non-brotli archives "
        "don't pay uv cold-start cost."
    )


def test_inflate_sh_stage0_inline_python_uses_brotli_directly(tmp_path: Path) -> None:
    """Stage 0's inline python should NOT depend on `tac.submission_archive`
    — in a clean contest env `tac` may not be installed at all. The
    `--with brotli` flag only guarantees `brotli` is importable. The Stage 0
    inline must use `import brotli` directly, not `from tac.submission_archive
    import decompress_brotli_files_in_dir`.
    """
    text = INFLATE_SH.read_text()
    block_match = re.search(
        r"Stage 0: brotli decompression.*?upscale_rgb_base_filter\(\)",
        text, flags=re.DOTALL,
    )
    assert block_match is not None
    block = block_match.group(0)
    assert "import brotli" in block, (
        "Stage 0 inline must `import brotli` directly. Without this it "
        "depends on `tac` being installed in the uv-resolved env, which "
        "breaks the clean-contest-env guarantee."
    )
    assert "from tac.submission_archive" not in block, (
        "Stage 0 inline must NOT depend on `tac.submission_archive` — "
        "that import is unsafe in a contest env that resolves the "
        "contest's pyproject (no tac installed)."
    )


def test_inflate_sh_renderer_branch_still_calls_inline_brotli_helper() -> None:
    """Defense-in-depth: even after the Stage 0 centralization, the renderer
    branch's inline `_decompress_brotli_in_archive(archive_dir)` call MUST
    remain so direct python invocation paths (unit tests, local debugging
    that bypasses inflate.sh) still work. The helper is a no-op when no .br
    files are present, so this is cheap.
    """
    text = INFLATE_PY.read_text()
    # There are TWO call sites historically (inflate_renderer + inflate_renderer_with_tto).
    call_count = text.count("_decompress_brotli_in_archive(archive_dir)")
    assert call_count >= 2, (
        f"Expected the inline brotli helper to be called at >=2 sites in "
        f"inflate_renderer.py (defense-in-depth for direct python use); "
        f"found {call_count}. If you removed a call site you also broke "
        f"the non-inflate.sh codepaths."
    )
