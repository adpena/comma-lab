"""SAFE STUB: experiments/repack_quantizr_faithful_qzs3_archive.py

Recovery note: this module was lost when subagent worktrees were
auto-cleaned without committing source. Only the .pyc cache survives at
experiments/__pycache__/repack_quantizr_faithful_qzs3_archive.cpython-312.pyc.
Never in git history.

Imported by src/tac/tests/test_qzs3_packer.py:
    from experiments.repack_quantizr_faithful_qzs3_archive import (
        RENDERER_CODEC_QZS3,
        RENDERER_CODEC_QZS4,
        build_archive,
        build_submission_archive,
    )

Per CLAUDE.md "no signal loss ever" mandate: this stub satisfies the import
contract (tests no longer ImportError at collection) but raises
NotImplementedError on any actual call so we don't silently mis-pack
contest archives.

The original module's job (per memory notes + test docstrings):
- Take a JointFrameGenerator state dict
- Encode via tac.quantizr_qzs3_codec.encode_qzs3_state_dict (this codec
  exists at src/tac/quantizr_qzs3_codec.py)
- Pack into a contest archive following the C067/QZS3/QP1 format used by
  PR #67 — produce build_archive() (compressed) and
  build_submission_archive() (full submission ZIP)

Recovery options:
  1. Restore from a Modal/Lightning/Vast.ai backup
  2. Rebuild from src/tac/quantizr_qzs3_codec.py + the C067 source archive
     bytes layout (see experiments/build_sjkl_c067_archive.py for a
     parallel pattern that does the sibling-layout safe path)
  3. Read the C067 PR submission's own packer

Until then, the test suite collects + imports cleanly, but any test that
actually exercises this module's behavior will fail-loud rather than
silently producing wrong contest-archive bytes.
"""
from __future__ import annotations


# Codec ID constants — matched to the runtime contract in
# src/tac/quantizr_qzs3_codec.py + submissions/robust_current/inflate_renderer.py
# (these constants are the documented runtime IDs; safe to define here as
# they're just enum values, no behavioral risk).
RENDERER_CODEC_QZS3 = 0x33  # 'Q' high-nibble + version 3
RENDERER_CODEC_QZS4 = 0x34  # 'Q' high-nibble + version 4 (block-search variant)


_STUB_RECOVERY_MESSAGE = (
    "experiments/repack_quantizr_faithful_qzs3_archive.py is a recovery stub: "
    "the original module was lost when subagent worktrees were auto-cleaned "
    "without committing source. Only the .pyc cache survives at "
    "experiments/__pycache__/. The C067/QZS3/QP1 archive packer cannot be "
    "rebuilt safely without the byte-layout spec. Recovery options: "
    "(1) restore from a Modal/Lightning/Vast.ai backup, "
    "(2) rebuild from src/tac/quantizr_qzs3_codec.py + the C067 source "
    "archive bytes (see experiments/build_sjkl_c067_archive.py for a "
    "parallel sibling-layout pattern), or (3) read the C067 PR submission's "
    "own packer source. Until then, this stub fails loud rather than "
    "silently mis-packing contest archives."
)


def build_archive(*args, **kwargs):
    """Pack a JointFrameGenerator state dict into a C067/QZS3 archive.

    Original signature lost. Raises NotImplementedError with recovery pointer.
    """
    raise NotImplementedError(_STUB_RECOVERY_MESSAGE)


def build_submission_archive(*args, **kwargs):
    """Pack into the full submission archive layout.

    Original signature lost. Raises NotImplementedError with recovery pointer.
    """
    raise NotImplementedError(_STUB_RECOVERY_MESSAGE)


__all__ = [
    "RENDERER_CODEC_QZS3",
    "RENDERER_CODEC_QZS4",
    "build_archive",
    "build_submission_archive",
]
