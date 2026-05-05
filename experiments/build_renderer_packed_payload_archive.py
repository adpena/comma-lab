"""SAFE STUB: experiments/build_renderer_packed_payload_archive.py

Recovery note: this module was lost when subagent worktrees were
auto-cleaned without committing source. Only the .pyc cache survives at
experiments/__pycache__/build_renderer_packed_payload_archive.cpython-312.pyc.
Never in git history.

Imported by src/tac/tests/test_qzs3_packer.py:
    from experiments.build_renderer_packed_payload_archive import (
        PAYLOAD_FORMAT_PR64_LEN_TABLE,
        POSE_QP1_CODEC,
        build_packed_archive,
    )

Per CLAUDE.md "no signal loss ever" mandate: this stub satisfies the import
contract (test_qzs3_packer.py collects cleanly) but raises
NotImplementedError on any actual call so we don't silently mis-pack
contest archives.

Original module's job (per import + neighbouring test context):
- Take a JointFrameGenerator state dict + pose array
- Pack into a "renderer payload" archive following PR #64's length-table
  format (pose codec QP1 + variable-length payload sections)
- Used as a building block for QZS4 block-search variants of the QZS3
  archive lineage

Recovery options (same as sibling lost packers):
  1. Restore from a Modal/Lightning/Vast.ai backup
  2. Rebuild from src/tac/quantizr_qzs3_codec.py + the PR #64 reference
     archive bytes (the constants below are runtime-public and known
     from inflate_renderer.py — but the build_packed_archive byte
     layout is undocumented)
  3. Read PR #64's own packer source

Until then, the test suite collects + imports cleanly, but any test that
actually exercises this module will fail-loud rather than silently
producing wrong contest-archive bytes.
"""
from __future__ import annotations


# Pose codec identifier — runtime public constant from PR #67 lineage.
# Safe to define here as it's just an enum value (no behavioral risk):
# inflate_renderer.py uses POSE_QP1_CODEC to dispatch the QP1 pose decoder.
POSE_QP1_CODEC = "QP1"


# PR #64 length-table format header (variable-length payload section
# lengths, in order). The exact constant table is documented as having
# 7 entries per PR #64's inflate.py + the bucket-lookup in QZS3 packer.
# Values restored as a placeholder — operators that depend on the EXACT
# table values must rebuild from the PR #64 reference (the table is
# version-locked to specific archive byte counts, so wrong values will
# produce wrong-bucket archives).
PAYLOAD_FORMAT_PR64_LEN_TABLE: tuple[int, ...] = (
    # 7-bucket lookup of compressed model.pt.br sizes — see
    # reference_pr65_pr67_blob_byte_layouts_proper_reverse_engineering_20260501.md
    # for the runtime constraint that builds must hit one of these buckets.
    # Original values lost; placeholders below would produce wrong-bucket
    # output if the build function were callable. Build function fails loud.
    0, 0, 0, 0, 0, 0, 0,
)


_STUB_RECOVERY_MESSAGE = (
    "experiments/build_renderer_packed_payload_archive.py is a recovery stub: "
    "the original module was lost when subagent worktrees were auto-cleaned "
    "without committing source. Only the .pyc cache survives at "
    "experiments/__pycache__/. The PR #64 length-table format byte layout is "
    "undocumented in surviving spec files; rebuilding without spec carries "
    "silent-corruption risk on archives this repo may have already validated. "
    "Recovery options: "
    "(1) restore from a Modal/Lightning/Vast.ai backup, "
    "(2) rebuild from src/tac/quantizr_qzs3_codec.py + PR #64 reference "
    "archive bytes (note: PAYLOAD_FORMAT_PR64_LEN_TABLE values must come "
    "from a real reference build — placeholders here are zeros), "
    "(3) read PR #64's own packer source. "
    "Until then, this stub fails loud rather than silently producing "
    "wrong-bucket contest archives."
)


def build_packed_archive(*args, **kwargs):
    """Build a PR #64-format packed renderer payload archive.

    Original signature lost. Raises NotImplementedError with recovery pointer.
    """
    raise NotImplementedError(_STUB_RECOVERY_MESSAGE)


__all__ = [
    "PAYLOAD_FORMAT_PR64_LEN_TABLE",
    "POSE_QP1_CODEC",
    "build_packed_archive",
]
