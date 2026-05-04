"""Inverse parser for the C067-style packed renderer payload (RPK1 family).

Recovery note: this module was lost when subagent worktrees were auto-cleaned
without committing source to git. Rebuilt 2026-05-04 as a SAFE STUB:

  - The API contract that src/tac/submission_archive.py expects is implemented
    (_parse_payload returns (header_dict, members_dict))
  - On real RPK1 payloads, it raises NotImplementedError with a clear recovery
    pointer rather than silently mis-decoding
  - Existing C067 archives will fail loud at validation time (caller catches
    and records the error) instead of being corrupted

Why a stub instead of a guess: The RPK1 binary layout was not documented in
either of the two surviving spec files (sjkl_c067_remote_dispatch_runbook +
sjkl_c067_shrink_addendum). Reverse-engineering it from the C067 source's `p`
container header bytes (5b 98 68 43 ...) without source is a silent-corruption
risk on archives this repo has already validated. Per CLAUDE.md "no signal
loss ever" mandate: better to fail loud than to silently produce wrong member
extractions.

Recovery path:
  1. Find the original `unpack_renderer_payload.py` in any local backup
     (Modal harvest, Lightning workspace, Vast.ai instance backups)
  2. OR: rebuild from `experiments/build_sjkl_c067_archive.py` (also missing,
     but its packer logic is the inverse — once recovered, RPK1 byte format
     is fully recoverable)
  3. OR: read the C067 PR submission's own decoder code (the format originated
     there)

The expected logical members per the SJ-KL runbook + addendum are:
  - renderer.bin    (JointFrameGenerator state, FP4/QZS3/MQZ1/QBF1)
  - masks.mkv       (compressed semantic-mask video)
  - optimized_poses.bin  (per-frame ego-motion poses)
  - sjkl.bin        (optional: Score-Jacobian Karhunen-Loève residual basis)

The expected header dict at minimum should include:
  - version: int
  - layout: str ("min_rpk1", "top_level_sibling", etc.)
  - member_order: list[str] (the deterministic Brotli-screen-optimal order)
"""
from __future__ import annotations


_STUB_RECOVERY_MESSAGE = (
    "RPK1 payload parser is a recovery stub: the original "
    "submissions/robust_current/unpack_renderer_payload.py was lost when "
    "subagent worktrees were auto-cleaned without committing source. The "
    "byte layout is not documented in surviving spec files. Recovery options: "
    "(1) restore from a Modal/Lightning/Vast.ai backup, "
    "(2) rebuild after experiments/build_sjkl_c067_archive.py is recovered "
    "(its packer is the inverse of this parser), "
    "(3) read the C067 PR submission's own decoder source. "
    "Until then, this stub fails loud rather than silently mis-decoding."
)


def _parse_payload(payload: bytes) -> tuple[dict, dict[str, bytes]]:
    """Parse a packed renderer payload into (header_dict, members_dict).

    Args:
        payload: raw bytes of the packed payload (typically the contents of
                 the single-member ZIP entry named `p`, `renderer_payload.bin`,
                 or `renderer_payload.bin.br` after Brotli decompression).

    Returns:
        (header, members) where header is a metadata dict and members is a
        dict mapping logical member name (e.g. "renderer.bin", "masks.mkv",
        "optimized_poses.bin", "sjkl.bin") to its raw bytes.

    Raises:
        NotImplementedError: with a detailed recovery message. The caller in
            src/tac/submission_archive.py catches Exception and records an
            error rather than crashing, so existing validators continue to
            function — they just report this stub's error in the errors list.
    """
    if not isinstance(payload, (bytes, bytearray)):
        raise TypeError(f"payload must be bytes, got {type(payload).__name__}")
    if len(payload) == 0:
        return ({"version": 0, "layout": "empty", "member_order": []}, {})
    raise NotImplementedError(_STUB_RECOVERY_MESSAGE)


def parse_payload(payload: bytes) -> tuple[dict, dict[str, bytes]]:
    """Public alias of _parse_payload (some callers may use the public name)."""
    return _parse_payload(payload)


__all__ = ["_parse_payload", "parse_payload"]
