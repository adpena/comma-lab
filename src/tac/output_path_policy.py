# SPDX-License-Identifier: MIT
"""Shared output-path policy for durable experiment artifacts.

Tools and trainers may use temporary scratch internally, but any path that is
advertised with ``no_tmp_paths`` or persisted as a result artifact must refuse
OS temporary roots before writing. Keep this helper tiny and dependency-free so
CLI tools can import it without pulling scorer/training stacks.
"""

from __future__ import annotations

from pathlib import Path

_TEMPORARY_OUTPUT_ROOTS = (
    Path("/tmp"),
    Path("/var/tmp"),
    Path("/private/tmp"),
)


def assert_not_temporary_output_dir(
    output_dir: Path | str,
    *,
    tool_name: str = "tool",
) -> Path:
    """Return resolved ``output_dir`` or raise ``ValueError`` for temp roots."""

    resolved = Path(output_dir).expanduser().resolve(strict=False)
    for root in _TEMPORARY_OUTPUT_ROOTS:
        root_resolved = root.resolve(strict=False)
        if resolved == root_resolved or root_resolved in resolved.parents:
            raise ValueError(
                f"[{tool_name}] refusing output directory under temporary "
                f"path {resolved}; no_tmp_paths compliance requires a durable "
                "output directory"
            )
    return resolved


__all__ = ["assert_not_temporary_output_dir"]
