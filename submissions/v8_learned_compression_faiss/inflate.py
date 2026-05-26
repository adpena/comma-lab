#!/usr/bin/env python3
"""V8 learned-compression Faiss local inflate runtime.

This runtime implements the minimal byte-closed V8 v1 raw-frame fixture
grammar. It remains explicitly research-only and non-promotable until the
learned encoder/export/training path lands and exact eval custody exists.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))  # SUBMISSION_PYTHONPATH_SHIM_OK:research_only_v1_raw_frame_fixture_scaffold_per_inflate_docstring_explicitly_non_promotable_until_learned_encoder_export_training_path_lands_and_exact_eval_custody_exists_no_contest_dispatch_enabled_for_this_substrate

from tac.substrates.v8_learned_compression_faiss.archive import (  # noqa: E402
    V8_MAGIC as MAGIC,
)
from tac.substrates.v8_learned_compression_faiss.archive import (  # noqa: E402
    decode_raw_frame_archive,
    parse_v8_header,
)


def select_inflate_device() -> str:
    """Honor ``PACT_INFLATE_DEVICE`` (auto/cpu/cuda); MPS is forbidden.

    The v1 fixture decoder is pure byte copying after validation, so the
    selected device is observability metadata only and never loads torch.
    """
    value = (os.environ.get("PACT_INFLATE_DEVICE") or "auto").strip().lower()
    if value == "auto":
        return "cpu"
    if value in {"cpu", "cuda"}:
        return value
    raise RuntimeError(f"unsupported PACT_INFLATE_DEVICE={value!r}; expected auto/cpu/cuda")


def parse_v8_archive_header(raw: bytes) -> dict[str, object]:
    return parse_v8_header(raw).as_dict()


def inflate(src_bin: str | Path, dst_raw: str | Path) -> None:
    raw = Path(src_bin).read_bytes()
    _device = select_inflate_device()
    payload = decode_raw_frame_archive(raw)
    dst = Path(dst_raw)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(f"{dst.name}.tmp")
    try:
        tmp.write_bytes(payload)
        os.replace(tmp, dst)
    finally:
        if tmp.exists():
            tmp.unlink()


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 2:
        sys.stderr.write("Usage: python inflate.py <src.bin> <dst.raw>\n")
        return 2
    inflate(args[0], args[1])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
