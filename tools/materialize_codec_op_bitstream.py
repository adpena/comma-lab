#!/usr/bin/env python3
"""Materialize a deterministic CodecOp bitstream golden vector.

The input is a CodecOp encode-result/manifest-like JSON object. If that JSON
only carries payload custody fields, pass ``--payload`` with the actual bytes.
The output is byte-custody/planning evidence only; it is not a score claim or
an exact-eval dispatch packet.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.codec_op_bitstream_materializer import (  # noqa: E402
    CodecOpBitstreamMaterializerError,
    materialize_codec_op_bitstream,
)


def _read_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise CodecOpBitstreamMaterializerError(f"input JSON must be an object: {path}")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-json", type=Path, required=True)
    parser.add_argument(
        "--payload",
        type=Path,
        default=None,
        help="Actual CodecOp payload bytes when --input-json only has custody metadata.",
    )
    parser.add_argument("--output-blob", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    parser.add_argument("--candidate-id", default=None)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite output paths if they already exist.",
    )
    args = parser.parse_args(argv)

    try:
        source = _read_json_object(args.input_json)
        manifest = materialize_codec_op_bitstream(
            source,
            output_blob=args.output_blob,
            manifest_output=args.manifest_output,
            source_manifest_path=args.input_json,
            payload_path=args.payload,
            candidate_id=args.candidate_id,
            force=args.force,
        )
    except (CodecOpBitstreamMaterializerError, OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"codec-op bitstream materialization failed: {exc}") from None

    blob = manifest["charged_byte_blob"]
    print(
        f"wrote {blob['path']} "
        f"({blob['bytes']} bytes, sha256={blob['sha256']})"
    )
    print(
        f"manifest {args.manifest_output} "
        f"ready_for_exact_eval_dispatch={manifest['ready_for_exact_eval_dispatch']} "
        f"blockers={len(manifest['blockers'])}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
