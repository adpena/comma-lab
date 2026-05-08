#!/usr/bin/env python3
"""Materialize a generated-schema HNeRV codec blob from a state dict."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tac.hnerv_generated_schema_codec import (  # noqa: E402
    decode_generated_schema_blob,
    encode_generated_schema_state_dict,
    manifest_without_header,
)

Schema = tuple[tuple[str, tuple[int, ...]], ...]


def _schema_from_json(path: Path) -> Schema:
    raw = json.loads(path.read_text(encoding="utf-8"))
    rows = raw.get("state_schema", raw)
    return tuple(
        (str(row["name"]), tuple(int(v) for v in row["shape"]))
        for row in rows
    )


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dict", required=True)
    parser.add_argument("--schema-json", required=True)
    parser.add_argument("--output-blob", required=True)
    parser.add_argument("--output-manifest", required=True)
    parser.add_argument("--brotli-quality", type=int, default=11)
    parser.add_argument("--brotli-lgwin", type=int)
    parser.add_argument("--brotli-lgblock", type=int)
    args = parser.parse_args(argv)

    state_dict = torch.load(args.state_dict, map_location="cpu", weights_only=True)
    schema = _schema_from_json(Path(args.schema_json))
    encoded = encode_generated_schema_state_dict(
        state_dict,
        schema=schema,
        brotli_quality=args.brotli_quality,
        brotli_lgwin=args.brotli_lgwin,
        brotli_lgblock=args.brotli_lgblock,
    )
    out_blob = Path(args.output_blob)
    out_blob.parent.mkdir(parents=True, exist_ok=True)
    out_blob.write_bytes(encoded.blob)

    _decoded, decode_manifest = decode_generated_schema_blob(encoded.blob)
    manifest = {
        **manifest_without_header(encoded),
        "tool": "tools/materialize_hnerv_generated_schema_codec.py",
        "score_claim": False,
        "ready_for_exact_eval_dispatch": False,
        "output_blob": str(out_blob),
        "source_state_dict": str(args.state_dict),
        "schema_json": str(args.schema_json),
        "decode_roundtrip_schema_fingerprint": decode_manifest["schema_fingerprint"],
        "decode_manifest": {
            key: value for key, value in decode_manifest.items() if key != "header"
        },
    }
    _write_json(Path(args.output_manifest), manifest)
    print(json.dumps({
        "blob": str(out_blob),
        "blob_bytes": manifest["blob_bytes"],
        "manifest": str(args.output_manifest),
        "ready_for_exact_eval_dispatch": False,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
