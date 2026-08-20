#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Materialize one typed G52 codec arm through the strict V10 receiver."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tac.admission_guard import assert_governed_admission  # noqa: E402
from tac.witness_dsl.taskspace_lossy_selected_plane_codec_v1 import (  # noqa: E402
    ARMS,
    LossySelectedPlaneCodecError,
    load_config,
    materialize_v10_scoring_raw,
    sha256_file,
)

BRIDGE_CONFIG_SCHEMA = "taskspace_lossy_selected_plane_raw_bridge_config.v1"
_FIELDS = frozenset(
    {
        "schema",
        "research_only",
        "candidate_lineage_allowed",
        "score_claim",
        "codec_config_path",
        "codec_config_file_sha256",
        "codec_output_root",
        "codec_aggregate_sha256",
        "endpoint",
        "arm",
        "output_root",
        "pair_count",
    }
)


def _load_bridge_config(path: Path) -> dict:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as exc:
        raise LossySelectedPlaneCodecError(f"cannot load raw-bridge config: {path}") from exc
    if not isinstance(value, dict) or frozenset(value) != _FIELDS:
        raise LossySelectedPlaneCodecError("raw-bridge config has missing or unknown fields")
    if (
        value["schema"] != BRIDGE_CONFIG_SCHEMA
        or value["research_only"] is not True
        or value["candidate_lineage_allowed"] is not False
        or value["score_claim"] is not False
        or value["pair_count"] != 600
        or value["arm"] not in ARMS
    ):
        raise LossySelectedPlaneCodecError("raw-bridge authority/type contract drift")
    codec_config_path = Path(value["codec_config_path"])
    if sha256_file(codec_config_path) != value["codec_config_file_sha256"]:
        raise LossySelectedPlaneCodecError("raw-bridge codec config file custody drift")
    aggregate_path = Path(value["codec_output_root"]) / "aggregate_receipt.json"
    if sha256_file(aggregate_path) != value["codec_aggregate_sha256"]:
        raise LossySelectedPlaneCodecError("raw-bridge codec aggregate custody drift")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    assert_governed_admission("taskspace_lossy_codec_raw_bridge_n600")
    try:
        bridge_config = _load_bridge_config(args.config)
        codec_config = load_config(bridge_config["codec_config_path"])
        receipt = materialize_v10_scoring_raw(
            codec_config,
            codec_output_root=bridge_config["codec_output_root"],
            endpoint_name=bridge_config["endpoint"],
            arm=bridge_config["arm"],
            output_root=bridge_config["output_root"],
        )
    except LossySelectedPlaneCodecError as exc:
        print(json.dumps({"status": "refused", "error": str(exc)}, sort_keys=True), file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "arm": receipt["arm"],
                "diagnostic_bundle_bytes": receipt["diagnostic_bundle_bytes"],
                "raw_path": receipt["raw_path"],
                "raw_sha256": receipt["raw_sha256"],
                "score_claim": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
