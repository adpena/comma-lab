# SPDX-License-Identifier: MIT
"""Compile post-charter DDM MENU1 rows from SHA-pinned measured receipts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from tac.optimization.ddm_realized_flip_menu import compile_postcharter_addendum


class PostcharterConfig(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        populate_by_name=True,
    )

    schema_id: Literal["DDMMenu1PostcharterConfigV1"] = Field(alias="schema")
    run_id: str
    menu1_receipt_path: str
    menu1_receipt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    mc1_receipt_path: str
    mc1_receipt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    ws1_receipt_path: str
    ws1_receipt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    rd1_receipt_path: str
    rd1_receipt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    e4_receipt_path: str
    e4_receipt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    execution_allowed: bool
    research_only: bool
    score_claim: bool

    def stable_hash(self) -> str:
        payload = json.dumps(
            self.model_dump(mode="json", by_alias=True),
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        return hashlib.sha256(payload).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path_text: str, expected_sha256: str, label: str) -> dict[str, Any]:
    path = Path(path_text)
    observed = _sha256(path)
    if observed != expected_sha256:
        raise ValueError(
            f"{label} SHA-256 differs: expected {expected_sha256}, observed {observed}"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def run(config_path: Path, output_path: Path) -> Path:
    config = PostcharterConfig.model_validate_json(config_path.read_bytes())
    if (
        config.execution_allowed is not True
        or config.research_only is not True
        or config.score_claim is not False
    ):
        raise ValueError("post-charter compiler authority must remain local research-only")
    paths = {
        "menu1": (config.menu1_receipt_path, config.menu1_receipt_sha256),
        "mc1": (config.mc1_receipt_path, config.mc1_receipt_sha256),
        "ws1": (config.ws1_receipt_path, config.ws1_receipt_sha256),
        "rd1": (config.rd1_receipt_path, config.rd1_receipt_sha256),
        "e4": (config.e4_receipt_path, config.e4_receipt_sha256),
    }
    receipts = {
        key: _load(path, digest, key.upper()) for key, (path, digest) in paths.items()
    }
    compiled = compile_postcharter_addendum(
        **receipts,
        input_sha256={key: digest for key, (_path, digest) in paths.items()},
    )
    compiled["run_id"] = config.run_id
    compiled["typed_config_path"] = config_path.as_posix()
    compiled["typed_config_sha256"] = config.stable_hash()
    compiled["triality"] = {
        "dag_feed": (
            ".omx/research/"
            "ddm_menu1_postcharter_addendum_DAG_FEED_20260724.md"
        ),
        "canonical_equations": (
            ".omx/research/"
            "ddm_menu1_postcharter_addendum_canonical_equations_20260724.md"
        ),
        "dsl_data": output_path.as_posix(),
    }
    payload = json.dumps(
        compiled,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ) + "\n"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(payload, encoding="utf-8")
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(run(args.config, args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
