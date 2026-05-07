#!/usr/bin/env python3
"""Build a deterministic fixture for categorical candidate readiness tooling."""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path
from typing import Any

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.categorical_candidate_plan import (  # noqa: E402
    build_categorical_charged_label_plan,
    build_categorical_class_codebook,
)
from tac.categorical_candidate_readiness import (  # noqa: E402
    ARCHIVE_MEMBER_MANIFEST_CONTRACT,
    CANDIDATE_MANIFEST_CONTRACT,
    RUNTIME_LOADER_PARITY_CONTRACT,
    audit_categorical_candidate_manifest,
)
from tac.categorical_label_prior_payload_manifest import (  # noqa: E402
    LABEL_PRIOR_PAYLOAD_MANIFEST_CONTRACT,
    LABEL_PRIOR_PAYLOAD_MANIFEST_MEMBER,
    LABEL_PRIOR_PAYLOAD_MANIFEST_ROLE,
    build_categorical_label_prior_payload_manifest,
)
from tac.categorical_openpilot_mask_prior_contract import RUNTIME_LABEL_CONTRACT  # noqa: E402
from tac.repo_io import json_text, sha256_bytes, sha256_file, write_json  # noqa: E402
from tac.semantic_label_contract import CONTEST_SEGNET_CLASS_NAME_TUPLE, SELFCOMP_CLASS_TO_GRAY  # noqa: E402
from tac.tool_manifest import attach_tool_run_manifest  # noqa: E402

FIXED_ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
FIXED_ZIP_FILE_MODE = 0o644
FIXED_ZIP_INFLATE_MODE = 0o755
RUNTIME_CONSUMER_REPO_PATH = "src/tac/qma9_range_mask_contract.py"
MEMBER_ROLES = {
    "inflate.sh": "inflate_entrypoint",
    "categorical_payload.bin": "categorical_payload",
    "class_codebook.json": "decoder_table",
    LABEL_PRIOR_PAYLOAD_MANIFEST_MEMBER: LABEL_PRIOR_PAYLOAD_MANIFEST_ROLE,
    "runtime_decoder.py": "decoder_or_runtime_consumer",
}


def _conditioning_priors(
    *,
    class_codebook_sha256: str,
) -> list[dict[str, Any]]:
    return [
        {
            "family": "openpilot_priors",
            "name": "ego_lane_atom_ranker",
            "usage": "compression_time_atom_ranking_only",
            "runtime_consumed": False,
            "label_contract": "openpilot_prior_hints_non_runtime",
            "source_provenance": {
                "kind": "compression_time_only_derivation",
                "source": "fixture_openpilot_prior_hints",
                "runtime_consumed": False,
            },
        },
        {
            "family": "clade_spade",
            "name": "canonical_class_codebook_conditioning",
            "usage": "inflate_runtime_conditioning",
            "runtime_consumed": True,
            "charged_member": "class_codebook.json",
            "charged_member_sha256": class_codebook_sha256,
            "label_contract": RUNTIME_LABEL_CONTRACT,
            "source_provenance": {
                "kind": "charged_archive_member",
                "charged_member": "class_codebook.json",
                "sha256": class_codebook_sha256,
                "source": "fixture_class_codebook",
            },
        },
    ]


def _member_payloads(*, source_archive_sha256: str) -> tuple[dict[str, bytes], list[dict[str, Any]]]:
    member_payloads = {
        "inflate.sh": (
            b"#!/usr/bin/env bash\nset -euo pipefail\necho 'fixture-only categorical candidate' >&2\nexit 2\n"
        ),
        "categorical_payload.bin": b"QMA9_fixture_payload_not_a_score_candidate\n",
        "class_codebook.json": json_text(build_categorical_class_codebook()).encode("utf-8"),
        "runtime_decoder.py": (REPO_ROOT / RUNTIME_CONSUMER_REPO_PATH).read_bytes(),
    }
    conditioning_priors = _conditioning_priors(
        class_codebook_sha256=sha256_bytes(member_payloads["class_codebook.json"]),
    )
    member_payloads[LABEL_PRIOR_PAYLOAD_MANIFEST_MEMBER] = json_text(
        build_categorical_label_prior_payload_manifest(
            source_archive_sha256=source_archive_sha256,
            payload_member="categorical_payload.bin",
            payload_member_sha256=sha256_bytes(member_payloads["categorical_payload.bin"]),
            class_codebook_member="class_codebook.json",
            class_codebook_sha256=sha256_bytes(member_payloads["class_codebook.json"]),
            conditioning_priors=conditioning_priors,
        )
    ).encode("utf-8")
    return member_payloads, conditioning_priors


def _zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, date_time=FIXED_ZIP_TIMESTAMP)
    info.compress_type = zipfile.ZIP_STORED
    mode = FIXED_ZIP_INFLATE_MODE if name == "inflate.sh" else FIXED_ZIP_FILE_MODE
    info.external_attr = mode << 16
    info.create_system = 3
    return info


def _write_archive(path: Path, member_payloads: dict[str, bytes]) -> list[dict[str, Any]]:
    path.parent.mkdir(parents=True, exist_ok=True)
    records = []
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
        for name in sorted(member_payloads):
            raw = member_payloads[name]
            archive.writestr(_zip_info(name), raw, compress_type=zipfile.ZIP_STORED)
            records.append(
                {
                    "name": name,
                    "role": MEMBER_ROLES[name],
                    "bytes": len(raw),
                    "sha256": sha256_bytes(raw),
                }
            )
    return records


def build_fixture(*, out_dir: Path, source_archive_sha256: str) -> dict[str, Any]:
    archive_path = out_dir / "archive.zip"
    member_payloads, conditioning_priors = _member_payloads(
        source_archive_sha256=source_archive_sha256,
    )
    member_records = _write_archive(archive_path, member_payloads)
    runtime_source_path = REPO_ROOT / RUNTIME_CONSUMER_REPO_PATH
    runtime_source_sha = sha256_file(runtime_source_path)
    runtime_loader_sha = sha256_bytes(member_payloads["runtime_decoder.py"])
    archive_member_manifest = {
        "schema_version": 1,
        "kind": "categorical_fixture_archive_member_manifest",
        "archive_member_manifest_contract": ARCHIVE_MEMBER_MANIFEST_CONTRACT,
        "fixture_only": True,
        "member_count": len(member_records),
        "member_order": [record["name"] for record in member_records],
        "members": member_records,
    }
    archive_member_manifest_path = out_dir / "archive_member_manifest.json"
    write_json(archive_member_manifest_path, archive_member_manifest)
    archive_member_manifest_sha = sha256_bytes(json_text(archive_member_manifest).encode("utf-8"))
    candidate_archive_sha = sha256_file(archive_path)
    construction_plan = build_categorical_charged_label_plan(
        source_archive_sha256=source_archive_sha256,
        charged_members=member_records,
        conditioning_priors=conditioning_priors,
        candidate_archive_sha256=candidate_archive_sha,
        archive_member_manifest_sha256=archive_member_manifest_sha,
    )
    candidate = {
        "schema_version": 1,
        "kind": "categorical_candidate_fixture_manifest",
        "candidate_manifest_contract": CANDIDATE_MANIFEST_CONTRACT,
        "fixture_only": True,
        "score_claim": False,
        "dispatch_attempted": False,
        "ready_for_exact_eval_dispatch": False,
        "source_archive_sha256": source_archive_sha256,
        "archive_member_manifest_sha256": archive_member_manifest_sha,
        "archive_member_manifest": {
            "path": "archive_member_manifest.json",
            "bytes": archive_member_manifest_path.stat().st_size,
            "sha256": sha256_file(archive_member_manifest_path),
        },
        "candidate_archive_contract": "contest_archive_zip",
        "candidate_archive": {
            "path": "archive.zip",
            "bytes": archive_path.stat().st_size,
            "sha256": candidate_archive_sha,
        },
        "semantic_class_order": list(CONTEST_SEGNET_CLASS_NAME_TUPLE),
        "selfcomp_gray_codebook": [SELFCOMP_CLASS_TO_GRAY[index] for index in range(len(SELFCOMP_CLASS_TO_GRAY))],
        "runtime_consumer": {
            "path": RUNTIME_CONSUMER_REPO_PATH,
            "consumes_charged_members": True,
        },
        "label_prior_payload_manifest": {
            "member": LABEL_PRIOR_PAYLOAD_MANIFEST_MEMBER,
            "bytes": len(member_payloads[LABEL_PRIOR_PAYLOAD_MANIFEST_MEMBER]),
            "sha256": sha256_bytes(member_payloads[LABEL_PRIOR_PAYLOAD_MANIFEST_MEMBER]),
            "contract": LABEL_PRIOR_PAYLOAD_MANIFEST_CONTRACT,
        },
        "runtime_loader_parity": {
            "schema_version": 1,
            "runtime_loader_parity_contract": RUNTIME_LOADER_PARITY_CONTRACT,
            "passed": True,
            "score_claim": False,
            "dispatch_attempted": False,
            "runtime_consumer_path": RUNTIME_CONSUMER_REPO_PATH,
            "runtime_consumer_sha256": runtime_source_sha,
            "loader_member": "runtime_decoder.py",
            "loader_member_sha256": runtime_loader_sha,
            "byte_identical_to_runtime_consumer": True,
            "sidecar_free": True,
            "fallback_used": False,
            "loaded_charged_members": [
                "categorical_payload.bin",
                "class_codebook.json",
                LABEL_PRIOR_PAYLOAD_MANIFEST_MEMBER,
            ],
        },
        "candidate_construction_plan": construction_plan,
        "conditioning_priors": construction_plan["conditioning_priors"],
        "charged_members": member_records,
        "no_op_controls": {
            "decode_reencode_identity_control": {
                "passed": True,
                "scope": "fixture_manifest_only",
            },
            "label_permutation_fail_closed_control": {
                "passed": True,
                "scope": "fixture_manifest_only",
            },
            "charged_member_presence_control": {
                "passed": True,
                "scope": "fixture_manifest_only",
            },
            "runtime_consumes_conditioning_control": {
                "passed": True,
                "scope": "fixture_manifest_only",
            },
        },
    }
    readiness = audit_categorical_candidate_manifest(
        candidate,
        repo_root=REPO_ROOT,
        manifest_dir=out_dir,
    )
    return {
        "schema_version": 1,
        "kind": "categorical_candidate_fixture_build",
        "fixture_only": True,
        "score_claim": False,
        "dispatch_attempted": False,
        "ready_for_exact_eval_dispatch": False,
        "archive_member_manifest": archive_member_manifest,
        "candidate_manifest": candidate,
        "readiness": readiness,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--source-archive-sha256", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(raw_argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    payload = build_fixture(
        out_dir=args.out_dir,
        source_archive_sha256=args.source_archive_sha256,
    )
    archive_member_manifest_path = args.out_dir / "archive_member_manifest.json"
    construction_plan_path = args.out_dir / "construction_plan.json"
    candidate_path = args.out_dir / "candidate.json"
    readiness_path = args.out_dir / "readiness.json"
    summary_path = args.out_dir / "summary.json"
    write_json(archive_member_manifest_path, payload["archive_member_manifest"])
    write_json(construction_plan_path, payload["candidate_manifest"]["candidate_construction_plan"])
    write_json(candidate_path, payload["candidate_manifest"])
    write_json(readiness_path, payload["readiness"])
    summary = {
        key: value
        for key, value in payload.items()
        if key not in {"archive_member_manifest", "candidate_manifest", "readiness"}
    }
    summary["paths"] = {
        "archive": str(args.out_dir / "archive.zip"),
        "archive_member_manifest": str(archive_member_manifest_path),
        "construction_plan": str(construction_plan_path),
        "candidate": str(candidate_path),
        "readiness": str(readiness_path),
    }
    summary["archive_sha256"] = payload["candidate_manifest"]["candidate_archive"]["sha256"]
    summary["readiness_blockers"] = payload["readiness"]["dispatch_blockers"]
    summary = attach_tool_run_manifest(
        summary,
        tool=Path(__file__).relative_to(REPO_ROOT).as_posix(),
        argv=raw_argv,
        input_paths=[],
        repo_root=REPO_ROOT,
        output_path=summary_path,
    )
    write_json(summary_path, summary)
    print(json_text(summary), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
