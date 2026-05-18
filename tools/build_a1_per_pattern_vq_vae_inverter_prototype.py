#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build an A1-specialized deterministic packet compiler prototype.

This is a thin operator CLI around ``tac.contest_exploits``.  It deliberately
produces a non-promotable prototype artifact: no paid dispatch, no score claim,
and no exact-eval readiness marker.
"""

from __future__ import annotations

import argparse
import json
import stat
import zipfile
from collections.abc import Sequence
from pathlib import Path

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.contest_exploits.a1_specialized_inverter import (  # noqa: E402
    A1SpecializedInverterConfig,
    build_a1_specialized_inverter,
    derive_proxy_features_from_archive_bytes,
    load_feature_matrix,
    sha256_bytes,
    write_a1_specialized_artifact,
)
from tac.packet_compiler.deterministic_compiler import (  # noqa: E402
    DETERMINISTIC_ZIP_DATE_TIME,
    MANIFEST_NAME,
    NON_EXECUTABLE_MODE,
    compile_packet,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--features",
        type=Path,
        help="Path to a .npy/.npz 2-D precomputed feature/pattern matrix.",
    )
    source.add_argument(
        "--archive-zip",
        type=Path,
        help="Archive bytes used only for explicit proxy-feature smoke builds.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--codebook-size", type=int, default=256)
    parser.add_argument("--sparsity", type=float, default=0.50)
    parser.add_argument("--kmeans-iterations", type=int, default=8)
    parser.add_argument("--brotli-quality", type=int, default=11)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--proxy-feature-dim", type=int, default=64)
    parser.add_argument("--proxy-max-patterns", type=int, default=600)
    parser.add_argument(
        "--skip-canonical-packet-compiler",
        action="store_true",
        help=(
            "Only write the raw prototype blob/report. Default also wraps the blob in a "
            "contest-shaped packet and runs the canonical deterministic packet compiler in identity mode."
        ),
    )
    parser.add_argument(
        "--allow-proxy-from-archive-bytes",
        action="store_true",
        help=(
            "Required with --archive-zip. Builds a smoke-test proxy matrix from raw archive bytes; "
            "this is not scorer-feature evidence."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = A1SpecializedInverterConfig(
        codebook_size=args.codebook_size,
        sparsity=args.sparsity,
        kmeans_iterations=args.kmeans_iterations,
        brotli_quality=args.brotli_quality,
        seed=args.seed,
    )

    if args.features is not None:
        features = load_feature_matrix(args.features)
        input_label = "precomputed_scorer_or_pattern_features"
        source_sha256 = sha256_bytes(args.features.read_bytes())
    else:
        if not args.allow_proxy_from_archive_bytes:
            raise SystemExit("--archive-zip requires --allow-proxy-from-archive-bytes")
        archive_bytes = args.archive_zip.read_bytes()
        features = derive_proxy_features_from_archive_bytes(
            archive_bytes,
            feature_dim=args.proxy_feature_dim,
            max_patterns=args.proxy_max_patterns,
        )
        input_label = "synthetic_archive_byte_proxy_not_scorer_features"
        source_sha256 = sha256_bytes(archive_bytes)

    artifact = build_a1_specialized_inverter(
        features,
        config=config,
        input_label=input_label,
        source_sha256=source_sha256,
    )
    paths = write_a1_specialized_artifact(artifact, args.output_dir)
    if not args.skip_canonical_packet_compiler:
        packet_paths = _build_canonical_custody_packet(
            output_dir=args.output_dir,
            artifact_blob=artifact.compressed_blob,
        )
        artifact.report.update(packet_paths)
        Path(paths["report_path"]).write_text(
            json.dumps(artifact.report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps({"paths": paths, "report": artifact.report}, indent=2, sort_keys=True))
    return 0


def _build_canonical_custody_packet(*, output_dir: Path, artifact_blob: bytes) -> dict[str, object]:
    """Wrap the A1 blob in the canonical deterministic packet compiler."""

    source_packet = output_dir / "a1_specialized_packet_source"
    source_packet.mkdir(parents=True, exist_ok=True)
    archive_path = source_packet / "archive.zip"
    _write_archive_with_a1_blob(archive_path, artifact_blob)
    _write_minimal_runtime(source_packet)

    custody_dir = output_dir / "a1_specialized_packet_custody"
    result = compile_packet(
        input_packet=source_packet,
        output_dir=custody_dir,
        mode="identity",
        target_profile="contest_one_video_replay",
        allow_existing_output_dir=True,
    )
    return {
        "archive_sha": result.archive_sha256,
        "canonical_packet_archive_size_bytes": result.archive_size_bytes,
        "canonical_packet_compiler_blockers": list(result.blockers),
        "canonical_packet_compiler_evidence_grade": "byte_custody_only",
        "canonical_packet_compiler_manifest_path": str(custody_dir / MANIFEST_NAME),
        "canonical_packet_output_dir": str(custody_dir),
        "canonical_packet_runtime_tree_sha256": result.runtime_tree_sha256,
    }


def _write_archive_with_a1_blob(path: Path, artifact_blob: bytes) -> None:
    info = zipfile.ZipInfo("a1_specialized_inverter.bin", date_time=DETERMINISTIC_ZIP_DATE_TIME)
    info.compress_type = zipfile.ZIP_STORED
    info.external_attr = NON_EXECUTABLE_MODE << 16
    info.create_system = 3
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr(info, artifact_blob, compress_type=zipfile.ZIP_STORED)


def _write_minimal_runtime(packet_dir: Path) -> None:
    inflate_sh = packet_dir / "inflate.sh"
    inflate_sh.write_text(
        '#!/usr/bin/env bash\nset -euo pipefail\n'
        'python3 "$(dirname "$0")/inflate.py" "$1" "$2" "$3"\n',
        encoding="utf-8",
    )
    inflate_sh.chmod(
        stat.S_IRUSR
        | stat.S_IWUSR
        | stat.S_IXUSR
        | stat.S_IRGRP
        | stat.S_IXGRP
        | stat.S_IROTH
        | stat.S_IXOTH
    )
    (packet_dir / "inflate.py").write_text(
        "import sys\n"
        "import zipfile\n"
        "from pathlib import Path\n"
        "archive_dir = Path(sys.argv[1])\n"
        "output_dir = Path(sys.argv[2])\n"
        "output_dir.mkdir(parents=True, exist_ok=True)\n"
        "archive_path = archive_dir / 'archive.zip'\n"
        "with zipfile.ZipFile(archive_path, 'r') as zf:\n"
        "    payload = zf.read('a1_specialized_inverter.bin')\n"
        "(output_dir / 'a1_specialized_inverter.bin').write_bytes(payload)\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())
