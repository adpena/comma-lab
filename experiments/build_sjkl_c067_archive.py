#!/usr/bin/env python3
"""Build a SJ-KL-charged archive from a C067-style source archive + sjkl.bin.

Recovery note: this script was lost when subagent worktrees were auto-cleaned
without committing source. Rebuilt 2026-05-04 as the LAST of 6 lost SJ-KL
pipeline modules. Two layouts:

  --archive-layout top_level_sibling   (DEFAULT, fully implemented)
      Preserves the C067 source archive's `p` payload bytes EXACTLY and adds
      sjkl.bin as a top-level ZIP sibling member. The runtime's
      submissions/robust_current/inflate_renderer.py reads sjkl.bin via the
      sibling member name (under SJKL_REQUIRE_APPLIED=1) and applies the
      residual to the JointFrameGenerator forward pass. No undocumented
      byte-format manipulation needed.

      Per .omx/research/sjkl_c067_shrink_addendum_20260502_worker.md (which
      describes the exact sibling layout used by the q6/min-RPK1 candidate
      that landed empirically), this is the production-safe layout.

  --archive-layout packed_rpk1         (SAFE STUB — fails loud)
      Would modify the C067 `p` payload to include sjkl.bin as a logical
      member of the RPK1 container itself. The RPK1 byte layout was not
      preserved in the surviving spec files; reverse-engineering it carries
      silent-corruption risk on existing validated archives. Per CLAUDE.md
      "no signal loss ever" mandate: fail loud rather than silently mis-pack.

Output:
  - <out-dir>/archive.zip         the SJ-KL-charged archive
  - <out-dir>/sjkl_c067_archive_manifest.json   build metadata + provenance

Per CLAUDE.md FORBIDDEN PATTERNS:
  - score_claim=false until contest_auth_eval.json from inflate.sh ->
    upstream/evaluate.py with --device cuda confirms a real score
  - No scorer loads at archive-build time
  - Deterministic ZIP output (ZipInfo + writestr with date_time=(1980, 1, 1, 0, 0, 0))
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import sys
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_SJKL_MEMBER_NAME = "sjkl.bin"
INFLATE_RENDERER_PATH = REPO_ROOT / "submissions" / "robust_current" / "inflate_renderer.py"


@dataclass(frozen=True)
class ArchiveBuildConfig:
    source_archive: Path
    sjkl_bin: Path
    output_dir: Path
    archive_layout: str
    sjkl_member_name: str
    max_sjkl_bytes: int | None = None


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _utc_now() -> str:
    return dt.datetime.now(tz=dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _repo_rel(path: Path) -> str:
    return str(path.relative_to(REPO_ROOT) if path.is_relative_to(REPO_ROOT) else path)


def _load_inflate_renderer_runtime() -> Any:
    if str(REPO_ROOT / "src") not in sys.path:
        sys.path.insert(0, str(REPO_ROOT / "src"))
    spec = importlib.util.spec_from_file_location(
        "_sjkl_archive_builder_inflate_renderer", INFLATE_RENDERER_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import inflate renderer from {INFLATE_RENDERER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _build_runtime_apply_proof(sjkl_bytes: bytes, *, sjkl_member_name: str) -> dict[str, Any]:
    """Prove the default inflate runtime can decode and apply the charged payload.

    This is a local proof hook only. It does not load scorer modules and it does
    not claim score movement; exact CUDA auth eval remains the score truth.
    """
    if sjkl_member_name != RUNTIME_SJKL_MEMBER_NAME:
        return {
            "schema": "sjkl_runtime_apply_proof_v1",
            "verified": False,
            "failure_class": "non_default_member_name",
            "reason": (
                f"default inflate runtime only auto-loads {RUNTIME_SJKL_MEMBER_NAME!r}; "
                f"got {sjkl_member_name!r}"
            ),
        }

    try:
        import torch
    except Exception as exc:  # pragma: no cover - environment-specific import failure
        return {
            "schema": "sjkl_runtime_apply_proof_v1",
            "verified": False,
            "failure_class": "torch_unavailable",
            "reason": repr(exc),
        }

    try:
        runtime = _load_inflate_renderer_runtime()
        with tempfile.TemporaryDirectory(prefix="sjkl-runtime-proof-") as tmp:
            archive_dir = Path(tmp)
            (archive_dir / RUNTIME_SJKL_MEMBER_NAME).write_bytes(sjkl_bytes)
            state = runtime._load_sjkl_residual_from_archive_dir(archive_dir)
            if state is None:
                raise RuntimeError("runtime loader returned None for sjkl.bin")
            basis = state["basis"]
            pair_indices = state.get("pair_indices")
            pair_start = 0 if pair_indices is None else int(pair_indices[0])
            pairs = torch.zeros(
                1,
                2,
                int(basis.target_h),
                int(basis.target_w),
                3,
                dtype=torch.float32,
            )
            before = pairs.clone()
            after = runtime._apply_sjkl_residual_to_pairs(
                pairs,
                state,
                pair_start=pair_start,
            )
            changed_l1 = float((after - before).abs().sum().item())
            applied = int(state.get("applied_pair_count", 0))
            verified = applied > 0 and changed_l1 > 0.0
            return {
                "schema": "sjkl_runtime_apply_proof_v1",
                "verified": verified,
                "failure_class": None if verified else "no_effect_on_probe_pair",
                "member_name": RUNTIME_SJKL_MEMBER_NAME,
                "runtime_loader": _repo_rel(INFLATE_RENDERER_PATH),
                "target_shape": [int(basis.target_h), int(basis.target_w)],
                "alpha_block_format": state.get("alpha_block_format"),
                "pair_start": int(pair_start),
                "applied_pair_count": applied,
                "probe_delta_l1": changed_l1,
                "skip_reasons": list(state.get("skip_reasons", [])),
                "scorer_modules_loaded": False,
            }
    except Exception as exc:
        return {
            "schema": "sjkl_runtime_apply_proof_v1",
            "verified": False,
            "failure_class": "runtime_decode_or_apply_failed",
            "reason": repr(exc),
            "member_name": sjkl_member_name,
            "runtime_loader": _repo_rel(INFLATE_RENDERER_PATH),
            "scorer_modules_loaded": False,
        }


def _build_top_level_sibling(cfg: ArchiveBuildConfig) -> dict:
    """Charged archive: copy source ZIP members verbatim + add sjkl.bin sibling.

    Source bytes preserved exactly: every existing member is read with
    ZipFile.read() and rewritten with the same ZipInfo (date_time + compress_type
    + extra) so the output is byte-deterministic. The sjkl.bin sibling is
    appended with the canonical date_time=(1980,1,1,0,0,0) + ZIP_STORED.
    """
    if not cfg.source_archive.is_file():
        raise SystemExit(f"FATAL: source archive not found: {cfg.source_archive}")
    if not cfg.sjkl_bin.is_file():
        raise SystemExit(f"FATAL: sjkl.bin not found: {cfg.sjkl_bin}")
    if cfg.sjkl_member_name != RUNTIME_SJKL_MEMBER_NAME:
        raise SystemExit(
            f"FATAL: SJ-KL exact-evaluable runtime contract requires member "
            f"'{RUNTIME_SJKL_MEMBER_NAME}', got '{cfg.sjkl_member_name}'. "
            "Do not use 'p' or a custom name unless a reviewed packed-payload "
            "runtime contract is implemented."
        )
    sjkl_bytes = cfg.sjkl_bin.read_bytes()
    if not sjkl_bytes:
        raise SystemExit("FATAL: sjkl.bin is empty")
    max_sjkl_bytes = int(cfg.max_sjkl_bytes) if cfg.max_sjkl_bytes is not None else len(sjkl_bytes)
    if max_sjkl_bytes <= 0:
        raise SystemExit(f"FATAL: max_sjkl_bytes must be positive, got {max_sjkl_bytes}")
    if len(sjkl_bytes) > max_sjkl_bytes:
        raise SystemExit(
            f"FATAL: sjkl.bin size {len(sjkl_bytes)} > max_sjkl_bytes {max_sjkl_bytes}"
        )
    runtime_apply_proof = _build_runtime_apply_proof(sjkl_bytes, sjkl_member_name=cfg.sjkl_member_name)
    if runtime_apply_proof.get("verified") is not True:
        raise SystemExit(f"FATAL: SJ-KL runtime apply proof failed: {runtime_apply_proof}")

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    out_archive = cfg.output_dir / "archive.zip"

    source_members: list[tuple[str, bytes, int]] = []  # (name, data, compress_type)
    with zipfile.ZipFile(cfg.source_archive, "r") as zin:
        for info in zin.infolist():
            if info.filename == cfg.sjkl_member_name:
                raise SystemExit(
                    f"FATAL: source archive already contains member '{cfg.sjkl_member_name}'; "
                    "refusing to silently overwrite."
                )
            data = zin.read(info)
            source_members.append((info.filename, data, info.compress_type))

    # Deterministic write: fixed date_time, fixed compress_type per member
    # DETERMINISTIC_ZIP_OK
    with zipfile.ZipFile(out_archive, "w", compression=zipfile.ZIP_STORED) as zout:
        for name, data, compress_type in source_members:
            zi = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            zi.compress_type = compress_type
            zout.writestr(zi, data)
        zi_sjkl = zipfile.ZipInfo(cfg.sjkl_member_name, date_time=(1980, 1, 1, 0, 0, 0))
        zi_sjkl.compress_type = zipfile.ZIP_STORED  # sjkl.bin is already brotli-compressed inside
        zout.writestr(zi_sjkl, sjkl_bytes)

    return {
        "out_archive_path": _repo_rel(out_archive),
        "out_archive_bytes": out_archive.stat().st_size,
        "out_archive_sha256": _sha256_file(out_archive),
        "source_archive_path": str(cfg.source_archive),
        "source_archive_bytes": cfg.source_archive.stat().st_size,
        "source_archive_sha256": _sha256_file(cfg.source_archive),
        "sjkl_bin_path": str(cfg.sjkl_bin),
        "sjkl_bin_bytes": len(sjkl_bytes),
        "sjkl_bin_sha256": _sha256_bytes(sjkl_bytes),
        "sjkl_member_name": cfg.sjkl_member_name,
        "sjkl_payload": {
            "member_name": cfg.sjkl_member_name,
            "path": str(cfg.sjkl_bin),
            "bytes": len(sjkl_bytes),
            "sha256": _sha256_bytes(sjkl_bytes),
            "max_bytes": max_sjkl_bytes,
        },
        "n_source_members": len(source_members),
        "source_member_names": [name for name, _, _ in source_members],
        "payload_member_names": {
            "source_archive_members": [name for name, _, _ in source_members],
            "output_archive_members": [name for name, _, _ in source_members] + [cfg.sjkl_member_name],
            "output_logical_runtime_members": [name for name, _, _ in source_members] + [cfg.sjkl_member_name],
            "sjkl_member_is_default_runtime_name": True,
        },
        "runtime_contract": {
            "schema": "sjkl_top_level_runtime_contract_v1",
            "layout": "top_level_sibling",
            "default_inflate_member_name": RUNTIME_SJKL_MEMBER_NAME,
            "score_affecting_payload_charged_in_archive": True,
            "scorer_load_allowed_at_inflate": False,
            "sidecars_allowed": False,
            "requires_sjkl_require_applied_for_exact_eval": True,
            "runtime_apply_proof": runtime_apply_proof,
        },
        "archive_layout": "top_level_sibling",
        "score_claim": False,
        "evidence_grade": "queued_exact_cuda_required_for_score",
        "produced_at_utc": _utc_now(),
        "produced_by": "experiments/build_sjkl_c067_archive.py",
    }


def _build_packed_rpk1_stub(cfg: ArchiveBuildConfig) -> dict:
    """SAFE STUB. The RPK1 byte format is not documented in surviving spec
    files; reverse-engineering it carries silent-corruption risk on existing
    validated archives. Use --archive-layout top_level_sibling instead, which
    preserves the source `p` payload bytes exactly.
    """
    raise NotImplementedError(
        "RPK1 packed layout is a recovery stub: the original "
        "experiments/build_sjkl_c067_archive.py packed_rpk1 packer was lost "
        "when subagent worktrees were auto-cleaned without committing source. "
        "The RPK1 byte layout was not documented in surviving spec files. "
        "Reverse-engineering it from the C067 source's `p` container header "
        "(5b 98 68 43 ...) carries silent-corruption risk on archives this "
        "repo has already validated. Use --archive-layout top_level_sibling "
        "instead — it preserves source `p` bytes exactly and adds sjkl.bin as "
        "a top-level ZIP sibling member that the runtime reads under "
        "SJKL_REQUIRE_APPLIED=1."
    )


def build_sjkl_c067_archive(cfg: ArchiveBuildConfig) -> dict:
    if cfg.archive_layout == "top_level_sibling":
        manifest = _build_top_level_sibling(cfg)
    elif cfg.archive_layout == "packed_rpk1":
        manifest = _build_packed_rpk1_stub(cfg)
    else:
        raise SystemExit(f"FATAL: unknown archive layout: {cfg.archive_layout}")

    manifest_path = cfg.output_dir / "sjkl_c067_archive_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    print(f"[sjkl-c067-archive] wrote {manifest['out_archive_path']} "
          f"({manifest['out_archive_bytes']} bytes, sha {manifest['out_archive_sha256'][:16]})", file=sys.stderr)
    print(f"[sjkl-c067-archive] wrote {manifest_path}", file=sys.stderr)
    print(f"[sjkl-c067-archive] layout={manifest['archive_layout']} score_claim=false; "
          f"auth eval through contest CUDA required", file=sys.stderr)

    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-archive", type=Path, required=True,
                        help="C067-style source archive.zip (single-member 'p' container).")
    parser.add_argument("--sjkl-bin", type=Path, required=True,
                        help="sjkl.bin payload (output of experiments/build_sjkl_residual.py).")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--archive-layout", default="top_level_sibling",
                        choices=["top_level_sibling", "packed_rpk1"],
                        help="top_level_sibling preserves source bytes exactly; packed_rpk1 is a safe stub.")
    parser.add_argument("--sjkl-member-name", default="sjkl.bin",
                        help="ZIP member name under which sjkl.bin will be added.")
    parser.add_argument("--max-sjkl-bytes", type=int, default=None,
                        help="Optional fail-closed cap for charged sjkl.bin bytes.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = ArchiveBuildConfig(
        source_archive=args.source_archive,
        sjkl_bin=args.sjkl_bin,
        output_dir=args.output_dir,
        archive_layout=args.archive_layout,
        sjkl_member_name=args.sjkl_member_name,
        max_sjkl_bytes=args.max_sjkl_bytes,
    )
    build_sjkl_c067_archive(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
