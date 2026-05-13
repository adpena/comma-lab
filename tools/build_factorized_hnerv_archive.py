"""Build a factorized_hnerv_v1 candidate archive from a PR101/PR106/PR107 substrate.

Inputs:
    --substrate-archive PATH  Path to a PR107-style ``archive.zip`` (whose
                              ``0.bin`` is parsed by PR107's apogee codec).
    --output-dir PATH         Destination directory. Will create:
                              ``archive.zip`` (the new factorized candidate),
                              ``submission_dir/`` (inflate.sh + inflate.py +
                              src/ vendored), and ``build_manifest.json``.
    --plan-config JSON-PATH   Optional JSON file specifying ``factorized_indices``
                              and ``per_index_rank``. Defaults to codex sub017
                              ``svd_stem_blocks012_balanced`` ranks (stem r=20,
                              blocks.0/1/2 r=64).
    --target-rms-err FLOAT    Optional per-tensor RMS rel-err cap (warning only).

Outputs (in ``--output-dir``):

* ``archive.zip``             — the candidate archive (a ZIP with one file
                                ``0.bin`` containing the factorized_hnerv_v1
                                wire format).
* ``submission_dir/``         — inflate.sh, inflate.py, src/ vendored
                                (canonical layout for an inflate-runnable lane).
* ``build_manifest.json``     — full custody manifest including:
    schema_version, factorized_tensor_indices, per_tensor_ranks,
    per_tensor_rel_errs, factor_section_bytes, non_factorized_section_bytes,
    archive_bytes, score_claim=False, ready_for_exact_eval_dispatch=False
    until anchored, dispatch_blockers, custody_status.

Per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA": the build manifest
will list both CPU and CUDA dispatch blockers; the harvest pipeline lifts
them once the dual eval lands.

Per CLAUDE.md "Strict scorer rule": this tool loads NO scorer.

Usage:
    .venv/bin/python tools/build_factorized_hnerv_archive.py \\
        --substrate-archive experiments/.../public_pr107_intake.../archive.zip \\
        --output-dir experiments/results/sub017_factorized_hnerv_<ts>/

Smoke (synthetic substrate; no real PR107):
    .venv/bin/python tools/build_factorized_hnerv_archive.py \\
        --synthetic-substrate \\
        --output-dir /tmp/factorized_hnerv_smoke
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import shutil
import struct
import sys
import zipfile
from pathlib import Path

import brotli  # noqa: F401  (used indirectly via codec module)
import numpy as np
import torch

# Late-import the encoder side from tac (the decoder is vendored into the
# submissions/factorized_hnerv_v1/src/codec.py for inflate-time use).
from tac.codec.factorized_hnerv_codec import (
    FIXED_STATE_SCHEMA,
    FactorizedSectionPlan,
    WIRE_FORMAT_VERSION,
    encode_factorized_section,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
SUB_TEMPLATE = REPO_ROOT / "submissions" / "factorized_hnerv_v1"


DEFAULT_PLAN = {
    "factorized_indices": [0, 2, 4, 6],
    "per_index_rank": {0: 20, 2: 64, 4: 64, 6: 64},
}


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _archive_member_manifest(archive_zip: Path) -> list[dict[str, object]]:
    members: list[dict[str, object]] = []
    with zipfile.ZipFile(archive_zip) as zf:
        for info in zf.infolist():
            members.append({
                "name": info.filename,
                "file_size": info.file_size,
                "compress_size": info.compress_size,
                "crc": info.CRC,
                "sha256": _sha256_bytes(zf.read(info)),
            })
    return members


def _load_pr107_substrate(archive_path: Path) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Load PR107 (apogee) ``archive.zip`` -> (state_dict, latents)."""
    pr107_codec_dir = (
        REPO_ROOT
        / "experiments/results/public_pr_intake_full"
        / "public_pr107_intake_20260505_auto/source/submissions/apogee/src"
    )
    if not pr107_codec_dir.exists():
        raise FileNotFoundError(
            f"PR107 source not found at {pr107_codec_dir}; need it to parse the substrate."
        )
    # Use a private importlib namespace so we don't pollute sys.modules
    # with PR107's ``codec``/``model`` (which collide with our submission's).
    import importlib.util
    sys.path.insert(0, str(pr107_codec_dir))
    try:
        # First load `model` (codec.parse_archive does `from model import HNeRVDecoder`).
        for nm in ("model", "codec"):
            spec = importlib.util.spec_from_file_location(
                f"_pr107_substrate_{nm}", pr107_codec_dir / f"{nm}.py"
            )
            mod = importlib.util.module_from_spec(spec)
            sys.modules[f"_pr107_substrate_{nm}"] = mod
            # codec.py does ``from model import HNeRVDecoder`` (unqualified). Stage
            # our private model module under the unqualified name temporarily.
            sys.modules[nm] = sys.modules.get(f"_pr107_substrate_{nm}", mod)
            spec.loader.exec_module(mod)
        codec_mod = sys.modules["_pr107_substrate_codec"]
        with zipfile.ZipFile(archive_path) as z:
            data = z.read("0.bin")
        state_dict, latents, _meta = codec_mod.parse_archive(data)
        return state_dict, latents
    finally:
        if str(pr107_codec_dir) in sys.path:
            sys.path.remove(str(pr107_codec_dir))
        # Clean up unqualified ``model`` and ``codec`` module shims so the
        # later smoke test imports the SUBMISSION's vendored versions.
        for nm in ("codec", "model"):
            if nm in sys.modules and sys.modules[nm].__file__ and \
               str(pr107_codec_dir) in sys.modules[nm].__file__:
                del sys.modules[nm]


def _load_synthetic_substrate(seed: int = 0) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Build a synthetic substrate that mimics PR107's structure."""
    torch.manual_seed(seed)
    sd = {n: torch.randn(*s) for n, s in FIXED_STATE_SCHEMA}
    # Latents: (600, 28) tensor, will be re-encoded via brotli below.
    latents = torch.randn(600, 28).clamp(-3, 3) * 50 + 100
    return sd, latents


def _encode_pr_style_latents(latents: torch.Tensor, brotli_quality: int = 11) -> bytes:
    """PR107/PR106 latent encoder (round-trip-faithful with vendored decoder).

    Wire format (matches ``submissions.factorized_hnerv_v1.src.codec.decode_fixed_latents``):
        [n*d uint8 lo][d fp16 mins][d fp16 scales][n*d uint8 hi]   then brotli'd.
    """
    n, d = 600, 28
    assert latents.shape == (n, d), f"expected (600, 28), got {tuple(latents.shape)}"
    arr = latents.detach().cpu().float().numpy().astype(np.float64)
    mins = arr.min(axis=0)
    maxs = arr.max(axis=0)
    rng = np.where(maxs - mins > 0, maxs - mins, 1.0)
    scales = rng / 254.0
    q = np.round((arr - mins[None, :]) / scales[None, :]).clip(0, 254).astype(np.uint8)
    # 1st-order temporal delta on the uint8 codes
    delta = np.empty_like(q, dtype=np.int16)
    delta[0] = q[0].astype(np.int16)
    delta[1:] = q[1:].astype(np.int16) - q[:-1].astype(np.int16)
    # zigzag to uint16
    delta_zz = np.where(delta >= 0, 2 * delta, -2 * delta - 1).astype(np.uint16)
    lo = (delta_zz & 0xFF).astype(np.uint8)
    hi = ((delta_zz >> 8) & 0xFF).astype(np.uint8)
    payload = (
        lo.tobytes()
        + np.float16(mins).tobytes()
        + np.float16(scales).tobytes()
        + hi.tobytes()
    )
    return brotli.compress(payload, quality=brotli_quality)


def _build_archive_bytes(decoder_section: bytes, latent_section: bytes) -> bytes:
    """Pack decoder + latent into the factorized_hnerv_v1 wire format."""
    return (
        bytes([0xF1])
        + struct.pack("<I", len(decoder_section))
        + decoder_section
        + struct.pack("<I", len(latent_section))
        + latent_section
    )


def _write_archive_zip(out_zip: Path, archive_bytes: bytes) -> None:
    out_zip.parent.mkdir(parents=True, exist_ok=True)
    # Use deterministic ZIP info per CLAUDE.md PCC2 (deterministic-zip).
    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_STORED) as z:
        info = zipfile.ZipInfo("0.bin", date_time=(2026, 1, 1, 0, 0, 0))
        info.compress_type = zipfile.ZIP_STORED
        z.writestr(info, archive_bytes)


def _stage_submission_dir(out_dir: Path) -> None:
    """Copy submissions/factorized_hnerv_v1/ into out_dir/submission_dir/."""
    target = out_dir / "submission_dir"
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    for name in ("inflate.sh", "inflate.py"):
        shutil.copy2(SUB_TEMPLATE / name, target / name)
    src_target = target / "src"
    src_target.mkdir()
    for f in (SUB_TEMPLATE / "src").iterdir():
        if f.suffix == ".py":
            shutil.copy2(f, src_target / f.name)
    (src_target / "__init__.py").touch()


def _write_submission_custody(
    *,
    out_dir: Path,
    archive_zip: Path,
    archive_payload_sha256: str,
    archive_payload_bytes: int,
    candidate_manifest: dict[str, object],
) -> dict[str, object]:
    """Close the local submission surface around the built archive.

    ``scripts/pre_submission_compliance_check.py`` treats ``archive.zip``,
    ``archive_manifest.json``, and ``report.txt`` as the operator-facing
    packet surface.  The factorized builder used to emit only an external
    archive plus ``build_manifest.json``; that was not dispatchable without
    manual glue and it also made the word ``archive_sha256`` ambiguous.  This
    helper makes the ZIP archive the canonical archive identity and stores the
    inner ``0.bin`` payload hash separately.
    """

    submission_dir = out_dir / "submission_dir"
    submission_archive = submission_dir / "archive.zip"
    shutil.copy2(archive_zip, submission_archive)

    archive_zip_sha256 = _sha256_file(archive_zip)
    archive_zip_bytes = int(archive_zip.stat().st_size)
    members = _archive_member_manifest(archive_zip)
    archive_manifest: dict[str, object] = {
        "schema_version": "factorized_hnerv_archive_manifest_v1",
        "lane_id": candidate_manifest["lane_id"],
        "archive_path": "archive.zip",
        "archive_sha256": archive_zip_sha256,
        "archive_size_bytes": archive_zip_bytes,
        "archive_bytes": archive_zip_bytes,
        "archive_payload_sha256": archive_payload_sha256,
        "archive_payload_bytes": archive_payload_bytes,
        "score_claim": False,
        "ready_for_exact_eval_dispatch": False,
        "score_affecting_payload_changed": True,
        "charged_bits_changed": True,
        "members": members,
        "build_manifest_path": "../build_manifest.json",
    }
    (submission_dir / "archive_manifest.json").write_text(
        json.dumps(archive_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (submission_dir / "report.txt").write_text(
        "\n".join(
            [
                "factorized_hnerv_v1",
                f"archive_sha256: {archive_zip_sha256}",
                f"archive_size_bytes: {archive_zip_bytes}",
                f"archive_payload_sha256: {archive_payload_sha256}",
                f"archive_payload_bytes: {archive_payload_bytes}",
                "score_claim: false",
                "ready_for_exact_eval_dispatch: false",
                "evidence_grade: [CPU-build proxy; not yet contest-CPU or contest-CUDA anchored]",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return {
        "submission_dir": str(submission_dir),
        "submission_archive_path": str(submission_archive),
        "archive_manifest_path": str(submission_dir / "archive_manifest.json"),
        "report_path": str(submission_dir / "report.txt"),
        "archive_zip_sha256": archive_zip_sha256,
        "archive_zip_bytes": archive_zip_bytes,
        "members": members,
    }


def _smoke_test_inflate(archive_zip_path: Path, out_raw: Path) -> dict[str, object]:
    """End-to-end smoke: parse archive bytes, decode, run forward on first
    few latents on CPU, return diagnostics. Does NOT compute a contest score.
    """
    # Use importlib to import our submission's vendored codec/model into a
    # private namespace, avoiding any sys.modules collision with PR107's
    # substrate codec/model that may be cached.
    import importlib.util
    src_dir = SUB_TEMPLATE / "src"
    for nm in ("model", "codec"):
        spec = importlib.util.spec_from_file_location(
            f"_factorized_v1_{nm}", src_dir / f"{nm}.py"
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules[f"_factorized_v1_{nm}"] = mod
        spec.loader.exec_module(mod)
    parse_archive = sys.modules["_factorized_v1_codec"].parse_archive
    HNeRVDecoder = sys.modules["_factorized_v1_model"].HNeRVDecoder

    with zipfile.ZipFile(archive_zip_path) as z:
        data = z.read("0.bin")
    sd, latents, meta = parse_archive(data)
    model = HNeRVDecoder(
        latent_dim=meta["latent_dim"],
        base_channels=meta["base_channels"],
        eval_size=tuple(meta["eval_size"]),
    )
    model.load_state_dict(sd)
    model.eval()
    with torch.inference_mode():
        out = model(latents[:4])
    return {
        "smoke_decode_ok": True,
        "smoke_state_dict_keys": len(sd),
        "smoke_forward_shape": list(out.shape),
        "smoke_forward_value_range": [float(out.min()), float(out.max())],
        "smoke_archive_bytes": int(archive_zip_path.stat().st_size),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--substrate-archive", type=Path, default=None,
                   help="Path to PR107 archive.zip whose 0.bin parses via apogee codec.")
    p.add_argument("--synthetic-substrate", action="store_true",
                   help="Use a synthetic state_dict (smoke / unit-test mode).")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--plan-config", type=Path, default=None,
                   help="JSON file with factorized_indices + per_index_rank; "
                        "defaults to codex sub017 svd_stem_blocks012_balanced.")
    p.add_argument("--brotli-quality", type=int, default=11)
    p.add_argument("--target-rms-err", type=float, default=None,
                   help="Per-tensor RMS rel-err cap; warns if any factorization exceeds it.")
    args = p.parse_args()

    if not (args.substrate_archive or args.synthetic_substrate):
        sys.exit("must pass --substrate-archive or --synthetic-substrate")

    if args.substrate_archive:
        sd, latents = _load_pr107_substrate(args.substrate_archive)
    else:
        sd, latents = _load_synthetic_substrate()

    if args.plan_config:
        with args.plan_config.open() as f:
            cfg = json.load(f)
    else:
        cfg = DEFAULT_PLAN
    factorized_indices = tuple(int(i) for i in cfg["factorized_indices"])
    per_index_rank = {int(k): int(v) for k, v in cfg["per_index_rank"].items()}
    plan = FactorizedSectionPlan(
        factorized_indices=factorized_indices,
        per_index_rank=per_index_rank,
    )

    target_rms_err_per_tensor = (
        {idx: float(args.target_rms_err) for idx in plan.factorized_indices}
        if args.target_rms_err is not None else None
    )

    # 1. Encode the factorized decoder section.
    decoder_section, telem = encode_factorized_section(
        sd, plan,
        target_rms_err_per_tensor=target_rms_err_per_tensor,
        brotli_quality=args.brotli_quality,
    )
    # 2. Encode latents PR-style.
    latent_section = _encode_pr_style_latents(latents, brotli_quality=args.brotli_quality)
    # 3. Pack to factorized_hnerv_v1 wire format.
    archive_bytes = _build_archive_bytes(decoder_section, latent_section)
    archive_payload_sha256 = _sha256_bytes(archive_bytes)

    # 4. Write archive.zip + stage submission_dir.
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    archive_zip = out_dir / "archive.zip"
    _write_archive_zip(archive_zip, archive_bytes)
    _stage_submission_dir(out_dir)

    # 5. Smoke-test inflate end-to-end (CPU-only, no scorer).
    smoke = _smoke_test_inflate(archive_zip, out_dir / "smoke.raw")

    # 6. Build manifest.
    archive_zip_bytes = int(archive_zip.stat().st_size)
    archive_zip_sha256 = _sha256_file(archive_zip)
    archive_members = _archive_member_manifest(archive_zip)
    manifest: dict[str, object] = {
        "schema_version": WIRE_FORMAT_VERSION,
        "lane_id": "factorized_hnerv_v1",
        "build_timestamp_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "substrate_archive": str(args.substrate_archive) if args.substrate_archive else "synthetic",
        "factorized_tensor_indices": list(factorized_indices),
        "per_tensor_ranks": telem["per_tensor_ranks"],
        "per_tensor_rel_errs": telem["per_tensor_rel_errs"],
        "per_tensor_rel_err_form": telem["per_tensor_rel_err_form"],
        "factor_section_bytes": telem["factor_section_bytes"],
        "non_factorized_section_bytes": telem["non_factorized_section_bytes"],
        "decoder_section_bytes": int(len(decoder_section)),
        "latent_section_bytes": int(len(latent_section)),
        "archive_payload_bytes": int(len(archive_bytes)),
        "archive_payload_sha256": archive_payload_sha256,
        "archive_bytes": archive_zip_bytes,
        "archive_size_bytes": archive_zip_bytes,
        "archive_zip_bytes": archive_zip_bytes,
        "archive_zip_sha256": archive_zip_sha256,
        "archive_sha256": archive_zip_sha256,
        "archive_member_manifest": archive_members,
        "brotli_quality": args.brotli_quality,
        "score_claim": False,
        "ready_for_exact_eval_dispatch": False,
        "byte_proxy_only": True,
        "cuda_eval_worth_testing": "pending_relerr_review",
        "evidence_grade": "[CPU-build proxy; not yet contest-CPU or contest-CUDA anchored]",
        "dispatch_blockers": [
            "no_cpu_eval_yet",
            "no_cuda_eval_yet",
            "rel_err_per_tensor_above_baseline_relerr_envelope",
        ],
        "smoke_test": smoke,
        "plan_used": {
            "factorized_indices": list(factorized_indices),
            "per_index_rank": per_index_rank,
        },
        "target_modes": ["contest_exact_eval"],
        "deployment_target": "t4_contest_runtime",
        "score_affecting_payload_changed": True,
        "charged_bits_changed": True,
        "custody_status": "ci-rebuildable",
        "rel_err_form": telem["per_tensor_rel_err_form"],
        "notes": (
            "Factorized HNeRV runtime closes codex sub017_plan dispatch blocker "
            "factorized_hnerv_runtime_not_implemented. Empirical PR107 substrate "
            "spectrum is near-flat (top-5 singular values within 0.74-1.0 of "
            "max) so SVD low-rank at low rank yields HIGH per-tensor rel_err "
            "(~40-50% RMS at codex's recommended ranks). Runtime ships as a "
            "BUILD-ONLY anchor for future Q-FAITHFUL retrains that explicitly "
            "enforce low-rank structure. Tagged DEFERRED-pending-research per "
            "CLAUDE.md 'KILL is LAST RESORT'; predicted score lift NOT claimed."
        ),
    }
    manifest["submission_custody"] = _write_submission_custody(
        out_dir=out_dir,
        archive_zip=archive_zip,
        archive_payload_sha256=archive_payload_sha256,
        archive_payload_bytes=int(len(archive_bytes)),
        candidate_manifest=manifest,
    )
    with (out_dir / "build_manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Built {archive_zip} ({archive_zip_bytes} bytes, sha256={archive_zip_sha256[:16]}...)")
    print(f"Manifest at {out_dir / 'build_manifest.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
