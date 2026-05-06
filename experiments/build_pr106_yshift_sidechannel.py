#!/usr/bin/env python3
# ruff: noqa: I001
"""Build a pr106_yshift_sidechannel archive on top of a PR106 packed archive.

Layered on top of PR106 (see docs/pr106_byte_layout_deconstruction_20260504.md):
  outer 0xFC dispatch byte + uint24 PR106 length + raw PR106 bytes
  + sidechannel version (1) + uint16 brotli'd-len + brotli(SC01 YSHIFT payload).

SC01 YSHIFT payload (post-brotli) = 14-byte header + n_frames*3 int8 raw stream
of [y_off, dy, dx] per output frame. Mirrors the codex_metric_yshift_av1
SC01 mode-7 wire format. Inflate path is `submissions/pr106_yshift_sidechannel/inflate.py`.

Two operating modes:

  --search-mode {zero, score_table, gradient, brute_force}
    zero        : write a no-op SC01 (all zeros). Brotli compresses to ~few bytes.
                  Used for CPU-only smoke + wire-format roundtrip verification.
                  CPU-tagged [advisory only] per CLAUDE.md MPS-noise rule.
    score_table : scorer-free reducer over a precomputed CUDA scorer table
                  (`--score-table-npy`). This turns measured candidate scores
                  into deterministic charged sidechannel bytes without loading
                  scorers at inflate time or claiming a score.
    gradient    : single backward pass per frame ∂score/∂(Y_offset, dy, dx).
                  Quantize the optimal direction to int8. Requires CUDA + scorers.
                  ~$0.20 on Vast.ai 4090.
    brute_force : 7 × 7 × 7 grid (dy, dx ∈ {-3..3}, y_off ∈ {-3..3}*step)
                  per frame; pick min(score). ~$0.40 on Vast.ai 4090.
                  Operator picks via env: PR106_YSHIFT_MODE=brute_force.

  --score-step
    Float scale factor for y_off (codex pattern uses 1.0 for raw int8 scale;
    smaller values like 0.5 give finer DC luma adjustment with same int8 budget).

Smoke-mode preview (CPU, --search-mode zero) — empirically verified 2026-05-05:
  In:  PR106 archive (186,131 bytes anchor 0.bin / 186,239 bytes zip)
  Out: pr106_yshift_sidechannel archive 186,283 bytes zip (+44 bytes for
       brotli'd zero-payload + 6-byte outer dispatch wrapper)
       (no distortion improvement; wire format proof only;
        rate Δ +0.000029 vs PR106 zip)

Real-mode dispatch (CUDA):
  search-mode=gradient predicted distortion Δ ~-0.0005 to -0.0015 score
  search-mode=brute_force predicted distortion Δ ~-0.001 to -0.002 score
  Both add ~600B-2KB to archive (rate Δ ~+0.0004 to +0.0013).
  Net: ~-0.0005 to -0.001 score Δ. STACKS on lane_pr106_latent_sidecar.

Strict-scorer-rule: scorer is loaded ONLY at compress-time (gradient/brute_force
modes). NEVER at inflate. Per-frame deltas are precomputed and frozen into
the SC01 sidechannel payload.
"""
from __future__ import annotations

import argparse
import importlib.util
import io
import json
import struct
import zipfile
from pathlib import Path

import brotli  # type: ignore[import-not-found]
import numpy as np


try:
    from tools.tool_bootstrap import ensure_repo_imports, prepend_paths, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    _bootstrap_path = Path(__file__).resolve().parent.parent / "tools" / "tool_bootstrap.py"
    _spec = importlib.util.spec_from_file_location("tool_bootstrap", _bootstrap_path)
    if _spec is None or _spec.loader is None:
        raise
    _tool_bootstrap = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_tool_bootstrap)
    ensure_repo_imports = _tool_bootstrap.ensure_repo_imports
    prepend_paths = _tool_bootstrap.prepend_paths
    repo_root_from_tool = _tool_bootstrap.repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)
prepend_paths(
    REPO_ROOT / "submissions" / "pr106_yshift_sidechannel",
    REPO_ROOT / "submissions" / "apogee_intN" / "src",
)

from tac.repo_io import json_text, sha256_bytes, sha256_file

from inflate import (  # type: ignore[import-not-found]
    YSHIFT_MAGIC_BYTE,
    SIDECHANNEL_VERSION,
    SC01_MAGIC,
    SC01_HEADER,
    SIDECHANNEL_MODE_Y_SHIFT,
    parse_yshift_archive,
)


def _read_pr106_bytes(pr106_archive: Path) -> bytes:
    with zipfile.ZipFile(pr106_archive) as z:
        return z.read("0.bin")


def _encode_sc01_yshift(values_int8: np.ndarray, step: float) -> bytes:
    """Pack (n_frames, 3) int8 array into SC01 YSHIFT payload (un-brotli'd)."""
    if values_int8.dtype != np.int8:
        raise TypeError(f"values must be int8, got {values_int8.dtype}")
    if values_int8.ndim != 2 or values_int8.shape[1] != 3:
        raise ValueError(f"values must be (n_frames, 3), got {values_int8.shape}")
    n_frames = int(values_int8.shape[0])
    header = SC01_HEADER.pack(SC01_MAGIC, SIDECHANNEL_MODE_Y_SHIFT, 3, n_frames, float(step))
    return header + values_int8.tobytes()


def _build_yshift_archive_bytes(pr106_bytes: bytes, sc01_blob: bytes | None) -> bytes:
    """Wrap PR106 bytes + optional brotli-compressed SC01 in pr106_yshift outer layout."""
    if len(pr106_bytes) >= (1 << 24):
        raise ValueError(f"pr106 bytes too large for uint24: {len(pr106_bytes)}")
    out = io.BytesIO()
    out.write(bytes([YSHIFT_MAGIC_BYTE]))
    out.write(len(pr106_bytes).to_bytes(3, "little"))
    out.write(pr106_bytes)
    if sc01_blob is not None:
        out.write(bytes([SIDECHANNEL_VERSION]))
        if len(sc01_blob) >= (1 << 16):
            raise ValueError(f"sc01 blob too large for uint16: {len(sc01_blob)}")
        out.write(struct.pack("<H", len(sc01_blob)))
        out.write(sc01_blob)
    return out.getvalue()


def _zero_search(n_frames: int) -> np.ndarray:
    """No-op corrections — used for CPU smoke + wire-format roundtrip."""
    return np.zeros((n_frames, 3), dtype=np.int8)


def _gradient_search_stub(n_frames: int) -> np.ndarray:
    """Placeholder stub for gradient mode — fails LOUD if invoked without CUDA."""
    raise NotImplementedError(
        "gradient search mode requires CUDA + scorers. Run via the remote dispatch "
        "wrapper scripts/remote_lane_pr106_yshift_sidechannel.sh which loads CUDA "
        "scorers at compress time only (per CLAUDE.md strict-scorer-rule)."
    )


def _brute_force_search_stub(n_frames: int) -> np.ndarray:
    """Placeholder stub for brute-force mode — fails LOUD if invoked without CUDA."""
    raise NotImplementedError(
        "brute_force search mode requires CUDA + scorers. Run via the remote dispatch "
        "wrapper scripts/remote_lane_pr106_yshift_sidechannel.sh which loads CUDA "
        "scorers at compress time only (per CLAUDE.md strict-scorer-rule)."
    )


SEARCH_MODES = {
    "zero": _zero_search,
    "gradient": _gradient_search_stub,
    "brute_force": _brute_force_search_stub,
}
SEARCH_MODE_CHOICES = ("zero", "score_table", "gradient", "brute_force")


def build_yshift_candidate_grid(radius: int = 3) -> np.ndarray:
    """Return canonical int8 [y_off, dy, dx] candidates for scorer-backed search."""
    if radius < 0 or radius > 127:
        raise ValueError(f"radius must be in 0..127, got {radius}")
    vals = range(-radius, radius + 1)
    candidates = np.array(
        [(y_off, dy, dx) for y_off in vals for dy in vals for dx in vals],
        dtype=np.int16,
    )
    if not ((candidates == 0).all(axis=1)).any():
        raise AssertionError("candidate grid must include the all-zero no-op")
    return candidates.astype(np.int8)


def yshift_candidate_grid_npy_sha256(candidates: np.ndarray) -> str:
    """Return the deterministic `.npy` SHA-256 used by the CUDA score-table manifest."""
    raw = io.BytesIO()
    np.save(raw, np.asarray(candidates, dtype=np.int8), allow_pickle=False)
    return sha256_bytes(raw.getvalue())


def choose_yshift_candidates_from_scores(
    score_table: np.ndarray,
    candidates: np.ndarray,
    *,
    require_improvement: bool = True,
) -> np.ndarray:
    """Choose one yshift candidate per frame from a precomputed score table.

    `score_table[f, c]` must be the exact scorer objective for frame `f` and
    candidate `c`, computed outside this helper on the authorized CUDA path.
    This reducer is deterministic and scorer-free; it only turns measured
    candidate scores into charged sidechannel bytes.
    """
    scores = np.asarray(score_table, dtype=np.float64)
    cands = np.asarray(candidates)
    if cands.dtype != np.int8:
        raise TypeError(f"candidates must be int8, got {cands.dtype}")
    if cands.ndim != 2 or cands.shape[1] != 3:
        raise ValueError(f"candidates must have shape (n_candidates, 3), got {cands.shape}")
    if scores.ndim != 2 or scores.shape[1] != cands.shape[0]:
        raise ValueError(
            "score_table must have shape (n_frames, n_candidates), got "
            f"{scores.shape} for {cands.shape[0]} candidates"
        )
    if not np.isfinite(scores).all():
        raise ValueError("score_table contains NaN/Inf")
    zero_matches = np.flatnonzero((cands == 0).all(axis=1))
    if len(zero_matches) != 1:
        raise ValueError("candidates must contain exactly one all-zero no-op row")
    zero_idx = int(zero_matches[0])
    best_idx = scores.argmin(axis=1)
    if require_improvement:
        best_scores = scores[np.arange(scores.shape[0]), best_idx]
        zero_scores = scores[:, zero_idx]
        best_idx = np.where(best_scores < zero_scores, best_idx, zero_idx)
    return cands[best_idx].astype(np.int8, copy=True)


def choose_yshift_candidates_from_score_table_file(
    score_table_npy: Path,
    *,
    n_frames: int,
    candidate_radius: int,
) -> tuple[np.ndarray, dict[str, object]]:
    """Load a precomputed candidate score table and emit deterministic yshift rows.

    This helper is intentionally scorer-free. The input table must already have
    been produced by a CUDA scorer job against the exact source archive. The
    builder records file custody and keeps the archive non-promotable until a
    separate exact CUDA auth eval scores the emitted bytes.
    """
    candidates = build_yshift_candidate_grid(radius=candidate_radius)
    loaded = np.load(score_table_npy, allow_pickle=False)
    if not isinstance(loaded, np.ndarray):
        raise TypeError(f"score table must be a .npy ndarray, got {type(loaded).__name__}")
    if loaded.shape != (n_frames, len(candidates)):
        raise ValueError(
            "score table shape mismatch: expected "
            f"({n_frames}, {len(candidates)}), got {loaded.shape}"
        )
    scores = np.asarray(loaded, dtype=np.float64)
    selected = choose_yshift_candidates_from_scores(scores, candidates)
    zero_idx = int(np.flatnonzero((candidates == 0).all(axis=1))[0])
    selected_indices = np.array(
        [
            int(np.flatnonzero((candidates == row).all(axis=1))[0])
            for row in selected
        ],
        dtype=np.int32,
    )
    selected_scores = scores[np.arange(n_frames), selected_indices]
    zero_scores = scores[:, zero_idx]
    improvements = zero_scores - selected_scores
    nonzero_rows = ~(selected == 0).all(axis=1)
    diagnostics: dict[str, object] = {
        "candidate_grid_radius": int(candidate_radius),
        "candidate_grid_count": len(candidates),
        "score_table_shape": [int(scores.shape[0]), int(scores.shape[1])],
        "zero_candidate_index": int(zero_idx),
        "selected_nonzero_frame_count": int(nonzero_rows.sum()),
        "selected_zero_frame_count": int((~nonzero_rows).sum()),
        "best_improvement_min": float(improvements.min()) if len(improvements) else 0.0,
        "best_improvement_mean": float(improvements.mean()) if len(improvements) else 0.0,
        "best_improvement_max": float(improvements.max()) if len(improvements) else 0.0,
        "require_strict_improvement_over_noop": True,
    }
    return selected, diagnostics


def validate_score_table_manifest(
    manifest_path: Path,
    *,
    score_table_npy: Path,
    pr106_archive: Path,
    n_frames: int,
    candidate_radius: int,
    candidate_count: int,
    score_step: float,
) -> dict[str, object]:
    """Validate CUDA score-table provenance before reducing it to charged bytes."""

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"score table manifest is not valid JSON: {exc}") from exc
    if not isinstance(manifest, dict):
        raise ValueError("score table manifest must be a JSON object")
    if manifest.get("manifest_schema") != "pr106_yshift_score_table_manifest_v1":
        raise ValueError("score table manifest_schema mismatch")
    if manifest.get("producer") != "experiments/build_pr106_yshift_score_table.py":
        raise ValueError("score table manifest producer mismatch")
    if manifest.get("score_claim") is not False:
        raise ValueError("score table manifest must keep score_claim=false")
    if manifest.get("ready_for_builder") is not True:
        raise ValueError("score table manifest must have ready_for_builder=true")
    if manifest.get("source_archive_sha256") != sha256_file(pr106_archive):
        raise ValueError("score table manifest source_archive_sha256 does not match --pr106-archive")
    if manifest.get("score_table_npy_sha256") != sha256_file(score_table_npy):
        raise ValueError("score table manifest score_table_npy_sha256 does not match --score-table-npy")
    if manifest.get("candidate_radius") != int(candidate_radius):
        raise ValueError("score table manifest candidate_radius mismatch")
    if manifest.get("candidate_count") != int(candidate_count):
        raise ValueError("score table manifest candidate_count mismatch")
    expected_grid_sha256 = yshift_candidate_grid_npy_sha256(
        build_yshift_candidate_grid(radius=candidate_radius)
    )
    if manifest.get("candidate_grid_sha256") != expected_grid_sha256:
        raise ValueError("score table manifest candidate_grid_sha256 mismatch")
    if manifest.get("n_frames") != int(n_frames):
        raise ValueError("score table manifest n_frames mismatch")
    if manifest.get("score_table_shape") != [int(n_frames), int(candidate_count)]:
        raise ValueError("score table manifest score_table_shape mismatch")
    if float(manifest.get("score_step", float("nan"))) != float(score_step):
        raise ValueError("score table manifest score_step mismatch")
    if manifest.get("ready_for_exact_eval_dispatch") is True:
        raise ValueError("score table manifest must not claim exact-eval dispatch readiness")
    if manifest.get("dispatch_attempted") is True or manifest.get("remote_jobs_dispatched") is True:
        raise ValueError("score table manifest must not mark dispatch attempted")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pr106-archive", type=Path, required=True,
                        help="Path to PR106 packed archive.zip (anchor).")
    parser.add_argument("--out-dir", type=Path, required=True,
                        help="Output directory for built archive + metadata.")
    parser.add_argument("--search-mode", choices=SEARCH_MODE_CHOICES, default="zero",
                        help="Sidechannel search strategy. zero = CPU smoke wire-format only; "
                             "score_table = reduce precomputed CUDA scorer table; "
                             "gradient/brute_force = CUDA scorer required (raises if invoked here).")
    parser.add_argument("--score-table-npy", type=Path, default=None,
                        help="Required for --search-mode score_table. Shape must be "
                             "(n_frames, candidate_count) for the selected --candidate-radius.")
    parser.add_argument("--score-table-manifest", type=Path, default=None,
                        help="Optional JSON provenance for the CUDA scorer table. Its SHA/path are "
                             "recorded, but exact CUDA auth eval remains required before promotion.")
    parser.add_argument("--candidate-radius", type=int, default=3,
                        help="Canonical score_table candidate grid radius for [y_off, dy, dx].")
    parser.add_argument("--score-step", type=float, default=1.0,
                        help="Y_offset scale factor (default 1.0). Smaller = finer DC adjustment.")
    parser.add_argument("--n-pairs", type=int, default=600,
                        help="Number of frame pairs (PR106 default = 600 → 1200 output frames).")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    pr106_bytes = _read_pr106_bytes(args.pr106_archive)
    pr106_size = len(pr106_bytes)
    print(f"[build-yshift] pr106 archive: {pr106_size} bytes")

    n_frames = args.n_pairs * 2
    print(f"[build-yshift] search-mode={args.search_mode}, n_frames={n_frames}, step={args.score_step}")
    search_diagnostics: dict[str, object] = {}
    if args.search_mode == "score_table":
        if args.score_table_npy is None:
            raise SystemExit("--score-table-npy is required when --search-mode score_table")
        values_int8, search_diagnostics = choose_yshift_candidates_from_score_table_file(
            args.score_table_npy,
            n_frames=n_frames,
            candidate_radius=args.candidate_radius,
        )
        print(
            "[build-yshift] score_table selected "
            f"{search_diagnostics['selected_nonzero_frame_count']} nonzero frame corrections "
            f"from {search_diagnostics['candidate_grid_count']} candidates"
        )
    else:
        search_fn = SEARCH_MODES[args.search_mode]
        values_int8 = search_fn(n_frames)
    if values_int8.dtype != np.int8 or values_int8.shape != (n_frames, 3):
        raise RuntimeError(
            f"search_fn returned bad shape/dtype: shape={values_int8.shape}, dtype={values_int8.dtype}"
        )

    sc01_payload = _encode_sc01_yshift(values_int8, args.score_step)
    sc01_blob = brotli.compress(sc01_payload, quality=11)
    print(f"[build-yshift] SC01 raw payload: {len(sc01_payload)} bytes")
    print(f"[build-yshift] SC01 brotli'd: {len(sc01_blob)} bytes "
          f"({100.0 * len(sc01_blob) / max(len(sc01_payload), 1):.1f}%)")

    new_bin = _build_yshift_archive_bytes(pr106_bytes, sc01_blob)

    # Roundtrip verify before writing zip
    sd_check, lat_check, meta_check, sc_check = parse_yshift_archive(new_bin)
    if sc_check is None:
        raise RuntimeError("sidechannel roundtrip parse failed — sidechannel missing")
    if sc_check["raw"].shape != (n_frames, 3):
        raise RuntimeError(
            f"sidechannel roundtrip shape mismatch: got {sc_check['raw'].shape}, expected ({n_frames}, 3)"
        )
    if not np.array_equal(sc_check["raw"], values_int8):
        raise RuntimeError("sidechannel roundtrip values mismatch (encode/decode bug)")
    if abs(sc_check["step"] - args.score_step) > 1e-6:
        raise RuntimeError(f"sidechannel roundtrip step mismatch: {sc_check['step']} vs {args.score_step}")
    print(f"[build-yshift] roundtrip OK: SC01 raw shape={sc_check['raw'].shape}, "
          f"step={sc_check['step']}, decoder_tensors={len(sd_check)}, "
          f"latents_shape={tuple(lat_check.shape)}, meta={meta_check}")

    archive_path = args.out_dir / "pr106_yshift_sidechannel_archive.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_STORED) as z:  # DETERMINISTIC_ZIP_OK
        zi = zipfile.ZipInfo("0.bin", date_time=(1980, 1, 1, 0, 0, 0))
        zi.compress_type = zipfile.ZIP_STORED
        z.writestr(zi, new_bin)
    archive_size = archive_path.stat().st_size
    pr106_zip_size = args.pr106_archive.stat().st_size
    delta = archive_size - pr106_zip_size
    score_delta = 25.0 * delta / 37545489.0
    print(f"[build-yshift] wrote {archive_path}: {archive_size} bytes")
    print(f"[build-yshift] PR106 zip: {pr106_zip_size} bytes; delta: {delta:+d} ({100*delta/pr106_zip_size:+.3f}%)")
    print(f"[build-yshift] estimated rate-component score Δ vs PR106: {score_delta:+.6f}")

    dispatch_blockers = [
        "requires_exact_cuda_auth_eval_on_built_archive",
        "requires_inflate_runtime_tree_hash_provenance",
        "requires_lane_claim_before_remote_gpu_or_eval_dispatch",
    ]
    if args.n_pairs != 600:
        dispatch_blockers.append("nonstandard_n_pairs_not_contest_promotable")
    if args.search_mode == "zero":
        dispatch_blockers.append("zero_search_is_wire_format_smoke_only")
    if args.search_mode == "score_table" and args.score_table_manifest is None:
        dispatch_blockers.append("missing_cuda_score_table_manifest")

    score_table_metadata: dict[str, object] | None = None
    if args.score_table_npy is not None:
        score_table_manifest: dict[str, object] | None = None
        if args.score_table_manifest is not None:
            score_table_manifest = validate_score_table_manifest(
                args.score_table_manifest,
                score_table_npy=args.score_table_npy,
                pr106_archive=args.pr106_archive,
                n_frames=n_frames,
                candidate_radius=args.candidate_radius,
                candidate_count=int(search_diagnostics.get("candidate_grid_count", 0)),
                score_step=args.score_step,
            )
        score_table_metadata = {
            "score_table_npy_path": str(args.score_table_npy),
            "score_table_npy_bytes": int(args.score_table_npy.stat().st_size),
            "score_table_npy_sha256": sha256_file(args.score_table_npy),
            "score_table_manifest_path": str(args.score_table_manifest) if args.score_table_manifest else None,
            "score_table_manifest_sha256": (
                sha256_file(args.score_table_manifest) if args.score_table_manifest else None
            ),
            "score_table_is_score_claim": False,
            "score_table_required_provenance": (
                "CUDA scorer table must be generated against the exact source archive; "
                "this builder only reduces the table into charged bytes."
            ),
            "score_table_manifest_validated": score_table_manifest is not None,
            "score_table_manifest_schema": (
                score_table_manifest.get("manifest_schema") if score_table_manifest is not None else None
            ),
            "search_diagnostics": search_diagnostics,
        }

    metadata = {
        "manifest_schema": "pr106_yshift_sidechannel_build_metadata_v2",
        "archive_path": str(archive_path),
        "archive_size_bytes": archive_size,
        "archive_sha256": sha256_file(archive_path),
        "source_archive_path": str(args.pr106_archive),
        "source_archive_sha256": sha256_file(args.pr106_archive),
        "pr106_size_bytes": pr106_zip_size,
        "delta_bytes": delta,
        "rate_component_score_delta_vs_pr106": score_delta,
        "search_mode": args.search_mode,
        "score_step": float(args.score_step),
        "n_pairs": int(args.n_pairs),
        "n_frames": int(n_frames),
        "sc01_payload_bytes": len(sc01_payload),
        "sc01_brotli_bytes": len(sc01_blob),
        "outer_dispatch_magic": f"0x{YSHIFT_MAGIC_BYTE:02X}",
        "sc01_mode_id": int(SIDECHANNEL_MODE_Y_SHIFT),
        "score_table": score_table_metadata,
        "tag": "[advisory only]" if args.search_mode == "zero" else "[score-table-reduction]" if args.search_mode == "score_table" else "[design-validation]",
        "council_status": "PROPOSAL — pre-registered at L1; gated on lane_pr106_latent_sidecar empirical",
        "score_claim": False,
        "dispatch_attempted": False,
        "remote_jobs_dispatched": False,
        "ready_for_exact_eval_dispatch": False,
        "dispatch_blockers": dispatch_blockers,
        "next_step": (
            "search-mode=zero is CPU smoke ONLY (all-zero corrections; pure wire-format proof). "
            "search-mode=score_table consumes a precomputed CUDA scorer table and emits "
            "charged sidechannel bytes, but it is not score evidence until exact CUDA "
            "auth eval is run on the built archive. Direct gradient/brute_force modes "
            "must run through a claimed remote lane and remain fail-closed in this local builder."
        ),
    }
    metadata_path = args.out_dir / "build_metadata.json"
    metadata_path.write_text(json_text(metadata), encoding="utf-8")
    print(f"[build-yshift] wrote {metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
