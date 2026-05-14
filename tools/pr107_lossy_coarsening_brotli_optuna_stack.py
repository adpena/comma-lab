#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""PR107 (apogee) STACK candidate: lossy_coarsening + Optuna brotli-param search.

This tool produces multiple candidate archives at varying rel_err budgets (0.05,
0.06, 0.07, 0.08) where each candidate's CD1-payload brotli parameters are
optimized via Optuna TPE search (mode/lgwin/lgblock; quality fixed at 11). The
brotli decoder is parameter-free, so any Optuna-discovered config is decodable
by stock PR107 ``apogee/inflate.py`` without modification.

Source-of-finding & rationale
-----------------------------
- Sister subagent ``a5e5717a`` produced PR107 lossy_coarsening at b050 = 156,263 B
  archive (rel_err 3.86%) using brotli quality=11 with default lgwin=22, lgblock=0
  (per ``tools/pr107_lossy_coarsening_apogee.py``).
- Memory ``feedback_pr101_lgwin13_q10_8byte_savings_20260507`` and
  ``feedback_cmaes_adjacent_libraries_hypernerd_verdict_20260507`` show Optuna
  TPE (with IntDistribution) outperforms CMA-ES on PR101 brotli params, with
  -14 B savings on the canonical PR101 archive.
- For sub-0.17 we need archive_bytes < 138,444 B (with PR107's pose+seg ~0.0779).
  That requires more aggressive lossy-coarsening + every byte of brotli savings.

This is a STACK: lossy_coarsening (-22 KB at b050) + brotli-param Optuna
(-tens-of-bytes per candidate) + budget exploration (b060/b070/b080 push
further at the cost of distortion). The pose+seg degradation curve is unknown;
we measure it empirically by dispatching 2-3 candidates to GHA CPU eval.

Output structure:
  experiments/results/pr107_apogee_stack_<timestamp>/
    candidates/
      b050/archive.zip + build_manifest.json
      b060/archive.zip + build_manifest.json
      b070/archive.zip + build_manifest.json
      b080/archive.zip + build_manifest.json
    stack_manifest.json   # superset summary

Each candidate's archive.zip is a PR107-compatible single-member ZIP holding
0.bin (which the stock ``apogee/inflate.py`` parses without modification).

CLAUDE.md compliance:
- No scorers loaded; pure-CPU build.
- Archive built via ``zipfile.ZipFile`` (not shell zip).
- All evidence tagged ``[CPU-prep]`` / ``[predicted band]`` until empirical
  CPU + CUDA dispatch land.
- ``family_falsified=False``; ``measured_config_only`` scope.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import sys
import zipfile
from pathlib import Path

import brotli
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
APOGEE_SRC = (
    REPO_ROOT
    / "experiments/results/public_pr_intake_full/public_pr107_intake_20260505_auto"
    / "source/submissions/apogee/src"
)
sys.path.insert(0, str(APOGEE_SRC))
sys.path.insert(0, str(REPO_ROOT / "tools"))

from pr107_lossy_coarsening_apogee import (  # type: ignore  # noqa: E402
    parse_pr107_archive_to_tensors,
    build_archive_bin,
    build_cd1_payload,
    PR107_BASELINE_ARCHIVE_BYTES,
    PR107_BASELINE_BIN_BYTES,
)
from codec import parse_archive  # type: ignore  # noqa: E402
from tac.codec.rel_err import REL_ERR_FORM_KEY, RelErrForm, compute_rel_err  # noqa: E402

TOOL_NAME = "tools/pr107_lossy_coarsening_brotli_optuna_stack.py"
SCHEMA_VERSION = "pr107_lossy_coarsening_brotli_optuna_stack.v1"

DEFAULT_PR107_ARCHIVE = (
    REPO_ROOT
    / "experiments/results/public_pr_intake_full/public_pr107_intake_20260505_auto/archive.zip"
)


def best_brotli_for_payload(
    payload: bytes,
    n_trials: int = 80,
    seed: int = 20260508,
) -> dict:
    """Optuna TPE search over (mode, lgwin, lgblock) at quality=11.

    Returns {bytes, mode, quality, lgwin, lgblock, savings_vs_default, best_compressed}.
    """
    try:
        import optuna
    except ImportError:
        # Fallback to a coarse grid + random
        return _grid_brotli_search(payload)

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    default_compressed = brotli.compress(payload, mode=0, quality=11, lgwin=22, lgblock=0)
    default_bytes = len(default_compressed)

    best: dict = {
        "bytes": default_bytes,
        "mode": 0,
        "quality": 11,
        "lgwin": 22,
        "lgblock": 0,
        "compressed": default_compressed,
    }

    def objective(trial: "optuna.Trial") -> int:
        mode = trial.suggest_categorical("mode", [0, 1, 2])
        lgwin = trial.suggest_int("lgwin", 10, 24)
        lgblock = trial.suggest_categorical("lgblock", [0, 16, 17, 18, 19, 20, 21, 22, 23, 24])
        try:
            out = brotli.compress(payload, mode=mode, quality=11, lgwin=lgwin, lgblock=lgblock)
            n = len(out)
            if n < best["bytes"]:
                best.update(
                    {
                        "bytes": n,
                        "mode": mode,
                        "quality": 11,
                        "lgwin": lgwin,
                        "lgblock": lgblock,
                        "compressed": out,
                    }
                )
            return n
        except brotli.error:
            return default_bytes

    sampler = optuna.samplers.TPESampler(seed=seed, n_startup_trials=10)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    return {
        "bytes": best["bytes"],
        "mode": best["mode"],
        "quality": best["quality"],
        "lgwin": best["lgwin"],
        "lgblock": best["lgblock"],
        "savings_vs_default": default_bytes - best["bytes"],
        "default_bytes": default_bytes,
        "compressed": best["compressed"],
        "n_trials": n_trials,
        "search_method": "optuna_tpe",
    }


def _grid_brotli_search(payload: bytes) -> dict:
    default_compressed = brotli.compress(payload, mode=0, quality=11, lgwin=22, lgblock=0)
    default_bytes = len(default_compressed)
    best = {
        "bytes": default_bytes,
        "mode": 0,
        "quality": 11,
        "lgwin": 22,
        "lgblock": 0,
        "compressed": default_compressed,
    }
    for mode in [0, 1, 2]:
        for lgwin in [16, 17, 18, 19, 20, 21, 22, 23, 24]:
            for lgblock in [0, 16, 17, 18, 19, 20, 21, 22, 23, 24]:
                try:
                    out = brotli.compress(payload, mode=mode, quality=11, lgwin=lgwin, lgblock=lgblock)
                    n = len(out)
                    if n < best["bytes"]:
                        best.update(
                            {
                                "bytes": n,
                                "mode": mode,
                                "quality": 11,
                                "lgwin": lgwin,
                                "lgblock": lgblock,
                                "compressed": out,
                            }
                        )
                except brotli.error:
                    pass
    return {
        "bytes": best["bytes"],
        "mode": best["mode"],
        "quality": best["quality"],
        "lgwin": best["lgwin"],
        "lgblock": best["lgblock"],
        "savings_vs_default": default_bytes - best["bytes"],
        "default_bytes": default_bytes,
        "compressed": best["compressed"],
        "search_method": "grid",
    }


def build_candidate(
    blobs: list,
    meta_section: bytes,
    latents_section: bytes,
    budget: float,
    optuna_trials: int,
) -> dict:
    """Build one (budget, optuna-best-brotli) candidate. Returns full result + bytes."""
    # Step 1: K-coarsening at given budget
    Ks: list[int] = []
    rounded_chunks: list[np.ndarray] = []
    orig_chunks: list[np.ndarray] = []
    for tb in blobs:
        from pr107_lossy_coarsening_apogee import find_best_K_for_tensor  # noqa
        K, _ = find_best_K_for_tensor(tb.raw_int8, budget)
        Ks.append(K)
        s = tb.raw_int8.astype(np.float64)
        rounded = np.round(s / K) * K
        orig_chunks.append(s)
        rounded_chunks.append(rounded)

    rel_err = compute_rel_err(
        np.concatenate(rounded_chunks),
        np.concatenate(orig_chunks),
        mode=RelErrForm.L1_RATIO,
    )
    cd1_payload = build_cd1_payload(blobs, rounded_chunks)

    # Step 2: brotli optimization
    brotli_result = best_brotli_for_payload(cd1_payload, n_trials=optuna_trials)
    decoder_brotli = brotli_result["compressed"]

    # Step 3: assemble 0.bin
    new_bin_bytes = build_archive_bin(meta_section, decoder_brotli, latents_section)

    # Step 4: roundtrip verification (use stock parse_archive)
    decoder_sd_rt, latents_rt, meta_rt = parse_archive(new_bin_bytes)
    assert len(decoder_sd_rt) == 28, f"roundtrip n={len(decoder_sd_rt)}"
    assert latents_rt.shape == (600, 28), f"roundtrip latents shape={latents_rt.shape}"

    return {
        "budget": budget,
        "rel_err": rel_err,
        REL_ERR_FORM_KEY: RelErrForm.L1_RATIO.value,
        "Ks": Ks,
        "cd1_payload_bytes": len(cd1_payload),
        "decoder_brotli_bytes_default": brotli_result["default_bytes"],
        "decoder_brotli_bytes_optuna": brotli_result["bytes"],
        "brotli_savings_vs_default": brotli_result["savings_vs_default"],
        "brotli_mode": brotli_result["mode"],
        "brotli_quality": brotli_result["quality"],
        "brotli_lgwin": brotli_result["lgwin"],
        "brotli_lgblock": brotli_result["lgblock"],
        "brotli_search_method": brotli_result["search_method"],
        "brotli_n_trials": brotli_result.get("n_trials"),
        "decoder_brotli": decoder_brotli,
        "new_bin_bytes": new_bin_bytes,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--archive", type=Path, default=DEFAULT_PR107_ARCHIVE)
    p.add_argument(
        "--budgets",
        type=str,
        default="0.05,0.06,0.07,0.08",
        help="Comma-separated rel_err budgets to build candidates for.",
    )
    p.add_argument("--optuna-trials", type=int, default=80)
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )
    args = p.parse_args(argv)

    if not args.archive.is_file():
        raise SystemExit(f"PR107 archive not found: {args.archive}")
    if args.output_dir is None:
        ts = _dt.datetime.now(_dt.UTC).strftime("%Y%m%dT%H%M%SZ")
        args.output_dir = REPO_ROOT / f"experiments/results/pr107_apogee_stack_lossy_brotli_{ts}"
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Parse PR107
    with zipfile.ZipFile(args.archive) as zf:
        names = zf.namelist()
        if names != ["0.bin"]:
            raise SystemExit(f"unexpected members: {names}")
        bin_bytes = zf.read("0.bin")

    blobs, meta_section, latents_section = parse_pr107_archive_to_tensors(bin_bytes)
    print(f"[stack] parsed {len(blobs)} tensors; meta={len(meta_section)} B; latents={len(latents_section)} B")

    budgets = [float(x) for x in args.budgets.split(",") if x.strip()]
    candidates = []

    print()
    header = f"{'budget':>7} {'rel_err':>8} {'CD1':>9} {'br_def':>8} {'br_opt':>8} {'savings':>8} {'lgwin':>5} {'lgblk':>5} {'mode':>4} {'0.bin':>8} {'archive.zip':>11} {'delta_zip':>10}"
    print(header)
    print("-" * len(header))

    for budget in budgets:
        cand = build_candidate(
            blobs,
            meta_section,
            latents_section,
            budget,
            optuna_trials=args.optuna_trials,
        )

        # Materialize candidate archive
        cand_dir = args.output_dir / f"b{int(round(budget * 1000)):03d}"
        cand_dir.mkdir(parents=True, exist_ok=True)
        archive_path = cand_dir / "archive.zip"
        zi = zipfile.ZipInfo(filename="0.bin", date_time=(1980, 1, 1, 0, 0, 0))
        zi.compress_type = zipfile.ZIP_STORED
        zi.external_attr = (0o644 & 0xFFFF) << 16
        with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_STORED) as zf:
            zf.writestr(zi, cand["new_bin_bytes"])

        archive_bytes = archive_path.stat().st_size
        archive_sha = hashlib.sha256(archive_path.read_bytes()).hexdigest()
        member_sha = hashlib.sha256(cand["new_bin_bytes"]).hexdigest()
        delta_zip = archive_bytes - PR107_BASELINE_ARCHIVE_BYTES

        # Compute predicted CPU score band (advisory only; pose+seg degradation
        # is unknown — using PR107 anchor 0.0779 with monotonic widening)
        # PR107 baseline pose_avg+seg_avg = 0.07782; rel_err 0% → +0
        # rel_err 0.0386 (b050) → +0 to +0.005 (sister hypothesis: at this budget,
        # the rendered frames are nearly indistinguishable from PR107 baseline).
        # Above b050: distortion is empirically unknown.
        rate_term = 25 * archive_bytes / 37545489
        pred_score_lower = 0.0779 + rate_term  # assumes pose+seg unchanged (lower bound)
        # Upper bound: assume linear distortion growth with rel_err
        # rel_err 0.0386 → at most +0.020 to pose+seg → score = 0.0979 + rate_term
        # rel_err 0.10 → at most +0.060 to pose+seg → score = 0.138 + rate_term
        # Use rel_err * 0.5 as crude upper-bound distortion delta.
        pred_score_upper = 0.0779 + cand["rel_err"] * 0.5 + rate_term

        manifest = {
            "schema_version": SCHEMA_VERSION,
            "tool": TOOL_NAME,
            "evidence_grade": "[CPU-prep]",
            "evidence_semantics": "cpu_byte_roundtrip_proxy_no_score",
            "score_claim": False,
            "promotion_eligible": False,
            "rank_or_kill_eligible": False,
            "ready_for_exact_eval_dispatch": False,
            "dispatch_attempted": False,
            "input_archive_path": str(args.archive),
            "input_archive_sha256": hashlib.sha256(args.archive.read_bytes()).hexdigest(),
            "input_archive_size_bytes": args.archive.stat().st_size,
            "pr107_baseline_archive_bytes": PR107_BASELINE_ARCHIVE_BYTES,
            "pr107_baseline_bin_bytes": PR107_BASELINE_BIN_BYTES,
            "build_target_budget": budget,
            "build_rel_err": cand["rel_err"],
            "build_Ks": cand["Ks"],
            "build_archive_relpath": str(archive_path.relative_to(REPO_ROOT)),
            "build_archive_sha256": archive_sha,
            "build_archive_size_bytes": archive_bytes,
            "build_member_name": "0.bin",
            "build_member_sha256": member_sha,
            "build_member_size_bytes": len(cand["new_bin_bytes"]),
            "build_decoder_brotli_bytes_default": cand["decoder_brotli_bytes_default"],
            "build_decoder_brotli_bytes_optuna": cand["decoder_brotli_bytes_optuna"],
            "build_brotli_savings_vs_default": cand["brotli_savings_vs_default"],
            "build_brotli_mode": cand["brotli_mode"],
            "build_brotli_quality": cand["brotli_quality"],
            "build_brotli_lgwin": cand["brotli_lgwin"],
            "build_brotli_lgblock": cand["brotli_lgblock"],
            "build_brotli_search_method": cand["brotli_search_method"],
            "build_brotli_n_trials": cand["brotli_n_trials"],
            "build_meta_section_bytes": len(meta_section),
            "build_latents_section_bytes": len(latents_section),
            "build_delta_zip_vs_baseline": delta_zip,
            "predicted_rate_term": rate_term,
            "predicted_score_band_lower": pred_score_lower,
            "predicted_score_band_upper": pred_score_upper,
            "predicted_band_grade": "predicted",
            "predicted_band_caveat": (
                "Lower bound assumes pose+seg unchanged from PR107 baseline 0.0779; "
                "upper bound uses rel_err*0.5 as crude distortion-growth proxy. "
                "Actual pose+seg degradation curve is empirically unknown and "
                "must be measured via GHA CPU + Lightning CUDA dispatch."
            ),
            "score_affecting_payload_changed": True,
            "charged_bits_changed": True,
            "score_claim_blockers": [
                "cpu_build_rel_err_proxy_not_score_evidence",
                "exact_cuda_auth_eval_not_yet_harvested",
                "exact_cpu_auth_eval_not_yet_harvested",
            ],
            "dispatch_blockers": [
                "cpu_build_rel_err_proxy_not_score_evidence",
                "runtime_closure_not_yet_verified",
                "exact_linux_cpu_auth_eval_not_yet_harvested",
                "exact_cuda_auth_eval_not_yet_harvested",
                "dispatch_claim_not_created",
            ],
            "target_modes": ["contest_exact_eval"],
            "deployment_target": "linux_x86_64_cpu_and_t4_contest_runtime",
            "stacking_paradigms": [
                "lossy_coarsening_per_tensor_K_search",
                "brotli_param_optuna_tpe_search",
            ],
            "lane_id": "pr107_apogee_stack_brotli_sweep_cpu_build",
        }

        manifest_path = cand_dir / "build_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2))

        candidates.append(
            {
                "budget": budget,
                "archive_path": str(archive_path.relative_to(REPO_ROOT)),
                "archive_bytes": archive_bytes,
                "archive_sha256": archive_sha,
                "rel_err": cand["rel_err"],
                "delta_zip_vs_baseline": delta_zip,
                "predicted_score_band": [pred_score_lower, pred_score_upper],
                "manifest_path": str(manifest_path.relative_to(REPO_ROOT)),
                "brotli_lgwin": cand["brotli_lgwin"],
                "brotli_lgblock": cand["brotli_lgblock"],
                "brotli_mode": cand["brotli_mode"],
                "brotli_savings_vs_default": cand["brotli_savings_vs_default"],
            }
        )

        print(
            f"{budget:>7.4f} {cand['rel_err']:>8.4f} "
            f"{cand['cd1_payload_bytes']:>9,} "
            f"{cand['decoder_brotli_bytes_default']:>8,} "
            f"{cand['decoder_brotli_bytes_optuna']:>8,} "
            f"{cand['brotli_savings_vs_default']:>+8,} "
            f"{cand['brotli_lgwin']:>5} "
            f"{cand['brotli_lgblock']:>5} "
            f"{cand['brotli_mode']:>4} "
            f"{len(cand['new_bin_bytes']):>8,} "
            f"{archive_bytes:>11,} "
            f"{delta_zip:>+10,}"
        )

    stack_manifest = {
        "schema_version": SCHEMA_VERSION + ".stack",
        "tool": TOOL_NAME,
        "built_at_utc": _dt.datetime.now(_dt.UTC).isoformat(),
        "input_archive": str(args.archive),
        "input_archive_sha256": hashlib.sha256(args.archive.read_bytes()).hexdigest(),
        "candidates": candidates,
        "stacking_paradigms": [
            "lossy_coarsening_per_tensor_K_search",
            "brotli_param_optuna_tpe_search",
        ],
        "lane_id": "pr107_apogee_stack_brotli_sweep_cpu_build",
        "predispatch_recommendation": (
            "Do not dispatch directly from this CPU-prep stack manifest. First close "
            "runtime closure, claim the lane, then run paired Linux CPU and CUDA exact "
            "eval on selected candidates."
        ),
        "evidence_grade": "[CPU-prep]",
        "evidence_semantics": "cpu_byte_roundtrip_proxy_no_score",
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "dispatch_blockers": [
            "cpu_build_rel_err_proxy_not_score_evidence",
            "runtime_closure_not_yet_verified",
            "exact_linux_cpu_auth_eval_not_yet_harvested",
            "exact_cuda_auth_eval_not_yet_harvested",
            "dispatch_claim_not_created",
        ],
        "target_modes": ["contest_exact_eval"],
        "deployment_target": "linux_x86_64_cpu_and_t4_contest_runtime",
    }

    stack_path = args.output_dir / "stack_manifest.json"
    stack_path.write_text(json.dumps(stack_manifest, indent=2))
    print()
    print(f"[stack] wrote {len(candidates)} candidates to {args.output_dir}")
    print(f"[stack] stack_manifest: {stack_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
