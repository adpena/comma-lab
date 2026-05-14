# /// script
# requires-python = ">=3.11"
# ///
"""F3 GTScorerCache trainer-side wire-in actionability checker.

Canonical tool promoted from the F3-BACKPORT-WAVE subagent's defensive reproducer
at `.omx/tmp/f3_backport_7_trainers_reproducer.py` per F3-BACKPORT-WAVE
operator-routable decision #4 + the premise-verification-before-edit pattern
(`feedback_prompt_premise_verification_before_edit_pattern_20260514.md`).

Classifies each `experiments/train_substrate_<id>.py` trainer into one of:

- `ALREADY_WIRED` — `build_gt_scorer_cache` / `build_optimized_training_context`
  AND `gt_cache.lookup(` AND `gt_pose_batch=` kwarg threaded into the loss call.
  F3 backport is done; no work.
- `NEEDS_F3_BACKPORT` — RESERVED `--enable-gt-scorer-cache` flag declared in the
  Catalog #151 manifest BUT no cache build / lookup / kwargs threaded. This is
  the F3-BACKPORT-WAVE-V2 actionable set.
- `FULL_MAIN_STUB_NO_SCORER_HOT_LOOP` — `_full_main` raises NotImplementedError;
  no hot loop to wire. DEFERRED-pending-stub-unlock.
- `D1_SEGNET_SIDECAR_NO_HOT_LOOP` — D1-class trainer; SegNet-only sidecar with
  `compute_logit_margin_map(...)` and no PoseNet hot loop. Structurally moot.
- `Z3_V1_RATE_ONLY_SCORERS_IDENTITY` — Z3-class trainer; `torch.nn.Identity()`
  placeholder scorers + `gt_pair=None`. F3 dead-code until v2.
- `RESERVED_FLAG_BUT_NO_SCORER_HOT_LOOP` — declares flag but doesn't load
  scorers at all. Investigate.
- `UNKNOWN_STATE_INVESTIGATE` — heuristics did not match; surface for review.
- `NO_TRAINER_FILE` — file missing.

Usage:

    # Default: scan all known substrate trainers in `experiments/`
    .venv/bin/python tools/check_f3_trainer_actionable.py

    # Target a specific trainer set:
    .venv/bin/python tools/check_f3_trainer_actionable.py --target sane_hnerv pr101_lc_v2_clone_enhanced_curriculum

    # JSON output for programmatic consumption:
    .venv/bin/python tools/check_f3_trainer_actionable.py --json

Exit code:
- 0: every trainer classified into a terminal verdict (no UNKNOWN/NO_TRAINER_FILE)
- 1: at least one trainer is in an unknown state and needs operator review

Per CLAUDE.md "NEVER invent CLI flags" non-negotiable + the
premise-verification-before-edit pattern: this tool is the operator's smoke
gate before any future F3-BACKPORT-WAVE subagent runs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# The canonical set of substrate trainers known to declare or potentially declare
# the RESERVED `--enable-gt-scorer-cache` flag in their Catalog #151 manifest.
# This is the universe of trainers F3 wire-in could apply to.
# Sourced from `experiments/train_substrate_*.py` listing 2026-05-14.
KNOWN_SUBSTRATE_TRAINERS = [
    # F3-BACKPORT-WAVE-V2 actual actionable set:
    "sane_hnerv",
    "pr101_lc_v2_clone_enhanced_curriculum",
    "balle_renderer",
    "tc_nerv",
    "block_nerv",
    "ff_nerv",
    "ds_nerv",
    "hi_nerv",
    "cool_chic",
    "self_compress_nn",
    "hybrid_renderer_residual",
    "vq_vae",
    "pretrained_driving_prior",
    # F3-BACKPORT-WAVE original 7 (predecessor-verified non-actionable):
    "d1_segnet_margin_polytope",
    "d4_wyner_ziv_frame_0",
    "c1_world_model_foveation",
    "c6_e4_mdl_ibps",
    "z3_balle_hyperprior_bolton",
    "z4_cooperative_receiver_loss",
    "z5_predictive_coding_world_model",
    # Other known substrate trainers
    "siren",
    "time_traveler_l5_autonomy",
]


def _has_token(text: str, *tokens: str) -> bool:
    return any(t in text for t in tokens)


def _substrate_accepts_f3_kwargs(trainer_id: str) -> bool:
    """Check whether the trainer's substrate-side score_aware_loss accepts
    F3 kwargs (gt_pose_batch / gt_seg_batch / gt_seg_already_probs).

    Without substrate-side acceptance, the F3 cache wire-in at trainer-side
    is dead code (the cache builds but the loss call discards the kwargs).
    """
    candidate_dirs = [
        REPO_ROOT / "src" / "tac" / "substrates" / trainer_id,
        # Some trainers map to a differently-named substrate package
        REPO_ROOT / "src" / "tac" / "substrates" / trainer_id.replace("_renderer", ""),
    ]
    for d in candidate_dirs:
        loss_file = d / "score_aware_loss.py"
        if loss_file.is_file():
            text = loss_file.read_text(encoding="utf-8", errors="replace")
            return "gt_pose_batch" in text
    return False


def _has_smoke_only_scorers(text: str) -> bool:
    """Detect synthetic smoke scorers (e.g. _SmokeSegScorer / _SmokePoseScorer)
    that bypass the contest scorers — F3 cache is structurally moot when
    the trainer's only real hot loop runs against synthetic scorers.
    """
    return _has_token(text, "_SmokeSegScorer", "_SmokePoseScorer")


def _has_tier1_recommended_flags(text: str) -> bool:
    """Check whether the trainer declares Tier-1 RESERVED optimization flags
    (autocast/torch_compile/gt_scorer_cache). Trainers missing all three need
    flag-declaration ALSO, not just wire-in.
    """
    return (
        _has_token(text, "enable-autocast-fp16", "enable_autocast_fp16")
        or _has_token(text, "enable-torch-compile", "enable_torch_compile")
        or _has_token(text, "enable-gt-scorer-cache", "enable_gt_scorer_cache")
    )


def _classify(trainer_id: str) -> dict[str, object]:
    path = REPO_ROOT / "experiments" / f"train_substrate_{trainer_id}.py"
    if not path.is_file():
        return {"trainer_id": trainer_id, "verdict": "NO_TRAINER_FILE", "path": None}

    text = path.read_text(encoding="utf-8", errors="replace")

    # F3 wire-in tokens (the canonical pattern)
    has_cache_build = _has_token(
        text,
        "build_gt_scorer_cache",
        "build_optimized_training_context",
    )
    has_cache_lookup = _has_token(text, "gt_cache.lookup")
    has_kwargs_threaded = _has_token(text, "gt_pose_batch=")
    # The RESERVED flag declaration tells us this trainer's Catalog #151
    # manifest claims F3 cache availability
    has_reserved_marker = _has_token(text, "enable-gt-scorer-cache", "enable_gt_scorer_cache")
    has_full_stub = _has_token(text, "raise NotImplementedError")
    # Real scorer hot loop indicator
    has_scorer_hot_loop = _has_token(
        text,
        "score_pair_components_dispatch",
        "score_pair_components(",
        "loss_fn(",
    )
    has_pose_seg_load = _has_token(
        text, "load_differentiable_scorers", "load_default_scorers"
    )
    has_identity_scorer = _has_token(
        text,
        "pose_scorer = torch.nn.Identity()",
        "seg_scorer = torch.nn.Identity()",
    )
    is_d1_sidecar = "compute_logit_margin_map" in text and trainer_id.startswith("d1")
    has_smoke_only_scorers = _has_smoke_only_scorers(text)
    substrate_accepts_kwargs = _substrate_accepts_f3_kwargs(trainer_id)
    has_tier1_flags = _has_tier1_recommended_flags(text)

    # Detect full-main-is-stub-but-smoke-has-hot-loop pattern (pr101_lc_v2 type):
    # smoke runs real loop with synthetic smoke scorers; full is a stub.
    smoke_with_synthetic_full_stub = (
        has_full_stub and has_smoke_only_scorers and has_scorer_hot_loop
    )

    # Verdict logic
    if has_cache_build and has_cache_lookup and has_kwargs_threaded:
        verdict = "ALREADY_WIRED"
    elif is_d1_sidecar:
        verdict = "D1_SEGNET_SIDECAR_NO_HOT_LOOP"
    elif has_identity_scorer:
        verdict = "Z3_V1_RATE_ONLY_SCORERS_IDENTITY"
    elif smoke_with_synthetic_full_stub:
        # Trainer's only hot loop is smoke + synthetic scorers; F3 dead code.
        verdict = "SMOKE_SYNTHETIC_SCORERS_FULL_STUB"
    elif has_full_stub and not has_scorer_hot_loop:
        verdict = "FULL_MAIN_STUB_NO_SCORER_HOT_LOOP"
    elif has_full_stub and has_scorer_hot_loop and not has_pose_seg_load:
        # _full_main stub + smoke runs but doesn't load real scorers (e.g. time_traveler)
        verdict = "FULL_MAIN_STUB_SMOKE_NO_REAL_SCORERS"
    elif (
        has_reserved_marker
        and has_scorer_hot_loop
        and has_pose_seg_load
        and substrate_accepts_kwargs
        and not has_full_stub
    ):
        # The CANONICAL actionable case
        verdict = "NEEDS_F3_BACKPORT"
    elif (
        has_scorer_hot_loop
        and has_pose_seg_load
        and substrate_accepts_kwargs
        and not has_tier1_flags
    ):
        # Needs flag-declaration ALSO (e.g. vq_vae)
        verdict = "NEEDS_F3_BACKPORT_PLUS_TIER1_FLAGS"
    elif has_scorer_hot_loop and has_pose_seg_load and not substrate_accepts_kwargs:
        # Trainer-side ready but substrate-side score_aware_loss doesn't accept F3 kwargs
        verdict = "NEEDS_SUBSTRATE_F3_WIRE_IN"
    elif has_reserved_marker and not has_scorer_hot_loop:
        verdict = "RESERVED_FLAG_BUT_NO_SCORER_HOT_LOOP"
    elif has_reserved_marker:
        verdict = "RESERVED_FLAG_NOT_CONSUMED"
    else:
        verdict = "UNKNOWN_STATE_INVESTIGATE"

    return {
        "trainer_id": trainer_id,
        "verdict": verdict,
        "path": str(path.relative_to(REPO_ROOT)),
        "has_cache_build": has_cache_build,
        "has_cache_lookup": has_cache_lookup,
        "has_kwargs_threaded": has_kwargs_threaded,
        "has_reserved_marker": has_reserved_marker,
        "has_full_stub": has_full_stub,
        "has_scorer_hot_loop": has_scorer_hot_loop,
        "has_pose_seg_load": has_pose_seg_load,
        "has_identity_scorer": has_identity_scorer,
        "has_smoke_only_scorers": has_smoke_only_scorers,
        "substrate_accepts_kwargs": substrate_accepts_kwargs,
        "has_tier1_flags": has_tier1_flags,
    }


def _format_table(rows: list[dict[str, object]]) -> str:
    cols = [
        "trainer_id",
        "verdict",
        "cache_build",
        "cache_lookup",
        "kwargs",
        "reserved",
        "stub",
        "hot_loop",
        "pose_seg_load",
    ]
    widths = {c: len(c) for c in cols}
    for r in rows:
        for c in cols:
            if c == "trainer_id" or c == "verdict":
                v = str(r.get(c, ""))
            else:
                # Map shortened col to full key
                key_map = {
                    "cache_build": "has_cache_build",
                    "cache_lookup": "has_cache_lookup",
                    "kwargs": "has_kwargs_threaded",
                    "reserved": "has_reserved_marker",
                    "stub": "has_full_stub",
                    "hot_loop": "has_scorer_hot_loop",
                    "pose_seg_load": "has_pose_seg_load",
                }
                v = str(r.get(key_map[c], ""))
            widths[c] = max(widths[c], len(v))

    def _line(cells: list[str]) -> str:
        return "| " + " | ".join(c.ljust(widths[col]) for c, col in zip(cells, cols)) + " |"

    out = []
    out.append(_line(cols))
    out.append("|" + "|".join("-" * (widths[c] + 2) for c in cols) + "|")
    for r in rows:
        cells = []
        for c in cols:
            if c in ("trainer_id", "verdict"):
                cells.append(str(r.get(c, "")))
            else:
                key_map = {
                    "cache_build": "has_cache_build",
                    "cache_lookup": "has_cache_lookup",
                    "kwargs": "has_kwargs_threaded",
                    "reserved": "has_reserved_marker",
                    "stub": "has_full_stub",
                    "hot_loop": "has_scorer_hot_loop",
                    "pose_seg_load": "has_pose_seg_load",
                }
                cells.append(str(r.get(key_map[c], "")))
        out.append(_line(cells))
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target",
        nargs="+",
        default=None,
        help="Specific trainer ids to classify (default: all known substrates)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Scan every experiments/train_substrate_*.py (auto-discover)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON instead of a markdown table",
    )
    args = parser.parse_args(argv)

    if args.all:
        # Auto-discover
        trainer_files = sorted(
            (REPO_ROOT / "experiments").glob("train_substrate_*.py")
        )
        targets = [f.stem.replace("train_substrate_", "") for f in trainer_files]
    elif args.target:
        targets = args.target
    else:
        targets = KNOWN_SUBSTRATE_TRAINERS

    rows = [_classify(tid) for tid in targets]

    if args.json:
        print(json.dumps(rows, indent=2))
    else:
        print(_format_table(rows))
        print()
        # Summary buckets
        buckets: dict[str, list[str]] = {}
        for r in rows:
            buckets.setdefault(str(r["verdict"]), []).append(str(r["trainer_id"]))
        print("# Verdict summary")
        for verdict in sorted(buckets):
            ids = buckets[verdict]
            print(f"- **{verdict}** ({len(ids)}): {', '.join(ids)}")
        actionable = buckets.get("NEEDS_F3_BACKPORT", [])
        print()
        print(f"# Actionable F3 backport targets: {len(actionable)}")
        for t in actionable:
            print(f"- {t}")

    # Exit code: 0 iff no UNKNOWN/NO_TRAINER_FILE
    unknown = [
        r for r in rows
        if r["verdict"] in ("UNKNOWN_STATE_INVESTIGATE", "NO_TRAINER_FILE")
    ]
    return 1 if unknown else 0


if __name__ == "__main__":
    sys.exit(main())
