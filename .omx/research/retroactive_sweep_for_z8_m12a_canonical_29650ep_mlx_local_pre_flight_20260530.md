# Retroactive sweep for Z8 M12a canonical 29,650-epoch MLX-LOCAL pre-flight verification — 2026-05-30

**Per Catalog #348 NEW-GATE-LANDING canonical retroactive sweep contract.**

## Bug-class symptom signature

Z8 M12a operator-attended 29,650-epoch MLX-LOCAL FULL RUN admissibility check
BEFORE committing to multi-hour MLX run. The canonical pre-flight verification
composes the 4 canonical primitives landed earlier today (alaska / m9-v3 PR95
curriculum / Yousfi-Tier-1 pose-axis / Z7+Z8 mamba2_adapter rewire) into ONE
end-to-end pre-flight that proves:

1. PR95 8-stage 29,650-epoch curriculum advances per L14 spec.
2. Muon optimizer fires final-stage-only per L15 spec.
3. Per-stage hparam transitions (cat_lambda + cat_sigma) advance per L16+L17.
4. EMA + canonical Provenance preserved per `[[macOS-MLX research-signal]]`.
5. MLX → numpy state_dict export bridge byte-stable per Catalog #1265.

The bug class this pre-flight prevents: dispatching 29,650 epochs only to
discover at hour 18 that the curriculum did not advance, Muon never fired, or
the export bridge produced different bytes on two consecutive writes. Each
such failure mode costs 8-20h wall-clock vs the pre-flight's 5.2s budget at
N=100 scaled.

## Pre-fix window

**2026-05-30 ~20:27 UTC** — operator binding correction: *"30k plus epoch
runs all for free on MLX for proving and if frontier and portable via numpy
then good. MLX to the full extent possible."*

Before this pre-flight, no single command sequence verified the canonical
composition of the 4 sister-landed primitives end-to-end at scaled epoch
budget.

## Historical KILL / DEFER / FALSIFY search results

Query: `grep -r "z8.*29650\|m12a\|pr95.*curriculum.*mlx" .omx/research/ --include="*killed*.md" --include="*defer*.md" --include="*falsif*.md" 2>/dev/null`

No historical KILL / DEFER / FALSIFY verdicts on the Z8 M12a canonical
29,650-epoch MLX-LOCAL pre-flight path. The Z8 substrate has been an active
build target since 2026-05-29 per
`feedback_z8_hierarchical_predictive_coding_binding_first_active_build_target_yousfi_grounded_20260529.md`;
the canonical curriculum + Muon helpers landed today via m9-v3 + Z7+Z8 rewire
landings.

## Per-finding RE-EVAL priority assignment

None — no historical verdicts to re-evaluate. The pre-flight is a NEW
canonical composition check, not a re-litigation of prior decisions.

## 4-field contract closure

- **bug-class symptom signature**: 29,650-epoch MLX-LOCAL FULL RUN admissibility check (above)
- **pre-fix window**: 2026-05-30 ~20:27 UTC operator binding correction (above)
- **historical-KILL/DEFER/FALSIFY search results**: 0 hits (above)
- **per-finding RE-EVAL priority assignment**: N/A — no historical verdicts (above)

## Sister relationships

- Catalog #325 per-substrate optimal-form symposium evidence — Z8 carries
  Phase 2 symposium anchor per the binding-first build target memo.
- Catalog #344 canonical equations registry — anchors appended to
  `pr95_family_l14_eight_stage_29650_epoch_curriculum_v1` (anchor count
  1→2) + `pr95_family_l15_muon_optimizer_final_stage_only_v1` (anchor count
  1→2). Catalog #371 auto-recalibrator may refit posteriors.
- Catalog #313 probe outcomes ledger — PROCEED advisory 14-day verdict.
- Catalog #192 macOS-MLX research-signal non-promotability — preserved across all artifacts.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
