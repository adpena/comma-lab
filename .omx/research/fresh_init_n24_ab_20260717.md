# p0_448 FreSh frequency-shift init A/B — FIRED; both arms FAIL CLOSED at the init seam (2026-07-17)

**Pointer 0.19108 UNMOVED.** This is a duty-to-measure burn-down row (MEANS, not goal progress).
No epochs-to-threshold number exists; the measured outcome is a fail-closed init-time blocker on
BOTH matched arms. Axis: `[macOS-MLX research-signal]`, `score_claim=false`, `promotable=false`.

## What fired (first-ever execution of the 2026-07-12 pre-registered A/B)

The 2026-07-12 build (`init_levers_fresh_metainit_20260712.md`) compiled but never executed
(no-Metal sandbox + governor refuse). This unit reactivated it:

1. **Sealed 20260712 arms are no longer runnable as-is**: `bash experiments/results/fresh_init_n8_control_seal_20260712/launch.sh`
   → trainer admission guard rc=8 `DSL COMPILE REFUSED: missing TAC_DSL_COMPILE_HASH / provenance / launch.sh custody`
   (Catalog #406 landed after those arms were sealed). MEASURED, not inferred.
2. **`v9_cgauge_432` base no longer recompiles**: launcher rc=8
   `#406 ... LawRef compiled value for 'hosc_beta_end' differs from WitnessProgram flag --hosc-beta-end: 10.0 != 3.177`
   (family-level LawRef drift; pre-existing, not introduced here).
3. **Recompiled both arms via the governed launcher** on the nearest compile-clean sibling base
   `v9_cgauge_ideal_mod19` (the sealed arms were mod-dim 19), `--dry-run` emit + `bash launch.sh`
   under `tools/safe_run.py`:
   - epochs 810 (the new CURRICULUM EPOCH-BUDGET FEASIBILITY gate refuses epochs=50 with stage
     starts ≤800; the pre-registered 50-epoch budget was retained as the measurement budget,
     bounded by safe_run `--timeout 2700`).
   - control: `--dsl-lever FreShInitControl --dsl-lever FreShFixedQualitySlice`
   - treatment: `--dsl-lever FreshFrequencyShift --dsl-lever FreShFixedQualitySlice`
   - matched: seed 0, gt `experiments/results/mlx_fleet_gt_cache/gt_n24.npz`, `--num-pairs 8`
     (the pre-registered n8 integration screen; the P0 text says "n24 A/B" — the gt_n24 cache is
     the n24 authority surface, the pre-registered slice subsamples 8 pairs; n64/n600 remain owed
     per the 20260712 protocol), flag-validated 205/205 (control) / 212 (treatment),
     system-admission ADMIT (~51 GiB projected vs 106.9 ceiling).

## Measured rows

| arm | candidates | outcome | elapsed | peak RSS |
|---|---|---|---|---|
| control (`fresh_init_n8_control_20260717`) | 1 (freq_along=8, bias_k=0) | `ValueError: all FreSh candidates are spectrally degenerate` → exit 1, **zero training epochs** | 5.6 s | 1.6 GiB |
| treatment (`fresh_init_n8_treatment_20260717`) | 93 (3 freqs {8, 8√3.2, 8·3.2} × 31 bias widths 0.0..3.0) | identical fail-closed error → exit 1, **zero training epochs** | 8.6 s | 1.7 GiB |

Durable fail-closed receipts (schema `tac.witness_init.fresh_blocker.v1`,
`claim_scope=init_blocker_not_contest_score`), both at seed 0,
git `a935724e4d2bff8b25d4c1c4313c81b0bac87d96` (dirty),
upstream `d46d89155dbf0848e357858c8f62e12ef450a2914ef65814a4359ef6768d2d41`:

- `experiments/results/fresh_init_n8_control_20260717/fresh_init_blocker.json` (candidate_count 1, 2026-07-17T05:07:40Z)
- `experiments/results/fresh_init_n8_treatment_20260717/fresh_init_blocker.json` (candidate_count 93, 2026-07-17T05:09:12Z)

(Both run dirs also hold the governed `launch.sh`, `dsl_provenance.json`, `constants_manifest.json`,
`launch_manifest.json` — gitignored bulk, durable on disk, no `/tmp` evidence.)

## Mechanism (measured to the branch, hypothesis beyond it)

`tac.witness_init.fresh_runtime.run_fresh_initialization_sweep` rejects every candidate before any
training. Per the code, the rejection is one of (per-candidate telemetry is only persisted on
SUCCESS receipts, so the specific reason per candidate is NOT recoverable from the blocker):

1. `zero_boundary` — the cold (epoch-0, pre-structured-prefit) witness's realized-through-R frozen
   SegNet **argmax is a constant map** (no 4-connected boundary anywhere). Consistent with our own
   level-set physics: a cold init sits deep inside one Fisher-flat argmax cell.
2. `zero_non_dc_spectrum` on the **residual-weighted** spectrum — the selection objective weights
   the candidate boundary by the GT **class-1 lane-thin maps** (`lane_thin_weight_map`, ~0.6% area
   support); a cold boundary that never intersects the GT dash support has zero weighted mass even
   if a boundary exists.

Either way the residual-conditioned FreSh objective is **undefined at the cold seam it was designed
for** on this vehicle, and the runtime correctly fails closed (NO-FAKE: it refuses to select on a
degenerate distribution rather than manufacturing a pick).

## Verdict (verdict-scope ladder)

**INSTANCE-level measurement blocker** — arms `v9_cgauge_ideal_mod19` base, n8 slice of gt_n24,
seed 0, both control and treatment: the pre-registered FreSh A/B **cannot produce an
epochs-to-fixed-d_seg number in its current formulation** because both arms abort at init.
This is NOT a family kill of frequency-shift initialization (FreSh, arXiv:2410.05050): the paper's
objective assumes a non-degenerate target/candidate spectrum; our residual-conditioned variant at
the pre-prefit seam is the falsified *formulation surface*, pending the telemetry disambiguation
below. WIN/LOSS per the pre-registered ≥15%-fewer-epochs criterion: **UNMEASURABLE** (right-blocked
at epoch 0, not right-censored).

Note the 20260712 arms would have failed identically had they ever run — the blocker is a property
of the init seam, not of this recompile (same seam, same seed, same candidate law).

## Routing (owed follow-ups; each needs a code-permitted unit — this unit was code-frozen by design)

1. **$0 disambiguator**: persist per-candidate `rejection_reason` + `boundary_pixels` into
   `fresh_init_blocker.json` on failure (telemetry already exists in-memory at
   `fresh_runtime.py:263-364`; it is dropped on the failure path). One measured rerun then splits
   mechanism 1 vs 2.
2. **Re-seam decision** (design change, DSL-held): either (a) move the FreSh seam AFTER the
   structured prefit (`--structured-init` produces a boundary-bearing partition, making the
   spectrum well-defined; changes the "before the structured prefit" pre-registration), or
   (b) fall back to the retained unweighted global-boundary W1 when the residual-weighted mass is
   zero (weakens the residual-conditioning that the 20260712 design argued for). Both need DSL
   lever changes + re-pre-registration; neither may be hand-flagged.
3. Canonical-equations leg: `fresh_frequency_shift_init_v1` needs a superseding/refining anchor row
   recording the degenerate-at-cold-seam domain restriction (registry edit = code; owed with #1/#2,
   hence this memo commits `[no-triality]`).
4. The sealed `fresh_init_n8_{control,treatment}_seal_20260712` dirs are historical provenance
   (pre-#406); do not re-run them.

## P0 ledger

`p0_448_fresh_init_never_fired` → **complete** (the lever FIRED; duty-to-measure satisfied with a
measured fail-closed outcome + named reactivation criteria). No fold-in to any Phase-2 arm config;
nothing to retire beyond the duty row — the lever family stays parked behind follow-ups 1–2.

## STORES CONSULTED

`CLAUDE.md` (NO-FAKE, verdict-scope ladder, governed-launch + admission disciplines) ·
`init_levers_fresh_metainit_20260712.md` (pre-registration; reactivation recipe) ·
`.omx/research/owed16v2_verdict_20260710.json` context (warm-start along26 stays closed) ·
`experiments/results/fresh_init_n8_*_seal_20260712/` (sealed arms) ·
`src/tac/witness_init/{fresh_runtime,fresh_frequency_shift}.py`, trainer FreSh seam
(`train_levelset_witness_realized_through_R_mlx.py:4949-5056`), `src/tac/admission_guard.py`,
`tools/launch_witness_run.py` (all read-only).
