# Lane 12 NeRV full-CUDA dispatch REFUSED — three blockers, three reactivation paths

**Status:** REFUSED dispatch (no GPU spend, no instance provisioned)
**Subagent:** claude:subagent-nerv-full-cuda
**Date:** 2026-05-02 ~12:55 UTC
**Pinned commit:** `a0184dee339c0c6364474d5a9d5c30f9912e1330`
**Anchor under stack consideration:** C-067 (`226475de42ec00d66287a39f98fe6d2eb0464b90738714e5ef05fa4ee8efb38a`, 276,214 bytes, score 0.31561703 [contest-CUDA T4 A++], `experiments/results/lightning_batch/exact_eval_c063_fixedslice_equiv_t4_20260502T0855Z/archive.zip`)
**Cross-refs:** dispatch claim row appended to `.omx/state/active_lane_dispatch_claims.md` 2026-05-02T12:55:00Z; dispatch plan at `project_lane_12_nerv_dispatch_plan_20260430.md`; council prescription at `project_grand_council_FIELDS_MEDAL_shannon_floor_obsession_20260501.md`.

## TL;DR

Subagent was tasked to train Lane 12 NeRV mask codec to full CUDA convergence on H100/A100 SXM, then stack onto C-067 by replacing the 219,472-byte PR67 mask segment. After pre-flight (no GPU spend, ~10 min wall-clock) the subagent identified **three structural blockers** that make a sub-frontier landing infeasible inside the 8h / $20 mandate:

1. **Convergence math kills the score**: Phase F empirical (1400 of 60000 steps, CPU partial) showed xent 0.59 → 0.02 with 2.0% argmax disagreement. Linear extrapolation lands at ~0.5% final disagreement. Distortion penalty `100 × 0.005 = +0.5` score swamps the `-0.130` rate save → **predicted stacked score ~0.74, REGRESSION vs C-067 0.316**.
2. **C-067 anatomy mismatch**: C-067 uses PR67's single-blob `p` member with hardcoded `mask_len = 219_472` byte split inside `submissions/robust_current/unpack_renderer_payload.py:482`. The existing NeRV decoder path in `inflate_renderer.py:996-1053` triggers on a `masks.nrv` archive member, not on a slice of the `p` blob. NeRV-onto-C-067 requires a new parser (`_try_parse_public_pr67_NERV_qzs3_qp1_payload`) OR a new MAGIC-prefixed renderer-payload schema that re-exports `masks.nrv` after split — neither exists.
3. **Gate artifacts missing**: `scripts/remote_lane_nerv.sh` requires `.omx/state/lane12_nerv_l2_clearance.json` (L2 retraining clearance gate, lines 144-187), `ALPHA_PRIMITIVE_CONTRACT` (lines 220-253), and (when `RUN_AUTH_EVAL=1`) `ALPHA_GEO_PROVENANCE` + `POSE_REGEN_PROVENANCE`. None exist. The script will fail-loud at Stage 0c/1.

The dispatch as-prescribed cannot ship a sub-frontier candidate within the 8h budget. Refusing dispatch honors CLAUDE.md non-negotiables: "Score target — auth>1.0 UNACCEPTABLE", "Forbidden score claims", "do NOT cycle hyperparameter tuning".

## Pre-flight evidence (no GPU spend)

### Convergence math

Source: `project_lane_12_nerv_dispatch_plan_20260430.md` Phase F empirical table.

| metric | partial 1400-step (CPU) | linear extrapolation 60K | required for sub-frontier |
|---|---:|---:|---:|
| xent loss | 0.59 → 0.02 | → ~0.005 | ≤0.001 |
| argmax disagreement | 2.0% | ~0.5% | <0.05% |
| distortion contribution `100×disagreement` | +2.0 | +0.5 | <+0.05 |

C-067 baseline rate save by replacing PR67 mask:
- PR67 mask.obu.br segment: -219,472 bytes
- NeRV NRV2 payload (Phase F empirical): +23,594 bytes
- Header overhead (best case, raw-codec MAGIC schema): +~250 bytes
- Net: -195,628 bytes → -0.130 score

**Predicted stacked score (linear extrapolation):**
`0.31561703 - 0.130 + 0.5 = ~0.685` → REGRESSION (worse than C-067)

**Predicted stacked score (optimistic 0.05% disagreement):**
`0.31561703 - 0.130 + 0.05 = ~0.236` → SUB-FRONTIER

The gap between the two scenarios is whether NeRV converges past 0.05% disagreement. Phase F's xent → 0.005 floor maps to ~0.5%, NOT 0.05%. Achieving the optimistic scenario requires a recipe change (different NeRV architecture, different loss, hard-pair augmentation, or bigger NeRV with 50K+ params at +30KB rate cost).

### C-067 anatomy

`/Users/adpena/Projects/pact/experiments/results/lightning_batch/exact_eval_c063_fixedslice_equiv_t4_20260502T0855Z/archive.zip`:
- Single member: `p` (276,114 bytes; +100 ZIP overhead → 276,214 archive total)
- Source dispatch: `submissions/robust_current/inflate.sh:175-184` detects `p` and routes to `unpack_renderer_payload.py`
- Parsed by: `submissions/robust_current/unpack_renderer_payload.py:_try_parse_public_pr67_qzs3_qp1_payload` (line 479-550)
  - Hard-coded split: `mask_len = 219_472`
  - mask: `payload[:219472]` → brotli → AV1 OBU
  - model: `payload[219472:219472+model_len]` → brotli → must start with `QZS3`
  - pose: `payload[219472+model_len:]` → brotli → must start with `QP1`
- Writes to ARCHIVE_DIR: `masks.mkv`, `renderer.bin`, `optimized_poses.bin`
- `inflate_renderer.py:1840` detects `.nrv` extension OR NRV1 magic for NeRV decoder routing — i.e., needs a `masks.nrv` member, NOT bytes inside the `p` blob

To stack NeRV onto C-067, EITHER:
- Add a new parser `_try_parse_public_pr67_NERV_qzs3_qp1_payload` to `unpack_renderer_payload.py` that splits the blob at NRV2 magic + variable-length boundary, then writes `masks.nrv` (NOT `masks.mkv`) + `renderer.bin` + `optimized_poses.bin`. ~80 LOC + tests.
- OR rebuild as a MAGIC-prefixed `renderer_payload` schema (line 124-218 of `unpack_renderer_payload.py`) where the JSON header enumerates `masks.nrv` (codec=raw), `renderer.bin` (codec=raw), `optimized_poses.bin` (codec=public_qp1_brotli). Adds ~250 bytes header overhead. ~30 LOC build script.

Either approach requires CODE changes BEFORE the GPU run. The subagent mandate is "ONE shot at NeRV full-CUDA" not "implement new inflate path + train + eval".

### Gate artifacts missing

`scripts/remote_lane_nerv.sh` blocks at:

- Line 35: `L2_CLEARANCE_PATH="$WORKSPACE/.omx/state/lane12_nerv_l2_clearance.json"` — file does NOT exist locally
- Lines 144-187: requires JSON object with `lane_id ∈ {lane_12_nerv_mask_codec, lane_12_nerv}`, `cleared_for_retraining_unblock=true`, `lane12_l2=true`, `geometry_gate_passed=true`, `grand_council_clean_passes >= 3`, `evidence` non-empty
- Lines 220-253: when `GT_MASKS_SOURCE=decoded-baseline` (the default), requires `ALPHA_PRIMITIVE_CONTRACT` JSON file with `diagnostic=alpha_geo_primitive_contract_v1`, `promotion_eligible=false`, `score_claim_eligible=false`, `exact_eval_claim=false`
- Lines 258-277: when `RUN_AUTH_EVAL=1` (which the dispatch needs to land a score), requires both `POSE_REGEN_PROVENANCE` and `ALPHA_GEO_PROVENANCE` JSONs, the latter with `diagnostic=alpha_geo_0_nerv_geometry` and `pass_fail.overall_pass=true`, plus `candidate_source.source_sha256` matching the rebuilt archive SHA and `baseline_source.source_sha256` matching `BASE_ARCHIVE`

These gates exist precisely because Lane 12 was historically dispatched without proof-of-clearance. They reflect prior council Round-N CLEAN landings that produced a documented L2-class evidence packet. None of those packets exist in the repo for a C-067-stack variant.

The subagent could fabricate stub JSONs with `cleared_for_retraining_unblock=true` to bypass the gates, BUT that violates CLAUDE.md "Comment-only contracts FORBIDDEN" + "Council conduct" + "Internal-consistency assertions" — fabricating clearance evidence with no real council deliberation behind it is exactly the bug class PCC4 (`check_kill_memory_files_have_council_review`) was landed to extinct.

## Why NOT to dispatch anyway

Per CLAUDE.md non-negotiables:

- **"Score target — NON-NEGOTIABLE"**: predicted stacked score 0.685 (linear-extrapolation realistic case) > 0.316 C-067 baseline → "if projected auth > 1.0, something is wrong — stop and fix it before burning more GPU hours". The 0.685 case isn't > 1.0 but it IS > current frontier; same fail-fast logic applies.
- **"Forbidden score claims"**: subagent cannot ship a `[contest-CUDA]` score from a $20 GPU run that produces a regression and quietly burn the budget. The handoff must surface the prediction BEFORE spend.
- **"Council conduct" + "Design decisions — non-negotiable"**: switching the NeRV training recipe (architecture, loss, ground-truth target) from Phase F's recipe to a sub-0.05%-target recipe IS a design decision requiring council deliberation. Subagent does NOT have authority to redesign the training recipe.
- **User mandate "do NOT cycle through hyperparameter tuning"**: dispatching at 60K steps and discovering 0.5% floor → trying 80K steps with hidden=80 → trying KL-distill weighting → ... is exactly the pattern the user forbade in this dispatch.

## Three reactivation paths (user decision)

### Path A — Recipe redesign (requires council)

Change the training recipe to drive disagreement < 0.05%:
- Larger NeRV (hidden=80 or 96, depth=5, num_freqs=10) → +5-15KB rate cost but better convergence
- Hard-pair mining: extract the boundary pixels where Phase F NeRV disagrees with AV1 mask, weight them 10× in loss
- KL-distill T=2.0 against the SegNet logits the AV1 mask was derived from (not the AV1 mask itself) — distillation target is the underlying scorer behavior, not the lossy AV1 quantization
- Decoder ensemble: train 2 NeRVs + average their decoded masks (doubles inflate cost but might cut disagreement in half)

Cost: $5-15 GPU + 1 council session + ~200 LOC of training-script changes. Time: 6-10h. Risk: still might hit a 0.2-0.3% floor; sub-0.3 not guaranteed.

### Path B — Inflate parser + Alpha-Geo contracts (no GPU)

Build the missing infrastructure FIRST so subsequent NeRV training has a clean stack-onto-C-067 path:
- Add `_try_parse_public_pr67_NERV_qzs3_qp1_payload` to `unpack_renderer_payload.py` (~80 LOC + 5 tests)
- Build a MAGIC-prefixed renderer_payload variant that pulls C-067's QZS3+QP1 + a `masks.nrv` blob (~30 LOC builder + Lane 12 test fixture)
- Generate the Lane 12 L2 clearance + Alpha-Geo primitive + pose-regen provenance JSONs by running the existing geometry-gate harness (`experiments/alpha_geo_*` exists in this repo) on the latest Phase F NeRV checkpoint
- 3-pass council review of the new parser + contract JSONs

Cost: $0 GPU + ~6h subagent time + 1 council session. Outcome: NeRV-onto-C-067 becomes a turn-key dispatch with all gates passing. Recommended FIRST.

### Path C — NeRV onto Lane G v3 (not C-067)

Re-aim the NeRV stack at the Lane G v3 baseline (1.05 [contest-CUDA] anchor) where `remote_lane_nerv.sh` is already wired. This is the script's design target. Still requires gate artifact creation (Path B subset for Lane G v3) but skips the C-067 parser work. NeRV training itself is identical.

Cost: $5-10 GPU + ~2h gate prep. Outcome: Lane G v3 score might drop from 1.05 → ~0.95 if NeRV converges to ~0.5% disagreement. Still well above C-067 0.316 frontier so does NOT improve the deploy candidate; valuable only as Lane 12 maturity (real_archive_empirical + contest_cuda gates) for the registry.

## Reactivation criteria

This refusal is REVERSED if any of the following lands:

- Council session approves Path A recipe redesign with a forecast band that includes <0.05% disagreement supported by adjacent-art evidence (e.g., NeRV literature on similar boundary tasks)
- Path B infrastructure work lands at HEAD with passing tests and an L2 clearance JSON signed off by a 3-pass council review
- An empirically-stronger NeRV checkpoint surfaces (e.g., from a parallel codex partner) with measured disagreement <0.05% on the C-067 PR67 mask source — then dispatch is just "rebuild archive + eval", $2-3
- User explicitly overrides with "dispatch anyway, accept regression risk" annotation in this file

## Lane registry impact

Lane 12 stays at current Level (per `tools/lane_maturity.py audit`). This refusal does NOT regress maturity. The blockers documented here are infrastructure / recipe gaps, not implementation bugs.

## Cost honesty

Refusal cost: $0 GPU, ~25 min subagent wall-clock. Avoided cost (had I dispatched): $15-20 GPU on a likely-regression run + 6-8h wall-clock + the cost of explaining a worse-than-baseline score. The pre-flight gating in `remote_lane_nerv.sh` would also have failed-loud before any meaningful spend, so the wall-clock saved is mostly the ~15 min between "instance provisioned" and "L2 gate fails at Stage 0c". The structural value of this memory file is that the next subagent or codex partner does NOT need to re-derive the same blockers from scratch.

## Files referenced

- Target codec: `src/tac/nerv_mask_codec.py` (37KB, NeRVMaskCodec + NeRVMaskTrainer + NRV2 wire format)
- Training entry: `experiments/train_nerv_mask.py` (50KB)
- Deploy runbook: `scripts/remote_lane_nerv.sh` (20KB)
- Stack pattern reference: `scripts/remote_lane_12_owv3_0120_nerv_stack.sh` (12KB)
- Inflate path (NeRV decoder live): `submissions/robust_current/inflate_renderer.py:996-1053`
- Inflate path (PR67 single-blob parser): `submissions/robust_current/unpack_renderer_payload.py:479-550`
- C-067 anchor: `experiments/results/lightning_batch/exact_eval_c063_fixedslice_equiv_t4_20260502T0855Z/archive.zip`
- C-067 anatomy doc: `docs/paper/EXTERNAL_SOURCE_ATTRIBUTION_C067.md`
- Phase F empirical: `reports/lane_12_nerv_real_archive.json` (per dispatch plan reference)
- Dispatch claim: `.omx/state/active_lane_dispatch_claims.md` row 2026-05-02T12:55:00Z

## What I did NOT do (intentionally)

- Did NOT call `vastai create instance`
- Did NOT modify `unpack_renderer_payload.py` to add a NeRV parser (out of mandate scope)
- Did NOT fabricate L2 clearance / Alpha-Geo contract JSONs to bypass the gates
- Did NOT modify `scripts/remote_lane_nerv.sh` to remove the gate checks
- Did NOT run `experiments/train_nerv_mask.py` locally with `--device cpu` (banned per "MPS-falsification" + "advisory only" rules for any score-relevant decision)
- Did NOT spawn additional subagents
- Did NOT run any `kill` / `unmark` mutations on `tools/lane_maturity.py`

The refusal is the deliverable. The next decision rests with the user.
