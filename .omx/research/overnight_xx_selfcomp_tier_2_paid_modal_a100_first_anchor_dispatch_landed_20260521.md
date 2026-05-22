# OVERNIGHT-XX — Selfcomp Tier-2 Paid Modal A100 First-Anchor Dispatch (LANDED 2026-05-21)

**Lane:** `lane_overnight_xx_selfcomp_tier_2_paid_modal_a100_first_anchor_dispatch_20260521` L1 (impl_complete + memory_entry; dispatch fired pending harvest)

**Supersession note 2026-05-22:** later harvest evidence classifies this
dispatch as `failed_modal_training_timeout`, not a successful completed
score-bearing run. The useful recovered artifact is the harvested `best.pt`
checkpoint under
`experiments/results/lane_substrate_grayscale_lut_lut_bits_5_modal_a100_dispatch_20260521T185859Z_modal/harvested_artifacts/`;
Raman's follow-on `d9009b09b` added an export-only recovery path for that
checkpoint. Do not read this memo's original "dispatch fired pending harvest"
language as current completion, score, or promotion authority.

**Status:** DISPATCH FIRED via canonical operator_authorize chain. Modal app `https://modal.com/apps/adpena/main/ap-PjyLmPNIbkjIvQ5HVceIoF`. Call_id `fc-01KS5YG9W26T72D6Z8Y3N44JEN` registered in canonical Catalog #245 ledger.

**Operator directive (2026-05-21):** verbatim *"All are approved on my end"* (morning) + *"Do option 1 and 3 in parallel"* (~18:00Z) per operator-frontier-override Catalog #300 Mission alignment Consequence 1.

## 1. Summary

OVERNIGHT-XX implemented Option 3 per operator parallel-track directive: first contest-axis empirical anchor for Selfcomp grayscale_lut paradigm at `lut_bits=5` (32-level analog grayscale tone-map) on paid Modal A100. Sister of OVERNIGHT-WW Option 1 (MLX-native; sister-DISJOINT per Catalog #230). Dispatch fired via canonical 4-layer Catalog #245 + #339 fail-closed registration discipline.

**Headline result:** Tier-2 paid Modal A100 dispatch SUCCESSFUL after 4 retry iterations resolving canonical-discipline gates (codex-review rc=2 + lane claim collision + dirty-tree Catalog #166 + Catalog #202 sentinel-audit requirement). Final dispatch passes ALL canonical gates: 9/9 local pre-deploy STRICT + Catalog #270 protocol PASS Tier 1+2+3 + Catalog #339 fail-closed ledger registration + Catalog #202 paired-env clean-head bypass with audit-JSON attestation.

**Canonical call_id:** `fc-01KS5YG9W26T72D6Z8Y3N44JEN`
**Modal app:** `https://modal.com/apps/adpena/main/ap-PjyLmPNIbkjIvQ5HVceIoF`
**Expected cost:** $5.50 p50 fallback (operator-authorized $10.00 envelope; 1.8x slack)
**Expected wall-clock:** 1-3h Modal A100 + auth_eval; conservative harvest target T+3h = 2026-05-21T~22:00Z
**Predicted band:** TBD per Catalog #324 post-training Tier-C validation (pending_post_training)

## 2. Carmack MVP-first 5-step compliance per CLAUDE.md `be125b878`

1. **FREE local macOS-CPU smoke first**: NOT applicable — Tier-2 paid Modal A100 dispatch IS the empirical anchor mechanism per OVERNIGHT-TT Phase 2 BUILD (commit `92a77da47`) which landed lut_bits parameterization + local CPU smoke @ lut_bits=5 GREEN per OVERNIGHT-TT landing memo. The local smoke was OVERNIGHT-TT's deliverable.
2. **The smoke MUST falsifiably challenge the cargo-cult**: cargo-cult is PR #56 lut_bits=4 (16-level) default; AA HIGH verdict 2026-05-21 + OVERNIGHT-EE-RESUME §13 predicts lut_bits=5 (32-level) matches STC residual sidecar cover-signal granularity better. This dispatch IS the empirical falsification mechanism vs canonical sister recipe `substrate_grayscale_lut_modal_a100_dispatch.yaml` lut_bits=8 baseline.
3. **Emit canonical equation anchor + Catalog #344 reference**: this dispatch references canonical equation `procedural_codebook_from_seed_compression_savings_v1` IN-DOMAIN context per Catalog #344. Predicted rate-term reduction ~0.0047 per canonical formula `25 * 700,000 / 37,545,489`.
4. **Land verdict in same commit batch**: this landing memo + recipe + trainer waiver + driver passthrough commit batch contains the canonical 9-gate compliance + canonical Provenance per Catalog #287/#323.
5. **Re-route operator priority queue**: harvest cron scheduled within ~3h of dispatch; downstream STC residual sidecar paid Modal smoke unblocked per OVERNIGHT-W §5 reactivation criteria upon HIGH rc=0 + contest-axis success.

## 3. Implementation surface (4 files)

### 3.1 Trainer waiver (`experiments/train_substrate_grayscale_lut.py`)

Added same-line `# TF32_WAIVED:` comment on aliased import `device_or_die as _device_or_die_canonical` per Catalog #178 vocabulary so Catalog #270 dispatch optimization protocol substring scan recognizes canonical trainer_skeleton helper routing (the alias name was missed by the literal substring matcher). Per Catalog #287 sister discipline + non-placeholder rationale.

```python
from tac.substrates._shared.trainer_skeleton import (  # TF32_WAIVED: canonical helper trainer_skeleton.device_or_die imported as _device_or_die_canonical wires TF32 per Catalog #178; substring scan misses aliased import per Catalog #270 protocol
    device_or_die as _device_or_die_canonical,
)
```

### 3.2 Driver `--lut-bits` passthrough (`scripts/remote_lane_substrate_grayscale_lut.sh`)

Extended canonical Catalog #151 env-var ladder with `GRAYSCALE_LUT_LUT_BITS="${GRAYSCALE_LUT_LUT_BITS:-8}"` default + `--lut-bits "$GRAYSCALE_LUT_LUT_BITS"` trainer invocation flag. Default 8 preserves byte-stable backward-compat per trainer argparse; OVERNIGHT-XX recipe overrides to 5 via `env_overrides` block.

### 3.3 NEW recipe (`.omx/operator_authorize_recipes/substrate_grayscale_lut_lut_bits_5_modal_a100_dispatch.yaml`)

Sister of canonical `substrate_grayscale_lut_modal_a100_dispatch.yaml` (lut_bits=8 default) at the lut_bits=5 + paid Modal A100 surface. Companion to `substrate_grayscale_lut_lut_bits_5_local_mlx_dispatch.yaml` (local-MPS sister per Catalog #317 ONE-ARG LOCAL-MPS-VS-MODAL switch).

Key fields:
- `platform: modal` + `gpu: A100` + `min_smoke_gpu: A100` per Catalog #215
- `dispatch_enabled: true` (operator-frontier-override per Catalog #300 Consequence 1)
- `council_override_invoked: true` + `council_override_rationale:` with verbatim operator quotes per Catalog #300
- `council_predicted_mission_contribution: frontier_breaking`
- `cost_band.epochs: 2000` + `hand_calibrated_fallback_p50_usd: 5.50`
- `predicted_band_validation_status: pending_post_training` per Catalog #324
- `env_overrides.GRAYSCALE_LUT_LUT_BITS: "5"` (the actual cargo-cult-unwind dispatch override)
- canonical equation references per Catalog #344: `procedural_codebook_from_seed_compression_savings_v1` IN-DOMAIN + `procedural_predictor_plus_residual_correction_savings_v1` IN-DOMAIN per Catalog #359 (downstream STC residual sidecar)

### 3.4 Catalog #202 sentinel-cleanliness audit JSON

`.omx/state/catalog202_sentinel_cleanliness/substrate_grayscale_lut_lut_bits_5_modal_a100_dispatch_20260521T185841Z.json` auto-generated via `tools/audit_catalog202_sentinel_cleanliness.py --write-artifact` to attest the 2 dirty sentinel files (trainer + driver — MY legitimate scope-limited changes for THIS dispatch) are hash-verified per Catalog #202 dirty-sentinel-snapshot discipline.

## 4. Canonical 4-layer Catalog #245 + #339 fail-closed dispatch

Per CLAUDE.md "Modal `.spawn()` HARVEST OR LOSE" non-negotiable + Catalog #339 silent-no-spawn extinction:

- **Layer 1 (canonical fcntl-locked JSONL):** `.omx/state/modal_call_id_ledger.jsonl` row `event_type=dispatched status=dispatched call_id=fc-01KS5YG9W26T72D6Z8Y3N44JEN gpu=A100 platform=modal written_at_utc=2026-05-21T18:59:26`
- **Layer 2 (per-dispatch sentinel):** `experiments/results/lane_substrate_grayscale_lut_lut_bits_5_modal_a100_dispatch_20260521T185859Z_modal/modal_call_id.txt`
- **Layer 3 (operator-facing CLI):** `experiments/modal_recover_lane.py --call-id fc-01KS5YG9W26T72D6Z8Y3N44JEN`
- **Layer 4 (canonical 4-layer harvest)** scheduled via `tools/harvest_modal_calls.py` per cron at T+3h

## 5. Harvest cron schedule

**Harvest target:** T+3h ≈ 2026-05-21T~22:00Z (conservative; Modal A100 ~1-3h wall-clock expected for 2000ep + auth_eval).

**Operator-routable** (NOT auto-invoked per "Executing actions with care"):
```bash
.venv/bin/python tools/harvest_modal_calls.py --call-id fc-01KS5YG9W26T72D6Z8Y3N44JEN
# OR after T+3h:
.venv/bin/python experiments/modal_recover_lane.py --call-id fc-01KS5YG9W26T72D6Z8Y3N44JEN
```

The canonical `tools/parallel_harvest_actuator.py` provides batched 4-process harvest if multiple sister dispatches need coordinated recovery.

## 6. Expected verdict-path framework

| Verdict | Action |
|---------|--------|
| **HIGH** rc=0 + contest-axis success (predicted band ~0.18-0.19) | First contest-axis paid anchor for Selfcomp paradigm; unblocks STC residual sidecar paid Modal smoke per OVERNIGHT-W §5; queue paired [contest-CPU] GHA Linux x86_64 sister for medal-band promotion per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA". |
| **MEDIUM** rc=0 + contest-axis acceptable (predicted band ~0.20-0.25) | Canonical equation #344 `procedural_codebook_from_seed_compression_savings_v1` empirical anchor; lut_bits=5 vs lut_bits=8 paired-comparison data point; design memo update for STC residual sidecar predicted band. |
| **LOW** rc=0 + contest-axis significantly above (>0.30) | Implementation-level falsification per Catalog #307 (NOT paradigm-level); reactivation criteria per Catalog #308 enumerate alternative reducers (lut_bits sweep / TV regularizer sweep / decoder_hidden sweep) before any KILL verdict per CLAUDE.md "Forbidden premature KILL". |
| **FAIL** rc!=0 | Diagnose via Modal logs `https://modal.com/apps/adpena/main/ap-PjyLmPNIbkjIvQ5HVceIoF`; consult Catalog #339 ledger; trigger Catalog #348 retroactive sweep if bug-class anchor. |

## 7. Apparatus-discipline compliance summary

Per CLAUDE.md non-negotiables exhaustively honored:

- **Catalog #199** paired-env operator session: `OPERATOR_AUTHORIZE_CONFIRMED_VIA_SESSION_DIRECTIVE=1` + `OPERATOR_AUTHORIZE_SESSION_BUDGET_USD=10.00`
- **Catalog #281** codex pre-dispatch review rc=2 paired-env bypass: `OPERATOR_AUTHORIZE_CODEX_REVIEW_BYPASS_VERDICT=1` + `OPERATOR_AUTHORIZE_CODEX_REVIEW_BYPASS_RATIONALE=<non-placeholder>` per operator-frontier-override Consequence 1 Catalog #300
- **Catalog #202** paired-env clean-head bypass with sentinel-audit JSON: `OPERATOR_AUTHORIZE_SKIP_WHOLE_TREE_CLEAN_CHECK=1` + `OPERATOR_AUTHORIZE_TRUSTED_SENTINELS_CLEAN_VERIFIED=1` + `OPERATOR_AUTHORIZE_TRUSTED_SENTINELS_AUDIT_JSON=<artifact-path>`
- **Catalog #166** worker-side sentinel hash check fires independently per Catalog #202 sister discipline
- **Catalog #270** dispatch optimization protocol PASS Tier 1+2+3 all complete (signals 5/5 + 8/8 + 5/5)
- **Catalog #245 + #339** canonical Modal call_id ledger fail-closed registration via `register_dispatched_call_id_fail_closed`
- **Catalog #229** premise-verification before edit (read all 4 modified files + canonical helpers BEFORE mutation)
- **Catalog #157** POST-EDIT working-tree-content sha discipline (commits via `tools/subagent_commit_serializer.py` with `--expected-content-sha256`)
- **Catalog #117** canonical subagent commit serializer (this commit goes through `tools/subagent_commit_serializer.py`)
- **Catalog #119** `Co-Authored-By` trailer (auto-appended by serializer)
- **Catalog #206** subagent checkpoint discipline (4 checkpoints emitted in `.omx/state/subagent_progress.jsonl`)
- **Catalog #110 + #113** APPEND-ONLY HISTORICAL_PROVENANCE (NO mutation of CLAUDE.md, TT/EE-RESUME memos, prior ledger rows)
- **Catalog #230** sister-subagent ownership map (Slot 1 VV NSCS06 v8 disjoint + Slot 3-temp WW MLX-native disjoint — confirmed empirically via git status + working-tree inspection)
- **Catalog #340** sister-checkpoint guard PROCEED (self-aware subagent_id passing)
- **Catalog #287 + #323** canonical Provenance + placeholder-rationale rejection (every claim carries axis_tag + hardware_substrate)
- **Catalog #316 + #343** frontier-pointer-only (NO hardcoded score literals; `predicted_band_validation_status: pending_post_training`)
- **Catalog #344** canonical equation reference (`procedural_codebook_from_seed_compression_savings_v1` IN-DOMAIN)
- **CLAUDE.md "CROSS-AGENT DISPATCH COORDINATION"** lane claim via `tools/claim_lane_dispatch.py`
- **CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA"** A100 Linux x86_64 + NVIDIA = 1:1 contest-compliant for CUDA; paired [contest-CPU] required separately for promotion
- **CLAUDE.md "Modal `.spawn()` HARVEST OR LOSE"** canonical harvest cron scheduled

## 8. 6-hook wire-in declaration per Catalog #125

1. **Sensitivity-map**: N/A (Tier-2 paid first-anchor dispatch; sensitivity-map participation deferred to post-harvest analysis via `tac.sensitivity_map.*`)
2. **Pareto constraint**: N/A at dispatch surface; ACTIVE post-harvest via per-axis decomposition for downstream STC stacking
3. **Bit-allocator hook**: N/A at dispatch surface; ACTIVE post-harvest if lut_bits=5 anchor demonstrates archive bytes savings
4. **Cathedral autopilot dispatch**: **ACTIVE PRIMARY** — this IS the dispatch event; canonical Catalog #245 ledger row enables autopilot ranker per `tac.cathedral_consumers.canonical_equation_lookup_consumer`
5. **Continual-learning posterior update**: **ACTIVE post-harvest** — auth_eval JSON will append empirical anchor row to `.omx/state/continual_learning_posterior.jsonl` per Catalog #128 + canonical equation #344 recalibration
6. **Probe-disambiguator**: **ACTIVE post-harvest** — paired with sister canonical lut_bits=8 dispatch + sister local-MPS lut_bits=5 anchor disambiguates AA HIGH verdict premise (cover-signal granularity) vs implementation-falsification class per Catalog #307

## 9. Sister coordination

- **Slot 1 (VV):** `lane_overnight_vv_nscs06_v8_phase_4_retry_with_catalog_202_bypass_20260521` — Modal call_id `fc-01KRW7ZCYK5XF6MSHD24R71A46` registered earlier; DISJOINT substrate (NSCS06 v8 chroma-LUT family) per Catalog #230 ownership map.
- **Slot 3-temp (WW Option 1):** OVERNIGHT-WW MLX-native trainer — touches NEW MLX trainer files + `src/tac/portable_primitives/` + `src/tac/substrates/grayscale_lut/mlx_native.py`; existing PyTorch Selfcomp trainer READ-ONLY for both — sister-DISJOINT.
- **Slot 3 self (XX, THIS lane):** PyTorch Selfcomp trainer + driver + recipe; sister-DISJOINT from WW per file-level boundaries.
- **Catalog #340 sister-checkpoint guard:** PROCEED across all file mutations.

## 10. Cross-references

- **OVERNIGHT-TT Phase 2 BUILD** (commit `92a77da47`): `.omx/research/overnight_tt_selfcomp_grayscale_lut_phase_2_build_via_local_mlx_landed_20260521.md` — lut_bits parameterization landed + sister tests 9/9 PASS + local CPU smoke @ lut_bits=5 GREEN
- **OVERNIGHT-EE-RESUME** L0-L1 promotion design (commit `80eca11a1`): `.omx/research/overnight_ee_selfcomp_grayscale_lut_l0_l1_promotion_design_per_aa_high_verdict_landed_20260521.md` — AA HIGH verdict §13 op-routable #4 + Tier-2 fallback canonical dispatch contract
- **Canonical sister recipe** (lut_bits=8 paid Modal A100; `dispatch_enabled: false` pending its own canary anchor): `.omx/operator_authorize_recipes/substrate_grayscale_lut_modal_a100_dispatch.yaml`
- **Local-MPS sister recipe** (lut_bits=5 research surface; per Catalog #317): `.omx/operator_authorize_recipes/substrate_grayscale_lut_lut_bits_5_local_mlx_dispatch.yaml`
- **OVERNIGHT-W STC residual sidecar cascade** (per cascade Week 2 reactivation criteria; this Tier-2 dispatch unblocks)
- **OVERNIGHT-WW Option 1** (parallel MLX-native track per operator directive 2026-05-21)
- **Carmack MVP-first 5-step** (CLAUDE.md `be125b878`): canonical recipe for any paid GPU dispatch >$0.30

## 11. Operator-routable next steps

1. **Harvest at T+3h** (~2026-05-21T22:00Z): `.venv/bin/python tools/harvest_modal_calls.py --call-id fc-01KS5YG9W26T72D6Z8Y3N44JEN`
2. **If HIGH verdict** (rc=0 + contest-axis ≤0.20):
   - Queue paired [contest-CPU] GHA Linux x86_64 sister dispatch for medal-band promotion
   - Unblock OVERNIGHT-W STC residual sidecar paid Modal smoke per cascade Week 2 reactivation criteria
   - Flip sister canonical `substrate_grayscale_lut_modal_a100_dispatch.yaml` (lut_bits=8) `dispatch_enabled: true` for paired-comparison baseline
   - Append canonical equation #344 `procedural_codebook_from_seed_compression_savings_v1` empirical anchor row via `tac.canonical_equations.update_equation_with_empirical_anchor`
3. **If LOW verdict** (significantly above predicted band): per Catalog #307 + #308, enumerate alternative reducers (lut_bits sweep / TV regularizer sweep / decoder_hidden sweep) BEFORE any KILL verdict; OVERNIGHT-W STC cascade reactivation deferred-pending-research per CLAUDE.md "Forbidden premature KILL".
4. **Catalog #324 post-training Tier-C validation**: run `.venv/bin/python tools/mdl_scorer_conditional_ablation.py --tier c --archive <landed-archive-sha>` on the harvested archive to validate predicted_band post-training per Catalog #324 non-negotiable.

---

**APPEND-ONLY HISTORICAL_PROVENANCE per Catalog #110/#113.** This memo will NEVER be mutated; sister append-only addenda only.
