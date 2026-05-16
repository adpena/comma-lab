# Dispatch plan ledger — 4 NSCS substrates + NSCS01+02+03 3-stack composition

**Date:** 2026-05-15
**Lane:** `lane_dispatch_planning_4sub_3stack_20260515`
**Subagent:** dispatch_planning_4sub_3stack
**Scope:** PLAN ONLY (no execution; READ-ONLY against source; writes only this ledger to `.omx/research/`).
**Disjoint sister-subagents:** `a4` (read-only audit) / `b3` (gate-author) per Catalog #230 ownership map.

This ledger drafts the paired CPU+CUDA Modal smoke-then-full dispatches for
NSCS01 / NSCS02 / NSCS03 / NSCS06, plus the proposed NSCS01+NSCS02+NSCS03
stack-of-stacks composition. Every CLI line cites the canonical entry point
(`tools/operator_authorize.py` / `tools/run_modal_smoke_before_full.py` /
`tools/dispatch_modal_paired_auth_eval.py` / `tools/harvest_modal_calls.py` /
`tools/local_pre_deploy_check.py`) and was verified against each tool's
`add_argument` surface — NO invented flags per CLAUDE.md "NEVER invent CLI
flags".

---

## Section 1 — Pre-dispatch readiness check (per substrate)

Recipe schema fields lifted via direct YAML read; trainer `_full_main`
implementation status verified via grep (all 4 now have `def _full_main`
bodies; no `raise NotImplementedError`).

| Substrate | Recipe | `dispatch_enabled` | `research_only` | `smoke_only` | `min_smoke_gpu` (Catalog #215) | `target_modes` (Catalog #182) | Trainer `_full_main` | Contest-CUDA dispatchable today? |
|---|---|---|---|---|---|---|---|---|
| NSCS01 nullspace-split | `substrate_nscs01_nullspace_split_renderer_modal_t4_dispatch.yaml` | **false** | **true** | **true** | T4 | `[research_substrate]` | implemented (PR95 paradigm; 2026-05-15) | NO — operator must flip `dispatch_enabled: true` AND clear `dispatch_blockers` after Phase 2 council green-up |
| NSCS02 downsampled-renderer | `substrate_nscs02_downsampled_renderer_modal_t4_dispatch.yaml` | **false** | **true** | **true** | T4 | `[research_substrate]` | implemented (UNIQUE-AND-COMPLETE PR95 paradigm; 2026-05-15) | NO — operator must flip + resolve resizing-chain ablation blocker |
| NSCS03 Ballé end-to-end joint | `substrate_nscs03_end_to_end_balle_joint_codec_modal_a100_dispatch.yaml` | **false** | **true** | **true** | **A100** (Catalog #215 — joint codec memory floor) | `[research_substrate]` | implemented (END-TO-END Ballé 2018; 2026-05-15) | NO — operator must flip + Phase 2 council λ_R sweep + σ-floor calibration |
| NSCS06 Carmack-Hotz strip-everything | `substrate_nscs06_carmack_hotz_strip_everything_modal_t4_dispatch.yaml` | **true** | **false** | (default false) | T4 | `[contest_one_video_replay, research_substrate]` | implemented (symposium #4; closed-form 1ep pass) | **YES** — contest-eligible; ONLY substrate of the four with dispatch_enabled=true |

**Operator-action gap (per Catalog #240 recipe-vs-trainer-state consistency):**
NSCS01/02/03 trainers now ship full `_full_main` bodies (per the just-landed
memory entries for `feedback_nscs01_full_main_implementation_pr95_paradigm_landed_20260515.md`
and `feedback_nscs03_full_main_implementation_pr95_balle_2018_paradigm_landed_20260515.md`),
but their recipes still declare `research_only: true` / `dispatch_enabled: false`.
This is the CORRECT state pending Phase 2 council green-up — flipping
`dispatch_enabled: true` is an OPERATOR-GATE-RAISE not a recipe-vs-trainer
divergence (the trainer is implemented; the operator chooses NOT to dispatch
yet). See Op-routable #1.

**Wrapper / driver readiness:**

| Substrate | `experiments/train_substrate_<id>.py` | `scripts/remote_lane_substrate_<id>.sh` | `scripts/operator_authorize_substrate_<id>_modal_*_dispatch.sh` |
|---|---|---|---|
| NSCS01 | EXISTS (42.4 KB) | EXISTS (7.6 KB) | **MISSING** — must be generated via `tac.substrate_registry.driver_generator` before operator-authorize routing (per Catalog #244 canonical NVML env block + Catalog #163 bootstrap sentinel + Catalog #166 sentinel files + Catalog #167 smoke-before-full wrapper). |
| NSCS02 | EXISTS (39.6 KB) | EXISTS (8.7 KB) | **MISSING** — same as NSCS01 |
| NSCS03 | EXISTS (57.1 KB) | EXISTS (9.6 KB) | **MISSING** — same |
| NSCS06 | EXISTS (31.6 KB) | EXISTS (8.1 KB) | EXISTS (1.7 KB) — only substrate of the four with the canonical authorize wrapper already in place |

**TIER_1_OPERATOR_REQUIRED_FLAGS manifests (Catalog #151):** all 4 trainers
declare the manifest dict — wrapper threading via Catalog #151 is structurally
intact.

---

## Section 2 — Per-substrate paired dispatch plan

For each of 4 substrates: SMOKE (~$0.30–0.50; Catalog #167 canonical pattern)
→ FULL ($5–15; Catalog #176 canonical routing through `operator_authorize.py`
which wires Catalog #243 local pre-deploy + Catalog #271 codex pre-dispatch
review + Catalog #270 production-hardened protocol) → PAIRED CPU
($0.10–0.50; Catalog #246 anchor-skip-aware after CUDA full lands).

**Required env vars (universal):**

```bash
export OPERATOR_AUTHORIZE_CONFIRMED_VIA_SESSION_DIRECTIVE=1  # Catalog #199
export OPERATOR_AUTHORIZE_SESSION_BUDGET_USD=50.00            # Catalog #199 paired requirement
```

CPU paired-eval invocation also requires (per Catalog #271 cost-gated codex
pre-dispatch review and Catalog #243 fail-closed local harness):

```bash
# Only when paired CPU eval is < $1 (codex pre-dispatch review is cost-gated)
# Cost gate IS the gate; no bypass needed when budget < $1.
```

### Section 2.1 — NSCS06 Carmack-Hotz strip-everything (READY TODAY)

Sole substrate with `dispatch_enabled: true`. Canary class `independent_substrate`
per recipe, so this substrate does NOT block / does NOT require dependent
canary-first sequencing.

**SMOKE (canonical via Catalog #167):**

```bash
.venv/bin/python tools/run_modal_smoke_before_full.py \
    --recipe substrate_nscs06_carmack_hotz_strip_everything_modal_t4_dispatch \
    --smoke-only \
    --operator-handle "claude:nscs06_carmack_hotz_first_anchor_smoke"
```

- Smoke GPU: T4 (per `min_smoke_gpu: T4`); CLI flag `--smoke-gpu T4` default.
- Smoke epochs: `DEFAULT_SMOKE_EPOCHS` (100; Catalog #167 default).
- Smoke timeout: `DEFAULT_SMOKE_TIMEOUT_HOURS` (1.0; per recipe `timeout_hours: 1.0`).
- Expected duration: ~30–60s (cost band `epochs: 1, all_flags_on: true`; Carmack-Hotz is closed-form so 1-epoch pass).
- Expected cost: **~$0.01–0.05** (cost-band p50 anchor: D4 100ep T4 = $0.35 ⇒ 1ep ≈ $0.0035; round to floor $0.01).
- Expected score axis tag: `[advisory: smoke validation only]` (smoke validates parser/runtime + archive grammar; does NOT produce contest-CUDA score claim per Catalog #167).
- Expected blockers: NONE; canonical wrapper handles Catalog #163 + #166 + #244 NVML block via auto-generated driver.
- Harvest path: `experiments/results/lane_substrate_nscs06_carmack_hotz_strip_everything_modal_t4_dispatch_<UTC>__smoke__1ep_modal/`.

**FULL (canonical via Catalog #176 routing through Catalog #243 + Catalog #271 + Catalog #270):**

```bash
.venv/bin/python tools/run_modal_smoke_before_full.py \
    --recipe substrate_nscs06_carmack_hotz_strip_everything_modal_t4_dispatch \
    --operator-handle "claude:nscs06_carmack_hotz_first_anchor_full"
```

(Default — runs smoke FIRST then full upon green; NEVER use `--full-only` per
CLAUDE.md "Race-mode rigor inversion" + Catalog #167.)

- Full GPU: T4 (per `gpu: ${MODAL_GPU:-T4}`); `min_vram_gb: 16`.
- Full epochs: 1 (per cost-band `epochs: 1`; closed-form 1-shot codec).
- Full timeout: 1.0h.
- Expected duration: ~60–300s (single forward pass per video × 600 pair compress).
- Expected cost: **$0.10–$8.00** (cost-band p50 fallback `$8.00`; predicted lower from D4 anchor extrapolation).
- Expected score axis tag: `[contest-CUDA Modal T4]` (cost-band <$1 ⇒ Catalog #271 codex pre-dispatch review NOT triggered; but Catalog #270 umbrella protocol still applies).
- Expected blockers per recipe risk: chroma-from-grayscale-LUT under-fidelity (predicted ΔS HIGH VARIANCE `[0.10, 0.20]`).
- Cost-band classification: `eval` / `smoke` (epochs=1 below typical `full` threshold; Catalog #239 boundary open `<=`).
- Harvest path: `experiments/results/lane_substrate_nscs06_carmack_hotz_strip_everything_modal_t4_dispatch_<UTC>__full__1ep_modal/`.

**PAIRED CPU (canonical via Catalog #246; only fires after CUDA full lands with promotable archive):**

```bash
.venv/bin/python tools/dispatch_modal_paired_auth_eval.py \
    --archive experiments/results/lane_substrate_nscs06_carmack_hotz_strip_everything_modal_t4_dispatch_<UTC>__full__1ep_modal/output/archive.zip \
    --inflate-sh submissions/nscs06_carmack_hotz_strip_everything/inflate.sh \
    --submission-dir submissions/nscs06_carmack_hotz_strip_everything \
    --label nscs06_carmack_hotz_strip_everything_paired \
    --lane-id-base lane_nscs06_paired_cpu_cuda_modal_dispatch_20260515 \
    --gpu T4 \
    --skip-axis-if-promotable-anchor-exists \
    --claim-agent "claude:nscs06_paired_cpu_anchor" \
    --execute
```

- Expected duration: 30–90 min for 600-sample CPU eval on Modal T4 CPU container.
- Expected cost: **$0.10–$0.50** (Modal CPU container ~$0.06/hr × ~1h).
- Expected score axis tag: `[contest-CPU Linux x86_64 Modal]` (per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA" + Catalog #127 custody validator).
- Catalog #246 skips the CUDA axis if the CUDA anchor already exists for the
  archive sha — saves ~$0.30 per re-fire.
- Harvest path: `experiments/results/modal_auth_eval_cpu/nscs06_carmack_hotz_strip_everything_paired_<UTC>_cpu/modal_cpu_auth_eval_result.json`.

### Section 2.2 — NSCS01 nullspace-split-renderer (BLOCKED on operator gate-flip)

**Prerequisite ops (sequence):**

1. Generate `scripts/operator_authorize_substrate_nscs01_nullspace_split_renderer_modal_t4_dispatch.sh` via canonical driver generator (per Catalog #167 wrapper).
2. Operator: review L1 SCAFFOLD evidence + Phase 2 council green-up criteria
   (frame-0/frame-1 PoseNet gradient norms + head0 CNN-vs-MLP probe).
3. Operator: flip recipe `dispatch_enabled: true` AND clear `dispatch_blockers`.
4. Backfill recipe `min_vram_gb: 16` (already declared) / `target_modes` (currently `research_substrate` only; flip to `contest_generalized` for contest dispatch).

**SMOKE (after gate-flip):**

```bash
.venv/bin/python tools/run_modal_smoke_before_full.py \
    --recipe substrate_nscs01_nullspace_split_renderer_modal_t4_dispatch \
    --smoke-only \
    --operator-handle "claude:nscs01_first_anchor_smoke"
```

- Smoke GPU: T4. Cost: **~$0.30** (per Z3 100ep T4 smoke posterior anchor).
- Expected duration: ~10–30 min (100 epochs × per-pair-latent renderer at bs=8).
- Expected score axis tag: `[advisory: smoke parser+runtime validation only]`.

**FULL (after smoke green):**

```bash
.venv/bin/python tools/run_modal_smoke_before_full.py \
    --recipe substrate_nscs01_nullspace_split_renderer_modal_t4_dispatch \
    --operator-handle "claude:nscs01_first_anchor_full" \
    --smoke-timeout-hours 1.0
```

- Full GPU: T4 (per recipe).
- Full epochs: 100 (cost-band default; consider override `--cost-band-epochs-override 1000` for full convergence).
- Expected cost: **$5–$15** (cost-band fallback p50 = $15.00; T4 long-burn).
- Expected score axis tag: `[contest-CUDA Modal T4]` (per Catalog #270 umbrella).
- Expected blockers per recipe risk: frame_0_head collapse if PoseNet gradient insufficient; reactivation criterion HEAD0_BITS=6 fallback.
- Cost-band classification: `full` (epochs ≥ 100 + cost ≥ $5).
- Harvest path: `experiments/results/lane_substrate_nscs01_nullspace_split_renderer_modal_t4_dispatch_<UTC>__full__100ep_modal/`.

**PAIRED CPU (after CUDA full + promotable archive):**

```bash
.venv/bin/python tools/dispatch_modal_paired_auth_eval.py \
    --archive experiments/results/lane_substrate_nscs01_nullspace_split_renderer_modal_t4_dispatch_<UTC>__full__100ep_modal/output/archive.zip \
    --inflate-sh submissions/nscs01_nullspace_split_renderer/inflate.sh \
    --submission-dir submissions/nscs01_nullspace_split_renderer \
    --label nscs01_nullspace_split_renderer_paired \
    --lane-id-base lane_nscs01_paired_cpu_cuda_modal_dispatch_20260515 \
    --gpu T4 \
    --skip-axis-if-promotable-anchor-exists \
    --claim-agent "claude:nscs01_paired_cpu_anchor" \
    --execute
```

- Expected duration / cost / harvest: same envelope as NSCS06 ($0.10–$0.50).

### Section 2.3 — NSCS02 downsampled-renderer (BLOCKED on operator gate-flip + resizing-chain ablation)

**Prerequisite ops (sequence):**

1. Generate `scripts/operator_authorize_substrate_nscs02_downsampled_renderer_modal_t4_dispatch.sh`.
2. Run NO-TRAIN A1 resizing-chain ablation (per recipe `dispatch_blockers`):
   compare direct 192→384, 192→874→1164→384, bilinear/bicubic, paired
   component deltas. This is a LOCAL CPU operation (~minutes; no GPU spend).
3. Operator: review ablation + flip `dispatch_enabled: true`.

**SMOKE (after gate-flip):**

```bash
.venv/bin/python tools/run_modal_smoke_before_full.py \
    --recipe substrate_nscs02_downsampled_renderer_modal_t4_dispatch \
    --smoke-only \
    --operator-handle "claude:nscs02_first_anchor_smoke"
```

- Smoke GPU: T4. Cost: **~$0.30** (recipe `hand_calibrated_fallback_p50_usd: 0.30` confirms).
- Expected duration: ~10–30 min.
- Recipe `cost_band.estimated_cost_usd_band: [5, 15]` for full.

**FULL (after smoke green):**

```bash
.venv/bin/python tools/run_modal_smoke_before_full.py \
    --recipe substrate_nscs02_downsampled_renderer_modal_t4_dispatch \
    --operator-handle "claude:nscs02_first_anchor_full" \
    --smoke-timeout-hours 1.5
```

- Expected cost: **$5–$15** (recipe-declared).
- Expected score axis tag: `[contest-CUDA Modal T4]`.
- Expected blockers per recipe risk: PoseNet bicubic-upsample anti-aliasing if `d_pose regression > 5e-4`.

**PAIRED CPU (after CUDA full):**

```bash
.venv/bin/python tools/dispatch_modal_paired_auth_eval.py \
    --archive experiments/results/lane_substrate_nscs02_downsampled_renderer_modal_t4_dispatch_<UTC>__full__100ep_modal/output/archive.zip \
    --inflate-sh submissions/nscs02_downsampled_renderer/inflate.sh \
    --submission-dir submissions/nscs02_downsampled_renderer \
    --label nscs02_downsampled_renderer_paired \
    --lane-id-base lane_nscs02_paired_cpu_cuda_modal_dispatch_20260515 \
    --gpu T4 \
    --skip-axis-if-promotable-anchor-exists \
    --claim-agent "claude:nscs02_paired_cpu_anchor" \
    --execute
```

### Section 2.4 — NSCS03 Ballé end-to-end joint codec (BLOCKED on operator gate-flip + λ_R sweep + σ-floor calibration; A100-only)

**HIGHEST COST envelope of the four** — recipe declares `min_smoke_gpu: A100`
+ `gpu: A100` + `hand_calibrated_fallback_p50_usd: 80.00` + `timeout_hours: 12.0`.
Per CLAUDE.md "Race-mode rigor inversion" + Catalog #173 canary_first ordering:
NSCS03 should be canary-first (no parallel fan-out at $80) UNLESS Phase 2
council has greenlit a parallel sweep.

**Prerequisite ops (sequence):**

1. Generate `scripts/operator_authorize_substrate_nscs03_end_to_end_balle_joint_codec_modal_a100_dispatch.sh`.
2. Phase 2 council: λ_R sweep design (typically logspace [0.01, 0.1, 0.5, 1.0, 5.0]).
3. Phase 2 council: σ-floor calibration (current default `1e-4`; needs low-rate op-point validation).
4. Operator: review council verdict + flip `dispatch_enabled: true`.

**SMOKE (after gate-flip; A100 smoke is REQUIRED per Catalog #215):**

```bash
.venv/bin/python tools/run_modal_smoke_before_full.py \
    --recipe substrate_nscs03_end_to_end_balle_joint_codec_modal_a100_dispatch \
    --smoke-only \
    --smoke-gpu A100 \
    --smoke-timeout-hours 1.5 \
    --operator-handle "claude:nscs03_first_anchor_smoke"
```

- Smoke GPU: A100 (CANNOT downgrade per Catalog #215; wrapper auto-rejects).
- Smoke cost: **~$1.50–$4.00** (A100 hourly $4.00/hr × 0.5–1.0h smoke).
- Expected duration: 30–60 min (entropy bottleneck + scale hyperprior + GDN at 384×512).

**FULL (after smoke green; A100 12h budget is expensive):**

```bash
.venv/bin/python tools/run_modal_smoke_before_full.py \
    --recipe substrate_nscs03_end_to_end_balle_joint_codec_modal_a100_dispatch \
    --smoke-gpu A100 \
    --smoke-timeout-hours 1.5 \
    --operator-handle "claude:nscs03_first_anchor_full"
```

- Full GPU: A100 + `min_vram_gb: 40`. Cost: **$60–$80** (p50 anchor $80; 12h timeout).
- **REQUIRES Catalog #271 codex pre-dispatch review** (cost > $1).
- Expected score axis tag: `[contest-CUDA Modal A100]`.
- Cost-band classification: `long_burn` (cost > $50; per Catalog #239 boundary).

**PAIRED CPU (after CUDA full):**

```bash
.venv/bin/python tools/dispatch_modal_paired_auth_eval.py \
    --archive experiments/results/lane_substrate_nscs03_end_to_end_balle_joint_codec_modal_a100_dispatch_<UTC>__full__100ep_modal/output/archive.zip \
    --inflate-sh submissions/nscs03_end_to_end_balle_joint_codec/inflate.sh \
    --submission-dir submissions/nscs03_end_to_end_balle_joint_codec \
    --label nscs03_balle_joint_codec_paired \
    --lane-id-base lane_nscs03_paired_cpu_cuda_modal_dispatch_20260515 \
    --gpu T4 \
    --skip-axis-if-promotable-anchor-exists \
    --claim-agent "claude:nscs03_paired_cpu_anchor" \
    --execute
```

- CPU paired eval runs on Modal T4 instance with `--device cpu` per upstream; cost identical to other substrates (**$0.10–$0.50**).

---

## Section 3 — 3-stack composition plan (NSCS01 + NSCS02 + NSCS03)

### Orthogonality argument (per 9-dim checklist Dimension 6 + ASSUMPTIONS-CHALLENGE-AUDIT composition matrix)

The three substrates exploit STRUCTURALLY ORTHOGONAL shared assumptions:

| Substrate | Axis varied | Shared assumption violated | Mechanism |
|---|---|---|---|
| NSCS01 | **Architecture × scorer-relationship** | SA02 `segnet_only_last_frame` | Split-head renderer: frame_0 head trained PoseNet-only (4-bit); frame_1 head trained SegNet+PoseNet (8-bit) |
| NSCS02 | **Decode-time-contract** | Scorer interpolation budget at (192, 256) effective receptive field | Render at (192, 256); inflate-time bicubic upsample to (384, 512) — scorer-invisible downscale |
| NSCS03 | **Training-time-paradigm** | Hand-designed pipeline (NeRV-family standard) | End-to-end joint codec: convolutional g_a + entropy bottleneck + scale hyperprior + convolutional g_s, joint trained with score-aware loss |

**Independence math** (per ASSUMPTIONS-CHALLENGE-AUDIT first-principles bounds):
each substrate's predicted ΔS is in the band `[-0.020, -0.030]` independently
(per `nscs01_nullspace_split_renderer_design_20260515.md` predicted [0.148, 0.178]
vs PR101 anchor 0.193 ⇒ ΔS≈-0.020; sister NSCS02/NSCS03 each likewise
predicted ~-0.020 band).

**Predicted compound ΔS (additive UPPER bound; saturating LOWER bound):**

- Additive (orthogonal axes assumption): `[-0.060, -0.090]` (sum of individual lower bounds)
- Saturating (composition_alpha < 1 per Catalog #227 ranker): `[-0.030, -0.045]` (50% sub-additive halving per default Tier C composition rule)

**Confidence:** LOW until empirical first anchor on at least 2-of-3 individuals lands AND Tier C ablation proves class-shift per Catalog #227.

### Composition runtime feasibility

**Inflate.sh budget on T4 (must be ≤30 min per contest):**

Per individual recipe `timeout_hours` declarations:
- NSCS01 inflate: ~1-3 min (split-head renderer per-pair forward)
- NSCS02 inflate: ~1-3 min (downsampled-renderer + bicubic upsample per-pair)
- NSCS03 inflate: ~3-8 min (Ballé conv g_s + hyperprior decode per-pair at 64-ch latent)

**Composition inflate estimate (cascade):** 5-14 minutes (sums additively because
each substrate's decoder runs sequentially per-pair, NOT parallel). **WITHIN
30-min contest budget.**

### Composition archive layout

Per CLAUDE.md HNeRV parity discipline L3: each component's bytes MUST occupy
declared offsets per parser-section manifest. The 3-stack composition layout:

```
composition_archive.zip
├── 0.bin (monolithic; sections at fixed offsets)
│   ├── HEADER     [offset 0x0000;  16 bytes] — magic "NSC03STK" + version
│   ├── NSCS01     [offset 0x0010; ~25 KB]    — split-head renderer state_dict + per-pair latents
│   ├── NSCS02     [offset NS01_END; ~15 KB] — downsampled-renderer + bicubic-mode meta
│   └── NSCS03     [offset NS02_END; ~80 KB] — Ballé joint codec (g_a, g_s, h_a, h_s, EB) state_dicts + 2 latent streams
└── (single-file per HNeRV parity L3)
```

**Total predicted archive bytes:** ~120 KB (sub-PR101 179 KB by ~33%).

### Composition trainer status (CRITICAL GAP)

**No 3-stack composition trainer exists today.** Search results:

- `experiments/train_substrate_nscs01_*.py` — single-substrate trainer
- `experiments/train_substrate_nscs02_*.py` — single-substrate trainer
- `experiments/train_substrate_nscs03_*.py` — single-substrate trainer
- No `experiments/train_composition_nscs01_nscs02_nscs03_*.py` or sister.

**Status per HNeRV parity discipline L7 (bolt-on size budget) + L5 (architecture must be the FULL renderer):**

DEFERRED-pending-design. The composition is structurally NEW substrate engineering
(not a bolt-on), so the LOC budget exceeds 350 explicitly per L7. Three design
paths possible:

1. **Sequential cascade (lowest engineering risk):** NSCS03 trains first to convergence;
   NSCS02 trains conditioned on NSCS03's reconstruction as ground-truth target;
   NSCS01 trains conditioned on both — sequential bootstrap.
2. **Joint multi-objective (highest expected ΔS; highest engineering risk):**
   all three sub-modules + composition score-aware loss trained jointly; α_NSCS01·loss_NSCS01 +
   α_NSCS02·loss_NSCS02 + α_NSCS03·loss_NSCS03 + α_composition·loss_composition_on_GT.
3. **Substrate-engineering scaffold (PR101 model):** lift PR101's ~600 LOC bind-all
   pattern to a 3-substrate variant; substrate_engineering tag exempts from bolt-on budget.

**Council adjudication required before any 3-stack dispatch.** Per CLAUDE.md
"Forbidden cross-archive composition (HStack/VStack/cross-paradigm) without a
single verified [contest-CUDA] substrate anchor": NSCS01/02/03 must each land
at least one contest-CUDA anchor before composition is dispatched. **Composition
is BLOCKED-pending-3-individual-anchors AND council design adjudication.**

### Composition dispatch plan (DEFERRED skeleton)

Once trainer exists + at least 2 individuals have [contest-CUDA] anchors,
proposed canonical CLI shape (PENDING TRAINER NAME — not yet wired):

```bash
# PLACEHOLDER — trainer + recipe + driver do not yet exist
.venv/bin/python tools/run_modal_smoke_before_full.py \
    --recipe substrate_nscs_composition_nscs01_nscs02_nscs03_modal_a100_dispatch \
    --smoke-only \
    --smoke-gpu A100 \
    --smoke-timeout-hours 2.0 \
    --operator-handle "claude:nscs_3stack_first_anchor_smoke"
```

- Expected smoke cost: ~$5–$10 (A100 × 2h, joint training of all 3 sub-modules).
- Expected full cost: $40–$120 (A100 × 10–30h; depends on joint vs sequential training path chosen by council).

---

## Section 4 — Total budget + sequencing

### Total cost envelope (across the 4 substrates + 1 composition)

| Dispatch | Cost band | Sequencing dependency |
|---|---|---|
| NSCS06 smoke | $0.01–$0.05 | Independent canary (ready today) |
| NSCS06 full | $0.10–$8.00 | After NSCS06 smoke green |
| NSCS06 paired CPU | $0.10–$0.50 | After NSCS06 CUDA full + promotable anchor |
| NSCS01 smoke | $0.30 | After operator gate-flip + driver gen |
| NSCS01 full | $5–$15 | After NSCS01 smoke green |
| NSCS01 paired CPU | $0.10–$0.50 | After NSCS01 CUDA full |
| NSCS02 smoke | $0.30 | After operator gate-flip + resizing-chain ablation + driver gen |
| NSCS02 full | $5–$15 | After NSCS02 smoke green |
| NSCS02 paired CPU | $0.10–$0.50 | After NSCS02 CUDA full |
| NSCS03 smoke | $1.50–$4.00 | After operator gate-flip + Phase 2 council + driver gen |
| NSCS03 full | $60–$80 | After NSCS03 smoke green (REQUIRES Catalog #271 codex pre-dispatch review) |
| NSCS03 paired CPU | $0.10–$0.50 | After NSCS03 CUDA full |
| 3-stack composition smoke | ~$5–$10 | After 2-of-3 individual CUDA anchors + council design adjudication + trainer wired |
| 3-stack composition full | $40–$120 | After composition smoke green |
| 3-stack composition paired CPU | $0.30–$1.00 | After composition CUDA full |

**Total estimated envelope (sum of all phases):**

- **Minimum (all-pass; lower-bound costs):** ~$117
- **Maximum (worst-case; upper-bound costs):** ~$253
- **Realistic p50:** ~$155 (Modal-anchor-weighted; biased by NSCS03 A100 cost which dominates 50%+ of envelope)

### Recommended dispatch sequencing

Per CLAUDE.md "Race-mode rigor inversion + parallel-dispatch first" + recipe
`canary_status: independent_substrate` (all 4 declare independent):

**Phase 1 (TODAY; ready immediately):** NSCS06 smoke → full → paired CPU.
Sole substrate with `dispatch_enabled: true`; closed-form 1-epoch pass; lowest
cost envelope; highest information-per-dollar.

**Phase 2 (in parallel after operator gate-flips):** NSCS01 + NSCS02 + NSCS03
smokes in PARALLEL per CLAUDE.md "parallel-dispatch is a FIRST-CLASS
DELIVERABLE" + Catalog #173 `independent_substrate` canary status. Use
`tools/parallel_dispatch_top_k.py` (or sister parallel actuator) to fan out
3 concurrent smokes; concurrency cap = 3.

**Phase 3 (sequential per individual smoke-green):** Each substrate's full
fires when its OWN smoke goes green per Catalog #167 smoke-before-full;
3 full dispatches can also fan out in parallel.

**Phase 4 (after all 3 NSCS01/02/03 CUDA anchors land):** Council adjudicates
composition trainer design + greenlights trainer authoring. THEN 3-stack
composition smoke → full → paired CPU.

**Phase 5 (autopilot reseed):** Per the cathedral autopilot Catalog #227 Tier
C ranker, every successful anchor (CUDA + paired CPU) reseeds the empirical
posterior; the ranker auto-promotes class-shift substrates over within-class
refinements per the Z1 empirical revision.

---

## Section 5 — Op-routables (operator-decision items ranked by EV)

Ranked by `|predicted ΔS lower bound| / cost` per the META-ASSUMPTION REVIEW
ranking discipline.

### OR-1 (HIGHEST EV; immediate): Fire NSCS06 canary today

**EV:** predicted band `[0.10, 0.20]` HIGH VARIANCE per symposium council; cost
envelope ~$8 maximum. **$0.5–2.5 per ΔS-point at upper-bound predicted
realization.** Sole substrate ready-to-fire without gate-flip work.

**Action:**
```bash
export OPERATOR_AUTHORIZE_CONFIRMED_VIA_SESSION_DIRECTIVE=1
export OPERATOR_AUTHORIZE_SESSION_BUDGET_USD=15.00
.venv/bin/python tools/run_modal_smoke_before_full.py \
    --recipe substrate_nscs06_carmack_hotz_strip_everything_modal_t4_dispatch \
    --operator-handle "claude:nscs06_carmack_hotz_first_anchor"
```

### OR-2 (HIGH EV; parallel after OR-1): Generate 3 missing operator-authorize wrappers + auto-driver shells

**EV:** UNBLOCKS NSCS01/02/03 dispatches (cumulative EV equal to OR-3 / OR-4 / OR-5 below).

**Action:** Operator runs `tac.substrate_registry.driver_generator` (or
`scripts/operator_authorize_<recipe>.sh` template generator) for each of
NSCS01/02/03 recipes. Per Catalog #244 the auto-generated drivers will carry
the canonical 3-export NVML env block, Catalog #163 bootstrap sentinel,
Catalog #166 sentinel files, and Catalog #167 smoke-before-full wrapper
structure.

### OR-3 (HIGH EV; after OR-2): NSCS01 + NSCS02 parallel fire (after gate-flips)

**EV:** Each substrate predicted ΔS `[-0.015, -0.030]` at ~$15 max cost
⇒ **$0.5–2.0 per ΔS-point.** Combined parallel fan-out doubles throughput.

**Pre-flight:**
1. Operator flips `dispatch_enabled: true` on both recipes
2. Operator clears recipe `dispatch_blockers` (or downgrades to `dispatch_warnings`)
3. NSCS02 only: run no-train A1 resizing-chain ablation locally (~10 min CPU)

**Action (parallel via shell + nohup per CLAUDE.md "Codex CLI invocation Pattern A"):**
```bash
export OPERATOR_AUTHORIZE_CONFIRMED_VIA_SESSION_DIRECTIVE=1
export OPERATOR_AUTHORIZE_SESSION_BUDGET_USD=50.00
nohup bash -c '.venv/bin/python tools/run_modal_smoke_before_full.py \
    --recipe substrate_nscs01_nullspace_split_renderer_modal_t4_dispatch \
    --operator-handle "claude:nscs01_first_anchor"' \
    < /dev/null > .omx/tmp/nscs01_dispatch.log 2>&1 &
disown
nohup bash -c '.venv/bin/python tools/run_modal_smoke_before_full.py \
    --recipe substrate_nscs02_downsampled_renderer_modal_t4_dispatch \
    --operator-handle "claude:nscs02_first_anchor"' \
    < /dev/null > .omx/tmp/nscs02_dispatch.log 2>&1 &
disown
```

### OR-4 (MEDIUM-HIGH EV; sequential — A100 cost-class jump): NSCS03 fire (after gate-flips + Phase 2 council)

**EV:** Predicted ΔS `[-0.020, -0.040]` (end-to-end Ballé end-to-end is
strongest theoretical performer) at ~$80 max cost ⇒ **$2.0–4.0 per
ΔS-point.** Higher per-ΔS-point cost than NSCS01/02 but predicted upper
band larger.

**Critical sequencing per Catalog #239 boundary:** NSCS03 full at $60–$80
crosses the `long_burn` boundary (>$50). Catalog #239 BOUNDARY-OPEN ensures
the routing is structurally correct, but the cost-class jump is real —
should be SEQUENTIAL (not parallel) after smoke confirms green per Catalog
#173 `independent_substrate` AND #271 codex pre-dispatch review at
cost-gated >$1.

**Pre-flight (council-grade per CLAUDE.md "Design decisions — non-negotiable"):**
1. Phase 2 council: λ_R sweep design (5/10 inner-council quintet sign-off)
2. Phase 2 council: σ-floor calibration
3. Operator flips `dispatch_enabled: true`

### OR-5 (DEFERRED-pending-design; HIGH long-term EV): Council adjudicates 3-stack composition trainer design

**EV:** Compound predicted ΔS `[-0.030, -0.090]` at composition full ~$120 max
⇒ **$1.3–4.0 per ΔS-point**, BUT the upside is the LARGEST single predicted
band since it stacks 3 orthogonal axes per ASSUMPTIONS-CHALLENGE-AUDIT.

**Action:** Council inner-quintet (Shannon / Dykstra / Yousfi / Fridrich /
Contrarian) + Assumption-Adversary seat per the sextet-pact (CLAUDE.md
"Council conduct" amendment 2026-05-15) adjudicate ONE of:

- Sequential-cascade trainer design (lowest risk, lowest predicted ΔS)
- Joint-multi-objective trainer design (highest risk, highest predicted ΔS)
- Substrate-engineering 3-substrate scaffold (PR101-style 600 LOC bind-all)

Per CLAUDE.md "Forbidden cross-archive composition without verified
[contest-CUDA] substrate anchor": composition is BLOCKED until at least 2-of-3
individuals (NSCS01/02 or NSCS01/03 or NSCS02/03) have landed contest-CUDA
anchors. **Predicted unblock date:** ~3-5 days after OR-3 + OR-4 complete.

### OR-6 (HOUSEKEEPING; LOW EV but high process-correctness): Harvest pending D4 100ep smoke (already harvested per ledger)

**Status:** ALREADY HARVESTED per ledger 2026-05-16T01:53:28 (event_type:
harvested for `fc-01KRPJZ9FY7N1HJH6HMEK6TX6C`). No action required —
mentioned in prompt as a potential outstanding item, but the canonical
`tools/harvest_modal_calls.py` has already completed the harvest. The
post-harvest cost-band anchor was logged at 2026-05-16T01:49:25 (rc=0,
6914.78s, $1.13 actual cost — within $0.50 of $0.35 prediction).

**If a NEW harvest is desired (e.g., to refresh `from-ledger` ingestion path
per Catalog #245):**

```bash
.venv/bin/python tools/harvest_modal_calls.py --execute --from-ledger
```

### OR-7 (META; for next session): Re-evaluate after NSCS06 first anchor lands

**EV:** Per Catalog #227 cathedral autopilot ranker + Z1 empirical revision,
the NSCS06 first anchor will reseed the cost-band posterior + composition
matrix. Re-rank the remaining 4 substrates against the updated empirical
band. May supersede OR-3 / OR-4 ordering if NSCS06 measures within band
[0.10, 0.20] (the predicted upper edge significantly outperforms PR101
0.193; would reroute all remaining budget to composition).

---

## Process compliance footer

- **Pre-flight reads completed (Catalog #229 premise verification):**
  CLAUDE.md (4 sections); 9-dim checklist memory; within-class vs class-shift
  memory; 4 recipe YAMLs; 3 canonical tool argparse surfaces (no flags invented);
  cost-band posterior (last 25 anchors); active claims (NSCS not in-flight; D4
  smoke `fc-01KRPJZ9FY7N1HJH6HMEK6TX6C` already harvested per ledger);
  trainer + driver script existence per substrate.
- **Checkpoint discipline (Catalog #206):** 3 checkpoints recorded at
  `.omx/state/subagent_progress.jsonl` with `lane_id=lane_dispatch_planning_4sub_3stack_20260515`.
- **Disjoint sister-subagent scope (Catalog #230):** `a4` audit READ-ONLY of
  source + writes `.omx/research/` audit memos (disjoint from this ledger);
  `b3` gate-author writes `src/tac/` + `CLAUDE.md` (disjoint from this
  read-only plan).
- **No CLI flags invented (CLAUDE.md "NEVER invent CLI flags"):** all flags
  verified via `grep "add_argument"` against the live tools (`tools/operator_authorize.py`,
  `tools/run_modal_smoke_before_full.py`, `tools/dispatch_modal_paired_auth_eval.py`,
  `tools/harvest_modal_calls.py`, `tools/local_pre_deploy_check.py`).
- **Apples-to-apples evidence discipline:** every score axis tagged
  `[contest-CUDA <gpu>]` / `[contest-CPU Linux x86_64 Modal]` / `[advisory: <reason>]`.
- **6-hook wire-in declaration (Catalog #125):**
  1. Sensitivity-map: N/A — this is a planning ledger, not a measurement.
  2. Pareto constraint: N/A — no new constraint declared.
  3. Bit-allocator hook: N/A — no per-tensor importance changes.
  4. Cathedral autopilot dispatch hook: ACTIVE — dispatches recommended in
     this plan will reseed the autopilot ranker after harvest.
  5. Continual-learning posterior update: ACTIVE — each dispatch will trigger
     `cost_band_calibration.append_anchor(outcome=...)` per Catalog #175/#177.
  6. Probe-disambiguator: N/A — no 2+ defensible interpretations to arbitrate
     at the plan level (per-substrate disambiguators are owned by per-substrate
     design memos).
