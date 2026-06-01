# Operator-routable command sheet — Slot WW canonical L28 PR98 patch wave + paired-CUDA RATIFICATION

**Date**: 2026-05-29
**Lane**: `lane_slot_ww_canonical_l28_pr98_patch_wave_build_only_top_4_canonical_frontier_candidates_paired_cuda_ratification_operator_routable_prep_20260529`
**Canonical frontier per Catalog #343**:
- CPU: sha `b7106c9bdbb8` score `0.19198533626623068` [contest-CPU]
- CUDA: sha `9cb989cef519` score `0.20533002902019143` [contest-CUDA]

**Per CLAUDE.md "Modal `.spawn()` HARVEST OR LOSE" non-negotiable + Catalog #246 paired CPU+CUDA discipline + canonical 11th ORDER directive Step 3 paired-CUDA RATIFICATION THIRD.**

## Scope reconciliation (HONEST classification per Slot QQ canonical META-lesson + Catalog #307)

Parent prompt named TOP-4 frontier candidates: V14-V2 DQS1 / fec6 / PR106 format0d / NSCS06 v8.

Per Slot SS canonical Phase B 24/26 L28_PATCH_MISSING_OPERATOR_ROUTABLE audit (`experiments/results/slot_ss_l28_pr98_cascade_application_audit_71_renderer_only_20260529T135256Z/empirical_verification.json`):

| Parent candidate | Maps to substrate inflate.py | In Slot SS scope? | Slot WW disposition |
|---|---|---|---|
| V14-V2 DQS1 (sha `7a0da5d0fc327cba`) | `pr101_lc_v2_clone` (V14-V2 is a DQS1 byte-variant of pr101 substrate) | YES (in 24 MISSING list) | **PATCHED** in `src/tac/substrates/pr101_lc_v2_clone/inflate.py` |
| fec6 (sha `b7106c9bdbb8`) | `pr101_lc_v2_clone` (fec6 is the canonical CPU frontier byte-variant of pr101 substrate via fp11_source_brotli_recode) | YES (in 24 MISSING list — same substrate as V14-V2) | **PATCHED** (same inflate.py as V14-V2 above) |
| PR106 format0d (sha `9cb989cef519`) | `submissions/pr106_*_sidecar/*` (sidecar architecture; multiple per-sidecar inflate.py files) | **OUT-OF-SCOPE** (sidecar architecture excluded from Slot OO 71-substrate renderer-only scope by design) | DEFERRED — separate per-sidecar L28 patch wave required (operator-routable as sister Slot subsequent) |
| NSCS06 v8 stacked | `nscs06_carmack_hotz_strip_everything` (v8 is a composite on the v7 baseline) | YES (in 24 MISSING list) | **PATCHED** in `src/tac/substrates/nscs06_carmack_hotz_strip_everything/inflate.py` |

**Net Slot WW BUILD scope**: 2 inflate.py patches + 2 paired-CUDA RATIFICATION recipes. PR106 format0d sidecar L28 patch wave deferred per HNeRV parity L4 inflate runtime budget + sidecar-architecture-specific patch geometry (the L28 PR98 pattern targets the full RGB renderer; sidecars adjust to the base substrate's output, not vice-versa).

## Phase A — Canonical L28 PR98 patches LANDED (BUILD complete)

### Patch 1 — `src/tac/substrates/pr101_lc_v2_clone/inflate.py`

- **Insertion point**: lines 100-122 (post-`F.interpolate` bicubic upsample, pre-`.clamp(0, 255).round().to(uint8)` cast).
- **Canonical helper**: `PR98_CHANNEL_BALANCE_OFFSETS_CANONICAL` imported from `tac.codec.pr98_channel_balance_zero_byte_bolt_on` (NOT copy-pasted; per operator binding directive #2 "no duplicative code").
- **Torch in-place pattern**: reshape `up` from `(batch*2, 3, H, W)` flat → `(batch, 2, 3, H, W)` paired → apply canonical offsets via `up_pair[:, frame, channel].sub_(value)` (3 in-place subtractions) → reshape back → clamp/round/uint8.
- **Verified imports clean** via `python -c "from tac.substrates.pr101_lc_v2_clone.inflate import inflate_one_video, main_cli"`.

### Patch 2 — `src/tac/substrates/nscs06_carmack_hotz_strip_everything/inflate.py`

- **Insertion point**: `inflate_one_video` per-pair write loop (after `_grayscale_plus_chroma_to_rgb` + `_affine_warp_frame1_from_frame0`, before per-frame raw byte write).
- **Canonical helper**: `PR98_CHANNEL_BALANCE_OFFSETS_CANONICAL` imported from `tac.codec.pr98_channel_balance_zero_byte_bolt_on` + canonical adapter function `_apply_pr98_l28_channel_balance_to_pair_uint8(frame_0, frame_1)` (int16 underflow-safe; clip back to uint8 [0, 255] per PR101 canonical clamp BEFORE round).
- **Numpy uint8 pattern**: int16 cast → apply canonical offsets per `PR98_CHANNEL_BALANCE_OFFSETS_CANONICAL` → `np.clip(..., 0, 255).astype(np.uint8)`.
- **Verified imports + L28 semantics + underflow clip** via `python -c "from tac.substrates.nscs06_carmack_hotz_strip_everything.inflate import ..."`.

## Phase B — Operator-authorize paired-CUDA RATIFICATION recipes LANDED

Two canonical recipes at `.omx/operator_authorize_recipes/`:

1. `slot_ww_l28_pr98_patch_pr101_lc_v2_clone_paired_modal_dispatch.yaml`
2. `slot_ww_l28_pr98_patch_nscs06_carmack_hotz_strip_everything_paired_modal_dispatch.yaml`

Per Catalog #170/#171/#181/#182/#215 canonical recipe schema fields: `min_vram_gb=16` + `video_input_strategy=per_dispatch_local_copy` + `pyav_decode_strategy=cpu_thread_async_upload` + `target_modes=[contest_one_video_replay]` + `canary_status=post_canary_dependent` + `min_smoke_gpu=T4`. Per Catalog #324 `predicted_band_validation_status=pending_post_training`. Per HNeRV parity L7 `lane_class=substrate_engineering`. Per Catalog #240 `dispatch_enabled=false` until operator authorization.

## Phase C — Canonical local pre-deploy check + smoke-before-full PV (operator-attended at dispatch time)

Operator runs (after authorization) for each candidate:

```bash
# 1. Local 30-second pre-deploy check (Catalog #243 8-check harness):
.venv/bin/python tools/local_pre_deploy_check.py \
  --recipe slot_ww_l28_pr98_patch_pr101_lc_v2_clone_paired_modal_dispatch \
  --strict

# 2. Smoke-before-full dry-run (Catalog #167):
.venv/bin/python tools/run_modal_smoke_before_full.py \
  --recipe slot_ww_l28_pr98_patch_pr101_lc_v2_clone_paired_modal_dispatch \
  --dry-run
```

Repeat for `slot_ww_l28_pr98_patch_nscs06_carmack_hotz_strip_everything_paired_modal_dispatch`.

## Phase D — Canonical paired-CUDA RATIFICATION (operator-attended)

**HARD CONSTRAINTS** per CLAUDE.md "GPU budget" + canonical Modal $5 hard-stop + Catalog #246 paired CPU+CUDA discipline:
- Per-candidate envelope: $0.30 (CPU $0.05 + CUDA T4 $0.25 estimated per Catalog #245 canonical Modal call_id ledger 4-layer pattern)
- Total envelope: $0.60 ($0.30 × 2 substrates; well under canonical Modal $5 hard-stop)
- Per Catalog #199 paired-env discipline: BOTH env vars REQUIRED for non-interactive subprocess dispatch.
- Per Catalog #271 PRE-DISPATCH-CODEX-REVIEW automation: codex adversarial review fires automatically for paid dispatches >$1 estimated cost; THIS recipe band $0.30 < $1 so cached canonical Codex review skipped per Catalog #271 cost gate.

### Candidate 1 — pr101_lc_v2_clone (V14-V2 DQS1 + fec6 frontier coverage)

```bash
OPERATOR_AUTHORIZE_CONFIRMED_VIA_SESSION_DIRECTIVE=1 \
OPERATOR_AUTHORIZE_SESSION_BUDGET_USD=5.00 \
.venv/bin/python tools/operator_authorize.py \
  --recipe slot_ww_l28_pr98_patch_pr101_lc_v2_clone_paired_modal_dispatch \
  --confirm
```

### Candidate 2 — nscs06_carmack_hotz_strip_everything (NSCS06 v8 stacked coverage)

```bash
OPERATOR_AUTHORIZE_CONFIRMED_VIA_SESSION_DIRECTIVE=1 \
OPERATOR_AUTHORIZE_SESSION_BUDGET_USD=5.00 \
.venv/bin/python tools/operator_authorize.py \
  --recipe slot_ww_l28_pr98_patch_nscs06_carmack_hotz_strip_everything_paired_modal_dispatch \
  --confirm
```

## Canonical FRONTIER-BREAKING acceptance criteria

Per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA, ON 1:1 CONTEST-COMPLIANT HARDWARE" + Catalog #246 + canonical "iterate not force":

**PR111-candidate disposition** — any of the 2 candidates achieves either:
- CPU axis < `0.19198533626623068` [contest-CPU] (current canonical frontier per Catalog #343), OR
- CUDA axis < `0.20533002902019143` [contest-CUDA] (current canonical frontier per Catalog #343)

→ candidate PROMOTES to PR111-candidate operator-attended RATIFICATION cascade. Canonical equation `pr98_zero_byte_decode_side_channel_balance_score_savings_v1` PROMOTION from FORMALIZATION_PENDING → REGISTERED per Catalog #344.

**Diagnostic-only disposition** — candidate achieves score in band `[predicted_band - 0.0005, predicted_band + 0.0005]`:

→ canonical PROCEED per Slot DD canonical band; canonical equation EmpiricalAnchor anchored at the empirical delta per Catalog #344.

**IMPLEMENTATION-LEVEL FALSIFICATION per Catalog #307** — candidate achieves score OUTSIDE band (e.g. -0.005 or worse, OR positive delta indicating L28 patch HURTS the score on this substrate):

→ canonical FALSIFICATION-of-the-specific-implementation per Catalog #307 + Catalog #308 alternative-probe enumeration per Slot QQ META-lesson. NSCS06-specific risk acknowledged in recipe per analytical-renderer-vs-PR95-family-substrate scorer-feedback geometry difference.

## Per Slot QQ canonical META-lesson — per-archive EMPIRICAL VERIFICATION REQUIRED

Slot SS canonical Phase A already verified per-substrate L28_APPLIED_EMPIRICALLY_VERIFIED via Slot LL helper for both pr101_lc_v2_clone + nscs06_carmack_hotz_strip_everything (canonical channel checks all True). The paired-CUDA RATIFICATION at dispatch time is the FULL empirical verification per Slot QQ META-lesson (predicted-band-overlay must be confirmed by per-archive empirical anchor BEFORE downstream consumers treat the delta as authoritative).

## Per Slot NN + PP + QQ canonical 3-retrospective-anchor META-pattern

The canonical apparatus IS working as designed:
- Slot NN: STAND_DOWN when sister-coherence-overlap detected at Phase 0 PV
- Slot PP: STAND_DOWN when sister-identity-predictor sister-coherence overlap detected at Phase 0 PV
- Slot QQ: IMPLEMENTATION-LEVEL FALSIFICATION when per-archive empirical verification reveals predicted overlay is artifact (not paradigm-level kill per Catalog #307)

Slot WW honored this canonical pattern at PHASE 0 PV: verified Slot SS canonical empirical artifact exists (`20507 bytes`); verified Slot LL canonical helper exists (`22.8K`); verified canonical frontier pointer (`b7106c9bdbb8 CPU + 9cb989cef519 CUDA`); verified no prior Slot WW landing memo (memory grep clean); verified 4 in-flight sister subagents (Slot TT/UU/VV per Catalog #340 DISJOINT); proceeded with HONEST scope reconciliation (4 parent candidates → 2 in-scope substrate inflate.py + 1 deferred sidecar + 1 same-substrate-as-another-candidate).

## Cross-references

- Canonical Slot LL helper: `src/tac/codec/pr98_channel_balance_zero_byte_bolt_on/__init__.py`
- Canonical Slot LL landing memo: `feedback_slot_ll_l28_pr98_zero_byte_decode_side_channel_balance_bolt_on_per_slot_dd_highest_ev_shortest_wc_rank_1_landed_20260529.md`
- Canonical Slot SS landing memo: `feedback_slot_ss_l28_pr98_cascade_application_audit_71_renderer_only_per_slot_oo_canonical_highest_ev_shortest_wall_clock_oproutable_landed_20260529.md`
- Canonical Slot SS empirical artifact: `experiments/results/slot_ss_l28_pr98_cascade_application_audit_71_renderer_only_20260529T135256Z/empirical_verification.json`
- Canonical Slot DD L14-L70 finding: `.omx/research/cross_pr_family_canonical_techniques_mining_L14_L70_20260529T075244Z.md`
- Canonical PR101 source-of-truth: `experiments/results/public_pr_intake_full/public_pr101_intake_20260505_auto/source/submissions/hnerv_ft_microcodec/inflate.py:49-51`
- Canonical frontier pointer: `.omx/state/canonical_frontier_pointer.json`
- Canonical Modal call_id ledger: `.omx/state/modal_call_id_ledger.jsonl`

**PR98 third-prize empirical anchor**: PR97 0.197 → PR98 0.196 (canonical -0.001 score delta) <!-- HISTORICAL_SCORE_LITERAL_OK:pr97_to_pr98_score_delta_l28_anchor_per_slot_dd_canonical_finding_2026-05-29 -->
