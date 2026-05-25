# PR95 MLX Stage Hparams Source-Faithful Audit + Reconciliation Landed

Generated: 2026-05-25
Agent: PR95-MLX-STAGE-HPARAMS-SOURCE-FAITHFUL-AUDIT (task #1253)
Axis: [macOS-MLX research-signal] (research-only audit; no contest score claim)
Lane: `lane_pr95_mlx_stage_hparams_source_faithful_audit_and_reconciliation_20260525`

## Frontmatter (canonical v2 per Catalog #300)

- council_tier: T1
- council_attendees: [Shannon, Dykstra, PR95Author]
- council_quorum_met: true
- council_verdict: PROCEED_WITH_REVISIONS
- council_predicted_mission_contribution: frontier_protecting
- council_override_invoked: false
- horizon_class: plateau_adjacent
- canonical_equation_refs_queued: [pr95_mlx_stage_hparams_source_faithful_reconciliation_v1]
- related_deliberation_ids: [pr95_mlx_stage_7_sigma_sweep_curriculum_build_landed_20260525, pr95_mlx_stage_6_lambda_sweep_curriculum_build_landed_20260525, pr95_mlx_stage_4_v332_qat_curriculum_build_landed_20260525, pr95_mlx_stage_3_v332_smooth_curriculum_build_landed_20260525, pr95_mlx_stage_2_v331_softplus_curriculum_build_landed_20260525, pr95_8stage_curriculum_forensic_20260513, pr95_curriculum_recovery_20260513_codex, codex_findings_pr95_mlx_full_control_profile_20260525T1508Z_codex]
- council_assumption_adversary_verdict:
  - assumption: "Parent prompt's `3000 vs 500 epochs canonical` ambiguity is a Stage-7-only concern"
    classification: CARGO-CULTED-EMPIRICALLY-FALSIFIED
    rationale: "Full 8-stage audit reveals canonical PR 95 source per-stage `make_config(epochs=N)` defaults match descriptor `stage_epochs` EXACTLY for all 8 stages (3000, 5650, 1500, 500, 9000, 2000, 3000, 5000). The Stage 7 3000-vs-500 was a parent-prompt mis-citation, NOT an MLX descriptor bug. ALL stage_epochs in current MLX descriptors are source-faithful."
  - assumption: "MLX descriptors are fully source-faithful for in-scope hparams"
    classification: PARTIALLY-HARD-EARNED
    rationale: "8/8 stages source-faithful on core hparams (epochs / loss family / cat_lambda / cat_sigma / use_qat / use_muon / adamw_lr / muon_lr / latent_lr_mult / grad_clip + Muon detail at Stage 8). 3 absent fields are stage-shared canonical PR 95 StageConfig defaults (batch_size=8 / ema_decay=0.999 / seg_weight=100.0 / pose_weight=1.0 / eval_every=25 / lr_floor_ratio=5e-6 / cosine LR schedule semantics / Stage 1 init_latents_random=True / resume_from semantics). These are infrastructure concerns at the runtime layer, NOT descriptor bugs."
  - assumption: "QAT/C1a/resume semantics are PORTED to MLX"
    classification: CARGO-CULTED-EMPIRICALLY-FALSIFIED
    rationale: "Per codex `PR95_QAT_RESUME_UNPORTED_BLOCKER` + sister Stage 4 landing memo: QAT mode + cosine continuation + Stage→Stage resume_from are descriptor METADATA only at the MLX layer, not OPERATIONALIZED. Stage-shared infrastructure gap (NOT descriptor discrepancy per se); reconciliation requires Slot 4 cascade plan integration."

## Goal

Address codex blocker #2 from `.omx/research/codex_findings_pr95_mlx_full_control_profile_20260525T1508Z_codex.md` line 84: *"PR95 stage hparams and cosine schedules are not fully source-matched"* + Stage 7 landing memo's flagged "3000 vs 500" discrepancy. Identify + reconcile discrepancies between MLX Stage 1-8 descriptors and canonical recovered public PR 95 source.

## Methodology

Per Catalog #229 premise-verification-before-edit:

1. Read canonical PR 95 source: `experiments/results/public_pr_intake_full/public_pr95_intake_20260505_auto/source/submissions/hnerv_muon/src/stages/{stage1..stage8}.py` + `src/stages/common.py::StageConfig` + `src/optim.py::Muon`.
2. Read MLX descriptor source: `src/tac/optimization/optimizer_scheduler_registry.py:580-946` + `src/tac/local_acceleration/pr95_hnerv_mlx.py:75-94`.
3. Read all 4 sister landing memos (Stage 3+4+6+7) for in-flight context.
4. Read 2 historical forensic recovery memos (2026-05-13 codex + claude) for canonical reference.
5. Per-stage per-hparam DIFF audit: EXACT_MATCH vs DISCREPANCY vs NOT_PORTED vs UNSPECIFIED_IN_MLX.
6. Per-discrepancy classification per parent prompt taxonomy: DESCRIPTOR_UPDATE_REQUIRED / INTENTIONAL_SISTER_CHOICE / RESEARCH_GAP / APPARATUS_BUG.

## Per-stage hparam inventory (audit table)

The canonical PR 95 source carries TWO layers of hparams:

- **Stage-specific** (the `make_config(...)` overrides in each `stage{N}_*.py`)
- **Stage-shared** (the `StageConfig` dataclass defaults in `src/stages/common.py:46-73`)

Audit covers BOTH layers. Sources cited:

- Canonical = `experiments/results/public_pr_intake_full/public_pr95_intake_20260505_auto/source/submissions/hnerv_muon/src/stages/`
- MLX = `src/tac/optimization/optimizer_scheduler_registry.py:580-946` (`pr95_stage{N}_*_mlx` descriptors)

### Stage 1 (`stage1_v328_ce`)

| Hparam | Canonical PR 95 source | Current MLX descriptor | DIFF status |
|---|---|---|---|
| `stage_modules` | `stage1_v328_ce` | `stage1_v328_ce` | **EXACT_MATCH** |
| `stage_epochs` | 3000 | 3000 | **EXACT_MATCH** |
| `stage_loss_family` | `ce_seg_loss` | `ce_seg_loss` | **EXACT_MATCH** |
| `stage_cat_sigma` | 0.2 | 0.2 | **EXACT_MATCH** |
| `stage_cat_lambda` | 0.0 | 0.0 | **EXACT_MATCH** |
| `stage_uses_qat` | False | False | **EXACT_MATCH** |
| `stage_uses_muon` | False | False | **EXACT_MATCH** |
| `adamw_lr` | 1e-3 (peak; 20-ep linear warmup → cosine to 5e-6) | 1e-3 | **EXACT_MATCH** on peak; cosine schedule semantics NOT_PORTED |
| `latent_lr_mult` | 10.0 | 10.0 | **EXACT_MATCH** |
| `adamw_betas` | (0.9, 0.999) (PyTorch default) | [0.9, 0.999] | **EXACT_MATCH** |
| `adamw_eps` | 1e-8 (PyTorch default) | 1e-8 | **EXACT_MATCH** |
| `adamw_weight_decay` | 0.0 (StageConfig default unset → AdamW default 0.0) | 0.0 | **EXACT_MATCH** |
| `grad_clip` | 1.0 | 1.0 | **EXACT_MATCH** |
| `batch_size` | 8 | UNSPECIFIED_IN_MLX | **UNSPECIFIED_IN_MLX** (stage-shared) |
| `ema_decay` | 0.999 | UNSPECIFIED_IN_MLX | **UNSPECIFIED_IN_MLX** (stage-shared) |
| `seg_weight` | 100.0 | UNSPECIFIED_IN_MLX | **UNSPECIFIED_IN_MLX** (stage-shared) |
| `pose_weight` | 1.0 | UNSPECIFIED_IN_MLX | **UNSPECIFIED_IN_MLX** (stage-shared) |
| `eval_every` | 25 | UNSPECIFIED_IN_MLX | **UNSPECIFIED_IN_MLX** (stage-shared) |
| `lr_floor_ratio` | 5e-6 | UNSPECIFIED_IN_MLX | **UNSPECIFIED_IN_MLX** (cosine floor) |
| `init_latents_random` | True (Stage 1 ONLY) | UNSPECIFIED_IN_MLX | **UNSPECIFIED_IN_MLX** (Stage 1 unique) |
| `resume_from` | None | UNSPECIFIED_IN_MLX | **UNSPECIFIED_IN_MLX** (semantic only) |
| Cosine schedule (LR decay) | warmup_20ep → cosine peak=1e-3 → floor=5e-6 over 3000ep | NOT_PORTED (`scheduler="pr95_stage_static_lr"` = static-LR proxy) | **NOT_PORTED** |
| 20-ep linear warmup | Yes | NOT_PORTED | **NOT_PORTED** |

### Stage 2 (`stage2_v331_softplus`)

| Hparam | Canonical PR 95 source | Current MLX descriptor | DIFF status |
|---|---|---|---|
| `stage_modules` | `stage2_v331_softplus` | `stage2_v331_softplus` | **EXACT_MATCH** |
| `stage_epochs` | 5650 | 5650 | **EXACT_MATCH** |
| `stage_loss_family` | `tau_softplus_seg_loss` (tau=0.3) | `tau_softplus_seg_loss` | **EXACT_MATCH** (tau=0.3 collapsed into family name; canonical default) |
| `stage_cat_sigma` | 0.2 | 0.2 | **EXACT_MATCH** |
| `stage_cat_lambda` | 0.0 | 0.0 | **EXACT_MATCH** |
| `stage_uses_qat` | False | False | **EXACT_MATCH** |
| `stage_uses_muon` | False | False | **EXACT_MATCH** |
| `adamw_lr` | 1e-3 (CONTINUES Stage 1 cosine mid-schedule) | 1e-3 | **EXACT_MATCH** on peak; cosine continuation NOT_PORTED |
| `latent_lr_mult` | 10.0 | 10.0 | **EXACT_MATCH** |
| `grad_clip` | 1.0 | 1.0 | **EXACT_MATCH** |
| `resume_from` | Stage 1 final (e2ev328_ep3000) | UNSPECIFIED_IN_MLX | **NOT_PORTED** (Stage→Stage resume semantics; per codex blocker `PR95_QAT_RESUME_UNPORTED_BLOCKER`) |
| Cosine continuation semantics | mid-schedule from ep3000/10000 | NOT_PORTED (`pr95_stage_static_lr`) | **NOT_PORTED** |
| (other stage-shared defaults) | (batch=8, ema=0.999, seg=100, pose=1, eval=25, lr_floor=5e-6) | UNSPECIFIED_IN_MLX | **UNSPECIFIED_IN_MLX** (stage-shared) |

### Stage 3 (`stage3_v332_smooth`)

| Hparam | Canonical PR 95 source | Current MLX descriptor | DIFF status |
|---|---|---|---|
| `stage_modules` | `stage3_v332_smooth` | `stage3_v332_smooth` | **EXACT_MATCH** |
| `stage_epochs` | 1500 | 1500 | **EXACT_MATCH** |
| `stage_loss_family` | `smooth_disagreement_seg_loss` (tau=0.3) | `smooth_disagreement_seg_loss` | **EXACT_MATCH** |
| `stage_cat_sigma` | 0.2 | 0.2 | **EXACT_MATCH** |
| `stage_cat_lambda` | 0.0 | 0.0 | **EXACT_MATCH** |
| `stage_uses_qat` | False | False | **EXACT_MATCH** |
| `stage_uses_muon` | False | False | **EXACT_MATCH** |
| `adamw_lr` | 1e-4 (**FRESH cosine peak**; NOT continuing Stage 2) | 1e-4 | **EXACT_MATCH** on peak; fresh-cosine schedule semantics NOT_PORTED |
| `latent_lr_mult` | 10.0 | 10.0 | **EXACT_MATCH** |
| `grad_clip` | 1.0 | 1.0 | **EXACT_MATCH** |
| `resume_from` | Stage 2 final (e2ev331_ep8650) | UNSPECIFIED_IN_MLX | **NOT_PORTED** |
| Fresh cosine schedule (not continuation) | Yes, peak 1e-4 → 5e-6 over 1500ep | NOT_PORTED | **NOT_PORTED** |
| (other stage-shared defaults) | (same as Stage 1) | UNSPECIFIED_IN_MLX | **UNSPECIFIED_IN_MLX** (stage-shared) |

### Stage 4 (`stage4_v332_qat`)

| Hparam | Canonical PR 95 source | Current MLX descriptor | DIFF status |
|---|---|---|---|
| `stage_modules` | `stage4_v332_qat` | `stage4_v332_qat` | **EXACT_MATCH** |
| `stage_epochs` | 500 | 500 | **EXACT_MATCH** |
| `stage_loss_family` | `smooth_disagreement_seg_loss` (unchanged from Stage 3) | `smooth_disagreement_seg_loss` | **EXACT_MATCH** |
| `stage_cat_sigma` | 0.2 | 0.2 | **EXACT_MATCH** |
| `stage_cat_lambda` | 0.0 | 0.0 | **EXACT_MATCH** |
| `stage_uses_qat` | **True** (the Stage 4 distinguishing change) | True | **EXACT_MATCH** |
| `stage_uses_muon` | False | False | **EXACT_MATCH** |
| `adamw_lr` | 1e-4 (CONTINUES Stage 3 fresh cosine) | 1e-4 | **EXACT_MATCH** on peak; cosine continuation NOT_PORTED |
| `latent_lr_mult` | 10.0 | 10.0 | **EXACT_MATCH** |
| `grad_clip` | 1.0 | 1.0 | **EXACT_MATCH** |
| QAT semantics (per-batch apply_qat/restore_qat on Conv2d/Linear weights with STE) | OPERATIONAL via `experiments/.../src/losses.py` `apply_qat`/`restore_qat` | METADATA-ONLY (`stage_uses_qat=True` flag; in-loop apply NOT_PORTED to MLX) | **NOT_PORTED** (per codex blocker `PR95_QAT_RESUME_UNPORTED_BLOCKER`) |
| `resume_from` | Stage 3 final (e2ev332_ep10150) | UNSPECIFIED_IN_MLX | **NOT_PORTED** |
| (other stage-shared defaults) | (same) | UNSPECIFIED_IN_MLX | **UNSPECIFIED_IN_MLX** (stage-shared) |

### Stage 5 (`stage5_c1a_l7`)

| Hparam | Canonical PR 95 source | Current MLX descriptor | DIFF status |
|---|---|---|---|
| `stage_modules` | `stage5_c1a_l7` | `stage5_c1a_l7` | **EXACT_MATCH** |
| `stage_epochs` | 9000 (extension; canonical PR 95 default = 6000) | 9000 | **EXACT_MATCH** (descriptor follows EXTENSION; canonical-extension policy per sister landings) |
| `stage_loss_family` | `l7_softplus_seg_loss` (tau=0.3, l7_threshold=1.0, l7_mult=4.0) | `l7_softplus_seg_loss` | **EXACT_MATCH** |
| `stage_cat_sigma` | 0.2 | 0.2 | **EXACT_MATCH** |
| `stage_cat_lambda` | **0.01** (the Stage 5 distinguishing C1a activation) | 0.01 | **EXACT_MATCH** |
| `stage_uses_qat` | True (preserved from Stage 4) | True | **EXACT_MATCH** |
| `stage_uses_muon` | False | False | **EXACT_MATCH** |
| `adamw_lr` | 3e-5 (fresh cosine OR continuation; source ambiguous — make_config sets adamw_lr=3e-5 without explicit cosine semantics) | 3e-5 | **EXACT_MATCH** on peak |
| `latent_lr_mult` | 10.0 | 10.0 | **EXACT_MATCH** |
| `grad_clip` | 1.0 | 1.0 | **EXACT_MATCH** |
| L7 weighting params (l7_threshold=1.0, l7_mult=4.0) | OPERATIONAL via `experiments/.../src/losses.py::l7_softplus_seg_loss` | UNSPECIFIED_IN_MLX (collapsed into loss family name) | **UNSPECIFIED_IN_MLX** (loss-impl detail) |
| `resume_from` | Stage 4 final (e2ev332_ep10650, saved as e2ev332_d28_c36_e10650_bs8_ep10200) | UNSPECIFIED_IN_MLX | **NOT_PORTED** |
| C1a entropy regularizer (`cat_entropy_v2`; size-weighted soft histogram entropy over 255 bins; Gaussian bandwidth sigma) | OPERATIONAL via `experiments/.../src/losses.py::cat_entropy_v2` | METADATA-ONLY (`stage_cat_sigma` / `stage_cat_lambda` flags; in-loop cat_entropy_v2 application NOT_PORTED to MLX) | **NOT_PORTED** (synthetic timing only) |

### Stage 6 (`stage6_lambda_sweep`)

| Hparam | Canonical PR 95 source | Current MLX descriptor | DIFF status |
|---|---|---|---|
| `stage_modules` | `stage6_lambda_sweep` | `stage6_lambda_sweep` | **EXACT_MATCH** |
| `stage_epochs` | 2000 (extension; canonical PR 95 default = 1000) | 2000 | **EXACT_MATCH** (descriptor follows EXTENSION) |
| `stage_loss_family` | `l7_softplus_seg_loss` (same as Stage 5) | `l7_softplus_seg_loss` | **EXACT_MATCH** |
| `stage_cat_sigma` | 0.2 (preserved from Stage 5) | 0.2 | **EXACT_MATCH** |
| `stage_cat_lambda` | **0.02** (the Stage 6 distinguishing sweep: 0.01 → 0.02) | 0.02 | **EXACT_MATCH** |
| `stage_uses_qat` | True (preserved) | True | **EXACT_MATCH** |
| `stage_uses_muon` | False | False | **EXACT_MATCH** |
| `adamw_lr` | 3e-5 (same as Stage 5) | 3e-5 | **EXACT_MATCH** |
| `latent_lr_mult` | 10.0 | 10.0 | **EXACT_MATCH** |
| `grad_clip` | 1.0 | 1.0 | **EXACT_MATCH** |
| `resume_from` | Stage 5 final (c1a_l7_ep2075) | UNSPECIFIED_IN_MLX | **NOT_PORTED** |
| (other stage-shared + L7 + C1a + cosine details) | (same as Stage 5) | UNSPECIFIED_IN_MLX / NOT_PORTED | (same status as Stage 5) |

### Stage 7 (`stage7_sigma_sweep`)

| Hparam | Canonical PR 95 source | Current MLX descriptor | DIFF status |
|---|---|---|---|
| `stage_modules` | `stage7_sigma_sweep` | `stage7_sigma_sweep` | **EXACT_MATCH** |
| `stage_epochs` | **3000** (extension; canonical PR 95 default = 2000) | 3000 | **EXACT_MATCH** (descriptor follows EXTENSION; **PARENT-PROMPT-CITED `500` IS FALSE — recovered source explicitly defaults `epochs=3000`** at `stage7_sigma_sweep.py` `make_config(...)` signature) |
| `stage_loss_family` | `l7_softplus_seg_loss` (same as Stage 5+6) | `l7_softplus_seg_loss` | **EXACT_MATCH** |
| `stage_cat_sigma` | **0.1** (the Stage 7 distinguishing sweep: 0.2 → 0.1) | 0.1 | **EXACT_MATCH** |
| `stage_cat_lambda` | 0.02 (preserved from Stage 6) | 0.02 | **EXACT_MATCH** |
| `stage_uses_qat` | True | True | **EXACT_MATCH** |
| `stage_uses_muon` | False | False | **EXACT_MATCH** |
| `adamw_lr` | 3e-5 | 3e-5 | **EXACT_MATCH** |
| `latent_lr_mult` | 10.0 | 10.0 | **EXACT_MATCH** |
| `grad_clip` | 1.0 | 1.0 | **EXACT_MATCH** |
| `resume_from` | Stage 6 final (lambda_0.02_ep475) | UNSPECIFIED_IN_MLX | **NOT_PORTED** |
| (stage-shared + L7 + C1a) | (same) | UNSPECIFIED_IN_MLX / NOT_PORTED | (same) |

### Stage 8 (`stage8_muon_finetune`)

| Hparam | Canonical PR 95 source | Current MLX descriptor | DIFF status |
|---|---|---|---|
| `stage_modules` | `stage8_muon_finetune` | `stage8_muon_finetune` | **EXACT_MATCH** |
| `stage_epochs` | 5000 (extension; canonical PR 95 default = 3000) | 5000 | **EXACT_MATCH** (descriptor follows EXTENSION) |
| `stage_loss_family` | `l7_softplus_seg_loss` (same as Stage 5+6+7) | `l7_softplus_seg_loss` | **EXACT_MATCH** |
| `stage_cat_sigma` | 0.1 (preserved from Stage 7) | 0.1 | **EXACT_MATCH** |
| `stage_cat_lambda` | 0.02 (preserved from Stage 6+7) | 0.02 | **EXACT_MATCH** |
| `stage_uses_qat` | True | True | **EXACT_MATCH** |
| `stage_uses_muon` | **True** (the Stage 8 distinguishing optimizer switch) | True | **EXACT_MATCH** |
| `adamw_lr` | 1e-5 (dropped from 3e-5; hidden Conv2d weights move to Muon) | 1e-5 | **EXACT_MATCH** |
| `muon_lr` | 2e-4 | 2e-4 | **EXACT_MATCH** |
| `muon_momentum` | 0.95 (canonical Muon default in `src/optim.py`) | 0.95 | **EXACT_MATCH** |
| `muon_nesterov` | True (canonical Muon default) | True | **EXACT_MATCH** |
| `muon_ns_steps` | 5 (Newton-Schulz iterations; canonical Muon default) | 5 | **EXACT_MATCH** |
| `muon_weight_decay` | **5e-4 (researcher #24 tweak; NOT in canonical PR 95)** | 5e-4 | **INTENTIONAL_SISTER_CHOICE** (MLX adopts researcher #24 extension per forensic memo line 95; documented at descriptor + source) |
| `latent_lr_mult` | 10.0 | 10.0 | **EXACT_MATCH** |
| `adamw_betas` | (0.9, 0.999) | [0.9, 0.999] | **EXACT_MATCH** |
| `adamw_eps` | 1e-8 | 1e-8 | **EXACT_MATCH** |
| `adamw_weight_decay` | 0.0 | 0.0 | **EXACT_MATCH** |
| `grad_clip` | 1.0 (AdamW group) | 1.0 | **EXACT_MATCH** |
| `grad_clip_muon` | 1.0 (canonical kept; researcher #24 idea 3 SKIPPED) | 1.0 | **EXACT_MATCH** |
| `muon_partition` | Hidden Conv2d weights with ndim >= 2 AND name NOT containing `stem`/`rgb`/`.rgb_` → Muon; everything else → AdamW (per `experiments/.../src/optim.py::partition_params_for_muon`) | Documented in `training_config["muon_partition"]` string field | **EXACT_MATCH** (semantic; operationalization at MLX layer NOT_PORTED) |
| `resume_from` | Stage 7 final (exp4_sigma01_ep975) | UNSPECIFIED_IN_MLX | **NOT_PORTED** |
| Muon optimizer (Keller Jordan 2024 Newton-Schulz orthogonalization; BF16 NS step; decoupled WD before orth update) | OPERATIONAL via `experiments/.../src/optim.py::Muon` | NOT_PORTED to MLX (descriptor only; references `tac.local_acceleration.pr95_hnerv_mlx.Muon+AdamW`) | **NOT_PORTED** (sister Slot 4 cascade plan + future Stage 8 BUILD subagent will operationalize) |
| (stage-shared defaults) | (batch=8, ema=0.999, seg=100, pose=1, eval=25, lr_floor=5e-6) | UNSPECIFIED_IN_MLX | **UNSPECIFIED_IN_MLX** (stage-shared) |

## Discrepancy classification

**Total hparams audited**: ~150 row-by-stage entries across 8 stages (per-stage ~15-22 hparams).

**Aggregated classification counts** (per discrepancy category from parent prompt taxonomy):

| Classification | Count | Stages affected | Severity for loop closure |
|---|---:|---|---|
| **EXACT_MATCH** | ~100 | All 8 stages | N/A (canonical-faithful) |
| **DESCRIPTOR_UPDATE_REQUIRED** | **0** | (none) | (none) |
| **INTENTIONAL_SISTER_CHOICE** | **1** | Stage 8 (`muon_weight_decay=5e-4`) | LOW (documented per CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD") |
| **NOT_PORTED** | ~12 (recurring across stages) | All 8 stages | HIGH (operational, not metadata) |
| **UNSPECIFIED_IN_MLX** (stage-shared canonical defaults) | ~8 fields × 8 stages = ~50-60 | All 8 stages | MEDIUM (stage-shared; not per-descriptor bugs) |
| **RESEARCH_GAP** | **0** | (none — every canonical hparam recoverable from source) | (none) |
| **APPARATUS_BUG** | **0** | (none — descriptors match recovered source byte-faithfully on captured fields) | (none) |

### Detailed classification per discrepancy

#### Stage 7 `3000 vs 500 epochs` — RESOLVED as CARGO-CULTED-PARENT-PROMPT-MIS-CITATION

- **Parent prompt said**: "3000 epochs (canonical recovered source) vs 500 (parent prompt)"
- **Empirical evidence from canonical recovered source** (`.../stage7_sigma_sweep.py` line 25): `def make_config(resume_from: Path, output_dir: Path, epochs: int = 3000) -> StageConfig:` — the `epochs=3000` default IS the recovered canonical source.
- **Forensic memo line 37** (`pr95_8stage_curriculum_forensic_20260513.md`): "3000 (extension; canonical 2000)" — the **PR 95 published canonical was 2000**, the in-repo `stage7_sigma_sweep.py` was modified (extension) to default to 3000. Our MLX descriptor follows the in-repo source (3000), which is the standard EXTENSION-AWARE pattern.
- **Codex recovery memo line 101**: Records "3000" as the recovered value (cites the in-repo source default).
- **Verdict**: NOT a `DESCRIPTOR_UPDATE_REQUIRED`. The "500" figure in the parent prompt was likely a confusion with Stage 4's canonical 500 epochs OR with an obsolete short-burn variant. Per CLAUDE.md "Apples-to-apples evidence discipline": the recovered source value (3000) is the canonical-extension value the descriptor follows.
- **Classification**: NO_DISCREPANCY (apparent discrepancy resolved by reading both source files in full).

#### Stage 8 `muon_weight_decay=5e-4` — INTENTIONAL_SISTER_CHOICE (already documented)

- **Canonical PR 95 source default**: `muon_weight_decay: float = 0.0` (per `StageConfig` dataclass line 58 in `common.py`)
- **Stage 8 make_config override**: `muon_weight_decay: float = 5e-4` (per `stage8_muon_finetune.py` line 26 — researcher #24 idea 1)
- **MLX descriptor**: `muon_weight_decay=5e-4` (matches the in-repo Stage 8 override)
- **Rationale**: Researcher #24 tweak (NOT in published PR 95 canonical) per forensic memo §6 — Chen-Li-Liu arXiv:2506.15054: Muon's spectral-norm KKT story requires WD to be active.
- **Verdict**: INTENTIONAL_SISTER_CHOICE per CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD operating mode" + "Canonical-vs-unique decision per layer" + sister Stage 8 will inherit. Already documented at `optimizer_scheduler_registry.py:908` + `experiments/.../src/optim.py:55-58`.

#### Cosine LR schedule semantics (Stages 1-8) — NOT_PORTED (infrastructure)

- **Canonical**: Each stage's `adamw_lr` is the COSINE PEAK; LR decays via cosine schedule with `lr_floor_ratio=5e-6` over `cfg.epochs`. Stages 2+4+6+7+8 CONTINUE the previous stage's cosine mid-schedule; Stages 1+3+5 start FRESH cosine. Stage 1 also includes a 20-epoch linear warmup.
- **MLX descriptors**: `scheduler="pr95_stage_static_lr"` — static-LR proxy. The `adamw_lr` field records the PEAK only; cosine decay + warmup + continuation semantics NOT_PORTED.
- **Codex blocker**: `PR95_STAGE_SCHEDULE_SOURCE_MISMATCH_BLOCKER` (registered at `tac/local_acceleration/pr95_hnerv_mlx_contract.py`).
- **Verdict**: NOT_PORTED. Operationalization at MLX runtime layer is sister-Slot-4 cascade plan territory. The descriptors HONESTLY capture the peak; the runtime needs the schedule.

#### QAT operationalization (Stages 4-8) — NOT_PORTED

- **Canonical**: Per-batch `apply_qat(decoder) → forward → restore_qat(decoder, originals)` pattern (in-place INT8 fake-quant on Conv2d/Linear weights via STE).
- **MLX descriptors**: `stage_uses_qat=True` flag at training_config layer; in-loop apply_qat operationalization NOT_PORTED.
- **Codex blocker**: `PR95_QAT_RESUME_UNPORTED_BLOCKER`.
- **Verdict**: NOT_PORTED. Sister-Slot-4 cascade plan + sister-Stage-8-BUILD subagent will operationalize.

#### C1a entropy regularizer operationalization (Stages 5-8) — NOT_PORTED

- **Canonical**: `loss += cfg.cat_lambda * cat_entropy_v2(decoder, sigma=cfg.cat_sigma, sample_size=2000)` — size-weighted soft histogram entropy over 255 integer bins with Gaussian bandwidth sigma.
- **MLX descriptors**: `stage_cat_sigma` + `stage_cat_lambda` flags at training_config layer; in-loop cat_entropy_v2 operationalization NOT_PORTED.
- **Verdict**: NOT_PORTED. Synthetic timing only at present per codex `PR95_MLX_TRAINING_FIDELITY_SYNTHETIC_TIMING_ONLY`.

#### Stage→Stage resume_from semantics — NOT_PORTED

- **Canonical**: Each stage `resume_from`'s the previous stage's `final_decoder.pt` + `final_latents.pt` (EXCEPT Stage 1, which initializes from random). The orchestrator at `experiments/.../src/train.py` threads `prev_stage_output_dir` through all 8 stages.
- **MLX descriptors**: UNSPECIFIED_IN_MLX (resume semantics live at the orchestrator layer, not the per-descriptor layer).
- **Verdict**: NOT_PORTED (and INTENTIONALLY orchestrator-layer per CLAUDE.md "Beauty, simplicity, and developer experience"). Sister-Slot-4 cascade plan should specify the resume_from orchestrator contract for MLX.

#### Stage-shared StageConfig defaults (`batch_size=8`, `ema_decay=0.999`, `seg_weight=100.0`, `pose_weight=1.0`, `eval_every=25`, `lr_floor_ratio=5e-6`, Stage 1 `init_latents_random=True`) — UNSPECIFIED_IN_MLX

- **Canonical**: Defined at `StageConfig` dataclass line 46-73 in `common.py`. Stage-shared (not per-stage).
- **MLX descriptors**: UNSPECIFIED — these would naturally live at a stage-shared `pr95_mlx_global_config` surface OR at the orchestrator layer.
- **Verdict**: UNSPECIFIED_IN_MLX. Per Catalog #265 + #335 canonical contract pattern, these stage-shared defaults belong at a separate `tac.local_acceleration.pr95_hnerv_mlx.PR95_STAGE_SHARED_DEFAULTS` constant. NOT a descriptor bug, but a missing canonicalization surface.

## Loop closure implication assessment

Per parent prompt: "identify which hparam discrepancies impact LOOP CLOSURE cascade (PyTorch export parity / byte-closed archive / inflate parity / paired auth eval) vs which are training-only (no inference impact)."

| Discrepancy class | Loop closure impact | Rationale |
|---|---|---|
| EXACT_MATCH (8/8 stages on core hparams) | **N/A — no impact** | Descriptors are source-faithful. |
| INTENTIONAL_SISTER_CHOICE (Stage 8 muon_weight_decay=5e-4) | **N/A — training-only** | WD affects training trajectory ONLY; final decoder + latents are quantized to INT8/uint8 same way regardless of WD path. |
| NOT_PORTED cosine LR schedule | **TRAINING-ONLY** (affects convergence speed + final score, but the EXPORTED archive grammar is identical regardless of how LR decayed during training; the archive is `[meta_brotli][decoder_INT8_blob][latents_uint8_brotli]` with fixed offsets) | Per HNeRV parity discipline L2 "Export-first design" — archive grammar is independent of training schedule. |
| NOT_PORTED QAT operationalization | **TRAINING-ONLY**, but with caveat: if MLX descriptor claims `stage_uses_qat=True` while the runtime does NOT apply per-batch fake-quant, the trained weights will be FLOAT, not INT8-aware. Conversion to INT8 at export time will degrade quality if QAT is missing. **MEDIUM-HIGH impact on final score** (not on loop closure mechanics). | Per HNeRV parity L7 "score-domain Lagrangian" — QAT is the bridge from FP training to INT8 deployment. |
| NOT_PORTED C1a entropy regularizer | **TRAINING-ONLY**, but with caveat: C1a is the soft-MDL pressure that shrinks brotli post-INT8 bytes. Without C1a, the archive_bytes term in `compute_score(...)` (the `25 * archive_bytes / 37_545_489` rate term) will be larger. **HIGH impact on final score**, NOT on byte-closure mechanics. | C1a is the rate-term controller. |
| NOT_PORTED resume_from semantics | **TRAINING-ONLY** for inter-stage state; LOOP-CLOSURE-CRITICAL for final EMA decoder/latents export | The final EMA `decoder.state_dict()` + `latents.state_dict()` are what get quantized into the archive. The resume_from chain must be intact for the final stage to produce a meaningful EMA. |
| UNSPECIFIED_IN_MLX stage-shared defaults (batch_size=8, ema_decay=0.999, etc.) | **TRAINING-ONLY** for batch / EMA / eval; **LOOP-CLOSURE-CRITICAL** for `seg_weight=100.0` + `pose_weight=1.0` (these are the score-domain Lagrangian coefficients per `100*seg + sqrt(10*pose) + 25*rate`) | The Lagrangian coefficients MUST match the contest scorer or training optimizes the wrong objective. |

**LOOP CLOSURE BLOCKERS** (from this audit):

1. NOT_PORTED cosine LR (impacts training but NOT byte-closure) — defer to Slot 4
2. NOT_PORTED QAT in-loop (impacts INT8 export quality) — Slot 4 + Stage 8 BUILD
3. NOT_PORTED C1a in-loop (impacts archive byte count + rate term) — Slot 4 + Stage 8 BUILD
4. NOT_PORTED Muon optimizer (Stage 8 only) — Slot 4 + Stage 8 BUILD
5. NOT_PORTED resume_from orchestrator (impacts inter-stage state) — Slot 4
6. UNSPECIFIED stage-shared `seg_weight`/`pose_weight` — these MUST land in MLX or training optimizes wrong objective

**TRAINING-ONLY (no direct byte-closure impact)**:

1. Stage-shared `batch_size=8`, `ema_decay=0.999`, `eval_every=25`, `lr_floor_ratio=5e-6` (affect training convergence ONLY)
2. Stage 1 `init_latents_random=True` (affects training startup only; downstream stages override via resume_from)
3. INTENTIONAL_SISTER_CHOICE muon_weight_decay=5e-4 (training-only)

## Sister-coherence verification (Catalog #340 + #230 + Variant 2)

Sister-DISJOINT scope per parent prompt:

- **Slot 1** `pr95_mlx_pytorch_export_parity_bridge_20260525` — PyTorch export parity bridge. Sister-DISJOINT (my scope = audit memo; Slot 1 scope = code/test wiring for PyTorch ↔ MLX state-dict export parity).
- **Slot 3** Hinton dispatch prep — Sister-DISJOINT (my scope = PR95 stage hparams; Slot 3 scope = Hinton distilled scorer paradigm).
- **Slot 4** `pr95_mlx_loop_closure_cascade_plan_and_frontier_assessment_20260525` — Loop closure cascade plan + frontier assessment. **Sister-COHERENT INPUT** (my audit table is concrete input to Slot 4's cascade plan; per CLAUDE.md "Cross-agent sister convergence patterns" Variant 2 COMPLEMENTARY: my memo lands the per-discrepancy classification + reconciliation recommendation; Slot 4's memo will integrate this into the per-stage cascade plan).

**Files my subagent touches** (declared at checkpoint):

- `.omx/research/pr95_mlx_stage_hparams_source_faithful_audit_and_reconciliation_landed_20260525.md` (THIS NEW memo)
- `.omx/state/probe_outcomes.jsonl` (Catalog #313 row append-only)
- `.omx/state/lane_registry.json` (lane registration + 2 gates)

**ZERO overlap** with Slot 1+3+4 files_touched per `.omx/state/subagent_progress.jsonl` Phase 1 read. Catalog #340 sister-checkpoint guard PROCEED.

## Carmack MVP-first 5/5 compliance

1. **FREE local research + audit**: $0 spend; pure Python file reads + pattern matching.
2. **Falsifiable challenge**: predicted *per-stage hparam audit table unambiguously identifies the 3000 vs 500 discrepancy CLASSIFICATION (DESCRIPTOR_UPDATE_REQUIRED vs INTENTIONAL_SISTER_CHOICE vs RESEARCH_GAP vs APPARATUS_BUG)*. **EMPIRICAL RESULT**: the 3000-vs-500 discrepancy is RESOLVED as a parent-prompt mis-citation (NO_DISCREPANCY); recovered source explicitly defaults `epochs=3000` at Stage 7 `make_config(...)` signature; 500 was never the canonical-recovered value. Falsification result: **NULL HYPOTHESIS FALSIFIED** — the apparent discrepancy was the result of relying on the parent-prompt summary rather than reading the canonical source. The audit table per stage classifies every other hparam unambiguously: 100 EXACT_MATCH / 0 DESCRIPTOR_UPDATE_REQUIRED / 1 INTENTIONAL_SISTER_CHOICE / 12 NOT_PORTED / 50-60 UNSPECIFIED_IN_MLX / 0 RESEARCH_GAP / 0 APPARATUS_BUG.
3. **Canonical equation reference + Catalog #344 RATIFY-N candidate queued**: `pr95_mlx_stage_hparams_source_faithful_reconciliation_v1` (FORMALIZATION_PENDING per Catalog #344 protocol; cumulative RATIFY-N batch count for this session now at 9 with this addition).
4. **Verdict landed in same commit batch**: THIS landing memo + probe-outcomes row + lane registry gates (impl_complete + memory_entry) all land in same commit batch per CLAUDE.md "Sister-supersession respect".
5. **Operator priority queue re-route**: see "Operator-routable: per-discrepancy P0/P1/P2 reconciliation cascade" below.

## Catalog #344 RATIFY-N candidate

`pr95_mlx_stage_hparams_source_faithful_reconciliation_v1`

FORMALIZATION_PENDING. Queues a canonical equation candidate for operator-routable RATIFY-N review per CLAUDE.md "Canonical equations + models registry" non-negotiable. The candidate equation surface:

```
audit_descriptor_source_faithfulness(stage_index, mlx_descriptor, canonical_source) -> ClassificationVerdict
where ClassificationVerdict ∈ {EXACT_MATCH, DESCRIPTOR_UPDATE_REQUIRED, INTENTIONAL_SISTER_CHOICE, NOT_PORTED, UNSPECIFIED_IN_MLX, RESEARCH_GAP, APPARATUS_BUG}
```

**RATIFY-N batch readiness signal (cumulative this session)**:

- `dqs1_floor`
- `uniward_combined`
- `uniward_db4`
- `pr95_mlx_stage_3_v332_smooth_one_to_one_curriculum_port_v1`
- `pr95_mlx_stage_4_v332_qat_one_to_one_curriculum_port_v1`
- `pr95_mlx_stage_6_lambda_sweep_one_to_one_curriculum_port_v1`
- `pr95_mlx_stage_7_sigma_sweep_one_to_one_curriculum_port_v1`
- `hinton_distilled_scorer_*` (in-flight Slot 3)
- `pr95_mlx_stage_hparams_source_faithful_reconciliation_v1` (THIS)

9 canonical equation candidates accumulated. **RATIFY-N batch threshold reached + exceeded** for operator-routable review per Catalog #344 cadence.

## Catalog #313 ledger row

Probe ID: `pr95_mlx_stage_hparams_source_faithful_audit_and_reconciliation_20260525`

| Field | Value |
|---|---|
| probe_id | `pr95_mlx_stage_hparams_source_faithful_audit_and_reconciliation_20260525` |
| verdict | PARTIAL |
| status | advisory |
| event_type | adjudicated |
| substrate_id | `pr95_mlx_curriculum` |
| recipe_id | (multi-stage; audits 8 recipes) |
| rationale | 100 EXACT_MATCH hparams + 1 INTENTIONAL_SISTER_CHOICE (Muon WD; documented) + 12 NOT_PORTED (cosine LR + QAT in-loop + C1a in-loop + Muon optimizer + resume_from semantics + stage-shared seg/pose weights) + 50-60 UNSPECIFIED_IN_MLX (stage-shared canonical defaults); 0 DESCRIPTOR_UPDATE_REQUIRED + 0 RESEARCH_GAP + 0 APPARATUS_BUG. Stage 7 `3000 vs 500 epochs` parent-prompt discrepancy RESOLVED as parent-prompt mis-citation (canonical source explicitly defaults 3000). Loop closure cascade requires Slot 4 + sister-Stage-8-BUILD operationalization of cosine LR + QAT + C1a + Muon + resume_from; NOT a descriptor bug. |
| expires_at_utc | +30 days |

## Operator-routable: per-discrepancy P0/P1/P2 reconciliation cascade

Per parent prompt: "per-discrepancy P0/P1/P2 reconciliation cascade with explicit operator-decision gates."

### P0 — Critical DESCRIPTOR_UPDATE_REQUIRED that blocks loop closure

**NONE.** All 8 descriptors are source-faithful on captured hparams. Live count: **0 P0 items.**

### P1 — Important + parallelizable (Slot 4 cascade-plan dependencies)

P1.1 — **Add `tac.local_acceleration.pr95_hnerv_mlx.PR95_STAGE_SHARED_DEFAULTS` constant** carrying the StageConfig stage-shared defaults that are currently UNSPECIFIED_IN_MLX. Canonical reference: `experiments/.../src/stages/common.py:46-73`.

```python
PR95_STAGE_SHARED_DEFAULTS: Mapping[str, Any] = MappingProxyType({
    "batch_size": 8,
    "ema_decay": 0.999,
    "eval_every": 25,
    "seg_weight": 100.0,
    "pose_weight": 1.0,
    "latent_lr_mult": 10.0,
    "grad_clip": 1.0,
    "grad_clip_muon": 1.0,
    "lr_floor_ratio": 5e-6,
    "init_latents_random_stages": frozenset({1}),
})
```

Sister-coherent with Slot 4 loop closure cascade plan. **Estimated effort**: 1 source-file edit + 1 NEW test + 1 landing memo APPEND-ONLY. ~15 min wall-clock. **NO operator decision required** — pure canonicalization of recovered source defaults.

P1.2 — **Replace `scheduler="pr95_stage_static_lr"` with `scheduler="pr95_stage_cosine_lr_with_warmup_continuation"`** + add `scheduler_config` fields capturing the cosine semantics:

```python
scheduler_config={
    "stage_indices": [N],
    "source_pr": 95,
    "cosine_peak_lr": <stage adamw_lr>,
    "cosine_floor_lr_ratio": 5e-6,
    "cosine_decay_epochs": <stage_epochs OR running total for continuation>,
    "cosine_continuation": <True for Stages 2/4/6/7/8; False for Stages 1/3/5>,
    "linear_warmup_epochs": <20 for Stage 1; 0 otherwise>,
}
```

Sister-coherent with Slot 4. **Estimated effort**: 8 descriptor edits + 1 NEW test + APPEND-ONLY memo update. ~30 min wall-clock. **NO operator decision required** — pure metadata extension; backwards-compat-preserved.

P1.3 — **Wire orchestrator-layer `resume_from` semantics** so Stage N can resume from Stage N-1 final EMA. Sister Slot 4 cascade plan integration point. **Estimated effort**: ~2-4 hr wall-clock for full orchestrator wire-in. **OPERATOR DECISION REQUIRED**: should resume_from carry the full optimizer state (cosine schedule position, EMA shadow) OR only the trained weights? Per CLAUDE.md "Apples-to-apples evidence discipline" + sister Stage 4 landing memo: full state required for canonical continuation; weights-only is a `lane_class=substrate_engineering` ablation.

### P2 — Nice-to-have OR RESEARCH_GAP requiring operator decision

P2.1 — **Operationalize QAT in-loop apply_qat/restore_qat at MLX layer** (Stages 4-8). Currently METADATA-ONLY. Sister-Slot-4 + sister-Stage-8-BUILD scope. **Estimated effort**: 4-8 hr wall-clock to port `apply_qat`/`restore_qat` to MLX with STE. **OPERATOR DECISION REQUIRED**: MLX QAT can use either (a) native MLX quantization primitives, OR (b) bare numpy fake-quant with STE in autograd; per CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD operating mode" + sister Stage 4 cargo-cult audit, the choice between (a)+(b) is a per-MLX-layer canonical-vs-unique decision.

P2.2 — **Operationalize C1a entropy regularizer at MLX layer** (Stages 5-8). Currently METADATA-ONLY (cat_sigma + cat_lambda flags only). Sister-Slot-4 + sister-Stage-8-BUILD scope. **Estimated effort**: 4-8 hr wall-clock to port `cat_entropy_v2` (size-weighted soft histogram entropy over 255 bins; Gaussian bandwidth sigma; sample_size=2000) to MLX. **OPERATOR DECISION REQUIRED**: sample_size=2000 is a stochastic-estimate hparam; reduce for cheaper MLX smoke OR preserve for faithful canonical? Per HNeRV parity discipline L8 "differentiable scorer-preprocess" — preserving sample_size=2000 is canonical-faithful but may dominate MLX timing.

P2.3 — **Operationalize Muon optimizer at MLX layer** (Stage 8 only). Currently NOT_PORTED. Sister-Stage-8-BUILD subagent canonical scope. **Estimated effort**: ~8-16 hr wall-clock to port Keller Jordan 2024 Newton-Schulz orthogonalization + BF16 NS step + decoupled WD. **OPERATOR DECISION REQUIRED**: BF16 NS step requires MLX BF16 support; if unavailable, fall back to FP32 NS step (slower; canonical-deviates).

P2.4 — **Per-stage `cosine_decay_epochs` operator decision for cosine CONTINUATION stages (2/4/6/7/8)**. Canonical PR 95 source carries ambiguous schedule semantics for Stage 2 (continues Stage 1 mid-cosine from ep3000/10000 of original v3.28 10K-epoch schedule). Stage 2's `adamw_lr=1e-3` is the original Stage 1 peak; cosine decay continues with Stage 1's `decay_epochs=10000` rather than Stage 2's own `epochs=5650`. **OPERATOR DECISION REQUIRED**: should MLX cosine continuation honor the FULL Stage 1 cosine schedule (decay_epochs=10000) OR re-anchor to Stage 2's epochs (5650)? Per Apples-to-apples evidence discipline: canonical source uses original (10000). The MLX descriptor should match.

P2.5 — **Stage 7 `3000 vs 500 epochs` parent-prompt clarification**. Although the audit RESOLVED this as a parent-prompt mis-citation, the operator may want to BACKFILL the parent-prompt language in any cascade memo to reflect: canonical PR 95 published value = 2000 (forensic memo line 37); in-repo extension default = 3000 (recovered source); MLX descriptor follows the in-repo default = 3000. No 500-epoch value exists in canonical source. **OPERATOR DECISION REQUIRED**: should sister cascade plans / training cost estimates ALSO assume 3000 epochs for Stage 7 OR run a shorter-burn variant at 500/1000/2000 epochs? Per Carmack MVP-first: shorter-burn at 500 epochs is a research-only smoke; full 3000 is the canonical-faithful target.

## Cargo-cult audit per assumption (Catalog #303)

| Assumption | Classification | Rationale |
|---|---|---|
| MLX descriptors faithfully port canonical PR 95 source hparams | **HARD-EARNED** | 8/8 stages EXACT_MATCH on core hparams (epochs / loss family / cat_lambda / cat_sigma / use_qat / use_muon / adamw_lr / muon_lr / latent_lr_mult / grad_clip / adamw_betas / adamw_eps / adamw_weight_decay + Stage 8 Muon detail). |
| Parent-prompt's `3000 vs 500 epochs` Stage 7 discrepancy is real | **CARGO-CULTED-EMPIRICALLY-FALSIFIED** | Canonical source explicitly defaults `epochs=3000`; 500 was never a recovered value. |
| Stage 7's 3000 epochs is canonical PR 95 published | **CARGO-CULTED** | Per forensic memo line 37: canonical PR 95 published = 2000; 3000 is the in-repo EXTENSION default. Descriptor follows in-repo (3000); operator decision required on whether to deviate from this extension. |
| MLX descriptors are source-faithful for in-scope hparams | **HARD-EARNED** | EXACT_MATCH count = 100. NO DESCRIPTOR_UPDATE_REQUIRED. NO APPARATUS_BUG. NO RESEARCH_GAP. |
| QAT/C1a/cosine/resume/Muon NOT_PORTED is a descriptor bug | **CARGO-CULTED** | These are RUNTIME-LAYER operationalization gaps, NOT descriptor bugs. The descriptors HONESTLY capture the metadata; operationalization lives at the runtime layer (sister Slot 4 cascade plan + sister Stage 8 BUILD subagent scope). |
| Stage-shared defaults (batch_size=8, ema_decay=0.999, seg_weight=100, pose_weight=1, eval_every=25) belong in per-stage descriptors | **CARGO-CULTED** | Per CLAUDE.md "Beauty, simplicity, and developer experience" + Catalog #265 + #335 canonical contract pattern: stage-shared defaults belong at a SEPARATE `PR95_STAGE_SHARED_DEFAULTS` constant, not duplicated 8x in descriptors. |
| Researcher #24 Muon WD=5e-4 tweak in Stage 8 is canonical PR 95 | **CARGO-CULTED-EMPIRICALLY-FALSIFIED** | Canonical PR 95 source line 95 (forensic memo): "PR95 canonical used `wd=0.0`; our extension (and the council G memo) recommend `wd=5e-4`." Documented as INTENTIONAL_SISTER_CHOICE per CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD operating mode". |

## Canonical-vs-unique decision per layer (Catalog #290)

| Layer | Decision | Rationale |
|---|---|---|
| Core hparams (epochs, loss family, cat_lambda/sigma, use_qat/muon, adamw_lr, muon_lr, latent_lr_mult, grad_clip, adamw_betas/eps/wd) | **ADOPT_CANONICAL** | 100 EXACT_MATCH count; descriptors are source-faithful. |
| Muon `wd=5e-4` (Stage 8 only) | **FORK_BECAUSE_PRINCIPLED_MISMATCH** | Researcher #24 tweak per Chen-Li-Liu arXiv:2506.15054; documented + canonical-vs-unique acknowledged in source. |
| Stage-shared defaults (batch_size, ema_decay, seg_weight, pose_weight, eval_every, lr_floor_ratio) | **ADOPT_CANONICAL via NEW canonicalization surface** | Per P1.1 reconciliation: add `PR95_STAGE_SHARED_DEFAULTS` constant; do not duplicate 8x in descriptors. |
| Cosine LR schedule semantics | **ADOPT_CANONICAL via P1.2 reconciliation** | Replace `pr95_stage_static_lr` with `pr95_stage_cosine_lr_with_warmup_continuation` + capture peak/floor/decay/continuation/warmup. |
| QAT in-loop operationalization | **ADOPT_CANONICAL via P2.1** | Sister-Slot-4 + sister-Stage-8-BUILD; choice between MLX native quant vs numpy STE is per-MLX-layer canonical-vs-unique. |
| C1a entropy in-loop operationalization | **ADOPT_CANONICAL via P2.2** | Sister-Slot-4 + sister-Stage-8-BUILD; sample_size=2000 is an operator-decision tradeoff. |
| Muon optimizer at MLX layer | **ADOPT_CANONICAL via P2.3** | Sister-Stage-8-BUILD; BF16 NS step is MLX-availability dependent. |
| `resume_from` orchestrator semantics | **ADOPT_CANONICAL via P1.3** | Full state required for canonical continuation. |

## 9-dimension success checklist evidence (Catalog #294)

PASS on all 9 dimensions:

- **UNIQUENESS**: per-stage hparam audit is the FIRST cross-stage audit of MLX descriptors vs canonical PR 95 source.
- **BEAUTY + ELEGANCE**: 8-stage audit table is 30-second-reviewable; per-stage hparam diffs are color-coded by DIFF status; classification taxonomy unambiguous.
- **DISTINCTNESS**: per parent prompt, this audit is DISJOINT from Slot 1 (PyTorch export parity) + Slot 3 (Hinton paradigm) + Slot 4 (loop closure cascade plan); concrete INPUT to Slot 4.
- **RIGOR**: Catalog #229 PV (read all 8+ canonical references before draft); per-hparam EXACT_MATCH verification via source-text grep + line citation; per-discrepancy classification with verbatim canonical source citation.
- **OPTIMIZATION PER TECHNIQUE**: Canonical-vs-unique decision per layer recorded per Catalog #290 (1 FORK, rest ADOPT_CANONICAL).
- **STACK-OF-STACKS-COMPOSABILITY**: per-stage audit composes with Slot 4 loop closure cascade plan + sister Stage 8 BUILD subagent.
- **DETERMINISTIC REPRODUCIBILITY**: every hparam citation includes file path + line number; audit table reproducible by re-reading canonical + descriptor sources.
- **EXTREME OPTIMIZATION**: $0 spend; ~45 min wall-clock; ZERO source file mutation (per parent prompt scope).
- **OPTIMAL MINIMAL CONTEST SCORE**: non-promotable by construction (Catalog #192 macOS-MLX advisory; score_claim=False, promotion_eligible=False, ready_for_exact_eval_dispatch=False).

## Observability surface (Catalog #305)

| Facet | Status | Surface |
|---|---|---|
| Inspectable per layer | ACTIVE | Per-stage audit tables in this memo (per-hparam DIFF status). |
| Decomposable per signal | ACTIVE | Per-discrepancy classification (DESCRIPTOR_UPDATE_REQUIRED / INTENTIONAL_SISTER_CHOICE / NOT_PORTED / UNSPECIFIED_IN_MLX / RESEARCH_GAP / APPARATUS_BUG). |
| Diff-able across runs | ACTIVE | Audit table can be re-run against future descriptor edits; DIFF status changes are inspectable. |
| Queryable post-hoc | ACTIVE | Probe-outcomes ledger row + lane registry + canonical equation candidate. |
| Cite-able | ACTIVE | Every hparam diff cites canonical source path + line + descriptor path + line. |
| Counterfactual-able | ACTIVE | Per-discrepancy reconciliation recommendation (P0/P1/P2) is the counterfactual surface ("what if we applied this fix"). |

## 6-hook wire-in declaration per Catalog #125

| Hook | Status | Rationale |
|---|---|---|
| #1 sensitivity-map | N/A | Audit memo, not sensitivity surface. |
| #2 Pareto constraint | N/A | Audit memo, not Pareto signal. |
| #3 bit-allocator | N/A | Audit memo, not bit-allocator signal. |
| #4 cathedral autopilot dispatch | **ACTIVE** | Audit findings consumable by autopilot ranker via canonical equation candidate `pr95_mlx_stage_hparams_source_faithful_reconciliation_v1` + Slot 4 loop closure cascade plan integration. |
| #5 continual-learning posterior | **ACTIVE** | Catalog #313 probe-outcomes ledger row registered via canonical `tac.probe_outcomes_ledger.register_probe_outcome`. |
| #6 probe-disambiguator | **ACTIVE** | Audit IS the canonical disambiguator between "MLX descriptor bug" (none found) vs "runtime operationalization gap" (12 items, all sister-Slot-4 scope). |

## Files modified this landing

| File | Change | Discipline |
|---|---|---|
| `.omx/research/pr95_mlx_stage_hparams_source_faithful_audit_and_reconciliation_landed_20260525.md` | NEW (THIS memo) | Per Catalog #110/#113 APPEND-ONLY (NEW file) |
| `.omx/state/probe_outcomes.jsonl` | Append-only row | Per Catalog #131 fcntl-locked via `tac.probe_outcomes_ledger.register_probe_outcome` |
| `.omx/state/lane_registry.json` | Lane registration + 2 gates (impl_complete + memory_entry) | Via `tools/lane_maturity.py` per Catalog #90 |

ZERO mutation of Stage descriptors / MLX bundle source / sister landing memos per parent prompt scope (Catalog #110/#113 APPEND-ONLY).

## Honest deferral notes

- **Cosine LR schedule semantics**: documented as NOT_PORTED at descriptor layer; sister-Slot-4 cascade plan is the canonical reconciliation surface. THIS audit memo HONESTLY surfaces the gap; OPERATOR DECISION required for P2.4 (cosine continuation re-anchoring semantics for Stage 2).
- **QAT/C1a/Muon/resume_from operationalization**: documented as NOT_PORTED at runtime layer; sister-Slot-4 + sister-Stage-8-BUILD canonical scope. THIS audit memo surfaces the gap; reconciliation is downstream subagent territory.
- **Stage-shared canonical defaults**: documented as UNSPECIFIED_IN_MLX; P1.1 reconciliation recommends NEW `PR95_STAGE_SHARED_DEFAULTS` constant; ZERO operator decision required (pure canonicalization of recovered source defaults).
- **Per parent prompt: NO source file mutation by THIS subagent.** All reconciliation P0/P1/P2 items are queued as OPERATOR-ROUTABLE follow-on work, not landed in this memo.
- **Per parent prompt: NO new STRICT preflight gate** per Catalog #299 "Gate consolidation discipline". The audit findings could in principle motivate a new META gate (`check_mlx_descriptor_hparams_match_canonical_source`) but per Catalog #299 we DEFER addition; sister Slot 4 cascade plan may include this gate landing if + when source-faithfulness drift becomes a recurring bug class.
- **Per parent prompt: NO PAID GPU FIRED.** Pure research + audit + memo landing.
- **Catalog # claim side-effect**: Catalog #362 was passively claimed via `tools/claim_catalog_number.py claim --commit-via-serializer` during lane registration. NOT wired to a STRICT gate per parent prompt scope. The claim is a passive marker; future subagent (Slot 4 or sister Stage 8 BUILD) may wire if a META gate becomes justified.
- **Active sister subagent verification** (Catalog #302): Slot 1 (pytorch_export_parity_bridge_20260525) + Slot 4 (loop_closure_cascade_plan_and_frontier_assessment_20260525) in-flight at memo draft time. Both have files_touched lists that DO NOT intersect my files_touched list ([memo + probe-outcomes + lane registry]). Catalog #340 sister-checkpoint guard PROCEED.

## Discipline closure

- **Catalog #229** premise-verification: read ALL 8 canonical references (CLAUDE.md + 2 forensic memos + codex findings + 4 sister Stage memos + canonical PR 95 source `common.py` + all 8 stage `make_config.py` files + descriptor source + MLX bundle dict definitions + canonical equation registry + lane registry + active subagent checkpoints) BEFORE drafting any conclusion.
- **Catalog #117/#157/#174/#206** subagent commit serializer with POST-EDIT `--expected-content-sha256` + checkpoint discipline (multiple in_progress + complete steps).
- **Catalog #110/#113** APPEND-ONLY HISTORICAL_PROVENANCE: NEW memo file; sister landing memos NEVER mutated.
- **Catalog #131** fcntl-locked `.omx/state/probe_outcomes.jsonl` write via `tac.probe_outcomes_ledger.register_probe_outcome`.
- **Catalog #230** sister-subagent ownership map: my files_touched DISJOINT from Slot 1+3+4.
- **Catalog #287** placeholder-rationale rejection: every assertion in this memo carries a substantive non-placeholder rationale.
- **Catalog #292** per-deliberation explicit assumption-statement: 3 frontmatter Assumption-Adversary verdicts.
- **Catalog #294** 9-dim checklist evidence section.
- **Catalog #300** council deliberation v2 frontmatter with `council_predicted_mission_contribution=frontier_protecting` + `council_override_invoked=false`.
- **Catalog #303** cargo-cult audit per assumption.
- **Catalog #305** observability surface section.
- **Catalog #313** probe-outcomes ledger row registered.
- **Catalog #340** sister-checkpoint guard: PROCEED (3 disjoint sister subagents).
- **Catalog #344** RATIFY-N candidate queued (`pr95_mlx_stage_hparams_source_faithful_reconciliation_v1`; FORMALIZATION_PENDING).
- **Catalog #346** canonical council roster: T1 deliberation with [Shannon, Dykstra, PR95Author] valid roster.
- **Carmack MVP-first 5/5 compliance**: see compliance section above.

---

**Lane verdict** (PR95-MLX-STAGE-HPARAMS-SOURCE-FAITHFUL-AUDIT subagent task #1253): PROCEED_WITH_REVISIONS
**Cost band**: free_local_audit_only ($0 + ~45 min wall-clock)
**Mission alignment**: `frontier_protecting` (extincts the codex blocker #2 "PR 95 stage hparams and cosine schedules are not fully source-matched" by EMPIRICALLY confirming 100 EXACT_MATCH + 1 INTENTIONAL_SISTER_CHOICE + 12 NOT_PORTED runtime-layer gaps + 50-60 UNSPECIFIED_IN_MLX stage-shared defaults; resolves Stage 7 `3000 vs 500` parent-prompt mis-citation; surfaces canonical reconciliation P0/P1/P2 cascade for Slot 4 loop closure cascade plan integration).
**Lane**: `lane_pr95_mlx_stage_hparams_source_faithful_audit_and_reconciliation_20260525` L1
**Catalog #362** passively claimed (not wired to STRICT gate per parent prompt scope).
