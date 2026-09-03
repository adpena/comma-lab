# ddm_ft1_shipped_renderer_aligned_finetune — the first fold-back on the FRONTIER object: fine-tune the shipped 30,856 B semantic renderer FROM ITS OWN WEIGHTS with the realized expected-flip margin loss (R → uint8 → SegNet argmax vs the DALI GT) + pose at step zero, QAT-int4 so the exported section is the same size, tokens untouched — a pure same-bytes distortion move, delivered as a candidate archive + a typed MAIN fire order (n600 advisory verdict → terminal pose re-solve → T4)

## MANDATE

Operator 2026-09-03: *"all of the stuff we have discovered and chain after training points to improvements to
the training and other steps themselves"* + standing GO. `ddm_fb1_foldback_program_20260903.md` maps every
post-hoc law to a training lever; this is fold #1 on the archive that holds the frontier. Facts: d_seg
2.0139e-4 (100·d_seg = 0.020139 S) is 95% MANUFACTURED by the render path (td1/rt1; mst1); the shipped renderer
(semantic_renderer section 30,856 B, sha 39d1be52…, at offset 13,529 of member `p`) was trained by the PR #130
authors with their curriculum and has NEVER been fine-tuned at its own size against the realized argmax
(rj1: realized deltas UNMEASURED; wd2: a smaller student; w96a/b: a from-birth lineage at d_seg 8e-4). Each
1e-5 of d_seg recovered = −0.001 S at ZERO archive bytes; halving d_seg would be ≈ −0.010 S (S ≈ 0.138) — the
largest same-object move available. Pose is the risk: a changed render shifts PoseNet; the pose term must be in
the loop from step zero and the carrier terminally re-solved (jg5/up2 law); pose must stay ≤ 1.25e-4 absolute
(memory m110; ∂S/∂d_pose = 626.5) and ideally at 6.37e-6.

## THE RECIPE (verify every element at source; never invent a flag)

1. **Identity gate (0 steps) FIRST.** Unpack the shipped semantic_renderer section from the AFR1 archive
   (`/Volumes/VertigoDataTier/pact/ddm_g8s_single_run_reproof/store_v2/retained/archive.zip` or the g8r
   compliance stage; sha cbb8d928…) with the shipping receiver's parser (`submissions/semantic_joint_ctxmix/cpr1/
   ddm_mp2_semantic_receiver.py` / `inflate.py` `unpack_semantic`, READ-ONLY tree — import, never edit) into the
   lifted trainer's checkpoint format; pack it back through the deployed packer
   (`src/tac/pr130_lift/lifted/pack_semantic_pose`) → the section must be BYTE-IDENTICAL (30,856 B, sha
   39d1be52…); render 8 pairs through the trainer's deployed inference path → raw identical to the shipped
   receiver's output. If this gate fails, STOP with the typed reason (the fine-tune would be on a different object).
2. **Fine-tune** with `src/tac/pr130_lift/train_semantic_quantized_resumable.py` (42d322db5: `--init`, `--bits 4`
   hard-pinned to the deployed int4 packer, expected-flip loss at line ~337, exact-R evaluation, deployed
   pack/parse argmax parity gate, EMA with `--ema-target-seed-fraction`): init = the unpacked shipped weights;
   loss = realized expected-flip margin `100·mean(sigmoid(−(z_target − max z_other)/τ))` on the R → uint8 →
   SegNet logits vs the DALI GT argmax table (`experiments/results/mlx_fleet_gt_cache/gt_n600.npz`, the
   T4-scored lineage — #1142: PyAV-vs-DALI seg fork 1.43×; train against DALI, report both) with τ linear
   0.15 → 0.05 over the window (w96b law, 3d9e021d0) + the pose term from step zero (PoseNet on the carrier-rendered
   pair vs the stored targets, weight as w96b/qbr1 derive it) + QAT int4 fake-quant on; EMA shadow saved per
   epoch (the EMA law resolved via `ema_decay_run_geometry_v1` — executable == sealed; the wc3 gate is STRICT);
   LR derived from the receipts (the 2026-08-09 BS16 smoke used 2e-7 for 30 steps; w96b's schedule and cosine
   1% floor are the reference) — declare the derivation; seeded; per-epoch checkpoints; `--resume-from`.
   Device: **CPU only** (the QBR1 burn owns Metal/MPS; the trainer's thread count capped so the burn's 4 threads
   are not starved — measure s/pair first). Chunked verdict every epoch on n600 (advisory, PyAV lineage,
   labeled; `--eval-batch-size` ≤ 32). Stop rule: jg5's derived materiality (a step is kept only if the
   advisory ΔS falls by more than the exchange floor); cap epochs by wall-clock, not by hope.
3. **Export + swap:** EMA weights → deployed packer → new section (must be exactly 30,856 B by the int4 law —
   verify; if the size changes, the law changed: STOP and report) → splice into a COPY of the AFR1 archive
   (all other bytes identical; recompute the container per the shipped grammar) → receiver parse-back +
   render identity vs the trainer's deployed inference (8 pairs full, 600 pairs via the receiver hash).
4. **Deliver a typed MAIN fire order** (do not run it): (a) n600 advisory verdict of the candidate through the
   real receiver (BR2/qxr1 payload-retaining protocol; scorer claim placeholder); (b) terminal pose re-solve
   on the candidate's renders (`experiments/ddm_up2_shipping_pose_solve.py` 2079b4bb9 / jg5 2079b4bb9: `solve
   --gt-cache --axis`) → re-spliced carrier; (c) the T4 buy (`tools/fire_modal_auth_eval.py --seal`) with the
   pre-registered decision: promote iff exact S < 0.14797617125559104 AND d_pose ≤ 1.25e-4.


## RECALL — DYNAMICS, INTERACTIONS, SOURCE (binding pre-read; operator 2026-09-03: "we have extensive research … understanding dynamics and interactions and source")

Do NOT design the loss, LR, τ, or the stop rule from the headlines above. Read these first and cite what you
consume; each carries a mechanism the fine-tune must respect:
- **What the render path manufactures and where:** `ddm_mst1_manufactured_stage_split_20260822.md` (78.71% of
  seg error at the native render; R + uint8 are net REPAIRERS), `ddm_rt2_manufactured_seg_mechanism_20260817.md`,
  `ddm_ms9_dx2_seg_manufactured_fraction_20260822.md`, `ddm_msr1_manufactured_seg_reduction_20260823.md`,
  `ddm_mf1_manufactured_seg_repair_20260823.md` — the sites the renderer gets wrong are NOT random; the fine-tune
  should target the manufactured class, and the memos say what those sites look like.
- **Seg ↔ pose coupling geometry:** `ddm_tv1_evaluator_tolerance_curve_20260824.md`, `ddm_tv2_…` (seg slack and
  pose damage are CO-LOCATED: boundaries no-slack vs interiors 47% slack — a seg-driven render change hits
  pose where the interiors move); `ddm_jg1_joint_solve_20260819.md` (104.6–822.7× pose damage before a joint
  solve); `ddm_pk4_optimal_form_frame0_pose_verdict_20260813.md` (frame-0 is the pose place);
  `collateral_coupling_geometry_and_film_flicker_sidecar_20260718.md` — the pose term's weight and step-zero
  placement follow from these, not from a default.
- **Collateral, not targeting, is what kills seg moves:** `ddm_qs3_saturation_compose_20260813.md` (loss is
  COLLATERAL; 97.4% of edits realize), `ddm_a1_bounded_collateral_realized_n64_20260723T031500Z/`, memory m132
  (B/H decomposition mandatory, collateral priced in) — report per-epoch B/H/W: sites fixed / newly broken /
  unchanged, never a headline d_seg alone.
- **The judge's source geometry:** CLAUDE.md "Exact scorer architectures" (SegNet reads frame_1 only; stride-2
  stem; ~85 px ERF; the canonical class order; PoseNet and SegNet share the SAME resize `D` — `ddm_pz1` in
  CLAUDE.md), `ddm_cfa1_closed_form_atlas_20260831.md` (the frozen piecewise-analytic chain; every non-analytic
  locus), `ddm_ce1_allocation_ladder_verdict_20260817.md` (the exact expected-flip margin law and its ladder).
- **Realization and the exchange rates:** `ddm_bz2d_distortion_verdict_20260830.md` (token error → argmax 1.157×;
  pose 152× worse on the PyAV fork), `ddm_dd1_displacement_dimensionality_20260803.md` (Lane ±1 px width =
  ±40% area), `ddm_r2s_stratified_and_sparse_residual_20260728.md` (warp-predict closed; flips are codim-1
  sub-pixel boundary shifts), `ddm_sg2b_falsifier_verdict_20260901.md` (X-alone falsifier 3/3), memory m65/m108
  (Euclid-vs-Fisher cosine sign-flips; the margin field IS the Fisher surrogate — use the margin table in
  `gt_n600.npz['margins']`), `ddm_dc1_correction_label_cost_and_qa03_censoring_20260801.md` (error sites are
  class-skewed; Lane error-rate dominates).
- **Renderer-specific prior art:** `ddm_rf1_renderer_film_rung_20260824.md` (FiLM rung; BAR-SEG Δd_seg <
  7.18e-6 for the seg leg alone — the admissibility bar you must beat per leg), `ddm_rj1_renderer_joint_move_20260823.md`,
  `ddm_wd2_ep60_advisory_refusal_verdict_20260815.md`, `ddm_w96b_aligned_loss_implementation_20260826.md` (the
  τ schedule and cosine floor with their derivations), `ddm_xs1_cross_section_conditioning_20260822.md`.
Run `tools/graph_memory_recall.py "<your query>"` before each design decision and record what changed.

## HARD CONSTRAINTS

- `upstream/` and `submissions/semantic_joint_ctxmix/` READ-ONLY (import the receiver; never edit the PR tree).
  NO scorer-lane verdict runs beyond the trainer's own chunked advisory eval (label it), NO Modal, NO Metal/MPS.
  Never write under `/Volumes/APDataStore/pact/ddm_wc3_qbr1_ema_law_cure/`.
- Any step > 30 min ONLY via `.venv/bin/python tools/launch_detached_process.py --output-dir <run_dir>
  --done-receipt <name> --nice 10 --nice-best-effort -- <cmd...>`; the arm monitors; per-epoch checkpoints.
- ALWAYS KEEP THE PAYLOAD under `/Volumes/VertigoDataTier/pact/ddm_ft1_shipped_renderer_aligned_finetune/`:
  every checkpoint (EMA), every exported section, every candidate archive, every verdict chunk — sha256 + bytes.
- Serializer commits w/ post-edit `--expected-content-sha256`; `.py` = 2 review passes; ruff clean; never bare git.
- NO-FAKE: an identity gate that fails, a size that changes, or a pose that blows past budget is reported as
  such; the advisory d_seg is never called a score.

## PRIOR NEGATIVE SIGNAL (bearing dead-ends this charter consumes)

- `ddm_rj1_renderer_joint_move_20260823.md` — three renderer representation rungs, realized deltas UNMEASURED,
  compensation NOT_SOLVED: the missing piece was training + re-solve, which this charter does.
- `ddm_wd2_ep60_advisory_refusal_verdict_20260815.md` — a SMALLER student's pose cost was tens of × its byte
  credit: keep the size; this is a same-size fine-tune.
- `ddm_w96a_aligned_config_renderer_window_20260826.md` / `ddm_w96b_…` — the from-birth aligned lineage (OFF
  d_seg 8.1e-4): use its LOSS LAW and τ schedule, not its lineage.
- `ddm_fcd2_distortion_legs_execute_20260829.md` — large render changes broke the pose gate 26,710×: pose in the
  loop from step zero + terminal re-solve are mandatory, and the fine-tune must stay small (LR, epochs).
- `n205_oom…` (memory) — chunk the verdict; never 600 pairs at once.
- `ddm_cpu1_gt_lineage_attribution.py` / #1142 — PyAV vs DALI GT: train on DALI (T4's lineage), report both.

## OPTIMAL FORM

- Family exemplar (reference): the lifted PR130 trainer `src/tac/pr130_lift/train_semantic_quantized_resumable.py` (receipt `/Volumes/VertigoDataTier/pact/ddm_pr130_train_20260809/reports/BS16.json`, verdict PASS; commit
  42d322db5) with its deployed pack/parse parity gate; the aligned loss law `ddm_w96b_aligned_loss_implementation_20260826.md`
  (commit 3d9e021d0); the terminal pose re-solve `experiments/ddm_up2_shipping_pose_solve.py` (commit 2079b4bb9).
- SCOPE reductions: epochs capped by wall-clock; the identity gate's 8-pair render check (legal). MECHANISM
  reductions FORBIDDEN: no float export; no proxy loss (the realized exact-R path only); no pose-free loss;
  no prefix subsets for the verdict (chunked all-600 or seeded random).
- **PRIOR-LAW PREDICTION (falsifiable):** the manufactured-error law (95% of seg error from the render path)
  predicts the aligned fine-tune lowers advisory d_seg by ≥ 25% within a few epochs at held pose (≤ 6.4e-6
  after re-solve). FALSIFIER: advisory d_seg does not fall ≥ 10% at any kept epoch, or pose cannot be re-solved
  under 1.25e-4 — count it plainly; that closes "renderer fine-tune" at formulation scope for this size.

## DELIVERABLE

`.omx/research/ddm_ft1_shipped_renderer_aligned_finetune_20260903.md` — identity-gate receipt, s/pair and the
LR derivation, per-epoch advisory table (d_seg PyAV+DALI, d_pose, section bytes), the candidate archive sha, the
typed MAIN fire order, RECALL EVIDENCE, NEXT_IF_RESUMED, LIVE-HYPOTHESES, DEAD-ENDS. Commit via the serializer.
Cite `docs/operating_manual_craft_handoff.md`. End with the own-vehicle frontier line.
