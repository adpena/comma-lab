# ORPHAN-HARVEST RECOVERY LEDGER — 2026-06-10

**Subagent:** `orphan_harvest_ready_made_sweep_20260610` (READ-ONLY sweep; this memo is the only artifact).
**Evidence grade:** `[macOS-CPU advisory]` / mechanism-only. NO score claims, NO dispatch, `promotable=false`.
**Operator alert (2026-06-10):** PR #112 (mattneel) cashed a ~$0 lossless byte win WE had already identified
but left BLOCKED (decoder-weight `entropy_recode`, candidate −16 KB, never materialized). Thesis: *"there are
likely OTHER blocked/planning-only/research-only things sitting orphaned but ready-made and almost fully baked."*
This is the comprehensive value-recovery sweep.

**Frontier at audit (pointer, never hardcoded — `tools/refresh_canonical_frontier.py`):**
contest-CPU **0.19198275** (178,495 B, `lane_pr110pp_r3_candidate_cpu`, sha `1ccae18d…`) /
contest-CUDA **0.20533003** (186,876 B, `lane_pr106_format0d_latent_score_table`, sha `9cb989ce…`).
Score law: `S = 100·d_seg + √(10·d_pose) + 25·B/N`, N=37,545,489. Byte price ≈ **6.66e-7 score/byte**.

---

## 0. HEADLINE — THE CONVERGENT FINDING

The 2026-06-09/10 exhaustion-map wave proved every **frozen-bytes distortion** lever is DEFER (Pareto
vertex). But PR #112 just demonstrated that the **lossless-entropy-recode axis is ORTHOGONAL and NOT
exhausted** — it changes ZERO pixels (d_seg/d_pose identical) and wins on rate alone. The exhaustion map's
"decoder coarsening FALSIFIED" verdicts are about *lossy* coarsening (which moves d_seg ~10× the rate gain);
they do NOT touch lossless recode. This is the single recoverable class the operator's thesis predicted.

**Count of READY-MADE-and-blocker-resolved items: 3** (R1, R2, R3 below). All are PR-#112-class lossless byte
wins materializable at ~$0 build + one paired replay. Everything else is either DEFER-killed-for-a-reason
(excluded with citation, §3) or NEEDS-REAL-WORK (training campaigns, §2).

---

## 1. THE RECOVERY LEDGER (ranked by value × readiness / blocker-cost)

Columns: **what** · **where** · **predicted value** · **blocker** · **BLOCKER NOW RESOLVED?** (the key column) ·
**readiness** · **reuse targets (named existing code)** · **dead-end?**

### R1 — Decoder-weight per-tensor adaptive entropy recode (THE PR-#112 ORPHAN ITSELF) ⭐ TOP
- **What:** Replace the frontier decoder's 7 Brotli streams with a tight **per-tensor adaptive order-0 range
  coder** (constriction ANS, geometric-primed priors (ρ,M,inc,ε) grid-searched by exact simulated code length,
  2 B/model header, IEEE-exact float64 tables for cross-platform losslessness). Pixels unchanged.
- **Where (orphan signal):** `byte_shaving_campaign_master_gradient_byte_range_suggested_contract_portfolio_20260523T154232Z.json`
  flagged spans `mg_byte_span_0162171_0166267 … _0178417`, `materializer_target_kind = archive_charged_byte_range_entropy_recode`,
  `candidate_saved_bytes` up to **16,246** (optimistic UB). PR #112 intake §5: `public_pr112_frontier_beat_intake_20260610.md`.
- **Predicted value:** PR #112 measured **−1,060 B on decoder** (real, byte-identical decode). On our R3
  frontier (orthogonal selector axis) → ~177,114 B → **S ≈ 0.191117**, beats PR #112 by ~+8.6e-6 and our
  frontier by **−0.00092**. (Campaign's 16 KB UB is the order-0 ceiling; PR #112's 950–1,060 B is the realized floor.)
- **Blocker (original):** `master_gradient_byte_ranges_are_planning_coordinates_only` +
  `requires_archive_grammar_mapping_before_materialization` + `requires_adapter_implementation_before_queue_dispatch`.
- **BLOCKER NOW RESOLVED? — PARTIAL→YES.** (a) Archive-grammar mapping EXISTS: `tac.pr101_split_brotli_codec.decode_decoder_compact`
  (brotli→7 raw streams) + PR101 `FIXED_STATE_SCHEMA`. (b) A materializer scaffold EXISTS:
  `src/tac/optimization/byte_range_entropy_recode_materializer.py` (delegates to PR103 static-histogram AC).
  (c) `constriction` is installed; `tac.pr103_arithmetic_codec`, `tac.shared_pmf_model` (== PR #112's
  `SHARED_MODEL_TENSORS` idea), `tac.arithmetic_qint_codec` all in-tree. **The ONLY missing primitive** is the
  adaptive **geometric-primed per-tensor model with grid-searched (ρ,M,inc,ε)** — PR #112's single innovation
  beyond our static-AC port (intake §5 line 145). ~30–60 LOC on top of `shared_pmf_model`.
- **Readiness: SMALL-BUILD** (one new ~50-LOC adaptive model + wire into existing materializer + re-pack via
  `pr101_split_brotli_codec` grammar + byte-close + 1 paired Modal CPU+CUDA replay ~$0.3 for the formal stamp).
- **Reuse targets:** `byte_range_entropy_recode_materializer.py` (extend), `pr103_arithmetic_codec.pack_ac_stream/unpack_ac_stream`,
  `shared_pmf_model.SharedPMFModel`, `pr101_split_brotli_codec.{decode_decoder_compact,decode_latents_compact}`,
  `constriction.RangeEncoder`. **R3 selector axis is ours and orthogonal — composes for the head-to-head win.**
- **Dead-end? NO.** Distinct from the FALSIFIED lossy-coarsening (§3-X1): that moves pixels; this is bit-identical
  decode (PR #112 proved sha `d1afc583`). The decoder-axis-waterfill verdict explicitly scoped its kill to
  *coarsening*, not lossless recode.

### R2 — Latent AR(1)+cross-dim+discrete-Gaussian range recode (PR-#112 latent sibling) ⭐
- **What:** Replace raw-LZMA1 latent coding with **per-dim causal AR(1) on own deltas + optional lag-2 + up to
  4 already-decoded cross-dims (integer-quantized LS coefficients) + discrete-Gaussian residual range coding**
  (precomputed `Q_TABLE`, no `exp()` at decode). Pixels unchanged.
- **Where:** PR #112 intake §3 (technique 2); our `frontier_latent_axis_waterfill_verdict_20260610.md`.
- **Predicted value:** PR #112 measured **−317 B on latents** (real). On our frontier, transfers cleanly (same
  PR101 INT8 latent codes). Compounds with R1: R1+R2 ≈ −1,377 B ≈ **−0.00092** (matches PR #112's total win,
  + our R3 selector −22 B edge).
- **Blocker (original):** same planning-coordinate + adapter blockers as R1.
- **BLOCKER NOW RESOLVED? — YES, with a NUANCE the kill-record does NOT cover.** Our latent verdict tested
  *LZMA retune / coder swap / **second-order** delta re-prediction* → all FALSIFIED (2nd-order entropy 7.52 >
  1st-order 7.03 bits, +977 B). **It did NOT test PR #112's exact combination** (1st-order AR + cross-dim LS +
  discrete-Gaussian *range coder replacing LZMA entirely*). PR #112 EMPIRICALLY achieved −317 B with that combo
  on the same PR101 latents → the lever is OPEN, not killed. Reuse `decode_latents_compact` (LZMA→raw codes)
  as the inverse + a new AR+range-coder (≈40 LOC; the discrete-Gaussian Q_TABLE is the only new piece).
- **Readiness: SMALL-BUILD** (≈40 LOC + wire into the R1 materializer + same re-pack/byte-close/replay).
- **Reuse targets:** `pr101_split_brotli_codec.decode_latents_compact`, `pr103_arithmetic_codec`, `constriction`,
  `arithmetic_qint_codec`, the latent-axis verdict's per-dim sensitivity ranking.
- **Dead-end? NO.** Distinct from the FALSIFIED 2nd-order re-prediction (different predictor + different coder).

### R3 — The canonical `pr110_payload_entropy_recode` materializer (closes the orphan permanently) ⭐
- **What:** Promote the BLOCKED byte-shaving span into a real, reusable materializer (R1+R2 packaged): extract
  via `decode_decoder_compact`/`decode_latents_compact` → recode (R1 adaptive decoder model + R2 latent AR/range)
  → re-pack → byte-close → paired-eval. This is MOVE 2 of the PR #112 intake.
- **Where:** PR #112 intake §5 MOVE 2; the BLOCKED byte-shaving campaign portfolios (§above).
- **Predicted value:** ≥ R1+R2's −1,377 B on ANY PR101-grammar archive; **reusable across R3/PSV3/future lanes**
  (durable, not one-shot). Also tests whether OUR distinct R3/PSV3 decoder weights have MORE recode headroom
  than PR101's (intake §5 line 171).
- **Blocker (original):** `materializer_backlog_is_planning_only` / `requires_adapter_implementation_before_queue_dispatch`.
- **BLOCKER NOW RESOLVED? — YES once R1+R2 land** (the adapters ARE R1's adaptive model + R2's AR coder; the
  scaffold `byte_range_entropy_recode_materializer.py` is the contract shell already in-tree).
- **Readiness: SMALL-BUILD** (R1+R2 + the packaging/test/wire-in; no new GPU; this is the "make the win durable" layer).
- **Reuse targets:** all of R1+R2 + `byte_range_entropy_recode_materializer.py` + `byte_range_entropy_recode_chain.py`
  + `tools/run_byte_range_entropy_recode_chain.py` + `tools/build_byte_range_entropy_recode_receiver_proof.py`.
- **Dead-end? NO.** It is the canonical home for the recovered class.

### R4 — Inflate-program-bytes-are-RATE-FREE (E1: migrate small payload sections into code) — SMALL-BUILD, COMPLIANCE-BOUND
- **What:** `upstream/evaluate.py:63` counts `archive.zip` ONLY; inflate.py/inflate.sh bytes are NOT in the rate
  term. Migrate compliance-defensible payload-class sections (constants, tables, procedural generators) into CODE
  at ZERO rate cost (precedent: PR110's mode catalog + Huffman codebook live in code, maintainer-accepted).
- **Where:** `evaluate_py_fresh_eurekas_20260610.md` E1 (committed `1264a4405`, 2026-06-10). NOT yet actioned
  (git log confirms no follow-up commit migrating sections to code).
- **Predicted value:** UNQUANTIFIED but potentially large — any section movable to code is pure rate reduction.
  The selector mode-catalog/codebook precedent is already in code; the open question is which OTHER small sections
  (sidecar tables, framing constants) are defensibly procedural. Bounded by review norms, not the formula.
- **Blocker:** "compliance-bounded audit of largest defensible inflate.py" is a NAMED follow-up, never executed.
- **BLOCKER NOW RESOLVED? — PARTIAL.** The finding is fresh + correct (source-cited). The blocker is a
  *judgment call* (compliance defensibility), not a missing primitive. Needs the audit, not new code.
- **Readiness: SMALL-BUILD + JUDGMENT** (audit which sections are compliance-defensible-as-code; sidecar/framing
  are the candidates — but sidecar is "< 7 B from optimal" and framing is 7 B, so the EV is small per the
  exhaustion map's "sidecar/selector = 0.5% of bytes"). Honest: low-byte-EV but ZERO-cost and a standing subsidy
  for any future witness-program (V6) carrier.
- **Reuse targets:** PR110 inflate.py mode-catalog pattern; AFSR-1 export placement logic.
- **Dead-end? NO** but **LOW-EV on the current frontier** (the big sections — decoder weights — are genuinely
  high-entropy payload, NOT movable to code). Flag as a V6 witness-program subsidy, not an immediate frontier move.

### R5 — `pr106_latent_sidecar` proposal (L27-style per-pair correction) — SUPERSEDED, NOT READY
- **What:** Per-pair `(dim_idx, delta_q)` sidecar, predicted −0.00218 (PR100-vs-PR105 empirical).
- **Where:** `lane_pr106_latent_sidecar` (L1 PROPOSAL, council-gated).
- **BLOCKER NOW RESOLVED? — NO / MOOT.** The current 0.19199 frontier ALREADY CARRIES the 607-byte L27
  correction sidecar (`frontier_latent_axis_waterfill_verdict_20260610.md`: dropping it = +0.0029 UNFAVORABLE;
  "sidecar pays rent ~8×"). The proposal targets the OLDER PR106 archive; on the current frontier the sidecar is
  exhausted (adding more or dropping is net-negative). **Superseded by frontier evolution, not a recovery item.**
- **Dead-end? Effectively yes on the current frontier** (the lever already fired into the frontier).

---

## 2. NEEDS-REAL-WORK (not ready-made; the structural doors, per MASTER_ROADMAP post-exhaustion-map)

These are the roadmap's RANK 1/2/5/6 — genuine but NOT "almost fully baked" (they are training campaigns / new
vehicles, weeks of work):
- **Aimed score-aware retraining campaign** (`lane_aimed_score_aware_retrain_20260610`, RANK 1): smaller decoder
  re-memorized at fewer bytes, aimed by flip-map/atlas/cone. NEEDS-REAL-WORK (multi-hour MLX + paired eval,
  descent-proof smoke gate first). The aiming surfaces ARE landed (flip map, per-tensor/per-pair maps, spectral
  atlas w_equiv≈294, B2 Y-fraction, invisibility basis, resize-null preimage) — but the campaign itself is unbuilt.
- **Faithful HF-mechanism vehicle** (`pact_nerv_vq` + F1 bilinear-skip, RANK 2): ~15-LOC kernel wire-in is small,
  but reaching a byte-closed descending exact score is real work.
- **Evaluator-inverse direct grammar** (RANK 5): weeks; primitives (read-surface atoms, invisibility basis) feed
  Phase 4, not the live loop.
- **Pose-output-entropy probe** (RANK 6): $0 measurement, aims RANK 1 — a probe, not a candidate.

The landed aiming surfaces (R-inventory of `evaluator_inverse_orphan_inventory_20260609.md`) are CONSUMED-or-
EXTEND, not orphaned: `lf_payload_rate_distortion`, `joint_p18_p19_waterfill`, `action_effect`,
`evaluator_action_waterfill`, `null_space_exploiter`, `evaluator_invisibility_basis`, `resize_null_preimage`,
`frame1_joint_safe_cone`, Class-2/3 atom generators. Do NOT rebuild (no-duplicative-code).

---

## 3. EXCLUDED — looks-ready-but-KILLED-for-a-reason (the guard; cite the verdict)

- **X1 — `lane_lossy_coarsening_analytical`** (registry note: "BREAKTHROUGH: 156,344 B archive (−21,800 B) at
  3.86% rel_err; needs CUDA dispatch to confirm"). **The registry note is STALE.** The CUDA dispatch ALREADY
  HAPPENED and KILLED it: `lossy_coarsening_T0312_retired_config_do_not_redispatch_20260508_claude.md` records
  **0.351718793 [contest-CUDA A-negative]** (sha `ab8a8a13…`, 156,404 B) — far WORSE than frontier. The 3.86%
  rel_err moves d_seg/d_pose; coarsening is FALSIFIED-AT-IMPL ×2 (naive uniform + grid-LSQ retrain,
  `frontier_decoder_axis_waterfill_verdict_20260610.md` + `frontier_decoder_qat_recovery_verdict_20260610.md`).
  DO NOT redispatch. *(Recommend: update the stale registry note for `lane_lossy_coarsening_analytical` to cite
  the A-negative result so it stops looking ready.)*
- **X2 — Decoder weight COARSENING / quantization (int4/int6 PTQ, QAT, per-channel, GPTQ, AWQ).**
  `lossy_falsification_scope_audit_20260508_codex.md`: all measured configs non-dispatchable (28–46% rel_err);
  re-confirmed 2026-06-10 by the decoder-axis-waterfill (c1/c2/c3 +0.0709/+0.0902/+0.1648 contest-CPU). DEFER,
  not killed — but NOT ready-made (any new config needs new research + paired CUDA). Distinct from R1 (lossless).
- **X3 — Latent SECOND-ORDER re-prediction.** FALSIFIED (`frontier_latent_axis_waterfill_verdict_20260610.md`:
  2nd-order 7.52 bits > 1st-order 7.03, +977 B). NOTE: this is NOT R2 (R2 = 1st-order AR + range coder, OPEN).
- **X4 — Selector recode.** At entropy floor; recode = sha-identical no-op (exhaustion map). PR #112 also left it
  (their selector is plain FEC6 248 B; our R3 on-host mode-table already shaves 22 B below — the orthogonal edge).
- **X5 — Frame-1 seg-repair correction sidecar.** Information-theoretically incapable: 1.525 B/flip position-only
  floor > 1.27 B/flip break-even (`frontier_seg_repair_pool_verdict_20260610.md`). 66,039 flips fully mapped; a
  sidecar cannot clear THE LAW. The fix is a better reconstruction (RANK 1 training), not a sidecar.
- **X6 — SNeRV stored-LF representations.** Every structured rung 280–530× the 178 KB frontier
  (`snerv_branch_b_round2_verdict_20260610.md`). DEFER; the binding rung is composing the frontier DIRECTLY.
- **X7 — R3 CPU-frontier → CUDA promotion.** NO TRANSFER (`cuda_axis_frontier_eval_verdict_20260610.md`:
  0.22616 CUDA vs control 0.20533; pose +0.0232 CUDA drift). Kill criterion HIT. CUDA axis needs its own attack.

---

## 4. THE PR-#112-CLASS FREE WINS (lossless byte wins materializable at ~$0)

**R1 + R2 + R3 are the entire PR-#112-class.** They are lossless (byte-identical decode, d_seg/d_pose UNCHANGED),
the reuse code is in-tree (constriction, pr103_arithmetic_codec, shared_pmf_model, pr101_split_brotli_codec,
the byte_range_entropy_recode_materializer scaffold), and the only missing primitive is PR #112's adaptive
geometric-primed per-tensor model (~50 LOC) + the latent AR+discrete-Gaussian range coder (~40 LOC). Cost: ~$0
build + ONE paired Modal CPU+CUDA replay (~$0.3) for the formal stamp. Predicted: ~177,114 B → S ≈ 0.191117,
**beats PR #112 + beats our frontier by −0.00092**, ZERO fidelity risk. This is the immediate harvest queue and
it is exactly the orphan the operator flagged — the planner signal existed (campaign portfolio, candidate −16 KB)
and we never built the ~90-LOC materializer that maps spans → PR101 schema → recode → re-pack.

---

## 5. WIRE-IN (Catalog #125) + provenance
- **Hook #1 sensitivity-map:** the byte-shaving campaign spans + per-tensor sensitivity map ARE the recode targets.
- **Hook #2 Pareto:** R1/R2/R3 move the RATE axis only (orthogonal to the saturated distortion vertex).
- **Hook #3 bit-allocator:** the adaptive per-tensor (ρ,M,inc,ε) model IS a bit-allocator primitive (PR95 L21–L32 family).
- **Hook #4 cathedral-autopilot:** the `byte_range_entropy_recode_chain` + materializer are the dispatch surface.
- **Hook #5 continual-learning:** PR #112's measured −1,381 B reseeds the V3 judge that lossless-recode is OPEN
  (NOT exhausted) — the one axis the frozen-bytes verdicts did not test.
- **Hook #6 probe-disambiguator:** R2's nuance (1st-order AR+range-coder UNTESTED vs 2nd-order FALSIFIED) is the
  disambiguator — our latent verdict's "LZMA optimal" conclusion is scoped to coders we tested, not PR #112's.
- **Provenance:** every claim cites `{memo, line/JSON, observed value}`. No score claim; `[macOS-CPU advisory]`.
  Frontier read from pointer. Reuse code paths verified present in-tree (find/grep, §sweep). Constriction import OK.
- **Cross-refs:** `public_pr112_frontier_beat_intake_20260610.md` (the PR-#112 analysis + MOVE 1/2/3) ·
  `MASTER_ROADMAP_post_exhaustion_map_20260610.md` (the DEFER verdicts = the kill record respected) ·
  `evaluator_inverse_orphan_inventory_20260609.md` (the CONSUMED-vs-orphan code map) ·
  `frontier_{decoder,latent,seg_repair}_*_verdict_20260610.md` + `lossy_coarsening_T0312_retired_*` (the §3 exclusions) ·
  `evaluate_py_fresh_eurekas_20260610.md` (R4 E1) · byte-shaving campaign portfolios `…20260523T154232Z.json`.
