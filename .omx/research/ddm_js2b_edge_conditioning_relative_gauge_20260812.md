# ddm_js2b edge conditioning on the corrected relative gauge

**Verdict:** **FORMULATION-DEAD on the fixed nine-seed, two-FiLM continuation catalog at stratified-random n32; selected exact-coder instance is RATE-DOMINATED.** F1 and F2 fired. F3 did not fire. No T4 acceptance row is queued.

**Authority:** `[macOS-CPU advisory, instrument floor 0.0131 S]`; local relative deltas only. No local absolute `d_seg` is progress, no exact CUDA score was measured, and Modal was not dispatched.

## Measured result

The corrected baseline is 50,389 local flips on the retained T4 scorer-input planes at batch 16 and eight CPU threads. The promoted scalar corresponds to 34,968 flips, a net difference of 15,421. The promoted per-pixel argmax field was not retained, so exact pixelwise local-vs-CUDA disagreement calibration is impossible from the scalar alone. The admissible bar used here is therefore a conservative rank calibration, not a fabricated disagreement map: take the lowest-margin 15,421 of the 50,389 local custody errors. Its upper edge is

`delta = 0.08036041259765625` SegNet logit units.

The full local-error margin distribution was measured from the retained six-flip-equivalent local logits: min `9.7752e-6`, p10 `0.0216868`, p25 `0.0618284`, p50 `0.1603413`, p75 `0.3237519`, p90 `0.5279029`, p95 `0.7041718`, p99 `1.2285750`, max `6.4442191`, denominator 50,389. The local logits' argmax differs from `lstars_local_on_custody.npy` at only six pixels, matching the corrected decode-equivalence result.

Nine real two-code FiLM states were rendered through the CP135/F26 receiver on a seeded stratified-random 32-pair sample. Their correction deltas were transported onto the retained T4 scorer-input planes and scored at the fixed batch/thread geometry. Across the fixed catalog, 15/15 beneficial flips had post-correction margin below delta. Thus F1 fires at **FORMULATION** scope for this catalog and sample. This does not kill learned implicit conditioning, other tensors, larger semantic moves, or a from-scratch training leg.

| rank | proposal | projected n600 total delta flips | projected n600 robust delta flips | tie-fragile beneficial fraction | disposition |
|---:|---|---:|---:|---:|---|
| 1 | `f26_pair_continue` | -37 | 0 | 1.000 | compensated and exact-coded; FOLDED |
| 2 | `f19_1512m_1335m` | +19 | 0 | 1.000 | semantic screen folded |
| 3 | `f19_1512m_399p` | +19 | 0 | 1.000 | semantic screen folded |
| 4 | `f19_623p_1335m` | +19 | 0 | 1.000 | semantic screen folded |
| 5 | `f19_623p_553p` | +19 | 0 | 1.000 | semantic screen folded |
| 6 | `f19_623p_1415m` | +37 | 0 | 1.000 | semantic screen folded |
| 7 | `f19_623p_1512m` | +38 | 0 | 1.000 | semantic screen folded |
| 8 | `f19_623p_399p` | +38 | 0 | 1.000 | semantic screen folded |
| 9 | `f19_623p_701m` | +38 | 0 | 1.000 | semantic screen folded |

The selected seed was put into the charter's full reference form rather than judged as a semantic-only toy. Exact signed-int12 carrier compensation evaluated all 24 singleton moves per eligible sampled row and converged after 12 passes, with accepted-row counts `32, 26, 22, 17, 14, 9, 4, 2, 2, 1, 1, 0`. Its stratified n32 custody-gauge pose mean moved from `0.0001757154697162195` to `0.00007209906008917241`, delta `-0.00010361640962704708`, so the local pose guard passed. This is a subset guard, not an n600 pose claim.

The compensated state rebuilt through the real F24S/WANS/CAP1 physical model, CP135 metadata pack, a retained 12-quality Brotli race per section, the fixed residual/RC64 token payload, deterministic stored ZIP, and the shipped CP135 receiver. All semantic, carrier, selector, HPAC, residual, and token parse-back identities passed. The archive is **186,294 B**, SHA-256 `057377f16867afa56df1677595f62de5ec532018dccb58c515894a9ef8f87d97`, exactly **+42 B** versus CP135. Because it buys zero projected margin-robust flips, F2 fires for this exact-coder instance: positive counted rate with no robust SEG denominator is an infinite price wall, worse than 3 B per robust flip.

The charter's T4 gate requires projected robust delta flips at most -2,000, archive delta at most +1,000 B, and pose delta below `+2e-6`. The byte and subset pose gates pass; the SEG gate misses by 2,000 robust flips. **QUEUED-WITH-A-FIRE-ORDER T4 rows: none.** The candidate disposition is **FOLDED**. No paid acceptance row is justified.

The stratified n32 pose/SEG/rate arithmetic gives a non-promotable policy-transfer estimate of `delta S = -0.0150392`, but it assumes the sampled-row carrier compensation can be solved across all 600 rows. The materialized archive modifies carrier coordinates only on the sampled rows and has no n600 pose score. It is therefore forbidden as an archive-level score projection and is not used for routing.

## Payload custody

- Final archive: `/Volumes/VertigoDataTier/pact/ddm_js2b_20260812/candidate/retained/archive.zip`, 186,294 B, SHA-256 `057377f16867afa56df1677595f62de5ec532018dccb58c515894a9ef8f87d97`.
- Calibration margins: `/Volumes/VertigoDataTier/pact/ddm_js2b_20260812/calibration/local_error_margins.six_flip_equivalent.npy`, 201,684 B, SHA-256 `8044dcca298a0324ab7f1c0bca02159d90d8b8036c8c499c37b63ae6b61481dc`.
- Every semantic seed retains WANS bytes, n32 receiver frames, fp32 logits, and argmax under `/Volumes/VertigoDataTier/pact/ddm_js2b_20260812/semantic_screen/candidates/`.
- Every carrier pass retains full n600 codes, n32 pose errors/outputs, and rendered frame 0 under `/Volumes/VertigoDataTier/pact/ddm_js2b_20260812/carrier_compensation/checkpoints/`.
- The final candidate retains semantic, CPR1, CAP1, selector carrier, raw F24S models, every Brotli race payload, split model pack, residual, RC64 token stream, member, archive, repeat archive, logits, and argmax under `/Volumes/VertigoDataTier/pact/ddm_js2b_20260812/candidate/`.
- Machine verdict: `/Volumes/VertigoDataTier/pact/ddm_js2b_20260812/FINAL_RESULT.json`.

## RECALL EVIDENCE

**Stores consulted:** `CLAUDE.md`; `AGENTS.md`; `PROGRAM.md`; `docs/operating_manual_craft_handoff.md`; `.omx/state/main_hot_state.md`; corrected js1 annex and CUDA custody JSON; js2/fd135 seeds; `.omx/research/CANONICAL_RESEARCH_INDEX*`; `sub015_DAG_*` FEED blocks; canonical equations registry; task/claim ledgers; the full PR135 ExperimentBook source/docs; CP135 composition code and receipts.

**Queries:** `implicit edge|edge-conditioned|joint int12|basis FiLM|quantize compensate|F26|batch shape|margin|CAP1|WANS|RC64|split Brotli|realization_breakeven_bytes_v1|frozen_scorer_fisher_curvature_margin_colocation_v1|cpu_cuda_score_gap_v1`.

Beyond the charter seeds, recall found four implementation-shaping facts:

1. `ddm_et4_pair17_c2_batch_seam_diagnosis_20260806.md` proves batch shape is part of the forward instrument. The runner fixes batch 16 and eight threads rather than treating deterministic mode as sufficient.
2. `screen_f25_f24_renderer_nearmisses.py` supplies the settled two-symbol continuation/near-miss catalog; pair-revert and single-code retries were not reopened. `solve_f26_iterative_joint_carrier.py` supplies the exact 24-neighbor, repeat-to-dry compensation form. This changed the plan from an invented perturbation sweep to a lineage-preserving catalog plus a 12-pass dry closure.
3. `ddm_lp135_lossless_pack_20260810.md` and `ddm_cp135_rate_compose.py` show that same-state ANS and CAP1/container work are already closed and that the real receiver requires the physical WANS/CAP1 grammar. This forced exact CP135 recompression and parse-back and prevented a raw semantic-byte estimate.
4. The canonical Fisher/margin, CPU-CUDA gap, ordinal-margin, and realization-breakeven laws support margin-first ranking and exact byte rent, but none can replace receiver/coder/scorer evidence. No registry equation displaced the charter's real-candidate requirement.

The live lane registry did not grant this arm the sole full-n600 scorer slot. The work therefore stayed within the charter-authorized stratified-random n32 scope and emitted no full-n600 local or T4 score row.

## Borrowed-substrate accounting

Borrowed/granted: CP135 archive and adapted receiver, PR135/F26 semantic renderer and W4 weights, HPAC/RC64 token state, CPR1/CAP1 carrier and basis, selector, scorer weights, GT cache, T4 scorer-input custody, and ExperimentBook joint-solve primitives. New here: corrected-gauge delta transport, retained nine-seed relative screen, conservative rank delta calibration, subset carrier compensation on CP135, exact CP135 recompression/parse-back, and the F1/F2 verdict. No original codec, frontier, or exact score is claimed.

## Skeleton annex queue

- `ddm_js2b`: **FORMULATION-DEAD at stratified-random n32; F1 + F2 FIRED** — the fixed nine two-FiLM seeds produced 15 beneficial flips and all 15 were below `delta=0.0803604`; the best compensated stack projected -37 total but 0 robust flips and exact-coded to 186,294 B (+42 B). Pose guard passed on the subset. T4 gate failed; no T4 row queued. **Disposition:** FOLDED. **Owner:** `ddm_js2b`. **Consumer store:** `/Volumes/VertigoDataTier/pact/ddm_js2b_20260812/FINAL_RESULT.json`. **Fire trigger:** none for this formulation; route the residual SEG obligation to the training leg.

## Follow-on disposition

- **QUEUED-WITH-A-FIRE-ORDER.** **Owner:** MAIN training-leg router. **Consumer store:** `/Volumes/VertigoDataTier/pact/ddm_js2b_20260812/training_route/ROUTE.json`. **Fire trigger:** a current-vehicle, from-scratch SEG training leg has a retained stage checkpoint and the sole scorer lane is free; consume the js2b F1/F2 receipt as a prohibition on direct two-W4-code continuation, and screen only a representation-changing learned implicit-conditioning actuator.

Own-vehicle frontier remains **LC2 S=0.16959899569230852 @ 187,226 B [contest-CUDA T4, n600]**. The effective CP135 frontier also remains unchanged at **S=0.16195513827824176 @ 186,252 B [contest-CUDA T4, n600]**.
