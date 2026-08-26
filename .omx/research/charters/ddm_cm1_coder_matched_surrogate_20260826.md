# ddm_cm1_coder_matched_surrogate — build+validate the coder-matched rate surrogate that unblocks the no1 row-1 three-term objective (canonical ledger row ddm_no1_row1_three_term_objective; memo ddm_no1_new_object_derivation_20260826.md)

## MANDATE

Operator 20260826: *"Codex is available too"* + *"Prefer codex"* (routing), executing the
routed finding from `.omx/research/ddm_no1_new_object_derivation_20260826.md` row-1: the
three-term objective (rate+pose in the live loss) is QUEUED behind a coder-matched rate
surrogate, because the `entropy` form is MEASURED anti-correlated with real coded bytes
(rsf1: ρ = −0.7235). Build and validate a surrogate whose correlation with REAL re-encode
bytes on the live dx2/gb1-lineage token stream is measured on held-out rows, so the
three-term training route either becomes fireable or is honestly re-priced. This is the
named prerequisite for one of only two live sub-0.12 routes surviving the no1 derivation.

## SCOPE

1. RECALL AT SOURCE before building: (a) the rsf1 memo
   `.omx/research/ddm_rsf1_rate_surrogate_fidelity_20260801.md` (+ its rows jsonl) — the
   anti-correlation law and its exact stream/object; (b) the sv2 memo
   `.omx/research/ddm_sv2_smevr_base_rule_race_20260803.md` (landing `fec6dae38b`) —
   mechanism: the live coder pays for LZ MATCH STRUCTURE / adaptive-context state, not
   symbol rank, which is WHY marginal entropy fails; (c) the sm2 artifacts dir
   `.omx/research/ddm_sm2_20260805/` (landing `29e47a600f`) — the raced-predictor protocol
   and its 152 banked real re-encode rows (best linear RMSE 41,187.6 vs entropy 41,866.2
   vs smevr 42,799.4): CONSUME these rows, do not re-generate what exists; (d) fs2's real
   re-encode price machinery, memo `.omx/research/ddm_fs2_rc4_drop_carrier_resolve_20260820.md`
   — the ground-truth producer for new rows.
2. CANDIDATE FAMILY (race, never presume): (i) EXACT incremental — the real F26/HPAC
   adaptive coder run on the perturbed stream (correlation 1.0 by construction; deliver its
   MEASURED per-eval wall-clock so the trainer economics are decidable); (ii) truncated/
   windowed exact coding (real coder on a local window around the edit, correlation +
   wall-clock measured); (iii) fitted calibrated proxies per the sm2 protocol trained on the
   banked rows + fresh fs2 rows, validated HELD-OUT. Report ρ (Pearson + Spearman) vs real
   re-encode bytes per candidate on held-out rows, plus per-eval cost.
3. VERDICT + WIRING: name the winner with its measured (ρ, cost) pair; state whether the
   three-term route is FIREABLE (surrogate exists with ρ ≥ 0.9 held-out at trainable cost)
   or must re-price to exact-incremental coding; write the recommendation into the memo and
   update the canonical ledger row `ddm_no1_row1_three_term_objective` (via
   `tools/canonical_task_status.py update`, actor ddm_cm1, session-id ddm_cm1) with the
   surrogate verdict — the row's other prerequisite (the wd3 single-seed variation, per
   the row-1 prerequisites in `.omx/research/ddm_no1_new_object_derivation_20260826.md`;
   Metal-bound) stays MAIN-owned.

## HARD CONSTRAINTS

- `upstream/` READ-ONLY. NO Modal fire from the arm (MAIN owns dispatch + single-flight).
- Serializer commits w/ post-edit `--expected-content-sha256`; `.py` = 2 genuine review passes.
- ALWAYS KEEP THE PAYLOAD; bulky receipts to `/Volumes/APDataStore/pact/ddm_cm1_coder_matched_surrogate/`.
- Every byte number from the REAL coder path (F26/HPAC machinery or fs2's re-encoder) —
  entropy estimates are banned as ground truth per the rsf1 law. Do NOT touch the ddm_lm1
  arm's surface (learned-model-vs-HPAC-tables falsifier, live in parallel) — cm1 owns the
  SURROGATE question, lm1 owns the TABLE-REPLACEMENT question; shared reads OK, no shared
  writes. No scorer runs needed (rate-only measurement).

## PRIOR NEGATIVE SIGNAL (bearing dead-ends this charter consumes)

- rsf1 (memo `.omx/research/ddm_rsf1_rate_surrogate_fidelity_20260801.md`): entropy
  surrogate ρ = −0.7235 vs real coded bytes — the static-marginal family is the named dead
  end this charter routes around.
- blind-subspace narrowing (memo dir `.omx/research/ddm_sm2_20260805/`, commit
  `29e47a600f`): marginal entropy is BLIND to the permutation class — live pair
  permutations move real bytes 13–13,466 B at max |Δentropy| = 1.38e-14 bits; any surviving
  surrogate must see ordering/context state.
- sv2 (memo `.omx/research/ddm_sv2_smevr_base_rule_race_20260803.md`, `fec6dae38b`): 30
  exact-rh1 arms + SMEVR itself LOSE on the IX2TOK01 bulk — symbol-rank surrogates
  mispredict because the live coder pays for LZ match structure; consume this as the
  mechanism constraint on proxy design.
- the SUM/affine arm (same sm2 dir `.omx/research/ddm_sm2_20260805/`): already BUILT and
  RACED — best linear fit only marginally beat entropy and was honestly NOT PROMOTED; do
  not rebuild it, extend from its banked rows.

## OPTIMAL FORM

- Family exemplar: the sm2 raced-predictor protocol is the reference form — memo landed at
  commit `29e47a600f` (152 banked real re-encode rows, held-out validation, honest
  non-promotion); receipt rows live in the sm2 landing artifacts. cm1 extends that exact
  protocol with the two exact-coding candidates and fresh fs2 ground truth.
- SCOPE reductions declared per row: held-out row count may be bounded (state n); window
  sizes for candidate (ii) swept over a small declared set. MECHANISM reductions FORBIDDEN:
  no toy coder, no synthetic token streams, no entropy ground truth.
- **PRIOR-LAW PREDICTION (falsifiable):** the rsf1 + blind-subspace laws
  (`.omx/research/ddm_rsf1_rate_surrogate_fidelity_20260801.md` +
  `.omx/research/ddm_sm2_20260805/`) predict every STATIC
  (context-free) surrogate stays below ρ ≈ 0.5 held-out, while exact/windowed real coding
  is the only family reaching ρ ≥ 0.9. FALSIFIER: a static fitted proxy achieving ρ ≥ 0.9
  on held-out real re-encode rows — if it lands, count it plainly and promote it (it is the
  cheapest trainer-compatible form).

## DELIVERABLE

`.omx/research/ddm_cm1_coder_matched_surrogate_20260826.md` — typed rows: per-candidate
(family, ρ_pearson, ρ_spearman, n_heldout, per-eval wall-clock, verdict) + the FIREABLE /
RE-PRICE routing verdict + ledger-row update receipt + GESTALT-DELTA line + payload paths
w/ sha256s. Commit via the serializer. End with the own-vehicle frontier line.
