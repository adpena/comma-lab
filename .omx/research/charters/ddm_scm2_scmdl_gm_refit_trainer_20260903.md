# ddm_scm2_scmdl_gm_refit_trainer — SCMDL gate 2 made EXECUTABLE: a fitted cross-group causal schedule + retrained COUNTED HPAC integer model, exact-priced through the retained RXC1 instrument on the already-retained SFP1/XOV1 fields (task #1374, the frontier's sole surviving rate route)

## MANDATE

Operator 2026-09-03 verbatim: *"let's get back to working on our frontier score lowering work"* +
*"continue with all"*, under the standing GO (`standing-go-full-authority-frontier-lowering-20260902`).
THE GOAL is sub-0.12. The live demand is the RATE corner: −42,016 B at held distortion (archive
≤ 137,986 B), or on the joint pool the jc1 arithmetic: joint field+model pool 126,926 B (tokens
113,411 + HPAC model 13,515), required joint-pool cut 39,522.14 B (31.14%), affine archive ceiling
140,479.86 B (`ddm_jc1_afr_rc64_joint_redesign_20260901.md`, commit f9937b4e3c). Seven of eight
mechanism classes FOLDED with receipts; the survivor is SCMDL: field X + causal schedule/context graph
G + probability model M co-designed. Gate 1 (RXC1) proved EXACT suffix restart (64/64 byte-identical)
but not economics (450.6 s/proposal; batch-of-proposals-per-suffix granularity) —
`ddm_rxc1_gen3_gate1_verdict_20260901.md`. Gate 2 has NEVER EXISTED as an executable: JBP1 measured
the one row it could (A `xov1_bhw5506`: 177,052 B, −2,950 B, 7.464% of demand, REFUSED
+36,572.14 B) and BLOCKED B1–B3 because every SFP1 proposal says `refit_required: true` and names
`refit_cross_group_causal_schedule`, for which NO implementation, trained model, decoder binding, or
counted model payload exists (`ddm_jbp1_joint_batch_price_verdict_20260901.md`, commit 625de245ee;
blocker JSON `/Volumes/APDataStore/pact/ddm_jbp1_joint_batch_price/SFP1_GM_REFIT_BLOCKER.json`).
JCB1 then TERMINAL-REFUSED a batched pricer because its charter neither pinned the candidate sources
nor reconciled DDS1/JBP1 (`ddm_jcb1_scmdl_batched_suffix_pricing_20260902.md`). This charter IS that
reconciliation, and it builds the missing executable. It is the frontier's critical path.

## RECONCILIATION (binding; the JCB1 refusal conditions, answered)

- DDS1 (`ddm_dds1_decoder_derivable_verdict_20260901.md` a19f9f2555 + its ceiling re-adjudication
  `ddm_dds1_ceiling_readjudication_20260901.md`): the born-statistics rider is CLOSED at a ~2.08 B
  ideal-coder ceiling; the GF1 packet is 77.6× underwater. This charter does NOT reopen any
  decoder-derivable born context. M here is the HPAC integer context-mixing model family ONLY.
- JBP1 (625de245ee): row A stands as MEASURED under shipped G/M (−2,950 B). This charter does NOT
  re-price A under shipped G/M. It prices A AND B1–B3 under the FITTED G/M this arm builds, which is
  the mechanism JBP1 said was missing — a different measurement, reported as such.
- JCB1's missing pins: every admitted source object is pinned BELOW by path + sha from JBP1's retained
  custody. Re-verify each sha before use; a mismatch is a typed refusal, never a re-derivation.

## ADMITTED SOURCE OBJECTS (pin, verify, refuse-on-mismatch)

- AFR1 archive `cbb8d928a8ccdd3f5103da1d4a8d38d0662a5e5615266b923b5f8350d405bf25` (180,002 B); RC64
  token stream `5601d6fd792c60c176e7cb7478e6033c4ed9a7e87404582340ed3f50ed60cfe3` (113,411 B);
  shipped HPAC section `602115b323b0e403d08287af9b273a2d4fb23e026d83c1f6e4609ed77ef98f98` (13,515 B).
- Fields + overlays: `/Volumes/APDataStore/pact/ddm_jbp1_joint_batch_price/retained/{fields,overlays,exact}/`
  — A field sha `e685c4bf7fbea1188b64f521487192196eaae99c8b8b335b770586ab984585fa`, overlay
  `7953c9164cc5ac4f3fa59b8715a6eaecdc9cb11e61cba58d20c3d24db38eea63`; B1–B3 per the JBP1 receipt
  manifest there (read the shas from the receipts; do not type them from memory).
- RXC1 instrument: `experiments/ddm_rxc1_restartable_exact_coder.py` (9cf2fd5d82) + custody
  `/Volumes/APDataStore/pact/ddm_jc1/restartable_exact_coder/` (PREFLIGHT/BASELINE/NULL_REPLAY/SCREEN).
- HPAC family source: `submissions/semantic_joint_ctxmix/cpr1/hpac_integer.py` (7ba53d1b84) — the
  SHIPPED integer context mixer whose cross-group index is simultaneously the coding partition and the
  causal mask baked into the trained convolution weights (source-verified in JBP1). The group-conditioned
  training we already did: `experiments/ddm_gb1_groupbin8_conditioning.py` (f00d7994a5) — reference
  form for "retrain the HPAC model under a different group plan and count it".

## SCOPE (build → price → refuse-or-fire; every payload retained)

1. **Executable cross-group schedule G.** Implement `refit_cross_group_causal_schedule` as a real,
   receiver-executable schedule object: a counted, parse-back-bound group plan that the SHIPPED
   receiver family can consume (extend the cpr1 receiver ONLY in a copy under this arm's tree; the
   shipped packet tree `submissions/semantic_joint_ctxmix/` is FROZEN — PR #140 is live on it).
   The schedule proposals come from the SFP1 schemas (`experiments/ddm_sfp1_scmdl_field_proposal_prep.py`)
   — implement what they declare, do not invent a fourth.
2. **Retrained counted M.** Train the HPAC integer model under each fitted schedule on the AFR1 field
   (and on each SFP1 edited field for its own row), integer-export it exactly as the shipped section
   is exported (bit-identical export path; `integer_model_io.py`), COUNT its bytes, and bind it to the
   receiver copy. Model bytes are counted; nothing video-derived may hide in code (rule 118).
3. **Exact price** each row {A, B1, B2, B3} × {fitted G/M} through the RXC1/RC64 instrument:
   full-state exact re-encode, receiver parse-back identity (decode the archive with the receiver
   copy → field byte-identical to the input field), determinism repeat (two encodes, identical bytes).
   Report per row: model bytes, token bytes, framing, total archive bytes, delta vs 180,002 B,
   fraction of the 39,522.14 B demand, and the joint-pool figure vs the 87,403.86 B allowance.
4. **Decision rule (pre-registered).** Any row whose complete receiver-closed archive is
   ≤ 140,479.86 B → emit a TYPED FIRE ORDER for MAIN's local scorer lane (full-n600 realized Seg/Pose,
   BR2 payload-retaining protocol; B rows change X so distortion is NOT inherited). Otherwise typed
   REFUSED with the exact shortfall. Rate rows are scorer-free; no scorer runs in this arm.
5. **Economics leg (the RXC1 outer-loop cost).** Report seconds per row and whether the fitted-G/M
   path amortizes the suffix replay; a bounded-reset causal state is OUT OF SCOPE (QUEUED-STRUCTURAL-
   LOCALITY row stays queued) — name it, do not build it.

## HARD CONSTRAINTS

- `upstream/` READ-ONLY. `submissions/semantic_joint_ctxmix/` READ-ONLY (live PR #140 tree).
- NO scorer run and NO Modal fire from the arm; MAIN owns the local scorer lane and dispatch —
  never write a lane-occupancy claim into this charter's reasoning (#1210 stale-precondition genus);
  emit a typed fire order and let MAIN fire it.
- Serializer commits w/ post-edit `--expected-content-sha256`; `.py` = 2 genuine review passes.
- ALWAYS KEEP THE PAYLOAD; bulky receipts to `/Volumes/APDataStore/pact/ddm_scm2_scmdl_gm_refit_trainer/`
  (AP has ~30 GiB free — budget ≤ 8 GiB; refuse if free < 1.5 GiB; Vertigo has 185 GiB as overflow).
- DETACHED >30-MIN COMPUTE: training + full-state re-encodes (~15 min each) exceed 30 min in
  aggregate → launch ONLY through the canonical launcher
  `.venv/bin/python tools/launch_detached_process.py --output-dir <run_dir> --done-receipt <name> -- <cmd...>`
  (true start_new_session detach + manifest + .done receipt the fleet watcher delivers to MAIN);
  hand-rolled detaches are blocked by the launch guard. Crash-resumable stage checkpoints; the arm
  MONITORS; MAIN harvests the done-receipt.
- CLOSED-FORM-FIRST: the group plan / causal mask is a deterministic combinatorial object — derive it
  exactly; only M is fitted, and its fit owes the one-line reason (a context-mixing model is a
  trained object by construction).
- Resumability P0: per-stage checkpoints, `--resume-from`, seeded determinism, provenance JSON.
- Do NOT touch `/Volumes/APDataStore/pact/ddm_jc1/restartable_exact_coder/` inputs (read-only custody).

## PRIOR NEGATIVE SIGNAL (bearing dead-ends this charter consumes)

- `ddm_jbp1_joint_batch_price_verdict_20260901.md` — A under shipped G/M cuts only 2,950 B; B1–B3
  BLOCKED-MISSING-EXECUTABLE-GM-REFIT (the exact gap this arm fills).
- `ddm_jcb1_scmdl_batched_suffix_pricing_20260902.md` — TERMINAL-REFUSED on source pins and
  DDS1/JBP1 supersession; answered in RECONCILIATION above.
- `ddm_dds1_ceiling_readjudication_20260901.md` — born-stats M rider ceiling ~2.08 B: CLOSED; not reopened.
- `ddm_rxc1_gen3_gate1_verdict_20260901.md` — 0/32 terminal adaptive-state reconvergence: prices are
  suffix-level, batch-of-proposals; no per-proposal outer loop.
- `ddm_ccs1_causal_schedule_builder_verdict_20260901.md` — a sparse 512-leaf causal table produced a
  607,228 B stream vs the shipped 113,411 B: a NEW model must be a parameter-sharing context mixer of
  the shipped family's class, not a table.
- `ddm_jt23_coder_collection_compose_verdict_20260826.md` — generic coder swaps on the fixed body: 0 B.
- `ddm_rr9_reorder_refit_20260824.md` — within-group reorder under the fixed model: 0 B; cross-group
  order = a different trained model (the reason M must be retrained here).

## OPTIMAL FORM

- Family exemplar: the shipped integer HPAC context mixer, reference
  `submissions/semantic_joint_ctxmix/cpr1/hpac_integer.py` (commit 7ba53d1b84), and the
  group-conditioned retrain that produced the GB1 rows, reference
  `experiments/ddm_gb1_groupbin8_conditioning.py` (commit f00d7994a5), receipt: the GB1 stage pins in
  `compress.py` (f20b5e4baf: gb1_pointer 180,215 B, gb1_joint 180,192 B).
- SCOPE reductions declared per row (n-pairs of the training set may be reduced for the SMOKE only;
  every PRICED row is the full 600-pair field). MECHANISM reductions FORBIDDEN: no fixed-G/M stand-in
  for a fitted row (JBP1 already refused that fake); no float model where the shipped is integer.
- **PRIOR-LAW PREDICTION (falsifiable):** the price = (form, conditioning) law (gp2) predicts a fitted
  cross-group G/M on the SAME field moves the joint pool by at most low single-digit percent — the
  refit re-partitions the same information. Concretely: every row lands ABOVE 140,479.86 B.
  FALSIFIER: any receiver-closed, repeat-identical row ≤ 140,479.86 B — count it plainly and fire the
  scorer order.

## DELIVERABLE

`.omx/research/ddm_scm2_scmdl_gm_refit_trainer_20260903.md` — typed rows: per candidate
{schedule id, model bytes, token bytes, framing bytes, archive bytes, Δ vs 180,002, fraction of
demand, parse-back identity, repeat identity, seconds} + the typed decision (REFUSED-WITH-SHORTFALL /
FIRE-ORDER) + RECALL EVIDENCE + NEXT_IF_RESUMED + LIVE-HYPOTHESES + DEAD-ENDS. Commit via the
serializer. Cite `docs/operating_manual_craft_handoff.md`. End with the own-vehicle frontier line.
