# ddm_tc1 — the token TAIL (120,321 B, 66 % of the archive): bound its remaining structure closed-form against the shipped HPAC coder, then price a counted shared mixer on top of HPAC — rc2's winning form applied to the largest section (charter, 2026-09-09)

Tokens: `[no-triality] [p0-ledger-ok]` · Owner: codex arm (astra xhigh) · Spawned by MAIN 2026-09-09 under the operator's standing GO ("big structural"; "All point to gestalt"). Source: rc1 ITEM 2 (`.omx/research/ddm_rc1_adaptive_recode_race_of_the_model_sections_20260905.md`: "whether the RC64 tail has an analogous width-change blind spot is unmeasured"), rc2's mixer result (move 33), gs3 Addendum 15 (the rate corner: −28,310 B at held distortion). Axes: bytes exact, zero distortion; `score_claim=false` until a T4 row; MAIN fires.

## MANDATE

The token tail is 120,321 B of the 181,414 B archive and is coded by the shipped HPAC context model through the RC64 range coder. Every rate arm so far touched the MODEL sections (rc1 −1,733 B, rc2 −231 B) or the tail's FIELD (sj1 pre-distortion). Nobody has bounded how much structure the tail still holds under a coder that mixes the HPAC predictor with cheap counted secondary predictors — the form that just paid on the model rows. This arm does the closed-form bound first, then prices a counted shared mixer over {HPAC p(token|context), order-k spatial neighbours, previous-plane co-located symbol, run-length state} — a zero-distortion rate move on the section where 1 % = 1,203 B = −8.0e-4 S.

## SCOPE

1. RECALL: the shipped tail coder path (find the RC64/HPAC encoder+decoder under `src/tac/pr130_runtime/fx1_runtime_tree/` and `experiments/ddm_cl2_hpac_prior_capacity_ladder.py` — the pricer that re-encodes a field), hc1 (`token-stream-is-one-binary-question`: 97.8 % of the stream is one binary question), bd1 (`temporal_context_is_saturated_on_the_label_field_bframes_do_not_transfer_20260905`: B-pyramid 2.8–5.7 % vs an 8 % bar; read WHY the bar was 8 %), mc1 (`motion_compensated_previous_plane_closed_at_formulation_scope_20260905`), the reorder law (`reordering-pays-iff-the-coder-has-no-context-model`), cl2/cl3 (capacity closed), fs2/fs3 (−log2p direction-dependent; AVERAGE ≠ MARGINAL), fe1's container-break law. The live pointer tree `/Volumes/VertigoDataTier/pact/ddm_rc2_hpac_semistatic_mixing/candidate_runtime` (READ ONLY; tail 120,321 B — verify by census). `tools/subagent_checkpoint.py read --subagent-id ddm_tc1` first.
2. Bound leg (closed-form, on the exact shipped field): (a) the shipped HPAC's realized bits per symbol by symbol class and by context (from the encoder's own probabilities — instrument it read-only); (b) empirical conditional entropies with Miller–Madow correction for candidate extra contexts (order-2/3 spatial, previous-plane co-located, run-length, row band) GIVEN the HPAC prediction bucket — i.e. how much of the residual −log2 p is explained by information HPAC does not condition on; (c) a table: context · bits saved (bound) · counted table/weight bytes · net. Report the ceiling honestly: the bound is what ANY counted mixer of these predictors could save at most.
3. If the bound net ≥ 500 B: build the mixer as a counted, receiver-decodable stage (online logistic mixing of HPAC with the admitted predictors; shared weights ≤ 64, int8/fp16 counted; no field change) and race it: encode ×2 (twin byte-identical) → receiver decode returns the EXACT field → census (only the tail + its reader moved) → container sweep (fe1's law) → exact archive bytes. If the bound net < 500 B: close the door at the bound with the table; build nothing.
4. If net bytes < 181,414 B: stage on the live tree, public-path decode identity, public-smoke block via the validator's own checker (frontier = live pointer), `tools/make_candidate_seal.py` contest-CUDA, name `ddm_tc1_tail_shared_mixer` (no `v<digit>`); "SEAL READY: <path>"; MAIN fires.
5. Memo `.omx/research/ddm_tc1_token_tail_bound_and_shared_mixer_pricing_20260909.md`; equations leg: register `token_tail_context_mixing_bound_v1` via `register_canonical_equation` (`tac.canonical_equations`) with the bound table as the first anchor (or anchor on hc1's law if it fits).

## HARD CONSTRAINTS

- `upstream/` READ-ONLY. NO Modal fire. No scorer (zero distortion — PROVE by exact field decode + census). Never write into `submissions/semantic_joint_ctxmix/`, the live pointer tree, or the sj1/pc2/fe1/rw1/rc3 trees; your tree `/Volumes/VertigoDataTier/pact/ddm_tc1_tail_shared_mixer/` (`df -h` first; fields are npz, small).
- COMPOSITION: sj1's pass 4 CHANGES the tail (new field) and will promote soon; your mixer must be field-agnostic (a coder change, not a field change) and you re-base onto the new tail when the pointer moves (re-read the pointer before every encode/seal); rc3 owns the hpac model section; do not touch it.
- The local SCORER LANE belongs to MAIN, always. Do NOT write who holds it into a charter (the #1210 stale-precondition genus, memo ddm_bz2_bornsmall_capacity_ceiling 2026-08-29).
- CPU ≤ 2 procs (the tail encode is minutes; the bound pass is one read of the field). Detached only via `tools/launch_detached_process.py … --nice 10 --nice-best-effort`; no `nohup`/`&`/clock waiters; artifact-bound waits ≤ 780 s; kill the process group on timeout.
- Serializer commits w/ post-edit `--expected-content-sha256`; `.py` = 2 review passes. Tokens `[no-triality] [p0-ledger-ok]`. NEVER a Co-Authored-By or AI-attribution trailer. Sandbox git refusal → serializer fallback bundle + receipt.
- ALWAYS KEEP THE PAYLOAD; VERIFIED-AT-SOURCE for 120,321 B, the HPAC context definition, the RC64 coder; PER-PAIR receipts n/a (zero distortion; the census is the receipt); CLOSED-FORM-FIRST is the gate.
- Checkpoint discipline: `tools/subagent_checkpoint.py --subagent-id ddm_tc1 …` every ~10 tool uses.

## PRIOR NEGATIVE SIGNAL (bearing dead-ends this charter consumes)

- Temporal context is SATURATED on the label field: B-pyramid 2.8–5.7 % vs the 8 % bar, P(32)/P(1) = 1.07–1.13 (bd1). A previous-plane predictor enters here ONLY as one input to a counted mixer, priced against that bar's reasons, never as a new reference structure.
- Motion-compensated previous plane: +160 B vs a 5,000 B bar (mc1) — field change ≠ rigid motion; do not re-run.
- Reordering pays IFF the coder has no context model; the tail HAS one (HPAC) — no permutation designs.
- HPAC capacity closed both ways (cl2/cl3); the coder, not the model, is the lever — and even there rc2's per-cell tables lost; only SHARED parameters paid.
- −log2p is direction-dependent and AVERAGE ≠ MARGINAL (fs2/fs3, 2.24×): the bound must be computed on the real encoder's probabilities, not an average bits/token.

## OPTIMAL FORM

- Family exemplar: rc2's counted shared logistic mixer (move 33; memo `.omx/research/ddm_rc2_hpac_semistatic_depth_mixing_prior_and_container_sweep_20260908.md`, landing fb39f2139) — the reference form for a counted mixer with exact encode×2, bit-identical decode, census, container sweep; the bound instrument follows rc1's Miller–Madow order-1 bound (`ddm_rc1_adaptive_recode_race_of_the_model_sections_20260905.md`).
- SCOPE reductions declared per row: the candidate-context set is the five named families (SCOPE); ≤ 1 mixer built (SCOPE). MECHANISM reductions FORBIDDEN: an average-bits bound, uncounted weights, a subset of pairs in the bound, library-path-only decode identity.
- **PRIOR-LAW PREDICTION (falsifiable):** the bound finds 0.8–2.5 % of the tail (960–3,000 B) reachable by contexts HPAC does not condition on, dominated by run-length/row-band state on the 97.8 % binary question; a ≤ 64-weight mixer converts 40–60 % of the bound (−400…−1,800 B). FALSIFIER: bound net < 500 B — the tail is at its context ceiling under HPAC; close with the table; build nothing.

## DELIVERABLE

The memo with the bound table (and the race if built), retained payload, the seal if admitted. Commit via the serializer. End with the live frontier line.
