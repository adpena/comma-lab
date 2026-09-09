# ddm_rc3 — rc2's gated successor: a richer GLOBALLY-SHARED low-parameter mixer for the IHS1 model rows, built ONLY if closed-form pricing predicts ≥ 150 B of the remaining ≈953 B (charter, 2026-09-09)

Tokens: `[no-triality] [p0-ledger-ok]` · Owner: codex arm (astra xhigh) · Spawned by MAIN 2026-09-09 under the operator's standing GO. Source: rc2's NEXT_IF_RESUMED fire trigger ("only after the sealed RC2 row passes T4 — it did, move 33 — and only with a closed-form counted-parameter design predicting at least another 150 B"), memo `.omx/research/ddm_rc2_hpac_semistatic_depth_mixing_prior_and_container_sweep_20260908.md` (landing `fb39f2139`). Axes: bytes exact, zero distortion; `score_claim=false` until a T4 row; MAIN fires.

## MANDATE

Move 33 (rc2): a COUNTED 8-byte shared logistic mixer across depths took −231 B off the IHS1 model rows at zero distortion; the 16-byte mixer won raw J by 5 B but lost the archive objective by 1 B; semi-static tables lost 1,210 B (closed). rc2 measured ≈953 B still between its raw J and the order-1 Miller–Madow bound, and named the live hypothesis: a richer but still globally shared low-parameter mixer (8–16 counted weights already converted 216–221 B where per-cell tables failed). This arm answers by CLOSED-FORM PRICING FIRST which mixer families can convert ≥ 150 B net of their counted parameters, builds only the winner, and seals a zero-distortion candidate on the LIVE pointer if the real encode confirms.

## SCOPE

1. RECALL: rc2 memo (whole; the mixer form, the 36 container cells, the 16 decoded designs), rc1 memo (`ddm_rc1_adaptive_recode_race_of_the_model_sections_20260905.md` §6/§7), `experiments/ddm_rc2_hpac_semistatic_depth_mixing.py` + `ddm_rc2_hpac_semistatic_mixing_codec.py` (the shipped reader path), the live pointer tree `/Volumes/VertigoDataTier/pact/ddm_rc2_hpac_semistatic_mixing/candidate_runtime` (READ ONLY; archive 181,414 B sha `c810c2c7f72e57670dc29bde27d584b18aa82feff68b063936a61dca89cf671e`). `tools/subagent_checkpoint.py read --subagent-id ddm_rc3` first.
2. Closed-form leg (before any coder run): on the exact IHS1 rows, compute the conditional-entropy bounds for candidate mixer contexts (depth × symbol position × previous symbol × sibling group; order-2 where the count supports Miller–Madow), and for each family the counted-parameter cost (weights at int8/fp16 + any context table) — a table: family · predicted coded bytes · counted parameter bytes · net vs rc2's 12,112 B. Only families with predicted net ≥ 150 B proceed.
3. Build the ≤ 2 admitted families as counted, receiver-decodable mixers (online logistic mixing, shared weights; learning-rate as a counted byte if not derivable); race through pack → stage → encode ×2 (twin byte-identical) → receiver decode returns the EXACT rows (bit-identical) → census (only the hpac section + reader moved) → container sweep (fe1's law: `(ck2, q, lgwin)`; ties to the shipped shape by byte identity).
4. If net bytes < 181,414: stage on the live tree, public-path decode identity, public-smoke block via `tac.candidate_seal._public_smoke_problems` (frontier role = the live pointer), `tools/make_candidate_seal.py` contest-CUDA, name `ddm_rc3_shared_mixer_successor` (no `v<digit>`); "SEAL READY: <path>"; MAIN fires.
5. Memo `.omx/research/ddm_rc3_shared_mixer_successor_closed_form_gated_20260909.md`: the pricing table, the race, prediction vs measured; equations leg: anchor on `model_section_adaptive_recode_ceiling_v1` via `update_equation_with_empirical_anchor` (`tac.canonical_equations`).

## HARD CONSTRAINTS

- `upstream/` READ-ONLY. NO Modal fire. No scorer (zero distortion by construction — PROVE by bit-identical decoded rows + census). Never write into `submissions/semantic_joint_ctxmix/`, the live pointer tree, or the sj1/pc2/fe1/rw1 trees; your tree `/Volumes/VertigoDataTier/pact/ddm_rc3_shared_mixer_successor/` (small; `df -h` first).
- COMPOSITION: sj1 pass 4 (tail + carrier), pc2 (carrier scales), fe1 (semantic FiLM codes), rw1 (renderer weights) are live; you touch ONLY the hpac model section + reader; whoever lands second re-bases (re-read `.omx/state/canonical_frontier_pointer.json` before staging/sealing).
- The local SCORER LANE belongs to MAIN, always. Do NOT write who holds it into a charter (the #1210 stale-precondition genus, memo ddm_bz2_bornsmall_capacity_ceiling 2026-08-29).
- CPU ≤ 2 procs. Detached only via `tools/launch_detached_process.py … --nice 10 --nice-best-effort`; no `nohup`/`&`/clock waiters; artifact-bound waits ≤ 780 s; kill the process group on timeout.
- Serializer commits w/ post-edit `--expected-content-sha256`; `.py` = 2 review passes. Tokens `[no-triality] [p0-ledger-ok]`. NEVER a Co-Authored-By or AI-attribution trailer. Sandbox git refusal → serializer fallback bundle + receipt, named in the final message.
- ALWAYS KEEP THE PAYLOAD; VERIFIED-AT-SOURCE for 12,112 / 953 / 8,469 / 231 (re-measure on THIS tree); CLOSED-FORM-FIRST is the gate: no coder run before the bound table.
- Checkpoint discipline: `tools/subagent_checkpoint.py --subagent-id ddm_rc3 …` every ~10 tool uses.

## PRIOR NEGATIVE SIGNAL (bearing dead-ends this charter consumes)

- Semi-static previous-value tables: CLOSED at formulation scope (best counted J 10,853 B, −1,210 B vs rc1) — rc2 memo. Per-cell adaptive designs converted 10.3 B of 1,139 B (rc1 §6). Shared parameters are the only form that has paid.
- 16-weight mixer lost the archive objective to the 8-weight by 1 B while winning raw J by 5 B — rc2: parameter bytes are counted at the archive, not at J.
- Container retuning alone: 0 B (q11 × lgwin 22/23/24 tie) — rc2; container search is a lever on EDITED bytes only (fe1).
- Capacity axis CLOSED both directions (cl2/cl3); coder strength substitutes for capacity — do not touch the model's capacity.

## OPTIMAL FORM

- Family exemplar: rc2's landing (`fb39f2139`, pointer move 33 memo `.omx/research/ddm_rc2_t4_hpac_semistatic_mixing_20260909_pointer_move_33_20260909.md`) — the reference form: counted mixer, exact pack/stage/encode×2, bit-identical decode, census, container sweep, seal with the public-smoke block.
- SCOPE reductions declared per row: ≤ 2 families built (SCOPE, chosen by the bound table). MECHANISM reductions FORBIDDEN: uncounted parameters, library-path-only decode identity, a fit presented as a bound.
- **PRIOR-LAW PREDICTION (falsifiable):** an order-2/position-aware shared mixer with ≤ 32 counted weights converts 200–400 B net; the bound table shows ≥ 600 B of the 953 B is context-reachable at order 2. FALSIFIER: no family predicts ≥ 150 B net — close the model-prior door at formulation scope with the bound table as the receipt; build nothing.

## DELIVERABLE

The memo with the pricing table and race, retained payload, the seal if admitted. Commit via the serializer. End with the live frontier line.
