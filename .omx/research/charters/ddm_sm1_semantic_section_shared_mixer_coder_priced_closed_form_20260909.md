# ddm_sm1 — the semantic renderer section (30,246 B, the largest model-like section, untried at the coder level): bound its remaining structure under the shipped rc1 adaptive coder, then price a COUNTED shared mixer — rc2/rc3's winning form — built only if the bound clears 150 B (charter, 2026-09-09)

Tokens: `[no-triality] [p0-ledger-ok]` · Owner: codex arm (astra xhigh) · Spawned by MAIN 2026-09-09 under the operator's standing GO ("rate representation mandatory"; gs3 Addendum 17: the seg half is at a capability floor, the live demand is the rate corner). Sources: rc1 (`.omx/research/ddm_rc1_adaptive_recode_race_of_the_model_sections_20260905.md`: the adaptive per-group tree coder took −610 B off the SM3R renderer body — packed integer codes of `unpack_semantic`'s int4 weights + fp16 scales — and −1,123 B off the hpac rows), rc2/rc3 (`35c631244`: counted shared logistic mixers converted 231 + 201 B of the hpac rows' order-1 gap where per-cell tables lost), tc1 (`token_tail_context_mixing_bound_v1`: the bound-then-mix instrument, 549 B on the tail). Axes: bytes exact, zero distortion; `score_claim=false`; MAIN fires.

## MANDATE

The semantic section is 30,246 B of the 181,521 B archive: the SM3R renderer body (int4 codes with per-row fp16 scales for `token_embed`, `coord_mix`, four TokenBlocks (dw/pw/norm/film), `head`; `frame_embed` 600×8 int4 — see `src/tac/pr130_runtime/fx1_runtime_tree/inflate.py:unpack_semantic`), coded by rc1's adaptive per-group tree coder inside the archive container. Every rate arm since rc1 touched the hpac rows (rc2, rc3) or the tail (tc1); nobody has bounded the semantic section's remaining structure under rc1's coder or priced the shared-mixer form on it. Per-tensor weight statistics (row-wise scale structure, sign/magnitude runs across a conv's input dimension, dw/pw correlations) are exactly what a counted shared mixer over the adaptive coder's predictor can exploit. Bound first; build only if the bound clears; seal a zero-distortion candidate (decoded weights bit-identical) on the LIVE pointer.

## SCOPE

1. RECALL: rc1 memo (§the SM3R race: what the adaptive tree coder conditions on, the 36,130 B raw body vs 30,856 → 30,246 B container figures — two bases, m99), rc2/rc3 memos and code (`experiments/ddm_rc2_hpac_semistatic_mixing_codec.py`, `experiments/ddm_rc3_shared_mixer_codec.py` — the counted mixer form + exact pack/stage/encode path), tc1's bound instrument (`experiments/ddm_tc1_token_tail_bound.py`, `ddm_tc1_shared_predictor_bound.py`), the semantic packer/unpacker on the SHIPPED tree, fe1's lottery law (container deltas are one-sample draws, sd 34.8 B — sample seg-neutral variants, ship the smallest), the live pointer (read `.omx/state/canonical_frontier_pointer.json` at run time; READ ONLY). `tools/subagent_checkpoint.py read --subagent-id ddm_sm1` first.
2. Bound leg (closed-form, on the exact shipped codes): per tensor, the realized bits under rc1's coder (instrument its probabilities read-only) vs Miller–Madow conditional entropies given candidate contexts the coder does not condition on (row position / row scale bucket / previous row's same column / sign of the left neighbour / dw↔pw co-location / depth of the block); table: tensor · context · bound bits saved · counted parameter bytes · net. Report the ceiling honestly.
3. If bound net ≥ 150 B: build ≤ 2 counted shared-mixer families (shared weights ≤ 64, int8/fp16 counted) on top of rc1's predictor; race: pack → stage → encode ×2 (twin byte-identical) → receiver decode returns BIT-IDENTICAL weights (every tensor; `frame_embed` too) → census (only the semantic section + reader moved; hpac, carrier, tail byte-identical) → container sweep with 3–5 seg-neutral variants sampled (lottery law), ship the smallest. If < 150 B: close the semantic-coder door with the bound table; build nothing.
4. If net bytes < the pointer: stage on the live tree, public-path decode identity, public-smoke block via `tac.candidate_seal._public_smoke_problems` (frontier = live pointer), `tools/make_candidate_seal.py` contest-CUDA, name `ddm_sm1_semantic_shared_mixer` (no `v<digit>`); "SEAL READY". MAIN fires.
5. Memo `.omx/research/ddm_sm1_semantic_section_shared_mixer_coder_priced_closed_form_20260909.md`; equations leg: anchor on `model_section_adaptive_recode_ceiling_v1` via `update_equation_with_empirical_anchor` (`tac.canonical_equations`) or register `semantic_section_context_mixing_bound_v1`.

## HARD CONSTRAINTS

- `upstream/` READ-ONLY. NO Modal fire. No scorer (zero distortion — PROVE by bit-identical decoded weights + census; the renderer's output is then identical by construction — state it, do not re-render to "check"). Never write into `submissions/semantic_joint_ctxmix/`, the live pointer tree, or the sj1/fe1/rw1/cmp1 trees; your tree `/Volumes/VertigoDataTier/pact/ddm_sm1_semantic_shared_mixer/` (`df -h` first; small).
- COMPOSITION: cmp1 (hpac + tail coder) is sealing on the same pointer and fe1 (one FiLM code in the SEMANTIC section) re-bases after it — coordinate through the pointer: re-read before staging/sealing; if fe1's row promotes, its one changed code is part of the bytes you re-code (a re-base for you is a re-encode). You touch ONLY the semantic section + its reader.
- The local SCORER LANE belongs to MAIN, always. Do NOT write who holds it into a charter (the #1210 stale-precondition genus, memo ddm_bz2_bornsmall_capacity_ceiling 2026-08-29).
- CPU ≤ 2 procs. Detached only via `tools/launch_detached_process.py … --nice 10 --nice-best-effort`; kill the process group on timeout; no `nohup`/`&`/clock waiters; artifact-bound waits ≤ 780 s.
- Serializer commits w/ post-edit `--expected-content-sha256`; `.py` = 2 review passes. Tokens `[no-triality] [p0-ledger-ok]`. NEVER a Co-Authored-By or AI-attribution trailer. Sandbox git refusal → serializer fallback bundle + receipt, named in the final message.
- ALWAYS KEEP THE PAYLOAD; VERIFIED-AT-SOURCE for 30,246 / 36,130 / 30,856 B and the tensor shapes; CLOSED-FORM-FIRST is the gate (no coder run before the bound table); m99: name which basis (container bytes vs raw body) every number is in.
- Checkpoint discipline: `tools/subagent_checkpoint.py --subagent-id ddm_sm1 …` every ~10 tool uses.

## PRIOR NEGATIVE SIGNAL (bearing dead-ends this charter consumes)

- rc1's −610 B on the SM3R body is already banked (move 27); rc1 §7: a generic byte coder loses boundary structure when the code width changes — width boundaries are contexts.
- Per-cell / semi-static tables LOSE (rc2: −1,210 B vs rc1; rc1 §6: five conditioned designs converted 10.3 B) — only SHARED parameters have paid; parameter bytes count at the archive (rc2: 16-weight lost to 8-weight by 1 B).
- Container retuning alone: 0 B; container deltas of an edited payload are a one-sample lottery (fe1) — sample variants, never search.
- Capacity (λ) closed both ways (cl2/cl3); coder strength substitutes for capacity — do not touch the renderer's weights or capacity; only their CODING.
- Reordering pays IFF the coder has no context model; rc1's coder has one — no permutation designs unless the bound says a specific order IS the context.

## OPTIMAL FORM

- Family exemplar: rc3's landing (`35c631244`; seal 181,213 B) and rc1's SM3R race (move 27) — the reference form: counted shared mixer over an adaptive predictor, exact encode×2, bit-identical decode, census, sampled container, seal with the public-smoke pair.
- SCOPE reductions declared per row: ≤ 2 families built (SCOPE, chosen by the bound table); the context-candidate set is the six named (SCOPE). MECHANISM reductions FORBIDDEN: an average-bits bound; uncounted mixer weights; library-path-only decode identity; a subset of tensors in the bound presented as the section's.
- **PRIOR-LAW PREDICTION (falsifiable):** the bound finds 1.5–3 % of the section (450–900 B) reachable by row-scale / co-location contexts rc1's coder ignores; a ≤ 32-weight shared mixer converts 40–60 % of it (−200…−500 B net). FALSIFIER: bound net < 150 B — the section is at its context ceiling under rc1's coder; close with the table; build nothing.

## DELIVERABLE

The memo with the bound table (and the race if built), retained payload, the seal if admitted. Commit via the serializer. End with the live frontier line.
