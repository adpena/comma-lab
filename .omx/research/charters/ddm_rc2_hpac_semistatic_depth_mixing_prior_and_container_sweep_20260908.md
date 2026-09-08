# ddm_rc2 — price a semi-static / depth-mixing prior for the IHS1 model rows (≈ 1,139 B of measured unconverted structure) + re-run the container argmax under the RC1 rider (charter, 2026-09-08)

Tokens: `[no-triality] [p0-ledger-ok]` · Owner: codex arm · Spawned by MAIN 2026-09-08 under the operator's standing GO. Source: rc1's owed ITEM 1 + ITEM 3 in `.omx/research/ddm_rc1_adaptive_recode_race_of_the_model_sections_20260905.md` (landing `8979e18aa`). Axes: bytes exact, zero distortion (the field is untouched); `score_claim=false` until a T4 row; MAIN fires.

## MANDATE

Operator 20260908: *"recover and respawn and continue with all, also codex is back"*. rc1's adaptive per-group tree coder took −1,123 B off the IHS1 HPAC model rows (and −610 B off the renderer body), zero distortion, and moved the pointer (move 27). It left a MEASURED gap: the Miller–Madow order-1 bound on the IHS1 rows is ≈ 8,469 B against the 9,639 B the adaptive tree realized — **≈ 1,139 B of real structure**, of which rc1's five conditioned designs converted 10.3 B (0.9%). Two forms are untested: (a) a two-pass SEMI-STATIC conditional table whose counted table bytes are paid explicitly and traded against the gain; (b) a LOGISTIC MIXTURE sharing one weight vector across depths so each cell is not learned alone. Both are pure local byte measurements on the retained body. ITEM 3: the rider changed what Brotli sees, so the container argmax over `(ck2, quality, lgwin)` may have moved (tens of bytes expected). Any net byte win at zero distortion is a pointer move by arithmetic (−1e-3 S ≈ 1,502 B).

## SCOPE

1. RECALL first: the rc1 memo (whole; §6 the five conditioned designs and WHY each converted ≈ 0; §7 the width-change blind spot), `experiments/ddm_rc1_*.py` (coder, group tree, pack/stage/encode path), the live pointer's model section (`/Volumes/VertigoDataTier/pact/ddm_sj1_multipass_token_predistortion/candidate_pass3/candidate_runtime`, archive 181,645 B sha `06c44dc4…`; the hpac model section is byte-identical to rc1's — verify by census before assuming). `tools/subagent_checkpoint.py read --subagent-id ddm_rc2` first.
2. Re-derive the bound on THIS tree's rows: order-0, order-1 Miller–Madow, and the realized adaptive-tree bytes; state the gap you are chasing with its derivation (CLOSED-FORM-FIRST).
3. Design (a): two-pass semi-static conditional coding — pass 1 counts the conditional histogram per context (contexts = the group tree's conditioning set, or better ones you derive from the row statistics), pass 2 codes with the frozen table; the TABLE is counted in the archive (serialize it compactly: sparse, Golomb/Elias-coded counts, or a parametric family with a few parameters — price each). Report net J = table bytes + coded bytes vs rc1's 9,639 B.
4. Design (b): logistic mixing across depths — one shared weight vector (sizes 8–64 weights, fp16/int8 quantized and COUNTED), adaptive per-symbol mixing of the depth predictors; report net bytes incl. the weights. Also test (a)+(b) if both help.
5. Race the winner through the shipped container path: pack → stage → encode ×2 (twin byte-identical) → receiver decode returns the EXACT model rows (bit-identical weights; the field and every other section byte-identical by census) → exact archive bytes.
6. ITEM 3: sweep `(ck2, quality ∈ {9,10,11}, lgwin ∈ {22,23,24})` on the winner and on the base; report the argmax and its bytes.
7. If net bytes < the pointer's: stage on the LIVE tree (model section + reader only), decode identity through the PUBLIC path, full CPU `f26_inflate` reaches token decode + `bash inflate.sh` to the CUDA gate (the seal smoke PAIR — required), `tools/make_candidate_seal.py` contest-CUDA vs 181,645 B / 0.13900437796841966; candidate name `ddm_rc2_hpac_semistatic_mixing` (NO `v<digit>` tokens). Zero distortion → the T4 row is a custody replay of the same field; MAIN fires. Report "SEAL READY: <path>" with bytes and sha.
8. Memo `.omx/research/ddm_rc2_hpac_semistatic_depth_mixing_prior_and_container_sweep_20260908.md`: bound derivation, design table (table/weights bytes, coded bytes, net J), container sweep, what was NOT done, equations leg (anchor on `model_section_adaptive_recode_ceiling_v1` and `coder_strength_substitutes_for_capacity_v1` via `update_equation_with_empirical_anchor`, `tac.canonical_equations`), frontier line last.

## HARD CONSTRAINTS

- `upstream/` READ-ONLY. NO Modal fire. Never write into `submissions/semantic_joint_ctxmix/` or sj1's tree (READ-ONLY for you); your tree: `/Volumes/VertigoDataTier/pact/ddm_rc2_hpac_semistatic_mixing/` (small: model rows are ~12 KB; check `df -h`; nothing bulky).
- COMPOSITION: sj1 (token tail + carrier) and pc2 (carrier format) are live on the same pointer; you touch ONLY the hpac model section + its reader. Whoever lands second re-bases; re-read `.omx/state/canonical_frontier_pointer.json` before staging/sealing. Do not edit `experiments/ddm_sj1_*` or `experiments/ddm_pc*`.
- The local SCORER LANE belongs to MAIN, always. Do NOT write who holds it into a charter (the #1210 stale-precondition genus, memo ddm_bz2_bornsmall_capacity_ceiling 2026-08-29). No scorer runs here (zero distortion by construction — PROVE it by census + bit-identical weights, never assume).
- CPU: ≤ 2 processes; no Metal/MPS (design (b) trains its mixing weights online per symbol — CPU). Detached only via `tools/launch_detached_process.py … --nice 10 --nice-best-effort`; no `nohup`/`&`/clock waiters; artifact-bound waits ≤ 780 s.
- Serializer commits w/ post-edit `--expected-content-sha256`; `.py` = 2 genuine review passes. Tokens `[no-triality] [p0-ledger-ok]`. NEVER a Co-Authored-By or AI-attribution trailer.
- ALWAYS KEEP THE PAYLOAD (tables, weights, coded streams, archives; sha256 + bytes). VERIFIED-AT-SOURCE LAW for 8,469 / 9,639 / 1,139 / 10.3 B — re-measure on THIS tree. PER-PAIR RECEIPTS: n/a (zero distortion) — state so with the census as the receipt. EQUATIONS-LEG LAW as in SCOPE 8; run the Catalog #344 check before the final message.
- Checkpoint discipline: `tools/subagent_checkpoint.py --subagent-id ddm_rc2 …` every ~10 tool uses.

## PRIOR NEGATIVE SIGNAL (bearing dead-ends this charter consumes)

- rc1's five CONDITIONED designs converted 10.3 B of the 1,139 B (0.9%) — `.omx/research/ddm_rc1_adaptive_recode_race_of_the_model_sections_20260905.md` §6. Read WHY (adaptive per-cell learning dilutes across depths) before designing; do not re-run those five.
- The generic byte coder loses boundary structure when the code width changes (§7) — width boundaries are contexts, not noise.
- Capacity axis CLOSED both directions (cl3: λ 1.0→2.0 +224 B; cl2: 1.0→0.5 +506 B) and a stronger coder returns LESS for the same weight shrink (`coder_strength_substitutes_for_capacity_v1`) — do not touch the model's capacity; only its CODING.
- Reordering pays IFF the coder has no context model (memory `reordering-pays-iff-the-coder-has-no-context-model`) — a permutation pass on top of a context coder is not a design here.

## OPTIMAL FORM

- Family exemplar: rc1's adaptive per-group tree coder landing (`8979e18aa`; pointer move 27 memo `.omx/research/*rc1_t4_model_section_adaptive_recode*pointer_move_27*`) — the reference form: exact pack/stage/encode×2, receiver decode identity, section census, zero-distortion custody row.
- SCOPE reductions declared per row: table-family search may be pruned by the closed-form bound (SCOPE). MECHANISM reductions FORBIDDEN: a table or weight vector that is not COUNTED in the archive is the hide-data-in-code fake; a decode identity through the library path only is not identity.
- **PRIOR-LAW PREDICTION (falsifiable):** a semi-static table converts 300–600 B of the 1,139 B net of its own counted bytes (order-1 tables on ~12 KB of rows are dense enough to pay); logistic depth-mixing converts a further 100–300 B; the container argmax moves by ≤ 60 B. FALSIFIER: neither design nets ≥ 150 B (the gap is table-cost-bound at this row count) — count it plainly and close the model-prior door at formulation scope.

## DELIVERABLE

The memo above + retained payload + the seal if admitted. Commit via the serializer. End with `sj1 S 0.13900437796841966 @ 181,645 B [contest-CUDA T4 n600]` (or the live row).
