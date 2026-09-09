# ddm_cmp1 — COMPOSE two sealed zero-distortion rate rows onto the LIVE pointer: rc3's hpac model section (−201 B) + tc1's token-tail shared-mixer coder (−549 B) → one candidate, one seal, one T4 call (charter, 2026-09-09)

Tokens: `[no-triality] [p0-ledger-ok]` · Owner: codex arm (astra high) · Spawned by MAIN after sj1's pass-4 row is decided (the charter reads the pointer at run time). Sources: rc3 seal `/Volumes/VertigoDataTier/pact/ddm_rc3_shared_mixer_successor/SEAL_ddm_rc3_shared_mixer_successor_contest_cuda.json` (memo `.omx/research/ddm_rc3_shared_mixer_successor_closed_form_gated_20260909.md`, landed 35c631244) and tc1 seal `/Volumes/VertigoDataTier/pact/ddm_tc1_tail_shared_mixer/rebase_pc2/SEAL_ddm_tc1_tail_shared_mixer.json` (memo `.omx/research/ddm_tc1_token_tail_bound_and_shared_mixer_pricing_20260909.md`, landed faac73963). Axes: bytes exact, zero distortion; `score_claim=false`; MAIN fires.

## MANDATE

Two arms sealed independent zero-distortion rate moves against the move-34 pointer: rc3 re-codes the IHS1 model rows (hpac section + reader, −201 B; semantic/carrier/tail byte-identical) and tc1 re-codes the token tail with a 35-weight shared mixer over HPAC (tail section + reader, −549 B; field byte-identical). They touch disjoint sections and disjoint reader code paths. Firing them separately costs two T4 calls and two re-bases; composing them is exact byte arithmetic plus the identity proofs. Compose both onto whatever pointer is LIVE when you run (re-read `.omx/state/canonical_frontier_pointer.json` first: if sj1's pass-4 row promoted, the pointer's TAIL is a new field — tc1's coder is field-agnostic and must RE-ENCODE that tail; rc3's section is field-independent).

## SCOPE

1. RECALL: both memos and seals; rc3's stage/race scripts (`experiments/ddm_rc3_shared_mixer_stage.py` etc.); tc1's codec/pricing/proof scripts (`experiments/ddm_tc1_mixer_codec.py`, `ddm_tc1_public_proof.py`, `ddm_tc1_receiver_checkpoint.py`); the live pointer tree (READ ONLY). `tools/subagent_checkpoint.py read --subagent-id ddm_cmp1` first. Read the pointer FIRST and name the base (lane, sha, bytes) in your memo.
2. Identity controls on the live base: rebuild the pointer's archive byte-identically from its own sections (null build).
3. Compose: (a) rc3's model section + its reader onto the base — decoded IHS1 rows bit-identical to the base's (rc3's own identity check); (b) tc1's tail coder: re-encode the BASE's tail field (whatever field the pointer ships) with the mixer, twin byte-identical, receiver decode returns the EXACT field; (c) census: only hpac + tail sections (and their readers) moved; semantic and carrier byte-identical to the base. Container sweep on the composed archive (fe1's law; ties to the shipped shape by byte identity). Report the composed bytes vs the base and vs the sum of the two separate savings (they need not add exactly).
4. Public-path decode identity through `bash inflate.sh`'s preamble + `f26_inflate` reach (the smoke PAIR, both roles, frontier = the live pointer) checked with `tac.candidate_seal._public_smoke_problems`; `tools/make_candidate_seal.py` contest-CUDA; name `ddm_cmp1_rc3_tc1_composed` (no `v<digit>`); "SEAL READY: <path>" with bytes, sha, and the two-row decomposition. MAIN fires.
5. Memo `.omx/research/ddm_cmp1_compose_rc3_tc1_20260909.md`; equations leg: anchors on `model_section_adaptive_recode_ceiling_v1` and `token_tail_context_mixing_bound_v1` via `update_equation_with_empirical_anchor` (`tac.canonical_equations`) with the composed bytes.

## HARD CONSTRAINTS

- `upstream/` READ-ONLY. NO Modal fire. No scorer (zero distortion — PROVE by exact field decode + bit-identical model rows + census). Never write into `submissions/semantic_joint_ctxmix/`, the live pointer tree, or the sj1/fe1/rc3/tc1 trees (read their retained artifacts); your tree `/Volumes/VertigoDataTier/pact/ddm_cmp1_compose/` — Vertigo is nearly FULL (~4 GiB): keep it under 1 GiB; anything larger to `/Volumes/APDataStore/pact/ddm_cmp1_compose/` (ExFAT payload blobs only); `df -h` before every write.
- COMPOSITION: fe1 (semantic FiLM codes) will re-base onto your row if it promotes; you touch ONLY hpac + tail sections and their readers.
- The local SCORER LANE belongs to MAIN, always. Do NOT write who holds it into a charter (the #1210 stale-precondition genus, memo ddm_bz2_bornsmall_capacity_ceiling 2026-08-29).
- CPU ≤ 2 procs. Detached only via `tools/launch_detached_process.py … --nice 10 --nice-best-effort`; kill the process group on timeout; no `nohup`/`&`/clock waiters; artifact-bound waits ≤ 780 s.
- Serializer commits w/ post-edit `--expected-content-sha256`; `.py` = 2 review passes. Tokens `[no-triality] [p0-ledger-ok]`. NEVER a Co-Authored-By or AI-attribution trailer. Sandbox git refusal → serializer fallback bundle + receipt, named in the final message.
- ALWAYS KEEP THE PAYLOAD; VERIFIED-AT-SOURCE for 201 / 549 / the base's section bytes; CLOSED-FORM-FIRST: state the expected composed bytes before encoding.
- Checkpoint discipline: `tools/subagent_checkpoint.py --subagent-id ddm_cmp1 …` every ~10 tool uses.

## PRIOR NEGATIVE SIGNAL (bearing dead-ends this charter consumes)

- A length-only container tie-break ships a DIFFERENT stream of the same length (fe1: two windows → 30,246 B both, 30,129 bytes differ) — pin the shape by byte identity.
- A check that encodes a round-one premise keeps passing after the premise moves (sj1's silent-revert classes; gs3 Addendum 14) — every identity control here is against the CURRENT pointer's bytes, re-read at seal time.
- tc1: dense HPAC and reconstructed null rows failed exact parity — keep the sparse HPAC and the original-row bypass; Brotli/DEFLATE containers were larger than raw RC64 with ZIP STORE for the tail.
- rc3: parameter bytes are counted at the archive, not at J (the 16-weight mixer lost by 1 B); ck2=false variants cannot ship through the current reader grammar.

## OPTIMAL FORM

- Family exemplar: rc2's move-33 landing (memo `.omx/research/ddm_rc2_t4_hpac_semistatic_mixing_20260909_pointer_move_33_20260909.md`) and pc2's move 34 (same-day re-base onto a moved pointer, `64cf42609`) — the reference form: section swap, null-build identity, exact encode×2, census, seal with the public-smoke pair; projection = exact row to the last digit for zero-distortion custody rows.
- SCOPE reductions declared per row: none (both inputs are complete sealed rows). MECHANISM reductions FORBIDDEN: no library-path-only decode identity; no assumed container shape; no assumed additivity — measure the composed bytes.
- **PRIOR-LAW PREDICTION (falsifiable):** composed saving = 201 + 549 = 750 B ± 20 B (the sections are disjoint and each at its own container optimum; the archive-level brotli may interact by tens of bytes); distortion identical by construction. FALSIFIER: composed saving < 700 B (an interaction at the container) — report both rows and the interaction; still seal if net < the pointer.

## DELIVERABLE

The memo, the composed candidate tree + seal, retained payload. Commit via the serializer. End with the live frontier line.
