# ddm_rc1 — adaptive (CABAC / context-model) lossless recode race of the two MODEL sections through the shipped container (charter, 2026-09-05)

Tokens: `[no-triality] [p0-ledger-ok]` · Owner: Opus arm · Spawned by MAIN 2026-09-05 ~18:00Z under the operator's standing GO. Axis: `[exact local
byte arithmetic, scorer-free]`; ZERO distortion by construction (lossless); `score_claim=false` until a T4 row confirms custody.

## The object (recall — state the BASIS with every byte count)
Frontier archive (cl2 repack, 179,982 B, S 0.14781744131049854). Container `RX1` (`runtime/residual_archive.py` in the receiver copy at
`/Volumes/VertigoDataTier/pact/ddm_cl2_hpac_prior_capacity_ladder/rungs/lambda_1p0/retained/receiver_copy_runtime/`): sections hpac (IHS1 integer
probability model; raw → XZ or Brotli → **13,466 B** container bytes), carrier (Rice payload + basis; ck2 2-plane de-interleave; **22,010 B**), semantic
(SM3R v1 MODE_ROW_PRUNE_MIXED body: width 96, 4 blocks, frame_dim 8, **66,339 params**, 15,363 zeros (23.16%), per-tensor depths {3,4}, fp16 scales,
prune masks; RAW body **36,130 B** → ck2 2-plane + Brotli q11 → **≈30,856 B** container bytes), token tail (RC64, 113,419 B — NOT in scope), framing 217 B.
Every coding applied to the two MODEL sections today is GENERIC (Brotli q0..q11 race, XZ, ck2's parameter-free 2-plane byte de-interleave — ck2 memo:
plane count k=2 −613 B, k=4 −330, k=8 −37). No arm has ever coded the unpacked 3/4-bit weight codes or the IHS1 integers with an ADAPTIVE per-tensor
context model, although the runtime already carries a CABAC (`runtime/dx2_cabac_coefficients.py`, used for carrier coefficients — dx1 measured the
carrier's own recode ceiling at −18 B, so the CARRIER is out of scope). Corpus grep (semantic/hpac × cabac/adaptive/arithmetic) returns nothing.

## PRIOR-LAW PREDICTION (m38)
- **SM3R body:** unpack the packed 3/4-bit signed codes per tensor; code them with an adaptive binary arithmetic coder under contexts
  {tensor id, |code| of the left neighbour (or previous row same column), magnitude bucket of the previous code}; code the prune masks and the fp16 scales
  separately (masks: run-length + adaptive bit; scales: fp16 high/low byte planes, adaptive). Quantized trained weights concentrate near zero: predicted
  entropy 2.4–2.9 bits/param vs Brotli's realized 3.72 → **container 26,500–28,800 B = −2,050…−4,350 B**.
- **IHS1 body:** the same per-tensor adaptive coder over its integer weights vs the shipped Brotli/XZ: **−600…−1,300 B**.
- **Total −2,650…−5,650 B at zero distortion = −0.0018…−0.0038 S**, admit iff total ≤ −300 B (10× the container-transform noise ck2 measured).
- **FALSIFIER:** best adaptive total within −300 B of the shipped container bytes → Brotli already sits at the empirical entropy of these bodies →
  CLOSE at family scope (adaptive recode of model sections). Write measured bytes beside each line; register `model_section_adaptive_recode_ceiling_v1`.

## What to do
A. RECALL: ck2 memo (`ddm_ck2_container_plane2_eleventh_move_20260819.md`), dx1 (`ddm_dx1_dxi_recode_and_fruit_sweep_20260820.md` — the CABAC precedent
   and its −18 B ceiling on the carrier), the receiver's `runtime/residual_archive.py` (section split, codec flags, ck2 uninterleave), `cpr1/ddm_mp2_semantic_receiver.py`
   (SM3R parse: masks, fp16 scales, `_unpack_signed_bits`), `cpr1/integer_model_io.py` (IHS1), `runtime/dx2_cabac_coefficients.py`. `tools/subagent_checkpoint.py read
   --subagent-id ddm_rc1` first.
B. Parse the frontier archive; extract both raw bodies; verify byte-identical re-pack through the existing packers (identity first). Report the empirical
   zeroth-order and first-order entropies of the unpacked codes per tensor (the bound the adaptive coder must approach).
C. Race: (i) baselines on the raw bodies — Brotli q11 lgwin 24 (shipped), xz -9e, zstd --ultra -22 (measurement only; a new dependency would need the rule-118
   bootstrap proof — prefer our own coder); (ii) per-tensor adaptive CABAC over unpacked codes with the context set above; (iii) the same with a small
   learned static table per tensor (counted bytes) vs adaptive-only. Exact bytes for each; decode with a fresh decoder and assert byte-identical raw bodies.
D. Winner → a NEW section codec flag in `residual_archive.py` (additive: old flags decode as before), the encoder in the shipped container path, the receiver
   copy updated in a staged tree (the cl2 pricer's `stage` step is the sister case) → receiver decode identity (field, render bytes) → inflate wall-clock →
   twin (a second encode is byte-identical) → `tools/make_candidate_seal.py` contest-CUDA. **Never dispatch Modal; MAIN fires.**
E. Memo `.omx/research/ddm_rc1_adaptive_recode_race_of_the_model_sections_20260905.md` (entropy table · race table · predictions vs measured · verdict_scope ·
   frontier line last) · law registered · lane `lane_ddm_rc1_model_section_adaptive_recode_20260905` · owed items as `## ITEM n — …` registered with
   `tools/extract_canonical_tasks_from_directive.py --directive <memo> --register-all --owner ddm_rc1`.

## OPTIMAL FORM
Reference form = the shipped container path + a real adaptive arithmetic coder decoded by the receiver itself (pure Python is fine: ≤ 80K symbols). A
"saving" measured with a coder the receiver cannot run, or on a body that does not re-pack to identity, is a TOY: refuse.

## Compute, disk, resumability (binding)
CPU only, light (≤ 4 threads). Trees on APFS (`/Volumes/VertigoDataTier/pact/ddm_rc1_model_section_adaptive_recode/` or `experiments/results/`); never on
APDataStore (ExFAT). Keep every payload (raw bodies, every coded body, the candidate archive; sha256 + bytes). `tools/subagent_checkpoint.py` every ~10 tool
uses. Commits ONLY via the serializer (`[no-triality] [p0-ledger-ok]`, post-edit shas); `.py` two review passes; NO co-author trailer; no `/tmp`; grep argparse
first. Read CLAUDE.md + `docs/operating_manual_craft_handoff.md`. Label every number MEASURED / DERIVED / PREDICTED with its BASIS (raw vs container). End with
`cl2 S 0.14781744131049854 @ 179,982 B [contest-CUDA T4 n600]` + any advisory candidate line.
