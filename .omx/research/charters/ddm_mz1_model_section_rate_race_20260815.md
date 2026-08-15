# ddm_mz1_model_section_rate_race — close the 52,566 B model-serialization gap (the sub-0.15 rung)

## THE OBJECT (all receipts SHA-pinned in FINAL_RESULT.json, schema ddm_rx2_final.v1)
The e480b identity-race winner archive (183,502 B, sha e3e6f440b45bbb92f2eeb58c7a56d74b3cd0a62bbcff01a26adcd008391c19d3,
/Volumes/APDataStore/pact/ddm_rx2_current_mc36_label_hpac/FINAL_RESULT.json) decodes token-identical
to the MC36 frontier (raw-identity PROVEN on CPU). Sections: token 112,749 · MODEL 70,557 · table 100 · residual 96.

## THE MEASURED GAP (this charter's entire reason)
Trainer endpoint (full_e480b/launcher/run.log, ep480): estimated_token_bytes 113,229 (realized 112,749 — estimate GOOD)
but estimated_model_bytes 17,991 vs REALIZED model section 70,557 B — a 3.92× serialization gap = 52,566 B = 0.0350 S.
Projected S now 0.16009202615715576. **Sub-0.15 needs only ≥15,153 B of lossless model-section savings**
(0.16009 − 0.15 = 0.01009 S → 15,153 B at 25/37,545,489 S/B). Full closure → ~0.1251.

## PRIOR-LAW PREDICTION (m38): the trainer's 17,991 B is a per-tensor entropy estimate; xz-of-serialized-tensors
loses vs per-tensor adaptive coding. Predict ≥25,000 B recoverable (≥47% of gap) via (b)+(c) below.

## FACTS ON THE PACK (FINAL_RESULT.pack): IHS1, quantized_tensor_count 28, state_tensor_count 37,
training_only_bit_depth_buffers 9, weight/activation bound 127 (int8-range), xz container,
verified_exact idempotent, decode_logit_diff 0.0. Raw checkpoint 1,099,767 B → 70,557 B (15.6×, naive).
The self-compress round-trip machinery was FIXED by hb2 (#988) — recall it, do not rebuild.

## RUNGS (in order; each = a measured row, keep every payload P0)
(a) SECTION AUTOPSY: decompose the 70,557 B — per-tensor bytes, quantized planes vs fp32 buffers vs
    indices/shape metadata vs xz overhead. Are the 9 training-only bit-depth buffers shipped? Strip-derive them.
(b) REAL CODER RACE on the quantized weight planes: PR130-lineage self-compress (hb2-fixed) ·
    per-tensor adaptive/context coder · brotli-q11 · lzma-raw · rc64 — race, never adopt by citation.
    Per-tensor byte-maps/perms (L14–L32 = intake intelligence: RACE as candidates only).
(c) DERIVE-don't-ship: any buffer reconstructible from config/seed at decode = rule-118 FREE receiver code.
(d) REBUILD archive (winner variant s1p25_c1p0) → identity race cpu-decode stage (the existing verifier,
    experiments/ddm_rx2_mc36_identity_race.py) → decoded-token identity MANDATORY, repeat-build determinism.
(e) Emit candidate + fire-order for MAIN's T4 dispatch (canonical tools/dispatch_modal_paired_auth_eval.py).

## HARD CONSTRAINTS
- LOSSLESS on decoded tokens (identity race = the gate). Any lossy variant is a DIFFERENT charter — refuse.
- ALWAYS KEEP THE PAYLOAD (P0 DEF CON 1000): persist every recoded section + sha256, SSD tier
  /Volumes/VertigoDataTier/pact/ddm_mz1_model_section_rate_race/.
- No score claims: byte deltas are exact; S projections labeled projected-not-authority.
- Serializer commits, post-edit working-tree SHAs, [no-triality] [p0-ledger-ok]; memo
  .omx/research/ddm_mz1_model_section_rate_race_20260815.md.
## OPTIMAL FORM
Reference form = the family's real coders already in-tree (hb2 self-compress, rc64, SMEVR/r7 race harness) on the
REAL 70,557 B section — no synthetic fixtures, no subset tensors. Scope reduction legal: rung (b) may race on the
3 largest tensors first, but the VERDICT row requires the full section. Provenance pins: FINAL_RESULT.json +
launcher/run.log paths above.
Provenance pins: winner archive sha256 e3e6f440b45bbb92f2eeb58c7a56d74b3cd0a62bbcff01a26adcd008391c19d3;
race receipts committed at git 82aaa5cf15 (wc2 §5h); T4 authority row S 0.1600920261571558
(call fc-01M02QMN3SQ9SNHXZMRWXYEJEW, custody dir experiments/results/modal_auth_eval/
ddm_rx2_e480b_hpac_winner_v2_paired_modal_auth_20260815T125117Z_cuda/).
