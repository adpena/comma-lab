# CHARTER — ddm_f26q_rc64_native_lowering (2026-08-14, Route B decode-engineering cure)

PARENT: ddm_f26p_runtime_cpu_lift_20260814.md + the Modal CPU exact-row attempt
(call fc-01M00WRPPSEE6HVT3Y5FFSTM37, CLOSED_REMOTE_FAILED on the 1,800 s
contest budget). READ THE PARENT MEMO FIRST.

## THE MEASURED PROBLEM (all receipts on disk)

The lifted F26 CPU decode is CORRECT cross-platform — Modal x86 decoded
tokens byte-identical to the retained checkpoint (exact sha pinned under
EXACT PINS below) — but OVER BUDGET:
- Modal x86 4-thread total: 2,933.2 s vs 1,800 s budget (report captured in
  experiments/results/ddm_f26p_mc36_contest_cpu_20260814/returned_artifacts/contest_auth_eval.stdout.log).
- Hot stage: token_decode_or_checkpoint_load (sequential Python RC64) =
  2,613.9 s = 89.1% of the wall. Render = 304.1 s. Everything else ≈ 15 s.
- M5 4-thread reference: 646.4 s total, token stage ≈ 383 s — Modal x86 is
  ~6.8× slower on this stage (single-core-bound Python).
- NEEDED: total ≤ ~1,700 s w/ margin ⇒ token stage ≤ ~1,380 s ⇒ ≥1.9×
  speedup on the RC64 stage on Modal-class x86. Target harder: 5-10×.

EXACT PINS (byte-identity gates — the ONLY admissible correctness proof):
- decoded_token_sha256 = 9ba2e52b3096585895970066b389bf1261ebc203d5b828cdea056c13858aea52
  (117,964,800 tokens; matches the retained checkpoint AND the Modal x86 run
  — platform-independent, so it is THE parity oracle).
- decoder_bit_position = 921964 · token_codec = rc64 ·
  corrected_cdf_input_sha256 ba0d529b… · corrected_quantized_logit_sha256 617e9fcf…
- Archive f0ba4bb41d55fff85542f2a17dfe682508aa4f9ab50ef51cda573d79f0c4b1de @186,269 B.
- Lifted runtime + runner: experiments/ddm_f26p_f26_cpu_lift.py (commit
  a5e1f6027018f001975619f1aff187c75777fc52); work dir
  /Volumes/VertigoDataTier/pact/ddm_f26p_runtime_cpu_lift_20260814/.

## THE TASK — lower the RC64 token-decode stage to native (Rust preferred,
C acceptable), bit-identical, under the full native grant
([[full_native_lowering_and_optimization_grant_20260810]]; runtime-rs
bit-parity precedent #58/#283; #214 took a prior inflate under budget).

STAGES (optimal form, no toys):
1. PROFILE the Python RC64 inner loop at element level (per-op counts on the
   real stream — not a synthetic stream; SCOPE reduction = a frame subset of
   the REAL archive, extrapolated honestly, full-field before any verdict).
2. LOWER: native RC64 decoder consuming the SAME corrected-CDF inputs,
   producing the SAME token bytes. Byte-identity gate: sha256 of the full
   decoded token field == 9ba2e52b… on the full n600 field, plus
   decoder_bit_position == 921964. A vectorized-numpy rung may be raced
   first if profiling shows it reaches ≥2× (cheaper than native), but the
   native rung is the deliverable unless numpy alone clears ≥3×.
3. MEASURE wall-clock on M5 (report M5 factor + projected Modal x86 via the
   measured 6.8× stage ratio — label the projection DERIVED, the M5 number
   MEASURED). Integrate into the lifted runner behind a flag defaulting to
   the proven path; the native path activates only after byte-identity.
4. SEALED FIRE-ORDER for MAIN: the re-fire of the contest-CPU exact row
   (same canonical modal_auth_eval_cpu chain, same archive) once projected
   Modal total ≤ ~1,600 s. Include raw-payload retention via the volume
   (the failed run's 3.6 GB raw was container-ephemeral — payload law says
   the re-fire must persist raw or its per-frame manifest to the volume).

## OPTIMAL FORM
PINS: lifted runtime commit a5e1f6027018f001975619f1aff187c75777fc52 ·
archive sha256 f0ba4bb41d55fff85542f2a17dfe682508aa4f9ab50ef51cda573d79f0c4b1de ·
token-parity oracle sha256 9ba2e52b3096585895970066b389bf1261ebc203d5b828cdea056c13858aea52 ·
parent memo ddm_f26p_runtime_cpu_lift_20260814.md.
Reference form = the live Python RC64 (the semantics oracle). MECHANISM
reductions (approximate CDF, lossy shortcut, different rounding) are
FORBIDDEN — bit-identity or nothing. Decode-time law binds: report typed
speedup receipts, never kill. Payload law: retain every decoded token field
sha + any native-decoder binary + build manifest (deterministic build,
rebuild_instructions per the native-runtime discipline). Rust in runtime-rs/
with python_reference_equivalence_test per the payload-cleanliness bundle.
Git-blocked ⇒ memo SHA handoff.

## OUTPUT
Work dir /Volumes/VertigoDataTier/pact/ddm_f26q_rc64_native_20260814/.
Memo .omx/research/ddm_f26q_rc64_native_lowering_20260814.md: profile table ·
parity receipt (sha match) · measured M5 + projected Modal wall · sealed
fire-order or typed decode-engineering-gated residual. Serializer commit,
[no-triality] [p0-ledger-ok], no co-author trailer.
NEXT_IF_RESUMED + LIVE-HYPOTHESES + DEAD-ENDS.
