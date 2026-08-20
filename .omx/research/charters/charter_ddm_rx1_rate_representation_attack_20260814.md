# CHARTER — ddm_rx1_rate_representation_attack (2026-08-14, operator "Not lowering fast enough")

OPERATOR VELOCITY MANDATE (08-14): "Not lowering fast enough." THE GOAL's
consequence 1 binds: bias every decision toward the action most likely to LAND
A LOWER EXACT SCORE soonest. This charter is that action by ARITHMETIC:

**S = 0.1619344579 (MC36 floor). Gap to 0.15 = −0.0119345. The RATE term is
0.1240289 — 76.6% of S. ΔS_rate = 25/37,545,489 per byte = 6.66e-7/B.
A −18,000 B cut alone = −0.0120 S = SUB-0.15 IN ONE ROW.** Pose (−0.0083 max)
and seg (js1) are slower routes; rate is the widest untried axis: #996 closed
only the CODER axis (each section vs its OWN memoryless bound under the
EXISTING prior). Changing the PRIOR/representation is open and unowned.

## THE TARGET

From the MC36 archive f0ba4bb41d55fff85542f2a17dfe682508aa4f9ab50ef51cda573d79f0c4b1de
(186,269 B), produce candidate archives with **maximal byte reduction at
≤ +0.0007 total distortion cost** (the crossing allowance at −18,000 B; scale
allowance linearly with realized cut). Rank levers by ΔS = −6.66e-7·ΔB + ΔD.

## LEVERS (measured-first order; recall each receipt before building)

1. **HPAC RETRAIN on our labels — the distortion-FREE lever (#982 fire-order,
   hb1/hb2 machinery LANDED + round-trip-FIXED).** The HPAC model is the
   entropy-coding PRIOR: a better prior recodes the SAME token stream
   losslessly → pure rate win, d_seg/d_pose byte-identical BY CONSTRUCTION
   (verify decoded-token byte-identity, that IS the gate). Net = token-stream
   savings − Δ(packed model size). tq1c stages 3-4 receipts + hb2's
   deploy-bound pack gate are the substrate. #996 does NOT cap this (it
   bounded the OLD prior's stream). Measure net ΔB on the REAL archive.
2. **HPAC/model-weight precision waterfill (pz4a lineage + quantization
   toolbox: adaptive per-cell · aware in-loop · sub-int16).** Sensitivity-
   allocated variable precision on the model section; distortion measured
   through the REAL local CPU decode (see instrument below), never assumed.
3. **fd135 section-level sweep**: per-section {re-representation, precision
   cut, derive-at-decode instead of store} candidates from the fd135
   decomposition table; price each with real coders; drop/merge anything
   decode-derivable (rule-118: generic algorithm free in inflate).
4. **Cross-section/joint recode** (headers, redundancy between pose carrier
   and token stream, container overhead) — measured, not assumed; lc2's −903 B
   lossless lever class shows this tier is real but small.

## THE INSTRUMENT (the f26p unlock makes this campaign CHEAP)

Local byte-closed iteration is now FAST: the f26p lifted CPU decode runs full
n600 in 646 s on M5 4-thread (commit a5e1f60270; lifted module + runner
experiments/ddm_f26p_f26_cpu_lift.py; work dir
/Volumes/VertigoDataTier/pact/ddm_f26p_runtime_cpu_lift_20260814/). For every
candidate: rebuild archive → local CPU decode → compare decoded tokens/raw vs
the base decode (retained CPU raw sha e5539653…, token checkpoint sha
9ba2e52b…, per-frame manifest in receipts/) → byte-identical ⇒ ZERO-distortion
rate row (no scorer needed!); else score the delta locally (CPU-torch scorers,
advisory) before any T4 spend. NO Modal — MAIN fires exact rows; your product
is candidate archives + measured ΔB + identity/advisory-distortion receipts +
a sealed fire-order ranked by projected ΔS.

## OPTIMAL FORM

PINS: archive f0ba4bb4…@186,269 B · fd135 decomposition memo · hb1/hb2
receipts + tq1c stage 3-4 · pz4a receipt · #996 coder-axis verdict (respect
its scope: do NOT re-race coders on unchanged streams) · f26p memo
ddm_f26p_runtime_cpu_lift_20260814.md. SCOPE reductions legal (lever subset
first, biggest first); MECHANISM reductions TOY-BRACKET (a projected byte
count without a rebuilt archive, or a distortion claim without a real decode,
cannot produce a row). Decode-time law binds: if a candidate slows decode,
type it decode-engineering-gated, never kill. Payload law: retain every
candidate archive + its decode receipts on Vertigo. Honest negative allowed:
if the total reachable cut measures ≪ 18,000 B, report the measured ceiling
and its mechanism — that re-ranks the forest routes, which is itself velocity.

## OUTPUT

Work dir /Volumes/VertigoDataTier/pact/ddm_rx1_rate_attack_20260814/. Memo
.omx/research/ddm_rx1_rate_representation_attack_20260814.md: per-lever
measured ΔB + distortion receipt + composed best candidate (sha, bytes,
projected S) + sealed T4 fire-order for MAIN. Commit via
tools/subagent_commit_serializer.py (post-edit shas, [no-triality]
[p0-ledger-ok], no co-author trailer; .py via review_tracker mark-file).
NEXT_IF_RESUMED + LIVE-HYPOTHESES + DEAD-ENDS. Git-blocked ⇒ memo SHA handoff.
