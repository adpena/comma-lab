# ddm_jt21 — joint 21-family re-encode: −176 B vs dx2, −23 B vs the gb1 pointer — BANKED below the pre-registered solo-fire bar

**Status:** VERDICT — **BANKED** (MAIN-executed, #1269). The marginal came in below the
pre-registered fire bar, so no seal, no C port, no T4 spend.
**verdict_scope:** INSTANCE — the dx2/gb1 pointer body's token stream, ONE joint F26 config
(19 fx5 families + `groupbin8_surprise` + `cls_groupbin8` = 21 families).
**axis:** `[macOS-CPU advisory / scorer-free EXACT byte measurement]` — `score_claim=false`
in the receipt; the byte delta is exact, the S projection is deterministic rate-only
arithmetic on a lossless (tokens_changed=0) family.
**Receipts:** `/Volumes/APDataStore/pact/ddm_gb1_groupbin8_conditioning/` —
`retained/{candidate_gb1_joint21.zip, S1_encode_gb1_joint21.json}` ·
`launch_joint21_r2/run.log` (done event) · `runtime_joint21/` (the 21-family decoder
config, content = runtime_groupbin8_surprise + one tuple line adding `cls_groupbin8`).

## 1. The measurement

The identical dx2 token stream re-encoded under the joint 21-family conditioning model,
spliced into the dx2 body (the body the encoder mirrored):

| quantity | value |
|---|---:|
| candidate | **180,192 B**, sha `ec0dd68ff241070f1c76d5d0da4d8a89b33039bcf56528729a791ec9fd66aef3` |
| vs dx2 base (180,368 B) | **−176 B** (ΔS_rate −1.1719e-4, receipt field) |
| **vs the gb1 POINTER base (180,215 B)** | **−23 B ≈ −1.531e-5 S** — the joint MARGINAL |
| token stream | 113,777 → 113,601 B; `tokens_changed` **0** (lossless family) |
| wall | r1 727 s (refused at splice) + r2 resume 29 s from the f575 checkpoint |

**Overlap MEASURED: 77 B = 30.4% of the naive sum.** Standalone rungs were
groupbin8_surprise −153 B + cls_groupbin8 −100 B = −253 B naive; the joint re-encode
delivers −176 B. The gb1 gestalt-delta's prediction — *same-axis families share the
conditional structure they price* — now carries a number.

## 2. Adjudication — BANK (per the rule derived BEFORE the harvest)

Pre-registered rule: solo T4 fire only if the marginal ≤ ~−30 B (≈ −2e-5 S), because the
fire costs a NEW gen-21 native C port (versioned generation — never pair a C generation
with a different family count, per `runtime-rs/native/f26-corrector/gb1_20family/README.md`)
+ decode-identity proof + canonical seal + ~$0.20 T4. Measured −23 B < the bar → **BANK**.

**Bank record:** candidate zip retained (sha above) + `runtime_joint21/` decoder config in
custody + decode-identity proof OWED at fire time (Python-corrector full decode + the
gen-21 C port). The rung fires by RIDING the next lossless fire on this lineage or any
composition that clears the bar; it is NOT additive with the shipped gb1 rung (it
SUPERSEDES it — the joint candidate replaces the same token stream).

## 3. The r1 refusal (lesson, one line)

r1 completed all 600 frames then correctly REFUSED at the splice: MAIN's launch passed
`--pointer-archive` at the gb1 candidate while the runtime tree mirrored the dx2 base —
the #1237 half-updated-pin genus, recurring in MAIN's own launch config. The encoder's
mirror assertion (jg2:926) fail-closed exactly as designed; the resumable checkpoint made
the cure cost 29 seconds instead of a 12-minute re-encode. Resumability-P0 paid for
itself again.

## 4. GESTALT-DELTA

The model-axis block of [[dx2-block-ceilings-are-measured-and-sum-to-5-percent]]
(ceiling 2,162 B) now has TWO collection points: gb1 −153 B (shipped, the twentieth
move) and joint21 −176 B total (banked). The per-family marginal is SHRINKING
(153 → 23): conditional structure inside this block exhibits strong overlap, so the
remaining ~1,986 B of measured ceiling will cost MORE per byte to collect than the first
rung did. This sharpens the block table's verdict — the model axis funds micro-moves,
never the 42,229 B demand. The campaign's live sub-0.12 route remains the S1
trained-renderer diagonal (#1270).

Sisters: `ddm_gb1_groupbin8_verdict_20260824.md` (the twentieth move + the successor
naming) · [[dx2-block-ceilings-are-measured-and-sum-to-5-percent]] · `#1237` (the pin
genus §3 re-instances).
