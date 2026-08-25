# ddm_jt22 — mixer-context race VERDICT: axis CLOSED measured-both-directions (2026-08-25)

**Task #1276. Verdict: REFUSED at the fire bar.** The F26 mixer-context axis is closed with
measurements in BOTH directions on the dx2-lineage base (pointer pin 976f706d…, archive
180,368 B, shipped token stream 113,777 B). No 21st pointer move comes from this axis.

STORES CONSULTED: hot-state POINTER_LINE + LIVE_PROCESSES · jt21 verdict memo
(ddm_jt21_joint_21family_reencode_verdict_20260825.md) · gb1 verdict memo · S1_encode receipts
(ctxA + ctxB, sha-pinned below) · task ledger #1276/#1269.

## The two variants, measured

Shipped mixer context: `cls_boundary_agree_homog_ubin8` — 4,000 mixing sets, uncertainty tail
`np.minimum(ubin >> 3, 7)` (fx2_model_axis_corrector.py:326). Harness:
`experiments/ddm_jg2_tail_reencode.py --stage encode` ($0, scorer-free, EXACT bytes; both runs
`control.byte_identical=True`, `tokens_changed=0`, `score_claim=false`).

| Variant | Context | Sets | Stream (B) | Δ vs 113,777 | Receipt |
|---|---|---|---|---|---|
| V-A replace | `cls_boundary_agree_homog_groupbin8` (position REPLACES ubin8) | 4,000 | 114,096 | **+319** | S1_encode_gb1_jt22_ctxA.json |
| V-B cross | `cls_boundary_agree_homog_ubin8_groupbin8` (position ADDED to ubin8) | 32,000 | 113,776 | **−1** | S1_encode_gb1_jt22_ctxB.json |

V-B receipt custody: candidate stream sha `c5013226…`, candidate archive
`f4d4fdc7…` @ 180,367 B, delta_S_rate −6.659e-7, base pointer pin `976f706d…` verified.
Retained at `/Volumes/APDataStore/pact/ddm_gb1_groupbin8_conditioning/retained/`.

## Adjudication

1. **Replace direction (V-A):** decode-position CANNOT substitute for the ubin8 uncertainty
   tail — +319 B. The model's uncertainty signal carries real conditional information that
   position does not.
2. **Add direction (V-B):** the full cross buys exactly **1 byte** while growing the context
   table 8× (4,000 → 32,000 sets). The residual position-conditional information GIVEN ubin8
   is ~zero — the gb1 groupbin8_surprise mixer FAMILY (−153 B, the 20th move) already absorbed
   what position knows; pushing position into the mixer CONTEXT is redundant with it.
3. **Vs the bars:** −1 B is 30× below the pre-registered ~30 B solo-fire bar and does not
   change the jt21 bank arithmetic (jt21 −23 B marginal stays banked, still sub-bar).

## Consequences

- **#1276 CLOSED.** The mixer-context rung joins the measured model-axis ledger: the
  2,162 B model-axis ceiling ([[dx2-block-ceilings-are-measured-and-sum-to-5-percent]])
  stands; per-family marginals keep SHRINKING (153 → 23 → 1 B) — the model axis funds
  micro-moves only and cannot supply the 42,229 B demand.
- The jt21 bank (−23 B, sha ec0dd68f…) rides the next lossless fire unchanged.
- Live route to sub-0.12 remains a DIFFERENT OBJECT (post-S1e state: all single axes
  measured short; see hot-state POINTER_LINE).

verdict_scope: FORMULATION (mixer-context conditioning on the dx2/gb1 lineage, both
directions measured at exact bytes). score_claim: false — no scorer ran; distortion legs
untouched by construction (token identity proven).
