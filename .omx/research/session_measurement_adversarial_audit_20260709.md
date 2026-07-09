# Adversarial audit — all 2026-07-09 session findings/measurements (naive/toy check)

**Operator directive:** *"Need to adversarial audit all findings and measurements to ensure not naive or
toy."* Triggered by the operator catching that the v8 scaffold's 707 B/frame byte-cost was naive. This
ledger applies the same scrutiny to EVERY measurement this session. Each row: what it claims → the
adversarial attack → verdict SOLID / PROXY-FLAGGED / TOY-corrected. Pointer 0.19110 UNMOVED. `[no-triality]`.

## The rate-thesis measurements (v8) — the ones with real toy risk

| # | claim | adversarial attack | verdict |
|---|---|---|---|
| 1 | scaffold `bulk_boundary_byte_cost` = **707 B/frame = 0.282 S** | brotli-on-bitmap is a naive coder AND it measured the FULL Road perimeter (wrong scope) | **TOY-corrected** (operator caught) → see #2 |
| 2 | edge-centric Road↔Undriv = **47 KB = 0.032 S, "lands in band"** | 1.5 b/px was an ASSUMED proxy; the contour is ragged (1.6–2.0 crossings/row MEASURED, not the smooth ~1.0 the proxy assumes) | **PROXY-FLAGGED** — honest range 0.021–0.064 S, leans high; "lands in band" is coder-dependent; a REAL measured contour coder is OWED. Neither 0.282 nor 0.032 is the answer. |
| 3 | Road↔Undriv edge = **19% of Road perimeter** (426/2228 px) | 4-connectivity + the "significant >0.5% frame" component threshold are choices | **SOLID** (the 19/47/23/5% split is robust to the coder question; it's a pure geometry measurement on n600 gt) |
| 4 | Road **multi-component in 37.2%** of frames | threshold-sensitive: "significant = >0.5% frame area" is arbitrary; a different min-area changes the % | **PROXY-FLAGGED** — the *direction* (Road is often multi-blob; Undriv is single) is robust, but the exact 37.2% is threshold-dependent. Undriv-single-connected (0%) is the load-bearing part and IS robust. |

## The apparatus/tooling measurements — lower toy risk (real artifacts, not proxies)

| # | claim | attack | verdict |
|---|---|---|---|
| 5 | scaffold self-detects **road=0, undriv=2** | could be a hardcoded fallback masquerading as detection | **SOLID** — verified: `classify_segnet_regions` returns full evidence (static IoU, row spans) on the real n600 cache; matches the measured comma10k order |
| 6 | scaffold byte-close **round-trip bit-exact** | tested on 1 frame only | **SOLID-narrow** — bit-exact confirmed, but on a single frame; a full-n600 round-trip is the honest completion (cheap, owed with the test file #376) |
| 7 | gitleaks **0 false-positives / 40 commits / sha256 provenance** | 40 commits is a sample, not all history | **SOLID** — real gitleaks run on real history; the claim is scoped ("40 commits") not universalised |
| 8 | Flowers = **AFFIRMATION-only** (learned bilinear-pullback = counted weights) | could be an abstract-only skim dressed as depth | **SOLID** — the agent read the actual `flower_standalone.py` code + adversarially self-reviewed its own "it's a lever" claim; the rule-118 counted-weights argument is mechanistically grounded |
| 9 | fmtools caught a **regex-missed embedded credential** | one example ≠ a rate | **SOLID-as-anecdote** — labelled correctly as a single demonstrated catch, and the sha256-false-positive weakness was measured + drove the log-only default (honest both ways) |

## Meta-finding
The one CLASS of naivety this session: **byte-cost / rate estimates via a b/px PROXY instead of a real
coder.** #1 and #2 both fell to it (naive-coder AND coder-assumption). The FIX is a standing rule for v8:
**no v8 rate number is load-bearing until it is a MEASURED byte count from a real contour/arithmetic coder
on our actual n600 boundary — never a bits-per-pixel proxy.** This is the #307 contour-string discipline
(0.820 B/flip MEASURED with a real chain-coder) applied to v8. The geometry measurements (#3, #5) and the
apparatus measurements (#7, #8) are solid; the RATE measurements (#1, #2, #4-exact-%) are proxy-flagged and
owe a real-coder measurement. Nothing here is n96/toy-scale — all ran on the full n600 gt cache — but
n600-scale is necessary NOT sufficient: a real-scale run through a PROXY coder is still a proxy.

**Consequence for v8 gating:** the increment-1 go/no-go must include a MEASURED real-coder contour byte count
(not the 1.5 b/px proxy) as a hard gate — added to the P-C blocking precondition.

## Real-machinery resolution (2026-07-09, operator "prefer the real thing to a proxy")
The "real-coder OWED" gate above was itself SATISFIED with the WRONG coder. Operator caught a deeper proxy:
a generic chain/arithmetic coder measures 600 INDEPENDENT contours, but the Road↔Undriv boundary IS the
horizon = **ego-rigid**. Measured with the real BASIS instead (degree-3 poly + ξ-delta, $0 on n600 gt):
horizon fits at **1.46 px** over 425/512 cols (curvelet-class low-order curve); coeffs are ego-coherent
(only the intercept moves ~1.2 px/frame = ego pitch = the stored ξ). **Real-coder store: 4.7 KB@n600 =
0.0032 S** (SOLID, zlib on delta-coded fp16 coeffs), dominant-arc only — 8× below the generic 0.026 S.
**Meta-lesson (deepened):** even a real arithmetic coder is a proxy if it ignores the representation's
physics. The standing rule tightens: *prefer the real BASIS (curvelet + ξ), not merely a real coder.* Row #2
above (PROXY-FLAGGED, 0.021–0.064 S) is now RESOLVED → the answer was below the whole range because the
generic coder was the wrong instrument. See DAG FEED-v8-realmachinery.
