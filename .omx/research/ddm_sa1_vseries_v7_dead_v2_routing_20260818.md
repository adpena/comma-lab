# V-series SD1M ladder: V7 is DEAD, the live-pathway rungs are contaminated, V2 is the rung

`verdict_scope`: **INSTANCE** — archive
`0ecf5d9c4a9b80820508592abebe8791d5c65292e55490fb712fc0e0272e56f0` @ 166,718 B
(V7_V6_livepw_q2), advisory n600 CPU, frozen-scorer instrument. The routing inference
(live-pathway rungs V3–V8 presumptively contaminated) is DERIVED from the ladder's
cumulative composition, not measured per rung.

## 1. The V7 advisory row

Receipt: `/Volumes/APDataStore/pact/ddm_sa1/advisory_n600_cpu/V7_V6_livepw_q2/attempt_0002/contest_auth_eval.json`
(rc=0, 1,191 s: decode 778.6 s + evaluate 408.7 s).

| | V7 | keep01 advisory (comparator, attempt_0006) | ratio |
|---|---:|---:|---:|
| d_seg | 0.05976591 | 0.00043191 | **138×** |
| d_pose | 9.83209991 | 0.00387399 | **2,538×** |
| bytes | 166,718 | 178,272 | −11,554 |
| S (advisory recompute) | 16.003 | 0.3587 | — |

The rate credit (−8.80e-3 S vs sz1) is overwhelmed ~675× by seg damage alone.
2-bit quantization of the LIVE pathways (blocks.0/1.pw at q2, on top of V6's q3 stack)
destroys BOTH scorers. No compensation solve can rescue this: the in-compile Schur
compensation addresses pose only; the seg leg has no compensator in this family.

This is the caveat firing that the ninth-move memo pre-registered: the sa1 "seg-INERT"
prior (`verdict_scope: formulation` — 3 measured members on the rr4/sz1 lineage) does
NOT extend to 2–3-bit depths on live tensors. Measured, one hour after scoping it.

## 2. The ladder's composition (decoded from `ddm_sa1/retained/` dir names)

| rung | cumulative edits | bytes | vs sz1 |
|---|---|---:|---:|
| V0 | all q4 control | 181,482 | +1,552 |
| V1 | dead pw (blocks.2/3.pw) → q2 | 175,787 | −4,143 |
| **V2** | + dead films (blocks.2/3.film) → q2 | **174,927** | **−5,003** |
| V3 | + LIVE pw → q3 | 172,785 | −7,145 |
| V4 | + coord_mix, head → q3 | 171,645 | −8,285 |
| V5 | + live films → q3 | 171,263 | −8,667 |
| V6 | + dw → q3 | 170,695 | −9,235 |
| V7 | live pw q3→q2 | 166,718 | **DEAD (S 16.0)** |
| V8 | V6 + coord_mix → q2 | 168,685 | −11,245 |

V3–V8 all stack onto live-tensor edits; V7's catastrophic row cannot be attributed to
its last edit alone, so the whole upper ladder is presumptively contaminated until a
rung is measured in isolation. **V1/V2 are the only dead-tensor-only rungs.** V2
dominates V1 (same dead-pw edits plus the two near-zero-render films — rgb_rms 0.032/0.034
per `receipts/render_waterfall.json` — for −860 B more). Waterfall receipts note: dead-pw
q2 weight-MSE is ~1.9e-7 (1,900× smaller than live-pw q2's 3.8e-4), but their render
amplification is large, so V2's survival is an open measurement, not a safe bet.

## 3. Routing

- **FIRED:** V2 advisory n600 CPU (attempt_0002, done-receipt `v2_advisory_n600_20260818`),
  archive sha `e6c37a28882e271ce74ccf85abe497cd911b28f7f0cd3d1a4fa9a82eeef8bf32` @ 174,927 B.
  If seg damage lands in-family and uncompensated pose is solvable, the chain is the proven
  keep01 chain: ~3h Metal authority solve → rebase compile onto sz1 → candidate_seal.v1 → T4.
  Rate leg −5,003 B ≈ −3.33e-3 S = 2.1× keep01's credit; at keep01-class retention (~35%)
  the rung projects ≈ −1.2e-3 S net — roughly 2× the ninth move.
- **If V2 dies too:** the SD1M mixed-precision family closes at FORMULATION scope on this
  lineage (dead-tensor q2 being the mildest member), and the mass axis routes back to the
  sa3×keep01 joint re-solve (ninth-move memo §5) or the js1 joint line.
- **NOT fired:** V1 (dominated by V2 — one advisory adjudicates both: films are the safe
  part), V3–V8 (contaminated until isolation evidence exists).

## 4. Apparatus incident (recorded for #1122)

V2's r1 died at t=5 s: V7's own evaluate phase imported upstream modules from the ExFAT
mirror, Python wrote `__pycache__/frame_utils.cpython-313.pyc` (mtime 12:36:33, mid-V7-r2),
ExFAT added the `._` AppleDouble sidecar, and `contest_compliance._iter_upstream_files`
then refused the snapshot ("canonical authority snapshot cannot contain executable
bytecode"). V7's own receipt is unaffected — its provenance hashed the mirror pre-
contamination. Cure: purged the rebuildable `__pycache__` (trivial-cache delete, allowed)
+ `PYTHONDONTWRITEBYTECODE=1` in the advisory launch env so the run cannot re-contaminate
its own instrument (the detector zeroes on the cure). Fourth confirmed AppleDouble
instance; the first BLOCKING one.
