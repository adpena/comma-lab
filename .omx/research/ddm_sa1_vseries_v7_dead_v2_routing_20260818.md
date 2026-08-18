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

## 5. V2 verdict (landed 2026-08-18 ~18:40Z) — DEAD, and the family CLOSES

`verdict_scope`: **FORMULATION** — SD1M mixed-precision (q2–q4) semantic-tensor quantization on
the rr4/sz1 lineage, bracketed by its mildest and near-deepest measured members.

V2 advisory n600 (rc=0, 1,186 s; receipt
`advisory_n600_cpu/V2_dead_pw_film_q2/attempt_0002/contest_auth_eval.json`):
d_seg **0.00342723** (≈8× keep01's advisory 0.00043191 → seg ΔS ≈ **+0.3 S**, ~90× the −3.33e-3 S
rate credit) · d_pose 0.28587887 (74× keep01's, uncompensated) · S 2.15 @ 174,927 B.

**Mechanism.** The "dead pathway" tensors are dead in weight-MSE only. blocks.2/3.pw at q2 carry
mse ≈ 2e-7 yet produce d_seg 0.0034 — render amplification ≈ 38,700× (vs ≈ 2,518× for live pw,
from the waterfall receipts). Films remain proven-cheap (sa3's S2 edit, which includes films-q2,
landed +2.04e-6 d_seg on T4) — the V2 damage is attributable to the dead-pw q2 legs, which every
rung V1+ shares. Scaling across the two measured members: damage ∝ weight-mse^~0.4, so damage
falls far SLOWER than rate credit as depth shallows — no rung of this ladder pays. V0 (all-q4) is
byte-WORSE than sz1's own coding (+1,552 B). Family closed; seg has no compensator in this family
(the Schur solve addresses pose only).

**Routing (per §3's pre-registered fork) — CORRECTED SAME-DAY.** The mass axis goes to the
**sa3×keep01 composed candidate**, but the marginal is priced against the LIVE keep01 pointer,
not sz1 (a delta without its baseline is unanchored; the baseline moved when keep01 was
admitted). Coverage measured at source: keep01's `PRUNE_NAMES` = {blocks.1/2/3.film} (pruned to
1% rows, survivors q4; all other 2-D tensors already q4 in the SM3R body); S2's recipe =
films 2/3 @ q2 + {films 0/1, frame_embed} @ q3 (`builders/final.py:37`). Overlap: films 1/2/3
are 99%-pruned in keep01, so S2's edits there buy ~nothing. The surviving marginal =
frame_embed q4→q3 (≈ −570 B) + blocks.0.film q4→q3 (≈ −243 B) ≈ **−800 B ≈ −5.4e-4 S rate**,
against S2-class seg (+~2e-4 S) and compensated-pose (+~2.7e-4 S) marginals →
**projected net ≈ −5e-5 to −1e-4 S** — a CONTRIBUTOR rung (~1% of the gap), an order of
magnitude below the earlier −0.9e-3 projection (which double-counted keep01's banked move; the
error is preserved here per append-only honesty and superseded by this paragraph). Build cost is
real: a composed SM3R mode (row-prune + per-tensor mixed depth) needs packer + receiver decode +
parse-back before the advisory/authority/compile chain. Whether this rung outranks the js8 pose
line (pose owns 69.38% of the gap per eu4) is a routing decision taken at the queue head, not
inside this memo.
