# Task #452: defect-network / tube-algebra per-boundary rate-code probe

**OUTCOME — NEEDS-MORE (instance/formulation scoped).** **MEASURED:** on the real n600 GT-cache
phase surface, the lossless standalone candidate section is 1,003,855 bytes versus 1,010,237 bytes
for incumbent `PHAS1`, a saving of 6,382 standalone-section bytes. Both the integer residual stream
and decoded phase fields are bit-identical under the same authenticated out-of-band GT-cache
geometry, so the standalone-section rate-code subverdict is GO. **MEASURED:**
the component streams themselves are 996,246 bytes versus 993,897 incumbent residual bytes, so
the defect-network correlation mechanism is NO-GO by 2,349 bytes; all net saving comes from
removing redundant per-frame counts from the header. **UNKNOWN:** that GT-cache geometry has not
been made derivable from the shipped witness receiver. The current byte-close receiver measures
`PHAS1` for accounting but does
not pack or consume it, so equal `d_seg` and `d_pose` for a carrier A/B are **UNKNOWN** rather
than measured. `score_claim=false`; `promotion_eligible=false`. Re-derived canonical state records
the submittable `[contest-CPU]` pointer as 0.1910828242 and the separate non-submission defensive
bank as 0.1880443980; neither was mutated by this probe.

**REVIEW STATUS:** `fresh-eyes-reviewed(3)` after three consecutive independent CLEAN passes on
the current post-fix bytes. The immutable measurement receipt retains its honest
`recovery-written-UNREVIEWED` at-measurement tag. **STORES CONSULTED:** loaded
`tools/corpus_query.py` across research, equations,
memory, DAG, council, tasks, and docs; the phase/covariance/flicker ledgers; canonical
task/lane/subagent stores; the real n600 cache and incumbent codec sources; and arXiv
2607.07786 HTML. Deliberately did not consult paid-provider or live-run actuation stores,
did not call `upstream/evaluate.py`, and did not touch the protected trainer or V9 result.

## Exact dictionary and its limits

The imported source is Nathan Benjamin, Ho Tat Lam, and Conghuan Luo (2026), *Chiral Tube
Algebras I: Topological Defect Lines, Twisted Modules, and Finite Gauging*, arXiv:2607.07786,
Sections 1.1 and 1.3. The identifier and named paper were resolved on the
[arXiv abstract/HTML page](https://arxiv.org/abs/2607.07786) before derivation.

| Pact object | Paper object | Status | Exact boundary |
|---|---|---|---|
| GT-cache-derived 8-connected active boundary component | codimension-one topological defect line (§1.1) | **ANALOGY** | The separatrix is codimension one in image geometry, but no invariance under arbitrary deformations or fusion-category structure was proved. Receiver derivability of this geometry is UNKNOWN. |
| Component neighborhood and lossless residual transform | tubular neighborhood and lasso/tube-algebra operators (§1.1, §1.3) | **ANALOGY**, with a rigorous integer transform underneath | `DTUB1` stores one first residual and raster-order first differences per connected component. Exact inversion is tested. The lasso algebra is not implemented or invoked as a theorem. |
| Residual sign orbit under a component-wide finite `Z2` action | twisted module / defect Hilbert-space sector (§1.3) | **ANALOGY**, with a rigorous finite coding action underneath | `r` and `-r` are canonicalized by first nonzero sign, and one orientation bit restores the original. This is an invertible coding quotient, not a CFT twisted module. |
| Orientation canonicalization plus counted group label | finite gauging (§1.3) | **ANALOGY** | The paper's gauging reorganizes local and defect sectors. Here only a `Z2` sequence action is quotiented; no ego-motion gauge theorem was established. |
| Three-class 2x2 label junction | defect-line fusion junction (§1.1) | **ANALOGY** | `PHAS1` has no separately stored junction field. There is therefore no free vertex datum to eliminate without changing ordinary residual samples. |

**DERIVED:** the only rigorous algebra used by the code is
`r_C -> (r_C,0, r_C,1-r_C,0, ..., r_C,n-r_C,n-1)`, with the optional invertible
component action `r_C -> -r_C` plus a counted group-label bit. **INFERRED:** connected-boundary
ordering can expose more local residual correlation than the incumbent global active-pixel
order. **UNKNOWN:** whether an actual receiver-consumed phase correction preserves or improves
evaluator cells. No paper supplies a visual-codec byte-saving theorem; none is claimed.

## Pre-registered controls and measured gate

**P1:** the cache, receipt, and artifact manifest each have one path/key; contour bytes are context,
not duplicated payload. **P2:** exact serialized-byte and array comparison has a 0-byte within-input
floor. Across-seed variance is **UNKNOWN** because the real probe uses the deterministic single
n600 spine. **P3:** the residual and decoded-phase mismatch budgets are exactly zero; no through-R
distortion budget is claimed. **P4 controls:** the positive control reconstructs exact residual and
phase arrays. Negative controls alter geometry without changing topology, add an extra frame,
declare unused non-`Z2` bytes, alter the active mask, or append trailing bytes; each must fail
closed. **P5:** incumbent, plain, and `Z2` arms run in one invocation on the same cache, classes,
quantization, and geometry. **P6:** this formulation constructs spatial components independently
per frame; a temporal component-tracking/zero-mode code is **UNMEASURED**, so no sequence-family
verdict is asserted. **P7:** any candidate not smaller than `PHAS1`, any component-stream regression,
or any equality/custody failure blocks the corresponding GO. **P8:** lossless decode has zero
allowed mismatch; optimization was restricted to the standalone section's byte gap.

**MEASURED:** the real cache contains 1,287,364 active residuals across 151,175 connected
components. The plain candidate section is 1,003,855 bytes, saving 6,382 standalone-section bytes.
Its base-plus-delta
component streams total 996,246 bytes, 2,349 bytes worse than the incumbent 993,897-byte residual
stream. The generic header deduplication is therefore the source of the section saving, not the
defect transform. The candidate's conditional rate-term change is -0.004249511838825697 score
units if and only if a future receiver derives the same geometry, consumes the identical decoded
phase field, and counts this section in the exact archive. The artifact manifest binds receipt,
candidate, canary, and execution-source hashes in the durable result directory.

**MEASURED mechanism falsifiers:** only 33.7258% of components and 3.98442% of residual pixels
are constant, so the proposed zero-mode-only code cannot exactly reconstruct 96.01558% of pixels.
The finite-`Z2` variant is 1,005,719 bytes. Its 18,897 group-label bytes make it 1,864
bytes larger than the plain candidate, so finite gauging as implemented is NO-GO. There are 6,703
three-class 2x2 junction cells, but `PHAS1` stores no independent vertex field; junction-field
elimination is NO-GO for this representation.

**MEASURED contour context:** the current lossless `contour_codec` totals 507,832 bytes over the
600 label maps, with 846.3867 bytes per map on average. It is not co-stored in `DTUB1`, so these
bytes are context rather than claimed savings; each geometric byte retains one home.

## Through-R boundary and verdict

**MEASURED canary:** `tools/levelset_byte_close_and_eval.py` inflated and scored six real pairs
from `snapshot_ema_BEST.npz`. The NumPy-fp32 bit-exact gate passed on four frames. The inflated
base witness measured `d_seg=0.003478156195746528` and `d_pose=146.77780276986093` on
`[macOS-CPU advisory]`; the checkpoint is pose-blind, so this is explicitly non-promotable. The
phase section remeasured at 1,010,237 bytes with bit-identical reconstruction.

**UNKNOWN / HARD WALL:** this canary is not a carrier A/B. Source inspection and the tool's own
report show that the phase section is constructed for rate accounting but is not packed into or
consumed by the inflated witness. Therefore the mission's equal-or-better through-R `d_seg` and
`d_pose` gate is unmet, and an overall GO would be fake.

**VERDICT: NEEDS-MORE overall.** The standalone-section formulation is GO by 6,382 bytes with
exact decoded phase fields under shared GT-cache geometry, but that saving is a generic header
deduplication and receiver derivability is UNKNOWN. The load-bearing
defect-network component transform is NO-GO by 2,349 stream bytes. Zero-mode-only coding, the
tested finite-`Z2` quotient, and separate junction-field elimination are also formulation-scoped
NO-GO results for the precise reasons above. No family or paradigm is killed. The named recess
measurement is a receiver-consumed `PHAS1` versus `DTUB1` exact-byte A/B through the existing R
operator; until that surface exists, the control law is a constant fail-closed gate:
`promotion_eligible=false`.

## Triality and durability

This is a `research_only=true` isolated codec, not a campaign Lever, exported policy, curriculum,
or trainer wire-in. No witness-DSL surface is therefore asserted. The measured law is
`defect_network_component_delta_rate_v1`; the DAG leg is the task #452 FEED; the durable receipt
is `experiments/results/defect_network_tube_rate_code_20260712T225958Z/measurement_receipt.json`.
