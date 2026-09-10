# TC3 mixer and pricing independent review

Reviewer: ddm_tc3 geometry subarm. Scope: source review only, no code/state
mutation or job launch. Parent owns implementation controls, native execution,
full-n600 measurements, public receiver proof and review-tracker marking.

Reviewed final files and SHA-256:
- `experiments/ddm_tc3_mixer.py`:
  `fe5aa8b922711f1919935ea2a513eaa3dd449ddbbffdfe8b9213ffd1b3409f7d`.
- `experiments/ddm_tc3_price.py`:
  `0613a39c1d7ac87d94f1d34c4791819c74d672aad6af7a3ce1201658923090a8`.

Supporting source inspected: `ddm_tc1_mixer_codec.py`,
`ddm_tc1_logistic_bound.c`, `ddm_tc3_geometry.py`, `ddm_tc3_trace.py`,
`ddm_jg2_tail_reencode.py`, and
`ddm_rc64p_native_cpu_decode/route_b_rc64.py`. Charter/common-contract recall
for this arm is recorded in the sibling `geometry_review.md` and the parent
receipt; no new scientific claim is made by this review.

## Findings corrected before the final clean passes

The initial pricing binding omitted the imported io helper and Python RC64
wrapper. Per-frame trace reads checked payload/field but ignored producer/native
lineage. The initial encode binding omitted the fit result that supplies float
weights and convergence status. Downstream stages did not compare inherited
source/field bindings. Preparation accepted an earlier stop than its checkpoint,
and the fit scope did not explicitly limit its tangent gap to the retained
fitting subset. An optional mixer active-frame restart guard was also absent.

Parent applied the corrections. Final code captures io and wrapper facts plus
the trace binding, checks frame receipts against the captured stage binding,
checks inherited common keys across stages, binds the fit result and rounded
weights jointly, rejects bad preparation boundaries, and rejects restarting an
active mixer frame. A and B zero-extra reconstruction checks now run in every
preparation group. `prepare_v2/` isolates the final source from the preserved
earlier two-frame control; no original receipt is retroactively relabeled.

## Final visible pass 1: lineage, causality and stage contract — CLEAN

Re-read both complete final files after the corrections. Traced preparation
through per-group features/observe, per-frame count updates, and complete
snapshots. Geometry observes only decoded prefixes; original TC1 neighbor maps
mask unavailable groups. Full-plane arguments in preparation do not grant
future symbols to either map. The sixth map is computed from copied previous
labels plus the decoded current prefix, not a stored encoder-only context table.

Trace frame lineage is checked against the captured pin rather than a moving
LATEST reference. Common source/archive/field/native-wrapper facts are compared
between preparation, fit and encode; stage-specific extra differs explicitly.
Encode binds both rounded weights and the entire fit result, then checks the
rounded matrix against its counted bytes. Checkpoints retain every completed
frame's complete mixer or encoder state. No unresolved blocker found in this
scope. This is source inspection, not a completed n600 execution.

Shared assumption: unchanged decoded labels imply the base HPAC/corrector
trajectory can be reused while mixer weights change. The source supports it:
the corrector consumes the fixed labels, and mixer feature counts use the fixed
pre-TC1 rows rather than its fitted output. Public regeneration is still the
execution gate; the trace is not free receiver side information.

## Final visible pass 2: objective, integer coder and retention — CLEAN

Re-read the computation and retention paths separately, with the native
objective and RC64 quantizer open. A uses unchanged TC1 followed by one feature
per output class; B uses eight features per class against the raw pre-TC1 row.
B's seven old features and lane KT feature do not depend on the fitted 40
coefficients, so the continuous objective is affine in those coefficients
inside log-softmax. Its bank selection matches the integer mixer.

The C objective uses `ln(2)/Q`; continuous weights initialize from counted int8
values divided by 32, and final bytes multiply by 32 before rounding. Thus the
continuous and integer exponent conventions agree. Shapes, flattened bank
indexing and dtype contracts agree for both F=1 and F=8. The native quantizer
and `base.frequencies` both floor positive float32 probabilities times 2^31,
apply minimum frequency one and assign the balance to the same argmax winner.
A forwards inactive original rows in both coding and offline pricing.

The fit gap is now explicitly a retained-subset numerical tangent, not a
full-n600 optimality bound. The later encode computes continuous and integer
codelengths on every symbol and compares actual rider sizes separately. All
candidate parameter vectors are retained in iteration/trial records; final
weights, per-frame states, both raw RC64 streams, padded envelopes and counted
riders are persisted. The 45-byte candidate rider and 40-byte source rider
account for every counted coefficient/header. No unresolved blocker found in
this scope. Both final source hashes remained unchanged across the two passes.

Shared assumption: the continuous retained-subset optimum is a useful proposal
for real integer coding. It remains a proposal: full-symbol rounding and actual
twins decide the gain, and the public receiver decides reproducibility. This
review does not promote the subset certificate into a geometry-family closure.

## Boundary and handoff

Parent reported Ruff green and the preserved original-source two-frame
preparation control; this reviewer did not rerun it or claim it covers changed
source. Parent's final-source controls and full-n600 byte/public proof remain
part of the active charter. No additional independent follow-on was created:
review disposition FOLDED into parent ddm_tc3; consumer store
`/Volumes/VertigoDataTier/pact/ddm_tc3_lane_predictor_seal/`; fire trigger is
parent's final-source implementation controls and completed current-field trace.

LIVE-HYPOTHESES: coherent run geometry and joint calibration may supply useful
information beyond existing contexts. This is plausible because B preserves
run identity across slots and permits the original weights to compensate for
the new feature. Neither gain nor full-n600 receiver identity was measured here.

DEAD-ENDS: the source omissions described above are fixed and must not recur;
do not reuse a retained-subset convergence gap as a full-population certificate.
No scientific geometry path was closed by this review.
