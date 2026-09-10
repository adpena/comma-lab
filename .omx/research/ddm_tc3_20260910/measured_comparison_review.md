# TC3 measured comparison and verdict-scope audit

Independent ddm_tc3 geometry-subarm audit of retained results. No encode,
optimizer, receiver, scorer or long job was launched. This note is the only
file written by this audit; canonical state and source files were not changed.

## Current measured comparison

Axis: `[exact serialized bytes, scorer-free macOS-CPU, n600]`.
Every current result uses **117,964,800 positions = 600 x 384 x 512**, move39
archive `8877f75d87bf25b410264e08682959c7710cf677307bd5452039ce53835f6bf4`,
with field SHA
`4aa519a25e4b02afb564498025b366cb9007ea663093c60bf8ce90d079dc8791`.
The field identity is read from the source-bound receipts; this audit did not
rehash the 118 MB field or rerun its decoder.

| Current move39 quantity | A: TC2 map, five new fitted weights | B: run tracks, joint 40-weight fit |
|---|---:|---:|
| Full-population continuous codelength saving, bits/8 | 84.123664 | 48.578657 |
| Full-population integer-probability codelength saving, bits/8 | 84.113623 | 41.417880 |
| Raw RC64 stream bytes | 119,484 | 119,527 |
| Raw RC64 saving against source 119,568 B | 84 | 41 |
| Counted rider bytes | 119,529 | 119,572 |
| Counted rider saving against source 119,608 B | **79** | **36** |
| Smallest actual archive bytes | **180,107** | **180,150** |
| Actual archive saving against source 180,186 B | **79** | **36** |

Both winners are the plain rider in a stored ZIP. B loses to A by 43 archive
bytes, so it fails the selection rule requiring B to beat A by at least 300 B.
A clears the 30 B byte-admit bar by 49 B. The current single staged candidate
is A, archive SHA
`2ee6e292255a63d48391f852cde26b573fe3e8de378fabfa056632c24e7c6c13`.
Its recorded S 0.13761671196479372 is a projection conditional on public
output identity and the later MAIN contest evaluation, not a measured score.

## Historical scale comparison: different fields

The requested TC2 denominators are from **move37**, not move39:
82.771457 continuous ideal bytes, 78 counted real bytes, and the noncausal
GT-distance reference's 5,480.397216 ideal bytes. These ratios compare scale;
they do not establish a matched field treatment effect, current-field oracle
headroom, or a universal upper bound on lane prediction.

| Current numerator divided by historical denominator | A | B |
|---|---:|---:|
| Current continuous / TC2 causal continuous 82.771457 B | 101.633663% | 58.690107% |
| Current real / TC2 real 78 B | 101.282051% | 46.153846% |
| Current continuous / GT-reference ideal 5,480.397216 B | 1.534992% | 0.886408% |
| Current integer ideal / GT-reference ideal 5,480.397216 B | 1.534809% | 0.755746% |
| Current real / GT-reference ideal 5,480.397216 B | 1.441501% | 0.656887% |

The last row deliberately mixes physical bytes with an ideal historical
denominator and is only a normalized scale comparison. A's 79 B versus TC2's
78 B cannot be attributed to a better predictor: the fields differ. No GT
distance oracle was rerun on move39.

## Joint fit and integerization checks

The retained fit uses 18,294,812 positions for A (15.508704% of all positions)
and 15,449,620 for B (13.096805%). Both selections span the full 600 frames;
they are probability-selected fitting subsets, not prefix samples. The final
continuous weights are evaluated over all 117,964,800 positions in encode.
A fits five new coefficients while preserving the old 35. B fits all 40
coefficients against the fixed pre-TC1 frequencies and eight fixed features.

I checked the four saved optimization iterations per variant: their losses
decrease, and the last stored supporting-tangent gap recomputes to the RESULT
value. A ends at 0.000005784036 ideal B; B at 0.000039284875 ideal B. These are
floating-point numerical convergence gaps on the retained fitting subsets.
They do not certify full-n600 coefficient optimality, the best int8 matrix,
another predictor, or a general geometry family.

The full-population continuous and integer savings recompute exactly from
each RESULT's three total-bit sums. The serialized weight bytes equal the
recorded rounded matrices. Twin raw streams, padded envelopes and counted
riders are byte-identical for each variant. This audit verified the retained
outputs and arithmetic; it did not rerun the native objective or add a new
gradient/Hessian test. The source-level affine-feature and quantization
conventions were reviewed in sibling `mixer_price_review.md`.

| Difference in current move39 saving | A | B |
|---|---:|---:|
| Continuous minus integer-probability ideal B | 0.010041 | 7.160777 |
| Integer-probability ideal minus actual raw-stream B | 0.113623 | 0.417880 |
| Raw-stream saving minus counted-rider saving B | 5 | 5 |
| Continuous minus actual counted saving B | 5.123664 | 12.578657 |

B loses 14.740583% of its measured continuous gain during integerization,
which includes int8 coefficient rounding and the deployed probability path;
the audit does not attribute the entire gap to coefficient rounding alone.
The new rider carries 40 coefficients instead of the source's 35, accounting
for the five additional bytes in both variants. Even B's measured continuous
48.578657 B is well below 150 B, so integerization is not the main observed
shortfall. Recovering its measured integerization loss alone cannot supply the
114 B missing from the real-byte threshold.

For an additional scope check, the existing TC2 supporting-tangent recipe
can grant zero candidate loss to every omitted position. Using the final
retained loss and gap gives optimistic full-population continuous bounds of
162.872250 B for A and 170.969687 B for B:
`source_full_bits/8 - retained_loss_nats/(8*ln(2)) + retained_gap_bytes`.
These are numerical fixed-feature, bounded-weight bounds under the same
exclusions as the registered equation. In particular, B's bound is above
150 B and does not prove a sub-150 B ceiling for every coefficient choice.
It is not an integer-coder bound. The measured 36 B result therefore supports
the charter's routing falsifier, not a mathematical impossibility claim.

## Verdict scope

**FORMULATION-scoped operational closure is supported:** B's real n600
36 B saving is below the charter's predeclared 150 B falsifier. The tested
slotwise-map and fixed run-tracker context extensions, with these bounded
shared weights, fitting procedure and counted precision, have failed to
produce the proposed large lane-context gain. The charter's 0.5-1.5 KB
central prediction is falsified for this move39 B implementation.

Do not generalize that result to all possible lane predictors, richer causal
contexts, different run dynamics, different integer optimization procedures,
or a task-space generator. The historical oracle is itself a specific
noncausal quantized reference. A and B also change both geometry and fitting
scope, so this comparison does not isolate tracking's contribution from joint
refitting's contribution. No broad geometry FAMILY closure or independent
causal claim about joint refitting is supported.

## RECALL EVIDENCE

Re-read the TC3 charter, current fit/encode source and sibling implementation
reviews. Searched the TC2 memo and charter with
`82.771457|5480.397216|5,480|78 B|field|FORMULATION` and read TC2's result,
bound and verdict-scope sections. Searched owned review/test files for
`joint|gradient|Hessian|hessian|finite|objective|round|gap|test` and enumerated
the retained fit/control evidence. The TC2 memo explicitly limits its
supporting tangent to a fixed coefficient family, distinguishes kept fitting
positions from full-symbol pricing, and rejects the GT map as a universal
oracle. That changed this audit's wording to historical scale ratios and an
operational formulation closure. Broader original recall is in the earlier
geometry note and parent receipt; no new scientific source was introduced.

## Evidence custody checked

Read and hashed each fit RESULT, each encode RESULT, CONTAINERS and CANDIDATE.
Verified both variants' fit-weight receipts and payload twins. Rehashed all
80 retained archive samples and their rider facts (40 per variant), the
source-control archive and the staged candidate archive. The smallest-sample
selection and A candidate identity agree. No payload was created or discarded.

| Receipt relative to move39 store | SHA-256 |
|---|---|
| fit/A/RESULT.json | `4aba45bfa0fcd2b096d26af6269ef1e738222466cce807bf4a31ce3cd30884e5` |
| fit/B/RESULT.json | `5e7c478ae3c800d098ac2924a810814dbe6b11e7c740a4e6ba5c4ffcfb5e0daf` |
| encode/A/RESULT.json | `578d3d83ce489f60801ab82e2ab192623db3977d54ea94751477982b10d9bb0e` |
| encode/B/RESULT.json | `973778c0b2d7d6440a224e4e2ae014fc7626429171f2cdef7d6be03665e61c69` |
| CONTAINERS.json | `99a3fb56be46252ec34408f8e64b04bb38edd49b4c3c1aa6c7b06cd3c0eec08a` |
| CANDIDATE.json | `6121005dcefad1d5757757da3931f830721fbe788cf81a61929eec84f0b11c7b` |

Store: `/Volumes/VertigoDataTier/pact/ddm_tc3_lane_predictor_seal/move39/`.
Parent is already executing A's authorized public proof. This audit supplies
no public receiver identity, scorer result, contest-time result or new score.
Disposition: FOLDED into parent ddm_tc3's active final memo and one-seal
handoff; owner parent ddm_tc3; consumer the move39 store and final TC3 memo;
fire trigger the currently active public-proof harvest.

LIVE-HYPOTHESES: A's 79 B actual archive saving may survive public inflation;
its causal receiver matches the encoded mechanism, but the full literal-shell
proof remains the parent's active execution gate. The generator remains
outside this audit's tested context-map formulations; no generator gain is
asserted or newly queued here.

DEAD-ENDS: the tested B run-track plus joint-refit recipe fails the declared
150 B real-byte bar on move39, so it should not be repeated unchanged. A broad
claim that all lane predictors are closed is rejected because the experiment
and numerical tangent cover only fixed tested formulations. Historical
move37 oracle fractions cannot be presented as current-field oracle recovery.
