# Current contract does not express the receiver-only charter

Source inspection of the immutable current contract (including pr19), independently
checked by `contract_review`, identifies the following controls. These are not
claimed producer execution results; PREFIRE_ATTEMPT.json will hold the real CLI
outcome if the measured timing gate permits attempting it.

1. `_pf_identity` requires `check_pin_consistency(...).ok`; the minimal receiver
   has no top-level ARCHIVE_SHA256 / ARCHIVE_BYTES assignments. Its existing
   equivalent checks inside main do not satisfy this recognizer. Expected typed
   refusal: PREFIRE_IDENTITY_DRIFT_REFUSED, receiver/archive pins disagree.
2. An internal MANIFEST.sha256 is optional: validate_receiver_manifest accepts
   absence. An external source/dependency manifest is still required.
3. `_pf_pointer` requires a strictly negative admit bar AND rate-only delta less
   than it. Identical pointer bytes give exactly zero; no negative bar can pass.
   The honest receiver-only threshold is zero, not a claimed score improvement.
4. `_pf_evidence` requires two independent n600 encoder executions and both
   retained 3,662,409,600-byte raws. This charter changes no encoder or payload,
   and its <=2 GiB retention cap cannot retain even one full raw. Copies or
   extraction are not independent encodes. Certified raw identity remains real
   evidence, but is not the object that this validator accepts.
5. The raw gate requires the sealed receiver's structured cold stdout schema;
   the minimal receiver does not emit it. The smoke checker requires a specific
   RuntimeError message, while the minimal receiver uses SystemExit. No log or
   exception is rewritten to look like the other receiver.
6. Risk hashing separately requires both top-level archive-pin variables.
   Ordinary normalized hashing does not. These digest definitions cannot be
   silently interchanged.
7. `completed_t4_receiver_delta` requires actual REFUSED diagnostics, candidate
   cold n600 and max(candidate walls)/one base wall with a nonnegative clamp.
   It cannot honestly encode sampled n48, restored pre-pair states and the
   median(B)/median(A) requested here. Neither successful diagnostic timings nor
   host contention manufactures a REFUSED verdict.
8. The newer identity-class mode refuses changed receiver bytes. It is not a
   route around the rewrite's own measurement requirement.

The original t4_direct leg validates without problems. Its exact wall time is
1185.899645171 s. Its legacy receiver digest is
fdef3b001dbdfc76ebeccfcf1fd1b33928e619fbc018bd2cb323cc67de566cbb;
its behavior digest is
9f6e71680a13d8598974ee13f78a1a72759a758681e6d86b7b288cc105442890.
The distinct definitions are retained; neither is this new receiver's identity.

If the producer refuses, it deletes only its newly written failed intent and
retains a typed refusal beside the output (rc3). There is then no surviving intent
whose file SHA, size or canonical digest can be reported, and no actual intent
for the normal validate_seal / --seal controls. These downstream controls are
NOT REACHED, not falsely reported as executed. The charter explicitly requires
stopping on a real refusal, with MAIN adjudicating the contract mismatch.
No contract file, public pin, score threshold fiction, fake twin or raw report
is introduced to get past these controls.

MAIN's ordering remains: land exact source/evidence; resolve the scoped refusal;
produce and COMMIT a valid intent; only then authorization, first measurement,
harvest and completion. This arm does none of those dispatch steps.
